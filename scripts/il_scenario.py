#!/usr/bin/env python3
"""
il_scenario.py  —  Scene & Task Generation for IL Dataset  v9 (Phase 3 + Profiles)

Provides:
  - CylinderObstacleSpec dataclass
  - ObstacleRegion definition
  - SceneGenerationProfile dataclass
  - DensityMode enum
  - SceneValidationResult / TaskValidationResult dataclasses
  - YamlCylinderSceneGenerator: procedural cylinder scene generation
  - CylinderSceneValidator: 2D topology, U-shape, dead-end checks
  - StartGoalTaskGenerator: task sampling with constraint enforcement
  - SideCostEvaluator: left/right portal path cost comparison
  - SceneManifestWriter: YAML/JSON manifest for reproducibility
  - SceneGenerationFailureManifestWriter: failure manifest
  - ObstacleVisibilityAuditor: runtime observability audit

Conventions:
  - All world coordinates: ROS frame (X-fwd, Y-left, Z-up).
  - Cylinder centers: (x, y, z) with z = cylinder centre height.
  - Height semantics: cylinder extends from (z - height/2) to (z + height/2).
  - Obstacle region: only limits where cylinder centres may appear.
  - Space outside obstacle region is freely traversable.

Only cylinder obstacles are generated.
The obstacle generation region is not treated as a wall or flight boundary.
Space outside the obstacle generation region remains traversable.
"""

from __future__ import print_function, division

import math, os, time, random, copy, yaml, json
import numpy as np
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum

import rospy


# ============================================================================
#  Density modes and helpers
# ============================================================================

class DensityMode(Enum):
    INFLATED_OCCUPANCY = "inflated_occupancy"
    RAW_OCCUPANCY = "raw_occupancy"
    OBSTACLES_PER_100M2 = "obstacles_per_100m2"
    FIXED_COUNT = "fixed_count"

    @staticmethod
    def from_string(s):
        for mode in DensityMode:
            if mode.value == s:
                return mode
        raise ValueError("Unknown density mode: '{}'. Supported: {}".format(
            s, [m.value for m in DensityMode]))


def compute_region_area(region):
    """Compute area of an ObstacleRegion in the XY plane (m²)."""
    return (region.x_max - region.x_min) * (region.y_max - region.y_min)


def compute_raw_occupancy(obstacles, region_area):
    """Σ π r_i² / region_area."""
    if region_area <= 0.0:
        return 0.0
    total = sum(math.pi * o.radius_m ** 2 for o in obstacles)
    return total / region_area


def compute_inflated_occupancy(obstacles, region_area, vehicle_r, safety_m):
    """Σ π (r_i + vehicle_r + safety_m)² / region_area.

    Note: Overlap is double-counted.  Current profiles forbid inflated
    overlap, so this approximation is acceptable.
    """
    if region_area <= 0.0:
        return 0.0
    infl = vehicle_r + safety_m
    total = sum(math.pi * (o.radius_m + infl) ** 2 for o in obstacles)
    return total / region_area


def compute_obstacles_per_100m2(obstacles, region_area):
    """obstacle_count / region_area × 100."""
    if region_area <= 0.0:
        return 0.0
    return len(obstacles) / region_area * 100.0


def compute_density(obstacles, region_area, mode, vehicle_r=0.30, safety_m=0.10):
    """Compute the requested density metric for a set of obstacles."""
    if mode == DensityMode.RAW_OCCUPANCY:
        return compute_raw_occupancy(obstacles, region_area)
    elif mode == DensityMode.INFLATED_OCCUPANCY:
        return compute_inflated_occupancy(obstacles, region_area, vehicle_r, safety_m)
    elif mode == DensityMode.OBSTACLES_PER_100M2:
        return compute_obstacles_per_100m2(obstacles, region_area)
    elif mode == DensityMode.FIXED_COUNT:
        return float(len(obstacles))
    else:
        raise ValueError("Unsupported density mode: {}".format(mode))


def compute_pairwise_min_gaps(obstacles, vehicle_r, safety_m):
    """Compute min surface gap and min post-inflation gap across all pairs.

    Returns:
        (min_surface_gap_m, min_post_inflation_gap_m)
    """
    min_surface = float('inf')
    min_post = float('inf')
    if len(obstacles) < 2:
        return (0.0, 0.0)
    infl = vehicle_r + safety_m
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            a, b = obstacles[i], obstacles[j]
            d = float(np.linalg.norm(a.center_xy() - b.center_xy()))
            sg = d - a.radius_m - b.radius_m
            pg = sg - 2.0 * infl
            if sg < min_surface:
                min_surface = sg
            if pg < min_post:
                min_post = pg
    if min_surface == float('inf'):
        min_surface = 0.0
    if min_post == float('inf'):
        min_post = 0.0
    return (min_surface, min_post)


def _deep_merge_dict(base, override):
    """Return a recursive copy of ``base`` updated by ``override``."""
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _infer_density_tier(profile_name, density_min, density_max, sg_cfg):
    """Classify a profile as sparse/medium/dense for coverage scheduling."""
    name = str(profile_name).lower()
    for tier in ("sparse", "medium", "dense"):
        if name.endswith("_{}".format(tier)):
            return tier

    coverage_cfg = sg_cfg.get("coverage_balancing", {})
    thresholds = coverage_cfg.get("density_tier_thresholds", {})
    sparse_max = float(thresholds.get("sparse_max", 0.16))
    medium_max = float(thresholds.get("medium_max", 0.24))
    midpoint = 0.5 * (float(density_min) + float(density_max))
    if midpoint <= sparse_max:
        return "sparse"
    if midpoint <= medium_max:
        return "medium"
    return "dense"


class _StratifiedGridSampler:
    """Yield jittered XY samples with one sample per grid stratum.

    The old generator drew every obstacle centre independently from a
    uniform distribution.  That creates avoidable clumps and large empty
    bands, especially in the long, narrow collection region.  This sampler
    traverses a shuffled grid and jitters once inside every cell before
    starting a new pass.  Bounds are supplied at draw time so cylinders with
    different radii retain their own valid boundary inset.
    """

    def __init__(self, rng, target_cell_count, aspect_ratio,
                 minimum_x_bands=1, minimum_y_bands=1):
        self._rng = rng
        target_cell_count = max(1, int(target_cell_count))
        aspect_ratio = max(float(aspect_ratio), 1.0e-6)
        derived_nx = max(
            1, int(math.ceil(math.sqrt(
                target_cell_count * aspect_ratio))))
        derived_ny = max(
            1, int(math.ceil(
                float(target_cell_count) / float(derived_nx))))
        self.nx = max(int(minimum_x_bands), derived_nx)
        self.ny = max(int(minimum_y_bands), derived_ny)
        self._points = []
        self._cursor = 0
        self._refill()

    def _refill(self):
        points = []
        for ix in range(self.nx):
            for iy in range(self.ny):
                u = (float(ix) + float(self._rng.uniform(0.0, 1.0))) / self.nx
                v = (float(iy) + float(self._rng.uniform(0.0, 1.0))) / self.ny
                points.append((u, v))
        self._rng.shuffle(points)
        self._points = points
        self._cursor = 0

    def draw(self, x_min, x_max, y_min, y_max):
        if x_min >= x_max or y_min >= y_max:
            return None
        if self._cursor >= len(self._points):
            self._refill()
        u, v = self._points[self._cursor]
        self._cursor += 1
        return (
            float(x_min + u * (x_max - x_min)),
            float(y_min + v * (y_max - y_min)),
        )


def _plan_obstacle_areas(rng, budget_m2, minimum_area_m2,
                         maximum_area_m2):
    """Plan individually feasible obstacle areas near a group budget.

    For a chosen obstacle count ``n``, the realisable interval is
    ``[n * min_area, n * max_area]``.  This function finds the closest such
    interval to the requested budget and then allocates every obstacle area
    while preserving feasibility for all remaining obstacles.  It avoids the
    systematic low-density under-fill caused by greedily placing a minimum
    obstacle and abandoning a residual smaller than another minimum.
    """
    budget = float(budget_m2)
    minimum_area = float(minimum_area_m2)
    maximum_area = float(maximum_area_m2)
    if (budget <= 0.0 or minimum_area <= 0.0 or
            maximum_area + 1.0e-12 < minimum_area):
        return []

    ideal_area = 0.5 * (minimum_area + maximum_area)
    ideal_count = max(1.0, budget / max(ideal_area, 1.0e-12))
    maximum_count = max(
        1, int(math.ceil(budget / minimum_area)) + 1)
    best = None
    for count in range(1, maximum_count + 1):
        lower_total = count * minimum_area
        upper_total = count * maximum_area
        realised_total = min(
            max(budget, lower_total), upper_total)
        error = abs(realised_total - budget)
        # Prefer an equally close under-fill over an over-fill, then the
        # count whose mean area is closest to the middle of the radius band.
        key = (
            error,
            1 if realised_total > budget else 0,
            abs(float(count) - ideal_count),
            count,
        )
        if best is None or key < best[0]:
            best = (key, count, realised_total)

    _key, count, realised_total = best
    remaining_total = realised_total
    areas = []
    for index in range(count):
        remaining_count = count - index
        if remaining_count == 1:
            area = remaining_total
        else:
            lower = max(
                minimum_area,
                remaining_total -
                (remaining_count - 1) * maximum_area)
            upper = min(
                maximum_area,
                remaining_total -
                (remaining_count - 1) * minimum_area)
            if upper < lower:
                upper = lower
            area = float(rng.uniform(lower, upper))
        area = min(max(area, minimum_area), maximum_area)
        areas.append(area)
        remaining_total -= area
    rng.shuffle(areas)
    return areas


# ============================================================================
#  SizeGroup — per-size-class params for density-driven generation
# ============================================================================

@dataclass
class SizeGroup:
    """Parameters for a single obstacle size class in density-driven generation.

    Three groups (large, medium, small) share the total density budget
    equally by default.  Each group places obstacles until its capacity is
    exhausted or consecutive placement failures exceed the threshold.
    """
    name: str               # "large", "medium", or "small"
    radius_min_m: float
    radius_max_m: float
    capacity_fraction: float = 1.0 / 3.0   # share of total inflated-area budget
    consecutive_fail_threshold: int = 100

    def __post_init__(self):
        if self.radius_min_m <= 0.0:
            raise ValueError("SizeGroup.radius_min_m must be > 0, got {}".format(
                self.radius_min_m))
        if self.radius_max_m < self.radius_min_m:
            raise ValueError("SizeGroup.radius_max_m ({}) < radius_min_m ({})".format(
                self.radius_max_m, self.radius_min_m))
        if self.capacity_fraction <= 0.0 or self.capacity_fraction > 1.0:
            raise ValueError("SizeGroup.capacity_fraction must be in (0, 1], got {}".format(
                self.capacity_fraction))
        if self.consecutive_fail_threshold < 1:
            raise ValueError("SizeGroup.consecutive_fail_threshold must be >= 1, got {}".format(
                self.consecutive_fail_threshold))

    def min_inflated_area_m2(self, infl_radius):
        """Minimum inflated area (m²) of one obstacle from this group."""
        return math.pi * (self.radius_min_m + infl_radius) ** 2

    def max_inflated_area_m2(self, infl_radius):
        """Maximum inflated area (m²) of one obstacle from this group."""
        return math.pi * (self.radius_max_m + infl_radius) ** 2

    def sample_radius(self, rng):
        """Sample a radius uniformly from [radius_min_m, radius_max_m]."""
        return float(rng.uniform(self.radius_min_m, self.radius_max_m))

    @staticmethod
    def from_dict(d, default_name="unnamed"):
        """Parse from a YAML dict."""
        return SizeGroup(
            name=str(d.get("name", default_name)),
            radius_min_m=float(d.get("radius_min_m", 0.10)),
            radius_max_m=float(d.get("radius_max_m", 0.80)),
            capacity_fraction=float(d.get("capacity_fraction", 1.0 / 3.0)),
            consecutive_fail_threshold=int(d.get("consecutive_fail_threshold", 100)),
        )


# ============================================================================
#  SceneGenerationProfile dataclass
# ============================================================================

@dataclass
class SceneGenerationProfile:
    """Runtime contract for one density-driven scene profile."""
    name: str
    enabled: bool
    scene_count: int
    seed_offset: int
    density_mode: str

    size_groups: list
    total_density_min: float
    total_density_max: float

    height_min_m: float
    height_max_m: float
    region_boundary_margin_m: float
    minimum_surface_gap_m: float
    minimum_post_inflation_gap_m: float

    vehicle_radius_m: float
    safety_margin_m: float

    density_tier: str
    coverage_balancing: dict

    @property
    def inflation_radius_m(self):
        return self.vehicle_radius_m + self.safety_margin_m

    @staticmethod
    def from_density_config(profile_dict, scene_generation_cfg):
        """Parse the sole density-driven profile schema."""
        name = str(profile_dict.get("name", "unnamed"))
        enabled = bool(profile_dict.get("enabled", True))
        scene_count = int(profile_dict.get("scene_count", 1))
        seed_offset = int(profile_dict.get("seed_offset", 0))

        groups_raw = profile_dict.get("size_groups", {})
        size_groups = []
        for group_name in ("large", "medium", "small"):
            group = SizeGroup.from_dict(
                groups_raw.get(group_name, {}),
                default_name=group_name)
            group.name = group_name
            size_groups.append(group)

        total_density_min = float(
            profile_dict.get("density_min", 0.05))
        total_density_max = float(
            profile_dict.get("density_max", 0.15))

        common_cylinder = scene_generation_cfg.get(
            "common_cylinder", {})
        vehicle = scene_generation_cfg.get("vehicle", {})
        common_task = scene_generation_cfg.get(
            "common_task_generation", {})

        return SceneGenerationProfile(
            name=name,
            enabled=enabled,
            scene_count=scene_count,
            seed_offset=seed_offset,
            density_mode="inflated_occupancy",
            size_groups=size_groups,
            total_density_min=total_density_min,
            total_density_max=total_density_max,
            height_min_m=float(
                common_cylinder.get("height_min_m", 8.0)),
            height_max_m=float(
                common_cylinder.get("height_max_m", 8.0)),
            region_boundary_margin_m=float(
                common_cylinder.get(
                    "region_boundary_margin_m", 0.30)),
            minimum_surface_gap_m=float(
                common_cylinder.get(
                    "minimum_surface_gap_m", 0.0)),
            minimum_post_inflation_gap_m=float(
                common_cylinder.get(
                    "minimum_post_inflation_gap_m", 0.15)),
            vehicle_radius_m=float(vehicle.get("radius_m", 0.30)),
            safety_margin_m=float(
                vehicle.get("safety_margin_m", 0.10)),
            density_tier=str(profile_dict.get(
                "density_tier",
                _infer_density_tier(
                    name, total_density_min, total_density_max,
                    scene_generation_cfg))),
            coverage_balancing=dict(
                common_task.get("coverage_balancing", {})),
        )



def load_scene_profiles(config):
    """Load enabled density-driven profiles in YAML order."""
    sg_cfg = config.get("global", {}).get("scene_generation", {})
    source = str(sg_cfg.get("source", "density_driven")).strip()
    if source != "density_driven":
        raise ValueError(
            "load_scene_profiles requires source='density_driven', got '{}'"
            .format(source))

    profiles_raw = sg_cfg.get("profiles", [])
    if not profiles_raw:
        rospy.logwarn(
            "[Profiles] density_driven profiles list is empty.")
        return []

    profiles = []
    seen_names = set()
    for p in profiles_raw:
        profile = SceneGenerationProfile.from_density_config(p, sg_cfg)
        if not profile.enabled:
            rospy.loginfo("[Profiles] Skipping disabled profile '%s'.", profile.name)
            continue
        if profile.name in seen_names:
            raise ValueError("Duplicate profile name: '{}'".format(profile.name))
        seen_names.add(profile.name)
        profiles.append(profile)

    coverage_cfg = sg_cfg.get("coverage_balancing", {})
    if (coverage_cfg.get("enabled", False) and
            coverage_cfg.get("require_density_tier_mix", True)):
        mix_scope = str(coverage_cfg.get(
            "density_tier_mix_scope", "all_multi_scene_runs")).lower()
        if mix_scope not in ("all_multi_scene_runs", "full_catalog"):
            raise ValueError(
                "coverage_balancing.density_tier_mix_scope must be "
                "'all_multi_scene_runs' or 'full_catalog'")
        full_catalog_enabled = all(
            bool(item.get("enabled", True)) for item in profiles_raw)
        check_density_mix = (
            mix_scope == "all_multi_scene_runs" or full_catalog_enabled)
        required = set(str(v).lower() for v in coverage_cfg.get(
            "required_density_tiers", ["sparse", "medium", "dense"]))
        scene_counts = {}
        for profile in profiles:
            tier = str(profile.density_tier).lower()
            scene_counts[tier] = (
                scene_counts.get(tier, 0) + int(profile.scene_count))
        total_scenes = sum(scene_counts.values())

        # A one-scene expert smoke test deliberately selects one profile.
        # The canonical full-catalog run must retain every requested density
        # domain.  Explicit profile subsets remain useful for diagnostics.
        if check_density_mix and total_scenes > 1:
            missing = sorted(required.difference(scene_counts))
            if missing:
                raise ValueError(
                    "Coverage-balanced generation needs density tiers {}; "
                    "missing {} from enabled profiles".format(
                        sorted(required), missing))
            minimum_fraction = float(coverage_cfg.get(
                "minimum_scene_fraction_per_density_tier", 0.0))
            maximum_fraction = float(coverage_cfg.get(
                "maximum_scene_fraction_per_density_tier", 1.0))
            for tier in sorted(required):
                fraction = float(scene_counts[tier]) / float(total_scenes)
                if fraction < minimum_fraction or fraction > maximum_fraction:
                    raise ValueError(
                        "Density tier '{}' occupies {:.1%} of {} scenes; "
                        "coverage limits are [{:.1%}, {:.1%}]".format(
                            tier, fraction, total_scenes,
                            minimum_fraction, maximum_fraction))
        elif not check_density_mix:
            rospy.logwarn(
                "[Profiles] Density-tier mix guard skipped for an explicit "
                "profile subset (scope=full_catalog).")

    rospy.loginfo("[Profiles] Loaded %d enabled profiles: %s",
                  len(profiles), [p.name for p in profiles])
    return profiles


# ============================================================================
#  Data classes
# ============================================================================

@dataclass
class CylinderObstacleSpec:
    """Specification of a single cylinder obstacle."""
    obstacle_id: str
    center_world: np.ndarray  # [x, y, z]
    radius_m: float
    height_m: float
    yaw_rad: float = 0.0

    def diameter_m(self):
        return 2.0 * self.radius_m

    def center_xy(self):
        return self.center_world[:2].copy()

    def to_dict(self):
        return OrderedDict([
            ("id", self.obstacle_id),
            ("type", "cylinder"),
            ("center", [float(self.center_world[0]),
                        float(self.center_world[1]),
                        float(self.center_world[2])]),
            ("radius_m", self.radius_m),
            ("height_m", self.height_m),
        ])


@dataclass
class ObstacleRegion:
    """Rectangular region where cylinder centres may appear."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains_cylinder(self, center_xy, radius_m, margin_m=0.0):
        """Check if a cylinder of given radius is within region bounds."""
        r = radius_m + margin_m
        return (self.x_min + r <= center_xy[0] <= self.x_max - r and
                self.y_min + r <= center_xy[1] <= self.y_max - r)

    def width_x(self):
        return self.x_max - self.x_min

    def width_y(self):
        return self.y_max - self.y_min


@dataclass
class SceneValidationResult:
    """Result of 2D topology validation."""
    valid: bool = False
    rejection_reason: str = ""

    obstacle_count: int = 0
    minimum_surface_gap_m: float = 0.0
    minimum_inflated_gap_m: float = 0.0

    inflated_components: int = 0
    enclosed_free_component_count: int = 0
    dead_end_detected: bool = False
    u_shape_detected: bool = False

    navigable_free_ratio: float = 0.0
    minimum_navigable_clearance_m: float = 0.0

    dead_end_max_depth_m: float = 0.0


@dataclass
class TaskValidationResult:
    """Result of start-goal task validation."""
    valid: bool = False
    rejection_reason: str = ""

    direct_path_blocked: bool = False
    direct_blocker_count: int = 0

    astar_reachable: bool = False
    direct_distance_m: float = 0.0
    astar_length_m: float = 0.0
    detour_ratio: float = 0.0

    dominant_obstacle_id: str = ""
    dominant_obstacle_lateral_offset_m: float = 0.0

    left_path_valid: bool = False
    right_path_valid: bool = False
    left_path_cost: float = 0.0
    right_path_cost: float = 0.0
    lower_cost_side: str = ""  # "LEFT" or "RIGHT"
    side_cost_difference_ratio: float = 0.0

    global_side_choice_valid: bool = False

    left_min_clearance_m: float = 0.0
    right_min_clearance_m: float = 0.0
    left_path_length_m: float = 0.0
    right_path_length_m: float = 0.0

    # Coverage-balanced task sampling metadata.  These are task-level
    # geometric proxies for downstream Guide-label coverage; the actual
    # frame-level label histogram remains the final source of truth.
    coverage_target_task_type: str = ""
    coverage_actual_task_type: str = ""
    coverage_target_blocker_distance_band: str = ""
    coverage_actual_blocker_distance_band: str = ""
    coverage_target_height_band: str = ""
    coverage_actual_height_band: str = ""
    coverage_region_pair_index: int = -1
    nearest_direct_blocker_distance_m: float = 0.0


@dataclass
class ObservabilityAuditResult:
    """Runtime observability audit for a single frame or episode."""
    observability_check_triggered: bool = False
    observed_side_cost_valid: bool = False

    left_observed_path_cost: float = 0.0
    right_observed_path_cost: float = 0.0

    observed_lower_cost_side: str = ""
    observed_side_cost_difference_ratio: float = 0.0

    side_choice_consistent: bool = False
    observable_expert_label: bool = False

    invalid_observability_frame_count: int = 0


# ============================================================================
#  YamlCylinderSceneGenerator
# ============================================================================

class YamlCylinderSceneGenerator:
    """Generate density-driven catalogs or explicit fixed test scenes.

    density_driven is the sole dataset-generation path. fixed_scenario
    reuses the same geometry contract for deterministic expert diagnostics.
    """

    _SUPPORTED_SOURCES = ("density_driven", "fixed_scenario")

    def __init__(self, config):
        cfg = config.get("global", {}).get("scene_generation", {})
        self._cfg = cfg
        self._source = str(
            cfg.get("source", "density_driven")).strip()
        if self._source not in self._SUPPORTED_SOURCES:
            raise ValueError(
                "scene_generation.source must be one of {}, got '{}'"
                .format(self._SUPPORTED_SOURCES, self._source))

        region = cfg.get("obstacle_region", {})
        self.obstacle_region = ObstacleRegion(
            x_min=float(region.get("x_min", -8.0)),
            x_max=float(region.get("x_max", 8.0)),
            y_min=float(region.get("y_min", -6.0)),
            y_max=float(region.get("y_max", 6.0)),
            z_min=float(region.get("z_min", 0.0)),
            z_max=float(region.get("z_max", 6.0)),
        )

        execution = cfg.get("execution", {})
        self.max_scene_attempts = int(execution.get(
            "max_generation_attempts_per_scene", 500))
        self.max_obs_attempts = int(execution.get(
            "max_obstacle_sampling_attempts", 5000))

        generation_quality = cfg.get("generation_quality", {})
        self.minimum_density_achievement_ratio = float(
            generation_quality.get(
                "minimum_density_achievement_ratio", 0.97))
        if not 0.0 < self.minimum_density_achievement_ratio <= 1.0:
            raise ValueError(
                "scene_generation.generation_quality."
                "minimum_density_achievement_ratio must be in (0, 1]")

        stratification = cfg.get("placement_stratification", {})
        self.placement_stratification_enabled = bool(
            stratification.get("enabled", True))
        self.placement_x_bands = int(
            stratification.get("x_bands", 1))
        self.placement_y_bands = int(
            stratification.get("y_bands", 1))
        if self.placement_x_bands < 1 or self.placement_y_bands < 1:
            raise ValueError(
                "scene_generation.placement_stratification x_bands and "
                "y_bands must both be >= 1")

        common_cylinder = cfg.get("common_cylinder", {})
        self.region_margin = float(common_cylinder.get(
            "region_boundary_margin_m", 0.30))
        self.base_seed = int(cfg.get("seed", 12345))

    def _make_np_rng(self, seed):
        """Create a reproducible NumPy generator."""
        return np.random.Generator(np.random.PCG64(int(seed)))


    @staticmethod
    def _allocate_group_budgets(groups, target_area, inflation_radius):
        """Allocate inflated-area budget while preserving every scale group.

        Each configured group first receives enough area for one minimum-size
        obstacle.  Remaining area is distributed by the normalized
        ``capacity_fraction`` values.  This makes small-heavy/balanced/
        large-heavy profiles real rather than cosmetic names.
        """
        if not groups:
            raise ValueError("at least one size group is required")
        target = float(target_area)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_area must be finite and positive")

        minimums = np.asarray([
            group.min_inflated_area_m2(inflation_radius)
            for group in groups
        ], dtype=np.float64)
        minimum_total = float(np.sum(minimums))
        if target + 1.0e-9 < minimum_total:
            return None

        fractions = np.asarray([
            group.capacity_fraction for group in groups
        ], dtype=np.float64)
        if (not np.all(np.isfinite(fractions)) or
                np.any(fractions <= 0.0)):
            raise ValueError(
                "size-group capacity fractions must be finite and positive")
        fractions /= float(np.sum(fractions))
        remaining = max(0.0, target - minimum_total)
        budgets = minimums + remaining * fractions
        return {
            group.name: float(budget)
            for group, budget in zip(groups, budgets)
        }

    def generate_scene(self, attempt_sub_seed=0):
        """Load the explicit obstacle list for a fixed diagnostic scene."""
        del attempt_sub_seed
        if self._source != "fixed_scenario":
            raise RuntimeError(
                "generate_scene is only valid for source='fixed_scenario'")

        fixed_specs = self._cfg.get("fixed_obstacles", [])
        common_cylinder = self._cfg.get("common_cylinder", {})
        default_height = 0.5 * (
            float(common_cylinder.get("height_min_m", 8.0)) +
            float(common_cylinder.get("height_max_m", 8.0)))

        obstacles = []
        seen_ids = set()
        for index, item in enumerate(fixed_specs):
            try:
                obstacle_id = str(
                    item.get("id", "fixed_{:03d}".format(index)))
                center = np.asarray(item["center"], dtype=np.float64)
                radius = float(item["radius_m"])
                height = float(item.get("height_m", default_height))
            except (KeyError, TypeError, ValueError):
                return [], "FIXED_OBSTACLE_SCHEMA_INVALID"
            if (obstacle_id in seen_ids or center.shape != (3,) or
                    not np.all(np.isfinite(center)) or radius <= 0.0 or
                    height <= 0.0):
                return [], "FIXED_OBSTACLE_SCHEMA_INVALID"
            if not self.obstacle_region.contains_cylinder(
                    center[:2], radius, self.region_margin):
                return [], "FIXED_OBSTACLE_OUTSIDE_REGION"
            if (center[2] - 0.5 * height <
                    self.obstacle_region.z_min - 1.0e-6 or
                    center[2] + 0.5 * height >
                    self.obstacle_region.z_max + 1.0e-6):
                return [], "FIXED_OBSTACLE_OUTSIDE_VERTICAL_REGION"
            seen_ids.add(obstacle_id)
            obstacles.append(CylinderObstacleSpec(
                obstacle_id=obstacle_id,
                center_world=center,
                radius_m=radius,
                height_m=height))

        if not obstacles:
            return [], "FIXED_OBSTACLE_LIST_EMPTY"
        return obstacles, ""

    # ── Density-driven dataset generation ────────────────────────────


    def generate_scene_density_driven(self, profile, effective_scene_seed,
                                       scene_index_in_profile, attempt_index=0):
        """Generate a stratified, capacity-controlled cylinder scene.

        Each size group is guaranteed at least one feasible obstacle, and the
        remaining inflated-area budget follows ``capacity_fraction``.  The
        caller must retry any layout that cannot realise the configured
        minimum fraction of its sampled target density.
        """
        rng = self._make_np_rng(effective_scene_seed + attempt_index * 10007)
        region_area = compute_region_area(self.obstacle_region)
        infl_r = profile.inflation_radius_m

        # Every scale group is part of the new density contract.
        if not profile.size_groups or len(profile.size_groups) != 3:
            return ([], "DENSITY_DRIVEN_NEEDS_EXACTLY_3_SIZE_GROUPS",
                    None, "inflated_occupancy")

        groups = sorted(
            profile.size_groups,
            key=lambda group: group.radius_max_m,
            reverse=True)
        g_large, g_medium, g_small = groups
        fraction_sum = float(sum(
            group.capacity_fraction for group in groups))
        if abs(fraction_sum - 1.0) > 1.0e-6:
            return ([], "DENSITY_DRIVEN_CAPACITY_FRACTIONS_MUST_SUM_TO_ONE",
                    None, "inflated_occupancy")

        target_density = float(rng.uniform(
            profile.total_density_min, profile.total_density_max))
        target_inflated_area = region_area * target_density

        # Reserve one minimum-radius obstacle in every group, then allocate
        # the discretionary area by the configured capacity fractions.
        minimum_areas = [
            group.min_inflated_area_m2(infl_r) for group in groups]
        mandatory_area = float(sum(minimum_areas))
        if target_inflated_area + 1.0e-9 < mandatory_area:
            return ([], "DENSITY_DRIVEN_AREA_TOO_SMALL_FOR_GROUP_MINIMA",
                    target_density, "inflated_occupancy")
        discretionary_area = target_inflated_area - mandatory_area
        budgets = [
            minimum_area + discretionary_area * group.capacity_fraction
            for group, minimum_area in zip(groups, minimum_areas)
        ]
        large_budget, medium_budget, small_budget = budgets

        # ── 4. Placement ──
        obstacles = []
        obstacle_id_counter = 0
        boundary_margin = profile.region_boundary_margin_m
        min_surface_gap = profile.minimum_surface_gap_m
        min_post_inflation_gap = profile.minimum_post_inflation_gap_m
        height_min = profile.height_min_m
        height_max = profile.height_max_m
        region_width = (
            self.obstacle_region.x_max - self.obstacle_region.x_min)
        region_height = (
            self.obstacle_region.y_max - self.obstacle_region.y_min)
        aspect_ratio = region_width / max(region_height, 1.0e-9)

        def _inflated_area(radius):
            return math.pi * (radius + infl_r) ** 2

        def _try_place(group, budget_limit, obstacles_list, id_counter):
            """Place obstacles from one size group.

            Returns: (new_obstacles_or_none, spent_area, updated_id_counter).
            """
            new_obs = []
            local_spent = 0.0
            local_id = id_counter
            attempts_used = 0
            min_area = group.min_inflated_area_m2(infl_r)
            max_area = group.max_inflated_area_m2(infl_r)
            planned_areas = _plan_obstacle_areas(
                rng, budget_limit, min_area, max_area)
            if not planned_areas:
                return None, local_spent, local_id
            sampler = _StratifiedGridSampler(
                rng,
                target_cell_count=max(4, 2 * len(planned_areas)),
                aspect_ratio=aspect_ratio,
                minimum_x_bands=(
                    self.placement_x_bands
                    if self.placement_stratification_enabled else 1),
                minimum_y_bands=(
                    self.placement_y_bands
                    if self.placement_stratification_enabled else 1))

            for area in planned_areas:
                radius = (
                    math.sqrt(max(area, 0.0) / math.pi) - infl_r)
                radius = min(max(
                    radius, group.radius_min_m), group.radius_max_m)
                placed = False
                consecutive_fails = 0
                while (attempts_used < self.max_obs_attempts and
                       consecutive_fails <
                       group.consecutive_fail_threshold):
                    height = float(rng.uniform(height_min, height_max))
                    z_min_bound = (
                        self.obstacle_region.z_min + height / 2.0)
                    z_max_bound = (
                        self.obstacle_region.z_max - height / 2.0)
                    if z_min_bound > z_max_bound + 1.0e-9:
                        return None, local_spent, local_id
                    if abs(z_max_bound - z_min_bound) <= 1.0e-9:
                        z = 0.5 * (z_min_bound + z_max_bound)
                    else:
                        z = float(rng.uniform(
                            z_min_bound, z_max_bound))

                    effective_margin = radius + boundary_margin
                    sampled_xy = sampler.draw(
                        self.obstacle_region.x_min + effective_margin,
                        self.obstacle_region.x_max - effective_margin,
                        self.obstacle_region.y_min + effective_margin,
                        self.obstacle_region.y_max - effective_margin)
                    attempts_used += 1
                    if sampled_xy is None:
                        return None, local_spent, local_id
                    cx, cy = sampled_xy
                    center_xy = np.array(
                        [cx, cy], dtype=np.float64)

                    pairwise_ok = True
                    for previous in obstacles_list + new_obs:
                        distance = float(np.linalg.norm(
                            center_xy - previous.center_xy()))
                        surface_gap = (
                            distance - radius - previous.radius_m)
                        post_gap = surface_gap - 2.0 * infl_r
                        if (surface_gap + 1.0e-9 <
                                min_surface_gap or
                                post_gap + 1.0e-9 <
                                min_post_inflation_gap):
                            pairwise_ok = False
                            break
                    if not pairwise_ok:
                        consecutive_fails += 1
                        continue

                    new_obs.append(CylinderObstacleSpec(
                        obstacle_id="cylinder_{:04d}".format(local_id),
                        center_world=np.array(
                            [cx, cy, z], dtype=np.float64),
                        radius_m=radius,
                        height_m=height))
                    local_id += 1
                    local_spent += _inflated_area(radius)
                    placed = True
                    break
                if not placed:
                    return None, local_spent, local_id

            return new_obs, local_spent, local_id

        placed_by_group = OrderedDict()
        for group, budget in zip(groups, budgets):
            group_obstacles, _spent, obstacle_id_counter = _try_place(
                group, budget, obstacles, obstacle_id_counter)
            if group_obstacles is None:
                return (
                    [],
                    "DENSITY_DRIVEN_GROUP_MINIMUM_PLACEMENT_FAILED_{}".format(
                        group.name.upper()),
                    target_density,
                    "inflated_occupancy")
            obstacles.extend(group_obstacles)
            placed_by_group[group.name] = len(group_obstacles)

        actual_density = compute_inflated_occupancy(
            obstacles, region_area,
            profile.vehicle_radius_m, profile.safety_margin_m)
        achievement_ratio = (
            actual_density / target_density
            if target_density > 1.0e-12 else 1.0)
        if (achievement_ratio + 1.0e-9 <
                self.minimum_density_achievement_ratio):
            rospy.logwarn(
                "[DensityDriven] profile='%s' scene=%d rejected: "
                "density %.4f / %.4f = %.1f%%, required %.1f%%",
                profile.name, scene_index_in_profile,
                actual_density, target_density,
                100.0 * achievement_ratio,
                100.0 * self.minimum_density_achievement_ratio)
            return ([], "DENSITY_DRIVEN_TARGET_DENSITY_UNDERSHOT",
                    target_density, "inflated_occupancy")

        rospy.loginfo(
            "[DensityDriven] profile='%s' scene=%d: placed %d obstacles "
            "(L=%d M=%d S=%d), density=%.3f (target=%.3f, %.1f%%), "
            "capacity=(%.2f,%.2f,%.2f), budget=(%.1f,%.1f,%.1f) m2",
            profile.name, scene_index_in_profile, len(obstacles),
            placed_by_group.get("large", 0),
            placed_by_group.get("medium", 0),
            placed_by_group.get("small", 0),
            actual_density, target_density, 100.0 * achievement_ratio,
            groups[0].capacity_fraction,
            groups[1].capacity_fraction,
            groups[2].capacity_fraction,
            large_budget, medium_budget, small_budget)

        return (obstacles, "", target_density, "inflated_occupancy")

    def generate_unity_objects(self, obstacles):
        """Convert CylinderObstacleSpec list to Unity Object_t dicts."""
        obj_list = []
        for obs in obstacles:
            obj_list.append({
                "ID": obs.obstacle_id,
                "prefabID": "Object",
                "position": [float(obs.center_world[0]),
                             float(obs.center_world[2]),
                             float(obs.center_world[1])],  # ROS → Unity
                "rotation": [0, 0, 0, 1],
                "size": [obs.diameter_m(), obs.height_m, obs.diameter_m()],
            })
        return obj_list


# ============================================================================
#  CylinderSceneValidator — 2D topology checks
# ============================================================================

class CylinderSceneValidator:
    """Simplified 2D topology validator (v10).

    Keeps only the essential checks:
      - Enclosed free components (unreachable from region exterior).
      - Inflated obstacle component merging.
    Removed: escape-sector U-shape/dead-end analysis, minimum corridor width.
    """

    def __init__(self, config):
        cfg = config.get("global", {}).get("scene_generation", {})
        topo = cfg.get("topology_validation", {})
        vehicle_cfg = cfg.get("vehicle", {})
        self._cfg = cfg

        self.res = float(topo.get("grid_resolution_m", 0.10))
        self.halo_m = float(topo.get("validation_halo_m", 3.0))
        self.vehicle_r = float(vehicle_cfg.get("radius_m",
            topo.get("vehicle_radius_m", 0.30)))
        self.safety_m = float(vehicle_cfg.get("safety_margin_m",
            topo.get("safety_margin_m", 0.10)))
        self.inflated_extra = self.vehicle_r + self.safety_m

        # Simplified flags (read from density_driven config or topo compat)
        self.forbid_enclosed = bool(cfg.get("forbid_enclosed_free_components",
            topo.get("forbid_enclosed_free_components", True)))
        self.forbid_merge = bool(cfg.get("forbid_inflated_component_merging",
            not cfg.get("cylinder", {}).get("allow_inflated_component_merging", False)))

    def validate(self, obstacles, obstacle_region):
        """Run simplified topology validation. Returns SceneValidationResult."""
        result = SceneValidationResult()
        result.obstacle_count = len(obstacles)

        if len(obstacles) == 0:
            result.valid = True
            return result

        # Compute pairwise minimum gaps (informational)
        min_surface_gap = float('inf')
        min_inflated_gap = float('inf')
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                a, b = obstacles[i], obstacles[j]
                d = float(np.linalg.norm(a.center_xy() - b.center_xy()))
                sg = d - a.radius_m - b.radius_m
                ig = d - (a.radius_m + self.inflated_extra) - (b.radius_m + self.inflated_extra)
                if sg < min_surface_gap:
                    min_surface_gap = sg
                if ig < min_inflated_gap:
                    min_inflated_gap = ig
        result.minimum_surface_gap_m = (min_surface_gap if min_surface_gap != float('inf') else 0.0)
        result.minimum_inflated_gap_m = (min_inflated_gap if min_inflated_gap != float('inf') else 0.0)

        # Build 2D inflated occupancy grid with halo
        x_min = obstacle_region.x_min - self.halo_m
        x_max = obstacle_region.x_max + self.halo_m
        y_min = obstacle_region.y_min - self.halo_m
        y_max = obstacle_region.y_max + self.halo_m

        gx = int(math.ceil((x_max - x_min) / self.res))
        gy = int(math.ceil((y_max - y_min) / self.res))
        if gx < 3 or gy < 3:
            result.valid = True
            return result

        occ = np.zeros((gx, gy), dtype=np.uint8)
        origin = np.array([x_min, y_min])

        def in_bounds(ix, iy):
            return 0 <= ix < gx and 0 <= iy < gy

        # Mark inflated cylinders on the grid
        for obs in obstacles:
            infl_r = obs.radius_m + self.inflated_extra
            cx, cy = obs.center_xy()
            gx0 = int(math.floor((cx - origin[0]) / self.res))
            gy0 = int(math.floor((cy - origin[1]) / self.res))
            ir = int(math.ceil(infl_r / self.res))
            for dx in range(-ir, ir + 1):
                for dy in range(-ir, ir + 1):
                    ix, iy = gx0 + dx, gy0 + dy
                    if not in_bounds(ix, iy):
                        continue
                    wx = origin[0] + (ix + 0.5) * self.res
                    wy = origin[1] + (iy + 0.5) * self.res
                    if (wx - cx)**2 + (wy - cy)**2 <= infl_r**2:
                        occ[ix, iy] = 1

        # ── 1. Enclosed free component check ──
        if self.forbid_enclosed:
            free_mask = (occ == 0)
            visited = np.zeros((gx, gy), dtype=bool)
            from collections import deque
            q = deque()
            # Seed all edge free cells
            for ix in range(gx):
                for iy in [0, gy - 1]:
                    if free_mask[ix, iy]:
                        visited[ix, iy] = True
                        q.append((ix, iy))
            for iy in range(gy):
                for ix in [0, gx - 1]:
                    if free_mask[ix, iy] and not visited[ix, iy]:
                        visited[ix, iy] = True
                        q.append((ix, iy))

            while q:
                ix, iy = q.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = ix + dx, iy + dy
                    if in_bounds(nx, ny) and free_mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        q.append((nx, ny))

            enclosed_count = int(np.sum(free_mask & ~visited))
            result.enclosed_free_component_count = enclosed_count
            if enclosed_count > 0:
                result.valid = False
                result.rejection_reason = "SCENE_ENCLOSED_FREE_COMPONENT"
                return result

            # Navigable free ratio (informational)
            navigable = free_mask & visited
            result.navigable_free_ratio = float(np.sum(navigable)) / max(gx * gy, 1)
        else:
            result.navigable_free_ratio = 1.0
            result.enclosed_free_component_count = 0

        # Informational: distance-transform based minimum clearance
        try:
            from scipy.ndimage import distance_transform_edt
            dt = distance_transform_edt(occ == 0, sampling=self.res)
            free_mask = (occ == 0)
            if np.any(free_mask):
                result.minimum_navigable_clearance_m = float(np.max(dt[free_mask]))
        except ImportError:
            result.minimum_navigable_clearance_m = 0.0

        # ── 2. Inflated component merge check ──
        if self.forbid_merge:
            comp_labels = np.zeros((gx, gy), dtype=np.int32)
            comp_id = 0
            for ix in range(gx):
                for iy in range(gy):
                    if occ[ix, iy] and comp_labels[ix, iy] == 0:
                        comp_id += 1
                        stack = [(ix, iy)]
                        comp_labels[ix, iy] = comp_id
                        while stack:
                            cx_, cy_ = stack.pop()
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = cx_ + dx, cy_ + dy
                                if in_bounds(nx, ny) and occ[nx, ny] and comp_labels[nx, ny] == 0:
                                    comp_labels[nx, ny] = comp_id
                                    stack.append((nx, ny))
            result.inflated_components = comp_id
            if comp_id != len(obstacles):
                result.valid = False
                result.rejection_reason = "SCENE_INFLATED_OBSTACLE_COMPONENT_MERGE"
                return result
        else:
            result.inflated_components = len(obstacles)

        # Not checked in simplified validator
        result.u_shape_detected = False
        result.dead_end_detected = False
        result.dead_end_max_depth_m = 0.0

        result.valid = True
        return result


# ============================================================================
#  StartGoalTaskGenerator
# ============================================================================

def _region_from_dict(d, z_min, z_max):
    """Build an ObstacleRegion from a dict with optional z_min/z_max override."""
    return ObstacleRegion(
        x_min=float(d.get("x_min", -8.0)),
        x_max=float(d.get("x_max", 8.0)),
        y_min=float(d.get("y_min", -2.0)),
        y_max=float(d.get("y_max", 2.0)),
        z_min=float(d.get("z_min", z_min)),
        z_max=float(d.get("z_max", z_max)),
    )


class StartGoalTaskGenerator:
    """Generate and validate start-goal task pairs."""

    def __init__(self, config):
        cfg = config.get("global", {}).get(
            "scene_generation", {}).get(
                "common_task_generation", {})
        self._cfg = cfg

        self.max_attempts = int(cfg.get("max_task_sampling_attempts", 1000))
        self.tasks_per_scene = int(cfg.get("tasks_per_scene", 20))

        self.start_h_min = float(cfg.get("start_height_min_m", 1.8))
        self.start_h_max = float(cfg.get("start_height_max_m", 2.2))
        self.goal_h_min = float(cfg.get("goal_height_min_m", 1.8))
        self.goal_h_max = float(cfg.get("goal_height_max_m", 2.2))
        self.max_h_diff = float(cfg.get("maximum_start_goal_height_difference_m", 0.20))

        self.start_clearance = float(cfg.get("start_clearance_m", 0.60))
        self.goal_clearance = float(cfg.get("goal_clearance_m", 0.60))

        self.min_dist = float(cfg.get("minimum_start_goal_distance_m", 8.0))
        self.max_dist = float(cfg.get("maximum_start_goal_distance_m", 30.0))

        self.require_blocked = bool(cfg.get("require_direct_path_blocked", True))
        self.corridor_r = float(cfg.get("direct_path_corridor_radius_m", 0.35))
        self.min_blockers = int(cfg.get("minimum_direct_blocker_count", 1))
        self.max_blockers = int(cfg.get("maximum_direct_blocker_count", 3))

        self.require_astar = bool(cfg.get("require_astar_reachable", True))
        self.min_detour = float(cfg.get("minimum_detour_ratio", 1.05))
        self.max_detour = float(cfg.get("maximum_detour_ratio", 1.80))
        self.fixed_tasks = list(cfg.get("fixed_tasks", []))

        start_sampling_regions = cfg.get("start_sampling_regions", [
            {"x_min": -12.0, "x_max": -8.0,
             "y_min": -8.0, "y_max": 8.0},
        ])
        goal_sampling_regions = cfg.get("goal_sampling_regions", [
            {"x_min": 8.0, "x_max": 12.0,
             "y_min": -8.0, "y_max": 8.0},
        ])
        self.start_regions = [
            _region_from_dict(
                region, self.start_h_min, self.start_h_max)
            for region in start_sampling_regions
        ]
        self.goal_regions = [
            _region_from_dict(
                region, self.goal_h_min, self.goal_h_max)
            for region in goal_sampling_regions
        ]

        self._coverage_density_tier = "default"
        self._configure_coverage(
            cfg.get("coverage_balancing", {}),
            density_tier=self._coverage_density_tier)

    @staticmethod
    def _normalise_weights(raw, allowed, field_name):
        if not isinstance(raw, dict):
            raise ValueError("{} must be a mapping".format(field_name))
        weights = OrderedDict()
        for key, value in raw.items():
            name = str(key)
            if name not in allowed:
                raise ValueError(
                    "{} contains unsupported bucket '{}'".format(
                        field_name, name))
            weight = float(value)
            if weight < 0.0:
                raise ValueError(
                    "{} weight '{}' must be >= 0".format(field_name, name))
            if weight > 0.0:
                weights[name] = weight
        if not weights:
            raise ValueError("{} needs at least one positive weight".format(
                field_name))
        return weights

    def _configure_coverage(self, coverage_cfg, density_tier="default"):
        """Resolve configurable task-coverage quotas.

        The buckets are geometric proxies available before a trajectory is
        flown.  They intentionally do not claim to be exact Guide labels.
        """
        cfg = copy.deepcopy(coverage_cfg) if isinstance(coverage_cfg, dict) else {}
        self.coverage_enabled = bool(cfg.get("enabled", False))
        self._coverage_density_tier = str(
            density_tier or "default").lower()
        self.coverage_rotate_region_pairs = bool(
            cfg.get("rotate_region_pairs_by_seed", True))
        self.coverage_center_lateral_fraction = float(
            cfg.get("center_lateral_fraction", 0.25))
        if not 0.0 <= self.coverage_center_lateral_fraction <= 1.0:
            raise ValueError(
                "coverage_balancing.center_lateral_fraction must be in [0, 1]")

        type_defaults = OrderedDict([
            ("clear", 0.25),
            ("single_left", 0.20),
            ("single_right", 0.20),
            ("single_center", 0.15),
            ("multi_blocker", 0.20),
        ])
        by_tier = cfg.get("task_type_weights_by_density_tier", {})
        raw_type_weights = (
            by_tier.get(self._coverage_density_tier)
            if isinstance(by_tier, dict) else None)
        if raw_type_weights is None and isinstance(by_tier, dict):
            raw_type_weights = by_tier.get("default")
        if raw_type_weights is None:
            raw_type_weights = cfg.get("task_type_weights", type_defaults)
        self.coverage_task_type_weights = self._normalise_weights(
            raw_type_weights,
            {"any", "clear", "single_left", "single_right",
             "single_center", "multi_blocker"},
            "coverage_balancing.task_type_weights")
        if self._coverage_density_tier == "sparse":
            self.coverage_task_type_weights.pop("multi_blocker", None)
        elif self._coverage_density_tier == "dense":
            self.coverage_task_type_weights.pop("clear", None)
        if not self.coverage_task_type_weights:
            raise ValueError(
                "coverage task weights are empty after density-tier "
                "feasibility filtering")

        raw_distance_bands = cfg.get(
            "blocker_distance_bands_m",
            OrderedDict([
                ("near", [0.0, 6.0]),
                ("middle", [6.0, 15.0]),
                ("far", [15.0, 1000.0]),
            ]))
        if not isinstance(raw_distance_bands, dict):
            raise ValueError(
                "coverage_balancing.blocker_distance_bands_m must be a mapping")
        self.coverage_distance_bands = OrderedDict()
        for name, limits in raw_distance_bands.items():
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                raise ValueError(
                    "blocker distance band '{}' must be [min, max]".format(
                        name))
            lower, upper = float(limits[0]), float(limits[1])
            if lower < 0.0 or upper <= lower:
                raise ValueError(
                    "invalid blocker distance band '{}': {}".format(
                        name, limits))
            self.coverage_distance_bands[str(name)] = (lower, upper)
        distance_weights_by_tier = cfg.get(
            "blocker_distance_weights_by_density_tier", {})
        raw_distance_weights = (
            distance_weights_by_tier.get(self._coverage_density_tier)
            if isinstance(distance_weights_by_tier, dict) else None)
        if raw_distance_weights is None and isinstance(
                distance_weights_by_tier, dict):
            raw_distance_weights = distance_weights_by_tier.get("default")
        if raw_distance_weights is None:
            raw_distance_weights = cfg.get(
                "blocker_distance_weights",
                OrderedDict([
                    ("near", 0.40),
                    ("middle", 0.40),
                    ("far", 0.20),
                ]))
        self.coverage_distance_weights = self._normalise_weights(
            raw_distance_weights,
            set(self.coverage_distance_bands),
            "coverage_balancing.blocker_distance_weights")
        if self._coverage_density_tier == "dense":
            self.coverage_distance_weights.pop("far", None)
        if not self.coverage_distance_weights:
            raise ValueError(
                "coverage distance weights are empty after density-tier "
                "feasibility filtering")

        self.coverage_level_height_delta_max = float(
            cfg.get("level_height_delta_max_m", 0.25))
        self.coverage_nonlevel_height_delta_min = float(
            cfg.get("minimum_nonlevel_height_delta_m", 0.75))
        if (self.coverage_level_height_delta_max < 0.0 or
                self.coverage_nonlevel_height_delta_min <=
                self.coverage_level_height_delta_max):
            raise ValueError(
                "coverage height thresholds need 0 <= level_max < nonlevel_min")
        self.coverage_height_weights = self._normalise_weights(
            cfg.get("height_delta_weights",
                    OrderedDict([
                        ("level", 1.0),
                        ("ascending", 0.0),
                        ("descending", 0.0),
                    ])),
            {"any", "level", "ascending", "descending"},
            "coverage_balancing.height_delta_weights")

    @staticmethod
    def _quota_sequence(weights, count, rng):
        """Build an exact largest-remainder quota and shuffle deterministically."""
        if count <= 0:
            return []
        names = list(weights.keys())
        total = float(sum(weights.values()))
        raw_counts = [float(weights[name]) / total * count for name in names]
        counts = [int(math.floor(value)) for value in raw_counts]
        remainder = count - sum(counts)
        order = sorted(
            range(len(names)),
            key=lambda index: (raw_counts[index] - counts[index], -index),
            reverse=True)
        for index in order[:remainder]:
            counts[index] += 1
        sequence = []
        for name, bucket_count in zip(names, counts):
            sequence.extend([name] * bucket_count)
        rng.shuffle(sequence)
        return sequence

    def _effective_task_type_weights(self):
        weights = OrderedDict(self.coverage_task_type_weights)
        if self.require_blocked:
            weights.pop("clear", None)
        if self.min_blockers >= 2:
            for name in ("single_left", "single_right", "single_center"):
                weights.pop(name, None)
        if self.max_blockers < 2:
            weights.pop("multi_blocker", None)
        if not weights:
            weights["any"] = 1.0
        return weights

    def _build_coverage_targets(self, count, seed):
        if not self.coverage_enabled:
            return [None] * count
        schedule_rng = random.Random(int(seed) ^ 0x5EEDC0DE)
        task_types = self._quota_sequence(
            self._effective_task_type_weights(), count, schedule_rng)
        blocked_count = sum(
            task_type not in ("clear", "any")
            for task_type in task_types)
        distances = self._quota_sequence(
            self.coverage_distance_weights, blocked_count, schedule_rng)
        heights = self._quota_sequence(
            self.coverage_height_weights, count, schedule_rng)
        distance_index = 0
        targets = []
        for task_index, task_type in enumerate(task_types):
            distance_band = ""
            if task_type not in ("clear", "any"):
                distance_band = distances[distance_index]
                distance_index += 1
            targets.append({
                "task_type": task_type,
                "blocker_distance_band": distance_band,
                "height_band": heights[task_index],
            })
        return targets

    def _sample_height_pair(self, rng, target_height_band):
        last_pair = (self.start_h_min, self.goal_h_min)
        for _ in range(64):
            start_z = rng.uniform(self.start_h_min, self.start_h_max)
            goal_z = rng.uniform(self.goal_h_min, self.goal_h_max)
            last_pair = (start_z, goal_z)
            if abs(goal_z - start_z) > self.max_h_diff:
                continue
            if (target_height_band in ("", "any") or
                    self._height_band(goal_z - start_z) ==
                    target_height_band):
                return last_pair
        return last_pair

    def _height_band(self, height_delta):
        if abs(float(height_delta)) <= self.coverage_level_height_delta_max:
            return "level"
        if float(height_delta) >= self.coverage_nonlevel_height_delta_min:
            return "ascending"
        if float(height_delta) <= -self.coverage_nonlevel_height_delta_min:
            return "descending"
        return "transition"

    def _blocker_distance_band(self, distance_m):
        if distance_m is None:
            return ""
        for name, (lower, upper) in self.coverage_distance_bands.items():
            if lower <= float(distance_m) < upper:
                return name
        return "out_of_range"

    def _classify_direct_path(self, start, goal, obstacles):
        """Classify direct-corridor geometry, including obstacle height."""
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        direct = goal - start
        direct_xy = direct[:2]
        xy_length_sq = float(np.dot(direct_xy, direct_xy))
        if xy_length_sq < 1.0e-12:
            return {
                "task_type": "clear",
                "blockers": [],
                "dominant": None,
                "nearest_distance_m": None,
                "dominant_lateral_offset_m": 0.0,
            }

        xy_length = math.sqrt(xy_length_sq)
        e_xy = direct_xy / xy_length
        n_xy = np.array([-e_xy[1], e_xy[0]], dtype=np.float64)
        direct_length = float(np.linalg.norm(direct))
        blockers = []
        for obstacle in obstacles:
            relative_xy = obstacle.center_xy() - start[:2]
            fraction = float(np.dot(relative_xy, direct_xy) / xy_length_sq)
            if not 0.0 < fraction < 1.0:
                continue
            closest_xy = start[:2] + fraction * direct_xy
            lateral_offset = float(np.dot(
                obstacle.center_xy() - closest_xy, n_xy))
            lateral_distance = abs(lateral_offset)
            path_z = float(start[2] + fraction * direct[2])
            half_height = 0.5 * float(obstacle.height_m)
            obstacle_bottom = float(obstacle.center_world[2]) - half_height
            obstacle_top = float(obstacle.center_world[2]) + half_height
            vertical_gap = max(
                obstacle_bottom - path_z, path_z - obstacle_top, 0.0)
            lateral_surface_gap = max(
                lateral_distance - float(obstacle.radius_m), 0.0)
            if math.hypot(lateral_surface_gap, vertical_gap) > self.corridor_r:
                continue
            blockers.append({
                "obstacle": obstacle,
                "fraction": fraction,
                "distance_m": fraction * direct_length,
                "lateral_offset_m": lateral_offset,
            })

        blockers.sort(key=lambda item: item["distance_m"])
        if not blockers:
            task_type = "clear"
            dominant = None
            nearest_distance = None
            lateral_offset = 0.0
        elif len(blockers) >= 2:
            task_type = "multi_blocker"
            dominant = blockers[0]
            nearest_distance = dominant["distance_m"]
            lateral_offset = dominant["lateral_offset_m"]
        else:
            dominant = blockers[0]
            nearest_distance = dominant["distance_m"]
            lateral_offset = dominant["lateral_offset_m"]
            normalizer = (
                float(dominant["obstacle"].radius_m) + self.corridor_r)
            centered = (
                abs(lateral_offset) <=
                self.coverage_center_lateral_fraction * normalizer)
            if centered:
                task_type = "single_center"
            elif lateral_offset > 0.0:
                task_type = "single_left"
            else:
                task_type = "single_right"

        return {
            "task_type": task_type,
            "blockers": blockers,
            "dominant": dominant,
            "nearest_distance_m": nearest_distance,
            "dominant_lateral_offset_m": lateral_offset,
        }

    def _matches_coverage_target(self, target, classification, start, goal):
        if target is None:
            return True
        task_type = target.get("task_type", "any")
        if (task_type != "any" and
                classification["task_type"] != task_type):
            return False
        distance_band = target.get("blocker_distance_band", "")
        if (distance_band and self._blocker_distance_band(
                classification["nearest_distance_m"]) != distance_band):
            return False
        height_band = target.get("height_band", "any")
        if (height_band != "any" and
                self._height_band(float(goal[2]) - float(start[2])) !=
                height_band):
            return False
        return True

    def _annotate_coverage_result(
            self, result, target, classification, start, goal,
            region_pair_index):
        if target is None:
            return
        result.coverage_target_task_type = target.get("task_type", "")
        result.coverage_actual_task_type = classification["task_type"]
        result.coverage_target_blocker_distance_band = target.get(
            "blocker_distance_band", "")
        result.coverage_actual_blocker_distance_band = (
            self._blocker_distance_band(
                classification["nearest_distance_m"]))
        result.coverage_target_height_band = target.get("height_band", "")
        result.coverage_actual_height_band = self._height_band(
            float(goal[2]) - float(start[2]))
        result.coverage_region_pair_index = int(region_pair_index)
        result.nearest_direct_blocker_distance_m = float(
            classification["nearest_distance_m"] or 0.0)

    def configure_from_profile(self, profile):
        """Select density-tier coverage weights for the active profile."""
        if profile is None:
            return
        self._configure_coverage(
            profile.coverage_balancing,
            density_tier=profile.density_tier)


    def generate_tasks(self, obstacles, esdf, esdf_origin, esdf_res,
                       astar_planner_fn, seed=0):
        """Generate validated start-goal task pairs.

        Args:
            obstacles: list of CylinderObstacleSpec.
            esdf: global ESDF array.
            esdf_origin: (ox, oy, oz).
            esdf_res: float.
            astar_planner_fn: function(esdf, origin, res, start, goal, min_cl) -> PlanResult.
            seed: random seed.

        Returns:
            list of (start, goal, TaskValidationResult).
        """
        rng = random.Random(int(seed))
        tasks = []

        if self.fixed_tasks:
            for task_idx, item in enumerate(self.fixed_tasks):
                try:
                    start = np.asarray(item["start"], dtype=np.float64)
                    goal = np.asarray(item["goal"], dtype=np.float64)
                except (KeyError, TypeError, ValueError):
                    raise ValueError(
                        "fixed task {} has an invalid schema".format(
                            task_idx))
                if (start.shape != (3,) or goal.shape != (3,) or
                        not np.all(np.isfinite(start)) or
                        not np.all(np.isfinite(goal))):
                    raise ValueError(
                        "fixed task {} must contain finite 3-D start/goal"
                        .format(task_idx))
                result = self._validate_task(
                    start, goal, obstacles, esdf, esdf_origin, esdf_res,
                    astar_planner_fn)
                if not result.valid:
                    raise ValueError(
                        "fixed task {} rejected: {}".format(
                            task_idx, result.rejection_reason))
                tasks.append((start.tolist(), goal.tolist(), result))
            return tasks

        targets = self._build_coverage_targets(
            self.tasks_per_scene, seed)
        pair_count = min(len(self.start_regions), len(self.goal_regions))
        if pair_count <= 0:
            raise ValueError(
                "Task generation needs at least one start/goal region pair")
        pair_offset = 0
        if self.coverage_enabled and self.coverage_rotate_region_pairs:
            pair_offset = int(seed) % pair_count

        def draw_candidate(start_region, goal_region, height_band):
            sx = rng.uniform(start_region.x_min, start_region.x_max)
            sy = rng.uniform(start_region.y_min, start_region.y_max)
            gx = rng.uniform(goal_region.x_min, goal_region.x_max)
            gy = rng.uniform(goal_region.y_min, goal_region.y_max)
            sz, gz = self._sample_height_pair(rng, height_band)
            return (np.array([sx, sy, sz], dtype=np.float64),
                    np.array([gx, gy, gz], dtype=np.float64))

        for task_idx in range(self.tasks_per_scene):
            accepted = False
            target = targets[task_idx]
            initial_pair_idx = (task_idx + pair_offset) % pair_count
            target_height = (
                target.get("height_band", "any")
                if target is not None else "any")

            # Strict phase: cheap geometric classification happens before A*,
            # so quota retries do not repeatedly invoke the expensive planner.
            # Rotate through every region pair instead of permanently binding
            # a bucket to one direction that may be structurally infeasible.
            for _attempt in range(self.max_attempts):
                pair_idx = (
                    initial_pair_idx + _attempt) % pair_count
                start_r = self.start_regions[pair_idx]
                goal_r = self.goal_regions[pair_idx]
                start, goal = draw_candidate(
                    start_r, goal_r, target_height)
                classification = self._classify_direct_path(
                    start, goal, obstacles)
                if not self._matches_coverage_target(
                        target, classification, start, goal):
                    continue
                result = self._validate_task(
                    start, goal, obstacles, esdf, esdf_origin, esdf_res,
                    astar_planner_fn)
                if not result.valid:
                    continue
                self._annotate_coverage_result(
                    result, target, classification, start, goal,
                    pair_idx)
                tasks.append((start.tolist(), goal.tolist(), result))
                accepted = True
                break

            if not accepted:
                target_description = (
                    "{}/{}/{}".format(
                        target.get("task_type", ""),
                        target.get("blocker_distance_band", ""),
                        target.get("height_band", ""))
                    if target is not None else "unconstrained")
                raise RuntimeError(
                    "Task coverage target unavailable: task={} target={} "
                    "initial_region_pair={} tried_region_pairs={} "
                    "attempts={}; rejecting the complete "
                    "scene instead of substituting a mismatched label"
                    .format(
                        task_idx, target_description, initial_pair_idx,
                        pair_count,
                        self.max_attempts))

        if len(tasks) != self.tasks_per_scene:
            raise RuntimeError(
                "Task generation incomplete: produced {}/{} validated tasks; "
                "rejecting the complete scene".format(
                    len(tasks), self.tasks_per_scene))
        return tasks

    def _validate_task(self, start, goal, obstacles, esdf, esdf_origin, esdf_res,
                       astar_planner_fn):
        """Validate a single start-goal pair. Returns TaskValidationResult."""
        result = TaskValidationResult()

        # Height check
        if abs(start[2] - goal[2]) > self.max_h_diff:
            result.rejection_reason = "TASK_HEIGHT_DIFFERENCE_EXCEEDED"
            return result

        # Distance check
        direct_vec = goal - start
        direct_dist = float(np.linalg.norm(direct_vec))
        result.direct_distance_m = direct_dist
        if direct_dist < self.min_dist:
            result.rejection_reason = "TASK_TOO_CLOSE"
            return result
        if direct_dist > self.max_dist:
            result.rejection_reason = "TASK_TOO_FAR"
            return result

        # Start/goal clearance (use ESDF)
        ox, oy, oz = esdf_origin
        inv = 1.0 / esdf_res

        def esdf_at(pt):
            gx_, gy_, gz_ = esdf.shape
            ix = max(0, min(gx_ - 1, int(math.floor((pt[0] - ox) * inv))))
            iy = max(0, min(gy_ - 1, int(math.floor((pt[1] - oy) * inv))))
            iz = max(0, min(gz_ - 1, int(math.floor((pt[2] - oz) * inv))))
            return float(esdf[ix, iy, iz])

        def path_length(path):
            return sum(float(np.linalg.norm(
                np.asarray(path[i], dtype=np.float64) -
                np.asarray(path[i - 1], dtype=np.float64)))
                for i in range(1, len(path)))

        if esdf_at(start) < self.start_clearance:
            result.rejection_reason = "TASK_START_OR_GOAL_NOT_FREE"
            return result
        if esdf_at(goal) < self.goal_clearance:
            result.rejection_reason = "TASK_START_OR_GOAL_NOT_FREE"
            return result

        # ── Direct path blocked check ──
        # Always classify the direct corridor for dataset metadata.  Blocking
        # is an optional acceptance requirement, allowing both straight-flight
        # and avoidance trajectories in the same generated dataset.
        classification = self._classify_direct_path(
            start, goal, obstacles)
        n_blockers = len(classification["blockers"])
        result.direct_blocker_count = n_blockers
        result.direct_path_blocked = (n_blockers > 0)
        result.nearest_direct_blocker_distance_m = float(
            classification["nearest_distance_m"] or 0.0)

        if self.require_blocked:
            if n_blockers < self.min_blockers:
                result.rejection_reason = "TASK_DIRECT_PATH_NOT_BLOCKED"
                return result
            if n_blockers > self.max_blockers:
                result.rejection_reason = "TASK_TOO_MANY_DIRECT_BLOCKERS"
                return result

        # ── A* reachability ──
        if self.require_astar and astar_planner_fn is not None:
            plan_result = astar_planner_fn(esdf, esdf_origin, esdf_res,
                                           start.tolist(), goal.tolist(),
                                           self.start_clearance)
            if not plan_result.reached_goal:
                result.rejection_reason = "TASK_ASTAR_UNREACHABLE"
                result.astar_reachable = False
                return result
            result.astar_reachable = True
            # Compute A* path length
            astar_len = 0.0
            for i in range(1, len(plan_result.path)):
                astar_len += float(np.linalg.norm(
                    np.array(plan_result.path[i]) - np.array(plan_result.path[i-1])))
            result.astar_length_m = astar_len
            result.detour_ratio = astar_len / max(direct_dist, 1e-6)

            if result.detour_ratio < self.min_detour:
                result.rejection_reason = "TASK_DETOUR_RATIO_TOO_LOW"
                return result
            if result.detour_ratio > self.max_detour:
                result.rejection_reason = "TASK_DETOUR_RATIO_TOO_HIGH"
                return result

        # ── Dominant obstacle identification ──
        dominant_info = classification["dominant"]
        dom_obs = (
            dominant_info["obstacle"]
            if dominant_info is not None else None)
        if dom_obs is not None:
            result.dominant_obstacle_id = dom_obs.obstacle_id
            result.dominant_obstacle_lateral_offset_m = float(
                classification["dominant_lateral_offset_m"])
        elif self.require_blocked:
            result.rejection_reason = "TASK_NO_DOMINANT_OBSTACLE"
            return result

        result.valid = True
        return result

    def _find_dominant_obstacle(self, start, goal, obstacles):
        """Find the primary blocking obstacle along the start→goal line."""
        direct_vec = goal[:2] - start[:2]
        direct_dist = float(np.linalg.norm(direct_vec))
        if direct_dist < 1e-6:
            return None
        e_dir = direct_vec / direct_dist

        best = None
        best_proj = float('inf')
        for obs in obstacles:
            obs_xy = obs.center_xy()
            proj = float(np.dot(obs_xy - start[:2], e_dir))
            if 0 < proj < direct_dist:
                closest = start[:2] + proj * e_dir
                lateral = float(np.linalg.norm(obs_xy - closest))
                if lateral < obs.radius_m + self.corridor_r:
                    if proj < best_proj:
                        best_proj = proj
                        best = obs
        return best


# ============================================================================
#  SideCostEvaluator
# ============================================================================

class SideCostEvaluator:
    """Compute left/right portal path costs around the dominant obstacle."""

    def __init__(self, config):
        cfg = config.get("global", {}).get("scene_generation", {}).get("side_cost", {})
        self._cfg = cfg
        topo = config.get("global", {}).get("scene_generation", {}).get("topology_validation", {})

        self.min_cost_diff_ratio = float(cfg.get("minimum_cost_difference_ratio", 0.15))
        self.portal_lat_clearance = float(cfg.get("portal_lateral_clearance_m", 0.60))
        self.portal_long_offset = float(cfg.get("portal_longitudinal_offset_m", 0.50))
        self.require_both = bool(cfg.get("require_both_sides_feasible", False))
        self.reject_equal = bool(cfg.get("reject_nearly_equal_sides", True))

        self.vehicle_r = float(topo.get("vehicle_radius_m", 0.30))
        self.safety_m = float(topo.get("safety_margin_m", 0.10))
        self.inflated_extra = self.vehicle_r + self.safety_m

    def evaluate(self, start, goal, dominant_obstacle, obstacles,
                 esdf, esdf_origin, esdf_res, astar_planner_fn):
        """Compute left/right costs. Returns (TaskValidationResult updated)."""
        result = TaskValidationResult()
        result.direct_distance_m = float(np.linalg.norm(goal - start))

        if dominant_obstacle is None:
            result.rejection_reason = "TASK_NO_DOMINANT_OBSTACLE"
            return result

        direct_vec = goal[:2] - start[:2]
        direct_dist = float(np.linalg.norm(direct_vec))
        if direct_dist < 1e-6:
            result.rejection_reason = "TASK_TOO_CLOSE"
            return result
        e_dir = direct_vec / direct_dist
        n_dir = np.array([-e_dir[1], e_dir[0]])

        infl_r = dominant_obstacle.radius_m + self.inflated_extra
        obs_xy = dominant_obstacle.center_xy()
        obs_proj = float(np.dot(obs_xy - start[:2], e_dir))

        # Left portal
        pL_xy = obs_xy + n_dir * (infl_r + self.portal_lat_clearance) + e_dir * self.portal_long_offset
        pL_z = 0.5 * (start[2] + goal[2])
        pL = np.array([pL_xy[0], pL_xy[1], pL_z])

        # Right portal
        pR_xy = obs_xy - n_dir * (infl_r + self.portal_lat_clearance) + e_dir * self.portal_long_offset
        pR_z = 0.5 * (start[2] + goal[2])
        pR = np.array([pR_xy[0], pR_xy[1], pR_z])

        # Check portal points are in free space
        ox, oy, oz = esdf_origin
        inv = 1.0 / esdf_res

        def esdf_at(pt):
            gx_, gy_, gz_ = esdf.shape
            ix = max(0, min(gx_ - 1, int(math.floor((pt[0] - ox) * inv))))
            iy = max(0, min(gy_ - 1, int(math.floor((pt[1] - oy) * inv))))
            iz = max(0, min(gz_ - 1, int(math.floor((pt[2] - oz) * inv))))
            return float(esdf[ix, iy, iz])

        def path_length(path):
            return sum(float(np.linalg.norm(
                np.asarray(path[i], dtype=np.float64) -
                np.asarray(path[i - 1], dtype=np.float64)))
                for i in range(1, len(path)))

        min_cl = 0.30  # minimum clearance for portal

        # Left path
        result.left_path_valid = False
        left_cost = 0.0
        if esdf_at(pL) >= min_cl:
            try:
                pr_s2l = astar_planner_fn(esdf, esdf_origin, esdf_res,
                                          start.tolist(), pL.tolist(), min_cl)
                pr_l2g = astar_planner_fn(esdf, esdf_origin, esdf_res,
                                          pL.tolist(), goal.tolist(), min_cl)
                if pr_s2l.reached_goal and pr_l2g.reached_goal:
                    result.left_path_valid = True
                    left_cost = path_length(pr_s2l.path) + path_length(pr_l2g.path)
                    result.left_path_length_m = left_cost
                    result.left_path_cost = left_cost
                    # Compute min clearance along path
                    min_cl_left = float('inf')
                    for pt in pr_s2l.path + pr_l2g.path:
                        cl = esdf_at(np.array(pt))
                        if cl < min_cl_left:
                            min_cl_left = cl
                    result.left_min_clearance_m = min_cl_left
            except Exception:
                pass

        # Right path
        result.right_path_valid = False
        right_cost = 0.0
        if esdf_at(pR) >= min_cl:
            try:
                pr_s2r = astar_planner_fn(esdf, esdf_origin, esdf_res,
                                          start.tolist(), pR.tolist(), min_cl)
                pr_r2g = astar_planner_fn(esdf, esdf_origin, esdf_res,
                                          pR.tolist(), goal.tolist(), min_cl)
                if pr_s2r.reached_goal and pr_r2g.reached_goal:
                    result.right_path_valid = True
                    right_cost = path_length(pr_s2r.path) + path_length(pr_r2g.path)
                    result.right_path_length_m = right_cost
                    result.right_path_cost = right_cost
                    min_cl_right = float('inf')
                    for pt in pr_s2r.path + pr_r2g.path:
                        cl = esdf_at(np.array(pt))
                        if cl < min_cl_right:
                            min_cl_right = cl
                    result.right_min_clearance_m = min_cl_right
            except Exception:
                pass

        # Determine lower-cost side
        if result.left_path_valid and result.right_path_valid:
            if left_cost < right_cost:
                result.lower_cost_side = "LEFT"
                result.side_cost_difference_ratio = (right_cost - left_cost) / max(left_cost, 1e-6)
            else:
                result.lower_cost_side = "RIGHT"
                result.side_cost_difference_ratio = (left_cost - right_cost) / max(right_cost, 1e-6)

            if self.reject_equal and result.side_cost_difference_ratio < self.min_cost_diff_ratio:
                # Costs nearly equal — default to RIGHT instead of rejecting.
                result.lower_cost_side = "RIGHT"
            result.global_side_choice_valid = True
        elif result.left_path_valid:
            result.lower_cost_side = "LEFT"
            result.global_side_choice_valid = not self.require_both
            if self.require_both:
                result.rejection_reason = "TASK_SIDE_PATHS_INVALID"
        elif result.right_path_valid:
            result.lower_cost_side = "RIGHT"
            result.global_side_choice_valid = not self.require_both
            if self.require_both:
                result.rejection_reason = "TASK_SIDE_PATHS_INVALID"
        else:
            result.rejection_reason = "TASK_SIDE_PATHS_INVALID"
            result.global_side_choice_valid = False

        if result.global_side_choice_valid:
            result.valid = True
        return result


# ============================================================================
#  SceneManifestWriter
# ============================================================================

class SceneManifestWriter:
    """Write scene and task manifests as JSON for reproducibility.

    v9: extended with profile, density, seed, and gap metadata fields.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def write_scene_manifest(self, scene_id, base_seed, profile_name,
                              profile_index, scene_index_in_profile,
                              effective_scene_seed, generation_attempt,
                              obstacles, validation, task_results,
                              obstacle_region,
                              target_density_mode, target_density,
                              actual_raw_occupancy, actual_inflated_occupancy,
                              actual_obstacles_per_100m2,
                              vehicle_radius_m, safety_margin_m,
                              min_surface_gap_required_m,
                              min_post_inflation_gap_required_m,
                              min_surface_gap_actual_m,
                              min_post_inflation_gap_actual_m,
                              generation_status="accepted"):
        """Write scene_manifest.json with full metadata."""
        radius_list = [o.radius_m for o in obstacles] if obstacles else [0.0]
        manifest = OrderedDict([
            ("scene_id", scene_id),
            ("scene_profile_name", profile_name),
            ("scene_profile_index", profile_index),
            ("scene_index_in_profile", scene_index_in_profile),
            ("base_seed", base_seed),
            ("profile_seed_offset", 0),  # filled by caller
            ("effective_scene_seed", effective_scene_seed),
            ("obstacle_type", "cylinder"),
            ("obstacle_region", OrderedDict([
                ("x_min", obstacle_region.x_min),
                ("x_max", obstacle_region.x_max),
                ("y_min", obstacle_region.y_min),
                ("y_max", obstacle_region.y_max),
                ("z_min", obstacle_region.z_min),
                ("z_max", obstacle_region.z_max),
            ])),
            ("target_density_mode", target_density_mode),
            ("target_density", target_density),
            ("actual_raw_occupancy_ratio", actual_raw_occupancy),
            ("actual_inflated_occupancy_ratio", actual_inflated_occupancy),
            ("actual_obstacles_per_100m2", actual_obstacles_per_100m2),
            ("obstacle_count", len(obstacles)),
            ("radius_min_actual_m", min(radius_list) if radius_list else 0.0),
            ("radius_max_actual_m", max(radius_list) if radius_list else 0.0),
            ("radius_mean_actual_m", sum(radius_list) / max(len(radius_list), 1)),
            ("vehicle_radius_m", vehicle_radius_m),
            ("safety_margin_m", safety_margin_m),
            ("minimum_surface_gap_required_m", min_surface_gap_required_m),
            ("minimum_post_inflation_gap_required_m", min_post_inflation_gap_required_m),
            ("minimum_surface_gap_actual_m", min_surface_gap_actual_m),
            ("minimum_post_inflation_gap_actual_m", min_post_inflation_gap_actual_m),
            ("generation_attempt_index", generation_attempt),
            ("generation_status", generation_status),
            ("obstacles", [o.to_dict() for o in obstacles]),
            ("validation", OrderedDict([
                ("valid", validation.valid),
                ("rejection_reason", validation.rejection_reason),
                ("minimum_surface_gap_m", validation.minimum_surface_gap_m),
                ("minimum_inflated_gap_m", validation.minimum_inflated_gap_m),
                ("inflated_components", validation.inflated_components),
                ("enclosed_free_component_count", validation.enclosed_free_component_count),
                ("dead_end_detected", validation.dead_end_detected),
                ("u_shape_detected", validation.u_shape_detected),
                ("navigable_free_ratio", validation.navigable_free_ratio),
                ("minimum_navigable_clearance_m", validation.minimum_navigable_clearance_m),
            ])),
        ])

        scene_dir = os.path.join(self.output_dir, "scenes", profile_name, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        path = os.path.join(scene_dir, "scene_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
        # Also write a YAML copy for readability
        yaml_path = os.path.join(scene_dir, "scene_manifest.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True,
                       sort_keys=False)
        return path

    def write_task_manifest(self, scene_id, task_id, start, goal, validation,
                             profile_name=""):
        """Write task manifest as JSON."""
        manifest = OrderedDict([
            ("task_id", task_id),
            ("scene_id", scene_id),
            ("scene_profile_name", profile_name),
            ("start", [float(v) for v in start]),
            ("goal", [float(v) for v in goal]),
            ("direct_distance_m", validation.direct_distance_m),
            ("direct_path_blocked", validation.direct_path_blocked),
            ("direct_blocker_count", validation.direct_blocker_count),
            ("nearest_direct_blocker_distance_m",
             validation.nearest_direct_blocker_distance_m),
            ("astar_length_m", validation.astar_length_m),
            ("detour_ratio", validation.detour_ratio),
            ("dominant_obstacle_id", validation.dominant_obstacle_id),
            ("dominant_obstacle_lateral_offset_m",
             validation.dominant_obstacle_lateral_offset_m),
            ("left_path_valid", validation.left_path_valid),
            ("right_path_valid", validation.right_path_valid),
            ("left_path_cost", validation.left_path_cost),
            ("right_path_cost", validation.right_path_cost),
            ("left_path_length_m", validation.left_path_length_m),
            ("right_path_length_m", validation.right_path_length_m),
            ("left_min_clearance_m", validation.left_min_clearance_m),
            ("right_min_clearance_m", validation.right_min_clearance_m),
            ("lower_cost_side", validation.lower_cost_side),
            ("side_cost_difference_ratio", validation.side_cost_difference_ratio),
            ("global_side_choice_valid", validation.global_side_choice_valid),
            ("coverage_target_task_type",
             validation.coverage_target_task_type),
            ("coverage_actual_task_type",
             validation.coverage_actual_task_type),
            ("coverage_target_blocker_distance_band",
             validation.coverage_target_blocker_distance_band),
            ("coverage_actual_blocker_distance_band",
             validation.coverage_actual_blocker_distance_band),
            ("coverage_target_height_band",
             validation.coverage_target_height_band),
            ("coverage_actual_height_band",
             validation.coverage_actual_height_band),
            ("coverage_region_pair_index",
             validation.coverage_region_pair_index),
        ])
        scene_dir = os.path.join(self.output_dir, "scenes", profile_name, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        path = os.path.join(scene_dir, "task_{}.json".format(task_id))
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
        return path


class SceneGenerationFailureManifestWriter:
    """Write failure manifests when scene generation is exhausted."""

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def write_failure_manifest(self, profile_name, profile_index,
                                scene_index_in_profile, effective_scene_seed,
                                failure_reason, num_attempts):
        """Write a generation failure manifest."""
        failed_dir = os.path.join(self.output_dir, "_failed", "scene_generation")
        os.makedirs(failed_dir, exist_ok=True)
        manifest = OrderedDict([
            ("profile_name", profile_name),
            ("profile_index", profile_index),
            ("scene_index_in_profile", scene_index_in_profile),
            ("effective_scene_seed", effective_scene_seed),
            ("failure_reason", failure_reason),
            ("generation_attempts_exhausted", num_attempts),
            ("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ])
        fname = "failure_{}_scene_{:06d}.json".format(
            profile_name, scene_index_in_profile)
        path = os.path.join(failed_dir, fname)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=False)
        return path


# ============================================================================
#  ObstacleVisibilityAuditor — runtime observability audit
# ============================================================================

class ObstacleVisibilityAuditor:
    """Runtime audit: check that observed side-cost choice matches global choice."""

    def __init__(self, config):
        obs_cfg = config.get("global", {}).get("scene_generation", {}).get(
            "observability_audit", {})
        self._cfg = obs_cfg

        self.require_consistency = bool(
            obs_cfg.get("require_runtime_side_consistency", True))
        self.min_obs_diff_ratio = float(
            obs_cfg.get("minimum_observed_cost_difference_ratio", 0.05))
        self.max_invalid_frames = int(
            obs_cfg.get("maximum_invalid_frames_before_reject", 5))
        self.reject_inconsistent = bool(
            obs_cfg.get("reject_episode_on_inconsistent_side_choice", True))
        topo = config.get("global", {}).get("scene_generation", {}).get(
            "topology_validation", {})
        side_cfg = config.get("global", {}).get("scene_generation", {}).get(
            "side_cost", {})
        self.inflation = (float(topo.get("vehicle_radius_m", 0.30)) +
                          float(topo.get("safety_margin_m", 0.10)))
        self.portal_clearance = float(
            side_cfg.get("portal_lateral_clearance_m", 0.60))
        self.corridor_spacing = float(topo.get("grid_resolution_m", 0.10))

    def audit(self, global_lower_cost_side, observed_map, observed_esdf,
              dominant_obstacle, current_position_world, current_yaw,
              start_world, goal_world, invalid_frame_count=0):
        """Run observability audit for current frame.

        Returns ObservabilityAuditResult.
        """
        result = ObservabilityAuditResult()

        if not self.require_consistency:
            result.observable_expert_label = True
            result.side_choice_consistent = True
            return result

        if observed_map is None or observed_esdf is None:
            result.side_choice_consistent = False
            return result

        if dominant_obstacle is not None:
            obs_xy = dominant_obstacle.center_xy()
            pos = np.asarray(current_position_world)
            radial = pos[:2] - obs_xy
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > 1e-6:
                surface_xy = obs_xy + radial / radial_norm * dominant_obstacle.radius_m
            else:
                surface_xy = obs_xy
            z_min = dominant_obstacle.center_world[2] - 0.5 * dominant_obstacle.height_m
            z_max = dominant_obstacle.center_world[2] + 0.5 * dominant_obstacle.height_m
            surface_z = float(np.clip(pos[2], z_min, z_max))
            visible_surface = np.array(
                [surface_xy[0], surface_xy[1], surface_z])
            surface_observed = any(
                observed_map.is_known(visible_surface + np.array([dx, dy, dz]))
                for dx in (-self.corridor_spacing, 0.0, self.corridor_spacing)
                for dy in (-self.corridor_spacing, 0.0, self.corridor_spacing)
                for dz in (-self.corridor_spacing, 0.0, self.corridor_spacing))

            if surface_observed:
                result.observability_check_triggered = True
                direct = np.asarray(goal_world, dtype=np.float64)[:2] - \
                    np.asarray(start_world, dtype=np.float64)[:2]
                direct_norm = float(np.linalg.norm(direct))
                if direct_norm < 1e-6:
                    return result
                e_dir = direct / direct_norm
                n_dir = np.array([-e_dir[1], e_dir[0]])
                portal_offset = (dominant_obstacle.radius_m + self.inflation +
                                 self.portal_clearance)
                altitude = float(current_position_world[2])
                left = np.array([obs_xy[0] + n_dir[0] * portal_offset,
                                 obs_xy[1] + n_dir[1] * portal_offset, altitude])
                right = np.array([obs_xy[0] - n_dir[0] * portal_offset,
                                  obs_xy[1] - n_dir[1] * portal_offset, altitude])
                current = np.asarray(current_position_world, dtype=np.float64)
                radius = self.inflation
                left_ratio = observed_map.sample_known_free_ratio_along_corridor(
                    current, left, radius, self.corridor_spacing)
                right_ratio = observed_map.sample_known_free_ratio_along_corridor(
                    current, right, radius, self.corridor_spacing)
                left_clearance = observed_esdf.value_at(left)
                right_clearance = observed_esdf.value_at(right)
                min_ratio = observed_map.min_known_free_ratio
                left_valid = left_ratio >= min_ratio and left_clearance is not None
                right_valid = right_ratio >= min_ratio and right_clearance is not None
                if left_valid and right_valid:
                    # Cost is based solely on observed path length and observed
                    # clearance; unknown voxels never receive a global fill-in.
                    left_cost = (float(np.linalg.norm(left - current)) +
                                 1.0 / max(left_clearance, 1e-3))
                    right_cost = (float(np.linalg.norm(right - current)) +
                                  1.0 / max(right_clearance, 1e-3))
                    result.left_observed_path_cost = left_cost
                    result.right_observed_path_cost = right_cost
                    result.observed_lower_cost_side = (
                        "LEFT" if left_cost < right_cost else "RIGHT")
                    result.observed_side_cost_difference_ratio = (
                        abs(left_cost - right_cost) /
                        max(min(left_cost, right_cost), 1e-6))
                    result.observed_side_cost_valid = (
                        result.observed_side_cost_difference_ratio >=
                        self.min_obs_diff_ratio)
                    result.side_choice_consistent = (
                        result.observed_side_cost_valid and
                        result.observed_lower_cost_side == global_lower_cost_side)
                    result.observable_expert_label = result.side_choice_consistent
                if not result.observable_expert_label:
                    result.invalid_observability_frame_count = invalid_frame_count + 1
            else:
                # Before the dominant obstacle is observed there is no side
                # decision to audit; do not spend the consecutive-invalid
                # budget on pre-trigger frames.
                result.side_choice_consistent = False
                result.invalid_observability_frame_count = 0
        else:
            result.observable_expert_label = True
            result.side_choice_consistent = True

        return result
