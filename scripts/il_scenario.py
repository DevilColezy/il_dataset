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


# ============================================================================
#  RadiusRange — multi-range cylinder sampling support
# ============================================================================

@dataclass
class RadiusRange:
    """A single radius sub-range with sampling weight.

    Used by mixed-scale profiles (e.g. S10–S12) that need cylinders from
    multiple distinct size classes within the same scene.

    Attributes:
        min_m: minimum radius in this sub-range (metres).
        max_m: maximum radius in this sub-range (metres).
        weight: relative probability of sampling from this range.
            Weights are normalised across all ranges in a profile.
    """
    min_m: float
    max_m: float
    weight: float = 1.0

    def __post_init__(self):
        if self.min_m <= 0.0:
            raise ValueError("RadiusRange.min_m must be > 0, got {}".format(self.min_m))
        if self.max_m < self.min_m:
            raise ValueError("RadiusRange.max_m ({}) < min_m ({})".format(
                self.max_m, self.min_m))
        if self.weight <= 0.0:
            raise ValueError("RadiusRange.weight must be > 0, got {}".format(self.weight))

    @staticmethod
    def from_dict(d):
        """Parse from a YAML dict with keys min, max, weight."""
        return RadiusRange(
            min_m=float(d.get("min", 0.0)),
            max_m=float(d.get("max", 0.0)),
            weight=float(d.get("weight", 1.0)),
        )


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
    """A single scene generation profile parsed from YAML."""
    name: str
    enabled: bool
    scene_count: int
    seed_offset: int

    # Cylinder params
    radius_min_m: float
    radius_max_m: float
    count_min: int
    count_max: int

    # Density params
    density_enabled: bool
    density_mode: str  # "inflated_occupancy" etc.
    density_target_min: float
    density_target_max: float

    # Resolved common params (merged from common_cylinder / common_task_generation)
    height_min_m: float
    height_max_m: float
    require_full_vertical_blocking: bool
    region_boundary_margin_m: float
    minimum_surface_gap_m: float
    minimum_post_inflation_gap_m: float
    allow_overlap: bool

    # Vehicle
    vehicle_radius_m: float
    safety_margin_m: float

    # Task params
    tasks_per_scene: int
    start_height_min_m: float
    start_height_max_m: float
    goal_height_min_m: float
    goal_height_max_m: float
    maximum_start_goal_height_difference_m: float
    minimum_start_goal_distance_m: float
    maximum_start_goal_distance_m: float
    require_direct_path_blocked: bool
    minimum_direct_blocker_count: int
    maximum_direct_blocker_count: int
    require_astar_reachable: bool
    minimum_detour_ratio: float
    maximum_detour_ratio: float

    # Overrides (any profile-level overrides dict, kept for metadata)
    overrides: dict = field(default_factory=dict)

    # Multi-range sampling (mixed-scale scenes)
    radius_ranges: list = field(default_factory=list)

    # ── Density-driven mode fields (v10) ──────────────────────────
    size_groups: list = field(default_factory=list)     # list of SizeGroup
    total_density_min: float = 0.05
    total_density_max: float = 0.20
    forbid_enclosed_free_components: bool = True
    forbid_inflated_component_merging: bool = True

    # Task params — legacy single-region (backward compat, must be after non-defaults)
    start_sampling_region: dict = field(default_factory=dict)
    goal_sampling_region: dict = field(default_factory=dict)
    # Task params — multi-region lists (v6, preferred)
    start_sampling_regions: list = field(default_factory=list)
    goal_sampling_regions: list = field(default_factory=list)

    @property
    def inflation_radius_m(self):
        return self.vehicle_radius_m + self.safety_margin_m

    @staticmethod
    def from_yaml_dict(profile_dict, common_cfg):
        """Parse a single profile dict with common config fallback.

        Profile fields override common_cfg fields.
        """
        name = str(profile_dict.get("name", "unnamed"))
        enabled = bool(profile_dict.get("enabled", True))
        scene_count = int(profile_dict.get("scene_count", 10))
        seed_offset = int(profile_dict.get("seed_offset", 0))

        # Cylinder params (profile overrides common_cylinder)
        cyl_cfg = profile_dict.get("cylinder", {})
        common_cyl = common_cfg.get("common_cylinder", {})

        radius_min_m = float(cyl_cfg.get("radius_min_m",
            common_cyl.get("radius_min_m", 0.30)))
        radius_max_m = float(cyl_cfg.get("radius_max_m",
            common_cyl.get("radius_max_m", 1.20)))
        count_min = int(cyl_cfg.get("count_min",
            common_cyl.get("count_min", 4)))
        count_max = int(cyl_cfg.get("count_max",
            common_cyl.get("count_max", 12)))

        # Multi-range sampling (mixed-scale scenes, e.g. S10–S12)
        radius_ranges = []
        ranges_raw = cyl_cfg.get("ranges", None)
        if ranges_raw is not None:
            for r in ranges_raw:
                radius_ranges.append(RadiusRange.from_dict(r))

        # Density params
        density_cfg = profile_dict.get("density", {})
        density_enabled = bool(density_cfg.get("enabled", True))
        density_mode = str(density_cfg.get("mode", "inflated_occupancy"))
        density_target_min = float(density_cfg.get("target_min", 0.05))
        density_target_max = float(density_cfg.get("target_max", 0.15))

        # Resolved common cylinder params
        height_min_m = float(common_cyl.get("height_min_m", 8.0))
        height_max_m = float(common_cyl.get("height_max_m", 8.0))
        require_full_vertical_blocking = bool(common_cyl.get(
            "require_full_vertical_blocking", True))
        region_boundary_margin_m = float(common_cyl.get(
            "region_boundary_margin_m", 0.30))
        minimum_surface_gap_m = float(common_cyl.get(
            "minimum_surface_gap_m", 0.0))
        minimum_post_inflation_gap_m = float(common_cyl.get(
            "minimum_post_inflation_gap_m", 0.15))
        allow_overlap = bool(common_cyl.get("allow_overlap", False))

        # Vehicle params
        vehicle_cfg = common_cfg.get("vehicle", {})
        vehicle_radius_m = float(vehicle_cfg.get("radius_m",
            common_cfg.get("topology_validation", {}).get("vehicle_radius_m", 0.30)))
        safety_margin_m = float(vehicle_cfg.get("safety_margin_m",
            common_cfg.get("topology_validation", {}).get("safety_margin_m", 0.10)))

        # Task params (profile overrides common_task_generation)
        task_cfg = profile_dict.get("task_generation", {})
        common_task = common_cfg.get("common_task_generation", {})

        tasks_per_scene = int(task_cfg.get("tasks_per_scene",
            common_task.get("tasks_per_scene", 3)))

        # Start/Goal sampling regions — support both single (legacy) and multi (v6)
        start_sampling_region = task_cfg.get("start_sampling_region", None)
        goal_sampling_region = task_cfg.get("goal_sampling_region", None)
        start_sampling_regions = task_cfg.get("start_sampling_regions", None)
        goal_sampling_regions = task_cfg.get("goal_sampling_regions", None)

        # Backward compat: if singular is provided, wrap as single-element list
        if start_sampling_regions is None and start_sampling_region is not None:
            start_sampling_regions = [start_sampling_region]
        if goal_sampling_regions is None and goal_sampling_region is not None:
            goal_sampling_regions = [goal_sampling_region]

        # Fallback from common if still None
        if start_sampling_regions is None:
            sr = common_task.get("start_sampling_region", {})
            start_sampling_regions = common_task.get("start_sampling_regions", [sr] if sr else [{"x_min": -8.0, "x_max": 10.0, "y_min": -2.0, "y_max": -1.0}])
        if goal_sampling_regions is None:
            gr = common_task.get("goal_sampling_region", {})
            goal_sampling_regions = common_task.get("goal_sampling_regions", [gr] if gr else [{"x_min": -8.0, "x_max": 10.0, "y_min": 30.0, "y_max": 32.0}])

        # Also keep the first entry as the legacy singular fields for backward compat
        start_sampling_region = start_sampling_regions[0] if start_sampling_regions else {}
        goal_sampling_region = goal_sampling_regions[0] if goal_sampling_regions else {}

        start_height_min_m = float(task_cfg.get("start_height_min_m",
            common_task.get("start_height_min_m", 1.8)))
        start_height_max_m = float(task_cfg.get("start_height_max_m",
            common_task.get("start_height_max_m", 2.2)))
        goal_height_min_m = float(task_cfg.get("goal_height_min_m",
            common_task.get("goal_height_min_m", 1.8)))
        goal_height_max_m = float(task_cfg.get("goal_height_max_m",
            common_task.get("goal_height_max_m", 2.2)))
        maximum_start_goal_height_difference_m = float(
            task_cfg.get("maximum_start_goal_height_difference_m",
                common_task.get("maximum_start_goal_height_difference_m", 0.20)))
        minimum_start_goal_distance_m = float(
            task_cfg.get("minimum_start_goal_distance_m",
                common_task.get("minimum_start_goal_distance_m", 28.0)))
        maximum_start_goal_distance_m = float(
            task_cfg.get("maximum_start_goal_distance_m",
                common_task.get("maximum_start_goal_distance_m", 40.0)))
        require_direct_path_blocked = bool(
            task_cfg.get("require_direct_path_blocked",
                common_task.get("require_direct_path_blocked", True)))
        minimum_direct_blocker_count = int(
            task_cfg.get("minimum_direct_blocker_count",
                common_task.get("minimum_direct_blocker_count", 1)))
        maximum_direct_blocker_count = int(
            task_cfg.get("maximum_direct_blocker_count",
                common_task.get("maximum_direct_blocker_count", 3)))
        require_astar_reachable = bool(
            task_cfg.get("require_astar_reachable",
                common_task.get("require_astar_reachable", True)))
        minimum_detour_ratio = float(
            task_cfg.get("minimum_detour_ratio",
                common_task.get("minimum_detour_ratio", 1.03)))
        maximum_detour_ratio = float(
            task_cfg.get("maximum_detour_ratio",
                common_task.get("maximum_detour_ratio", 2.00)))

        # Explicit overrides
        overrides = profile_dict.get("overrides", {})

        return SceneGenerationProfile(
            name=name,
            enabled=enabled,
            scene_count=scene_count,
            seed_offset=seed_offset,
            radius_min_m=radius_min_m,
            radius_max_m=radius_max_m,
            count_min=count_min,
            count_max=count_max,
            density_enabled=density_enabled,
            density_mode=density_mode,
            density_target_min=density_target_min,
            density_target_max=density_target_max,
            height_min_m=height_min_m,
            height_max_m=height_max_m,
            require_full_vertical_blocking=require_full_vertical_blocking,
            region_boundary_margin_m=region_boundary_margin_m,
            minimum_surface_gap_m=minimum_surface_gap_m,
            minimum_post_inflation_gap_m=minimum_post_inflation_gap_m,
            allow_overlap=allow_overlap,
            vehicle_radius_m=vehicle_radius_m,
            safety_margin_m=safety_margin_m,
            tasks_per_scene=tasks_per_scene,
            start_sampling_region=start_sampling_region,
            goal_sampling_region=goal_sampling_region,
            start_sampling_regions=start_sampling_regions,
            goal_sampling_regions=goal_sampling_regions,
            start_height_min_m=start_height_min_m,
            start_height_max_m=start_height_max_m,
            goal_height_min_m=goal_height_min_m,
            goal_height_max_m=goal_height_max_m,
            maximum_start_goal_height_difference_m=maximum_start_goal_height_difference_m,
            minimum_start_goal_distance_m=minimum_start_goal_distance_m,
            maximum_start_goal_distance_m=maximum_start_goal_distance_m,
            require_direct_path_blocked=require_direct_path_blocked,
            minimum_direct_blocker_count=minimum_direct_blocker_count,
            maximum_direct_blocker_count=maximum_direct_blocker_count,
            require_astar_reachable=require_astar_reachable,
            minimum_detour_ratio=minimum_detour_ratio,
            maximum_detour_ratio=maximum_detour_ratio,
            radius_ranges=radius_ranges,
            overrides=overrides,
        )

    @staticmethod
    def from_yaml_dict_v2(profile_dict, sg_cfg):
        """Parse a density-driven profile (v10 schema).

        Uses size_groups + total_density instead of legacy cylinder params.
        """
        name = str(profile_dict.get("name", "unnamed"))
        enabled = bool(profile_dict.get("enabled", True))
        scene_count = int(profile_dict.get("scene_count", 1))
        seed_offset = int(profile_dict.get("seed_offset", 0))

        # Size groups
        groups_raw = profile_dict.get("size_groups", {})
        size_groups = []
        for group_name in ("large", "medium", "small"):
            gd = groups_raw.get(group_name, {})
            sg = SizeGroup.from_dict(gd, default_name=group_name)
            sg.name = group_name
            size_groups.append(sg)

        # Density range
        total_density_min = float(profile_dict.get("density_min", 0.05))
        total_density_max = float(profile_dict.get("density_max", 0.15))

        # Core constraint params (from common level)
        common_cyl = sg_cfg.get("common_cylinder", {})
        vehicle_cfg = sg_cfg.get("vehicle", {})
        topo_cfg = sg_cfg.get("topology_validation", {})

        height_min_m = float(common_cyl.get("height_min_m", 8.0))
        height_max_m = float(common_cyl.get("height_max_m", 8.0))
        require_full_vertical_blocking = bool(common_cyl.get(
            "require_full_vertical_blocking", True))
        region_boundary_margin_m = float(sg_cfg.get("region_boundary_margin_m",
            common_cyl.get("region_boundary_margin_m", 0.30)))
        minimum_surface_gap_m = float(sg_cfg.get("minimum_surface_gap_m",
            common_cyl.get("minimum_surface_gap_m", 0.0)))
        minimum_post_inflation_gap_m = float(sg_cfg.get("post_inflation_gap_m",
            common_cyl.get("minimum_post_inflation_gap_m", 0.15)))
        allow_overlap = bool(sg_cfg.get("allow_overlap",
            common_cyl.get("allow_overlap", False)))

        vehicle_radius_m = float(vehicle_cfg.get("radius_m",
            topo_cfg.get("vehicle_radius_m", 0.30)))
        safety_margin_m = float(vehicle_cfg.get("safety_margin_m",
            topo_cfg.get("safety_margin_m", 0.10)))

        forbid_enclosed = bool(sg_cfg.get("forbid_enclosed_free_components",
            topo_cfg.get("forbid_enclosed_free_components", True)))
        forbid_merge = bool(sg_cfg.get("forbid_inflated_component_merging",
            not common_cyl.get("allow_overlap", False)))

        # Task params (from common_task_generation, profile may override)
        task_cfg = profile_dict.get("task_generation", {})
        common_task = sg_cfg.get("common_task_generation", {})

        tasks_per_scene = int(task_cfg.get("tasks_per_scene",
            common_task.get("tasks_per_scene", 12)))
        start_sampling_regions = task_cfg.get("start_sampling_regions",
            common_task.get("start_sampling_regions", None))
        goal_sampling_regions = task_cfg.get("goal_sampling_regions",
            common_task.get("goal_sampling_regions", None))
        if start_sampling_regions is None:
            start_sampling_regions = common_task.get("start_sampling_regions",
                [{"x_min": -8.0, "x_max": 10.0, "y_min": -2.0, "y_max": -1.0}])
        if goal_sampling_regions is None:
            goal_sampling_regions = common_task.get("goal_sampling_regions",
                [{"x_min": -8.0, "x_max": 10.0, "y_min": 30.0, "y_max": 32.0}])

        start_height_min_m = float(task_cfg.get("start_height_min_m",
            common_task.get("start_height_min_m", 1.8)))
        start_height_max_m = float(task_cfg.get("start_height_max_m",
            common_task.get("start_height_max_m", 2.2)))
        goal_height_min_m = float(task_cfg.get("goal_height_min_m",
            common_task.get("goal_height_min_m", 1.8)))
        goal_height_max_m = float(task_cfg.get("goal_height_max_m",
            common_task.get("goal_height_max_m", 2.2)))
        maximum_start_goal_height_difference_m = float(
            task_cfg.get("maximum_start_goal_height_difference_m",
                common_task.get("maximum_start_goal_height_difference_m", 1.50)))
        minimum_start_goal_distance_m = float(
            task_cfg.get("minimum_start_goal_distance_m",
                common_task.get("minimum_start_goal_distance_m", 10.0)))
        maximum_start_goal_distance_m = float(
            task_cfg.get("maximum_start_goal_distance_m",
                common_task.get("maximum_start_goal_distance_m", 40.0)))
        require_direct_path_blocked = bool(
            task_cfg.get("require_direct_path_blocked",
                common_task.get("require_direct_path_blocked", True)))
        minimum_direct_blocker_count = int(
            task_cfg.get("minimum_direct_blocker_count",
                common_task.get("minimum_direct_blocker_count", 1)))
        maximum_direct_blocker_count = int(
            task_cfg.get("maximum_direct_blocker_count",
                common_task.get("maximum_direct_blocker_count", 3)))
        require_astar_reachable = bool(
            task_cfg.get("require_astar_reachable",
                common_task.get("require_astar_reachable", True)))
        minimum_detour_ratio = float(
            task_cfg.get("minimum_detour_ratio",
                common_task.get("minimum_detour_ratio", 1.03)))
        maximum_detour_ratio = float(
            task_cfg.get("maximum_detour_ratio",
                common_task.get("maximum_detour_ratio", 2.00)))

        return SceneGenerationProfile(
            name=name,
            enabled=enabled,
            scene_count=scene_count,
            seed_offset=seed_offset,
            radius_min_m=size_groups[2].radius_min_m if size_groups else 0.10,
            radius_max_m=size_groups[0].radius_max_m if size_groups else 3.50,
            count_min=0,  # n/a for density-driven
            count_max=0,  # n/a for density-driven
            density_enabled=True,
            density_mode="inflated_occupancy",
            density_target_min=total_density_min,
            density_target_max=total_density_max,
            height_min_m=height_min_m,
            height_max_m=height_max_m,
            require_full_vertical_blocking=require_full_vertical_blocking,
            region_boundary_margin_m=region_boundary_margin_m,
            minimum_surface_gap_m=minimum_surface_gap_m,
            minimum_post_inflation_gap_m=minimum_post_inflation_gap_m,
            allow_overlap=allow_overlap,
            vehicle_radius_m=vehicle_radius_m,
            safety_margin_m=safety_margin_m,
            tasks_per_scene=tasks_per_scene,
            start_sampling_region=start_sampling_regions[0] if start_sampling_regions else {},
            goal_sampling_region=goal_sampling_regions[0] if goal_sampling_regions else {},
            start_sampling_regions=start_sampling_regions,
            goal_sampling_regions=goal_sampling_regions,
            start_height_min_m=start_height_min_m,
            start_height_max_m=start_height_max_m,
            goal_height_min_m=goal_height_min_m,
            goal_height_max_m=goal_height_max_m,
            maximum_start_goal_height_difference_m=maximum_start_goal_height_difference_m,
            minimum_start_goal_distance_m=minimum_start_goal_distance_m,
            maximum_start_goal_distance_m=maximum_start_goal_distance_m,
            require_direct_path_blocked=require_direct_path_blocked,
            minimum_direct_blocker_count=minimum_direct_blocker_count,
            maximum_direct_blocker_count=maximum_direct_blocker_count,
            require_astar_reachable=require_astar_reachable,
            minimum_detour_ratio=minimum_detour_ratio,
            maximum_detour_ratio=maximum_detour_ratio,
            radius_ranges=[],
            size_groups=size_groups,
            total_density_min=total_density_min,
            total_density_max=total_density_max,
            forbid_enclosed_free_components=forbid_enclosed,
            forbid_inflated_component_merging=forbid_merge,
            overrides=profile_dict.get("overrides", {}),
        )


def load_scene_profiles(config):
    """Load enabled scene generation profiles from config.

    Supports sources: procedural_yaml, procedural_profiles, density_driven.

    Returns:
        list of SceneGenerationProfile (only enabled profiles, in YAML order).
    """
    sg_cfg = config.get("global", {}).get("scene_generation", {})
    source = sg_cfg.get("source", "procedural_yaml")

    if source not in ("procedural_profiles", "density_driven"):
        # Legacy mode: no profiles
        return []

    profiles_raw = sg_cfg.get("profiles", [])
    if not profiles_raw:
        rospy.logwarn("[Profiles] source='%s' but profiles list is empty.", source)
        return []

    profiles = []
    seen_names = set()
    for p in profiles_raw:
        if source == "density_driven":
            profile = SceneGenerationProfile.from_yaml_dict_v2(p, sg_cfg)
        else:
            profile = SceneGenerationProfile.from_yaml_dict(p, sg_cfg)
        if not profile.enabled:
            rospy.loginfo("[Profiles] Skipping disabled profile '%s'.", profile.name)
            continue
        if profile.name in seen_names:
            raise ValueError("Duplicate profile name: '{}'".format(profile.name))
        seen_names.add(profile.name)
        profiles.append(profile)

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
    """Generate random cylinder obstacle scenes from YAML configuration.

    Supports two modes:
      - Legacy (source="procedural_yaml"): single flat cylinder config.
      - Profile (source="procedural_profiles"): density-driven per-profile generation.

    Only cylinder obstacles are generated.
    The obstacle generation region is not treated as a wall or flight boundary.
    Space outside the obstacle generation region remains traversable.
    """

    def __init__(self, config):
        cfg = config.get("global", {}).get("scene_generation", {})
        self._cfg = cfg
        self._source = cfg.get("source", "procedural_yaml")
        self._legacy_cyl_cfg = cfg.get("cylinder", {})
        self._topo_cfg = cfg.get("topology_validation", {})

        self.obstacle_region = ObstacleRegion(
            x_min=float(cfg.get("obstacle_region", {}).get("x_min", -8.0)),
            x_max=float(cfg.get("obstacle_region", {}).get("x_max", 8.0)),
            y_min=float(cfg.get("obstacle_region", {}).get("y_min", -6.0)),
            y_max=float(cfg.get("obstacle_region", {}).get("y_max", 6.0)),
            z_min=float(cfg.get("obstacle_region", {}).get("z_min", 0.0)),
            z_max=float(cfg.get("obstacle_region", {}).get("z_max", 6.0)),
        )

        exec_cfg = cfg.get("execution", {})
        self.max_scene_attempts = int(exec_cfg.get(
            "max_generation_attempts_per_scene",
            cfg.get("max_scene_generation_attempts", 500)))
        self.max_obs_attempts = int(exec_cfg.get(
            "max_obstacle_sampling_attempts",
            cfg.get("max_obstacle_sampling_attempts", 5000)))

        # Legacy params (used when source != "procedural_profiles")
        self.region_margin = float(self._legacy_cyl_cfg.get("region_boundary_margin_m", 0.30))
        self.min_surface_gap = float(self._legacy_cyl_cfg.get("minimum_surface_gap_m", 1.20))
        self.min_inflated_gap = float(self._legacy_cyl_cfg.get("minimum_inflated_gap_m", 0.60))
        self.allow_overlap = bool(self._legacy_cyl_cfg.get("allow_overlap", False))
        self.allow_inflated_merge = bool(self._legacy_cyl_cfg.get("allow_inflated_component_merging", False))

        self.vehicle_r = float(self._topo_cfg.get("vehicle_radius_m", 0.30))
        self.safety_m = float(self._topo_cfg.get("safety_margin_m", 0.10))
        self.inflated_extra = self.vehicle_r + self.safety_m

        self.base_seed = int(cfg.get("seed", 12345))
        self._rng = None

        # Profile mode: loaded profiles
        self._profiles = []
        if self._source == "procedural_profiles":
            self._profiles = load_scene_profiles(config)

    @property
    def is_profile_mode(self):
        return self._source == "procedural_profiles" and len(self._profiles) > 0

    def get_profiles(self):
        return list(self._profiles)

    def set_seed(self, seed):
        self.base_seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(self.base_seed))

    def _make_np_rng(self, seed):
        """Create a numpy Generator for reproducible randomness."""
        return np.random.Generator(np.random.PCG64(int(seed)))

    def _make_rng(self, sub_seed):
        """Legacy random.Random (for backward compat)."""
        return random.Random(self.base_seed + int(sub_seed))

    # ── Legacy generation ────────────────────────────────────────────

    def generate_scene(self, attempt_sub_seed=0):
        """Legacy generate one obstacle layout. Returns (obstacles, rejection).

        Used when source != "procedural_profiles".
        On failure, returns ([], reason_string).
        """
        fixed_specs = self._cfg.get("fixed_obstacles", None)
        if fixed_specs is not None:
            obstacles = []
            seen_ids = set()
            for index, item in enumerate(fixed_specs):
                try:
                    obstacle_id = str(
                        item.get("id", "fixed_{:03d}".format(index)))
                    center = np.asarray(item["center"], dtype=np.float64)
                    radius = float(item["radius_m"])
                    height = float(item.get("height_m", 8.0))
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

        rng = self._make_rng(attempt_sub_seed)

        count_min = int(self._legacy_cyl_cfg.get("count_min", 4))
        count_max = int(self._legacy_cyl_cfg.get("count_max", 12))
        n_obs = rng.randint(count_min, count_max)

        radius_min = float(self._legacy_cyl_cfg.get("radius_min_m", 0.40))
        radius_max = float(self._legacy_cyl_cfg.get("radius_max_m", 1.20))
        height_min = float(self._legacy_cyl_cfg.get("height_min_m", 5.0))
        height_max = float(self._legacy_cyl_cfg.get("height_max_m", 6.0))

        obstacles = []

        for i in range(n_obs):
            placed = False
            for _attempt in range(self.max_obs_attempts):
                radius = rng.uniform(radius_min, radius_max)
                height = rng.uniform(height_min, height_max)
                z = rng.uniform(self.obstacle_region.z_min + height / 2.0,
                                self.obstacle_region.z_max - height / 2.0)
                if height / 2.0 >= z - self.obstacle_region.z_min:
                    z = self.obstacle_region.z_min + height / 2.0 + 0.01

                cx = rng.uniform(
                    self.obstacle_region.x_min + radius + self.region_margin,
                    self.obstacle_region.x_max - radius - self.region_margin)
                cy = rng.uniform(
                    self.obstacle_region.y_min + radius + self.region_margin,
                    self.obstacle_region.y_max - radius - self.region_margin)

                valid = True
                center_xy = np.array([cx, cy])
                for prev in obstacles:
                    prev_xy = prev.center_xy()
                    d = float(np.linalg.norm(center_xy - prev_xy))
                    surface_gap = d - radius - prev.radius_m
                    if surface_gap < self.min_surface_gap:
                        valid = False
                        break
                    infl_r_i = radius + self.inflated_extra
                    infl_r_j = prev.radius_m + self.inflated_extra
                    inflated_gap = d - infl_r_i - infl_r_j
                    if inflated_gap < self.min_inflated_gap and not self.allow_inflated_merge:
                        valid = False
                        break

                if valid:
                    obs_id = "cylinder_{:04d}".format(i)
                    obs = CylinderObstacleSpec(
                        obstacle_id=obs_id,
                        center_world=np.array([cx, cy, z]),
                        radius_m=radius,
                        height_m=height,
                    )
                    obstacles.append(obs)
                    placed = True
                    break

            if not placed:
                return [], "SCENE_OBSTACLE_SAMPLING_EXHAUSTED"

        if len(obstacles) < count_min:
            return [], "SCENE_OBSTACLE_SAMPLING_EXHAUSTED"

        return obstacles, ""

    # ── Profile-based generation ─────────────────────────────────────

    def generate_scene_from_profile(self, profile, effective_scene_seed,
                                     scene_index_in_profile, attempt_index=0):
        """Generate one scene using a specific profile.

        Density-driven algorithm:
          1. Sample target_density from [density_target_min, density_target_max]
          2. Iteratively sample candidate cylinders
          3. Check region boundary and pairwise passable gap
          4. Accept if conditions satisfied
          5. Stop when count >= count_min AND actual density >= target_density

        Args:
            profile: SceneGenerationProfile.
            effective_scene_seed: seed for this scene (= base_seed + seed_offset + scene_index).
            scene_index_in_profile: 0-based index within the profile.
            attempt_index: generation retry index (0 for first attempt).

        Returns:
            (obstacles, rejection_reason, actual_target_density, density_mode)
        """
        rng = self._make_np_rng(effective_scene_seed + attempt_index * 10007)

        region_area = compute_region_area(self.obstacle_region)
        density_mode = DensityMode.from_string(profile.density_mode)

        # Sample target density
        if profile.density_enabled:
            target_density = float(rng.uniform(
                profile.density_target_min, profile.density_target_max))
        else:
            # Fixed count mode: no density target
            target_density = None

        obstacles = []
        count_min = profile.count_min
        count_max = profile.count_max
        height_min = profile.height_min_m
        height_max = profile.height_max_m
        boundary_margin = profile.region_boundary_margin_m
        min_surface_gap = profile.minimum_surface_gap_m
        min_post_inflation_gap = profile.minimum_post_inflation_gap_m
        infl_radius = profile.inflation_radius_m

        # ── Radius sampling strategy ──────────────────────────────
        # When radius_ranges is provided (mixed-scale scenes), sample a
        # sub-range by weight first, then draw uniformly within it.
        # Otherwise fall back to the legacy single [min, max] range.
        use_multi_range = (
            profile.radius_ranges is not None
            and len(profile.radius_ranges) > 0)
        if use_multi_range:
            _range_weights = np.array(
                [r.weight for r in profile.radius_ranges],
                dtype=np.float64)
            _range_weight_sum = float(np.sum(_range_weights))
            if _range_weight_sum <= 0.0:
                use_multi_range = False  # safety fallback
            else:
                _range_weights /= _range_weight_sum
                _range_cumulative = np.cumsum(_range_weights)

        def _sample_radius(rng_state):
            """Sample a cylinder radius using the active strategy."""
            if use_multi_range:
                # Weighted range selection
                p = float(rng_state.uniform(0.0, 1.0))
                range_idx = int(np.searchsorted(_range_cumulative, p, side="right"))
                range_idx = min(range_idx, len(profile.radius_ranges) - 1)
                sel = profile.radius_ranges[range_idx]
                return float(rng_state.uniform(sel.min_m, sel.max_m))
            else:
                return float(rng_state.uniform(
                    profile.radius_min_m, profile.radius_max_m))

        # For fixed-count (density disabled): pick a target count
        if not profile.density_enabled:
            target_count = int(rng.integers(count_min, count_max + 1))
        else:
            target_count = None

        obstacle_id_counter = 0
        sampling_attempts_used = 0

        while sampling_attempts_used < self.max_obs_attempts:
            # Check count limit
            if len(obstacles) >= count_max:
                if profile.density_enabled:
                    # Reached count_max but density may not be reached
                    actual_density = compute_density(
                        obstacles, region_area, density_mode,
                        profile.vehicle_radius_m, profile.safety_margin_m)
                    if actual_density < target_density:
                        return ([], "SCENE_COUNT_LIMIT_REACHED_BEFORE_DENSITY",
                                target_density, density_mode.value)
                break

            # Sample radius (multi-range or single-range)
            radius = _sample_radius(rng)
            height = float(rng.uniform(height_min, height_max))

            # Sample z
            z_min_bound = self.obstacle_region.z_min + height / 2.0
            z_max_bound = self.obstacle_region.z_max - height / 2.0
            if z_min_bound >= z_max_bound:
                z = self.obstacle_region.z_min + height / 2.0 + 0.01
            else:
                z = float(rng.uniform(z_min_bound, z_max_bound))

            # Sample center (x, y) within region accounting for radius + margin
            eff_margin = radius + boundary_margin
            x_min_bound = self.obstacle_region.x_min + eff_margin
            x_max_bound = self.obstacle_region.x_max - eff_margin
            y_min_bound = self.obstacle_region.y_min + eff_margin
            y_max_bound = self.obstacle_region.y_max - eff_margin

            if x_min_bound >= x_max_bound or y_min_bound >= y_max_bound:
                sampling_attempts_used += 1
                continue

            cx = float(rng.uniform(x_min_bound, x_max_bound))
            cy = float(rng.uniform(y_min_bound, y_max_bound))

            center_xy = np.array([cx, cy])

            # Check region boundary
            if not self.obstacle_region.contains_cylinder(center_xy, radius, boundary_margin):
                sampling_attempts_used += 1
                continue

            # Check pairwise passable gap against all existing obstacles
            pairwise_ok = True
            for prev in obstacles:
                prev_xy = prev.center_xy()
                d = float(np.linalg.norm(center_xy - prev_xy))

                # Surface gap
                surface_gap = d - radius - prev.radius_m
                if surface_gap < min_surface_gap:
                    pairwise_ok = False
                    break

                # Post-inflation gap
                # post_inflation_gap = surface_gap - 2 * (vehicle_r + safety_m)
                post_inflation_gap = d - (radius + infl_radius) - (prev.radius_m + infl_radius)
                if post_inflation_gap < min_post_inflation_gap:
                    pairwise_ok = False
                    break

            if not pairwise_ok:
                sampling_attempts_used += 1
                continue

            # Accept candidate
            obs_id = "cylinder_{:04d}".format(obstacle_id_counter)
            obstacle_id_counter += 1
            obs = CylinderObstacleSpec(
                obstacle_id=obs_id,
                center_world=np.array([cx, cy, z]),
                radius_m=radius,
                height_m=height,
            )
            obstacles.append(obs)

            # Check stopping condition
            if profile.density_enabled and target_density is not None:
                if len(obstacles) >= count_min:
                    actual_density = compute_density(
                        obstacles, region_area, density_mode,
                        profile.vehicle_radius_m, profile.safety_margin_m)
                    if actual_density >= target_density:
                        # Success: density target reached with at least count_min obstacles
                        break
            elif target_count is not None:
                if len(obstacles) >= target_count:
                    break

        # Post-generation checks
        if len(obstacles) < count_min:
            return ([], "SCENE_OBSTACLE_SAMPLING_EXHAUSTED",
                    target_density, density_mode.value)

        if profile.density_enabled and target_density is not None:
            actual_density = compute_density(
                obstacles, region_area, density_mode,
                profile.vehicle_radius_m, profile.safety_margin_m)
            if actual_density < target_density:
                return ([], "SCENE_TARGET_DENSITY_UNREACHABLE",
                        target_density, density_mode.value)

        return (obstacles, "", target_density, density_mode.value)

    # ── Density-driven generation (v10) ──────────────────────────────

    def generate_scene_density_driven(self, profile, effective_scene_seed,
                                       scene_index_in_profile, attempt_index=0):
        """Generate a scene using the density-driven algorithm (v10).

        Algorithm:
          1. Compute target inflated area = region_area × target_density.
          2. Feasibility: target_area > 2 × max(inflated_area_over_all_groups).
          3. Split budget equally among large/medium/small groups.
          4. Large → medium → small placement order:
             - Borrow from small group if large can't fit min obstacle.
             - Place obstacles randomly; check inflated gap + boundary.
             - Stop when: consecutive_fails >= threshold OR remaining < min_area.
          5. Remaining capacity cascades to next group.

        Returns:
            (obstacles, rejection_reason, target_density, density_mode_str)
        """
        rng = self._make_np_rng(effective_scene_seed + attempt_index * 10007)
        region_area = compute_region_area(self.obstacle_region)
        infl_r = profile.inflation_radius_m

        # ── Sanity: need at least 3 size groups ──
        if not profile.size_groups or len(profile.size_groups) < 3:
            return ([], "DENSITY_DRIVEN_NEEDS_3_SIZE_GROUPS",
                    None, "inflated_occupancy")

        # Sort groups by radius: large (idx 0), medium (idx 1), small (idx 2)
        groups = sorted(profile.size_groups,
                        key=lambda g: g.radius_max_m, reverse=True)
        g_large, g_medium, g_small = groups[0], groups[1], groups[2]

        # ── 1. Target density and area ──
        target_density = float(rng.uniform(
            profile.total_density_min, profile.total_density_max))
        target_inflated_area = region_area * target_density

        # ── 2. Feasibility check ──
        max_inflated = g_large.max_inflated_area_m2(infl_r)
        if target_inflated_area <= 2.0 * max_inflated:
            return ([], "DENSITY_DRIVEN_AREA_TOO_SMALL_FOR_TWO_MAX_OBSTACLES",
                    target_density, "inflated_occupancy")

        # ── 3. Budget allocation (1/3 each) ──
        budget_per_group = target_inflated_area / 3.0

        # Capacity borrowing: if large group can't fit even 1 min obstacle
        min_large_area = g_large.min_inflated_area_m2(infl_r)
        if budget_per_group < min_large_area:
            deficit = min_large_area - budget_per_group
            small_budget = max(0.0, budget_per_group - deficit)
            large_budget = budget_per_group + deficit
            medium_budget = budget_per_group
        else:
            large_budget = budget_per_group
            medium_budget = budget_per_group
            small_budget = budget_per_group

        # ── 4. Placement ──
        obstacles = []
        obstacle_id_counter = 0
        boundary_margin = profile.region_boundary_margin_m
        min_post_inflation_gap = profile.minimum_post_inflation_gap_m
        height_min = profile.height_min_m
        height_max = profile.height_max_m

        # Track spent inflated area per group
        spent_large = 0.0
        spent_medium = 0.0
        spent_small = 0.0

        def _inflated_area(radius):
            return math.pi * (radius + infl_r) ** 2

        def _try_place(group, spent_budget, budget_limit, obstacles_list,
                       consecutive_fail_threshold, rng_state, id_counter):
            """Place obstacles from one size group.

            Returns: (updated_spent, new_obstacles, updated_id_counter)
            """
            consecutive_fails = 0
            new_obs = []
            local_spent = spent_budget
            local_id = id_counter

            while True:
                # Stop conditions
                min_area = group.min_inflated_area_m2(infl_r)
                remaining = budget_limit - local_spent
                if remaining < min_area:
                    break  # can't fit even smallest obstacle in this group
                if consecutive_fails >= consecutive_fail_threshold:
                    break  # too many consecutive placement failures

                # Sample radius and position
                radius = group.sample_radius(rng_state)
                area = _inflated_area(radius)
                if area > remaining:
                    consecutive_fails += 1
                    continue

                height = float(rng_state.uniform(height_min, height_max))

                # Sample z
                z_min_bound = self.obstacle_region.z_min + height / 2.0
                z_max_bound = self.obstacle_region.z_max - height / 2.0
                if z_min_bound >= z_max_bound:
                    z = self.obstacle_region.z_min + height / 2.0 + 0.01
                else:
                    z = float(rng_state.uniform(z_min_bound, z_max_bound))

                # Sample (x, y) within region
                eff_margin = radius + boundary_margin
                x_min_bound = self.obstacle_region.x_min + eff_margin
                x_max_bound = self.obstacle_region.x_max - eff_margin
                y_min_bound = self.obstacle_region.y_min + eff_margin
                y_max_bound = self.obstacle_region.y_max - eff_margin

                if x_min_bound >= x_max_bound or y_min_bound >= y_max_bound:
                    consecutive_fails += 1
                    continue

                cx = float(rng_state.uniform(x_min_bound, x_max_bound))
                cy = float(rng_state.uniform(y_min_bound, y_max_bound))
                center_xy = np.array([cx, cy])

                # Check region boundary
                if not self.obstacle_region.contains_cylinder(
                        center_xy, radius, boundary_margin):
                    consecutive_fails += 1
                    continue

                # Check pairwise inflated gap against ALL existing obstacles
                all_obs = obstacles_list + new_obs
                pairwise_ok = True
                for prev in all_obs:
                    prev_xy = prev.center_xy()
                    d = float(np.linalg.norm(center_xy - prev_xy))
                    # Post-inflation gap (the one that matters)
                    post_gap = d - (radius + infl_r) - (prev.radius_m + infl_r)
                    if post_gap < min_post_inflation_gap:
                        pairwise_ok = False
                        break
                    # No overlap (hard constraint)
                    surface_gap = d - radius - prev.radius_m
                    if surface_gap < 0.0:
                        pairwise_ok = False
                        break

                if not pairwise_ok:
                    consecutive_fails += 1
                    continue

                # Accept
                obs_id = "cylinder_{:04d}".format(local_id)
                local_id += 1

                obs = CylinderObstacleSpec(
                    obstacle_id=obs_id,
                    center_world=np.array([cx, cy, z]),
                    radius_m=radius,
                    height_m=height,
                )
                new_obs.append(obs)
                local_spent += area
                consecutive_fails = 0

            return local_spent, new_obs, local_id

        # ── Place large obstacles ──
        spent_large, large_obs, obstacle_id_counter = _try_place(
            g_large, spent_large, large_budget, obstacles,
            g_large.consecutive_fail_threshold, rng, obstacle_id_counter)
        obstacles.extend(large_obs)

        # Remaining large budget → medium
        remaining_large = max(0.0, large_budget - spent_large)
        medium_budget += remaining_large

        # ── Place medium obstacles ──
        spent_medium, medium_obs, obstacle_id_counter = _try_place(
            g_medium, spent_medium, medium_budget, obstacles,
            g_medium.consecutive_fail_threshold, rng, obstacle_id_counter)
        obstacles.extend(medium_obs)

        # Remaining medium budget → small (regardless of medium success)
        remaining_medium = max(0.0, medium_budget - spent_medium)
        small_budget += remaining_medium

        # ── Place small obstacles ──
        spent_small, small_obs, obstacle_id_counter = _try_place(
            g_small, spent_small, small_budget, obstacles,
            g_small.consecutive_fail_threshold, rng, obstacle_id_counter)
        obstacles.extend(small_obs)

        # ── Post-generation: any obstacles at all? ──
        if len(obstacles) == 0:
            return ([], "DENSITY_DRIVEN_NO_OBSTACLES_PLACED",
                    target_density, "inflated_occupancy")

        # ── Density report ──
        actual_density = compute_inflated_occupancy(
            obstacles, region_area,
            profile.vehicle_radius_m, profile.safety_margin_m)
        rospy.loginfo("[DensityDriven] profile='%s' scene=%d: placed %d obstacles "
                      "(L=%d M=%d S=%d), density=%.3f (target=%.3f), "
                      "budget: L=%.1f M=%.1f S=%.1f m²",
                      profile.name, scene_index_in_profile, len(obstacles),
                      len(large_obs), len(medium_obs), len(small_obs),
                      actual_density, target_density,
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
        cfg = config.get("global", {}).get("scene_generation", {}).get("task_generation", {})
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

        sr = cfg.get("start_sampling_region", {})
        self.start_regions = [ObstacleRegion(
            x_min=float(sr.get("x_min", -12.0)),
            x_max=float(sr.get("x_max", -8.0)),
            y_min=float(sr.get("y_min", -8.0)),
            y_max=float(sr.get("y_max", 8.0)),
            z_min=float(cfg.get("start_height_min_m", 1.8)),
            z_max=float(cfg.get("start_height_max_m", 2.2)),
        )]
        gr = cfg.get("goal_sampling_region", {})
        self.goal_regions = [ObstacleRegion(
            x_min=float(gr.get("x_min", 8.0)),
            x_max=float(gr.get("x_max", 12.0)),
            y_min=float(gr.get("y_min", -8.0)),
            y_max=float(gr.get("y_max", 8.0)),
            z_min=float(cfg.get("goal_height_min_m", 1.8)),
            z_max=float(cfg.get("goal_height_max_m", 2.2)),
        )]

        # Multi-region support (v6) — overwrites single if present
        srs = cfg.get("start_sampling_regions", None)
        grs = cfg.get("goal_sampling_regions", None)
        if srs is not None and grs is not None:
            self.start_regions = [_region_from_dict(r, self.start_h_min, self.start_h_max) for r in srs]
            self.goal_regions = [_region_from_dict(r, self.goal_h_min, self.goal_h_max) for r in grs]

    def set_tasks_per_scene(self, n):
        """Override the number of tasks to generate per scene.

        Used by profile mode to apply profile-specific tasks_per_scene.
        """
        self.tasks_per_scene = int(n)

    def configure_from_profile(self, profile):
        """Apply all profile-specific task generation parameters.

        Called before generate_tasks() when running in profile mode.
        This fixes the bug where profile-level start_sampling_region,
        goal_sampling_region, height ranges, distance limits, and
        blocking requirements were parsed but never applied at runtime.

        Args:
            profile: SceneGenerationProfile or None.
        """
        if profile is None:
            return

        # ── Task count ──────────────────────────────────────────
        self.tasks_per_scene = int(profile.tasks_per_scene)

        # ── Height ranges ───────────────────────────────────────
        self.start_h_min = float(profile.start_height_min_m)
        self.start_h_max = float(profile.start_height_max_m)
        self.goal_h_min = float(profile.goal_height_min_m)
        self.goal_h_max = float(profile.goal_height_max_m)
        self.max_h_diff = float(profile.maximum_start_goal_height_difference_m)

        # ── Distance limits ─────────────────────────────────────
        self.min_dist = float(profile.minimum_start_goal_distance_m)
        self.max_dist = float(profile.maximum_start_goal_distance_m)

        # ── Blocked-path requirements ───────────────────────────
        self.require_blocked = bool(profile.require_direct_path_blocked)
        self.min_blockers = int(profile.minimum_direct_blocker_count)
        self.max_blockers = int(profile.maximum_direct_blocker_count)

        # ── A* reachability ─────────────────────────────────────
        self.require_astar = bool(profile.require_astar_reachable)
        self.min_detour = float(profile.minimum_detour_ratio)
        self.max_detour = float(profile.maximum_detour_ratio)

        # ── Sampling regions (v6: multi-region) ────────────────
        if profile.start_sampling_regions:
            self.start_regions = [
                _region_from_dict(r, self.start_h_min, self.start_h_max)
                for r in profile.start_sampling_regions
            ]
        elif profile.start_sampling_region:
            sr = profile.start_sampling_region
            self.start_regions = [ObstacleRegion(
                x_min=float(sr.get("x_min", self.start_regions[0].x_min)),
                x_max=float(sr.get("x_max", self.start_regions[0].x_max)),
                y_min=float(sr.get("y_min", self.start_regions[0].y_min)),
                y_max=float(sr.get("y_max", self.start_regions[0].y_max)),
                z_min=self.start_h_min,
                z_max=self.start_h_max,
            )]

        if profile.goal_sampling_regions:
            self.goal_regions = [
                _region_from_dict(r, self.goal_h_min, self.goal_h_max)
                for r in profile.goal_sampling_regions
            ]
        elif profile.goal_sampling_region:
            gr = profile.goal_sampling_region
            self.goal_regions = [ObstacleRegion(
                x_min=float(gr.get("x_min", self.goal_regions[0].x_min)),
                x_max=float(gr.get("x_max", self.goal_regions[0].x_max)),
                y_min=float(gr.get("y_min", self.goal_regions[0].y_min)),
                y_max=float(gr.get("y_max", self.goal_regions[0].y_max)),
                z_min=self.goal_h_min,
                z_max=self.goal_h_max,
            )]

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

        for task_idx in range(self.tasks_per_scene):
            accepted = False
            for attempt in range(self.max_attempts):
                # Round-robin distribution across region pairs (v6)
                # Ensures tasks are evenly spread across all start/goal pairs.
                pair_idx = task_idx % len(self.start_regions)
                start_r = self.start_regions[pair_idx]
                goal_r = self.goal_regions[min(pair_idx, len(self.goal_regions) - 1)]

                sx = rng.uniform(start_r.x_min, start_r.x_max)
                sy = rng.uniform(start_r.y_min, start_r.y_max)
                sz = rng.uniform(self.start_h_min, self.start_h_max)

                gx = rng.uniform(goal_r.x_min, goal_r.x_max)
                gy = rng.uniform(goal_r.y_min, goal_r.y_max)
                gz = rng.uniform(self.goal_h_min, self.goal_h_max)

                start = np.array([sx, sy, sz])
                goal = np.array([gx, gy, gz])

                result = self._validate_task(
                    start, goal, obstacles, esdf, esdf_origin, esdf_res,
                    astar_planner_fn)

                if result.valid:
                    tasks.append((start.tolist(), goal.tolist(), result))
                    accepted = True
                    break

            if not accepted:
                rospy.logwarn("[TaskGen] Could not generate valid task %d after %d attempts.",
                              task_idx, self.max_attempts)

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
        direct_dir = direct_vec / direct_dist
        n_blockers = 0
        for obs in obstacles:
            obs_xy = obs.center_xy()
            proj = float(np.dot(
                obs_xy - start[:2], direct_dir[:2]))
            if 0 < proj < direct_dist:
                closest = start[:2] + proj * direct_dir[:2]
                lateral = float(np.linalg.norm(obs_xy - closest))
                if lateral < obs.radius_m + self.corridor_r:
                    n_blockers += 1
        result.direct_blocker_count = n_blockers
        result.direct_path_blocked = (n_blockers > 0)

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
        dom_obs = self._find_dominant_obstacle(start, goal, obstacles)
        if dom_obs is not None:
            result.dominant_obstacle_id = dom_obs.obstacle_id
            # Lateral offset
            direct_dir_xy = direct_vec[:2] / max(float(np.linalg.norm(direct_vec[:2])), 1e-6)
            n_xy = np.array([-direct_dir_xy[1], direct_dir_xy[0]])
            result.dominant_obstacle_lateral_offset_m = float(
                np.dot(dom_obs.center_xy() - start[:2], n_xy))
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
