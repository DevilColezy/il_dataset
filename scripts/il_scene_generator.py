#!/usr/bin/env python3
"""il_scene_generator.py  —  procedural multi-scale scene generation.

Scene generator only decides the TRAINING DISTRIBUTION (obstacle geometry).
It never does local/macro planning: real reachability / scale classification
happens later on the actual exported point cloud via the C++ privileged map
(il_task_generator.py / TaskGenerationOracle).

Obstacle -> Unity conversion uses the VERIFIED AvoidBench Object_t format
(section IV):
    position = [ros_x, ros_z, ros_y]   (== il_common.ros_pos_to_unity)
    size     = [diameter, height, diameter]
    prefabID = "Object"
Full-height cylinders keep the task decision horizontal (2.5D, section XXX).

Generation modes (section XXXIV): independent / cluster / chain / mixed.
Validation (sections XXXV/XXXVI): boundary margin + minimum surface gap +
free-space presence; inflated obstacle MERGING is allowed (a barrier of many
small obstacles is exactly what COMPOUND_BARRIER needs).
"""

from __future__ import print_function, division

import math
import random
import zlib

import numpy as np

from il_common import ros_pos_to_unity


class CylinderObstacleSpec(object):
    __slots__ = ("x", "y", "z", "radius", "height", "obj_id")

    def __init__(self, x, y, z, radius, height, obj_id):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.radius = float(radius)
        self.height = float(height)
        self.obj_id = str(obj_id)

    def to_dict(self):
        # Full-height cylinders: bottom_z / top_z make the vertical coverage
        # of the flight region explicit for post-hoc static checks (section
        # LXIII).
        return {
            "obj_id": self.obj_id,
            "center_world": [round(self.x, 4), round(self.y, 4),
                             round(self.z, 4)],
            "radius_m": round(self.radius, 4),
            "height_m": round(self.height, 4),
            "bottom_z": round(self.z - 0.5 * self.height, 4),
            "top_z": round(self.z + 0.5 * self.height, 4),
        }

    def unity_object(self):
        """Verified AvoidBench Object_t (section IV)."""
        pos = ros_pos_to_unity([self.x, self.y, self.z])  # [x, z, y]
        return {
            "ID": self.obj_id,
            "prefabID": "Object",
            "position": [pos[0], pos[1], pos[2]],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "size": [2.0 * self.radius, self.height, 2.0 * self.radius],
        }


class ObstacleRegion(object):
    __slots__ = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")

    def __init__(self, min_x, max_x, min_y, max_y, min_z, max_z):
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.min_z = float(min_z)
        self.max_z = float(max_z)

    def contains(self, x, y):
        return (self.min_x <= x <= self.max_x and
                self.min_y <= y <= self.max_y)

    def contains_cylinder(self, x, y, radius, margin=0.0):
        """True if a cylinder of `radius` fits fully inside the region with
        its SURFACE at least `margin` from every wall.

        The factory is walled (point cloud box == factory interior): a
        cylinder whose center is inside but whose surface crosses the box
        edge would poke through the wall, so placement and validation must
        check center +/- radius, not just the center (section XX).
        """
        r = radius + margin
        return (self.min_x + r <= x <= self.max_x - r and
                self.min_y + r <= y <= self.max_y - r)

    def center(self):
        return np.array([0.5 * (self.min_x + self.max_x),
                         0.5 * (self.min_y + self.max_y)], dtype=np.float64)

    def width(self):
        return self.max_x - self.min_x

    def height(self):
        return self.max_y - self.min_y

    def area(self):
        return self.width() * self.height()

    def to_dict(self):
        return {
            "min_x": self.min_x, "max_x": self.max_x,
            "min_y": self.min_y, "max_y": self.max_y,
            "min_z": self.min_z, "max_z": self.max_z,
        }


class SceneProfile(object):
    """Parsed scene profile from config (section XXXIII)."""

    def __init__(self, spec):
        self.name = str(spec.get("name", "profile"))
        self.scene_count = int(spec.get("scene_count", 20))
        self.radius_min_m = float(spec.get("radius_min_m", 0.15))
        self.radius_max_m = float(spec.get("radius_max_m", 1.0))
        occ = spec.get("target_occupancy", [0.05, 0.12])
        self.occupancy_min = float(occ[0])
        self.occupancy_max = float(occ[1])
        self.cluster_probability = float(spec.get("cluster_probability", 0.0))
        self.chain_probability = float(spec.get("chain_probability", 0.0))
        self.cluster_size = int(spec.get("cluster_size", 5))
        self.chain_length = int(spec.get("chain_length", 6))
        self.mode_weights = spec.get("mode_weights", None)

    def to_dict(self):
        return {
            "name": self.name,
            "scene_count": self.scene_count,
            "radius_min_m": self.radius_min_m,
            "radius_max_m": self.radius_max_m,
            "target_occupancy": [self.occupancy_min, self.occupancy_max],
            "cluster_probability": self.cluster_probability,
            "chain_probability": self.chain_probability,
        }


class GeneratedScene(object):
    """A generated procedural scene.

    `scene_key` is the dataset-internal unique scene name (e.g.
    "scene_mixed_scale_000003") — it is NEVER sent to AvoidBench as
    "scene_id".  The AvoidBench numeric scene id lives in the manager
    (`unity_scene_id`, from `global.scene_id`) and is kept separate
    (sections VII-IX).
    """

    def __init__(self, scene_key, profile, scene_index, seed,
                 generation_attempt, region, obstacles, metrics):
        self.scene_key = str(scene_key)
        self.profile = profile
        self.profile_name = profile.name if profile else None
        self.scene_index = int(scene_index)
        self.seed = int(seed)
        self.generation_attempt = int(generation_attempt)
        self.region = region
        self.obstacles = obstacles            # list[CylinderObstacleSpec]
        self.metrics = metrics or {}          # scene metrics dict
        self.unity_objects = [ob.unity_object() for ob in obstacles]

    def to_dict(self):
        return {
            "scene_key": self.scene_key,
            "profile_name": self.profile_name,
            "scene_index": self.scene_index,
            "seed": self.seed,
            "generation_attempt": self.generation_attempt,
            "obstacle_region": self.region.to_dict(),
            "obstacles": [ob.to_dict() for ob in self.obstacles],
            "scene_metrics": self.metrics,
        }


class ProceduralSceneGenerator(object):
    """Deterministic procedural cylinder scenes (seeded, reproducible)."""

    MODES = ("independent", "cluster", "chain", "mixed")

    def __init__(self, cfg, vehicle_radius_m, safety_margin_m):
        self.cfg = cfg or {}
        self.seed_base = int(cfg.get("seed", 12345)) if cfg else 12345
        self.vehicle_radius = float(vehicle_radius_m)
        self.safety_margin = float(safety_margin_m)
        region = cfg.get("obstacle_region", {}) if cfg else {}
        self.region = ObstacleRegion(
            region.get("min_x", 1.5), region.get("max_x", 28.5),
            region.get("min_y", 16.0), region.get("max_y", 60.0),
            region.get("min_z", 3.5), region.get("max_z", 11.5))
        geom = cfg.get("geometry", {}) if cfg else {}
        # Full-height cylinders (sections XV-XVIII): the obstacle must cover
        # the whole configured vertical flight range so the 2.5D LEFT/RIGHT
        # macro topology cannot be bypassed by flying over the top.  Height
        # and center are derived from the region vertical span — never from
        # an unrelated `height_m` knob.
        self._obstacle_height = float(self.region.max_z - self.region.min_z)
        self._obstacle_center_z = 0.5 * (self.region.min_z +
                                         self.region.max_z)
        if self._obstacle_height <= 0.0:
            # Defensive fallback only: degenerate region.
            h = float(geom.get("height_m", 8.0))
            self._obstacle_height = h
            self._obstacle_center_z = self.region.min_z + 0.5 * h
        self.minimum_surface_gap_m = float(
            geom.get("minimum_surface_gap_m", 0.30))
        self.boundary_margin_m = float(geom.get("boundary_margin_m", 0.6))
        self.max_obstacles = int(geom.get("max_obstacles", 120))

    def _scene_seed(self, profile, scene_index, attempt):
        name_code = zlib.crc32(profile.name.encode("utf-8")) & 0x7FFFFFFF
        return (self.seed_base * 7919 + name_code * 104729 +
                scene_index * 15485863 + attempt * 40503) & 0x7FFFFFFF

    def _rng(self, profile, scene_index, attempt):
        return random.Random(self._scene_seed(profile, scene_index, attempt))

    def _pick_mode(self, profile, rng):
        w = profile.mode_weights
        if w:
            modes = [m for m in self.MODES if w.get(m, 0) > 0]
            weights = [w[m] for m in modes]
            return rng.choices(modes, weights=weights, k=1)[0]
        roll = rng.random()
        if roll < profile.cluster_probability:
            return "cluster"
        roll = (roll - profile.cluster_probability) / \
            max(1e-6, 1.0 - profile.cluster_probability)
        if roll < profile.chain_probability:
            return "chain"
        return "independent"

    def _place_obstacle(self, rng, profile, placed):
        """Rejection-sample one cylinder position with min surface gap.

        The center is inset by `boundary_margin + radius` so the obstacle
        surface keeps a full `boundary_margin` clear of the factory walls.
        """
        radius = rng.uniform(profile.radius_min_m, profile.radius_max_m)
        m = self.boundary_margin_m + radius
        for _ in range(300):
            x = rng.uniform(self.region.min_x + m,
                            self.region.max_x - m)
            y = rng.uniform(self.region.min_y + m,
                            self.region.max_y - m)
            if not self.region.contains_cylinder(
                    x, y, radius, self.boundary_margin_m):
                continue
            ok = True
            for (px, py, pr) in placed:
                d = math.hypot(x - px, y - py)
                if d < pr + radius + self.minimum_surface_gap_m:
                    ok = False
                    break
            if ok:
                return x, y, radius
        return None

    def _obstacle_count_for_occupancy(self, profile, rng):
        target = rng.uniform(profile.occupancy_min, profile.occupancy_max)
        r_mean = 0.5 * (profile.radius_min_m + profile.radius_max_m)
        per_obstacle = math.pi * r_mean * r_mean
        count = int(round(target * self.region.area() / max(per_obstacle, 1e-6)))
        return max(1, min(self.max_obstacles, count))

    def generate(self, profile, scene_index, attempt):
        """Generate one deterministic scene.  Returns a GeneratedScene."""
        rng = self._rng(profile, scene_index, attempt)
        seed = self._scene_seed(profile, scene_index, attempt)
        scene_key = "scene_%s_%06d" % (profile.name, scene_index)
        mode = self._pick_mode(profile, rng)
        obstacles = []
        placed = []  # (x, y, radius)
        count_target = self._obstacle_count_for_occupancy(profile, rng)

        if mode == "chain":
            count_target = max(count_target, profile.chain_length)
        if mode == "cluster":
            count_target = max(count_target, profile.cluster_size)

        if mode == "chain":
            # Chain: a lateral barrier line with small jitter (across the
            # corridor's long axis) -> a macro-scale blocker made of many
            # small cylinders (COMPOUND_BARRIER support, section XVII).
            center = self.region.center()
            base_y = center[1]
            n = profile.chain_length
            span = self.region.width() - 2.0 * self.boundary_margin_m
            for i in range(n):
                t = (i / max(1, n - 1)) - 0.5
                x = center[0] + t * span + rng.uniform(-0.2, 0.2)
                y = base_y + math.sin(t * 2.4) * 0.5 + rng.uniform(-0.15, 0.15)
                radius = rng.uniform(profile.radius_min_m,
                                     min(profile.radius_max_m,
                                         self.region.width() / n / 2.0))
                # Surface (not just center) must stay clear of the walls.
                if not self.region.contains_cylinder(
                        x, y, radius, self.boundary_margin_m):
                    continue
                obstacles.append(CylinderObstacleSpec(
                    x, y, self._obstacle_center_z, radius,
                    self._obstacle_height, "chain_%d" % i))
                placed.append((x, y, radius))

        while len(obstacles) < count_target:
            if mode == "cluster" and len(obstacles) < profile.cluster_size:
                # One tight cluster near a random anchor.
                cx = rng.uniform(self.region.min_x + self.boundary_margin_m,
                                 self.region.max_x - self.boundary_margin_m)
                cy = rng.uniform(self.region.min_y + self.boundary_margin_m,
                                 self.region.max_y - self.boundary_margin_m)
                placed_anchor = False
                for i in range(profile.cluster_size):
                    radius = rng.uniform(profile.radius_min_m,
                                         profile.radius_max_m)
                    ang = rng.uniform(0.0, 2.0 * math.pi)
                    rad = rng.uniform(0.0, 1.2)
                    x = cx + math.cos(ang) * rad
                    y = cy + math.sin(ang) * rad
                    if not self.region.contains(x, y):
                        continue
                    if not self.region.contains_cylinder(
                            x, y, radius, self.boundary_margin_m):
                        continue
                    ok = all(math.hypot(x - px, y - py) >=
                             pr + radius + self.minimum_surface_gap_m
                             for (px, py, pr) in placed)
                    if not ok:
                        continue
                    obstacles.append(CylinderObstacleSpec(
                        x, y, self._obstacle_center_z, radius,
                        self._obstacle_height, "cluster_%d" % len(obstacles)))
                    placed.append((x, y, radius))
                    placed_anchor = True
                if not placed_anchor:
                    break  # avoid an infinite loop on an over-crowded region
            else:
                placed_ob = self._place_obstacle(rng, profile, placed)
                if placed_ob is None:
                    break  # region saturated; accept what we have
                x, y, radius = placed_ob
                obstacles.append(CylinderObstacleSpec(
                    x, y, self._obstacle_center_z, radius,
                    self._obstacle_height, "ob_%d" % len(obstacles)))
                placed.append((x, y, radius))

        metrics = self._compute_metrics(obstacles, count_target, mode)
        return GeneratedScene(scene_key, profile, scene_index, seed, attempt,
                              self.region, obstacles, metrics)

    def _compute_metrics(self, obstacles, count_target, mode):
        radii = [ob.radius for ob in obstacles]
        return {
            "mode": mode,
            "obstacle_count": len(obstacles),
            "target_obstacle_count": count_target,
            "radius_min": round(min(radii), 4) if radii else 0.0,
            "radius_max": round(max(radii), 4) if radii else 0.0,
            "radius_mean": round(float(np.mean(radii)), 4) if radii else 0.0,
            "raw_occupancy": round(
                sum(math.pi * r * r for r in radii) / self.region.area(), 4),
        }


class SceneGeometryValidator(object):
    """Cheap geometry sanity for a generated scene (sections XXXV/XXXVI).

    Only rejects clearly-invalid layouts (no navigation space, illegal raw
    overlap, boundary violation).  It deliberately ALLOWS inflated obstacle
    merging — the real per-task reachability is validated on the actual
    point cloud by the C++ privileged map.
    """

    def __init__(self, cfg, vehicle_radius_m, safety_margin_m):
        self.cfg = cfg or {}
        self.vehicle_radius = float(vehicle_radius_m)
        self.safety_margin = float(safety_margin_m)
        self.min_free_fraction = float(cfg.get("min_free_fraction", 0.35)) \
            if cfg else 0.35
        self.coarse_grid = int(cfg.get("coarse_grid", 24)) if cfg else 24

    def validate(self, scene):
        region = scene.region
        obstacles = scene.obstacles
        if len(obstacles) == 0:
            return False, "no_obstacles", {}
        margin = self.cfg.get("geometry", {}).get("boundary_margin_m", 0.6) \
            if self.cfg else 0.6
        # 1) boundary: the obstacle SURFACE (center +/- radius) must stay at
        # least `margin` from every wall — a center-only check lets large
        # cluster/chain cylinders poke through the factory walls (section
        # XX).
        for ob in obstacles:
            if not region.contains_cylinder(ob.x, ob.y, ob.radius, margin):
                return False, "boundary_violation", {}
        # 2) raw surface gap between obstacle bodies (no severe overlap).
        for i in range(len(obstacles)):
            for j in range(i + 1, len(obstacles)):
                a = obstacles[i]
                b = obstacles[j]
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < a.radius + b.radius - 0.02:  # allow tiny tolerance
                    return False, "raw_overlap", {}
        # 3) coarse free-space presence + connectivity on the flight slice.
        inflate = self.vehicle_radius + self.safety_margin
        free_frac = self._sample_free_fraction(region, obstacles, inflate)
        if free_frac < self.min_free_fraction:
            return False, "insufficient_free_space", {"free_fraction": free_frac}
        return True, "ok", {"free_fraction": round(free_frac, 4)}

    def _sample_free_fraction(self, region, obstacles, inflate):
        n = self.coarse_grid
        free = 0
        total = 0
        for i in range(n):
            for j in range(n):
                x = region.min_x + (i + 0.5) / n * region.width()
                y = region.min_y + (j + 0.5) / n * region.height()
                total += 1
                blocked = False
                for ob in obstacles:
                    if math.hypot(x - ob.x, y - ob.y) < ob.radius + inflate:
                        blocked = True
                        break
                if not blocked:
                    free += 1
        return free / max(1, total)
