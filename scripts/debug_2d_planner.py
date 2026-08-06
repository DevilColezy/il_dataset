#!/usr/bin/env python3
"""
2-D offline debug harness for the observed-map macro expert and bounded A*.

Run this on Windows WITHOUT ROS / Flightmare / Unity to verify that the
complete pipeline (depth → observed map → ESDF → frontier extraction →
bounded A* → B-spline seed) can navigate around obstacles using only
causal observations.  The C++ pybind11 backend is replaced by pure-Python
fallbacks that match its semantics as closely as possible.

Usage:
    python debug_2d_planner.py                     # interactive default scene
    python debug_2d_planner.py --scene narrow_gap  # predefined scene
    python debug_2d_planner.py --scene random      # random cylinder field

Controls (when interactive window is open):
    space / enter  — advance one macro tick (0.2 s)
    r              — run until goal reached or blocked
    q / esc        — quit
"""

from __future__ import division, print_function

import argparse
import math
import os
import sys
import time
from collections import deque

import numpy as np

# ── find the scripts directory ──────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Stub ROS
try:
    import rospy
except ImportError:
    import types
    _stub = types.ModuleType("rospy")
    _stub.loginfo = lambda *a, **kw: None
    _stub.logwarn = lambda *a, **kw: None
    _stub.logerr = lambda *a, **kw: None
    sys.modules["rospy"] = _stub
try:
    import rospkg
except ImportError:
    sys.modules["rospkg"] = types.ModuleType("rospkg")


# ══════════════════════════════════════════════════════════════════════════
#  Scene definitions
# ══════════════════════════════════════════════════════════════════════════

SCENES = {}


def _register(name):
    def dec(fn):
        SCENES[name] = fn
        return fn
    return dec


@_register("default")
def scene_default():
    """One big cylinder left, one right — forces a bypass decision."""
    return {
        "start": np.array([0.0, 0.5, 2.0], dtype=np.float64),
        "goal": np.array([0.0, 8.0, 2.0], dtype=np.float64),
        "obstacles": [
            {"cx": -0.5, "cy": 3.5, "r": 0.45},
            {"cx":  0.5, "cy": 5.0, "r": 0.50},
        ],
    }


@_register("narrow_gap")
def scene_narrow_gap():
    """Two cylinders with a narrow gap — tests tight-corridor planning."""
    return {
        "start": np.array([0.0, 1.0, 2.0], dtype=np.float64),
        "goal": np.array([0.0, 8.0, 2.0], dtype=np.float64),
        "obstacles": [
            {"cx": -0.45, "cy": 3.0, "r": 0.50},
            {"cx":  0.45, "cy": 3.0, "r": 0.50},
            {"cx": -0.35, "cy": 5.5, "r": 0.55},
            {"cx":  0.40, "cy": 5.5, "r": 0.40},
        ],
    }


@_register("offset_left")
def scene_offset_left():
    """One obstacle left of the direct line — should bypass right."""
    return {
        "start": np.array([0.0, 0.5, 2.0], dtype=np.float64),
        "goal": np.array([0.0, 8.0, 2.0], dtype=np.float64),
        "obstacles": [
            {"cx": -1.2, "cy": 4.0, "r": 0.80},
        ],
    }


@_register("offset_right")
def scene_offset_right():
    """One obstacle right — should bypass left."""
    return {
        "start": np.array([0.0, 0.5, 2.0], dtype=np.float64),
        "goal": np.array([0.0, 8.0, 2.0], dtype=np.float64),
        "obstacles": [
            {"cx": 1.2, "cy": 4.0, "r": 0.80},
        ],
    }


@_register("slalom")
def scene_slalom():
    """Alternating obstacles — requires weaving."""
    return {
        "start": np.array([0.0, 0.5, 2.0], dtype=np.float64),
        "goal": np.array([0.0, 9.0, 2.0], dtype=np.float64),
        "obstacles": [
            {"cx":  0.8, "cy": 2.0, "r": 0.45},
            {"cx": -0.8, "cy": 3.5, "r": 0.45},
            {"cx":  0.8, "cy": 5.0, "r": 0.45},
            {"cx": -0.8, "cy": 6.5, "r": 0.45},
            {"cx":  0.0, "cy": 8.0, "r": 0.30},
        ],
    }


@_register("M01_like")
def scene_M01_like():
    """Mimics the M01_micro_medium_000000 layout: start ~(-2.6,13.8), goal ~(-2.6,17.0)."""
    return {
        "start": np.array([-2.62, 13.81, 2.0], dtype=np.float64),
        "goal": np.array([-2.65, 16.95, 2.0], dtype=np.float64),
        "obstacles": [
            {"cx": -1.8, "cy": 14.8, "r": 0.35},
            {"cx": -3.0, "cy": 15.3, "r": 0.30},
            {"cx": -2.2, "cy": 15.8, "r": 0.40},
            {"cx": -3.2, "cy": 14.2, "r": 0.25},
            {"cx": -1.5, "cy": 14.0, "r": 0.30},
        ],
    }


@_register("random")
def scene_random():
    """Random cylinder field."""
    rng = np.random.RandomState(42)
    n = 12
    obstacles = []
    for _ in range(n):
        cx = (rng.rand() - 0.5) * 6.0
        cy = 1.5 + rng.rand() * 6.0
        r = 0.15 + rng.rand() * 0.50
        obstacles.append({"cx": float(cx), "cy": float(cy), "r": float(r)})
    return {
        "start": np.array([0.0, 0.5, 2.0], dtype=np.float64),
        "goal": np.array([0.0, 8.0, 2.0], dtype=np.float64),
        "obstacles": obstacles,
    }


# ══════════════════════════════════════════════════════════════════════════
#  2-D depth camera simulation
# ══════════════════════════════════════════════════════════════════════════

class SimDepthCamera:
    """Ray-cast in 2-D (horizontal slice).  Returns a 1-D "depth image"."""

    def __init__(self, hfov_rad, max_range_m, n_rays=90):
        self.hfov = hfov_rad
        self.max_range = max_range_m
        self.n_rays = n_rays
        self.angles = np.linspace(-self.hfov / 2, self.hfov / 2, n_rays)

    def cast(self, position_world, yaw, obstacles):
        """Return (depths_m, ray_endpoints_world)."""
        px, py = position_world[0], position_world[1]
        depths = np.full(self.n_rays, self.max_range, dtype=np.float64)
        endpoints = np.zeros((self.n_rays, 3), dtype=np.float64)
        for i, a in enumerate(self.angles):
            # Package convention B: yaw=0 => nose (body +Y) faces world +Y, so
            # the world heading of the nose is yaw + pi/2.
            ray_angle = yaw + a + math.pi / 2.0
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)
            hit_dist = self.max_range
            for obs in obstacles:
                ocx, ocy, r = obs["cx"], obs["cy"], obs["r"]
                # Ray-circle intersection
                ocx_arr = np.array([ocx, ocy])
                f = np.array([px - ocx, py - ocy])
                d = np.array([dx, dy])
                b = np.dot(f, d)
                c = np.dot(f, f) - r * r
                disc = b * b - c
                if disc >= 0.0:
                    sqrt_disc = math.sqrt(disc)
                    t = -b - sqrt_disc
                    if t < 0.0:
                        t = -b + sqrt_disc
                    if 0.0 < t < hit_dist:
                        hit_dist = t
            depths[i] = hit_dist
            endpoints[i, 0] = px + dx * hit_dist
            endpoints[i, 1] = py + dy * hit_dist
            endpoints[i, 2] = 0.0  # z not used
        return depths, endpoints


# ══════════════════════════════════════════════════════════════════════════
#  Pure-Python bounded A* (matches C++ searchLocalSeed semantics)
# ══════════════════════════════════════════════════════════════════════════

def py_bounded_astar(start_world, start_vel, goal_world,
                     esdf_grid, known_mask, origin, resolution,
                     forbid_unknown=True, target_clearance=0.20,
                     min_clearance=0.05, search_resolution=0.10,
                     max_expansions=50000, time_budget_ms=15.0):
    """Pure-Python bounded A* on the observed ESDF.

    Returns (reachable_distance, seed_path_or_None).
    Uses the SAME semantics as the C++ searchLocalSeed.
    """
    t0 = time.time()
    gx, gy, gz = esdf_grid.shape
    inv_res = 1.0 / resolution

    def world_to_grid(pt):
        return (pt - origin) * inv_res

    def grid_to_world(idx):
        return idx * resolution + origin

    def is_known(wx, wy, wz=2.0):
        gi = world_to_grid(np.array([wx, wy, wz]))
        ix0 = int(math.floor(gi[0]))
        iy0 = int(math.floor(gi[1]))
        iz0 = int(math.floor(gi[2]))
        for ix in (ix0, min(ix0 + 1, gx - 1)):
            for iy in (iy0, min(iy0 + 1, gy - 1)):
                for iz in (iz0, min(iz0 + 1, gz - 1)):
                    if (ix < 0 or ix >= gx or iy < 0 or iy >= gy or
                            iz < 0 or iz >= gz):
                        return False
                    if not known_mask[ix, iy, iz]:
                        return False
        return True

    def esdf_value(wx, wy, wz=2.0):
        gi = world_to_grid(np.array([wx, wy, wz]))
        ix0 = int(math.floor(gi[0]))
        iy0 = int(math.floor(gi[1]))
        iz0 = int(math.floor(gi[2]))
        wx_grid = gi[0] - ix0
        wy_grid = gi[1] - iy0
        wz_grid = gi[2] - iz0
        def _clamp(v, mx):
            return max(0, min(mx - 1, v))
        ix0_c = _clamp(ix0, gx)
        ix1_c = _clamp(ix0 + 1, gx)
        iy0_c = _clamp(iy0, gy)
        iy1_c = _clamp(iy0 + 1, gy)
        iz0_c = _clamp(iz0, gz)
        iz1_c = _clamp(iz0 + 1, gz)
        c000 = esdf_grid[ix0_c, iy0_c, iz0_c]
        c100 = esdf_grid[ix1_c, iy0_c, iz0_c]
        c010 = esdf_grid[ix0_c, iy1_c, iz0_c]
        c110 = esdf_grid[ix1_c, iy1_c, iz0_c]
        c001 = esdf_grid[ix0_c, iy0_c, iz1_c]
        c101 = esdf_grid[ix1_c, iy0_c, iz1_c]
        c011 = esdf_grid[ix0_c, iy1_c, iz1_c]
        c111 = esdf_grid[ix1_c, iy1_c, iz1_c]
        c00 = c000 * (1 - wx_grid) + c100 * wx_grid
        c01 = c001 * (1 - wx_grid) + c101 * wx_grid
        c10 = c010 * (1 - wx_grid) + c110 * wx_grid
        c11 = c011 * (1 - wx_grid) + c111 * wx_grid
        c0 = c00 * (1 - wy_grid) + c10 * wy_grid
        c1 = c01 * (1 - wy_grid) + c11 * wy_grid
        return c0 * (1 - wz_grid) + c1 * wz_grid

    # ── Start check ──
    if forbid_unknown and not is_known(start_world[0], start_world[1]):
        return 0.0, None

    # ── Terminal check ──
    term_val = esdf_value(goal_world[0], goal_world[1])
    if not np.isfinite(term_val) or term_val <= min_clearance:
        return 0.0, None

    # ── Direct path check ──
    direct_vec = goal_world - start_world
    direct_dist = float(np.linalg.norm(direct_vec))
    if direct_dist < 1e-6:
        return direct_dist, [start_world, goal_world]

    n_direct = max(2, int(direct_dist / (0.5 * search_resolution)) + 1)
    direct_clear = True
    for i in range(n_direct):
        frac = i / max(n_direct - 1, 1)
        pt = start_world + frac * direct_vec
        if forbid_unknown and not is_known(pt[0], pt[1]):
            direct_clear = False
            break
        val = esdf_value(pt[0], pt[1])
        if val <= target_clearance:
            direct_clear = False
            break
    if direct_clear:
        return direct_dist, [start_world, goal_world]

    # ── Bounded A* ──
    expansion = 1.0
    minimum = np.minimum(start_world, goal_world) - expansion
    maximum = np.maximum(start_world, goal_world) + expansion
    minimum[0] = max(minimum[0], origin[0] + 0.5 * resolution)
    minimum[1] = max(minimum[1], origin[1] + 0.5 * resolution)
    maximum[0] = min(maximum[0], origin[0] + (gx - 0.5) * resolution)
    maximum[1] = min(maximum[1], origin[1] + (gy - 0.5) * resolution)

    dims = ((maximum - minimum)[:2] / search_resolution).astype(int) + 1
    if dims[0] <= 1 or dims[1] <= 1:
        return 0.0, None
    nx, ny = int(dims[0]), int(dims[1])
    total = nx * ny
    if total > 250000:
        return 0.0, None

    def _encode(ix, iy):
        return ix * ny + iy

    def _decode(idx):
        return idx // ny, idx % ny

    def _position(ix, iy):
        pt = minimum[:2] + search_resolution * np.array([ix, iy])
        # z follows the direct profile
        frac = 0.0
        xy_len_sq = float(np.sum((goal_world[:2] - start_world[:2]) ** 2))
        if xy_len_sq > 1e-9:
            frac = max(0.0, min(1.0, float(np.dot(
                pt - start_world[:2],
                goal_world[:2] - start_world[:2])) / xy_len_sq))
        z = start_world[2] + frac * (goal_world[2] - start_world[2])
        return np.array([pt[0], pt[1], z])

    def _nearest_idx(pt):
        gi = ((pt[:2] - minimum[:2]) / search_resolution)
        return (max(0, min(nx - 1, int(round(gi[0])))),
                max(0, min(ny - 1, int(round(gi[1])))))

    start_ix, start_iy = _nearest_idx(start_world)
    goal_ix, goal_iy = _nearest_idx(goal_world)
    start_idx = _encode(start_ix, start_iy)
    goal_idx = _encode(goal_ix, goal_iy)

    cost = np.full(total, np.inf)
    parent = np.full(total, -1, dtype=int)
    closed = np.zeros(total, dtype=np.uint8)
    import heapq
    open_q = []
    cost[start_idx] = 0.0
    heapq.heappush(open_q, (float(np.linalg.norm(goal_world - start_world)),
                             start_idx))
    search_clearance = min_clearance + min(0.15, 0.75 * max(0.0, target_clearance - min_clearance))
    found = False
    expansions = 0
    deadline = t0 + time_budget_ms * 1e-3

    while open_q and time.time() < deadline and expansions < max_expansions:
        _, cur_idx = heapq.heappop(open_q)
        if closed[cur_idx]:
            continue
        closed[cur_idx] = 1
        expansions += 1
        if cur_idx == goal_idx:
            found = True
            break
        cx, cy = _decode(cur_idx)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_i, ny_i = cx + dx, cy + dy
                if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                    continue
                n_idx = _encode(nx_i, ny_i)
                if closed[n_idx]:
                    continue
                n_pos = _position(nx_i, ny_i)
                val = esdf_value(n_pos[0], n_pos[1])
                is_endpoint = (n_idx == start_idx or n_idx == goal_idx)
                # Frontier endpoint check — same as C++ fix
                if n_idx == goal_idx:
                    if not (np.isfinite(val) and val > search_clearance):
                        continue
                else:
                    if forbid_unknown and not is_known(n_pos[0], n_pos[1]):
                        continue
                if not is_endpoint and val <= search_clearance:
                    continue
                step = search_resolution * math.sqrt(dx * dx + dy * dy)
                tentative = cost[cur_idx] + step
                if tentative >= cost[n_idx]:
                    continue
                cost[n_idx] = tentative
                parent[n_idx] = cur_idx
                h = float(np.linalg.norm(goal_world - n_pos))
                heapq.heappush(open_q, (tentative + h, n_idx))

    if not found:
        return 0.0, None

    # Reconstruct path
    path = []
    idx = goal_idx
    while idx >= 0:
        ix, iy = _decode(idx)
        path.append(_position(ix, iy))
        if idx == start_idx:
            break
        idx = parent[idx]
    path.reverse()
    if len(path) < 2:
        return 0.0, None
    path[0] = start_world.copy()
    path[-1] = goal_world.copy()

    # Compute reachable distance from arc length
    arc = 0.0
    for i in range(1, len(path)):
        arc += float(np.linalg.norm(path[i] - path[i - 1]))
    return arc, path


# ══════════════════════════════════════════════════════════════════════════
#  Main debug runner
# ══════════════════════════════════════════════════════════════════════════

class Debug2DRunner:
    """Runs the full pipeline frame-by-frame, collecting diagnostics."""

    def __init__(self, scene, config_overrides=None):
        self.scene = scene
        self.start = scene["start"].copy()
        self.goal = scene["goal"].copy()
        self.obstacles = scene["obstacles"]

        # Drone state
        self.pos = self.start.copy()
        self.yaw = 0.0  # radians, 0 = facing +Y (forward in FLU: X=fwd)
        self.vel = np.zeros(3)
        self.t = 0.0
        self.macro_tick = 0
        self.frame = 0

        # Camera — use enough rays that the observed cone is dense
        self.camera = SimDepthCamera(
            hfov_rad=math.radians(90.0), max_range_m=5.0, n_rays=180)

        # Observed map
        from il_observed_map import (
            RollingObservedOccupancyMap, ObservedESDF, FREE, OCCUPIED, UNKNOWN)
        map_cfg = {"global": {
            "depth": {"fov": 90.0, "max_m": 5.0},
            "esdf": {"drone_radius": 0.30},
            "observed_map": {
                "resolution": 0.10,
                "size_x_m": 8.0,
                "size_y_m": 8.0,
                "size_z_m": 2.0,
                "history_seconds": 8.0,
                "depth_integration_step": 1,   # dense rays for debug
                "free_space_sample_spacing_m": 0.05,  # mark every voxel along ray
            },
        }}
        self.obs_map = RollingObservedOccupancyMap(map_cfg)
        self.obs_map.reset(self.pos)
        self.esdf_builder = ObservedESDF(map_cfg)
        self.esdf_data = None
        self.known_mask = None

        # Macro expert
        from il_macro_expert import MacroExpert, MacroExpertConfig
        macro_cfg = MacroExpertConfig(
            enter_blocked_frames=3,
            exit_clear_frames=8,
            active_scan_yaw_rate_rps=2.0,
            active_scan_min_angle_deg=15.0,
            active_scan_max_duration_s=5.0,
            map_history_seconds=8.0,
            effective_guide_range_m=4.45,
        )
        macro_cfg.validate(5.0)
        self.macro = MacroExpert(macro_cfg)
        self.macro.set_guide_reachability_checker(self._check_reachability)

        # Diagnostics
        self.history = []
        self.last_seed_path = None
        self.last_frontier_candidates = []

    def _check_reachability(self, position, velocity, direction,
                            desired, minimum, step):
        """Bridge: macro → Python bounded A*."""
        if self.esdf_data is None:
            return 0.0
        for d in np.arange(desired, minimum - 0.01, -step):
            terminal = position + direction * d
            dist, path = py_bounded_astar(
                position, velocity, terminal,
                self.esdf_data, self.known_mask,
                self.obs_map.get_origin(), self.obs_map.get_resolution(),
                forbid_unknown=True,
                target_clearance=0.20, min_clearance=0.05,
                search_resolution=0.10,
                time_budget_ms=15.0)
            if path is not None and dist >= minimum:
                self.last_seed_path = path
                return max(0.0, min(float(dist), desired))
        return 0.0

    def _integrate_depth(self):
        """Cast depth rays and directly mark the 2-D grid — bypass the
        pinhole camera model which may misalign with the horizontal slice."""
        from il_observed_map import UNKNOWN, FREE, OCCUPIED
        depths, endpoints = self.camera.cast(self.pos, self.yaw, self.obstacles)
        occ = self.obs_map.get_occupancy(copy=False)
        origin = self.obs_map.get_origin()
        res = self.obs_map.get_resolution()
        gz = int((self.pos[2] - origin[2]) / res)
        gz = max(0, min(occ.shape[2] - 1, gz))
        changed = False

        for i, d in enumerate(depths):
            end_pt = endpoints[i]
            # Mark the ray from camera to endpoint as FREE
            n_steps = max(2, int(d / (0.5 * res)) + 1)
            for s in range(n_steps):
                frac = s / max(n_steps - 1, 1)
                pt = self.pos + frac * (end_pt - self.pos)
                gi = ((pt - origin) / res).astype(int)
                if (0 <= gi[0] < occ.shape[0] and
                        0 <= gi[1] < occ.shape[1]):
                    if occ[gi[0], gi[1], gz] == UNKNOWN:
                        occ[gi[0], gi[1], gz] = FREE
                        changed = True
            # Mark the endpoint as OCCUPIED (if within range)
            if d < self.camera.max_range - 0.05:
                gi = ((end_pt - origin) / res).astype(int)
                if (0 <= gi[0] < occ.shape[0] and
                        0 <= gi[1] < occ.shape[1]):
                    if occ[gi[0], gi[1], gz] != OCCUPIED:
                        occ[gi[0], gi[1], gz] = OCCUPIED
                        changed = True

        # Mark vehicle free bubble (same as production code)
        cx, cy, cz = (self.pos - origin) / res
        bubble_r = 0.30 + math.sqrt(3.0) * res
        r_vox = int(math.ceil(bubble_r / res))
        ix0, iy0 = int(math.floor(cx)), int(math.floor(cy))
        for dx in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                ix, iy = ix0 + dx, iy0 + dy
                if 0 <= ix < occ.shape[0] and 0 <= iy < occ.shape[1]:
                    vx, vy = float(ix) + 0.5, float(iy) + 0.5
                    dist2 = (vx - cx)**2 + (vy - cy)**2
                    if dist2 * res * res <= bubble_r * bubble_r + 1e-9:
                        if occ[ix, iy, gz] == UNKNOWN:
                            occ[ix, iy, gz] = FREE
                            changed = True

        # Copy the 2-D horizontal slice to enough z-levels that the 3-D
        # ESDF erosion (vehicle_radius = 0.30 m → 3 voxels) has vertical
        # support.  In production the 640×480 depth camera covers the full
        # vertical FOV naturally; the debug tool needs this workaround.
        for dz in range(-4, 5):
            z_copy = gz + dz
            if 0 <= z_copy < occ.shape[2]:
                occ[:, :, z_copy] = occ[:, :, gz].copy()

        if changed:
            self.obs_map._revision += 1

    def _build_esdf(self):
        occ = self.obs_map.get_occupancy(copy=False)
        self.esdf_builder.rebuild(
            occupancy=occ,
            known_mask=self.obs_map.get_known_mask(),
            origin_world=self.obs_map.get_origin(),
            resolution=self.obs_map.get_resolution())
        self.esdf_data = self.esdf_builder.get_esdf()
        self.known_mask = self.esdf_builder.get_known_mask()

    def step(self):
        """Advance one macro tick (0.2 s)."""
        dt = 0.2  # 5 Hz macro
        goal_dir_flu = self._goal_direction_flu()
        goal_dist = float(np.linalg.norm(self.goal - self.pos))
        if goal_dist < 0.30:
            return "goal_reached"

        # Integrate one depth frame per 30Hz frame (6 per macro tick)
        for _ in range(6):
            self._integrate_depth()
            self.t += 0.033
            self.frame += 1
        self._build_esdf()

        quat = np.array([0.0, 0.0, math.sin(0.5 * self.yaw),
                         math.cos(0.5 * self.yaw)], dtype=np.float64)
        depth_for_macro = np.full((48, 64), 5.0, dtype=np.float32)

        guide = self.macro.update(
            goal_direction_flu=goal_dir_flu,
            goal_distance_m=goal_dist,
            depth_m=depth_for_macro,
            observed_map=self.obs_map,
            current_position_world=self.pos,
            current_yaw=self.yaw,
            current_quaternion_xyzw=quat,
            current_velocity_world=self.vel,
            local_blocked=False,
            local_progress_rate=1.0,
            local_feasible=True,
            dt_since_last_macro=dt)

        # Execute guide
        macro_state = guide.macro_state
        move_dist = guide.move_distance_m

        if move_dist > 0.01:
            # Move toward guide direction, limited to realistic speed
            mv = guide.move_direction_flu
            yaw_dir = guide.yaw_direction_flu_xy
            # FLU: X=forward, Y=left.  Convention B (yaw=0 => nose faces
            # world +Y):  forward -> world (-sin yaw, cos yaw),
            # left -> world (-cos yaw, -sin yaw).
            cy = math.cos(self.yaw)
            sy = math.sin(self.yaw)
            move_dir_world = np.array([
                -sy * mv[0] - cy * mv[1],
                 cy * mv[0] - sy * mv[1],
                0.0
            ])
            # Limit per-tick movement to nominal speed * dt
            max_step = 1.8 * dt  # nominal_speed * dt
            actual_dist = min(move_dist, max_step)
            self.pos += move_dir_world * actual_dist
            # Track yaw toward intended direction (smooth turn).
            # Convention B: target yaw = world heading - pi/2.
            target_yaw = (math.atan2(move_dir_world[1], move_dir_world[0])
                          - math.pi / 2.0)
            yaw_err = self._wrap(target_yaw - self.yaw)
            self.yaw += max(-0.8, min(0.8, yaw_err))  # limited yaw rate
            self.vel = move_dir_world * actual_dist / dt
        elif macro_state == "PROBE":
            # Pure rotation — pan left then right (handled by macro guide
            # yaw); here we simply rotate in place.
            self.vel = np.zeros(3)
        elif macro_state == "GOAL_HOLD":
            self.vel = np.zeros(3)
            return "goal_hold"
        else:
            self.vel = np.zeros(3)

        # Record history
        self.history.append({
            "pos": self.pos.copy(),
            "yaw": self.yaw,
            "macro_state": macro_state,
            "move_dist": move_dist,
            "decision_reason": guide.decision_reason,
            "extracted_frontier": guide.extracted_frontier_candidate_count,
            "reachable_frontier": guide.reachable_frontier_candidate_count,
            "rejection_summary": guide.frontier_rejection_summary,
            "seed_path": (self.last_seed_path.copy()
                          if self.last_seed_path is not None else None),
        })

        if self.history[-1]["seed_path"] is not None:
            self.last_seed_path = None  # consume

        self.macro_tick += 1
        return "ok"

    def _goal_direction_flu(self):
        vec = self.goal - self.pos
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        world_dir = vec / dist
        # World -> FLU (Convention B: yaw=0 => nose faces world +Y):
        #   forward = -sin(yaw)*wx + cos(yaw)*wy
        #   left    = -cos(yaw)*wx - sin(yaw)*wy
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)
        flu_x = -sy * world_dir[0] + cy * world_dir[1]
        flu_y = -cy * world_dir[0] - sy * world_dir[1]
        flu = np.array([flu_x, flu_y, world_dir[2]], dtype=np.float64)
        norm = float(np.linalg.norm(flu))
        return flu / max(norm, 1e-9)

    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    def print_diagnostics(self):
        """Print a summary of the last few macro ticks."""
        recent = self.history[-10:]
        if not recent:
            return
        print("\n=== Macro Diagnostics (last {} ticks) ===".format(
            len(recent)))
        for h in recent[-6:]:
            pos = h["pos"]
            print("  pos=({:.2f},{:.2f}) yaw={:.0f}° state={} dist={:.2f}m "
                  "extracted={} reachable={} reject={} reason={}".format(
                      pos[0], pos[1], math.degrees(h["yaw"]),
                      h["macro_state"], h["move_dist"],
                      h["extracted_frontier"], h["reachable_frontier"],
                      h["rejection_summary"], h["decision_reason"]))


# ══════════════════════════════════════════════════════════════════════════
#  Matplotlib visualisation
# ══════════════════════════════════════════════════════════════════════════

def visualise(runner, title="2-D Planner Debug"):
    """Interactive matplotlib window."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyArrowPatch
    except ImportError:
        print("[WARN] matplotlib not available — visualisation disabled.")
        return None

    plt.ion()
    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(16, 7),
                                           gridspec_kw={"width_ratios": [3, 1]})
    fig.canvas.manager.set_window_title(title)

    class State:
        running = True
        step_mode = "manual"  # manual | auto

    state = State()

    def on_key(event):
        if event.key in (" ", "enter"):
            result = runner.step()
            runner.print_diagnostics()
            if result == "goal_reached":
                print("\n*** GOAL REACHED! ***")
                state.running = False
            update_plot()
        elif event.key == "r":
            state.step_mode = "auto"
            _auto_run()
        elif event.key in ("q", "escape"):
            state.running = False
            plt.close()
        elif event.key == "p":
            runner.print_diagnostics()

    def _auto_run():
        max_steps = 200
        for _ in range(max_steps):
            if not state.running or state.step_mode != "auto":
                break
            result = runner.step()
            if result == "goal_reached":
                print("\n*** GOAL REACHED! ***")
                state.running = False
                break
            if result == "goal_hold":
                print("\n*** GOAL HOLD ***")
                break
        runner.print_diagnostics()
        update_plot()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update_plot():
        ax_map.clear()
        ax_info.clear()

        # ── obstacles ──
        for obs in runner.obstacles:
            c = Circle((obs["cx"], obs["cy"]), obs["r"],
                       fc="gray", ec="black", alpha=0.6, zorder=3)
            ax_map.add_patch(c)

        # ── observed map ──
        occ = runner.obs_map.get_occupancy(copy=False)
        origin = runner.obs_map.get_origin()
        res = runner.obs_map.get_resolution()
        gx, gy, gz = occ.shape

        # Build a 2-D known/free/occupied image (horizontal slice at drone z)
        drone_gz = int((runner.pos[2] - origin[2]) / res)
        drone_gz = max(0, min(gz - 1, drone_gz))
        slice_2d = occ[:, :, drone_gz]
        from il_observed_map import UNKNOWN, FREE, OCCUPIED
        # known-free = green, occupied = red, unknown = dark
        rgba = np.zeros((gx, gy, 4), dtype=np.float64)
        rgba[slice_2d == FREE, :] = [0.2, 0.8, 0.2, 0.4]       # free → green
        rgba[slice_2d == OCCUPIED, :] = [0.8, 0.2, 0.2, 0.6]    # occupied → red
        rgba[slice_2d == UNKNOWN, :] = [0.15, 0.15, 0.15, 0.3]   # unknown → dark
        extent = [origin[0], origin[0] + gx * res,
                  origin[1], origin[1] + gy * res]
        ax_map.imshow(np.transpose(rgba, (1, 0, 2)),
                      origin="lower", extent=extent, zorder=1)

        # ── safe-known overlay ──
        if runner.known_mask is not None:
            km = runner.known_mask[:, :, drone_gz].astype(np.float64)
            km_rgba = np.zeros((gx, gy, 4))
            km_rgba[km > 0.5] = [0.0, 1.0, 0.0, 0.25]
            ax_map.imshow(np.transpose(km_rgba, (1, 0, 2)),
                          origin="lower", extent=extent, zorder=2)

        # ── start / goal ──
        ax_map.scatter(*runner.start[:2], c="cyan", s=120, marker="o",
                       edgecolors="black", zorder=5, label="Start")
        ax_map.scatter(*runner.goal[:2], c="gold", s=120, marker="*",
                       edgecolors="black", zorder=5, label="Goal")

        # ── drone ──
        ax_map.scatter(*runner.pos[:2], c="blue", s=80, marker="s",
                       zorder=6, label="Drone")
        # yaw indicator
        yl = 0.3
        ax_map.arrow(runner.pos[0], runner.pos[1],
                     math.cos(runner.yaw) * yl,
                     math.sin(runner.yaw) * yl,
                     head_width=0.08, head_length=0.12,
                     fc="blue", ec="blue", zorder=6)

        # ── trajectory history ──
        if len(runner.history) > 1:
            traj = np.array([h["pos"][:2] for h in runner.history])
            ax_map.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.0,
                        alpha=0.5, zorder=4)

        # ── last seed path ──
        last_seed = None
        for h in reversed(runner.history):
            if h.get("seed_path") is not None:
                last_seed = h["seed_path"]
                break
        if last_seed is not None:
            seed_pts = np.array(last_seed)
            ax_map.plot(seed_pts[:, 0], seed_pts[:, 1], "m-",
                        linewidth=2.0, alpha=0.9, zorder=7, label="A* seed")

        # ── guide direction ──
        if runner.history:
            h = runner.history[-1]
            if h["move_dist"] > 0.01:
                guide_dir = np.array([
                    math.cos(h["yaw"]),
                    math.sin(h["yaw"])
                ]) * h["move_dist"]
                ax_map.arrow(h["pos"][0], h["pos"][1],
                             guide_dir[0], guide_dir[1],
                             head_width=0.1, head_length=0.15,
                             fc="orange", ec="orange",
                             linewidth=2, zorder=8)

        ax_map.set_xlim(origin[0] - 0.5, origin[0] + gx * res + 0.5)
        ax_map.set_ylim(origin[1] - 0.5, origin[1] + gy * res + 0.5)
        ax_map.set_aspect("equal")
        ax_map.legend(loc="upper right", fontsize=7)
        ax_map.set_title("t={:.1f}s  frame={}  macro_tick={}".format(
            runner.t, runner.frame, runner.macro_tick))

        # ── info panel ──
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        lines = []
        lines.append("=== State ===")
        if runner.history:
            h = runner.history[-1]
            lines.append("state: {}".format(h["macro_state"]))
            lines.append("reason: {}".format(h["decision_reason"]))
            lines.append("move_dist: {:.2f}m".format(h["move_dist"]))
            lines.append("yaw: {:.0f}°".format(math.degrees(h["yaw"])))
        goal_dist = float(np.linalg.norm(runner.goal - runner.pos))
        lines.append("goal_dist: {:.2f}m".format(goal_dist))
        lines.append("")
        lines.append("=== Map ===")
        free = int(np.sum(runner.obs_map.get_occupancy(copy=False) == FREE))
        occ_v = int(np.sum(runner.obs_map.get_occupancy(copy=False) == OCCUPIED))
        unk = int(np.sum(runner.obs_map.get_occupancy(copy=False) == UNKNOWN))
        lines.append("free: {}  occ: {}  unk: {}".format(free, occ_v, unk))
        lines.append("")
        lines.append("=== Last 8 ticks ===")
        for h in runner.history[-8:]:
            lines.append("  {} dist={:.2f} extr={} reach={}".format(
                h["macro_state"][:12], h["move_dist"],
                h["extracted_frontier"], h["reachable_frontier"]))
        for i, line in enumerate(lines):
            ax_info.text(0.02, 0.98 - i * 0.027, line,
                         fontfamily="monospace", fontsize=8,
                         verticalalignment="top")

        fig.canvas.draw_idle()

    update_plot()
    print("\nControls: [space/enter]=step  [r]=auto-run  [p]=print diag  [q]=quit")
    print("Close the window to exit.")
    plt.show(block=True)


# ══════════════════════════════════════════════════════════════════════════
#  CLI entry
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="2-D Planner Debug")
    parser.add_argument("--scene", default="default",
                        choices=sorted(SCENES.keys()),
                        help="Scene to load")
    parser.add_argument("--no-viz", action="store_true",
                        help="Run headless, print diagnostics only")
    parser.add_argument("--steps", type=int, default=0,
                        help="Run N steps headless (0 = viz mode)")
    args = parser.parse_args()

    scene_fn = SCENES.get(args.scene)
    if scene_fn is None:
        print("Unknown scene: {}".format(args.scene))
        print("Available: {}".format(sorted(SCENES.keys())))
        return 1
    scene = scene_fn()
    print("Scene: {}".format(args.scene))
    print("  start: ({:.2f}, {:.2f})".format(*scene["start"][:2]))
    print("  goal:  ({:.2f}, {:.2f})".format(*scene["goal"][:2]))
    print("  obstacles: {}".format(len(scene["obstacles"])))

    runner = Debug2DRunner(scene)

    if args.steps > 0 or args.no_viz:
        for i in range(args.steps or 60):
            result = runner.step()
            if result != "ok":
                print("Step {}: {}".format(i + 1, result))
                break
            if (i + 1) % 10 == 0:
                runner.print_diagnostics()
        runner.print_diagnostics()
        return 0

    visualise(runner, title="2-D Planner Debug — {}".format(args.scene))
    return 0


if __name__ == "__main__":
    sys.exit(main())
