#!/usr/bin/env python3
"""
il_guide_selector.py  —  Farthest-visible A* waypoint selector for IL dataset v8.

Selects the farthest currently visible A* waypoint (Guide) and a dynamically
reachable Terminal. Guide is used for trend labels; Terminal is the trajectory
optimization target.

Key design:
  - Guide = farthest forward A* path index satisfying:
      range, FOV, and current-depth visibility constraints.
  - Terminal = farthest dynamically reachable path point not beyond Guide.
  - The optional known-free corridor check is independent and disabled in the
    global-ESDF production mode.
"""

from __future__ import print_function, division

import math
import numpy as np
from dataclasses import dataclass, field

from il_common import (world_vector_to_body_flu,
                       world_vector_to_body_flu_quat)
from il_observed_map import PinholeCameraModel


@dataclass
class GuideSelection:
    """Result of guide/terminal selection."""
    valid: bool = False

    # Guide (farthest visible A* waypoint) — for trend labels
    guide_position_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    guide_path_index: int = -1
    progress_path_index: int = -1

    # Terminal (dynamically reachable point) — for trajectory optimization
    terminal_position_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    terminal_path_index: int = -1
    terminal_distance_m: float = 0.0
    terminal_path_arc_length_m: float = 0.0

    # Guide direction in FLU (for trend labels)
    guide_direction_flu: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    guide_distance_m: float = 0.0
    guide_distance_norm: float = 0.0

    # Angular coordinates for soft labels
    azimuth_rad: float = 0.0
    elevation_rad: float = 0.0

    # Diagnostic fields
    candidate_count: int = 0
    visible: bool = False
    depth_visible: bool = False
    corridor_check_enabled: bool = False
    corridor_known_free_ratio: float = -1.0
    rejection_reason: str = ""


class GuideSelector:
    """Selects farthest visible A* waypoint as trend Guide.

    Also selects a dynamically reachable Terminal for trajectory optimization,
    which is not beyond the Guide and within planner horizon.
    """

    def __init__(self, config, camera_model):
        gs_cfg = config.get("global", {}).get("guide_selector", {})
        depth_cfg = config.get("global", {}).get("depth", {})

        self.max_range_m = min(
            float(gs_cfg.get("max_range_m", 5.0)),
            float(depth_cfg.get("max_m", 5.0)))
        self.h_fov_margin_deg = float(gs_cfg.get("horizontal_fov_margin_deg", 3.0))
        self.v_fov_margin_deg = float(gs_cfg.get("vertical_fov_margin_deg", 3.0))
        self.h_fov_margin_rad = math.radians(self.h_fov_margin_deg)
        self.v_fov_margin_rad = math.radians(self.v_fov_margin_deg)

        self.depth_visibility_tol_m = float(
            gs_cfg.get("depth_visibility_tolerance_m", 0.20))
        self.corridor_radius_m = float(gs_cfg.get("corridor_radius_m", 0.35))
        self.corridor_sample_spacing_m = float(
            gs_cfg.get("corridor_sample_spacing_m", 0.10))
        self.min_clearance_m = float(gs_cfg.get("min_clearance_m", 0.35))
        self.path_search_max_points = int(gs_cfg.get("path_search_max_points", 200))

        self.terminal_horizon_s = float(gs_cfg.get("terminal_horizon_s", 1.5))
        self.terminal_accel_limit = float(
            gs_cfg.get("terminal_acceleration_limit_mps2", 3.0))
        self.terminal_min_distance_m = float(
            gs_cfg.get("terminal_min_distance_m", 0.40))

        self.use_depth_projection = bool(
            gs_cfg.get("use_depth_projection_visibility", True))
        self.use_known_free_corridor = bool(
            gs_cfg.get("use_known_free_corridor", False))

        self._camera = camera_model

    def reset(self):
        """Reset selector state for new episode (currently no persistent state)."""
        pass

    def select(self, global_path_world, previous_progress_index,
               current_position_world, current_yaw,
               current_velocity_world, depth_m,
               observed_map=None, observed_esdf=None,
               current_quaternion_xyzw=None):
        """Select Guide and Terminal from global A* path.

        Args:
            global_path_world: list of [x, y, z] waypoints.
            previous_progress_index: last progress index (negative = first call).
            current_position_world: [x, y, z].
            current_yaw: radians.
            current_velocity_world: [vx, vy, vz].
            depth_m: (H, W) depth image.
            observed_map: optional RollingObservedOccupancyMap, required only
                when known-free corridor checking is enabled.
            observed_esdf: optional ObservedESDF (reserved for corridor modes).

        Returns:
            GuideSelection with guide and terminal positions.
        """
        sel = GuideSelection()

        if not global_path_world or len(global_path_world) < 2:
            sel.rejection_reason = "empty_global_path"
            return sel
        if self.use_depth_projection and depth_m is None:
            sel.rejection_reason = "missing_depth"
            return sel
        if self.use_known_free_corridor and observed_map is None:
            sel.rejection_reason = "known_free_corridor_requires_observed_map"
            return sel

        sel.corridor_check_enabled = self.use_known_free_corridor

        pos = np.asarray(current_position_world, dtype=np.float64)
        vel = np.asarray(current_velocity_world, dtype=np.float64)

        # ── Find nearest forward path index ─────────────────────
        start_index = self._find_forward_start_index(
            global_path_world, previous_progress_index, pos)
        sel.progress_path_index = start_index

        # ── Compute camera FOV limits ──────────────────────────
        h_half = self._camera.hfov / 2.0 - self.h_fov_margin_rad
        v_half = self._camera.vfov / 2.0 - self.v_fov_margin_rad

        # ── Search forward for farthest visible waypoint ───────
        candidate_count = 0
        best_guide_idx = -1

        candidate_begin = min(start_index + 1, len(global_path_world) - 1)
        for i in range(candidate_begin, min(len(global_path_world),
                                             start_index + self.path_search_max_points)):
            pt = np.array(global_path_world[i], dtype=np.float64)

            # 1. Range check
            dist = float(np.linalg.norm(pt - pos))
            if dist > self.max_range_m:
                continue

            # 2. Camera forward check (FLU)
            delta_world = pt - pos
            delta_flu = (world_vector_to_body_flu_quat(
                delta_world, current_quaternion_xyzw)
                if current_quaternion_xyzw is not None else
                world_vector_to_body_flu(delta_world, float(current_yaw)))
            if delta_flu[0] <= 0:  # not in front
                continue

            # 3. FOV check
            norm = float(np.linalg.norm(delta_flu))
            if norm < 1e-9:
                continue
            azimuth = math.atan2(delta_flu[1], delta_flu[0])
            elevation = math.atan2(delta_flu[2],
                                   math.sqrt(delta_flu[0]**2 + delta_flu[1]**2))

            if abs(azimuth) > h_half or abs(elevation) > v_half:
                continue

            candidate_count += 1
            visible = True

            # 4. Depth visibility check
            depth_visible = True
            if self.use_depth_projection and depth_m is not None:
                depth_visible = self._check_depth_visibility(
                    pt, delta_flu, depth_m, dist)
            if not depth_visible:
                visible = False
                continue

            # 5. Known-free corridor check
            if self.use_known_free_corridor:
                corridor_ratio = observed_map.sample_known_free_ratio_along_corridor(
                    pos, pt, self.corridor_radius_m,
                    self.corridor_sample_spacing_m,
                    self.min_clearance_m)
                min_ratio = getattr(observed_map, 'min_known_free_ratio', 0.95)
                if corridor_ratio < min_ratio:
                    visible = False
                    continue

            # All checks passed
            best_guide_idx = i

        if best_guide_idx < 0:
            sel.rejection_reason = "no_visible_waypoint"
            sel.candidate_count = candidate_count
            return sel

        sel.candidate_count = candidate_count

        # ── Populate Guide ──────────────────────────────────────
        guide_pt = np.array(global_path_world[best_guide_idx], dtype=np.float64)
        sel.guide_position_world = guide_pt
        sel.guide_path_index = best_guide_idx
        sel.visible = True

        delta_w = guide_pt - pos
        guide_dist = float(np.linalg.norm(delta_w))
        sel.guide_distance_m = guide_dist
        sel.guide_distance_norm = min(guide_dist, self.max_range_m) / max(self.max_range_m, 1e-9)

        delta_flu = (world_vector_to_body_flu_quat(
            delta_w, current_quaternion_xyzw)
            if current_quaternion_xyzw is not None else
            world_vector_to_body_flu(delta_w, float(current_yaw)))
        norm_f = float(np.linalg.norm(delta_flu))
        if norm_f > 1e-9:
            sel.guide_direction_flu = delta_flu / norm_f
        sel.azimuth_rad = math.atan2(float(delta_flu[1]), float(delta_flu[0]))
        sel.elevation_rad = math.atan2(
            float(delta_flu[2]),
            math.sqrt(float(delta_flu[0])**2 + float(delta_flu[1])**2))

        sel.depth_visible = True

        if self.use_known_free_corridor:
            sel.corridor_known_free_ratio = observed_map.sample_known_free_ratio_along_corridor(
                pos, guide_pt, self.corridor_radius_m,
                self.corridor_sample_spacing_m, self.min_clearance_m)
        else:
            sel.corridor_known_free_ratio = -1.0

        # ── Select Terminal (dynamically reachable) ─────────────
        speed = float(np.linalg.norm(vel))
        L_max = speed * self.terminal_horizon_s + \
            0.5 * self.terminal_accel_limit * self.terminal_horizon_s**2

        # Walk from start_index to guide_index, accumulate arc length
        term_idx = start_index
        arc_len = 0.0
        for i in range(start_index, best_guide_idx):
            seg = np.linalg.norm(
                np.array(global_path_world[i + 1]) -
                np.array(global_path_world[i]))
            if arc_len + seg > L_max:
                break
            arc_len += seg
            term_idx = i + 1

        # Ensure minimum distance
        if term_idx == start_index:
            term_idx = min(start_index + 1, best_guide_idx)

        term_pt = np.array(global_path_world[term_idx], dtype=np.float64)
        term_dist = float(np.linalg.norm(term_pt - pos))
        if term_dist < self.terminal_min_distance_m:
            # Push forward to min distance
            for i in range(term_idx, best_guide_idx + 1):
                pt_i = np.array(global_path_world[i], dtype=np.float64)
                if float(np.linalg.norm(pt_i - pos)) >= self.terminal_min_distance_m:
                    term_idx = i
                    term_pt = pt_i
                    break

        # Record the actual A* arc length after any minimum-distance push.
        terminal_arc_len = 0.0
        for i in range(start_index, term_idx):
            terminal_arc_len += float(np.linalg.norm(
                np.asarray(global_path_world[i + 1], dtype=np.float64) -
                np.asarray(global_path_world[i], dtype=np.float64)))
        term_dist = float(np.linalg.norm(term_pt - pos))

        sel.terminal_position_world = term_pt
        sel.terminal_path_index = term_idx
        sel.terminal_distance_m = term_dist
        sel.terminal_path_arc_length_m = terminal_arc_len
        sel.valid = True

        return sel

    def _find_forward_start_index(self, global_path, prev_progress_idx, pos):
        """Find the nearest forward path index to start searching from."""
        if prev_progress_idx < 0:
            # First call: find nearest point
            best_idx = 0
            best_dist = float('inf')
            for i, pt in enumerate(global_path):
                d = float(np.linalg.norm(np.array(pt) - pos))
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            return best_idx

        # Re-localise in a bounded forward window without ever regressing.
        begin = max(0, int(prev_progress_idx))
        end = min(len(global_path), begin + self.path_search_max_points)
        best_idx = begin
        best_dist = float('inf')
        for i in range(begin, end):
            d = float(np.linalg.norm(np.asarray(global_path[i]) - pos))
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _check_depth_visibility(self, point_world, delta_flu, depth_m, dist):
        """Check if a world point is visible in the current depth image.

        Uses FLU direction to the point and projects it to image coordinates.
        Then compares the point's distance to the depth at that pixel.

        Args:
            point_world: [x, y, z] world position.
            delta_flu: FLU vector from camera to point.
            depth_m: (H, W) depth image in metres.
            dist: Euclidean distance to point.
        Returns:
            True if point is depth-visible (not occluded).
        """
        # FLU [forward, left, up] -> Camera [right, down, forward]
        # cam_x = -flu_y (left -> right negates)
        # cam_y = -flu_z (up -> down negates)
        # cam_z = flu_x (forward)

        if depth_m is None:
            return True

        flu = np.asarray(delta_flu, dtype=np.float64)
        cam_x = -flu[1]
        cam_y = -flu[2]
        cam_z = flu[0]

        if cam_z <= 1e-6:
            return False

        # Project to pixel
        u = (cam_x / cam_z) * self._camera.fx + self._camera.cx
        v = (cam_y / cam_z) * self._camera.fy + self._camera.cy

        h, w = depth_m.shape
        ui = int(round(u))
        vi = int(round(v))

        # Check 3x3 neighbourhood for robust depth
        r_min = max(0, vi - 1)
        r_max = min(h - 1, vi + 1)
        c_min = max(0, ui - 1)
        c_max = min(w - 1, ui + 1)

        if r_min > r_max or c_min > c_max:
            return False

        patch = depth_m[r_min:r_max + 1, c_min:c_max + 1]
        if patch.size == 0:
            return False

        # Use max depth in neighbourhood (conservative occlusion check)
        max_depth = float(np.max(patch[np.isfinite(patch)])) if np.any(np.isfinite(patch)) else 0.0
        if max_depth <= 0:
            return False

        # Point is visible if closer than or equal to depth + tolerance
        # Flightmare depth is camera-axis (z) depth, not Euclidean range.
        return cam_z <= max_depth + self.depth_visibility_tol_m

    def get_reference_segment(self, global_path_world, start_index, end_index):
        """Extract A* sub-path from start_index to end_index for planner init.

        Returns:
            List of [x, y, z] waypoints, at most path_search_max_points.
        """
        seg = global_path_world[start_index:end_index + 1]
        if len(seg) > self.path_search_max_points:
            # Subsample evenly
            keep = self.path_search_max_points
            indices = np.linspace(0, len(seg) - 1, keep, dtype=int)
            seg = [seg[i] for i in indices]
        return seg
