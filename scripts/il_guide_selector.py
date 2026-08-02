#!/usr/bin/env python3
"""Current-observation local viewpoint selector for IL collection.

The selector derives every Guide from only the current depth frame and the
relative mission goal.  It deliberately does not read future global-path
geometry, preventing stacked sensing horizons in Trend supervision.
"""

from __future__ import print_function, division

import math
import numpy as np
from dataclasses import dataclass, field

from il_common import (body_flu_to_rfu,
                       body_flu_to_world_quat,
                       body_vel_to_world,
                       world_vector_to_body_flu,
                       world_vector_to_body_flu_quat)
@dataclass
class GuideSelection:
    """Result of guide/terminal selection."""
    valid: bool = False

    # Guide (farthest visible A* waypoint) — for trend labels
    guide_position_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    guide_path_index: int = -1
    progress_path_index: int = -1

    # Terminal aliases Guide for trajectory optimization.
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
    guide_is_final: bool = False

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
    selection_mode: str = ""
    recovery_target_valid: bool = False
    recovery_target_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    avoidance_side: int = 0


class GuideSelector:
    """Select a causal local Guide from the current depth observation."""

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

        self.corridor_radius_m = float(gs_cfg.get("corridor_radius_m", 0.35))
        self.path_search_max_points = int(gs_cfg.get("path_search_max_points", 200))

        self.selection_mode = str(
            gs_cfg.get("selection_mode", "local_goal_explorer")).strip().lower()
        if self.selection_mode != "local_goal_explorer":
            raise ValueError(
                "guide_selector.selection_mode must be local_goal_explorer, "
                "got {!r}".format(self.selection_mode))
        self.explorer_angle_step_rad = math.radians(float(
            gs_cfg.get("explorer_angle_step_deg", 3.0)))
        self.explorer_refine_step_rad = math.radians(float(
            gs_cfg.get("explorer_refine_step_deg", 1.0)))
        self.explorer_refine_half_width_rad = math.radians(float(
            gs_cfg.get("explorer_refine_half_width_deg", 3.0)))
        self.explorer_max_angle_rad = math.radians(float(
            gs_cfg.get("explorer_max_angle_deg", 55.0)))
        self.explorer_min_advance_m = float(
            gs_cfg.get("explorer_min_advance_m", 1.0))
        self.explorer_escape_min_advance_m = float(
            gs_cfg.get("explorer_escape_min_advance_m", 0.40))
        self.explorer_max_clearance_drop_m = float(
            gs_cfg.get("explorer_max_clearance_drop_m", 0.02))
        self.explorer_recovery_turn_rad = math.radians(float(
            gs_cfg.get("explorer_recovery_turn_deg", 25.0)))
        self.explorer_depth_step = max(
            1, int(gs_cfg.get("explorer_depth_step", 6)))
        self.explorer_obstacle_margin_m = float(
            gs_cfg.get("explorer_obstacle_margin_m", 0.05))
        self.depth_max_m = float(depth_cfg.get("max_m", self.max_range_m))
        esdf_cfg = config.get("global", {}).get("esdf", {})
        local_planner_cfg = config.get("global", {}).get(
            "planning", {}).get("local_planner", {})
        self.vehicle_radius_m = float(esdf_cfg.get("drone_radius", 0.30))
        self.local_target_clearance_m = float(
            local_planner_cfg.get("target_clearance", 0.20))
        self.esdf_resolution_m = float(esdf_cfg.get("resolution", 0.10))
        # A viewpoint is the vehicle centre, not a point ray.  Its complete
        # clearance ball must remain inside the measured depth range;
        # otherwise an obstacle just beyond max range can already collide
        # with the vehicle at the nominal Guide.
        self.explorer_required_radius_m = max(
            self.corridor_radius_m,
            self.vehicle_radius_m + self.local_target_clearance_m +
            0.5 * self.esdf_resolution_m)
        self.explorer_esdf_validation_clearance_m = (
            self.local_target_clearance_m +
            0.5 * self.esdf_resolution_m)
        self.explorer_usable_range_m = max(
            self.explorer_min_advance_m,
            min(self.max_range_m,
                self.depth_max_m - self.explorer_required_radius_m))

        self._camera = camera_model
        self._last_guide_path_index = -1
        self._safety_esdf = None
        self._safety_esdf_origin = None
        self._safety_esdf_resolution = None
        self._avoidance_side = 0
        self._last_escape_direction_world = None

    def reset(self):
        """Reset monotonic Guide state for a new episode."""
        self._last_guide_path_index = -1
        self._avoidance_side = 0
        self._last_escape_direction_world = None

    def set_safety_esdf(self, esdf, origin_world, resolution):
        """Attach the simulator ESDF as an in-frustum consistency checker.

        The map is queried only along candidate corridors whose complete
        swept-radius volume lies inside the current depth range.  It corrects
        depth discretisation/surface errors; it does not enlarge the causal
        sensing horizon.
        """
        self._safety_esdf = np.asarray(esdf, dtype=np.float32)
        self._safety_esdf_origin = np.asarray(
            origin_world, dtype=np.float64).reshape(3)
        self._safety_esdf_resolution = float(resolution)

    def select(self, global_path_world, previous_progress_index,
               current_position_world, current_yaw,
               current_velocity_world, depth_m,
               observed_map=None, observed_esdf=None,
               current_quaternion_xyzw=None,
               goal_position_world=None):
        """Select one Guide/Terminal from the current observation.

        Args:
            global_path_world: list of [x, y, z] waypoints.
            previous_progress_index: last progress index (negative = first call).
            current_position_world: [x, y, z].
            current_yaw: radians.
            current_velocity_world: [vx, vy, vz].
            depth_m: (H, W) depth image.
            observed_map: accepted for the manager's stable call interface.
            observed_esdf: accepted for the manager's stable call interface.

        Returns:
            GuideSelection with guide and terminal positions.
        """
        return self._select_local_goal_explorer(
            global_path_world, previous_progress_index,
            current_position_world, current_yaw, depth_m,
            current_quaternion_xyzw, goal_position_world)

    def _select_local_goal_explorer(
            self, global_path_world, previous_progress_index,
            current_position_world, current_yaw, depth_m,
            current_quaternion_xyzw, goal_position_world):
        """Choose a causal local viewpoint from this depth frame and the goal.

        The global path is used only to attach a monotonic diagnostic index;
        none of its future waypoint geometry participates in viewpoint
        selection.
        """
        sel = GuideSelection(selection_mode="local_goal_explorer")
        if goal_position_world is None:
            sel.rejection_reason = "missing_goal"
            return sel
        if depth_m is None:
            sel.rejection_reason = "missing_depth"
            return sel

        pos = np.asarray(current_position_world, dtype=np.float64)
        goal = np.asarray(goal_position_world, dtype=np.float64)
        goal_delta_world = goal - pos
        goal_distance = float(np.linalg.norm(goal_delta_world))
        if not np.isfinite(goal_distance) or goal_distance < 1.0e-6:
            sel.rejection_reason = "goal_reached"
            return sel

        goal_flu = (
            world_vector_to_body_flu_quat(
                goal_delta_world, current_quaternion_xyzw)
            if current_quaternion_xyzw is not None else
            world_vector_to_body_flu(goal_delta_world, float(current_yaw)))
        desired_azimuth = math.atan2(
            float(goal_flu[1]), float(goal_flu[0]))
        desired_elevation = math.atan2(
            float(goal_flu[2]),
            math.sqrt(float(goal_flu[0]) ** 2 +
                      float(goal_flu[1]) ** 2))

        h_half = self._camera.hfov / 2.0 - self.h_fov_margin_rad
        v_half = self._camera.vfov / 2.0 - self.v_fov_margin_rad
        goal_in_fov = (
            goal_flu[0] > 0.0 and abs(desired_azimuth) <= h_half)
        # An avoidance-side commitment chooses between obstacle branches; it
        # must never grant permission to translate while the mission goal is
        # outside the current camera FOV.  In that case the manager rotates
        # toward the goal, while this selector preserves the commitment for
        # use after the goal direction becomes observable again.
        if not goal_in_fov:
            sel.rejection_reason = "goal_direction_outside_fov"
            sel.avoidance_side = self._avoidance_side
            return sel
        if abs(desired_elevation) > v_half:
            sel.rejection_reason = "goal_elevation_outside_fov"
            return sel

        target_distance = min(
            goal_distance, self.explorer_usable_range_m)
        step = max(math.radians(1.0), self.explorer_angle_step_rad)

        # Scan the complete current FOV.  Restricting candidates to a fixed
        # deviation around the goal creates a deadlock when the escape tangent
        # lies just outside that cone.  Goal deviation is a ranking criterion,
        # not a candidate-generation boundary.
        azimuths = list(np.arange(
            -h_half, h_half + 0.5 * step, step, dtype=np.float64))
        refine_anchors = [-h_half, h_half]
        if goal_in_fov:
            refine_anchors.append(float(desired_azimuth))
        if self._avoidance_side != 0:
            refine_anchors.append(self._previous_direction_azimuth(
                current_yaw, current_quaternion_xyzw))
        refine_step = max(
            math.radians(0.5), self.explorer_refine_step_rad)
        refine_half_width = max(0.0, self.explorer_refine_half_width_rad)
        for anchor in refine_anchors:
            azimuths.extend(np.arange(
                anchor - refine_half_width,
                anchor + refine_half_width + 0.5 * refine_step,
                refine_step, dtype=np.float64))
        azimuths = sorted(set(round(float(a), 10) for a in azimuths
                              if -h_half <= a <= h_half))
        if not azimuths:
            sel.rejection_reason = "no_explorer_candidate_in_fov"
            return sel

        elevation = desired_elevation
        units = np.asarray([
            [math.cos(elevation) * math.cos(azimuth),
             math.cos(elevation) * math.sin(azimuth),
             math.sin(elevation)]
            for azimuth in azimuths
        ], dtype=np.float64)
        safe_ranges = self._explorer_safe_ranges(
            depth_m, units, target_distance)
        if self._safety_esdf is not None:
            esdf_safe_ranges = self._explorer_esdf_safe_ranges(
                pos, units, target_distance, current_yaw,
                current_quaternion_xyzw)
            safe_ranges = np.minimum(safe_ranges, esdf_safe_ranges)
        sel.candidate_count = len(azimuths)

        full_threshold = max(
            0.0, target_distance - self.explorer_obstacle_margin_m)
        full_indices = [
            i for i, safe_range in enumerate(safe_ranges)
            if safe_range >= full_threshold]
        offsets = np.asarray([
            self._wrap_angle(azimuth - desired_azimuth)
            for azimuth in azimuths], dtype=np.float64)
        direct_index = (
            min(range(len(azimuths)),
                key=lambda i: abs(azimuths[i] - desired_azimuth))
            if goal_in_fov else -1)
        direct_clear = (
            direct_index >= 0 and
            abs(azimuths[direct_index] - desired_azimuth) < 1.0e-6 and
            safe_ranges[direct_index] >= full_threshold)

        chosen_index = -1
        guide_distance = target_distance
        direct_selected = False

        if direct_clear:
            chosen_index = direct_index
            direct_selected = True
            self._avoidance_side = 0
            self._last_escape_direction_world = None
        else:
            if self._avoidance_side == 0:
                # First obstacle decision: minimum goal deviation, exact ties
                # choose RIGHT (negative FLU azimuth offset).
                side_candidates = [
                    i for i in full_indices if abs(offsets[i]) > 1.0e-6]
                if side_candidates:
                    chosen_index = min(
                        side_candidates,
                        key=lambda i: (
                            abs(offsets[i]),
                            0 if offsets[i] < 0.0 else 1))
                    self._avoidance_side = (
                        -1 if offsets[chosen_index] < 0.0 else 1)
            else:
                committed = [
                    i for i in full_indices
                    if self._avoidance_side * offsets[i] > 1.0e-6]
                if committed:
                    previous_azimuth = self._previous_direction_azimuth(
                        current_yaw, current_quaternion_xyzw)
                    chosen_index = min(
                        committed,
                        key=lambda i: abs(self._wrap_angle(
                            azimuths[i] - previous_azimuth)))

            if chosen_index < 0 and self._avoidance_side != 0:
                # A shortened escape step is allowed only when it does not
                # trade away the current ESDF clearance.  This prevents the
                # old 1.28 -> 1.15 -> 1.02 m shrinking-guide deadlock.
                committed = [
                    i for i in range(len(azimuths))
                    if self._avoidance_side * offsets[i] > 1.0e-6]
                escape = self._select_clearance_preserving_escape(
                    committed, units, safe_ranges, target_distance, pos,
                    current_yaw, current_quaternion_xyzw, azimuths)
                if escape is not None:
                    chosen_index, guide_distance = escape

            if chosen_index < 0:
                sel.rejection_reason = "no_safe_local_viewpoint"
                sel.avoidance_side = self._avoidance_side
                self._populate_explore_recovery(
                    sel, pos, current_yaw, current_quaternion_xyzw)
                return sel

        chosen_unit = units[chosen_index]
        chosen_flu = chosen_unit * guide_distance
        chosen_world_unit = self._unit_flu_to_world(
            chosen_unit, current_yaw, current_quaternion_xyzw)
        chosen_world_delta = chosen_world_unit * guide_distance
        # Final invariant against a stale commitment or frame-conversion
        # regression: a translational Guide must make strictly positive
        # progress along the current position-to-goal vector.  Recovery toward
        # the mission goal is safer than emitting a label that flies away.
        if float(np.dot(chosen_world_delta, goal_delta_world)) <= 1.0e-9:
            sel.rejection_reason = "non_goal_advancing_viewpoint"
            sel.avoidance_side = self._avoidance_side
            return sel
        guide_pt = pos + chosen_world_delta
        if not direct_selected:
            self._last_escape_direction_world = chosen_world_unit.copy()

        if global_path_world:
            start_index = self._find_forward_start_index(
                global_path_world, previous_progress_index, pos)
            is_final = (
                goal_distance <= self.explorer_usable_range_m + 1.0e-6 and
                direct_selected and
                guide_distance >= goal_distance -
                self.explorer_obstacle_margin_m)
            guide_index = (
                len(global_path_world) - 1 if is_final else
                min(start_index + 1, max(0, len(global_path_world) - 2)))
        else:
            start_index = max(0, int(previous_progress_index))
            is_final = (
                goal_distance <= self.explorer_usable_range_m + 1.0e-6 and
                direct_selected)
            guide_index = start_index

        delta_flu = chosen_flu
        sel.valid = True
        sel.visible = True
        sel.depth_visible = True
        sel.progress_path_index = start_index
        sel.guide_position_world = guide_pt
        sel.guide_path_index = guide_index
        sel.guide_is_final = is_final
        sel.avoidance_side = self._avoidance_side
        sel.guide_direction_flu = chosen_unit
        sel.guide_distance_m = guide_distance
        sel.guide_distance_norm = (
            guide_distance / max(self.explorer_usable_range_m, 1.0e-9))
        sel.azimuth_rad = float(azimuths[chosen_index])
        sel.elevation_rad = float(elevation)
        sel.terminal_position_world = guide_pt.copy()
        sel.terminal_path_index = guide_index
        sel.terminal_distance_m = guide_distance
        sel.terminal_path_arc_length_m = guide_distance
        sel.corridor_check_enabled = True
        sel.corridor_known_free_ratio = 1.0
        return sel

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))

    def _unit_flu_to_world(
            self, unit_flu, current_yaw, current_quaternion_xyzw):
        if current_quaternion_xyzw is not None:
            return body_flu_to_world_quat(
                unit_flu, current_quaternion_xyzw)
        unit_rfu = body_flu_to_rfu(unit_flu)
        return np.asarray(body_vel_to_world(
            float(unit_rfu[0]), float(unit_rfu[1]),
            float(unit_rfu[2]), float(current_yaw)),
            dtype=np.float64)

    def _previous_direction_azimuth(
            self, current_yaw, current_quaternion_xyzw):
        if self._last_escape_direction_world is None:
            return 0.0
        direction_flu = (
            world_vector_to_body_flu_quat(
                self._last_escape_direction_world,
                current_quaternion_xyzw)
            if current_quaternion_xyzw is not None else
            world_vector_to_body_flu(
                self._last_escape_direction_world, float(current_yaw)))
        return math.atan2(
            float(direction_flu[1]), float(direction_flu[0]))

    def _select_clearance_preserving_escape(
            self, candidate_indices, units, safe_ranges, target_distance,
            current_position_world, current_yaw, current_quaternion_xyzw,
            azimuths):
        if not candidate_indices:
            return None

        start_clearance = None
        if self._safety_esdf is not None:
            start_clearance = float(self._sample_safety_esdf(
                np.asarray(current_position_world, dtype=np.float64)
                .reshape(1, 3))[0])
        previous_azimuth = self._previous_direction_azimuth(
            current_yaw, current_quaternion_xyzw)
        eligible = []
        for index in candidate_indices:
            distance = min(
                target_distance,
                max(0.0, float(safe_ranges[index]) -
                    self.explorer_obstacle_margin_m))
            if distance < self.explorer_escape_min_advance_m:
                continue

            endpoint_clearance = float("inf")
            if self._safety_esdf is not None:
                unit_world = self._unit_flu_to_world(
                    units[index], current_yaw, current_quaternion_xyzw)
                endpoint = (
                    np.asarray(current_position_world, dtype=np.float64) +
                    distance * unit_world)
                endpoint_clearance = float(self._sample_safety_esdf(
                    endpoint.reshape(1, 3))[0])
                clearance_floor = max(
                    self.local_target_clearance_m,
                    start_clearance -
                    self.explorer_max_clearance_drop_m)
                if endpoint_clearance < clearance_floor:
                    continue
            eligible.append((
                index, distance, endpoint_clearance,
                abs(self._wrap_angle(
                    azimuths[index] - previous_azimuth))))

        if not eligible:
            return None
        best = min(
            eligible,
            key=lambda item: (
                -item[2], item[3], -item[1]))
        return int(best[0]), float(best[1])

    def _populate_explore_recovery(
            self, selection, current_position_world, current_yaw,
            current_quaternion_xyzw):
        if self._avoidance_side == 0:
            return
        azimuth = (
            self._avoidance_side * self.explorer_recovery_turn_rad)
        unit_flu = np.asarray(
            [math.cos(azimuth), math.sin(azimuth), 0.0],
            dtype=np.float64)
        unit_world = self._unit_flu_to_world(
            unit_flu, current_yaw, current_quaternion_xyzw)
        selection.recovery_target_valid = True
        selection.recovery_target_world = (
            np.asarray(current_position_world, dtype=np.float64) +
            2.0 * unit_world)

    def _explorer_safe_ranges(self, depth_m, candidate_units_flu,
                              target_distance):
        """Return collision-free range of each ray from current depth only."""
        points_cam, _ = self._camera.backproject_depth(
            depth_m, step=self.explorer_depth_step)
        if len(points_cam) == 0:
            return np.zeros(len(candidate_units_flu), dtype=np.float64)

        # Max-range pixels represent unknown/background rather than obstacle
        # surfaces.  Only finite returns before the sensor limit constrain the
        # viewpoint rays.
        hit_mask = (
            np.isfinite(points_cam[:, 2]) &
            (points_cam[:, 2] > 0.0) &
            (points_cam[:, 2] <
             self.depth_max_m - self.explorer_obstacle_margin_m))
        points_cam = points_cam[hit_mask]
        if len(points_cam) == 0:
            return np.full(
                len(candidate_units_flu), target_distance, dtype=np.float64)

        points_flu = self._camera.cam_to_flu(points_cam)
        units = np.asarray(candidate_units_flu, dtype=np.float64)
        projection = points_flu.dot(units.T)
        point_norm_sq = np.sum(points_flu * points_flu, axis=1)[:, None]
        perpendicular_sq = np.maximum(
            0.0, point_norm_sq - projection * projection)

        # Inflate slightly for depth subsampling.  This is a local observed
        # corridor check, not an ESDF/global-map query.
        radius = self.explorer_required_radius_m
        collision = (
            (projection > 0.0) &
            (projection <= target_distance + radius) &
            (perpendicular_sq <= radius * radius))
        longitudinal_half = np.sqrt(np.maximum(
            0.0, radius * radius - perpendicular_sq))
        entry_range = projection - longitudinal_half
        entry_range = np.where(collision, entry_range, np.inf)
        safe_ranges = np.min(entry_range, axis=0)
        safe_ranges = np.where(
            np.isfinite(safe_ranges),
            np.maximum(0.0, safe_ranges),
            target_distance)
        return np.minimum(safe_ranges, target_distance)

    def _explorer_esdf_safe_ranges(
            self, current_position_world, candidate_units_flu,
            target_distance, current_yaw, current_quaternion_xyzw):
        """Validate observed candidate corridors with planner ESDF semantics."""
        spacing = max(
            0.02, min(0.05, 0.5 * self._safety_esdf_resolution))
        sample_distances = np.arange(
            spacing, target_distance + 0.5 * spacing, spacing,
            dtype=np.float64)
        if len(sample_distances) == 0:
            sample_distances = np.asarray(
                [target_distance], dtype=np.float64)

        pos = np.asarray(current_position_world, dtype=np.float64)
        units_world = []
        for unit_flu in candidate_units_flu:
            if current_quaternion_xyzw is not None:
                unit_world = body_flu_to_world_quat(
                    unit_flu, current_quaternion_xyzw)
            else:
                unit_rfu = body_flu_to_rfu(unit_flu)
                unit_world = np.asarray(body_vel_to_world(
                    float(unit_rfu[0]), float(unit_rfu[1]),
                    float(unit_rfu[2]), float(current_yaw)),
                    dtype=np.float64)
            units_world.append(unit_world)

        units_world = np.asarray(units_world, dtype=np.float64)
        points_world = (
            pos[None, None, :] +
            units_world[:, None, :] *
            sample_distances[None, :, None])
        clearances = self._sample_safety_esdf(
            points_world.reshape(-1, 3)).reshape(
                len(units_world), len(sample_distances))
        unsafe = (
            clearances <
            self.explorer_esdf_validation_clearance_m)
        has_unsafe = np.any(unsafe, axis=1)
        first_unsafe = np.argmax(unsafe, axis=1)
        safe_ranges = np.full(
            len(units_world), target_distance, dtype=np.float64)
        rows = np.flatnonzero(has_unsafe)
        if len(rows) > 0:
            first = first_unsafe[rows]
            previous = np.maximum(0, first - 1)
            safe_ranges[rows] = sample_distances[previous]
            safe_ranges[rows[first == 0]] = 0.0
        return safe_ranges

    def _sample_safety_esdf(self, points_world):
        """Vectorised trilinear ESDF sampling; outside the map is unsafe."""
        points = np.asarray(points_world, dtype=np.float64)
        grid = (
            (points - self._safety_esdf_origin[None, :]) /
            self._safety_esdf_resolution)
        base = np.floor(grid).astype(np.int64)
        frac = grid - base
        shape = np.asarray(self._safety_esdf.shape, dtype=np.int64)
        valid = np.all(
            (base >= 0) & (base + 1 < shape[None, :]), axis=1)
        values = np.full(len(points), -np.inf, dtype=np.float64)
        if not np.any(valid):
            return values

        b = base[valid]
        f = frac[valid]
        accum = np.zeros(len(b), dtype=np.float64)
        for dx in (0, 1):
            wx = (1.0 - f[:, 0]) if dx == 0 else f[:, 0]
            for dy in (0, 1):
                wy = (1.0 - f[:, 1]) if dy == 0 else f[:, 1]
                for dz in (0, 1):
                    wz = (1.0 - f[:, 2]) if dz == 0 else f[:, 2]
                    accum += wx * wy * wz * self._safety_esdf[
                        b[:, 0] + dx, b[:, 1] + dy, b[:, 2] + dz]
        values[valid] = accum
        return values

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

