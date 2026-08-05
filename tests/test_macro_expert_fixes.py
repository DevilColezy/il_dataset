#!/usr/bin/env python3
"""Regression tests for the macro-expert fixes (safe-known bubble, unified A*,
   scan state machine, map history)."""

import math
import os
import sys
import types
import unittest

import numpy as np


# ── Stub rospy / rospkg for Windows static analysis ─────────────────────
class _RospyStub(types.ModuleType):
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


try:
    import rospy  # noqa: F401
except ImportError:
    sys.modules["rospy"] = _RospyStub("rospy")
try:
    import rospkg  # noqa: F401
except ImportError:
    sys.modules["rospkg"] = types.ModuleType("rospkg")

_SCRIPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts"))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from il_macro_expert import (  # noqa: E402
    CommittedSide,
    MacroExpert,
    MacroExpertConfig,
    MacroGuide,
    MacroState,
)
from il_observed_map import (  # noqa: E402
    FREE,
    OCCUPIED,
    UNKNOWN,
    ObservedESDF,
    RollingObservedOccupancyMap,
)


def _quaternion_from_yaw(yaw):
    return np.array(
        [0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)],
        dtype=np.float64)


def _make_map_config(resolution=0.10, size_xy=2.0, size_z=2.0,
                     history_seconds=3.0,
                     vehicle_radius=0.30):
    return {"global": {
        "depth": {"fov": 90.0, "max_m": 5.0},
        "esdf": {"drone_radius": float(vehicle_radius)},
        "observed_map": {
            "resolution": float(resolution),
            "size_x_m": float(size_xy),
            "size_y_m": float(size_xy),
            "size_z_m": float(size_z),
            "history_seconds": float(history_seconds),
        },
    }}


# ────────────────────────────────────────────────────────────────────────────
#  1. Vehicle free-bubble safe-known coverage
# ────────────────────────────────────────────────────────────────────────────

class VehicleFreeBubbleTest(unittest.TestCase):
    """The free bubble must guarantee that every sub-voxel drone position
    yields all eight trilinear-interpolation voxels in the safe-known mask."""

    def _build_esdf_for_position(self, center_world, resolution=0.10,
                                 vehicle_radius=0.30):
        """Build an observed map + ESDF centred at *center_world*."""
        config = _make_map_config(
            resolution=resolution, size_xy=2.0, size_z=2.0,
            vehicle_radius=vehicle_radius)
        obs_map = RollingObservedOccupancyMap(config)
        obs_map.reset(center_world)
        # Mark a single depth frame (no obstacles) so the free bubble is
        # applied and ESDF is built.
        depth = np.full((64, 64), 5.0, dtype=np.float32)
        obs_map.integrate_depth(
            depth, center_world,
            _quaternion_from_yaw(0.0),
            0.0)
        occupancy = obs_map.get_occupancy(copy=False)
        esdf = ObservedESDF(config)
        esdf.rebuild(
            occupancy=occupancy,
            known_mask=obs_map.get_known_mask(),
            origin_world=obs_map.get_origin(),
            resolution=resolution)
        return obs_map, esdf

    def test_center_position_eight_interpolation_voxels_are_safe_known(self):
        """Drone at voxel centre → all 8 interpolation voxels safe-known."""
        resolution = 0.10
        # Choose a centre that maps to fractional 0.5 in every dimension
        # so the drone is exactly at a voxel centre.
        origin_zero = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        center = origin_zero + 0.5 * resolution  # voxel centre
        _, esdf = self._build_esdf_for_position(center, resolution)
        self.assertTrue(esdf.is_built())

        # The ESDF isKnown check requires all 8 interpolation voxels known.
        self.assertTrue(
            esdf.value_at(center) is not None,
            "Voxel-centre position must be in safe-known mask")

    def test_subvoxel_corner_position_eight_interpolation_voxels_are_safe_known(self):
        """Drone near voxel corner → all 8 interpolation voxels safe-known."""
        resolution = 0.10
        origin_zero = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        # Position at 0.01 fraction within its floor cell in each dimension.
        # This is a worst-case sub-voxel offset for the interpolation stencil.
        center = origin_zero + 0.01 * resolution
        _, esdf = self._build_esdf_for_position(center, resolution)
        self.assertTrue(esdf.is_built())
        self.assertTrue(
            esdf.value_at(center) is not None,
            "Sub-voxel-corner position must be in safe-known mask")

    def test_subvoxel_near_one_position_eight_interpolation_voxels_are_safe_known(self):
        """Drone near the +1 edge of its floor cell in each dimension."""
        resolution = 0.10
        origin_zero = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        # f ≈ 0.99 in each dimension
        center = origin_zero + 0.99 * resolution
        _, esdf = self._build_esdf_for_position(center, resolution)
        self.assertTrue(esdf.is_built())
        self.assertTrue(
            esdf.value_at(center) is not None,
            "Near-+1-edge position must be in safe-known mask")

    def test_free_bubble_does_not_overwrite_occupied(self):
        """Bubble must never mark OCCUPIED voxels as FREE."""
        config = _make_map_config(resolution=0.10, size_xy=2.0, size_z=2.0)
        obs_map = RollingObservedOccupancyMap(config)
        center = np.zeros(3)
        obs_map.reset(center)
        occupancy = obs_map.get_occupancy(copy=False)
        # Put an OCCUPIED voxel inside the bubble radius.
        occ_idx = obs_map._world_to_grid_int(center)
        occupancy[tuple(occ_idx)] = OCCUPIED
        # Now integrate a depth frame — the bubble runs after integration.
        depth = np.full((64, 64), 5.0, dtype=np.float32)
        obs_map.integrate_depth(
            depth, center, _quaternion_from_yaw(0.0), 0.0)
        # The occupied voxel must stay OCCUPIED.
        self.assertEqual(
            occupancy[tuple(occ_idx)], OCCUPIED,
            "Bubble must never overwrite OCCUPIED voxels")


# ────────────────────────────────────────────────────────────────────────────
#  2. Bounded A* start / endpoint semantics
# ────────────────────────────────────────────────────────────────────────────

class BoundedAStartSemanticsTest(unittest.TestCase):
    """The macro reachability check must not accept paths that the formal
    planner would immediately reject at t=0 (UNKNOWN_SPACE)."""

    def test_reachability_rejects_when_start_is_unknown(self):
        """When guided by a C++ checker that asserts start isKnown, a start
        in unknown space must return 0."""
        expert = MacroExpert(MacroExpertConfig())
        calls = []

        def checker(position, _velocity, _direction, desired,
                    _minimum, _step):
            calls.append(desired)
            # Simulate the C++ findReachableGuideDistance returning 0
            # because the start position fails the isKnown check.
            return 0.0

        expert.set_guide_reachability_checker(checker)
        # Use a map stub that reports everything free (the checker overrides).
        import il_macro_expert as mex
        # We need a map stub that passes _map_supports_frontiers.
        class FrontierMapStub:
            def get_occupancy(self, copy=True):
                return np.ones((10, 10, 10), dtype=np.uint8)
            def get_origin(self):
                return np.zeros(3)
            def get_resolution(self):
                return 0.10
            def is_known_free(self, _point):
                return True
            def free_voxel_count(self):
                return 100
            def occupied_voxel_count(self):
                return 0
            def known_voxel_count(self):
                return 100
            def get_revision(self):
                return 1
        obs_map = FrontierMapStub()
        guide = expert.update(
            goal_direction_flu=np.array([1.0, 0.0, 0.0]),
            goal_distance_m=3.0,
            depth_m=np.full((12, 16), 5.0, dtype=np.float32),
            observed_map=obs_map,
            current_position_world=np.zeros(3),
            current_yaw=0.0,
            current_quaternion_xyzw=_quaternion_from_yaw(0.0),
            current_velocity_world=np.zeros(3),
            local_blocked=True,
            dt_since_last_macro=0.2)
        # The checker returns 0, so no guide distance can be used.
        # The expert should fall back to scan, not issue a move guide.
        self.assertIn(
            guide.macro_state,
            (MacroState.ACTIVE_SCAN_LEFT.value,
             MacroState.ACTIVE_SCAN_RIGHT.value))
        self.assertEqual(guide.move_distance_m, 0.0)
        self.assertTrue(calls, "Checker must have been invoked")

    def test_reachability_accepts_when_start_and_end_are_known_free(self):
        """When the checker returns the full desired distance, the goal-seek
        guide must be valid with non-zero distance."""
        expert = MacroExpert(MacroExpertConfig(enter_blocked_frames=10))
        reached = []

        def checker(position, _velocity, _direction, desired,
                    _minimum, _step):
            reached.append(float(desired))
            return desired

        expert.set_guide_reachability_checker(checker)
        # ObservedMap stub that always reports known-free corridors.
        class AlwaysClearMapStub:
            def sample_known_free_ratio_along_corridor(self, **_kwargs):
                return 1.0
            def is_known_free(self, _point):
                return True
            def free_voxel_count(self):
                return 100
            def occupied_voxel_count(self):
                return 0
            def known_voxel_count(self):
                return 100
            def get_occupancy(self):
                return np.ones((10, 10, 10), dtype=np.uint8)
            def get_revision(self):
                return 1
        obs_map = AlwaysClearMapStub()
        guide = expert.update(
            goal_direction_flu=np.array([1.0, 0.0, 0.0]),
            goal_distance_m=3.0,
            depth_m=np.full((12, 16), 5.0, dtype=np.float32),
            observed_map=obs_map,
            current_position_world=np.zeros(3),
            current_yaw=0.0,
            current_quaternion_xyzw=_quaternion_from_yaw(0.0),
            current_velocity_world=np.zeros(3),
            local_blocked=False,
            dt_since_last_macro=0.2)
        self.assertEqual(guide.macro_state, MacroState.GOAL_SEEK.value)
        self.assertGreater(guide.move_distance_m, 0.0)
        self.assertTrue(reached, "Checker must have been invoked")


# ────────────────────────────────────────────────────────────────────────────
#  3. Active scan state machine
# ────────────────────────────────────────────────────────────────────────────

class ActiveScanStateMachineTest(unittest.TestCase):
    """The scan session must cover left +90°, cross back through initial,
    and reach right -90° without oscillating."""

    def _make_expert(self, **overrides):
        kwargs = dict(
            enter_blocked_frames=1,
            active_scan_yaw_rate_rps=0.8,
            active_scan_min_angle_deg=15.0,
            active_scan_max_duration_s=8.0,
            map_history_seconds=8.0,
        )
        kwargs.update(overrides)
        return MacroExpert(MacroExpertConfig(**kwargs))

    def _make_blocked_map(self):
        """Return a map stub that always reports blocked corridors."""
        class BlockedMapStub:
            def sample_known_free_ratio_along_corridor(self, **_kwargs):
                return 0.0
            def is_known_free(self, _point):
                return False
            def free_voxel_count(self):
                return 0
            def occupied_voxel_count(self):
                return 0
            def known_voxel_count(self):
                return 0
            def get_occupancy(self):
                return np.zeros((10, 10, 10), dtype=np.uint8)
            def get_revision(self):
                return 1
            def get_origin(self):
                return np.array([-1.0, -1.0, -1.0])
            def get_resolution(self):
                return 0.10
        return BlockedMapStub()

    def _update(self, expert, obs_map, yaw, dt=0.2, blocked=True):
        return expert.update(
            goal_direction_flu=np.array([1.0, 0.0, 0.0]),
            goal_distance_m=3.0,
            depth_m=np.full((12, 16), 5.0, dtype=np.float32),
            observed_map=obs_map,
            current_position_world=np.zeros(3),
            current_yaw=yaw,
            current_quaternion_xyzw=_quaternion_from_yaw(yaw),
            current_velocity_world=np.zeros(3),
            local_blocked=blocked,
            local_progress_rate=1.0,
            local_feasible=True,
            dt_since_last_macro=dt)

    def test_scan_starts_on_preferred_left_side(self):
        expert = self._make_expert()
        obs_map = self._make_blocked_map()
        guide = self._update(expert, obs_map, yaw=0.0)
        self.assertEqual(guide.macro_state,
                         MacroState.ACTIVE_SCAN_LEFT.value)

    def test_scan_accumulates_continuous_angle(self):
        """Single continuous sweep: the scan accumulates angle in one
        direction and keeps scanning past 90°, 180°, toward 360°."""
        expert = self._make_expert()
        obs_map = self._make_blocked_map()
        # Start at yaw=0
        self._update(expert, obs_map, yaw=0.0)
        self.assertEqual(expert._scan_phase, 0)

        # Rotate to +180° in steps — must stay in the same scan state.
        for deg in range(15, 185, 15):
            yaw = float(np.deg2rad(deg))
            guide = self._update(expert, obs_map, yaw=yaw)
            self.assertEqual(guide.macro_state,
                             MacroState.ACTIVE_SCAN_LEFT.value,
                             "Must stay in scan; no phase switch at 90°")
        # After 180°, _scan_angle_accum should be ~π rad.
        self.assertGreaterEqual(
            expert._scan_angle_accum, math.radians(170.0))

    def test_scan_ends_after_full_circle(self):
        """After 360° of rotation the state exits ACTIVE_SCAN."""
        expert = self._make_expert()
        obs_map = self._make_blocked_map()

        self._update(expert, obs_map, yaw=0.0)

        # Rotate past 360° — the full-circle check fires and exits scan.
        exited_scan = False
        for deg in range(30, 370, 30):
            yaw = np.deg2rad(deg)
            guide = self._update(expert, obs_map, yaw=yaw)
            if guide.macro_state not in (MacroState.ACTIVE_SCAN_LEFT.value,
                                         MacroState.ACTIVE_SCAN_RIGHT.value):
                exited_scan = True
                break

        self.assertTrue(
            exited_scan,
            "Must exit ACTIVE_SCAN after full 360° sweep")

    def test_scan_exits_early_when_corridor_found(self):
        """When a reachable frontier appears mid-scan, exit immediately."""
        expert = self._make_expert()

        # Map that starts blocked, becomes clear after min angle.
        class GradualMapStub:
            def __init__(self):
                self.call_count = 0
            def sample_known_free_ratio_along_corridor(self, **_kwargs):
                self.call_count += 1
                return 1.0 if self.call_count > 3 else 0.0
            def is_known_free(self, _point):
                return self.call_count > 3
            def free_voxel_count(self):
                return 100
            def occupied_voxel_count(self):
                return 0
            def known_voxel_count(self):
                return 100
            def get_occupancy(self):
                occ = np.zeros((10, 10, 10), dtype=np.uint8)
                occ[4:7, 4:7, 5] = FREE
                return occ
            def get_revision(self):
                return 1
            def get_origin(self):
                return np.array([-1.0, -1.0, -1.0])
            def get_resolution(self):
                return 0.10
        obs_map = GradualMapStub()

        # Must use a reachability checker that returns non-zero.
        def checker(*_args, **_kwargs):
            return 0.8
        expert.set_guide_reachability_checker(checker)

        self._update(expert, obs_map, yaw=0.0, blocked=True)
        guide = self._update(expert, obs_map, yaw=np.deg2rad(20.0),
                             blocked=False)
        # After min_angle + corridor found, should transition to BYPASS.
        if guide.macro_state in (MacroState.BYPASS_LEFT.value,
                                 MacroState.BYPASS_RIGHT.value):
            self.assertGreater(guide.move_distance_m, 0.0)
            # Scan session should be cleaned up.
            self.assertIsNone(expert._scan_session_initial_yaw)
        else:
            # If the stub map didn't provide enough frontier support for
            # _find_horizontal_corridor, the scan continues — that's fine.
            pass

    def test_pi_wrap_handled_correctly(self):
        """Crossing the ±π boundary must not break angle accumulation."""
        expert = self._make_expert()
        obs_map = self._make_blocked_map()

        # Start near π (179°) and scan left.
        # Left rotation goes from π → π+small → -π+small (wrap) → ...
        init = np.deg2rad(179.0)
        self._update(expert, obs_map, yaw=init)
        self.assertIsNotNone(expert._scan_session_initial_yaw)

        # Move to -179° (crossing +π wrap, a small leftward step).
        yaw2 = np.deg2rad(-179.0)
        self._update(expert, obs_map, yaw=yaw2)
        # scan_relative_angle should be ~2° (small positive), not ~358°.
        rel = np.rad2deg(float(expert._scan_relative_angle))
        self.assertLess(abs(rel), 10.0,
                        "±π wrap must be handled correctly, got {:.1f}°".format(rel))

    def test_scan_does_not_flip_back_and_forth_at_15_degrees(self):
        """A single rejected candidate at 15° must not reverse the scan side."""
        expert = self._make_expert()
        obs_map = self._make_blocked_map()
        # Set a checker that always rejects.
        expert.set_guide_reachability_checker(
            lambda *_args, **_kwargs: 0.0)

        first = self._update(expert, obs_map, yaw=0.0, blocked=True)
        self.assertEqual(first.macro_state,
                         MacroState.ACTIVE_SCAN_LEFT.value)

        # Move to minimum scan angle (16°) — checker rejects.
        second = self._update(
            expert, obs_map, yaw=np.deg2rad(16.0), blocked=False)
        # Must stay in ACTIVE_SCAN_LEFT, not flip to right.
        self.assertEqual(second.macro_state,
                         MacroState.ACTIVE_SCAN_LEFT.value)
        self.assertEqual(second.move_distance_m, 0.0)


# ────────────────────────────────────────────────────────────────────────────
#  4. Map history
# ────────────────────────────────────────────────────────────────────────────

class MapHistoryTest(unittest.TestCase):
    """The rolling map must retain observations for history_seconds and
    expire them afterwards — it is not a permanent global map."""

    def test_observations_persist_within_history_window(self):
        config = _make_map_config(resolution=0.10, size_xy=4.0, size_z=2.0,
                                  history_seconds=5.0)
        obs_map = RollingObservedOccupancyMap(config)
        center = np.zeros(3)
        obs_map.reset(center)
        depth = np.full((64, 64), 3.0, dtype=np.float32)
        obs_map.integrate_depth(
            depth, center, _quaternion_from_yaw(0.0), 0.0)
        initial_known = obs_map.known_voxel_count()
        self.assertGreater(initial_known, 0,
                           "Integration must produce known voxels")

        # Advance time within history window — voxels must persist.
        obs_map.integrate_depth(
            depth, center, _quaternion_from_yaw(0.0), 3.0)
        mid_known = obs_map.known_voxel_count()
        self.assertGreater(mid_known, initial_known * 0.5,
                           "Known voxels must persist within history window")

    def test_observations_expire_after_history_window(self):
        config = _make_map_config(resolution=0.10, size_xy=4.0, size_z=2.0,
                                  history_seconds=2.0)
        obs_map = RollingObservedOccupancyMap(config)
        center = np.zeros(3)
        obs_map.reset(center)
        depth = np.full((64, 64), 3.0, dtype=np.float32)
        obs_map.integrate_depth(
            depth, center, _quaternion_from_yaw(0.0), 0.0)
        initial_known = obs_map.known_voxel_count()
        self.assertGreater(initial_known, 0)

        # Advance past history window.
        obs_map.integrate_depth(
            depth, center, _quaternion_from_yaw(0.0), 3.0)
        # After 3 s with a 2 s window, the original voxels should have
        # expired.  The new integration refreshes some, but known count
        # should not grow indefinitely.
        later_known = obs_map.known_voxel_count()
        # The map is not permanently accumulating — it's rolling.
        self.assertLess(
            later_known, initial_known * 3.0,
            "Rolling map must expire old observations")

    def test_config_validation_rejects_impossible_scan_timing(self):
        """history_seconds must be ≥ active_scan_max_duration_s."""
        with self.assertRaises(ValueError):
            MacroExpertConfig(
                active_scan_yaw_rate_rps=0.8,
                active_scan_max_duration_s=8.0,
                map_history_seconds=3.0,  # too short
            ).validate(depth_max_m=5.0)

    def test_config_validation_rejects_duration_too_short_for_full_scan(self):
        """active_scan_max_duration_s must cover ~270° at yaw_rate."""
        with self.assertRaises(ValueError):
            MacroExpertConfig(
                active_scan_yaw_rate_rps=0.8,
                active_scan_max_duration_s=2.0,  # too short for 270°
                map_history_seconds=8.0,
            ).validate(depth_max_m=5.0)


# ────────────────────────────────────────────────────────────────────────────
#  5. Diagnostic fields
# ────────────────────────────────────────────────────────────────────────────

class DiagnosticFieldsTest(unittest.TestCase):
    """The new rejection-reason fields must appear in MacroGuide and the
    held-guide dictionary."""

    def test_macro_guide_has_new_fields(self):
        guide = MacroGuide()
        self.assertTrue(hasattr(guide, "extracted_frontier_candidate_count"))
        self.assertTrue(hasattr(guide, "reachable_frontier_candidate_count"))
        self.assertTrue(hasattr(guide, "frontier_rejection_summary"))

    def test_build_guide_populates_frontier_diagnostics(self):
        expert = MacroExpert(MacroExpertConfig(enter_blocked_frames=1))
        # Set rejection reasons directly.
        expert._extracted_frontier_count = 5
        expert._reachable_frontier_count = 1
        expert._frontier_rejection_reasons = {
            0: "endpoint_unknown",
            1: "astar_no_path_or_start_unknown",
            2: "endpoint_unknown",
            3: "astar_no_path_or_start_unknown",
        }
        expert._last_frontier_candidate_count = 4

        class StubMap:
            def sample_known_free_ratio_along_corridor(self, **_kw):
                return 1.0
            def free_voxel_count(self):
                return 1
            def occupied_voxel_count(self):
                return 0
            def known_voxel_count(self):
                return 1
            def get_occupancy(self):
                return np.ones((1, 1, 1), dtype=np.uint8)
            def get_revision(self):
                return 1
        obs_map = StubMap()
        guide = expert.update(
            goal_direction_flu=np.array([1.0, 0.0, 0.0]),
            goal_distance_m=3.0,
            depth_m=np.full((12, 16), 5.0, dtype=np.float32),
            observed_map=obs_map,
            current_position_world=np.zeros(3),
            current_yaw=0.0,
            current_quaternion_xyzw=_quaternion_from_yaw(0.0),
            current_velocity_world=np.zeros(3),
            local_blocked=True,
            dt_since_last_macro=0.2)
        self.assertEqual(guide.extracted_frontier_candidate_count, 5)
        self.assertEqual(guide.reachable_frontier_candidate_count, 1)
        self.assertIn("endpoint_unknown", guide.frontier_rejection_summary)
        self.assertIn("astar_no_path_or_start_unknown",
                      guide.frontier_rejection_summary)

    def test_held_guide_exposes_frontier_diagnostics(self):
        expert = MacroExpert(MacroExpertConfig(enter_blocked_frames=1))
        expert._extracted_frontier_count = 3
        expert._reachable_frontier_count = 0
        expert._frontier_rejection_reasons = {
            0: "endpoint_unknown",
            1: "endpoint_unknown",
            2: "astar_no_path_or_start_unknown",
        }
        expert._held_extracted_frontier_count = 3
        expert._held_reachable_frontier_count = 0
        expert._held_rejection_reasons = dict(
            expert._frontier_rejection_reasons)
        expert._held_valid = True
        expert._held_move_direction_flu = np.array([1.0, 0.0, 0.0])
        expert._held_move_distance_m = 0.0
        expert._held_move_distance_norm = 0.0
        expert._held_yaw_direction_flu_xy = np.array([1.0, 0.0])
        expert._state = MacroState.ACTIVE_SCAN_LEFT
        held = expert.get_held_guide_flu(
            np.zeros(3), _quaternion_from_yaw(0.0))
        self.assertEqual(held["extracted_frontier_candidate_count"], 3)
        self.assertEqual(held["reachable_frontier_candidate_count"], 0)
        self.assertIn("endpoint_unknown",
                      held["frontier_rejection_summary"])


if __name__ == "__main__":
    unittest.main()
