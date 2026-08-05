#!/usr/bin/env python3
"""Regression tests for actual-yaw scanning and swept-guide certification."""

import os
import sys
import types
import unittest

import numpy as np


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
    MacroExpert,
    MacroExpertConfig,
    MacroState,
)
from il_observed_map import (  # noqa: E402
    FREE,
    OCCUPIED,
    UNKNOWN,
    RollingObservedOccupancyMap,
)


class _ObservedMapStub:
    def __init__(self, corridor_ratio):
        self.corridor_ratio = float(corridor_ratio)

    def sample_known_free_ratio_along_corridor(self, **_kwargs):
        return self.corridor_ratio

    def is_known_free(self, _point):
        # Deliberately true: the guide test must reject based on the complete
        # corridor rather than accepting this endpoint-only result.
        return True

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


def _quaternion_from_yaw(yaw):
    return np.array(
        [0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)],
        dtype=np.float64)


def _update(expert, observed_map, yaw, local_blocked):
    return expert.update(
        goal_direction_flu=np.array([1.0, 0.0, 0.0]),
        goal_distance_m=3.0,
        depth_m=np.full((12, 16), 5.0, dtype=np.float32),
        observed_map=observed_map,
        current_position_world=np.zeros(3),
        current_yaw=yaw,
        current_quaternion_xyzw=_quaternion_from_yaw(yaw),
        current_velocity_world=np.zeros(3),
        local_blocked=local_blocked,
        local_progress_rate=1.0,
        local_feasible=True,
        dt_since_last_macro=0.2)


class MacroExpertObservabilityTest(unittest.TestCase):
    def test_scan_waits_for_actual_yaw_motion(self):
        config = MacroExpertConfig(enter_blocked_frames=1)
        expert = MacroExpert(config)
        observed_map = _ObservedMapStub(corridor_ratio=0.0)

        guide = _update(expert, observed_map, yaw=0.0, local_blocked=True)
        self.assertEqual(guide.macro_state, MacroState.ACTIVE_SCAN_LEFT.value)

        # A now-clear map must not end scanning while the real camera yaw is
        # unchanged, even though configured_yaw_rate * elapsed_time exceeds
        # the old 15 degree threshold.
        observed_map.corridor_ratio = 1.0
        for _ in range(3):
            guide = _update(
                expert, observed_map, yaw=0.0, local_blocked=False)
            self.assertEqual(
                guide.macro_state, MacroState.ACTIVE_SCAN_LEFT.value)

        actual_yaw = np.deg2rad(16.0)
        guide = _update(
            expert, observed_map, yaw=actual_yaw, local_blocked=False)
        self.assertEqual(guide.macro_state, MacroState.BYPASS_LEFT.value)
        self.assertGreater(guide.move_distance_m, 0.0)

    def test_free_endpoint_does_not_bypass_blocked_swept_volume(self):
        expert = MacroExpert(MacroExpertConfig())
        observed_map = _ObservedMapStub(corridor_ratio=0.0)

        guide = _update(expert, observed_map, yaw=0.0, local_blocked=False)

        self.assertIn(guide.macro_state, (
            MacroState.ACTIVE_SCAN_LEFT.value,
            MacroState.ACTIVE_SCAN_RIGHT.value))
        self.assertEqual(guide.move_distance_m, 0.0)

    def test_cpp_reachability_checker_replaces_straight_chord_gate(self):
        expert = MacroExpert(MacroExpertConfig())
        observed_map = _ObservedMapStub(corridor_ratio=0.0)
        calls = []

        def reachable(_position, _velocity, _direction, desired,
                      _minimum, _step):
            calls.append(desired)
            # Represents a curved bounded-A* path to an endpoint whose direct
            # chord is not fully observed/free.
            return desired

        expert.set_guide_reachability_checker(reachable)
        guide = _update(expert, observed_map, yaw=0.0, local_blocked=False)

        self.assertTrue(calls)
        self.assertEqual(guide.macro_state, MacroState.GOAL_SEEK.value)
        self.assertGreater(guide.move_distance_m, 0.0)

    def test_rejected_scanned_candidate_continues_same_perception_side(self):
        expert = MacroExpert(MacroExpertConfig())
        observed_map = _ObservedMapStub(corridor_ratio=1.0)
        expert.set_guide_reachability_checker(
            lambda *_args, **_kwargs: 0.0)

        first = _update(
            expert, observed_map, yaw=0.0, local_blocked=False)
        self.assertEqual(
            first.macro_state, MacroState.ACTIVE_SCAN_LEFT.value)

        # A single rejected candidate at the minimum scan angle must not
        # reverse the camera. Continue expanding the same observed sector;
        # side switching is allowed only at the scan duration/angle limit.
        second = _update(
            expert, observed_map, yaw=np.deg2rad(16.0),
            local_blocked=False)
        self.assertEqual(
            second.macro_state, MacroState.ACTIVE_SCAN_LEFT.value)
        self.assertEqual(second.move_distance_m, 0.0)

    def test_local_plan_failure_atomically_replaces_move_with_scan(self):
        expert = MacroExpert(MacroExpertConfig())
        observed_map = _ObservedMapStub(corridor_ratio=1.0)
        scan = expert.force_active_scan(
            goal_direction_flu=np.array([1.0, 0.0, 0.0]),
            goal_distance_m=3.0,
            observed_map=observed_map,
            current_position_world=np.zeros(3),
            current_quaternion_xyzw=_quaternion_from_yaw(0.0))

        self.assertEqual(
            scan.macro_state, MacroState.ACTIVE_SCAN_LEFT.value)
        self.assertEqual(scan.move_distance_m, 0.0)
        np.testing.assert_allclose(scan.move_target_world, np.zeros(3))
        self.assertGreater(
            np.linalg.norm(scan.look_target_world), 0.9)

    def test_corridor_separates_known_body_from_obstacle_clearance_ring(self):
        config = {"global": {
            "depth": {"fov": 90.0, "max_m": 5.0},
            "esdf": {"drone_radius": 0.10},
            "observed_map": {
                "resolution": 0.10,
                "size_x_m": 2.0,
                "size_y_m": 2.0,
                "size_z_m": 2.0,
            },
        }}
        observed_map = RollingObservedOccupancyMap(config)
        observed_map.reset(np.zeros(3))
        occupancy = observed_map.get_occupancy(copy=False)
        occupancy.fill(FREE)

        # This voxel is in the extra clearance ring but outside the 0.10 m
        # body support. Unknown is allowed there because only the vehicle body
        # must be observed; an actual obstacle in the same ring is not.
        outer = observed_map._world_to_grid_int(np.zeros(3))
        outer[1] += 2
        occupancy[tuple(outer)] = UNKNOWN
        ratio = observed_map.sample_known_free_ratio_along_corridor(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            radius_m=0.10,
            spacing_m=0.05,
            min_clearance_m=0.10)
        self.assertAlmostEqual(ratio, 1.0)

        occupancy[tuple(outer)] = OCCUPIED
        ratio = observed_map.sample_known_free_ratio_along_corridor(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            radius_m=0.10,
            spacing_m=0.05,
            min_clearance_m=0.10)
        self.assertLess(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
