#!/usr/bin/env python3
"""Offline regression tests for deterministic terminal GOAL_HOLD semantics."""

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

from il_common import (  # noqa: E402
    ControlMode,
    PlannerMode,
    TrendMode,
    goal_hold_guide_labels,
    make_goal_hold_decision,
    update_goal_hold_latch,
)


class GoalHoldSemanticsTest(unittest.TestCase):
    def test_goal_tolerance_entry_latches_across_inertial_overshoot(self):
        goal = np.array([0.0, 10.0, 2.0])

        self.assertFalse(update_goal_hold_latch(
            False, [0.0, 9.699, 2.0], goal, 0.30))
        self.assertTrue(update_goal_hold_latch(
            False, [0.0, 9.700, 2.0], goal, 0.30))
        self.assertTrue(update_goal_hold_latch(
            True, [0.0, 10.45, 2.0], goal, 0.30))

    def test_goal_hold_releases_only_after_stopping_outside_tolerance(self):
        goal = np.array([0.0, 10.0, 2.0])
        self.assertTrue(update_goal_hold_latch(
            True, [0.0, 10.45, 2.0], goal, 0.30,
            current_speed_mps=0.20, goal_speed_tolerance_mps=0.10))
        self.assertFalse(update_goal_hold_latch(
            True, [0.0, 10.45, 2.0], goal, 0.30,
            current_speed_mps=0.05, goal_speed_tolerance_mps=0.10))

    def test_goal_hold_keeps_existing_13_class_network_interface(self):
        h_class, v_class, h_soft, v_soft = goal_hold_guide_labels(11, 7)

        self.assertEqual(h_class, 6)
        self.assertEqual(v_class, 3)
        self.assertEqual(h_soft.shape, (13,))
        self.assertEqual(v_soft.shape, (7,))
        np.testing.assert_array_equal(
            np.flatnonzero(h_soft), np.array([6]))
        np.testing.assert_array_equal(
            np.flatnonzero(v_soft), np.array([3]))
        self.assertAlmostEqual(float(h_soft.sum()), 1.0)
        self.assertAlmostEqual(float(v_soft.sum()), 1.0)

    def test_goal_hold_decision_is_normal_center_guide_with_zero_control(self):
        position = np.array([0.0, 10.0, 2.0])
        decision = make_goal_hold_decision(
            position, position.copy(), goal_path_index=42)

        self.assertEqual(decision.planner_mode, PlannerMode.GOAL_HOLD.value)
        self.assertEqual(decision.trend_mode, TrendMode.GOAL_HOLD.value)
        self.assertEqual(
            decision.control_mode, ControlMode.HOLD_POSITION.value)
        self.assertEqual(decision.guide_source, "goal_tolerance_hold")
        self.assertEqual(decision.selected_actor, "goal_hold")
        np.testing.assert_array_equal(
            decision.guide_target_world, position)
        np.testing.assert_array_equal(
            decision.expert_velocity_flu, np.zeros(3))
        np.testing.assert_array_equal(
            decision.selected_velocity_flu, np.zeros(3))
        self.assertEqual(decision.expert_yaw_rate, 0.0)
        self.assertEqual(decision.selected_yaw_rate, 0.0)

    def test_invalid_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            update_goal_hold_latch(
                False, np.zeros(3), np.zeros(3), float("nan"))


if __name__ == "__main__":
    unittest.main()
