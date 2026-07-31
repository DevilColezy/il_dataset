#!/usr/bin/env python3
"""Offline regression checks for observation-conditioned global rollout."""

import importlib.util
import os
import sys
import unittest

import numpy as np


class _RospyStub:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


sys.modules.setdefault("rospy", _RospyStub())
_SCRIPT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts", "il_trajectory.py"))
_SPEC = importlib.util.spec_from_file_location(
    "il_trajectory_under_test", _SCRIPT_PATH)
_TRAJECTORY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAJECTORY)


def _cylinder_esdf(obstacles):
    """Build the 0.1 m analytic-grid ESDF used by the legacy scenarios."""
    resolution = 0.1
    origin = np.array([-12.0, -3.0, 0.0], dtype=np.float64)
    gx, gy, gz = 240, 380, 80
    xs = origin[0] + (np.arange(gx) + 0.5) * resolution
    ys = origin[1] + (np.arange(gy) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    planar_esdf = np.full_like(xx, 1.0e3, dtype=np.float64)
    for center_x, center_y, radius in obstacles:
        # Dataset ESDF values already have the 0.3 m vehicle radius removed.
        obstacle_esdf = (
            np.hypot(xx - center_x, yy - center_y) - radius - 0.3)
        planar_esdf = np.minimum(planar_esdf, obstacle_esdf)
    esdf = np.repeat(
        planar_esdf[:, :, None].astype(np.float32), gz, axis=2)
    return esdf, origin, resolution


class ObservationConditionedRolloutTest(unittest.TestCase):
    _CONFIG = {
        "step_length_m": 0.10,
        "angular_step_deg": 2.0,
        "observation_range_m": 5.0,
        "horizontal_fov_deg": 90.0,
        "fov_margin_deg": 3.0,
        "maximum_deviation_deg": 120.0,
        "goal_tolerance_m": 0.10,
        "maximum_rollout_steps": 10000,
        "maximum_planning_time_s": 45.0,
    }

    def test_open_space_is_one_exact_goal_segment(self):
        esdf, origin, resolution = _cylinder_esdf([])
        path, report = _TRAJECTORY.observation_conditioned_rollout_path(
            (0.0, -1.0, 2.0), (0.0, 32.0, 2.0),
            esdf, origin, resolution, 0.3, 0.05, self._CONFIG)

        self.assertGreater(len(path), 2)
        self.assertLess(
            np.max(np.abs(np.asarray(path)[:, 0])), 1.0e-9)
        self.assertEqual(report["decision_events"], [])
        self.assertAlmostEqual(report["length"], 33.0, places=9)

    def test_scale_transition_walks_straight_then_branches(self):
        esdf, origin, resolution = _cylinder_esdf([
            (0.25, 9.0, 0.60),
            (-0.40, 22.0, 2.00),
        ])
        path, report = _TRAJECTORY.observation_conditioned_rollout_path(
            (0.0, -1.0, 2.0), (0.0, 32.0, 2.0),
            esdf, origin, resolution, 0.3, 0.05,
            self._CONFIG)

        self.assertIsNotNone(path, report)
        points = np.asarray(path)

        # The search must not anticipate a distant obstacle: it walks exactly
        # toward the goal until the next fixed step would violate clearance.
        first_decision_index = report["decision_events"][0]["path_index"]
        before_first = points[:first_decision_index + 1]
        self.assertGreater(len(before_first), 2)
        self.assertLess(np.max(np.abs(before_first[:, 0])), 1.0e-9)

        # Equal-altitude cylinder tasks stay planar; avoidance cannot create a
        # spurious up/down training label.
        self.assertLess(
            np.max(np.abs(points[:, 2] - 2.0)), 1.0e-9)
        self.assertGreater(len(report["decision_events"]), 0)
        self.assertAlmostEqual(report["cost"], report["length"], places=9)
        self.assertEqual(
            report["policy"], "minimum_visible_safe_deviation")
        self.assertEqual(
            report["equal_deviation_tie_break"], "right")
        self.assertGreater(report["left_selected_events"], 0)
        first = report["decision_events"][0]
        self.assertEqual(first["selected_side"], "LEFT")
        self.assertLess(
            first["left_min_deviation_deg"],
            first["right_min_deviation_deg"])
        self.assertLess(report["length"], 36.0)

        valid, worst = _TRAJECTORY._validate_polyline(
            points, esdf, origin, resolution, 0.3, 0.05)
        self.assertTrue(valid)
        self.assertGreaterEqual(worst, 0.299)

    def test_two_large_symmetric_blockers_keep_distinct_branches(self):
        esdf, origin, resolution = _cylinder_esdf([
            (0.0, 5.0, 1.80),
            (0.0, 27.0, 1.80),
        ])
        path, report = _TRAJECTORY.observation_conditioned_rollout_path(
            (0.0, -1.0, 2.0), (0.0, 32.0, 2.0),
            esdf, origin, resolution, 0.3, 0.05,
            self._CONFIG)

        self.assertIsNotNone(path, report)
        points = np.asarray(path)
        self.assertGreaterEqual(len(report["decision_events"]), 2)
        self.assertGreater(report["right_selected_events"], 0)
        self.assertEqual(report["left_selected_events"], 0)
        self.assertGreater(np.max(points[:, 0]), 0.5)
        self.assertLess(report["length"], 45.0)
        self.assertLess(
            np.max(np.abs(points[:, 2] - 2.0)), 1.0e-9)

        valid, worst = _TRAJECTORY._validate_polyline(
            points, esdf, origin, resolution, 0.3, 0.05)
        self.assertTrue(valid)
        self.assertGreaterEqual(worst, 0.299)

    def test_later_obstacle_cannot_reverse_first_equal_length_choice(self):
        first_choices = []
        for obstacles in (
                [(0.0, 5.0, 1.20)],
                [(0.0, 5.0, 1.20), (2.0, 20.0, 1.80)]):
            esdf, origin, resolution = _cylinder_esdf(obstacles)
            path, report = (
                _TRAJECTORY.observation_conditioned_rollout_path(
                (0.0, -1.0, 2.0), (0.0, 32.0, 2.0),
                esdf, origin, resolution, 0.3, 0.05,
                self._CONFIG))
            self.assertIsNotNone(path, report)
            points = np.asarray(path)
            lateral = np.flatnonzero(np.abs(points[:, 0]) > 1.0e-6)
            self.assertGreater(len(lateral), 0)
            first_choices.append(points[lateral[0], 0])

        # A centered first blocker is an equal-length decision. RIGHT is +x
        # for this +y task, and a later blocker must not change that decision.
        self.assertGreater(first_choices[0], 0.0)
        self.assertGreater(first_choices[1], 0.0)

    def test_visible_right_blocker_makes_left_deviation_smaller(self):
        esdf, origin, resolution = _cylinder_esdf([
            (0.0, 9.0, 0.80),
            (1.5, 7.5, 0.60),
        ])
        path, report = _TRAJECTORY.observation_conditioned_rollout_path(
            (0.0, -1.0, 2.0), (0.0, 32.0, 2.0),
            esdf, origin, resolution, 0.3, 0.05,
            self._CONFIG)

        self.assertIsNotNone(path, report)
        first = report["decision_events"][0]
        self.assertEqual(first["selected_side"], "LEFT")
        self.assertLess(
            first["left_min_deviation_deg"],
            first["right_min_deviation_deg"])

    def test_global_planner_entrypoint_uses_no_line_push_fallback(self):
        esdf, origin, resolution = _cylinder_esdf([
            (0.25, 9.0, 0.60),
        ])
        planner = _TRAJECTORY.GlobalPathPlanner(
            esdf, origin, resolution,
            {
                "planning": {
                    "global_planner": {
                        "algorithm": "observation_rollout",
                        "observation_rollout": self._CONFIG,
                        "min_clearance": 0.30,
                        "collision_check_spacing_m": 0.05,
                        "reference_resample_spacing_m": 0.10,
                    }
                }
            })
        plan = planner.plan_global(
            (0.0, -1.0, 2.0), (0.0, 16.0, 2.0))

        self.assertIsNotNone(plan)
        report = plan["validation_report"]
        self.assertEqual(
            report["planner_algorithm"], "observation_rollout")
        self.assertIn("observation_rollout", report)
        self.assertNotIn("line_push", report)


if __name__ == "__main__":
    unittest.main()
