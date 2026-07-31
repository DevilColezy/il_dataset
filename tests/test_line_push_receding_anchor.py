#!/usr/bin/env python3
"""Offline regression checks for receding-anchor LinePush."""

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


class RecedingAnchorLinePushTest(unittest.TestCase):
    def test_scale_transition_uses_shorter_side_for_each_obstacle(self):
        esdf, origin, resolution = _cylinder_esdf([
            (0.25, 9.0, 0.60),
            (-0.40, 22.0, 2.00),
        ])
        path, report = _TRAJECTORY.line_push_path(
            (0.0, -1.0, 2.0), (0.0, 32.0, 2.0),
            esdf, origin, resolution, 0.3, 0.05,
            {
                "point_spacing_m": 0.10,
                "push_margin_m": 0.10,
                "max_iterations": 80,
                "max_offset_m": 12.0,
                "max_control_points": 5000,
            })

        self.assertIsNotNone(path, report)
        points = np.asarray(path)
        early_anchor = points[np.argmin(np.abs(points[:, 1] - 9.0))]
        late_anchor = points[np.argmin(np.abs(points[:, 1] - 22.0))]

        # Travel is along +Y: negative X is left and positive X is right.
        self.assertLess(early_anchor[0], 0.0)
        self.assertGreater(late_anchor[0], 0.0)

        # Collision-free prefixes must remain on the active straight ray.
        before_first = points[points[:, 1] <= 7.5]
        between_obstacles = points[
            (points[:, 1] >= 10.5) & (points[:, 1] <= 19.0)]
        self.assertGreater(len(before_first), 2)
        self.assertGreater(len(between_obstacles), 2)
        self.assertLess(np.max(np.abs(before_first[:, 0])), 1.0e-9)
        self.assertLess(np.max(np.abs(between_obstacles[:, 0])), 1.0e-9)
        self.assertNotIn("tightened_points_removed", report)
        self.assertAlmostEqual(report["cost"], report["length"], places=9)

        valid, worst = _TRAJECTORY._validate_polyline(
            points, esdf, origin, resolution, 0.3, 0.05)
        self.assertTrue(valid)
        self.assertGreaterEqual(worst, 0.299)


if __name__ == "__main__":
    unittest.main()
