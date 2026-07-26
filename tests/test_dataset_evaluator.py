#!/usr/bin/env python3

import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import dataset_evaluator as evaluator


class DatasetEvaluatorTest(unittest.TestCase):

    def test_csv_inference_uses_all_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "data.csv")
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["planner_compute_ms"])
                for _ in range(20):
                    writer.writerow(["0"])
                writer.writerow(["12.5"])

            data = evaluator.load_trajectory_csv(path)
            self.assertEqual(data.dtype["planner_compute_ms"].kind, "f")
            self.assertEqual(float(data["planner_compute_ms"][-1]), 12.5)

    def test_schema_v15_recovery_masks_and_planner_attempts(self):
        dtype = [
            ("trajectory_time_s", "f8"),
            ("trend_horizontal_class_13", "i8"),
            ("trend_horizontal_loss_valid", "i8"),
            ("guide_elevation_bin", "i8"),
            ("trend_vertical_loss_valid", "i8"),
            ("guide_mode", "U20"),
            ("trend_mode", "U20"),
            ("recovery_direction", "U8"),
            ("planner_attempted", "i8"),
            ("planner_success", "i8"),
            ("planner_compute_ms", "f8"),
            ("planner_min_clearance", "f8"),
            ("state_vx_flu", "f8"),
            ("state_vy_flu", "f8"),
            ("state_vz_flu", "f8"),
        ]
        rows = [
            (0.0, 0, 1, 2, 1, "RECOVER_LEFT", "RECOVERY", "left",
             0, 0, 0.0, 0.0, 1.0, 0.0, 0.0),
            (0.1, 12, 1, 3, 1, "RECOVER_RIGHT", "RECOVERY", "right",
             1, 1, 12.5, 0.8, 1.0, 0.0, 0.0),
            # recovery_direction remains populated after recovery; this is normal.
            (0.2, 11, 1, -1, 0, "NORMAL", "TRACK_GUIDE", "right",
             0, 1, 0.0, 0.8, 1.0, 0.0, 0.0),
            # Invalid horizontal target must not enter the class histogram.
            (0.3, 5, 0, 4, 1, "NORMAL", "TRACK_GUIDE", "right",
             1, 0, 7.5, 0.0, 1.0, 0.0, 0.0),
        ]
        data = np.array(rows, dtype=dtype)
        metrics = evaluator.compute_trajectory_metrics(
            data, {}, "traj", "scene", "task")

        self.assertEqual(metrics.recovery_frame_count, 2)
        self.assertEqual(metrics.trend_class_distribution,
                         {0: 1, 12: 1, 11: 1})
        self.assertEqual(metrics.guide_elevation_distribution,
                         {2: 1, 3: 1, 4: 1})
        self.assertEqual(metrics.num_replans, 2)
        self.assertEqual(metrics.num_failed_replans, 1)
        self.assertAlmostEqual(metrics.planner_success_rate, 0.5)
        self.assertAlmostEqual(metrics.mean_planning_ms, 10.0)
        self.assertAlmostEqual(metrics.mean_clearance, 0.8)

    def test_discovery_accepts_only_committed_schema_v15(self):
        with tempfile.TemporaryDirectory() as root:
            cases = [
                ("scene/deep/traj_ok", 15, "committed", True),
                ("scene/traj_old", 14, "committed", False),
                ("scene/traj_open", 15, "collecting", False),
                ("scene/traj_partial.inprogress", 15, "committed", False),
            ]
            for relative, schema, status, _ in cases:
                directory = os.path.join(root, *relative.split("/"))
                os.makedirs(directory)
                with open(os.path.join(directory, "data.csv"), "w") as stream:
                    stream.write("x\n1\n")
                with open(os.path.join(directory, "metadata.json"), "w") as stream:
                    json.dump({"schema_version": schema, "status": status}, stream)

            found = evaluator.discover_datasets(root)
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0].endswith(os.path.join(
                "scene", "deep", "traj_ok")))


if __name__ == "__main__":
    unittest.main()
