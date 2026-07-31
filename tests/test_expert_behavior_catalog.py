#!/usr/bin/env python3
"""Offline validation for deterministic expert-review scenarios and suites."""

import math
import os
import sys
import types
import unittest
import xml.etree.ElementTree as ET


PACKAGE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.join(PACKAGE_DIR, "scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from expert_behavior_catalog import (  # noqa: E402
    load_scenario_catalog,
    select_scenarios,
)


CATALOG_PATH = os.path.join(
    PACKAGE_DIR, "config", "expert_behavior_scenarios.yaml")


def _point_segment_distance(point, start, goal):
    vx = goal[0] - start[0]
    vy = goal[1] - start[1]
    length_sq = vx * vx + vy * vy
    projection = (
        (point[0] - start[0]) * vx +
        (point[1] - start[1]) * vy) / length_sq
    projection = max(0.0, min(1.0, projection))
    closest = (
        start[0] + projection * vx,
        start[1] + projection * vy,
    )
    return math.hypot(point[0] - closest[0], point[1] - closest[1])


class ExpertBehaviorCatalogTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (cls.resolved, cls.catalog, cls.scenarios, cls.suites,
         cls.acceptance) = load_scenario_catalog(CATALOG_PATH)

    def test_human_review_covers_required_behavior_families(self):
        selected = set(self.suites["human_review"])
        required = {
            "open_baseline",
            "small_mid_left_offset",
            "small_mid_right_offset",
            "large_mid_left_offset",
            "large_mid_right_offset",
            "scale_transition",
            "large_to_small_transition",
            "narrow_gate",
            "double_slalom",
            "sparse_mixed_field",
            "dense_mixed_field",
        }
        self.assertTrue(required.issubset(selected))

    def test_suite_and_mixed_selection_are_ordered_and_deduplicated(self):
        self.assertEqual(
            select_scenarios(
                "quick_check,small_mid_center", self.scenarios, self.suites),
            self.suites["quick_check"])
        self.assertEqual(
            select_scenarios(
                "open_baseline,dense_mixed_field",
                self.scenarios, self.suites),
            ["open_baseline", "dense_mixed_field"])
        with self.assertRaises(ValueError):
            select_scenarios("missing_suite", self.scenarios, self.suites)

    def test_declared_direct_blocker_counts_match_geometry(self):
        corridor_radius = 0.35
        for name, scenario in self.scenarios.items():
            start = scenario["start"]
            goal = scenario["goal"]
            blockers = 0
            for obstacle in scenario["obstacles"]:
                distance = _point_segment_distance(
                    obstacle["center"], start, goal)
                if distance <= float(
                        obstacle["radius_m"]) + corridor_radius:
                    blockers += 1
            if scenario.get("require_direct_path_blocked", True):
                minimum = int(
                    scenario.get("minimum_direct_blocker_count", 1))
                maximum = int(scenario.get(
                    "maximum_direct_blocker_count",
                    max(1, len(scenario["obstacles"]))))
                self.assertGreaterEqual(
                    blockers, minimum, "{} blockers".format(name))
                self.assertLessEqual(
                    blockers, maximum, "{} blockers".format(name))
            else:
                self.assertEqual(
                    blockers, 0, "{} should have a clear direct path".format(
                        name))

    def test_obstacles_keep_canonical_raw_surface_gap(self):
        for name, scenario in self.scenarios.items():
            obstacles = scenario["obstacles"]
            for first_index, first in enumerate(obstacles):
                for second in obstacles[first_index + 1:]:
                    center_a = first["center"]
                    center_b = second["center"]
                    center_distance = math.hypot(
                        center_a[0] - center_b[0],
                        center_a[1] - center_b[1])
                    surface_gap = center_distance - (
                        float(first["radius_m"]) +
                        float(second["radius_m"]))
                    self.assertGreaterEqual(
                        surface_gap + 1.0e-9, 1.20,
                        "{}: {} vs {}".format(
                            name, first["id"], second["id"]))

    def test_launch_defaults_to_manual_review_suite(self):
        launch_path = os.path.join(
            PACKAGE_DIR, "launch", "expert_behavior_test.launch")
        root = ET.parse(launch_path).getroot()
        args = {
            element.attrib["name"]: element.attrib.get("default")
            for element in root.findall("arg")
        }
        self.assertEqual(args["scenario"], "")
        self.assertEqual(args["suite"], "human_review")
        self.assertEqual(args["show_plot"], "true")

    def test_single_scenario_uses_only_fixed_scenario_configuration(self):
        rospy_stub = types.ModuleType("rospy")
        il_config_stub = types.ModuleType("il_config")
        il_config_stub.load_config = lambda: None
        il_manager_stub = types.ModuleType("il_manager")
        il_manager_stub.ILManager = object
        saved = {
            name: sys.modules.get(name)
            for name in ("rospy", "il_config", "il_manager",
                         "expert_behavior_test")
        }
        try:
            sys.modules["rospy"] = rospy_stub
            sys.modules["il_config"] = il_config_stub
            sys.modules["il_manager"] = il_manager_stub
            sys.modules.pop("expert_behavior_test", None)
            from expert_behavior_test import _configure_fixed_scenario

            cfg = {
                "global": {
                    "scene_generation": {
                        "profiles": [{"name": "formal_collection_only"}],
                        "task_generation": {"stale": True},
                        "common_task_generation": {},
                        "execution": {},
                    },
                    "start_goal": {},
                    "dagger": {},
                }
            }
            _configure_fixed_scenario(
                cfg, "open_baseline", CATALOG_PATH,
                os.path.join(PACKAGE_DIR, "_test_output"), 7)
            scene_cfg = cfg["global"]["scene_generation"]
            self.assertEqual(scene_cfg["source"], "fixed_scenario")
            self.assertNotIn("profiles", scene_cfg)
            self.assertNotIn("task_generation", scene_cfg)
            common = scene_cfg["common_task_generation"]
            self.assertEqual(common["tasks_per_scene"], 1)
            self.assertEqual(
                common["fixed_tasks"][0]["start"],
                self.scenarios["open_baseline"]["start"])
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value


if __name__ == "__main__":
    unittest.main()
