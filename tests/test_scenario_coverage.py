#!/usr/bin/env python3
"""Offline regression tests for coverage-balanced scene/task generation."""

import copy
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np
import yaml


class _RospyStub:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _RosPackStub:
    def get_path(self, _package_name):
        return os.path.abspath(os.path.join(
            os.path.dirname(__file__), ".."))


sys.modules.setdefault("rospy", _RospyStub())
sys.modules.setdefault(
    "rospkg", type("_RospkgStub", (), {"RosPack": _RosPackStub})())
_SCRIPT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts", "il_scenario.py"))
_SPEC = importlib.util.spec_from_file_location(
    "il_scenario_under_test", _SCRIPT_PATH)
_SCENARIO = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCENARIO
_SPEC.loader.exec_module(_SCENARIO)

_CONFIG_SCRIPT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts", "il_config.py"))
_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "il_config_under_test", _CONFIG_SCRIPT_PATH)
_CONFIG = importlib.util.module_from_spec(_CONFIG_SPEC)
sys.modules[_CONFIG_SPEC.name] = _CONFIG
_CONFIG_SPEC.loader.exec_module(_CONFIG)


def _task_config(tasks_per_scene=4, task_type="any"):
    coverage = {
        "enabled": True,
        "rotate_region_pairs_by_seed": True,
        "task_type_weights": {task_type: 1.0},
        "blocker_distance_bands_m": {"all": [0.0, 1000.0]},
        "blocker_distance_weights": {"all": 1.0},
        "height_delta_weights": {"level": 1.0},
        "level_height_delta_max_m": 0.25,
        "minimum_nonlevel_height_delta_m": 0.75,
    }
    return {
        "global": {
            "scene_generation": {
                "common_task_generation": {
                    "max_task_sampling_attempts": 3,
                    "tasks_per_scene": tasks_per_scene,
                    "start_height_min_m": 2.0,
                    "start_height_max_m": 2.0,
                    "goal_height_min_m": 2.0,
                    "goal_height_max_m": 2.0,
                    "maximum_start_goal_height_difference_m": 0.1,
                    "start_clearance_m": 0.0,
                    "goal_clearance_m": 0.0,
                    "minimum_start_goal_distance_m": 1.0,
                    "maximum_start_goal_distance_m": 100.0,
                    "require_direct_path_blocked": False,
                    "direct_path_corridor_radius_m": 0.35,
                    "minimum_direct_blocker_count": 1,
                    "maximum_direct_blocker_count": 3,
                    "require_astar_reachable": False,
                    "minimum_detour_ratio": 1.0,
                    "maximum_detour_ratio": 1000.0,
                    "start_sampling_regions": [
                        {"x_min": 0.0, "x_max": 0.0,
                         "y_min": 0.0, "y_max": 0.0},
                        {"x_min": 10.0, "x_max": 10.0,
                         "y_min": 0.0, "y_max": 0.0},
                    ],
                    "goal_sampling_regions": [
                        {"x_min": 0.0, "x_max": 0.0,
                         "y_min": 20.0, "y_max": 20.0},
                        {"x_min": -10.0, "x_max": -10.0,
                         "y_min": 0.0, "y_max": 0.0},
                    ],
                    "coverage_balancing": coverage,
                },
            },
        },
    }


def _obstacle(obstacle_id, x, y, radius=0.5, z=4.0, height=8.0):
    return _SCENARIO.CylinderObstacleSpec(
        obstacle_id=obstacle_id,
        center_world=np.array([x, y, z], dtype=np.float64),
        radius_m=radius,
        height_m=height)


def _density_config(minimum_achievement_ratio=0.85):
    """Small deterministic density-driven fixture for offline tests."""
    return {
        "global": {
            "scene_generation": {
                "source": "density_driven",
                "seed": 31,
                "obstacle_region": {
                    "x_min": 0.0, "x_max": 20.0,
                    "y_min": 0.0, "y_max": 20.0,
                    "z_min": 0.0, "z_max": 8.0,
                },
                "execution": {
                    "max_obstacle_sampling_attempts": 5000,
                },
                "generation_quality": {
                    "minimum_density_achievement_ratio":
                        minimum_achievement_ratio,
                },
                "placement_stratification": {
                    "enabled": True,
                    "x_bands": 3,
                    "y_bands": 5,
                },
                "vehicle": {
                    "radius_m": 0.0,
                    "safety_margin_m": 0.0,
                },
                "common_cylinder": {
                    "height_min_m": 8.0,
                    "height_max_m": 8.0,
                    "region_boundary_margin_m": 0.0,
                    "minimum_surface_gap_m": 0.0,
                    "minimum_post_inflation_gap_m": 0.0,
                },
                "common_task_generation": {
                    "tasks_per_scene": 1,
                },
                "profiles": [{
                    "name": "capacity_fixture",
                    "enabled": True,
                    "scene_count": 1,
                    "seed_offset": 0,
                    "density_min": 0.05,
                    "density_max": 0.05,
                    "size_groups": {
                        "large": {
                            "radius_min_m": 1.0,
                            "radius_max_m": 1.0,
                            "capacity_fraction": 0.10,
                            "consecutive_fail_threshold": 500,
                        },
                        "medium": {
                            "radius_min_m": 0.5,
                            "radius_max_m": 0.5,
                            "capacity_fraction": 0.20,
                            "consecutive_fail_threshold": 500,
                        },
                        "small": {
                            "radius_min_m": 0.2,
                            "radius_max_m": 0.2,
                            "capacity_fraction": 0.70,
                            "consecutive_fail_threshold": 500,
                        },
                    },
                }],
            },
        },
    }


class ScenarioGeometryContractTest(unittest.TestCase):
    def test_2p5d_profile_requires_cylinders_to_span_region_height(self):
        config = _density_config()
        scene_cfg = config["global"]["scene_generation"]
        scene_cfg["common_cylinder"]["height_min_m"] = 7.5
        with self.assertRaisesRegex(ValueError, "must fully cover"):
            _SCENARIO.load_scene_profiles(config)

    def test_fixed_obstacle_must_span_the_complete_vertical_region(self):
        config = _density_config()
        scene_cfg = config["global"]["scene_generation"]
        scene_cfg["source"] = "fixed_scenario"
        scene_cfg["fixed_obstacles"] = [{
            "id": "short",
            "center": [10.0, 10.0, 4.0],
            "radius_m": 0.5,
            "height_m": 7.0,
        }]
        generator = _SCENARIO.YamlCylinderSceneGenerator(config)
        obstacles, reason = generator.generate_scene()
        self.assertEqual(obstacles, [])
        self.assertEqual(reason, "FIXED_OBSTACLE_NOT_FULL_HEIGHT")

    def test_all_geometry_consumers_use_scene_vehicle_only(self):
        config = {
            "global": {
                "scene_generation": {
                    "vehicle": {
                        "radius_m": 0.41,
                        "safety_margin_m": 0.17,
                    },
                    "topology_validation": {
                        "vehicle_radius_m": 9.0,
                        "safety_margin_m": 8.0,
                        "grid_resolution_m": 0.12,
                    },
                    "side_cost": {},
                    "observability_audit": {},
                },
            },
        }
        validator = _SCENARIO.CylinderSceneValidator(config)
        side_cost = _SCENARIO.SideCostEvaluator(config)
        auditor = _SCENARIO.ObstacleVisibilityAuditor(config)
        self.assertAlmostEqual(validator.vehicle_r, 0.41)
        self.assertAlmostEqual(validator.safety_m, 0.17)
        self.assertAlmostEqual(side_cost.vehicle_r, 0.41)
        self.assertAlmostEqual(side_cost.safety_m, 0.17)
        self.assertAlmostEqual(auditor.inflation, 0.58)


class CoverageBalancedTaskTest(unittest.TestCase):
    def test_largest_remainder_quota_is_exact_and_deterministic(self):
        cfg = _task_config(tasks_per_scene=8)
        coverage = cfg["global"]["scene_generation"][
            "common_task_generation"]["coverage_balancing"]
        coverage["task_type_weights"] = {
            "clear": 0.25,
            "single_left": 0.25,
            "single_right": 0.25,
            "multi_blocker": 0.25,
        }
        coverage["blocker_distance_bands_m"] = {
            "near": [0.0, 10.0],
            "far": [10.0, 1000.0],
        }
        coverage["blocker_distance_weights"] = {
            "near": 0.5, "far": 0.5}

        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        first = generator._build_coverage_targets(8, seed=17)
        second = generator._build_coverage_targets(8, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(item["task_type"] for item in first),
            Counter({
                "clear": 2,
                "single_left": 2,
                "single_right": 2,
                "multi_blocker": 2,
            }))
        blocked_distances = [
            item["blocker_distance_band"] for item in first
            if item["task_type"] != "clear"]
        self.assertEqual(
            Counter(blocked_distances),
            Counter({"near": 3, "far": 3}))

    def test_task_generator_reads_only_common_task_generation(self):
        cfg = _task_config(tasks_per_scene=4)
        cfg["global"]["scene_generation"]["task_generation"] = {
            "tasks_per_scene": 99,
            "max_task_sampling_attempts": 999,
            "fixed_tasks": [{"start": [9, 9, 9], "goal": [8, 8, 8]}],
        }
        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        self.assertEqual(generator.tasks_per_scene, 4)
        self.assertEqual(generator.max_attempts, 3)
        self.assertEqual(generator.fixed_tasks, [])

    def test_fixed_tasks_remain_supported_in_common_schema(self):
        cfg = _task_config(tasks_per_scene=1)
        common = cfg["global"]["scene_generation"][
            "common_task_generation"]
        common["fixed_tasks"] = [{
            "start": [0.0, 0.0, 2.0],
            "goal": [0.0, 20.0, 2.0],
        }]
        common["require_astar_reachable"] = False
        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        esdf = np.full((32, 32, 8), 100.0, dtype=np.float32)
        tasks = generator.generate_tasks(
            [], esdf, (-10.0, -10.0, 0.0), 1.0, None, seed=0)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0][0], [0.0, 0.0, 2.0])
        self.assertEqual(tasks[0][1], [0.0, 20.0, 2.0])

    def test_distance_quota_is_resolved_by_density_tier(self):
        cfg = _task_config(tasks_per_scene=6, task_type="single_center")
        coverage = cfg["global"]["scene_generation"][
            "common_task_generation"]["coverage_balancing"]
        coverage["blocker_distance_bands_m"] = {
            "near": [0.0, 6.0],
            "middle": [6.0, 15.0],
            "far": [15.0, 1000.0],
        }
        coverage["blocker_distance_weights_by_density_tier"] = {
            "default": {"near": 0.5, "middle": 0.5},
            "sparse": {"middle": 0.25, "far": 0.75},
            "dense": {"near": 0.75, "middle": 0.25, "far": 0.50},
        }
        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        generator._configure_coverage(coverage, density_tier="sparse")
        sparse = generator._build_coverage_targets(4, seed=4)
        self.assertEqual(
            Counter(item["blocker_distance_band"] for item in sparse),
            Counter({"far": 3, "middle": 1}))

        generator._configure_coverage(coverage, density_tier="dense")
        dense = generator._build_coverage_targets(4, seed=4)
        dense_distances = [
            item["blocker_distance_band"] for item in dense]
        self.assertNotIn("far", dense_distances)
        self.assertEqual(
            Counter(dense_distances),
            Counter({"near": 3, "middle": 1}))

    def test_direct_path_classification_balances_left_right_and_height(self):
        generator = _SCENARIO.StartGoalTaskGenerator(_task_config())
        start = np.array([0.0, 0.0, 2.0])
        goal = np.array([0.0, 20.0, 2.0])

        left = generator._classify_direct_path(
            start, goal, [_obstacle("left", -0.6, 8.0)])
        right = generator._classify_direct_path(
            start, goal, [_obstacle("right", 0.6, 8.0)])
        centered = generator._classify_direct_path(
            start, goal, [_obstacle("center", 0.0, 8.0)])
        multi = generator._classify_direct_path(
            start, goal, [
                _obstacle("first", -0.6, 6.0),
                _obstacle("second", 0.6, 13.0),
            ])

        self.assertEqual(left["task_type"], "single_left")
        self.assertEqual(right["task_type"], "single_right")
        self.assertEqual(centered["task_type"], "single_center")
        self.assertEqual(multi["task_type"], "multi_blocker")

        high_start = np.array([0.0, 0.0, 4.0])
        high_goal = np.array([0.0, 20.0, 4.0])
        below = _obstacle(
            "below", 0.0, 8.0, z=1.0, height=1.0)
        at_height = _obstacle(
            "at_height", 0.0, 8.0, z=4.0, height=1.0)
        self.assertEqual(
            generator._classify_direct_path(
                high_start, high_goal, [below])["task_type"],
            "clear")
        self.assertEqual(
            generator._classify_direct_path(
                high_start, high_goal, [at_height])["task_type"],
            "single_center")

        esdf = np.full((32, 32, 8), 100.0, dtype=np.float32)
        below_result = generator._validate_task(
            high_start, high_goal, [below], esdf,
            (-10.0, -10.0, 0.0), 1.0, None)
        at_height_result = generator._validate_task(
            high_start, high_goal, [at_height], esdf,
            (-10.0, -10.0, 0.0), 1.0, None)
        self.assertFalse(below_result.direct_path_blocked)
        self.assertEqual(below_result.direct_blocker_count, 0)
        self.assertTrue(at_height_result.direct_path_blocked)
        self.assertEqual(at_height_result.direct_blocker_count, 1)

    def test_impossible_bucket_rejects_complete_scene_without_fallback(self):
        cfg = _task_config(tasks_per_scene=4, task_type="multi_blocker")
        # A removed compatibility key must not re-enable label substitution.
        cfg["global"]["scene_generation"][
            "common_task_generation"]["coverage_balancing"][
                "fallback_to_any_valid"] = True
        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        esdf = np.full((32, 32, 8), 100.0, dtype=np.float32)
        with self.assertRaisesRegex(
                RuntimeError, "rejecting the complete scene"):
            generator.generate_tasks(
                [], esdf, (-10.0, -10.0, 0.0), 1.0, None, seed=5)

    def test_strict_bucket_rotates_across_region_pairs(self):
        cfg = _task_config(tasks_per_scene=1, task_type="multi_blocker")
        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        obstacles = [
            _obstacle("horizontal_first", 3.0, 0.0),
            _obstacle("horizontal_second", -3.0, 0.0),
        ]
        esdf = np.full((32, 32, 8), 100.0, dtype=np.float32)
        tasks = generator.generate_tasks(
            obstacles, esdf, (-10.0, -10.0, 0.0), 1.0,
            None, seed=0)
        self.assertEqual(len(tasks), 1)
        # Pair 0 is the clear vertical route. Pair 1 is the horizontal route
        # containing both blockers and must be tried on the second attempt.
        self.assertEqual(tasks[0][2].coverage_region_pair_index, 1)
        self.assertEqual(
            tasks[0][2].coverage_actual_task_type, "multi_blocker")

    def test_incomplete_task_count_is_rejected_instead_of_silently_shrunk(self):
        cfg = _task_config(tasks_per_scene=1, task_type="multi_blocker")
        coverage = cfg["global"]["scene_generation"][
            "common_task_generation"]["coverage_balancing"]
        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        esdf = np.full((32, 32, 8), 100.0, dtype=np.float32)

        with self.assertRaisesRegex(
                RuntimeError, "Task coverage target unavailable"):
            generator.generate_tasks(
                [], esdf, (-10.0, -10.0, 0.0), 1.0, None, seed=5)

    def test_task_manifest_records_requested_and_actual_coverage(self):
        cfg = _task_config(tasks_per_scene=1, task_type="multi_blocker")
        generator = _SCENARIO.StartGoalTaskGenerator(cfg)
        esdf = np.full((32, 32, 8), 100.0, dtype=np.float32)
        obstacles = [
            _obstacle("first", -0.6, 6.0),
            _obstacle("second", 0.6, 13.0),
        ]
        start, goal, result = generator.generate_tasks(
            obstacles, esdf, (-10.0, -10.0, 0.0), 1.0,
            None, seed=0)[0]

        with tempfile.TemporaryDirectory() as output_dir:
            writer = _SCENARIO.SceneManifestWriter(output_dir)
            path = writer.write_task_manifest(
                "scene", "task_000", start, goal, result, "sparse")
            with open(path, "r") as stream:
                manifest = json.load(stream)
        self.assertEqual(
            manifest["coverage_target_task_type"], "multi_blocker")
        self.assertEqual(
            manifest["coverage_actual_task_type"], "multi_blocker")
        self.assertNotIn("coverage_fallback", manifest)


class DensityDrivenGenerationTest(unittest.TestCase):
    @staticmethod
    def _generator_and_profile(minimum_achievement_ratio):
        cfg = _density_config(minimum_achievement_ratio)
        generator = _SCENARIO.YamlCylinderSceneGenerator(cfg)
        # The Windows CI image carries an old NumPy without Generator/PCG64.
        # RandomState provides the uniform/shuffle API used by this test.
        generator._make_np_rng = (
            lambda seed: np.random.RandomState(int(seed)))
        profile = _SCENARIO.load_scene_profiles(cfg)[0]
        return generator, profile

    def test_stratified_sampler_visits_each_cell_once_per_pass(self):
        sampler = _SCENARIO._StratifiedGridSampler(
            np.random.RandomState(7), target_cell_count=12,
            aspect_ratio=1.0)
        points = [
            sampler.draw(0.0, 1.0, 0.0, 1.0)
            for _ in range(sampler.nx * sampler.ny)
        ]
        occupied_cells = {
            (
                min(sampler.nx - 1, int(x * sampler.nx)),
                min(sampler.ny - 1, int(y * sampler.ny)),
            )
            for x, y in points
        }
        self.assertEqual(
            len(occupied_cells), sampler.nx * sampler.ny)

    def test_capacity_fraction_controls_area_and_keeps_every_scale(self):
        generator, profile = self._generator_and_profile(0.85)
        obstacles, reason, target, mode = (
            generator.generate_scene_density_driven(
                profile, effective_scene_seed=31,
                scene_index_in_profile=0, attempt_index=0))
        self.assertEqual(reason, "")
        self.assertEqual(mode, "inflated_occupancy")
        counts = Counter()
        areas = Counter()
        for obstacle in obstacles:
            if obstacle.radius_m >= 0.99:
                group = "large"
            elif obstacle.radius_m >= 0.49:
                group = "medium"
            else:
                group = "small"
            counts[group] += 1
            areas[group] += math.pi * obstacle.radius_m ** 2
            self.assertAlmostEqual(obstacle.center_world[2], 4.0)
        self.assertTrue(all(counts[name] >= 1 for name in (
            "large", "medium", "small")))
        self.assertGreater(areas["small"], areas["medium"])
        self.assertGreater(areas["small"], areas["large"])
        actual = _SCENARIO.compute_inflated_occupancy(
            obstacles, 400.0, 0.0, 0.0)
        self.assertGreaterEqual(actual / target, 0.85)

    def test_underfilled_density_is_rejected_at_configured_ratio(self):
        generator, profile = self._generator_and_profile(1.0)
        original_compute = _SCENARIO.compute_inflated_occupancy
        _SCENARIO.compute_inflated_occupancy = (
            lambda *_args, **_kwargs: 0.0)
        try:
            obstacles, reason, target, mode = (
                generator.generate_scene_density_driven(
                    profile, effective_scene_seed=31,
                    scene_index_in_profile=0, attempt_index=0))
        finally:
            _SCENARIO.compute_inflated_occupancy = original_compute
        self.assertEqual(obstacles, [])
        self.assertEqual(
            reason, "DENSITY_DRIVEN_TARGET_DENSITY_UNDERSHOT")
        self.assertEqual(mode, "inflated_occupancy")
        self.assertEqual(target, 0.05)

    def test_group_area_planner_hits_feasible_budget_exactly(self):
        areas = _SCENARIO._plan_obstacle_areas(
            np.random.RandomState(9),
            budget_m2=5.0,
            minimum_area_m2=1.0,
            maximum_area_m2=3.0)
        self.assertAlmostEqual(sum(areas), 5.0)
        self.assertTrue(all(1.0 <= area <= 3.0 for area in areas))

    def test_capacity_fractions_must_sum_to_one(self):
        generator, profile = self._generator_and_profile(0.85)
        profile.size_groups[0].capacity_fraction = 0.20
        obstacles, reason, target, _mode = (
            generator.generate_scene_density_driven(
                profile, effective_scene_seed=31,
                scene_index_in_profile=0, attempt_index=0))
        self.assertEqual(obstacles, [])
        self.assertEqual(
            reason,
            "DENSITY_DRIVEN_CAPACITY_FRACTIONS_MUST_SUM_TO_ONE")
        self.assertIsNone(target)


class SceneDensityMixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "config",
            "il_dataset_config.yaml"))
        with open(cls.config_path, "r", encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)

    def test_canonical_config_uses_more_scenes_and_fewer_tasks(self):
        profiles = _SCENARIO.load_scene_profiles(self.config)
        self.assertEqual(len(profiles), 12)
        self.assertEqual(sum(profile.scene_count for profile in profiles), 48)
        self.assertEqual(
            self.config["global"]["scene_generation"][
                "common_task_generation"]["tasks_per_scene"],
            12)
        self.assertEqual(
            len(self.config["global"]["scene_generation"][
                "common_task_generation"]["start_sampling_regions"]),
            8)
        self.assertEqual(
            {profile.density_tier for profile in profiles},
            {"sparse", "medium", "dense"})
        tiers = {profile.name: profile.density_tier for profile in profiles}
        self.assertEqual(tiers["S02_small_medium"], "sparse")
        self.assertEqual(tiers["S03_small_dense"], "medium")
        self.assertEqual(tiers["S04_medium_sparse"], "sparse")
        self.assertEqual(tiers["S06_medium_dense"], "dense")

        task_generator = _SCENARIO.StartGoalTaskGenerator(self.config)
        by_name = {profile.name: profile for profile in profiles}
        task_generator.configure_from_profile(
            by_name["S01_small_sparse"])
        sparse_targets = task_generator._build_coverage_targets(12, seed=1)
        task_generator.configure_from_profile(
            by_name["S06_medium_dense"])
        dense_targets = task_generator._build_coverage_targets(12, seed=1)
        sparse_counts = Counter(
            item["task_type"] for item in sparse_targets)
        dense_counts = Counter(
            item["task_type"] for item in dense_targets)
        self.assertGreater(sparse_counts["clear"], dense_counts["clear"])
        self.assertLess(
            sparse_counts["multi_blocker"],
            dense_counts["multi_blocker"])
        self.assertEqual(sparse_counts["multi_blocker"], 0)
        self.assertEqual(dense_counts["clear"], 0)
        self.assertNotIn(
            "far",
            [item["blocker_distance_band"] for item in dense_targets])

    def test_s01_low_density_reaches_the_97_percent_contract(self):
        generator = _SCENARIO.YamlCylinderSceneGenerator(self.config)
        generator._make_np_rng = (
            lambda seed: np.random.RandomState(int(seed)))
        profiles = {
            profile.name: profile
            for profile in _SCENARIO.load_scene_profiles(self.config)}
        profile = profiles["S01_small_sparse"]
        obstacles = []
        target = None
        for attempt in range(20):
            obstacles, reason, target, _mode = (
                generator.generate_scene_density_driven(
                    profile,
                    generator.base_seed + profile.seed_offset,
                    scene_index_in_profile=0,
                    attempt_index=attempt))
            if obstacles:
                break
        self.assertTrue(obstacles, reason)
        area = _SCENARIO.compute_region_area(
            generator.obstacle_region)
        actual = _SCENARIO.compute_inflated_occupancy(
            obstacles, area,
            profile.vehicle_radius_m, profile.safety_margin_m)
        self.assertGreaterEqual(actual / target, 0.97)
        self.assertEqual(generator.placement_x_bands, 4)
        self.assertEqual(generator.placement_y_bands, 6)

    def test_all_shipped_configs_use_the_strict_density_flow(self):
        config_dir = os.path.dirname(self.config_path)
        for filename in (
                "il_dataset_config.yaml",
                "il_dataset_config_1.yaml",
                "il_dataset_config_2.yaml"):
            with open(
                    os.path.join(config_dir, filename),
                    "r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream)
            _CONFIG._validate_config(loaded)
            self.assertEqual(loaded["global"]["data"]["schema_version"], 16)
            self.assertEqual(
                loaded["global"]["data"]["collection_mode"],
                "deterministic_lockstep")
            self.assertNotIn("obstacle", loaded["global"])
            self.assertNotIn("start_goal", loaded["global"])
            scene_cfg = loaded["global"]["scene_generation"]
            self.assertEqual(scene_cfg["source"], "density_driven")
            self.assertNotIn("cylinder", scene_cfg)
            self.assertNotIn("task_generation", scene_cfg)
            self.assertNotIn(
                "require_full_vertical_blocking",
                scene_cfg["common_cylinder"])
            self.assertEqual(
                set(scene_cfg["topology_validation"]),
                {"grid_resolution_m", "validation_halo_m"})
            self.assertEqual(
                scene_cfg["vehicle"]["radius_m"],
                loaded["global"]["esdf"]["drone_radius"])
            self.assertNotIn("scenes", loaded)
            self.assertEqual(len(scene_cfg["profiles"]), 12)
            self.assertEqual(
                scene_cfg["common_task_generation"][
                    "max_task_sampling_attempts"],
                2000)

    def test_explicit_profile_subset_is_allowed_for_diagnostics(self):
        dense_only = copy.deepcopy(self.config)
        for profile in dense_only["global"]["scene_generation"]["profiles"]:
            profile["enabled"] = (
                profile["name"] in {
                    "S09_large_dense", "S12_mixed_large_heavy"})
        profiles = _SCENARIO.load_scene_profiles(dense_only)
        self.assertEqual(len(profiles), 2)

    def test_removed_global_obstacle_schema_is_rejected(self):
        old_style = copy.deepcopy(self.config)
        old_style["global"]["obstacle"] = {
            "drone_radius": 0.30,
            "min_gap": 0.20,
        }
        with self.assertRaisesRegex(ValueError, "global.obstacle was removed"):
            _CONFIG._validate_config(old_style)

    def test_full_catalog_without_density_mix_is_rejected(self):
        dense_only = copy.deepcopy(self.config)
        for profile in dense_only["global"]["scene_generation"]["profiles"]:
            profile["density_min"] = 0.20
            profile["density_max"] = 0.22
        with self.assertRaisesRegex(ValueError, "missing"):
            _SCENARIO.load_scene_profiles(dense_only)


if __name__ == "__main__":
    unittest.main()
