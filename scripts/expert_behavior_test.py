#!/usr/bin/env python3
"""Collect exactly one expert trajectory, then open its diagnostic figure."""

from __future__ import print_function

import glob
import os
import subprocess
import sys

import rospy
import yaml

# catkin's generated launcher executes this source file from devel/lib.  Keep
# the package's script modules importable in both source and devel spaces.
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from il_config import load_config
from il_manager import ILManager


PRESETS = {
    "small_sparse": "S01_small_sparse",
    "medium_dense": "S06_medium_dense",
    "large_sparse": "S07_large_sparse",
    "mixed_dense": "S12_mixed_large_heavy",
}


def _configure_single_test(cfg, distribution, output_dir, seed):
    if distribution not in PRESETS:
        raise ValueError(
            "Unknown distribution '{}'; choose one of {}".format(
                distribution, sorted(PRESETS)))

    g = cfg["global"]
    scene_cfg = g["scene_generation"]
    selected_name = PRESETS[distribution]
    selected = None
    for profile in scene_cfg.get("profiles", []):
        profile["enabled"] = profile.get("name") == selected_name
        if profile["enabled"]:
            selected = profile
    if selected is None:
        raise ValueError(
            "Profile '{}' is absent from the base config".format(
                selected_name))

    selected["scene_count"] = 1
    selected.setdefault("task_generation", {})["tasks_per_scene"] = 1
    scene_cfg["seed"] = int(seed)
    scene_cfg.setdefault("execution", {})[
        "stop_after_all_profiles"] = True
    g["start_goal"]["num_pairs_per_config"] = 1
    g.setdefault("dagger", {})["enabled"] = False
    g["output_dir"] = os.path.abspath(os.path.expanduser(output_dir))
    return selected_name


def _load_scenario_catalog(scenario_file):
    path = os.path.abspath(os.path.expanduser(scenario_file))
    with open(path, "r") as stream:
        catalog = yaml.safe_load(stream) or {}
    scenarios = catalog.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError(
            "Scenario catalog contains no 'scenarios': {}".format(path))
    return path, catalog


def _configure_fixed_scenario(cfg, scenario_name, scenario_file,
                              output_dir, seed):
    scenario_file, catalog = _load_scenario_catalog(scenario_file)
    scenarios = catalog["scenarios"]
    if scenario_name not in scenarios:
        raise ValueError(
            "Unknown scenario '{}'; choose one of {}".format(
                scenario_name, sorted(scenarios)))

    scenario = scenarios[scenario_name]
    if not isinstance(scenario, dict):
        raise ValueError(
            "Scenario '{}' must be a mapping".format(scenario_name))
    start = scenario.get("start")
    goal = scenario.get("goal")
    obstacles = scenario.get("obstacles")
    if (not isinstance(start, list) or len(start) != 3 or
            not isinstance(goal, list) or len(goal) != 3 or
            not isinstance(obstacles, list) or not obstacles):
        raise ValueError(
            "Scenario '{}' needs 3-D start/goal and at least one obstacle"
            .format(scenario_name))

    g = cfg["global"]
    scene_cfg = g["scene_generation"]
    scene_cfg["enabled"] = True
    scene_cfg["source"] = "procedural_yaml"
    scene_cfg["seed"] = int(seed)
    scene_cfg["fixed_scene_name"] = scenario_name
    scene_cfg["fixed_obstacles"] = obstacles
    # Legacy (non-profile) FSM uses this as its scene-count limit.
    scene_cfg["max_scene_generation_attempts"] = 1
    for profile in scene_cfg.get("profiles", []):
        profile["enabled"] = False

    execution = scene_cfg.setdefault("execution", {})
    execution["max_generation_attempts_per_scene"] = 1
    execution["max_task_sampling_attempts_per_scene"] = 1
    execution["stop_after_all_profiles"] = True

    task_cfg = scene_cfg.setdefault("task_generation", {})
    task_cfg.update({
        "enabled": True,
        "tasks_per_scene": 1,
        "max_task_sampling_attempts": 1,
        "fixed_tasks": [{"start": start, "goal": goal}],
        "start_height_min_m": float(start[2]),
        "start_height_max_m": float(start[2]),
        "goal_height_min_m": float(goal[2]),
        "goal_height_max_m": float(goal[2]),
        "maximum_start_goal_height_difference_m":
            abs(float(goal[2]) - float(start[2])) + 1.0e-6,
        "start_clearance_m": 0.30,
        "goal_clearance_m": 0.30,
        "minimum_start_goal_distance_m": 0.0,
        "maximum_start_goal_distance_m": 1000.0,
        "require_direct_path_blocked": bool(
            scenario.get("require_direct_path_blocked", True)),
        "minimum_direct_blocker_count": int(
            scenario.get("minimum_direct_blocker_count", 1)),
        "maximum_direct_blocker_count": int(
            scenario.get(
                "maximum_direct_blocker_count", max(1, len(obstacles)))),
        "require_astar_reachable": True,
        "minimum_detour_ratio": 1.0,
        "maximum_detour_ratio": 1000.0,
    })

    g["start_goal"]["num_pairs_per_config"] = 1
    g.setdefault("dagger", {})["enabled"] = False
    g["output_dir"] = os.path.abspath(os.path.expanduser(output_dir))
    return scenario, scenario_file


def _find_recorded_trajectory(output_dir):
    candidates = glob.glob(
        os.path.join(output_dir, "**", "data.csv"), recursive=True)
    nonempty_candidates = []
    for path in candidates:
        try:
            with open(path, "r") as stream:
                # A schema header alone is not a recorded trajectory.
                next(stream, None)
                if next(stream, None) is not None:
                    nonempty_candidates.append(path)
        except OSError:
            continue
    if not nonempty_candidates:
        return None
    newest = max(nonempty_candidates, key=os.path.getmtime)
    return os.path.dirname(newest)


def main():
    rospy.init_node("expert_behavior_test", anonymous=False)
    cfg = load_config()
    distribution = rospy.get_param("~distribution", "small_sparse")
    scenario = str(rospy.get_param("~scenario", "")).strip()
    scenario_file = rospy.get_param(
        "~scenario_file",
        os.path.join(os.path.dirname(SCRIPT_DIR), "config",
                     "expert_behavior_scenarios.yaml"))
    output_dir = rospy.get_param(
        "~output_dir",
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "dataset", "expert_behavior_test"))
    seed = int(rospy.get_param("~seed", 12345))
    show_plot_param = rospy.get_param("~show_plot", True)
    show_plot = (show_plot_param if isinstance(show_plot_param, bool)
                 else str(show_plot_param).lower() in ("1", "true", "yes"))

    if scenario:
        scenario_cfg, resolved_scenario_file = _configure_fixed_scenario(
            cfg, scenario, scenario_file, output_dir, seed)
        rospy.loginfo(
            "[ExpertTest] scenario=%s seed=%d catalog=%s output=%s",
            scenario, seed, resolved_scenario_file,
            cfg["global"]["output_dir"])
        rospy.loginfo(
            "[ExpertTest] purpose: %s",
            scenario_cfg.get("description", ""))
    else:
        profile_name = _configure_single_test(
            cfg, distribution, output_dir, seed)
        rospy.loginfo(
            "[ExpertTest] distribution=%s profile=%s seed=%d output=%s",
            distribution, profile_name, seed, cfg["global"]["output_dir"])

    manager = ILManager(cfg)
    manager.run()

    trajectory_dir = _find_recorded_trajectory(
        cfg["global"]["output_dir"])
    if trajectory_dir is None:
        rospy.logerr(
            "[ExpertTest] No completed or rejected trajectory was recorded.")
        return

    rospy.loginfo("[ExpertTest] Opening diagnostics for %s", trajectory_dir)
    visualizer = os.path.join(SCRIPT_DIR, "expert_behavior_visualizer.py")
    command = [sys.executable, visualizer, trajectory_dir]
    if not show_plot:
        command.append("--no-show")
    return_code = subprocess.call(command)
    if return_code != 0:
        rospy.logerr(
            "[ExpertTest] Visualizer exited with code %d", return_code)
    overview = os.path.join(
        trajectory_dir, "expert_behavior_overview.png")
    rospy.loginfo("[ExpertTest] Overview saved to %s", overview)


if __name__ == "__main__":
    main()
