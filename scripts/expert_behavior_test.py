#!/usr/bin/env python3
"""Collect exactly one expert trajectory, then open its diagnostic figure."""

from __future__ import print_function

import glob
import os
import subprocess
import sys

import rospy

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


def _find_recorded_trajectory(output_dir):
    candidates = glob.glob(
        os.path.join(output_dir, "**", "data.csv"), recursive=True)
    if not candidates:
        return None
    newest = max(candidates, key=os.path.getmtime)
    return os.path.dirname(newest)


def main():
    rospy.init_node("expert_behavior_test", anonymous=False)
    cfg = load_config()
    distribution = rospy.get_param("~distribution", "small_sparse")
    output_dir = rospy.get_param(
        "~output_dir",
        os.path.join(os.path.dirname(os.path.dirname(__file__)),
                     "dataset", "expert_behavior_test"))
    seed = int(rospy.get_param("~seed", 12345))
    show_plot_param = rospy.get_param("~show_plot", True)
    show_plot = (show_plot_param if isinstance(show_plot_param, bool)
                 else str(show_plot_param).lower() in ("1", "true", "yes"))

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
