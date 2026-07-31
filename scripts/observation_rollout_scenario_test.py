#!/usr/bin/env python3
"""Run configured observation-rollout scenarios without Flightmare."""

from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
import rospy


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from il_config import load_config
from expert_behavior_catalog import (
    load_scenario_catalog,
    select_scenarios,
)
from il_trajectory import observation_conditioned_rollout_path


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _build_cylinder_esdf(global_cfg, obstacles):
    """Build a deterministic analytic ESDF for the configured cylinders."""
    pc_cfg = global_cfg["pointcloud"]
    esdf_cfg = global_cfg["esdf"]
    resolution = float(esdf_cfg["resolution"])
    pc_range = np.asarray(pc_cfg["range"], dtype=np.float64)
    pc_center = np.asarray(pc_cfg["origin"], dtype=np.float64)
    origin = pc_center - 0.5 * pc_range
    shape = tuple(
        int(math.floor(float(extent) / resolution)) + 1
        for extent in pc_range)

    xs = origin[0] + np.arange(shape[0], dtype=np.float64) * resolution
    ys = origin[1] + np.arange(shape[1], dtype=np.float64) * resolution
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    planar = np.full(xx.shape, 1.0e3, dtype=np.float64)
    drone_radius = float(esdf_cfg.get("drone_radius", 0.30))

    for obstacle in obstacles:
        if str(obstacle.get("type", "cylinder")).lower() != "cylinder":
            raise ValueError(
                "Only cylinder obstacles are supported by this offline "
                "test node: {}".format(obstacle))
        center = obstacle.get("center", [])
        radius = float(obstacle.get("radius_m", 0.0))
        if len(center) != 3 or radius <= 0.0:
            raise ValueError("Invalid cylinder obstacle: {}".format(obstacle))
        distance = (
            np.hypot(xx - float(center[0]), yy - float(center[1])) -
            radius - drone_radius)
        planar = np.minimum(planar, distance)

    esdf = np.repeat(
        planar.astype(np.float32)[:, :, None], shape[2], axis=2)
    return esdf, origin, resolution


def _planner_config(global_cfg):
    gp_cfg = global_cfg["planning"]["global_planner"]
    rollout_cfg = dict(gp_cfg.get("observation_rollout", {}))
    depth_cfg = global_cfg.get("depth", {})
    rollout_cfg.setdefault(
        "horizontal_fov_deg", float(depth_cfg.get("fov", 90.0)))
    rollout_cfg.setdefault(
        "observation_range_m", float(depth_cfg.get("max_m", 5.0)))
    return gp_cfg, rollout_cfg


def _format_angle(value):
    return "--" if value is None else "{:.1f}".format(float(value))


def _run_one(name, scenario, global_cfg, print_decisions):
    obstacles = scenario.get("obstacles", [])
    start = scenario.get("start", [])
    goal = scenario.get("goal", [])
    if len(start) != 3 or len(goal) != 3:
        raise ValueError(
            "Scenario '{}' needs 3-D start and goal".format(name))

    esdf, origin, resolution = _build_cylinder_esdf(
        global_cfg, obstacles)
    gp_cfg, rollout_cfg = _planner_config(global_cfg)
    min_clearance = float(gp_cfg.get("min_clearance", 0.30))
    check_spacing = min(
        float(gp_cfg.get("collision_check_spacing_m", 0.05)),
        resolution * 0.5)

    started_at = time.time()
    path, report = observation_conditioned_rollout_path(
        start, goal, esdf, origin, resolution,
        min_clearance, check_spacing, rollout_cfg)
    elapsed_ms = (time.time() - started_at) * 1000.0
    success = path is not None

    if success:
        rospy.loginfo(
            "[ObsRolloutTest] %-24s PASS length=%6.3fm "
            "clearance=%6.3fm decisions=%d checks=%d time=%7.1fms",
            name,
            float(report.get("length", 0.0)),
            float(report.get("worst_clearance", -1.0)),
            len(report.get("decision_events", [])),
            int(report.get("corridor_checks", 0)),
            elapsed_ms)
    else:
        rospy.logerr(
            "[ObsRolloutTest] %-24s FAIL reason=%s time=%.1fms",
            name, report.get("reason", "unknown"), elapsed_ms)

    if print_decisions:
        for index, decision in enumerate(
                report.get("decision_events", []), start=1):
            rospy.loginfo(
                "[ObsRolloutTest]   decision=%d path_idx=%s "
                "pos=%s LEFT=%sdeg RIGHT=%sdeg -> %s (%s)",
                index,
                decision.get("path_index", "?"),
                decision.get("position", []),
                _format_angle(decision.get("left_min_deviation_deg")),
                _format_angle(decision.get("right_min_deviation_deg")),
                decision.get("selected_side", "?"),
                decision.get("selection_reason", "?"))

    del esdf
    return success, report, path


def _show_interactive_results(results, global_cfg):
    """Open one interactive figure for reviewing every planned scenario."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.widgets import Button, TextBox

    if not results:
        return

    drone_radius = float(
        global_cfg.get("esdf", {}).get("drone_radius", 0.30))
    min_clearance = float(
        global_cfg["planning"]["global_planner"].get(
            "min_clearance", 0.30))
    observation_range = float(
        global_cfg.get("depth", {}).get("max_m", 5.0))

    figure, axis = plt.subplots(figsize=(11.5, 8.0))
    figure.canvas.manager.set_window_title(
        "Observation rollout scenario test")
    plt.subplots_adjust(bottom=0.18, top=0.90)
    previous_axis = plt.axes([0.07, 0.055, 0.12, 0.055])
    next_axis = plt.axes([0.205, 0.055, 0.12, 0.055])
    scenario_axis = plt.axes([0.43, 0.055, 0.35, 0.055])
    previous_button = Button(previous_axis, "Previous")
    next_button = Button(next_axis, "Next")
    scenario_box = TextBox(
        scenario_axis, "Scenario ", initial=results[0]["name"])
    state = {"index": 0, "updating_box": False}

    def draw_arrow(position, goal, angle_deg, side, selected):
        if angle_deg is None:
            return
        delta = np.asarray(goal[:2]) - np.asarray(position[:2])
        if float(np.linalg.norm(delta)) < 1e-9:
            return
        base = math.atan2(delta[1], delta[0])
        signed = -float(angle_deg) if side == "RIGHT" else float(angle_deg)
        heading = base + math.radians(signed)
        arrow_length = 0.9 if selected else 0.65
        color = "#d62728" if side == "RIGHT" else "#1f77b4"
        axis.arrow(
            position[0], position[1],
            arrow_length * math.cos(heading),
            arrow_length * math.sin(heading),
            width=0.025 if selected else 0.012,
            head_width=0.16, head_length=0.20,
            length_includes_head=True,
            color=color, alpha=0.95 if selected else 0.45,
            zorder=8)

    def render():
        item = results[state["index"]]
        scenario = item["scenario"]
        report = item["report"]
        path = item["path"]
        start = np.asarray(scenario["start"], dtype=np.float64)
        goal = np.asarray(scenario["goal"], dtype=np.float64)
        obstacles = scenario.get("obstacles", [])

        axis.clear()
        axis.plot(
            [start[0], goal[0]], [start[1], goal[1]],
            linestyle="-.", color="0.65", linewidth=1.2,
            label="straight start-goal")
        for obstacle_index, obstacle in enumerate(obstacles):
            center = obstacle["center"]
            radius = float(obstacle["radius_m"])
            axis.add_patch(Circle(
                (center[0], center[1]), radius,
                facecolor="0.75", edgecolor="0.35",
                linewidth=1.2, alpha=0.85,
                label=("physical obstacle"
                       if obstacle_index == 0 else None)))
            axis.add_patch(Circle(
                (center[0], center[1]),
                radius + drone_radius + min_clearance,
                fill=False, edgecolor="#ff7f0e",
                linestyle=":", linewidth=1.1, alpha=0.75,
                label=("vehicle + global clearance"
                       if obstacle_index == 0 else None)))
            axis.text(
                center[0], center[1],
                str(obstacle.get("id", obstacle_index)),
                ha="center", va="center", fontsize=8,
                color="0.20", zorder=6)

        if path is not None:
            points = np.asarray(path, dtype=np.float64)
            axis.plot(
                points[:, 0], points[:, 1],
                color="#9467bd", linewidth=2.4,
                label="observation rollout", zorder=7)

        axis.scatter(
            [start[0]], [start[1]], s=75, color="green",
            edgecolors="white", linewidths=0.8,
            label="start", zorder=9)
        axis.scatter(
            [goal[0]], [goal[1]], s=90, color="red",
            marker="x", linewidths=2.0,
            label="goal", zorder=9)

        decisions = report.get("decision_events", [])
        for decision_index, decision in enumerate(decisions, start=1):
            position = np.asarray(
                decision.get("position", [0.0, 0.0, 0.0]),
                dtype=np.float64)
            selected = decision.get("selected_side", "")
            axis.scatter(
                [position[0]], [position[1]],
                marker="*", s=125, color="#f1c40f",
                edgecolors="0.25", linewidths=0.7, zorder=10,
                label=("local decision"
                       if decision_index == 1 else None))
            axis.add_patch(Circle(
                (position[0], position[1]), observation_range,
                fill=False, edgecolor="#17becf",
                linestyle="--", linewidth=0.7, alpha=0.16))
            draw_arrow(
                position, goal,
                decision.get("left_min_deviation_deg"),
                "LEFT", selected == "LEFT")
            draw_arrow(
                position, goal,
                decision.get("right_min_deviation_deg"),
                "RIGHT", selected == "RIGHT")
            axis.annotate(
                "#{}  L={}deg  R={}deg  -> {}".format(
                    decision_index,
                    _format_angle(
                        decision.get("left_min_deviation_deg")),
                    _format_angle(
                        decision.get("right_min_deviation_deg")),
                    selected),
                (position[0], position[1]),
                xytext=(7, 9), textcoords="offset points",
                fontsize=8, color="0.15",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white", alpha=0.78,
                    edgecolor="0.75"))

        all_x = [float(start[0]), float(goal[0])]
        all_y = [float(start[1]), float(goal[1])]
        for obstacle in obstacles:
            center = obstacle["center"]
            radius = (
                float(obstacle["radius_m"]) +
                drone_radius + min_clearance)
            all_x.extend([float(center[0]) - radius,
                          float(center[0]) + radius])
            all_y.extend([float(center[1]) - radius,
                          float(center[1]) + radius])
        if path is not None:
            all_x.extend(points[:, 0].tolist())
            all_y.extend(points[:, 1].tolist())
        x_margin = max(2.0, 0.12 * (max(all_x) - min(all_x) + 1.0))
        y_margin = max(2.0, 0.06 * (max(all_y) - min(all_y) + 1.0))
        axis.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        axis.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")

        if item["success"]:
            status = (
                "PASS | length={:.3f} m | clearance={:.3f} m | "
                "decisions={} | checks={}".format(
                    float(report.get("length", 0.0)),
                    float(report.get("worst_clearance", -1.0)),
                    len(decisions),
                    int(report.get("corridor_checks", 0))))
        else:
            status = "FAIL | reason={}".format(
                report.get("reason", "unknown"))
        axis.set_title(
            "{}/{}  {}\n{}\n{}".format(
                state["index"] + 1, len(results), item["name"],
                scenario.get("description", ""), status),
            fontsize=11)
        axis.legend(loc="best", fontsize=8)
        state["updating_box"] = True
        scenario_box.set_val(item["name"])
        state["updating_box"] = False
        figure.canvas.draw_idle()

    def change_index(offset):
        state["index"] = (
            state["index"] + offset) % len(results)
        render()

    def select_scenario(text):
        if state["updating_box"]:
            return
        query = str(text).strip()
        for index, item in enumerate(results):
            if item["name"] == query:
                state["index"] = index
                render()
                return
        try:
            numeric_index = int(query) - 1
        except ValueError:
            numeric_index = -1
        if 0 <= numeric_index < len(results):
            state["index"] = numeric_index
            render()
        else:
            rospy.logwarn(
                "[ObsRolloutTest] Unknown plot scenario/index: %s", query)

    previous_button.on_clicked(lambda _event: change_index(-1))
    next_button.on_clicked(lambda _event: change_index(1))
    scenario_box.on_submit(select_scenario)

    def on_key(event):
        if event.key in ("left", "pageup"):
            change_index(-1)
        elif event.key in ("right", "pagedown"):
            change_index(1)

    figure.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


def main():
    rospy.init_node(
        "observation_rollout_scenario_test", anonymous=False)
    cfg = load_config()
    global_cfg = cfg["global"]
    scenario_file = rospy.get_param(
        "~scenario_file",
        os.path.join(
            os.path.dirname(SCRIPT_DIR), "config",
            "expert_behavior_scenarios.yaml"))
    selector = rospy.get_param("~scenario", "all")
    print_decisions = _as_bool(
        rospy.get_param("~print_decisions", True))
    show_plot = _as_bool(
        rospy.get_param("~show_plot", True))
    fail_on_error = _as_bool(
        rospy.get_param("~fail_on_error", True))

    resolved, _, scenarios, suites, _ = load_scenario_catalog(
        scenario_file)
    names = select_scenarios(selector, scenarios, suites)
    rospy.loginfo(
        "[ObsRolloutTest] catalog=%s scenarios=%s planner=%s",
        resolved, names,
        global_cfg["planning"]["global_planner"].get(
            "algorithm", ""))

    passed = []
    failed = []
    results = []
    for name in names:
        try:
            success, report, path = _run_one(
                name, scenarios[name], global_cfg, print_decisions)
        except Exception as exc:
            rospy.logerr(
                "[ObsRolloutTest] %-24s ERROR %s", name, exc)
            success = False
            report = {"reason": "exception:{}".format(exc)}
            path = None
        results.append({
            "name": name,
            "scenario": scenarios[name],
            "success": success,
            "report": report,
            "path": path})
        if success:
            passed.append(name)
        else:
            failed.append((name, report.get("reason", "unknown")))

    rospy.loginfo(
        "[ObsRolloutTest] SUMMARY passed=%d/%d failed=%d",
        len(passed), len(names), len(failed))
    if passed:
        rospy.loginfo(
            "[ObsRolloutTest] passed: %s", ", ".join(passed))
    for name, reason in failed:
        rospy.logerr(
            "[ObsRolloutTest] failed: %s reason=%s", name, reason)

    if show_plot:
        rospy.loginfo(
            "[ObsRolloutTest] Opening interactive review for %d scenarios.",
            len(results))
        _show_interactive_results(results, global_cfg)

    if failed and fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
