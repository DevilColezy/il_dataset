#!/usr/bin/env python3
"""Interactive top-view diagnostics for one expert trajectory."""

from __future__ import print_function

import csv
import json
import math
import os


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _yaw_from_row(row):
    qx = _number(row, "qx", 0.0)
    qy = _number(row, "qy", 0.0)
    qz = _number(row, "qz", 0.0)
    qw = _number(row, "qw", 1.0)
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def _load_obstacles(trajectory_dir):
    meta_path = os.path.join(trajectory_dir, "metadata.json")
    candidates = []
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as stream:
            metadata = json.load(stream)
        manifest = metadata.get("scene_manifest_path", "")
        if manifest:
            candidates.append(manifest)

    output_root = os.path.dirname(os.path.dirname(trajectory_dir))
    for root, _, files in os.walk(output_root):
        if "scene_manifest.json" in files:
            candidates.append(os.path.join(root, "scene_manifest.json"))

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r") as stream:
                    return json.load(stream).get("obstacles", [])
            except (OSError, ValueError):
                continue
    return []


class ExpertBehaviorFigure(object):
    def __init__(self, trajectory_dir):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox

        self.plt = plt
        self.trajectory_dir = os.path.abspath(trajectory_dir)
        self.frames = _read_csv(os.path.join(self.trajectory_dir, "data.csv"))
        self.global_path = _read_csv(
            os.path.join(self.trajectory_dir, "global_path.csv"))
        self.raw_global_path = _read_csv(
            os.path.join(self.trajectory_dir, "raw_global_path.csv"))
        self.local_plans = _read_csv(
            os.path.join(self.trajectory_dir, "local_plans.csv"))
        points = _read_csv(
            os.path.join(self.trajectory_dir, "local_plan_points.csv"))
        self.obstacles = _load_obstacles(self.trajectory_dir)

        if not self.frames:
            raise ValueError("data.csv is missing or empty: {}".format(
                self.trajectory_dir))

        self.points_by_plan = {}
        for point in points:
            plan_id = int(_number(point, "plan_id", -1))
            self.points_by_plan.setdefault(plan_id, []).append(point)
        for plan_points in self.points_by_plan.values():
            plan_points.sort(key=lambda row: int(_number(
                row, "point_index", 0)))

        summary_ids = [
            int(_number(row, "plan_id", -1)) for row in self.local_plans]
        self.plan_ids = []
        for plan_id in summary_ids + sorted(self.points_by_plan):
            if plan_id >= 0 and plan_id not in self.plan_ids:
                self.plan_ids.append(plan_id)
        self.plan_ordinal_by_id = {
            plan_id: ordinal
            for ordinal, plan_id in enumerate(self.plan_ids)}
        self.plan_ordinal_by_frame = {}
        for row in self.local_plans:
            source_frame = int(_number(row, "source_frame_id", -1))
            plan_id = int(_number(row, "plan_id", -1))
            if source_frame >= 0 and plan_id in self.plan_ordinal_by_id:
                self.plan_ordinal_by_frame[source_frame] = \
                    self.plan_ordinal_by_id[plan_id]

        self.fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        self.ax_overall, self.ax_guide, self.ax_local = axes
        self.fig.subplots_adjust(bottom=0.19, wspace=0.25)
        try:
            self.fig.canvas.manager.set_window_title(
                "Expert behavior test")
        except AttributeError:
            pass

        guide_box_ax = self.fig.add_axes([0.38, 0.07, 0.10, 0.045])
        plan_box_ax = self.fig.add_axes([0.72, 0.07, 0.10, 0.045])
        self.guide_box = TextBox(
            guide_box_ax, "Frame index ", initial="0")
        self.plan_box = TextBox(
            plan_box_ax, "Plan ordinal ", initial="0")
        self.status = self.fig.text(
            0.5, 0.015, "", ha="center", va="bottom", color="firebrick")
        self.guide_box.on_submit(self._on_guide_index)
        self.plan_box.on_submit(self._on_plan_index)

        self._draw_overall()
        self._draw_guide(0)
        initial_plan = self._plan_ordinal_for_frame(0)
        self._draw_local(initial_plan)

    def _plan_ordinal_for_frame(self, frame_ordinal):
        row = self.frames[frame_ordinal]
        source_frame = int(_number(
            row, "frame_id", _number(
                row, "episode_frame_index", frame_ordinal)))
        if source_frame in self.plan_ordinal_by_frame:
            return self.plan_ordinal_by_frame[source_frame]
        plan_id = int(_number(row, "plan_id", -1))
        return self.plan_ordinal_by_id.get(plan_id)

    def _base_map(self, ax):
        from matplotlib.patches import Circle

        for obstacle in self.obstacles:
            center = obstacle.get("center", [0.0, 0.0, 0.0])
            radius = float(obstacle.get(
                "radius_m", obstacle.get("radius", 0.0)))
            ax.add_patch(Circle(
                (float(center[0]), float(center[1])), radius,
                facecolor="0.72", edgecolor="0.35", linewidth=0.7,
                alpha=0.65, zorder=0))
        if self.raw_global_path:
            ax.plot([_number(row, "x") for row in self.raw_global_path],
                    [_number(row, "y") for row in self.raw_global_path],
                    ":", color="cornflowerblue", linewidth=1.0,
                    label="mission endpoints (diagnostic)")
        if self.global_path:
            ax.plot([_number(row, "x") for row in self.global_path],
                    [_number(row, "y") for row in self.global_path],
                    "--", color="tab:blue", linewidth=1.6,
                    label="mission axis (not used as Guide)")
        ax.set_xlabel("world x [m]")
        ax.set_ylabel("world y [m]")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    def _draw_overall(self):
        ax = self.ax_overall
        ax.clear()
        self._base_map(ax)
        ax.plot([_number(row, "x") for row in self.frames],
                [_number(row, "y") for row in self.frames],
                color="tab:orange", linewidth=2.0, label="executed flight")
        start = self.frames[0]
        goal = self.frames[-1]
        ax.scatter([_number(start, "x")], [_number(start, "y")],
                   marker="o", s=50, color="green", label="start")
        ax.scatter([_number(goal, "x")], [_number(goal, "y")],
                   marker="x", s=65, color="red", label="final position")
        # ── Straight line start → planned goal ──────────────────
        if self.global_path:
            gp_start = self.global_path[0]
            gp_goal = self.global_path[-1]
            ax.plot([_number(gp_start, "x"), _number(gp_goal, "x")],
                    [_number(gp_start, "y"), _number(gp_goal, "y")],
                    "-.", color="gray", linewidth=1.2, alpha=0.7,
                    label="straight line start→goal")
        ax.set_title("1. Goal mission axis and complete flight")
        ax.legend(loc="best", fontsize=8)

    def _draw_guide(self, index):
        ax = self.ax_guide
        ax.clear()
        self._base_map(ax)
        row = self.frames[index]
        x, y = _number(row, "x"), _number(row, "y")
        gx, gy = _number(row, "guide_x_world"), _number(
            row, "guide_y_world")
        yaw = _yaw_from_row(row)
        arrow_len = 1.2
        ax.arrow(x, y, arrow_len * math.cos(yaw),
                 arrow_len * math.sin(yaw), width=0.045,
                 head_width=0.30, color="tab:red",
                 length_includes_head=True, label="drone heading")
        ax.scatter([x], [y], color="tab:red", s=40, zorder=4)
        if math.isfinite(gx) and math.isfinite(gy):
            ax.plot([x, gx], [y, gy], color="tab:green",
                    linewidth=2.0, marker="o", label="selected guide")
        rejection = str(row.get("guide_rejection_reason", "") or "")
        diagnostic = (
            "\ncandidates={}, rejection={}".format(
                row.get("guide_candidate_count", "?"), rejection)
            if rejection else "")
        ax.set_title(
            "2. Expert guide at frame {}\npath index={}, mode={}{}".format(
                index, row.get("guide_path_index", "?"),
                row.get("guide_mode", "?"), diagnostic))
        ax.legend(loc="best", fontsize=8)

    def _draw_local(self, ordinal):
        ax = self.ax_local
        ax.clear()
        self._base_map(ax)
        if not self.plan_ids:
            ax.set_title("3. No local plans recorded")
            return
        if ordinal is None:
            ax.set_title(
                "3. No local plan associated with this frame")
            return
        plan_id = self.plan_ids[ordinal]
        points = self.points_by_plan.get(plan_id, [])
        summary = next((
            row for row in self.local_plans
            if int(_number(row, "plan_id", -1)) == plan_id), None)
        if points:
            ax.plot([_number(row, "x") for row in points],
                    [_number(row, "y") for row in points],
                    color="tab:purple", linewidth=2.3,
                    marker=".", markersize=3, label="local trajectory")
        if summary is not None:
            sx, sy = _number(summary, "state_x"), _number(
                summary, "state_y")
            yaw = _number(summary, "state_yaw", 0.0)
            ax.arrow(sx, sy, math.cos(yaw), math.sin(yaw),
                     width=0.04, head_width=0.27, color="tab:red",
                     length_includes_head=True)
            ax.scatter([_number(summary, "local_goal_x")],
                       [_number(summary, "local_goal_y")],
                       marker="*", s=90, color="goldenrod",
                       label="local goal")
        ax.set_title(
            "3. Plan ordinal {} (plan_id={}, source frame={}, status={})\n"
            "{} points, duration={:.2f}s, terminal scale={:.2f}".format(
                ordinal, plan_id,
                summary.get("source_frame_id", "?")
                if summary is not None else "?",
                summary.get("status", "?")
                if summary is not None else "?",
                len(points),
                _number(summary, "trajectory_duration_s", 0.0)
                if summary is not None else 0.0,
                _number(summary, "terminal_scale", 1.0)
                if summary is not None else 1.0))
        ax.legend(loc="best", fontsize=8)

    def _on_guide_index(self, value):
        try:
            index = int(value)
            if not 0 <= index < len(self.frames):
                raise IndexError
        except (ValueError, IndexError):
            self.status.set_text(
                "Frame index must be in [0, {}]".format(
                    len(self.frames) - 1))
            self.fig.canvas.draw_idle()
            return
        self.status.set_text("")
        self._draw_guide(index)
        plan_ordinal = self._plan_ordinal_for_frame(index)
        if plan_ordinal is not None:
            self.plan_box.set_val(str(plan_ordinal))
        else:
            self._draw_local(None)
            self.status.set_text(
                "No local plan is associated with frame {}".format(
                    index))
        self.fig.canvas.draw_idle()

    def _on_plan_index(self, value):
        try:
            index = int(value)
            if not 0 <= index < len(self.plan_ids):
                raise IndexError
        except (ValueError, IndexError):
            upper = len(self.plan_ids) - 1
            self.status.set_text(
                "Plan ordinal must be in [0, {}]".format(upper))
            self.fig.canvas.draw_idle()
            return
        self.status.set_text("")
        self._draw_local(index)
        self.fig.canvas.draw_idle()

    def save_overview(self):
        path = os.path.join(
            self.trajectory_dir, "expert_behavior_overview.png")
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def show(self):
        backend = str(self.plt.get_backend()).lower()
        non_gui_backends = {
            "agg", "pdf", "ps", "svg", "template", "cairo", "pgf"}
        if backend in non_gui_backends:
            self.plt.close(self.fig)
            return False
        self.plt.show()
        return True


def visualize(trajectory_dir, show=True):
    figure = ExpertBehaviorFigure(trajectory_dir)
    output = figure.save_overview()
    if show:
        shown = figure.show()
        if not shown:
            print(
                "[ExpertTest] Matplotlib backend '{}' is non-interactive; "
                "saved the figure without opening a window.".format(
                    figure.plt.get_backend()))
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize one collected expert trajectory")
    parser.add_argument("trajectory_dir")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    print(visualize(args.trajectory_dir, show=not args.no_show))


if __name__ == "__main__":
    main()
