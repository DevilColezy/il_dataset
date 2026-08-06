#!/usr/bin/env python3
"""Interactive frame-by-frame trajectory replay.

Usage:
    python trajectory_replay.py <trajectory_dir>

Controls:
    Left/Right arrows   step one frame
    Up/Down arrows      skip 10 frames
    Home/End            go to first/last frame
    Space               toggle auto-play
    q/w                 adjust auto-play speed
    s                   print frame summary to terminal
"""

from __future__ import print_function, division
import csv
import json
import math
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button

FREE_COLOR = (0.95, 0.95, 0.95)
OCC_COLOR = (0.3, 0.3, 0.3)
UNK_COLOR = (0.7, 0.7, 0.85)
DRONE_COLOR = (0.0, 0.6, 0.0)
GOAL_COLOR = (1.0, 0.0, 0.0)
GUIDE_COLOR = (0.0, 0.3, 1.0)
PATH_COLOR = (0.8, 0.0, 0.8)
PLAN_FAIL_COLOR = (1.0, 0.2, 0.2)
DANGER_COLOR = (1.0, 0.05, 0.05)
SCAN_COLOR = (1.0, 0.5, 0.0)
BYPASS_COLOR = (0.0, 0.7, 0.7)
SEEK_COLOR = (0.0, 0.6, 0.0)
HOLD_COLOR = (0.0, 1.0, 0.0)

# A spline sample whose clearance dips below this (ESDF value; the vehicle
# radius is already subtracted) is drawn in red as a danger segment.
CLEARANCE_WARN_M = 0.15

MACRO_COLORS = {
    "GOAL_SEEK": SEEK_COLOR,
    "BYPASS": BYPASS_COLOR,
    "PROBE": SCAN_COLOR,
    # V15: legacy rows (pre-guide-line) kept for reading old CSVs.
    "GOAL_HOLD": HOLD_COLOR,
    "BYPASS_LEFT": BYPASS_COLOR,
    "BYPASS_RIGHT": (0.7, 0.0, 0.7),
    "ACTIVE_SCAN_LEFT": SCAN_COLOR,
    "ACTIVE_SCAN_RIGHT": (1.0, 0.3, 0.0),
}


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as stream:
        return list(csv.DictReader(stream))


def _expect_float(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _load_obstacles(traj_dir):
    """Load obstacles from metadata or scene_manifest."""
    meta_path = os.path.join(traj_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as stream:
            metadata = json.load(stream)
        manifest = metadata.get("scene_manifest_path", "")
        if manifest and os.path.isfile(manifest):
            with open(manifest, "r") as stream:
                scene = json.load(stream)
                return scene.get("obstacles", [])
    return []


def _yaw_direction(yaw_rad):
    return (-math.sin(yaw_rad), math.cos(yaw_rad))


class TrajectoryReplay:
    def __init__(self, traj_dir):
        self.traj_dir = os.path.abspath(traj_dir)
        self.frames = _read_csv(os.path.join(traj_dir, "data.csv"))
        self.plans = _read_csv(os.path.join(traj_dir, "local_plans.csv"))
        self.plan_points = _read_csv(
            os.path.join(traj_dir, "local_plan_points.csv"))
        self.obstacles = _load_obstacles(traj_dir)

        if not self.frames:
            raise ValueError("data.csv is empty or missing")

        self.n_frames = len(self.frames)
        self.current = 0
        self.playing = False
        self.play_speed = 10  # frames per second
        self.show_all_plans = False
        self.plan_by_source = {}
        for plan in self.plans:
            sf = int(_expect_float(plan, "source_frame_id", -1))
            if sf >= 0:
                self.plan_by_source.setdefault(sf, []).append(plan)
        # Dense B-spline samples, grouped by plan_id and ordered by t.
        self.plan_curves = {}
        for pt in self.plan_points:
            pid = int(_expect_float(pt, "plan_id", -1))
            if pid >= 0:
                self.plan_curves.setdefault(pid, []).append(pt)
        for pid in self.plan_curves:
            self.plan_curves[pid].sort(key=lambda p: _expect_float(p, "t"))

        # Compute path bounds
        xs = [_expect_float(r, "x") for r in self.frames]
        ys = [_expect_float(r, "y") for r in self.frames]
        valid_xs = [x for x in xs if np.isfinite(x)]
        valid_ys = [y for y in ys if np.isfinite(y)]
        # Include every planned-curve sample so trajectories near the edge
        # of the view are not clipped.
        for pt in self.plan_points:
            px = _expect_float(pt, "x")
            py = _expect_float(pt, "y")
            if np.isfinite(px):
                valid_xs.append(px)
            if np.isfinite(py):
                valid_ys.append(py)
        gx = _expect_float(self.frames[0], "goal_x", valid_xs[0])
        gy = _expect_float(self.frames[0], "goal_y", valid_ys[0])
        if np.isfinite(gx):
            valid_xs.append(gx)
        if np.isfinite(gy):
            valid_ys.append(gy)
        if self.obstacles:
            for obs in self.obstacles:
                cx = float(obs.get("center", [0, 0, 0])[0])
                cy = float(obs.get("center", [0, 0, 0])[1])
                r = float(obs.get("radius_m", 0.3))
                valid_xs.extend([cx - r, cx + r])
                valid_ys.extend([cy - r, cy + r])
        xmin, xmax = min(valid_xs) - 1, max(valid_xs) + 1
        ymin, ymax = min(valid_ys) - 1, max(valid_ys) + 1
        span = max(xmax - xmin, ymax - ymin) * 0.05
        self.bounds = (xmin - span, xmax + span, ymin - span, ymax + span)

        self.fig, (self.ax_main, self.ax_info) = plt.subplots(
            1, 2, figsize=(16, 8),
            gridspec_kw={"width_ratios": [3, 1]})
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.ax_info.axis("off")
        self._timer = None
        self._draw()

    def _on_key(self, event):
        if event.key == "right":
            self.current = min(self.n_frames - 1, self.current + 1)
            self._draw()
        elif event.key == "left":
            self.current = max(0, self.current - 1)
            self._draw()
        elif event.key == "up":
            self.current = min(self.n_frames - 1, self.current + 10)
            self._draw()
        elif event.key == "down":
            self.current = max(0, self.current - 10)
            self._draw()
        elif event.key == "home":
            self.current = 0
            self._draw()
        elif event.key == "end":
            self.current = self.n_frames - 1
            self._draw()
        elif event.key == " ":
            self.playing = not self.playing
            if self.playing:
                self._start_play()
            else:
                self._stop_play()
        elif event.key == "q":
            self.play_speed = max(1, self.play_speed // 2)
        elif event.key == "w":
            self.play_speed = min(120, self.play_speed * 2)
        elif event.key == "s":
            self._print_summary()
        elif event.key == "a":
            self.show_all_plans = not self.show_all_plans
            self._draw()

    def _start_play(self):
        self._play_tick()

    def _stop_play(self):
        if self._timer is not None:
            self._timer.remove_callbacks()
            self._timer = None

    def _play_tick(self):
        if not self.playing:
            return
        self.current = (self.current + 1) % self.n_frames
        self._draw()
        interval = 1000 // self.play_speed
        self._timer = self.fig.canvas.new_timer(interval=interval)
        self._timer.single_shot = True
        self._timer.add_callback(self._play_tick)
        self._timer.start()

    def _print_summary(self):
        row = self.frames[self.current]
        print("\n=== Frame {}/{} ({}) ===".format(
            self.current, self.n_frames - 1,
            row.get("episode_frame_index", "?")))
        keys = ["macro_mode", "macro_decision_reason", "planner_status",
                "planner_success", "planner_min_clearance",
                "distance_to_final_goal", "local_feasible",
                "local_progress_rate", "plan_is_fresh",
                "guide_azimuth_rad", "guide_distance_m",
                "planner_mode", "guide_mode", "recovery_entered"]
        for k in keys:
            v = row.get(k, "?")
            if k.endswith("_rad") and v not in ("?", ""):
                try:
                    v = "{:.1f}\u00b0".format(math.degrees(float(v)))
                except Exception:
                    pass
            print("  {}: {}".format(k, v))

    def _plot_plan_curve(self, plan, alpha=1.0, bright=False):
        """Draw one plan's dense B-spline trajectory.

        The curve is drawn with a single LineCollection (segment colours)
        so samples whose ESDF clearance dips below CLEARANCE_WARN_M turn
        red.  Successful plans are purple, rejected plans red.  On the
        bright (current-frame) curve, short ticks show the planned world
        yaw along the spline.  Returns True when a curve was drawn.
        """
        pid = int(_expect_float(plan, "plan_id", -1))
        curve = self.plan_curves.get(pid, [])
        if len(curve) < 2:
            return False
        xs = np.array([_expect_float(p, "x") for p in curve], dtype=np.float64)
        ys = np.array([_expect_float(p, "y") for p in curve], dtype=np.float64)
        cl = np.array([_expect_float(p, "clearance") for p in curve],
                      dtype=np.float64)
        yaws = np.array([_expect_float(p, "yaw") for p in curve],
                        dtype=np.float64)
        success = str(plan.get("success", "0")) == "True"
        base = PATH_COLOR if success else PLAN_FAIL_COLOR
        segments = np.stack([
            np.column_stack([xs[:-1], ys[:-1]]),
            np.column_stack([xs[1:], ys[1:]])], axis=1)
        colors = np.tile(base, (len(segments), 1))
        danger = (cl[:-1] < CLEARANCE_WARN_M) | (cl[1:] < CLEARANCE_WARN_M)
        colors[danger] = DANGER_COLOR
        self.ax_main.add_collection(LineCollection(
            segments, colors=colors,
            linewidths=2.4 if bright else 1.0,
            alpha=alpha, capstyle="round"))
        # Planned-yaw heading ticks along the current plan only.
        if bright and len(xs) > 0:
            step = max(1, len(xs) // 20)
            for i in range(0, len(xs), step):
                if not (np.isfinite(xs[i]) and np.isfinite(ys[i]) and
                        np.isfinite(yaws[i])):
                    continue
                dx, dy = _yaw_direction(yaws[i])
                self.ax_main.plot([xs[i], xs[i] + dx * 0.25],
                                  [ys[i], ys[i] + dy * 0.25],
                                  color=base, lw=1.0, alpha=0.7)
        if bright:
            self.ax_main.plot(xs, ys, ".", color=base, ms=2.5,
                              alpha=min(1.0, alpha * 1.2))
        return True

    def _draw(self):
        self.ax_main.clear()
        self.ax_info.clear()
        self.ax_info.axis("off")

        row = self.frames[self.current]
        x = _expect_float(row, "x")
        y = _expect_float(row, "y")
        yaw = _expect_float(row, "executed_next_yaw",
                            _expect_float(row, "state_angular_velocity_z_body"))
        goal_x = _expect_float(row, "goal_x")
        goal_y = _expect_float(row, "goal_y")
        guide_x = _expect_float(row, "guide_x_world")
        guide_y = _expect_float(row, "guide_y_world")
        macro_mode = row.get("macro_mode", "?")
        planner_status = row.get("planner_status", "?")
        planner_success = row.get("planner_success", "0")
        dist_to_goal = _expect_float(row, "distance_to_final_goal")
        clearance = _expect_float(row, "planner_min_clearance")
        reason = row.get("macro_decision_reason", "?")
        progress = _expect_float(row, "global_progress_s")
        coll = row.get("collision", "0")

        # Obstacles
        for obs in self.obstacles:
            cx = float(obs.get("center", [0, 0, 0])[0])
            cy = float(obs.get("center", [0, 0, 0])[1])
            r = float(obs.get("radius_m", 0.3))
            self.ax_main.add_patch(Circle((cx, cy), r, fill=True,
                                          color=OCC_COLOR, ec="black", lw=0.5))

        # Drone path (trail)
        trail = min(self.current + 1, 100)
        trail_xs = [_expect_float(self.frames[i], "x")
                    for i in range(max(0, self.current - trail), self.current + 1)]
        trail_ys = [_expect_float(self.frames[i], "y")
                    for i in range(max(0, self.current - trail), self.current + 1)]
        valid_trail = [(xx, yy) for xx, yy in zip(trail_xs, trail_ys)
                       if np.isfinite(xx) and np.isfinite(yy)]
        if valid_trail:
            txs, tys = zip(*valid_trail)
            self.ax_main.plot(txs, tys, color=DRONE_COLOR, lw=1.5, alpha=0.5)

        # Drone position
        if np.isfinite(x) and np.isfinite(y):
            self.ax_main.plot(x, y, "o", color=DRONE_COLOR, ms=8, zorder=5)
            # FOV cone
            if np.isfinite(yaw):
                dx, dy = _yaw_direction(yaw)
                fov_half = math.radians(45.0)
                for a in [-fov_half, fov_half]:
                    ax = math.cos(yaw + a + math.pi / 2.0)
                    ay = math.sin(yaw + a + math.pi / 2.0)
                    self.ax_main.plot([x, x + ax * 5.0], [y, y + ay * 5.0],
                                      color=DRONE_COLOR, lw=0.5, alpha=0.3)

        # Goal
        if np.isfinite(goal_x) and np.isfinite(goal_y):
            self.ax_main.plot(goal_x, goal_y, "r*", ms=15, zorder=4)
            self.ax_main.annotate("GOAL", (goal_x, goal_y),
                                  xytext=(5, 5), textcoords="offset points",
                                  color="red", fontsize=9, fontweight="bold")

        # Guide
        if np.isfinite(guide_x) and np.isfinite(guide_y):
            color = MACRO_COLORS.get(macro_mode, "gray")
            self.ax_main.plot(guide_x, guide_y, "D", color=color,
                              ms=10, zorder=3, markeredgecolor="black")
            if np.isfinite(x) and np.isfinite(y):
                guide_dx = guide_x - x
                guide_dy = guide_y - y
                gdist = math.sqrt(guide_dx * guide_dx + guide_dy * guide_dy)
                if gdist > 0.01:
                    self.ax_main.arrow(x, y, guide_dx * 0.85, guide_dy * 0.85,
                                       head_width=0.15, head_length=0.2,
                                       fc=color, ec=color, alpha=0.5,
                                       length_includes_head=True)

        # Planned trajectory curves (dense B-spline from
        # local_plan_points.csv).  The current frame's plan(s) are drawn
        # bright; 'a' overlays every plan in the episode faintly.
        sf = int(row.get("frame_id",
                         row.get("episode_frame_index", self.current)))
        plans_here = self.plan_by_source.get(sf, [])
        if not plans_here:
            plans_here = self.plan_by_source.get(self.current, [])
        if self.show_all_plans:
            for other in self.plans:
                self._plot_plan_curve(other, alpha=0.12, bright=False)
        drawn_curve = False
        for plan in plans_here:
            if self._plot_plan_curve(plan, alpha=0.9, bright=True):
                drawn_curve = True
        if not drawn_curve:
            # Fallback: no dense sidecar — draw the guide→local_goal line.
            for plan in plans_here:
                gx_p = _expect_float(plan, "guide_x")
                gy_p = _expect_float(plan, "guide_y")
                lgx = _expect_float(plan, "local_goal_x")
                lgy = _expect_float(plan, "local_goal_y")
                if (np.isfinite(gx_p) and np.isfinite(gy_p) and
                        np.isfinite(lgx) and np.isfinite(lgy)):
                    success = plan.get("success", "0")
                    plan_color = PATH_COLOR if success == "True" else "red"
                    self.ax_main.plot([gx_p, lgx], [gy_p, lgy],
                                      "--", color=plan_color, lw=1, alpha=0.6)

        self.ax_main.set_xlim(self.bounds[0], self.bounds[1])
        self.ax_main.set_ylim(self.bounds[2], self.bounds[3])
        self.ax_main.set_aspect("equal")
        self.ax_main.set_xlabel("X (world, m)")
        self.ax_main.set_ylabel("Y (world, m)")
        self.ax_main.grid(True, alpha=0.3)

        # Info panel
        info_lines = [
            "=== Frame {}/{} ===".format(self.current, self.n_frames - 1),
            "Episode frame: {}".format(
                row.get("episode_frame_index", "?")),
            "Time: {:.2f}s".format(
                _expect_float(row, "trajectory_time_s")),
            "",
            "--- Position ---",
            "X: {:.3f} m".format(x),
            "Y: {:.3f} m".format(y),
            "Z: {:.3f} m".format(_expect_float(row, "z")),
            "Yaw: {:.1f}\u00b0".format(math.degrees(yaw) if np.isfinite(yaw) else 0),
            "Dist to goal: {:.3f} m".format(
                dist_to_goal if np.isfinite(dist_to_goal) else 0),
            "",
            "--- Macro Expert ---",
            "Mode: {}".format(macro_mode),
            "Reason: {}".format(reason[:60] if reason else "?"),
            "Progress: {:.3f}".format(
                progress if np.isfinite(progress) else 0),
            "",
            "--- Planner ---",
            "Status: {}".format(planner_status),
            "Success: {}".format(planner_success),
            "Clearance: {:.4f} m".format(
                clearance if np.isfinite(clearance) else 0),
            "Plan ids: {}".format(
                ",".join(str(int(_expect_float(p, "plan_id", -1)))
                          for p in plans_here) or "-"),
            "Plan ok: {}".format(
                ",".join("1" if str(p.get("success", "0")) == "True"
                           else "0" for p in plans_here) or "-"),
            "Traj pts: {}".format(
                ",".join(str(len(self.plan_curves.get(
                    int(_expect_float(p, "plan_id", -1)), [])))
                          for p in plans_here) or "-"),
            "",
            "--- Curve legend ---",
            "purple=plan ok  red=rejected",
            "dark red=clearance < {:.2f} m".format(CLEARANCE_WARN_M),
            "tick=planned yaw  (a=all plans)",
            "",
            "--- Diagnostics ---",
            "Collision: {}".format(coll),
            "Planner mode: {}".format(row.get("planner_mode", "?")),
            "Guide mode: {}".format(row.get("guide_mode", "?")),
            "Safety override: {}".format(row.get("safety_override", "0")),
            "",
            "Controls:",
            "\u2190\u2192 step 1  \u2191\u2193 step 10",
            "Space=play  q/w=speed  s=summary",
            "a=all plan curves  Home/End=first/last",
        ]
        if self.playing:
            info_lines.append("\n[AUTO-PLAY {} fps]".format(self.play_speed))

        self.ax_info.text(0.02, 0.98, "\n".join(info_lines),
                          transform=self.ax_info.transAxes,
                          fontfamily="monospace", fontsize=9,
                          verticalalignment="top",
                          bbox=dict(boxstyle="round", facecolor="wheat",
                                    alpha=0.5))

        self.fig.suptitle(
            "Trajectory Replay — {}  (frame {}/{})".format(
                os.path.basename(self.traj_dir),
                self.current, self.n_frames - 1),
            fontsize=12, fontweight="bold")
        self.fig.canvas.draw_idle()


def main():
    if len(sys.argv) < 2:
        print("Usage: python trajectory_replay.py <trajectory_dir>")
        print("Example: python trajectory_replay.py "
              "dataset/il_data/_failed/M01_micro_medium_000000_traj_001")
        sys.exit(1)

    traj_dir = sys.argv[1]
    if not os.path.isdir(traj_dir):
        print("Error: directory not found: {}".format(traj_dir))
        sys.exit(1)

    replay = TrajectoryReplay(traj_dir)
    plt.show()


if __name__ == "__main__":
    main()
