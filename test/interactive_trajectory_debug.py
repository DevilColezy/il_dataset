#!/usr/bin/env python3
"""Interactive top-down debugger for collected il_dataset episodes.

Usage (inside WSL with a GUI-capable matplotlib backend)::

    python3 interactive_trajectory_debug.py /path/to/il_data_joint_v2
    python3 interactive_trajectory_debug.py /path/to/il_data_joint_v2 \
        --include-failed

    The script discovers episode directories containing ``data.csv``.  By
    default it selects committed episodes; ``--include-failed`` also adds
    rejected episodes under ``_failed``.  After
choosing an episode, use the keyboard to step through the 30 Hz rows:

    right / down / n / space : next timestamp
    left / up / p            : previous timestamp
    home / end               : first / last timestamp
    q / escape               : quit
    h                        : print controls

The left panel is a world-XY top-down view.  It shows the travelled path,
the current local plan, the original navigation goal, the effective target,
and the current 30 Hz / 5 Hz target directions.  The right panel and the
terminal print the current row's state, target, planner, macro, and safety
fields.  A blueprint manifest is loaded automatically when it is found next
to the selected data root; pass ``--manifest`` to override it.
"""

import argparse
import csv
import json
import math
import os
import sys
def _float(row, key, default=float("nan")):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _text(row, key, default=""):
    value = row.get(key, default)
    return default if value is None else str(value)


def _fmt(value, digits=3):
    return "--" if not math.isfinite(value) else ("%." + str(digits) + "f") % value


INFERENCE_FIELDS = (
    "inference_ms",
    "model_inference_ms",
    "inference_time_ms",
    "model_inference_time_ms",
)


def find_inference_series(rows):
    """Return ``(field, values)`` for an optional model inference metric.

    Older collector datasets do not contain inference timing.  In that case,
    or when the column is present but empty, the viewer simply omits the
    inference subplot.
    """
    for field in INFERENCE_FIELDS:
        values = [_float(row, field) for row in rows]
        if any(math.isfinite(value) for value in values):
            return field, values
    return None, []


def discover_episodes(root, include_failed=False):
    """Return episode directories, optionally including rejected episodes,
    newest (by directory mtime) first."""
    if os.path.isfile(root):
        return [os.path.dirname(os.path.abspath(root))]
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    episodes = []
    for dirpath, dirnames, filenames in os.walk(root):
        # In-progress collections are never valid debug inputs.  Rejected
        # collections are opt-in so normal committed-data browsing remains
        # unchanged.
        excluded = {"_inprogress"}
        if not include_failed:
            excluded.add("_failed")
        dirnames[:] = [
            d for d in dirnames
            if d not in excluded and
            not d.endswith(".inprogress")
        ]
        if "data.csv" in filenames:
            episodes.append(os.path.abspath(dirpath))
    # Newest collection first (most recent episode is the default target).
    return sorted(episodes, key=lambda p: os.path.getmtime(p), reverse=True)


def load_rows(episode_dir):
    path = os.path.join(episode_dir, "data.csv")
    with open(path, "r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty data.csv: %s" % path)
    return rows


def find_manifest(root, explicit=None, scene_id=None):
    """Return the manifest matching the given scene_id, else the single /
    preferred manifest."""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    if os.path.isfile(root):
        root = os.path.dirname(root)
    candidates = []
    for name in os.listdir(root):
        if name.endswith("_manifest.json"):
            candidates.append(os.path.join(root, name))
    if len(candidates) == 1:
        return candidates[0]
    # Newest manifest first: the most recent collection is what you debug.
    candidates.sort(key=os.path.getmtime, reverse=True)
    # Multiple manifests (e.g. avoid_scenes / test_blueprint / joint_v2 in the
    # same root): pick the newest one whose scenes contain the episode's
    # scene_id so the obstacle overlay actually matches the collection.
    if scene_id is not None:
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    payload = json.load(stream)
                scene_ids = {
                    int(s.get("scene_id"))
                    for s in payload.get("scenes", [])
                    if s.get("scene_id") is not None
                }
                if int(scene_id) in scene_ids:
                    return path
            except (OSError, ValueError, TypeError):
                continue
    preferred = os.path.join(root, "joint_v2_blueprint_manifest.json")
    return preferred if os.path.isfile(preferred) else None


def load_scene_obstacles(manifest_path):
    """Return scene_id -> [(x, y, radius), ...] from an optional manifest."""
    if not manifest_path:
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError) as exc:
        print("warning: cannot read manifest %s: %s" % (manifest_path, exc),
              file=sys.stderr)
        return {}
    result = {}
    for scene in payload.get("scenes", []):
        obstacles = []
        for obstacle in scene.get("obstacles", []):
            try:
                obstacles.append((float(obstacle["x"]), float(obstacle["y"]),
                                  float(obstacle["radius"])))
            except (KeyError, TypeError, ValueError):
                continue
        try:
            result[int(scene["scene_id"])] = obstacles
        except (KeyError, TypeError, ValueError):
            continue
    return result


def parse_plan_points(value):
    """Decode writer format ``x1,y1;x2,y2;...``."""
    points = []
    if not value:
        return points
    for item in str(value).split(";"):
        try:
            x, y = item.split(",", 1)
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    return points


def body_direction_to_world_values(row, bx, by):
    yaw = _float(row, "yaw")
    if not all(math.isfinite(v) for v in (bx, by, yaw)):
        return None
    # Flightmare yaw convention: yaw=0 points the FLU nose toward world +Y.
    # Therefore forward=(-sin(yaw), cos(yaw)) and left=(-cos(yaw), -sin(yaw)).
    return (-math.sin(yaw) * bx - math.cos(yaw) * by,
            math.cos(yaw) * bx - math.sin(yaw) * by)


def body_direction_to_world(row, prefix):
    return body_direction_to_world_values(
        row, _float(row, prefix + "_x"), _float(row, prefix + "_y"))


def row_report(index, row, total):
    """Print the current timestamp's important diagnostic information."""
    speed = math.hypot(_float(row, "target_velocity_flu_x", 0.0),
                       _float(row, "target_velocity_flu_y", 0.0))
    state_speed = math.hypot(_float(row, "state_vx_flu", 0.0),
                             _float(row, "state_vy_flu", 0.0))
    print("\n" + "=" * 76)
    print("frame %d/%d  episode_frame_index=%s  t=%ss  dt=%ss" % (
        index + 1, total, _text(row, "episode_frame_index"),
        _fmt(_float(row, "trajectory_time_s"), 3),
        _fmt(_float(row, "control_dt_s"), 4)))
    print("state: pos=(%s, %s, %s) yaw=%s yaw_rate=%s speed_flu=%s" % (
        _fmt(_float(row, "x")), _fmt(_float(row, "y")),
        _fmt(_float(row, "z")), _fmt(_float(row, "yaw")),
        _fmt(_float(row, "yaw_rate")), _fmt(state_speed)))
    print("effective target: world=(%s, %s) body_dir=(%s, %s) norm_dist=%s" % (
        _fmt(_float(row, "effective_target_world_x")),
        _fmt(_float(row, "effective_target_world_y")),
        _fmt(_float(row, "goal_direction_flu_x")),
        _fmt(_float(row, "goal_direction_flu_y")),
        _fmt(_float(row, "goal_distance_norm"))))
    print("command: v_flu=(%s, %s, %s) speed=%s yaw_rate=%s" % (
        _fmt(_float(row, "target_velocity_flu_x")),
        _fmt(_float(row, "target_velocity_flu_y")),
        _fmt(_float(row, "target_velocity_flu_z")), _fmt(speed),
        _fmt(_float(row, "target_yaw_rate"))))
    print("planner: mode=%s status=%s failure=%s plan_valid=%s terminal=%s" % (
        _text(row, "hierarchical_mode", "?"), _text(row, "planner_status", "?"),
        _text(row, "planner_failure_reason", "?"),
        _text(row, "plan_valid", "?"), _text(row, "plan_terminal", "?")))
    print("macro: mask=%s valid=%s type=%s token=%s dir=(%s,%s) dist=%s" % (
        _text(row, "macro_update_mask", "0"),
        _text(row, "macro_label_valid", "?"),
        _text(row, "macro_correction_type", ""),
        _text(row, "macro_direction_token", ""),
        _fmt(_float(row, "macro_direction_flu_x")),
        _fmt(_float(row, "macro_direction_flu_y")),
        _fmt(_float(row, "macro_distance_norm"))))
    print("safety: observed_clearance=%s truth_clearance=%s brake_risk=%s "
          "truth_brake=%s emergency=%s corridor_blocked=%s" % (
              _fmt(_float(row, "min_observed_clearance_m")),
              _fmt(_float(row, "truth_minimum_clearance_m")),
              _fmt(_float(row, "truth_brake_risk")),
              _text(row, "truth_brake_would_trigger", "?"),
              _text(row, "emergency_brake", "?"),
              _text(row, "local_corridor_blocked", "?")))
    print("fsm=%s target_source=%s correction_active=%s obs_reason=%s" % (
        _text(row, "fsm_state", "?"), _text(row, "effective_target_source", "?"),
        _text(row, "target_correction_active", "?"),
        _text(row, "observability_reason", "?")))


class TopDownViewer:
    def __init__(self, rows, episode_dir, obstacles):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required in the WSL Python environment") from exc
        self.plt = plt
        self.Circle = Circle
        self.rows = rows
        self.episode_dir = episode_dir
        self.obstacles = obstacles
        self.index = 0
        self.inference_field, self.inference_values = find_inference_series(rows)
        self.inference_times = []
        if self.inference_field is not None:
            self.inference_times = [
                (_float(row, "trajectory_time_s", float(index))
                 if math.isfinite(_float(row, "trajectory_time_s"))
                 else float(index))
                for index, row in enumerate(rows)
            ]

        self.x = [_float(r, "x", 0.0) for r in rows]
        self.y = [_float(r, "y", 0.0) for r in rows]
        self.fig = plt.figure(figsize=(14, 8))
        if self.inference_field is not None:
            self.ax = self.fig.add_axes([0.05, 0.31, 0.62, 0.63])
            self.inference_ax = self.fig.add_axes([0.05, 0.08, 0.62, 0.16])
        else:
            self.ax = self.fig.add_axes([0.05, 0.08, 0.62, 0.86])
            self.inference_ax = None
        self.info = self.fig.add_axes([0.70, 0.05, 0.28, 0.90])
        self.info.set_axis_off()
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update(0, announce=True)

    def _limits(self):
        points = list(zip(self.x, self.y))
        for row in self.rows:
            for px, py in (("navigation_goal_world_x", "navigation_goal_world_y"),
                           ("effective_target_world_x", "effective_target_world_y"),
                           ("original_navigation_goal_world_x",
                            "original_navigation_goal_world_y")):
                a, b = _float(row, px), _float(row, py)
                if math.isfinite(a) and math.isfinite(b):
                    points.append((a, b))
        if not points:
            return (-1, 1, -1, 1)
        xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
        ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
        span = max(xmax - xmin, ymax - ymin, 1.0)
        margin = 0.08 * span + 0.5
        return xmin - margin, xmax + margin, ymin - margin, ymax + margin

    def update(self, index, announce=True):
        self.index = max(0, min(len(self.rows) - 1, index))
        row = self.rows[self.index]
        self.ax.clear()
        self.ax.set_title("Top-down trajectory: %s  [%d/%d]" % (
            os.path.basename(self.episode_dir), self.index + 1, len(self.rows)))
        self.ax.set_xlabel("world x (m)")
        self.ax.set_ylabel("world y (m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(*self._limits()[:2])
        self.ax.set_ylim(*self._limits()[2:])

        self.ax.plot(self.x, self.y, color="0.75", linewidth=1.0,
                     label="full trajectory")
        self.ax.plot(self.x[:self.index + 1], self.y[:self.index + 1],
                     color="tab:blue", linewidth=2.0, label="history")

        scene_id = _int(row, "scene_id", -1)
        for ox, oy, radius in self.obstacles.get(scene_id, []):
            self.ax.add_patch(self.Circle((ox, oy), radius, color="0.25",
                                           alpha=0.25, linewidth=0))

        sx, sy = _float(row, "x", 0.0), _float(row, "y", 0.0)
        self.ax.scatter([sx], [sy], c="tab:blue", s=70, zorder=8,
                        label="current")
        self._draw_heading(row)

        self._scatter_target(row, "navigation_goal_world_x",
                             "navigation_goal_world_y", "original goal",
                             "tab:red", "*")
        self._scatter_target(row, "effective_target_world_x",
                             "effective_target_world_y", "effective target",
                             "tab:orange", "x")
        self._scatter_target(row, "original_navigation_goal_world_x",
                             "original_navigation_goal_world_y", "original goal",
                             "tab:red", "*")

        plan = parse_plan_points(_text(row, "plan_points_xy"))
        if plan:
            self.ax.plot([p[0] for p in plan], [p[1] for p in plan],
                         color="tab:green", linewidth=2, alpha=0.9,
                         label="local plan")

        self._draw_body_arrow(row, "goal_direction_flu", "effective dir",
                              "tab:orange", 1.2)
        if _int(row, "macro_update_mask"):
            self._draw_body_arrow(row, "macro_direction_flu", "macro dir",
                                  "tab:purple", 1.5)
        self.ax.legend(loc="upper left", fontsize=8)

        self._draw_inference_plot()
        lines = self._info_lines(row, self.inference_field)
        self.info.clear()
        self.info.set_axis_off()
        self.info.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                       family="monospace", fontsize=9)
        self.fig.canvas.draw_idle()
        if announce:
            row_report(self.index, row, len(self.rows))

    def _draw_inference_plot(self):
        if self.inference_ax is None:
            return
        self.inference_ax.clear()
        self.inference_ax.plot(
            self.inference_times, self.inference_values,
            color="tab:purple", linewidth=1.2, label="inference time")
        self.inference_ax.axvline(
            self.inference_times[self.index], color="tab:red",
            linewidth=1.0, alpha=0.8)
        current = self.inference_values[self.index]
        if math.isfinite(current):
            self.inference_ax.scatter(
                [self.inference_times[self.index]], [current],
                color="tab:red", s=28, zorder=3)
        self.inference_ax.set_title(
            "Model inference: %s (lower is faster)" % self.inference_field,
            fontsize=9)
        self.inference_ax.set_xlabel("trajectory time (s)")
        self.inference_ax.set_ylabel("ms")
        self.inference_ax.grid(True, alpha=0.3)



    def _scatter_target(self, row, xkey, ykey, label, color, marker):
        x, y = _float(row, xkey), _float(row, ykey)
        if math.isfinite(x) and math.isfinite(y):
            self.ax.scatter([x], [y], c=color, marker=marker, s=100,
                            zorder=7, label=label)

    def _draw_body_arrow(self, row, prefix, label, color, length):
        direction = body_direction_to_world(row, prefix)
        if direction is None:
            return
        sx, sy = _float(row, "x", 0.0), _float(row, "y", 0.0)
        self.ax.arrow(sx, sy, length * direction[0], length * direction[1],
                      color=color, width=0.015, head_width=0.16,
                      length_includes_head=True, zorder=10, label=label)

    def _draw_heading(self, row):
        direction = body_direction_to_world_values(row, 1.0, 0.0)
        if direction is None:
            return
        sx, sy = _float(row, "x", 0.0), _float(row, "y", 0.0)
        self.ax.arrow(sx, sy, 0.85 * direction[0], 0.85 * direction[1],
                      color="tab:blue", width=0.02, head_width=0.18,
                      length_includes_head=True, zorder=10,
                      label="nose / +X_FLU")

    @staticmethod
    def _info_lines(row, inference_field=None):
        speed = math.hypot(_float(row, "target_velocity_flu_x", 0.0),
                           _float(row, "target_velocity_flu_y", 0.0))
        lines = [
            "frame: %s / t=%ss" % (_text(row, "episode_frame_index", "?"),
                                    _fmt(_float(row, "trajectory_time_s"))),
            "scene/task: %s / %s" % (_text(row, "scene_id", "?"),
                                     _text(row, "task_id", "?")),
            "",
            "STATE",
            "  pos: (%s, %s)" % (_fmt(_float(row, "x")), _fmt(_float(row, "y"))),
            "  yaw: %s rad" % _fmt(_float(row, "yaw")),
            "  v_flu: (%s, %s)" % (_fmt(_float(row, "state_vx_flu")),
                                   _fmt(_float(row, "state_vy_flu"))),
            "",
            "TARGET / COMMAND",
            "  mode: %s" % _text(row, "hierarchical_mode", "?"),
            "  target: (%s, %s)" % (_fmt(_float(row, "effective_target_world_x")),
                                    _fmt(_float(row, "effective_target_world_y"))),
            "  label: dir=(%s,%s) d=%s" % (
                _fmt(_float(row, "goal_direction_flu_x")),
                _fmt(_float(row, "goal_direction_flu_y")),
                _fmt(_float(row, "goal_distance_norm"))),
            "  command speed/yaw: %s / %s" % (
                _fmt(speed), _fmt(_float(row, "target_yaw_rate"))),
            "",
            "PLANNER",
            "  status: %s" % _text(row, "planner_status", "?"),
            "  failure: %s" % _text(row, "planner_failure_reason", "?"),
            "  plan/terminal: %s / %s" % (_text(row, "plan_valid", "?"),
                                          _text(row, "plan_terminal", "?")),
            "  macro: %s (mask=%s)" % (
                _text(row, "macro_correction_type", ""),
                _text(row, "macro_update_mask", "0")),
            "",
            "SAFETY",
            "  clearance obs/truth: %s / %s" % (
                _fmt(_float(row, "min_observed_clearance_m")),
                _fmt(_float(row, "truth_minimum_clearance_m"))),
            "  brake/emergency: %s / %s" % (
                _text(row, "truth_brake_would_trigger", "?"),
                _text(row, "emergency_brake", "?")),
            "  corridor blocked: %s" % _text(row, "local_corridor_blocked", "?"),
        ]
        if inference_field is not None:
            inference_ms = _float(row, inference_field)
            if math.isfinite(inference_ms):
                lines.extend([
                    "",
                    "MODEL",
                    "  inference: %sms (%s FPS)" % (
                        _fmt(inference_ms, 3),
                        _fmt(1000.0 / inference_ms, 1)
                        if inference_ms > 0.0 else "--"),
                ])
        return lines

    def on_key(self, event):
        if event.key in ("right", "down", "n", " "):
            self.update(self.index + 1)
        elif event.key in ("left", "up", "p"):
            self.update(self.index - 1)
        elif event.key == "home":
            self.update(0)
        elif event.key == "end":
            self.update(len(self.rows) - 1)
        elif event.key in ("q", "escape"):
            self.plt.close(self.fig)
        elif event.key == "h":
            print("controls: right/down/n/space next; left/up/p previous; "
                  "home/end jump; q/escape quit")


def choose_episode(episodes, auto=False):
    if len(episodes) == 1 or auto:
        return episodes[0]
    print("Available episodes:")
    for i, path in enumerate(episodes):
        try:
            rows = load_rows(path)
            label = "%d rows, task=%s" % (len(rows), _text(rows[0], "task_id", "?"))
            failure_path = os.path.join(path, "failure_reason.json")
            if os.path.isfile(failure_path):
                try:
                    with open(failure_path, "r", encoding="utf-8") as stream:
                        reason = json.load(stream).get("reason", "unknown")
                    label += ", FAILED: %s" % reason
                except (OSError, ValueError):
                    label += ", FAILED"
        except (OSError, ValueError):
            label = "unreadable"
        print("  [%d] %s (%s)" % (i, path, label))
    while True:
        answer = input("Select episode index (q to quit): ").strip()
        if answer.lower() == "q":
            raise SystemExit(0)
        try:
            index = int(answer)
            if 0 <= index < len(episodes):
                return episodes[index]
        except ValueError:
            pass
        print("invalid index")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", help="episode directory, data.csv, or output root")
    parser.add_argument("--manifest", default=None,
                        help="optional blueprint manifest for obstacle overlay")
    parser.add_argument("--include-failed", action="store_true",
                        help="include episodes under _failed in the selector "
                             "(default: auto-included for debugging)")
    parser.add_argument("--committed-only", action="store_true",
                        help="only committed episodes (exclude _failed)")
    parser.add_argument("--auto", action="store_true",
                        help="auto-select the newest episode without prompting")
    parser.add_argument("--index", type=int, default=None,
                        help="select episode by index (newest=0); implies --auto")
    args = parser.parse_args()

    include_failed = not args.committed_only
    episodes = discover_episodes(args.data_root,
                                 include_failed=include_failed)
    if not episodes:
        parser.error("no debug episode containing data.csv: %s" % args.data_root)
    if args.index is not None:
        try:
            episode = episodes[args.index]
        except IndexError:
            parser.error("index %d out of range (0..%d)" % (args.index,
                                                             len(episodes) - 1))
    else:
        episode = choose_episode(episodes, auto=args.auto)
    rows = load_rows(episode)
    # Scene id of the first row -> auto-match the correct manifest overlay.
    try:
        first_scene_id = _int(rows[0], "scene_id", None)
    except Exception:
        first_scene_id = None
    manifest = find_manifest(args.data_root, args.manifest,
                             scene_id=first_scene_id)
    obstacles = load_scene_obstacles(manifest)
    print("selected: %s (%d rows, %s)" % (
        episode, len(rows),
        "FAILED: %s" % _text(rows[0], "failure_taxonomy", "?") if
        _text(rows[0], "episode_valid", "1") != "1" else "committed"))
    print("manifest: %s" % (manifest or "not found; obstacle overlay disabled"))

    try:
        viewer = TopDownViewer(rows, episode, obstacles)
        viewer.plt.show()
    except RuntimeError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
