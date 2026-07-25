#!/usr/bin/env python3
"""
Dataset Evaluator for IL Dataset Collection Pipeline.

Features:
- Auto-discover the latest dataset directory
- Label distribution analysis (trend horizontal/vertical, guide modes, recovery)
- Trajectory quality metrics (collision, clearance, smoothness, goal-reaching)
- Planner performance analysis (success rate, timing)
- Scene-level and profile-level aggregation
- Rich terminal output with color-coded summaries
- Visualization: trajectory paths, label histograms, clearance/speed profiles
- JSON export for downstream tools

Usage:
    # Full evaluation with plots
    python dataset_evaluator.py --mode full

    # Only label distribution
    python dataset_evaluator.py --mode labels --data-dir /path/to/il_data

    # Planner analysis only, no plots
    python dataset_evaluator.py --mode planner --no-plots

    # Scene-level comparison
    python dataset_evaluator.py --mode scene --output-dir ./reports
"""

import argparse
import json
import os
import sys
import csv
import math
import warnings
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# --- Optional imports with graceful fallback ---
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Circle, FancyBboxPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Constants & Style
# ============================================================================

# Expected data.csv columns (Schema v15)
TREND_HORIZONTAL_CLASS_COL = "trend_horizontal_class_13"
GUIDE_ELEVATION_BIN_COL = "guide_elevation_bin"
GUIDE_MODE_COL = "trend_mode"  # trend_mode doubles as guide/control mode
TREND_MODE_COL = "trend_mode"
RECOVERY_DIRECTION_COL = "recovery_direction"  # non-zero indicates recovery active
RECOVERY_ELAPSED_COL = "recovery_elapsed_s"
COLLISION_COL = "collision"
FRAME_VALID_COL = "frame_valid"
PLANNER_SUCCESS_COL = "planner_success"
PLANNER_COMPUTE_MS_COL = "planner_compute_ms"
PLANNER_MIN_CLEARANCE_COL = "planner_min_clearance"
EXECUTED_CLEARANCE_COL = "minimum_executed_clearance"  # in metadata
TRAJECTORY_TIME_COL = "trajectory_time_s"
DISTANCE_TO_GOAL_COL = "distance_to_final_goal"
EPISODE_FRAME_INDEX_COL = "episode_frame_index"
SCENE_ID_COL = "scene_id"
TASK_ID_COL = "task_id"
VELOCITY_X_COL = "state_vx_flu"
VELOCITY_Y_COL = "state_vy_flu"
VELOCITY_Z_COL = "state_vz_flu"
POS_X_COL = "x"
POS_Y_COL = "y"
POS_Z_COL = "z"
YAW_RATE_COL = "executed_next_yaw_rate"
EXPERT_LABEL_VALID_COL = "expert_label_valid"
TREND_LABEL_VALID_COL = "trend_label_valid"
TREND_H_LOSS_VALID_COL = "trend_horizontal_loss_valid"
TREND_V_LOSS_VALID_COL = "trend_vertical_loss_valid"
TREND_VALUE_LOSS_VALID_COL = "trend_value_loss_valid"
CONTROL_LOSS_VALID_COL = "control_loss_valid"
START_X_COL = "start_x"
START_Y_COL = "start_y"
START_Z_COL = "start_z"
GOAL_X_COL = "goal_x"
GOAL_Y_COL = "goal_y"
GOAL_Z_COL = "goal_z"
APPLIED_VX_COL = "applied_command_vx_flu"
APPLIED_VY_COL = "applied_command_vy_flu"
APPLIED_VZ_COL = "applied_command_vz_flu"
APPLIED_YAW_RATE_COL = "applied_command_yaw_rate"
EXPERT_VX_COL = "expert_vx_flu"
EXPERT_VY_COL = "expert_vy_flu"
EXPERT_VZ_COL = "expert_vz_flu"
EXPERT_YAW_RATE_COL = "expert_yaw_rate"
VELOCITY_TRACKING_COL = "velocity_tracking_error"
YAW_RATE_TRACKING_COL = "yaw_rate_tracking_error"
EXIT_REASON_KEY = "exit_reason"
FINAL_GOAL_ERROR_KEY = "final_goal_error"

# 13-class horizontal trend label names
TREND_CLASS_NAMES = {
    0: "fwd center", 1: "fwd slightly L", 2: "fwd L",
    3: "fwd slightly R", 4: "fwd R",
    5: "left", 6: "back L", 7: "back", 8: "back R", 9: "right",
    10: "up", 11: "down",
    12: "recovery_L", 13: "recovery_R",
}
# Actually 11 normal + 2 recovery = 13 classes (indices 0-12)
# But the mapping may vary; we'll be data-driven

# Guide mode names
GUIDE_MODE_NAMES = {
    "NORMAL": "Normal (guide tracking)",
    "RECOVER_LEFT": "Recovery Left",
    "RECOVER_RIGHT": "Recovery Right",
    "HOVER": "Hover/Hold",
    "UNKNOWN": "Unknown",
}

# ANSI color codes for terminal
class Term:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# ============================================================================
# Data Loading
# ============================================================================

def discover_datasets(data_root: str) -> List[str]:
    """Discover all trajectory directories under a data root.

    Returns list of absolute paths to committed trajectory directories.
    Skips `.inprogress`, `_failed`, `_debug`, `scenes/` directories.
    """
    root = Path(data_root)
    if not root.exists():
        return []

    traj_dirs = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        # Skip special directories
        if name.startswith("_") or name == "scenes":
            continue
        # Check for trajectories inside
        for sub in sorted(item.iterdir()):
            if sub.is_dir() and not sub.name.endswith(".inprogress"):
                # Verify it has data.csv
                if (sub / "data.csv").exists():
                    traj_dirs.append(str(sub))

    return traj_dirs


def find_latest_dataset(search_paths: List[str]) -> str:
    """Find the most recently modified dataset directory."""
    best_dir = None
    best_mtime = 0

    for search_path in search_paths:
        p = Path(search_path)
        if not p.exists():
            continue
        # Check if it directly contains trajectories
        if any((d / "data.csv").exists() for d in p.iterdir() if d.is_dir() and not d.name.startswith("_")):
            mtime = p.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best_dir = str(p)
        # Also check subdirectories
        for sub in p.iterdir():
            if sub.is_dir() and not sub.name.startswith("_"):
                if any((d / "data.csv").exists() for d in sub.iterdir() if d.is_dir() and not d.name.endswith(".inprogress")):
                    mtime = sub.stat().st_mtime
                    if mtime > best_mtime:
                        best_mtime = mtime
                        best_dir = str(sub)
                        # Also go one level deeper to get all trajectories
                        # We want the PARENT that contains scene_* dirs
    if best_dir is None and search_paths:
        # Fallback: return first that exists and has csv files anywhere
        for sp in search_paths:
            for root, dirs, files in os.walk(sp):
                if "data.csv" in files:
                    # Get parent of the traj dir → scene dir → root
                    parts = Path(root).relative_to(sp).parts
                    if len(parts) >= 2 and not parts[0].startswith("_"):
                        best_dir = str(Path(sp) / parts[0])
                        break
            if best_dir:
                break

    return best_dir or ""


def _infer_column_type(values: List[str]) -> str:
    """Infer numpy dtype for a column by sampling non-empty values."""
    sample = [v for v in values[:20] if v and v.strip()]
    if not sample:
        return "U32"  # default to string
    # Try float first
    try:
        for v in sample:
            float(v)
        # Check if all are integers
        all_int = True
        for v in sample:
            f = float(v)
            if f != int(f):
                all_int = False
                break
        if all_int:
            return "i8"
        return "f8"
    except ValueError:
        return "U64"


def load_trajectory_csv(csv_path: str) -> Optional[np.ndarray]:
    """Load a data.csv file into a structured numpy array with mixed types."""
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        if not rows or not header:
            return None

        n_cols = len(header)
        # Transpose to get column values
        columns = [[] for _ in range(n_cols)]
        for row in rows:
            # Pad/truncate to match header length
            if len(row) < n_cols:
                row = row + [""] * (n_cols - len(row))
            for i in range(n_cols):
                columns[i].append(row[i] if i < len(row) else "")

        # Build dtype from inferred column types
        dtype_parts = []
        for i, col_name in enumerate(header):
            col_type = _infer_column_type(columns[i])
            # Sanitize column name for numpy (no special chars)
            safe_name = col_name.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            if not safe_name:
                safe_name = f"col_{i}"
            dtype_parts.append((safe_name, col_type))

        # Parse rows into tuples with proper types
        data = []
        for row in rows:
            parsed = []
            for i in range(n_cols):
                dt = dtype_parts[i][1]
                raw = row[i] if i < len(row) else ""
                if dt in ("f8", "i8"):
                    try:
                        val = float(raw)
                        parsed.append(int(val) if dt == "i8" else val)
                    except (ValueError, TypeError):
                        parsed.append(0 if dt == "i8" else np.nan)
                else:
                    parsed.append(str(raw) if raw else "")
            data.append(tuple(parsed))

        arr = np.array(data, dtype=np.dtype(dtype_parts))
        return arr
    except Exception as e:
        import traceback
        print(f"  {Term.RED}Error loading {csv_path}: {e}{Term.RESET}")
        traceback.print_exc()
        return None


def load_metadata(meta_path: str) -> Dict[str, Any]:
    """Load metadata.json."""
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_global_path(gp_path: str) -> Optional[np.ndarray]:
    """Load global_path.csv into numpy array."""
    try:
        with open(gp_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        pts = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
        return pts
    except Exception:
        return None


# ============================================================================
# Metrics Computation
# ============================================================================

class TrajectoryMetrics:
    """Holds computed metrics for a single trajectory."""

    __slots__ = (
        "traj_id", "scene_id", "task_id", "num_frames",
        "duration_s", "collision_frames", "collision_rate",
        "frame_valid_rate", "expert_label_valid_rate", "trend_label_valid_rate",
        "trend_h_loss_valid_rate", "trend_v_loss_valid_rate",
        "trend_value_loss_valid_rate", "control_loss_valid_rate",
        "exit_reason", "final_goal_error_m",
        "trend_class_distribution", "guide_elevation_distribution",
        "guide_mode_distribution", "recovery_frame_count", "recovery_rate",
        "mean_speed", "max_speed", "mean_yaw_rate", "max_yaw_rate",
        "mean_clearance", "min_clearance", "clearance_profile",
        "speed_profile", "yaw_rate_profile",
        "jerk_rms", "acceleration_rms",
        "planner_success_rate", "mean_planning_ms", "p95_planning_ms",
        "num_replans", "num_failed_replans",
        "velocity_tracking_rmse", "yaw_rate_tracking_rmse",
        "start_pos", "goal_pos", "positions",
        "planner_success_rate_detail",
    )

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self.trend_class_distribution = Counter()
        self.guide_elevation_distribution = Counter()
        self.guide_mode_distribution = Counter()
        self.positions = None


def compute_trajectory_metrics(
    data: np.ndarray, meta: Dict[str, Any], traj_id: str, scene_id: str, task_id: str
) -> TrajectoryMetrics:
    """Compute all metrics for a single trajectory."""
    m = TrajectoryMetrics()
    m.traj_id = traj_id
    m.scene_id = scene_id
    m.task_id = task_id
    m.num_frames = len(data)

    if m.num_frames == 0:
        return m

    # Time
    if TRAJECTORY_TIME_COL in data.dtype.names:
        m.duration_s = float(data[TRAJECTORY_TIME_COL][-1] - data[TRAJECTORY_TIME_COL][0])
    else:
        m.duration_s = 0.0

    # Collision
    if COLLISION_COL in data.dtype.names:
        coll = data[COLLISION_COL]
        m.collision_frames = int(np.nansum(coll))
        m.collision_rate = m.collision_frames / m.num_frames if m.num_frames > 0 else 0.0

    # Frame validity
    if FRAME_VALID_COL in data.dtype.names:
        valid = data[FRAME_VALID_COL]
        m.frame_valid_rate = float(np.nanmean(valid)) if len(valid) > 0 else 1.0

    # Label validity
    for col, attr in [
        (EXPERT_LABEL_VALID_COL, "expert_label_valid_rate"),
        (TREND_LABEL_VALID_COL, "trend_label_valid_rate"),
        (TREND_H_LOSS_VALID_COL, "trend_h_loss_valid_rate"),
        (TREND_V_LOSS_VALID_COL, "trend_v_loss_valid_rate"),
        (TREND_VALUE_LOSS_VALID_COL, "trend_value_loss_valid_rate"),
        (CONTROL_LOSS_VALID_COL, "control_loss_valid_rate"),
    ]:
        if col in data.dtype.names:
            vals = data[col]
            v = float(np.nanmean(vals)) if len(vals) > 0 else 1.0
            # loss_valid=1 means valid, loss_valid=0 means masked (invalid for training)
            setattr(m, attr, v)

    # Exit reason and goal error
    m.exit_reason = meta.get(EXIT_REASON_KEY, "unknown")
    m.final_goal_error_m = meta.get(FINAL_GOAL_ERROR_KEY, None)

    # Trend class distribution
    if TREND_HORIZONTAL_CLASS_COL in data.dtype.names:
        classes = data[TREND_HORIZONTAL_CLASS_COL]
        for c in classes:
            if not np.isnan(c):
                m.trend_class_distribution[int(c)] += 1

    # Guide elevation distribution
    if GUIDE_ELEVATION_BIN_COL in data.dtype.names:
        bins = data[GUIDE_ELEVATION_BIN_COL]
        for b in bins:
            if not np.isnan(b):
                m.guide_elevation_distribution[int(b)] += 1

    # Guide mode distribution (trend_mode column: TRACK_GUIDE, RECOVER_LEFT, RECOVER_RIGHT, HOVER, etc.)
    if GUIDE_MODE_COL in data.dtype.names:
        modes = data[GUIDE_MODE_COL]
        for mode in modes:
            if isinstance(mode, str) and mode.strip():
                m.guide_mode_distribution[mode.strip()] += 1
            elif isinstance(mode, (int, float)) and not np.isnan(mode):
                m.guide_mode_distribution[str(int(mode))] += 1

    # Recovery frames: recovery_direction != 0 or recovery_elapsed_s > 0
    recovery_mask = None
    if RECOVERY_DIRECTION_COL in data.dtype.names:
        rd = data[RECOVERY_DIRECTION_COL]
        try:
            rd_vals = rd.astype(float)
            recovery_mask = np.abs(rd_vals) > 0.01
        except (ValueError, TypeError):
            recovery_mask = np.array([str(v).strip() not in ("", "0", "0.0", "None", "none") for v in rd])
    elif RECOVERY_ELAPSED_COL in data.dtype.names:
        re = data[RECOVERY_ELAPSED_COL]
        try:
            re_vals = re.astype(float)
            recovery_mask = re_vals > 0.0
        except (ValueError, TypeError):
            pass

    if recovery_mask is not None:
        m.recovery_frame_count = int(np.sum(recovery_mask))
        m.recovery_rate = m.recovery_frame_count / m.num_frames if m.num_frames > 0 else 0.0

    # Speed
    if VELOCITY_X_COL in data.dtype.names and VELOCITY_Y_COL in data.dtype.names and VELOCITY_Z_COL in data.dtype.names:
        vx = data[VELOCITY_X_COL]
        vy = data[VELOCITY_Y_COL]
        vz = data[VELOCITY_Z_COL]
        speeds = np.sqrt(vx**2 + vy**2 + vz**2)
        m.speed_profile = speeds
        m.mean_speed = float(np.nanmean(speeds)) if len(speeds) > 0 else 0.0
        m.max_speed = float(np.nanmax(speeds)) if len(speeds) > 0 else 0.0

    # Yaw rate
    if YAW_RATE_COL in data.dtype.names:
        yr = data[YAW_RATE_COL]
        m.yaw_rate_profile = yr
        m.mean_yaw_rate = float(np.nanmean(np.abs(yr))) if len(yr) > 0 else 0.0
        m.max_yaw_rate = float(np.nanmax(np.abs(yr))) if len(yr) > 0 else 0.0

    # Clearance
    if PLANNER_MIN_CLEARANCE_COL in data.dtype.names:
        cl = data[PLANNER_MIN_CLEARANCE_COL]
        m.clearance_profile = cl
        m.mean_clearance = float(np.nanmean(cl)) if len(cl) > 0 else 0.0
        m.min_clearance = float(np.nanmin(cl)) if len(cl) > 0 else 0.0

    # Jerk / smoothness (from acceleration difference)
    if VELOCITY_X_COL in data.dtype.names:
        vx = data[VELOCITY_X_COL]
        vy = data[VELOCITY_Y_COL]
        vz = data[VELOCITY_Z_COL]
        if len(vx) > 2:
            dt = float(data[TRAJECTORY_TIME_COL][1] - data[TRAJECTORY_TIME_COL][0]) if TRAJECTORY_TIME_COL in data.dtype.names else 0.033
            if dt > 0:
                ax = np.gradient(vx, dt)
                ay = np.gradient(vy, dt)
                az = np.gradient(vz, dt)
                jerk_x = np.gradient(ax, dt)
                jerk_y = np.gradient(ay, dt)
                jerk_z = np.gradient(az, dt)
                jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
                m.jerk_rms = float(np.sqrt(np.nanmean(jerk_mag**2)))
                accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
                m.acceleration_rms = float(np.sqrt(np.nanmean(accel_mag**2)))

    # Velocity tracking error
    if VELOCITY_TRACKING_COL in data.dtype.names:
        err = data[VELOCITY_TRACKING_COL]
        m.velocity_tracking_rmse = float(np.sqrt(np.nanmean(err**2))) if len(err) > 0 else 0.0
    if YAW_RATE_TRACKING_COL in data.dtype.names:
        err = data[YAW_RATE_TRACKING_COL]
        m.yaw_rate_tracking_rmse = float(np.sqrt(np.nanmean(err**2))) if len(err) > 0 else 0.0

    # Planner stats
    if PLANNER_SUCCESS_COL in data.dtype.names:
        ps = data[PLANNER_SUCCESS_COL]
        try:
            if isinstance(ps[0], (str, np.str_)):
                # String values like "True"/"False"
                m.planner_success_rate = float(np.mean([1.0 if str(v).strip().lower() == "true" else 0.0 for v in ps]))
            else:
                m.planner_success_rate = float(np.nanmean(ps.astype(float))) if len(ps) > 0 else 1.0
        except Exception:
            pass
    if PLANNER_COMPUTE_MS_COL in data.dtype.names:
        pt = data[PLANNER_COMPUTE_MS_COL]
        pt_valid = pt[~np.isnan(pt)]
        m.mean_planning_ms = float(np.nanmean(pt_valid)) if len(pt_valid) > 0 else 0.0
        m.p95_planning_ms = float(np.percentile(pt_valid, 95)) if len(pt_valid) > 0 else 0.0

    # Planner replans from metadata
    m.num_replans = meta.get("fresh_plan_control_frame_count", 0)
    m.num_failed_replans = meta.get("failed_replans", 0)

    # Start/goal
    sx = sy = sz = gx = gy = gz = None
    for col in [START_X_COL, START_Y_COL, START_Z_COL, GOAL_X_COL, GOAL_Y_COL, GOAL_Z_COL]:
        if col in data.dtype.names:
            vals = data[col]
            # Filter NaN from numeric columns
            try:
                valid_vals = vals[~np.isnan(vals.astype(float))]
            except (TypeError, ValueError):
                valid_vals = vals
            v = float(valid_vals[0]) if len(valid_vals) > 0 else None
            if col == START_X_COL: sx = v
            elif col == START_Y_COL: sy = v
            elif col == START_Z_COL: sz = v
            elif col == GOAL_X_COL: gx = v
            elif col == GOAL_Y_COL: gy = v
            elif col == GOAL_Z_COL: gz = v

    m.start_pos = (sx, sy, sz) if sx is not None else None
    m.goal_pos = (gx, gy, gz) if gx is not None else None

    # Positions for trajectory plot
    if POS_X_COL in data.dtype.names and POS_Y_COL in data.dtype.names and POS_Z_COL in data.dtype.names:
        m.positions = np.column_stack([data[POS_X_COL], data[POS_Y_COL], data[POS_Z_COL]])

    return m


# ============================================================================
# Aggregation
# ============================================================================

class DatasetSummary:
    """Aggregated metrics across all trajectories in a dataset."""

    def __init__(self):
        self.num_trajectories = 0
        self.num_total_frames = 0
        self.total_duration_s = 0.0
        self.trajectory_metrics: List[TrajectoryMetrics] = []

        # Aggregate counters
        self.global_trend_classes = Counter()
        self.global_elevation_bins = Counter()
        self.global_guide_modes = Counter()
        self.total_collision_frames = 0
        self.total_recovery_frames = 0
        self.exit_reasons = Counter()
        self.scene_stats: Dict[str, Dict] = defaultdict(lambda: {
            "num_trajs": 0, "total_frames": 0, "collision_frames": 0, "goal_errors": [],
        })

        # Per-profile aggregation (extracted from scene_id naming)
        self.profile_stats: Dict[str, Dict] = defaultdict(lambda: {
            "num_trajs": 0, "total_frames": 0, "goal_errors": [], "collision_rates": [],
            "mean_speeds": [], "durations": [],
        })

    def add_trajectory(self, tm: TrajectoryMetrics):
        self.num_trajectories += 1
        self.num_total_frames += tm.num_frames
        self.total_duration_s += tm.duration_s
        self.trajectory_metrics.append(tm)

        self.global_trend_classes.update(tm.trend_class_distribution)
        self.global_elevation_bins.update(tm.guide_elevation_distribution)
        self.global_guide_modes.update(tm.guide_mode_distribution)
        self.total_collision_frames += tm.collision_frames or 0
        self.total_recovery_frames += tm.recovery_frame_count or 0
        self.exit_reasons[tm.exit_reason] += 1

        # Scene stats
        sid = tm.scene_id or "unknown"
        ss = self.scene_stats[sid]
        ss["num_trajs"] += 1
        ss["total_frames"] += tm.num_frames
        ss["collision_frames"] += tm.collision_frames or 0
        if tm.final_goal_error_m is not None:
            ss["goal_errors"].append(tm.final_goal_error_m)

        # Profile stats: try to parse profile name from scene_id
        # scene_id like "small_only_000000" → profile "small_only"
        # or "scene_0000_sub0000" → no profile
        if sid:
            # Try profile_mode naming: {profile_name}_{scene_index:06d}
            parts = sid.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 4:
                profile_name = parts[0]
            else:
                profile_name = "unknown_profile"
            ps = self.profile_stats[profile_name]
            ps["num_trajs"] += 1
            ps["total_frames"] += tm.num_frames
            if tm.final_goal_error_m is not None:
                ps["goal_errors"].append(tm.final_goal_error_m)
            if tm.collision_rate is not None:
                ps["collision_rates"].append(tm.collision_rate)
            if tm.mean_speed is not None:
                ps["mean_speeds"].append(tm.mean_speed)
            if tm.duration_s is not None:
                ps["durations"].append(tm.duration_s)


# ============================================================================
# Rich Terminal Output
# ============================================================================

def print_header(title: str, width: int = 70):
    print(f"\n{Term.BOLD}{Term.CYAN}{'='*width}{Term.RESET}")
    print(f"{Term.BOLD}{Term.CYAN}  {title}{Term.RESET}")
    print(f"{Term.BOLD}{Term.CYAN}{'='*width}{Term.RESET}")

def print_subheader(title: str):
    print(f"\n{Term.BOLD}{Term.YELLOW}--- {title} ---{Term.RESET}")

def print_stat(label: str, value: Any, unit: str = "", good: bool = None, warn_thresh: float = None):
    """Print a labeled statistic with optional color coding."""
    if isinstance(value, float):
        val_str = f"{value:.3f}"
    elif isinstance(value, int):
        val_str = f"{value:,}"
    else:
        val_str = str(value)

    color = Term.RESET
    if good is True:
        color = Term.GREEN
    elif good is False:
        color = Term.RED
    elif warn_thresh is not None and isinstance(value, (int, float)):
        if value > warn_thresh:
            color = Term.YELLOW

    unit_str = f" {unit}" if unit else ""
    print(f"  {label:.<40s} {color}{val_str}{unit_str}{Term.RESET}")

def print_bar_chart(counter: Dict, max_bars: int = 20, width: int = 40):
    """Print a horizontal bar chart using ASCII characters."""
    if not counter:
        print("  (no data)")
        return
    total = sum(counter.values())
    if total == 0:
        print("  (no data)")
        return

    items = sorted(counter.items())[:max_bars]
    max_count = max(c[1] for c in items) if items else 1
    for key, count in items:
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        pct = (count / total) * 100
        key_str = str(key)[:25]
        print(f"  {key_str:.<28s} {bar} {count:,} ({pct:.1f}%)")


# ============================================================================
# Report Generation
# ============================================================================

def generate_full_report(ds: DatasetSummary, traj_dirs: List[str], data_root: str):
    """Print a comprehensive terminal report."""
    print_header("📊 IL Dataset Evaluation Report")
    print(f"  Dataset root: {data_root}")
    print(f"  Evaluated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Trajectories found: {len(traj_dirs)}")
    print(f"  Trajectories loaded: {ds.num_trajectories}")

    # === 1. Overview ===
    print_header("1. DATASET OVERVIEW")
    print_stat("Total trajectories", ds.num_trajectories)
    print_stat("Total frames", ds.num_total_frames)
    print_stat("Total duration", ds.total_duration_s, "s")
    print_stat("Avg frames per trajectory", ds.num_total_frames / max(1, ds.num_trajectories), "")
    print_stat("Avg duration per trajectory", ds.total_duration_s / max(1, ds.num_trajectories), "s")

    # === 2. Label Distribution ===
    print_header("2. LABEL DISTRIBUTION")

    print_subheader("2.1 Horizontal Trend Classes (13-class)")
    if ds.global_trend_classes:
        total_labels = sum(ds.global_trend_classes.values())
        print(f"  Total labeled frames: {total_labels:,}")
        # Group into normal (indices 0-10) and recovery (indices 11-12 or similar)
        # We'll just print all
        print_bar_chart(dict(ds.global_trend_classes), max_bars=15, width=35)
        # Highlight recovery vs normal
        normal_count = sum(c for k, c in ds.global_trend_classes.items() if k < 11)
        recovery_count = sum(c for k, c in ds.global_trend_classes.items() if k >= 11)
        if total_labels > 0:
            print(f"\n  {Term.GREEN}Normal frames: {normal_count:,} ({normal_count/total_labels*100:.1f}%){Term.RESET}")
            print(f"  {Term.YELLOW}Recovery frames: {recovery_count:,} ({recovery_count/total_labels*100:.1f}%){Term.RESET}")
    else:
        print("  No trend class data found")

    print_subheader("2.2 Guide Elevation Bins")
    if ds.global_elevation_bins:
        print_bar_chart(dict(ds.global_elevation_bins), max_bars=15, width=35)
    else:
        print("  No elevation bin data found")

    print_subheader("2.3 Guide Mode Distribution")
    if ds.global_guide_modes:
        print_bar_chart(dict(ds.global_guide_modes), max_bars=10, width=35)
    else:
        print("  No guide mode data found")

    print_subheader("2.4 Label Validity Rates (per-frame average)")
    if ds.trajectory_metrics:
        avg_expert = np.mean([m.expert_label_valid_rate for m in ds.trajectory_metrics if m.expert_label_valid_rate is not None])
        avg_trend = np.mean([m.trend_label_valid_rate for m in ds.trajectory_metrics if m.trend_label_valid_rate is not None])
        avg_h_loss = np.mean([m.trend_h_loss_valid_rate for m in ds.trajectory_metrics if m.trend_h_loss_valid_rate is not None])
        avg_v_loss = np.mean([m.trend_v_loss_valid_rate for m in ds.trajectory_metrics if m.trend_v_loss_valid_rate is not None])
        avg_val_loss = np.mean([m.trend_value_loss_valid_rate for m in ds.trajectory_metrics if m.trend_value_loss_valid_rate is not None])
        avg_ctrl_loss = np.mean([m.control_loss_valid_rate for m in ds.trajectory_metrics if m.control_loss_valid_rate is not None])
        print_stat("Expert label valid", avg_expert, "", good=avg_expert >= 0.99, warn_thresh=0.95)
        print_stat("Trend label valid", avg_trend, "", good=avg_trend >= 0.99, warn_thresh=0.95)
        print_stat("Horizontal loss valid", avg_h_loss, "", good=avg_h_loss >= 0.99, warn_thresh=0.95)
        print_stat("Vertical loss valid", avg_v_loss, "", good=avg_v_loss >= 0.99, warn_thresh=0.95)
        print_stat("Value loss valid", avg_val_loss, "", good=avg_val_loss >= 0.99, warn_thresh=0.95)
        print_stat("Control loss valid", avg_ctrl_loss, "", good=avg_ctrl_loss >= 0.99, warn_thresh=0.95)

    # === 3. Trajectory Quality ===
    print_header("3. TRAJECTORY QUALITY")

    # Collision
    print_subheader("3.1 Collision Statistics")
    coll_rates = [m.collision_rate for m in ds.trajectory_metrics if m.collision_rate is not None]
    if coll_rates:
        print_stat("Mean collision rate", np.mean(coll_rates))
        print_stat("Max collision rate", np.max(coll_rates))
        print_stat("Trajs with zero collisions", sum(1 for r in coll_rates if r == 0), f"/ {len(coll_rates)}")
        print_stat("Trajs with >5% collisions", sum(1 for r in coll_rates if r > 0.05), f"/ {len(coll_rates)}")

    # Goal reaching
    print_subheader("3.2 Goal Reaching")
    goal_errors = [m.final_goal_error_m for m in ds.trajectory_metrics if m.final_goal_error_m is not None]
    if goal_errors:
        print_stat("Mean final goal error", np.mean(goal_errors), "m")
        print_stat("Median final goal error", np.median(goal_errors), "m")
        print_stat("Max final goal error", np.max(goal_errors), "m")
    print_stat("Exit reasons", "")
    print_bar_chart(dict(ds.exit_reasons), max_bars=8, width=30)

    # Clearance
    print_subheader("3.3 Clearance (ESDF)")
    mean_clearances = [m.mean_clearance for m in ds.trajectory_metrics if m.mean_clearance is not None]
    min_clearances = [m.min_clearance for m in ds.trajectory_metrics if m.min_clearance is not None]
    if mean_clearances:
        print_stat("Avg mean clearance", np.mean(mean_clearances), "m")
        print_stat("Avg min clearance", np.mean(min_clearances), "m")
        print_stat("Worst min clearance", np.min(min_clearances), "m",
                   good=np.min(min_clearances) > 0.05, warn_thresh=0.1)

    # Smoothness
    print_subheader("3.4 Trajectory Smoothness")
    jerks = [m.jerk_rms for m in ds.trajectory_metrics if m.jerk_rms is not None]
    accels = [m.acceleration_rms for m in ds.trajectory_metrics if m.acceleration_rms is not None]
    if jerks:
        print_stat("Mean jerk RMS", np.mean(jerks), "m/s³")
        print_stat("Mean acceleration RMS", np.mean(accels), "m/s²")

    # Speed
    print_subheader("3.5 Speed Profile")
    speeds = [m.mean_speed for m in ds.trajectory_metrics if m.mean_speed is not None]
    max_speeds = [m.max_speed for m in ds.trajectory_metrics if m.max_speed is not None]
    if speeds:
        print_stat("Mean speed", np.mean(speeds), "m/s")
        print_stat("Mean max speed", np.mean(max_speeds), "m/s")
        print_stat("Max speed (any frame)", np.max(max_speeds), "m/s")

    # Tracking
    print_subheader("3.6 Command Tracking")
    vt = [m.velocity_tracking_rmse for m in ds.trajectory_metrics if m.velocity_tracking_rmse is not None]
    yt = [m.yaw_rate_tracking_rmse for m in ds.trajectory_metrics if m.yaw_rate_tracking_rmse is not None]
    if vt:
        print_stat("Mean velocity tracking RMSE", np.mean(vt), "m/s")
        print_stat("Mean yaw rate tracking RMSE", np.mean(yt), "rad/s")

    # === 4. Planner Performance ===
    print_header("4. PLANNER PERFORMANCE")
    psr = [m.planner_success_rate for m in ds.trajectory_metrics if m.planner_success_rate is not None]
    pmt = [m.mean_planning_ms for m in ds.trajectory_metrics if m.mean_planning_ms is not None]
    pp95 = [m.p95_planning_ms for m in ds.trajectory_metrics if m.p95_planning_ms is not None]
    if psr:
        print_stat("Mean planner success rate", np.mean(psr), "", good=np.mean(psr) >= 0.99, warn_thresh=0.95)
    if pmt:
        print_stat("Mean planning time", np.mean(pmt), "ms")
        print_stat("P95 planning time", np.mean(pp95), "ms")
        print_stat("Max planning time", np.max(pp95) if pp95 else 0, "ms")

    total_replans = sum(m.num_replans or 0 for m in ds.trajectory_metrics)
    total_failed = sum(m.num_failed_replans or 0 for m in ds.trajectory_metrics)
    print_stat("Total replans", total_replans)
    print_stat("Total failed replans", total_failed)

    # === 5. Recovery Statistics ===
    print_header("5. RECOVERY STATISTICS")
    recovery_rates = [m.recovery_rate for m in ds.trajectory_metrics if m.recovery_rate is not None]
    if recovery_rates:
        print_stat("Mean recovery frame rate", np.mean(recovery_rates))
        print_stat("Trajs with any recovery", sum(1 for r in recovery_rates if r > 0), f"/ {len(recovery_rates)}")
        print_stat("Total recovery frames", ds.total_recovery_frames)

    # === 6. Per-Scene Summary ===
    print_header("6. PER-SCENE SUMMARY")
    if ds.scene_stats:
        print(f"  {'Scene':<30s} {'Trajs':>6s} {'Frames':>8s} {'Coll%':>7s} {'GoalErr':>8s}")
        print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")
        for sid in sorted(ds.scene_stats.keys()):
            ss = ds.scene_stats[sid]
            coll_rate = ss["collision_frames"] / max(1, ss["total_frames"])
            goal_err = np.mean(ss["goal_errors"]) if ss["goal_errors"] else float("nan")
            goal_str = f"{goal_err:.3f}" if not math.isnan(goal_err) else "N/A"
            print(f"  {sid:<30s} {ss['num_trajs']:>6d} {ss['total_frames']:>8,d} {coll_rate:>6.1%} {goal_str:>8s}")

    # === 7. Per-Profile Summary ===
    if ds.profile_stats and len(ds.profile_stats) > 1:
        print_header("7. PER-PROFILE SUMMARY")
        print(f"  {'Profile':<35s} {'Trajs':>6s} {'Frames':>8s} {'GoalErr':>8s} {'Coll%':>7s} {'Speed':>7s}")
        print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
        for pid in sorted(ds.profile_stats.keys()):
            ps = ds.profile_stats[pid]
            goal_err = np.mean(ps["goal_errors"]) if ps["goal_errors"] else float("nan")
            coll_rate = np.mean(ps["collision_rates"]) if ps["collision_rates"] else float("nan")
            speed = np.mean(ps["mean_speeds"]) if ps["mean_speeds"] else float("nan")
            gs = f"{goal_err:.3f}" if not math.isnan(goal_err) else "N/A"
            cs = f"{coll_rate:.1%}" if not math.isnan(coll_rate) else "N/A"
            ss = f"{speed:.2f}" if not math.isnan(speed) else "N/A"
            print(f"  {pid:<35s} {ps['num_trajs']:>6d} {ps['total_frames']:>8,d} {gs:>8s} {cs:>7s} {ss:>7s}")

    # === 8. Warnings & Recommendations ===
    print_header("8. WARNINGS & RECOMMENDATIONS")
    warnings_list = []

    if psr and np.mean(psr) < 0.95:
        warnings_list.append(f"⚠  Planner success rate is {np.mean(psr):.2%} (below 95%)")
    if goal_errors and np.mean(goal_errors) > 0.5:
        warnings_list.append(f"⚠  Mean goal error is {np.mean(goal_errors):.3f}m (above 0.5m)")
    if coll_rates and np.mean(coll_rates) > 0.01:
        warnings_list.append(f"⚠  Mean collision rate is {np.mean(coll_rates):.2%} (above 1%)")
    if ds.num_trajectories == 0:
        warnings_list.append("⚠  No valid trajectories found!")
    recovery_count_total = sum(1 for m in ds.trajectory_metrics if (m.recovery_rate or 0) > 0.2)
    if recovery_count_total > ds.num_trajectories * 0.3:
        warnings_list.append(f"⚠  {recovery_count_total}/{ds.num_trajectories} trajectories have >20% recovery frames")
    if ds.num_trajectories < 10:
        warnings_list.append(f"ℹ  Small dataset ({ds.num_trajectories} trajectories). Consider collecting more data.")

    if warnings_list:
        for w in warnings_list:
            print(f"  {w}")
    else:
        print(f"  {Term.GREEN}✓ No major issues detected{Term.RESET}")

    print_header("END OF REPORT")


# ============================================================================
# Visualizations
# ============================================================================

def generate_plots(ds: DatasetSummary, output_dir: str, data_root: str):
    """Generate visualization plots."""
    if not HAS_MPL:
        print(f"\n{Term.YELLOW}⚠ matplotlib not available — skipping plots{Term.RESET}")
        return

    os.makedirs(output_dir, exist_ok=True)
    trajs = ds.trajectory_metrics

    # ---- Plot 1: Label Distribution ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Label Distribution", fontsize=14, fontweight="bold")

    # Horizontal trend classes
    ax = axes[0]
    if ds.global_trend_classes:
        items = sorted(ds.global_trend_classes.items())
        labels = [str(k) for k, _ in items]
        values = [v for _, v in items]
        colors = ["#3498db" if int(k) < 11 else "#e74c3c" for k, _ in items]
        ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Frame Count")
        ax.set_title("Horizontal Trend Classes\n(blue=normal, red=recovery)")
        ax.grid(axis="y", alpha=0.3)
        # Add percentage annotations for top classes
        total = sum(values)
        top_n = min(5, len(items))
        for i in range(top_n):
            pct = values[i] / total * 100
            ax.text(i, values[i], f"{pct:.1f}%", ha="center", va="bottom", fontsize=7, color="#555555")

    # Elevation bins
    ax = axes[1]
    if ds.global_elevation_bins:
        items = sorted(ds.global_elevation_bins.items())
        ax.bar([str(k) for k, _ in items], [v for _, v in items],
               color="#2ecc71", edgecolor="white", linewidth=0.5)
        ax.set_title("Guide Elevation Bins")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Frame Count")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    # Guide modes
    ax = axes[2]
    if ds.global_guide_modes:
        items = sorted(ds.global_guide_modes.items(), key=lambda x: -x[1])
        labels = [k[:20] for k, _ in items]
        values = [v for _, v in items]
        mode_colors = {"NORMAL": "#3498db", "RECOVER_LEFT": "#e67e22", "RECOVER_RIGHT": "#e67e22",
                        "HOVER": "#9b59b6"}
        colors = [mode_colors.get(str(l), "#95a5a6") for l, _ in items]
        ax.bar(range(len(values)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title("Guide Modes")
        ax.set_ylabel("Frame Count")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "01_label_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---- Plot 2: Trajectory Quality Overview ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Trajectory Quality Overview", fontsize=14, fontweight="bold")

    # Collision rate histogram
    ax = axes[0][0]
    coll_rates = [m.collision_rate or 0 for m in trajs]
    ax.hist(coll_rates, bins=30, color="#e74c3c", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(coll_rates), color="black", linestyle="--", label=f"Mean={np.mean(coll_rates):.3f}")
    ax.set_xlabel("Collision Rate")
    ax.set_ylabel("Trajectories")
    ax.set_title("Collision Rate Distribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Goal error histogram
    ax = axes[0][1]
    goal_errs = [m.final_goal_error_m for m in trajs if m.final_goal_error_m is not None]
    if goal_errs:
        ax.hist(goal_errs, bins=30, color="#2ecc71", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(goal_errs), color="black", linestyle="--", label=f"Mean={np.mean(goal_errs):.3f}m")
        ax.set_xlabel("Goal Error (m)")
        ax.set_ylabel("Trajectories")
        ax.set_title("Final Goal Error Distribution")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Mean clearance histogram
    ax = axes[0][2]
    clearances = [m.mean_clearance for m in trajs if m.mean_clearance is not None]
    if clearances:
        ax.hist(clearances, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(clearances), color="black", linestyle="--", label=f"Mean={np.mean(clearances):.3f}m")
        ax.set_xlabel("Mean Clearance (m)")
        ax.set_ylabel("Trajectories")
        ax.set_title("ESDF Clearance Distribution")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Speed histogram
    ax = axes[1][0]
    speeds = [m.mean_speed for m in trajs if m.mean_speed is not None]
    if speeds:
        ax.hist(speeds, bins=30, color="#f39c12", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(speeds), color="black", linestyle="--", label=f"Mean={np.mean(speeds):.2f}m/s")
        ax.set_xlabel("Mean Speed (m/s)")
        ax.set_ylabel("Trajectories")
        ax.set_title("Speed Distribution")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Jerk histogram
    ax = axes[1][1]
    jerks = [m.jerk_rms for m in trajs if m.jerk_rms is not None and m.jerk_rms < 500]
    if jerks:
        ax.hist(jerks, bins=30, color="#9b59b6", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(jerks), color="black", linestyle="--", label=f"Mean={np.mean(jerks):.1f}")
        ax.set_xlabel("Jerk RMS (m/s³)")
        ax.set_ylabel("Trajectories")
        ax.set_title("Trajectory Smoothness (Jerk)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Frame count per trajectory
    ax = axes[1][2]
    frame_counts = [m.num_frames for m in trajs]
    ax.bar(range(len(frame_counts)), sorted(frame_counts, reverse=True),
           color="#1abc9c", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Trajectory Rank")
    ax.set_ylabel("Frames")
    ax.set_title("Frame Count per Trajectory (sorted)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "02_quality_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---- Plot 3: Trajectory Paths Overlay (sample) ----
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Trajectory Paths (XY Projection)", fontsize=14, fontweight="bold")

        # All trajectories overlaid
        ax = axes[0]
        for i, m in enumerate(trajs):
            if m.positions is not None and len(m.positions) > 1:
                alpha = min(0.3, 5.0 / max(1, ds.num_trajectories))
                color = "#e74c3c" if (m.collision_rate or 0) > 0 else "#3498db"
                ax.plot(m.positions[:, 0], m.positions[:, 1], linewidth=0.5, alpha=alpha, color=color)
        # Plot start/goal markers for first few
        for m in trajs[:5]:
            if m.start_pos is not None:
                ax.scatter(*m.start_pos[:2], marker="o", s=30, c="green", edgecolors="black", linewidths=0.5, zorder=5)
            if m.goal_pos is not None:
                ax.scatter(*m.goal_pos[:2], marker="*", s=60, c="gold", edgecolors="black", linewidths=0.5, zorder=5)
        ax.set_xlabel("X World (m)")
        ax.set_ylabel("Y World (m)")
        ax.set_title(f"All {ds.num_trajectories} Trajectories\n(blue=no collision, red=has collision)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        # Single representative trajectory with detail
        ax = axes[1]
        if trajs and trajs[0].positions is not None:
            m = trajs[0]
            pos = m.positions
            # Color by speed if available, else use time
            if m.speed_profile is not None and len(m.speed_profile) == len(pos):
                colors = m.speed_profile
                cmap = plt.cm.viridis
            else:
                colors = np.linspace(0, 1, len(pos))
                cmap = plt.cm.plasma
            scatter = ax.scatter(pos[:, 0], pos[:, 1], c=colors, cmap=cmap, s=2, alpha=0.8)
            plt.colorbar(scatter, ax=ax, label="Speed (m/s)" if m.speed_profile is not None else "Time")

            if m.start_pos is not None:
                ax.scatter(*m.start_pos[:2], marker="o", s=80, c="green", edgecolors="black", linewidths=1,
                          zorder=5, label="Start")
            if m.goal_pos is not None:
                ax.scatter(*m.goal_pos[:2], marker="*", s=120, c="gold", edgecolors="black", linewidths=1,
                          zorder=5, label="Goal")
            # Add obstacle circles based on metadata if available
            ax.set_xlabel("X World (m)")
            ax.set_ylabel("Y World (m)")
            ax.set_title(f"Sample Trajectory: {m.traj_id}")
            ax.set_aspect("equal")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(output_dir, "03_trajectory_paths.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")

    # ---- Plot 4: Planner Performance ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Planner Performance", fontsize=14, fontweight="bold")

    # Planning time histogram
    ax = axes[0]
    ptimes = [m.mean_planning_ms for m in trajs if m.mean_planning_ms is not None and m.mean_planning_ms > 0]
    if ptimes:
        ax.hist(ptimes, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(ptimes), color="black", linestyle="--", label=f"Mean={np.mean(ptimes):.2f}ms")
        ax.set_xlabel("Mean Planning Time (ms)")
        ax.set_ylabel("Trajectories")
        ax.set_title("Planning Time Distribution")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Planner success rate
    ax = axes[1]
    psr = [m.planner_success_rate for m in trajs if m.planner_success_rate is not None]
    if psr:
        bins = np.linspace(0, 1, 21)
        ax.hist(psr, bins=bins, color="#2ecc71", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(psr), color="black", linestyle="--", label=f"Mean={np.mean(psr):.2%}")
        ax.set_xlabel("Success Rate")
        ax.set_ylabel("Trajectories")
        ax.set_title("Planner Success Rate")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Replans per trajectory
    ax = axes[2]
    replans = [m.num_replans or 0 for m in trajs]
    failed = [m.num_failed_replans or 0 for m in trajs]
    x = np.arange(min(10, len(replans)))
    wid = 0.35
    # Show top 10 by replan count
    sorted_idx = np.argsort(replans)[::-1][:10]
    ax.bar(x - wid/2, [replans[i] for i in sorted_idx], wid, label="Replans", color="#3498db", edgecolor="white")
    ax.bar(x + wid/2, [failed[i] for i in sorted_idx], wid, label="Failed", color="#e74c3c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in sorted_idx], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Count")
    ax.set_title("Top 10 Trajs by Replan Count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "04_planner_performance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # ---- Plot 5: Time-Series Profiles (sample traj) ----
    if trajs and trajs[0].speed_profile is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(f"Time-Series Profiles — {trajs[0].traj_id}", fontsize=14, fontweight="bold")

        m = trajs[0]
        t = m.duration_s
        if m.num_frames > 1 and t > 0:
            time = np.linspace(0, t, m.num_frames)

            # Speed
            ax = axes[0][0]
            if m.speed_profile is not None:
                ax.plot(time, m.speed_profile, linewidth=0.5, color="#3498db")
                ax.set_ylabel("Speed (m/s)")
                ax.set_title("Speed Profile")
                ax.grid(alpha=0.3)

            # Clearance
            ax = axes[0][1]
            if m.clearance_profile is not None:
                ax.plot(time, m.clearance_profile, linewidth=0.5, color="#2ecc71")
                ax.axhline(y=0.05, color="red", linestyle="--", linewidth=0.5, label="Danger")
                ax.set_ylabel("Clearance (m)")
                ax.set_title("ESDF Clearance Profile")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)

            # Yaw rate
            ax = axes[1][0]
            if m.yaw_rate_profile is not None:
                ax.plot(time, m.yaw_rate_profile, linewidth=0.5, color="#e67e22")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Yaw Rate (rad/s)")
                ax.set_title("Yaw Rate Profile")
                ax.grid(alpha=0.3)

            # Altitude (Z)
            ax = axes[1][1]
            if m.positions is not None:
                ax.plot(time, m.positions[:, 2], linewidth=0.5, color="#9b59b6")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Z (m)")
                ax.set_title("Altitude Profile")
                ax.grid(alpha=0.3)

        plt.tight_layout()
        out = os.path.join(output_dir, "05_time_series_profiles.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ============================================================================
# JSON Export
# ============================================================================

def export_json(ds: DatasetSummary, output_dir: str, data_root: str):
    """Export a structured JSON summary."""
    report = {
        "dataset_root": data_root,
        "evaluated_at": datetime.now().isoformat(),
        "overview": {
            "num_trajectories": ds.num_trajectories,
            "num_total_frames": ds.num_total_frames,
            "total_duration_s": ds.total_duration_s,
            "avg_frames_per_trajectory": ds.num_total_frames / max(1, ds.num_trajectories),
            "avg_duration_per_trajectory_s": ds.total_duration_s / max(1, ds.num_trajectories),
        },
        "label_distribution": {
            "horizontal_trend_classes": {str(k): v for k, v in ds.global_trend_classes.items()},
            "guide_elevation_bins": {str(k): v for k, v in ds.global_elevation_bins.items()},
            "guide_modes": {str(k): v for k, v in ds.global_guide_modes.items()},
            "total_collision_frames": ds.total_collision_frames,
            "total_recovery_frames": ds.total_recovery_frames,
        },
        "exit_reasons": dict(ds.exit_reasons),
        "quality_summary": {},
        "planner_summary": {},
        "per_scene": {},
        "per_profile": {},
        "per_trajectory": [],
    }

    # Quality summary
    trajs = ds.trajectory_metrics
    for key, vals_fn in [
        ("collision_rate", lambda: [(m.collision_rate or 0) for m in trajs]),
        ("final_goal_error_m", lambda: [m.final_goal_error_m for m in trajs if m.final_goal_error_m is not None]),
        ("mean_clearance_m", lambda: [m.mean_clearance for m in trajs if m.mean_clearance is not None]),
        ("min_clearance_m", lambda: [m.min_clearance for m in trajs if m.min_clearance is not None]),
        ("mean_speed_mps", lambda: [m.mean_speed for m in trajs if m.mean_speed is not None]),
        ("jerk_rms", lambda: [m.jerk_rms for m in trajs if m.jerk_rms is not None]),
        ("velocity_tracking_rmse", lambda: [m.velocity_tracking_rmse for m in trajs if m.velocity_tracking_rmse is not None]),
    ]:
        vals = vals_fn()
        if vals:
            report["quality_summary"][key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "std": float(np.std(vals)),
            }

    # Planner summary
    for key, vals_fn in [
        ("planner_success_rate", lambda: [m.planner_success_rate for m in trajs if m.planner_success_rate is not None]),
        ("mean_planning_ms", lambda: [m.mean_planning_ms for m in trajs if m.mean_planning_ms is not None]),
        ("p95_planning_ms", lambda: [m.p95_planning_ms for m in trajs if m.p95_planning_ms is not None]),
    ]:
        vals = vals_fn()
        if vals:
            report["planner_summary"][key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "max": float(np.max(vals)),
            }

    # Per scene
    for sid, ss in ds.scene_stats.items():
        report["per_scene"][sid] = {
            "num_trajectories": ss["num_trajs"],
            "total_frames": ss["total_frames"],
            "collision_rate": ss["collision_frames"] / max(1, ss["total_frames"]),
            "mean_goal_error_m": float(np.mean(ss["goal_errors"])) if ss["goal_errors"] else None,
        }

    # Per profile
    for pid, ps in ds.profile_stats.items():
        report["per_profile"][pid] = {
            "num_trajectories": ps["num_trajs"],
            "total_frames": ps["total_frames"],
            "mean_goal_error_m": float(np.mean(ps["goal_errors"])) if ps["goal_errors"] else None,
            "mean_collision_rate": float(np.mean(ps["collision_rates"])) if ps["collision_rates"] else None,
            "mean_speed_mps": float(np.mean(ps["mean_speeds"])) if ps["mean_speeds"] else None,
            "mean_duration_s": float(np.mean(ps["durations"])) if ps["durations"] else None,
        }

    # Per trajectory (compact)
    for m in trajs:
        report["per_trajectory"].append({
            "traj_id": m.traj_id,
            "scene_id": m.scene_id,
            "num_frames": m.num_frames,
            "duration_s": m.duration_s,
            "collision_rate": m.collision_rate,
            "final_goal_error_m": m.final_goal_error_m,
            "exit_reason": m.exit_reason,
            "mean_speed": m.mean_speed,
            "mean_clearance": m.mean_clearance,
            "planner_success_rate": m.planner_success_rate,
            "recovery_rate": m.recovery_rate,
        })

    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved JSON report: {json_path}")
    return report


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="IL Dataset Evaluator — analyze collected imitation learning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_evaluator.py --mode full
  python dataset_evaluator.py --mode labels --data-dir ./il_data
  python dataset_evaluator.py --mode planner --no-plots
  python dataset_evaluator.py --mode scene --output-dir ./reports
        """,
    )
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "labels", "quality", "planner", "scene", "compact"],
                        help="Evaluation mode: full (all), labels, quality, planner, scene, compact (terminal only)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to dataset root directory (auto-detected if not specified)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for saving plots and JSON report (default: <data-dir>/_eval/)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip JSON export")
    parser.add_argument("--max-trajs", type=int, default=0,
                        help="Limit number of trajectories to evaluate (0 = all)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine data directory
    if args.data_dir:
        data_root = args.data_dir
    else:
        # Auto-detect from common locations
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "il_data"),
            os.path.join(os.getcwd(), "il_data"),
            "g:\\Code\\flightmare_ws\\il_data",
        ]
        data_root = find_latest_dataset(search_paths)
        if not data_root:
            print(f"{Term.RED}Error: No dataset found. Specify --data-dir explicitly.{Term.RESET}")
            sys.exit(1)

    print(f"{Term.CYAN}Dataset root: {data_root}{Term.RESET}")

    # Discover trajectories
    traj_dirs = discover_datasets(data_root)
    if not traj_dirs:
        print(f"{Term.RED}Error: No valid trajectory directories found in {data_root}{Term.RESET}")
        sys.exit(1)

    traj_dirs = sorted(traj_dirs)
    if args.max_trajs > 0:
        traj_dirs = traj_dirs[:args.max_trajs]

    print(f"{Term.CYAN}Found {len(traj_dirs)} trajectory directories{Term.RESET}")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(data_root, "_eval")
    os.makedirs(output_dir, exist_ok=True)

    # Load all trajectories
    ds = DatasetSummary()
    loaded = 0
    skipped = 0

    for i, td in enumerate(traj_dirs):
        csv_path = os.path.join(td, "data.csv")
        meta_path = os.path.join(td, "metadata.json")

        data = load_trajectory_csv(csv_path)
        meta = load_metadata(meta_path)

        if data is None or len(data) == 0:
            skipped += 1
            print(f"  [{i+1}/{len(traj_dirs)}] {os.path.basename(td)} ... {Term.YELLOW}SKIPPED (empty/invalid){Term.RESET}")
            continue

        # Extract scene/task from data or path
        scene_id = ""
        task_id = ""
        try:
            if SCENE_ID_COL in data.dtype.names:
                vals = data[SCENE_ID_COL]
                scene_id = str(vals[0]) if len(vals) > 0 else ""
            if TASK_ID_COL in data.dtype.names:
                vals = data[TASK_ID_COL]
                task_id = str(vals[0]) if len(vals) > 0 else ""
        except Exception:
            pass

        if not scene_id:
            # Derive from path: il_data/scene_xxx/traj_NNN
            parts = Path(td).parts
            scene_id = parts[-2] if len(parts) >= 2 else "unknown"
        traj_name = Path(td).name

        tm = compute_trajectory_metrics(data, meta, traj_name, scene_id, task_id)
        ds.add_trajectory(tm)
        loaded += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(traj_dirs):
            print(f"  [{i+1}/{len(traj_dirs)}] Loaded {loaded} trajectories, skipped {skipped} ...")

    if loaded == 0:
        print(f"{Term.RED}Error: No valid trajectories could be loaded.{Term.RESET}")
        sys.exit(1)

    # Generate report based on mode
    if args.mode in ("full", "compact"):
        generate_full_report(ds, traj_dirs, data_root)
    elif args.mode == "labels":
        print_header("LABEL DISTRIBUTION REPORT")
        print_subheader("Horizontal Trend Classes")
        print_bar_chart(dict(ds.global_trend_classes), max_bars=15)
        print_subheader("Guide Elevation Bins")
        print_bar_chart(dict(ds.global_elevation_bins), max_bars=15)
        print_subheader("Guide Modes")
        print_bar_chart(dict(ds.global_guide_modes), max_bars=10)
    elif args.mode == "quality":
        print_header("TRAJECTORY QUALITY REPORT")
        for m in ds.trajectory_metrics:
            print(f"\n  {m.traj_id}:")
            print_stat("  Frames", m.num_frames)
            print_stat("  Collision rate", m.collision_rate or 0)
            print_stat("  Goal error", m.final_goal_error_m or float("nan"), "m")
            print_stat("  Mean clearance", m.mean_clearance or 0, "m")
            print_stat("  Jerk RMS", m.jerk_rms or 0, "m/s³")
            print_stat("  Exit reason", m.exit_reason)
    elif args.mode == "planner":
        print_header("PLANNER PERFORMANCE REPORT")
        for m in ds.trajectory_metrics:
            print(f"\n  {m.traj_id}:")
            print_stat("  Success rate", m.planner_success_rate or 0)
            print_stat("  Mean time", m.mean_planning_ms or 0, "ms")
            print_stat("  P95 time", m.p95_planning_ms or 0, "ms")
            print_stat("  Replans", m.num_replans or 0)
            print_stat("  Failed", m.num_failed_replans or 0)
    elif args.mode == "scene":
        generate_full_report(ds, traj_dirs, data_root)

    # Plots
    if not args.no_plots and args.mode in ("full",):
        print(f"\n{Term.CYAN}Generating plots...{Term.RESET}")
        generate_plots(ds, output_dir, data_root)
    elif args.no_plots:
        print(f"\n{Term.DIM}Plots disabled (--no-plots){Term.RESET}")

    # JSON export
    if not args.no_json and args.mode in ("full",):
        export_json(ds, output_dir, data_root)

    print(f"\n{Term.GREEN}✓ Evaluation complete. Output: {output_dir}{Term.RESET}")


if __name__ == "__main__":
    main()
