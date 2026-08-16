#!/usr/bin/env python3
"""
il_dataset_writer.py  —  Dataset recording for the privileged micro-detour
expert.

Owns the per-episode directory, the v25 student/label CSV schema, depth
PNGs, privileged micro-plan debug traces and the atomic commit / reject
lifecycle.
"""

from __future__ import print_function, division

import csv
import json
import os
import shutil
import time

import numpy as np


def _wall_time_ns():
    """Python 3.6-compatible diagnostic wall-clock timestamp."""
    if hasattr(time, "time_ns"):
        return time.time_ns()
    return int(time.time() * 1000000000)

# ── Schema (v25 + two_level_expert_labels_v1) — NEW architecture ─────
# Order matters: this is the CSV header.  All fields are written on every
# row; missing keys are filled with defaults by write_row().
#
# The old micro-detour / goal-switching diagnostics (micro_detour_*,
# goal_switch_*, local_target_*, desired_yaw_*, abrupt_target_change_event
# ...) are REMOVED — the new architecture has no such semantic.
#
# STUDENT INPUTS (future LSTM local-avoidance network, 30 Hz):
#   depth_file, gravity_flu_*, velocity_flu_* (current), yaw_rate,
#   goal_direction_flu_*, goal_distance_clipped_m, goal_distance_norm
#   (goal_* = the CURRENT effective target of the hierarchical expert:
#   PASS: original goal; NORMAL: live re-expressed world-latched
#   correction; TURN: live re-expressed world-latched turn direction with
#   distance_norm EXACTLY 1 — NEVER the raw navigation_goal)
# SUPERVISION LABELS (30 Hz):
#   target_velocity_flu_*, target_yaw_rate — the FINAL command actually
#   sent to the Flightmare backend (velocity/yaw-rate constrained; NOT an
#   intent, NOT a trajectory-end velocity).
#   (velocity_command_flu_* / yaw_rate_command are identical aliases)
# 5 Hz SUPERVISION (two_level_expert_labels_v1):
#   macro_update_mask / macro_label_valid / macro_correction_type /
#   macro_direction_token / macro_direction_flu_* / macro_distance_norm /
#   macro_param_valid
# 5 Hz STUDENT INPUTS:
#   navigation_goal_direction_flu_* / navigation_goal_distance_clipped_m /
#   navigation_goal_distance_norm   (the ORIGINAL goal, never the
#   effective goal_*)
# hierarchical_mode: the FULL new-architecture expert state (direct /
#   local_avoidance / macro_normal / macro_turn_left / macro_turn_right /
#   turn_to_target / goal_capture / blocked).  This is the ONLY mode label
#   (no legacy lossy expert_mode projection is kept; save_net reads
#   hierarchical_mode directly).
# Everything else is a privileged / expert diagnostic and is NEVER a
# student input.
DATA_SCHEMA_V25_FIELDS = [
    # time / matching
    "timestamp_ns", "receive_timestamp_ns", "episode_id", "frame_id",
    "episode_frame_index", "control_dt_s", "trajectory_time_s",
    "latency_ms", "match_method", "frame_valid", "frame_invalid_reason",
    # state (diagnostic + used to derive FLU inputs)
    "x", "y", "z", "qx", "qy", "qz", "qw",
    "state_vx_world", "state_vy_world", "state_vz_world",
    "state_vx_flu", "state_vy_flu", "state_vz_flu",
    "yaw", "yaw_rate",
    # ── student inputs (30 Hz) ─────────────────────────────────────
    "depth_file", "raw_depth_finite_ratio",
    "gravity_flu_x", "gravity_flu_y", "gravity_flu_z",
    "velocity_flu_x", "velocity_flu_y", "velocity_flu_z",
    "yaw_rate_flu",
    "goal_direction_flu_x", "goal_direction_flu_y", "goal_direction_flu_z",
    "goal_distance_clipped_m", "goal_distance_norm",
    # ── supervision labels (30 Hz) — the ACTUAL sent command ───────
    "target_velocity_flu_x", "target_velocity_flu_y", "target_velocity_flu_z",
    "target_yaw_rate",
    # compatibility aliases (identical to target_* / documented in metadata)
    "velocity_command_flu_x", "velocity_command_flu_y",
    "velocity_command_flu_z", "yaw_rate_command",
    # ── expert diagnostics (never student inputs) ──────────────────
    "hierarchical_mode", "planner_status",
    "planner_failure_reason", "fsm_state", "effective_target_source",
    "target_correction_active", "effective_direction_token",
    "directive_update_event", "mission_revision", "reentry_guard_ticks",
    "obstacle_first_observed_event", "selected_output_speed_mps",
    "local_target_distance_m", "min_observed_clearance_m",
    "obstacle_risk_cost", "avoidance_active", "local_corridor_blocked",
    "emergency_brake", "immediate_avoidance", "local_limit_cycle_detected",
    "target_bearing_error_deg", "consecutive_failures_30hz",
    "unknown_recovery_ticks",
    # ── world-frame diagnostics (privileged, never student inputs) ─
    "navigation_goal_world_x", "navigation_goal_world_y",
    "navigation_goal_world_z",
    "effective_target_world_x", "effective_target_world_y",
    "effective_target_world_z",
    "original_navigation_goal_world_x", "original_navigation_goal_world_y",
    "original_navigation_goal_world_z",
    # truth (exact generated cylinders) continuous audit of the executed
    # trajectory (judge only)
    "truth_minimum_clearance_m", "truth_brake_risk", "observed_brake_risk",
    "truth_brake_would_trigger",
    # episode
    "scene_id", "task_id", "episode_valid",
    "failure_taxonomy", "failure_reason",
    # ── 5 Hz supervision (two_level_expert_labels_v1) ──────────────
    "macro_update_mask", "macro_label_valid", "macro_correction_type",
    "macro_direction_token", "macro_direction_flu_x",
    "macro_direction_flu_y", "macro_direction_flu_z", "macro_distance_norm",
    "macro_param_valid",
    # ── 5 Hz student inputs (the ORIGINAL navigation goal) ─────────
    "navigation_goal_direction_flu_x", "navigation_goal_direction_flu_y",
    "navigation_goal_direction_flu_z",
    "navigation_goal_distance_clipped_m", "navigation_goal_distance_norm",
    # ── 5 Hz / two-level privileged diagnostics (never student inputs) ─
    "correction_enter_event", "correction_exit_event",
    "correction_update_event",
    "observability_reason", "observability_goal_inside_fov",
    "observability_direct_corridor_blocked",
    "observability_left_bypass_visible", "observability_right_bypass_visible",
    "observability_local_avoidance_observable",
]

_DEFAULT_ROW = {field: "" for field in DATA_SCHEMA_V25_FIELDS}

STUDENT_INPUT_FIELDS = [
    "depth_file",
    "gravity_flu_x", "gravity_flu_y", "gravity_flu_z",
    "velocity_flu_x", "velocity_flu_y", "velocity_flu_z",
    "yaw_rate_flu",
    "goal_direction_flu_x", "goal_direction_flu_y",
    "goal_direction_flu_z",
    "goal_distance_clipped_m", "goal_distance_norm",
]

SUPERVISION_FIELDS = [
    "target_velocity_flu_x", "target_velocity_flu_y",
    "target_velocity_flu_z", "target_yaw_rate",
]

# ── two_level_expert_labels_v1: 5 Hz student input / supervision ────
STUDENT_INPUT_FIELDS_5HZ = [
    "depth_file",
    "gravity_flu_x", "gravity_flu_y", "gravity_flu_z",
    "velocity_flu_x", "velocity_flu_y", "velocity_flu_z",
    "yaw_rate_flu",
    "navigation_goal_direction_flu_x", "navigation_goal_direction_flu_y",
    "navigation_goal_direction_flu_z",
    "navigation_goal_distance_clipped_m", "navigation_goal_distance_norm",
]

SUPERVISION_FIELDS_5HZ = [
    "macro_update_mask", "macro_label_valid", "macro_correction_type",
    "macro_direction_token", "macro_direction_flu_x",
    "macro_direction_flu_y", "macro_direction_flu_z", "macro_distance_norm",
    "macro_param_valid",
]

# Full new-architecture expert states (hierarchical_mode).
HIERARCHICAL_MODES = [
    "direct", "local_avoidance", "macro_normal", "macro_turn_left",
    "macro_turn_right", "turn_to_target", "goal_capture", "blocked",
]

PRIVILEGED_DIAGNOSTIC_FIELDS = [
    "planner_status", "planner_failure_reason", "fsm_state",
    "effective_target_source", "target_correction_active",
    "effective_direction_token", "directive_update_event",
    "mission_revision", "reentry_guard_ticks",
    "obstacle_first_observed_event", "selected_output_speed_mps",
    "local_target_distance_m", "min_observed_clearance_m",
    "obstacle_risk_cost", "avoidance_active", "local_corridor_blocked",
    "emergency_brake", "immediate_avoidance", "local_limit_cycle_detected",
    "target_bearing_error_deg", "consecutive_failures_30hz",
    "unknown_recovery_ticks",
    "navigation_goal_world_x", "navigation_goal_world_y",
    "navigation_goal_world_z",
    "effective_target_world_x", "effective_target_world_y",
    "effective_target_world_z",
    "original_navigation_goal_world_x", "original_navigation_goal_world_y",
    "original_navigation_goal_world_z",
    "truth_minimum_clearance_m", "truth_brake_risk", "observed_brake_risk",
    "truth_brake_would_trigger",
    "failure_taxonomy", "failure_reason",
    # ── 5 Hz / two-level diagnostics (privileged) ─────────────────
    "correction_enter_event", "correction_exit_event",
    "correction_update_event",
    "observability_reason", "observability_goal_inside_fov",
    "observability_direct_corridor_blocked",
    "observability_left_bypass_visible", "observability_right_bypass_visible",
    "observability_local_avoidance_observable",
]


class DatasetWriter(object):
    """Per-episode writer with atomic commit / reject."""

    def __init__(self, cfg, episode_id, output_root, scene_id, task_id,
                 start_world, goal_world, initial_yaw, depth_cfg,
                 control_hz=30.0, macro_update_hz=5.0):
        self.cfg = cfg
        self._control_hz = float(control_hz)
        self._macro_update_hz = float(macro_update_hz)
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.task_id = task_id
        self.start_world = np.asarray(start_world, dtype=np.float64)
        self.goal_world = np.asarray(goal_world, dtype=np.float64)
        self.initial_yaw = float(initial_yaw)
        self.depth_cfg = depth_cfg
        # The manager passes the dataset_logging SUB-config.
        self._flush_interval_rows = int(cfg.get("flush_interval_rows", 64))
        self._depth_png_compress_level = int(
            cfg.get("depth_png_compress_level", 4))

        self._output_root = output_root
        self._inprogress_root = os.path.join(output_root, "_inprogress")
        if not os.path.isdir(self._inprogress_root):
            os.makedirs(self._inprogress_root)
        self._inprogress_dir = os.path.join(
            self._inprogress_root, "%s.inprogress" % episode_id)
        if not os.path.isdir(self._inprogress_dir):
            os.makedirs(self._inprogress_dir)

        self._data_file = os.path.join(self._inprogress_dir, "data.csv")
        self._sync_file = os.path.join(self._inprogress_dir, "sync.csv")
        self._depth_dir = os.path.join(self._inprogress_dir, "depth")
        # Depth is a mandatory student input.  The legacy depth_in_memory
        # switch skipped directory creation while write_depth() still wrote
        # files, so disabling it made collection fail at the first frame.
        os.makedirs(self._depth_dir)
        self._metadata_file = os.path.join(self._inprogress_dir, "metadata.json")

        self._data_csv = open(self._data_file, "w", newline="")
        self._data_writer = csv.DictWriter(
            self._data_csv, fieldnames=DATA_SCHEMA_V25_FIELDS)
        self._data_writer.writeheader()

        self._sync_csv = open(self._sync_file, "w", newline="")
        self._sync_writer = csv.writer(self._sync_csv)
        self._sync_writer.writerow([
            "frame_id", "latency_ms", "match_method", "is_dropped",
            "exact_matches", "unmatched_frames"])

        self._depth_rows = {}        # frame_id -> row dict (depth + meta)
        self._metadata = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "task_id": task_id,
            "start_world": self.start_world.tolist(),
            "goal_world": self.goal_world.tolist(),
            "initial_yaw": self.initial_yaw,
            "depth_config": dict(depth_cfg),
            "data_contract": (
                "Fixed cadence: local_control_hz (=30) control ticks, one "
                "row per tick at control_dt_s; macro_update_hz (=5) "
                "decisions on macro_update_mask==1 rows (every 6th control "
                "tick).  The 30 Hz sequence is continuous across the whole "
                "episode; the 5 Hz student sequence is the mask==1 rows in "
                "time order."),
            "depth_encoding_contract": {
                "format": "uint16_png",
                "png_mode": "I;16",
                "meters_per_unit": 0.01,
                "decode_formula": "depth_m = uint16_pixel / 100.0",
                "invalid_pixel_value": 0,
                "invalid_semantics": (
                    "pixel 0 = invalid / non-finite / <=0 depth (no "
                    "return).  Loaders MUST mask it to max range, never "
                    "treat it as a real 0-metre obstacle."),
                "clip_range_pixels": [0, 65535],
                "clip_range_m": [0.0, 655.35],
                "unit": "metres",
                "orientation": (
                    "row-major, top-left origin; the AvoidBench flipud is "
                    "applied before encoding, so decode needs no flip."),
            },
            "schema_version": int(cfg.get("schema_version", 25)),
            "schema_extensions": ["two_level_expert_labels_v1"],
            "expert_stack_revision": "hierarchical_local_v1",
            "hierarchical_modes": HIERARCHICAL_MODES,
            "student_input_fields_30hz": STUDENT_INPUT_FIELDS,
            "supervision_fields_30hz": SUPERVISION_FIELDS,
            "student_input_fields_5hz": STUDENT_INPUT_FIELDS_5HZ,
            "supervision_fields_5hz": SUPERVISION_FIELDS_5HZ,
            "privileged_diagnostic_fields": PRIVILEGED_DIAGNOSTIC_FIELDS,
            "macro_update_hz": self._macro_update_hz,
            "local_control_hz": self._control_hz,
            "target_encoding_contract": (
                "R = perception_range_m (5.0). ordinary target "
                "distance_norm = min(real_horizontal_distance, R - 0.5) / R "
                "< 1 (max 0.9); TURN_LEFT/RIGHT distance_norm == 1.0 exactly "
                "(pure rotation, horizontal translation 0); goal reached "
                "distance_norm == 0 with canonical direction (1,0,0). "
                "direction classes: 0=TURN_LEFT, 1..11 ordinary in-FOV bins "
                "left-to-right (includes 0 deg), 12=TURN_RIGHT.  TURN is a "
                "finite world-latched rotation step, re-expressed in the "
                "live body frame every 30 Hz tick."),
            "information_boundary_contract": (
                "30 Hz student input = effective target (PASS: original "
                "goal; NORMAL: live re-expressed world-latched correction; "
                "TURN: live re-expressed world-latched turn direction).  "
                "5 Hz student input = the ORIGINAL navigation goal, never "
                "the effective goal_*.  World-frame goals and observability "
                "fields are privileged diagnostics, never student inputs.  "
                "No global ESDF / PLY truth / global path / future "
                "observations ever generate expert actions."),
            "compatibility_aliases": {
                "velocity_command_flu_x": "target_velocity_flu_x",
                "velocity_command_flu_y": "target_velocity_flu_y",
                "velocity_command_flu_z": "target_velocity_flu_z",
                "yaw_rate_command": "target_yaw_rate",
            },
            "sequence_contract": (
                "Rows with frame_valid=1 form one independent LSTM sequence; "
                "episode_frame_index is contiguous from zero.  The 30 Hz "
                "sequence is continuous across the whole episode; 5 Hz "
                "target correction never resets the 30 Hz history; "
                "NORMAL/TURN/PASS switches never cut the sequence.  The 5 Hz "
                "student sequence is the macro_update_mask==1 rows in time "
                "order (never a repeated 5 Hz decision re-used 6 times)."),
            "local_target_contract": (
                "goal_direction_flu_* / goal_distance_* encode the CURRENT "
                "effective target of the hierarchical expert (PASS / NORMAL "
                "/ TURN), zero-order held at 30 Hz and live re-expressed "
                "every tick.  A directive update does NOT reset the LSTM / "
                "planner history.  navigation_goal_world_* is diagnostic + "
                "task termination only.  truth_brake_would_trigger is the "
                "judge-only EXACT-CYLINDER-TRUTH brake flag (continuous "
                "swept audit against the generated cylinders; there is NO "
                "real-PLY audit in this pipeline) that rejects an episode."),
            "macro_label_contract": (
                "macro_update_mask==1 marks real 5 Hz decision frames; "
                "training of the 5 Hz student must use ONLY those frames. "
                "macro_label_valid==1 means the 5 Hz decision was complete; "
                "an update frame that cannot produce a legal label rejects "
                "the whole episode. macro_direction_flu_* is the unit "
                "direction decoded at the 5 Hz decision instant (z=0); "
                "macro_param_valid==1 for NORMAL/TURN, 0 for PASS_THROUGH."),
            "debug_artifacts": {
                "privileged_only": True,
            },
            "status": "inprogress",
            "created_at_ns": _wall_time_ns(),
        }
        self._rows_written = 0
        self._closed = False
        # Publish an initial snapshot so a live `_inprogress` episode can be
        # inspected (scene/map + schema info).
        self._write_metadata_snapshot()

    # ── Rows ─────────────────────────────────────────────────────────
    def _write_metadata_snapshot(self):
        with open(self._metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def write_row(self, fields):
        """Persist one data row.  Missing fields get schema defaults."""
        row = dict(_DEFAULT_ROW)
        row.update(fields)
        self._data_writer.writerow(row)
        self._rows_written += 1
        if self._flush_interval_rows > 0 and \
                self._rows_written % self._flush_interval_rows == 0:
            self._data_csv.flush()

    def write_sync(self, frame_id, latency_ms, match_method, is_dropped,
                   exact_matches, unmatched_frames):
        self._sync_writer.writerow([
            frame_id, latency_ms, match_method, int(is_dropped),
            exact_matches, unmatched_frames])

    # ── Depth ────────────────────────────────────────────────────────
    def write_depth(self, frame_id, depth_m, raw_finite_ratio):
        """Encode and store the canonicalised single-channel depth frame
        (16-bit PNG).  raw_finite_ratio is a pure diagnostic (the fraction
        of finite/valid pixels in the RAW frame), never a student input."""
        valid = np.isfinite(depth_m) & (depth_m > 0)
        u16 = np.zeros(depth_m.shape, dtype=np.uint16)
        finite = depth_m.copy()
        finite[~valid] = 0.0
        scaled = np.clip(finite * 100.0, 0, 65535)
        u16[...] = np.round(scaled).astype(np.uint16)
        from PIL import Image
        png_path = os.path.join(self._depth_dir, "%06d.png" % frame_id)
        Image.fromarray(u16, mode="I;16").save(
            png_path, compress_level=self._depth_png_compress_level)
        self._depth_rows[frame_id] = {
            "depth_file": "depth/%06d.png" % frame_id,
            "raw_depth_finite_ratio": float(raw_finite_ratio),
        }

    def depth_file_for(self, frame_id):
        row = self._depth_rows.get(frame_id)
        return row["depth_file"] if row else ""

    # ── Lifecycle ────────────────────────────────────────────────────
    def add_metadata(self, extra):
        self._metadata.update(extra)

    def finish(self, success, reason, extra_metadata=None):
        """Close writers, write metadata, atomically commit or reject."""
        if self._closed:
            return
        self._closed = True
        self._data_csv.flush()
        self._data_csv.close()
        self._sync_csv.flush()
        self._sync_csv.close()

        self._metadata["status"] = "committed" if success else "rejected"
        self._metadata["exit_reason"] = reason
        self._metadata["rows_written"] = self._rows_written
        self._metadata["finished_at_ns"] = _wall_time_ns()
        if extra_metadata:
            self._metadata.update(extra_metadata)

        self._write_metadata_snapshot()

        if success:
            final_dir = os.path.join(self._output_root, self.episode_id)
            if os.path.isdir(final_dir):
                shutil.rmtree(final_dir, ignore_errors=True)
            os.rename(self._inprogress_dir, final_dir)
            return final_dir
        else:
            failed_root = os.path.join(self._output_root, "_failed")
            if not os.path.isdir(failed_root):
                os.makedirs(failed_root)
            failed_dir = os.path.join(failed_root, self.episode_id)
            if os.path.isdir(failed_dir):
                shutil.rmtree(failed_dir, ignore_errors=True)
            os.rename(self._inprogress_dir, failed_dir)
            with open(os.path.join(failed_dir, "failure_reason.json"), "w") as f:
                json.dump({"reason": reason}, f)
            return failed_dir

    def close(self):
        if not self._closed:
            self.finish(False, "aborted")
