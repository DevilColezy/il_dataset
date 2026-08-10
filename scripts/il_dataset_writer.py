#!/usr/bin/env python3
"""
il_dataset_writer.py  —  Dataset recording for the two-level navigation
expert (section XIV).

Owns the per-episode directory, the CSV schema (macro + local labels +
student inputs + privileged diagnostics), depth PNGs, local-plan logs and
the atomic commit / reject lifecycle.
"""

from __future__ import print_function, division

import csv
import json
import os
import shutil
import time

import numpy as np

# ── Schema (v24) ──────────────────────────────────────────────────────
# Order matters: this is the CSV header.  All fields are written on every
# row; missing keys are filled with defaults by write_row().
# NO depth-valid-mask fields: the student input is a single depth channel
# (section XII).  raw_depth_finite_ratio is a PURE diagnostic.
DATA_SCHEMA_V24_FIELDS = [
    # time / matching
    "timestamp_ns", "receive_timestamp_ns", "episode_id", "frame_id",
    "episode_frame_index", "control_dt_s", "trajectory_time_s",
    "latency_ms", "match_method", "frame_valid", "frame_invalid_reason",
    # state
    "x", "y", "z", "qx", "qy", "qz", "qw",
    "state_vx_world", "state_vy_world", "state_vz_world",
    "state_vx_flu", "state_vy_flu", "state_vz_flu",
    "yaw", "yaw_rate",
    "gravity_dx_flu", "gravity_dy_flu", "gravity_dz_flu",
    # student inputs (single-channel depth, no mask)
    "depth_file", "raw_depth_finite_ratio",
    "goal_direction_flu_x", "goal_direction_flu_y", "goal_direction_flu_z",
    "goal_distance_m", "goal_distance_norm",
    "velocity_flu_x", "velocity_flu_y", "velocity_flu_z",
    # macro labels (5 Hz; held between ticks, macro_is_new_tick flags)
    "macro_is_new_tick", "macro_mode", "macro_committed_side",
    "macro_observe_side",
    # P2 side-selection consistency diagnostics (never student input)
    "macro_chosen_side", "side_rejection_reason",
    "side_candidate_full_left", "side_candidate_full_right",
    "side_candidate_connected_left", "side_candidate_connected_right",
    "macro_confidence", "macro_decision_reason",
    "macro_decision_observable", "macro_decision_confidence",
    "macro_decision_margin",
    "causal_intervention_evidence",
    "macro_guide_world_x", "macro_guide_world_y", "macro_guide_world_z",
    "macro_guide_flu_x", "macro_guide_flu_y", "macro_guide_flu_z",
    "macro_guide_direction_flu_x", "macro_guide_direction_flu_y",
    "macro_guide_direction_flu_z", "macro_guide_distance_m",
    "desired_yaw_world", "desired_yaw_delta",
    "desired_yaw_sin", "desired_yaw_cos",
    # local labels (30 Hz) — EXECUTED plan semantics (section XVIII)
    "local_terminal_valid",
    "local_terminal_world_x", "local_terminal_world_y", "local_terminal_world_z",
    "local_terminal_flu_x", "local_terminal_flu_y", "local_terminal_flu_z",
    "execution_mode",
    "velocity_command_flu_x", "velocity_command_flu_y", "velocity_command_flu_z",
    "yaw_rate_command",
    "fresh_plan", "cached_plan_used",
    "active_plan_is_fresh", "active_plan_is_cached",
    "planning_status", "minimum_clearance", "trajectory_duration_s",
    "fresh_planning_status",
    # privileged diagnostics (never part of student inputs)
    "local_recoverable", "blocker_signature", "blocker_ray_depth",
    "blocker_cell_count", "blocker_track_id",
    "left_edge_visible", "right_edge_visible",
    "left_corridor_known", "right_corridor_known",
    "privileged_best_side",
    "privileged_local_recoverable", "privileged_rejoin_reached",
    "privileged_future_intervention_required",
    "privileged_local_path_length", "privileged_local_duration",
    "privileged_detour_ratio", "privileged_min_clearance",
    "privileged_goal_progress",
    # observed/privileged recoverability audit fields (section XXV)
    "observed_rejoin_distance", "observed_path_length",
    "observed_detour_ratio", "observed_terminal_alignment",
    "privileged_rejoin_distance", "privileged_terminal_alignment",
    "direct_no_progress_time", "observe_no_information_time",
    # macro-interval feedback diagnostics (sections XXVIII/XXIX)
    "macro_feedback_is_new",
    "macro_interval_frame_count", "macro_interval_planning_failures",
    "macro_interval_cached_frames", "macro_interval_brake_frames",
    "macro_interval_emergency_frames",
    "macro_interval_local_unrecoverable_frames",
    "global_cost_to_go", "global_clearance", "global_candidate_costs",
    # goal / plan bookkeeping — executed-plan semantics (XVII)
    "goal_world_x", "goal_world_y", "goal_world_z", "distance_to_final_goal",
    "plan_id", "plan_age_s", "plan_is_fresh", "plan_status", "plan_compute_ms",
    # episode
    "scene_id", "task_id", "episode_valid",
    # P5 failure taxonomy + P1 stale-plan diagnostics (never student input)
    "failure_taxonomy", "critical_plan_failure_status",
    "stale_plan_invalidations",
    # active-observation diagnostics (pure diagnostics, never student
    # input; held from the last 5 Hz macro tick)
    "observe_scan_side", "left_scan_exhausted", "right_scan_exhausted",
    "observe_rotation_exhausted", "observe_stagnant_rotate_time",
    "observe_raw_candidate_count", "observe_lattice_candidate_count",
    "observe_frontier_candidate_count", "observe_endpoint_known_free_count",
    "observe_local_full_count",
    "observe_forward_full_count", "observe_retreat_full_count",
    "observe_retreat_candidate_count", "observe_recovery_active",
    "observe_reject_unknown",
    "observe_reject_endpoint_clearance", "observe_reject_min_distance",
    "observe_reject_max_distance", "observe_reject_partial",
    "observe_reject_no_path",
    "observe_left_valid_count", "observe_right_valid_count",
    "observe_center_valid_count",
    "observe_selected_source", "observe_selected_side",
    "observe_selected_distance", "observe_selected_path_length",
    "observe_selected_info_gain", "observe_selected_clearance",
]

_DEFAULT_ROW = {field: "" for field in DATA_SCHEMA_V24_FIELDS}


class DatasetWriter(object):
    """Per-episode writer with atomic commit / reject."""

    def __init__(self, cfg, episode_id, output_root, scene_id, task_id,
                 start_world, goal_world, initial_yaw, depth_cfg):
        self.cfg = cfg
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.task_id = task_id
        self.start_world = np.asarray(start_world, dtype=np.float64)
        self.goal_world = np.asarray(goal_world, dtype=np.float64)
        self.initial_yaw = float(initial_yaw)
        self.depth_cfg = depth_cfg
        # The manager passes the dataset_logging SUB-config (section XXII).
        self._depth_in_memory = bool(cfg.get("depth_in_memory", True))
        self._flush_interval_rows = int(cfg.get("flush_interval_rows", 64))
        self._depth_png_compress_level = int(
            cfg.get("depth_png_compress_level", 4))
        # Debug trace (section "debug"): when enabled, every 5 Hz macro
        # tick writes the expert's internal state to trace.jsonl for
        # post-hoc review with scripts/debug_viewer.py.
        self._debug_trace = bool(cfg.get("debug_trace", False))

        self._inprogress_dir = os.path.join(output_root, "%s.inprogress" % episode_id)
        if not os.path.isdir(self._inprogress_dir):
            os.makedirs(self._inprogress_dir)

        self._data_file = os.path.join(self._inprogress_dir, "data.csv")
        self._sync_file = os.path.join(self._inprogress_dir, "sync.csv")
        self._local_plans_file = os.path.join(self._inprogress_dir, "local_plans.csv")
        self._local_plan_points_file = os.path.join(
            self._inprogress_dir, "local_plan_points.csv")
        self._depth_dir = os.path.join(self._inprogress_dir, "depth")
        if self._depth_in_memory:
            os.makedirs(self._depth_dir)
        self._metadata_file = os.path.join(self._inprogress_dir, "metadata.json")

        self._data_csv = open(self._data_file, "w", newline="")
        self._data_writer = csv.DictWriter(
            self._data_csv, fieldnames=DATA_SCHEMA_V24_FIELDS)
        self._data_writer.writeheader()

        self._sync_csv = open(self._sync_file, "w", newline="")
        self._sync_writer = csv.writer(self._sync_csv)
        self._sync_writer.writerow([
            "frame_id", "latency_ms", "match_method", "is_dropped",
            "exact_matches", "unmatched_frames"])

        self._plans_csv = open(self._local_plans_file, "w", newline="")
        self._plans_writer = csv.writer(self._plans_csv)
        self._plans_writer.writerow([
            "plan_id", "source_frame_id", "status", "success",
            "planning_time_ms", "min_clearance", "duration_s",
            "guide_waypoint_x", "guide_waypoint_y", "guide_waypoint_z",
            "terminal_x", "terminal_y", "terminal_z",
            "search_status", "trajectory_point_count"])

        self._plan_points_csv = open(self._local_plan_points_file, "w", newline="")
        self._plan_points_writer = csv.writer(self._plan_points_csv)
        self._plan_points_writer.writerow([
            "plan_id", "point_index", "t", "x", "y", "z",
            "vx", "vy", "vz", "ax", "ay", "az",
            "yaw", "yaw_rate", "clearance"])

        self._trace_csv = None
        if self._debug_trace:
            self._trace_file = os.path.join(self._inprogress_dir, "trace.jsonl")
            self._trace_csv = open(self._trace_file, "w", newline="")

        self._depth_rows = {}        # frame_id -> row dict (depth + meta)
        self._metadata = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "task_id": task_id,
            "start_world": start_world,
            "goal_world": goal_world,
            "initial_yaw": initial_yaw,
            "schema_version": int(cfg.get("schema_version", 24)),
            "status": "inprogress",
            "created_at_ns": time.time_ns(),
        }
        self._rows_written = 0
        self._closed = False

    # ── Rows ─────────────────────────────────────────────────────────
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

    # ── Local plan logs ──────────────────────────────────────────────
    def write_trace(self, record):
        """Append one macro-tick trace line (debug mode only)."""
        if self._trace_csv is None:
            return
        self._trace_csv.write(json.dumps(record, default=str) + "\n")
        self._trace_csv.flush()

    def write_local_plan(self, plan_id, source_frame_id, status, success,
                         planning_time_ms, min_clearance, duration_s,
                         guide_waypoint, terminal, search_status,
                         trajectory):
        self._plans_writer.writerow([
            plan_id, source_frame_id, status, int(success),
            planning_time_ms, min_clearance, duration_s,
            guide_waypoint[0], guide_waypoint[1], guide_waypoint[2],
            terminal[0], terminal[1], terminal[2],
            search_status, len(trajectory)])
        for i, point in enumerate(trajectory):
            self._plan_points_writer.writerow([
                plan_id, i, point.t,
                point.position[0], point.position[1], point.position[2],
                point.velocity[0], point.velocity[1], point.velocity[2],
                point.acceleration[0], point.acceleration[1], point.acceleration[2],
                point.yaw, point.yaw_rate,
                point.clearance if np.isfinite(point.clearance) else -1.0])

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
        self._plans_csv.flush()
        self._plans_csv.close()
        self._plan_points_csv.flush()
        self._plan_points_csv.close()
        if self._trace_csv is not None:
            self._trace_csv.flush()
            self._trace_csv.close()
            self._trace_csv = None

        self._metadata["status"] = "committed" if success else "rejected"
        self._metadata["exit_reason"] = reason
        self._metadata["rows_written"] = self._rows_written
        self._metadata["finished_at_ns"] = time.time_ns()
        if extra_metadata:
            self._metadata.update(extra_metadata)

        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        if success:
            final_dir = os.path.join(
                os.path.dirname(self._inprogress_dir), self.episode_id)
            if os.path.isdir(final_dir):
                shutil.rmtree(final_dir, ignore_errors=True)
            os.rename(self._inprogress_dir, final_dir)
            return final_dir
        else:
            failed_root = os.path.join(
                os.path.dirname(self._inprogress_dir), "_failed")
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
