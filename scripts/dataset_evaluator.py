#!/usr/bin/env python3
"""Offline quality evaluator for the strict Schema-v17 IL dataset.

The evaluator never repairs or filters samples.  It audits complete committed
episodes, rejected/in-progress episodes, training-label coverage, scene split
coverage, and expert-control statistics, then writes JSON and Markdown reports.
"""

from __future__ import print_function, division

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA_VERSION = 17
STRICT_POLICY = "reject_trajectory_if_any_required_label_is_invalid_v1"
DENSITY_TIERS = ("sparse", "medium", "dense")
H_CLASSES = 13
V_CLASSES = 7
H_SOFT = tuple("trend_horizontal_soft_{:02d}".format(i) for i in range(H_CLASSES))
V_SOFT = tuple("guide_elevation_soft_{}".format(i) for i in range(V_CLASSES))
VALIDITY_FIELDS = (
    "frame_valid", "expert_label_valid",
    "global_direction_valid", "macro_label_valid",
)
LEGACY_LOSS_MASKS = (
    "trend_horizontal_loss_valid", "trend_vertical_loss_valid",
    "trend_value_loss_valid", "control_loss_valid",
)
REQUIRED_COLUMNS = (
    "timestamp_ns", "episode_id", "frame_id", "episode_frame_index",
    "sequence_reset", "control_dt_s", "depth_file", "scene_id", "task_id",
    "global_dir_x_flu", "global_dir_y_flu", "global_dir_z_flu",
    "global_distance_norm", "gravity_direction_x_flu",
    "gravity_direction_y_flu", "gravity_direction_z_flu",
    "state_vx_flu", "state_vy_flu", "state_vz_flu",
    "state_angular_velocity_x_flu", "state_angular_velocity_y_flu",
    "state_angular_velocity_z_flu", "macro_update", "macro_mode",
    "macro_committed_side", "macro_move_dir_x_flu",
    "macro_move_dir_y_flu", "macro_move_dir_z_flu",
    "macro_move_distance_norm", "macro_yaw_dir_x_flu",
    "macro_yaw_dir_y_flu", "local_feasible", "local_progress_rate",
    "trend_horizontal_class_13",
    "guide_elevation_bin", "guide_distance_norm", "guide_mode",
    "trend_mode", "planner_mode", "expert_vx_flu", "expert_vy_flu",
    "expert_vz_flu", "expert_yaw_rate", "collision",
) + VALIDITY_FIELDS + H_SOFT + V_SOFT


class RunningStats(object):
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def add(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite value")
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def result(self):
        if not self.count:
            return {"count": 0, "mean": None, "std": None,
                    "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class IssueStore(list):
    """Bounded detail storage with an unbounded total issue counter."""

    def __init__(self):
        super(IssueStore, self).__init__()
        self.total = 0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a committed strict Schema-v16 IL dataset")
    parser.add_argument("--dataset-root", required=True,
                        help="Root containing committed trajectories")
    parser.add_argument("--config-file", default="",
                        help="Collection YAML used to derive expected counts/limits")
    parser.add_argument("--output-dir", default="",
                        help="Optional report directory; omit for terminal-only output")
    parser.add_argument("--check-images", choices=("none", "existence", "decode"),
                        default="existence")
    parser.add_argument("--min-class-frames", type=int, default=30,
                        help="Minimum samples per class for coverage readiness")
    parser.add_argument("--max-errors", type=int, default=200,
                        help="Maximum detailed errors retained in the report")
    parser.add_argument("--plots", choices=("true", "false"), default="false",
                        help="Write plots when --output-dir is also provided")
    parser.add_argument("--fail-on-error", choices=("true", "false"),
                        default="false",
                        help="Return exit code 2 on integrity failure (useful for CI)")
    return parser.parse_args()


def _load_config(path):
    defaults = {
        "expected_trajectories": None,
        "record_hz": None,
        "max_velocity": 2.5,
        "max_yaw_rate": 2.0,
        "goal_hold_ticks": 3,
        "expected_profile_trajectories": {},
    }
    if not path:
        return defaults
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}
        global_cfg = cfg.get("global", {})
        scene_cfg = global_cfg.get("scene_generation", {})
        task_cfg = scene_cfg.get("common_task_generation", {})
        enabled_profiles = [
            p for p in scene_cfg.get("profiles", [])
            if p.get("enabled", True)
        ]
        scene_count = sum(int(p.get("scene_count", 0)) for p in enabled_profiles)
        tasks_per_scene = int(task_cfg.get("tasks_per_scene", 0))
        if scene_count > 0 and tasks_per_scene > 0:
            defaults["expected_trajectories"] = scene_count * tasks_per_scene
            defaults["expected_profile_trajectories"] = {
                str(profile.get("name", "unnamed")):
                    int(profile.get("scene_count", 0)) * tasks_per_scene
                for profile in enabled_profiles
            }
        control_cfg = global_cfg.get("control", {})
        planner_cfg = global_cfg.get("planning", {}).get("local_planner", {})
        defaults["record_hz"] = float(control_cfg.get("record_hz", 0.0)) or None
        defaults["max_velocity"] = float(planner_cfg.get("max_velocity", 2.5))
        defaults["max_yaw_rate"] = float(planner_cfg.get("max_yaw_rate", 2.0))
        defaults["goal_hold_ticks"] = int(planner_cfg.get("goal_hold_ticks", 3))
        return defaults
    except Exception as exc:
        raise ValueError("cannot load config '{}': {}".format(path, exc))


def _discover(root):
    committed = []
    inprogress = []
    failure_manifests = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        path = Path(dirpath)
        if path.name == "_eval":
            dirnames[:] = []
            continue
        if path.name.endswith(".inprogress"):
            inprogress.append(path)
            dirnames[:] = []
            continue
        if "failure_reason.json" in filenames:
            failure_manifests.append(path / "failure_reason.json")
        if {"data.csv", "metadata.json"}.issubset(set(filenames)):
            if "_failed" not in path.parts:
                committed.append(path)
            dirnames[:] = []
    return sorted(committed), sorted(inprogress), sorted(failure_manifests)


def _counter_dict(counter, keys=None):
    if keys is None:
        keys = sorted(counter, key=lambda value: str(value))
    return {str(key): int(counter.get(key, 0)) for key in keys}


def _distribution(counter, keys):
    total = float(sum(counter.get(key, 0) for key in keys))
    return {
        str(key): {
            "count": int(counter.get(key, 0)),
            "fraction": (counter.get(key, 0) / total if total else 0.0),
        }
        for key in keys
    }


def _add_issue(store, message, limit):
    store.total += 1
    if len(store) < limit:
        store.append(message)


def _read_failure_reasons(paths):
    counts = Counter()
    unreadable = 0
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
            reasons = payload.get("reasons", [])
            if not isinstance(reasons, list):
                reasons = [str(reasons)]
            for reason in reasons:
                reason = str(reason)
                category = reason.split(":", 1)[0].strip()
                counts[category or "unspecified"] += 1
        except Exception:
            unreadable += 1
    return counts, unreadable


def _evaluate(root, config, args):
    episode_dirs, inprogress_dirs, failure_paths = _discover(root)
    errors = IssueStore()
    warnings = []
    metadata_warnings = []
    h_counts = Counter()
    v_counts = Counter()
    tier_episodes = Counter()
    tier_frames = Counter()
    tier_scenes = defaultdict(set)
    profile_episodes = Counter()
    task_types = Counter()
    task_type_targets = Counter()
    blocker_bands = Counter()
    blocker_band_targets = Counter()
    height_bands = Counter()
    height_band_targets = Counter()
    coverage_target_mismatches = Counter()
    guide_modes = Counter()
    trend_modes = Counter()
    exit_reasons = Counter()
    scene_ids = set()
    episode_ids = set()
    control_stats = {name: RunningStats() for name in
                     ("vx", "vy", "vz", "yaw_rate", "speed", "delta_speed",
                      "delta_yaw_rate", "guide_distance_norm")}
    total_frames = 0
    valid_episodes = 0
    missing_images = 0
    corrupt_images = 0
    extra_images = 0
    collision_frames = 0
    goal_hold_frames = 0
    recovery_frames = 0
    stationary_frames = 0
    turning_frames = 0
    lateral_frames = 0
    saturated_speed_frames = 0
    saturated_yaw_frames = 0
    soft_label_mismatches = 0
    duplicate_episode_ids = 0
    episode_lengths = []
    episode_durations = []

    for episode_dir in episode_dirs:
        csv_path = episode_dir / "data.csv"
        meta_path = episode_dir / "metadata.json"
        depth_dir = episode_dir / "depth"
        episode_error_count_before = errors.total
        try:
            with open(meta_path, "r", encoding="utf-8") as stream:
                meta = json.load(stream)
        except Exception as exc:
            _add_issue(errors, "{}: unreadable metadata: {}".format(meta_path, exc),
                       args.max_errors)
            continue

        contracts = (
            (meta.get("schema_version") == SCHEMA_VERSION,
             "schema_version must be 17 (got {!r})".format(meta.get("schema_version"))),
            (meta.get("status") == "committed",
             "status must be committed (got {!r})".format(meta.get("status"))),
            (meta.get("terminal_label_semantics") == "goal_hold_v1",
             "terminal_label_semantics must be goal_hold_v1"),
            (meta.get("episode_validity_policy") == STRICT_POLICY,
             "invalid episode_validity_policy"),
            (meta.get("collection_mode") == "deterministic_lockstep",
             "collection_mode must be deterministic_lockstep"),
            (meta.get("dynamics_backend") == "flightmare",
             "dynamics_backend must be flightmare"),
            (bool(meta.get("reached_goal", False)), "reached_goal must be true"),
            (str(meta.get("exit_reason", "")) == "goal_reached",
             "exit_reason must be goal_reached"),
        )
        for passed, message in contracts:
            if not passed:
                _add_issue(errors, "{}: {}".format(meta_path, message), args.max_errors)

        tier = str(meta.get("scene_density_tier", "")).strip().lower()
        profile = str(meta.get("scene_profile_name", "")).strip()
        if tier not in DENSITY_TIERS:
            _add_issue(errors, "{}: invalid scene_density_tier={!r}".format(meta_path, tier),
                       args.max_errors)
        if not profile:
            _add_issue(errors, "{}: empty scene_profile_name".format(meta_path),
                       args.max_errors)
        if int(meta.get("invalid_frame_count", 0)) != 0:
            _add_issue(errors, "{}: invalid_frame_count is nonzero".format(meta_path),
                       args.max_errors)
        if int(meta.get("invalid_expert_label_count", 0)) != 0:
            _add_issue(errors, "{}: invalid_expert_label_count is nonzero".format(meta_path),
                       args.max_errors)
        if int(meta.get("invalid_trend_label_count", 0)) != 0:
            _add_issue(errors, "{}: invalid_trend_label_count is nonzero".format(meta_path),
                       args.max_errors)
        if int(meta.get("goal_hold_frame_count", 0)) < config["goal_hold_ticks"]:
            _add_issue(errors, "{}: goal_hold_frame_count < {}".format(
                meta_path, config["goal_hold_ticks"]), args.max_errors)

        exit_reasons[str(meta.get("exit_reason", "missing"))] += 1
        actual_task_type = str(meta.get(
            "task_coverage_actual_task_type", "unknown")) or "unknown"
        target_task_type = str(meta.get(
            "task_coverage_target_task_type", "unknown")) or "unknown"
        actual_blocker_band = str(meta.get(
            "task_coverage_actual_blocker_distance_band", "unknown")) or "unknown"
        target_blocker_band = str(meta.get(
            "task_coverage_target_blocker_distance_band", "unknown")) or "unknown"
        actual_height_band = str(meta.get(
            "task_coverage_actual_height_band", "unknown")) or "unknown"
        target_height_band = str(meta.get(
            "task_coverage_target_height_band", "unknown")) or "unknown"
        task_types[actual_task_type] += 1
        task_type_targets[target_task_type] += 1
        blocker_bands[actual_blocker_band] += 1
        blocker_band_targets[target_blocker_band] += 1
        height_bands[actual_height_band] += 1
        height_band_targets[target_height_band] += 1
        if target_task_type != actual_task_type:
            coverage_target_mismatches["task_type"] += 1
        if target_blocker_band != actual_blocker_band:
            coverage_target_mismatches["blocker_distance_band"] += 1
        if target_height_band != actual_height_band:
            coverage_target_mismatches["height_band"] += 1

        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                headers = tuple(reader.fieldnames or ())
                missing = [field for field in REQUIRED_COLUMNS if field not in headers]
                legacy = [field for field in LEGACY_LOSS_MASKS if field in headers]
                if missing:
                    raise ValueError("missing required columns: {}".format(missing))
                if legacy:
                    raise ValueError("legacy per-head loss-mask columns present: {}".format(legacy))

                expected_frame_index = 0
                previous_frame_id = None
                previous_timestamp = None
                previous_command = None
                referenced_images = set()
                csv_episode_id = None
                csv_scene_id = None
                row_count = 0
                first_time = None
                last_time = None

                for row_number, row in enumerate(reader, 1):
                    row_count += 1
                    location = "{}:row {}".format(csv_path, row_number)
                    invalid = [field for field in VALIDITY_FIELDS
                               if str(row.get(field, "0")).strip() != "1"]
                    if invalid:
                        _add_issue(errors, "{}: invalid supervision {}".format(
                            location, invalid), args.max_errors)

                    try:
                        frame_index = int(row["episode_frame_index"])
                        frame_id = int(row["frame_id"])
                        timestamp = int(row["timestamp_ns"])
                        sequence_reset = int(row["sequence_reset"])
                        dt_s = float(row["control_dt_s"])
                        if frame_index != expected_frame_index:
                            raise ValueError("episode_frame_index {} != {}".format(
                                frame_index, expected_frame_index))
                        if sequence_reset != (1 if row_number == 1 else 0):
                            raise ValueError("invalid sequence_reset={}".format(sequence_reset))
                        if previous_frame_id is not None and frame_id != previous_frame_id + 1:
                            raise ValueError("non-consecutive frame_id {} -> {}".format(
                                previous_frame_id, frame_id))
                        if previous_timestamp is not None and timestamp <= previous_timestamp:
                            raise ValueError("timestamp is not strictly increasing")
                        if not math.isfinite(dt_s) or dt_s <= 0.0:
                            raise ValueError("invalid control_dt_s")
                        if config["record_hz"] is not None:
                            expected_dt = 1.0 / config["record_hz"]
                            if abs(dt_s - expected_dt) > max(1e-6, 0.05 * expected_dt):
                                raise ValueError("control_dt_s does not match config")
                        expected_frame_index += 1
                        previous_frame_id = frame_id
                        previous_timestamp = timestamp
                        first_time = timestamp if first_time is None else first_time
                        last_time = timestamp
                    except Exception as exc:
                        _add_issue(errors, "{}: {}".format(location, exc), args.max_errors)

                    current_episode_id = row["episode_id"].strip()
                    current_scene_id = row["scene_id"].strip()
                    if csv_episode_id is None:
                        csv_episode_id = current_episode_id
                        csv_scene_id = current_scene_id
                    if current_episode_id != csv_episode_id or current_scene_id != csv_scene_id:
                        _add_issue(errors, "{}: episode/scene identifier changed".format(location),
                                   args.max_errors)

                    try:
                        h_class = int(row["trend_horizontal_class_13"])
                        v_class = int(row["guide_elevation_bin"])
                        if not 0 <= h_class < H_CLASSES:
                            raise ValueError("horizontal class out of range: {}".format(h_class))
                        if not 0 <= v_class < V_CLASSES:
                            raise ValueError("vertical class out of range: {}".format(v_class))
                        h_values = [float(row[name]) for name in H_SOFT]
                        v_values = [float(row[name]) for name in V_SOFT]
                        if any(not math.isfinite(value) or value < -1e-6 for value in
                               h_values + v_values):
                            raise ValueError("invalid soft-label value")
                        if abs(sum(h_values) - 1.0) > 0.02 or abs(sum(v_values) - 1.0) > 0.02:
                            raise ValueError("soft-label probabilities do not sum to one")
                        if h_values.index(max(h_values)) != h_class:
                            soft_label_mismatches += 1
                            _add_issue(errors,
                                       "{}: horizontal soft-label argmax mismatch".format(
                                           location), args.max_errors)
                        if v_values.index(max(v_values)) != v_class:
                            soft_label_mismatches += 1
                            _add_issue(errors,
                                       "{}: vertical soft-label argmax mismatch".format(
                                           location), args.max_errors)
                        h_counts[h_class] += 1
                        v_counts[v_class] += 1
                    except Exception as exc:
                        _add_issue(errors, "{}: {}".format(location, exc), args.max_errors)

                    try:
                        values = {
                            "vx": float(row["expert_vx_flu"]),
                            "vy": float(row["expert_vy_flu"]),
                            "vz": float(row["expert_vz_flu"]),
                            "yaw_rate": float(row["expert_yaw_rate"]),
                        }
                        if any(not math.isfinite(value) for value in values.values()):
                            raise ValueError("non-finite expert command")
                        speed = math.sqrt(values["vx"] ** 2 + values["vy"] ** 2 +
                                          values["vz"] ** 2)
                        for name, value in values.items():
                            control_stats[name].add(value)
                        control_stats["speed"].add(speed)
                        if speed < 0.05:
                            stationary_frames += 1
                        if abs(values["yaw_rate"]) > 0.10:
                            turning_frames += 1
                        if abs(values["vy"]) > 0.10:
                            lateral_frames += 1
                        if speed >= 0.98 * config["max_velocity"]:
                            saturated_speed_frames += 1
                        if abs(values["yaw_rate"]) >= 0.98 * config["max_yaw_rate"]:
                            saturated_yaw_frames += 1
                        if speed > config["max_velocity"] + 1e-4:
                            raise ValueError("expert speed exceeds configured maximum")
                        if abs(values["yaw_rate"]) > config["max_yaw_rate"] + 1e-4:
                            raise ValueError("expert yaw rate exceeds configured maximum")
                        if previous_command is not None:
                            delta_speed = math.sqrt(sum(
                                (values[name] - previous_command[name]) ** 2
                                for name in ("vx", "vy", "vz")))
                            control_stats["delta_speed"].add(delta_speed)
                            control_stats["delta_yaw_rate"].add(
                                abs(values["yaw_rate"] - previous_command["yaw_rate"]))
                        previous_command = values
                    except Exception as exc:
                        _add_issue(errors, "{}: {}".format(location, exc), args.max_errors)

                    try:
                        guide_value = float(row["guide_distance_norm"])
                        global_value = float(row["global_distance_norm"])
                        if not (-1e-6 <= guide_value <= 1.0 + 1e-6):
                            raise ValueError("guide_distance_norm outside [0,1]")
                        if not (-1e-6 <= global_value <= 1.0 + 1e-6):
                            raise ValueError("global_distance_norm outside [0,1]")
                        control_stats["guide_distance_norm"].add(guide_value)
                        global_norm = math.sqrt(sum(float(row[name]) ** 2 for name in (
                            "global_dir_x_flu", "global_dir_y_flu",
                            "global_dir_z_flu")))
                        if abs(global_norm - 1.0) > 0.02:
                            raise ValueError("global direction is not unit length")
                        gravity_norm = math.sqrt(sum(float(row[name]) ** 2 for name in (
                            "gravity_direction_x_flu", "gravity_direction_y_flu",
                            "gravity_direction_z_flu")))
                        if abs(gravity_norm - 1.0) > 0.02:
                            raise ValueError("gravity direction is not unit length")
                    except Exception as exc:
                        _add_issue(errors, "{}: {}".format(location, exc), args.max_errors)

                    mode = row["guide_mode"].strip()
                    trend_mode = row["trend_mode"].strip()
                    guide_modes[mode] += 1
                    trend_modes[trend_mode] += 1
                    if trend_mode == "RECOVERY":
                        recovery_frames += 1
                        try:
                            if int(row["trend_horizontal_class_13"]) not in (0, 12):
                                raise ValueError("RECOVERY must use class 0 or 12")
                            if abs(float(row["guide_distance_norm"])) > 1e-6:
                                raise ValueError("RECOVERY guide value must be zero")
                        except Exception as exc:
                            _add_issue(errors, "{}: {}".format(location, exc), args.max_errors)
                    if trend_mode == "GOAL_HOLD":
                        goal_hold_frames += 1
                        try:
                            if (int(row["trend_horizontal_class_13"]) != 6 or
                                    int(row["guide_elevation_bin"]) != 3 or
                                    abs(float(row["guide_distance_norm"])) > 1e-6):
                                raise ValueError("invalid GOAL_HOLD guide target")
                            if max(abs(float(row[name])) for name in (
                                    "expert_vx_flu", "expert_vy_flu",
                                    "expert_vz_flu", "expert_yaw_rate")) > 1e-6:
                                raise ValueError("nonzero GOAL_HOLD expert command")
                        except Exception as exc:
                            _add_issue(errors, "{}: {}".format(location, exc), args.max_errors)
                    if int(float(row["collision"])) != 0:
                        collision_frames += 1
                        _add_issue(errors, "{}: collision frame".format(location), args.max_errors)

                    depth_name = row["depth_file"].strip()
                    if not depth_name or Path(depth_name).name != depth_name:
                        _add_issue(errors, "{}: invalid depth_file={!r}".format(
                            location, depth_name), args.max_errors)
                    elif depth_name in referenced_images:
                        _add_issue(errors, "{}: duplicate depth_file={!r}".format(
                            location, depth_name), args.max_errors)
                    else:
                        referenced_images.add(depth_name)
                        if args.check_images != "none":
                            image_path = depth_dir / depth_name
                            if not image_path.is_file():
                                missing_images += 1
                                _add_issue(errors, "{}: missing depth image {}".format(
                                    location, image_path), args.max_errors)
                            elif args.check_images == "decode":
                                try:
                                    from PIL import Image
                                    with Image.open(str(image_path)) as image:
                                        expected_size = (
                                            int(meta.get("depth_w", 0)),
                                            int(meta.get("depth_h", 0)))
                                        if image.size != expected_size:
                                            raise ValueError(
                                                "image size {} != metadata {}".format(
                                                    image.size, expected_size))
                                        if image.mode not in ("I", "I;16", "I;16B", "I;16L"):
                                            raise ValueError(
                                                "depth PNG is not 16-bit (mode={})".format(
                                                    image.mode))
                                        image.verify()
                                except Exception as exc:
                                    corrupt_images += 1
                                    _add_issue(errors, "{}: corrupt depth image: {}".format(
                                        image_path, exc), args.max_errors)

                if row_count == 0:
                    raise ValueError("empty data.csv")
                actual_pngs = set(path.name for path in depth_dir.glob("*.png")) \
                    if depth_dir.is_dir() else set()
                extra_images += len(actual_pngs - referenced_images)
                if int(meta.get("written_rows", -1)) != row_count:
                    raise ValueError("metadata written_rows={} but CSV has {}".format(
                        meta.get("written_rows"), row_count))
                if csv_episode_id in episode_ids:
                    duplicate_episode_ids += 1
                    raise ValueError("duplicate episode_id={!r}".format(csv_episode_id))
                episode_ids.add(csv_episode_id)
                scene_ids.add(csv_scene_id)
                total_frames += row_count
                episode_lengths.append(row_count)
                if first_time is not None and last_time is not None:
                    episode_durations.append(max(0.0, (last_time - first_time) * 1e-9))
                tier_episodes[tier] += 1
                tier_frames[tier] += row_count
                tier_scenes[tier].add(csv_scene_id)
                profile_episodes[profile] += 1
        except Exception as exc:
            _add_issue(errors, "{}: {}".format(csv_path, exc), args.max_errors)

        if errors.total == episode_error_count_before:
            valid_episodes += 1
        elif len(errors) >= args.max_errors:
            metadata_warnings.append(
                "Detailed error list reached --max-errors; counts may be under-reported.")

    failure_counts, unreadable_failures = _read_failure_reasons(failure_paths)
    if inprogress_dirs:
        warnings.append("{} interrupted .inprogress trajectory directories found".format(
            len(inprogress_dirs)))
    if failure_paths:
        warnings.append("{} rejected trajectories found; they are excluded from training".format(
            len(failure_paths)))
    if unreadable_failures:
        warnings.append("{} failure manifests could not be read".format(unreadable_failures))
    if extra_images:
        warnings.append("{} unreferenced PNG files found".format(extra_images))
    if soft_label_mismatches:
        warnings.append("{} soft-label argmax/class mismatches found".format(
            soft_label_mismatches))

    min_frames = max(1, args.min_class_frames)
    missing_h = [index for index in range(H_CLASSES) if h_counts[index] < min_frames]
    missing_v = [index for index in range(V_CLASSES) if v_counts[index] < min_frames]
    split_tier_failures = [tier for tier in DENSITY_TIERS if len(tier_scenes[tier]) < 2]
    if missing_h:
        warnings.append("horizontal classes below {} frames: {}".format(min_frames, missing_h))
    if missing_v:
        warnings.append("vertical classes below {} frames: {}".format(min_frames, missing_v))
    if split_tier_failures:
        warnings.append("density tiers with fewer than two scenes: {}".format(
            split_tier_failures))
    if coverage_target_mismatches:
        warnings.append("task coverage targets were not met: {}".format(
            _counter_dict(coverage_target_mismatches)))

    profile_shortfalls = {}
    for profile, expected_count in config["expected_profile_trajectories"].items():
        actual_count = int(profile_episodes[profile])
        if actual_count != expected_count:
            profile_shortfalls[profile] = {
                "expected": expected_count, "actual": actual_count,
                "missing": expected_count - actual_count,
            }

    expected = config["expected_trajectories"]
    structurally_valid = errors.total == 0 and valid_episodes == len(episode_dirs)
    collection_complete = None if expected is None else len(episode_dirs) == expected
    ready_2p5d = bool(structurally_valid and len(scene_ids) >= 2 and
                     not split_tier_failures and not missing_h)
    ready_3d = bool(ready_2p5d and not missing_v)
    total = float(total_frames) if total_frames else 1.0

    result = {
        "evaluator": "strict_schema_v17_dataset_evaluator_v1",
        "dataset_root": str(root),
        "verdict": {
            "structural_integrity": "PASS" if structurally_valid else "FAIL",
            "collection_complete": (
                "UNKNOWN" if collection_complete is None else
                ("PASS" if collection_complete else "FAIL")),
            "ready_for_2p5d_training": ready_2p5d,
            "ready_for_full_3d_training": ready_3d,
        },
        "collection": {
            "expected_committed_trajectories": expected,
            "discovered_committed_trajectories": len(episode_dirs),
            "valid_committed_trajectories": valid_episodes,
            "rejected_trajectories": len(failure_paths),
            "inprogress_trajectories": len(inprogress_dirs),
            "total_frames": total_frames,
            "total_duration_hours": sum(episode_durations) / 3600.0,
            "unique_scenes": len(scene_ids),
            "unique_episode_ids": len(episode_ids),
            "duplicate_episode_ids": duplicate_episode_ids,
            "episode_frames": _summary_list(episode_lengths),
            "episode_duration_s": _summary_list(episode_durations),
        },
        "integrity": {
            "error_count": errors.total,
            "error_count_retained": len(errors),
            "errors": list(errors),
            "missing_depth_images": missing_images,
            "corrupt_depth_images": corrupt_images,
            "unreferenced_depth_images": extra_images,
            "collision_frames": collision_frames,
            "soft_label_argmax_mismatches": soft_label_mismatches,
        },
        "scene_coverage": {
            "density_tiers": {
                tier: {"episodes": int(tier_episodes[tier]),
                       "frames": int(tier_frames[tier]),
                       "scenes": len(tier_scenes[tier])}
                for tier in DENSITY_TIERS
            },
            "profiles": _counter_dict(profile_episodes),
            "expected_profile_trajectories": config["expected_profile_trajectories"],
            "profile_shortfalls": profile_shortfalls,
            "task_types": {"target": _counter_dict(task_type_targets),
                           "actual": _counter_dict(task_types)},
            "blocker_distance_bands": {
                "target": _counter_dict(blocker_band_targets),
                "actual": _counter_dict(blocker_bands)},
            "height_bands": {"target": _counter_dict(height_band_targets),
                             "actual": _counter_dict(height_bands)},
            "coverage_target_mismatches": _counter_dict(
                coverage_target_mismatches),
            "leakage_free_split_ready": not split_tier_failures and len(scene_ids) >= 2,
        },
        "guide_labels": {
            "horizontal": _distribution(h_counts, range(H_CLASSES)),
            "vertical": _distribution(v_counts, range(V_CLASSES)),
            "horizontal_classes_below_minimum": missing_h,
            "vertical_classes_below_minimum": missing_v,
            "minimum_frames_per_class": min_frames,
            "guide_modes": _counter_dict(guide_modes),
            "trend_modes": _counter_dict(trend_modes),
            "guide_distance_norm": control_stats["guide_distance_norm"].result(),
            "recovery_frames": recovery_frames,
            "goal_hold_frames": goal_hold_frames,
        },
        "control_labels": {
            name: stats.result() for name, stats in control_stats.items()
            if name != "guide_distance_norm"
        },
        "control_behavior": {
            "stationary_fraction": stationary_frames / total,
            "turning_fraction": turning_frames / total,
            "lateral_motion_fraction": lateral_frames / total,
            "speed_saturation_fraction": saturated_speed_frames / total,
            "yaw_saturation_fraction": saturated_yaw_frames / total,
        },
        "failures": {
            "reason_categories": _counter_dict(failure_counts),
            "unreadable_manifests": unreadable_failures,
            "exit_reasons": _counter_dict(exit_reasons),
        },
        "warnings": warnings + metadata_warnings,
    }
    recommendations = []
    if not structurally_valid:
        recommendations.append(
            "Do not train: fix every integrity error and recollect each rejected episode.")
    if collection_complete is False:
        recommendations.append(
            "Continue collection until committed trajectories reach the configured target.")
    if split_tier_failures:
        recommendations.append(
            "Collect at least two independent scenes in every density tier before scene-level splitting.")
    if missing_h:
        recommendations.append(
            "Add tasks that deliberately elicit the under-covered horizontal/recovery guide classes.")
    if missing_v:
        recommendations.append(
            "The vertical branch is not fully supervised; keep this dataset for 2.5D validation or add true 3D height variation.")
    if coverage_target_mismatches:
        recommendations.append(
            "Inspect task generation quotas because requested task/band labels do not match accepted tasks.")
    result["recommendations"] = recommendations
    return result


def _summary_list(values):
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    middle = count // 2
    median = (ordered[middle] if count % 2 else
              0.5 * (ordered[middle - 1] + ordered[middle]))
    return {"count": count, "min": ordered[0], "median": median,
            "mean": sum(ordered) / count, "max": ordered[-1]}


def _write_markdown(result, path):
    verdict = result["verdict"]
    collection = result["collection"]
    coverage = result["scene_coverage"]
    guide = result["guide_labels"]
    behavior = result["control_behavior"]
    lines = [
        "# Schema-v16 imitation-learning dataset evaluation",
        "",
        "## Verdict",
        "",
        "- Structural integrity: **{}**".format(verdict["structural_integrity"]),
        "- Collection complete: **{}**".format(verdict["collection_complete"]),
        "- Ready for 2.5D training: **{}**".format(verdict["ready_for_2p5d_training"]),
        "- Ready for full 3D training: **{}**".format(verdict["ready_for_full_3d_training"]),
        "",
        "## Scale",
        "",
        "- Committed trajectories: {} / {}".format(
            collection["discovered_committed_trajectories"],
            collection["expected_committed_trajectories"]),
        "- Valid committed trajectories: {}".format(collection["valid_committed_trajectories"]),
        "- Frames: {}".format(collection["total_frames"]),
        "- Scenes: {}".format(collection["unique_scenes"]),
        "- Rejected / in-progress: {} / {}".format(
            collection["rejected_trajectories"], collection["inprogress_trajectories"]),
        "",
        "## Density coverage",
        "",
        "| Tier | Scenes | Episodes | Frames |",
        "|---|---:|---:|---:|",
    ]
    for tier in DENSITY_TIERS:
        item = coverage["density_tiers"][tier]
        lines.append("| {} | {} | {} | {} |".format(
            tier, item["scenes"], item["episodes"], item["frames"]))
    lines.extend([
        "",
        "## Guide coverage",
        "",
        "- Horizontal classes below minimum: `{}`".format(
            guide["horizontal_classes_below_minimum"]),
        "- Vertical classes below minimum: `{}`".format(
            guide["vertical_classes_below_minimum"]),
        "- Recovery frames: {}".format(guide["recovery_frames"]),
        "- Goal-hold frames: {}".format(guide["goal_hold_frames"]),
        "",
        "## Control behavior",
        "",
        "- Stationary: {:.2%}".format(behavior["stationary_fraction"]),
        "- Turning: {:.2%}".format(behavior["turning_fraction"]),
        "- Lateral motion: {:.2%}".format(behavior["lateral_motion_fraction"]),
        "- Speed saturation: {:.2%}".format(behavior["speed_saturation_fraction"]),
        "- Yaw saturation: {:.2%}".format(behavior["yaw_saturation_fraction"]),
        "",
        "## Warnings",
        "",
    ])
    lines.extend("- {}".format(item) for item in result["warnings"])
    if not result["warnings"]:
        lines.append("- None")
    lines.extend(["", "## Recommendations", ""])
    lines.extend("- {}".format(item) for item in result["recommendations"])
    if not result["recommendations"]:
        lines.append("- Dataset meets the configured readiness checks.")
    lines.extend(["", "## Integrity errors", ""])
    errors = result["integrity"]["errors"]
    lines.extend("- {}".format(item) for item in errors)
    if not errors:
        lines.append("- None")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def _write_plots(result, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return "plots skipped because matplotlib is unavailable: {}".format(exc)

    horizontal = result["guide_labels"]["horizontal"]
    vertical = result["guide_labels"]["vertical"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].bar(range(H_CLASSES), [horizontal[str(i)]["count"] for i in range(H_CLASSES)])
    axes[0].set_title("Horizontal guide labels (0/12 are recovery)")
    axes[0].set_xticks(range(H_CLASSES))
    axes[0].set_ylabel("frames")
    axes[1].bar(range(V_CLASSES), [vertical[str(i)]["count"] for i in range(V_CLASSES)])
    axes[1].set_title("Vertical guide labels")
    axes[1].set_xticks(range(V_CLASSES))
    axes[1].set_xlabel("class")
    axes[1].set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(str(output_dir / "guide_label_distribution.png"), dpi=150)
    plt.close(fig)

    tiers = result["scene_coverage"]["density_tiers"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(DENSITY_TIERS, [tiers[tier]["frames"] for tier in DENSITY_TIERS])
    ax.set_title("Frames by scene density tier")
    ax.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(str(output_dir / "density_tier_distribution.png"), dpi=150)
    plt.close(fig)
    return None


def _print_terminal_report(result):
    verdict = result["verdict"]
    collection = result["collection"]
    integrity = result["integrity"]
    coverage = result["scene_coverage"]
    guide = result["guide_labels"]
    control = result["control_labels"]
    behavior = result["control_behavior"]

    print("=" * 78)
    print("Schema-v16 模仿学习数据集评价")
    print("=" * 78)
    print("数据集路径: {}".format(result["dataset_root"]))
    print("结构完整性: {}".format(verdict["structural_integrity"]))
    print("采集完整性: {}".format(verdict["collection_complete"]))
    print("2.5D训练就绪: {}".format(
        "是" if verdict["ready_for_2p5d_training"] else "否"))
    print("完整3D训练就绪: {}".format(
        "是" if verdict["ready_for_full_3d_training"] else "否"))

    print("\n[数据规模]")
    print("  已提交轨迹: {} / {}".format(
        collection["discovered_committed_trajectories"],
        collection["expected_committed_trajectories"]
        if collection["expected_committed_trajectories"] is not None else "未知"))
    print("  完全有效轨迹: {}".format(collection["valid_committed_trajectories"]))
    print("  总帧数: {}".format(collection["total_frames"]))
    print("  总时长: {:.3f} 小时".format(collection["total_duration_hours"]))
    print("  独立场景: {}".format(collection["unique_scenes"]))
    print("  拒绝轨迹: {}".format(collection["rejected_trajectories"]))
    print("  未完成轨迹: {}".format(collection["inprogress_trajectories"]))
    frame_stats = collection["episode_frames"]
    print("  单轨迹帧数: min={} median={} mean={} max={}".format(
        _fmt(frame_stats["min"]), _fmt(frame_stats["median"]),
        _fmt(frame_stats["mean"]), _fmt(frame_stats["max"])))

    print("\n[密度覆盖]")
    print("  {:<10} {:>8} {:>10} {:>12}".format(
        "等级", "场景", "轨迹", "帧"))
    for tier in DENSITY_TIERS:
        item = coverage["density_tiers"][tier]
        print("  {:<10} {:>8} {:>10} {:>12}".format(
            tier, item["scenes"], item["episodes"], item["frames"]))
    print("  可进行无场景泄漏划分: {}".format(
        "是" if coverage["leakage_free_split_ready"] else "否"))
    if coverage["profile_shortfalls"]:
        print("  Profile数量缺口: {}".format(coverage["profile_shortfalls"]))
    if coverage["coverage_target_mismatches"]:
        print("  任务配额不匹配: {}".format(
            coverage["coverage_target_mismatches"]))

    print("\n[Guide水平标签：0/12为恢复，1-11为正常方向]")
    for index in range(H_CLASSES):
        item = guide["horizontal"][str(index)]
        print("  class {:>2}: {:>10} ({:>7.3%})".format(
            index, item["count"], item["fraction"]))
    print("  低于最小样本数的水平类别: {}".format(
        guide["horizontal_classes_below_minimum"]))

    print("\n[Guide垂直标签]")
    for index in range(V_CLASSES):
        item = guide["vertical"][str(index)]
        print("  class {:>2}: {:>10} ({:>7.3%})".format(
            index, item["count"], item["fraction"]))
    print("  低于最小样本数的垂直类别: {}".format(
        guide["vertical_classes_below_minimum"]))
    print("  Recovery帧: {}  Goal-hold帧: {}".format(
        guide["recovery_frames"], guide["goal_hold_frames"]))

    print("\n[Control标签统计]")
    print("  {:<18} {:>12} {:>12} {:>12} {:>12}".format(
        "字段", "mean", "std", "min", "max"))
    for name in ("vx", "vy", "vz", "yaw_rate", "speed",
                 "delta_speed", "delta_yaw_rate"):
        item = control[name]
        print("  {:<18} {:>12} {:>12} {:>12} {:>12}".format(
            name, _fmt(item["mean"]), _fmt(item["std"]),
            _fmt(item["min"]), _fmt(item["max"])))
    print("  静止帧比例: {:.3%}".format(behavior["stationary_fraction"]))
    print("  转向帧比例: {:.3%}".format(behavior["turning_fraction"]))
    print("  侧向运动比例: {:.3%}".format(behavior["lateral_motion_fraction"]))
    print("  速度饱和比例: {:.3%}".format(behavior["speed_saturation_fraction"]))
    print("  偏航饱和比例: {:.3%}".format(behavior["yaw_saturation_fraction"]))

    print("\n[完整性]")
    print("  错误总数: {}".format(integrity["error_count"]))
    print("  缺失/损坏/未引用深度图: {} / {} / {}".format(
        integrity["missing_depth_images"], integrity["corrupt_depth_images"],
        integrity["unreferenced_depth_images"]))
    print("  碰撞帧: {}".format(integrity["collision_frames"]))
    print("  Soft-label与类别不一致: {}".format(
        integrity["soft_label_argmax_mismatches"]))

    print("\n[警告]")
    if result["warnings"]:
        for item in result["warnings"]:
            print("  - {}".format(item))
    else:
        print("  - 无")

    print("\n[建议]")
    if result["recommendations"]:
        for item in result["recommendations"]:
            print("  - {}".format(item))
    else:
        print("  - 数据集通过当前评价条件")

    if integrity["errors"]:
        print("\n[完整性错误明细，最多显示配置的max-errors条]")
        for item in integrity["errors"]:
            print("  - {}".format(item))
    print("=" * 78)


def _fmt(value):
    if value is None:
        return "N/A"
    return "{:.6g}".format(float(value))


def main():
    args = _parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit("dataset root does not exist: {}".format(root))
    config = _load_config(args.config_file)
    result = _evaluate(root, config, args)
    _print_terminal_report(result)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.plots == "true":
            plot_warning = _write_plots(result, output_dir)
            if plot_warning:
                result["warnings"].append(plot_warning)
        json_path = output_dir / "dataset_evaluation.json"
        markdown_path = output_dir / "dataset_evaluation.md"
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, ensure_ascii=False)
        _write_markdown(result, markdown_path)
        print("报告已保存:")
        print("  {}".format(json_path))
        print("  {}".format(markdown_path))
    if (args.fail_on_error == "true" and
            result["verdict"]["structural_integrity"] != "PASS"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
