#!/usr/bin/env python3
"""
il_config.py  —  Configuration loading / validation for the two-level
navigation expert dataset collector.

The YAML config is organized into the modules defined in the architecture:
task_oracle, observed_map, macro_expert, macro_candidates,
local_recoverability, local_path_search, trajectory_optimization,
yaw_planning, trajectory_controller, execution_safety, dataset_logging
plus infrastructure (fsm, depth, pointcloud, vehicle, control, sync, commit).

Every parameter read here is consumed by the runtime code.
"""

from __future__ import print_function, division

import os

import yaml

import rospkg
import rospy

_PKG = "il_dataset"
_DEFAULT_CONFIG_REL = "config/il_dataset_config.yaml"

# Modules that must exist under `global`.
REQUIRED_MODULES = [
    "fsm", "depth", "pointcloud", "vehicle", "control", "dynamics",
    "task_oracle", "observed_map", "macro_expert", "macro_candidates",
    "local_recoverability", "local_path_search", "trajectory_optimization",
    "yaw_planning", "trajectory_controller", "execution_safety",
    "dataset_logging", "sync", "commit", "privileged_intervention",
    "navigation",
]


def _resolve_path(cfg_path):
    if cfg_path and os.path.isfile(cfg_path):
        return cfg_path
    try:
        rospack = rospkg.RosPack()
        pkg_dir = rospack.get_path(_PKG)
        return os.path.join(pkg_dir, _DEFAULT_CONFIG_REL)
    except Exception:
        # Fall back to a path relative to this file (scripts/..).
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "..", _DEFAULT_CONFIG_REL)
        return os.path.normpath(candidate)


def _get_nested(cfg, path, default=None):
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _positive(value, name, errors, allow_zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append("%s must be a number" % name)
        return
    if allow_zero:
        if value < 0.0:
            errors.append("%s must be >= 0" % name)
    elif value <= 0.0:
        errors.append("%s must be > 0" % name)


def _bounded(value, name, lo, hi, errors):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append("%s must be a number" % name)
        return
    if not (lo <= value <= hi):
        errors.append("%s must be in [%s, %s]" % (name, lo, hi))


def _validate_config(cfg):
    errors = []
    g = cfg.get("global", {})
    for module in REQUIRED_MODULES:
        if not isinstance(g.get(module), dict):
            errors.append("missing global.%s module" % module)

    # Cross-module consistency
    depth_max = _get_nested(g, ["depth", "max_m"], 5.0)
    om = g.get("observed_map", {})
    me = g.get("macro_expert", {})
    mc = g.get("macro_candidates", {})
    lr = g.get("local_recoverability", {})
    lps = g.get("local_path_search", {})
    to = g.get("trajectory_optimization", {})
    yp = g.get("yaw_planning", {})
    tc = g.get("trajectory_controller", {})
    ctrl = g.get("control", {})
    veh = g.get("vehicle", {})
    ds = g.get("dataset_logging", {})

    control_hz = ctrl.get("control_hz", None)
    record_hz = ctrl.get("record_hz", 30.0)
    if control_hz is not None:
        errors.append("global.control.control_hz is removed; the control "
                      "sample rate is dynamics.control_hz, the record rate "
                      "is global.control.record_hz")
    _positive(record_hz, "global.control.record_hz", errors)

    # Wall-clock task watchdog (section "time"): independent of the sim
    # trajectory_timeout, bounds the PROCESS time of one episode so a
    # stalled simulation / degraded loop cannot hang the collector.
    _positive(g.get("fsm", {}).get("trajectory_wall_timeout_s", 600.0),
              "global.fsm.trajectory_wall_timeout_s", errors)

    me_hz = me.get("update_hz", 5.0)
    _positive(me_hz, "global.macro_expert.update_hz", errors)
    if me_hz >= record_hz:
        errors.append("global.macro_expert.update_hz must be < record_hz")
    frames_per_macro = int(round(record_hz / me_hz))
    if abs(record_hz / me_hz - frames_per_macro) > 1e-6:
        errors.append("record_hz / macro update_hz must be an integer")

    _positive(om.get("resolution", 0.10), "observed_map.resolution", errors)
    _positive(om.get("size_x_m", 12.0), "observed_map.size_x_m", errors)
    _positive(om.get("size_y_m", 12.0), "observed_map.size_y_m", errors)
    _positive(om.get("size_z_m", 5.0), "observed_map.size_z_m", errors)
    _positive(om.get("history_seconds", 8.0), "observed_map.history_seconds", errors)
    if me.get("map_history_seconds") is not None:
        errors.append(
            "global.macro_expert.map_history_seconds is removed; use "
            "observed_map.history_seconds")

    # macro lookahead must fit within depth range and observed map size.
    lookahead = me.get("macro_lookahead_distance_m", 4.5)
    _positive(lookahead, "macro_expert.macro_lookahead_distance_m", errors)
    if lookahead > depth_max:
        errors.append("macro_lookahead_distance_m must be <= depth.max_m")
    half_x = om.get("size_x_m", 12.0) / 2.0
    half_y = om.get("size_y_m", 12.0) / 2.0
    if lookahead > 0.9 * min(half_x, half_y):
        errors.append("macro_lookahead_distance_m must fit the observed map")

    # causal_feedback (sections XXV/XXVI): 30 Hz -> 5 Hz interval
    # thresholds for DIRECT causal evidence.
    cf = me.get("causal_feedback", {})
    _positive(cf.get("interval_failure_threshold", 2),
              "macro_expert.causal_feedback.interval_failure_threshold",
              errors, allow_zero=True)
    _positive(cf.get("local_failure_macro_ticks", 2),
              "macro_expert.causal_feedback.local_failure_macro_ticks",
              errors, allow_zero=True)
    _bounded(cf.get("cached_ratio_threshold", 0.5),
             "macro_expert.causal_feedback.cached_ratio_threshold",
             0.0, 1.0, errors)
    _positive(cf.get("cached_macro_ticks", 2),
              "macro_expert.causal_feedback.cached_macro_ticks",
              errors, allow_zero=True)
    _bounded(cf.get("brake_ratio_threshold", 0.5),
             "macro_expert.causal_feedback.brake_ratio_threshold",
             0.0, 1.0, errors)
    _positive(cf.get("brake_macro_ticks", 2),
              "macro_expert.causal_feedback.brake_macro_ticks",
              errors, allow_zero=True)
    _positive(cf.get("emergency_macro_ticks", 1),
              "macro_expert.causal_feedback.emergency_macro_ticks",
              errors, allow_zero=True)
    _positive(me.get("goal_tolerance_m", 0.30), "macro_expert.goal_tolerance_m", errors)
    _positive(me.get("direct_intervention_timeout", 5.0),
              "macro_expert.direct_intervention_timeout", errors)
    _positive(me.get("side_no_progress_seconds", 6.0),
              "macro_expert.side_no_progress_seconds", errors)
    _positive(me.get("observe_no_information_timeout", 4.0),
              "macro_expert.observe_no_information_timeout", errors)
    _positive(me.get("observe_no_progress_fail_ticks", 35),
              "macro_expert.observe_no_progress_fail_ticks", errors)
    _positive(me.get("viewpoint_reset_distance_m", 0.35),
              "macro_expert.viewpoint_reset_distance_m", errors)
    _positive(me.get("local_path_fail_threshold", 2),
              "macro_expert.local_path_fail_threshold", errors)
    _positive(me.get("blocker_rebind_ticks", 2),
              "macro_expert.blocker_rebind_ticks", errors)
    _positive(me.get("blocker_association_distance_m", 1.5),
              "macro_expert.blocker_association_distance_m", errors)
    _positive(me.get("blocker_overlap_pad_m", 0.5),
              "macro_expert.blocker_overlap_pad_m", errors, allow_zero=True)
    _positive(me.get("blocker_lost_grace_s", 2.0),
              "macro_expert.blocker_lost_grace_s", errors)
    _positive(me.get("direct_release_ticks", 3),
              "macro_expert.direct_release_ticks", errors)
    _positive(me.get("macro_intervention_absolute_safety_timeout", 45.0),
              "macro_expert.macro_intervention_absolute_safety_timeout",
              errors)
    if me.get("macro_intervention_absolute_safety_timeout", 45.0) <= \
            me.get("observe_no_information_timeout", 4.0):
        errors.append(
            "macro_expert.macro_intervention_absolute_safety_timeout must "
            "be well above observe_no_information_timeout")
    # P2/P4 side-commitment consistency and anti-oscillation.
    _positive(me.get("side_min_hold_s", 1.5),
              "macro_expert.side_min_hold_s", errors, allow_zero=True)
    _positive(me.get("side_release_cooldown_s", 1.5),
              "macro_expert.side_release_cooldown_s", errors, allow_zero=True)
    _positive(me.get("side_unsafe_clearance_margin_m", 0.05),
              "macro_expert.side_unsafe_clearance_margin_m", errors,
              allow_zero=True)
    _positive(me.get("observed_side_score_margin", 0.15),
              "macro_expert.observed_side_score_margin", errors,
              allow_zero=True)
    _positive(me.get("observe_min_visible_gain", 0.05),
              "macro_expert.observe_min_visible_gain", errors,
              allow_zero=True)
    _positive(me.get("observe_virtual_frontier_distance_m", 2.0),
              "macro_expert.observe_virtual_frontier_distance_m", errors)
    _positive(me.get("side_goal_regress_m", 0.35),
              "macro_expert.side_goal_regress_m", errors, allow_zero=True)
    _positive(me.get("side_goal_regress_ticks", 3),
              "macro_expert.side_goal_regress_ticks", errors,
              allow_zero=True)

    # candidate geometry
    _positive(mc.get("side_corridor_radius_m", 0.55),
              "macro_candidates.side_corridor_radius_m", errors)
    _positive(mc.get("min_observe_move_distance_m", 0.15),
              "macro_candidates.min_observe_move_distance_m", errors)
    if mc.get("side_corridor_radius_m", 0.55) <= veh.get("radius_m", 0.30):
        errors.append(
            "macro_candidates.side_corridor_radius_m must exceed vehicle.radius_m")
    # Active observation viewpoint search (section XV).
    obs = mc.get("observation", {})
    obs_lat = obs.get("lateral_distances_m", [0.4, 0.8, 1.2, 1.6])
    obs_fwd = obs.get("forward_distances_m", [0.0, 0.4, 0.8, 1.2])
    if not isinstance(obs_lat, list) or not obs_lat or \
            any(float(v) <= 0 for v in obs_lat):
        errors.append("macro_candidates.observation.lateral_distances_m "
                      "must be a non-empty list of positive numbers")
    if not isinstance(obs_fwd, list) or not obs_fwd or \
            any(float(v) < 0 for v in obs_fwd):
        errors.append("macro_candidates.observation.forward_distances_m "
                      "must be a non-empty list of >= 0 numbers")
    _positive(obs.get("max_viewpoint_candidates", 24),
              "macro_candidates.observation.max_viewpoint_candidates", errors)
    _positive(obs.get("max_viewpoint_searches_per_tick", 8),
              "macro_candidates.observation.max_viewpoint_searches_per_tick",
              errors)
    _positive(obs.get("min_frontier_searches_per_tick", 2),
              "macro_candidates.observation.min_frontier_searches_per_tick",
              errors)
    if int(obs.get("min_frontier_searches_per_tick", 2)) > \
            int(obs.get("max_viewpoint_searches_per_tick", 8)):
        errors.append("macro_candidates.observation."
                      "min_frontier_searches_per_tick must be <= "
                      "max_viewpoint_searches_per_tick")
    _positive(obs.get("max_observe_move_distance_m", 6.0),
              "macro_candidates.observation.max_observe_move_distance_m", errors)
    _positive(obs.get("visibility_fov_deg", 90.0),
              "macro_candidates.observation.visibility_fov_deg", errors)
    if float(obs.get("visibility_fov_deg", 90.0)) > 180.0:
        errors.append("macro_candidates.observation.visibility_fov_deg "
                      "must be <= 180")
    _positive(obs.get("visibility_ray_count", 31),
              "macro_candidates.observation.visibility_ray_count", errors)
    _positive(obs.get("visibility_range_m", 4.0),
              "macro_candidates.observation.visibility_range_m", errors)
    if int(obs.get("max_viewpoint_searches_per_tick", 8)) > \
            int(obs.get("max_viewpoint_candidates", 24)):
        errors.append("macro_candidates.observation."
                      "max_viewpoint_searches_per_tick must be <= "
                      "max_viewpoint_candidates")
    # P3 known-free recovery (retreat) viewpoints.
    _positive(obs.get("retreat_searches_per_tick", 3),
              "macro_candidates.observation.retreat_searches_per_tick",
              errors, allow_zero=True)
    ret_dist = obs.get("retreat_distances_m", [0.5, 1.0, 1.5])
    if not isinstance(ret_dist, list) or not ret_dist or \
            any(float(v) <= 0 for v in ret_dist):
        errors.append("macro_candidates.observation.retreat_distances_m "
                      "must be a non-empty list of positive numbers")
    _positive(obs.get("retreat_lateral_m", 0.6),
              "macro_candidates.observation.retreat_lateral_m", errors,
              allow_zero=True)
    if int(obs.get("min_frontier_searches_per_tick", 2)) + \
            int(obs.get("retreat_searches_per_tick", 3)) > \
            int(obs.get("max_viewpoint_searches_per_tick", 8)):
        errors.append("macro_candidates.observation."
                      "min_frontier_searches_per_tick + "
                      "retreat_searches_per_tick must be <= "
                      "max_viewpoint_searches_per_tick")

    # recoverability — unified LOCAL capability bounds (section II): the
    # privileged audit shares the SAME rejoin distance / duration / path
    # length / detour limits.
    _positive(lr.get("rejoin_distance_m", 2.5),
              "local_recoverability.rejoin_distance_m", errors)
    _positive(lr.get("search_lateral_margin_m", 2.0),
              "local_recoverability.search_lateral_margin_m", errors)
    _positive(lr.get("search_longitudinal_margin_m", 2.0),
              "local_recoverability.search_longitudinal_margin_m", errors)
    _positive(lr.get("max_path_length_m", 6.0),
              "local_recoverability.max_path_length_m", errors)
    _positive(lr.get("max_duration_s", 2.5),
              "local_recoverability.max_duration_s", errors)
    _bounded(lr.get("min_terminal_alignment", 0.5),
             "local_recoverability.min_terminal_alignment", 0.0, 1.0, errors)
    if lr.get("max_detour_ratio", 1.6) < 1.0:
        errors.append("local_recoverability.max_detour_ratio must be >= 1.0")
    if lr.get("rejoin_distance_m", 2.5) > mc.get("lookahead_distance_m", 4.5):
        errors.append(
            "local_recoverability.rejoin_distance_m must be <= "
            "macro_candidates.lookahead_distance_m")

    # local path search
    _positive(lps.get("max_time_ms", 20.0), "local_path_search.max_time_ms", errors)

    # trajectory optimization
    _positive(to.get("planning_time_budget_ms", 30.0),
              "trajectory_optimization.planning_time_budget_ms", errors)
    _positive(to.get("trajectory_dt", 0.04),
              "trajectory_optimization.trajectory_dt", errors)
    _positive(to.get("nominal_speed", 1.8),
              "trajectory_optimization.nominal_speed", errors)
    _positive(to.get("max_velocity", 2.5),
              "trajectory_optimization.max_velocity", errors)
    if to.get("max_velocity", 2.5) < to.get("nominal_speed", 1.8):
        errors.append("trajectory_optimization.max_velocity must be >= nominal_speed")
    _positive(to.get("max_acceleration", 8.0),
              "trajectory_optimization.max_acceleration", errors)
    if to.get("optimizer", "auto") not in ("auto", "nlopt", "native"):
        errors.append("trajectory_optimization.optimizer must be auto|nlopt|native")
    _positive(to.get("goal_stop_tolerance_m", 0.4),
              "trajectory_optimization.goal_stop_tolerance_m", errors)

    # yaw planning
    _positive(yp.get("max_yaw_rate", 2.0), "yaw_planning.max_yaw_rate", errors)
    _positive(yp.get("max_yaw_accel", 8.0), "yaw_planning.max_yaw_accel", errors)
    _bounded(yp.get("fov_half_deg", 45.0), "yaw_planning.fov_half_deg", 1.0, 89.0, errors)
    _positive(yp.get("fov_margin_deg", 5.0), "yaw_planning.fov_margin_deg", errors, allow_zero=True)

    # trajectory controller
    _positive(tc.get("velocity_lookahead_time_s", 0.08),
              "trajectory_controller.velocity_lookahead_time_s", errors)
    _positive(tc.get("position_gain", 2.0),
              "trajectory_controller.position_gain", errors)
    _positive(tc.get("max_velocity_mps", 2.5),
              "trajectory_controller.max_velocity_mps", errors)
    _positive(tc.get("max_acceleration_mps2", 3.5),
              "trajectory_controller.max_acceleration_mps2", errors)
    _positive(tc.get("max_yaw_rate_rps", 2.0),
              "trajectory_controller.max_yaw_rate_rps", errors)
    ah = tc.get("altitude_hold", {})
    if not isinstance(ah, dict):
        errors.append("trajectory_controller.altitude_hold must be a mapping")
    else:
        _positive(ah.get("kp", 1.5),
                  "trajectory_controller.altitude_hold.kp", errors)
        _positive(ah.get("kd", 0.6),
                  "trajectory_controller.altitude_hold.kd", errors,
                  allow_zero=True)
        _positive(ah.get("deadband_m", 0.02),
                  "trajectory_controller.altitude_hold.deadband_m", errors,
                  allow_zero=True)
        _positive(ah.get("max_speed_mps", 0.6),
                  "trajectory_controller.altitude_hold.max_speed_mps", errors)
    _positive(tc.get("max_yaw_accel_rps2", 8.0),
              "trajectory_controller.max_yaw_accel_rps2", errors)
    _positive(tc.get("emergency_brake_distance_m", 0.8),
              "trajectory_controller.emergency_brake_distance_m", errors)
    if tc.get("command_change_rate_limit_mps2") is not None:
        errors.append(
            "global.trajectory_controller.command_change_rate_limit_mps2 is "
            "removed; use max_jerk_mps3 + max_acceleration_mps2 instead")
    _positive(tc.get("max_jerk_mps3", 25.0),
              "trajectory_controller.max_jerk_mps3", errors)

    # execution safety
    es = g.get("execution_safety", {})
    _positive(es.get("max_plan_age_s", 0.5),
              "execution_safety.max_plan_age_s", errors)
    _positive(es.get("min_remaining_trajectory_s", 0.25),
              "execution_safety.min_remaining_trajectory_s", errors)
    _positive(es.get("max_position_error_m", 0.6),
              "execution_safety.max_position_error_m", errors)
    _positive(es.get("max_velocity_error_mps", 1.0),
              "execution_safety.max_velocity_error_mps", errors)
    _positive(es.get("emergency_deceleration_mps2", 3.0),
              "execution_safety.emergency_deceleration_mps2", errors)
    _positive(es.get("brake_reaction_delay_s", 0.10),
              "execution_safety.brake_reaction_delay_s", errors)
    _positive(es.get("max_brake_hold_seconds", 1.0),
              "execution_safety.max_brake_hold_seconds", errors)
    _positive(es.get("max_emergency_stop_seconds", 2.0),
              "execution_safety.max_emergency_stop_seconds", errors)

    # dataset logging
    _positive(ds.get("schema_version", 23),
              "dataset_logging.schema_version", errors)
    if ds.get("schema_version", 23) != 23:
        errors.append("dataset_logging.schema_version must be 23")
    _positive(ds.get("perception_range_m", 5.0),
              "dataset_logging.perception_range_m", errors)
    _positive(ds.get("flush_interval_rows", 64),
              "dataset_logging.flush_interval_rows", errors)
    _bounded(ds.get("depth_png_compress_level", 4),
             "dataset_logging.depth_png_compress_level", 0, 9, errors)

    # privileged intervention (privileged LOCAL-SCALE audit, section II)
    pi = g.get("privileged_intervention", {})
    _positive(pi.get("search_max_time_ms", 20.0),
              "privileged_intervention.search_max_time_ms", errors)
    # Capability bounds are shared with local_recoverability (single
    # source).
    _positive(pi.get("rejoin_radius_m", 0.6),
              "privileged_intervention.rejoin_radius_m", errors)
    _positive(pi.get("loop_ignore_recent_s", 2.5),
              "privileged_intervention.loop_ignore_recent_s", errors)
    if pi.get("loop_leave_radius_m", 1.6) < pi.get("loop_revisit_radius_m", 0.8):
        errors.append(
            "privileged_intervention.loop_leave_radius_m must be >= "
            "loop_revisit_radius_m")
    _positive(pi.get("loop_revisit_radius_m", 0.8),
              "privileged_intervention.loop_revisit_radius_m", errors)
    _positive(pi.get("loop_leave_radius_m", 1.6),
              "privileged_intervention.loop_leave_radius_m", errors)
    _positive(pi.get("loop_min_speed_mps", 0.3),
              "privileged_intervention.loop_min_speed_mps", errors)
    _positive(pi.get("loop_min_revisits", 2),
              "privileged_intervention.loop_min_revisits", errors)
    _positive(pi.get("cost_margin_m", 2.0),
              "privileged_intervention.cost_margin_m", errors)

    # task oracle
    to_ = g.get("task_oracle", {})
    _positive(to_.get("map_resolution_m", 0.10),
              "task_oracle.map_resolution_m", errors)

    # ── UNIFIED navigation clearance (problem 4) ────────────────────
    # `navigation.clearance_m` is the SINGLE effective safety boundary for
    # every module (global connectivity/cost-to-go, local A*, recoverability,
    # privileged intervention, trajectory optimisation/validation, goal
    # stop, braking risk, start/goal/task generation).  The ESDFs already
    # subtract the vehicle radius, so this is purely the additional margin.
    nav = g.get("navigation", {})
    _positive(nav.get("clearance_m", 0.20), "navigation.clearance_m", errors)
    # P1 dynamic executability margin: speed-dependent extra buffer for the
    # LOCAL layer only (planner + braking).  Never a second clearance value.
    _positive(nav.get("margin_tracking_m", 0.05),
              "navigation.margin_tracking_m", errors, allow_zero=True)
    _positive(nav.get("margin_latency_s", 0.10),
              "navigation.margin_latency_s", errors, allow_zero=True)
    _positive(nav.get("margin_max_m", 0.25),
              "navigation.margin_max_m", errors, allow_zero=True)

    # scene generation (section LXXV)
    sg = g.get("scene_generation", {})
    if sg.get("enabled", True):
        _positive(sg.get("seed", 12345), "scene_generation.seed", errors)
        _positive(sg.get("tasks_per_scene", 12),
                  "scene_generation.tasks_per_scene", errors, allow_zero=True)
        _positive(sg.get("minimum_tasks_per_scene", 1),
                  "scene_generation.minimum_tasks_per_scene", errors,
                  allow_zero=True)
        if sg.get("minimum_tasks_per_scene", 1) > \
                sg.get("tasks_per_scene", 12):
            errors.append(
                "scene_generation.minimum_tasks_per_scene must be <= "
                "tasks_per_scene")
        if not sg.get("profiles"):
            errors.append("scene_generation.profiles must not be empty")

    # task generation (section LXXV)
    tg = g.get("task_generation", {})
    if tg.get("enabled", True):
        _positive(tg.get("validation_speed_mps", 0.0),
                  "task_generation.validation_speed_mps", errors,
                  allow_zero=True)
        _positive(tg.get("candidate_batch_size", 64),
                  "task_generation.candidate_batch_size", errors)
        _positive(tg.get("maximum_batches_per_scene", 6),
                  "task_generation.maximum_batches_per_scene", errors)
        _positive(tg.get("flight_height_m", 5.0),
                  "task_generation.flight_height_m", errors)
        if not tg.get("class_weights"):
            errors.append("task_generation.class_weights must not be empty")

    # sync sanity
    sync = g.get("sync", {})
    if sync.get("unity_response_timeout_s", 2.0) <= 0:
        errors.append("sync.unity_response_timeout_s must be > 0")
    if sync.get("max_acceptable_latency_ms", 250.0) <= 0:
        errors.append("sync.max_acceptable_latency_ms must be > 0")

    if errors:
        raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))


def load_config(config_path=None, validate=True):
    """Load and validate the YAML configuration.

    Returns the full config dict (``config["global"]`` holds the modules).
    """
    path = _resolve_path(config_path)
    if not os.path.isfile(path):
        raise ValueError("Config file not found: %s" % path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "global" not in cfg:
        raise ValueError("Config must have a top-level 'global' section")
    g = cfg["global"]

    # ROS param overrides (ports and scene id).
    try:
        if rospy.has_param("~pub_port"):
            g["pub_port"] = rospy.get_param("~pub_port")
        if rospy.has_param("~sub_port"):
            g["sub_port"] = rospy.get_param("~sub_port")
        if rospy.has_param("~scene_id"):
            g["scene_id"] = int(rospy.get_param("~scene_id"))
        if rospy.has_param("~config_file"):
            g["_config_source"] = str(rospy.get_param("~config_file"))
    except Exception:
        pass

    if not g.get("output_dir"):
        try:
            rospack = rospkg.RosPack()
            pkg_dir = rospack.get_path(_PKG)
            g["output_dir"] = os.path.join(pkg_dir, "dataset", "il_data")
        except Exception:
            g["output_dir"] = "dataset/il_data"

    if validate:
        _validate_config(cfg)
    return cfg


# ============================================================================
#  Builders for the C++ (pybind) configuration objects.
#  Every parameter below is consumed by the corresponding C++ module.
# ============================================================================

def build_observed_map_config(g, module):
    om = g.get("observed_map", {})
    cfg = module.ObservedMapConfig()
    cfg.resolution = float(om.get("resolution", 0.10))
    cfg.size_x_m = float(om.get("size_x_m", 12.0))
    cfg.size_y_m = float(om.get("size_y_m", 12.0))
    cfg.size_z_m = float(om.get("size_z_m", 5.0))
    cfg.history_seconds = float(om.get("history_seconds", 8.0))
    cfg.occupied_endpoint_margin_m = float(
        om.get("occupied_endpoint_margin_m", 0.05))
    cfg.vehicle_radius_m = float(g.get("vehicle", {}).get("radius_m", 0.30))
    cfg.max_depth_m = float(g.get("depth", {}).get("max_m", 5.0))
    cfg.horizontal_fov_deg = float(g.get("depth", {}).get("fov", 90.0))
    cfg.esdf_max_distance_m = float(om.get("esdf_max_distance_m", 5.0))
    cfg.free_space_spacing_m = float(om.get("free_space_sample_spacing_m", 0.10))
    cfg.depth_integration_step = int(om.get("depth_integration_step", 2))
    cfg.rebuild_every_n_frames = int(om.get("rebuild_every_n_frames", 3))
    cfg.recenter_threshold_m = float(om.get("recenter_threshold_m", 3.0))
    return cfg


def build_oracle_config(g, module):
    to_ = g.get("task_oracle", {})
    nav = g.get("navigation", {})
    cfg = module.PrivilegedOracleConfig()
    cfg.resolution = float(to_.get("map_resolution_m", 0.10))
    cfg.vehicle_radius_m = float(g.get("vehicle", {}).get("radius_m", 0.30))
    # UNIFIED navigation clearance (problem 4): the single additional
    # safety margin used by every module.
    cfg.clearance_m = float(nav.get("clearance_m", 0.20))
    cfg.max_esdf_distance_m = float(to_.get("max_esdf_distance_m", 8.0))
    cfg.map_margin_m = float(to_.get("map_margin_m", 2.0))
    cfg.min_z_m = float(to_.get("min_z_m", 0.0))
    cfg.max_z_m = float(to_.get("max_z_m", 8.0))
    cfg.cost_to_go_cap_m = float(to_.get("cost_to_go_cap_m", 30.0))
    s = cfg.scoring
    s.weight_observed_cost = float(to_.get("weight_observed_cost", 1.0))
    s.weight_cost_to_go = float(to_.get("weight_cost_to_go", 2.0))
    s.weight_connectivity = float(to_.get("weight_connectivity", 6.0))
    s.weight_clearance = float(to_.get("weight_clearance", 1.0))
    s.weight_goal_progress = float(to_.get("weight_goal_progress", 1.0))
    s.weight_information = float(to_.get("weight_information", 0.5))
    s.weight_yaw_cost = float(to_.get("weight_yaw_cost", 0.5))
    s.weight_side_switch = float(to_.get("weight_side_switch", 1.0))
    s.weight_repeat = float(to_.get("weight_repeat", 0.5))
    s.side_switch_penalty = float(to_.get("side_switch_penalty", 1.0))
    s.repeat_penalty = float(to_.get("repeat_penalty", 1.0))
    s.clearance_target_m = float(to_.get("clearance_target_m", 0.6))
    s.yaw_cost_scale_rad = float(to_.get("yaw_cost_scale_rad", 1.0))
    return cfg


def build_macro_candidate_config(g, module):
    mc = g.get("macro_candidates", {})
    nav = g.get("navigation", {})
    cfg = module.MacroCandidateConfig()
    cfg.lookahead_distance_m = float(mc.get("lookahead_distance_m", 4.5))
    cfg.side_corridor_length_m = float(mc.get("side_corridor_length_m", 4.0))
    cfg.side_corridor_radius_m = float(mc.get("side_corridor_radius_m", 0.55))
    cfg.edge_search_radius_m = float(mc.get("edge_search_radius_m", 5.0))
    # UNIFIED navigation clearance (problem 4): candidate known-free and
    # observed LocalPathSearch reachability use the SAME additional margin
    # as every other module.
    cfg.clearance_m = float(nav.get("clearance_m", 0.20))
    # Round 5: the SAME dynamic-executability margin parameters the 30 Hz
    # LocalPlanner uses (single source = navigation), so candidate endpoint
    # filters and FULL-reachable A* are evaluated at a clearance never
    # below what the planner validates with.
    cfg.clearance_margin_tracking_m = float(
        nav.get("margin_tracking_m", 0.05))
    cfg.clearance_margin_latency_s = float(
        nav.get("margin_latency_s", 0.10))
    cfg.clearance_margin_max_m = float(nav.get("margin_max_m", 0.25))
    cfg.candidate_spacing_m = float(mc.get("candidate_spacing_m", 0.5))
    cfg.observe_step_m = float(mc.get("observe_step_m", 0.6))
    cfg.min_observe_move_distance_m = float(
        mc.get("min_observe_move_distance_m", 0.15))
    # Active observation viewpoint search: lattice + FULL LocalPathSearch
    # budget + FOV/known-occlusion-aware expected visibility.
    obs = mc.get("observation", {})
    cfg.observe_lateral_distances_m = [float(v) for v in
        obs.get("lateral_distances_m", [0.4, 0.8, 1.2, 1.6])]
    cfg.observe_forward_distances_m = [float(v) for v in
        obs.get("forward_distances_m", [0.0, 0.4, 0.8, 1.2])]
    cfg.max_viewpoint_candidates = int(obs.get("max_viewpoint_candidates", 24))
    cfg.max_viewpoint_searches_per_tick = int(
        obs.get("max_viewpoint_searches_per_tick", 8))
    cfg.min_frontier_searches_per_tick = int(
        obs.get("min_frontier_searches_per_tick", 2))
    # P3 known-free recovery (retreat) viewpoints.
    cfg.retreat_searches_per_tick = int(
        obs.get("retreat_searches_per_tick", 3))
    cfg.retreat_distances_m = [float(v) for v in
        obs.get("retreat_distances_m", [0.5, 1.0, 1.5])]
    cfg.retreat_lateral_m = float(obs.get("retreat_lateral_m", 0.6))
    cfg.max_observe_move_distance_m = float(
        obs.get("max_observe_move_distance_m", 6.0))
    cfg.observe_visibility_fov_deg = float(
        obs.get("visibility_fov_deg", 90.0))
    cfg.observe_visibility_ray_count = int(
        obs.get("visibility_ray_count", 31))
    cfg.observe_visibility_range_m = float(
        obs.get("visibility_range_m", 4.0))
    cfg.max_frontier_candidates = int(mc.get("max_frontier_candidates", 8))
    cfg.frontier_standoff_m = float(mc.get("frontier_standoff_m", 0.45))
    cfg.goal_frontier_cone_deg = float(mc.get("goal_frontier_cone_deg", 70.0))
    cfg.corridor_check_spacing_m = float(mc.get("corridor_check_spacing_m", 0.10))
    # Observed-map path-search parameters for REAL SIDE-candidate
    # reachability (section XII).  Taken from local_path_search.
    lps = g.get("local_path_search", {})
    cfg.search_max_time_ms = float(lps.get("max_time_ms", 20.0))
    cfg.search_region_margin_m = float(lps.get("region_margin_m", 2.0))
    cfg.side_bias_gain = float(lps.get("side_bias_gain", 2.0))
    return cfg


def build_intervention_config(g, module):
    # The privileged LOCAL-SCALE audit shares the SAME capability bounds as
    # the observed local recoverability (section II).
    lr = g.get("local_recoverability", {})
    pi = g.get("privileged_intervention", {})
    nav = g.get("navigation", {})
    cfg = module.PrivilegedInterventionConfig()
    # UNIFIED navigation clearance (problem 4).
    cfg.clearance_m = float(nav.get("clearance_m", 0.20))
    cfg.search_max_time_ms = float(pi.get("search_max_time_ms", 20.0))
    cfg.rejoin_distance_m = float(lr.get("rejoin_distance_m", 2.5))
    cfg.search_lateral_margin_m = float(lr.get("search_lateral_margin_m", 2.0))
    cfg.search_longitudinal_margin_m = float(
        lr.get("search_longitudinal_margin_m", 2.0))
    cfg.max_duration_s = float(lr.get("max_duration_s", 2.5))
    cfg.max_path_length_m = float(lr.get("max_path_length_m", 6.0))
    cfg.nominal_speed_mps = float(lr.get("nominal_speed_mps", 1.8))
    cfg.max_detour_ratio = float(lr.get("max_detour_ratio", 1.6))
    cfg.min_goal_progress_m = float(lr.get("min_goal_progress_m", 0.30))
    cfg.min_terminal_alignment = float(lr.get("min_terminal_alignment", 0.5))
    cfg.terminal_tangent_min_baseline = float(
        lr.get("terminal_tangent_min_baseline", 0.3))
    cfg.loop_ignore_recent_s = float(pi.get("loop_ignore_recent_s", 2.5))
    cfg.loop_leave_radius_m = float(pi.get("loop_leave_radius_m", 1.6))
    cfg.loop_revisit_radius_m = float(pi.get("loop_revisit_radius_m", 0.8))
    cfg.loop_min_speed_mps = float(pi.get("loop_min_speed_mps", 0.3))
    cfg.loop_min_revisits = int(pi.get("loop_min_revisits", 2))
    cfg.loop_history_size = int(pi.get("loop_history_size", 60))
    return cfg


def build_recoverability_config(g, module):
    lr = g.get("local_recoverability", {})
    nav = g.get("navigation", {})
    cfg = module.RecoverabilityConfig()
    cfg.rejoin_distance_m = float(lr.get("rejoin_distance_m", 2.5))
    # UNIFIED navigation clearance (problem 4).
    cfg.clearance_m = float(nav.get("clearance_m", 0.20))
    # Round 6: the SAME dynamic-executability margin parameters the 30 Hz
    # LocalPlanner and the macro candidate search use (single source =
    # navigation), so the recoverability query is never more permissive
    # than actual local planning.
    cfg.clearance_margin_tracking_m = float(
        nav.get("margin_tracking_m", 0.05))
    cfg.clearance_margin_latency_s = float(
        nav.get("margin_latency_s", 0.10))
    cfg.clearance_margin_max_m = float(nav.get("margin_max_m", 0.25))
    cfg.max_duration_s = float(lr.get("max_duration_s", 2.5))
    cfg.max_path_length_m = float(lr.get("max_path_length_m", 6.0))
    cfg.min_goal_progress_m = float(lr.get("min_goal_progress_m", 0.30))
    cfg.min_terminal_alignment = float(lr.get("min_terminal_alignment", 0.5))
    cfg.max_detour_ratio = float(lr.get("max_detour_ratio", 1.6))
    cfg.nominal_speed_mps = float(lr.get("nominal_speed_mps", 1.8))
    cfg.terminal_tangent_min_baseline = float(
        lr.get("terminal_tangent_min_baseline", 0.3))
    cfg.side_corridor_length_m = float(lr.get("side_corridor_length_m", 4.0))
    cfg.side_corridor_radius_m = float(lr.get("side_corridor_radius_m", 0.55))
    cfg.edge_search_radius_m = float(lr.get("edge_search_radius_m", 5.0))
    return cfg


def build_task_generation_config(g, module):
    """Build the C++ TaskGenerationConfig (sections XXVIII/XXXIX).

    The clearance is the single UNIFIED navigation clearance
    (`navigation.clearance_m`, problem 4): the global ESDF already subtracts
    the vehicle radius, so this is purely the additional safety margin used
    for start/goal free tests, the direct corridor, the lateral probes AND
    the local audit.  The capability bounds come from local_recoverability
    / local_path_search so the generated classes match the real behaviour
    scale.
    """
    tg = g.get("task_generation", {})
    lr = g.get("local_recoverability", {})
    lps = g.get("local_path_search", {})
    nav = g.get("navigation", {})
    cfg = module.TaskGenerationConfig()
    cfg.clearance_m = float(nav.get("clearance_m", 0.20))
    cfg.clearance_margin_tracking_m = float(
        nav.get("margin_tracking_m", 0.05))
    cfg.clearance_margin_latency_s = float(
        nav.get("margin_latency_s", 0.10))
    cfg.clearance_margin_max_m = float(nav.get("margin_max_m", 0.25))
    # Stationary validation includes the non-zero tracking floor.  Increasing
    # this optional speed makes task generation stricter for faster expected
    # execution without creating another clearance definition.
    cfg.validation_speed_mps = float(tg.get("validation_speed_mps", 0.0))
    bands = tg.get("distance_bands", {}) or {}
    mins = [float(b.get("min_m", 4.0)) for b in bands.values()]
    maxs = [float(b.get("max_m", 28.0)) for b in bands.values()]
    cfg.min_task_distance_m = float(min(mins)) if mins else 3.0
    cfg.max_task_distance_m = float(max(maxs)) if maxs else 30.0
    sampling = tg.get("sampling", {}) or {}
    cfg.lateral_probe_offset_m = float(
        sampling.get("lateral_probe_offset_m", 1.2))
    cfg.lateral_probe_spacing_m = float(
        sampling.get("lateral_probe_spacing_m", 0.6))
    cfg.lateral_probe_count = int(sampling.get("lateral_probe_count", 4))
    # Local-scale audit capability (same as local_recoverability).  The
    # clearance is the SAME unified navigation clearance set above.
    cfg.search_max_time_ms = float(lps.get("max_time_ms", 20.0))
    cfg.rejoin_distance_m = float(lr.get("rejoin_distance_m", 2.5))
    cfg.max_duration_s = float(lr.get("max_duration_s", 2.5))
    cfg.max_path_length_m = float(lr.get("max_path_length_m", 6.0))
    cfg.nominal_speed_mps = float(lr.get("nominal_speed_mps", 1.8))
    cfg.max_detour_ratio = float(lr.get("max_detour_ratio", 1.6))
    cfg.min_goal_progress_m = float(lr.get("min_goal_progress_m", 0.30))
    cfg.min_terminal_alignment = float(lr.get("min_terminal_alignment", 0.5))
    cfg.terminal_tangent_min_baseline = float(
        lr.get("terminal_tangent_min_baseline", 0.3))
    cfg.search_lateral_margin_m = float(lr.get("search_lateral_margin_m", 2.0))
    cfg.search_longitudinal_margin_m = float(
        lr.get("search_longitudinal_margin_m", 2.0))
    return cfg


def build_planner_config(g, module):
    to = g.get("trajectory_optimization", {})
    lps = g.get("local_path_search", {})
    yp = g.get("yaw_planning", {})
    nav = g.get("navigation", {})
    cfg = module.TrajectoryOptimizationConfig()
    cfg.planning_time_budget_ms = float(to.get("planning_time_budget_ms", 30.0))
    cfg.trajectory_dt = float(to.get("trajectory_dt", 0.04))
    cfg.horizon_time = float(to.get("horizon_time", 2.5))
    cfg.optimizer = str(to.get("optimizer", "auto"))
    cfg.control_points = int(to.get("control_points", 12))
    cfg.max_iterations = int(to.get("max_iterations", 10000))
    cfg.convergence_tolerance = float(to.get("convergence_tolerance", 1.0e-4))
    cfg.initial_step_size = float(to.get("initial_step_size", 0.1))
    cfg.minimum_step_size = float(to.get("minimum_step_size", 1.0e-4))
    cfg.seed_trust_radius = float(to.get("seed_trust_radius", 0.35))
    cfg.horizontal_avoidance_only = bool(to.get("horizontal_avoidance_only", True))
    # UNIFIED navigation clearance (problem 4): the single effective safety
    # boundary — both the optimizer's soft target and the hard validation
    # floor.  The observed ESDF already subtracts the vehicle radius.
    cfg.clearance_m = float(nav.get("clearance_m", 0.20))
    # P1 dynamic executability margin (single source with the Python
    # braking check): speed-dependent extra buffer for the LOCAL layer.
    cfg.clearance_margin_tracking_m = float(
        nav.get("margin_tracking_m", 0.05))
    cfg.clearance_margin_latency_s = float(
        nav.get("margin_latency_s", 0.10))
    cfg.clearance_margin_max_m = float(nav.get("margin_max_m", 0.25))
    cfg.collision_check_spacing = float(to.get("collision_check_spacing", 0.05))
    cfg.weight_path_length = float(to.get("weight_path_length", 0.05))
    cfg.weight_smooth = float(to.get("weight_smooth", 1.0))
    cfg.weight_jerk = float(to.get("weight_jerk", 0.2))
    cfg.weight_obstacle = float(to.get("weight_obstacle", 4.0))
    cfg.weight_dynamics = float(to.get("weight_dynamics", 1.0))
    cfg.nominal_speed = float(to.get("nominal_speed", 1.8))
    cfg.max_velocity = float(to.get("max_velocity", 2.5))
    cfg.max_acceleration = float(to.get("max_acceleration", 8.0))
    cfg.max_jerk = float(to.get("max_jerk", 50.0))
    cfg.lookahead_distance = float(to.get("lookahead_distance", 4.0))
    cfg.terminal_speed_ratio = float(to.get("terminal_speed_ratio", 0.85))
    cfg.goal_stop_tolerance_m = float(to.get("goal_stop_tolerance_m", 0.4))
    cfg.warm_start_max_age_s = float(to.get("warm_start_max_age_s", 0.25))
    cfg.warm_start_max_terminal_deviation_m = float(
        to.get("warm_start_max_terminal_deviation_m", 1.5))
    # Local A* search parameters come from the local_path_search module.
    # The clearance is the SAME unified navigation clearance set above.
    cfg.search_max_time_ms = float(lps.get("max_time_ms", 18.0))
    cfg.search_region_margin_m = float(lps.get("region_margin_m", 2.0))
    cfg.search_side_bias_gain = float(lps.get("side_bias_gain", 2.0))
    # Yaw planning parameters come from the yaw_planning module.
    cfg.yaw_max_rate = float(yp.get("max_yaw_rate", 2.0))
    cfg.yaw_max_accel = float(yp.get("max_yaw_accel", 8.0))
    cfg.yaw_fov_half_deg = float(yp.get("fov_half_deg", 45.0))
    cfg.yaw_fov_margin_deg = float(yp.get("fov_margin_deg", 5.0))
    cfg.yaw_speed_threshold_mps = float(yp.get("speed_threshold_mps", 0.20))
    return cfg
