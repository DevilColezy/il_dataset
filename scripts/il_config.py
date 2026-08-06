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

import math
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
    "dataset_logging", "sync", "commit",
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

    for key in ("direct_to_side_frames", "side_to_direct_frames"):
        _positive(me.get(key), "macro_expert.%s" % key, errors, allow_zero=True)
    _positive(me.get("goal_tolerance_m", 0.30), "macro_expert.goal_tolerance_m", errors)

    # candidate geometry
    _positive(mc.get("side_corridor_radius_m", 0.55),
              "macro_candidates.side_corridor_radius_m", errors)
    _positive(mc.get("min_candidate_clearance_m", 0.25),
              "macro_candidates.min_candidate_clearance_m", errors)
    if mc.get("side_corridor_radius_m", 0.55) <= veh.get("radius_m", 0.30):
        errors.append(
            "macro_candidates.side_corridor_radius_m must exceed vehicle.radius_m")

    # recoverability horizon must match the local planning horizon.
    _positive(lr.get("max_execution_time_s", 2.5),
              "local_recoverability.max_execution_time_s", errors)
    _positive(lr.get("rejoin_distance_m", 2.5),
              "local_recoverability.rejoin_distance_m", errors)
    _bounded(lr.get("min_terminal_alignment", 0.5),
             "local_recoverability.min_terminal_alignment", 0.0, 1.0, errors)
    _positive(lr.get("max_loop_ratio", 1.6),
              "local_recoverability.max_loop_ratio", errors)
    if lr.get("rejoin_distance_m", 2.5) > mc.get("lookahead_distance_m", 4.5):
        errors.append(
            "local_recoverability.rejoin_distance_m must be <= "
            "macro_candidates.lookahead_distance_m")

    # local path search
    _positive(lps.get("search_clearance_m", 0.25),
              "local_path_search.search_clearance_m", errors)
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
    _positive(to.get("min_clearance", 0.02),
              "trajectory_optimization.min_clearance", errors, allow_zero=True)
    if to.get("target_clearance", 0.20) < to.get("min_clearance", 0.02):
        errors.append("target_clearance must be >= min_clearance")
    if to.get("optimizer", "auto") not in ("auto", "nlopt", "native"):
        errors.append("trajectory_optimization.optimizer must be auto|nlopt|native")

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
    _positive(tc.get("command_change_rate_limit_mps2", 25.0),
              "trajectory_controller.command_change_rate_limit_mps2", errors)
    _positive(tc.get("emergency_brake_distance_m", 0.8),
              "trajectory_controller.emergency_brake_distance_m", errors)

    # dataset logging
    _positive(ds.get("schema_version", 18),
              "dataset_logging.schema_version", errors)
    if ds.get("schema_version", 18) != 18:
        errors.append("dataset_logging.schema_version must be 18")

    # task oracle
    to_ = g.get("task_oracle", {})
    _positive(to_.get("map_resolution_m", 0.10),
              "task_oracle.map_resolution_m", errors)
    _positive(to_.get("inflation_m", 0.30), "task_oracle.inflation_m", errors)
    if not to_.get("task_manifest_dir"):
        errors.append("task_oracle.task_manifest_dir must be set")

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
    cfg = module.PrivilegedOracleConfig()
    cfg.resolution = float(to_.get("map_resolution_m", 0.10))
    cfg.vehicle_radius_m = float(g.get("vehicle", {}).get("radius_m", 0.30))
    cfg.inflation_m = float(to_.get("inflation_m", 0.30))
    cfg.max_esdf_distance_m = float(to_.get("max_esdf_distance_m", 8.0))
    cfg.map_margin_m = float(to_.get("map_margin_m", 2.0))
    cfg.min_z_m = float(to_.get("min_z_m", 0.0))
    cfg.max_z_m = float(to_.get("max_z_m", 8.0))
    cfg.free_clearance_m = float(to_.get("free_clearance_m", 0.10))
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
    cfg = module.MacroCandidateConfig()
    cfg.lookahead_distance_m = float(mc.get("lookahead_distance_m", 4.5))
    cfg.side_corridor_length_m = float(mc.get("side_corridor_length_m", 4.0))
    cfg.side_corridor_radius_m = float(mc.get("side_corridor_radius_m", 0.55))
    cfg.edge_search_radius_m = float(mc.get("edge_search_radius_m", 5.0))
    cfg.min_candidate_clearance_m = float(mc.get("min_candidate_clearance_m", 0.25))
    cfg.candidate_spacing_m = float(mc.get("candidate_spacing_m", 0.5))
    cfg.observe_step_m = float(mc.get("observe_step_m", 0.6))
    cfg.max_frontier_candidates = int(mc.get("max_frontier_candidates", 8))
    cfg.frontier_standoff_m = float(mc.get("frontier_standoff_m", 0.45))
    cfg.goal_frontier_cone_deg = float(mc.get("goal_frontier_cone_deg", 70.0))
    cfg.corridor_check_spacing_m = float(mc.get("corridor_check_spacing_m", 0.10))
    return cfg


def build_recoverability_config(g, module):
    lr = g.get("local_recoverability", {})
    lps = g.get("local_path_search", {})
    cfg = module.RecoverabilityConfig()
    cfg.rejoin_distance_m = float(lr.get("rejoin_distance_m", 2.5))
    cfg.search_clearance_m = float(lps.get("search_clearance_m", 0.25))
    cfg.max_execution_time_s = float(lr.get("max_execution_time_s", 2.5))
    cfg.min_goal_progress_m = float(lr.get("min_goal_progress_m", 0.30))
    cfg.min_terminal_alignment = float(lr.get("min_terminal_alignment", 0.5))
    cfg.max_loop_ratio = float(lr.get("max_loop_ratio", 1.6))
    cfg.nominal_speed_mps = float(lr.get("nominal_speed_mps", 1.8))
    cfg.side_corridor_length_m = float(lr.get("side_corridor_length_m", 4.0))
    cfg.side_corridor_radius_m = float(lr.get("side_corridor_radius_m", 0.55))
    cfg.edge_search_radius_m = float(lr.get("edge_search_radius_m", 5.0))
    return cfg


def build_planner_config(g, module):
    to = g.get("trajectory_optimization", {})
    lps = g.get("local_path_search", {})
    yp = g.get("yaw_planning", {})
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
    cfg.min_clearance = float(to.get("min_clearance", 0.02))
    cfg.target_clearance = float(to.get("target_clearance", 0.20))
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
    cfg.warm_start_max_age_s = float(to.get("warm_start_max_age_s", 0.25))
    cfg.warm_start_max_terminal_deviation_m = float(
        to.get("warm_start_max_terminal_deviation_m", 1.5))
    # Local A* search parameters come from the local_path_search module.
    cfg.search_clearance_m = float(lps.get("search_clearance_m", 0.25))
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
