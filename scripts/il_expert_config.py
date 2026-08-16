#!/usr/bin/env python3
"""
il_expert_config.py  —  Builds the C++ Params2D (the single authoritative
expert parameter set) from the loaded YAML configuration.

There is exactly ONE source of truth for every expert parameter: the
``global.hierarchical_expert`` YAML section.  This module only re-orders
and validates those values into the pybind Params2D object — it never
introduces a second set of defaults.
"""

from __future__ import print_function, division

import math

try:
    from _il_hierarchical_expert import Params2D
except ImportError:  # pragma: no cover - module not built yet
    Params2D = None


def _get(cfg, path, default=None):
    node = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _num(value, name, errors, default=None, allow_zero=False):
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append("%s must be a number" % name)
        return default
    if not allow_zero and float(value) <= 0.0:
        errors.append("%s must be > 0" % name)
    if not math.isfinite(float(value)):
        errors.append("%s must be finite" % name)
    return float(value)


def _num_list(value, name, errors, default=None):
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or not value or \
            any(not isinstance(v, (int, float)) or isinstance(v, bool) or
                not math.isfinite(float(v)) for v in value):
        errors.append("%s must be a non-empty list of finite numbers" % name)
        return default
    return [float(v) for v in value]


def _parse_t_bc(value, errors, name="depth.t_bc"):
    """Parse a Unity T_BC 4x4 (16 floats, row-major) into
    (translation [tx, ty, tz], rotation 3x3 row-major [9 floats]).

    Row-major 4x4: translation = column 3 = [m3, m7, m11]; the rotation
    is the leading 3x3 = m[0:9].  Defaults to the identity rotation +
    [0, 0, 0.3] forward offset, which MUST match the camera actually sent
    to Unity (il_common.make_depth_vehicle).
    """
    if value is None:
        return [0.0, 0.0, 0.3], [1, 0, 0, 0, 1, 0, 0, 0, 1]
    if not isinstance(value, (list, tuple)) or len(value) != 16 or \
            any(not isinstance(v, (int, float)) or isinstance(v, bool) or
                not math.isfinite(float(v)) for v in value):
        errors.append("%s must be a list of 16 finite numbers (row-major 4x4)"
                      % name)
        return [0.0, 0.0, 0.3], [1, 0, 0, 0, 1, 0, 0, 0, 1]
    m = [float(v) for v in value]
    translation = [m[3], m[7], m[11]]
    rotation = m[0:9]
    return translation, rotation


def build_params(global_cfg, errors=None):
    """Build and return the pybind Params2D object from
    ``global_cfg["hierarchical_expert"]`` (or defaults when absent).

    ``errors`` (optional list) collects validation problems instead of
    raising; the caller decides whether to fail.
    """
    if Params2D is None:  # pragma: no cover
        raise RuntimeError(
            "_il_hierarchical_expert module not built; run "
            "`catkin build il_dataset` first")
    problems = [] if errors is None else errors
    he = global_cfg.get("hierarchical_expert", {}) or {}
    p = Params2D()

    region = he.get("region", {}) or {}
    p.region_min_x = _num(region.get("min_x"), "he.region.min_x", problems,
                          -20.0, allow_zero=True)
    p.region_max_x = _num(region.get("max_x"), "he.region.max_x", problems,
                          20.0, allow_zero=True)
    p.region_min_y = _num(region.get("min_y"), "he.region.min_y", problems,
                          -20.0, allow_zero=True)
    p.region_max_y = _num(region.get("max_y"), "he.region.max_y", problems,
                          20.0, allow_zero=True)

    p.drone_radius = _num(he.get("drone_radius_m"),
                          "he.drone_radius_m", problems, 0.15)
    p.scene_safety_clearance = _num(
        he.get("scene_safety_clearance_m"),
        "he.scene_safety_clearance_m", problems, 0.5)
    p.macro_route_clearance_margin = _num(
        he.get("macro_route_clearance_margin_m"),
        "he.macro_route_clearance_margin_m", problems, 0.1,
        allow_zero=True)
    p.task_goal_tolerance = _num(
        he.get("goal_tolerance_m"), "he.goal_tolerance_m", problems, 0.4)
    p.task_episode_timeout_s = _num(
        he.get("episode_timeout_s"), "he.episode_timeout_s", problems, 0.0,
        allow_zero=True)

    # ── observation (R = range_m, THE perception range) ───────────
    obs = he.get("observation", {}) or {}
    p.obs_fov_deg = _num(obs.get("fov_deg"), "he.observation.fov_deg",
                         problems, 90.0)
    p.obs_range_m = _num(obs.get("range_m"), "he.observation.range_m",
                         problems, 5.0)
    p.obs_resolution = _num(obs.get("resolution"),
                            "he.observation.resolution", problems, 0.1)
    p.obs_ray_angular_res_deg = _num(
        obs.get("ray_angular_res_deg"),
        "he.observation.ray_angular_res_deg", problems, 0.5)
    p.obs_history_max_age_ticks = int(_num(
        obs.get("history_max_age_ticks"),
        "he.observation.history_max_age_ticks", problems, 120))

    # ── depth camera extrinsic (Unity T_BC, camera->body) ─────────
    # Single source: the same camera sent to Unity (il_common builds it
    # from depth.*).  The C++ observation builder derives the true camera
    # pose from the vehicle pose + this extrinsic.
    depth_cfg = global_cfg.get("depth", {}) or {}
    _t_bc, _r_bc = _parse_t_bc(
        depth_cfg.get("t_bc"), problems, "depth.t_bc")
    p.cam_t_bc_x, p.cam_t_bc_y, p.cam_t_bc_z = _t_bc
    p.cam_r_bc = list(_r_bc)

    # ── local planner (30 Hz) ─────────────────────────────────────
    lp = he.get("local_planner", {}) or {}
    p.lp_horizon_s = _num(lp.get("horizon_s"), "he.lp.horizon_s",
                          problems, 2.5)
    p.lp_dt = _num(lp.get("dt"), "he.lp.dt", problems, 0.1)
    p.lp_speed_samples = _num_list(
        lp.get("speed_samples"), "he.lp.speed_samples", problems,
        [0.0, 0.3, 0.6, 1.2, 1.8, 2.5])
    p.lp_lateral_ratio_samples = _num_list(
        lp.get("lateral_ratio_samples"), "he.lp.lateral_ratio_samples",
        problems, [-0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5])
    p.lp_yaw_rate_samples = _num_list(
        lp.get("yaw_rate_samples"), "he.lp.yaw_rate_samples", problems,
        [-2.0, -1.0, -0.5, -0.25, -0.15, 0.0, 0.15, 0.25, 0.5, 1.0, 2.0])
    p.lp_max_speed = _num(lp.get("max_speed"), "he.lp.max_speed",
                          problems, 3.0)
    p.lp_max_accel = _num(lp.get("max_accel"), "he.lp.max_accel",
                          problems, 2.0)
    p.lp_max_yaw_rate = _num(lp.get("max_yaw_rate"),
                             "he.lp.max_yaw_rate", problems, 2.0)
    p.lp_max_yaw_accel = _num(lp.get("max_yaw_accel"),
                              "he.lp.max_yaw_accel", problems, 4.0)
    p.lp_min_clearance = _num(lp.get("min_clearance"),
                              "he.lp.min_clearance", problems, 0.5)
    p.lp_soft_clearance_radius_m = _num(
        lp.get("soft_clearance_radius_m"),
        "he.lp.soft_clearance_radius_m", problems, 2.0)
    p.lp_clearance_discretization_margin_m = _num(
        lp.get("clearance_discretization_margin_m"),
        "he.lp.clearance_discretization_margin_m", problems, 0.05,
        allow_zero=True)
    p.lp_obstacle_reaction_time_s = _num(
        lp.get("obstacle_reaction_time_s"),
        "he.lp.obstacle_reaction_time_s", problems, 0.20)
    p.lp_control_period_s = _num(lp.get("control_period_s"),
                                 "he.lp.control_period_s", problems,
                                 1.0 / 30.0)
    p.lp_max_allowed_regress_m = _num(
        lp.get("max_allowed_regress_m"),
        "he.lp.max_allowed_regress_m", problems, 0.05,
        allow_zero=True)
    p.lp_limit_cycle_window_ticks = int(_num(
        lp.get("limit_cycle_window_ticks"),
        "he.lp.limit_cycle_window_ticks", problems, 15))
    p.lp_limit_cycle_net_progress_m = _num(
        lp.get("limit_cycle_net_progress_m"),
        "he.lp.limit_cycle_net_progress_m", problems, 0.10,
        allow_zero=True)
    p.lp_limit_cycle_min_blocked_ticks = int(_num(
        lp.get("limit_cycle_min_blocked_ticks"),
        "he.lp.limit_cycle_min_blocked_ticks", problems, 8))
    p.lp_limit_cycle_lateral_flip_count = int(_num(
        lp.get("limit_cycle_lateral_flip_count"),
        "he.lp.limit_cycle_lateral_flip_count", problems, 2))
    p.lp_turn_enter_deg = _num(lp.get("turn_enter_deg"),
                               "he.lp.turn_enter_deg", problems, 42.0)
    p.lp_turn_exit_deg = _num(lp.get("turn_exit_deg"),
                              "he.lp.turn_exit_deg", problems, 8.0)
    p.lp_turn_exit_max_yaw_rate = _num(
        lp.get("turn_exit_max_yaw_rate"),
        "he.lp.turn_exit_max_yaw_rate", problems, 0.15,
        allow_zero=True)
    p.lp_turn_k = _num(lp.get("turn_k"), "he.lp.turn_k", problems, 2.5)
    p.lp_near_goal_heading_relax_distance = _num(
        lp.get("near_goal_heading_relax_distance"),
        "he.lp.near_goal_heading_relax_distance", problems, 1.0)
    p.lp_near_goal_turn_enter_deg = _num(
        lp.get("near_goal_turn_enter_deg"),
        "he.lp.near_goal_turn_enter_deg", problems, 75.0)
    p.lp_terminal_control_distance = _num(
        lp.get("terminal_control_distance"),
        "he.lp.terminal_control_distance", problems, 1.2)
    p.lp_terminal_speed_gain = _num(
        lp.get("terminal_speed_gain"), "he.lp.terminal_speed_gain",
        problems, 1.0)
    p.lp_terminal_max_speed = _num(
        lp.get("terminal_max_speed"), "he.lp.terminal_max_speed",
        problems, 0.6)
    p.lp_terminal_max_yaw_rate = _num(
        lp.get("terminal_max_yaw_rate"), "he.lp.terminal_max_yaw_rate",
        problems, 0.5)
    p.lp_min_progress_m = _num(lp.get("min_progress_m"),
                               "he.lp.min_progress_m", problems, 0.05,
                               allow_zero=True)
    p.lp_min_progress_speed_mps = _num(
        lp.get("min_progress_speed_mps"),
        "he.lp.min_progress_speed_mps", problems, 0.03,
        allow_zero=True)
    p.lp_min_progress_epsilon_m = _num(
        lp.get("min_progress_epsilon_m"),
        "he.lp.min_progress_epsilon_m", problems, 0.01,
        allow_zero=True)
    p.lp_target_discontinuity_reset_m = _num(
        lp.get("target_discontinuity_reset_m"),
        "he.lp.target_discontinuity_reset_m", problems, 1.5)
    p.lp_nominal_clearance_m = _num(
        lp.get("nominal_clearance_m"), "he.lp.nominal_clearance_m",
        problems, 0.65)
    p.lp_risk_corridor_half_width = _num(
        lp.get("risk_corridor_half_width"),
        "he.lp.risk_corridor_half_width", problems, 1.0)
    p.lp_risk_distance_horizon_m = _num(
        lp.get("risk_distance_horizon_m"),
        "he.lp.risk_distance_horizon_m", problems, 5.0)
    p.lp_risk_ttc_horizon_s = _num(lp.get("risk_ttc_horizon_s"),
                                   "he.lp.risk_ttc_horizon_s", problems, 2.5)
    p.lp_risk_trajectory_radius_m = _num(
        lp.get("risk_trajectory_radius_m"),
        "he.lp.risk_trajectory_radius_m", problems, 1.0)
    p.lp_avoidance_active_threshold = _num(
        lp.get("avoidance_active_threshold"),
        "he.lp.avoidance_active_threshold", problems, 0.10,
        allow_zero=True)
    p.lp_brake_stop_margin_m = _num(
        lp.get("brake_stop_margin_m"), "he.lp.brake_stop_margin_m",
        problems, 0.3, allow_zero=True)
    p.lp_min_executable_prefix_s = _num(
        lp.get("min_executable_prefix_s"),
        "he.lp.min_executable_prefix_s", problems, 0.2)
    p.lp_scoring_horizon_s = _num(
        lp.get("scoring_horizon_s"), "he.lp.scoring_horizon_s",
        problems, 0.8)
    p.lp_cost_tie_tolerance = _num(
        lp.get("cost_tie_tolerance"), "he.lp.cost_tie_tolerance",
        problems, 1e-6, allow_zero=True)
    p.lp_cross_track_normalize_m = _num(
        lp.get("cross_track_normalize_m"),
        "he.lp.cross_track_normalize_m", problems, 2.0)

    cw = lp.get("cost_weights", {}) or {}
    p.cost_w_progress = _num(cw.get("progress"), "he.lp.cost.progress",
                             problems, 1.0, allow_zero=True)
    p.cost_w_clearance = _num(cw.get("clearance"), "he.lp.cost.clearance",
                              problems, 2.0, allow_zero=True)
    p.cost_w_smoothness = _num(cw.get("smoothness"),
                               "he.lp.cost.smoothness", problems, 0.5,
                               allow_zero=True)
    p.cost_w_speed_change = _num(cw.get("speed_change"),
                                 "he.lp.cost.speed_change", problems, 0.3,
                                 allow_zero=True)
    p.cost_w_yaw_rate_change = _num(cw.get("yaw_rate_change"),
                                    "he.lp.cost.yaw_rate_change", problems,
                                    0.3, allow_zero=True)
    p.cost_w_terminal_heading = _num(
        cw.get("terminal_heading"), "he.lp.cost.terminal_heading",
        problems, 1.0, allow_zero=True)
    p.cost_w_velocity_alignment = _num(
        cw.get("velocity_alignment"), "he.lp.cost.velocity_alignment",
        problems, 1.2, allow_zero=True)
    p.cost_w_cross_track = _num(cw.get("cross_track"),
                                "he.lp.cost.cross_track", problems, 0.8,
                                allow_zero=True)
    p.cost_w_obstacle_risk = _num(cw.get("obstacle_risk"),
                                  "he.lp.cost.obstacle_risk", problems, 3.0,
                                  allow_zero=True)

    # ── 5 Hz corrector (macro) ────────────────────────────────────
    mc = he.get("corrector", {}) or {}
    p.macro_local_failure_duration_s = _num(
        mc.get("local_failure_duration_s"),
        "he.corrector.local_failure_duration_s", problems, 0.4)
    p.macro_reentry_guard_ticks = int(_num(
        mc.get("reentry_guard_ticks"),
        "he.corrector.reentry_guard_ticks", problems, 30))
    p.macro_correction_enter_stable_ticks = int(_num(
        mc.get("correction_enter_stable_ticks"),
        "he.corrector.correction_enter_stable_ticks", problems, 1))
    p.macro_observable_frontier_min_distance_m = _num(
        mc.get("observable_frontier_min_distance_m"),
        "he.corrector.observable_frontier_min_distance_m", problems, 1.5)
    p.macro_observable_frontier_min_progress_m = _num(
        mc.get("observable_frontier_min_progress_m"),
        "he.corrector.observable_frontier_min_progress_m", problems, 0.5,
        allow_zero=True)
    p.macro_observable_unknown_margin_cells = int(_num(
        mc.get("observable_unknown_margin_cells"),
        "he.corrector.observable_unknown_margin_cells", problems, 3))
    p.macro_side_evidence_margin = _num(
        mc.get("side_evidence_margin"),
        "he.corrector.side_evidence_margin", problems, 0.5,
        allow_zero=True)
    p.macro_evidence_ray_step_deg = _num(
        mc.get("evidence_ray_step_deg"),
        "he.corrector.evidence_ray_step_deg", problems, 1.0)
    p.macro_min_evidence_ray_pairs = int(_num(
        mc.get("min_evidence_ray_pairs"),
        "he.corrector.min_evidence_ray_pairs", problems, 4))
    p.macro_corridor_half_width = _num(
        mc.get("corridor_half_width"),
        "he.corrector.corridor_half_width", problems, 1.5)
    p.macro_corridor_rear_tolerance_m = _num(
        mc.get("corridor_rear_tolerance_m"),
        "he.corrector.corridor_rear_tolerance_m", problems, 0.5,
        allow_zero=True)
    p.macro_local_recovery_prefix_m = _num(
        mc.get("local_recovery_prefix_m"),
        "he.corrector.local_recovery_prefix_m", problems, 0.8,
        allow_zero=True)
    p.macro_local_candidate_bearing_step_deg = _num(
        mc.get("local_candidate_bearing_step_deg"),
        "he.corrector.local_candidate_bearing_step_deg", problems, 5.0)
    p.macro_local_candidate_distance_step_m = _num(
        mc.get("local_candidate_distance_step_m"),
        "he.corrector.local_candidate_distance_step_m", problems, 0.5)
    p.macro_local_target_event_tolerance_m = _num(
        mc.get("local_target_event_tolerance_m"),
        "he.corrector.local_target_event_tolerance_m", problems, 0.05,
        allow_zero=True)
    p.macro_unknown_recovery_threshold_ticks = int(_num(
        mc.get("unknown_recovery_threshold_ticks"),
        "he.corrector.unknown_recovery_threshold_ticks", problems, 60))

    # ── target encoding protocol ──────────────────────────────────
    te = he.get("target_encoding", {}) or {}
    p.te_direction_bin_count = int(_num(
        te.get("direction_bin_count"),
        "he.target_encoding.direction_bin_count", problems, 11))
    p.te_normal_distance_reserve_m = _num(
        te.get("normal_distance_reserve_m"),
        "he.target_encoding.normal_distance_reserve_m", problems, 0.5,
        allow_zero=True)
    p.te_turn_ray_margin_deg = _num(
        te.get("turn_ray_margin_deg"),
        "he.target_encoding.turn_ray_margin_deg", problems, 10.0)

    # ── vehicle thresholds ────────────────────────────────────────
    vh = he.get("vehicle", {}) or {}
    p.vehicle_goal_stop_speed_mps = _num(
        vh.get("goal_stop_speed_mps"),
        "he.vehicle.goal_stop_speed_mps", problems, 0.2,
        allow_zero=True)
    p.vehicle_stationary_speed_mps = _num(
        vh.get("stationary_speed_mps"),
        "he.vehicle.stationary_speed_mps", problems, 0.05,
        allow_zero=True)

    # ── Cross-checks (single source of truth, never hard-coded) ───
    if p.te_direction_bin_count < 3 or p.te_direction_bin_count % 2 == 0:
        problems.append(
            "he.target_encoding.direction_bin_count must be odd and >= 3")
    if not (p.te_normal_distance_reserve_m < p.obs_range_m):
        problems.append(
            "he.target_encoding.normal_distance_reserve_m must be < "
            "he.observation.range_m")
    if p.te_direction_bin_count >= 3:
        half_span = math.radians(p.obs_fov_deg) / 2.0 - \
            math.radians(p.te_turn_ray_margin_deg)
        if half_span <= 0.0:
            problems.append(
                "he.target_encoding.turn_ray_margin_deg must be < "
                "he.observation.fov_deg / 2")
    if p.region_max_x <= p.region_min_x or p.region_max_y <= p.region_min_y:
        problems.append("he.region max bounds must exceed min bounds")
    return p


def build_scene_bounds(global_cfg):
    """Return (min_bounds, max_bounds) world XY for the grid anchor."""
    he = global_cfg.get("hierarchical_expert", {}) or {}
    region = he.get("region", {}) or {}
    return ([float(region.get("min_x", -20.0)),
             float(region.get("min_y", -20.0))],
            [float(region.get("max_x", 20.0)),
             float(region.get("max_y", 20.0))])
