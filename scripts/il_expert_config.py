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


def _bool(value, name, errors, default=None):
    """Accept a real YAML boolean (or a numeric / string flag) and return
    a Python bool.  `_num` deliberately rejects bool, so boolean knobs must
    use this helper instead of being wrapped in bool(_num(...))."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    errors.append("%s must be a boolean" % name)
    return default


def _parse_t_bc(value, errors, name="depth.t_bc"):
    """Parse a Unity T_BC 4x4 (16 floats, row-major) into
    (translation [tx, ty, tz], rotation 3x3 row-major [9 floats]).

    Row-major 4x4: translation = column 3 = [m3, m7, m11]; the rotation
    is the leading 3x3 spanning rows 0..2, which SKIPS the 4th column:
    [m0,m1,m2, m4,m5,m6, m8,m9,m10].  (m[0:9] would wrongly include the
    4th-column elements m3/m7 and drop m9/m10, turning even the identity
    matrix into a singular one.)  Defaults to the identity rotation +
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
    rotation = [m[0], m[1], m[2],
                m[4], m[5], m[6],
                m[8], m[9], m[10]]
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
        "he.scene_safety_clearance_m", problems, 0.5, allow_zero=True)
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
    p.obs_free_clear_confirmations = int(_num(
        obs.get("free_clear_confirmations"),
        "he.observation.free_clear_confirmations", problems, 3))
    p.lp_planning_history_max_age_ticks = int(_num(
        obs.get("planning_history_max_age_ticks"),
        "he.observation.planning_history_max_age_ticks", problems, 45))
    p.obs_ground_clearance_m = _num(
        obs.get("ground_clearance_m"),
        "he.observation.ground_clearance_m", problems, 0.5)

    # ── depth camera extrinsic (Unity T_BC, camera->body) ─────────
    # Single source: the same camera sent to Unity (il_common builds it
    # from depth.*).  The C++ observation builder derives the true camera
    # pose from the vehicle pose + this extrinsic.
    depth_cfg = global_cfg.get("depth", {}) or {}
    _t_bc, _r_bc = _parse_t_bc(
        depth_cfg.get("t_bc"), problems, "depth.t_bc")
    p.cam_t_bc_x, p.cam_t_bc_y, p.cam_t_bc_z = _t_bc
    p.cam_r_bc = list(_r_bc)
    # Defensive: the C++ CameraRig2D assumes cam_r_bc is a proper rotation
    # (orthonormal, det +1).  A singular / reflected matrix silently
    # collapses the 2D FOV to a degenerate line (every preflight stalls).
    # Fail loudly here instead of at runtime.
    try:
        ra, rb, rc = _r_bc[0], _r_bc[1], _r_bc[2]
        rd, re, rf = _r_bc[3], _r_bc[4], _r_bc[5]
        rg, rh, ri = _r_bc[6], _r_bc[7], _r_bc[8]
        det = (ra * (re * ri - rf * rh) - rb * (rd * ri - rf * rg) +
               rc * (rd * rh - re * rg))
        if not math.isfinite(det) or abs(det - 1.0) > 1e-6:
            problems.append(
                "depth.t_bc 3x3 rotation det must be +1 (got %.6f); "
                "camera FOV would be degenerate" % det)
    except Exception:
        pass

    # ── local planner (30 Hz) ─────────────────────────────────────
    lp = he.get("local_planner", {}) or {}
    p.lp_horizon_s = _num(lp.get("horizon_s"), "he.lp.horizon_s",
                          problems, 4.0)
    p.lp_dt = _num(lp.get("dt"), "he.lp.dt", problems, 0.1)
    p.lp_max_speed = _num(lp.get("max_speed"), "he.lp.max_speed",
                          problems, 3.0)
    p.lp_cruise_speed_mps = _num(
        lp.get("cruise_speed_mps"), "he.lp.cruise_speed_mps", problems, 2.0,
        allow_zero=True)
    # ── R29h: simplified speed law ────────────────────────────────
    # v_des = cruise · goal_decay(goal_along_ray) · yaw_decay(|ray_b|);
    # min lp_vmin_speed_mps while progressing on a clear ray; nose_clear ≤
    # handoff → hard stop (0).  Lost target / all rays blocked / out of FOV
    # → 0 via the existing hand-off branches.
    p.lp_goal_decay_range_m = _num(
        lp.get("goal_decay_range_m"), "he.lp.goal_decay_range_m",
        problems, 2.0)
    p.lp_vmin_speed_mps = _num(
        lp.get("vmin_speed_mps"), "he.lp.vmin_speed_mps", problems, 0.5)
    p.lp_yaw_decay_per_deg = _num(
        lp.get("yaw_decay_per_deg"), "he.lp.yaw_decay_per_deg",
        problems, 0.0111)
    p.lp_yaw_decay_min = _num(
        lp.get("yaw_decay_min"), "he.lp.yaw_decay_min", problems, 0.5)
    p.lp_ray_target_rel_max_deg = _num(
        lp.get("ray_target_rel_max_deg"), "he.lp.ray_target_rel_max_deg",
        problems, p.obs_fov_deg - 10.0)
    p.lp_terminal_micro_approach_m = _num(
        lp.get("terminal_micro_approach_m"),
        "he.lp.terminal_micro_approach_m", problems, 0.8,
        allow_zero=True)
    # ── R27: receding-horizon tracking (plan/track split) ─────────
    # The B-spline is re-optimised every lp_replan_interval_ticks control
    # ticks (3 = 10 Hz at 30 Hz); between replans the drone PURSUES the
    # committed trajectory (arc-length lookahead) instead of chasing a
    # freshly re-solved head every tick — kills the receding-horizon
    # head-drift accumulation.  1 disables tracking (legacy behaviour).
    p.lp_replan_interval_ticks = int(_num(
        lp.get("replan_interval_ticks"),
        "he.lp.replan_interval_ticks", problems, 3))
    p.lp_pursuit_lookahead_m = _num(
        lp.get("pursuit_lookahead_m"),
        "he.lp.pursuit_lookahead_m", problems, 0.6)
    p.lp_track_max_cross_track_m = _num(
        lp.get("track_max_cross_track_m"),
        "he.lp.track_max_cross_track_m", problems, 0.5)
    p.lp_track_min_front_m = _num(
        lp.get("track_min_front_m"),
        "he.lp.track_min_front_m", problems, 0.8)
    p.lp_max_accel = _num(lp.get("max_accel"), "he.lp.max_accel",
                          problems, 2.0)
    # The EFFECTIVE (physically achieved) horizontal acceleration of the
    # closed loop.  With the command-ramp feedforward the drone tracks the
    # lp_max_accel ramp, so this equals max_accel = 2.0.  Used for braking
    # / clearance / trajectory profiles so the planner can actually stop
    # where it thinks it can.
    p.lp_eff_accel_mps2 = _num(
        lp.get("eff_accel_mps2"), "he.lp.eff_accel_mps2", problems, 2.0)
    p.lp_max_yaw_rate = _num(lp.get("max_yaw_rate"),
                             "he.lp.max_yaw_rate", problems, 2.0)
    p.lp_max_yaw_accel = _num(lp.get("max_yaw_accel"),
                              "he.lp.max_yaw_accel", problems, 8.0)
    # R28c/R28g: local plan endpoint must stay within this bearing band of the
    # current target direction (deg).  Big lateral detours are the UPPER
    # planner's job; the local only makes small adjustments here and hands
    # back NO_SAFE_CANDIDATE otherwise.  35 = the +-35° scan band: a local
    # detour up to the band is legitimate (task 401 needs ~31° around obs7);
    # beyond it (task-33-style 40°+ spirals) it is rejected for the macro.
    p.lp_max_local_deviation_deg = _num(
        lp.get("max_local_deviation_deg"),
        "he.lp.max_local_deviation_deg", problems, 35.0)
    p.lp_preferred_local_deviation_deg = _num(
        lp.get("preferred_local_deviation_deg"),
        "he.lp.preferred_local_deviation_deg", problems, 20.0)
    # ── vertical channel (3D expert extension) ────────────────────
    p.lp_max_vz = _num(lp.get("max_vz"), "he.lp.max_vz", problems, 1.0)
    p.lp_max_v_accel = _num(lp.get("max_v_accel"),
                            "he.lp.max_v_accel", problems, 2.0)
    p.lp_vz_kp = _num(lp.get("vz_kp"), "he.lp.vz_kp", problems, 1.0)
    p.lp_z_min_m = _num(lp.get("z_min_m"), "he.lp.z_min_m", problems, 0.8)
    p.lp_z_max_m = _num(lp.get("z_max_m"), "he.lp.z_max_m", problems, 3.0)
    p.lp_vertical_clearance_m = _num(
        lp.get("vertical_clearance_m"),
        "he.lp.vertical_clearance_m", problems, 0.3)
    p.lp_min_clearance = _num(lp.get("min_clearance"),
                              "he.lp.min_clearance", problems, 0.4)
    p.lp_soft_clearance_radius_m = _num(
        lp.get("soft_clearance_radius_m"),
        "he.lp.soft_clearance_radius_m", problems, 1.0)
    p.lp_clearance_discretization_margin_m = _num(
        lp.get("clearance_discretization_margin_m"),
        "he.lp.clearance_discretization_margin_m", problems, 0.15,
        allow_zero=True)
    p.lp_obstacle_reaction_time_s = _num(
        lp.get("obstacle_reaction_time_s"),
        "he.lp.obstacle_reaction_time_s", problems, 0.20)
    # ── EGO-style optimisation B-spline (R19) ─────────────────────
    # Optimization-based local path (structure from ZJU-FAST-Lab/ego-planner
    # BsplineOptimizer + okazaki L-BFGS).  Bends the cubic B-spline around
    # observed obstacles instead of the straight-ray degeneracy of the old
    # core.  The optimised geometry is still validated with the hard
    # clearance + dynamic envelope and falls back to the straight-line
    # planner / escape-rotate / brake.
    p.ego_enabled = _bool(
        lp.get("ego_enabled"), "he.lp.ego_enabled", problems, True)
    p.ego_lambda_smooth = _num(
        lp.get("ego_lambda_smooth"), "he.lp.ego_lambda_smooth",
        problems, 0.5, allow_zero=True)
    p.ego_lambda_collision = _num(
        lp.get("ego_lambda_collision"), "he.lp.ego_lambda_collision",
        problems, 2.0, allow_zero=True)
    p.ego_lambda_feasibility = _num(
        lp.get("ego_lambda_feasibility"), "he.lp.ego_lambda_feasibility",
        problems, 0.2, allow_zero=True)
    p.ego_lambda_fitness = _num(
        lp.get("ego_lambda_fitness"), "he.lp.ego_lambda_fitness",
        problems, 0.8, allow_zero=True)
    p.ego_lambda_fov = _num(
        lp.get("ego_lambda_fov"), "he.lp.ego_lambda_fov",
        problems, 0.3, allow_zero=True)
    p.ego_clearance_m = _num(
        lp.get("ego_clearance_m"), "he.lp.ego_clearance_m",
        problems, 0.55)
    p.ego_ts = _num(lp.get("ego_ts"), "he.lp.ego_ts", problems, 0.4)
    p.ego_n_segments = int(_num(
        lp.get("ego_n_segments"), "he.lp.ego_n_segments", problems, 8))
    p.ego_max_iter = int(_num(
        lp.get("ego_max_iter"), "he.lp.ego_max_iter", problems, 60))
    # R27: temporal anchoring weight — a soft cost pulling consecutive EGO
    # replans toward the previous committed plan (receding-horizon
    # continuity).  The warm-start init is applied whenever a compatible
    # reference exists; 0 disables only the cost term.
    p.ego_lambda_ref = _num(
        lp.get("ego_lambda_ref"), "he.lp.ego_lambda_ref",
        problems, 0.3, allow_zero=True)
    p.lp_control_period_s = _num(lp.get("control_period_s"),
                                 "he.lp.control_period_s", problems,
                                 1.0 / 30.0)
    p.lp_turn_enter_deg = _num(lp.get("turn_enter_deg"),
                               "he.lp.turn_enter_deg", problems, 42.0)
    p.lp_turn_exit_deg = _num(lp.get("turn_exit_deg"),
                              "he.lp.turn_exit_deg", problems, 8.0)
    p.lp_turn_exit_max_yaw_rate = _num(
        lp.get("turn_exit_max_yaw_rate"),
        "he.lp.turn_exit_max_yaw_rate", problems, 0.15,
        allow_zero=True)
    p.lp_turn_k = _num(lp.get("turn_k"), "he.lp.turn_k", problems, 2.5)
    p.lp_yaw_smooth_alpha = _num(
        lp.get("yaw_smooth_alpha"), "he.lp.yaw_smooth_alpha",
        problems, 0.35, allow_zero=True)
    p.lp_near_goal_heading_relax_distance = _num(
        lp.get("near_goal_heading_relax_distance"),
        "he.lp.near_goal_heading_relax_distance", problems, 1.0)
    p.lp_near_goal_turn_enter_deg = _num(
        lp.get("near_goal_turn_enter_deg"),
        "he.lp.near_goal_turn_enter_deg", problems, 75.0)
    p.lp_terminal_speed_gain = _num(
        lp.get("terminal_speed_gain"), "he.lp.terminal_speed_gain",
        problems, 1.0)
    p.lp_terminal_max_speed = _num(
        lp.get("terminal_max_speed"), "he.lp.terminal_max_speed",
        problems, 0.6)
    p.lp_terminal_max_yaw_rate = _num(
        lp.get("terminal_max_yaw_rate"), "he.lp.terminal_max_yaw_rate",
        problems, 0.5)
    p.lp_min_progress_speed_mps = _num(
        lp.get("min_progress_speed_mps"),
        "he.lp.min_progress_speed_mps", problems, 0.03,
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
    p.lp_brake_stop_margin_m = _num(
        lp.get("brake_stop_margin_m"), "he.lp.brake_stop_margin_m",
        problems, 0.3, allow_zero=True)
    # ── 5 Hz corrector (macro) ────────────────────────────────────
    mc = he.get("corrector", {}) or {}
    p.macro_observable_frontier_min_distance_m = _num(
        mc.get("observable_frontier_min_distance_m"),
        "he.corrector.observable_frontier_min_distance_m", problems, 1.5)
    p.macro_observable_frontier_min_progress_m = _num(
        mc.get("observable_frontier_min_progress_m"),
        "he.corrector.observable_frontier_min_progress_m", problems, 0.5,
        allow_zero=True)
    # R29k: only re-acquire the original goal via search rotation when its
    # bearing is traversable (continuous FREE run >= this range).  Behind a
    # blocker → keep the locked bypass side.
    p.macro_goal_direction_min_range_m = _num(
        mc.get("goal_direction_min_range_m"),
        "he.corrector.goal_direction_min_range_m", problems, 2.0)
    # R29l: a fresh 5 Hz waypoint candidate must beat the held waypoint by
    # this along-goal progress before it is adopted (anti-jitter margin).
    p.macro_waypoint_update_along_margin = _num(
        mc.get("waypoint_update_along_margin"),
        "he.corrector.waypoint_update_along_margin", problems, 0.3,
        allow_zero=True)
    # R29m: SEARCH_ROTATION_TOWARD_ORIGINAL_GOAL cooldown (5 Hz updates);
    # prevents depth-evidence flips from oscillating the turn direction.
    p.macro_search_rotation_cooldown_5hz = int(_num(
        mc.get("search_rotation_cooldown_5hz"),
        "he.corrector.search_rotation_cooldown_5hz", problems, 12))
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
    p.macro_blocking_lateral_span_ratio = _num(
        mc.get("blocking_lateral_span_ratio"),
        "he.corrector.blocking_lateral_span_ratio", problems, 0.5,
        allow_zero=True)
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
    p.macro_guide_horizon_m = _num(
        mc.get("guide_horizon_m"), "he.corrector.guide_horizon_m",
        problems, 4.8)
    p.macro_local_target_event_tolerance_m = _num(
        mc.get("local_target_event_tolerance_m"),
        "he.corrector.local_target_event_tolerance_m", problems, 0.05,
        allow_zero=True)
    p.macro_takeover_confirm_ticks_30hz = int(_num(
        mc.get("takeover_confirm_ticks_30hz"),
        "he.corrector.takeover_confirm_ticks_30hz", problems, 12))
    p.macro_unknown_recovery_threshold_ticks = int(_num(
        mc.get("unknown_recovery_threshold_ticks"),
        "he.corrector.unknown_recovery_threshold_ticks", problems, 60))
    p.macro_brake_confirm_ticks_5hz = int(_num(
        mc.get("brake_confirm_ticks_5hz"),
        "he.corrector.brake_confirm_ticks_5hz", problems, 2))
    p.macro_waypoint_reached_tolerance_m = _num(
        mc.get("waypoint_reached_tolerance_m"),
        "he.corrector.waypoint_reached_tolerance_m", problems, 0.5,
        allow_zero=True)
    p.macro_terminal_capture_radius_m = _num(
        mc.get("terminal_capture_radius_m"),
        "he.corrector.terminal_capture_radius_m", problems, 1.0,
        allow_zero=True)
    p.macro_limit_cycle_goal_progress_m = _num(
        mc.get("limit_cycle_goal_progress_m"),
        "he.corrector.limit_cycle_goal_progress_m", problems, 0.1,
        allow_zero=True)
    p.macro_limit_cycle_window_5hz = int(_num(
        mc.get("limit_cycle_window_5hz"),
        "he.corrector.limit_cycle_window_5hz", problems, 15))

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
