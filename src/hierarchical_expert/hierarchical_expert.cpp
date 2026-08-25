#include "il_dataset/hierarchical_expert/hierarchical_expert.hpp"

#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"
#include "il_dataset/hierarchical_expert/effective_target_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

namespace {

/// The FULL new-architecture expert state (hierarchical_mode).
std::string mapHierarchicalMode(const FsmStepOutput& out) {
    switch (out.state) {
        case FsmState::GOAL_REACHED:
            return "goal_capture";
        case FsmState::COLLISION:
        case FsmState::TASK_INVALID:
        case FsmState::TIMEOUT:
            return "blocked";
        default:
            break;
    }
    // A live 5 Hz correction directive takes precedence: it is the
    // semantically most informative state (the 30 Hz layer may be turning
    // or translating toward the corrected target underneath it).
    if (out.target_correction_active) {
        switch (static_cast<TargetCorrectionType>(out.target_correction_type)) {
            case TargetCorrectionType::TURN_LEFT:
                return "macro_turn_left";
            case TargetCorrectionType::TURN_RIGHT:
                return "macro_turn_right";
            case TargetCorrectionType::NORMAL_CORRECTION:
                return "macro_normal";
            default:
                break;  // PASS_THROUGH with active flag is transient
        }
    }
    if (out.local.turn_mode ||
        out.local.planner_status == PlannerStatus::TURNING) {
        return "turn_to_target";
    }
    if (out.local.planner_status == PlannerStatus::TERMINAL_SETTLING ||
        out.local.local_target_distance <= 0.0) {
        return "goal_capture";
    }
    if (out.local.planner_status == PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE ||
        out.local.planner_status == PlannerStatus::EMERGENCY_BRAKE ||
        out.local.planner_status == PlannerStatus::NO_SAFE_CANDIDATE ||
        out.local.planner_status == PlannerStatus::STALLED_WITHOUT_PROGRESS ||
        out.local.planner_status == PlannerStatus::NO_TARGET) {
        return "blocked";
    }
    if (out.local.avoidance_active || out.local.local_corridor_blocked ||
        out.local.immediate_avoidance) {
        return "local_avoidance";
    }
    return "direct";
}

}  // namespace

// ────────────────────────────────────────────────────────────────────
//  Configuration / lifecycle
// ────────────────────────────────────────────────────────────────────
void HierarchicalExpert::configure(const Params2D& p, const Vec2d& min_bounds,
                                   const Vec2d& max_bounds) {
    p_ = p;
    min_bounds_ = min_bounds;
    max_bounds_ = max_bounds;
    history_.configure(min_bounds_, max_bounds_, p_.obs_resolution,
                       p_.obs_history_max_age_ticks,
                       p_.obs_free_clear_confirmations);
    // The FSM is stateless until the first resetTask; reconfigure its
    // parameter copy.
    fsm_ = HierarchicalExpertFsm(p_);
    obs_builder_ = Flightmare2DObservation(p_);
    configured_ = true;
}

void HierarchicalExpert::resetTask(const Vec2d& start, const Vec2d& goal,
                                   double initial_yaw_fm, uint64_t tick,
                                   double flight_z) {
    task_.start = start;
    task_.goal = goal;
    task_.initial_yaw = CoordinateAdapter::flightmareYawToExpert(initial_yaw_fm);
    task_.valid = true;
    flight_z_ = flight_z;
    // ── 3D extension: canonical 3D navigation task (mission altitude +
    //    the configured flight band). ───────────────────────────────
    navigation_task_.start = HorizontalProjection::lift(start, flight_z);
    navigation_task_.goal = HorizontalProjection::lift(goal, flight_z);
    navigation_task_.initial_yaw = task_.initial_yaw;
    navigation_task_.z_min = p_.lp_z_min_m;
    navigation_task_.z_max = p_.lp_z_max_m;
    navigation_task_.task_id = task_.task_id;
    navigation_task_.scene_id = task_.scene_id;
    navigation_task_.seed = task_.seed;
    navigation_task_.valid = true;
    history_.reset();
    fsm_.reset(task_, tick);
    // Re-arm the vertical command-ramp base for the new task.
    last_vz_command_ = 0.0;
    // Re-arm the ZOH mirror with a fresh PASS_THROUGH block.
    last_macro_label_valid_ = 1;
    last_macro_correction_type_ = "PASS_THROUGH";
    last_macro_direction_token_ = -1;
    last_macro_direction_flu_x_ = 1.0;
    last_macro_direction_flu_y_ = 0.0;
    last_macro_direction_flu_z_ = 0.0;
    last_macro_distance_norm_ = 0.0;
    last_macro_param_valid_ = 0;
}

void HierarchicalExpert::acceptNewGoal(const Vec2d& goal, uint64_t tick) {
    task_.goal = goal;
    fsm_.acceptNewGoal(goal, tick);
}

// ────────────────────────────────────────────────────────────────────
//  Output filling (single place that maps FsmStepOutput → CSV fields)
// ────────────────────────────────────────────────────────────────────
void HierarchicalExpert::fillOutput(ExpertStepOutput& out,
                                    const FsmStepOutput& fsm_out,
                                    uint64_t tick, double flight_z) {
    out.tick = tick;
    out.macro_update_mask = (tick % 6 == 0);
    out.macro_tick_ran = fsm_out.macro_tick_ran;
    out.fsm_state = fsmStateName(fsm_out.state);
    out.fsm_prev_state = fsmStateName(fsm_out.prev_state);
    out.terminal = fsm_out.state == FsmState::GOAL_REACHED ||
                   fsm_out.state == FsmState::COLLISION ||
                   fsm_out.state == FsmState::TIMEOUT ||
                   fsm_out.state == FsmState::TASK_INVALID;

    // ── 30 Hz effective target (live re-expressed every tick) ─────
    out.goal_direction_flu_x = fsm_out.target_direction_x_body;
    out.goal_direction_flu_y = fsm_out.target_direction_y_body;
    out.goal_direction_flu_z = 0.0;
    out.goal_distance_norm = fsm_out.target_distance_normalized;
    out.goal_distance_clipped_m =
        fsm_out.target_distance_normalized * std::max(1e-9, p_.obs_range_m);
    out.effective_target_source = fsm_out.target_correction_type_name;
    out.target_correction_active = fsm_out.target_correction_active;
    out.directive_terminal_stop = fsm_out.directive_terminal_stop;
    // Quantized token of the LIVE effective direction (diagnostic only).
    const EffectiveTargetAdapter adapter(p_);
    out.effective_direction_token =
        fsm_out.target_distance_normalized >= 1.0 - 1e-9
            ? fsm_out.target_direction_token
            : adapter.quantizeBearing(std::atan2(
                  fsm_out.target_direction_y_body,
                  fsm_out.target_direction_x_body));
    out.effective_target_world_x = fsm_out.effective_target_x;
    out.effective_target_world_y = fsm_out.effective_target_y;
    out.effective_target_world_z = fsm_out.local_target.z;
    out.effective_target_world_valid = fsm_out.effective_target_world_valid;

    // ── 30 Hz executable label.  The planner result is PLANAR; the final
    //    3D BODY/FLU command (including vz) is composed by
    //    CommandComposer3D in step()/stepFromPatch() and overrides these
    //    horizontal placeholders. ─────────────────────────────────────
    out.target_velocity_flu_x = fsm_out.local.vx_body;
    out.target_velocity_flu_y = fsm_out.local.vy_body;
    out.target_velocity_flu_z = 0.0;
    out.target_yaw_rate = fsm_out.local.yaw_rate;
    out.intent_vx_body = fsm_out.local.intent_vx_body;
    out.intent_vy_body = fsm_out.local.intent_vy_body;
    out.intent_yaw_rate = fsm_out.local.intent_yaw_rate;
    out.intent_vz_body = 0.0;

    // ── 30 Hz diagnostics ──────────────────────────────────────────
    out.hierarchical_mode = mapHierarchicalMode(fsm_out);
    out.planner_status = plannerStatusName(fsm_out.local.planner_status);
    out.failure_reason = failureReasonName(fsm_out.local.failure_reason);
    out.selected_output_speed_mps = fsm_out.local.selected_output_speed_mps;
    out.local_target_distance_m = fsm_out.local.local_target_distance;
    out.min_observed_clearance_m = fsm_out.local.min_observed_clearance;
    out.obstacle_risk_cost = fsm_out.local.obstacle_risk_cost;
    out.avoidance_active = fsm_out.local.avoidance_active;
    out.local_corridor_blocked = fsm_out.local.local_corridor_blocked;
    out.risk_corridor_near_obstacle =
        fsm_out.local.risk_corridor_near_obstacle;
    // R25 structured corridor diagnostics (never student inputs).
    out.corridor_block_reason =
        corridorBlockReasonName(fsm_out.local.corridor_block_reason);
    out.corridor_block_source =
        fsm_out.local.corridor_block_reason ==
                CorridorBlockReason::CURRENT_OCCUPIED
            ? "current"
            : (fsm_out.local.corridor_block_reason ==
                       CorridorBlockReason::HISTORY_OCCUPIED
                   ? "history"
                   : "none");
    out.first_blocking_distance_m =
        fsm_out.local.first_blocking_obstacle_distance;
    out.first_block_x = fsm_out.local.first_block_x;
    out.first_block_y = fsm_out.local.first_block_y;
    out.first_block_age_ticks = fsm_out.local.first_block_age_ticks;
    out.emergency_brake = fsm_out.local.emergency_brake;
    out.immediate_avoidance = fsm_out.local.immediate_avoidance;
    out.local_limit_cycle_detected = fsm_out.local.local_limit_cycle_detected;
    out.target_bearing_error_deg = fsm_out.local.target_bearing_error_deg;
    out.consecutive_failures_30hz = fsm_out.consecutive_failures_30hz;
    out.unknown_recovery_ticks = fsm_out.unknown_recovery_ticks;
    // ── 30 Hz planned trajectory (diagnostic; world XY, downsampled) ──
    const PlanarTrajectory& traj = fsm_out.local.selected;
    out.plan_valid = traj.valid && traj.points.size() >= 2;
    out.plan_terminal = fsm_out.local.plan_terminal;
    out.plan_end_speed_mps = fsm_out.local.plan_end_speed_mps;
    out.plan_executed_speed_mps = fsm_out.local.plan_executed_speed_mps;
    out.plan_points_x.clear();
    out.plan_points_y.clear();
    if (out.plan_valid) {
        constexpr size_t kMaxPlanPts = 16;
        const size_t n = traj.points.size();
        const size_t stride = std::max<size_t>(1, (n + kMaxPlanPts - 1) /
                                                      kMaxPlanPts);
        for (size_t i = 0; i < n; i += stride) {
            out.plan_points_x.push_back(traj.points[i].x());
            out.plan_points_y.push_back(traj.points[i].y());
        }
        // Always include the exact endpoint.
        if ((n - 1) % stride != 0) {
            out.plan_points_x.push_back(traj.points.back().x());
            out.plan_points_y.push_back(traj.points.back().y());
        }
    }
    // ── 30 Hz candidate-rejection breakdown (diagnostic) ───────────
    out.reject_not_known_free = fsm_out.local.reject_not_known_free;
    out.reject_outside_current_fov =
        fsm_out.local.reject_outside_current_fov;
    out.reject_observed_clearance_too_small =
        fsm_out.local.reject_observed_clearance_too_small;
    out.reject_no_progress = fsm_out.local.reject_no_progress;
    out.reject_insufficient_braking_clearance =
        fsm_out.local.reject_insufficient_braking_clearance;
    out.reject_other = fsm_out.local.reject_other;

    // ── 5 Hz labels (ZOH between 5 Hz boundaries) ──────────────────
    // The PASS/NORMAL/TURN classification is unchanged, but the NORMAL /
    // TURN numeric labels are NOT the directive-creation-time parameters:
    // every 5 Hz boundary records the LIVE Effective Target — the
    // world-latched target (NORMAL world point / TURN world direction)
    // re-projected into the current body frame at the CURRENT pose
    // (fsm_out.target_direction_*_body / target_distance_normalized are
    // last_encoded_ of this very tick).  The NORMAL direction token is
    // RE-QUANTIZED from that live body direction; TURN_* keep the fixed
    // class tokens (0 / N+1); PASS_THROUGH stays -1.
    if (out.macro_update_mask) {
        const TargetCorrectionDirective& d = fsm_.lastDirective();
        last_macro_label_valid_ = d.valid ? 1 : 0;
        last_macro_correction_type_ = targetCorrectionTypeName(d.type);
        const double live_x = fsm_out.target_direction_x_body;
        const double live_y = fsm_out.target_direction_y_body;
        last_macro_direction_flu_x_ = live_x;
        last_macro_direction_flu_y_ = live_y;
        last_macro_direction_flu_z_ = 0.0;
        last_macro_distance_norm_ = fsm_out.target_distance_normalized;
        if (d.type == TargetCorrectionType::NORMAL_CORRECTION) {
            last_macro_direction_token_ =
                adapter.quantizeBearing(std::atan2(live_y, live_x));
        } else {
            last_macro_direction_token_ = d.direction_token;
        }
        last_macro_param_valid_ =
            (d.type == TargetCorrectionType::NORMAL_CORRECTION ||
             d.type == TargetCorrectionType::TURN_LEFT ||
             d.type == TargetCorrectionType::TURN_RIGHT)
                ? 1
                : 0;
    }
    out.macro_label_valid = last_macro_label_valid_;
    out.macro_correction_type = last_macro_correction_type_;
    out.macro_direction_token = last_macro_direction_token_;
    out.macro_direction_flu_x = last_macro_direction_flu_x_;
    out.macro_direction_flu_y = last_macro_direction_flu_y_;
    out.macro_direction_flu_z = last_macro_direction_flu_z_;
    out.macro_distance_norm = last_macro_distance_norm_;
    out.macro_param_valid = last_macro_param_valid_;

    // ── 5 Hz student input: original navigation goal (live) ────────
    // NOTE: the live body direction of the ORIGINAL goal is recomputed in
    // step()/stepFromPatch() below from the world goal + live pose (the
    // effective target may be a NORMAL/TURN correction, never a substitute
    // for the original goal).  The world goal is exported here as a
    // privileged diagnostic.
    out.original_navigation_goal_world_x = fsm_out.original_goal.x();
    out.original_navigation_goal_world_y = fsm_out.original_goal.y();
    out.original_navigation_goal_world_z = flight_z;

    // Diagnostics.
    out.correction_enter_event = fsm_out.correction_enter_event;
    out.correction_exit_event = fsm_out.correction_exit_event;
    out.correction_update_event = fsm_out.correction_update_event;
    out.observability_reason = fsm_out.observability_reason;
    out.observability_goal_inside_fov = fsm_out.observability_goal_inside_fov;
    out.observability_direct_corridor_blocked =
        fsm_out.observability_direct_corridor_blocked;
    out.observability_left_bypass_visible =
        fsm_out.observability_left_bypass_visible;
    out.observability_right_bypass_visible =
        fsm_out.observability_right_bypass_visible;
    out.observability_local_avoidance_observable =
        fsm_out.observability_local_avoidance_observable;
    out.directive_update_event = fsm_out.directive_update_event;
    out.mission_revision = fsm_out.mission_revision;
    out.reentry_guard_ticks = fsm_out.reentry_guard;
    out.obstacle_first_observed_event = obstacleFirstObservedEvent();
}

// ────────────────────────────────────────────────────────────────────
//  3D goal directions (3D expert extension)
// ────────────────────────────────────────────────────────────────────
void HierarchicalExpert::apply3DGoalDirections(ExpertStepOutput& out,
                                               const PlanarState& st,
                                               double z, double flight_z) {
    // ── 3D effective-target direction (30 Hz student input).  The
    //    effective target XY comes from the adapter; its altitude is the
    //    mission z for PASS/NORMAL and the live state z for TURN (pure
    //    rotation).  The 3D unit direction is projected into the FLU body
    //    frame (forward / left / up), so goal_direction_flu_z reflects the
    //    real vertical component. ────────────────────────────────────
    {
        double ex = out.effective_target_world_x;
        double ey = out.effective_target_world_y;
        if (!out.effective_target_world_valid) {
            ex = task_.goal.x();
            ey = task_.goal.y();
        }
        const double ez = out.effective_target_world_z;
        const Vec2d d2(ex - st.position.x(), ey - st.position.y());
        const double dz = ez - z;
        const double d3 = std::sqrt(d2.squaredNorm() + dz * dz);
        out.goal_distance_raw_m = d3;  // R29r: raw unclipped 3D range
        if (d3 > 1e-9) {
            const Vec2d xy = d2 / d3;  // normalized by the 3D distance
            const double dn = dz / d3;
            const Vec2d body = rot2(xy, -st.yaw);
            out.goal_direction_flu_x = body.x();
            out.goal_direction_flu_y = body.y();
            out.goal_direction_flu_z = dn;
        } else {
            out.goal_direction_flu_x = 1.0;
            out.goal_direction_flu_y = 0.0;
            out.goal_direction_flu_z = 0.0;
        }
        // The 5 Hz macro direction is the LIVE effective-target direction
        // at the 5 Hz boundary — record the 3D version (also into the ZOH
        // mirror so non-macro rows stay consistent).
        if (out.macro_update_mask) {
            last_macro_direction_flu_x_ = out.goal_direction_flu_x;
            last_macro_direction_flu_y_ = out.goal_direction_flu_y;
            last_macro_direction_flu_z_ = out.goal_direction_flu_z;
        }
    }
    // ── 3D original-goal direction + distance (5 Hz student input). ──
    {
        const Vec2d to2 = task_.goal - st.position;
        const double dz = flight_z - z;
        const double d3 = std::sqrt(to2.squaredNorm() + dz * dz);
        out.navigation_goal_distance_raw_m = d3;  // R29r: raw unclipped range
        Vec2d body(1.0, 0.0);
        double dn = 0.0;
        if (d3 > 1e-9) {
            const Vec2d xy = to2 / d3;
            dn = dz / d3;
            body = rot2(xy, -st.yaw);
        }
        out.navigation_goal_direction_flu_x = body.x();
        out.navigation_goal_direction_flu_y = body.y();
        out.navigation_goal_direction_flu_z = dn;
        const double R = std::max(1e-9, p_.obs_range_m);
        const double reserve = std::max(0.0, p_.te_normal_distance_reserve_m);
        const double clip = std::min(d3, R - reserve);
        out.navigation_goal_distance_clipped_m = clip;
        out.navigation_goal_distance_norm = clip / R;
    }

    // Canonical terminal label.  The final committed row previously kept
    // the residual within-goal-tolerance distance (for example 0.0385),
    // contradicting the public contract and teaching the student that a
    // reached goal still has non-zero range.
    if (out.fsm_state == "GOAL_REACHED") {
        out.goal_direction_flu_x = 1.0;
        out.goal_direction_flu_y = 0.0;
        out.goal_direction_flu_z = 0.0;
        out.goal_distance_clipped_m = 0.0;
        out.goal_distance_norm = 0.0;
        out.goal_distance_raw_m = 0.0;
        out.effective_direction_token =
            EffectiveTargetAdapter(p_).quantizeBearing(0.0);

        out.navigation_goal_direction_flu_x = 1.0;
        out.navigation_goal_direction_flu_y = 0.0;
        out.navigation_goal_direction_flu_z = 0.0;
        out.navigation_goal_distance_clipped_m = 0.0;
        out.navigation_goal_distance_norm = 0.0;
        out.navigation_goal_distance_raw_m = 0.0;
    }
}

// ────────────────────────────────────────────────────────────────────
//  Step (Flightmare path)
// ────────────────────────────────────────────────────────────────────
ExpertStepOutput HierarchicalExpert::step(
    const double pos[3], double yaw_fm, const double vel_world[3],
    double yaw_rate_fm, const std::vector<float>& depth_m, int depth_w,
    int depth_h, const double cam_pos[3], const double cam_q[4],
    double flight_z, uint64_t tick, bool collision) {
    // ── 3D extension: build the canonical 3D state, then project the
    //    horizontal part for the planar expert layers.  Coordinate
    //    adaptation (Flightmare → expert frame) lives in CoordinateAdapter.
    VehicleState3D st3;
    st3.position = Vec3d(pos[0], pos[1], pos[2]);
    st3.velocity_world = Vec3d(vel_world[0], vel_world[1], vel_world[2]);
    st3.yaw = CoordinateAdapter::flightmareYawToExpert(yaw_fm);
    st3.yaw_rate = yaw_rate_fm;  // left-turn positive in both frames
    st3.pitch = 0.0;
    flight_z_ = flight_z;
    const PlanarState st = HorizontalProjection::state(st3);

    // Depth → current FOV patch (grid-aligned to the global grid).  The
    // camera FOV/range/resolution come from Params2D only; the true camera
    // pose is derived from the vehicle pose + Params2D T_BC.
    const LocalObservation current_patch = obs_builder_.build(
        depth_m, depth_w, depth_h, cam_pos, cam_q, min_bounds_, tick);

    // Causal merge into the persistent local history (30 Hz planner input).
    history_.integrate(current_patch, tick);

    FsmInput in{task_, st, current_patch, history_.observation(), tick,
                collision, flight_z, st3.position.z(),
                st3.velocity_world.z()};
    const FsmStepOutput fsm_out = fsm_.step(in);

    ExpertStepOutput out;
    fillOutput(out, fsm_out, tick, flight_z);
    // ── 3D extension: live 3D goal directions (effective target + the
    //    original goal) projected into the FLU body frame. ──────────
    apply3DGoalDirections(out, st, st3.position.z(), flight_z);

    // ── 3D extension: compose the FINAL BODY/FLU command.  The planar
    //    planner emits horizontal vx/vy/yaw_rate; the VerticalController
    //    regulates the altitude toward the effective target z; the
    //    CommandComposer3D merges them (no XY/Z frame mixing). ───────
    {
        VerticalController vc(p_);
        const VerticalCommand vcmd = vc.compute(
            st3, fsm_out.local_target.z, p_.lp_horizon_s, p_.lp_dt,
            last_vz_command_);
        CommandComposer3D composer(p_);
        const VelocityCommand3D cmd = composer.compose(fsm_out.local, vcmd);
        out.target_velocity_flu_x = cmd.vx_body;
        out.target_velocity_flu_y = cmd.vy_body;
        out.target_velocity_flu_z = cmd.vz_body;
        out.target_yaw_rate = cmd.yaw_rate;
        out.intent_vz_body = vcmd.intent_vz_body;
        // Remember the executable vz for the next tick's command ramp.
        last_vz_command_ = vcmd.vz_body;
    }
    return out;
}

// ────────────────────────────────────────────────────────────────────
//  Step (preflight path: patch already synthesized from the truth scene)
// ────────────────────────────────────────────────────────────────────
ExpertStepOutput HierarchicalExpert::stepFromPatch(
    const VehicleState3D& expert_state, const LocalObservation& current_patch,
    double flight_z, uint64_t tick, bool collision) {
    flight_z_ = flight_z;
    history_.integrate(current_patch, tick);

    const PlanarState st = HorizontalProjection::state(expert_state);
    FsmInput in{task_, st, current_patch, history_.observation(), tick,
                collision, flight_z, expert_state.position.z(),
                expert_state.velocity_world.z()};
    const FsmStepOutput fsm_out = fsm_.step(in);

    ExpertStepOutput out;
    fillOutput(out, fsm_out, tick, flight_z);
    // ── 3D extension: live 3D goal directions (preflight path). ─────
    apply3DGoalDirections(out, st, expert_state.position.z(), flight_z);

    // ── 3D extension: compose the FINAL BODY/FLU command (preflight). ─
    {
        VerticalController vc(p_);
        const VerticalCommand vcmd = vc.compute(
            expert_state, fsm_out.local_target.z, p_.lp_horizon_s, p_.lp_dt,
            last_vz_command_);
        CommandComposer3D composer(p_);
        const VelocityCommand3D cmd = composer.compose(fsm_out.local, vcmd);
        out.target_velocity_flu_x = cmd.vx_body;
        out.target_velocity_flu_y = cmd.vy_body;
        out.target_velocity_flu_z = cmd.vz_body;
        out.target_yaw_rate = cmd.yaw_rate;
        out.intent_vz_body = vcmd.intent_vz_body;
        // Remember the executable vz for the next tick's command ramp.
        last_vz_command_ = vcmd.vz_body;
    }
    return out;
}

}  // namespace expert
}  // namespace il_dataset
