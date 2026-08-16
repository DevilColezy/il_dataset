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
                       p_.obs_history_max_age_ticks);
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
    history_.reset();
    fsm_.reset(task_, tick);
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
    out.effective_target_world_valid = fsm_out.effective_target_world_valid;

    // ── 30 Hz executable label ─────────────────────────────────────
    out.target_velocity_flu_x = fsm_out.local.vx_body;
    out.target_velocity_flu_y = fsm_out.local.vy_body;
    out.target_yaw_rate = fsm_out.local.yaw_rate;
    out.intent_vx_body = fsm_out.local.intent_vx_body;
    out.intent_vy_body = fsm_out.local.intent_vy_body;
    out.intent_yaw_rate = fsm_out.local.intent_yaw_rate;

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
    out.emergency_brake = fsm_out.local.emergency_brake;
    out.immediate_avoidance = fsm_out.local.immediate_avoidance;
    out.local_limit_cycle_detected = fsm_out.local.local_limit_cycle_detected;
    out.target_bearing_error_deg = fsm_out.local.target_bearing_error_deg;
    out.consecutive_failures_30hz = fsm_out.consecutive_failures_30hz;
    out.unknown_recovery_ticks = fsm_out.unknown_recovery_ticks;

    // ── 5 Hz labels (ZOH between 5 Hz boundaries) ──────────────────
    if (out.macro_update_mask) {
        const TargetCorrectionDirective& d = fsm_.lastDirective();
        last_macro_label_valid_ = d.valid ? 1 : 0;
        last_macro_correction_type_ = targetCorrectionTypeName(d.type);
        last_macro_direction_token_ = d.direction_token;
        last_macro_direction_flu_x_ = d.decoded_direction_body.x();
        last_macro_direction_flu_y_ = d.decoded_direction_body.y();
        last_macro_direction_flu_z_ = 0.0;
        last_macro_distance_norm_ = d.normalized_distance;
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
//  Step (Flightmare path)
// ────────────────────────────────────────────────────────────────────
ExpertStepOutput HierarchicalExpert::step(
    const double pos[3], double yaw_fm, const double vel_world[3],
    double yaw_rate_fm, const std::vector<float>& depth_m, int depth_w,
    int depth_h, const double cam_pos[3], const double cam_q[4],
    double flight_z, uint64_t tick, bool collision) {
    // Coordinate adaptation (single layer): Flightmare → expert frame.
    VehicleState2D st;
    st.position = Vec2d(pos[0], pos[1]);
    st.yaw = CoordinateAdapter::flightmareYawToExpert(yaw_fm);
    st.velocity_world = Vec2d(vel_world[0], vel_world[1]);
    st.yaw_rate = yaw_rate_fm;  // left-turn positive in both frames
    flight_z_ = flight_z;

    // Depth → current FOV patch (grid-aligned to the global grid).  The
    // camera FOV/range/resolution come from Params2D only; the true camera
    // pose is derived from the vehicle pose + Params2D T_BC.
    const LocalObservation current_patch = obs_builder_.build(
        depth_m, depth_w, depth_h, cam_pos, cam_q, min_bounds_, tick);

    // Causal merge into the persistent local history (30 Hz planner input).
    history_.integrate(current_patch, tick);

    FsmInput in{task_, st, current_patch, history_.observation(), tick,
                collision};
    const FsmStepOutput fsm_out = fsm_.step(in);

    ExpertStepOutput out;
    fillOutput(out, fsm_out, tick, flight_z);
    // The 5 Hz student input needs the live body direction of the ORIGINAL
    // goal.  Rebuild it here from the world goal and the expert pose.
    {
        const Vec2d to = task_.goal - st.position;
        const double d = to.norm();
        Vec2d dir_body(1.0, 0.0);
        if (d > 1e-9) {
            dir_body = rot2(to / d, -st.yaw);
        }
        out.navigation_goal_direction_flu_x = dir_body.x();
        out.navigation_goal_direction_flu_y = dir_body.y();
        out.navigation_goal_direction_flu_z = 0.0;
        const double R = std::max(1e-9, p_.obs_range_m);
        const double reserve = std::max(0.0, p_.te_normal_distance_reserve_m);
        const double clip = std::min(d, R - reserve);
        out.navigation_goal_distance_clipped_m = clip;
        out.navigation_goal_distance_norm = clip / R;
    }
    return out;
}

// ────────────────────────────────────────────────────────────────────
//  Step (preflight path: patch already synthesized from the truth scene)
// ────────────────────────────────────────────────────────────────────
ExpertStepOutput HierarchicalExpert::stepFromPatch(
    const VehicleState2D& expert_state, const LocalObservation& current_patch,
    double flight_z, uint64_t tick, bool collision) {
    flight_z_ = flight_z;
    history_.integrate(current_patch, tick);
    FsmInput in{task_, expert_state, current_patch, history_.observation(),
                tick, collision};
    const FsmStepOutput fsm_out = fsm_.step(in);

    ExpertStepOutput out;
    fillOutput(out, fsm_out, tick, flight_z);
    {
        const Vec2d to = task_.goal - expert_state.position;
        const double d = to.norm();
        Vec2d dir_body(1.0, 0.0);
        if (d > 1e-9) {
            dir_body = rot2(to / d, -expert_state.yaw);
        }
        out.navigation_goal_direction_flu_x = dir_body.x();
        out.navigation_goal_direction_flu_y = dir_body.y();
        out.navigation_goal_direction_flu_z = 0.0;
        const double R = std::max(1e-9, p_.obs_range_m);
        const double reserve = std::max(0.0, p_.te_normal_distance_reserve_m);
        const double clip = std::min(d, R - reserve);
        out.navigation_goal_distance_clipped_m = clip;
        out.navigation_goal_distance_norm = clip / R;
    }
    return out;
}

}  // namespace expert
}  // namespace il_dataset
