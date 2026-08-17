#include "il_dataset/hierarchical_expert/hierarchical_expert_fsm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

// ────────────────────────────────────────────────────────────────────
//  Construction / reset
// ────────────────────────────────────────────────────────────────────
HierarchicalExpertFsm::HierarchicalExpertFsm(const Params2D& p)
    : p_(p), local_planner_(p), corrector_(p), adapter_(p) {}

void HierarchicalExpertFsm::reset(const Task2D& task, uint64_t tick) {
    state_ = FsmState::DIRECT_LOCAL;
    tick_ = tick;
    reset_tick_ = tick;
    consecutive_failures_ = 0;
    failure_start_tick_ = 0;
    unknown_recovery_ticks_ = 0;
    unknown_recovery_episode_count_ = 0;
    goal_revision_ = 0;
    mission_revision_ = 0;
    reentry_guard_ = 0;
    macro_tick_event_ = 0;
    corrector_.reset();
    // Initial directive: PASS_THROUGH toward the task goal, event 0.
    directive_ = TargetCorrectionDirective{};
    directive_.valid = true;
    directive_.reason = "PASS_THROUGH";
    last_encoded_ = EncodedTargetInput{};
    directive_updated_ = false;
    last_delivered_event_ = 0;
    last_original_goal_ = task.goal;
    local_planner_.reset();
}

// ────────────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────────────
bool HierarchicalExpertFsm::isTerminal(FsmState s) const {
    return s == FsmState::GOAL_REACHED || s == FsmState::COLLISION ||
           s == FsmState::TIMEOUT || s == FsmState::TASK_INVALID;
}

void HierarchicalExpertFsm::transition(FsmStepOutput& out, FsmState next,
                                       const std::string& reason) {
    if (next == state_) return;
    (void)reason;
    out.prev_state = state_;
    state_ = next;
}

LocalTarget HierarchicalExpertFsm::makeLocalTarget(
    const EncodedTargetInput& encoded) const {
    LocalTarget t;
    t.position = encoded.effective_target_world;
    t.valid = encoded.valid && encoded.effective_target_world_valid;
    t.update_event = corrector_.directiveUpdateEvent();
    t.mission_revision = mission_revision_;
    t.normalized_distance = encoded.normalized_distance;
    // ── 3D extension: target altitude (mission z; TURN keeps state z). ─
    t.z = encoded.z;
    return t;
}

bool HierarchicalExpertFsm::goalReached(const FsmInput& in) const {
    // ── 3D extension: the goal tolerance is now judged on the 3D distance
    //    (horizontal + altitude).  The vertical channel is regulated to
    //    the mission altitude by the planner, so the drone must also be
    //    near the goal height before the episode is committed. ─────────
    const double dx = in.state.position.x() - in.task.goal.x();
    const double dy = in.state.position.y() - in.task.goal.y();
    const double dz = in.state.z - in.goal_z;
    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double v = in.state.velocity_world.norm();
    const double vz = std::fabs(in.state.vz_world);
    const double yr = std::fabs(in.state.yaw_rate);
    return d <= p_.task_goal_tolerance &&
           v < p_.vehicle_goal_stop_speed_mps &&
           vz < p_.vehicle_goal_stop_speed_mps &&
           yr <= p_.lp_turn_exit_max_yaw_rate;
}

void HierarchicalExpertFsm::updateFailureBookkeeping(
    const FsmStepOutput& out, const FsmInput& in) {
    // v9: DIAGNOSTIC ONLY — never read by the 5 Hz corrector.
    const bool blocked =
        out.local.failure_reason == FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE;
    if (out.local.success || out.local.turn_mode) {
        consecutive_failures_ = 0;
        failure_start_tick_ = 0;
        unknown_recovery_ticks_ = 0;
    } else if (blocked) {
        if (consecutive_failures_ == 0) failure_start_tick_ = in.tick;
        ++consecutive_failures_;
        unknown_recovery_ticks_ = 0;
    } else if (out.local.failure_reason == FailureReason::NO_SAFE_CANDIDATE) {
        consecutive_failures_ = 0;
        failure_start_tick_ = 0;
        if (unknown_recovery_ticks_ == 0) ++unknown_recovery_episode_count_;
        ++unknown_recovery_ticks_;
    } else {
        consecutive_failures_ = 0;
        failure_start_tick_ = 0;
        unknown_recovery_ticks_ = 0;
    }
}

void HierarchicalExpertFsm::fillObservability(FsmStepOutput& out) const {
    out.target_correction_type = static_cast<uint8_t>(directive_.type);
    out.target_correction_type_name =
        targetCorrectionTypeName(directive_.type);
    out.target_correction_active = corrector_.correctionActive();
    out.target_direction_token = directive_.direction_token;
    out.target_direction_x_body = last_encoded_.direction_body.x();
    out.target_direction_y_body = last_encoded_.direction_body.y();
    out.target_distance_normalized = last_encoded_.normalized_distance;
    out.effective_target_x = last_encoded_.effective_target_world.x();
    out.effective_target_y = last_encoded_.effective_target_world.y();
    out.effective_target_world_valid =
        last_encoded_.effective_target_world_valid;
    out.original_goal = last_original_goal_;
    const AvoidanceObservability& o = corrector_.lastObservability();
    out.observability_goal_inside_fov = o.goal_inside_fov;
    out.observability_direct_corridor_blocked = o.direct_corridor_blocked;
    out.observability_blocker_observed = o.blocker_observed;
    out.observability_left_bypass_visible = o.left_bypass_observable;
    out.observability_right_bypass_visible = o.right_bypass_observable;
    out.observability_local_avoidance_observable =
        o.local_avoidance_observable;
    out.observability_fov_boundary_truncated = o.fov_boundary_truncated;
    out.observability_unknown_occluded = o.unknown_occluded;
    out.observability_reason = o.reason;
    out.observability_left_score = o.left_score;
    out.observability_right_score = o.right_score;
    out.correction_enter_event = corrector_.correctionEnterEvent();
    out.correction_exit_event = corrector_.correctionExitEvent();
    out.correction_update_event = corrector_.correctionUpdateEvent();
    out.side = corrector_.lockedSide();
    out.directive_update_event = corrector_.directiveUpdateEvent();
    out.mission_revision = mission_revision_;
    out.consecutive_failures_30hz = consecutive_failures_;
    out.unknown_recovery_ticks = unknown_recovery_ticks_;
    out.unknown_recovery_active =
        static_cast<int>(unknown_recovery_ticks_) >=
        p_.macro_unknown_recovery_threshold_ticks;
    out.unknown_recovery_episode_count = unknown_recovery_episode_count_;
    out.macro_tick_event = macro_tick_event_;
    out.reentry_guard = reentry_guard_;
    if (!out.local_target.valid) {
        out.local_target =
            makeLocalTarget(last_encoded_);
    }
}

void HierarchicalExpertFsm::forceCollision(FsmStepOutput& out) {
    if (isTerminal(state_)) return;
    out.prev_state = state_;
    out.state = FsmState::COLLISION;
    state_ = FsmState::COLLISION;
    out.macro_tick_event = macro_tick_event_;
    out.local.success = false;
    out.local.vx_body = 0.0;
    out.local.vy_body = 0.0;
    out.local.yaw_rate = 0.0;
    fillObservability(out);
}

// ────────────────────────────────────────────────────────────────────
//  Formal acceptance of a new final goal (5 Hz boundary only)
// ────────────────────────────────────────────────────────────────────
void HierarchicalExpertFsm::acceptNewGoal(const Vec2d& new_goal,
                                          uint64_t tick) {
    tick_ = tick;
    ++goal_revision_;
    // A formally accepted final-goal revision is a NEW MISSION CONTRACT:
    // mission_revision_ increments (the 30 Hz planner may reset its
    // per-mission memory).  5 Hz correction enter / refresh / exit NEVER
    // change mission_revision_.
    ++mission_revision_;
    corrector_.resetForNewGoal();
    directive_ = TargetCorrectionDirective{};
    directive_.valid = true;
    directive_.update_event = corrector_.bumpDirectiveEvent();
    directive_.reason = "NEW_FINAL_GOAL";
    (void)new_goal;
}

// ────────────────────────────────────────────────────────────────────
//  Main step
// ────────────────────────────────────────────────────────────────────
FsmStepOutput HierarchicalExpertFsm::step(const FsmInput& in) {
    FsmStepOutput out;
    tick_ = in.tick;
    out.prev_state = state_;
    out.state = state_;
    out.macro_tick_event = macro_tick_event_;
    last_original_goal_ = in.task.goal;

    // Terminal states are frozen but remain FULLY observable.
    if (isTerminal(state_)) {
        out.local.success = false;
        out.local.vx_body = 0.0;
        out.local.vy_body = 0.0;
        out.local.yaw_rate = 0.0;
        fillObservability(out);
        return out;
    }

    // ── 5 Hz boundary: run the VisibilityTargetCorrector (exactly once
    //    per boundary, ZOH between boundaries). ──
    const bool is_5hz_tick = (in.tick % 6 == 0);
    if (is_5hz_tick) {
        ++macro_tick_event_;
        out.macro_tick_ran = true;
        directive_ = corrector_.update(in.state, in.task.goal,
                                       in.current_patch, in.history, in.tick);
    } else {
        out.macro_tick_ran = false;
    }
    directive_updated_ = directive_.update_event != last_delivered_event_;
    last_delivered_event_ = directive_.update_event;

    // ── EffectiveTargetAdapter EVERY real 30 Hz tick. ──
    last_encoded_ =
        adapter_.encode(in.state, in.task.goal, directive_, in.goal_z);

    // ── LocalTarget for the 30 Hz planner. ──
    out.local_target_updated = directive_updated_;
    out.local_target = makeLocalTarget(last_encoded_);
    out.local_target.update_event = directive_.update_event;

    // ── 30 Hz local plan (merged HISTORY map only). ──
    out.local = local_planner_.plan(in.state, in.history, out.local_target);

    // ── Terminal checks (goal reached judged on the ORIGINAL goal). ──
    if (in.collision) {
        transition(out, FsmState::COLLISION, "COLLISION_DETECTED");
    } else if (goalReached(in)) {
        transition(out, FsmState::GOAL_REACHED, "GOAL_REACHED");
    } else if (p_.task_episode_timeout_s > 0.0 &&
               static_cast<double>(in.tick - reset_tick_) / 30.0 >=
                   p_.task_episode_timeout_s) {
        transition(out, FsmState::TIMEOUT, "EPISODE_TIMEOUT");
    } else {
        updateFailureBookkeeping(out, in);
        switch (state_) {
            case FsmState::DIRECT_LOCAL:
                if (out.local.turn_mode) {
                    transition(out, FsmState::TURN_TO_TARGET,
                               "TARGET_OUTSIDE_FOV");
                }
                break;
            case FsmState::TURN_TO_TARGET:
                if (!out.local.turn_mode) {
                    transition(out, FsmState::DIRECT_LOCAL, "TARGET_IN_FOV");
                }
                break;
            default:
                break;
        }
    }

    reentry_guard_ = corrector_.reentryGuardRemaining(in.tick);
    out.macro_tick_event = macro_tick_event_;
    out.state = state_;
    fillObservability(out);
    return out;
}

}  // namespace expert
}  // namespace il_dataset
