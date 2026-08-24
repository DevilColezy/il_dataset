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
    consecutive_hold_ticks_ = 0;
    failure_start_tick_ = 0;
    last_failure_tick_ = 0;
    unknown_recovery_ticks_ = 0;
    unknown_recovery_episode_count_ = 0;
    progress_window_ticks_ = 0;
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
    last_local_result_ = PlannerResult{};
    has_last_local_result_ = false;
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
    const EncodedTargetInput& encoded,
    const TargetCorrectionDirective& directive) const {
    LocalTarget t;
    // The C++ expert may consume the world target directly.  The body
    // direction and normalized distance travel alongside it solely as the
    // student/data-label contract.
    t.planar.position_world = encoded.effective_target_world;
    t.planar.world_valid =
        encoded.valid && encoded.effective_target_world_valid;
    t.planar.direction_body = encoded.direction_body;
    t.planar.normalized_distance = encoded.normalized_distance;
    // R24: a directive flagged terminal_stop (brake-before-search) is a
    // PERSISTENT stop semantic.  Fly-through is decided by that flag, NOT
    // re-derived from the live distance every 30 Hz tick — otherwise a
    // brake point the vehicle coasts a few cm past flips back to
    // fly-through and the planner accelerates through its own brake.
    t.planar.flythrough =
        directive.type == TargetCorrectionType::NORMAL_CORRECTION &&
        !directive.terminal_stop &&
        encoded.normalized_distance > 1e-9;
    t.planar.update_event = corrector_.directiveUpdateEvent();
    t.planar.mission_revision = mission_revision_;
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
    const double dz = in.z - in.goal_z;  // 3D altitude from FsmInput
    const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double v = in.state.velocity_world.norm();
    const double vz = std::fabs(in.vz_world);  // 3D vertical velocity
    const double yr = std::fabs(in.state.yaw_rate);
    return d <= p_.task_goal_tolerance &&
           v < p_.vehicle_goal_stop_speed_mps &&
           vz < p_.vehicle_goal_stop_speed_mps &&
           yr <= p_.lp_turn_exit_max_yaw_rate;
}

LocalPlanningAssessment HierarchicalExpertFsm::assessOriginalTarget(
    const FsmInput& in) const {
    TargetCorrectionDirective pass;
    pass.type = TargetCorrectionType::PASS_THROUGH;
    pass.valid = true;
    return assessDirectiveTarget(in, pass);
}

LocalPlanningAssessment HierarchicalExpertFsm::assessDirectiveTarget(
    const FsmInput& in,
    const TargetCorrectionDirective& directive) const {
    const EncodedTargetInput encoded =
        adapter_.encode(in.state, in.task.goal, directive, in.goal_z, in.z);
    const LocalTarget target = makeLocalTarget(encoded, directive);
    const PreviewResult preview =
        local_planner_.previewPlan(in.state, in.history, target.planar);

    LocalPlanningAssessment assessment;
    const double target_bearing = wrapAngle(
        std::atan2(encoded.effective_target_world.y() - in.state.position.y(),
                   encoded.effective_target_world.x() - in.state.position.x()) -
        in.state.yaw);
    assessment.target_outside_fov =
        std::fabs(target_bearing) > 0.5 * deg2rad(p_.obs_fov_deg);

    assessment.plan_valid = preview.plan_valid;
    assessment.progress_qualified = preview.progress_qualified;
    assessment.local_corridor_blocked = preview.local_corridor_blocked;
    assessment.planner_status = preview.planner_status;
    assessment.failure_reason = preview.failure_reason;

    // Rotation and translation are deliberately separate capabilities.  A
    // TURNING preview means the local layer can look at the target; it does
    // not prove that a collision-free translational route exists.
    assessment.rotation_available =
        preview.success && !preview.emergency_brake &&
        preview.failure_reason == FailureReason::NONE &&
        (preview.turn_mode ||
         (preview.planner_status == PlannerStatus::SAFE_HOLD &&
          assessment.target_outside_fov &&
          !preview.local_corridor_blocked));
    const bool temporary_correction =
        directive.type == TargetCorrectionType::NORMAL_CORRECTION;
    // A terminal_stop brake is a genuine STOP target: its preview must be
    // allowed to be plan_terminal (and must not require fly-through
    // progress).  Ordinary temporary waypoints stay fly-through semantics.
    const bool terminal_brake = temporary_correction && directive.terminal_stop;
    const bool translation_semantics_valid =
        temporary_correction
            ? (terminal_brake
                   ? (preview.progress_qualified || preview.plan_terminal)
                   : (preview.progress_qualified && !preview.plan_terminal))
            : (preview.progress_qualified || preview.plan_terminal);
    assessment.translation_plan_valid =
        preview.success && !preview.emergency_brake &&
        preview.failure_reason == FailureReason::NONE &&
        !preview.turn_mode && preview.plan_valid &&
        preview.planner_status != PlannerStatus::SAFE_HOLD &&
        preview.planner_status != PlannerStatus::TURNING &&
        translation_semantics_valid;
    assessment.terminal_plan_valid =
        assessment.translation_plan_valid && preview.plan_terminal;
    return assessment;
}

void HierarchicalExpertFsm::updateFailureBookkeeping(
    const FsmStepOutput& out, const FsmInput& in) {
    // The 5 Hz corrector consumes this actual 30 Hz execution history.  A
    // cold preview miss alone never authorizes macro takeover.
    const bool blocked =
        out.local.failure_reason == FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE;
    // R28h (P0#4 refined): a SAFE_HOLD is benign while TRANSIENT — a brief
    // replan wait, a pre-rotation brake, or an already-safe motion state.
    // Counting every hold frame inflated consecutive_failures_ and drove
    // spurious 5 Hz takeovers on momentary replan blips (task 236: 6
    // corrections with ZERO NO_SAFE_CANDIDATE).  But a hold that PERSISTS
    // with no plan is a genuine dead-end and MUST hand over to the macro
    // (task 401 r28h: counting nothing left the drone at a 0.1 m corridor
    // block with consecutive_failures_ = 0 forever, episode interrupted).
    // Track consecutive hold frames; count a hold as a failure only once it
    // is sustained (>= kSustainedHoldTicks, ~0.4 s).
    const bool hold = out.local.planner_status == PlannerStatus::SAFE_HOLD;
    consecutive_hold_ticks_ = hold ? consecutive_hold_ticks_ + 1 : 0;
    const bool sustained_hold = consecutive_hold_ticks_ >= kSustainedHoldTicks;
    const bool no_safe =
        out.local.failure_reason == FailureReason::NO_SAFE_CANDIDATE ||
        sustained_hold;
    const bool limit_cycle = out.local.local_limit_cycle_detected;

    // R25 (Fix #4): "genuine" progress that is allowed to DECAY the
    // failure evidence instead of hard-resetting it.  Measured
    // (joint_v2_000004_4ab1e354): sporadic TERMINAL_SETTLING frames inside
    // a ~50 s blocked deadlock kept resetting consecutive_failures_ 37→0,
    // so the 5 Hz layer never took over.  Now:
    //   * TERMINAL_SETTLING counts only when the ORIGINAL goal is genuinely
    //     near (a real terminal approach), not a short-lived artifact of a
    //     blocked cycle;
    //   * turn_mode counts only when the goal is OUTSIDE the FOV (a real
    //     re-acquisition turn); a spin in place with the goal visible is
    //     not recovery;
    //   * real progress subtracts 2 per frame (floor 0) and fully clears
    //     the recovery state only after ~0.5 s of consecutive progress.
    const double goal_dist = (in.task.goal - in.state.position).norm();
    const bool near_goal =
        goal_dist <= adapter_.normalMaxDistanceM() + 1e-9;
    const bool terminal_settling_real =
        out.local.planner_status == PlannerStatus::TERMINAL_SETTLING &&
        near_goal;
    const Vec2d to_goal = in.task.goal - in.state.position;
    const double b_goal = std::fabs(wrapAngle(
        std::atan2(to_goal.y(), to_goal.x()) - in.state.yaw));
    const bool turning_real =
        out.local.turn_mode &&
        b_goal > 0.5 * deg2rad(p_.obs_fov_deg);
    const bool genuine_progress =
        out.local.success && !out.local.emergency_brake &&
        (out.local.progress_qualified || terminal_settling_real ||
         turning_real);

    constexpr uint32_t kProgressClearWindowTicks = 15;  // 0.5 s at 30 Hz
    if (blocked || no_safe || limit_cycle) {
        if (consecutive_failures_ == 0) failure_start_tick_ = in.tick;
        // R28j: remember when the CURRENT failure happened so the 5 Hz
        // takeover can require a RECENT failure (not a decaying leftover
        // from an earlier stall).
        last_failure_tick_ = in.tick;
        ++consecutive_failures_;
        progress_window_ticks_ = 0;
        if (no_safe) {
            if (unknown_recovery_ticks_ == 0) ++unknown_recovery_episode_count_;
            ++unknown_recovery_ticks_;
        } else {
            unknown_recovery_ticks_ = 0;
        }
    } else if (genuine_progress) {
        // Decay by 2 per genuine-progress frame (never a hard reset).
        consecutive_failures_ =
            consecutive_failures_ >= 2 ? consecutive_failures_ - 2 : 0;
        ++progress_window_ticks_;
        if (progress_window_ticks_ >= kProgressClearWindowTicks) {
            failure_start_tick_ = 0;
            unknown_recovery_ticks_ = 0;
        }
    } else {
        // Neutral frame (e.g. SAFE_PROGRESSING without qualified progress):
        // decay one step; the window still accumulates real progress.
        consecutive_failures_ =
            consecutive_failures_ >= 1 ? consecutive_failures_ - 1 : 0;
        ++progress_window_ticks_;
        if (progress_window_ticks_ >= kProgressClearWindowTicks) {
            failure_start_tick_ = 0;
            unknown_recovery_ticks_ = 0;
        }
    }
}

void HierarchicalExpertFsm::fillObservability(FsmStepOutput& out) const {
    out.target_correction_type = static_cast<uint8_t>(directive_.type);
    out.target_correction_type_name =
        targetCorrectionTypeName(directive_.type);
    out.target_correction_active = corrector_.correctionActive();
    out.directive_terminal_stop = directive_.terminal_stop;
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
    if (!out.local_target.planar.valid()) {
        out.local_target =
            makeLocalTarget(last_encoded_, directive_);
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
    consecutive_failures_ = 0;
    consecutive_hold_ticks_ = 0;
    failure_start_tick_ = 0;
    last_failure_tick_ = 0;
    unknown_recovery_ticks_ = 0;
    progress_window_ticks_ = 0;
    last_local_result_ = PlannerResult{};
    has_last_local_result_ = false;
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
        LocalPlanningAssessment assessment = assessOriginalTarget(in);
        bool live_directive_usable = false;
        if (has_last_local_result_) {
            const PlannerResult& live = last_local_result_;
            const bool live_rotation =
                live.success && !live.emergency_brake && live.turn_mode;
            const bool live_translation =
                live.success && !live.emergency_brake && !live.turn_mode &&
                live.failure_reason == FailureReason::NONE &&
                (live.progress_qualified || live.plan_terminal ||
                 live.planner_status == PlannerStatus::TERMINAL_SETTLING);
            live_directive_usable = live_rotation || live_translation;
        }
        // A cold-start preview is advisory.  When PASS_THROUGH is actually
        // executing a safe original-goal trajectory, preserve that plan and
        // let the 30 Hz layer react to fresh observations first.
        const bool pass_executing =
            directive_.type == TargetCorrectionType::PASS_THROUGH &&
            !corrector_.correctionActive();
        if (pass_executing && has_last_local_result_) {
            const PlannerResult& live = last_local_result_;
            const bool live_rotation =
                live.success && !live.emergency_brake && live.turn_mode;
            const bool live_translation =
                live.success && !live.emergency_brake && !live.turn_mode &&
                live.failure_reason == FailureReason::NONE &&
                (live.progress_qualified || live.plan_terminal ||
                 live.planner_status == PlannerStatus::TERMINAL_SETTLING);
            assessment.live_original_plan_usable =
                live_rotation || live_translation;
            if (live_rotation) assessment.rotation_available = true;
            if (live_translation) {
                assessment.translation_plan_valid = true;
                assessment.terminal_plan_valid = live.plan_terminal;
                assessment.plan_valid = true;
            }
        }
        const uint32_t confirm_ticks = static_cast<uint32_t>(
            std::max(1, p_.macro_takeover_confirm_ticks_30hz));
        // R28j: the takeover must reflect a CURRENT inability to plan, not
        // a decaying failure counter.  Measured (task 475 r28i): after an
        // emergency brake the counter stayed above the confirm threshold
        // for ~0.3 s of recovery, and the 5 Hz layer published TURN_RIGHT
        // at frame 60 while the local was already planner_failure_reason=
        // NONE with both bypass corridors visible.  Require the last
        // failure frame to be RECENT (within one 6-tick macro window) in
        // addition to the count and the current block evidence.
        const bool failure_recent =
            last_failure_tick_ != 0 &&
            in.tick - last_failure_tick_ <= kTakeoverFailureRecencyTicks;
        const bool persistent_failure =
            (consecutive_failures_ >= confirm_ticks ||
             unknown_recovery_ticks_ >= confirm_ticks) &&
            failure_recent;
        // A speed-dependent clearance failure means "brake/slow down",
        // not "the topology is blocked".  It remains owned by local.
        const bool dynamic_braking_only =
            has_last_local_result_ &&
            last_local_result_.dynamic_clearance_blocked &&
            !last_local_result_.local_corridor_blocked;
        // R28h (P0#4): a stationary body is NOT topology evidence — the
        // drone may simply be holding/braking (SAFE_HOLD no longer counts
        // as a failure above).  Only an OBSERVED blocked corridor confirms
        // the topology is genuinely blocked; together with persistent real
        // failures this is the sustained large-scale evidence the upper
        // layer must see before taking over.
        const bool topology_evidence_ready =
            has_last_local_result_ &&
            last_local_result_.local_corridor_blocked;
        assessment.takeover_confirmed =
            persistent_failure && topology_evidence_ready &&
            !dynamic_braking_only;
        const VisibilityTargetCorrector::DirectiveAssessmentFn
            assess_directive = [this, &in](
                const TargetCorrectionDirective& candidate) {
                return assessDirectiveTarget(in, candidate);
            };
        directive_ = corrector_.update(in.state, in.task.goal,
                                       in.current_patch, in.history,
                                       assessment, live_directive_usable,
                                       assess_directive);
    } else {
        out.macro_tick_ran = false;
    }
    directive_updated_ = directive_.update_event != last_delivered_event_;
    last_delivered_event_ = directive_.update_event;

    // ── EffectiveTargetAdapter EVERY real 30 Hz tick. ──
    last_encoded_ = adapter_.encode(in.state, in.task.goal, directive_,
                                    in.goal_z, in.z);

    // ── LocalTarget for the 30 Hz planner (planar part) + the vertical
    //    controller (target altitude). ──
    out.local_target_updated = directive_updated_;
    out.local_target = makeLocalTarget(last_encoded_, directive_);
    out.local_target.planar.update_event = directive_.update_event;

    // ── 30 Hz local plan (merged HISTORY map only; PLANAR). ──
    out.local =
        local_planner_.plan(in.state, in.history, out.local_target.planar);
    last_local_result_ = out.local;
    has_last_local_result_ = true;

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
