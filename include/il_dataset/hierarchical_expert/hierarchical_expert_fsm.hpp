#pragma once
/// @file   hierarchical_expert_fsm.hpp
/// @brief  Two-level expert state machine (v9).
///
/// The 5 Hz VisibilityTargetCorrector runs on every 5 Hz boundary
/// (tick % 6 == 0) inside the single 30 Hz step — independent of the
/// 30 Hz outcome — and produces a zero-order-held
/// TargetCorrectionDirective.  The EffectiveTargetAdapter converts that
/// directive EVERY 30 Hz tick into the LocalTarget the 30 Hz planner sees
/// (world point for the C++ expert, body direction + normalized distance
/// for the future student).
///
/// States: DIRECT_LOCAL, TURN_TO_TARGET, GOAL_REACHED, TASK_INVALID,
/// COLLISION, TIMEOUT.  The 30 Hz planner keeps its own TURN_TO_TARGET
/// hysteresis; the 5 Hz layer never reads the 30 Hz result.

#include "il_dataset/hierarchical_expert/local_planner_30hz.hpp"
#include "il_dataset/hierarchical_expert/macro_expert_5hz.hpp"
#include "il_dataset/hierarchical_expert/effective_target_adapter.hpp"
#include "il_dataset/hierarchical_expert/types.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

/// Everything the FSM needs from the outside world for one tick.
struct FsmInput {
    const Task2D& task;
    const VehicleState2D& state;
    /// INSTANTANEOUS FOV patch of THIS tick (before merging into history).
    /// Only the 5 Hz VisibilityTargetCorrector reads it.
    const LocalObservation& current_patch;
    /// Merged short-term HISTORY map (ObservedGrid2D).  Only the 30 Hz
    /// planner reads it.
    const LocalObservation& history;
    uint64_t tick;
    bool collision;
};

/// Flat per-tick output consumed by the Python manager / CSV writer.
struct FsmStepOutput {
    FsmState state = FsmState::DIRECT_LOCAL;
    FsmState prev_state = FsmState::DIRECT_LOCAL;
    PlannerResult local;
    LocalTarget local_target;
    bool local_target_updated = false;

    // ── v9: 5 Hz target correction + effective-target encoding ──────
    uint8_t target_correction_type = 0;  // TargetCorrectionType
    std::string target_correction_type_name = "PASS_THROUGH";
    bool target_correction_active = false;
    int32_t target_direction_token = -1;
    double target_direction_x_body = 1.0;
    double target_direction_y_body = 0.0;
    double target_distance_normalized = 0.0;
    double effective_target_x = 0.0;
    double effective_target_y = 0.0;
    bool effective_target_world_valid = false;
    Vec2d original_goal{0.0, 0.0};

    // ── 5 Hz local observability diagnostics ────────────────────────
    bool observability_goal_inside_fov = false;
    bool observability_direct_corridor_blocked = false;
    bool observability_blocker_observed = false;
    bool observability_left_bypass_visible = false;
    bool observability_right_bypass_visible = false;
    bool observability_local_avoidance_observable = false;
    bool observability_fov_boundary_truncated = false;
    bool observability_unknown_occluded = false;
    std::string observability_reason = "NONE";
    double observability_left_score = 0.0;
    double observability_right_score = 0.0;
    uint64_t correction_enter_event = 0;
    uint64_t correction_exit_event = 0;
    uint64_t correction_update_event = 0;
    SideSelection side = SideSelection::NONE;

    // ── counters / diagnostics ─────────────────────────────────────
    uint64_t directive_update_event = 0;
    uint64_t mission_revision = 0;
    uint32_t consecutive_failures_30hz = 0;
    uint32_t unknown_recovery_ticks = 0;
    bool unknown_recovery_active = false;
    uint32_t unknown_recovery_episode_count = 0;
    bool macro_tick_ran = false;
    uint64_t macro_tick_event = 0;
    int reentry_guard = 0;
    uint64_t obstacle_first_observed_event = 0;
};

class HierarchicalExpertFsm {
public:
    explicit HierarchicalExpertFsm(const Params2D& p);

    /// Reset all state (call on every new task / scene).
    void reset(const Task2D& task, uint64_t tick);

    /// Advance one 30 Hz tick.  The 5 Hz corrector runs at tick % 6 == 0;
    /// the EffectiveTargetAdapter runs every tick.
    FsmStepOutput step(const FsmInput& in);

    /// Force the COLLISION terminal state.
    void forceCollision(FsmStepOutput& out);

    /// Formally accept a NEW final navigation goal at a 5 Hz boundary.
    void acceptNewGoal(const Vec2d& new_goal, uint64_t tick);

    FsmState state() const { return state_; }
    uint64_t effectiveLocalTargetEvent() const {
        return corrector_.directiveUpdateEvent();
    }
    uint64_t macroTickEvent() const { return macro_tick_event_; }
    uint64_t acceptedGoalEvent() const { return goal_revision_; }
    int reentryGuardTicks() const { return reentry_guard_; }
    const VisibilityTargetCorrector& corrector() const { return corrector_; }

    /// Direct access for the wrapper / pybind (observe the current ZOH
    /// directive without running a step).
    const TargetCorrectionDirective& lastDirective() const {
        return corrector_.lastDirective();
    }
    const EncodedTargetInput& lastEncoded() const { return last_encoded_; }

private:
    bool isTerminal(FsmState s) const;
    void transition(FsmStepOutput& out, FsmState next,
                    const std::string& reason);
    void fillObservability(FsmStepOutput& out) const;
    bool goalReached(const FsmInput& in) const;
    void updateFailureBookkeeping(const FsmStepOutput& out,
                                  const FsmInput& in);
    LocalTarget makeLocalTarget(const EncodedTargetInput& encoded) const;

    Params2D p_;
    FsmState state_ = FsmState::DIRECT_LOCAL;
    uint64_t tick_ = 0;
    /// Tick at the last reset() — the optional episode timeout is measured
    /// as (tick - reset_tick_) so it never depends on the absolute tick
    /// (which is a multiple of 600000 per episode).
    uint64_t reset_tick_ = 0;

    LocalPlanner30Hz local_planner_;
    VisibilityTargetCorrector corrector_;
    EffectiveTargetAdapter adapter_;

    // ── v9 per-tick state ───────────────────────────────────────────
    TargetCorrectionDirective directive_;
    EncodedTargetInput last_encoded_;
    bool directive_updated_ = false;
    uint64_t last_delivered_event_ = 0;
    Vec2d last_original_goal_{0.0, 0.0};

    uint32_t consecutive_failures_ = 0;
    uint64_t failure_start_tick_ = 0;
    uint32_t unknown_recovery_ticks_ = 0;
    uint32_t unknown_recovery_episode_count_ = 0;
    uint64_t goal_revision_ = 0;
    uint64_t mission_revision_ = 0;
    int reentry_guard_ = 0;
    uint64_t macro_tick_event_ = 0;
};

}  // namespace expert
}  // namespace il_dataset
