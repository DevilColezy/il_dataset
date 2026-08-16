#pragma once
/// @file   hierarchical_expert.hpp
/// @brief  Top-level wrapper: ONE C++ instance owns the 5 Hz corrector,
///         the 30 Hz planner, the effective-target adapter, the FSM, the
///         causal local-history grid and the Flightmare depth→grid
///         observation builder.
///
/// Python only:
///   * drives the ROS/Flightmare lifecycle,
///   * synchronises depth frames + state,
///   * calls step() on this class,
///   * records the flat ExpertStepOutput,
///   * merges the altitude-hold vz into the 30 Hz command.
///
/// Coordinate convention: the wrapper accepts the Flightmare state
/// directly (world XY + Flightmare yaw + FLU velocity + yaw rate) and
/// converts through CoordinateAdapter (the single adaptation layer).

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/kinematics.hpp"
#include "il_dataset/hierarchical_expert/observed_grid_2d.hpp"
#include "il_dataset/hierarchical_expert/effective_target_adapter.hpp"
#include "il_dataset/hierarchical_expert/macro_expert_5hz.hpp"
#include "il_dataset/hierarchical_expert/local_planner_30hz.hpp"
#include "il_dataset/hierarchical_expert/hierarchical_expert_fsm.hpp"
#include "il_dataset/hierarchical_expert/flightmare_2d_observation.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

/// Flat per-tick output consumed by the Python manager / CSV writer.
/// All coordinates are already in the Flightmare FLU / world frame.
struct ExpertStepOutput {
    // ── timing / cadence ───────────────────────────────────────────
    uint64_t tick = 0;
    bool macro_update_mask = false;   // true exactly on tick % 6 == 0

    // ── FSM ────────────────────────────────────────────────────────
    std::string fsm_state = "DIRECT_LOCAL";
    std::string fsm_prev_state = "DIRECT_LOCAL";
    bool terminal = false;

    // ── 30 Hz effective target (student input, live re-expressed) ──
    double goal_direction_flu_x = 1.0;
    double goal_direction_flu_y = 0.0;
    double goal_direction_flu_z = 0.0;
    double goal_distance_clipped_m = 0.0;
    double goal_distance_norm = 0.0;
    // Effective target source (PASS_THROUGH / NORMAL_CORRECTION /
    // TURN_LEFT / TURN_RIGHT) — diagnostic only.
    std::string effective_target_source = "PASS_THROUGH";
    bool target_correction_active = false;
    int effective_direction_token = -1;  // quantized token of live direction
    // Effective world target (diagnostic; world-latched for NORMAL).
    double effective_target_world_x = 0.0;
    double effective_target_world_y = 0.0;
    bool effective_target_world_valid = false;

    // ── 30 Hz executable label (body FLU; z merged by the altitude
    //    controller in Python) ──────────────────────────────────────
    double target_velocity_flu_x = 0.0;
    double target_velocity_flu_y = 0.0;
    double target_yaw_rate = 0.0;
    double intent_vx_body = 0.0;
    double intent_vy_body = 0.0;
    double intent_yaw_rate = 0.0;

    // ── 30 Hz diagnostics ──────────────────────────────────────────
    // hierarchical_mode: the FULL new-architecture expert state
    // (direct / local_avoidance / macro_normal / macro_turn_left /
    //  macro_turn_right / turn_to_target / goal_capture / blocked).
    // There is NO legacy lossy expert_mode projection (save_net reads
    // hierarchical_mode directly).
    std::string hierarchical_mode = "direct";
    std::string planner_status = "NO_SAFE_CANDIDATE";
    std::string failure_reason = "NONE";
    double selected_output_speed_mps = 0.0;
    double local_target_distance_m = 0.0;
    double min_observed_clearance_m = std::numeric_limits<double>::infinity();
    double obstacle_risk_cost = 0.0;
    bool avoidance_active = false;
    bool local_corridor_blocked = false;
    bool emergency_brake = false;
    bool immediate_avoidance = false;
    bool local_limit_cycle_detected = false;
    double target_bearing_error_deg = 0.0;
    uint32_t consecutive_failures_30hz = 0;
    uint32_t unknown_recovery_ticks = 0;
    // ── 30 Hz candidate-rejection breakdown (diagnostic) ───────────
    uint32_t reject_not_known_free = 0;
    uint32_t reject_outside_current_fov = 0;
    uint32_t reject_observed_clearance_too_small = 0;
    uint32_t reject_no_progress = 0;
    uint32_t reject_insufficient_braking_clearance = 0;
    uint32_t reject_other = 0;

    // ── 5 Hz labels (valid on macro_update_mask == 1; zero-order held
    //    on the other frames) ───────────────────────────────────────
    int macro_label_valid = 1;
    std::string macro_correction_type = "PASS_THROUGH";
    int macro_direction_token = -1;
    double macro_direction_flu_x = 1.0;
    double macro_direction_flu_y = 0.0;
    double macro_direction_flu_z = 0.0;
    double macro_distance_norm = 0.0;
    int macro_param_valid = 0;

    // ── 5 Hz student input: the ORIGINAL navigation goal (live
    //    re-expressed every frame; never confused with the 30 Hz
    //    effective goal_*) ──────────────────────────────────────────
    double navigation_goal_direction_flu_x = 1.0;
    double navigation_goal_direction_flu_y = 0.0;
    double navigation_goal_direction_flu_z = 0.0;
    double navigation_goal_distance_clipped_m = 0.0;
    double navigation_goal_distance_norm = 0.0;

    // ── privileged diagnostics (world-frame; never student inputs) ──
    double original_navigation_goal_world_x = 0.0;
    double original_navigation_goal_world_y = 0.0;
    double original_navigation_goal_world_z = 0.0;
    uint64_t correction_enter_event = 0;
    uint64_t correction_exit_event = 0;
    uint64_t correction_update_event = 0;
    std::string observability_reason = "NONE";
    bool observability_goal_inside_fov = false;
    bool observability_direct_corridor_blocked = false;
    bool observability_left_bypass_visible = false;
    bool observability_right_bypass_visible = false;
    bool observability_local_avoidance_observable = false;
    uint64_t directive_update_event = 0;
    uint64_t mission_revision = 0;
    int reentry_guard_ticks = 0;
    uint64_t obstacle_first_observed_event = 0;
    bool macro_tick_ran = false;
};

class HierarchicalExpert {
public:
    HierarchicalExpert() = default;

    /// Configure the expert with the single authoritative parameter set.
    /// `min_bounds` is the Flightmare scene horizontal min corner (the
    /// global grid anchor for both the FOV patch and the history map).
    void configure(const Params2D& p, const Vec2d& min_bounds,
                   const Vec2d& max_bounds);

    /// Begin a new episode (new task).  `start`/`goal` are world XY,
    /// `initial_yaw` is the Flightmare yaw; all converted internally.
    /// `flight_z` is the fixed 2D flight height (used only for logging).
    void resetTask(const Vec2d& start, const Vec2d& goal,
                   double initial_yaw_fm, uint64_t tick, double flight_z);

    /// Advance one 30 Hz tick.  The Flightmare state is given directly;
    /// the wrapper converts through CoordinateAdapter.  `depth_m` is the
    /// current depth frame (metres, row-major), `cam_pos`/`cam_q` the
    /// VEHICLE pose (the true camera pose is derived from Params2D T_BC
    /// inside the observation builder; FOV/R/resolution come from
    /// Params2D only).  `flight_z` is the current altitude (for the
    /// world-z diagnostics).  `collision` is the (privileged, judge-only)
    /// collision flag.
    ExpertStepOutput step(const double pos[3], double yaw_fm,
                          const double vel_world[3], double yaw_rate_fm,
                          const std::vector<float>& depth_m, int depth_w,
                          int depth_h, const double cam_pos[3],
                          const double cam_q[4],
                          double flight_z, uint64_t tick, bool collision);

    /// Formally accept a new final navigation goal (only meaningful at a
    /// 5 Hz boundary; joint_v2 keeps the original goal fixed per episode).
    void acceptNewGoal(const Vec2d& goal, uint64_t tick);

    /// Preflight access: run a step from an already-integrated observation
    /// without a new depth frame (used by the dry-run simulator which
    /// synthesizes depth rays from the truth scene).  See preflight.cpp.
    ExpertStepOutput stepFromPatch(const VehicleState2D& expert_state,
                                   const LocalObservation& current_patch,
                                   double flight_z, uint64_t tick,
                                   bool collision);

    const Params2D& params() const { return p_; }
    const TargetCorrectionDirective& lastDirective() const {
        return fsm_.lastDirective();
    }
    const EncodedTargetInput& lastEncoded() const { return fsm_.lastEncoded(); }
    const AvoidanceObservability& lastObservability() const {
        return fsm_.corrector().lastObservability();
    }
    uint64_t obstacleFirstObservedEvent() const {
        return history_.obstacleFirstObservedEvent();
    }
    const Task2D& task() const { return task_; }
    double flightZ() const { return flight_z_; }

private:
    void fillOutput(ExpertStepOutput& out, const FsmStepOutput& fsm_out,
                    uint64_t tick, double flight_z);

    Params2D p_;
    Vec2d min_bounds_{-20.0, -20.0};
    Vec2d max_bounds_{20.0, 20.0};
    bool configured_ = false;
    ObservedGrid2D history_;
    Flightmare2DObservation obs_builder_;
    HierarchicalExpertFsm fsm_{Params2D{}};
    Task2D task_;
    double flight_z_ = 2.0;

    // ZOH mirror of the last 5 Hz label block.
    int last_macro_label_valid_ = 1;
    std::string last_macro_correction_type_ = "PASS_THROUGH";
    int last_macro_direction_token_ = -1;
    double last_macro_direction_flu_x_ = 1.0;
    double last_macro_direction_flu_y_ = 0.0;
    double last_macro_direction_flu_z_ = 0.0;
    double last_macro_distance_norm_ = 0.0;
    int last_macro_param_valid_ = 0;
};

}  // namespace expert
}  // namespace il_dataset
