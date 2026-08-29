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
///   * records the flat ExpertStepOutput.
///
/// 3D extension: the expert itself commands the full body-FLU velocity
/// [vx, vy, vz] + yaw_rate; no Python-side altitude merge is performed.
///
/// Coordinate convention: the wrapper accepts the Flightmare state
/// directly (world XY + world Z + Flightmare yaw + FLU velocity + yaw
/// rate) and converts through CoordinateAdapter (the single adaptation
/// layer).

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/kinematics.hpp"
#include "il_dataset/hierarchical_expert/observed_grid_2d.hpp"
#include "il_dataset/hierarchical_expert/effective_target_adapter.hpp"
#include "il_dataset/hierarchical_expert/macro_expert_5hz.hpp"
#include "il_dataset/hierarchical_expert/local_planner_30hz.hpp"
#include "il_dataset/hierarchical_expert/hierarchical_expert_fsm.hpp"
#include "il_dataset/hierarchical_expert/flightmare_2d_observation.hpp"
#include "il_dataset/hierarchical_expert/vertical_controller.hpp"
#include "il_dataset/hierarchical_expert/command_composer_3d.hpp"

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
    // R29r: RAW (unclipped) 3D distance to the effective target, so a
    // downstream re-normalisation never needs to reconstruct the original
    // range from the clipped/normalised pair.
    double goal_distance_raw_m = 0.0;
    // Effective target source (PASS_THROUGH / NORMAL_CORRECTION /
    // TURN_LEFT / TURN_RIGHT) — diagnostic only.
    std::string effective_target_source = "PASS_THROUGH";
    bool target_correction_active = false;
    // Persistent terminal-stop flag of the current ZOH directive (always
    // false: the arbiter never issues terminal stops).  Diagnostic.
    bool directive_terminal_stop = false;
    int effective_direction_token = -1;  // quantized token of live direction
    // Effective world target (diagnostic; world-latched for NORMAL).
    double effective_target_world_x = 0.0;
    double effective_target_world_y = 0.0;
    double effective_target_world_z = 2.0;
    bool effective_target_world_valid = false;

    // ── 30 Hz executable label (body FLU; the vertical channel is now
    //    commanded by the expert itself — 3D extension) ────────────
    double target_velocity_flu_x = 0.0;
    double target_velocity_flu_y = 0.0;
    double target_velocity_flu_z = 0.0;
    double target_yaw_rate = 0.0;
    double intent_vx_body = 0.0;
    double intent_vy_body = 0.0;
    double intent_yaw_rate = 0.0;
    double intent_vz_body = 0.0;

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
    // R26: soft 1 m risk-corridor proximity (diagnostic; NEVER a hard
    // topology block — see corridor_block_reason for the hard verdict).
    bool risk_corridor_near_obstacle = false;
    // R25 structured corridor diagnostics: WHY the straight corridor to the
    // target was judged blocked + the first blocking cell (age 0 = current
    // frame, > 0 = stale history).  Never student inputs.
    std::string corridor_block_reason = "CLEAR";
    std::string corridor_block_source = "none";
    double first_blocking_distance_m =
        std::numeric_limits<double>::quiet_NaN();
    double first_block_x = std::numeric_limits<double>::quiet_NaN();
    double first_block_y = std::numeric_limits<double>::quiet_NaN();
    int first_block_age_ticks = 0;
    bool emergency_brake = false;
    bool immediate_avoidance = false;
    bool local_limit_cycle_detected = false;
    double target_bearing_error_deg = 0.0;
    uint32_t consecutive_failures_30hz = 0;
    uint32_t unknown_recovery_ticks = 0;
    // ── 30 Hz planned trajectory (diagnostic for the stepped viewer).
    //    plan_points_* are world XY (positions are identical between the
    //    expert and Flightmare frames), downsampled to ≤16 points.
    bool plan_valid = false;
    bool plan_terminal = false;
    double plan_end_speed_mps = 0.0;
    double plan_executed_speed_mps = 0.0;
    std::vector<double> plan_points_x;
    std::vector<double> plan_points_y;
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
    // R29r: RAW (unclipped) 3D distance to the ORIGINAL navigation goal.
    double navigation_goal_distance_raw_m = 0.0;

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

    // ── External 5 Hz directive injection (student-upper rollouts) ──
    // When set, the 5 Hz corrector is bypassed and the injected directive
    // is used on every 5 Hz boundary; the 30 Hz LocalPlanner30Hz runs
    // completely unchanged.  ``type`` 0=PASS,1=NORMAL,2=TURN_LEFT,
    // 3=TURN_RIGHT; ``corrected_*`` is the world-latched NORMAL target
    // point, ``turn_*`` the world-latched unit TURN direction, and
    // ``normalized_distance`` the distance label (used for NORMAL).
    void setExternalDirective(int type, double corrected_x, double corrected_y,
                              double turn_dir_x, double turn_dir_y,
                              double normalized_distance,
                              const std::string& reason);
    void clearExternalDirective();
    bool externalDirectiveActive() const {
        return fsm_.externalDirectiveActive();
    }

    /// Preflight access: run a step from an already-integrated observation
    /// without a new depth frame (used by the dry-run simulator which
    /// synthesizes depth rays from the truth scene).  Takes the canonical
    /// 3D state; the planar layers receive the horizontal projection.
    ExpertStepOutput stepFromPatch(const VehicleState3D& expert_state,
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
    const NavigationTask3D& navigationTask() const { return navigation_task_; }
    double flightZ() const { return flight_z_; }

private:
    void fillOutput(ExpertStepOutput& out, const FsmStepOutput& fsm_out,
                    uint64_t tick, double flight_z);
    /// ── 3D extension: overwrite the 2D goal directions with the live 3D
    ///    unit direction (effective target + original goal) projected into
    ///    the FLU body frame, and the 3D macro direction on 5 Hz frames. ─
    void apply3DGoalDirections(ExpertStepOutput& out, const PlanarState& st,
                               double z, double flight_z);

    Params2D p_;
    Vec2d min_bounds_{-20.0, -20.0};
    Vec2d max_bounds_{20.0, 20.0};
    bool configured_ = false;
    ObservedGrid2D history_;
    Flightmare2DObservation obs_builder_;
    HierarchicalExpertFsm fsm_{Params2D{}};
    Task2D task_;
    /// ── 3D extension: canonical 3D navigation task (start/goal with the
    ///    mission altitude + flight band). ────────────────────────────
    NavigationTask3D navigation_task_;
    double flight_z_ = 2.0;
    // Vertical command-ramp base (previous executable vz), reset per task.
    // The VerticalController ramps the executable vz relative to this so it
    // can lead a recovery from a sink (state-pinning it to vz_world made the
    // drone drop ~1.3 m under horizontal-acceleration tilt, joint_v2
    // episode 000000_3d0c3119).
    double last_vz_command_ = 0.0;

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
