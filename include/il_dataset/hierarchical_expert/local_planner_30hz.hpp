#pragma once
/// @file   local_planner_30hz.hpp
/// @brief  30 Hz local obstacle-avoidance expert.
///
/// INFORMATION BOUNDARY (enforced by the interface):
///   plan(VehicleState2D, LocalObservation, LocalTarget)
///   — the planner CANNOT see the global ESDF, the obstacle truth, the
///     global path or the scene.  It sees only the current state, the
///     local observation (with short-term history) and the current local
///     target (position + update_event, zero-order held by the FSM).

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/kinematics.hpp"

#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

struct LocalPlannerCandidate {
    // ── Rollout INTENT (long-term desired control). ──
    double desired_vx_body = 0.0;
    double desired_vy_body = 0.0;
    double desired_yaw_rate = 0.0;
    // ── EXECUTABLE OUTPUT: the one-control-period reachable command.
    double vx_body = 0.0;
    double vy_body = 0.0;
    double yaw_rate = 0.0;
    // ── 3D extension: vertical intent / output (body FLU +up, m/s) ──
    double desired_vz_body = 0.0;
    double vz_body = 0.0;

    Trajectory2D nominal_traj;
    Trajectory2D traj;
    double cost = std::numeric_limits<double>::infinity();
    double min_clearance = std::numeric_limits<double>::infinity();
    bool feasible = false;
    std::string reject_reason = "";
    double cost_progress = 0.0;
    double cost_clearance = 0.0;
    double cost_smoothness = 0.0;
    double cost_speed_change = 0.0;
    double cost_yaw_rate_change = 0.0;
    double cost_terminal_heading = 0.0;
    double cost_velocity_alignment = 0.0;
    double cost_cross_track = 0.0;
    double terminal_heading_error_rad = 0.0;
    double velocity_alignment_error_rad = 0.0;
    double cross_track_error_m = 0.0;
    int stable_index = 0;
    double soft_min_clearance = std::numeric_limits<double>::infinity();
    double max_dynamic_required_clearance = 0.0;
    double max_closing_speed = 0.0;
    uint64_t tie_hash = 0;
    double safe_prefix_duration_s = 0.0;
    double nominal_progress_m = 0.0;
    double executable_progress_m = 0.0;
    double achievable_progress_m = 0.0;
    bool progress_qualified = false;
    bool stationary = false;
    // ── 3D extension: vertical rollout diagnostics ─────────────────
    double z_min = std::numeric_limits<double>::infinity();
    double z_max = -std::numeric_limits<double>::infinity();
    bool z_bounds_ok = true;
    PlannerStatus status = PlannerStatus::NO_SAFE_CANDIDATE;
    double obstacle_risk_cost = 0.0;
    double predicted_closest_clearance = std::numeric_limits<double>::infinity();
    double time_to_collision = std::numeric_limits<double>::infinity();
    double avoidance_strength = 0.0;
    bool avoidance_active = false;
};

/// Light result of a NON-mutating preview plan.
struct PreviewResult {
    bool success = false;
    bool turn_mode = false;
    bool emergency_brake = false;
    FailureReason failure_reason = FailureReason::NONE;
    PlannerStatus planner_status = PlannerStatus::NO_SAFE_CANDIDATE;
    bool has_progressing_trajectory = false;
    double executable_progress_m = 0.0;
    double safe_prefix_duration_s = 0.0;
    double selected_output_speed_mps = 0.0;
};

class LocalPlanner30Hz {
public:
    explicit LocalPlanner30Hz(const Params2D& p) : p_(p) {}

    /// Full planning step.  Mutates internal hysteresis + current
    /// trajectory bookkeeping.
    PlannerResult plan(const VehicleState2D& state, const LocalObservation& obs,
                       const LocalTarget& target);

    /// Non-mutating preview.  Same algorithm, no state change.
    PreviewResult previewPlan(const VehicleState2D& state,
                              const LocalObservation& obs,
                              const LocalTarget& target);

    /// True iff the vehicle can come to a full stop (under max accel)
    /// within the observed free space ahead.
    bool canBrakeSafely(const VehicleState2D& state,
                        const LocalObservation& obs) const;

    bool turnHysteresisActive() const { return turn_hysteresis_active_; }
    void resetTurnHysteresis() { turn_hysteresis_active_ = false; }
    /// Reset all per-task planner memory, including command-change history.
    void reset();

private:
    /// Static macro-handoff base clearance used by this planner:
    ///   handoff_clearance = scene_safety_clearance
    ///                     + macro_route_clearance_margin
    ///                     + clearance_discretization_margin_m
    double handoffClearance() const {
        const double geometric =
            p_.scene_safety_clearance + p_.macro_route_clearance_margin +
            p_.lp_clearance_discretization_margin_m;
        return std::max(p_.lp_nominal_clearance_m, geometric);
    }

    /// Speed-dependent required dynamic clearance for a candidate sample
    /// closing on an observed obstacle.
    double requiredClearance(double closing_speed) const {
        const double v = std::max(0.0, closing_speed);
        if (v <= 1e-9) return p_.lp_min_clearance;
        return handoffClearance() + v * p_.lp_obstacle_reaction_time_s +
               (v * v) / std::max(1e-6, 2.0 * p_.lp_max_accel);
    }

    /// Search radius for the nearest-OCCUPIED query at a candidate sample.
    double clearanceSearchRadius(double total_speed) const {
        const double dyn =
            handoffClearance() +
            total_speed * p_.lp_obstacle_reaction_time_s +
            (total_speed * total_speed) /
                std::max(1e-6, 2.0 * p_.lp_max_accel);
        return std::max(p_.lp_soft_clearance_radius_m, dyn);
    }

    struct LimitCycleSample {
        uint64_t target_update_event = 0;
        uint64_t mission_revision = 0;
        double dist_to_target = 0.0;
        double vx_body = 0.0;
        double vy_body = 0.0;
        double yaw_rate = 0.0;
        bool blocked = false;
        Vec2d position{0.0, 0.0};
        Vec2d target_position{0.0, 0.0};
    };

    static uint64_t commandTieHash(double vx, double vy, double yr);

    bool updateLimitCycle(const PlannerResult& res, const LocalTarget& target,
                          const VehicleState2D& state);

    PlannerResult computePlan(const VehicleState2D& state,
                              const LocalObservation& obs,
                              const LocalTarget& target, bool mutate);
    bool currentTrajectoryBlocked(const VehicleState2D& state,
                                  const LocalObservation& obs,
                                  bool& dynamic_violation) const;
    std::vector<LocalPlannerCandidate> generateCandidates(
        const VehicleState2D& state, const LocalTarget& target) const;
    BodyCommand2D reachableCommand(const VehicleState2D& state,
                                   const BodyCommand2D& intent) const;
    Vec2d bodyVelocity(const VehicleState2D& state) const;
    /// ── 3D extension: deterministic altitude-regulation vertical
    ///    intent toward the target altitude (m/s, FLU +up). ─────────
    double verticalIntent(const VehicleState2D& state,
                          const LocalTarget& target) const;
    bool updateMissionState(const LocalTarget& target, PlannerResult& res);
    BodyCommand2D terminalIntent(const VehicleState2D& state,
                                 const LocalTarget& target) const;
    LocalPlannerCandidate makeTerminalCandidate(
        const VehicleState2D& state, const LocalTarget& target) const;
    void collectOccupiedCells(const Trajectory2D& traj,
                              const LocalObservation& obs,
                              std::vector<Vec2d>& out) const;
    void collectRiskOccupiedCells(const VehicleState2D& state,
                                  const LocalObservation& obs,
                                  std::vector<Vec2d>& out) const;
    void computeObstacleRisk(LocalPlannerCandidate& c,
                             const VehicleState2D& state,
                             const std::vector<Vec2d>& occ_cells) const;
    void assessLocalCorridor(const VehicleState2D& state,
                             const LocalObservation& obs,
                             const LocalTarget& target, bool& blocked,
                             double& first_blocking_distance_m) const;
    bool evaluateCandidate(LocalPlannerCandidate& c, const VehicleState2D& state,
                           const LocalObservation& obs, const LocalTarget& target,
                           const std::vector<Vec2d>& risk_occ_cells,
                           std::string& reject_reason,
                           CandidateRejectReason& reject_enum) const;
    bool corridorBlockedByObserved(const VehicleState2D& state,
                                   const LocalObservation& obs,
                                   const LocalTarget& target) const;
    double stoppingDistance(const VehicleState2D& state) const;
    bool spaceToStop(const VehicleState2D& state, const LocalObservation& obs,
                     double dist) const;

    Params2D p_;
    bool turn_hysteresis_active_ = false;
    Trajectory2D current_trajectory_;
    BodyCommand2D last_command_;
    bool has_last_command_ = false;
    uint64_t last_mission_revision_ = 0;
    Vec2d last_target_position_{0.0, 0.0};
    bool last_target_valid_ = false;
    std::vector<LimitCycleSample> limit_cycle_window_;
    bool limit_cycle_detected_ = false;
    uint64_t last_cycle_mission_revision_ = 0;
};

}  // namespace expert
}  // namespace il_dataset
