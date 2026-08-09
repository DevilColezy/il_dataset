#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class ObservedMap;

/// Configuration for the 30 Hz local trajectory optimization
/// (section X.4, "trajectory_optimization" config module).
struct TrajectoryOptimizationConfig {
    // Timing
    double planning_time_budget_ms = 30.0;
    double trajectory_dt = 0.04;
    double horizon_time = 2.5;

    // B-spline
    std::string optimizer = "auto";  // auto | nlopt | native
    int control_points = 12;
    int max_iterations = 10000;
    double convergence_tolerance = 1.0e-4;
    double initial_step_size = 0.1;
    double minimum_step_size = 1.0e-4;
    double seed_trust_radius = 0.35;
    bool horizontal_avoidance_only = true;

    // Clearance — the single UNIFIED navigation clearance (problem 4).
    // The observed ESDF already subtracts the vehicle radius, so this is
    // the same additional margin used by the global connectivity and every
    // other module.  It is BOTH the trajectory optimizer's soft target AND
    // the hard validation floor: the local trajectory / goal-stop /
    // braking checks are never more permissive than global connectivity.
    double clearance_m = 0.20;
    // ── Dynamic executability margin (P1) ───────────────────────────
    // The LOCAL layer adds a speed-dependent buffer ON TOP of the unified
    // clearance so accepted trajectories never hug the 0.20 m floor: it
    // composes the control latency (margin_latency_s * speed, the distance
    // travelled before the command takes effect) and a constant tracking-
    // error floor, capped by margin_max_m.  The effective boundary is
    // `clearance_m + min(margin_max_m, tracking + latency*speed)` and is
    // applied consistently to the A* seed, the optimizer cost/floor, the
    // fresh validation AND the cached-suffix validation of the LOCAL
    // layer.  All OTHER modules (recoverability, candidates, privileged
    // audit, global connectivity) keep the unified base `clearance_m` so
    // the decision boundary stays module-consistent; only the executable
    // 30 Hz trajectory is planned/validated with the extra buffer.
    double clearance_margin_tracking_m = 0.05;
    double clearance_margin_latency_s = 0.10;
    double clearance_margin_max_m = 0.25;
    double collision_check_spacing = 0.05;

    // Cost weights
    double weight_path_length = 0.05;
    double weight_smooth = 1.0;
    double weight_jerk = 0.2;
    double weight_obstacle = 4.0;
    double weight_dynamics = 1.0;

    // Dynamics
    double nominal_speed = 1.8;
    double max_velocity = 2.5;
    double max_acceleration = 8.0;
    double max_jerk = 50.0;
    double lookahead_distance = 4.0;
    /// Fraction of nominal speed for non-final terminals (cruise-through).
    double terminal_speed_ratio = 0.85;
    /// Distance tolerance for treating the trajectory terminal as the goal
    /// (then the terminal velocity is driven to zero).  Section XVII.
    double goal_stop_tolerance_m = 0.4;

    // Warm start / trajectory continuity
    double warm_start_max_age_s = 0.25;
    double warm_start_max_terminal_deviation_m = 1.5;

    // Local A* seed (populated from the local_path_search config module).
    // Uses the SAME unified clearance_m (never a second, more permissive
    // boundary).
    double search_max_time_ms = 18.0;
    double search_region_margin_m = 2.0;
    double search_side_bias_gain = 2.0;

    // Yaw planning (populated from the yaw_planning config module).
    double yaw_max_rate = 2.0;
    double yaw_max_accel = 8.0;
    double yaw_fov_half_deg = 45.0;
    double yaw_fov_margin_deg = 5.0;
    double yaw_speed_threshold_mps = 0.20;
};

/// The 30 Hz local trajectory planner (sections X, XII, XVII).
///
/// Planning flow:
///   observed-map A* seed path
///   -> previous-trajectory warm start (known-free revalidated)
///   -> B-spline optimization
///   -> dynamic retiming
///   -> strict known-mask + clearance validation
///   -> yaw planning (FOV-constrained)
///
/// The planner consumes ONLY the observed map, the current state, the held
/// macro guide/yaw and the previous trajectory.  It never touches the
/// privileged global map.
class LocalPlanner {
public:
    explicit LocalPlanner(const TrajectoryOptimizationConfig& config);

    /// Point the planner at the observed map (must outlive the planner).
    void setMap(const ObservedMap* map);

    /// Plan a 30 Hz local trajectory.
    LocalPlanResult plan(const LocalPlanRequest& request) const;

    /// Strict validation of a complete trajectory (section XVI): every
    /// point must be known AND clearance above the SPEED-DEPENDENT
    /// effective clearance (round 6), plus finite state and dynamics
    /// feasibility.  Spatially interpolates so the max collision-check
    /// spacing along the trajectory is <= collision_check_spacing.
    ///
    /// `state` is the vehicle state at the START of the trajectory: it
    /// provides the speed at which the unified dynamic-executability margin
    /// is evaluated, so the public validator is never more permissive than
    /// what `plan()` actually validates its executable trajectories with
    /// (no base-clearance-only validation path remains).
    ValidationResult validateTrajectory(
        const std::vector<TrajectoryPoint>& trajectory,
        const VehicleState& state) const;

    /// Validate the SUFFIX of a previously planned trajectory that is being
    /// re-executed from the cache (section VIII/X).  The current state is
    /// compared against the trajectory INTERPOLATED at the current age
    /// (position_at_age, velocity_at_age) — never against trajectory[0].
    /// Safety validation starts AT the current age, uses the SAME unified
    /// `config_.clearance_m` and the SAME spatial interpolation
    /// (collision_check_spacing) as the fresh final validation (sections
    /// XIX/XX).  No external clearance is accepted: the unified
    /// navigation clearance is the ONLY effective safety boundary.
    ///  - plan_start_time: wall time when the trajectory was planned (s).
    ///  - current_time:    wall time now (s).
    /// Returns a ValidationResult.
    ValidationResult validateTrajectorySuffix(
        const std::vector<TrajectoryPoint>& trajectory,
        double plan_start_time,
        double current_time,
        const VehicleState& state,
        double max_position_error,
        double max_velocity_error) const;

    /// The SINGLE C++ dynamic effective-clearance computation (round 5/6):
    /// `clearance_m + min(margin_max_m, margin_tracking_m +
    /// margin_latency_s * speed)`, evaluated at the current state speed.
    /// PUBLIC so the pybind interface (`effective_clearance_for`) and any
    /// other module can call it directly — there is exactly one formula in
    /// `effectiveClearanceForSpeed()` (types.hpp) and no second copy.
    double effectiveClearance(const VehicleState& state) const;

    uint64_t currentPlanId() const { return plan_id_counter_; }
    const TrajectoryOptimizationConfig& config() const { return config_; }

private:
    /// Shared continuous spatial validator used by BOTH the fresh final
    /// validation and the cached suffix validation.  Checks every sample
    /// with t >= start_t and spatially interpolates so the maximum gap is
    /// <= collision_check_spacing.  Every interpolated point is checked for
    /// known + clearance.  `clearance` is the effective safety boundary
    /// (always the speed-dependent effective clearance; the public
    /// `validateTrajectory`, `plan()` and the cached-suffix validation all
    /// pass it — never a bare base clearance).
    ValidationResult validateTrajectorySegmentSpatially(
        const std::vector<TrajectoryPoint>& trajectory,
        double start_t,
        double clearance) const;

    TrajectoryOptimizationConfig config_;
    const ObservedMap* map_ = nullptr;
    mutable std::uint64_t plan_id_counter_ = 0;
};

}  // namespace il_dataset
