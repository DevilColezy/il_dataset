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

    // Clearance
    double min_clearance = 0.02;
    double target_clearance = 0.20;
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
    double search_clearance_m = 0.25;
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

    /// Strict validation: every point must be known AND clearance above
    /// `min_clearance`, plus finite state and dynamics feasibility.
    /// Spatially interpolates so the max collision-check spacing along the
    /// trajectory is <= collision_check_spacing (section XVI).
    ValidationResult validateTrajectory(
        const std::vector<TrajectoryPoint>& trajectory) const;

    /// Validate the SUFFIX of a previously planned trajectory that is being
    /// re-executed from the cache (section VIII/X).  The current state is
    /// compared against the trajectory INTERPOLATED at the current age
    /// (position_at_age, velocity_at_age) — never against trajectory[0].
    /// Safety validation starts AT the current age and uses the SAME
    /// spatial interpolation (collision_check_spacing) as the fresh final
    /// validation (sections XIX/XX).
    ///  - plan_start_time: wall time when the trajectory was planned (s).
    ///  - current_time:    wall time now (s).
    /// Returns a ValidationResult.
    ValidationResult validateTrajectorySuffix(
        const std::vector<TrajectoryPoint>& trajectory,
        double plan_start_time,
        double current_time,
        const VehicleState& state,
        double min_clearance,
        double max_position_error,
        double max_velocity_error) const;

    uint64_t currentPlanId() const { return plan_id_counter_; }
    const TrajectoryOptimizationConfig& config() const { return config_; }

private:
    /// Shared continuous spatial validator used by BOTH the fresh final
    /// validation and the cached suffix validation.  Checks every sample
    /// with t >= start_t and spatially interpolates so the maximum gap is
    /// <= collision_check_spacing.  Every interpolated point is checked for
    /// known + clearance.
    ValidationResult validateTrajectorySegmentSpatially(
        const std::vector<TrajectoryPoint>& trajectory,
        double start_t,
        double min_clearance) const;

    TrajectoryOptimizationConfig config_;
    const ObservedMap* map_ = nullptr;
    mutable std::uint64_t plan_id_counter_ = 0;
};

}  // namespace il_dataset
