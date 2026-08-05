#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <string>
#include <vector>

namespace il_dataset {

/// Vehicle state in ROS world coordinates (x-fwd, y-left, z-up).
struct VehicleState {
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    double yaw = 0.0;
    double yaw_rate = 0.0;
};

/// A single point on a dense time-sampled trajectory.
struct TrajectoryPoint {
    double t = 0.0;
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
    double yaw = 0.0;
    double yaw_rate = 0.0;
    /// ESDF clearance at this point (negative = collision, after drone_radius subtraction).
    double clearance = 0.0;
};

/// Status codes returned by the local planner.
enum class PlannerStatus : int {
    SUCCESS = 0,
    INVALID_INPUT = 1,
    NO_GLOBAL_PATH = 2,
    LOCAL_GOAL_INVALID = 3,
    OPTIMIZATION_FAILED = 4,
    COLLISION = 5,
    DYNAMICS_VIOLATION = 6,
    OUTSIDE_MAP = 7,
    EMERGENCY_HOLD = 8,
    UNKNOWN_SPACE = 9  // Phase 2: trajectory enters unknown space
};

/// Teacher-only local trajectory-planning request.
struct LocalPlanningRequest {
    VehicleState state;

    double previous_progress_s{0.0};

    /// World endpoint reconstructed from the complete macro guide. It may
    /// lie outside the latest FOV when known by the causal rolling map.
    Eigen::Vector3d guide_waypoint{Eigen::Vector3d::Zero()};
    int guide_waypoint_index{-1};

    /// Hard optimization endpoint for the complete teacher trajectory.
    Eigen::Vector3d trajectory_terminal{Eigen::Vector3d::Zero()};
    int trajectory_terminal_index{-1};

    /// World-frame yaw intent supplied by the complete macro guide.  The
    /// teacher planner time-parameterizes this together with translation so
    /// the local expert only tracks one coherent SE(2.5) trajectory.
    bool has_target_yaw{false};
    double target_yaw{0.0};

    /// DEPRECATED: reference_path_segment is no longer used.
    /// The local planner uses the terminal plus ESDF to initialize its local
    /// B-spline (bounded local A* is used only when the direct seed is
    /// obstructed).  This field is kept for backward compatibility but MUST
    /// be empty.
    std::vector<Eigen::Vector3d> reference_path_segment;

    /// If true, use isKnownFree() instead of isFree() for collision checks.
    bool forbid_unknown_space{true};

    /// DEPRECATED: allow_global_map_fallback — NO LONGER SUPPORTED.
    /// The local planner MUST NOT fall back to the global path on failure.
    bool allow_global_map_fallback{false};
};

/// Complete result of a single local-planning invocation.
struct LocalPlanResult {
    bool success = false;
    PlannerStatus status = PlannerStatus::SUCCESS;
    std::string message;

    /// Dense time-sampled trajectory (dt = trajectory_dt).
    std::vector<TrajectoryPoint> trajectory;

    double planning_time_ms = 0.0;
    double min_clearance = 0.0;    ///< minimum ESDF clearance along the planned trajectory
    double progress_s = 0.0;       ///< legacy progress diagnostic
    int progress_index = -1;       ///< legacy waypoint-index diagnostic
    int local_goal_index = -1;     ///< legacy terminal-index diagnostic
    Eigen::Vector3d local_goal{Eigen::Vector3d::Zero()};
    uint64_t plan_id = 0;          ///< monotonically increasing plan identifier

    // Phase 2: explicit guide/terminal separation
    Eigen::Vector3d guide_waypoint{Eigen::Vector3d::Zero()};
    int guide_waypoint_index{-1};
    Eigen::Vector3d trajectory_terminal{Eigen::Vector3d::Zero()};
    int trajectory_terminal_index{-1};
    bool used_global_fallback{false};
    bool used_observed_esdf{false};
};

/// Validation result for a trajectory.
struct ValidationResult {
    bool all_clear = false;
    bool any_collision = false;
    double min_clearance = 0.0;
    int clearance_violation_count = 0;
    Eigen::Vector3d worst_position{Eigen::Vector3d::Zero()};
    double worst_time = 0.0;
    double worst_clearance = 0.0;
};

}  // namespace il_dataset
