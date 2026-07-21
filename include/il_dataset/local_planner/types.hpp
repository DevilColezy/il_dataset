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
    EMERGENCY_HOLD = 8
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
    double progress_s = 0.0;       ///< current progress along global path (arc-length)
    int progress_index = -1;       ///< global-path waypoint index for progress_s
    int local_goal_index = -1;     ///< global-path waypoint index of the local goal
    Eigen::Vector3d local_goal{Eigen::Vector3d::Zero()};
    uint64_t plan_id = 0;          ///< monotonically increasing plan identifier
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
