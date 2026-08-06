#pragma once

#include <Eigen/Core>
#include <vector>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

/// Configuration for the 30 Hz yaw trajectory planner (section XI).
struct YawPlannerConfig {
    double max_yaw_rate = 2.0;
    double max_yaw_accel = 8.0;
    /// Camera horizontal FOV half-angle (rad).
    double fov_half_deg = 45.0;
    /// Keep the motion direction inside fov_half - margin.
    double fov_margin_deg = 5.0;
    /// Speed below which the drone is treated as stationary (macro yaw
    /// governs).
    double speed_threshold_mps = 0.20;
};

/// Plans the yaw trajectory for a planned local trajectory.
///
/// Rules (section XI):
///  - When the drone is moving, the yaw follows the horizontal motion
///    direction so the actual heading always keeps the motion in the camera
///    FOV.  During a small-scale bypass this intentionally deviates from the
///    macro yaw.
///  - When nearly stationary (OBSERVE rotation, hover), the yaw follows the
///    macro desired yaw.
///  - The yaw rate and yaw acceleration are hard-limited; the trajectory
///    duration must already allow enough time for the required rotation.
///  - No fixed goal/motion weight blend is used.
class YawPlanner {
public:
    explicit YawPlanner(const YawPlannerConfig& config);

    /// Fill `yaw` and `yaw_rate` on every trajectory point starting from
    /// `initial_yaw`.  Returns the final yaw.
    double planYaw(std::vector<TrajectoryPoint>* trajectory,
                   double initial_yaw,
                   double macro_yaw,
                   bool has_macro_yaw) const;

private:
    YawPlannerConfig config_;
};

}  // namespace il_dataset
