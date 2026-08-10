#pragma once

#include <Eigen/Core>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

struct GoalCaptureConfig {
    double position_tolerance_m = 0.30;
    double speed_tolerance_mps = 0.20;
    double approach_deceleration_mps2 = 2.5;
    double max_approach_speed_mps = 0.80;
    double return_speed_mps = 0.25;
    double position_gain = 1.0;
    double max_acceleration_mps2 = 3.5;
    double max_jerk_mps3 = 25.0;
};

struct GoalCaptureCommand {
    Eigen::Vector3d velocity_world{Eigen::Vector3d::Zero()};
    bool valid = false;
    bool braking = false;
};

/// State-aware terminal controller for a known-clear straight goal segment.
/// Before entering the tolerance ball it applies a stopping-distance speed
/// envelope.  After the external capture latch is set it brakes first and
/// only makes a slow return to the goal after the vehicle has settled.
class GoalCaptureController {
public:
    explicit GoalCaptureController(const GoalCaptureConfig& config);

    void reset();
    GoalCaptureCommand compute(const VehicleState& state,
                               const Eigen::Vector3d& goal_world,
                               bool capture_latched,
                               double dt_s);

private:
    GoalCaptureConfig config_;
    Eigen::Vector3d previous_velocity_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d previous_acceleration_{Eigen::Vector3d::Zero()};
    bool has_previous_ = false;
};

}  // namespace il_dataset
