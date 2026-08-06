#include "il_dataset/local_planner/yaw_planner.hpp"

#include <algorithm>
#include <cmath>

namespace il_dataset {

namespace {
constexpr double kPi = 3.14159265358979323846;

double wrapAngleLocal(double angle) {
    constexpr double kTwoPi = 2.0 * kPi;
    angle = std::fmod(angle, kTwoPi);
    if (angle > kPi) angle -= kTwoPi;
    if (angle < -kPi) angle += kTwoPi;
    return angle;
}

double yawFromVelocity(const Eigen::Vector3d& velocity) {
    return std::atan2(velocity.y(), velocity.x()) - 0.5 * kPi;
}
}  // namespace

YawPlanner::YawPlanner(const YawPlannerConfig& config) : config_(config) {}

double YawPlanner::planYaw(std::vector<TrajectoryPoint>* trajectory,
                           double initial_yaw,
                           double macro_yaw,
                           bool has_macro_yaw) const {
    if (trajectory == nullptr || trajectory->empty()) return initial_yaw;
    const double fov_limit =
        (config_.fov_half_deg - config_.fov_margin_deg) * kPi / 180.0;

    double yaw = wrapAngleLocal(initial_yaw);
    double yaw_rate = 0.0;
    (*trajectory)[0].yaw = yaw;
    (*trajectory)[0].yaw_rate = 0.0;
    double previous_time = (*trajectory)[0].t;
    for (size_t i = 1; i < trajectory->size(); ++i) {
        TrajectoryPoint& point = (*trajectory)[i];
        const double dt = std::max(1.0e-4, point.t - previous_time);
        previous_time = point.t;

        double desired = yaw;
        const double speed =
            point.velocity.head<2>().norm();
        if (speed > config_.speed_threshold_mps) {
            // Moving: follow the horizontal motion so it stays in the FOV.
            // The FOV constraint is satisfied structurally because the
            // motion direction is centred in the camera.  During a bypass
            // this deviates from the macro yaw temporarily.
            desired = yawFromVelocity(point.velocity);
        } else if (has_macro_yaw) {
            // Stationary (OBSERVE rotation / hover): follow the macro yaw.
            desired = wrapAngleLocal(macro_yaw);
        } else {
            desired = yaw;
        }

        // Hard yaw-rate and yaw-acceleration limits.
        const double d = wrapAngleLocal(desired - yaw);
        const double target_rate =
            std::max(-config_.max_yaw_rate,
                     std::min(config_.max_yaw_rate, d / dt));
        const double rate_change_limit = config_.max_yaw_accel * dt;
        yaw_rate = std::max(yaw_rate - rate_change_limit,
                            std::min(yaw_rate + rate_change_limit, target_rate));
        yaw += yaw_rate * dt;
        yaw = wrapAngleLocal(yaw);

        point.yaw = yaw;
        point.yaw_rate = yaw_rate;
    }
    return yaw;
}

}  // namespace il_dataset
