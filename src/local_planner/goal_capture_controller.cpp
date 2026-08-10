#include "il_dataset/local_planner/goal_capture_controller.hpp"

#include <algorithm>
#include <cmath>

namespace il_dataset {
namespace {

Eigen::Vector3d clampNorm(const Eigen::Vector3d& value, double limit) {
    const double norm = value.norm();
    if (!std::isfinite(norm) || limit < 0.0) {
        return Eigen::Vector3d::Zero();
    }
    if (norm <= limit || norm <= 1.0e-9) return value;
    return value * (limit / norm);
}

}  // namespace

GoalCaptureController::GoalCaptureController(const GoalCaptureConfig& config)
    : config_(config) {}

void GoalCaptureController::reset() {
    previous_velocity_.setZero();
    previous_acceleration_.setZero();
    has_previous_ = false;
}

GoalCaptureCommand GoalCaptureController::compute(
    const VehicleState& state,
    const Eigen::Vector3d& goal_world,
    bool capture_latched,
    double dt_s) {
    GoalCaptureCommand output;
    if (!state.position.allFinite() || !state.velocity.allFinite() ||
        !goal_world.allFinite() || !std::isfinite(dt_s) || dt_s <= 0.0) {
        return output;
    }

    Eigen::Vector3d delta = goal_world - state.position;
    // Altitude is controlled by the manager's common flight-slice hold.
    delta.z() = 0.0;
    const double distance = delta.norm();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    if (distance > 1.0e-9) direction = delta / distance;
    const double horizontal_speed = state.velocity.head<2>().norm();
    Eigen::Vector3d base_velocity = has_previous_
        ? previous_velocity_ : state.velocity;
    base_velocity.z() = 0.0;

    Eigen::Vector3d desired = Eigen::Vector3d::Zero();
    if (capture_latched &&
        horizontal_speed > config_.speed_tolerance_mps) {
        // Once the vehicle has crossed the capture ball, never request
        // another forward approach while it is still moving.  Brake the
        // previously commanded velocity first; this removes the old
        // replan/reaccelerate
        // cycle around the goal.
        const double commanded_speed = base_velocity.norm();
        const double reduced_speed = std::max(
            0.0, commanded_speed -
                     config_.approach_deceleration_mps2 * dt_s);
        desired = base_velocity *
                  (reduced_speed / std::max(commanded_speed, 1.0e-9));
        output.braking = true;
    } else if (capture_latched &&
               distance <= config_.position_tolerance_m) {
        desired.setZero();
        output.braking = true;
    } else if (capture_latched) {
        const double return_speed = std::min(
            config_.return_speed_mps, config_.position_gain * distance);
        desired = direction * return_speed;
    } else {
        const double remaining = std::max(
            0.0, distance - config_.position_tolerance_m);
        const double stopping_speed = std::sqrt(
            2.0 * config_.approach_deceleration_mps2 * remaining);
        const double desired_speed = std::min({
            config_.max_approach_speed_mps,
            config_.position_gain * distance,
            stopping_speed});
        desired = direction * desired_speed;
    }

    Eigen::Vector3d acceleration =
        clampNorm((desired - base_velocity) / dt_s,
                  config_.max_acceleration_mps2);
    if (has_previous_ && !output.braking) {
        const Eigen::Vector3d jerk =
            (acceleration - previous_acceleration_) / dt_s;
        acceleration = previous_acceleration_ +
            clampNorm(jerk, config_.max_jerk_mps3) * dt_s;
        acceleration = clampNorm(acceleration,
                                 config_.max_acceleration_mps2);
    }

    Eigen::Vector3d command = base_velocity + acceleration * dt_s;
    command.z() = 0.0;
    if (output.braking) {
        const double base_speed = base_velocity.norm();
        const double command_speed = command.norm();
        if (command.dot(base_velocity) < 0.0) {
            command.setZero();
        } else if (command_speed > base_speed && command_speed > 1.0e-9) {
            command *= base_speed / command_speed;
        }
    }
    if (!command.allFinite()) return output;

    previous_velocity_ = command;
    previous_acceleration_ = acceleration;
    has_previous_ = true;
    output.velocity_world = command;
    output.valid = true;
    return output;
}

}  // namespace il_dataset
