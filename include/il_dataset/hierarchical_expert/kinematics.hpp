#pragma once
/// @file   kinematics.hpp
/// @brief  Shared deterministic 2D kinematic integration (pure C++).
///
/// The SAME function is used by:
///   * the preflight 2D simulator step — actual execution,
///   * LocalPlanner30Hz candidate prediction,
/// so prediction and execution can never diverge.  Every step applies:
///   * linear acceleration limit,
///   * radial speed limit (max_speed),
///   * yaw acceleration limit,
///   * yaw rate limit (max_yaw_rate).

#include "il_dataset/hierarchical_expert/types.hpp"

#include <cmath>

namespace il_dataset {
namespace expert {

/// Body-frame velocity + yaw-rate command.
struct BodyCommand2D {
    double vx_body = 0.0;  // forward (+X body)
    double vy_body = 0.0;  // lateral  (+Y body = left, CCW)
    double yaw_rate = 0.0;
};

/// Advance the planar vehicle state by dt under the limits above.
inline VehicleState2D integrateKinematicStep(const VehicleState2D& s,
                                             const BodyCommand2D& cmd,
                                             double dt, const Params2D& p) {
    VehicleState2D ns;
    const double c = std::cos(s.yaw), sn = std::sin(s.yaw);

    // world → body
    const Vec2d v_body(c * s.velocity_world.x() + sn * s.velocity_world.y(),
                       -sn * s.velocity_world.x() + c * s.velocity_world.y());

    // linear acceleration limit toward the commanded body velocity
    const Vec2d v_body_new(
        v_body.x() + clamp(cmd.vx_body - v_body.x(), -p.lp_max_accel * dt,
                           p.lp_max_accel * dt),
        v_body.y() + clamp(cmd.vy_body - v_body.y(), -p.lp_max_accel * dt,
                           p.lp_max_accel * dt));

    // radial speed limit
    const double speed = v_body_new.norm();
    const double cap = speed > p.lp_max_speed ? p.lp_max_speed / speed : 1.0;
    const Vec2d v_body_capped = v_body_new * cap;

    // yaw acceleration limit, then yaw rate max clamp
    double yaw_rate = s.yaw_rate + clamp(cmd.yaw_rate - s.yaw_rate,
                                         -p.lp_max_yaw_accel * dt,
                                         p.lp_max_yaw_accel * dt);
    yaw_rate = clamp(yaw_rate, -p.lp_max_yaw_rate, p.lp_max_yaw_rate);

    // body → world
    const Vec2d v_world(c * v_body_capped.x() - sn * v_body_capped.y(),
                        sn * v_body_capped.x() + c * v_body_capped.y());

    ns.position = s.position + v_world * dt;
    ns.yaw = wrapAngle(s.yaw + yaw_rate * dt);
    ns.velocity_world = v_world;
    ns.yaw_rate = yaw_rate;
    return ns;
}

}  // namespace expert
}  // namespace il_dataset
