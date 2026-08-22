#pragma once
/// @file   kinematics.hpp
/// @brief  Shared deterministic kinematic integration (pure C++).
///
/// The SAME planar function is used by:
///   * the preflight simulator step — actual execution,
///   * LocalPlanner30Hz candidate prediction,
/// so prediction and execution can never diverge.  Every step applies:
///   * linear acceleration limit,
///   * radial speed limit (max_speed),
///   * yaw acceleration limit,
///   * yaw rate limit (max_yaw_rate).
///
/// Frame discipline: VelocityCommand3D is ALWAYS BODY/FLU
/// (forward / left / up).  The planar integrator uses only the horizontal
/// channels (vx_body / vy_body / yaw_rate); the 3D integrator additionally
/// integrates the vertical channel (vz_body, body-up == world-up at level)
/// into the world z.

#include "il_dataset/hierarchical_expert/types.hpp"

#include <cmath>

namespace il_dataset {
namespace expert {

/// Full 3D BODY/FLU velocity + yaw-rate command — the ONLY command type
/// that ever leaves the expert (composed by CommandComposer3D).
struct VelocityCommand3D {
    double vx_body = 0.0;  // forward  (+X body)
    double vy_body = 0.0;  // lateral  (+Y body = left, CCW)
    double vz_body = 0.0;  // vertical (+Z body = up, FLU)
    double yaw_rate = 0.0;
};

/// Advance the PLANAR (horizontal) state by dt under the limits above.
/// The vertical channel is ignored — it is owned by VerticalController /
/// CommandComposer3D, never by the planar planner.
inline PlanarState integratePlanarStep(const PlanarState& s,
                                       const VelocityCommand3D& cmd,
                                       double dt, const Params2D& p) {
    PlanarState ns;
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

/// Advance the canonical 3D state by dt (horizontal via the planar step,
/// vertical integrated from the BODY/FLU vz command with the vertical
/// acceleration / speed limits).  Used by the preflight simulator (and any
/// future 3D execution prediction).
inline VehicleState3D integrateVehicle3DStep(const VehicleState3D& s,
                                             const VelocityCommand3D& cmd,
                                             double dt, const Params2D& p) {
    VehicleState3D ns;
    const PlanarState ps = HorizontalProjection::state(s);
    const PlanarState psn = integratePlanarStep(ps, cmd, dt, p);
    ns.position = Vec3d(psn.position.x(), psn.position.y(), s.position.z());
    ns.velocity_world =
        Vec3d(psn.velocity_world.x(), psn.velocity_world.y(),
              s.velocity_world.z());
    ns.yaw = psn.yaw;
    ns.yaw_rate = psn.yaw_rate;
    ns.pitch = s.pitch;
    // vertical channel (body-up == world-up at level)
    double vz_new = s.velocity_world.z() +
                    clamp(cmd.vz_body - s.velocity_world.z(),
                          -p.lp_max_v_accel * dt, p.lp_max_v_accel * dt);
    vz_new = clamp(vz_new, -p.lp_max_vz, p.lp_max_vz);
    ns.position.z() = s.position.z() + vz_new * dt;
    ns.velocity_world.z() = vz_new;
    return ns;
}

}  // namespace expert
}  // namespace il_dataset
