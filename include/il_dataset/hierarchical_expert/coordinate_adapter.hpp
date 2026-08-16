#pragma once
/// @file   coordinate_adapter.hpp
/// @brief  THE single coordinate adaptation layer between the Flightmare
///         FLU world (il_dataset convention B) and the 2D expert frame.
///
/// This is the ONLY place in the whole package that converts between the
/// two conventions.  Python (via the pybind module), the C++ observation
/// builder and the preflight simulator all call these helpers — nobody
/// hand-writes a symbol conversion anywhere else.
///
/// ── Flightmare / il_dataset convention (production "B") ─────────────
///   world   : x-fwd, y-left, z-up (Flightmare world)
///   body    : FLU (+x forward, +y left, +z up)
///   yaw     : forward_world(yaw) = (-sin yaw,  cos yaw)
///             left_world(yaw)    = (-cos yaw, -sin yaw)
///   yaw     = atan2(world_y, world_x) - pi/2   (yaw=0 ⇒ nose faces +Y)
///   yaw left turn is positive.
///
/// ── Expert 2D convention (internal, debug-package style) ────────────
///   world   : X → right, Y → up (standard 2D math frame over the same
///             horizontal plane)
///   body    : +X forward (nose), +Y left (90° CCW from nose)
///   yaw     : forward_world(psi) = (cos psi, sin psi)
///
/// ── Derivation (verified) ───────────────────────────────────────────
///   cos(psi) = cos(yaw+pi/2) = -sin yaw   ✓
///   sin(psi) = sin(yaw+pi/2) =  cos yaw   ✓
///   ⇒ expert_yaw = flightmare_yaw + pi/2.
///   Under this substitution the body frames coincide (+X fwd, +Y left),
///   so world positions, body FLU velocities, yaw rates and body commands
///   are all IDENTITY between the two frames.  Only the yaw zero offset
///   differs.

#include "il_dataset/hierarchical_expert/types.hpp"

#include <cmath>

namespace il_dataset {
namespace expert {

class CoordinateAdapter {
public:
    /// Flightmare yaw (convention B) → expert 2D yaw (CCW from world X).
    static double flightmareYawToExpert(double yaw_fm) {
        return yaw_fm + M_PI / 2.0;
    }

    /// Expert 2D yaw → Flightmare yaw (convention B).
    static double expertYawToFlightmare(double yaw_expert) {
        return yaw_expert - M_PI / 2.0;
    }

    /// World XY velocity → FLU XY velocity (identity, level flight).
    static Vec2d worldVelToFluXY(const Vec2d& v_world, double yaw_fm) {
        // Body +X forward = (-sin yaw, cos yaw), body +Y left =
        // (-cos yaw, -sin yaw) in the Flightmare world frame.
        const double c = std::cos(yaw_fm), s = std::sin(yaw_fm);
        return Vec2d(-s * v_world.x() + c * v_world.y(),
                     -c * v_world.x() - s * v_world.y());
    }

    /// FLU XY velocity → world XY velocity (identity, level flight).
    static Vec2d fluXYToWorldVel(const Vec2d& v_flu, double yaw_fm) {
        const double c = std::cos(yaw_fm), s = std::sin(yaw_fm);
        return Vec2d(-s * v_flu.x() - c * v_flu.y(),
                     c * v_flu.x() - s * v_flu.y());
    }

    /// Expert body-frame command (vx, vy, yaw_rate) → Flightmare FLU
    /// command.  Both frames are +X forward / +Y left, so this is the
    /// identity; documented here so the mapping has exactly one home.
    static void expertCommandToFlu(double vx_body, double vy_body,
                                   double yaw_rate,
                                   double& vx_flu, double& vy_flu,
                                   double& yaw_rate_flu) {
        vx_flu = vx_body;
        vy_flu = vy_body;
        yaw_rate_flu = yaw_rate;
    }

    /// Unit body bearing (rad, CCW + = LEFT) of a world point seen from a
    /// level vehicle with Flightmare yaw `yaw_fm`.  Uses the expert frame
    /// internally (identical geometry) and returns the FLU bearing.
    static double bearingToWorldPoint(const Vec2d& world_point,
                                      const Vec2d& position,
                                      double yaw_fm) {
        const Vec2d rel = world_point - position;
        const double expert_yaw = flightmareYawToExpert(yaw_fm);
        return wrapAngle(std::atan2(rel.y(), rel.x()) - expert_yaw);
    }
};

}  // namespace expert
}  // namespace il_dataset
