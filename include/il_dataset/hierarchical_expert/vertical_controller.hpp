#pragma once
/// @file   vertical_controller.hpp
/// @brief  VerticalController — the vertical (z) channel of the 3D expert,
///         separated from the planar planner (interface refactor).
///
/// Owns the vertical channel exclusively:
///   * altitude regulation toward the target altitude (vz intent),
///   * vertical reachability (acceleration / speed limits),
///   * flight-band safety bounds (floor / ceiling) with recovery from a
///     transient overshoot (no deadlock).
/// The planar (2D) planner NEVER sees the vertical channel; the composed
/// 3D command is built by CommandComposer3D from this controller's output.
///
/// Frame discipline: vz is ALWAYS BODY/FLU (+up).  For the level-flight
/// model used here body-up == world-up, so it integrates directly into the
/// world z.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace il_dataset {
namespace expert {

/// Vertical command / diagnostics produced by the VerticalController.
struct VerticalCommand {
    double vz_body = 0.0;         // executable BODY/FLU vertical velocity (m/s)
    double intent_vz_body = 0.0;  // long-term altitude-regulation intent (m/s)
    double z_min_m = std::numeric_limits<double>::quiet_NaN();
    double z_max_m = std::numeric_limits<double>::quiet_NaN();
    bool z_bounds_ok = true;
};

class VerticalController {
public:
    explicit VerticalController(const Params2D& p) : p_(p) {}

    /// vz intent (BODY/FLU +up) toward target_z (world z, m) with the
    /// vertical speed limit.
    double vzIntent(double z, double target_z) const {
        return clamp(p_.lp_vz_kp * (target_z - z), -p_.lp_max_vz,
                     p_.lp_max_vz);
    }

    /// Executable vz — COMMAND-RAMP relative to the PREVIOUS vz command so
    /// the vertical command can LEAD a recovery from a sink.  (State-pinning
    /// it to vz_world made the command follow the sink: when horizontal
    /// acceleration tilted the drone and it sank at ~0.8 m/s, the command
    /// was clamped to vz_world ± lp_max_v_accel*dt — i.e. still descending —
    /// and the drone dropped ~1.3 m before the disturbance ended, joint_v2
    /// episode 000000_3d0c3119.  The backend feedforward then delivers the
    /// ramp, same as the horizontal channel.)  Falls back to the state
    /// velocity when no previous command is available or it has diverged
    /// (teleport / reset without a planner reset).
    double executableVz(double prev_vz_cmd, double vz_world, double vz_intent,
                        double dt) const {
        const double dvz = std::max(1e-9, p_.lp_max_v_accel * dt);
        const bool diverged =
            std::fabs(prev_vz_cmd - vz_world) > 2.0 * p_.lp_max_vz;
        const double base = diverged ? vz_world : prev_vz_cmd;
        double vz = clamp(vz_intent, base - dvz, base + dvz);
        return clamp(vz, -p_.lp_max_vz, p_.lp_max_vz);
    }

    /// Clamp the executable vz so z stays inside the hard flight band
    /// [z_min, z_max] over one step.  A transient overshoot is allowed to
    /// recover back into the band (never a recovery deadlock).
    double clampToBand(double z, double vz, double dt) const {
        const double z_lo = p_.lp_z_min_m;
        const double z_hi = p_.lp_z_max_m;
        const double z_next = z + vz * dt;
        if (z_next < z_lo) {
            vz = std::max(vz, (z_lo - z) / std::max(1e-9, dt));
        }
        if (z_next > z_hi) {
            vz = std::min(vz, (z_hi - z) / std::max(1e-9, dt));
        }
        return vz;
    }

    /// Compute the full VerticalCommand for the current 3D state + target
    /// altitude.  `horizon_s` is used only for the predicted band
    /// diagnostics (z_min/z_max, z_bounds_ok).  `prev_vz_cmd` is the
    /// previous executable vz (the command-ramp base; see executableVz).
    VerticalCommand compute(const VehicleState3D& state, double target_z,
                            double horizon_s, double dt,
                            double prev_vz_cmd) const {
        const double z0 = state.position.z();
        const double vz0 = state.velocity_world.z();
        VerticalCommand out;
        out.intent_vz_body = vzIntent(z0, target_z);
        out.vz_body =
            executableVz(prev_vz_cmd, vz0, out.intent_vz_body, dt);
        out.vz_body = clampToBand(z0, out.vz_body, dt);

        // Predicted band over the horizon (diagnostics only).
        double zmin = z0, zmax = z0;
        bool ok = true;
        const double z_lo = p_.lp_z_min_m;
        const double z_hi = p_.lp_z_max_m;
        const double z_lo_inner = p_.lp_z_min_m + p_.lp_vertical_clearance_m;
        const double z_hi_inner = p_.lp_z_max_m - p_.lp_vertical_clearance_m;
        double zz = z0, vv = vz0;
        const double step = std::max(1e-3, dt);
        for (double t = step; t <= horizon_s + 1e-6; t += step) {
            const double intent = vzIntent(zz, target_z);
            vv = executableVz(vv, vv, intent, step);
            vv = clampToBand(zz, vv, step);
            zz += vv * step;
            zmin = std::min(zmin, zz);
            zmax = std::max(zmax, zz);
            if (zz < z_lo - 1e-9 || zz > z_hi + 1e-9) ok = false;
        }
        // The final altitude must converge inside the clearance-margin
        // inner band (transient overshoot recovery is allowed).
        if (zz < z_lo_inner - 1e-9 || zz > z_hi_inner + 1e-9) ok = false;
        out.z_min_m = zmin;
        out.z_max_m = zmax;
        out.z_bounds_ok = ok;
        return out;
    }

    /// Predicted altitude profile over [0, horizon] under the altitude
    /// regulator (used to assemble the canonical 3D trajectory).
    std::vector<double> predictZ(double z, double vz_world, double target_z,
                                 double horizon_s, double dt) const {
        std::vector<double> zs;
        zs.push_back(z);
        double zz = z, vv = vz_world;
        const double step = std::max(1e-3, dt);
        for (double t = step; t <= horizon_s + 1e-6; t += step) {
            const double intent = vzIntent(zz, target_z);
            vv = executableVz(vv, vv, intent, step);
            vv = clampToBand(zz, vv, step);
            zz += vv * step;
            zs.push_back(zz);
        }
        return zs;
    }

    bool zInBand(double z) const {
        return z >= p_.lp_z_min_m && z <= p_.lp_z_max_m;
    }

private:
    Params2D p_;
};

}  // namespace expert
}  // namespace il_dataset
