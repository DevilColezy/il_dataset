#pragma once
/// @file   flightmare_2d_observation.hpp
/// @brief  Builds the instantaneous 2D FOV patch from a real Flightmare
///         depth frame (the Flightmare "sensor synthesis" stage).
///
/// This replaces the debug harness's truth-based FovRaycaster2D: the
/// patch is produced ONLY from the current depth frame + camera pose.
///   * cells inside the current horizontal FOV wedge and within the
///     perception range R are classified from the depth rays;
///   * 0 < depth < R: FREE before the first surface, OCCUPIED at the
///     surface, UNKNOWN behind it (occlusion);
///   * valid no-hit / depth >= R: the ray is FREE all the way to R and
///     NO OCCUPIED endpoint is produced (an open FOV never stays UNKNOWN);
///   * invalid depth (0 / NaN / Inf) stays UNKNOWN;
///   * cells outside the FOV stay UNKNOWN;
///   * no truth / PLY / ESDF data is ever used here.
///
/// ═══════════════════════════════════════════════════════════════════
///  COORDINATE CONTRACT (single, unambiguous)
/// ═══════════════════════════════════════════════════════════════════
///   input pose  : cam_pos = VEHICLE world position [x, y, z] in the
///                 Flightmare world (x-fwd, y-left, z-up);
///                 cam_q    = VEHICLE body→world quaternion [x, y, z, w]
///                 in the FLIGHTLIB body frame.
///   flightlib body : x right, y forward, z up.
///   Unity body     : x right, y up, z forward   (permutation P: [x,y,z]
///                 → flightlib [x, z, y]).
///   Unity camera   : x right, y up, z forward (= image optical with the
///                 vertical axis flipped; C = diag(1,-1,1)).
///   image optical  : x right, y DOWN, z forward.  The Unity image is
///                 vertically flipped by Python (flipud(raw*100)) so the
///                 C++ receives `depth_m` in the SAME optical convention:
///                 row v=0 is the top, y_opt = (v - cy)·d/fy points DOWN.
///   T_BC (depth.t_bc, row-major 4x4) : camera → Unity body.
///                 translation t_u = [tx, ty, tz] in Unity body coords
///                 → flightlib body t_fl = [tx, tz, ty];
///                 rotation R_unity (row-major 3x3, camera→Unity body)
///                 → optical→flightlib-body R_bc_fl = P·R_unity·C.
///                 For the default identity rotation this reduces exactly
///                 to  R_bc_fl · [x_opt, y_opt, d] = [x_opt, d, -y_opt].
///
///   SINGLE EXTRINSIC FORMULA (the translation is applied EXACTLY once):
///     cam_world   = vehicle_world + R_WB · t_fl
///     point_world = cam_world + R_WB · (R_bc_fl · point_optical)
///   Equivalently point_world = vehicle_world + R_WB · (t_fl +
///   R_bc_fl·point_optical).  R_WB is the vehicle body→world rotation from
///   cam_q.  R_bc_fl is applied BEFORE R_WB, and t_fl is NOT re-added to
///   the per-pixel body vector.
///
///   CAMERA WORLD ORIENTATION INCLUDES R_BC: the camera FOV centre heading
///   is the XY heading of `R_WB · R_bc_fl · [0,0,1]` (not merely the
///   vehicle yaw), and every ray direction is `R_WB · R_bc_fl ·
///   rotZ(bearing)·[0,0,1]` projected onto the world XY plane.
///
///   The runtime observation (here) and the PreflightSimulator use the
///   SAME camera origin, orientation, FOV, range and ray semantics: the
///   preflight synthesizes ray hits from the camera (not the drone centre)
///   and feeds them through buildFromRays().
///
///   FOV / range / resolution come ONLY from Params2D (obs_fov_deg /
///   obs_range_m / obs_resolution) — never from a per-call argument.
///
/// GRID ALIGNMENT: the returned patch shares the EXACT global grid of
/// ObservedGrid2D — its origin is snapped to a global grid boundary
/// derived from `min_bounds + n*resolution` (never position - range).

#include "il_dataset/hierarchical_expert/types.hpp"

#include <cstdint>
#include <vector>

namespace il_dataset {
namespace expert {

/// Reusable camera rig (the SINGLE place the Unity T_BC extrinsic and the
/// camera world orientation — INCLUDING R_BC — are computed).  Shared by
/// the runtime observation builder and the PreflightSimulator so both use
/// the identical camera origin, orientation, FOV and ray semantics.
class CameraRig2D {
public:
    /// Build from a VEHICLE world position + flightlib body→world
    /// quaternion.  cam_world = vehicle + R_WB·t_fl (translation applied
    /// EXACTLY once); cam_yaw_world is the XY heading of
    /// R_WB·(R_bc_fl·[0,0,1]) (the camera FOV centre, INCLUDING R_BC).
    CameraRig2D(const Params2D& p, const double cam_pos[3],
                const double cam_q[4]);

    double worldX() const { return cam_world_[0]; }
    double worldY() const { return cam_world_[1]; }
    const double* camWorld() const { return cam_world_; }
    /// Camera world yaw in the expert frame (rad), INCLUDING R_BC.
    double yawWorld() const { return cam_yaw_world_; }

    /// World XY unit direction of a camera-frame ray at `bearing` (rad,
    /// CCW from the camera forward): R_WB·(R_bc_fl·rotZ(bearing)·[0,0,1])
    /// projected onto the world XY plane.
    void rayWorldDirXY(double bearing, double out[2]) const;

    /// Apply the vehicle body→world rotation R_WB to a body vector.
    void applyR(double x, double y, double z, double out[3]) const;
    /// Apply the optical→flightlib-body rotation R_bc_fl to an optical
    /// vector (the T_BC rotation, WITHOUT the translation).
    void applyRbc(double x, double y, double z, double out[3]) const;

    /// Flightlib body→world quaternion [x,y,z,w] for a level body whose
    /// nose points at expert yaw `yaw_expert` (used by the preflight sim,
    /// which integrates in the expert 2D frame).
    static void quatFromExpertYaw(double yaw_expert, double out[4]);

private:
    double R_[3][3];
    double R_bc_fl_[3][3];
    double cam_world_[3];
    double cam_yaw_world_ = 0.0;
};

class Flightmare2DObservation {
public:
    explicit Flightmare2DObservation(const Params2D& p) : p_(p) {}

    /// Build a three-state current-patch LocalObservation from a depth
    /// frame (HxW, float32, metres — 0 / NaN / inf treated as no return).
    ///
    /// @param depth_m       row-major depth image [height*width].
    /// @param width,height  image size.
    /// @param cam_pos       VEHICLE world position [x, y, z] (Flightmare).
    /// @param cam_q         vehicle body->world quaternion [x, y, z, w] in
    ///                      the FLIGHTLIB body frame.
    /// @param min_bounds    global grid anchor (scene min bounds XY).
    /// @param tick          30 Hz tick (stamped into the patch).
    LocalObservation build(const std::vector<float>& depth_m, int width,
                           int height, const double cam_pos[3],
                           const double cam_q[4],
                           const Vec2d& min_bounds, uint64_t tick) const;

    /// Build a patch directly from precomputed per-ray hits (used by the
    /// PreflightSimulator with the SAME camera pose/extrinsic/FOV/range and
    /// ray semantics).  `ray_hit[i]` = horizontal range to the first
    /// surface along ray i (infinity = valid no-hit), `ray_seen[i]` =
    /// the ray had at least one valid return.  Sizes must be n_rays+1 with
    /// n_rays = ceil(fov/ray_angular_res).
    LocalObservation buildFromRays(const std::vector<double>& ray_hit,
                                   const std::vector<bool>& ray_seen,
                                   const double cam_pos[3],
                                   const double cam_q[4],
                                   const Vec2d& min_bounds,
                                   uint64_t tick) const;

private:
    Params2D p_;
};

}  // namespace expert
}  // namespace il_dataset
