#include "il_dataset/hierarchical_expert/flightmare_2d_observation.hpp"

#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

namespace {

/// Body->world rotation matrix from a [x, y, z, w] quaternion (flightlib
/// body frame: x right, y forward, z up).
inline void quatToRotation(const double q[4], double r[3][3]) {
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    const double n = std::sqrt(x * x + y * y + z * z + w * w);
    const double inv = n > 1e-12 ? 1.0 / n : 0.0;
    const double qx = x * inv, qy = y * inv, qz = z * inv, qw = w * inv;
    r[0][0] = 1 - 2 * (qy * qy + qz * qz);
    r[0][1] = 2 * (qx * qy - qz * qw);
    r[0][2] = 2 * (qx * qz + qy * qw);
    r[1][0] = 2 * (qx * qy + qz * qw);
    r[1][1] = 1 - 2 * (qx * qx + qz * qz);
    r[1][2] = 2 * (qy * qz - qx * qw);
    r[2][0] = 2 * (qx * qz - qy * qw);
    r[2][1] = 2 * (qy * qz + qx * qw);
    r[2][2] = 1 - 2 * (qx * qx + qy * qy);
}

/// out = M · v for a 3x3 matrix M and 3-vector v.
inline void applyR3(const double m[3][3], double x, double y, double z,
                    double out[3]) {
    out[0] = m[0][0] * x + m[0][1] * y + m[0][2] * z;
    out[1] = m[1][0] * x + m[1][1] * y + m[1][2] * z;
    out[2] = m[2][0] * x + m[2][1] * y + m[2][2] * z;
}

}  // namespace

// ────────────────────────────────────────────────────────────────────
//  CameraRig2D (shared by runtime observation + preflight simulator)
// ────────────────────────────────────────────────────────────────────
CameraRig2D::CameraRig2D(const Params2D& p, const double cam_pos[3],
                         const double cam_q[4]) {
    quatToRotation(cam_q, R_);
    // T_BC translation: Unity body [x right, y up, z forward] → flightlib
    // body [tx, tz, ty].
    const double t_fl[3] = {p.cam_t_bc_x, p.cam_t_bc_z, p.cam_t_bc_y};
    // R_bc_fl = P · R_unity · C (optical → flightlib body).  For the
    // default identity rotation: R_bc_fl·[x_opt,y_opt,d] = [x_opt,d,-y_opt].
    const auto* r = p.cam_r_bc.data();
    const int perm[3] = {0, 2, 1};  // flightlib axis i -> Unity axis perm[i]
    for (int i = 0; i < 3; ++i) {
        const int iu = perm[i];
        for (int j = 0; j < 3; ++j) {
            // C[j][j] = -1 for the optical Y axis (j == 1), else +1.
            const double s = (j == 1) ? -1.0 : 1.0;
            R_bc_fl_[i][j] = s * r[iu * 3 + j];
        }
    }
    // SINGLE application of the translation:
    //   cam_world = vehicle_world + R_WB · t_fl
    for (int i = 0; i < 3; ++i) {
        cam_world_[i] = cam_pos[i] + R_[i][0] * t_fl[0] +
                        R_[i][1] * t_fl[1] + R_[i][2] * t_fl[2];
    }
    // Camera FOV-centre heading INCLUDING R_BC:
    //   fwd_world = R_WB · (R_bc_fl · [0,0,1])
    double fwd_fl[3], fwd_world[3];
    applyR3(R_bc_fl_, 0.0, 0.0, 1.0, fwd_fl);
    applyR3(R_, fwd_fl[0], fwd_fl[1], fwd_fl[2], fwd_world);
    cam_yaw_world_ = std::atan2(fwd_world[1], fwd_world[0]);
}

void CameraRig2D::rayWorldDirXY(double bearing, double out[2]) const {
    // rotZ(bearing)·[0,0,1] = [sin(bearing), 0, cos(bearing)] (camera frame).
    const double dc[3] = {std::sin(bearing), 0.0, std::cos(bearing)};
    double dfl[3], dw[3];
    applyR3(R_bc_fl_, dc[0], dc[1], dc[2], dfl);
    applyR3(R_, dfl[0], dfl[1], dfl[2], dw);
    const double n = std::hypot(dw[0], dw[1]);
    if (n > 1e-12) {
        out[0] = dw[0] / n;
        out[1] = dw[1] / n;
    } else {
        out[0] = std::cos(cam_yaw_world_ + bearing);
        out[1] = std::sin(cam_yaw_world_ + bearing);
    }
}

void CameraRig2D::applyR(double x, double y, double z, double out[3]) const {
    applyR3(R_, x, y, z, out);
}

void CameraRig2D::applyRbc(double x, double y, double z, double out[3]) const {
    applyR3(R_bc_fl_, x, y, z, out);
}

void CameraRig2D::quatFromExpertYaw(double yaw_expert, double out[4]) {
    // For a level body with expert yaw ψ: the flightlib body→world rotation
    // is rotZ(φ) with φ = ψ - π/2 (derived so body-forward (0,1,0) maps to
    // the expert forward (cos ψ, sin ψ)).
    const double phi = yaw_expert - M_PI / 2.0;
    out[0] = 0.0;
    out[1] = 0.0;
    out[2] = std::sin(phi * 0.5);
    out[3] = std::cos(phi * 0.5);
}

// ────────────────────────────────────────────────────────────────────
//  Flightmare2DObservation
// ────────────────────────────────────────────────────────────────────
LocalObservation Flightmare2DObservation::build(
    const std::vector<float>& depth_m, int width, int height,
    const double cam_pos[3], const double cam_q[4],
    const Vec2d& min_bounds, uint64_t tick) const {
    const double range = p_.obs_range_m;
    const double fov = deg2rad(p_.obs_fov_deg);

    CameraRig2D rig(p_, cam_pos, cam_q);

    if (width <= 0 || height <= 0 ||
        depth_m.size() < static_cast<size_t>(width) * height) {
        return buildFromRays({}, {}, cam_pos, cam_q, min_bounds, tick);
    }

    // ── Focal length (horizontal) from Params2D FOV and image width. ──
    const double cx = 0.5 * static_cast<double>(width - 1);
    const double cy = 0.5 * static_cast<double>(height - 1);
    const double fx = (0.5 * width) / std::max(1e-9, std::tan(0.5 * fov));
    const double fy = fx;

    // ── Per-ray nearest-hit accumulation (discretised horizontal rays) ──
    const double ray_da = deg2rad(p_.obs_ray_angular_res_deg);
    const int n_rays = std::max(
        1, static_cast<int>(std::ceil(fov / std::max(1e-9, ray_da))));
    std::vector<double> ray_hit(n_rays + 1,
                                std::numeric_limits<double>::infinity());
    std::vector<bool> ray_seen(n_rays + 1, false);

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            const float d_raw = depth_m[static_cast<size_t>(v) * width + u];
            // Invalid depth (0 / NaN / Inf / near-plane clip) → UNKNOWN.
            if (!std::isfinite(static_cast<double>(d_raw))) continue;
            const double d = static_cast<double>(d_raw);
            if (d <= 1e-3) continue;
            // Image optical frame: X right, Y down (flipud applied), Z fwd.
            const double x_opt = (u - cx) * d / fx;
            const double y_opt = (v - cy) * d / fy;
            // point_optical -> flightlib body (R_bc_fl, NO t_fl) -> world.
            double p_fl[3], p_world[3];
            rig.applyRbc(x_opt, y_opt, d, p_fl);
            rig.applyR(p_fl[0], p_fl[1], p_fl[2], p_world);
            // rel = point_world - cam_world = R_WB · (R_bc_fl · p_opt).
            const double rel_x = p_world[0];
            const double rel_y = p_world[1];
            const double horiz = std::hypot(rel_x, rel_y);
            const double bearing = wrapAngle(
                std::atan2(rel_y, rel_x) - rig.yawWorld());
            if (std::fabs(bearing) > fov / 2.0 + 1e-9) continue;
            int ri = static_cast<int>(std::floor(
                (bearing + fov / 2.0) / (fov / n_rays)));
            ri = clamp(ri, 0, n_rays);
            if (horiz > range) {
                ray_seen[ri] = true;  // valid no-hit beyond R
                continue;
            }
            ray_seen[ri] = true;
            if (horiz > 1e-6 && horiz < ray_hit[ri]) ray_hit[ri] = horiz;
        }
    }

    return buildFromRays(ray_hit, ray_seen, cam_pos, cam_q, min_bounds, tick);
}

LocalObservation Flightmare2DObservation::buildFromRays(
    const std::vector<double>& ray_hit, const std::vector<bool>& ray_seen,
    const double cam_pos[3], const double cam_q[4],
    const Vec2d& min_bounds, uint64_t tick) const {
    const double res = p_.obs_resolution;
    const double range = p_.obs_range_m;
    const double fov = deg2rad(p_.obs_fov_deg);

    CameraRig2D rig(p_, cam_pos, cam_q);
    const Vec2d cam_pos2(rig.worldX(), rig.worldY());

    // ── GRID-ALIGNED PATCH (same contract as the debug FovRaycaster2D) ──
    const double cov_eps = 1e-6;
    const GridIndex2D lo = worldToGrid(cam_pos2 - Vec2d(range, range),
                                       min_bounds, res);
    const GridIndex2D hi = worldToGrid(
        cam_pos2 + Vec2d(range + cov_eps, range + cov_eps), min_bounds, res);

    LocalObservation patch;
    patch.resolution = res;
    patch.width = (hi.ix - lo.ix) + 1;
    patch.height = (hi.iy - lo.iy) + 1;
    patch.origin = Vec2d(min_bounds.x() + lo.ix * res,
                         min_bounds.y() + lo.iy * res);
    patch.cells.assign(static_cast<size_t>(patch.width) * patch.height,
                       CellState::UNKNOWN);
    patch.age_ticks.assign(static_cast<size_t>(patch.width) * patch.height, 0);
    patch.max_age_ticks = 1;
    patch.tick = tick;

    auto patchCell = [&](double wx, double wy, CellState s) {
        const GridIndex2D g = worldToGrid(Vec2d(wx, wy), patch.origin, res);
        if (!patch.inGrid(g.ix, g.iy)) return;
        patch.cells[patch.idx(g.ix, g.iy)] = s;
    };

    const double ray_da = deg2rad(p_.obs_ray_angular_res_deg);
    const int n_rays = std::max(
        1, static_cast<int>(std::ceil(fov / std::max(1e-9, ray_da))));
    if (static_cast<int>(ray_hit.size()) != n_rays + 1 ||
        static_cast<int>(ray_seen.size()) != n_rays + 1) {
        return patch;  // mismatched ray arrays → all UNKNOWN
    }

    // ── Pass 1: ray march from the CAMERA origin along the world XY ray
    //    directions (INCLUDING R_BC). ────────────────────────────────
    const double march_step = res * 0.5;
    for (int i = 0; i <= n_rays; ++i) {
        if (!ray_seen[i]) continue;
        const double bearing =
            -fov / 2.0 + fov * (static_cast<double>(i) / n_rays);
        double dir[2];
        rig.rayWorldDirXY(bearing, dir);
        const double h = ray_hit[i];
        if (std::isfinite(h)) {
            for (double d = march_step; d <= h - 1e-6; d += march_step) {
                patchCell(cam_pos2.x() + dir[0] * d,
                          cam_pos2.y() + dir[1] * d, CellState::FREE);
            }
            patchCell(cam_pos2.x() + dir[0] * h,
                      cam_pos2.y() + dir[1] * h, CellState::OCCUPIED);
        } else {
            // Valid no-hit: FREE to R, no OCCUPIED endpoint.
            for (double d = march_step; d <= range; d += march_step) {
                patchCell(cam_pos2.x() + dir[0] * d,
                          cam_pos2.y() + dir[1] * d, CellState::FREE);
            }
        }
    }

    // ── Pass 2: cell-wise classification for still-UNKNOWN cells inside
    //    the FOV wedge and range (occlusion by the nearest surface). ──
    for (int iy = 0; iy < patch.height; ++iy) {
        for (int ix = 0; ix < patch.width; ++ix) {
            const size_t id = patch.idx(ix, iy);
            if (patch.cells[id] != CellState::UNKNOWN) continue;
            const Vec2d cw = gridCellCenter(ix, iy, patch.origin, res);
            const Vec2d rel = cw - cam_pos2;
            const double dist = rel.norm();
            if (dist > range + 0.5 * res) continue;
            if (dist < 1e-9) {
                patch.cells[id] = CellState::FREE;  // sensor-origin cell
                continue;
            }
            const double bearing = wrapAngle(
                std::atan2(rel.y(), rel.x()) - rig.yawWorld());
            if (std::fabs(bearing) > fov / 2.0 + 1e-9) continue;
            const int ri = clamp(
                static_cast<int>(std::floor(
                    (bearing + fov / 2.0) / (fov / n_rays))),
                0, n_rays);
            const double nearest = ray_hit[ri];
            if (std::isfinite(nearest) && nearest < dist - 1e-3) {
                continue;  // occluded → stays UNKNOWN
            }
            patch.cells[id] = CellState::FREE;
        }
    }

    // ── Sensor-origin cell is known usable. ─────────────────────────
    const GridIndex2D veh = worldToGrid(cam_pos2, patch.origin, res);
    if (patch.inGrid(veh.ix, veh.iy)) {
        const size_t vid = patch.idx(veh.ix, veh.iy);
        patch.cells[vid] = CellState::FREE;
        patch.age_ticks[vid] = 0;
    }

    return patch;
}

}  // namespace expert
}  // namespace il_dataset
