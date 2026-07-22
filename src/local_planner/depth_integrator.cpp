#include "il_dataset/local_planner/depth_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace il_dataset {

namespace {

// Inline helper: world → integer grid index (floor).
inline void world_to_grid(double wx, double wy, double wz,
                          double ox, double oy, double oz,
                          double inv_res,
                          int& ix, int& iy, int& iz) {
    ix = static_cast<int>(std::floor((wx - ox) * inv_res));
    iy = static_cast<int>(std::floor((wy - oy) * inv_res));
    iz = static_cast<int>(std::floor((wz - oz) * inv_res));
}

// Inline helper: check bounds.
inline bool in_bounds(int ix, int iy, int iz,
                      int gx, int gy, int gz) {
    return (static_cast<unsigned>(ix) < static_cast<unsigned>(gx)) &&
           (static_cast<unsigned>(iy) < static_cast<unsigned>(gy)) &&
           (static_cast<unsigned>(iz) < static_cast<unsigned>(gz));
}

}  // anonymous namespace

int integrate_depth(py::array_t<double> points_world,
                    py::array_t<double> points_cam_z,
                    py::array_t<double> cam_pos,
                    py::array_t<std::uint8_t> occ_grid,
                    py::array_t<double> last_obs_time,
                    double occ_endpoint_margin,
                    double free_space_spacing,
                    double resolution,
                    double max_depth_m,
                    double timestamp_s,
                    py::array_t<double> origin,
                    py::array_t<int> grid_dims) {
    // ── Unpack array buffers ──────────────────────────────────────
    auto pw = points_world.unchecked<2>();
    auto pz = points_cam_z.unchecked<1>();
    auto cp = cam_pos.unchecked<1>();
    auto occ = occ_grid.mutable_unchecked<3>();
    auto lot = last_obs_time.mutable_unchecked<3>();
    auto orig = origin.unchecked<1>();
    auto dims = grid_dims.unchecked<1>();

    const int N = static_cast<int>(pw.shape(0));
    const double ox = orig(0), oy = orig(1), oz = orig(2);
    const int gx = dims(0), gy = dims(1), gz = dims(2);
    const double inv_res = 1.0 / resolution;
    const double cam_x = cp(0), cam_y = cp(1), cam_z = cp(2);

    // Effective max depth for occupancy marking
    const double occ_max_depth = max_depth_m - std::max(occ_endpoint_margin, 1e-3);

    // Ensure free_space_spacing is at least 1 voxel
    const double eff_spacing = std::max(free_space_spacing, resolution);

    int changed = 0;

    // ── Pass 1: mark occupied endpoints ──────────────────────────
    for (int k = 0; k < N; ++k) {
        // Skip max-range (no-return) rays for occupancy marking
        if (pz(k) >= occ_max_depth) continue;

        int ix, iy, iz;
        world_to_grid(pw(k, 0), pw(k, 1), pw(k, 2),
                      ox, oy, oz, inv_res, ix, iy, iz);
        if (!in_bounds(ix, iy, iz, gx, gy, gz)) continue;

        if (occ(ix, iy, iz) != OCCUPIED) {
            occ(ix, iy, iz) = OCCUPIED;
            ++changed;
        }
        lot(ix, iy, iz) = timestamp_s;
    }

    // ── Pass 2: mark free space along rays ───────────────────────
    for (int k = 0; k < N; ++k) {
        const double ex = pw(k, 0), ey = pw(k, 1), ez = pw(k, 2);
        double dx = ex - cam_x;
        double dy = ey - cam_y;
        double dz = ez - cam_z;
        double ray_len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (ray_len < 1e-6) continue;

        const double inv_len = 1.0 / ray_len;
        dx *= inv_len;
        dy *= inv_len;
        dz *= inv_len;

        // Effective length: stop before the occupied endpoint
        const double effective_len = ray_len - occ_endpoint_margin;
        const int n_samples = std::max(1,
            static_cast<int>(effective_len / eff_spacing));

        for (int s = 0; s < n_samples; ++s) {
            const double frac = static_cast<double>(s) /
                                static_cast<double>(n_samples);
            const double px = cam_x + frac * dx * effective_len;
            const double py = cam_y + frac * dy * effective_len;
            const double pz_val = cam_z + frac * dz * effective_len;

            int ix, iy, iz;
            world_to_grid(px, py, pz_val, ox, oy, oz, inv_res, ix, iy, iz);
            if (!in_bounds(ix, iy, iz, gx, gy, gz)) continue;

            // Occupy-first policy: only set FREE if currently UNKNOWN
            if (occ(ix, iy, iz) == UNKNOWN) {
                occ(ix, iy, iz) = FREE;
                lot(ix, iy, iz) = timestamp_s;
                ++changed;
            }
        }
    }

    return changed;
}

}  // namespace il_dataset
