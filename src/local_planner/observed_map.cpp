#include "il_dataset/local_planner/observed_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1.0e-9;

// ── Separable exact squared Euclidean distance transform ─────────────
// Felzenszwalb & Huttenlocher (2012) 1-D quadratic lower-envelope pass.
// `f` is a vector of squared distances (0 for seeds, +inf otherwise).
void edt1d(std::vector<double>& f) {
    const int n = static_cast<int>(f.size());
    if (n == 0) return;
    std::vector<int> v(n, 0);
    std::vector<double> z(n + 1, 0.0);
    int k = 0;
    v[0] = 0;
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();
    for (int q = 1; q < n; ++q) {
        double s = ((f[q] + static_cast<double>(q) * q) -
                    (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
                   (2.0 * static_cast<double>(q - v[k]));
        while (s <= z[k]) {
            --k;
            s = ((f[q] + static_cast<double>(q) * q) -
                 (f[v[k]] + static_cast<double>(v[k]) * v[k])) /
                (2.0 * static_cast<double>(q - v[k]));
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = std::numeric_limits<double>::infinity();
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < static_cast<double>(q)) ++k;
        const double dx = static_cast<double>(q - v[k]);
        f[q] = dx * dx + f[v[k]];
    }
}

/// Exact 3-D squared EDT.  `seed` marks cells at distance 0 (true).
/// Boundary faces are treated as infinite (no erosion from the map edge).
std::vector<double> squaredEdt3d(const std::vector<std::uint8_t>& seed,
                                 int gx, int gy, int gz) {
    auto at = [&](int ix, int iy, int iz) {
        return (ix * gy + iy) * gz + iz;
    };
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> f(static_cast<size_t>(gx) * gy * gz, inf);
    for (int ix = 0; ix < gx; ++ix) {
        for (int iy = 0; iy < gy; ++iy) {
            for (int iz = 0; iz < gz; ++iz) {
                if (seed[static_cast<size_t>(at(ix, iy, iz))] != 0) {
                    f[static_cast<size_t>(at(ix, iy, iz))] = 0.0;
                }
            }
        }
    }
    // Pass along x.
    for (int iy = 0; iy < gy; ++iy) {
        for (int iz = 0; iz < gz; ++iz) {
            std::vector<double> column(static_cast<size_t>(gx), inf);
            for (int ix = 0; ix < gx; ++ix) {
                column[static_cast<size_t>(ix)] =
                    f[static_cast<size_t>(at(ix, iy, iz))];
            }
            edt1d(column);
            for (int ix = 0; ix < gx; ++ix) {
                f[static_cast<size_t>(at(ix, iy, iz))] =
                    column[static_cast<size_t>(ix)];
            }
        }
    }
    // Pass along y.
    for (int ix = 0; ix < gx; ++ix) {
        for (int iz = 0; iz < gz; ++iz) {
            std::vector<double> column(static_cast<size_t>(gy), inf);
            for (int iy = 0; iy < gy; ++iy) {
                column[static_cast<size_t>(iy)] =
                    f[static_cast<size_t>(at(ix, iy, iz))];
            }
            edt1d(column);
            for (int iy = 0; iy < gy; ++iy) {
                f[static_cast<size_t>(at(ix, iy, iz))] =
                    column[static_cast<size_t>(iy)];
            }
        }
    }
    // Pass along z.
    for (int ix = 0; ix < gx; ++ix) {
        for (int iy = 0; iy < gy; ++iy) {
            std::vector<double> column(static_cast<size_t>(gz), inf);
            for (int iz = 0; iz < gz; ++iz) {
                column[static_cast<size_t>(iz)] =
                    f[static_cast<size_t>(at(ix, iy, iz))];
            }
            edt1d(column);
            for (int iz = 0; iz < gz; ++iz) {
                f[static_cast<size_t>(at(ix, iy, iz))] =
                    column[static_cast<size_t>(iz)];
            }
        }
    }
    return f;
}

}  // namespace

ObservedMap::ObservedMap(const ObservedMapConfig& config) : config_(config) {}

std::int64_t ObservedMap::indexOf(int ix, int iy, int iz) const {
    if (ix < 0 || ix >= gx_ || iy < 0 || iy >= gy_ || iz < 0 || iz >= gz_) {
        return -1;
    }
    return (static_cast<std::int64_t>(ix) * gy_ + iy) * gz_ + iz;
}

void ObservedMap::reset(const Eigen::Vector3d& center_world) {
    center_world_ = center_world;
    origin_world_ = center_world_ -
                    Eigen::Vector3d(config_.size_x_m * 0.5,
                                    config_.size_y_m * 0.5,
                                    config_.size_z_m * 0.5);
    gx_ = static_cast<int>(std::ceil(config_.size_x_m / config_.resolution));
    gy_ = static_cast<int>(std::ceil(config_.size_y_m / config_.resolution));
    gz_ = static_cast<int>(std::ceil(config_.size_z_m / config_.resolution));
    if (gx_ % 2 == 0) ++gx_;
    if (gy_ % 2 == 0) ++gy_;
    if (gz_ % 2 == 0) ++gz_;
    occ_.assign(static_cast<size_t>(gx_) * gy_ * gz_, UNKNOWN);
    last_obs_time_.assign(static_cast<size_t>(gx_) * gy_ * gz_, -1.0);
    known_mask_.assign(static_cast<size_t>(gx_) * gy_ * gz_, 0);
    esdf_.assign(static_cast<size_t>(gx_) * gy_ * gz_, 0.0f);
    revision_ = 0;
    esdf_revision_ = -1;
    frames_since_esdf_ = 0;
    esdf_built_ = false;
    initialized_ = true;
}

bool ObservedMap::recenterIfNeeded(const Eigen::Vector3d& center_world) {
    if (!initialized_) {
        reset(center_world);
        return true;
    }
    const double displacement = (center_world - center_world_).norm();
    if (displacement > config_.recenter_threshold_m) {
        reset(center_world);
        return true;
    }
    return false;
}

bool ObservedMap::inBounds(double x, double y, double z) const {
    if (!initialized_) return false;
    const Eigen::Vector3d g = worldToGrid(Eigen::Vector3d(x, y, z));
    return g.x() >= 0.0 && g.x() < gx_ && g.y() >= 0.0 && g.y() < gy_ &&
           g.z() >= 0.0 && g.z() < gz_;
}

BrakeRiskResult ObservedMap::sweptBrakeRisk(
    const VehicleState& state,
    double reaction_delay_s,
    double deceleration_mps2,
    double body_radius_m,
    double safety_margin_m,
    double sample_spacing_m) const {
    BrakeRiskResult result;
    if (!esdf_built_ || !state.position.allFinite() ||
        !state.velocity.allFinite()) {
        result.risk = true;
        result.min_clearance = 0.0;
        return result;
    }
    const double required_clearance = body_radius_m + safety_margin_m;
    const double speed = state.velocity.norm();
    Eigen::Vector3d vel_dir(Eigen::Vector3d::Zero());
    if (speed > kEpsilon) vel_dir = state.velocity / speed;

    // Reaction phase: constant velocity.
    const double reaction_distance = speed * std::max(0.0, reaction_delay_s);
    // Braking phase: uniform deceleration from `speed` to rest.
    const double braking_distance =
        deceleration_mps2 > kEpsilon
            ? speed * speed / (2.0 * deceleration_mps2)
            : 0.0;
    result.braking_distance = reaction_distance + braking_distance;
    const double total_distance = result.braking_distance;

    const int reaction_samples = std::max(
        1, static_cast<int>(std::ceil(
               reaction_distance / std::max(0.02, sample_spacing_m))));
    const int brake_samples = std::max(
        1, static_cast<int>(std::ceil(
               braking_distance / std::max(0.02, sample_spacing_m))));

    double min_clearance = std::numeric_limits<double>::infinity();
    double first_risk_time = -1.0;

    auto probe = [&](const Eigen::Vector3d& point, double t) {
        const bool known =
            isKnown(point.x(), point.y(), point.z());
        const double clearance = esdfValue(point.x(), point.y(), point.z());
        if (std::isfinite(clearance)) {
            min_clearance = std::min(min_clearance, clearance);
        }
        if (!known || !std::isfinite(clearance) ||
            clearance < required_clearance) {
            if (first_risk_time < 0.0) first_risk_time = t;
            result.risk = true;
        }
    };

    // Reaction phase (constant velocity), times [0, reaction_delay_s].
    for (int i = 0; i <= reaction_samples; ++i) {
        const double s = reaction_distance * i / reaction_samples;
        const double t = reaction_delay_s * i / reaction_samples;
        probe(state.position + vel_dir * s, t);
    }
    // Braking phase (deceleration), times [reaction_delay_s, t_stop].
    const double brake_duration =
        deceleration_mps2 > kEpsilon ? speed / deceleration_mps2 : 0.0;
    for (int i = 1; i <= brake_samples; ++i) {
        const double s = reaction_distance +
                         braking_distance * i / brake_samples;
        const double f = static_cast<double>(i) / brake_samples;
        const double t = reaction_delay_s + brake_duration * f;
        probe(state.position + vel_dir * s, t);
    }

    result.min_clearance =
        std::isfinite(min_clearance) ? min_clearance : 0.0;
    return result;
}

Eigen::Vector3d ObservedMap::worldToGrid(const Eigen::Vector3d& world) const {
    return (world - origin_world_) / config_.resolution;
}

Eigen::Vector3i ObservedMap::worldToGridInt(const Eigen::Vector3d& world) const {
    return worldToGrid(world).array().floor().cast<int>();
}

Eigen::Vector3d ObservedMap::gridToWorld(const Eigen::Vector3d& grid) const {
    return grid * config_.resolution + origin_world_;
}

void ObservedMap::integrateDepth(const float* depth,
                                 int height,
                                 int width,
                                 const Eigen::Vector3d& cam_pos_world,
                                 const Eigen::Vector4d& quat_xyzw,
                                 double timestamp_s) {
    if (!initialized_ || depth == nullptr || height <= 0 || width <= 0) {
        return;
    }
    const double hfov = config_.horizontal_fov_deg * kPi / 180.0;
    const double vfov =
        2.0 * std::atan(std::tan(hfov * 0.5) *
                        static_cast<double>(height) /
                        std::max(1.0, static_cast<double>(width)));
    const double fx = (static_cast<double>(width) * 0.5) / std::tan(hfov * 0.5);
    const double fy = (static_cast<double>(height) * 0.5) / std::tan(vfov * 0.5);
    const double cx = (static_cast<double>(width) - 1.0) * 0.5;
    const double cy = (static_cast<double>(height) - 1.0) * 0.5;

    const Eigen::Quaterniond q(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1],
                               quat_xyzw[2]);
    const Eigen::Matrix3d body_to_world = q.toRotationMatrix();

    const int step = std::max(1, config_.depth_integration_step);
    const double max_depth = config_.max_depth_m;
    const double occ_margin = config_.occupied_endpoint_margin_m;

    purgeExpired(timestamp_s);

    // ── Pass 1: occupied endpoints ──────────────────────────────
    std::vector<Eigen::Vector3d> endpoints;
    endpoints.reserve(static_cast<size_t>(height / step + 1) *
                      (width / step + 1));
    for (int row = 0; row < height; row += step) {
        for (int col = 0; col < width; col += step) {
            const float z = depth[static_cast<size_t>(row) * width + col];
            if (!std::isfinite(z) || z <= 0.0f || z >= 1000.0f) continue;
            const double cam_x = (static_cast<double>(col) - cx) * z / fx;
            const double cam_y = (static_cast<double>(row) - cy) * z / fy;
            const double cam_z = static_cast<double>(z);
            // flightlib body frame: [x_right, y_forward, z_up]
            const Eigen::Vector3d flightlib_body(cam_x, cam_z, -cam_y);
            const Eigen::Vector3d world_point =
                body_to_world * flightlib_body + cam_pos_world;
            if (cam_z < max_depth - occ_margin) {
                const Eigen::Vector3i g = worldToGridInt(world_point);
                const std::int64_t index = indexOf(g.x(), g.y(), g.z());
                if (index >= 0) {
                    if (occ_[static_cast<size_t>(index)] != OCCUPIED) {
                        occ_[static_cast<size_t>(index)] = OCCUPIED;
                        last_obs_time_[static_cast<size_t>(index)] =
                            timestamp_s;
                        ++revision_;
                    }
                }
            }
            endpoints.push_back(world_point);
        }
    }

    // ── Pass 2: free-space ray casting ──────────────────────────
    const double spacing =
        std::max(config_.free_space_spacing_m, config_.resolution);
    for (const Eigen::Vector3d& endpoint : endpoints) {
        const Eigen::Vector3d delta = endpoint - cam_pos_world;
        const double length = delta.norm();
        const double effective = std::max(0.0, length - occ_margin);
        if (effective <= 1.0e-6) continue;
        const Eigen::Vector3d direction = delta / length;
        const int samples =
            std::max(1, static_cast<int>(std::ceil(effective / spacing)));
        for (int i = 0; i < samples; ++i) {
            const Eigen::Vector3d point =
                cam_pos_world + direction *
                    (static_cast<double>(i) * spacing + 0.5 * spacing);
            const Eigen::Vector3i g = worldToGridInt(point);
            const std::int64_t index = indexOf(g.x(), g.y(), g.z());
            if (index < 0) continue;
            if (occ_[static_cast<size_t>(index)] == UNKNOWN) {
                occ_[static_cast<size_t>(index)] = FREE;
                last_obs_time_[static_cast<size_t>(index)] = timestamp_s;
                ++revision_;
            }
        }
    }

    markVehicleFreeBubble(cam_pos_world, timestamp_s);
}

void ObservedMap::markVehicleFreeBubble(const Eigen::Vector3d& center_world,
                                        double timestamp_s) {
    if (!initialized_) return;
    const double radius = config_.vehicle_radius_m + std::sqrt(3.0) *
                                                       config_.resolution;
    const int cells = std::max(1, static_cast<int>(std::ceil(
                                      radius / config_.resolution)));
    const Eigen::Vector3i center = worldToGridInt(center_world);
    for (int dx = -cells; dx <= cells; ++dx) {
        for (int dy = -cells; dy <= cells; ++dy) {
            for (int dz = -cells; dz <= cells; ++dz) {
                const std::int64_t index =
                    indexOf(center.x() + dx, center.y() + dy, center.z() + dz);
                if (index < 0) continue;
                if (occ_[static_cast<size_t>(index)] == OCCUPIED) continue;
                if (occ_[static_cast<size_t>(index)] != FREE) {
                    occ_[static_cast<size_t>(index)] = FREE;
                    last_obs_time_[static_cast<size_t>(index)] = timestamp_s;
                    ++revision_;
                }
            }
        }
    }
}

void ObservedMap::purgeExpired(double now_s) {
    if (!initialized_ || config_.history_seconds <= 0.0) return;
    const double cutoff = now_s - config_.history_seconds;
    for (size_t i = 0; i < occ_.size(); ++i) {
        if (occ_[i] != UNKNOWN && last_obs_time_[i] < cutoff) {
            occ_[i] = UNKNOWN;
            last_obs_time_[i] = -1.0;
            ++revision_;
        }
    }
}

void ObservedMap::buildEsdfImpl() {
    const size_t total = static_cast<size_t>(gx_) * gy_ * gz_;
    // Distance to the nearest OCCUPIED cell only.  Known-free cells keep
    // their true clearance (they may legitimately approach the frontier of
    // known space).  Unknown cells are explicitly set to a non-free value
    // (0.0) and are further excluded by the separate known mask, so unknown
    // space is never disguised as free.
    std::vector<std::uint8_t> seed(total, 0);
    known_mask_.assign(total, 0);
    for (size_t i = 0; i < total; ++i) {
        if (occ_[i] == OCCUPIED) {
            seed[i] = 1;
            known_mask_[i] = 1;
        } else if (occ_[i] == FREE) {
            known_mask_[i] = 1;
        }
    }
    const std::vector<double> dist2 = squaredEdt3d(seed, gx_, gy_, gz_);
    const double res = config_.resolution;
    const double max_val = config_.esdf_max_distance_m;
    esdf_.resize(total);
    for (size_t i = 0; i < total; ++i) {
        double value;
        if (occ_[i] == FREE) {
            value = std::sqrt(dist2[i]) * res - config_.vehicle_radius_m;
        } else if (occ_[i] == OCCUPIED) {
            value = -config_.vehicle_radius_m;
        } else {
            value = 0.0;  // unknown: never treated as free
        }
        esdf_[i] = static_cast<float>(
            std::max(-max_val, std::min(max_val, value)));
    }
    esdf_built_ = true;
    ++esdf_revision_;
}

bool ObservedMap::rebuildEsdf() {
    if (!initialized_) return false;
    ++frames_since_esdf_;
    if (frames_since_esdf_ < std::max(1, config_.rebuild_every_n_frames)) {
        return false;
    }
    frames_since_esdf_ = 0;
    buildEsdfImpl();
    return true;
}

void ObservedMap::forceRebuildEsdf() {
    if (!initialized_) return;
    frames_since_esdf_ = 0;
    buildEsdfImpl();
}

std::uint8_t ObservedMap::occupancyAt(double x, double y, double z) const {
    if (!initialized_) return UNKNOWN;
    const Eigen::Vector3i g = worldToGridInt(Eigen::Vector3d(x, y, z));
    const std::int64_t index = indexOf(g.x(), g.y(), g.z());
    if (index < 0) return UNKNOWN;
    return occ_[static_cast<size_t>(index)];
}

bool ObservedMap::isKnown(double x, double y, double z) const {
    if (!initialized_ || !esdf_built_) return false;
    // All 8 corners of the trilinear interpolation footprint must be known.
    const Eigen::Vector3d g = worldToGrid(Eigen::Vector3d(x, y, z));
    const int ix = static_cast<int>(std::floor(g.x()));
    const int iy = static_cast<int>(std::floor(g.y()));
    const int iz = static_cast<int>(std::floor(g.z()));
    for (int dx = 0; dx <= 1; ++dx) {
        for (int dy = 0; dy <= 1; ++dy) {
            for (int dz = 0; dz <= 1; ++dz) {
                const std::int64_t index =
                    indexOf(ix + dx, iy + dy, iz + dz);
                if (index < 0) return false;
                if (known_mask_[static_cast<size_t>(index)] == 0) return false;
            }
        }
    }
    return true;
}

double ObservedMap::esdfValue(double x, double y, double z) const {
    if (!initialized_ || !esdf_built_) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::Vector3d g = worldToGrid(Eigen::Vector3d(x, y, z));
    const double ix = g.x() - 0.5;
    const double iy = g.y() - 0.5;
    const double iz = g.z() - 0.5;
    if (ix < -0.5 || ix > gx_ - 0.5 || iy < -0.5 || iy > gy_ - 0.5 ||
        iz < -0.5 || iz > gz_ - 0.5) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int x0 = std::max(0, std::min(gx_ - 1, static_cast<int>(std::floor(ix))));
    const int x1 = std::min(gx_ - 1, x0 + 1);
    const int y0 = std::max(0, std::min(gy_ - 1, static_cast<int>(std::floor(iy))));
    const int y1 = std::min(gy_ - 1, y0 + 1);
    const int z0 = std::max(0, std::min(gz_ - 1, static_cast<int>(std::floor(iz))));
    const int z1 = std::min(gz_ - 1, z0 + 1);
    const double fx = std::max(0.0, std::min(1.0, ix - std::floor(ix)));
    const double fy = std::max(0.0, std::min(1.0, iy - std::floor(iy)));
    const double fz = std::max(0.0, std::min(1.0, iz - std::floor(iz)));
    auto val = [&](int a, int b, int c) {
        return esdf_[(static_cast<size_t>(a) * gy_ + b) * gz_ + c];
    };
    const double c000 = val(x0, y0, z0);
    const double c100 = val(x1, y0, z0);
    const double c010 = val(x0, y1, z0);
    const double c110 = val(x1, y1, z0);
    const double c001 = val(x0, y0, z1);
    const double c101 = val(x1, y0, z1);
    const double c011 = val(x0, y1, z1);
    const double c111 = val(x1, y1, z1);
    const double c00 = c000 * (1.0 - fx) + c100 * fx;
    const double c10 = c010 * (1.0 - fx) + c110 * fx;
    const double c01 = c001 * (1.0 - fx) + c101 * fx;
    const double c11 = c011 * (1.0 - fx) + c111 * fx;
    const double c0 = c00 * (1.0 - fy) + c10 * fy;
    const double c1 = c01 * (1.0 - fy) + c11 * fy;
    return c0 * (1.0 - fz) + c1 * fz;
}

bool ObservedMap::isKnownFree(double x, double y, double z,
                              double min_clearance) const {
    if (!isKnown(x, y, z)) return false;
    const double value = esdfValue(x, y, z);
    return std::isfinite(value) && value > min_clearance;
}

int ObservedMap::knownCount() const {
    if (!initialized_) return 0;
    return static_cast<int>(std::count(known_mask_.begin(), known_mask_.end(),
                                       static_cast<std::uint8_t>(1)));
}

int ObservedMap::occupiedCount() const {
    if (!initialized_) return 0;
    return static_cast<int>(std::count(occ_.begin(), occ_.end(),
                                       static_cast<std::uint8_t>(OCCUPIED)));
}

int ObservedMap::freeCount() const {
    if (!initialized_) return 0;
    return static_cast<int>(std::count(occ_.begin(), occ_.end(),
                                       static_cast<std::uint8_t>(FREE)));
}

int ObservedMap::unknownCount() const {
    if (!initialized_) return 0;
    return static_cast<int>(std::count(occ_.begin(), occ_.end(),
                                       static_cast<std::uint8_t>(UNKNOWN)));
}

}  // namespace il_dataset
