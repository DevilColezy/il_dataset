#include "il_dataset/local_planner/privileged_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace il_dataset {

namespace {

constexpr double kEpsilon = 1.0e-9;

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

std::vector<double> squaredEdt3d(const std::vector<std::uint8_t>& seed,
                                 int gx, int gy, int gz) {
    auto at = [&](int ix, int iy, int iz) {
        return (ix * gy + iy) * gz + iz;
    };
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> f(static_cast<size_t>(gx) * gy * gz, inf);
    for (size_t i = 0; i < seed.size(); ++i) {
        if (seed[i] != 0) f[i] = 0.0;
    }
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

bool PrivilegedOracle::build(const std::vector<Eigen::Vector3d>& points,
                             const Eigen::Vector3d& start,
                             const Eigen::Vector3d& goal,
                             const PrivilegedOracleConfig& config) {
    config_ = config;
    built_ = false;
    task_reachable_ = false;
    start_goal_distance_ = (goal - start).norm();
    if (points.empty() || !start.allFinite() || !goal.allFinite()) {
        return false;
    }
    const double res = config_.resolution;

    // Grid bounds from the point cloud, clamped to the start-goal bbox with
    // a margin so cost-to-go and connectivity stay meaningful.
    Eigen::Vector3d min_p(std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity());
    Eigen::Vector3d max_p(-std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity());
    for (const Eigen::Vector3d& p : points) {
        min_p = min_p.cwiseMin(p);
        max_p = max_p.cwiseMax(p);
    }
    Eigen::Vector3d min_w = start.cwiseMin(goal).cwiseMin(min_p);
    Eigen::Vector3d max_w = start.cwiseMax(goal).cwiseMax(max_p);
    min_w.array() -= config_.map_margin_m;
    max_w.array() += config_.map_margin_m;
    min_w.z() = std::max(min_w.z(), config_.min_z_m);
    max_w.z() = std::min(max_w.z(), config_.max_z_m);
    if (min_w.z() >= max_w.z() - 0.01) {
        min_w.z() = config_.min_z_m;
        max_w.z() = std::max(config_.min_z_m + 0.5, config_.max_z_m);
    }

    origin_world_ = min_w;
    gx_ = std::max(2, static_cast<int>(std::ceil((max_w.x() - min_w.x()) / res)));
    gy_ = std::max(2, static_cast<int>(std::ceil((max_w.y() - min_w.y()) / res)));
    gz_ = std::max(2, static_cast<int>(std::ceil((max_w.z() - min_w.z()) / res)));
    const size_t total = static_cast<size_t>(gx_) * gy_ * gz_;
    occupancy_.assign(total, 0);

    auto voxelize = [&](double x, double y, double z) -> bool {
        const Eigen::Vector3d g = (Eigen::Vector3d(x, y, z) - origin_world_) / res;
        const int ix = static_cast<int>(std::floor(g.x()));
        const int iy = static_cast<int>(std::floor(g.y()));
        const int iz = static_cast<int>(std::floor(g.z()));
        if (ix < 0 || ix >= gx_ || iy < 0 || iy >= gy_ || iz < 0 || iz >= gz_) {
            return false;
        }
        const size_t index = (static_cast<size_t>(ix) * gy_ + iy) * gz_ + iz;
        occupancy_[index] = 1;
        return true;
    };
    for (const Eigen::Vector3d& p : points) voxelize(p.x(), p.y(), p.z());

    // Global ESDF.  UNIFIED SEMANTICS (section XIV/XV): the ESDF value is
    //   esdf = distance_to_obstacle_surface - vehicle_radius
    // i.e. clearance from the INFLATED vehicle body.  Free space is
    // everywhere defined by  esdf > inflation_m  (never re-subtracting the
    // vehicle radius).
    const std::vector<double> dist2 =
        squaredEdt3d(occupancy_, gx_, gy_, gz_);
    esdf_.resize(total);
    const double max_val = config_.max_esdf_distance_m;
    for (size_t i = 0; i < total; ++i) {
        const double value =
            occupancy_[i] == 0
                ? std::sqrt(dist2[i]) * res - config_.vehicle_radius_m
                : -config_.vehicle_radius_m;
        esdf_[i] = static_cast<float>(
            std::max(-max_val, std::min(max_val, value)));
    }

    buildConnectivityAndCostToGo(start, goal);
    built_ = true;
    return true;
}

Eigen::Vector3i PrivilegedOracle::worldToGridInt(double x, double y,
                                                 double z) const {
    const Eigen::Vector3d g =
        (Eigen::Vector3d(x, y, z) - origin_world_) / config_.resolution;
    return Eigen::Vector3i(static_cast<int>(std::floor(g.x())),
                           static_cast<int>(std::floor(g.y())),
                           static_cast<int>(std::floor(g.z())));
}

int PrivilegedOracle::zIndexAt(double z) const {
    const double gz = (z - origin_world_.z()) / config_.resolution;
    return std::max(0, std::min(gz_ - 1, static_cast<int>(std::floor(gz))));
}

bool PrivilegedOracle::isOccupied(double x, double y, double z) const {
    if (!built_) return true;
    const Eigen::Vector3i g = worldToGridInt(x, y, z);
    if (g.x() < 0 || g.x() >= gx_ || g.y() < 0 || g.y() >= gy_ ||
        g.z() < 0 || g.z() >= gz_) {
        return true;
    }
    return occupancy_[(static_cast<size_t>(g.x()) * gy_ + g.y()) * gz_ +
                      g.z()] != 0;
}

double PrivilegedOracle::clearance(double x, double y, double z) const {
    if (!built_) return std::numeric_limits<double>::quiet_NaN();
    const Eigen::Vector3d g =
        (Eigen::Vector3d(x, y, z) - origin_world_) / config_.resolution;
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

bool PrivilegedOracle::isFree(double x, double y, double z,
                              double required_clearance) const {
    if (!built_) return false;
    const Eigen::Vector3d g =
        (Eigen::Vector3d(x, y, z) - origin_world_) / config_.resolution;
    const int ix = static_cast<int>(std::floor(g.x()));
    const int iy = static_cast<int>(std::floor(g.y()));
    const int iz = static_cast<int>(std::floor(g.z()));
    if (ix < 0 || ix >= gx_ || iy < 0 || iy >= gy_ || iz < 0 || iz >= gz_) {
        return false;
    }
    const size_t index = (static_cast<size_t>(ix) * gy_ + iy) * gz_ + iz;
    if (occupancy_[index] != 0) return false;
    return esdf_[index] > static_cast<float>(required_clearance);
}

void PrivilegedOracle::buildConnectivityAndCostToGo(
    const Eigen::Vector3d& start, const Eigen::Vector3d& goal) {
    const int z = zIndexAt(0.5 * (start.z() + goal.z()));
    cost_to_go_z_ = z;
    const size_t slice = static_cast<size_t>(gx_) * gy_;
    cost_to_go_.assign(slice, std::numeric_limits<float>::infinity());
    connected_.assign(slice, 0);

    // UNIFIED free-space definition (section XIV): free = not occupied AND
    // esdf > inflation_m  (ESDF already subtracts the vehicle radius).
    const double free_threshold = config_.inflation_m;
    auto cell_free = [&](int ix, int iy) {
        if (ix < 0 || ix >= gx_ || iy < 0 || iy >= gy_) return false;
        const size_t index = (static_cast<size_t>(ix) * gy_ + iy) * gz_ + z;
        if (occupancy_[index] != 0) return false;
        return esdf_[index] > static_cast<float>(free_threshold);
    };

    const Eigen::Vector3i start_g = worldToGridInt(start.x(), start.y(), start.z());
    const Eigen::Vector3i goal_g = worldToGridInt(goal.x(), goal.y(), goal.z());
    if (!cell_free(start_g.x(), start_g.y()) ||
        !cell_free(goal_g.x(), goal_g.y())) {
        task_reachable_ = false;
        return;
    }
    const int start_index = start_g.y() * gx_ + start_g.x();
    const int goal_index = goal_g.y() * gx_ + goal_g.x();

    // 8-connected with NO diagonal corner cutting (section XI): a diagonal
    // move requires BOTH orthogonal neighbours to be free as well.
    const int di[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dj[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    auto diagonal_corner_free = [&](int ix, int iy, int n) {
        if (di[n] != 0 && dj[n] != 0) {
            return cell_free(ix + di[n], iy) && cell_free(ix, iy + dj[n]);
        }
        return true;
    };

    // Flood fill from the start to check goal connectivity.
    std::vector<int> stack;
    stack.reserve(slice);
    std::vector<std::uint8_t> visited(slice, 0);
    visited[static_cast<size_t>(start_index)] = 1;
    stack.push_back(start_index);
    bool goal_connected = false;
    while (!stack.empty()) {
        const int index = stack.back();
        stack.pop_back();
        if (index == goal_index) {
            goal_connected = true;
        }
        const int ix = index % gx_;
        const int iy = index / gx_;
        for (int n = 0; n < 8; ++n) {
            const int nxi = ix + di[n];
            const int nyi = iy + dj[n];
            if (nxi < 0 || nxi >= gx_ || nyi < 0 || nyi >= gy_) continue;
            if (!diagonal_corner_free(ix, iy, n)) continue;
            const int nindex = nyi * gx_ + nxi;
            if (visited[static_cast<size_t>(nindex)] != 0) continue;
            if (!cell_free(nxi, nyi)) continue;
            visited[static_cast<size_t>(nindex)] = 1;
            stack.push_back(nindex);
        }
    }
    for (size_t i = 0; i < slice; ++i) {
        connected_[i] = visited[i];
    }
    task_reachable_ = goal_connected;
    if (!goal_connected) return;

    // Dijkstra cost-to-go from the goal over free cells (corner-cutting
    // rule applied, section XI).
    struct Entry {
        double cost;
        int index;
        bool operator<(const Entry& other) const { return cost > other.cost; }
    };
    std::priority_queue<Entry> open;
    std::vector<double> dist(slice, std::numeric_limits<double>::infinity());
    dist[static_cast<size_t>(goal_index)] = 0.0;
    open.push({0.0, goal_index});
    while (!open.empty()) {
        const Entry current = open.top();
        open.pop();
        if (current.cost > dist[static_cast<size_t>(current.index)] + 1.0e-6) {
            continue;
        }
        const int ix = current.index % gx_;
        const int iy = current.index / gx_;
        for (int n = 0; n < 8; ++n) {
            const int nxi = ix + di[n];
            const int nyi = iy + dj[n];
            if (nxi < 0 || nxi >= gx_ || nyi < 0 || nyi >= gy_) continue;
            if (!diagonal_corner_free(ix, iy, n)) continue;
            const int nindex = nyi * gx_ + nxi;
            if (!cell_free(nxi, nyi)) continue;
            const double step =
                (di[n] == 0 || dj[n] == 0)
                    ? config_.resolution
                    : config_.resolution * std::sqrt(2.0);
            const double tentative = current.cost + step;
            if (tentative >= dist[static_cast<size_t>(nindex)]) continue;
            dist[static_cast<size_t>(nindex)] = tentative;
            open.push({tentative, nindex});
        }
    }
    for (size_t i = 0; i < slice; ++i) {
        cost_to_go_[i] = static_cast<float>(
            std::min(config_.cost_to_go_cap_m, dist[i]));
    }
}

double PrivilegedOracle::costToGo(double x, double y, double z) const {
    if (!built_ || cost_to_go_z_ < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const Eigen::Vector3i g = worldToGridInt(x, y, z);
    if (g.x() < 0 || g.x() >= gx_ || g.y() < 0 || g.y() >= gy_) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const float value =
        cost_to_go_[static_cast<size_t>(g.y()) * gx_ + g.x()];
    return std::isfinite(value) ? static_cast<double>(value)
                                : std::numeric_limits<double>::quiet_NaN();
}

bool PrivilegedOracle::connectedToGoal(double x, double y, double z) const {
    if (!built_ || cost_to_go_z_ < 0) return false;
    const Eigen::Vector3i g = worldToGridInt(x, y, z);
    if (g.x() < 0 || g.x() >= gx_ || g.y() < 0 || g.y() >= gy_) return false;
    return connected_[static_cast<size_t>(g.y()) * gx_ + g.x()] != 0;
}

double PrivilegedOracle::directionCostToGo(
    const Eigen::Vector3d& position_world,
    const Eigen::Vector3d& direction_world,
    double distance) const {
    const double norm = direction_world.norm();
    if (norm <= kEpsilon) return std::numeric_limits<double>::quiet_NaN();
    const Eigen::Vector3d point =
        position_world + (direction_world / norm) * distance;
    return costToGo(point.x(), point.y(), point.z());
}

void PrivilegedOracle::scoreCandidates(
    std::vector<MacroCandidate>* candidates,
    const VehicleState& state,
    const Eigen::Vector3d& goal_world,
    Side committed_side,
    const Eigen::Vector3d* previous_guide_world) const {
    if (candidates == nullptr || !built_) return;
    const double cap = config_.cost_to_go_cap_m;
    const double clearance_target = config_.scoring.clearance_target_m;
    for (MacroCandidate& candidate : *candidates) {
        const Eigen::Vector3d& p = candidate.position_world;
        const double c = clearance(p.x(), p.y(), p.z());
        const double ctg = costToGo(p.x(), p.y(), p.z());
        const bool conn = connectedToGoal(p.x(), p.y(), p.z());
        candidate.global_clearance =
            std::isfinite(c) ? c : 0.0;
        candidate.global_cost_to_go =
            std::isfinite(ctg) ? std::min(cap, ctg) : cap;
        candidate.connected_to_goal = conn;

        double score = 0.0;
        const double norm_observed =
            candidate.observed_path_cost /
            std::max(0.5, config_.map_margin_m + 2.0);
        score += config_.scoring.weight_observed_cost *
                 std::min(1.0, norm_observed);
        score += config_.scoring.weight_cost_to_go *
                 (candidate.global_cost_to_go / cap);
        if (!conn) {
            score += config_.scoring.weight_connectivity;
        }
        if (std::isfinite(c) && c < clearance_target) {
            score += config_.scoring.weight_clearance *
                     (clearance_target - c) / clearance_target;
        }
        const Eigen::Vector2d travel =
            goal_world.head<2>() - state.position.head<2>();
        const double travel_len = travel.norm();
        if (travel_len > kEpsilon) {
            const Eigen::Vector2d offset =
                p.head<2>() - state.position.head<2>();
            const double progress = offset.dot(travel / travel_len);
            if (progress < 0.0) {
                score += config_.scoring.weight_goal_progress *
                         (-progress) / std::max(0.5, travel_len);
            }
        }
        score += config_.scoring.weight_information *
                 (1.0 - candidate.unknown_information_gain);
        // Yaw cost: how far is the candidate direction from the current
        // heading?
        const Eigen::Vector2d offset =
            p.head<2>() - state.position.head<2>();
        const double offset_norm = offset.norm();
        if (offset_norm > kEpsilon) {
            const double yaw_world =
                std::atan2(offset.y(), offset.x()) - 0.5 * 3.14159265358979323846;
            double diff = yaw_world - state.yaw;
            while (diff > 3.14159265358979323846) diff -= 2.0 * 3.14159265358979323846;
            while (diff < -3.14159265358979323846) diff += 2.0 * 3.14159265358979323846;
            score += config_.scoring.weight_yaw_cost *
                     std::abs(diff) /
                     std::max(0.1, config_.scoring.yaw_cost_scale_rad);
        }
        // Side-switch penalty: candidate on the opposite side of the
        // committed side.
        if (committed_side != Side::NONE && candidate.side != Side::NONE &&
            candidate.side != committed_side) {
            score += config_.scoring.weight_side_switch *
                     config_.scoring.side_switch_penalty;
        }
        // Repeat penalty: near the previous guide.
        if (previous_guide_world != nullptr) {
            const double d =
                (p - *previous_guide_world).head<2>().norm();
            if (d < config_.scoring.repeat_penalty) {
                score += config_.scoring.weight_repeat *
                         (config_.scoring.repeat_penalty - d) /
                         config_.scoring.repeat_penalty;
            }
        }
        candidate.privileged_score = score;
        // Approximate long-term path length (observed + remaining).
        candidate.global_path_length =
            candidate.observed_path_cost + candidate.global_cost_to_go;
    }
}

Side PrivilegedOracle::privilegedBestSide(
    const std::vector<MacroCandidate>& candidates,
    double* margin_out) const {
    if (margin_out != nullptr) *margin_out = 0.0;
    const MacroCandidate* left = nullptr;
    const MacroCandidate* right = nullptr;
    for (const MacroCandidate& candidate : candidates) {
        if (candidate.type != CandidateType::SIDE) continue;
        if (candidate.side == Side::LEFT && (left == nullptr ||
                                             candidate.privileged_score <
                                                 left->privileged_score)) {
            left = &candidate;
        } else if (candidate.side == Side::RIGHT &&
                   (right == nullptr ||
                    candidate.privileged_score < right->privileged_score)) {
            right = &candidate;
        }
    }
    if (left == nullptr && right == nullptr) return Side::NONE;
    if (left == nullptr) return Side::RIGHT;
    if (right == nullptr) return Side::LEFT;
    const double margin = left->privileged_score - right->privileged_score;
    if (margin_out != nullptr) *margin_out = std::abs(margin);
    constexpr double kTieTolerance = 0.05;
    if (std::abs(margin) < kTieTolerance) return Side::NONE;
    return margin < 0.0 ? Side::LEFT : Side::RIGHT;
}

}  // namespace il_dataset
