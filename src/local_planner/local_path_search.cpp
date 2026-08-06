#include "il_dataset/local_planner/local_path_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>

#include "il_dataset/local_planner/observed_map.hpp"

namespace il_dataset {

namespace {

constexpr double kEpsilon = 1.0e-9;

struct OpenEntry {
    double score = 0.0;
    int index = -1;
    double g = 0.0;
    double remaining = 0.0;
    bool operator<(const OpenEntry& other) const {
        if (std::abs(score - other.score) > kEpsilon) return score > other.score;
        return remaining > other.remaining;
    }
};

}  // namespace

LocalSearchResult LocalPathSearch::search(const ObservedMap& map,
                                          const VehicleState& state,
                                          const Eigen::Vector3d& goal_world,
                                          const LocalSearchConfig& config) const {
    using Clock = std::chrono::steady_clock;
    const auto start_time = Clock::now();
    const auto deadline = start_time + std::chrono::duration_cast<Clock::duration>(
                                           std::chrono::duration<double, std::milli>(
                                               std::max(1.0, config.max_time_ms)));
    LocalSearchResult result;

    if (!map.esdfBuilt() || !state.position.allFinite() ||
        !goal_world.allFinite()) {
        result.failure_reason = "map_not_ready_or_invalid_state";
        return result;
    }
    const double res = map.resolution();
    const double z = state.position.z();

    // ── Search region on the map's own grid ──────────────────────────
    const Eigen::Vector3d start_g = map.worldToGrid(state.position);
    const Eigen::Vector3d goal_g = map.worldToGrid(goal_world);
    const int margin = std::max(1, static_cast<int>(std::ceil(
                                       config.region_margin_m / res)));
    const int min_ix = std::max(0, static_cast<int>(std::floor(
                                       std::min(start_g.x(), goal_g.x()))) -
                                       margin);
    const int max_ix = std::min(map.gx() - 1, static_cast<int>(std::floor(
                                                  std::max(start_g.x(), goal_g.x()))) +
                                                  margin);
    const int min_iy = std::max(0, static_cast<int>(std::floor(
                                       std::min(start_g.y(), goal_g.y()))) -
                                       margin);
    const int max_iy = std::min(map.gy() - 1, static_cast<int>(std::floor(
                                                  std::max(start_g.y(), goal_g.y()))) +
                                                  margin);
    if (max_ix < min_ix || max_iy < min_iy) {
        result.failure_reason = "search_region_empty";
        return result;
    }
    const int nx = max_ix - min_ix + 1;
    const int ny = max_iy - min_iy + 1;
    const size_t total = static_cast<size_t>(nx) * ny;
    if (total > static_cast<size_t>(config.max_expansions) * 2ULL) {
        result.failure_reason = "search_region_too_large";
        return result;
    }
    auto encode = [ny](int ix, int iy) { return ix * ny + iy; };
    auto worldPoint = [&](int ix, int iy) {
        return Eigen::Vector3d(
            map.origin().x() + (static_cast<double>(ix) + 0.5) * res,
            map.origin().y() + (static_cast<double>(iy) + 0.5) * res, z);
    };

    const int start_ix = std::max(min_ix, std::min(max_ix, static_cast<int>(
                                                             std::floor(start_g.x()))));
    const int start_iy = std::max(min_iy, std::min(max_iy, static_cast<int>(
                                                             std::floor(start_g.y()))));
    const int goal_ix = std::max(min_ix, std::min(max_ix, static_cast<int>(
                                                             std::floor(goal_g.x()))));
    const int goal_iy = std::max(min_iy, std::min(max_iy, static_cast<int>(
                                                             std::floor(goal_g.y()))));
    const int start_index = encode(start_ix - min_ix, start_iy - min_iy);
    const int goal_index = encode(goal_ix - min_ix, goal_iy - min_iy);

    // The continuous start must be in known (observed) space.  It does not
    // need the full search clearance — the drone is already physically
    // there — but it must be observed so the search can escape toward
    // higher-clearance cells.
    if (config.forbid_unknown &&
        !map.isKnown(state.position.x(), state.position.y(),
                     state.position.z())) {
        result.failure_reason = "start_not_known";
        return result;
    }

    const Eigen::Vector3d travel = goal_world - state.position;
    const double travel_len = travel.norm();
    Eigen::Vector2d ray_dir(1.0, 0.0);
    if (travel_len > kEpsilon) {
        ray_dir = travel.head<2>() / travel_len;
    }

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> g_cost(total, inf);
    std::vector<int> parent(total, -1);
    std::vector<std::uint8_t> closed(total, 0);
    std::priority_queue<OpenEntry> open;

    g_cost[static_cast<size_t>(start_index)] = 0.0;
    open.push({(state.position - goal_world).norm(), start_index, 0.0,
               (state.position - goal_world).norm()});

    bool goal_reached = false;
    int expansions = 0;
    // Best partial node: reachable node minimizing remaining distance to the
    // goal (tie-break by larger g = farther along the path).
    int best_index = start_index;
    double best_remaining = (state.position - goal_world).norm();
    double best_g = 0.0;

    const int di[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dj[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    while (!open.empty() && Clock::now() < deadline &&
           expansions < config.max_expansions) {
        const OpenEntry current = open.top();
        open.pop();
        const size_t ci = static_cast<size_t>(current.index);
        if (closed[ci] != 0) continue;
        closed[ci] = 1;
        ++expansions;
        if (current.index == goal_index) {
            goal_reached = true;
            break;
        }
        if (current.remaining < best_remaining - kEpsilon ||
            (std::abs(current.remaining - best_remaining) <= kEpsilon &&
             current.g > best_g)) {
            best_remaining = current.remaining;
            best_g = current.g;
            best_index = current.index;
        }
        const int cx = current.index / ny + min_ix;
        const int cy = current.index % ny + min_iy;
        const Eigen::Vector3d current_world = worldPoint(cx, cy);
        for (int n = 0; n < 8; ++n) {
            const int nxi = cx + di[n];
            const int nyi = cy + dj[n];
            if (nxi < min_ix || nxi > max_ix || nyi < min_iy || nyi > max_iy) {
                continue;
            }
            const int next_index = encode(nxi - min_ix, nyi - min_iy);
            const size_t ni = static_cast<size_t>(next_index);
            if (closed[ni] != 0) continue;
            const Eigen::Vector3d next_world = worldPoint(nxi, nyi);
            const double step =
                (next_world - current_world).norm();
            if (config.forbid_unknown &&
                !map.isKnownFree(next_world.x(), next_world.y(),
                                 next_world.z(), config.search_clearance_m)) {
                continue;
            }
            double cost = step;
            // Committed-side lateral bias (soft, never hard-blocking).
            if (config.committed_side != Side::NONE &&
                config.side_bias_gain > kEpsilon && travel_len > kEpsilon) {
                const Eigen::Vector2d offset =
                    next_world.head<2>() - state.position.head<2>();
                // cross(ray, offset).z > 0 => offset is left of the ray.
                const double lateral =
                    ray_dir.x() * offset.y() - ray_dir.y() * offset.x();
                const bool wrong_side =
                    (config.committed_side == Side::LEFT && lateral < 0.0) ||
                    (config.committed_side == Side::RIGHT && lateral > 0.0);
                if (wrong_side) {
                    cost += config.side_bias_gain * std::abs(lateral);
                }
            }
            const double tentative = g_cost[ci] + cost;
            if (tentative >= g_cost[ni]) continue;
            g_cost[ni] = tentative;
            parent[ni] = current.index;
            const double remaining = (next_world - goal_world).norm();
            open.push({tentative + remaining, next_index, tentative,
                       remaining});
        }
    }

    // Reconstruct the path.
    auto reconstruct = [&](int terminal_index) {
        std::vector<Eigen::Vector3d> reverse_path;
        for (int index = terminal_index; index >= 0;
             index = parent[static_cast<size_t>(index)]) {
            const int ix = index / ny + min_ix;
            const int iy = index % ny + min_iy;
            reverse_path.push_back(worldPoint(ix, iy));
            if (index == start_index) break;
        }
        std::reverse(reverse_path.begin(), reverse_path.end());
        return reverse_path;
    };

    double min_clearance = inf;
    if (goal_reached) {
        std::vector<Eigen::Vector3d> path = reconstruct(goal_index);
        if (path.empty() || (path.back() - worldPoint(goal_ix, goal_iy)).norm() >
                                res * 2.0) {
            result.failure_reason = "path_reconstruction_failed";
            return result;
        }
        path.front() = state.position;
        path.back() = goal_world;
        result.success = true;
        result.found_any = true;
        result.path = std::move(path);
        result.terminal = goal_world;
    } else if (best_index != start_index) {
        std::vector<Eigen::Vector3d> path = reconstruct(best_index);
        if (!path.empty()) {
            path.front() = state.position;
            result.found_any = true;
            result.path = std::move(path);
            result.terminal = result.path.back();
            result.failure_reason = "partial_path_only";
        } else {
            result.failure_reason = "no_partial_path";
            return result;
        }
    } else {
        result.failure_reason = "no_safe_motion";
        return result;
    }

    for (const Eigen::Vector3d& point : result.path) {
        const double c = map.esdfValue(point.x(), point.y(), point.z());
        if (std::isfinite(c)) min_clearance = std::min(min_clearance, c);
    }
    result.min_clearance =
        std::isfinite(min_clearance) ? min_clearance : 0.0;
    double length = 0.0;
    for (size_t i = 1; i < result.path.size(); ++i) {
        length += (result.path[i] - result.path[i - 1]).norm();
    }
    result.path_length = length;
    result.expanded_nodes = expansions;
    result.compute_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start_time)
            .count();
    return result;
}

}  // namespace il_dataset
