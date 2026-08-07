#include "il_dataset/local_planner/task_generation_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "il_dataset/local_planner/privileged_intervention_oracle.hpp"

namespace il_dataset {

namespace {

constexpr double kEpsilon = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;

/// Per-candidate 2D search on the scene grid at a reference z slice:
/// flood-free start/goal tests plus a goal-reversed Dijkstra with parent
/// pointers (for path reconstruction / clearance and lateral probes).
struct SliceSearch {
    bool built = false;
    int gx = 0;
    int gy = 0;
    int iz = 0;
    double z_world = 0.0;
    std::vector<double> dist;  // from goal (inf = unreachable)
    std::vector<int> parent;   // -1 = none; self for goal
};

SliceSearch runSliceSearch(const PrivilegedOracle& oracle,
                           const Eigen::Vector3d& start,
                           const Eigen::Vector3d& goal,
                           double clearance) {
    SliceSearch s;
    const Eigen::Vector3d& origin = oracle.origin();
    const double res = oracle.resolution();
    const int gx = oracle.gx();
    const int gy = oracle.gy();
    const int gz = oracle.gz();
    s.gx = gx;
    s.gy = gy;
    const double z_ref = 0.5 * (start.z() + goal.z());
    s.iz = std::max(0, std::min(
                           gz - 1, static_cast<int>(std::floor(
                                       (z_ref - origin.z()) / res))));
    s.z_world = origin.z() + (static_cast<double>(s.iz) + 0.5) * res;
    const size_t slice = static_cast<size_t>(gx) * gy;
    s.dist.assign(slice, std::numeric_limits<double>::infinity());
    s.parent.assign(slice, -1);

    auto cell_free = [&](int ix, int iy) {
        if (ix < 0 || ix >= gx || iy < 0 || iy >= gy) return false;
        const double wx = origin.x() + (static_cast<double>(ix) + 0.5) * res;
        const double wy = origin.y() + (static_cast<double>(iy) + 0.5) * res;
        return oracle.isFree(wx, wy, s.z_world, clearance);
    };
    auto world_to_cell = [&](double x, double y) {
        const int ix = static_cast<int>(std::floor((x - origin.x()) / res));
        const int iy = static_cast<int>(std::floor((y - origin.y()) / res));
        return std::make_pair(ix, iy);
    };

    const std::pair<int, int> sg = world_to_cell(start.x(), start.y());
    const std::pair<int, int> gg = world_to_cell(goal.x(), goal.y());
    if (!cell_free(sg.first, sg.second) || !cell_free(gg.first, gg.second)) {
        return s;
    }
    s.built = true;
    const int start_index = sg.second * gx + sg.first;
    const int goal_index = gg.second * gx + gg.first;

    struct Entry {
        double cost;
        int index;
        bool operator<(const Entry& other) const { return cost > other.cost; }
    };
    std::priority_queue<Entry> open;
    s.dist[static_cast<size_t>(goal_index)] = 0.0;
    s.parent[static_cast<size_t>(goal_index)] = goal_index;
    open.push({0.0, goal_index});

    const int di[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dj[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    auto corner_ok = [&](int ix, int iy, int n) {
        if (di[n] != 0 && dj[n] != 0) {
            return cell_free(ix + di[n], iy) && cell_free(ix, iy + dj[n]);
        }
        return true;
    };
    while (!open.empty()) {
        const Entry current = open.top();
        open.pop();
        if (current.cost > s.dist[static_cast<size_t>(current.index)] + 1.0e-6) {
            continue;
        }
        const int ix = current.index % gx;
        const int iy = current.index / gx;
        for (int n = 0; n < 8; ++n) {
            const int nxi = ix + di[n];
            const int nyi = iy + dj[n];
            if (nxi < 0 || nxi >= gx || nyi < 0 || nyi >= gy) continue;
            const int nindex = nyi * gx + nxi;
            if (!cell_free(nxi, nyi)) continue;
            if (!corner_ok(ix, iy, n)) continue;
            const double step =
                (di[n] == 0 || dj[n] == 0) ? res : res * std::sqrt(2.0);
            const double tentative = current.cost + step;
            if (tentative >= s.dist[static_cast<size_t>(nindex)]) continue;
            s.dist[static_cast<size_t>(nindex)] = tentative;
            s.parent[static_cast<size_t>(nindex)] = current.index;
            open.push({tentative, nindex});
        }
    }
    return s;
}

}  // namespace

TaskGenerationOracle::TaskGenerationOracle(const TaskGenerationConfig& config)
    : config_(config) {}

TaskCandidateResult TaskGenerationOracle::evaluate(
    const PrivilegedOracle& oracle,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& goal) const {
    TaskCandidateResult r;
    if (!oracle.built() || !start.allFinite() || !goal.allFinite()) {
        r.reason = "invalid_input";
        return r;
    }
    r.start_free = oracle.isFree(start.x(), start.y(), start.z(),
                                 config_.start_clearance_m);
    r.goal_free = oracle.isFree(goal.x(), goal.y(), goal.z(),
                                config_.goal_clearance_m);
    r.straight_distance = (goal - start).norm();
    if (!r.start_free || !r.goal_free) {
        r.reason = "endpoint_not_free";
        return r;
    }
    if (r.straight_distance < config_.min_task_distance_m) {
        r.reason = "too_short";
        return r;
    }
    if (r.straight_distance > config_.max_task_distance_m) {
        r.reason = "too_long";
        return r;
    }

    const double res = oracle.resolution();
    const Eigen::Vector2d travel = (goal - start).head<2>();
    const double travel_len = travel.norm();
    if (travel_len <= kEpsilon) {
        r.reason = "zero_horizontal";
        return r;
    }
    const Eigen::Vector2d dir = travel / travel_len;
    const Eigen::Vector2d perp(-dir.y(), dir.x());  // + = left (world)
    const double z_walk = 0.5 * (start.z() + goal.z());

    // Direct corridor ray walk.
    bool in_block = false;
    for (double d = res; d <= travel_len + 1.0e-6; d += res) {
        const double px = start.x() + dir.x() * d;
        const double py = start.y() + dir.y() * d;
        const bool free =
            oracle.isFree(px, py, z_walk, config_.direct_corridor_clearance_m);
        if (!free) {
            if (!in_block) {
                ++r.direct_blocker_count;
                if (r.nearest_blocker_distance_m < 0.0) {
                    r.nearest_blocker_distance_m = d;
                }
                in_block = true;
            }
            r.direct_blocked = true;
        } else {
            in_block = false;
        }
    }

    // Global connectivity + cost-to-go from this candidate's goal (local
    // copy; the oracle's persistent task state is left untouched).
    const SliceSearch s = runSliceSearch(oracle, start, goal,
                                         config_.lateral_path_clearance_m);
    if (!s.built) {
        r.reason = "endpoint_not_in_grid";
        return r;
    }
    const int gx = s.gx;
    auto index_of = [gx](int ix, int iy) { return iy * gx + ix; };
    const int sgx = static_cast<int>(std::floor(
        (start.x() - oracle.origin().x()) / res));
    const int sgy = static_cast<int>(std::floor(
        (start.y() - oracle.origin().y()) / res));
    const double start_dist = s.dist[static_cast<size_t>(index_of(sgx, sgy))];
    r.goal_reachable = std::isfinite(start_dist);
    if (!r.goal_reachable) {
        r.reason = "goal_not_reachable";
        return r;
    }
    r.global_path_length = start_dist;
    r.global_detour_ratio = start_dist / std::max(0.1, r.straight_distance);

    // Reconstruct the path for minimum clearance.
    double min_clear = std::numeric_limits<double>::infinity();
    int cur = index_of(sgx, sgy);
    int guard = 0;
    while (cur >= 0 && cur != s.parent[static_cast<size_t>(cur)] &&
           guard < 2000000) {
        const int ix = cur % gx;
        const int iy = cur / gx;
        const double wx = oracle.origin().x() + (static_cast<double>(ix) + 0.5) * res;
        const double wy = oracle.origin().y() + (static_cast<double>(iy) + 0.5) * res;
        const double c = oracle.clearance(wx, wy, s.z_world);
        if (std::isfinite(c)) min_clear = std::min(min_clear, c);
        cur = s.parent[static_cast<size_t>(cur)];
        ++guard;
    }
    r.global_min_clearance = std::isfinite(min_clear) ? min_clear : 0.0;

    // Left / right global feasibility via lateral probes past the blocker.
    const double d0 = r.nearest_blocker_distance_m >= 0.0
                          ? r.nearest_blocker_distance_m
                          : 0.5 * r.straight_distance;
    for (int pass = 0; pass < 2; ++pass) {
        const int side_sign = pass == 0 ? 1 : -1;  // +1 left, -1 right
        bool feasible = false;
        double best_len = std::numeric_limits<double>::infinity();
        for (int k = 1; k <= config_.lateral_probe_count; ++k) {
            const double along = d0 + static_cast<double>(k) *
                                          config_.lateral_probe_spacing_m;
            const Eigen::Vector2d probe =
                start.head<2>() + dir * along +
                perp * (static_cast<double>(side_sign) *
                        config_.lateral_probe_offset_m);
            if (!oracle.isFree(probe.x(), probe.y(), s.z_world,
                               config_.lateral_path_clearance_m)) {
                continue;
            }
            const int pix = static_cast<int>(std::floor(
                (probe.x() - oracle.origin().x()) / res));
            const int piy = static_cast<int>(std::floor(
                (probe.y() - oracle.origin().y()) / res));
            if (pix < 0 || pix >= gx || piy < 0 || piy >= s.gy) continue;
            const double pd =
                s.dist[static_cast<size_t>(index_of(pix, piy))];
            if (!std::isfinite(pd)) continue;
            feasible = true;
            best_len = std::min(best_len, pd);
        }
        if (pass == 0) {
            r.left_global_feasible = feasible;
            r.left_path_length = std::isfinite(best_len) ? best_len : -1.0;
        } else {
            r.right_global_feasible = feasible;
            r.right_path_length = std::isfinite(best_len) ? best_len : -1.0;
        }
    }

    // Privileged local-scale audit on the FULL map: short-range rejoin
    // search from the start toward the goal ray, allowing local bypass.
    PrivilegedInterventionConfig lic;
    lic.search_clearance_m = config_.search_clearance_m;
    lic.search_max_time_ms = config_.search_max_time_ms;
    lic.rejoin_distance_m = config_.rejoin_distance_m;
    lic.max_duration_s = config_.max_duration_s;
    lic.max_path_length_m = config_.max_path_length_m;
    lic.nominal_speed_mps = config_.nominal_speed_mps;
    lic.max_detour_ratio = config_.max_detour_ratio;
    lic.min_goal_progress_m = config_.min_goal_progress_m;
    lic.min_terminal_alignment = config_.min_terminal_alignment;
    lic.terminal_tangent_min_baseline = config_.terminal_tangent_min_baseline;
    lic.search_lateral_margin_m = config_.search_lateral_margin_m;
    lic.search_longitudinal_margin_m = config_.search_longitudinal_margin_m;
    PrivilegedInterventionOracle local_audit(lic);
    VehicleState vs;
    vs.position = start;
    vs.velocity = Eigen::Vector3d::Zero();
    vs.acceleration = Eigen::Vector3d::Zero();
    vs.yaw = std::atan2(dir.y(), dir.x()) - 0.5 * kPi;
    vs.yaw_rate = 0.0;
    const double look = std::min(travel_len, config_.rejoin_distance_m);
    const Eigen::Vector3d direct_guide =
        start + Eigen::Vector3d(dir.x(), dir.y(), 0.0) * look;
    const PrivilegedInterventionResult pir =
        local_audit.evaluate(oracle, vs, direct_guide, goal, 0.0);
    r.privileged_local_recoverable = pir.privileged_local_recoverable;
    r.reason = "ok";
    return r;
}

std::vector<TaskCandidateResult> TaskGenerationOracle::evaluateCandidates(
    const PrivilegedOracle& oracle,
    const std::vector<Eigen::Vector3d>& starts,
    const std::vector<Eigen::Vector3d>& goals) const {
    std::vector<TaskCandidateResult> results;
    const size_t n = std::min(starts.size(), goals.size());
    results.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        results.push_back(evaluate(oracle, starts[i], goals[i]));
    }
    return results;
}

}  // namespace il_dataset
