#include "il_dataset/local_planner/privileged_intervention_oracle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>

#include "il_dataset/local_planner/privileged_oracle.hpp"

namespace il_dataset {

namespace {

constexpr double kEpsilon = 1.0e-9;

/// Continuous free check on the global map: all samples must satisfy the
/// unified free-space definition (esdf > required_clearance).
bool segmentFree(const PrivilegedOracle& oracle,
                 const Eigen::Vector3d& from,
                 const Eigen::Vector3d& to,
                 double required_clearance,
                 double spacing) {
    const double distance = (to - from).norm();
    const int samples = std::max(
        1, static_cast<int>(std::ceil(distance / std::max(0.02, spacing))));
    for (int i = 0; i <= samples; ++i) {
        const double alpha = static_cast<double>(i) / samples;
        const Eigen::Vector3d point = (1.0 - alpha) * from + alpha * to;
        if (!oracle.isFree(point.x(), point.y(), point.z(),
                           required_clearance)) {
            return false;
        }
    }
    return true;
}

struct OpenEntry {
    double score = 0.0;
    int index = -1;
    double g = 0.0;
    bool operator<(const OpenEntry& other) const {
        if (std::abs(score - other.score) > kEpsilon) return score > other.score;
        return g > other.g;
    }
};

}  // namespace

PrivilegedInterventionOracle::PrivilegedInterventionOracle(
    const PrivilegedInterventionConfig& config)
    : config_(config) {}

void PrivilegedInterventionOracle::reset() { history_.clear(); }

bool PrivilegedInterventionOracle::detectLoopRisk(
    const Eigen::Vector2d& position, double now_s) const {
    if (history_.size() < 4) return false;
    const double ignore = config_.loop_ignore_recent_s;
    const double revisit_r = config_.loop_revisit_radius_m;
    const double leave_r = config_.loop_leave_radius_m;
    int revisits = 0;
    // For every old sample (older than the ignore window) within the
    // revisit radius of the current position, check that the drone LEFT
    // that region (beyond leave_radius) somewhere between the old sample
    // and now.  Only such excursions count as genuine revisits; recent
    // contiguous motion is ignored.
    for (size_t i = 0; i < history_.size(); ++i) {
        const Eigen::Vector2d& old_pos = history_[i].first;
        const double old_t = history_[i].second;
        if (now_s - old_t < ignore) continue;
        if ((old_pos - position).norm() > revisit_r) continue;
        bool left_region = false;
        for (size_t j = i + 1; j < history_.size(); ++j) {
            if ((history_[j].first - old_pos).norm() > leave_r) {
                left_region = true;
                break;
            }
        }
        if (left_region) ++revisits;
    }
    return revisits >= config_.loop_min_revisits;
}

PrivilegedInterventionResult PrivilegedInterventionOracle::evaluate(
    const PrivilegedOracle& oracle,
    const VehicleState& state,
    const Eigen::Vector3d& direct_guide_world,
    const Eigen::Vector3d& goal_world,
    double current_time_s) {
    PrivilegedInterventionResult result;
    if (!oracle.built() || !state.position.allFinite() ||
        !direct_guide_world.allFinite() || !goal_world.allFinite()) {
        result.privileged_local_recoverable = false;
        result.privileged_future_intervention_required = true;
        result.reason = InterventionReason::NO_GLOBAL_ROUTE;
        return result;
    }

    const Eigen::Vector2d position2 = state.position.head<2>();
    const double speed = state.velocity.norm();

    // ── Loop-history update (section IX) ─────────────────────────────
    if (speed >= config_.loop_min_speed_mps && std::isfinite(current_time_s)) {
        history_.emplace_back(position2, current_time_s);
        while (static_cast<int>(history_.size()) > config_.loop_history_size) {
            history_.pop_front();
        }
    }
    const bool loop_risk =
        detectLoopRisk(position2, current_time_s);
    result.loop_risk = loop_risk;

    // ── Global cost-to-go diagnostics ────────────────────────────────
    const double cur_ctg = oracle.costToGo(
        state.position.x(), state.position.y(), state.position.z());
    result.current_cost_to_go =
        std::isfinite(cur_ctg) ? cur_ctg : std::numeric_limits<double>::infinity();
    const double direct_ctg = oracle.costToGo(
        direct_guide_world.x(), direct_guide_world.y(), direct_guide_world.z());
    result.direct_cost_to_go =
        std::isfinite(direct_ctg) ? direct_ctg : std::numeric_limits<double>::infinity();

    // ── Privileged LOCAL-SCALE recoverability (section II) ──────────
    // Run a short-range A* on the FULL map from the current state to the
    // direct guide, ALLOWING local bypass (never a straight-line ray
    // check).  This mirrors the observed local recoverability geometry as
    // closely as the full map allows.
    const double z = state.position.z();
    const double res = oracle.resolution();
    const Eigen::Vector3d origin = oracle.origin();
    const int gx = oracle.gx();
    const int gy = oracle.gy();
    auto world_point = [&](int ix, int iy) {
        return Eigen::Vector3d(origin.x() + (static_cast<double>(ix) + 0.5) * res,
                               origin.y() + (static_cast<double>(iy) + 0.5) * res,
                               z);
    };
    auto cell_free = [&](int ix, int iy) {
        if (ix < 0 || ix >= gx || iy < 0 || iy >= gy) return false;
        const Eigen::Vector3d p = world_point(ix, iy);
        return oracle.isFree(p.x(), p.y(), p.z(), config_.search_clearance_m);
    };

    const Eigen::Vector3d start_g = (state.position - origin) / res;
    const Eigen::Vector3d goal_g = (direct_guide_world - origin) / res;
    const int min_ix = std::max(0, static_cast<int>(std::floor(
                                       std::min(start_g.x(), goal_g.x()))) - 2);
    const int max_ix = std::min(gx - 1, static_cast<int>(std::floor(
                                                std::max(start_g.x(), goal_g.x()))) + 2);
    const int min_iy = std::max(0, static_cast<int>(std::floor(
                                       std::min(start_g.y(), goal_g.y()))) - 2);
    const int max_iy = std::min(gy - 1, static_cast<int>(std::floor(
                                                std::max(start_g.y(), goal_g.y()))) + 2);
    if (max_ix < min_ix || max_iy < min_iy) {
        result.privileged_local_recoverable = false;
        result.privileged_future_intervention_required = true;
        result.reason = InterventionReason::NO_GLOBAL_ROUTE;
        return result;
    }
    const int nx = max_ix - min_ix + 1;
    const int ny = max_iy - min_iy + 1;
    const size_t total = static_cast<size_t>(nx) * ny;
    auto encode = [ny](int ix, int iy) { return ix * ny + iy; };

    const int start_ix =
        std::max(min_ix, std::min(max_ix, static_cast<int>(std::floor(start_g.x()))));
    const int start_iy =
        std::max(min_iy, std::min(max_iy, static_cast<int>(std::floor(start_g.y()))));
    const int goal_ix =
        std::max(min_ix, std::min(max_ix, static_cast<int>(std::floor(goal_g.x()))));
    const int goal_iy =
        std::max(min_iy, std::min(max_iy, static_cast<int>(std::floor(goal_g.y()))));
    const int start_index = encode(start_ix - min_ix, start_iy - min_iy);
    const int goal_index = encode(goal_ix - min_ix, goal_iy - min_iy);

    // The start must be free under the unified global definition.
    if (!oracle.isFree(state.position.x(), state.position.y(), z,
                       config_.search_clearance_m)) {
        result.privileged_local_recoverable = false;
        result.privileged_future_intervention_required = true;
        result.reason = InterventionReason::NO_GLOBAL_ROUTE;
        return result;
    }
    const bool goal_cell_free = cell_free(goal_ix, goal_iy);

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> g_cost(total, inf);
    std::vector<int> parent(total, -1);
    std::vector<std::uint8_t> closed(total, 0);
    std::priority_queue<OpenEntry> open;
    g_cost[static_cast<size_t>(start_index)] = 0.0;
    open.push({(state.position - direct_guide_world).norm(), start_index, 0.0});

    const int di[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dj[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(
                std::max(1.0, config_.search_max_time_ms)));

    bool goal_reached = false;
    int expansions = 0;
    const int max_expansions = 200000;
    while (!open.empty() && expansions < max_expansions &&
           std::chrono::steady_clock::now() < deadline) {
        const OpenEntry current = open.top();
        open.pop();
        const size_t ci = static_cast<size_t>(current.index);
        if (closed[ci] != 0) continue;
        closed[ci] = 1;
        ++expansions;
        if (goal_cell_free && current.index == goal_index) {
            goal_reached = true;
            break;
        }
        const int cx = current.index / ny + min_ix;
        const int cy = current.index % ny + min_iy;
        const Eigen::Vector3d current_world = world_point(cx, cy);
        for (int n = 0; n < 8; ++n) {
            const int nxi = cx + di[n];
            const int nyi = cy + dj[n];
            if (nxi < min_ix || nxi > max_ix || nyi < min_iy || nyi > max_iy) {
                continue;
            }
            // Diagonal corner-cutting rule (section XI): both orthogonal
            // neighbours must be free.
            if (di[n] != 0 && dj[n] != 0) {
                if (!cell_free(cx + di[n], cy) || !cell_free(cx, cy + dj[n])) {
                    continue;
                }
            }
            const int next_index = encode(nxi - min_ix, nyi - min_iy);
            const size_t ni = static_cast<size_t>(next_index);
            if (closed[ni] != 0) continue;
            if (!cell_free(nxi, nyi)) continue;
            const Eigen::Vector3d next_world = world_point(nxi, nyi);
            const double step = (next_world - current_world).norm();
            const double tentative = g_cost[ci] + step;
            if (tentative >= g_cost[ni]) continue;
            g_cost[ni] = tentative;
            parent[ni] = current.index;
            open.push({tentative + (next_world - direct_guide_world).norm(),
                       next_index, tentative});
        }
    }

    auto reconstruct = [&](int terminal_index) {
        std::vector<Eigen::Vector3d> reverse_path;
        for (int index = terminal_index; index >= 0;
             index = parent[static_cast<size_t>(index)]) {
            const int ix = index / ny + min_ix;
            const int iy = index % ny + min_iy;
            reverse_path.push_back(world_point(ix, iy));
            if (index == start_index) break;
        }
        std::reverse(reverse_path.begin(), reverse_path.end());
        return reverse_path;
    };

    std::vector<Eigen::Vector3d> path;
    Eigen::Vector3d terminal{Eigen::Vector3d::Zero()};
    if (goal_reached) {
        path = reconstruct(goal_index);
        if (!path.empty()) {
            path.front() = state.position;
        }
        // Rejoin requires the exact goal region to be free and the final
        // segment continuous (no unknown here, only clearance).
        const Eigen::Vector3d goal_cell = world_point(goal_ix, goal_iy);
        if (!segmentFree(oracle, goal_cell, direct_guide_world,
                         config_.search_clearance_m, res)) {
            goal_reached = false;
            path.clear();
        } else {
            path.back() = direct_guide_world;
            terminal = direct_guide_world;
        }
    }

    double path_length = 0.0;
    double min_clearance = inf;
    for (size_t i = 1; i < path.size(); ++i) {
        path_length += (path[i] - path[i - 1]).norm();
    }
    for (const Eigen::Vector3d& p : path) {
        const double c = oracle.clearance(p.x(), p.y(), p.z());
        if (std::isfinite(c)) min_clearance = std::min(min_clearance, c);
    }
    result.privileged_local_path_length = path_length;
    result.privileged_min_clearance =
        std::isfinite(min_clearance) ? min_clearance : 0.0;
    result.privileged_local_duration =
        path_length / std::max(0.1, config_.nominal_speed_mps);
    const double straight = (direct_guide_world - state.position).norm();
    result.privileged_detour_ratio =
        path_length / std::max(0.5, straight);

    // Goal progress: projection of terminal motion onto the goal ray.
    const Eigen::Vector3d travel = direct_guide_world - state.position;
    const double travel_len = travel.norm();
    double goal_progress = 0.0;
    double terminal_alignment = 0.0;
    double rejoin_lateral = 0.0;
    if (goal_reached && travel_len > kEpsilon) {
        const Eigen::Vector3d motion = terminal - state.position;
        goal_progress = motion.dot(travel) / travel_len;
        const double motion_len = motion.norm();
        if (motion_len > kEpsilon) {
            terminal_alignment =
                motion.head<2>().dot(travel.head<2>() / travel_len) /
                motion_len;
        }
        // Lateral offset of the terminal from the direct-guide ray: the
        // terminal must re-enter the guide's forward region.
        const Eigen::Vector2d ray_dir = travel.head<2>() / travel_len;
        const Eigen::Vector2d rel = motion.head<2>();
        const Eigen::Vector2d proj = ray_dir * rel.dot(ray_dir);
        rejoin_lateral = (rel - proj).norm();
    }
    result.privileged_goal_progress = goal_progress;

    // Rejoin condition (mirrors observed local recoverability):
    // full arrival + within local path budget + sufficient progress +
    // terminal re-enters the direct guide's forward region (aligned +
    // laterally within rejoin_radius_m) + minimum clearance.
    const bool within_budget =
        path_length <= config_.max_path_length_m + 1.0e-3 &&
        result.privileged_local_duration <= config_.horizon_time_s + 1.0e-3;
    const bool progress_ok =
        goal_progress >= config_.min_goal_progress_m;
    const bool alignment_ok =
        terminal_alignment >= config_.min_terminal_alignment &&
        rejoin_lateral <= config_.rejoin_radius_m;
    const bool clearance_ok =
        result.privileged_min_clearance >= config_.search_clearance_m;

    result.privileged_rejoin_reached =
        goal_reached && within_budget && progress_ok && alignment_ok &&
        clearance_ok;
    result.privileged_local_recoverable =
        result.privileged_rejoin_reached && !loop_risk;
    result.privileged_future_intervention_required =
        !result.privileged_local_recoverable;

    // ── Reason (enumerated) ──────────────────────────────────────────
    const bool state_connected = oracle.connectedToGoal(
        state.position.x(), state.position.y(), state.position.z());
    if (!state_connected) {
        result.reason = InterventionReason::NO_GLOBAL_ROUTE;
    } else if (loop_risk) {
        result.reason = InterventionReason::DIRECT_LOOP_RISK;
    } else if (!goal_reached) {
        result.reason = InterventionReason::DIRECT_LONG_WALL_BLOCKED;
    } else if (!within_budget) {
        result.reason = InterventionReason::DIRECT_EXCESSIVE_DETOUR;
    } else if (!clearance_ok) {
        result.reason = InterventionReason::DIRECT_LONG_WALL_BLOCKED;
    } else if (!progress_ok || !alignment_ok) {
        result.reason = InterventionReason::DIRECT_EXCESSIVE_DETOUR;
    } else {
        result.reason = InterventionReason::DIRECT_GLOBALLY_VALID;
    }
    return result;
}

}  // namespace il_dataset
