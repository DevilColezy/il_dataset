#include "il_dataset/local_planner/local_recoverability.hpp"

#include <cmath>
#include <limits>

#include "il_dataset/local_planner/local_path_search.hpp"
#include "il_dataset/local_planner/observed_map.hpp"

namespace il_dataset {

namespace {
constexpr double kEpsilon = 1.0e-9;

/// Estimate the terminal TANGENT of a reconstructed path (section VI).
/// Uses path.back() - path[k] for the farthest point back that is at least
/// `min_baseline` away (a tiny last segment cannot dominate).
bool terminalTangent(const std::vector<Eigen::Vector3d>& path,
                     double min_baseline,
                     Eigen::Vector2d* tangent_out) {
    if (path.size() < 2) return false;
    const Eigen::Vector2d end = path.back().head<2>();
    for (size_t i = path.size() - 1; i-- > 0;) {
        const Eigen::Vector2d delta = end - path[i].head<2>();
        const double len = delta.norm();
        if (len >= min_baseline) {
            *tangent_out = delta / len;
            return true;
        }
    }
    const Eigen::Vector2d delta = end - path.front().head<2>();
    const double len = delta.norm();
    if (len <= kEpsilon) return false;
    *tangent_out = delta / len;
    return true;
}

}  // namespace

LocalRecoverability::LocalRecoverability(const RecoverabilityConfig& config)
    : config_(config) {}

RecoverabilityResult LocalRecoverability::test(
    const ObservedMap& map,
    const VehicleState& state,
    const Eigen::Vector3d& direct_guide_world) const {
    RecoverabilityResult result;
    if (!map.esdfBuilt()) {
        result.status = RecoverabilityStatus::NO_SAFE_MOTION;
        result.reason = "observed_map_not_built";
        return result;
    }

    // The direct guide lies along the goal ray; the rejoin point is placed
    // at the configured rejoin distance along that direction (clamped to
    // the guide distance).  Same geometry as the privileged audit.
    const Eigen::Vector3d travel = direct_guide_world - state.position;
    const double travel_len = travel.norm();
    Eigen::Vector2d goal_dir(1.0, 0.0);
    if (travel_len > kEpsilon) {
        goal_dir = travel.head<2>() / travel_len;
    }
    const Eigen::Vector3d rejoin_dir =
        travel_len > kEpsilon ? travel / travel_len : Eigen::Vector3d::UnitX();
    const double rejoin_dist =
        std::min(std::max(0.1, travel_len), config_.rejoin_distance_m);
    const Eigen::Vector3d rejoin_point =
        state.position + rejoin_dir * rejoin_dist;
    result.rejoin_point = rejoin_point;
    result.rejoin_distance = rejoin_dist;

    LocalSearchConfig search_config;
    search_config.search_clearance_m = config_.search_clearance_m;
    search_config.committed_side = Side::NONE;  // direct intent has no side
    search_config.forbid_unknown = true;

    LocalPathSearch search;
    const LocalPathResult search_result =
        search.search(map, state, rejoin_point, search_config);
    result.minimum_clearance = search_result.minimum_clearance;
    result.path_length = search_result.path_cost;
    const double straight_rejoin =
        std::max(0.1, (rejoin_point - state.position).norm());
    result.detour_ratio =
        straight_rejoin > kEpsilon
            ? result.path_length / straight_rejoin
            : 1.0;

    // DIRECT_REJOIN_SUCCESS requires FULL arrival at the rejoin point
    // (section VI).  A partial path is never "recoverable".
    if (search_result.status == LocalPathResult::Status::FULL_GOAL_REACHED) {
        // Full path to the rejoin point found in known free space.
        result.known_free = true;
        result.feasible = true;
        result.estimated_duration =
            search_result.path_cost /
            std::max(0.1, config_.nominal_speed_mps);
        if (travel_len > kEpsilon) {
            const Eigen::Vector3d motion = rejoin_point - state.position;
            result.goal_progress = motion.dot(travel) / travel_len;
        }
        // Terminal alignment uses the path TERMINAL TANGENT (section VI),
        // not the start-end chord.
        Eigen::Vector2d tangent;
        if (terminalTangent(search_result.path,
                            config_.terminal_tangent_min_baseline, &tangent)) {
            result.terminal_guide_alignment = tangent.dot(goal_dir);
        }
        if (result.goal_progress < config_.min_goal_progress_m) {
            result.status = RecoverabilityStatus::PARTIAL_PROGRESS_ONLY;
            result.reason = "insufficient_goal_progress";
            result.feasible = false;
            return result;
        }
        if (result.terminal_guide_alignment <
            config_.min_terminal_alignment) {
            result.status = RecoverabilityStatus::PARTIAL_PROGRESS_ONLY;
            result.reason = "terminal_not_aligned_with_guide";
            result.feasible = false;
            return result;
        }
        if (result.estimated_duration > config_.max_duration_s) {
            result.status = RecoverabilityStatus::PARTIAL_PROGRESS_ONLY;
            result.reason = "exceeds_local_horizon";
            result.feasible = false;
            return result;
        }
        if (result.path_length > config_.max_path_length_m) {
            result.status = RecoverabilityStatus::PARTIAL_PROGRESS_ONLY;
            result.reason = "exceeds_local_path_length";
            result.feasible = false;
            return result;
        }
        if (straight_rejoin > kEpsilon &&
            result.detour_ratio > config_.max_detour_ratio) {
            result.status = RecoverabilityStatus::PARTIAL_PROGRESS_ONLY;
            result.reason = "path_loops_or_backtracks";
            result.feasible = false;
            return result;
        }
        result.status = RecoverabilityStatus::DIRECT_REJOIN_SUCCESS;
        result.reason = "direct_rejoin_success";
        return result;
    }

    // ── Not fully recoverable: classify the blocker ─────────────────
    result.feasible = false;
    MacroCandidateConfig blocker_config;
    blocker_config.edge_search_radius_m = config_.edge_search_radius_m;
    blocker_config.side_corridor_length_m = config_.side_corridor_length_m;
    blocker_config.side_corridor_radius_m = config_.side_corridor_radius_m;
    blocker_config.min_candidate_clearance_m = config_.search_clearance_m;
    const GoalBlocker blocker =
        analyzeGoalBlocker(map, state, direct_guide_world, blocker_config);
    result.blocking_component_id = blocker.component_id;
    result.left_edge_visible = blocker.left_edge_visible;
    result.right_edge_visible = blocker.right_edge_visible;
    result.left_corridor_known = blocker.left_corridor_known;
    result.right_corridor_known = blocker.right_corridor_known;

    if (!search_result.found_partial) {
        // No safe motion at all.  Classify whether the immediate blockage
        // is known or unknown.
        const std::uint8_t state_at = map.occupancyAt(
            state.position.x(), state.position.y(), state.position.z());
        if (state_at == OCCUPIED) {
            result.status = RecoverabilityStatus::BLOCKED_BY_KNOWN;
            result.reason = "no_safe_motion_blocked_by_known";
        } else if (state_at == UNKNOWN) {
            result.status = RecoverabilityStatus::BLOCKED_BY_UNKNOWN;
            result.reason = "no_safe_motion_blocked_by_unknown";
        } else {
            result.status = RecoverabilityStatus::NO_SAFE_MOTION;
            result.reason = "no_safe_motion";
        }
        return result;
    }

    // Partial progress exists but the rejoin point is unreachable.
    result.status = blocker.found && blocker.blocked_by_known
                        ? RecoverabilityStatus::BLOCKED_BY_KNOWN
                        : RecoverabilityStatus::BLOCKED_BY_UNKNOWN;
    result.reason = result.status == RecoverabilityStatus::BLOCKED_BY_KNOWN
                        ? "partial_progress_blocked_by_known"
                        : "partial_progress_blocked_by_unknown";
    if (!blocker.found) {
        result.status = RecoverabilityStatus::PARTIAL_PROGRESS_ONLY;
        result.reason = "partial_progress_blocker_unclassified";
    }
    return result;
}

}  // namespace il_dataset
