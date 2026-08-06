#include "il_dataset/local_planner/local_recoverability.hpp"

#include <cmath>
#include <limits>

#include "il_dataset/local_planner/local_path_search.hpp"
#include "il_dataset/local_planner/observed_map.hpp"

namespace il_dataset {

namespace {
constexpr double kEpsilon = 1.0e-9;
}

LocalRecoverability::LocalRecoverability(const RecoverabilityConfig& config)
    : config_(config) {}

RecoverabilityResult LocalRecoverability::test(
    const ObservedMap& map,
    const VehicleState& state,
    const Eigen::Vector3d& direct_guide_world) const {
    RecoverabilityResult result;
    result.rejoin_point = direct_guide_world;
    if (!map.esdfBuilt()) {
        result.status = RecoverabilityStatus::NO_SAFE_MOTION;
        result.reason = "observed_map_not_built";
        return result;
    }

    // The direct guide lies along the goal ray; use it as the goal
    // direction reference.
    const Eigen::Vector3d travel = direct_guide_world - state.position;
    const double travel_len = travel.norm();
    Eigen::Vector2d goal_dir(1.0, 0.0);
    if (travel_len > kEpsilon) {
        goal_dir = travel.head<2>() / travel_len;
    }

    LocalSearchConfig search_config;
    search_config.search_clearance_m = config_.search_clearance_m;
    search_config.committed_side = Side::NONE;  // direct intent has no side
    search_config.forbid_unknown = true;

    LocalPathSearch search;
    const LocalSearchResult search_result =
        search.search(map, state, direct_guide_world, search_config);
    result.minimum_clearance = search_result.min_clearance;
    result.path_length = search_result.path_length;

    if (search_result.success) {
        // Full path to the rejoin point found in known free space.
        result.known_free = true;
        result.feasible = true;
        result.estimated_duration =
            search_result.path_length /
            std::max(0.1, config_.nominal_speed_mps);
        const Eigen::Vector3d terminal = search_result.terminal;
        const Eigen::Vector3d motion = terminal - state.position;
        const double motion_len = motion.norm();
        if (travel_len > kEpsilon) {
            result.goal_progress = motion.dot(travel) / travel_len;
        }
        if (motion_len > kEpsilon) {
            result.terminal_guide_alignment =
                motion.head<2>().dot(goal_dir) / motion_len;
        }
        // Rejoin point must be at least at the direct-guide distance.
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
        if (result.estimated_duration > config_.max_execution_time_s) {
            result.status = RecoverabilityStatus::PARTIAL_PROGRESS_ONLY;
            result.reason = "exceeds_local_horizon";
            result.feasible = false;
            return result;
        }
        const double straight = travel_len;
        if (straight > kEpsilon &&
            search_result.path_length / straight > config_.max_loop_ratio) {
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

    if (!search_result.found_any) {
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
