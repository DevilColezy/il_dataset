#include "il_dataset/local_planner/privileged_intervention_oracle.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "il_dataset/local_planner/privileged_oracle.hpp"

namespace il_dataset {

namespace {
constexpr double kEpsilon = 1.0e-9;
}

PrivilegedInterventionOracle::PrivilegedInterventionOracle(
    const PrivilegedInterventionConfig& config)
    : config_(config) {}

void PrivilegedInterventionOracle::reset() {
    history_.clear();
    revisit_count_ = 0;
    outside_revisit_radius_ = true;
}

bool PrivilegedInterventionOracle::detectLoopRisk(
    const Eigen::Vector3d& position, double speed) const {
    // Stationary hover / rotation is handled by the macro's strategic
    // progress metrics, not loop detection.
    if (speed < config_.loop_min_speed_mps) return false;
    if (history_.size() < 2) return false;
    const double radius = config_.loop_revisit_radius_m;
    // Revisit: the drone returned to a previously visited region.
    bool revisited = false;
    for (const Eigen::Vector3d& old : history_) {
        if ((old - position).head<2>().norm() <= radius) {
            revisited = true;
            break;
        }
    }
    if (!revisited) return false;
    // A genuine loop requires having left the radius since the last visit.
    return revisit_count_ >= config_.loop_min_revisits;
}

PrivilegedInterventionResult PrivilegedInterventionOracle::evaluate(
    const PrivilegedOracle& oracle,
    const VehicleState& state,
    const Eigen::Vector3d& direct_guide_world,
    const Eigen::Vector3d& goal_world) {
    PrivilegedInterventionResult result;

    const Eigen::Vector3d position = state.position;
    const double speed = state.velocity.norm();

    // ── Loop-history update (used by the next evaluate call). ────────
    if (!history_.empty()) {
        const double radius = config_.loop_revisit_radius_m;
        const bool inside =
            (history_.back() - position).head<2>().norm() <= radius;
        if (outside_revisit_radius_ && inside && speed >= config_.loop_min_speed_mps) {
            ++revisit_count_;
        }
        outside_revisit_radius_ = !inside;
    }
    history_.push_back(position);
    while (static_cast<int>(history_.size()) > config_.loop_history_size) {
        history_.pop_front();
    }
    const bool loop_risk = detectLoopRisk(position, speed);

    // ── Direct ray geometry ──────────────────────────────────────────
    const Eigen::Vector2d travel = direct_guide_world.head<2>() - position.head<2>();
    const double travel_len = travel.norm();
    Eigen::Vector2d dir(1.0, 0.0);
    if (travel_len > kEpsilon) dir = travel / travel_len;

    const double direct_ctg = oracle.costToGo(
        direct_guide_world.x(), direct_guide_world.y(), direct_guide_world.z());
    result.direct_cost_to_go =
        std::isfinite(direct_ctg) ? direct_ctg : std::numeric_limits<double>::infinity();

    // Minimum clearance along the direct ray (global map).
    double direct_clearance = std::numeric_limits<double>::infinity();
    const double ray_len = std::max(0.1, travel_len);
    const int samples = std::max(2, static_cast<int>(std::ceil(
                                         ray_len / config_.lookahead_sampling_m)));
    for (int i = 0; i <= samples; ++i) {
        const Eigen::Vector3d point(
            position.x() + dir.x() * (ray_len * i / samples),
            position.y() + dir.y() * (ray_len * i / samples),
            position.z());
        const double c = oracle.clearance(point.x(), point.y(), point.z());
        if (std::isfinite(c)) direct_clearance = std::min(direct_clearance, c);
    }
    result.direct_min_clearance =
        std::isfinite(direct_clearance) ? direct_clearance : 0.0;

    // Straight-line distance from the direct guide to the goal.
    const double guide_goal_dist =
        (direct_guide_world - goal_world).head<2>().norm();
    result.direct_detour_ratio =
        std::isfinite(direct_ctg)
            ? direct_ctg / std::max(0.5, guide_goal_dist)
            : std::numeric_limits<double>::infinity();

    // ── Left / right lateral alternatives at the direct guide ─────────
    const Eigen::Vector3d perp(-dir.y(), dir.x(), 0.0);
    const Eigen::Vector3d left_point =
        direct_guide_world + perp * config_.lateral_offset_m;
    const Eigen::Vector3d right_point =
        direct_guide_world - perp * config_.lateral_offset_m;
    const double left_ctg = oracle.costToGo(
        left_point.x(), left_point.y(), left_point.z());
    const double right_ctg = oracle.costToGo(
        right_point.x(), right_point.y(), right_point.z());
    result.left_cost_to_go =
        std::isfinite(left_ctg) ? left_ctg : std::numeric_limits<double>::infinity();
    result.right_cost_to_go =
        std::isfinite(right_ctg) ? right_ctg : std::numeric_limits<double>::infinity();
    result.left_globally_feasible =
        oracle.connectedToGoal(left_point.x(), left_point.y(), left_point.z()) &&
        oracle.clearance(left_point.x(), left_point.y(), left_point.z()) >=
            config_.min_global_clearance_m;
    result.right_globally_feasible =
        oracle.connectedToGoal(right_point.x(), right_point.y(), right_point.z()) &&
        oracle.clearance(right_point.x(), right_point.y(), right_point.z()) >=
            config_.min_global_clearance_m;

    // Decision margin between the feasible lateral alternatives.
    double best_side_cost = std::numeric_limits<double>::infinity();
    if (result.left_globally_feasible) {
        best_side_cost = std::min(best_side_cost, result.left_cost_to_go);
    }
    if (result.right_globally_feasible) {
        best_side_cost = std::min(best_side_cost, result.right_cost_to_go);
    }
    if (result.left_globally_feasible && result.right_globally_feasible) {
        result.decision_margin =
            std::abs(result.left_cost_to_go - result.right_cost_to_go);
    } else {
        result.decision_margin = 0.0;
    }

    // ── Viability ────────────────────────────────────────────────────
    const bool state_connected = oracle.connectedToGoal(
        position.x(), position.y(), position.z());
    const bool guide_connected =
        std::isfinite(direct_ctg) &&
        oracle.connectedToGoal(direct_guide_world.x(), direct_guide_world.y(),
                               direct_guide_world.z());
    const bool wrong_homotopy =
        std::isfinite(best_side_cost) && std::isfinite(direct_ctg) &&
        direct_ctg > best_side_cost + config_.cost_margin_m;

    result.direct_viable =
        state_connected && guide_connected &&
        result.direct_min_clearance >= config_.min_global_clearance_m &&
        result.direct_detour_ratio <= config_.max_direct_detour_ratio &&
        !wrong_homotopy && !loop_risk;
    result.intervention_required = !result.direct_viable;

    // ── Reason (enumerated) ──────────────────────────────────────────
    if (!state_connected &&
        !result.left_globally_feasible && !result.right_globally_feasible) {
        result.reason = InterventionReason::NO_GLOBAL_ROUTE;
    } else if (!guide_connected) {
        result.reason = InterventionReason::DIRECT_GLOBAL_DISCONNECTED;
    } else if (loop_risk) {
        result.reason = InterventionReason::DIRECT_LOOP_RISK;
    } else if (result.direct_min_clearance < config_.min_global_clearance_m) {
        result.reason = InterventionReason::DIRECT_LONG_WALL_BLOCKED;
    } else if (wrong_homotopy) {
        result.reason = InterventionReason::DIRECT_WRONG_HOMOTOPY;
    } else if (result.direct_detour_ratio > config_.max_direct_detour_ratio) {
        result.reason = InterventionReason::DIRECT_EXCESSIVE_DETOUR;
    } else if (result.left_globally_feasible && result.right_globally_feasible) {
        result.reason = InterventionReason::BOTH_SIDES_FEASIBLE;
    } else if (result.left_globally_feasible) {
        result.reason = InterventionReason::LEFT_ONLY_FEASIBLE;
    } else if (result.right_globally_feasible) {
        result.reason = InterventionReason::RIGHT_ONLY_FEASIBLE;
    } else {
        result.reason = InterventionReason::DIRECT_GLOBALLY_VALID;
    }
    return result;
}

}  // namespace il_dataset
