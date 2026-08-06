#pragma once

#include <Eigen/Core>
#include <vector>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class ObservedMap;

/// Configuration for macro candidate generation (section VIII).
struct MacroCandidateConfig {
    /// Direct-guide lookahead used when no blocker is present.
    double lookahead_distance_m = 4.5;
    /// Length of the known side-corridor used to place SIDE candidates.
    double side_corridor_length_m = 4.0;
    /// Swept radius of the side-corridor / candidate clearance check.
    double side_corridor_radius_m = 0.55;
    /// Radius around the blocker used to find its visible edges.
    double edge_search_radius_m = 5.0;
    /// Required clearance for a candidate to count as known-reachable.
    double min_candidate_clearance_m = 0.25;
    /// Spacing between consecutive SIDE candidates along the corridor.
    double candidate_spacing_m = 0.5;
    /// Forward step of OBSERVE candidates (kept very short / known-safe).
    double observe_step_m = 0.6;
    /// Max number of goal-directed frontier candidates.
    int max_frontier_candidates = 8;
    /// Pull-back of frontier candidates into known space.
    double frontier_standoff_m = 0.45;
    /// Half-cone (deg) around the goal direction for frontier extraction.
    double goal_frontier_cone_deg = 70.0;
    /// Straight-line corridor check spacing.
    double corridor_check_spacing_m = 0.10;
};

/// Shared blocker analysis used by both the recoverability query and the
/// candidate search.  Walks the ray from `state` toward `goal_world` and
/// identifies the first non-known-free region, its connected component,
/// visible edges and known side corridors.
GoalBlocker analyzeGoalBlocker(const ObservedMap& map,
                               const VehicleState& state,
                               const Eigen::Vector3d& goal_world,
                               const MacroCandidateConfig& config);

/// 5 Hz macro candidate generator.  All candidates are built from the
/// observed map and current observable structure only; the privileged
/// oracle later scores them (never the other way around).
class MacroCandidateSearch {
public:
    explicit MacroCandidateSearch(const MacroCandidateConfig& config);

    /// Generate the candidate set: direct, left/right side, left/right
    /// observe, goal-directed frontiers and the previous strategic
    /// continuation.  `prev_candidate_world` is the previous macro guide
    /// (world) used for the continuation candidate.
    std::vector<MacroCandidate> generateCandidates(
        const ObservedMap& map,
        const VehicleState& state,
        const Eigen::Vector3d& goal_world,
        const GoalBlocker& blocker,
        const Eigen::Vector3d* prev_candidate_world) const;

private:
    void scoreObserved(MacroCandidate* candidate,
                       const ObservedMap& map,
                       const VehicleState& state,
                       const Eigen::Vector3d& goal_world) const;
    MacroCandidate makeSideCandidate(const ObservedMap& map,
                                     const VehicleState& state,
                                     const Eigen::Vector3d& goal_world,
                                     const GoalBlocker& blocker,
                                     Side side,
                                     double offset_m) const;
    std::vector<MacroCandidate> makeFrontierCandidates(
        const ObservedMap& map,
        const VehicleState& state,
        const Eigen::Vector3d& goal_world) const;

    MacroCandidateConfig config_;
};

}  // namespace il_dataset
