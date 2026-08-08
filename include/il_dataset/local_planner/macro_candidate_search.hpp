#pragma once

#include <Eigen/Core>
#include <vector>

#include "il_dataset/local_planner/local_path_search.hpp"
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
    /// Minimum candidate->current distance for an OBSERVE candidate to be
    /// emitted at all.  A zero-distance probe must never be generated: it
    /// would be trivially FULL-reachable and masquerade as an OBSERVE_MOVE.
    double min_observe_move_distance_m = 0.15;
    // ── Active observation viewpoint search (section XV) ───────────────
    /// Lattice of lateral (perpendicular to the goal ray) and forward
    /// offsets around the current position.  Every (forward, lateral)
    /// combination on BOTH sides becomes a raw observation-viewpoint
    /// candidate; this replaces the old single fixed ±observe_step_m probes.
    std::vector<double> observe_lateral_distances_m{0.4, 0.8, 1.2, 1.6};
    std::vector<double> observe_forward_distances_m{0.0, 0.4, 0.8, 1.2};
    /// Cap on raw viewpoint candidates kept after the cheap endpoint
    /// filter (before any A* runs).
    int max_viewpoint_candidates = 24;
    /// Cap on observed LocalPathSearch FULL validations for OBSERVE +
    /// GOAL_FRONTIER movement candidates per tick.  The cheap endpoint
    /// filter and the information-gain / clearance / distance rank run
    /// first; only the top candidates get the full search.
    int max_viewpoint_searches_per_tick = 8;
    /// Minimum FULL searches reserved for GOAL_FRONTIER candidates each
    /// tick (so the lattice can never starve the frontier source).
    int min_frontier_searches_per_tick = 2;
    /// Max current->viewpoint distance for a valid OBSERVE_MOVE.
    double max_observe_move_distance_m = 6.0;
    /// Radius around a viewpoint used to estimate unknown information gain.
    double observe_info_gain_radius_m = 1.2;
    /// Max number of goal-directed frontier candidates.
    int max_frontier_candidates = 8;
    /// Pull-back of frontier candidates into known space.
    double frontier_standoff_m = 0.45;
    /// Half-cone (deg) around the goal direction for frontier extraction.
    double goal_frontier_cone_deg = 70.0;
    /// Straight-line corridor check spacing.
    double corridor_check_spacing_m = 0.10;
    // Observed-map path-search parameters used for REAL reachability of
    // SIDE candidates (section XII).
    double search_clearance_m = 0.25;
    double search_max_time_ms = 20.0;
    double search_region_margin_m = 2.0;
    double side_bias_gain = 2.0;
};

/// Shared blocker analysis used by both the recoverability query and the
/// candidate search.  Walks the ray from `state` toward `goal_world` and
/// identifies the first non-known-free region, its connected component,
/// visible edges and known side corridors.
GoalBlocker analyzeGoalBlocker(const ObservedMap& map,
                               const VehicleState& state,
                               const Eigen::Vector3d& goal_world,
                               const MacroCandidateConfig& config);

/// Per-tick OBSERVE viewpoint generation diagnostics (pure diagnostics,
/// never part of any student input).  Reset at the start of every
/// generateCandidates() call.
struct ObserveDiagnostics {
    /// Raw lattice / frontier viewpoints generated this tick.
    int raw_candidate_count = 0;
    /// Lattice (OBSERVE) candidates emitted after the cheap filter.
    int lattice_candidate_count = 0;
    /// GOAL_FRONTIER candidates emitted after the cheap filter.
    int frontier_candidate_count = 0;
    /// Candidates whose endpoint passed the cheap known-free + clearance
    /// filter (these get ranked and possibly the FULL LocalPathSearch).
    int endpoint_known_free_count = 0;
    /// OBSERVE/GOAL_FRONTIER candidates verified FULL_GOAL_REACHED.
    int full_local_count = 0;
    /// OBSERVE/GOAL_FRONTIER candidates with PARTIAL_TERMINAL_REACHED.
    int partial_count = 0;
    /// OBSERVE/GOAL_FRONTIER candidates with NO_PATH.
    int no_path_count = 0;
    int reject_unknown = 0;             // endpoint not known at all
    int reject_endpoint_clearance = 0;  // endpoint known but not clear enough
    int reject_min_distance = 0;        // endpoint closer than min move dist
    int reject_max_distance = 0;        // endpoint farther than max move dist
};

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

    /// Diagnostics of the LAST generateCandidates() call (see
    /// ObserveDiagnostics).  Holds the previous tick's values until the
    /// next call.
    const ObserveDiagnostics& lastObserveDiagnostics() const {
        return observe_diag_;
    }

private:
    /// REAL observed-map reachability for movement candidates (section
    /// XIII/XV).  SIDE / OBSERVE / GOAL_FRONTIER candidates all run the
    /// observed LocalPathSearch; only FULL_GOAL_REACHED sets
    /// `known_reachable` / `full_goal_reached`.  PARTIAL paths never count
    /// as reachable for OBSERVE / GOAL_FRONTIER (a viewpoint is a real
    /// movement terminal); for SIDE the partial terminal is adopted as
    /// before.  DIRECT / PREVIOUS_CONTINUATION keep the cheap straight
    /// segment check (they are not committed movement targets of this
    /// tick).  The reachability query uses the SAME search clearance as
    /// the 30 Hz planner (search_clearance_m) — never a second inflation.
    void scoreObserved(MacroCandidate* candidate,
                       const ObservedMap& map,
                       const VehicleState& state,
                       const Eigen::Vector3d& goal_world) const;

    /// Unified observed LocalPathSearch reachability for a movement
    /// candidate (SIDE / OBSERVE / GOAL_FRONTIER).  `adopt_partial`
    /// controls whether a PARTIAL terminal replaces the candidate
    /// position (true only for SIDE).
    void evaluateObservedReachability(
        MacroCandidate* candidate,
        const ObservedMap& map,
        const VehicleState& state,
        bool adopt_partial) const;

    /// Small local viewpoint lattice around the current position (section
    /// XV): forward x lateral offsets on BOTH sides, cheap endpoint filter
    /// (in map, known, free with search_clearance_m, distance bounds),
    /// then cheap information-gain / clearance / distance rank.  Returns
    /// at most `max_viewpoint_candidates` candidates; only the top
    /// `emit_budget` are emitted (the ones that will get the FULL
    /// LocalPathSearch in scoreObserved).
    std::vector<MacroCandidate> makeObserveCandidates(
        const ObservedMap& map,
        const VehicleState& state,
        const Eigen::Vector3d& goal_world,
        int emit_budget) const;

    MacroCandidate makeSideCandidate(const ObservedMap& map,
                                     const VehicleState& state,
                                     const Eigen::Vector3d& goal_world,
                                     const GoalBlocker& blocker,
                                     Side side,
                                     double offset_m) const;
    /// Goal-directed frontier candidates (known-free standoff on the known
    /// side of the frontier).  `remaining_searches` is the leftover FULL
    /// search budget after the OBSERVE lattice has taken its share.
    std::vector<MacroCandidate> makeFrontierCandidates(
        const ObservedMap& map,
        const VehicleState& state,
        const Eigen::Vector3d& goal_world,
        int remaining_searches) const;

    /// Fraction of unknown cells in a disk of
    /// `observe_info_gain_radius_m` around `position` (used for cheap
    /// viewpoint ranking and the final candidate score).
    double estimateInfoGain(const ObservedMap& map,
                            const Eigen::Vector3d& position) const;

    MacroCandidateConfig config_;
    /// Per-tick observation diagnostics (mutable: updated inside the const
    /// generateCandidates()/scoreObserved() path).
    mutable ObserveDiagnostics observe_diag_;
};

}  // namespace il_dataset
