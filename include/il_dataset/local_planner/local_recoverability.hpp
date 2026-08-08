#pragma once

#include <Eigen/Core>

#include "il_dataset/local_planner/macro_candidate_search.hpp"
#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class ObservedMap;

/// Configuration for the 5 Hz local-recoverability query (section VI).
struct RecoverabilityConfig {
    /// Direct-guide lookahead used to define the rejoin point.
    double rejoin_distance_m = 2.5;
    /// The single UNIFIED navigation clearance (problem 4): clearance
    /// required along the local path.  The observed ESDF already subtracts
    /// the vehicle radius, so this is the SAME additional margin used by
    /// every other navigation module.
    double clearance_m = 0.20;
    /// Fixed local planning horizon: the path must be executable within it.
    double max_duration_s = 2.5;
    /// Max allowed local path length (m).
    double max_path_length_m = 6.0;
    /// Minimum forward progress toward the goal the local path must yield.
    double min_goal_progress_m = 0.30;
    /// Minimum cosine alignment between the terminal TANGENT and the guide
    /// direction.
    double min_terminal_alignment = 0.5;
    /// Maximum path_length / straight_REJOIN_distance ratio (no loops /
    /// backtracking).  The denominator is the actual rejoin distance, not
    /// the macro-guide distance.
    double max_detour_ratio = 1.6;
    /// Cruise speed used to estimate path duration.
    double nominal_speed_mps = 1.8;
    /// Minimum segment length (m) used to estimate the terminal tangent.
    double terminal_tangent_min_baseline = 0.3;
    /// Side-corridor geometry passed to the blocker analysis.
    double side_corridor_length_m = 4.0;
    double side_corridor_radius_m = 0.55;
    double edge_search_radius_m = 5.0;
};

/// Low-frequency interface exposed by the 30 Hz local layer to the 5 Hz
/// macro expert (section VI).  `DIRECT_REJOIN_SUCCESS` is returned only
/// when the local system can genuinely recover the direct goal intent;
/// a partial path alone is never reported as success.
class LocalRecoverability {
public:
    explicit LocalRecoverability(const RecoverabilityConfig& config);

    /// Test whether the direct goal intent (`direct_guide_world`, a point
    /// on the goal ray) can be resolved by the local system in the observed
    /// map within the fixed horizon.
    RecoverabilityResult test(const ObservedMap& map,
                              const VehicleState& state,
                              const Eigen::Vector3d& direct_guide_world) const;

private:
    RecoverabilityConfig config_;
};

}  // namespace il_dataset
