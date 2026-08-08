#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class ObservedMap;

/// Configuration for the 30 Hz observed-map A* / JPS-style search.
struct LocalSearchConfig {
    /// The single UNIFIED navigation clearance (problem 4): the extra
    /// margin on top of the vehicle radius required for a node to be safe.
    /// The observed ESDF already subtracts the vehicle radius, so this is
    /// the SAME additional margin used by the global connectivity and every
    /// other module — never a second, more permissive boundary.
    double clearance_m = 0.20;
    /// Lateral bias toward the committed side: per-metre path-cost penalty
    /// for nodes on the opposite side of the goal ray (0 disables).
    double side_bias_gain = 2.0;
    /// Committed macro side. SIDE_GUIDE searches must respect it (soft bias,
    /// never a hard block). DIRECT_GUIDE passes NONE.
    Side committed_side = Side::NONE;
    int max_expansions = 200000;
    double max_time_ms = 20.0;
    /// Search region margin around the start/goal bbox.
    double region_margin_m = 2.0;
    bool forbid_unknown = true;
};

/// Result of the local path search (section V).
///
/// FULL_GOAL_REACHED is reported only when the ORIGINAL goal lies inside
/// the map, is known, has enough clearance, the A* connects to the goal
/// grid cell AND the continuous segment from the goal grid centre to the
/// exact goal is verified known-safe.  Otherwise a strictly-marked partial
/// terminal (a genuinely verified known-safe position) is returned.
struct LocalPathResult {
    enum class Status {
        FULL_GOAL_REACHED,
        PARTIAL_TERMINAL_REACHED,
        NO_PATH,
    };
    Status status = Status::NO_PATH;
    bool full_goal_reached = false;  // == (status == FULL_GOAL_REACHED)
    bool found_partial = false;      // status != NO_PATH
    /// Verified known-safe terminal (the goal for FULL_GOAL_REACHED).
    Eigen::Vector3d terminal{Eigen::Vector3d::Zero()};
    std::vector<Eigen::Vector3d> path;  // world points, start -> terminal
    double path_cost = 0.0;
    double minimum_clearance = 0.0;
    std::string failure_reason;
    int expanded_nodes = 0;
    double compute_ms = 0.0;
};

/// 2-D observed-map A* (8-connected) on the drone's height slice.  Used by
/// the 30 Hz local planner for its seed path / terminal selection and by
/// the local recoverability query.
///
///  - Start is the current state; goal is the macro guide.
///  - When the guide is not a valid known-free in-map target, or is
///    unreachable, the search returns the farthest known-safe point toward
///    the guide as a strictly-marked partial result.
///  - The committed side is respected through a lateral bias that never
///    hard-blocks the opposite side (DIRECT_GUIDE allows small detours).
class LocalPathSearch {
public:
    LocalPathResult search(const ObservedMap& map,
                           const VehicleState& state,
                           const Eigen::Vector3d& goal_world,
                           const LocalSearchConfig& config) const;
};

}  // namespace il_dataset
