#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class ObservedMap;

/// Configuration for the 30 Hz observed-map A* / JPS-style search.
struct LocalSearchConfig {
    /// Extra margin added on top of the vehicle radius when deciding node
    /// safety (the ESDF already subtracts the vehicle radius).
    double search_clearance_m = 0.25;
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

/// Result of the local path search.  `success` (full path to the requested
/// goal) and `found_any` (partial path toward it) are strictly separated;
/// a failed search never masquerades as a complete path.
struct LocalSearchResult {
    bool success = false;
    bool found_any = false;
    std::vector<Eigen::Vector3d> path;  // world points, start -> terminal
    Eigen::Vector3d terminal{Eigen::Vector3d::Zero()};
    double path_length = 0.0;
    double min_clearance = 0.0;
    std::string failure_reason;
    int expanded_nodes = 0;
    double compute_ms = 0.0;
};

/// 2-D observed-map A* (8-connected) on the drone's height slice.  Used by
/// the 30 Hz local planner for its seed path / terminal selection and by
/// the local recoverability query.
///
///  - Start is the current state; goal is the macro guide.
///  - When the guide is not in known free space, or is unreachable, the
///    search returns the farthest known-safe point toward the guide as a
///    strictly-marked partial result.
///  - The committed side is respected through a lateral bias that never
///    hard-blocks the opposite side (DIRECT_GUIDE allows small detours).
class LocalPathSearch {
public:
    LocalSearchResult search(const ObservedMap& map,
                             const VehicleState& state,
                             const Eigen::Vector3d& goal_world,
                             const LocalSearchConfig& config) const;
};

}  // namespace il_dataset
