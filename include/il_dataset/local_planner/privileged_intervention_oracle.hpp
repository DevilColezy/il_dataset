#pragma once

#include <Eigen/Core>
#include <deque>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class PrivilegedOracle;

/// Configuration for the privileged macro-intervention evaluation
/// (section III).
struct PrivilegedInterventionConfig {
    /// Lateral offset (m) used to sample the left/right alternatives at
    /// the direct guide point.
    double lateral_offset_m = 1.5;
    /// Minimum global clearance (m) required along the direct ray.
    double min_global_clearance_m = 0.20;
    /// Max ratio of direct cost-to-go over straight-line distance before
    /// the direct corridor is considered excessively long.
    double max_direct_detour_ratio = 1.6;
    /// A side alternative must beat the direct cost-to-go by at least this
    /// margin (m) to be a "wrong homotopy" signal.
    double cost_margin_m = 2.0;
    /// Distance ahead (m) at which the direct guide consequence is sampled.
    double lookahead_sampling_m = 4.0;
    // Loop-risk history (section III / XIV).
    double loop_revisit_radius_m = 0.8;
    int loop_history_size = 40;
    /// Revisit count (after leaving the radius) that signals a loop.
    int loop_min_revisits = 2;
    double loop_min_speed_mps = 0.3;
};

/// Low-frequency privileged evaluator: decides whether the direct goal
/// intent remains globally viable, i.e. whether continuing to provide a
/// direct-to-goal macro guide lets the 30 Hz local system reach the goal
/// efficiently.  It never outputs hidden waypoints or a hidden "right/left
/// answer"; it only reports whether macro intervention is needed and
/// ranks sides that are already observable.
///
/// Evaluated over: global connectivity, global cost-to-go, direct-ray
/// clearance and detour ratio, wrong-homotopy margin, and loop history.
class PrivilegedInterventionOracle {
public:
    explicit PrivilegedInterventionOracle(
        const PrivilegedInterventionConfig& config);

    /// Evaluate the direct intent.  Keeps a bounded position history for
    /// loop-risk detection.
    PrivilegedInterventionResult evaluate(const PrivilegedOracle& oracle,
                                          const VehicleState& state,
                                          const Eigen::Vector3d& direct_guide_world,
                                          const Eigen::Vector3d& goal_world);

    /// Clear the loop-history at episode start.
    void reset();

private:
    bool detectLoopRisk(const Eigen::Vector3d& position,
                        double speed) const;

    PrivilegedInterventionConfig config_;
    // Position history (bounded) for loop-risk detection.
    std::deque<Eigen::Vector3d> history_;
    // Number of times the drone left and re-entered the revisit radius.
    int revisit_count_ = 0;
    bool outside_revisit_radius_ = true;
};

}  // namespace il_dataset
