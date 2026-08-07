#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

/// Configuration for the privileged global map (section VII).
///
/// UNIFIED ESDF SEMANTICS (section XIV/XV): the global ESDF value is
///     esdf = distance_to_obstacle_surface - vehicle_radius
/// i.e. it represents clearance from the INFLATED vehicle body.  Free
/// space everywhere uses
///     esdf > additional_safety_margin
/// where the additional safety margin is `inflation_m` (connectivity,
/// Dijkstra, candidate feasibility, privileged local recoverability).
/// The vehicle radius is NEVER double counted.
struct PrivilegedOracleConfig {
    double resolution = 0.10;
    double vehicle_radius_m = 0.30;
    /// The single additional safety margin (m) on top of the vehicle
    /// radius used for ALL global free-space definitions.
    double inflation_m = 0.30;
    double max_esdf_distance_m = 8.0;
    /// Margin added around the start-goal bounding box for the grid.
    double map_margin_m = 2.0;
    double min_z_m = 0.0;
    double max_z_m = 8.0;
    /// Cap for cost-to-go values.
    double cost_to_go_cap_m = 30.0;
    /// Candidate scoring weights (lower score = better).
    struct Scoring {
        double weight_observed_cost = 1.0;
        double weight_cost_to_go = 2.0;
        double weight_connectivity = 6.0;
        double weight_clearance = 1.0;
        double weight_goal_progress = 1.0;
        double weight_information = 0.5;
        double weight_yaw_cost = 0.5;
        double weight_side_switch = 1.0;
        double weight_repeat = 0.5;
        double side_switch_penalty = 1.0;
        double repeat_penalty = 1.0;
        double clearance_target_m = 0.6;
        double yaw_cost_scale_rad = 1.0;
    } scoring;
};

/// Privileged global map built once per task from the exported global
/// point cloud: occupancy, global ESDF (distance - vehicle_radius), global
/// connectivity and a goal-reversed cost-to-go.  Free space is everywhere
///  esdf > inflation_m  (unified definition, section XIV).  Used ONLY for
/// macro candidate evaluation and dataset diagnostics — never for the
/// 30 Hz observed-map planning or the student inputs.
class PrivilegedOracle {
public:
    PrivilegedOracle() = default;

    /// Build the SCENE map from a point cloud (world frame): grid +
    /// occupancy + global ESDF only.  A scene is exported and built ONCE
    /// and shared by all of its tasks (section XLIV/LXXI).  `region_min` /
    /// `region_max` optionally force the grid to cover a task domain even
    /// where the point cloud is sparse (e.g. OPEN_DIRECT scenes).
    bool buildScene(const std::vector<Eigen::Vector3d>& points,
                    const PrivilegedOracleConfig& config,
                    const Eigen::Vector3d* region_min = nullptr,
                    const Eigen::Vector3d* region_max = nullptr);

    /// Prepare one TASK on the already-built scene: updates the goal
    /// reversed cost-to-go / connectivity for this task's start-goal.
    /// Returns whether the task is reachable.
    bool setTask(const Eigen::Vector3d& start,
                 const Eigen::Vector3d& goal);

    /// Convenience: buildScene + setTask (single-task legacy path).
    bool build(const std::vector<Eigen::Vector3d>& points,
               const Eigen::Vector3d& start,
               const Eigen::Vector3d& goal,
               const PrivilegedOracleConfig& config);

    bool built() const { return built_; }
    bool taskReachable() const { return task_reachable_; }
    double startGoalDistance() const { return start_goal_distance_; }

    // ── Queries (world -> privileged values) ────────────────────────
    bool isOccupied(double x, double y, double z) const;
    /// Global ESDF clearance (vehicle radius already subtracted).
    double clearance(double x, double y, double z) const;
    /// Unified global free-space test: in-bounds AND
    /// clearance(x,y,z) > required_clearance.  The vehicle radius is
    /// already inside the ESDF value, so `required_clearance` is an extra
    /// safety margin (same definition as inflation_m).
    bool isFree(double x, double y, double z, double required_clearance) const;
    /// Goal-reversed cost-to-go at the drone's height slice (NaN when the
    /// cell is not free / outside the grid).
    double costToGo(double x, double y, double z) const;
    bool connectedToGoal(double x, double y, double z) const;

    /// Fill the privileged fields + score of every candidate.
    void scoreCandidates(std::vector<MacroCandidate>* candidates,
                         const VehicleState& state,
                         const Eigen::Vector3d& goal_world,
                         Side committed_side,
                         const Eigen::Vector3d* previous_guide_world) const;

    /// Privileged best side between left/right side candidates.  Returns
    /// Side::NONE when neither is clearly better.
    Side privilegedBestSide(const std::vector<MacroCandidate>& candidates,
                            double* margin_out = nullptr) const;

    /// Evaluate a world look direction for information value: returns the
    /// privileged cost-to-go at a point `distance` ahead in that direction
    /// (lower = this direction leads toward the goal).
    double directionCostToGo(const Eigen::Vector3d& position_world,
                             const Eigen::Vector3d& direction_world,
                             double distance) const;

    // ── Diagnostics / accessors ─────────────────────────────────────
    int gx() const { return gx_; }
    int gy() const { return gy_; }
    int gz() const { return gz_; }
    const Eigen::Vector3d& origin() const { return origin_world_; }
    double resolution() const { return config_.resolution; }
    const std::vector<float>& esdf() const { return esdf_; }
    const std::vector<float>& costToGoGrid() const { return cost_to_go_; }

private:
    Eigen::Vector3i worldToGridInt(double x, double y, double z) const;
    int zIndexAt(double z) const;
    void buildConnectivityAndCostToGo(const Eigen::Vector3d& start,
                                      const Eigen::Vector3d& goal);

    PrivilegedOracleConfig config_;
    bool built_ = false;
    bool task_reachable_ = false;
    double start_goal_distance_ = 0.0;
    int gx_ = 0;
    int gy_ = 0;
    int gz_ = 0;
    Eigen::Vector3d origin_world_{Eigen::Vector3d::Zero()};
    std::vector<std::uint8_t> occupancy_;  // 0 free, 1 occupied
    std::vector<float> esdf_;
    std::vector<float> cost_to_go_;  // 2D [gx, gy] at a reference z slice
    int cost_to_go_z_ = -1;
    std::vector<std::uint8_t> connected_;  // 2D [gx, gy]
};

}  // namespace il_dataset
