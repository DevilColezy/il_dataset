#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

/// Configuration for the privileged global map (section VII).
struct PrivilegedOracleConfig {
    double resolution = 0.10;
    double vehicle_radius_m = 0.30;
    /// Extra inflation added on top of the vehicle radius for the
    /// connectivity / cost-to-go free-space definition.
    double inflation_m = 0.30;
    double max_esdf_distance_m = 8.0;
    /// Margin added around the start-goal bounding box for the grid.
    double map_margin_m = 2.0;
    double min_z_m = 0.0;
    double max_z_m = 8.0;
    /// Minimum clearance for a cell to be considered free (connectivity).
    double free_clearance_m = 0.10;
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
/// point cloud: occupancy, inflated obstacle map, global ESDF, global
/// connectivity and a goal-reversed cost-to-go.  Used ONLY for macro
/// candidate evaluation and dataset diagnostics — never for the 30 Hz
/// observed-map planning or the student inputs.
class PrivilegedOracle {
public:
    PrivilegedOracle() = default;

    /// Build the global map from a point cloud (world frame).  Returns
    /// false when the point cloud is empty or the grid is degenerate.
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
    const std::vector<std::uint8_t>& inflatedOccupancy() const {
        return inflated_;
    }

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
    std::vector<std::uint8_t> inflated_;   // 0 free, 1 blocked (inflated)
    std::vector<float> esdf_;
    std::vector<float> cost_to_go_;  // 2D [gx, gy] at a reference z slice
    int cost_to_go_z_ = -1;
    std::vector<std::uint8_t> connected_;  // 2D [gx, gy]
};

}  // namespace il_dataset
