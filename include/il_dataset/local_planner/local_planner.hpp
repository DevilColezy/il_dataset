#pragma once

#include "il_dataset/local_planner/types.hpp"
#include "il_dataset/local_planner/esdf_grid.hpp"

#include <Eigen/Core>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace il_dataset {

/// Configuration for the local receding-horizon planner.
struct LocalPlannerConfig {
    // Timing
    double planner_hz = 10.0;
    double horizon_time = 2.5;
    double execute_prefix_time = 0.60;
    double max_plan_age = 0.75;
    double planning_time_budget_ms = 40.0;
    double trajectory_dt = 0.04;

    // Lookahead / local goal
    double lookahead_distance = 4.0;
    double min_lookahead_distance = 2.0;
    double max_lookahead_distance = 6.0;
    double lookahead_velocity_gain = 0.6;
    double curvature_lookahead_gain = 1.0;
    double local_map_radius = 6.0;
    int max_reference_points = 32;

    // Optimization
    std::string optimizer = "auto";  // auto | nlopt | native
    int control_points = 12;
    double control_point_spacing = 0.20;
    int max_iterations = 60;
    double convergence_tolerance = 1.0e-4;
    double initial_step_size = 0.1;
    double minimum_step_size = 1.0e-4;
    int max_cost_samples_per_segment = 64;

    // Clearance
    double min_clearance = 0.10;
    double target_clearance = 0.20;
    double collision_check_spacing = 0.05;

    // Cost weights
    double weight_smooth = 1.0;
    double weight_jerk = 0.2;
    double weight_guide = 0.8;
    double weight_obstacle = 4.0;
    double weight_goal = 2.0;
    double weight_dynamics = 1.0;

    // Dynamics limits
    double nominal_speed = 1.8;
    double max_velocity = 2.5;
    double max_acceleration = 3.5;
    double max_jerk = 15.0;
    double max_yaw_rate = 2.0;

    // Goal conditions
    double goal_tolerance = 0.30;
    double goal_speed_tolerance = 0.20;
    int goal_hold_ticks = 3;

    // Failure recovery
    int max_consecutive_failures = 3;
    bool reduce_lookahead_on_failure = true;
    bool emergency_hold_enabled = true;
};

/// Receding-horizon local trajectory planner.
///
/// Thread safety: setESDF() and setGlobalPath() are NOT thread-safe and
/// should be called before planning begins.  planLocal() and
/// validateTrajectory() are const from the data perspective and can be
/// called from multiple threads if the ESDF/global path are not being
/// modified concurrently.
class LocalPlanner {
public:
    /// Construct with configuration.
    explicit LocalPlanner(const LocalPlannerConfig& config);

    ~LocalPlanner();

    // ── Initialization (call once per scene, NOT thread-safe) ──────

    /// Set the ESDF map.  Data is COPIED once into the planner.
    /// @param data  float32 numpy array, shape [gx, gy, gz], C-order
    /// @param gx, gy, gz  dimensions
    /// @param origin_x, origin_y, origin_z  world corner of voxel (0,0,0)
    /// @param resolution  voxel size
    /// @return true on success
    bool setESDF(const float* data,
                 int gx, int gy, int gz,
                 double origin_x, double origin_y, double origin_z,
                 double resolution);

    /// Set observed ESDF with known mask (Phase 2).
    /// @param data  float32 ESDF array [gx, gy, gz]
    /// @param known_mask  uint8 array [gx, gy, gz]
    /// @param gx, gy, gz  dimensions
    /// @param origin_x, origin_y, origin_z  corner
    /// @param resolution  voxel size
    /// @param unknown_is_free  legacy vs observed policy
    /// @return true on success
    bool setObservedESDF(const float* data,
                         const uint8_t* known_mask,
                         int gx, int gy, int gz,
                         double origin_x, double origin_y, double origin_z,
                         double resolution,
                         bool unknown_is_free);

    /// Set the global reference path (A* shortcut output).
    /// Data is COPIED once.
    /// @param path  float64 numpy array, shape [N, 3], C-order
    /// @param n_points  number of waypoints
    /// @return true on success
    bool setGlobalPath(const double* path, int n_points);

    /// Reset planner state for a new trajectory.
    /// @param initial_state  the reset position/orientation of the drone
    void reset(const VehicleState& initial_state);

    // ── Online planning (MUST be called with GIL released) ─────────

    /// Plan a local trajectory from the current drone state (legacy).
    /// Uses internal selectLocalGoal() and global ESDF.
    /// @param current_state  latest kinematic state
    /// @param previous_progress_s  progress along global path from last plan
    /// @return  LocalPlanResult with trajectory and metadata
    LocalPlanResult planLocal(const VehicleState& current_state,
                              double previous_progress_s) const;

    /// Plan a local trajectory with explicit request (Phase 2).
    /// Uses guide_waypoint for reference, trajectory_terminal as optimization
    /// target, and reference_path_segment for B-spline initialisation.
    /// Collision checks use isKnownFree() when forbid_unknown_space is true.
    /// @param request  full planning request
    /// @return  LocalPlanResult with trajectory and metadata
    LocalPlanResult planLocalWithRequest(
        const LocalPlanningRequest& request) const;

    /// Validate a trajectory for collisions and clearance.
    /// @param trajectory  the trajectory to validate
    /// @return  ValidationResult
    ValidationResult validateTrajectory(
        const std::vector<TrajectoryPoint>& trajectory) const;

    // ── Accessors ──────────────────────────────────────────────────

    const ESDFGrid& esdf() const { return esdf_; }
    const std::vector<Eigen::Vector3d>& globalPath() const { return global_path_; }
    const LocalPlannerConfig& config() const { return config_; }

    /// Return whether the planner has been initialized with ESDF and global path.
    bool isReady() const;

    /// Get the current plan ID counter.
    uint64_t currentPlanId() const { return plan_id_counter_; }

private:
    // ── Internal helpers ───────────────────────────────────────────

    /// Compute progress along the global path via arc-length projection.
    /// Searches near previous_progress_s for continuity.
    struct ProgressResult {
        double progress_s = 0.0;
        int segment_index = -1;  // index of segment [idx, idx+1] containing the projection
        double t = 0.0;          // interpolation parameter in [0, 1] within segment
        bool valid = false;
    };
    ProgressResult computeProgress(const Eigen::Vector3d& position,
                                   double previous_progress_s) const;

    /// Select a local goal on the global path given progress and lookahead.
    /// @deprecated Phase 2: use external GuideSelector instead. Kept for legacy.
    struct LocalGoalResult {
        Eigen::Vector3d position{Eigen::Vector3d::Zero()};
        int waypoint_index = -1;
        double arc_length_from_start = 0.0;
        bool is_final_goal = false;
        bool valid = false;
    };
    [[deprecated("Phase 2: use external GuideSelector. Kept for legacy async mode.")]]
    LocalGoalResult selectLocalGoal(double progress_s,
                                    const Eigen::Vector3d& current_position,
                                    double current_speed) const;

    /// Generate yaw from trajectory tangent.
    static double yawFromTangent(const Eigen::Vector3d& direction);

    /// Wrap angle to [-pi, pi].
    static double wrapAngle(double angle);

    /// Densely sample a trajectory from control points (B-spline or minimum-jerk).
    std::vector<TrajectoryPoint> sampleTrajectory(
        const std::vector<Eigen::Vector3d>& control_points,
        const Eigen::Vector3d& start_pos,
        const Eigen::Vector3d& start_vel,
        const Eigen::Vector3d& start_acc,
        const Eigen::Vector3d& goal_pos,
        double start_yaw,
        double dt,
        int num_samples) const;

    /// Sample with an automatically allocated duration, lengthening time
    /// until finite-difference velocity/acceleration limits are satisfied.
    std::vector<TrajectoryPoint> sampleTrajectoryFeasible(
        const std::vector<Eigen::Vector3d>& control_points,
        const Eigen::Vector3d& start_pos,
        const Eigen::Vector3d& start_vel,
        const Eigen::Vector3d& start_acc,
        const Eigen::Vector3d& goal_pos,
        double start_yaw) const;

    /// Compute total cost for a set of control points.
    double computeCost(const std::vector<Eigen::Vector3d>& control_points,
                       const Eigen::Vector3d& start_pos,
                       const Eigen::Vector3d& start_vel,
                       const Eigen::Vector3d& local_goal,
                       const std::vector<Eigen::Vector3d>& global_ref_segment,
                       bool near_final_goal) const;

    /// Optimize control points via gradient descent with line search.
    bool optimizeControlPoints(std::vector<Eigen::Vector3d>& control_points,
                               const Eigen::Vector3d& start_pos,
                               const Eigen::Vector3d& start_vel,
                               const Eigen::Vector3d& local_goal,
                               const std::vector<Eigen::Vector3d>& global_ref_segment,
                               bool near_final_goal) const;

    /// Generate an emergency hold/brake trajectory.
    std::vector<TrajectoryPoint> generateEmergencyHold(
        const VehicleState& current_state) const;

    /// Generate a conservative fallback from global path segment.
    std::vector<TrajectoryPoint> generateGlobalPathFallback(
        const VehicleState& current_state,
        double progress_s,
        double lookahead) const;

    // Precomputed arc-lengths for global path segments.
    void recomputeArcLengths();

    // ── Data ───────────────────────────────────────────────────────

    LocalPlannerConfig config_;
    ESDFGrid esdf_;
    std::vector<Eigen::Vector3d> global_path_;
    std::vector<double> arc_lengths_;  // cumulative arc-length at each waypoint

    // Mutable state for plan_id counter (atomic for thread safety).
    mutable std::atomic<uint64_t> plan_id_counter_{0};

    // Initial/reset state (used as fallback start).
    VehicleState initial_state_;
};

}  // namespace il_dataset
