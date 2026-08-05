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
    double max_plan_age = 0.75;
    double planning_time_budget_ms = 30.0;
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
    int max_iterations = 10000;
    double convergence_tolerance = 1.0e-4;
    double initial_step_size = 0.1;
    double minimum_step_size = 1.0e-4;
    int max_cost_samples_per_segment = 64;
    // Each free control point stays near the corresponding collision-free
    // seed point.  This preserves the seed homotopy without attracting the
    // trajectory to the straight Guide chord.
    double seed_trust_radius = 0.35;
    // Search/optimize obstacle detours only in x-y. Height follows the direct
    // start-to-terminal profile and is never selected as an avoidance axis.
    bool horizontal_avoidance_only = true;

    // ESDF already subtracts the vehicle radius.  This is only the extra
    // local hard margin; target_clearance remains the soft preference.
    double min_clearance = 0.02;
    double target_clearance = 0.20;
    double collision_check_spacing = 0.05;

    // Cost weights
    double weight_path_length = 0.05;
    double weight_smooth = 1.0;
    double weight_jerk = 0.2;
    double weight_guide = 0.0;
    double weight_obstacle = 4.0;
    double weight_goal = 2.0;
    double weight_dynamics = 1.0;

    // Dynamics limits
    double nominal_speed = 1.8;
    double max_velocity = 2.5;
    double max_acceleration = 8.0;
    double max_jerk = 50.0;
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
/// Thread safety: map/path setters are NOT thread-safe and
/// should be called before planning begins.  planLocalWithRequest() and
/// validateTrajectory() are const from the data perspective and can be
/// called from multiple threads if the ESDF/optional diagnostics are not being
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

    /// Set an optional legacy global reference path for diagnostics only.
    /// Data is COPIED once.
    /// @param path  float64 numpy array, shape [N, 3], C-order
    /// @param n_points  number of waypoints
    /// @return true on success
    bool setGlobalPath(const double* path, int n_points);

    /// Reset planner state for a new trajectory.
    /// @param initial_state  the reset position/orientation of the drone
    void reset(const VehicleState& initial_state);

    // ── Online planning (MUST be called with GIL released) ─────────

    /// Plan a local trajectory with explicit request (Phase 2).
    /// Uses guide_waypoint/trajectory_terminal as the exact local target.
    /// The deprecated reference_path_segment is ignored; an obstructed direct
    /// seed is initialized by a bounded local ESDF search.
    /// Collision checks use isKnownFree() when forbid_unknown_space is true.
    /// @param request  full planning request
    /// @return  LocalPlanResult with trajectory and metadata
    LocalPlanResult planLocalWithRequest(
        const LocalPlanningRequest& request) const;

    /// Return the farthest candidate distance whose terminal is hard-safe
    /// and reachable through the causal observed ESDF. A bounded local A*
    /// search is used when the direct segment is blocked. Full B-spline and
    /// dynamics optimization is intentionally performed only by the formal
    /// 30 Hz local planner after the macro Guide has been selected.
    double findReachableGuideDistance(
        const VehicleState& state,
        const Eigen::Vector3d& direction_world,
        double desired_distance,
        double minimum_distance,
        double distance_step,
        bool forbid_unknown_space) const;

    /// Validate a trajectory for collisions and clearance.
    /// @param trajectory  the trajectory to validate
    /// @return  ValidationResult
    ValidationResult validateTrajectory(
        const std::vector<TrajectoryPoint>& trajectory) const;

    // ── Accessors ──────────────────────────────────────────────────

    const ESDFGrid& esdf() const { return esdf_; }
    const std::vector<Eigen::Vector3d>& globalPath() const { return global_path_; }
    const LocalPlannerConfig& config() const { return config_; }

    /// Return whether the planner has an initialized ESDF.
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

    /// Production explicit-Guide planner.  Uses one cubic B-spline for
    /// optimization, sampling and validation.
    LocalPlanResult planSplineWithRequest(
        const LocalPlanningRequest& request) const;

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
