#pragma once

#include <Eigen/Core>
#include <deque>
#include <utility>

#include "il_dataset/local_planner/types.hpp"

namespace il_dataset {

class PrivilegedOracle;

/// Configuration for the privileged LOCAL-SCALE audit (section II/III).
///
/// The oracle runs a short-range A* on the FULL map with geometry as
/// consistent as possible with the OBSERVED local recoverability.  It
/// ALLOWS local bypass around obstacles — it never requires the direct
/// ray to be collision free.  It uses the SAME rejoin target distance and
/// the SAME capability bounds as the observed local recoverability.
struct PrivilegedInterventionConfig {
    /// The single UNIFIED navigation clearance (problem 4): additional
    /// clearance (m) required for the privileged search.  The global ESDF
    /// already subtracts the vehicle radius, so this is the SAME "extra
    /// safety margin" used by every global module.
    double clearance_m = 0.20;
    double search_max_time_ms = 20.0;
    /// Rejoin target distance (m) — SAME as local_recoverability.
    double rejoin_distance_m = 2.5;
    /// Search region expansion (m) perpendicular to the guide direction.
    /// Must be real metres (a few cells of 0.1 m is NOT enough to bypass).
    double search_lateral_margin_m = 2.0;
    /// Search region expansion (m) along the guide direction.
    double search_longitudinal_margin_m = 2.0;
    /// Local planning horizon / duration bound (s).
    double max_duration_s = 2.5;
    /// Max allowed local path length (m).
    double max_path_length_m = 6.0;
    double nominal_speed_mps = 1.8;
    /// Max detour ratio relative to the STRAIGHT rejoin distance.
    double max_detour_ratio = 1.6;
    /// Minimum forward progress along the goal ray the local path must
    /// yield before "rejoin" counts.
    double min_goal_progress_m = 0.30;
    /// Minimum cosine alignment between the terminal TANGENT and the guide
    /// direction.
    double min_terminal_alignment = 0.5;
    /// Minimum segment length (m) used to estimate the terminal tangent.
    double terminal_tangent_min_baseline = 0.3;
    // ── Loop detection (section IX) ─────────────────────────────────
    /// Recent history within this window (s) is ignored (normal motion).
    double loop_ignore_recent_s = 2.5;
    /// A real revisit requires having left the region beyond this radius.
    double loop_leave_radius_m = 1.6;
    double loop_revisit_radius_m = 0.8;
    double loop_min_speed_mps = 0.3;
    int loop_min_revisits = 2;
    int loop_history_size = 60;
};

/// Low-frequency privileged evaluator (sections II, III, IX, XXI).
///
/// Responsibilities (ONLY):
///  - true local-scale audit (privileged local recoverability),
///  - long-term candidate / route evaluation (via PrivilegedOracle),
///  - global-connectivity filtering of candidates,
///  - loop / dead-end identification,
///  - future-intervention diagnosis (auxiliary labels, never the main
///    macro mode).
///
/// It never emits hidden waypoints and never directly changes the main
/// macro mode without causal history evidence.
class PrivilegedInterventionOracle {
public:
    explicit PrivilegedInterventionOracle(
        const PrivilegedInterventionConfig& config);

    /// Evaluate the direct intent on the full map.  `current_time_s` is
    /// used for loop detection (wall-clock, seconds).
    PrivilegedInterventionResult evaluate(const PrivilegedOracle& oracle,
                                          const VehicleState& state,
                                          const Eigen::Vector3d& direct_guide_world,
                                          const Eigen::Vector3d& goal_world,
                                          double current_time_s);

    /// Clear the loop-history at episode start.
    void reset();

private:
    bool detectLoopRisk(const Eigen::Vector2d& position,
                        double now_s) const;

    PrivilegedInterventionConfig config_;
    // Position history (bounded) for loop detection: (2D position, time).
    std::deque<std::pair<Eigen::Vector2d, double>> history_;
};

}  // namespace il_dataset
