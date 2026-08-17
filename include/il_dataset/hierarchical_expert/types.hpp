#pragma once
/// @file   types.hpp
/// @brief  Shared data types and parameters for the hierarchical local
///         expert integrated into il_dataset (joint_v2 collection).
///
/// This is a SELF-CONTAINED port of the stable "5 Hz local target
/// corrector + 30 Hz local obstacle-avoidance expert" from
/// il_2d_multiscale_debug (v9).  It has NO dependency on the debug
/// package, NO Qt/GUI and NO ROS headers; the ROS/Flightmare lifecycle
/// lives in Python, the expert itself is pure C++17 + Eigen.
///
/// ── 2D frame & yaw convention (documented, self-consistent) ─────────
///   World frame: X → right, Y → up (standard 2D math frame).
///   Yaw is CCW-positive, vehicle forward direction = (cos yaw, sin yaw).
///   Body frame: +X forward (nose), +Y left (90° CCW from nose).
///   A target with positive bearing (atan2(dy,dx) - yaw) is on the LEFT.
///
///   Flightmare FLU adaptation (see coordinate_adapter.hpp — the ONE
///   place that converts between this frame and the Flightmare world):
///       expert_yaw = flightmare_yaw + pi/2
///       expert_position   = flightmare_position  (XY, identity)
///       expert_body vel   = Flightmare FLU velocity (identity)
///       expert yaw_rate   = Flightmare yaw_rate   (identity)
///       expert commands   = Flightmare FLU commands (identity)

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

using Vec2d = Eigen::Vector2d;

// ═══════════════════════════════════════════════════════════════════
//  Small math helpers
// ═══════════════════════════════════════════════════════════════════
inline double deg2rad(double d) { return d * M_PI / 180.0; }
inline double rad2deg(double r) { return r * 180.0 / M_PI; }
inline double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}
/// Wrap an angle into [-pi, pi].
inline double wrapAngle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
}
/// 2D rotation of v by CCW angle a.
inline Vec2d rot2(const Vec2d& v, double a) {
    const double c = std::cos(a), s = std::sin(a);
    return Vec2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}
/// Signed 2D cross product z = a.x*b.y - a.y*b.x.
inline double cross2(const Vec2d& a, const Vec2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

/// TRUE iff the axis-aligned rectangle [min_b, max_b] fully contains the
/// disk of radius `r` centred at `p` (edges AND corners, exact).
inline bool diskInsideBounds(const Vec2d& p, double r, const Vec2d& min_b,
                             const Vec2d& max_b) {
    return p.x() - r >= min_b.x() && p.x() + r <= max_b.x() &&
           p.y() - r >= min_b.y() && p.y() + r <= max_b.y();
}

/// TRUE iff the STRAIGHT segment (x0,y0)→(x1,y1) swept by a disk of
/// radius `r` stays inside the axis-aligned rectangle [min_b, max_b].
///
/// EXACT for axis-aligned rectangles and a straight segment: the
/// rectangle is convex, so the whole segment lies inside the r-shrunk
/// rectangle IFF BOTH endpoints do; every disk along the segment is then
/// fully inside the original rectangle (and any endpoint outside the
/// r-shrunk rectangle leaves it).  This is the SINGLE shared boundary-
/// sweep helper used by TruthCylinderAudit AND PreflightSimulator — no
/// per-class copies and no endpoint-only segSegDist approximation.
inline bool segmentDiskInsideBounds(double x0, double y0, double x1,
                                    double y1, double r, const Vec2d& min_b,
                                    const Vec2d& max_b) {
    return diskInsideBounds(Vec2d(x0, y0), r, min_b, max_b) &&
           diskInsideBounds(Vec2d(x1, y1), r, min_b, max_b);
}

// ═══════════════════════════════════════════════════════════════════
//  WORLD ↔ GLOBAL-GRID index convention (ONE package-wide convention)
//  ═══════════════════════════════════════════════════════════════════
//  Every world-aligned grid (the instantaneous FOV patch AND the merged
//  history map) is anchored at the SAME global grid whose origin is
//  `min_bounds + n*resolution` (grid-aligned).  The unique index rule is:
//      ix = floor((world - min_bounds) / resolution)
//      cell_centre = min_bounds + (ix + 0.5) * resolution
//  floor() handles negative coordinates correctly, and an exact grid
//  boundary is never ambiguous (floor of an integer is that integer).
//  All consumers MUST use these helpers — never re-derive a different
//  formula — so a patch cell and the history cell it merges into are
//  ALWAYS the same global cell.
struct GridIndex2D {
    int ix = 0;
    int iy = 0;
};

/// floor((world - min_bounds) / resolution) for a grid anchored at
/// `min_bounds`.
inline GridIndex2D worldToGrid(const Vec2d& world, const Vec2d& min_bounds,
                               double resolution) {
    const double inv = 1.0 / resolution;
    return {static_cast<int>(std::floor((world.x() - min_bounds.x()) * inv)),
            static_cast<int>(std::floor((world.y() - min_bounds.y()) * inv))};
}

/// World position of a cell centre on a grid anchored at `min_bounds`.
inline Vec2d gridCellCenter(int ix, int iy, const Vec2d& min_bounds,
                            double resolution) {
    return Vec2d(min_bounds.x() + (static_cast<double>(ix) + 0.5) * resolution,
                 min_bounds.y() + (static_cast<double>(iy) + 0.5) * resolution);
}

// ═══════════════════════════════════════════════════════════════════
//  Enum: candidate rejection reasons (first decisive reason only)
// ═══════════════════════════════════════════════════════════════════
enum class CandidateRejectReason : uint8_t {
    NONE = 0,
    NOT_KNOWN_FREE = 1,
    OUTSIDE_CURRENT_FOV = 2,
    OBSERVED_CLEARANCE_TOO_SMALL = 3,
    NO_PROGRESS = 4,
    OTHER = 5,
    INSUFFICIENT_BRAKING_CLEARANCE = 6,
};

inline const char* candidateRejectReasonName(CandidateRejectReason r) {
    switch (r) {
        case CandidateRejectReason::NONE: return "NONE";
        case CandidateRejectReason::NOT_KNOWN_FREE: return "NOT_KNOWN_FREE";
        case CandidateRejectReason::OUTSIDE_CURRENT_FOV: return "OUTSIDE_CURRENT_FOV";
        case CandidateRejectReason::OBSERVED_CLEARANCE_TOO_SMALL:
            return "OBSERVED_CLEARANCE_TOO_SMALL";
        case CandidateRejectReason::NO_PROGRESS: return "NO_PROGRESS";
        case CandidateRejectReason::OTHER: return "OTHER";
        case CandidateRejectReason::INSUFFICIENT_BRAKING_CLEARANCE:
            return "INSUFFICIENT_BRAKING_CLEARANCE";
    }
    return "UNKNOWN";
}

// ═══════════════════════════════════════════════════════════════════
//  Enum: cell state of the local observation
// ═══════════════════════════════════════════════════════════════════
enum class CellState : uint8_t { FREE = 0, OCCUPIED = 1, UNKNOWN = 2 };

// ═══════════════════════════════════════════════════════════════════
//  Enum: FSM states
// ═══════════════════════════════════════════════════════════════════
enum class FsmState : uint8_t {
    DIRECT_LOCAL = 0,
    TURN_TO_TARGET = 1,
    GOAL_REACHED = 2,
    TASK_INVALID = 3,
    COLLISION = 4,
    TIMEOUT = 5,
};

inline const char* fsmStateName(FsmState s) {
    switch (s) {
        case FsmState::DIRECT_LOCAL: return "DIRECT_LOCAL";
        case FsmState::TURN_TO_TARGET: return "TURN_TO_TARGET";
        case FsmState::GOAL_REACHED: return "GOAL_REACHED";
        case FsmState::TASK_INVALID: return "TASK_INVALID";
        case FsmState::COLLISION: return "COLLISION";
        case FsmState::TIMEOUT: return "TIMEOUT";
    }
    return "UNKNOWN";
}

// ═══════════════════════════════════════════════════════════════════
//  Enum: local planner failure reasons
// ═══════════════════════════════════════════════════════════════════
enum class FailureReason : uint8_t {
    NONE = 0,
    TARGET_OUTSIDE_FOV = 1,
    NO_SAFE_CANDIDATE = 2,
    BLOCKED_BY_OBSERVED_OBSTACLE = 3,
    STALLED_WITHOUT_PROGRESS = 4,
};

inline const char* failureReasonName(FailureReason r) {
    switch (r) {
        case FailureReason::NONE: return "NONE";
        case FailureReason::TARGET_OUTSIDE_FOV: return "TARGET_OUTSIDE_FOV";
        case FailureReason::NO_SAFE_CANDIDATE: return "NO_SAFE_CANDIDATE";
        case FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE: return "BLOCKED_BY_OBSERVED_OBSTACLE";
        case FailureReason::STALLED_WITHOUT_PROGRESS: return "STALLED_WITHOUT_PROGRESS";
    }
    return "UNKNOWN";
}

// ═══════════════════════════════════════════════════════════════════
//  Enum: planner candidate-category / outcome
// ═══════════════════════════════════════════════════════════════════
enum class PlannerStatus : uint8_t {
    SAFE_PROGRESSING = 0,
    SAFE_HOLD = 1,
    TERMINAL_SETTLING = 2,
    TURNING = 3,
    EMERGENCY_BRAKE = 4,
    BLOCKED_BY_OBSERVED_OBSTACLE = 5,
    NO_SAFE_CANDIDATE = 6,
    STALLED_WITHOUT_PROGRESS = 7,
    NO_TARGET = 8,
};

inline const char* plannerStatusName(PlannerStatus s) {
    switch (s) {
        case PlannerStatus::SAFE_PROGRESSING: return "SAFE_PROGRESSING";
        case PlannerStatus::SAFE_HOLD: return "SAFE_HOLD";
        case PlannerStatus::TERMINAL_SETTLING: return "TERMINAL_SETTLING";
        case PlannerStatus::TURNING: return "TURNING";
        case PlannerStatus::EMERGENCY_BRAKE: return "EMERGENCY_BRAKE";
        case PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE: return "BLOCKED_BY_OBSERVED_OBSTACLE";
        case PlannerStatus::NO_SAFE_CANDIDATE: return "NO_SAFE_CANDIDATE";
        case PlannerStatus::STALLED_WITHOUT_PROGRESS: return "STALLED_WITHOUT_PROGRESS";
        case PlannerStatus::NO_TARGET: return "NO_TARGET";
    }
    return "UNKNOWN";
}

// ═══════════════════════════════════════════════════════════════════
//  Enum: left/right side selection
// ═══════════════════════════════════════════════════════════════════
enum class SideSelection : uint8_t { NONE = 0, LEFT = 1, RIGHT = 2 };

inline const char* sideName(SideSelection s) {
    switch (s) {
        case SideSelection::NONE: return "NONE";
        case SideSelection::LEFT: return "LEFT";
        case SideSelection::RIGHT: return "RIGHT";
    }
    return "UNKNOWN";
}

// ═══════════════════════════════════════════════════════════════════
//  Parameters (single authoritative source for the whole expert).
//  Python builds this struct from config YAML via the pybind module;
//  there is no second set of default values anywhere else.
// ═══════════════════════════════════════════════════════════════════
struct Params2D {
    // ── region (the Flightmare scene horizontal bounds; grid anchor) ──
    double region_min_x = -20.0, region_max_x = 20.0;
    double region_min_y = -20.0, region_max_y = 20.0;
    // Physical drone radius (m).  Used only for preflight collision audit,
    // never inside the observation-to-command path.
    double drone_radius = 0.15;

    // ── scene safety parameters (STATIC configuration; used by the
    //    30 Hz handoff envelope, never runtime privileged info) ─────
    double scene_safety_clearance = 0.5;
    double macro_route_clearance_margin = 0.1;

    // ── task sampling / judgement ──────────────────────────────────
    double task_goal_tolerance = 0.4;
    // <= 0 disables the episode timeout (wall-tick based).
    double task_episode_timeout_s = 0.0;

    // ── observation (perception range R is the single student input
    //    distance; keep R = 5.0 m for joint_v2) ─────────────────────
    double obs_fov_deg = 90.0;
    double obs_range_m = 5.0;
    double obs_resolution = 0.1;
    double obs_ray_angular_res_deg = 0.5;
    uint32_t obs_history_max_age_ticks = 120;
    // Ground / below-flight-plane filtering: pixels whose 3D point lies
    // more than this far BELOW the camera are the floor / below the flight
    // plane (e.g. the ground at z=0 seen from a ~2 m flight) and are NOT
    // horizontal obstacles.  Without this the expert sees a near floor band
    // as an impassable wall and keeps issuing TURNs.
    double obs_ground_clearance_m = 0.5;

    // ── local planner (30 Hz) ──────────────────────────────────────
    double lp_horizon_s = 2.5;
    double lp_dt = 0.1;
    std::vector<double> lp_speed_samples{0.0, 0.3, 0.6, 1.2, 1.8, 2.5};
    std::vector<double> lp_lateral_ratio_samples{
        -0.5, -0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3, 0.5};
    std::vector<double> lp_yaw_rate_samples{
        -2.0, -1.0, -0.5, -0.25, -0.15, 0.0,
         0.15, 0.25, 0.5, 1.0, 2.0};
    double lp_max_speed = 3.0;
    double lp_max_accel = 2.0;
    double lp_max_yaw_rate = 2.0;
    double lp_max_yaw_accel = 4.0;
    double lp_min_clearance = 0.5;
    double lp_soft_clearance_radius_m = 2.0;
    double lp_clearance_discretization_margin_m = 0.05;
    double lp_obstacle_reaction_time_s = 0.20;
    double lp_control_period_s = 0.0333333333;
    double lp_max_allowed_regress_m = 0.05;
    int    lp_limit_cycle_window_ticks = 15;
    double lp_limit_cycle_net_progress_m = 0.10;
    int    lp_limit_cycle_min_blocked_ticks = 8;
    int    lp_limit_cycle_lateral_flip_count = 2;
    double lp_turn_enter_deg = 42.0;
    double lp_turn_exit_deg = 8.0;
    double lp_turn_exit_max_yaw_rate = 0.15;
    double lp_turn_k = 2.5;
    double lp_near_goal_heading_relax_distance = 1.0;
    double lp_near_goal_turn_enter_deg = 75.0;
    double lp_terminal_control_distance = 1.2;
    double lp_terminal_speed_gain = 1.0;
    double lp_terminal_max_speed = 0.6;
    double lp_terminal_max_yaw_rate = 0.5;
    double lp_min_progress_m = 0.05;       // legacy / diagnostic
    double lp_min_progress_speed_mps = 0.03;
    double lp_min_progress_epsilon_m = 0.01;
    double lp_target_discontinuity_reset_m = 1.5;
    double lp_nominal_clearance_m = 0.65;
    double lp_risk_corridor_half_width = 1.0;
    double lp_risk_distance_horizon_m = 5.0;
    double lp_risk_ttc_horizon_s = 2.5;
    double lp_risk_trajectory_radius_m = 1.0;
    double lp_avoidance_active_threshold = 0.10;
    double lp_brake_stop_margin_m = 0.3;
    double lp_min_executable_prefix_s = 0.2;
    double lp_scoring_horizon_s = 0.8;
    double lp_cost_tie_tolerance = 1e-6;
    double lp_cross_track_normalize_m = 2.0;
    double cost_w_progress = 1.0;
    double cost_w_clearance = 2.0;
    double cost_w_smoothness = 0.5;
    double cost_w_speed_change = 0.3;
    double cost_w_yaw_rate_change = 0.3;
    double cost_w_terminal_heading = 1.0;
    double cost_w_velocity_alignment = 1.2;
    double cost_w_cross_track = 0.8;
    double cost_w_obstacle_risk = 3.0;

    // ── 5 Hz local corrector (visibility judge + target corrector) ──
    double macro_local_failure_duration_s = 0.4;  // legacy, unused by v9
    int    macro_reentry_guard_ticks = 30;
    int    macro_correction_enter_stable_ticks = 1;
    double macro_observable_frontier_min_distance_m = 1.5;
    double macro_observable_frontier_min_progress_m = 0.5;
    int    macro_observable_unknown_margin_cells = 3;
    double macro_side_evidence_margin = 0.5;
    double macro_evidence_ray_step_deg = 1.0;
    int    macro_min_evidence_ray_pairs = 4;
    double macro_corridor_half_width = 1.5;
    double macro_corridor_rear_tolerance_m = 0.5;
    double macro_local_recovery_prefix_m = 0.8;
    double macro_local_candidate_bearing_step_deg = 5.0;
    double macro_local_candidate_distance_step_m = 0.5;
    double macro_local_target_event_tolerance_m = 0.05;
    int    macro_unknown_recovery_threshold_ticks = 60;

    // ── target encoding protocol (R = obs_range_m, reserve 0.5 m) ───
    int    te_direction_bin_count = 11;
    double te_normal_distance_reserve_m = 0.5;
    double te_turn_ray_margin_deg = 10.0;

    // ── depth camera extrinsic (Unity T_BC, camera->body) ──────────
    // T_BC translation in Unity body coords [x right, y up, z forward];
    // rotation row-major 3x3 (identity by default).  These MUST match the
    // camera actually sent to Unity (il_common.make_depth_vehicle), so
    // Flightmare2DObservation derives the TRUE camera pose from the
    // vehicle pose + T_BC instead of assuming cam_pos == vehicle pos.
    double cam_t_bc_x = 0.0;
    double cam_t_bc_y = 0.0;
    double cam_t_bc_z = 0.3;
    std::vector<double> cam_r_bc{1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0};

    // ── vehicle/referee thresholds ─────────────────────────────────
    double vehicle_goal_stop_speed_mps = 0.2;
    double vehicle_stationary_speed_mps = 0.05;
};

// ═══════════════════════════════════════════════════════════════════
//  2D geometric primitives (scene geometry used ONLY for preflight
//  simulation and for the privileged offline scene/task generation; the
//  runtime expert never receives them).
// ═══════════════════════════════════════════════════════════════════
struct Obstacle2D {
    Vec2d center{0.0, 0.0};
    double radius = 0.0;
    int id = -1;
};

struct Scene2D {
    Vec2d min_bounds{-20.0, -20.0};
    Vec2d max_bounds{20.0, 20.0};
    std::vector<Obstacle2D> obstacles;
    uint64_t seed = 0;
    uint64_t scene_id = 0;
    bool valid = false;
};

struct VehicleState2D {
    Vec2d position{0.0, 0.0};
    double yaw = 0.0;
    Vec2d velocity_world{0.0, 0.0};  // world-frame velocity
    double yaw_rate = 0.0;
};

struct Task2D {
    Vec2d start{0.0, 0.0};
    Vec2d goal{0.0, 0.0};
    double initial_yaw = 0.0;
    uint64_t task_id = 0;
    uint64_t scene_id = 0;
    uint64_t seed = 0;
    bool valid = false;
};

// ═══════════════════════════════════════════════════════════════════
//  Result of a bounded nearest-OCCUPIED-cell query on a LocalObservation.
// ═══════════════════════════════════════════════════════════════════
struct NearestOccupiedResult {
    bool found = false;
    double distance = std::numeric_limits<double>::infinity();
    Vec2d cell_center{0.0, 0.0};
};

// ═══════════════════════════════════════════════════════════════════
//  Local observation (world-aligned grid with short-term history)
// ═══════════════════════════════════════════════════════════════════
struct LocalObservation {
    double resolution = 0.1;
    int width = 0, height = 0;
    Vec2d origin{0.0, 0.0};  // world position of cell (0,0)
    std::vector<CellState> cells;
    std::vector<uint32_t> age_ticks;  // ticks since last observed
    uint32_t max_age_ticks = 120;
    uint64_t tick = 0;

    bool valid() const { return width > 0 && height > 0 && !cells.empty(); }
    bool inGrid(int ix, int iy) const {
        return ix >= 0 && iy >= 0 && ix < width && iy < height;
    }
    size_t idx(int ix, int iy) const { return static_cast<size_t>(iy) * width + ix; }
    CellState at(int ix, int iy) const {
        return inGrid(ix, iy) ? cells[idx(ix, iy)] : CellState::UNKNOWN;
    }
    CellState atWorld(double x, double y) const {
        const GridIndex2D g = worldToGrid(Vec2d(x, y), origin, resolution);
        return at(g.ix, g.iy);
    }
    bool isKnownFree(double x, double y) const { return atWorld(x, y) == CellState::FREE; }
    bool isOccupied(double x, double y) const { return atWorld(x, y) == CellState::OCCUPIED; }

    /// Visit every observed OCCUPIED cell whose centre lies inside the
    /// closed radius-r disk around p.
    template <typename Visitor>
    void forEachOccupiedWithin(const Vec2d& p, double search_radius,
                               Visitor&& visitor) const {
        if (!valid() || !(search_radius > 0.0) ||
            !std::isfinite(search_radius)) {
            return;
        }
        const double r = search_radius;
        const double r2 = r * r;
        const int ix0 = static_cast<int>(std::floor(
            (p.x() - r - resolution - origin.x()) / resolution));
        const int ix1 = static_cast<int>(std::ceil(
            (p.x() + r + resolution - origin.x()) / resolution));
        const int iy0 = static_cast<int>(std::floor(
            (p.y() - r - resolution - origin.y()) / resolution));
        const int iy1 = static_cast<int>(std::ceil(
            (p.y() + r + resolution - origin.y()) / resolution));
        for (int iy = std::max(0, iy0); iy <= std::min(height - 1, iy1);
             ++iy) {
            for (int ix = std::max(0, ix0); ix <= std::min(width - 1, ix1);
                 ++ix) {
                if (cells[idx(ix, iy)] != CellState::OCCUPIED) continue;
                const Vec2d centre(origin.x() + (ix + 0.5) * resolution,
                                   origin.y() + (iy + 0.5) * resolution);
                const double d2 = (centre - p).squaredNorm();
                if (d2 > r2 + 1e-12) continue;
                visitor(centre, std::sqrt(std::max(0.0, d2)));
            }
        }
    }

    /// Nearest observed OCCUPIED cell (centre + distance) inside the
    /// radius-r search neighbourhood.
    NearestOccupiedResult nearestOccupied(const Vec2d& p,
                                          double search_radius) const {
        NearestOccupiedResult out;
        double best = std::numeric_limits<double>::infinity();
        Vec2d best_centre(0.0, 0.0);
        forEachOccupiedWithin(
            p, search_radius, [&](const Vec2d& centre, double distance) {
                if (distance < best) {
                    best = distance;
                    best_centre = centre;
                }
            });
        if (best < std::numeric_limits<double>::infinity()) {
            out.found = true;
            out.distance = best;
            out.cell_center = best_centre;
        }
        return out;
    }

    /// Minimum distance from p to the centre of an observed OCCUPIED cell
    /// inside the radius-r search neighbourhood.
    double minClearanceToOccupied(const Vec2d& p, double r) const {
        return nearestOccupied(p, r).distance;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  Local target (the ONLY goal information the 30 Hz planner receives)
// ═══════════════════════════════════════════════════════════════════
struct LocalTarget {
    Vec2d position{0.0, 0.0};
    bool valid = false;
    // Directive update event (5 Hz ZOH boundary).  A change alone NEVER
    // resets planner memory.
    uint64_t update_event = 0;
    // MISSION revision.  Changed ONLY on a new task / scene reset and on a
    // formally accepted final navigation-goal revision.  5 Hz correction
    // enter / refresh / exit NEVER change it.
    uint64_t mission_revision = 0;
    // The second channel of the public 30 Hz target contract.  Ordinary
    // targets are strictly below 1; an exact value of 1 is the reserved
    // pure-rotation command.
    double normalized_distance = 0.0;
};

// ═══════════════════════════════════════════════════════════════════
//  Trajectory / planner result
// ═══════════════════════════════════════════════════════════════════
struct Trajectory2D {
    std::vector<Vec2d> points;
    std::vector<double> yaw;
    std::vector<double> t;  // seconds from plan start
    bool valid = false;
};

struct PlannerResult {
    bool success = false;
    bool turn_mode = false;
    FailureReason failure_reason = FailureReason::NONE;
    // intent_* = the LONG-TERM rollout intent.  vx_body/vy_body/yaw_rate
    // are the EXECUTABLE OUTPUT — the only thing sent to the backend and
    // recorded as the 30 Hz expert label.
    double vx_body = 0.0;
    double vy_body = 0.0;
    double yaw_rate = 0.0;
    double intent_vx_body = 0.0;
    double intent_vy_body = 0.0;
    double intent_yaw_rate = 0.0;
    PlannerStatus planner_status = PlannerStatus::NO_SAFE_CANDIDATE;
    bool candidate_progress_qualified = false;
    bool output_progress_qualified = false;
    bool progress_qualified = false;
    double nominal_progress_m = 0.0;
    double executable_progress_m = 0.0;
    double safe_prefix_duration_s = 0.0;
    double selected_output_speed_mps = 0.0;
    bool stationary_candidate_selected = false;
    std::string stationary_selection_reason;
    bool target_discontinuity_reset = false;
    uint64_t target_mission_revision = 0;
    Trajectory2D selected;
    std::vector<Trajectory2D> rejected_candidates;
    bool blocked_observed = false;
    bool immediate_avoidance = false;
    bool emergency_brake = false;
    double min_observed_clearance = std::numeric_limits<double>::infinity();
    double selected_soft_min_clearance_m = std::numeric_limits<double>::quiet_NaN();
    double selected_dynamic_required_clearance_m = std::numeric_limits<double>::quiet_NaN();
    double selected_closing_speed_mps = std::numeric_limits<double>::quiet_NaN();
    double handoff_clearance_m = std::numeric_limits<double>::quiet_NaN();
    bool dynamic_clearance_blocked = false;
    bool local_limit_cycle_detected = false;
    uint32_t dynamic_window_candidate_count = 0;
    uint32_t reject_not_known_free = 0;
    uint32_t reject_outside_current_fov = 0;
    uint32_t reject_observed_clearance_too_small = 0;
    uint32_t reject_no_progress = 0;
    uint32_t reject_other = 0;
    uint32_t reject_insufficient_braking_clearance = 0;
    double target_bearing_error_deg = 0.0;
    double selected_terminal_heading_error_deg = 0.0;
    double selected_velocity_alignment_error_deg = 0.0;
    double selected_cross_track_error_m = 0.0;
    double selected_cost_total = 0.0;
    double selected_cost_progress = 0.0;
    double selected_cost_clearance = 0.0;
    double selected_cost_smoothness = 0.0;
    double selected_cost_speed_change = 0.0;
    double selected_cost_yaw_rate_change = 0.0;
    double selected_cost_terminal_heading = 0.0;
    double selected_cost_velocity_alignment = 0.0;
    double selected_cost_cross_track = 0.0;
    double selected_cost_obstacle_risk = 0.0;
    bool local_corridor_blocked = false;
    double first_blocking_obstacle_distance = std::numeric_limits<double>::quiet_NaN();
    double predicted_closest_clearance = std::numeric_limits<double>::quiet_NaN();
    double time_to_collision = std::numeric_limits<double>::quiet_NaN();
    double obstacle_risk_cost = 0.0;
    double avoidance_strength = 0.0;
    bool avoidance_active = false;
    double local_target_distance = 0.0;
};

// ═══════════════════════════════════════════════════════════════════
//  v9: 5 Hz target-correction types (local observability judge)
// ═══════════════════════════════════════════════════════════════════
enum class TargetCorrectionType : uint8_t {
    PASS_THROUGH = 0,
    NORMAL_CORRECTION = 1,
    TURN_LEFT = 2,
    TURN_RIGHT = 3,
};

inline const char* targetCorrectionTypeName(TargetCorrectionType t) {
    switch (t) {
        case TargetCorrectionType::PASS_THROUGH: return "PASS_THROUGH";
        case TargetCorrectionType::NORMAL_CORRECTION: return "NORMAL_CORRECTION";
        case TargetCorrectionType::TURN_LEFT: return "TURN_LEFT";
        case TargetCorrectionType::TURN_RIGHT: return "TURN_RIGHT";
    }
    return "UNKNOWN";
}

/// The 5 Hz corrector's output, zero-order held between 5 Hz boundaries.
struct TargetCorrectionDirective {
    TargetCorrectionType type = TargetCorrectionType::PASS_THROUGH;
    bool valid = true;
    // Student direction class: 0=TURN_LEFT; 1..N ordinary in-FOV bins
    // ordered LEFT-to-RIGHT; N+1=TURN_RIGHT.  -1 for PASS_THROUGH.
    int direction_token = -1;
    // Direction decoded at the 5 Hz decision instant (bin centre for an
    // ordinary target, initial FOV-external ray for a bounded TURN step).
    Vec2d decoded_direction_body{1.0, 0.0};
    // DECODED normalized distance (student label).  Ordinary classes are
    // clamped to normal_distance_max < 1; TURN classes are EXACTLY 1.0;
    // PASS_THROUGH is meaningless (0, masked by macro_param_valid).
    double normalized_distance = 0.0;
    // NORMAL_CORRECTION: world point locked during the 5 Hz period.
    Vec2d corrected_target_world{0.0, 0.0};
    bool corrected_target_world_valid = false;
    // TURN_LEFT / TURN_RIGHT: world-frame UNIT direction captured when the
    // bounded turn step is issued.
    Vec2d turn_direction_world{1.0, 0.0};
    bool turn_direction_world_valid = false;
    // Side locked for the current correction episode (NONE when not
    // correcting).
    SideSelection locked_side = SideSelection::NONE;
    // Directive update event: increments ONLY when the directive value
    // changes on a real 5 Hz boundary.
    uint64_t update_event = 0;
    std::string reason = "PASS_THROUGH";
};

/// Local observability judgement of the 5 Hz corrector.
struct AvoidanceObservability {
    bool goal_inside_fov = false;
    bool direct_corridor_blocked = false;
    bool blocker_observed = false;
    bool left_bypass_observable = false;
    bool right_bypass_observable = false;
    bool local_avoidance_observable = false;
    bool fov_boundary_truncated = false;
    bool unknown_occluded = false;
    double left_score = 0.0;
    double right_score = 0.0;
    std::string reason = "NONE";
};

/// The adapter's per-30Hz-tick output — the exact information bottleneck
/// shared by the C++ 30 Hz expert (world target) and a future 30 Hz
/// student (body direction + normalized distance).
struct EncodedTargetInput {
    bool valid = false;
    Vec2d direction_body{1.0, 0.0};
    double normalized_distance = 0.0;
    Vec2d effective_target_world{0.0, 0.0};
    bool effective_target_world_valid = false;
    TargetCorrectionType source_type = TargetCorrectionType::PASS_THROUGH;
};

}  // namespace expert
}  // namespace il_dataset
