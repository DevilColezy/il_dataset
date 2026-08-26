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
#include <utility>
#include <vector>

namespace il_dataset {
namespace expert {

using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;

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
    Z_BOUNDS_VIOLATED = 7,
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
        case CandidateRejectReason::Z_BOUNDS_VIOLATED:
            return "Z_BOUNDS_VIOLATED";
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
//  Enum: why the local corridor to the target was judged BLOCKED (R25)
// ═══════════════════════════════════════════════════════════════════
//  CURRENT_OCCUPIED  — blocking OCCUPIED cell observed THIS tick (age 0).
//  HISTORY_OCCUPIED  — blocking OCCUPIED cell only present in the merged
//                      short-term history (age > 0; possibly stale).
//  TARGET_NOT_FREE   — the target cell itself is OCCUPIED in the map.
//  UNKNOWN_TARGET    — the target cell is UNKNOWN (never observed free).
//  START_NOT_FREE    — the vehicle's own cell is not known-free.
//  CLEAR             — corridor free (diagnostic default).
enum class CorridorBlockReason : uint8_t {
    CLEAR = 0,
    CURRENT_OCCUPIED = 1,
    HISTORY_OCCUPIED = 2,
    TARGET_NOT_FREE = 3,
    UNKNOWN_TARGET = 4,
    START_NOT_FREE = 5,
};

inline const char* corridorBlockReasonName(CorridorBlockReason r) {
    switch (r) {
        case CorridorBlockReason::CLEAR: return "CLEAR";
        case CorridorBlockReason::CURRENT_OCCUPIED: return "CURRENT_OCCUPIED";
        case CorridorBlockReason::HISTORY_OCCUPIED: return "HISTORY_OCCUPIED";
        case CorridorBlockReason::TARGET_NOT_FREE: return "TARGET_NOT_FREE";
        case CorridorBlockReason::UNKNOWN_TARGET: return "UNKNOWN_TARGET";
        case CorridorBlockReason::START_NOT_FREE: return "START_NOT_FREE";
    }
    return "CLEAR";
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
    // R25: a cell previously observed OCCUPIED is cleared back to FREE only
    // after this many CONSECUTIVE current-frame FREE confirmations (a
    // single-frame depth gap must not erase a real obstacle, but stale
    // history must not fabricate a permanent blockage either).
    uint32_t obs_free_clear_confirmations = 3;
    // Occupied cells older than this are retained for observability but are
    // not allowed to hard-block a new local trajectory.  The local planner
    // still sees current and recent history, while a stale cell must be
    // refreshed before it can force a HOLD/macro takeover.
    uint32_t lp_planning_history_max_age_ticks = 45;
    // R26: within this distance of the ORIGINAL goal the macro layer locks
    // terminal capture: it must never issue a locked-side search TURN; the
    // target is handed to local (rotate toward the goal / micro-approach).
    // Only a genuine HARD corridor block may release the capture state.
    double macro_terminal_capture_radius_m = 1.0;
    // R26: macro-level limit-cycle watchdog on the ORIGINAL goal.  The
    // LOCAL detector watches the current effective target, but the macro
    // switches goal<->TURN so each switch resets the local bearing
    // evidence.  When the original-goal distance has not decreased by
    // >= macro_limit_cycle_goal_progress_m for macro_limit_cycle_window_5hz
    // consecutive 5 Hz updates, the macro forces a handoff to local.
    double macro_limit_cycle_goal_progress_m = 0.1;
    int    macro_limit_cycle_window_5hz = 15;   // 3 s at 5 Hz
    // Ground / below-flight-plane filtering: pixels whose 3D point lies
    // more than this far BELOW the camera are the floor / below the flight
    // plane (e.g. the ground at z=0 seen from a ~2 m flight) and are NOT
    // horizontal obstacles.  Without this the expert sees a near floor band
    // as an impassable wall and keeps issuing TURNs.
    double obs_ground_clearance_m = 0.5;

    // ── local planner (30 Hz) ──────────────────────────────────────
    double lp_horizon_s = 4.0;
    double lp_dt = 0.1;
    // Max speed cap (m/s).  Set equal to the desired cruise speed (2.0) so
    // the command can never exceed the validated cruise level — the
    // reachableCommand hard cap and every min(cruise, max_speed) now agree.
    double lp_max_speed = 2.0;
    // Desired cruise speed for the FOV-boundary planner (m/s).  The boundary
    // subgoal is driven at this speed ("末点速度最高可达期望速度") and the
    // terminal approach caps at it.  Lower than lp_max_speed for smoother,
    // more conservative flight (user: 2 m/s).
    double lp_cruise_speed_mps = 2.0;
    // ── R29h: simplified speed law for the ray sector (user redesign) ──
    //   v_des = cruise · goal_decay(goal_along_ray) · yaw_decay(|ray_b|)
    //   goal_decay: linear 1.0 → 0 as the goal's projection along the ray
    //     drops from lp_goal_decay_range_m to the capture radius — the
    //     closer to the point, the slower, stop AT it (velocity 0).
    //   yaw_decay: 1.0 when the velocity runs along the nose, falling to
    //     lp_yaw_decay_min the more the velocity direction deviates from
    //     the nose (sideways flight is slower), never below
    //     lp_vmin_speed_mps while still progressing on a clear ray.
    //   Nose at/into a blocker (nose_clear ≤ handoffClearance) → HARD STOP
    //     (velocity 0); the yaw intent keeps slewing so flight resumes once
    //     the nose clears.  Lost target / every ray blocked / target out of
    //     FOV → 0 via the existing hand-off branches.
    //   Replaces the old √(2·a·clearance) raySectorSpeed law + avoid_scale
    //     + nose-halving (R29c).
    double lp_goal_decay_range_m = 2.0;   // full speed beyond this projection
    double lp_vmin_speed_mps = 0.5;       // min forward speed on a clear ray
    double lp_yaw_decay_per_deg = 0.0111; // 45° off-nose → ×0.5
    double lp_yaw_decay_min = 0.5;        // yaw-decay floor
    // R26: inside this distance to a TERMINAL target (in FOV, hard-clear
    // straight path) the planner skips the B-spline and drives the
    // proportional terminal controller directly.  Bridges the 0.4-0.8 m
    // dead zone where the exact-goal B-spline + pullbacks fail and the
    // drone reported BLOCKED at 0.48 m (task 65, 36 s infinite turn loop).
    double lp_terminal_micro_approach_m = 0.8;
    // ── R27: receding-horizon tracking (plan/track split) ─────────
    // The B-spline is re-optimised every `lp_replan_interval_ticks` control
    // ticks (3 = 10 Hz); between replans the drone PURSUES the committed
    // trajectory instead of chasing a freshly re-solved head every tick.
    // This stops the receding-horizon head-drift accumulation (the executed
    // path becomes the envelope of per-tick constraint-satisfying heads
    // rather than one spline).  1 disables tracking (replan every tick).
    int    lp_replan_interval_ticks = 3;
    // Pure-pursuit lookahead along the planned B-spline (m).  The command
    // steers at the trajectory point this far AHEAD of the closest point
    // (instead of only plan.points[1]), so each executed segment is longer
    // and the plan tail is consumed rather than re-solved away.
    double lp_pursuit_lookahead_m = 0.6;
    // Max cross-track deviation from the stored trajectory before a forced
    // replan (m).
    double lp_track_max_cross_track_m = 0.5;
    // Minimum arc length still ahead of the drone on the stored trajectory
    // before a forced replan (m).  Below this the plan is nearly consumed
    // and the next hop must be planned.
    double lp_track_min_front_m = 0.8;
    // Command-ramp limit for reachableCommand (m/s^2): the executable
    // command changes by at most lp_max_accel*dt per tick, and because
    // reachableCommand ramps relative to the PREVIOUS COMMAND the command
    // (and hence the backend feedforward) runs at this rate — this is the
    // achieved closed-loop acceleration.
    double lp_max_accel = 2.0;
    // The EFFECTIVE (physically achieved) horizontal acceleration of the
    // closed loop (m/s^2).  With the command-ramp + backend feedforward
    // (kp=8/kd=1.2, max_accel 4) the drone tracks the lp_max_accel ramp,
    // so this equals lp_max_accel = 2.0.  All braking / clearance /
    // trajectory-profile calculations MUST use this value.  (Historically
    // state-pinning capped it at ~0.42 m/s^2 and made the planner believe
    // it could stop in 1 m at 2 m/s when it actually needed ~5 m — goal
    // overshoot collisions, sc3/tk254.)
    double lp_eff_accel_mps2 = 2.0;
    double lp_max_yaw_rate = 2.0;
    // R28c: raise the yaw command-ramp rate (was 4 -> ~0.6 s to reverse a
    // full-rate turn; too clumsy to follow a freshly re-planned trajectory).
    // The command is slew-limited by lp_max_yaw_accel*dt per tick; 8 rad/s^2
    // halves the reversal time (~0.3 s).  The backend yaw controller tracks
    // the command with feedforward, so a faster ramp is achievable.
    double lp_max_yaw_accel = 8.0;
    // Unified collision distance (USER DIRECTIVE 2026-08-20): obstacle
    // minimum collision = 4 grid cells = 0.4 m from an OCCUPIED cell centre
    // (= drone radius 0.3 + cell 0.1).  This is the shared static base;
    // validation adds discretisation, reaction and stopping distance.
    double lp_min_clearance = 0.4;
    // Soft clearance radius for the early-avoidance gradient.  User
    // directive: the drone must be able to navigate at 0.2 m from a surface
    // (centre 0.5 m = drone radius 0.3 + safety 0.2).  A 2.0 m radius made a
    // 1.94 m corridor gap "impassable" (every path inside it maxed the soft
    // cost) and pushed the planner into wide detours / traps.  1.0 m keeps a
    // gradient while leaving the 0.5..1.0 m band cheap to navigate.
    double lp_soft_clearance_radius_m = 1.0;
    // ── vertical channel (3D extension) ────────────────────────────
    double lp_max_vz = 1.0;        // max vertical speed (m/s, +up)
    double lp_max_v_accel = 2.0;   // vertical acceleration limit (m/s^2)
    double lp_vz_kp = 1.0;         // altitude regulation gain (1/s)
    double lp_z_min_m = 0.8;       // lower altitude bound (floor safety)
    double lp_z_max_m = 3.0;       // upper altitude bound (ceiling safety)
    double lp_vertical_clearance_m = 0.3;  // margin inside the band
    // Robustness for grid quantisation, depth projection and one-period
    // closed-loop tracking between occupied-cell and truth geometry.
    double lp_clearance_discretization_margin_m = 0.15;
    double lp_obstacle_reaction_time_s = 0.20;
    // ── EGO-style optimisation B-spline (R19) ─────────────────────
    // Optimization-based local path planning (structure copied from the
    // open-source ZJU-FAST-Lab/ego-planner BsplineOptimizer + the okazaki
    // L-BFGS solver, see ego_bspline.hpp/.cpp).  Bends the cubic B-spline
    // around OBSERVED obstacles instead of the straight-ray degeneracy of
    // the old core (collinear control points = zero curvature = cannot
    // detour -> boxed-in deadlocks in joint_v2).  The optimised geometry
    // is still validated with the hard clearance + dynamic envelope and
    // falls back to the straight-line planner / escape-rotate / brake.
    bool ego_enabled = true;
    double ego_lambda_smooth = 0.5;      // jerk elastic band
    double ego_lambda_collision = 2.0;   // obstacle push (sample-based)
    double ego_lambda_feasibility = 0.2; // velocity/accel feasibility
    double ego_lambda_fitness = 0.8;     // guidance toward the detour guide
    double ego_lambda_fov = 0.3;         // soft FOV-wedge penalty
    double ego_clearance_m = 0.4;        // collision distance = 4 cells (0.3 radius + 0.1 cell)
    double ego_ts = 0.4;                 // initial B-spline knot span (s)
    int    ego_n_segments = 8;
    int    ego_max_iter = 60;
    // R27: temporal anchoring weight for the EGO B-spline.  When > 0 a soft
    // ego_lambda_ref * |q - ref|^2 cost pulls consecutive replans toward the
    // previous plan (receding-horizon continuity); the free control points
    // are ALWAYS warm-started from a compatible previous plan regardless of
    // this weight.  0 disables only the cost term.
    double ego_lambda_ref = 0.3;
    double lp_control_period_s = 0.0333333333;
    double lp_turn_enter_deg = 42.0;
    double lp_turn_exit_deg = 8.0;
    double lp_turn_exit_max_yaw_rate = 0.15;
    double lp_turn_k = 2.5;
    // R29c: EMA weight for the ray-sector yaw intent (1.0 = no smoothing,
    // smaller = smoother heading but slower response).  0.35 eases a 5°
    // ray-switch step over ~3-4 ticks while a sustained goal-regaining
    // turn still converges.
    double lp_yaw_smooth_alpha = 0.35;
    // R28c: the local planner must NOT produce big lateral detours — that is
    // the UPPER planner's job (it places bypass waypoints around large
    // blockers).  A selected plan whose endpoint direction deviates from the
    // current target direction by more than this (deg) is rejected and
    // handed back to the macro (NO_SAFE_CANDIDATE).  The local only makes
    // SMALL adjustments inside this band, never a ~40°-off-target side
    // excursion (measured task 33: 315° plan vs 274° goal -> east spiral).
    // R28g: 30 -> 35 (= the +-35° scan band).  A razor-thin 30° cliff killed
    // a working local detour when the nose drifted a few degrees across it
    // (task 401: the east-side pass around obs7 needs ~31°, the plan was
    // 0.8 m clear, yet NO_SAFE_CANDIDATE -> 14 s stall).  At 35° the local
    // may use its full scan band; beyond it (task-33-style 40°+) the plan is
    // still rejected and handed to the macro.
    double lp_max_local_deviation_deg = 35.0;
    // Prefer the smallest local steering correction first.  The planner may
    // expand to lp_max_local_deviation_deg only when no candidate exists in
    // this preferred band, preventing an otherwise clear small opening from
    // being replaced by a FOV-edge detour.
    double lp_preferred_local_deviation_deg = 20.0;
    double lp_near_goal_heading_relax_distance = 1.0;
    double lp_near_goal_turn_enter_deg = 75.0;
    double lp_terminal_speed_gain = 1.0;
    double lp_terminal_max_speed = 0.6;
    double lp_terminal_max_yaw_rate = 0.5;
    double lp_min_progress_speed_mps = 0.03;
    double lp_target_discontinuity_reset_m = 1.5;
    double lp_nominal_clearance_m = 0.4;  // = 4 cells (collision distance)
    double lp_risk_corridor_half_width = 1.0;
    double lp_brake_stop_margin_m = 0.1;  // truth-judge edge floor: 0.4 - 0.3 = 0.1 m from surface

    // ── 5 Hz local corrector (visibility judge + target corrector) ──
    double macro_observable_frontier_min_distance_m = 1.5;
    double macro_observable_frontier_min_progress_m = 0.5;
    // R29k: SEARCH_ROTATION_TOWARD_ORIGINAL_GOAL only re-acquires the
    // original goal when the goal bearing is traversable — a continuous
    // FREE run along it of at least this range (m).  When the goal sits
    // behind the blocker (occluded / UNKNOWN), pulling the nose back to it
    // just yaws the drone back and forth across the blocked bearing
    // (measured _failed/000006_02cac96b: TURN_LEFT↔TURN_RIGHT, zero
    // displacement); the corrector keeps the LOCKED bypass side instead.
    double macro_goal_direction_min_range_m = 2.0;
    // R29l: minimum along-goal progress a fresh 5 Hz waypoint candidate
    // must beat the currently held waypoint by (m) before it is adopted.
    // A margin prevents preview noise from jittering the waypoint every
    // boundary while still letting a nearer detour exit take over.
    double macro_waypoint_update_along_margin = 0.3;
    // R29m: SEARCH_ROTATION_TOWARD_ORIGINAL_GOAL cooldown (in 5 Hz
    // corrector updates).  After the corrector pulled the nose toward the
    // (possibly occluded) original goal once, it must not immediately
    // re-pull when depth-derived corridor/clearance evidence flips — that
    // caused the TURN_LEFT↔TURN_RIGHT oscillation (_failed/000006).  During
    // the cooldown the corrector keeps the LOCKED bypass side.
    uint32_t macro_search_rotation_cooldown_5hz = 12;
    int    macro_observable_unknown_margin_cells = 3;
    double macro_side_evidence_margin = 0.5;
    double macro_evidence_ray_step_deg = 1.0;
    int    macro_min_evidence_ray_pairs = 4;
    double macro_corridor_half_width = 1.5;
    // A blocker must LATERALLY SPAN >= this fraction of the corridor
    // half-width before the corridor is declared blocked (aligned with
    // il_2d_multiscale_debug macro/blocking_lateral_span_ratio).  A small /
    // single obstacle that only nicks the corridor edge is left to the
    // 30 Hz planner to weave around; the 5 Hz corrector must not take over.
    double macro_blocking_lateral_span_ratio = 0.5;
    double macro_corridor_rear_tolerance_m = 0.5;
    double macro_local_recovery_prefix_m = 0.8;
    double macro_local_candidate_bearing_step_deg = 5.0;
    double macro_local_candidate_distance_step_m = 0.5;
    // Maximum distance for a fixed NORMAL_CORRECTION world waypoint.  The
    // internal fly-through bit, rather than distance inflation, separates a
    // temporary waypoint from the original terminal goal.
    double macro_guide_horizon_m = 4.8;
    double macro_local_target_event_tolerance_m = 0.05;
    int    macro_takeover_confirm_ticks_30hz = 12;
    int    macro_unknown_recovery_threshold_ticks = 60;
    // Consecutive 5 Hz updates the vehicle must remain at/below the
    // stationary speed before a latched brake-before-search is released
    // and the world-latched TURN step is published.  2 updates = 0.4 s.
    int    macro_brake_confirm_ticks_5hz = 2;
    // R25: distance at which a fixed NORMAL_CORRECTION waypoint counts as
    // REACHED (start searching for the next target).  Must be >= the
    // 30 Hz stop distance (goal_tolerance_m = 0.4): the unified speed law
    // stops the vehicle 0.4 m short of ANY target, so a smaller reach
    // threshold would leave it parked short forever and trip the
    // no-progress refresh (waypoint_execution_fail_updates_ >= 3).
    double macro_waypoint_reached_tolerance_m = 0.5;

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

/// Canonical 3D vehicle state — the ONLY 3D state the pipeline carries.
/// World frame: X right, Y up (2D plane), Z world-up.  Yaw uses the expert
/// frame (see the file header); pitch is diagnostic only.
struct VehicleState3D {
    Vec3d position{0.0, 0.0, 2.0};        // world (m)
    Vec3d velocity_world{0.0, 0.0, 0.0};  // world-frame velocity (m/s)
    double yaw = 0.0;                     // world heading (rad, expert frame)
    double pitch = 0.0;                   // diagnostic pitch (rad)
    double yaw_rate = 0.0;                // body yaw rate (rad/s)
};

/// The horizontal projection of the vehicle state — the ONLY state the
/// planar (2D) expert layers (5 Hz corrector / adapter / 30 Hz planner)
/// ever see.  The vertical channel lives exclusively in VehicleState3D.
struct PlanarState {
    Vec2d position{0.0, 0.0};
    double yaw = 0.0;
    Vec2d velocity_world{0.0, 0.0};  // world-frame horizontal velocity
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

/// Canonical 3D navigation task — the ONLY task the logger / labels see.
/// The planar (2D) layers receive Task2D (projected) + goal_z + the flight
/// band separately.
struct NavigationTask3D {
    Vec3d start{0.0, 0.0, 2.0};
    Vec3d goal{0.0, 0.0, 2.0};
    double initial_yaw = 0.0;
    double z_min = 0.8;  // flight band lower bound (m)
    double z_max = 3.0;  // flight band upper bound (m)
    uint64_t task_id = 0;
    uint64_t scene_id = 0;
    uint64_t seed = 0;
    bool valid = false;
};

/// Explicit projection between the canonical 3D types and the planar (2D)
/// types used by the 2D expert layers.  The ONLY sanctioned place to
/// convert — keeps XY and Z frames from being mixed anywhere else.
struct HorizontalProjection {
    static PlanarState state(const VehicleState3D& s) {
        PlanarState p;
        p.position = Vec2d(s.position.x(), s.position.y());
        p.yaw = s.yaw;
        p.velocity_world =
            Vec2d(s.velocity_world.x(), s.velocity_world.y());
        p.yaw_rate = s.yaw_rate;
        return p;
    }
    static Vec2d position(const Vec3d& p) { return Vec2d(p.x(), p.y()); }
    static Vec3d lift(const Vec2d& xy, double z) {
        return Vec3d(xy.x(), xy.y(), z);
    }
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
        forEachOccupiedWithin(
            p, search_radius, std::numeric_limits<uint32_t>::max(),
            std::forward<Visitor>(visitor));
    }

    /// Same query with an explicit history-age limit.  This lets the local
    /// planner keep short-term memory without allowing an old, unrefreshed
    /// cell to permanently veto a visible route.
    template <typename Visitor>
    void forEachOccupiedWithin(const Vec2d& p, double search_radius,
                               uint32_t max_age_ticks,
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
                const size_t cell_id = idx(ix, iy);
                if (cells[cell_id] != CellState::OCCUPIED) continue;
                if (max_age_ticks != std::numeric_limits<uint32_t>::max() &&
                    cell_id < age_ticks.size() &&
                    age_ticks[cell_id] > max_age_ticks) {
                    continue;
                }
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
        return nearestOccupied(p, search_radius,
                               std::numeric_limits<uint32_t>::max());
    }

    NearestOccupiedResult nearestOccupied(const Vec2d& p,
                                          double search_radius,
                                          uint32_t max_age_ticks) const {
        NearestOccupiedResult out;
        double best = std::numeric_limits<double>::infinity();
        Vec2d best_centre(0.0, 0.0);
        forEachOccupiedWithin(
            p, search_radius, max_age_ticks,
            [&](const Vec2d& centre, double distance) {
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

    double minClearanceToOccupied(const Vec2d& p, double r,
                                  uint32_t max_age_ticks) const {
        return nearestOccupied(p, r, max_age_ticks).distance;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  Planar target — expert world point plus the body-frame training label.
//  The target altitude travels separately.
// ═══════════════════════════════════════════════════════════════════
struct PlanarTarget {
    // Expert-space target. Both expert planners may use/latch world points;
    // only the student/data label below is constrained to the body frame.
    Vec2d position_world{0.0, 0.0};
    bool world_valid = false;

    // Student/data-label contract: unit direction in the LIVE body FLU
    // plane (+X forward, +Y left), re-expressed on every 30 Hz tick.
    Vec2d direction_body{1.0, 0.0};
    // Student/data-label distance channel:
    //   ordinary target: min(real_distance, 4.5 m) / 5 m in [0, 0.9]
    //   pure rotation:   exactly 1.0
    // The exact ordinary ceiling is parameterized as
    // (obs_range_m - te_normal_distance_reserve_m) / obs_range_m.
    double normalized_distance = 0.0;
    // Internal expert semantic, never persisted as a student input. A fixed
    // macro waypoint remains fly-through inside the 4.5 m label ceiling.
    bool flythrough = false;
    // Directive update event (5 Hz ZOH boundary).  A change alone NEVER
    // resets planner memory.
    uint64_t update_event = 0;
    // MISSION revision.  Changed ONLY on a new task / scene reset and on a
    // formally accepted final navigation-goal revision.  5 Hz correction
    // enter / refresh / exit NEVER change it.
    uint64_t mission_revision = 0;

    bool valid() const {
        return world_valid && position_world.allFinite() &&
               direction_body.allFinite() &&
               direction_body.squaredNorm() > 1e-12 &&
               std::isfinite(normalized_distance) &&
               normalized_distance >= 0.0 && normalized_distance <= 1.0;
    }
};

/// FSM output consumed by the vertical controller / 3D composer: the
/// planar target plus its altitude (world z).  PASS / NORMAL carry the
/// mission altitude; TURN keeps the current altitude (pure rotation).
struct LocalTarget {
    PlanarTarget planar;
    double z = 2.0;
};

// ═══════════════════════════════════════════════════════════════════
//  Trajectories
// ═══════════════════════════════════════════════════════════════════
/// Planar (2D) trajectory — produced/consumed by the 30 Hz planar planner
/// and the preflight simulator.
struct PlanarTrajectory {
    std::vector<Vec2d> points;
    std::vector<double> yaw;
    std::vector<double> t;  // seconds from plan start
    bool valid = false;
    // The cruise level at which this trajectory was validated by the
    // multi-cruise retry (m/s).  The EXECUTED command must never exceed it
    // — a path only certified at 0.25 m/s (tight clearance) must not be
    // driven at the nominal 2 m/s.
    double cruise_mps = 0.0;
};

/// Canonical 3D trajectory (world points) for logging / diagnostics.
/// Assembled by CommandComposer3D from a PlanarTrajectory + the vertical
/// prediction.
struct Trajectory3D {
    std::vector<Vec3d> points;
    std::vector<double> yaw;
    std::vector<double> t;
    bool valid = false;
};

struct PlannerResult {
    bool success = false;
    bool turn_mode = false;
    FailureReason failure_reason = FailureReason::NONE;
    // intent_* = the LONG-TERM rollout intent.  vx_body/vy_body/yaw_rate
    // are the EXECUTABLE OUTPUT (PLANAR, body FLU horizontal).  The full
    // 3D BODY/FLU command is composed by CommandComposer3D (horizontal
    // here + vertical from VerticalController) — the planner NEVER emits
    // a vertical component.
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
    PlanarTrajectory selected;
    std::vector<PlanarTrajectory> rejected_candidates;
    // Diagnostic: the planned trajectory's semantics (for the viewer).
    // plan_terminal == true  → the selected spline ends in a full stop at
    //                          the target (TERMINAL_SETTLING);
    // plan_terminal == false → boundary waypoint, end speed = cruise.
    bool plan_terminal = false;
    double plan_end_speed_mps = 0.0;
    double plan_executed_speed_mps = 0.0;
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
    // R29i: the R29h speed law hard-stopped because the NOSE points at/into
    // an observed blocker (nose_clear <= handoffClearance).  A clear ray may
    // still exist, so this is NOT a corridor block — but it IS "too close to
    // an obstacle to proceed", and it must accumulate failure evidence so
    // the 5 Hz macro takes over and rescues (replan to the original goal /
    // pure rotation around the blocker) instead of parking forever.
    bool nose_blocked_stop = false;
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
    double selected_cost_lateral_drift = 0.0;
    double selected_cost_obstacle_risk = 0.0;
    bool local_corridor_blocked = false;
    // R26: the 1 m risk corridor is a SOFT proximity signal (cost / speed
    // only), NEVER a hard topology block.  A cell 0.5-1 m from the direct
    // path sets this flag but must not trigger macro takeover.
    bool risk_corridor_near_obstacle = false;
    // R25 structured corridor-block diagnostics: WHY the straight corridor
    // to the target was judged blocked, the first blocking OCCUPIED cell
    // (world XY) and its history age in ticks (0 = observed this tick =
    // CURRENT frame; > 0 = stale HISTORY cell).  CLEAR / no info otherwise.
    CorridorBlockReason corridor_block_reason = CorridorBlockReason::CLEAR;
    double first_block_x = std::numeric_limits<double>::quiet_NaN();
    double first_block_y = std::numeric_limits<double>::quiet_NaN();
    uint32_t first_block_age_ticks = 0;
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
    // PERSISTENT terminal/brake-only semantic (R24).  When true, the
    // corrected_target_world is a TERMINAL brake point: the 30 Hz planner
    // must STOP at it and never fly through it.  This is decided ONCE at
    // 5 Hz and held on the directive — it must NEVER be re-derived from
    // the live distance every 30 Hz tick (the vehicle coasts a few cm past
    // the point and would otherwise flip from terminal-brake back to
    // fly-through).  Set only by the brake-before-search path; all other
    // directives keep it false.
    bool terminal_stop = false;
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
    // Along-goal-axis distance (m) from the drone to the NEAREST surface
    // of a corridor-blocking obstacle (from extractBlocker; +inf when no
    // blocker is observed).  Positive = blocker still AHEAD along the
    // goal line; <= 0 = at/behind the drone.  Used by the exit / re-entry
    // gates so a correction is only released once the current blocker is
    // actually bypassed (not merely because a bypass ray is visible).
    double blocker_min_along = std::numeric_limits<double>::infinity();
    std::string reason = "NONE";
};

/// Local-planner feedback used by the upper planner.  The cold preview uses
/// the same local planner and local information; actual 30 Hz execution
/// history confirms takeover so one preview miss cannot replace a working
/// plan.  No scene truth or global route is involved.
struct LocalPlanningAssessment {
    bool target_outside_fov = false;
    bool rotation_available = false;
    bool translation_plan_valid = false;
    bool terminal_plan_valid = false;
    bool plan_valid = false;
    bool progress_qualified = false;
    bool local_corridor_blocked = false;
    // R29i: the local speed law hard-stopped because the nose faces an
    // observed blocker (see PlannerResult::nose_blocked_stop).  Carried
    // through the 5 Hz assessment so the macro treats it as takeover
    // evidence ("too close to an obstacle") and issues the rescue directive.
    bool nose_blocked_stop = false;
    bool live_original_plan_usable = false;
    bool takeover_confirmed = false;
    PlannerStatus planner_status = PlannerStatus::NO_SAFE_CANDIDATE;
    FailureReason failure_reason = FailureReason::NO_SAFE_CANDIDATE;
};

/// The adapter's per-30Hz-tick output shared by the C++ expert (full world
/// target) and the 30 Hz student/data contract (live body direction +
/// clipped normalized distance).
struct EncodedTargetInput {
    bool valid = false;
    Vec2d direction_body{1.0, 0.0};
    double normalized_distance = 0.0;
    Vec2d effective_target_world{0.0, 0.0};
    bool effective_target_world_valid = false;
    TargetCorrectionType source_type = TargetCorrectionType::PASS_THROUGH;
    // ── 3D extension: effective target altitude (world z, m).  TURN
    //    targets are pure rotation ⇒ z = state.z; PASS / NORMAL carry the
    //    mission altitude. ────────────────────────────────────────────
    double z = 2.0;
};

}  // namespace expert
}  // namespace il_dataset
