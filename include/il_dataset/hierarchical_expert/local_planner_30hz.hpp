#pragma once
/// @file   local_planner_30hz.hpp
/// @brief  30 Hz local obstacle-avoidance expert.
///
/// INFORMATION BOUNDARY (enforced by the interface):
///   plan(PlanarState, LocalObservation, PlanarTarget)
///   — the planner CANNOT see the global ESDF, the obstacle truth, the
///     global path or the scene.  It sees only the current PLANAR state,
///     the local observation (with short-term history) and the current
///     planar target (expert world point plus the live body-frame training
///     label).  It NEVER receives an original/corrected-target semantic
///     flag.  It NEVER sees the vertical channel
///     (VehicleState3D / vz):
///     the 3D command is composed by CommandComposer3D.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/kinematics.hpp"
#include "il_dataset/hierarchical_expert/ego_bspline.hpp"

#include <algorithm>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

/// Light result of a NON-mutating preview plan.
struct PreviewResult {
    bool success = false;
    bool turn_mode = false;
    bool emergency_brake = false;
    bool plan_valid = false;
    bool plan_terminal = false;
    bool progress_qualified = false;
    bool local_corridor_blocked = false;
    bool avoidance_active = false;
    // R29i: nose-blocked hard stop (see PlannerResult::nose_blocked_stop).
    bool nose_blocked_stop = false;
    double selected_output_speed_mps = 0.0;
    double min_observed_clearance_m =
        std::numeric_limits<double>::infinity();
    FailureReason failure_reason = FailureReason::NO_SAFE_CANDIDATE;
    PlannerStatus planner_status = PlannerStatus::NO_SAFE_CANDIDATE;
};

class LocalPlanner30Hz {
public:
    explicit LocalPlanner30Hz(const Params2D& p) : p_(p) {}

    /// Full planning step.  Mutates internal hysteresis + current
    /// trajectory bookkeeping.  PLANAR only — the vertical channel is
    /// composed externally.
    PlannerResult plan(const PlanarState& state, const LocalObservation& obs,
                       const PlanarTarget& target);

    /// Non-mutating, memory-independent feasibility preview.  The upper
    /// planner runs this at its lower cadence to answer whether the local
    /// planner can handle the original target by itself (including the
    /// local yaw-first rotation responsibility).
    PreviewResult previewPlan(const PlanarState& state,
                              const LocalObservation& obs,
                              const PlanarTarget& target) const;

    /// True iff the vehicle can come to a full stop (under max accel)
    /// within the observed free space ahead.
    bool canBrakeSafely(const PlanarState& state,
                        const LocalObservation& obs) const;

    bool turnHysteresisActive() const { return turn_hysteresis_active_; }
    void resetTurnHysteresis() { turn_hysteresis_active_ = false; }
    /// Reset all per-task planner memory, including command-change history.
    void reset();

private:
    /// Geometry resolved privately from the expert target at the live pose.
    /// This type never crosses the local-planner boundary.
    struct ResolvedPlanarTarget {
        Vec2d position{0.0, 0.0};
        bool valid = false;
        uint64_t update_event = 0;
        uint64_t mission_revision = 0;
        double normalized_distance = 0.0;
        // The original goal becomes terminal below the clip ceiling.
        // Temporary macro waypoints carry explicit internal fly-through
        // semantics even when their remaining distance falls below it.
        bool terminal = false;
        // R29j: macro NORMAL_CORRECTION detour waypoint flag (from
        // PlanarTarget::flythrough).  The R29h speed law is relaxed for
        // these: off-nose flight is expected (sideways detour) and a
        // nose-facing blocker is not a hard stop.
        bool flythrough = false;
    };

    ResolvedPlanarTarget resolveTarget(const PlanarState& state,
                                       const PlanarTarget& target) const;

    /// Pure-rotation safety: rotating
    /// at speed sweeps the drone sideways into unobserved space (scene
    /// walls / obstacles) — repeated collisions happened at 1.4-1.8 m/s
    /// while TURNING (the reachableCommand clamp retains the forward
    /// velocity, it does not brake to zero).  Brake first, rotate in place.
    /// Yaw rate for a turn, defensively scaled by any residual translation.
    /// Normal turn branches enter only after the explicit brake stage.
    /// (The old hard gate — speed > kTurnMaxSpeedMps → yaw_rate 0 — froze
    /// the nose during the ~1 s brake and looked "clumsy / nose fixed in
    /// place" at macro→PASS_THROUGH and planning transitions, joint_v2 ep
    /// 000000_b99de7ca: yaw_rate_command=0 for a full second while the
    /// target sat 55° off-nose.)  The turn-sweep collisions were at
    /// SUSTAINED 1.4-1.8 m/s with full yaw rate; the speed scaling here
    /// yields a gentle yaw rate at speed (large radius, little sweep,
    /// and the drone is decelerating at lp_max_accel the whole time) that
    /// grows to the full rate only near standstill, so the swept arc stays
    /// tight.  scale = 1/(1 + 3·spd): spd 0→1.0, 0.2→0.63, 0.5→0.4,
    /// 1.0→0.25, 1.8→0.16.
    double turnYawRate(const PlanarState& state, double bearing) const {
        const double spd = state.velocity_world.norm();
        const double scale = 1.0 / (1.0 + 3.0 * spd);
        return clamp(p_.lp_turn_k * bearing * scale, -p_.lp_max_yaw_rate,
                     p_.lp_max_yaw_rate);
    }

    /// Unified collision distance used by this planner (USER DIRECTIVE
    /// 2026-08-20): obstacle minimum collision = 4 cells = 0.4 m from an
    /// OCCUPIED cell centre (= drone radius 0.3 + cell 0.1).  No ESDF
    /// geometric envelope — lp_min_clearance is the common static base for
    /// every planner (EGO, straight ray, A*).
    /// UNIFIED minimum clearance (USER DIRECTIVE 2026-08-24): every
    /// planner/judge clearance is a single 0.5 m value measured from an
    /// occupied-cell centre (drone radius 0.3 + cell 0.1 + 0.1 safety) —
    /// the SAME threshold for ray blocking, trajectory validation, the
    /// speed brake, the hand-off-to-upper-planner gate and the EGO config.
    /// No per-planner clearance tuning.
    static constexpr double kMinClearanceM = 0.5;
    double handoffClearance() const { return kMinClearanceM; }

    /// Required centre-to-cell clearance: static handoff robustness plus
    /// one observation reaction interval and the physical stopping distance.
    double stoppingEnvelope(double speed) const {
        const double s = std::max(0.0, speed);
        const double a = std::max(1e-6, p_.lp_eff_accel_mps2);
        return handoffClearance() +
               s * std::max(0.0, p_.lp_obstacle_reaction_time_s) +
               s * s / (2.0 * a);
    }

    double requiredClearance(double closing_speed) const {
        return stoppingEnvelope(closing_speed);
    }

    /// Search radius for the nearest-OCCUPIED query at a candidate sample.
    double clearanceSearchRadius(double total_speed) const {
        return std::max(p_.lp_soft_clearance_radius_m,
                        stoppingEnvelope(total_speed));
    }

    /// Plan a dynamics-feasible reference path from the current state to an
    /// endpoint (the target when terminal, else a FOV-boundary point).  The
    /// path stays inside the FOV and clears observed OCCUPIED cells (the
    /// local ESDF); UNKNOWN is PASSABLE.  Terminal → decelerate to a full
    /// stop at the endpoint; boundary → cruise up to v_end.  Returns false
    /// when no safe path reaches the endpoint within the horizon.
    /// min_clear is the smallest centre-to-surface distance along the path.
    /// `min_validate_m` (R28) forces the non-terminal receding-horizon
    /// validation to cover at least this arc (the first observed corridor
    /// block): a straight chord through a blocked corridor can no longer
    /// "validate" because the blocker sits beyond the ~3 m front window.
    bool planFovTrajectory(const PlanarState& state,
                           const LocalObservation& obs,
                           const Vec2d& endpoint, double v_end,
                           bool terminal, PlanarTrajectory& out,
                           double& min_clear,
                           double min_validate_m = 0.0) const;

    /// EGO-style optimisation B-spline first, straight-ray fallback.
    /// Tries the EGO optimiser (bends the spline around observed
    /// obstacles) when p_.ego_enabled; if it yields no validated path,
    /// falls back to the legacy straight planFovTrajectory so existing
    /// success cases never regress.
    bool planEgoOrStraight(const PlanarState& state,
                           const LocalObservation& obs,
                           const Vec2d& endpoint, double v_end,
                           bool terminal, PlanarTrajectory& out,
                           double& min_clear,
                           double min_validate_m = 0.0) const;
    /// A*-route initialised EGO B-spline (USER ARCHITECTURE A+B): the EGO
    /// optimiser is given the REAL A* route to the endpoint so the curve
    /// follows the obstacle-free corridor and ends AT the endpoint.  Falls
    /// back to the straight validated planFovTrajectory (architecture C)
    /// when the EGO plan fails.
    bool planEgoOrStraightWithPath(const PlanarState& state,
                                   const LocalObservation& obs,
                                   const Vec2d& endpoint, double v_end,
                                   bool terminal,
                                   const std::vector<Vec2d>& astar_path,
                                   PlanarTrajectory& out,
                                   double& min_clear,
                                   double min_validate_m = 0.0) const;
    /// Build the EGO Config from the current Params2D (ego_* section).
    static EgoBsplineOptimizer::Config egoConfig(const Params2D& p);

    /// Lightweight 8-connected A* over the observed grid (EGO-style routing,
    /// R20d).  Routes from `start` toward `goal` through non-OCCUPIED cells
    /// (UNKNOWN passable with a small penalty — it may be free beyond the
    /// current observation).  Returns the world-space polyline of cell
    /// centres ending at the goal cell when reachable, else at the best
    /// (lowest f) cell reached.  Empty on failure (start/goal occupied or
    /// out of grid).  Bounded by `max_expansions` for 30 Hz real-time.
    /// When `fov_front_m > 0`, cells within that distance of `start` that
    /// lie OUTSIDE the FOV cone centred on `yaw` get a per-cell cost
    /// penalty, so the route leaves through the VISIBLE corridor — the
    /// B-spline's front-3 m FOV validation (45°) then passes.  Without it
    /// A* routes "over the top" of a blocker (shorter but heading behind
    /// the drone, outside the FOV) and the B-spline is rejected -> scan
    /// fallback -> the drone creeps to the blocker edge and deadlocks
    /// (measured: joint_v2_000000_316ed0e2 TURN_RIGHT spin for 6.5 s).
    std::vector<Vec2d> routeAStar(const LocalObservation& obs,
                                  const Vec2d& start, const Vec2d& goal,
                                  int max_expansions,
                                  double fov_front_m = 0.0,
                                  double yaw = 0.0) const;

    /// Executed-front distance used to constrain the macro-guide A* route
    /// to the visible corridor.  Matches the EGO B-spline front-3 m FOV
    /// validation (buildAndValidate front_dist = max(2, min(3, cruise·1.5)))
    /// so the optimised curve's front can always be validated.
    double macroGuideFrontFovM() const {
        const double cruise =
            std::min(p_.lp_cruise_speed_mps, p_.lp_max_speed);
        return std::max(2.0, std::min(3.0, cruise * 1.5));
    }
    /// R28j: the receding local route horizon — the arc-length at which the
    /// B-spline endpoint / routing waypoint sits for NON-terminal plans.
    /// The local B-spline plans only to this ~3 m waypoint (never the full
    /// far target); the executed yaw control stays on the short pure-pursuit
    /// lookahead (lp_pursuit_lookahead_m, 0.6 m).  Routing heading = the
    /// direction to this waypoint; control heading = the pursuit tangent —
    /// the two layers are deliberately separate.
    double localRouteHorizon() const {
        const double cruise =
            std::min(p_.lp_cruise_speed_mps, p_.lp_max_speed);
        return std::max(2.0, std::min(3.0, cruise * 1.5));
    }

    PlannerResult computePlan(const PlanarState& state,
                              const LocalObservation& obs,
                              const ResolvedPlanarTarget& target, bool mutate);
    /// R27: true when the drone can (and should) PURSUE the committed
    /// current_trajectory_ this tick instead of re-optimising (tracking
    /// fast-path gate).  False when: no stored plan, target changed,
    /// terminal micro-approach / goal-capture zone, target out of FOV,
    /// stored path cut by a newly observed obstacle, large cross-track
    /// deviation, or the plan front is nearly consumed.
    bool trackingAvailable(const PlanarState& state,
                           const LocalObservation& obs,
                           const ResolvedPlanarTarget& target) const;
    /// R27: pure-pursuit along the stored trajectory (no re-optimisation).
    /// Mirrors the planned-branch command semantics so the FSM / CSV see
    /// the same fields (status / selected / plan_terminal / progress bits).
    PlannerResult trackStoredTrajectory(const PlanarState& state,
                                        const LocalObservation& obs,
                                        const ResolvedPlanarTarget& target);
    /// R27: EGO temporal-anchoring reference (the previous plan's world
    /// points), or nullptr when no compatible stored plan exists.  The
    /// reference must start near the current state and end near the new
    /// endpoint, else it is ignored.
    const std::vector<Vec2d>* anchoredReference(const PlanarState& state,
                                                const Vec2d& endpoint) const;
    /// R27: near-obstacle proximity signal (mirrors computePlan's lambda).
    bool nearObservedObstacle(const LocalObservation& obs, const Vec2d& pos,
                              double proximity_m) const;
    /// R25: after a full mutate plan, push this tick's outcome into the
    /// limit-cycle window and set res.local_limit_cycle_detected.  Called
    /// from plan() (every branch reaches it; preview probes never mutate).
    void updateLimitCycleWindow(const PlanarState& state,
                                const ResolvedPlanarTarget& target,
                                PlannerResult& res);
    bool currentTrajectoryBlocked(const PlanarState& state,
                                  const LocalObservation& obs,
                                  bool& dynamic_violation) const;
    /// R28b: true if any point of `plan` (from the closest point to the
    /// state onward) comes within the hard/dynamic clearance of an observed
    /// OCCUPIED cell.  Reconciles the corridor assessment (which looks
    /// toward the GOAL) with the ACTUAL planned path: a clear line parallel
    /// to a blocked goal corridor, or a short hop whose tail ends before
    /// the blocker, must NOT be rejected (task 440: BLOCKED + false macro
    /// takeover while flying a 0.66 m-clear line past obs5).
    bool planPathBlocked(const PlanarState& state,
                         const LocalObservation& obs,
                         const PlanarTrajectory& plan) const;
    VelocityCommand3D reachableCommand(const PlanarState& state,
                                       const VelocityCommand3D& intent) const;
    Vec2d bodyVelocity(const PlanarState& state) const;
    bool updateMissionState(const ResolvedPlanarTarget& target,
                            PlannerResult& res);
    VelocityCommand3D terminalIntent(const PlanarState& state,
                                     const ResolvedPlanarTarget& target) const;
    void assessLocalCorridor(const PlanarState& state,
                             const LocalObservation& obs,
                             const ResolvedPlanarTarget& target, bool& blocked,
                             double& first_blocking_distance_m,
                             CorridorBlockReason& reason, Vec2d& first_block,
                             uint32_t& first_block_age_ticks,
                             bool& risk_near_obstacle) const;
    /// R26: true iff the straight segment from the state to `goal` keeps
    /// at least handoffClearance() from every observed OCCUPIED cell.
    /// Used by the terminal micro-approach gate (0.4-0.8 m dead zone).
    bool microApproachSafe(const PlanarState& state,
                           const LocalObservation& obs,
                           const Vec2d& goal) const;
    /// Check a straight world-frame ray through the full observable range.
    /// The executed spline may end at the shorter receding horizon, but a
    /// far obstacle must still be noticed before the vehicle advances into
    /// a locally avoidable dead end.
    bool forwardCorridorSafe(const PlanarState& state,
                             const LocalObservation& obs,
                             const Vec2d& world_direction,
                             double distance_m) const;
    /// ── RAY-SECTOR SELECTION (user redesign) ─────────────────────
    /// Cast `kRays` rays every 5° across the camera FOV from the current
    /// pose, each to (obs_range_m - 0.5) m.  A ray is BLOCKED when any
    /// sampled point has an observed OCCUPIED cell centre within
    /// handoffClearance() (the same clearance the trajectory validators
    /// use, so a chosen ray always passes the executed spline's hard gate).
    /// Selection (per user design):
    ///   1. start at the ray whose bearing is CLOSEST to the target
    ///      bearing; if clear, choose it;
    ///   2. if blocked, expand outward in PAIRS (one step each side);
    ///      one clear side -> choose that ray; BOTH clear -> choose the
    ///      RIGHT one (smaller bearing), unless a committed side is active
    ///      (last_plan_side_ hysteresis) in which case that side wins;
    ///   3. both blocked -> keep expanding; when one side is exhausted
    ///      only the other side advances;
    ///   4. every ray blocked -> return NaN (the upper planner must take
    ///      over: macro correction or pure rotation).
    /// `clear_range` returns the free distance along the chosen ray (to the
    /// first blocking cell); `nose_clear` returns the free distance along
    /// the CURRENT NOSE direction (yaw) — the executed body still flies
    /// along the nose while the yaw slews to the chosen ray, so the speed
    /// must brake against the nose clearance, not only the ray clearance.
    /// Returns the chosen ray bearing RELATIVE to yaw, or NaN when all rays
    /// are blocked.
    double raySectorSelect(const PlanarState& state,
                           const LocalObservation& obs, double b_t,
                           double& clear_range, double& nose_clear) const;
    /// Speed along the chosen ray: braking-feasible against the observed
    /// free range (v = sqrt(2·a·(clear_range − handoffClearance))), capped
    /// by cruise.  Zero when the free range barely exceeds the clearance.
    double raySectorSpeed(double clear_range) const;
    double stoppingDistance(const PlanarState& state) const;
    bool spaceToStop(const PlanarState& state, const LocalObservation& obs,
                     double dist) const;

    Params2D p_;
    EgoBsplineOptimizer ego_bspline_;
    bool turn_hysteresis_active_ = false;
    // ── Turn spin guard ────────────────────────────────────────────
    // Pure-rotation turn hysteresis has no natural timeout: if the
    // corrected target is unreachable the bearing stays large and the
    // drone spins in place at max yaw rate forever (observed after the
    // corrector latched an unreachable bypass target).  Cap the turn
    // duration and force-release into the normal (brake/hold) path with a
    // short cooldown so the 5 Hz corrector can re-plan.
    uint64_t turn_ticks_ = 0;       // mutating-plan ticks while turning
    uint64_t turn_hold_until_ = 0;  // plan-tick cooldown before re-entry
    uint64_t plan_ticks_ = 0;       // monotonic mutating-plan tick counter
    PlanarTrajectory current_trajectory_;
    VelocityCommand3D last_command_;
    bool has_last_command_ = false;
    uint64_t last_mission_revision_ = 0;
    Vec2d last_target_position_{0.0, 0.0};
    bool last_target_valid_ = false;
    // ── R27 plan/track state ──────────────────────────────────────
    // Metadata of the last committed planned trajectory (current_trajectory_
    // holds its geometry).  The tracking fast-path uses these to decide
    // whether the stored plan still applies to the current target.
    uint64_t stored_mission_revision_ = 0;
    Vec2d stored_target_pos_{0.0, 0.0};
    bool stored_terminal_ = false;
    Vec2d stored_endpoint_{0.0, 0.0};
    double stored_v_end_ = 0.0;
    double stored_min_clear_ = std::numeric_limits<double>::infinity();
    // ── R28i side-commitment ──────────────────────────────────────
    // Side (sign of chosen_b - b_t) of the last committed non-terminal plan:
    // +1 = plan deviates to the +b (counter-clockwise / left) side of the
    // target direction, -1 = the -b (clockwise / right) side, 0 = none yet.
    // The bearing scan uses it to keep expanding on the SAME side first so
    // the drone commits to one side of a blocker instead of flapping between
    // both sides as the observed-grid validation toggles (task 401: the plan
    // flipped right (toward goal) <-> left (away) around obs7 every ~0.3 s).
    double last_plan_side_ = 0.0;
    // A side change is forbidden for a short dwell after choosing a bypass.
    // This is deliberately a minimum dwell, not a permanent latch: a side
    // that becomes genuinely infeasible may still be abandoned afterwards.
    uint64_t side_commit_until_tick_ = 0;
    uint32_t direct_clear_ticks_ = 0;
    // ── R29c yaw-intent EMA (ray-sector smoothing) ──────────────────
    // Last smoothed yaw intent, so neighbouring-ray switches (avg_b steps)
    // ease the nose instead of jerking it at 30 Hz while the velocity stays
    // on the clear ray.  Cleared per task; updated only on the mutate path.
    double last_yaw_intent_ = 0.0;
    bool has_last_yaw_intent_ = false;
    // ── Ray-sector FOV shrink (user directive) ────────────────────
    // When the drone is STUCK repeatedly selecting the OUTERMOST ray with
    // no progress, shrink the ray FOV half-angle (e.g. 45° -> 40°) so the
    // sector excludes the edge and the "no solution" hand-off to the upper
    // planner fires sooner, instead of parking on the edge ray.  0 means
    // "use the default obs_fov/2".  Eased back open once progress resumes.
    double ray_fov_half_ = 0.0;  // rad; 0 = default (obs_fov_deg / 2)
    uint32_t edge_stuck_ticks_ = 0;
    // ── R25 limit-cycle / stagnation detector (mutate path only) ──
    // Sliding window over the distance/bearing to the EFFECTIVE target and
    // the blocked-frame ratio.  When the target is not being approached and
    // >70% of the window is BLOCKED / SAFE_HOLD / NO_SAFE_CANDIDATE, the
    // planner reports local_limit_cycle_detected so the 5 Hz corrector can
    // force a takeover (issue a new waypoint / change search strategy).
    static constexpr size_t kLimitCycleWindowTicks = 90;   // 3 s at 30 Hz
    static constexpr double kLimitCycleMinProgressM = 0.20;   // over window
    static constexpr double kLimitCycleMinTurnProgressDeg = 6.0;
    static constexpr double kLimitCycleBlockedRatio = 0.70;
    std::deque<double> limit_cycle_dist_window_;
    std::deque<double> limit_cycle_bearing_window_;
    std::deque<uint8_t> limit_cycle_blocked_window_;
};

}  // namespace expert
}  // namespace il_dataset
