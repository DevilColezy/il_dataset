#pragma once
/// @file   macro_expert_5hz.hpp
/// @brief  5 Hz VisibilityTargetCorrector — local observability judge
///         + target corrector.
///
/// The 5 Hz expert answers exactly one question on every real 5 Hz
/// boundary (tick % 6 == 0):
///
///   "Do the current FOV and its causal local history contain enough
///    information for the 30 Hz expert to finish its OWN local avoidance?"
///
///   * YES → PASS_THROUGH the original target (the local planner owns
///     out-of-FOV target rotation as well as visible reactive avoidance);
///   * NO  → temporarily correct the tracked target:
///       - NORMAL_CORRECTION: a quantized in-FOV frontier on the locked
///         side;
///       - TURN_LEFT / TURN_RIGHT: one bounded world-latched direction
///         step when the current local information has no usable frontier.
///
/// INFORMATION BOUNDARY (enforced by the interface).  At runtime the
/// corrector may ONLY read:
///   * PlanarState (pose, velocity, yaw, yaw rate) — the horizontal
///     projection of the canonical VehicleState3D;
///   * the ORIGINAL final goal;
///   * the CURRENT INSTANTANEOUS FOV patch (current_patch);
///   * the causal, decaying local history map built only from past patches;
///   * the current tick;
///   * its own memory.
/// It additionally receives compact feasibility assessments produced by
/// memory-independent local-planner previews: one for the original goal and
/// one for each proposed correction.  It never receives local commands,
/// candidate trajectories, scene truth, global ESDF or global paths.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/effective_target_adapter.hpp"

#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

class VisibilityTargetCorrector {
public:
    using DirectiveAssessmentFn = std::function<LocalPlanningAssessment(
        const TargetCorrectionDirective&)>;

    explicit VisibilityTargetCorrector(const Params2D& p)
        : p_(p), adapter_(p) {}

    /// Run on every real 5 Hz boundary (called exactly at tick % 6 == 0).
    TargetCorrectionDirective update(const PlanarState& state,
                                     const Vec2d& original_goal,
                                     const LocalObservation& current_patch,
                                     const LocalObservation& local_history,
                                     const LocalPlanningAssessment& assessment,
                                     bool live_directive_usable,
                                     const DirectiveAssessmentFn& assess_directive);

    /// Reset per-episode correction state (new task / scene).
    void reset();

    /// Clear only goal-dependent correction memory while preserving all
    /// monotonic diagnostic/update counters.
    void resetForNewGoal();

    /// Bump and return the directive update event.
    uint64_t bumpDirectiveEvent();

    // ── Diagnostics ────────────────────────────────────────────────
    const AvoidanceObservability& lastObservability() const {
        return last_obs_;
    }
    bool correctionActive() const { return correction_active_; }
    SideSelection lockedSide() const { return locked_side_; }
    uint64_t correctionEnterEvent() const { return correction_enter_event_; }
    uint64_t correctionExitEvent() const { return correction_exit_event_; }
    uint64_t correctionUpdateEvent() const { return correction_update_event_; }
    uint64_t directiveUpdateEvent() const { return update_event_; }
    const TargetCorrectionDirective& lastDirective() const {
        return last_directive_;
    }
    /// Recovery/re-entry guard (reported in 30 Hz ticks; 1 means the macro
    /// layer still owns the target).  Losing sight of a blocker during search
    /// is not evidence that it has been bypassed.
    int reentryGuardRemaining(uint64_t /*tick*/) const {
        return correction_active_ ? 1 : 0;
    }

private:
    struct LocalFreeGrid {
        double resolution = 0.1;
        int width = 0, height = 0;
        Vec2d origin{0.0, 0.0};
        std::vector<uint8_t> free;
        std::vector<uint8_t> blocked;
        std::vector<uint8_t> reachable;
        bool valid() const { return width > 0 && height > 0; }
        bool inGrid(int ix, int iy) const {
            return ix >= 0 && iy >= 0 && ix < width && iy < height;
        }
        size_t idx(int ix, int iy) const {
            return static_cast<size_t>(iy) * width + ix;
        }
        bool freeAt(const Vec2d& p) const {
            const GridIndex2D g = worldToGrid(p, origin, resolution);
            if (!inGrid(g.ix, g.iy)) return false;
            return free[idx(g.ix, g.iy)] != 0;
        }
        bool traversableAt(const Vec2d& p) const {
            const GridIndex2D g = worldToGrid(p, origin, resolution);
            if (!inGrid(g.ix, g.iy)) return false;
            return free[idx(g.ix, g.iy)] != 0 &&
                   blocked[idx(g.ix, g.iy)] == 0;
        }
        bool reachableAt(const Vec2d& p) const {
            const GridIndex2D g = worldToGrid(p, origin, resolution);
            if (!inGrid(g.ix, g.iy)) return false;
            return reachable[idx(g.ix, g.iy)] != 0;
        }
    };
    LocalFreeGrid buildLocalFreeGrid(const PlanarState& state,
                                     const LocalObservation& patch) const;

    struct SideCandidate {
        Vec2d endpoint{0.0, 0.0};
        double bearing = 0.0;
        double dist = 0.0;
        double along_progress = 0.0;
        bool certified = false;
    };
    std::vector<SideCandidate> sampleSideCandidates(
        const PlanarState& state, const Vec2d& goal,
        const LocalObservation& patch, const LocalFreeGrid& grid,
        bool has_blocker, double blocker_min_along, SideSelection side,
        bool strict) const;

    double freeRangeAlongFrom(const LocalObservation& obs, const Vec2d& from,
                              double bearing_world) const;
    double freeRangeAlong(const PlanarState& state,
                          const LocalObservation& obs,
                          double bearing_body) const;
    // R29k: distance along a world bearing to the first OBSERVED OCCUPIED
    // cell (UNKNOWN is passable).  Used to gate SEARCH_ROTATION_TOWARD_ORIG
    // INAL_GOAL on the goal bearing not being blocked.
    double occupiedRangeAlongFrom(const LocalObservation& obs,
                                  const Vec2d& from, double bearing_world,
                                  double max_range) const;

    bool extractBlocker(const PlanarState& state, const Vec2d& goal,
                        const LocalObservation& patch,
                        double& blocker_min_along,
                        double& blocker_lat_min,
                        double& blocker_lat_max) const;

    SideSelection selectSide(const PlanarState& state,
                             const LocalObservation& patch,
                             const Vec2d& goal) const;

    TargetCorrectionDirective makeCorrectionDirective(
        const PlanarState& state, const Vec2d& goal,
        const LocalObservation& patch, const LocalFreeGrid& grid,
        SideSelection side, bool live_directive_usable,
        bool drop_held_waypoint_allowed,
        const DirectiveAssessmentFn& assess_directive) const;

    /// Distance at which a fixed NORMAL_CORRECTION waypoint counts as
    /// REACHED (max of task tolerance and the dedicated waypoint
    /// tolerance).  Deliberately separate from the recovery-prefix lookahead.
    double waypointReachedTolerance() const {
        return std::max(p_.task_goal_tolerance,
                        p_.macro_waypoint_reached_tolerance_m);
    }

    bool directiveChanged(const TargetCorrectionDirective& a,
                          const TargetCorrectionDirective& b) const;

    AvoidanceObservability assessObservability(
        const PlanarState& state, const Vec2d& goal,
        const LocalObservation& patch) const;

    Params2D p_;
    EffectiveTargetAdapter adapter_;
    // ── 5 Hz internal memory ──
    bool correction_active_ = false;
    SideSelection locked_side_ = SideSelection::NONE;
    uint64_t update_event_ = 0;
    uint64_t correction_enter_event_ = 0;
    uint64_t correction_exit_event_ = 0;
    uint64_t correction_update_event_ = 0;
    // Progress watchdog for a correction episode.  The upper planner runs at
    // 5 Hz, so a bounded counter is enough to break a repeated brake/hold
    // loop without introducing global or scene state.
    uint32_t stagnant_update_count_ = 0;
    Vec2d last_state_position_{0.0, 0.0};
    bool has_last_state_position_ = false;
    uint32_t reentry_success_updates_ = 0;
    // A search episode includes both its braking handoff and the following
    // TURN steps.  Keeping this phase explicit prevents the 5 Hz brake
    // directive from resetting the accumulated yaw sweep every cycle.
    bool search_episode_active_ = false;
    double search_swept_rad_ = 0.0;
    double last_search_yaw_ = 0.0;
    bool has_last_search_yaw_ = false;
    // R29m: monotonic update counter + last SEARCH_ROTATION_TOWARD_ORIGINAL
    // _GOAL update, used to cooldown that re-acquisition so depth evidence
    // flips cannot oscillate the turn direction every boundary.
    uint64_t update_count_ = 0;
    uint64_t last_search_rotation_update_ = 0;
    // ── R24 brake-before-search latch ──────────────────────────────
    // A zero-distance brake is a TERMINAL semantic decided once at 5 Hz.
    // The world point is latched at first issue and held FIXED (never
    // rewritten to the live pose every cycle — that turned a brake into a
    // moving target) until the vehicle has been stationary for the
    // confirmation window; only then may the TURN search step be released.
    bool brake_latched_ = false;
    Vec2d brake_world_point_{0.0, 0.0};
    uint32_t brake_stationary_updates_ = 0;
    // ── R25 held-waypoint execution-failure counter ───────────────
    // Consecutive 5 Hz updates in which the currently held NORMAL_CORRECTION
    // waypoint is NOT actually executing (live_directive_usable=false) AND
    // its cold preview also fails.  Only after >= 3 consecutive failures is
    // the waypoint allowed to be dropped (a single cold preview miss must
    // not discard an actively-executing safe waypoint).
    uint32_t waypoint_execution_fail_updates_ = 0;
    // ── R26 macro-level limit-cycle watchdog (original-goal based) ─
    // The local detector watches the CURRENT effective target, but the
    // macro switches goal<->TURN so each switch resets the local bearing
    // evidence (measured: task 65 PASS/TURN_RIGHT loop at 0.48 m, ~36 s).
    // Track the ORIGINAL goal: no progress for the window forces a fresh
    // handoff to local.
    double lc_goal_dist_start_ = std::numeric_limits<double>::infinity();
    uint64_t lc_no_progress_ticks_ = 0;
    TargetCorrectionDirective last_directive_;
    AvoidanceObservability last_obs_;
};

}  // namespace expert
}  // namespace il_dataset
