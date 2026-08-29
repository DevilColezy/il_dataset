#pragma once
/// @file   macro_expert_5hz.hpp
/// @brief  5 Hz local-target arbiter (the upper expert).
///
/// The upper expert is NOT a second local planner.  It answers exactly one
/// question on every real 5 Hz boundary (tick % 6 == 0):
///
///   "Which target should the 30 Hz local expert chase right now?"
///
/// It never plans trajectories and never decides HOW the local expert
/// moves.  It only chooses among three directives:
///
///   * PASS        — keep chasing the ORIGINAL task goal;
///   * TURN(dir)   — stop translating and rotate the requested direction
///                   into the FOV (pure rotation, world-latched);
///   * CORRECTION  — chase a visible, in-FOV, locally-executable
///                   temporary waypoint (world-latched, event-driven).
///
/// The arbiter has exactly two modes:
///
///   DIRECT — the original goal is the attention centre:
///       goal trackable             → PASS
///       goal outside FOV           → TURN(goal direction)
///       local persistently blocked → BYPASS
///
///   BYPASS — the blocker / bypass frontier is the attention centre.  On
///       entry the PRIMARY blocker (the occupied component crossing the
///       vehicle→goal corridor) is identified and ONE side (LEFT/RIGHT) is
///       committed to.  Corrections are HELD (event-driven update, never
///       re-sampled every tick) until they are reached or no longer locally
///       feasible.  The original goal's FOV status is deliberately ignored
///       while in BYPASS.  BYPASS exits only when the original goal
///       re-enters the local expert's capability set.
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
    /// True while the current directive is NOT PASS_THROUGH (the macro is
    /// overriding the original target with a TURN or a CORRECTION).
    bool correctionActive() const {
        return last_directive_.type != TargetCorrectionType::PASS_THROUGH;
    }
    /// True while the arbiter is in BYPASS mode (see the file header).
    bool bypassActive() const { return bypass_active_; }
    SideSelection lockedSide() const { return locked_side_; }
    uint64_t correctionEnterEvent() const { return correction_enter_event_; }
    uint64_t correctionExitEvent() const { return correction_exit_event_; }
    uint64_t correctionUpdateEvent() const { return correction_update_event_; }
    uint64_t directiveUpdateEvent() const { return update_event_; }
    const TargetCorrectionDirective& lastDirective() const {
        return last_directive_;
    }
    /// Macro-owns-target guard (reported in 30 Hz ticks; 1 means the macro
    /// layer still owns the target).
    int reentryGuardRemaining(uint64_t /*tick*/) const {
        return correctionActive() ? 1 : 0;
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
    // DIRECT(false) / BYPASS(true) mode flag (see the file header).  In
    // DIRECT the original goal is the attention centre; in BYPASS the
    // committed blocker side / bypass frontier is the attention centre.
    bool bypass_active_ = false;
    // Side committed for the current BYPASS episode (NONE in DIRECT).
    SideSelection locked_side_ = SideSelection::NONE;
    uint64_t update_event_ = 0;
    uint64_t correction_enter_event_ = 0;
    uint64_t correction_exit_event_ = 0;
    uint64_t correction_update_event_ = 0;
    // Progress watchdog: consecutive 5 Hz updates with negligible motion
    // while a BYPASS correction is active.  Forces an event-driven
    // correction refresh instead of holding a stale waypoint forever.
    uint32_t stagnant_update_count_ = 0;
    Vec2d last_state_position_{0.0, 0.0};
    bool has_last_state_position_ = false;
    // Consecutive 5 Hz updates in which the original goal is locally
    // trackable (translation valid + in FOV + clear corridor).  BYPASS
    // exits only after this is confirmed several times in a row.
    uint32_t reentry_success_updates_ = 0;
    // Consecutive 5 Hz updates in which the committed BYPASS side produced
    // no feasible correction.  After a threshold the side commitment flips
    // (a rare, decisive event — never a per-tick side switch).
    uint32_t side_exhaustion_updates_ = 0;
    // Consecutive 5 Hz updates in which the currently held CORRECTION
    // waypoint is NOT actually executing (live_directive_usable=false) AND
    // its cold preview also fails.  Only after >= 3 consecutive failures is
    // the waypoint allowed to be dropped (a single cold preview miss must
    // not discard an actively-executing safe waypoint).
    uint32_t waypoint_execution_fail_updates_ = 0;
    TargetCorrectionDirective last_directive_;
    AvoidanceObservability last_obs_;
};

}  // namespace expert
}  // namespace il_dataset
