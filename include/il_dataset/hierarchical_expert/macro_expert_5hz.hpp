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
///   * YES outside a correction episode → PASS_THROUGH;
///   * YES during a correction episode → PASS_THROUGH only after the
///     ORIGINAL goal also returns to the safe ordinary-direction handoff
///     cone; until then, keep the locked side and issue NORMAL_CORRECTION;
///   * NO  → temporarily correct the tracked target:
///       - NORMAL_CORRECTION: a quantized in-FOV frontier on the locked
///         side;
///       - TURN_LEFT / TURN_RIGHT: one bounded world-latched direction
///         step, initially just outside the FOV.
///
/// INFORMATION BOUNDARY (enforced by the interface).  At runtime the
/// corrector may ONLY read:
///   * VehicleState2D (pose, velocity, yaw, yaw rate);
///   * the ORIGINAL final goal;
///   * the CURRENT INSTANTANEOUS FOV patch (current_patch);
///   * the causal, decaying local history map built only from past patches;
///   * the current tick;
///   * its own memory.
/// It NEVER reads / receives / uses PlannerResult / PreviewResult /
/// failure counters / candidate trajectories / scene truth / global ESDF
/// / global paths.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/effective_target_adapter.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

class VisibilityTargetCorrector {
public:
    explicit VisibilityTargetCorrector(const Params2D& p)
        : p_(p), adapter_(p) {}

    /// Run on every real 5 Hz boundary (called exactly at tick % 6 == 0).
    TargetCorrectionDirective update(const VehicleState2D& state,
                                     const Vec2d& original_goal,
                                     const LocalObservation& current_patch,
                                     const LocalObservation& local_history,
                                     uint64_t tick);

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
    /// Remaining re-entry hysteresis guard in 30 Hz TICKS (0 = free).
    int reentryGuardRemaining(uint64_t tick) const {
        return tick < reentry_guard_until_tick_
                   ? static_cast<int>(reentry_guard_until_tick_ - tick)
                   : 0;
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
    LocalFreeGrid buildLocalFreeGrid(const VehicleState2D& state,
                                     const LocalObservation& patch) const;

    struct SideCandidate {
        Vec2d endpoint{0.0, 0.0};
        double bearing = 0.0;
        double dist = 0.0;
        double along_progress = 0.0;
        bool certified = false;
    };
    std::vector<SideCandidate> sampleSideCandidates(
        const VehicleState2D& state, const Vec2d& goal,
        const LocalObservation& patch, const LocalFreeGrid& grid,
        bool has_blocker, double blocker_min_along, SideSelection side,
        bool strict) const;

    bool chordClear(const VehicleState2D& state,
                    const LocalObservation& patch, const LocalFreeGrid& grid,
                    const Vec2d& endpoint) const;

    double freeRangeAlongFrom(const LocalObservation& obs, const Vec2d& from,
                              double bearing_world) const;
    double freeRangeAlong(const VehicleState2D& state,
                          const LocalObservation& obs,
                          double bearing_body) const;

    bool extractBlocker(const VehicleState2D& state, const Vec2d& goal,
                        const LocalObservation& patch,
                        double& blocker_min_along,
                        double& blocker_max_lateral) const;

    SideSelection selectSide(const VehicleState2D& state,
                             const LocalObservation& patch,
                             const Vec2d& goal) const;

    TargetCorrectionDirective makeCorrectionDirective(
        const VehicleState2D& state, const Vec2d& goal,
        const LocalObservation& patch, const LocalFreeGrid& grid,
        SideSelection side) const;

    bool directiveChanged(const TargetCorrectionDirective& a,
                          const TargetCorrectionDirective& b) const;

    AvoidanceObservability assessObservability(
        const VehicleState2D& state, const Vec2d& goal,
        const LocalObservation& patch) const;

    Params2D p_;
    EffectiveTargetAdapter adapter_;
    // ── 5 Hz internal memory (never anything from the 30 Hz outcome) ──
    bool correction_active_ = false;
    SideSelection locked_side_ = SideSelection::NONE;
    int enter_stable_count_ = 0;
    uint64_t reentry_guard_until_tick_ = 0;
    uint64_t update_event_ = 0;
    uint64_t correction_enter_event_ = 0;
    uint64_t correction_exit_event_ = 0;
    uint64_t correction_update_event_ = 0;
    TargetCorrectionDirective last_directive_;
    AvoidanceObservability last_obs_;
};

}  // namespace expert
}  // namespace il_dataset
