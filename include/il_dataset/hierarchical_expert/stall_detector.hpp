#pragma once
/// @file   stall_detector.hpp
/// @brief  Tiny PURE stall detector for the blueprint early-termination
///         (kept as a small state struct so the regression tests can
///         exercise it without constructing a full preflight / expert).
///
/// Semantics (blueprint-only; never changes expert labels):
///   * each tick the caller feeds the ACTUAL step displacement
///     (m/tick, computed BEFORE the previous-position bookkeeping is
///     updated — the P0 bug that made it always 0) and an `in_turn` flag;
///   * a tick counts as "stalled" when the drone is NOT in a legitimate
///     TURN update AND the displacement is below
///     `stall_speed_mps * dt` (threshold = speed x time per tick);
///   * the stall triggers after `window_ticks` consecutive stalled ticks;
///   * a pure TURN keeps position ~constant but is exempt via `in_turn`,
///     so turning in place can never accumulate a stall.

#include <cstdint>

namespace il_dataset {
namespace expert {

struct StallDetector {
    double disp_threshold = 0.0;  // stall_speed_mps * dt [m/tick]
    int window_ticks = 90;        // consecutive stalled ticks to trigger
    int consecutive = 0;

    /// Feed one tick.  Returns true exactly when the stall condition is
    /// reached on this tick (the caller then early-terminates).
    bool update(double step_disp, bool in_turn) {
        if (!in_turn && step_disp < disp_threshold) {
            ++consecutive;
        } else {
            consecutive = 0;
        }
        return consecutive >= window_ticks;
    }
    void reset() { consecutive = 0; }
};

}  // namespace expert
}  // namespace il_dataset
