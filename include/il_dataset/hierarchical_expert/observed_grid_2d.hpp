#pragma once
/// @file   observed_grid_2d.hpp
/// @brief  Persistent world-aligned local observation with short-term
///         history (parameterised max age).
///
/// Rules:
///   * a cell observed FREE or OCCUPIED stays known for up to
///     history_max_age_ticks, then decays back to UNKNOWN;
///   * an incoming UNKNOWN patch cell never overwrites a known cell;
///   * UNKNOWN is never usable as safe free space by the planner;
///   * R25: a cell previously OCCUPIED is cleared back to FREE only after
///     free_clear_confirmations CONSECUTIVE current-frame FREE
///     observations (a single-frame depth gap must not erase a real
///     obstacle, but a stale history cell must not fabricate a permanent
///     blockage — measured: 1655/1791 frames reported local BLOCKED from
///     a clear current frame in joint_v2_000004_4ab1e354).
///
/// Also counts `obstacle_first_observed_event` — the first time any
/// OCCUPIED cell enters the local map since the last reset (one-shot
/// audit event per episode).

#include "il_dataset/hierarchical_expert/types.hpp"

namespace il_dataset {
namespace expert {

class ObservedGrid2D {
public:
    ObservedGrid2D() = default;

    /// Configure the persistent grid over the whole region.
    void configure(const Vec2d& min_bounds, const Vec2d& max_bounds,
                   double resolution, uint32_t max_age_ticks,
                   uint32_t free_clear_confirmations = 3);

    void reset();

    /// Merge one instantaneous FOV patch and age all cells by one tick.
    void integrate(const LocalObservation& patch, uint64_t tick);

    const LocalObservation& observation() const { return obs_; }
    LocalObservation& observation() { return obs_; }

    /// Min distance from `p` to any OCCUPIED cell centre inside the
    /// radius-r search box; inf if none.
    double minClearanceToOccupied(const Vec2d& p, double r) const;

    /// One-shot audit event: 0 until the first OCCUPIED cell is seen,
    /// then 1 (reset with the grid).
    uint64_t obstacleFirstObservedEvent() const { return first_observed_event_; }

private:
    LocalObservation obs_;
    // R25: per-cell counter of CONSECUTIVE current-frame FREE observations
    // while the merged cell is still OCCUPIED.  Reaches free_clear_confirmations_
    // -> the stale OCCUPIED cell is cleared to FREE.
    std::vector<uint8_t> free_confirm_;
    uint32_t free_clear_confirmations_ = 3;
    uint64_t first_observed_event_ = 0;
    bool seen_occupied_ = false;
};

}  // namespace expert
}  // namespace il_dataset
