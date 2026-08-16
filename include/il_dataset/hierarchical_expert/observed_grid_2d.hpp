#pragma once
/// @file   observed_grid_2d.hpp
/// @brief  Persistent world-aligned local observation with short-term
///         history (parameterised max age).
///
/// Rules:
///   * a cell observed FREE or OCCUPIED stays known for up to
///     history_max_age_ticks, then decays back to UNKNOWN;
///   * an incoming UNKNOWN patch cell never overwrites a known cell;
///   * UNKNOWN is never usable as safe free space by the planner.
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
                   double resolution, uint32_t max_age_ticks);

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
    uint64_t first_observed_event_ = 0;
    bool seen_occupied_ = false;
};

}  // namespace expert
}  // namespace il_dataset
