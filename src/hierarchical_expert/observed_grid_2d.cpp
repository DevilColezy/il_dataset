#include "il_dataset/hierarchical_expert/observed_grid_2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

void ObservedGrid2D::configure(const Vec2d& min_bounds, const Vec2d& max_bounds,
                               double resolution, uint32_t max_age_ticks,
                               uint32_t free_clear_confirmations) {
    obs_.resolution = resolution;
    obs_.origin = min_bounds;
    obs_.width = static_cast<int>(
        std::ceil((max_bounds.x() - min_bounds.x()) / resolution));
    obs_.height = static_cast<int>(
        std::ceil((max_bounds.y() - min_bounds.y()) / resolution));
    obs_.max_age_ticks = max_age_ticks;
    free_clear_confirmations_ = std::max<uint32_t>(1, free_clear_confirmations);
    reset();
}

void ObservedGrid2D::reset() {
    obs_.cells.assign(static_cast<size_t>(obs_.width) * obs_.height,
                      CellState::UNKNOWN);
    obs_.age_ticks.assign(static_cast<size_t>(obs_.width) * obs_.height, 0);
    free_confirm_.assign(static_cast<size_t>(obs_.width) * obs_.height, 0);
    obs_.tick = 0;
    first_observed_event_ = 0;
    seen_occupied_ = false;
}

void ObservedGrid2D::integrate(const LocalObservation& patch, uint64_t tick) {
    if (!patch.valid() || patch.resolution <= 0.0 || obs_.resolution <= 0.0)
        return;
    const double resolution_scale = patch.resolution / obs_.resolution;
    if (std::fabs(resolution_scale - 1.0) > 1e-9) {
        return;
    }
    const double offset_x_real =
        (patch.origin.x() - obs_.origin.x()) / obs_.resolution;
    const double offset_y_real =
        (patch.origin.y() - obs_.origin.y()) / obs_.resolution;
    const int offset_x = static_cast<int>(std::llround(offset_x_real));
    const int offset_y = static_cast<int>(std::llround(offset_y_real));
    if (std::fabs(offset_x_real - offset_x) > 1e-7 ||
        std::fabs(offset_y_real - offset_y) > 1e-7) {
        return;
    }

    obs_.tick = tick;

    // Age everything one tick first (cells older than max_age decay).
    const size_t n = obs_.cells.size();
    for (size_t i = 0; i < n; ++i) {
        if (obs_.cells[i] == CellState::UNKNOWN) continue;
        ++obs_.age_ticks[i];
        if (obs_.age_ticks[i] > obs_.max_age_ticks) {
            obs_.cells[i] = CellState::UNKNOWN;
            obs_.age_ticks[i] = 0;
        }
    }

    // Merge the fresh patch: only observed (FREE/OCCUPIED) patch cells
    // overwrite; UNKNOWN patch cells never erase known cells.
    //
    // R25: a cell the current frame sees FREE is no longer treated as
    // permanently occupied.  The merged OCCUPIED cell is kept only until
    // it has been re-confirmed FREE for `free_clear_confirmations_`
    // CONSECUTIVE current frames, then it is cleared to FREE (fresh
    // evidence outranks stale history).  Any fresh OCCUPIED observation
    // resets the confirmation counter.
    for (int iy = 0; iy < patch.height; ++iy) {
        for (int ix = 0; ix < patch.width; ++ix) {
            const CellState s = patch.cells[patch.idx(ix, iy)];
            if (s == CellState::UNKNOWN) continue;
            const int gx = offset_x + ix;
            const int gy = offset_y + iy;
            if (!obs_.inGrid(gx, gy)) continue;
            const size_t id = obs_.idx(gx, gy);
            if (s == CellState::FREE) {
                if (obs_.cells[id] == CellState::OCCUPIED) {
                    // Fresh FREE evidence against a stale OCCUPIED cell:
                    // confirm over several frames before clearing.
                    ++free_confirm_[id];
                    if (free_confirm_[id] >= free_clear_confirmations_) {
                        obs_.cells[id] = CellState::FREE;
                        obs_.age_ticks[id] = 0;
                        free_confirm_[id] = 0;
                    }
                    // else: stay OCCUPIED (conservative, not yet confirmed)
                } else {
                    obs_.cells[id] = CellState::FREE;
                    obs_.age_ticks[id] = 0;
                    free_confirm_[id] = 0;
                }
            } else {  // OCCUPIED (fresh hard evidence)
                obs_.cells[id] = CellState::OCCUPIED;
                obs_.age_ticks[id] = 0;
                free_confirm_[id] = 0;
            }
        }
    }

    // One-shot audit: first observed obstacle in this episode.
    if (!seen_occupied_) {
        for (const auto& s : obs_.cells) {
            if (s == CellState::OCCUPIED) {
                seen_occupied_ = true;
                first_observed_event_ = 1;
                break;
            }
        }
    }
}

double ObservedGrid2D::minClearanceToOccupied(const Vec2d& p, double r) const {
    return obs_.minClearanceToOccupied(p, r);
}

}  // namespace expert
}  // namespace il_dataset
