#include "il_dataset/hierarchical_expert/macro_expert_5hz.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>

namespace il_dataset {
namespace expert {

// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
//  Local free-grid construction + observability judgement
//  (current patch + decaying local history 鈥?deterministic, local, causal)
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
VisibilityTargetCorrector::LocalFreeGrid
VisibilityTargetCorrector::buildLocalFreeGrid(const PlanarState& state,
                                              const LocalObservation& patch) const {
    LocalFreeGrid grid;
    if (!patch.valid()) return grid;
    grid.resolution = patch.resolution;

    const double half = p_.obs_range_m + p_.lp_min_clearance + 2.0;
    const Vec2d lo = state.position - Vec2d(half, half);
    const Vec2d hi = state.position + Vec2d(half, half);
    const GridIndex2D g0 = worldToGrid(lo, patch.origin, patch.resolution);
    const GridIndex2D g1 = worldToGrid(hi, patch.origin, patch.resolution);
    const int ix0 = std::max(0, g0.ix);
    const int iy0 = std::max(0, g0.iy);
    const int ix1 = std::min(patch.width - 1, g1.ix);
    const int iy1 = std::min(patch.height - 1, g1.iy);
    if (ix1 < ix0 || iy1 < iy0) return grid;
    grid.width = ix1 - ix0 + 1;
    grid.height = iy1 - iy0 + 1;
    grid.origin = Vec2d(patch.origin.x() + ix0 * grid.resolution,
                        patch.origin.y() + iy0 * grid.resolution);
    grid.free.assign(static_cast<size_t>(grid.width) * grid.height, 0);
    grid.blocked.assign(static_cast<size_t>(grid.width) * grid.height, 0);
    grid.reachable.assign(static_cast<size_t>(grid.width) * grid.height, 0);

    // Unified collision distance (USER DIRECTIVE): 4 cells = 0.4 m from an
    // OCCUPIED cell centre.  No ESDF geometric envelope.
    const double clearance_margin =
        std::max(0.0, p_.lp_clearance_discretization_margin_m);
    const double req = std::max(
        {0.0, p_.lp_min_clearance + clearance_margin,
         p_.lp_nominal_clearance_m + clearance_margin});
    const double inv = 1.0 / std::max(1e-6, grid.resolution);
    const int k = static_cast<int>(std::ceil(req * inv));
    const double kr2 = (req * inv) * (req * inv);

    for (int iy = iy0; iy <= iy1; ++iy) {
        for (int ix = ix0; ix <= ix1; ++ix) {
            const CellState st = patch.at(ix, iy);
            const int lx = ix - ix0, ly = iy - iy0;
            if (st == CellState::FREE) {
                grid.free[grid.idx(lx, ly)] = 1;
                continue;
            }
            if (st != CellState::OCCUPIED) continue;  // UNKNOWN stays 0
            for (int dy = -k; dy <= k; ++dy) {
                for (int dx = -k; dx <= k; ++dx) {
                    const double d2 = static_cast<double>(dx) * dx +
                                      static_cast<double>(dy) * dy;
                    if (d2 > kr2 + 1e-9) continue;
                    const int nx = lx + dx, ny = ly + dy;
                    if (nx < 0 || ny < 0 || nx >= grid.width ||
                        ny >= grid.height) {
                        continue;
                    }
                    grid.blocked[grid.idx(nx, ny)] = 1;
                }
            }
        }
    }

    // 8-connected flood fill over known-FREE, clearance-valid cells from
    // the vehicle cell.
    const GridIndex2D gv =
        worldToGrid(state.position, grid.origin, grid.resolution);
    if (!grid.inGrid(gv.ix, gv.iy)) return grid;
    const size_t seed = grid.idx(gv.ix, gv.iy);
    if (grid.free[seed] == 0) return grid;
    std::deque<size_t> stack{seed};
    grid.reachable[seed] = 1;
    const int di[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dj[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const auto cellAllowed = [&](int x, int y) {
        if (!grid.inGrid(x, y)) return false;
        const size_t id = grid.idx(x, y);
        if (grid.free[id] == 0) return false;
        if (grid.blocked[id] == 0) return true;
        const Vec2d w =
            gridCellCenter(x, y, grid.origin, grid.resolution);
        return (w - state.position).norm() <=
               p_.macro_local_recovery_prefix_m;
    };
    while (!stack.empty()) {
        const size_t cur = stack.front();
        stack.pop_front();
        const int cx = static_cast<int>(cur % static_cast<size_t>(grid.width));
        const int cy = static_cast<int>(cur / static_cast<size_t>(grid.width));
        for (int d = 0; d < 8; ++d) {
            const int nx = cx + dj[d], ny = cy + di[d];
            if (!grid.inGrid(nx, ny)) continue;
            const size_t ni = grid.idx(nx, ny);
            if (grid.reachable[ni] != 0) continue;
            if (!cellAllowed(nx, ny)) continue;
            if (nx != cx && ny != cy &&
                (!cellAllowed(cx, ny) || !cellAllowed(nx, cy))) {
                continue;
            }
            grid.reachable[ni] = 1;
            stack.push_back(ni);
        }
    }
    return grid;
}

double VisibilityTargetCorrector::freeRangeAlongFrom(
    const LocalObservation& obs, const Vec2d& from,
    double bearing_world) const {
    if (!obs.valid()) return 0.0;
    const Vec2d dir(std::cos(bearing_world), std::sin(bearing_world));
    const double step = obs.resolution * 0.5;
    double range = 0.0;
    for (double d = step; d <= p_.obs_range_m + 1e-9; d += step) {
        const Vec2d p = from + dir * d;
        if (obs.atWorld(p.x(), p.y()) != CellState::FREE) break;
        range = d;
    }
    return range;
}

double VisibilityTargetCorrector::freeRangeAlong(
    const PlanarState& state, const LocalObservation& obs,
    double bearing_body) const {
    return freeRangeAlongFrom(obs, state.position, state.yaw + bearing_body);
}

/// Distance along a world bearing to the FIRST observed OCCUPIED cell
/// (UNKNOWN is PASSABLE — the R29 contract).  Returns max_range when no
/// occupied cell is met.  Unlike freeRangeAlongFrom (which stops on any
/// non-FREE cell), this ignores UNKNOWN, so a bearing that is merely
/// outside the current FOV does not look blocked.
double VisibilityTargetCorrector::occupiedRangeAlongFrom(
    const LocalObservation& obs, const Vec2d& from,
    double bearing_world, double max_range) const {
    if (!obs.valid() || max_range <= 0.0) return max_range;
    const Vec2d dir(std::cos(bearing_world), std::sin(bearing_world));
    const double step = obs.resolution * 0.5;
    double range = max_range;
    for (double d = step; d <= max_range + 1e-9; d += step) {
        const Vec2d p = from + dir * d;
        if (obs.atWorld(p.x(), p.y()) == CellState::OCCUPIED) {
            range = d;
            break;
        }
    }
    return range;
}

bool VisibilityTargetCorrector::extractBlocker(
    const PlanarState& state, const Vec2d& goal,
    const LocalObservation& patch, double& blocker_min_along,
    double& blocker_lat_min, double& blocker_lat_max) const {
    blocker_min_along = std::numeric_limits<double>::infinity();
    blocker_lat_min = std::numeric_limits<double>::infinity();
    blocker_lat_max = -std::numeric_limits<double>::infinity();
    if (!patch.valid()) return false;
    const Vec2d axis = goal - state.position;
    const double axis_len = std::max(1e-6, axis.norm());
    const Vec2d dir = axis / axis_len;
    const double hw = p_.macro_corridor_half_width;
    // An obstacle only blocks this task if it is reached before the goal.
    // (blocker along <= min(perception_range, goal_distance + tolerance))
    const double max_along = std::min(
        p_.obs_range_m,
        axis_len + std::max(0.0, p_.task_goal_tolerance));
    bool found = false;
    for (int iy = 0; iy < patch.height; ++iy) {
        for (int ix = 0; ix < patch.width; ++ix) {
            if (patch.cells[patch.idx(ix, iy)] != CellState::OCCUPIED) {
                continue;
            }
            const Vec2d p(patch.origin.x() + (ix + 0.5) * patch.resolution,
                          patch.origin.y() + (iy + 0.5) * patch.resolution);
            const Vec2d rel = p - state.position;
            // SIGNED lateral, goal-axis convention: cross2(dir, rel) > 0
            // is LEFT, < 0 is RIGHT (matches the best_clear lat test and
            // sampleSideCandidates).
            const double lat = cross2(dir, rel);
            const double along = rel.dot(dir);
            if (std::fabs(lat) <= hw &&
                along >= -p_.macro_corridor_rear_tolerance_m &&
                along <= max_along) {
                blocker_min_along = std::min(blocker_min_along, along);
                blocker_lat_min = std::min(blocker_lat_min, lat);
                blocker_lat_max = std::max(blocker_lat_max, lat);
                found = true;
            }
        }
    }
    // Lateral-span gate (align with il_2d_multiscale_debug
    // macro/blocking_lateral_span_ratio): the corridor is only BLOCKED when
    // the observed OCCUPIED cells span >= this fraction of the corridor
    // half-width across the goal axis.  A single small obstacle (or a 1-cell
    // depth artifact) that barely touches the corridor edge is left to the
    // 30 Hz planner to weave around; the 5 Hz corrector must NOT take over
    // for it.  (v8 identifyBlocker design: corridor-crossing cluster,
    // lateral span >= corridor_half_width * ratio.)
    if (found &&
        (blocker_lat_max - blocker_lat_min) <
            hw * p_.macro_blocking_lateral_span_ratio) {
        found = false;
    }
    return found;
}

AvoidanceObservability VisibilityTargetCorrector::assessObservability(
    const PlanarState& state, const Vec2d& goal,
    const LocalObservation& patch) const {
    AvoidanceObservability obs;
    obs.reason = "NO_PATCH";
    if (!patch.valid()) return obs;

    const double fov_half = deg2rad(p_.obs_fov_deg) / 2.0;
    const Vec2d to_goal = goal - state.position;
    const double goal_dist = to_goal.norm();
    const Vec2d axis_dir =
        goal_dist > 1e-9 ? to_goal / goal_dist : Vec2d(1.0, 0.0);
    const double b_goal =
        wrapAngle(std::atan2(to_goal.y(), to_goal.x()) - state.yaw);
    obs.goal_inside_fov = std::fabs(b_goal) <= fov_half + 1e-9;

    // 1) Direct corridor check (vehicle 鈫?original goal, to perception
    //    range). Only locally observed OCCUPIED blocks.
    double blocker_min_along = std::numeric_limits<double>::infinity();
    double blocker_lat_min = std::numeric_limits<double>::infinity();
    double blocker_lat_max = -std::numeric_limits<double>::infinity();
    const bool has_blocker =
        extractBlocker(state, goal, patch, blocker_min_along,
                       blocker_lat_min, blocker_lat_max);
    obs.direct_corridor_blocked = has_blocker;
    obs.blocker_observed = has_blocker;
    obs.blocker_min_along = blocker_min_along;

    // UNKNOWN occlusion along the goal-axis corridor (diagnostic).
    {
        const double range = std::min(p_.obs_range_m, goal_dist);
        const double step = patch.resolution * 0.5;
        for (double d = step; d <= range + 1e-9; d += step) {
            const Vec2d p = state.position + axis_dir * d;
            if (patch.atWorld(p.x(), p.y()) == CellState::UNKNOWN) {
                obs.unknown_occluded = true;
                break;
            }
        }
    }

    // 2) Local bypass exits: known-free grid + flood fill + per-side
    //    certified bypass candidates (strict test).
    const LocalFreeGrid grid = buildLocalFreeGrid(state, patch);
    if (!grid.valid()) {
        obs.local_avoidance_observable =
            obs.goal_inside_fov && !obs.direct_corridor_blocked;
        if (obs.direct_corridor_blocked) {
            obs.reason = "NO_GRID_BLOCKED";
        } else if (!obs.goal_inside_fov) {
            obs.reason = "GOAL_OUTSIDE_FOV";
        } else {
            obs.reason = "NO_GRID_CLEAR";
        }
        return obs;
    }
    const std::vector<SideCandidate> left =
        sampleSideCandidates(state, goal, patch, grid, has_blocker,
                             blocker_min_along, SideSelection::LEFT,
                             /*strict=*/true);
    const std::vector<SideCandidate> right =
        sampleSideCandidates(state, goal, patch, grid, has_blocker,
                             blocker_min_along, SideSelection::RIGHT,
                             /*strict=*/true);
    obs.left_bypass_observable = !left.empty();
    obs.right_bypass_observable = !right.empty();
    for (const SideCandidate& c : left) {
        obs.left_score = std::max(obs.left_score, c.dist);
    }
    for (const SideCandidate& c : right) {
        obs.right_score = std::max(obs.right_score, c.dist);
    }
    obs.local_avoidance_observable =
        (obs.goal_inside_fov && !obs.direct_corridor_blocked) ||
        obs.left_bypass_observable ||
        obs.right_bypass_observable;

    // 3) Truncation diagnostics (only meaningful when a blocker exists and
    //    no bypass is visible).
    if (obs.direct_corridor_blocked && !obs.left_bypass_observable &&
        !obs.right_bypass_observable) {
        const double fov_hw_at_blocker =
            std::tan(fov_half) * std::max(1e-3, blocker_min_along);
        const double blocker_max_abs = std::max(
            std::fabs(blocker_lat_min), std::fabs(blocker_lat_max));
        obs.fov_boundary_truncated =
            blocker_min_along >= p_.obs_range_m - 1.0 ||
            blocker_max_abs >= 0.9 * fov_hw_at_blocker;
    }

    if (obs.left_bypass_observable || obs.right_bypass_observable) {
        obs.reason = "BYPASS_VISIBLE";
    } else if (obs.local_avoidance_observable) {
        obs.reason = "CORRIDOR_CLEAR";
    } else if (!obs.goal_inside_fov && !obs.direct_corridor_blocked) {
        obs.reason = "GOAL_OUTSIDE_FOV";
    } else if (obs.unknown_occluded) {
        obs.reason = "BLOCKED_UNKNOWN_OCCLUDED";
    } else if (obs.fov_boundary_truncated) {
        obs.reason = "BLOCKED_FOV_TRUNCATED";
    } else {
        obs.reason = "BLOCKED_NO_BYPASS";
    }
    return obs;
}

// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
//  Side candidate sampling / certification
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
std::vector<VisibilityTargetCorrector::SideCandidate>
VisibilityTargetCorrector::sampleSideCandidates(
    const PlanarState& state, const Vec2d& goal,
    const LocalObservation& patch, const LocalFreeGrid& grid,
    bool has_blocker, double blocker_min_along, SideSelection side,
    bool strict) const {
    std::vector<SideCandidate> out;
    if (!patch.valid() || !grid.valid()) return out;

    const double fov_half = deg2rad(p_.obs_fov_deg) / 2.0;
    const double margin = deg2rad(p_.te_turn_ray_margin_deg);
    const double b_lo = -fov_half + margin;
    const double b_hi = fov_half - margin;
    if (!(b_hi > b_lo)) return out;

    const Vec2d to_goal = goal - state.position;
    const double goal_dist = to_goal.norm();
    const Vec2d axis =
        goal_dist > 1e-9 ? to_goal / goal_dist : Vec2d(1.0, 0.0);
    const double b_goal =
        wrapAngle(std::atan2(to_goal.y(), to_goal.x()) - state.yaw);

    const double step_b = deg2rad(p_.macro_local_candidate_bearing_step_deg);
    const double dmin = std::max(p_.macro_observable_frontier_min_distance_m,
                                 2.0 * patch.resolution);
    const double dmax = p_.obs_range_m - 0.5 * patch.resolution;
    // A macro correction is a bounded temporary waypoint, not a terminal
    // mission goal.  Candidate distance is limited by local observability;
    // the progress checks below also prevent successive waypoints from
    // walking indefinitely away from the original goal.
    const double candidate_max_distance =
        std::min(dmax, p_.macro_guide_horizon_m);
    const double step_d = p_.macro_local_candidate_distance_step_m;
    if (!(step_b > 0.0) || !(step_d > 0.0) ||
        !(candidate_max_distance >= dmin)) {
        return out;
    }

    // Strict bypass exits must lie PAST the nearest blocker surface.
    const double beyond =
        (strict && has_blocker)
            ? blocker_min_along + patch.resolution
            : -std::numeric_limits<double>::infinity();

    const int dir = side == SideSelection::LEFT ? 1 : -1;
    const double b_start =
        side == SideSelection::LEFT ? std::max(b_goal, b_lo)
                                    : std::min(b_goal, b_hi);
    const double b_end = side == SideSelection::LEFT ? b_hi : b_lo;

    for (int i = 0; i < 2000; ++i) {
        const double b = b_start + static_cast<double>(dir * i) * step_b;
        if (dir > 0 && b > b_end + 1e-9) break;
        if (dir < 0 && b < b_end - 1e-9) break;
        if (std::fabs(b) > fov_half - margin + 1e-9) continue;
        const Vec2d dir_world(std::cos(state.yaw + b),
                              std::sin(state.yaw + b));
        for (double d = dmin; d <= candidate_max_distance + 1e-9;
             d += step_d) {
            const Vec2d endpoint = state.position + dir_world * d;
            // Lateral sign must match the side (relative to the goal axis):
            // cross(axis, rel) > 0 = LEFT, < 0 = RIGHT.
            const double lat = cross2(axis, endpoint - state.position);
            if ((side == SideSelection::LEFT && lat < -0.05) ||
                (side == SideSelection::RIGHT && lat > 0.05)) {
                continue;
            }
            // A correction on the goal line is not a bypass when an observed
            // blocker spans the direct corridor.  Require lateral separation
            // so the local planner receives an executable detour target.
            if (has_blocker) {
                const double min_lateral = std::max(
                    0.5, p_.macro_corridor_half_width *
                             p_.macro_blocking_lateral_span_ratio);
                if (std::fabs(lat) < min_lateral) continue;
            }
            const double along = (endpoint - state.position).dot(axis);
            if (along < p_.macro_observable_frontier_min_progress_m) {
                continue;
            }
            // A temporary waypoint must make measurable progress toward the
            // original mission goal.  Mere local reachability is not enough:
            // an indefinitely reachable side ray previously carried the
            // vehicle from 7.2 m to 14.3 m away from the goal.
            const double endpoint_goal_dist = (goal - endpoint).norm();
            if (endpoint_goal_dist >
                goal_dist - p_.macro_observable_frontier_min_progress_m) {
                continue;
            }
            if (strict &&
                std::fabs(wrapAngle(b - b_goal)) > fov_half + 1e-9) {
                continue;
            }
            if (strict && along < beyond) continue;
            if (!grid.reachableAt(endpoint)) continue;
            // USER DIRECTIVE: NO straight-chord validation for the guide 鈥?
            // the guide only needs its OWN point to be free with
            // >= lp_min_clearance (0.5) to the nearest observed cell (the
            // 0.5-inflated grid enforces that via traversableAt); the
            // 30 Hz planner routes around obstacles itself.  A straight-
            // chord requirement forced the guide to the SIDE of every
            // blocker (measured: task254 first guide ~50掳 off the goal
            // line) instead of letting it sit on the goal line beyond the
            // blocker, and pinned it at the perception edge.
            // Applied to BOTH strict (bypass certification + side lock)
            // and non-strict (guide placement): a strict bypass that only
            // threads 0.37 m from a surface is not a certifiable escape,
            // and a guide placed inside the 0.5 m band is unreachable by
            // the 30 Hz A* (its 0.5 m inflation blocks the goal cell) 鈥?
            // measured: joint_v2_000000_0e798c93 guide (2.88,15.86) at
            // 0.37 m from obstacle (4.07,15.05) r1.07; the local plan
            // could never end there and fell to the FOV-edge scan.
            if (!grid.traversableAt(endpoint)) continue;
            // NOT truncated: known-FREE continuation beyond the endpoint
            // along the same ray.
            const double cont = freeRangeAlongFrom(
                patch, endpoint, state.yaw + b);
            if (cont < p_.macro_observable_unknown_margin_cells *
                           patch.resolution) {
                continue;
            }
            SideCandidate c;
            c.endpoint = endpoint;
            c.bearing = b;
            c.dist = d;
            c.along_progress = along;
            c.certified = true;
            out.push_back(c);
        }
    }
    return out;
}

// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
//  Side selection (strict bypass certification first, then free-range)
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
SideSelection VisibilityTargetCorrector::selectSide(
    const PlanarState& state, const LocalObservation& patch,
    const Vec2d& goal) const {
    if (!patch.valid()) return SideSelection::RIGHT;

    // 鈹€鈹€ PRIMARY: strict bypass-observability preference 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    // If exactly ONE side has a certifiable strict bypass on the current
    // local map, lock that side.  The free-range metric below can disagree
    // with the strict certification (measured: task30 entry had
    // right_bypass_observable=1 / left=0 yet the free-range + goal-side
    // fallback locked LEFT, leading the drone into a west dead-end pocket
    // where it was boxed in forever).  A certified strict bypass (known-
    // free reachable path + free continuation beyond the FOV boundary) is
    // real evidence of an escape route; the free-range average is not.
    {
        const LocalFreeGrid grid = buildLocalFreeGrid(state, patch);
        if (grid.valid()) {
            double blocker_min_along = std::numeric_limits<double>::infinity();
            double blocker_lat_min = std::numeric_limits<double>::infinity();
            double blocker_lat_max = -std::numeric_limits<double>::infinity();
            const bool has_blocker =
                extractBlocker(state, goal, patch, blocker_min_along,
                               blocker_lat_min, blocker_lat_max);
            const bool left_ok = !sampleSideCandidates(
                state, goal, patch, grid, has_blocker, blocker_min_along,
                SideSelection::LEFT, /*strict=*/true).empty();
            const bool right_ok = !sampleSideCandidates(
                state, goal, patch, grid, has_blocker, blocker_min_along,
                SideSelection::RIGHT, /*strict=*/true).empty();
            if (left_ok != right_ok) {
                return left_ok ? SideSelection::LEFT : SideSelection::RIGHT;
            }
        }
    }

    // 鈹€鈹€ FALLBACK: paired-ray free-range evidence 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    const double fov = deg2rad(p_.obs_fov_deg);
    const double b_goal =
        wrapAngle(std::atan2(goal.y() - state.position.y(),
                             goal.x() - state.position.x()) -
                  state.yaw);
    const double d_beta = deg2rad(p_.macro_evidence_ray_step_deg);

    double left_total = 0.0, right_total = 0.0;
    int pairs = 0;
    for (double db = d_beta; db <= fov / 2.0 - 1e-6; db += d_beta) {
        const double bl = b_goal + db;  // LEFT (positive bearing)
        const double br = b_goal - db;  // RIGHT
        if (std::fabs(bl) > fov / 2.0 || std::fabs(br) > fov / 2.0) continue;
        left_total += freeRangeAlong(state, patch, bl);
        right_total += freeRangeAlong(state, patch, br);
        ++pairs;
    }
    if (pairs < std::max(1, p_.macro_min_evidence_ray_pairs)) {
        // Too little paired evidence to judge free space 鈥?prefer the
        // side the ORIGINAL goal lies on (same rule as the ambiguity
        // fallback below; fixed RIGHT only when the goal is dead ahead).
        if (std::fabs(b_goal) > 1e-9) {
            return b_goal > 0.0 ? SideSelection::LEFT
                                : SideSelection::RIGHT;
        }
        return SideSelection::RIGHT;
    }
    const double left_avg = left_total / static_cast<double>(pairs);
    const double right_avg = right_total / static_cast<double>(pairs);
    const double m = p_.macro_side_evidence_margin;
    if (left_avg > right_avg + m) return SideSelection::LEFT;
    if (right_avg > left_avg + m) return SideSelection::RIGHT;

    // Ambiguous free-range evidence 鈫?prefer the side the ORIGINAL goal
    // lies on (b_goal > 0 = goal LEFT of the nose in the expert frame).
    // The drone enters correction heading at the goal line, so the
    // goal-side bypass is the minimal-turn detour (no backtracking across
    // the blocker).  Fixed-RIGHT here sent joint_v2_000002 on an east
    // detour away from a west goal (it then stalled at the blocker
    // boundary) and turned joint_v2_000000 (a turn_right task) into a
    // LEFT bypass that the preflight predicted as TURN_RIGHT.
    if (std::fabs(b_goal) > 1e-9) {
        return b_goal > 0.0 ? SideSelection::LEFT : SideSelection::RIGHT;
    }
    return SideSelection::RIGHT;  // goal exactly dead ahead 鈫?fixed RIGHT
}

// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
//  Correction directive construction
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
TargetCorrectionDirective VisibilityTargetCorrector::makeCorrectionDirective(
    const PlanarState& state, const Vec2d& goal,
    const LocalObservation& patch, const LocalFreeGrid& grid,
    SideSelection side, bool live_directive_usable,
    bool drop_held_waypoint_allowed,
    const DirectiveAssessmentFn& assess_directive) const {
    TargetCorrectionDirective d;
    d.valid = true;
    d.locked_side = side;
    d.type = side == SideSelection::LEFT
                 ? TargetCorrectionType::TURN_LEFT
                 : TargetCorrectionType::TURN_RIGHT;
    // Default: one bounded view-rotation step (world-latched).
    const int n = std::max(3, p_.te_direction_bin_count);
    d.direction_token = side == SideSelection::LEFT ? 0 : n + 1;
    d.decoded_direction_body = adapter_.decodeDirectionToken(d.direction_token);
    d.normalized_distance = 1.0;  // TURN classes: EXACT 1.0
    d.turn_direction_world =
        rot2(d.decoded_direction_body, state.yaw).normalized();
    d.turn_direction_world_valid = true;
    d.reason = side == SideSelection::LEFT ? "TURN_LEFT_NO_FRONTIER"
                                           : "TURN_RIGHT_NO_FRONTIER";

    // Try NORMAL_CORRECTION candidates on the locked side.  Geometry only
    // proposes candidates; the exact same 30 Hz local planner preview that
    // executes the target is the final feasibility authority.
    double blocker_min_along = std::numeric_limits<double>::infinity();
    double blocker_lat_min = std::numeric_limits<double>::infinity();
    double blocker_lat_max = -std::numeric_limits<double>::infinity();
    const bool has_blocker =
        extractBlocker(state, goal, patch, blocker_min_along,
                       blocker_lat_min, blocker_lat_max);
    std::vector<SideCandidate> cands =
        sampleSideCandidates(state, goal, patch, grid, has_blocker,
                             blocker_min_along, side, /*strict=*/false);
    std::sort(cands.begin(), cands.end(),
              [](const SideCandidate& a, const SideCandidate& b) {
                  if (a.along_progress != b.along_progress) {
                      return a.along_progress > b.along_progress;
                  }
                  return a.dist > b.dist;
              });

    // R29l (user redesign): re-sample every 5 Hz and adopt a BETTER
    // candidate, but stay on a currently executing waypoint unless the
    // fresh sample is clearly better (more progress along the goal axis)
    // or the held waypoint is no longer executable.  This keeps the 5 Hz
    // observation fresh — a nearer detour exit is picked up immediately
    // instead of waiting for the held waypoint to be reached (the old
    // "arrive-then-update" stalled around large cylinders, e.g. large_short
    // stuck at (-4.05, 12.97) after reaching its waypoint) — while the
    // along-progress margin prevents preview noise from jittering the
    // waypoint every boundary.
    //
    // R25 (kept): a single cold-start preview miss must NOT discard a
    // waypoint that is still actually executing; and a waypoint is never
    // extrapolated from the live pose (that turns a finite waypoint into
    // an endless ray).  The held waypoint is only abandoned for a fresh
    // candidate that is strictly better, or when it stops being executable.
    if (last_directive_.type == TargetCorrectionType::NORMAL_CORRECTION &&
        last_directive_.normalized_distance > 1e-9 &&
        last_directive_.corrected_target_world_valid) {
        const Vec2d previous_delta =
            last_directive_.corrected_target_world - state.position;
        const double waypoint_reached_tolerance = waypointReachedTolerance();
        if (previous_delta.norm() > waypoint_reached_tolerance) {
            TargetCorrectionDirective held = last_directive_;
            const double bearing = wrapAngle(
                std::atan2(previous_delta.y(), previous_delta.x()) -
                state.yaw);
            held.direction_token = adapter_.quantizeBearing(bearing);
            held.decoded_direction_body =
                adapter_.decodeDirectionToken(held.direction_token);
            held.normalized_distance =
                adapter_.clampNormalizedDistance(previous_delta.norm());
            held.turn_direction_world_valid = false;
            held.locked_side = side;
            held.reason = "FIXED_WAYPOINT_HELD";
            const LocalPlanningAssessment held_assessment =
                assess_directive(held);
            const bool held_ok =
                held_assessment.translation_plan_valid ||
                held_assessment.rotation_available || live_directive_usable;
            // Progress of the held waypoint along the goal axis (the same
            // axis sampleSideCandidates uses for along_progress).
            const Vec2d to_goal = goal - state.position;
            const double goal_dist = to_goal.norm();
            const Vec2d axis =
                goal_dist > 1e-9 ? to_goal / goal_dist : Vec2d(1.0, 0.0);
            const double held_along = previous_delta.dot(axis);
            // Best fresh candidate: cands are sorted by along_progress
            // descending, so the first preview-valid one is the best.
            double best_cand_along = -1e9;
            size_t n_preview = 0;
            constexpr size_t kHeldPreviewBudget = 8;
            std::vector<uint8_t> held_tokens(
                static_cast<size_t>(std::max(3, p_.te_direction_bin_count) + 2),
                0);
            for (const SideCandidate& cand : cands) {
                if (n_preview >= kHeldPreviewBudget) break;
                const int token = adapter_.quantizeBearing(cand.bearing);
                if (held_tokens[static_cast<size_t>(token)] != 0) continue;
                held_tokens[static_cast<size_t>(token)] = 1;
                TargetCorrectionDirective cand_dir = d;
                cand_dir.direction_token = token;
                cand_dir.decoded_direction_body =
                    adapter_.decodeDirectionToken(token);
                cand_dir.normalized_distance =
                    adapter_.clampNormalizedDistance(cand.dist);
                const Vec2d cdw = rot2(cand_dir.decoded_direction_body,
                                       state.yaw);
                cand_dir.corrected_target_world =
                    state.position + cdw * adapter_.normalizedToWorld(
                        cand_dir.normalized_distance);
                cand_dir.corrected_target_world_valid = true;
                cand_dir.turn_direction_world_valid = false;
                cand_dir.type = TargetCorrectionType::NORMAL_CORRECTION;
                const LocalPlanningAssessment ca = assess_directive(cand_dir);
                ++n_preview;
                if (ca.translation_plan_valid) {
                    best_cand_along = cand.along_progress;
                    break;
                }
            }
            const bool adopt_new =
                !held_ok ||
                (best_cand_along > held_along +
                     p_.macro_waypoint_update_along_margin);
            if (!adopt_new) {
                return held;
            }
            // Otherwise fall through to the candidate loop below: a fresh,
            // strictly-better candidate (or the held one is no longer
            // executable) is selected there.
        }
    }

    std::vector<uint8_t> previewed_tokens(
        static_cast<size_t>(std::max(3, p_.te_direction_bin_count) + 2), 0);
    size_t preview_count = 0;
    constexpr size_t kMaxCandidatePreviews = 16;
    for (const SideCandidate& chosen : cands) {
        TargetCorrectionDirective candidate = d;
        const int token = adapter_.quantizeBearing(chosen.bearing);
        if (previewed_tokens[static_cast<size_t>(token)] != 0) continue;
        previewed_tokens[static_cast<size_t>(token)] = 1;
        // Use the EXACT geometry-checked world point (chosen.endpoint) as
        // the corrected target — no quantization re-projection.  The 5 Hz
        // student regresses the continuous FLU direction, which is
        // re-derived live from corrected_target_world in the encoder, so a
        // quantized re-projection (up to ~3.5°, ~0.29 m at 4.8 m) would
        // silently teach/execute a target off the geometry-validated point.
        // The direction token is quantized from the exact bearing purely
        // for the class/token fields.
        candidate.corrected_target_world = chosen.endpoint;
        candidate.corrected_target_world_valid = true;
        candidate.turn_direction_world_valid = false;
        candidate.type = TargetCorrectionType::NORMAL_CORRECTION;
        candidate.reason = "NORMAL_CORRECTION_PREVIEW_CERTIFIED";
        const Vec2d delta = chosen.endpoint - state.position;
        const double d_norm = delta.norm();
        const Vec2d dir_world =
            d_norm > 1e-9 ? delta / d_norm : Vec2d(1.0, 0.0);
        candidate.decoded_direction_body = rot2(dir_world, -state.yaw);
        candidate.direction_token = token;
        candidate.normalized_distance =
            adapter_.clampNormalizedDistance(d_norm);
        if ((goal - candidate.corrected_target_world).norm() >
            (goal - state.position).norm() -
                p_.macro_observable_frontier_min_progress_m) {
            continue;
        }

        ++preview_count;
        const LocalPlanningAssessment candidate_assessment =
            assess_directive(candidate);
        if (candidate_assessment.translation_plan_valid) {
            return candidate;
        }
        if (preview_count >= kMaxCandidatePreviews) break;
    }

    // R29m (user redesign, il_2d_multiscale_debug style): when the 30 Hz
    // preview rejects EVERY candidate, fall back to the best OBSERVABLE
    // frontier candidate — cands are sorted by along-progress descending and
    // sampleSideCandidates already guarantees reachable + traversable
    // (0.5 m-inflated) + known-FREE continuation.  Real depth around a large
    // cylinder makes the B-spline clearance / UNKNOWN gates fail right at
    // the obstacle silhouette even though the candidate is genuinely
    // reachable (measured _failed/000006_0ba8dd16: NORMAL_CORRECTION=0 and
    // the corrector spun in TURN_LEFT↔TURN_RIGHT forever).  With the R29j
    // speed-law relaxation the 30 Hz layer can execute such a fly-through
    // detour at vmin, so a rejected preview must not leave the corrector
    // with pure rotation only.
    if (!cands.empty()) {
        const SideCandidate& chosen = cands[0];
        TargetCorrectionDirective fallback = d;
        // Same exact-point contract as the preview loop above: publish the
        // geometry-checked endpoint directly; token quantized from the
        // exact bearing for the class/token fields only.
        fallback.corrected_target_world = chosen.endpoint;
        fallback.corrected_target_world_valid = true;
        fallback.turn_direction_world_valid = false;
        fallback.type = TargetCorrectionType::NORMAL_CORRECTION;
        fallback.reason = "NORMAL_CORRECTION_OBSERVABLE_FRONTIER";
        const Vec2d delta = chosen.endpoint - state.position;
        const double d_norm = delta.norm();
        const Vec2d dir_world =
            d_norm > 1e-9 ? delta / d_norm : Vec2d(1.0, 0.0);
        fallback.decoded_direction_body = rot2(dir_world, -state.yaw);
        fallback.direction_token =
            adapter_.quantizeBearing(chosen.bearing);
        fallback.normalized_distance =
            adapter_.clampNormalizedDistance(d_norm);
        return fallback;
    }
    return d;
}

bool VisibilityTargetCorrector::directiveChanged(
    const TargetCorrectionDirective& a,
    const TargetCorrectionDirective& b) const {
    if (a.type != b.type) return true;
    // A flip between fly-through and terminal-brake semantics is a change
    // even when the world point is identical (R24 terminal_stop).
    if (a.terminal_stop != b.terminal_stop) return true;
    if (a.locked_side != b.locked_side) return true;
    if (a.type == TargetCorrectionType::NORMAL_CORRECTION ||
        a.type == TargetCorrectionType::TURN_LEFT ||
        a.type == TargetCorrectionType::TURN_RIGHT) {
        const double tol = p_.macro_local_target_event_tolerance_m;
        if (a.type == TargetCorrectionType::NORMAL_CORRECTION) {
            if (a.corrected_target_world_valid !=
                b.corrected_target_world_valid) {
                return true;
            }
            if ((a.corrected_target_world - b.corrected_target_world).norm() >
                tol) {
                return true;
            }
        } else {
            if (a.turn_direction_world_valid !=
                b.turn_direction_world_valid) {
                return true;
            }
            if ((a.turn_direction_world - b.turn_direction_world).norm() >
                1e-9) {
                return true;
            }
        }
    }
    return false;
}

// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
//  Public API: reset / bump / update
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
void VisibilityTargetCorrector::reset() {
    correction_active_ = false;
    locked_side_ = SideSelection::NONE;
    update_event_ = 0;
    correction_enter_event_ = 0;
    correction_exit_event_ = 0;
    correction_update_event_ = 0;
    stagnant_update_count_ = 0;
    reentry_success_updates_ = 0;
    search_episode_active_ = false;
    search_swept_rad_ = 0.0;
    last_search_yaw_ = 0.0;
    has_last_search_yaw_ = false;
    brake_latched_ = false;
    brake_world_point_ = Vec2d(0.0, 0.0);
    brake_stationary_updates_ = 0;
    waypoint_execution_fail_updates_ = 0;
    lc_goal_dist_start_ = std::numeric_limits<double>::infinity();
    lc_no_progress_ticks_ = 0;
    last_state_position_ = Vec2d(0.0, 0.0);
    has_last_state_position_ = false;
    last_directive_ = TargetCorrectionDirective{};
    last_obs_ = AvoidanceObservability{};
}

void VisibilityTargetCorrector::resetForNewGoal() {
    correction_active_ = false;
    locked_side_ = SideSelection::NONE;
    stagnant_update_count_ = 0;
    reentry_success_updates_ = 0;
    search_episode_active_ = false;
    search_swept_rad_ = 0.0;
    last_search_yaw_ = 0.0;
    has_last_search_yaw_ = false;
    brake_latched_ = false;
    brake_world_point_ = Vec2d(0.0, 0.0);
    brake_stationary_updates_ = 0;
    waypoint_execution_fail_updates_ = 0;
    lc_goal_dist_start_ = std::numeric_limits<double>::infinity();
    lc_no_progress_ticks_ = 0;
    last_state_position_ = Vec2d(0.0, 0.0);
    has_last_state_position_ = false;
    last_directive_ = TargetCorrectionDirective{};
    last_directive_.update_event = update_event_;
    last_obs_ = AvoidanceObservability{};
}

uint64_t VisibilityTargetCorrector::bumpDirectiveEvent() {
    ++update_event_;
    return update_event_;
}

TargetCorrectionDirective VisibilityTargetCorrector::update(
    const PlanarState& state, const Vec2d& original_goal,
    const LocalObservation& current_patch,
    const LocalObservation& local_history,
    const LocalPlanningAssessment& assessment,
    bool live_directive_usable,
    const DirectiveAssessmentFn& assess_directive) {
    // R29m: monotonic 5 Hz update counter (for the search-rotation
    // cooldown below).
    ++update_count_;
    // The corrector runs at 5 Hz.  Count consecutive updates with negligible
    // motion while a correction is active so a stale target cannot hold the
    // vehicle in a permanent brake/hold loop.
    const bool previous_normal_translation =
        correction_active_ &&
        last_directive_.type == TargetCorrectionType::NORMAL_CORRECTION &&
        last_directive_.normalized_distance > 1e-9;
    if (has_last_state_position_ && previous_normal_translation) {
        const double moved =
            (state.position - last_state_position_).norm();
        // At 5 Hz, twice the configured stationary-speed displacement is a
        // conservative no-progress threshold.  The old 0.12 m/update test
        // misclassified legitimate slow avoidance (~0.5 m/s) as stagnation.
        const double progress_epsilon = std::max(
            0.01, 0.4 * p_.vehicle_stationary_speed_mps);
        if (moved < progress_epsilon) {
            ++stagnant_update_count_;
        } else {
            stagnant_update_count_ = 0;
        }
    } else {
        stagnant_update_count_ = 0;
    }
    last_state_position_ = state.position;
    has_last_state_position_ = true;

    // Search is monotonic on one locked side.  Accumulate actual yaw swept
    // while TURN directives are active; after a half turn without finding a
    // preview-certified translational guide, try the opposite side once
    // instead of oscillating around the original-goal bearing.
    const bool previous_search_phase =
        correction_active_ && search_episode_active_;
    bool switched_search_side = false;
    if (previous_search_phase) {
        if (has_last_search_yaw_) {
            search_swept_rad_ +=
                std::fabs(wrapAngle(state.yaw - last_search_yaw_));
        }
        last_search_yaw_ = state.yaw;
        has_last_search_yaw_ = true;
        if (search_swept_rad_ >= deg2rad(180.0) - 1e-6) {
            locked_side_ = locked_side_ == SideSelection::LEFT
                               ? SideSelection::RIGHT
                               : SideSelection::LEFT;
            search_swept_rad_ = 0.0;
            last_search_yaw_ = state.yaw;
            switched_search_side = true;
        }
    } else {
        search_swept_rad_ = 0.0;
        has_last_search_yaw_ = false;
    }

    // Geometric observability remains diagnostic/search evidence. The
    // authoritative handoff decision is the real local-planner assessment.
    last_obs_ = assessObservability(state, original_goal, local_history);
    const AvoidanceObservability fresh_obs =
        assessObservability(state, original_goal, current_patch);
    last_obs_.goal_inside_fov = fresh_obs.goal_inside_fov;
    last_obs_.direct_corridor_blocked =
        fresh_obs.direct_corridor_blocked;
    last_obs_.blocker_observed = fresh_obs.blocker_observed;
    last_obs_.blocker_min_along = fresh_obs.blocker_min_along;
    last_obs_.left_bypass_observable = fresh_obs.left_bypass_observable;
    last_obs_.right_bypass_observable = fresh_obs.right_bypass_observable;
    last_obs_.fov_boundary_truncated = fresh_obs.fov_boundary_truncated;
    last_obs_.unknown_occluded = fresh_obs.unknown_occluded;
    last_obs_.local_avoidance_observable =
        assessment.translation_plan_valid;
    last_obs_.reason = fresh_obs.reason;
    last_obs_.left_score = fresh_obs.left_score;
    last_obs_.right_score = fresh_obs.right_score;

    // Outside recovery, yaw-first capability is sufficient to pass the
    // original goal to the local layer.  During recovery, however, TURNING
    // only proves that the target can be looked at.  Release the corrected
    // target solely after three consecutive 5 Hz previews prove a genuine
    // translational path to the original target.
    constexpr uint32_t kReentrySuccessUpdates = 3;
    if (correction_active_) {
        if (assessment.translation_plan_valid) {
            reentry_success_updates_ = std::min(
                reentry_success_updates_ + 1, kReentrySuccessUpdates);
        } else {
            reentry_success_updates_ = 0;
        }
    } else {
        reentry_success_updates_ = 0;
    }
    const bool direct_local_available =
        assessment.translation_plan_valid || assessment.rotation_available;
    const double original_goal_distance =
        (original_goal - state.position).norm();
    // R26: macro-level limit-cycle watchdog on the ORIGINAL goal distance.
    // The local detector watches the current EFFECTIVE target, but the
    // macro switches goal<->TURN, so each switch resets the local bearing
    // evidence (measured: task 65 PASS/TURN_RIGHT loop at 0.48 m for ~36 s
    // with 2458° of yaw).  Whenever the original-goal distance decreases by
    // >= macro_limit_cycle_goal_progress_m the window resets; after
    // macro_limit_cycle_window_5hz updates without such progress the macro
    // treats the episode as cycling and forces a fresh handoff to local.
    if (lc_goal_dist_start_ - original_goal_distance >=
        p_.macro_limit_cycle_goal_progress_m) {
        lc_goal_dist_start_ = original_goal_distance;
        lc_no_progress_ticks_ = 0;
    } else if (std::isfinite(lc_goal_dist_start_)) {
        ++lc_no_progress_ticks_;
    } else {
        lc_goal_dist_start_ = original_goal_distance;
        lc_no_progress_ticks_ = 0;
    }
    const bool macro_limit_cycle =
        lc_no_progress_ticks_ >=
        static_cast<uint64_t>(std::max(1, p_.macro_limit_cycle_window_5hz));
    // Terminal capture zone: inside macro_terminal_capture_radius_m of the
    // original goal the macro must NEVER issue a locked-side search TURN.
    // As soon as the HARD corridor is clear (and local can rotate or the
    // goal is inside the FOV), hand the target back to local so it
    // micro-approaches / rotates toward the goal.  Only a genuine HARD
    // corridor block may keep the macro engaged.  (task 65: the macro
    // issued TURN_RIGHT at 0.48 m with the goal in FOV and a clear
    // corridor — the 1 m risk corridor was treated as a hard block.)
    const bool terminal_capture_zone =
        original_goal_distance <= p_.macro_terminal_capture_radius_m;
    const bool local_can_take_goal =
        assessment.rotation_available || !assessment.target_outside_fov ||
        assessment.translation_plan_valid;
    const bool terminal_capture_lock =
        (terminal_capture_zone || macro_limit_cycle) &&
        !assessment.local_corridor_blocked && local_can_take_goal;
    // Final approach has higher priority than macro recovery.  When the
    // original goal is visible, near and geometrically unobstructed, keep
    // the original terminal target even if a cold-start preview temporarily
    // cannot certify the current-speed stopping trajectory.
    //
    // R25 (measured deadlock joint_v2_000004_4ab1e354 / task 491): the
    // near-goal capture lock is an INITIAL priority for the LOCAL layer,
    // not a permanent veto on macro recovery.  It must never fire while a
    // correction is ACTIVE (it revoked a working bypass waypoint the
    // instant the goal crossed 4.5 m, leaving PASS_THROUGH + local BLOCKED
    // for ~50 s), and it must not block a CONFIRMED takeover: when the
    // local planner is persistently failing, the 5 Hz layer must be allowed
    // to re-issue a temporary target.  During correction only
    // recovery_reentry_confirmed (3 consecutive translational previews)
    // releases the corrected target.
    const bool goal_capture_lock =
        !correction_active_ &&
        original_goal_distance <= adapter_.normalMaxDistanceM() + 1e-9 &&
        fresh_obs.goal_inside_fov &&
        !fresh_obs.direct_corridor_blocked &&
        !assessment.takeover_confirmed;
    const bool recovery_reentry_confirmed =
        correction_active_ &&
        reentry_success_updates_ >= kReentrySuccessUpdates;
    // R24 (legacy): when the goal was OUTSIDE the FOV with a clear corridor,
    // the local planner used to rotate to re-acquire it, so the macro handed
    // PASS_THROUGH back to local.  R29 removed local self-rotation: for an
    // out-of-FOV goal the local always reports NO_SAFE_CANDIDATE and
    // rotation_available is false, so this branch no longer fires — the
    // macro stays in correction and issues a goal-toward SEARCH_ROTATION
    // TURN below instead.
    const bool reacquire_original_goal =
        correction_active_ &&
        assessment.target_outside_fov &&
        !fresh_obs.direct_corridor_blocked &&
        assessment.rotation_available;
    if (goal_capture_lock ||
        (!correction_active_ && direct_local_available) ||
        recovery_reentry_confirmed ||
        reacquire_original_goal ||
        terminal_capture_lock) {
        TargetCorrectionDirective d;
        d.valid = true;
        d.type = TargetCorrectionType::PASS_THROUGH;
        d.locked_side = SideSelection::NONE;
        const bool leaving_correction = correction_active_;
        d.reason = goal_capture_lock
                       ? "ORIGINAL_GOAL_CAPTURE_LOCK"
                       : (terminal_capture_lock
                              ? (terminal_capture_zone
                                     ? "TERMINAL_CAPTURE_ZONE_LOCAL_APPROACH"
                                     : "MACRO_LIMIT_CYCLE_FORCE_LOCAL")
                              : (reacquire_original_goal
                                     ? "REACQUIRE_ORIGINAL_GOAL_LOCAL_ROTATE"
                                     : (assessment.rotation_available &&
                                        !assessment.translation_plan_valid
                                           ? "LOCAL_CAN_ROTATE_TO_ORIGINAL_TARGET"
                                           : (leaving_correction
                                                  ? "ORIGINAL_TRANSLATION_REENTRY_CONFIRMED"
                                                  : "LOCAL_CAN_PLAN_ORIGINAL_TARGET"))));

        const bool changed =
            leaving_correction ||
            last_directive_.type != TargetCorrectionType::PASS_THROUGH;
        if (leaving_correction) ++correction_exit_event_;
        correction_active_ = false;
        locked_side_ = SideSelection::NONE;
        stagnant_update_count_ = 0;
        reentry_success_updates_ = 0;
        search_episode_active_ = false;
        search_swept_rad_ = 0.0;
        has_last_search_yaw_ = false;
        // R24: a latched brake belongs to the correction episode only; it
        // must never leak into the next episode via a stale world point.
        brake_latched_ = false;
        brake_stationary_updates_ = 0;
        waypoint_execution_fail_updates_ = 0;
        if (changed) ++update_event_;
        d.update_event = update_event_;
        last_directive_ = d;
        return d;
    }

    // A preview miss does not authorize an upper-layer target change.  Wait
    // until the actually executing 30 Hz planner has reported continuous
    // failure for the configured confirmation window.  During this window
    // PASS_THROUGH lets local brake, slow down or retry the same goal.
    //
    // R24: the LOCAL planner's own corridor verdict is authoritative macro
    // topology evidence.  The geometric extractBlocker lateral-span gate
    // deliberately ignores small/single obstacles, but the 30 Hz A* (with
    // its 0.5 m inflation) can be genuinely blocked by exactly those small
    // obstacles.  Relying only on the geometric gate created a
    // responsibility vacuum: the upper layer said "small obstacle → local
    // can handle it" while the local planner itself continuously reported
    // it could not (joint_v2_000001_469baa3b: 108/122 locally-BLOCKED
    // frames had no geometric corridor blocker; 133/141 for
    // joint_v2_000000_a411ef5a).
    //
    // R29 (single-mode expert): the local never rotates by itself, so a
    // goal outside the FOV is unambiguous "local cannot proceed" evidence
    // even when the corridor is clear — the macro must issue a goal-toward
    // TURN to re-acquire it (otherwise local NO_SAFE_CANDIDATE + macro
    // PASS_THROUGH deadlock, measured joint_v2_000000).
    // R29i: a nose-blocked hard stop (too close to an obstacle) is takeover
    // evidence too — without it the FSM confirms takeover but the macro
    // still PASS_THROUGHs (its own topology gate below is false) and the
    // drone parks beside the blocker forever.
    const bool macro_topology_evidence =
        !current_patch.valid() || fresh_obs.direct_corridor_blocked ||
        fresh_obs.unknown_occluded || fresh_obs.fov_boundary_truncated ||
        assessment.local_corridor_blocked ||
        assessment.nose_blocked_stop ||
        assessment.target_outside_fov;
    if (!correction_active_ &&
        (!assessment.takeover_confirmed || !macro_topology_evidence)) {
        TargetCorrectionDirective d;
        d.valid = true;
        d.type = TargetCorrectionType::PASS_THROUGH;
        d.locked_side = SideSelection::NONE;
        d.reason = !assessment.takeover_confirmed
                       ? "LOCAL_TAKEOVER_PENDING_CONFIRMED_FAILURE"
                       : "LOCAL_FAILURE_WITHOUT_MACRO_TOPOLOGY_EVIDENCE";
        const bool changed =
            last_directive_.type != TargetCorrectionType::PASS_THROUGH;
        if (changed) ++update_event_;
        d.update_event = update_event_;
        last_directive_ = d;
        return d;
    }

    // 2/3) The original target is not locally plannable. Lock a bypass
    // side for this episode, then publish either an in-FOV temporary target
    // or a bounded pure-rotation search step when route evidence is weak.
    const bool entering = !correction_active_;
    if (entering) {
        locked_side_ = selectSide(state, local_history, original_goal);
        if (locked_side_ == SideSelection::NONE) {
            locked_side_ = selectSide(state, current_patch, original_goal);
        }
        if (locked_side_ == SideSelection::NONE) {
            locked_side_ = SideSelection::RIGHT;
        }
        correction_active_ = true;
        ++correction_enter_event_;
        stagnant_update_count_ = 0;
        reentry_success_updates_ = 0;
        search_episode_active_ = false;
        search_swept_rad_ = 0.0;
        has_last_search_yaw_ = false;
        // R24: never carry a latched brake / stationary count into a new
        // correction episode.
        brake_latched_ = false;
        brake_stationary_updates_ = 0;
        waypoint_execution_fail_updates_ = 0;
    }

    // If the current patch already contains the blocker, build the guide from
    // that patch.  History may still be used when the current view is clear,
    // but it must not override fresh blocker/bypass evidence with an older
    // straight frontier.
    const LocalObservation& planning_observation =
        fresh_obs.direct_corridor_blocked
            ? current_patch
            : (local_history.valid() ? local_history : current_patch);
    const LocalFreeGrid grid =
        buildLocalFreeGrid(state, planning_observation);

    // R25 (Fix #6): a single cold preview miss must not discard a waypoint
    // that is still actually executing.  Track consecutive 5 Hz updates in
    // which the held waypoint is neither reached nor executing; only after
    // >= 3 such updates may makeCorrectionDirective drop it.
    {
        const bool held_waypoint =
            last_directive_.type == TargetCorrectionType::NORMAL_CORRECTION &&
            last_directive_.normalized_distance > 1e-9 &&
            last_directive_.corrected_target_world_valid;
        if (held_waypoint) {
            const Vec2d pd =
                last_directive_.corrected_target_world - state.position;
            if (pd.norm() > waypointReachedTolerance()) {
                waypoint_execution_fail_updates_ =
                    live_directive_usable
                        ? 0
                        : waypoint_execution_fail_updates_ + 1;
            } else {
                waypoint_execution_fail_updates_ = 0;
            }
        } else {
            waypoint_execution_fail_updates_ = 0;
        }
    }
    const bool drop_held_allowed =
        waypoint_execution_fail_updates_ >= 3;
    TargetCorrectionDirective d = makeCorrectionDirective(
        state, original_goal, planning_observation, grid, locked_side_,
        live_directive_usable, drop_held_allowed, assess_directive);
    d.locked_side = locked_side_;

    // R24: a pure-rotation search step must first serve "re-acquire the
    // original goal".  When the direct corridor has NO blocker evidence,
    // turn TOWARD the original goal (bring it back into the FOV) instead
    // of blindly following the locked bypass side — the locked side is
    // only a bypass aid while a blocker is actually present.
    // (joint_v2_000001_469baa3b @22.4 s: goal 5.6° outside the LEFT FOV,
    // no corridor blocker; the expert rotated RIGHT along the locked side
    // and pushed the goal to the tail.)
    //
    // R29k (measured _failed/000006_02cac96b): re-acquiring the original
    // goal is only useful when that bearing is actually TRAVERSABLE.  With
    // the drone beside a large cylinder the depth-derived
    // direct_corridor_blocked flips 0/1 as the nose sweeps, and every
    // "no blocker" frame re-issued SEARCH_ROTATION_TOWARD_ORIGINAL_GOAL,
    // pulling the nose back toward the occluded goal — TURN_LEFT↔TURN_RIGHT
    // every ~1 s, zero displacement, goal_no_progress.  Gate the
    // re-acquisition on a continuous FREE run of >=
    // macro_goal_direction_min_range_m along the goal bearing; when it is
    // blocked / occluded keep the LOCKED bypass side (the swept-angle
    // logic flips sides only after ~180° of search).  large_short still
    // re-acquires because its goal bearing (up over the cylinder) has a
    // long FREE run.
    // R29m: SEARCH_ROTATION_TOWARD_ORIGINAL_GOAL cooldown.  Once the
    // corrector has pulled the nose toward the (possibly occluded) goal, it
    // must not immediately re-pull when depth evidence flips — that is the
    // TURN_LEFT↔TURN_RIGHT oscillation (_failed/000006_0ba8dd16).  During
    // the cooldown the corrector keeps the LOCKED bypass side.
    const bool sr_cooldown_ok =
        last_search_rotation_update_ == 0 ||
        (update_count_ - last_search_rotation_update_) >=
            static_cast<uint64_t>(p_.macro_search_rotation_cooldown_5hz);
    {
        const bool proposed_search_turn_rd =
            d.type == TargetCorrectionType::TURN_LEFT ||
            d.type == TargetCorrectionType::TURN_RIGHT;
        if (proposed_search_turn_rd && sr_cooldown_ok &&
            !fresh_obs.direct_corridor_blocked) {
            const Vec2d to_goal = original_goal - state.position;
            const double b_goal = wrapAngle(
                std::atan2(to_goal.y(), to_goal.x()) - state.yaw);
            const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
            const bool goal_left = b_goal > fov_half + 1e-9;
            const bool goal_right = b_goal < -(fov_half + 1e-9);
            if (goal_left || goal_right) {
                // R29k: only re-acquire when the goal bearing is not
                // OBSERVED-OCCUPIED within the minimum range.  We check
                // OCCUPIED, not FREE: a bearing outside the FOV is UNKNOWN
                // (passable) and must not suppress re-acquisition
                // (large_short needs it), while a blocker directly on the
                // goal bearing keeps the locked bypass side instead
                // (_failed/000006: pulling the nose back to an occluded
                // goal yaws the drone back and forth with zero progress).
                const double goal_occ_range = occupiedRangeAlongFrom(
                    current_patch, state.position,
                    state.yaw + b_goal, p_.obs_range_m);
                if (goal_occ_range <
                    p_.macro_goal_direction_min_range_m) {
                    // Goal bearing blocked → keep the locked bypass side
                    // (leave d as makeCorrectionDirective produced).
                } else {
                    const int n = std::max(3, p_.te_direction_bin_count);
                    d.type = goal_left ? TargetCorrectionType::TURN_LEFT
                                       : TargetCorrectionType::TURN_RIGHT;
                    d.direction_token = goal_left ? 0 : n + 1;
                    d.decoded_direction_body =
                        adapter_.decodeDirectionToken(d.direction_token);
                    d.normalized_distance = 1.0;
                    d.corrected_target_world_valid = false;
                    d.turn_direction_world =
                        rot2(d.decoded_direction_body, state.yaw).normalized();
                    d.turn_direction_world_valid = true;
                    d.reason = "SEARCH_ROTATION_TOWARD_ORIGINAL_GOAL";
                    // R29m: start the cooldown so the next re-pull waits.
                    last_search_rotation_update_ = update_count_;
                }
            }
        }
    }

    // Candidate generation already falls back to pure rotation when current
    // plus causal-history information cannot produce a certified guide.  The
    // watchdog handles the remaining case where a nominal guide is feasible
    // on paper but repeatedly produces no translation.
    const uint32_t stagnation_limit = static_cast<uint32_t>(std::max(
        3, p_.macro_unknown_recovery_threshold_ticks / 6));
    const bool progress_watchdog =
        stagnant_update_count_ >= stagnation_limit;
    // Geometry is diagnostic here and must not veto a preview-certified
    // target from the other planner.
    if (progress_watchdog &&
        d.type == TargetCorrectionType::NORMAL_CORRECTION) {
        const int n = std::max(3, p_.te_direction_bin_count);
        const bool left = locked_side_ == SideSelection::LEFT;
        d.type = left ? TargetCorrectionType::TURN_LEFT
                      : TargetCorrectionType::TURN_RIGHT;
        d.direction_token = left ? 0 : n + 1;
        d.decoded_direction_body = adapter_.decodeDirectionToken(
            d.direction_token);
        d.normalized_distance = 1.0;
        d.corrected_target_world_valid = false;
        d.turn_direction_world =
            rot2(d.decoded_direction_body, state.yaw).normalized();
        d.turn_direction_world_valid = true;
        d.reason = "LOCAL_BLOCKED_SEARCH_ROTATION_NO_PROGRESS";
    }

    // A distance==1 directive is a strict pure-rotation label.  R28h
    // (P0#5): the 30 Hz planner ALREADY brakes before rotating for a
    // pure-rotation target (local_planner_30hz rotationBrakeRequired ->
    // brakeBeforeRotation), so we NO LONGER issue a zero-distance
    // "stop at current position" NORMAL_CORRECTION first.  That label
    // (effective_target = current position, distance_norm = 0,
    // terminal_stop = true) conflicted with the goal-distance contract and
    // forced a full stop before every search rotation — task 440: ~0.3 s
    // after one NO_SAFE the corrector published a NORMAL_CORRECTION stop at
    // the vehicle pose, turning a transient local replan blip into an
    // upper-layer brake event.  The world-latched TURN is published
    // directly; the local layer brakes, then rotates, and the label is a
    // clean turn_to_target.  (R24's brake-point latching is obsolete here —
    // there is no brake point; the local's own brake-before-rotation owns
    // the stop, and the TURN anchor is direction-latched, not position-
    // latched, so the "moving target" bug cannot reappear.)
    // Keep the (always-false) flag so the shared reason / search-phase
    // bookkeeping below still compiles; no brake-before-search phase exists.
    const bool brake_before_search = false;

    // A TURN is a bounded world-latched step. Keep its anchor until it
    // enters the FOV; only then may the 5 Hz layer issue another step.
    const bool previous_turn =
        last_directive_.type == TargetCorrectionType::TURN_LEFT ||
        last_directive_.type == TargetCorrectionType::TURN_RIGHT;
    const bool proposed_turn =
        d.type == TargetCorrectionType::TURN_LEFT ||
        d.type == TargetCorrectionType::TURN_RIGHT;
    if (!entering && !switched_search_side && previous_turn && proposed_turn &&
        last_directive_.turn_direction_world_valid) {
        const Vec2d direction = last_directive_.turn_direction_world;
        if (direction.squaredNorm() > 1e-12) {
            const double bearing = wrapAngle(
                std::atan2(direction.y(), direction.x()) - state.yaw);
            const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
            if (std::fabs(bearing) > fov_half + 1e-9) {
                d = last_directive_;
                d.reason = "SEARCH_TURN_STEP_PENDING";
            }
        }
    }

    if (d.type == TargetCorrectionType::NORMAL_CORRECTION) {
        if (brake_before_search) {
            d.reason = "BRAKE_BEFORE_SEARCH_ROTATION";
        } else {
            d.reason = entering
                           ? "LOCAL_BLOCKED_CORRECTION_ENTER_PREVIEW_CERTIFIED"
                           : "LOCAL_BLOCKED_CORRECTION_UPDATE_PREVIEW_CERTIFIED";
        }
    } else if (d.reason != "SEARCH_TURN_STEP_PENDING" &&
               d.reason != "LOCAL_BLOCKED_SEARCH_ROTATION_NO_PROGRESS" &&
               d.reason != "SEARCH_ROTATION_TOWARD_ORIGINAL_GOAL") {
        d.reason = entering ? "LOCAL_BLOCKED_SEARCH_ROTATION_ENTER"
                            : "LOCAL_BLOCKED_SEARCH_ROTATION_UPDATE";
    }

    const bool changed = entering || directiveChanged(last_directive_, d);
    if (changed) {
        ++update_event_;
        if (!entering) ++correction_update_event_;
    } else if (!brake_before_search) {
        const std::string reason = d.reason;
        d = last_directive_;
        d.reason = reason;
    }
    d.valid = true;
    d.update_event = update_event_;
    last_directive_ = d;
    const bool next_search_phase = proposed_turn || brake_before_search;
    search_episode_active_ = next_search_phase;
    if (next_search_phase) {
        if (!has_last_search_yaw_) {
            last_search_yaw_ = state.yaw;
            has_last_search_yaw_ = true;
        }
    } else {
        search_swept_rad_ = 0.0;
        has_last_search_yaw_ = false;
    }
    return d;
}

}  // namespace expert
}  // namespace il_dataset
