#include "il_dataset/hierarchical_expert/macro_expert_5hz.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
                             /*strict=*/true, axis_dir,
                             p_.macro_guide_horizon_m);
    const std::vector<SideCandidate> right =
        sampleSideCandidates(state, goal, patch, grid, has_blocker,
                             blocker_min_along, SideSelection::RIGHT,
                             /*strict=*/true, axis_dir,
                             p_.macro_guide_horizon_m);
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
    bool strict, const Vec2d& axis, double max_distance_m) const {
    std::vector<SideCandidate> out;
    if (!patch.valid() || !grid.valid()) return out;

    const double fov_half = deg2rad(p_.obs_fov_deg) / 2.0;
    const double margin = deg2rad(p_.te_turn_ray_margin_deg);
    const double b_lo = -fov_half + margin;
    const double b_hi = fov_half - margin;
    if (!(b_hi > b_lo)) return out;

    const Vec2d to_goal = goal - state.position;
    const double goal_dist = to_goal.norm();
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
        std::min(dmax, std::max(0.0, max_distance_m));
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

    // Two-pass collection: (1) STRICT progress — every candidate must move
    // measurably closer to the goal (the original semantic, keeps the drone
    // goal-bound); (2) RELAXED retreat — only when pass (1) yields nothing
    // (a dense cluster forces lateral motion that temporarily increases the
    // goal distance), allow a candidate up to max_retreat_m beyond the
    // current goal distance so the corrector can still ISSUE a translational
    // guide instead of stranding in a pure-rotation search forever.
    auto collect = [&](bool relaxed) {
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
                // 绕大障碍的第一步是沿障碍表面切向的,相对目标轴可能带负
                // 分量(暂时"后退")。strict 模式仍强制沿轴前进;仅 relaxed
                // 回退允许负 along,且不得超出 max_retreat(否则候选会无限
                // 远离原目标,永远桥接不出绕过障碍的推进)。
                if (!relaxed ||
                    along < -p_.macro_observable_frontier_max_retreat_m) {
                    continue;
                }
            }
            // A temporary waypoint must make measurable progress toward the
            // original mission goal.  Mere local reachability is not enough:
            // an indefinitely reachable side ray previously carried the
            // vehicle from 7.2 m to 14.3 m away from the goal.
            const double endpoint_goal_dist = (goal - endpoint).norm();
            if (relaxed) {
                // Fallback pass: allow a bounded temporary retreat, never
                // beyond max_retreat_m farther than the current distance.
                if (endpoint_goal_dist >
                    goal_dist + p_.macro_observable_frontier_max_retreat_m) {
                    continue;
                }
            } else {
                if (endpoint_goal_dist >
                    goal_dist - p_.macro_observable_frontier_min_progress_m) {
                    continue;
                }
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
            c.cont = cont;
            c.certified = true;
            out.push_back(c);
        }
    }
    };  // end collect(lambda)
    if (strict) {
        collect(false);   // strict certification: goal progress required
    } else {
        collect(true);    // BYPASS: all bounded-retreat candidates, uniform
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

    const Vec2d to_goal = goal - state.position;
    const double goal_dist = to_goal.norm();
    const Vec2d axis =
        goal_dist > 1e-9 ? to_goal / goal_dist : Vec2d(1.0, 0.0);

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
                SideSelection::LEFT, /*strict=*/true, axis,
                p_.macro_guide_horizon_m).empty();
            const bool right_ok = !sampleSideCandidates(
                state, goal, patch, grid, has_blocker, blocker_min_along,
                SideSelection::RIGHT, /*strict=*/true, axis,
                p_.macro_guide_horizon_m).empty();
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
        // The paired-ray window is centred on the ORIGINAL-GOAL bearing:
        // when the goal is outside the FOV the window falls completely
        // outside it and no evidence is gathered (pairs==0), so the old
        // code degraded straight to a goal-side guess.  But the local map
        // already knows which side has the free space (the drone just flew
        // in from there / a wall spans one side).  Re-scan the evidence
        // rays centred on the NOSE instead and prefer the more open side
        // (measured: cylinder-wall test t=240 goal 135 deg out of FOV and
        // the LEFT side is the wall; nose-centred scan favours RIGHT).
        double nose_left = 0.0, nose_right = 0.0;
        int nose_pairs = 0;
        for (double db = d_beta; db <= fov / 2.0 - 1e-6; db += d_beta) {
            const double bl = db;     // LEFT of the nose
            const double br = -db;    // RIGHT of the nose
            nose_left += freeRangeAlong(state, patch, bl);
            nose_right += freeRangeAlong(state, patch, br);
            ++nose_pairs;
        }
        if (nose_pairs > 0) {
            const double m = p_.macro_side_evidence_margin;
            const double nla = nose_left / nose_pairs;
            const double nra = nose_right / nose_pairs;
            if (nla > nra + m) return SideSelection::LEFT;
            if (nra > nla + m) return SideSelection::RIGHT;
        }
        // Still ambiguous: prefer RIGHT (user design: tie -> right).
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
    // Ambiguous free-range evidence: prefer RIGHT (user design: tie -> right).
    return SideSelection::RIGHT;  // goal exactly dead ahead 鈫?fixed RIGHT
}

// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
//  Passable angular frontier on the committed side
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
double VisibilityTargetCorrector::sideFrontierBearing(
    const PlanarState& state, const LocalObservation& patch,
    SideSelection side) const {
    if (!patch.valid()) return std::numeric_limits<double>::quiet_NaN();
    const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
    const double margin = deg2rad(p_.te_turn_ray_margin_deg);
    const double min_free = p_.macro_observable_unknown_margin_cells *
                            patch.resolution;
    const double step_b = deg2rad(p_.macro_local_candidate_bearing_step_deg);
    const int dir = side == SideSelection::LEFT ? 1 : -1;
    // Deepest sideward bearing that still observes a known-FREE ray of at
    // least min_free.  Scan from the nose outward to the side FOV edge and
    // keep the LAST (deepest) passable bearing.
    double best = std::numeric_limits<double>::quiet_NaN();
    const double b_start =
        side == SideSelection::LEFT ? -fov_half + margin : fov_half - margin;
    const double b_end =
        side == SideSelection::LEFT ? fov_half - margin : -fov_half + margin;
    for (double b = b_start; ; b += dir * step_b) {
        if (dir > 0 && b > b_end + 1e-9) break;
        if (dir < 0 && b < b_end - 1e-9) break;
        if (freeRangeAlong(state, patch, b) >= min_free) {
            best = b;
        }
    }
    return best;
}

// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
//  Correction directive construction
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
// ─────────────────────────────────────────────────────────────────────
//  Correction directive construction
// ─────────────────────────────────────────────────────────────────────
TargetCorrectionDirective VisibilityTargetCorrector::makeCorrectionDirective(
    const PlanarState& state, const Vec2d& goal,
    const LocalObservation& patch, const LocalFreeGrid& grid,
    SideSelection side, bool live_directive_usable,
    bool drop_held_waypoint_allowed,
    const DirectiveAssessmentFn& assess_directive) const {
    // A BYPASS correction is a SHORT observation hop, not a full bypass
    // plan: [1.5, 2.5] m.  The drone hops, observes, and re-decides.
    constexpr double kMinCorrectionDistance = 1.5;
    constexpr double kMaxCorrectionDistance = 2.5;

    // Live goal axis.  The committed side is a GUIDANCE constraint, not a
    // fixed motion direction: every regeneration re-derives the side
    // geometry from the CURRENT depth and the live goal bearing.
    const Vec2d to_goal = goal - state.position;
    const double goal_dist = std::max(1e-6, to_goal.norm());
    const Vec2d axis = to_goal / goal_dist;

    // Fresh PRIMARY blocker on the CURRENT patch (never a stale, locked
    // entry frame): nearest surface + lateral edges.
    double blocker_min_along = std::numeric_limits<double>::infinity();
    double blocker_lat_min = std::numeric_limits<double>::infinity();
    double blocker_lat_max = -std::numeric_limits<double>::infinity();
    const bool has_blocker =
        extractBlocker(state, goal, patch, blocker_min_along,
                       blocker_lat_min, blocker_lat_max);

    // Default directive: a world-latched TURN pointing the nose at the
    // side's passable angular frontier (where new bypass space appears).
    const int n = std::max(3, p_.te_direction_bin_count);
    TargetCorrectionDirective d;
    d.valid = true;
    d.locked_side = side;
    d.type = side == SideSelection::LEFT
                 ? TargetCorrectionType::TURN_LEFT
                 : TargetCorrectionType::TURN_RIGHT;
    d.direction_token = side == SideSelection::LEFT ? 0 : n + 1;
    const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
    const double margin = deg2rad(p_.te_turn_ray_margin_deg);
    double frontier_b = sideFrontierBearing(state, patch, side);
    if (std::isnan(frontier_b)) {
        frontier_b = side == SideSelection::LEFT ? fov_half - margin
                                                 : -(fov_half - margin);
    }
    d.decoded_direction_body =
        Vec2d(std::cos(frontier_b), std::sin(frontier_b));
    d.normalized_distance = 1.0;  // TURN classes: EXACT 1.0
    d.turn_direction_world =
        rot2(d.decoded_direction_body, state.yaw).normalized();
    d.turn_direction_world_valid = true;
    d.reason = side == SideSelection::LEFT ? "TURN_LEFT_NO_BYPASS_FRONTIER"
                                           : "TURN_RIGHT_NO_BYPASS_FRONTIER";

    std::vector<SideCandidate> cands =
        sampleSideCandidates(state, goal, patch, grid, has_blocker,
                             blocker_min_along, side, /*strict=*/false,
                             axis, kMaxCorrectionDistance);

    // ── NEW-VISIBILITY ranking.  BYPASS no longer asks "which waypoint
    //    scores highest on side/goal progress"; it asks "which currently
    //    visible, locally-executable SHORT target reveals the most new
    //    bypass visibility".  Primary signal: the known-FREE continuation
    //    beyond the endpoint (more = more new map revealed).  Secondary: a
    //    modest lateral depth into the committed side (rounds the blocker
    //    corner).  When the direct corridor is no longer blocked, free space
    //    has wrapped back toward the goal: converge by closing goal distance.
    // ─────────────────────────────────────────────────────────────────────
    const double lat_sign = side == SideSelection::LEFT ? 1.0 : -1.0;
    const double w_cont = 1.0;   // free continuation (m) — information gain
    const double w_lat = 0.4;    // lateral depth into the side (m)
    const double w_wrap = 1.0;   // goal convergence (only when corridor clear)
    const double w_turn = 0.1;   // bearing magnitude penalty (rad)
    const auto scoreOf = [&](const SideCandidate& c) {
        const Vec2d rel = c.endpoint - state.position;
        const double lat = lat_sign * cross2(axis, rel);
        double s = w_cont * c.cont + w_lat * lat;
        if (!has_blocker) {
            s += w_wrap *
                 std::max(0.0, goal_dist - (goal - c.endpoint).norm());
        }
        s -= w_turn * std::fabs(c.bearing);
        return s;
    };
    std::vector<std::pair<double, const SideCandidate*>> ranked;
    ranked.reserve(cands.size());
    for (const SideCandidate& c : cands) {
        if (c.dist > kMaxCorrectionDistance + 1e-9) continue;
        if (c.dist < kMinCorrectionDistance - 1e-9) continue;
        ranked.emplace_back(scoreOf(c), &c);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const std::pair<double, const SideCandidate*>& a,
                 const std::pair<double, const SideCandidate*>& b) {
                  return a.first > b.first;
              });
    const double best_fresh_score =
        ranked.empty() ? -std::numeric_limits<double>::infinity()
                       : ranked.front().first;

    // ── Continuous 5 Hz re-evaluation of the HELD waypoint.  The previous
    //    waypoint is NOT locked until arrival: it participates as one more
    //    candidate scored on the CURRENT depth, with a small hysteresis
    //    bonus (prevents per-tick churn).  It is kept only while it is still
    //    the best available point AND locally feasible; a clearly better
    //    fresh point replaces it IMMEDIATELY (no waiting for arrival). ─────
    constexpr double kHoldHysteresisM = 0.2;
    const bool have_held =
        last_directive_.type == TargetCorrectionType::NORMAL_CORRECTION &&
        last_directive_.normalized_distance > 1e-9 &&
        last_directive_.corrected_target_world_valid &&
        !drop_held_waypoint_allowed;
    TargetCorrectionDirective held;
    bool held_feasible = false;
    double held_score = -std::numeric_limits<double>::infinity();
    if (have_held) {
        const Vec2d previous_delta =
            last_directive_.corrected_target_world - state.position;
        const double d_held = previous_delta.norm();
        if (d_held > waypointReachedTolerance()) {
            const double bearing = wrapAngle(
                std::atan2(previous_delta.y(), previous_delta.x()) -
                state.yaw);
            SideCandidate held_cand;
            held_cand.endpoint = last_directive_.corrected_target_world;
            held_cand.bearing = bearing;
            held_cand.dist = d_held;
            held_cand.cont = freeRangeAlongFrom(
                patch, last_directive_.corrected_target_world,
                state.yaw + bearing);
            held_score = scoreOf(held_cand) + kHoldHysteresisM;

            held = last_directive_;
            held.direction_token = adapter_.quantizeBearing(bearing);
            held.decoded_direction_body =
                adapter_.decodeDirectionToken(held.direction_token);
            held.normalized_distance =
                adapter_.clampNormalizedDistance(d_held);
            held.turn_direction_world_valid = false;
            held.locked_side = side;
            held.reason = "FIXED_WAYPOINT_HELD";
            const LocalPlanningAssessment held_assessment =
                assess_directive(held);
            held_feasible = held_assessment.translation_plan_valid ||
                            held_assessment.rotation_available ||
                            live_directive_usable;
        }
    }
    // Keep the held waypoint only while it is still the best AND feasible.
    if (have_held && held_feasible && held_score >= best_fresh_score) {
        return held;
    }

    std::vector<uint8_t> previewed_tokens(
        static_cast<size_t>(std::max(3, p_.te_direction_bin_count) + 2), 0);
    size_t preview_count = 0;
    constexpr size_t kMaxCandidatePreviews = 16;
    for (const auto& entry : ranked) {
        const SideCandidate& chosen = *entry.second;
        TargetCorrectionDirective candidate = d;
        const int token = adapter_.quantizeBearing(chosen.bearing);
        if (previewed_tokens[static_cast<size_t>(token)] != 0) continue;
        previewed_tokens[static_cast<size_t>(token)] = 1;
        // Publish the EXACT geometry-checked world point — no quantization
        // re-projection.  The 5 Hz student regresses the continuous FLU
        // direction re-derived live from corrected_target_world.
        candidate.corrected_target_world = chosen.endpoint;
        candidate.corrected_target_world_valid = true;
        candidate.turn_direction_world_valid = false;
        candidate.type = TargetCorrectionType::NORMAL_CORRECTION;
        candidate.reason = "BYPASS_CORRECTION_PREVIEW_CERTIFIED";
        const Vec2d delta = chosen.endpoint - state.position;
        const double d_norm = delta.norm();
        const Vec2d dir_world =
            d_norm > 1e-9 ? delta / d_norm : Vec2d(1.0, 0.0);
        candidate.decoded_direction_body = rot2(dir_world, -state.yaw);
        candidate.direction_token = token;
        candidate.normalized_distance =
            adapter_.clampNormalizedDistance(d_norm);

        ++preview_count;
        const LocalPlanningAssessment candidate_assessment =
            assess_directive(candidate);
        if (candidate_assessment.translation_plan_valid) {
            return candidate;
        }
        if (preview_count >= kMaxCandidatePreviews) break;
    }
    // No fresh candidate is executable: keep the still-feasible held
    // waypoint rather than dropping to a TURN (it is already executing or
    // reachable).
    if (have_held && held_feasible) {
        return held;
    }
    return d;  // TURN toward the bypass frontier (no feasible correction).
}

bool VisibilityTargetCorrector::directiveChanged(
    const TargetCorrectionDirective& a,
    const TargetCorrectionDirective& b) const {
    if (a.type != b.type) return true;
    // A flip in the terminal-stop flag is a change even when the world
    // point is identical.
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
    bypass_active_ = false;
    locked_side_ = SideSelection::NONE;
    update_event_ = 0;
    correction_enter_event_ = 0;
    correction_exit_event_ = 0;
    correction_update_event_ = 0;
    stagnant_update_count_ = 0;
    reentry_success_updates_ = 0;
    waypoint_execution_fail_updates_ = 0;
    search_active_ = false;
    last_state_position_ = Vec2d(0.0, 0.0);
    has_last_state_position_ = false;
    last_directive_ = TargetCorrectionDirective{};
    last_obs_ = AvoidanceObservability{};
}

void VisibilityTargetCorrector::resetForNewGoal() {
    bypass_active_ = false;
    locked_side_ = SideSelection::NONE;
    stagnant_update_count_ = 0;
    reentry_success_updates_ = 0;
    waypoint_execution_fail_updates_ = 0;
    search_active_ = false;
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
    constexpr uint32_t kReentrySuccessUpdates = 3;

    // ── 5 Hz stagnation watchdog (event-driven correction refresh). ──
    // Count consecutive updates with negligible motion while a BYPASS
    // correction is active, so a stale waypoint cannot hold the vehicle in
    // a permanent brake/hold loop.
    const bool previous_normal_translation =
        bypass_active_ &&
        last_directive_.type == TargetCorrectionType::NORMAL_CORRECTION &&
        last_directive_.normalized_distance > 1e-9;
    if (has_last_state_position_ && previous_normal_translation) {
        const double moved =
            (state.position - last_state_position_).norm();
        const double progress_epsilon = std::max(
            0.01, 0.4 * p_.vehicle_stationary_speed_mps);
        stagnant_update_count_ =
            (moved < progress_epsilon) ? stagnant_update_count_ + 1 : 0;
    } else {
        stagnant_update_count_ = 0;
    }
    last_state_position_ = state.position;
    has_last_state_position_ = true;

    // ── Observability (diagnostic + fresh goal / blocker evidence). ──
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

    const bool goal_in_fov = fresh_obs.goal_inside_fov;
    const bool corridor_blocked = fresh_obs.direct_corridor_blocked;
    const bool local_can_track_goal =
        assessment.translation_plan_valid || assessment.rotation_available;

    // ── World-locked TURN.  A TURN latches a world HEADING and is held until
    //    the residual heading is consumed (the FLU label then decays toward 0
    //    as the drone rotates) instead of the teacher re-issuing a fresh
    //    step every 5 Hz.  A DIRECT goal-acquisition turn completes early
    //    once the goal re-enters the FOV; a BYPASS observation turn is held
    //    until the nose actually reaches the side frontier (that completed
    //    scan is the only evidence used to flip the side). ────────────────
    const bool last_is_turn =
        last_directive_.type == TargetCorrectionType::TURN_LEFT ||
        last_directive_.type == TargetCorrectionType::TURN_RIGHT;
    if (last_is_turn && last_directive_.turn_direction_world_valid &&
        (search_active_ || !bypass_active_)) {
        const Vec2d dir = last_directive_.turn_direction_world;
        const double residual =
            wrapAngle(std::atan2(dir.y(), dir.x()) - state.yaw);
        const double complete_rad = deg2rad(p_.lp_turn_exit_deg);
        bool turn_done = std::fabs(residual) <= complete_rad + 1e-9;
        if (!bypass_active_ && goal_in_fov) {
            turn_done = true;  // goal re-acquired mid-turn
        }
        if (!turn_done) {
            TargetCorrectionDirective held = last_directive_;
            held.reason = "TURN_HELD_WORLD_HEADING";
            held.update_event = update_event_;
            last_directive_ = held;
            return held;
        }
    }

    // Helper: emit a PASS directive (the original goal is chased again).
    auto emitPass = [&](const char* reason) {
        TargetCorrectionDirective d;
        d.valid = true;
        d.type = TargetCorrectionType::PASS_THROUGH;
        d.locked_side = SideSelection::NONE;
        d.reason = reason;
        const bool was_override =
            last_directive_.type != TargetCorrectionType::PASS_THROUGH;
        if (was_override) ++correction_exit_event_;
        bypass_active_ = false;
        locked_side_ = SideSelection::NONE;
        stagnant_update_count_ = 0;
        reentry_success_updates_ = 0;
        search_active_ = false;
        waypoint_execution_fail_updates_ = 0;
        if (was_override) ++update_event_;
        d.update_event = update_event_;
        last_directive_ = d;
        return d;
    };

    // Helper: emit a world-latched TURN that brings the ORIGINAL goal back
    // into the FOV (a DIRECT-mode re-acquisition, not a bypass).
    auto emitTurnTowardGoal = [&]() {
        const Vec2d to_goal = original_goal - state.position;
        const double b_goal = wrapAngle(
            std::atan2(to_goal.y(), to_goal.x()) - state.yaw);
        const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
        const SideSelection side = b_goal > fov_half + 1e-9
                                       ? SideSelection::LEFT
                                       : SideSelection::RIGHT;
        const int n = std::max(3, p_.te_direction_bin_count);
        TargetCorrectionDirective d;
        d.valid = true;
        d.type = side == SideSelection::LEFT
                     ? TargetCorrectionType::TURN_LEFT
                     : TargetCorrectionType::TURN_RIGHT;
        d.direction_token = side == SideSelection::LEFT ? 0 : n + 1;
        d.decoded_direction_body =
            adapter_.decodeDirectionToken(d.direction_token);
        d.normalized_distance = 1.0;
        d.corrected_target_world_valid = false;
        d.turn_direction_world =
            rot2(d.decoded_direction_body, state.yaw).normalized();
        d.turn_direction_world_valid = true;
        d.locked_side = side;
        d.reason = "TURN_TO_ACQUIRE_GOAL";
        const bool was_override =
            last_directive_.type != TargetCorrectionType::PASS_THROUGH;
        const bool changed = !was_override ||
                             directiveChanged(last_directive_, d);
        if (!was_override) ++correction_enter_event_;
        if (changed) ++update_event_;
        d.update_event = update_event_;
        last_directive_ = d;
        return d;
    };

    // ═══════════════════════════════════════════════════════════════
    //  DIRECT mode — the original goal is the attention centre.
    // ═══════════════════════════════════════════════════════════════
    if (!bypass_active_) {
        if (local_can_track_goal) {
            return emitPass("LOCAL_CAN_PLAN_ORIGINAL_TARGET");
        }
        if (!goal_in_fov) {
            // Goal outside the FOV: rotate it back into view.  This is a
            // DIRECT-mode re-acquisition, not a bypass.
            return emitTurnTowardGoal();
        }
        // Goal inside the FOV but the local expert keeps reporting it
        // cannot proceed.  Only the LOCAL verdict (not our own obstacle
        // reading) authorizes a takeover, and only after the persistent
        // failure confirmation window (assessment.takeover_confirmed).
        if (!assessment.takeover_confirmed) {
            return emitPass("LOCAL_TAKEOVER_PENDING_CONFIRMED_FAILURE");
        }
        // Enter BYPASS: commit to one side ONCE for this blocker episode.
        // The side is a guidance constraint — every correction re-derives
        // its geometry from the current depth and live goal bearing.
        locked_side_ = selectSide(state, current_patch, original_goal);
        if (locked_side_ == SideSelection::NONE) {
            locked_side_ = SideSelection::RIGHT;
        }
        bypass_active_ = true;
        stagnant_update_count_ = 0;
        reentry_success_updates_ = 0;
        search_active_ = false;
        waypoint_execution_fail_updates_ = 0;
    }

    // ═══════════════════════════════════════════════════════════════
    //  BYPASS mode — the blocker / bypass frontier is the attention
    //  centre.  The original goal's FOV status is deliberately ignored.
    // ═══════════════════════════════════════════════════════════════
    // Exit only when the original goal re-enters the local expert's
    // capability set: translation-valid AND in FOV AND the direct corridor
    // is no longer blocked, confirmed for several consecutive updates.
    if (assessment.translation_plan_valid && goal_in_fov &&
        !corridor_blocked) {
        reentry_success_updates_ = std::min(
            reentry_success_updates_ + 1, kReentrySuccessUpdates);
        if (reentry_success_updates_ >= kReentrySuccessUpdates) {
            return emitPass("ORIGINAL_TRANSLATION_REENTRY_CONFIRMED");
        }
    } else {
        reentry_success_updates_ = 0;
    }

    // If the current patch already contains the blocker, build the guide
    // from that patch; otherwise fall back to the causal history.
    const LocalObservation& planning_observation =
        corridor_blocked
            ? current_patch
            : (local_history.valid() ? local_history : current_patch);
    const LocalFreeGrid grid =
        buildLocalFreeGrid(state, planning_observation);

    // Held-waypoint drop guard: a held CORRECTION is only dropped after >=
    // 3 consecutive updates in which it is neither executing nor cold-
    // preview-feasible, or after a sustained no-progress stagnation.
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
    const uint32_t stagnation_limit = static_cast<uint32_t>(std::max(
        3, p_.macro_unknown_recovery_threshold_ticks / 6));
    const bool drop_held_allowed =
        waypoint_execution_fail_updates_ >= 3 ||
        stagnant_update_count_ >= stagnation_limit;

    TargetCorrectionDirective d = makeCorrectionDirective(
        state, original_goal, planning_observation, grid, locked_side_,
        live_directive_usable, drop_held_allowed, assess_directive);

    // ── BYPASS observation phase.  A TURN is held (world-latched) at the
    //    top of update() until the nose reaches the side frontier.  We only
    //    reach this point with search_active_ set AFTER that scan completed
    //    (a still-incomplete held TURN returns earlier).  So: a TURN proposal
    //    on an ALREADY-ACTIVE observation means the fresh depth STILL finds
    //    no passable frontier on the committed side → flip once.  A TURN on
    //    an inactive observation just starts the scan (no flip). ──────────
    bool proposed_turn =
        d.type == TargetCorrectionType::TURN_LEFT ||
        d.type == TargetCorrectionType::TURN_RIGHT;
    if (proposed_turn && search_active_) {
        // Completed scan still shows no frontier on this side: flip.
        locked_side_ = locked_side_ == SideSelection::LEFT
                           ? SideSelection::RIGHT
                           : SideSelection::LEFT;
        waypoint_execution_fail_updates_ = 0;
        search_active_ = false;
        d = makeCorrectionDirective(state, original_goal, planning_observation,
                                    grid, locked_side_, live_directive_usable,
                                    true, assess_directive);
        proposed_turn = d.type == TargetCorrectionType::TURN_LEFT ||
                        d.type == TargetCorrectionType::TURN_RIGHT;
    }
    if (proposed_turn) {
        search_active_ = true;   // start (or continue) an observation scan
    } else {
        search_active_ = false;
    }
    d.locked_side = locked_side_;

    // ── Finalize: bump events on directive changes; ZOH-hold otherwise. ──
    const bool was_override =
        last_directive_.type != TargetCorrectionType::PASS_THROUGH;
    const bool is_override = d.type != TargetCorrectionType::PASS_THROUGH;
    const bool changed = was_override != is_override ||
                         directiveChanged(last_directive_, d);
    if (changed) {
        ++update_event_;
        if (is_override && !was_override) {
            ++correction_enter_event_;
        } else if (!is_override && was_override) {
            ++correction_exit_event_;
        } else if (is_override) {
            ++correction_update_event_;
        }
    } else if (is_override) {
        const std::string reason = d.reason;
        d = last_directive_;
        d.reason = reason;
    }
    d.valid = true;
    d.update_event = update_event_;
    last_directive_ = d;
    return d;
}

}  // namespace expert
}  // namespace il_dataset
