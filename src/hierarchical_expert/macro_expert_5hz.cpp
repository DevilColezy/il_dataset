#include "il_dataset/hierarchical_expert/macro_expert_5hz.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>

namespace il_dataset {
namespace expert {

// ═══════════════════════════════════════════════════════════════════
//  Local free-grid construction + observability judgement
//  (current patch + decaying local history — deterministic, local, causal)
// ═══════════════════════════════════════════════════════════════════
VisibilityTargetCorrector::LocalFreeGrid
VisibilityTargetCorrector::buildLocalFreeGrid(const VehicleState2D& state,
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

    const double req = std::max(
        {0.0, p_.lp_min_clearance, p_.lp_nominal_clearance_m,
         p_.scene_safety_clearance + p_.macro_route_clearance_margin +
             p_.lp_clearance_discretization_margin_m});
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

bool VisibilityTargetCorrector::chordClear(
    const VehicleState2D& state, const LocalObservation& patch,
    const LocalFreeGrid& grid, const Vec2d& endpoint) const {
    if (!patch.valid() || !grid.valid()) return false;
    const Vec2d delta = endpoint - state.position;
    const double dist = delta.norm();
    if (dist < 1e-9) return false;
    const double recovery = p_.macro_local_recovery_prefix_m;
    const double step = std::max(1e-3, 0.5 * patch.resolution);
    const int samples = std::max(1, static_cast<int>(std::ceil(dist / step)));
    for (int s = 0; s <= samples; ++s) {
        const double u = static_cast<double>(s) / samples;
        const Vec2d p = state.position + u * delta;
        if (!grid.freeAt(p)) return false;
        if (u * dist > recovery && !grid.traversableAt(p)) return false;
    }
    return true;
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
    const VehicleState2D& state, const LocalObservation& obs,
    double bearing_body) const {
    return freeRangeAlongFrom(obs, state.position, state.yaw + bearing_body);
}

bool VisibilityTargetCorrector::extractBlocker(
    const VehicleState2D& state, const Vec2d& goal,
    const LocalObservation& patch, double& blocker_min_along,
    double& blocker_max_lateral) const {
    blocker_min_along = std::numeric_limits<double>::infinity();
    blocker_max_lateral = 0.0;
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
            const double lateral = std::fabs(cross2(rel, dir));
            const double along = rel.dot(dir);
            if (lateral <= hw &&
                along >= -p_.macro_corridor_rear_tolerance_m &&
                along <= max_along) {
                blocker_min_along = std::min(blocker_min_along, along);
                blocker_max_lateral = std::max(blocker_max_lateral, lateral);
                found = true;
            }
        }
    }
    return found;
}

AvoidanceObservability VisibilityTargetCorrector::assessObservability(
    const VehicleState2D& state, const Vec2d& goal,
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

    // 1) Direct corridor check (vehicle → original goal, to perception
    //    range). Only locally observed OCCUPIED blocks.
    double blocker_min_along = std::numeric_limits<double>::infinity();
    double blocker_max_lateral = 0.0;
    const bool has_blocker =
        extractBlocker(state, goal, patch, blocker_min_along,
                       blocker_max_lateral);
    obs.direct_corridor_blocked = has_blocker;
    obs.blocker_observed = has_blocker;

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
        obs.fov_boundary_truncated =
            blocker_min_along >= p_.obs_range_m - 1.0 ||
            blocker_max_lateral >= 0.9 * fov_hw_at_blocker;
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

// ═══════════════════════════════════════════════════════════════════
//  Side candidate sampling / certification
// ═══════════════════════════════════════════════════════════════════
std::vector<VisibilityTargetCorrector::SideCandidate>
VisibilityTargetCorrector::sampleSideCandidates(
    const VehicleState2D& state, const Vec2d& goal,
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
    // A local correction is guidance toward the current mission goal, not a
    // request to travel beyond it.  (NORMAL_CORRECTION candidates must not
    // cross the goal.)
    const double candidate_max_distance = std::min(
        dmax, goal_dist + std::max(0.0, p_.task_goal_tolerance));
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
            const double along = (endpoint - state.position).dot(axis);
            if (along < p_.macro_observable_frontier_min_progress_m) {
                continue;
            }
            if (along > goal_dist + p_.task_goal_tolerance) {
                continue;
            }
            if (strict &&
                std::fabs(wrapAngle(b - b_goal)) > fov_half + 1e-9) {
                continue;
            }
            if (strict && along < beyond) continue;
            if (!grid.reachableAt(endpoint)) continue;
            if (!strict && !chordClear(state, patch, grid, endpoint)) continue;
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

// ═══════════════════════════════════════════════════════════════════
//  Side selection (current patch visible evidence only)
// ═══════════════════════════════════════════════════════════════════
SideSelection VisibilityTargetCorrector::selectSide(
    const VehicleState2D& state, const LocalObservation& patch,
    const Vec2d& goal) const {
    if (!patch.valid()) return SideSelection::RIGHT;
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
        return SideSelection::RIGHT;  // ambiguous → fixed RIGHT
    }
    const double left_avg = left_total / static_cast<double>(pairs);
    const double right_avg = right_total / static_cast<double>(pairs);
    const double m = p_.macro_side_evidence_margin;
    if (left_avg > right_avg + m) return SideSelection::LEFT;
    if (right_avg > left_avg + m) return SideSelection::RIGHT;
    return SideSelection::RIGHT;  // indistinguishable → fixed RIGHT
}

// ═══════════════════════════════════════════════════════════════════
//  Correction directive construction
// ═══════════════════════════════════════════════════════════════════
TargetCorrectionDirective VisibilityTargetCorrector::makeCorrectionDirective(
    const VehicleState2D& state, const Vec2d& goal,
    const LocalObservation& patch, const LocalFreeGrid& grid,
    SideSelection side) const {
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

    // Try NORMAL_CORRECTION: a safe ordinary observation frontier on the
    // locked side (strict=false).  The expert's CONTINUOUS bearing is then
    // quantized through the student adapter and the executed world point is
    // REBUILT from the quantized direction + clamped distance.
    double blocker_min_along = std::numeric_limits<double>::infinity();
    double blocker_max_lateral = 0.0;
    const bool has_blocker =
        extractBlocker(state, goal, patch, blocker_min_along,
                       blocker_max_lateral);
    const std::vector<SideCandidate> cands =
        sampleSideCandidates(state, goal, patch, grid, has_blocker,
                             blocker_min_along, side, /*strict=*/false);
    if (!cands.empty()) {
        const SideCandidate* best = nullptr;
        for (const SideCandidate& c : cands) {
            if (!best || c.along_progress > best->along_progress ||
                (c.along_progress == best->along_progress &&
                 c.dist > best->dist)) {
                best = &c;
            }
        }
        const int token = adapter_.quantizeBearing(best->bearing);
        d.direction_token = token;
        d.decoded_direction_body = adapter_.decodeDirectionToken(token);
        const double normalized =
            adapter_.clampNormalizedDistance(best->dist);
        d.normalized_distance = normalized;
        const Vec2d dir_world = rot2(d.decoded_direction_body, state.yaw);
        d.corrected_target_world =
            state.position + dir_world * adapter_.normalizedToWorld(normalized);
        d.corrected_target_world_valid = true;
        d.turn_direction_world_valid = false;
        d.type = TargetCorrectionType::NORMAL_CORRECTION;
        d.reason = "NORMAL_CORRECTION_FRONTIER";
    }
    return d;
}

bool VisibilityTargetCorrector::directiveChanged(
    const TargetCorrectionDirective& a,
    const TargetCorrectionDirective& b) const {
    if (a.type != b.type) return true;
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

// ═══════════════════════════════════════════════════════════════════
//  Public API: reset / bump / update
// ═══════════════════════════════════════════════════════════════════
void VisibilityTargetCorrector::reset() {
    correction_active_ = false;
    locked_side_ = SideSelection::NONE;
    enter_stable_count_ = 0;
    reentry_guard_until_tick_ = 0;
    update_event_ = 0;
    correction_enter_event_ = 0;
    correction_exit_event_ = 0;
    correction_update_event_ = 0;
    last_directive_ = TargetCorrectionDirective{};
    last_obs_ = AvoidanceObservability{};
}

void VisibilityTargetCorrector::resetForNewGoal() {
    correction_active_ = false;
    locked_side_ = SideSelection::NONE;
    enter_stable_count_ = 0;
    reentry_guard_until_tick_ = 0;
    last_directive_ = TargetCorrectionDirective{};
    last_directive_.update_event = update_event_;
    last_obs_ = AvoidanceObservability{};
}

uint64_t VisibilityTargetCorrector::bumpDirectiveEvent() {
    ++update_event_;
    return update_event_;
}

TargetCorrectionDirective VisibilityTargetCorrector::update(
    const VehicleState2D& state, const Vec2d& original_goal,
    const LocalObservation& current_patch,
    const LocalObservation& local_history, uint64_t tick) {

    // 1) Local observability judgement.  The fused history contains the
    //    current patch plus recent causal observations.  A fresh-patch
    //    judgement is retained for the correction-entry gate and side
    //    choice, preventing stale-only entry.
    last_obs_ = assessObservability(state, original_goal, local_history);
    const AvoidanceObservability& obs = last_obs_;
    const AvoidanceObservability fresh_obs =
        assessObservability(state, original_goal, current_patch);

    // 2) Enter / exit conditions (5 Hz's own judgement — NEVER the 30 Hz
    //    outcome):
    const bool enter_ready =
        fresh_obs.goal_inside_fov && fresh_obs.direct_corridor_blocked &&
        !obs.local_avoidance_observable;
    const Vec2d to_goal = original_goal - state.position;
    const double goal_bearing =
        to_goal.squaredNorm() > 1e-12
            ? wrapAngle(std::atan2(to_goal.y(), to_goal.x()) - state.yaw)
            : 0.0;
    const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
    const double handoff_half =
        std::max(0.0, fov_half - deg2rad(p_.te_turn_ray_margin_deg));
    const bool goal_handoff_ready =
        std::fabs(goal_bearing) <= handoff_half + 1e-9;
    const bool yaw_rate_handoff_ready =
        std::fabs(state.yaw_rate) <= p_.lp_turn_exit_max_yaw_rate + 1e-9;
    const bool base_exit_ready =
        goal_handoff_ready && yaw_rate_handoff_ready &&
        obs.local_avoidance_observable;

    TargetCorrectionDirective d;
    d.valid = true;
    d.type = TargetCorrectionType::PASS_THROUGH;
    d.locked_side = correction_active_ ? locked_side_ : SideSelection::NONE;
    d.reason = "PASS_THROUGH";
    d.update_event = update_event_;

    bool changed = false;
    if (!correction_active_) {
        if (enter_ready && tick >= reentry_guard_until_tick_) {
            ++enter_stable_count_;
            if (enter_stable_count_ >= p_.macro_correction_enter_stable_ticks) {
                // ENTER correction: lock the side from current-patch
                // evidence only (ambiguous → fixed RIGHT).
                locked_side_ = selectSide(state, current_patch, original_goal);
                if (locked_side_ == SideSelection::NONE) {
                    locked_side_ = SideSelection::RIGHT;
                }
                correction_active_ = true;
                enter_stable_count_ = 0;
                ++correction_enter_event_;
                const LocalFreeGrid grid = buildLocalFreeGrid(state, local_history);
                d = makeCorrectionDirective(state, original_goal, local_history,
                                            grid, locked_side_);
                d.reason = "CORRECTION_ENTER " + d.reason;
                changed = true;
            } else {
                d.reason = "CORRECTION_ENTER_PENDING";
            }
        } else {
            enter_stable_count_ = 0;
        }
    } else {
        // Construct the next local-only correction before deciding whether
        // to release.
        const LocalFreeGrid grid = buildLocalFreeGrid(state, local_history);
        TargetCorrectionDirective proposed = makeCorrectionDirective(
            state, original_goal, local_history, grid, locked_side_);
        proposed.locked_side = locked_side_;

        const bool previous_was_turn =
            last_directive_.type == TargetCorrectionType::TURN_LEFT ||
            last_directive_.type == TargetCorrectionType::TURN_RIGHT;
        const bool visible_bypass =
            obs.direct_corridor_blocked &&
            (obs.left_bypass_observable || obs.right_bypass_observable);
        const bool normal_bridge_available =
            proposed.type == TargetCorrectionType::NORMAL_CORRECTION;

        // A TURN token is a bounded, world-latched angular step.  If no
        // ordinary frontier is available yet and the previous turn anchor
        // is still outside the current FOV, keep that exact anchor.  Once
        // it has entered the FOV, this 5 Hz boundary may issue another
        // bounded step (finite turns, no infinite in-place spin).
        if (previous_was_turn && !normal_bridge_available &&
            last_directive_.turn_direction_world_valid) {
            const Vec2d old_direction =
                last_directive_.turn_direction_world;
            if (old_direction.squaredNorm() > 1e-12) {
                const double old_bearing = wrapAngle(
                    std::atan2(old_direction.y(), old_direction.x()) -
                    state.yaw);
                if (std::fabs(old_bearing) > fov_half + 1e-9) {
                    proposed = last_directive_;
                    proposed.reason = "TURN_STEP_PENDING";
                }
            }
        }
        const bool require_turn_to_normal_bridge =
            previous_was_turn && visible_bypass && normal_bridge_available;
        const bool exit_ready =
            base_exit_ready && !require_turn_to_normal_bridge;

        if (exit_ready) {
            // Release immediately on the first real 5 Hz observation that
            // certifies a locally observable route, a safe original-goal
            // handoff bearing, and sufficiently low actual yaw rate.
            correction_active_ = false;
            locked_side_ = SideSelection::NONE;
            enter_stable_count_ = 0;
            ++correction_exit_event_;
            reentry_guard_until_tick_ =
                tick + static_cast<uint64_t>(
                           std::max(0, p_.macro_reentry_guard_ticks));
            d.type = TargetCorrectionType::PASS_THROUGH;
            d.locked_side = SideSelection::NONE;
            d.reason = "CORRECTION_EXIT_HANDOFF_READY";
            changed = true;
        } else {
            d = std::move(proposed);
            if (require_turn_to_normal_bridge) {
                d.reason = "TURN_TO_NORMAL_HANDOFF_BRIDGE " + d.reason;
            } else if (obs.local_avoidance_observable &&
                       !yaw_rate_handoff_ready) {
                d.reason = "HANDOFF_PENDING_YAW_RATE " + d.reason;
            } else if (obs.local_avoidance_observable &&
                       !goal_handoff_ready) {
                d.reason = "HANDOFF_PENDING_GOAL_OUTSIDE_CONE " + d.reason;
            }
            if (directiveChanged(last_directive_, d)) {
                changed = true;
                ++correction_update_event_;
            } else {
                // Keep the previous numeric directive exactly.
                const std::string current_reason = d.reason;
                d = last_directive_;
                d.reason = current_reason;
            }
        }
    }

    if (changed) ++update_event_;
    d.update_event = update_event_;
    d.valid = true;
    last_directive_ = d;
    return d;
}

}  // namespace expert
}  // namespace il_dataset
