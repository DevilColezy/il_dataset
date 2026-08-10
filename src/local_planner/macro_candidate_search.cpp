#include "il_dataset/local_planner/macro_candidate_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "il_dataset/local_planner/local_path_search.hpp"
#include "il_dataset/local_planner/observed_map.hpp"

namespace il_dataset {

namespace {

constexpr double kEpsilon = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;

/// Straight known-and-clear segment check with a SINGLE required ESDF
/// clearance.  The ESDF already subtracts the vehicle radius, so callers
/// pass the extra margin they actually mean — never radius + margin again
/// (that would double-inflate).  Used only where a straight segment is the
/// right abstraction (side-corridor extent, DIRECT lookahead); OBSERVE and
/// GOAL_FRONTIER movement candidates use the real observed LocalPathSearch
/// instead (section XV).
bool segmentKnownAndClear(const ObservedMap& map,
                          const Eigen::Vector3d& start,
                          const Eigen::Vector3d& end,
                          double required_esdf_clearance) {
    const double length = (end - start).norm();
    const double spacing = 0.10;
    if (length <= kEpsilon) {
        return map.isKnownFree(start.x(), start.y(), start.z(),
                               required_esdf_clearance);
    }
    const Eigen::Vector3d direction = (end - start) / length;
    const int samples =
        std::max(1, static_cast<int>(std::ceil(length / std::max(0.02, spacing))));
    for (int i = 0; i <= samples; ++i) {
        const Eigen::Vector3d point = start + direction *
                                                 (length * i / samples);
        if (!map.isKnownFree(point.x(), point.y(), point.z(),
                             required_esdf_clearance)) {
            return false;
        }
    }
    return true;
}

/// Build the search config for a candidate's reachability query.
///
/// Round 5: the observed FULL-reachable A* uses the SAME nominal planning
/// clearance as the 30 Hz LocalPlanner, so a candidate marked FULL is never
/// immediately
/// rejected by the local planner over an inconsistent safety margin.  The
/// formula is the shared C++ effectiveClearanceForSpeed() — never a local
/// copy.
LocalSearchConfig makeSearchConfig(const MacroCandidateConfig& config,
                                   Side committed_side) {
    LocalSearchConfig search_config;
    const DynamicClearanceConfig clearance{
        config.clearance_m, config.clearance_margin_tracking_m,
        config.clearance_margin_latency_s, config.clearance_margin_max_m};
    search_config.clearance_m = planningClearanceForSpeed(
        clearance, config.nominal_speed_mps,
        config.planning_clearance_margin_m);
    search_config.max_time_ms = config.search_max_time_ms;
    search_config.region_margin_m = config.search_region_margin_m;
    search_config.side_bias_gain = config.side_bias_gain;
    search_config.committed_side = committed_side;
    search_config.forbid_unknown = true;
    return search_config;
}

/// Round 5: the candidate-search view of the shared dynamic clearance.
double effectiveCandidateClearance(const MacroCandidateConfig& config) {
    const DynamicClearanceConfig clearance{
        config.clearance_m, config.clearance_margin_tracking_m,
        config.clearance_margin_latency_s, config.clearance_margin_max_m};
    return planningClearanceForSpeed(clearance, config.nominal_speed_mps,
                                     config.planning_clearance_margin_m);
}

/// World delta -> FLU using yaw only (level body).  UNIFIED FLU convention
/// (section XVIII): +x forward, +y left, +z up.
///   yaw = 0  =>  nose points toward world +Y
///   nose world direction  = (-sin yaw,  cos yaw)
///   left  world direction = (-cos yaw, -sin yaw)
/// This is the exact same math as Python `il_common.world_to_flu_xy` and
/// matches the quaternion version numerically at level attitude.
Eigen::Vector3d worldToFlu(const Eigen::Vector3d& world_delta, double yaw) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    return Eigen::Vector3d(-s * world_delta.x() + c * world_delta.y(),
                           -c * world_delta.x() - s * world_delta.y(),
                           world_delta.z());
}

/// Select an observation yaw using only the candidate and the final task
/// goal.  The camera faces the goal-side of the candidate so a lateral peek
/// exposes the previously occluded forward region rather than merely the
/// route used to arrive at the viewpoint.
double observationYawTowardGoal(const Eigen::Vector3d& position,
                                const Eigen::Vector3d& goal_world,
                                double fallback_yaw) {
    const Eigen::Vector2d delta =
        goal_world.head<2>() - position.head<2>();
    if (delta.norm() <= kEpsilon) return fallback_yaw;
    return std::atan2(delta.y(), delta.x()) - 0.5 * kPi;
}

/// Flood-fill the blocked component on the drone's height slice containing
/// `seed_ix, seed_iy`.  A cell is blocked when it is not known-free with
/// `block_clearance`.
struct BlockComponent {
    int id = -1;
    int count = 0;
    Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
    double extent = 0.0;
    bool blocked_by_known = false;
    int min_ix = std::numeric_limits<int>::max();
    int max_ix = std::numeric_limits<int>::min();
    int min_iy = std::numeric_limits<int>::max();
    int max_iy = std::numeric_limits<int>::min();
};

BlockComponent floodBlockedComponent(const ObservedMap& map,
                                     int seed_ix,
                                     int seed_iy,
                                     int z_index,
                                     double block_clearance,
                                     int region_radius_cells) {
    BlockComponent comp;
    const int gx = map.gx();
    const int gy = map.gy();
    const double res = map.resolution();
    const Eigen::Vector3d origin = map.origin();
    const double z_world = origin.z() + (static_cast<double>(z_index) + 0.5) * res;
    const double seed_wx =
        origin.x() + (static_cast<double>(seed_ix) + 0.5) * res;
    const double seed_wy =
        origin.y() + (static_cast<double>(seed_iy) + 0.5) * res;
    const bool seed_known = map.isKnown(seed_wx, seed_wy, z_world);
    const double seed_clearance = map.esdfValue(seed_wx, seed_wy, z_world);
    // A FREE voxel inside the ESDF safety band is blocked by known geometry,
    // not by missing information.  Treating only raw OCCUPIED seeds as known
    // sent grazing corridors into an OBSERVE scan loop.
    comp.blocked_by_known =
        seed_known && std::isfinite(seed_clearance) &&
        seed_clearance <= block_clearance;
    auto blocked_at = [&](int ix, int iy) {
        if (std::abs(ix - seed_ix) > region_radius_cells ||
            std::abs(iy - seed_iy) > region_radius_cells) {
            return false;
        }
        const double wx =
            origin.x() + (static_cast<double>(ix) + 0.5) * res;
        const double wy =
            origin.y() + (static_cast<double>(iy) + 0.5) * res;
        const bool known = map.isKnown(wx, wy, z_world);
        if (!comp.blocked_by_known) return !known;
        const double clearance = map.esdfValue(wx, wy, z_world);
        return known && std::isfinite(clearance) &&
               clearance <= block_clearance;
    };

    std::vector<int> queue;
    queue.reserve(4096);
    std::vector<std::uint8_t> visited(static_cast<size_t>(gx) * gy, 0);
    const int seed_index = seed_iy * gx + seed_ix;
    if (seed_ix < 0 || seed_ix >= gx || seed_iy < 0 || seed_iy >= gy) {
        return comp;
    }
    visited[static_cast<size_t>(seed_index)] = 1;
    queue.push_back(seed_index);
    int sum_x = 0;
    int sum_y = 0;
    int count = 0;
    const int di[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    const int dj[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    while (!queue.empty()) {
        const int index = queue.back();
        queue.pop_back();
        const int ix = index % gx;
        const int iy = index / gx;
        ++count;
        sum_x += ix;
        sum_y += iy;
        comp.min_ix = std::min(comp.min_ix, ix);
        comp.max_ix = std::max(comp.max_ix, ix);
        comp.min_iy = std::min(comp.min_iy, iy);
        comp.max_iy = std::max(comp.max_iy, iy);
        if (!blocked_at(ix, iy)) continue;
        for (int n = 0; n < 8; ++n) {
            const int nxi = ix + di[n];
            const int nyi = iy + dj[n];
            if (nxi < 0 || nxi >= gx || nyi < 0 || nyi >= gy) continue;
            const int nindex = nyi * gx + nxi;
            if (visited[static_cast<size_t>(nindex)] != 0) continue;
            if (blocked_at(nxi, nyi)) {
                visited[static_cast<size_t>(nindex)] = 1;
                queue.push_back(nindex);
            }
        }
    }
    if (count == 0) return comp;
    comp.count = count;
    const double cx = static_cast<double>(sum_x) / count;
    const double cy = static_cast<double>(sum_y) / count;
    comp.centroid = Eigen::Vector3d(
        origin.x() + (cx + 0.5) * res,
        origin.y() + (cy + 0.5) * res, z_world);
    for (int ix = 0; ix < gx; ++ix) {
        for (int iy = 0; iy < gy; ++iy) {
            const int index = iy * gx + ix;
            if (visited[static_cast<size_t>(index)] == 0) continue;
            const double wx =
                origin.x() + (static_cast<double>(ix) + 0.5) * res;
            const double wy =
                origin.y() + (static_cast<double>(iy) + 0.5) * res;
            const double d = std::hypot(wx - comp.centroid.x(),
                                        wy - comp.centroid.y());
            comp.extent = std::max(comp.extent, d);
        }
    }
    return comp;
}

}  // namespace

GoalBlocker analyzeGoalBlocker(const ObservedMap& map,
                               const VehicleState& state,
                               const Eigen::Vector3d& goal_world,
                               const MacroCandidateConfig& config) {
    GoalBlocker blocker;
    if (!map.esdfBuilt()) return blocker;
    const double res = map.resolution();
    const Eigen::Vector3d origin = map.origin();
    const Eigen::Vector2d travel = goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len <= kEpsilon) return blocker;
    const Eigen::Vector2d goal_dir = travel / travel_len;

    const double z = state.position.z();
    const double effective_clearance =
        effectiveCandidateClearance(config);
    const double max_walk = std::min(travel_len, config.edge_search_radius_m);
    Eigen::Vector3d block_point{Eigen::Vector3d::Zero()};
    bool found = false;
    int block_ix = 0;
    int block_iy = 0;
    int block_iz = 0;
    // Explicitly saved ray depth: the loop variable `d` is scoped to the
    // for-loop, so it can never be referenced afterwards (section II/IV).
    double blocking_ray_depth = -1.0;
    for (double d = res; d <= max_walk + 1.0e-6; d += res) {
        const Eigen::Vector3d point(
            state.position.x() + goal_dir.x() * d,
            state.position.y() + goal_dir.y() * d, z);
        if (!map.isKnownFree(point.x(), point.y(), point.z(),
                             effective_clearance)) {
            block_point = point;
            found = true;
            blocking_ray_depth = d;
            const Eigen::Vector3i g = map.worldToGridInt(point);
            block_ix = g.x();
            block_iy = g.y();
            block_iz = g.z();
            break;
        }
    }
    if (!found) return blocker;
    blocker.found = true;
    blocker.blocking_ray_depth = blocking_ray_depth;

    const int region_radius =
        std::max(8, static_cast<int>(std::ceil(config.edge_search_radius_m / res)));
    const BlockComponent comp =
        floodBlockedComponent(map, block_ix, block_iy, block_iz,
                              effective_clearance,
                              region_radius);
    if (comp.count > 0) {
        blocker.centroid = comp.centroid;
        blocker.extent = comp.extent;
        blocker.component_cell_count = comp.count;
        blocker.bbox_min_world = Eigen::Vector3d(
            origin.x() + (static_cast<double>(comp.min_ix) + 0.5) * res,
            origin.y() + (static_cast<double>(comp.min_iy) + 0.5) * res,
            z);
        blocker.bbox_max_world = Eigen::Vector3d(
            origin.x() + (static_cast<double>(comp.max_ix) + 0.5) * res,
            origin.y() + (static_cast<double>(comp.max_iy) + 0.5) * res,
            z);
        // Stable-ish signature: quantize the world geometry on a 0.5 m
        // grid (section VIII).  The macro layer still performs the real
        // association via centroid distance + bbox overlap.
        int signature = 2166136261;
        auto mix = [&signature](int v) {
            signature = (signature ^ v) * 16777619;
        };
        mix(static_cast<int>(std::floor(comp.centroid.x() / 0.5)));
        mix(static_cast<int>(std::floor(comp.centroid.y() / 0.5)));
        mix(static_cast<int>(std::floor(comp.centroid.z() / 0.5)));
        mix(static_cast<int>(std::floor(comp.extent / 0.5)));
        mix(static_cast<int>(std::floor(blocking_ray_depth / 0.5)));
        blocker.blocker_signature = signature;
    }
    blocker.blocked_by_known = comp.blocked_by_known;

    // ── Edge visibility + known corridor on each side ──────────────
    const Eigen::Vector2d perp(-goal_dir.y(), goal_dir.x());  // + = left
    const double corridor_clearance =
        std::max(config.side_corridor_radius_m, effective_clearance);
    const int max_edge_cells =
        static_cast<int>(std::ceil(config.edge_search_radius_m / res));
    for (int side_sign = -1; side_sign <= 1; side_sign += 2) {
        const bool is_left = side_sign > 0;
        Eigen::Vector3d* edge_world =
            is_left ? &blocker.left_edge_world : &blocker.right_edge_world;
        bool* edge_visible =
            is_left ? &blocker.left_edge_visible : &blocker.right_edge_visible;
        bool* corridor_known =
            is_left ? &blocker.left_corridor_known : &blocker.right_corridor_known;
        Eigen::Vector3d* corridor_point =
            is_left ? &blocker.left_corridor_point : &blocker.right_corridor_point;
        for (int cell = 1; cell <= max_edge_cells; ++cell) {
            const Eigen::Vector3d probe(
                block_point.x() + perp.x() * side_sign * cell * res,
                block_point.y() + perp.y() * side_sign * cell * res, z);
            if (map.isKnownFree(probe.x(), probe.y(), probe.z(),
                                corridor_clearance)) {
                *edge_world = probe;
                *edge_visible = true;
                *corridor_point = probe;
                *corridor_known = segmentKnownAndClear(
                    map, probe,
                    Eigen::Vector3d(probe.x() + goal_dir.x() *
                                                     config.side_corridor_length_m,
                                    probe.y() + goal_dir.y() *
                                                     config.side_corridor_length_m,
                                    z),
                    corridor_clearance);
                break;
            }
        }
    }
    return blocker;
}

double MacroCandidateSearch::requiredClearance() const {
    return effectiveCandidateClearance(config_);
}

void MacroCandidateSearch::scoreObserved(MacroCandidate* candidate,
                                         const ObservedMap& map,
                                         const VehicleState& state,
                                         const Eigen::Vector3d& goal_world) const {
    const Eigen::Vector3d delta = candidate->position_world - state.position;
    const double dist = delta.norm();
    candidate->position_flu = worldToFlu(delta, state.yaw);

    // REAL observed-map reachability for every movement candidate (section
    // XIII/XV): SIDE, OBSERVE and GOAL_FRONTIER all run the observed
    // LocalPathSearch, and only FULL_GOAL_REACHED counts as reachable.  A
    // blocked straight line from current->candidate is NOT a reason to
    // reject an observation viewpoint — the 30 Hz planner can detour around
    // known obstacles, so the macro candidate validator must reflect that
    // real capability.  DIRECT and PREVIOUS_CONTINUATION keep the cheap
    // segment check (they are not committed movement targets of this tick).
    if (candidate->type == CandidateType::SIDE) {
        evaluateObservedReachability(candidate, map, state, true);
    } else if (candidate->type == CandidateType::OBSERVE ||
               candidate->type == CandidateType::GOAL_FRONTIER) {
        evaluateObservedReachability(candidate, map, state, false);
        if (candidate->full_goal_reached) {
            ++observe_diag_.full_local_count;
            if (candidate->source == "observe_retreat") {
                ++observe_diag_.retreat_full_count;
            }
        } else if (candidate->found_partial) {
            ++observe_diag_.partial_count;
        } else {
            ++observe_diag_.no_path_count;
        }
    } else {
        // DIRECT / PREVIOUS_CONTINUATION keep the cheap straight-segment
        // check, but with the SAME dynamic effective clearance the 30 Hz
        // planner uses (round 5): a continuation point that would not be
        // executable locally is not reported as reachable either.
        const double effective_clearance =
            effectiveCandidateClearance(config_);
        candidate->known_reachable = segmentKnownAndClear(
            map, state.position, candidate->position_world,
            effective_clearance);
        candidate->full_goal_reached = candidate->known_reachable;
        candidate->found_partial = candidate->known_reachable;
        candidate->observed_path_cost = dist;
        candidate->observed_path_length = dist;
        candidate->minimum_clearance = 0.0;
        if (dist > kEpsilon) {
            const Eigen::Vector3d dir = delta / dist;
            double min_clearance = std::numeric_limits<double>::infinity();
            const int n = std::max(
                1, static_cast<int>(std::ceil(
                       dist / std::max(0.02, config_.corridor_check_spacing_m))));
            for (int i = 0; i <= n; ++i) {
                const Eigen::Vector3d point =
                    state.position + dir * (dist * i / n);
                const double c = map.esdfValue(point.x(), point.y(), point.z());
                if (std::isfinite(c)) min_clearance = std::min(min_clearance, c);
            }
            candidate->minimum_clearance =
                std::isfinite(min_clearance) ? min_clearance : 0.0;
        }
    }
    // Goal progress: projection of the candidate offset onto the goal ray.
    const Eigen::Vector2d travel = goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len > kEpsilon) {
        candidate->goal_progress = delta.head<2>().dot(travel / travel_len);
    }
    candidate->observation_yaw_world = observationYawTowardGoal(
        candidate->position_world, goal_world, state.yaw);
    candidate->unknown_information_gain = estimateVisibleUnknownGain(
        map, candidate->position_world, candidate->observation_yaw_world);

    // Online candidate utility contains only causal quantities.  It is the
    // score used by the macro state machine; privileged_score remains a
    // diagnostics/auxiliary-training field and never selects an action.
    const double clearance = std::max(0.0, candidate->minimum_clearance);
    const double reachability = candidate->full_goal_reached ? 1.0 : 0.0;
    candidate->observed_score =
        2.0 * reachability +
        1.5 * candidate->unknown_information_gain +
        0.15 * candidate->goal_progress +
        0.10 * clearance -
        0.05 * candidate->observed_path_length;
}

void MacroCandidateSearch::evaluateObservedReachability(
    MacroCandidate* candidate,
    const ObservedMap& map,
    const VehicleState& state,
    bool adopt_partial) const {
    LocalPathSearch search;
    // OBSERVE / GOAL_FRONTIER use NO side bias: the reachability query must
    // reflect the true 30 Hz planner capability, not a preference.
    const Side bias_side =
        candidate->type == CandidateType::SIDE ? candidate->side : Side::NONE;
    // Round 5: the FULL-reachable A* uses the shared dynamic clearance at
    // the CURRENT vehicle speed — the exact boundary the 30 Hz planner
    // validates the executed trajectory with at the macro tick.  No more
    // fixed-clearance FULL (macro) vs dynamic-clearance execution (local)
    // mismatch.
    const LocalSearchConfig search_config =
        makeSearchConfig(config_, bias_side);
    const LocalPathResult path_result =
        search.search(map, state, candidate->position_world, search_config);
    candidate->full_goal_reached =
        path_result.status == LocalPathResult::Status::FULL_GOAL_REACHED;
    candidate->found_partial =
        path_result.status != LocalPathResult::Status::NO_PATH;
    candidate->known_reachable = candidate->full_goal_reached;
    candidate->observed_path_cost = path_result.path_cost;
    candidate->observed_path_length = path_result.path_cost;
    candidate->minimum_clearance = path_result.minimum_clearance;
    // A PARTIAL path is NEVER a reachable OBSERVE/GOAL_FRONTIER viewpoint
    // (it is a real movement terminal); only SIDE adopts the partial
    // terminal as its fallback position (existing behaviour).
    if (adopt_partial && candidate->found_partial &&
        !candidate->full_goal_reached) {
        candidate->position_world = path_result.terminal;
    }
}

double MacroCandidateSearch::estimateVisibleUnknownGain(
    const ObservedMap& map,
    const Eigen::Vector3d& position,
    double yaw_world) const {
    if (!map.esdfBuilt()) return 0.0;
    const int ray_count = std::max(3, config_.observe_visibility_ray_count);
    const double range = std::max(
        map.resolution(), config_.observe_visibility_range_m);
    const double half_fov = std::max(1.0, config_.observe_visibility_fov_deg) *
                           kPi / 360.0;
    const double spacing = std::max(map.resolution(), 0.05);
    const int steps = std::max(1, static_cast<int>(std::ceil(range / spacing)));
    const int iz = map.worldToGridInt(position).z();
    if (iz < 0 || iz >= map.gz()) return 0.0;

    auto occupancyAt = [&](const Eigen::Vector3d& point) -> std::uint8_t {
        const Eigen::Vector3i grid = map.worldToGridInt(point);
        if (grid.x() < 0 || grid.x() >= map.gx() ||
            grid.y() < 0 || grid.y() >= map.gy() ||
            grid.z() < 0 || grid.z() >= map.gz()) {
            return OCCUPIED;  // map boundary is not an observable frontier
        }
        const std::int64_t index =
            (static_cast<std::int64_t>(grid.x()) * map.gy() + grid.y()) *
                map.gz() + grid.z();
        return map.occupancy()[static_cast<size_t>(index)];
    };

    double gain = 0.0;
    for (int ray = 0; ray < ray_count; ++ray) {
        const double fraction = ray_count == 1 ? 0.5 :
            static_cast<double>(ray) / static_cast<double>(ray_count - 1);
        const double yaw = yaw_world + (2.0 * fraction - 1.0) * half_fov;
        // Project convention: yaw=0 faces world +Y.
        const Eigen::Vector3d direction(-std::sin(yaw), std::cos(yaw), 0.0);
        for (int step = 1; step <= steps; ++step) {
            const double distance = std::min(range, step * spacing);
            const std::uint8_t occupancy =
                occupancyAt(position + direction * distance);
            if (occupancy == OCCUPIED) break;
            if (occupancy == UNKNOWN) {
                // Only the first UNKNOWN on a visible ray counts.  The map
                // cannot causally assert visibility behind that frontier.
                gain += 1.0 - 0.5 * distance / range;
                break;
            }
        }
    }
    return gain / static_cast<double>(ray_count);
}

MacroCandidateSearch::MacroCandidateSearch(const MacroCandidateConfig& config)
    : config_(config) {}

std::vector<MacroCandidate> MacroCandidateSearch::makeObserveCandidates(
    const ObservedMap& map,
    const VehicleState& state,
    const Eigen::Vector3d& goal_world,
    int emit_budget) const {
    std::vector<MacroCandidate> raw;
    emit_budget = std::max(0, emit_budget);
    if (emit_budget == 0) return raw;
    const Eigen::Vector2d travel =
        goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len <= kEpsilon) return raw;
    const Eigen::Vector2d goal_dir = travel / travel_len;
    const Eigen::Vector2d left_world(-goal_dir.y(), goal_dir.x());
    const double z = state.position.z();

    const auto& laterals = config_.observe_lateral_distances_m;
    const auto& forwards = config_.observe_forward_distances_m;
    if (laterals.empty()) return raw;
    const double min_move = config_.min_observe_move_distance_m;
    const double max_move = config_.max_observe_move_distance_m;
    // Round 5: the lattice endpoint must be known-free with the SAME
    // dynamic effective clearance the 30 Hz planner validates with, never
    // the fixed base clearance.
    const double clear = effectiveCandidateClearance(config_);
    // Lateral x forward x {LEFT, RIGHT}: a small local viewpoint lattice
    // (section XV).  Forward 0 still yields near-side viewpoints at
    // lateral offsets; every candidate is a real world position the drone
    // may move to.
    for (double fwd : forwards) {
        for (double lat : laterals) {
            for (int side_sign = -1; side_sign <= 1; side_sign += 2) {
                const double lat_signed = side_sign * lat;
                const Eigen::Vector3d pos(
                    state.position.x() + goal_dir.x() * fwd +
                        left_world.x() * lat_signed,
                    state.position.y() + goal_dir.y() * fwd +
                        left_world.y() * lat_signed,
                    z);
                const double move_dist = (pos - state.position).norm();
                ++observe_diag_.raw_candidate_count;
                // Cheap endpoint filter (section XII): distance bounds,
                // known, free with the SAME clearance the 30 Hz planner
                // uses (clearance_m — never a second inflation).
                if (move_dist < min_move) {
                    ++observe_diag_.reject_min_distance;
                    continue;
                }
                if (move_dist > max_move) {
                    ++observe_diag_.reject_max_distance;
                    continue;
                }
                if (!map.isKnown(pos.x(), pos.y(), pos.z())) {
                    ++observe_diag_.reject_unknown;
                    continue;
                }
                if (!map.isKnownFree(pos.x(), pos.y(), pos.z(), clear)) {
                    ++observe_diag_.reject_endpoint_clearance;
                    continue;
                }
                ++observe_diag_.endpoint_known_free_count;
                MacroCandidate c;
                c.type = CandidateType::OBSERVE;
                c.position_world = pos;
                if (lat_signed > 1e-3) {
                    c.side = Side::LEFT;
                    c.source = "observe_left";
                } else if (lat_signed < -1e-3) {
                    c.side = Side::RIGHT;
                    c.source = "observe_right";
                } else {
                    c.side = Side::NONE;
                    c.source = "observe_center";
                }
                const double ec = map.esdfValue(pos.x(), pos.y(), pos.z());
                c.minimum_clearance = std::isfinite(ec) ? ec : 0.0;
                c.observation_yaw_world = observationYawTowardGoal(
                    pos, goal_world, state.yaw);
                // Causal FOV-aware proxy used before the expensive FULL A*.
                c.unknown_information_gain = estimateVisibleUnknownGain(
                    map, pos, c.observation_yaw_world);
                raw.push_back(c);
            }
        }
    }

    // Cheap rank: information gain desc, endpoint clearance desc, distance
    // asc.  Only the top candidates are kept; the FULL LocalPathSearch
    // budget (max_viewpoint_searches_per_tick) limits how many of them
    // actually get the A* in scoreObserved.
    std::sort(raw.begin(), raw.end(),
              [&](const MacroCandidate& a, const MacroCandidate& b) {
                  const double ia = a.unknown_information_gain;
                  const double ib = b.unknown_information_gain;
                  if (std::abs(ia - ib) > 1e-6) return ia > ib;
                  const double ca = a.minimum_clearance;
                  const double cb = b.minimum_clearance;
                  if (std::abs(ca - cb) > 1e-6) return ca > cb;
                  const double da =
                      (a.position_world - state.position).norm();
                  const double db =
                      (b.position_world - state.position).norm();
                  return da < db;
              });
    const int pool_limit =
        std::min(config_.max_viewpoint_candidates, static_cast<int>(raw.size()));
    const int emit_limit =
        std::min(emit_budget, pool_limit);
    raw.resize(static_cast<size_t>(emit_limit));
    observe_diag_.lattice_candidate_count = emit_limit;
    return raw;
}

MacroCandidate MacroCandidateSearch::makeSideCandidate(
    const ObservedMap& map,
    const VehicleState& state,
    const Eigen::Vector3d& goal_world,
    const GoalBlocker& blocker,
    Side side,
    double offset_m) const {
    MacroCandidate candidate;
    candidate.type = CandidateType::SIDE;
    candidate.side = side;
    const bool is_left = side == Side::LEFT;
    const Eigen::Vector3d edge =
        is_left ? blocker.left_edge_world : blocker.right_edge_world;
    const bool corridor_known =
        is_left ? blocker.left_corridor_known : blocker.right_corridor_known;
    const Eigen::Vector2d travel = goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len <= kEpsilon) return candidate;
    const Eigen::Vector2d goal_dir = travel / travel_len;
    if (corridor_known) {
        // Advance along the corridor as far as known-free allows, expressing
        // long-term bypass intent (never the fixed 0.7 m offset).
        const double z = state.position.z();
        Eigen::Vector3d advance = edge;
        for (double d = offset_m; d <= config_.side_corridor_length_m + 1.0e-6;
             d += config_.candidate_spacing_m) {
            const Eigen::Vector3d probe(
                edge.x() + goal_dir.x() * d,
                edge.y() + goal_dir.y() * d, z);
            if (segmentKnownAndClear(
                    map, edge, probe,
                    std::max(config_.side_corridor_radius_m,
                             requiredClearance()))) {
                advance = probe;
            } else {
                break;
            }
        }
        candidate.position_world = advance;
    } else if (blocker.left_edge_visible || blocker.right_edge_visible) {
        candidate.position_world = edge;
    } else {
        // Neither edge visible: no usable side candidate.
        candidate.position_world = state.position;
    }
    candidate.source = is_left ? "side_left" : "side_right";
    return candidate;
}

std::vector<MacroCandidate> MacroCandidateSearch::makeFrontierCandidates(
    const ObservedMap& map,
    const VehicleState& state,
    const Eigen::Vector3d& goal_world,
    int remaining_searches) const {
    std::vector<MacroCandidate> candidates;
    if (remaining_searches <= 0) return candidates;
    const double res = map.resolution();
    const double z = state.position.z();
    const Eigen::Vector2d travel = goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len <= kEpsilon) return candidates;
    const Eigen::Vector2d goal_dir = travel / travel_len;
    const double cone = config_.goal_frontier_cone_deg * kPi / 180.0;
    const double cos_cone = std::cos(cone);

    struct FrontierCell {
        Eigen::Vector3d world;
        double standoff;
        bool is_safe_prefix = false;
    };
    std::vector<FrontierCell> cells;
    cells.reserve(512);
    const Eigen::Vector3d origin = map.origin();
    const int iz = static_cast<int>(std::floor(
        (z - origin.z()) / res));
    // First emit the farthest short segment on the goal ray that is fully
    // observed and clear at the execution clearance.  Unlike a grid A*
    // terminal at the distant map frontier, this is a straight, conservative
    // prefix that the spline planner can execute without cutting a corner.
    const double clear = effectiveCandidateClearance(config_);
    const double prefix_horizon = std::min(
        std::min(travel_len, config_.edge_search_radius_m),
        config_.frontier_prefix_horizon_m);
    Eigen::Vector3d safe_prefix = state.position;
    const double prefix_step = std::max(res, 0.10);
    for (double d = config_.min_observe_move_distance_m;
         d <= prefix_horizon + 1.0e-6; d += prefix_step) {
        const Eigen::Vector3d point(
            state.position.x() + goal_dir.x() * d,
            state.position.y() + goal_dir.y() * d, z);
        if (!segmentKnownAndClear(map, state.position, point, clear)) break;
        safe_prefix = point;
    }
    if ((safe_prefix - state.position).norm() >=
        config_.min_observe_move_distance_m) {
        cells.push_back({safe_prefix, 0.0, true});
    }
    for (int ix = 0; ix < map.gx(); ++ix) {
        for (int iy = 0; iy < map.gy(); ++iy) {
            const std::int64_t idx =
                (static_cast<std::int64_t>(ix) * map.gy() + iy) * map.gz() + iz;
            if (idx < 0 || idx >= static_cast<std::int64_t>(map.occupancy().size())) {
                continue;
            }
            if (map.occupancy()[static_cast<size_t>(idx)] != FREE) continue;
            const double wx = origin.x() + (static_cast<double>(ix) + 0.5) * res;
            const double wy = origin.y() + (static_cast<double>(iy) + 0.5) * res;
            const Eigen::Vector2d offset(wx - state.position.x(),
                                         wy - state.position.y());
            const double dist = offset.norm();
            if (dist <= config_.min_observe_move_distance_m ||
                dist > config_.edge_search_radius_m) {
                continue;
            }
            if (dist > kEpsilon) {
                const double cos_angle = offset.dot(goal_dir) / dist;
                if (cos_angle < cos_cone) continue;
            }
            // Adjacent to unknown?
            bool adjacent_unknown = false;
            for (int dx = -1; dx <= 1 && !adjacent_unknown; ++dx) {
                for (int dy = -1; dy <= 1 && !adjacent_unknown; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    const int nix = ix + dx;
                    const int niy = iy + dy;
                    if (nix < 0 || nix >= map.gx() || niy < 0 || niy >= map.gy()) {
                        continue;
                    }
                    const std::int64_t nidx =
                        (static_cast<std::int64_t>(nix) * map.gy() + niy) *
                            map.gz() +
                        iz;
                    if (map.occupancy()[static_cast<size_t>(nidx)] == UNKNOWN) {
                        adjacent_unknown = true;
                        break;
                    }
                }
            }
            if (!adjacent_unknown) continue;
            cells.push_back({Eigen::Vector3d(wx, wy, z),
                             config_.frontier_standoff_m, false});
        }
    }
    observe_diag_.raw_candidate_count += static_cast<int>(cells.size());
    // Cluster by angular bin and keep the best cell per bin.
    const double bin_rad = 10.0 * kPi / 180.0;
    std::vector<std::pair<int, FrontierCell>> best_per_bin;
    for (const FrontierCell& cell : cells) {
        const Eigen::Vector2d offset(cell.world.x() - state.position.x(),
                                     cell.world.y() - state.position.y());
        const double rel = std::atan2(
            goal_dir.x() * offset.y() - goal_dir.y() * offset.x(),
            offset.dot(goal_dir));
        const int bin = static_cast<int>(std::floor(rel / bin_rad));
        const double score = offset.dot(goal_dir);
        bool inserted = false;
        for (auto& entry : best_per_bin) {
            if (entry.first == bin) {
                const double old_score =
                    entry.second.world.head<2>().dot(goal_dir);
                if ((cell.is_safe_prefix && !entry.second.is_safe_prefix) ||
                    (cell.is_safe_prefix == entry.second.is_safe_prefix &&
                     ((cell.is_safe_prefix && score < old_score) ||
                      (!cell.is_safe_prefix && score > old_score)))) {
                    entry.second = cell;
                }
                inserted = true;
                break;
            }
        }
        if (!inserted) best_per_bin.emplace_back(bin, cell);
    }
    std::sort(best_per_bin.begin(), best_per_bin.end(),
              [&](const std::pair<int, FrontierCell>& a,
                  const std::pair<int, FrontierCell>& b) {
                   if (a.second.is_safe_prefix != b.second.is_safe_prefix) {
                       return a.second.is_safe_prefix;
                   }
                   const double pa = a.second.world.head<2>().dot(goal_dir);
                   const double pb = b.second.world.head<2>().dot(goal_dir);
                   return pa > pb;
              });
    const int limit =
        std::min(config_.max_frontier_candidates,
                 static_cast<int>(best_per_bin.size()));
    for (int i = 0; i < limit; ++i) {
        const FrontierCell& cell = best_per_bin[static_cast<size_t>(i)].second;
        // Pull back into known space.
        const Eigen::Vector2d dir = (cell.world.head<2>() -
                                     state.position.head<2>())
                                        .normalized();
        // A freshly integrated map can expose a frontier only a few cells
        // ahead.  Preserve a nonzero forward prefix instead of pulling that
        // endpoint back to (or behind) the vehicle.
        const double standoff = std::min(
            cell.standoff,
            std::max(0.0, (cell.world - state.position).norm() -
                              config_.min_observe_move_distance_m));
        const Eigen::Vector3d pulled(
            cell.world.x() - dir.x() * standoff,
            cell.world.y() - dir.y() * standoff, z);
        // Endpoint sits on the KNOWN-FREE side of the frontier with the
        // SAME dynamic effective clearance the 30 Hz planner uses (round
        // 5) — never a narrower fixed boundary.
        if (!map.isKnownFree(pulled.x(), pulled.y(), pulled.z(),
                             effectiveCandidateClearance(config_))) {
            continue;
        }
        MacroCandidate candidate;
        candidate.type = CandidateType::GOAL_FRONTIER;
        // Classify the frontier by its lateral sign relative to the goal
        // ray (section XVI): +left = LEFT, -left = RIGHT, |lateral| small
        // -> NONE (central).  This lets OBSERVE_MOVE strictly follow the
        // current observation side.
        const Eigen::Vector2d rel_frontier =
            pulled.head<2>() - state.position.head<2>();
        const double lateral =
            (-goal_dir.y()) * rel_frontier.x() + goal_dir.x() * rel_frontier.y();
        const double side_eps = config_.candidate_spacing_m;
        if (lateral > side_eps) {
            candidate.side = Side::LEFT;
        } else if (lateral < -side_eps) {
            candidate.side = Side::RIGHT;
        } else {
            candidate.side = Side::NONE;
        }
        candidate.position_world = pulled;
        candidate.source = cell.is_safe_prefix ? "goal_safe_prefix"
                                                : "goal_frontier";
        candidate.observation_yaw_world = observationYawTowardGoal(
            pulled, goal_world, state.yaw);
        candidate.unknown_information_gain = estimateVisibleUnknownGain(
            map, pulled, candidate.observation_yaw_world);
        candidates.push_back(candidate);
    }
    // Preserve the short executable prefix before optional distant frontier
    // candidates when applying the bounded FULL-search budget.
    if (static_cast<int>(candidates.size()) > remaining_searches) {
        std::sort(candidates.begin(), candidates.end(),
                  [&](const MacroCandidate& a, const MacroCandidate& b) {
                       const bool a_prefix = a.source == "goal_safe_prefix";
                       const bool b_prefix = b.source == "goal_safe_prefix";
                       if (a_prefix != b_prefix) return a_prefix;
                       const double da =
                           (a.position_world - state.position).norm();
                       const double db =
                           (b.position_world - state.position).norm();
                       if (a_prefix) return da > db;
                       return a.unknown_information_gain >
                              b.unknown_information_gain;
                   });
        candidates.resize(static_cast<size_t>(remaining_searches));
    }
    observe_diag_.frontier_candidate_count =
        static_cast<int>(candidates.size());
    return candidates;
}

std::vector<MacroCandidate> MacroCandidateSearch::makeRetreatCandidates(
    const ObservedMap& map,
    const VehicleState& state,
    const Eigen::Vector3d& goal_world,
    int budget) const {
    std::vector<MacroCandidate> raw;
    if (budget <= 0) return raw;
    const Eigen::Vector2d travel =
        goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len <= kEpsilon) return raw;
    const Eigen::Vector2d goal_dir = travel / travel_len;
    const Eigen::Vector2d left_world(-goal_dir.y(), goal_dir.x());
    const double z = state.position.z();
    const auto& distances = config_.retreat_distances_m;
    if (distances.empty()) return raw;
    const double min_move = config_.min_observe_move_distance_m;
    const double max_move = config_.max_observe_move_distance_m;
    // Round 5: the retreat endpoint must be known-free with the SAME
    // dynamic effective clearance as the 30 Hz planner (and FULL-verified
    // by the observed A* in scoreObserved).  Unknown is never traversable.
    const double clear = effectiveCandidateClearance(config_);
    const double lateral = config_.retreat_lateral_m;
    // Backward x {-lateral, 0, +lateral}: known-free points BEHIND the
    // drone.  Moderate negative goal progress is allowed — the whole point
    // is a safe retreat when forward observation is stuck — but the
    // endpoint must be known-free with the unified clearance and the FULL
    // observed LocalPathSearch (in scoreObserved) must reach it.
    for (double dist : distances) {
        for (int side_sign = -1; side_sign <= 1; ++side_sign) {
            const Eigen::Vector3d pos(
                state.position.x() - goal_dir.x() * dist +
                    left_world.x() * (side_sign * lateral),
                state.position.y() - goal_dir.y() * dist +
                    left_world.y() * (side_sign * lateral),
                z);
            const double move_dist = (pos - state.position).norm();
            if (move_dist < min_move || move_dist > max_move) continue;
            if (!map.isKnown(pos.x(), pos.y(), pos.z())) continue;
            if (!map.isKnownFree(pos.x(), pos.y(), pos.z(), clear)) continue;
            MacroCandidate candidate;
            candidate.type = CandidateType::OBSERVE;
            candidate.position_world = pos;
            candidate.source = "observe_retreat";
            if (side_sign > 1e-3) {
                candidate.side = Side::LEFT;
            } else if (side_sign < -1e-3) {
                candidate.side = Side::RIGHT;
            } else {
                candidate.side = Side::NONE;
            }
            const double ec = map.esdfValue(pos.x(), pos.y(), pos.z());
            candidate.minimum_clearance = std::isfinite(ec) ? ec : 0.0;
            candidate.observation_yaw_world = observationYawTowardGoal(
                pos, goal_world, state.yaw);
            candidate.unknown_information_gain = estimateVisibleUnknownGain(
                map, pos, candidate.observation_yaw_world);
            raw.push_back(candidate);
        }
    }
    // Cheap rank: information gain desc, clearance desc, distance asc
    // (same tie-breaks as the forward lattice).
    std::sort(raw.begin(), raw.end(),
              [&](const MacroCandidate& a, const MacroCandidate& b) {
                  const double ia = a.unknown_information_gain;
                  const double ib = b.unknown_information_gain;
                  if (std::abs(ia - ib) > 1e-6) return ia > ib;
                  const double ca = a.minimum_clearance;
                  const double cb = b.minimum_clearance;
                  if (std::abs(ca - cb) > 1e-6) return ca > cb;
                  return (a.position_world - state.position).norm() <
                         (b.position_world - state.position).norm();
              });
    if (static_cast<int>(raw.size()) > budget) {
        raw.resize(static_cast<size_t>(budget));
    }
    observe_diag_.retreat_candidate_count = static_cast<int>(raw.size());
    return raw;
}

std::vector<MacroCandidate> MacroCandidateSearch::generateCandidates(
    const ObservedMap& map,
    const VehicleState& state,
    const Eigen::Vector3d& goal_world,
    const GoalBlocker& blocker,
    const Eigen::Vector3d* prev_candidate_world) const {
    std::vector<MacroCandidate> candidates;
    const Eigen::Vector2d travel = goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    const double z = state.position.z();
    // Per-tick observation diagnostics (mutable member, reset here).
    observe_diag_ = ObserveDiagnostics();

    // P3: a few known-free recovery (retreat) viewpoints behind the drone
    // are generated regardless of the forward lattice outcome.  They only
    // become the ACTIVE OBSERVE_MOVE when rotation produced no usable
    // forward FULL viewpoint (the Python macro ranks "observe_retreat"
    // below every forward candidate), which is exactly the
    // observe_deadlock case — raw candidates exist but no FULL forward
    // viewpoint -> the drone moves to a safe retreat instead of rotating
    // forever.  Negative goal progress is allowed; the endpoint must be
    // known-free with the unified clearance and FULL-reachable via the
    // real observed LocalPathSearch (never a straight-corridor proxy).
    if (travel_len > kEpsilon) {
        const Eigen::Vector2d goal_dir = travel / travel_len;

        // ── Direct candidate ───────────────────────────────────────
        MacroCandidate direct;
        direct.type = CandidateType::DIRECT;
        direct.side = Side::NONE;
        direct.position_world = Eigen::Vector3d(
            state.position.x() +
                goal_dir.x() * std::min(travel_len, config_.lookahead_distance_m),
            state.position.y() +
                goal_dir.y() * std::min(travel_len, config_.lookahead_distance_m),
            z);
        direct.source = "direct";
        candidates.push_back(direct);

        // ── Side candidates (only from observable edges/corridors) ──
        if (blocker.found) {
            if (blocker.left_edge_visible || blocker.left_corridor_known) {
                MacroCandidate left = makeSideCandidate(
                    map, state, goal_world, blocker, Side::LEFT, 0.0);
                if (left.position_world != state.position) {
                    candidates.push_back(left);
                }
            }
            if (blocker.right_edge_visible || blocker.right_corridor_known) {
                MacroCandidate right = makeSideCandidate(
                    map, state, goal_world, blocker, Side::RIGHT, 0.0);
                if (right.position_world != state.position) {
                    candidates.push_back(right);
                }
            }
        }

        // ── Active observation viewpoints (section XV) ──────────────
        // A lateral x forward lattice on BOTH sides replaces the old two
        // fixed ±observe_step_m probes.  The cheap endpoint filter runs
        // first; only the top candidates (by information gain / clearance /
        // distance) get the FULL observed LocalPathSearch in scoreObserved.
        // A viewpoint never needs to be straight-line visible from here —
        // the A* can detour around known obstacles exactly like the 30 Hz
        // planner, and a blocked direct corridor is NOT a rejection.
        if (config_.max_viewpoint_searches_per_tick > 0) {
        const int frontier_reserve = std::min(
            config_.max_frontier_candidates,
            config_.min_frontier_searches_per_tick);
        const int retreat_budget =
            std::max(0, config_.retreat_searches_per_tick);
        const int lattice_budget = std::max(
            0, config_.max_viewpoint_searches_per_tick - frontier_reserve -
                   retreat_budget);
        std::vector<MacroCandidate> observe_candidates =
            makeObserveCandidates(map, state, goal_world, lattice_budget);
        const int lattice_count =
            static_cast<int>(observe_candidates.size());
        candidates.insert(candidates.end(), observe_candidates.begin(),
                          observe_candidates.end());

        // ── Goal-directed frontiers (known-free standoff) ──────────
        const int remaining =
            std::max(frontier_reserve,
                     config_.max_viewpoint_searches_per_tick -
                         lattice_count - retreat_budget);
        std::vector<MacroCandidate> frontiers =
            makeFrontierCandidates(map, state, goal_world, remaining);
        candidates.insert(candidates.end(), frontiers.begin(), frontiers.end());

        // ── P3 known-free recovery viewpoints (observe_retreat) ─────
        std::vector<MacroCandidate> retreats =
            makeRetreatCandidates(map, state, goal_world, retreat_budget);
        candidates.insert(candidates.end(), retreats.begin(), retreats.end());
        }
    }

    // ── Previous strategic continuation ────────────────────────────
    if (prev_candidate_world != nullptr && prev_candidate_world->allFinite()) {
        MacroCandidate continuation;
        continuation.type = CandidateType::PREVIOUS_CONTINUATION;
        continuation.side = Side::NONE;
        continuation.position_world = *prev_candidate_world;
        continuation.source = "previous_continuation";
        candidates.push_back(continuation);
    }

    // Observed-side scoring for every candidate.
    for (MacroCandidate& candidate : candidates) {
        if (candidate.position_world == state.position) continue;
        scoreObserved(&candidate, map, state, goal_world);
    }
    return candidates;
}

}  // namespace il_dataset
