#include "il_dataset/local_planner/macro_candidate_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

#include "il_dataset/local_planner/observed_map.hpp"

namespace il_dataset {

namespace {

constexpr double kEpsilon = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;

bool corridorIsKnownFree(const ObservedMap& map,
                         const Eigen::Vector3d& start,
                         const Eigen::Vector3d& end,
                         double radius,
                         double spacing,
                         double min_clearance) {
    const double length = (end - start).norm();
    if (length <= kEpsilon) {
        return map.isKnownFree(start.x(), start.y(), start.z(),
                               radius + min_clearance);
    }
    const Eigen::Vector3d direction = (end - start) / length;
    const int samples =
        std::max(1, static_cast<int>(std::ceil(length / std::max(0.02, spacing))));
    for (int i = 0; i <= samples; ++i) {
        const Eigen::Vector3d point = start + direction *
                                                 (length * i / samples);
        if (!map.isKnownFree(point.x(), point.y(), point.z(),
                             radius + min_clearance)) {
            return false;
        }
    }
    return true;
}

Eigen::Vector3d worldToFlu(const Eigen::Vector3d& world_delta, double yaw) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    return Eigen::Vector3d(c * world_delta.x() + s * world_delta.y(),
                           -s * world_delta.x() + c * world_delta.y(),
                           world_delta.z());
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
    auto occupancy_at = [&](int ix, int iy) -> std::uint8_t {
        if (ix < 0 || ix >= gx || iy < 0 || iy >= gy || z_index < 0 ||
            z_index >= map.gz()) {
            return UNKNOWN;
        }
        const std::int64_t idx =
            (static_cast<std::int64_t>(ix) * gy + iy) * map.gz() + z_index;
        return map.occupancy()[static_cast<size_t>(idx)];
    };
    auto known_free_at = [&](int ix, int iy) {
        if (std::abs(ix - seed_ix) > region_radius_cells ||
            std::abs(iy - seed_iy) > region_radius_cells) {
            return true;  // treat far cells as free to bound the region
        }
        const double wx = origin.x() + (static_cast<double>(ix) + 0.5) * res;
        const double wy = origin.y() + (static_cast<double>(iy) + 0.5) * res;
        return map.isKnownFree(wx, wy, z_world, block_clearance);
    };
    const std::uint8_t seed_state = occupancy_at(seed_ix, seed_iy);
    comp.blocked_by_known = (seed_state == OCCUPIED);

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
        if (known_free_at(ix, iy)) continue;
        for (int n = 0; n < 8; ++n) {
            const int nxi = ix + di[n];
            const int nyi = iy + dj[n];
            if (nxi < 0 || nxi >= gx || nyi < 0 || nyi >= gy) continue;
            const int nindex = nyi * gx + nxi;
            if (visited[static_cast<size_t>(nindex)] != 0) continue;
            if (!known_free_at(nxi, nyi)) {
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
    const Eigen::Vector2d travel = goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len <= kEpsilon) return blocker;
    const Eigen::Vector2d goal_dir = travel / travel_len;

    const double z = state.position.z();
    const double max_walk = std::min(travel_len, config.edge_search_radius_m);
    Eigen::Vector3d block_point{Eigen::Vector3d::Zero()};
    bool found = false;
    int block_ix = 0;
    int block_iy = 0;
    int block_iz = 0;
    for (double d = res; d <= max_walk + 1.0e-6; d += res) {
        const Eigen::Vector3d point(
            state.position.x() + goal_dir.x() * d,
            state.position.y() + goal_dir.y() * d, z);
        if (!map.isKnownFree(point.x(), point.y(), point.z(), 0.0)) {
            block_point = point;
            found = true;
            const Eigen::Vector3i g = map.worldToGridInt(point);
            block_ix = g.x();
            block_iy = g.y();
            block_iz = g.z();
            break;
        }
    }
    if (!found) return blocker;
    blocker.found = true;

    const int region_radius =
        std::max(8, static_cast<int>(std::ceil(config.edge_search_radius_m / res)));
    const BlockComponent comp =
        floodBlockedComponent(map, block_ix, block_iy, block_iz,
                              config.min_candidate_clearance_m,
                              region_radius);
    blocker.component_id = comp.count > 0 ? 0 : -1;
    blocker.blocked_by_known = comp.blocked_by_known;
    if (comp.count > 0) {
        blocker.centroid = comp.centroid;
        blocker.extent = comp.extent;
    }

    // ── Edge visibility + known corridor on each side ──────────────
    const Eigen::Vector2d perp(-goal_dir.y(), goal_dir.x());  // + = left
    const double corridor_clearance = config.side_corridor_radius_m;
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
                *corridor_known = corridorIsKnownFree(
                    map, probe,
                    Eigen::Vector3d(probe.x() + goal_dir.x() *
                                                     config.side_corridor_length_m,
                                    probe.y() + goal_dir.y() *
                                                     config.side_corridor_length_m,
                                    z),
                    corridor_clearance, config.corridor_check_spacing_m,
                    config.min_candidate_clearance_m);
                break;
            }
        }
    }
    return blocker;
}

void MacroCandidateSearch::scoreObserved(MacroCandidate* candidate,
                                         const ObservedMap& map,
                                         const VehicleState& state,
                                         const Eigen::Vector3d& goal_world) const {
    const Eigen::Vector3d delta = candidate->position_world - state.position;
    const double dist = delta.norm();
    candidate->position_flu = worldToFlu(delta, state.yaw);
    candidate->observed_path_cost = dist;
    candidate->known_reachable =
        corridorIsKnownFree(map, state.position, candidate->position_world,
                            config_.min_candidate_clearance_m,
                            config_.corridor_check_spacing_m,
                            config_.min_candidate_clearance_m);
    // Minimum clearance along the candidate corridor.
    double min_clearance = std::numeric_limits<double>::infinity();
    if (dist > kEpsilon) {
        const Eigen::Vector3d dir = delta / dist;
        const int n = std::max(
            1, static_cast<int>(std::ceil(
                   dist / std::max(0.02, config_.corridor_check_spacing_m))));
        for (int i = 0; i <= n; ++i) {
            const Eigen::Vector3d point =
                state.position + dir * (dist * i / n);
            const double c = map.esdfValue(point.x(), point.y(), point.z());
            if (std::isfinite(c)) min_clearance = std::min(min_clearance, c);
        }
    }
    candidate->minimum_clearance =
        std::isfinite(min_clearance) ? min_clearance : 0.0;
    // Goal progress: projection of the candidate offset onto the goal ray.
    const Eigen::Vector2d travel = goal_world.head<2>() - state.position.head<2>();
    const double travel_len = travel.norm();
    if (travel_len > kEpsilon) {
        candidate->goal_progress = delta.head<2>().dot(travel / travel_len);
    }
    // Information gain: unknown cells in a disk around the candidate.
    const double z = state.position.z();
    const int radius_cells = std::max(
        2, static_cast<int>(std::ceil(0.8 / std::max(0.05, map.resolution()))));
    int unknown_count = 0;
    int total = 0;
    const Eigen::Vector3i center = map.worldToGridInt(candidate->position_world);
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
        for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
            const int ix = center.x() + dx;
            const int iy = center.y() + dy;
            if (ix < 0 || ix >= map.gx() || iy < 0 || iy >= map.gy()) continue;
            const double wx = map.origin().x() + (static_cast<double>(ix) + 0.5) *
                                                     map.resolution();
            const double wy = map.origin().y() + (static_cast<double>(iy) + 0.5) *
                                                     map.resolution();
            if (std::hypot(wx - candidate->position_world.x(),
                           wy - candidate->position_world.y()) > 0.8) {
                continue;
            }
            ++total;
            const Eigen::Vector3i g(ix, iy, center.z());
            const std::int64_t idx =
                (static_cast<std::int64_t>(g.x()) * map.gy() + g.y()) *
                    map.gz() +
                g.z();
            if (idx >= 0 &&
                idx < static_cast<std::int64_t>(map.occupancy().size()) &&
                map.occupancy()[static_cast<size_t>(idx)] == UNKNOWN) {
                ++unknown_count;
            }
        }
    }
    candidate->unknown_information_gain =
        total > 0 ? static_cast<double>(unknown_count) / total : 0.0;
}

MacroCandidateSearch::MacroCandidateSearch(const MacroCandidateConfig& config)
    : config_(config) {}

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
            if (corridorIsKnownFree(map, edge, probe,
                                    config_.side_corridor_radius_m,
                                    config_.corridor_check_spacing_m,
                                    config_.min_candidate_clearance_m)) {
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
    const Eigen::Vector3d& goal_world) const {
    std::vector<MacroCandidate> candidates;
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
    };
    std::vector<FrontierCell> cells;
    cells.reserve(512);
    const Eigen::Vector3d origin = map.origin();
    const int iz = static_cast<int>(std::floor(
        (z - origin.z()) / res));
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
            if (dist <= 1.0 || dist > config_.edge_search_radius_m) continue;
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
                             config_.frontier_standoff_m});
        }
    }
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
                if (score > entry.second.world.head<2>().dot(goal_dir)) {
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
                  return a.second.world.head<2>().dot(goal_dir) >
                         b.second.world.head<2>().dot(goal_dir);
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
        const Eigen::Vector3d pulled(
            cell.world.x() - dir.x() * cell.standoff,
            cell.world.y() - dir.y() * cell.standoff, z);
        if (!map.isKnownFree(pulled.x(), pulled.y(), pulled.z(),
                             config_.min_candidate_clearance_m)) {
            continue;
        }
        MacroCandidate candidate;
        candidate.type = CandidateType::GOAL_FRONTIER;
        candidate.side = Side::NONE;
        candidate.position_world = pulled;
        candidate.source = "goal_frontier";
        candidates.push_back(candidate);
    }
    return candidates;
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

        // ── Observe candidates (short known-safe moves, mostly yaw) ──
        const Eigen::Vector2d perp(-goal_dir.y(), goal_dir.x());
        for (int side_sign = -1; side_sign <= 1; side_sign += 2) {
            const Side side = side_sign > 0 ? Side::LEFT : Side::RIGHT;
            const Eigen::Vector3d probe(
                state.position.x() + perp.x() * side_sign * config_.observe_step_m,
                state.position.y() + perp.y() * side_sign * config_.observe_step_m,
                z);
            MacroCandidate observe;
            observe.type = CandidateType::OBSERVE;
            observe.side = side;
            observe.position_world =
                map.isKnownFree(probe.x(), probe.y(), probe.z(),
                                config_.min_candidate_clearance_m)
                    ? probe
                    : state.position;
            observe.source = side == Side::LEFT ? "observe_left" : "observe_right";
            candidates.push_back(observe);
        }

        // ── Goal-directed frontiers ───────────────────────────────
        std::vector<MacroCandidate> frontiers =
            makeFrontierCandidates(map, state, goal_world);
        candidates.insert(candidates.end(), frontiers.begin(), frontiers.end());
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
