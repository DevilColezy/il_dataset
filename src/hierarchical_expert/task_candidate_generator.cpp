#include "il_dataset/hierarchical_expert/task_candidate_generator.hpp"

#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

namespace {

/// Distance from a point to the finite segment a→b (exact).
inline double distToSeg(const Vec2d& p, const Vec2d& a, const Vec2d& b) {
    const Vec2d ab = b - a;
    const double len2 = ab.squaredNorm();
    if (len2 <= 1e-12) return (p - a).norm();
    const double t = clamp((p - a).dot(ab) / len2, 0.0, 1.0);
    return (p - (a + ab * t)).norm();
}

/// Signed distance from a point to the segment (negative = left side in
/// the expert frame; +X right / +Y up, so positive = LEFT).
inline double signedDistToSeg(const Vec2d& p, const Vec2d& a, const Vec2d& b) {
    const Vec2d ab = b - a;
    const double len2 = ab.squaredNorm();
    if (len2 <= 1e-12) return (p - a).norm();
    const double t = clamp((p - a).dot(ab) / len2, 0.0, 1.0);
    const Vec2d proj = a + ab * t;
    const double d = (p - proj).norm();
    const double s = cross2(ab, p - a);
    return s >= 0.0 ? d : -d;
}

}  // namespace

std::array<bool, kNumTaskGeomTypes> TaskCandidateGenerator::feasibilityFor(
    const BlueprintScene& scene, const SceneGeometryCache& geo) const {
    (void)scene;
    std::array<bool, kNumTaskGeomTypes> m;
    m.fill(false);
    const size_t n_obs = geo.obstacleCenters().size();
    const bool has_large = !geo.largeObstacles().empty();
    const bool has_narrow = !geo.narrowPassages().empty();

    // CLEAR is always feasible (empty scenes must be able to produce it).
    m[static_cast<size_t>(TaskGeomType::CLEAR)] = true;
    m[static_cast<size_t>(TaskGeomType::LOCAL_AVOIDANCE)] = n_obs >= 1;
    m[static_cast<size_t>(TaskGeomType::OFFSET_AVOIDANCE)] = n_obs >= 1;
    m[static_cast<size_t>(TaskGeomType::MULTI_OBSTACLE)] = n_obs >= 2;
    m[static_cast<size_t>(TaskGeomType::LARGE_OCCLUSION)] = has_large;
    // CHICANE proxy needs >= 4 obstacles with alternating sides; with fewer
    // the sampler would only waste attempts.
    m[static_cast<size_t>(TaskGeomType::CHICANE)] = n_obs >= 4;
    m[static_cast<size_t>(TaskGeomType::NARROW_BUT_PLANNABLE)] = has_narrow;
    // LONG_DETOUR requires a blocker / large obstacle on the path.
    m[static_cast<size_t>(TaskGeomType::LONG_DETOUR)] = has_large;
    return m;
}

TaskGeomType TaskCandidateGenerator::classifyGeometry(
    const SceneGeometryCache& geo, const Vec2d& start,
    const Vec2d& goal) const {
    const Vec2d axis = goal - start;
    const double len = std::max(1e-6, axis.norm());
    const double chw = corridorHalfWidth();
    const double free_clr = cfg_.free_cell_surface_clearance_m;
    const auto& centers = geo.obstacleCenters();
    const auto& radii = geo.obstacleRadii();
    const auto& large = geo.largeObstacles();

    int near_count = 0;
    bool straight_blocked = false;
    bool large_blocker = false;
    int left_count = 0, right_count = 0;

    // O(N) per task (N <= ~50); the O(N^2) all-pair narrow-gap search is
    // done ONCE per scene in SceneGeometryCache.
    for (size_t i = 0; i < centers.size(); ++i) {
        const Vec2d& c = centers[i];
        const double d = distToSeg(c, start, goal);
        const double along =
            clamp((c - start).dot(axis) / (len * len), 0.0, 1.0) * len;
        // Only obstacles "ahead" of the start along the segment matter for
        // the proxy (behind the vehicle is irrelevant for classification).
        if (along < 0.5) continue;

        const double clear = d - radii[i];
        if (clear < chw) {
            ++near_count;
        }
        if (clear < free_clr) {
            straight_blocked = true;
            if (std::find(large.begin(), large.end(), static_cast<int>(i)) !=
                large.end()) {
                large_blocker = true;
            }
        }
        // Chicane: alternate left/right obstacles around the segment.
        if (clear < chw + 1.0) {
            if (signedDistToSeg(c, start, goal) >= 0.0) {
                ++left_count;
            } else {
                ++right_count;
            }
        }
    }

    // NARROW_BUT_PLANNABLE must be TASK-RELEVANT: the task's start-goal
    // corridor must actually pass through a cached narrow passage (never a
    // scene-corner gap unrelated to the path).  O(#narrow_passages).
    bool narrow_relevant = false;
    for (const auto& np : geo.narrowPassages()) {
        if (distToSeg(np.center, start, goal) < chw + 1.0) {
            narrow_relevant = true;
            break;
        }
    }

    const bool is_long = len >= cfg_.path_long_min_m - 1e-9;
    const bool chicane =
        near_count >= 2 && left_count >= 1 && right_count >= 1;

    // Priority order (geometric proxy, refined by preflight afterwards).
    if (is_long && (straight_blocked || large_blocker)) {
        return TaskGeomType::LONG_DETOUR;
    }
    if (large_blocker && straight_blocked) {
        return TaskGeomType::LARGE_OCCLUSION;
    }
    if (chicane) {
        return TaskGeomType::CHICANE;
    }
    if (straight_blocked && near_count >= 2) {
        return TaskGeomType::MULTI_OBSTACLE;
    }
    if (straight_blocked && near_count == 1) {
        return TaskGeomType::LOCAL_AVOIDANCE;
    }
    if (near_count >= 2) {
        return TaskGeomType::MULTI_OBSTACLE;
    }
    if (near_count == 1) {
        return TaskGeomType::OFFSET_AVOIDANCE;
    }
    if (narrow_relevant) {
        return TaskGeomType::NARROW_BUT_PLANNABLE;
    }
    return TaskGeomType::CLEAR;
}

TaskGeomType TaskCandidateGenerator::classifyQualified(
    const SceneGeometryCache& geo, const BlueprintScene& scene,
    const Vec2d& start, const Vec2d& goal,
    const TaskQualificationSummary& q) const {
    const Vec2d axis = goal - start;
    const double len = std::max(1e-6, axis.norm());
    const double chw = corridorHalfWidth();
    const double free_clr = cfg_.free_cell_surface_clearance_m;
    const auto& centers = geo.obstacleCenters();
    const auto& radii = geo.obstacleRadii();
    const auto& large = geo.largeObstacles();

    // Recompute the O(N) segment-relative statistics (also used by
    // classifyGeometry); cheap and scene-cached.
    int near_count = 0;
    bool large_blocker = false;
    for (size_t i = 0; i < centers.size(); ++i) {
        const Vec2d& c = centers[i];
        const double d = distToSeg(c, start, goal);
        const double along = clamp((c - start).dot(axis) / (len * len),
                                   0.0, 1.0) * len;
        if (along < 0.5) continue;  // only obstacles ahead of the start
        const double clear = d - radii[i];
        if (clear < chw) ++near_count;
        if (clear < free_clr &&
            std::find(large.begin(), large.end(), static_cast<int>(i)) !=
                large.end()) {
            large_blocker = true;
        }
    }
    // CHICANE evidence: real lateral sign alternation of the obstacles
    // NEAR the task corridor, sorted by their along-coordinate relative to
    // the task forward direction.  (+ - + -) is a chicane; (+ + - -) is
    // not, even though both left and right obstacles exist.
    const int chicane_flips = countTaskChicaneAlternations(centers, radii,
                                                           start, goal);

    // ── Straight corridor clear at the route qualification clearance ──
    if (q.straight_corridor_clear) {
        // A grazing single obstacle may still cause a local deflection.
        if (near_count == 1 && !large_blocker) {
            return TaskGeomType::OFFSET_AVOIDANCE;
        }
        if (near_count >= 2) {
            return TaskGeomType::MULTI_OBSTACLE;
        }
        return TaskGeomType::CLEAR;
    }

    // ── Blocked: classification priority is EXPLICIT (documented + tested,
    //    not an implicit if-order side effect):
    //    CHICANE > NARROW_BUT_PLANNABLE > LONG_DETOUR > LARGE_OCCLUSION >
    //    MULTI_OBSTACLE > LOCAL_AVOIDANCE > OFFSET_AVOIDANCE > CLEAR ─────
    const double stretch = q.privileged_min_route_stretch;
    const bool blocker_large =
        q.primary_blocker_radius >= largeRadiusThreshold();

    if (chicane_flips >= std::max(1, cfg_.min_chicane_alternations)) {
        return TaskGeomType::CHICANE;
    }
    // NARROW requires PROOF that an accepted qualified route actually
    // traverses a cached narrow passage (set by TaskRouteQualifier).
    // Priority NARROW > LARGE_OCCLUSION: a route squeezed through a narrow
    // gap is NARROW even when the flanking obstacles are large.
    if (q.route_traverses_narrow && q.narrow_passage_id >= 0) {
        return TaskGeomType::NARROW_BUT_PLANNABLE;
    }
    // LONG_DETOUR: the CORE criterion is the privileged route stretch
    // (min feasible route / straight distance), independent of whether the
    // straight distance itself is "long".  A 12 m straight with a 20 m
    // route (stretch 1.67) is a LONG_DETOUR even though 12 m < 20 m.
    if (stretch >= cfg_.qualification.min_route_stretch_for_long_detour) {
        return TaskGeomType::LONG_DETOUR;
    }
    if (blocker_large) {
        return TaskGeomType::LARGE_OCCLUSION;
    }
    if (q.blocking_obstacle_ids.size() >= 2 || near_count >= 2) {
        return TaskGeomType::MULTI_OBSTACLE;
    }
    return TaskGeomType::LOCAL_AVOIDANCE;
}

int TaskCandidateGenerator::countTaskChicaneAlternations(
    const std::vector<Vec2d>& centers, const std::vector<double>& radii,
    const Vec2d& start, const Vec2d& goal) const {
    const Vec2d axis = goal - start;
    const double len = std::max(1e-6, axis.norm());
    const Vec2d fwd = axis / len;
    const Vec2d left(-fwd.y(), fwd.x());
    const double chw = corridorHalfWidth();
    // Candidate obstacles: those whose surface is within a lateral band of
    // the task corridor (reuse the chw+1 band used by the classifier).
    std::vector<std::pair<double, double>> seq;  // (along, lateral)
    for (size_t i = 0; i < centers.size() && i < radii.size(); ++i) {
        const Vec2d rel = centers[i] - start;
        const double along = rel.dot(fwd);
        const double lateral = rel.dot(left);
        if (along < 0.5 || along > len) continue;
        const double off = std::fabs(lateral);
        if (off - radii[i] > chw + 1.0) continue;  // not near the corridor
        seq.emplace_back(along, lateral);
    }
    if (seq.size() < 4) return 0;
    std::sort(seq.begin(), seq.end());
    int flips = 0;
    double prev = 0.0;
    bool first = true;
    for (const auto& s : seq) {
        const double side = s.second;  // lateral sign (LEFT positive)
        if (first) {
            prev = side;
            first = false;
            continue;
        }
        if (prev * side < 0.0) ++flips;
        prev = side;
    }
    return flips;
}

double TaskCandidateGenerator::sampleInitialYaw(
    double goal_bearing_expert, const std::vector<double>& yaw_weights,
    Rng& rng) const {
    const auto& edges = cfg_.yaw_edges_deg;
    const int n = static_cast<int>(edges.size()) - 1;
    if (n < 1) return goal_bearing_expert - M_PI / 2.0;
    // Weighted stratum pick.
    std::vector<double> w;
    w.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        double ww = 1.0;
        if (i < static_cast<int>(yaw_weights.size()) &&
            yaw_weights[static_cast<size_t>(i)] > 0.0) {
            ww = yaw_weights[static_cast<size_t>(i)];
        }
        w.push_back(ww);
    }
    const int si = rng.weightedPick(w);
    const double lo = edges[static_cast<size_t>(si)];
    const double hi = edges[static_cast<size_t>(si + 1)];
    const double mag = rng.uniform(lo, hi);
    // Mirror-balanced sign: left/right equally likely.
    const double sign = (rng.uniform(0.0, 1.0) < 0.5) ? 1.0 : -1.0;
    const double yaw_error_deg = sign * mag;
    // expert_yaw = goal_bearing - yaw_error  (positive error = goal LEFT)
    const double expert_yaw = wrapAngle(
        goal_bearing_expert - deg2rad(yaw_error_deg));
    // Store the Flightmare yaw (convention B) in the task.
    return CoordinateAdapter::expertYawToFlightmare(expert_yaw);
}

bool TaskCandidateGenerator::sample(
    const BlueprintScene& scene, const SceneGeometryCache& geo,
    const std::vector<double>& task_type_weights,
    const std::vector<double>& yaw_weights, uint64_t seed, uint64_t task_id,
    uint64_t scene_id, BlueprintTask& out, TaskGeomType& geom_out,
    double& yaw_error_signed_deg) const {
    const auto& cells = geo.validCells();
    if (cells.size() < 2) return false;
    Rng rng(seed);

    // Desired proxy class (weighted; fall back to CLEAR-biased sampling
    // when the weight vector is empty).
    const int ntypes = kNumTaskGeomTypes;
    int desired_type = static_cast<int>(TaskGeomType::CLEAR);
    if (!task_type_weights.empty()) {
        desired_type = rng.weightedPick(task_type_weights);
        desired_type = clamp(desired_type, 0, ntypes - 1);
    }

    // Goal-distance band: expand for LONG_DETOUR.
    const double dmin = cfg_.min_task_distance_m;
    const double dmax = cfg_.max_task_distance_m;
    double band_lo = dmin, band_hi = dmax;
    if (static_cast<TaskGeomType>(desired_type) == TaskGeomType::LONG_DETOUR) {
        band_lo = std::min(std::max(dmin, cfg_.path_long_min_m),
                           dmax - 1e-6);
    }

    // ── Directional pre-pass (task twenty): for LONG_DETOUR and
    //    NARROW_BUT_PLANNABLE, generate the pair from SCENE-LOCAL hints
    //    (start/goal on opposite sides of a large blocker or a cached
    //    narrow passage) instead of waiting for a lucky random pair. ──
    if (desired_type == static_cast<int>(TaskGeomType::LONG_DETOUR) &&
        !geo.largeObstacles().empty()) {
        const size_t li = static_cast<size_t>(rng.uniformInt(
            0, static_cast<int>(geo.largeObstacles().size()) - 1));
        const Vec2d blocker =
            geo.obstacleCenters()[static_cast<size_t>(geo.largeObstacles()[li])];
        // Split the region by a line through the blocker; try both axes.
        for (int k = 0; k < 2; ++k) {
            const Vec2d normal = (k == 0) ? Vec2d(1.0, 0.0) : Vec2d(0.0, 1.0);
            if (sampleAcrossReference(geo, blocker, normal,
                                      std::max(band_lo, cfg_.path_long_min_m),
                                      TaskGeomType::LONG_DETOUR, yaw_weights,
                                      seed, rng, task_id, scene_id, out,
                                      geom_out, yaw_error_signed_deg)) {
                return true;
            }
        }
    }
    if (desired_type == static_cast<int>(TaskGeomType::NARROW_BUT_PLANNABLE) &&
        !geo.narrowPassages().empty()) {
        const NarrowPassage& np =
            geo.narrowPassages()[static_cast<size_t>(rng.uniformInt(
                0, static_cast<int>(geo.narrowPassages().size()) - 1))];
        // Normal to the passage axis: start/goal land on opposite sides of
        // the narrow gap, forcing the path through it.
        const Vec2d normal(-np.axis.y(), np.axis.x());
        if (sampleAcrossReference(geo, np.center, normal, dmin,
                                  TaskGeomType::NARROW_BUT_PLANNABLE,
                                  yaw_weights, seed, rng, task_id, scene_id,
                                  out, geom_out, yaw_error_signed_deg)) {
            return true;
        }
    }

    // A candidate that matches the desired proxy class (or is at least a
    // valid connected pair) is kept as a fallback when the budget runs out.
    bool have_fallback = false;
    TaskGeomType fallback_type = TaskGeomType::CLEAR;
    Vec2d fallback_start{0, 0}, fallback_goal{0, 0};
    double fallback_goal_bearing = 0.0;

    const int max_attempts = std::max(1, cfg_.task_sample_attempts);
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        // Start: random valid cell of the main component.
        const size_t sid = cells[rng.uniformInt(0, static_cast<int>(cells.size()) - 1)];
        const int sx = static_cast<int>(sid % static_cast<size_t>(geo.w()));
        const int sy = static_cast<int>(sid / static_cast<size_t>(geo.w()));
        const Vec2d start = geo.cellCenter(sx, sy);

        // Goal: another valid cell of the main component in the band.
        for (int g = 0; g < std::max(4, cfg_.task_goal_attempts); ++g) {
            const size_t gid = cells[rng.uniformInt(0, static_cast<int>(cells.size()) - 1)];
            const int gx = static_cast<int>(gid % static_cast<size_t>(geo.w()));
            const int gy = static_cast<int>(gid / static_cast<size_t>(geo.w()));
            const Vec2d goal = geo.cellCenter(gx, gy);
            const double d = (goal - start).norm();
            if (d < band_lo - 1e-9 || d > band_hi + 1e-9) continue;
            // Cheap analytic re-check (surface clearance + boundary).
            if (!geo.pointFreeMain(goal, cfg_.free_cell_surface_clearance_m)) {
                continue;
            }
            const TaskGeomType proxy = classifyGeometry(geo, start, goal);
            // Distance band of the desired class is always satisfied by
            // construction; accept an exact match or (for CLEAR) any
            // non-blocking proxy.
            bool accept = false;
            if (static_cast<int>(proxy) == desired_type) {
                accept = true;
            } else if (desired_type == static_cast<int>(TaskGeomType::CLEAR) &&
                       proxy == TaskGeomType::OFFSET_AVOIDANCE) {
                accept = true;  // near-clear (single grazing obstacle)
            } else if (desired_type == static_cast<int>(TaskGeomType::LOCAL_AVOIDANCE) &&
                       (proxy == TaskGeomType::OFFSET_AVOIDANCE ||
                        proxy == TaskGeomType::MULTI_OBSTACLE)) {
                accept = true;  // compatible avoidance proxies
            }
            if (!accept) {
                // Keep the first valid connected pair as fallback.
                if (!have_fallback) {
                    have_fallback = true;
                    fallback_type = proxy;
                    fallback_start = start;
                    fallback_goal = goal;
                    fallback_goal_bearing =
                        std::atan2(goal.y() - start.y(), goal.x() - start.x());
                }
                continue;
            }

            // ── Initial yaw (layered; sign mirrors left/right) ─────
            const double goal_bearing_expert =
                std::atan2(goal.y() - start.y(), goal.x() - start.x());
            const double initial_yaw_fm =
                sampleInitialYaw(goal_bearing_expert, yaw_weights, rng);
            // Signed error: goal bearing relative to the initial heading.
            const double expert_yaw =
                CoordinateAdapter::flightmareYawToExpert(initial_yaw_fm);
            const double err = wrapAngle(goal_bearing_expert - expert_yaw);

            out.scene_id = scene_id;
            out.task_id = task_id;
            out.seed = seed;
            out.start_x = start.x();
            out.start_y = start.y();
            out.goal_x = goal.x();
            out.goal_y = goal.y();
            out.initial_yaw = initial_yaw_fm;
            out.flight_height_m = cfg_.flight_height_m;
            out.audit.goal_distance_m = d;
            out.geom_type = taskGeomTypeName(proxy);
            out.audit.straight_distance_m = d;
            geom_out = proxy;
            yaw_error_signed_deg = rad2deg(err);
            return true;
        }
    }

    // Budget exhausted: emit the fallback (valid connected pair) so the
    // pool can still fill; preflight re-decides the true class.
    if (have_fallback) {
        const double goal_bearing_expert = fallback_goal_bearing;
        const double initial_yaw_fm =
            sampleInitialYaw(goal_bearing_expert, yaw_weights, rng);
        const double expert_yaw =
            CoordinateAdapter::flightmareYawToExpert(initial_yaw_fm);
        const double err = wrapAngle(goal_bearing_expert - expert_yaw);
        out.scene_id = scene_id;
        out.task_id = task_id;
        out.seed = seed;
        out.start_x = fallback_start.x();
        out.start_y = fallback_start.y();
        out.goal_x = fallback_goal.x();
        out.goal_y = fallback_goal.y();
        out.initial_yaw = initial_yaw_fm;
        out.flight_height_m = cfg_.flight_height_m;
        out.audit.goal_distance_m = (fallback_goal - fallback_start).norm();
        out.audit.straight_distance_m = out.audit.goal_distance_m;
        out.geom_type = taskGeomTypeName(fallback_type);
        geom_out = fallback_type;
        yaw_error_signed_deg = rad2deg(err);
        return true;
    }
    return false;
}

bool TaskCandidateGenerator::sampleAcrossReference(
    const SceneGeometryCache& geo, const Vec2d& ref, const Vec2d& normal,
    double min_dist, TaskGeomType required,
    const std::vector<double>& yaw_weights, uint64_t seed, Rng& rng,
    uint64_t task_id, uint64_t scene_id, BlueprintTask& out,
    TaskGeomType& geom_out, double& yaw_error_signed_deg) const {
    const auto& cells = geo.validCells();
    if (cells.size() < 2) return false;

    // Partition the valid cells by which side of the reference line they
    // lie on (O(#cells) once per directional attempt; cells ~ thousands).
    std::vector<Vec2d> plus, minus;
    plus.reserve(cells.size() / 2);
    minus.reserve(cells.size() / 2);
    const double side_margin = 2.0;
    for (size_t idx : cells) {
        const int ix = static_cast<int>(idx % static_cast<size_t>(geo.w()));
        const int iy = static_cast<int>(idx / static_cast<size_t>(geo.w()));
        const Vec2d p = geo.cellCenter(ix, iy);
        const double s = (p - ref).dot(normal);
        if (s > side_margin) {
            plus.push_back(p);
        } else if (s < -side_margin) {
            minus.push_back(p);
        }
    }
    if (plus.size() < 2 || minus.size() < 2) return false;

    const double dmax = cfg_.max_task_distance_m;
    const int tries = std::max(24, cfg_.task_goal_attempts);
    for (int a = 0; a < tries; ++a) {
        const Vec2d& start =
            (a % 2 == 0)
                ? plus[static_cast<size_t>(rng.uniformInt(
                      0, static_cast<int>(plus.size()) - 1))]
                : minus[static_cast<size_t>(rng.uniformInt(
                      0, static_cast<int>(minus.size()) - 1))];
        const Vec2d& goal =
            (a % 2 == 0)
                ? minus[static_cast<size_t>(rng.uniformInt(
                      0, static_cast<int>(minus.size()) - 1))]
                : plus[static_cast<size_t>(rng.uniformInt(
                      0, static_cast<int>(plus.size()) - 1))];
        const double d = (goal - start).norm();
        if (d < min_dist - 1e-9 || d > dmax + 1e-9) continue;
        if (!geo.pointFreeMain(goal, cfg_.free_cell_surface_clearance_m)) {
            continue;
        }
        const TaskGeomType proxy = classifyGeometry(geo, start, goal);
        if (proxy != required) continue;

        const double goal_bearing_expert =
            std::atan2(goal.y() - start.y(), goal.x() - start.x());
        const double initial_yaw_fm =
            sampleInitialYaw(goal_bearing_expert, yaw_weights, rng);
        const double expert_yaw =
            CoordinateAdapter::flightmareYawToExpert(initial_yaw_fm);
        const double err = wrapAngle(goal_bearing_expert - expert_yaw);

        out.scene_id = scene_id;
        out.task_id = task_id;
        out.seed = seed;
        out.start_x = start.x();
        out.start_y = start.y();
        out.goal_x = goal.x();
        out.goal_y = goal.y();
        out.initial_yaw = initial_yaw_fm;
        out.flight_height_m = cfg_.flight_height_m;
        out.audit.goal_distance_m = d;
        out.audit.straight_distance_m = d;
        out.geom_type = taskGeomTypeName(proxy);
        geom_out = proxy;
        yaw_error_signed_deg = rad2deg(err);
        return true;
    }
    return false;
}

}  // namespace expert
}  // namespace il_dataset
