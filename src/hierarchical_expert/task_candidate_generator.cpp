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

TaskGeomType TaskCandidateGenerator::classifyGeometry(
    const BlueprintScene& scene, const Vec2d& start,
    const Vec2d& goal) const {
    const Vec2d axis = goal - start;
    const double len = std::max(1e-6, axis.norm());
    const double chw = corridorHalfWidth();
    const double large_r = largeRadiusThreshold();
    const double free_clr = cfg_.free_cell_surface_clearance_m;

    int near_count = 0;
    double max_near_radius = 0.0;
    bool straight_blocked = false;
    bool large_blocker = false;
    int left_count = 0, right_count = 0;
    double narrowest_gap = std::numeric_limits<double>::infinity();

    for (const auto& o : scene.obstacles) {
        const Vec2d c(o.x, o.y);
        const double d = distToSeg(c, start, goal);
        const double along =
            clamp((c - start).dot(axis) / (len * len), 0.0, 1.0) * len;
        // Only obstacles "ahead" of the start along the segment matter for
        // the proxy (behind the vehicle is irrelevant for classification).
        if (along < 0.5) continue;

        const double clear = d - o.radius;
        if (clear < chw) {
            ++near_count;
            max_near_radius = std::max(max_near_radius, o.radius);
        }
        if (clear < free_clr) {
            straight_blocked = true;
            if (o.radius >= large_r) large_blocker = true;
        }
        // Chicane: alternate left/right obstacles around the segment.
        if (clear < chw + 1.0) {
            if (signedDistToSeg(c, start, goal) >= 0.0) {
                ++left_count;
            } else {
                ++right_count;
            }
        }
        // Narrow passage: gap between two obstacles straddling the segment.
        for (const auto& o2 : scene.obstacles) {
            if (o2.id == o.id) continue;
            const double gap =
                (c - Vec2d(o2.x, o2.y)).norm() - o.radius - o2.radius;
            narrowest_gap = std::min(narrowest_gap, gap);
        }
    }

    const bool is_long = len >= cfg_.path_long_min_m - 1e-9;
    const double narrow_max = 2.0 * cfg_.plannerRequiredPassage();
    const bool has_narrow = narrowest_gap >= cfg_.plannerRequiredPassage() - 1e-9 &&
                            narrowest_gap <= narrow_max + 1e-9;
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
    if (has_narrow) {
        return TaskGeomType::NARROW_BUT_PLANNABLE;
    }
    return TaskGeomType::CLEAR;
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
            const TaskGeomType proxy = classifyGeometry(scene, start, goal);
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

}  // namespace expert
}  // namespace il_dataset
