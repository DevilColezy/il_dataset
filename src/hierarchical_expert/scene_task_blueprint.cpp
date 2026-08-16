#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include "il_dataset/hierarchical_expert/blueprint_generation_controller.hpp"
#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

namespace {

/// Distance from a point to a finite segment (exact, for swept audit).
inline double distToSegment(const Vec2d& p, const Vec2d& a, const Vec2d& b) {
    const Vec2d ab = b - a;
    const double len2 = ab.squaredNorm();
    if (len2 <= 1e-12) return (p - a).norm();
    const double t = clamp((p - a).dot(ab) / len2, 0.0, 1.0);
    return (p - (a + ab * t)).norm();
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════
//  TruthCylinderAudit
// ═══════════════════════════════════════════════════════════════════
void TruthCylinderAudit::configure(const std::vector<BlueprintObstacle>& obs,
                                   double vehicle_radius,
                                   const Vec2d& min_bounds,
                                   const Vec2d& max_bounds) {
    obstacles_ = obs;
    vehicle_radius_ = vehicle_radius;
    min_bounds_ = min_bounds;
    max_bounds_ = max_bounds;
    has_bounds_ = true;
}

double TruthCylinderAudit::pointClearance(double x, double y) const {
    double best = std::numeric_limits<double>::infinity();
    const Vec2d p(x, y);
    for (const auto& o : obstacles_) {
        const double d = (p - Vec2d(o.x, o.y)).norm() - o.radius;
        best = std::min(best, d);
    }
    return best;
}

double TruthCylinderAudit::segmentMinClearance(double x0, double y0, double x1,
                                               double y1) const {
    const Vec2d a(x0, y0), b(x1, y1);
    double best = std::numeric_limits<double>::infinity();
    for (const auto& o : obstacles_) {
        const double d = distToSegment(Vec2d(o.x, o.y), a, b) - o.radius;
        best = std::min(best, d);
    }
    return best;
}

bool TruthCylinderAudit::segmentCollision(double x0, double y0, double x1,
                                          double y1) const {
    const Vec2d a(x0, y0), b(x1, y1);
    for (const auto& o : obstacles_) {
        if (distToSegment(Vec2d(o.x, o.y), a, b) <
            o.radius + vehicle_radius_ - 1e-9) {
            return true;
        }
    }
    return false;
}

bool TruthCylinderAudit::segmentCrossesBounds(double x0, double y0, double x1,
                                              double y1, double r) const {
    if (!has_bounds_) return false;
    // Convex safe rectangle: a straight segment stays inside the r-shrunk
    // rectangle IFF both endpoints do, so the swept disk is inside the
    // original rectangle exactly when both endpoints' disks are.  This is
    // the SHARED helper (types.hpp) — no segSegDist approximation.
    return !segmentDiskInsideBounds(x0, y0, x1, y1, r, min_bounds_,
                                    max_bounds_);
}

bool TruthCylinderAudit::pointOutOfBounds(double x, double y, double r) const {
    if (!has_bounds_) return false;
    return x - r < min_bounds_.x() || x + r > max_bounds_.x() ||
           y - r < min_bounds_.y() || y + r > max_bounds_.y();
}

void TruthCylinderAudit::brakeRisk(double x, double y, double vx, double vy,
                                   double max_decel, double stop_margin,
                                   double& risk, bool& would_trigger) const {
    risk = 0.0;
    would_trigger = false;
    const Vec2d p(x, y);
    const Vec2d v(vx, vy);
    const double speed = v.norm();

    // ── Per-obstacle brake risk: use the closing speed along the vector
    //    to THAT obstacle and take the MAXIMUM over all obstacles.  The
    //    nearest side/behind obstacle never masks the front obstacle. ──
    auto eval = [&](const Vec2d& rel, double d_surface, double& out_risk,
                    bool& out_would) {
        out_risk = 0.0;
        out_would = false;
        const double rn = rel.norm();
        const double d_free = d_surface - vehicle_radius_;
        // Closing speed = -(v·rel)/|rel| (positive when moving toward).
        double closing = 0.0;
        if (rn > 1e-9) closing = -(v.dot(rel)) / rn;
        if (closing <= 0.0) return;  // moving away / stationary
        const double stop_dist =
            closing * closing / (2.0 * std::max(1e-6, max_decel));
        const double envelope = stop_dist + stop_margin;
        if (d_free < envelope) {
            out_would = true;
            out_risk = clamp((envelope - d_free) / (envelope + 1e-9), 0.0, 1.0);
        }
    };

    for (const auto& o : obstacles_) {
        const Vec2d rel = p - Vec2d(o.x, o.y);
        const double d_surface = rel.norm() - o.radius;
        double rk = 0.0;
        bool wd = false;
        eval(rel, d_surface, rk, wd);
        if (rk > risk) risk = rk;
        would_trigger = would_trigger || wd;
    }

    // ── Region boundary brake risk (the disk cannot stop before leaving
    //    the configured region). ─────────────────────────────────────
    if (has_bounds_) {
        // Distance to the nearest boundary along the current motion.
        const Vec2d dir = speed > 1e-9 ? v / speed : Vec2d(1.0, 0.0);
        // Find where the drone disk centre would hit the boundary in the
        // motion direction.
        double t_hit = std::numeric_limits<double>::infinity();
        if (dir.x() > 1e-9) {
            t_hit = std::min(t_hit, (max_bounds_.x() - x) / dir.x());
        } else if (dir.x() < -1e-9) {
            t_hit = std::min(t_hit, (min_bounds_.x() - x) / dir.x());
        }
        if (dir.y() > 1e-9) {
            t_hit = std::min(t_hit, (max_bounds_.y() - y) / dir.y());
        } else if (dir.y() < -1e-9) {
            t_hit = std::min(t_hit, (min_bounds_.y() - y) / dir.y());
        }
        if (std::isfinite(t_hit)) {
            const double d_edge = t_hit - vehicle_radius_;  // disk outer edge to boundary
            const double stop_dist =
                speed * speed / (2.0 * std::max(1e-6, max_decel));
            const double envelope = stop_dist + stop_margin;
            if (d_edge < envelope) {
                would_trigger = true;
                risk = std::max(
                    risk, clamp((envelope - d_edge) / (envelope + 1e-9), 0.0,
                                1.0));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  SceneTaskBlueprintGenerator (facade over BlueprintGenerationController)
// ═══════════════════════════════════════════════════════════════════
void SceneTaskBlueprintGenerator::configure(const Params2D& params,
                                            const Config& cfg) {
    p_ = params;
    cfg_ = cfg;
    configured_ = true;
}

BlueprintGenerationConfig SceneTaskBlueprintGenerator::makeBlueprintConfig() const {
    BlueprintGenerationConfig b = cfg_.blueprint;
    if (!cfg_.blueprint_explicit) {
        // ── Backward-compatible fallback: derive the new-style config
        //    from the legacy Config fields (no blueprint_generation YAML
        //    section was provided). ─────────────────────────────────
        b.warehouse = WarehouseGeometry{
            p_.region_min_x, p_.region_max_x, p_.region_min_y,
            p_.region_max_y, 1.0};
        b.vehicle_radius_m = cfg_.vehicle_radius_m;
        b.navigation_clearance_m = cfg_.navigation_clearance_m;
        b.min_surface_gap_m = std::max(
            cfg_.min_surface_gap_m, b.plannerRequiredPassage());
        b.boundary_margin_m = cfg_.boundary_margin_m;
        b.free_cell_surface_clearance_m = cfg_.free_cell_surface_clearance_m;
        b.esdf_resolution_m = cfg_.esdf_resolution_m;
        b.min_task_distance_m = cfg_.min_task_distance_m;
        b.max_task_distance_m = cfg_.max_task_distance_m;
        b.flight_height_m = cfg_.flight_height_m;
        b.obstacle_height_m = cfg_.obstacle_height_m;
        b.task_sample_attempts = cfg_.task_sample_attempts;
        b.max_task_generation_attempts = std::max(
            1, static_cast<int>(cfg_.qualification_attempt_budget));
        b.max_preflight_ticks_per_task = static_cast<int>(
            cfg_.preflight_qualification_max_ticks);
        b.max_tasks_per_scene = std::max(1, cfg_.tasks_per_scene);
        b.min_tasks_per_scene = std::max(1, cfg_.minimum_tasks_per_scene);
        b.max_task_candidates_per_scene = std::max(
            1, cfg_.tasks_per_scene * std::max(1, cfg_.candidate_pool_multiplier));
        b.min_scenes = std::max(1, cfg_.scene_count);
        b.min_tasks = std::max(1, cfg_.scene_count * cfg_.tasks_per_scene);
        b.base_seed = cfg_.base_seed;
        b.density_sparse_max = cfg_.density_sparse_max;
        b.density_dense_min = cfg_.density_dense_min;
        b.radius_small_max_m = cfg_.radius_small_max_m;
        b.radius_large_min_m = cfg_.radius_large_min_m;
        // Legacy yaw bias becomes a coarse two-bin approximation of the
        // layered distribution: [-bias, +bias] plus a wide tail.
        if (cfg_.initial_yaw_bias_deg > 0.0) {
            b.yaw_edges_deg = {0.0, cfg_.initial_yaw_bias_deg, 180.0};
            b.yaw_weights = {2.0, 1.0};
        }
    }
    // ── Planner-compatibility guarantee (never below the required
    //    traversable passage for the configured vehicle/clearance). ──
    b.min_surface_gap_m = std::max(b.min_surface_gap_m,
                                   b.plannerRequiredPassage());
    // ── Single warehouse source for the whole pipeline ─────────────
    if (b.warehouse.area() <= 0.0) {
        b.warehouse = WarehouseGeometry{-7.0, 10.0, 0.0, 30.0, 1.0};
    }
    return b;
}

// The legacy stratum schedule / ESDF / sampling / quota methods were
// REPLACED by BlueprintGenerationController (see blueprint_generation_*
// and the scene_profile_* / task_candidate_* / distribution_analyzer_*
// components).  Only the facade below remains.

// ── One scene with an explicit stratum schedule + REAL geometry ─────
//    (REMOVED — replaced by SceneProfileGenerator)
#if 0
BlueprintScene SceneTaskBlueprintGenerator::makeScene(uint64_t scene_id,
                                                      uint64_t seed) const {
    BlueprintScene scene;
    scene.scene_id = scene_id;
    scene.seed = seed;

    if (scene_id == 0) {
        // ── Explicit empty / clear scene ───────────────────────────
        scene.is_empty = true;
        scene.stratum_id = -1;
        scene.count_stratum = -1;
        scene.radius_stratum = -1;
        scene.planned_density_class = "sparse";
        scene.planned_radius_class = "none";
        scene.requested_obstacle_count = 0;
        scene.actual_obstacle_count = 0;
        scene.generation_valid = true;
        scene.failure_reason = "";
        scene.density_class = "sparse";
        scene.actual_density_class = "sparse";
        scene.actual_radius_class = "none";
        scene.actual_min_radius_m = 0.0;
        scene.actual_max_radius_m = 0.0;
        return scene;
    }

    const uint64_t nid = scene_id - 1;
    scene.stratum_id = static_cast<int>(nid % 9);
    scene.count_stratum = scene.stratum_id % 3;   // 0=sparse 1=medium 2=dense
    scene.radius_stratum = scene.stratum_id / 3;  // 0=small 1=medium 2=large
    scene.planned_density_class = densityStratumName(scene.count_stratum);
    scene.planned_radius_class = radiusStratumName(scene.radius_stratum);

    // Non-empty count band aligned to the actual classification thresholds.
    int lo_count = 0, hi_count = 0;
    switch (scene.count_stratum) {
        case 0:
            lo_count = 1;
            hi_count = std::max(1, static_cast<int>(cfg_.density_sparse_max));
            break;
        case 1:
            lo_count = std::max(
                1, static_cast<int>(cfg_.density_sparse_max) + 1);
            hi_count = std::max(
                lo_count,
                std::max(lo_count,
                         static_cast<int>(cfg_.density_dense_min) - 1));
            break;
        default:
            lo_count = std::max(1, static_cast<int>(cfg_.density_dense_min));
            hi_count = std::max(lo_count, cfg_.max_obstacles);
            break;
    }
    // Non-empty radius band aligned to the actual classification thresholds.
    double lo_r = cfg_.radius_min_m, hi_r = cfg_.radius_max_m;
    switch (scene.radius_stratum) {
        case 0:
            lo_r = cfg_.radius_min_m;
            hi_r = cfg_.radius_small_max_m;
            break;
        case 1:
            lo_r = cfg_.radius_small_max_m;
            hi_r = cfg_.radius_large_min_m;
            break;
        default:
            lo_r = cfg_.radius_large_min_m;
            hi_r = cfg_.radius_max_m;
            break;
    }
    if (lo_r > hi_r) lo_r = hi_r;

    // Requested count sampled ONCE from the scene BASE seed (item 八.1);
    // all whole-scene retries reuse this SAME count + strata.
    Rng count_rng(seed);
    const int desired = count_rng.uniformInt(lo_count, hi_count);
    scene.requested_obstacle_count = desired;

    const double min_gap = cfg_.min_surface_gap_m;
    const double margin = cfg_.boundary_margin_m;
    const double x0 = p_.region_min_x + margin;
    const double x1 = p_.region_max_x - margin;
    const double y0 = p_.region_min_y + margin;
    const double y1 = p_.region_max_y - margin;
    const double width = std::max(1.0, x1 - x0);
    const double height = std::max(1.0, y1 - y0);

    int generation_attempt = 0;
    while (generation_attempt <
           std::max(1, cfg_.max_generation_attempts)) {
        ++generation_attempt;
        // Fresh deterministic attempt seed for placement ONLY (the
        // requested count / strata are already fixed).
        const uint64_t attempt_seed =
            seed + static_cast<uint64_t>(generation_attempt) * 0x9E3779B97F4A7C15ull;
        Rng rng(attempt_seed);
        scene.obstacles.clear();

        const int max_attempts = std::max(
            64, cfg_.max_generation_attempts * 64 * std::max(1, desired));
        int attempts = 0;
        while (static_cast<int>(scene.obstacles.size()) < desired &&
               attempts < max_attempts) {
            ++attempts;
            const double r = rng.uniform(lo_r, hi_r);
            if (r > width * 0.5 || r > height * 0.5) continue;
            const double x = rng.uniform(x0 + r, x1 - r);
            const double y = rng.uniform(y0 + r, y1 - r);
            bool ok = true;
            for (const auto& o : scene.obstacles) {
                const double d = std::hypot(x - o.x, y - o.y) - r - o.radius;
                if (d < min_gap - 1e-6) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            BlueprintObstacle ob;
            ob.x = x;
            ob.y = y;
            ob.radius = r;
            ob.height_m = cfg_.obstacle_height_m;
            ob.id = static_cast<int>(scene.obstacles.size());
            scene.obstacles.push_back(ob);
        }
        scene.actual_obstacle_count = static_cast<int>(scene.obstacles.size());

        // Count validity: actual == requested is the ONLY condition (the
        // meaningless "(desired > 0 || true)" expression is removed; a
        // shortfall always invalidates the scene).
        bool valid = (scene.actual_obstacle_count == desired);
        std::string reason = "";
        if (!valid) {
            reason = "requested=" + std::to_string(desired) + " actual=" +
                     std::to_string(scene.actual_obstacle_count);
        }
        if (valid) {
            for (const auto& a : scene.obstacles) {
                if (a.x - a.radius < x0 - 1e-6 || a.x + a.radius > x1 + 1e-6 ||
                    a.y - a.radius < y0 - 1e-6 || a.y + a.radius > y1 + 1e-6) {
                    valid = false;
                    reason = "boundary_gap_violation";
                    break;
                }
                for (const auto& b : scene.obstacles) {
                    if (a.id == b.id) continue;
                    const double d = std::hypot(a.x - b.x, a.y - b.y) -
                                     a.radius - b.radius;
                    if (d < min_gap - 1e-6) {
                        valid = false;
                        reason = "surface_gap_violation";
                        break;
                    }
                }
                if (!valid) break;
            }
        }
        if (valid) {
            scene.generation_valid = true;
            scene.failure_reason = "";
            break;
        }
        scene.generation_valid = false;
        scene.failure_reason = reason;
    }

    // ── ACTUAL classes from the REAL obstacle set (item 九) ────────
    scene.actual_density_class =
        densityClassOf(static_cast<double>(scene.actual_obstacle_count));
    scene.density_class = scene.actual_density_class;
    double min_r = std::numeric_limits<double>::infinity();
    double max_r = 0.0;
    for (const auto& o : scene.obstacles) {
        min_r = std::min(min_r, o.radius);
        max_r = std::max(max_r, o.radius);
    }
    scene.actual_min_radius_m = scene.obstacles.empty() ? 0.0 : min_r;
    scene.actual_max_radius_m = scene.obstacles.empty() ? 0.0 : max_r;
    scene.actual_radius_class = radiusClassOf(scene.actual_max_radius_m,
                                              scene.is_empty);
    return scene;
}

// ── Analytic re-verification of a start/goal pair (independent of the
//    grid): surface distance > free_cell_surface_clearance, boundary
//    clearance, and the distance lies in the requested stratum. ──────
bool SceneTaskBlueprintGenerator::verifyEndpointPair(
    const BlueprintScene& scene, double sx, double sy, double gx, double gy,
    int distance_stratum) const {
    const double margin = cfg_.boundary_margin_m;
    const double x0 = p_.region_min_x + margin, x1 = p_.region_max_x - margin;
    const double y0 = p_.region_min_y + margin, y1 = p_.region_max_y - margin;
    auto safe = [&](double x, double y) -> bool {
        if (x < x0 || x > x1 || y < y0 || y > y1) return false;
        for (const auto& o : scene.obstacles) {
            const double d = std::hypot(x - o.x, y - o.y) - o.radius;
            if (d <= cfg_.free_cell_surface_clearance_m + 1e-9) return false;
        }
        return true;
    };
    if (!safe(sx, sy) || !safe(gx, gy)) return false;
    const double d = std::hypot(gx - sx, gy - sy);
    const double dmin = cfg_.min_task_distance_m;
    const double dmax = cfg_.max_task_distance_m;
    const double band = (dmax - dmin) / 3.0;
    const double band_lo = dmin + distance_stratum * band;
    const double band_hi = dmin + (distance_stratum + 1) * band;
    return d >= band_lo - 1e-9 && d <= band_hi + 1e-9;
}

// ── Sample ONE connected task: start AND goal from free-cell CENTRES of
//    the SAME component, verified analytically before writing. ───────
bool SceneTaskBlueprintGenerator::sampleOneTask(
    const EsdfGrid& g, const BlueprintScene& scene, uint64_t seed,
    uint64_t task_id, int distance_stratum, BlueprintTask& out) const {
    Rng rng(seed);

    // Group free-cell indices by component.
    std::map<int, std::vector<size_t>> comp_cells;
    for (size_t id = 0; id < g.comp.size(); ++id) {
        if (g.comp[id] > 0) comp_cells[g.comp[id]].push_back(id);
    }
    if (comp_cells.empty()) return false;

    for (int attempt = 0; attempt < cfg_.task_sample_attempts; ++attempt) {
        // Pick a start component (weighted toward larger components) and a
        // start free-cell centre within it.
        auto it = comp_cells.begin();
        std::advance(it, rng.uniformInt(0, static_cast<int>(comp_cells.size()) - 1));
        const std::vector<size_t>& cells = it->second;
        if (cells.size() < 2) continue;
        const size_t start_id = cells[rng.uniformInt(
            0, static_cast<int>(cells.size()) - 1)];
        const int sx = static_cast<int>(start_id % g.w);
        const int sy = static_cast<int>(start_id / g.w);
        const Vec2d start = gridCellCenter(sx, sy, g.min_bounds, g.res);

        // Sample a GOAL free-cell centre from the SAME component whose
        // distance lies in the requested stratum.
        bool found = false;
        for (int goal_try = 0; goal_try < std::max(4, cfg_.task_sample_attempts / 4);
             ++goal_try) {
            const size_t goal_id = cells[rng.uniformInt(
                0, static_cast<int>(cells.size()) - 1)];
            const int gx = static_cast<int>(goal_id % g.w);
            const int gy = static_cast<int>(goal_id / g.w);
            const Vec2d goal = gridCellCenter(gx, gy, g.min_bounds, g.res);
            const double d = (goal - start).norm();
            const double dmin = cfg_.min_task_distance_m;
            const double dmax = cfg_.max_task_distance_m;
            const double band = (dmax - dmin) / 3.0;
            const double band_lo = dmin + distance_stratum * band;
            const double band_hi = dmin + (distance_stratum + 1) * band;
            if (d < band_lo - 1e-9 || d > band_hi + 1e-9) continue;

            // Analytic re-verification (item 二.3).
            if (!verifyEndpointPair(scene, start.x(), start.y(), goal.x(),
                                    goal.y(), distance_stratum)) {
                continue;
            }
            out.scene_id = scene.scene_id;
            out.task_id = task_id;
            out.seed = seed;
            out.start_x = start.x();
            out.start_y = start.y();
            out.goal_x = goal.x();
            out.goal_y = goal.y();
            const double aim_yaw =
                std::atan2(goal.y() - start.y(), goal.x() - start.x()) -
                M_PI / 2.0;
            const double bias =
                rng.uniform(-cfg_.initial_yaw_bias_deg,
                            cfg_.initial_yaw_bias_deg) *
                M_PI / 180.0;
            out.initial_yaw = aim_yaw + bias;
            out.flight_height_m = cfg_.flight_height_m;
            out.audit.goal_distance_m = d;
            assignGeometryClasses(out, scene);
            found = true;
            break;
        }
        if (found) return true;
    }
    return false;
}

// ── Geometry classes from scene geometry + task distance ───────────
void SceneTaskBlueprintGenerator::assignGeometryClasses(
    BlueprintTask& task, const BlueprintScene& scene) const {
    task.density_class = scene.actual_density_class;
    // Empty scenes have radius class "none" — their tasks NEVER contribute
    // to the small/medium/large radius quotas (item 九).
    task.radius_class = radiusClassOf(scene.actual_max_radius_m,
                                      scene.is_empty);
    const double d = task.audit.goal_distance_m;
    if (d <= cfg_.distance_short_max_m) {
        task.distance_class = "short";
    } else if (d >= cfg_.distance_long_min_m) {
        task.distance_class = "long";
    } else {
        task.distance_class = "medium";
    }
}

// ── Closed-loop preflight with the SAME expert + C++ classification ─
namespace {
struct BehaviorStats {
    uint64_t direct = 0, local_avoidance = 0, turn_to_target = 0;
    uint64_t goal_capture = 0, blocked = 0;
    uint64_t macro_frames = 0, pass = 0, normal = 0;
    uint64_t turn_left = 0, turn_right = 0;
    uint64_t correction_enter = 0, correction_update = 0;
    uint64_t max_consecutive_active = 0, cur_consecutive_active = 0;
    bool saw_turn = false;
    bool truth_collision = false, out_of_bounds = false;
    bool macro_label_invalid = false, reached = false;
    bool qualification_exceeded = false;
    double min_clearance = 1e9;
};
}  // namespace

void SceneTaskBlueprintGenerator::preflightTask(BlueprintTask& task,
                                                const BlueprintScene& scene,
                                                uint64_t tick_base) const {
    PreflightSimulator sim(p_);
    Scene2D s2d;
    s2d.min_bounds = Vec2d(p_.region_min_x, p_.region_min_y);
    s2d.max_bounds = Vec2d(p_.region_max_x, p_.region_max_y);
    s2d.valid = true;
    for (const auto& o : scene.obstacles) {
        Obstacle2D ob;
        ob.center = Vec2d(o.x, o.y);
        ob.radius = o.radius;
        ob.id = o.id;
        s2d.obstacles.push_back(ob);
    }
    sim.configure(s2d, s2d.min_bounds, s2d.max_bounds);

    TruthCylinderAudit truth;
    truth.configure(scene.obstacles, p_.drone_radius, s2d.min_bounds,
                    s2d.max_bounds);

    sim.resetTask(Vec2d(task.start_x, task.start_y),
                  Vec2d(task.goal_x, task.goal_y), task.initial_yaw,
                  tick_base, task.flight_height_m);

    BehaviorStats stats;
    Vec2d prev(task.start_x, task.start_y);
    const uint64_t budget =
        std::max<uint64_t>(1, cfg_.preflight_qualification_max_ticks);
    uint64_t ticks = 0;
    for (uint64_t t = 0; t < budget; ++t) {
        const auto res = sim.step(tick_base + t, false);
        ticks = t + 1;
        const ExpertStepOutput& out = res.output;

        // Continuous swept truth clearance + collision over the segment
        // prev→new (the sim already does its own; we double-check).
        const double seg_clr = truth.segmentMinClearance(
            prev.x(), prev.y(), res.state.position.x(), res.state.position.y());
        stats.min_clearance = std::min(stats.min_clearance, seg_clr);
        if (truth.segmentCollision(prev.x(), prev.y(),
                                   res.state.position.x(),
                                   res.state.position.y())) {
            stats.truth_collision = true;
            break;
        }
        if (res.truth_collision) {
            stats.truth_collision = true;
            break;
        }
        if (res.out_of_bounds ||
            truth.segmentCrossesBounds(prev.x(), prev.y(),
                                       res.state.position.x(),
                                       res.state.position.y(),
                                       p_.drone_radius)) {
            stats.out_of_bounds = true;
            break;
        }
        prev = res.state.position;

        const std::string& hm = out.hierarchical_mode;
        if (hm == "direct") {
            ++stats.direct;
        } else if (hm == "local_avoidance") {
            ++stats.local_avoidance;
        } else if (hm == "turn_to_target") {
            ++stats.turn_to_target;
        } else if (hm == "goal_capture") {
            ++stats.goal_capture;
        } else {
            ++stats.blocked;
        }
        if (out.macro_update_mask) {
            ++stats.macro_frames;
            const std::string& ct = out.macro_correction_type;
            if (ct == "PASS_THROUGH") {
                ++stats.pass;
            } else if (ct == "NORMAL_CORRECTION") {
                ++stats.normal;
            } else if (ct == "TURN_LEFT") {
                ++stats.turn_left;
                stats.saw_turn = true;
            } else if (ct == "TURN_RIGHT") {
                ++stats.turn_right;
                stats.saw_turn = true;
            }
            if (out.macro_label_valid != 1) {
                stats.macro_label_invalid = true;
                break;
            }
        }
        if (out.target_correction_active) {
            ++stats.cur_consecutive_active;
            stats.max_consecutive_active = std::max(
                stats.max_consecutive_active, stats.cur_consecutive_active);
        } else {
            stats.cur_consecutive_active = 0;
        }
        stats.correction_enter = std::max(stats.correction_enter,
                                          out.correction_enter_event);
        stats.correction_update = std::max(stats.correction_update,
                                           out.correction_update_event);

        if (res.goal_reached) {
            stats.reached = true;
            break;
        }
    }
    if (ticks >= budget && !stats.reached) {
        stats.qualification_exceeded = true;
    }

    // ── Audit ──────────────────────────────────────────────────────
    task.audit.preflight_ticks = ticks;
    task.audit.min_truth_clearance_m = stats.min_clearance;
    task.audit.reached_goal = stats.reached;
    task.audit.truth_collision = stats.truth_collision;
    task.audit.out_of_bounds = stats.out_of_bounds;
    task.audit.macro_label_ok = !stats.macro_label_invalid;
    task.audit.qualification_exceeded = stats.qualification_exceeded;
    task.audit.accepted = stats.reached && !stats.truth_collision &&
                          !stats.out_of_bounds && !stats.macro_label_invalid &&
                          !stats.qualification_exceeded;

    if (!task.audit.accepted) {
        task.behavior_class = "rejected";
        task.side_class = "none";
        task.audit.preflight_status =
            stats.truth_collision
                ? "preflight_rejected:truth_collision"
                : (stats.out_of_bounds
                       ? "preflight_rejected:out_of_bounds"
                       : (stats.qualification_exceeded
                              ? "preflight_rejected:qualification_budget"
                              : (!stats.reached
                                     ? "preflight_rejected:goal_not_reached"
                                     : "preflight_rejected:macro_label_invalid")));
        return;
    }
    task.audit.preflight_status = "preflight_accepted";

    // ── Behavior classification (priority order) ───────────────────
    if (stats.saw_turn && stats.normal > 0) {
        task.behavior_class = "turn_normal";
    } else if (stats.turn_left > 0 && stats.turn_right == 0) {
        task.behavior_class = "turn_left";
    } else if (stats.turn_right > 0 && stats.turn_left == 0) {
        task.behavior_class = "turn_right";
    } else if (stats.turn_left > 0 && stats.turn_right > 0) {
        task.behavior_class = "turn_both";
    } else if (stats.correction_enter >= 2 || stats.correction_update >= 2) {
        task.behavior_class = "multi_correction";
    } else if (stats.max_consecutive_active >=
               cfg_.long_takeover_min_ticks) {
        task.behavior_class = "long_takeover";
    } else if (stats.normal > 0) {
        task.behavior_class = "normal";
    } else if (stats.local_avoidance > 0) {
        task.behavior_class = "local_avoidance";
    } else {
        task.behavior_class = "clear";
    }

    // ── Real macro-directive statistics (item 六) ──────────────────
    // The TURN left/right quotas count ONLY tasks that ACTUALLY saw a
    // TURN_LEFT / TURN_RIGHT directive.  NORMAL_CORRECTION direction-token
    // laterality is never a TURN and never contributes to min_turn_per_side.
    task.saw_turn_left = stats.turn_left > 0;
    task.saw_turn_right = stats.turn_right > 0;
    task.saw_normal_correction = stats.normal > 0;
    task.turn_update_count = stats.turn_left + stats.turn_right;
    task.normal_update_count = stats.normal;

    // ── Side class from REAL TURN directives only ──────────────────
    if (stats.turn_left > 0 && stats.turn_right > 0) {
        task.side_class = "both";
    } else if (stats.turn_left > 0) {
        task.side_class = "left";
    } else if (stats.turn_right > 0) {
        task.side_class = "right";
    } else {
        task.side_class = "none";
    }
}

// ── Dataset quota selection (greedy, deterministic, HARD verification) ──
//    Order (item 五): (A) per-scene floors for EVERY scene_id 0..n-1 first,
//    (B) global required behaviors + real TURN side quotas, (C) geometry
//    level floors, (D) fill to capacity.  Every accept is capped per scene
//    at tasks_per_scene.  side_count counts ONLY real TURN tasks (item 六):
//    a NORMAL_CORRECTION direction-token laterality is never a TURN.
std::vector<BlueprintTask> SceneTaskBlueprintGenerator::applyQuotas(
    std::vector<BlueprintTask> all, std::vector<std::string>& unmet_quotas,
    std::map<std::string, uint64_t>& category_counts,
    std::vector<uint64_t>& per_scene_accepted) const {
    unmet_quotas.clear();
    category_counts.clear();

    const int n_scenes = std::max(1, cfg_.scene_count);
    const int max_tasks = std::max(1, n_scenes * cfg_.tasks_per_scene);
    const int bucket_cap =
        std::max(1, static_cast<int>(std::ceil(max_tasks / 27.0)) + 2);

    std::map<std::string, int> behavior_count, side_count, level_count;
    // Fixed length = cfg_.scene_count; a scene with NO accepted task is
    // counted as 0 (item 五) — it can never be skipped.
    std::vector<int> scene_count(static_cast<size_t>(n_scenes), 0);
    std::set<uint64_t> used;
    std::vector<BlueprintTask> accepted;

    auto bucketKey = [](const BlueprintTask& t) {
        return t.density_class + "|" + t.radius_class + "|" +
               t.distance_class;
    };
    auto sceneOk = [&](const BlueprintTask& t) -> bool {
        const size_t s = static_cast<size_t>(t.scene_id);
        return s < static_cast<size_t>(n_scenes) &&
               scene_count[s] < cfg_.tasks_per_scene;
    };
    auto sideImbalanced = [&](const BlueprintTask& t) -> bool {
        if (t.side_class == "left") {
            return side_count["left"] - side_count["right"] >=
                   cfg_.max_left_right_imbalance;
        }
        if (t.side_class == "right") {
            return side_count["right"] - side_count["left"] >=
                   cfg_.max_left_right_imbalance;
        }
        return false;
    };
    auto accept = [&](const BlueprintTask& t) {
        used.insert(t.task_id);
        accepted.push_back(t);
        ++behavior_count[t.behavior_class];
        ++side_count[t.side_class];
        ++level_count[bucketKey(t)];
        scene_count[static_cast<size_t>(t.scene_id)] += 1;
    };

    // ── Phase A: per-scene floors (EVERY scene_id 0..scene_count-1) ──
    for (int sid = 0; sid < n_scenes; ++sid) {
        for (const auto& t : all) {
            if (static_cast<int>(accepted.size()) >= max_tasks) break;
            if (static_cast<int>(t.scene_id) != sid) continue;
            if (used.count(t.task_id)) continue;
            if (!sceneOk(t)) continue;
            if (scene_count[static_cast<size_t>(sid)] >=
                cfg_.minimum_tasks_per_scene) {
                break;  // floor satisfied (deterministic pool order)
            }
            if (level_count[bucketKey(t)] >= bucket_cap) continue;
            accept(t);
        }
    }

    // ── Phase B: required behaviors + REAL TURN side quotas ────────
    // need_behavior applies ONLY to kRequiredBehaviors: turn_both /
    // multi_correction / long_takeover never fill capacity ahead of the
    // hard-required classes (item 六).
    auto needBehavior = [&](const BlueprintTask& t) -> bool {
        for (const char* b : kRequiredBehaviors) {
            if (std::string(b) == t.behavior_class &&
                behavior_count[t.behavior_class] < cfg_.min_per_behavior) {
                return true;
            }
        }
        return false;
    };
    auto needTurnSide = [&](const BlueprintTask& t) -> bool {
        return (t.side_class == "left" &&
                side_count["left"] < cfg_.min_turn_per_side) ||
               (t.side_class == "right" &&
                side_count["right"] < cfg_.min_turn_per_side);
    };
    for (const auto& t : all) {
        if (static_cast<int>(accepted.size()) >= max_tasks) break;
        if (used.count(t.task_id) || !sceneOk(t)) continue;
        if (level_count[bucketKey(t)] >= bucket_cap) continue;
        if (sideImbalanced(t)) continue;
        if (needBehavior(t) || needTurnSide(t)) accept(t);
    }

    // ── Phase C: geometry level floors (density / radius / distance) ──
    auto densityTotal = [&](const std::string& lvl) {
        int c = 0;
        for (const auto& kv : level_count) {
            if (kv.first.rfind(lvl + "|", 0) == 0) c += kv.second;
        }
        return c;
    };
    auto radiusTotal = [&](const std::string& lvl) {
        int c = 0;
        for (const auto& kv : level_count) {
            const auto p1 = kv.first.find('|');
            const auto p2 = kv.first.find('|', p1 + 1);
            if (p1 != std::string::npos && p2 != std::string::npos &&
                kv.first.substr(p1 + 1, p2 - p1 - 1) == lvl) {
                c += kv.second;
            }
        }
        return c;
    };
    auto distanceTotal = [&](const std::string& lvl) {
        int c = 0;
        for (const auto& kv : level_count) {
            const auto p1 = kv.first.find('|');
            const auto p2 = kv.first.find('|', p1 + 1);
            if (p1 != std::string::npos && p2 != std::string::npos &&
                kv.first.substr(p2 + 1) == lvl) {
                c += kv.second;
            }
        }
        return c;
    };
    for (const auto& t : all) {
        if (static_cast<int>(accepted.size()) >= max_tasks) break;
        if (used.count(t.task_id) || !sceneOk(t)) continue;
        if (level_count[bucketKey(t)] >= bucket_cap) continue;
        if (sideImbalanced(t)) continue;
        const bool need_level =
            densityTotal(t.density_class) < cfg_.min_per_density_level ||
            (t.radius_class != "none" &&
             radiusTotal(t.radius_class) < cfg_.min_per_radius_level) ||
            distanceTotal(t.distance_class) < cfg_.min_per_distance_level;
        if (need_level) accept(t);
    }

    // ── Phase D: fill remaining capacity (per-scene cap included) ──
    for (const auto& t : all) {
        if (static_cast<int>(accepted.size()) >= max_tasks) break;
        if (used.count(t.task_id) || !sceneOk(t)) continue;
        if (level_count[bucketKey(t)] >= bucket_cap) continue;
        if (sideImbalanced(t)) continue;
        accept(t);
    }

    // ── HARD verification (every scene_id checked, missing = 0) ────
    auto fail = [&](const std::string& msg) { unmet_quotas.push_back(msg); };

    for (const char* b : kRequiredBehaviors) {
        if (behavior_count[b] < cfg_.min_per_behavior) {
            fail("behavior:" + std::string(b) + "=" +
                 std::to_string(behavior_count[b]) + "<" +
                 std::to_string(cfg_.min_per_behavior));
        }
    }
    if (side_count["left"] < cfg_.min_turn_per_side) {
        fail("turn_left=" + std::to_string(side_count["left"]) + "<" +
             std::to_string(cfg_.min_turn_per_side));
    }
    if (side_count["right"] < cfg_.min_turn_per_side) {
        fail("turn_right=" + std::to_string(side_count["right"]) + "<" +
             std::to_string(cfg_.min_turn_per_side));
    }
    if (std::abs(side_count["left"] - side_count["right"]) >
        cfg_.max_left_right_imbalance) {
        fail("turn_left_right_imbalance=" +
             std::to_string(std::abs(side_count["left"] -
                                     side_count["right"])) +
             ">" + std::to_string(cfg_.max_left_right_imbalance));
    }
    for (const char* lvl : {"sparse", "medium", "dense"}) {
        if (densityTotal(lvl) < cfg_.min_per_density_level) {
            fail("density:" + std::string(lvl) + "=" +
                 std::to_string(densityTotal(lvl)) + "<" +
                 std::to_string(cfg_.min_per_density_level));
        }
    }
    for (const char* lvl : {"small", "medium", "large"}) {
        if (radiusTotal(lvl) < cfg_.min_per_radius_level) {
            fail("radius:" + std::string(lvl) + "=" +
                 std::to_string(radiusTotal(lvl)) + "<" +
                 std::to_string(cfg_.min_per_radius_level));
        }
    }
    for (const char* lvl : {"short", "medium", "long"}) {
        if (distanceTotal(lvl) < cfg_.min_per_distance_level) {
            fail("distance:" + std::string(lvl) + "=" +
                 std::to_string(distanceTotal(lvl)) + "<" +
                 std::to_string(cfg_.min_per_distance_level));
        }
    }
    // Per-scene floor AND cap, iterating 0..scene_count-1 (item 五): a
    // scene absent from the pool is counted as 0 and fails explicitly.
    for (int sid = 0; sid < n_scenes; ++sid) {
        const int c = scene_count[static_cast<size_t>(sid)];
        if (c < cfg_.minimum_tasks_per_scene) {
            fail("scene:" + std::to_string(sid) + "=" + std::to_string(c) +
                 "<" + std::to_string(cfg_.minimum_tasks_per_scene));
        }
        if (c > cfg_.tasks_per_scene) {
            fail("scene:" + std::to_string(sid) + "=" + std::to_string(c) +
                 ">" + std::to_string(cfg_.tasks_per_scene));
        }
    }

    // ── category counts ────────────────────────────────────────────
    for (const auto& kv : behavior_count) {
        category_counts["behavior:" + kv.first] = kv.second;
    }
    for (const char* lvl : {"sparse", "medium", "dense"}) {
        category_counts["density:" + std::string(lvl)] = densityTotal(lvl);
    }
    for (const char* lvl : {"small", "medium", "large"}) {
        category_counts["radius:" + std::string(lvl)] = radiusTotal(lvl);
    }
    for (const char* lvl : {"short", "medium", "long"}) {
        category_counts["distance:" + std::string(lvl)] = distanceTotal(lvl);
    }
    category_counts["turn_left"] = side_count["left"];
    category_counts["turn_right"] = side_count["right"];

    per_scene_accepted.assign(static_cast<size_t>(n_scenes), 0);
    for (int sid = 0; sid < n_scenes; ++sid) {
        per_scene_accepted[static_cast<size_t>(sid)] =
            scene_count[static_cast<size_t>(sid)];
    }
    return accepted;
}

#endif  // legacy generation engine (replaced by BlueprintGenerationController)

// ── generate(): delegate to the deficit-driven controller ─────────
BlueprintResult SceneTaskBlueprintGenerator::generate() {
    BlueprintResult result;
    if (!configured_) {
        result.failure_reason = "generator not configured";
        return result;
    }
    const BlueprintGenerationConfig b = makeBlueprintConfig();
    BlueprintGenerationController controller(p_, b);
    result = controller.generate();
    return result;
}

}  // namespace expert
}  // namespace il_dataset
