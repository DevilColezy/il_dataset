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

    // ── Per-obstacle brake risk: edge-clearance floor ONLY.  USER
    //    DIRECTIVE (2026-08-20): the speed-dependent dynamic braking
    //    envelope is removed from all planners; the judge keeps the same
    //    rule — reject only when the drone's edge is closer than stop_margin
    //    (0.1 m) to a surface, i.e. centre closer than 0.4 m, at any speed. ──
    auto eval = [&](const Vec2d& rel, double d_surface, double& out_risk,
                    bool& out_would) {
        out_risk = 0.0;
        out_would = false;
        const double d_free = d_surface - vehicle_radius_;
        const double envelope = stop_margin;
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
        b.flight_height_min_m = cfg_.flight_height_min_m;
        b.flight_height_max_m = cfg_.flight_height_max_m;
        b.obstacle_height_m = cfg_.obstacle_height_m;
        b.obstacle_height_min_m = cfg_.obstacle_height_min_m;
        b.obstacle_height_max_m = cfg_.obstacle_height_max_m;
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
