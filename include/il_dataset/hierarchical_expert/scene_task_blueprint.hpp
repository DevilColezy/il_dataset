#pragma once
/// @file   scene_task_blueprint.hpp
/// @brief  FULL C++ scene / truth-ESDF / task / blueprint generation +
///         C++ behavior classifier + dataset quota selection.
///
/// This replaces the Python TruthAudit / _generate_scene / _sample_task /
/// JointV2BehaviorClassifier.  Python (il_manager.py) only:
///   * configures the generator from YAML,
///   * calls generate() once (BOTH normal collection and blueprint_only),
///   * writes the manifest,
///   * then (normal mode only) collects strictly per the manifest.
///
/// ═══════════════════════════════════════════════════════════════════
///  ESDF / free-cell semantics (single, never mixed)
/// ═══════════════════════════════════════════════════════════════════
///   ESDF value = distance from the DRONE CENTRE point to the nearest
///   obstacle SURFACE (analytic |p−centre| − radius).
///   A free candidate cell (start / goal / connectivity) requires
///   ESDF > free_cell_surface_clearance_m, where
///   free_cell_surface_clearance_m >= vehicle_radius (validated).  The
///   body-edge clearance of a free cell is therefore ESDF − vehicle_radius.
///   All connectivity / start / goal / analytic re-checks use THIS
///   centre-to-surface convention — never the body-edge distance.
///
///  STRATA SCHEDULE (explicit, recorded):
///   scene 0 = explicit EMPTY / CLEAR scene (0 obstacles, radius "none").
///   scenes 1..9 = the non-empty 3×3 joint coverage; for scene_id >= 1,
///   stratum_id = (scene_id-1) % 9; count_stratum = stratum_id % 3;
///   radius_stratum = stratum_id / 3.  The non-empty count/radius bands are
///   aligned to the CONFIGURED classification thresholds
///   (density_sparse_max / density_dense_min / radius_small_max /
///   radius_large_min), so planned == actual class on success.  A stratum
///   is COVERED only when the ACTUAL obstacle set classifies into the
///   planned density AND radius class (never merely scene.generation_valid).
///   scene_count < 10 NEVER claims full joint coverage.
///
///  CONNECTIVITY (no corner-cutting): 8-connected flood fill where a
///   diagonal move (dx,dy both nonzero) is allowed ONLY when the two
///   orthogonal neighbours (cx+dx, cy) and (cx, cy+dy) are also free.
///
///  TASKS (oversampling): tasks_per_scene is the FINAL expected accepted
///   count per scene.  Each scene samples + preflights candidates until its
///   candidate pool reaches tasks_per_scene * max(1, candidate_pool_multiplier)
///   OR the finite qualification_attempt_budget is exhausted.  A budget
///   shortfall below the candidate target is an EXPLICIT failure (the
///   quota selector would be forced to fill classes from a too-small pool;
///   it is never reported as "oversampled").  start and goal are sampled
///   from free-cell CENTRES of the SAME connected component and are
///   analytically re-verified (surface clearance + boundary + distance
///   stratum) before being written into BlueprintTask.
///
///  HARD QUOTAS: applyQuotas() verifies every configured quota and returns
///   unmet_quotas; generate() sets generation_ok=false if any hard quota or
///   any scene is invalid.  The manager MUST abort before connecting
///   Flightmare when generation_ok == false.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/preflight_simulator.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

// ═══════════════════════════════════════════════════════════════════
//  POD results (exposed to Python via pybind)
// ═══════════════════════════════════════════════════════════════════
struct BlueprintObstacle {
    double x = 0.0, y = 0.0;
    double radius = 0.0;
    double height_m = 8.0;
    int id = -1;
};

struct BlueprintScene {
    uint64_t scene_id = 0;
    uint64_t seed = 0;
    int stratum_id = 0;  // 0..8 for non-empty scenes; -1 for the empty scene
    int count_stratum = 0;  // 0=sparse 1=medium 2=dense; -1 for empty
    int radius_stratum = 0;  // 0=small 1=medium 2=large; -1 for empty
    bool is_empty = false;   // explicit CLEAR scene with 0 obstacles
    int requested_obstacle_count = 0;
    int actual_obstacle_count = 0;
    bool generation_valid = true;
    std::string failure_reason = "";
    std::vector<BlueprintObstacle> obstacles;
    // ── strata (actual geometry, never the scene_id schedule) ──────
    // planned_* is the schedule target; actual_* is computed from the REAL
    // obstacle set.  A stratum is covered ONLY when actual == planned.
    std::string planned_density_class = "";  // sparse / medium / dense
    std::string planned_radius_class = "";   // small / medium / large / none
    std::string density_class = "medium";    // == actual_density_class
    std::string actual_density_class = "medium";
    std::string actual_radius_class = "none";
    double actual_min_radius_m = 0.0;
    double actual_max_radius_m = 0.0;
};

struct BlueprintTaskAudit {
    bool accepted = false;
    bool reached_goal = false;
    bool truth_collision = false;      // point OR continuous swept collision
    bool out_of_bounds = false;        // drone disk crossed the region boundary
    bool macro_label_ok = true;
    bool qualification_exceeded = false;  // ran out of the preflight tick budget
    uint64_t preflight_ticks = 0;
    double min_truth_clearance_m = 1e9;
    double goal_distance_m = 0.0;
    std::string preflight_status = "";
};

struct BlueprintTask {
    uint64_t scene_id = 0;
    uint64_t task_id = 0;
    uint64_t seed = 0;
    double start_x = 0.0, start_y = 0.0;
    double goal_x = 0.0, goal_y = 0.0;
    double initial_yaw = 0.0;  // Flightmare yaw (convention B)
    double flight_height_m = 2.0;
    std::string behavior_class = "clear";
    std::string density_class = "medium";
    std::string radius_class = "medium";  // "none" for empty-scene tasks
    std::string distance_class = "medium";
    std::string side_class = "none";  // left / right / both / none
    // ── REAL macro-directive statistics (item 六) ──────────────────
    // The TURN left/right quota counts ONLY tasks that actually saw a
    // TURN_LEFT / TURN_RIGHT directive.  NORMAL_CORRECTION direction-token
    // laterality is never a TURN and never contributes to min_turn_per_side.
    bool saw_turn_left = false;
    bool saw_turn_right = false;
    bool saw_normal_correction = false;
    uint64_t turn_update_count = 0;    // TURN_LEFT/RIGHT update frames
    uint64_t normal_update_count = 0;  // NORMAL_CORRECTION update frames
    BlueprintTaskAudit audit;
};

struct BlueprintResult {
    bool generation_ok = false;
    std::string failure_reason = "";
    std::vector<std::string> unmet_quotas;
    std::vector<BlueprintScene> scenes;
    /// quota-accepted tasks (normal collection iterates EXACTLY these)
    std::vector<BlueprintTask> tasks;
    /// all preflight-accepted tasks (the oversampled candidate pool)
    std::vector<BlueprintTask> preflighted;
    uint64_t requested_scenes = 0;
    uint64_t requested_tasks_per_scene = 0;
    uint64_t scenes_generated = 0;
    uint64_t scenes_valid = 0;
    uint64_t tasks_sampled = 0;      // sampled start/goal candidates
    uint64_t tasks_preflighted = 0;  // candidates actually preflighted
    uint64_t tasks_pool_target = 0;  // sum of per-scene oversample targets
    uint64_t tasks_pool_accepted = 0;  // preflight-accepted candidate pool
    uint64_t tasks_quota_accepted = 0;  // final quota-selected tasks
    bool pool_budget_exhausted = false;  // a scene hit the budget below target
    uint64_t strata_required = 0;
    uint64_t strata_covered = 0;
    std::vector<uint64_t> strata_covered_flags;  // 1 if stratum i covered by ACTUAL geometry
    /// per-scene accepted-task counts (indexed by scene_id)
    std::vector<uint64_t> per_scene_accepted;
    /// final per-category counts: "behavior:clear", "density:sparse", ...
    std::map<std::string, uint64_t> category_counts;
    uint64_t base_seed = 0;
};

// ═══════════════════════════════════════════════════════════════════
//  Exact-cylinder truth audit (judge-only; continuous swept segments)
// ═══════════════════════════════════════════════════════════════════
class TruthCylinderAudit {
public:
    void configure(const std::vector<BlueprintObstacle>& obstacles,
                   double vehicle_radius, const Vec2d& min_bounds,
                   const Vec2d& max_bounds);

    /// Clearance from the point to the nearest obstacle SURFACE (m).
    double pointClearance(double x, double y) const;

    /// MINIMUM swept clearance of the moving body (a disk of
    /// `vehicle_radius`) along the segment P0→P1, w.r.t. every exact
    /// cylinder: min over obstacles of (dist(segment, center) - radius).
    /// This is EXACT (analytic segment–circle distance), not a sampled
    /// approximation.  Value = drone centre→surface distance along the
    /// swept path.
    double segmentMinClearance(double x0, double y0, double x1,
                               double y1) const;

    /// Swept collision of the body disk along the segment: true iff any
    /// cylinder surface comes within `vehicle_radius` of the segment.
    bool segmentCollision(double x0, double y0, double x1, double y1) const;

    /// True if the drone disk (radius r) swept along the segment crosses
    /// the configured region boundary (edges AND corners).
    bool segmentCrossesBounds(double x0, double y0, double x1, double y1,
                              double r) const;

    /// True if the drone disk at a point crosses the region boundary.
    bool pointOutOfBounds(double x, double y, double r) const;

    /// Truth brake risk at the current state.  For EVERY obstacle AND the
    /// region boundary independently: compute the surface/body-edge
    /// distance, the closing speed along that direction, the stopping
    /// distance based on the closing speed, the brake envelope and the
    /// per-obstacle risk; the result is the MAXIMUM risk over all of them
    /// (a nearest side/behind obstacle can never mask the front obstacle
    /// that really needs braking).  `would_trigger` is true iff any
    /// envelope is violated while approaching.
    void brakeRisk(double x, double y, double vx, double vy,
                   double max_decel, double stop_margin, double& risk,
                   bool& would_trigger) const;

private:
    std::vector<BlueprintObstacle> obstacles_;
    double vehicle_radius_ = 0.3;
    Vec2d min_bounds_{-1e9, -1e9};
    Vec2d max_bounds_{1e9, 1e9};
    bool has_bounds_ = false;
};

// ═══════════════════════════════════════════════════════════════════
//  Scene / ESDF / task / blueprint generator + classifier + quotas
// ═══════════════════════════════════════════════════════════════════
class SceneTaskBlueprintGenerator {
public:
    struct Config {
        // ── scenes ────────────────────────────────────────────────
        int scene_count = 10;       // scene 0 = explicit EMPTY/CLEAR scene
                                    // + 9 non-empty sparse/medium/dense ×
                                    //     small/medium/large coverage
        int tasks_per_scene = 8;    // FINAL expected accepted tasks/scene
        int minimum_tasks_per_scene = 6;
        uint64_t base_seed = 260812;
        double flight_height_m = 2.0;
        double obstacle_height_m = 8.0;
        bool require_full_strata_coverage = true;
        // ── geometry ──────────────────────────────────────────────
        double min_surface_gap_m = 1.20;
        double boundary_margin_m = 1.20;
        double radius_min_m = 0.10;
        double radius_max_m = 2.00;
        int max_obstacles = 20;
        double vehicle_radius_m = 0.30;
        double navigation_clearance_m = 0.30;
        double free_cell_surface_clearance_m = 0.50;  // centre→surface, ≥ vehicle_radius
        double esdf_resolution_m = 0.10;
        int max_generation_attempts = 24;   // whole-scene regeneration attempts
        // ── tasks (oversampling) ──────────────────────────────────
        double min_task_distance_m = 4.0;
        double max_task_distance_m = 20.0;
        double initial_yaw_bias_deg = 15.0;  // left/right symmetric
        int task_sample_attempts = 200;      // per-candidate start/goal attempts
        int candidate_pool_multiplier = 4;   // oversample preflighted pool
        uint64_t qualification_attempt_budget = 400;  // per-scene preflight attempts
        // ── preflight (named, finite qualification budget) ────────
        uint64_t preflight_qualification_max_ticks = 1800;  // 60 s @ 30 Hz
        // ── dataset quotas (hard) ─────────────────────────────────
        int min_per_behavior = 2;      // each required behavior class
        int min_turn_per_side = 2;     // turn_left / turn_right tasks
        int max_left_right_imbalance = 2;
        int min_per_density_level = 4;   // sparse / medium / dense
        int min_per_radius_level = 4;    // small / medium / large
        int min_per_distance_level = 4;  // short / medium / long
        double distance_short_max_m = 9.0;
        double distance_long_min_m = 15.0;
        double radius_small_max_m = 0.6;
        double radius_large_min_m = 1.4;
        double density_sparse_max = 7.0;
        double density_dense_min = 14.0;
        uint64_t long_takeover_min_ticks = 30;  // >= 1.0 s @ 30 Hz
    };

    void configure(const Params2D& params, const Config& cfg);
    BlueprintResult generate();
    Config& config() { return cfg_; }
    const Config& config() const { return cfg_; }

private:
    struct EsdfGrid {
        double res = 0.1;
        Vec2d min_bounds{-20.0, -20.0};
        int w = 0, h = 0;
        std::vector<double> dist;  // drone-centre→surface distance per cell
        std::vector<int> comp;     // free component id (-1 = not free)
    };

    EsdfGrid buildEsdf(const BlueprintScene& scene) const;
    void labelComponents(EsdfGrid& grid) const;
    /// Generate ONE scene, retrying the whole scene with a fresh attempt
    /// seed until the requested count is met (or attempts are exhausted).
    BlueprintScene makeScene(uint64_t scene_id, uint64_t seed) const;
    /// Analytically re-verify a start/goal pair (surface clearance +
    /// boundary + distance stratum).  Returns false on ANY violation.
    bool verifyEndpointPair(const BlueprintScene& scene, double sx, double sy,
                            double gx, double gy, int distance_stratum) const;
    /// Sample one connected task: start AND goal from free-cell CENTRES of
    /// the SAME component, with the distance in the requested stratum.
    bool sampleOneTask(const EsdfGrid& g, const BlueprintScene& scene,
                       uint64_t seed, uint64_t task_id, int distance_stratum,
                       BlueprintTask& out) const;
    void preflightTask(BlueprintTask& task, const BlueprintScene& scene,
                       uint64_t tick_base) const;
    void assignGeometryClasses(BlueprintTask& task,
                               const BlueprintScene& scene) const;
    /// Actual density class from the REAL obstacle count (sparse/medium/dense)
    /// using the configured classification thresholds.
    std::string densityClassOf(double count) const;
    /// Actual radius class from the REAL maximum obstacle radius; empty
    /// scenes are ALWAYS "none" (never "small").
    std::string radiusClassOf(double max_radius, bool is_empty) const;
    /// Greedy deterministic quota selector.  Returns the accepted tasks,
    /// fills `unmet_quotas`, `category_counts` and `per_scene_accepted`.
    std::vector<BlueprintTask> applyQuotas(
        std::vector<BlueprintTask> preflighted,
        std::vector<std::string>& unmet_quotas,
        std::map<std::string, uint64_t>& category_counts,
        std::vector<uint64_t>& per_scene_accepted) const;

    Params2D p_;
    Config cfg_;
    bool configured_ = false;
};

}  // namespace expert
}  // namespace il_dataset
