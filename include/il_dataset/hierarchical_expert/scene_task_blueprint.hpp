#pragma once
/// @file   scene_task_blueprint.hpp
/// @brief  FULL C++ scene / truth-ESDF / task / blueprint generation +
///         C++ behavior classifier + distribution-driven selection.
///
/// This replaces the Python TruthAudit / _generate_scene / _sample_task /
/// JointV2BehaviorClassifier.  Python (il_manager.py) only:
///   * configures the generator from YAML,
///   * calls generate() once (BOTH normal collection and blueprint_only),
///   * writes the manifest,
///   * then (normal mode only) collects strictly per the manifest.
///
/// The generation engine is BlueprintGenerationController (deficit-driven,
/// budgeted, deterministic).  The class below is the stable public entry
/// point; its structs are the manifest contract (backward compatible —
/// fields are added, never removed).
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
///  WAREHOUSE (single source): the only free region is
///  [-7,10] x [0,30] (BluePrintGenerationConfig::warehouse); the 1 m outer
///  shell is the wall envelope.  Scene generator, task generator, cheap
///  filter, preflight and truth audit all read the SAME WarehouseGeometry.
///
///  SCENES (profiles, not strata): scenes are random realizations of
///  SceneProfile entries (empty / sparse_tiny ... mixed_all / clustered /
///  corridor / bottleneck / chicane / central_blocker / edge_clutter).
///
///  TASKS (fast candidates): start/goal are sampled from the pre-computed
///  VALID task cells of the main connected component (SceneGeometryCache
///  built ONCE per scene — never rebuilt per task), classified by a cheap
///  geometric PROXY (CLEAR ... LONG_DETOUR) and preflighted with the SAME
///  expert.  Each preflight produces a TaskDistributionSummary (path /
///  depth proxy / 5 Hz tick-level macro / 30 Hz deflection / yaw / stretch)
///  that flows into the global DistributionAnalyzer.
///
///  SELECTION (distribution targets, not hard quotas): a deterministic
///  greedy selector scores each candidate by its contribution to the
///  currently-deficient histogram/count targets (minus over-supply
///  penalties), with a scene_switch_penalty so the final blueprint prefers
///  fewer scenes with more complementary tasks each.  generation_ok is
///  false ONLY when a hard minimum coverage / structural balance check
///  fails; unmet SOFT targets are warnings + remaining_deficits.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/types.hpp"

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
    // ── new: scene profile + full metadata ─────────────────────────
    std::string profile = "empty";
    // Orientation recorded at realization time (NONE/HORIZONTAL/VERTICAL).
    // Single source for validation + manifest; never re-derived.
    StructureOrientation structure_orientation = StructureOrientation::NONE;
    SceneMetadata metadata;
    // ── legacy fields (report-only, manifest compatibility) ────────
    int stratum_id = -1;   // legacy 3x3 stratum id; -1 for empty
    int count_stratum = -1;  // legacy density stratum
    int radius_stratum = -1; // legacy radius stratum
    bool is_empty = false;   // explicit CLEAR scene with 0 obstacles
    int requested_obstacle_count = 0;
    int actual_obstacle_count = 0;
    bool generation_valid = true;
    std::string failure_reason = "";
    std::vector<BlueprintObstacle> obstacles;
    std::string planned_density_class = "";  // legacy (== actual for profiles)
    std::string planned_radius_class = "";   // legacy
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
    bool truth_brake_triggered = false;  // runtime edge-clearance judge
    bool out_of_bounds = false;        // drone disk crossed the region boundary
    bool macro_label_ok = true;
    bool qualification_exceeded = false;  // ran out of the preflight tick budget
    uint64_t preflight_ticks = 0;
    double min_truth_clearance_m = 1e9;
    double goal_distance_m = 0.0;
    std::string preflight_status = "";
    // ── new: actual path statistics (replaces Euclidean-only length) ─
    double straight_distance_m = 0.0;
    double path_length_m = 0.0;
    double path_stretch_ratio = 1.0;
    double preflight_duration_s = 0.0;
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
    std::string distance_class = "medium";  // short/medium/long (path-based)
    std::string side_class = "none";  // left / right / both / none
    // ── REAL macro-directive statistics (item 六) ──────────────────
    bool saw_turn_left = false;
    bool saw_turn_right = false;
    bool saw_normal_correction = false;
    uint64_t turn_update_count = 0;    // TURN_LEFT/RIGHT update frames
    uint64_t normal_update_count = 0;  // NORMAL_CORRECTION update frames
    BlueprintTaskAudit audit;
    // ── new: geometric proxy class + full distribution summary ─────
    std::string geom_type = "CLEAR";
    TaskDistributionSummary summary;
    double selection_score = 0.0;  // final greedy selection score
    // ── behaviour SEGMENT label counts (six classes) observed in the
    //    preflight trajectory — used by the offline test blueprint to
    //    measure multi-avoidance / consecutive-avoidance coverage. ──
    std::map<std::string, uint64_t> segment_label_counts;
    // ── privileged task-qualification diagnostics (manifest only; never
    //    a student input).  Side routes / blocker / stretch are PRIVILEGED
    //    truth and are NEVER fed to the expert. ──────────────────────
    TaskQualificationSummary qualification;
};

struct BlueprintResult {
    bool generation_ok = false;
    std::string failure_reason = "";
    std::vector<std::string> unmet_quotas;
    std::vector<BlueprintScene> scenes;
    /// selected tasks (normal collection iterates EXACTLY these)
    std::vector<BlueprintTask> tasks;
    /// all preflight-accepted tasks (the candidate pool)
    std::vector<BlueprintTask> preflighted;
    uint64_t requested_scenes = 0;
    uint64_t requested_tasks_per_scene = 0;
    uint64_t scenes_generated = 0;
    uint64_t scenes_valid = 0;
    uint64_t tasks_sampled = 0;      // sampled start/goal candidates
    uint64_t tasks_preflighted = 0;  // candidates actually preflighted
    uint64_t tasks_pool_target = 0;  // sum of per-scene pool targets
    uint64_t tasks_pool_accepted = 0;  // preflight-accepted candidate pool
    uint64_t tasks_quota_accepted = 0;  // final selected tasks
    uint64_t total_task_candidates = 0;  // preflight-accepted per scene, summed
    uint64_t preflight_success_tasks = 0;
    uint64_t cheap_filter_rejected = 0;
    uint64_t qualification_rejected = 0;  // rejected by the privileged gate
    bool pool_budget_exhausted = false;
    uint64_t strata_required = 0;   // legacy report
    uint64_t strata_covered = 0;    // legacy report
    std::vector<uint64_t> strata_covered_flags;  // legacy report
    /// per-scene selected-task counts (indexed by scene_id)
    std::vector<uint64_t> per_scene_accepted;
    /// final per-category counts: "behavior:clear", "density:sparse", ...
    std::map<std::string, uint64_t> category_counts;
    uint64_t base_seed = 0;
    // ── new: distribution report (manifest) ────────────────────────
    std::map<std::string, uint64_t> distribution_counts;
    std::map<std::string, std::vector<uint64_t>> distribution_histograms;
    std::vector<std::string> remaining_deficits;
    std::vector<std::string> warnings;
    uint64_t generation_rounds = 0;
    std::map<std::string, double> timing_ms;
    std::vector<uint64_t> selected_scene_ids;
    bool hard_minimums_met = false;
    bool soft_targets_met = false;
    // ── new: efficiency + budget diagnostics ───────────────────────
    uint64_t preflight_attempt_count = 0;     // total preflight calls
    uint64_t preflight_success_count = 0;     // accepted candidates
    uint64_t preflight_failure_count = 0;     // rejected candidates
    uint64_t total_preflight_ticks = 0;       // 30 Hz ticks, all attempts
    uint64_t full_preflight_attempted = 0;    // attempts NOT early-terminated
    uint64_t full_preflight_success = 0;      // accepted AND ran to completion
    uint64_t selected_scene_count = 0;        // scenes in the final selection
    double preflight_acceptance_ratio = 0.0;  // success / attempts
    double selected_per_preflight_ratio = 0.0;  // selected / attempts
    std::string budget_exhausted_reason = "none";
    std::vector<RoundStats> round_logs;
    // ── privileged task-qualification efficiency (aggregate) ───────
    uint64_t task_candidates_generated = 0;
    uint64_t endpoint_pass_count = 0;
    uint64_t connectivity_pass_count = 0;
    uint64_t straight_clear_count = 0;
    uint64_t blocked_count = 0;
    uint64_t side_qualification_attempt_count = 0;
    uint64_t both_sides_feasible_count = 0;
    uint64_t qualification_accept_count = 0;
    uint64_t total_astar_expansions = 0;
    double qualification_pass_ratio = 0.0;
    double full_preflight_success_after_qualification_ratio = 0.0;
    QualificationCounters qualification;
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
//  Scene / ESDF / task / blueprint generator + classifier + selection
// ═══════════════════════════════════════════════════════════════════
//  Public entry point (pybind / il_manager).  The heavy pipeline lives in
//  BlueprintGenerationController; this class is a thin configured facade
//  that keeps the pybind + manifest contract stable.
class SceneTaskBlueprintGenerator {
public:
    // ── legacy Config fields (backward compatible; the NEW pipeline is
    //    driven by `blueprint` below, with these as the fallback) ───
    struct Config {
        int scene_count = 10;
        int tasks_per_scene = 8;
        int minimum_tasks_per_scene = 6;
        uint64_t base_seed = 260812;
        double flight_height_m = 2.0;
        double obstacle_height_m = 8.0;
        bool require_full_strata_coverage = true;
        double min_surface_gap_m = 1.40;
        double boundary_margin_m = 1.20;
        double radius_min_m = 0.10;
        double radius_max_m = 6.00;
        int max_obstacles = 30;
        double vehicle_radius_m = 0.30;
        double navigation_clearance_m = 0.30;
        double free_cell_surface_clearance_m = 0.50;
        double esdf_resolution_m = 0.10;
        int max_generation_attempts = 96;
        double min_task_distance_m = 4.0;
        double max_task_distance_m = 28.0;
        double initial_yaw_bias_deg = 15.0;   // legacy; replaced by yaw strata
        int task_sample_attempts = 300;
        int candidate_pool_multiplier = 4;
        uint64_t qualification_attempt_budget = 600;
        uint64_t preflight_qualification_max_ticks = 900;
        int min_per_behavior = 2;      // legacy (informational)
        int min_turn_per_side = 2;
        int max_left_right_imbalance = 2;
        int min_per_density_level = 4;
        int min_per_radius_level = 4;
        int min_per_distance_level = 4;
        double distance_short_max_m = 9.0;
        double distance_long_min_m = 15.0;
        double radius_small_max_m = 0.6;
        double radius_large_min_m = 1.4;
        double density_sparse_max = 7.0;
        double density_dense_min = 14.0;
        uint64_t long_takeover_min_ticks = 30;
        // ── new: the full blueprint-generation config (YAML
        //    blueprint_generation section); when left at defaults the
        //    controller falls back to the legacy fields above ───────
        BlueprintGenerationConfig blueprint;
        bool blueprint_explicit = false;
    };

    void configure(const Params2D& params, const Config& cfg);
    BlueprintResult generate();
    Config& config() { return cfg_; }
    const Config& config() const { return cfg_; }

private:
    /// Build the effective BlueprintGenerationConfig (from the explicit
    /// blueprint section, or from the legacy fields when not provided).
    BlueprintGenerationConfig makeBlueprintConfig() const;
    Params2D p_;
    Config cfg_;
    bool configured_ = false;
};

}  // namespace expert
}  // namespace il_dataset
