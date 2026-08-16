#pragma once
/// @file   blueprint_types.hpp
/// @brief  Data structures for the NEW offline Scene/Task Blueprint
///         generation pipeline (il_dataset-feature-v4).
///
/// Everything in this header is PURE C++17 + Eigen, no ROS / Unity /
/// Flightmare.  It is the single source of:
///   * WarehouseGeometry  — the ONLY warehouse coordinate definition
///     (free region [-7,10] x [0,30] + 1 m wall envelope).  Scene
///     generator, task generator, preflight and truth audit all read the
///     same structure; no hard-coded duplicate ranges anywhere else.
///   * Histogram1D        — fixed-edge integer histogram (5 Hz correction
///     angle / correction distance, 30 Hz deflection / yaw rate / speed,
///     yaw error, path length, stretch ratio).
///   * SceneProfile       — the profile catalog + random realization
///     parameters (empty / sparse_tiny ... mixed_all / clustered /
///     corridor / bottleneck / chicane / central_blocker / edge_clutter).
///   * SceneMetadata      — per-scene statistics (radius bands, density
///     proxy, cluster count, free-space ratio, corridor width proxy).
///   * TaskGeomType       — geometric PROXY task classes used to bias
///     candidate sampling (the real class is re-decided by preflight).
///   * TaskDistributionSummary — the per-task structured summary that
///     flows into the global DistributionAccumulator.
///   * DistributionTarget / DistributionAccumulator / DistributionDeficit
///     — the quota replacement: soft targets + hard minimums + greedy
///     scoring.
///   * BlueprintGenerationConfig — all budgets / thresholds / targets.
///
/// Frame note: obstacle / start / goal coordinates are in the EXPERT 2D
/// frame (X right, Y up) which equals the Flightmare world XY (identity
/// horizontal map, see coordinate_adapter.hpp).  `initial_yaw` stored in
/// BlueprintTask is the Flightmare yaw (convention B), converted by
/// TaskCandidateGenerator.

#include "il_dataset/hierarchical_expert/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

// ═══════════════════════════════════════════════════════════════════
//  Rng — deterministic per-scene / per-task RNG (mt19937_64)
//  (moved out of scene_task_blueprint.cpp's anonymous namespace so every
//   blueprint component shares the SAME deterministic helper)
// ═══════════════════════════════════════════════════════════════════
class Rng {
public:
    explicit Rng(uint64_t seed) : gen_(seed) {}
    /// Re-seed (fresh per task / per scene attempt).
    void seed(uint64_t s) { gen_.seed(s); }
    double uniform(double lo, double hi) {
        if (!(hi > lo)) return lo;
        std::uniform_real_distribution<double> d(lo, hi);
        return d(gen_);
    }
    int uniformInt(int lo, int hi) {
        if (hi < lo) std::swap(lo, hi);
        std::uniform_int_distribution<int> d(lo, hi);
        return d(gen_);
    }
    /// Draw one index from a weight vector (weights must be >= 0, sum>0).
    int weightedPick(const std::vector<double>& weights) {
        double total = 0.0;
        for (double w : weights) total += std::max(0.0, w);
        if (total <= 0.0) return 0;
        double r = uniform(0.0, total);
        for (size_t i = 0; i < weights.size(); ++i) {
            r -= std::max(0.0, weights[i]);
            if (r <= 0.0) return static_cast<int>(i);
        }
        return static_cast<int>(weights.size()) - 1;
    }

private:
    std::mt19937_64 gen_;
};

// ═══════════════════════════════════════════════════════════════════
//  WarehouseGeometry — THE single warehouse coordinate definition
// ═══════════════════════════════════════════════════════════════════
//  Internal free region : x in [free_min_x, free_max_x],
//                         y in [free_min_y, free_max_y]  ([-7,10]x[0,30]).
//  Obstacle generation, start/goal generation and flight tasks may only
//  use the free region.  The 1 m outer shell (wall_extension_m) is the
//  warehouse envelope — non-traversable, no fake internal static
//  structure is ever constructed.
struct WarehouseGeometry {
    double free_min_x = -7.0;
    double free_max_x = 10.0;
    double free_min_y = 0.0;
    double free_max_y = 30.0;
    double wall_extension_m = 1.0;

    Vec2d freeMin() const { return Vec2d(free_min_x, free_min_y); }
    Vec2d freeMax() const { return Vec2d(free_max_x, free_max_y); }
    Vec2d envelopeMin() const {
        return Vec2d(free_min_x - wall_extension_m,
                     free_min_y - wall_extension_m);
    }
    Vec2d envelopeMax() const {
        return Vec2d(free_max_x + wall_extension_m,
                     free_max_y + wall_extension_m);
    }
    double width() const { return free_max_x - free_min_x; }
    double height() const { return free_max_y - free_min_y; }
    double area() const { return std::max(0.0, width()) * std::max(0.0, height()); }

    /// Point inside the FREE region (with an optional inset margin).
    bool inFree(double x, double y, double margin = 0.0) const {
        return x >= free_min_x + margin && x <= free_max_x - margin &&
               y >= free_min_y + margin && y <= free_max_y - margin;
    }
    /// Point inside the outer warehouse envelope (walls included).
    bool inEnvelope(double x, double y) const {
        return x >= free_min_x - wall_extension_m &&
               x <= free_max_x + wall_extension_m &&
               y >= free_min_y - wall_extension_m &&
               y <= free_max_y + wall_extension_m;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  Histogram1D — fixed-edge integer histogram (NaN-safe: non-finite
//  values are dropped)
// ═══════════════════════════════════════════════════════════════════
struct Histogram1D {
    std::vector<double> edges;     // n+1 strictly increasing edges
    std::vector<uint64_t> counts;  // n bins

    void configure(const std::vector<double>& e) {
        edges = e;
        std::sort(edges.begin(), edges.end());
        // Deduplicate (keep first occurrence of each value).
        std::vector<double> uniq;
        for (double v : edges) {
            if (uniq.empty() || v > uniq.back() + 1e-12) uniq.push_back(v);
        }
        edges = std::move(uniq);
        if (edges.size() < 2) edges = {0.0, 1.0};
        counts.assign(edges.size() - 1, 0);
    }

    void clear() { std::fill(counts.begin(), counts.end(), 0); }

    bool valid() const { return edges.size() >= 2 && counts.size() == edges.size() - 1; }

    /// Bin index for value v (clamped to the first/last bin).
    int binOf(double v) const {
        if (!valid() || !std::isfinite(v)) return -1;
        if (v <= edges.front()) return 0;
        if (v >= edges.back()) return static_cast<int>(edges.size()) - 2;
        const auto it = std::upper_bound(edges.begin(), edges.end(), v);
        return static_cast<int>((it - edges.begin()) - 1);
    }

    void add(double v) {
        const int b = binOf(v);
        if (b >= 0 && b < static_cast<int>(counts.size())) ++counts[b];
    }

    uint64_t at(int i) const {
        return (i >= 0 && i < static_cast<int>(counts.size())) ? counts[i] : 0;
    }
    uint64_t total() const {
        uint64_t s = 0;
        for (uint64_t c : counts) s += c;
        return s;
    }
    std::string binLabel(int i) const {
        if (!valid() || i < 0 || i >= static_cast<int>(edges.size()) - 1)
            return "";
        return std::to_string(edges[i]) + "_" + std::to_string(edges[i + 1]);
    }
};

// ═══════════════════════════════════════════════════════════════════
//  SceneStructure / SceneProfile — the profile catalog
// ═══════════════════════════════════════════════════════════════════
enum class SceneStructure : uint8_t {
    EMPTY = 0,
    UNIFORM = 1,
    CLUSTERED = 2,
    CORRIDOR = 3,
    BOTTLENECK = 4,
    CHICANE = 5,
    CENTRAL_BLOCKER = 6,
    EDGE_CLUTTER = 7,
};

inline const char* sceneStructureName(SceneStructure s) {
    switch (s) {
        case SceneStructure::EMPTY: return "empty";
        case SceneStructure::UNIFORM: return "uniform";
        case SceneStructure::CLUSTERED: return "clustered";
        case SceneStructure::CORRIDOR: return "corridor";
        case SceneStructure::BOTTLENECK: return "bottleneck";
        case SceneStructure::CHICANE: return "chicane";
        case SceneStructure::CENTRAL_BLOCKER: return "central_blocker";
        case SceneStructure::EDGE_CLUTTER: return "edge_clutter";
    }
    return "uniform";
}

struct SceneProfile {
    std::string name = "empty";
    int count_min = 0, count_max = 0;
    double radius_min = 0.1, radius_max = 1.0;
    // "log_uniform" | "uniform" | "fixed"
    std::string radius_mode = "log_uniform";
    double fixed_radius = 0.1;
    SceneStructure structure = SceneStructure::UNIFORM;
    int cluster_count = 0;
    double cluster_spread_m = 0.0;  // cluster disk radius
    double passage_width_m = 0.0;   // corridor / bottleneck / chicane
    double weight = 1.0;
    std::vector<std::string> tags;  // "sparse","dense","tiny","large","blocker"...
};

// ═══════════════════════════════════════════════════════════════════
//  SceneMetadata — per-scene statistics recorded in the manifest
// ═══════════════════════════════════════════════════════════════════
struct SceneMetadata {
    std::string profile = "empty";
    int obstacle_count = 0;
    double radius_min = 0.0, radius_max = 0.0, radius_mean = 0.0;
    int tiny_count = 0, small_count = 0, medium_count = 0, large_count = 0;
    // obstacles per m^2, scaled by 1000 for readability
    double local_density_proxy = 0.0;
    double largest_obstacle_radius = 0.0;
    uint64_t scene_seed = 0;
    int generation_attempt = 0;
    int cluster_count = 0;
    // main free component area / free region area (filled by geometry cache)
    double free_space_ratio = 1.0;
    // narrowest obstacle-pair surface gap in the scene (0 when <2 obstacles)
    double estimated_corridor_width = 0.0;
    bool geometry_valid = true;
    std::string geometry_failure_reason;
    bool planning_valid = true;   // main component area >= min area
    std::string planning_failure_reason;
};

// ═══════════════════════════════════════════════════════════════════
//  TaskGeomType — geometric PROXY classes (sampling bias + recording)
// ═══════════════════════════════════════════════════════════════════
enum class TaskGeomType : uint8_t {
    CLEAR = 0,
    LOCAL_AVOIDANCE = 1,
    OFFSET_AVOIDANCE = 2,
    LARGE_OCCLUSION = 3,
    MULTI_OBSTACLE = 4,
    CHICANE = 5,
    NARROW_BUT_PLANNABLE = 6,
    LONG_DETOUR = 7,
};

inline const char* taskGeomTypeName(TaskGeomType t) {
    switch (t) {
        case TaskGeomType::CLEAR: return "CLEAR";
        case TaskGeomType::LOCAL_AVOIDANCE: return "LOCAL_AVOIDANCE";
        case TaskGeomType::OFFSET_AVOIDANCE: return "OFFSET_AVOIDANCE";
        case TaskGeomType::LARGE_OCCLUSION: return "LARGE_OCCLUSION";
        case TaskGeomType::MULTI_OBSTACLE: return "MULTI_OBSTACLE";
        case TaskGeomType::CHICANE: return "CHICANE";
        case TaskGeomType::NARROW_BUT_PLANNABLE: return "NARROW_BUT_PLANNABLE";
        case TaskGeomType::LONG_DETOUR: return "LONG_DETOUR";
    }
    return "CLEAR";
}

/// Number of geometric proxy classes (shared by the sampler and the
/// distribution analyzer; never hard-code 8 elsewhere).
inline constexpr int kNumTaskGeomTypes = 8;

// ═══════════════════════════════════════════════════════════════════
//  DepthProxySample — one synthetic 2D raycast depth sample
// ═══════════════════════════════════════════════════════════════════
struct DepthProxySample {
    uint64_t near_count = 0, mid_count = 0, far_count = 0, free_count = 0;
    uint64_t occupied_rays = 0, total_rays = 0;
    double min_visible = std::numeric_limits<double>::infinity();
    double sum_visible = 0.0;
    uint64_t visible_count = 0;
    // largest angular span (deg) of consecutive rays that hit within range
    double max_angular_occlusion_deg = 0.0;

    void reset() {
        near_count = mid_count = far_count = free_count = 0;
        occupied_rays = total_rays = 0;
        min_visible = std::numeric_limits<double>::infinity();
        sum_visible = 0.0;
        visible_count = 0;
        max_angular_occlusion_deg = 0.0;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  TaskDistributionSummary — per-task structured distribution summary
// ═══════════════════════════════════════════════════════════════════
//  Every preflight candidate produces one; the global accumulator merges
//  them.  All angles are signed with the expert-frame sign convention
//  (+ = LEFT).  5 Hz correction angle / 30 Hz deflection are computed in
//  the shared body frame (+X forward, +Y left) so the sign is identical
//  between the expert 2D frame and the FLU body frame.
struct TaskDistributionSummary {
    uint64_t task_id = 0;
    uint64_t scene_id = 0;
    std::string scene_profile;
    std::string task_geom_type = "CLEAR";

    // geometry / path
    double straight_distance_m = 0.0;
    double preflight_path_length_m = 0.0;
    double path_stretch_ratio = 0.0;
    double preflight_duration_s = 0.0;
    uint64_t preflight_ticks = 0;
    double initial_yaw_error_signed_deg = 0.0;
    double initial_yaw_error_abs_deg = 0.0;

    // depth proxy (raw counts; ratios derived on demand)
    uint64_t depth_samples = 0;
    uint64_t depth_near_count = 0, depth_mid_count = 0;
    uint64_t depth_far_count = 0, depth_free_count = 0;
    uint64_t depth_visible_count = 0;  // rays that hit within range
    // Running mean starts at 0 (never inf: inf*0 = NaN in the first merge).
    double depth_min_visible_m = std::numeric_limits<double>::infinity();
    double depth_mean_visible_m = 0.0;
    double depth_max_angular_occlusion_deg = 0.0;
    double depth_occupied_ray_ratio = 0.0;

    // 5 Hz tick-level macro directives
    uint64_t macro_tick_total = 0;
    uint64_t macro_pass_count = 0;
    uint64_t macro_normal_count = 0;
    uint64_t macro_turn_left_count = 0;
    uint64_t macro_turn_right_count = 0;
    Histogram1D macro_correction_angle_hist;    // signed deg
    Histogram1D macro_correction_distance_hist; // normalized 0..1

    // 30 Hz tick-level local behaviour
    uint64_t local_direct_count = 0;
    uint64_t local_avoidance_count = 0;
    Histogram1D local_deflection_hist;  // signed deg
    Histogram1D local_yaw_rate_hist;    // rad/s
    Histogram1D local_speed_hist;       // m/s
    double min_observed_clearance_m = std::numeric_limits<double>::infinity();
    double mean_observed_clearance_m = std::numeric_limits<double>::infinity();

    // quality
    bool reached_goal = false;
    bool collision = false;
    bool out_of_bounds = false;
    double minimum_clearance_m = std::numeric_limits<double>::infinity();

    double nearDepthRatio() const {
        return depth_samples ? static_cast<double>(depth_near_count) /
                                   static_cast<double>(depth_samples)
                             : 0.0;
    }
    double midDepthRatio() const {
        return depth_samples ? static_cast<double>(depth_mid_count) /
                                   static_cast<double>(depth_samples)
                             : 0.0;
    }
    double farDepthRatio() const {
        return depth_samples ? static_cast<double>(depth_far_count) /
                                   static_cast<double>(depth_samples)
                             : 0.0;
    }
    double freeDepthRatio() const {
        return depth_samples ? static_cast<double>(depth_free_count) /
                                   static_cast<double>(depth_samples)
                             : 0.0;
    }
};

// ═══════════════════════════════════════════════════════════════════
//  DistributionTarget / DistributionAccumulator / DistributionDeficit
// ═══════════════════════════════════════════════════════════════════
//  metric grammar:
//    "count:<key>"                    → flat counter in the accumulator
//    "hist:<name>"                    → total of a histogram
//    "hist:<name>:<bin>"              → one bin of a histogram
//    "frac:<num>/<den>"               → count(num)/max(1,count(den))  [verify only]
//    "balance:<left>/<right>"         → |l-r|/(l+r+1)  [verify only]
struct DistributionTarget {
    std::string key;        // unique id for reporting
    std::string metric;     // see grammar above
    int bin_index = -1;     // for "hist:<name>:<bin>"
    std::string num_key;    // for frac/balance
    std::string den_key;
    double target = 0.0;    // ideal count / fraction
    double minimum = 0.0;   // hard minimum (violation -> generation_ok=false)
    double maximum = 1e18;  // soft cap
    double weight = 1.0;
    double tolerance = 0.0; // relative tolerance on target
    bool minimum_hard = true;
};

// The DistributionAccumulator (merges task summaries) and the target /
// deficit / selection logic live in distribution_analyzer.hpp.

// ═══════════════════════════════════════════════════════════════════
//  BlueprintGenerationConfig — the full new-style pipeline config
// ═══════════════════════════════════════════════════════════════════
struct BlueprintGenerationConfig {
    // ── warehouse (single source) ─────────────────────────────────
    WarehouseGeometry warehouse;

    // ── planner compatibility (reused from the existing expert) ───
    double vehicle_radius_m = 0.30;
    double navigation_clearance_m = 0.30;
    double clearance_discretization_margin_m = 0.05;
    double generation_margin_m = 0.05;  // extra safety margin (added once per side)
    double min_surface_gap_m = 1.40;    // validated >= planner-required passage
    double boundary_margin_m = 1.20;
    double free_cell_surface_clearance_m = 0.50;  // drone centre -> surface
    double esdf_resolution_m = 0.10;
    double min_main_component_area_m2 = 60.0;

    /// Planner-required traversable passage between two obstacle surfaces:
    ///   2 * (vehicle_radius + navigation_clearance + discretisation margin)
    ///   + 2 * generation_margin.
    double plannerRequiredPassage() const {
        return 2.0 * (vehicle_radius_m + navigation_clearance_m +
                      clearance_discretization_margin_m) +
               2.0 * generation_margin_m;
    }

    // ── scene profiles ─────────────────────────────────────────────
    std::vector<SceneProfile> profiles;   // default catalog (see .cpp)
    bool use_profile_catalog = true;
    // optional explicit profile schedule; when non-empty the controller
    // realizes these profiles in order (rounds then repeat with deficits)
    std::vector<std::string> profile_sequence;

    // ── task generation ────────────────────────────────────────────
    double min_task_distance_m = 4.0;
    double max_task_distance_m = 28.0;  // free region allows up to ~34 m
    double flight_height_m = 2.0;
    double obstacle_height_m = 8.0;
    int task_sample_attempts = 300;     // per candidate start/goal draws
    int task_goal_attempts = 120;       // per candidate goal draws

    // ── initial yaw (layered sampling) ─────────────────────────────
    std::vector<double> yaw_edges_deg{0.0, 15.0, 35.0, 55.0, 90.0, 150.0, 180.0};
    std::vector<double> yaw_weights{0.8, 1.2, 2.2, 1.6, 1.0, 0.9};

    // ── depth proxy (2D synthetic raycast, NOT a student input) ────
    int depth_proxy_num_rays = 96;
    int depth_proxy_sample_stride_ticks = 6;  // every 0.2 s @ 30 Hz
    double depth_near_max_m = 1.5;
    double depth_mid_max_m = 3.0;
    double depth_far_max_m = 5.0;  // == perception range (validated)

    // ── behaviour histograms ───────────────────────────────────────
    std::vector<double> correction_angle_edges_deg{-90, -60, -45, -30, -15,
                                                   0, 15, 30, 45, 60, 90};
    std::vector<double> correction_distance_edges{0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
    std::vector<double> deflection_edges_deg{-90, -60, -30, -10, 10, 30, 60, 90};
    std::vector<double> yaw_rate_edges{-2.0, -1.0, -0.5, -0.2, 0.0,
                                       0.2, 0.5, 1.0, 2.0};
    std::vector<double> speed_edges{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    double min_deflection_speed_mps = 0.10;  // below this: no deflection stats

    // ── path length classes (from PREFLIGHT path length) ───────────
    double path_short_max_m = 12.0;
    double path_long_min_m = 20.0;

    // ── distribution targets (defaults built in the .cpp) ──────────
    std::vector<DistributionTarget> targets;

    // ── performance budgets (strict, finite) ───────────────────────
    int max_scene_candidates = 64;
    int max_task_candidates_per_scene = 40;
    int max_generation_rounds = 6;
    int max_total_preflight_tasks = 1800;
    int max_preflight_ticks_per_task = 900;   // 30 s @ 30 Hz
    int max_scene_generation_attempts = 96;
    int max_task_generation_attempts = 600;
    bool parallel_tasks = false;  // optional; kept false by default
    double scene_switch_penalty = 0.25;

    // ── result requirements ────────────────────────────────────────
    int min_scenes = 4;
    int min_tasks = 24;
    int min_tasks_per_scene = 4;
    int max_tasks_per_scene = 12;
    int min_macro_ticks_per_class = 24;    // pass/normal/turn_left/turn_right
    int min_depth_samples_per_band = 40;   // near/mid/far/free
    int min_yaw_samples_per_bin = 3;       // per absolute-yaw bin
    int min_path_samples_per_class = 4;    // short/medium/long
    double max_turn_imbalance_ratio = 0.66;
    double max_yaw_imbalance_ratio = 0.66;

    // ── legacy strata thresholds (manifest compatibility only) ─────
    // Used to map realized scenes onto the legacy 3x3 density x radius
    // strata coverage report; the NEW pipeline balances profiles directly.
    double density_sparse_max = 7.0;
    double density_dense_min = 14.0;
    double radius_small_max_m = 0.6;
    double radius_large_min_m = 1.4;

    // ── base seed ──────────────────────────────────────────────────
    uint64_t base_seed = 260812;
};

// ═══════════════════════════════════════════════════════════════════
//  GenerationTiming — lightweight performance statistics
// ═══════════════════════════════════════════════════════════════════
struct GenerationTiming {
    double scene_generation_ms = 0.0;
    double scene_geometry_cache_ms = 0.0;
    double task_candidate_generation_ms = 0.0;
    double cheap_filter_ms = 0.0;
    double preflight_total_ms = 0.0;
    double preflight_average_ms = 0.0;
    uint64_t preflight_count = 0;
    uint64_t cheap_filter_rejected = 0;
    double depth_proxy_total_ms = 0.0;
    double selection_ms = 0.0;
    double total_ms = 0.0;

    std::map<std::string, double> asMap() const {
        return {{"scene_generation_ms", scene_generation_ms},
                {"scene_geometry_cache_ms", scene_geometry_cache_ms},
                {"task_candidate_generation_ms", task_candidate_generation_ms},
                {"cheap_filter_ms", cheap_filter_ms},
                {"preflight_total_ms", preflight_total_ms},
                {"preflight_average_ms", preflight_average_ms},
                {"depth_proxy_total_ms", depth_proxy_total_ms},
                {"selection_ms", selection_ms},
                {"total_ms", total_ms}};
    }
};

}  // namespace expert
}  // namespace il_dataset
