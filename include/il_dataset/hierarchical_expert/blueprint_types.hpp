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

/// A single 2.5D cylinder obstacle (x, y, radius, height).  Shared by the
/// blueprint types and the scene/task blueprint result types.
struct BlueprintObstacle {
    double x = 0.0, y = 0.0;
    double radius = 0.0;
    double height_m = 8.0;
    int id = -1;
};

/// An axis-aligned known-obstacle rectangle (point-cloud occupancy cluster
/// bounding box).  Cells inside [min,max] are treated as occupied in the
/// scene grid; random cylinders keep a surface gap away from it.
struct KnownRect {
    double min_x = 0.0, max_x = 0.0;
    double min_y = 0.0, max_y = 0.0;
    double height_m = 8.0;
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

/// Orientation of a directional structured scene (corridor / bottleneck /
/// chicane).  Recorded ONCE by the SceneProfileGenerator at realization
/// time and reused by validation / manifest / debug — never re-derived
/// from the obstacle spread by a heuristic.
///   HORIZONTAL : the passage runs along the X axis (obstacles alternate
///                around the X centre-line in Y)
///   VERTICAL   : the passage runs along the Y axis (obstacles alternate
///                around the Y centre-line in X)
enum class StructureOrientation : uint8_t {
    NONE = 0,
    HORIZONTAL = 1,
    VERTICAL = 2,
};

inline const char* structureOrientationName(StructureOrientation o) {
    switch (o) {
        case StructureOrientation::HORIZONTAL: return "horizontal";
        case StructureOrientation::VERTICAL: return "vertical";
        default: return "none";
    }
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
    // Orientation of directional structures, recorded at realization time
    // ("horizontal" / "vertical" / "none").  Single source — validation
    // and manifest both use it (never re-derived heuristically).
    std::string structure_orientation = "none";
    int obstacle_count = 0;
    double radius_min = 0.0, radius_max = 0.0, radius_mean = 0.0;
    int tiny_count = 0, small_count = 0, medium_count = 0, large_count = 0;
    // obstacles per m^2, scaled by 1000 for readability
    double local_density_proxy = 0.0;
    // 占地密度 Σ(π r²)/free-region-area (0..1) — density classification
    // and reporting use this (replaces obstacle-count based density).
    double occupancy_ratio = 0.0;
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

    // ── Clearance hierarchy (port of the 2D reference semantics).  Every
    //    component (SceneGenerator / SceneGeometryCache / TaskSampler /
    //    RouteQualifier / LocalPlanner) MUST use these accessors instead of
    //    scattering magic numbers.  The 2D reference uses:
    //      scene_safety_clearance 0.5  -> endpoint/connectivity
    //      route clearance = safety + macro_route_clearance_margin(0.1)
    //                        + clearance_discretization_margin(0.05) = 0.65
    //    Mapped to the 2.5D parameters below (same values, formula kept).
    /// Endpoint (start/goal centre->surface) clearance: the drone centre
    /// must sit on a cell strictly beyond this from any obstacle surface.
    double endpointRequiredClearance() const {
        return free_cell_surface_clearance_m;
    }
    /// Connectivity / traversable-component clearance: the same base
    /// safety clearance as endpoints (strict `>`).
    double connectivityRequiredClearance() const {
        return free_cell_surface_clearance_m;
    }
    /// LocalPlanner-required clearance from an obstacle surface to the
    /// drone centre (per side): vehicle radius + navigation clearance +
    /// grid discretisation margin.  A corridor narrower than twice this is
    /// not traversable by the current planner.
    double plannerRequiredClearance() const {
        return vehicle_radius_m + navigation_clearance_m +
               clearance_discretization_margin_m;
    }
    /// Required passage width (obstacle-surface to obstacle-surface) for
    /// the current planner — the planner-required traversable passage.
    double requiredPassageWidth() const { return plannerRequiredPassage(); }
    /// Route-qualification clearance (side A* / causal qualification): the
    /// privileged global-route search keeps this much centre->surface
    /// clearance so the local planner actually has room to execute.
    double routeQualificationClearance() const {
        return free_cell_surface_clearance_m + route_clearance_margin_m +
               clearance_discretization_margin_m;
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
    // NEW: reject super-detours.  A start/goal on opposite sides of a wall
    // / big building is A*-connected only by walking around it; cap the
    // greedy-A* route at stretch_ratio x straight + slack so such tasks are
    // never admitted (they waste a long episode on a useless detour).
    double max_route_stretch_ratio = 4.0;
    double max_route_stretch_slack_m = 3.0;
    double flight_height_m = 2.0;
    // NEW: per-task random operating height (2.5D).  When the range is
    // left at the default (equal to flight_height_m) every task keeps the
    // legacy single height; otherwise each task draws U[min,max] and its
    // start/goal share that height (start/goal height are equal).
    double flight_height_min_m = 2.0;
    double flight_height_max_m = 2.0;
    double obstacle_height_m = 8.0;
    // NEW: per-obstacle random height.  Defaults keep the legacy single
    // height; otherwise each cylinder draws U[min,max] (must exceed the
    // operating height so the cylinder spans the flight band).
    double obstacle_height_min_m = 8.0;
    double obstacle_height_max_m = 8.0;
    // NEW: fixed known-obstacle AABBs (point-cloud occupancy clusters of
    // the real Unity scene).  Injected into EVERY generated scene so
    // obstacle placement, start/goal sampling and A* connectivity all
    // account for them.  The AABB cells are marked occupied in the scene
    // grid and random cylinders never overlap them.
    std::vector<KnownRect> known_rects;
    int task_sample_attempts = 300;     // per candidate start/goal draws
    int task_goal_attempts = 120;       // per candidate goal draws
    // Extra route margin (m) on top of the endpoint clearance used by the
    // privileged route qualification (mirrors the 2D macro_route_clearance
    // _margin=0.1).  routeQualificationClearance() =
    // endpointRequiredClearance() + route_clearance_margin_m +
    // clearance_discretization_margin_m.
    double route_clearance_margin_m = 0.10;

    // ── initial yaw (layered sampling) ─────────────────────────────
    std::vector<double> yaw_edges_deg{0.0, 15.0, 35.0, 55.0, 90.0, 150.0, 180.0};
    std::vector<double> yaw_weights{0.8, 1.2, 2.2, 1.6, 1.0, 0.9};

    // Rare macro-turn probe.  This is only a sampling hint; preflight must
    // still observe the requested TURN_LEFT/TURN_RIGHT label from the
    // closed-loop expert before admitting the candidate.
    bool macro_probe_enabled = true;
    double macro_probe_yaw_error_deg = 70.0;
    // A probe is a sampling hint; false keeps candidates whose observed
    // expert label differs (for example local yaw-first vs macro TURN).
    bool macro_probe_require_match = false;

    // Explore a random candidate pool before switching to deficit-driven
    // supplementation and final greedy selection.
    bool pool_first_exploration = true;
    int exploration_rounds = 2;
    uint64_t exploration_min_pool_tasks = 0;

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
    // Every full preflight call — success OR failure — consumes 1 unit.
    int max_total_preflight_tasks = 1800;
    // Total 30 Hz ticks across ALL preflight calls (success + failure).
    uint64_t max_total_preflight_ticks = 500000;
    int max_preflight_ticks_per_task = 900;   // 30 s @ 30 Hz
    int max_scene_generation_attempts = 96;
    int max_task_generation_attempts = 600;
    // Parallel closed-loop task preflight workers.  0 or 1 = fully serial
    // (identical behaviour to the pre-parallel pipeline); N >= 2 runs up
    // to N preflight simulations concurrently.  Sampling / cheap filter /
    // qualification stay serial; only the expensive 30 Hz preflight runs
    // in parallel and its outcomes are merged in submission order.
    int parallel_tasks = 0;  // worker threads for task preflight (0/1 = serial)
    // Legacy constant penalty (weak); the selector now uses a two-stage
    // coverage + scene-consolidation flow (see DistributionAnalyzer::select).
    double scene_switch_penalty = 0.10;

    // ── scene-level parallel generation (new architecture) ─────────
    // Main thread first generates scene_levels x scenes_per_level scene
    // SPECS (each level's cylinders sized by level_radius_min/max, the 10
    // scenes per level go sparse -> dense), then scene_parallel_threads
    // workers take over whole scenes: each worker builds the 2D grid,
    // samples many random start/goal pairs, runs a greedy toward-goal A*
    // connectivity check, balances short/medium/long (direct-line)
    // distances, generates tasks, and runs the QUICK expert preflight
    // (quick_preflight_max_ticks) to label them.  After all scenes finish
    // the main thread merges, balances labels across the 4 levels
    // (pick/drop/top-up, approximate), and reports the expected collection
    // count vs the target (expected_collect_tasks).
    bool scene_level_parallel = false;   // enable the scene-level pipeline
    int scene_parallel_threads = 8;      // worker threads for scenes
    int scene_levels = 4;                // small / medium / large / mixed
    int scenes_per_level = 10;           // scenes per level (sparse -> dense)
    // Optional per-level scene counts (sparse -> dense), overriding
    // scenes_per_level when its size == scene_levels.  Lets large/mixed
    // levels carry more scenes for more macro-planning labels.
    std::vector<int> scenes_per_level_list;
    // Cylinder radius bands per level: [min_m[i], max_m[i]].
    std::vector<double> level_radius_min_m{0.15, 0.5, 1.5, 0.15};
    std::vector<double> level_radius_max_m{0.5, 1.5, 3.0, 3.0};
    // Target occupied-area fraction per level (sparse edge -> dense edge).
    // makeSceneSpec derives the obstacle COUNT from occupancy_target x
    // free-area / E[π r²] — the same occupancy reads "not empty" for small
    // cylinders and "not crammed" for large ones (count alone does not).
    std::vector<double> level_occupancy_min{0.05, 0.07, 0.04, 0.08};
    std::vector<double> level_occupancy_max{0.08, 0.11, 0.07, 0.12};
    // Per-level adaptive distance floor: a medium/large level's minimum
    // start-goal distance is max(min_task_distance_m, scale * rmax), so
    // big cylinders are never asked to host a short path they cannot
    // physically connect.  small/mixed keep the full [min, max] band.
    double distance_min_radius_scale = 2.5;
    // Placement constraints: pairwise SURFACE gap >= obstacle_surface_gap
    // _min_m (must be traversable), obstacle centre >= boundary_min_m from
    // the free-region border.
    double obstacle_surface_gap_min_m = 1.6;
    double obstacle_boundary_min_m = 0.6;
    // Direct-line distance bands for the task balance (start-goal
    // Euclidean): short < short_max, medium [short_max, medium_max],
    // long > medium_max.  Sampled with a target ratio (1:1:1 default).
    double task_distance_short_max_m = 8.0;
    double task_distance_medium_max_m = 16.0;
    // Quick expert preflight tick cap (simplified expert: same 30 Hz
    // closed-loop expert, but each task runs at most this many ticks).
    int quick_preflight_max_ticks = 150;
    // Coarse-step multiplier for the QUICK preflight: the dynamics
    // integration step becomes dt_scale/30 s per tick, so the same
    // physical path needs dt_scale fewer ticks — a full start->goal
    // preflight finishes faster.  The expert decision stream is unchanged
    // (one decision per tick, 5 Hz macro every 6 ticks).  Use 1 for the
    // exact real-time rate.  NOTE: the preflight must reach the GOAL (not
    // a short truncated window) for the detour / long_detour segment
    // labels to be meaningful, hence the large tick cap below.
    double quick_preflight_dt_scale = 3.0;
    // Expected final collection size — used by the merge stage for the
    // "how many tasks per scene / per level" budget and the reasonableness
    // report.
    int expected_collect_tasks = 400;

    // ── synthetic observation / preflight behaviour ────────────────
    // Whether the warehouse wall envelope appears in the synthetic depth
    // observation (depth proxy + preflight patch).  NEVER changes the
    // out-of-bounds semantics (that stays the FREE region, matching the
    // real Flightmare truth audit).
    // ── early termination / synthetic observation ──────────────────
    // Preflight control rate (Hz) — dt = 1/control_rate_hz is used for
    // the stall-displacement threshold and duration conversions (no magic
    // 30.0 scattered in the code).
    double control_rate_hz = 30.0;
    bool walls_visible_in_observation = true;
    // Early termination (blueprint-only, never changes expert labels):
    //  * no-progress: original-goal distance shrinks by less than
    //    no_progress_min_progress_m over no_progress_window_ticks;
    //  * stall: |velocity| < stall_speed_mps for stall_window_ticks while
    //    not in a legitimate TURN.
    // DEFAULT OFF: a pure "goal distance must shrink by X over the window"
    // rule wrongly kills legitimate long detours that need lateral movement
    // or a temporary distance increase.  Only when enabled (>0) does the
    // controller use the COMBINED criterion (goal shrink AND window motion
    // below thresholds); the stall + global tick budget remain the primary
    // guards.  See blueprint_generation_controller.cpp.
    int    no_progress_window_ticks = 0;        // 0 = disabled
    double no_progress_min_progress_m = 1.0;
    // Combined-criterion motion floor (m): when no-progress is enabled the
    // drone must ALSO have travelled less than this over the window, so a
    // long detour that is actually moving is never killed.
    double no_progress_window_min_motion_m = 2.0;
    int    stall_window_ticks = 90;             // 3 s @ 30 Hz
    double stall_speed_mps = 0.02;
    // Minimum required sign alternations of a chicane realisation along
    // its recorded orientation (4 obstacles => at least 2 flips).
    int min_chicane_alternations = 2;
    // Per-round sanity log to stderr (one line per round).
    bool log_rounds = true;

    // ── privileged task qualification (port of the 2D causal
    //    qualification; see route_qualifier.*) ──────────────────────
    // Runs BEFORE the full HierarchicalExpert preflight: endpoint safety,
    // global connectivity, straight-corridor blocker analysis, and (only
    // for blocked tasks) LEFT / RIGHT side-constrained A* around the
    // primary blocker.  The A* / side routes are PRIVILEGED truth — they
    // are used ONLY to decide task fairness, never fed to the expert.
    struct TaskQualificationConfig {
        bool enabled = true;
        // Blocked tasks must have BOTH homotopy branches globally feasible
        // (a local-causal expert cannot know a hidden one-sided dead end).
        bool require_both_sides_feasible = true;
        // Global A* connectivity confirmation (in addition to the cached
        // component lookup).  Cheap and bounded; only run when a scene has
        // many components / large obstacles.
        bool run_astar_confirmation = true;
        // Node-expansion caps (hard; a search that hits the cap is a
        // reject reason, never an unbounded search).
        int max_astar_expansions = 30000;        // global connectivity A*
        int max_side_route_expansions = 20000;   // per segment A*
        // Per-TASK shared cap across LEFT + RIGHT (each segment deducts
        // from this; a query whose remaining budget is 0 stops and reports
        // side_search_budget_exceeded).
        int max_total_side_route_expansions = 120000;
        // Blueprint-generation-level hard upper bound on ALL A* expansions
        // (global connectivity + side routes across every candidate).
        // Once reached, no further qualification searches run and the
        // generation ends with a clear reason.
        uint64_t max_total_qualification_expansions = 400000;
        // Per-scene cap prevents one difficult scene from consuming the
        // entire generation-wide qualification budget.
        uint64_t max_expansions_per_scene = 0;
        // Bounded radius (m) used by the start-clearance recovery (mirrors
        // the 2D macro_start_recovery_max_radius_m = 0.5).
        double start_recovery_max_radius_m = 0.5;
        // Side-A* lateral bias (0 = side-neutral).  Mirrors the 2D
        // macro_route_side_bias=0.4; the SAME fixed axis (start->goal) and
        // blocker centre define LEFT/RIGHT for the whole qualification.
        double side_bias = 0.4;
        // Homotopy check tolerance (m): the continuous route must pass the
        // blocker with this much lateral margin on the requested side.
        double homotopy_side_tolerance_m = 0.05;
        // Bounded radius used to project a tangent gateway into a STRICTLY
        // legal cell (mirrors 2D macro_gateway_projection_radius_m).
        double gateway_projection_radius_m = 0.8;
        // Route-stretch threshold: a blocked task whose shortest qualified
        // route is >= this many times the straight distance is LONG_DETOUR.
        double min_route_stretch_for_long_detour = 1.5;
        // Diagnostic counters gate (aggregate only).
        bool log_qualification_stats = true;
    };
    TaskQualificationConfig qualification;

    // ── result requirements ────────────────────────────────────────
    int min_scenes = 4;              // must be met by the SELECTED scenes
    int min_selected_scenes = 4;     // selected-scene diversity gate
    int min_tasks = 24;
    int min_tasks_per_scene = 4;
    int max_tasks_per_scene = 12;
    int min_macro_ticks_per_class = 24;    // pass/normal/turn_left/turn_right
    int min_depth_samples_per_band = 40;   // near/mid/far/free
    int min_yaw_samples_per_bin = 3;       // per absolute-yaw bin
    int min_path_samples_per_class = 4;    // short/medium/long
    // Initial yaw is directly sampled, so it can be tightly balanced.
    double max_yaw_imbalance_ratio = 0.20;
    // TURN behaviour depends on real geometry; slightly looser.
    double max_turn_imbalance_ratio = 0.30;
    // Grouped coverage (avoid the "only near-direct samples" trap):
    //  * deflection: strong-right-group / right / near-direct / left /
    //    strong-left-group each need at least this many 30 Hz ticks;
    //  * correction: right / near-forward / left groups each need at
    //    least this many 5 Hz NORMAL_CORRECTION ticks.
    int min_grouped_deflection_samples = 8;
    int min_grouped_correction_samples = 4;

    // ── legacy strata thresholds (manifest compatibility only) ─────
    // Used to map realized scenes onto the legacy 3x3 density x radius
    // strata coverage report.  The density thresholds are OCCUPANCY
    // RATIOS (Σπ r² / free area), not obstacle counts — the same count of
    // big cylinders is far denser than small ones.
    double density_sparse_max = 0.06;
    double density_dense_min = 0.12;
    double radius_small_max_m = 0.6;
    double radius_large_min_m = 1.4;

    // ── base seed ──────────────────────────────────────────────────
    uint64_t base_seed = 260812;
};

// ═══════════════════════════════════════════════════════════════════
//  Privileged task qualification results (port of the 2D causal
//  qualification).  ALL fields are PRIVILEGED truth / diagnostics: they
//  may go to the Blueprint manifest and generation statistics but MUST
//  NEVER be fed to the 5 Hz / 30 Hz expert or the DatasetWriter student
//  inputs.
// ═══════════════════════════════════════════════════════════════════
struct SideRouteResult {
    bool checked = false;
    bool feasible = false;
    double path_length_m = 0.0;
    double min_clearance_m = 0.0;   // min route-clearance along the path
    uint32_t expanded_nodes = 0;    // A* node expansions for this side
    std::string reject_reason;      // "" when feasible
};

struct TaskQualificationSummary {
    bool endpoint_valid = false;        // both endpoints >= endpoint clearance
    bool connectivity_valid = false;    // same main traversable component
    bool straight_corridor_clear = false;  // direct route-clear corridor
    // Primary straight-corridor blocker (0 when clear).  PRIVILEGED truth.
    int primary_blocker_id = -1;
    double primary_blocker_x = 0.0, primary_blocker_y = 0.0;
    double primary_blocker_radius = 0.0;
    std::vector<int> blocking_obstacle_ids;
    // LEFT / RIGHT detour routes (only filled when the corridor is blocked
    // and qualification runs).
    SideRouteResult left;
    SideRouteResult right;
    // min(left.length, right.length) / straight_distance (0 when clear or
    // no feasible side).
    double privileged_min_route_stretch = 0.0;
    // Narrow-passage traversal evidence: set when at least one FEASIBLE
    // qualified side route actually traverses a cached narrow passage
    // (route crosses the passage corridor from one side to the other).
    int narrow_passage_id = -1;
    bool route_traverses_narrow = false;
    // Realized geometric class (CLEAR / LOCAL_AVOIDANCE / ... ) decided
    // from the qualification geometry, NOT from the scene profile alone.
    std::string realized_geom_type = "CLEAR";
    // "clear" / "blocked_both_feasible" / "blocked_single_side" /
    // "blocked_no_side" / "endpoint_invalid" / "different_component".
    std::string qualification_class = "clear";
    std::string reject_reason = "";   // "" when accepted
    bool accepted = false;            // passes the qualification gate
};

/// Aggregate per-round qualification counters (only aggregate, never
/// per-task logs).
struct QualificationCounters {
    uint64_t candidates_checked = 0;
    uint64_t endpoint_pass = 0;
    uint64_t connectivity_pass = 0;
    uint64_t straight_clear = 0;
    uint64_t blocked = 0;
    uint64_t side_qualification_attempt = 0;
    uint64_t both_sides_feasible = 0;
    uint64_t accepted = 0;                  // passes the qualification gate
    uint64_t reject_endpoint = 0;
    uint64_t reject_clearance = 0;
    uint64_t reject_different_component = 0;
    uint64_t reject_global_route = 0;
    uint64_t reject_global_astar_budget = 0;
    uint64_t reject_left_infeasible = 0;
    uint64_t reject_right_infeasible = 0;
    uint64_t reject_both_sides_required = 0;
    uint64_t reject_side_search_budget = 0;
    uint64_t reject_geom_mismatch = 0;
    uint64_t total_astar_expansions = 0;
};

// ═══════════════════════════════════════════════════════════════════
//  BudgetExhaustion — which budget stopped the generation (diagnostic)
// ═══════════════════════════════════════════════════════════════════
enum class BudgetExhaustion : uint8_t {
    NONE = 0,
    SCENE_BUDGET = 1,
    PREFLIGHT_ATTEMPT_BUDGET = 2,
    PREFLIGHT_TICK_BUDGET = 3,
    GENERATION_ROUND_BUDGET = 4,
    TASK_CANDIDATE_BUDGET = 5,
    // Privileged qualification A* expansion budget (generation-wide).
    QUALIFICATION_EXPANSION_BUDGET = 6,
};

inline const char* budgetExhaustionName(BudgetExhaustion b) {
    switch (b) {
        case BudgetExhaustion::NONE: return "none";
        case BudgetExhaustion::SCENE_BUDGET: return "scene_budget";
        case BudgetExhaustion::PREFLIGHT_ATTEMPT_BUDGET:
            return "preflight_attempt_budget";
        case BudgetExhaustion::PREFLIGHT_TICK_BUDGET:
            return "preflight_tick_budget";
        case BudgetExhaustion::GENERATION_ROUND_BUDGET:
            return "generation_round_budget";
        case BudgetExhaustion::TASK_CANDIDATE_BUDGET:
            return "task_candidate_budget";
        case BudgetExhaustion::QUALIFICATION_EXPANSION_BUDGET:
            return "qualification_expansion_budget";
    }
    return "none";
}

// ═══════════════════════════════════════════════════════════════════
//  GenerationTiming — lightweight performance statistics (real timers)
// ═══════════════════════════════════════════════════════════════════
struct GenerationTiming {
    double scene_generation_ms = 0.0;
    double scene_geometry_cache_ms = 0.0;
    double task_candidate_generation_ms = 0.0;
    double cheap_filter_ms = 0.0;
    double task_qualification_ms = 0.0;
    double preflight_total_ms = 0.0;
    double preflight_average_ms = 0.0;
    uint64_t preflight_count = 0;          // attempts (success + failure)
    uint64_t preflight_success_count = 0;
    uint64_t preflight_failure_count = 0;
    uint64_t preflight_ticks = 0;          // total ticks across attempts
    uint64_t cheap_filter_rejected = 0;
    double depth_proxy_total_ms = 0.0;     // timed inside preflightOne
    double selection_ms = 0.0;
    double consolidation_ms = 0.0;
    double total_ms = 0.0;

    std::map<std::string, double> asMap() const {
        return {{"scene_generation_ms", scene_generation_ms},
                {"scene_geometry_cache_ms", scene_geometry_cache_ms},
                {"task_candidate_generation_ms", task_candidate_generation_ms},
                {"cheap_filter_ms", cheap_filter_ms},
                {"task_qualification_ms", task_qualification_ms},
                {"preflight_total_ms", preflight_total_ms},
                {"preflight_average_ms", preflight_average_ms},
                {"depth_proxy_total_ms", depth_proxy_total_ms},
                {"selection_ms", selection_ms},
                {"consolidation_ms", consolidation_ms},
                {"total_ms", total_ms}};
    }
};

/// One round of generation (lightweight sanity log + manifest report).
struct RoundStats {
    uint64_t round = 0;
    uint64_t scenes_generated = 0;
    uint64_t scenes_valid = 0;
    uint64_t task_candidates = 0;     // sampled candidates this round
    uint64_t cheap_rejected = 0;
    uint64_t preflight_attempted = 0;
    uint64_t preflight_success = 0;
    uint64_t selected_pool = 0;       // pool size after this round
    double elapsed_ms = 0.0;
    double preflight_avg_ms = 0.0;
    /// Per-rejection-category preflight counts for this round:
    /// accepted / collision / timeout / no_progress / stall /
    /// out_of_bounds / macro_label / goal_not_reached.
    std::map<std::string, uint64_t> failure_breakdown;
    /// Aggregate privileged task-qualification counters for this round.
    QualificationCounters qualification;
    std::vector<std::string> remaining_deficits;
};

}  // namespace expert
}  // namespace il_dataset
