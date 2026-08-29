/// @file   pybind_module.cpp
/// @brief  pybind11 bindings for the hierarchical local expert
///         (_il_hierarchical_expert).
///
/// Exposes:
///   * Params2D              — the single authoritative parameter set
///                             (Python fills it from YAML);
///   * CoordinateAdapter     — the single coordinate adaptation layer;
///   * HierarchicalExpert    — the ONE state-owning expert instance;
///   * ExpertStepOutput      — flat per-tick output (CSV fields);
///   * PreflightSimulator    — dry-run closed-loop harness (same expert);
///   * SceneTaskBlueprintGenerator — full C++ scene/ESDF/task/blueprint
///                             generation + C++ classifier + quotas;
///   * TruthCylinderAudit    — exact-cylinder swept truth audit.

#include "il_dataset/hierarchical_expert/types.hpp"
#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"
#include "il_dataset/hierarchical_expert/hierarchical_expert.hpp"
#include "il_dataset/hierarchical_expert/preflight_simulator.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace il_dataset::expert;

namespace {

double* toArray3(const py::object& obj, double out[3]) {
    const auto list = py::cast<py::sequence>(obj);
    if (py::len(list) < 3) {
        throw std::invalid_argument("expected a 3-element sequence");
    }
    for (int i = 0; i < 3; ++i) {
        out[i] = py::cast<double>(list[i]);
    }
    return out;
}

double* toArray4(const py::object& obj, double out[4]) {
    const auto list = py::cast<py::sequence>(obj);
    if (py::len(list) < 4) {
        throw std::invalid_argument("expected a 4-element sequence");
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = py::cast<double>(list[i]);
    }
    return out;
}

/// Parse a numeric-list key of a dict (used by the blueprint config).
void parseDoubleList(const py::dict& d, const char* key,
                     std::vector<double>& out) {
    if (!d.contains(key)) return;
    const auto seq = py::cast<py::sequence>(d[key]);
    out.clear();
    out.reserve(static_cast<size_t>(py::len(seq)));
    for (py::handle h : seq) out.push_back(py::cast<double>(h));
}

void parseStringList(const py::dict& d, const char* key,
                     std::vector<std::string>& out) {
    if (!d.contains(key)) return;
    const auto seq = py::cast<py::sequence>(d[key]);
    out.clear();
    out.reserve(static_cast<size_t>(py::len(seq)));
    for (py::handle h : seq) out.push_back(py::cast<std::string>(h));
}

/// Parse the YAML `blueprint_generation` dict into the C++
/// BlueprintGenerationConfig (single source for the new pipeline).
void parseBlueprintConfig(const py::dict& bp, BlueprintGenerationConfig& b) {
    auto get_d = [&](const char* k, double dflt) {
        return bp.contains(k) ? py::cast<double>(bp[k]) : dflt;
    };
    auto get_i = [&](const char* k, int dflt) {
        return bp.contains(k) ? py::cast<int>(bp[k]) : dflt;
    };
    auto get_u = [&](const char* k, uint64_t dflt) {
        return bp.contains(k) ? py::cast<uint64_t>(bp[k]) : dflt;
    };
    auto get_b = [&](const char* k, bool dflt) {
        return bp.contains(k) ? py::cast<bool>(bp[k]) : dflt;
    };

    // ── warehouse (THE single coordinate source) ──────────────────
    if (bp.contains("warehouse")) {
        const py::dict wh = py::cast<py::dict>(bp["warehouse"]);
        if (wh.contains("free_region")) {
            const auto fr = py::cast<py::sequence>(wh["free_region"]);
            if (py::len(fr) >= 4) {
                b.warehouse.free_min_x = py::cast<double>(fr[0]);
                b.warehouse.free_max_x = py::cast<double>(fr[1]);
                b.warehouse.free_min_y = py::cast<double>(fr[2]);
                b.warehouse.free_max_y = py::cast<double>(fr[3]);
            }
        }
        if (wh.contains("wall_extension_m")) {
            b.warehouse.wall_extension_m = py::cast<double>(wh["wall_extension_m"]);
        }
    }

    // ── planner compatibility ─────────────────────────────────────
    b.vehicle_radius_m = get_d("vehicle_radius_m", b.vehicle_radius_m);
    b.navigation_clearance_m =
        get_d("navigation_clearance_m", b.navigation_clearance_m);
    b.clearance_discretization_margin_m = get_d(
        "clearance_discretization_margin_m", b.clearance_discretization_margin_m);
    b.generation_margin_m = get_d("generation_margin_m", b.generation_margin_m);
    b.min_surface_gap_m = get_d("min_surface_gap_m", b.min_surface_gap_m);
    b.boundary_margin_m = get_d("boundary_margin_m", b.boundary_margin_m);
    b.free_cell_surface_clearance_m =
        get_d("free_cell_surface_clearance_m", b.free_cell_surface_clearance_m);
    b.esdf_resolution_m = get_d("esdf_resolution_m", b.esdf_resolution_m);
    b.min_main_component_area_m2 =
        get_d("min_main_component_area_m2", b.min_main_component_area_m2);

    // ── profiles / sequence ───────────────────────────────────────
    // use_profile_catalog MUST reach C++: false disables the built-in
    // default catalog (only user-provided `profiles` remain).
    b.use_profile_catalog =
        get_b("use_profile_catalog", b.use_profile_catalog);
    if (bp.contains("profiles")) {
        std::vector<SceneProfile> profiles;
        for (py::handle h : py::cast<py::sequence>(bp["profiles"])) {
            const py::dict pd = py::cast<py::dict>(h);
            SceneProfile p;
            auto pg = [&](const char* k, auto dflt) {
                return pd.contains(k) ? py::cast<decltype(dflt)>(pd[k]) : dflt;
            };
            p.name = pg("name", std::string("profile"));
            p.count_min = pg("count_min", 0);
            p.count_max = pg("count_max", 0);
            p.radius_min = pg("radius_min", 0.1);
            p.radius_max = pg("radius_max", 1.0);
            p.radius_mode = pg("radius_mode", std::string("log_uniform"));
            p.fixed_radius = pg("fixed_radius", 0.1);
            const std::string st = pg("structure", std::string("uniform"));
            if (st == "clustered") p.structure = SceneStructure::CLUSTERED;
            else if (st == "corridor") p.structure = SceneStructure::CORRIDOR;
            else if (st == "bottleneck") p.structure = SceneStructure::BOTTLENECK;
            else if (st == "chicane") p.structure = SceneStructure::CHICANE;
            else if (st == "central_blocker") p.structure = SceneStructure::CENTRAL_BLOCKER;
            else if (st == "edge_clutter") p.structure = SceneStructure::EDGE_CLUTTER;
            else if (st == "empty") p.structure = SceneStructure::EMPTY;
            p.cluster_count = pg("cluster_count", 0);
            p.cluster_spread_m = pg("cluster_spread_m", 0.0);
            p.passage_width_m = pg("passage_width_m", 0.0);
            p.weight = pg("weight", 1.0);
            if (pd.contains("tags")) {
                for (py::handle th : py::cast<py::sequence>(pd["tags"])) {
                    p.tags.push_back(py::cast<std::string>(th));
                }
            }
            profiles.push_back(p);
        }
        b.profiles = std::move(profiles);
    }
    parseStringList(bp, "profile_sequence", b.profile_sequence);

    // ── tasks ─────────────────────────────────────────────────────
    b.min_task_distance_m = get_d("min_task_distance_m", b.min_task_distance_m);
    b.max_task_distance_m = get_d("max_task_distance_m", b.max_task_distance_m);
    b.max_route_stretch_ratio =
        get_d("max_route_stretch_ratio", b.max_route_stretch_ratio);
    b.max_route_stretch_slack_m =
        get_d("max_route_stretch_slack_m", b.max_route_stretch_slack_m);
    b.flight_height_m = get_d("flight_height_m", b.flight_height_m);
    b.flight_height_min_m = get_d("flight_height_min_m", b.flight_height_m);
    b.flight_height_max_m = get_d("flight_height_max_m", b.flight_height_m);
    b.obstacle_height_m = get_d("obstacle_height_m", b.obstacle_height_m);
    b.obstacle_height_min_m = get_d("obstacle_height_min_m", b.obstacle_height_m);
    b.obstacle_height_max_m = get_d("obstacle_height_max_m", b.obstacle_height_m);
    if (bp.contains("known_rects")) {
        b.known_rects.clear();
        for (py::handle h : py::cast<py::sequence>(bp["known_rects"])) {
            const py::dict rd = py::cast<py::dict>(h);
            KnownRect r;
            r.min_x = py::cast<double>(rd["min_x"]);
            r.max_x = py::cast<double>(rd["max_x"]);
            r.min_y = py::cast<double>(rd["min_y"]);
            r.max_y = py::cast<double>(rd["max_y"]);
            r.height_m = rd.contains("height_m")
                             ? py::cast<double>(rd["height_m"])
                             : b.obstacle_height_m;
            b.known_rects.push_back(r);
        }
    }
    b.task_sample_attempts = get_i("task_sample_attempts", b.task_sample_attempts);
    b.task_goal_attempts = get_i("task_goal_attempts", b.task_goal_attempts);

    // ── initial yaw (layered) ─────────────────────────────────────
    if (bp.contains("initial_yaw")) {
        const py::dict y = py::cast<py::dict>(bp["initial_yaw"]);
        parseDoubleList(y, "edges_deg", b.yaw_edges_deg);
        parseDoubleList(y, "weights", b.yaw_weights);
    }

    // ── depth proxy ───────────────────────────────────────────────
    if (bp.contains("depth_proxy")) {
        const py::dict d = py::cast<py::dict>(bp["depth_proxy"]);
        auto gd = [&](const char* k, double dflt) {
            return d.contains(k) ? py::cast<double>(d[k]) : dflt;
        };
        auto gi = [&](const char* k, int dflt) {
            return d.contains(k) ? py::cast<int>(d[k]) : dflt;
        };
        b.depth_proxy_num_rays = gi("num_rays", b.depth_proxy_num_rays);
        b.depth_proxy_sample_stride_ticks =
            gi("sample_stride_ticks", b.depth_proxy_sample_stride_ticks);
        b.depth_near_max_m = gd("near_max_m", b.depth_near_max_m);
        b.depth_mid_max_m = gd("mid_max_m", b.depth_mid_max_m);
        b.depth_far_max_m = gd("far_max_m", b.depth_far_max_m);
    }

    // ── behaviour histograms ──────────────────────────────────────
    if (bp.contains("histograms")) {
        const py::dict d = py::cast<py::dict>(bp["histograms"]);
        parseDoubleList(d, "correction_angle_edges_deg",
                        b.correction_angle_edges_deg);
        parseDoubleList(d, "correction_distance_edges",
                        b.correction_distance_edges);
        parseDoubleList(d, "deflection_edges_deg", b.deflection_edges_deg);
        parseDoubleList(d, "yaw_rate_edges", b.yaw_rate_edges);
        parseDoubleList(d, "speed_edges", b.speed_edges);
        if (d.contains("min_deflection_speed_mps")) {
            b.min_deflection_speed_mps =
                py::cast<double>(d["min_deflection_speed_mps"]);
        }
    }

    // ── path classes ──────────────────────────────────────────────
    if (bp.contains("path")) {
        const py::dict d = py::cast<py::dict>(bp["path"]);
        if (d.contains("short_max_m")) b.path_short_max_m = py::cast<double>(d["short_max_m"]);
        if (d.contains("long_min_m")) b.path_long_min_m = py::cast<double>(d["long_min_m"]);
    }

    // ── performance budgets ───────────────────────────────────────
    if (bp.contains("performance")) {
        const py::dict d = py::cast<py::dict>(bp["performance"]);
        auto gi = [&](const char* k, int dflt) {
            return d.contains(k) ? py::cast<int>(d[k]) : dflt;
        };
        b.max_scene_candidates = gi("max_scene_candidates", b.max_scene_candidates);
        b.max_task_candidates_per_scene =
            gi("max_task_candidates_per_scene", b.max_task_candidates_per_scene);
        b.max_generation_rounds = gi("max_generation_rounds", b.max_generation_rounds);
        b.max_total_preflight_tasks =
            gi("max_total_preflight_tasks", b.max_total_preflight_tasks);
        b.max_total_preflight_ticks = get_u(
            "max_total_preflight_ticks", b.max_total_preflight_ticks);
        b.max_preflight_ticks_per_task =
            gi("max_preflight_ticks_per_task", b.max_preflight_ticks_per_task);
        b.max_scene_generation_attempts =
            gi("max_scene_generation_attempts", b.max_scene_generation_attempts);
        b.max_task_generation_attempts =
            gi("max_task_generation_attempts", b.max_task_generation_attempts);
        // int worker count (legacy `false` bool casts to 0 = serial).
        if (d.contains("parallel_tasks")) b.parallel_tasks = py::cast<int>(d["parallel_tasks"]);
        if (d.contains("scene_switch_penalty")) b.scene_switch_penalty = py::cast<double>(d["scene_switch_penalty"]);
    }

    // ── scene-level parallel pipeline (new architecture) ─────────
    if (bp.contains("scene_parallel")) {
        const py::dict d = py::cast<py::dict>(bp["scene_parallel"]);
        auto gi = [&](const char* k, int dflt) {
            return d.contains(k) ? py::cast<int>(d[k]) : dflt;
        };
        auto gd = [&](const char* k, double dflt) {
            return d.contains(k) ? py::cast<double>(d[k]) : dflt;
        };
        auto gb = [&](const char* k, bool dflt) {
            return d.contains(k) ? py::cast<bool>(d[k]) : dflt;
        };
        auto gvl = [&](const char* k, const std::vector<double>& dflt) {
            std::vector<double> v = dflt;
            if (d.contains(k)) {
                v.clear();
                for (auto item : py::cast<py::list>(d[k])) {
                    v.push_back(py::cast<double>(item));
                }
            }
            return v;
        };
        b.scene_level_parallel = gb("enabled", false);
        b.scene_parallel_threads = gi("threads", b.scene_parallel_threads);
        b.scene_levels = gi("levels", b.scene_levels);
        b.scenes_per_level = gi("scenes_per_level", b.scenes_per_level);
        if (d.contains("scenes_per_level_list")) {
            b.scenes_per_level_list.clear();
            for (auto item : py::cast<py::list>(d["scenes_per_level_list"])) {
                b.scenes_per_level_list.push_back(py::cast<int>(item));
            }
        }
        b.level_radius_min_m = gvl("level_radius_min", b.level_radius_min_m);
        b.level_radius_max_m = gvl("level_radius_max", b.level_radius_max_m);
        b.level_occupancy_min = gvl("level_occupancy_min", b.level_occupancy_min);
        b.level_occupancy_max = gvl("level_occupancy_max", b.level_occupancy_max);
        b.distance_min_radius_scale =
            gd("distance_min_radius_scale", b.distance_min_radius_scale);
        b.obstacle_surface_gap_min_m =
            gd("surface_gap_min_m", b.obstacle_surface_gap_min_m);
        b.obstacle_boundary_min_m =
            gd("boundary_min_m", b.obstacle_boundary_min_m);
        b.task_distance_short_max_m =
            gd("distance_short_max_m", b.task_distance_short_max_m);
        b.task_distance_medium_max_m =
            gd("distance_medium_max_m", b.task_distance_medium_max_m);
        b.quick_preflight_max_ticks =
            gi("quick_preflight_max_ticks", b.quick_preflight_max_ticks);
        b.quick_preflight_dt_scale =
            gd("quick_preflight_dt_scale", b.quick_preflight_dt_scale);
        b.expected_collect_tasks =
            gi("expected_collect_tasks", b.expected_collect_tasks);
    }

    // ── result requirements ───────────────────────────────────────
    if (bp.contains("requirements")) {
        const py::dict d = py::cast<py::dict>(bp["requirements"]);
        auto gi = [&](const char* k, int dflt) {
            return d.contains(k) ? py::cast<int>(d[k]) : dflt;
        };
        auto gd = [&](const char* k, double dflt) {
            return d.contains(k) ? py::cast<double>(d[k]) : dflt;
        };
        b.min_scenes = gi("min_scenes", b.min_scenes);
        b.min_tasks = gi("min_tasks", b.min_tasks);
        b.min_tasks_per_scene = gi("min_tasks_per_scene", b.min_tasks_per_scene);
        b.max_tasks_per_scene = gi("max_tasks_per_scene", b.max_tasks_per_scene);
        b.min_macro_ticks_per_class =
            gi("min_macro_ticks_per_class", b.min_macro_ticks_per_class);
        b.min_depth_samples_per_band =
            gi("min_depth_samples_per_band", b.min_depth_samples_per_band);
        b.min_yaw_samples_per_bin =
            gi("min_yaw_samples_per_bin", b.min_yaw_samples_per_bin);
        b.min_path_samples_per_class =
            gi("min_path_samples_per_class", b.min_path_samples_per_class);
        b.max_turn_imbalance_ratio =
            gd("max_turn_imbalance_ratio", b.max_turn_imbalance_ratio);
        b.max_yaw_imbalance_ratio =
            gd("max_yaw_imbalance_ratio", b.max_yaw_imbalance_ratio);
        b.min_selected_scenes =
            gi("min_selected_scenes", b.min_selected_scenes);
        b.min_grouped_deflection_samples = gi(
            "min_grouped_deflection_samples",
            b.min_grouped_deflection_samples);
        b.min_grouped_correction_samples = gi(
            "min_grouped_correction_samples",
            b.min_grouped_correction_samples);
    }

    // ── synthetic observation / early termination ─────────────────
    if (bp.contains("synthetic_observation")) {
        const py::dict d = py::cast<py::dict>(bp["synthetic_observation"]);
        auto gb = [&](const char* k, bool dflt) {
            return d.contains(k) ? py::cast<bool>(d[k]) : dflt;
        };
        b.walls_visible_in_observation = gb(
            "walls_visible_in_observation", b.walls_visible_in_observation);
    }
    if (bp.contains("early_termination")) {
        const py::dict d = py::cast<py::dict>(bp["early_termination"]);
        auto gi2 = [&](const char* k, int dflt) {
            return d.contains(k) ? py::cast<int>(d[k]) : dflt;
        };
        auto gd2 = [&](const char* k, double dflt) {
            return d.contains(k) ? py::cast<double>(d[k]) : dflt;
        };
        auto gb = [&](const char* k, bool dflt) {
            return d.contains(k) ? py::cast<bool>(d[k]) : dflt;
        };
        b.no_progress_window_ticks =
            gi2("no_progress_window_ticks", b.no_progress_window_ticks);
        b.no_progress_min_progress_m = gd2(
            "no_progress_min_progress_m", b.no_progress_min_progress_m);
        b.stall_window_ticks =
            gi2("stall_window_ticks", b.stall_window_ticks);
        b.stall_speed_mps = gd2("stall_speed_mps", b.stall_speed_mps);
        b.log_rounds = gb("log_rounds", b.log_rounds);
        b.min_chicane_alternations =
            gi2("min_chicane_alternations", b.min_chicane_alternations);
    }
    // Preflight control rate (Hz) — dt = 1/control_rate_hz is the stall /
    // duration time base.  Defaults to 30.0 (the preflight tick grid).
    b.control_rate_hz = get_d("control_rate_hz", b.control_rate_hz);

    // Rare macro-turn probe.  By default it only biases sampling; strict
    // requested-side admission is opt-in for focused diagnostics.
    if (bp.contains("macro_probe")) {
        const py::dict d = py::cast<py::dict>(bp["macro_probe"]);
        if (d.contains("enabled")) {
            b.macro_probe_enabled = py::cast<bool>(d["enabled"]);
        }
        if (d.contains("yaw_error_deg")) {
            b.macro_probe_yaw_error_deg =
                py::cast<double>(d["yaw_error_deg"]);
        }
        if (d.contains("require_match")) {
            b.macro_probe_require_match =
                py::cast<bool>(d["require_match"]);
        }
    }

    if (bp.contains("exploration")) {
        const py::dict d = py::cast<py::dict>(bp["exploration"]);
        if (d.contains("pool_first")) {
            b.pool_first_exploration = py::cast<bool>(d["pool_first"]);
        }
        if (d.contains("rounds")) {
            b.exploration_rounds = py::cast<int>(d["rounds"]);
        }
        if (d.contains("min_pool_tasks")) {
            b.exploration_min_pool_tasks =
                py::cast<uint64_t>(d["min_pool_tasks"]);
        }
    }

    // ── privileged task qualification (2D causal-qualification port) ─
    if (bp.contains("task_qualification")) {
        const py::dict d = py::cast<py::dict>(bp["task_qualification"]);
        auto gb = [&](const char* k, bool dflt) {
            return d.contains(k) ? py::cast<bool>(d[k]) : dflt;
        };
        auto gi = [&](const char* k, int dflt) {
            return d.contains(k) ? py::cast<int>(d[k]) : dflt;
        };
        auto gu = [&](const char* k, uint64_t dflt) {
            return d.contains(k) ? py::cast<uint64_t>(d[k]) : dflt;
        };
        auto gd = [&](const char* k, double dflt) {
            return d.contains(k) ? py::cast<double>(d[k]) : dflt;
        };
        b.qualification.enabled =
            gb("enabled", b.qualification.enabled);
        b.qualification.require_both_sides_feasible =
            gb("require_both_sides_feasible",
               b.qualification.require_both_sides_feasible);
        b.qualification.run_astar_confirmation =
            gb("run_astar_confirmation",
               b.qualification.run_astar_confirmation);
        b.qualification.max_astar_expansions =
            gi("max_astar_expansions", b.qualification.max_astar_expansions);
        b.qualification.max_side_route_expansions = gi(
            "max_side_route_expansions",
            b.qualification.max_side_route_expansions);
        b.qualification.max_total_side_route_expansions = gu(
            "max_total_side_route_expansions",
            b.qualification.max_total_side_route_expansions);
        // Generation-wide cap on all privileged qualification A* work.
        b.qualification.max_total_qualification_expansions = gu(
            "max_total_qualification_expansions",
            b.qualification.max_total_qualification_expansions);
        b.qualification.max_expansions_per_scene = gu(
            "max_expansions_per_scene",
            b.qualification.max_expansions_per_scene);
        // Start-endpoint recovery disk radius (2D macro start recovery).
        b.qualification.start_recovery_max_radius_m = gd(
            "start_recovery_max_radius_m",
            b.qualification.start_recovery_max_radius_m);
        b.qualification.side_bias =
            gd("side_bias", b.qualification.side_bias);
        b.qualification.homotopy_side_tolerance_m = gd(
            "homotopy_side_tolerance_m",
            b.qualification.homotopy_side_tolerance_m);
        b.qualification.gateway_projection_radius_m = gd(
            "gateway_projection_radius_m",
            b.qualification.gateway_projection_radius_m);
        b.qualification.min_route_stretch_for_long_detour = gd(
            "min_route_stretch_for_long_detour",
            b.qualification.min_route_stretch_for_long_detour);
        b.qualification.log_qualification_stats = gb(
            "log_qualification_stats",
            b.qualification.log_qualification_stats);
    }
    // Route-clearance margin (m) on top of the endpoint clearance.
    b.route_clearance_margin_m =
        get_d("route_clearance_margin_m", b.route_clearance_margin_m);

    // ── explicit distribution targets (empty => buildDefaultTargets) ─
    if (bp.contains("distribution_targets")) {
        const auto seq = py::cast<py::sequence>(bp["distribution_targets"]);
        b.targets.clear();
        for (py::handle h : seq) {
            const py::dict td = py::cast<py::dict>(h);
            DistributionTarget t;
            t.key = py::cast<std::string>(td["key"]);
            t.metric = py::cast<std::string>(td["metric"]);
            if (td.contains("target")) t.target = py::cast<double>(td["target"]);
            if (td.contains("minimum")) t.minimum = py::cast<double>(td["minimum"]);
            if (td.contains("maximum")) t.maximum = py::cast<double>(td["maximum"]);
            if (td.contains("weight")) t.weight = py::cast<double>(td["weight"]);
            if (td.contains("tolerance")) t.tolerance = py::cast<double>(td["tolerance"]);
            if (td.contains("bin_index")) t.bin_index = py::cast<int>(td["bin_index"]);
            if (td.contains("minimum_hard")) t.minimum_hard = py::cast<bool>(td["minimum_hard"]);
            b.targets.push_back(t);
        }
    }

    // ── legacy strata thresholds / seed ───────────────────────────
    if (bp.contains("legacy")) {
        const py::dict d = py::cast<py::dict>(bp["legacy"]);
        if (d.contains("density_sparse_max")) b.density_sparse_max = py::cast<double>(d["density_sparse_max"]);
        if (d.contains("density_dense_min")) b.density_dense_min = py::cast<double>(d["density_dense_min"]);
        if (d.contains("radius_small_max_m")) b.radius_small_max_m = py::cast<double>(d["radius_small_max_m"]);
        if (d.contains("radius_large_min_m")) b.radius_large_min_m = py::cast<double>(d["radius_large_min_m"]);
    }
    b.base_seed = get_u("base_seed", b.base_seed);
}

}  // namespace

PYBIND11_MODULE(_il_hierarchical_expert, m) {
    m.doc() = "Hierarchical local expert (5 Hz corrector + 30 Hz planner)";

    // Build-identity marker.  Bump this string whenever the C++ expert
    // behaviour changes so a stale .so is instantly detectable: read it at
    // runtime (il_manager logs it and writes it into metadata.json).
    // Current source state:
    //   R2 signed per-side blocker clearance (Fix A)
    //   R3 30 Hz planner spin guard (70-tick release + cooldown)
    //   R4 spin-settle gate + TURN_STEP_PENDING exit_ang hold
    //   R5 minimal-turn best_clear (min |body bearing| among clearing)
    //   R6 CameraRig2D::rayWorldDirXY left-right MIRROR FIX
    //      (positive bearing now = LEFT; +sin -> -sin).  The mirror made
    //      every right-side obstacle appear on the LEFT, so the corrector
    //      always turned left into the mirrored image (the +28 deg target /
    //      in-place spin in joint_v2 episodes).  Verified offline: with the
    //      fix the corrector picks a ~straight target until the real
    //      blocker enters range, then a small ~10 deg left nudge.
    //   R7 selectSide MINIMAL-TURN tie-break: when free-range evidence is
    //      ambiguous, pick the side whose blocker-clearing bypass needs the
    //      smaller turn (LEFT ~0.5 deg vs RIGHT ~24.5 deg in task369).
    //      Previously the ambiguous case defaulted RIGHT -> big right
    //      detour -> stall + spin (episode 97845b3f).
    //   R8 obs_body_ignore_radius_m: mask the drone's OWN body out of the
    //      depth (the render shows body/rotors up to ~1.5 m from the vehicle
    //      centre at the FOV edges).  Without this the own body became a
    //      false OCCUPIED blob at spawn, the corrector entered on it
    //      (corridor blocked) and issued TURN_LEFT forever (stuck in place,
    //      cvx=0; episode d3e14461).  REMOVED in R18: with the camera
    //      mounted 0.3 m FORWARD of the centre the 0.3 m centre-radius mask
    //      is geometrically dead (every valid depth pixel sits >= 0.31 m
    //      from the centre) and the body never renders in practice.
    //   R10 r20260819_cmd_ramp_feedforward: reachableCommand switched from
    //      state-pinning to COMMAND-RAMP (clamp relative to the previous
    //      command), so the backend VelocityYawRateController feedforward
    //      delivers the lp_max_accel=2 m/s^2 ramp and the closed loop
    //      reaches ~2 m/s^2 (P error ~0.3 m/s -> P = 2.4 < 4, tilt ~11.5 deg
    //      < 35 deg).  The historical ±15 deg limit cycle (af4159ce) was
    //      with kp=36 (P saturated on any error > 0.11 m/s); kp=8/kd=1.2
    //      keeps 5x the headroom.  lp_eff_accel_mps2 restored to 2.0 to
    //      match.
    //   R11 r20260819_nodding_fix: real episodes under R10 showed a ~3 Hz
    //      tilt limit cycle ("nodding", roll swings to 16-19 deg, a_cmd
    //      clipping at max_accel on ~10% of ticks).  Root causes (from CSV
    //      analysis): (1) the Python feedforward divided d(cmd) by the
    //      1/50 control dt while the command updates at 30 Hz -> a 1.67x
    //      FF over-pulse that, with the P term, clipped at max_accel;
    //      fixed in il_dynamics.py by computing the FF once per 30 Hz
    //      command interval (d(cmd)/duration) and holding it; (2) at a 5 Hz
    //      correction exit the effective target jumps (~4.8 m), triggering a
    //      planner memory reset that state-pinned the command down from its
    //      leading value -> P-error spike; fixed by seeding last_command_
    //      at the current state velocity on reset.
    //   R12 r20260819_vz_cmd_ramp: episodes under R11 showed large altitude
    //      fluctuation (z range ~1.4 m, min 0.59 below the 0.8 floor) during
    //      horizontal acceleration.  Root cause: VerticalController::
    //      executableVz state-pinned the vertical command to vz_world ±
    //      lp_max_v_accel*dt, so when tilt made the drone sink (~0.8 m/s) the
    //      command was forced to stay negative and could not command a climb
    //      until the disturbance ended.  Fixed like the horizontal channel:
    //      the executable vz now command-ramps relative to the PREVIOUS vz
    //      command (last_vz_command_, reset per task), so it leads the
    //      recovery; divergence > 2*lp_max_vz falls back to vz_world.
    //   R13 r20260819_roll_coord_turn: episodes under R12 succeeded but still
    //      showed a ~2.5 Hz roll oscillation (roll swings to ~19-20 deg)
    //      during yaw-while-moving.  Root cause: UNCOORDINATED turns — the
    //      planner commands yaw_rate + forward velocity but vy_cmd = 0, so
    //      the velocity vector does not rotate with the body; the body-frame
    //      side-slip vy grows at rate -w*v and the lateral velocity loop
    //      over-banks to fight it.  Fixed in il_dynamics.py by adding a
    //      coordinated-turn centripetal feedforward to the lateral accel:
    //      accel_y += v_fwd * yaw_rate (Python controller update(), using
    //      the measured body yaw rate).  Cancels the side-slip rate; loop
    //      only trims the residual.
    //   R14 r20260819_avoidance_labels_macro_v9: user reported local
    //      planning "does not avoid" and macro waypoints "stiff/fixed".
    //      A side-by-side diff vs il_2d_multiscale_debug showed: (1) the
    //      local planner's avoidance_active barely fired (only in the
    //      planned-trajectory branch on a >3° bearing deviation; false
    //      during turns/escapes/brakes and when the B-spline weaved with
    //      chosen_b==b_t) — now set in ALL live avoidance branches (turn,
    //      planned detour incl. plan cross-track > 0.35 m, escape rotation,
    //      blocked brake) plus an obstacle-proximity signal (observed
    //      OCCUPIED within 2 m); (2) the macro corrector had 5 deliberate
    //      anti-spin modifications that made it stiffer than v9 — removed
    //      to restore il_2d v9 behavior: selectSide minimal-turn tie-break,
    //      makeCorrectionDirective best_clear nose-nudge (back to max
    //      along-progress), spin-settle HOLD_SPINNING gate, TURN re-step at
    //      exit_ang 8° (back to fov_half 45°), NORMAL_CORRECTION
    //      world-latch + stall-refresh.
    //   R15 r20260819_align_while_planning: user architecture spec — local
    //      planner keeps ONE target (original or macro sub-target, world-
    //      latched and converted to body at 30 Hz); macro takeover has 4
    //      behaviours (TURN_LEFT / TURN_RIGHT / NORMAL_CORRECTION sub-
    //      target / PASS_THROUGH keep-original).  Gap fixed: when the
    //      target is inside the ±35° usable band but planning fails, the
    //      planner now KEEPS turning the nose toward the target and
    //      re-plans every 30 Hz tick; only once aligned (|b_t| <=
    //      lp_turn_exit_deg = 8°) and still unplannable does it hand over
    //      to the macro takeover (escape rotation / brake).  Removes the
    //      30-35° "dead zone" brake.
    //   R16 r20260819_yaw_cmd_ramp: user reported a "clumsy nose" — the
    //      drone braked to a stop, sat still ~1 s, then took ~1.5 s to
    //      build yaw rate (CSV: command -0.12 -> -0.93 rad/s over 1.5 s
    //      while turnYawRate wanted -1.5 immediately, target 65° off-nose,
    //      episode 000000_b99de7ca).  Root cause: reachableCommand state-
    //      pinned the yaw-rate COMMAND to the actual yaw rate ± dyr, so it
    //      tracked the slowly-building measured rate.  Fixed: the yaw rate
    //      now command-ramps relative to the PREVIOUS yaw-rate command at
    //      lp_max_yaw_accel = 4 rad/s^2 (~0.4 s to full rate), with a
    //      divergence guard (> 2*lp_max_yaw_rate falls back to the state
    //      yaw rate).
    //   R17 r20260819_brake_and_rotate (R16 REVERTED + turnYawRate scaling):
    //      the user clarified the real complaint was NOT yaw build speed —
    //      it was the nose FROZEN IN PLACE during macro -> PASS_THROUGH and
    //      planning transitions.  turnYawRate had a hard gate
    //      (speed > kTurnMaxSpeedMps=0.2 -> yaw_rate 0) that froze the nose
    //      for the full ~1 s brake while the target sat 55-65° off-nose
    //      (yaw_rate_command=0.000, yaw constant, speed braking 1.83->0.77,
    //      joint_v2 ep 000000_99ca0c06 t=4.43-4.97).  Now turnYawRate
    //      scales the yaw rate by 1/(1+3*spd): gentle rotation while
    //      braking (large radius, small sweep, drone decelerating at
    //      lp_max_accel), full rate near standstill.  R16 itself is
    //      REVERTED (reachableCommand yaw back to state-pinning): the
    //      command-ramp executed the fast-flipping goal bearing at the
    //      terminal approach and produced a sustained yaw oscillation
    //      (final yaw rate never < lp_turn_exit_max_yaw_rate ->
    //      terminal_yaw_rate episode failures, ep 000000_99ca0c06).
    //      Also added goal-proximity guards so the terminal phase SETTLES:
    //      the yaw-first gate, the align-while-planning turn, and the
    //      planned-terminal yaw intent all command 0 yaw once
    //      dist <= task_goal_tolerance (the bearing flips with sub-
    //      tolerance jitter; chasing it blocks goalReached()).
    //   R18: removed the geometrically-dead drone-body mask
    //      (obs_body_ignore_radius_m): with the camera mounted 0.3 m FORWARD
    //      of the vehicle centre every valid depth pixel sits >= 0.31 m from
    //      the centre, so the 0.3 m centre-radius mask could never fire, and
    //      the body never renders (zero masked pixels over full episodes).
    //   R19 r20260819_ego_bspline: EGO-style optimisation-based B-spline
    //      local path (structure copied from the open-source ZJU-FAST-Lab/
    //      ego-planner BsplineOptimizer + the okazaki L-BFGS solver — see
    //      ego_bspline.hpp/.cpp and lbfgs.hpp).  The cubic B-spline is BENT
    //      around observed obstacles (sample-based collision + a lateral
    //      detour guide, our substitute for EGO's A* guide path) instead of
    //      the old straight-ray core (collinear control points = zero
    //      curvature = boxed-in deadlocks).  The optimised path is still
    //      validated with the hard clearance + dynamic envelope and falls
    //      back to the straight-line planner / escape-rotate / brake
    //      (planEgoOrStraight in local_planner_30hz), so existing success
    //      cases never regress.  Tuned via Params2D.ego_*.
    //   R20 r20260820_smooth_turn_brake: vector-magnitude XY slew limiting,
    //      previous-command yaw slew limiting, brake-before-rotate staging,
    //      strict zero-translation TURN execution, terminal speed/yaw caps,
    //      and canonical zero-distance GOAL_REACHED labels.
    //   R21 r20260822_persistent_recovery: separate rotation availability
    //      from translational feasibility, keep macro recovery persistent
    //      until translation re-entry is confirmed, and preview-certify
    //      every buffered fly-through correction with the 30 Hz local planner
    //      while its student distance label remains clipped to 4.5/5.
    //   R22 r20260822_receding_safe_recovery: keep temporary guides as
    //      receding non-terminal fly-through targets, preserve search yaw
    //      across brake staging, restore a physical stopping envelope, and
    //      make blueprint truth-brake acceptance match runtime collection.
    //   R23 r20260822_fixed_waypoint_handoff: gate macro takeover on actual
    //      persistent 30 Hz failure plus topology evidence, hold finite
    //      world-frame waypoints instead of extending a receding ray, keep
    //      live feasible execution across cold previews, and lock the near
    //      visible original goal for terminal capture.
    //   R24 r20260822_brake_latch_terminal_stop: persistent terminal_stop
    //      brake semantics (never re-derived from live distance), latched
    //      brake-before-search point, exit correction when the goal is
    //      outside the FOV with a clear corridor, goal-directed search
    //      rotation, and local-planner corridor evidence for takeover.
    //   R25 r20260822_goal_capture_recovery: goal_capture_lock can no
    //      longer revoke an ACTIVE correction (measured 50 s deadlock in
    //      joint_v2_000004_4ab1e354), stale-history FREE clearing with
    //      confirmation, decayed failure counting (no single-frame
    //      TERMINAL_SETTLING reset), real local_limit_cycle_detected,
    //      separate waypoint-reached tolerance, held-waypoint persistent
    //      failure gate, and structured corridor-block diagnostics.
    //   R26 r20260822_terminal_capture_microapproach: fix the 0.4-0.8 m
    //      terminal dead zone (task 65: 0.48 m for 36 s, 2458° of yaw) via
    //      a hard-clear terminal micro-approach; split the 1 m RISK
    //      corridor from the HARD (handoffClearance) corridor so a cell
    //      ~0.9 m beyond the goal can no longer trigger macro takeover;
    //      terminal-capture lock (<=1 m: no locked-side search TURN, only
    //      a real hard block releases it); macro-level limit-cycle
    //      watchdog on the ORIGINAL goal; Python judge-only quality
    //      watchdogs (no-progress / near-goal timeout / cumulative yaw).
    // R27 (2026-08-22, user: "为什么重新规划导致轨迹偏移"): the local
    //      planner becomes a PLAN/TRACK hierarchy — the B-spline is
    //      re-optimised every lp_replan_interval_ticks ticks (10 Hz) and
    //      PURSUED between replans (longer lp_pursuit_lookahead_m arc
    //      lookahead), and the EGO optimiser is temporally anchored
    //      (warm-start + lambda_ref cost toward the previous plan) so the
    //      executed path follows ONE committed spline instead of the
    //      envelope of per-tick heads (kills receding-horizon head drift).
    // R28 (2026-08-22, task 440 analysis): local-vs-upper consistency fixes.
    //      (1) non-terminal A* now routes to a LOCAL horizon (4.5 m) instead
    //      of the far target (out-of-grid -> EMPTY -> FOV scan -> straight
    //      chord blind to a blocker just beyond the scan endpoint ->
    //      spurious BLOCKED + macro takeover); (2) when the corridor is
    //      blocked the receding-horizon validation is extended out to the
    //      first block (min_validate_m / ego validate_front_m) so a straight
    //      chord through the block can never "validate" — the scan continues
    //      to a genuine detour; (3) a stale HISTORY cell parked next to the
    //      drone and OUTSIDE the FOV (cannot be re-confirmed, blocks every
    //      bearing for its full 120-tick lifetime) triggers a bounded
    //      in-place re-orientation toward the most-open edge instead of a
    //      ~4 s NO_SAFE_CANDIDATE stall; (4) A* budget exhaustion now picks
    //      the min-heuristic frontier (not min g+h) and rejects degenerate
    //      no-progress partial paths.
    // R28b (2026-08-23, task 440 R28 batch regression): the R28
    // planFovTrajectory clearance-gate edit used `if (!in_front_clr)
    // continue;` BEFORE the point push + `reached` flag — every non-terminal
    // straight-fallback plan failed validation (reached never set), the
    // local planner collapsed to NO_SAFE_CANDIDATE, and the upper layer
    // took over falsely.  Fixed by wrapping the gate (matching EGO
    // buildAndValidate).  Also REVERTED the R28 "local-horizon A*" routing
    // (it drifted the drone east around the obstacle chain into a dead-end
    // at the next blocker); the original A*-to-target + scan fallback is
    // restored.  min_validate_m / stale-history re-orientation / A* min-h
    // budget fix are kept.
    // R28c (2026-08-23, task 33 analysis — user architecture guidance):
    //      (1) the LOCAL planner must NOT produce big lateral detours — the
    //      UPPER planner owns bypasses.  Selected non-terminal plans whose
    //      endpoint deviates > lp_max_local_deviation_deg (30) from the
    //      target direction are rejected -> NO_SAFE_CANDIDATE -> macro
    //      places a bypass waypoint (kills the ~40°-off-target side plans
    //      that spiralled the drone east, task 33);
    //      (2) multi-cruise retry now picks the SMALLEST lateral bend across
    //      cruise levels (planFovTrajectory + EGO planImpl) — a slow
    //      straight thread through a gap beats a fast wide detour (the
    //      "search the neighbourhood for a satisfying point" request);
    //      (3) lp_max_yaw_accel 4 -> 8 rad/s^2 so the drone can reverse a
    //      full-rate turn in ~0.3 s and promptly follow a freshly
    //      re-planned trajectory (was ~0.6 s, "too clumsy").
    // R28d (2026-08-23, task 194 analysis — user question: "does the local
    // yaw aim at the target or just pick the current velocity?"):
    //      EGO dep_dir now ALWAYS aims at the endpoint/route — never the
    //      current velocity.  Old code used the current velocity for moving
    //      terminal stops and (first) for A*-guided plans; the B-spline
    //      then departed along the nose (20-50° off the goal), the 0.6 m
    //      pursuit lookahead saw only that nose-aligned start -> yaw_cmd
    //      ~0.05 -> the drone crabbed the whole flight, and the terminal
    //      approach (goal CLEAR, plan endpoint == goal) still departed west,
    //      lost the plan at 4.2 m, stopped dead west of the goal and
    //      re-oriented (~1.2 s stall).  Terminal now departs toward the
    //      goal (front-3 m FOV gate passes within the 45° camera FOV; a
    //      blocked straight departure falls to the bearing scan); A*-guided
    //      plans depart along the route's initial corridor segment.
    // R28e (2026-08-23, task 401 analysis — user question: "planner flips to
    // the other side of an obstacle when close / after the nose turns; is it
    // a cost weight? the local plan seems to prefer the FOV edges"):
    //      NOT a cost-weight issue.  When the direct corridor to the goal is
    //      blocked, R28's min_validate_m = first_blocking_distance (~3.1 m)
    //      EXTENDED the clearance validation of EVERY bearing-scan candidate
    //      (not just the direct chord).  As the drone moved past the blocker
    //      (obs5, task 401), its observed cells toggled each side plan
    //      valid/invalid -> the plan flipped between the west (toward goal)
    //      and south (east of blocker, misses goal) sides every ~0.6 s ->
    //      zigzag + brief east drift.  FIX: only the DIRECT bearing (k==0)
    //      keeps the corridor-block extended validation (so the straight
    //      chord still fails — task 440 stays fixed); side/detour bearings
    //      validate to the base 3 m front only (receding-horizon contract:
    //      only the executed front must be clear).  Side plans near the
    //      target direction now stay valid consistently -> the drone commits
    //      to one side and completes the detour.
    // R28f (2026-08-23, task 401 analysis — user question: "does the planner
    // use the local map? why can it plan far but struggles to advance near?"):
    //      (1) YES — the planner uses the ObservedGrid2D (5m, 0.1m, 4s-expiry)
    //      built from depth for corridor assessment, clearance validation and
    //      A*.  Far: grid clear near the path -> straight plan validates.
    //      Near obs7 (-2.57,12.90): its cells fill the grid -> direct corridor
    //      BLOCKED -> the local constraint set (+-35° scan band + 30°
    //      deviation + rotate-only-outside-FOV) finds no plan -> NO_SAFE_
    //      CANDIDATE, and the macro's bypass waypoints were themselves
    //      unreachable -> the drone stalled ~14 s (joint_v2_000010_2f0cc210).
    //      (2) FIX: the "keep-aligning" rotation (section 2.5) used the CAMERA
    //      FOV (45°) as its threshold, so a target 35-45° off the nose was
    //      inside the FOV (no pure-rotate at section 1) yet outside the +-35°
    //      scan band (no plan) -> dead-zone oscillation around 45°.  The
    //      rotation now fires whenever |b_t| > planning_fov_half (35°) and no
    //      plan was found, so the drone aligns to the opening and hands a
    //      properly-oriented NO_SAFE_CANDIDATE to the macro.
    // R28g (2026-08-23, task 401 debug — user: "near the obstacle, no matter
    // how the map fills in, planning must not fail"):
    //      The user was RIGHT: the far plan (t=4-5) already routed EAST
    //      around obs7 (-2.57,12.90) at ~287° with 0.8 m truth clearance and
    //      advanced toward the goal.  At t=5.27 the nose had drifted 284° ->
    //      287° (goal bearing error 27° -> 31°), so the same east detour's
    //      deviation from the goal crossed the R28c 30° cliff (30.0° was
    //      accepted, 30.6° rejected) -> NO_SAFE_CANDIDATE -> ~14 s stall
    //      (joint_v2_000010_2f0cc210).  FIX: lp_max_local_deviation_deg
    //      30 -> 35 (= the +-35° scan band), in lock-step across types.hpp /
    //      pybind / il_expert_config.py / both YAMLs.  The local may use its
    //      full scan band for a legitimate minimal detour; task-33-style
    //      40-48° spirals are still rejected for the macro.  Combines with
    //      R28f (rotate to |b_t|<=35° before handing NO_SAFE_CANDIDATE).
    // R28h (2026-08-23, 9-trajectory batch review — 7 success / 2 fail):
    // cross-checked the other-AI findings on the r28g data (task 440/33/
    // 236/491/452/194/475/493/65) and implemented the P0 items:
    //  P0#1/#2 (local_planner_30hz.cpp): straight-through priority + local
    //    horizon cap.  14.9% of valid plans had endpoints >5 m (max 8.24 m,
    //    from the global-history-grid A* returning route.back()); clear-area
    //    plans bent 0.31-0.53 m.  Now: (a) the straight plan to a target-
    //    direction endpoint at local range is tried FIRST (planFovTrajectory,
    //    no EGO lateral freedom); (b) the A* endpoint AND EGO guide are
    //    capped to the ~3 m receding horizon (route goal stays the full
    //    target — this is NOT the R28b-reverted local-horizon-A*).
    //  P0#3: 1° fine re-scan between b0 and the first valid coarse bearing
    //    -> minimal-necessary detour instead of a 5°-step jump.
    //  P0#4 (hierarchical_expert_fsm.cpp): SAFE_HOLD no longer counts as a
    //    failure (task 236 had 6 corrections with ZERO NO_SAFE_CANDIDATE);
    //    a stationary body is no longer topology evidence — takeover needs
    //    sustained real NO_SAFE/BLOCKED + observed corridor block.
    //  P0#5 (macro_expert_5hz.cpp): removed the "stop at current position"
    //    NORMAL_CORRECTION label (distance_norm=0, terminal_stop) before a
    //    search turn — the local brakes-then-rotates for pure-rotation
    //    targets itself, so the TURN is published directly (clean label).
    // R28i (2026-08-23, task 401 joint_v2_000010_5d9e5002):
    //  side-commitment + target-side preference so the local planner stops
    //  flapping right<->left around a blocker and picks the GOAL side.
    //  Root cause: obs7 (-2.57,12.90) sat ON the goal line (goal 12° right
    //  of the nose, b_t=-12); the WEST (goal-side) corridor was in obs7's
    //  occlusion shadow -> UNKNOWN (+3*res/cell), the EAST corridor was
    //  observed FREE, so the global-grid A* rationally routed EAST (away
    //  from the goal) and the 30 Hz scan, whose fixed +1 (left)-first
    //  order tried the wrong side first, flipped between 235° (right,
    //  correct) and 287° (left, wrong).
    //  Fixes:
    //   (a) scan expansion order now tries the TARGET's side first
    //       ((b_t>=0)?+1:-1) instead of always +1 (left) — the plan bends
    //       toward the goal, not away (local_planner_30hz.cpp).
    //   (b) last_plan_side_ commitment: once a side is chosen (set from
    //       wrapAngle(chosen_b-b_t) in the commit block), the scan keeps
    //       expanding that side first; it only flips when that side has NO
    //       valid bearing.  Reset per task (reset()).
    //   (c) routeAStar goal-line lateral penalty (+0.25/cell per metre off
    //       the direct start->goal line): the A* hugs the minimal-detour
    //       (goal) side instead of wandering onto a far observed corridor,
    //       so the A* and the scan agree on the side (this was the actual
    //       side-decider in task 401 — it routed EAST at 3.83-4.57 s with
    //       only a one-tick WEST scan blip at 4.27 s).
    // R28j (2026-08-23, r28i batch review — 9/9 success but structural):
    //  FOV-edge saturation + three inconsistent endpoint semantics + A*
    //  "pretend-plan to the far world goal through UNKNOWN" + upper
    //  takeover lag.  Data (3670 valid plans): 179 endpoints >30° off the
    //  nose, 225 >25° off the target, many pinned at 34-35° (the 35°
    //  deviation guard equals the 35° scan band so it never constrained
    //  the scanner); the A* 3 m-horizon branch produces 74% of all plans.
    //  Fixes:
    //   (a) local_planner_30hz.cpp — A* ROUTE GOAL is now the LOCAL
    //       OBSERVABLE HORIZON (target direction clamped to ~4.95 m), not
    //       the full world target threaded through ~11 m of UNKNOWN cells.
    //       Far UNKNOWN regions no longer distort the current topology or
    //       the bypass-side choice (R28i's task-401 east choice was exactly
    //       that).  The full target stays the attract direction; only the
    //       search is local.  The executed endpoint/guide stay at the ~3 m
    //       receding horizon (localRouteHorizon()).
    //   (b) local_planner_30hz.cpp — the legacy "first-valid ±35° scan +
    //       fine re-scan" is replaced by a SCORED local-frontier scan:
    //       every bearing in the band is evaluated at 2° and the best is
    //       scored by J = |b−b_t| + 0.5·lateral + 0.3·curvature +
    //       0.35·side_flip − 0.25·clearance.  Direct bearing keeps the
    //       full-range endpoint + corridor-block extended validation
    //       (R28e); side bearings use the 3 m endpoint == validated front.
    //       The scanner now optimises minimal detour / progress /
    //       clearance / side commitment instead of grabbing an arbitrary
    //       edge bearing.
    //   (c) local_planner_30hz.cpp — routing/control layering made
    //       explicit: routing heading = the direction to the 3 m waypoint
    //       (chosen_b / deviation guard); actual yaw control = the 0.6 m
    //       pure-pursuit tangent (unchanged).  goal_direction_flu_* still
    //       encodes the effective (upper/original) target, never the 3 m
    //       waypoint.
    //   (d) hierarchical_expert_fsm.cpp — takeover now requires a RECENT
    //       failure (last_failure_tick_ within one 6-tick macro window) in
    //       addition to the sustained count and the current corridor-block
    //       evidence.  A decaying counter from an earlier emergency brake
    //       no longer authorises takeover once the local has recovered
    //       (task 475 r28i: TURN_RIGHT at frame 60 with NONE/corridor
    //       clear).
    //  NOT changed: the straight-through priority keeps its full-range
    //  endpoint — the corridor-block extended validation needs the endpoint
    //  at/ beyond the first blocking cell to reject a blocked chord (a 3 m
    //  endpoint would validate a clear front-3 m even with a blocker at
    //  3.5 m).  The 3 m waypoint principle applies to the DETOUR branches.
    // R29 (2026-08-23): stale-history cells older than the local planning
    // age no longer hard-veto trajectory validation; local frontier scans
    // search a preferred small-deviation band before expanding to the full
    // FOV band; stored-plan tracking reports its real target distance; and
    // macro takeover requires a longer confirmed failure window.
    // R30 (2026-08-27): (1) the ray-sector rays shrink to the target
    // distance once the target is inside the 4.5 m planning range — the
    // centre ray only validates the path TO the goal, so an obstacle
    // BEHIND the goal no longer forces an end-zone detour; (2) the
    // avoidance ray must stay within FOV−10° of the target bearing — the
    // target and the avoidance direction can no longer wedge at opposite
    // FOV edges.
    //
    // r20260829_d435i_fov72.95_noise: the camera is now D435i-aligned —
    // Unity vertical FOV 58° (depth.fov), expert HORIZONTAL FOV 72.95°
    // (obs_fov_deg), near=0.28 m (D435i Min-Z), far=10 m, and the depth
    // frame gets multiplicative Gaussian noise (sigma = 0.02*depth) before
    // the expert step.  All labels are generated against the D435i FOV.
    //
    // r20260829_d435i_fullh88.8_noise_5m: FULL-HORIZONTAL capture —
    // depth is 848x480 (D435i nominal 87°x58° @848x480), expert
    // HORIZONTAL FOV 88.80° (obs_fov_deg = 2*atan(tan(29°)*848/480)).
    // The camera captures [0.28, 10] m but only 0-5 m is used (expert
    // range_m=5; PNG encoded with max_m=5 clips the rest to the far
    // marker).  Noise injection unchanged (sigma = 0.02*depth).
    //
    // r20260829_d435i_fullh89.2_640x360_noise_5m: switched capture to
    // 640x360 (16:9) — expert HORIZONTAL FOV 89.16° (obs_fov_deg =
    // 2*atan(tan(29°)*640/360)); everything else unchanged (vertical
    // FOV 58°, near 0.28, far 10, use 0-5 m only, noise 0.02*depth).
    m.attr("EXPERT_REVISION") =
        std::string("r20260829_d435i_fullh89.2_640x360_noise_5m_TGS");

    // ── Params2D: the single authoritative parameter source ─────────
    py::class_<Params2D>(m, "Params2D")
        .def(py::init<>())
        // region
        .def_readwrite("region_min_x", &Params2D::region_min_x)
        .def_readwrite("region_max_x", &Params2D::region_max_x)
        .def_readwrite("region_min_y", &Params2D::region_min_y)
        .def_readwrite("region_max_y", &Params2D::region_max_y)
        .def_readwrite("drone_radius", &Params2D::drone_radius)
        .def_readwrite("scene_safety_clearance",
                       &Params2D::scene_safety_clearance)
        .def_readwrite("macro_route_clearance_margin",
                       &Params2D::macro_route_clearance_margin)
        .def_readwrite("task_goal_tolerance", &Params2D::task_goal_tolerance)
        .def_readwrite("task_episode_timeout_s",
                       &Params2D::task_episode_timeout_s)
        // observation
        .def_readwrite("obs_fov_deg", &Params2D::obs_fov_deg)
        .def_readwrite("obs_range_m", &Params2D::obs_range_m)
        .def_readwrite("obs_resolution", &Params2D::obs_resolution)
        .def_readwrite("obs_ray_angular_res_deg",
                       &Params2D::obs_ray_angular_res_deg)
        .def_readwrite("obs_history_max_age_ticks",
                       &Params2D::obs_history_max_age_ticks)
        .def_readwrite("obs_free_clear_confirmations",
                       &Params2D::obs_free_clear_confirmations)
        .def_readwrite("lp_planning_history_max_age_ticks",
                       &Params2D::lp_planning_history_max_age_ticks)
        .def_readwrite("obs_ground_clearance_m",
                       &Params2D::obs_ground_clearance_m)
        // local planner
        .def_readwrite("lp_horizon_s", &Params2D::lp_horizon_s)
        .def_readwrite("lp_dt", &Params2D::lp_dt)
        .def_readwrite("ego_enabled", &Params2D::ego_enabled)
        .def_readwrite("ego_lambda_smooth", &Params2D::ego_lambda_smooth)
        .def_readwrite("ego_lambda_collision", &Params2D::ego_lambda_collision)
        .def_readwrite("ego_lambda_feasibility",
                       &Params2D::ego_lambda_feasibility)
        .def_readwrite("ego_lambda_fitness", &Params2D::ego_lambda_fitness)
        .def_readwrite("ego_lambda_fov", &Params2D::ego_lambda_fov)
        .def_readwrite("ego_clearance_m", &Params2D::ego_clearance_m)
        .def_readwrite("ego_ts", &Params2D::ego_ts)
        .def_readwrite("ego_n_segments", &Params2D::ego_n_segments)
        .def_readwrite("ego_max_iter", &Params2D::ego_max_iter)
        .def_readwrite("ego_lambda_ref", &Params2D::ego_lambda_ref)
        .def_readwrite("lp_max_speed", &Params2D::lp_max_speed)
        .def_readwrite("lp_cruise_speed_mps", &Params2D::lp_cruise_speed_mps)
        .def_readwrite("lp_goal_decay_range_m",
                       &Params2D::lp_goal_decay_range_m)
        .def_readwrite("lp_vmin_speed_mps", &Params2D::lp_vmin_speed_mps)
        .def_readwrite("lp_yaw_decay_per_deg",
                       &Params2D::lp_yaw_decay_per_deg)
        .def_readwrite("lp_yaw_decay_min", &Params2D::lp_yaw_decay_min)
        .def_readwrite("lp_ray_target_rel_max_deg",
                       &Params2D::lp_ray_target_rel_max_deg)
        .def_readwrite("lp_terminal_micro_approach_m",
                       &Params2D::lp_terminal_micro_approach_m)
        .def_readwrite("lp_replan_interval_ticks",
                       &Params2D::lp_replan_interval_ticks)
        .def_readwrite("lp_pursuit_lookahead_m",
                       &Params2D::lp_pursuit_lookahead_m)
        .def_readwrite("lp_track_max_cross_track_m",
                       &Params2D::lp_track_max_cross_track_m)
        .def_readwrite("lp_track_min_front_m",
                       &Params2D::lp_track_min_front_m)
        .def_readwrite("lp_max_accel", &Params2D::lp_max_accel)
        .def_readwrite("lp_eff_accel_mps2", &Params2D::lp_eff_accel_mps2)
        .def_readwrite("lp_max_yaw_rate", &Params2D::lp_max_yaw_rate)
        .def_readwrite("lp_max_yaw_accel", &Params2D::lp_max_yaw_accel)
        .def_readwrite("lp_max_vz", &Params2D::lp_max_vz)
        .def_readwrite("lp_max_v_accel", &Params2D::lp_max_v_accel)
        .def_readwrite("lp_vz_kp", &Params2D::lp_vz_kp)
        .def_readwrite("lp_z_min_m", &Params2D::lp_z_min_m)
        .def_readwrite("lp_z_max_m", &Params2D::lp_z_max_m)
        .def_readwrite("lp_vertical_clearance_m",
                       &Params2D::lp_vertical_clearance_m)
        .def_readwrite("lp_min_clearance", &Params2D::lp_min_clearance)
        .def_readwrite("lp_soft_clearance_radius_m",
                       &Params2D::lp_soft_clearance_radius_m)
        .def_readwrite("lp_clearance_discretization_margin_m",
                       &Params2D::lp_clearance_discretization_margin_m)
        .def_readwrite("lp_obstacle_reaction_time_s",
                       &Params2D::lp_obstacle_reaction_time_s)
        .def_readwrite("lp_control_period_s", &Params2D::lp_control_period_s)
        .def_readwrite("lp_turn_enter_deg", &Params2D::lp_turn_enter_deg)
        .def_readwrite("lp_turn_exit_deg", &Params2D::lp_turn_exit_deg)
        .def_readwrite("lp_turn_exit_max_yaw_rate",
                       &Params2D::lp_turn_exit_max_yaw_rate)
        .def_readwrite("lp_turn_k", &Params2D::lp_turn_k)
        .def_readwrite("lp_yaw_smooth_alpha", &Params2D::lp_yaw_smooth_alpha)
        .def_readwrite("lp_max_local_deviation_deg",
                       &Params2D::lp_max_local_deviation_deg)
        .def_readwrite("lp_preferred_local_deviation_deg",
                       &Params2D::lp_preferred_local_deviation_deg)
        .def_readwrite("lp_near_goal_heading_relax_distance",
                       &Params2D::lp_near_goal_heading_relax_distance)
        .def_readwrite("lp_near_goal_turn_enter_deg",
                       &Params2D::lp_near_goal_turn_enter_deg)
        .def_readwrite("lp_terminal_speed_gain",
                       &Params2D::lp_terminal_speed_gain)
        .def_readwrite("lp_terminal_max_speed", &Params2D::lp_terminal_max_speed)
        .def_readwrite("lp_terminal_max_yaw_rate",
                       &Params2D::lp_terminal_max_yaw_rate)
        .def_readwrite("lp_min_progress_speed_mps",
                       &Params2D::lp_min_progress_speed_mps)
        .def_readwrite("lp_target_discontinuity_reset_m",
                       &Params2D::lp_target_discontinuity_reset_m)
        .def_readwrite("lp_nominal_clearance_m",
                       &Params2D::lp_nominal_clearance_m)
        .def_readwrite("lp_risk_corridor_half_width",
                       &Params2D::lp_risk_corridor_half_width)
        .def_readwrite("lp_brake_stop_margin_m",
                       &Params2D::lp_brake_stop_margin_m)
        // macro / corrector
        .def_readwrite("macro_observable_frontier_min_distance_m",
                       &Params2D::macro_observable_frontier_min_distance_m)
        .def_readwrite("macro_observable_frontier_min_progress_m",
                       &Params2D::macro_observable_frontier_min_progress_m)
        .def_readwrite("macro_observable_frontier_max_retreat_m",
                       &Params2D::macro_observable_frontier_max_retreat_m)
        .def_readwrite("macro_goal_direction_min_range_m",
                       &Params2D::macro_goal_direction_min_range_m)
        .def_readwrite("macro_waypoint_update_along_margin",
                       &Params2D::macro_waypoint_update_along_margin)
        .def_readwrite("macro_search_rotation_cooldown_5hz",
                       &Params2D::macro_search_rotation_cooldown_5hz)
        .def_readwrite("macro_observable_unknown_margin_cells",
                       &Params2D::macro_observable_unknown_margin_cells)
        .def_readwrite("macro_side_evidence_margin",
                       &Params2D::macro_side_evidence_margin)
        .def_readwrite("macro_evidence_ray_step_deg",
                       &Params2D::macro_evidence_ray_step_deg)
        .def_readwrite("macro_min_evidence_ray_pairs",
                       &Params2D::macro_min_evidence_ray_pairs)
        .def_readwrite("macro_corridor_half_width",
                       &Params2D::macro_corridor_half_width)
        .def_readwrite("macro_blocking_lateral_span_ratio",
                       &Params2D::macro_blocking_lateral_span_ratio)
        .def_readwrite("macro_corridor_rear_tolerance_m",
                       &Params2D::macro_corridor_rear_tolerance_m)
        .def_readwrite("macro_local_recovery_prefix_m",
                       &Params2D::macro_local_recovery_prefix_m)
        .def_readwrite("macro_local_candidate_bearing_step_deg",
                       &Params2D::macro_local_candidate_bearing_step_deg)
        .def_readwrite("macro_local_candidate_distance_step_m",
                       &Params2D::macro_local_candidate_distance_step_m)
        .def_readwrite("macro_guide_horizon_m",
                       &Params2D::macro_guide_horizon_m)
        .def_readwrite("macro_local_target_event_tolerance_m",
                       &Params2D::macro_local_target_event_tolerance_m)
        .def_readwrite("macro_takeover_confirm_ticks_30hz",
                       &Params2D::macro_takeover_confirm_ticks_30hz)
        .def_readwrite("macro_unknown_recovery_threshold_ticks",
                       &Params2D::macro_unknown_recovery_threshold_ticks)
        .def_readwrite("macro_brake_confirm_ticks_5hz",
                       &Params2D::macro_brake_confirm_ticks_5hz)
        .def_readwrite("macro_waypoint_reached_tolerance_m",
                       &Params2D::macro_waypoint_reached_tolerance_m)
        .def_readwrite("macro_terminal_capture_radius_m",
                       &Params2D::macro_terminal_capture_radius_m)
        .def_readwrite("macro_limit_cycle_goal_progress_m",
                       &Params2D::macro_limit_cycle_goal_progress_m)
        .def_readwrite("macro_limit_cycle_window_5hz",
                       &Params2D::macro_limit_cycle_window_5hz)
        // target encoding
        .def_readwrite("te_direction_bin_count",
                       &Params2D::te_direction_bin_count)
        .def_readwrite("te_normal_distance_reserve_m",
                       &Params2D::te_normal_distance_reserve_m)
        .def_readwrite("te_turn_ray_margin_deg",
                       &Params2D::te_turn_ray_margin_deg)
        // depth camera extrinsic (Unity T_BC)
        .def_readwrite("cam_t_bc_x", &Params2D::cam_t_bc_x)
        .def_readwrite("cam_t_bc_y", &Params2D::cam_t_bc_y)
        .def_readwrite("cam_t_bc_z", &Params2D::cam_t_bc_z)
        .def_readwrite("cam_r_bc", &Params2D::cam_r_bc)
        // vehicle
        .def_readwrite("vehicle_goal_stop_speed_mps",
                       &Params2D::vehicle_goal_stop_speed_mps)
        .def_readwrite("vehicle_stationary_speed_mps",
                       &Params2D::vehicle_stationary_speed_mps);

    // ── CoordinateAdapter (the single coordinate adaptation layer) ──
    py::class_<CoordinateAdapter>(m, "CoordinateAdapter")
        .def_static("flightmare_yaw_to_expert",
                    &CoordinateAdapter::flightmareYawToExpert,
                    py::arg("yaw_fm"))
        .def_static("expert_yaw_to_flightmare",
                    &CoordinateAdapter::expertYawToFlightmare,
                    py::arg("yaw_expert"))
        .def_static(
            "bearing_to_world_point",
            [](double px, double py, double tx, double ty, double yaw_fm) {
                const Vec2d world_point(tx, ty);
                const Vec2d position(px, py);
                return CoordinateAdapter::bearingToWorldPoint(
                    world_point, position, yaw_fm);
            },
            py::arg("px"), py::arg("py"), py::arg("tx"), py::arg("ty"),
            py::arg("yaw_fm"));

    // ── ExpertStepOutput (flat, read-only from Python) ──────────────
    py::class_<ExpertStepOutput>(m, "ExpertStepOutput")
        .def(py::init<>())
        .def_readonly("tick", &ExpertStepOutput::tick)
        .def_readonly("macro_update_mask", &ExpertStepOutput::macro_update_mask)
        .def_readonly("fsm_state", &ExpertStepOutput::fsm_state)
        .def_readonly("fsm_prev_state", &ExpertStepOutput::fsm_prev_state)
        .def_readonly("terminal", &ExpertStepOutput::terminal)
        .def_readonly("goal_direction_flu_x",
                      &ExpertStepOutput::goal_direction_flu_x)
        .def_readonly("goal_direction_flu_y",
                      &ExpertStepOutput::goal_direction_flu_y)
        .def_readonly("goal_direction_flu_z",
                      &ExpertStepOutput::goal_direction_flu_z)
        .def_readonly("goal_distance_clipped_m",
                      &ExpertStepOutput::goal_distance_clipped_m)
        .def_readonly("goal_distance_norm", &ExpertStepOutput::goal_distance_norm)
        .def_readonly("goal_distance_raw_m",
                      &ExpertStepOutput::goal_distance_raw_m)
        .def_readonly("effective_target_source",
                      &ExpertStepOutput::effective_target_source)
        .def_readonly("target_correction_active",
                      &ExpertStepOutput::target_correction_active)
        .def_readonly("directive_terminal_stop",
                      &ExpertStepOutput::directive_terminal_stop)
        .def_readonly("effective_direction_token",
                      &ExpertStepOutput::effective_direction_token)
        .def_readonly("effective_target_world_x",
                      &ExpertStepOutput::effective_target_world_x)
        .def_readonly("effective_target_world_y",
                      &ExpertStepOutput::effective_target_world_y)
        .def_readonly("effective_target_world_z",
                      &ExpertStepOutput::effective_target_world_z)
        .def_readonly("effective_target_world_valid",
                      &ExpertStepOutput::effective_target_world_valid)
        .def_readonly("target_velocity_flu_x",
                      &ExpertStepOutput::target_velocity_flu_x)
        .def_readonly("target_velocity_flu_y",
                      &ExpertStepOutput::target_velocity_flu_y)
        .def_readonly("target_velocity_flu_z",
                      &ExpertStepOutput::target_velocity_flu_z)
        .def_readonly("target_yaw_rate", &ExpertStepOutput::target_yaw_rate)
        .def_readonly("intent_vx_body", &ExpertStepOutput::intent_vx_body)
        .def_readonly("intent_vy_body", &ExpertStepOutput::intent_vy_body)
        .def_readonly("intent_yaw_rate", &ExpertStepOutput::intent_yaw_rate)
        .def_readonly("intent_vz_body", &ExpertStepOutput::intent_vz_body)
        .def_readonly("hierarchical_mode", &ExpertStepOutput::hierarchical_mode)
        .def_readonly("planner_status", &ExpertStepOutput::planner_status)
        .def_readonly("failure_reason", &ExpertStepOutput::failure_reason)
        .def_readonly("selected_output_speed_mps",
                      &ExpertStepOutput::selected_output_speed_mps)
        .def_readonly("local_target_distance_m",
                      &ExpertStepOutput::local_target_distance_m)
        .def_readonly("min_observed_clearance_m",
                      &ExpertStepOutput::min_observed_clearance_m)
        .def_readonly("obstacle_risk_cost", &ExpertStepOutput::obstacle_risk_cost)
        .def_readonly("avoidance_active", &ExpertStepOutput::avoidance_active)
        .def_readonly("local_corridor_blocked",
                      &ExpertStepOutput::local_corridor_blocked)
        .def_readonly("risk_corridor_near_obstacle",
                      &ExpertStepOutput::risk_corridor_near_obstacle)
        .def_readonly("corridor_block_reason",
                      &ExpertStepOutput::corridor_block_reason)
        .def_readonly("corridor_block_source",
                      &ExpertStepOutput::corridor_block_source)
        .def_readonly("first_blocking_distance_m",
                      &ExpertStepOutput::first_blocking_distance_m)
        .def_readonly("first_block_x", &ExpertStepOutput::first_block_x)
        .def_readonly("first_block_y", &ExpertStepOutput::first_block_y)
        .def_readonly("first_block_age_ticks",
                      &ExpertStepOutput::first_block_age_ticks)
        .def_readonly("emergency_brake", &ExpertStepOutput::emergency_brake)
        .def_readonly("immediate_avoidance",
                      &ExpertStepOutput::immediate_avoidance)
        .def_readonly("local_limit_cycle_detected",
                      &ExpertStepOutput::local_limit_cycle_detected)
        .def_readonly("target_bearing_error_deg",
                      &ExpertStepOutput::target_bearing_error_deg)
        .def_readonly("consecutive_failures_30hz",
                      &ExpertStepOutput::consecutive_failures_30hz)
        .def_readonly("unknown_recovery_ticks",
                      &ExpertStepOutput::unknown_recovery_ticks)
        .def_readonly("plan_valid", &ExpertStepOutput::plan_valid)
        .def_readonly("plan_terminal", &ExpertStepOutput::plan_terminal)
        .def_readonly("plan_end_speed_mps",
                      &ExpertStepOutput::plan_end_speed_mps)
        .def_readonly("plan_executed_speed_mps",
                      &ExpertStepOutput::plan_executed_speed_mps)
        .def_readonly("plan_points_x", &ExpertStepOutput::plan_points_x)
        .def_readonly("plan_points_y", &ExpertStepOutput::plan_points_y)
        .def_readonly("macro_label_valid", &ExpertStepOutput::macro_label_valid)
        .def_readonly("macro_correction_type",
                      &ExpertStepOutput::macro_correction_type)
        .def_readonly("macro_direction_token",
                      &ExpertStepOutput::macro_direction_token)
        .def_readonly("macro_direction_flu_x",
                      &ExpertStepOutput::macro_direction_flu_x)
        .def_readonly("macro_direction_flu_y",
                      &ExpertStepOutput::macro_direction_flu_y)
        .def_readonly("macro_direction_flu_z",
                      &ExpertStepOutput::macro_direction_flu_z)
        .def_readonly("macro_distance_norm",
                      &ExpertStepOutput::macro_distance_norm)
        .def_readonly("macro_param_valid", &ExpertStepOutput::macro_param_valid)
        .def_readonly("navigation_goal_direction_flu_x",
                      &ExpertStepOutput::navigation_goal_direction_flu_x)
        .def_readonly("navigation_goal_direction_flu_y",
                      &ExpertStepOutput::navigation_goal_direction_flu_y)
        .def_readonly("navigation_goal_direction_flu_z",
                      &ExpertStepOutput::navigation_goal_direction_flu_z)
        .def_readonly("navigation_goal_distance_clipped_m",
                      &ExpertStepOutput::navigation_goal_distance_clipped_m)
        .def_readonly("navigation_goal_distance_norm",
                      &ExpertStepOutput::navigation_goal_distance_norm)
        .def_readonly("navigation_goal_distance_raw_m",
                      &ExpertStepOutput::navigation_goal_distance_raw_m)
        .def_readonly("original_navigation_goal_world_x",
                      &ExpertStepOutput::original_navigation_goal_world_x)
        .def_readonly("original_navigation_goal_world_y",
                      &ExpertStepOutput::original_navigation_goal_world_y)
        .def_readonly("original_navigation_goal_world_z",
                      &ExpertStepOutput::original_navigation_goal_world_z)
        .def_readonly("correction_enter_event",
                      &ExpertStepOutput::correction_enter_event)
        .def_readonly("correction_exit_event",
                      &ExpertStepOutput::correction_exit_event)
        .def_readonly("correction_update_event",
                      &ExpertStepOutput::correction_update_event)
        .def_readonly("observability_reason",
                      &ExpertStepOutput::observability_reason)
        .def_readonly("observability_goal_inside_fov",
                      &ExpertStepOutput::observability_goal_inside_fov)
        .def_readonly("observability_direct_corridor_blocked",
                      &ExpertStepOutput::observability_direct_corridor_blocked)
        .def_readonly("observability_left_bypass_visible",
                      &ExpertStepOutput::observability_left_bypass_visible)
        .def_readonly("observability_right_bypass_visible",
                      &ExpertStepOutput::observability_right_bypass_visible)
        .def_readonly("observability_local_avoidance_observable",
                      &ExpertStepOutput::observability_local_avoidance_observable)
        .def_readonly("directive_update_event",
                      &ExpertStepOutput::directive_update_event)
        .def_readonly("mission_revision", &ExpertStepOutput::mission_revision)
        .def_readonly("reentry_guard_ticks",
                      &ExpertStepOutput::reentry_guard_ticks)
        .def_readonly("obstacle_first_observed_event",
                      &ExpertStepOutput::obstacle_first_observed_event)
        .def_readonly("macro_tick_ran", &ExpertStepOutput::macro_tick_ran);

    // ── HierarchicalExpert ─────────────────────────────────────────
    py::class_<HierarchicalExpert>(m, "HierarchicalExpert")
        .def(py::init<>())
        .def("configure",
             [](HierarchicalExpert& self, const Params2D& p,
                const std::vector<double>& min_bounds,
                const std::vector<double>& max_bounds) {
                 if (min_bounds.size() < 2 || max_bounds.size() < 2) {
                     throw std::invalid_argument(
                         "min_bounds/max_bounds need 2 elements");
                 }
                 // configure(params, min_bounds, max_bounds) — the params
                 // are required (they drive the FSM / observation builder).
                 self.configure(p, Vec2d(min_bounds[0], min_bounds[1]),
                                Vec2d(max_bounds[0], max_bounds[1]));
             },
             py::arg("params"), py::arg("min_bounds"), py::arg("max_bounds"))
        .def("reset_task",
             [](HierarchicalExpert& self, const std::vector<double>& start,
                const std::vector<double>& goal, double initial_yaw_fm,
                uint64_t tick, double flight_z) {
                 self.resetTask(Vec2d(start[0], start[1]),
                                Vec2d(goal[0], goal[1]), initial_yaw_fm, tick,
                                flight_z);
             },
             py::arg("start"), py::arg("goal"), py::arg("initial_yaw_fm"),
             py::arg("tick"), py::arg("flight_z"))
        .def("accept_new_goal",
             [](HierarchicalExpert& self, const std::vector<double>& goal,
                uint64_t tick) {
                 self.acceptNewGoal(Vec2d(goal[0], goal[1]), tick);
             },
             py::arg("goal"), py::arg("tick"))
        .def("set_external_directive",
             [](HierarchicalExpert& self, int type, double corrected_x,
                double corrected_y, double turn_dir_x, double turn_dir_y,
                double normalized_distance, const std::string& reason) {
                 self.setExternalDirective(type, corrected_x, corrected_y,
                                           turn_dir_x, turn_dir_y,
                                           normalized_distance, reason);
             },
             py::arg("type"), py::arg("corrected_x"), py::arg("corrected_y"),
             py::arg("turn_dir_x"), py::arg("turn_dir_y"),
             py::arg("normalized_distance"), py::arg("reason"))
        .def("clear_external_directive",
             [](HierarchicalExpert& self) {
                 self.clearExternalDirective();
             })
        .def("external_directive_active",
             [](const HierarchicalExpert& self) -> bool {
                 return self.externalDirectiveActive();
             })
        .def("step",
             [](HierarchicalExpert& self, const py::object& pos, double yaw_fm,
                const py::object& vel_world, double yaw_rate_fm,
                const py::object& depth_m, int depth_w, int depth_h,
                const py::object& cam_pos, const py::object& cam_q,
                double flight_z, uint64_t tick, bool collision) {
                 double pos_arr[3], vel_arr[3], cam_pos_arr[3], cam_q_arr[4];
                 toArray3(pos, pos_arr);
                 toArray3(vel_world, vel_arr);
                 toArray3(cam_pos, cam_pos_arr);
                 toArray4(cam_q, cam_q_arr);
                 const auto depth_seq = py::cast<py::sequence>(depth_m);
                 std::vector<float> depth_vec;
                 depth_vec.reserve(static_cast<size_t>(py::len(depth_seq)));
                 for (py::handle item : depth_seq) {
                     depth_vec.push_back(static_cast<float>(py::cast<double>(item)));
                 }
                 return self.step(pos_arr, yaw_fm, vel_arr, yaw_rate_fm,
                                  depth_vec, depth_w, depth_h, cam_pos_arr,
                                  cam_q_arr, flight_z, tick, collision);
             },
             py::arg("pos"), py::arg("yaw_fm"), py::arg("vel_world"),
             py::arg("yaw_rate_fm"), py::arg("depth_m"), py::arg("depth_w"),
             py::arg("depth_h"), py::arg("cam_pos"), py::arg("cam_q"),
             py::arg("flight_z"), py::arg("tick"), py::arg("collision"))
        .def("last_directive_type",
             [](const HierarchicalExpert& self) -> std::string {
                 return targetCorrectionTypeName(self.lastDirective().type);
             })
        .def("last_directive_token",
             [](const HierarchicalExpert& self) -> int {
                 return self.lastDirective().direction_token;
             })
        .def("last_observability_reason",
             [](const HierarchicalExpert& self) -> std::string {
                 return self.lastObservability().reason;
             })
        .def("params", &HierarchicalExpert::params,
             py::return_value_policy::reference_internal)
        .def("task_goal",
             [](const HierarchicalExpert& self) -> std::vector<double> {
                 const Vec2d& g = self.task().goal;
                 return {g.x(), g.y(), self.flightZ()};
             });

    // ── PreflightSimulator (dry-run; same expert, truth-synthesized) ──
    py::class_<PreflightSimulator>(m, "PreflightSimulator")
        .def(py::init<const Params2D&>(), py::arg("params"))
        .def("configure",
             [](PreflightSimulator& self, const std::vector<double>& min_bounds,
                const std::vector<double>& max_bounds,
                const std::vector<std::vector<double>>& obstacles,
                const std::vector<std::vector<double>>& known_rects) {
                 Scene2D scene;
                 scene.min_bounds = Vec2d(min_bounds[0], min_bounds[1]);
                 scene.max_bounds = Vec2d(max_bounds[0], max_bounds[1]);
                 scene.valid = true;
                 int id = 0;
                 for (const auto& o : obstacles) {
                     if (o.size() < 3) continue;
                     Obstacle2D ob;
                     ob.center = Vec2d(o[0], o[1]);
                     ob.radius = o[2];
                     ob.id = id++;
                     scene.obstacles.push_back(ob);
                 }
                 std::vector<Vec2d> rmin, rmax;
                 for (const auto& r : known_rects) {
                     if (r.size() < 4) continue;
                     rmin.emplace_back(r[0], r[1]);
                     rmax.emplace_back(r[2], r[3]);
                 }
                 self.configure(scene, scene.min_bounds, scene.max_bounds,
                                nullptr, nullptr, rmin, rmax);
             },
             py::arg("min_bounds"), py::arg("max_bounds"),
             py::arg("obstacles"),
             py::arg("known_rects") = std::vector<std::vector<double>>{})
        .def("reset_task",
             [](PreflightSimulator& self, const std::vector<double>& start,
                const std::vector<double>& goal, double initial_yaw_fm,
                uint64_t tick, double flight_z) {
                 self.resetTask(Vec2d(start[0], start[1]),
                                Vec2d(goal[0], goal[1]), initial_yaw_fm, tick,
                                flight_z);
             },
             py::arg("start"), py::arg("goal"), py::arg("initial_yaw_fm"),
             py::arg("tick"), py::arg("flight_z"))
        .def("step",
             [](PreflightSimulator& self, uint64_t tick, bool collision_override) {
                 const auto res = self.step(tick, collision_override);
                 std::vector<double> state_vec{
                     res.state.position.x(), res.state.position.y(),
                     res.state.yaw, res.state.velocity_world.x(),
                     res.state.velocity_world.y(), res.state.yaw_rate};
                 return py::make_tuple(res.output, state_vec,
                                       res.truth_collision, res.goal_reached,
                                       res.out_of_bounds);
             },
             py::arg("tick"), py::arg("collision_override") = false);

    // ── Scene / ESDF / task / blueprint generator + classifier ─────
    py::class_<BlueprintObstacle>(m, "BlueprintObstacle")
        .def(py::init<>())
        .def_readwrite("x", &BlueprintObstacle::x)
        .def_readwrite("y", &BlueprintObstacle::y)
        .def_readwrite("radius", &BlueprintObstacle::radius)
        .def_readwrite("height_m", &BlueprintObstacle::height_m)
        .def_readwrite("id", &BlueprintObstacle::id);

    py::class_<BlueprintScene>(m, "BlueprintScene")
        .def(py::init<>())
        .def_readwrite("scene_id", &BlueprintScene::scene_id)
        .def_readwrite("seed", &BlueprintScene::seed)
        .def_readwrite("profile", &BlueprintScene::profile)
        // Exposed as a STRING ("horizontal"/"vertical"/"none") — the raw
        // scoped enum is internal; the manifest consumes the name.
        .def_property_readonly(
            "structure_orientation",
            [](const BlueprintScene& s) {
                return std::string(
                    structureOrientationName(s.structure_orientation));
            })
        .def_readwrite("metadata", &BlueprintScene::metadata)
        .def_readwrite("stratum_id", &BlueprintScene::stratum_id)
        .def_readwrite("count_stratum", &BlueprintScene::count_stratum)
        .def_readwrite("radius_stratum", &BlueprintScene::radius_stratum)
        .def_readwrite("is_empty", &BlueprintScene::is_empty)
        .def_readwrite("requested_obstacle_count",
                       &BlueprintScene::requested_obstacle_count)
        .def_readwrite("actual_obstacle_count",
                       &BlueprintScene::actual_obstacle_count)
        .def_readwrite("generation_valid", &BlueprintScene::generation_valid)
        .def_readwrite("failure_reason", &BlueprintScene::failure_reason)
        .def_readwrite("obstacles", &BlueprintScene::obstacles)
        .def_readwrite("planned_density_class",
                       &BlueprintScene::planned_density_class)
        .def_readwrite("planned_radius_class",
                       &BlueprintScene::planned_radius_class)
        .def_readwrite("density_class", &BlueprintScene::density_class)
        .def_readwrite("actual_density_class",
                       &BlueprintScene::actual_density_class)
        .def_readwrite("actual_radius_class",
                       &BlueprintScene::actual_radius_class)
        .def_readwrite("actual_min_radius_m",
                       &BlueprintScene::actual_min_radius_m)
        .def_readwrite("actual_max_radius_m",
                       &BlueprintScene::actual_max_radius_m);

    py::class_<BlueprintTaskAudit>(m, "BlueprintTaskAudit")
        .def(py::init<>())
        .def_readwrite("accepted", &BlueprintTaskAudit::accepted)
        .def_readwrite("reached_goal", &BlueprintTaskAudit::reached_goal)
        .def_readwrite("truth_collision", &BlueprintTaskAudit::truth_collision)
        .def_readwrite("truth_brake_triggered",
                       &BlueprintTaskAudit::truth_brake_triggered)
        .def_readwrite("out_of_bounds", &BlueprintTaskAudit::out_of_bounds)
        .def_readwrite("macro_label_ok", &BlueprintTaskAudit::macro_label_ok)
        .def_readwrite("qualification_exceeded",
                       &BlueprintTaskAudit::qualification_exceeded)
        .def_readwrite("preflight_ticks", &BlueprintTaskAudit::preflight_ticks)
        .def_readwrite("min_truth_clearance_m",
                       &BlueprintTaskAudit::min_truth_clearance_m)
        .def_readwrite("goal_distance_m", &BlueprintTaskAudit::goal_distance_m)
        .def_readwrite("preflight_status", &BlueprintTaskAudit::preflight_status)
        .def_readwrite("straight_distance_m",
                       &BlueprintTaskAudit::straight_distance_m)
        .def_readwrite("path_length_m", &BlueprintTaskAudit::path_length_m)
        .def_readwrite("path_stretch_ratio",
                       &BlueprintTaskAudit::path_stretch_ratio)
        .def_readwrite("preflight_duration_s",
                       &BlueprintTaskAudit::preflight_duration_s);

    py::class_<SideRouteResult>(m, "SideRouteResult")
        .def(py::init<>())
        .def_readwrite("checked", &SideRouteResult::checked)
        .def_readwrite("feasible", &SideRouteResult::feasible)
        .def_readwrite("path_length_m", &SideRouteResult::path_length_m)
        .def_readwrite("min_clearance_m", &SideRouteResult::min_clearance_m)
        .def_readwrite("expanded_nodes", &SideRouteResult::expanded_nodes)
        .def_readwrite("reject_reason", &SideRouteResult::reject_reason);

    py::class_<TaskQualificationSummary>(m, "TaskQualificationSummary")
        .def(py::init<>())
        .def_readwrite("endpoint_valid",
                       &TaskQualificationSummary::endpoint_valid)
        .def_readwrite("connectivity_valid",
                       &TaskQualificationSummary::connectivity_valid)
        .def_readwrite("straight_corridor_clear",
                       &TaskQualificationSummary::straight_corridor_clear)
        .def_readwrite("primary_blocker_id",
                       &TaskQualificationSummary::primary_blocker_id)
        .def_readwrite("primary_blocker_x",
                       &TaskQualificationSummary::primary_blocker_x)
        .def_readwrite("primary_blocker_y",
                       &TaskQualificationSummary::primary_blocker_y)
        .def_readwrite("primary_blocker_radius",
                       &TaskQualificationSummary::primary_blocker_radius)
        .def_readwrite("blocking_obstacle_ids",
                       &TaskQualificationSummary::blocking_obstacle_ids)
        .def_readwrite("left", &TaskQualificationSummary::left)
        .def_readwrite("right", &TaskQualificationSummary::right)
        .def_readwrite("narrow_passage_id",
                       &TaskQualificationSummary::narrow_passage_id)
        .def_readwrite("route_traverses_narrow",
                       &TaskQualificationSummary::route_traverses_narrow)
        .def_readwrite("privileged_min_route_stretch",
                       &TaskQualificationSummary::privileged_min_route_stretch)
        .def_readwrite("realized_geom_type",
                       &TaskQualificationSummary::realized_geom_type)
        .def_readwrite("qualification_class",
                       &TaskQualificationSummary::qualification_class)
        .def_readwrite("reject_reason", &TaskQualificationSummary::reject_reason)
        .def_readwrite("accepted", &TaskQualificationSummary::accepted);

    py::class_<QualificationCounters>(m, "QualificationCounters")
        .def(py::init<>())
        .def_readonly("candidates_checked", &QualificationCounters::candidates_checked)
        .def_readonly("endpoint_pass", &QualificationCounters::endpoint_pass)
        .def_readonly("connectivity_pass", &QualificationCounters::connectivity_pass)
        .def_readonly("straight_clear", &QualificationCounters::straight_clear)
        .def_readonly("blocked", &QualificationCounters::blocked)
        .def_readonly("side_qualification_attempt",
                      &QualificationCounters::side_qualification_attempt)
        .def_readonly("both_sides_feasible", &QualificationCounters::both_sides_feasible)
        .def_readonly("accepted", &QualificationCounters::accepted)
        .def_readonly("reject_endpoint", &QualificationCounters::reject_endpoint)
        .def_readonly("reject_clearance", &QualificationCounters::reject_clearance)
        .def_readonly("reject_different_component",
                      &QualificationCounters::reject_different_component)
        .def_readonly("reject_global_route", &QualificationCounters::reject_global_route)
        .def_readonly("reject_global_astar_budget",
                      &QualificationCounters::reject_global_astar_budget)
        .def_readonly("reject_left_infeasible", &QualificationCounters::reject_left_infeasible)
        .def_readonly("reject_right_infeasible", &QualificationCounters::reject_right_infeasible)
        .def_readonly("reject_both_sides_required",
                      &QualificationCounters::reject_both_sides_required)
        .def_readonly("reject_side_search_budget",
                      &QualificationCounters::reject_side_search_budget)
        .def_readonly("reject_geom_mismatch", &QualificationCounters::reject_geom_mismatch)
        .def_readonly("total_astar_expansions",
                      &QualificationCounters::total_astar_expansions);

    py::class_<BlueprintTask>(m, "BlueprintTask")
        .def(py::init<>())
        .def_readwrite("scene_id", &BlueprintTask::scene_id)
        .def_readwrite("task_id", &BlueprintTask::task_id)
        .def_readwrite("seed", &BlueprintTask::seed)
        .def_readwrite("start_x", &BlueprintTask::start_x)
        .def_readwrite("start_y", &BlueprintTask::start_y)
        .def_readwrite("goal_x", &BlueprintTask::goal_x)
        .def_readwrite("goal_y", &BlueprintTask::goal_y)
        .def_readwrite("initial_yaw", &BlueprintTask::initial_yaw)
        .def_readwrite("flight_height_m", &BlueprintTask::flight_height_m)
        .def_readwrite("behavior_class", &BlueprintTask::behavior_class)
        .def_readwrite("density_class", &BlueprintTask::density_class)
        .def_readwrite("radius_class", &BlueprintTask::radius_class)
        .def_readwrite("distance_class", &BlueprintTask::distance_class)
        .def_readwrite("side_class", &BlueprintTask::side_class)
        .def_readwrite("saw_turn_left", &BlueprintTask::saw_turn_left)
        .def_readwrite("saw_turn_right", &BlueprintTask::saw_turn_right)
        .def_readwrite("saw_normal_correction",
                       &BlueprintTask::saw_normal_correction)
        .def_readwrite("turn_update_count", &BlueprintTask::turn_update_count)
        .def_readwrite("normal_update_count",
                       &BlueprintTask::normal_update_count)
        .def_readwrite("audit", &BlueprintTask::audit)
        .def_readwrite("geom_type", &BlueprintTask::geom_type)
        .def_readwrite("summary", &BlueprintTask::summary)
        .def_readwrite("selection_score", &BlueprintTask::selection_score)
        .def_readwrite("qualification", &BlueprintTask::qualification)
        .def_readwrite("segment_label_counts",
                       &BlueprintTask::segment_label_counts);

    py::class_<RoundStats>(m, "RoundStats")
        .def(py::init<>())
        .def_readonly("round", &RoundStats::round)
        .def_readonly("scenes_generated", &RoundStats::scenes_generated)
        .def_readonly("scenes_valid", &RoundStats::scenes_valid)
        .def_readonly("task_candidates", &RoundStats::task_candidates)
        .def_readonly("cheap_rejected", &RoundStats::cheap_rejected)
        .def_readonly("preflight_attempted", &RoundStats::preflight_attempted)
        .def_readonly("preflight_success", &RoundStats::preflight_success)
        .def_readonly("selected_pool", &RoundStats::selected_pool)
        .def_readonly("elapsed_ms", &RoundStats::elapsed_ms)
        .def_readonly("preflight_avg_ms", &RoundStats::preflight_avg_ms)
        .def_readonly("failure_breakdown", &RoundStats::failure_breakdown)
        .def_readonly("qualification", &RoundStats::qualification)
        .def_readonly("remaining_deficits", &RoundStats::remaining_deficits);

    py::class_<BlueprintResult>(m, "BlueprintResult")
        .def(py::init<>())
        .def_readonly("generation_ok", &BlueprintResult::generation_ok)
        .def_readonly("failure_reason", &BlueprintResult::failure_reason)
        .def_readonly("unmet_quotas", &BlueprintResult::unmet_quotas)
        .def_readonly("scenes", &BlueprintResult::scenes)
        .def_readonly("tasks", &BlueprintResult::tasks)
        .def_readonly("preflighted", &BlueprintResult::preflighted)
        .def_readonly("requested_scenes", &BlueprintResult::requested_scenes)
        .def_readonly("requested_tasks_per_scene",
                      &BlueprintResult::requested_tasks_per_scene)
        .def_readonly("scenes_generated", &BlueprintResult::scenes_generated)
        .def_readonly("scenes_valid", &BlueprintResult::scenes_valid)
        .def_readonly("tasks_sampled", &BlueprintResult::tasks_sampled)
        .def_readonly("tasks_preflighted", &BlueprintResult::tasks_preflighted)
        .def_readonly("tasks_pool_target", &BlueprintResult::tasks_pool_target)
        .def_readonly("tasks_pool_accepted",
                      &BlueprintResult::tasks_pool_accepted)
        .def_readonly("tasks_quota_accepted",
                      &BlueprintResult::tasks_quota_accepted)
        .def_readonly("pool_budget_exhausted",
                      &BlueprintResult::pool_budget_exhausted)
        .def_readonly("strata_required", &BlueprintResult::strata_required)
        .def_readonly("strata_covered", &BlueprintResult::strata_covered)
        .def_readonly("strata_covered_flags",
                      &BlueprintResult::strata_covered_flags)
        .def_readonly("per_scene_accepted",
                      &BlueprintResult::per_scene_accepted)
        .def_readonly("category_counts", &BlueprintResult::category_counts)
        .def_readonly("base_seed", &BlueprintResult::base_seed)
        .def_readonly("total_task_candidates",
                      &BlueprintResult::total_task_candidates)
        .def_readonly("preflight_success_tasks",
                      &BlueprintResult::preflight_success_tasks)
        .def_readonly("cheap_filter_rejected",
                      &BlueprintResult::cheap_filter_rejected)
        .def_readonly("distribution_counts",
                      &BlueprintResult::distribution_counts)
        .def_readonly("distribution_histograms",
                      &BlueprintResult::distribution_histograms)
        .def_readonly("remaining_deficits",
                      &BlueprintResult::remaining_deficits)
        .def_readonly("warnings", &BlueprintResult::warnings)
        .def_readonly("generation_rounds", &BlueprintResult::generation_rounds)
        .def_readonly("timing_ms", &BlueprintResult::timing_ms)
        .def_readonly("selected_scene_ids", &BlueprintResult::selected_scene_ids)
        .def_readonly("hard_minimums_met", &BlueprintResult::hard_minimums_met)
        .def_readonly("soft_targets_met", &BlueprintResult::soft_targets_met)
        .def_readonly("preflight_attempt_count",
                      &BlueprintResult::preflight_attempt_count)
        .def_readonly("preflight_success_count",
                      &BlueprintResult::preflight_success_count)
        .def_readonly("preflight_failure_count",
                      &BlueprintResult::preflight_failure_count)
        .def_readonly("total_preflight_ticks",
                      &BlueprintResult::total_preflight_ticks)
        .def_readonly("full_preflight_attempted",
                      &BlueprintResult::full_preflight_attempted)
        .def_readonly("full_preflight_success",
                      &BlueprintResult::full_preflight_success)
        .def_readonly("selected_scene_count",
                      &BlueprintResult::selected_scene_count)
        .def_readonly("preflight_acceptance_ratio",
                      &BlueprintResult::preflight_acceptance_ratio)
        .def_readonly("selected_per_preflight_ratio",
                      &BlueprintResult::selected_per_preflight_ratio)
        .def_readonly("budget_exhausted_reason",
                      &BlueprintResult::budget_exhausted_reason)
        .def_readonly("round_logs", &BlueprintResult::round_logs)
        .def_readonly("qualification_rejected",
                      &BlueprintResult::qualification_rejected)
        .def_readonly("task_candidates_generated",
                      &BlueprintResult::task_candidates_generated)
        .def_readonly("endpoint_pass_count",
                      &BlueprintResult::endpoint_pass_count)
        .def_readonly("connectivity_pass_count",
                      &BlueprintResult::connectivity_pass_count)
        .def_readonly("straight_clear_count",
                      &BlueprintResult::straight_clear_count)
        .def_readonly("blocked_count", &BlueprintResult::blocked_count)
        .def_readonly("side_qualification_attempt_count",
                      &BlueprintResult::side_qualification_attempt_count)
        .def_readonly("both_sides_feasible_count",
                      &BlueprintResult::both_sides_feasible_count)
        .def_readonly("qualification_accept_count",
                      &BlueprintResult::qualification_accept_count)
        .def_readonly("total_astar_expansions",
                      &BlueprintResult::total_astar_expansions)
        .def_readonly("qualification_pass_ratio",
                      &BlueprintResult::qualification_pass_ratio)
        .def_readonly("full_preflight_success_after_qualification_ratio",
                      &BlueprintResult::full_preflight_success_after_qualification_ratio)
        .def_readonly("qualification", &BlueprintResult::qualification);

    // ── TruthCylinderAudit (exact cylinder swept audit, judge-only) ─
    py::class_<TruthCylinderAudit>(m, "TruthCylinderAudit")
        .def(py::init<>())
        .def("configure",
             [](TruthCylinderAudit& self,
                const std::vector<std::vector<double>>& obstacles,
                double vehicle_radius,
                const std::vector<double>& min_bounds,
                const std::vector<double>& max_bounds) {
                 std::vector<BlueprintObstacle> obs;
                 int id = 0;
                 for (const auto& o : obstacles) {
                     if (o.size() < 3) continue;
                     BlueprintObstacle ob;
                     ob.x = o[0];
                     ob.y = o[1];
                     ob.radius = o[2];
                     ob.height_m = o.size() > 3 ? o[3] : 8.0;
                     ob.id = id++;
                     obs.push_back(ob);
                 }
                 Vec2d mn(-1e9, -1e9), mx(1e9, 1e9);
                 if (min_bounds.size() >= 2) {
                     mn = Vec2d(min_bounds[0], min_bounds[1]);
                 }
                 if (max_bounds.size() >= 2) {
                     mx = Vec2d(max_bounds[0], max_bounds[1]);
                 }
                 self.configure(obs, vehicle_radius, mn, mx);
             },
             py::arg("obstacles"), py::arg("vehicle_radius"),
             py::arg("min_bounds"), py::arg("max_bounds"))
        .def("point_clearance", &TruthCylinderAudit::pointClearance,
             py::arg("x"), py::arg("y"))
        .def("segment_min_clearance",
             &TruthCylinderAudit::segmentMinClearance,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"))
        .def("segment_collision", &TruthCylinderAudit::segmentCollision,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"))
        .def("segment_crosses_bounds",
             &TruthCylinderAudit::segmentCrossesBounds,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("radius"))
        .def("point_out_of_bounds", &TruthCylinderAudit::pointOutOfBounds,
             py::arg("x"), py::arg("y"), py::arg("radius"))
        .def("brake_risk",
             [](TruthCylinderAudit& self, double x, double y, double vx,
                double vy, double max_decel, double stop_margin) {
                 double risk = 0.0;
                 bool would_trigger = false;
                 self.brakeRisk(x, y, vx, vy, max_decel, stop_margin, risk,
                                would_trigger);
                 return py::make_tuple(risk, would_trigger);
             },
             py::arg("x"), py::arg("y"), py::arg("vx"), py::arg("vy"),
             py::arg("max_decel"), py::arg("stop_margin"));

    // ── SceneTaskBlueprintGenerator (config read from a py dict) ───
    py::class_<SceneTaskBlueprintGenerator>(m, "SceneTaskBlueprintGenerator")
        .def(py::init<>())
        .def("configure",
             [](SceneTaskBlueprintGenerator& self, const Params2D& params,
                const py::dict& cfg) {
                 SceneTaskBlueprintGenerator::Config c;
                 auto get_int = [&](const char* k, int dflt) {
                     return cfg.contains(k) ? py::cast<int>(cfg[k]) : dflt;
                 };
                 auto get_uint = [&](const char* k, uint64_t dflt) {
                     return cfg.contains(k) ? py::cast<uint64_t>(cfg[k]) : dflt;
                 };
                 auto get_dbl = [&](const char* k, double dflt) {
                     return cfg.contains(k) ? py::cast<double>(cfg[k]) : dflt;
                 };
                 c.scene_count = get_int("scene_count", 10);
                 c.tasks_per_scene = get_int("tasks_per_scene", 8);
                 c.minimum_tasks_per_scene =
                     get_int("minimum_tasks_per_scene", 6);
                 c.base_seed = get_uint("base_seed", 260812);
                 c.flight_height_m = get_dbl("flight_height_m", 2.0);
                 c.flight_height_min_m =
                     get_dbl("flight_height_min_m", c.flight_height_m);
                 c.flight_height_max_m =
                     get_dbl("flight_height_max_m", c.flight_height_m);
                 c.obstacle_height_m = get_dbl("obstacle_height_m", 8.0);
                 c.obstacle_height_min_m =
                     get_dbl("obstacle_height_min_m", c.obstacle_height_m);
                 c.obstacle_height_max_m =
                     get_dbl("obstacle_height_max_m", c.obstacle_height_m);
                 c.require_full_strata_coverage =
                     cfg.contains("require_full_strata_coverage")
                         ? py::cast<bool>(cfg["require_full_strata_coverage"])
                         : true;
                 c.min_surface_gap_m = get_dbl("min_surface_gap_m", 1.2);
                 c.boundary_margin_m = get_dbl("boundary_margin_m", 1.2);
                 c.radius_min_m = get_dbl("radius_min_m", 0.1);
                 c.radius_max_m = get_dbl("radius_max_m", 2.0);
                 c.max_obstacles = get_int("max_obstacles", 20);
                 c.vehicle_radius_m = get_dbl("vehicle_radius_m", 0.30);
                 c.navigation_clearance_m =
                     get_dbl("navigation_clearance_m", 0.30);
                 c.free_cell_surface_clearance_m =
                     get_dbl("free_cell_surface_clearance_m", 0.50);
                 c.esdf_resolution_m = get_dbl("esdf_resolution_m", 0.1);
                 c.max_generation_attempts =
                     get_int("max_generation_attempts", 24);
                 c.min_task_distance_m = get_dbl("min_task_distance_m", 4.0);
                 c.max_task_distance_m = get_dbl("max_task_distance_m", 20.0);
                 c.initial_yaw_bias_deg =
                     get_dbl("initial_yaw_bias_deg", 15.0);
                 c.task_sample_attempts = get_int("task_sample_attempts", 200);
                 c.candidate_pool_multiplier =
                     get_int("candidate_pool_multiplier", 4);
                 c.qualification_attempt_budget = get_uint(
                     "qualification_attempt_budget", 400);
                 c.preflight_qualification_max_ticks = get_uint(
                     "preflight_qualification_max_ticks", 1800);
                 c.min_per_behavior = get_int("min_per_behavior", 2);
                 c.min_turn_per_side = get_int("min_turn_per_side", 2);
                 c.max_left_right_imbalance =
                     get_int("max_left_right_imbalance", 2);
                 c.min_per_density_level =
                     get_int("min_per_density_level", 4);
                 c.min_per_radius_level =
                     get_int("min_per_radius_level", 4);
                 c.min_per_distance_level =
                     get_int("min_per_distance_level", 4);
                 c.distance_short_max_m =
                     get_dbl("distance_short_max_m", 9.0);
                 c.distance_long_min_m = get_dbl("distance_long_min_m", 15.0);
                 c.radius_small_max_m = get_dbl("radius_small_max_m", 0.6);
                 c.radius_large_min_m = get_dbl("radius_large_min_m", 1.4);
                 c.density_sparse_max = get_dbl("density_sparse_max", 7.0);
                 c.density_dense_min = get_dbl("density_dense_min", 14.0);
                 c.long_takeover_min_ticks =
                     get_uint("long_takeover_min_ticks", 30);
                 // ── NEW: the `blueprint_generation` section drives the
                 //    deficit-driven pipeline (optional; legacy keys above
                 //    remain the fallback when it is absent) ─────────
                 if (cfg.contains("blueprint")) {
                     const py::dict bp = py::cast<py::dict>(cfg["blueprint"]);
                     parseBlueprintConfig(bp, c.blueprint);
                     c.blueprint_explicit = true;
                 }
                 self.configure(params, c);
             },
             py::arg("params"), py::arg("config"))
        .def("generate", &SceneTaskBlueprintGenerator::generate);

    // ── NEW: SceneMetadata / Histogram1D / TaskDistributionSummary ─
    py::class_<SceneMetadata>(m, "SceneMetadata")
        .def(py::init<>())
        .def_readwrite("profile", &SceneMetadata::profile)
        .def_readwrite("structure_orientation",
                       &SceneMetadata::structure_orientation)
        .def_readwrite("obstacle_count", &SceneMetadata::obstacle_count)
        .def_readwrite("radius_min", &SceneMetadata::radius_min)
        .def_readwrite("radius_max", &SceneMetadata::radius_max)
        .def_readwrite("radius_mean", &SceneMetadata::radius_mean)
        .def_readwrite("tiny_count", &SceneMetadata::tiny_count)
        .def_readwrite("small_count", &SceneMetadata::small_count)
        .def_readwrite("medium_count", &SceneMetadata::medium_count)
        .def_readwrite("large_count", &SceneMetadata::large_count)
        .def_readwrite("local_density_proxy", &SceneMetadata::local_density_proxy)
        .def_readwrite("largest_obstacle_radius",
                       &SceneMetadata::largest_obstacle_radius)
        .def_readwrite("scene_seed", &SceneMetadata::scene_seed)
        .def_readwrite("generation_attempt", &SceneMetadata::generation_attempt)
        .def_readwrite("cluster_count", &SceneMetadata::cluster_count)
        .def_readwrite("free_space_ratio", &SceneMetadata::free_space_ratio)
        .def_readwrite("estimated_corridor_width",
                       &SceneMetadata::estimated_corridor_width)
        .def_readwrite("geometry_valid", &SceneMetadata::geometry_valid)
        .def_readwrite("geometry_failure_reason",
                       &SceneMetadata::geometry_failure_reason)
        .def_readwrite("planning_valid", &SceneMetadata::planning_valid)
        .def_readwrite("planning_failure_reason",
                       &SceneMetadata::planning_failure_reason);

    py::class_<Histogram1D>(m, "Histogram1D")
        .def(py::init<>())
        .def_readwrite("edges", &Histogram1D::edges)
        .def_readwrite("counts", &Histogram1D::counts)
        .def("total", &Histogram1D::total);

    py::class_<TaskDistributionSummary>(m, "TaskDistributionSummary")
        .def(py::init<>())
        .def_readwrite("task_id", &TaskDistributionSummary::task_id)
        .def_readwrite("scene_id", &TaskDistributionSummary::scene_id)
        .def_readwrite("scene_profile", &TaskDistributionSummary::scene_profile)
        .def_readwrite("task_geom_type", &TaskDistributionSummary::task_geom_type)
        .def_readwrite("straight_distance_m",
                       &TaskDistributionSummary::straight_distance_m)
        .def_readwrite("preflight_path_length_m",
                       &TaskDistributionSummary::preflight_path_length_m)
        .def_readwrite("path_stretch_ratio",
                       &TaskDistributionSummary::path_stretch_ratio)
        .def_readwrite("preflight_duration_s",
                       &TaskDistributionSummary::preflight_duration_s)
        .def_readwrite("preflight_ticks", &TaskDistributionSummary::preflight_ticks)
        .def_readwrite("initial_yaw_error_signed_deg",
                       &TaskDistributionSummary::initial_yaw_error_signed_deg)
        .def_readwrite("initial_yaw_error_abs_deg",
                       &TaskDistributionSummary::initial_yaw_error_abs_deg)
        .def_readwrite("depth_samples", &TaskDistributionSummary::depth_samples)
        .def_readwrite("depth_near_count", &TaskDistributionSummary::depth_near_count)
        .def_readwrite("depth_mid_count", &TaskDistributionSummary::depth_mid_count)
        .def_readwrite("depth_far_count", &TaskDistributionSummary::depth_far_count)
        .def_readwrite("depth_free_count", &TaskDistributionSummary::depth_free_count)
        .def_readwrite("depth_visible_count",
                       &TaskDistributionSummary::depth_visible_count)
        .def_readwrite("depth_min_visible_m",
                       &TaskDistributionSummary::depth_min_visible_m)
        .def_readwrite("depth_mean_visible_m",
                       &TaskDistributionSummary::depth_mean_visible_m)
        .def_readwrite("depth_max_angular_occlusion_deg",
                       &TaskDistributionSummary::depth_max_angular_occlusion_deg)
        .def_readwrite("depth_occupied_ray_ratio",
                       &TaskDistributionSummary::depth_occupied_ray_ratio)
        .def_readwrite("macro_tick_total", &TaskDistributionSummary::macro_tick_total)
        .def_readwrite("macro_pass_count", &TaskDistributionSummary::macro_pass_count)
        .def_readwrite("macro_normal_count", &TaskDistributionSummary::macro_normal_count)
        .def_readwrite("macro_turn_left_count",
                       &TaskDistributionSummary::macro_turn_left_count)
        .def_readwrite("macro_turn_right_count",
                       &TaskDistributionSummary::macro_turn_right_count)
        .def_readwrite("macro_correction_angle_hist",
                       &TaskDistributionSummary::macro_correction_angle_hist)
        .def_readwrite("macro_correction_distance_hist",
                       &TaskDistributionSummary::macro_correction_distance_hist)
        .def_readwrite("local_direct_count", &TaskDistributionSummary::local_direct_count)
        .def_readwrite("local_avoidance_count",
                       &TaskDistributionSummary::local_avoidance_count)
        .def_readwrite("local_deflection_hist",
                       &TaskDistributionSummary::local_deflection_hist)
        .def_readwrite("local_yaw_rate_hist",
                       &TaskDistributionSummary::local_yaw_rate_hist)
        .def_readwrite("local_speed_hist", &TaskDistributionSummary::local_speed_hist)
        .def_readwrite("min_observed_clearance_m",
                       &TaskDistributionSummary::min_observed_clearance_m)
        .def_readwrite("mean_observed_clearance_m",
                       &TaskDistributionSummary::mean_observed_clearance_m)
        .def_readwrite("reached_goal", &TaskDistributionSummary::reached_goal)
        .def_readwrite("collision", &TaskDistributionSummary::collision)
        .def_readwrite("out_of_bounds", &TaskDistributionSummary::out_of_bounds)
        .def_readwrite("minimum_clearance_m",
                       &TaskDistributionSummary::minimum_clearance_m)
        .def("near_depth_ratio", &TaskDistributionSummary::nearDepthRatio)
        .def("mid_depth_ratio", &TaskDistributionSummary::midDepthRatio)
        .def("far_depth_ratio", &TaskDistributionSummary::farDepthRatio)
        .def("free_depth_ratio", &TaskDistributionSummary::freeDepthRatio);
}
