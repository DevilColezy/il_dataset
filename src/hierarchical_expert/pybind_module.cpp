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
    b.flight_height_m = get_d("flight_height_m", b.flight_height_m);
    b.obstacle_height_m = get_d("obstacle_height_m", b.obstacle_height_m);
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
        if (d.contains("parallel_tasks")) b.parallel_tasks = py::cast<bool>(d["parallel_tasks"]);
        if (d.contains("scene_switch_penalty")) b.scene_switch_penalty = py::cast<double>(d["scene_switch_penalty"]);
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
        // local planner
        .def_readwrite("lp_horizon_s", &Params2D::lp_horizon_s)
        .def_readwrite("lp_dt", &Params2D::lp_dt)
        .def_readwrite("lp_speed_samples", &Params2D::lp_speed_samples)
        .def_readwrite("lp_lateral_ratio_samples",
                       &Params2D::lp_lateral_ratio_samples)
        .def_readwrite("lp_yaw_rate_samples", &Params2D::lp_yaw_rate_samples)
        .def_readwrite("lp_max_speed", &Params2D::lp_max_speed)
        .def_readwrite("lp_max_accel", &Params2D::lp_max_accel)
        .def_readwrite("lp_max_yaw_rate", &Params2D::lp_max_yaw_rate)
        .def_readwrite("lp_max_yaw_accel", &Params2D::lp_max_yaw_accel)
        .def_readwrite("lp_min_clearance", &Params2D::lp_min_clearance)
        .def_readwrite("lp_soft_clearance_radius_m",
                       &Params2D::lp_soft_clearance_radius_m)
        .def_readwrite("lp_clearance_discretization_margin_m",
                       &Params2D::lp_clearance_discretization_margin_m)
        .def_readwrite("lp_obstacle_reaction_time_s",
                       &Params2D::lp_obstacle_reaction_time_s)
        .def_readwrite("lp_control_period_s", &Params2D::lp_control_period_s)
        .def_readwrite("lp_max_allowed_regress_m",
                       &Params2D::lp_max_allowed_regress_m)
        .def_readwrite("lp_limit_cycle_window_ticks",
                       &Params2D::lp_limit_cycle_window_ticks)
        .def_readwrite("lp_limit_cycle_net_progress_m",
                       &Params2D::lp_limit_cycle_net_progress_m)
        .def_readwrite("lp_limit_cycle_min_blocked_ticks",
                       &Params2D::lp_limit_cycle_min_blocked_ticks)
        .def_readwrite("lp_limit_cycle_lateral_flip_count",
                       &Params2D::lp_limit_cycle_lateral_flip_count)
        .def_readwrite("lp_turn_enter_deg", &Params2D::lp_turn_enter_deg)
        .def_readwrite("lp_turn_exit_deg", &Params2D::lp_turn_exit_deg)
        .def_readwrite("lp_turn_exit_max_yaw_rate",
                       &Params2D::lp_turn_exit_max_yaw_rate)
        .def_readwrite("lp_turn_k", &Params2D::lp_turn_k)
        .def_readwrite("lp_near_goal_heading_relax_distance",
                       &Params2D::lp_near_goal_heading_relax_distance)
        .def_readwrite("lp_near_goal_turn_enter_deg",
                       &Params2D::lp_near_goal_turn_enter_deg)
        .def_readwrite("lp_terminal_control_distance",
                       &Params2D::lp_terminal_control_distance)
        .def_readwrite("lp_terminal_speed_gain",
                       &Params2D::lp_terminal_speed_gain)
        .def_readwrite("lp_terminal_max_speed", &Params2D::lp_terminal_max_speed)
        .def_readwrite("lp_terminal_max_yaw_rate",
                       &Params2D::lp_terminal_max_yaw_rate)
        .def_readwrite("lp_min_progress_m", &Params2D::lp_min_progress_m)
        .def_readwrite("lp_min_progress_speed_mps",
                       &Params2D::lp_min_progress_speed_mps)
        .def_readwrite("lp_min_progress_epsilon_m",
                       &Params2D::lp_min_progress_epsilon_m)
        .def_readwrite("lp_target_discontinuity_reset_m",
                       &Params2D::lp_target_discontinuity_reset_m)
        .def_readwrite("lp_nominal_clearance_m",
                       &Params2D::lp_nominal_clearance_m)
        .def_readwrite("lp_risk_corridor_half_width",
                       &Params2D::lp_risk_corridor_half_width)
        .def_readwrite("lp_risk_distance_horizon_m",
                       &Params2D::lp_risk_distance_horizon_m)
        .def_readwrite("lp_risk_ttc_horizon_s", &Params2D::lp_risk_ttc_horizon_s)
        .def_readwrite("lp_risk_trajectory_radius_m",
                       &Params2D::lp_risk_trajectory_radius_m)
        .def_readwrite("lp_avoidance_active_threshold",
                       &Params2D::lp_avoidance_active_threshold)
        .def_readwrite("lp_brake_stop_margin_m",
                       &Params2D::lp_brake_stop_margin_m)
        .def_readwrite("lp_min_executable_prefix_s",
                       &Params2D::lp_min_executable_prefix_s)
        .def_readwrite("lp_scoring_horizon_s", &Params2D::lp_scoring_horizon_s)
        .def_readwrite("lp_cost_tie_tolerance",
                       &Params2D::lp_cost_tie_tolerance)
        .def_readwrite("lp_cross_track_normalize_m",
                       &Params2D::lp_cross_track_normalize_m)
        .def_readwrite("cost_w_progress", &Params2D::cost_w_progress)
        .def_readwrite("cost_w_clearance", &Params2D::cost_w_clearance)
        .def_readwrite("cost_w_smoothness", &Params2D::cost_w_smoothness)
        .def_readwrite("cost_w_speed_change", &Params2D::cost_w_speed_change)
        .def_readwrite("cost_w_yaw_rate_change",
                       &Params2D::cost_w_yaw_rate_change)
        .def_readwrite("cost_w_terminal_heading",
                       &Params2D::cost_w_terminal_heading)
        .def_readwrite("cost_w_velocity_alignment",
                       &Params2D::cost_w_velocity_alignment)
        .def_readwrite("cost_w_cross_track", &Params2D::cost_w_cross_track)
        .def_readwrite("cost_w_obstacle_risk",
                       &Params2D::cost_w_obstacle_risk)
        // macro / corrector
        .def_readwrite("macro_local_failure_duration_s",
                       &Params2D::macro_local_failure_duration_s)
        .def_readwrite("macro_reentry_guard_ticks",
                       &Params2D::macro_reentry_guard_ticks)
        .def_readwrite("macro_correction_enter_stable_ticks",
                       &Params2D::macro_correction_enter_stable_ticks)
        .def_readwrite("macro_observable_frontier_min_distance_m",
                       &Params2D::macro_observable_frontier_min_distance_m)
        .def_readwrite("macro_observable_frontier_min_progress_m",
                       &Params2D::macro_observable_frontier_min_progress_m)
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
        .def_readwrite("macro_corridor_rear_tolerance_m",
                       &Params2D::macro_corridor_rear_tolerance_m)
        .def_readwrite("macro_local_recovery_prefix_m",
                       &Params2D::macro_local_recovery_prefix_m)
        .def_readwrite("macro_local_candidate_bearing_step_deg",
                       &Params2D::macro_local_candidate_bearing_step_deg)
        .def_readwrite("macro_local_candidate_distance_step_m",
                       &Params2D::macro_local_candidate_distance_step_m)
        .def_readwrite("macro_local_target_event_tolerance_m",
                       &Params2D::macro_local_target_event_tolerance_m)
        .def_readwrite("macro_unknown_recovery_threshold_ticks",
                       &Params2D::macro_unknown_recovery_threshold_ticks)
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
        .def_readonly("effective_target_source",
                      &ExpertStepOutput::effective_target_source)
        .def_readonly("target_correction_active",
                      &ExpertStepOutput::target_correction_active)
        .def_readonly("effective_direction_token",
                      &ExpertStepOutput::effective_direction_token)
        .def_readonly("effective_target_world_x",
                      &ExpertStepOutput::effective_target_world_x)
        .def_readonly("effective_target_world_y",
                      &ExpertStepOutput::effective_target_world_y)
        .def_readonly("effective_target_world_valid",
                      &ExpertStepOutput::effective_target_world_valid)
        .def_readonly("target_velocity_flu_x",
                      &ExpertStepOutput::target_velocity_flu_x)
        .def_readonly("target_velocity_flu_y",
                      &ExpertStepOutput::target_velocity_flu_y)
        .def_readonly("target_yaw_rate", &ExpertStepOutput::target_yaw_rate)
        .def_readonly("intent_vx_body", &ExpertStepOutput::intent_vx_body)
        .def_readonly("intent_vy_body", &ExpertStepOutput::intent_vy_body)
        .def_readonly("intent_yaw_rate", &ExpertStepOutput::intent_yaw_rate)
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
                 self.configure(Vec2d(min_bounds[0], min_bounds[1]),
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
                const std::vector<std::vector<double>>& obstacles) {
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
                 self.configure(scene, scene.min_bounds, scene.max_bounds);
             },
             py::arg("min_bounds"), py::arg("max_bounds"),
             py::arg("obstacles"))
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
        .def_readwrite("qualification", &BlueprintTask::qualification);

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
                 c.obstacle_height_m = get_dbl("obstacle_height_m", 8.0);
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
