#!/usr/bin/env python3
"""Offline verification of the parallel task-preflight change.

Runs the SAME C++ SceneTaskBlueprintGenerator with
blueprint_generation.performance.parallel_tasks = 0 (serial) vs N (parallel)
and reports wall time, budget usage and the produced blueprint so the two
runs can be compared.  No ROS required.

Usage:
  PATH=/home/rgzn/anaconda3/bin:$PATH python3 verify_parallel.py
"""
import copy
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))

import il_config            # noqa: E402
import il_expert_config     # noqa: E402
import _il_hierarchical_expert as expert_mod  # noqa: E402

CONFIG = os.path.join(_HERE, "config", "il_dataset_joint_v2_config.yaml")


def build_blueprint_config(g, vehicle_radius, clearance):
    """Mirror of il_manager._blueprint_config_dict (kept in sync)."""
    sg = g.get("scene_generation", {}) or {}
    tg = g.get("task_generation", {}) or {}
    geo = sg.get("geometry", {}) or {}
    q = tg.get("quotas", {}) or {}
    bp = g.get("blueprint_generation", {}) or {}

    legacy = {
        "scene_count": int(sg.get("scene_count", 10)),
        "tasks_per_scene": int(sg.get("tasks_per_scene", 8)),
        "minimum_tasks_per_scene": int(sg.get("minimum_tasks_per_scene", 6)),
        "base_seed": int(sg.get("seed", 260812)),
        "flight_height_m": float(tg.get("flight_height_m", 2.0)),
        "obstacle_height_m": float(geo.get("height_m", 8.0)),
        "require_full_strata_coverage": bool(
            sg.get("require_full_strata_coverage", True)),
        "min_surface_gap_m": float(geo.get("minimum_surface_gap_m", 1.40)),
        "boundary_margin_m": float(geo.get("boundary_margin_m", 1.2)),
        "radius_min_m": float(geo.get("radius_min_m", 0.10)),
        "radius_max_m": float(geo.get("radius_max_m", 6.0)),
        "max_obstacles": int(geo.get("max_obstacles", 30)),
        "vehicle_radius_m": float(vehicle_radius),
        "navigation_clearance_m": float(clearance),
        "free_cell_surface_clearance_m": float(
            geo.get("free_cell_surface_clearance_m", 0.5)),
        "esdf_resolution_m": float(geo.get("esdf_resolution_m", 0.1)),
        "max_generation_attempts": int(sg.get("max_generation_attempts", 96)),
        "min_task_distance_m": float(tg.get("min_task_distance_m", 4.0)),
        "max_task_distance_m": float(tg.get("max_task_distance_m", 20.0)),
        "initial_yaw_bias_deg": float(tg.get("initial_yaw_bias_deg", 15.0)),
        "task_sample_attempts": int(tg.get("task_sample_attempts", 300)),
        "candidate_pool_multiplier": int(tg.get("candidate_pool_multiplier", 4)),
        "qualification_attempt_budget": int(tg.get("qualification_attempt_budget", 600)),
        "preflight_qualification_max_ticks": int(
            tg.get("preflight_qualification_max_ticks", 900)),
        "min_per_behavior": int(q.get("min_per_behavior", 2)),
        "min_turn_per_side": int(q.get("min_turn_per_side", 2)),
        "max_left_right_imbalance": int(q.get("max_left_right_imbalance", 2)),
        "min_per_density_level": int(q.get("min_per_density_level", 4)),
        "min_per_radius_level": int(q.get("min_per_radius_level", 4)),
        "min_per_distance_level": int(q.get("min_per_distance_level", 4)),
        "distance_short_max_m": float(q.get("distance_short_max_m", 9.0)),
        "distance_long_min_m": float(q.get("distance_long_min_m", 15.0)),
        "radius_small_max_m": float(q.get("radius_small_max_m", 0.6)),
        "radius_large_min_m": float(q.get("radius_large_min_m", 1.4)),
        "density_sparse_max": float(q.get("density_sparse_max", 7.0)),
        "density_dense_min": float(q.get("density_dense_min", 14.0)),
        "long_takeover_min_ticks": int(q.get("long_takeover_min_ticks", 30)),
    }

    bsg = bp.get("scene_generation", {}) or {}
    btg = bp.get("task_generation", {}) or {}
    perf = bp.get("performance", {}) or {}
    req = bp.get("requirements", {}) or {}
    wh = bp.get("warehouse", {}) or {}
    fr = wh.get("free_region", []) or []
    if not fr or len(fr) < 4:
        he_region = g.get("hierarchical_expert", {}).get("region", {}) or {}
        fr = [float(he_region.get("min_x", -7.0)),
              float(he_region.get("max_x", 10.0)),
              float(he_region.get("min_y", 0.0)),
              float(he_region.get("max_y", 30.0))]
    else:
        fr = [float(v) for v in fr]

    blueprint = {
        "base_seed": int(bp.get("base_seed", sg.get("seed", 260812))),
        "warehouse": {
            "free_region": fr,
            "wall_extension_m": float(wh.get("wall_extension_m", 1.0)),
        },
        "vehicle_radius_m": float(vehicle_radius),
        "navigation_clearance_m": float(clearance),
        "clearance_discretization_margin_m": float(
            bsg.get("clearance_discretization_margin_m", 0.05)),
        "generation_margin_m": float(bsg.get("generation_margin_m", 0.05)),
        "min_surface_gap_m": float(
            bsg.get("min_surface_gap_m", geo.get("minimum_surface_gap_m", 1.40))),
        "boundary_margin_m": float(
            bsg.get("boundary_margin_m", geo.get("boundary_margin_m", 1.20))),
        "free_cell_surface_clearance_m": float(
            bsg.get("free_cell_surface_clearance_m",
                    geo.get("free_cell_surface_clearance_m", 0.5))),
        "esdf_resolution_m": float(
            bsg.get("esdf_resolution_m", geo.get("esdf_resolution_m", 0.1))),
        "min_main_component_area_m2": float(
            bsg.get("min_main_component_area_m2", 60.0)),
        "use_profile_catalog": bool(bsg.get("use_profile_catalog", True)),
        "profiles": list(bsg.get("profiles", []) or []),
        "profile_sequence": list(bsg.get("profile_sequence", []) or []),
        "min_task_distance_m": float(
            btg.get("min_task_distance_m", tg.get("min_task_distance_m", 4.0))),
        "max_task_distance_m": float(
            btg.get("max_task_distance_m", tg.get("max_task_distance_m", 20.0))),
        "flight_height_m": float(
            btg.get("flight_height_m", tg.get("flight_height_m", 2.0))),
        "obstacle_height_m": float(
            btg.get("obstacle_height_m", geo.get("height_m", 8.0))),
        "task_sample_attempts": int(
            btg.get("task_sample_attempts", tg.get("task_sample_attempts", 300))),
        "task_goal_attempts": int(btg.get("task_goal_attempts", 120)),
        "initial_yaw": dict(btg.get("initial_yaw", {}) or {}),
        "depth_proxy": dict(btg.get("depth_proxy", {}) or {}),
        "histograms": dict(btg.get("histograms", {}) or {}),
        "path": dict(btg.get("path", {}) or {}),
        "performance": dict(perf),
        "scene_parallel": dict(bp.get("scene_parallel", {}) or {}),
        "requirements": dict(req),
        "synthetic_observation": dict(bp.get("synthetic_observation", {}) or {}),
        "early_termination": dict(bp.get("early_termination", {}) or {}),
        "task_qualification": dict(bp.get("task_qualification", {}) or {}),
        "distribution_targets": list(bp.get("distribution_targets", []) or []),
        "control_rate_hz": float(
            bp.get("control_rate_hz",
                   (g.get("hierarchical_expert", {}) or {}).get("control_hz", 30.0))),
        "macro_probe": dict(btg.get("macro_probe", {}) or {}),
        "exploration": dict(bp.get("exploration", {}) or {}),
        "legacy": dict(legacy),
    }
    legacy["blueprint"] = blueprint
    return legacy


def run(parallel, seed_override=None, label=""):
    cfg = il_config.load_config(CONFIG)
    g = cfg["global"]
    params = il_expert_config.build_params(g, [])
    vradius = float(g.get("vehicle", {}).get("radius_m", 0.30))
    clearance = float(g.get("navigation", {}).get("clearance_m", 0.30))
    bp = build_blueprint_config(g, vradius, clearance)
    perf = bp["blueprint"]["performance"]
    perf["parallel_tasks"] = parallel
    if seed_override is not None:
        bp["blueprint"]["base_seed"] = seed_override
        bp["base_seed"] = seed_override

    gen = expert_mod.SceneTaskBlueprintGenerator()
    gen.configure(params, bp)
    t0 = time.time()
    result = gen.generate()
    wall = time.time() - t0
    t = dict(result.timing_ms)
    print("=" * 70)
    print("[%s] parallel_tasks=%d  wall=%.1f s" % (label, parallel, wall))
    print("  scenes_gen/valid      : %d / %d" % (result.scenes_generated,
                                                 result.scenes_valid))
    print("  preflighted/attempts  : %d / %d" % (result.tasks_preflighted,
                                                 result.preflight_attempt_count))
    print("  pool/selected         : %d / %d" % (result.tasks_pool_accepted,
                                                 result.tasks_quota_accepted))
    print("  preflight ticks       : %d" % result.total_preflight_ticks)
    print("  generation_ok         : %s" % result.generation_ok)
    print("  budget exhausted      : %s" % result.budget_exhausted_reason)
    print("  timing breakdown (ms) :")
    for k in ("scene_generation", "scene_geometry_cache", "task_qualification",
              "preflight", "selection", "total"):
        kk = k + "_ms"
        if kk in t:
            print("      %-24s %10.1f" % (kk, t[kk]))
    return wall, result


if __name__ == "__main__":
    # Same seed for a fair A/B comparison (default base_seed is 260812).
    seed = 260812
    t_serial, r_serial = run(0, seed_override=seed, label="SERIAL ")
    t_par, r_par = run(8, seed_override=seed, label="PARALLEL (8 workers)")
    print("=" * 70)
    print("SPEEDUP: serial=%.1f s  parallel=%.1f s  ->  %.2fx"
          % (t_serial, t_par, t_serial / max(1e-9, t_par)))
