#!/usr/bin/env python3
"""Unit verification for Phase-1 random heights (per-task flight height +
per-obstacle height).  Generates a FULL blueprint from a config and asserts
every task flight_height_m and every obstacle height_m falls inside the
configured ranges.

Usage:
  source /opt/ros/noetic/setup.bash && source devel/setup.bash
  PATH=/home/rgzn/anaconda3/bin:$PATH python3 verify_random_heights.py \
      config/il_dataset_indoor_config.yaml
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_PKG, "scripts"))

import il_config          # noqa: E402
import il_expert_config   # noqa: E402
import _il_hierarchical_expert as expert_mod  # noqa: E402


def _load_known_rects(bp):
    """Merge inline known_rects + known_obstacles_file clusters -> AABBs."""
    rects = list(bp.get("known_rects", []) or [])
    fpath = bp.get("known_obstacles_file", "") or ""
    if fpath:
        import json as _json
        with open(os.path.expanduser(str(fpath))) as f:
            data = _json.load(f)
        btg = bp.get("task_generation", {}) or {}
        hgt = float(btg.get(
            "obstacle_height_max_m", btg.get("obstacle_height_m", 8.0)))
        for c in (data.get("clusters", []) or []):
            w = float(c.get("w", 0.0)); hh = float(c.get("h", 0.0))
            x = float(c.get("x", 0.0)); y = float(c.get("y", 0.0))
            if w <= 0.0 or hh <= 0.0:
                continue
            rects.append({"min_x": x - w / 2.0, "max_x": x + w / 2.0,
                          "min_y": y - hh / 2.0, "max_y": y + hh / 2.0,
                          "height_m": hgt})
    return rects


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(_PKG, "config", "il_dataset_indoor_config.yaml")
    cfg = il_config.load_config(cfg_path)
    g = cfg["global"]
    params = il_expert_config.build_params(g, [])
    min_b, max_b = il_expert_config.build_scene_bounds(g)

    # Expected ranges (must match the config under test).
    btg = g.get("blueprint_generation", {}).get("task_generation", {})
    f_min = float(btg.get("flight_height_min_m", 2.0))
    f_max = float(btg.get("flight_height_max_m", 2.0))
    o_min = float(btg.get("obstacle_height_min_m", 8.0))
    o_max = float(btg.get("obstacle_height_max_m", 8.0))

    # Build the SAME legacy dict il_manager passes to C++ (minimal but
    # faithful: blueprint section with the height ranges + base seed).
    legacy = {
        "base_seed": int(g.get("blueprint_generation", {}).get(
            "base_seed", g.get("scene_generation", {}).get("seed", 260812))),
        "scene_count": int(g.get("scene_generation", {}).get(
            "scene_count", 10)),
        "tasks_per_scene": int(g.get("scene_generation", {}).get(
            "tasks_per_scene", 8)),
        "minimum_tasks_per_scene": int(g.get("scene_generation", {}).get(
            "minimum_tasks_per_scene", 6)),
        "flight_height_m": float(g.get("task_generation", {}).get(
            "flight_height_m", 2.0)),
        "obstacle_height_m": float(g.get("scene_generation", {}).get(
            "geometry", {}).get("height_m", 8.0)),
        "require_full_strata_coverage": bool(
            g.get("scene_generation", {}).get(
                "require_full_strata_coverage", True)),
        "min_surface_gap_m": float(g.get("scene_generation", {}).get(
            "geometry", {}).get("minimum_surface_gap_m", 1.40)),
        "boundary_margin_m": float(g.get("scene_generation", {}).get(
            "geometry", {}).get("boundary_margin_m", 1.2)),
        "radius_min_m": float(g.get("scene_generation", {}).get(
            "geometry", {}).get("radius_min_m", 0.10)),
        "radius_max_m": float(g.get("scene_generation", {}).get(
            "geometry", {}).get("radius_max_m", 6.0)),
        "max_obstacles": int(g.get("scene_generation", {}).get(
            "geometry", {}).get("max_obstacles", 30)),
        "vehicle_radius_m": float(g.get("vehicle", {}).get("radius_m", 0.30)),
        "navigation_clearance_m": float(g.get("navigation", {}).get(
            "clearance_m", 0.30)),
        "free_cell_surface_clearance_m": float(
            g.get("scene_generation", {}).get("geometry", {}).get(
                "free_cell_surface_clearance_m", 0.5)),
        "esdf_resolution_m": float(g.get("scene_generation", {}).get(
            "geometry", {}).get("esdf_resolution_m", 0.1)),
        "min_task_distance_m": float(g.get("task_generation", {}).get(
            "min_task_distance_m", 4.0)),
        "max_task_distance_m": float(g.get("task_generation", {}).get(
            "max_task_distance_m", 28.0)),
        "task_sample_attempts": int(g.get("task_generation", {}).get(
            "task_sample_attempts", 300)),
        "task_goal_attempts": int(g.get("task_generation", {}).get(
            "task_goal_attempts", 120)),
        "initial_yaw": dict(g.get("blueprint_generation", {}).get(
            "task_generation", {}).get("initial_yaw", {}) or {}),
        "depth_proxy": dict(g.get("blueprint_generation", {}).get(
            "task_generation", {}).get("depth_proxy", {}) or {}),
        "histograms": dict(g.get("blueprint_generation", {}).get(
            "task_generation", {}).get("histograms", {}) or {}),
        "path": dict(g.get("blueprint_generation", {}).get(
            "task_generation", {}).get("path", {}) or {}),
        "performance": dict(g.get("blueprint_generation", {}).get(
            "performance", {}) or {}),
        "scene_parallel": dict(g.get("blueprint_generation", {}).get(
            "scene_parallel", {}) or {}),
        "requirements": dict(g.get("blueprint_generation", {}).get(
            "requirements", {}) or {}),
        "synthetic_observation": dict(
            g.get("blueprint_generation", {}).get(
                "synthetic_observation", {}) or {}),
        "early_termination": dict(g.get("blueprint_generation", {}).get(
            "early_termination", {}) or {}),
        "task_qualification": dict(g.get("blueprint_generation", {}).get(
            "task_qualification", {}) or {}),
        "distribution_targets": list(
            g.get("blueprint_generation", {}).get(
                "distribution_targets", []) or []),
        "control_rate_hz": float(g.get("blueprint_generation", {}).get(
            "control_rate_hz",
            g.get("hierarchical_expert", {}).get("control_hz", 30.0))),
        "macro_probe": dict(g.get("blueprint_generation", {}).get(
            "task_generation", {}).get("macro_probe", {}) or {}),
        "exploration": dict(g.get("blueprint_generation", {}).get(
            "exploration", {}) or {}),
        "legacy": dict(g.get("blueprint_generation", {}).get(
            "legacy", {}) or {}),
    }
    # blueprint sub-dict (single source of truth).
    bp = g.get("blueprint_generation", {}) or {}
    wh = bp.get("warehouse", {}) or {}
    fr = wh.get("free_region", [])
    if not fr or len(fr) < 4:
        he_region = g.get("hierarchical_expert", {}).get("region", {}) or {}
        fr = [float(he_region.get("min_x", -7.0)),
              float(he_region.get("max_x", 10.0)),
              float(he_region.get("min_y", 0.0)),
              float(he_region.get("max_y", 30.0))]
    else:
        fr = [float(v) for v in fr]
    bsg = bp.get("scene_generation", {}) or {}
    known_rects = _load_known_rects(bp)
    legacy["blueprint"] = {
        "warehouse": {"free_region": fr,
                      "wall_extension_m": float(wh.get(
                          "wall_extension_m", 1.0))},
        "vehicle_radius_m": float(g.get("vehicle", {}).get("radius_m", 0.30)),
        "navigation_clearance_m": float(g.get("navigation", {}).get(
            "clearance_m", 0.30)),
        "clearance_discretization_margin_m": float(bsg.get(
            "clearance_discretization_margin_m", 0.05)),
        "generation_margin_m": float(bsg.get("generation_margin_m", 0.05)),
        "min_surface_gap_m": float(bsg.get(
            "min_surface_gap_m",
            g.get("scene_generation", {}).get("geometry", {}).get(
                "minimum_surface_gap_m", 1.40))),
        "boundary_margin_m": float(bsg.get(
            "boundary_margin_m",
            g.get("scene_generation", {}).get("geometry", {}).get(
                "boundary_margin_m", 1.20))),
        "free_cell_surface_clearance_m": float(bsg.get(
            "free_cell_surface_clearance_m",
            g.get("scene_generation", {}).get("geometry", {}).get(
                "free_cell_surface_clearance_m", 0.5))),
        "esdf_resolution_m": float(bsg.get(
            "esdf_resolution_m",
            g.get("scene_generation", {}).get("geometry", {}).get(
                "esdf_resolution_m", 0.1))),
        "min_main_component_area_m2": float(bsg.get(
            "min_main_component_area_m2", 60.0)),
        "use_profile_catalog": bool(bsg.get("use_profile_catalog", True)),
        "profiles": list(bsg.get("profiles", []) or []),
        "profile_sequence": list(bsg.get("profile_sequence", []) or []),
        "min_task_distance_m": float(btg.get(
            "min_task_distance_m",
            g.get("task_generation", {}).get("min_task_distance_m", 4.0))),
        "max_task_distance_m": float(btg.get(
            "max_task_distance_m",
            g.get("task_generation", {}).get("max_task_distance_m", 28.0))),
        "flight_height_m": float(btg.get(
            "flight_height_m",
            g.get("task_generation", {}).get("flight_height_m", 2.0))),
        "flight_height_min_m": float(btg.get(
            "flight_height_min_m", f_min)),
        "flight_height_max_m": float(btg.get(
            "flight_height_max_m", f_max)),
        "obstacle_height_m": float(btg.get(
            "obstacle_height_m",
            g.get("scene_generation", {}).get("geometry", {}).get(
                "height_m", 8.0))),
        "obstacle_height_min_m": float(btg.get(
            "obstacle_height_min_m", o_min)),
        "obstacle_height_max_m": float(btg.get(
            "obstacle_height_max_m", o_max)),
        "task_sample_attempts": int(btg.get(
            "task_sample_attempts",
            g.get("task_generation", {}).get("task_sample_attempts", 300))),
        "task_goal_attempts": int(btg.get("task_goal_attempts", 120)),
        "initial_yaw": dict(btg.get("initial_yaw", {}) or {}),
        "depth_proxy": dict(btg.get("depth_proxy", {}) or {}),
        "histograms": dict(btg.get("histograms", {}) or {}),
        "path": dict(btg.get("path", {}) or {}),
        "known_rects": known_rects,
        "performance": dict(bp.get("performance", {}) or {}),
        "scene_parallel": dict(bp.get("scene_parallel", {}) or {}),
        "requirements": dict(bp.get("requirements", {}) or {}),
        "synthetic_observation": dict(
            bp.get("synthetic_observation", {}) or {}),
        "early_termination": dict(bp.get("early_termination", {}) or {}),
        "task_qualification": dict(bp.get("task_qualification", {}) or {}),
        "distribution_targets": list(
            bp.get("distribution_targets", []) or []),
        "control_rate_hz": float(bp.get(
            "control_rate_hz",
            g.get("hierarchical_expert", {}).get("control_hz", 30.0))),
        "macro_probe": dict(btg.get("macro_probe", {}) or {}),
        "exploration": dict(bp.get("exploration", {}) or {}),
        "legacy": dict(bp.get("legacy", {}) or {}),
    }

    gen = expert_mod.SceneTaskBlueprintGenerator()
    gen.configure(params, legacy)
    result = gen.generate()

    n_scenes = int(result.scenes_generated)
    n_tasks = len(result.tasks)
    print("scenes_generated=%d tasks=%d preflighted=%d quota=%d" % (
        n_scenes, n_tasks, int(result.tasks_preflighted),
        int(result.tasks_quota_accepted)))

    bad_f = 0
    bad_o = 0
    f_seen = set()
    o_seen = set()
    for sc in result.scenes:
        for ob in sc.obstacles:
            h = float(ob.height_m)
            o_seen.add(round(h, 2))
            if not (o_min - 1e-9 <= h <= o_max + 1e-9):
                bad_o += 1
                print("BAD obstacle height %.3f (scene %d)" % (h, sc.scene_id))
    for t in result.tasks:
        h = float(t.flight_height_m)
        f_seen.add(round(h, 2))
        if not (f_min - 1e-9 <= h <= f_max + 1e-9):
            bad_f += 1
            print("BAD flight height %.3f (task %d)" % (h, t.task_id))

    # Known-rect avoidance: no start/goal may sit inside a known AABB.
    def in_known(px, py):
        for r in known_rects:
            if r["min_x"] <= px <= r["max_x"] and r["min_y"] <= py <= r["max_y"]:
                return True
        return False

    bad_k = 0
    for t in result.tasks:
        if in_known(float(t.start_x), float(t.start_y)) or \
           in_known(float(t.goal_x), float(t.goal_y)):
            bad_k += 1
            print("BAD task %d start/goal inside known rect"
                  % (t.task_id))

    # No random cylinder may overlap a known AABB (surface gap check).
    bad_co = 0
    for sc in result.scenes:
        for ob in sc.obstacles:
            for r in known_rects:
                dx = max(r["min_x"] - ob.x, 0.0, ob.x - r["max_x"])
                dy = max(r["min_y"] - ob.y, 0.0, ob.y - r["max_y"])
                if (dx ** 2 + dy ** 2) ** 0.5 < ob.radius:
                    bad_co += 1
                    print("BAD scene %d cylinder (%.2f,%.2f,r=%.2f) overlaps "
                          "known rect" % (sc.scene_id, ob.x, ob.y, ob.radius))
                    break

    print("flight_height range [%.2f, %.2f]: %d distinct values, "
          "%d out of range" % (f_min, f_max, len(f_seen), bad_f))
    print("obstacle_height range [%.2f, %.2f]: %d distinct values, "
          "%d out of range" % (o_min, o_max, len(o_seen), bad_o))
    print("known_rects=%d: %d tasks inside known rect, %d cylinders "
          "overlapping" % (len(known_rects), bad_k, bad_co))

    if n_scenes == 0 or n_tasks == 0:
        print("FAIL: no scenes/tasks generated")
        return 1
    if bad_f or bad_o or bad_k or bad_co:
        print("FAIL: out-of-range heights or known-rect violations")
        return 1
    if len(f_seen) < 2 and f_min != f_max:
        print("WARN: flight heights did not vary (may be degenerate)")
    print("PASS: all heights inside configured ranges, known rects respected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
