#!/usr/bin/env python3
"""Generate a TEST blueprint (no roslaunch / no Unity) into the dataset root
and report three capability dimensions of the ray-sector planner:

  1. SCALE (大中小): obstacle radius scale — small / medium / large / mixed
     levels, per-scale behaviour distribution (straight / avoidance /
     detour) and how many avoidance/detour segments a trajectory gets.
  2. DISTANCE (远中近): task straight-line distance — short / medium / long,
     per-distance behaviour distribution.
  3. CONSECUTIVE AVOIDANCE (连续避障): how many avoidance segments (light +
     large) a single trajectory accumulates — 0 (clear), 1 (single weave),
     >=2 (consecutive / multi-obstacle weaving) — broken down by scale and
     distance, plus the detour-classified segments (macro handoff).

Writes a test manifest to:
    /home/rgzn/flightmare_ws/il_data_joint_v2/test_blueprint_manifest.json
(collection uses the SAME task/scene schema as the production manifest, so
the test blueprint can be flown with
    manifest_file:=.../test_blueprint_manifest.json)

Usage:
  PATH=/home/rgzn/anaconda3/bin:$PATH python3 gen_test_blueprint.py
"""
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))
sys.path.insert(0, _HERE)

import il_config            # noqa: E402
import il_expert_config     # noqa: E402
import _il_hierarchical_expert as expert_mod  # noqa: E402
from verify_parallel import build_blueprint_config  # noqa: E402

CONFIG = os.path.join(_HERE, "config", "il_dataset_joint_v2_config.yaml")
OUT_DIR = "/home/rgzn/flightmare_ws/il_data_joint_v2"
MANIFEST = os.path.join(OUT_DIR, "test_blueprint_manifest.json")

LEVEL_NAMES = {0: "small", 1: "medium", 2: "large", 3: "mixed"}
AVOID_LABELS = ("light_avoidance", "large_avoidance")
DETOUR_LABELS = ("detour", "medium_detour", "long_detour")


def _fin(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def _summary_dict(sm):
    return {
        "macro5hz": {
            "tick_total": int(sm.macro_tick_total),
            "pass_count": int(sm.macro_pass_count),
            "normal_count": int(sm.macro_normal_count),
            "turn_left_count": int(sm.macro_turn_left_count),
            "turn_right_count": int(sm.macro_turn_right_count),
            "correction_angle_hist": list(
                sm.macro_correction_angle_hist.counts),
            "correction_distance_hist": list(
                sm.macro_correction_distance_hist.counts),
        },
        "local30hz": {
            "direct_count": int(sm.local_direct_count),
            "avoidance_count": int(sm.local_avoidance_count),
            "deflection_hist": list(sm.local_deflection_hist.counts),
            "yaw_rate_hist": list(sm.local_yaw_rate_hist.counts),
            "speed_hist": list(sm.local_speed_hist.counts),
            "min_observed_clearance_m": _fin(sm.min_observed_clearance_m),
            "mean_observed_clearance_m": _fin(sm.mean_observed_clearance_m),
        },
        "quality": {
            "reached_goal": bool(sm.reached_goal),
            "collision": bool(sm.collision),
            "out_of_bounds": bool(sm.out_of_bounds),
            "minimum_clearance_m": _fin(sm.minimum_clearance_m),
        },
    }


def _task_dict(t, level):
    ad = t.audit
    return {
        "scene_id": int(t.scene_id),
        "task_id": int(t.task_id),
        "seed": int(t.seed),
        "start": [float(t.start_x), float(t.start_y)],
        "goal": [float(t.goal_x), float(t.goal_y)],
        "initial_yaw": float(t.initial_yaw),
        "flight_height_m": float(t.flight_height_m),
        "behavior_class": str(t.behavior_class),
        "density_class": str(t.density_class),
        "radius_class": str(t.radius_class),
        "distance_class": str(t.distance_class),
        "side_class": str(t.side_class),
        "geom_type": str(t.geom_type),
        "level": int(level),
        "segment_label_counts": dict(t.segment_label_counts or {}),
        "audit": {
            "accepted": bool(ad.accepted),
            "truth_brake_triggered": bool(ad.truth_brake_triggered),
            "preflight_ticks": int(ad.preflight_ticks),
            "min_truth_clearance_m": float(ad.min_truth_clearance_m),
            "goal_distance_m": float(ad.goal_distance_m),
            "preflight_status": str(ad.preflight_status),
        },
        "summary": _summary_dict(t.summary),
    }


def main():
    cfg = il_config.load_config(CONFIG)
    g = cfg["global"]
    params = il_expert_config.build_params(g, [])
    vradius = float(g.get("vehicle", {}).get("radius_m", 0.30))
    clearance = float(g.get("navigation", {}).get("clearance_m", 0.30))
    bp = build_blueprint_config(g, vradius, clearance)
    sp = bp["blueprint"]["scene_parallel"]

    print(">>> TEST blueprint: levels=%s per_level=%s threads=%s expected=%s"
          % (sp.get("levels"), sp.get("scenes_per_level"), sp.get("threads"),
             sp.get("expected_collect_tasks")))
    t0 = time.time()
    gen = expert_mod.SceneTaskBlueprintGenerator()
    gen.configure(params, bp)
    result = gen.generate()
    wall = time.time() - t0
    print("generate() wall=%.2f s  scenes=%d/%d selected=%d  ok=%s"
          % (wall, result.scenes_valid, result.scenes_generated,
             result.tasks_quota_accepted, result.generation_ok))

    # per-scene level.  With per-level scene counts (scenes_per_level_list)
    # the level of scene_id is the cumulative-prefix band, NOT scene_id //
    # scenes_per_level (that old formula mislabels scenes once the per-level
    # counts differ).  Falls back to the uniform scenes_per_level when the
    # list is absent.
    per_level_n = int(sp.get("scenes_per_level", 10))
    spl = sp.get("scenes_per_level_list", []) or []
    scene_level = {}
    acc = 0
    for level_i in range(int(sp.get("levels", 4))):
        cnt = int(spl[level_i]) if level_i < len(spl) else per_level_n
        for sid in range(acc, acc + max(1, cnt)):
            scene_level[sid] = level_i
        acc += max(1, cnt)
    task_level = [scene_level.get(int(t.scene_id), 0) for t in result.tasks]

    # ── write test manifest (same schema as production) ────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {
        "manifest_kind": "TEST_BLUEPRINT",
        "expert_revision": "r20260823_pool_first_exploration_r30",
        "base_seed": int(result.base_seed),
        "generation_ok": bool(result.generation_ok),
        "scenes": [
            {
                "scene_id": int(s.scene_id),
                "seed": int(s.seed),
                "profile": str(s.profile),
                "planned_radius_class": str(s.planned_radius_class),
                "actual_radius_class": str(s.actual_radius_class),
                "actual_min_radius_m": float(s.actual_min_radius_m),
                "actual_max_radius_m": float(s.actual_max_radius_m),
                "density_class": str(s.density_class),
                "actual_obstacle_count": int(s.actual_obstacle_count),
                "obstacles": [
                    {"id": int(o.id), "x": float(o.x), "y": float(o.y),
                     "radius": float(o.radius),
                     "height_m": float(o.height_m)}
                    for o in s.obstacles],
            }
            for s in result.scenes],
        "tasks": [_task_dict(t, task_level[i])
                  for i, t in enumerate(result.tasks)],
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(">>> manifest written: %s  (tasks=%d scenes=%d)"
          % (MANIFEST, len(manifest["tasks"]), len(manifest["scenes"])))
    def seg_counts(t):
        c = dict(t.segment_label_counts or {})
        return {
            "avoid": sum(c.get(k, 0) for k in AVOID_LABELS),
            "detour": sum(c.get(k, 0) for k in DETOUR_LABELS),
            "straight": c.get("straight", 0),
            "total": sum(c.values()),
        }

    def avoid_bucket(n):
        return "multi(>=2)" if n >= 2 else ("single(1)" if n == 1 else "clear(0)")

    # 1) SCALE (大中小) ─────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("1) SCALE (大中小) — per-level selected tasks + behaviour")
    scale_rows = []
    for L in sorted(set(task_level)):
        name = LEVEL_NAMES.get(L, "level%d" % L)
        ts = [t for t, lv in zip(result.tasks, task_level) if lv == L]
        beh = Counter(t.behavior_class for t in ts)
        dist = Counter(t.distance_class for t in ts)
        avoid = Counter(avoid_bucket(seg_counts(t)["avoid"]) for t in ts)
        det = sum(seg_counts(t)["detour"] for t in ts)
        n_cyl = [s.actual_obstacle_count
                 for s in result.scenes if scene_level.get(int(s.scene_id)) == L]
        print("  [%s] n=%d  radius=%.2f..%.2fm  cyl=%s" % (
            name, len(ts),
            min((s.actual_min_radius_m for s in result.scenes
                 if scene_level.get(int(s.scene_id)) == L), default=0),
            max((s.actual_max_radius_m for s in result.scenes
                 if scene_level.get(int(s.scene_id)) == L), default=0),
            sorted(n_cyl)))
        print("        behavior=%s" % dict(beh))
        print("        distance=%s" % dict(dist))
        print("        avoidance-segments/task=%s  detour-segs=%d"
              % (dict(avoid), det))
        scale_rows.append((name, len(ts), dict(beh), dict(avoid), det))

    # 2) DISTANCE (远中近) ──────────────────────────────────────────
    print("\n" + "=" * 66)
    print("2) DISTANCE (远中近) — per distance-class behaviour")
    for dc in ("short", "medium", "long"):
        ts = [t for t in result.tasks if t.distance_class == dc]
        beh = Counter(t.behavior_class for t in ts)
        avoid = Counter(avoid_bucket(seg_counts(t)["avoid"]) for t in ts)
        det = sum(seg_counts(t)["detour"] for t in ts)
        dmin = min((seg_counts(t)["total"] for t in ts), default=0)
        dmax = max((seg_counts(t)["total"] for t in ts), default=0)
        print("  [%s] n=%d  behavior=%s" % (dc, len(ts), dict(beh)))
        print("        avoidance-segments/task=%s  detour-segs=%d"
              % (dict(avoid), det))

    # 3) CONSECUTIVE AVOIDANCE (连续避障) ───────────────────────────
    print("\n" + "=" * 66)
    print("3) CONSECUTIVE AVOIDANCE (连续避障) — avoidance segments per task")
    all_avoid = Counter(avoid_bucket(seg_counts(t)["avoid"])
                        for t in result.tasks)
    print("  pool (400 tasks): %s" % dict(all_avoid))
    # multi-avoidance tasks that ALSO have macro detour segments
    multi_det = [t for t in result.tasks
                 if seg_counts(t)["avoid"] >= 2 and seg_counts(t)["detour"] >= 1]
    single_det = [t for t in result.tasks
                  if seg_counts(t)["avoid"] == 1 and seg_counts(t)["detour"] >= 1]
    print("  multi-avoid + macro-detour tasks = %d" % len(multi_det))
    print("  single-avoid + macro-detour tasks = %d" % len(single_det))
    # top multi-avoid tasks by scale and distance
    print("  multi-avoid (>=2) by scale:")
    for L in sorted(set(task_level)):
        name = LEVEL_NAMES.get(L, "L%d" % L)
        n = sum(1 for t, lv in zip(result.tasks, task_level)
                if lv == L and seg_counts(t)["avoid"] >= 2)
        print("    %-7s %d" % (name, n))
    print("  multi-avoid (>=2) by distance:")
    for dc in ("short", "medium", "long"):
        n = sum(1 for t in result.tasks
                if t.distance_class == dc and seg_counts(t)["avoid"] >= 2)
        print("    %-7s %d" % (dc, n))
    # examples: the tasks with the most avoidance segments
    ranked = sorted(result.tasks,
                    key=lambda t: seg_counts(t)["avoid"], reverse=True)[:8]
    print("  top multi-avoid tasks (task_id, scale, dist, seg):")
    for t in ranked:
        lv = scene_level.get(int(t.scene_id), 0)
        print("    task=%d %s %s %s"
              % (int(t.task_id), LEVEL_NAMES.get(lv, "L%d" % lv),
                 t.distance_class, dict(t.segment_label_counts or {})))


if __name__ == "__main__":
    main()
