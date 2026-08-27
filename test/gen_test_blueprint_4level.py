#!/usr/bin/env python3
"""Generate a TEST blueprint for the 4-level handcrafted scenes and run the
NEW local planner (two-segment detour fallback) through the PreflightSimulator
to report per-level behaviour: planner-status distribution, goal-reached rate,
blocked rate and macro labels.  Writes a production-schema manifest loadable by
il_manager collection-only mode so the same scenes can be flown for real.

Usage:
  PATH=/home/rgzn/anaconda3/bin:$PATH python3 gen_test_blueprint_4level.py
"""
import importlib.util
import json
import math
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.expanduser(os.environ.get(
    "IL_DATASET_OUTPUT_DIR", "~/flightmare_ws/il_data_joint_v2"))
_MANIFEST = os.path.join(_OUT_DIR, "test_blueprint_4level_manifest.json")
_CONFIG = "/home/rgzn/flightmare_ws/src/il_dataset/config/il_dataset_joint_v2_config.yaml"

sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import il_config            # noqa: E402
import il_expert_config     # noqa: E402
import _il_hierarchical_expert as expert_mod  # noqa: E402


def _load_scenes():
    path = os.path.join(_HERE, "gen_avoid_scenes_4level.py")
    spec = importlib.util.spec_from_file_location("_gen4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SCENES, mod.sample_initial_yaw, mod.LEVEL_NAMES


def main():
    SCENES, sample_initial_yaw, LEVEL_NAMES = _load_scenes()
    cfg = il_config.load_config(_CONFIG)
    params = il_expert_config.build_params(cfg["global"], [])
    min_b = [-7.0, 0.0]
    max_b = [10.0, 30.0]

    # Per-level behaviour counters.
    level_stat = {lvl: Counter() for lvl in LEVEL_NAMES}
    level_reached = {lvl: [0, 0] for lvl in LEVEL_NAMES}  # [reached, total]

    tasks_out = []
    scenes_out = []
    scene_id = 0
    task_id = 0

    for sc in SCENES:
        level = sc["level"]
        level_name = LEVEL_NAMES[level]
        scenes_out.append({
            "scene_id": scene_id, "profile": sc["name"], "level": level,
            "level_name": level_name,
            "actual_min_radius_m": min(o[2] for o in sc["obstacles"]),
            "actual_max_radius_m": max(o[2] for o in sc["obstacles"]),
            "actual_obstacle_count": len(sc["obstacles"]),
            "obstacles": [{"id": i, "x": float(o[0]), "y": float(o[1]),
                           "radius": float(o[2]), "height_m": 6.0}
                          for i, o in enumerate(sc["obstacles"])],
        })

        sim = expert_mod.PreflightSimulator(params)
        sim.configure(min_b, max_b, [list(o) for o in sc["obstacles"]])

        import random
        rng = random.Random(20260824 + scene_id * 7919)
        for (sx, sy, gx, gy, label) in sc["tasks"]:
            dx, dy = gx - sx, gy - sy
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)
            yaw_fm = sample_initial_yaw(bearing, rng)
            sim.reset_task([sx, sy], [gx, gy], yaw_fm, 0, 2.0)

            ps = Counter()
            macro = Counter()
            reached = False
            collision = False
            for tick in range(200):
                out, st, coll, gr, oob = sim.step(tick, False)
                ps[out.planner_status] += 1
                if out.macro_update_mask:
                    macro[out.macro_correction_type] += 1
                if coll or oob:
                    collision = True
                    break
                if gr:
                    reached = True
                    break

            level_stat[level]["frames"] += sum(ps.values())
            for k, v in ps.items():
                level_stat[level][k] += v
            for k, v in macro.items():
                level_stat[level]["macro:" + k] += v
            if reached:
                level_reached[level][0] += 1
            level_reached[level][1] += 1

            tasks_out.append({
                "scene_id": scene_id, "task_id": task_id,
                "start": [float(sx), float(sy)], "goal": [float(gx), float(gy)],
                "initial_yaw": float(yaw_fm), "flight_height_m": 2.0,
                "test_label": label, "level": level,
                "preflight": {"reached_goal": reached, "collision": collision,
                              "planner_status": dict(ps),
                              "macro": dict(macro)},
            })
            task_id += 1
        scene_id += 1

    manifest = {
        "manifest_kind": "TEST_4LEVEL_NEW_EXPERT",
        "expert_revision": "r20260823_pool_first_exploration_r30",
        "generation_ok": True,
        "scenes": scenes_out,
        "tasks": tasks_out,
        "preflighted": [],
    }
    os.makedirs(_OUT_DIR, exist_ok=True)
    with open(_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(">>> manifest written: %s  tasks=%d scenes=%d"
          % (_MANIFEST, len(tasks_out), len(scenes_out)))

    # ── report ────────────────────────────────────────────────────
    print()
    print("%-6s %-8s %-8s %s" % ("level", "reached", "blocked%", "planner_status / macro"))
    for lvl in sorted(level_stat):
        st = level_stat[lvl]
        frames = st.get("frames", 0)
        blocked = (st.get("NO_SAFE_CANDIDATE", 0) +
                   st.get("BLOCKED_BY_OBSERVED_OBSTACLE", 0) +
                   st.get("EMERGENCY_BRAKE", 0))
        r = level_reached[lvl]
        print("%-6s %-8s %-8s" % (
            LEVEL_NAMES[lvl], "%d/%d" % (r[0], r[1]),
            "%.1f%%" % (100.0 * blocked / max(1, frames))))
        for k in sorted(st):
            if k == "frames":
                continue
            print("       %-28s %d (%.1f%%)" % (k, st[k], 100.0 * st[k] / max(1, frames)))


if __name__ == "__main__":
    main()
