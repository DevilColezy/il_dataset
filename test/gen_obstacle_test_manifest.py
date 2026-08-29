#!/usr/bin/env python3
"""Generate the outdoor-high obstacle-test manifest.

21 single-obstacle cases (radius x surface-distance grid) + 1 multi-obstacle
case (a 10 m cylinder with 6 small obstacles on both bypass routes), all at
the outdoor-high operating band (flight z 15-17 m, scene 0 INDUSTRIAL).

Output: <out_dir>/joint_v2_blueprint_manifest.json (collection-only mode).

Geometry (single obstacle):
    start  = (SX, 0), heading +X (toward the obstacle)
    centre = (SX + surf + R, 0)          # surface distance `surf` from start
    goal   = (cx + R + clear, 0)         # `clear` behind the far surface,
                                         # clipped to the region [SX, 25]
"""
import argparse
import json
import math
import os
import sys

EXPERTS = "r20260827_raylimit_fovrel_r30"
STACK = "hierarchical_local_v1"
SCHEMA = ["two_level_expert_labels_v1"]

RADII = [0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 15.0]
SURF_DISTS = [0.6, 2.5, 6.0]
SX = -20.0            # single-obstacle start x (region edge)
REGION_MAX_X = 25.0   # hierarchical_expert.region.max_x
FLIGHT_Z = 16.0       # operating altitude (inside [15,17])
OBST_H = 24.0         # obstacle height (pierces the 15-17 m band)

# Multi-obstacle case: 10 m main cylinder at the origin, small obstacles on
# both bypass routes (upper + lower), all outside the main cylinder surface.
M_START = -15.0
M_CENTRE = (0.0, 0.0)
M_R = 10.0
M_GOAL = (18.0, 0.0)
M_SMALL = [
    {"id": 1, "x": 2.0, "y": 12.0, "radius": 0.8, "height_m": OBST_H},
    {"id": 2, "x": 9.0, "y": 13.0, "radius": 1.0, "height_m": OBST_H},
    {"id": 3, "x": 15.0, "y": 11.0, "radius": 0.7, "height_m": OBST_H},
    {"id": 4, "x": 2.0, "y": -12.0, "radius": 1.0, "height_m": OBST_H},
    {"id": 5, "x": 9.0, "y": -13.0, "radius": 0.6, "height_m": OBST_H},
    {"id": 6, "x": 15.0, "y": -11.0, "radius": 1.2, "height_m": OBST_H},
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/home/rgzn/flightmare_ws/il_data_test_obstacles")
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    scenes, tasks = [], []
    sid = tid = 0

    # ── 21 single-obstacle cases ─────────────────────────────────
    for R in RADII:
        for surf in SURF_DISTS:
            cx = SX + surf + R
            clear = min(10.0, REGION_MAX_X - (cx + R) - 2.0)
            clear = max(clear, 3.0)
            gx = cx + R + clear
            name = "r%.1f_surf%.1f" % (R, surf)
            scenes.append({
                "scene_id": sid,
                "obstacles": [{"id": 0, "x": cx, "y": 0.0,
                               "radius": R, "height_m": OBST_H}],
            })
            tasks.append({
                "scene_id": sid, "task_id": tid, "seed": 1000 + sid,
                "start": [SX, 0.0], "goal": [round(gx, 3), 0.0],
                "initial_yaw": -math.pi / 2.0,   # expert yaw 0 = +X
                "flight_height_m": FLIGHT_Z,
                "behavior_class": "blocked", "density_class": "dense",
                "radius_class": "large", "distance_class": "long",
                "side_class": "none",
                "_name": name,
            })
            sid += 1
            tid += 1

    # ── 1 multi-obstacle case ────────────────────────────────────
    obs = [{"id": 0, "x": M_CENTRE[0], "y": M_CENTRE[1],
            "radius": M_R, "height_m": OBST_H}] + M_SMALL
    scenes.append({"scene_id": sid, "obstacles": obs})
    tasks.append({
        "scene_id": sid, "task_id": tid, "seed": 2000,
        "start": [M_START, 0.0], "goal": list(M_GOAL),
        "initial_yaw": -math.pi / 2.0,
        "flight_height_m": FLIGHT_Z,
        "behavior_class": "blocked", "density_class": "dense",
        "radius_class": "large", "distance_class": "long",
        "side_class": "none",
        "_name": "multi_r10_small_obstacles",
    })

    manifest = {
        "expert_revision": EXPERTS,
        "expert_stack_revision": STACK,
        "schema_extensions": SCHEMA,
        "scenes": scenes,
        "tasks": tasks,
        "preflighted": [],
        "note": "manual obstacle-test manifest: %d single + 1 multi" % len(RADII),
    }
    out = os.path.join(a.out_dir, "joint_v2_blueprint_manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    print("wrote %s" % out)
    print("  scenes=%d tasks=%d" % (len(scenes), len(tasks)))
    for t in tasks:
        print("  task %2d %-22s start=%s goal=%s" % (
            t["task_id"], t.get("_name", ""), t["start"], t["goal"]))


if __name__ == "__main__":
    main()
