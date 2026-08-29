#!/usr/bin/env python3
"""Step ep2 with the known-rect preflight and print the 5 Hz macro decision
trail: side evidence (left/right bypass, goal in FOV, corridor blocked),
the chosen correction type and the vehicle position — to answer why the
drone detours AWAY from the goal.
"""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_PKG, "scripts"))

import il_config          # noqa: E402
import il_expert_config   # noqa: E402
import _il_hierarchical_expert as expert_mod  # noqa: E402

DATA = "/home/rgzn/flightmare_ws/il_data_joint_v2_col5_pilot_outdoor_low"
CFG = os.path.join(_PKG, "config", "il_dataset_outdoor_low_config.yaml")
EP = "joint_v2_000002_7fe79a3c"
MAX_TICKS = 400


def main():
    cfg = il_config.load_config(CFG)
    g = cfg["global"]
    params = il_expert_config.build_params(g, [])
    min_b, max_b = il_expert_config.build_scene_bounds(g)

    man = json.load(open(os.path.join(DATA, "joint_v2_blueprint_manifest.json")))
    scenes = {s["scene_id"]: s for s in man["scenes"]}

    bp = g.get("blueprint_generation", {}) or {}
    known_rects = []
    fpath = bp.get("known_obstacles_file", "") or ""
    if fpath:
        data = json.load(open(os.path.expanduser(str(fpath))))
        for c in (data.get("clusters", []) or []):
            w = float(c.get("w", 0.0)); hh = float(c.get("h", 0.0))
            x = float(c.get("x", 0.0)); y = float(c.get("y", 0.0))
            if w <= 0.0 or hh <= 0.0:
                continue
            known_rects.append([x - w / 2.0, y - hh / 2.0,
                                x + w / 2.0, y + hh / 2.0])

    md = json.load(open(os.path.join(DATA, "_failed", EP, "metadata.json")))
    sc = scenes[md["scene_id"]]
    obstacles = [[o["x"], o["y"], o["radius"]] for o in sc["obstacles"]]

    sim = expert_mod.PreflightSimulator(params)
    sim.configure(min_b, max_b, obstacles, known_rects)
    sx, sy = md["start_world"][0], md["start_world"][1]
    gx, gy = md["goal_world"][0], md["goal_world"][1]
    sim.reset_task([sx, sy], [gx, gy], md["initial_yaw"], 0, md["start_world"][2])

    def goal_bearing(x, y, yaw):
        b = math.atan2(gy - y, gx - x) - yaw
        return math.degrees(wrap(b))

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    print("goal=(%.2f,%.2f) start=(%.2f,%.2f)" % (gx, gy, sx, sy))
    print("tick | x | y | yaw | 目标bearing | corr_type | left_bypass | "
          "right_bypass | goal_fov | corridor_blocked | obs_reason")
    for t in range(1, MAX_TICKS + 1):
        out, st, tc, gr, oob = sim.step(t, False)
        if out.macro_update_mask:
            x, y, yaw = st[0], st[1], st[2]
            print("%4d %7.2f %7.2f %7.1f %8.1f %-12s %s %s %s %s %s" % (
                t, x, y, math.degrees(yaw) % 360, goal_bearing(x, y, yaw),
                out.macro_correction_type,
                int(out.observability_left_bypass_visible),
                int(out.observability_right_bypass_visible),
                int(out.observability_goal_inside_fov),
                int(out.observability_direct_corridor_blocked),
                out.observability_reason))
        if tc or gr or oob:
            print("END tick=%d collision=%s reached=%s oob=%s" %
                  (t, tc, gr, oob))
            break


if __name__ == "__main__":
    sys.exit(main())
