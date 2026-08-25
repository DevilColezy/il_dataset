#!/usr/bin/env python3
"""Preflight-verify the handcrafted avoid scenes manifest with the SAME
C++ expert (PreflightSimulator, truth-synthesized obstacles)."""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))
sys.path.insert(0, _HERE)

import il_config            # noqa: E402
import il_expert_config     # noqa: E402
import _il_hierarchical_expert as expert_mod  # noqa: E402

CONFIG = os.path.join(_HERE, "config", "il_dataset_joint_v2_config.yaml")
MANIFEST = "/home/rgzn/flightmare_ws/il_data_joint_v2/avoid_scenes_manifest.json"


def main():
    cfg = il_config.load_config(CONFIG)
    g = cfg["global"]
    errors = []
    params = il_expert_config.build_params(g, errors)
    if errors:
        print("PARAM ERRORS:", errors)
        sys.exit(1)
    min_b, max_b = il_expert_config.build_scene_bounds(g)
    flight_z = float((g.get("task_generation", {}) or {}).get(
        "flight_height_m", 2.0))

    m = json.load(open(MANIFEST))
    sim = expert_mod.PreflightSimulator(params)
    sim.configure(min_b, max_b, [])

    max_ticks = 2400
    print("%-6s %-16s %-10s %-8s %-8s %-10s %s" % (
        "scene", "label", "result", "ticks", "dist_m", "min_clr", "reason"))
    n_ok = 0
    for sc in m["scenes"]:
        obstacles = [[o["x"], o["y"], o["radius"]] for o in sc["obstacles"]]
        sim.configure(min_b, max_b, obstacles)
        for t in m["tasks"]:
            if t["scene_id"] != sc["scene_id"]:
                continue
            start = t["start"]
            goal = t["goal"]
            yaw = t["initial_yaw"]
            sim.reset_task(start, goal, yaw, 0, flight_z)
            tick = 0
            min_clr = 1e9
            last_dist = None
            result = "TIMEOUT"
            while tick < max_ticks:
                tick += 1
                out, state, truth_coll, goal_reached, oob = sim.step(
                    tick)
                min_clr = min(min_clr, float(out.min_observed_clearance_m))
                # state_vec = [px, py, yaw, vx, vy, yaw_rate]
                last_dist = math.hypot(state[0] - goal[0],
                                       state[1] - goal[1])
                if truth_coll or oob:
                    result = "COLLISION" if truth_coll else "OUT_OF_BOUNDS"
                    break
                if goal_reached:
                    result = "GOAL_REACHED"
                    break
                if tick % 300 == 0:
                    # progress check
                    if last_dist is not None and last_dist < 0.05:
                        result = "STALL"
                        break
            label = t.get("test_label", "?")
            ok = result == "GOAL_REACHED"
            n_ok += ok
            print("%-6d %-16s %-10s %-8d %-8.2f %-10.2f %s" % (
                sc["scene_id"], label, result, tick,
                float(last_dist if last_dist is not None else -1), min_clr,
                "" if ok else "FAIL"))
    print("=" * 70)
    print("avoid scenes preflight: %d / %d tasks reached the goal" % (
        n_ok, len(m["tasks"])))


if __name__ == "__main__":
    main()
