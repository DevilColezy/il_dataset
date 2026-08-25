#!/usr/bin/env python3
"""Trace a single avoid-scene task through preflight: dump planner status
over time to see where/why it stalls."""
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
    scene_id = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    label = sys.argv[2] if len(sys.argv) > 2 else None
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
    sc = next(s for s in m["scenes"] if s["scene_id"] == scene_id)
    sim.configure(min_b, max_b, [[o["x"], o["y"], o["radius"]]
                                 for o in sc["obstacles"]])
    task = None
    for t in m["tasks"]:
        if t["scene_id"] == scene_id and (label is None or
                                          t.get("test_label") == label):
            task = t
            break
    if task is None:
        print("task not found")
        sys.exit(1)
    print("tracing scene %d task %s start=%s goal=%s yaw=%.2f" % (
        scene_id, task.get("test_label"), task["start"], task["goal"],
        task["initial_yaw"]))

    sim.reset_task(task["start"], task["goal"], task["initial_yaw"], 0,
                   flight_z)
    prev = {}
    for tick in range(1, 2400):
        out, state, truth_coll, goal_reached, oob = sim.step(tick)
        px, py, yaw = state[0], state[1], state[2]
        d = math.hypot(px - task["goal"][0], py - task["goal"][1])
        key = (out.fsm_state, out.planner_status, out.effective_target_source,
               out.target_correction_active)
        if key != prev or tick % 60 == 0 or tick < 12:
            print("t=%4d d=%.2f p=(%.1f,%.1f) yaw=%.2f %-14s %-16s src=%s "
                  "corr=%d fail=%d tok=%s brg=%.0f vx=%.2f vy=%.2f" % (
                      tick, d, px, py, yaw, out.fsm_state,
                      out.planner_status, out.effective_target_source,
                      out.target_correction_active,
                      out.consecutive_failures_30hz,
                      out.effective_direction_token,
                      out.target_bearing_error_deg,
                      out.target_velocity_flu_x, out.target_velocity_flu_y))
            prev = key
        if truth_coll or oob or goal_reached:
            print("END tick=%d coll=%s oob=%s goal=%s d=%.2f" % (
                tick, truth_coll, oob, goal_reached, d))
            break


if __name__ == "__main__":
    main()
