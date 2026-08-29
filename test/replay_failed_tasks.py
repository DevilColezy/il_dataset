#!/usr/bin/env python3
"""Replay the previously failed outdoor_low pilot tasks with the REBUILT
expert .so to check whether the detour-bridge fix reduces goal_no_progress.

Uses PreflightSimulator (same expert, truth-synthesized depth) with the FULL
30 Hz tick (no dt_scale) so a limit cycle reproduces faithfully.

Usage:
  source /opt/ros/noetic/setup.bash && source devel/setup.bash
  PATH=/home/rgzn/anaconda3/bin:$PATH python3 replay_failed_tasks.py
"""
import csv
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
MAX_TICKS = 3000  # 100 s at 30 Hz


def main():
    cfg = il_config.load_config(CFG)
    g = cfg["global"]
    params = il_expert_config.build_params(g, [])
    min_b, max_b = il_expert_config.build_scene_bounds(g)

    man = json.load(open(os.path.join(DATA, "joint_v2_blueprint_manifest.json")))
    scenes = {s["scene_id"]: s for s in man["scenes"]}

    # Collect failed goal_no_progress episodes.
    failed = []
    for ep in sorted(os.listdir(os.path.join(DATA, "_failed"))):
        fr = os.path.join(DATA, "_failed", ep, "failure_reason.json")
        if not os.path.isfile(fr):
            continue
        if "progress" not in open(fr).read():
            continue
        md = json.load(open(os.path.join(DATA, "_failed", ep, "metadata.json")))
        failed.append((ep, md))

    # Load fixed known-obstacle AABBs from the config's cluster file so the
    # synthetic patch matches the REAL scene depth.
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
    print("known_rects=%d" % len(known_rects))

    print("replaying %d goal_no_progress episodes (max %d ticks)..." %
          (len(failed), MAX_TICKS))
    for ep, md in failed:
        sc = scenes.get(md["scene_id"])
        if not sc:
            print("%s: scene %d missing" % (ep, md["scene_id"]))
            continue
        obstacles = [[o["x"], o["y"], o["radius"]] for o in sc["obstacles"]]

        sim = expert_mod.PreflightSimulator(params)
        sim.configure(min_b, max_b, obstacles, known_rects)
        sx, sy = md["start_world"][0], md["start_world"][1]
        gx, gy = md["goal_world"][0], md["goal_world"][1]
        fz = md["start_world"][2]
        sim.reset_task([sx, sy], [gx, gy], md["initial_yaw"], 0, fz)

        reached = False
        collided = False
        oob = False
        last_state = None
        last_mode = ""
        n_ticks = 0
        for t in range(1, MAX_TICKS + 1):
            out, state, tc, gr, oob_ = sim.step(t, False)
            last_state = state
            last_mode = getattr(out, "correction_type", "") or ""
            if tc:
                collided = True
                break
            if oob_:
                oob = True
                break
            if gr:
                reached = True
                break
            n_ticks = t

        d = math.hypot(last_state[0] - gx, last_state[1] - gy) if last_state else -1
        print("%s: %s (ticks=%d, final_dist=%.2fm)" % (
            ep,
            "goal_reached" if reached else
            ("collision" if collided else ("out_of_bounds" if oob else "STUCK")),
            n_ticks, d))


if __name__ == "__main__":
    sys.exit(main())
