#!/usr/bin/env python3
"""Verify the scene-level parallel pipeline (new architecture).

Loads the joint_v2 config (which has scene_parallel.enabled=true), runs the
C++ generator and prints the scene/level/task/label summary plus the timing.
Also reports peak process threads observed during generate().

Usage:
  PATH=/home/rgzn/anaconda3/bin:$PATH python3 verify_scene_parallel.py
"""
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))
sys.path.insert(0, _HERE)

import il_config            # noqa: E402
import il_expert_config     # noqa: E402
import _il_hierarchical_expert as expert_mod  # noqa: E402
from verify_parallel import build_blueprint_config  # noqa: E402

CONFIG = os.path.join(_HERE, "config", "il_dataset_joint_v2_config.yaml")


def main():
    cfg = il_config.load_config(CONFIG)  # validates scene_parallel too
    g = cfg["global"]
    params = il_expert_config.build_params(g, [])
    vradius = float(g.get("vehicle", {}).get("radius_m", 0.30))
    clearance = float(g.get("navigation", {}).get("clearance_m", 0.30))
    bp = build_blueprint_config(g, vradius, clearance)
    sp = bp["blueprint"]["scene_parallel"]
    print(">>> scene_parallel config: enabled=%s threads=%s levels=%s "
          "per_level=%s expected=%s"
          % (sp.get("enabled"), sp.get("threads"), sp.get("levels"),
             sp.get("scenes_per_level"), sp.get("expected_collect_tasks")))

    peak_threads = [0]
    stop = threading.Event()

    def monitor():
        pid = os.getpid()
        while not stop.is_set():
            try:
                with open("/proc/%d/status" % pid) as f:
                    for line in f:
                        if line.startswith("Threads:"):
                            n = int(line.split()[1])
                            peak_threads[0] = max(peak_threads[0], n)
                            break
            except Exception:
                pass
            time.sleep(0.1)

    mon = threading.Thread(target=monitor, daemon=True)
    mon.start()

    t0 = time.time()
    gen = expert_mod.SceneTaskBlueprintGenerator()
    gen.configure(params, bp)
    result = gen.generate()
    wall = time.time() - t0
    stop.set()
    mon.join(timeout=2)

    print("=" * 60)
    print("wall=%.2f s  peak threads=%d" % (wall, peak_threads[0]))
    print("scenes ok/planned = %d / %d" % (result.scenes_valid,
                                           result.scenes_generated))
    print("candidates/preflighted = %d / %d"
          % (result.tasks_sampled, result.tasks_preflighted))
    print("pool accepted / selected = %d / %d"
          % (result.tasks_pool_accepted, result.tasks_quota_accepted))
    print("generation_ok = %s" % result.generation_ok)
    if result.failure_reason:
        print("failure_reason = %s" % result.failure_reason)
    # per-scene obstacle counts (sparse -> dense check)
    print("obstacle counts per scene:")
    for s in result.scenes:
        print("  scene %d level-profile=%s count=%d rmin=%.2f rmax=%.2f"
              % (s.scene_id, s.profile, s.actual_obstacle_count,
                 s.actual_min_radius_m, s.actual_max_radius_m))
    # task distance / behavior classes
    from collections import Counter
    dist = Counter(t.distance_class for t in result.tasks)
    beh = Counter(t.behavior_class for t in result.tasks)
    print("distance classes (selected): %s" % dict(dist))
    print("behavior classes (selected): %s" % dict(beh))

    # per-level breakdown: level = scene_id // scenes_per_level
    sp_cfg = bp["blueprint"]["scene_parallel"]
    per_level_n = int(sp_cfg.get("scenes_per_level", 10))
    levels = int(sp_cfg.get("levels", 4))
    names = ["small", "medium", "large", "mixed"]
    print("=" * 60)
    print("PER-LEVEL BREAKDOWN (selected tasks):")
    for L in range(levels):
        lname = names[L] if L < len(names) else "level%d" % L
        ltasks = [t for t in result.tasks
                  if t.scene_id // per_level_n == L]
        ld = Counter(t.distance_class for t in ltasks)
        lb = Counter(t.behavior_class for t in ltasks)
        print("  [%s] n=%d  dist=%s" % (lname, len(ltasks), dict(ld)))
        print("          beh=%s" % dict(lb))
    # scenes per level: obstacle counts to confirm sparse->dense
    for L in range(levels):
        lname = names[L] if L < len(names) else "level%d" % L
        counts = sorted(s.actual_obstacle_count for s in result.scenes
                        if s.scene_id // per_level_n == L)
        print("  [%s] obstacle counts (sparse->dense): %s" % (lname, counts))
    t = dict(result.timing_ms)
    print("timing: %s" % t)


if __name__ == "__main__":
    main()
