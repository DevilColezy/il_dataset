#!/usr/bin/env python3
"""Generate a HAND-DESIGNED AVOIDANCE test blueprint (no random sampling):
every task's straight start->goal line provably crosses an obstacle CORE, so
every collected trajectory must avoid / detour (no clear flights).

Scene layout (Unity world, warehouse x in [-7,10], y in [0,30], flight z=2):

  S_small   1 small  cylinder r=0.40 -> light local avoidance
  S_medium  1 medium cylinder r=1.00 -> local avoidance / short macro
  S_large   1 large  cylinder r=2.50 -> macro detour (big blocker)
  S_chain2  2 cylinders on the start->goal line -> CONSECUTIVE avoidance
  S_chain3  3 cylinders on the line      -> consecutive avoidance (x3)
  S_bigsmall large + 2 small on the line -> big macro detour then weaving

Single-obstacle scenes get short(6m)/medium(12m)/long(18m) tasks built by
placing start and goal on the same axis as the cylinder centre (the line
provably cuts the core).  Chain scenes get long line-piercing tasks.

Writes a production-schema manifest to:
    /home/rgzn/flightmare_ws/il_data_joint_v2/avoid_scenes_manifest.json
(loadable by il_manager collection-only mode.)

Usage:
  PATH=/home/rgzn/anaconda3/bin:$PATH python3 gen_avoid_scenes.py
"""
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/home/rgzn/flightmare_ws/il_data_joint_v2"
MANIFEST = os.path.join(OUT_DIR, "avoid_scenes_manifest.json")
REGION = {"min_x": -7.0, "max_x": 10.0, "min_y": 0.0, "max_y": 30.0}

# ── Initial-yaw bias: EXACT mirror of TaskCandidateGenerator::
#    sampleInitialYaw (il_dataset_config.yaml blueprint.task_generation.
#    initial_yaw) — the drone is allowed a layered absolute goal-bearing
#    error at start, sign mirrored left/right, so handcrafted scenes behave
#    like real generated tasks (the 35-55 deg bin is the in-FOV <-> just
#    out-of-FOV boundary; 90-180 deg keep out-of-FOV / rear-goal TURN).
YAW_EDGES_DEG = [0.0, 15.0, 35.0, 55.0, 90.0, 150.0, 180.0]
YAW_WEIGHTS = [0.8, 1.2, 2.2, 1.6, 1.0, 0.9]


def _wrap_angle(a):
    return ((a + math.pi) % (2.0 * math.pi)) - math.pi


def sample_initial_yaw(goal_bearing_expert, rng):
    """Mirror TaskCandidateGenerator::sampleInitialYaw.

    goal_bearing_expert = atan2(goal-start) in the EXPERT frame (yaw=0 -> +X).
    Returns the manifest initial_yaw in FM convention B (expert_yaw - pi/2):
      - weighted stratum pick -> mag uniform in [lo,hi) degrees
      - sign mirrored 50/50 (positive error = goal LEFT of the nose)
      - expert_yaw = wrap(goal_bearing - yaw_error)
      - store expert_yaw - pi/2 (FM yaw=0 -> nose +Y)
    """
    si = rng.choices(range(len(YAW_WEIGHTS)), weights=YAW_WEIGHTS, k=1)[0]
    lo, hi = YAW_EDGES_DEG[si], YAW_EDGES_DEG[si + 1]
    mag = rng.uniform(lo, hi)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    yaw_error_deg = sign * mag
    expert_yaw = _wrap_angle(
        goal_bearing_expert - math.radians(yaw_error_deg))
    return expert_yaw - math.pi / 2.0

# ── scene definitions ─────────────────────────────────────────────
# each scene: name, obstacles [(x,y,r)...], tasks [(sx,sy,gx,gy,label)...]
SCENES = [
    {
        "name": "S_small",
        "obstacles": [(0.0, 10.0, 0.40)],
        "tasks": [
            (0.0, 7.0, 0.0, 13.0, "small_short"),   # 6 m line pierces core
            (0.0, 4.0, 0.0, 16.0, "small_medium"),  # 12 m
            (0.0, 1.0, 0.0, 19.0, "small_long"),    # 18 m
        ],
    },
    {
        "name": "S_medium",
        "obstacles": [(0.0, 12.0, 1.00)],
        "tasks": [
            (0.0, 9.0, 0.0, 15.0, "medium_short"),
            (0.0, 6.0, 0.0, 18.0, "medium_medium"),
            (0.0, 3.0, 0.0, 21.0, "medium_long"),
        ],
    },
    {
        "name": "S_large",
        "obstacles": [(0.0, 13.0, 2.50)],
        "tasks": [
            (0.0, 8.0, 0.0, 16.0, "large_short"),   # 8 m; start 5 m from the
                                                    # r=2.5 core (was 3 m =
                                                    # surface-tangent spawn)
            (0.0, 7.0, 0.0, 19.0, "large_medium"),
            (0.0, 4.0, 0.0, 22.0, "large_long"),
        ],
    },
    {
        "name": "S_chain2",
        "obstacles": [(-2.4, 9.0, 0.50), (2.4, 17.0, 0.70)],
        "tasks": [
            (-6.0, 3.0, 6.0, 23.0, "chain2_long_a"),
            (-5.0, 5.0, 5.0, 21.0, "chain2_long_b"),
        ],
    },
    {
        "name": "S_chain3",
        "obstacles": [(-3.0, 8.5, 0.50), (0.6, 15.1, 0.60),
                      (4.2, 21.7, 0.50)],
        "tasks": [
            (-6.0, 3.0, 6.0, 25.0, "chain3_long"),
        ],
    },
    {
        "name": "S_bigsmall",
        "obstacles": [(-3.0, 8.5, 2.20), (0.6, 15.1, 0.50),
                      (4.2, 21.7, 0.60)],
        "tasks": [
            (-6.0, 3.0, 6.0, 25.0, "bigsmall_long"),
        ],
    },
    # R29o: larger-scale single obstacles (beyond the r=2.5 S_large) to
    # exercise the macro detour around very big blockers.
    {
        "name": "S_xlarge",
        "obstacles": [(0.0, 15.0, 3.00)],
        "tasks": [
            (0.0, 10.0, 0.0, 20.0, "xlarge_short"),   # 10 m, pierces r=3 core
            (0.0, 8.0, 0.0, 22.0, "xlarge_medium"),  # 14 m
            (0.0, 6.0, 0.0, 24.0, "xlarge_long"),    # 18 m
        ],
    },
    {
        "name": "S_xlarge2",
        "obstacles": [(0.0, 15.0, 3.50)],
        "tasks": [
            (0.0, 10.0, 0.0, 20.0, "xlarge2_short"),  # 10 m, pierces r=3.5 core
            (0.0, 7.0, 0.0, 23.0, "xlarge2_medium"), # 16 m
        ],
    },
    # R29o: two obstacles forming a NARROW PASSAGE whose surface gap is
    # 1.6 m (>= the required passable clearance: 2*0.5 handoff + margin;
    # widened from 1.2 m because the 5-deg ray grid cannot thread the
    # ~0.06 m effective corridor of a 1.2 m gap).  The start->goal line
    # must thread the gap, so the drone has to fly through the passage
    # (not around it).
    {
        "name": "S_gap",
        "gap": True,
        "obstacles": [(-1.4, 12.0, 0.60), (1.4, 12.0, 0.60)],
        "tasks": [
            (0.0, 8.0, 0.0, 16.0, "gap_short"),      # straight through the gap
            (0.0, 6.0, 0.0, 18.0, "gap_medium"),     # longer run through the gap
            (-2.0, 7.0, 2.0, 17.0, "gap_diagonal"),  # diagonal through the gap
        ],
    },
]


def line_dist_to_point(a, b, p):
    """Minimum distance from point p to segment a->b."""
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def gap_pierced(sc, sx, sy, gx, gy):
    """True if the start->goal line threads a narrow passage formed by two
    obstacles whose SURFACE separation is >= 1.2 m (the required passable
    clearance).  The line must come within half the gap width of the gap
    midpoint so the drone is forced through the passage."""
    obs = sc["obstacles"]
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            oxi, oyi, ri = obs[i]
            oxj, oyj, rj = obs[j]
            d = math.hypot(oxj - oxi, oyj - oyi)
            surface = d - ri - rj
            if surface < 1.2 - 1e-9:
                continue
            midx = 0.5 * (oxi + oxj)
            midy = 0.5 * (oyi + oyj)
            if line_dist_to_point((sx, sy), (gx, gy),
                                  (midx, midy)) < 0.5 * surface:
                return True
    return False


def verify(scenes):
    problems = []
    total = 0
    for sc in scenes:
        for (sx, sy, gx, gy, label) in sc["tasks"]:
            total += 1
            # 1) inside region (with margin)
            for (nx, ny) in ((sx, sy), (gx, gy)):
                if not (REGION["min_x"] + 0.6 <= nx <= REGION["max_x"] - 0.6
                        and REGION["min_y"] + 0.6 <= ny <= REGION["max_y"] - 0.6):
                    problems.append("%s/%s: point (%g,%g) out of region"
                                    % (sc["name"], label, nx, ny))
            # 2) line pierces at least one obstacle CORE (< 0.6*r) — OR, for
            # a "gap" scene, threads the >=1.2 m narrow passage.
            pierced = False
            if sc.get("gap", False):
                pierced = gap_pierced(sc, sx, sy, gx, gy)
            else:
                for (ox, oy, r) in sc["obstacles"]:
                    d = line_dist_to_point((sx, sy), (gx, gy), (ox, oy))
                    if d < 0.6 * r:
                        pierced = True
                        break
            if not pierced:
                problems.append("%s/%s: line does NOT pierce a core / "
                                "thread the gap" % (sc["name"], label))
    return total, problems


def main():
    total, problems = verify(SCENES)
    print("scenes=%d tasks=%d  verification problems=%d"
          % (len(SCENES), total, len(problems)))
    for p in problems:
        print("  PROBLEM: %s" % p)
    if problems:
        sys.exit(1)

    # ── build production-schema manifest ──────────────────────────
    def empty_summary():
        return {
            "macro5hz": {
                "tick_total": 0, "pass_count": 0, "normal_count": 0,
                "turn_left_count": 0, "turn_right_count": 0,
                "correction_angle_hist": [], "correction_distance_hist": [],
            },
            "local30hz": {
                "direct_count": 0, "avoidance_count": 0,
                "deflection_hist": [], "yaw_rate_hist": [], "speed_hist": [],
                "min_observed_clearance_m": 0.0,
                "mean_observed_clearance_m": 0.0,
            },
            "quality": {"reached_goal": False, "collision": False,
                        "out_of_bounds": False, "minimum_clearance_m": 0.0},
        }

    def empty_audit():
        return {"accepted": True, "truth_brake_triggered": False,
                "preflight_ticks": 0, "min_truth_clearance_m": 0.0,
                "goal_distance_m": 0.0, "preflight_status": "handcrafted"}

    scenes_out = []
    tasks_out = []
    scene_id = 0
    task_id = 0
    for sc in SCENES:
        scenes_out.append({
            "scene_id": scene_id,
            "seed": 0,
            "profile": sc["name"],
            "planned_radius_class": "handcrafted",
            "actual_radius_class": "handcrafted",
            "actual_min_radius_m": min(o[2] for o in sc["obstacles"]),
            "actual_max_radius_m": max(o[2] for o in sc["obstacles"]),
            "density_class": "handcrafted",
            "actual_obstacle_count": len(sc["obstacles"]),
            "obstacles": [
                {"id": i, "x": float(o[0]), "y": float(o[1]),
                 "radius": float(o[2]), "height_m": 6.0}
                for i, o in enumerate(sc["obstacles"])],
        })
        # Deterministic per-scene RNG for the layered initial-yaw bias.
        rng = random.Random(20260824 + scene_id * 7919)
        for (sx, sy, gx, gy, label) in sc["tasks"]:
            dx = gx - sx
            dy = gy - sy
            dist = math.hypot(dx, dy)
            dc = "short" if dist < 8.0 else ("medium" if dist < 16.0
                                             else "long")
            goal_bearing_expert = math.atan2(dy, dx)
            tasks_out.append({
                "scene_id": scene_id,
                "task_id": task_id,
                "seed": 0,
                "start": [float(sx), float(sy)],
                "goal": [float(gx), float(gy)],
                # Layered initial-yaw bias (mirrors the C++ generator):
                # the drone starts slightly off the goal line (in-FOV
                # mostly, some out-of-FOV / rear-goal turns).
                "initial_yaw": sample_initial_yaw(goal_bearing_expert, rng),
                "flight_height_m": 2.0,
                "behavior_class": "local_avoidance",
                "density_class": "handcrafted",
                "radius_class": "handcrafted",
                "distance_class": dc,
                "side_class": "none",
                "geom_type": "HANDCRAFTED_AVOID",
                "test_label": label,
                "audit": empty_audit(),
                "summary": empty_summary(),
            })
            task_id += 1
        scene_id += 1

    manifest = {
        "manifest_kind": "HANDCRAFTED_AVOID_TEST",
        "expert_revision": "r20260823_pool_first_exploration_r30",
        "base_seed": 0,
        "generation_ok": True,
        "scenes": scenes_out,
        "tasks": tasks_out,
        "preflighted": [],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(">>> manifest written: %s  tasks=%d scenes=%d"
          % (MANIFEST, len(tasks_out), len(scenes_out)))
    for sc in scenes_out:
        n = sum(1 for t in tasks_out if t["scene_id"] == sc["scene_id"])
        print("  %-11s cyl=%d tasks=%d r=[%.2f, %.2f]"
              % (sc["profile"], sc["actual_obstacle_count"], n,
                 sc["actual_min_radius_m"], sc["actual_max_radius_m"]))


if __name__ == "__main__":
    main()
