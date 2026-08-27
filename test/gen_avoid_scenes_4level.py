#!/usr/bin/env python3
"""Generate a HAND-DESIGNED 4-LEVEL avoidance test blueprint mirroring the
scene_parallel collection recipe (small / medium / large / mixed), but with
a tiny deterministic layout: 2 scenes per level x 2 tasks per scene = 8
scenes / 16 tasks.

Level structure mirrors il_dataset_joint_v2_config.yaml scene_parallel:

  level 0  small : radius [0.15, 0.5]   occupancy 0.05..0.08 (dense small cyls)
  level 1  medium: radius [0.5, 1.5]    occupancy 0.09..0.13
  level 2  large : radius [1.5, 3.5]    occupancy 0.12..0.18 (big blockers)
  level 3  mixed : radius [0.15, 3.5]   occupancy 0.12..0.16 (big + small)

Every task's straight start->goal line provably crosses an obstacle CORE
(or threads a >=1.2 m narrow passage in the gap scene), so every collected
trajectory must avoid / detour — same verification contract as
gen_avoid_scenes.py.

Writes a production-schema manifest to:
    ~/flightmare_ws/il_data_joint_v2/avoid_scenes_4level_manifest.json
(loadable by il_manager collection-only mode and by rollout_hierarchical.py
via --tasks 4level)

Usage:
  python3 gen_avoid_scenes_4level.py
  IL_DATASET_OUTPUT_DIR=/path/to/output python3 gen_avoid_scenes_4level.py
"""
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.expanduser(os.environ.get(
    "IL_DATASET_OUTPUT_DIR", "~/flightmare_ws/il_data_joint_v2"))
MANIFEST = os.path.join(OUT_DIR, "avoid_scenes_4level_manifest.json")
REGION = {"min_x": -7.0, "max_x": 10.0, "min_y": 0.0, "max_y": 30.0}

# ── Initial-yaw bias: EXACT mirror of gen_avoid_scenes.py (which mirrors
#    TaskCandidateGenerator::sampleInitialYaw). ─────────────────────
YAW_EDGES_DEG = [0.0, 15.0, 35.0, 55.0, 90.0, 150.0, 180.0]
YAW_WEIGHTS = [0.8, 1.2, 2.2, 1.6, 1.0, 0.9]


def _wrap_angle(a):
    return ((a + math.pi) % (2.0 * math.pi)) - math.pi


def sample_initial_yaw(goal_bearing_expert, rng):
    """Mirror TaskCandidateGenerator::sampleInitialYaw (FM convention B)."""
    si = rng.choices(range(len(YAW_WEIGHTS)), weights=YAW_WEIGHTS, k=1)[0]
    lo, hi = YAW_EDGES_DEG[si], YAW_EDGES_DEG[si + 1]
    mag = rng.uniform(lo, hi)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    yaw_error_deg = sign * mag
    expert_yaw = _wrap_angle(
        goal_bearing_expert - math.radians(yaw_error_deg))
    return expert_yaw - math.pi / 2.0


# ── 4-level scene definitions ─────────────────────────────────────
# Each scene carries its scene_parallel level (0=small,1=medium,2=large,
# 3=mixed) so the manifest / rollout can group by level exactly like the
# production collection.  Tasks are start/goal pairs whose straight line
# provably crosses an obstacle core (or threads the gap).
LEVEL_NAMES = {0: "small", 1: "medium", 2: "large", 3: "mixed"}
# Dense layouts auto-generated around the two per-scene task lines
# (on-line core piercers + a zig-zag corridor), keeping surface gap >= 1.6 m.
SCENES = [
    # ── level 0: small (dense small cylinders, r 0.15..0.5) ───────
    {
        "name": "S_small_a", "level": 0,
        "obstacles": [(0, 10, 0.3), (0, 19, 0.35), (2.62, 11.31, 0.22),
                      (-2.22, 20.44, 0.34), (1.9, 20.42, 0.21),
                      (1.98, 7.58, 0.3), (-1.42, 7.98, 0.43),
                      (-0.67, 16.93, 0.21), (2.73, 16.8, 0.43),
                      (-2.9, 17.93, 0.32)],
        "tasks": [
            (0, 7, 0, 13, "small_a1"),    # 6 m, pierces (0,10) r=0.30
            (0, 16, 0, 22, "small_a2"),   # 6 m, pierces (0,19) r=0.35
        ],
    },
    {
        "name": "S_small_b", "level": 0,
        "obstacles": [(-2, 10.5, 0.3), (2, 10.5, 0.3), (2.41, 12.85, 0.3),
                      (3.48, 8.06, 0.38), (-0.66, 7.92, 0.22),
                      (-3.78, 13.06, 0.32), (-1.04, 13.35, 0.21),
                      (-3.6, 8.25, 0.36)],
        "tasks": [
            (-2, 7, -2, 14, "small_b1"),  # 7 m, pierces (-2,10.5) r=0.30
            (2, 7, 2, 14, "small_b2"),    # 7 m, pierces (2,10.5) r=0.30
        ],
    },
    # ── level 1: medium (r 0.5..1.5) ──────────────────────────────
    {
        "name": "S_medium_a", "level": 1,
        "obstacles": [(0, 11.5, 0.8), (0, 20.5, 1), (-1.83, 14.28, 0.87),
                      (2.39, 14.12, 1.07), (2.86, 17.7, 0.76),
                      (-2.84, 9.85, 0.68)],
        "tasks": [
            (0, 8, 0, 15, "medium_a1"),   # 7 m, pierces (0,11.5) r=0.80
            (0, 17, 0, 24, "medium_a2"),  # 7 m, pierces (0,20.5) r=1.00
        ],
    },
    {
        "name": "S_medium_b", "level": 1,
        "obstacles": [(-2, 12, 0.9), (2, 12, 0.7), (-4.19, 9.4, 0.64),
                      (-0.03, 8.67, 0.92), (-3.49, 15.28, 0.99),
                      (4.08, 15.04, 1.26)],
        "tasks": [
            (-2, 8, -2, 16, "medium_b1"),  # 8 m, pierces (-2,12) r=0.90
            (2, 8, 2, 16, "medium_b2"),    # 8 m, pierces (2,12) r=0.70
        ],
    },
    # ── level 2: large (big blockers, r 1.5..3.5) ─────────────────
    {
        "name": "S_large_a", "level": 2,
        "obstacles": [(0, 11, 2), (0, 18, 2.2), (-3.87, 6.43, 2.11)],
        "tasks": [
            (0, 8, 0, 14, "large_a1"),    # 6 m, pierces (0,11) r=2.00
            (0, 4, 0, 24, "large_a2"),    # 20 m, pierces (0,18) r=2.20
        ],
    },
    {
        "name": "S_large_b", "level": 2,
        "obstacles": [(-3, 14, 2.2), (3, 14, 2), (6.59, 18.8, 2.13)],
        "tasks": [
            (-3, 8, -3, 20, "large_b1"),  # 12 m, pierces (-3,14) r=2.20
            (3, 8, 3, 20, "large_b2"),    # 12 m, pierces (3,14) r=2.00
        ],
    },
    # ── level 3: mixed (big + medium + small, r 0.15..3.5) ────────
    {
        "name": "S_mixed_a", "level": 3,
        "obstacles": [(0, 13, 1.8), (0, 21.5, 1.5), (-3.17, 18.39, 0.61),
                      (-3.42, 9.28, 1.15), (2.02, 18.54, 0.39),
                      (3.28, 9.15, 1.6), (-0.3, 9.17, 0.32)],
        "tasks": [
            (0, 8, 0, 18, "mixed_a1"),    # 10 m, pierces (0,13) r=1.80
            (0, 17, 0, 26, "mixed_a2"),   # 9 m, pierces (0,21.5) r=1.50
        ],
    },
    {
        "name": "S_mixed_b", "level": 3,
        "obstacles": [(-2.5, 13, 1.5), (2.5, 13, 1.3), (0.65, 9.59, 0.68),
                      (-5.14, 16.87, 0.76), (3.91, 16.95, 1.26),
                      (-0.69, 16.41, 0.41), (4.51, 9.51, 0.88),
                      (-5.29, 8.86, 0.76)],
        "tasks": [
            (-2.5, 8, -2.5, 18, "mixed_b1"),  # 10 m, pierces (-2.5,13) r=1.50
            (2.5, 8, 2.5, 18, "mixed_b2"),    # 10 m, pierces (2.5,13) r=1.30
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
    obstacles whose SURFACE separation is >= 1.2 m."""
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
    """Same contract as gen_avoid_scenes: in-region, line pierces a core
    (or threads a gap).  Also checks each scene carries a valid level."""
    problems = []
    total = 0
    for sc in scenes:
        if sc.get("level", -1) not in LEVEL_NAMES:
            problems.append("%s: invalid level %r" % (sc["name"],
                                                      sc.get("level")))
        for (sx, sy, gx, gy, label) in sc["tasks"]:
            total += 1
            for (nx, ny) in ((sx, sy), (gx, gy)):
                if not (REGION["min_x"] + 0.6 <= nx <= REGION["max_x"] - 0.6
                        and REGION["min_y"] + 0.6 <= ny <= REGION["max_y"] - 0.6):
                    problems.append("%s/%s: point (%g,%g) out of region"
                                    % (sc["name"], label, nx, ny))
            d = math.hypot(gx - sx, gy - sy)
            if not (4.0 <= d <= 28.0):
                problems.append("%s/%s: task distance %g outside 4..28 m"
                                % (sc["name"], label, d))
            pierced = False
            if sc.get("gap", False):
                pierced = gap_pierced(sc, sx, sy, gx, gy)
            else:
                for (ox, oy, r) in sc["obstacles"]:
                    if line_dist_to_point((sx, sy), (gx, gy), (ox, oy)) < 0.6 * r:
                        pierced = True
                        break
            if not pierced:
                problems.append("%s/%s: line does NOT pierce a core / "
                                "thread the gap" % (sc["name"], label))
    return total, problems


def main():
    total, problems = verify(SCENES)
    print("4-level scenes=%d tasks=%d  verification problems=%d"
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
        level = sc.get("level", 0)
        level_name = LEVEL_NAMES[level]
        # Radius / density class mirror the scene_parallel strata.
        radius_class = ("small" if level == 0 else
                        "medium" if level == 1 else
                        "large" if level == 2 else "mixed")
        density_class = ("sparse" if level == 0 else
                         "medium" if level == 1 else
                         "dense" if level == 2 else "mixed")
        scenes_out.append({
            "scene_id": scene_id,
            "seed": 0,
            "profile": sc["name"],
            "level": level,
            "level_name": level_name,
            "planned_radius_class": "handcrafted",
            "actual_radius_class": radius_class,
            "actual_min_radius_m": min(o[2] for o in sc["obstacles"]),
            "actual_max_radius_m": max(o[2] for o in sc["obstacles"]),
            "density_class": density_class,
            "actual_obstacle_count": len(sc["obstacles"]),
            "obstacles": [
                {"id": i, "x": float(o[0]), "y": float(o[1]),
                 "radius": float(o[2]), "height_m": 6.0}
                for i, o in enumerate(sc["obstacles"])],
        })
        # Deterministic per-scene RNG (same formula as gen_avoid_scenes so
        # the rollout initial yaw matches the collector blueprint).
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
                "initial_yaw": sample_initial_yaw(goal_bearing_expert, rng),
                "flight_height_m": 2.0,
                "behavior_class": "local_avoidance",
                "density_class": density_class,
                "radius_class": radius_class,
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
        "manifest_kind": "HANDCRAFTED_4LEVEL_AVOID_TEST",
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
        print("  L%d %-12s cyl=%d tasks=%d r=[%.2f, %.2f]"
              % (sc["level"], sc["profile"], sc["actual_obstacle_count"], n,
                 sc["actual_min_radius_m"], sc["actual_max_radius_m"]))


if __name__ == "__main__":
    main()
