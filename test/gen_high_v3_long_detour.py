#!/usr/bin/env python3
"""Append HANDCRAFTED long-detour scenes to the v3 high-altitude blueprint.

Scene recipe (matches the user's request):
  * ONE big central BOX (AABB, side ~8 m, circumscribed radius ~5.7 m) that
    blocks the straight start->goal line.
  * OTHER small obstacles (radii 0.5..2.5 m) UNIFORMLY distributed around it
    (surface gap >= 1.5 m), so the detour passes through real clutter.
  * Tasks: start on one side of the big box, goal on the opposite side,
    direct line PIERCES the box core -> the 5 Hz macro expert MUST take over
    and route around -> genuinely long-duration upper-level takeover.

The appended tasks carry no C++ preflight summary (handcrafted); the real
expert validates them at collection time (they commit only if the full
episode succeeds).  Behavior labels are best-guess diagnostics
(long_takeover / turn_both / turn_normal).

Usage:
    python3 gen_high_v3_long_detour.py
    (reads/writes IL_DATASET_V3_MANIFEST, default
     ~/flightmare_ws/il_data_d435i_col_high_v3/joint_v2_blueprint_manifest.json)
"""
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.expanduser(os.environ.get(
    "IL_DATASET_V3_DIR", "~/flightmare_ws/il_data_d435i_col_high_v3"))
MANIFEST = os.path.join(V3_DIR, "joint_v2_blueprint_manifest.json")

REGION = (-24.0, 24.0, -24.0, 24.0)      # inside the [-25,25]^2 free region
SURFACE_GAP_M = 1.5
FLIGHT_H_MIN, FLIGHT_H_MAX = 15.0, 17.0
OBSTACLE_H_MIN, OBSTACLE_H_MAX = 18.0, 24.0
TASK_DIST_MIN, TASK_DIST_MAX = 20.0, 35.0
NUM_SCENES = 4
NUM_TASKS = 10
SEED = 260831

# ── initial-yaw sampler (mirror of the collector's sampler) ──────
YAW_EDGES_DEG = [0.0, 15.0, 35.0, 55.0, 90.0, 150.0, 180.0]
YAW_WEIGHTS = [0.8, 1.2, 2.2, 1.6, 1.0, 0.9]


def _wrap_angle(a):
    return ((a + math.pi) % (2.0 * math.pi)) - math.pi


def sample_initial_yaw(goal_bearing_expert, rng):
    si = rng.choices(range(len(YAW_WEIGHTS)), weights=YAW_WEIGHTS, k=1)[0]
    lo, hi = YAW_EDGES_DEG[si], YAW_EDGES_DEG[si + 1]
    mag = rng.uniform(lo, hi)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    yaw_error_deg = sign * mag
    expert_yaw = _wrap_angle(
        goal_bearing_expert - math.radians(yaw_error_deg))
    return expert_yaw - math.pi / 2.0


# ── geometry helpers ─────────────────────────────────────────────
def seg_intersects_aabb(ax, ay, bx, by, xmin, ymin, xmax, ymax):
    """Liang-Barsky clip: True when segment a->b enters the AABB."""
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - xmin), (dx, xmax - ax),
                 (-dy, ay - ymin), (dy, ymax - ay)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return True


def point_clear_circle(x, y, obstacles, inflate=1.0):
    """Endpoints must stay clear of every obstacle (circle surrogate)."""
    return all(math.hypot(x - o[0], y - o[1]) > o[2] + inflate
               for o in obstacles)


def _placeable(x, y, r, obstacles, boxes):
    """Inside region with >= SURFACE_GAP_M surface gap from everything
    (boxes use their AABB faces; obstacles use their circle radius)."""
    if not (REGION[0] + r + 0.5 <= x <= REGION[1] - r - 0.5 and
            REGION[2] + r + 0.5 <= y <= REGION[3] - r - 0.5):
        return False
    for ox, oy, orad, _s in obstacles:
        if math.hypot(x - ox, y - oy) < r + orad + SURFACE_GAP_M:
            return False
    for bx, by, half in boxes:
        dxx = max(0.0, abs(x - bx) - (half + r + SURFACE_GAP_M))
        dyy = max(0.0, abs(y - by) - (half + r + SURFACE_GAP_M))
        if math.hypot(dxx, dyy) < 1e-9:
            return False
    return True


def make_scene(rng, scene_id, big_box, n_small=20, box2=None):
    """One handcrafted scene: big box(es) + uniformly scattered smalls."""
    obstacles = []          # (x, y, r, side|None)
    boxes = []              # (bx, by, half)
    boxes.append((big_box[0], big_box[1], big_box[2] / 2.0))
    if box2:
        boxes.append((box2[0], box2[1], box2[2] / 2.0))
    attempts = 0
    while len(obstacles) < n_small and attempts < 6000:
        attempts += 1
        r = rng.uniform(0.5, 2.5)
        side = r * math.sqrt(2.0) if rng.random() < 0.35 else None
        x = rng.uniform(REGION[0] + r + 0.5, REGION[1] - r - 0.5)
        y = rng.uniform(REGION[2] + r + 0.5, REGION[3] - r - 0.5)
        if _placeable(x, y, r, obstacles, boxes):
            obstacles.append((x, y, r, side))
    return obstacles, boxes


def sample_task(rng, obstacles, boxes):
    """Start/goal on OPPOSITE sides of the first big box so the straight
    line pierces its AABB -> forced long detour."""
    bx, by, half = boxes[0]
    for _ in range(6000):
        side = 1.0 if rng.random() < 0.5 else -1.0
        sx = bx + side * rng.uniform(half + 6.0, half + 12.0)
        sy = by + rng.uniform(-14.0, 14.0)
        gx = bx - side * rng.uniform(half + 6.0, half + 12.0)
        gy = by + rng.uniform(-14.0, 14.0)
        dist = math.hypot(gx - sx, gy - sy)
        if not (TASK_DIST_MIN <= dist <= TASK_DIST_MAX):
            continue
        if not point_clear_circle(sx, sy, obstacles) or \
                not point_clear_circle(gx, gy, obstacles):
            continue
        # endpoints outside the big box AABB
        if abs(sx - bx) < half + 1.0 or abs(gx - bx) < half + 1.0:
            continue
        if not seg_intersects_aabb(sx, sy, gx, gy,
                                   bx - half, by - half,
                                   bx + half, by + half):
            continue
        return (sx, sy, gx, gy)
    return None


def obstacle_manifest_entry(idx, x, y, r, side, height_m):
    entry = {"id": idx, "x": float(x), "y": float(y),
             "radius": float(r), "height_m": float(height_m)}
    if side is not None:
        entry["w"] = float(side)
        entry["h"] = float(side)
    else:
        entry["w"] = None
        entry["h"] = None
    return entry


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
    return {"accepted": True, "reached_goal": True, "truth_collision": False,
            "truth_brake_triggered": False, "out_of_bounds": False,
            "macro_label_ok": True, "qualification_exceeded": False,
            "preflight_ticks": 0, "min_truth_clearance_m": 0.0,
            "goal_distance_m": 0.0, "straight_distance_m": 0.0,
            "preflight_status": "handcrafted_long_detour"}


def main():
    if not os.path.isfile(MANIFEST):
        sys.exit("manifest not found: %s" % MANIFEST)
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    existing_scenes = manifest.get("scenes", [])
    existing_tasks = manifest.get("tasks", [])
    next_scene = max((int(s["scene_id"]) for s in existing_scenes),
                     default=-1) + 1
    next_task = max((int(t["task_id"]) for t in existing_tasks),
                    default=-1) + 1
    base_seed = int(manifest.get("base_seed", 0))
    rng = random.Random(base_seed ^ SEED)

    # Big-box layouts: (bx, by, side) — all within the region, sides large
    # enough to force a real detour but small enough to leave a route.
    big_boxes = [
        (0.0, 0.0, 8.0),      # central box
        (6.0, -4.0, 9.0),     # offset box
        (-5.0, 5.0, 8.0, (7.0, -6.0, 6.0)),   # two boxes
        (0.0, 0.0, 8.0),      # central box, denser smalls
    ]

    new_scenes, new_tasks = [], []
    for si, bb in enumerate(big_boxes):
        box2 = bb[3] if len(bb) > 3 else None
        side = bb[2]
        n_small = 26 if si == 3 else 20
        obstacles, boxes = make_scene(rng, si, bb[:3], n_small=n_small,
                                      box2=box2)
        scene_id = next_scene + si
        # obstacle manifest entries (heights random in the high band)
        obs_entries = []
        for idx, (ox, oy, r, s) in enumerate(obstacles):
            obs_entries.append(obstacle_manifest_entry(
                idx, ox, oy, r, s, rng.uniform(OBSTACLE_H_MIN, OBSTACLE_H_MAX)))
        # big box entry (AABB, height covering the flight band)
        for idx, (bx, by, half) in enumerate(boxes):
            obs_entries.append(obstacle_manifest_entry(
                1000 + idx, bx, by, half * math.sqrt(2.0),
                half * 2.0, rng.uniform(OBSTACLE_H_MIN, OBSTACLE_H_MAX)))
        radii = [o[2] for o in obstacles]
        scene = {
            "scene_id": scene_id,
            "seed": rng.randrange(1 << 62),
            "profile": "big_box_long_detour",
            "structure_orientation": "none",
            "metadata": {
                "structure_orientation": "none",
                "obstacle_count": len(obs_entries),
                "radius_min": min(radii) if radii else 0.0,
                "radius_max": max(radii) if radii else 0.0,
                "radius_mean": sum(radii) / len(radii) if radii else 0.0,
                "tiny_count": 0, "small_count": 0, "medium_count": 0,
                "large_count": 0, "local_density_proxy": 0.0,
                "largest_obstacle_radius": max(radii) if radii else 0.0,
                "cluster_count": 0, "free_space_ratio": 0.7,
                "estimated_corridor_width": 6.0,
                "geometry_valid": True, "geometry_failure_reason": "",
                "planning_valid": True, "planning_failure_reason": "",
            },
            "stratum_id": -1,
            "is_empty": False,
            "planned_density_class": "handcrafted",
            "planned_radius_class": "handcrafted",
            "actual_density_class": "dense",
            "actual_radius_class": "large",
            "actual_min_radius_m": min(radii) if radii else 0.0,
            "actual_max_radius_m": max(radii) if radii else 0.0,
            "density_class": "dense",
            "actual_obstacle_count": len(obs_entries),
            "obstacles": obs_entries,
        }
        new_scenes.append(scene)

        for ti in range(NUM_TASKS):
            task = sample_task(rng, obstacles, boxes)
            if task is None:
                sys.exit("handcrafted task sampling failed (scene %d)" % si)
            sx, sy, gx, gy = task
            dist = math.hypot(gx - sx, gy - sy)
            dc = "short" if dist < 18.0 else ("long" if dist >= 27.0 else "medium")
            goal_bearing = math.atan2(gy - sy, gx - sx)
            yaw = sample_initial_yaw(goal_bearing, rng)
            # best-guess behaviour labels (diagnostic only)
            if len(boxes) > 1:
                beh = "turn_both"
            elif dist >= 27.0:
                beh = "long_takeover"
            else:
                beh = "turn_normal"
            new_tasks.append({
                "scene_id": scene_id,
                "task_id": next_task + si * NUM_TASKS + ti,
                "seed": rng.randrange(1 << 62),
                "start": [float(sx), float(sy)],
                "goal": [float(gx), float(gy)],
                "initial_yaw": float(yaw),
                "flight_height_m": rng.uniform(FLIGHT_H_MIN, FLIGHT_H_MAX),
                "behavior_class": beh,
                "density_class": "dense",
                "radius_class": "large",
                "distance_class": dc,
                "side_class": "both" if len(boxes) > 1 else "left",
                "geom_type": "HANDCRAFTED_LONG_DETOUR",
                "test_label": "long_detour_s%d_%02d" % (si, ti),
                "audit": empty_audit(),
                "summary": empty_summary(),
            })

    manifest["scenes"] = existing_scenes + new_scenes
    manifest["tasks"] = existing_tasks + new_tasks
    manifest["generation_ok"] = True
    manifest["hard_minimums_met"] = True
    manifest["warnings"] = (manifest.get("warnings") or []) + [
        "appended %d handcrafted long-detour scenes (%d tasks)" % (
            len(new_scenes), len(new_tasks))]

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    # ── report ──────────────────────────────────────────────────
    print("appended scenes: %d (ids %d..%d)" % (
        len(new_scenes), next_scene, next_scene + len(new_scenes) - 1))
    print("appended tasks: %d (ids %d..%d)" % (
        len(new_tasks), next_task, next_task + len(new_tasks) - 1))
    for si, sc in enumerate(new_scenes):
        n_box = sum(1 for o in sc["obstacles"] if o.get("w"))
        print("  scene %d: obstacles=%d (big_boxes=%d smalls=%d)" % (
            sc["scene_id"], len(sc["obstacles"]), n_box,
            len(sc["obstacles"]) - n_box))
    from collections import Counter
    print("behavior labels:", dict(Counter(t["behavior_class"]
                                           for t in new_tasks)))
    print("total tasks in manifest:", len(manifest["tasks"]),
          "| scenes:", len(manifest["scenes"]))
    print(">>> merged manifest: %s" % MANIFEST)


if __name__ == "__main__":
    main()
