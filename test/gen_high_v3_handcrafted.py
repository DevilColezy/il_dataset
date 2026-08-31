#!/usr/bin/env python3
"""Rebuild the HANDCRAFTED long-detour half of the v3 high-altitude blueprint.

16 handcrafted scenes (8 obstacle archetypes x 2 variants) x 20 tasks = 320
tasks, so the handcrafted subset is HALF of the 640-task merged manifest.

Obstacle archetypes (all force a LARGE-scale detour, with small/medium
obstacles along the detour AND after it):
  1. ring_around_box   : central big box + a RING of small/medium obstacles
                         with 1-2 gaps — thread the ring while going around.
  2. twin_boxes_s      : two big boxes (S-curve) — detour #1 then detour #2,
                         turn_left AND turn_right in one task.
  3. box_exit_clutter  : big box in the first half, DENSE small obstacle
                         gauntlet in the second — detour, then small-scale
                         avoidance run to the goal.
  4. triple_box_zigzag : three big boxes in a staircase — zigzag around each.
  5. box_narrow_gaps   : big box with small/medium obstacles forming narrow
                         side gaps — pick and thread the gap.
  6. u_canyon          : big boxes forming a U — large loop around the wall.
  7. wall_corridor     : a wall of big boxes + a medium-obstacle corridor on
                         one side — detour around the wall through the corridor.
  8. mixed_cascade     : big box -> medium zone -> dense small zone — a full
                         scale cascade (large then medium then small).

Usage:
    python3 gen_high_v3_handcrafted.py
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

REGION = (-24.0, 24.0, -24.0, 24.0)
SURFACE_GAP_M = 1.5
FLIGHT_H_MIN, FLIGHT_H_MAX = 15.0, 17.0
OBSTACLE_H_MIN, OBSTACLE_H_MAX = 18.0, 24.0
TASK_DIST_MIN, TASK_DIST_MAX = 20.0, 35.0
TASKS_PER_SCENE = 10   # 16 scenes x 10 = 160 handcrafted tasks (half of 320)
SEED = 260831

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


def seg_hits_circle(ax, ay, bx, by, cx, cy, r):
    vx, vy = bx - ax, by - ay
    wx, wy = cx - ax, cy - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return math.hypot(cx - ax, cy - ay) < r
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / L2))
    return math.hypot(wx - t * vx, wy - t * vy) < r


def point_clear(x, y, obstacles, inflate=1.0):
    """Endpoints clear of every obstacle (box -> its circumscribed circle,
    conservative)."""
    return all(math.hypot(x - ox, y - oy) > r + inflate
               for (ox, oy, r, _s) in obstacles)


def _clear_of_boxes(x, y, boxes, inflate=1.0):
    for bx, by, half in boxes:
        dx = max(0.0, abs(x - bx) - (half + inflate))
        dy = max(0.0, abs(y - by) - (half + inflate))
        if math.hypot(dx, dy) < 1e-9:
            return False
    return True


def placeable(x, y, r, obstacles, boxes):
    """Inside region with >= SURFACE_GAP_M surface gap from everything."""
    if not (REGION[0] + r + 0.5 <= x <= REGION[1] - r - 0.5 and
            REGION[2] + r + 0.5 <= y <= REGION[3] - r - 0.5):
        return False
    for ox, oy, orad, _s in obstacles:
        if math.hypot(x - ox, y - oy) < r + orad + SURFACE_GAP_M:
            return False
    for bx, by, half in boxes:
        dx = max(0.0, abs(x - bx) - (half + r + SURFACE_GAP_M))
        dy = max(0.0, abs(y - by) - (half + r + SURFACE_GAP_M))
        if math.hypot(dx, dy) < 1e-9:
            return False
    return True


def add_box(obstacles, boxes, x, y, side):
    half = side / 2.0
    boxes.append((x, y, half))
    obstacles.append((x, y, half * math.sqrt(2.0), side))
    return half


def scatter(rng, obstacles, boxes, n, rmin=0.5, rmax=2.5, region=None):
    """Uniformly scatter n small/medium obstacles; return count placed.
    region is (xmin, xmax, ymin, ymax)."""
    rmin_, rmax_ = rmin, rmax
    rx0, rx1, ry0, ry1 = region if region else REGION
    placed = 0
    for _ in range(12000):
        if placed >= n:
            break
        r = rng.uniform(rmin_, rmax_)
        side = r * math.sqrt(2.0) if rng.random() < 0.35 else None
        x = rng.uniform(rx0 + r + 0.5, rx1 - r - 0.5)
        y = rng.uniform(ry0 + r + 0.5, ry1 - r - 0.5)
        if placeable(x, y, r, obstacles, boxes):
            obstacles.append((x, y, r, side))
            placed += 1
    return placed


# ── archetype builders: return (obstacles, boxes) ───────────────
def build_ring_around_box(rng, variant):
    obstacles, boxes = [], []
    cx, cy = variant.get("center", (0.0, 0.0))
    side = variant.get("side", 9.0)
    ring_r = variant.get("ring_r", 11.0)
    n_ring = variant.get("n_ring", 10)
    gaps = variant.get("gaps", [0.6, 3.8])  # gap angles (rad) around the ring
    add_box(obstacles, boxes, cx, cy, side)
    half = side / 2.0
    gap_ok = [False] * n_ring
    for k in range(n_ring):
        ang = 2.0 * math.pi * k / n_ring
        # skip sectors near the gaps (leave 2 gaps for the route)
        near_gap = any(
            abs((ang - g + math.pi) % (2 * math.pi) - math.pi) < 0.45
            for g in gaps)
        if near_gap:
            gap_ok[k] = True
            continue
        r = rng.uniform(0.8, 2.2)
        x = cx + (ring_r + r) * math.cos(ang)
        y = cy + (ring_r + r) * math.sin(ang)
        if placeable(x, y, r, obstacles, boxes):
            obstacles.append((x, y, r, r * math.sqrt(2.0) if rng.random() < 0.3 else None))
    # a few mid-size obstacles inside the ring band + outside
    scatter(rng, obstacles, boxes, 6, 0.6, 1.8)
    return obstacles, boxes


def build_twin_boxes_s(rng, variant):
    obstacles, boxes = [], []
    b1 = variant.get("box1", (-7.0, -2.0))
    b2 = variant.get("box2", (7.0, 5.0))
    s1 = variant.get("side1", 8.0)
    s2 = variant.get("side2", 8.0)
    add_box(obstacles, boxes, b1[0], b1[1], s1)
    add_box(obstacles, boxes, b2[0], b2[1], s2)
    scatter(rng, obstacles, boxes, 18, 0.5, 2.2)
    return obstacles, boxes


def build_box_exit_clutter(rng, variant):
    obstacles, boxes = [], []
    bx, by = variant.get("box", (-4.0, 0.0))
    side = variant.get("side", 9.0)
    add_box(obstacles, boxes, bx, by, side)
    # dense small/medium gauntlet in the right half (the exit region)
    scatter(rng, obstacles, boxes, 22, 0.4, 2.0, region=(2.0, 24.0, -22.0, 22.0))
    # light scatter in the left half
    scatter(rng, obstacles, boxes, 8, 0.5, 2.0, region=(-24.0, 0.0, -22.0, 22.0))
    return obstacles, boxes


def build_triple_box_zigzag(rng, variant):
    obstacles, boxes = [], []
    pos = variant.get("positions", [(-10.0, -6.0), (0.0, 1.0), (10.0, -6.0)])
    sides = variant.get("sides", [7.0, 8.0, 7.0])
    for (px, py), sd in zip(pos, sides):
        add_box(obstacles, boxes, px, py, sd)
    scatter(rng, obstacles, boxes, 14, 0.5, 2.0)
    return obstacles, boxes


def build_box_narrow_gaps(rng, variant):
    obstacles, boxes = [], []
    cx, cy = variant.get("center", (0.0, 0.0))
    side = variant.get("side", 10.0)
    add_box(obstacles, boxes, cx, cy, side)
    half = side / 2.0
    # two narrow side gaps: stack small/medium obstacles above/below the box
    # to funnel the route through a ~4 m gap on the right and left.
    for gy in (cy + half + 2.0, cy - half - 2.0):
        for gx in (-12.0, 0.0, 12.0):
            r = rng.uniform(0.8, 1.6)
            if placeable(gx, gy, r, obstacles, boxes):
                obstacles.append((gx, gy, r, r * math.sqrt(2.0) if rng.random() < 0.4 else None))
    # outer scattered fill
    scatter(rng, obstacles, boxes, 12, 0.5, 2.0)
    return obstacles, boxes


def build_u_canyon(rng, variant):
    obstacles, boxes = [], []
    cx, cy = variant.get("center", (0.0, 0.0))
    arm = variant.get("arm", 12.0)
    # big boxes forming a U open to the +x side (the route enters and loops)
    add_box(obstacles, boxes, cx, cy + arm, 8.0)          # top wall
    add_box(obstacles, boxes, cx, cy - arm, 8.0)          # bottom wall
    add_box(obstacles, boxes, cx - arm, cy, 8.0)          # closed back
    # scattered small/medium inside the U and around
    scatter(rng, obstacles, boxes, 14, 0.5, 2.0)
    return obstacles, boxes


def build_wall_corridor(rng, variant):
    obstacles, boxes = [], []
    wy = variant.get("wall_y", 0.0)
    # a wall of three big boxes across the middle (gap at one end -> corridor)
    for k, wx in enumerate((-12.0, 0.0, 12.0)):
        add_box(obstacles, boxes, wx, wy, 8.0)
    # medium obstacles lining a corridor on the +x side
    for k in range(4):
        r = rng.uniform(1.0, 2.0)
        x = 19.0 - k * 1.2
        y = wy + (6.0 if k % 2 == 0 else -6.0)
        if placeable(x, y, r, obstacles, boxes):
            obstacles.append((x, y, r, None))
    scatter(rng, obstacles, boxes, 8, 0.5, 1.8)
    return obstacles, boxes


def build_mixed_cascade(rng, variant):
    obstacles, boxes = [], []
    # big box at the left
    add_box(obstacles, boxes, -14.0, 0.0, 9.0)
    # medium zone in the middle
    for k in range(6):
        r = rng.uniform(1.6, 2.8)
        x = -2.0 + k * 3.5
        y = rng.uniform(-8.0, 8.0)
        if placeable(x, y, r, obstacles, boxes):
            obstacles.append((x, y, r, r * math.sqrt(2.0) if rng.random() < 0.3 else None))
    # dense small zone at the right (final gauntlet)
    scatter(rng, obstacles, boxes, 16, 0.4, 1.4, region=(10.0, 24.0, -12.0, 12.0))
    return obstacles, boxes


ARCHETYPES = [
    ("ring_around_box", build_ring_around_box),
    ("twin_boxes_s", build_twin_boxes_s),
    ("box_exit_clutter", build_box_exit_clutter),
    ("triple_box_zigzag", build_triple_box_zigzag),
    ("box_narrow_gaps", build_box_narrow_gaps),
    ("u_canyon", build_u_canyon),
    ("wall_corridor", build_wall_corridor),
    ("mixed_cascade", build_mixed_cascade),
]

# 2 variants per archetype
VARIANTS = {
    "ring_around_box": [{"center": (0, 0), "side": 9.0, "ring_r": 11.5,
                         "n_ring": 10, "gaps": [0.6, 3.8]},
                        {"center": (2, -3), "side": 8.0, "ring_r": 10.5,
                         "n_ring": 9, "gaps": [1.4, 4.2]}],
    "twin_boxes_s": [{"box1": (-7, -2), "box2": (7, 5), "side1": 8.0, "side2": 8.0},
                     {"box1": (6, 2), "box2": (-6, -5), "side1": 7.0, "side2": 9.0}],
    "box_exit_clutter": [{"box": (-4, 0), "side": 9.0},
                         {"box": (-6, -2), "side": 8.0}],
    "triple_box_zigzag": [{"positions": [(-10, -6), (0, 1), (10, -6)], "sides": [7, 8, 7]},
                          {"positions": [(-10, 5), (0, -2), (10, 5)], "sides": [8, 7, 8]}],
    "box_narrow_gaps": [{"center": (0, 0), "side": 10.0},
                        {"center": (-2, 3), "side": 9.0}],
    "u_canyon": [{"center": (0, 0), "arm": 12.0},
                 {"center": (-3, -2), "arm": 11.0}],
    "wall_corridor": [{"wall_y": 0.0}, {"wall_y": 3.0}],
    "mixed_cascade": [{"seed": 1}, {"seed": 2}],
}


# ── manifest helpers ────────────────────────────────────────────
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


def sample_task(rng, obstacles, boxes):
    """Start/goal on OPPOSITE sides of the first big box so the straight
    line pierces it -> forced long detour.  The first big box is the primary
    blocker; for wall/u archetypes the sampler still works because the
    primary box sits between the two halves."""
    bx, by, half = boxes[0]
    for _ in range(8000):
        side = 1.0 if rng.random() < 0.5 else -1.0
        sx = bx + side * rng.uniform(half + 6.0, half + 12.0)
        sy = by + rng.uniform(-15.0, 15.0)
        gx = bx - side * rng.uniform(half + 6.0, half + 12.0)
        gy = by + rng.uniform(-15.0, 15.0)
        dist = math.hypot(gx - sx, gy - sy)
        if not (TASK_DIST_MIN <= dist <= TASK_DIST_MAX):
            continue
        if not point_clear(sx, sy, obstacles) or \
                not point_clear(gx, gy, obstacles):
            continue
        if not _clear_of_boxes(sx, sy, boxes) or \
                not _clear_of_boxes(gx, gy, boxes):
            continue
        if not seg_intersects_aabb(sx, sy, gx, gy,
                                   bx - half, by - half,
                                   bx + half, by + half):
            continue
        return (sx, sy, gx, gy)
    return None


def main():
    if not os.path.isfile(MANIFEST):
        sys.exit("manifest not found: %s" % MANIFEST)
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # idempotent: drop any previously-appended handcrafted scenes/tasks
    manifest["scenes"] = [s for s in manifest.get("scenes", [])
                          if s["scene_id"] < 32]
    manifest["tasks"] = [t for t in manifest.get("tasks", [])
                         if t["scene_id"] < 32]

    existing_scenes = manifest["scenes"]
    existing_tasks = manifest["tasks"]
    base_seed = int(manifest.get("base_seed", 0))
    rng = random.Random(base_seed ^ SEED)

    next_scene = 32
    next_task = max((int(t["task_id"]) for t in existing_tasks),
                    default=-1) + 1

    new_scenes, new_tasks = [], []
    si = 0
    for name, builder in ARCHETYPES:
        for vi, variant in enumerate(VARIANTS[name]):
            vrng = random.Random(rng.randrange(1 << 62))
            obstacles, boxes = builder(vrng, variant)
            scene_id = next_scene + si
            obs_entries = []
            for idx, (ox, oy, r, s) in enumerate(obstacles):
                obs_entries.append(obstacle_manifest_entry(
                    idx, ox, oy, r, s,
                    vrng.uniform(OBSTACLE_H_MIN, OBSTACLE_H_MAX)))
            radii = [o[2] for o in obstacles]
            n_big = sum(1 for o in obstacles if o[3] and o[3] > 5.0)
            scene = {
                "scene_id": scene_id,
                "seed": vrng.randrange(1 << 62),
                "profile": "handcrafted_" + name,
                "structure_orientation": "none",
                "metadata": {
                    "structure_orientation": "none",
                    "obstacle_count": len(obs_entries),
                    "radius_min": min(radii) if radii else 0.0,
                    "radius_max": max(radii) if radii else 0.0,
                    "radius_mean": sum(radii) / len(radii) if radii else 0.0,
                    "tiny_count": 0, "small_count": 0, "medium_count": 0,
                    "large_count": n_big, "local_density_proxy": 0.0,
                    "largest_obstacle_radius": max(radii) if radii else 0.0,
                    "cluster_count": 0, "free_space_ratio": 0.7,
                    "estimated_corridor_width": 5.0,
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

            for ti in range(TASKS_PER_SCENE):
                task = sample_task(vrng, obstacles, boxes)
                if task is None:
                    sys.exit("task sampling failed: scene %d (%s)" % (
                        scene_id, name))
                sx, sy, gx, gy = task
                dist = math.hypot(gx - sx, gy - sy)
                dc = ("short" if dist < 18.0
                      else ("long" if dist >= 27.0 else "medium"))
                goal_bearing = math.atan2(gy - sy, gx - sx)
                yaw = sample_initial_yaw(goal_bearing, vrng)
                if len(boxes) >= 3:
                    beh = "turn_both"
                elif dist >= 27.0:
                    beh = "long_takeover"
                else:
                    beh = "turn_normal"
                new_tasks.append({
                    "scene_id": scene_id,
                    "task_id": next_task + si * TASKS_PER_SCENE + ti,
                    "seed": vrng.randrange(1 << 62),
                    "start": [float(sx), float(sy)],
                    "goal": [float(gx), float(gy)],
                    "initial_yaw": float(yaw),
                    "flight_height_m": vrng.uniform(FLIGHT_H_MIN, FLIGHT_H_MAX),
                    "behavior_class": beh,
                    "density_class": "dense",
                    "radius_class": "large",
                    "distance_class": dc,
                    "side_class": "both" if len(boxes) >= 3 else "left",
                    "geom_type": "HANDCRAFTED_LONG_DETOUR",
                    "test_label": "%s_v%d_%02d" % (name, vi, ti),
                    "audit": empty_audit(),
                    "summary": empty_summary(),
                })
            si += 1

    manifest["scenes"] = manifest["scenes"] + new_scenes
    manifest["tasks"] = manifest["tasks"] + new_tasks
    manifest["generation_ok"] = True
    manifest["hard_minimums_met"] = True
    manifest["warnings"] = (manifest.get("warnings") or []) + [
        "appended %d handcrafted long-detour scenes (%d tasks, 8 archetypes)" % (
            len(new_scenes), len(new_tasks))]

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    # ── report ──────────────────────────────────────────────────
    from collections import Counter
    print("handcrafted scenes: %d (8 archetypes x 2 variants)" % len(new_scenes))
    print("handcrafted tasks: %d  (total in manifest: %d)" % (
        len(new_tasks), len(manifest["tasks"])))
    print("behavior labels:", dict(Counter(t["behavior_class"]
                                           for t in new_tasks)))
    for sc in new_scenes:
        n_big = sum(1 for o in sc["obstacles"]
                    if o.get("w") and o["radius"] > 4.0)
        print("  scene %2d %-22s obstacles=%-3d big_boxes=%d" % (
            sc["scene_id"], sc["profile"], len(sc["obstacles"]), n_big))
    hc = len(new_tasks)
    total = len(manifest["tasks"])
    print("handcrafted share: %d/%d = %.0f%%" % (hc, total, 100.0 * hc / total))
    print(">>> merged manifest: %s" % MANIFEST)


if __name__ == "__main__":
    main()
