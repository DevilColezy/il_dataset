#!/usr/bin/env python3
"""Generate a DENSE multi-scale HIGH-ALTITUDE avoidance TEST blueprint.

Outdoor high-altitude (scene 0 INDUSTRIAL, flight z ~ 15 m) test with ONE
dense 15 x 15 m mixed-scale cylinder forest (radii 0.1 .. 3.0 m, surface gap
>= 1.6 m) and 10 start/goal pairs sampled RANDOMLY INSIDE the forest.

Each task:
  * start/goal lie inside the forest (a blocker within 6 m of the surface)
  * start/goal never intersect an inflated cylinder (radius + 0.9 m)
  * start->goal straight line pierces at least one obstacle core (0.6*r)
  * start/goal distance in [8, 20] m
  * initial yaw is a RANDOM offset from the goal bearing (0..180 deg,
    weighted edges, same sampler as the collector blueprint)

This makes the test genuinely hard: the drone takes off surrounded by
cylinders and must thread a 1.6 m-gap forest.

Usage:
    python3 gen_high_test_blueprint.py
    IL_DATASET_OUTPUT_DIR=/path/to/output python3 gen_high_test_blueprint.py

Writes into IL_DATASET_OUTPUT_DIR (default ~/flightmare_ws/il_data_high_test):
    joint_v2_blueprint_manifest.json   # collection-only blueprint
    config.yaml                        # test config (extends outdoor high-alt
                                       # config, output_dir -> this dir)
"""
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.expanduser(os.environ.get(
    "IL_DATASET_OUTPUT_DIR", "~/flightmare_ws/il_data_high_test"))
MANIFEST = os.path.join(OUT_DIR, "joint_v2_blueprint_manifest.json")
TEST_CONFIG = os.path.join(OUT_DIR, "config.yaml")

# ── forest / task parameters ──────────────────────────────────────
REGION = (-7.5, 7.5, -7.5, 7.5)      # 15 x 15 m forest
SURFACE_GAP_M = 1.6                   # min surface gap between cylinders
RADIUS_MIN, RADIUS_MAX = 0.1, 3.0     # mixed-scale radii
TARGET_CYLINDERS = 18
N_CLUSTERS = 2                        # tight cylinder clusters (composite
                                      # large-scale obstacles)
CLUSTER_RING = 4                      # cylinders around each cluster centre
FOREST_SEED = 20260902
NUM_TASKS = 10
TASK_DIST_MIN, TASK_DIST_MAX = 8.0, 20.0

FLIGHT_HEIGHT_M = 15.0
OBSTACLE_HEIGHT_M = 16.0              # covers the 15 m flight band
EXPERT_REVISION = "r20260829_d435i_fullh89.2_640x360_noise_5m"

# ── initial-yaw sampler (mirrors TaskCandidateGenerator::sampleInitialYaw) ──
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


# ── geometry helpers ──────────────────────────────────────────────
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


def point_clear(x, y, obstacles, inflate=0.9):
    """True if (x,y) is at least `inflate` m outside every obstacle surface
    (boxes use their circumscribed-circle radius)."""
    return all(math.hypot(x - ox, y - oy) > r + inflate
               for (ox, oy, r, _side) in obstacles)


def _placeable(x, y, r, obstacles):
    """Inside region and >= SURFACE_GAP_M surface gap from every obstacle
    (boxes use their circumscribed-circle radius as the surface)."""
    if not (REGION[0] + r + 0.4 <= x <= REGION[1] - r - 0.4 and
            REGION[2] + r + 0.4 <= y <= REGION[3] - r - 0.4):
        return False
    return all(math.hypot(x - ox, y - oy) >= r + orad + SURFACE_GAP_M
               for (ox, oy, orad, _side) in obstacles)


def generate_forest(rng, target=TARGET_CYLINDERS, n_clusters=N_CLUSTERS,
                    cluster_ring=CLUSTER_RING, max_attempts=8000):
    """Mixed forest: a few TIGHT cylinder CLUSTERS plus scattered singles.

    Each cluster is a central cylinder surrounded by `cluster_ring` members at
    exactly SURFACE_GAP_M surface distance — together they read as ONE large
    composite obstacle (a dense block the drone must route around/through), so
    the scene has genuine large-scale structure instead of only scattered
    singles.  Scattered small cylinders then fill the remaining space.
    """
    obstacles = []
    # ── 1) tight cylinder clusters (composite large-scale obstacles) ──
    for _ in range(n_clusters * 300):
        if len(obstacles) >= target:
            break
        cx = rng.uniform(REGION[0] + 3.0, REGION[1] - 3.0)
        cy = rng.uniform(REGION[2] + 3.0, REGION[3] - 3.0)
        # cluster centre must be clear of everything already placed AND stay
        # well separated from other clusters (>= 6.5 m centre gap) so every
        # cluster is an ISOLATED composite obstacle with open space between.
        if any(math.hypot(cx - ox, cy - oy) < 6.5 + orad
               for (ox, oy, orad, _s) in obstacles):
            continue
        r0 = rng.uniform(0.8, 1.2)   # centre obstacle (medium core)
        # ~half of the cluster cores are square BOXES (side = r0*sqrt(2),
        # circumscribed-circle radius == r0): a sharp-edged large blocker.
        side0 = r0 * math.sqrt(2.0) if rng.random() < 0.5 else None
        if not _placeable(cx, cy, r0, obstacles):
            continue
        obstacles.append((cx, cy, r0, side0))
        ring = 0
        for k in range(cluster_ring):
            rr = rng.uniform(0.4, 0.9)   # ring members (medium cylinders)
            ang = 2.0 * math.pi * k / cluster_ring + rng.uniform(-0.25, 0.25)
            dist = r0 + SURFACE_GAP_M + rr
            x = cx + dist * math.cos(ang)
            y = cy + dist * math.sin(ang)
            if _placeable(x, y, rr, obstacles):
                obstacles.append((x, y, rr, None))
                ring += 1
        if ring < 3:
            # did not actually form a cluster: drop the centre again
            obstacles.pop()
    # ── 2) scattered fill cylinders ──────────────────────────────
    for _ in range(max_attempts):
        if len(obstacles) >= target:
            break
        r = rng.uniform(RADIUS_MIN, RADIUS_MAX)
        # ~30% of the scattered obstacles are square boxes
        side = r * math.sqrt(2.0) if rng.random() < 0.3 else None
        x = rng.uniform(REGION[0] + r + 0.4, REGION[1] - r - 0.4)
        y = rng.uniform(REGION[2] + r + 0.4, REGION[3] - r - 0.4)
        if _placeable(x, y, r, obstacles):
            obstacles.append((x, y, r, side))
    return obstacles


def nearest_surface(x, y, obstacles):
    """Distance from (x,y) to the nearest obstacle surface (boxes use their
    circumscribed-circle radius)."""
    if not obstacles:
        return float("inf")
    return min(math.hypot(x - ox, y - oy) - r for (ox, oy, r, _side) in obstacles)


def sample_task(rng, obstacles):
    """Random start/goal INSIDE the forest with the difficulty contract."""
    for _ in range(6000):
        sx = rng.uniform(REGION[0] + 0.8, REGION[1] - 0.8)
        sy = rng.uniform(REGION[2] + 0.8, REGION[3] - 0.8)
        gx = rng.uniform(REGION[0] + 0.8, REGION[1] - 0.8)
        gy = rng.uniform(REGION[2] + 0.8, REGION[3] - 0.8)
        dist = math.hypot(gx - sx, gy - sy)
        if not (TASK_DIST_MIN <= dist <= TASK_DIST_MAX):
            continue
        # endpoints must not sit inside an inflated cylinder
        if not point_clear(sx, sy, obstacles) or \
                not point_clear(gx, gy, obstacles):
            continue
        # endpoints must be INSIDE the forest (a blocker nearby)
        if nearest_surface(sx, sy, obstacles) > 6.0 or \
                nearest_surface(gx, gy, obstacles) > 6.0:
            continue
        # the straight line must pierce at least one obstacle core (the
        # circumscribed circle is the core for a box)
        if not any(line_dist_to_point((sx, sy), (gx, gy), (ox, oy)) < 0.6 * r
                   for (ox, oy, r, _side) in obstacles):
            continue
        return (sx, sy, gx, gy)
    return None


def _obstacles_manifest(obstacles):
    """Obstacle list for the blueprint manifest.  Cylinders carry `radius`;
    square boxes additionally carry `w == h` (the side length) so il_manager
    renders them as Transparen_Cube AABBs.  The circumscribed-circle radius is
    kept in `radius` for collision / spacing statistics."""
    out = []
    for i, (x, y, r, side) in enumerate(obstacles):
        entry = {"id": i, "x": float(x), "y": float(y),
                 "radius": float(r), "height_m": OBSTACLE_HEIGHT_M}
        if side is not None:
            entry["w"] = float(side)
            entry["h"] = float(side)
        out.append(entry)
    return out


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


def build_big_box_scene():
    """Single 10 m circumscribed-radius box with two bypass tasks.

    The box SIDE equals the circumscribed-circle DIAMETER (side = 2*r), so
    the AABB faces sit ON the circle's surface along the x/y axes.  That is
    CRITICAL: the depth camera sees the AABB faces, and truth collision uses
    the circumscribed circle — if the box were smaller (e.g. side = r*sqrt2,
    half-width 0.707r) the expert sees the faces ~2.9 m closer than the
    collision circle and flies into the invisible ring.  With side = 2*r the
    visible face distance == collision distance, so the expert plans and
    detours correctly.  Endpoints use off = radius + surf (6 / 2.5 m outside
    the circle)."""
    radius = 10.0
    side = radius * 2.0
    half = side / 2.0
    obstacles = [(0.0, 0.0, radius, side)]
    scene = {
        "scene_id": 1,
        "seed": 0,
        "profile": "H_big_box_10m",
        "planned_radius_class": "handcrafted",
        "actual_radius_class": "large",
        "actual_min_radius_m": radius,
        "actual_max_radius_m": radius,
        "density_class": "sparse",
        "actual_obstacle_count": len(obstacles),
        "obstacles": _obstacles_manifest(obstacles),
    }
    tasks = []
    for i, (label, surf) in enumerate([
            ("h_bigbox_6m", 6.0), ("h_bigbox_2p5m", 2.5)]):
        # Distance from the CIRCUMSCRIBED circle centre (truth collision uses
        # the circle), so surf is the clearance to the circle surface.
        off = radius + surf
        tasks.append({
            "scene_id": 1,
            "task_id": 100 + i,
            "seed": 0,
            "start": [off, 0.0],
            "goal": [-off, 0.0],
            # FIX (2026-08-29): was pi/2 (90°) = facing +Y, i.e. SIDEWAYS to
            # the -X goal/obstacle — the obstacle sat outside the FOV, depth
            # never saw it (obs_clr=inf) and every detour failed.  Must be pi
            # (180°) = face -X toward the goal.
            "initial_yaw": math.pi,
            "flight_height_m": FLIGHT_HEIGHT_M,
            "behavior_class": "bypass_big_box",
            "density_class": "sparse",
            "radius_class": "large",
            "distance_class": "long",
            "side_class": "none",
            "geom_type": "HANDCRAFTED_BIG_BOX",
            "test_label": label,
            "audit": empty_audit(),
            "summary": empty_summary(),
        })
    return scene, tasks


def main():
    rng = random.Random(FOREST_SEED)
    obstacles = generate_forest(rng)
    if len(obstacles) < 10:
        sys.exit("forest too sparse: %d cylinders" % len(obstacles))

    tasks_raw = []
    for i in range(NUM_TASKS):
        t = sample_task(rng, obstacles)
        if t is None:
            sys.exit("task sampling failed at pair %d" % i)
        tasks_raw.append(t)

    radii = [o[2] for o in obstacles]
    scene = {
        "scene_id": 0,
        "seed": FOREST_SEED,
        "profile": "H_forest_15x15_mixed",
        "planned_radius_class": "handcrafted",
        "actual_radius_class": "mixed",
        "actual_min_radius_m": min(radii),
        "actual_max_radius_m": max(radii),
        "density_class": "dense",
        "actual_obstacle_count": len(obstacles),
        "obstacles": _obstacles_manifest(obstacles),
    }

    tasks_out = []
    for i, (sx, sy, gx, gy) in enumerate(tasks_raw):
        dist = math.hypot(gx - sx, gy - sy)
        dc = ("short" if dist < 18.0 else "medium")
        goal_bearing_expert = math.atan2(gy - sy, gx - sx)
        yaw = sample_initial_yaw(goal_bearing_expert, rng)
        tasks_out.append({
            "scene_id": 0,
            "task_id": i,
            "seed": 0,
            "start": [float(sx), float(sy)],
            "goal": [float(gx), float(gy)],
            "initial_yaw": float(yaw),
            "flight_height_m": FLIGHT_HEIGHT_M,
            "behavior_class": "forest_navigation",
            "density_class": "dense",
            "radius_class": "mixed",
            "distance_class": dc,
            "side_class": "none",
            "geom_type": "HANDCRAFTED_HIGH_AVOID",
            "test_label": "h_forest_%02d" % i,
            "audit": empty_audit(),
            "summary": empty_summary(),
        })

    big_scene, big_tasks = build_big_box_scene()
    manifest = {
        "manifest_kind": "HANDCRAFTED_HIGH_AVOID_TEST",
        "expert_revision": EXPERT_REVISION,
        "base_seed": FOREST_SEED,
        "generation_ok": True,
        "scenes": [scene, big_scene],
        "tasks": tasks_out + big_tasks,
        "preflighted": [],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    outdoor_config = os.path.normpath(os.path.join(
        os.path.dirname(_HERE),
        "config", "il_dataset_outdoor_config.yaml"))
    config_text = (
        "# Auto-generated HIGH-ALTITUDE TEST config (collection-only).\n"
        "# Extends the outdoor high-altitude recipe; only output_dir differs\n"
        "# so the test never touches il_data_d435i_col_high.\n"
        "extends: %s\n"
        "global:\n"
        '  output_dir: "%s"\n'
        "  scene_id: 0   # INDUSTRIAL (outdoor)\n"
        % (outdoor_config, OUT_DIR))
    with open(TEST_CONFIG, "w", encoding="utf-8") as f:
        f.write(config_text)

    # ── report ──────────────────────────────────────────────────
    print("forest: %d cylinders  r=[%.2f, %.2f]  surface_gap>=%.1f m"
          % (len(obstacles), min(radii), max(radii), SURFACE_GAP_M))
    print("tasks: %d  (region 15x15, dist %.0f..%.0f m)"
          % (len(tasks_out), TASK_DIST_MIN, TASK_DIST_MAX))
    for i, t in enumerate(tasks_out):
        sx, sy = t["start"]
        gx, gy = t["goal"]
        pierce = [j for j, (ox, oy, r, _s) in enumerate(obstacles)
                  if line_dist_to_point((sx, sy), (gx, gy), (ox, oy)) < 0.6 * r]
        print("  %02d s=(%5.2f,%5.2f) g=(%5.2f,%5.2f) d=%5.1fm "
              "yaw=%6.2f pierce_cyl=%s" % (
                  i, sx, sy, gx, gy,
                  math.hypot(gx - sx, gy - sy), t["initial_yaw"], pierce))
    print("big_box: 1 x r_circ=10m box (side=2r=%.2fm), tasks=%d"
          % (10.0 * 2.0, len(big_tasks)))
    for t in big_tasks:
        print("  %s s=(%5.2f,%5.2f) g=(%5.2f,%5.2f) d=%.1fm yaw=%.2f" % (
            t["test_label"], t["start"][0], t["start"][1],
            t["goal"][0], t["goal"][1],
            math.hypot(t["goal"][0] - t["start"][0],
                       t["goal"][1] - t["start"][1]), t["initial_yaw"]))
    print(">>> manifest written: %s" % MANIFEST)
    print(">>> test config written: %s" % TEST_CONFIG)


if __name__ == "__main__":
    main()
