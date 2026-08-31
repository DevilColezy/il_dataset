#!/usr/bin/env python3
"""Rollout-v3 test scene blueprint.

Redesigned rollout stack scene set (10 scenes / 10 tasks, one task per scene):

INDOOR (Unity scene 1 WAREHOUSE, flight z = 2.0, region x[-7,10] y[0,30]):
  I_small_field    dense field of SMALL cylinders (r 0.25..0.5) between
                   start/goal — can the drone thread the field?
  I_medium_field   field of MEDIUM cylinders (r 0.8..1.3)
  I_large_field    field of LARGE cylinders (r 2.0) + narrow 1.6 m gaps
  I_mixed_field    MIXED field (r 0.3..2.5)
  I_slalom_field   [self-designed] staggered big cylinders -> S slalom
  I_gate_dense     [self-designed] central big gate + dense small fillers

OUTDOOR (Unity scene 0 INDUSTRIAL, flight z = 2.0, large region):
  O_wall_small     20 m wall + SMALL obstacles in the bypass zone
                   (wall = 5 x r=2.0 tangent cylinders x[-10,10] at y=15)
  O_wall_medium    20 m wall + MEDIUM obstacles in the bypass zone
  O_twin_wall      [self-designed] two staggered 9 m walls -> S detour
  O_wall_mixed     [self-designed] 20 m wall + MIXED bypass obstacles

Contract (mirrors gen_avoid_scenes_4level.py):
  * every task's straight start->goal line provably pierces an obstacle core
  * non-wall obstacle SURFACE gap >= 1.6 m
  * wall cylinders may touch (gap 0) — they form one continuous wall
  * bypass obstacles may hug the wall, but keep >= 1.6 m between each other

Writes the production-schema blueprint to:
    ~/flightmare_ws/il_data_joint_v2/rollout_v3_scenes_manifest.json
(loadable by rollout_stack.py via --scene-set v3)
"""
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.expanduser(os.environ.get(
    "IL_DATASET_OUTPUT_DIR", "~/flightmare_ws/il_data_joint_v2"))
MANIFEST = os.path.join(OUT_DIR, "rollout_v3_scenes_manifest.json")

INDOOR_REGION = {"min_x": -7.0, "max_x": 10.0, "min_y": 0.0, "max_y": 30.0}
OUTDOOR_REGION = {"min_x": -28.0, "max_x": 28.0, "min_y": -28.0, "max_y": 30.0}

# Wall helper: a continuous wall = ODD number of tangent cylinders along x
# at y=wall_y so a straight line down the wall's axis pierces the centre core.
def _wall(centre_x, wall_y, half_len, r):
    """Odd number of r-cylinders tangent along x spanning centre_x±half_len."""
    n = int(round((2.0 * half_len) / (2.0 * r))) + 1
    if n % 2 == 0:
        n += 1  # force a centre cylinder so the axis line pierces a core
    xs = [centre_x + (i - (n - 1) / 2.0) * 2.0 * r for i in range(n)]
    return [(x, wall_y, r) for x in xs]


# Each scene: {"name", "environment", "desc", "walls", "obstacles",
# "tasks"}.  "walls" are continuous walls (cylinders may TOUCH, gap 0) and
# bypass obstacles may HUG a wall (gap < 1.6 allowed); only obstacle-obstacle
# surface gap is enforced at >= MIN_GAP.  task = (sx, sy, gx, gy, label).
SCENES = [
    # ── INDOOR: scale-class obstacle fields ─────────────────────────
    {
        "name": "I_small_field", "environment": "indoor",
        "desc": "室内小尺度障碍区: 起飞→终点间密集小柱(r 0.25-0.5), 测试穿越",
        "walls": [],
        "obstacles": [
            (0, 10, 0.35), (0, 15, 0.3), (0, 20, 0.4),
            (-2.5, 12, 0.3), (2.5, 12, 0.35), (-2.5, 18, 0.35), (2.5, 18, 0.3),
            (-3.8, 15, 0.3), (3.8, 15, 0.35),
        ],
        "tasks": [(0, 3, 0, 25, "I_small_field")],
    },
    {
        "name": "I_medium_field", "environment": "indoor",
        "desc": "室内中尺度障碍区: 起飞→终点间中柱(r 0.8-1.3), 测试穿越",
        "walls": [],
        "obstacles": [
            (0, 11, 1.0), (0, 18, 1.1), (0, 23, 0.9),
            (-2.8, 14, 0.85), (2.8, 14, 0.9), (-3.0, 20, 0.8), (3.0, 20, 0.9),
        ],
        "tasks": [(0, 3, 0, 25, "I_medium_field")],
    },
    {
        "name": "I_large_field", "environment": "indoor",
        "desc": "室内大尺度障碍区: 大柱(r 2.0)交错+1.6m窄缝, 测试绕行",
        "walls": [],
        "obstacles": [
            (-3.5, 11, 2.0), (2.1, 11, 2.0), (-3.5, 17.5, 2.0),
            (2.1, 17.5, 2.0), (0.7, 23, 2.0),
        ],
        "tasks": [(0.7, 3, 0.7, 28, "I_large_field")],
    },
    {
        "name": "I_mixed_field", "environment": "indoor",
        "desc": "室内混合尺度障碍区: r 0.3-2.5 混合, 测试穿越+绕行",
        "walls": [],
        "obstacles": [
            (0, 11, 2.0), (0, 18, 1.5), (0, 24, 0.9),
            (-3, 14, 0.4), (3, 14, 0.35), (-3.5, 20, 0.3), (3.2, 20, 0.4),
        ],
        "tasks": [(0, 3, 0, 27, "I_mixed_field")],
    },
    {
        "name": "I_slalom_field", "environment": "indoor",
        "desc": "自设计-室内蛇形走廊: 大柱错位迫使S形连续绕障",
        "walls": [],
        "obstacles": [
            (0, 10, 1.8), (3.5, 15, 1.6), (-3.5, 20, 1.7), (1.0, 25, 1.5),
        ],
        "tasks": [(0, 3, 0, 28, "I_slalom_field")],
    },
    {
        "name": "I_gate_dense", "environment": "indoor",
        "desc": "自设计-室内大门+密集填充: 中央r2.5大柱堵路+周边小障碍",
        "walls": [],
        "obstacles": [
            (0, 13, 2.5), (0, 19, 1.2), (0, 23, 0.9),
            (-4.5, 13, 0.4), (-5.2, 18, 0.3), (3.2, 18, 0.4),
            (-2.8, 23, 0.3), (2.9, 23, 0.35),
        ],
        "tasks": [(0, 4, 0, 27, "I_gate_dense")],
    },
    # ── OUTDOOR: 10 m-radius BIG BOX bypass ────────────────────────
    # A 10 m-radius cylinder renders as a 20x20 m BOX (Unity "Object"
    # prefab, size 2r x h x 2r) — the same big-box obstacle class the
    # training data contains (e.g. big8 r=8 box), unlike a 20 m linear
    # wall.  The drone must go around the box end; bypass obstacles sit on
    # the detour path (they may hug the box, only >=1.6 m between each
    # ── OUTDOOR: r=15 m cylinder, four bypass-belt classes ─────────
    # Main body: a single r=15 m cylinder centred at (0,8) (Object prefab,
    # 30 m diameter).  Task crosses it along x: (-20,8)->(20,8), so the
    # drone must detour around the right end (x>15).  The bypass belt sits
    # on the detour path with 4 obstacle classes (small/medium/large/mixed);
    # belt obstacles keep >= 1.6 m surface gap between EACH OTHER but may
    # hug the cylinder.
    {
        "name": "O15_small", "environment": "outdoor",
        "desc": "室外r15m圆柱+小尺度绕行带: 绕过圆柱右端, 途中穿小障碍(r 0.3-0.4)",
        "walls": [(0.0, 8.0, 15.0)],
        "obstacles": [(17, 6, 0.35), (19.5, 8, 0.3), (17.5, 12, 0.4),
                      (20, 14, 0.35), (18.5, 16, 0.3)],
        "tasks": [(-20, 8, 21, 8, "O15_small")],
    },
    {
        "name": "O15_medium", "environment": "outdoor",
        "desc": "室外r15m圆柱+中尺度绕行带: 绕过圆柱右端, 途中穿中障碍(r 0.7-0.9)",
        "walls": [(0.0, 8.0, 15.0)],
        "obstacles": [(17, 7, 0.8), (20, 9, 0.7), (17.5, 13, 0.9),
                      (20.5, 15, 0.75), (18.5, 17.5, 0.8)],
        "tasks": [(-20, 8, 21, 8, "O15_medium")],
    },
    {
        "name": "O15_large", "environment": "outdoor",
        "desc": "室外r15m圆柱+大尺度绕行带: 绕过圆柱右端, 途中穿大障碍(r 1.5)",
        "walls": [(0.0, 8.0, 15.0)],
        "obstacles": [(17.5, 8, 1.5), (20.5, 13, 1.5), (18.5, 18, 1.5)],
        "tasks": [(-20, 8, 21, 8, "O15_large")],
    },
    {
        "name": "O15_mixed", "environment": "outdoor",
        "desc": "室外r15m圆柱+混合绕行带: 绕过圆柱右端, 途中穿混合障碍(r 0.4-1.8)",
        "walls": [(0.0, 8.0, 15.0)],
        "obstacles": [(17, 7, 0.4), (20, 9.5, 0.8), (17.0, 13, 1.8),
                      (20.5, 15, 0.5), (18.5, 17.5, 0.8)],
        "tasks": [(-20, 8, 21, 8, "O15_mixed")],
    },
    # ── OUTDOOR: 20 m linear WALL (harder, wall-like shape) ────────
    # A 20 m continuous wall along x centred at (0,8), built from tangent
    # r=0.8 cylinders (training data has no such linear-wall shape, so this
    # is strictly harder than the r=15 cylinder).  Task crosses along the
    # wall axis so the straight line pierces the core; the drone must
    # detour around the right end (x>10).  Bypass belt has 4 classes.
    {
        "name": "Owall_small", "environment": "outdoor",
        "desc": "室外20m长墙(任务垂直穿墙)+左右端小绕行带: 绕过墙端途中穿小障碍(r 0.3-0.4)",
        "walls": _wall(0.0, 8.0, 10.0, 0.8),
        "obstacles": [(11, -1, 0.35), (13.5, 3, 0.3), (12, 7, 0.4),
                      (15.5, 10, 0.35), (13.5, 14, 0.3), (17, 17, 0.4),
                      (15.5, 20, 0.35),
                      (-11, -1, 0.35), (-13.5, 3, 0.3), (-12, 7, 0.4),
                      (-15.5, 10, 0.35), (-13.5, 14, 0.3), (-17, 17, 0.4),
                      (-15.5, 20, 0.35)],
        "tasks": [(0, -6, 0, 22, "Owall_small")],
    },
    {
        "name": "Owall_medium", "environment": "outdoor",
        "desc": "室外20m长墙(任务垂直穿墙)+左右端中绕行带: 绕过墙端途中穿中障碍(r 0.7-0.9)",
        "walls": _wall(0.0, 8.0, 10.0, 0.8),
        "obstacles": [(11, 0, 0.8), (14, 4, 0.7), (12.5, 8.5, 0.9),
                      (16, 12, 0.75), (14, 16.5, 0.8), (17.5, 19.5, 0.7),
                      (-11, 0, 0.8), (-14, 4, 0.7), (-12.5, 8.5, 0.9),
                      (-16, 12, 0.75), (-14, 16.5, 0.8), (-17.5, 19.5, 0.7)],
        "tasks": [(0, -6, 0, 22, "Owall_medium")],
    },
    {
        "name": "Owall_large", "environment": "outdoor",
        "desc": "室外20m长墙(任务垂直穿墙)+左右端大绕行带: 绕过墙端途中穿大障碍(r 1.5)",
        "walls": _wall(0.0, 8.0, 10.0, 0.8),
        "obstacles": [(11.5, 2, 1.5), (15, 8, 1.5), (13.5, 15, 1.5),
                      (17, 20, 1.5),
                      (-11.5, 2, 1.5), (-15, 8, 1.5), (-13.5, 15, 1.5),
                      (-17, 20, 1.5)],
        "tasks": [(0, -6, 0, 22, "Owall_large")],
    },
    {
        "name": "Owall_mixed", "environment": "outdoor",
        "desc": "室外20m长墙(任务垂直穿墙)+左右端混合绕行带: 绕过墙端途中穿混合障碍(r 0.4-1.8)",
        "walls": _wall(0.0, 8.0, 10.0, 0.8),
        "obstacles": [(11, -1, 0.4), (13.5, 3.5, 0.8), (14.5, 9, 1.8),
                      (16, 13, 0.5), (14, 17.5, 0.8), (17.5, 20.5, 0.5),
                      (-11, -1, 0.4), (-13.5, 3.5, 0.8), (-14.5, 9, 1.8),
                      (-16, 13, 0.5), (-14, 17.5, 0.8), (-17.5, 20.5, 0.5)],
        "tasks": [(0, -6, 0, 22, "Owall_mixed")],
    },
    # ── OUTDOOR: benchmark high_altitude_side_band 复刻 ──────────
    # 主圆柱 r10/15/20 居中, 两列 r1.0 圆柱填满两侧绕行带 (1.8m 表面间距,
    # 内列贴主圆柱侧面). 起点距圆柱面仅 6m, 一起飞即面对密集障碍群.
    {
        "name": "Osideband_r10", "environment": "outdoor",
        "desc": "复刻benchmark side_band: r10主圆柱+两列r1.0密集侧带(1.8m间距), 起点距圆柱面6m",
        "walls": [(0.0, 0.0, 10.0)],
        "obstacles": [
            (-11.0, -7.6, 1.0), (-11.0, -3.8, 1.0), (-11.0, 0.0, 1.0),
            (-11.0, 3.8, 1.0), (-11.0, 7.6, 1.0),
            (-14.8, -7.6, 1.0), (-14.8, -3.8, 1.0), (-14.8, 0.0, 1.0),
            (-14.8, 3.8, 1.0), (-14.8, 7.6, 1.0),
            (11.0, -7.6, 1.0), (11.0, -3.8, 1.0), (11.0, 0.0, 1.0),
            (11.0, 3.8, 1.0), (11.0, 7.6, 1.0),
            (14.8, -7.6, 1.0), (14.8, -3.8, 1.0), (14.8, 0.0, 1.0),
            (14.8, 3.8, 1.0), (14.8, 7.6, 1.0),
        ],
        "tasks": [(0, -16, 0, 16, "Osideband_r10")],
    },
    {
        "name": "Osideband_r15", "environment": "outdoor",
        "desc": "复刻benchmark side_band: r15主圆柱+两列r1.0密集侧带(1.8m间距), 起点距圆柱面6m",
        "walls": [(0.0, 0.0, 15.0)],
        "obstacles": [
            (-16.0, -11.4, 1.0), (-16.0, -7.6, 1.0), (-16.0, -3.8, 1.0),
            (-16.0, 0.0, 1.0), (-16.0, 3.8, 1.0), (-16.0, 7.6, 1.0),
            (-16.0, 11.4, 1.0),
            (-19.8, -11.4, 1.0), (-19.8, -7.6, 1.0), (-19.8, -3.8, 1.0),
            (-19.8, 0.0, 1.0), (-19.8, 3.8, 1.0), (-19.8, 7.6, 1.0),
            (-19.8, 11.4, 1.0),
            (16.0, -11.4, 1.0), (16.0, -7.6, 1.0), (16.0, -3.8, 1.0),
            (16.0, 0.0, 1.0), (16.0, 3.8, 1.0), (16.0, 7.6, 1.0),
            (16.0, 11.4, 1.0),
            (19.8, -11.4, 1.0), (19.8, -7.6, 1.0), (19.8, -3.8, 1.0),
            (19.8, 0.0, 1.0), (19.8, 3.8, 1.0), (19.8, 7.6, 1.0),
            (19.8, 11.4, 1.0),
        ],
        "tasks": [(0, -21, 0, 21, "Osideband_r15")],
    },
    {
        "name": "Osideband_r20", "environment": "outdoor",
        "desc": "复刻benchmark side_band: r20主圆柱+两列r1.0密集侧带(1.8m间距), 起点距圆柱面6m",
        "walls": [(0.0, 0.0, 20.0)],
        "obstacles": [
            (-21.0, -19.0, 1.0), (-21.0, -15.2, 1.0), (-21.0, -11.4, 1.0),
            (-21.0, -7.6, 1.0), (-21.0, -3.8, 1.0), (-21.0, 0.0, 1.0),
            (-21.0, 3.8, 1.0), (-21.0, 7.6, 1.0), (-21.0, 11.4, 1.0),
            (-21.0, 15.2, 1.0), (-21.0, 19.0, 1.0),
            (-24.8, -19.0, 1.0), (-24.8, -15.2, 1.0), (-24.8, -11.4, 1.0),
            (-24.8, -7.6, 1.0), (-24.8, -3.8, 1.0), (-24.8, 0.0, 1.0),
            (-24.8, 3.8, 1.0), (-24.8, 7.6, 1.0), (-24.8, 11.4, 1.0),
            (-24.8, 15.2, 1.0), (-24.8, 19.0, 1.0),
            (21.0, -19.0, 1.0), (21.0, -15.2, 1.0), (21.0, -11.4, 1.0),
            (21.0, -7.6, 1.0), (21.0, -3.8, 1.0), (21.0, 0.0, 1.0),
            (21.0, 3.8, 1.0), (21.0, 7.6, 1.0), (21.0, 11.4, 1.0),
            (21.0, 15.2, 1.0), (21.0, 19.0, 1.0),
            (24.8, -19.0, 1.0), (24.8, -15.2, 1.0), (24.8, -11.4, 1.0),
            (24.8, -7.6, 1.0), (24.8, -3.8, 1.0), (24.8, 0.0, 1.0),
            (24.8, 3.8, 1.0), (24.8, 7.6, 1.0), (24.8, 11.4, 1.0),
            (24.8, 15.2, 1.0), (24.8, 19.0, 1.0),
        ],
        "tasks": [(0, -26, 0, 26, "Osideband_r20")],
    },
]

MIN_GAP = 1.6          # non-wall obstacle surface gap
WALL_TOUCH = 0.1       # obstacles with surface gap below this are "one wall"


def line_dist_to_point(a, b, p):
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    L2 = abx * abx + aby * aby
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _is_box(o):
    """Obstacle tuple is a box: (x, y, 0, half_w, half_h)."""
    return len(o) >= 5 and o[3] > 0.0 and o[4] > 0.0


def _obs_gap(o1, o2):
    """Surface gap between two obstacles (cylinder (x,y,r) or box)."""
    if _is_box(o1) and _is_box(o2):
        gx = abs(o1[0] - o2[0]) - (o1[3] + o2[3])
        gy = abs(o1[1] - o2[1]) - (o1[4] + o2[4])
        if gx <= 0 and gy <= 0:
            return -math.hypot(gx, gy)
        if gx <= 0:
            return gy
        if gy <= 0:
            return gx
        return math.hypot(gx, gy)
    if _is_box(o1) or _is_box(o2):
        box, cyl = (o1, o2) if _is_box(o1) else (o2, o1)
        dx = max(abs(box[0] - cyl[0]) - box[3], 0.0)
        dy = max(abs(box[1] - cyl[1]) - box[4], 0.0)
        return max(math.hypot(dx, dy) - cyl[2], 0.0)
    d = math.hypot(o1[0] - o2[0], o1[1] - o2[1])
    return d - o1[2] - o2[2]


def _point_obs_dist(px, py, o):
    """Distance from point to obstacle surface."""
    if _is_box(o):
        dx = max(abs(px - o[0]) - o[3], 0.0)
        dy = max(abs(py - o[1]) - o[4], 0.0)
        return math.hypot(dx, dy)
    return max(math.hypot(px - o[0], py - o[1]) - o[2], 0.0)


def _line_pierces(sx, sy, gx, gy, o):
    """Straight segment start->goal pierces an obstacle CORE."""
    if _is_box(o):
        # Slab test: segment intersects the box interior.
        x0, x1 = o[0] - o[3], o[0] + o[3]
        y0, y1 = o[1] - o[4], o[1] + o[4]
        dx, dy = gx - sx, gy - sy
        tmin, tmax = 0.0, 1.0
        if abs(dx) < 1e-12:
            if sx < x0 or sx > x1:
                return False
        else:
            t1, t2 = (x0 - sx) / dx, (x1 - sx) / dx
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
            if tmin > tmax:
                return False
        if abs(dy) < 1e-12:
            if sy < y0 or sy > y1:
                return False
        else:
            t1, t2 = (y0 - sy) / dy, (y1 - sy) / dy
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
            if tmin > tmax:
                return False
        return True
    return line_dist_to_point((sx, sy), (gx, gy), (o[0], o[1])) < 0.6 * o[2]


def verify(scenes):
    """Contract: in-region, line pierces a core, obstacle-obstacle gap
    >= MIN_GAP; wall cylinders may touch; obstacles may hug walls."""
    problems = []
    total = 0
    for sc in scenes:
        region = INDOOR_REGION if sc["environment"] == "indoor" \
            else OUTDOOR_REGION
        walls = list(sc.get("walls", []))
        obs = list(sc.get("obstacles", []))
        # wall cylinders may touch each other (continuous wall), so only
        # check obstacle-obstacle surface gaps (>= MIN_GAP).
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                gap = _obs_gap(obs[i], obs[j])
                if gap + 1e-9 < MIN_GAP:
                    problems.append("%s: obstacle pair %d/%d gap %.2f < %.1f"
                                    % (sc["name"], i, j, gap, MIN_GAP))
        all_obs = walls + obs
        for (sx, sy, gx, gy, label) in sc["tasks"]:
            total += 1
            for (nx, ny) in ((sx, sy), (gx, gy)):
                if not (region["min_x"] + 0.6 <= nx <= region["max_x"] - 0.6
                        and region["min_y"] + 0.6 <= ny <= region["max_y"] - 0.6):
                    problems.append("%s/%s: point (%g,%g) out of region"
                                    % (sc["name"], label, nx, ny))
            d = math.hypot(gx - sx, gy - sy)
            if not (4.0 <= d <= 55.0):
                problems.append("%s/%s: distance %g outside 4..55 m"
                                % (sc["name"], label, d))
            pierced = any(_line_pierces(sx, sy, gx, gy, o) for o in all_obs)
            if not pierced:
                problems.append("%s/%s: line does NOT pierce a core"
                                % (sc["name"], label))
            for (nx, ny) in ((sx, sy), (gx, gy)):
                for o in all_obs:
                    if _point_obs_dist(nx, ny, o) < 0.6:
                        problems.append("%s/%s: endpoint (%.1f,%.1f) inside "
                                        "inflated obstacle"
                                        % (sc["name"], label, nx, ny))
    return total, problems


def main():
    total, problems = verify(SCENES)
    print("rollout-v3 scenes=%d tasks=%d  verification problems=%d"
          % (len(SCENES), total, len(problems)))
    for p in problems:
        print("  PROBLEM: %s" % p)
    if problems:
        sys.exit(1)

    scenes_out = []
    tasks_out = []
    scene_id = 0
    task_id = 0
    for sc in SCENES:
        walls = list(sc.get("walls", []))
        obs = list(sc.get("obstacles", []))
        all_obs = walls + obs
        rads = [o[2] for o in all_obs]
        if sc["name"].startswith("I_"):
            radius_class = ("small" if "small" in sc["name"] else
                            "medium" if "medium" in sc["name"] else
                            "large" if "large" in sc["name"] else "mixed")
        else:
            radius_class = ("small" if "small" in sc["name"] else
                            "medium" if "medium" in sc["name"] else
                            "large" if "large" in sc["name"] else "mixed")
        scenes_out.append({
            "scene_id": scene_id,
            "environment": sc["environment"],
            "unity_scene_id": 1 if sc["environment"] == "indoor" else 0,
            "profile": sc["name"],
            "desc": sc["desc"],
            "actual_radius_class": radius_class,
            "actual_min_radius_m": min(rads),
            "actual_max_radius_m": max(rads),
            "actual_obstacle_count": len(all_obs),
            "wall_count": len(walls),
            "obstacles": [
                ({"id": i, "x": float(o[0]), "y": float(o[1]),
                  "radius": float(o[2]),
                  "height_m": 18.0 if sc["environment"] == "outdoor" else 6.0,
                  "w": 2.0 * float(o[3]), "h": 2.0 * float(o[4]),
                  "wall": i < len(walls)}
                 if _is_box(o) else
                 {"id": i, "x": float(o[0]), "y": float(o[1]),
                  "radius": float(o[2]),
                  "height_m": 18.0 if sc["environment"] == "outdoor" else 6.0,
                  "wall": i < len(walls)})
                for i, o in enumerate(all_obs)],
        })
        rng = random.Random(20260830 + scene_id * 7919)
        for (sx, sy, gx, gy, label) in sc["tasks"]:
            dx, dy = gx - sx, gy - sy
            dist = math.hypot(dx, dy)
            dc = "short" if dist < 8.0 else ("medium" if dist < 16.0
                                             else "long")
            goal_bearing = math.atan2(dy, dx)
            tasks_out.append({
                "scene_id": scene_id,
                "task_id": task_id,
                "start": [float(sx), float(sy)],
                "goal": [float(gx), float(gy)],
                "initial_yaw": math.atan2(dy, dx) - math.pi / 2.0,
                "flight_height_m": 2.0,
                "behavior_class": "local_avoidance",
                "radius_class": radius_class,
                "distance_class": dc,
                "geom_type": "HANDCRAFTED_ROLLOUT_V3",
                "test_label": label,
            })
            task_id += 1
        scene_id += 1

    manifest = {
        "manifest_kind": "HANDCRAFTED_ROLLOUT_V3",
        "min_surface_gap_m": MIN_GAP,
        "wall_touch_tolerance_m": WALL_TOUCH,
        "generation_ok": True,
        "scenes": scenes_out,
        "tasks": tasks_out,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(">>> blueprint written: %s  scenes=%d tasks=%d"
          % (MANIFEST, len(scenes_out), len(tasks_out)))
    for sc in scenes_out:
        n = sum(1 for t in tasks_out if t["scene_id"] == sc["scene_id"])
        print("  %-6s %-14s cyl=%2d task=%d r=[%.2f, %.2f]  %s"
              % (sc["environment"], sc["profile"], sc["actual_obstacle_count"],
                 n, sc["actual_min_radius_m"], sc["actual_max_radius_m"],
                 sc["desc"]))


if __name__ == "__main__":
    main()
