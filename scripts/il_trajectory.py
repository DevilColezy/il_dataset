#!/usr/bin/env python3
"""
Global Path Planning Module  —  A* with shortcut, NO full-trajectory smoothing.

=== v5: Receding-Horizon Refactor ===
This module now ONLY provides:
  - Grid utilities (world↔grid, ESDF lookup, coarse ESDF)
  - A* planner (26-neighbour, corner-cutting prevention)
  - Greedy shortcut / string pulling with adaptive ESDF segment check
  - GlobalPathPlanner: plan_global() returns raw A* + shortcut global path

The following functions are DEPRECATED and MUST NOT be called by the new pipeline:
  - smooth_trajectory
  - smooth_position_path
  - resample_path (as full-trajectory step)
  - time_parameterize
  - sample_trajectory
  - generate_controls
  - validate_dynamics

They are kept only for backward compatibility and debugging.

All coordinates are ROS world frame (x-fwd, y-left, z-up).
"""

from __future__ import print_function, division

import math, heapq, time
import numpy as np

import rospy


# ============================================================================
#  0.  Constants & conventions
# ============================================================================

# Grid index convention:
#   grid[i, j, k] corresponds to the VOXEL CENTER at world position:
#     x = origin[0] + (i + 0.5) * resolution
#     y = origin[1] + (j + 0.5) * resolution
#     z = origin[2] + (k + 0.5) * resolution
#
#   world → grid conversion uses floor:
#     i = floor((x - origin[0]) / resolution)
#   This is monotonic across zero and consistent for negative coordinates.

# 26-neighbor offsets (including diagonals)
_NEIGHBOURS_26 = [(dx, dy, dz)
                  for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                  if not (dx == 0 and dy == 0 and dz == 0)]


def _diagonal_heuristic_3d(node, goal, resolution):
    """Exact obstacle-free cost on a 26-connected 3-D grid.

    This is admissible and consistent, but tighter than Euclidean distance.
    Therefore epsilon=1 still returns an optimal grid path while expanding
    substantially fewer nearly equivalent nodes.
    """
    a, b, c = sorted((abs(node[0] - goal[0]),
                      abs(node[1] - goal[1]),
                      abs(node[2] - goal[2])))
    return resolution * (math.sqrt(3.0) * a
                         + math.sqrt(2.0) * (b - a)
                         + (c - b))


# ============================================================================
#  1.  Grid utilities
# ============================================================================

def _w2g(pos, origin, resolution):
    """World → grid index using floor (safe for negative coordinates)."""
    ox, oy, oz = origin
    inv = 1.0 / resolution
    return (int(math.floor((pos[0] - ox) * inv)),
            int(math.floor((pos[1] - oy) * inv)),
            int(math.floor((pos[2] - oz) * inv)))


def _g2w(idx, origin, resolution):
    """Grid index → voxel CENTER world position."""
    ox, oy, oz = origin
    return (ox + (idx[0] + 0.5) * resolution,
            oy + (idx[1] + 0.5) * resolution,
            oz + (idx[2] + 0.5) * resolution)


def _in_bounds(idx, shape):
    return (0 <= idx[0] < shape[0] and
            0 <= idx[1] < shape[1] and
            0 <= idx[2] < shape[2])


def _esdf_at_grid(esdf, idx):
    if _in_bounds(idx, esdf.shape):
        return float(esdf[idx[0], idx[1], idx[2]])
    return -1.0  # outside = obstacle


def _is_free(esdf, idx, min_clearance=0.0):
    return _esdf_at_grid(esdf, idx) > min_clearance


def _esdf_at_world(esdf, origin, resolution, pos):
    """ESDF value at world position (nearest-neighbour, for A* use)."""
    gx, gy, gz = esdf.shape
    idx = _w2g(pos, origin, resolution)
    idx = (max(0, min(gx - 1, idx[0])),
           max(0, min(gy - 1, idx[1])),
           max(0, min(gz - 1, idx[2])))
    return float(esdf[idx[0], idx[1], idx[2]])


def _trilinear_esdf(esdf, origin, resolution, pos):
    """Trilinear interpolated ESDF value at world position."""
    ox, oy, oz = origin
    inv = 1.0 / resolution
    gx, gy, gz = esdf.shape
    
    # ESDF samples live at voxel centres: origin + (index + 0.5) * res.
    # Convert world coordinates into that centre-indexed lattice before
    # interpolation.  Without the -0.5 offset every query was displaced by
    # half a voxel in x/y/z relative to A* and the C++ planner.
    gx_f = (pos[0] - ox) * inv - 0.5
    gy_f = (pos[1] - oy) * inv - 0.5
    gz_f = (pos[2] - oz) * inv - 0.5
    
    if (gx_f < -0.5 or gx_f > gx - 0.5 or
        gy_f < -0.5 or gy_f > gy - 0.5 or
        gz_f < -0.5 or gz_f > gz - 0.5):
        return -1.0  # outside map
    
    ix0 = int(math.floor(gx_f)); ix1 = ix0 + 1
    iy0 = int(math.floor(gy_f)); iy1 = iy0 + 1
    iz0 = int(math.floor(gz_f)); iz1 = iz0 + 1
    
    ix0 = max(0, min(gx - 1, ix0)); ix1 = max(0, min(gx - 1, ix1))
    iy0 = max(0, min(gy - 1, iy0)); iy1 = max(0, min(gy - 1, iy1))
    iz0 = max(0, min(gz - 1, iz0)); iz1 = max(0, min(gz - 1, iz1))
    
    wx = gx_f - ix0; wy = gy_f - iy0; wz = gz_f - iz0
    
    c000 = float(esdf[ix0, iy0, iz0])
    c100 = float(esdf[ix1, iy0, iz0])
    c010 = float(esdf[ix0, iy1, iz0])
    c110 = float(esdf[ix1, iy1, iz0])
    c001 = float(esdf[ix0, iy0, iz1])
    c101 = float(esdf[ix1, iy0, iz1])
    c011 = float(esdf[ix0, iy1, iz1])
    c111 = float(esdf[ix1, iy1, iz1])
    
    c00 = c000 * (1 - wx) + c100 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c11 = c011 * (1 - wx) + c111 * wx
    
    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy
    
    return c0 * (1 - wz) + c1 * wz


# ============================================================================
#  2.  Diagonal corner-cutting prevention
# ============================================================================

def _enumerate_diagonal_corners(i0, i1):
    """Enumerate intermediate grid cells crossed by a diagonal step.

    For a step from grid index i0 to i1, returns the set of all
    grid cells whose volume is intersected by the line segment.

    If (dx, dy, dz) is a pure axis-aligned step (only one component non-zero),
    returns empty set (no corners to check).

    If it's a face diagonal (2 components non-zero), returns the corner cells.
    If it's a space diagonal (3 components non-zero), returns all intermediate
    cells that must also be free to prevent corner-cutting.
    """
    dx = i1[0] - i0[0]
    dy = i1[1] - i0[1]
    dz = i1[2] - i0[2]

    nz = (1 if dx != 0 else 0) + (1 if dy != 0 else 0) + (1 if dz != 0 else 0)

    if nz <= 1:
        return set()

    corners = set()
    if nz == 2:
        if dx != 0 and dy != 0:
            corners.add((i0[0] + dx, i0[1], i0[2]))
            corners.add((i0[0], i0[1] + dy, i0[2]))
        elif dx != 0 and dz != 0:
            corners.add((i0[0] + dx, i0[1], i0[2]))
            corners.add((i0[0], i0[1], i0[2] + dz))
        else:
            corners.add((i0[0], i0[1] + dy, i0[2]))
            corners.add((i0[0], i0[1], i0[2] + dz))
    else:
        corners.add((i0[0] + dx, i0[1] + dy, i0[2]))
        corners.add((i0[0] + dx, i0[1], i0[2] + dz))
        corners.add((i0[0], i0[1] + dy, i0[2] + dz))
        corners.add((i0[0] + dx, i0[1], i0[2]))
        corners.add((i0[0], i0[1] + dy, i0[2]))
        corners.add((i0[0], i0[1], i0[2] + dz))

    return corners


# ============================================================================
#  3.  Conservative coarse ESDF
# ============================================================================

def make_coarse_esdf(esdf, factor=2):
    """Downsample ESDF using conservative min-pooling.

    Each coarse voxel = min(ESDF values of all fine voxels in that block).
    This guarantees small obstacles (single voxel) are NEVER lost.

    Args:
        esdf:  fine ESDF array (gx, gy, gz)
        factor:  integer downsampling factor

    Returns:
        coarse_esdf array
    """
    if factor <= 1:
        return esdf

    gx, gy, gz = esdf.shape
    cgx = (gx + factor - 1) // factor
    cgy = (gy + factor - 1) // factor
    cgz = (gz + factor - 1) // factor

    pad_x = cgx * factor - gx
    pad_y = cgy * factor - gy
    pad_z = cgz * factor - gz

    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        padded = np.pad(esdf, ((0, pad_x), (0, pad_y), (0, pad_z)),
                        mode='constant', constant_values=-1.0)
    else:
        padded = esdf

    reshaped = padded.reshape(cgx, factor, cgy, factor, cgz, factor)
    coarse = reshaped.min(axis=(1, 3, 5))

    return coarse


# ============================================================================
#  4.  A* planner with proper failure modes
# ============================================================================

class PlanResult:
    """Structured result from A* planning."""
    def __init__(self, path=None, reached_goal=False, iterations=0,
                 failure_reason="", goal_error=None):
        self.path = path or []
        self.reached_goal = reached_goal
        self.iterations = iterations
        self.failure_reason = failure_reason
        self.goal_error = goal_error

    def __bool__(self):
        return self.reached_goal and len(self.path) >= 2

    def __len__(self):
        return len(self.path)


class AStarPlanner:
    """Grid-based A* search on an ESDF (positive = free space).

    Uses floor() world→grid, diagonal corner-cutting prevention,
    clear failure reporting, and stale-queue detection.
    """

    def __init__(self, esdf, resolution, origin, cost_weight=2.0,
                 clearance_target=0.6):
        self.e = esdf
        self.res = resolution
        self.origin = origin
        self.ox, self.oy, self.oz = origin
        self.cw = cost_weight
        self.clearance_target = max(0.0, float(clearance_target))
        self.shape = esdf.shape

    def _w2g(self, p):
        return _w2g(p, self.origin, self.res)

    def _g2w(self, i):
        return _g2w(i, self.origin, self.res)

    def _esdf_at(self, idx):
        return _esdf_at_grid(self.e, idx)

    def _free(self, idx, min_clearance=0.0):
        return _is_free(self.e, idx, min_clearance)

    def _nearest_free(self, idx, min_clearance=0.0, max_radius=25):
        """Find nearest free grid cell within max_radius.

        Returns the free cell index, or None if none found
        (NEVER returns the original collision index).
        """
        if self._free(idx, min_clearance):
            return idx

        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != r:
                            continue
                        n = (idx[0] + dx, idx[1] + dy, idx[2] + dz)
                        if self._free(n, min_clearance):
                            return n
        return None

    def plan(self, start_world, goal_world, min_clearance=0.0,
             max_iterations=500000, epsilon=1.5, max_time_sec=None):
        """Plan a path from start_world to goal_world.

        Returns:
            PlanResult object.  Check result.reached_goal.
        """
        s = self._w2g(start_world)
        g = self._w2g(goal_world)

        if not self._free(s, min_clearance):
            s_free = self._nearest_free(s, min_clearance)
            if s_free is None:
                return PlanResult(
                    reached_goal=False, iterations=0,
                    failure_reason="start_in_collision_no_free_nearby",
                    goal_error=np.linalg.norm(
                        np.array(goal_world) - np.array(start_world)))
            rospy.loginfo("[A*] Start moved: (%.2f,%.2f,%.2f) → (%.2f,%.2f,%.2f)",
                          start_world[0], start_world[1], start_world[2],
                          self._g2w(s_free)[0], self._g2w(s_free)[1], self._g2w(s_free)[2])
            s = s_free

        if not self._free(g, min_clearance):
            g_free = self._nearest_free(g, min_clearance)
            if g_free is None:
                return PlanResult(
                    reached_goal=False, iterations=0,
                    failure_reason="goal_in_collision_no_free_nearby",
                    goal_error=np.linalg.norm(
                        np.array(goal_world) - np.array(start_world)))
            rospy.loginfo("[A*] Goal moved: (%.2f,%.2f,%.2f) → (%.2f,%.2f,%.2f)",
                          goal_world[0], goal_world[1], goal_world[2],
                          self._g2w(g_free)[0], self._g2w(g_free)[1], self._g2w(g_free)[2])
            g = g_free

        if s == g:
            path = [self._g2w(s)]
            return PlanResult(path=path, reached_goal=True, iterations=0, goal_error=0.0)

        start_priority = _diagonal_heuristic_3d(s, g, self.res) * epsilon
        opens = [(start_priority, 0, s)]
        came_from = {}
        g_score = {s: 0.0}
        closed = set()
        iterations = 0
        tie = 0
        gi, gj, gk = g
        best_node = s
        best_goal_dist_sq = ((s[0] - gi) ** 2 + (s[1] - gj) ** 2
                             + (s[2] - gk) ** 2)
        deadline = None
        if max_time_sec is not None and float(max_time_sec) > 0.0:
            deadline = time.monotonic() + float(max_time_sec)

        def partial_failure(reason):
            partial_path = []
            cur = best_node
            while cur in came_from:
                partial_path.append(self._g2w(cur))
                cur = came_from[cur]
            partial_path.append(self._g2w(s))
            partial_path.reverse()
            goal_err_dist = np.linalg.norm(
                np.array(self._g2w(best_node)) - np.array(goal_world))
            return PlanResult(
                path=partial_path, reached_goal=False,
                iterations=iterations, failure_reason=reason,
                goal_error=goal_err_dist)

        while opens:
            iterations += 1
            if iterations > max_iterations:
                return partial_failure("max_iterations_reached")

            # Keep the hot loop cheap, but still honour Ctrl-C and a wall-clock
            # limit promptly instead of requiring roslaunch to send SIGKILL.
            if (iterations & 0xff) == 0:
                if rospy.is_shutdown():
                    return partial_failure("shutdown_requested")
                if deadline is not None and time.monotonic() >= deadline:
                    return partial_failure("planning_timeout")

            _, _, cur = heapq.heappop(opens)

            if cur in closed:
                continue
            closed.add(cur)

            if cur == g:
                path = [self._g2w(cur)]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(self._g2w(cur))
                path.reverse()
                return PlanResult(
                    path=path, reached_goal=True,
                    iterations=iterations, goal_error=0.0)

            for d in _NEIGHBOURS_26:
                n = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])

                corners = _enumerate_diagonal_corners(cur, n)
                corner_blocked = False
                for corner in corners:
                    if not self._free(corner, min_clearance):
                        corner_blocked = True
                        break
                if corner_blocked:
                    continue

                if n in closed:
                    continue
                if not self._free(n, min_clearance):
                    continue

                dist = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2) * self.res
                clearance = self._esdf_at(n)
                # Express clearance preference as a dimensionless per-metre
                # multiplier.  The previous fixed penalty was charged once per
                # voxel, so changing map resolution changed the selected route
                # and strongly favoured long paths through open perimeter space.
                clearance_penalty = max(
                    0.0, self.clearance_target - clearance) * self.cw
                tg = g_score[cur] + dist * (1.0 + clearance_penalty)

                if n not in g_score or tg < g_score[n]:
                    g_score[n] = tg
                    came_from[n] = cur
                    goal_dist_sq = ((n[0] - gi) ** 2 + (n[1] - gj) ** 2
                                    + (n[2] - gk) ** 2)
                    if goal_dist_sq < best_goal_dist_sq:
                        best_goal_dist_sq = goal_dist_sq
                        best_node = n
                    h = _diagonal_heuristic_3d(n, g, self.res)
                    tie += 1
                    heapq.heappush(opens, (tg + h * epsilon, tie, n))

        return PlanResult(
            reached_goal=False, iterations=iterations,
            failure_reason="no_path_exists",
            goal_error=float("inf"))


# ============================================================================
#  5.  Shortcut / greedy string-pulling for global path
# ============================================================================

def _check_segment_clearance_adaptive(p0, p1, esdf, origin, resolution,
                                       min_clearance, check_spacing):
    """Check line segment p0→p1 at adaptive spacing for ESDF clearance.
    
    Returns (is_clear, worst_clearance).
    """
    p0a, p1a = np.array(p0), np.array(p1)
    seg_len = np.linalg.norm(p1a - p0a)
    if seg_len < 1e-9:
        cl = _trilinear_esdf(esdf, origin, resolution, p0)
        return cl >= min_clearance - 1e-3, cl
    
    steps = max(2, int(math.ceil(seg_len / check_spacing)) + 1)
    worst = float("inf")
    for i in range(steps + 1):
        alpha = i / max(steps, 1)
        pt = p0a + alpha * (p1a - p0a)
        cl = _trilinear_esdf(esdf, origin, resolution, pt)
        worst = min(worst, cl)
        # Match the C++ validator's 1 mm numerical tolerance.  ESDF values are
        # float32 and interpolation can produce 0.0999 at a 0.10 m boundary.
        if cl < min_clearance - 1e-3:
            return False, worst
    return True, worst


def shortcut_path(raw_path, esdf, origin, resolution, 
                  min_clearance, check_spacing):
    """Greedy shortcut / string-pulling with adaptive ESDF segment check.
    
    For each point, try to connect to the furthest visible successor.
    Only removes points when the shortcut segment is safe.
    
    If start→goal is entirely clear, returns [start, goal] directly.
    
    Args:
        raw_path: list of (x, y, z) from A*
        esdf: ESDF array
        origin: (ox, oy, oz)
        resolution: voxel size
        min_clearance: hard safety boundary
        check_spacing: max spacing between consecutive checks (should be <= resolution/2)
    
    Returns:
        shortcut list of (x, y, z)
    """
    if len(raw_path) < 2:
        return list(raw_path)
    if len(raw_path) == 2:
        clear, _ = _check_segment_clearance_adaptive(
            raw_path[0], raw_path[1], esdf, origin, resolution,
            min_clearance, check_spacing)
        return list(raw_path) if clear else []
    
    # Check if direct start→goal is entirely clear
    start = raw_path[0]
    goal = raw_path[-1]
    direct_clear, _ = _check_segment_clearance_adaptive(
        start, goal, esdf, origin, resolution, min_clearance, check_spacing)
    if direct_clear:
        return [tuple(start), tuple(goal)]
    
    n = len(raw_path)
    shortcut = [tuple(raw_path[0])]
    current_idx = 0
    
    while current_idx < n - 1:
        best_next = None
        # Try to connect to the furthest visible point.  The adjacent edge is
        # checked too; the old default silently accepted current_idx + 1 even
        # when continuous ESDF validation said that edge was unsafe.
        for j in range(n - 1, current_idx, -1):
            clear, _ = _check_segment_clearance_adaptive(
                raw_path[current_idx], raw_path[j],
                esdf, origin, resolution, min_clearance, check_spacing)
            if clear:
                best_next = j
                break

        if best_next is None:
            rospy.logwarn(
                "[Shortcut] No continuously safe outgoing edge at raw index %d/%d.",
                current_idx, n - 1)
            return []
        
        shortcut.append(tuple(raw_path[best_next]))
        current_idx = best_next
    
    # Ensure goal is the last point
    if np.linalg.norm(np.array(shortcut[-1]) - np.array(goal)) > 1e-6:
        shortcut.append(tuple(goal))
    
    return shortcut


# ============================================================================
#  6.  GlobalPathPlanner — A* + shortcut only, no full-trajectory smoothing
# ============================================================================

class GlobalPathPlanner:
    """Global path planner: A* search + greedy shortcut.
    
    This class ONLY produces a global reference path. It does NOT:
    - Smooth the entire path iteratively
    - Time-parameterize
    - Sample at control rate
    - Generate controls
    
    Those steps are handled by the online C++ local planner.
    """
    
    def __init__(self, esdf, esdf_origin, esdf_resolution, config):
        self.esdf = esdf
        self.esdf_origin = esdf_origin
        self.esdf_res = esdf_resolution
        self.cfg = config
        
        gp_cfg = config.get("planning", {}).get("global_planner", {})
        self.coarse_factor = gp_cfg.get("coarse_factor", 2)
        self.epsilon = gp_cfg.get("epsilon", 1.2)
        self.max_iter_coarse = gp_cfg.get("max_iterations_coarse", 300000)
        self.max_iter_full = gp_cfg.get("max_iterations_full", 800000)
        self.max_time_coarse = gp_cfg.get("max_planning_time_coarse_s", 8.0)
        self.max_time_full = gp_cfg.get("max_planning_time_full_s", 45.0)
        self.cost_weight = gp_cfg.get("cost_weight", 0.35)
        self.clearance_target = gp_cfg.get("clearance_target", 0.20)
        self.min_clearance = gp_cfg.get("min_clearance", 0.10)
        self.shortcut_enabled = gp_cfg.get("shortcut_enabled", True)
        self.shortcut_check_spacing = gp_cfg.get("shortcut_check_spacing", 0.05)
        
        # Ensure check spacing <= resolution / 2
        self.shortcut_check_spacing = min(
            self.shortcut_check_spacing, self.esdf_res * 0.5)
        
        self.a_star = AStarPlanner(
            esdf, esdf_resolution, esdf_origin,
            cost_weight=self.cost_weight,
            clearance_target=self.clearance_target)
    
    def plan_global(self, start, goal):
        """Plan a global reference path from start to goal.
        
        Returns dict:
            start, goal, raw_path, global_path, valid, 
            validation_report, raw_path_points, shortcut_path_points,
            global_path_length
        Returns None on total A* failure.
        """
        t0 = time.time()
        
        # ── Coarse A* first ────────────────────────────────────
        # factor=1 already means full resolution.  Do not search the identical
        # grid twice after hitting the first iteration limit.
        if self.coarse_factor <= 1:
            rospy.loginfo("[GlobalPath] Full-resolution A* (single pass).")
            result = self.a_star.plan(
                start, goal, min_clearance=self.min_clearance,
                max_iterations=self.max_iter_full, epsilon=self.epsilon,
                max_time_sec=self.max_time_full)
        else:
            coarse_esdf = make_coarse_esdf(
                self.esdf, factor=self.coarse_factor)
            coarse_res = self.esdf_res * self.coarse_factor
            coarse_astar = AStarPlanner(
                coarse_esdf, coarse_res, self.esdf_origin,
                cost_weight=self.cost_weight,
                clearance_target=self.clearance_target)

            result = coarse_astar.plan(
                start, goal, min_clearance=self.min_clearance,
                max_iterations=self.max_iter_coarse, epsilon=self.epsilon,
                max_time_sec=self.max_time_coarse)

            if (not result.reached_goal
                    and result.failure_reason != "shutdown_requested"):
                rospy.loginfo(
                    "[GlobalPath] Coarse A* failed (%s), trying full-res...",
                    result.failure_reason)
                result = self.a_star.plan(
                    start, goal, min_clearance=self.min_clearance,
                    max_iterations=self.max_iter_full, epsilon=self.epsilon,
                    max_time_sec=self.max_time_full)
        
        if not result.reached_goal:
            rospy.logwarn("[GlobalPath] A* FAILED: %s (iter=%d, goal_err=%.2f m)",
                          result.failure_reason, result.iterations,
                          result.goal_error if result.goal_error else -1)
            return None
        
        # Restore exact start/goal endpoints
        raw_path = list(result.path)
        raw_path[0] = tuple(start)
        raw_path[-1] = tuple(goal)
        
        # ── Shortcut (greedy string-pulling) ────────────────────
        if self.shortcut_enabled and len(raw_path) > 2:
            global_path = shortcut_path(
                raw_path, self.esdf, self.esdf_origin, self.esdf_res,
                self.min_clearance, self.shortcut_check_spacing)
        else:
            global_path = list(raw_path)
        
        # ── Compute path length ─────────────────────────────────
        global_path_length = 0.0
        for i in range(1, len(global_path)):
            global_path_length += np.linalg.norm(
                np.array(global_path[i]) - np.array(global_path[i-1]))
        
        # ── Validate global path ────────────────────────────────
        has_path = len(global_path) >= 2
        all_clear = has_path
        worst_cl = float("inf")
        violations = 0
        any_collision = False
        
        for i in range(1, len(global_path)):
            clear, wc = _check_segment_clearance_adaptive(
                global_path[i-1], global_path[i],
                self.esdf, self.esdf_origin, self.esdf_res,
                self.min_clearance, self.shortcut_check_spacing)
            worst_cl = min(worst_cl, wc)
            if not clear:
                all_clear = False
                violations += 1
            if wc <= 0:
                any_collision = True
        
        if not has_path:
            worst_cl = -1.0
        valid = all_clear and not any_collision and has_path
        
        # ── Build validation report ─────────────────────────────
        report = {
            "astar_reached_goal": result.reached_goal,
            "astar_iterations": result.iterations,
            "astar_failure_reason": result.failure_reason if not result.reached_goal else "",
            "raw_path_points": len(raw_path),
            "shortcut_path_points": len(global_path),
            "global_path_length_m": round(global_path_length, 3),
            "all_segments_clear": all_clear,
            "worst_clearance": round(float(worst_cl), 4),
            "clearance_violations": violations,
            "any_collision": any_collision,
            "valid": valid,
            "planning_time_sec": round(time.time() - t0, 3),
        }
        
        if not valid:
            reasons = []
            if not has_path:
                reasons.append("shortcut_failed_no_safe_edge")
            elif not all_clear:
                reasons.append("clearance_violations={}".format(violations))
            if any_collision:
                reasons.append("collision_detected")
            report["invalid_reasons"] = reasons
        
        plan = {
            "start": list(start),
            "goal": list(goal),
            "raw_path": raw_path,
            "global_path": global_path,
            "global_path_length": global_path_length,
            "valid": valid,
            "validation_report": report,
            # Legacy keys for backward compat (set to empty/None)
            "sampled_traj": [],
            "controls": [],
            "optimised_path": None,
            "resampled_path": None,
            "timed_waypoints": None,
            "total_time": 0.0,
        }
        
        rospy.loginfo("[GlobalPath] A* %d pts → shortcut %d pts, length=%.1fm, "
                      "valid=%s, time=%.2fs",
                      len(raw_path), len(global_path), global_path_length,
                      valid, report["planning_time_sec"])
        
        return plan


# ============================================================================
#  ==== DEPRECATED FUNCTIONS BELOW — NOT USED BY NEW PIPELINE ====
# ============================================================================
# These are kept ONLY for backward compatibility and debugging.
# The new pipeline does NOT call them.

def _deprecated_warning(name):
    rospy.logwarn_once("[il_trajectory] DEPRECATED function called: %s "
                        "(not used by v5 pipeline)", name)

# ============================================================================
#  7.  DEPRECATED: Constrained trajectory smoother with segment checking
# ============================================================================

def _check_segment_clearance(p0, p1, esdf, origin, resolution,
                             min_clearance, check_steps=10):
    """DEPRECATED. Kept for backward compatibility only."""
    _deprecated_warning("_check_segment_clearance")
    p0a, p1a = np.array(p0), np.array(p1)
    worst = float("inf")
    for i in range(check_steps + 1):
        alpha = i / max(check_steps, 1)
        pt = p0a + alpha * (p1a - p0a)
        cl = _esdf_at_world(esdf, origin, resolution, pt)
        worst = min(worst, cl)
        if cl <= min_clearance:
            return False, worst
    return True, worst


def smooth_trajectory(path, esdf, origin, resolution,
                      smooth_iterations=300, smooth_step=0.5,
                      push_iterations=60, push_step=0.03,
                      min_clearance=0.05, target_clearance=0.6,
                      final_smooth_iterations=60):
    """Smooth a coarse A* path while maintaining ESDF collision-free.

    Start and goal points are KEPT FIXED.
    Segment clearance is checked between neighbors, not just at waypoints.
    """
    if len(path) < 3:
        return list(path)

    ox, oy, oz = origin
    inv_res = 1.0 / resolution
    gx, gy, gz = esdf.shape

    def _eval(p):
        ix = int(math.floor((p[0] - ox) * inv_res))
        iy = int(math.floor((p[1] - oy) * inv_res))
        iz = int(math.floor((p[2] - oz) * inv_res))
        ix = max(0, min(gx - 1, ix))
        iy = max(0, min(gy - 1, iy))
        iz = max(0, min(gz - 1, iz))
        return float(esdf[ix, iy, iz])

    def _grad(p):
        ix = int(math.floor((p[0] - ox) * inv_res))
        iy = int(math.floor((p[1] - oy) * inv_res))
        iz = int(math.floor((p[2] - oz) * inv_res))
        ix = max(1, min(gx - 2, ix))
        iy = max(1, min(gy - 2, iy))
        iz = max(1, min(gz - 2, iz))
        grad = np.zeros(3)
        grad[0] = (esdf[ix + 1, iy, iz] - esdf[ix - 1, iy, iz]) / (2 * resolution)
        grad[1] = (esdf[ix, iy + 1, iz] - esdf[ix, iy - 1, iz]) / (2 * resolution)
        grad[2] = (esdf[ix, iy, iz + 1] - esdf[ix, iy, iz - 1]) / (2 * resolution)
        return grad

    pts = [np.array(p, dtype=np.float64) for p in path]
    n = len(pts)

    def _laplacian_smooth(points, iterations, step):
        """Smooth without moving endpoints or crossing the clearance bound."""
        for _ in range(iterations):
            new_points = [points[0].copy()]
            max_move = 0.0
            for i in range(1, n - 1):
                lap = (points[i - 1] + points[i + 1]) / 2.0 - points[i]
                move = step * lap
                candidate = points[i].copy()
                for frac in (1.0, 0.5, 0.25, 0.1):
                    trial = points[i] + move * frac
                    prev_ok, _ = _check_segment_clearance(
                        new_points[-1], trial, esdf, origin, resolution,
                        min_clearance)
                    next_ok, _ = _check_segment_clearance(
                        trial, points[i + 1], esdf, origin, resolution,
                        min_clearance)
                    if (_eval(trial) > min_clearance and
                            prev_ok and next_ok):
                        candidate = trial
                        break
                max_move = max(
                    max_move, np.linalg.norm(candidate - points[i]))
                new_points.append(candidate)
            new_points.append(points[-1].copy())
            points = new_points
            if max_move < 1e-4:
                break
        return points

    # Phase 1: remove the grid stair-step pattern from A*.
    pts = _laplacian_smooth(pts, smooth_iterations, smooth_step)

    # Phase 2: push only points that are below the requested clearance.  The
    # previous implementation pushed every point for every iteration, so even
    # already-safe paths drifted towards the edge of the scene.
    target_clearance = max(float(target_clearance), float(min_clearance))
    for _ in range(push_iterations):
        new_pts = [pts[0].copy()]
        max_move = 0.0
        for i in range(1, n - 1):
            clearance = _eval(pts[i])
            deficit = target_clearance - clearance
            if deficit <= 0.0:
                new_pts.append(pts[i].copy())
                continue
            g = _grad(pts[i])
            gn = np.linalg.norm(g)
            if gn < 1e-9:
                new_pts.append(pts[i].copy())
                continue
            g = g / gn
            step = min(float(push_step), deficit)
            candidate = pts[i] + step * g

            if _eval(candidate) <= min_clearance:
                found = False
                for frac in (0.5, 0.25):
                    c2 = pts[i] + step * frac * g
                    if _eval(c2) > min_clearance:
                        candidate = c2
                        found = True
                        break
                if not found:
                    candidate = pts[i].copy()

            if i > 0:
                prev_ok, _ = _check_segment_clearance(
                    new_pts[-1], candidate, esdf, origin, resolution, min_clearance)
                next_ok, _ = _check_segment_clearance(
                    candidate, pts[i + 1], esdf, origin, resolution,
                    min_clearance)
                if not (prev_ok and next_ok):
                    candidate = pts[i].copy()

            new_pts.append(candidate)
            max_move = max(max_move, np.linalg.norm(candidate - pts[i]))
        new_pts.append(pts[-1].copy())
        pts = new_pts
        if max_move < 1e-5:
            break

    # The gradient stage can introduce small zig-zags.  Finish with a gentler,
    # collision-constrained pass so the time-parameterised path stays smooth.
    pts = _laplacian_smooth(pts, final_smooth_iterations,
                            min(float(smooth_step), 0.25))

    return [tuple(p) for p in pts]


# ============================================================================
#  6.  Path resampling
# ============================================================================

def resample_path(path, spacing=0.2):
    """Uniformly resample to *spacing* metres between consecutive points.

    The last point of the input path is always the last point of the output.
    """
    if len(path) < 2:
        return list(path)
    out = [path[0]]
    for i in range(1, len(path)):
        a, b = np.array(path[i - 1]), np.array(path[i])
        L = np.linalg.norm(b - a)
        n = max(1, int(L / spacing))
        for j in range(1, n + 1):
            out.append(tuple(a + (b - a) * j / n))
    out[-1] = path[-1]
    return out


def smooth_position_path(path, window=5):
    """Apply moving-average filter to position path.

    Start and goal endpoints are KEPT FIXED.
    """
    n = len(path)
    if n < 2 * window + 1:
        return list(path)
    smoothed = [path[0]]
    for i in range(1, n - 1):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        avg = np.mean([path[k] for k in range(lo, hi + 1)], axis=0)
        smoothed.append(tuple(avg))
    smoothed.append(path[-1])
    return smoothed


# ============================================================================
#  7.  Yaw profile generation (unified canonical yaw)
# ============================================================================

def generate_yaw_profile(waypoints):
    """Generate a canonical yaw profile from a position path.

    Yaw convention (ROS world): yaw=0 → nose faces +Y (left).
    Yaw is unwrapped before any processing.
    """
    n = len(waypoints)
    if n < 2:
        return np.zeros(n)

    pts = np.array(waypoints)
    yaws = np.zeros(n)

    for i in range(n - 1):
        dx = pts[i + 1, 0] - pts[i, 0]
        dy = pts[i + 1, 1] - pts[i, 1]
        yaws[i] = math.atan2(dy, dx) - math.pi / 2.0

    yaws[-1] = yaws[-2] if n >= 2 else 0.0
    return np.unwrap(yaws)


def smooth_yaw_profile(yaws_unwrapped, alpha=0.85):
    """Apply exponential moving average (EMA) smoothing to yaw.

    alpha = 0: no smoothing (raw yaw)
    alpha close to 1: strong smoothing

    EMA: y_smooth[i] = y_smooth[i-1] + (1-alpha) * (y_raw[i] - y_smooth[i-1])
    """
    n = len(yaws_unwrapped)
    if n <= 1:
        return yaws_unwrapped.copy()

    smoothed = np.zeros(n)
    smoothed[0] = yaws_unwrapped[0]
    for i in range(1, n):
        smoothed[i] = smoothed[i - 1] + (1.0 - alpha) * (yaws_unwrapped[i] - smoothed[i - 1])

    return smoothed


# ============================================================================
#  8.  Time parameterisation
# ============================================================================

def _smooth_velocity_profile(velocities, max_acc, segment_lengths):
    """Forward-backward pass to enforce max_acceleration."""
    n = len(velocities)
    v = list(velocities)
    for i in range(n - 1):
        d = max(1e-6, segment_lengths[i])
        v[i + 1] = min(v[i + 1], math.sqrt(max(0.0, v[i]**2 + 2 * max_acc * d)))
    for i in range(n - 2, -1, -1):
        d = max(1e-6, segment_lengths[i])
        v[i] = min(v[i], math.sqrt(max(0.0, v[i + 1]**2 + 2 * max_acc * d)))
    return v


def time_parameterize(waypoints, constraints):
    """Assign timestamps to waypoints respecting dynamics limits."""
    if len(waypoints) < 2:
        return [(0.0,) + waypoints[0] + (0.0,)] if waypoints else []

    v_nom  = constraints.get("nominal_speed", 2.5)
    v_max  = constraints.get("max_velocity", 5.0)
    a_max  = constraints.get("max_acceleration", 4.0)
    yr_max = constraints.get("max_yaw_rate", 2.0)
    curv_f = constraints.get("curvature_slowdown", True)
    curv_g = constraints.get("curvature_gain", 0.8)

    n = len(waypoints)
    pts = np.array(waypoints)
    seg_vec = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_dir = np.zeros_like(seg_vec)
    for i in range(n - 1):
        if seg_len[i] > 1e-9:
            seg_dir[i] = seg_vec[i] / seg_len[i]

    velocities = np.full(n, v_nom, dtype=np.float64)
    velocities[0] = 0.0
    velocities[-1] = 0.0

    if curv_f and n > 2:
        for i in range(1, n - 1):
            d1 = seg_dir[i - 1] if i - 1 < n - 1 else np.zeros(3)
            d2 = seg_dir[i] if i < n - 1 else np.zeros(3)
            dot = np.clip(np.dot(d1, d2), -1.0, 1.0)
            angle = math.acos(dot)
            avg_L = 0.5 * (seg_len[max(0, i - 1)] + seg_len[min(n - 2, i)]) + 1e-6
            curvature = angle / avg_L
            if curvature > 1e-6:
                v_yr = yr_max / curvature
                velocities[i] = min(velocities[i], v_yr)
            slowdown = 1.0 / (1.0 + curv_g * curvature * 10.0)
            velocities[i] = min(velocities[i], v_nom * slowdown)

    velocities = _smooth_velocity_profile(velocities, a_max, seg_len)
    velocities = np.clip(velocities, 0.0, v_max)

    t = 0.0
    timed = []
    for i in range(n):
        yaw = math.atan2(seg_dir[i][1], seg_dir[i][0]) - math.pi / 2.0 if i < n - 1 else (
            math.atan2(seg_dir[-1][1], seg_dir[-1][0]) - math.pi / 2.0 if n > 1 else 0.0)
        timed.append((t, waypoints[i][0], waypoints[i][1], waypoints[i][2], yaw))
        if i < n - 1:
            v_avg = max(0.5 * (velocities[i] + velocities[i + 1]), 0.2)
            dt = seg_len[i] / v_avg
            t += dt

    return timed


# ============================================================================
#  9.  Trajectory sampling
# ============================================================================

def sample_trajectory(timed_waypoints, dt_sample):
    """Evenly sample the timed trajectory at *dt_sample* seconds.

    The LAST sample is at t = total_time with EXACT goal position and
    final yaw from the timed waypoints.
    """
    if len(timed_waypoints) < 2:
        return list(timed_waypoints)

    total_t = timed_waypoints[-1][0]
    n_samples = max(2, int(total_t / dt_sample) + 1)
    sampled = []
    wp_idx = 0

    for k in range(n_samples):
        t = k * dt_sample
        t = min(t, total_t)

        while (wp_idx + 1 < len(timed_waypoints) and
               timed_waypoints[wp_idx + 1][0] < t):
            wp_idx += 1

        if wp_idx + 1 >= len(timed_waypoints):
            w = timed_waypoints[-1]
            sampled.append((t, w[1], w[2], w[3], w[4]))
            continue

        w0, w1 = timed_waypoints[wp_idx], timed_waypoints[wp_idx + 1]
        dt_seg = w1[0] - w0[0]
        alpha = (t - w0[0]) / max(dt_seg, 1e-9)
        alpha = max(0.0, min(1.0, alpha))
        x = w0[1] + alpha * (w1[1] - w0[1])
        y = w0[2] + alpha * (w1[2] - w0[2])
        z = w0[3] + alpha * (w1[3] - w0[3])

        y0, y1 = w0[4], w1[4]
        dy = y1 - y0
        dy = math.atan2(math.sin(dy), math.cos(dy))
        yaw = y0 + alpha * dy

        sampled.append((t, x, y, z, yaw))

    if len(sampled) > 0:
        w_last = timed_waypoints[-1]
        sampled[-1] = (w_last[0], w_last[1], w_last[2], w_last[3], w_last[4])

    return sampled


# ============================================================================
#  10.  Control generation (unified yaw profile)
# ============================================================================

def generate_controls(sampled_traj, dt, lookahead=6, yaw_smooth_alpha=0.85,
                      max_vel=5.0, max_yaw_rate=None):
    """Generate smooth velocity + yaw-rate control commands.

    Uses a UNIFIED canonical yaw profile:
      1. Extract raw yaw from sampled trajectory
      2. Unwrap
      3. Apply EMA smoothing
      4. Compute yaw-rate by discrete derivative of smoothed yaw
      5. Clamp yaw-rate to max_yaw_rate and re-integrate for corrected yaw
      6. Pose quaternion uses the corrected yaw (same canonical profile)

    Returns:
        list of (t, vx_world, vy_world, vz_world, yaw_rate, corrected_yaw)
    """
    n = len(sampled_traj)
    if n < 2:
        return [(sampled_traj[0][0], 0.0, 0.0, 0.0, 0.0, sampled_traj[0][4])] if n else []

    yaws_raw = np.array([w[4] for w in sampled_traj])
    yaws_unwrapped = np.unwrap(yaws_raw)
    yaws_smooth_unwrapped = smooth_yaw_profile(yaws_unwrapped, yaw_smooth_alpha)

    yr_raw = np.zeros(n)
    for i in range(1, n):
        yr_raw[i] = (yaws_smooth_unwrapped[i] - yaws_smooth_unwrapped[i - 1]) / dt
    yr_raw[0] = yr_raw[1] if n >= 2 else 0.0

    if max_yaw_rate is not None and max_yaw_rate > 0:
        yr = np.clip(yr_raw, -max_yaw_rate, max_yaw_rate)
        yaws_corrected_unwrapped = np.zeros(n)
        yaws_corrected_unwrapped[0] = yaws_unwrapped[0]
        for i in range(1, n):
            yaws_corrected_unwrapped[i] = yaws_corrected_unwrapped[i - 1] + yr[i] * dt
    else:
        yr = yr_raw
        yaws_corrected_unwrapped = yaws_smooth_unwrapped

    controls = []
    for i in range(n):
        t_i = sampled_traj[i][0]

        j = min(i + lookahead, n - 1)
        if j == i:
            vx = vy = vz = 0.0
        else:
            cur = np.array([sampled_traj[i][1], sampled_traj[i][2], sampled_traj[i][3]])
            nxt = np.array([sampled_traj[j][1], sampled_traj[j][2], sampled_traj[j][3]])
            dpos = nxt - cur
            L = np.linalg.norm(dpos)
            if L > 0.01:
                direction = dpos / L
                speed = min(L / (dt * lookahead), max_vel)
                vx, vy, vz = direction * speed
            else:
                vx = vy = vz = 0.0

        corrected_yaw = math.atan2(math.sin(yaws_corrected_unwrapped[i]),
                                   math.cos(yaws_corrected_unwrapped[i]))

        controls.append((t_i, vx, vy, vz, yr[i], corrected_yaw))

    return controls


# ============================================================================
#  11.  Comprehensive dynamics & safety validation
# ============================================================================

def _check_path_segment_clearance(path, esdf, origin, resolution,
                                  min_clearance, check_steps_per_seg=5):
    """Check clearance along all segments of a path.

    Returns (all_clear, worst_clearance, clearance_violations,
             worst_violation_margin, any_collision).
    """
    n = len(path)
    if n < 2:
        return True, float("inf"), 0, 0.0, False

    worst_cl = float("inf")
    violations = 0
    worst_margin = 0.0
    any_collision = False

    ox, oy, oz = origin
    inv = 1.0 / resolution
    gx, gy, gz = esdf.shape

    for i in range(n - 1):
        p0 = np.array(path[i])
        p1 = np.array(path[i + 1])
        for s in range(check_steps_per_seg + 1):
            alpha = s / max(check_steps_per_seg, 1)
            pt = p0 + alpha * (p1 - p0)
            ix = int(math.floor((pt[0] - ox) * inv))
            iy = int(math.floor((pt[1] - oy) * inv))
            iz = int(math.floor((pt[2] - oz) * inv))
            if 0 <= ix < gx and 0 <= iy < gy and 0 <= iz < gz:
                cl = float(esdf[ix, iy, iz])
            else:
                cl = -1.0

            worst_cl = min(worst_cl, cl)
            if cl <= min_clearance:
                violations += 1
                margin = min_clearance - cl
                worst_margin = max(worst_margin, margin)
            if cl <= 0:
                any_collision = True

    return violations == 0, worst_cl, violations, worst_margin, any_collision


def validate_dynamics(timed_waypoints, constraints, esdf=None,
                      esdf_origin=None, esdf_res=None):
    """Comprehensive dynamics and safety validation.

    Checks: velocity, acceleration, jerk, yaw-rate, obstacle clearance
    along SEGMENTS (not just waypoints).

    Always returns a full report; caller decides validity based on
    thresholds.
    """
    v_max   = constraints.get("max_velocity", 5.0)
    a_max   = constraints.get("max_acceleration", 4.0)
    j_max   = constraints.get("max_jerk", 10.0)
    yr_max  = constraints.get("max_yaw_rate", 2.0)
    min_cl  = constraints.get("min_obstacle_clearance", 0.3)

    n = len(timed_waypoints)
    if n < 2:
        return True, {"total_violations": 0, "note": "too_short"}

    times = np.array([w[0] for w in timed_waypoints])
    pos = np.array([[w[1], w[2], w[3]] for w in timed_waypoints])
    yaws = np.array([w[4] for w in timed_waypoints])

    dt = np.diff(times)
    report = {
        "num_waypoints": n,
        "total_time": float(times[-1]),
        "max_velocity": 0.0,
        "max_velocity_limit": v_max,
        "max_velocity_excess": 0.0,
        "velocity_violations": 0,
        "velocity_violation_pct": 0.0,
        "max_acceleration": 0.0,
        "max_acceleration_limit": a_max,
        "max_acceleration_excess": 0.0,
        "acceleration_violations": 0,
        "acceleration_violation_pct": 0.0,
        "max_jerk": 0.0,
        "max_jerk_limit": j_max,
        "max_jerk_excess": 0.0,
        "jerk_violations": 0,
        "jerk_violation_pct": 0.0,
        "max_yaw_rate": 0.0,
        "max_yaw_rate_limit": yr_max,
        "max_yaw_rate_excess": 0.0,
        "yaw_rate_violations": 0,
        "yaw_rate_violation_pct": 0.0,
        "min_clearance": float("inf"),
        "min_clearance_limit": min_cl,
        "clearance_violations": 0,
        "any_collision": False,
        "start_error": 0.0,
        "goal_error": 0.0,
        "total_violations": 0,
    }

    max_v = 0.0; max_v_excess = 0.0; v_viol = 0
    for i in range(n - 1):
        d = pos[i + 1] - pos[i]
        v = np.linalg.norm(d) / max(dt[i], 1e-6)
        max_v = max(max_v, v)
        if v > v_max:
            v_viol += 1
            max_v_excess = max(max_v_excess, v - v_max)
    report["max_velocity"] = float(max_v)
    report["max_velocity_excess"] = float(max_v_excess)
    report["velocity_violations"] = v_viol
    report["velocity_violation_pct"] = round(100.0 * v_viol / max(n - 1, 1), 1)

    max_a = 0.0; max_a_excess = 0.0; a_viol = 0
    for i in range(1, n - 1):
        v1 = (pos[i] - pos[i - 1]) / max(dt[i - 1], 1e-6)
        v2 = (pos[i + 1] - pos[i]) / max(dt[i], 1e-6)
        a = np.linalg.norm(v2 - v1) / max(0.5 * (dt[i - 1] + dt[i]), 1e-6)
        max_a = max(max_a, a)
        if a > a_max:
            a_viol += 1
            max_a_excess = max(max_a_excess, a - a_max)
    report["max_acceleration"] = float(max_a)
    report["max_acceleration_excess"] = float(max_a_excess)
    report["acceleration_violations"] = a_viol
    report["acceleration_violation_pct"] = round(100.0 * a_viol / max(n - 2, 1), 1)

    max_j = 0.0; max_j_excess = 0.0; j_viol = 0
    for i in range(2, n - 1):
        v0 = (pos[i - 1] - pos[i - 2]) / max(dt[i - 2], 1e-6)
        v1 = (pos[i] - pos[i - 1]) / max(dt[i - 1], 1e-6)
        v2 = (pos[i + 1] - pos[i]) / max(dt[i], 1e-6)
        a1 = (v1 - v0) / max(dt[i - 1], 1e-6)
        a2 = (v2 - v1) / max(dt[i], 1e-6)
        j = np.linalg.norm(a2 - a1) / max(0.5 * (dt[i - 2] + dt[i - 1] + dt[i]), 1e-6)
        max_j = max(max_j, j)
        if j > j_max:
            j_viol += 1
            max_j_excess = max(max_j_excess, j - j_max)
    report["max_jerk"] = float(max_j)
    report["max_jerk_excess"] = float(max_j_excess)
    report["jerk_violations"] = j_viol
    report["jerk_violation_pct"] = round(100.0 * j_viol / max(n - 3, 1), 1)

    max_yr = 0.0; max_yr_excess = 0.0; yr_viol = 0
    for i in range(1, n):
        dy = yaws[i] - yaws[i - 1]
        dy = math.atan2(math.sin(dy), math.cos(dy))
        yr = abs(dy) / max(dt[i - 1], 1e-6)
        max_yr = max(max_yr, yr)
        if yr > yr_max:
            yr_viol += 1
            max_yr_excess = max(max_yr_excess, yr - yr_max)
    report["max_yaw_rate"] = float(max_yr)
    report["max_yaw_rate_excess"] = float(max_yr_excess)
    report["yaw_rate_violations"] = yr_viol
    report["yaw_rate_violation_pct"] = round(100.0 * yr_viol / max(n - 1, 1), 1)

    if esdf is not None and esdf_origin is not None and esdf_res is not None:
        pos_list = [(w[1], w[2], w[3]) for w in timed_waypoints]
        _, worst_cl, cl_viol, worst_margin, any_col = _check_path_segment_clearance(
            pos_list, esdf, esdf_origin, esdf_res, min_cl)
        report["min_clearance"] = float(worst_cl) if worst_cl != float("inf") else 0.0
        report["clearance_violations"] = cl_viol
        report["worst_clearance_margin"] = float(worst_margin)
        report["any_collision"] = any_col

    report["total_violations"] = (v_viol + a_viol + j_viol + yr_viol +
                                   report.get("clearance_violations", 0))

    return True, report


# ============================================================================
#  12.  DEPRECATED: TrajectoryPlanner — backward-compatible wrapper
# ============================================================================

class TrajectoryPlanner:
    """DEPRECATED — Full pipeline planner for v3/v4 compatibility.
    
    This class wraps GlobalPathPlanner and also calls the deprecated
    smoothing/time_param/sample/control functions to produce the old
    output format.  New code should use GlobalPathPlanner.plan_global()
    and the C++ local planner instead.
    """
    
    def __init__(self, esdf, esdf_origin, esdf_resolution, config):
        _deprecated_warning("TrajectoryPlanner")
        self.esdf = esdf
        self.esdf_origin = esdf_origin
        self.esdf_res = esdf_resolution
        self.cfg = config
        self.pl_cfg = config.get("planning", {})
        self.global_planner = GlobalPathPlanner(esdf, esdf_origin, esdf_resolution, config)
    
    def plan_one(self, start, goal):
        """Plan a single trajectory using the full v3/v4 pipeline.
        
        DEPRECATED: Use GlobalPathPlanner.plan_global() instead.
        Returns dict with all planning results, or None on A* failure.
        """
        _deprecated_warning("TrajectoryPlanner.plan_one")
        
        tp_cfg = self.pl_cfg.get("time_param", {})
        opt_cfg = self.pl_cfg.get("esdf_optimize", {})
        
        # 1. Global A* + shortcut
        global_plan = self.global_planner.plan_global(start, goal)
        if global_plan is None:
            return None
        
        raw_path = global_plan["raw_path"]
        global_path = global_plan["global_path"]
        
        # 2. (DEPRECATED) Full smooth
        if opt_cfg.get("enabled", False):
            optimised = smooth_trajectory(
                raw_path, self.esdf, self.esdf_origin, self.esdf_res,
                smooth_iterations=opt_cfg.get("smooth_iterations", 250),
                smooth_step=opt_cfg.get("smooth_step", 0.35),
                push_iterations=opt_cfg.get("push_iterations", 25),
                push_step=opt_cfg.get("push_step", 0.02),
                min_clearance=opt_cfg.get("min_clearance", 0.35),
                target_clearance=opt_cfg.get("target_clearance", 0.55),
                final_smooth_iterations=opt_cfg.get("final_smooth_iterations", 80))
        else:
            optimised = list(global_path)
        
        # 3. Resample
        resample_sp = self.pl_cfg.get("resample_spacing", 0.15)
        dense = resample_path(optimised, resample_sp)
        
        # 4. Position smooth
        pos_smooth_window = self.pl_cfg.get("pos_smooth_window", 10)
        if pos_smooth_window > 0 and len(dense) > pos_smooth_window * 2:
            dense = smooth_position_path(dense, pos_smooth_window)
        
        # 5. Time param
        timed_dense = time_parameterize(dense, tp_cfg)
        
        # 6. Sample
        ctrl_hz = self.cfg.get("control", {}).get("control_hz", 25.0)
        dt_sample = 1.0 / ctrl_hz
        sampled = sample_trajectory(timed_dense, dt_sample)
        
        # 7. Controls
        ctrl_lookahead = self.pl_cfg.get("control_lookahead", 8)
        ctrl_yaw_smooth = self.pl_cfg.get("control_yaw_smooth", 0.92)
        max_v = tp_cfg.get("max_velocity", 2.5)
        max_yr = tp_cfg.get("max_yaw_rate", 2.0)
        controls = generate_controls(sampled, dt_sample,
                                     lookahead=ctrl_lookahead,
                                     yaw_smooth_alpha=ctrl_yaw_smooth,
                                     max_vel=max_v, max_yaw_rate=max_yr)
        
        sampled = [(w[0], w[1], w[2], w[3], controls[i][5])
                   for i, w in enumerate(sampled)]
        
        # 8. Validate
        start_err = np.linalg.norm(
            np.array([sampled[0][1], sampled[0][2], sampled[0][3]]) - np.array(start))
        goal_err = np.linalg.norm(
            np.array([sampled[-1][1], sampled[-1][2], sampled[-1][3]]) - np.array(goal))
        
        _, report = validate_dynamics(sampled, tp_cfg,
                                      esdf=self.esdf, esdf_origin=self.esdf_origin,
                                      esdf_res=self.esdf_res)
        report["start_error"] = float(start_err)
        report["goal_error"] = float(goal_err)
        
        val_cfg = self.pl_cfg.get("validation", {})
        max_cl_v = val_cfg.get("max_clearance_violations", 0)
        max_dyn_v = val_cfg.get("max_dynamics_violations", 20)
        dyn_v = (report.get("velocity_violations", 0) +
                 report.get("acceleration_violations", 0) +
                 report.get("jerk_violations", 0) +
                 report.get("yaw_rate_violations", 0))
        any_collision = report.get("any_collision", False)
        
        valid = (not any_collision and
                 report.get("clearance_violations", 0) <= max_cl_v and
                 dyn_v <= max_dyn_v)
        
        if not valid:
            reasons = []
            if any_collision:
                reasons.append("collision_detected")
            if report.get("clearance_violations", 0) > max_cl_v:
                reasons.append("clearance_violations")
            if dyn_v > max_dyn_v:
                reasons.append("dynamics_violations={}".format(dyn_v))
            report["invalid_reasons"] = reasons
        
        return {
            "raw_path": raw_path,
            "optimised_path": optimised,
            "resampled_path": dense,
            "timed_waypoints": timed_dense,
            "valid": valid,
            "validation_report": report,
            "sampled_traj": sampled,
            "controls": controls,
            "start": list(start),
            "goal": list(goal),
            "total_time": sampled[-1][0] if sampled else 0.0,
        }
