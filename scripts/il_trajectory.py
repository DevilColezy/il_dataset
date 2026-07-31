#!/usr/bin/env python3
"""
Global Path Planning Module — goal-directed branching search.

=== v7: Goal-directed Branching Global Path ===
This module provides the SINGLE, DETERMINISTIC global path algorithm:
  - Advance toward the final goal in fixed collision-checked steps
  - Branch only when the next goal-directed step violates ESDF clearance
  - For every angular branch, use its smallest collision-free deviation
  - Keep all branches in an A*-ordered queue and return the shortest route
  - GlobalPathPlanner: plan_global() returns the only path or failure

NO clearance reward, NO unconstrained smoothing, NO fallback planners.

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


def _validate_polyline(path, esdf, origin, resolution,
                       min_clearance, check_spacing):
    """Return (valid, worst_clearance) for a continuously checked polyline."""
    if len(path) < 2:
        return False, -1.0
    worst = float("inf")
    for index in range(1, len(path)):
        clear, segment_worst = _check_segment_clearance_adaptive(
            path[index - 1], path[index], esdf, origin, resolution,
            min_clearance, check_spacing)
        worst = min(worst, segment_worst)
        if not clear:
            return False, worst
    return True, worst


# ============================================================================
#  5.  Goal-directed branching search
# ============================================================================

def _goal_branch_state_key(position, origin, merge_resolution):
    """Quantise a continuous search point for deterministic branch merging."""
    scaled = (
        (np.asarray(position, dtype=np.float64) -
         np.asarray(origin, dtype=np.float64)) / merge_resolution)
    return tuple(int(math.floor(value + 0.5)) for value in scaled)


def _goal_branch_directions(goal_direction, angular_step_rad,
                            azimuth_bins, max_deviation_rad,
                            planar_search=False):
    """Yield one angular ray at a time around the goal direction.

    Every azimuth defines an independent potential avoidance branch.  In 3-D,
    the caller keeps the smallest collision-free deviation on each ray.  The
    planar search can retain multiple clear deviations on its left/right rays
    so the shortest-path search does not lose a necessary boundary branch.
    """
    direction = np.asarray(goal_direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-9:
        return []
    direction = direction / direction_norm

    if planar_search:
        base_angle = math.atan2(direction[1], direction[0])
        angle_count = max(
            1, int(math.ceil(max_deviation_rad / angular_step_rad)))
        rays = []
        for side_sign in (-1.0, 1.0):
            ray = []
            for angle_index in range(1, angle_count + 1):
                deviation = min(
                    max_deviation_rad,
                    angle_index * angular_step_rad)
                heading = base_angle + side_sign * deviation
                ray.append((
                    deviation,
                    np.array(
                        [math.cos(heading), math.sin(heading), 0.0],
                        dtype=np.float64)))
            rays.append(ray)
        return rays

    # Pick the least-parallel world axis to build a stable perpendicular basis.
    axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    reference = min(axes, key=lambda axis: abs(float(np.dot(axis, direction))))
    basis_u = np.cross(direction, reference)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-12)
    basis_v = np.cross(direction, basis_u)
    basis_v /= max(float(np.linalg.norm(basis_v)), 1e-12)

    angle_count = max(
        1, int(math.ceil(max_deviation_rad / angular_step_rad)))
    rays = []
    for azimuth_index in range(azimuth_bins):
        azimuth = 2.0 * math.pi * azimuth_index / float(azimuth_bins)
        lateral = (
            math.cos(azimuth) * basis_u +
            math.sin(azimuth) * basis_v)
        ray = []
        for angle_index in range(1, angle_count + 1):
            angle = min(
                max_deviation_rad, angle_index * angular_step_rad)
            candidate = (
                math.cos(angle) * direction +
                math.sin(angle) * lateral)
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm > 1e-9:
                ray.append((angle, candidate / candidate_norm))
        rays.append(ray)
    return rays


def _goal_directed_branch_path_global_competition_legacy(
        start, goal, esdf, origin, resolution,
        min_clearance, check_spacing, config):
    """Plan by walking toward the goal and branching only at collisions.

    Search states are continuous points merged at a configurable spatial
    resolution.  A free goal-directed step has exactly one successor.  If that
    step is blocked, collision-free angular successors become branches.  The
    open set is ordered by
    ``travelled_length + Euclidean_distance_to_goal``, so all surviving
    branches compete on complete geometric path length.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if (start.shape != (3,) or goal.shape != (3,) or
            not np.all(np.isfinite(start)) or
            not np.all(np.isfinite(goal))):
        return None, {"reason": "invalid_start_or_goal"}

    direct_distance = float(np.linalg.norm(goal - start))
    if direct_distance < 1e-9:
        return [tuple(start)], {
            "cost": 0.0,
            "length": 0.0,
            "expanded_states": 0,
            "generated_branches": 0,
            "direct_steps": 0,
            "branch_events": 0}

    step_length = max(
        resolution * 0.5,
        float(config.get("step_length_m", resolution)))
    merge_resolution = max(
        resolution * 0.5,
        float(config.get(
            "state_merge_resolution_m",
            min(resolution, step_length * 0.75))))
    angular_step_deg = max(
        1.0, float(config.get("angular_step_deg", 10.0)))
    angular_step_rad = math.radians(angular_step_deg)
    azimuth_bins = max(4, int(config.get("azimuth_bins", 12)))
    max_deviation_deg = min(
        179.0, max(angular_step_deg,
                   float(config.get("maximum_deviation_deg", 120.0))))
    max_deviation_rad = math.radians(max_deviation_deg)
    goal_tolerance = max(
        step_length * 0.5,
        float(config.get("goal_tolerance_m", step_length)))
    max_expanded_states = max(
        1, int(config.get("maximum_expanded_states", 300000)))
    max_planning_time_s = max(
        0.1, float(config.get("maximum_planning_time_s", 45.0)))
    planar_when_equal_altitude = bool(
        config.get("planar_when_equal_altitude", True))
    altitude_tolerance = max(
        0.0, float(config.get(
            "altitude_equality_tolerance_m", 0.05)))
    planar_search = bool(
        planar_when_equal_altitude and
        abs(float(start[2] - goal[2])) <= altitude_tolerance)

    start_clearance = _trilinear_esdf(esdf, origin, resolution, start)
    goal_clearance = _trilinear_esdf(esdf, origin, resolution, goal)
    if start_clearance < min_clearance - 1e-3:
        return None, {
            "reason": "start_below_minimum_clearance",
            "start_clearance": float(start_clearance)}
    if goal_clearance < min_clearance - 1e-3:
        return None, {
            "reason": "goal_below_minimum_clearance",
            "goal_clearance": float(goal_clearance)}

    positions = [start.copy()]
    parents = [-1]
    path_costs = [0.0]
    start_key = _goal_branch_state_key(
        start, origin, merge_resolution)
    best_cost_by_key = {start_key: 0.0}

    open_heap = []
    heap_counter = 0
    heapq.heappush(
        open_heap, (direct_distance, heap_counter, 0))

    best_goal_cost = float("inf")
    best_goal_parent = -1
    expanded_states = 0
    generated_branches = 0
    direct_steps = 0
    branch_events = 0
    stale_states = 0
    started_at = time.time()

    while open_heap:
        if expanded_states >= max_expanded_states:
            break
        if time.time() - started_at > max_planning_time_s:
            break

        estimated_total, _, node_index = heapq.heappop(open_heap)
        if estimated_total >= best_goal_cost - 1e-9:
            break

        position = positions[node_index]
        travelled = path_costs[node_index]
        state_key = _goal_branch_state_key(
            position, origin, merge_resolution)
        if travelled > best_cost_by_key.get(
                state_key, float("inf")) + 1e-9:
            stale_states += 1
            continue

        expanded_states += 1
        to_goal = goal - position
        distance_to_goal = float(np.linalg.norm(to_goal))
        if distance_to_goal <= goal_tolerance:
            candidate_cost = travelled + distance_to_goal
            if candidate_cost < best_goal_cost:
                best_goal_cost = candidate_cost
                best_goal_parent = node_index
            continue

        goal_segment_clear, _ = _check_segment_clearance_adaptive(
            position, goal, esdf, origin, resolution,
            min_clearance, check_spacing)
        if goal_segment_clear:
            candidate_cost = travelled + distance_to_goal
            if candidate_cost < best_goal_cost:
                best_goal_cost = candidate_cost
                best_goal_parent = node_index
            continue

        goal_direction = to_goal / distance_to_goal
        actual_step = min(step_length, distance_to_goal)
        direct_successor = position + actual_step * goal_direction
        direct_clear, _ = _check_segment_clearance_adaptive(
            position, direct_successor,
            esdf, origin, resolution,
            min_clearance, check_spacing)

        successors = []
        if direct_clear:
            successors.append((direct_successor, actual_step))
            direct_steps += 1
        else:
            branch_events += 1
            seen_successor_keys = set()
            direction_rays = _goal_branch_directions(
                goal_direction, angular_step_rad,
                azimuth_bins, max_deviation_rad,
                planar_search=planar_search)
            for ray in direction_rays:
                # In 3-D, each azimuthal ray contributes its minimum-angle
                # action.  In the planar two-sided case, retain the remaining
                # clear angular cells as independent branches as well.  This
                # lets the shortest-route search choose a larger immediate
                # deviation when repeated minimum-angle steps would only
                # oscillate against the same obstacle boundary.
                for _, branch_direction in ray:
                    successor = position + step_length * branch_direction
                    clear, _ = _check_segment_clearance_adaptive(
                        position, successor,
                        esdf, origin, resolution,
                        min_clearance, check_spacing)
                    if not clear:
                        continue
                    successor_key = _goal_branch_state_key(
                        successor, origin, merge_resolution)
                    if successor_key not in seen_successor_keys:
                        seen_successor_keys.add(successor_key)
                        successors.append((successor, step_length))
                    if not planar_search:
                        break
            generated_branches += len(successors)

        for successor, edge_length in successors:
            successor_cost = travelled + edge_length
            successor_key = _goal_branch_state_key(
                successor, origin, merge_resolution)
            if successor_cost >= best_cost_by_key.get(
                    successor_key, float("inf")) - 1e-9:
                continue

            best_cost_by_key[successor_key] = successor_cost
            positions.append(successor)
            parents.append(node_index)
            path_costs.append(successor_cost)
            successor_index = len(positions) - 1
            heuristic = float(np.linalg.norm(goal - successor))
            heap_counter += 1
            heapq.heappush(
                open_heap,
                (successor_cost + heuristic,
                 heap_counter, successor_index))

    if best_goal_parent < 0:
        reason = (
            "maximum_expanded_states_reached"
            if expanded_states >= max_expanded_states
            else "planning_time_limit_reached"
            if time.time() - started_at > max_planning_time_s
            else "no_branch_reaches_goal")
        return None, {
            "reason": reason,
            "expanded_states": expanded_states,
            "generated_branches": generated_branches,
            "direct_steps": direct_steps,
            "branch_events": branch_events,
            "stale_states": stale_states,
            "planning_time_sec": float(time.time() - started_at)}

    reverse_indices = []
    cursor = best_goal_parent
    while cursor >= 0:
        reverse_indices.append(cursor)
        cursor = parents[cursor]
    reverse_indices.reverse()
    path = [tuple(positions[index]) for index in reverse_indices]
    if float(np.linalg.norm(np.asarray(path[-1]) - goal)) > 1e-9:
        path.append(tuple(goal))

    valid, worst_clearance = _validate_polyline(
        np.asarray(path, dtype=np.float64),
        esdf, origin, resolution,
        min_clearance, check_spacing)
    if not valid:
        return None, {
            "reason": "final_path_validation_failed",
            "expanded_states": expanded_states,
            "worst_clearance": float(worst_clearance)}

    length = sum(
        float(np.linalg.norm(
            np.asarray(path[index]) -
            np.asarray(path[index - 1])))
        for index in range(1, len(path)))
    return path, {
        "cost": float(length),
        "length": float(length),
        "worst_clearance": float(worst_clearance),
        "expanded_states": expanded_states,
        "generated_branches": generated_branches,
        "direct_steps": direct_steps,
        "branch_events": branch_events,
        "stale_states": stale_states,
        "search_nodes": len(positions),
        "step_length_m": float(step_length),
        "state_merge_resolution_m": float(merge_resolution),
        "angular_step_deg": float(angular_step_deg),
        "azimuth_bins": int(azimuth_bins),
        "maximum_deviation_deg": float(max_deviation_deg),
        "planar_search": bool(planar_search),
        "planning_time_sec": float(time.time() - started_at)}


def _goal_directed_branch_path_event_length_legacy(
        start, goal, esdf, origin, resolution,
        min_clearance, check_spacing, config):
    """Walk toward the goal and solve each blocked step independently.

    The normal state has exactly one action: one fixed step toward the goal.
    If that step is blocked, a temporary avoidance search branches over clear
    angular steps.  A branch is complete as soon as its next goal-directed
    step is clear.  The shortest completed avoidance segment is committed
    immediately; later obstacles cannot change that decision.  Equal-length
    planar branches prefer the right side.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if (start.shape != (3,) or goal.shape != (3,) or
            not np.all(np.isfinite(start)) or
            not np.all(np.isfinite(goal))):
        return None, {"reason": "invalid_start_or_goal"}

    direct_distance = float(np.linalg.norm(goal - start))
    if direct_distance < 1e-9:
        return [tuple(start)], {
            "cost": 0.0,
            "length": 0.0,
            "expanded_states": 0,
            "generated_branches": 0,
            "direct_steps": 0,
            "branch_events": 0,
            "right_tie_breaks": 0}

    step_length = max(
        resolution * 0.5,
        float(config.get("step_length_m", resolution)))
    merge_resolution = max(
        resolution * 0.5,
        float(config.get(
            "state_merge_resolution_m",
            min(resolution, step_length * 0.75))))
    angular_step_deg = max(
        1.0, float(config.get("angular_step_deg", 10.0)))
    angular_step_rad = math.radians(angular_step_deg)
    azimuth_bins = max(4, int(config.get("azimuth_bins", 12)))
    max_deviation_deg = min(
        179.0, max(
            angular_step_deg,
            float(config.get("maximum_deviation_deg", 120.0))))
    max_deviation_rad = math.radians(max_deviation_deg)
    goal_tolerance = max(
        step_length * 0.5,
        float(config.get("goal_tolerance_m", step_length)))
    max_expanded_states = max(
        1, int(config.get("maximum_expanded_states", 300000)))
    max_planning_time_s = max(
        0.1, float(config.get("maximum_planning_time_s", 45.0)))
    planar_when_equal_altitude = bool(
        config.get("planar_when_equal_altitude", True))
    altitude_tolerance = max(
        0.0, float(config.get(
            "altitude_equality_tolerance_m", 0.05)))
    planar_search = bool(
        planar_when_equal_altitude and
        abs(float(start[2] - goal[2])) <= altitude_tolerance)

    start_clearance = _trilinear_esdf(
        esdf, origin, resolution, start)
    goal_clearance = _trilinear_esdf(
        esdf, origin, resolution, goal)
    if start_clearance < min_clearance - 1e-3:
        return None, {
            "reason": "start_below_minimum_clearance",
            "start_clearance": float(start_clearance)}
    if goal_clearance < min_clearance - 1e-3:
        return None, {
            "reason": "goal_below_minimum_clearance",
            "goal_clearance": float(goal_clearance)}

    path = [start.copy()]
    current = start.copy()
    expanded_states = 0
    generated_branches = 0
    direct_steps = 0
    branch_events = 0
    right_selected_events = 0
    left_selected_events = 0
    other_selected_events = 0
    started_at = time.time()

    while float(np.linalg.norm(goal - current)) > goal_tolerance:
        if expanded_states >= max_expanded_states:
            return None, {
                "reason": "maximum_expanded_states_reached",
                "expanded_states": expanded_states,
                "generated_branches": generated_branches,
                "direct_steps": direct_steps,
                "branch_events": branch_events}
        if time.time() - started_at > max_planning_time_s:
            return None, {
                "reason": "planning_time_limit_reached",
                "expanded_states": expanded_states,
                "generated_branches": generated_branches,
                "direct_steps": direct_steps,
                "branch_events": branch_events}

        # Connecting the complete remaining chord is only a compact form of
        # repeatedly taking the same goal-directed action.
        goal_segment_clear, _ = _check_segment_clearance_adaptive(
            current, goal, esdf, origin, resolution,
            min_clearance, check_spacing)
        if goal_segment_clear:
            path.append(goal.copy())
            current = goal.copy()
            break

        to_goal = goal - current
        distance_to_goal = float(np.linalg.norm(to_goal))
        goal_direction = to_goal / distance_to_goal
        actual_step = min(step_length, distance_to_goal)
        direct_successor = current + actual_step * goal_direction
        direct_clear, _ = _check_segment_clearance_adaptive(
            current, direct_successor,
            esdf, origin, resolution,
            min_clearance, check_spacing)
        if direct_clear:
            path.append(direct_successor.copy())
            current = direct_successor
            direct_steps += 1
            continue

        # A blocked normal action starts one self-contained avoidance event.
        # Search cost is ONLY distance travelled since this event's anchor.
        branch_events += 1
        anchor = current.copy()
        positions = [anchor]
        parents = [-1]
        episode_costs = [0.0]
        first_side_ranks = [2]
        angle_costs = [0.0]
        start_key = _goal_branch_state_key(
            anchor, origin, merge_resolution)
        best_by_key = {start_key: (0.0, 2, 0.0)}
        open_heap = []
        heap_counter = 0
        heapq.heappush(open_heap, (0.0, 2, 0.0, heap_counter, 0))
        selected_index = -1
        selected_side_rank = 2

        while open_heap:
            if expanded_states >= max_expanded_states:
                break
            if time.time() - started_at > max_planning_time_s:
                break

            (travelled, side_rank, angle_cost,
             _, node_index) = heapq.heappop(open_heap)
            position = positions[node_index]
            state_key = _goal_branch_state_key(
                position, origin, merge_resolution)
            state_score = (travelled, side_rank, angle_cost)
            if state_score > best_by_key.get(
                    state_key,
                    (float("inf"), 3, float("inf"))):
                continue

            expanded_states += 1
            to_goal = goal - position
            distance_to_goal = float(np.linalg.norm(to_goal))
            if distance_to_goal <= goal_tolerance:
                selected_index = node_index
                selected_side_rank = side_rank
                break

            goal_direction = to_goal / distance_to_goal
            direct_successor = (
                position +
                min(step_length, distance_to_goal) * goal_direction)
            direct_clear, _ = _check_segment_clearance_adaptive(
                position, direct_successor,
                esdf, origin, resolution,
                min_clearance, check_spacing)

            # The anchor is known to be blocked.  For every other state, one
            # clear goal-directed step ends this avoidance event immediately.
            if node_index != 0 and direct_clear:
                selected_index = node_index
                selected_side_rank = side_rank
                break

            direction_rays = _goal_branch_directions(
                goal_direction, angular_step_rad,
                azimuth_bins, max_deviation_rad,
                planar_search=planar_search)
            seen_successor_keys = set()
            for ray_index, ray in enumerate(direction_rays):
                # Once a planar avoidance branch has chosen RIGHT or LEFT,
                # keep that side for the lifetime of this event.  Allowing a
                # branch to swap sides at every blocked step creates a short
                # alternating zig-zag rather than a coherent bypass.
                if (planar_search and node_index != 0 and
                        ray_index != side_rank):
                    continue
                for deviation, branch_direction in ray:
                    successor = (
                        position + step_length * branch_direction)
                    clear, _ = _check_segment_clearance_adaptive(
                        position, successor,
                        esdf, origin, resolution,
                        min_clearance, check_spacing)
                    if not clear:
                        continue

                    successor_key = _goal_branch_state_key(
                        successor, origin, merge_resolution)
                    if successor_key in seen_successor_keys:
                        continue
                    seen_successor_keys.add(successor_key)

                    if node_index == 0:
                        # _goal_branch_directions emits clockwise/right first
                        # in planar mode.  In 3-D the azimuth order is simply a
                        # deterministic tie-break because "right" is undefined.
                        successor_side_rank = (
                            ray_index if planar_search else ray_index + 2)
                    else:
                        successor_side_rank = side_rank
                    successor_cost = travelled + step_length
                    successor_angle_cost = angle_cost + deviation
                    successor_score = (
                        successor_cost,
                        successor_side_rank,
                        successor_angle_cost)
                    if successor_score >= best_by_key.get(
                            successor_key,
                            (float("inf"), 1000000, float("inf"))):
                        continue

                    best_by_key[successor_key] = successor_score
                    positions.append(successor)
                    parents.append(node_index)
                    episode_costs.append(successor_cost)
                    first_side_ranks.append(successor_side_rank)
                    angle_costs.append(successor_angle_cost)
                    successor_index = len(positions) - 1
                    heap_counter += 1
                    heapq.heappush(
                        open_heap,
                        (successor_cost,
                         successor_side_rank,
                         successor_angle_cost,
                         heap_counter,
                         successor_index))
                    generated_branches += 1

        if selected_index < 0:
            reason = (
                "maximum_expanded_states_reached"
                if expanded_states >= max_expanded_states
                else "planning_time_limit_reached"
                if time.time() - started_at > max_planning_time_s
                else "no_avoidance_branch_restores_goal_step")
            return None, {
                "reason": reason,
                "expanded_states": expanded_states,
                "generated_branches": generated_branches,
                "direct_steps": direct_steps,
                "branch_events": branch_events}

        reverse_indices = []
        cursor = selected_index
        while cursor > 0:
            reverse_indices.append(cursor)
            cursor = parents[cursor]
        reverse_indices.reverse()
        if not reverse_indices:
            return None, {
                "reason": "empty_avoidance_branch",
                "expanded_states": expanded_states,
                "branch_events": branch_events}
        for index in reverse_indices:
            path.append(positions[index].copy())
        current = positions[selected_index].copy()

        if planar_search and selected_side_rank == 0:
            right_selected_events += 1
        elif planar_search and selected_side_rank == 1:
            left_selected_events += 1
        else:
            other_selected_events += 1

    if float(np.linalg.norm(np.asarray(path[-1]) - goal)) > 1e-9:
        path.append(goal.copy())

    valid, worst_clearance = _validate_polyline(
        np.asarray(path, dtype=np.float64),
        esdf, origin, resolution,
        min_clearance, check_spacing)
    if not valid:
        return None, {
            "reason": "final_path_validation_failed",
            "expanded_states": expanded_states,
            "worst_clearance": float(worst_clearance)}

    length = sum(
        float(np.linalg.norm(path[index] - path[index - 1]))
        for index in range(1, len(path)))
    return [tuple(point) for point in path], {
        "cost": float(length),
        "length": float(length),
        "avoidance_score": "segment_length_only",
        "equal_length_tie_break": "right",
        "worst_clearance": float(worst_clearance),
        "expanded_states": expanded_states,
        "generated_branches": generated_branches,
        "direct_steps": direct_steps,
        "branch_events": branch_events,
        "right_selected_events": right_selected_events,
        "left_selected_events": left_selected_events,
        "other_selected_events": other_selected_events,
        "step_length_m": float(step_length),
        "state_merge_resolution_m": float(merge_resolution),
        "angular_step_deg": float(angular_step_deg),
        "azimuth_bins": int(azimuth_bins),
        "maximum_deviation_deg": float(max_deviation_deg),
        "planar_search": bool(planar_search),
        "planning_time_sec": float(time.time() - started_at)}


def _rotate_horizontal(direction, signed_angle):
    """Rotate a 3-D direction in world XY while preserving its elevation."""
    direction = np.asarray(direction, dtype=np.float64)
    horizontal_norm = float(np.linalg.norm(direction[:2]))
    if horizontal_norm < 1e-9:
        return None
    heading = math.atan2(direction[1], direction[0]) + signed_angle
    rotated = np.array([
        horizontal_norm * math.cos(heading),
        horizontal_norm * math.sin(heading),
        direction[2]], dtype=np.float64)
    norm = float(np.linalg.norm(rotated))
    return rotated / norm if norm > 1e-9 else None


def _visible_corridor_clear(position, direction, length,
                            esdf, origin, resolution,
                            min_clearance, check_spacing):
    """Check one finite swept corridor available to the local observation.

    The ESDF already has vehicle radius subtracted, so testing the candidate
    centreline at ``min_clearance`` is equivalent to checking a capsule with
    vehicle radius plus the requested safety margin.  Only the finite segment
    inside the simulated depth range is queried; geometry beyond it cannot
    influence the decision.
    """
    if length <= 1e-9:
        return True, float("inf")
    endpoint = (
        np.asarray(position, dtype=np.float64) +
        float(length) * np.asarray(direction, dtype=np.float64))
    return _check_segment_clearance_adaptive(
        position, endpoint, esdf, origin, resolution,
        min_clearance, check_spacing)


def observation_conditioned_rollout_path(
        start, goal, esdf, origin, resolution,
        min_clearance, check_spacing, config):
    """Generate a global reference by rolling out a local-observation policy.

    The complete ESDF acts as an offline sensor simulator and final safety
    oracle.  Direction decisions query only finite swept corridors inside the
    configured depth range and horizontal FOV.  At the first blocked forward
    corridor, LEFT and RIGHT are compared by their minimum visible safe
    deviation; equal deviations choose RIGHT.  The selected side is retained
    until the forward corridor becomes locally clear again.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if (start.shape != (3,) or goal.shape != (3,) or
            not np.all(np.isfinite(start)) or
            not np.all(np.isfinite(goal))):
        return None, {"reason": "invalid_start_or_goal"}

    direct_distance = float(np.linalg.norm(goal - start))
    if direct_distance < 1e-9:
        return [tuple(start)], {
            "cost": 0.0,
            "length": 0.0,
            "decision_events": [],
            "right_selected_events": 0,
            "left_selected_events": 0}

    step_length = max(
        resolution * 0.5,
        float(config.get("step_length_m", resolution)))
    angular_step_deg = max(
        0.25, float(config.get("angular_step_deg", 2.0)))
    angular_step_rad = math.radians(angular_step_deg)
    deviation_tie_tolerance_deg = max(
        0.0, float(config.get(
            "deviation_tie_tolerance_deg", angular_step_deg)))
    deviation_tie_tolerance_rad = math.radians(
        deviation_tie_tolerance_deg)
    observation_range = max(
        step_length,
        float(config.get("observation_range_m", 5.0)))
    horizontal_fov_deg = min(
        179.0, max(
            1.0, float(config.get("horizontal_fov_deg", 90.0))))
    fov_margin_deg = max(
        0.0, float(config.get("fov_margin_deg", 3.0)))
    visible_half_fov_deg = max(
        angular_step_deg,
        horizontal_fov_deg * 0.5 - fov_margin_deg)
    maximum_deviation_deg = min(
        179.0,
        max(
            angular_step_deg,
            float(config.get("maximum_deviation_deg", 120.0))))
    maximum_angle_index = max(
        1, int(math.floor(
            maximum_deviation_deg / angular_step_deg + 1e-9)))
    goal_tolerance = max(
        step_length * 0.5,
        float(config.get("goal_tolerance_m", step_length)))
    maximum_rollout_steps = max(
        1, int(config.get("maximum_rollout_steps", 10000)))
    maximum_planning_time_s = max(
        0.1, float(config.get("maximum_planning_time_s", 45.0)))

    start_clearance = _trilinear_esdf(
        esdf, origin, resolution, start)
    goal_clearance = _trilinear_esdf(
        esdf, origin, resolution, goal)
    if start_clearance < min_clearance - 1e-3:
        return None, {
            "reason": "start_below_minimum_clearance",
            "start_clearance": float(start_clearance)}
    if goal_clearance < min_clearance - 1e-3:
        return None, {
            "reason": "goal_below_minimum_clearance",
            "goal_clearance": float(goal_clearance)}

    path = [start.copy()]
    current = start.copy()
    camera_direction = (goal - start) / direct_distance
    committed_side = None
    commitment_anchor = None
    commitment_right_normal = None
    decision_events = []
    right_selected_events = 0
    left_selected_events = 0
    direct_steps = 0
    avoidance_steps = 0
    corridor_checks = 0
    started_at = time.time()

    def corridor_length(distance_to_goal):
        return min(observation_range, distance_to_goal)

    def direction_in_camera_fov(direction):
        candidate_heading = math.atan2(direction[1], direction[0])
        camera_heading = math.atan2(
            camera_direction[1], camera_direction[0])
        delta = math.atan2(
            math.sin(candidate_heading - camera_heading),
            math.cos(candidate_heading - camera_heading))
        return abs(math.degrees(delta)) <= visible_half_fov_deg + 1e-9

    def minimum_safe_deviation(position, goal_direction,
                               distance_to_goal, side):
        nonlocal corridor_checks
        side_sign = -1.0 if side == "RIGHT" else 1.0
        length = corridor_length(distance_to_goal)
        for angle_index in range(1, maximum_angle_index + 1):
            deviation = angle_index * angular_step_rad
            direction = _rotate_horizontal(
                goal_direction, side_sign * deviation)
            if direction is None:
                continue
            if not direction_in_camera_fov(direction):
                continue
            corridor_checks += 1
            clear, worst = _visible_corridor_clear(
                position, direction, length,
                esdf, origin, resolution,
                min_clearance, check_spacing)
            if not clear:
                continue
            # Validate the actually executed prefix independently. Sampling a
            # long corridor and a 0.1 m prefix uses different sample phases;
            # both must accept the direction at the hard boundary.
            prefix_length = min(step_length, distance_to_goal)
            prefix_clear, prefix_worst = _visible_corridor_clear(
                position, direction, prefix_length,
                esdf, origin, resolution,
                min_clearance, check_spacing)
            if prefix_clear:
                return (
                    deviation, direction,
                    min(float(worst), float(prefix_worst)))
        return None, None, None

    def minimum_safe_committed_direction(
            position, goal_direction, distance_to_goal):
        """Find the smallest turn that stays on the committed path side."""
        nonlocal corridor_checks
        length = corridor_length(distance_to_goal)
        preferred_sign = -1.0 if committed_side == "RIGHT" else 1.0
        for angle_index in range(1, maximum_angle_index + 1):
            deviation = angle_index * angular_step_rad
            for side_sign in (preferred_sign, -preferred_sign):
                direction = _rotate_horizontal(
                    goal_direction, side_sign * deviation)
                if direction is None:
                    continue
                if not direction_in_camera_fov(direction):
                    continue
                prefix_length = min(step_length, distance_to_goal)
                successor = position + prefix_length * direction
                lateral = float(np.dot(
                    successor[:2] - commitment_anchor[:2],
                    commitment_right_normal))
                if committed_side == "RIGHT" and lateral < -1e-6:
                    continue
                if committed_side == "LEFT" and lateral > 1e-6:
                    continue

                corridor_checks += 1
                clear, worst = _visible_corridor_clear(
                    position, direction, length,
                    esdf, origin, resolution,
                    min_clearance, check_spacing)
                if not clear:
                    continue
                prefix_clear, prefix_worst = _visible_corridor_clear(
                    position, direction, prefix_length,
                    esdf, origin, resolution,
                    min_clearance, check_spacing)
                if prefix_clear:
                    return (
                        deviation, direction,
                        min(float(worst), float(prefix_worst)))
        return None, None, None

    for rollout_step in range(maximum_rollout_steps):
        if time.time() - started_at > maximum_planning_time_s:
            return None, {
                "reason": "planning_time_limit_reached",
                "rollout_steps": rollout_step,
                "decision_events": decision_events,
                "corridor_checks": corridor_checks}

        to_goal = goal - current
        distance_to_goal = float(np.linalg.norm(to_goal))
        if distance_to_goal <= goal_tolerance:
            path.append(goal.copy())
            current = goal.copy()
            break
        goal_direction = to_goal / distance_to_goal
        local_length = corridor_length(distance_to_goal)
        forward_clear = False
        if direction_in_camera_fov(goal_direction):
            corridor_checks += 1
            forward_clear, _ = _visible_corridor_clear(
                current, goal_direction, local_length,
                esdf, origin, resolution,
                min_clearance, check_spacing)
        if forward_clear:
            prefix_clear, _ = _visible_corridor_clear(
                current, goal_direction,
                min(step_length, distance_to_goal),
                esdf, origin, resolution,
                min_clearance, check_spacing)
            forward_clear = bool(prefix_clear)

        if forward_clear:
            committed_side = None
            commitment_anchor = None
            commitment_right_normal = None
            direction = goal_direction
            direct_steps += 1
        else:
            if committed_side is None:
                right = minimum_safe_deviation(
                    current, goal_direction, distance_to_goal, "RIGHT")
                left = minimum_safe_deviation(
                    current, goal_direction, distance_to_goal, "LEFT")
                right_angle, right_direction, right_worst = right
                left_angle, left_direction, left_worst = left

                if right_angle is None and left_angle is None:
                    return None, {
                        "reason": "no_visible_safe_avoidance_direction",
                        "rollout_steps": rollout_step,
                        "position": [float(v) for v in current],
                        "decision_events": decision_events,
                        "corridor_checks": corridor_checks}
                if right_angle is None:
                    committed_side = "LEFT"
                    direction = left_direction
                    selection_reason = "right_unavailable"
                elif left_angle is None:
                    committed_side = "RIGHT"
                    direction = right_direction
                    selection_reason = "left_unavailable"
                elif (left_angle <
                      right_angle - deviation_tie_tolerance_rad - 1e-12):
                    committed_side = "LEFT"
                    direction = left_direction
                    selection_reason = "smaller_visible_deviation"
                elif (right_angle <
                      left_angle - deviation_tie_tolerance_rad - 1e-12):
                    committed_side = "RIGHT"
                    direction = right_direction
                    selection_reason = "smaller_visible_deviation"
                else:
                    # One angular bin is the measurement resolution, not a
                    # meaningful geometric difference. RIGHT wins such ties.
                    committed_side = "RIGHT"
                    direction = right_direction
                    selection_reason = (
                        "equal_visible_deviation_choose_right")

                if committed_side == "RIGHT":
                    right_selected_events += 1
                else:
                    left_selected_events += 1
                commitment_anchor = current.copy()
                forward_xy = goal_direction[:2]
                forward_xy_norm = float(np.linalg.norm(forward_xy))
                if forward_xy_norm < 1e-9:
                    return None, {
                        "reason": "horizontal_side_choice_is_undefined",
                        "rollout_steps": rollout_step,
                        "decision_events": decision_events}
                forward_xy = forward_xy / forward_xy_norm
                commitment_right_normal = np.array(
                    [forward_xy[1], -forward_xy[0]],
                    dtype=np.float64)
                decision_events.append({
                    "path_index": len(path) - 1,
                    "position": [
                        round(float(v), 4) for v in current],
                    "right_min_deviation_deg": (
                        None if right_angle is None
                        else round(math.degrees(right_angle), 4)),
                    "left_min_deviation_deg": (
                        None if left_angle is None
                        else round(math.degrees(left_angle), 4)),
                    "right_corridor_worst_clearance": (
                        None if right_worst is None
                        else round(float(right_worst), 4)),
                    "left_corridor_worst_clearance": (
                        None if left_worst is None
                        else round(float(left_worst), 4)),
                    "selected_side": committed_side,
                    "selection_reason": selection_reason})
            else:
                deviation = minimum_safe_committed_direction(
                    current, goal_direction, distance_to_goal)
                _, direction, _ = deviation
                if direction is None:
                    return None, {
                        "reason": "committed_side_has_no_visible_safe_direction",
                        "committed_side": committed_side,
                        "rollout_steps": rollout_step,
                        "position": [float(v) for v in current],
                        "decision_events": decision_events,
                        "corridor_checks": corridor_checks}
            avoidance_steps += 1

        actual_step = min(step_length, distance_to_goal)
        successor = current + actual_step * direction
        step_clear, step_worst = _check_segment_clearance_adaptive(
            current, successor, esdf, origin, resolution,
            min_clearance, check_spacing)
        if not step_clear:
            return None, {
                "reason": "selected_visible_direction_failed_truth_check",
                "rollout_steps": rollout_step,
                "step_worst_clearance": float(step_worst),
                "decision_events": decision_events}
        path.append(successor.copy())
        current = successor
        camera_direction = direction.copy()
    else:
        return None, {
            "reason": "maximum_rollout_steps_reached",
            "rollout_steps": maximum_rollout_steps,
            "decision_events": decision_events,
            "corridor_checks": corridor_checks}

    if float(np.linalg.norm(np.asarray(path[-1]) - goal)) > 1e-9:
        path.append(goal.copy())
    valid, worst_clearance = _validate_polyline(
        np.asarray(path, dtype=np.float64),
        esdf, origin, resolution,
        min_clearance, check_spacing)
    if not valid:
        return None, {
            "reason": "final_path_validation_failed",
            "worst_clearance": float(worst_clearance),
            "decision_events": decision_events}

    length = sum(
        float(np.linalg.norm(path[index] - path[index - 1]))
        for index in range(1, len(path)))
    return [tuple(point) for point in path], {
        "cost": float(length),
        "length": float(length),
        "policy": "minimum_visible_safe_deviation",
        "equal_deviation_tie_break": "right",
        "observation_source": "finite_esdf_swept_corridors",
        "observation_range_m": float(observation_range),
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "fov_margin_deg": float(fov_margin_deg),
        "maximum_deviation_deg": float(maximum_deviation_deg),
        "angular_step_deg": float(angular_step_deg),
        "deviation_tie_tolerance_deg": float(
            deviation_tie_tolerance_deg),
        "step_length_m": float(step_length),
        "worst_clearance": float(worst_clearance),
        "rollout_steps": len(path) - 1,
        "direct_steps": direct_steps,
        "avoidance_steps": avoidance_steps,
        "corridor_checks": corridor_checks,
        "right_selected_events": right_selected_events,
        "left_selected_events": left_selected_events,
        "decision_events": decision_events,
        "planning_time_sec": float(time.time() - started_at)}


# ============================================================================
#  6.  Legacy LinePush implementation (not used by GlobalPathPlanner)
# ============================================================================
#  Retained for compatibility with old offline tools. Runtime global planning
#  no longer calls this implementation.
#
#  Algorithm:
#    1. Generate initial control points on the straight line start→goal.
#    2. Group consecutive control points with ESDF < 0.3 into violation intervals.
#    3. For each interval, generate LEFT and RIGHT candidate paths.
#    4. Compare complete candidate-path cost; select lower side (RIGHT on tie).
#    5. For every adjacent pair of safe control points, sample the segment at
#       check_spacing; if any interior sample has ESDF < 0.3, insert a NEW
#       control point at the worst location.  NEVER move the two safe endpoints.
#    6. Only newly-inserted control points (tagged "segment_insert") may be pushed.
#       Original safe control points ("original") are IMMUTABLE.
#    7. Iterate until the full polyline passes ESDF ≥ 0.3 everywhere.
#    8. On failure: return None.  No fallback, no A*, no unsafe bypass.
#
#  The ESDF threshold 0.3 is the ONLY safety criterion.
# ============================================================================

# Global ESDF safety threshold — THE ONLY threshold used across all modules.
_GLOBAL_ESDF_THRESHOLD = 0.3


def _line_push_cost(path, esdf, origin, resolution):
    """Compute geometric length for collision-side selection.

    Clearance is a hard feasibility constraint, not a reward.  Smoothness and
    dynamics belong to the local trajectory planner; including a curvature
    term here can make voxel-gradient noise select a geometrically longer side
    and would contaminate the intended pure collision-avoidance reference.
    """
    if len(path) < 2:
        return float("inf")
    length = 0.0
    for i in range(1, len(path)):
        length += float(np.linalg.norm(path[i] - path[i - 1]))
    return length


def _line_push_interval_cost(path, interval_indices, esdf, origin,
                             resolution):
    """Cost the complete candidate after changing one collision interval.

    Unchanged intervals are identical between the left and right candidates,
    so their contribution cancels in the comparison.  Keeping the complete
    prefix/suffix in the objective is nevertheless essential: it accounts for
    the distance needed to reconnect the repaired interval to the final goal
    instead of making a myopic decision from only two adjacent samples.
    """
    if path is None or not interval_indices:
        return float("inf")
    return _line_push_cost(path, esdf, origin, resolution)


def _compute_esdf_gradient(point, esdf, origin, resolution, eps=0.10):
    """Compute ESDF gradient at a world point via central differences.

    Gradient points in the direction of INCREASING ESDF — away from obstacles,
    toward free space.  This is the natural "escape direction".

    Args:
        point: (3,) numpy array, world coordinates
        esdf, origin, resolution: ESDF query parameters
        eps: finite-difference step size (metres)

    Returns:
        (3,) numpy array — unit-norm gradient direction, or zero if flat.
    """
    grad = np.zeros(3, dtype=np.float64)
    for d in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[d] = eps
        fwd = _trilinear_esdf(esdf, origin, resolution, point + offset)
        bwd = _trilinear_esdf(esdf, origin, resolution, point - offset)
        grad[d] = (fwd - bwd) / (2.0 * eps)
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return grad / grad_norm


def _gradient_push_side(start_point, esdf, origin, resolution,
                         left_normal, side_sign, threshold=0.3,
                         max_steps=500, step_size=0.15,
                         max_push_distance=0.0):
    """Push a point to safety using the ESDF gradient, biased to one side.

    The gradient points toward higher ESDF (free space).  This function
    follows it preferentially, falling back to pure side-direction push
    when the gradient is ambiguous or temporarily points the wrong way.

    KEY: the gradient determines the PREFERRED direction but does NOT
    gate the push.  Local gradient fluctuations (common when navigating
    around multiple obstacles) do not cause premature failure.

    Args:
        start_point: (3,) world coords of the violating point
        left_normal: unit vector pointing LEFT of the local anchor→goal line
        side_sign: +1.0 for LEFT, -1.0 for RIGHT
        threshold: ESDF value to reach (0.3)
        max_steps: iteration limit
        step_size: metres per step
        max_push_distance: maximum displacement from the seed; <=0 disables

    Returns:
        (final_point, success, diagnostics)
    """
    current = start_point.copy().astype(np.float64)
    initial_point = current.copy()
    side_dir = side_sign * left_normal  # +left_normal or -left_normal

    initial_cl = _trilinear_esdf(
        esdf, origin, resolution, current)
    best_cl = initial_cl
    best_pos = current.copy()
    steps_since_improvement = 0

    def finish(position, success, reason, steps):
        return position, success, {
            "reason": reason,
            "steps": int(steps),
            "initial_clearance": float(initial_cl),
            "best_clearance": float(best_cl),
            "displacement": float(np.linalg.norm(
                np.asarray(position) - initial_point)),
        }

    for step_i in range(max_steps):
        cl = _trilinear_esdf(esdf, origin, resolution, current)

        # Track best position seen
        if cl > best_cl:
            best_cl = cl
            best_pos = current.copy()
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        # Success
        if cl >= threshold:
            return finish(
                current, True, "reached_clearance", step_i)

        # Stuck detection: no improvement for many steps
        if steps_since_improvement > 80:
            return finish(
                best_pos, best_cl >= threshold,
                "no_clearance_improvement", step_i)

        # ── ESDF gradient: steepest ascent toward free space ──
        grad = _compute_esdf_gradient(current, esdf, origin, resolution,
                                       eps=0.10)
        grad_norm = float(np.linalg.norm(grad))

        if grad_norm < 1e-9:
            # Flat ESDF — pure side direction push
            step_dir = side_dir.copy()
        else:
            side_component = float(np.dot(grad, side_dir))

            if side_component > 0.05:
                # Gradient strongly agrees — follow it directly
                step_dir = grad.copy()
            elif side_component > -0.05:
                # Gradient is roughly parallel to path or weakly opposing
                # Blend: mostly side direction, some gradient influence
                blend = 0.7 * side_dir + 0.3 * grad
                blend_norm = float(np.linalg.norm(blend))
                step_dir = (blend / blend_norm) if blend_norm > 1e-9 else side_dir.copy()
            else:
                # Gradient strongly opposes — but don't give up!
                # Push pure side direction; the local gradient may be from
                # a nearby obstacle surface that we need to push past.
                step_dir = side_dir.copy()

        current = current + step_size * step_dir

        # Deeply colliding seeds need multiple steps before they become safe.
        # Bound displacement from the seed instead of rejecting on the current
        # signed distance alone.
        if (max_push_distance > 0.0 and
                float(np.linalg.norm(current - initial_point)) >
                max_push_distance):
            return finish(
                best_pos, False, "max_push_distance_exceeded",
                step_i + 1)

    # Exhausted steps — return best position found
    final_cl = _trilinear_esdf(
        esdf, origin, resolution, current)
    if final_cl > best_cl:
        best_cl = final_cl
        best_pos = current.copy()
    if final_cl >= threshold:
        return finish(
            current, True, "reached_clearance", max_steps)
    return finish(
        best_pos, False, "max_push_steps_exhausted", max_steps)


def _generate_left_right_candidates(violation_indices, all_points, original_tags,
                                     delta, left_normal, esdf, origin, resolution,
                                     max_push_distance, max_push_steps,
                                     push_clearance):
    """Generate LEFT and RIGHT candidate paths for a violation interval.

    The start→goal straight line divides the plane into left and right.
    For each violating point:
      - LEFT  candidate: follow the gradient ONLY if it points leftward.
      - RIGHT candidate: follow the gradient ONLY if it points rightward.
      - If the gradient has equal components on both sides → RIGHT wins.

    No blending weights, no preference mixing — the ESDF gradient alone
    determines the viable escape direction(s).

    Args:
        violation_indices: list of consecutive indices where ESDF < 0.3
        all_points: numpy array [N, 3] of all control points
        original_tags: list of str, "original" or "segment_insert"
        delta: start→goal vector (unused, kept for interface compat)
        left_normal: left-pointing unit vector in XY plane
        esdf, origin, resolution: ESDF query parameters
        max_push_distance: maximum displacement of each pushed point
        max_push_steps: max gradient-ascent steps per point
        push_clearance: ESDF target for moved points (hard limit + margin)

    Returns:
        (left_path, right_path) — each is None if that side failed.
    """
    n = len(all_points)
    left_path = all_points.copy()
    right_path = all_points.copy()

    left_ok = True
    right_ok = True
    first_left_failure = None
    first_right_failure = None

    for idx in violation_indices:
        if original_tags[idx] not in ("segment_insert", "original_violating"):
            continue

        base_point = all_points[idx].copy()

        # ── ALWAYS try both sides.  The ESDF gradient is informative
        #     but can be misleading at a single point (e.g. narrow corridor
        #     where the nearest wall biases the gradient).  The push itself
        #     will determine which side actually reaches safe ESDF.
        #     Per spec: if both succeed with equal cost → RIGHT wins.

        # ── LEFT candidate ──────────────────────────────────────
        left_safe, left_success, left_report = _gradient_push_side(
            base_point, esdf, origin, resolution,
            left_normal, side_sign=+1.0,
            threshold=push_clearance,
            max_steps=max_push_steps,
            max_push_distance=max_push_distance)
        if left_success:
            left_path[idx] = left_safe
        else:
            left_ok = False
            if first_left_failure is None:
                first_left_failure = (idx, left_report)

        # ── RIGHT candidate ─────────────────────────────────────
        right_safe, right_success, right_report = _gradient_push_side(
            base_point, esdf, origin, resolution,
            left_normal, side_sign=-1.0,
            threshold=push_clearance,
            max_steps=max_push_steps,
            max_push_distance=max_push_distance)
        if right_success:
            right_path[idx] = right_safe
        else:
            right_ok = False
            if first_right_failure is None:
                first_right_failure = (idx, right_report)

    if not left_ok:
        idx, report = first_left_failure
        rospy.logwarn(
            "[LinePush] LEFT candidate failed at point=%d: reason=%s "
            "steps=%d initial_esdf=%.3f best_esdf=%.3f offset=%.2fm",
            idx, report["reason"], report["steps"],
            report["initial_clearance"], report["best_clearance"],
            report["displacement"])
        left_path = None
    if not right_ok:
        idx, report = first_right_failure
        rospy.logwarn(
            "[LinePush] RIGHT candidate failed at point=%d: reason=%s "
            "steps=%d initial_esdf=%.3f best_esdf=%.3f offset=%.2fm",
            idx, report["reason"], report["steps"],
            report["initial_clearance"], report["best_clearance"],
            report["displacement"])
        right_path = None

    return left_path, right_path


def line_push_path(start, goal, esdf, origin, resolution,
                   min_clearance, check_spacing, config):
    """THE ONLY global path planning algorithm.

    Straight-line initial control points + ESDF violation bilateral push.
    Original safe control points are never displaced or bypassed.  This keeps
    every collision-free prefix exactly on its current anchor→goal ray and
    confines all lateral motion to actual collision intervals.
    Segment-inserted control points are pushed left/right only.
    No A*, no RRT, no smoothing, no shortcut.

    Args:
        start, goal: (x, y, z) tuples or arrays
        esdf: 3D numpy array
        origin: (ox, oy, oz)
        resolution: voxel size
        min_clearance: ESDF threshold (MUST be 0.3)
        check_spacing: segment sampling spacing
        config: line_push configuration dict

    Returns:
        (path_list, report_dict) on success, (None, report) on failure.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return None, {"reason": "degenerate_start_goal"}

    # ── Quick direct check ────────────────────────────────────────
    direct_clear, direct_worst = _check_segment_clearance_adaptive(
        start, goal, esdf, origin, resolution,
        min_clearance, check_spacing)
    if direct_clear:
        return [tuple(start), tuple(goal)], {
            "side": "direct", "cost": distance,
            "worst_clearance": direct_worst, "iterations": 0}

    # ── Config params ─────────────────────────────────────────────
    point_spacing = max(
        resolution * 0.5, float(config.get("point_spacing_m", 0.10)))
    max_iterations = max(1, int(config.get("max_iterations", 80)))
    max_control_points = max(
        16, int(config.get("max_control_points", 5000)))
    max_push_distance = max(
        0.0, float(config.get("max_offset_m", 12.0)))
    push_clearance = (
        min_clearance +
        max(0.0, float(config.get("push_margin_m", 0.0))))
    # Gradient-ascent push steps: each step is 0.15 m, 300 steps = 45 m reach.
    # No distance limit — push continues until ESDF >= 0.3 or steps exhausted.
    max_push_steps = max(100, int(config.get("max_iterations", 80)) * 3)

    # ── Generate initial control points on straight line ──────────
    # min_clearance is the ESDF threshold (0.3); use it directly.
    esdf_threshold = min_clearance  # 0.3

    point_count = max(3, int(math.ceil(distance / point_spacing)) + 1)
    fractions = np.linspace(0.0, 1.0, point_count)
    base = start[None, :] + fractions[:, None] * delta[None, :]
    base[0] = start
    base[-1] = goal

    # Tags: classify original points by their ESDF
    #   "original_safe"      — ESDF >= 0.3, NEVER move
    #   "original_violating" — ESDF < 0.3, pushable (the point is inside an obstacle)
    #   "segment_insert"     — inserted later, pushable
    tags = []
    for i in range(point_count):
        cl = _trilinear_esdf(esdf, origin, resolution, base[i])
        if cl >= esdf_threshold:
            tags.append("original_safe")
        else:
            tags.append("original_violating")

    n_orig_safe = sum(1 for t in tags if t == "original_safe")
    n_orig_violating = sum(1 for t in tags if t == "original_violating")
    rospy.loginfo("[LinePush] %d initial pts: %d safe (immutable), %d violating (pushable)",
                  point_count, n_orig_safe, n_orig_violating)

    # The initial normal only verifies that this 2.5-D bilateral planner has a
    # horizontal planning direction.  Each obstacle interval recomputes its
    # own normal from the current prefix anchor to the final goal.
    xy_norm = float(np.linalg.norm(delta[:2]))
    if xy_norm < 1e-6:
        return None, {"reason": "vertical_line_has_no_horizontal_normal"}

    # ── Main iteration loop ───────────────────────────────────────
    path = base.copy()
    total_segment_insertions = 0
    best_worst = float("-inf")

    for iteration in range(max_iterations):
        n = len(path)

        # 1. Validate all points
        valid, worst = _validate_polyline(
            path, esdf, origin, resolution,
            esdf_threshold, check_spacing)
        if valid:
            path_list = [tuple(point) for point in path]
            length = sum(float(np.linalg.norm(
                path[i] - path[i - 1])) for i in range(1, len(path)))
            cost = _line_push_cost(path, esdf, origin, resolution)
            rospy.loginfo("[LinePush] converged: iter=%d pts=%d len=%.1fm cost=%.2f",
                          iteration, len(path), length, cost)
            return path_list, {
                "side": "bilateral_push",
                "cost": float(cost),
                "length": float(length),
                "worst_clearance": float(worst),
                "iterations": iteration,
                "control_points": len(path),
                "segment_insertions": total_segment_insertions}

        # 2. Identify violation points (ESDF < threshold)
        violation_flags = [False] * n
        for i in range(n):
            cl = _trilinear_esdf(esdf, origin, resolution, path[i])
            if cl < esdf_threshold:
                violation_flags[i] = True

        # 3. Find continuously unsafe segments.  Insert their actual worst
        #    interior sample so the new segment_insert point is violating and
        #    therefore gets pushed on the next iteration.
        insertions_needed = []
        for i in range(n - 1):
            seg_clear, seg_worst = _check_segment_clearance_adaptive(
                path[i], path[i + 1], esdf, origin, resolution,
                esdf_threshold, check_spacing)
            if seg_clear:
                continue

            cl_i = _trilinear_esdf(esdf, origin, resolution, path[i])
            cl_j = _trilinear_esdf(esdf, origin, resolution, path[i + 1])

            # Two violating endpoints already belong to a push interval.
            both_violating = (cl_i < esdf_threshold and cl_j < esdf_threshold)
            if both_violating:
                continue

            # Do not insert an ESDF ~= threshold crossing: float interpolation
            # can classify it as safe, causing endless subdivision without a
            # point that is eligible for pushing.
            seg_vec = path[i + 1] - path[i]
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len < 1e-9:
                continue

            sample_steps = max(
                2, int(math.ceil(seg_len / check_spacing)) + 1)
            worst_alpha = None
            worst_point = None
            worst_clearance = float("inf")
            for sample_index in range(1, sample_steps):
                alpha = sample_index / float(sample_steps)
                sample = path[i] + alpha * seg_vec
                clearance = _trilinear_esdf(
                    esdf, origin, resolution, sample)
                if clearance < worst_clearance:
                    worst_clearance = clearance
                    worst_alpha = alpha
                    worst_point = sample

            if (worst_point is not None and
                    worst_clearance < esdf_threshold):
                insertions_needed.append(
                    (i, worst_alpha, worst_point, worst_clearance))

        if not insertions_needed and not any(violation_flags):
            # No violations found, but polyline check failed — unexpected
            break

        # 4. Handle segment insertions first (insert new control points)
        if insertions_needed and len(path) < max_control_points:
            # Sort by segment index (descending so insertions don't shift indices)
            insertions_needed.sort(key=lambda x: x[0], reverse=True)
            # Deduplicate: at most one insertion per segment
            seen_segments = set()
            unique_insertions = []
            for (seg_idx, alpha, point, worst_cl) in insertions_needed:
                if seg_idx not in seen_segments:
                    seen_segments.add(seg_idx)
                    unique_insertions.append((seg_idx, alpha, point, worst_cl))
            unique_insertions.sort(key=lambda x: x[0], reverse=True)

            for (seg_idx, alpha, point, worst_cl) in unique_insertions:
                if len(path) >= max_control_points:
                    break
                path = np.insert(path, seg_idx + 1, point, axis=0)
                tags.insert(seg_idx + 1, "segment_insert")
                total_segment_insertions += 1

            # After insertions, re-validate
            continue

        # 5. Find violation intervals — pushable points are:
        #    "segment_insert" and "original_violating"
        #    "original_safe" is IMMUTABLE and breaks intervals.
        PUSHABLE_TAGS = {"segment_insert", "original_violating"}
        violation_intervals = []
        in_violation = False
        current_interval = []
        for i in range(len(path)):
            cl = _trilinear_esdf(esdf, origin, resolution, path[i])
            is_violating = (cl < esdf_threshold)
            is_pushable = (tags[i] in PUSHABLE_TAGS)
            if is_violating and is_pushable:
                if not in_violation:
                    in_violation = True
                    current_interval = [i]
                else:
                    current_interval.append(i)
            else:
                if in_violation:
                    in_violation = False
                    violation_intervals.append(list(current_interval))
                    current_interval = []
        if in_violation:
            violation_intervals.append(list(current_interval))

        if not violation_intervals:
            n_violating = sum(1 for i in range(len(path))
                              if _trilinear_esdf(esdf, origin, resolution, path[i]) < esdf_threshold)
            n_safe_violating = sum(1 for i in range(len(path))
                                   if tags[i] == "original_safe"
                                   and _trilinear_esdf(esdf, origin, resolution, path[i]) < esdf_threshold)
            rospy.logwarn("[LinePush] iter=%d: no pushable intervals. "
                          "total_violations=%d safe_immutable_violating=%d pts=%d insertions=%d",
                          iteration, n_violating, n_safe_violating,
                          len(path), total_segment_insertions)
            break

        # Diagnostic: first and every 10th iteration
        if iteration == 0 or iteration % 10 == 0:
            n_viol = sum(1 for iv in violation_intervals for _ in iv)
            rospy.loginfo("[LinePush] iter=%d: %d intervals, %d violating pts, "
                          "path=%d pts, %d insertions total",
                          iteration, len(violation_intervals), n_viol,
                          len(path), total_segment_insertions)

        # 6. Process each violation interval — bilateral push
        pushes_attempted = 0
        pushes_left_win = 0
        pushes_right_win = 0
        pushes_both_fail = 0
        for interval_indices in violation_intervals:
            # Receding anchor: define LEFT/RIGHT from the last fixed point
            # before this collision interval toward the final goal, not from
            # the episode's original start->goal line.
            anchor_index = max(0, int(interval_indices[0]) - 1)
            local_delta = goal - path[anchor_index]
            local_xy_norm = float(np.linalg.norm(local_delta[:2]))
            if local_xy_norm < 1e-6:
                rospy.logerr(
                    "[LinePush] interval=[%d,%d] has no horizontal "
                    "anchor->goal direction.",
                    interval_indices[0], interval_indices[-1])
                return None, {
                    "reason": "interval_anchor_has_no_horizontal_direction",
                    "interval": interval_indices,
                    "iteration": iteration}
            local_left_normal = np.array(
                [-local_delta[1] / local_xy_norm,
                 local_delta[0] / local_xy_norm,
                 0.0],
                dtype=np.float64)

            left_candidate, right_candidate = _generate_left_right_candidates(
                interval_indices, path, tags,
                local_delta, local_left_normal, esdf, origin, resolution,
                max_push_distance,
                max_push_steps,
                push_clearance)

            pushes_attempted += 1
            # Compare complete candidates.  Other not-yet-repaired intervals
            # are unchanged between both branches and therefore cancel, while
            # the full prefix/suffix captures reconnection distance to goal.
            left_cost = _line_push_interval_cost(
                left_candidate, interval_indices,
                esdf, origin, resolution)
            right_cost = _line_push_interval_cost(
                right_candidate, interval_indices,
                esdf, origin, resolution)

            if left_cost == float("inf") and right_cost == float("inf"):
                # Both sides failed — global path planning fails
                pushes_both_fail += 1
                rospy.logerr("[LinePush] iter=%d interval=%d pts: BOTH sides failed",
                             iteration, len(interval_indices))
                return None, {
                    "reason": "bilateral_push_both_sides_failed",
                    "interval": interval_indices,
                    "iteration": iteration}

            chosen_side = ""
            if left_cost == float("inf"):
                path = right_candidate
                pushes_right_win += 1
                chosen_side = "RIGHT"
            elif right_cost == float("inf"):
                path = left_candidate
                pushes_left_win += 1
                chosen_side = "LEFT"
            elif abs(left_cost - right_cost) <= 1e-6:
                # Equal cost → choose RIGHT (per spec)
                path = right_candidate
                pushes_right_win += 1
                chosen_side = "RIGHT"
            elif left_cost < right_cost:
                path = left_candidate
                pushes_left_win += 1
                chosen_side = "LEFT"
            else:
                path = right_candidate
                pushes_right_win += 1
                chosen_side = "RIGHT"
            rospy.loginfo(
                "[LinePush] iter=%d interval=[%d,%d] anchor=%d global_cost "
                "left=%.3f right=%.3f chosen=%s",
                iteration, interval_indices[0], interval_indices[-1],
                anchor_index, left_cost, right_cost, chosen_side)

        if iteration == 0 or iteration % 10 == 0:
            rospy.loginfo("[LinePush] iter=%d push results: left=%d right=%d both_fail=%d",
                          iteration, pushes_left_win, pushes_right_win, pushes_both_fail)

        # Re-validate after all interval pushes
        valid, worst = _validate_polyline(
            path, esdf, origin, resolution,
            esdf_threshold, check_spacing)
        if valid:
            path_list = [tuple(point) for point in path]
            length = sum(float(np.linalg.norm(
                path[i] - path[i - 1])) for i in range(1, len(path)))
            cost = _line_push_cost(path, esdf, origin, resolution)
            return path_list, {
                "side": "bilateral_push",
                "cost": float(cost),
                "length": float(length),
                "worst_clearance": float(worst),
                "iterations": iteration + 1,
                "control_points": len(path),
                "segment_insertions": total_segment_insertions}

        best_worst = max(best_worst, worst)

    # ── Max iterations reached without success ────────────────────
    # Diagnostic: count remaining violations
    n_pts_violating = sum(1 for i in range(len(path))
                          if _trilinear_esdf(esdf, origin, resolution, path[i]) < esdf_threshold)
    n_safe_violating = sum(1 for i in range(len(path))
                           if tags[i] == "original_safe"
                           and _trilinear_esdf(esdf, origin, resolution, path[i]) < esdf_threshold)
    n_pushable_violating = n_pts_violating - n_safe_violating
    rospy.logerr("[LinePush] FAILED after %d iterations: "
                 "%d/%d points violating (safe_immutable=%d pushable=%d), "
                 "%d total insertions, worst ESDF=%.3f",
                 max_iterations, n_pts_violating, len(path),
                 n_safe_violating, n_pushable_violating,
                 total_segment_insertions, float(best_worst))
    return None, {
        "reason": "max_iterations_reached",
        "iterations": max_iterations,
        "control_points": len(path),
        "worst_clearance": float(best_worst),
        "segment_insertions": total_segment_insertions,
        "points_violating": n_pts_violating,
        "safe_immutable_violating": n_safe_violating}


# ============================================================================
#  7.  GlobalPathPlanner — observation-conditioned rollout only
# ============================================================================

class GlobalPathPlanner:
    """Global path planner: local-observation policy rollout ONLY.

    This is the SINGLE, DETERMINISTIC global path algorithm.
    There is NO LinePush fallback, NO RRT, and NO unconstrained smoothing.

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
        self.min_clearance = gp_cfg.get("min_clearance", 0.30)
        self.algorithm = str(
            gp_cfg.get(
                "algorithm", "observation_rollout")).lower()
        self.observation_rollout_config = dict(
            gp_cfg.get("observation_rollout", {}))
        depth_cfg = config.get("depth", {})
        self.observation_rollout_config.setdefault(
            "horizontal_fov_deg", float(depth_cfg.get("fov", 90.0)))
        self.observation_rollout_config.setdefault(
            "observation_range_m", float(depth_cfg.get("max_m", 5.0)))
        self.collision_check_spacing = gp_cfg.get(
            "collision_check_spacing_m", 0.05)
        self.reference_spacing = float(
            gp_cfg.get("reference_resample_spacing_m", 0.10))
        
        # Ensure check spacing <= resolution / 2
        self.collision_check_spacing = min(
            self.collision_check_spacing, self.esdf_res * 0.5)
        
        # No fallback planner is created.
        self.a_star = None
    
    def plan_global(self, start, goal):
        """Plan a global reference path from start to goal.

        Uses ONLY an observation-conditioned rollout. No LinePush fallback.

        Returns dict:
            start, goal, raw_path, global_path, valid,
            validation_report, raw_path_points, global_path_points,
            global_path_length
        Returns None on planning failure.
        """
        t0 = time.time()
        geometric_report = {}
        algorithm_used = "observation_rollout"

        # ── THE ONLY global path algorithm ────────────────────────
        geometric_path, geometric_report = (
            observation_conditioned_rollout_path(
            start, goal, self.esdf, self.esdf_origin, self.esdf_res,
            self.min_clearance, self.collision_check_spacing,
            self.observation_rollout_config))

        if geometric_path is None:
            rospy.logerr(
                "[GlobalPath] observation rollout FAILED: %s. "
                "No fallback available; global planning failed.",
                geometric_report.get("reason", "unknown"))
            return None

        rospy.loginfo(
            "[GlobalPath] observation rollout SUCCESS: "
            "cost=%.2f pts=%d decisions=%d corridor_checks=%d.",
            geometric_report.get("cost", 0.0),
            len(geometric_path),
            len(geometric_report.get("decision_events", [])),
            geometric_report.get("corridor_checks", 0))

        raw_path = geometric_path
        global_path = list(raw_path)

        # ── Resample to uniform spacing (for stable guide indexing) ─
        # This does NOT move points — it just adds intermediate points
        # on straight segments for uniform arc-length spacing.
        global_path = resample_path(global_path, self.reference_spacing)

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
                self.min_clearance, self.collision_check_spacing)
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
            "planner_algorithm": algorithm_used,
            "observation_rollout": geometric_report,
            "raw_path_points": len(raw_path),
            "reference_path_points": len(global_path),
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
                reasons.append("no_valid_path")
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
        
        rospy.loginfo("[GlobalPath] %s %d pts -> reference %d pts, length=%.1fm, "
                      "valid=%s, time=%.2fs",
                      algorithm_used, len(raw_path), len(global_path),
                      global_path_length,
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
        n = max(1, int(math.ceil(L / spacing)))
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
