#!/usr/bin/env python3
"""Path validation and observation-conditioned reference planning.

``AStarPlanner`` remains the deterministic full-map validator used while
accepting generated tasks.  Expert behaviour uses the single causal
``observation_conditioned_rollout_path`` flow through ``GlobalPathPlanner``;
there are no alternative runtime planners or post-processing fallbacks.

All coordinates are ROS world frame (x-forward, y-left, z-up).
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
#  6.  Observation-conditioned rollout helpers
# ============================================================================

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


def resample_path(path, spacing=0.2):
    """Return a polyline with uniform maximum segment length.

    Resampling only inserts points on existing straight segments, so it does
    not change path geometry or clearance.  The final input point is preserved
    exactly.
    """
    if len(path) < 2:
        return list(path)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")

    out = [path[0]]
    for index in range(1, len(path)):
        start = np.asarray(path[index - 1], dtype=np.float64)
        end = np.asarray(path[index], dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        segments = max(1, int(math.ceil(length / spacing)))
        for segment in range(1, segments + 1):
            alpha = segment / float(segments)
            out.append(tuple(start + alpha * (end - start)))
    out[-1] = path[-1]
    return out


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
        if self.algorithm != "observation_rollout":
            raise ValueError(
                "planning.global_planner.algorithm must be "
                "observation_rollout, got {!r}".format(self.algorithm))
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
        }
        
        rospy.loginfo("[GlobalPath] %s %d pts -> reference %d pts, length=%.1fm, "
                      "valid=%s, time=%.2fs",
                      algorithm_used, len(raw_path), len(global_path),
                      global_path_length,
                      valid, report["planning_time_sec"])
        
        return plan


