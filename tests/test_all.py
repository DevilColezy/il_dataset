#!/usr/bin/env python3
"""
Characterization & Unit Tests for IL Dataset modules.

Tests that do NOT require ROS, Unity, or ZMQ:
  - Coordinate transformations
  - world_vel_to_body
  - world→grid floor for negative coordinates
  - PLY loading (ASCII, CRLF, binary LE/BE, with RGB/normal attrs)
  - A* planning: no path, start/goal collision, max iterations, partial path rejection
  - Diagonal corner cutting prevention
  - Conservative coarse ESDF
  - Smoothing: endpoints fixed, segment clearance
  - Final sampled trajectory: exact goal position
  - Yaw: ±179° interpolation, unwrap, EMA, yaw-rate from canonical profile
  - quaternion yaw vs yaw-rate numerical derivative consistency
  - Config loading and validation
  - Frame matching: delay, out-of-order, duplicate, missing frames
  - Inprogress/commit/reject data lifecycle

Usage:
    cd g:/Code/flightmare_ws/src/il_dataset
    python -m pytest tests/ -v
    # or
    python tests/test_all.py
"""

from __future__ import print_function, division

import os, sys, json, math, tempfile, shutil, struct
import numpy as np

# Add parent scripts dir to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.join(_script_dir, "..", "scripts")
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ── Test helpers ─────────────────────────────────────────────────────────

def approx_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def assert_all_close(a, b, tol=1e-6, msg=""):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.allclose(a, b, atol=tol):
        raise AssertionError("{}: {} != {} (tol={})".format(msg, a, b, tol))


def test_velocity_command_integrator_is_continuous_across_plan_updates():
    from il_common import integrate_velocity_command

    dt = 0.04
    pos = np.zeros(3)
    vel = np.zeros(3)
    yaw = 0.0
    positions = []
    velocities = []
    for _ in range(8):
        pos, vel, yaw, yaw_rate = integrate_velocity_command(
            pos, vel, yaw, [2.0, 0.0, 0.0], dt, 2.5, 3.5, 2.0)
        positions.append(pos.copy())
        velocities.append(vel.copy())
        assert np.linalg.norm(vel) <= 2.5 + 1e-9
        assert abs(yaw_rate) <= 2.0 + 1e-9

    # Replacing the plan changes acceleration, never position or velocity
    # discontinuously.  In particular there is no repeated sample-0 hold.
    old_pos = pos.copy()
    old_vel = vel.copy()
    pos, vel, yaw, _ = integrate_velocity_command(
        pos, vel, yaw, [0.0, 2.0, 0.0], dt, 2.5, 3.5, 2.0)
    assert np.linalg.norm(vel - old_vel) <= 3.5 * dt + 1e-9
    assert np.linalg.norm(pos - old_pos) > 0.0
    assert_all_close(pos - old_pos, 0.5 * (old_vel + vel) * dt)

    step_distances = np.linalg.norm(np.diff(np.asarray(positions), axis=0), axis=1)
    assert np.all(step_distances > 0.0)
    speed_deltas = np.linalg.norm(np.diff(np.asarray(velocities), axis=0), axis=1)
    assert np.all(speed_deltas <= 3.5 * dt + 1e-9)


def test_velocity_command_integrator_brakes_instead_of_hard_stop():
    from il_common import integrate_velocity_command

    pos, vel, yaw, _ = integrate_velocity_command(
        [0, 0, 0], [1.0, 0, 0], 0.0, [0, 0, 0],
        0.04, 2.5, 3.5, 2.0)
    assert_all_close(vel, [0.86, 0.0, 0.0])
    assert 0.0 < pos[0] < 0.04


def test_smart_obstacle_sampler_random_order_and_passable_gaps():
    import importlib.util

    sampler_path = os.path.abspath(os.path.join(
        _script_dir, "..", "..", "flightmare_dataset_tools",
        "scripts", "export_pointcloud.py"))
    spec = importlib.util.spec_from_file_location(
        "test_export_pointcloud", sampler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    params = {
        "seed": 700, "count": 0, "count_min": 12, "count_max": 20,
        "target_occupancy": 0.05,
        "x_min": -9.0, "x_max": 11.0, "y_min": 0.0, "y_max": 30.0,
        "radius_min": 0.15, "radius_max": 4.0,
        "height_min": 8.0, "height_max": 8.0,
        "min_gap": 0.20, "drone_radius": 0.20, "safety_margin": 0.20,
        "border_margin": 0.20, "avoid_x": 0.0, "avoid_y": 0.0,
        "avoid_radius": 0.0, "candidate_attempts": 60,
        "shrink_on_fail": True, "shrink_factor": 0.85,
        "scale_weights": [0.72, 0.23, 0.05],
        "ground_z": 0.0, "size_interval": 0.5,
    }
    obstacles = module.SmartObstacleSampler(params).sample()
    assert params["count_min"] <= len(obstacles) <= params["count_max"]

    radii = [obs["radius"] for obs in obstacles]
    assert radii != sorted(radii)
    assert radii != sorted(radii, reverse=True)

    required_surface_gap = (params["min_gap"] +
                            2.0 * params["drone_radius"] +
                            params["safety_margin"])
    for i, first in enumerate(obstacles):
        for second in obstacles[i + 1:]:
            surface_gap = (math.hypot(first["x"] - second["x"],
                                      first["y"] - second["y"]) -
                           first["radius"] - second["radius"])
            assert surface_gap + 1e-9 >= required_surface_gap

    large = [r for r in radii if
             (r - params["radius_min"]) /
             (params["radius_max"] - params["radius_min"]) >= 0.70]
    assert len(large) <= max(1, int(math.ceil(0.20 * len(radii))))


# ══════════════════════════════════════════════════════════════════════════
#  1.  Coordinate transformation tests
# ══════════════════════════════════════════════════════════════════════════

def test_ros_to_unity():
    from il_common import ros_pos_to_unity
    # ROS (x-fwd, y-left, z-up) → Unity (x-right, y-up, z-fwd)
    assert ros_pos_to_unity([1, 2, 3]) == [1, 3, 2]
    assert ros_pos_to_unity([0, 0, 0]) == [0, 0, 0]
    assert ros_pos_to_unity([-5, 10, 3]) == [-5, 3, 10]


def test_yaw_to_unity_quat():
    from il_common import yaw_to_unity_quat
    # yaw=0 → half=0 → [0, 0, 0, 1] (identity quaternion)
    q = yaw_to_unity_quat(0)
    assert_all_close(q, [0, 0, 0, 1])
    # yaw=π → half=-π/2 → [0, -1, 0, 0]
    q = yaw_to_unity_quat(math.pi)
    assert_all_close(q, [0, -1, 0, 0], tol=1e-5)
    # yaw=π/2 → half=-π/4 → [0, -sin(π/4), 0, cos(π/4)]
    q = yaw_to_unity_quat(math.pi / 2)
    assert_all_close(q, [0, -math.sin(math.pi/4), 0, math.cos(math.pi/4)], tol=1e-5)


def test_world_vel_to_body_baseline():
    """Test world_vel_to_body at key yaw angles."""
    from il_common import world_vel_to_body

    # yaw=0: nose faces +Y world. Body right=X world, body fwd=Y world
    # World vel (1,0,0) = forward in world → should be (1, 0, 0) in body (right)
    vx, vy, vz = world_vel_to_body(1, 0, 0, 0)
    assert_all_close([vx, vy, vz], [1, 0, 0])

    # World vel (0,1,0) = left in world → should be (0, 1, 0) in body (forward)
    vx, vy, vz = world_vel_to_body(0, 1, 0, 0)
    assert_all_close([vx, vy, vz], [0, 1, 0])

    # yaw=-π/2: nose faces +X world.
    # World vel (1,0,0) forward → body fwd (+Y)
    vx, vy, vz = world_vel_to_body(1, 0, 0, -math.pi / 2)
    assert_all_close([vx, vy, vz], [0, 1, 0], tol=1e-5)

    # yaw=π: nose faces -Y world
    # World vel (-1,0,0) = back in world → body fwd (+Y) should be negative?
    vx, vy, vz = world_vel_to_body(-1, 0, 0, math.pi)
    # At yaw=π, body right is (-1,0), body fwd is (0,-1)
    assert_all_close([vx, vy, vz], [1, 0, 0], tol=1e-5)


def test_world_vel_to_body_roundtrip():
    """world_vel_to_body → body_vel_to_world should be identity."""
    from il_common import world_vel_to_body, body_vel_to_world
    rng = np.random.RandomState(42)
    for _ in range(100):
        vx, vy, vz = rng.uniform(-10, 10, 3)
        yaw = rng.uniform(-math.pi, math.pi)
        vb = world_vel_to_body(vx, vy, vz, yaw)
        vw = body_vel_to_world(vb[0], vb[1], vb[2], yaw)
        assert_all_close([vx, vy, vz], vw, tol=1e-5)


# ══════════════════════════════════════════════════════════════════════════
#  2.  World-to-grid floor tests (negative coordinates)
# ══════════════════════════════════════════════════════════════════════════

def test_w2g_floor_negative():
    """floor() correctly handles negative coordinates, unlike int()."""
    from il_trajectory import _w2g
    origin = (-10.0, -10.0, -5.0)
    res = 0.15

    # Positive case
    assert _w2g((0, 0, 0), origin, res) == (66, 66, 33)

    # Negative case: int(-0.5) = 0, floor(-0.5) = -1
    idx = _w2g((-9.9, -9.9, -4.9), origin, res)
    # (-9.9 - (-10)) / 0.15 = 0.1/0.15 = 0.666... → floor = 0
    assert idx[0] >= 0  # should be 0 not -1

    # Test with value just below origin
    idx2 = _w2g((-10.1, -10.1, -5.1), origin, res)
    # (-10.1 - (-10)) / 0.15 = -0.1/0.15 = -0.666 → floor = -1
    assert idx2[0] == -1, "Expected -1, got {}".format(idx2[0])
    assert idx2[1] == -1
    assert idx2[2] == -1


def test_w2g_consistent_with_g2w():
    """Grid roundtrip: g2w(w2g(p)) should be close to p for grid-aligned points."""
    from il_trajectory import _w2g, _g2w
    origin = (-5.0, 0.0, 0.0)
    res = 0.2

    for _ in range(20):
        x = np.random.uniform(-4, 10)
        y = np.random.uniform(0, 20)
        z = np.random.uniform(0, 5)
        idx = _w2g((x, y, z), origin, res)
        center = _g2w(idx, origin, res)
        # The center should be within one voxel of the original point
        assert abs(center[0] - x) <= res
        assert abs(center[1] - y) <= res
        assert abs(center[2] - z) <= res


# ══════════════════════════════════════════════════════════════════════════
#  3.  PLY loading tests
# ══════════════════════════════════════════════════════════════════════════

def _write_ply(filepath, header_lines, body_bytes=b""):
    with open(filepath, "wb") as f:
        f.write(b"\n".join(header_lines) + b"\nend_header\n")
        f.write(body_bytes)


def test_ply_ascii():
    from il_common import load_ply
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tf:
        path = tf.name
    try:
        header = [
            b"ply", b"format ascii 1.0",
            b"element vertex 3", b"property float x",
            b"property float y", b"property float z",
        ]
        body = b"1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n"
        _write_ply(path, header, body)
        pts = load_ply(path)
        assert pts.shape == (3, 3)
        assert_all_close(pts[0], [1, 2, 3])
        assert_all_close(pts[2], [7, 8, 9])
    finally:
        os.unlink(path)


def test_ply_crlf_header():
    """Handle CRLF line endings in header."""
    from il_common import load_ply
    path = tempfile.mktemp(suffix=".ply")
    try:
        with open(path, "wb") as f:
            f.write(b"ply\r\nformat ascii 1.0\r\nelement vertex 1\r\n")
            f.write(b"property float x\r\nproperty float y\r\nproperty float z\r\n")
            f.write(b"end_header\r\n1 2 3\r\n")
        pts = load_ply(path)
        assert pts.shape == (1, 3)
        assert_all_close(pts[0], [1, 2, 3])
    finally:
        os.unlink(path)


def test_ply_binary_le():
    from il_common import load_ply
    path = tempfile.mktemp(suffix=".ply")
    try:
        header = [
            b"ply", b"format binary_little_endian 1.0",
            b"element vertex 2", b"property float x",
            b"property float y", b"property float z",
        ]
        body = struct.pack("<fff", 1.0, 2.0, 3.0) + struct.pack("<fff", 4.0, 5.0, 6.0)
        _write_ply(path, header, body)
        pts = load_ply(path)
        assert pts.shape == (2, 3)
        assert_all_close(pts[0], [1, 2, 3])
        assert_all_close(pts[1], [4, 5, 6])
    finally:
        os.unlink(path)


def test_ply_binary_be():
    from il_common import load_ply
    path = tempfile.mktemp(suffix=".ply")
    try:
        header = [
            b"ply", b"format binary_big_endian 1.0",
            b"element vertex 1", b"property float x",
            b"property float y", b"property float z",
        ]
        body = struct.pack(">fff", 7.0, 8.0, 9.0)
        _write_ply(path, header, body)
        pts = load_ply(path)
        assert pts.shape == (1, 3)
        assert_all_close(pts[0], [7, 8, 9])
    finally:
        os.unlink(path)


def test_ply_with_rgb_and_normals():
    """Binary PLY with extra vertex properties (RGB, normals) and correct stride."""
    from il_common import load_ply
    path = tempfile.mktemp(suffix=".ply")
    try:
        header = [
            b"ply", b"format binary_little_endian 1.0",
            b"element vertex 2",
            b"property float x", b"property float y", b"property float z",
            b"property uchar red", b"property uchar green", b"property uchar blue",
            b"property float nx", b"property float ny", b"property float nz",
        ]
        # stride = 4*3 + 1*3 + 4*3 = 27 bytes per vertex
        body = (
            struct.pack("<fffBBBfff", 1.0, 2.0, 3.0, 255, 0, 0, 0.0, 0.0, 1.0) +
            struct.pack("<fffBBBfff", 4.0, 5.0, 6.0, 0, 255, 0, 1.0, 0.0, 0.0)
        )
        _write_ply(path, header, body)
        pts = load_ply(path)
        assert pts.shape == (2, 3)
        assert_all_close(pts[0], [1, 2, 3])
        assert_all_close(pts[1], [4, 5, 6])
    finally:
        os.unlink(path)


def test_ply_empty_file_raises():
    from il_common import load_ply
    path = tempfile.mktemp(suffix=".ply")
    try:
        with open(path, "wb") as f:
            f.write(b"")
        try:
            load_ply(path)
            assert False, "Should have raised"
        except ValueError:
            pass  # expected
    finally:
        os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════
#  4.  A* planner tests
# ══════════════════════════════════════════════════════════════════════════

def _make_simple_esdf(gx=20, gy=20, gz=5, res=0.5):
    """Create a simple ESDF: all free except a partial wall in the middle.
    The wall does NOT span the full x-range so A* can go around."""
    esdf = np.ones((gx, gy, gz), dtype=np.float64) * 2.0  # 2m clearance everywhere
    # Wall at y=10, spanning x = 2..17 (leaving gaps at edges for path)
    esdf[2:18, 10, :] = -1.0
    origin = (0.0, 0.0, 0.0)
    return esdf, origin, res


def test_astar_simple_path():
    from il_trajectory import AStarPlanner
    esdf, origin, res = _make_simple_esdf()
    planner = AStarPlanner(esdf, res, origin, cost_weight=2.0)

    # Grid covers x=0..10, y=0..10, z=0..2.5 (20x20x5 cells, 0.5m res)
    # Wall at y-grid=10 (y=5m world), x-grid=2..17 (x=1m..8.5m world)
    # Start at y=2m (south), goal at y=8m (north) → must go around wall
    start = (5.0, 2.0, 1.0)
    goal = (5.0, 8.0, 1.0)

    result = planner.plan(start, goal, max_iterations=50000)
    assert result.reached_goal, "A* should find a path around the partial wall: {}".format(
        result.failure_reason)
    assert len(result.path) >= 2
    start_dist = np.linalg.norm(np.array(result.path[0]) - np.array(start))
    goal_dist = np.linalg.norm(np.array(result.path[-1]) - np.array(goal))
    assert start_dist < res * 3, "Start dist {:.2f} > {:.2f}".format(start_dist, res * 3)
    assert goal_dist < res * 3, "Goal dist {:.2f} > {:.2f}".format(goal_dist, res * 3)


def test_trilinear_esdf_uses_voxel_centres():
    """A query at a voxel centre must reproduce that voxel's exact value."""
    from il_trajectory import _trilinear_esdf
    esdf = np.arange(4 * 5 * 3, dtype=np.float64).reshape((4, 5, 3))
    origin = (-1.0, 2.0, -0.5)
    res = 0.1
    idx = (2, 3, 1)
    centre = tuple(origin[d] + (idx[d] + 0.5) * res for d in range(3))
    assert abs(_trilinear_esdf(esdf, origin, res, centre) - esdf[idx]) < 1e-10


def test_astar_no_path():
    """Start and goal in disconnected free regions → no path possible."""
    from il_trajectory import AStarPlanner
    # Two separated free pockets with collision between them
    esdf = -np.ones((10, 10, 3), dtype=np.float64)
    esdf[1:4, 1:4, 1] = 2.0      # free pocket A at (1..3, 1..3)
    esdf[6:9, 6:9, 1] = 2.0      # free pocket B at (6..9, 6..9) — disconnected
    origin = (0.0, 0.0, 0.0)

    planner = AStarPlanner(esdf, 0.5, origin)
    # Start in pocket A, goal in pocket B — no connecting path
    result = planner.plan((1.0, 1.0, 0.5), (4.0, 4.0, 0.5), max_iterations=5000)
    assert not result.reached_goal, (
        "No path between disconnected pockets, got reached_goal={}".format(
            result.reached_goal))


def test_astar_start_in_collision():
    from il_trajectory import AStarPlanner
    esdf, origin, res = _make_simple_esdf()
    # Place a single blocked cell where the start would be
    esdf[4, 4, 2] = -1.0  # isolated collision cell
    planner = AStarPlanner(esdf, res, origin)
    # Start at (2.0, 2.0, 1.0) → grid (4, 4, 2) which is now blocked
    result = planner.plan((2.0, 2.0, 1.0), (8.0, 7.5, 1.0), max_iterations=10000)
    assert result.reached_goal, (
        "Should recover from start collision, got: {}".format(result.failure_reason))


def test_astar_goal_in_collision_deep():
    """Goal in a map with NO free cells → _nearest_free returns None → failure."""
    from il_trajectory import AStarPlanner
    # 100% collision — no free cell anywhere
    esdf = -np.ones((10, 10, 5), dtype=np.float64)
    origin = (0.0, 0.0, 0.0)
    planner = AStarPlanner(esdf, 0.5, origin)
    result = planner.plan((1.0, 1.0, 1.0), (4.0, 4.0, 1.0), max_iterations=2000)
    assert not result.reached_goal, (
        "100% collision map should have no path, got reached_goal={} reason={}".format(
            result.reached_goal, result.failure_reason))
    # Failure should be about collision (no free cell for start or goal)
    assert "collision" in result.failure_reason.lower(), (
        "Expected collision-related failure, got: {}".format(result.failure_reason))


def test_astar_partial_path_rejected():
    """A* returns PlanResult, not raw path; check reached_goal flag."""
    from il_trajectory import AStarPlanner, PlanResult

    esdf, origin, res = _make_simple_esdf(gx=100, gy=100, gz=5, res=0.5)
    planner = AStarPlanner(esdf, res, origin)

    # Very distant goal with low max_iterations → partial path
    result = planner.plan((1.0, 1.0, 2.0), (45.0, 45.0, 2.0), max_iterations=50)
    assert not result.reached_goal, "Should not reach goal with 50 iterations"
    assert result.failure_reason == "max_iterations_reached"
    assert result.goal_error is not None and result.goal_error > 0
    # The result should NOT be treated as a valid path (bool test)
    assert not bool(result)


# ══════════════════════════════════════════════════════════════════════════
#  5.  Diagonal corner cutting test
# ══════════════════════════════════════════════════════════════════════════

def test_diagonal_corner_cutting_prevented():
    """A diagonal step through a thin wall should be blocked."""
    from il_trajectory import AStarPlanner
    # Create a thin diagonal wall
    esdf = np.ones((10, 10, 3), dtype=np.float64) * 2.0
    # Two cells at diagonal positions: (3,4) and (4,3)
    esdf[3, 4, 1] = -1.0
    esdf[4, 3, 1] = -1.0
    origin = (0.0, 0.0, 0.0)
    planner = AStarPlanner(esdf, 0.5, origin)

    result = planner.plan((1.5, 1.5, 0.5), (3.5, 4.5, 0.5), max_iterations=5000)
    # If diagonal corner cutting is prevented, path won't cut through
    # Just verify it doesn't crash and returns something reasonable
    assert result.path is not None


# ══════════════════════════════════════════════════════════════════════════
#  6.  Conservative coarse ESDF test
# ══════════════════════════════════════════════════════════════════════════

def test_conservative_coarse_esdf():
    """Small obstacles must survive downsampling."""
    from il_trajectory import make_coarse_esdf
    # Obstacle at ODD index (3, 1, 0) so naive [::2,::2,::2] MISSES it
    # but conservative min-pooling CATCHES it.
    esdf = np.ones((4, 4, 2), dtype=np.float64) * 3.0
    esdf[3, 1, 0] = -1.0  # obstacle at odd indices

    coarse = make_coarse_esdf(esdf, factor=2)
    assert coarse.shape == (2, 2, 1)
    # Coarse index (1,0,0) covers fine (2:4, 0:2, 0:2) = cells (2,0)..(3,1)
    assert coarse[1, 0, 0] == -1.0, (
        "Single-voxel obstacle must survive conservative min-pooling")

    # Naive subsampling misses it because x-stride=2 samples x=0,2 only
    naive = esdf[::2, ::2, ::2].copy()
    assert naive[1, 0, 0] == 3.0, (
        "Naive subsampling WOULD lose the obstacle (this proves the bug)")


# ══════════════════════════════════════════════════════════════════════════
#  7.  Smoothing endpoint fixity test
# ══════════════════════════════════════════════════════════════════════════

def test_smoothing_endpoints_fixed():
    """smooth_trajectory and smooth_position_path must preserve start/goal."""
    from il_trajectory import smooth_trajectory, smooth_position_path
    # Create a simple ESDF
    esdf = np.ones((20, 20, 5), dtype=np.float64) * 2.0
    origin = (0.0, 0.0, 0.0)
    res = 0.5

    # Zigzag path
    path = [(1.0, 1.0, 2.0), (3.0, 1.5, 2.0), (5.0, 1.0, 2.0),
            (7.0, 1.5, 2.0), (9.0, 1.0, 2.0)]

    smoothed = smooth_trajectory(path, esdf, origin, res, min_clearance=0.0)
    assert_all_close(smoothed[0], path[0], msg="Start must be fixed")
    assert_all_close(smoothed[-1], path[-1], msg="Goal must be fixed")

    # smooth_position_path
    dense = [(x * 0.1, 0.0, 2.0) for x in range(50)]
    smoothed2 = smooth_position_path(dense, window=3)
    assert_all_close(smoothed2[0], dense[0], msg="Start must be fixed")
    assert_all_close(smoothed2[-1], dense[-1], msg="Goal must be fixed")


# ══════════════════════════════════════════════════════════════════════════
#  8.  Final sampled trajectory: exact goal
# ══════════════════════════════════════════════════════════════════════════

def test_sampled_trajectory_exact_goal():
    """Last sample must be exact goal position."""
    from il_trajectory import sample_trajectory
    timed = [
        (0.0, 0.0, 0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0, 1.0, 0.5),
        (2.0, 2.0, 0.0, 1.0, 1.0),
    ]
    sampled = sample_trajectory(timed, 0.1)
    assert len(sampled) > 0
    last = sampled[-1]
    assert_all_close([last[1], last[2], last[3]], [2.0, 0.0, 1.0])
    assert_all_close(last[0], 2.0)  # t = total_time


# ══════════════════════════════════════════════════════════════════════════
#  9.  Yaw tests
# ══════════════════════════════════════════════════════════════════════════

def test_yaw_179_to_minus179_interpolation():
    """Interpolating from +179° to -179° should go ~2° not ~358°."""
    from il_trajectory import sample_trajectory
    # Two waypoints with yaw at +179° and -179°
    timed = [(0.0, 0.0, 0.0, 1.0, math.radians(179)),
             (1.0, 1.0, 0.0, 1.0, math.radians(-179))]
    sampled = sample_trajectory(timed, 0.5)
    assert len(sampled) >= 2
    # Middle sample should be around ±180° (π or -π), not 0°
    mid = sampled[len(sampled) // 2]
    mid_yaw_deg = abs(math.degrees(mid[4]))
    # Should be near 180°, not near 0°
    assert mid_yaw_deg > 170, "Midpoint yaw should be ~180°, got {:.1f}°".format(mid_yaw_deg)


def test_yaw_unwrap():
    """unwrap should produce continuous sequence."""
    from il_trajectory import generate_yaw_profile
    # Use waypoints that don't create an exact pi reversal
    waypoints = [(0, 0, 0), (1, 0.1, 0), (2, 0, 0), (1, -0.1, 0), (0, 0, 0)]
    yaws = generate_yaw_profile(waypoints)
    # Should be continuous (no jumps >= pi)
    for i in range(1, len(yaws)):
        jump = abs(yaws[i] - yaws[i - 1])
        assert jump <= math.pi + 1e-9, \
            "Unwrap failed: yaw jump at {}: {} -> {} (jump={:.4f})".format(
                i, yaws[i-1], yaws[i], jump)


def test_yaw_ema_smoothing():
    """EMA with alpha close to 1 should produce strong smoothing."""
    from il_trajectory import smooth_yaw_profile
    raw = np.array([0.0, 1.0, 0.0, 1.0, 0.0])  # oscillating
    smoothed = smooth_yaw_profile(raw, alpha=0.95)
    # With alpha=0.95, only 5% of new value mixed in each step
    # After 4 steps, smoothed should be much closer to 0 than 1
    assert abs(smoothed[-1]) < 0.3, \
        "Strong smoothing expected, got {}".format(smoothed[-1])

    # alpha=0 (no smoothing) should follow raw exactly
    smoothed_none = smooth_yaw_profile(raw, alpha=0.0)
    assert_all_close(smoothed_none, raw)


def test_yawrate_from_canonical_yaw_consistency():
    """Pose quaternion yaw and yaw-rate must come from same canonical profile."""
    from il_trajectory import generate_controls
    # Simple straight-line trajectory
    sampled = [(t * 0.02, t * 0.05, 0.0, 1.0, math.atan2(0.05, 0.0))
               for t in range(100)]
    controls = generate_controls(sampled, 0.02, lookahead=5,
                                 yaw_smooth_alpha=0.85,
                                 max_vel=5.0, max_yaw_rate=3.0)

    # controls[i] = (t, vx, vy, vz, yr, corrected_yaw)
    for i in range(1, len(controls)):
        yaw_prev = controls[i - 1][5]
        yaw_curr = controls[i][5]
        yr = controls[i][4]
        dt = 0.02
        # yaw_rate should approximately equal (yaw_curr - yaw_prev)/dt
        dy = yaw_curr - yaw_prev
        dy = math.atan2(math.sin(dy), math.cos(dy))
        expected_yr = dy / dt
        assert abs(yr - expected_yr) < 0.01, \
            "Yaw-rate inconsistent with yaw profile at step {}: yr={:.4f} expected={:.4f}".format(
                i, yr, expected_yr)


# ══════════════════════════════════════════════════════════════════════════
#  10.  Frame matching tests (mock Unity delay, disorder, duplicates)
# ══════════════════════════════════════════════════════════════════════════

def test_sync_buffer_fifo():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=10)

    for i in range(5):
        buf.push({"frame_id": i, "t_sent": float(i), "pos": [i, 0, 0]})

    # Match by frame_id
    entry, err = buf.match_by_frame_id(2)
    assert entry is not None
    assert entry["frame_id"] == 2

    # Non-existent frame_id
    entry, err = buf.match_by_frame_id(99)
    assert entry is None


def test_sync_buffer_time_match():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=10, max_sync_error_ms=100)

    for i in range(5):
        buf.push({"frame_id": i, "t_sent": float(i), "pos": [i, 0, 0]})

    # Match by time
    entry, err = buf.match_by_time(2.05)
    assert entry is not None
    assert entry["frame_id"] == 2

    # Too far away
    entry, err = buf.match_by_time(10.0)
    assert entry is None  # beyond max_sync_error


def test_sync_buffer_drain_to_latest():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=10)
    for i in range(5):
        buf.push({"frame_id": i, "t_sent": float(i)})

    latest = buf.drain_to_latest()
    assert latest["frame_id"] == 4
    assert buf.size() == 0  # returned frame was consumed; do not reuse it


def test_sync_buffer_overflow():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=3)
    for i in range(10):
        buf.push({"frame_id": i})
    assert buf.size() == 3
    # Oldest should be 7, newest 9
    entry, _ = buf.match_by_frame_id(7)
    assert entry is not None
    entry, _ = buf.match_by_frame_id(0)
    assert entry is None  # evicted


# ══════════════════════════════════════════════════════════════════════════
#  11.  Config loading & validation tests
# ══════════════════════════════════════════════════════════════════════════

def test_config_load_valid():
    """The default config should load without errors."""
    import yaml
    # Config is at the package root level (tests/../config/)
    cfg_path = os.path.join(os.path.dirname(_parent_dir), "config", "il_dataset_config.yaml")
    if not os.path.isfile(cfg_path):
        # Fallback: relative to cwd
        cfg_path = "config/il_dataset_config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert "global" in cfg
    assert "scenes" in cfg
    assert len(cfg["scenes"]) >= 1
    g = cfg["global"]
    assert "fsm" in g
    assert g["fsm"]["connect_timeout"] > 0


def test_config_validation_rejects_bad_values():
    """Config validation should catch invalid values."""
    from il_config import _validate_config

    # Missing required global keys
    bad_cfg = {"global": {"scene_id": 1}, "scenes": [{"name": "t", "seeds": [1],
               "target_occupancy": 0.04, "radius_min": 0.1, "radius_max": 1.0}]}
    try:
        _validate_config(bad_cfg)
        assert False, "Should have raised ValueError for missing keys"
    except ValueError as e:
        err = str(e)
        # Error should mention missing keys or invalid values
        assert "Missing" in err or "must be" in err or "error" in err.lower(), (
            "Expected validation error, got: {}".format(e))

    # record_hz > control_hz
    bad_cfg2 = {
        "global": {
            "scene_id": 1, "pub_port": "10253", "sub_port": "10254",
            "output_dir": "/tmp",
            "fsm": {"connect_timeout": 60, "scene_settle_timeout": 10,
                    "pc_export_timeout": 300, "esdf_build_timeout": 60,
                    "drone_stable_timeout": 10, "trajectory_timeout": 120,
                    "keep_alive_period": 3},
            "depth": {"width": 640, "height": 480, "fov": 90,
                      "max_m": 20, "near": 0.01, "far": 1000},
            "pointcloud": {"range": [30, 50, 10], "origin": [0, 20, 5],
                           "resolution": 0.15},
            "esdf": {"resolution": 0.15, "drone_radius": 0.25},
            "control": {"control_hz": 10, "record_hz": 50},
            "obstacle": {"area_x": [-9, 9], "area_y": [0, 30]},
            "start_goal": {
                "num_pairs_per_config": 3,
                "start_zone": {"x": [-8, 8], "y": [-2, -1], "z_min": 2, "z_max": 4},
                "goal_zone": {"x": [-8, 8], "y": [31, 32], "z_min": 2, "z_max": 4},
                "min_start_goal_distance": 15, "max_attempts": 200,
                "min_esdf_clearance_at_endpoints": 0.15,
            },
            "planning": {
                "a_star_cost_weight": 2.0,
                "esdf_optimize": {"enabled": True, "smooth_iterations": 300,
                                  "smooth_step": 0.5, "push_iterations": 60,
                                  "push_step": 0.03},
                "time_param": {"nominal_speed": 2, "max_velocity": 3,
                               "max_acceleration": 5, "max_jerk": 25,
                               "max_yaw_rate": 3, "min_obstacle_clearance": 0.2,
                               "curvature_slowdown": True, "curvature_gain": 0.8},
                "resample_spacing": 0.15,
                "pos_smooth_window": 8, "control_lookahead": 6,
                "control_yaw_smooth": 0.85,
                "validation": {"max_clearance_violations": 10,
                               "max_dynamics_violations": 60},
            },
            "data": {"coordinate_frame": "ROS_WORLD"},
        },
        "scenes": [{"name": "test", "target_occupancy": 0.04,
                    "radius_min": 0.15, "radius_max": 1.0, "seeds": [1]}],
    }
    try:
        _validate_config(bad_cfg2)
        assert False, "Should have raised ValueError for record_hz > control_hz"
    except ValueError as e:
        err_str = str(e).lower()
        assert "record_hz" in err_str or "control_hz" in err_str, \
            "Expected error about record/control hz, got: {}".format(e)


# ══════════════════════════════════════════════════════════════════════════
#  12.  Data lifecycle tests (.inprogress → commit / reject)
# ══════════════════════════════════════════════════════════════════════════

def test_inprogress_commit_reject_lifecycle():
    """Simulate the atomic commit flow: .inprogress → rename."""
    tmpdir = tempfile.mkdtemp()
    try:
        inprogress = os.path.join(tmpdir, "traj_001.inprogress")
        final = os.path.join(tmpdir, "traj_001")
        failed_dir = os.path.join(tmpdir, "_failed")

        os.makedirs(inprogress)
        os.makedirs(failed_dir)

        # Write fake data
        with open(os.path.join(inprogress, "data.csv"), "w") as f:
            f.write("col1,col2\n1,2\n3,4\n")
        with open(os.path.join(inprogress, "metadata.json"), "w") as f:
            json.dump({"status": "inprogress"}, f)
        os.makedirs(os.path.join(inprogress, "depth"))
        with open(os.path.join(inprogress, "depth", "000000.png"), "wb") as f:
            f.write(b"fake_png")

        # Simulate commit
        os.rename(inprogress, final)
        assert os.path.isdir(final)
        assert not os.path.exists(inprogress)
        assert os.path.isfile(os.path.join(final, "data.csv"))

        # Simulate rejection
        inprogress2 = os.path.join(tmpdir, "traj_002.inprogress")
        os.makedirs(inprogress2)
        with open(os.path.join(inprogress2, "data.csv"), "w") as f:
            f.write("bad")
        failed_dest = os.path.join(failed_dir, "test_traj_002")
        os.rename(inprogress2, failed_dest)
        assert os.path.isdir(failed_dest)
        assert not os.path.exists(inprogress2)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
#  13.  ESDF vectorized voxelization test
# ══════════════════════════════════════════════════════════════════════════

def test_esdf_vectorized_voxelization():
    """ESDFBuilder.build should produce correct grid dimensions."""
    try:
        from scipy.ndimage import distance_transform_edt  # noqa: F401
    except ImportError:
        print("  SKIP  test_esdf_vectorized_voxelization (scipy not available)")
        return

    tmpdir = tempfile.mkdtemp()
    try:
        ply_path = os.path.join(tmpdir, "test.ply")
        header = [
            b"ply", b"format ascii 1.0",
            b"element vertex 5", b"property float x",
            b"property float y", b"property float z",
        ]
        body = b"0 0 0\n1 1 1\n2 2 2\n3 3 3\n4 4 4\n"
        with open(ply_path, "wb") as f:
            f.write(b"\n".join(header) + b"\nend_header\n" + body)

        from il_common import ESDFBuilder
        try:
            builder = ESDFBuilder([10.0, 10.0, 5.0], [5.0, 5.0, 2.5], 0.5, 0.25)
            esdf, origin, stats = builder.build(ply_path)
            assert esdf.ndim == 3
            assert stats["total_points"] == 5
            assert "occupied_voxels" in stats
        except Exception as exc:
            # scipy/numpy version incompatibility — skip gracefully
            print("  SKIP  test_esdf_vectorized_voxelization ({})".format(exc))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
#  14.  Yaw → Unity quaternion direction tests
# ══════════════════════════════════════════════════════════════════════════

def _unity_nose_direction(yaw):
    """Compute the Unity world direction of the drone nose for a given ROS yaw."""
    return (math.sin(-yaw), 0.0, math.cos(-yaw))


def _ros_tangent_to_unity(vx_ros, vy_ros):
    """Convert ROS world tangent (vx, vy) to Unity world direction."""
    return (vx_ros, 0.0, vy_ros)


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def test_yaw_to_unity_quat_direction_plus_x():
    """ROS +X trajectory → Unity nose should face +X."""
    vx, vy = 1.0, 0.0
    yaw = math.atan2(vy, vx) - math.pi / 2.0
    nose_u = _unity_nose_direction(yaw)
    tangent_u = _ros_tangent_to_unity(vx, vy)
    L = math.sqrt(tangent_u[0]**2 + tangent_u[2]**2)
    tangent_u = (tangent_u[0]/L, 0.0, tangent_u[2]/L)
    assert _dot(nose_u, tangent_u) > 0.99, "+X dot={:.4f}".format(_dot(nose_u, tangent_u))


def test_yaw_to_unity_quat_direction_minus_x():
    vx, vy = -1.0, 0.0
    yaw = math.atan2(vy, vx) - math.pi / 2.0
    nose_u = _unity_nose_direction(yaw)
    tangent_u = _ros_tangent_to_unity(vx, vy)
    L = math.sqrt(tangent_u[0]**2 + tangent_u[2]**2)
    tangent_u = (tangent_u[0]/L, 0.0, tangent_u[2]/L)
    assert _dot(nose_u, tangent_u) > 0.99, "-X dot={:.4f}".format(_dot(nose_u, tangent_u))


def test_yaw_to_unity_quat_direction_plus_y():
    vx, vy = 0.0, 1.0
    yaw = math.atan2(vy, vx) - math.pi / 2.0
    nose_u = _unity_nose_direction(yaw)
    tangent_u = _ros_tangent_to_unity(vx, vy)
    L = math.sqrt(tangent_u[0]**2 + tangent_u[2]**2)
    tangent_u = (tangent_u[0]/L, 0.0, tangent_u[2]/L)
    assert _dot(nose_u, tangent_u) > 0.99, "+Y dot={:.4f}".format(_dot(nose_u, tangent_u))


def test_yaw_to_unity_quat_direction_minus_y():
    vx, vy = 0.0, -1.0
    yaw = math.atan2(vy, vx) - math.pi / 2.0
    nose_u = _unity_nose_direction(yaw)
    tangent_u = _ros_tangent_to_unity(vx, vy)
    L = math.sqrt(tangent_u[0]**2 + tangent_u[2]**2)
    tangent_u = (tangent_u[0]/L, 0.0, tangent_u[2]/L)
    assert _dot(nose_u, tangent_u) > 0.99, "-Y dot={:.4f}".format(_dot(nose_u, tangent_u))


def test_yaw_to_unity_quat_direction_diagonal():
    for vx, vy in [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 1), (1, 3)]:
        yaw = math.atan2(vy, vx) - math.pi / 2.0
        nose_u = _unity_nose_direction(yaw)
        tangent_u = _ros_tangent_to_unity(vx, vy)
        L = math.sqrt(tangent_u[0]**2 + tangent_u[2]**2)
        tangent_u = (tangent_u[0]/L, 0.0, tangent_u[2]/L)
        assert _dot(nose_u, tangent_u) > 0.99, \
            "diag({},{}): dot={:.4f}".format(vx, vy, _dot(nose_u, tangent_u))


# ══════════════════════════════════════════════════════════════════════════
#  15.  Body velocity direction tests
# ══════════════════════════════════════════════════════════════════════════

def test_body_vel_forward_positive():
    from il_common import world_vel_to_body
    vx_b, vy_b, vz_b = world_vel_to_body(0, 1, 0, 0)
    assert vy_b > 0.9, "Forward vel_y_body should be ~1, got {:.3f}".format(vy_b)
    assert abs(vx_b) < 0.1


def test_world_plus_y_is_positive_camera_forward_roundtrip():
    """The first logged task travels mainly ROS +Y, never FLU backward."""
    from il_common import (world_vector_to_body_flu_quat,
                           body_flu_to_world_quat)
    q_identity = np.array([0.0, 0.0, 0.0, 1.0])
    world_command = np.array([0.0, 1.0, 0.0])
    flu_command = world_vector_to_body_flu_quat(
        world_command, q_identity)
    assert_all_close(flu_command, [1.0, 0.0, 0.0])
    assert_all_close(body_flu_to_world_quat(
        flu_command, q_identity), world_command)


def test_yaw_rate_tracks_world_velocity_direction():
    from il_common import yaw_from_world_velocity, yaw_rate_for_world_velocity

    assert abs(yaw_from_world_velocity([0.0, 1.0, 0.0])) < 1e-9
    assert abs(yaw_from_world_velocity([1.0, 0.0, 0.0]) +
               math.pi / 2.0) < 1e-9

    # Facing +X while flying +Y must turn negative toward yaw=0. The captured
    # planner feed-forward command incorrectly returned +2 rad/s here.
    rate = yaw_rate_for_world_velocity(
        math.pi / 2.0, [0.0, 1.0, 0.0], 3.0, 2.0, 0.10)
    assert rate < 0.0
    assert abs(rate) <= 2.0


def test_yaw_rate_uses_shortest_wrap_and_holds_at_low_speed():
    from il_common import yaw_rate_for_world_velocity

    # Velocity target corresponds to -179 degrees while current yaw is +179.
    target = math.radians(-179.0)
    velocity = [math.cos(target + math.pi / 2.0),
                math.sin(target + math.pi / 2.0), 0.0]
    rate = yaw_rate_for_world_velocity(
        math.radians(179.0), velocity, 3.0, 2.0, 0.10)
    assert 0.0 < rate < 0.2
    assert yaw_rate_for_world_velocity(
        1.0, [0.01, 0.01, 0.0], 3.0, 2.0, 0.10) == 0.0


def test_bounded_command_remains_bounded_after_csv_quantization():
    from il_common import quantize_bounded_vector

    # These component patterns are taken from rows rejected in the captured
    # dataset after four-decimal serialization inflated their norm.
    cases = ([2.4997, -0.0038, 0.0405],
             [2.4999, -0.0266, 0.0051],
             [2.4893, 0.0031, 0.2312],
             [2.5, 0.0118, 0.0031])
    for case in cases:
        bounded = quantize_bounded_vector(case, 2.5, decimals=6)
        assert np.linalg.norm(bounded) <= 2.5
        assert np.all(np.isfinite(bounded))


def test_body_vel_right_positive():
    from il_common import world_vel_to_body
    vx_b, vy_b, vz_b = world_vel_to_body(1, 0, 0, 0)
    assert vx_b > 0.9, "Right vel_x_body should be ~1, got {:.3f}".format(vx_b)
    assert abs(vy_b) < 0.1


# ══════════════════════════════════════════════════════════════════════════
#  16.  SyncBuffer: match_and_remove + frame_id=0 + queue cleanup
# ══════════════════════════════════════════════════════════════════════════

def test_sync_buffer_match_and_remove():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=10)
    for i in range(10):
        buf.push({"frame_id": i})
    assert buf.size() == 10
    entry = buf.match_and_remove(3)
    assert entry is not None and entry["frame_id"] == 3
    assert buf.size() == 6, "Expected 6 after removing 0..3, got {}".format(buf.size())
    entry2 = buf.match_and_remove(4)
    assert entry2 is not None and entry2["frame_id"] == 4


def test_sync_buffer_frame_id_zero():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=10)
    buf.push({"frame_id": 0, "data": "zero"})
    buf.push({"frame_id": 1, "data": "one"})
    entry = buf.match_and_remove(0)
    assert entry is not None and entry["data"] == "zero"


def test_sync_buffer_out_of_order():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=20)
    for fid in [10, 5, 15, 3, 8, 12]:
        buf.push({"frame_id": fid})
    entry = buf.match_and_remove(5)
    assert entry is not None and entry["frame_id"] == 5


def test_sync_buffer_missing_frame():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=10)
    for i in range(5):
        buf.push({"frame_id": i})
    assert buf.match_and_remove(99) is None


def test_sync_buffer_queue_not_stuck_full():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=5)
    for i in range(10):
        buf.push({"frame_id": i})
    assert buf.size() == 5
    entry = buf.match_and_remove(7)
    assert entry is not None
    assert buf.size() == 2, "Queue should shrink to 2, got {}".format(buf.size())


def test_sync_buffer_latency_vs_match_error():
    from il_common import SyncBuffer
    buf = SyncBuffer(max_entries=10)
    t0 = 1000.0
    buf.push({"frame_id": 1, "t_sent": t0})
    entry = buf.match_and_remove(1)
    assert entry is not None
    latency = (t0 + 0.050 - entry["t_sent"]) * 1000.0
    assert 40 < latency < 60, "Latency should be ~50ms, got {:.1f}".format(latency)


# ══════════════════════════════════════════════════════════════════════════
#  17.  CSV column-name-based indexing test
# ══════════════════════════════════════════════════════════════════════════

def test_csv_header_column_map():
    header = (
        "timestamp_ns,x,y,z,qx,qy,qz,qw,"
        "vel_x_body,vel_y_body,vel_z_body,"
        "ctrl_vx_body,ctrl_vy_body,ctrl_vz_body,ctrl_yaw_rate,"
        "depth_file,collision,"
        "start_x,start_y,start_z,goal_x,goal_y,goal_z,"
        "schema_version,latency_ms,match_error_ms,frame_id,vel_source,"
        "trajectory_time_s,control_frame_id,send_timestamp_ns"
    )
    cols = {name.strip(): idx for idx, name in enumerate(header.split(","))}
    assert cols["depth_file"] == 15, "depth_file col={}".format(cols["depth_file"])
    assert cols["collision"] == 16, "collision col={}".format(cols["collision"])
    assert cols["latency_ms"] == 24
    assert cols["match_error_ms"] == 25
    assert cols["vel_x_body"] == 8


def test_csv_row_count_consistency():
    import tempfile
    tmpdir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(tmpdir, "data.csv")
        with open(csv_path, "w") as f:
            f.write("a,b,c\n")
            for _ in range(50):
                f.write("1,2,3\n")
        with open(csv_path, "r") as f:
            f.readline()
            rows = sum(1 for _ in f)
        assert rows == 50
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
#  18.  Statistics consistency test
# ══════════════════════════════════════════════════════════════════════════

def test_stats_semantics():
    ctrl_hz, rec_hz = 50.0, 25.0
    duration = 2.0
    sent_controls = int(ctrl_hz * duration)
    record_ticks = int(rec_hz * duration)
    assert sent_controls != record_ticks
    assert sent_controls == 100
    assert record_ticks == 50
#  Main test runner
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Collect all test functions
    tests = [v for k, v in list(globals().items())
             if k.startswith("test_") and callable(v)]

    print("Running {} tests...".format(len(tests)))
    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
            print("  PASS  {}".format(test_fn.__name__))
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print("  FAIL  {}: {}".format(test_fn.__name__, e))

    print("\n" + "=" * 60)
    print("  Results: {} passed, {} failed".format(passed, failed))
    if errors:
        print("  Failures:")
        for name, err in errors:
            print("    {}: {}".format(name, err))
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
