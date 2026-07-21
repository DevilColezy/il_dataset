#!/usr/bin/env python3
"""
il_common.py  —  Shared components for IL dataset collection.

Contains:
  - Coordinate helpers (ROS ↔ Unity, world ↔ body)
  - Vehicle / camera builders
  - PLY loader (ASCII, CRLF, binary LE/BE, arbitrary vertex properties)
  - ESDF builder (with statistics)
  - Unity ZMQ bridge (with proper lifecycle)
  - Obstacle generator
  - Start/goal pair generator
  - Frame synchronization buffer
"""

from __future__ import print_function, division

import json, math, os, sys, time, random, struct, hashlib, threading
import numpy as np

import rospy
import rospkg

try:
    import zmq
except ImportError:
    sys.exit("pyzmq not installed –  sudo apt install python3-zmq")

try:
    from PIL import Image
except ImportError:
    Image = None

# ============================================================================
#  Coordinate helpers
# ============================================================================

def ros_pos_to_unity(p):
    """ROS (x-fwd, y-left, z-up) → Unity (x-right, y-up, z-fwd)."""
    return [p[0], p[2], p[1]]


def yaw_to_unity_quat(yaw):
    """ROS yaw (z-up) → Unity quaternion [x, y, z, w] (y-up).

    Derivation:
      ROS world: X-fwd, Y-left, Z-up.  Positive yaw rotates nose from +Y toward +X.
      ROS→Unity pos mapping: (rx, ry, rz) → (rx, rz, ry).
      In Unity: X-right, Y-up, Z-fwd.  Rotation around Y-up axis.
      Unity Y+ rotation moves local +Z (fwd) toward +X.
      ROS +yaw moves nose from +Y_world toward +X_world.
      After mapping: +Y_world → +Z_unity, +X_world → +X_unity.
      So +yaw should move Unity fwd (+Z) toward +X.
      Unity Y+ rotation does exactly this.  Therefore Unity yaw = +ros_yaw.
      HOWEVER: the Unity vehicle's local forward is +Z, and the quaternion
      [0, sin(θ/2), 0, cos(θ/2)] rotates around Y by θ.
      R_y(+π/2)·(0,0,1) = (+1,0,0) = +X_unity.  ✓
      For ROS +X motion: yaw = -π/2.  We need Unity nose = +X.
      R_y(+π/2)·(0,0,1) = +X.  So θ = +π/2 = -yaw.  → θ = -ros_yaw.
      Hence: half = -0.5 * yaw.
    """
    half = -0.5 * yaw
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def world_vel_to_body(vx_w, vy_w, vz_w, yaw):
    """Convert velocity from ROS world frame to drone body frame.

    ROS world:  X-fwd, Y-left, Z-up
    Drone body: X-right, Y-fwd(nose), Z-up

    At yaw=0 the nose (body +Y) points to world +Y (left).
    At yaw=-π/2 the nose points to world +X (forward).

    Args:
        vx_w, vy_w, vz_w:  velocity in ROS world frame
        yaw:  drone yaw (0 = nose faces world +Y)
    Returns:
        (vx_body, vy_body, vz_body)   vx=right, vy=forward, vz=up
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_b =  vx_w * c + vy_w * s     # right
    vy_b = -vx_w * s + vy_w * c     # forward (nose)
    vz_b =  vz_w
    return vx_b, vy_b, vz_b


def body_vel_to_world(vx_b, vy_b, vz_b, yaw):
    """Inverse of world_vel_to_body."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    vx_w = vx_b * c - vy_b * s
    vy_w = vx_b * s + vy_b * c
    vz_w = vz_b
    return vx_w, vy_w, vz_w


def normalize_angle(a):
    """Wrap angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


def shortest_angle_diff(a, b):
    """Shortest signed angle from a to b, in [-π, π]."""
    return normalize_angle(b - a)


def integrate_velocity_command(position, velocity, yaw, desired_velocity, dt,
                               max_velocity, max_acceleration,
                               max_yaw_rate, yaw_speed_threshold=0.05):
    """Integrate a planner velocity command without pose discontinuities."""
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    pos = np.asarray(position, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    desired = np.asarray(desired_velocity, dtype=np.float64)
    if pos.shape != (3,) or vel.shape != (3,) or desired.shape != (3,):
        raise ValueError("position, velocity and desired_velocity must be 3-vectors")
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)) and
            np.all(np.isfinite(desired))):
        raise ValueError("velocity executor inputs must be finite")

    desired_speed = float(np.linalg.norm(desired))
    if desired_speed > max_velocity:
        desired = desired * (max_velocity / desired_speed)

    delta_v = desired - vel
    max_delta_v = max_acceleration * dt
    delta_norm = float(np.linalg.norm(delta_v))
    if delta_norm > max_delta_v:
        delta_v *= max_delta_v / delta_norm
    next_vel = vel + delta_v

    next_speed = float(np.linalg.norm(next_vel))
    if next_speed > max_velocity:
        next_vel *= max_velocity / next_speed

    # Trapezoidal integration is continuous even when a new plan replaces the
    # desired velocity between two controller ticks.
    next_pos = pos + 0.5 * (vel + next_vel) * dt

    horizontal_speed = float(np.linalg.norm(next_vel[:2]))
    if horizontal_speed > yaw_speed_threshold:
        # Project convention: yaw=0 points along world +Y.
        desired_yaw = math.atan2(next_vel[1], next_vel[0]) - math.pi / 2.0
        yaw_error = shortest_angle_diff(yaw, desired_yaw)
        yaw_step = max(-max_yaw_rate * dt,
                       min(max_yaw_rate * dt, yaw_error))
    else:
        yaw_step = 0.0
    next_yaw = normalize_angle(yaw + yaw_step)
    yaw_rate = yaw_step / dt

    return next_pos, next_vel, next_yaw, yaw_rate


# ============================================================================
#  FLU coordinate conversion helpers  (schema v7)
# ============================================================================

def body_rfu_to_flu(vector):
    """RFU [right, forward, up] -> FLU [forward, left, up].

    Training coordinate frame for neural network input/output.
    """
    right, forward, up = vector
    return np.array([forward, -right, up], dtype=np.float64)


def body_flu_to_rfu(vector):
    """FLU [forward, left, up] -> RFU [right, forward, up].

    Inverse of body_rfu_to_flu.
    """
    forward, left, up = vector
    return np.array([-left, forward, up], dtype=np.float64)


def world_vector_to_body_flu(vector_world, yaw):
    """ROS world vector -> body/navigation FLU.

    Convenience wrapper: world -> body RFU (via world_vel_to_body)
    then RFU -> FLU (via body_rfu_to_flu).

    Args:
        vector_world: 3-vector in ROS world frame (X-fwd, Y-left, Z-up).
        yaw: drone yaw in radians.

    Returns:
        np.array of shape (3,) in FLU: [forward, left, up].
    """
    right, forward, up = world_vel_to_body(
        float(vector_world[0]),
        float(vector_world[1]),
        float(vector_world[2]),
        float(yaw),
    )
    return body_rfu_to_flu([right, forward, up])


# ============================================================================
#  Vehicle / camera builders  (KEPT EXACTLY AS ORIGINAL – compatibility)
# ============================================================================

def make_depth_vehicle(ros_pos, yaw, depth_cfg):
    """Return a Unity vehicle dict with a depth camera.

    **DO NOT MODIFY** the camera configuration, T_BC, depthScale,
    isDepth, enabledLayers, or depth value conversion logic without
    explicit test proof of correctness.
    """
    return {
        "ID": "quadrotor0",
        "position": ros_pos_to_unity(ros_pos),
        "rotation": yaw_to_unity_quat(yaw),
        "size": [0.5, 0.5, 0.5],
        "cameras": [{
            "ID": "quadrotor0_0", "channels": 3,
            "width": depth_cfg["width"], "height": depth_cfg["height"],
            "fov": depth_cfg["fov"],
            "nearClipPlane": [depth_cfg["near"]] * 4,
            "farClipPlane": [depth_cfg["far"], 100, depth_cfg["far"], depth_cfg["far"]],
            "T_BC": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.3, 0, 0, 0, 1],
            "isDepth": False, "enabledLayers": [True, False, False],
            "depthScale": 0.2, "outputIndex": 0,
        }],
        "lidars": [], "hasCollisionCheck": True,
    }


def make_dummy_vehicle():
    return {"ID": "quadrotor0", "position": [0, 0, 0],
            "rotation": [0, 0, 0, 1], "size": [1, 1, 1],
            "cameras": [], "lidars": [], "hasCollisionCheck": True}


# ============================================================================
#  PLY loader  —  robust ASCII / binary LE / BE, arbitrary vertex properties
# ============================================================================

# Map PLY type tokens to Python struct format characters
_PLY_TYPE_SIZES = {
    "char":    (1, "b"),   "int8":    (1, "b"),
    "uchar":   (1, "B"),   "uint8":   (1, "B"),
    "short":   (2, "h"),   "int16":   (2, "h"),
    "ushort":  (2, "H"),   "uint16":  (2, "H"),
    "int":     (4, "i"),   "int32":   (4, "i"),
    "uint":    (4, "I"),   "uint32":  (4, "I"),
    "float":   (4, "f"),   "float32": (4, "f"),
    "double":  (8, "d"),   "float64": (8, "d"),
}

# Big-endian variants remap the struct char
_BE_REMAP = {"b": "b", "B": "B", "h": ">h", "H": ">H",
             "i": ">i", "I": ">I", "f": ">f", "d": ">d"}


def load_ply(filepath):
    """Load (N,3) float32 array from a PLY file.

    Supports:
      - ASCII, binary_little_endian, binary_big_endian
      - LF and CRLF line endings in header
      - Vertex properties other than just xyz (color, normals, etc.)
      - Non-contiguous xyz columns (reads correct byte offset per property)

    Returns:
        np.ndarray of shape (N, 3), dtype float32

    Raises:
        ValueError on unparseable or empty files.
    """
    # ── Check file exists and is non-empty ──────────────────────
    if not os.path.isfile(filepath):
        raise ValueError("PLY file not found: {}".format(filepath))
    fsize = os.path.getsize(filepath)
    if fsize == 0:
        raise ValueError("PLY file is empty: {}".format(filepath))

    with open(filepath, "rb") as f:
        raw = f.read()

    if len(raw) == 0:
        raise ValueError("PLY file read 0 bytes: {}".format(filepath))

    # ── Find header end (handle CRLF) ──────────────────────────
    header_end = raw.find(b"end_header")
    if header_end < 0:
        raise ValueError("Not a valid PLY file (no 'end_header' marker): {}".format(filepath))

    body_start = header_end + len(b"end_header")
    # Skip line ending after end_header
    if body_start < len(raw) and raw[body_start:body_start + 1] == b"\n":
        body_start += 1
    elif body_start + 1 < len(raw) and raw[body_start:body_start + 2] == b"\r\n":
        body_start += 2

    header_text = raw[:header_end].decode("ascii", errors="replace")
    body_bytes = raw[body_start:]

    # ── Parse header ───────────────────────────────────────────
    fmt = "ascii"
    n_verts = 0
    vertex_props = []  # list of (name, ply_type_string)

    for line in header_text.splitlines():
        line = line.strip()
        if line.startswith("format "):
            parts = line.split()
            if len(parts) >= 2:
                fmt = parts[1]
        if line.startswith("element vertex"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    n_verts = int(parts[-1])
                except ValueError:
                    pass
        if line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3:
                vertex_props.append((parts[2], parts[1]))  # (name, type)

    if n_verts <= 0:
        rospy.logwarn("[PLY] vertex count=%d in %s", n_verts, os.path.basename(filepath))
        return np.zeros((0, 3), dtype=np.float32)

    rospy.loginfo("[PLY] %d vertices, format=%s, %d properties from %s",
                  n_verts, fmt, len(vertex_props), os.path.basename(filepath))

    # ── ASCII path ─────────────────────────────────────────────
    if "ascii" in fmt:
        text = body_bytes.decode("ascii", errors="replace").strip()
        rows = [ln.split() for ln in text.split("\n") if ln.strip()]
        if not rows:
            return np.zeros((0, 3), dtype=np.float32)
        data = np.array(rows, dtype=np.float32)
        if data.shape[1] < 3:
            raise ValueError("PLY has < 3 columns: {}".format(filepath))
        xyz = data[:, :3].astype(np.float32)
        _validate_ply_points(xyz, filepath)
        return xyz

    # ── Binary path ────────────────────────────────────────────
    if "binary" not in fmt:
        raise ValueError("Unknown PLY format: {}".format(fmt))

    big_endian = "big_endian" in fmt

    # Compute byte offsets and stride from property list
    xyz_indices = []  # which property indices are x, y, z
    xyz_offsets = []  # byte offsets of x, y, z within a vertex record
    stride = 0
    for idx, (name, ptype) in enumerate(vertex_props):
        sz, _ = _PLY_TYPE_SIZES.get(ptype, (4, "f"))
        if name in ("x", "y", "z"):
            xyz_indices.append(idx)
            xyz_offsets.append(stride)
        stride += sz

    if stride == 0:
        stride = 12  # fallback: assume 3 × float32

    # Trim body to whole vertices
    n_actual = len(body_bytes) // stride
    if n_actual == 0:
        rospy.logwarn("[PLY] Zero complete vertices in body (stride=%d, body=%d bytes).",
                      stride, len(body_bytes))
        return np.zeros((0, 3), dtype=np.float32)

    body_bytes = body_bytes[:n_actual * stride]

    # ── Extract xyz with correct stride (AOS format) ──────────
    pts = np.zeros((n_actual, 3), dtype=np.float32)

    # Determine if all properties are float32 (common case)
    all_float32 = all(
        _PLY_TYPE_SIZES.get(pt, (4, "f"))[0] == 4
        for _, pt in vertex_props)

    if all_float32 and len(xyz_offsets) >= 3:
        # Fast path: all properties are 4-byte floats, body is float32-aligned
        flat_all = np.frombuffer(body_bytes, dtype=np.float32)
        if big_endian:
            flat_all = flat_all.byteswap()
        stride_f32 = stride // 4
        col_f32 = [off // 4 for off in xyz_offsets[:3]]
        for c, start in enumerate(col_f32):
            pts[:, c] = flat_all[start::stride_f32][:n_actual]
    elif len(xyz_offsets) >= 3:
        # Mixed-type record: read each xyz float32 at its byte offset
        for v in range(n_actual):
            base = v * stride
            for c, off in enumerate(xyz_offsets[:3]):
                bstart = base + off
                pts[v, c] = np.frombuffer(
                    body_bytes[bstart:bstart + 4], dtype=np.float32)[0]
        if big_endian:
            pts = pts.byteswap()
    else:
        # Fallback: assume first 12 bytes per vertex are 3 × float32
        usable_len = (len(body_bytes) // 4) * 4
        flat_all = np.frombuffer(body_bytes[:usable_len], dtype=np.float32)
        if big_endian:
            flat_all = flat_all.byteswap()
        stride_f32 = max(stride // 4, 3)
        if stride_f32 == 3:
            pts = flat_all[:n_actual * 3].reshape(-1, 3)
        else:
            for i in range(n_actual):
                base = i * stride_f32
                if base + 2 < len(flat_all):
                    pts[i, :] = flat_all[base:base + 3]

    _validate_ply_points(pts, filepath)
    return pts


def _validate_ply_points(pts, filepath):
    """Check that loaded points are reasonable."""
    if len(pts) == 0:
        return
    n_finite = np.isfinite(pts).all(axis=1).sum()
    if n_finite < len(pts):
        rospy.logwarn("[PLY] %d/%d non-finite points in %s",
                      len(pts) - n_finite, len(pts), os.path.basename(filepath))
    # Check for NaN/Inf
    if not np.all(np.isfinite(pts)):
        rospy.logwarn("[PLY] Contains NaN/Inf values – cleaning")
        pts = pts[np.isfinite(pts).all(axis=1)]


def wait_for_stable_file(filepath, stable_sec=1.0, max_wait=30.0):
    """Wait until file exists and size stops changing for *stable_sec* seconds.

    Returns True if file is stable, False on timeout.
    """
    deadline = time.time() + max_wait
    last_size = -1
    stable_since = 0.0
    while time.time() < deadline:
        if os.path.exists(filepath):
            sz = os.path.getsize(filepath)
            if sz == last_size and sz > 0:
                if stable_since == 0.0:
                    stable_since = time.time()
                elif time.time() - stable_since >= stable_sec:
                    return True
            else:
                stable_since = 0.0
                last_size = sz
        time.sleep(0.1)
    return False


# ============================================================================
#  ESDF builder
# ============================================================================

class ESDFBuilder:
    """Build an Euclidean Signed Distance Field from a PLY point cloud.

    Uses vectorized voxelization (no per-point Python loop in the hot path).
    """

    def __init__(self, pc_range, pc_origin, esdf_res, drone_radius):
        self.range = list(pc_range)   # [rx, ry, rz]
        self.origin = list(pc_origin) # [ox, oy, oz] – center of the volume
        self.res = esdf_res
        self.drone_r = drone_radius

    @staticmethod
    def load_ply(filepath):
        """Delegate to module-level load_ply."""
        return load_ply(filepath)

    def build(self, ply_path):
        """Return (esdf_3d_array, grid_origin_tuple, stats_dict)."""
        from scipy.ndimage import distance_transform_edt

        t0 = time.time()
        pts = self.load_ply(ply_path)
        n_pts = len(pts)

        # ── Grid dimensions ────────────────────────────────────
        gx = int(math.floor(self.range[0] / self.res)) + 1
        gy = int(math.floor(self.range[1] / self.res)) + 1
        gz = int(math.floor(self.range[2] / self.res)) + 1

        origin = (self.origin[0] - self.range[0] / 2.0,
                  self.origin[1] - self.range[1] / 2.0,
                  self.origin[2] - self.range[2] / 2.0)

        inv_res = 1.0 / self.res
        ox, oy, oz = origin

        # ── Vectorized voxelization ────────────────────────────
        # Convert points to grid indices using floor
        ix = np.floor((pts[:, 0] - ox) * inv_res).astype(np.int32)
        iy = np.floor((pts[:, 1] - oy) * inv_res).astype(np.int32)
        iz = np.floor((pts[:, 2] - oz) * inv_res).astype(np.int32)

        # Filter to valid range
        mask = (ix >= 0) & (ix < gx) & (iy >= 0) & (iy < gy) & (iz >= 0) & (iz < gz)
        n_in_range = mask.sum()
        n_out = n_pts - n_in_range

        occ = np.zeros((gx, gy, gz), dtype=np.uint8)
        np.add.at(occ, (ix[mask], iy[mask], iz[mask]), 1)
        occ = (occ > 0).astype(np.uint8)

        # ── EDT ────────────────────────────────────────────────
        fd = distance_transform_edt(1 - occ, sampling=self.res)
        od = distance_transform_edt(occ, sampling=self.res)
        esdf = fd - od - self.drone_r

        t_build = time.time() - t0

        # ── Statistics ─────────────────────────────────────────
        n_occ = int(occ.sum())
        mem_est_mb = esdf.nbytes / (1024 * 1024)
        stats = {
            "total_points": n_pts,
            "points_in_range": int(n_in_range),
            "points_out_of_range": int(n_out),
            "out_of_range_pct": round(100.0 * n_out / max(n_pts, 1), 1),
            "grid_shape": [gx, gy, gz],
            "occupied_voxels": n_occ,
            "occupancy_pct": round(100.0 * n_occ / max(gx * gy * gz, 1), 3),
            "esdf_min": float(esdf.min()),
            "esdf_max": float(esdf.max()),
            "esdf_mean": float(esdf.mean()),
            "memory_mb": round(mem_est_mb, 1),
            "build_time_sec": round(t_build, 2),
        }

        rospy.loginfo("[ESDF] %d pts, grid %dx%dx%d, %d occupied voxels, "
                      "min=%.2f max=%.2f, %.1f MB, %.1f s",
                      n_pts, gx, gy, gz, n_occ,
                      esdf.min(), esdf.max(), mem_est_mb, t_build)

        return esdf, origin, stats


# ============================================================================
#  Unity ZMQ Bridge  (single-owner socket model)
# ============================================================================

class UnityBridge:
    """Manages ZMQ Pub/Sub connection to AvoidBench Unity.

    Socket ownership: the bridge owns both PUB and SUB sockets.
    External threads must use the thread-safe send queue, not raw sockets.
    """

    def __init__(self, pub_port, sub_port):
        self.pub_port = str(pub_port)
        self.sub_port = str(sub_port)
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.setsockopt(zmq.RCVHWM, 10)
        self.pub.setsockopt(zmq.SNDHWM, 10)
        self.sub.setsockopt(zmq.RCVTIMEO, 200)
        self._send_lock = threading.Lock()
        self._bound = False

    def bind(self):
        """Bind to ports. Must be called once."""
        self.pub.bind("tcp://*:" + self.pub_port)
        self.sub.bind("tcp://*:" + self.sub_port)
        self._bound = True
        rospy.loginfo("[Bridge] Bound to PUB:%s SUB:%s", self.pub_port, self.sub_port)

    def close(self):
        """Close sockets and terminate context."""
        if self._bound:
            try:
                self.pub.close(0)
                self.sub.close(0)
                self.ctx.term()
            except Exception:
                pass
            self._bound = False

    def try_recv(self):
        """Non-blocking receive.

        Returns:
            (merged_dict, raw_parts) or None if no message available.
            merged_dict is the union of all JSON sub-messages.
            raw_parts are non-JSON binary parts (depth images).
        """
        try:
            parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        if not parts:
            return None

        merged = {}
        for p in parts:
            try:
                obj = json.loads(p.decode("utf-8"))
                if isinstance(obj, dict):
                    merged.update(obj)
            except (ValueError, UnicodeDecodeError):
                pass

        return merged, parts[1:] if len(parts) > 1 else []

    def send_pose(self, msg_dict):
        """Thread-safe Pose message send."""
        with self._send_lock:
            self.pub.send_multipart([b"Pose", json.dumps(msg_dict).encode("utf-8")])

    def send_pc_request(self, req_dict):
        """Send a PointCloud export request."""
        with self._send_lock:
            self.pub.send_multipart([b"PointCloud", json.dumps(req_dict).encode("utf-8")])

    def connect_handshake(self, scene_id, depth_cfg, timeout=60.0):
        """Handshake with Unity. Returns True when 'ready' received."""
        vehicle = make_depth_vehicle([0, 0, 5], 0, depth_cfg)
        settings = {"scene_id": scene_id, "vehicles": [vehicle], "objects": []}
        deadline = time.time() + timeout
        rospy.loginfo("[Bridge] Connecting to Unity (scene %d) ...", scene_id)
        while time.time() < deadline and not rospy.is_shutdown():
            self.send_pose(settings)
            r = self.try_recv()
            if r is not None and r[0].get("ready"):
                rospy.loginfo("[Bridge] Unity ready.")
                return True
            time.sleep(0.2)
        rospy.logerr("[Bridge] Connection timeout after %.0f s.", timeout)
        return False


# ============================================================================
#  Frame synchronization buffer
# ============================================================================

class SyncBuffer:
    """Match depth frames to control states using frame_id.

    If frame_id is not available in Unity messages, falls back to
    a bounded FIFO queue with maximum matching error.

    Each entry in the control ring (v5 extended):
        {
            "frame_id": int,        # sent frame_id
            "t_sent": float,        # monotonic send time
            "pos": [x, y, z],
            "quat": [qx, qy, qz, qw],
            "vel_body": [vx, vy, vz],      # actual/estimated body velocity
            "vel_world": [vx, vy, vz],     # world velocity from local planner
            "ctrl_v_body": [vx, vy, vz],   # control velocity body
            "ctrl_yr": float,              # control yaw rate
            "yaw": float,
            # ── v5 planner metadata ──
            "local_goal_world": [x, y, z],
            "global_progress_s": float,
            "global_progress_ratio": float,
            "global_progress_index": int,
            "local_goal_index": int,
            "plan_id": int,
            "plan_sample_time_s": float,
            "plan_age_ms": float,
            "planner_status": str,
            "planner_success": bool,
            "planner_compute_ms": float,
            "planner_min_clearance": float,
            "distance_to_final_goal": float,
            "state_source": str,
        }
    """

    def __init__(self, max_entries=128, max_sync_error_ms=100.0):
        self._ring = []  # newest at end
        self._max_entries = max_entries
        self._max_sync_error = max_sync_error_ms * 0.001  # to seconds
        self._lock = threading.Lock()

    def push(self, entry):
        """Add a control state entry. Thread-safe."""
        with self._lock:
            self._ring.append(dict(entry))
            while len(self._ring) > self._max_entries:
                self._ring.pop(0)

    def match_by_frame_id(self, frame_id):
        """Exact match on frame_id. Returns (entry, error_sec) or (None, None).
        Does NOT remove the entry (read-only)."""
        with self._lock:
            for entry in reversed(self._ring):
                if entry.get("frame_id") == frame_id:
                    return dict(entry), 0.0
        return None, None

    def match_and_remove(self, frame_id):
        """Exact match AND remove the matched entry plus all older entries.
        Returns the matched entry dict, or None.
        Thread-safe."""
        with self._lock:
            found_idx = None
            for i in range(len(self._ring) - 1, -1, -1):
                if self._ring[i].get("frame_id") == frame_id:
                    found_idx = i
                    break
            if found_idx is None:
                return None
            entry = dict(self._ring[found_idx])
            # Remove matched entry and all older entries
            del self._ring[:found_idx + 1]
            return entry

    def match_by_time(self, recv_time, max_error=None):
        """Match by nearest receive time (fallback). Returns (entry, error_sec)."""
        if max_error is None:
            max_error = self._max_sync_error
        with self._lock:
            if not self._ring:
                return None, None
            best = min(self._ring, key=lambda x: abs(x["t_sent"] - recv_time))
            error = abs(best["t_sent"] - recv_time)
            if error <= max_error:
                return dict(best), error
        return None, None

    def drain_to_latest(self):
        """Return the most recent entry and consume the complete buffer."""
        with self._lock:
            if not self._ring:
                return None
            latest = dict(self._ring[-1])
            # Retaining ``latest`` made the next fallback reuse the same
            # control frame, producing duplicate frame ids in the dataset.
            self._ring = []
            return latest

    def size(self):
        with self._lock:
            return len(self._ring)

    def clear(self):
        with self._lock:
            self._ring = []


# ============================================================================
#  Obstacle generator  (wraps SmartObstacleSampler from flightmare_dataset_tools)
# ============================================================================

class ObstacleGenerator:
    """Generate obstacle configurations for a given scene + seed."""

    def __init__(self, global_cfg, scene_cfg):
        self.g = global_cfg
        self.s = scene_cfg

    def _sampler_params(self, seed):
        obs = self.g["obstacle"]
        s = self.s
        return {
            "seed": seed, "count": 0,
            "count_min": s.get("obstacle_count_min", 0),
            "count_max": s.get("obstacle_count_max", 0),
            "target_occupancy": s["target_occupancy"],
            "x_min": obs["area_x"][0], "x_max": obs["area_x"][1],
            "y_min": obs["area_y"][0], "y_max": obs["area_y"][1],
            "radius_min": s["radius_min"], "radius_max": s["radius_max"],
            "height_min": obs["height"], "height_max": obs["height"],
            "min_gap": obs["min_gap"], "drone_radius": obs["drone_radius"],
            "safety_margin": obs["safety_margin"],
            "border_margin": obs["border_margin"],
            "avoid_x": obs["avoid_x"], "avoid_y": obs["avoid_y"],
            "avoid_radius": obs["avoid_radius"],
            "candidate_attempts": obs["candidate_attempts"],
            "shrink_on_fail": obs["shrink_on_fail"],
            "shrink_factor": obs["shrink_factor"],
            "scale_weights": obs.get("scale_weights", [0.70, 0.25, 0.05]),
            "ground_z": obs["ground_z"],
            "size_interval": obs["size_interval"],
        }

    def generate(self, seed):
        from export_pointcloud import SmartObstacleSampler
        sampler = SmartObstacleSampler(self._sampler_params(seed))
        obstacles = sampler.sample()
        rospy.loginfo("[ObstacleGen] seed=%d → %d obstacles", seed, len(obstacles))
        return obstacles

    @staticmethod
    def to_unity_objects(obstacles):
        """Convert sampled obstacle dicts → Unity Object_t list."""
        obj_list = []
        for obs in obstacles:
            obj_list.append({
                "ID": obs["id"],
                "prefabID": "Object",
                "position": [obs["x"], obs["z"], obs["y"]],  # ROS → Unity
                "rotation": [0, 0, 0, 1],
                "size": [obs["diameter"], obs["height"], obs["diameter"]],
            })
        return obj_list


# ============================================================================
#  Start / Goal pair generator
# ============================================================================

class StartGoalGenerator:
    """Generate random start-goal pairs with ESDF clearance validation."""

    def __init__(self, sg_cfg):
        self.cfg = sg_cfg

    def _sample_point(self, zone, rng):
        x = rng.uniform(*zone["x"])
        y = rng.uniform(*zone["y"])
        z = rng.uniform(zone["z_min"], zone["z_max"])
        return (x, y, z)

    @staticmethod
    def _esdf_clearance(pt, esdf, esdf_origin, esdf_res):
        """Return ESDF value at point pt. Negative = in collision."""
        ox, oy, oz = esdf_origin
        inv = 1.0 / esdf_res
        gx, gy, gz = esdf.shape
        # Use floor for consistent world-to-grid conversion
        ix = int(math.floor((pt[0] - ox) * inv))
        iy = int(math.floor((pt[1] - oy) * inv))
        iz = int(math.floor((pt[2] - oz) * inv))
        if 0 <= ix < gx and 0 <= iy < gy and 0 <= iz < gz:
            return float(esdf[ix, iy, iz])
        return -1.0  # outside map = invalid

    @staticmethod
    def _point_in_map(pt, esdf, esdf_origin, esdf_res):
        """Check if point is within ESDF grid bounds."""
        ox, oy, oz = esdf_origin
        inv = 1.0 / esdf_res
        gx, gy, gz = esdf.shape
        ix = int(math.floor((pt[0] - ox) * inv))
        iy = int(math.floor((pt[1] - oy) * inv))
        iz = int(math.floor((pt[2] - oz) * inv))
        return 0 <= ix < gx and 0 <= iy < gy and 0 <= iz < gz

    def generate_pairs(self, num_pairs, esdf, esdf_origin, esdf_res, seed=0):
        """Generate start-goal pairs with clearance checks.

        Returns list of {start: [x,y,z], goal: [x,y,z]}.
        Pairs that cannot find valid endpoints after max_attempts are still
        included but marked with metadata showing the failure reason.
        """
        rng = random.Random(seed)
        pairs = []
        min_dist = self.cfg.get("min_start_goal_distance", 15.0)
        min_cl = self.cfg.get("min_esdf_clearance_at_endpoints", 0.5)
        max_attempts = self.cfg.get("max_attempts", 200)

        for pair_idx in range(num_pairs):
            best_s, best_g = None, None
            valid = False
            for attempt in range(max_attempts):
                s = self._sample_point(self.cfg["start_zone"], rng)
                g = self._sample_point(self.cfg["goal_zone"], rng)
                dist = np.linalg.norm(np.array(g) - np.array(s))
                if dist < min_dist:
                    continue
                # Check points are within ESDF bounds
                if not self._point_in_map(s, esdf, esdf_origin, esdf_res):
                    continue
                if not self._point_in_map(g, esdf, esdf_origin, esdf_res):
                    continue
                cl_s = self._esdf_clearance(s, esdf, esdf_origin, esdf_res)
                cl_g = self._esdf_clearance(g, esdf, esdf_origin, esdf_res)
                if cl_s >= min_cl and cl_g >= min_cl:
                    best_s, best_g = s, g
                    valid = True
                    break
                # Track best so far (closest to valid)
                if best_s is None or (cl_s + cl_g) > 0:
                    best_s, best_g = s, g

            if valid:
                pairs.append({
                    "start": list(best_s), "goal": list(best_g),
                    "valid_endpoints": True
                })
            elif best_s is not None:
                cl_s = self._esdf_clearance(best_s, esdf, esdf_origin, esdf_res)
                cl_g = self._esdf_clearance(best_g, esdf, esdf_origin, esdf_res)
                in_map_s = self._point_in_map(best_s, esdf, esdf_origin, esdf_res)
                in_map_g = self._point_in_map(best_g, esdf, esdf_origin, esdf_res)
                pairs.append({
                    "start": list(best_s), "goal": list(best_g),
                    "valid_endpoints": False,
                    "failure_reason": (
                        "endpoint_clearance" if (not in_map_s or not in_map_g)
                        else "low_clearance_s={:.3f}_g={:.3f}".format(cl_s, cl_g)
                    ),
                })
                rospy.logwarn("[SG] Pair %d: no valid endpoint after %d attempts "
                              "(cl_s=%.3f, cl_g=%.3f, in_map_s=%s, in_map_g=%s).",
                              pair_idx + 1, max_attempts, cl_s, cl_g, in_map_s, in_map_g)
            else:
                # Complete failure – use random point
                s = self._sample_point(self.cfg["start_zone"], rng)
                g = self._sample_point(self.cfg["goal_zone"], rng)
                pairs.append({
                    "start": list(s), "goal": list(g),
                    "valid_endpoints": False,
                    "failure_reason": "no_valid_point_found",
                })
                rospy.logwarn("[SG] Pair %d: complete failure after %d attempts.",
                              pair_idx + 1, max_attempts)

        return pairs
