#!/usr/bin/env python3
"""
il_common.py  —  Shared low-level utilities for IL dataset collection.

Contains:
  - Coordinate helpers (ROS <-> Unity, world <-> body FLU, yaw convention B)
  - Vehicle / camera builders (Unity)
  - PLY loader (ASCII / binary, arbitrary properties)
  - Unity ZMQ bridge and frame synchronization buffer

Yaw convention (unified across the package):
    yaw = atan2(world_y, world_x) - pi/2   (yaw=0 => body +Y faces world +Y)
    nose world direction = (-sin yaw, cos yaw); body quaternion = rotZ(yaw).
"""

from __future__ import print_function, division

import json, math, os, sys, time, struct, threading
import numpy as np

import rospy

try:
    import zmq
except ImportError:
    sys.exit("pyzmq not installed –  sudo apt install python3-zmq")

# ============================================================================
#  Coordinate helpers
# ============================================================================

def ros_pos_to_unity(p):
    """ROS (x-fwd, y-left, z-up) -> Unity (x-right, y-up, z-fwd)."""
    return [p[0], p[2], p[1]]


def yaw_to_unity_quat(yaw):
    """ROS yaw (z-up) -> Unity quaternion [x, y, z, w] (y-up)."""
    half = -0.5 * yaw
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def ros_quat_to_unity_quat(quaternion_xyzw):
    """Convert a full ROS-world body quaternion to Unity coordinates."""
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion_xyzw must contain four finite values")
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    r_ros = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    basis = np.array([[1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0]], dtype=np.float64)
    r = basis.dot(r_ros).dot(basis.T)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    out = np.array([qx, qy, qz, qw], dtype=np.float64)
    out /= max(float(np.linalg.norm(out)), 1e-12)
    return out.tolist()


def normalize_angle(angle):
    """Wrap to [-pi, pi)."""
    while angle >= math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# ── FLU transforms (training frame: [forward, left, up], camera aligned) ──
#
# PROJECT YAW CONVENTION (unified, section XVIII):
#   yaw = atan2(world_y, world_x) - pi/2
#   yaw = 0  =>  nose (forward) points toward world +Y
#   nose world direction        = (-sin yaw,  cos yaw)
#   body-left world direction   = (-cos yaw, -sin yaw)
#   FLU: +x forward, +y left, +z up.
# All yaw-only transforms below are derived from these basis vectors, so
# they match the quaternion version numerically at level attitude.

def forward_world_from_yaw(yaw):
    """Nose (FLU +x) direction in the world XY plane for a level body."""
    return np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)


def left_world_from_yaw(yaw):
    """Body-left (FLU +y) direction in the world XY plane (level body)."""
    return np.array([-math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)


def world_to_flu_xy(v_world, yaw):
    """Project a world XY vector onto the FLU XY basis (level body)."""
    v = np.asarray(v_world, dtype=np.float64)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    return np.array([
        -sin_y * v[0] + cos_y * v[1],   # forward
        -cos_y * v[0] - sin_y * v[1],   # left
    ], dtype=np.float64)


def flu_to_world_xy(v_flu, yaw):
    """Inverse of world_to_flu_xy (level body)."""
    v = np.asarray(v_flu, dtype=np.float64)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    # [f; l] = R [vx; vy]; R = [[-sin, cos], [-cos, -sin]]; R^-1 = R^T.
    return np.array([
        -sin_y * v[0] - cos_y * v[1],
        cos_y * v[0] - sin_y * v[1],
    ], dtype=np.float64)


def world_vector_to_body_flu(vector_world, yaw):
    """World vector -> FLU using yaw only (level body).  Matches the
    quaternion version numerically at level attitude (section XVIII)."""
    v = np.asarray(vector_world, dtype=np.float64)
    flu_xy = world_to_flu_xy(v, yaw)
    return np.array([flu_xy[0], flu_xy[1], v[2]], dtype=np.float64)


def quaternion_xyzw_to_rotation(quaternion_xyzw):
    """Build a body->world rotation matrix from a ROS [x,y,z,w] quaternion."""
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def world_vector_to_body_flu_quat(vector_world, quaternion_xyzw):
    """World vector -> FLU using the full body->world quaternion."""
    v = np.asarray(vector_world, dtype=np.float64)
    r = quaternion_xyzw_to_rotation(quaternion_xyzw)
    body = r.T.dot(v)
    return np.array([body[1], -body[0], body[2]], dtype=np.float64)


def body_flu_vector_to_world_quat(vector_flu, quaternion_xyzw):
    """Inverse of :func:`world_vector_to_body_flu_quat`."""
    v = np.asarray(vector_flu, dtype=np.float64)
    r = quaternion_xyzw_to_rotation(quaternion_xyzw)
    body = np.array([-v[1], v[0], v[2]], dtype=np.float64)
    return r.dot(body)


def body_flu_to_flightlib_body(vector_flu):
    """FLU [forward, left, up] -> Flightlib body [right, forward, up]."""
    v = np.asarray(vector_flu, dtype=np.float64)
    return np.array([-v[1], v[0], v[2]], dtype=np.float64)


def quantize_bounded_vector(vector, max_norm, decimals=3):
    """Norm-preserving quantization for recorded command labels."""
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm > max_norm and norm > 1e-9:
        v = v * (max_norm / norm)
    return np.round(v, decimals=decimals)


# ============================================================================
#  Vehicle / camera builders
# ============================================================================

def make_depth_vehicle(ros_pos, yaw, depth_cfg, quaternion_xyzw=None):
    """Return a Unity Vehicle_t dict with a depth camera.

    VERIFIED AvoidBench wire contract (pre-reshape, working handshake):
    the vehicle MUST be a `Vehicle_t` with `ID` / `size` / `cameras` /
    `lidars` / `hasCollisionCheck`, and the camera MUST carry
    `nearClipPlane` / `farClipPlane` / a 16-element 4x4 `T_BC` / `isDepth` /
    `enabledLayers` / `outputIndex` / `depthScale`.

    AvoidBench reads these fields during the handshake init:
      - SettingsMessage_t.InitParamsters():  mainVehicle.cameras.Count()
      - instantiateCameras():  camera.nearClipPlane[0], farClipPlane[0],
        ListToMatrix4x4(camera.T_BC)  (requires 16 floats)
    A missing / renamed field makes the handshake init throw, `ready` is
    never sent, and the manager times out with unity_connect_timeout.
    DO NOT change this schema without test proof.
    """
    pos = ros_pos_to_unity(ros_pos)
    if quaternion_xyzw is not None:
        quat = ros_quat_to_unity_quat(quaternion_xyzw)
    else:
        quat = yaw_to_unity_quat(yaw)
    near = float(depth_cfg.get("near", 0.01))
    far = float(depth_cfg.get("far", 1000.0))
    return {
        "ID": "quadrotor0",
        "position": pos,
        "rotation": quat,
        "size": [0.5, 0.5, 0.5],
        "cameras": [{
            "ID": "quadrotor0_0", "channels": 3,
            "width": int(depth_cfg.get("width", 640)),
            "height": int(depth_cfg.get("height", 480)),
            "fov": float(depth_cfg.get("fov", 90.0)),
            "nearClipPlane": [near] * 4,
            "farClipPlane": [far, 100.0, far, far],
            "T_BC": [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.3,
                     0.0, 0.0, 0.0, 1.0],
            "isDepth": False,
            "enabledLayers": [True, False, False],
            "depthScale": 0.2,
            "outputIndex": 0,
        }],
        "lidars": [],
        "hasCollisionCheck": True,
        "hasVehicleCollision": False,
    }


# ============================================================================
#  PLY loader
# ============================================================================

def _read_ply_header(f):
    header = []
    while True:
        line = f.readline()
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        header.append(line)
        if line.strip() == "end_header":
            break
    return header


def _parse_ply_header(header):
    fmt = "ascii"
    elements = {}
    order = []
    current = None
    for line in header:
        stripped = line.strip()
        if not stripped or stripped.startswith("comment"):
            continue
        tokens = stripped.split()
        if tokens[0] == "format":
            fmt = tokens[1]
        elif tokens[0] == "element":
            current = tokens[1]
            count = int(tokens[2])
            elements[current] = {"count": count, "props": []}
            order.append(current)
        elif tokens[0] == "property" and current is not None:
            if tokens[1] == "list":
                elements[current]["props"].append(
                    ("list", tokens[2], tokens[3], tokens[4]))
            else:
                elements[current]["props"].append(
                    ("scalar", tokens[1], tokens[2]))
    return fmt, elements, order


_PLY_TYPE_SIZES = {
    "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
    "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
    "int": 4, "uint": 4, "int32": 4, "uint32": 4,
    "float": 4, "float32": 4,
    "double": 8, "float64": 8,
}
_PLY_TYPE_FORMATS = {
    "char": "b", "uchar": "B", "int8": "b", "uint8": "B",
    "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
    "int": "i", "uint": "I", "int32": "i", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_ply(filepath):
    """Load vertex x/y/z from a PLY file (ASCII or binary LE/BE).

    Returns an (N, 3) float64 array.  ``filepath`` may be ``str`` or
    ``os.PathLike``.
    """
    if isinstance(filepath, (tuple, list)):
        raise TypeError("load_ply expects a single path, got a list")
    fmt, elements, order = None, None, None
    with open(filepath, "rb") as f:
        header = _read_ply_header(f)
        fmt, elements, order = _parse_ply_header(header)
        if "vertex" not in elements:
            raise ValueError("PLY has no vertex element")
        vertex = elements["vertex"]
        count = vertex["count"]
        props = vertex["props"]
        xyz = np.full((count, 3), np.nan, dtype=np.float64)
        if fmt == "ascii":
            for i in range(count):
                line = f.readline()
                if not line:
                    break
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                tokens = line.split()
                if len(tokens) >= 3:
                    xyz[i, 0] = _as_float(tokens[0])
                    xyz[i, 1] = _as_float(tokens[1])
                    xyz[i, 2] = _as_float(tokens[2])
        elif fmt.startswith("binary"):
            little = "little" in fmt
            endian = "<" if little else ">"
            offset = 0
            values = np.frombuffer(f.read(), dtype=np.uint8)
            idx = 0
            for i in range(count):
                for prop in props:
                    if prop[0] == "list":
                        count_type = prop[1]
                        elem_type = prop[3]
                        size = _PLY_TYPE_SIZES.get(count_type, 4)
                        if idx + size > len(values):
                            return xyz
                        count_val = int(np.frombuffer(
                            values[idx:idx + size].tobytes(),
                            dtype=endian + _PLY_TYPE_FORMATS.get(count_type, "i"))[0])
                        idx += size
                        elem_size = _PLY_TYPE_SIZES.get(elem_type, 4)
                        idx += count_val * elem_size
                    else:
                        ptype = prop[1]
                        size = _PLY_TYPE_SIZES.get(ptype, 4)
                        if idx + size > len(values):
                            return xyz
                        value = float(np.frombuffer(
                            values[idx:idx + size].tobytes(),
                            dtype=endian + _PLY_TYPE_FORMATS.get(ptype, "f"))[0])
                        idx += size
                        if prop[2] == "x":
                            xyz[i, 0] = value
                        elif prop[2] == "y":
                            xyz[i, 1] = value
                        elif prop[2] == "z":
                            xyz[i, 2] = value
        else:
            raise ValueError("Unsupported PLY format: %s" % fmt)
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.any():
        raise ValueError("PLY contains no finite vertices")
    return xyz[finite]


def wait_for_stable_file(filepath, timeout=60.0, stable_s=0.5, min_bytes=1):
    """Wait until a file exists and its size is stable."""
    deadline = time.time() + timeout
    last_size = -1
    while time.time() < deadline:
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            if size >= min_bytes and size == last_size:
                return True
            last_size = size
        time.sleep(stable_s)
    return os.path.isfile(filepath)


# ============================================================================
#  Unity ZMQ bridge
# ============================================================================

class UnityBridge:
    """ZMQ pub/sub connection to the Unity renderer."""

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
        self.pub.bind("tcp://*:" + self.pub_port)
        self.sub.bind("tcp://*:" + self.sub_port)
        self._bound = True

    def close(self):
        if self._bound:
            try:
                self.pub.close(0)
                self.sub.close(0)
                self.ctx.term()
            except Exception:
                pass
            self._bound = False

    def try_recv(self):
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
        with self._send_lock:
            self.pub.send_multipart([b"Pose", json.dumps(msg_dict).encode("utf-8")])

    def send_pc_request(self, req_dict):
        with self._send_lock:
            self.pub.send_multipart([b"PointCloud", json.dumps(req_dict).encode("utf-8")])

    def connect_handshake(self, scene_id, depth_cfg, timeout=60.0):
        vehicle = make_depth_vehicle([0, 0, 5], 0, depth_cfg)
        handshake_msg = {"scene_id": scene_id, "vehicles": [vehicle],
                         "objects": []}
        deadline = time.time() + timeout
        rospy.loginfo(
            "[Bridge] Waiting for AvoidBench ready handshake (scene id=%d, "
            "tcp://*:%s / tcp://*:%s)...", scene_id, self.pub_port,
            self.sub_port)
        last_log = 0.0
        while time.time() < deadline and not rospy.is_shutdown():
            self.send_pose(handshake_msg)
            r = self.try_recv()
            if r is not None and r[0].get("ready"):
                return True
            now = time.time()
            if now - last_log >= 5.0:
                rospy.logwarn(
                    "[Bridge] Still waiting for AvoidBench ready (%.0fs/%0.fs) "
                    "— is AvoidBench running and connected to this host? "
                    "Start it with: ./AvoidBench.x86_64 -input-port %s "
                    "-output-port %s -client-ip 127.0.0.1",
                    now - (deadline - timeout), timeout, self.pub_port,
                    self.sub_port)
                last_log = now
            time.sleep(0.2)
        return False
