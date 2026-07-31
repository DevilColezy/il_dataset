#!/usr/bin/env python3
"""
Scene Grid Map Visualizer — Point Cloud → ESDF / Occupancy Grid

Two modes:
  python-only (default) : cylinder obstacles → synthetic point cloud → grid
  unity                  : cylinder obstacles → Unity render → PLY export → grid

Usage
-----
    # Python-only: fast, no Unity needed
    python3 scene_grid_visualizer.py

    # Unity mode: real render via Flightmare
    python3 scene_grid_visualizer.py --unity

    # Specify profile, resolution
    python3 scene_grid_visualizer.py --profile S05 --grid-res 0.1 --pc-res 0.05 --unity

    # Save to file
    python3 scene_grid_visualizer.py --unity --output grid.png --no-show
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# --- matplotlib ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# --- Project imports ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_SCRIPTS = os.path.join(_SCRIPT_DIR, "..", "scripts")
if _PROJECT_SCRIPTS not in sys.path:
    sys.path.insert(0, _PROJECT_SCRIPTS)

try:
    from il_scenario import YamlCylinderSceneGenerator, CylinderObstacleSpec
    from il_common import ESDFBuilder, load_ply, UnityBridge, make_dummy_vehicle
    _HAS_PROJECT = True
except ImportError as e:
    print(f"[WARN] Cannot import project modules: {e}")
    _HAS_PROJECT = False

try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# Config loading (ROS-free)
# ============================================================================

def _find_config() -> str:
    candidates = [
        os.path.join(_SCRIPT_DIR, "..", "config", "il_dataset_config.yaml"),
    ]
    try:
        import rospkg
        rp = rospkg.RosPack()
        candidates.insert(0, os.path.join(
            rp.get_path("il_dataset"), "config", "il_dataset_config.yaml"))
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def _load_config_ros_free(config_path: str):
    """Load YAML config without needing a ROS master."""
    import il_config
    _orig = il_config._apply_ros_overrides
    il_config._apply_ros_overrides = lambda cfg: None
    try:
        cfg = il_config.load_config(config_path, validate=False)
    finally:
        il_config._apply_ros_overrides = _orig
    return cfg


# ============================================================================
# Scene generation
# ============================================================================

def generate_scene(seed: int, profile_hint: str, scene_index: int = 0):
    """Generate obstacles; return (obstacles, profile, region, generator, raw_cfg)."""
    config_path = _find_config()
    if not config_path:
        raise FileNotFoundError("Cannot find il_dataset_config.yaml")

    cfg = _load_config_ros_free(config_path)
    gen = YamlCylinderSceneGenerator(cfg)

    profiles = gen.get_profiles()
    profile = next((p for p in profiles if p.name and profile_hint in p.name),
                   profiles[0])

    region = gen.obstacle_region
    print(f"[INFO] Profile: {profile.name}")
    print(f"[INFO] Region: x=[{region.x_min:.1f},{region.x_max:.1f}]  "
          f"y=[{region.y_min:.1f},{region.y_max:.1f}]  "
          f"z=[{region.z_min:.1f},{region.z_max:.1f}]")

    eff_seed = seed + getattr(profile, 'seed_offset', 0)
    obstacles, rejection, _, _ = gen.generate_scene_from_profile(
        profile, eff_seed, scene_index_in_profile=scene_index)
    if rejection:
        print(f"[WARN] Scene rejected: {rejection}")
    print(f"[INFO] {len(obstacles)} obstacles generated")

    return obstacles, profile, region, gen, cfg


# ============================================================================
# Unity Point Cloud export (standalone ZMQ, no rospy needed)
# ============================================================================

def export_pointcloud_unity(
    obstacles: List,
    gen,
    cfg: dict,
    output_dir: str,
    scene_label: str = "grid_viz",
    timeout_s: float = 60.0,
    settle_s: float = 3.0,
) -> Optional[str]:
    """Send scene to Unity, request PLY export, wait for file.

    Mirrors the ILManager pipeline:
      WAIT_SCENE_READY → EXPORT_POINTCLOUD → WAIT_POINTCLOUD_READY

    Requires Unity + Flightmare running, ZMQ ports matching config.
    Does NOT need a ROS master — uses standalone ZMQ sockets.

    Returns path to the PLY file, or None on failure.
    """
    import zmq

    g = cfg.get("global", {})
    pub_port = str(g.get("pub_port", "10253"))
    sub_port = str(g.get("sub_port", "10254"))
    scene_id = int(g.get("scene_id", 1))
    pc_cfg = g.get("pointcloud", {})
    pc_range = pc_cfg.get("range", [30, 50, 8])
    pc_origin = pc_cfg.get("origin", [0, 20, 3.5])
    pc_res = float(pc_cfg.get("resolution", 0.1))
    fsm_cfg = g.get("fsm", {})
    keep_alive_period = float(fsm_cfg.get("keep_alive_period", 4.0))

    # Compute actual coverage
    rx, ry, rz = float(pc_range[0]), float(pc_range[1]), float(pc_range[2])
    ox, oy, oz = float(pc_origin[0]), float(pc_origin[1]), float(pc_origin[2])
    pc_xmin, pc_xmax = ox - rx / 2, ox + rx / 2
    pc_ymin, pc_ymax = oy - ry / 2, oy + ry / 2
    pc_zmin, pc_zmax = oz - rz / 2, oz + rz / 2

    print(f"[Unity] PC coverage:")
    print(f"         X: [{pc_xmin:.1f}, {pc_xmax:.1f}]  ({rx:.0f}m)")
    print(f"         Y: [{pc_ymin:.1f}, {pc_ymax:.1f}]  ({ry:.0f}m)")
    print(f"         Z: [{pc_zmin:.1f}, {pc_zmax:.1f}]  ({rz:.0f}m)")
    print(f"         resolution: {pc_res:.2f}m")
    print(f"[Unity] PC range={pc_range} origin={pc_origin} res={pc_res}")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.setsockopt(zmq.RCVHWM, 10)
    sub.setsockopt(zmq.RCVTIMEO, 100)

    try:
        pub.bind(f"tcp://*:{pub_port}")
        sub.bind(f"tcp://*:{sub_port}")
    except zmq.ZMQError as e:
        print(f"[Unity] ERROR binding ZMQ: {e}")
        print("[Unity] Is Unity running? Ports in use?")
        ctx.term()
        return None

    print(f"[Unity] Bound PUB:{pub_port} SUB:{sub_port}")

    ply_path = None
    try:
        # ── Step 1: Send scene (mirrors _st_wait_scene_ready) ──────
        obj_list = gen.generate_unity_objects(obstacles)
        vehicle = make_dummy_vehicle()
        # Unity coordinates: x-right, y-up, z-forward. ROS [0,0,5] → Unity [0,5,0].
        vehicle["position"] = [0.0, 5.0, 0.0]
        vehicle["size"] = [0.5, 0.5, 0.5]

        scene_msg = {
            "scene_id": scene_id,
            "frame_id": 0,
            "vehicles": [vehicle],
            "objects": obj_list,
        }

        print(f"[Unity] Sending scene (scene_id={scene_id}, settle={settle_s:.0f}s) ...")
        settle_deadline = time.time() + settle_s
        while time.time() < settle_deadline:
            pub.send_multipart([b"Pose", json.dumps(scene_msg).encode("utf-8")])
            # Drain stale Unity messages (prevents RCVHWM overflow)
            while True:
                try:
                    _ = sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            time.sleep(0.1)

        # ── Step 2: Export point cloud (mirrors _st_export_pointcloud) ──
        # Let Unity settle before PC request
        time.sleep(1.0)

        # Drain queue one last time
        while True:
            try:
                _ = sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        os.makedirs(output_dir, exist_ok=True)
        expected_ply = os.path.join(output_dir, scene_label + ".ply")

        # Remove existing PLY if present (mirrors ILManager)
        if os.path.exists(expected_ply):
            os.remove(expected_ply)

        pc_req = {
            "range": list(pc_range),
            "origin": list(pc_origin),
            "resolution": pc_res,
            "path": output_dir + "/",
            "file_name": scene_label,
        }
        print(f"[Unity] PC request → {expected_ply}")
        pub.send_multipart([b"PointCloud", json.dumps(pc_req).encode("utf-8")])

        # ── Step 3: Wait for PLY (mirrors _st_wait_pointcloud_ready) ──
        pc_acked = False
        save_success = False
        pc_request_time = time.time()
        pc_next_ka = pc_request_time + keep_alive_period
        last_progress = time.time()

        print(f"[Unity] Waiting for PLY (timeout={timeout_s:.0f}s, "
              f"expected={os.path.basename(expected_ply)}) ...")
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # --- Drain Unity messages ---
            while True:
                try:
                    parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                    for p in parts:
                        try:
                            obj = json.loads(p.decode("utf-8"))
                            if obj.get("get_pc_msg"):
                                if not pc_acked:
                                    elapsed = time.time() - pc_request_time
                                    print(f"[Unity] PC ack received (+{elapsed:.1f}s)")
                                pc_acked = True
                            if obj.get("save_pc_success"):
                                save_success = True
                                elapsed = time.time() - pc_request_time
                                print(f"[Unity] save_pc_success (+{elapsed:.1f}s)")
                        except (ValueError, UnicodeDecodeError):
                            pass
                except zmq.Again:
                    break

            # --- Progress indicator every 5s ---
            now = time.time()
            if now - last_progress > 5.0:
                elapsed = now - pc_request_time
                file_exists = os.path.exists(expected_ply)
                file_size = os.path.getsize(expected_ply) if file_exists else 0
                print(f"[Unity] ... waiting ({elapsed:.0f}s elapsed, "
                      f"file_exists={file_exists}, size={file_size})")
                last_progress = now

            # --- Keep-alive (WITHOUT objects to avoid resetting Unity scene) ---
            if now >= pc_next_ka:
                ka_vehicle = make_dummy_vehicle()
                ka_vehicle["position"] = [0.0, 5.0, 0.0]
                ka_vehicle["size"] = [0.5, 0.5, 0.5]
                ka_msg = {
                    "scene_id": scene_id,
                    "frame_id": -999,
                    "vehicles": [ka_vehicle],
                    "objects": [],   # empty — don't re-send objects during PC wait
                }
                pub.send_multipart([b"Pose", json.dumps(ka_msg).encode("utf-8")])
                pc_next_ka = now + keep_alive_period
            # --- Check for PLY file ---
            if save_success and os.path.exists(expected_ply):
                if _wait_for_stable_file(expected_ply, stable_sec=1.0, max_wait=10.0):
                    ply_path = expected_ply
                    break

            if os.path.exists(expected_ply):
                if _wait_for_stable_file(expected_ply, stable_sec=0.5, max_wait=5.0):
                    ply_path = expected_ply
                    break

            time.sleep(0.05)

        if ply_path is not None:
            size = os.path.getsize(ply_path)
            print(f"[Unity] PLY ready: {ply_path} ({size:,} bytes)")
        else:
            print("[Unity] ERROR: PLY export timeout.")

    finally:
        pub.close(0)
        sub.close(0)
        ctx.term()
        print("[Unity] ZMQ closed.")

    return ply_path


def _wait_for_stable_file(filepath: str, stable_sec: float = 1.0, max_wait: float = 10.0) -> bool:
    """Wait until file size stops changing (like il_common.wait_for_stable_file)."""
    if not os.path.exists(filepath):
        return False
    deadline = time.time() + max_wait
    last_size = -1
    last_change = time.time()
    while time.time() < deadline:
        try:
            cur_size = os.path.getsize(filepath)
        except OSError:
            time.sleep(0.3)
            continue
        if cur_size != last_size:
            last_size = cur_size
            last_change = time.time()
        elif time.time() - last_change >= stable_sec and cur_size > 0:
            return True
        time.sleep(0.2)
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


# ============================================================================
# Python-only point cloud
# ============================================================================

def _ring(cx, cy, r, n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def cylinders_to_pointcloud(obstacles, region_3d, pc_res=0.05, seed=0):
    """Rasterize cylinders into (N,3) point cloud (no Unity needed)."""
    rng = np.random.RandomState(seed)
    xmn, xmx, ymn, ymx, zmn, zmx = region_3d
    parts = []
    for obs in obstacles:
        cx, cy, cz = float(obs.center_world[0]), float(obs.center_world[1]), float(obs.center_world[2])
        r, h = obs.radius_m, obs.height_m
        zb = max(cz - h / 2, zmn)
        zt = min(cz + h / 2, zmx)
        nz = max(2, int(math.ceil((zt - zb) / pc_res)))
        na = max(8, int(math.ceil(2 * math.pi * r / pc_res)))
        for z in np.linspace(zb, zt, nz):
            ring = _ring(cx, cy, r, na)
            parts.append(np.column_stack([ring, np.full(len(ring), z, dtype=np.float32)]))
            n_fill = max(0, int(r / pc_res) - 1)
            for fi in range(1, n_fill + 1):
                ir = r * (1 - fi / (n_fill + 1))
                ni = max(6, int(na * ir / r))
                inner = _ring(cx, cy, ir, ni)
                parts.append(np.column_stack([inner, np.full(len(inner), z, dtype=np.float32)]))
    if not parts:
        return np.zeros((1, 3), dtype=np.float32)
    pts = np.vstack(parts).astype(np.float32)
    pts += rng.normal(0, pc_res * 0.1, pts.shape).astype(np.float32)
    m = ((pts[:, 0] >= xmn) & (pts[:, 0] <= xmx) &
         (pts[:, 1] >= ymn) & (pts[:, 1] <= ymx) &
         (pts[:, 2] >= zmn) & (pts[:, 2] <= zmx))
    pts = pts[m]
    print(f"[INFO] Synthetic point cloud: {len(pts):,} points")
    return pts


# ============================================================================
# Grid & ESDF
# ============================================================================

def pointcloud_to_grid(points, region_xy, grid_res, dilate_radius_m=0.0):
    """2D binary occupancy from XY-projected point cloud.

    Parameters
    ----------
    dilate_radius_m : float
        If > 0, apply morphological dilation to fill surface-only point clouds
        (e.g. Unity PLY export which only captures cylinder surfaces, not interiors).
        Use ~vehicle_radius to fill hollow cylinders properly.
    """
    xmn, xmx, ymn, ymx = region_xy
    gx = int(math.ceil((xmx - xmn) / grid_res))
    gy = int(math.ceil((ymx - ymn) / grid_res))
    ox, oy = xmn, ymn
    ix = np.floor((points[:, 0] - ox) / grid_res).astype(np.int32)
    iy = np.floor((points[:, 1] - oy) / grid_res).astype(np.int32)
    valid = (ix >= 0) & (ix < gx) & (iy >= 0) & (iy < gy)
    occ = np.zeros((gx, gy), dtype=np.uint8)
    if valid.any():
        np.add.at(occ, (ix[valid], iy[valid]), 1)
    occ = (occ > 0).astype(np.uint8)

    # Morphological closing to fill hollow surface-only point clouds
    # (Unity PLY only captures cylinder surfaces, not interiors)
    if dilate_radius_m > 0 and _HAS_SCIPY:
        from scipy.ndimage import binary_closing, generate_binary_structure
        radius_cells = max(1, int(math.ceil(dilate_radius_m / grid_res)))
        # Use a disk-like structuring element (cross with radius)
        se = generate_binary_structure(2, 1)
        for _ in range(radius_cells):
            se = binary_closing(se, structure=generate_binary_structure(2, 1))
        occ = binary_closing(occ, structure=se, iterations=1).astype(np.uint8)
        print(f"[Grid] Morphological closing applied (r={dilate_radius_m:.2f}m = {radius_cells} cells)")

    return occ, (ox, oy), (gx, gy)


def build_esdf(occ, grid_res, drone_r=0.0):
    """2D ESDF via scipy distance_transform_edt."""
    if not _HAS_SCIPY:
        return None
    fd = distance_transform_edt(1 - occ, sampling=grid_res)
    od = distance_transform_edt(occ, sampling=grid_res)
    return (fd - od - drone_r).astype(np.float32)


# ============================================================================
# Plotting
# ============================================================================

def _draw_cyl(ax, obstacles, edge_only, alpha, color):
    for obs in obstacles:
        cx, cy = float(obs.center_world[0]), float(obs.center_world[1])
        r = obs.radius_m
        kw = dict(edgecolor=color, linewidth=0.8, alpha=alpha)
        if edge_only:
            kw["fill"] = False
        else:
            kw["facecolor"] = color
        ax.add_patch(Circle((cx, cy), r, **kw))


def _setup_ax(ax, xmn, xmx, ymn, ymx, xl, yl):
    ax.set_xlim(xmn, xmx)
    ax.set_ylim(ymn, ymx)
    ax.set_xlabel(xl, fontsize=11, fontweight="bold")
    ax.set_ylabel(yl, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, which="major", lw=0.4, alpha=0.3, color="#555555")
    ax.grid(True, which="minor", lw=0.3, alpha=0.2, color="#888888")
    ax.tick_params(which="both", direction="in", labelsize=8)


def plot_grid_map(esdf, occ, origin_xy, grid_res, obstacles,
                  region_bounds, vehicle_r, infl_r, title, figsize=(18, 8),
                  raw_points=None):
    """Multi-panel: Occupancy | ESDF | Obstacles+Contours | Raw Point Cloud (optional)."""
    xmn, xmx, ymn, ymx = region_bounds
    ox, oy = origin_xy
    gx, gy = occ.shape
    extent = [ox, ox + gx * grid_res, oy, oy + gy * grid_res]
    has_raw = raw_points is not None and len(raw_points) > 0
    n = 3 if esdf is not None else 2
    if has_raw:
        n += 1

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n <= 2:
        axes = [axes] if n == 1 else list(axes)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    p = 0  # panel index

    # Panel 1: Occupancy
    ax = axes[p]; p += 1
    ax.imshow(occ.T, origin="lower", extent=extent, cmap="Greys",
              vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    _draw_cyl(ax, obstacles, edge_only=True, alpha=0.5, color="#e74c3c")
    ax.add_patch(Rectangle((xmn, ymn), xmx - xmn, ymx - ymn,
                           fill=False, edgecolor="#00ff00", lw=1.5, ls="--"))
    ax.set_title("Occupancy Grid\n(black=occupied, white=free)", fontsize=11)
    _setup_ax(ax, xmn, xmx, ymn, ymx, "X (m)", "Y (m)")

    if esdf is not None:
        # Panel 2: ESDF
        ax = axes[p]; p += 1
        v = max(1.0, np.percentile(np.abs(esdf), 99))
        im = ax.imshow(esdf.T, origin="lower", extent=extent, cmap="RdYlBu_r",
                       vmin=-v, vmax=v, interpolation="bilinear", aspect="equal")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Signed Distance (m)", fontsize=9)
        ax.contour(esdf.T, levels=[0], extent=extent, colors="black", lw=0.8)
        _draw_cyl(ax, obstacles, edge_only=True, alpha=0.4, color="#e74c3c")
        ax.add_patch(Rectangle((xmn, ymn), xmx - xmn, ymx - ymn,
                               fill=False, edgecolor="#00ff00", lw=1.5, ls="--"))
        ax.set_title("ESDF\n(red=occupied, blue=free)", fontsize=11)
        _setup_ax(ax, xmn, xmx, ymn, ymx, "X (m)", "Y (m)")

    # Raw point cloud panel (before contours panel if esdf, else here)
    if has_raw:
        ax = axes[p]; p += 1
        # Subsample for performance (max 100K points)
        pts = raw_points
        if len(pts) > 100000:
            idx = np.random.RandomState(42).choice(len(pts), 100000, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c="black", alpha=0.6, rasterized=True)
        _draw_cyl(ax, obstacles, edge_only=True, alpha=0.3, color="#e74c3c")
        ax.add_patch(Rectangle((xmn, ymn), xmx - xmn, ymx - ymn,
                               fill=False, edgecolor="#00ff00", lw=1.5, ls="--"))
        ax.set_title(f"Raw Point Cloud (XY)\n({len(pts):,} shown, black=Unity export, red=cylinders)",
                     fontsize=11)
        _setup_ax(ax, xmn, xmx, ymn, ymx, "X (m)", "Y (m)")

    # Panel: Obstacles + contours (always last)
    ax = axes[p]; p += 1
    _draw_cyl(ax, obstacles, edge_only=False, alpha=0.5, color="#e74c3c")
    if esdf is not None:
        for lvl, clr, lw, ls in [
            (0.0, "#e74c3c", 1.5, "-"),
            (vehicle_r, "#2ecc71", 1.0, "--"),
            (infl_r, "#f39c12", 1.0, ":"),
        ]:
            ax.contour(esdf.T, levels=[lvl], extent=extent, colors=clr,
                       linewidths=lw, linestyles=ls, alpha=0.8)

        ax.legend(handles=[
            Line2D([0], [0], color="#e74c3c", lw=1.5, label="surface (0 m)"),
            Line2D([0], [0], color="#2ecc71", lw=1.0, ls="--",
                   label=f"vehicle clearance ({vehicle_r:.2f} m)"),
            Line2D([0], [0], color="#f39c12", lw=1.0, ls=":",
                   label=f"inflation ({infl_r:.2f} m)"),
        ], loc="lower right", fontsize=8, framealpha=0.85)

    ax.add_patch(Rectangle((xmn, ymn), xmx - xmn, ymx - ymn,
                           fill=False, edgecolor="#00ff00", lw=1.5, ls="--"))
    ax.set_title("Obstacles + Clearance Contours", fontsize=11)
    _setup_ax(ax, xmn, xmx, ymn, ymx, "X (m)", "Y (m)")

    plt.tight_layout()
    return fig, axes


# ============================================================================
# Stats
# ============================================================================

def print_stats(occ, esdf, grid_res, obstacles, ply_path=None):
    total = occ.size
    n_occ = int(occ.sum())
    print(f"\n{'='*60}")
    print(f"  Grid: {occ.shape[0]}×{occ.shape[1]} @ {grid_res:.2f}m  "
          f"({occ.shape[0]*grid_res:.1f}×{occ.shape[1]*grid_res:.1f}m)")
    print(f"  Occupied: {n_occ:,} / {total:,} ({n_occ/total*100:.2f}%)")
    if esdf is not None:
        pos, neg = esdf[esdf > 0], esdf[esdf < 0]
        if len(pos):
            print(f"  ESDF free: μ={np.mean(pos):.3f}  max={np.max(pos):.3f} m")
        if len(neg):
            print(f"  ESDF penetration: μ={np.mean(neg):.3f}  min={np.min(neg):.3f} m")
    radii = [o.radius_m for o in obstacles]
    print(f"  Obstacles: {len(obstacles)}, r∈[{min(radii):.3f},{max(radii):.3f}] m, "
          f"μ={np.mean(radii):.3f} m")
    if ply_path:
        print(f"  PLY source: {ply_path}")
    print(f"{'='*60}\n")


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Scene Grid Map Visualizer")
    p.add_argument("--profile", default="S01")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--grid-res", type=float, default=0.1)
    p.add_argument("--pc-res", type=float, default=0.05,
                   help="Point cloud sampling in m (python-only mode)")
    p.add_argument("--unity", action="store_true",
                   help="Use Unity to render real point cloud via ZMQ")
    p.add_argument("--ply-dir", default=None,
                   help="Output dir for Unity PLY file (default: /tmp/unity_grid_viz/)")
    p.add_argument("--timeout", type=float, default=120.0,
                   help="Unity PC export timeout in seconds (default: 120)")
    p.add_argument("--settle", type=float, default=3.0,
                   help="Scene settle time before PC request (default: 3s)")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--figsize", type=float, nargs=2, default=(18, 8))
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    if not _HAS_PROJECT:
        print("ERROR: Cannot import il_scenario / il_common.")
        sys.exit(1)

    # --- 1. Generate obstacles ---
    obstacles, profile, region, gen, cfg = generate_scene(
        args.seed, args.profile, args.scene_index)
    region_xy = (region.x_min, region.x_max, region.y_min, region.y_max)
    region_3d = (region.x_min, region.x_max, region.y_min, region.y_max,
                 region.z_min, region.z_max)

    # --- 2. Get point cloud ---
    ply_path = None
    if args.unity:
        # Print PC vs scene coverage comparison
        pc_cfg = cfg.get("global", {}).get("pointcloud", {})
        pc_r = pc_cfg.get("range", [30, 50, 8])
        pc_o = pc_cfg.get("origin", [0, 20, 3.5])
        print(f"\n[Coverage] Scene region:  "
              f"X=[{region.x_min:.1f},{region.x_max:.1f}]  "
              f"Y=[{region.y_min:.1f},{region.y_max:.1f}]  "
              f"Z=[{region.z_min:.1f},{region.z_max:.1f}]")
        print(f"[Coverage] PC   region:  "
              f"X=[{pc_o[0]-pc_r[0]/2:.1f},{pc_o[0]+pc_r[0]/2:.1f}]  "
              f"Y=[{pc_o[1]-pc_r[1]/2:.1f},{pc_o[1]+pc_r[1]/2:.1f}]  "
              f"Z=[{pc_o[2]-pc_r[2]/2:.1f},{pc_o[2]+pc_r[2]/2:.1f}]")
        x_ok = (pc_o[0]-pc_r[0]/2 <= region.x_min and pc_o[0]+pc_r[0]/2 >= region.x_max)
        y_ok = (pc_o[1]-pc_r[1]/2 <= region.y_min and pc_o[1]+pc_r[1]/2 >= region.y_max)
        z_ok = (pc_o[2]-pc_r[2]/2 <= region.z_min and pc_o[2]+pc_r[2]/2 >= region.z_max)
        covers = "✓ FULLY COVERED" if (x_ok and y_ok and z_ok) else "⚠ PARTIAL — check Z/height!"
        print(f"[Coverage] Scene ⊂ PC ?  {covers}")
        print()

        ply_dir = args.ply_dir or "/tmp/unity_grid_viz"
        ply_path = export_pointcloud_unity(
            obstacles, gen, cfg,
            output_dir=ply_dir,
            scene_label=f"{profile.name}_scene{args.scene_index}",
            timeout_s=args.timeout,
            settle_s=args.settle,
        )
        if ply_path is None:
            print("ERROR: Unity point cloud export failed.")
            print("Make sure Unity + Flightmare is running and ZMQ ports match.")
            sys.exit(1)
        points = load_ply(ply_path)
        print(f"[INFO] Loaded PLY: {len(points):,} points")
    else:
        points = cylinders_to_pointcloud(obstacles, region_3d,
                                         pc_res=args.pc_res,
                                         seed=args.seed + 1000)

    # --- 3. Occupancy grid ---
    # --- 3. Occupancy grid ---
    # For Unity mode: use the full PC coverage as grid bounds
    # For Python mode: use the scene region (point cloud only exists there)
    if ply_path is not None:
        # Use PC coverage from config
        pc_cfg = cfg.get("global", {}).get("pointcloud", {})
        pc_r = pc_cfg.get("range", [30, 50, 8])
        pc_o = pc_cfg.get("origin", [0, 20, 3.5])
        grid_xmin = pc_o[0] - pc_r[0] / 2
        grid_xmax = pc_o[0] + pc_r[0] / 2
        grid_ymin = pc_o[1] - pc_r[1] / 2
        grid_ymax = pc_o[1] + pc_r[1] / 2
    else:
        # Use scene region for synthetic point cloud
        grid_xmin, grid_xmax = region_xy[0], region_xy[1]
        grid_ymin, grid_ymax = region_xy[2], region_xy[3]

    grid_region_xy = (grid_xmin, grid_xmax, grid_ymin, grid_ymax)
    print(f"[Grid] Region: X=[{grid_xmin:.1f},{grid_xmax:.1f}]  "
          f"Y=[{grid_ymin:.1f},{grid_ymax:.1f}]  res={args.grid_res:.2f}m")

    # For Unity mode: apply dilation to fill hollow cylinder surfaces
    dilate_r = profile.inflation_radius_m if ply_path is not None else 0.0
    occ, origin_xy, grid_shape = pointcloud_to_grid(
        points, grid_region_xy, args.grid_res, dilate_radius_m=dilate_r)

    # --- 4. ESDF ---
    esdf = build_esdf(occ, args.grid_res, drone_r=profile.vehicle_radius_m)

    # --- 5. Stats ---
    print_stats(occ, esdf, args.grid_res, obstacles, ply_path)

    # --- 6. Plot ---
    src_tag = "Unity PLY" if ply_path else "Python synthetic"
    title = (f"Grid Map — {profile.name}  |  {len(obstacles)} obstacles  |  "
             f"{grid_shape[0]}×{grid_shape[1]} @ {args.grid_res:.2f}m  |  {src_tag}")

    # Adjust figsize for extra raw-points panel
    out_figsize = list(args.figsize)
    if ply_path is not None:
        out_figsize[0] += 6  # extra width for raw point cloud panel

    fig, _ = plot_grid_map(esdf, occ, origin_xy, args.grid_res, obstacles,
                           grid_region_xy, profile.vehicle_radius_m,
                           profile.inflation_radius_m, title, tuple(out_figsize),
                           raw_points=points if ply_path is not None else None)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
