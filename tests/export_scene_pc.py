#!/usr/bin/env python3
"""
Export Scene Point Cloud from Unity (standalone, no ROS master needed).

Sends cylinder obstacles to Unity via ZMQ, requests a PLY point cloud export,
and saves the PLY file to the specified output directory.

Usage
-----
    # Export S01 scene from config
    python3 export_scene_pc.py

    # Specify profile, seed, output
    python3 export_scene_pc.py --profile S05 --seed 42 --output ./my_scene.ply

    # Adjust settle time and timeout
    python3 export_scene_pc.py --settle 5 --timeout 300

    # Preview after export
    python3 export_scene_pc.py --profile S05 --preview
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── project imports ────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_SCRIPTS = os.path.join(_SCRIPT_DIR, "..", "scripts")
if _PROJECT_SCRIPTS not in sys.path:
    sys.path.insert(0, _PROJECT_SCRIPTS)

try:
    from il_scenario import YamlCylinderSceneGenerator
    from il_common import make_dummy_vehicle
    _HAS_PROJECT = True
except ImportError as e:
    print(f"ERROR: Cannot import project modules: {e}")
    sys.exit(1)


# ============================================================================
# Config
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


def _load_config(config_path: str):
    """Load YAML config without ROS master."""
    import il_config
    _orig = il_config._apply_ros_overrides
    il_config._apply_ros_overrides = lambda cfg: None
    try:
        return il_config.load_config(config_path, validate=False)
    finally:
        il_config._apply_ros_overrides = _orig


# ============================================================================
# Scene generation
# ============================================================================

def generate_obstacles(seed: int, profile_hint: str, scene_index: int = 0):
    """Return (obstacles, profile, region, generator, cfg)."""
    config_path = _find_config()
    if not config_path:
        raise FileNotFoundError("Cannot find il_dataset_config.yaml")

    cfg = _load_config(config_path)
    gen = YamlCylinderSceneGenerator(cfg)
    profiles = gen.get_profiles()
    profile = next((p for p in profiles if p.name and profile_hint in p.name),
                   profiles[0])

    region = gen.obstacle_region
    print(f"[Scene] Profile: {profile.name}")
    print(f"[Scene] Region: x=[{region.x_min:.1f},{region.x_max:.1f}]  "
          f"y=[{region.y_min:.1f},{region.y_max:.1f}]  "
          f"z=[{region.z_min:.1f},{region.z_max:.1f}]")

    eff_seed = seed + getattr(profile, 'seed_offset', 0)
    obstacles, rejection, _, _ = gen.generate_scene_from_profile(
        profile, eff_seed, scene_index_in_profile=scene_index)

    if rejection:
        print(f"[Scene] WARN: rejection={rejection}")
    print(f"[Scene] {len(obstacles)} obstacles generated")

    radii = [o.radius_m for o in obstacles]
    if radii:
        print(f"[Scene] Radius: [{min(radii):.3f}, {max(radii):.3f}] m  "
              f"mean={sum(radii)/len(radii):.3f} m")

    return obstacles, profile, region, gen, cfg


# ============================================================================
# Unity PC Export
# ============================================================================

def export_ply(
    obstacles: List,
    gen,
    cfg: dict,
    output_path: str,
    settle_s: float = 3.0,
    timeout_s: float = 300.0,
) -> bool:
    """Export PLY from Unity via ZMQ.

    Mirrors the ILManager pipeline:
      Send scene repeatedly for settle_s → request PC → wait for file.

    Returns True on success.
    """
    import zmq

    g = cfg.get("global", {})
    pub_port = str(g.get("pub_port", "10253"))
    sub_port = str(g.get("sub_port", "10254"))
    scene_id = int(g.get("scene_id", 1))
    pc_cfg = g.get("pointcloud", {})
    pc_range = [30, 60, 8]
    pc_origin = [0, 20, 3.5]
    pc_res = float(pc_cfg.get("resolution", 0.1))
    keep_alive_period = float(g.get("fsm", {}).get("keep_alive_period", 4.0))

    # Resolve output path
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    file_base = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    # PC coverage info
    rx, ry, rz = map(float, pc_range)
    ox, oy, oz = map(float, pc_origin)
    print(f"[Unity] PC coverage: X=[{ox-rx/2:.1f},{ox+rx/2:.1f}]  "
          f"Y=[{oy-ry/2:.1f},{oy+ry/2:.1f}]  "
          f"Z=[{oz-rz/2:.1f},{oz+rz/2:.1f}]  res={pc_res}m")
    print(f"[Unity] Output: {output_path}")

    # ── ZMQ setup ──
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
        ctx.term()
        return False

    print(f"[Unity] Bound PUB:{pub_port} SUB:{sub_port}")
    success = False

    try:
        # ── Step 1: Send scene (settle period) ──
        obj_list = gen.generate_unity_objects(obstacles)
        vehicle = make_dummy_vehicle()
        vehicle["position"] = [0.0, 5.0, 0.0]
        vehicle["size"] = [0.5, 0.5, 0.5]

        scene_msg = {
            "scene_id": scene_id,
            "frame_id": 0,
            "vehicles": [vehicle],
            "objects": obj_list,
        }

        print(f"[Unity] Sending scene ({len(obj_list)} objects, settle={settle_s:.0f}s) ...")
        deadline = time.time() + settle_s
        while time.time() < deadline:
            pub.send_multipart([b"Pose", json.dumps(scene_msg).encode("utf-8")])
            # Drain stale messages
            while True:
                try:
                    _ = sub.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
            time.sleep(0.1)

        # ── Step 2: Request PC export ──
        time.sleep(1.0)
        # Final drain
        while True:
            try:
                _ = sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        # Remove old file
        if os.path.exists(output_path):
            os.remove(output_path)

        pc_req = {
            "range": list(pc_range),
            "origin": list(pc_origin),
            "resolution": pc_res,
            "path": out_dir + "/",
            "file_name": file_base,
        }
        print(f"[Unity] PC request sent")
        pub.send_multipart([b"PointCloud", json.dumps(pc_req).encode("utf-8")])

        # ── Step 3: Wait for file ──
        pc_acked = False
        save_success = False
        pc_request_time = time.time()
        pc_next_ka = pc_request_time + keep_alive_period
        last_progress = time.time()

        print(f"[Unity] Waiting for PLY (timeout={timeout_s:.0f}s) ...")
        deadline_t = time.time() + timeout_s

        while time.time() < deadline_t:
            # Drain + parse Unity messages
            while True:
                try:
                    parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                    for p in parts:
                        try:
                            obj = json.loads(p.decode("utf-8"))
                            if obj.get("get_pc_msg") and not pc_acked:
                                print(f"[Unity] PC ack (+{time.time()-pc_request_time:.1f}s)")
                                pc_acked = True
                            if obj.get("save_pc_success") and not save_success:
                                print(f"[Unity] save_pc_success (+{time.time()-pc_request_time:.1f}s)")
                                save_success = True
                        except (ValueError, UnicodeDecodeError):
                            pass
                except zmq.Again:
                    break

            # Progress
            now = time.time()
            if now - last_progress > 10.0:
                elapsed = now - pc_request_time
                exists = os.path.exists(output_path)
                size = os.path.getsize(output_path) if exists else 0
                print(f"[Unity] ... {elapsed:.0f}s  file_exists={exists}  size={size:,}")
                last_progress = now

            # Keep-alive (without objects)
            if now >= pc_next_ka:
                ka = make_dummy_vehicle()
                ka["position"] = [0.0, 5.0, 0.0]
                ka["size"] = [0.5, 0.5, 0.5]
                ka_msg = {"scene_id": scene_id, "frame_id": -999,
                          "vehicles": [ka], "objects": []}
                pub.send_multipart([b"Pose", json.dumps(ka_msg).encode("utf-8")])
                pc_next_ka = now + keep_alive_period

            # Check for file
            if os.path.exists(output_path):
                if _wait_stable(output_path):
                    success = True
                    break

            time.sleep(0.05)

        if success:
            size = os.path.getsize(output_path)
            print(f"[Unity] ✓ PLY saved: {output_path} ({size:,} bytes)")
        else:
            print(f"[Unity] ✗ Export failed or timed out.")

    finally:
        pub.close(0)
        sub.close(0)
        ctx.term()

    return success


def _wait_stable(filepath: str, stable_sec: float = 1.0, max_wait: float = 10.0) -> bool:
    """Wait for file size to stabilise."""
    if not os.path.exists(filepath):
        return False
    deadline = time.time() + max_wait
    last_size = -1
    last_change = time.time()
    while time.time() < deadline:
        try:
            cur = os.path.getsize(filepath)
        except OSError:
            time.sleep(0.3)
            continue
        if cur != last_size:
            last_size = cur
            last_change = time.time()
        elif time.time() - last_change >= stable_sec and cur > 0:
            return True
        time.sleep(0.2)
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Export Scene Point Cloud from Unity")
    p.add_argument("--profile", default="S01", help="Profile name/substring")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--output", "-o", default=None,
                   help="Output PLY path (default: ./<profile>_seed<seed>.ply)")
    p.add_argument("--settle", type=float, default=3.0,
                   help="Scene settle time before PC request (default: 3s)")
    p.add_argument("--timeout", type=float, default=300.0,
                   help="PC export timeout in seconds (default: 300)")
    p.add_argument("--preview", action="store_true",
                   help="After export, launch view_pc_grid.py to visualise")
    args = p.parse_args()

    # Generate obstacles
    obstacles, profile, region, gen, cfg = generate_obstacles(
        args.seed, args.profile, args.scene_index)

    # Output path
    if args.output:
        out_path = args.output
    else:
        out_path = f"{profile.name}_seed{args.seed}.ply"

    # Export
    ok = export_ply(obstacles, gen, cfg, out_path,
                    settle_s=args.settle, timeout_s=args.timeout)

    if not ok:
        print("\nExport FAILED. Check that Unity is running and ZMQ ports match.")
        sys.exit(1)

    # Save metadata alongside PLY
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    meta = {
        "profile": profile.name,
        "seed": args.seed,
        "scene_index": args.scene_index,
        "num_obstacles": len(obstacles),
        "obstacle_radii": [o.radius_m for o in obstacles],
        "region": {
            "x_min": region.x_min, "x_max": region.x_max,
            "y_min": region.y_min, "y_max": region.y_max,
            "z_min": region.z_min, "z_max": region.z_max,
        },
        "pc_config": {
            "range": cfg["global"]["pointcloud"]["range"],
            "origin": cfg["global"]["pointcloud"]["origin"],
            "resolution": cfg["global"]["pointcloud"]["resolution"],
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Meta] Saved: {meta_path}")

    # Preview
    if args.preview:
        viewer_script = os.path.join(_SCRIPT_DIR, "view_pc_grid.py")
        if os.path.exists(viewer_script):
            print(f"\n[Preview] Launching viewer...")
            os.system(f"python3 {viewer_script} {out_path}")
        else:
            print(f"\n[Preview] view_pc_grid.py not found in {_SCRIPT_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()
