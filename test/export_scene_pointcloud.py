#!/usr/bin/env python3
"""Export a Flightmare/AvoidBench Unity scene as a point cloud (PLY) and
render a top-down (2D) occupancy view.

Usage:
    python3 export_scene_pointcloud.py --scene-id 0 \
        --range 40 40 40 --resolution 0.2 \
        --out /home/rgzn/flightmare_ws/pic/scene0_topdown.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_IL_SCRIPTS = _THIS_DIR.parent / "scripts"
if str(_IL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_IL_SCRIPTS))

import il_common  # noqa: E402

# Unity process runs from this directory, so the PLY is written relative to it.
UNITY_CWD = Path("/home/rgzn/flightmare_ws/RPG_Flightmare/AvoidBench_x")
PC_OUT_DIR = UNITY_CWD / "point_clouds_data"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene-id", type=int, default=0,
                   help="Unity scene: 0=INDUSTRIAL(outdoor) 1=WAREHOUSE 2=GARAGE "
                        "3=NATUREFOREST 4=TUNELS")
    p.add_argument("--range", nargs=3, type=float, default=[40.0, 40.0, 40.0],
                   help="point cloud box range [x y z] metres")
    p.add_argument("--origin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    p.add_argument("--resolution", type=float, default=0.2)
    p.add_argument("--file-name", default="scene_pc")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--ply", default="",
                  help="Skip Unity and parse an existing PLY file directly.")
    p.add_argument("--out", default="/tmp/scene_topdown.png",
                   help="output top-down PNG")
    p.add_argument("--keep-ply", action="store_true")
    a = p.parse_args()

    if a.ply:
        ply_path = Path(a.ply)
    else:
        depth_cfg = {"width": 640, "height": 480, "fov": 58.0, "near": 0.28,
                     "far": 10.0,
                     "t_bc": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
        print(f"[pc-export] Connecting to Unity scene {a.scene_id} ...")
        bridge = il_common.UnityBridge(pub_port="10253", sub_port="10254")
        bridge.bind()
        ok = bridge.connect_handshake(a.scene_id, depth_cfg, timeout=60.0)
        if not ok:
            print("ERROR: Unity handshake failed")
            sys.exit(1)
        print("[pc-export] Unity handshake OK. Sending PointCloud request ...")

        PC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        ply_path = PC_OUT_DIR / (a.file_name + ".ply")
        if ply_path.exists():
            ply_path.unlink()

        req = {
            "range": list(a.range),
            "origin": list(a.origin),
            "resolution": a.resolution,
            "path": "./point_clouds_data/",
            "file_name": a.file_name,
        }
        bridge.send_pc_request(req)
        print(f"[pc-export] waiting for {ply_path} ...")

        deadline = time.time() + a.timeout
        while time.time() < deadline:
            if ply_path.exists() and ply_path.stat().st_size > 0:
                break
            time.sleep(1.0)
        if not ply_path.exists() or ply_path.stat().st_size == 0:
            print("ERROR: PLY not generated within %.0f s "
                  "(Unity may not support PointCloud export)." % a.timeout)
            sys.exit(1)
        print(f"[pc-export] PLY saved: {ply_path} "
              f"({ply_path.stat().st_size/1e6:.1f} MB)")

    # ── Parse PLY (ascii or binary_little_endian vertex cloud) ─────
    with ply_path.open("rb") as fh:
        header = b""
        while True:
            line = fh.readline()
            header += line
            if line.strip() == b"end_header":
                break
        header_text = header.decode("ascii", errors="replace")
        fmt = "ascii"
        n_verts = None
        props = []
        for line in header_text.splitlines():
            line = line.strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif line.startswith("property "):
                props.append(line.split()[1])
        if n_verts is None:
            raise RuntimeError("PLY has no vertex element")
        if fmt == "ascii":
            raw = fh.read().decode("ascii", errors="replace")
            vals = np.fromstring(raw, sep=" ")
            pts = vals.reshape(-1, len(props))[:, :3]
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            dtype = "<f4" if fmt == "binary_little_endian" else ">f4"
            n_float = len(props)
            count = n_verts * n_float
            data = np.frombuffer(fh.read(count * 4), dtype=dtype)
            if data.size < count:
                raise RuntimeError("PLY binary data truncated")
            pts = data.reshape(n_verts, n_float)[:, :3].astype(np.float64)
        else:
            raise RuntimeError("unsupported PLY format %r" % fmt)
    print(f"[pc-export] parsed {len(pts)} points, "
          f"bounds x[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}] "
          f"z[{pts[:,2].min():.1f},{pts[:,2].max():.1f}]")

    # ── Top-down occupancy (z-projection) ──────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = pts[:, 0]
    y = pts[:, 1]
    res = a.resolution
    x0, x1 = float(np.floor(x.min() / res) * res), float(np.ceil(x.max() / res) * res)
    y0, y1 = float(np.floor(y.min() / res) * res), float(np.ceil(y.max() / res) * res)
    nx = int(round((x1 - x0) / res)) + 1
    ny = int(round((y1 - y0) / res)) + 1
    occ = np.zeros((ny, nx), dtype=np.int32)
    ix = np.clip(((x - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((y - y0) / res).astype(int), 0, ny - 1)
    np.add.at(occ, (iy, ix), 1)
    occupied = occ > 0

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(occupied, origin="lower", extent=[x0, x1, y0, y1],
              cmap="Greys", interpolation="nearest")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Unity scene {a.scene_id} top-down occupancy "
                 f"(res={res}m, {len(pts)} pts)")
    ax.grid(alpha=0.3)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"[pc-export] top-down saved: {out}")

    # also dump the occupancy as a compact numpy for later use
    npz = out.with_suffix(".npz")
    np.savez_compressed(npz, occupied=occupied, x0=x0, y0=y0, res=res,
                        points=pts)
    print(f"[pc-export] occupancy npz saved: {npz}")

    if not a.keep_ply:
        ply_path.unlink()
    try:
        bridge.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
