#!/usr/bin/env python3
"""Slice an exported scene point cloud by an operating-height band and build
the 2D known-obstacle occupancy grid used by obstacle/task generation.

The point cloud is a surface sample of the Unity scene.  For a 2.5D flight
band [z_min, z_max], any structure whose surface has points INSIDE the band
blocks horizontal motion, so we project those points to a 2D occupancy grid.

Outputs (per scene x band):
  - `baseline/<scene>_<band>.npz`: occupied grid + meta (resolution, extent)
  - `baseline/<scene>_<band>.png`: top-down view with known-obstacle clusters
  - `baseline/<scene>_<band>_clusters.json`: cluster list (center/size/height)

Usage:
    python3 slice_pointcloud_layers.py \
        --scene0 pic/scene0_topdown.npz --scene1 pic/scene1_topdown.npz \
        --band-a 0.5 2.0 --band-b 10.0 13.0 \
        --out-dir /home/rgzn/flightmare_ws/il_data_baseline
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

DRONE_RADIUS = 0.30
RES = 0.10  # match expert ESDF resolution


def build_occupancy(pts, zmin, zmax, extent, res=RES):
    """Project band [zmin,zmax] points into a 2D occupancy grid over extent."""
    m = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    band = pts[m]
    x0, x1, y0, y1 = extent
    nx = int(round((x1 - x0) / res)) + 1
    ny = int(round((y1 - y0) / res)) + 1
    occ = np.zeros((ny, nx), dtype=np.uint8)
    if len(band) == 0:
        return occ, band, (x0, x1, y0, y1)
    ix = np.clip(((band[:, 0] - x0) / res).astype(int), 0, nx - 1)
    iy = np.clip(((band[:, 1] - y0) / res).astype(int), 0, ny - 1)
    np.add.at(occ, (iy, ix), 1)
    occ = (occ > 0).astype(np.uint8)
    # morphological close + dilate by drone radius (surface margin)
    dil = int(round(DRONE_RADIUS / res))
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3)))
    occ = ndimage.binary_dilation(occ, iterations=max(1, dil))
    return occ.astype(np.uint8), band, (x0, x1, y0, y1)


def clusters(occ, extent, band, res=RES, min_cells=6):
    """Return known-obstacle clusters (center/size/max-height)."""
    x0, x1, y0, y1 = extent
    lab, nlab = ndimage.label(occ)
    out = []
    for i in range(1, nlab + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) < min_cells:
            continue
        cx = x0 + xs.mean() * res
        cy = y0 + ys.mean() * res
        wx = (xs.max() - xs.min() + 1) * res
        wy = (ys.max() - ys.min() + 1) * res
        mm = (band[:, 0] >= cx - wx) & (band[:, 0] <= cx + wx) & \
             (band[:, 1] >= cy - wy) & (band[:, 1] <= cy + wy)
        zmax = float(band[mm, 2].max()) if mm.any() else 0.0
        out.append({"x": round(cx, 2), "y": round(cy, 2),
                    "w": round(wx, 2), "h": round(wy, 2),
                    "z_max": round(zmax, 2)})
    return out


def plot(occ, extent, clus, band_name, out_path):
    x0, x1, y0, y1 = extent
    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    ax.imshow(occ, origin="lower", extent=[x0, x1, y0, y1],
              cmap="Greys", interpolation="nearest")
    for c in clus:
        ax.add_patch(plt.Rectangle((c["x"] - c["w"] / 2, c["y"] - c["h"] / 2),
                                   c["w"], c["h"], fill=False,
                                   edgecolor="red", lw=1.2))
        ax.text(c["x"], c["y"], "%.0fx%.0f" % (c["w"], c["h"]),
                fontsize=6, color="red", ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.6, pad=1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{band_name} known obstacles ({len(clus)} clusters)")
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene0", default="pic/scene0_topdown.npz")
    p.add_argument("--scene1", default="pic/scene1_topdown.npz")
    p.add_argument("--band-a", nargs=2, type=float, default=[0.5, 2.0],
                   help="class A operating band z [min max]")
    p.add_argument("--band-b", nargs=2, type=float, default=[10.0, 13.0],
                   help="class B operating band z [min max]")
    p.add_argument("--out-dir", default="/home/rgzn/flightmare_ws/il_data_baseline")
    p.add_argument("--resolution", type=float, default=RES)
    a = p.parse_args()

    res = a.resolution
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = {"scene0": Path(a.scene0), "scene1": Path(a.scene1)}
    bands = {"classA": tuple(a.band_a), "classB": tuple(a.band_b)}

    for sname, spath in scenes.items():
        if not spath.exists():
            print(f"skip {sname}: {spath} missing")
            continue
        d = np.load(spath)
        pts = d["points"].astype(np.float64)
        # extent = point cloud XY bounds (rounded out)
        x0 = float(np.floor(pts[:, 0].min() / 1.0) * 1.0)
        x1 = float(np.ceil(pts[:, 0].max() / 1.0) * 1.0)
        y0 = float(np.floor(pts[:, 1].min() / 1.0) * 1.0)
        y1 = float(np.ceil(pts[:, 1].max() / 1.0) * 1.0)
        extent = (x0, x1, y0, y1)
        print(f"{sname}: {len(pts)} pts, extent x[{x0},{x1}] y[{y0},{y1}]")
        for bname, (zmin, zmax) in bands.items():
            occ, band, ext = build_occupancy(pts, zmin, zmax, extent, res)
            clus = clusters(occ, ext, band, res)
            occ_frac = float(occ.mean())
            base = f"{sname}_{bname}_{zmin:.1f}-{zmax:.1f}m"
            np.savez_compressed(out_dir / (base + ".npz"),
                                occupied=occ, extent=ext, res=res,
                                z_band=[zmin, zmax], points=band)
            with (out_dir / (base + "_clusters.json")).open("w") as fh:
                json.dump({"band": [zmin, zmax], "occupancy_fraction":
                           occ_frac, "clusters": clus}, fh, indent=1)
            plot(occ, ext, clus, f"{sname} {bname} z[{zmin},{zmax}]",
                 out_dir / (base + ".png"))
            print(f"  {bname} z[{zmin:.1f},{zmax:.1f}]: "
                  f"{int(occ.sum())} cells occ={occ_frac*100:.1f}% "
                  f"clusters={len(clus)}")


if __name__ == "__main__":
    main()
