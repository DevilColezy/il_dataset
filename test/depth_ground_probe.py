#!/usr/bin/env python3
"""Probe whether real depth frames contain a dominant GROUND band that the
2D horizontal expert would treat as near obstacles.

Mirrors Flightmare2DObservation::build() projection for the default
identity T_BC (R_bc_fl = [[1,0,0],[0,0,1],[0,-1,0]]), so for a level body:
    rel_z (point relative to camera, up) = -y_opt = -(v-cy)*d/fy
    horiz = hypot(rel_x, rel_y)

Pixels whose 3D point lies clearly BELOW the flight plane (rel_z < -0.5 m)
are ground / floor; the horizontal expert currently maps them to near
obstacles.  This script quantifies how many pixels fall in that band and
at what horizontal distance they sit.

Usage:
    python depth_ground_probe.py <episode_dir> [--frames 8] [--step 200]
"""
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
    from PIL import Image


def read_png(path):
    if cv2 is not None:
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    from PIL import Image
    return np.asarray(Image.open(str(path)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    episode = sys.argv[1]
    frames = 8
    step = 200
    for i, a in enumerate(sys.argv):
        if a == "--frames" and i + 1 < len(sys.argv):
            frames = int(sys.argv[i + 1])
        if a == "--step" and i + 1 < len(sys.argv):
            step = int(sys.argv[i + 1])
    depth_dir = os.path.join(episode, "depth")
    if not os.path.isdir(depth_dir):
        print("no depth dir at %s" % depth_dir)
        return 1
    pngs = sorted(f for f in os.listdir(depth_dir)
                  if f.lower().endswith(".png"))
    if not pngs:
        print("no PNGs")
        return 1
    chosen = pngs[::step][:frames]
    print("probe %d frames (every %d-th): %s ... %s" %
          (len(chosen), step, chosen[0], chosen[-1]))

    # Camera parameters (from il_dataset config): fov 90 deg, near 0.01,
    # far 1000; image 640x480.  Focal from FOV/width like the C++ build.
    fov_deg = 90.0
    width, height = 640, 480
    fov = math.radians(fov_deg)
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    fx = (0.5 * width) / max(1e-9, math.tan(0.5 * fov))
    fy = fx
    # Default identity T_BC -> R_bc_fl (flightlib body frame):
    #   [[1,0,0],[0,0,1],[0,-1,0]]  => rel_z = -y_opt, horiz uses (x_opt, d).
    ground_threshold_m = -0.5   # point clearly below the flight plane

    # Rows of the image (v) and columns (u).
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    for name in chosen:
        img = read_png(os.path.join(depth_dir, name))
        if img is None or img.ndim != 2:
            continue
        # Normalized 16-bit encoding: depth_m = pixel/65535 * max_m.
        depth_m = img.astype(np.float64) / 65535.0 * 5.0
        d = depth_m
        x_opt = (us[None, :] - cx) * d / fx        # [H,W]
        y_opt = (vs[:, None] - cy) * d / fy
        rel_z = -y_opt                              # level body
        horiz = np.hypot(x_opt, d)                  # horizontal distance
        valid = np.isfinite(d) & (d > 0.01)
        ground = valid & (rel_z < ground_threshold_m)
        n_valid = int(np.count_nonzero(valid))
        n_ground = int(np.count_nonzero(ground))
        if n_valid == 0:
            print("  %s: no valid depth" % name)
            continue
        gh = horiz[ground]
        frac = 100.0 * n_ground / n_valid
        # Of the ground pixels, how many map to < R=5 m (near obstacles)?
        near_frac = (100.0 * np.count_nonzero(gh < 5.0) / n_ground
                     if n_ground else 0.0)
        med = np.median(gh) if n_ground else float("nan")
        print(("  %s: valid=%d ground(rel_z<%.1f)=%d (%.1f%%)  "
               "of ground: <5m=%.1f%% median_horiz=%.1fm")
              % (name, n_valid, ground_threshold_m, n_ground, frac,
                 near_frac, med))
    print("\nInterpretation:")
    print("  ground% high + ground(<5m)% high -> the horizontal expert sees")
    print("  a near floor band as obstacles -> persistent TURN / spin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
