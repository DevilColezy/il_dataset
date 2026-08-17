#!/usr/bin/env python3
"""Inspect saved depth PNGs to diagnose "all black" depth.

Usage:
    python depth_stats.py <episode_dir> [--frames 12] [--step 1]

Reads a few depth PNGs (16-bit uint16, meters_per_unit=0.01 per the data
contract) and prints pixel statistics, so we can tell apart:
  * all-zero        -> Unity returned no valid depth at all
  * values 10..500  -> 0.1..5.0 m (normal; a generic 8-bit viewer shows
                       these as nearly black)
  * values 1..50    -> 0.01..0.5 m (depth01 mis-decoded as "hectometres";
                       real distances are ~25x larger)
"""
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
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return img
    from PIL import Image
    return np.asarray(Image.open(str(path)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    episode = sys.argv[1]
    frames = 12
    step = 1
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
        print("no PNGs in %s" % depth_dir)
        return 1
    print("total PNGs: %d" % len(pngs))
    chosen = pngs[::step][:frames]
    print("checking %d frames: %s ... %s" % (len(chosen), chosen[0],
                                             chosen[-1]))
    for name in chosen:
        img = read_png(os.path.join(depth_dir, name))
        if img is None or img.ndim != 2:
            print("  %s: bad image" % name)
            continue
        arr = img.astype(np.float64)
        total = arr.size
        n_zero = int(np.count_nonzero(arr == 0))
        nz = arr[arr > 0]
        if nz.size == 0:
            print("  %s: ALL ZERO (total=%d)" % (name, total))
            continue
        p = np.percentile(nz, [1, 10, 50, 90, 99])
        print(("  %s: zero=%.1f%%  nonzero: min=%.1f p01=%.1f p10=%.1f "
               "p50=%.1f p90=%.1f p99=%.1f max=%.1f  (dtype=%s)")
              % (name, 100.0 * n_zero / total,
                 nz.min(), p[0], p[1], p[2], p[3], p[4], nz.max(),
                 img.dtype))
    print("\nInterpretation (normalized uint16: depth_m = pixel/65535 * 5m):")
    print("  all-zero          -> Unity produced no valid depth")
    print("  median < 2000     -> mostly far (>5 m) / invalid")
    print("  near pixels 13000..65535 -> 1..5 m (bright, clearly visible)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
