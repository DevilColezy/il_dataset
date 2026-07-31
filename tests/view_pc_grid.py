#!/usr/bin/env python3
"""
Interactive Point Cloud Top-Down Viewer

Loads a PLY point cloud and displays an interactive XY (top-down) scatter view
with coordinate input fields to select the region of interest.

Usage
-----
    python3 view_pc_grid.py scene.ply

    # With initial region
    python3 view_pc_grid.py scene.ply --region -15 15 -5 45

    # Auto-fit to occupied area
    python3 view_pc_grid.py scene.ply --auto
"""

import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib.ticker as ticker

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_SCRIPTS = os.path.join(_SCRIPT_DIR, "..", "scripts")
if _PROJECT_SCRIPTS not in sys.path:
    sys.path.insert(0, _PROJECT_SCRIPTS)
try:
    from il_common import load_ply as _load_ply_lib
except ImportError:
    _load_ply_lib = None


# ============================================================================
# PLY loader
# ============================================================================

def load_ply_file(path: str) -> np.ndarray:
    if _load_ply_lib is not None:
        try:
            return _load_ply_lib(path)
        except Exception:
            pass
    # fallback binary parser
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            hdr += f.read(256)
    nv = 0
    for line in hdr.decode("ascii", "ignore").split("\n"):
        if line.startswith("element vertex"):
            nv = int(line.split()[-1])
    body = open(path, "rb").read()
    off = body.find(b"end_header\n") + len(b"end_header\n")
    pts = np.frombuffer(body[off:], dtype=np.float32).reshape(-1, 3)[:nv]
    return pts


# ============================================================================
# Viewer
# ============================================================================

class PcTopDownViewer:
    def __init__(self, ply_path: str, region=None, max_scatter=200_000):
        self.ply_path = ply_path
        self.max_scatter = max_scatter

        # Load
        print(f"Loading {ply_path} ...")
        self.points_full = load_ply_file(ply_path)
        self.n_total = len(self.points_full)
        print(f"Loaded {self.n_total:,} points")

        # Full bounds
        self.x_min_all = float(np.min(self.points_full[:, 0]))
        self.x_max_all = float(np.max(self.points_full[:, 0]))
        self.y_min_all = float(np.min(self.points_full[:, 1]))
        self.y_max_all = float(np.max(self.points_full[:, 1]))
        print(f"Full bounds: X=[{self.x_min_all:.2f}, {self.x_max_all:.2f}]  "
              f"Y=[{self.y_min_all:.2f}, {self.y_max_all:.2f}]")

        # Initial region
        if region is not None:
            self.xmin, self.xmax, self.ymin, self.ymax = region
        else:
            self.xmin, self.xmax = self.x_min_all, self.x_max_all
            self.ymin, self.ymax = self.y_min_all, self.y_max_all

        self._subsample()

        # Build UI
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.suptitle(
            f"Point Cloud Viewer — {os.path.basename(ply_path)}  "
            f"({self.n_total:,} pts total, {self.n_show:,} shown)",
            fontsize=11, fontweight="bold")

        # Main plot
        self.ax = self.fig.add_axes([0.08, 0.18, 0.84, 0.77])

        # Placeholder for status text (created before _draw populates it)
        bh, yr = 0.045, 0.08
        self.status_text = self.fig.text(0.86, yr + bh / 2, "", fontsize=8,
                                         va="center", family="monospace")

        self._draw()

        # --- Input fields ---
        bh, bw = 0.045, 0.10
        yr = 0.08

        self.tb_xmin = TextBox(self.fig.add_axes([0.08, yr, bw, bh]),
                               "X min:", initial=f"{self.xmin:.2f}")
        self.tb_xmin.on_submit(self._on_region_change)

        self.tb_xmax = TextBox(self.fig.add_axes([0.20, yr, bw, bh]),
                               "X max:", initial=f"{self.xmax:.2f}")
        self.tb_xmax.on_submit(self._on_region_change)

        self.tb_ymin = TextBox(self.fig.add_axes([0.34, yr, bw, bh]),
                               "Y min:", initial=f"{self.ymin:.2f}")
        self.tb_ymin.on_submit(self._on_region_change)

        self.tb_ymax = TextBox(self.fig.add_axes([0.46, yr, bw, bh]),
                               "Y max:", initial=f"{self.ymax:.2f}")
        self.tb_ymax.on_submit(self._on_region_change)

        self.btn_update = Button(self.fig.add_axes([0.59, yr, 0.07, bh]), "Update")
        self.btn_update.on_clicked(self._on_update)

        self.btn_reset = Button(self.fig.add_axes([0.68, yr, 0.07, bh]), "Reset")
        self.btn_reset.on_clicked(self._on_reset)

        self.btn_auto = Button(self.fig.add_axes([0.77, yr, 0.07, bh]), "Auto")
        self.btn_auto.on_clicked(self._on_auto)

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        print("\nMouse over plot → see coordinates in title bar.")
        print("Enter numbers in text boxes + Enter (or click 'Update').")

    def _subsample(self):
        mask = ((self.points_full[:, 0] >= self.xmin) &
                (self.points_full[:, 0] <= self.xmax) &
                (self.points_full[:, 1] >= self.ymin) &
                (self.points_full[:, 1] <= self.ymax))
        self.points_in_region = self.points_full[mask]
        n_in = len(self.points_in_region)
        if n_in > self.max_scatter:
            idx = np.random.RandomState(42).choice(n_in, self.max_scatter, replace=False)
            self.points_show = self.points_in_region[idx]
            self.n_show = self.max_scatter
            self.subsampled = True
        else:
            self.points_show = self.points_in_region
            self.n_show = n_in
            self.subsampled = False

    def _draw(self):
        self.ax.clear()

        self.ax.scatter(self.points_show[:, 0], self.points_show[:, 1],
                        s=0.5, c="black", alpha=0.7, rasterized=True)

        # Selected region (green solid)
        self.ax.add_patch(plt.Rectangle(
            (self.xmin, self.ymin), self.xmax - self.xmin, self.ymax - self.ymin,
            fill=False, edgecolor="#00cc00", linewidth=2.0, alpha=0.9))

        # Full bounds (orange dashed)
        self.ax.add_patch(plt.Rectangle(
            (self.x_min_all, self.y_min_all),
            self.x_max_all - self.x_min_all,
            self.y_max_all - self.y_min_all,
            fill=False, edgecolor="#ff6600", linewidth=1.0, linestyle="--", alpha=0.5))

        self.ax.set_xlabel("X World (m)")
        self.ax.set_ylabel("Y World (m)")
        self.ax.set_aspect("equal")

        # Auto tick spacing
        span = max(self.xmax - self.xmin, self.ymax - self.ymin)
        step = 10 ** math.floor(math.log10(span / 5))
        step = max(step, 0.5)
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(step))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        self.ax.xaxis.set_minor_locator(ticker.MultipleLocator(step / 5))
        self.ax.yaxis.set_minor_locator(ticker.MultipleLocator(step / 5))
        self.ax.grid(True, which="major", lw=0.5, alpha=0.4, color="#555555")
        self.ax.grid(True, which="minor", lw=0.3, alpha=0.2, color="#888888")
        self.ax.tick_params(labelsize=9)

        self.status_text.set_text(
            f"Region: {self.n_show:,}/{self.n_total:,} pts"
            + (" (subsampled)" if self.subsampled else ""))

        self.fig.canvas.draw_idle()

    def _update_region(self):
        try:
            self.xmin = float(self.tb_xmin.text)
            self.xmax = float(self.tb_xmax.text)
            self.ymin = float(self.tb_ymin.text)
            self.ymax = float(self.tb_ymax.text)
        except ValueError:
            print("Invalid number — use values like: -15.0")
            return
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            print("Need xmin < xmax and ymin < ymax")
            return
        self._subsample()
        self._draw()

    def _on_region_change(self, text):
        self._update_region()

    def _on_update(self, event):
        self._update_region()

    def _on_reset(self, event):
        self.xmin, self.xmax = self.x_min_all, self.x_max_all
        self.ymin, self.ymax = self.y_min_all, self.y_max_all
        self.tb_xmin.set_val(f"{self.xmin:.2f}")
        self.tb_xmax.set_val(f"{self.xmax:.2f}")
        self.tb_ymin.set_val(f"{self.ymin:.2f}")
        self.tb_ymax.set_val(f"{self.ymax:.2f}")
        self._subsample()
        self._draw()

    def _on_auto(self, event):
        """Auto-fit to occupied region (trim outliers)."""
        m = ((self.points_full[:, 0] >= self.x_min_all) &
             (self.points_full[:, 0] <= self.x_max_all) &
             (self.points_full[:, 1] >= self.y_min_all) &
             (self.points_full[:, 1] <= self.y_max_all))
        pts = self.points_full[m]
        if len(pts) == 0:
            return
        self.xmin = float(np.percentile(pts[:, 0], 0.5))
        self.xmax = float(np.percentile(pts[:, 0], 99.5))
        self.ymin = float(np.percentile(pts[:, 1], 0.5))
        self.ymax = float(np.percentile(pts[:, 1], 99.5))
        pad_x = (self.xmax - self.xmin) * 0.02
        pad_y = (self.ymax - self.ymin) * 0.02
        self.xmin -= pad_x; self.xmax += pad_x
        self.ymin -= pad_y; self.ymax += pad_y
        self.tb_xmin.set_val(f"{self.xmin:.2f}")
        self.tb_xmax.set_val(f"{self.xmax:.2f}")
        self.tb_ymin.set_val(f"{self.ymin:.2f}")
        self.tb_ymax.set_val(f"{self.ymax:.2f}")
        self._subsample()
        self._draw()

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            self.fig.suptitle(
                f"Point Cloud Viewer — {os.path.basename(self.ply_path)}  "
                f"({self.n_total:,} pts)  |  cursor = ({x:.3f}, {y:.3f})",
                fontsize=11, fontweight="bold")
            self.fig.canvas.draw_idle()


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Interactive Point Cloud Top-Down Viewer")
    p.add_argument("ply", help="Path to PLY file")
    p.add_argument("--region", type=float, nargs=4, default=None,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="Initial view region (default: full point cloud bounds)")
    p.add_argument("--auto", action="store_true",
                   help="Auto-fit to occupied area on startup")
    args = p.parse_args()

    if not os.path.exists(args.ply):
        print(f"ERROR: {args.ply} not found")
        sys.exit(1)

    viewer = PcTopDownViewer(args.ply, region=args.region)
    if args.auto:
        viewer._on_auto(None)

    plt.show()


if __name__ == "__main__":
    main()
