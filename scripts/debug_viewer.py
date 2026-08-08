#!/usr/bin/env python3
"""debug_viewer.py  —  interactive episode browser for the il_dataset output.

Opens a GUI (tkinter + matplotlib) that:
  * scans the dataset directory (default: dataset/il_data) and lists all
    success / failed / in-progress episodes in a tree panel
  * steps through frames with a slider, buttons, arrow keys or playback
  * shows, per frame: flight trajectory, current pose + heading arrow,
    macro guide waypoint, active local trajectory + terminal, macro
    mode timeline, and macro candidate points (from trace.jsonl)

Usage:
    python debug_viewer.py                 # default dataset/il_data
    python debug_viewer.py /path/to/dir

Pure read-only tool: nothing is saved, no images are written.
"""

from __future__ import print_function, division

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict

try:
    import numpy as np
except ImportError:
    np = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None

# Stable enum names (mirror types.hpp / il_macro_expert.SideFailure).
_MODE_NAMES = {0: "DIRECT_GUIDE", 1: "SIDE_GUIDE", 2: "OBSERVE",
               3: "GOAL_REACHED", 4: "FAILED"}
_SIDE_NAMES = {0: "NONE", 1: "LEFT", -1: "RIGHT"}
_REC_STATUS_NAMES = {0: "DIRECT_REJOIN_SUCCESS", 1: "PARTIAL_PROGRESS_ONLY",
                     2: "BLOCKED_BY_KNOWN", 3: "BLOCKED_BY_UNKNOWN",
                     4: "NO_SAFE_MOTION"}
_CAND_TYPE_NAMES = {0: "DIRECT", 1: "SIDE", 2: "OBSERVE",
                    3: "GOAL_FRONTIER", 4: "PREVIOUS_CONT"}
_MODE_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c",
                3: "#d62728", 4: "#9467bd"}
_CAND_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c",
                3: "#d62728", 4: "#9467bd"}

# Map legend rows: (swatch color, label) — what is drawn on the map.
_LEGEND_ITEMS = [
    ("#b0b0b0", "Global map (scene PLY)"),
    ("#4db3ff", "Local observation - free"),
    ("#e63826", "Local observation - obstacle"),
    ("#7f7f7f", "Flight trajectory"),
    ("#2ca02c", "Start"),
    ("#d62728", "Goal"),
    ("#000000", "Current pose + heading"),
    ("#1f77b4", "Macro guide waypoint"),
    ("#ff7f0e", "Local trajectory (plan)"),
    ("#e377c2", "Local terminal"),
    ("#9467bd", "Candidates"),
]

_IGNORED_DIRS = {"_inprogress", "maps", "scenes", "_failed", "_debug"}


# ── Small conversion helpers ──────────────────────────────────────────
def _f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _b(value, default=False):
    if value in (None, ""):
        return default
    return str(value).lower() in ("1", "true", "yes")


def _fmt_bool(value):
    return "yes" if _b(value) else "no"


# ── Data loaders ──────────────────────────────────────────────────────
def _read_metadata(path):
    meta_path = os.path.join(path, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, IOError):
        return None


def _load_csv(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _load_trace(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                print("[debug_viewer] skipping bad trace line: %s" % exc,
                      file=sys.stderr)
    return rows


def _load_plan_points(path):
    """plan_id -> list of [x, y, z] in trajectory order."""
    if not os.path.isfile(path):
        return {}
    plans = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = row.get("plan_id", "")
            if pid == "":
                continue
            try:
                pt = [_f(row.get("x")), _f(row.get("y")), _f(row.get("z"))]
            except (TypeError, ValueError):
                continue
            plans.setdefault(pid, []).append(pt)
    return plans


# ── PLY / 2D map helpers ──────────────────────────────────────────────
_PLY_TYPE_SIZES = {"char": 1, "uchar": 1, "short": 2, "ushort": 2,
                   "int": 4, "uint": 4, "float": 4, "double": 8}
_PLY_TYPE_FORMATS = {"char": "i1", "uchar": "u1", "short": "i2",
                     "ushort": "u2", "int": "i4", "uint": "u4",
                     "float": "f4", "double": "f8"}


def _load_ply_points(path):
    """Return (N, 3) float64 xyz from an ASCII or binary PLY, or None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            header = b""
            while b"end_header" not in header:
                chunk = f.readline()
                if not chunk:
                    return None
                header += chunk
        header_size = len(header)
        lines = header.decode("ascii", errors="replace").splitlines()
        fmt, n_vert, props = "ascii", 0, []
        in_vertex = False
        for ln in lines:
            s = ln.strip()
            if s.startswith("format"):
                fmt = s.split()[1]
            elif s.startswith("element vertex"):
                n_vert = int(s.split()[2])
                in_vertex = True
            elif s.startswith("element"):
                in_vertex = False
            elif in_vertex and s.startswith("property"):
                parts = s.split()
                if len(parts) >= 3:
                    props.append((parts[2], parts[1]))
        if n_vert <= 0:
            return None
        if fmt == "ascii":
            with open(path, "rb") as f:
                f.seek(header_size)
                rest = f.read()
            tokens = rest.split()
            arr = np.array(tokens, dtype=np.float64)
            if arr.size == n_vert * 3:
                arr = arr.reshape(n_vert, 3)
            elif arr.size == n_vert * len(props):
                arr = arr.reshape(n_vert, len(props))
            else:
                return None
            pts = arr[:, :3].astype(np.float64)
        else:
            little = "little" in fmt
            endian = "<" if little else ">"
            dt = np.dtype([(name, endian + _PLY_TYPE_FORMATS.get(ptype, "f4"))
                           for name, ptype in props])
            with open(path, "rb") as f:
                f.seek(header_size)
                raw = f.read(n_vert * dt.itemsize)
            arr = np.frombuffer(raw, dtype=dt, count=n_vert)
            if not all(n in arr.dtype.names for n in ("x", "y", "z")):
                return None
            pts = np.column_stack(
                [arr["x"], arr["y"], arr["z"]]).astype(np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return pts if len(pts) else None
    except Exception:
        return None


def _build_occupancy(pts, res=0.2):
    """Bin world XY points into a 2D occupancy grid.

    Returns (grid (ny, nx) uint8, extent [minx, maxx, miny, maxy]) or
    (None, None) when there is nothing to draw.
    """
    if pts is None or len(pts) == 0:
        return None, None
    minx, miny = float(pts[:, 0].min()), float(pts[:, 1].min())
    maxx, maxy = float(pts[:, 0].max()), float(pts[:, 1].max())
    nx = int(math.floor((maxx - minx) / res)) + 1
    ny = int(math.floor((maxy - miny) / res)) + 1
    grid = np.zeros((ny, nx), dtype=np.uint8)
    ix = np.floor((pts[:, 0] - minx) / res).astype(np.int64)
    iy = np.floor((pts[:, 1] - miny) / res).astype(np.int64)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(grid, (iy[m], ix[m]), 1)
    return grid, [minx, maxx, miny, maxy]


class LocalMapBuilder(object):
    """2D local occupancy around the drone, rebuilt from the stored depth
    PNG of the current frame.

    Projection matches the C++ ObservedMap::integrateDepth (flightlib body
    frame [x_right, y_forward, z_up]; pixel -> cam: x=(col-cx)*z/fx,
    y=(row-cy)*z/fy, z=depth).  The 2D map uses the horizontal depth band
    around the image centre: for every grid cell, the range at the cell's
    bearing decides free / occupied / unknown.
    """

    W = 640
    H = 480
    HFOV_DEG = 90.0
    MAX_DEPTH = 5.0
    RES = 0.1
    HALF = 7.0
    BAND_ROWS = 21

    def __init__(self, episode):
        self.episode = episode
        self.available = True
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            self.available = False
            return
        hfov = math.radians(self.HFOV_DEG)
        self.fx = (self.W * 0.5) / math.tan(hfov * 0.5)
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * self.H / self.W)
        self.fy = (self.H * 0.5) / math.tan(vfov * 0.5)
        self.cx = (self.W - 1.0) * 0.5
        self.cy = (self.H - 1.0) * 0.5
        n = int(round(2.0 * self.HALF / self.RES))
        self.n = n
        r = (np.arange(n) + 0.5) * self.RES - self.HALF
        self.rx, self.ry = np.meshgrid(r, r)
        half_rows = self.BAND_ROWS // 2
        self.rows = np.arange(int(round(self.cy)) - half_rows,
                              int(round(self.cy)) + half_rows + 1)
        self._band_cache = OrderedDict()

    def _band_for(self, frame_id):
        """Min depth per column over the centre band (0 = invalid column)."""
        band = self._band_cache.get(frame_id)
        if band is not None:
            self._band_cache.move_to_end(frame_id)
            return band
        if not self.available:
            return None
        png = os.path.join(self.episode.path, "depth",
                           "%06d.png" % int(frame_id))
        if not os.path.isfile(png):
            return None
        try:
            from PIL import Image
            img = np.asarray(Image.open(png), dtype=np.uint16)
        except Exception:
            return None
        if img.shape[0] < self.H or img.shape[1] < self.W:
            img = img[:self.H, :self.W]
        band = img[self.rows, :].astype(np.float64) / 100.0
        valid = band > 0.0
        col_valid = valid.any(axis=0)
        masked = np.where(valid, band, self.MAX_DEPTH)
        bmin = masked.min(axis=0)
        bmin = np.where(col_valid, bmin, 0.0)
        self._band_cache[frame_id] = bmin
        if len(self._band_cache) > 128:
            self._band_cache.popitem(last=False)
        return bmin

    def grid_for(self, frame_id, x, y, yaw):
        """Return (free (ny, nx) bool, occ (ny, nx) bool) around (x, y)."""
        band = self._band_for(frame_id)
        if band is None:
            return None, None
        cosy, siny = math.cos(yaw), math.sin(yaw)
        bx = cosy * self.rx + siny * self.ry      # body right
        by = -siny * self.rx + cosy * self.ry     # body forward
        dist = np.hypot(self.rx, self.ry)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(by > 1e-6, bx / np.maximum(by, 1e-6), 1e9)
        col = self.cx + self.fx * ratio
        col_int = np.clip(np.floor(col).astype(np.int64), 0, self.W - 1)
        in_fov = (by > 1e-6) & (np.abs(ratio) <= 1.001)
        depth_at = band[col_int]
        valid = in_fov & (depth_at > 0.0)
        free = valid & (dist < depth_at - 0.06)
        hit = valid & (depth_at < self.MAX_DEPTH - 0.01)
        occ = hit & ~free & (dist <= depth_at + 0.25)
        return free, occ


# ── Episode ───────────────────────────────────────────────────────────
_SCENE_MAP_CACHE = {}


class Episode(object):
    """One collected episode: data.csv (+ optional trace.jsonl and
    local_plan_points.csv).  Data is loaded lazily on first use."""

    def __init__(self, path, status, meta, root):
        self.path = path
        self.root = root
        self.status = status            # "success" | "failed" | "inprogress"
        self.meta = meta or {}
        self.episode_id = self.meta.get("episode_id") or \
            os.path.basename(path).replace(".inprogress", "")
        self.exit_reason = self.meta.get("exit_reason", "")
        self.scene_id = self.meta.get("scene_id", "")
        self.task_id = self.meta.get("task_id", "")
        self.has_trace = os.path.isfile(os.path.join(path, "trace.jsonl"))
        self._rows = None
        self._trace = None
        self._plan_points = None
        self._tick_for_frame = None
        self._scene_map = None

    def rows(self):
        if self._rows is None:
            self._rows = _load_csv(os.path.join(self.path, "data.csv"))
        return self._rows

    def trace(self):
        if self._trace is None:
            self._trace = _load_trace(os.path.join(self.path, "trace.jsonl"))
        return self._trace

    def plan_points(self):
        if self._plan_points is None:
            self._plan_points = _load_plan_points(
                os.path.join(self.path, "local_plan_points.csv"))
        return self._plan_points

    def tick_for_frame(self):
        """Index into trace() for each data.csv row (last tick <= frame)."""
        if self._tick_for_frame is None:
            rows = self.rows()
            tr = self.trace()
            out = [None] * len(rows)
            j = -1
            frames = [_i(r.get("frame"), -1) for r in tr]
            for i, row in enumerate(rows):
                fid = _i(row.get("frame_id"), -1)
                while j + 1 < len(tr) and frames[j + 1] <= fid:
                    j += 1
                out[i] = j if j >= 0 else None
            self._tick_for_frame = out
        return self._tick_for_frame

    def scene_map(self):
        """Global 2D occupancy grid from maps/<scene_key>.ply (cached)."""
        if self._scene_map is None:
            ply = os.path.join(self.root, "maps",
                               str(self.scene_id) + ".ply")
            if ply in _SCENE_MAP_CACHE:
                self._scene_map = _SCENE_MAP_CACHE[ply]
            else:
                grid, extent = _build_occupancy(_load_ply_points(ply))
                self._scene_map = (grid, extent)
                _SCENE_MAP_CACHE[ply] = self._scene_map
        return self._scene_map


# ── Dataset scan ──────────────────────────────────────────────────────
def _episode_status(meta, fallback):
    """Status from metadata.json when present, else location fallback.

    The writer's output_root is the manager's `_inprogress/` directory, so
    COMMITTED episodes live under _inprogress/<ep> (final, not in-progress)
    and REJECTED ones under _inprogress/_failed/<ep>.  The recorded status
    is therefore authoritative.
    """
    if meta is not None:
        mstatus = meta.get("status", "")
        if mstatus == "committed":
            return "success"
        if mstatus == "rejected":
            return "failed"
        if mstatus == "inprogress":
            return "inprogress"
    return fallback


def scan_dataset(root):
    """Return (success, failed, inprogress) episode lists under `root`."""
    success, failed, inprogress = [], [], []
    if not os.path.isdir(root):
        return success, failed, inprogress
    for name in sorted(os.listdir(root)):
        if name.startswith(".") or name in _IGNORED_DIRS:
            continue
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        meta = _read_metadata(path)
        if meta is None and not os.path.isfile(
                os.path.join(path, "data.csv")):
            continue
        ep = Episode(path, _episode_status(meta, "inprogress"), meta, root)
        if ep.status == "success":
            success.append(ep)
        elif ep.status == "failed":
            failed.append(ep)
        else:
            inprogress.append(ep)
    # Writer sub-directories: root _failed/ and _inprogress/, plus the
    # nested _inprogress/_failed/ produced when the writer output_root is
    # the manager's _inprogress_root.
    for sub_root, fallback in (
            (os.path.join(root, "_failed"), "failed"),
            (os.path.join(root, "_inprogress", "_failed"), "failed"),
            (os.path.join(root, "_inprogress"), "inprogress")):
        if not os.path.isdir(sub_root):
            continue
        for name in sorted(os.listdir(sub_root)):
            if name.startswith(".") or name == "_failed":
                continue
            path = os.path.join(sub_root, name)
            if not os.path.isdir(path):
                continue
            meta = _read_metadata(path)
            ep = Episode(path, _episode_status(meta, fallback), meta, root)
            if ep.status == "success":
                success.append(ep)
            elif ep.status == "failed":
                failed.append(ep)
            else:
                inprogress.append(ep)
    return success, failed, inprogress


class ViewerApp(object):
    """tkinter GUI: episode tree + matplotlib frame viewer."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.episodes = {"success": [], "failed": [], "inprogress": []}
        self.current = None            # Episode
        self.frame = 0
        self._play = False
        self._play_job = None
        self._view_vars = {}
        self._local_builder = None
        self.show = {"yaw": True, "guide": True, "local": True,
                     "cands": True, "side": True, "mode": True,
                     "global_map": True, "local_map": True, "follow": True}

        self.root = tk.Tk()
        self.root.title("IL Dataset Debug Viewer — %s" % root_dir)
        self.root.geometry("1500x860")

        self._build_menu()
        self._build_layout()
        self._build_figure()
        self._build_controls()
        self._bind_keys()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._scan()
        self.root.mainloop()

    # ── UI construction ─────────────────────────────────────────────
    def _build_menu(self):
        menubar = tk.Menu(self.root)

        m_file = tk.Menu(menubar, tearoff=0)
        m_file.add_command(label="Open Dataset Dir…", command=self._open_dir)
        m_file.add_command(label="Refresh List", command=self._scan)
        m_file.add_separator()
        m_file.add_command(label="Quit", command=self._on_close)
        menubar.add_cascade(label="File", menu=m_file)

        m_view = tk.Menu(menubar, tearoff=0)
        for key, label in (("yaw", "Heading Arrow"), ("guide", "Macro Guide"),
                           ("local", "Local Trajectory"), ("cands", "Candidates"),
                           ("side", "Side View"), ("mode", "Mode Timeline"),
                           ("global_map", "Global Map"),
                           ("local_map", "Local Observation"),
                           ("follow", "Auto Follow")):
            var = tk.BooleanVar(value=self.show[key])
            self._view_vars[key] = var
            m_view.add_checkbutton(
                label=label, variable=var,
                command=(lambda k=key: self._on_view_toggle(k)))
        menubar.add_cascade(label="View", menu=m_view)

        m_help = tk.Menu(menubar, tearoff=0)
        m_help.add_command(label="Usage", command=self._show_help)
        menubar.add_cascade(label="Help", menu=m_help)
        self.root.config(menu=menubar)

    def _build_layout(self):
        self.pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.pane.pack(fill=tk.BOTH, expand=True)

        # Left column: episode tree (top) + map legend (bottom).
        self.left = ttk.Frame(self.pane, width=300)
        self.pane.add(self.left, weight=0)
        self.left.pack_propagate(False)
        self._build_tree(self.left)
        self._build_legend(self.left)

        # Centre: figure + playback controls.
        self.center = ttk.Frame(self.pane)
        self.pane.add(self.center, weight=1)

        # Right column: current-frame data.
        self.right = ttk.Frame(self.pane, width=370)
        self.pane.add(self.right, weight=0)
        self.right.pack_propagate(False)
        self._build_frame_panel(self.right)

    def _build_tree(self, parent):
        box = ttk.LabelFrame(parent, text="Episodes")
        box.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.tree = ttk.Treeview(box, show="tree headings",
                                 selectmode="browse")
        self.tree["columns"] = ("reason",)
        self.tree.column("#0", width=190, anchor=tk.W)
        self.tree.column("reason", width=90, anchor=tk.W)
        self.tree.heading("reason", text="Reason")
        scroll = ttk.Scrollbar(box, orient=tk.VERTICAL,
                               command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _build_legend(self, parent):
        box = ttk.LabelFrame(parent, text="Map Layers")
        box.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)
        for color, label in _LEGEND_ITEMS:
            row = ttk.Frame(box)
            row.pack(fill=tk.X, padx=8, pady=1)
            swatch = tk.Canvas(row, width=14, height=10,
                               highlightthickness=1,
                               highlightbackground="#999999")
            swatch.pack(side=tk.LEFT, padx=(0, 6))
            swatch.create_rectangle(0, 0, 14, 10, fill=color, outline="")
            ttk.Label(row, text=label, anchor=tk.W).pack(side=tk.LEFT)

    def _build_frame_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Current Frame")
        box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.info = tk.Text(box, width=44, state=tk.DISABLED,
                            font=("TkFixedFont", 9), wrap=tk.WORD)
        scroll = ttk.Scrollbar(box, orient=tk.VERTICAL,
                               command=self.info.yview)
        self.info.configure(yscrollcommand=scroll.set)
        self.info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_figure(self):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(11.5, 6.8), dpi=100)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2, 1],
                                   height_ratios=[1, 1])
        self.ax_xy = self.fig.add_subplot(gs[0, 0])
        self.ax_xz = self.fig.add_subplot(gs[0, 1])
        self.ax_mode = self.fig.add_subplot(gs[1, :])
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.center)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                         expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.center)
        toolbar.update()

    def _build_controls(self):
        bar = ttk.Frame(self.center)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)

        btns = ttk.Frame(bar)
        btns.pack(side=tk.LEFT)
        for text, cmd in (("|<", self._goto_first), ("<", self._step_back),
                          (">", self._step_fwd), (">|", self._goto_last)):
            ttk.Button(btns, text=text, width=3, command=cmd).pack(
                side=tk.LEFT, padx=1)
        self.btn_play = ttk.Button(btns, text="Play", width=6,
                                   command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        self.frame_var = tk.IntVar(value=0)
        self.slider = tk.Scale(bar, from_=0, to=0, orient=tk.HORIZONTAL,
                               variable=self.frame_var,
                               command=self._on_slider, showvalue=True,
                               length=520)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.status = tk.StringVar(value="")
        status_bar = tk.Label(self.root, textvariable=self.status, anchor=tk.W,
                              relief=tk.SUNKEN, bd=1)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        self.root.bind("<Left>", lambda e: self._step_back())
        self.root.bind("<Right>", lambda e: self._step_fwd())
        self.root.bind("<Prior>", lambda e: self._goto_first())
        self.root.bind("<Next>", lambda e: self._goto_last())
        self.root.bind("<space>", lambda e: self._toggle_play())

    # ── Menus / tree ────────────────────────────────────────────────
    def _on_view_toggle(self, key):
        self.show[key] = bool(self._view_vars[key].get())
        if self.current is not None:
            self._draw()

    def _open_dir(self):
        d = filedialog.askdirectory(initialdir=self.root_dir,
                                    title="Select dataset directory")
        if d:
            self.root_dir = d
            self.root.title("IL Dataset Debug Viewer — %s" % d)
            self._scan()

    def _show_help(self):
        messagebox.showinfo(
            "Usage",
            "Left / Right arrows: previous / next frame\n"
            "PageUp / PageDown: jump to start / end\n"
            "Space: play / pause (real 30 Hz time)\n"
            "View menu: global map (scene PLY), local observation (2D\n"
            "  occupancy rebuilt from the current frame's depth), auto\n"
            "  follow (camera follows the drone), and overlay toggles\n"
            "Left panel: choose success / failed / in-progress trajectories\n"
            "Figure toolbar: zoom / pan")  # noqa: E501

    def _scan(self):
        success, failed, inprogress = scan_dataset(self.root_dir)
        self.episodes = {"success": success, "failed": failed,
                         "inprogress": inprogress}
        for item in self.tree.get_children():
            self.tree.delete(item)
        for status, label in (("success", "Successful"), ("failed", "Failed"),
                              ("inprogress", "In Progress")):
            eps = self.episodes[status]
            root_iid = "root:" + status
            self.tree.insert("", "end", iid=root_iid, open=True,
                             text="%s (%d)" % (label, len(eps)))
            for ep in eps:
                iid = "ep:%s:%s" % (status, ep.episode_id)
                self.tree.insert(root_iid, "end", iid=iid,
                                 text=ep.episode_id,
                                 values=(ep.exit_reason,))
        self.status.set("Scanned %s: success %d / failed %d / in-progress %d"
                        % (self.root_dir, len(success), len(failed),
                           len(inprogress)))

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel or not sel[0].startswith("ep:"):
            return
        _, status, ep_id = sel[0].split(":", 2)
        for ep in self.episodes.get(status, []):
            if ep.episode_id == ep_id:
                self._load_episode(ep)
                return

    def _load_episode(self, ep):
        self._play = False
        self.btn_play.config(text="Play")
        self.current = ep
        n = len(ep.rows())
        self.slider.config(to=max(0, n - 1))
        self._set_frame(0)

    # ── Frame navigation ────────────────────────────────────────────
    def _goto_first(self):
        if self.current is not None:
            self._set_frame(0)

    def _goto_last(self):
        if self.current is not None:
            self._set_frame(len(self.current.rows()) - 1)

    def _step_back(self):
        if self.current is not None and self.frame > 0:
            self._set_frame(self.frame - 1)

    def _step_fwd(self):
        if self.current is not None and \
                self.frame < len(self.current.rows()) - 1:
            self._set_frame(self.frame + 1)

    def _on_slider(self, value):
        if self.current is not None:
            self._set_frame(int(float(value)), update_slider=False)

    def _set_frame(self, index, update_slider=True):
        self.frame = index
        if update_slider:
            self.frame_var.set(index)
        self._draw()

    def _toggle_play(self):
        self._play = not self._play
        self.btn_play.config(text="Pause" if self._play else "Play")
        if self._play:
            self._play_loop()

    def _play_loop(self):
        if not self._play:
            return
        n = len(self.current.rows()) if self.current else 0
        if self.frame >= n - 1:
            self._play = False
            self.btn_play.config(text="Play")
            return
        self.frame += 1
        self._set_frame(self.frame)
        self._play_job = self.root.after(33, self._play_loop)

    # ── Drawing ─────────────────────────────────────────────────────
    @staticmethod
    def _vec(v):
        try:
            if v is None:
                return None
            return [float(v[0]), float(v[1]), float(v[2])]
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _trajectory(rows):
        xs = [_f(r.get("x")) for r in rows]
        ys = [_f(r.get("y")) for r in rows]
        zs = [_f(r.get("z")) for r in rows]
        return xs, ys, zs

    def _draw(self):
        ep = self.current
        if ep is None:
            return
        rows = ep.rows()
        if not rows:
            return
        i = min(max(self.frame, 0), len(rows) - 1)
        self.frame = i
        row = rows[i]
        self._draw_xy(ep, rows, i, row)
        self._draw_side(ep, rows, i, row)
        self._draw_mode(ep, rows, i, row)
        self._update_info(ep, row, i)
        self._update_status(ep, row, i)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_xy(self, ep, rows, i, row):
        ax = self.ax_xy
        ax.clear()
        xs, ys, _ = self._trajectory(rows)

        # Global static map (scene PLY) as the 2D plane background.
        if self.show["global_map"]:
            self._draw_global_map(ax, ep)

        ax.plot(xs, ys, color="0.7", lw=1.2, zorder=3, label="Flight trajectory")

        start = self._vec(ep.meta.get("start_world"))
        goal = self._vec(ep.meta.get("goal_world"))
        if start is None and rows:
            start = [xs[0], ys[0], _f(rows[0].get("z"))]
        if goal is None and rows:
            goal = [_f(row.get("goal_world_x")), _f(row.get("goal_world_y")),
                    _f(row.get("goal_world_z"))]
        if start:
            ax.plot([start[0]], [start[1]], marker="s", color="#2ca02c",
                    ms=9, zorder=7, label="Start")
        if goal:
            ax.plot([goal[0]], [goal[1]], marker="*", color="#d62728",
                    ms=16, zorder=7, label="Goal")

        x, y = _f(row.get("x")), _f(row.get("y"))
        yaw = _f(row.get("yaw"))

        # Local observed map rebuilt from the depth PNG of this frame.
        if self.show["local_map"]:
            self._draw_local_map(ax, ep, row, x, y, yaw)

        if self.show["guide"] and row.get("macro_guide_world_x") not in ("", None):
            gx, gy = _f(row.get("macro_guide_world_x")), \
                _f(row.get("macro_guide_world_y"))
            ax.plot([gx], [gy], marker="X", color="#1f77b4", ms=11,
                    zorder=8, label="Macro guide")
            ax.plot([x, gx], [y, gy], ls="--", color="#1f77b4", lw=1.0,
                    alpha=0.6, zorder=5)

        if self.show["local"]:
            pid = row.get("plan_id", "")
            pts = ep.plan_points().get(pid) if pid else None
            if pts:
                px = [p[0] for p in pts]
                py = [p[1] for p in pts]
                ax.plot(px, py, color="#ff7f0e", lw=2.0, zorder=4,
                        label="Local trajectory")
                ax.plot([px[-1]], [py[-1]], marker="^", color="#e377c2",
                        ms=10, zorder=8, label="Local terminal")
            elif row.get("local_terminal_world_x") not in ("", None):
                tx, ty = _f(row.get("local_terminal_world_x")), \
                    _f(row.get("local_terminal_world_y"))
                ax.plot([tx], [ty], marker="^", color="#e377c2",
                        ms=10, zorder=8, label="Local terminal")

        if self.show["cands"]:
            self._draw_candidates(ax, ep, i)

        ax.plot([x], [y], marker="o", color="black", ms=7, zorder=9,
                label="Current")
        if self.show["yaw"]:
            dx, dy = -math.sin(yaw), math.cos(yaw)
            ax.arrow(x, y, dx, dy, head_width=0.35, head_length=0.45,
                     fc="black", ec="black", zorder=10)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Top-down (x-y)")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8, ncol=2)

        if self.show["follow"]:
            half = 9.0
            ax.set_xlim(x - half, x + half)
            ax.set_ylim(y - half, y + half)
        else:
            ax.autoscale(enable=True, axis="both", tight=True)

    def _draw_global_map(self, ax, ep):
        grid, extent = ep.scene_map()
        if grid is None or extent is None:
            return
        ax.imshow((grid > 0).astype(np.float32), extent=extent,
                  origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.5,
                  interpolation="nearest", zorder=0)

    def _draw_local_map(self, ax, ep, row, x, y, yaw):
        if self._local_builder is None or \
                self._local_builder.episode is not ep:
            self._local_builder = LocalMapBuilder(ep)
        if not self._local_builder.available:
            return
        fid = _i(row.get("frame_id"), -1)
        if fid < 0:
            return
        free, occ = self._local_builder.grid_for(fid, x, y, yaw)
        if free is None:
            return
        h, w = free.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[free, 0] = 0.30
        rgba[free, 1] = 0.70
        rgba[free, 2] = 1.00
        rgba[free, 3] = 0.30
        rgba[occ, 0] = 0.90
        rgba[occ, 1] = 0.22
        rgba[occ, 2] = 0.15
        rgba[occ, 3] = 0.80
        half = LocalMapBuilder.HALF
        ax.imshow(rgba, extent=[x - half, x + half, y - half, y + half],
                  origin="lower", zorder=2)

    def _draw_candidates(self, ax, ep, i):
        tick = ep.tick_for_frame()[i]
        if tick is None:
            return
        rec = ep.trace()[tick]
        for c in rec.get("candidates") or []:
            pos = c.get("pos")
            if not pos or len(pos) < 2:
                continue
            ctype = _i(c.get("type"), 0)
            color = _CAND_COLORS.get(ctype, "gray")
            ax.plot([pos[0]], [pos[1]], marker="+", color=color, ms=12,
                    lw=2, zorder=6)
            label = "%s/%s %.2f" % (_CAND_TYPE_NAMES.get(ctype, "?"),
                                     c.get("side", "?"), _f(c.get("score")))
            ax.annotate(label, (pos[0], pos[1]), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color=color)

    def _draw_side(self, ep, rows, i, row):
        ax = self.ax_xz
        ax.clear()
        if not self.show["side"]:
            ax.set_visible(False)
            return
        ax.set_visible(True)
        xs, _, zs = self._trajectory(rows)
        ax.plot(xs, zs, color="0.7", lw=1.2, zorder=1)
        goal = self._vec(ep.meta.get("goal_world"))
        if goal:
            ax.plot([goal[0]], [goal[2]], marker="*", color="#d62728",
                    ms=14, zorder=5)
        x, z = _f(row.get("x")), _f(row.get("z"))
        ax.plot([x], [z], marker="o", color="black", ms=7, zorder=6)
        ax.axhline(z, color="black", lw=0.5, alpha=0.4, ls=":")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title("Side view (x-z)")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    def _draw_mode(self, ep, rows, i, row):
        ax = self.ax_mode
        ax.clear()
        if not self.show["mode"]:
            ax.set_visible(False)
            return
        ax.set_visible(True)
        tframes, tmodes = [], []
        for r in rows:
            if _b(r.get("macro_is_new_tick")):
                tframes.append(_i(r.get("frame_id")))
                tmodes.append(_i(r.get("macro_mode")))
        if tframes:
            ax.step(tframes, tmodes, where="post", color="0.3", lw=1.2)
            ax.scatter(tframes, tmodes,
                       c=[_MODE_COLORS.get(m, "gray") for m in tmodes],
                       s=30, zorder=3)
        ax.axvline(_i(row.get("frame_id"), i), color="red", lw=1.0,
                   ls="--", zorder=2)
        ax.set_yticks(sorted(_MODE_NAMES.keys()))
        ax.set_yticklabels(
            [_MODE_NAMES[k] for k in sorted(_MODE_NAMES.keys())], fontsize=8)
        ax.set_xlabel("frame")
        ax.set_title("Macro Mode Timeline")
        ax.grid(True, alpha=0.25)

    # ── Text panels ─────────────────────────────────────────────────
    def _update_info(self, ep, row, i):
        mode = _MODE_NAMES.get(_i(row.get("macro_mode"), -1), "?")
        side = _SIDE_NAMES.get(_i(row.get("macro_committed_side"), 0), "?")
        obs = _SIDE_NAMES.get(_i(row.get("macro_observe_side"), 0), "?")
        lines = [
            "Frame %d  t=%.2fs  pos=(%.2f, %.2f, %.2f)  yaw=%.3f  dyaw=%.3f"
            % (i, _f(row.get("trajectory_time_s")), _f(row.get("x")),
               _f(row.get("y")), _f(row.get("z")), _f(row.get("yaw")),
               _f(row.get("desired_yaw_delta"))),
            "Macro %s  side=%s  observe_side=%s  conf=%.3f  %s"
            % (mode, side, obs, _f(row.get("macro_confidence")),
               row.get("macro_decision_reason", "")),
            "evidence=%s  observable=%s  no_progress=%.2fs  observe_no_info=%.2fs  blocker=%s"
            % (_fmt_bool(row.get("causal_intervention_evidence")),
               _fmt_bool(row.get("macro_decision_observable")),
               _f(row.get("direct_no_progress_time")),
               _f(row.get("observe_no_information_time")),
               row.get("blocker_track_id", "")),
            "Local plan=%s  status=%s  clearance=%.3f  dur=%.3fs  cached=%s"
            % (row.get("plan_id", ""), row.get("plan_status", ""),
               _f(row.get("minimum_clearance")),
               _f(row.get("trajectory_duration_s")),
               _fmt_bool(row.get("active_plan_is_cached"))),
        ]
        tick = ep.tick_for_frame()[i]
        if tick is not None:
            rec = ep.trace()[tick]
            lines.append("Recoverability=%s  failedL=%s failedR=%s  candidates=%d"
                         % (_REC_STATUS_NAMES.get(
                                _i(rec.get("rec_status"), -1), "?"),
                            rec.get("failed_left", ""),
                            rec.get("failed_right", ""),
                            len(rec.get("candidates") or [])))
            for c in rec.get("candidates") or []:
                pos = c.get("pos")
                pstr = "(%.2f, %.2f, %.2f)" % (pos[0], pos[1], pos[2]) \
                    if pos and len(pos) >= 3 else "(no pos)"
                lines.append("  cand %s/%s score=%.3f %s full=%s conn=%s"
                             % (_CAND_TYPE_NAMES.get(
                                    _i(c.get("type"), -1), "?"),
                                c.get("side", "?"), _f(c.get("score")),
                                pstr, _fmt_bool(c.get("full")),
                                _fmt_bool(c.get("conn"))))
        self.info.config(state=tk.NORMAL)
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, "\n".join(lines))
        self.info.config(state=tk.DISABLED)

    def _update_status(self, ep, row, i):
        n = len(ep.rows())
        status = {"success": "success", "failed": "failed",
                  "inprogress": "in-progress"}.get(ep.status, ep.status)
        self.status.set("%s [%s]  frame %d/%d  t=%.2fs  mode %s  side %s  yaw=%.3f"
                        % (ep.episode_id, status, i + 1, n,
                           _f(row.get("trajectory_time_s")),
                           _MODE_NAMES.get(_i(row.get("macro_mode"), -1), "?"),
                           _SIDE_NAMES.get(
                               _i(row.get("macro_committed_side"), 0), "?"),
                           _f(row.get("yaw"))))

    def _on_close(self):
        if self._play_job:
            self.root.after_cancel(self._play_job)
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive episode browser for the il_dataset output.")
    parser.add_argument("dir", nargs="?", default="dataset/il_data",
                        help="dataset directory (default: dataset/il_data)")
    args = parser.parse_args(argv)

    if tk is None:
        print("[debug_viewer] tkinter not available.", file=sys.stderr)
        print("  Install it (Ubuntu/Debian):  sudo apt install python3-tk",
              file=sys.stderr)
        return 1
    if np is None:
        print("[debug_viewer] numpy not available.", file=sys.stderr)
        return 1
    try:
        import matplotlib
        matplotlib.use("TkAgg")
    except ImportError:
        print("[debug_viewer] matplotlib not available.", file=sys.stderr)
        return 1
    if not os.path.isdir(args.dir):
        print("[debug_viewer] dataset dir not found: %s" % args.dir,
              file=sys.stderr)
        print("  Use: python debug_viewer.py /path/to/dataset", file=sys.stderr)
        return 1

    ViewerApp(os.path.abspath(args.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
