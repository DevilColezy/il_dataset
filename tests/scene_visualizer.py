#!/usr/bin/env python3
"""
Scene Obstacle Top-Down Visualization

Generates or loads a cylinder-obstacle scene and renders a 2D top-down view
with coordinate axes, obstacle circles, and region boundaries.

Usage:
    # Default: load S01 scene from config, render top-down view
    python scene_visualizer.py

    # Specify a particular scene index and seed
    python scene_visualizer.py --scene-index 0 --seed 42

    # Manual mode: place a few random obstacles without config
    python scene_visualizer.py --manual

    # Custom figure size and DPI
    python scene_visualizer.py --figsize 14 10 --dpi 200

    # Save without displaying
    python scene_visualizer.py --output scene_topdown.png --no-show
"""

import argparse
import os
import sys
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# --- matplotlib setup ---
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

# Try to import project modules (may need ROS or local path)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_SCRIPTS = os.path.join(_SCRIPT_DIR, "..", "scripts")
_PROJECT_CONFIG = os.path.join(_SCRIPT_DIR, "..", "config")
if _PROJECT_SCRIPTS not in sys.path:
    sys.path.insert(0, _PROJECT_SCRIPTS)
if _PROJECT_CONFIG not in sys.path:
    sys.path.insert(0, _PROJECT_CONFIG)

_HAS_PROJECT = False
_YamlCylinderSceneGenerator = None
_CylinderObstacleSpec = None
_ObstacleRegion = None
_SceneGenerationProfile = None
_load_config = None

try:
    from il_scenario import (
        YamlCylinderSceneGenerator,
        CylinderObstacleSpec,
        ObstacleRegion,
        SceneGenerationProfile,
    )
    from il_config import load_config
    _HAS_PROJECT = True
except ImportError as e:
    print(f"[WARN] Cannot import il_scenario/il_config: {e}")
    print("[WARN] Falling back to manual obstacle mode")


# ============================================================================
# Scene Generation
# ============================================================================

def _resolve_config_path() -> str:
    """Find the YAML config file."""
    candidates = [
        os.path.join(_PROJECT_CONFIG, "il_dataset_config.yaml"),
        os.path.join(_SCRIPT_DIR, "..", "config", "il_dataset_config.yaml"),
        os.path.join(os.path.dirname(_PROJECT_SCRIPTS), "config", "il_dataset_config.yaml"),
    ]
    # Also try ROS-style paths
    try:
        import rospkg
        rp = rospkg.RosPack()
        pkg_path = rp.get_path("il_dataset")
        candidates.insert(0, os.path.join(pkg_path, "config", "il_dataset_config.yaml"))
    except Exception:
        pass
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def _load_config_ros_free(config_path: str):
    """Load config without requiring a running ROS master.

    Monkey-patches il_config._apply_ros_overrides to be a no-op.
    """
    import il_config

    # Stash original
    _original_apply = il_config._apply_ros_overrides

    def _noop_ros_overrides(cfg):
        pass

    il_config._apply_ros_overrides = _noop_ros_overrides
    try:
        cfg = il_config.load_config(config_path, validate=False)
    finally:
        il_config._apply_ros_overrides = _original_apply

    # Fill in default output_dir if missing (normally set after ROS overrides)
    if not cfg.get("global", {}).get("output_dir"):
        cfg["global"]["output_dir"] = os.path.join(
            os.path.dirname(config_path), "..", "dataset", "il_data"
        )
    return cfg


def generate_scene_s01(seed: int = 42, scene_index: int = 0) -> Tuple[List, str]:
    """Generate an S01 scene using the YAML config generator.

    Returns (obstacles_list, profile_name).
    """
    config_path = _resolve_config_path()
    if not config_path:
        raise FileNotFoundError("Cannot find il_dataset_config.yaml")

    print(f"[INFO] Config: {config_path}")

    # Use ROS-free loading — we only need scene geometry, not ROS params
    cfg = _load_config_ros_free(config_path)
    gen = YamlCylinderSceneGenerator(cfg)

    # Find S01 profile
    profiles = gen.get_profiles()
    s01 = None
    for p in profiles:
        if p.name and "S01" in p.name:
            s01 = p
            break
    if s01 is None:
        # Fallback to first profile
        s01 = profiles[0]
        print(f"[INFO] S01 not found by name, using first profile: {s01.name}")

    print(f"[INFO] Profile: {s01.name}")
    region = gen.obstacle_region
    print(f"[INFO] Region: x=[{region.x_min:.1f}, {region.x_max:.1f}], "
          f"y=[{region.y_min:.1f}, {region.y_max:.1f}]")
    print(f"[INFO] Cylinder radius: [{s01.radius_min_m:.2f}, {s01.radius_max_m:.2f}] m, "
          f"count: [{s01.count_min}, {s01.count_max}], height: [{s01.height_min_m:.1f}, {s01.height_max_m:.1f}] m")
    print(f"[INFO] Vehicle radius: {s01.vehicle_radius_m:.2f} m, safety margin: {s01.safety_margin_m:.2f} m")

    effective_seed = seed + s01.seed_offset if hasattr(s01, 'seed_offset') else seed
    print(f"[INFO] Seed: {seed}, offset: {getattr(s01, 'seed_offset', 0)}, effective: {effective_seed}")

    obstacles, rejection, target_density, density_mode = \
        gen.generate_scene_from_profile(s01, effective_seed, scene_index_in_profile=scene_index)

    if rejection:
        print(f"[WARN] Scene rejected: {rejection}")

    print(f"[INFO] Generated {len(obstacles)} obstacles")
    return obstacles, s01.name


def generate_manual_obstacles(
    seed: int = 42,
    region: Optional[Tuple[float, float, float, float]] = None,
) -> List:
    """Generate a few random cylinder obstacles manually (no config needed).

    Uses CylinderObstacleSpec if available, otherwise returns plain dicts.
    """
    rng = np.random.RandomState(seed)

    if region is None:
        # Default: 20m x 30m region like S01
        region = (-9.0, 11.0, 0.0, 30.0)

    x_min, x_max, y_min, y_max = region
    z_ground = 0.0
    height = 8.0
    z_center = z_ground + height / 2.0  # 4.0

    # Generate 8-15 obstacles with varying radii
    n = rng.randint(8, 16)
    obstacles = []
    for i in range(n):
        # Avoid edges
        margin = 1.0
        cx = float(rng.uniform(x_min + margin, x_max - margin))
        cy = float(rng.uniform(y_min + margin, y_max - margin))
        radius = float(rng.uniform(0.15, 0.50))
        obs_id = f"cyl_{i:04d}"

        if _CylinderObstacleSpec is not None:
            obs = _CylinderObstacleSpec(
                obstacle_id=obs_id,
                center_world=np.array([cx, cy, z_center]),
                radius_m=radius,
                height_m=height,
            )
        else:
            # Fallback: plain dict
            obs = {
                "id": obs_id,
                "center": np.array([cx, cy, z_center]),
                "radius_m": radius,
                "height_m": height,
            }
        obstacles.append(obs)

    print(f"[INFO] Manual mode: generated {n} obstacles in region "
          f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    return obstacles


# ============================================================================
# Obstacle Access Helpers
# ============================================================================

def obs_center_xy(obs) -> np.ndarray:
    """Get XY center of an obstacle (supports CylinderObstacleSpec and dict)."""
    if hasattr(obs, "center_xy"):
        return obs.center_xy()
    elif hasattr(obs, "center_world"):
        return obs.center_world[:2].copy()
    else:
        return np.array(obs["center"][:2], dtype=float)

def obs_center_xyz(obs) -> np.ndarray:
    """Get XYZ center of an obstacle."""
    if hasattr(obs, "center_world"):
        return obs.center_world.copy()
    else:
        return np.array(obs.get("center", [0.0, 0.0, 4.0]), dtype=float)

def obs_radius(obs) -> float:
    if hasattr(obs, "radius_m"):
        return obs.radius_m
    return obs["radius_m"]

def obs_height(obs) -> float:
    if hasattr(obs, "height_m"):
        return obs.height_m
    return obs.get("height_m", 8.0)

def obs_id(obs) -> str:
    if hasattr(obs, "obstacle_id"):
        return obs.obstacle_id
    return obs.get("id", "?")


# ============================================================================
# Plotting
# ============================================================================

def plot_scene_topdown(
    obstacles: List,
    region_bounds: Tuple[float, float, float, float],
    vehicle_radius: float = 0.30,
    safety_margin: float = 0.10,
    title: str = "Obstacle Scene — Top-Down View",
    show_vehicle_scale: bool = True,
    show_grid: bool = True,
    figsize: Tuple[float, float] = (14, 10),
) -> Tuple[plt.Figure, plt.Axes]:
    """Render a top-down 2D view of cylinder obstacles.

    Parameters
    ----------
    obstacles : list
        List of CylinderObstacleSpec or dict obstacles.
    region_bounds : (x_min, x_max, y_min, y_max)
        The rectangular region boundary.
    vehicle_radius : float
        Vehicle collision radius for scale reference circle.
    safety_margin : float
        Additional safety margin (inflation radius shown as dashed).
    title : str
        Plot title.
    show_vehicle_scale : bool
        Draw a reference circle showing vehicle scale.
    show_grid : bool
        Show grid lines.
    figsize : (w, h)
        Figure size in inches.

    Returns
    -------
    (figure, axes)
    """
    x_min, x_max, y_min, y_max = region_bounds

    # Compute reasonable aspect ratio
    x_span = x_max - x_min
    y_span = y_max - y_min

    fig, ax = plt.subplots(figsize=figsize)

    # ---- Region boundary ----
    region_rect = Rectangle(
        (x_min, y_min), x_span, y_span,
        fill=False, edgecolor="black", linewidth=2.0,
        linestyle="--", alpha=0.7, zorder=1,
        label="Region boundary"
    )
    ax.add_patch(region_rect)

    # ---- Obstacles as filled circles ----
    radii = [obs_radius(o) for o in obstacles]
    if radii:
        r_min, r_max = min(radii), max(radii)

    for obs in obstacles:
        cx, cy = obs_center_xy(obs)
        r = obs_radius(obs)
        oid = obs_id(obs)

        # Color map: small = blue, large = red
        if r_max > r_min:
            t = (r - r_min) / (r_max - r_min)
        else:
            t = 0.5
        color = plt.cm.plasma(0.2 + 0.6 * t)

        # Cylinder body
        circle = Circle(
            (cx, cy), r,
            fill=True, facecolor=color, edgecolor="black",
            linewidth=1.2, alpha=0.85, zorder=3,
        )
        ax.add_patch(circle)

        # Label small obstacles
        if len(obstacles) <= 20 or r > 0.3:
            ax.annotate(
                f"r={r:.2f}",
                (cx, cy),
                fontsize=6,
                ha="center", va="center",
                color="white" if t > 0.5 else "black",
                fontweight="bold",
                zorder=4,
            )

    # ---- Vehicle scale reference ----
    if show_vehicle_scale:
        ref_x = x_min + 1.5
        ref_y = y_min + 1.5
        # Vehicle body
        v_circle = Circle(
            (ref_x, ref_y), vehicle_radius,
            fill=False, edgecolor="#e74c3c", linewidth=2.0,
            linestyle="-", zorder=5,
            label=f"Vehicle (r={vehicle_radius:.2f}m)",
        )
        ax.add_patch(v_circle)
        # Inflation boundary
        infl_r = vehicle_radius + safety_margin
        i_circle = Circle(
            (ref_x, ref_y), infl_r,
            fill=False, edgecolor="#e67e22", linewidth=1.5,
            linestyle=":", zorder=4,
            label=f"Inflation (r={infl_r:.2f}m)",
        )
        ax.add_patch(i_circle)
        # Center dot
        ax.plot(ref_x, ref_y, "o", color="#e74c3c", markersize=4, zorder=6)

    # ---- Axes configuration ----
    ax.set_xlim(x_min - 2.0, x_max + 2.0)
    ax.set_ylim(y_min - 2.0, y_max + 2.0)

    ax.set_xlabel("X World (m) — Forward →", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y World (m) — Left →", fontsize=13, fontweight="bold")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    # Aspect ratio = 1 for non-distorted circles
    ax.set_aspect("equal")

    # Coordinate ticks: major every 2m, minor every 1m
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1.0))

    if show_grid:
        ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.4, color="#555555")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.25, color="#888888")

    # Tick labels on all sides
    ax.tick_params(which="both", direction="in", labelsize=9)
    ax.tick_params(which="major", length=6, width=1.0)
    ax.tick_params(which="minor", length=3, width=0.5)

    # ---- Legend ----
    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # ---- Info text ----
    text_lines = [
        f"Obstacles: {len(obstacles)}",
        f"Radius range: {min(radii):.2f}–{max(radii):.2f} m" if radii else "",
        f"Region: {x_span:.0f}×{y_span:.0f} m",
        f"Vehicle r: {vehicle_radius:.2f} m",
    ]
    text_str = "\n".join(l for l in text_lines if l)
    ax.text(
        0.02, 0.98, text_str,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    plt.tight_layout()
    return fig, ax


# ============================================================================
# Obstacle Stats Printer
# ============================================================================

def print_obstacle_stats(obstacles: List):
    """Print a summary table of obstacle positions and radii."""
    print(f"\n{'ID':<20s} {'X (m)':>8s} {'Y (m)':>8s} {'Z (m)':>8s} {'Radius':>8s} {'Height':>8s}")
    print("-" * 64)
    for obs in obstacles:
        cx, cy = obs_center_xy(obs)
        cz = obs_center_xyz(obs)[2] if hasattr(obs, "center_world") else obs.get("center", [0, 0, 4.0])[2]
        print(f"{obs_id(obs):<20s} {cx:>8.2f} {cy:>8.2f} {cz:>8.2f} {obs_radius(obs):>7.2f}m {obs_height(obs):>7.2f}m")

    radii = [obs_radius(o) for o in obstacles]
    print(f"\n  Count: {len(obstacles)},  Radius: min={min(radii):.3f}  max={max(radii):.3f}  "
          f"mean={np.mean(radii):.3f}  median={np.median(radii):.3f}")


# ============================================================================# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scene Obstacle Top-Down Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scene_visualizer.py                          # S01 scene from config
  python scene_visualizer.py --manual                 # Random obstacles, no config
  python scene_visualizer.py --scene-index 1 --seed 123
  python scene_visualizer.py --output scene.png --no-show --dpi 200
        """,
    )
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: generate random obstacles without config files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--scene-index", type=int, default=0,
                        help="Scene index for profile generation (default: 0)")
    parser.add_argument("--region", type=float, nargs=4,
                        default=None,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Region bounds for manual mode (default: -9 11 0 30)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the plot window")
    parser.add_argument("--figsize", type=float, nargs=2, default=(14, 10),
                        metavar=("W", "H"),
                        help="Figure size in inches (default: 14 10)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI (default: 150)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Hide grid lines")
    parser.add_argument("--no-vehicle-scale", action="store_true",
                        help="Hide vehicle scale reference")
    return parser.parse_args()


def main():
    args = parse_args()

    # Generate scene
    if args.manual or not _HAS_PROJECT:
        region_bounds = tuple(args.region) if args.region else (-9.0, 11.0, 0.0, 30.0)
        obstacles = generate_manual_obstacles(seed=args.seed, region=region_bounds)
        profile_name = "manual"
        vehicle_radius = 0.30
        safety_margin = 0.10
    else:
        obstacles, profile_name = generate_scene_s01(seed=args.seed, scene_index=args.scene_index)
        # Extract region from the profile used
        _DEFAULT_REGION = (-9.0, 11.0, 0.0, 30.0)
        try:
            config_path = _resolve_config_path()
            cfg = _load_config_ros_free(config_path)
            gen2 = YamlCylinderSceneGenerator(cfg)
            profiles = gen2.get_profiles()
            s01 = next((p for p in profiles if p.name and "S01" in p.name), profiles[0])
            rb = gen2.obstacle_region
            region_bounds = (rb.x_min, rb.x_max, rb.y_min, rb.y_max)
            vehicle_radius = s01.vehicle_radius_m
            safety_margin = s01.safety_margin_m
        except Exception:
            region_bounds = _DEFAULT_REGION
            vehicle_radius = 0.30
            safety_margin = 0.10

    if not obstacles:
        print("ERROR: No obstacles generated!")
        sys.exit(1)

    # Print stats
    print_obstacle_stats(obstacles)

    # Plot
    title = f"Obstacle Scene — Top-Down View\nProfile: {profile_name}  |  Seed: {args.seed}  |  "
    title += f"{len(obstacles)} obstacles"

    fig, ax = plot_scene_topdown(
        obstacles,
        region_bounds=region_bounds,
        vehicle_radius=vehicle_radius,
        safety_margin=safety_margin,
        title=title,
        show_vehicle_scale=not args.no_vehicle_scale,
        show_grid=not args.no_grid,
        figsize=args.figsize,
    )

    # Output
    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved: {out_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
