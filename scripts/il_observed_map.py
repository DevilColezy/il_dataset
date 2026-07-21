#!/usr/bin/env python3
"""
il_observed_map.py  —  Observed local occupancy map and ESDF for IL dataset v8.

Phase 2: Builds a rolling occupancy map from depth history, then constructs
a local ESDF from observed occupancy for the C++ local planner.

Key design:
  - Voxel states: UNKNOWN=0, FREE=1, OCCUPIED=2 (np.uint8).
  - Pinhole camera model for depth back-projection.
  - Depth integration with configurable downsampling.
  - Rolling window with optional history time window.
  - ESDF built via scipy.ndimage.distance_transform_edt.
  - Unknown space is NOT treated as free; must use known_mask.
"""

from __future__ import print_function, division

import math, time, collections
import numpy as np

# Camera coordinate convention (documented):
#   Flightmare depth camera: optical axis = +Z (forward in Unity),
#   image x = right, image y = down (OpenCV convention after flip).
#   After flipud in il_manager, image y = up.
#   We convert to FLU: forward=X, left=Y, up=Z for navigation.

UNKNOWN = np.uint8(0)
FREE = np.uint8(1)
OCCUPIED = np.uint8(2)


class PinholeCameraModel:
    """Pinhole camera model for depth back-projection.

    Convention:
      - Camera frame: X=right, Y=down, Z=forward (optical axis).
      - Output points are in camera frame (not FLU).
      - Callers convert to FLU/world as needed.

    Flightmare depth: z-depth along optical axis (parallel projection),
    not Euclidean ray length. Confirmed by the depth processing in
    il_manager which multiplies by 100 to convert to cm, then clips.
    """

    def __init__(self, width, height, horizontal_fov_rad,
                 vertical_fov_rad=None):
        self.width = int(width)
        self.height = int(height)
        self.hfov = float(horizontal_fov_rad)

        if vertical_fov_rad is None:
            # Compute vertical FOV from aspect ratio
            self.vfov = 2.0 * math.atan(
                math.tan(self.hfov / 2.0) * self.height / max(self.width, 1))
        else:
            self.vfov = float(vertical_fov_rad)

        # Focal lengths in pixels
        self.fx = (self.width / 2.0) / math.tan(self.hfov / 2.0)
        self.fy = (self.height / 2.0) / math.tan(self.vfov / 2.0)
        self.cx = (self.width - 1) / 2.0
        self.cy = (self.height - 1) / 2.0

    def backproject_depth(self, depth_m, step=4):
        """Back-project depth image to camera-frame 3D points.

        Args:
            depth_m: np.ndarray (H, W) of depth in metres (z-depth).
            step: pixel subsampling step (1 = full resolution).

        Returns:
            (points_cam, pixel_indices):
              points_cam: np.ndarray (N, 3) in camera frame [x_right, y_down, z_forward].
              pixel_indices: np.ndarray (N, 2) of (row, col) pixel positions.
        """
        h, w = depth_m.shape
        rows = np.arange(0, h, step)
        cols = np.arange(0, w, step)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        rr = rr.ravel()
        cc = cc.ravel()

        z = depth_m[rr, cc]
        valid = np.isfinite(z) & (z > 0) & (z < 1000.0)
        rr = rr[valid]
        cc = cc[valid]
        z = z[valid]

        x = (cc - self.cx) * z / self.fx
        y = (rr - self.cy) * z / self.fy

        points_cam = np.column_stack([x, y, z]).astype(np.float64)
        pixel_indices = np.column_stack([rr, cc]).astype(np.int32)
        return points_cam, pixel_indices

    def cam_to_flu(self, points_cam):
        """Convert camera-frame points to FLU frame.

        Camera: X=right, Y=down, Z=forward
        FLU:     X=forward, Y=left, Z=up

        Mapping: camera_x(right) -> FLU -Y (left=-right)
                 camera_y(down)  -> FLU -Z (up=-down)
                 camera_z(fwd)   -> FLU +X (forward)
        """
        flu = np.zeros_like(points_cam)
        flu[:, 0] = points_cam[:, 2]   # forward  = camera Z
        flu[:, 1] = -points_cam[:, 0]   # left     = -camera X
        flu[:, 2] = -points_cam[:, 1]   # up       = -camera Y
        return flu


class RollingObservedOccupancyMap:
    """Rolling occupancy map from depth history.

    Maintains a local voxel grid centered on the drone. Integrates
    depth observations and maintains a time-since-last-observation
    for history window management.
    """

    def __init__(self, config):
        cfg = config.get("global", {}).get("observed_map", {})
        self.resolution = float(cfg.get("resolution", 0.10))
        self.size_x_m = float(cfg.get("size_x_m", 12.0))
        self.size_y_m = float(cfg.get("size_y_m", 12.0))
        self.size_z_m = float(cfg.get("size_z_m", 5.0))
        self.history_seconds = float(cfg.get("history_seconds", 4.0))
        self.occ_endpoint_margin = float(cfg.get("occupied_endpoint_margin_m", 0.05))
        self.unknown_is_free = bool(cfg.get("unknown_is_free", False))
        self.min_known_free_ratio = float(cfg.get("min_known_free_ratio", 0.95))
        self.depth_step = int(cfg.get("depth_integration_step", 4))

        # Grid dimensions
        inv_res = 1.0 / self.resolution
        self.gx = int(math.ceil(self.size_x_m * inv_res))
        self.gy = int(math.ceil(self.size_y_m * inv_res))
        self.gz = int(math.ceil(self.size_z_m * inv_res))

        # Make dimensions odd for centering
        if self.gx % 2 == 0:
            self.gx += 1
        if self.gy % 2 == 0:
            self.gy += 1
        if self.gz % 2 == 0:
            self.gz += 1

        # Occupancy grid
        self._occ = np.full((self.gx, self.gy, self.gz), UNKNOWN, dtype=np.uint8)
        # Last observed time per voxel (monotonic seconds)
        self._last_obs_time = np.full((self.gx, self.gy, self.gz), -1.0, dtype=np.float64)
        # Current world center
        self._center_world = np.zeros(3, dtype=np.float64)
        # Grid origin (corner of voxel (0,0,0))
        self._origin_world = np.zeros(3, dtype=np.float64)
        self._initialized = False
        self._revision = 0

        # Stats
        self.total_integrations = 0

    # ── Grid coordinate helpers ──────────────────────────────────

    def world_to_grid(self, points_world):
        """Convert world points to continuous grid indices."""
        pts = np.asarray(points_world, dtype=np.float64)
        return (pts - self._origin_world) / self.resolution

    def grid_to_world(self, indices):
        """Convert grid indices (int or float) to world coordinates."""
        idx = np.asarray(indices, dtype=np.float64)
        return idx * self.resolution + self._origin_world

    def _world_to_grid_int(self, points_world):
        """Convert world points to integer grid indices (floor)."""
        g = self.world_to_grid(points_world)
        return np.floor(g).astype(np.int32)

    def _in_bounds(self, ix, iy, iz):
        """Check if integer grid indices are within bounds."""
        return ((ix >= 0) & (ix < self.gx) &
                (iy >= 0) & (iy < self.gy) &
                (iz >= 0) & (iz < self.gz))

    # ── Public API ───────────────────────────────────────────────

    def reset(self, center_world):
        """Reset map centered at a new world position."""
        self._center_world = np.asarray(center_world, dtype=np.float64)
        half_sizes = np.array([self.size_x_m, self.size_y_m, self.size_z_m]) / 2.0
        self._origin_world = self._center_world - half_sizes

        self._occ.fill(UNKNOWN)
        self._last_obs_time.fill(-1.0)
        self._initialized = True
        self._revision = 0

    def recenter_if_needed(self, center_world):
        """Re-center the map if the drone has moved too far."""
        center = np.asarray(center_world, dtype=np.float64)
        displacement = np.linalg.norm(center - self._center_world)
        # Recenter if displacement exceeds 1/4 of map extent
        half_size = min(self.size_x_m, self.size_y_m) / 4.0
        if displacement > half_size:
            self.reset(center_world)

    def integrate_depth(self, depth_m, camera_position_world,
                        camera_rotation_world, timestamp_s):
        """Integrate a depth image into the occupancy map.

        Args:
            depth_m: (H, W) float array of depth in metres.
            camera_position_world: [x, y, z] in ROS world.
            camera_rotation_world: yaw in radians (ROS convention).
            timestamp_s: monotonic time for history window.
        """
        if not self._initialized:
            return

        cam = PinholeCameraModel(
            depth_m.shape[1], depth_m.shape[0],
            math.radians(90.0))  # FOV from config, placeholder

        # Back-project depth
        points_cam, _ = cam.backproject_depth(depth_m, step=self.depth_step)
        if len(points_cam) == 0:
            return

        # Camera → FLU → World
        points_flu = cam.cam_to_flu(points_cam)
        yaw = float(camera_rotation_world)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        # FLU to world: rotate by yaw around Z, then translate
        points_world = np.zeros_like(points_flu)
        points_world[:, 0] = points_flu[:, 0] * cos_y - points_flu[:, 1] * sin_y
        points_world[:, 1] = points_flu[:, 0] * sin_y + points_flu[:, 1] * cos_y
        points_world[:, 2] = points_flu[:, 2]
        points_world += np.asarray(camera_position_world, dtype=np.float64)

        # Camera position in world
        cam_pos = np.asarray(camera_position_world, dtype=np.float64)

        # Purge expired history
        self._purge_expired(timestamp_s)

        # Rasterize each ray end-point
        occ_grid_ix = self._world_to_grid_int(points_world)
        occ_grid_iy = occ_grid_ix[:, 1]
        occ_grid_iz = occ_grid_ix[:, 2]
        occ_grid_ix = occ_grid_ix[:, 0]
        in_b = self._in_bounds(occ_grid_ix, occ_grid_iy, occ_grid_iz)

        # Mark occupied endpoints
        for k in np.where(in_b)[0]:
            ix, iy, iz = occ_grid_ix[k], occ_grid_iy[k], occ_grid_iz[k]
            if self._occ[ix, iy, iz] != OCCUPIED:
                self._occ[ix, iy, iz] = OCCUPIED
            self._last_obs_time[ix, iy, iz] = timestamp_s

        # Mark free space along rays using simplified sampling
        # Sample at resolution intervals along each ray
        for k in range(len(points_world)):
            if not in_b[k]:
                # Check if at least the ray enters the grid
                pass  # skip rays ending outside

            end_pt = points_world[k]
            ray_dir = end_pt - cam_pos
            ray_len = float(np.linalg.norm(ray_dir))
            if ray_len < 1e-6:
                continue

            ray_dir /= ray_len
            # Sample along ray at resolution intervals
            n_samples = max(1, int(ray_len / self.resolution))
            for s in range(n_samples):
                frac = s / max(n_samples, 1)
                pt = cam_pos + frac * ray_dir * (ray_len - self.occ_endpoint_margin)
                g = self._world_to_grid_int(pt)
                if self._in_bounds(g[0], g[1], g[2]):
                    ix, iy, iz = int(g[0]), int(g[1]), int(g[2])
                    # Only set FREE if not already OCCUPIED (occupy-first policy)
                    if self._occ[ix, iy, iz] == UNKNOWN:
                        self._occ[ix, iy, iz] = FREE
                        self._last_obs_time[ix, iy, iz] = timestamp_s

        self._revision += 1
        self.total_integrations += 1

    def _purge_expired(self, now_s):
        """Reset expired voxels to UNKNOWN."""
        if self.history_seconds <= 0:
            return
        expired = (now_s - self._last_obs_time) > self.history_seconds
        self._occ[expired & (self._occ != UNKNOWN)] = UNKNOWN
        self._last_obs_time[expired] = -1.0

    def get_occupancy(self):
        """Return the occupancy grid (gx, gy, gz) uint8."""
        return self._occ.copy()

    def get_known_mask(self):
        """Return boolean mask where voxels are known (FREE or OCCUPIED)."""
        return self._occ != UNKNOWN

    def get_origin(self):
        """Return grid origin world coordinates."""
        return self._origin_world.copy()

    def get_resolution(self):
        return self.resolution

    def get_center(self):
        return self._center_world.copy()

    def get_revision(self):
        return self._revision

    def known_voxel_count(self):
        return int(np.sum(self._occ != UNKNOWN))

    def occupied_voxel_count(self):
        return int(np.sum(self._occ == OCCUPIED))

    def free_voxel_count(self):
        return int(np.sum(self._occ == FREE))

    def is_known(self, point_world):
        """Check if a world point is in known space."""
        g = self._world_to_grid_int(point_world)
        if not self._in_bounds(g[0], g[1], g[2]):
            return False
        return self._occ[int(g[0]), int(g[1]), int(g[2])] != UNKNOWN

    def is_known_free(self, point_world):
        """Check if a world point is known and free."""
        g = self._world_to_grid_int(point_world)
        if not self._in_bounds(g[0], g[1], g[2]):
            return False
        return self._occ[int(g[0]), int(g[1]), int(g[2])] == FREE

    def sample_known_free_ratio_along_corridor(
            self, start_world, end_world, radius_m, spacing_m,
            min_clearance_m=0.0):
        """Sample points along a corridor and compute the known-free ratio.

        This is a simplified 2.5D cylinder check. For each sample point
        along the line, we check if it is known and has sufficient clearance.

        Returns:
            float in [0, 1]: fraction of samples that are known-free.
        """
        start = np.asarray(start_world, dtype=np.float64)
        end = np.asarray(end_world, dtype=np.float64)
        vec = end - start
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            return 0.0

        n_samples = max(2, int(length / spacing_m) + 1)
        n_samples = min(n_samples, 200)  # upper bound for performance

        known_free_count = 0
        total_samples = 0

        for i in range(n_samples):
            frac = i / max(n_samples - 1, 1)
            center_pt = start + frac * vec
            g = self._world_to_grid_int(center_pt)
            if not self._in_bounds(g[0], g[1], g[2]):
                continue
            total_samples += 1
            if self._occ[int(g[0]), int(g[1]), int(g[2])] == FREE:
                # Simplified: only check center point for now.
                # Full drone-radius check would require ESDF.
                known_free_count += 1

        if total_samples == 0:
            return 0.0
        return known_free_count / total_samples


class ObservedESDF:
    """Builds an ESDF from an observed occupancy map.

    Uses scipy.ndimage.distance_transform_edt for the first version.
    Unknown space is NOT treated as free — use known_mask separately.
    """

    def __init__(self, config):
        cfg = config.get("global", {}).get("observed_map", {})
        self.esdf_max_distance_m = float(cfg.get("esdf_max_distance_m", 5.0))
        self.rebuild_every_n_frames = int(cfg.get("rebuild_every_n_frames", 1))

        self._esdf = None  # (gx, gy, gz) float32
        self._known_mask = None  # (gx, gy, gz) bool
        self._origin = None
        self._resolution = 0.0
        self._built = False
        self._build_count = 0

    def rebuild(self, occupancy, known_mask, origin_world, resolution):
        """Rebuild ESDF from occupancy grid.

        Args:
            occupancy: (gx, gy, gz) uint8 array (UNKNOWN=0, FREE=1, OCCUPIED=2).
            known_mask: (gx, gy, gz) bool array.
            origin_world: [ox, oy, oz] corner of voxel (0,0,0).
            resolution: voxel size in metres.
        """
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            raise ImportError(
                "scipy is required for ObservedESDF. Install: pip install scipy")

        # Occupied voxels for EDT
        occupied = (occupancy == OCCUPIED).astype(np.uint8)

        n_occ = int(occupied.sum())
        if n_occ == 0:
            # No known obstacles: all known free gets max distance
            self._esdf = np.full_like(occupancy, self.esdf_max_distance_m,
                                      dtype=np.float32)
            # But unknown stays at 0 (not free!)
            self._esdf[~known_mask] = 0.0
        else:
            fd = distance_transform_edt(1 - occupied, sampling=resolution)
            od = distance_transform_edt(occupied, sampling=resolution)
            esdf_raw = fd - od

            # Clamp to max distance
            esdf_raw = np.clip(esdf_raw, -self.esdf_max_distance_m,
                               self.esdf_max_distance_m)

            # Unknown voxels: set to 0 (not usable as free)
            self._esdf = esdf_raw.astype(np.float32)
            self._esdf[~known_mask] = 0.0

        self._known_mask = known_mask.copy()
        self._origin = np.asarray(origin_world, dtype=np.float64).copy()
        self._resolution = float(resolution)
        self._built = True
        self._build_count += 1

    def get_esdf(self):
        """Return ESDF float32 array (gx, gy, gz)."""
        return self._esdf

    def get_known_mask(self):
        """Return known mask bool array (gx, gy, gz)."""
        return self._known_mask

    def get_origin(self):
        return self._origin

    def get_resolution(self):
        return self._resolution

    def is_built(self):
        return self._built
