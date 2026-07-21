#!/usr/bin/env python3
"""
il_esdf_cache.py  —  ESDF Caching for Global and Observed Maps  (Phase 4)

Provides:
  - GlobalESDFCache: persistent disk cache for global ESDF, keyed by
    scene content hash + builder parameters.  Reused across episodes,
    tasks, and DAgger rounds.
  - ObservedESDFCache: in-memory revision tracker to avoid redundant
    rebuilds and C++ uploads of the observed (rolling) ESDF.

Global ESDF caching does not alter finite-observation expert semantics.
Observed ESDF cache is cleared at every episode reset.
"""

from __future__ import print_function, division

import hashlib, json, math, os, shutil, time, collections
import numpy as np

import rospy

# Omit scipy import here — GlobalESDFCache lazy-imports from il_common


# ============================================================================
#  Utility: Stable JSON serialisation for hashing
# ============================================================================

def _stable_json_dumps(obj):
    """Deterministic JSON serialisation with sorted keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str)


def _sha256_hex(data_bytes):
    return hashlib.sha256(data_bytes).hexdigest()


# ============================================================================
#  Lightweight file lock (no extra dependency)
# ============================================================================

class _SimpleFileLock:
    """Lightweight advisory file lock via atomic file creation."""

    def __init__(self, lock_path, timeout_s=120.0):
        self._path = lock_path
        self._timeout = timeout_s
        self._fd = None

    def acquire(self):
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline and not rospy.is_shutdown():
            try:
                fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
                return True
            except OSError:
                time.sleep(0.05)
        return False

    def release(self):
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(
                "Could not acquire ESDF cache lock: {} (timeout {:.0f}s)".format(
                    self._path, self._timeout))
        return self

    def __exit__(self, *args):
        self.release()


# ============================================================================
#  Global ESDF Cache
# ============================================================================

class GlobalESDFCache:
    """Persistent disk cache for the global (ground-truth) ESDF.

    Cache key is derived from:
      - scene manifest / explicit layout content (hash of obstacle list)
      - point cloud file content hash
      - ESDF resolution, origin, dimensions
      - vehicle radius, safety margin
      - builder version
    """

    def __init__(self, config):
        cfg = config.get("global", {}).get("esdf_cache", {})
        self._enabled = bool(cfg.get("enabled", True))
        g_cfg = cfg.get("global", {})

        self.cache_root = os.path.abspath(
            cfg.get("cache_root", os.path.join(
                config.get("global", {}).get("output_dir", "."),
                "cache", "esdf")))
        self._format = g_cfg.get("format", "npz")
        self._compression = bool(g_cfg.get("compression", True))
        self._validate_hash = bool(g_cfg.get("validate_hash", True))
        self._validate_meta = bool(g_cfg.get("validate_metadata", True))
        self._rebuild_on_mismatch = bool(g_cfg.get("rebuild_on_mismatch", True))
        self._atomic_write = bool(g_cfg.get("atomic_write", True))
        self._lock_timeout = float(g_cfg.get("lock_timeout_s", 120.0))
        self._builder_version = int(g_cfg.get("builder_version", 1))

        # Global ESDF parameters (set when building)
        self._esdf_cfg = config.get("global", {}).get("esdf", {})
        self._pointcloud_cfg = config.get("global", {}).get("pointcloud", {})
        self._obs_cfg = config.get("global", {}).get("obstacle", {})

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_build_ms = 0.0
        self.total_load_ms = 0.0
        self.total_write_ms = 0.0

    @property
    def enabled(self):
        return self._enabled

    def compute_cache_key(self, scene_manifest_dict, ply_path):
        """Compute a deterministic cache key from scene and ESDF parameters.

        Args:
            scene_manifest_dict: OrderedDict representation of the scene
                (obstacles list, region, etc.) or explicit layout dict.
            ply_path: path to point cloud PLY file.

        Returns:
            cache_key string (hex) or None if PLY not found.
        """
        if not os.path.isfile(ply_path):
            rospy.logwarn("[ESDFCache] PLY file not found for hashing: %s", ply_path)
            return None

        # Hash the point cloud file
        ply_hash = _sha256_hex(open(ply_path, "rb").read())

        # Build metadata dict for hashing
        meta = collections.OrderedDict([
            ("scene", scene_manifest_dict),
            ("ply_hash", ply_hash),
            ("esdf_resolution", float(self._esdf_cfg.get("resolution", 0.1))),
            ("esdf_drone_radius", float(self._esdf_cfg.get("drone_radius", 0.2))),
            ("pc_range", list(self._pointcloud_cfg.get("range", [30, 50, 8]))),
            ("pc_origin", list(self._pointcloud_cfg.get("origin", [0, 20, 3.5]))),
            ("pc_resolution", float(self._pointcloud_cfg.get("resolution", 0.1))),
            ("builder_version", self._builder_version),
        ])

        meta_json = _stable_json_dumps(meta)
        return _sha256_hex(meta_json.encode("utf-8"))

    def get_cache_dir(self, cache_key):
        return os.path.join(self.cache_root, "global", cache_key)

    def load(self, cache_key):
        """Load ESDF from cache. Returns (esdf, origin, resolution) or None on miss."""
        if not self._enabled:
            return None

        cache_dir = self.get_cache_dir(cache_key)
        esdf_path = os.path.join(cache_dir, "esdf.npz")
        meta_path = os.path.join(cache_dir, "metadata.json")

        if not os.path.isfile(esdf_path) or not os.path.isfile(meta_path):
            self.cache_misses += 1
            return None

        t0 = time.monotonic()
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Validate metadata
            if self._validate_meta:
                if not meta.get("complete", False):
                    rospy.logwarn("[ESDFCache] Incomplete cache at %s", cache_key)
                    self.cache_misses += 1
                    return None
                if meta.get("builder_version", 0) != self._builder_version:
                    rospy.loginfo("[ESDFCache] Builder version mismatch; rebuilding.")
                    self.cache_misses += 1
                    return None
                if (meta.get("cache_key", "") != cache_key and
                        self._validate_hash):
                    rospy.logwarn("[ESDFCache] Cache key mismatch; rebuilding.")
                    self.cache_misses += 1
                    return None

            # Load ESDF
            data = np.load(esdf_path)
            esdf = data["esdf"]
            origin = data["origin"]
            resolution = float(data["resolution"])

            # Validation
            if self._validate_meta:
                expected_shape = tuple(meta.get("shape", []))
                if esdf.shape != expected_shape:
                    rospy.logwarn("[ESDFCache] Shape mismatch in cache %s", cache_key)
                    self.cache_misses += 1
                    return None
            if not np.all(np.isfinite(esdf)):
                rospy.logwarn("[ESDFCache] NaN/Inf in cached ESDF %s", cache_key)
                self.cache_misses += 1
                return None

            load_ms = (time.monotonic() - t0) * 1000.0
            self.total_load_ms += load_ms
            self.cache_hits += 1
            rospy.loginfo("[ESDFCache] Loaded global ESDF from cache (%.1f ms).", load_ms)
            return esdf, origin, resolution

        except Exception as e:
            rospy.logwarn("[ESDFCache] Failed to load cache %s: %s", cache_key, e)
            self.cache_misses += 1
            return None

    def save(self, cache_key, esdf, origin, resolution, scene_id=""):
        """Save ESDF to cache atomically."""
        if not self._enabled:
            return

        cache_dir = self.get_cache_dir(cache_key)

        # Acquire lock
        lock_path = os.path.join(self.cache_root, "global", cache_key + ".lock")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        lock = _SimpleFileLock(lock_path, self._lock_timeout)

        try:
            with lock:
                # If already exists and valid, skip
                if os.path.isdir(cache_dir):
                    meta_path = os.path.join(cache_dir, "metadata.json")
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r") as f:
                                m = json.load(f)
                            if m.get("complete", False) and m.get("cache_key") == cache_key:
                                rospy.loginfo("[ESDFCache] Cache already valid; skipping write.")
                                return
                        except Exception:
                            pass

                t0 = time.monotonic()

                # Write to temp directory
                tmp_dir = cache_dir + ".tmp"
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                os.makedirs(tmp_dir, exist_ok=True)

                # Save ESDF
                esdf_path = os.path.join(tmp_dir, "esdf.npz")
                if self._compression:
                    np.savez_compressed(esdf_path, esdf=esdf,
                                        origin=np.asarray(origin, dtype=np.float64),
                                        resolution=np.float64(resolution))
                else:
                    np.savez(esdf_path, esdf=esdf,
                             origin=np.asarray(origin, dtype=np.float64),
                             resolution=np.float64(resolution))

                # Save metadata
                meta = collections.OrderedDict([
                    ("cache_key", cache_key),
                    ("builder_version", self._builder_version),
                    ("scene_id", str(scene_id)),
                    ("shape", list(esdf.shape)),
                    ("origin", [float(x) for x in origin]),
                    ("resolution", float(resolution)),
                    ("dtype", str(esdf.dtype)),
                    ("sign_convention", "positive_is_free_negative_is_obstacle"),
                    ("vehicle_radius_m", float(self._esdf_cfg.get("drone_radius", 0.2))),
                    ("safety_margin_m", float(self._obs_cfg.get("safety_margin", 0.2))),
                    ("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                    ("complete", True),
                ])
                meta_path_tmp = os.path.join(tmp_dir, "metadata.json")
                with open(meta_path_tmp, "w") as f:
                    json.dump(meta, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                os.rename(tmp_dir, cache_dir)

                write_ms = (time.monotonic() - t0) * 1000.0
                self.total_write_ms += write_ms
                rospy.loginfo("[ESDFCache] Saved global ESDF cache %s (%.1f ms).",
                              cache_key[:16], write_ms)

        except RuntimeError as e:
            rospy.logerr("[ESDFCache] Lock timeout for %s: %s", cache_key, e)

    def stats_summary(self):
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_load_ms": round(self.total_load_ms, 1),
            "total_write_ms": round(self.total_write_ms, 1),
            "total_build_ms": round(self.total_build_ms, 1),
        }


# ============================================================================
#  Observed ESDF Cache (in-memory, revision-based)
# ============================================================================

class ObservedESDFCache:
    """Tracks observed ESDF revisions to avoid redundant rebuilds and C++ uploads.

    Observed ESDF cache is cleared at every episode reset.
    """

    def __init__(self, config):
        cfg = config.get("global", {}).get("esdf_cache", {}).get("observed", {})
        self._enabled = bool(cfg.get("enabled", True))

        self._rebuild_only_on_change = bool(
            cfg.get("rebuild_only_on_map_change", True))
        self._upload_only_on_change = bool(
            cfg.get("upload_only_on_revision_change", True))
        self._occ_change_threshold = int(
            cfg.get("occupancy_change_threshold_voxels", 1))
        self._max_cached_revisions = int(
            cfg.get("maximum_cached_revisions", 2))
        self._persist = bool(cfg.get("persist_across_episodes", False))

        # State per episode
        self._last_map_revision = -1
        self._last_esdf_revision = -1
        self._uploaded_revision = -1

        # Cached ESDF data (max revisions)
        self._cached_esdf = collections.OrderedDict()  # revision -> (esdf, known_mask, origin, res)

        # Stats
        self.rebuild_count = 0
        self.rebuild_skip_count = 0
        self.upload_count = 0
        self.upload_skip_count = 0

    @property
    def enabled(self):
        return self._enabled

    def reset(self):
        """Clear all cached state. MUST be called at episode start."""
        self._last_map_revision = -1
        self._last_esdf_revision = -1
        self._uploaded_revision = -1
        self._cached_esdf.clear()
        # Stats persist across episodes for logging

    def should_rebuild(self, current_map_revision):
        """Return True if ESDF should be rebuilt based on map revision."""
        if not self._enabled:
            return True
        if current_map_revision == self._last_map_revision:
            self.rebuild_skip_count += 1
            return False
        return True

    def on_rebuilt(self, map_revision, esdf, known_mask, origin, resolution):
        """Record that ESDF was rebuilt at given map revision."""
        self._last_map_revision = map_revision
        self._last_esdf_revision = map_revision
        self.rebuild_count += 1

        # Cache the result
        key = map_revision
        self._cached_esdf[key] = (esdf.copy(), known_mask.copy(),
                                   np.asarray(origin, dtype=np.float64).copy(),
                                   float(resolution))
        # Prune old revisions
        while len(self._cached_esdf) > self._max_cached_revisions:
            self._cached_esdf.popitem(last=False)

    def should_upload_to_cpp(self):
        """Return True if ESDF should be uploaded to the C++ planner."""
        if not self._enabled or not self._upload_only_on_change:
            return True
        if self._last_esdf_revision == self._uploaded_revision:
            self.upload_skip_count += 1
            return False
        return True

    def on_uploaded(self):
        """Mark current revision as uploaded to C++."""
        self._uploaded_revision = self._last_esdf_revision
        self.upload_count += 1

    def get_cached(self, revision=None):
        """Retrieve a cached ESDF revision. If revision is None, returns latest."""
        if revision is None:
            revision = self._last_esdf_revision
        return self._cached_esdf.get(revision)

    def stats_summary(self):
        return {
            "rebuild_count": self.rebuild_count,
            "rebuild_skip_count": self.rebuild_skip_count,
            "upload_count": self.upload_count,
            "upload_skip_count": self.upload_skip_count,
            "current_revision": self._last_esdf_revision,
        }
