#!/usr/bin/env python3
"""
il_config.py  —  Configuration loader with validation and migration.

Loads YAML config, applies ROS parameter overrides, validates all fields,
and warns about deprecated/unused fields.

Old config structure (il_pipeline.py) is detected and a migration warning
is issued.  The canonical structure is the v2 YAML used by il_manager.py.
"""

from __future__ import print_function, division

import os, sys, math, copy

import rospy
import rospkg

try:
    import yaml
except ImportError:
    yaml = None


# ── Expected config schema (v2) ──────────────────────────────────────────────
REQUIRED_GLOBAL_KEYS = [
    "scene_id", "pub_port", "sub_port", "output_dir",
    "fsm", "depth", "pointcloud", "esdf", "control",
    "obstacle", "start_goal", "planning", "data",
]

REQUIRED_FSM_KEYS = [
    "connect_timeout", "scene_settle_timeout", "pc_export_timeout",
    "esdf_build_timeout", "drone_stable_timeout", "trajectory_timeout",
    "keep_alive_period",
]

REQUIRED_DEPTH_KEYS = ["width", "height", "fov", "max_m", "near", "far"]

REQUIRED_PC_KEYS = ["range", "origin", "resolution"]
REQUIRED_ESDF_KEYS = ["resolution", "drone_radius"]
REQUIRED_CONTROL_KEYS = ["control_hz", "record_hz"]

# Keys that exist in the current YAML but are NOT actually used by
# il_manager.py v5 – we warn about these so the user knows they're dead.
UNUSED_KEYS = {
    "control.settle_time": "Defined but not consumed by the FSM.  Drone settle is managed by WAIT_DRONE_STABLE timeout.",
    "data.save_obstacle_ply": "Defined but PLY is always saved per-scene for ESDF; individual trajectory PLY is not implemented.",
    "data.save_trajectory_meta": "Metadata is always saved; this flag has no effect.",

    # v5 deprecations: these are superseded by the C++ local planner
    "planning.esdf_optimize.enabled": "v5: ESDF smoothing is superseded by C++ local planner. This flag is ignored.",
    "planning.pos_smooth_window": "v5: Position smoothing is superseded by C++ local planner. This parameter is ignored.",
    "planning.resample_spacing": "v5: Path resampling is superseded by C++ local planner time-sampling. This parameter is ignored.",
    "planning.control_lookahead": "v5: Control lookahead is superseded by local planner receding horizon. This parameter is ignored.",
    "planning.control_yaw_smooth": "v5: Yaw smoothing is handled by the local planner. This parameter is ignored.",
    "planning.time_param": "v5: Time parameterization is handled by the C++ local planner. These parameters are inactive.",
}

# v1 → v2 migration map (old il_pipeline.py config keys)
V1_TO_V2_MIGRATION = {
    "global.connect_timeout": "global.fsm.connect_timeout",
    "global.flight.speed": "global.planning.time_param.nominal_speed",
    "global.flight.data_hz": "global.control.record_hz",
    "global.flight.settle_time": "global.control.settle_time (note: unused)",
    "global.flight.keep_alive_period": "global.fsm.keep_alive_period",
    "scene.trajectories": "global.start_goal (automatic pair generation replaces manual trajectories)",
}


def _resolve_path(pkg_rel):
    """Resolve a path relative to the il_dataset package."""
    return os.path.join(rospkg.RosPack().get_path("il_dataset"), pkg_rel)


def _get_nested(d, *keys, default=None):
    """Safely get a nested dict value."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
        else:
            return default
    return d if d != {} else default


def _validate_positive(name, value):
    """Check value > 0."""
    if value <= 0:
        raise ValueError("{} must be > 0, got {}".format(name, value))


def _validate_list_of_n(name, value, n, min_val=None, max_val=None):
    """Check value is a list of n numbers."""
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError("{} must be a list of {} numbers, got {}".format(name, n, value))
    for v in value:
        if not isinstance(v, (int, float)):
            raise ValueError("{} elements must be numeric, got {}".format(name, v))
        if min_val is not None and v < min_val:
            raise ValueError("{} values must be >= {}, got {}".format(name, min_val, v))
        if max_val is not None and v > max_val:
            raise ValueError("{} values must be <= {}, got {}".format(name, max_val, v))


def _validate_zone(name, zone):
    """Validate a start/goal zone dict."""
    for key in ("x", "y"):
        if key not in zone:
            raise ValueError("{} missing '{}'".format(name, key))
        _validate_list_of_n("{}.{}".format(name, key), zone[key], 2)
        if zone[key][0] >= zone[key][1]:
            raise ValueError("{}.{} min >= max: {}".format(name, key, zone[key]))
    for key in ("z_min", "z_max"):
        if key not in zone:
            raise ValueError("{} missing '{}'".format(name, key))
        if not isinstance(zone[key], (int, float)):
            raise ValueError("{}.{} must be numeric".format(name, key))
    if zone["z_min"] >= zone["z_max"]:
        raise ValueError("{}.z_min >= z_max".format(name))


def load_config(config_path=None, validate=True):
    """Load and validate YAML configuration.

    Args:
        config_path:  Path to YAML file.  None = use ROS param ~config_file.
        validate:     If True, run full schema validation.

    Returns:
        Validated config dict.

    Raises:
        IOError: config file not found.
        ImportError: PyYAML not installed (when file is .yaml/.yml).
        ValueError: invalid config values.
    """
    # ── Resolve path ────────────────────────────────────────────
    if config_path is None:
        config_path = rospy.get_param(
            "~config_file",
            _resolve_path("config/il_dataset_config.yaml"))

    if not os.path.isfile(config_path):
        raise IOError("Config file not found: {}".format(config_path))

    # ── Load ────────────────────────────────────────────────────
    ext = os.path.splitext(config_path)[1].lower()
    is_yaml = ext in (".yaml", ".yml")

    if is_yaml:
        if yaml is None:
            raise ImportError(
                "PyYAML is required to parse '{}'. "
                "Install: pip install pyyaml".format(config_path))
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    else:
        import json
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

    if cfg is None:
        raise ValueError("Config file is empty: {}".format(config_path))

    # ── Detect v1 config (old il_pipeline.py format) ────────────
    _warn_v1_migration(cfg)

    # ── ROS parameter overrides ─────────────────────────────────
    _apply_ros_overrides(cfg)

    # ── Set default output_dir ──────────────────────────────────
    if not cfg.get("global", {}).get("output_dir"):
        cfg["global"]["output_dir"] = _resolve_path("dataset/il_data")

    # ── Normalize port types ────────────────────────────────────
    g = cfg["global"]
    g["pub_port"] = str(g["pub_port"])
    g["sub_port"] = str(g["sub_port"])
    g["scene_id"] = int(g["scene_id"])

    # ── Validate ────────────────────────────────────────────────
    if validate:
        _validate_config(cfg)

    # ── Warn about unused keys ──────────────────────────────────
    _warn_unused_keys(cfg)

    return cfg


def _warn_v1_migration(cfg):
    """Detect v1 config structure and warn."""
    g = cfg.get("global", {})
    v1_markers = []

    # v1 uses global.flight instead of global.control
    if "flight" in g:
        v1_markers.append("global.flight (v1) — should be global.control (v2)")

    # v1 uses global.connect_timeout instead of global.fsm.connect_timeout
    if "connect_timeout" in g:
        v1_markers.append("global.connect_timeout (v1) — should be global.fsm.connect_timeout (v2)")

    # v1 scenes use trajectories field
    for s in cfg.get("scenes", []):
        if "trajectories" in s:
            v1_markers.append(
                "scene.{}.trajectories (v1) — manual trajectories replaced by "
                "automatic start/goal pair generation via global.start_goal (v2)".format(
                    s.get("name", "?")))
            break

    if v1_markers:
        rospy.logwarn("=" * 60)
        rospy.logwarn("  CONFIG MIGRATION:  v1 (il_pipeline.py) → v2 (il_manager.py)")
        for m in v1_markers:
            rospy.logwarn("    • %s", m)
        rospy.logwarn("  Please update your config to the v2 schema.")
        rospy.logwarn("  See config/il_dataset_config.yaml for the canonical format.")
        rospy.logwarn("=" * 60)


def _apply_ros_overrides(cfg):
    """Apply ROS parameter overrides to the config.

    Compatible with both v1 and v2 key paths.
    """
    g = cfg["global"]

    # Standard overrides
    for key in ("scene_id", "pub_port", "sub_port"):
        if rospy.has_param("~" + key):
            g[key] = rospy.get_param("~" + key)

    # connect_timeout override → correct path: global.fsm.connect_timeout
    if rospy.has_param("~connect_timeout"):
        val = float(rospy.get_param("~connect_timeout"))
        fsm = g.setdefault("fsm", {})
        fsm["connect_timeout"] = val
        rospy.loginfo("[Config] Override fsm.connect_timeout = %.1f", val)

    # fly_speed override
    if rospy.has_param("~fly_speed"):
        val = float(rospy.get_param("~fly_speed"))
        planning = g.setdefault("planning", {})
        tp = planning.setdefault("time_param", {})
        tp["nominal_speed"] = val
        rospy.loginfo("[Config] Override nominal_speed = %.1f", val)


def _validate_config(cfg):
    """Full schema validation. Raises ValueError on problems."""
    errors = []
    g = cfg.get("global", {})
    if not isinstance(g, dict):
        raise ValueError("'global' section must be a dict")

    # ── Required top-level keys ─────────────────────────────────
    for key in REQUIRED_GLOBAL_KEYS:
        if key not in g:
            errors.append("Missing global.{}".format(key))

    # ── FSM timeouts ────────────────────────────────────────────
    fsm = g.get("fsm", {})
    for key in REQUIRED_FSM_KEYS:
        if key not in fsm:
            errors.append("Missing global.fsm.{}".format(key))
        elif not isinstance(fsm[key], (int, float)) or fsm[key] <= 0:
            errors.append("global.fsm.{} must be positive number, got {}".format(key, fsm[key]))

    # ── Depth ───────────────────────────────────────────────────
    depth = g.get("depth", {})
    for key in REQUIRED_DEPTH_KEYS:
        if key not in depth:
            errors.append("Missing global.depth.{}".format(key))
    if "width" in depth:
        _validate_positive("global.depth.width", depth["width"])
    if "height" in depth:
        _validate_positive("global.depth.height", depth["height"])
    if "fov" in depth:
        _validate_positive("global.depth.fov", depth["fov"])
    if "max_m" in depth:
        _validate_positive("global.depth.max_m", depth["max_m"])

    # ── Pointcloud ──────────────────────────────────────────────
    pc = g.get("pointcloud", {})
    for key in REQUIRED_PC_KEYS:
        if key not in pc:
            errors.append("Missing global.pointcloud.{}".format(key))
    if "range" in pc:
        _validate_list_of_n("global.pointcloud.range", pc["range"], 3, min_val=0.1)
    if "origin" in pc:
        _validate_list_of_n("global.pointcloud.origin", pc["origin"], 3)
    if "resolution" in pc:
        _validate_positive("global.pointcloud.resolution", pc["resolution"])

    # ── ESDF ────────────────────────────────────────────────────
    esdf = g.get("esdf", {})
    for key in REQUIRED_ESDF_KEYS:
        if key not in esdf:
            errors.append("Missing global.esdf.{}".format(key))
    if "resolution" in esdf:
        _validate_positive("global.esdf.resolution", esdf["resolution"])
    if "drone_radius" in esdf:
        _validate_positive("global.esdf.drone_radius", esdf["drone_radius"])

    # ── Control ─────────────────────────────────────────────────
    ctrl = g.get("control", {})
    for key in REQUIRED_CONTROL_KEYS:
        if key not in ctrl:
            errors.append("Missing global.control.{}".format(key))
    if "control_hz" in ctrl:
        _validate_positive("global.control.control_hz", ctrl["control_hz"])
    if "record_hz" in ctrl:
        _validate_positive("global.control.record_hz", ctrl["record_hz"])
    # record_hz should be <= control_hz (or at least a clean divisor)
    if ctrl.get("record_hz", 0) > ctrl.get("control_hz", 0):
        errors.append("global.control.record_hz ({}) > control_hz ({})".format(
            ctrl["record_hz"], ctrl["control_hz"]))

    # ── Data (schema v7) ────────────────────────────────────────
    data = g.get("data", {})
    schema_version = data.get("schema_version", 5)
    if schema_version >= 7:
        # label_lookahead_time_s validation
        lookahead = data.get("label_lookahead_time_s", 0.08)
        if lookahead <= 0:
            errors.append(
                "global.data.label_lookahead_time_s must be > 0, got {}".format(lookahead))

        # trend soft sigma
        sigma = data.get("trend_soft_sigma_bins", 0.75)
        if sigma <= 0:
            errors.append(
                "global.data.trend_soft_sigma_bins must be > 0, got {}".format(sigma))

        # trend bins must be odd > 1
        for bin_name in ("trend_horizontal_bins", "trend_vertical_bins"):
            nb = data.get(bin_name, 1)
            if not isinstance(nb, int) or nb < 3 or nb % 2 == 0:
                errors.append(
                    "global.data.{} must be an odd integer > 1, got {}".format(bin_name, nb))

        # collection_mode validation
        mode = data.get("collection_mode", "deterministic_lockstep")
        accepted_modes = ("deterministic_lockstep", "legacy_async")
        if mode not in accepted_modes:
            errors.append(
                "global.data.collection_mode must be one of {}, got '{}'".format(
                    accepted_modes, mode))

    # ── Observed map (Phase 2) ──────────────────────────────────
    obs_map = g.get("observed_map", {})
    if obs_map.get("enabled", False):
        for key in ("resolution", "size_x_m", "size_y_m", "size_z_m"):
            val = obs_map.get(key, 0)
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("global.observed_map.{} must be > 0, got {}".format(key, val))
        # Grid dimensions should map to reasonable integer counts
        for dim_name, size_key in [("x", "size_x_m"), ("y", "size_y_m"), ("z", "size_z_m")]:
            res = obs_map.get("resolution", 0.1)
            size = obs_map.get(size_key, 12.0)
            if res > 0:
                n = size / res
                if n < 2:
                    errors.append(
                        "global.observed_map.{}/resolution yields <2 voxels in {} dim".format(
                            size_key, dim_name))
        hist = obs_map.get("history_seconds", 4.0)
        if hist < 0:
            errors.append("global.observed_map.history_seconds must be >= 0, got {}".format(hist))
        ratio = obs_map.get("min_known_free_ratio", 0.95)
        if ratio < 0 or ratio > 1:
            errors.append("global.observed_map.min_known_free_ratio must be in [0,1], got {}".format(ratio))

    # ── Guide selector (Phase 2) ─────────────────────────────────
    gs = g.get("guide_selector", {})
    if gs:
        mr = gs.get("max_range_m", 5.0)
        dmax = depth.get("max_m", 5.0)
        if mr > dmax + 1e-9:
            errors.append("guide_selector.max_range_m ({}) > depth.max_m ({})".format(mr, dmax))
        for key in ("corridor_radius_m", "corridor_sample_spacing_m",
                     "terminal_horizon_s", "terminal_acceleration_limit_mps2"):
            val = gs.get(key, 0)
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("global.guide_selector.{} must be > 0, got {}".format(key, val))
        # FOV margins
        h_margin = gs.get("horizontal_fov_margin_deg", 3.0)
        v_margin = gs.get("vertical_fov_margin_deg", 3.0)
        hfov = depth.get("fov", 90.0)
        if h_margin < 0 or h_margin >= hfov / 2.0:
            errors.append("guide_selector.horizontal_fov_margin_deg ({}) must be in [0, {})".format(
                h_margin, hfov / 2.0))
        if v_margin < 0:
            errors.append("guide_selector.vertical_fov_margin_deg must be >= 0")

    # ── Phase 2: local_planner observed ESDF config ──────────────
    lp = g.get("planning", {}).get("local_planner", {})
    if lp:
        use_obs = lp.get("use_observed_esdf", True)
        forbid_unk = lp.get("forbid_unknown_space", True)
        allow_fb = lp.get("allow_global_map_fallback", False)
        if not use_obs:
            errors.append("Phase 2 requires local_planner.use_observed_esdf = true")
        if allow_fb:
            errors.append("Phase 2 requires local_planner.allow_global_map_fallback = false")

    # ── Phase 3: scene_generation validation ────────────────────
    sg_cfg = g.get("scene_generation", {})
    if sg_cfg.get("enabled", False):
        # Basic type checks
        if sg_cfg.get("obstacle_type", "cylinder") != "cylinder":
            errors.append("Phase 3 only supports obstacle_type='cylinder'")
        if sg_cfg.get("outside_obstacle_region_policy", "free") != "free":
            errors.append("outside_obstacle_region_policy must be 'free'")

        # Obstacle region
        oreg = sg_cfg.get("obstacle_region", {})
        for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            if k not in oreg:
                errors.append("Missing global.scene_generation.obstacle_region.{}".format(k))
        if oreg.get("x_min", 0) >= oreg.get("x_max", 0):
            errors.append("obstacle_region.x_min >= x_max")
        if oreg.get("y_min", 0) >= oreg.get("y_max", 0):
            errors.append("obstacle_region.y_min >= y_max")
        if oreg.get("z_min", 0) >= oreg.get("z_max", 0):
            errors.append("obstacle_region.z_min >= z_max")

        # Cylinder params
        cyl = sg_cfg.get("cylinder", {})
        for k in ("radius_min_m", "radius_max_m", "height_min_m", "height_max_m"):
            v = cyl.get(k, 0)
            if v <= 0:
                errors.append("global.scene_generation.cylinder.{} must be > 0".format(k))
        if cyl.get("radius_min_m", 0) > cyl.get("radius_max_m", 0):
            errors.append("cylinder.radius_min_m > radius_max_m")
        if cyl.get("height_min_m", 0) > cyl.get("height_max_m", 0):
            errors.append("cylinder.height_min_m > height_max_m")
        if cyl.get("count_min", 0) > cyl.get("count_max", 0):
            errors.append("cylinder.count_min > count_max")
        if cyl.get("minimum_surface_gap_m", 0) < 0:
            errors.append("cylinder.minimum_surface_gap_m must be >= 0")
        if cyl.get("minimum_inflated_gap_m", 0) < 0:
            errors.append("cylinder.minimum_inflated_gap_m must be >= 0")

        # Topology validation
        topo = sg_cfg.get("topology_validation", {})
        if topo.get("grid_resolution_m", 0) <= 0:
            errors.append("topology_validation.grid_resolution_m must be > 0")
        if topo.get("validation_halo_m", 0) <= 0:
            errors.append("topology_validation.validation_halo_m must be > 0")
        if topo.get("escape_ray_count", 0) < 8:
            errors.append("topology_validation.escape_ray_count must be >= 8")
        if topo.get("minimum_escape_sector_width_deg", 0) <= 0:
            errors.append("topology_validation.minimum_escape_sector_width_deg must be > 0")
        if topo.get("minimum_separated_escape_sectors", 0) < 1:
            errors.append("minimum_separated_escape_sectors must be >= 1")

        # Task generation
        tg = sg_cfg.get("task_generation", {})
        if tg.get("minimum_start_goal_distance_m", 0) > tg.get("maximum_start_goal_distance_m", 0):
            errors.append("task_generation.minimum_start_goal_distance > maximum")
        if tg.get("minimum_detour_ratio", 0) < 1.0:
            errors.append("task_generation.minimum_detour_ratio must be >= 1")
        if tg.get("maximum_detour_ratio", 0) < tg.get("minimum_detour_ratio", 0):
            errors.append("task_generation.maximum_detour_ratio < minimum_detour_ratio")
        if tg.get("maximum_start_goal_height_difference_m", 0) < 0:
            errors.append("maximum_start_goal_height_difference_m must be >= 0")
        if tg.get("minimum_direct_blocker_count", 0) < 0:
            errors.append("minimum_direct_blocker_count must be >= 0")
        if tg.get("maximum_direct_blocker_count", 0) < tg.get("minimum_direct_blocker_count", 0):
            errors.append("maximum_direct_blocker_count < minimum_direct_blocker_count")

        # Side cost
        sc = sg_cfg.get("side_cost", {})
        if sc.get("minimum_cost_difference_ratio", 0) < 0:
            errors.append("side_cost.minimum_cost_difference_ratio must be >= 0")

        # Observability
        obs_audit = sg_cfg.get("observability_audit", {})
        if obs_audit.get("maximum_invalid_frames_before_reject", 0) < 0:
            errors.append("maximum_invalid_frames_before_reject must be >= 0")

        # source check
        if sg_cfg.get("source") == "explicit_yaml":
            layout = sg_cfg.get("layout_file", "")
            if not layout:
                errors.append("explicit_yaml mode requires layout_file")
            elif not os.path.isfile(layout):
                errors.append("layout_file not found: {}".format(layout))

    # ── Obstacle ────────────────────────────────────────────────
    obs = g.get("obstacle", {})
    if "area_x" in obs:
        _validate_list_of_n("global.obstacle.area_x", obs["area_x"], 2)
    if "area_y" in obs:
        _validate_list_of_n("global.obstacle.area_y", obs["area_y"], 2)

    # ── Start/Goal zones ────────────────────────────────────────
    sg = g.get("start_goal", {})
    if "start_zone" in sg:
        _validate_zone("global.start_goal.start_zone", sg["start_zone"])
    if "goal_zone" in sg:
        _validate_zone("global.start_goal.goal_zone", sg["goal_zone"])
    if "num_pairs_per_config" in sg:
        if sg["num_pairs_per_config"] < 1:
            errors.append("global.start_goal.num_pairs_per_config must be >= 1")
    if "candidate_pair_multiplier" in sg:
        value = sg["candidate_pair_multiplier"]
        if not isinstance(value, int) or value < 1:
            errors.append(
                "global.start_goal.candidate_pair_multiplier must be an integer >= 1")
    if "min_esdf_clearance_at_endpoints" in sg:
        _validate_positive("global.start_goal.min_esdf_clearance_at_endpoints",
                           sg["min_esdf_clearance_at_endpoints"])

    # ── Planning (v5: new local_planner validation) ──────────────
    planning = g.get("planning", {})

    # New v5 sections
    gp = planning.get("global_planner", {})
    lp = planning.get("local_planner", {})

    # Keep all geometry parameters on one physical scale.  The point cloud
    # must not be sparser than the ESDF, and obstacle generation must use the
    # same body radius that ESDF inflation subtracts.
    pc_res = float(pc.get("resolution", 0.0) or 0.0)
    esdf_res = float(esdf.get("resolution", 0.0) or 0.0)
    esdf_radius = float(esdf.get("drone_radius", 0.0) or 0.0)
    obstacle_radius = float(obs.get("drone_radius", esdf_radius) or 0.0)
    if pc_res > esdf_res + 1e-9:
        errors.append("pointcloud resolution ({}) > esdf resolution ({}) may create surface holes".format(
            pc_res, esdf_res))
    if esdf_res > 0.5 * esdf_radius + 1e-9:
        errors.append("esdf resolution ({}) must be <= half drone_radius ({})".format(
            esdf_res, esdf_radius))
    if abs(obstacle_radius - esdf_radius) > 1e-9:
        errors.append("obstacle.drone_radius ({}) != esdf.drone_radius ({})".format(
            obstacle_radius, esdf_radius))

    hard_margin = max(float(gp.get("min_clearance", 0.0) or 0.0),
                      float(lp.get("min_clearance", 0.0) or 0.0))
    generated_gap = (float(obs.get("min_gap", 0.0) or 0.0)
                     + 2.0 * obstacle_radius
                     + float(obs.get("safety_margin", 0.0) or 0.0))
    required_gap = 2.0 * (esdf_radius + hard_margin)
    if generated_gap + 1e-9 < required_gap:
        errors.append("generated obstacle surface gap ({}) < planner-required width ({})".format(
            generated_gap, required_gap))
    scale_weights = obs.get("scale_weights", [0.70, 0.25, 0.05])
    if (not isinstance(scale_weights, list) or len(scale_weights) != 3 or
            any(not isinstance(w, (int, float)) or w < 0 for w in scale_weights) or
            sum(scale_weights) <= 0):
        errors.append("global.obstacle.scale_weights must be three non-negative values with positive sum")

    # Validate global_planner
    for key in ("min_clearance", "clearance_target", "shortcut_check_spacing"):
        if key in gp:
            _validate_positive("global.planning.global_planner.{}".format(key), gp[key])
    for key in ("max_planning_time_coarse_s", "max_planning_time_full_s"):
        if key in gp:
            _validate_positive("global.planning.global_planner.{}".format(key), gp[key])

    # Validate local_planner (if present and backend != python_fallback)
    if lp and lp.get("backend") != "python_fallback":
        _validate_positive("global.planning.local_planner.planner_hz",
                           lp.get("planner_hz", 10.0))
        _validate_positive("global.planning.local_planner.horizon_time",
                           lp.get("horizon_time", 2.5))
        _validate_positive("global.planning.local_planner.execute_prefix_time",
                           lp.get("execute_prefix_time", 0.12))
        _validate_positive("global.planning.local_planner.velocity_command_lookahead_time",
                           lp.get("velocity_command_lookahead_time", 0.12))
        _validate_positive("global.planning.local_planner.velocity_tracking_gain",
                           lp.get("velocity_tracking_gain", 1.5))
        _validate_positive("global.planning.local_planner.max_plan_age",
                           lp.get("max_plan_age", 0.50))
        _validate_positive("global.planning.local_planner.lookahead_distance",
                           lp.get("lookahead_distance", 4.0))
        _validate_positive("global.planning.local_planner.min_clearance",
                           lp.get("min_clearance", 0.10))
        _validate_positive("global.planning.local_planner.target_clearance",
                           lp.get("target_clearance", 0.20))
        _validate_positive("global.planning.local_planner.max_velocity",
                           lp.get("max_velocity", 2.5))
        _validate_positive("global.planning.local_planner.max_acceleration",
                           lp.get("max_acceleration", 3.5))
        _validate_positive("global.planning.local_planner.failure_grace_time",
                           lp.get("failure_grace_time", 1.0))

        # Cross-validations
        planner_hz = lp.get("planner_hz", 10.0)
        control_hz = ctrl.get("control_hz", 25.0)
        record_hz_val = ctrl.get("record_hz", 25.0)
        if planner_hz > control_hz:
            errors.append("local_planner.planner_hz ({}) > control.control_hz ({})".format(
                planner_hz, control_hz))
        if record_hz_val > control_hz:
            errors.append("control.record_hz ({}) > control.control_hz ({})".format(
                record_hz_val, control_hz))

        exec_prefix = lp.get("execute_prefix_time", 0.12)
        horizon = lp.get("horizon_time", 2.5)
        if exec_prefix >= horizon:
            errors.append("execute_prefix_time ({}) >= horizon_time ({})".format(
                exec_prefix, horizon))

        min_look = lp.get("min_lookahead_distance", 2.0)
        look = lp.get("lookahead_distance", 4.0)
        max_look = lp.get("max_lookahead_distance", 6.0)
        if not (min_look <= look <= max_look):
            errors.append("min_lookahead ({}) <= lookahead ({}) <= max_lookahead ({}) violated".format(
                min_look, look, max_look))

        min_cl = lp.get("min_clearance", 0.10)
        target_cl = lp.get("target_clearance", 0.20)
        if min_cl > target_cl:
            errors.append("min_clearance ({}) > target_clearance ({})".format(min_cl, target_cl))

        coll_spacing = lp.get("collision_check_spacing", 0.05)
        esdf_res = g.get("esdf", {}).get("resolution", 0.10)
        if coll_spacing > 0.5 * esdf_res + 1e-9:
            errors.append("collision_check_spacing ({}) > half esdf resolution ({})".format(
                coll_spacing, esdf_res))

        cp = lp.get("control_points", 12)
        if cp < 4:
            errors.append("control_points ({}) must be >= 4".format(cp))

        # Validate all positive
        for key in ("planner_hz", "horizon_time", "execute_prefix_time",
                     "velocity_command_lookahead_time", "max_plan_age",
                     "velocity_tracking_gain",
                     "lookahead_distance", "min_lookahead_distance", "max_lookahead_distance",
                     "lookahead_velocity_gain", "local_map_radius", "max_reference_points",
                     "control_points", "max_iterations", "max_cost_samples_per_segment",
                     "min_clearance", "target_clearance", "collision_check_spacing",
                     "nominal_speed", "max_velocity", "max_acceleration", "max_jerk",
                     "max_yaw_rate", "goal_tolerance", "goal_speed_tolerance",
                     "max_consecutive_failures", "failure_grace_time"):
            if key in lp:
                val = lp[key]
                if isinstance(val, (int, float)) and val <= 0:
                    errors.append("local_planner.{} must be > 0, got {}".format(key, val))

    # ── Legacy time_param validation (deprecated but still in file) ──
    tp = planning.get("time_param", {})
    for key in ("nominal_speed", "max_velocity", "max_acceleration",
                "max_yaw_rate", "min_obstacle_clearance"):
        if key in tp:
            _validate_positive("global.planning.time_param.{}".format(key), tp[key])

    # ── Check start/goal zones are within pointcloud range ──────
    if "range" in pc and "origin" in pc:
        rng = pc["range"]
        org = pc["origin"]
        x_min = org[0] - rng[0] / 2.0
        x_max = org[0] + rng[0] / 2.0
        y_min = org[1] - rng[1] / 2.0
        y_max = org[1] + rng[1] / 2.0
        z_min = org[2] - rng[2] / 2.0
        z_max = org[2] + rng[2] / 2.0

        for zone_name, zone in [("start_zone", sg.get("start_zone", {})),
                                 ("goal_zone", sg.get("goal_zone", {}))]:
            if not zone:
                continue
            if zone.get("x", [0, 0])[0] < x_min or zone.get("x", [0, 0])[1] > x_max:
                errors.append("global.start_goal.{} x-range {} outside pointcloud range [{:.1f}, {:.1f}]".format(
                    zone_name, zone.get("x"), x_min, x_max))
            if zone.get("y", [0, 0])[0] < y_min or zone.get("y", [0, 0])[1] > y_max:
                errors.append("global.start_goal.{} y-range {} outside pointcloud range [{:.1f}, {:.1f}]".format(
                    zone_name, zone.get("y"), y_min, y_max))
            if zone.get("z_min", 0) < z_min or zone.get("z_max", 0) > z_max:
                errors.append("global.start_goal.{} z-range [{}, {}] outside pointcloud range [{:.1f}, {:.1f}]".format(
                    zone_name, zone.get("z_min", 0), zone.get("z_max", 0), z_min, z_max))

    # ── Scenes ──────────────────────────────────────────────────
    scenes = cfg.get("scenes", [])
    if not scenes:
        errors.append("No scenes defined")
    for i, s in enumerate(scenes):
        if "name" not in s:
            errors.append("scenes[{}] missing 'name'".format(i))
        if "seeds" not in s:
            errors.append("scenes[{}] missing 'seeds'".format(i))
        elif not isinstance(s["seeds"], list) or len(s["seeds"]) == 0:
            errors.append("scenes[{}].seeds must be non-empty list".format(i))
        for key in ("target_occupancy", "radius_min", "radius_max"):
            if key not in s:
                errors.append("scenes[{}] missing '{}'".format(i, key))
        if s.get("radius_min", 0) > s.get("radius_max", 0):
            errors.append("scenes[{}].radius_min > radius_max".format(i))
        count_min = s.get("obstacle_count_min", 0)
        count_max = s.get("obstacle_count_max", 0)
        if (not isinstance(count_min, int) or not isinstance(count_max, int) or
                count_min < 0 or count_max < count_min):
            errors.append(
                "scenes[{}] obstacle_count range must satisfy 0 <= min <= max".format(i))

    if errors:
        msg = "Config validation failed with {} error(s):\n  • ".format(len(errors))
        msg += "\n  • ".join(errors)
        raise ValueError(msg)


def _warn_unused_keys(cfg):
    """Warn about config keys that are declared but not used."""
    for key_path, msg in UNUSED_KEYS.items():
        parts = key_path.split(".")
        val = cfg
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p)
            else:
                val = None
                break
        if val is not None:
            rospy.logwarn("[Config] UNUSED KEY '%s' — %s", key_path, msg)
