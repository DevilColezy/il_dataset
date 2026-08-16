#!/usr/bin/env python3
"""
il_config.py  -  Configuration loading / validation for the NEW
hierarchical local-expert IL dataset collector (both
il_dataset_collect.launch and il_dataset_joint_v2_collect.launch).

The production runtime uses exactly one expert parameter source:
  global.hierarchical_expert
plus the expert-independent infrastructure (fsm, depth, vehicle, control,
dynamics, navigation, scene_generation, task_generation, dataset_logging,
sync, commit).

The old expert modules (micro_detour_planner, goal_switching,
trajectory_controller, execution_safety, task_qualifier, task_oracle,
pointcloud, observed_map, macro_* / local_* / trajectory_optimization /
yaw_planning / privileged_intervention) were REMOVED entirely: their
sources, tests, old config recipes and builder functions are gone, they are
not required and not validated, and the production manager never calls them.
"""

from __future__ import print_function, division

import copy
import math
import os

import yaml

import rospkg
import rospy

_PKG = "il_dataset"
_DEFAULT_CONFIG_REL = "config/il_dataset_config.yaml"

# Modules that must exist under `global` (single production path).
REQUIRED_MODULES = [
    "fsm", "depth", "vehicle", "control", "dynamics", "navigation",
    "scene_generation", "task_generation", "hierarchical_expert",
    "dataset_logging", "sync", "commit",
]

# The ONE default Unity T_BC (camera->body, 16 floats row-major).  This is
# the single default used when a config omits `depth.t_bc`; il_common.
# make_depth_vehicle() reads the SAME validated matrix and il_expert_config
# feeds it into the C++ CameraRig2D — there is no second default anywhere.
DEFAULT_T_BC = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.3,
    0.0, 0.0, 0.0, 1.0,
]


def _resolve_path(cfg_path):
    if cfg_path:
        requested = os.path.abspath(os.path.expanduser(str(cfg_path)))
        if not os.path.isfile(requested):
            # A launch-supplied config must never silently fall back to the
            # default dataset recipe: that can erase the wrong output root.
            raise ValueError("Config file not found: %s" % requested)
        return requested
    try:
        rospack = rospkg.RosPack()
        pkg_dir = rospack.get_path(_PKG)
        return os.path.join(pkg_dir, _DEFAULT_CONFIG_REL)
    except Exception:
        # Fall back to a path relative to this file (scripts/..).
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(here, "..", _DEFAULT_CONFIG_REL)
        return os.path.normpath(candidate)


def _deep_merge_config(base, override):
    """Recursively merge a small override YAML onto a complete base YAML.

    Scene profiles are a named list, not a positional list: an override such
    as ``- name: dense_small; scene_count: 1`` changes only that profile and
    retains its geometry/distribution fields from the base configuration.
    Other lists deliberately replace their base value in full.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = _deep_merge_config(merged[key], value) \
                if key in merged else copy.deepcopy(value)
        return merged
    if isinstance(base, list) and isinstance(override, list) and \
            all(isinstance(v, dict) and "name" in v for v in base) and \
            all(isinstance(v, dict) and "name" in v for v in override):
        patches = {str(value["name"]): value for value in override}
        merged = []
        seen = set()
        for value in base:
            name = str(value["name"])
            merged.append(_deep_merge_config(value, patches[name])
                          if name in patches else copy.deepcopy(value))
            seen.add(name)
        merged.extend(copy.deepcopy(value) for value in override
                      if str(value["name"]) not in seen)
        return merged
    return copy.deepcopy(override)


def _load_yaml_with_extends(path, ancestry=None):
    """Load YAML, resolving an optional relative ``extends`` parent."""
    path = os.path.abspath(path)
    ancestry = list(ancestry or [])
    if path in ancestry:
        raise ValueError("circular config extends: %s" % " -> ".join(
            ancestry + [path]))
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping: %s" % path)
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    if not isinstance(parent, str) or not parent:
        raise ValueError("config extends must be a non-empty relative path")
    parent_path = os.path.normpath(os.path.join(os.path.dirname(path), parent))
    if not os.path.isfile(parent_path):
        raise ValueError("base config not found: %s" % parent_path)
    base = _load_yaml_with_extends(parent_path, ancestry + [path])
    return _deep_merge_config(base, cfg)


def _inject_defaults(cfg):
    """Fill the UNIQUE default for `global.depth.t_bc` when the config
    omits it (item 十: single default, never a second copy on the Unity
    wire side).  ``make_depth_vehicle`` and the C++ camera rig both read
    the same `depth.t_bc` afterwards."""
    global_cfg = cfg.get("global", {})
    depth = global_cfg.get("depth")
    if depth is not None and not isinstance(depth, dict):
        return
    if depth is None:
        global_cfg["depth"] = {}
        depth = global_cfg["depth"]
    if depth.get("t_bc") is None:
        depth["t_bc"] = list(DEFAULT_T_BC)


def _positive(value, name, errors, allow_zero=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append("%s must be a number" % name)
        return
    if allow_zero:
        if value < 0.0:
            errors.append("%s must be >= 0" % name)
    elif value <= 0.0:
        errors.append("%s must be > 0" % name)


def _bounded(value, name, lo, hi, errors):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append("%s must be a number" % name)
        return
    if not (lo <= value <= hi):
        errors.append("%s must be in [%s, %s]" % (name, lo, hi))


def _pos_list(value, name, errors):
    if not isinstance(value, list) or not value or \
            any(not isinstance(v, (int, float)) or isinstance(v, bool)
                or float(v) <= 0.0 for v in value):
        errors.append("%s must be a non-empty list of positive numbers"
                      % name)


def _validate_config(cfg):
    """Validate the NEW hierarchical local-expert configuration (the single
    production path used by both launches).

    Only the expert-independent infrastructure (fsm / depth / vehicle /
    control / dynamics / navigation / scene_generation / task_generation /
    dataset_logging / sync / commit) plus the ONE expert source
    `global.hierarchical_expert` are validated.  The old expert modules
    (micro_detour_planner, goal_switching, trajectory_controller,
    execution_safety, task_qualifier, task_oracle, pointcloud, ...) were
    REMOVED entirely and are not required and not validated.
    """
    errors = []
    g = cfg.get("global", {})
    for module in REQUIRED_MODULES:
        if not isinstance(g.get(module), dict):
            errors.append("missing global.%s module" % module)

    # ── fsm (lifecycle timeouts) ──────────────────────────────────
    _positive(g.get("fsm", {}).get("connect_timeout", 60.0),
              "global.fsm.connect_timeout", errors)
    # trajectory_wall_timeout_s == 0 DISABLES the production flight wall
    # timeout (episodes end only on an expert terminal state or shutdown).
    _positive(g.get("fsm", {}).get("trajectory_wall_timeout_s", 600.0),
              "global.fsm.trajectory_wall_timeout_s", errors, allow_zero=True)
    _positive(g.get("fsm", {}).get("depth_warmup_timeout", 30.0),
              "global.fsm.depth_warmup_timeout", errors)

    # ── depth (fixed camera; R is the single perception range) ────
    depth = g.get("depth", {})
    _positive(depth.get("width", 640), "global.depth.width", errors)
    _positive(depth.get("height", 480), "global.depth.height", errors)
    _positive(depth.get("fov", 90.0), "global.depth.fov", errors)
    _positive(depth.get("max_m", 5.0), "global.depth.max_m", errors)
    depth_max = float(depth.get("max_m", 5.0))
    depth_fov = float(depth.get("fov", 90.0))
    # ── depth.t_bc (item 十): the SINGLE camera->body matrix shared by the
    #    Unity wire (make_depth_vehicle) and the C++ CameraRig2D.  It must
    #    be 16 finite row-major floats, the last row must be (0,0,0,1) and
    #    the 3x3 rotation must be a proper rotation (orthonormal, det +1).
    t_bc = depth.get("t_bc")
    if t_bc is not None:
        if (not isinstance(t_bc, (list, tuple)) or len(t_bc) != 16 or
                any(not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not math.isfinite(float(v)) for v in t_bc)):
            errors.append("global.depth.t_bc must be 16 finite numbers "
                          "(row-major 4x4, camera->body)")
        else:
            m = [[float(t_bc[r * 4 + c]) for c in range(4)] for r in range(4)]
            last = m[3]
            if any(abs(last[c] - (1.0 if c == 3 else 0.0)) > 1e-6
                   for c in range(4)):
                errors.append("global.depth.t_bc last row must be "
                              "[0, 0, 0, 1] (affine 4x4)")
            r00, r01, r02 = m[0][0], m[0][1], m[0][2]
            r10, r11, r12 = m[1][0], m[1][1], m[1][2]
            r20, r21, r22 = m[2][0], m[2][1], m[2][2]
            rows = [(r00, r01, r02), (r10, r11, r12), (r20, r21, r22)]
            cols = [(r00, r10, r20), (r01, r11, r21), (r02, r12, r22)]
            ortho_ok = all(
                abs(sum(a * b for a, b in zip(rows[i], rows[j])) -
                    (1.0 if i == j else 0.0)) < 1e-6
                for i in range(3) for j in range(3))
            det = (r00 * (r11 * r22 - r12 * r21) -
                   r01 * (r10 * r22 - r12 * r20) +
                   r02 * (r10 * r21 - r11 * r20))
            if not ortho_ok:
                errors.append("global.depth.t_bc 3x3 rotation must be "
                              "orthonormal (identity for the default rig)")
            if ortho_ok and abs(det - 1.0) > 1e-6:
                errors.append("global.depth.t_bc 3x3 rotation must have "
                              "determinant +1 (proper rotation)")

    # ── vehicle / navigation (truth audit geometry) ────────────────
    veh = g.get("vehicle", {})
    nav = g.get("navigation", {})
    _positive(veh.get("radius_m", 0.30), "global.vehicle.radius_m", errors)
    _positive(nav.get("clearance_m", 0.30), "global.navigation.clearance_m",
              errors)

    # ── control / dynamics (single control backend) ────────────────
    ctrl = g.get("control", {})
    record_hz = float(ctrl.get("record_hz", 30.0))
    _positive(record_hz, "global.control.record_hz", errors)
    dyn = g.get("dynamics", {})
    if dyn.get("backend", "flightmare") != "flightmare":
        errors.append("global.dynamics.backend must be 'flightmare'")
    if dyn.get("control_mode", "velocity_yaw_rate") != "velocity_yaw_rate":
        errors.append("global.dynamics.control_mode must be velocity_yaw_rate")
    _positive(dyn.get("simulation_hz", 200.0),
              "global.dynamics.simulation_hz", errors)
    _positive(dyn.get("control_hz", 50.0),
              "global.dynamics.control_hz", errors)
    if float(dyn.get("simulation_hz", 200.0)) < \
            float(dyn.get("control_hz", 50.0)):
        errors.append("dynamics.simulation_hz must be >= control_hz")
    vc = (dyn.get("velocity_controller", {}) or {})
    backend_yaw_rate = float(vc.get("maximum_yaw_rate_rps", 1.5))
    backend_yaw_accel = float(vc.get("maximum_yaw_acceleration_rps2", 4.0))
    if not bool(vc.get("use_existing_flightmare_controller", False)):
        errors.append(
            "global.dynamics.velocity_controller."
            "use_existing_flightmare_controller must be true")

    # ── scene / task generation (truth, expert-independent) ────────
    sg = g.get("scene_generation", {}) or {}
    _positive(sg.get("seed", 12345), "scene_generation.seed", errors)
    _positive(sg.get("scene_count", 10), "scene_generation.scene_count",
              errors)
    _positive(sg.get("tasks_per_scene", 8),
              "scene_generation.tasks_per_scene", errors, allow_zero=True)
    _positive(sg.get("minimum_tasks_per_scene", 6),
              "scene_generation.minimum_tasks_per_scene", errors,
              allow_zero=True)
    if int(sg.get("minimum_tasks_per_scene", 6)) > \
            int(sg.get("tasks_per_scene", 8)):
        errors.append("scene_generation.minimum_tasks_per_scene must be "
                      "<= tasks_per_scene")
    full_strata = bool(sg.get("require_full_strata_coverage", True))
    if full_strata and int(sg.get("scene_count", 10)) < 10:
        errors.append(
            "scene_generation.scene_count must be >= 10 when "
            "require_full_strata_coverage is true (scene 0 = explicit empty "
            "CLEAR scene + 9 non-empty sparse/medium/dense x small/medium/"
            "large strata)")
    scene_key_prefix = str(sg.get("scene_key_prefix", ""))
    if scene_key_prefix and any(
            not (char.isalnum() or char in "_-")
            for char in scene_key_prefix):
        errors.append("scene_generation.scene_key_prefix may contain only "
                      "letters, digits, '_' and '-'")
    sg_geometry = sg.get("geometry", {}) or {}
    _positive(sg_geometry.get("minimum_surface_gap_m", 1.2),
              "scene_generation.geometry.minimum_surface_gap_m", errors)
    _positive(sg_geometry.get("boundary_margin_m", 1.2),
              "scene_generation.geometry.boundary_margin_m", errors,
              allow_zero=True)
    _positive(sg_geometry.get("radius_min_m", 0.10),
              "scene_generation.geometry.radius_min_m", errors)
    _positive(sg_geometry.get("radius_max_m", 2.0),
              "scene_generation.geometry.radius_max_m", errors)
    if float(sg_geometry.get("radius_max_m", 2.0)) < \
            float(sg_geometry.get("radius_min_m", 0.10)) - 1e-9:
        errors.append("scene_generation.geometry.radius_max_m must be "
                      ">= radius_min_m")
    # ESDF free-cell semantics: dist = drone CENTRE -> obstacle SURFACE.
    # A free cell needs dist > free_cell_surface_clearance_m, which must
    # itself be >= vehicle radius (body-edge clearance = this - radius).
    _positive(sg_geometry.get("free_cell_surface_clearance_m", 0.5),
              "scene_generation.geometry.free_cell_surface_clearance_m",
              errors)
    if float(sg_geometry.get("free_cell_surface_clearance_m", 0.5)) < \
            float(veh.get("radius_m", 0.30)) - 1e-9:
        errors.append(
            "scene_generation.geometry.free_cell_surface_clearance_m "
            "must be >= vehicle.radius_m (drone centre->surface)")
    _bounded(sg_geometry.get("esdf_resolution_m", 0.1),
             "scene_generation.geometry.esdf_resolution_m", 0.02, 0.5,
             errors)
    required_surface_gap = 2.0 * (
        float(veh.get("radius_m", 0.30)) + float(nav.get("clearance_m", 0.30)))
    if float(sg_geometry.get("minimum_surface_gap_m", 1.2)) < \
            required_surface_gap - 1e-9:
        errors.append(
            "scene_generation.geometry.minimum_surface_gap_m must be "
            ">= %.3f m = 2 * (vehicle radius + clearance)" %
            required_surface_gap)

    tg = g.get("task_generation", {}) or {}
    _positive(tg.get("flight_height_m", 2.0),
              "task_generation.flight_height_m", errors)
    _positive(tg.get("min_task_distance_m", 4.0),
              "task_generation.min_task_distance_m", errors)
    _positive(tg.get("max_task_distance_m", 20.0),
              "task_generation.max_task_distance_m", errors)
    if float(tg.get("min_task_distance_m", 4.0)) > \
            float(tg.get("max_task_distance_m", 20.0)):
        errors.append("task_generation.min_task_distance_m must be <= "
                      "max_task_distance_m")
    _positive(tg.get("initial_yaw_bias_deg", 15.0),
              "task_generation.initial_yaw_bias_deg", errors, allow_zero=True)
    _positive(tg.get("preflight_qualification_max_ticks", 1800),
              "task_generation.preflight_qualification_max_ticks", errors)
    # Oversampling / hard-quota budget (blueprint-only, never a flight
    # timeout).  Finite bounds prevent silent under-generation.
    _positive(tg.get("task_sample_attempts", 200),
              "task_generation.task_sample_attempts", errors)
    _positive(tg.get("candidate_pool_multiplier", 4),
              "task_generation.candidate_pool_multiplier", errors)
    _positive(tg.get("qualification_attempt_budget", 400),
              "task_generation.qualification_attempt_budget", errors)
    quotas = tg.get("quotas", {}) or {}
    for qkey in ("min_per_behavior", "min_turn_per_side",
                 "max_left_right_imbalance", "long_takeover_min_ticks",
                 "min_per_density_level", "min_per_radius_level",
                 "min_per_distance_level"):
        _positive(quotas.get(qkey, 2),
                  "task_generation.quotas.%s" % qkey, errors,
                  allow_zero=True)
    for qkey in ("distance_short_max_m", "distance_long_min_m",
                 "radius_small_max_m", "radius_large_min_m",
                 "density_sparse_max", "density_dense_min"):
        _positive(quotas.get(qkey, 1.0),
                  "task_generation.quotas.%s" % qkey, errors,
                  allow_zero=True)
    # Quota band ordering: the short/small/sparse bands must sit strictly
    # below the long/large/dense bands (else the strata are degenerate).
    if float(quotas.get("distance_short_max_m", 9.0)) >= \
            float(quotas.get("distance_long_min_m", 15.0)):
        errors.append("task_generation.quotas.distance_short_max_m must "
                      "be < distance_long_min_m")
    if float(quotas.get("radius_small_max_m", 0.6)) >= \
            float(quotas.get("radius_large_min_m", 1.4)):
        errors.append("task_generation.quotas.radius_small_max_m must "
                      "be < radius_large_min_m")
    if float(quotas.get("density_sparse_max", 7.0)) >= \
            float(quotas.get("density_dense_min", 14.0)):
        errors.append("task_generation.quotas.density_sparse_max must "
                      "be < density_dense_min")

    # ── blueprint_generation: deficit-driven offline pipeline ──────
    bp = g.get("blueprint_generation", {}) or {}
    if bool(bp.get("enabled", True)):
        # Warehouse free region is THE single source: when present it must
        # equal hierarchical_expert.region (the expert grid anchor) so the
        # blueprint, preflight and runtime never disagree on coordinates.
        bp_wh = bp.get("warehouse", {}) or {}
        fr = bp_wh.get("free_region", None)
        he_region = (g.get("hierarchical_expert", {}) or {}).get(
            "region", {}) or {}
        if fr is not None:
            if not isinstance(fr, (list, tuple)) or len(fr) < 4 or \
                    not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in fr):
                errors.append(
                    "blueprint_generation.warehouse.free_region must be "
                    "[min_x, max_x, min_y, max_y]")
            else:
                if abs(float(fr[0]) - float(he_region.get("min_x", 0.0))) > 1e-6 or \
                        abs(float(fr[1]) - float(he_region.get("max_x", 0.0))) > 1e-6 or \
                        abs(float(fr[2]) - float(he_region.get("min_y", 0.0))) > 1e-6 or \
                        abs(float(fr[3]) - float(he_region.get("max_y", 0.0))) > 1e-6:
                    errors.append(
                        "blueprint_generation.warehouse.free_region must "
                        "equal hierarchical_expert.region (single source "
                        "of the warehouse free region)")
        _bounded(bp_wh.get("wall_extension_m", 1.0),
                 "blueprint_generation.warehouse.wall_extension_m",
                 0.0, 100.0, errors)

        bsg = bp.get("scene_generation", {}) or {}
        _positive(bsg.get("min_surface_gap_m", 1.40),
                  "blueprint_generation.scene_generation.min_surface_gap_m",
                  errors)
        _positive(bsg.get("free_cell_surface_clearance_m", 0.5),
                  "blueprint_generation.scene_generation."
                  "free_cell_surface_clearance_m", errors)
        # planner-required traversable passage:
        #   2 * (vehicle radius + navigation clearance + discretisation
        #   margin) + 2 * generation margin
        req_passage = 2.0 * (
            float(veh.get("radius_m", 0.30)) +
            float(nav.get("clearance_m", 0.30)) +
            float(bsg.get("clearance_discretization_margin_m", 0.05)) +
            float(bsg.get("generation_margin_m", 0.05)))
        if float(bsg.get("min_surface_gap_m", 1.40)) < req_passage - 1e-9:
            errors.append(
                "blueprint_generation.scene_generation.min_surface_gap_m "
                "must be >= %.3f m (planner-required traversable passage)"
                % req_passage)

        btg = bp.get("task_generation", {}) or {}
        _positive(btg.get("min_task_distance_m", 4.0),
                  "blueprint_generation.task_generation.min_task_distance_m",
                  errors)
        _positive(btg.get("max_task_distance_m", 20.0),
                  "blueprint_generation.task_generation.max_task_distance_m",
                  errors)
        if float(btg.get("min_task_distance_m", 4.0)) > \
                float(btg.get("max_task_distance_m", 20.0)):
            errors.append("blueprint_generation.task_generation."
                          "min_task_distance_m must be <= max_task_distance_m")

        # layered initial yaw: strictly increasing edges, non-negative
        # weights, same bin count.
        yaw = btg.get("initial_yaw", {}) or {}
        yedges = yaw.get("edges_deg", [0.0, 15.0, 35.0, 55.0, 90.0, 150.0, 180.0])
        yw = yaw.get("weights", [0.8, 1.2, 2.2, 1.6, 1.0, 0.9])
        if isinstance(yedges, (list, tuple)) and len(yedges) >= 2 and \
                all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in yedges):
            if any(yedges[i] >= yedges[i + 1] for i in range(len(yedges) - 1)):
                errors.append("blueprint_generation.task_generation."
                              "initial_yaw.edges_deg must be strictly "
                              "increasing")
            if isinstance(yw, (list, tuple)) and len(yw) != len(yedges) - 1:
                errors.append("blueprint_generation.task_generation."
                              "initial_yaw.weights must have length "
                              "len(edges_deg) - 1")
        if isinstance(yw, (list, tuple)) and \
                any(not isinstance(v, (int, float)) or isinstance(v, bool)
                    or float(v) < 0.0 for v in yw):
            errors.append("blueprint_generation.task_generation."
                          "initial_yaw.weights must be non-negative numbers")

        dp = btg.get("depth_proxy", {}) or {}
        near_m = float(dp.get("near_max_m", 1.5))
        mid_m = float(dp.get("mid_max_m", 3.0))
        far_m = float(dp.get("far_max_m", 5.0))
        if not (0.0 < near_m < mid_m < far_m):
            errors.append("blueprint_generation.task_generation.depth_proxy "
                          "thresholds must satisfy 0 < near_max_m < "
                          "mid_max_m < far_max_m")

        perf = bp.get("performance", {}) or {}
        for pk in ("max_scene_candidates", "max_task_candidates_per_scene",
                   "max_generation_rounds", "max_total_preflight_tasks",
                   "max_total_preflight_ticks",
                   "max_preflight_ticks_per_task",
                   "max_scene_generation_attempts",
                   "max_task_generation_attempts"):
            _positive(perf.get(pk, 1),
                      "blueprint_generation.performance.%s" % pk, errors)
        if bool(perf.get("parallel_tasks", False)):
            errors.append(
                "blueprint_generation.performance.parallel_tasks is not "
                "implemented; it must be false (set it to false or remove "
                "the key)")

        req = bp.get("requirements", {}) or {}
        _positive(req.get("min_selected_scenes", 4),
                  "blueprint_generation.requirements.min_selected_scenes",
                  errors)
        if int(req.get("min_selected_scenes", 4)) > \
                int(req.get("min_scenes", 4)):
            errors.append(
                "blueprint_generation.requirements.min_selected_scenes must "
                "be <= min_scenes")
        _positive(req.get("min_grouped_deflection_samples", 8),
                  "blueprint_generation.requirements."
                  "min_grouped_deflection_samples", errors)
        _positive(req.get("min_grouped_correction_samples", 4),
                  "blueprint_generation.requirements."
                  "min_grouped_correction_samples", errors)

        et = bp.get("early_termination", {}) or {}
        # 0 = the no-progress watchdog is DISABLED (valid, intended
        # production default).  Only negative values are rejected.
        _positive(et.get("no_progress_window_ticks", 0),
                  "blueprint_generation.early_termination."
                  "no_progress_window_ticks", errors, allow_zero=True)
        _positive(et.get("stall_window_ticks", 90),
                  "blueprint_generation.early_termination.stall_window_ticks",
                  errors)
        _positive(et.get("min_chicane_alternations", 2),
                  "blueprint_generation.early_termination."
                  "min_chicane_alternations", errors)

        # ── privileged task qualification (2D causal-qualification port) ─
        tq = bp.get("task_qualification", {}) or {}
        _positive(tq.get("max_astar_expansions", 30000),
                  "blueprint_generation.task_qualification."
                  "max_astar_expansions", errors)
        _positive(tq.get("max_side_route_expansions", 20000),
                  "blueprint_generation.task_qualification."
                  "max_side_route_expansions", errors)
        _positive(tq.get("max_total_side_route_expansions", 120000),
                  "blueprint_generation.task_qualification."
                  "max_total_side_route_expansions", errors)
        _positive(tq.get("max_total_qualification_expansions", 400000),
                  "blueprint_generation.task_qualification."
                  "max_total_qualification_expansions", errors)
        _positive(tq.get("start_recovery_max_radius_m", 0.5),
                  "blueprint_generation.task_qualification."
                  "start_recovery_max_radius_m", errors)
        if float(tq.get("side_bias", 0.4)) < 0.0:
            errors.append("blueprint_generation.task_qualification.side_bias "
                          "must be >= 0")
        if float(tq.get("min_route_stretch_for_long_detour", 1.5)) < 1.0:
            errors.append("blueprint_generation.task_qualification."
                          "min_route_stretch_for_long_detour must be >= 1.0")

        # Preflight control rate (Hz): the production tick grid is FIXED at
        # 30 Hz (dt = 1/30 s is the preflight/stall time base everywhere).
        bp_rate = float(bp.get("control_rate_hz", 30.0))
        he_rate = float((g.get("hierarchical_expert", {}) or {}).get(
            "control_hz", 30.0))
        if abs(bp_rate - 30.0) > 1e-6:
            errors.append(
                "blueprint_generation.control_rate_hz must be exactly "
                "30.0 Hz (production tick grid); got %g" % bp_rate)
        if abs(he_rate - 30.0) > 1e-6:
            errors.append(
                "hierarchical_expert.control_hz must be exactly 30.0 Hz "
                "(production tick grid); got %g" % he_rate)
        if abs(bp_rate - he_rate) > 1e-6:
            errors.append(
                "blueprint_generation.control_rate_hz must equal "
                "hierarchical_expert.control_hz (%g)" % he_rate)

        # use_profile_catalog=false must NOT silently produce zero scenes:
        # it requires explicit user profiles (the C++ controller also fails
        # fast; this surfaces the error at config-load time too).  Profiles
        # live under blueprint_generation.scene_generation.profiles.
        bsg2 = bp.get("scene_generation", {}) or {}
        if not bool(bsg2.get("use_profile_catalog", True)) and \
                not (bsg2.get("profiles") or []):
            errors.append(
                "blueprint_generation.scene_generation.use_profile_catalog "
                "is false but scene_generation.profiles is empty; either "
                "set use_profile_catalog to true or provide a non-empty "
                "profiles list")

    # ── hierarchical_expert: THE single expert parameter source ────
    he = g.get("hierarchical_expert", {}) or {}
    control_hz = float(he.get("control_hz", 30.0))
    macro_hz = float(he.get("macro_update_hz", 5.0))
    _positive(control_hz, "hierarchical_expert.control_hz", errors)
    _positive(macro_hz, "hierarchical_expert.macro_update_hz", errors)
    if abs(record_hz - control_hz) > 1e-6:
        errors.append("hierarchical_expert.control_hz must equal "
                      "control.record_hz (%g)" % record_hz)
    if macro_hz <= 0.0 or \
            abs(control_hz / macro_hz - round(control_hz / macro_hz)) > 1e-6:
        errors.append("hierarchical_expert.macro_update_hz must divide "
                      "control_hz exactly (5 Hz on a 30 Hz tick grid)")

    region = he.get("region", {}) or {}
    if float(region.get("max_x", 0.0)) <= float(region.get("min_x", 0.0)) or \
            float(region.get("max_y", 0.0)) <= float(region.get("min_y", 0.0)):
        errors.append("hierarchical_expert.region max bounds must exceed "
                      "min bounds")
    # The region (also the ESDF / start-goal domain) must fit the largest
    # cylinder plus its boundary margin on BOTH axes.
    fit_need = 2.0 * (float(sg_geometry.get("radius_max_m", 2.0)) +
                      float(sg_geometry.get("boundary_margin_m", 1.2)))
    region_w = float(region.get("max_x", 0.0)) - float(region.get("min_x", 0.0))
    region_h = float(region.get("max_y", 0.0)) - float(region.get("min_y", 0.0))
    if region_w < fit_need - 1e-9 or region_h < fit_need - 1e-9:
        errors.append(
            "hierarchical_expert.region must fit the largest obstacle: "
            "both extents >= %.3f m = 2 * (radius_max + boundary_margin)"
            % fit_need)

    obs = he.get("observation", {}) or {}
    fov = float(obs.get("fov_deg", 90.0))
    perception = float(obs.get("range_m", 5.0))
    _bounded(fov, "hierarchical_expert.observation.fov_deg", 1.0, 180.0,
             errors)
    _positive(perception, "hierarchical_expert.observation.range_m", errors)
    _positive(obs.get("resolution", 0.1),
              "hierarchical_expert.observation.resolution", errors)
    _positive(obs.get("ray_angular_res_deg", 0.5),
              "hierarchical_expert.observation.ray_angular_res_deg", errors)
    # Item 十: depth.fov and observation.fov_deg are the SAME camera FOV
    # (Unity wire + expert), and depth.max_m must cover observation.range_m.
    if abs(depth_fov - fov) > 1e-6:
        errors.append(
            "global.depth.fov must equal "
            "hierarchical_expert.observation.fov_deg (single camera source)")
    if perception > depth_max + 1e-6:
        errors.append("hierarchical_expert.observation.range_m must be "
                      "<= depth.max_m")

    te = he.get("target_encoding", {}) or {}
    bin_count = int(te.get("direction_bin_count", 11))
    reserve = float(te.get("normal_distance_reserve_m", 0.5))
    margin_deg = float(te.get("turn_ray_margin_deg", 10.0))
    if bin_count < 3 or bin_count % 2 == 0:
        errors.append("hierarchical_expert.target_encoding."
                      "direction_bin_count must be odd and >= 3")
    if not (0.0 <= reserve < perception):
        errors.append("hierarchical_expert.target_encoding."
                      "normal_distance_reserve_m must be in [0, range_m)")
    if not (0.0 <= margin_deg < fov / 2.0):
        errors.append("hierarchical_expert.target_encoding."
                      "turn_ray_margin_deg must be in [0, fov_deg/2)")

    lp = he.get("local_planner", {}) or {}
    _positive(lp.get("max_speed", 3.0),
              "hierarchical_expert.local_planner.max_speed", errors)
    _positive(lp.get("max_accel", 2.0),
              "hierarchical_expert.local_planner.max_accel", errors)
    _positive(lp.get("max_yaw_rate", 1.5),
              "hierarchical_expert.local_planner.max_yaw_rate", errors)
    _positive(lp.get("max_yaw_accel", 4.0),
              "hierarchical_expert.local_planner.max_yaw_accel", errors)
    _positive(lp.get("min_clearance", 0.5),
              "hierarchical_expert.local_planner.min_clearance", errors)
    _positive(lp.get("control_period_s", 1.0 / 30.0),
              "hierarchical_expert.local_planner.control_period_s", errors)
    if float(lp.get("max_yaw_rate", 1.5)) > backend_yaw_rate + 1e-6:
        errors.append(
            "hierarchical_expert.local_planner.max_yaw_rate must not exceed "
            "the Flightmare backend limit (%g)" % backend_yaw_rate)
    if float(lp.get("max_yaw_accel", 4.0)) > backend_yaw_accel + 1e-6:
        errors.append(
            "hierarchical_expert.local_planner.max_yaw_accel must not exceed "
            "the Flightmare backend limit (%g)" % backend_yaw_accel)
    cw = lp.get("cost_weights", {}) or {}
    for name, default in (("progress", 1.0), ("clearance", 2.0),
                          ("obstacle_risk", 3.0)):
        _bounded(cw.get(name, default),
                 "hierarchical_expert.local_planner.cost_weights.%s" % name,
                 0.0, 100.0, errors)

    mc = he.get("corrector", {}) or {}
    _positive(mc.get("reentry_guard_ticks", 30),
              "hierarchical_expert.corrector.reentry_guard_ticks", errors,
              allow_zero=True)
    _positive(mc.get("correction_enter_stable_ticks", 1),
              "hierarchical_expert.corrector."
              "correction_enter_stable_ticks", errors, allow_zero=True)
    _positive(mc.get("observable_frontier_min_distance_m", 1.5),
              "hierarchical_expert.corrector."
              "observable_frontier_min_distance_m", errors)
    _positive(mc.get("corridor_half_width", 1.5),
              "hierarchical_expert.corrector.corridor_half_width", errors)

    ah = he.get("altitude_hold", {}) or {}
    _positive(ah.get("kp", 1.5), "hierarchical_expert.altitude_hold.kp",
              errors)
    _positive(ah.get("kd", 0.6), "hierarchical_expert.altitude_hold.kd",
              errors, allow_zero=True)
    _positive(ah.get("deadband_m", 0.02),
              "hierarchical_expert.altitude_hold.deadband_m", errors,
              allow_zero=True)
    _positive(ah.get("max_speed_mps", 0.6),
              "hierarchical_expert.altitude_hold.max_speed_mps", errors)

    # ── dataset logging (schema v25; R must match the expert) ─────
    ds = g.get("dataset_logging", {}) or {}
    _positive(ds.get("schema_version", 25),
              "dataset_logging.schema_version", errors)
    if int(ds.get("schema_version", 25)) != 25:
        errors.append("dataset_logging.schema_version must remain 25 "
                      "(save_net compatibility)")
    ds_perception = float(ds.get("perception_range_m", 5.0))
    if abs(ds_perception - perception) > 1e-6:
        errors.append(
            "dataset_logging.perception_range_m must equal "
            "hierarchical_expert.observation.range_m (single source)")
    _positive(ds.get("flush_interval_rows", 64),
              "dataset_logging.flush_interval_rows", errors)
    _bounded(ds.get("depth_png_compress_level", 4),
             "dataset_logging.depth_png_compress_level", 0, 9, errors)

    # ── sync / commit sanity ───────────────────────────────────────
    sync = g.get("sync", {})
    if float(sync.get("unity_response_timeout_s", 2.0)) <= 0:
        errors.append("sync.unity_response_timeout_s must be > 0")
    if float(sync.get("max_acceptable_latency_ms", 250.0)) <= 0:
        errors.append("sync.max_acceptable_latency_ms must be > 0")
    # Percentages must be within [0, 100] (item 十三).
    for pkey in ("max_unmatched_pct", "max_none_depth_pct",
                 "max_latency_violation_pct"):
        _bounded(sync.get(pkey, 1.0),
                 "sync.%s" % pkey, 0.0, 100.0, errors)
    # Catastrophic latency must be at least the acceptable threshold.
    if float(sync.get("catastrophic_latency_ms", 5000.0)) < \
            float(sync.get("max_acceptable_latency_ms", 250.0)):
        errors.append("sync.catastrophic_latency_ms must be >= "
                      "sync.max_acceptable_latency_ms")
    # Strict per-frame exact-match wait for ONE render attempt.  > 0 only;
    # this is per-frame retry waiting, never a trajectory timeout.
    _positive(sync.get("frame_match_timeout_s", 0.15),
              "sync.frame_match_timeout_s", errors)
    commit = g.get("commit", {}) or {}
    if not bool(commit.get("atomic_rename", True)):
        errors.append("commit.atomic_rename must remain true")
    # Max render attempts per control tick on the SAME saved state.
    _positive(commit.get("max_frame_retries", 5),
              "commit.max_frame_retries", errors, allow_zero=True)

    if errors:
        raise ValueError("Configuration errors:\n  - " + "\n  - ".join(errors))


def load_config(config_path=None, validate=True):
    """Load and validate the YAML configuration.

    Returns the full config dict (``config["global"]`` holds the modules).
    """
    # roslaunch provides ``config_file`` as a private node parameter.  It
    # must be read *before* resolving the YAML path; the old order always
    # resolved the default first, then merely recorded this parameter in the
    # already-loaded configuration.
    requested_path = config_path
    if not requested_path:
        try:
            if rospy.has_param("~config_file"):
                requested_path = rospy.get_param("~config_file")
        except Exception:
            pass
    path = _resolve_path(requested_path)
    if not os.path.isfile(path):
        raise ValueError("Config file not found: %s" % path)
    # Explicit UTF-8: the YAML comments use box-drawing characters and the
    # default locale on Windows (cp936) cannot decode them.
    cfg = _load_yaml_with_extends(path)
    if not isinstance(cfg, dict) or "global" not in cfg:
        raise ValueError("Config must have a top-level 'global' section")
    # Item 十: fill the UNIQUE default depth.t_bc before anything reads it,
    # so the Unity wire (make_depth_vehicle) and the C++ camera rig share
    # exactly one matrix.
    _inject_defaults(cfg)
    g = cfg["global"]

    # ROS param overrides (ports and scene id).
    try:
        if rospy.has_param("~pub_port"):
            g["pub_port"] = rospy.get_param("~pub_port")
        if rospy.has_param("~sub_port"):
            g["sub_port"] = rospy.get_param("~sub_port")
        if rospy.has_param("~scene_id"):
            g["scene_id"] = int(rospy.get_param("~scene_id"))
        g["_config_source"] = path
    except Exception:
        pass

    if not g.get("output_dir"):
        try:
            rospack = rospkg.RosPack()
            pkg_dir = rospack.get_path(_PKG)
            g["output_dir"] = os.path.join(pkg_dir, "dataset", "il_data")
        except Exception:
            g["output_dir"] = "dataset/il_data"

    if validate:
        _validate_config(cfg)
    return cfg


