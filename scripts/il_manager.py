#!/usr/bin/env python3
"""
il_manager.py  —  The ONLY collection manager (both
`il_dataset_joint_v2_collect.launch`.  Production path:

    C++ scene/ESDF/task blueprint generation + C++ classifier + quotas
        -> full manifest written FIRST
        -> then connect Unity + FlightmareDynamics
        -> strict frame-synchronised Flightmare collection per manifest
        -> C++ HierarchicalExpert (5 Hz corrector + 30 Hz planner)
        -> new two-level labels written to data.csv
        -> real episode commit audit (continuous exact-cylinder truth)

Responsibility split (strict):
  * Python (this file) owns ROS/Flightmare lifecycle, the Unity binary
    depth decode, STRICT frame synchronisation, expert invocation and
    dataset writing.  It never runs a planning / map / scene / ESDF /
    classifier algorithm.
  * The expert, the scene/ESDF/task/blueprint generator, the behavior
    classifier + quotas, the preflight simulator and the exact-cylinder
    truth audit are ALL pure C++17 in `_il_hierarchical_expert`.
  * Truth (generated cylinder geometry) is used ONLY for scene generation,
    start/goal connectivity screening, collision/clearance audits and the
    episode commit decision — never to generate expert actions.

Data-collection sequence (item 五):
    generate ALL scenes -> generate ALL connected tasks -> closed-loop
    preflight EVERY task with the SAME expert -> C++ classifier + quota
    selection -> write the FULL manifest -> only then connect Unity /
    create dynamics -> collect STRICTLY per the manifest.  Normal mode
    never re-randomises tasks and never generates scenes while flying.

There is exactly ONE HierarchicalExpert instance; it is the sole generator
of every control command.  The expert outputs the full 3D BODY/FLU
velocity [vx, vy, vz] + yaw_rate (the vertical channel is regulated toward
the mission goal z by the C++ VerticalController / CommandComposer3D).  No
Python-side altitude merge exists.

The following are NOT part of the production / preflight / blueprint /
label paths and have been REMOVED entirely (deleted sources and tests):
  * CausalLocalTargetStream / PrivilegedMicroDetourPlanner (C++ legacy),
  * scripts/il_macro_expert.py, scripts/il_micro_expert.py,
  * the old goal switcher, micro-detour controller, old
    behavior-classifier / blueprint generator and debug_viewer.py.
"""

from __future__ import print_function, division

import csv
import glob
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from types import SimpleNamespace

import numpy as np

import rospy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import il_config
import il_common
import il_expert_config
import il_dataset_writer

try:
    import _il_hierarchical_expert as expert_mod
except Exception as e:  # pragma: no cover
    rospy.logfatal(
        "_il_hierarchical_expert module is unavailable: %s.  Rebuild with "
        "`catkin build il_dataset` (the expert must be C++, never Python).",
        e)
    raise


# ============================================================================
#  Episode commit audit (real; geometry in C++ TruthCylinderAudit)
# ============================================================================

class EpisodeAudit(object):
    """Accumulates every commit-relevant counter for ONE episode.

    The heavy geometry (continuous swept collision / clearance / brake
    risk against the exact cylinders) lives in the C++
    `expert_mod.TruthCylinderAudit`; this class only counts and records.

    The strict 30 Hz state machine (item 七) separates:
      * RENDER attempts (pose sends to Unity; sync.csv) — may be retried on
        the SAME control state without advancing dynamics;
      * CONTROL ticks (committed rows in data.csv) — each requires an exact
        frame_id depth, a valid expert output and an executed command.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.render_attempts = 0
        self.matched = 0
        self.unmatched = 0          # render attempts with no exact frame_id
        self.none_depth = 0         # render attempts with a payload error
        self.stale = 0
        self.parse_failures = 0
        self.latency_violations = 0
        self.catastrophic_latency = False
        self.unity_collision = False
        self.frame_retries_exceeded = False
        self.macro_label_invalid = False
        self.non_finite_label = False
        self.command_rejected = False
        self.truth_swept_collision = False
        self.out_of_bounds = False
        self.truth_brake_triggered = False
        self.min_truth_clearance_m = float("inf")
        self.max_observed_brake_risk = 0.0
        self.state_segments = 0
        self.reached_goal = False
        self.terminal_state = ""
        self.final_speed_mps = -1.0
        self.final_yaw_rate_rps = -1.0
        # R26 judge-only quality-watchdog flags (reject pathological
        # episodes that churn/spin instead of progressing).
        self.goal_no_progress = False
        self.near_goal_timeout = False
        self.excessive_yaw = False

    def note_state_segment(self, x0, y0, x1, y1, truth):
        """Continuous swept audit of one EXECUTED dynamics segment."""
        self.state_segments += 1
        self.min_truth_clearance_m = min(
            self.min_truth_clearance_m,
            truth.segment_min_clearance(x0, y0, x1, y1))
        if truth.segment_collision(x0, y0, x1, y1):
            self.truth_swept_collision = True


# ============================================================================
#  Joint-v2 collection manager
# ============================================================================

class JointV2Manager(object):
    def __init__(self, config_path=None, blueprint_only=False, dry_run=False,
                 manifest_file=None):
        self._config = il_config.load_config(config_path)
        self._g = self._config["global"]
        self._blueprint_only = bool(blueprint_only)
        self._dry_run = bool(dry_run)
        # Expand `~` so a user-supplied `manifest_file:=~/...` is matched by
        # os.path.isfile() (a literal `~` would otherwise silently fall back
        # to blueprint regeneration, which also clears old manifests).
        self._manifest_file = (
            os.path.expanduser(manifest_file) if manifest_file else None)
        self._manifest_expert_revision = ""
        self._episode_id_counter = 0
        self._current_episode_id = ""

        he = self._g.get("hierarchical_expert", {}) or {}
        self._control_hz = float(he.get("control_hz", 30.0))
        self._macro_update_hz = float(he.get("macro_update_hz", 5.0))
        if self._macro_update_hz <= 0.0 or \
                abs(self._control_hz / self._macro_update_hz -
                    round(self._control_hz / self._macro_update_hz)) > 1e-6:
            raise ValueError(
                "hierarchical_expert.macro_update_hz must divide control_hz "
                "exactly (5 Hz corrector on a 30 Hz tick grid)")

        self._errors = []
        self._params = il_expert_config.build_params(self._g, self._errors)
        if self._errors:
            raise ValueError("hierarchical_expert config errors:\n  - " +
                             "\n  - ".join(self._errors))
        min_b, max_b = il_expert_config.build_scene_bounds(self._g)
        self._region_min = np.asarray(min_b, dtype=np.float64)
        self._region_max = np.asarray(max_b, dtype=np.float64)

        # The ONE C++ expert instance (owns all expert state).
        self._expert = expert_mod.HierarchicalExpert()
        rospy.loginfo(
            "[Manager] expert .so revision = %s",
            str(getattr(expert_mod, "EXPERT_REVISION",
                        "<no EXPERT_REVISION — STALE .so>")))
        self._expert.configure(self._params, list(min_b), list(max_b))

        # ROS params (launch-provided).
        try:
            if rospy.has_param("~pub_port"):
                self._g["pub_port"] = rospy.get_param("~pub_port")
            if rospy.has_param("~sub_port"):
                self._g["sub_port"] = rospy.get_param("~sub_port")
            if rospy.has_param("~scene_id"):
                self._g["scene_id"] = int(rospy.get_param("~scene_id"))
        except Exception:
            pass

        self._depth_cfg = self._g.get("depth", {})
        # Deterministic RNG for D435i-style depth noise (sim-to-real).
        self._noise_rng = np.random.default_rng(20260829)
        self._perception_range = float(
            self._g.get("dataset_logging", {}).get("perception_range_m",
                                                   self._params.obs_range_m))
        if abs(self._perception_range - self._params.obs_range_m) > 1e-6:
            raise ValueError(
                "dataset_logging.perception_range_m must equal "
                "hierarchical_expert.observation.range_m (single source)")

        self._vehicle_radius = float(self._g.get("vehicle", {}).get(
            "radius_m", 0.30))
        self._clearance = float(self._g.get("navigation", {}).get(
            "clearance_m", 0.30))
        self._flight_height = float(
            (self._g.get("task_generation", {}) or {}).get(
                "flight_height_m", 2.0))

        # Sync / commit thresholds (strict frame synchronisation).
        self._sync_cfg = self._g.get("sync", {}) or {}
        self._commit_cfg = self._g.get("commit", {}) or {}
        self._max_latency_ms = float(
            self._sync_cfg.get("max_acceptable_latency_ms", 250.0))
        self._catastrophic_latency_ms = float(
            self._sync_cfg.get("catastrophic_latency_ms", 5000.0))
        self._max_unmatched_pct = float(
            self._sync_cfg.get("max_unmatched_pct", 1.0))
        self._max_none_depth_pct = float(
            self._sync_cfg.get("max_none_depth_pct", 20.0))
        self._max_latency_violation_pct = float(
            self._sync_cfg.get("max_latency_violation_pct", 1.0))
        # Strict single-frame exact-match wait (NOT a flight trajectory
        # timeout) and the maximum render retries per control tick.
        self._frame_match_timeout_s = float(
            self._sync_cfg.get("frame_match_timeout_s", 0.15))
        self._max_frame_retries = int(
            self._commit_cfg.get("max_frame_retries", 5))
        self._wall_timeout_s = float(
            self._g.get("fsm", {}).get("trajectory_wall_timeout_s", 0.0))
        # R26 judge-only quality watchdogs (reject pathological episodes).
        self._goal_no_progress_ticks = int(
            self._commit_cfg.get("goal_no_progress_ticks", 150))
        self._goal_no_progress_min_m = float(
            self._commit_cfg.get("goal_no_progress_min_m", 0.15))
        self._goal_no_progress_floor_m = float(
            self._commit_cfg.get("goal_no_progress_floor_m", 0.5))
        self._near_goal_radius_m = float(
            self._commit_cfg.get("near_goal_radius_m", 1.0))
        self._near_goal_timeout_ticks = int(
            self._commit_cfg.get("near_goal_timeout_ticks", 300))
        self._max_cumulative_yaw_rad = float(
            self._commit_cfg.get("max_cumulative_yaw_rad", 12.566))

        self._bridge = None
        self._dynamics = None
        # Monotonic Unity render frame id (never reset across episodes, so
        # a stale response from a previous scene/task can never match).
        self._next_frame_id = 1
        # Persistent set of Unity object IDs already sent (for retiring
        # obstacles of a previous scene via build_replacing_object_update).
        self._known_object_ids = set()
        # The current scene's Unity object list, re-sent on EVERY pose (like
        # the v1/v3/master collection loops).  Sending objects=[] would make
        # the AvoidBench binary drop the scene obstacles after the first
        # pose, so the expert would fly through an empty scene.
        self._current_unity_objects = []
        self._truth = expert_mod.TruthCylinderAudit()

    # ═══════════════════════════════════════════════════════════════
    #  Full C++ blueprint generation (deficit-driven pipeline)
    # ═══════════════════════════════════════════════════════════════
    def _known_rects(self, bp):
        """Load fixed known-obstacle AABBs.

        Two sources (merged):
          * blueprint_generation.known_rects — inline list of
            {min_x,max_x,min_y,max_y,height_m}
          * blueprint_generation.known_obstacles_file — a cluster JSON
            (from slice_pointcloud_layers.py) whose clusters are converted
            to their bounding-box AABB with a height spanning the flight
            band (obstacle_height_max_m).
        """
        rects = list(bp.get("known_rects", []) or [])
        fpath = bp.get("known_obstacles_file", "") or ""
        if fpath:
            fpath = os.path.expanduser(str(fpath))
            if not os.path.isfile(fpath):
                raise ValueError("known_obstacles_file not found: %s" % fpath)
            with open(fpath) as f:
                data = json.load(f)
            btg = bp.get("task_generation", {}) or {}
            hgt = float(btg.get(
                "obstacle_height_max_m",
                btg.get("obstacle_height_m", 8.0)))
            for c in (data.get("clusters", []) or []):
                w = float(c.get("w", 0.0))
                hh = float(c.get("h", 0.0))
                x = float(c.get("x", 0.0))
                y = float(c.get("y", 0.0))
                if w <= 0.0 or hh <= 0.0:
                    continue
                rects.append({
                    "min_x": x - w / 2.0,
                    "max_x": x + w / 2.0,
                    "min_y": y - hh / 2.0,
                    "max_y": y + hh / 2.0,
                    "height_m": hgt,
                })
        return rects

    def _blueprint_config_dict(self):
        """Build the C++ SceneTaskBlueprintGenerator.Config from YAML.

        Legacy top-level keys (scene_generation / task_generation / quotas)
        stay for backward compatibility; the NEW `blueprint_generation`
        section (when enabled) provides the full deficit-driven pipeline
        config under the `blueprint` sub-dict (warehouse single source,
        profiles, layered initial yaw, depth proxy, distribution targets,
        budgets).  When `blueprint_generation.enabled` is false only the
        legacy keys are sent (C++ falls back).
        """
        sg = self._g.get("scene_generation", {}) or {}
        tg = self._g.get("task_generation", {}) or {}
        geo = sg.get("geometry", {}) or {}
        q = tg.get("quotas", {}) or {}
        bp = self._g.get("blueprint_generation", {}) or {}
        bp_enabled = bool(bp.get("enabled", True))

        legacy = {
            "scene_count": int(sg.get("scene_count", 10)),
            "tasks_per_scene": int(sg.get("tasks_per_scene", 8)),
            "minimum_tasks_per_scene": int(
                sg.get("minimum_tasks_per_scene", 6)),
            "base_seed": int(sg.get("seed", 260812)),
            "flight_height_m": float(tg.get("flight_height_m", 2.0)),
            "obstacle_height_m": float(geo.get("height_m", 8.0)),
            "require_full_strata_coverage": bool(
                sg.get("require_full_strata_coverage", True)),
            "min_surface_gap_m": float(
                geo.get("minimum_surface_gap_m", 1.40)),
            "boundary_margin_m": float(geo.get("boundary_margin_m", 1.2)),
            "radius_min_m": float(geo.get("radius_min_m", 0.10)),
            "radius_max_m": float(geo.get("radius_max_m", 6.0)),
            "max_obstacles": int(geo.get("max_obstacles", 30)),
            "vehicle_radius_m": float(self._vehicle_radius),
            "navigation_clearance_m": float(self._clearance),
            "free_cell_surface_clearance_m": float(
                geo.get("free_cell_surface_clearance_m", 0.5)),
            "esdf_resolution_m": float(geo.get("esdf_resolution_m", 0.1)),
            "max_generation_attempts": int(
                sg.get("max_generation_attempts", 96)),
            "min_task_distance_m": float(tg.get("min_task_distance_m", 4.0)),
            "max_task_distance_m": float(tg.get("max_task_distance_m", 20.0)),
            "initial_yaw_bias_deg": float(
                tg.get("initial_yaw_bias_deg", 15.0)),
            "task_sample_attempts": int(
                tg.get("task_sample_attempts", 300)),
            "candidate_pool_multiplier": int(
                tg.get("candidate_pool_multiplier", 4)),
            "qualification_attempt_budget": int(
                tg.get("qualification_attempt_budget", 600)),
            "preflight_qualification_max_ticks": int(
                tg.get("preflight_qualification_max_ticks", 900)),
            "min_per_behavior": int(q.get("min_per_behavior", 2)),
            "min_turn_per_side": int(q.get("min_turn_per_side", 2)),
            "max_left_right_imbalance": int(
                q.get("max_left_right_imbalance", 2)),
            "min_per_density_level": int(
                q.get("min_per_density_level", 4)),
            "min_per_radius_level": int(
                q.get("min_per_radius_level", 4)),
            "min_per_distance_level": int(
                q.get("min_per_distance_level", 4)),
            "distance_short_max_m": float(
                q.get("distance_short_max_m", 9.0)),
            "distance_long_min_m": float(q.get("distance_long_min_m", 15.0)),
            "radius_small_max_m": float(q.get("radius_small_max_m", 0.6)),
            "radius_large_min_m": float(q.get("radius_large_min_m", 1.4)),
            "density_sparse_max": float(q.get("density_sparse_max", 7.0)),
            "density_dense_min": float(q.get("density_dense_min", 14.0)),
            "long_takeover_min_ticks": int(
                q.get("long_takeover_min_ticks", 30)),
        }
        if not bp_enabled:
            return legacy

        # ── NEW: blueprint_generation section (deficit-driven) ─────
        bsg = bp.get("scene_generation", {}) or {}
        btg = bp.get("task_generation", {}) or {}
        perf = bp.get("performance", {}) or {}
        req = bp.get("requirements", {}) or {}
        leg = bp.get("legacy", {}) or {}
        wh = bp.get("warehouse", {}) or {}
        fr = wh.get("free_region", []) or []
        if not fr or len(fr) < 4:
            # Single-source fallback: hierarchical_expert.region IS the
            # warehouse free region ([-7,10] x [0,30]).
            he_region = self._g.get("hierarchical_expert", {}).get(
                "region", {}) or {}
            fr = [float(he_region.get("min_x", -7.0)),
                  float(he_region.get("max_x", 10.0)),
                  float(he_region.get("min_y", 0.0)),
                  float(he_region.get("max_y", 30.0))]
        else:
            fr = [float(v) for v in fr]

        blueprint = {
            "base_seed": int(bp.get("base_seed", sg.get("seed", 260812))),
            "warehouse": {
                "free_region": fr,
                "wall_extension_m": float(wh.get("wall_extension_m", 1.0)),
            },
            "vehicle_radius_m": float(self._vehicle_radius),
            "navigation_clearance_m": float(self._clearance),
            "clearance_discretization_margin_m": float(
                bsg.get("clearance_discretization_margin_m", 0.05)),
            "generation_margin_m": float(
                bsg.get("generation_margin_m", 0.05)),
            "min_surface_gap_m": float(
                bsg.get("min_surface_gap_m",
                        geo.get("minimum_surface_gap_m", 1.40))),
            "boundary_margin_m": float(
                bsg.get("boundary_margin_m",
                        geo.get("boundary_margin_m", 1.20))),
            "free_cell_surface_clearance_m": float(
                bsg.get("free_cell_surface_clearance_m",
                        geo.get("free_cell_surface_clearance_m", 0.5))),
            "esdf_resolution_m": float(
                bsg.get("esdf_resolution_m",
                        geo.get("esdf_resolution_m", 0.1))),
            "min_main_component_area_m2": float(
                bsg.get("min_main_component_area_m2", 60.0)),
            "use_profile_catalog": bool(
                bsg.get("use_profile_catalog", True)),
            "profiles": list(bsg.get("profiles", []) or []),
            "profile_sequence": list(bsg.get("profile_sequence", []) or []),
            "min_task_distance_m": float(
                btg.get("min_task_distance_m",
                        tg.get("min_task_distance_m", 4.0))),
            "max_task_distance_m": float(
                btg.get("max_task_distance_m",
                        tg.get("max_task_distance_m", 20.0))),
            "flight_height_m": float(
                btg.get("flight_height_m", tg.get("flight_height_m", 2.0))),
            "flight_height_min_m": float(
                btg.get("flight_height_min_m",
                        btg.get("flight_height_m",
                                tg.get("flight_height_m", 2.0)))),
            "flight_height_max_m": float(
                btg.get("flight_height_max_m",
                        btg.get("flight_height_m",
                                tg.get("flight_height_m", 2.0)))),
            "obstacle_height_m": float(
                btg.get("obstacle_height_m", geo.get("height_m", 8.0))),
            "obstacle_height_min_m": float(
                btg.get("obstacle_height_min_m",
                        btg.get("obstacle_height_m",
                                geo.get("height_m", 8.0)))),
            "obstacle_height_max_m": float(
                btg.get("obstacle_height_max_m",
                        btg.get("obstacle_height_m",
                                geo.get("height_m", 8.0)))),
            "task_sample_attempts": int(
                btg.get("task_sample_attempts",
                        tg.get("task_sample_attempts", 300))),
            "task_goal_attempts": int(btg.get("task_goal_attempts", 120)),
            "initial_yaw": dict(btg.get("initial_yaw", {}) or {}),
            "depth_proxy": dict(btg.get("depth_proxy", {}) or {}),
            "histograms": dict(btg.get("histograms", {}) or {}),
            "path": dict(btg.get("path", {}) or {}),
            "known_rects": self._known_rects(bp),
            "performance": dict(perf),
            "scene_parallel": dict(bp.get("scene_parallel", {}) or {}),
            "requirements": dict(req),
            "synthetic_observation": dict(
                bp.get("synthetic_observation", {}) or {}),
            "early_termination": dict(bp.get("early_termination", {}) or {}),
            "task_qualification": dict(
                bp.get("task_qualification", {}) or {}),
            "distribution_targets": list(
                bp.get("distribution_targets", []) or []),
            # Preflight control rate (Hz); dt = 1/control_rate_hz is the
            # stall-displacement / duration time base (no magic 30.0).
            "control_rate_hz": float(
                bp.get("control_rate_hz",
                       (self._g.get("hierarchical_expert", {}) or {}).get(
                           "control_hz", 30.0))),
            "macro_probe": dict(btg.get("macro_probe", {}) or {}),
            # Candidate-pool-first exploration is intentionally separate from
            # distribution targets: early random coverage is collected first,
            # then the C++ selector supplements deficits from that pool.
            "exploration": dict(bp.get("exploration", {}) or {}),
            "legacy": dict(leg),
        }
        legacy["blueprint"] = blueprint
        return legacy

    def _generate_blueprint(self):
        """ONE C++ call that returns the FULL blueprint (both modes)."""
        gen = expert_mod.SceneTaskBlueprintGenerator()
        gen.configure(self._params, self._blueprint_config_dict())
        result = gen.generate()
        if result.scenes_generated == 0 or not result.tasks:
            rospy.logerr("[Manager] blueprint produced no usable tasks "
                         "(scenes=%d preflighted=%d quota=%d)",
                         result.scenes_generated, result.tasks_preflighted,
                         result.tasks_quota_accepted)
        return result

    def _write_blueprint_manifest(self, result):
        output_root = os.path.expanduser(self._g.get("output_dir", ""))
        if not output_root:
            output_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dataset", "il_data")
        os.makedirs(output_root, exist_ok=True)
        # Regeneration always starts from a clean task folder: every old
        # blueprint manifest is removed before the new one is written.
        for old in sorted(glob.glob(
                os.path.join(output_root, "*_manifest.json"))):
            try:
                os.remove(old)
            except OSError:
                pass
        manifest = os.path.join(
            output_root, "joint_v2_blueprint_manifest.json")

        scenes = []
        for s in result.scenes:
            md = s.metadata
            scenes.append({
                "scene_id": int(s.scene_id),
                "seed": int(s.seed),
                "profile": str(s.profile),
                "structure_orientation": str(s.structure_orientation),
                "metadata": {
                    "structure_orientation": str(md.structure_orientation),
                    "obstacle_count": int(md.obstacle_count),
                    "radius_min": float(md.radius_min),
                    "radius_max": float(md.radius_max),
                    "radius_mean": float(md.radius_mean),
                    "tiny_count": int(md.tiny_count),
                    "small_count": int(md.small_count),
                    "medium_count": int(md.medium_count),
                    "large_count": int(md.large_count),
                    "local_density_proxy": float(md.local_density_proxy),
                    "largest_obstacle_radius":
                        float(md.largest_obstacle_radius),
                    "cluster_count": int(md.cluster_count),
                    "free_space_ratio": float(md.free_space_ratio),
                    "estimated_corridor_width":
                        float(md.estimated_corridor_width),
                    "geometry_valid": bool(md.geometry_valid),
                    "geometry_failure_reason":
                        str(md.geometry_failure_reason),
                    "planning_valid": bool(md.planning_valid),
                    "planning_failure_reason":
                        str(md.planning_failure_reason),
                },
                "stratum_id": int(s.stratum_id),
                "is_empty": bool(s.is_empty),
                "planned_density_class": str(s.planned_density_class),
                "planned_radius_class": str(s.planned_radius_class),
                "actual_density_class": str(s.actual_density_class),
                "actual_radius_class": str(s.actual_radius_class),
                "actual_min_radius_m": float(s.actual_min_radius_m),
                "actual_max_radius_m": float(s.actual_max_radius_m),
                "density_class": str(s.density_class),
                "requested_obstacle_count": int(s.requested_obstacle_count),
                "actual_obstacle_count": int(s.actual_obstacle_count),
                "generation_valid": bool(s.generation_valid),
                "failure_reason": str(s.failure_reason),
                "obstacles": [
                    {"id": int(o.id), "x": float(o.x), "y": float(o.y),
                     "radius": float(o.radius), "height_m": float(o.height_m)}
                    for o in s.obstacles],
            })

        def _fin(x):
            """Finite-only JSON sanitizer (inf/NaN -> 0.0)."""
            try:
                v = float(x)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(v):
                return 0.0
            return v

        def _summary_dict(sm):
            return {
                "straight_distance_m": float(sm.straight_distance_m),
                "preflight_path_length_m": float(sm.preflight_path_length_m),
                "path_stretch_ratio": float(sm.path_stretch_ratio),
                "preflight_duration_s": float(sm.preflight_duration_s),
                "preflight_ticks": int(sm.preflight_ticks),
                "initial_yaw_error_signed_deg":
                    float(sm.initial_yaw_error_signed_deg),
                "initial_yaw_error_abs_deg":
                    float(sm.initial_yaw_error_abs_deg),
                "depth": {
                    "samples": int(sm.depth_samples),
                    "near_count": int(sm.depth_near_count),
                    "mid_count": int(sm.depth_mid_count),
                    "far_count": int(sm.depth_far_count),
                    "free_count": int(sm.depth_free_count),
                    "near_ratio": float(sm.near_depth_ratio()),
                    "mid_ratio": float(sm.mid_depth_ratio()),
                    "far_ratio": float(sm.far_depth_ratio()),
                    "free_ratio": float(sm.free_depth_ratio()),
                    "min_visible_m": _fin(sm.depth_min_visible_m),
                    "mean_visible_m": _fin(sm.depth_mean_visible_m),
                    "max_angular_occlusion_deg":
                        float(sm.depth_max_angular_occlusion_deg),
                    "occupied_ray_ratio": float(sm.depth_occupied_ray_ratio),
                },
                "macro5hz": {
                    "tick_total": int(sm.macro_tick_total),
                    "pass_count": int(sm.macro_pass_count),
                    "normal_count": int(sm.macro_normal_count),
                    "turn_left_count": int(sm.macro_turn_left_count),
                    "turn_right_count": int(sm.macro_turn_right_count),
                    "correction_angle_hist": list(
                        sm.macro_correction_angle_hist.counts),
                    "correction_angle_edges": list(
                        sm.macro_correction_angle_hist.edges),
                    "correction_distance_hist": list(
                        sm.macro_correction_distance_hist.counts),
                },
                "local30hz": {
                    "direct_count": int(sm.local_direct_count),
                    "avoidance_count": int(sm.local_avoidance_count),
                    "deflection_hist": list(sm.local_deflection_hist.counts),
                    "deflection_edges": list(sm.local_deflection_hist.edges),
                    "yaw_rate_hist": list(sm.local_yaw_rate_hist.counts),
                    "speed_hist": list(sm.local_speed_hist.counts),
                    "min_observed_clearance_m":
                        _fin(sm.min_observed_clearance_m),
                    "mean_observed_clearance_m":
                        _fin(sm.mean_observed_clearance_m),
                },
                "quality": {
                    "reached_goal": bool(sm.reached_goal),
                    "collision": bool(sm.collision),
                    "out_of_bounds": bool(sm.out_of_bounds),
                    "minimum_clearance_m": _fin(sm.minimum_clearance_m),
                },
            }

        def _task_dict(t):
            return {
                "scene_id": int(t.scene_id),
                "task_id": int(t.task_id),
                "seed": int(t.seed),
                "start": [float(t.start_x), float(t.start_y)],
                "goal": [float(t.goal_x), float(t.goal_y)],
                "initial_yaw": float(t.initial_yaw),
                "flight_height_m": float(t.flight_height_m),
                "behavior_class": str(t.behavior_class),
                "density_class": str(t.density_class),
                "radius_class": str(t.radius_class),
                "distance_class": str(t.distance_class),
                "side_class": str(t.side_class),
                "saw_turn_left": bool(t.saw_turn_left),
                "saw_turn_right": bool(t.saw_turn_right),
                "saw_normal_correction": bool(t.saw_normal_correction),
                "turn_update_count": int(t.turn_update_count),
                "normal_update_count": int(t.normal_update_count),
                "geom_type": str(t.geom_type),
                "selection_score": float(t.selection_score),
                # ── privileged task-qualification diagnostics (manifest /
                #    generation statistics ONLY; never student inputs) ──
                "qualification": {
                    "endpoint_valid": bool(t.qualification.endpoint_valid),
                    "connectivity_valid": bool(
                        t.qualification.connectivity_valid),
                    "straight_corridor_clear": bool(
                        t.qualification.straight_corridor_clear),
                    "primary_blocker_id":
                        int(t.qualification.primary_blocker_id),
                    "primary_blocker_x":
                        float(t.qualification.primary_blocker_x),
                    "primary_blocker_y":
                        float(t.qualification.primary_blocker_y),
                    "primary_blocker_radius":
                        float(t.qualification.primary_blocker_radius),
                    "blocking_obstacle_ids":
                        [int(i) for i in t.qualification.blocking_obstacle_ids],
                    "left": {
                        "checked": bool(t.qualification.left.checked),
                        "feasible": bool(t.qualification.left.feasible),
                        "path_length_m":
                            _fin(t.qualification.left.path_length_m),
                        "min_clearance_m":
                            _fin(t.qualification.left.min_clearance_m),
                        "expanded_nodes":
                            int(t.qualification.left.expanded_nodes),
                        "reject_reason":
                            str(t.qualification.left.reject_reason),
                    },
                    "right": {
                        "checked": bool(t.qualification.right.checked),
                        "feasible": bool(t.qualification.right.feasible),
                        "path_length_m":
                            _fin(t.qualification.right.path_length_m),
                        "min_clearance_m":
                            _fin(t.qualification.right.min_clearance_m),
                        "expanded_nodes":
                            int(t.qualification.right.expanded_nodes),
                        "reject_reason":
                            str(t.qualification.right.reject_reason),
                    },
                    "privileged_min_route_stretch":
                        _fin(t.qualification.privileged_min_route_stretch),
                    "narrow_passage_id":
                        int(t.qualification.narrow_passage_id),
                    "route_traverses_narrow":
                        bool(t.qualification.route_traverses_narrow),
                    "realized_geom_type":
                        str(t.qualification.realized_geom_type),
                    "qualification_class":
                        str(t.qualification.qualification_class),
                    "reject_reason": str(t.qualification.reject_reason),
                    "accepted": bool(t.qualification.accepted),
                },
                "audit": {
                    "accepted": bool(t.audit.accepted),
                    "reached_goal": bool(t.audit.reached_goal),
                    "truth_collision": bool(t.audit.truth_collision),
                    "truth_brake_triggered": bool(
                        t.audit.truth_brake_triggered),
                    "out_of_bounds": bool(t.audit.out_of_bounds),
                    "macro_label_ok": bool(t.audit.macro_label_ok),
                    "qualification_exceeded":
                        bool(t.audit.qualification_exceeded),
                    "preflight_ticks": int(t.audit.preflight_ticks),
                    "min_truth_clearance_m":
                        _fin(t.audit.min_truth_clearance_m),
                    "goal_distance_m": float(t.audit.goal_distance_m),
                    "straight_distance_m": float(t.audit.straight_distance_m),
                    "path_length_m": float(t.audit.path_length_m),
                    "path_stretch_ratio": float(t.audit.path_stretch_ratio),
                    "preflight_duration_s": float(t.audit.preflight_duration_s),
                    "preflight_status": str(t.audit.preflight_status),
                },
                "summary": _summary_dict(t.summary),
            }

        payload = {
            "expert_stack_revision": "hierarchical_local_v1",
            "expert_revision": str(getattr(
                expert_mod, "EXPERT_REVISION",
                "<no EXPERT_REVISION in .so — stale build>")),
            "schema_extensions": ["two_level_expert_labels_v1"],
            "generation_ok": bool(result.generation_ok),
            "failure_reason": str(result.failure_reason),
            "unmet_quotas": [str(u) for u in result.unmet_quotas],
            "base_seed": int(result.base_seed),
            "requested_scenes": int(result.requested_scenes),
            "requested_tasks_per_scene": int(result.requested_tasks_per_scene),
            "scenes_generated": int(result.scenes_generated),
            "scenes_valid": int(result.scenes_valid),
            "strata_required": int(result.strata_required),
            "strata_covered": int(result.strata_covered),
            "strata_covered_flags":
                [int(f) for f in result.strata_covered_flags],
            "per_scene_accepted": [int(c) for c in result.per_scene_accepted],
            "category_counts": {str(k): int(v) for k, v in
                                result.category_counts.items()},
            "tasks_sampled": int(result.tasks_sampled),
            "tasks_preflighted": int(result.tasks_preflighted),
            "tasks_pool_target": int(result.tasks_pool_target),
            "tasks_pool_accepted": int(result.tasks_pool_accepted),
            "tasks_quota_accepted": int(result.tasks_quota_accepted),
            "total_task_candidates": int(result.total_task_candidates),
            "preflight_success_tasks": int(result.preflight_success_tasks),
            "cheap_filter_rejected": int(result.cheap_filter_rejected),
            "pool_budget_exhausted": bool(result.pool_budget_exhausted),
            # ── NEW: efficiency + budget diagnostics ───────────────
            "preflight_attempt_count": int(result.preflight_attempt_count),
            "preflight_success_count": int(result.preflight_success_count),
            "preflight_failure_count": int(result.preflight_failure_count),
            "total_preflight_ticks": int(result.total_preflight_ticks),
            "full_preflight_attempted": int(result.full_preflight_attempted),
            "full_preflight_success": int(result.full_preflight_success),
            "selected_scene_count": int(result.selected_scene_count),
            "preflight_acceptance_ratio":
                float(result.preflight_acceptance_ratio),
            "selected_per_preflight_ratio":
                float(result.selected_per_preflight_ratio),
            "budget_exhausted_reason": str(result.budget_exhausted_reason),
            # ── privileged task-qualification efficiency (aggregate) ──
            "qualification_rejected": int(result.qualification_rejected),
            "task_candidates_generated": int(result.task_candidates_generated),
            "endpoint_pass_count": int(result.endpoint_pass_count),
            "connectivity_pass_count": int(result.connectivity_pass_count),
            "straight_clear_count": int(result.straight_clear_count),
            "blocked_count": int(result.blocked_count),
            "side_qualification_attempt_count":
                int(result.side_qualification_attempt_count),
            "both_sides_feasible_count": int(result.both_sides_feasible_count),
            "qualification_accept_count": int(result.qualification_accept_count),
            "total_astar_expansions": int(result.total_astar_expansions),
            "qualification_pass_ratio": float(result.qualification_pass_ratio),
            "full_preflight_success_after_qualification_ratio":
                float(result.full_preflight_success_after_qualification_ratio),
            "qualification": {
                "candidates_checked":
                    int(result.qualification.candidates_checked),
                "endpoint_pass": int(result.qualification.endpoint_pass),
                "connectivity_pass":
                    int(result.qualification.connectivity_pass),
                "straight_clear": int(result.qualification.straight_clear),
                "blocked": int(result.qualification.blocked),
                "side_qualification_attempt":
                    int(result.qualification.side_qualification_attempt),
                "both_sides_feasible":
                    int(result.qualification.both_sides_feasible),
                "accepted": int(result.qualification.accepted),
                "reject_endpoint": int(result.qualification.reject_endpoint),
                "reject_clearance": int(result.qualification.reject_clearance),
                "reject_different_component":
                    int(result.qualification.reject_different_component),
                "reject_global_route":
                    int(result.qualification.reject_global_route),
                "reject_global_astar_budget":
                    int(result.qualification.reject_global_astar_budget),
                "reject_left_infeasible":
                    int(result.qualification.reject_left_infeasible),
                "reject_right_infeasible":
                    int(result.qualification.reject_right_infeasible),
                "reject_both_sides_required":
                    int(result.qualification.reject_both_sides_required),
                "reject_side_search_budget":
                    int(result.qualification.reject_side_search_budget),
                "reject_geom_mismatch":
                    int(result.qualification.reject_geom_mismatch),
                "total_astar_expansions":
                    int(result.qualification.total_astar_expansions),
            },
            "round_logs": [
                {
                    "round": int(r.round),
                    "scenes_generated": int(r.scenes_generated),
                    "scenes_valid": int(r.scenes_valid),
                    "task_candidates": int(r.task_candidates),
                    "cheap_rejected": int(r.cheap_rejected),
                    "preflight_attempted": int(r.preflight_attempted),
                    "preflight_success": int(r.preflight_success),
                    "selected_pool": int(r.selected_pool),
                    "elapsed_ms": _fin(r.elapsed_ms),
                    "preflight_avg_ms": _fin(r.preflight_avg_ms),
                    "failure_breakdown": {
                        str(k): int(v)
                        for k, v in r.failure_breakdown.items()
                    },
                    "qualification": {
                        "candidates_checked":
                            int(r.qualification.candidates_checked),
                        "endpoint_pass": int(r.qualification.endpoint_pass),
                        "connectivity_pass":
                            int(r.qualification.connectivity_pass),
                        "straight_clear": int(r.qualification.straight_clear),
                        "blocked": int(r.qualification.blocked),
                        "side_qualification_attempt":
                            int(r.qualification.side_qualification_attempt),
                        "both_sides_feasible":
                            int(r.qualification.both_sides_feasible),
                        "accepted": int(r.qualification.accepted),
                        "reject_endpoint":
                            int(r.qualification.reject_endpoint),
                        "reject_different_component":
                            int(r.qualification.reject_different_component),
                        "reject_global_route":
                            int(r.qualification.reject_global_route),
                        "reject_global_astar_budget":
                            int(r.qualification.reject_global_astar_budget),
                        "reject_both_sides_required":
                            int(r.qualification.reject_both_sides_required),
                        "total_astar_expansions":
                            int(r.qualification.total_astar_expansions),
                    },
                    "remaining_deficits": [str(d) for d in r.remaining_deficits],
                }
                for r in result.round_logs
            ],
            # ── NEW: distribution report + iteration summary ───────
            "generation_rounds": int(result.generation_rounds),
            "hard_minimums_met": bool(result.hard_minimums_met),
            "soft_targets_met": bool(result.soft_targets_met),
            "distribution_counts": {
                str(k): int(v) for k, v in result.distribution_counts.items()},
            "distribution_histograms": {
                str(k): [int(v) for v in hist]
                for k, hist in result.distribution_histograms.items()},
            "remaining_deficits": [str(d) for d in result.remaining_deficits],
            "warnings": [str(w) for w in result.warnings],
            "timing_ms": {str(k): float(v) for k, v in result.timing_ms.items()},
            "selected_scene_ids": [int(s) for s in result.selected_scene_ids],
            "scenes": scenes,
            "tasks": [_task_dict(t) for t in result.tasks],
            # The full preflight-accepted candidate pool, so a later
            # collection-only run (manifest_file) can still top-up gaps.
            "preflighted": [_task_dict(t) for t in result.preflighted],
            "note": (
                "Manifest stores ONLY original start/goal/initial_yaw, scene "
                "obstacles, seed, C++ behavior classification, the "
                "judge-only preflight audit and the per-task distribution "
                "summary (depth proxy / 5 Hz / 30 Hz histograms).  NO "
                "future 5 Hz correction sequence and NO local-target "
                "stream is stored.  generation_ok is false only when a "
                "HARD distribution minimum is unmet; soft-target shortfalls "
                "appear in remaining_deficits / warnings."),
        }
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return manifest

    # ═══════════════════════════════════════════════════════════════
    #  Collection-only mode: load a previously generated task manifest
    # ═══════════════════════════════════════════════════════════════
    def _load_collection_manifest(self, path):
        """Rebuild the collection plan (tasks / scenes / preflighted pool)
        from a previously written blueprint manifest.

        Returns a SimpleNamespace with `tasks`, `scenes` and `preflighted`
        objects exposing the SAME field names as the pybind BlueprintTask /
        BlueprintScene, so the collection loop and the committed top-up run
        unchanged.  The candidate pool (preflighted) carries the C++
        preflight distribution summaries (histograms as count lists).
        """
        class HistLite(object):
            def __init__(self, counts):
                self.counts = list(counts or [])
                self.edges = list(range(len(self.counts) + 1))

            def total(self):
                return float(sum(self.counts))

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        manifest_revision = str(payload.get("expert_revision", ""))
        runtime_revision = str(getattr(expert_mod, "EXPERT_REVISION", ""))
        # R26: record the manifest revision so every episode metadata can
        # distinguish the RUNNING expert from the (possibly older) preflight
        # expert whose preflight statistics are diagnostic only.
        self._manifest_expert_revision = manifest_revision
        self._runtime_expert_revision = runtime_revision
        if not manifest_revision:
            rospy.logwarn(
                "[Manager] loaded manifest has no expert_revision; its "
                "preflight acceptance may use stale planner behaviour")
        elif manifest_revision != runtime_revision:
            rospy.logwarn(
                "[Manager] manifest expert revision %s != runtime %s; "
                "regenerate blueprint before production collection",
                manifest_revision, runtime_revision)

        scenes = []
        for sd in payload.get("scenes", []):
            s = SimpleNamespace()
            s.scene_id = int(sd["scene_id"])
            s.obstacles = []
            for o in sd.get("obstacles", []):
                nobj = SimpleNamespace(
                    id=int(o["id"]), x=float(o["x"]), y=float(o["y"]),
                    radius=float(o.get("radius", 0.0)),
                    height_m=float(o["height_m"]))
                # 矩形障碍(可选 w/h):存在时按 AABB 渲染(size=[w,h,h]),
                # 否则按圆柱(radius)渲染。
                nobj.w = (float(o["w"]) if o.get("w") is not None
                          else None)
                nobj.h = (float(o["h"]) if o.get("h") is not None
                          else None)
                s.obstacles.append(nobj)
            scenes.append(s)

        def _task(td):
            t = SimpleNamespace()
            t.task_id = int(td["task_id"])
            t.scene_id = int(td["scene_id"])
            t.seed = int(td.get("seed", 0))
            t.start_x = float(td["start"][0])
            t.start_y = float(td["start"][1])
            t.goal_x = float(td["goal"][0])
            t.goal_y = float(td["goal"][1])
            t.initial_yaw = float(td["initial_yaw"])
            t.flight_height_m = float(td["flight_height_m"])
            t.behavior_class = str(td.get("behavior_class", "clear"))
            t.density_class = str(td.get("density_class", "medium"))
            t.radius_class = str(td.get("radius_class", "medium"))
            t.distance_class = str(td.get("distance_class", "medium"))
            t.side_class = str(td.get("side_class", "none"))
            ad = td.get("audit", {}) or {}
            t.audit = SimpleNamespace(
                accepted=bool(ad.get("accepted", True)),
                truth_brake_triggered=bool(
                    ad.get("truth_brake_triggered", False)),
                preflight_ticks=int(ad.get("preflight_ticks", 0)),
                min_truth_clearance_m=float(
                    ad.get("min_truth_clearance_m", 0.0)),
                goal_distance_m=float(ad.get("goal_distance_m", 0.0)),
                preflight_status=str(ad.get("preflight_status", "loaded")))
            sm = td.get("summary", {}) or {}
            m5 = sm.get("macro5hz", {}) or {}
            l30 = sm.get("local30hz", {}) or {}
            s = SimpleNamespace()
            s.macro_tick_total = int(m5.get("tick_total", 0))
            s.macro_pass_count = int(m5.get("pass_count", 0))
            s.macro_normal_count = int(m5.get("normal_count", 0))
            s.macro_turn_left_count = int(m5.get("turn_left_count", 0))
            s.macro_turn_right_count = int(m5.get("turn_right_count", 0))
            s.local_direct_count = int(l30.get("direct_count", 0))
            s.local_avoidance_count = int(l30.get("avoidance_count", 0))
            s.macro_correction_angle_hist = HistLite(
                m5.get("correction_angle_hist", []))
            s.macro_correction_distance_hist = HistLite(
                m5.get("correction_distance_hist", []))
            s.local_deflection_hist = HistLite(
                l30.get("deflection_hist", []))
            s.local_yaw_rate_hist = HistLite(
                l30.get("yaw_rate_hist", []))
            s.local_speed_hist = HistLite(
                l30.get("speed_hist", []))
            t.summary = s
            return t

        tasks = [_task(td) for td in payload.get("tasks", [])]
        if not tasks:
            raise ValueError(
                "manifest %s contains no tasks" % path)
        preflighted = [_task(td) for td in payload.get("preflighted", [])]
        return SimpleNamespace(tasks=tasks, scenes=scenes,
                               preflighted=preflighted)

    # ═══════════════════════════════════════════════════════════════
    #  Unity binary depth decode (item 一)
    # ═══════════════════════════════════════════════════════════════
    def _decode_depth(self, merged, raw_parts):
        """Decode a Unity depth frame from `raw_parts`.

        `UnityBridge.try_recv()` returns (merged, raw_parts); the binary
        depth is one of `raw_parts` (float32, width*height*4 bytes) and the
        AvoidBench convention is `depth_m = flipud(raw * 100.0)`.

        Returns (depth_m, raw_finite_ratio, frame_id, collision, error).
        On ANY failure depth_m is None and error is a clear reason — the
        caller must NEVER fabricate an all-zero frame and continue.
        """
        w = int(self._depth_cfg.get("width", 640))
        h = int(self._depth_cfg.get("height", 360))
        expected_len = w * h * 4  # float32

        frame_id = merged.get("pub_frame_id")
        if frame_id is None:
            frame_id = merged.get("frame_id")
        collision = bool(merged.get("collision", False))
        if not collision:
            vehicles = merged.get("pub_vehicles") or \
                merged.get("vehicles") or []
            if vehicles:
                collision = bool(vehicles[0].get("collision", False))
        if frame_id is None:
            return None, 0.0, None, collision, "frame_id_missing"
        frame_id = int(frame_id)

        if not raw_parts:
            return None, 0.0, frame_id, collision, "payload_missing"
        part = None
        for p in raw_parts:
            if len(p) >= expected_len:
                part = p
                break
        if part is None:
            return None, 0.0, frame_id, collision, "payload_wrong_length"
        try:
            arr = np.frombuffer(part[:expected_len], dtype=np.float32)
            arr = arr.reshape((h, w))
        except (ValueError, TypeError) as e:
            return None, 0.0, frame_id, collision, \
                "payload_shape_error_%s" % type(e).__name__
        # AvoidBench convention: depth_m = flipud(raw * 100.0).
        depth_m = np.flipud(arr.astype(np.float64) * 100.0)
        valid = np.isfinite(depth_m) & (depth_m > 0.01)
        ratio = float(valid.mean()) if valid.size else 0.0
        if not np.any(valid):
            # All-zero / all-invalid is a payload failure, NEVER "free".
            return None, ratio, frame_id, collision, "no_valid_depth"
        return depth_m, ratio, frame_id, collision, ""

    def _apply_depth_noise(self, depth_m):
        """D435i stereo-depth noise at the COLLECTOR (kept for legacy/debug
        only).  The production pipeline stores CLEAN depth and applies the
        D435i noise at TRAINING time instead (train.py
        --depth-noise-std-ratio), so this is disabled by default
        (noise_std_ratio: 0.0 in the YAML).  When enabled it injects a
        multiplicative Gaussian sigma = noise_std_ratio * depth (m) AFTER
        decode and BEFORE the expert step AND the PNG encode, so labels
        stay self-consistent with the noised student input.  Invalid
        (<=0 / non-finite) pixels stay invalid; results clip to [0, max_m]."""
        ratio = float(self._depth_cfg.get("noise_std_ratio", 0.0))
        if ratio <= 0.0:
            return depth_m
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if not np.any(valid):
            return depth_m
        max_m = float(self._depth_cfg.get("max_m", 5.0))
        sigma = ratio * np.abs(depth_m)
        noisy = depth_m + self._noise_rng.normal(
            0.0, 1.0, depth_m.shape).astype(np.float64) * sigma
        noisy = np.clip(noisy, 0.0, max_m)
        noisy[~valid] = depth_m[~valid]
        return noisy

    # ═══════════════════════════════════════════════════════════════
    #  Unity / Flightmare lifecycle
    # ═══════════════════════════════════════════════════════════════
    def _connect(self):
        pub_port = str(self._g.get("pub_port", "10253"))
        sub_port = str(self._g.get("sub_port", "10254"))
        self._bridge = il_common.UnityBridge(pub_port, sub_port)
        self._bridge.bind()
        scene_id = int(self._g.get("scene_id", 1))
        timeout = float(self._g.get("fsm", {}).get("connect_timeout", 60.0))
        if not self._bridge.connect_handshake(
                scene_id, self._depth_cfg, timeout=timeout):
            raise RuntimeError("AvoidBench handshake failed (scene id=%d)" %
                               scene_id)
        rospy.loginfo("[Manager] AvoidBench ready")

    def _create_dynamics(self):
        import il_dynamics
        self._dynamics = il_dynamics.create_dynamics_backend(self._config)
        return self._dynamics

    def _flush_bridge(self):
        """Drain every pending Unity response (scene/task change)."""
        if self._bridge is None:
            return
        while True:
            r = self._bridge.try_recv()
            if r is None:
                break

    def _send_scene_to_unity(self, scene):
        """Send one blueprint scene; retire a previous scene's obstacles via
        build_replacing_object_update() so no stale object survives."""
        if self._bridge is None:
            return
        objects = []
        for o in scene.obstacles:
            is_rect = getattr(o, "w", None) is not None and \
                getattr(o, "h", None) is not None
            if is_rect:
                # AABB box via the Transparen_Cube prefab (benchmark format).
                # Position = [world x, centre height, world y], size =
                # [x-width, height, y-length].  The generator sets the side =
                # 2*r so the AABB faces sit on the circumscribed circle along
                # the axes — depth reads the TRUE collision distance and the
                # expert detours instead of flying into the circle ring.
                objects.append({
                    "ID": "cyl_s%d_%d" % (int(scene.scene_id), int(o.id)),
                    "prefabID": "Transparen_Cube",
                    "position": [float(o.x), float(self._flight_height),
                                 float(o.y)],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "size": [float(o.w), float(o.height_m), float(o.h)],
                })
            else:
                objects.append({
                    "ID": "cyl_s%d_%d" % (int(scene.scene_id), int(o.id)),
                    "prefabID": "Object",
                    # Unity coords [x, y(up), z]: x_fwd -> x, y_left -> z.
                    "position": [float(o.x), float(self._flight_height),
                                 float(o.y)],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "size": [float(o.radius) * 2.0, float(o.height_m),
                             float(o.radius) * 2.0],
                })
        wire_objects, retired = il_common.build_replacing_object_update(
            objects, self._known_object_ids)
        # Remember the scene objects so every following pose re-sends them.
        self._current_unity_objects = list(wire_objects)
        self._bridge.send_pose({
            "scene_id": int(self._g.get("scene_id", 1)),
            "vehicles": [
                il_common.make_depth_vehicle(
                    [0.0, 0.0, self._flight_height], 0.0,
                    self._depth_cfg)],
            "objects": wire_objects,
        })
        settle = float(self._g.get("scene_runtime", {}).get(
            "settle_time_s", 8.0))
        rospy.sleep(settle)
        # Discard handshake / scene-settle depth — the first episode frame
        # must never consume it.
        self._flush_bridge()
        # Warm-up render (item): after scene objects are (re)placed, Unity's
        # FIRST render can take far longer than the per-frame timeout, which
        # otherwise rejects the FIRST episode of a scene at control tick 0
        # (frame_retries_exceeded / all attempts unmatched).  Actively
        # request one frame and confirm a matching response before the
        # episode starts; the warm-up frame is discarded.
        #
        # R29d: keep retrying until a matching frame actually arrives
        # (bounded by a total warm-up budget, small pause between attempts)
        # instead of giving up after 3 quick tries — a fresh scene's first
        # render often needs longer, and starting the first episode before
        # the render pipeline is aligned reliably trips
        # unmatched_render_rate (measured: first episode of a new scene at
        # ~9% unmatched, 3/33, even though the 8 s settle had run).
        warm_ok = False
        warmup_budget_s = float(self._g.get("scene_runtime", {}).get(
            "warmup_budget_s", 15.0))
        warmup_deadline = time.time() + max(1.0, warmup_budget_s)
        warm_attempts = 0
        while time.time() < warmup_deadline and not warm_ok:
            warm_attempts += 1
            warm_id = self._next_frame_id
            self._next_frame_id += 1
            self._bridge.send_pose(self._pose_message(
                [0.0, 0.0, self._flight_height], None, warm_id))
            m = self._wait_for_matching_frame(warm_id, 3.0)
            if m is not None:
                warm_ok = True
                break
            # Brief pause before retrying so Unity can finish the (slow)
            # first render of the freshly loaded scene.
            rospy.sleep(0.5)
        if not warm_ok:
            rospy.logwarn(
                "[Manager] scene %d: warm-up render did not return a "
                "matching frame within the warm-up budget (%.1f s, %d "
                "attempts); the first episode of this scene may be rejected",
                int(scene.scene_id), warmup_budget_s, warm_attempts)
        if retired:
            rospy.loginfo("[Manager] scene %d: retired %d stale objects",
                          int(scene.scene_id), retired)

    def _pose_message(self, pos, q, frame_id):
        """Unity Pose message that CARRIES the render frame_id.

        `q` is the quaternion_xyzw, or None to use the yaw-0 identity
        default of make_depth_vehicle (used by the scene warm-up render).
        """
        if q is None:
            vehicle = il_common.make_depth_vehicle(
                [float(pos[0]), float(pos[1]), float(pos[2])],
                0.0, self._depth_cfg)
        else:
            vehicle = il_common.make_depth_vehicle(
                [float(pos[0]), float(pos[1]), float(pos[2])],
                0.0, self._depth_cfg,
                quaternion_xyzw=[float(q[0]), float(q[1]),
                                 float(q[2]), float(q[3])])
        return {
            "scene_id": int(self._g.get("scene_id", 1)),
            "frame_id": int(frame_id),
            "vehicles": [vehicle],
            # Always re-send the current scene objects: the AvoidBench
            # binary drops obstacles not present in a pose message.
            "objects": list(self._current_unity_objects),
        }

    @staticmethod
    def _yaw_from_xyzw(q):
        return math.atan2(
            2.0 * (q[3] * q[2] + q[0] * q[1]),
            1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))

    # ═══════════════════════════════════════════════════════════════
    #  Strict frame synchronised episode (item 二)
    # ═══════════════════════════════════════════════════════════════
    def _wait_for_matching_frame(self, frame_id, timeout_s):
        """Wait for a Unity response whose pub_frame_id/frame_id equals the
        render frame we sent.  Stale responses are counted and discarded.
        Returns (merged, raw_parts) or None on timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline and not rospy.is_shutdown():
            r = self._bridge.try_recv()
            if r is None:
                time.sleep(0.001)
                continue
            merged, raw_parts = r
            fid = merged.get("pub_frame_id")
            if fid is None:
                fid = merged.get("frame_id")
            if fid is None:
                self._stale_responses += 1
                continue
            if int(fid) == int(frame_id):
                return merged, raw_parts
            self._stale_responses += 1
        return None

    def _episode(self, scene, task, writer, tick_base):
        """Run one episode with STRICT frame synchronisation.

        Returns (committed, reason, audit_summary).
        """
        episode_id = writer.episode_id
        self._current_episode_id = episode_id
        start = [float(task.start_x), float(task.start_y)]
        goal = [float(task.goal_x), float(task.goal_y)]
        initial_yaw = float(task.initial_yaw)
        flight_h = float(task.flight_height_m)
        scene_id = int(scene.scene_id)
        task_id = int(task.task_id)

        # Configure the exact-cylinder truth audit for this scene.
        obstacles = [[float(o.x), float(o.y), float(o.radius),
                      float(o.height_m)] for o in scene.obstacles]
        self._truth.configure(
            obstacles, float(self._vehicle_radius),
            [float(self._region_min[0]), float(self._region_min[1])],
            [float(self._region_max[0]), float(self._region_max[1])])

        # Reset the C++ expert for the new episode (fresh FSM / history).
        self._expert.reset_task(start, goal, initial_yaw, tick_base,
                                flight_h)
        if self._dynamics is not None:
            self._dynamics.reset(
                np.asarray(start + [flight_h], dtype=np.float64),
                initial_yaw)
        # Task reset: discard any stale Unity response.
        self._flush_bridge()

        audit = EpisodeAudit()
        self._stale_responses = 0
        self._episode_rows = []       # buffered row dicts (small)
        self._episode_start_wall = time.time()
        committed = False
        reason = ""
        control_tick = 0  # data.csv episode_frame_index (committed only)
        dt = 1.0 / self._control_hz

        # ── R26 judge-only quality-watchdog state ──────────────────
        goal_wd_x, goal_wd_y = float(goal[0]), float(goal[1])
        wd_last_goal_dist = math.hypot(
            float(start[0]) - goal_wd_x, float(start[1]) - goal_wd_y)
        wd_no_progress_ticks = 0
        wd_near_goal_ticks = 0
        wd_cum_yaw = 0.0
        wd_last_yaw = None

        try:
            while not rospy.is_shutdown():
                tick = tick_base + control_tick

                # 1. Read the current dynamics state.
                try:
                    state = self._dynamics.get_state()
                except Exception as e:  # noqa: BLE001
                    reason = "state_read_failed_%s" % type(e).__name__
                    audit.command_rejected = True
                    rospy.logwarn("[Manager] episode %s: state read failed: %s",
                                  episode_id, e)
                    break
                pos = state.position_world
                q = state.quaternion_world_body
                vel = state.velocity_world
                yaw = self._yaw_from_xyzw(q)
                yaw_rate = float(state.angular_velocity_body[2])
                sim_t = float(state.simulation_time_s)

                # ── R26 judge-only quality watchdog (never student
                #    inputs; rejects pathological spin/stop/replan loops
                #    such as task 65: 0.48 m for 36 s, 2458° of yaw). ──
                wd_goal_dist = math.hypot(
                    float(pos[0]) - goal_wd_x, float(pos[1]) - goal_wd_y)
                if wd_last_yaw is not None:
                    wd_dy = yaw - wd_last_yaw
                    while wd_dy > math.pi:
                        wd_dy -= 2.0 * math.pi
                    while wd_dy < -math.pi:
                        wd_dy += 2.0 * math.pi
                    wd_cum_yaw += abs(wd_dy)
                wd_last_yaw = yaw
                # A moving drone counts as progress even when the straight
                # goal distance is temporarily stagnant (legitimate lateral
                # detour around an obstacle).  Only a STATIONARY drone with
                # no goal progress is a true stall / spin loop.
                wd_speed = math.hypot(float(vel[0]), float(vel[1]))
                if wd_goal_dist <= wd_last_goal_dist - \
                        self._goal_no_progress_min_m:
                    wd_no_progress_ticks = 0
                    wd_last_goal_dist = wd_goal_dist
                elif wd_speed < 0.15:
                    wd_no_progress_ticks += 1
                else:
                    wd_no_progress_ticks = 0
                if wd_goal_dist <= self._near_goal_radius_m:
                    wd_near_goal_ticks += 1
                else:
                    wd_near_goal_ticks = 0
                if wd_no_progress_ticks >= self._goal_no_progress_ticks and \
                        wd_goal_dist > self._goal_no_progress_floor_m:
                    reason = "goal_no_progress"
                    audit.goal_no_progress = True
                    rospy.logwarn(
                        "[Manager] episode %s: stationary with no goal "
                        "progress for %d ticks at %.2f m; rejecting",
                        episode_id, wd_no_progress_ticks, wd_goal_dist)
                    break
                if wd_near_goal_ticks >= self._near_goal_timeout_ticks:
                    reason = "near_goal_timeout"
                    audit.near_goal_timeout = True
                    rospy.logwarn(
                        "[Manager] episode %s: inside %.2f m of the goal "
                        "for %d ticks; rejecting",
                        episode_id, self._near_goal_radius_m,
                        wd_near_goal_ticks)
                    break
                if wd_cum_yaw > self._max_cumulative_yaw_rad and \
                        control_tick > 30:
                    reason = "excessive_yaw"
                    audit.excessive_yaw = True
                    rospy.logwarn(
                        "[Manager] episode %s: cumulative yaw %.1f deg "
                        "exceeds cap; rejecting",
                        episode_id, math.degrees(wd_cum_yaw))
                    break

                # 2. Strict render-attempt loop on the SAME saved state:
                #    send a NEW render frame_id until an exact-match valid
                #    depth arrives, or max_frame_retries is exceeded.  A
                #    failed attempt NEVER advances dynamics, NEVER calls the
                #    expert and NEVER writes a row (item 七).
                matched = None
                for attempt in range(self._max_frame_retries + 1):
                    frame_id = self._next_frame_id
                    self._next_frame_id += 1
                    audit.render_attempts += 1
                    send_t = time.time()
                    self._bridge.send_pose(
                        self._pose_message(pos, q, frame_id))
                    m = self._wait_for_matching_frame(
                        frame_id, self._frame_match_timeout_s)
                    if m is None:
                        audit.unmatched += 1
                        writer.write_sync(frame_id, -1.0, "no_match", 1,
                                          audit.matched, audit.unmatched)
                        continue
                    merged, raw_parts = m
                    recv_t = time.time()
                    depth_m, raw_ratio, matched_fid, unity_collision, \
                        depth_err = self._decode_depth(merged, raw_parts)
                    latency_ms = (recv_t - send_t) * 1000.0
                    if depth_err:
                        # Payload missing / wrong length / shape / no valid
                        # depth: NEVER fabricate zeros, NEVER call the expert.
                        audit.none_depth += 1
                        audit.parse_failures += 1
                        writer.write_sync(frame_id, latency_ms,
                                          "frame_id_exact_%s" % depth_err, 1,
                                          audit.matched, audit.unmatched)
                        continue
                    matched = (frame_id, depth_m, raw_ratio, unity_collision,
                               latency_ms, recv_t, send_t)
                    break
                if matched is None:
                    audit.frame_retries_exceeded = True
                    reason = "frame_retries_exceeded"
                    rospy.logwarn(
                        "[Manager] episode %s: no valid depth after %d "
                        "render attempts on control tick %d; rejecting",
                        episode_id, self._max_frame_retries + 1, control_tick)
                    break

                frame_id, depth_m, raw_ratio, unity_collision, latency_ms, \
                    recv_t, send_t = matched
                audit.matched += 1
                # D435i sim-to-real: inject stereo-depth noise on the SAME
                # depth_m that feeds the expert AND the PNG (labels stay
                # self-consistent with the noised input).
                depth_m = self._apply_depth_noise(depth_m)
                if latency_ms > self._max_latency_ms:
                    audit.latency_violations += 1
                if latency_ms > self._catastrophic_latency_ms:
                    audit.catastrophic_latency = True
                if unity_collision:
                    audit.unity_collision = True

                # ── Judge-only point checks at the current state. ──
                truth_point_col = self._truth.segment_collision(
                    float(pos[0]), float(pos[1]),
                    float(pos[0]), float(pos[1]))
                truth_oob = self._truth.point_out_of_bounds(
                    float(pos[0]), float(pos[1]),
                    float(self._vehicle_radius))
                audit.min_truth_clearance_m = min(
                    audit.min_truth_clearance_m,
                    self._truth.segment_min_clearance(
                        float(pos[0]), float(pos[1]),
                        float(pos[0]), float(pos[1])))
                if truth_oob:
                    audit.out_of_bounds = True
                collision_flag = (unity_collision or truth_point_col or
                                  truth_oob)

                # 3. Expert step with the SAVED state + matched depth.
                out = self._expert.step(
                    [float(pos[0]), float(pos[1]), float(pos[2])], yaw,
                    [float(vel[0]), float(vel[1]), float(vel[2])], yaw_rate,
                    np.ascontiguousarray(depth_m, dtype=np.float32).ravel(),
                    int(self._depth_cfg.get("width", 640)),
                    int(self._depth_cfg.get("height", 360)),
                    [float(pos[0]), float(pos[1]), float(pos[2])],
                    [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                    flight_h, int(tick), collision_flag)

                # 4. 5 Hz label legality on a real update frame.
                if out.macro_update_mask and out.macro_label_valid != 1:
                    audit.macro_label_invalid = True
                    reason = "macro_label_invalid"
                    rospy.logwarn(
                        "[Manager] episode %s: invalid 5 Hz label at frame %d"
                        % (episode_id, control_tick))
                    break

                # 5. The 3D expert commands the vertical channel itself
                #    (C++ VerticalController / CommandComposer3D regulate
                #    the altitude toward the mission goal z).  The FINAL
                #    command is the full 3D BODY/FLU velocity + yaw rate —
                #    no Python-side altitude merge.  The backend still
                #    clamps to its own physical limits.
                final_vel = np.array([float(out.target_velocity_flu_x),
                                      float(out.target_velocity_flu_y),
                                      float(out.target_velocity_flu_z)],
                                     dtype=np.float64)
                final_yaw_rate = float(out.target_yaw_rate)

                # 6. Non-finite input/label guard.
                label_values = [
                    out.goal_direction_flu_x, out.goal_direction_flu_y,
                    out.goal_direction_flu_z, out.goal_distance_norm,
                    out.target_velocity_flu_x, out.target_velocity_flu_y,
                    out.target_velocity_flu_z, out.target_yaw_rate,
                    out.navigation_goal_direction_flu_x,
                    out.navigation_goal_direction_flu_y,
                    out.navigation_goal_direction_flu_z,
                    out.navigation_goal_distance_norm,
                    out.macro_distance_norm,
                ]
                if not all(np.isfinite(float(v)) for v in label_values):
                    audit.non_finite_label = True
                    reason = "non_finite_label"
                    rospy.logwarn(
                        "[Manager] episode %s: non-finite label at frame %d"
                        % (episode_id, control_tick))
                    break

                # 7. Execute the expert command (recorded label).  If the
                # dynamics REJECTS it, the episode is rejected immediately
                # and the command is never recorded as executed.
                if not self._dynamics.step_velocity_command(
                        final_vel, final_yaw_rate, dt):
                    reason = "command_rejected"
                    audit.command_rejected = True
                    rospy.logwarn(
                        "[Manager] episode %s: dynamics rejected the command "
                        "at frame %d" % (episode_id, control_tick))
                    break

                # 8. Read the RESULTING state and audit the EXECUTED
                #    segment pos → pos_after continuously (item 八): every
                #    dynamics step is covered; the final step before a
                #    terminal is audited too.
                try:
                    state_after = self._dynamics.get_state()
                except Exception as e:  # noqa: BLE001
                    reason = "state_read_failed_after_%s" % type(e).__name__
                    audit.command_rejected = True
                    break
                pos_after = state_after.position_world
                audit.note_state_segment(
                    float(pos[0]), float(pos[1]),
                    float(pos_after[0]), float(pos_after[1]), self._truth)
                if self._truth.segment_crosses_bounds(
                        float(pos[0]), float(pos[1]),
                        float(pos_after[0]), float(pos_after[1]),
                        float(self._vehicle_radius)):
                    audit.out_of_bounds = True

                # 9. Write the CSV row with the SAME state/depth/command.
                row = self._build_row(
                    writer, tick, control_tick, out, state, yaw, yaw_rate,
                    depth_m, raw_ratio, final_vel, final_yaw_rate, pos, q,
                    scene_id, task_id, goal, flight_h, frame_id, send_t,
                    recv_t, latency_ms, unity_collision, audit)
                writer.write_depth(int(frame_id), depth_m, float(raw_ratio))
                row["depth_file"] = writer.depth_file_for(int(frame_id))
                self._episode_rows.append(row)
                writer.write_sync(frame_id, latency_ms, "frame_id_exact", 0,
                                  audit.matched, audit.unmatched)

                control_tick += 1
                if out.terminal:
                    audit.terminal_state = str(out.fsm_state)
                    # R26: measure the terminal speed at the DECISION state
                    # (state read at this tick's step 1 — the same state the
                    # FSM's goalReached() gated on), NOT after executing the
                    # terminal command.  The post-command state can sit a few
                    # mm/s ABOVE the goal-stop threshold (measured 0.203-0.208
                    # vs 0.20) while the FSM legitimately declared
                    # GOAL_REACHED, causing 3 false terminal_speed rejects.
                    audit.final_speed_mps = float(
                        np.linalg.norm(state.velocity_world))
                    audit.final_yaw_rate_rps = float(
                        state.angular_velocity_body[2])
                    if out.fsm_state == "GOAL_REACHED":
                        audit.reached_goal = True
                        committed = True
                        reason = "goal_reached"
                    else:
                        reason = "terminal_" + out.fsm_state.lower()
                    break
                # Wall-clock watchdog ONLY when enabled (0 disables it).
                if self._wall_timeout_s > 0.0 and \
                        time.time() - self._episode_start_wall > \
                        self._wall_timeout_s:
                    reason = "trajectory_wall_timeout"
                    break
        except Exception as e:  # noqa: BLE001
            reason = "exception_%s" % type(e).__name__
            audit.command_rejected = True
            rospy.logwarn("[Manager] episode %s aborted: %s", episode_id, e)
            traceback.print_exc()

        # ── Commit decision (item 七): every condition must pass. ────
        success, taxonomy, failure = self._commit_decision(
            audit, control_tick, goal)
        if not success:
            committed = False
            reason = taxonomy if taxonomy else reason
        audit_summary = {
            "render_attempts": audit.render_attempts,
            "matched": audit.matched,
            "unmatched": audit.unmatched,
            "none_depth": audit.none_depth,
            "stale": audit.stale + self._stale_responses,
            "parse_failures": audit.parse_failures,
            "latency_violations": audit.latency_violations,
            "catastrophic_latency": audit.catastrophic_latency,
            "unity_collision": audit.unity_collision,
            "frame_retries_exceeded": audit.frame_retries_exceeded,
            "macro_label_invalid": audit.macro_label_invalid,
            "non_finite_label": audit.non_finite_label,
            "command_rejected": audit.command_rejected,
            "truth_swept_collision": audit.truth_swept_collision,
            "out_of_bounds": audit.out_of_bounds,
            "truth_brake_triggered": audit.truth_brake_triggered,
            "min_truth_clearance_m": audit.min_truth_clearance_m,
            "max_observed_brake_risk": audit.max_observed_brake_risk,
            "state_segments": audit.state_segments,
            "terminal_state": audit.terminal_state,
            "reached_goal": audit.reached_goal,
            "final_speed_mps": audit.final_speed_mps,
            "final_yaw_rate_rps": audit.final_yaw_rate_rps,
        }

        # ── Flush the buffered rows with the REAL per-episode fields. ─
        ep_valid = 1 if success else 0
        for row in self._episode_rows:
            row["episode_valid"] = ep_valid
            row["failure_taxonomy"] = taxonomy
            row["failure_reason"] = failure
            writer.write_row(row)
        return committed, reason, audit_summary

    def _commit_decision(self, audit, control_ticks, goal):
        """Real commit gate.  Returns (success, taxonomy, reason).

        Render-attempt statistics (unmatched / none-depth) are expressed
        against audit.render_attempts (the sync.csv cadence); the latency /
        terminal checks are expressed against the committed control ticks
        (data.csv cadence).  Every condition must pass (item 七).
        """
        if audit.unity_collision:
            return False, "unity_collision", \
                "Unity reported a vehicle collision"
        if audit.truth_swept_collision:
            return False, "truth_collision", \
                "continuous truth swept collision on an executed segment"
        if audit.out_of_bounds:
            return False, "out_of_bounds", \
                "the drone disk crossed the configured region boundary"
        if audit.truth_brake_triggered:
            return False, "truth_brake_would_trigger", \
                "truth brake would have triggered before an obstacle"
        if audit.non_finite_label:
            return False, "non_finite_label", \
                "a non-finite input/label was produced"
        if audit.macro_label_invalid:
            return False, "macro_label_invalid", \
                "a 5 Hz update frame had an invalid label"
        if audit.command_rejected:
            return False, "command_rejected", \
                "dynamics step_velocity_command returned False"
        if audit.frame_retries_exceeded:
            return False, "frame_retries_exceeded", \
                "no valid depth within the max render retries"
        if audit.catastrophic_latency:
            return False, "catastrophic_latency", \
                "a frame exceeded the catastrophic latency"
        # ── R26 judge-only quality watchdogs (pathological behaviour is
        #    rejected even when the episode would eventually reach the
        #    goal — spin / stop / replan loops are not expert data). ──
        if audit.goal_no_progress:
            return False, "goal_no_progress", \
                "no original-goal progress for the configured window"
        if audit.near_goal_timeout:
            return False, "near_goal_timeout", \
                "inside the near-goal radius for longer than the window"
        if audit.excessive_yaw:
            return False, "excessive_yaw", \
                "cumulative yaw exceeded the cap"
        total = max(1, audit.render_attempts)
        if 100.0 * audit.unmatched / total > self._max_unmatched_pct:
            return False, "unmatched_render_rate", \
                "unmatched render attempts %.1f%% > %.1f%%" % (
                    100.0 * audit.unmatched / total, self._max_unmatched_pct)
        if 100.0 * audit.none_depth / total > self._max_none_depth_pct:
            return False, "none_depth_rate", \
                "none-depth render attempts %.1f%% > %.1f%%" % (
                    100.0 * audit.none_depth / total,
                    self._max_none_depth_pct)
        ctrl_total = max(1, control_ticks)
        if 100.0 * audit.latency_violations / ctrl_total > \
                self._max_latency_violation_pct:
            return False, "latency_violation_rate", \
                "latency violations %.1f%% > %.1f%%" % (
                    100.0 * audit.latency_violations / ctrl_total,
                    self._max_latency_violation_pct)
        # Keep the commit reason diagnostic.  A collision or an externally
        # interrupted run must not be flattened into goal_not_reached.
        if audit.terminal_state == "COLLISION":
            return False, "collision", \
                "expert terminated in COLLISION state"
        if not audit.reached_goal:
            if not audit.terminal_state:
                return False, "episode_interrupted", \
                    "episode ended before an expert terminal state"
            return (False, "goal_not_reached",
                    "episode did not reach the original goal %s "
                    "(terminal=%s)" % (list(goal), audit.terminal_state))
        # Final-state terminal conditions (only meaningful when the episode
        # actually reached a terminal state).
        if audit.final_speed_mps >= 0.0 and \
                audit.final_speed_mps > \
                self._params.vehicle_goal_stop_speed_mps + 1e-6:
            return False, "terminal_speed", \
                "final speed %.3f m/s exceeds the goal-stop threshold" % \
                audit.final_speed_mps
        if audit.final_yaw_rate_rps >= 0.0 and \
                abs(audit.final_yaw_rate_rps) > \
                self._params.lp_turn_exit_max_yaw_rate + 1e-6:
            return False, "terminal_yaw_rate", \
                "final yaw rate %.3f rad/s exceeds the terminal threshold" % \
                abs(audit.final_yaw_rate_rps)
        return True, "", ""

    # ═══════════════════════════════════════════════════════════════
    #  Committed-episode distribution stats + gap-driven top-up (item 十)
    #
    #  After the real Flightmare collection the 5 Hz (PASS / NORMAL /
    #  TURN / direction / distance) and 30 Hz (deflection / speed /
    #  yaw-rate) label distributions are RE-DERIVED from the COMMITTED
    #  episodes ONLY (never from the blueprint preflight prediction).  Any
    #  soft gap caused by real episode rejects is auto top-up-collected
    #  from the blueprint candidate pool (BlueprintResult.preflighted),
    #  ranking tasks by their predicted marginal contribution to the
    #  current gaps.  Depth bands are NOT part of the top-up (their real
    #  distribution would require reading every committed depth PNG).
    # ═══════════════════════════════════════════════════════════════
    def _distribution_cfg(self):
        """Histogram edges / thresholds for the committed label stats.

        Mirrors the C++ blueprint config (il_dataset_config.yaml
        blueprint_generation.task_generation.histograms + requirements).
        """
        bp = self._g.get("blueprint_generation", {}) or {}
        btg = bp.get("task_generation", {}) or {}
        hist = btg.get("histograms", {}) or {}
        req = bp.get("requirements", {}) or {}

        def _edges(key, dflt):
            v = hist.get(key)
            if v:
                return [float(x) for x in v]
            return list(dflt)

        return {
            "correction_angle_edges": _edges(
                "correction_angle_edges_deg",
                [-90.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0,
                 60.0, 90.0]),
            "correction_distance_edges": _edges(
                "correction_distance_edges", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            "deflection_edges": _edges(
                "deflection_edges_deg",
                [-90.0, -60.0, -30.0, -10.0, 10.0, 30.0, 60.0, 90.0]),
            "yaw_rate_edges": _edges(
                "yaw_rate_edges",
                [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]),
            "speed_edges": _edges(
                "speed_edges", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            "min_deflection_speed_mps": float(
                hist.get("min_deflection_speed_mps", 0.10)),
            "mtc": int(req.get("min_macro_ticks_per_class", 24)),
        }

    @staticmethod
    def _hist_bin_of(edges, v):
        """Bin index for v (same clamp semantics as C++ Histogram1D)."""
        if not edges or not math.isfinite(v):
            return -1
        if v <= edges[0]:
            return 0
        if v >= edges[-1]:
            return len(edges) - 2
        lo, hi = 0, len(edges)
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo - 1

    @staticmethod
    def _wrap_angle_deg(a):
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    def _new_committed_stats(self, cfg):
        """Empty committed-label distribution accumulator."""
        def _zeros(edges):
            return [0] * max(0, len(edges) - 1)
        return {
            "counts": {
                "macro:total": 0, "macro:pass": 0, "macro:normal": 0,
                "macro:turn_left": 0, "macro:turn_right": 0,
                "local:direct": 0, "local:avoidance": 0,
            },
            "hists": {
                "macro_correction_angle": _zeros(
                    cfg["correction_angle_edges"]),
                "macro_correction_distance": _zeros(
                    cfg["correction_distance_edges"]),
                "local_deflection": _zeros(cfg["deflection_edges"]),
                "local_yaw_rate": _zeros(cfg["yaw_rate_edges"]),
                "local_speed": _zeros(cfg["speed_edges"]),
            },
            "episodes": 0,
            "rows": 0,
        }

    @staticmethod
    def _hist_add(counts, edges, v):
        b = JointV2Manager._hist_bin_of(edges, v)
        if 0 <= b < len(counts):
            counts[b] += 1

    def _add_committed_episode(self, stats, episode_dir, cfg):
        """Accumulate the label distribution of ONE committed episode.

        Mirrors the C++ preflight summary statistics exactly:
          * 5 Hz (macro_update_mask==1 rows): PASS / NORMAL / TURN_LEFT /
            TURN_RIGHT counts; for NORMAL_CORRECTION with
            target_correction_active: correction angle = effective-target
            direction vs original navigation-goal direction, and the
            normalized distance.
          * 30 Hz (every row): direct / avoidance counts, command speed,
            yaw rate, and deflection = command velocity direction vs the
            effective-target direction (only when speed >= the minimum).
        """
        csv_path = os.path.join(episode_dir, "data.csv")
        if not os.path.isfile(csv_path):
            return 0
        n = 0
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("episode_valid") != "1":
                    continue
                n += 1
                # ── 30 Hz behaviour ────────────────────────────────
                if row.get("hierarchical_mode") == "direct":
                    stats["counts"]["local:direct"] += 1
                if row.get("avoidance_active") == "1":
                    stats["counts"]["local:avoidance"] += 1
                try:
                    vx = float(row.get("target_velocity_flu_x", 0.0))
                    vy = float(row.get("target_velocity_flu_y", 0.0))
                except (TypeError, ValueError):
                    vx = vy = 0.0
                speed = math.hypot(vx, vy)
                self._hist_add(stats["hists"]["local_speed"],
                               cfg["speed_edges"], speed)
                try:
                    yr = float(row.get("target_yaw_rate", 0.0))
                except (TypeError, ValueError):
                    yr = 0.0
                self._hist_add(stats["hists"]["local_yaw_rate"],
                               cfg["yaw_rate_edges"], yr)
                gx = gy = 0.0
                try:
                    gx = float(row.get("goal_direction_flu_x", 0.0))
                    gy = float(row.get("goal_direction_flu_y", 0.0))
                except (TypeError, ValueError):
                    pass
                if speed >= cfg["min_deflection_speed_mps"]:
                    gl = math.hypot(gx, gy)
                    if gl > 1e-6:
                        ang = self._wrap_angle_deg(math.degrees(
                            math.atan2(gx * vy - gy * vx,
                                       gx * vx + gy * vy)))
                        self._hist_add(stats["hists"]["local_deflection"],
                                       cfg["deflection_edges"], ang)
                # ── 5 Hz labels ────────────────────────────────────
                if row.get("macro_update_mask") == "1":
                    stats["counts"]["macro:total"] += 1
                    ct = row.get("macro_correction_type", "")
                    if ct == "PASS_THROUGH":
                        stats["counts"]["macro:pass"] += 1
                    elif ct == "NORMAL_CORRECTION":
                        stats["counts"]["macro:normal"] += 1
                    elif ct == "TURN_LEFT":
                        stats["counts"]["macro:turn_left"] += 1
                    elif ct == "TURN_RIGHT":
                        stats["counts"]["macro:turn_right"] += 1
                    if ct == "NORMAL_CORRECTION" and \
                            row.get("target_correction_active") == "1":
                        ngx = ngy = 0.0
                        try:
                            ngx = float(row.get(
                                "navigation_goal_direction_flu_x", 0.0))
                            ngy = float(row.get(
                                "navigation_goal_direction_flu_y", 0.0))
                        except (TypeError, ValueError):
                            pass
                        ngl = math.hypot(ngx, ngy)
                        el = math.hypot(gx, gy)
                        if ngl > 1e-6 and el > 1e-6:
                            ang = self._wrap_angle_deg(math.degrees(
                                math.atan2(ngx * gy - ngy * gx,
                                           ngx * gx + ngy * gy)))
                            self._hist_add(
                                stats["hists"]["macro_correction_angle"],
                                cfg["correction_angle_edges"], ang)
                        try:
                            dn = float(row.get("macro_distance_norm", 0.0))
                        except (TypeError, ValueError):
                            dn = 0.0
                        if math.isfinite(dn):
                            self._hist_add(
                                stats["hists"][
                                    "macro_correction_distance"],
                                cfg["correction_distance_edges"],
                                max(0.0, min(1.0, dn)))
        stats["episodes"] += 1
        stats["rows"] += n
        return n

    def _gap_targets(self, cfg):
        """Soft distribution targets (replicates the C++ buildDefaultTargets
        target values; depth bands are deliberately excluded).  Returns a
        list of (key, metric, target, weight)."""
        mtc = max(1, cfg["mtc"])
        targets = []

        def _add(key, metric, target, weight):
            targets.append((key, metric, float(target), float(weight)))

        # ── 5 Hz macro coverage ────────────────────────────────────
        _add("macro:total", "count:macro:total", 4.0 * mtc, 0.5)
        _add("macro:pass", "count:macro:pass", 3.0 * mtc, 1.0)
        _add("macro:normal", "count:macro:normal", 3.0 * mtc, 1.0)
        _add("macro:turn_left", "count:macro:turn_left", 3.0 * mtc, 2.0)
        _add("macro:turn_right", "count:macro:turn_right", 3.0 * mtc, 2.0)
        _add("corr_angle_total", "hist_total:macro_correction_angle",
             4.0 * mtc, 1.0)
        for i in range(len(cfg["correction_angle_edges"]) - 1):
            _add("corr_angle:bin%d" % i,
                 "hist_bin:macro_correction_angle:%d" % i, 2.0, 0.8)
        for i in range(len(cfg["correction_distance_edges"]) - 1):
            _add("corr_dist:bin%d" % i,
                 "hist_bin:macro_correction_distance:%d" % i, 2.0, 0.6)
        # ── 30 Hz avoidance coverage ───────────────────────────────
        _add("local:avoidance", "count:local:avoidance", 3.0 * mtc, 1.2)
        _add("local:direct", "count:local:direct", 3.0 * mtc, 0.4)
        _add("deflection_total", "hist_total:local_deflection",
             4.0 * mtc, 1.0)
        n_def = len(cfg["deflection_edges"]) - 1
        for i in range(n_def):
            strong = (i == 0 or i == n_def - 1)
            medium = (i == 1 or i == n_def - 2)
            tgt = 6.0 if strong else (5.0 if medium else 4.0)
            _add("deflection:bin%d" % i,
                 "hist_bin:local_deflection:%d" % i, tgt,
                 1.4 if strong else 1.0)
        # ── 30 Hz speed / yaw-rate bin coverage (soft, uniform) ────
        sp_tgt = max(1, mtc // 2)
        for i in range(len(cfg["speed_edges"]) - 1):
            _add("speed:bin%d" % i, "hist_bin:local_speed:%d" % i,
                 sp_tgt, 0.6)
        for i in range(len(cfg["yaw_rate_edges"]) - 1):
            _add("yaw_rate:bin%d" % i, "hist_bin:local_yaw_rate:%d" % i,
                 sp_tgt, 0.6)
        return targets

    def _achieved_metric(self, stats, metric):
        if metric.startswith("count:"):
            return float(stats["counts"].get(metric[6:], 0))
        if metric.startswith("hist_total:"):
            return float(sum(stats["hists"].get(metric[11:], [])))
        if metric.startswith("hist_bin:"):
            rest = metric[9:]
            name, _, bs = rest.rpartition(":")
            counts = stats["hists"].get(name, [])
            b = int(bs) if bs.isdigit() else -1
            return float(counts[b]) if 0 <= b < len(counts) else 0.0
        return 0.0

    def _evaluate_gaps(self, stats, targets):
        """Soft gaps: {key: (achieved, target, deficit, weight)}."""
        gaps = {}
        for key, metric, target, weight in targets:
            a = self._achieved_metric(stats, metric)
            d = max(0.0, target - a)
            if d > 1e-9:
                gaps[key] = (a, target, d, weight)
        return gaps

    @staticmethod
    def _task_hist(s, name):
        if name == "macro_correction_angle":
            return s.macro_correction_angle_hist
        if name == "macro_correction_distance":
            return s.macro_correction_distance_hist
        if name == "local_deflection":
            return s.local_deflection_hist
        if name == "local_yaw_rate":
            return s.local_yaw_rate_hist
        if name == "local_speed":
            return s.local_speed_hist
        return None

    def _task_contribution(self, task, metric):
        """Predicted marginal contribution of a blueprint task to a metric
        (from its C++ preflight TaskDistributionSummary)."""
        s = task.summary
        if metric.startswith("count:"):
            key = metric[6:]
            m = {
                "macro:total": s.macro_tick_total,
                "macro:pass": s.macro_pass_count,
                "macro:normal": s.macro_normal_count,
                "macro:turn_left": s.macro_turn_left_count,
                "macro:turn_right": s.macro_turn_right_count,
                "local:direct": s.local_direct_count,
                "local:avoidance": s.local_avoidance_count,
            }
            return float(m.get(key, 0.0))
        if metric.startswith("hist_total:"):
            h = self._task_hist(s, metric[11:])
            return float(h.total()) if h is not None else 0.0
        if metric.startswith("hist_bin:"):
            rest = metric[9:]
            name, _, bs = rest.rpartition(":")
            h = self._task_hist(s, name)
            b = int(bs) if bs.isdigit() else -1
            if h is None or not (0 <= b < len(h.counts)):
                return 0.0
            return float(h.counts[b])
        return 0.0

    def _score_task_for_gaps(self, task, stats, targets):
        """Greedy gain score (replicates the C++ scoreTask)."""
        score = 0.0
        for key, metric, target, weight in targets:
            c = self._task_contribution(task, metric)
            if c <= 0.0:
                continue
            cur = self._achieved_metric(stats, metric)
            after = cur + c
            if after <= target + 1e-9:
                gain = c
            elif cur < target:
                gain = max(0.0, target - cur)
            else:
                gain = 0.0
            score += weight * gain
        return score

    def _clear_collection_output(self, output_root):
        """Remove every file/dir under `output_root` EXCEPT the blueprint
        manifest(s) (`*_manifest.json`) and the YAML config(s) (`*.yaml`), so
        each real collection run starts from a clean dataset, never mixes old
        episodes into the new one, and never deletes a test config that sits
        beside the manifest (e.g. il_data_high_test/config.yaml).
        Never touches anything outside `output_root`."""
        if not output_root or not os.path.isdir(output_root):
            return
        kept = set()
        for m in glob.glob(os.path.join(output_root, "joint_v2_blueprint_manifest*.json")):
            kept.add(os.path.abspath(m))
        for m in glob.glob(os.path.join(output_root, "*.yaml")):
            kept.add(os.path.abspath(m))
        for name in sorted(os.listdir(output_root)):
            p = os.path.join(output_root, name)
            if os.path.abspath(p) in kept:
                continue
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
            except OSError:
                pass

    def _collect_task_once(self, scene, task, tick_base):
        """Run ONE real Flightmare episode for a blueprint task.

        Returns (committed, reason, audit_summary, final_dir).  Handles
        the scene switch (once), the writer lifecycle and the episode
        audit; the caller tracks committed directories / used task ids.
        """
        if self._last_scene_id is None or \
                int(scene.scene_id) != self._last_scene_id:
            self._send_scene_to_unity(scene)
            self._last_scene_id = int(scene.scene_id)
        task_id = int(task.task_id)
        # Episode ids carry the scene key prefix so multiple collection runs
        # (indoor / outdoor_high / outdoor_low) can be merged into ONE
        # dataset root without id collisions.
        key_prefix = (self._g.get("scene_generation", {}) or {}).get(
            "scene_key_prefix", "joint_v2")
        episode_id = "%s_%06d_%s" % (
            key_prefix, self._episode_id_counter, uuid.uuid4().hex[:8])
        self._episode_id_counter += 1
        ds_cfg = self._g.get("dataset_logging", {}) or {}
        writer = il_dataset_writer.DatasetWriter(
            ds_cfg, episode_id, self._output_root, int(scene.scene_id),
            task_id,
            [float(task.start_x), float(task.start_y),
             float(task.flight_height_m)],
            [float(task.goal_x), float(task.goal_y),
             float(task.flight_height_m)],
            float(task.initial_yaw), self._depth_cfg,
            control_hz=self._control_hz,
            macro_update_hz=self._macro_update_hz)
        writer.add_metadata({
            "expert_stack_revision": "hierarchical_local_v1",
            "expert_revision": str(getattr(
                expert_mod, "EXPERT_REVISION",
                "<no EXPERT_REVISION in .so — stale build>")),
            # R26: the revision of the blueprint manifest whose preflight
            # statistics are diagnostic only (may differ from the runtime
            # expert revision during behaviour-debug collection).
            "manifest_expert_revision": str(
                getattr(self, "_manifest_expert_revision", "") or ""),
            "schema_extensions": ["two_level_expert_labels_v1"],
            "blueprint_seed": int(task.seed),
            "blueprint_behavior_class": str(task.behavior_class),
            "blueprint_density_class": str(task.density_class),
            "blueprint_radius_class": str(task.radius_class),
            "blueprint_distance_class": str(task.distance_class),
            "blueprint_side_class": str(task.side_class),
            "blueprint_preflight_audit": {
                "accepted": bool(task.audit.accepted),
                "truth_brake_triggered": bool(getattr(
                    task.audit, "truth_brake_triggered", False)),
                "preflight_ticks": int(task.audit.preflight_ticks),
                "min_truth_clearance_m":
                    float(task.audit.min_truth_clearance_m)
                    if math.isfinite(float(task.audit.min_truth_clearance_m))
                    else 0.0,
                "goal_distance_m": float(task.audit.goal_distance_m),
                "preflight_status": str(task.audit.preflight_status),
            },
        })
        committed, reason, audit_summary = self._episode(
            scene, task, writer, tick_base)
        success = bool(committed) and reason == "goal_reached"
        writer.add_metadata({
            "reached_goal": bool(audit_summary.get("reached_goal", False)),
            "quality_committed": success,
            "episode_audit": audit_summary,
        })
        final_dir = writer.finish(success, reason, {
            "audit": audit_summary,
        })
        rospy.loginfo(
            "[Manager] episode %s %s (%s) matched=%d unmatched=%d "
            "none_depth=%d min_clr=%.3f",
            episode_id, "committed" if success else "rejected", reason,
            audit_summary.get("matched", 0),
            audit_summary.get("unmatched", 0),
            audit_summary.get("none_depth", 0),
            audit_summary.get("min_truth_clearance_m", 0.0))
        audit_summary = dict(audit_summary)
        audit_summary["episode_id"] = episode_id
        return committed, reason, audit_summary, final_dir

    def _run_topup(self, plan, scene_map):
        """Recompute the 5 Hz / 30 Hz label distributions from the
        COMMITTED episodes only, then auto top-up the remaining soft gaps
        from the blueprint candidate pool (preflighted) — the real
        collection outcome never trusts the blueprint preflight
        prediction.  Budget: at most `max_rounds` top-up rounds; stops
        when the pool is exhausted or a round produces no new committed
        episode.
        """
        cfg = self._distribution_cfg()
        targets = self._gap_targets(cfg)
        topup_cfg = ((self._g.get("blueprint_generation", {}) or {}).get(
            "topup", {}) or {})
        max_rounds = int(topup_cfg.get("max_rounds", 3))
        stats = self._new_committed_stats(cfg)
        for d in self._committed_dirs:
            self._add_committed_episode(stats, d, cfg)
        rospy.loginfo(
            "[Manager] topup: committed episodes=%d rows=%d (targets=%d)",
            stats["episodes"], stats["rows"], len(targets))

        for rnd in range(1, max_rounds + 1):
            gaps = self._evaluate_gaps(stats, targets)
            if not gaps:
                rospy.loginfo(
                    "[Manager] topup: committed distribution targets met "
                    "after round %d", rnd - 1)
                break
            pool = [t for t in plan.preflighted
                    if int(t.task_id) not in self._used_task_ids]
            if not pool:
                rospy.logwarn(
                    "[Manager] topup round %d: candidate pool exhausted; "
                    "%d soft gaps remain", rnd, len(gaps))
                break
            scored = sorted(
                pool, key=lambda t: -self._score_task_for_gaps(
                    t, stats, targets))
            collected = 0
            for t in scored:
                if rospy.is_shutdown():
                    break
                if not gaps:
                    break
                scene = scene_map.get(int(t.scene_id))
                if scene is None:
                    continue
                committed, reason, audit_summary, final_dir = \
                    self._collect_task_once(
                        scene, t, int(t.task_id) * 600000)
                self._used_task_ids.add(int(t.task_id))
                if committed and final_dir:
                    self._committed_dirs.append(final_dir)
                    self._add_committed_episode(stats, final_dir, cfg)
                    collected += 1
                    gaps = self._evaluate_gaps(stats, targets)
                    rospy.loginfo(
                        "[Manager] topup round %d: committed %s; "
                        "soft gaps remaining=%d", rnd, reason, len(gaps))
            if collected == 0:
                rospy.logwarn(
                    "[Manager] topup round %d: no new committed episode; "
                    "stopping", rnd)
                break

        gaps = self._evaluate_gaps(stats, targets)
        if gaps:
            detail = "; ".join("%s=%.0f/%.0f" % (k, v[0], v[1])
                                for k, v in sorted(gaps.items()))
            rospy.logwarn(
                "[Manager] topup finished: %d soft gaps remaining: %s",
                len(gaps), detail)
        else:
            rospy.loginfo(
                "[Manager] topup finished: committed distribution targets "
                "met (episodes=%d rows=%d)", stats["episodes"],
                stats["rows"])
        return stats, gaps

    def _gravity_flu(self, q_xyzw):
        """Unit gravity direction in the FLU body frame (student input)."""
        q = np.asarray(q_xyzw, dtype=np.float64)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        x, y, z, w = q
        r = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)
        g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return r.T.dot(g_world)

    @staticmethod
    def _encode_plan_points(px, py):
        """Encode a planned trajectory (world XY) as 'x1,y1;x2,y2;...'.

        The points are already in the Flightmare world frame (the expert
        frame is position-identical, yaw-offset only), so the stepped
        viewer can draw them directly.  Empty/None -> ''.
        """
        try:
            xs = list(px or [])
            ys = list(py or [])
        except TypeError:
            return ""
        if not xs or not ys or len(xs) != len(ys):
            return ""
        return ";".join("%.3f,%.3f" % (xs[i], ys[i]) for i in range(len(xs)))

    def _build_row(self, writer, tick, frame_index, out, state, yaw,
                   yaw_rate, depth_m, raw_ratio, final_vel, final_yaw_rate,
                   pos, q, scene_id, task_id, goal, flight_h, frame_id,
                   send_t, recv_t, latency_ms, unity_collision, audit):
        """Build one row dict (buffer; final fields set at episode end)."""
        flu_vel = il_common.world_vector_to_body_flu_quat(
            state.velocity_world, q)
        gravity = self._gravity_flu(q)

        # ── Truth audit values for THIS frame (judge-only) ─────────
        speed = float(np.linalg.norm(state.velocity_world[0:2]))
        # The EFFECTIVE acceleration (lp_eff_accel_mps2, 2.0 with the
        # command-ramp feedforward) — the brake-risk audit must use the
        # physically achieved stopping ability, not a nominal overestimate.
        eff_accel = float(getattr(
            self._params, "lp_eff_accel_mps2",
            float(self._params.lp_max_accel)))
        truth_risk, truth_would = self._truth.brake_risk(
            float(pos[0]), float(pos[1]),
            float(state.velocity_world[0]), float(state.velocity_world[1]),
            eff_accel,
            float(self._params.lp_brake_stop_margin_m))
        if truth_would:
            audit.truth_brake_triggered = True
        # Observed (causal) brake risk from the expert's local observation.
        obs_min_clr = float(out.min_observed_clearance_m)
        obs_risk = 0.0
        if math.isfinite(obs_min_clr) and obs_min_clr >= 0.0:
            stop_dist = speed * speed / (2.0 * max(1e-6, eff_accel))
            static_handoff = (
                float(self._params.lp_min_clearance) +
                max(0.0, float(
                    self._params.lp_clearance_discretization_margin_m)))
            reaction_dist = speed * max(
                0.0, float(self._params.lp_obstacle_reaction_time_s))
            # Same centre-to-occupied-cell envelope used by all local path
            # validators; this diagnostic must not compare against the
            # unrelated truth surface-edge margin.
            env = static_handoff + reaction_dist + stop_dist
            if obs_min_clr < env:
                obs_risk = min(1.0, max(0.0, (env - obs_min_clr) / env))
        audit.max_observed_brake_risk = max(audit.max_observed_brake_risk,
                                            obs_risk)
        # Episode-cumulative REAL minimum truth clearance (running min).
        truth_min = audit.min_truth_clearance_m
        if not math.isfinite(truth_min):
            truth_min = 0.0

        row = {
            "timestamp_ns": int(send_t * 1e9),
            "receive_timestamp_ns": int(recv_t * 1e9),
            "episode_id": self._current_episode_id,
            "frame_id": int(frame_id),
            "episode_frame_index": int(frame_index),
            "control_dt_s": 1.0 / self._control_hz,
            "trajectory_time_s": float(frame_index) / self._control_hz,
            "latency_ms": float(latency_ms),
            "match_method": "frame_id_exact",
            "frame_valid": 1,
            "frame_invalid_reason": "",
            "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]),
            "qx": float(q[0]), "qy": float(q[1]), "qz": float(q[2]),
            "qw": float(q[3]),
            "state_vx_world": float(state.velocity_world[0]),
            "state_vy_world": float(state.velocity_world[1]),
            "state_vz_world": float(state.velocity_world[2]),
            "state_vx_flu": float(flu_vel[0]),
            "state_vy_flu": float(flu_vel[1]),
            "state_vz_flu": float(flu_vel[2]),
            "yaw": float(yaw),
            "yaw_rate": float(yaw_rate),
            # ── student inputs (30 Hz) ─────────────────────────────
            "depth_file": "",
            "raw_depth_finite_ratio": float(raw_ratio),
            "gravity_flu_x": float(gravity[0]),
            "gravity_flu_y": float(gravity[1]),
            "gravity_flu_z": float(gravity[2]),
            "velocity_flu_x": float(flu_vel[0]),
            "velocity_flu_y": float(flu_vel[1]),
            "velocity_flu_z": float(flu_vel[2]),
            "yaw_rate_flu": float(yaw_rate),
            "goal_direction_flu_x": float(out.goal_direction_flu_x),
            "goal_direction_flu_y": float(out.goal_direction_flu_y),
            "goal_direction_flu_z": float(out.goal_direction_flu_z),
            "goal_distance_clipped_m": float(out.goal_distance_clipped_m),
            "goal_distance_norm": float(out.goal_distance_norm),
            "goal_distance_raw_m": float(out.goal_distance_raw_m),
            # ── supervision labels (30 Hz) — the ACTUAL sent command ─
            "target_velocity_flu_x": float(final_vel[0]),
            "target_velocity_flu_y": float(final_vel[1]),
            "target_velocity_flu_z": float(final_vel[2]),
            "target_yaw_rate": float(final_yaw_rate),
            "velocity_command_flu_x": float(final_vel[0]),
            "velocity_command_flu_y": float(final_vel[1]),
            "velocity_command_flu_z": float(final_vel[2]),
            "yaw_rate_command": float(final_yaw_rate),
            # ── expert diagnostics (never student inputs) ─────────
            "hierarchical_mode": str(out.hierarchical_mode),
            "planner_status": str(out.planner_status),
            "planner_failure_reason": str(out.failure_reason),
            "fsm_state": str(out.fsm_state),
            "effective_target_source": str(out.effective_target_source),
            "target_correction_active": int(out.target_correction_active),
            "directive_terminal_stop": int(out.directive_terminal_stop),
            "effective_direction_token": int(out.effective_direction_token),
            "directive_update_event": int(out.directive_update_event),
            "mission_revision": int(out.mission_revision),
            "reentry_guard_ticks": int(out.reentry_guard_ticks),
            "obstacle_first_observed_event":
                int(out.obstacle_first_observed_event),
            "selected_output_speed_mps": float(out.selected_output_speed_mps),
            "local_target_distance_m": float(out.local_target_distance_m),
            "min_observed_clearance_m": float(out.min_observed_clearance_m),
            "obstacle_risk_cost": float(out.obstacle_risk_cost),
            "avoidance_active": int(out.avoidance_active),
            "local_corridor_blocked": int(out.local_corridor_blocked),
            "risk_corridor_near_obstacle":
                int(out.risk_corridor_near_obstacle),
            # R25 structured corridor diagnostics (privileged).
            "corridor_block_reason": str(out.corridor_block_reason),
            "corridor_block_source": str(out.corridor_block_source),
            "first_blocking_distance_m":
                float(out.first_blocking_distance_m),
            "first_block_x": float(out.first_block_x),
            "first_block_y": float(out.first_block_y),
            "first_block_age_ticks": int(out.first_block_age_ticks),
            "emergency_brake": int(out.emergency_brake),
            "immediate_avoidance": int(out.immediate_avoidance),
            "local_limit_cycle_detected": int(out.local_limit_cycle_detected),
            "target_bearing_error_deg": float(out.target_bearing_error_deg),
            "consecutive_failures_30hz": int(out.consecutive_failures_30hz),
            "unknown_recovery_ticks": int(out.unknown_recovery_ticks),
            # ── planned trajectory (world XY, diagnostic for the viewer) ─
            "plan_valid": int(out.plan_valid),
            "plan_terminal": int(out.plan_terminal),
            "plan_end_speed_mps": float(out.plan_end_speed_mps),
            "plan_executed_speed_mps": float(out.plan_executed_speed_mps),
            "plan_points_xy": self._encode_plan_points(
                out.plan_points_x, out.plan_points_y),
            # ── world-frame diagnostics (privileged) ────────────────
            "navigation_goal_world_x": float(goal[0]),
            "navigation_goal_world_y": float(goal[1]),
            "navigation_goal_world_z": float(flight_h),
            "effective_target_world_x": float(out.effective_target_world_x),
            "effective_target_world_y": float(out.effective_target_world_y),
            "effective_target_world_z": float(out.effective_target_world_z),
            "original_navigation_goal_world_x":
                float(out.original_navigation_goal_world_x),
            "original_navigation_goal_world_y":
                float(out.original_navigation_goal_world_y),
            "original_navigation_goal_world_z":
                float(out.original_navigation_goal_world_z),
            # ── truth audit (exact generated cylinders, judge-only;
            #    continuous swept, NOT discrete points; NO real PLY) ─
            "truth_minimum_clearance_m": float(truth_min),
            "truth_brake_risk": float(truth_risk),
            "observed_brake_risk": float(obs_risk),
            "truth_brake_would_trigger": int(1 if truth_would else 0),
            "scene_id": int(scene_id),
            "task_id": int(task_id),
            "episode_valid": 0,   # final value set at the episode end
            "failure_taxonomy": "",
            "failure_reason": "",
            # ── 5 Hz fields (new, two-level schema extension) ──────
            "macro_update_mask": int(out.macro_update_mask),
            "macro_label_valid": int(out.macro_label_valid),
            "macro_correction_type": str(out.macro_correction_type),
            "macro_direction_token": int(out.macro_direction_token),
            "macro_direction_flu_x": float(out.macro_direction_flu_x),
            "macro_direction_flu_y": float(out.macro_direction_flu_y),
            "macro_direction_flu_z": float(out.macro_direction_flu_z),
            "macro_distance_norm": float(out.macro_distance_norm),
            "macro_param_valid": int(out.macro_param_valid),
            "navigation_goal_direction_flu_x":
                float(out.navigation_goal_direction_flu_x),
            "navigation_goal_direction_flu_y":
                float(out.navigation_goal_direction_flu_y),
            "navigation_goal_direction_flu_z":
                float(out.navigation_goal_direction_flu_z),
            "navigation_goal_distance_clipped_m":
                float(out.navigation_goal_distance_clipped_m),
            "navigation_goal_distance_norm":
                float(out.navigation_goal_distance_norm),
            "navigation_goal_distance_raw_m":
                float(out.navigation_goal_distance_raw_m),
            # ── 5 Hz diagnostics (privileged) ───────────────────────
            "correction_enter_event": int(out.correction_enter_event),
            "correction_exit_event": int(out.correction_exit_event),
            "correction_update_event": int(out.correction_update_event),
            "observability_reason": str(out.observability_reason),
            "observability_goal_inside_fov":
                int(out.observability_goal_inside_fov),
            "observability_direct_corridor_blocked":
                int(out.observability_direct_corridor_blocked),
            "observability_left_bypass_visible":
                int(out.observability_left_bypass_visible),
            "observability_right_bypass_visible":
                int(out.observability_right_bypass_visible),
            "observability_local_avoidance_observable":
                int(out.observability_local_avoidance_observable),
        }
        return row

    # ═══════════════════════════════════════════════════════════════
    #  main loop: FULL blueprint FIRST, then Flightmare per manifest
    # ═══════════════════════════════════════════════════════════════
    def run(self):
        rospy.loginfo("[Manager] joint_v2 hierarchical local expert "
                      "(30Hz=%.0f, 5Hz=%.0f, R=%.2f m)",
                      self._control_hz, self._macro_update_hz,
                      self._params.obs_range_m)

        # 1. Task plan: collection-only mode loads a previously generated
        #    manifest (NEVER re-generate / re-randomise); otherwise a fresh
        #    C++ blueprint is generated and saved first.
        if self._manifest_file and os.path.isfile(self._manifest_file):
            plan = self._load_collection_manifest(self._manifest_file)
            rospy.loginfo(
                "[Manager] collection from manifest %s: tasks=%d scenes=%d "
                "pool=%d", self._manifest_file, len(plan.tasks),
                len(plan.scenes), len(plan.preflighted))
        else:
            blueprint = self._generate_blueprint()
            manifest = self._write_blueprint_manifest(blueprint)
            rospy.loginfo(
                "[Manager] blueprint: generation_ok=%s (hard=%s soft=%s) "
                "rounds=%d scenes=%d/%d sampled=%d preflight=%d pool=%d/%d "
                "selected=%d strata=%d/%d cheap_rej=%d deficits=%d -> %s",
                blueprint.generation_ok, blueprint.hard_minimums_met,
                blueprint.soft_targets_met, blueprint.generation_rounds,
                blueprint.scenes_valid, blueprint.scenes_generated,
                blueprint.tasks_sampled, blueprint.tasks_preflighted,
                blueprint.tasks_pool_accepted, blueprint.tasks_pool_target,
                blueprint.tasks_quota_accepted, blueprint.strata_covered,
                blueprint.strata_required, blueprint.cheap_filter_rejected,
                len(blueprint.remaining_deficits), manifest)
            # HARD GATE: a broken blueprint never prints a success line and
            # normal mode never connects Flightmare.
            if not blueprint.generation_ok:
                rospy.logerr(
                    "[Manager] blueprint generation FAILED (%s); NOT "
                    "connecting Flightmare.  hard_minimums_met=%s "
                    "deficits=%s pool_budget_exhausted=%s",
                    blueprint.failure_reason, blueprint.hard_minimums_met,
                    list(blueprint.remaining_deficits),
                    blueprint.pool_budget_exhausted)
                raise RuntimeError(
                    "blueprint generation failed: %s" %
                    blueprint.failure_reason)
            if self._blueprint_only or self._dry_run:
                rospy.loginfo("[Manager] %s mode: generation_ok, manifest "
                              "written; no Unity collection",
                              "blueprint_only" if self._blueprint_only
                              else "dry_run")
                return
            plan = SimpleNamespace(tasks=blueprint.tasks,
                                   scenes=blueprint.scenes,
                                   preflighted=blueprint.preflighted)

        # ── Sort tasks by scene so episodes run scene-contiguously ──
        # The blueprint pool is ordered by distance-class balancing, which
        # interleaves the scenes; re-switching the Unity scene on almost
        # every episode burns the warm-up render budget and the FIRST
        # episode of each scene gets rejected (unmatched_render_rate).
        # Grouping by scene keeps each scene's episodes contiguous — one
        # Unity scene switch per scene instead of one per episode.
        plan.tasks.sort(key=lambda t: int(t.scene_id))

        # 2. Only NOW connect Unity + create dynamics.
        self._connect()
        self._create_dynamics()

        # 3. Strictly per plan (never re-randomises).
        scene_map = {int(s.scene_id): s for s in plan.scenes}
        output_root = os.path.expanduser(self._g.get("output_dir", ""))
        if not output_root:
            output_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dataset", "il_data")
        self._output_root = output_root
        # Fresh collection: clear every previous dataset artifact EXCEPT the
        # blueprint manifest(s) (the task list), so a new run never mixes
        # old episodes / failed runs / stale in-progress dirs into the new
        # data.  blueprint_only / dry_run already returned above.
        self._clear_collection_output(output_root)
        self._last_scene_id = None
        self._committed_dirs = []
        self._used_task_ids = set()
        for task in plan.tasks:
            if rospy.is_shutdown():
                break
            scene = scene_map.get(int(task.scene_id))
            if scene is None:
                rospy.logwarn("[Manager] task %d: unknown scene %d; skipped",
                              int(task.task_id), int(task.scene_id))
                continue
            committed, reason, audit_summary, final_dir = \
                self._collect_task_once(scene, task,
                                        int(task.task_id) * 600000)
            # A scene switch can make Unity's first render of the new scene
            # exceed the strict unmatched-render budget
            # (unmatched_render_rate) even though the warm-up already ran:
            # the scene is now warm, so retry ONCE — the identical episode
            # normally commits on the second attempt.  Non-render rejections
            # (collision / timeout / label issues) are NOT retried.
            if not committed and reason == "unmatched_render_rate":
                rospy.logwarn(
                    "[Manager] episode %s rejected (%s); retrying once after "
                    "scene warm-up", audit_summary.get("episode_id", "?"),
                    reason)
                # Retry with a DIFFERENT but still 6-aligned tick base.  The
                # naive +1 shift makes the 5 Hz macro grid land on
                # episode_frame_index % 6 == 5, which violates the committed
                # data contract (macro_update_mask==1 must sit on % 6 == 0)
                # and fails the strict loader audit.  +6 keeps the grid
                # aligned while still changing the tick origin.
                committed, reason, audit_summary, final_dir = \
                    self._collect_task_once(scene, task,
                                            int(task.task_id) * 600000 + 6)
            self._used_task_ids.add(int(task.task_id))
            if committed and final_dir:
                self._committed_dirs.append(final_dir)

        # 4. Real-collection commit audit (item 十): re-derive the 5 Hz /
        #    30 Hz label distributions from the COMMITTED episodes ONLY
        #    and auto top-up the soft gaps from the candidate pool — never
        #    trust the preflight prediction for the real collection.
        self._run_topup(plan, scene_map)


def main():
    rospy.init_node("il_dataset_manager", anonymous=True)
    config_file = None
    try:
        if rospy.has_param("~config_file"):
            config_file = rospy.get_param("~config_file")
    except Exception:
        pass
    blueprint_only = False
    dry_run = False
    manifest_file = None
    try:
        if rospy.has_param("~blueprint_only"):
            blueprint_only = bool(rospy.get_param("~blueprint_only"))
        if rospy.has_param("~dry_run"):
            dry_run = bool(rospy.get_param("~dry_run"))
        if rospy.has_param("~manifest_file"):
            manifest_file = rospy.get_param("~manifest_file")
    except Exception:
        pass
    manager = JointV2Manager(
        config_path=config_file,
        blueprint_only=blueprint_only,
        dry_run=dry_run,
        manifest_file=manifest_file)
    interrupted = False
    failed = False
    try:
        manager.run()
    except rospy.ROSInterruptException:
        interrupted = True
        print("[Manager] EXIT CAUSE: ROSInterruptException "
              "(rospy_shutdown=%s)" % rospy.is_shutdown(), flush=True)
    except KeyboardInterrupt:
        # rospy usually converts SIGINT into ROSInterruptException, but a
        # raw Ctrl-C during early init can still surface as
        # KeyboardInterrupt; treat it identically.
        interrupted = True
        print("[Manager] EXIT CAUSE: KeyboardInterrupt", flush=True)
    except Exception:
        # Real failure: let the traceback propagate so roslaunch reports it.
        failed = True
        print("[Manager] EXIT CAUSE: exception (propagating)", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        # Individual guards: one resource failing to close must never mask
        # the other.
        try:
            if manager._bridge is not None:
                manager._bridge.close()
        except Exception:
            pass
        try:
            if manager._dynamics is not None:
                manager._dynamics.close()
        except Exception:
            pass
        print("[Manager] EXIT: interrupted=%s failed=%s "
              "rospy_shutdown=%s" % (interrupted, failed,
                                     rospy.is_shutdown()), flush=True)
        if interrupted or not failed:
            # Interrupted OR clean completion: hard-exit, skipping the
            # interpreter's atexit / pybind static-destructor phase.  Clean
            # completion ALSO used to crash there ("double free or
            # corruption (out)", exit -6) because the flightlib / zmq /
            # expert static teardown corrupts the heap during interpreter
            # shutdown.  os._exit(0) is clean: every dataset file is
            # already flushed/closed by the writer, and the OS reclaims the
            # ZMQ sockets / shared libs on process exit.  Real failures
            # still propagate normally so they are never masked.
            os._exit(0)


if __name__ == "__main__":
    main()
