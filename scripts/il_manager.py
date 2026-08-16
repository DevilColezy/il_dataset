#!/usr/bin/env python3
"""
il_manager.py  —  The ONLY collection manager (both
`il_dataset_collect.launch` and `il_dataset_joint_v2_collect.launch` start
this node).  Production path:

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
of every horizontal control command (vx/vy/yaw_rate).  The vertical vz
comes from the altitude hold under global.hierarchical_expert.altitude_hold.

The following are NOT part of the production / preflight / blueprint /
label paths and have been REMOVED entirely (deleted sources and tests):
  * CausalLocalTargetStream / PrivilegedMicroDetourPlanner (C++ legacy),
  * scripts/il_macro_expert.py, scripts/il_micro_expert.py,
  * the old goal switcher, micro-detour controller, old
    behavior-classifier / blueprint generator and debug_viewer.py.
"""

from __future__ import print_function, division

import json
import math
import os
import sys
import time
import traceback
import uuid

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
#  Altitude hold (vertical channel only — the 2D expert never sees z)
# ============================================================================

class AltitudeHold(object):
    """Simple PD altitude keeper producing vz (FLU +up).

    The 2D expert outputs only horizontal vx/vy and yaw_rate; z control is
    merged here and the FINAL command (including this vz) is what gets
    recorded as target_velocity_flu_z and sent to Flightmare.
    """

    def __init__(self, cfg):
        ah = (cfg.get("global", {}).get("hierarchical_expert", {}) or {}).get(
            "altitude_hold", {}) or {}
        self._kp = float(ah.get("kp", 1.5))
        self._kd = float(ah.get("kd", 0.6))
        self._deadband = float(ah.get("deadband_m", 0.02))
        self._max_speed = float(ah.get("max_speed_mps", 0.6))
        self._prev_z = None
        self._prev_t = None

    def reset(self):
        self._prev_z = None
        self._prev_t = None

    def compute(self, target_z, z, vz_world, now_s):
        dz = float(target_z - z)
        if abs(dz) <= self._deadband:
            vz = 0.0
        else:
            vz = float(self._kp * dz)
        if self._prev_z is not None and now_s is not None and \
                self._prev_t is not None and now_s > self._prev_t:
            vz -= float(self._kd * vz_world)
        self._prev_z = z
        self._prev_t = now_s
        return float(np.clip(vz, -self._max_speed, self._max_speed))


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
    def __init__(self, config_path=None, blueprint_only=False, dry_run=False):
        self._config = il_config.load_config(config_path)
        self._g = self._config["global"]
        self._blueprint_only = bool(blueprint_only)
        self._dry_run = bool(dry_run)
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
        self._altitude_hold = AltitudeHold(self._config)

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

        self._bridge = None
        self._dynamics = None
        # Monotonic Unity render frame id (never reset across episodes, so
        # a stale response from a previous scene/task can never match).
        self._next_frame_id = 1
        # Persistent set of Unity object IDs already sent (for retiring
        # obstacles of a previous scene via build_replacing_object_update).
        self._known_object_ids = set()
        self._truth = expert_mod.TruthCylinderAudit()

    # ═══════════════════════════════════════════════════════════════
    #  Full C++ blueprint generation (deficit-driven pipeline)
    # ═══════════════════════════════════════════════════════════════
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
            "obstacle_height_m": float(
                btg.get("obstacle_height_m", geo.get("height_m", 8.0))),
            "task_sample_attempts": int(
                btg.get("task_sample_attempts",
                        tg.get("task_sample_attempts", 300))),
            "task_goal_attempts": int(btg.get("task_goal_attempts", 120)),
            "initial_yaw": dict(btg.get("initial_yaw", {}) or {}),
            "depth_proxy": dict(btg.get("depth_proxy", {}) or {}),
            "histograms": dict(btg.get("histograms", {}) or {}),
            "path": dict(btg.get("path", {}) or {}),
            "performance": dict(perf),
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
        h = int(self._depth_cfg.get("height", 480))
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
        if retired:
            rospy.loginfo("[Manager] scene %d: retired %d stale objects",
                          int(scene.scene_id), retired)

    def _pose_message(self, pos, q, frame_id):
        """Unity Pose message that CARRIES the render frame_id."""
        return {
            "scene_id": int(self._g.get("scene_id", 1)),
            "frame_id": int(frame_id),
            "vehicles": [
                il_common.make_depth_vehicle(
                    [float(pos[0]), float(pos[1]), float(pos[2])],
                    0.0, self._depth_cfg,
                    quaternion_xyzw=[float(q[0]), float(q[1]),
                                     float(q[2]), float(q[3])])],
            "objects": [],
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
        self._altitude_hold.reset()
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
                    int(self._depth_cfg.get("height", 480)),
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

                # 5. Merge altitude-hold vz; build the final command.
                vz = self._altitude_hold.compute(
                    flight_h, float(pos[2]), float(vel[2]), sim_t)
                final_vel = np.array([float(out.target_velocity_flu_x),
                                      float(out.target_velocity_flu_y),
                                      vz], dtype=np.float64)
                final_yaw_rate = float(out.target_yaw_rate)

                # 6. Non-finite input/label guard.
                label_values = [
                    out.goal_direction_flu_x, out.goal_direction_flu_y,
                    out.goal_distance_norm, out.target_velocity_flu_x,
                    out.target_velocity_flu_y, out.target_yaw_rate,
                    out.navigation_goal_direction_flu_x,
                    out.navigation_goal_direction_flu_y,
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
                    audit.final_speed_mps = float(
                        np.linalg.norm(state_after.velocity_world))
                    audit.final_yaw_rate_rps = float(
                        state_after.angular_velocity_body[2])
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
        if not audit.reached_goal:
            return (False, "goal_not_reached",
                    "episode did not reach the original goal %s "
                    "(terminal=%s)" % (list(goal), audit.terminal_state))
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
        truth_risk, truth_would = self._truth.brake_risk(
            float(pos[0]), float(pos[1]),
            float(state.velocity_world[0]), float(state.velocity_world[1]),
            float(self._params.lp_max_accel),
            float(self._params.lp_brake_stop_margin_m))
        if truth_would:
            audit.truth_brake_triggered = True
        # Observed (causal) brake risk from the expert's local observation.
        obs_min_clr = float(out.min_observed_clearance_m)
        obs_risk = 0.0
        if math.isfinite(obs_min_clr) and obs_min_clr >= 0.0:
            stop_dist = speed * speed / (
                2.0 * max(1e-6, float(self._params.lp_max_accel)))
            env = stop_dist + float(self._params.lp_brake_stop_margin_m)
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
            "emergency_brake": int(out.emergency_brake),
            "immediate_avoidance": int(out.immediate_avoidance),
            "local_limit_cycle_detected": int(out.local_limit_cycle_detected),
            "target_bearing_error_deg": float(out.target_bearing_error_deg),
            "consecutive_failures_30hz": int(out.consecutive_failures_30hz),
            "unknown_recovery_ticks": int(out.unknown_recovery_ticks),
            # ── world-frame diagnostics (privileged) ────────────────
            "navigation_goal_world_x": float(goal[0]),
            "navigation_goal_world_y": float(goal[1]),
            "navigation_goal_world_z": float(flight_h),
            "effective_target_world_x": float(out.effective_target_world_x),
            "effective_target_world_y": float(out.effective_target_world_y),
            "effective_target_world_z": float(pos[2]),
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

        # 1. FULL blueprint (C++): all scenes -> all connected tasks ->
        #    per-task preflight with the SAME expert -> classifier + quotas.
        blueprint = self._generate_blueprint()
        # 2. Write the manifest (normal AND blueprint_only use the SAME
        #    C++ generator), recording ACTUAL counts / quota status.
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
        # 3. HARD GATE for ALL modes: a broken blueprint never prints a
        # success line, and normal mode never connects Flightmare.  The
        # gate runs BEFORE any blueprint_only/dry_run early-return.
        if not blueprint.generation_ok:
            rospy.logerr(
                "[Manager] blueprint generation FAILED (%s); NOT connecting "
                "Flightmare.  hard_minimums_met=%s deficits=%s "
                "pool_budget_exhausted=%s",
                blueprint.failure_reason, blueprint.hard_minimums_met,
                list(blueprint.remaining_deficits),
                blueprint.pool_budget_exhausted)
            raise RuntimeError(
                "blueprint generation failed: %s" % blueprint.failure_reason)
        if self._blueprint_only or self._dry_run:
            rospy.loginfo("[Manager] %s mode: generation_ok, manifest "
                          "written; no Unity collection",
                          "blueprint_only" if self._blueprint_only
                          else "dry_run")
            return

        # 4. Only NOW connect Unity + create dynamics.
        self._connect()
        self._create_dynamics()

        # 5. Strictly per manifest (normal mode never re-randomises).
        scene_map = {int(s.scene_id): s for s in blueprint.scenes}
        output_root = os.path.expanduser(self._g.get("output_dir", ""))
        if not output_root:
            output_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "dataset", "il_data")
        ds_cfg = self._g.get("dataset_logging", {}) or {}
        last_scene_id = None
        for task in blueprint.tasks:
            if rospy.is_shutdown():
                break
            scene = scene_map.get(int(task.scene_id))
            if scene is None:
                rospy.logwarn("[Manager] task %d: unknown scene %d; skipped",
                              int(task.task_id), int(task.scene_id))
                continue
            # Scene change: send + settle + retire old objects once.
            if last_scene_id is None or int(scene.scene_id) != last_scene_id:
                self._send_scene_to_unity(scene)
                last_scene_id = int(scene.scene_id)

            task_id = int(task.task_id)
            episode_id = "joint_v2_%06d_%s" % (
                self._episode_id_counter, uuid.uuid4().hex[:8])
            self._episode_id_counter += 1
            tick_base = task_id * 600000
            writer = il_dataset_writer.DatasetWriter(
                ds_cfg, episode_id, output_root, int(scene.scene_id),
                task_id,
                [float(task.start_x), float(task.start_y),
                 float(task.flight_height_m)],
                [float(task.goal_x), float(task.goal_y),
                 float(task.flight_height_m)],
                float(task.initial_yaw), self._depth_cfg)
            writer.add_metadata({
                "expert_stack_revision": "hierarchical_local_v1",
                "schema_extensions": ["two_level_expert_labels_v1"],
                "blueprint_seed": int(task.seed),
                "blueprint_behavior_class": str(task.behavior_class),
                "blueprint_density_class": str(task.density_class),
                "blueprint_radius_class": str(task.radius_class),
                "blueprint_distance_class": str(task.distance_class),
                "blueprint_side_class": str(task.side_class),
                "blueprint_preflight_audit": {
                    "accepted": bool(task.audit.accepted),
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
            writer.finish(success, reason, {
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
    try:
        if rospy.has_param("~blueprint_only"):
            blueprint_only = bool(rospy.get_param("~blueprint_only"))
        if rospy.has_param("~dry_run"):
            dry_run = bool(rospy.get_param("~dry_run"))
    except Exception:
        pass
    manager = JointV2Manager(
        config_path=config_file,
        blueprint_only=blueprint_only,
        dry_run=dry_run)
    try:
        manager.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if manager._bridge is not None:
            manager._bridge.close()
        if manager._dynamics is not None:
            manager._dynamics.close()


if __name__ == "__main__":
    main()
