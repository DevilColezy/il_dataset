#!/usr/bin/env python3
"""Schema-v16 imitation-learning dataset collection manager.

The sole production flow generates a quota-balanced 2.5D cylinder scene and
task set, derives causal local Guide labels from the current observation,
executes Control through the Flightmare/C++ receding-horizon stack, records in
deterministic lockstep, and commits only wholly valid episodes.

Usage:
    roslaunch il_dataset il_dataset_collect.launch
"""

from __future__ import print_function, division

import json, math, os, sys, time, random, copy, threading, shutil, traceback, csv, inspect, queue
import numpy as np
from dataclasses import replace

import rospy
import rospkg

# Ensure this script's directory is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Also add flightmare_dataset_tools scripts dir for point-cloud utilities.
_rp = rospkg.RosPack()
try:
    _ft_scripts = os.path.join(_rp.get_path("flightmare_dataset_tools"), "scripts")
    if os.path.isdir(_ft_scripts) and _ft_scripts not in sys.path:
        sys.path.insert(0, _ft_scripts)
except rospkg.common.ResourceNotFound:
    pass  # flightmare_dataset_tools not installed; unused in profile mode

# Also add the devel lib path for the C++ module
_devel_lib = os.path.join(_rp.get_path("il_dataset"), "..", "..", "devel", "lib")
if os.path.isdir(_devel_lib):
    # Try to find the python module directory
    for _root, _dirs, _files in os.walk(_devel_lib):
        if "_il_local_planner" in " ".join(_files):
            if _root not in sys.path:
                sys.path.insert(0, _root)
            break

# Optional debug-viz imports
_MPL_AVAILABLE = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    _MPL_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
except ImportError:
    Image = None

# ── Project modules ───────────────────────────────────────────────────
from il_common import (
    UnityBridge, ESDFBuilder, SyncBuffer,
    make_depth_vehicle, make_dummy_vehicle,
    load_ply, wait_for_stable_file,
    yaw_rate_for_world_velocity,
    quantize_bounded_vector,
    body_rfu_to_flu, body_flu_to_rfu,
    world_vector_to_body_flu_quat,
    body_flu_to_world_quat,
    # Runtime enums, constants, and immutable plan snapshot.
    PlannerMode, ControlMode, TrendMode,
    LocalPlanSnapshot, RuntimeDecision,
    update_goal_hold_latch, goal_hold_guide_labels,
    make_goal_hold_decision,
    TREND_NORMAL_HORIZONTAL_BIN_COUNT,
    TREND_HORIZONTAL_CLASS_COUNT,
    TREND_RECOVER_LEFT_CLASS,
    TREND_NORMAL_CLASS_OFFSET,
    TREND_RECOVER_RIGHT_CLASS,
    TREND_VERTICAL_CLASS_COUNT,
)

# Phase 2: observed map and guide selector
try:
    from il_observed_map import (
        RollingObservedOccupancyMap, ObservedESDF, PinholeCameraModel)
    _OBSERVED_MAP_AVAILABLE = True
except ImportError:
    _OBSERVED_MAP_AVAILABLE = False

try:
    from il_guide_selector import GuideSelector, GuideSelection
    _GUIDE_SELECTOR_AVAILABLE = True
except ImportError:
    _GUIDE_SELECTOR_AVAILABLE = False

# Phase 3: scene & task generation
try:
    from il_scenario import (
        CylinderObstacleSpec, ObstacleRegion,
        SceneValidationResult, TaskValidationResult, ObservabilityAuditResult,
        YamlCylinderSceneGenerator, CylinderSceneValidator,
        StartGoalTaskGenerator, SideCostEvaluator,
        SceneManifestWriter, SceneGenerationFailureManifestWriter,
        ObstacleVisibilityAuditor, load_scene_profiles,
        compute_raw_occupancy, compute_inflated_occupancy,
        compute_obstacles_per_100m2, compute_region_area,
        compute_pairwise_min_gaps,
    )
    _SCENARIO_AVAILABLE = True
except ImportError:
    _SCENARIO_AVAILABLE = False
    CylinderObstacleSpec = None

# Phase 4: ESDF cache, DAgger, dynamics
try:
    from il_esdf_cache import GlobalESDFCache, ObservedESDFCache
    _ESDF_CACHE_AVAILABLE = True
except ImportError:
    _ESDF_CACHE_AVAILABLE = False
    GlobalESDFCache = None; ObservedESDFCache = None

try:
    from il_dagger import PolicyProvider, DaggerController, PolicyOutput
    _DAGGER_AVAILABLE = True
except ImportError:
    _DAGGER_AVAILABLE = False
    PolicyProvider = None; DaggerController = None; PolicyOutput = None

try:
    from il_dynamics import (
        create_dynamics_backend, DynamicsState, DynamicsBackend,
        FlightmareDynamicsBackend)
    _DYNAMICS_AVAILABLE = True
except ImportError:
    _DYNAMICS_AVAILABLE = False
    create_dynamics_backend = None; DynamicsState = None

from il_config import load_config

# Import il_trajectory from THIS package (avoid shadowing by flightmare_dataset_tools)
import importlib.util as _importlib_util
_il_traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "il_trajectory.py")
_il_traj_spec = _importlib_util.spec_from_file_location("il_trajectory", _il_traj_path)
_il_traj = _importlib_util.module_from_spec(_il_traj_spec)
sys.modules["il_trajectory"] = _il_traj  # register before exec to prevent re-import
_il_traj_spec.loader.exec_module(_il_traj)

# ── C++ local planner (optional — will be None if not available) ─────
_CPP_PLANNER_AVAILABLE = False
_LocalPlanner = None
_LocalPlannerConfig = None
_VehicleState = None
_TrajectoryPoint = None
_LocalPlanResult = None
_PlannerStatus = None
_LocalPlanningRequest = None

try:
    import _il_local_planner as _cpp_planner
    _LocalPlanner = _cpp_planner.LocalPlanner
    _LocalPlannerConfig = _cpp_planner.LocalPlannerConfig
    _VehicleState = _cpp_planner.VehicleState
    _TrajectoryPoint = _cpp_planner.TrajectoryPoint
    _LocalPlanResult = _cpp_planner.LocalPlanResult
    _PlannerStatus = _cpp_planner.PlannerStatus
    _LocalPlanningRequest = _cpp_planner.LocalPlanningRequest
    _CPP_PLANNER_AVAILABLE = True
    rospy.loginfo("[Manager] C++ local planner loaded successfully.")
except ImportError as e:
    rospy.logerr("[Manager] Required C++ local planner is unavailable: %s", e)


class _AsyncPlannerWorker(object):
    """Single-owner planner worker with no request backlog."""

    def __init__(self, planner):
        self._planner = planner
        self._cv = threading.Condition()
        self._request = None
        self._completed = None
        self._busy = False
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name="il_local_planner_worker")
        self._thread.daemon = True
        self._thread.start()

    def submit(self, request):
        with self._cv:
            if self._stop or self._busy or self._request is not None:
                return False
            self._request = request
            self._cv.notify()
            return True

    def take_completed(self):
        with self._cv:
            completed = self._completed
            self._completed = None
            return completed

    def busy(self):
        with self._cv:
            return self._busy or self._request is not None

    def stop(self):
        with self._cv:
            self._stop = True
            self._request = None
            self._cv.notify_all()
        self._thread.join()

    def _run(self):
        while True:
            with self._cv:
                while self._request is None and not self._stop:
                    self._cv.wait()
                if self._stop and self._request is None:
                    return
                request = self._request
                self._request = None
                self._busy = True

            completed = dict(request)
            try:
                completed["result"] = self._planner.plan_local(
                    request["state"], request["previous_progress_s"])
                completed["exception"] = None
            except Exception as exc:
                completed["result"] = None
                completed["exception"] = exc
            completed["completed_mono"] = time.monotonic()

            with self._cv:
                # Retain only the latest completion; never build a result
                # backlog that could later overwrite a newer vehicle state.
                self._completed = completed
                self._busy = False


class _AsyncImageWriter(object):
    """Bounded background writer for depth PNG compression and disk I/O."""

    def __init__(self, max_pending=16):
        self._queue = queue.Queue(maxsize=max(1, int(max_pending)))
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="il_depth_image_writer")
        self._thread.daemon = True
        self._thread.start()

    def submit(self, path, depth_u16):
        if self._error is not None:
            raise RuntimeError("depth image writer failed: {}".format(
                self._error))
        self._queue.put((path, depth_u16))

    def close(self):
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("depth image writer failed: {}".format(
                self._error))

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, depth_u16 = item
                Image.fromarray(depth_u16, mode="I;16").save(path)
            except Exception as exc:
                self._error = exc
            finally:
                self._queue.task_done()


def _resolve_path(pkg_rel):
    return os.path.join(rospkg.RosPack().get_path("il_dataset"), pkg_rel)


# ============================================================================
#  FSM states
# ============================================================================
from enum import Enum


class State(Enum):
    BOOT = "BOOT"
    WAIT_UNITY_CONNECTED = "WAIT_UNITY_CONNECTED"
    GENERATE_OBSTACLE_CONFIG = "GENERATE_OBSTACLE_CONFIG"
    WAIT_SCENE_READY = "WAIT_SCENE_READY"
    EXPORT_POINTCLOUD = "EXPORT_POINTCLOUD"
    WAIT_POINTCLOUD_READY = "WAIT_POINTCLOUD_READY"
    BUILD_ESDF = "BUILD_ESDF"
    GENERATE_START_GOAL_PAIRS = "GENERATE_START_GOAL_PAIRS"
    # v5: renamed PLAN_ALL_TRAJECTORIES → PLAN_GLOBAL_PATHS
    PLAN_GLOBAL_PATHS = "PLAN_GLOBAL_PATHS"
    VALIDATE_GLOBAL_PATHS = "VALIDATE_GLOBAL_PATHS"
    RESET_DRONE = "RESET_DRONE"
    WAIT_DRONE_STABLE = "WAIT_DRONE_STABLE"
    INIT_LOCAL_PLANNER = "INIT_LOCAL_PLANNER"
    START_RECORDING = "START_RECORDING"
    # v5: renamed TRACK_TRAJECTORY → ONLINE_PLAN_AND_RECORD
    ONLINE_PLAN_AND_RECORD = "ONLINE_PLAN_AND_RECORD"
    STOP_RECORDING = "STOP_RECORDING"
    VALIDATE_AND_COMMIT = "VALIDATE_AND_COMMIT"
    NEXT_CONFIG = "NEXT_CONFIG"
    DONE = "DONE"
    ERROR = "ERROR"


class ILManager:
    """State-machine orchestrator for IL dataset collection."""

    # ── Schema v16 ordered field list ────────────────────────────────
    # A committed trajectory contains only fully supervised frames.
    # Row-level validity fields remain as audit invariants; per-head loss
    # masks are intentionally absent because partial supervision is not a
    # supported dataset mode.
    DATA_SCHEMA_V16_FIELDS = [
        # -- time & matching --
        "timestamp_ns", "receive_timestamp_ns", "episode_id", "frame_id",
        "episode_frame_index", "sequence_reset", "control_dt_s",
        "trajectory_time_s", "latency_ms", "match_method",
        "frame_valid", "frame_invalid_reason",
        # -- current state (x_t, before executing expert command) --
        "x", "y", "z", "qx", "qy", "qz", "qw",
        "state_vx_world", "state_vy_world", "state_vz_world",
        "state_vx_flu", "state_vy_flu", "state_vz_flu",
        "state_angular_velocity_x_body", "state_angular_velocity_y_body",
        "state_angular_velocity_z_body",
        # -- v12: FLU angular velocity (current frame, before action) --
        "state_angular_velocity_x_flu", "state_angular_velocity_y_flu",
        "state_angular_velocity_z_flu",
        # -- v12: FLU gravity direction --
        "gravity_direction_x_flu", "gravity_direction_y_flu",
        "gravity_direction_z_flu",
        # -- last upper-level command actually executed before depth_t --
        "previous_executed_command_valid",
        "previous_executed_command_frame_id", "previous_executed_actor",
        "previous_executed_command_vx_flu",
        "previous_executed_command_vy_flu",
        "previous_executed_command_vz_flu",
        "previous_executed_command_yaw_rate",
        # -- expert supervision (sampled from plan generated from x_t) --
        "expert_label_valid",
        "expert_vx_world", "expert_vy_world", "expert_vz_world",
        "expert_vx_flu", "expert_vy_flu", "expert_vz_flu",
        "expert_yaw_rate",
        # -- learner output and DAgger actor selection --
        "learner_output_valid",
        "learner_vx_flu", "learner_vy_flu", "learner_vz_flu",
        "learner_yaw_rate", "learner_inference_ms",
        "dagger_round_id", "dagger_beta", "dagger_random_value",
        "initial_selected_actor", "safety_override", "override_reason",
        "selected_command_vx_flu", "selected_command_vy_flu",
        "selected_command_vz_flu", "selected_command_yaw_rate",
        "final_executed_actor",
        # -- v12: applied command (current control interval) --
        "applied_command_vx_flu", "applied_command_vy_flu",
        "applied_command_vz_flu", "applied_command_yaw_rate",
        "applied_command_actor", "applied_command_valid",
        # -- executed next state (x_(t+1), after dt_sample integration) --
        "executed_next_x", "executed_next_y", "executed_next_z",
        "executed_next_vx_world", "executed_next_vy_world", "executed_next_vz_world",
        "executed_next_vx_flu", "executed_next_vy_flu", "executed_next_vz_flu",
        "actual_next_vx_flu", "actual_next_vy_flu", "actual_next_vz_flu",
        "executed_next_yaw", "executed_next_yaw_rate",
        "actual_acceleration_x_world", "actual_acceleration_y_world",
        "actual_acceleration_z_world", "actual_acceleration_x_flu",
        "actual_acceleration_y_flu", "actual_acceleration_z_flu",
        "actual_angular_velocity_x", "actual_angular_velocity_y",
        "actual_angular_velocity_z", "velocity_tracking_error",
        "yaw_rate_tracking_error",
        # -- global navigation labels --
        "global_direction_valid",
        "global_dir_x_flu", "global_dir_y_flu", "global_dir_z_flu",
        "global_distance_m", "global_distance_norm",
        # -- v11: runtime mode enums --
        "planner_mode", "control_mode", "trend_mode",
        # -- temporary trend labels (v11: 13-class horizontal) --
        "trend_label_valid", "guide_source", "guide_mode",
        "guide_mode_changed", "recovery_entered", "recovery_exited",
        "guide_x_world", "guide_y_world", "guide_z_world",
        "guide_dir_x_flu_exact", "guide_dir_y_flu_exact", "guide_dir_z_flu_exact",
        "guide_distance_m", "guide_distance_norm",
        "guide_azimuth_rad", "guide_elevation_rad",
        "guide_azimuth_bin", "guide_elevation_bin",
        # v11: 13-class trend horizontal
        "trend_horizontal_class_13", "trend_horizontal_class_count",
        "trend_normal_horizontal_class_11",
        # guide_azimuth_soft_0 ... guide_azimuth_soft_12 (appended dynamically)
        # guide_elevation_soft_0 ... guide_elevation_soft_{V-1} (appended dynamically)
        # -- v11: guide cache fields --
        "guide_plan_id", "guide_cache_age_s", "guide_cache_valid",
        "guide_target_x_world", "guide_target_y_world", "guide_target_z_world",
        "guide_target_path_index",
        # -- depth & collision --
        "depth_file", "collision",
        # -- start/goal --
        "start_x", "start_y", "start_z",
        "goal_x", "goal_y", "goal_z",
        # -- planner & debug fields --
        "global_progress_s", "global_progress_ratio", "global_progress_index",
        "local_goal_index", "plan_id", "plan_time_from_start_s",
        "planner_status", "planner_success", "planner_compute_ms",
        "planner_min_clearance", "distance_to_final_goal",
        # -- v11: planner scheduling & retry --
        "plan_source_frame_id", "plan_age_s", "plan_is_fresh",
        "plan_cache_valid",
        "planner_due", "planner_attempted", "planner_retry_count",
        "terminal_scale_used",
        # -- v11: recovery fields --
        "recovery_direction",
        "recovery_target_x_world", "recovery_target_y_world",
        "recovery_target_z_world", "recovery_target_path_index",
        "recovery_target_distance_m",
        "recovery_target_distance_norm_debug",
        "recovery_elapsed_s", "recovery_azimuth_rad",
        # -- v12: trajectory reference fields --
        "trajectory_reference_vx_flu", "trajectory_reference_vy_flu",
        "trajectory_reference_vz_flu",
        "trajectory_feedback_vx_flu", "trajectory_feedback_vy_flu",
        "trajectory_feedback_vz_flu",
        "trajectory_sample_time_s",
        # -- Phase 2: observed map diagnostics --
        "observed_map_revision",
        "observed_esdf_revision",
        "observed_known_voxel_count",
        "observed_occupied_voxel_count",
        "observed_free_voxel_count",
        # -- Phase 2: guide selection diagnostics --
        "guide_candidate_count",
        "guide_visible",
        "guide_depth_visible",
        "guide_corridor_check_enabled",
        "guide_corridor_known_free_ratio",
        "guide_path_index",
        "guide_rejection_reason",
        # -- Phase 2: terminal diagnostics --
        "terminal_path_index",
        "terminal_x_world", "terminal_y_world", "terminal_z_world",
        "terminal_distance_m",
        "terminal_path_arc_length_m",
        # -- Phase 2: planner observed ESDF flags --
        "planner_used_observed_esdf",
        "planner_map_source",
        "planner_unknown_is_free",
        "planner_used_global_fallback",
        "reference_segment_point_count",
        "scene_id", "task_id",
        "observability_check_triggered", "observed_left_cost",
        "observed_right_cost", "observed_lower_cost_side",
        "observed_side_cost_difference_ratio", "observability_consistent",
    ]

    def __init__(self, config):
        self.cfg = config
        self.g = config["global"]
        self._depth_cfg = self.g["depth"]

        # Debug mode
        self.debug = rospy.get_param("~debug", False)
        self.debug_dir = None
        if self.debug and _MPL_AVAILABLE:
            self.debug_dir = os.path.join(
                self.g.get("output_dir", _resolve_path("dataset/il_data")), "_debug")
            if not os.path.isdir(self.debug_dir):
                os.makedirs(self.debug_dir)
            rospy.loginfo("[DEBUG] Visualisation PNGs saved to %s", self.debug_dir)
        elif self.debug:
            rospy.logwarn("[DEBUG] matplotlib not available.")

        # Paths
        self.output_root = self.g["output_dir"]
        self.map_dir = _resolve_path("map")
        self._failed_dir = os.path.join(self.output_root, "_failed")
        for d in [self.output_root, self.map_dir, self._failed_dir]:
            if not os.path.isdir(d):
                os.makedirs(d)

        # Bridge (created in run(), closed in finally)
        self.bridge = None

        # ESDF builder
        self.esdf_builder = ESDFBuilder(
            self.g["pointcloud"]["range"],
            self.g["pointcloud"]["origin"],
            self.g["esdf"]["resolution"],
            self.g["esdf"]["drone_radius"])

        # ── v5: C++ local planner ───────────────────────────────
        self._cpp_planner = None
        self._planner_thread = None
        self._planner_stop_event = threading.Event()
        self._latest_plan = None  # LocalPlanResult
        self._latest_plan_lock = threading.Lock()
        self._planner_hz = 10.0
        self._control_hz = self.g["control"]["control_hz"]
        self._record_hz = self.g["control"]["record_hz"]
        self._trajectory_dt = 1.0 / self._control_hz

        # Determine planner backend
        lp_cfg = self.g.get("planning", {}).get("local_planner", {})
        self._planner_backend = lp_cfg.get("backend", "cpp_pybind")

        if self._planner_backend == "cpp_pybind":
            if not _CPP_PLANNER_AVAILABLE:
                raise RuntimeError(
                    "Configured C++ local planner is unavailable; formal "
                    "collection cannot fall back to Python")
            self._init_cpp_planner(lp_cfg)
        else:
            raise RuntimeError(
                "Only cpp_pybind is permitted by the formal collection path")

        # ── Phase 2: observed map, ESDF, camera model, guide selector ──
        obs_cfg = self.g.get("observed_map", {})
        self._use_observed_map = bool(obs_cfg.get("enabled", False))
        self._use_observed_esdf = bool(
            lp_cfg.get("use_observed_esdf", False))
        self._observed_map = None
        self._observed_esdf = None
        self._camera_model = None
        self._guide_selector = None
        self._guide_progress_index = -1
        self._consecutive_guide_failures = 0

        if self._use_observed_map and _OBSERVED_MAP_AVAILABLE:
            self._observed_map = RollingObservedOccupancyMap(config)
            self._observed_esdf = ObservedESDF(config)
        elif self._use_observed_map:
            raise RuntimeError("Observed-map module is required but unavailable")
        if self.g.get("scene_generation", {}).get("enabled", False) and not _SCENARIO_AVAILABLE:
            raise RuntimeError("Scenario-validation module is required but unavailable")
        if self.g.get("dagger", {}).get("enabled", False) and not _DAGGER_AVAILABLE:
            raise RuntimeError("DAgger module is required but unavailable")
        if not _ESDF_CACHE_AVAILABLE:
            raise RuntimeError("ESDF cache module is required but unavailable")
        # Guide selection depends on the current camera frame, not on whether
        # depth history is fused into an observed occupancy map.
        if _GUIDE_SELECTOR_AVAILABLE and _OBSERVED_MAP_AVAILABLE:
            self._camera_model = PinholeCameraModel(
                self._depth_cfg["width"], self._depth_cfg["height"],
                math.radians(self._depth_cfg["fov"]))
            self._guide_selector = GuideSelector(config, self._camera_model)
            rospy.loginfo(
                "[Manager] Local goal explorer enabled "
                "(current depth + relative mission goal, "
                "certified_range=%.2fm, swept_radius=%.2fm, "
                "esdf_check=%.2fm).",
                self._guide_selector.explorer_usable_range_m,
                self._guide_selector.explorer_required_radius_m,
                self._guide_selector.explorer_esdf_validation_clearance_m)
        else:
            raise RuntimeError(
                "GuideSelector and camera model are required for formal collection")

        # ── Phase 3: scene & task generation ────────────────────────
        sg_cfg = self.g.get("scene_generation", {})
        self._use_scene_gen = bool(sg_cfg.get("enabled", False))
        self._scene_generator = None
        self._scene_validator = None
        self._task_generator = None
        self._side_cost_eval = None
        self._manifest_writer = None
        self._failure_manifest_writer = None
        self._obs_auditor = None
        self._scene_generation_source = str(
            sg_cfg.get("source", "density_driven")).strip()
        self._use_profile_mode = False
        self._enabled_scene_profiles = []
        self._scene_profile_index = 0
        self._scene_index_in_profile = 0
        self._task_index_in_scene = 0
        self._current_profile = None
        self._current_profile_name = ""

        if self._use_scene_gen and _SCENARIO_AVAILABLE:
            self._scene_generator = YamlCylinderSceneGenerator(config)
            self._scene_validator = CylinderSceneValidator(config)
            self._task_generator = StartGoalTaskGenerator(config)
            side_cost_enabled = bool(
                sg_cfg.get("side_cost", {}).get("enabled", False))
            observability_enabled = bool(
                sg_cfg.get("observability_audit", {}).get(
                    "enabled", False))
            if side_cost_enabled:
                self._side_cost_eval = SideCostEvaluator(config)
            if side_cost_enabled and observability_enabled:
                self._obs_auditor = ObstacleVisibilityAuditor(config)
            self._manifest_writer = SceneManifestWriter(self.output_root)
            self._failure_manifest_writer = SceneGenerationFailureManifestWriter(self.output_root)
            rospy.loginfo(
                "[Manager] Phase 3: scene/task generation enabled "
                "(side_cost=%s, observability_audit=%s).",
                side_cost_enabled,
                side_cost_enabled and observability_enabled)

            if self._scene_generation_source == "density_driven":
                self._enabled_scene_profiles = load_scene_profiles(config)
                if not self._enabled_scene_profiles:
                    raise RuntimeError(
                        "density_driven requires at least one enabled profile")
                self._use_profile_mode = True
                self._scene_profile_index = 0
                self._scene_index_in_profile = 0
                self._task_index_in_scene = 0
                self._current_profile = self._enabled_scene_profiles[0]
                self._current_profile_name = self._current_profile.name
                rospy.loginfo(
                    "[Manager] Density-driven collection: %d profiles loaded. "
                    "Starting with '%s'.",
                    len(self._enabled_scene_profiles),
                    self._current_profile_name)
            elif self._scene_generation_source == "fixed_scenario":
                self._current_profile_name = "fixed_scenario"
                rospy.loginfo(
                    "[Manager] Fixed-scenario diagnostic flow enabled.")
            else:
                raise RuntimeError(
                    "Unsupported scene_generation.source '{}'"
                    .format(self._scene_generation_source))
        elif self._use_scene_gen:
            rospy.logwarn("[Manager] Phase 3 modules not available.")
            self._use_scene_gen = False

        # ── Phase 4: ESDF cache, DAgger, dynamics ────────────────────
        self._global_esdf_cache = None
        self._observed_esdf_cache = None
        if _ESDF_CACHE_AVAILABLE and self.g.get("esdf_cache", {}).get("enabled", True):
            self._global_esdf_cache = GlobalESDFCache(config)
            self._observed_esdf_cache = ObservedESDFCache(config)
            rospy.loginfo("[Manager] Phase 4: ESDF caching enabled.")

        self._dagger_ctrl = None
        self._policy_provider = None
        dagger_cfg = self.g.get("dagger", {})
        if dagger_cfg.get("enabled", False) and _DAGGER_AVAILABLE:
            self._policy_provider = PolicyProvider(config)
            self._dagger_ctrl = DaggerController(config)
            rospy.loginfo("[Manager] Phase 4: DAgger enabled (round %d, beta=%.2f).",
                          self._dagger_ctrl.round_id, self._dagger_ctrl.current_beta)

        self._dynamics = None
        if not _DYNAMICS_AVAILABLE:
            raise RuntimeError(
                "Flightmare dynamics module is required for collection")
        self._dynamics = create_dynamics_backend(config)
        rospy.loginfo("[Manager] Phase 4: Dynamics backend = %s.",
                      self._dynamics.backend_name)

        # Scene/task tracking
        self._current_scene_obstacles = []  # list of CylinderObstacleSpec
        self._current_scene_validation = None
        self._current_task_validation = None
        self._current_dominant_obstacle = None
        self._current_scene_subseed = 0
        self._current_scene_attempt = 0
        self._current_scene_manifest_path = ""
        self._current_task_manifest_paths = []
        self._invalid_obs_frame_count = 0
        self._observability_trigger_count = 0
        self._observability_consistent_count = 0

        # ── Scene scheduling state ───────────────────────────────────
        if not self._use_profile_mode:
            self._enabled_scene_profiles = []
            self._scene_profile_index = 0
            self._scene_index_in_profile = 0
            self._task_index_in_scene = 0
            self._current_profile = None
        self._current_effective_scene_seed = 0
        self._current_target_density = None
        self._current_target_density_mode = ""
        # Zero-based density-layout attempt for the current profile scene.
        # A task-quota miss is a scene-level failure, so the next pass resumes
        # from a fresh layout instead of reproducing attempt zero forever.
        self._scene_generation_retry_offset = 0

        # State machine
        self.state = State.BOOT
        self.state_start_time = 0.0
        self.state_timeout = 0.0

        # Iteration state
        self.scene_idx = 0
        self.seed_idx = 0
        self.traj_idx = 0
        self.current_scene_cfg = None
        self.current_seed = 0
        self.current_obstacles = []
        self.current_obj_list = []
        self.current_esdf = None
        self.current_esdf_origin = None
        self.current_esdf_stats = None
        self.current_ply_path = None
        self._current_esdf_cache_key = ""
        self._current_esdf_cache_hit = False
        self.current_pairs = []
        self.current_planned = []
        self.scene_label = ""

        # Per-trajectory in-progress tracking
        self._inprogress_dir = None
        self._inprogress_file = None
        self._sync_file = None
        self._global_path_file = None
        self._raw_global_path_file = None
        self._local_plans_file = None
        self._local_plan_points_file = None

        # ── v5: online planning stats ───────────────────────────
        self._total_replans = 0
        self._successful_replans = 0
        self._failed_replans = 0
        self._emergency_hold_count = 0
        self._planning_times_ms = []
        self._executed_clearances = []
        self._trajectory_exit_reason = "not_started"
        self._trajectory_reached_goal = False
        self._final_executed_position = None
        self._final_executed_velocity = None

        # Stats
        self.total_trajectories = 0
        self.total_committed = 0
        self.total_frames_sent = 0
        self.total_frames_received = 0
        self.total_frames_dropped = 0
        self.total_frames_committed = 0

        # Keep-alive
        self._last_keep_alive = 0.0

    def _init_cpp_planner(self, lp_cfg):
        """Initialize the C++ local planner with config."""
        cfg = _LocalPlannerConfig()
        cfg.planner_hz = float(lp_cfg.get("planner_hz", 10.0))
        cfg.horizon_time = float(lp_cfg.get("horizon_time", 2.5))
        cfg.max_plan_age = float(lp_cfg.get("max_plan_age", 0.75))
        cfg.planning_time_budget_ms = float(
            lp_cfg.get("planning_time_budget_ms", 30.0))
        cfg.trajectory_dt = float(self._trajectory_dt)
        cfg.optimizer = str(lp_cfg.get("optimizer", "auto"))

        cfg.lookahead_distance = float(lp_cfg.get("lookahead_distance", 4.0))
        cfg.min_lookahead_distance = float(lp_cfg.get("min_lookahead_distance", 2.0))
        cfg.max_lookahead_distance = float(lp_cfg.get("max_lookahead_distance", 6.0))
        cfg.lookahead_velocity_gain = float(lp_cfg.get("lookahead_velocity_gain", 0.6))
        cfg.curvature_lookahead_gain = float(lp_cfg.get("curvature_lookahead_gain", 1.0))
        cfg.local_map_radius = float(lp_cfg.get("local_map_radius", 6.0))
        cfg.max_reference_points = int(lp_cfg.get("max_reference_points", 32))

        cfg.control_points = int(lp_cfg.get("control_points", 12))
        cfg.control_point_spacing = float(
            lp_cfg.get("control_point_spacing", 0.20))
        cfg.max_iterations = int(lp_cfg.get("max_iterations", 10000))
        cfg.convergence_tolerance = float(lp_cfg.get("convergence_tolerance", 1e-4))
        cfg.initial_step_size = float(lp_cfg.get("initial_step_size", 0.1))
        cfg.minimum_step_size = float(lp_cfg.get("minimum_step_size", 1e-4))
        cfg.max_cost_samples_per_segment = int(
            lp_cfg.get("max_cost_samples_per_segment", 64))
        cfg.seed_trust_radius = float(
            lp_cfg.get("seed_trust_radius", 0.75))

        cfg.min_clearance = float(lp_cfg.get("min_clearance", 0.05))
        cfg.target_clearance = float(lp_cfg.get("target_clearance", 0.20))
        cfg.collision_check_spacing = float(lp_cfg.get("collision_check_spacing", 0.05))

        cfg.weight_path_length = float(
            lp_cfg.get("weight_path_length", 0.05))
        cfg.weight_smooth = float(lp_cfg.get("weight_smooth", 1.0))
        cfg.weight_jerk = float(lp_cfg.get("weight_jerk", 0.2))
        cfg.weight_guide = float(lp_cfg.get("weight_guide", 0.8))
        cfg.weight_obstacle = float(lp_cfg.get("weight_obstacle", 4.0))
        cfg.weight_goal = float(lp_cfg.get("weight_goal", 2.0))
        cfg.weight_dynamics = float(lp_cfg.get("weight_dynamics", 1.0))

        cfg.nominal_speed = float(lp_cfg.get("nominal_speed", 1.8))
        cfg.max_velocity = float(lp_cfg.get("max_velocity", 2.5))
        cfg.max_acceleration = float(lp_cfg.get("max_acceleration", 3.5))
        cfg.max_jerk = float(lp_cfg.get("max_jerk", 15.0))
        cfg.max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))

        cfg.goal_tolerance = float(lp_cfg.get("goal_tolerance", 0.30))
        cfg.goal_speed_tolerance = float(lp_cfg.get("goal_speed_tolerance", 0.20))
        cfg.goal_hold_ticks = int(lp_cfg.get("goal_hold_ticks", 3))

        cfg.max_consecutive_failures = int(lp_cfg.get("max_consecutive_failures", 3))
        cfg.reduce_lookahead_on_failure = bool(lp_cfg.get("reduce_lookahead_on_failure", True))
        cfg.emergency_hold_enabled = bool(lp_cfg.get("emergency_hold_enabled", True))

        self._cpp_planner = _LocalPlanner(cfg)
        self._planner_hz = cfg.planner_hz
        rospy.loginfo("[Manager] C++ local planner initialized (%.0f Hz, %.1fs horizon).",
                      cfg.planner_hz, cfg.horizon_time)

    # ═══════════════════════════════════════════════════════════════
    #  FSM core
    # ═══════════════════════════════════════════════════════════════

    def _clean_output_dirs(self):
        """Clean output directories before starting a new run to prevent
        old files from mixing with new ones.

        Cleans:
          - output_root/          (the main dataset directory)
          - output_root/_debug/   (debug visualisation PNGs)
          - output_root/_failed/  (previously rejected trajectories)
          - map_dir/              (cached PLY point clouds)
        """
        dirs_to_clean = [
            (self.output_root, "output"),
            (os.path.join(self.output_root, "_debug"), "debug"),
            (os.path.join(self.output_root, "_failed"), "failed"),
            (self.map_dir, "map (PLY cache)"),
        ]

        for d, label in dirs_to_clean:
            if not os.path.isdir(d):
                continue
            try:
                for entry in os.listdir(d):
                    entry_path = os.path.join(d, entry)
                    if os.path.isfile(entry_path) or os.path.islink(entry_path):
                        os.unlink(entry_path)
                    elif os.path.isdir(entry_path):
                        shutil.rmtree(entry_path, ignore_errors=True)
                rospy.loginfo("[Cleanup] %s dir cleared: %s", label, d)
            except Exception as exc:
                rospy.logwarn("[Cleanup] Could not fully clean %s (%s): %s",
                              label, d, exc)

        # Recreate essential subdirs that were cleaned
        for sub in ("_debug", "_failed"):
            sub_path = os.path.join(self.output_root, sub)
            if not os.path.isdir(sub_path):
                os.makedirs(sub_path)

        if not os.path.isdir(self.map_dir):
            os.makedirs(self.map_dir)

    def run(self):
        """Main entry point with full cleanup guarantee."""
        rospy.loginfo("=" * 60)
        rospy.loginfo("  IL Manager v9 – FSM starting")
        rospy.loginfo("  Output: %s", self.output_root)
        if self._use_profile_mode:
            total_scenes = sum(p.scene_count for p in self._enabled_scene_profiles)
            rospy.loginfo("  Profiles: %d  |  Total scenes: %d",
                          len(self._enabled_scene_profiles), total_scenes)
            for p in self._enabled_scene_profiles:
                size_summary = ", ".join(
                    "{}={:.2f}-{:.2f}m".format(
                        group.name,
                        group.radius_min_m,
                        group.radius_max_m)
                    for group in p.size_groups)
                rospy.loginfo(
                    "    - %s: %d scenes (density=%s/%.3f-%.3f, "
                    "tier=%s, sizes=[%s])",
                    p.name, p.scene_count, p.density_mode,
                    p.total_density_min, p.total_density_max,
                    p.density_tier, size_summary)
        elif self._use_scene_gen:
            fixed_name = str(self.g.get("scene_generation", {}).get(
                "fixed_scene_name", "")).strip()
            rospy.loginfo(
                "  Fixed diagnostic scenario: %s",
                fixed_name if fixed_name else "unnamed")
        else:
            rospy.loginfo("  Scene generation disabled")
        rospy.loginfo("=" * 60)

        # ── Clean output directories before starting ──────────
        self._clean_output_dirs()

        # Create bridge inside run() so finally can close it
        self.bridge = UnityBridge(self.g["pub_port"], self.g["sub_port"])
        self.bridge.bind()

        try:
            self._fsm_loop()
        except Exception as exc:
            rospy.logerr("Unhandled exception in FSM: %s", exc)
            rospy.logerr(traceback.format_exc())
            self._cleanup_inprogress()
        finally:
            if self.bridge is not None:
                self.bridge.close()
            self._close_open_files()
            rospy.loginfo("Manager shut down.  Committed: %d traj, %d frames",
                          self.total_committed, self.total_frames_committed)

    def _fsm_loop(self):
        """Main FSM tick loop."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state in (State.DONE, State.ERROR):
                self._handle_terminal_state()
                break
            try:
                self._tick()
            except Exception as exc:
                rospy.logerr("Exception in state %s: %s", self.state.value, exc)
                rospy.logerr(traceback.format_exc())
                self._enter_state(State.ERROR)
            rate.sleep()

    def _handle_terminal_state(self):
        """Execute terminal state handler then break loop."""
        if self.state == State.DONE:
            rospy.loginfo("[FSM] All scenes complete.")
        elif self.state == State.ERROR:
            rospy.logerr("[FSM] Error state – shutting down.")
            rospy.signal_shutdown("FSM error.")

    def _tick(self):
        """Execute one FSM tick."""
        handler = getattr(self, "_st_" + self.state.value.lower(), None)
        if handler is None:
            rospy.logerr("No handler for state %s", self.state)
            self._enter_state(State.ERROR)
            return
        handler()

    def _enter_state(self, new_state, timeout=0.0):
        old = self.state
        self.state = new_state
        self.state_start_time = time.monotonic()
        self.state_timeout = timeout
        rospy.loginfo("[FSM] %s → %s  (timeout=%.1fs)", old.value, new_state.value, timeout)

    def _timed_out(self):
        if self.state_timeout <= 0:
            return False
        return (time.monotonic() - self.state_start_time) > self.state_timeout

    def _keep_alive(self):
        """Send a Pose keep-alive if enough time has passed."""
        now = time.monotonic()
        if now - self._last_keep_alive > self.g["fsm"]["keep_alive_period"]:
            ka = {
                "scene_id": self.g["scene_id"], "frame_id": 0,
                "vehicles": [make_depth_vehicle([0, 0, 5], 0, self._depth_cfg)],
                "objects": self.current_obj_list,
            }
            self.bridge.send_pose(ka)
            self._last_keep_alive = now

    def _drain_unity_messages(self):
        """Drain all pending messages from Unity's ZMQ queue.
        Call before PC export to clear stale trajectory messages."""
        drained = 0
        while self.bridge.try_recv() is not None:
            drained += 1
        if drained > 0:
            rospy.loginfo("[FSM] Drained %d stale Unity messages before PC export.", drained)

    # ── Keep-alive: send via the main bridge during long blocking ops ──
    def _keep_alive_during_blocking(self):
        """Send keep-alive via main bridge — safe to call from any FSM state.
        Uses a dummy position high above the scene to keep Unity alive without
        affecting the current drone state."""
        ka = {
            "scene_id": self.g["scene_id"], "frame_id": -999,
            "vehicles": [make_depth_vehicle([0, 0, 5], 0, self._depth_cfg)],
            "objects": self.current_obj_list,
        }
        self.bridge.send_pose(ka)

    # ── File lifecycle ──────────────────────────────────────────────
    def _open_local_plan_points_file(self):
        """Open the dense local-plan sidecar used by expert diagnostics."""
        path = os.path.join(self._inprogress_dir, "local_plan_points.csv")
        self._local_plan_points_file = open(path, "w")
        self._local_plan_points_file.write(
            "plan_id,point_index,t,x,y,z,vx,vy,vz,ax,ay,az,"
            "yaw,yaw_rate,clearance\n")

    def _write_raw_global_path(self, raw_path):
        """Persist the un-shortcut global-planner path for diagnostics."""
        path = os.path.join(self._inprogress_dir, "raw_global_path.csv")
        self._raw_global_path_file = open(path, "w")
        self._raw_global_path_file.write("index,x,y,z\n")
        for index, point in enumerate(raw_path or []):
            self._raw_global_path_file.write(
                "{},{:.6f},{:.6f},{:.6f}\n".format(
                    index, float(point[0]), float(point[1]),
                    float(point[2])))
        self._raw_global_path_file.flush()

    def _write_local_plan_points(self, result):
        """Persist every dense point without changing local_plans.csv."""
        out = self._local_plan_points_file
        if out is None or result is None:
            return
        for point_index, point in enumerate(result.trajectory):
            pos = point.position
            vel = point.velocity
            acc = point.acceleration
            out.write(
                "{},{},{:.6f},{:.6f},{:.6f},{:.6f},"
                "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},"
                "{:.6f},{:.6f},{:.6f}\n".format(
                    int(result.plan_id), point_index, float(point.t),
                    float(pos[0]), float(pos[1]), float(pos[2]),
                    float(vel[0]), float(vel[1]), float(vel[2]),
                    float(acc[0]), float(acc[1]), float(acc[2]),
                    float(point.yaw), float(point.yaw_rate),
                    float(point.clearance)))
        out.flush()

    def _write_local_plan_summary(
            self, result, state, request_timestamp_ns,
            source_frame_id=-1, terminal_scale=1.0):
        """Write one plan summary with a stable frame/plan association."""
        if self._local_plans_file is None or result is None:
            return
        trajectory = result.trajectory
        duration = float(trajectory[-1].t) if len(trajectory) > 0 else 0.0
        guide = result.guide_waypoint
        goal = result.local_goal
        values = [
            int(result.plan_id), int(source_frame_id),
            int(request_timestamp_ns),
            "{:.6f}".format(float(state.position[0])),
            "{:.6f}".format(float(state.position[1])),
            "{:.6f}".format(float(state.position[2])),
            "{:.6f}".format(float(state.velocity[0])),
            "{:.6f}".format(float(state.velocity[1])),
            "{:.6f}".format(float(state.velocity[2])),
            "{:.6f}".format(float(state.acceleration[0])),
            "{:.6f}".format(float(state.acceleration[1])),
            "{:.6f}".format(float(state.acceleration[2])),
            "{:.6f}".format(float(state.yaw)),
            "{:.6f}".format(float(guide[0])),
            "{:.6f}".format(float(guide[1])),
            "{:.6f}".format(float(guide[2])),
            int(result.guide_waypoint_index),
            "{:.6f}".format(float(goal[0])),
            "{:.6f}".format(float(goal[1])),
            "{:.6f}".format(float(goal[2])),
            "{:.6f}".format(float(result.progress_s)),
            int(result.progress_index), int(result.local_goal_index),
            int(result.status), bool(result.success),
            "{:.6f}".format(float(result.planning_time_ms)),
            "{:.6f}".format(float(result.min_clearance)),
            len(trajectory), "{:.6f}".format(duration),
            "{:.4f}".format(float(terminal_scale)),
        ]
        self._local_plans_file.write(
            ",".join(str(value) for value in values) + "\n")
        self._local_plans_file.flush()
        self._write_local_plan_points(result)

    def _close_open_files(self):
        # v11: flush buffered depth images to disk before closing
        depth_buffer = getattr(self, "_depth_buffer", None)
        if depth_buffer:
            self._flush_depth_buffer()
        image_writer = getattr(self, "_image_writer", None)
        if image_writer is not None:
            try:
                image_writer.close()
            except Exception as exc:
                rospy.logerr("[Recorder] Async image writer failed: %s", exc)
                self._trajectory_exit_reason = "image_writer_failure"
            self._image_writer = None
        for attr in ("_inprogress_file", "_sync_file", "_global_path_file",
                     "_raw_global_path_file", "_local_plans_file",
                     "_local_plan_points_file"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _flush_depth_buffer(self):
        """Write all buffered depth images to disk in one batch.

        Uses a background thread for PNG compression so the main thread
        isn't blocked.  Depth arrays are released from memory as they
        are written.
        """
        depth_dir = os.path.join(self._inprogress_dir, "depth")
        if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir)
        buffer = self._depth_buffer
        self._depth_buffer = []

        if not buffer or Image is None:
            return

        rospy.loginfo("[Recorder] Flushing %d depth images to disk...", len(buffer))
        # Write in a background thread so we don't block the FSM transition
        errors = []

        def _write_all():
            for png_name, depth_u16 in buffer:
                try:
                    if not png_name.endswith(".png"):
                        png_name = png_name + ".png"
                    path = os.path.join(depth_dir, png_name)
                    Image.fromarray(depth_u16, mode="I;16").save(path)
                except Exception as exc:
                    errors.append(str(exc))

        thread = threading.Thread(target=_write_all, name="il_depth_flush")
        thread.start()
        thread.join(timeout=120.0)  # 2-minute safety timeout
        if thread.is_alive():
            rospy.logerr("[Recorder] Depth image flush timed out!")
        if errors:
            rospy.logerr("[Recorder] Depth image flush errors: %s",
                         "; ".join(errors[:5]))
        rospy.loginfo("[Recorder] Depth image flush complete (%d images).", len(buffer))

    def _cleanup_inprogress(self):
        """Clean up in-progress data on abnormal exit."""
        self._close_open_files()
        # Leave .inprogress dir for manual inspection / recovery

    # ═══════════════════════════════════════════════════════════════
    #  Debug plotting
    # ═══════════════════════════════════════════════════════════════

    def _debug_plot_esdf(self, tag, start, goal, raw_path, opt_path):
        if not _MPL_AVAILABLE or self.current_esdf is None or self.debug_dir is None:
            return
        esdf = self.current_esdf
        ox, oy, oz = self.current_esdf_origin
        res = self.g["esdf"]["resolution"]
        gx, gy, gz = esdf.shape

        z_mid = (start[2] + goal[2]) / 2.0
        iz = int(math.floor((z_mid - oz) / res))
        iz = max(0, min(gz - 1, iz))
        esdf_slice = esdf[:, :, iz].T

        fig, ax = plt.subplots(figsize=(14, 16))
        vmax = max(3.0, float(np.percentile(esdf_slice[esdf_slice > 0], 95))
                   if np.any(esdf_slice > 0) else 3.0)
        im = ax.imshow(esdf_slice, origin="lower", cmap="RdYlBu",
                       extent=[ox, ox + gx * res, oy, oy + gy * res],
                       vmin=-2, vmax=vmax, aspect="equal", interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="ESDF (m)", shrink=0.8)

        for obs in self.current_obstacles:
            c = Circle((obs["x"], obs["y"]), obs["radius"],
                       facecolor="black", edgecolor="gray", alpha=0.55, linewidth=0.4)
            ax.add_patch(c)

        ax.plot(start[0], start[1], "go", markersize=12, markeredgewidth=2,
                markeredgecolor="darkgreen", label="Start")
        ax.plot(goal[0], goal[1], "r*", markersize=16, markeredgewidth=1.5,
                markeredgecolor="darkred", label="Goal")

        if raw_path and len(raw_path) > 1:
            rx, ry = [p[0] for p in raw_path], [p[1] for p in raw_path]
            ax.plot(rx, ry, "y-", linewidth=1.5, alpha=0.7, label="A* raw")
        if opt_path and len(opt_path) > 1:
            oxx, oyy = [p[0] for p in opt_path], [p[1] for p in opt_path]
            ax.plot(oxx, oyy, "c-", linewidth=2.0, label="Optimised")

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("ESDF slice z≈{:.1f}m  —  {}".format(z_mid, tag))
        ax.legend(loc="upper right")
        ax.set_xlim(ox - 2, ox + gx * res + 2)
        ax.set_ylim(oy - 2, oy + gy * res + 2)

        out_path = os.path.join(self.debug_dir,
                                "{}_debug_{}.png".format(self.scene_label, tag))
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        rospy.loginfo("[DEBUG] Saved %s", out_path)

    def _debug_plot_executed(self, tag, start, goal, raw_path, opt_path,
                             executed_positions):
        """Plot the executed trajectory alongside the planned paths on ESDF.

        This is called AFTER recording to compare planned vs. actual flight.

        Args:
            tag:  label string for the filename
            start, goal:  (x, y, z) world coordinates
            raw_path:  A* raw waypoints (list of (x,y,z)) or None
            opt_path:  optimised (smoothed) waypoints (list of (x,y,z)) or None
            executed_positions:  list of (x, y) world coords of actual drone flight
        """
        if not _MPL_AVAILABLE or self.current_esdf is None or self.debug_dir is None:
            return
        esdf = self.current_esdf
        ox, oy, oz = self.current_esdf_origin
        res = self.g["esdf"]["resolution"]
        gx, gy, gz = esdf.shape

        z_mid = (start[2] + goal[2]) / 2.0
        iz = int(math.floor((z_mid - oz) / res))
        iz = max(0, min(gz - 1, iz))
        esdf_slice = esdf[:, :, iz].T

        fig, ax = plt.subplots(figsize=(14, 16))
        vmax = max(3.0, float(np.percentile(esdf_slice[esdf_slice > 0], 95))
                   if np.any(esdf_slice > 0) else 3.0)
        im = ax.imshow(esdf_slice, origin="lower", cmap="RdYlBu",
                       extent=[ox, ox + gx * res, oy, oy + gy * res],
                       vmin=-2, vmax=vmax, aspect="equal", interpolation="bilinear")
        plt.colorbar(im, ax=ax, label="ESDF (m)", shrink=0.8)

        # ── Obstacles ──────────────────────────────────────────
        for obs in self.current_obstacles:
            c = Circle((obs["x"], obs["y"]), obs["radius"],
                       facecolor="black", edgecolor="gray", alpha=0.55, linewidth=0.4)
            ax.add_patch(c)

        # ── Start / Goal ───────────────────────────────────────
        ax.plot(start[0], start[1], "go", markersize=12, markeredgewidth=2,
                markeredgecolor="darkgreen", label="Start")
        ax.plot(goal[0], goal[1], "r*", markersize=16, markeredgewidth=1.5,
                markeredgecolor="darkred", label="Goal")

        # ── Planned paths ──────────────────────────────────────
        if raw_path and len(raw_path) > 1:
            rx, ry = [p[0] for p in raw_path], [p[1] for p in raw_path]
            ax.plot(rx, ry, "y-", linewidth=1.5, alpha=0.5, label="A* raw")
        if opt_path and len(opt_path) > 1:
            oxx, oyy = [p[0] for p in opt_path], [p[1] for p in opt_path]
            ax.plot(oxx, oyy, "c-", linewidth=2.0, alpha=0.7, label="Optimised")

        # ── Executed trajectory ────────────────────────────────
        if executed_positions and len(executed_positions) > 1:
            ex = [p[0] for p in executed_positions]
            ey = [p[1] for p in executed_positions]
            ax.plot(ex, ey, "m-", linewidth=2.5, alpha=0.9, label="Executed")
            ax.plot(ex[0], ey[0], "mo", markersize=8, markeredgewidth=1.5,
                    markeredgecolor="darkmagenta")
            ax.plot(ex[-1], ey[-1], "mX", markersize=10, markeredgewidth=1.5,
                    markeredgecolor="darkmagenta")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("ESDF slice z≈{:.1f}m  —  {}  (executed)".format(z_mid, tag))
        ax.legend(loc="upper right")
        ax.set_xlim(ox - 2, ox + gx * res + 2)
        ax.set_ylim(oy - 2, oy + gy * res + 2)

        out_path = os.path.join(self.debug_dir,
                                "{}_debug_{}_executed.png".format(self.scene_label, tag))
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        rospy.loginfo("[DEBUG] Saved %s", out_path)

    # ═══════════════════════════════════════════════════════════════
    #  FSM state handlers
    # ═══════════════════════════════════════════════════════════════

    def _scene_geometry_contract(self):
        """Return the sole configured vehicle and cylinder-gap contract."""
        scene_cfg = self.g["scene_generation"]
        vehicle_cfg = scene_cfg["vehicle"]
        cylinder_cfg = scene_cfg["common_cylinder"]
        return (
            float(vehicle_cfg["radius_m"]),
            float(vehicle_cfg["safety_margin_m"]),
            float(cylinder_cfg["minimum_surface_gap_m"]),
            float(cylinder_cfg["minimum_post_inflation_gap_m"]),
        )

    def _st_boot(self):
        self._enter_state(State.WAIT_UNITY_CONNECTED,
                          self.g["fsm"]["connect_timeout"])

    def _st_wait_unity_connected(self):
        if self._timed_out():
            rospy.logerr("Unity connection timeout.")
            self._enter_state(State.ERROR)
            return

        vehicle = make_depth_vehicle([0, 0, 5], 0, self._depth_cfg)
        settings = {"scene_id": self.g["scene_id"], "vehicles": [vehicle], "objects": []}
        self.bridge.send_pose(settings)
        r = self.bridge.try_recv()
        if r is not None and r[0].get("ready"):
            rospy.loginfo("[FSM] Unity connected.")
            self._enter_state(State.GENERATE_OBSTACLE_CONFIG)
        else:
            time.sleep(0.2)

    def _st_generate_obstacle_config(self):
        # ── Formal collection: density-driven multi-profile scenes ──
        if (self._use_profile_mode and self._scene_generator is not None
                and len(self._enabled_scene_profiles) > 0):
            # Check if all profiles are done
            if self._scene_profile_index >= len(self._enabled_scene_profiles):
                rospy.loginfo("[FSM] All profiles complete. Entering DONE.")
                self._enter_state(State.DONE)
                return

            self._current_profile = self._enabled_scene_profiles[self._scene_profile_index]
            self._current_profile_name = self._current_profile.name

            # Check if current profile scene count is done
            if self._scene_index_in_profile >= self._current_profile.scene_count:
                rospy.loginfo("[FSM] Profile '%s' complete (%d scenes). Advancing to next profile.",
                              self._current_profile_name, self._scene_index_in_profile)
                self._scene_profile_index += 1
                self._scene_index_in_profile = 0
                self._task_index_in_scene = 0
                self._scene_generation_retry_offset = 0
                if self._scene_profile_index >= len(self._enabled_scene_profiles):
                    rospy.loginfo("[FSM] All profiles complete. Entering DONE.")
                    self._enter_state(State.DONE)
                    return
                self._current_profile = self._enabled_scene_profiles[self._scene_profile_index]
                self._current_profile_name = self._current_profile.name

            profile = self._current_profile
            rospy.loginfo("=" * 60)
            rospy.loginfo("  PROFILE '%s' [%d/%d]  scene %d/%d",
                          profile.name,
                          self._scene_profile_index + 1,
                          len(self._enabled_scene_profiles),
                          self._scene_index_in_profile + 1,
                          profile.scene_count)
            rospy.loginfo("=" * 60)

            # Compute effective scene seed
            base_seed = self._scene_generator.base_seed
            effective_scene_seed = (base_seed
                                    + profile.seed_offset
                                    + self._scene_index_in_profile)
            self._current_effective_scene_seed = effective_scene_seed

            max_attempts = self._scene_generator.max_scene_attempts

            obstacles = []
            validation = None
            target_density = None
            density_mode_str = profile.density_mode
            final_attempt_index = 0

            attempt_start = int(self._scene_generation_retry_offset)
            for attempt in range(attempt_start, max_attempts):
                (obstacles, rejection,
                 target_density, density_mode_str) = \
                    self._scene_generator.generate_scene_density_driven(
                        profile, effective_scene_seed,
                        self._scene_index_in_profile, attempt)

                if not obstacles:
                    rospy.logwarn("[SceneGen] Profile '%s' scene %d attempt %d: %s",
                                  profile.name, self._scene_index_in_profile,
                                  attempt + 1, rejection)
                    continue

                # Validate topology
                validation = self._scene_validator.validate(
                    obstacles, self._scene_generator.obstacle_region)
                if validation.valid:
                    self._current_scene_obstacles = obstacles
                    self._current_scene_validation = validation
                    self._current_scene_subseed = effective_scene_seed
                    self._current_scene_attempt = attempt + 1
                    self._current_target_density = target_density
                    self._current_target_density_mode = density_mode_str
                    final_attempt_index = attempt

                    # Convert to Unity object format
                    self.current_obstacles = [
                        {"id": o.obstacle_id, "x": float(o.center_world[0]),
                         "y": float(o.center_world[1]), "z": float(o.center_world[2]),
                         "radius": o.radius_m, "diameter": o.diameter_m(),
                         "height": o.height_m}
                        for o in obstacles]
                    self.current_obj_list = self._scene_generator.generate_unity_objects(obstacles)

                    self.scene_label = "{}_{:06d}".format(
                        profile.name, self._scene_index_in_profile)
                    rospy.loginfo("[SceneGen] ACCEPTED: profile='%s' scene=%d obstacles=%d "
                                  "attempt=%d/%d target_density=%.3f mode=%s",
                                  profile.name, self._scene_index_in_profile,
                                  len(obstacles), attempt + 1, max_attempts,
                                  target_density if target_density is not None else -1.0,
                                  density_mode_str)
                    break
                else:
                    rospy.logwarn("[SceneGen] Profile '%s' scene %d attempt %d REJECTED: %s",
                                  profile.name, self._scene_index_in_profile,
                                  attempt + 1, validation.rejection_reason)

            if not obstacles or (validation is not None and not validation.valid):
                # Generation exhausted — write failure manifest and terminate
                failure_reason = (validation.rejection_reason if validation is not None
                                  else "SCENE_OBSTACLE_SAMPLING_EXHAUSTED")
                rospy.logerr("[SceneGen] Profile '%s' scene %d: exhausted all %d attempts. "
                             "Reason: %s. Terminating collection.",
                             profile.name, self._scene_index_in_profile,
                             max_attempts, failure_reason)
                if self._failure_manifest_writer is not None:
                    self._failure_manifest_writer.write_failure_manifest(
                        profile.name,
                        self._scene_profile_index,
                        self._scene_index_in_profile,
                        effective_scene_seed,
                        failure_reason,
                        max_attempts)
                # Do NOT silently skip — terminate with ERROR
                self._enter_state(State.ERROR)
                return

            self._enter_state(State.WAIT_SCENE_READY,
                              self.g["fsm"]["scene_settle_timeout"])
            return

        # ── Diagnostic collection: deterministic fixed scenario ─────
        if (self._use_scene_gen and self._scene_generator is not None and
                self._scene_generation_source == "fixed_scenario"):
            if self.scene_idx >= 1:
                self._enter_state(State.DONE)
                return

            rospy.loginfo("=" * 60)
            rospy.loginfo(
                "  FIXED-SCENARIO DIAGNOSTIC: %s",
                self.g.get("scene_generation", {}).get(
                    "fixed_scene_name", "unnamed"))
            rospy.loginfo("=" * 60)

            # Try generating a valid scene
            obstacles = []
            validation = None
            for attempt in range(self._scene_generator.max_scene_attempts):
                sub_seed = self.scene_idx * 1000 + attempt
                obstacles, rejection = self._scene_generator.generate_scene(sub_seed)
                if not obstacles:
                    rospy.logwarn("[SceneGen] Attempt %d: %s", attempt + 1, rejection)
                    continue

                # Validate topology
                validation = self._scene_validator.validate(
                    obstacles, self._scene_generator.obstacle_region)
                if validation.valid:
                    self._current_scene_obstacles = obstacles
                    self._current_scene_validation = validation
                    self._current_scene_subseed = sub_seed
                    self._current_scene_attempt = attempt + 1

                    # Convert to Unity object format
                    self.current_obstacles = [
                        {"id": o.obstacle_id, "x": float(o.center_world[0]),
                         "y": float(o.center_world[1]), "z": float(o.center_world[2]),
                         "radius": o.radius_m, "diameter": o.diameter_m(),
                         "height": o.height_m}
                        for o in obstacles]
                    self.current_obj_list = self._scene_generator.generate_unity_objects(obstacles)

                    fixed_scene_name = str(
                        self.g.get("scene_generation", {}).get(
                            "fixed_scene_name", "")).strip()
                    self.scene_label = (
                        fixed_scene_name
                        if fixed_scene_name else
                        "scene_{:04d}_sub{:04d}".format(
                            self.scene_idx, sub_seed))
                    rospy.loginfo("[SceneGen] ACCEPTED: %d obstacles, subseed=%d, attempt=%d/%d",
                                  len(obstacles), sub_seed, attempt + 1,
                                  self._scene_generator.max_scene_attempts)
                    break
                else:
                    rospy.logwarn("[SceneGen] Attempt %d REJECTED: %s",
                                  attempt + 1, validation.rejection_reason)

            if not obstacles or (validation is not None and not validation.valid):
                rospy.logerr(
                    "[SceneGen] Fixed scenario failed validation after %d attempts.",
                    self._scene_generator.max_scene_attempts)
                self._enter_state(State.ERROR)
                return

            self.scene_idx += 1
            self._enter_state(State.WAIT_SCENE_READY,
                              self.g["fsm"]["scene_settle_timeout"])
            return

        rospy.logerr(
            "[FSM] Scene generation reached an invalid source/state: %s",
            self._scene_generation_source)
        self._enter_state(State.ERROR)

    def _st_wait_scene_ready(self):
        if self._timed_out():
            self._enter_state(State.EXPORT_POINTCLOUD)
            return

        vehicle = make_depth_vehicle(
            [0, 0, 5], 0, self._depth_cfg
        )

        msg = {
            "scene_id": self.g["scene_id"],
            "frame_id": 0,
            "vehicles": [vehicle],
            "objects": self.current_obj_list,
        }

        self.bridge.send_pose(msg)

        # 防止深度/Pose 回复积压到 RCVHWM
        while self.bridge.try_recv() is not None:
            pass

        time.sleep(0.1)

    def _st_export_pointcloud(self):
        # 先让 Unity 完成最后一批障碍物和相机消息处理
        time.sleep(1.0)

        # 必须在发送 PC 请求前的最后一刻清空队列
        self._drain_unity_messages()

        req = {
            "range": list(self.g["pointcloud"]["range"]),
            "origin": list(self.g["pointcloud"]["origin"]),
            "resolution": self.g["pointcloud"]["resolution"],
            "path": self.map_dir + "/",
            "file_name": self.scene_label,
        }

        self.current_ply_path = os.path.join(
            self.map_dir, self.scene_label + ".ply"
        )

        if os.path.exists(self.current_ply_path):
            os.remove(self.current_ply_path)

        self._pc_acked = False
        self._pc_req_count = 1
        self._pc_ack_warned = False

        # 记录发送时间，但不要每 4 秒无限重发
        self._pc_request_time = time.monotonic()
        self._pc_next_ka = (
            self._pc_request_time
            + self.g["fsm"]["keep_alive_period"]
        )

        self.bridge.send_pc_request(req)

        rospy.loginfo(
            "[FSM] PointCloud request #1 sent: %s",
            self.scene_label,
        )

        self._enter_state(
            State.WAIT_POINTCLOUD_READY,
            self.g["fsm"]["pc_export_timeout"],
        )

    def _st_wait_pointcloud_ready(self):
        if self._timed_out():
            rospy.logerr(
                "Point cloud export timed out: scene=%s acked=%s requests=%d path=%s",
                self.scene_label,
                self._pc_acked,
                self._pc_req_count,
                self.current_ply_path,
            )
            self._enter_state(State.ERROR)
            return

        now = time.monotonic()
        save_success = False

        # 一次取完当前所有待处理消息，避免 ack 被旧帧挡在后面
        recv_count = 0

        while True:
            r = self.bridge.try_recv()
            if r is None:
                break

            recv_count += 1
            msg = r[0]

            if msg.get("get_pc_msg"):
                if not self._pc_acked:
                    rospy.loginfo(
                        "[FSM] PointCloud request acknowledged by Unity."
                    )
                self._pc_acked = True

            if msg.get("save_pc_success"):
                save_success = True

        # 没有收到 ack 不代表第一次请求没有被 Unity 接收。
        # 禁止每 4 秒重发，以免反复重启或阻塞导出任务。
        if (
            not self._pc_acked
            and not self._pc_ack_warned
            and now - self._pc_request_time > 10.0
        ):
            rospy.logwarn(
                "[FSM] No PointCloud ack after 10 s; "
                "continuing to wait for the PLY file without resending."
            )
            self._pc_ack_warned = True

        # 点云等待期间发送无相机 keep-alive
        if now >= self._pc_next_ka:
            self._send_pointcloud_keep_alive()
            self._pc_next_ka = (
                now + self.g["fsm"]["keep_alive_period"]
            )

        if save_success:
            if wait_for_stable_file(
                self.current_ply_path,
                stable_sec=1.0,
                max_wait=10.0,
            ):
                rospy.loginfo("[FSM] Point cloud saved and stable.")
                self._enter_state(State.BUILD_ESDF)
                return

        # 即使 save_pc_success 或 ack 丢失，也以文件为最终依据
        if os.path.exists(self.current_ply_path):
            if wait_for_stable_file(
                self.current_ply_path,
                stable_sec=0.5,
                max_wait=5.0,
            ):
                rospy.loginfo(
                    "[FSM] Point cloud file appeared and stable."
                )
                self._enter_state(State.BUILD_ESDF)
                return

        time.sleep(0.05)

    def _send_pointcloud_keep_alive(self):
        """Keep Unity alive without requesting depth-camera rendering."""
        vehicle = make_dummy_vehicle()

        # Unity coordinates: x-right, y-up, z-forward
        # ROS [0, 0, 5] -> Unity [0, 5, 0]
        vehicle["position"] = [0.0, 5.0, 0.0]
        vehicle["size"] = [0.5, 0.5, 0.5]

        msg = {
            "scene_id": self.g["scene_id"],
            "frame_id": -999,
            "vehicles": [vehicle],
            "objects": self.current_obj_list,
        }

        self.bridge.send_pose(msg)

    def _st_build_esdf(self):
        # Send keep-alive before potentially long ESDF build
        self._keep_alive_during_blocking()
        try:
            cache_key = None
            cached = None
            if self._global_esdf_cache is not None and self._global_esdf_cache.enabled:
                scene_manifest = {
                    "scene_label": self.scene_label,
                    "obstacles": self.current_obstacles,
                }
                cache_key = self._global_esdf_cache.compute_cache_key(
                    scene_manifest, self.current_ply_path)
                if cache_key:
                    cached = self._global_esdf_cache.load(cache_key)
            if cached is not None:
                self.current_esdf, self.current_esdf_origin, cached_res = cached
                if abs(float(cached_res) - float(self.g["esdf"]["resolution"])) > 1e-12:
                    raise RuntimeError("Validated ESDF cache returned a resolution mismatch")
                self.current_esdf_stats = {"cache_hit": True}
                self._current_esdf_cache_hit = True
            else:
                self.current_esdf, self.current_esdf_origin, self.current_esdf_stats = \
                    self.esdf_builder.build(self.current_ply_path)
                self._current_esdf_cache_hit = False
                if cache_key:
                    self._global_esdf_cache.save(
                        cache_key, self.current_esdf, self.current_esdf_origin,
                        self.g["esdf"]["resolution"], self.scene_label)
            self._current_esdf_cache_key = cache_key or ""
            self._enter_state(State.GENERATE_START_GOAL_PAIRS)
        except Exception as exc:
            rospy.logerr("ESDF build failed: %s", exc)
            traceback.print_exc()
            self._enter_state(State.ERROR)

    def _reject_current_scene_for_task_failure(self, reason_code, detail):
        """Reject a whole scene when its final task quota is unavailable.

        Task generation and every task-level post-filter are part of one
        atomic scene acceptance decision.  A density-driven scene therefore
        retries with the next layout attempt; a fixed diagnostic scene cannot
        change its layout and fails immediately.
        """
        detail = str(detail)
        reason = "{}:{}".format(str(reason_code), detail)
        if self._use_profile_mode:
            # _current_scene_attempt is the accepted layout's one-based
            # attempt number and therefore also the next zero-based attempt
            # index consumed by _st_generate_obstacle_config.
            next_attempt = int(self._current_scene_attempt)
            max_attempts = int(self._scene_generator.max_scene_attempts)
            if next_attempt < max_attempts:
                self._scene_generation_retry_offset = next_attempt
                rospy.logwarn(
                    "[TaskGen] Rejecting profile '%s' scene %d layout "
                    "attempt %d: %s. Regenerating with attempt %d/%d.",
                    self._current_profile_name,
                    self._scene_index_in_profile,
                    next_attempt, detail,
                    next_attempt + 1, max_attempts)
                self._enter_state(State.GENERATE_OBSTACLE_CONFIG)
                return

            rospy.logerr(
                "[TaskGen] Profile '%s' scene %d exhausted %d layout "
                "attempts: %s",
                self._current_profile_name,
                self._scene_index_in_profile,
                max_attempts, detail)
            if self._failure_manifest_writer is not None:
                self._failure_manifest_writer.write_failure_manifest(
                    self._current_profile_name,
                    self._scene_profile_index,
                    self._scene_index_in_profile,
                    self._current_effective_scene_seed,
                    reason,
                    max_attempts)
            self._enter_state(State.ERROR)
            return

        fixed_scene_name = str(
            self.g.get("scene_generation", {}).get(
                "fixed_scene_name", "unnamed")).strip()
        rospy.logerr(
            "[TaskGen] Fixed scenario '%s' is invalid and cannot regenerate "
            "its layout: %s", fixed_scene_name or "unnamed", detail)
        self._enter_state(State.ERROR)

    def _st_generate_start_goal_pairs(self):
        # Phase 3: new task generation pipeline
        if (self._use_scene_gen and self._task_generator is not None and
                self._scene_generator is not None and
                len(self._current_scene_obstacles) > 0):
            rospy.loginfo("[FSM] Phase 3: generating tasks via StartGoalTaskGenerator...")

            # Apply ALL profile-specific task generation parameters
            # (sampling regions, height ranges, distance limits, blocking
            #  requirements, etc.) — not just tasks_per_scene.
            if self._use_profile_mode and self._current_profile is not None:
                self._task_generator.configure_from_profile(
                    self._current_profile)

            # Use the il_trajectory module already loaded via importlib at
            # module level (line ~133), which is guaranteed to come from
            # THIS package regardless of devel-space symlink resolution.
            _AStarPlanner = _il_traj.AStarPlanner

            def astar_fn(esdf, origin, res, start, goal, min_cl):
                planner = _AStarPlanner(
                    esdf, res, origin,
                    cost_weight=float(self.g.get("planning", {}).get(
                        "global_planner", {}).get("cost_weight", 0.0)),
                    clearance_target=float(self.g.get("planning", {}).get(
                        "global_planner", {}).get("clearance_target", 0.25)))
                return planner.plan(start, goal, min_clearance=min_cl,
                                    epsilon=float(self.g.get("planning", {}).get(
                                        "global_planner", {}).get("epsilon", 1.10)),
                                    max_iterations=int(self.g.get("planning", {}).get(
                                        "global_planner", {}).get("max_iterations_full", 800000)))

            try:
                tasks = self._task_generator.generate_tasks(
                    self._current_scene_obstacles,
                    self.current_esdf, self.current_esdf_origin,
                    self.g["esdf"]["resolution"],
                    astar_fn,
                    seed=self._current_scene_subseed)
            except (RuntimeError, ValueError) as exc:
                self._reject_current_scene_for_task_failure(
                    "TASK_COVERAGE_UNAVAILABLE", exc)
                return

            # For blocked tasks, evaluate configured left/right portal costs
            # with the same A* implementation.  Unblocked tasks have no
            # meaningful dominant obstacle or side-choice audit.
            if self._side_cost_eval is not None:
                side_validated_tasks = []
                for start, goal, task_val in tasks:
                    dominant = next((o for o in self._current_scene_obstacles
                                     if o.obstacle_id == task_val.dominant_obstacle_id), None)
                    # Open baselines and passable-gate tasks deliberately have
                    # no obstacle intersecting the direct corridor.  Left/right
                    # portal costs are undefined there, but that is not a task
                    # failure: the global path itself is the reference.
                    if dominant is None:
                        side_validated_tasks.append(
                            (start, goal, task_val))
                        continue
                    side = self._side_cost_eval.evaluate(
                        np.asarray(start, dtype=np.float64),
                        np.asarray(goal, dtype=np.float64), dominant,
                        self._current_scene_obstacles, self.current_esdf,
                        self.current_esdf_origin, self.g["esdf"]["resolution"],
                        astar_fn)
                    for field_name in (
                            "left_path_valid", "right_path_valid",
                            "left_path_cost", "right_path_cost",
                            "left_path_length_m", "right_path_length_m",
                            "left_min_clearance_m", "right_min_clearance_m",
                            "lower_cost_side", "side_cost_difference_ratio",
                            "global_side_choice_valid"):
                        setattr(task_val, field_name, getattr(side, field_name))
                    if side.valid:
                        side_validated_tasks.append((start, goal, task_val))
                    else:
                        rospy.logwarn("[TaskGen] Side-cost rejection: %s",
                                      side.rejection_reason)
                tasks = side_validated_tasks

            expected_task_count = int(self._task_generator.tasks_per_scene)
            if len(tasks) != expected_task_count:
                self._reject_current_scene_for_task_failure(
                    "TASK_POSTFILTER_QUOTA_UNAVAILABLE",
                    "side-cost/observability prerequisites retained {} of "
                    "{} required tasks".format(
                        len(tasks), expected_task_count))
                return

            # Convert to current_pairs format
            self.current_pairs = []
            for task_index, (start, goal, task_val) in enumerate(tasks):
                pair = {"start": start, "goal": goal, "valid_endpoints": True,
                        "_task_validation": task_val,
                        "task_id": "task_{:03d}".format(task_index)}
                self.current_pairs.append(pair)

            self._desired_pair_count = len(self.current_pairs)
            self.traj_idx = 0
            self.current_planned = []
            rospy.loginfo("[FSM] Phase 3: Generated %d validated tasks.", len(self.current_pairs))

            # Write scene manifest
            if self._manifest_writer is not None and self._current_scene_validation is not None:
                if self._use_profile_mode and self._current_profile is not None:
                    scene_id = "{}_{:06d}".format(
                        self._current_profile_name, self._scene_index_in_profile)
                else:
                    fixed_scene_name = str(
                        self.g.get("scene_generation", {}).get(
                            "fixed_scene_name", "")).strip()
                    scene_id = (
                        fixed_scene_name
                        if fixed_scene_name else
                        "scene_{:04d}".format(self.scene_idx - 1))

                task_results_for_manifest = [
                    pair.get("_task_validation") for pair in self.current_pairs
                    if pair.get("_task_validation") is not None]

                # Compute density metrics
                obstacles = self._current_scene_obstacles
                region = self._scene_generator.obstacle_region
                region_area = (region.x_max - region.x_min) * (region.y_max - region.y_min)
                actual_raw = compute_raw_occupancy(obstacles, region_area) if obstacles else 0.0
                (vehicle_r, safety_m,
                 common_req_sg, common_req_pg) = (
                    self._scene_geometry_contract())
                actual_inflated = compute_inflated_occupancy(obstacles, region_area, vehicle_r, safety_m)
                actual_per_100m2 = compute_obstacles_per_100m2(obstacles, region_area)

                # Compute gap stats
                min_sg, min_pg = compute_pairwise_min_gaps(obstacles, vehicle_r, safety_m)

                # Profile-aware gap requirements
                if self._use_profile_mode and self._current_profile is not None:
                    req_sg = self._current_profile.minimum_surface_gap_m
                    req_pg = self._current_profile.minimum_post_inflation_gap_m
                    profile_index = self._scene_profile_index
                    profile_name = self._current_profile_name
                    seed_offset = self._current_profile.seed_offset
                else:
                    req_sg = common_req_sg
                    req_pg = common_req_pg
                    profile_index = 0
                    profile_name = "fixed_scenario"
                    seed_offset = 0

                self._current_scene_manifest_path = self._manifest_writer.write_scene_manifest(
                    scene_id,
                    self._scene_generator.base_seed,
                    profile_name,
                    profile_index,
                    self._scene_index_in_profile,
                    self._current_effective_scene_seed,
                    self._current_scene_attempt,
                    obstacles,
                    self._current_scene_validation,
                    task_results_for_manifest,
                    self._scene_generator.obstacle_region,
                    self._current_target_density_mode,
                    self._current_target_density,
                    actual_raw,
                    actual_inflated,
                    actual_per_100m2,
                    vehicle_r,
                    safety_m,
                    req_sg,
                    req_pg,
                    min_sg,
                    min_pg,
                    generation_status="accepted",
                    profile_seed_offset=seed_offset)

                # Write task manifests
                self._current_task_manifest_paths = []
                for ti, pair in enumerate(self.current_pairs):
                    tv = pair.get("_task_validation")
                    if tv is not None:
                        task_manifest_path = self._manifest_writer.write_task_manifest(
                            scene_id,
                            "task_{:03d}".format(ti),
                            pair["start"], pair["goal"], tv,
                            profile_name=profile_name)
                        self._current_task_manifest_paths.append(task_manifest_path)

            self._enter_state(State.PLAN_GLOBAL_PATHS)
            return

        rospy.logerr(
            "[FSM] Start/goal generation requires density_driven or "
            "fixed_scenario scene generation.")
        self._enter_state(State.ERROR)

    def _st_plan_global_paths(self):
        """Prepare the goal-only mission axis used by the local explorer."""

        for pi, pair in enumerate(self.current_pairs):
            if rospy.is_shutdown():
                break
            if not pair.get("valid_endpoints", True):
                rospy.logwarn("  Skipping candidate %d with invalid endpoints: %s",
                              pi + 1, pair.get("failure_reason", "unknown"))
                continue
            self._keep_alive_during_blocking()
            start = tuple(pair["start"])
            goal = tuple(pair["goal"])
            rospy.loginfo("  Global path %d/%d: (%.1f,%.1f,%.1f) → (%.1f,%.1f,%.1f)",
                          pi + 1, len(self.current_pairs),
                          start[0], start[1], start[2],
                          goal[0], goal[1], goal[2])

            if self.debug and _MPL_AVAILABLE:
                self._debug_plot_esdf(
                    "plan_{:02d}_pre".format(pi + 1), start, goal, None, None)

            plan = self._build_goal_explorer_mission_axis(start, goal)
            if plan is None:
                rospy.logwarn(
                    "  Global planner FAILED for start→goal pair %d",
                    pi + 1)
                plan = {
                    "start": list(start), "goal": list(goal), "valid": False,
                    "validation_report": {"total_violations": 999,
                                          "invalid_reasons": [
                                              "global_planner_failed"]},
                    "raw_path": [], "global_path": [],
                    "global_path_length": 0.0,
                }
            plan["_task_validation"] = pair.get("_task_validation")
            plan["task_id"] = pair.get("task_id", "task_{:03d}".format(pi))
            if plan.get("valid"):
                self.current_planned.append(plan)
            rpt = plan.get("validation_report", {})
            rospy.loginfo("    valid=%s  raw_pts=%d  shortcut_pts=%d  length=%.1fm",
                          plan["valid"],
                          len(plan.get("raw_path", [])),
                          len(plan.get("global_path", [])),
                          plan.get("global_path_length", 0.0))

            if self.debug and _MPL_AVAILABLE:
                self._debug_plot_esdf(
                    "plan_{:02d}_post".format(pi + 1), start, goal,
                    plan.get("raw_path"), plan.get("global_path"))

            if len(self.current_planned) >= getattr(
                    self, "_desired_pair_count", len(self.current_pairs)):
                break

        self._enter_state(State.VALIDATE_GLOBAL_PATHS)

    def _build_goal_explorer_mission_axis(self, start, goal):
        """Build bookkeeping samples on start-goal axis, without path search.

        The samples support progress, final braking and existing sidecar
        schemas. They are not collision-checked and are never candidates for
        the online Guide selector.
        """
        start_np = np.asarray(start, dtype=np.float64)
        goal_np = np.asarray(goal, dtype=np.float64)
        delta = goal_np - start_np
        distance = float(np.linalg.norm(delta))
        spacing = float(self.g.get("planning", {}).get(
            "global_planner", {}).get(
                "reference_resample_spacing_m", 0.10))
        sample_count = max(
            2, int(math.ceil(distance / max(0.02, spacing))) + 1)
        axis = [
            (start_np + (float(i) / float(sample_count - 1)) * delta).tolist()
            for i in range(sample_count)
        ]
        return {
            "start": start_np.tolist(),
            "goal": goal_np.tolist(),
            "valid": True,
            "validation_report": {
                "algorithm": "local_goal_explorer",
                "global_path_search_performed": False,
                "mission_axis_only": True,
                "invalid_reasons": [],
            },
            "raw_path": [start_np.tolist(), goal_np.tolist()],
            "global_path": axis,
            "global_path_length": distance,
        }

    def _st_validate_global_paths(self):
        """v5: Validate global paths (simpler than old full-trajectory validation)."""
        valid_count = sum(1 for p in self.current_planned if p.get("valid"))
        invalid = [p for p in self.current_planned if not p.get("valid")]
        rospy.loginfo("[FSM] %d/%d global paths valid.",
                      valid_count, len(self.current_planned))
        if invalid:
            for p in invalid:
                r = p.get("validation_report", {})
                reasons = r.get("invalid_reasons", ["unknown"])
                rospy.logwarn("  INVALID %s→%s: reasons=%s",
                              p["start"], p["goal"], reasons)
        self.traj_idx = 0
        self._enter_state(State.RESET_DRONE)

    def _get_current_initial_yaw(self):
        """Return mission-axis yaw plus a scene-stratified offset.

        The base yaw points the drone nose along the first path segment.
        The tasks in one scene cover equal-width bins across
        ``scene_generation.common_task_generation.
        initial_yaw_randomization_deg``.  A scene-seeded permutation removes
        correlation with task order while guaranteeing symmetric catalog
        coverage and exact reproducibility.
        """
        plan = self.current_planned[self.traj_idx]
        global_path = plan.get("global_path", [])
        if len(global_path) >= 2:
            dx = global_path[1][0] - global_path[0][0]
            dy = global_path[1][1] - global_path[0][1]
            base_yaw = math.atan2(dy, dx) - math.pi / 2.0
        else:
            base_yaw = 0.0

        # Apply one centre-of-bin offset from a deterministic scene-wide
        # stratification.  Bin centres avoid the exact +/-limit endpoints.
        task_cfg = self.g["scene_generation"]["common_task_generation"]
        max_offset_deg = float(task_cfg["initial_yaw_randomization_deg"])
        if max_offset_deg > 0.0:
            max_offset_rad = math.radians(max_offset_deg)
            bin_count = max(1, len(self.current_planned))
            bin_order = list(range(bin_count))
            rng = random.Random("{}_initial_yaw_bins".format(
                self.scene_label))
            rng.shuffle(bin_order)
            bin_index = bin_order[self.traj_idx % bin_count]
            bin_fraction = (
                (float(bin_index) + 0.5) / float(bin_count))
            offset = -max_offset_rad + 2.0 * max_offset_rad * bin_fraction
            base_yaw += offset
            base_yaw = math.atan2(math.sin(base_yaw), math.cos(base_yaw))  # wrap

        return base_yaw

    def _st_reset_drone(self):
        if self.traj_idx >= len(self.current_planned):
            self._enter_state(State.VALIDATE_AND_COMMIT)
            return
        plan = self.current_planned[self.traj_idx]
        if not plan.get("valid") or not plan.get("global_path"):
            rospy.logwarn("[FSM] Skipping invalid trajectory %d.", self.traj_idx + 1)
            self.traj_idx += 1
            self._enter_state(State.RESET_DRONE if self.traj_idx < len(self.current_planned)
                              else State.VALIDATE_AND_COMMIT)
            return

        start = plan["start"]
        init_yaw = self._get_current_initial_yaw()
        if self._dynamics is None:
            rospy.logerr("[FSM] No dynamics backend; refusing to collect an episode.")
            self._enter_state(State.ERROR)
            return
        try:
            self._dynamics.reset(start, init_yaw)
        except Exception as exc:
            rospy.logerr("[FSM] Dynamics reset failed: %s", exc)
            self._enter_state(State.ERROR)
            return
        reset_state = self._dynamics.get_state()
        vehicle = make_depth_vehicle(
            reset_state.position_world.tolist(), init_yaw, self._depth_cfg,
            quaternion_xyzw=reset_state.quaternion_world_body)
        msg = {"scene_id": self.g["scene_id"], "frame_id": 0,
               "vehicles": [vehicle], "objects": self.current_obj_list}
        self.bridge.send_pose(msg)
        self._enter_state(State.WAIT_DRONE_STABLE,
                          self.g["fsm"]["drone_stable_timeout"])

    def _st_wait_drone_stable(self):
        if self._timed_out():
            rospy.logwarn("[FSM] Drone stable timeout reached; proceeding anyway.")
            self._enter_state(State.INIT_LOCAL_PLANNER)
            return

        # ── Real stability check: velocity must be near zero ──
        stab_cfg = self.g.get("fsm", {}).get("drone_stable", {})
        vel_thresh = stab_cfg.get("velocity_threshold", 0.05)   # m/s
        ticks_req  = stab_cfg.get("consecutive_ticks", 20)       # 2 s @ 10 Hz

        state = self._dynamics.get_state()
        vel_norm = float(np.linalg.norm(state.velocity_world))

        if not hasattr(self, "_drone_stable_ticks"):
            self._drone_stable_ticks = 0

        if vel_norm < vel_thresh:
            self._drone_stable_ticks += 1
            if self._drone_stable_ticks >= ticks_req:
                rospy.loginfo("[FSM] Drone stable (|v|=%.3f m/s for %d ticks).",
                              vel_norm, self._drone_stable_ticks)
                self._enter_state(State.INIT_LOCAL_PLANNER)
                return
        else:
            if self._drone_stable_ticks > 0:
                rospy.logdebug("[FSM] Drone velocity %.3f > %.3f, resetting stable counter.",
                               vel_norm, vel_thresh)
            self._drone_stable_ticks = 0

        plan = self.current_planned[self.traj_idx]
        start = plan["start"]
        init_yaw = self._get_current_initial_yaw()
        reset_state = self._dynamics.get_state()
        vehicle = make_depth_vehicle(
            reset_state.position_world.tolist(), init_yaw, self._depth_cfg,
            quaternion_xyzw=reset_state.quaternion_world_body)
        msg = {"scene_id": self.g["scene_id"], "frame_id": -1,
               "vehicles": [vehicle], "objects": self.current_obj_list}
        self.bridge.send_pose(msg)
        self.bridge.try_recv()
        time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════════
    #  v5: INIT_LOCAL_PLANNER — set ESDF, global path, reset state
    # ═══════════════════════════════════════════════════════════════

    def _st_init_local_planner(self):
        """Initialize C++ local planner with ESDF and global path for current trajectory."""
        plan = self.current_planned[self.traj_idx]
        start = plan["start"]
        init_yaw = self._get_current_initial_yaw()
        global_path = plan.get("global_path", [])

        # Phase 3: track task validation for this trajectory
        self._current_task_validation = plan.get("_task_validation", None)
        self._current_dominant_obstacle = None
        self._invalid_obs_frame_count = 0
        self._observability_trigger_count = 0
        self._observability_consistent_count = 0
        if (self._current_task_validation is not None and
                self._current_task_validation.dominant_obstacle_id):
            for obs in getattr(self, "_current_scene_obstacles", []):
                if obs.obstacle_id == self._current_task_validation.dominant_obstacle_id:
                    self._current_dominant_obstacle = obs
                    break

        if not global_path:
            rospy.logerr("[FSM] No global path for trajectory %d", self.traj_idx + 1)
            self.traj_idx += 1
            self._enter_state(State.RESET_DRONE if self.traj_idx < len(self.current_planned)
                              else State.VALIDATE_AND_COMMIT)
            return

        if self._cpp_planner is not None:
            try:
                # ── Phase 2: Reset observed map for new episode ──────
                if self._use_observed_map and self._observed_map is not None:
                    self._observed_map.reset(start)
                    self._observed_esdf.reset()
                    if self._observed_esdf_cache is not None:
                        self._observed_esdf_cache.reset()
                if self._guide_selector is not None:
                    self._guide_selector.reset()
                    self._guide_selector.set_safety_esdf(
                        self.current_esdf,
                        self.current_esdf_origin,
                        float(self.g["esdf"]["resolution"]))
                if self._dagger_ctrl is not None:
                    episode_seed = ((self.scene_idx & 0xffff) << 16) + self.traj_idx
                    self._dagger_ctrl.reset_episode(episode_seed)
                    self._policy_provider.reset()
                    if not self._policy_provider.load():
                        raise RuntimeError("DAgger policy failed to load")
                    rospy.loginfo("[FSM] Observed map reset at start position.")
                self._guide_progress_index = -1
                self._consecutive_guide_failures = 0

                # Set ESDF and global path in C++ planner
                if not self._use_observed_esdf and self.current_esdf is not None:
                    esdf_c = np.ascontiguousarray(
                        self.current_esdf, dtype=np.float32)
                    origin = np.array(self.current_esdf_origin,
                                      dtype=np.float64).reshape(3, 1)
                    ok = self._cpp_planner.set_esdf(
                        esdf_c, origin,
                        float(self.g["esdf"]["resolution"]))
                    if not ok:
                        rospy.logerr("[FSM] Failed to set global ESDF in C++ planner")
                        self._enter_state(State.ERROR)
                        return
                    rospy.loginfo("[FSM] Uploaded global ESDF (%s) to C++ planner.",
                                  "×".join(str(d) for d in self.current_esdf.shape))
                gp_np = np.array(global_path, dtype=np.float64, order='C')
                ok = self._cpp_planner.set_global_path(gp_np)
                if not ok:
                    rospy.logerr("[FSM] Failed to set global path in C++ planner")
                    self._enter_state(State.ERROR)
                    return

                # ── Build the planner's pre-recording reset state ──
                # The warmed Flightmare rigid-body state is restored to this
                # same task start below, before any frame is recorded.
                init_state = _VehicleState()
                init_state.position = tuple(float(v) for v in start)
                init_state.velocity = (0.0, 0.0, 0.0)
                init_state.acceleration = (0.0, 0.0, 0.0)
                init_state.yaw = float(init_yaw)
                init_state.yaw_rate = 0.0
                self._cpp_planner.reset(init_state)

                rospy.loginfo("[FSM] C++ local planner initialized for trajectory %d "
                              "(start=%.2f,%.2f,%.2f yaw=%.1f°).",
                              self.traj_idx + 1, *start,
                              math.degrees(init_yaw))
            except Exception as exc:
                rospy.logerr("[FSM] C++ planner init failed: %s", exc)
                traceback.print_exc()
                if self._planner_backend == "cpp_pybind":
                    self._enter_state(State.ERROR)
                    return
        else:
            rospy.logerr("[FSM] Required C++ local planner is unavailable.")
            self._enter_state(State.ERROR)
            return

        # ── Just-in-time dynamics reset (after planner init) ─────
        # The Flightmare sim falls whenever no hover command is active.
        # Reset the drone NOW (after all slow Python work is done) and
        # extend the settle time to bridge the gap until the first
        # control frame arrives in ONLINE_PLAN_AND_RECORD.
        if self._dynamics is not None:
            try:
                self._dynamics.reset(start, init_yaw)
                # Extended hover to cover START_RECORDING transition
                extra_settle = float(
                    self.g.get("fsm", {}).get("jit_settle_extra_s", 2.0))
                if extra_settle > 0.0:
                    self._dynamics.step_velocity_command(
                        np.zeros(3, dtype=np.float64), 0.0, extra_settle)
                # Keep the warmed-up motor state, but undo the position loss
                # accumulated while the motors spun up from zero.  A second
                # reset() would clear the motors again and repeat the fall.
                self._dynamics.restore_state_keep_motors(
                    start, init_yaw,
                    np.zeros(3, dtype=np.float64),
                    np.zeros(3, dtype=np.float64))
            except Exception as exc:
                rospy.logerr("[FSM] JIT dynamics reset failed: %s", exc)
                self._enter_state(State.ERROR)
                return
            # Sync Unity (frame_id=-1 so lockstep counter stays clean)
            reset_state = self._dynamics.get_state()
            start_error = float(np.linalg.norm(
                reset_state.position_world -
                np.asarray(start, dtype=np.float64)))
            start_speed = float(np.linalg.norm(reset_state.velocity_world))
            if start_error > 1.0e-4 or start_speed > 1.0e-4:
                rospy.logerr(
                    "[FSM] Warm dynamics state does not match task start: "
                    "position_error=%.6fm speed=%.6fm/s",
                    start_error, start_speed)
                self._enter_state(State.ERROR)
                return
            vehicle = make_depth_vehicle(
                reset_state.position_world.tolist(), init_yaw, self._depth_cfg,
                quaternion_xyzw=reset_state.quaternion_world_body)
            msg = {"scene_id": self.g["scene_id"], "frame_id": -1,
                   "vehicles": [vehicle], "objects": self.current_obj_list}
            self.bridge.send_pose(msg)

            # Point-cloud export and the subsequent blocking map/planning work
            # can leave old render replies queued at both ZMQ endpoints.  Do
            # not let the first recorded frame double as a connection probe:
            # first prove that Unity is again processing Pose messages and
            # returning an actual depth payload for the current scene.
            if not self._warm_up_unity_depth_stream(vehicle):
                rospy.logerr(
                    "[FSM] Unity depth stream did not recover after point-cloud "
                    "export; refusing to start an empty recording.")
                self._enter_state(State.ERROR)
                return

        self._enter_state(State.START_RECORDING)

    def _st_start_recording(self):
        """Set up recording files for deterministic lockstep collection."""
        self._enter_state(
            State.ONLINE_PLAN_AND_RECORD,
            self.g["fsm"]["trajectory_timeout"])

        # Create .inprogress directory
        traj_name = "traj_{:03d}".format(self.traj_idx + 1)
        episode_root = (self._dagger_ctrl.get_output_dir()
                        if self._dagger_ctrl is not None else self.output_root)
        self._inprogress_dir = os.path.join(
            episode_root, self.scene_label, traj_name + ".inprogress")
        self._final_dir = os.path.join(
            episode_root, self.scene_label, traj_name)
        self._episode_failed_dir = os.path.join(episode_root, "_failed")
        os.makedirs(self._episode_failed_dir, exist_ok=True)

        if os.path.isdir(self._inprogress_dir):
            rospy.logwarn("[FSM] Removing stale .inprogress: %s", self._inprogress_dir)
            shutil.rmtree(self._inprogress_dir, ignore_errors=True)

        depth_dir = os.path.join(self._inprogress_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

        # Per-trajectory counters
        self._rec_sent_control_frames = 0
        self._rec_raw_received_frames = 0
        self._rec_written_rows = 0
        self._rec_discarded_extra_frames = 0
        self._rec_missing_record_ticks = 0
        self._rec_unmatched_frames = 0
        self._rec_exact_matches = 0
        self._rec_fallback_matches = 0
        self._rec_start_mono = time.monotonic()
        self._rec_start_epoch_ns = int(time.time() * 1e9)

        # v7: Reset planner and validation stats
        self._total_replans = 0
        self._successful_replans = 0
        self._failed_replans = 0
        self._emergency_hold_count = 0
        self._planning_times_ms = []
        self._executed_clearances = []
        self._trajectory_exit_reason = "running"
        self._trajectory_reached_goal = False
        self._final_executed_position = None
        self._final_executed_velocity = None
        self._invalid_expert_label_count = 0
        self._invalid_trend_label_count = 0
        self._episode_invalid_frame_count = 0
        self._episode_invalid_reason_counts = {}

    # ═══════════════════════════════════════════════════════════════
    #  v7: ONLINE_PLAN_AND_RECORD — deterministic lockstep loop
    # ═══════════════════════════════════════════════════════════════

    def _st_online_plan_and_record(self):
        """v7: Strict single-frame lockstep data collection.

        Semantics:
            depth_t, state_t -> plan -> expert_command_t
            → execute dt_sample → state_(t+1)

        Timeline per sample:
            1. Send x_t pose to Unity with frame_id
            2. Wait for exact frame_id depth response
            3. Get depth_t, collision status
            4. Sync call local planner from x_t
            5. Sample expert command u_t* from new trajectory
            6. Compute global nav label and local trend label
            7. Execute dt_sample with sub-steps to reach x_(t+1)
            8. Save row: depth_t, state_t, nav_t, expert_cmd_t, state_(t+1)
        """
        plan = self.current_planned[self.traj_idx]
        global_path = plan.get("global_path", [])
        if not global_path:
            rospy.logwarn("[ONLINE] Empty global path — finishing.")
            self._st_finish_recording()
            return

        # ── Config shorthand ────────────────────────────────────────
        ctrl_hz = self._control_hz
        rec_hz = self._record_hz
        dt_ctrl = 1.0 / ctrl_hz
        dt_sample = 1.0 / rec_hz
        depth_max_m = self._depth_cfg["max_m"]
        img_w, img_h = self._depth_cfg["width"], self._depth_cfg["height"]
        depth_float_len = img_w * img_h * 4

        data_cfg = self.g.get("data", {})
        online_rt = self.g.get("online_runtime", {})
        label_lookahead_time_s = float(data_cfg.get("label_lookahead_time_s", 0.08))
        max_guide_range = float(
            self._guide_selector.explorer_usable_range_m)

        lp_cfg = self.g.get("planning", {}).get("local_planner", {})
        nominal_speed = float(lp_cfg.get("nominal_speed", 1.8))
        max_velocity = float(lp_cfg.get("max_velocity", 2.5))
        max_acceleration = float(lp_cfg.get("max_acceleration", 3.5))
        planner_acceleration_filter_tau = float(
            lp_cfg.get("planner_acceleration_filter_time_constant_s", 0.15))
        planner_acceleration_warm_start_blend = float(
            lp_cfg.get("planner_acceleration_warm_start_blend", 0.80))
        planner_velocity_ramp_time_s = float(
            lp_cfg.get("planner_velocity_ramp_time_s", 1.0))
        max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))
        command_lookahead_time = float(
            lp_cfg.get("velocity_command_lookahead_time", 0.08))
        velocity_tracking_gain = float(
            lp_cfg.get("velocity_tracking_gain", 2.0))
        yaw_tracking_gain = float(lp_cfg.get("yaw_tracking_gain", 3.0))
        yaw_speed_threshold = float(lp_cfg.get("yaw_speed_threshold", 0.10))
        goal_tolerance = float(lp_cfg.get("goal_tolerance", 0.30))
        goal_speed_tol = float(lp_cfg.get("goal_speed_tolerance", 0.20))
        goal_hold_ticks = int(lp_cfg.get("goal_hold_ticks", 3))
        configured_failure_limit = int(
            lp_cfg.get("max_consecutive_failures", 3))
        failure_grace_time = float(lp_cfg.get("failure_grace_time", 1.0))
        effective_failure_limit = max(
            configured_failure_limit,
            int(math.ceil(failure_grace_time * self._planner_hz)))
        unity_response_timeout_s = max(
            0.25, float(self.g.get("sync", {}).get(
                "unity_response_timeout_s", 2.0)))

        # ── Trend bin config (v11: 11 normal FOV bins + 2 recovery = 13 classes) ──
        trend_cfg = data_cfg["trend"]
        trend_normal_h_bins = int(trend_cfg["normal_horizontal_bins"])
        trend_v_bins = int(trend_cfg["vertical_bins"])
        trend_sigma_bins = float(trend_cfg["soft_sigma_bins"])
        # v11: normal FOV uses exactly 11 bin EDGES (12 edges for 11 intervals)
        # NOT 13. The 13-class output is: 0=RECOVER_LEFT, 1-11=normal FOV, 12=RECOVER_RIGHT.
        h_fov_deg = float(self._depth_cfg["fov"])  # horizontal FOV
        h_fov_rad = math.radians(h_fov_deg)
        v_fov_rad = 2.0 * math.atan(
            math.tan(h_fov_rad / 2.0) * img_h / max(img_w, 1))
        # Normal FOV bin edges: 12 edges for 11 bins
        normal_h_bin_edges = np.linspace(
            -h_fov_rad / 2.0, h_fov_rad / 2.0,
            TREND_NORMAL_HORIZONTAL_BIN_COUNT + 1)
        v_bin_edges = np.linspace(-v_fov_rad / 2.0, v_fov_rad / 2.0, trend_v_bins)

        # ── Build dynamic field list with 13-class soft label columns ──
        schema_fields = list(self.DATA_SCHEMA_V16_FIELDS)
        # Insert soft label columns before depth_file
        depth_idx = schema_fields.index("depth_file")
        azi_soft_names = []
        ele_soft_names = []
        # v11: exactly TREND_HORIZONTAL_CLASS_COUNT (13) horizontal soft columns
        for i in range(TREND_HORIZONTAL_CLASS_COUNT):
            name = "trend_horizontal_soft_{:02d}".format(i)
            azi_soft_names.append(name)
            schema_fields.insert(depth_idx, name)
            depth_idx += 1
        for i in range(trend_v_bins):
            name = "guide_elevation_soft_{}".format(i)
            ele_soft_names.append(name)
            schema_fields.insert(depth_idx, name)
            depth_idx += 1

        # ── Directories & files ──────────────────────────────────────
        depth_dir = os.path.join(self._inprogress_dir, "depth")
        data_path = os.path.join(self._inprogress_dir, "data.csv")
        sync_path = os.path.join(self._inprogress_dir, "sync.csv")
        gp_path = os.path.join(self._inprogress_dir, "global_path.csv")
        lp_path = os.path.join(self._inprogress_dir, "local_plans.csv")

        # ── Open data.csv with DictWriter (schema v7) ─────────────────
        self._inprogress_file = open(data_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._inprogress_file, fieldnames=schema_fields)
        self._csv_writer.writeheader()

        # ── Sync CSV (kept for per-frame diagnostics) ─────────────────
        self._sync_file = open(sync_path, "w")
        self._sync_file.write(
            "recv_step,frame_id_matched,latency_ms,match_error_ms,match_method,"
            "is_dropped,ctrl_queue_len,exact_matches,fallback_matches\n")

        # ── Global path CSV ──────────────────────────────────────────
        self._global_path_file = open(gp_path, "w")
        self._global_path_file.write("index,x,y,z,s\n")
        cum_s = 0.0
        for idx, pt in enumerate(global_path):
            if idx > 0:
                cum_s += np.linalg.norm(np.array(pt) - np.array(global_path[idx - 1]))
            self._global_path_file.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                idx, pt[0], pt[1], pt[2], cum_s))
        self._global_path_file.flush()
        self._write_raw_global_path(plan.get("raw_path", []))

        # ── Local plans CSV ──────────────────────────────────────────
        self._local_plans_file = open(lp_path, "w")
        self._local_plans_file.write(
            "plan_id,source_frame_id,request_timestamp_ns,"
            "state_x,state_y,state_z,state_vx,state_vy,state_vz,"
            "state_ax,state_ay,state_az,state_yaw,"
            "guide_x,guide_y,guide_z,guide_path_index,"
            "local_goal_x,local_goal_y,local_goal_z,"
            "progress_s,progress_index,local_goal_index,"
            "status,success,planning_time_ms,min_clearance,traj_point_count,"
            "trajectory_duration_s,terminal_scale\n")
        self._open_local_plan_points_file()
        self._image_writer = None
        self._depth_buffer = []  # list of (png_name, depth_u16)
        self._depth_buffer_enabled = bool(
            self.g.get("online_runtime", {}).get("depth_in_memory", True))

        # ── State variables ──────────────────────────────────────────
        goal_pt = plan["goal"]
        goal_np = np.array(goal_pt)
        global_path_length = plan.get("global_path_length", 0.0)
        if self._dynamics is None:
            rospy.logerr("[ONLINE-LOCKSTEP] Dynamics backend unavailable.")
            self._trajectory_exit_reason = "dynamics_backend_unavailable"
            self._st_finish_recording()
            return
        dynamics_state = self._dynamics.get_state()
        cur_pos = dynamics_state.position_world.copy()
        cur_vel = dynamics_state.velocity_world.copy()
        qx, qy, qz, qw = dynamics_state.quaternion_world_body
        cur_yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))

        consecutive_failures = 0
        sample_index = 0
        sent_frame_id = 0
        recording_start_mono = time.monotonic()
        recording_start_epoch_ns = int(time.time() * 1e9)
        last_valid_response_mono = time.monotonic()
        previous_progress_s = -1.0
        goal_hold_counter = 0
        goal_hold_active = False
        previous_command_valid = False
        previous_command_frame_id = -1
        previous_command_actor = "episode_start"
        previous_command_velocity_flu = np.zeros(3, dtype=np.float64)
        previous_command_yaw_rate = 0.0
        previous_guide_mode = None

        # ── v11: online runtime state ────────────────────────────────
        # v13: planner runs every record tick (30 Hz = record rate)
        runtime_loop_hz = rec_hz

        # v11 FIX: speed_scale removed — C++ pybind does not export it.

        rec_cfg = online_rt.get("recovery", {})
        recovery_enabled = bool(rec_cfg.get("enabled", True))
        recovery_lookahead_m = float(rec_cfg.get(
            "path_lookahead_distance_m", 2.0))
        recovery_yaw_deadband = float(rec_cfg.get("yaw_deadband_rad", 0.05))
        recovery_max_yaw_rate = float(rec_cfg.get("max_yaw_rate_rps", 0.80))
        recovery_yaw_gain = float(rec_cfg.get("yaw_gain", 1.50))
        recovery_max_duration_s = float(rec_cfg.get(
            "maximum_recovery_duration_s", 3.0))
        recovery_max_steps = int(rec_cfg.get(
            "maximum_recovery_control_steps", 90))
        recovery_tie_break = str(rec_cfg.get("tie_break_direction", "right"))

        # v11: cached plan state
        self._latest_plan_snapshot = None  # LocalPlanSnapshot or None
        planner_mode = PlannerMode.FRESH_PLAN
        previous_planner_mode = PlannerMode.FRESH_PLAN  # for recovery-entry detection
        control_mode = ControlMode.TRACK_TRAJECTORY
        trend_mode = TrendMode.TRACK_GUIDE
        # v11: recovery state
        recovery_elapsed_s = 0.0
        recovery_step_count = 0
        recovery_last_direction = ""  # "left" or "right"
        recovery_target_world = np.zeros(3, dtype=np.float64)
        recovery_target_path_index = -1
        recovery_azimuth_rad = 0.0  # v11 FIX: computed from atan2, used for yaw-rate

        # v11: episode stats
        self._fresh_plan_frame_count = 0
        self._cached_plan_frame_count = 0
        self._recovery_frame_count = 0
        self._planner_attempt_count = 0
        self._planner_success_count = 0
        self._planner_failure_count = 0
        self._planner_retry_success_count = 0
        self._deadline_retime_success_count = 0
        self._last_planner_rejection_status = ""
        self._last_planner_rejection_message = ""
        self._recovery_entry_count = 0
        self._recovery_success_count = 0
        self._recovery_timeout_count = 0
        self._max_plan_age_s = 0.0
        self._max_guide_cache_age_s = 0.0
        self._recover_left_frame_count = 0
        self._recover_right_frame_count = 0
        self._normal_guide_frame_count = 0
        self._goal_hold_frame_count = 0
        self._recovery_exit_count = 0
        self._horizontal_class_counts = [0] * TREND_HORIZONTAL_CLASS_COUNT
        self._vertical_class_counts = [0] * trend_v_bins
        self._guide_value_zero_count = 0
        self._guide_value_saturated_count = 0
        self._global_direction_invalid_count = 0
        self._sequence_reset_count = 0
        filtered_planner_acceleration = np.zeros(3, dtype=np.float64)

        rospy.loginfo("[ONLINE-LOCKSTEP] Starting. runtime_loop=%.0fHz dt_sample=%.3fs",
                      runtime_loop_hz, dt_sample)

        while not rospy.is_shutdown():
            if self._timed_out():
                rospy.logwarn("[ONLINE-LOCKSTEP] Timeout.")
                self._trajectory_exit_reason = "trajectory_timeout"
                break

            # ── Step 1: Request depth for current state ──────────────
            # The depth image and state fields belong to x_t.
            t_request_mono = time.monotonic()
            frame_id = sent_frame_id
            sent_frame_id += 1

            dynamics_state = self._dynamics.get_state()
            cur_pos = dynamics_state.position_world.copy()
            cur_vel = dynamics_state.velocity_world.copy()
            raw_planner_acceleration = np.asarray(
                dynamics_state.acceleration_world, dtype=np.float64)
            if not np.all(np.isfinite(raw_planner_acceleration)):
                raw_planner_acceleration = np.zeros(3, dtype=np.float64)
            raw_acceleration_norm = float(np.linalg.norm(
                raw_planner_acceleration))
            if raw_acceleration_norm > max_acceleration:
                raw_planner_acceleration *= (
                    max_acceleration / raw_acceleration_norm)
            acceleration_filter_alpha = 1.0 - math.exp(
                -dt_sample /
                max(1.0e-3, planner_acceleration_filter_tau))
            filtered_planner_acceleration += acceleration_filter_alpha * (
                raw_planner_acceleration - filtered_planner_acceleration)
            qx, qy, qz, qw = dynamics_state.quaternion_world_body
            cur_yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                                 1.0 - 2.0 * (qy * qy + qz * qz))
            goal_hold_was_active = goal_hold_active
            goal_hold_active = update_goal_hold_latch(
                goal_hold_active, cur_pos, goal_np, goal_tolerance,
                current_speed_mps=float(np.linalg.norm(cur_vel)),
                goal_speed_tolerance_mps=goal_speed_tol)
            if goal_hold_active and not goal_hold_was_active:
                consecutive_failures = 0
                rospy.loginfo(
                    "[ONLINE-LOCKSTEP] Entering terminal GOAL_HOLD at "
                    "sample %d (distance %.3f m).",
                    sample_index, float(np.linalg.norm(cur_pos - goal_np)))
            elif goal_hold_was_active and not goal_hold_active:
                goal_hold_counter = 0
                rospy.logwarn(
                    "[ONLINE-LOCKSTEP] Releasing terminal GOAL_HOLD at "
                    "sample %d: vehicle stopped %.3f m from the goal, "
                    "outside the %.3f m tolerance; planner will reacquire.",
                    sample_index, float(np.linalg.norm(cur_pos - goal_np)),
                    goal_tolerance)
            vehicle = make_depth_vehicle(
                cur_pos.tolist(), float(cur_yaw), self._depth_cfg,
                quaternion_xyzw=dynamics_state.quaternion_world_body)
            msg = {"scene_id": self.g["scene_id"], "frame_id": frame_id,
                   "vehicles": [vehicle], "objects": self.current_obj_list}
            self.bridge.send_pose(msg)

            # ── Step 2: Wait for exact frame_id depth response ───────
            depth_u16, collision, recv_frame_id, recv_time_mono = \
                self._wait_for_exact_depth_frame(
                    frame_id, depth_float_len, img_w, img_h,
                    depth_max_m, unity_response_timeout_s,
                    last_valid_response_mono)
            if depth_u16 is None and recv_frame_id is None:
                # timeout
                self._trajectory_exit_reason = "unity_response_timeout"
                rospy.logerr("[ONLINE-LOCKSTEP] Unity response timeout for frame_id=%d.",
                             frame_id)
                break
            if depth_u16 is None:
                # received wrong frame_id, continue loop
                continue
            last_valid_response_mono = recv_time_mono
            latency_ms = (recv_time_mono - t_request_mono) * 1000.0
            self._rec_raw_received_frames += 1
            self._rec_exact_matches += 1

            # ── Save depth PNG ───────────────────────────────────────
            png_name = "none"
            depth_m = None
            if depth_u16 is not None:
                depth_m = depth_u16.astype(np.float64) / 65535.0 * depth_max_m
                png_name = "{:06d}.png".format(sample_index)

            # ── Phase 2: Integrate depth into observed map ─────────
            t_map_start = time.monotonic()
            guide_sel = GuideSelection()
            observability_result = None
            ref_segment = []
            t_int_ms = 0.0
            t_esdf_ms = 0.0
            t_guide_ms = 0.0
            if self._use_observed_map and self._observed_map is not None and depth_m is not None:
                timestamp_s = sample_index * dt_sample
                if self._observed_map.recenter_if_needed(cur_pos):
                    self._observed_esdf.reset()
                    if self._observed_esdf_cache is not None:
                        self._observed_esdf_cache.reset()
                t_int_start = time.monotonic()
                self._observed_map.integrate_depth(
                    depth_m, cur_pos.tolist(),
                    dynamics_state.quaternion_world_body, timestamp_s)
                t_int_ms = (time.monotonic() - t_int_start) * 1000.0

                # Rebuild observed ESDF
                map_revision = self._observed_map.get_revision()
                cache_requests_rebuild = (
                    self._observed_esdf_cache is None or
                    self._observed_esdf_cache.should_rebuild(map_revision))
                if (cache_requests_rebuild and
                        (map_revision % self._observed_esdf.rebuild_every_n_frames == 0 or
                         not self._observed_esdf.is_built())):
                    t_esdf_start = time.monotonic()
                    occ = self._observed_map.get_occupancy()
                    known = self._observed_map.get_known_mask()
                    self._observed_esdf.rebuild(
                        occ, known,
                        self._observed_map.get_origin(),
                        self._observed_map.get_resolution())
                    if self._observed_esdf_cache is not None:
                        self._observed_esdf_cache.on_rebuilt(
                            map_revision, self._observed_esdf.get_esdf(),
                            self._observed_esdf.get_known_mask(),
                            self._observed_esdf.get_origin(),
                            self._observed_esdf.get_resolution())
                    t_esdf_ms = (time.monotonic() - t_esdf_start) * 1000.0

                if (self._obs_auditor is not None and
                        self._current_task_validation is not None and
                        self._current_dominant_obstacle is not None):
                    observability_result = self._obs_auditor.audit(
                        self._current_task_validation.lower_cost_side,
                        self._observed_map, self._observed_esdf,
                        self._current_dominant_obstacle, cur_pos, cur_yaw,
                        np.asarray(plan["start"], dtype=np.float64), goal_np,
                        self._invalid_obs_frame_count)
                    if observability_result.observable_expert_label:
                        self._invalid_obs_frame_count = 0
                    else:
                        self._invalid_obs_frame_count = \
                            observability_result.invalid_observability_frame_count
                    if observability_result.observability_check_triggered:
                        self._observability_trigger_count += 1
                    if observability_result.side_choice_consistent:
                        self._observability_consistent_count += 1

            # Select Guide/Terminal from the current depth frame regardless of
            # whether observed occupancy fusion is enabled. The quaternion is
            # used for the full world-to-camera FLU transform.
            if goal_hold_active:
                # Terminal HOLD owns the vehicle once goal tolerance has been
                # entered.  Do not ask GuideSelector for a direction that may
                # point behind the camera after a small inertial overshoot.
                guide_sel.guide_position_world = cur_pos.copy()
                guide_sel.guide_path_index = len(global_path) - 1
                guide_sel.progress_path_index = len(global_path) - 1
                guide_sel.terminal_position_world = cur_pos.copy()
                guide_sel.terminal_path_index = len(global_path) - 1
                guide_sel.guide_is_final = True
                guide_sel.selection_mode = "goal_tolerance_hold"
                self._consecutive_guide_failures = 0
            elif self._guide_selector is not None:
                t_guide_start = time.monotonic()
                guide_sel = self._guide_selector.select(
                    global_path, self._guide_progress_index,
                    cur_pos, float(cur_yaw), cur_vel, depth_m,
                    self._observed_map, self._observed_esdf,
                    dynamics_state.quaternion_world_body,
                    goal_position_world=goal_np)
                t_guide_ms = (time.monotonic() - t_guide_start) * 1000.0
                if guide_sel.valid:
                    self._guide_progress_index = max(
                        self._guide_progress_index,
                        guide_sel.progress_path_index)
                    self._consecutive_guide_failures = 0
                else:
                    self._consecutive_guide_failures += 1
            t_map_ms = (time.monotonic() - t_map_start) * 1000.0
            # Diagnostic: log Phase-2 timing breakdown when it exceeds 100 ms
            if t_map_ms > 100.0:
                _parts = ["map=%.0fms" % t_map_ms,
                          "int=%.0fms" % t_int_ms]
                try:
                    _parts.append("esdf=%.0fms" % t_esdf_ms)
                except NameError:
                    pass
                try:
                    _parts.append("guide=%.0fms" % t_guide_ms)
                except NameError:
                    pass
                rospy.logwarn("[LOCKSTEP] Phase-2 timing: %s (frame %d)",
                              ", ".join(_parts), sample_index)

            obs_cfg_runtime = self.g.get("scene_generation", {}).get(
                "observability_audit", {})
            if (obs_cfg_runtime.get("enabled", False) and
                    self._invalid_obs_frame_count >= int(obs_cfg_runtime.get(
                        "maximum_invalid_frames_before_reject", 5))):
                self._trajectory_exit_reason = "runtime_side_choice_unobservable"
                rospy.logerr("[ONLINE-LOCKSTEP] Runtime side choice remained "
                             "unobservable for %d frames.",
                             self._invalid_obs_frame_count)
                break

            # ── Step 3: Planner (v11: every frame at 30 Hz) ──────────

            result = None
            plan_success = False
            planner_compute_ms = 0.0
            planner_status_str = "NO_PLAN"
            planner_used_obs = 0
            planner_used_fallback = 0
            plan_is_fresh = 0
            plan_cache_valid_int = 0
            plan_age_s = 0.0
            plan_source_frame_id = -1
            terminal_scale_used = 1.0
            planner_retry_count = 0
            planner_attempted_int = 0

            trajectory_time_s = sample_index * dt_sample

            # ── Every frame runs planner (v11: 30 Hz planner = 30 Hz control) ──
            if goal_hold_active:
                planner_status_str = "GOAL_HOLD"
                consecutive_failures = 0
            elif self._cpp_planner is not None:
                # Project state to global path for progress tracking
                prog_idx, prog_s = self._project_to_global_path(
                    cur_pos, global_path, self._guide_progress_index)
                if prog_idx >= 0:
                    self._guide_progress_index = max(
                        self._guide_progress_index, prog_idx)

                # v11: Guide visible → plan trajectory; not visible → recovery (no plan needed)
                if guide_sel.valid:
                    planner_attempted_int = 1
                    self._planner_attempt_count += 1
                    self._total_replans += 1
                    # ── Normal: visible Guide → plan trajectory ──────
                    # Trend and Control use the exact same target. A planning
                    # failure is a label failure; never shorten the target to
                    # manufacture a different successful action.
                    scales_for_plan = [1.0]

                    for ts in scales_for_plan:
                        planner_retry_count += 1

                        scaled_terminal = np.asarray(
                            guide_sel.guide_position_world,
                            dtype=np.float64)
                        scaled_terminal_idx = guide_sel.guide_path_index
                        # No reference_path_segment — local planner uses
                        # straight-line initialization, not global path sub-segment

                        t_plan_start = time.monotonic()
                        try:
                            planner_boundary_acceleration = \
                                self._compute_planner_boundary_acceleration(
                                    measured_acceleration_world=
                                    filtered_planner_acceleration,
                                    current_position_world=cur_pos,
                                    current_velocity_world=cur_vel,
                                    guide_position_world=
                                    guide_sel.guide_position_world,
                                    guide_is_final=guide_sel.guide_is_final,
                                    previous_snapshot=
                                    self._latest_plan_snapshot,
                                    current_time_s=trajectory_time_s,
                                    command_lookahead_time_s=
                                    command_lookahead_time,
                                    nominal_speed=nominal_speed,
                                    max_velocity=max_velocity,
                                    max_acceleration=max_acceleration,
                                    lookahead_distance=float(lp_cfg.get(
                                        "lookahead_distance", 4.0)),
                                    goal_tolerance=goal_tolerance,
                                    max_plan_age_s=float(lp_cfg.get(
                                        "max_plan_age", 0.75)),
                                    ramp_time_s=
                                    planner_velocity_ramp_time_s,
                                    warm_start_blend=
                                    planner_acceleration_warm_start_blend)
                            state = _VehicleState()
                            state.position = (
                                float(cur_pos[0]), float(cur_pos[1]),
                                float(cur_pos[2]))
                            state.velocity = (
                                float(cur_vel[0]), float(cur_vel[1]),
                                float(cur_vel[2]))
                            state.acceleration = (
                                float(planner_boundary_acceleration[0]),
                                float(planner_boundary_acceleration[1]),
                                float(planner_boundary_acceleration[2]))
                            state.yaw = float(cur_yaw)
                            state.yaw_rate = 0.0

                            req = _LocalPlanningRequest()
                            req.state = state
                            req.previous_progress_s = previous_progress_s
                            req.guide_waypoint = (
                                float(guide_sel.guide_position_world[0]),
                                float(guide_sel.guide_position_world[1]),
                                float(guide_sel.guide_position_world[2]))
                            req.guide_waypoint_index = guide_sel.guide_path_index
                            req.trajectory_terminal = (
                                float(scaled_terminal[0]),
                                float(scaled_terminal[1]),
                                float(scaled_terminal[2]))
                            req.trajectory_terminal_index = scaled_terminal_idx
                            # reference_path_segment is NOT populated —
                            # the local planner uses straight-line initialization
                            req.forbid_unknown_space = False
                            req.allow_global_map_fallback = False
                            result = self._cpp_planner.plan_local_with_request(req)

                            t_plan_end = time.monotonic()
                            planner_compute_ms = (t_plan_end - t_plan_start) * 1000.0
                            self._planning_times_ms.append(
                                result.planning_time_ms
                                if hasattr(result, 'planning_time_ms')
                                else planner_compute_ms)
                            # Persist every planner result, including rejected
                            # trajectories, so the debug view can show the
                            # actual failing purple trajectory.
                            if result is not None:
                                self._write_local_plan_summary(
                                    result, state,
                                    int(t_request_mono * 1e9),
                                    source_frame_id=frame_id,
                                    terminal_scale=ts)

                            if result is not None and result.success:
                                plan_success = True
                                terminal_scale_used = ts
                                planner_status_str = str(result.status)
                                if planner_retry_count > 1:
                                    self._planner_retry_success_count += 1
                                if "final retime" in str(result.message):
                                    self._deadline_retime_success_count += 1
                                break
                            else:
                                planner_status_str = (
                                    str(result.status)
                                    if result is not None
                                    else "PLAN_FAILED")
                                self._last_planner_rejection_status = \
                                    planner_status_str
                                self._last_planner_rejection_message = (
                                    str(result.message)
                                    if result is not None
                                    else "no result")
                                rospy.logwarn_throttle(
                                    1.0,
                                    "[LOCAL-PLAN] rejected: frame=%d "
                                    "status=%s message=%s guide_idx=%d "
                                    "guide_dist=%.2fm guide_azimuth=%.1fdeg",
                                    frame_id, planner_status_str,
                                    str(result.message)
                                    if result is not None else "no result",
                                    int(guide_sel.guide_path_index),
                                    float(guide_sel.guide_distance_m),
                                    math.degrees(
                                        float(guide_sel.azimuth_rad)))
                        except Exception as exc:
                            rospy.logerr(
                                "[ONLINE-LOCKSTEP] Planner exception: %s", exc)
                            planner_status_str = "PLANNER_EXCEPTION"

                        if plan_success:
                            break

                    # The CSV field counts retries after the initial request.
                    planner_retry_count = max(0, planner_retry_count - 1)

                    if plan_success:
                        self._successful_replans += 1
                        self._planner_success_count += 1
                        plan_is_fresh = 1
                        plan_source_frame_id = frame_id
                        previous_progress_s = max(
                            previous_progress_s, result.progress_s)
                        self._executed_clearances.append(result.min_clearance)

                        snapshot = LocalPlanSnapshot(
                            plan_id=int(result.plan_id),
                            plan_timestamp_s=trajectory_time_s,
                            source_frame_id=frame_id,
                            source_state_timestamp_s=trajectory_time_s,
                            guide_world=guide_sel.guide_position_world.copy(),
                            guide_path_index=guide_sel.guide_path_index,
                            guide_is_final=guide_sel.guide_is_final,
                            terminal_world=np.array(
                                [float(result.local_goal[0]),
                                 float(result.local_goal[1]),
                                 float(result.local_goal[2])],
                                dtype=np.float64),
                            terminal_path_index=result.local_goal_index,
                            reference_path_start_index=self._guide_progress_index,
                            reference_path_end_index=result.local_goal_index,
                            trajectory=result.trajectory,
                            trajectory_duration_s=float(
                                result.trajectory[-1].t)
                            if len(result.trajectory) > 0 else 0.0,
                            planner_status=planner_status_str,
                            minimum_clearance_m=float(result.min_clearance),
                            terminal_scale=terminal_scale_used,
                        )
                        self._latest_plan_snapshot = snapshot
                        planner_mode = PlannerMode.FRESH_PLAN

                    else:
                        self._failed_replans += 1
                        self._planner_failure_count += 1
                else:
                    # ── Recovery: no visible guide → rotate in place, NO planner call ──
                    # During an active avoidance commitment, keep exploring
                    # toward that side.  Otherwise rotate toward the mission
                    # goal.  No trajectory is labelled from this state.
                    if guide_sel.recovery_target_valid:
                        recovery_target_world = \
                            guide_sel.recovery_target_world.copy()
                        recovery_target_path_index = \
                            self._guide_progress_index
                    else:
                        recovery_target_world = goal_np.copy()
                        recovery_target_path_index = len(global_path) - 1
                    planner_status_str = "RECOVERY_NO_GUIDE"
                    rospy.logwarn_throttle(
                        1.0,
                        "[GUIDE] unavailable: frame=%d reason=%s "
                        "candidates=%d progress_idx=%d",
                        frame_id, str(guide_sel.rejection_reason),
                        int(guide_sel.candidate_count),
                        int(self._guide_progress_index))
                    # planner statistics: not a planner attempt (no C++ call)
                    # below handles whether to enter RECOVERY or use cached plan.

            # ── v11: Determine final mode from snapshot validity ──────
            plan_cache_valid_int = 0
            plan_age_s = 0.0
            previous_planner_mode = planner_mode

            if self._latest_plan_snapshot is not None:
                plan_age_s = trajectory_time_s - \
                    self._latest_plan_snapshot.plan_timestamp_s
                self._max_plan_age_s = max(self._max_plan_age_s, plan_age_s)

            if goal_hold_active:
                planner_mode = PlannerMode.GOAL_HOLD
                recovery_elapsed_s = 0.0
                recovery_step_count = 0
                recovery_last_direction = ""
                recovery_azimuth_rad = 0.0
                recovery_target_world = goal_np.copy()
                recovery_target_path_index = len(global_path) - 1
            elif guide_sel.valid and plan_is_fresh:
                planner_mode = PlannerMode.FRESH_PLAN
            elif guide_sel.valid:
                # A visible Guide requires a fresh Control label generated
                # from that exact Guide on this frame. Cached trajectories and
                # RECOVERY actions are not substitute labels.
                planner_mode = PlannerMode.ABORT
            elif recovery_enabled:
                planner_mode = PlannerMode.RECOVERY
                if previous_planner_mode != PlannerMode.RECOVERY:
                    self._recovery_entry_count += 1
                    recovery_elapsed_s = 0.0
                    recovery_step_count = 0
                    if guide_sel.recovery_target_valid:
                        recovery_target_world = \
                            guide_sel.recovery_target_world.copy()
                        recovery_target_path_index = \
                            self._guide_progress_index
                    else:
                        recovery_target_world = goal_np.copy()
                        recovery_target_path_index = len(global_path) - 1
            else:
                planner_mode = PlannerMode.ABORT

            plan_success = bool(
                plan_is_fresh and planner_mode == PlannerMode.FRESH_PLAN)

            # Trend is a perception label: it follows current Guide
            # visibility, independently of whether the local optimizer found
            # a Control trajectory on this tick.
            recovery_timed_out = False
            trend_mode = (
                TrendMode.GOAL_HOLD
                if goal_hold_active else (
                    TrendMode.TRACK_GUIDE
                    if guide_sel.valid else TrendMode.RECOVERY))
            if planner_mode == PlannerMode.GOAL_HOLD:
                control_mode = ControlMode.HOLD_POSITION
            elif planner_mode == PlannerMode.RECOVERY:
                control_mode = ControlMode.ROTATE_IN_PLACE
                # v12: compute recovery direction BEFORE row build
                # so Trend label and yaw-rate use the same azimuth
                recovery_direction, recovery_azimuth_rad = \
                    self._compute_recovery_direction(
                        recovery_target_world, cur_pos,
                        dynamics_state.quaternion_world_body,
                        recovery_yaw_deadband,
                        recovery_last_direction, recovery_tie_break)
                recovery_last_direction = recovery_direction
            elif planner_mode in (PlannerMode.FRESH_PLAN, PlannerMode.CACHED_PLAN):
                control_mode = ControlMode.TRACK_TRAJECTORY
            else:
                control_mode = ControlMode.EMERGENCY_STOP

            # ── v13: Compute control ONCE, build RuntimeDecision ────
            # Normal tracking: compute trajectory tracking command
            recovery_timeout_pending = (
                planner_mode == PlannerMode.RECOVERY and
                (recovery_elapsed_s + dt_sample >= recovery_max_duration_s or
                 recovery_step_count + 1 >= recovery_max_steps))
            tracking_cmd = None
            if (self._latest_plan_snapshot is not None and
                    planner_mode in (PlannerMode.FRESH_PLAN, PlannerMode.CACHED_PLAN)):
                tracking_cmd = self._compute_trajectory_tracking_command(
                    self._latest_plan_snapshot, trajectory_time_s,
                    cur_pos, cur_vel, cur_yaw,
                    dynamics_state.quaternion_world_body,
                    command_lookahead_time=command_lookahead_time)

            if planner_mode == PlannerMode.GOAL_HOLD:
                ref_vel_flu = np.zeros(3, dtype=np.float64)
                fb_vel_flu = np.zeros(3, dtype=np.float64)
                final_vel_flu = np.zeros(3, dtype=np.float64)
                final_yr = 0.0
                traj_sample_t = -1.0
                sel_actor = "goal_hold"
            elif planner_mode == PlannerMode.RECOVERY:
                # Apply the angular deadband to the action as well as the
                # left/right label hysteresis.  Without this, a small positive
                # azimuth can retain a RIGHT label while commanding left yaw.
                if abs(recovery_azimuth_rad) <= recovery_yaw_deadband:
                    recovery_yaw_rate = 0.0
                else:
                    recovery_yaw_rate = max(
                        -recovery_max_yaw_rate,
                        min(recovery_max_yaw_rate,
                            recovery_yaw_gain * recovery_azimuth_rad))
                ref_vel_flu = np.zeros(3, dtype=np.float64)
                fb_vel_flu = np.zeros(3, dtype=np.float64)
                final_vel_flu = np.zeros(3, dtype=np.float64)
                final_yr = (
                    0.0 if recovery_timeout_pending else recovery_yaw_rate)
                traj_sample_t = -1.0
                sel_actor = (
                    "recovery_timeout_stop"
                    if recovery_timeout_pending else "recovery_controller")
            elif tracking_cmd is not None and tracking_cmd.get("valid"):
                ref_vel_flu = tracking_cmd["reference_velocity_flu"].copy()
                fb_vel_flu = tracking_cmd["feedback_velocity_flu"].copy()
                final_vel_flu = tracking_cmd["final_velocity_flu"].copy()
                final_yr = tracking_cmd["final_yaw_rate"]
                traj_sample_t = tracking_cmd["trajectory_sample_time_s"]
                sel_actor = "expert"
            else:
                ref_vel_flu = np.zeros(3, dtype=np.float64)
                fb_vel_flu = np.zeros(3, dtype=np.float64)
                final_vel_flu = np.zeros(3, dtype=np.float64)
                final_yr = 0.0
                traj_sample_t = -1.0
                sel_actor = "safety"

            if planner_mode == PlannerMode.GOAL_HOLD:
                decision = make_goal_hold_decision(
                    cur_pos, goal_np, len(global_path) - 1,
                    plan_snapshot=self._latest_plan_snapshot)
            else:
                decision = RuntimeDecision(
                    planner_mode=planner_mode.value,
                    trend_mode=trend_mode.value,
                    control_mode=control_mode.value,
                    guide_source=(
                        "current_depth_goal_explorer"
                        if guide_sel.valid
                        else (
                            "recovery_explore_direction"
                            if guide_sel.recovery_target_valid
                            else "recovery_goal_direction")),
                    guide_target_world=(
                        guide_sel.guide_position_world.copy()
                        if guide_sel.valid else recovery_target_world.copy()),
                    guide_target_path_index=(
                        guide_sel.guide_path_index
                        if guide_sel.valid else recovery_target_path_index),
                    recovery_direction=recovery_last_direction,
                    recovery_azimuth_rad=recovery_azimuth_rad,
                    plan_snapshot=self._latest_plan_snapshot,
                    recovery_target_world=recovery_target_world.copy(),
                    recovery_target_path_index=recovery_target_path_index,
                    trajectory_sample_time_s=traj_sample_t,
                    trajectory_reference_velocity_flu=ref_vel_flu.copy(),
                    trajectory_feedback_velocity_flu=fb_vel_flu.copy(),
                    expert_velocity_flu=final_vel_flu.copy(),
                    expert_yaw_rate=final_yr,
                    selected_velocity_flu=final_vel_flu.copy(),
                    selected_yaw_rate=final_yr,
                    selected_actor=sel_actor,
                )

            # ── Step 4: Build training row (v13) ────────────────────
            row = self._build_training_row_v16(
                cur_pos, cur_vel, cur_yaw,
                decision,
                plan_success, planner_compute_ms,
                planner_status_str,
                goal_np, goal_pt, global_path_length,
                max_guide_range,
                png_name, collision,
                plan, sample_index, dt_sample,
                frame_id, recording_start_mono, recording_start_epoch_ns,
                t_request_mono, recv_time_mono, latency_ms,
                h_fov_rad, v_fov_rad,
                trend_normal_h_bins, trend_v_bins,
                trend_sigma_bins, normal_h_bin_edges, v_bin_edges,
                azi_soft_names, ele_soft_names,
                previous_progress_s,
                guide_sel, ref_segment,
                self._observed_map, self._observed_esdf,
                planner_used_obs, planner_used_fallback,
                dynamics_state.quaternion_world_body,
                # v11 extras
                plan_is_fresh, plan_cache_valid_int,
                plan_source_frame_id, plan_age_s,
                True, planner_attempted_int,
                planner_retry_count, terminal_scale_used,
                recovery_elapsed_s,
                self._guide_progress_index, global_path,
            )
            if result is not None and planner_attempted_int:
                # The row builder normally reads the successful immutable
                # snapshot. For a rejected request, use the current result
                # instead of leaking the preceding plan's diagnostics.
                row["plan_id"] = int(result.plan_id)
                row["local_goal_index"] = int(result.local_goal_index)
                row["planner_min_clearance"] = round(
                    float(result.min_clearance), 4)
                row["global_progress_s"] = round(
                    float(result.progress_s), 4)
                row["global_progress_index"] = int(result.progress_index)
                row["plan_source_frame_id"] = int(frame_id)
            current_guide_mode = row["guide_mode"]
            if previous_guide_mode is not None:
                mode_changed = current_guide_mode != previous_guide_mode
                row["guide_mode_changed"] = int(mode_changed)
                row["recovery_entered"] = int(
                    mode_changed and previous_guide_mode == "NORMAL" and
                    current_guide_mode != "NORMAL")
                row["recovery_exited"] = int(
                    mode_changed and previous_guide_mode != "NORMAL" and
                    current_guide_mode == "NORMAL")
            for axis, value in zip(
                    ("x", "y", "z"), dynamics_state.angular_velocity_body):
                row["state_angular_velocity_{}_body".format(axis)] = round(
                    float(value), 6)
            row["previous_executed_command_valid"] = int(
                previous_command_valid)
            row["previous_executed_command_frame_id"] = int(
                previous_command_frame_id)
            row["previous_executed_actor"] = previous_command_actor
            row["previous_executed_command_vx_flu"] = round(
                float(previous_command_velocity_flu[0]), 6)
            row["previous_executed_command_vy_flu"] = round(
                float(previous_command_velocity_flu[1]), 6)
            row["previous_executed_command_vz_flu"] = round(
                float(previous_command_velocity_flu[2]), 6)
            row["previous_executed_command_yaw_rate"] = round(
                float(previous_command_yaw_rate), 6)
            row["observability_check_triggered"] = int(
                observability_result is not None and
                observability_result.observability_check_triggered)
            row["observed_left_cost"] = round(
                observability_result.left_observed_path_cost, 6) \
                if observability_result is not None else 0.0
            row["observed_right_cost"] = round(
                observability_result.right_observed_path_cost, 6) \
                if observability_result is not None else 0.0
            row["observed_lower_cost_side"] = (
                observability_result.observed_lower_cost_side
                if observability_result is not None else "")
            row["observed_side_cost_difference_ratio"] = round(
                observability_result.observed_side_cost_difference_ratio, 6) \
                if observability_result is not None else 0.0
            row["observability_consistent"] = int(
                observability_result is not None and
                observability_result.side_choice_consistent)

            learner_output = PolicyOutput() if PolicyOutput is not None else None
            # v13: selected command = expert command = from RuntimeDecision
            candidate_actor = decision.selected_actor
            safety_override = False
            override_reason = ""
            candidate_velocity_flu = decision.selected_velocity_flu.copy()
            candidate_yaw_rate = decision.selected_yaw_rate

            if planner_mode == PlannerMode.GOAL_HOLD:
                # Terminal state-machine ownership has priority over DAgger;
                # learner commands must not restart motion inside the goal.
                pass
            elif self._dagger_ctrl is not None:
                learner_output = self._policy_provider.infer(
                    depth_m.astype(np.float32) if depth_m is not None else None,
                    {"direction_flu": np.array([
                         row["global_dir_x_flu"], row["global_dir_y_flu"],
                         row["global_dir_z_flu"]], dtype=np.float32),
                     "distance_norm": row["global_distance_norm"]},
                    {"vel_flu": world_vector_to_body_flu_quat(
                         cur_vel, dynamics_state.quaternion_world_body),
                     "yaw": cur_yaw,
                     "yaw_rate": float(dynamics_state.angular_velocity_body[2])},
                    float(dynamics_state.simulation_time_s))
                observed_clearance = (self._observed_esdf.value_at(cur_pos)
                                      if self._observed_esdf is not None else None)
                ttc = float("inf")
                if (learner_output is not None and learner_output.valid and
                        self._observed_esdf is not None):
                    safety_cfg = self.g.get("dagger", {}).get("safety", {})
                    horizon = float(safety_cfg.get("prediction_horizon_s", 1.0))
                    min_safe = float(safety_cfg.get(
                        "minimum_observed_clearance_m", 0.35))
                    velocity_world = body_flu_to_world_quat(
                        learner_output.velocity_flu,
                        dynamics_state.quaternion_world_body)
                    for prediction_t in np.linspace(0.0, horizon, 11):
                        prediction_pos = cur_pos + velocity_world * prediction_t
                        clearance = self._observed_esdf.value_at(prediction_pos)
                        clearance = 0.0 if clearance is None else float(clearance)
                        if clearance < min_safe and not np.isfinite(ttc):
                            ttc = float(prediction_t)
                (candidate_actor, safety_override, override_reason,
                 learner_command, learner_yaw_rate) = self._dagger_ctrl.select_actor(
                    bool(row["expert_label_valid"]), learner_output,
                    observed_clearance, ttc,
                    self._policy_provider.hidden_state_valid)
                if learner_command is not None:
                    candidate_velocity_flu = np.asarray(
                        learner_command, dtype=np.float64)
                    candidate_yaw_rate = float(learner_yaw_rate)
            elif not bool(row["expert_label_valid"]):
                candidate_actor = "safety"
                safety_override = True
                override_reason = "expert_action_invalid"
                candidate_velocity_flu = np.zeros(3, dtype=np.float64)
                candidate_yaw_rate = 0.0

            learner_velocity = (learner_output.velocity_flu
                                if learner_output is not None else np.zeros(3))
            if (candidate_actor != decision.selected_actor or
                    not np.array_equal(
                        candidate_velocity_flu,
                        decision.selected_velocity_flu) or
                    candidate_yaw_rate != decision.selected_yaw_rate):
                decision = replace(
                    decision,
                    selected_velocity_flu=candidate_velocity_flu,
                    selected_yaw_rate=candidate_yaw_rate,
                    selected_actor=candidate_actor)
            row.update({
                "learner_output_valid": int(learner_output is not None and learner_output.valid),
                "learner_vx_flu": round(float(learner_velocity[0]), 6),
                "learner_vy_flu": round(float(learner_velocity[1]), 6),
                "learner_vz_flu": round(float(learner_velocity[2]), 6),
                "learner_yaw_rate": round(float(learner_output.yaw_rate) if learner_output is not None else 0.0, 6),
                "learner_inference_ms": round(float(learner_output.inference_ms) if learner_output is not None else 0.0, 3),
                "dagger_round_id": self._dagger_ctrl.round_id if self._dagger_ctrl is not None else -1,
                "dagger_beta": round(self._dagger_ctrl.current_beta if self._dagger_ctrl is not None else 1.0, 6),
                "dagger_random_value": round(self._dagger_ctrl.last_random_value if self._dagger_ctrl is not None else -1.0, 8),
                "initial_selected_actor": self._dagger_ctrl.last_initial_actor if self._dagger_ctrl is not None else "expert",
                "safety_override": int(safety_override),
                "override_reason": override_reason,
                "selected_command_vx_flu": round(
                    float(decision.selected_velocity_flu[0]), 6),
                "selected_command_vy_flu": round(
                    float(decision.selected_velocity_flu[1]), 6),
                "selected_command_vz_flu": round(
                    float(decision.selected_velocity_flu[2]), 6),
                "selected_command_yaw_rate": round(
                    float(decision.selected_yaw_rate), 6),
                "final_executed_actor": decision.selected_actor,
            })

            # ── Step 5: Execute dt_sample to advance to x_(t+1) ──────
            # ── Step 5: Execute dt_sample (v11: recovery / trajectory / hover) ──
            if planner_mode == PlannerMode.RECOVERY:
                # Recovery: zero translation + yaw rotation toward recovery target
                # (trend_mode, control_mode, recovery_direction, yaw-rate already set
                #  in RuntimeDecision before row construction)

                (exec_next_pos, exec_next_vel, exec_next_yaw,
                 exec_next_yaw_rate, executed_command_velocity_flu,
                 executed_command_yaw_rate, executed_command_actor) = \
                    self._execute_fixed_velocity_command_segment(
                        decision.selected_velocity_flu,
                        decision.selected_yaw_rate,
                        dt_sample, dt_ctrl,
                        cur_pos.copy(), cur_vel.copy(), float(cur_yaw),
                        max_velocity, max_acceleration, max_yaw_rate,
                        decision.selected_actor)

                recovery_elapsed_s += dt_sample
                recovery_step_count += 1
                row["recovery_elapsed_s"] = round(recovery_elapsed_s, 6)
                self._recovery_frame_count += 1
                if decision.recovery_direction == "left":
                    self._recover_left_frame_count += 1
                else:
                    self._recover_right_frame_count += 1

                # v11 FIX: check recovery timeout (timers accumulate across ticks)
                if (recovery_elapsed_s >= recovery_max_duration_s or
                        recovery_step_count >= recovery_max_steps):
                    recovery_timed_out = True
                    self._recovery_timeout_count += 1
                    self._trajectory_exit_reason = "RECOVERY_TIMEOUT"
                    rospy.logerr("[ONLINE-LOCKSTEP] Recovery timeout after "
                                 "%.2fs / %d steps.",
                                 recovery_elapsed_s, recovery_step_count)

                # v13: if Guide became visible, the next planner tick will use

            elif (planner_mode in (PlannerMode.FRESH_PLAN, PlannerMode.CACHED_PLAN)
                  and self._latest_plan_snapshot is not None):
                (exec_next_pos, exec_next_vel, exec_next_yaw,
                 exec_next_yaw_rate, executed_command_velocity_flu,
                 executed_command_yaw_rate, executed_command_actor) = \
                    self._execute_fixed_velocity_command_segment(
                        decision.selected_velocity_flu,
                        decision.selected_yaw_rate,
                        dt_sample, dt_ctrl,
                        cur_pos.copy(), cur_vel.copy(), float(cur_yaw),
                        max_velocity, max_acceleration, max_yaw_rate,
                        decision.selected_actor)
                if planner_mode == PlannerMode.FRESH_PLAN:
                    self._fresh_plan_frame_count += 1
                else:
                    self._cached_plan_frame_count += 1
            elif decision.selected_actor == "learner":
                (exec_next_pos, exec_next_vel, exec_next_yaw,
                 exec_next_yaw_rate, executed_command_velocity_flu,
                 executed_command_yaw_rate, executed_command_actor) = \
                    self._execute_fixed_velocity_command_segment(
                        decision.selected_velocity_flu,
                        decision.selected_yaw_rate,
                        dt_sample, dt_ctrl,
                        cur_pos.copy(), cur_vel.copy(), float(cur_yaw),
                        max_velocity, max_acceleration, max_yaw_rate,
                        decision.selected_actor)
            else:
                # Safety actor or no valid plan: hover
                (exec_next_pos, exec_next_vel, exec_next_yaw,
                 exec_next_yaw_rate, executed_command_velocity_flu,
                 executed_command_yaw_rate, executed_command_actor) = \
                    self._execute_fixed_velocity_command_segment(
                        decision.selected_velocity_flu,
                        decision.selected_yaw_rate,
                        dt_sample, dt_ctrl,
                        cur_pos.copy(), cur_vel.copy(), float(cur_yaw),
                        max_velocity, max_acceleration, max_yaw_rate,
                        decision.selected_actor)

            # Patch executed_next fields with post-execution values
            row["executed_next_x"] = round(float(exec_next_pos[0]), 4)
            row["executed_next_y"] = round(float(exec_next_pos[1]), 4)
            row["executed_next_z"] = round(float(exec_next_pos[2]), 4)
            row["executed_next_vx_world"] = round(float(exec_next_vel[0]), 4)
            row["executed_next_vy_world"] = round(float(exec_next_vel[1]), 4)
            row["executed_next_vz_world"] = round(float(exec_next_vel[2]), 4)
            actual_state = self._dynamics.get_state()
            exec_next_vel_flu = world_vector_to_body_flu_quat(
                exec_next_vel, actual_state.quaternion_world_body)
            row["executed_next_vx_flu"] = round(float(exec_next_vel_flu[0]), 4)
            row["executed_next_vy_flu"] = round(float(exec_next_vel_flu[1]), 4)
            row["executed_next_vz_flu"] = round(float(exec_next_vel_flu[2]), 4)
            row["actual_next_vx_flu"] = round(float(exec_next_vel_flu[0]), 4)
            row["actual_next_vy_flu"] = round(float(exec_next_vel_flu[1]), 4)
            row["actual_next_vz_flu"] = round(float(exec_next_vel_flu[2]), 4)
            row["executed_next_yaw"] = round(float(exec_next_yaw), 6)
            row["executed_next_yaw_rate"] = round(float(exec_next_yaw_rate), 6)

            for axis, value in zip(("x", "y", "z"), actual_state.acceleration_world):
                row["actual_acceleration_{}_world".format(axis)] = round(float(value), 5)
            for axis, value in zip(("x", "y", "z"), actual_state.acceleration_flu):
                row["actual_acceleration_{}_flu".format(axis)] = round(float(value), 5)
            for axis, value in zip(("x", "y", "z"), actual_state.angular_velocity_body):
                row["actual_angular_velocity_{}".format(axis)] = round(float(value), 6)
            row["velocity_tracking_error"] = round(float(np.linalg.norm(
                exec_next_vel_flu - decision.selected_velocity_flu)), 6)
            row["yaw_rate_tracking_error"] = round(
                abs(float(exec_next_yaw_rate) - decision.selected_yaw_rate), 6)

            # ── v12: applied command (current frame) ─────────────────
            row["applied_command_vx_flu"] = round(
                float(executed_command_velocity_flu[0]), 6)
            row["applied_command_vy_flu"] = round(
                float(executed_command_velocity_flu[1]), 6)
            row["applied_command_vz_flu"] = round(
                float(executed_command_velocity_flu[2]), 6)
            row["applied_command_yaw_rate"] = round(
                float(executed_command_yaw_rate), 6)
            row["applied_command_actor"] = executed_command_actor
            row["applied_command_valid"] = int(
                np.all(np.isfinite(executed_command_velocity_flu)) and
                np.isfinite(executed_command_yaw_rate))

            # ── v12: FLU angular velocity (current frame, before action) ──
            omega_body = dynamics_state.angular_velocity_body
            omega_flu = np.array([
                omega_body[1], -omega_body[0], omega_body[2]
            ], dtype=np.float64)
            row["state_angular_velocity_x_flu"] = round(float(omega_flu[0]), 6)
            row["state_angular_velocity_y_flu"] = round(float(omega_flu[1]), 6)
            row["state_angular_velocity_z_flu"] = round(float(omega_flu[2]), 6)

            # ── v12: FLU gravity direction (world [0,0,-1] → FLU) ──
            g_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            g_flu = world_vector_to_body_flu_quat(
                g_world, dynamics_state.quaternion_world_body)
            row["gravity_direction_x_flu"] = round(float(g_flu[0]), 6)
            row["gravity_direction_y_flu"] = round(float(g_flu[1]), 6)
            row["gravity_direction_z_flu"] = round(float(g_flu[2]), 6)

            # v11: A frame is only invalid when it truly cannot produce
            # safe, finite, semantically clear Trend and Control labels.
            # Transient planner failures are NOT automatically invalid when
            # a valid cached trajectory or recovery command exists.
            frame_invalid_reasons = []
            if depth_m is None or png_name == "none":
                frame_invalid_reasons.append("missing_depth")
            if bool(collision):
                frame_invalid_reasons.append("collision")
            if not np.all(np.isfinite(
                    dynamics_state.angular_velocity_body)):
                frame_invalid_reasons.append("invalid_image_time_angular_velocity")
            if not np.all(np.isfinite(cur_pos)) or \
                    not np.all(np.isfinite(cur_vel)):
                frame_invalid_reasons.append("invalid_current_state")
            if not bool(row.get("global_direction_valid", 0)):
                frame_invalid_reasons.append("invalid_global_guide_input")
            if guide_sel.valid and not plan_is_fresh:
                frame_invalid_reasons.append(
                    "local_plan_failed_for_visible_guide")
            if (planner_attempted_int and
                    planner_compute_ms > dt_sample * 1000.0):
                frame_invalid_reasons.append("planner_deadline_miss")
            if planner_mode == PlannerMode.ABORT:
                frame_invalid_reasons.append("planner_abort")
            if planner_mode == PlannerMode.GOAL_HOLD:
                self._goal_hold_frame_count += 1
            if planner_mode == PlannerMode.RECOVERY:
                # Recovery frames are valid IF we have a finite recovery target
                if not np.all(np.isfinite(recovery_target_world)):
                    frame_invalid_reasons.append("invalid_recovery_target")
                if not np.isfinite(decision.selected_yaw_rate):
                    frame_invalid_reasons.append("invalid_recovery_command")
                if recovery_timed_out:
                    frame_invalid_reasons.append("recovery_timeout")
            elif planner_mode == PlannerMode.GOAL_HOLD:
                if (decision.trend_mode != TrendMode.GOAL_HOLD.value or
                        decision.control_mode !=
                        ControlMode.HOLD_POSITION.value):
                    frame_invalid_reasons.append(
                        "invalid_goal_hold_runtime_mode")
                if (np.linalg.norm(decision.expert_velocity_flu) > 1.0e-9 or
                        abs(decision.expert_yaw_rate) > 1.0e-9 or
                        np.linalg.norm(decision.selected_velocity_flu) >
                        1.0e-9 or
                        abs(decision.selected_yaw_rate) > 1.0e-9):
                    frame_invalid_reasons.append(
                        "nonzero_goal_hold_command")
            elif planner_mode not in (
                    PlannerMode.FRESH_PLAN, PlannerMode.CACHED_PLAN):
                # Unknown planner mode
                frame_invalid_reasons.append("unknown_planner_mode")
            if not bool(row.get("expert_label_valid", 0)):
                frame_invalid_reasons.append("invalid_expert_label")
            # Check that trend label can be generated
            if not bool(row.get("trend_label_valid", 0)):
                frame_invalid_reasons.append("invalid_trend_label")
            if (not np.all(np.isfinite(executed_command_velocity_flu)) or
                    not np.isfinite(executed_command_yaw_rate)):
                frame_invalid_reasons.append("invalid_executed_command")
            elif (float(np.linalg.norm(executed_command_velocity_flu)) >
                    max_velocity + 1e-6 or
                    abs(float(executed_command_yaw_rate)) >
                    max_yaw_rate + 1e-6):
                frame_invalid_reasons.append("executed_command_out_of_bounds")
            if (not np.all(np.isfinite(actual_state.position_world)) or
                    not np.all(np.isfinite(actual_state.velocity_world)) or
                    not np.all(np.isfinite(actual_state.angular_velocity_body))):
                frame_invalid_reasons.append("invalid_actual_next_state")

            # Preserve order while avoiding duplicate reasons.
            frame_invalid_reasons = list(dict.fromkeys(frame_invalid_reasons))
            row["frame_valid"] = int(not frame_invalid_reasons)
            row["frame_invalid_reason"] = (
                ";".join(frame_invalid_reasons)
                if frame_invalid_reasons else "none")

            if row["guide_mode"] == "NORMAL":
                self._normal_guide_frame_count += 1
            if row["recovery_exited"]:
                self._recovery_exit_count += 1
            if row["frame_valid"]:
                h_class = int(row["trend_horizontal_class_13"])
                if 0 <= h_class < len(self._horizontal_class_counts):
                    self._horizontal_class_counts[h_class] += 1
            if row["frame_valid"]:
                v_class = int(row["guide_elevation_bin"])
                if 0 <= v_class < len(self._vertical_class_counts):
                    self._vertical_class_counts[v_class] += 1
            guide_value = float(row["guide_distance_norm"])
            if abs(guide_value) <= 1e-9:
                self._guide_value_zero_count += 1
            if guide_value >= 1.0 - 1e-9:
                self._guide_value_saturated_count += 1
            if not row["global_direction_valid"]:
                self._global_direction_invalid_count += 1
            if row["sequence_reset"]:
                self._sequence_reset_count += 1
            if frame_invalid_reasons:
                self._episode_invalid_frame_count += 1
                for reason in frame_invalid_reasons:
                    self._episode_invalid_reason_counts[reason] = (
                        self._episode_invalid_reason_counts.get(reason, 0) + 1)

            # track invalid counts for metadata
            if row.get("expert_label_valid", 0) == 0:
                self._invalid_expert_label_count += 1
            if row.get("trend_label_valid", 0) == 0:
                self._invalid_trend_label_count += 1

            # v11: buffer depth in memory, flush to disk at end
            if depth_u16 is not None and self._depth_buffer_enabled:
                self._depth_buffer.append((png_name, depth_u16))
            elif self._image_writer is not None and depth_u16 is not None:
                self._image_writer.submit(
                    os.path.join(depth_dir, png_name), depth_u16)
            self._csv_writer.writerow(row)
            self._rec_written_rows += 1
            previous_guide_mode = current_guide_mode

            # ── Sync diagnostics ─────────────────────────────────────
            self._sync_file.write("{},{},{:.2f},0.00,frame_id_exact,0,0,{},{}\n".format(
                sample_index, frame_id, latency_ms,
                self._rec_exact_matches, self._rec_fallback_matches))

            # Dataset validity is trajectory-level. Stop immediately after
            # recording the diagnostic row; validation will move the whole
            # episode to _failed instead of committing a partial sample.
            if frame_invalid_reasons:
                self._trajectory_exit_reason = (
                    "invalid_training_frame:" +
                    ";".join(frame_invalid_reasons))
                rospy.logerr(
                    "[ONLINE-LOCKSTEP] Invalid training label at frame %d: "
                    "%s. Rejecting the complete trajectory.",
                    frame_id, ";".join(frame_invalid_reasons))
                break

            # Advance state to x_(t+1)
            cur_pos = exec_next_pos
            cur_vel = exec_next_vel
            cur_yaw = exec_next_yaw
            previous_command_valid = bool(
                np.all(np.isfinite(executed_command_velocity_flu)) and
                np.isfinite(executed_command_yaw_rate))
            previous_command_frame_id = frame_id
            previous_command_actor = executed_command_actor
            previous_command_velocity_flu = np.asarray(
                executed_command_velocity_flu, dtype=np.float64).copy()
            previous_command_yaw_rate = float(executed_command_yaw_rate)
            if recovery_timed_out:
                break

            # ── Goal check ───────────────────────────────────────────
            dist_to_goal = float(np.linalg.norm(cur_pos - goal_np))
            speed = float(np.linalg.norm(cur_vel))
            if (goal_hold_active and dist_to_goal <= goal_tolerance and
                    speed <= goal_speed_tol):
                goal_hold_counter += 1
                if goal_hold_counter >= goal_hold_ticks:
                    rospy.loginfo("[ONLINE-LOCKSTEP] Goal reached at sample %d.", sample_index)
                    self._trajectory_reached_goal = True
                    self._trajectory_exit_reason = "goal_reached"
                    break
            else:
                goal_hold_counter = 0

            # ── Consecutive failure check ───────────────────────────
            if consecutive_failures >= effective_failure_limit:
                self._trajectory_exit_reason = "consecutive_planner_failures"
                rospy.logerr("[ONLINE-LOCKSTEP] %d consecutive planner failures; aborting.",
                             consecutive_failures)
                break

            sample_index += 1
            self._rec_sent_control_frames += 1

            # Deterministic simulation steps still need a wall-clock cadence
            # for smooth Unity playback.  Rendering/PNG work can overlap this
            # wait, but the vehicle will never run faster than record_hz.
            next_sample_deadline = recording_start_mono + sample_index * dt_sample
            remaining = next_sample_deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

        self._final_executed_position = cur_pos.tolist()
        self._final_executed_velocity = cur_vel.tolist()
        if self._trajectory_exit_reason == "running":
            self._trajectory_exit_reason = "shutdown" if rospy.is_shutdown() else "loop_exited"
        self._close_open_files()
        self._st_finish_recording()

    # ═══════════════════════════════════════════════════════════════
    #  Lockstep helper methods
    # ═══════════════════════════════════════════════════════════════

    def _warm_up_unity_depth_stream(self, vehicle):
        """Require a non-recorded exact depth reply before lockstep frame 0.

        Unity publishes the latest pose continuously, so old reset/keep-alive
        replies may remain queued after a long point-cloud export.  A unique
        negative frame id distinguishes this readiness probe from both those
        replies and the non-negative training-frame sequence.
        """
        sync_cfg = self.g.get("sync", {})
        attempts = max(
            1, int(sync_cfg.get("unity_startup_warmup_attempts", 8)))
        timeout_s = max(
            0.25, float(sync_cfg.get(
                "unity_startup_warmup_timeout_s",
                sync_cfg.get("unity_response_timeout_s", 2.0))))
        img_w = int(self._depth_cfg["width"])
        img_h = int(self._depth_cfg["height"])
        depth_float_len = img_w * img_h * 4
        depth_max_m = float(self._depth_cfg["max_m"])

        for attempt in range(attempts):
            self._drain_unity_messages()
            probe_frame_id = -1000000 - self.traj_idx * attempts - attempt
            probe = {
                "scene_id": self.g["scene_id"],
                "frame_id": probe_frame_id,
                "vehicles": [vehicle],
                "objects": self.current_obj_list,
            }
            self.bridge.send_pose(probe)
            depth_u16, _, recv_frame_id, _ = \
                self._wait_for_exact_depth_frame(
                    probe_frame_id, depth_float_len, img_w, img_h,
                    depth_max_m, timeout_s, time.monotonic())
            if depth_u16 is not None and recv_frame_id == probe_frame_id:
                # Remove duplicate probe frames produced before the next Pose
                # update so frame 0 starts with a clean receive queue.
                self._drain_unity_messages()
                rospy.loginfo(
                    "[FSM] Unity depth stream ready (warm-up attempt %d/%d).",
                    attempt + 1, attempts)
                return True
            rospy.logwarn(
                "[FSM] Unity depth warm-up attempt %d/%d failed "
                "(frame_id=%d, reply_frame_id=%s, depth=%s).",
                attempt + 1, attempts, probe_frame_id, recv_frame_id,
                "present" if depth_u16 is not None else "missing")

        return False

    def _wait_for_exact_depth_frame(self, target_frame_id, depth_float_len,
                                     img_w, img_h, depth_max_m,
                                     unity_response_timeout_s,
                                     last_valid_response_mono):
        """Wait for a Unity depth response with exact frame_id match.

        Drains stale (non-matching) frames internally until the correct
        frame_id arrives or the deadline expires.

        Returns:
            (depth_u16, collision, recv_frame_id, recv_time_mono)
            If timeout, returns (None, 0, None, None).
        """
        deadline = time.monotonic() + unity_response_timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            r = self.bridge.try_recv()
            if r is None:
                time.sleep(0.001)
                continue

            msg_dict, img_parts = r
            _fid = msg_dict.get("pub_frame_id")
            if _fid is None:
                _fid = msg_dict.get("frame_id")
            recv_time_mono = time.monotonic()

            # Check collision
            collision = 0
            vehicles = msg_dict.get("pub_vehicles", [])
            if vehicles and vehicles[0].get("collision", False):
                collision = 1

            # Verify frame_id match — drain stale frames, keep waiting
            if _fid != target_frame_id:
                rospy.logwarn("[LOCKSTEP] Discarding stale frame_id=%s (expected %d)",
                              _fid, target_frame_id)
                continue

            # Extract depth image
            depth_u16 = None
            for part in img_parts:
                if len(part) >= depth_float_len:
                    raw = part[:depth_float_len]
                    df32 = np.frombuffer(raw, dtype=np.float32).reshape((img_h, img_w))
                    dm = np.flipud(df32 * 100.0)
                    dm = np.nan_to_num(dm, nan=depth_max_m,
                                       posinf=depth_max_m, neginf=0)
                    depth_u16 = np.clip(
                        dm / max(1e-6, depth_max_m) * 65535,
                        0, 65535).astype(np.uint16)
                    break

            return depth_u16, collision, _fid, recv_time_mono

        return None, 0, None, None

    # ═══════════════════════════════════════════════════════════════
    #  v11: Dual-frequency scheduler, cache, recovery helpers
    # ═══════════════════════════════════════════════════════════════

    def _execute_fixed_velocity_command_segment(
            self, command_velocity_flu, command_yaw_rate,
            duration_s, dt_ctrl,
            cur_pos, cur_vel, cur_yaw,
            max_velocity, max_acceleration, max_yaw_rate,
            actor):
        """Execute one immutable upper-level command for a record interval."""
        pos = np.asarray(cur_pos, dtype=np.float64).copy()
        vel = np.asarray(cur_vel, dtype=np.float64).copy()
        yaw = float(cur_yaw)
        total_yaw_change = 0.0
        cmd_vel_flu = np.asarray(
            command_velocity_flu, dtype=np.float64).copy()
        cmd_yaw_rate = float(command_yaw_rate)
        if (not np.all(np.isfinite(cmd_vel_flu)) or
                not np.isfinite(cmd_yaw_rate) or not actor):
            raise ValueError("fixed velocity command must be finite and attributed")

        elapsed = 0.0
        epsilon = 1e-9

        while elapsed < duration_s - epsilon:
            step_dt = min(dt_ctrl, duration_s - elapsed)
            if not self._dynamics.step_velocity_command(
                    cmd_vel_flu, cmd_yaw_rate, step_dt):
                raise RuntimeError("Flightmare step failed")
            ds = self._dynamics.get_state()
            pos, vel = ds.position_world.copy(), ds.velocity_world.copy()
            qx, qy, qz, qw = ds.quaternion_world_body
            yaw = math.atan2(2.0 * (qw*qz + qx*qy),
                             1.0 - 2.0 * (qy*qy + qz*qz))
            yr = float(ds.angular_velocity_body[2])
            total_yaw_change += yr * step_dt
            elapsed += step_dt

        avg_yaw_rate = total_yaw_change / max(duration_s, 1e-9)
        return (pos, vel, yaw, avg_yaw_rate,
                cmd_vel_flu.copy(), float(cmd_yaw_rate), str(actor))

    def _scale_reference_segment(self, ref_segment, global_path,
                                  progress_index, terminal_idx):
        """Truncate reference segment to match a shortened terminal."""
        if not ref_segment or len(ref_segment) < 2:
            return ref_segment
        # Keep points from progress_index to terminal_idx (inclusive)
        result = []
        for pt in ref_segment:
            # Find closest index in global_path
            pt_arr = np.array(pt[:3], dtype=np.float64)
            best_idx = -1
            best_dist = float("inf")
            for i in range(max(0, progress_index),
                           min(len(global_path), terminal_idx + 5)):
                d = float(np.linalg.norm(
                    np.array(global_path[i]) - pt_arr))
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0 and progress_index <= best_idx <= terminal_idx:
                result.append(pt)
        if len(result) < 2:
            return ref_segment[:2]  # fallback to first two points
        return result

    def _project_to_global_path(self, position_world, global_path,
                                 prev_progress_index):
        """Project current position onto the global A* path.

        Searches from ``prev_progress_index - 5`` forward to find the
        closest path point.  Returns (progress_index, progress_s).

        Returns (-1, -1.0) if projection fails.
        """
        if not global_path or len(global_path) < 2:
            return -1, -1.0
        pos = np.asarray(position_world, dtype=np.float64)
        search_start = max(0, prev_progress_index - 5)
        best_idx = search_start
        best_dist = float("inf")
        for i in range(search_start, len(global_path)):
            d = float(np.linalg.norm(np.array(global_path[i]) - pos))
            if d < best_dist:
                best_dist = d
                best_idx = i
        # Compute cumulative arc length
        cum_s = 0.0
        for i in range(1, best_idx + 1):
            cum_s += float(np.linalg.norm(
                np.array(global_path[i]) - np.array(global_path[i - 1])))
        return best_idx, cum_s

    def _is_guide_cache_valid(self, snapshot, current_time_s,
                               progress_index, global_path, cur_pos):
        """Check whether the cached Guide from the last successful plan
        is still usable for Trend labels.

        v11 FIX: validates current drone position deviation from the
        global path, NOT distance between cached guide and a path point
        (which is trivially near zero).  Also checks that the Guide
        distance from the drone is within limits.
        """
        if snapshot is None:
            return False
        online_rt = self.g.get("online_runtime", {})
        gc_cfg = online_rt.get("guide_cache", {})
        max_age = float(gc_cfg.get("max_age_s", 0.50))
        max_dist = float(gc_cfg.get("maximum_distance_m", 7.0))
        max_dev = float(gc_cfg.get("maximum_path_deviation_m", 1.5))
        max_behind = int(gc_cfg.get("maximum_indices_behind_progress", 0))

        age = current_time_s - snapshot.plan_timestamp_s
        if age > max_age:
            return False
        # Guide path index must not be behind current progress
        if snapshot.guide_path_index < progress_index - max_behind:
            return False
        # Guide world coords must be finite
        if not np.all(np.isfinite(snapshot.guide_world)):
            return False
        # v11 FIX: check drone-to-guide distance (not guide-to-path distance)
        guide_dist_from_drone = float(
            np.linalg.norm(snapshot.guide_world - cur_pos))
        if guide_dist_from_drone > max_dist:
            return False
        # v11 FIX: check current drone deviation from global path
        if global_path and len(global_path) > 0:
            # Search a local window near the progress index
            search_start = max(0, progress_index - 5)
            search_end = min(len(global_path), progress_index + 10)
            min_path_dev = float("inf")
            for i in range(search_start, search_end):
                d = float(np.linalg.norm(
                    np.array(global_path[i]) - cur_pos))
                if d < min_path_dev:
                    min_path_dev = d
            if min_path_dev > max_dev:
                return False
        return True

    def _is_trajectory_cache_valid(self, snapshot, current_time_s,
                                    cur_pos, cur_vel, planner_mode):
        """Check whether the cached local trajectory is still usable
        for Control labels.

        v11 FIX: uses trajectory_relative_time to sample the reference
        point, NOT always traj[0].  Also checks velocity error.
        """
        if snapshot is None:
            return False
        online_rt = self.g.get("online_runtime", {})
        tc_cfg = online_rt.get("trajectory_cache", {})
        max_age = float(tc_cfg.get("max_plan_age_s", 0.20))
        min_remaining = float(tc_cfg.get("minimum_remaining_time_s", 0.10))
        max_pos_err = float(tc_cfg.get("maximum_position_error_m", 0.50))
        max_vel_err = float(tc_cfg.get("maximum_velocity_error_mps", 1.00))

        age = current_time_s - snapshot.plan_timestamp_s
        if age > max_age:
            return False

        # v11 FIX: use current relative time, not traj[0]
        traj = snapshot.trajectory
        if traj is None or len(traj) == 0:
            return False
        traj_dur = snapshot.trajectory_duration_s
        rel_time = max(0.0, age)  # relative time within trajectory
        if rel_time >= traj_dur - min_remaining:
            return False

        # v11 FIX: sample at current relative time, not always index 0
        clamped_time = min(rel_time, traj_dur)
        best_idx = min(range(len(traj)),
                       key=lambda i: abs(float(traj[i].t) - clamped_time))
        ref_pos = np.array(traj[best_idx].position, dtype=np.float64)
        ref_vel = np.array(traj[best_idx].velocity, dtype=np.float64)

        pos_err = float(np.linalg.norm(cur_pos - ref_pos))
        if pos_err > max_pos_err:
            return False
        vel_err = float(np.linalg.norm(cur_vel - ref_vel))
        if vel_err > max_vel_err:
            return False

        # v11: ESDF suffix re-validation not yet implemented
        # (recorded as 'pending' in config)
        return True

    def _compute_recovery_target(self, global_path, progress_index,
                                  lookahead_distance_m):
        """Extract a recovery target from the global A* path at a fixed
        arc-length lookahead from the current progress index.

        Returns (target_world, target_path_index).
        If the remaining path is too short, returns the goal point.
        """
        if not global_path or progress_index < 0:
            goal_pt = global_path[-1] if global_path else np.zeros(3)
            return np.array(goal_pt, dtype=np.float64), len(global_path) - 1

        cum = 0.0
        for i in range(progress_index, len(global_path) - 1):
            seg_len = float(np.linalg.norm(
                np.array(global_path[i + 1]) - np.array(global_path[i])))
            if cum + seg_len >= lookahead_distance_m:
                # Interpolate within this segment
                frac = (lookahead_distance_m - cum) / max(seg_len, 1e-9)
                pt = (np.array(global_path[i]) +
                      frac * (np.array(global_path[i + 1]) -
                              np.array(global_path[i])))
                return pt, i
            cum += seg_len

        # Fall back to goal
        return np.array(global_path[-1], dtype=np.float64), len(global_path) - 1

    def _compute_recovery_direction(self, recovery_target_world, cur_pos,
                                     cur_quaternion_xyzw,
                                     yaw_deadband_rad,
                                     last_direction, tie_break):
        """Classify recovery direction as 'left' or 'right'.

        v11 FIX: computes recovery_azimuth_rad via atan2 of the FLU
        direction vector, then compares against yaw_deadband_rad
        (both are in radians).  Returns (direction_str, azimuth_rad).
        """
        delta_world = np.asarray(recovery_target_world) - np.asarray(cur_pos)
        # Convert world delta to body FLU using full quaternion
        delta_flu = world_vector_to_body_flu_quat(
            delta_world, cur_quaternion_xyzw)
        # FLU: x=forward, y=left, z=up
        azimuth_rad = math.atan2(
            float(delta_flu[1]), float(delta_flu[0]))

        if azimuth_rad > yaw_deadband_rad:
            return "left", azimuth_rad
        elif azimuth_rad < -yaw_deadband_rad:
            return "right", azimuth_rad
        # Deadband: use last direction or tie-break
        if last_direction in ("left", "right"):
            return last_direction, azimuth_rad
        return tie_break, azimuth_rad

    @staticmethod
    def _compute_speed_reference(
            guide_distance,
            guide_is_final,
            nominal_speed,
            max_velocity,
            max_acceleration,
            lookahead_distance,
            goal_tolerance,
            ramp_time_s):
        """Return one longitudinal speed reference for planning and control.

        A moving visible Guide uses its range to approach cruise speed.  The
        final Guide uses the same reference with a zero-speed endpoint and a
        bounded braking envelope.  Geometry and avoidance direction remain
        entirely owned by the collision-checked local trajectory.
        """
        distance = max(0.0, float(guide_distance))
        cruise_speed = max(
            0.0, min(float(nominal_speed), float(max_velocity)))
        if cruise_speed <= 0.0:
            return 0.0

        if guide_is_final:
            braking_distance = max(
                0.0, distance - 0.5 * float(goal_tolerance))
            # Use the acceleration that reaches cruise speed over the normal
            # response time, rather than the hard dynamics limit.  This keeps
            # the same rule physically achievable by the velocity controller.
            braking_acceleration = min(
                max(0.1, float(max_acceleration)),
                cruise_speed / max(0.1, float(ramp_time_s)))
            speed_reference = min(
                cruise_speed,
                math.sqrt(max(
                    0.0, 2.0 * braking_acceleration * braking_distance)))
        else:
            # A shortened moving Guide represents limited observable/control
            # horizon, so reduce cruise speed without treating it as a stop.
            distance_ratio = max(
                0.25, min(
                    1.0,
                    distance / max(0.5, float(lookahead_distance))))
            speed_reference = cruise_speed * distance_ratio
        return speed_reference

    @staticmethod
    def _compute_planner_boundary_acceleration(
            measured_acceleration_world,
            current_position_world,
            current_velocity_world,
            guide_position_world,
            guide_is_final,
            previous_snapshot,
            current_time_s,
            command_lookahead_time_s,
            nominal_speed,
            max_velocity,
            max_acceleration,
            lookahead_distance,
            goal_tolerance,
            max_plan_age_s,
            ramp_time_s,
            warm_start_blend):
        """Choose a time-consistent acceleration boundary for a fresh plan.

        Position and velocity always come from the current measured state.
        Acceleration is less reliable as a geometric spline boundary: using
        the nearly-zero measured value on every 30 Hz replan repeatedly
        restarts the cubic spline's acceleration phase.  Continue the
        preceding collision-checked plan's acceleration when it is fresh.
        Preserve the preceding plan's transverse avoidance acceleration while
        always closing longitudinal speed error toward the shared Guide speed
        reference.  This prevents 30 Hz replanning from indefinitely replaying
        a low-acceleration trajectory prefix.
        """
        measured = np.asarray(
            measured_acceleration_world, dtype=np.float64)
        position = np.asarray(
            current_position_world, dtype=np.float64)
        velocity = np.asarray(
            current_velocity_world, dtype=np.float64)
        guide = np.asarray(guide_position_world, dtype=np.float64)
        if measured.shape != (3,) or not np.all(np.isfinite(measured)):
            measured = np.zeros(3, dtype=np.float64)

        displacement = guide - position
        distance = float(np.linalg.norm(displacement))
        planned = None
        planned_tangent = None
        if previous_snapshot is not None:
            age = float(
                current_time_s - previous_snapshot.plan_timestamp_s)
            trajectory = previous_snapshot.trajectory
            if (0.0 <= age <= float(max_plan_age_s) and
                    trajectory is not None and len(trajectory) > 0):
                # Align the acceleration boundary with the same preview phase
                # that generated the preceding velocity command.
                sample_time = min(
                    max(
                        0.0,
                        age + max(
                            0.0, float(command_lookahead_time_s))),
                    float(previous_snapshot.trajectory_duration_s))
                point = min(
                    trajectory,
                    key=lambda tp: abs(float(tp.t) - sample_time))
                candidate = np.asarray(
                    point.acceleration, dtype=np.float64)
                if candidate.shape == (3,) and np.all(
                        np.isfinite(candidate)):
                    planned = candidate
                candidate_velocity = np.asarray(
                    point.velocity, dtype=np.float64)
                candidate_speed = float(np.linalg.norm(
                    candidate_velocity))
                if (candidate_velocity.shape == (3,) and
                        np.all(np.isfinite(candidate_velocity)) and
                        candidate_speed > 1.0e-6):
                    planned_tangent = (
                        candidate_velocity / candidate_speed)

        bootstrap = np.zeros(3, dtype=np.float64)
        direction = None
        if distance > 1.0e-9:
            direction = (
                planned_tangent
                if planned_tangent is not None
                else displacement / distance)
            target_speed = ILManager._compute_speed_reference(
                distance, guide_is_final, nominal_speed, max_velocity,
                max_acceleration, lookahead_distance, goal_tolerance,
                ramp_time_s)
            target_velocity = target_speed * direction
            bootstrap = (
                target_velocity - velocity) / max(
                    0.1, float(ramp_time_s))

        if planned is None or direction is None:
            continuation = bootstrap
        else:
            # Keep lateral/vertical avoidance curvature from the preceding
            # collision-checked plan.  Replace only its longitudinal component
            # with the current speed-error acceleration so cruise acceleration
            # cannot disappear merely because the horizon was replanned.
            planned_longitudinal = float(np.dot(planned, direction))
            planned_transverse = (
                planned - planned_longitudinal * direction)
            bootstrap_longitudinal = float(np.dot(bootstrap, direction))
            continuation = (
                planned_transverse +
                bootstrap_longitudinal * direction)
        blend = max(0.0, min(1.0, float(warm_start_blend)))
        boundary = (1.0 - blend) * measured + blend * continuation
        # Keep reserve below the hard dynamics limit for optimization and
        # for disagreement between planned and measured acceleration.
        boundary_limit = 0.85 * max(
            0.1, float(max_acceleration))
        boundary_norm = float(np.linalg.norm(boundary))
        if boundary_norm > boundary_limit:
            boundary *= boundary_limit / boundary_norm
        return boundary

    def _compute_trajectory_tracking_command(
            self, snapshot, current_time_s,
            cur_pos, cur_vel, cur_yaw, current_quaternion_xyzw,
                                              command_lookahead_time=0.08):
        """Unified trajectory tracking command computation (v12).

        Used by BOTH the expert label generation AND the execution step.
        Returns the final upper-level FLU velocity + yaw-rate command
        that should be sent to Flightmare, including position feedback,
        velocity limiting, and closed-loop yaw control.

        Returns dict with keys:
            reference_position_world, reference_velocity_world,
            reference_velocity_flu, reference_yaw_rate,
            feedback_velocity_flu,
            final_velocity_flu, final_yaw_rate,
            trajectory_sample_time_s, valid
        """
        result = {
            "reference_position_world": np.zeros(3, dtype=np.float64),
            "reference_velocity_world": np.zeros(3, dtype=np.float64),
            "reference_velocity_flu": np.zeros(3, dtype=np.float64),
            "reference_yaw_rate": 0.0,
            "feedback_velocity_flu": np.zeros(3, dtype=np.float64),
            "final_velocity_flu": np.zeros(3, dtype=np.float64),
            "final_yaw_rate": 0.0,
            "trajectory_sample_time_s": 0.0,
            "valid": False,
        }
        if snapshot is None:
            return result
        traj = snapshot.trajectory
        if traj is None or len(traj) == 0:
            return result

        lp_cfg = self.g.get("planning", {}).get("local_planner", {})
        max_velocity = float(lp_cfg.get("max_velocity", 2.5))
        nominal_speed = float(lp_cfg.get("nominal_speed", 1.8))
        max_acceleration = float(lp_cfg.get("max_acceleration", 3.5))
        goal_tolerance = float(lp_cfg.get("goal_tolerance", 0.30))
        max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))
        yaw_tracking_gain = float(lp_cfg.get("yaw_tracking_gain", 3.0))
        yaw_speed_threshold = float(lp_cfg.get("yaw_speed_threshold", 0.10))
        velocity_tracking_gain = float(
            lp_cfg.get("velocity_tracking_gain", 2.0))
        speed_ramp_time = float(
            lp_cfg.get("planner_velocity_ramp_time_s", 1.0))
        lookahead_distance = float(
            lp_cfg.get("lookahead_distance", 4.0))
        acceleration_feedforward_time = float(lp_cfg.get(
            "velocity_acceleration_feedforward_time_s", 0.30))

        rel_time = current_time_s - snapshot.plan_timestamp_s
        sample_time = rel_time + max(0.0, command_lookahead_time)
        traj_dur = snapshot.trajectory_duration_s
        current_clamped = max(0.0, min(rel_time, traj_dur))
        preview_clamped = max(0.0, min(sample_time, traj_dur))
        current_idx = min(
            range(len(traj)),
            key=lambda i: abs(float(traj[i].t) - current_clamped))
        preview_idx = min(
            range(len(traj)),
            key=lambda i: abs(float(traj[i].t) - preview_clamped))
        current_tp = traj[current_idx]
        preview_tp = traj[preview_idx]

        # Position feedback belongs to the reference at the current plan
        # time.  Comparing the future preview position with the current
        # vehicle position treats normal forward motion as tracking error and
        # adds Kp * velocity * lookahead on every fresh replan.
        ref_pos = np.array(current_tp.position, dtype=np.float64)
        feedforward_vel = np.array(preview_tp.velocity, dtype=np.float64)
        feedforward_acceleration = np.array(
            preview_tp.acceleration, dtype=np.float64)

        result["reference_position_world"] = ref_pos
        result["reference_velocity_world"] = feedforward_vel
        result["trajectory_sample_time_s"] = float(preview_tp.t)

        # Position feedback is independent of the acceleration composition
        # below.  A short 80 ms geometric preview alone realizes only a small
        # fraction of the acceleration needed by the velocity controller.
        desired_vel_world = (
            feedforward_vel +
            velocity_tracking_gain * (ref_pos - cur_pos))

        # The same longitudinal speed reference is used by the planner
        # boundary and by trajectory tracking.  Correct only along the local
        # trajectory tangent: depth/ESDF optimization continues to own the
        # fine-grained avoidance direction.
        guide_distance = float(np.linalg.norm(snapshot.guide_world - cur_pos))
        speed_reference = self._compute_speed_reference(
            guide_distance, snapshot.guide_is_final,
            nominal_speed, max_velocity, max_acceleration,
            lookahead_distance, goal_tolerance, speed_ramp_time)

        tangent = feedforward_vel.copy()
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-6:
            tangent = (
                np.asarray(preview_tp.position, dtype=np.float64) -
                np.asarray(current_tp.position, dtype=np.float64))
            tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1.0e-6:
            tangent = snapshot.guide_world - cur_pos
            tangent_norm = float(np.linalg.norm(tangent))
        tracking_acceleration = feedforward_acceleration.copy()
        if tangent_norm > 1.0e-6:
            tangent /= tangent_norm
            tangential_speed = float(np.dot(cur_vel, tangent))
            longitudinal_acceleration = max(
                -max_acceleration, min(
                    max_acceleration,
                    (speed_reference - tangential_speed) /
                    max(0.1, speed_ramp_time)))
            # The spline owns transverse acceleration because it encodes
            # collision avoidance curvature.  Its longitudinal component can
            # oppose the shared speed reference on every fresh horizon and
            # cancel cruise acceleration, so replace that component instead
            # of adding two competing longitudinal accelerations.
            planned_longitudinal = float(np.dot(
                feedforward_acceleration, tangent))
            tracking_acceleration = (
                feedforward_acceleration -
                planned_longitudinal * tangent +
                longitudinal_acceleration * tangent)
        desired_vel_world += (
            acceleration_feedforward_time * tracking_acceleration)

        # Velocity limit
        desired_speed = float(np.linalg.norm(desired_vel_world))
        if desired_speed > speed_reference:
            desired_vel_world *= (
                speed_reference / max(desired_speed, 1e-9))

        # Convert to FLU
        desired_vel_flu = world_vector_to_body_flu_quat(
            desired_vel_world, current_quaternion_xyzw)
        feedforward_vel_flu = world_vector_to_body_flu_quat(
            feedforward_vel, current_quaternion_xyzw)

        result["reference_velocity_flu"] = feedforward_vel_flu
        result["feedback_velocity_flu"] = (
            desired_vel_flu - feedforward_vel_flu)
        result["final_velocity_flu"] = quantize_bounded_vector(
            desired_vel_flu, max_velocity, decimals=6)

        # Closed-loop yaw rate
        yaw_rate_cmd = yaw_rate_for_world_velocity(
            cur_yaw, desired_vel_world, yaw_tracking_gain,
            max_yaw_rate, yaw_speed_threshold)
        result["final_yaw_rate"] = float(yaw_rate_cmd)
        result["reference_yaw_rate"] = float(yaw_rate_cmd)
        result["valid"] = True

        return result

    def _execute_rotate_in_place(self, cur_pos, cur_vel, cur_yaw,
                                  yaw_rate_command, dt_sample, dt_ctrl,
                                  max_velocity, max_acceleration, max_yaw_rate):
        """Execute a pure rotation for dt_sample with zero translational velocity.

        Returns (next_pos, next_vel, next_yaw, avg_yaw_rate,
                 executed_command_flu, executed_command_yaw_rate).
        """
        pos = np.asarray(cur_pos, dtype=np.float64).copy()
        vel = np.asarray(cur_vel, dtype=np.float64).copy()
        yaw = float(cur_yaw)
        total_yaw_change = 0.0
        elapsed = 0.0
        epsilon = 1e-9
        yr_cmd = float(yaw_rate_command)

        while elapsed < dt_sample - epsilon:
            step_dt = min(dt_ctrl, dt_sample - elapsed)
            if not self._dynamics.step_velocity_command(
                    np.zeros(3, dtype=np.float64), yr_cmd, step_dt):
                raise RuntimeError("Flightmare rotate step failed")
            ds = self._dynamics.get_state()
            pos, vel = ds.position_world.copy(), ds.velocity_world.copy()
            qx, qy, qz, qw = ds.quaternion_world_body
            yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                             1.0 - 2.0 * (qy * qy + qz * qz))
            yr = float(ds.angular_velocity_body[2])
            total_yaw_change += yr * step_dt
            elapsed += step_dt

        avg_yaw_rate = total_yaw_change / max(dt_sample, 1e-9)
        return (pos, vel, yaw, avg_yaw_rate,
                np.zeros(3, dtype=np.float64), yr_cmd)

    def _sample_expert_command(self, result, label_lookahead_time_s,
                               current_yaw):
        """Sample expert velocity command from a plan trajectory at fixed lookahead.

        The expert command is sampled from a plan generated from x_t.

        Args:
            result: LocalPlanResult from C++ planner.
            label_lookahead_time_s: fixed future time for supervision.

        Returns:
            (expert_velocity_world, expert_yaw_rate) as numpy arrays/scalars.
        """
        if result is None or not result.success or len(result.trajectory) == 0:
            return np.zeros(3, dtype=np.float64), 0.0

        traj = result.trajectory
        # Find trajectory point closest to label_lookahead_time_s
        best_idx = min(range(len(traj)),
                       key=lambda i: abs(float(traj[i].t) - label_lookahead_time_s))
        command_point = traj[best_idx]
        expert_vel_world = np.array(command_point.velocity, dtype=np.float64)
        lp_cfg = self.g.get("planning", {}).get("local_planner", {})
        max_velocity = float(lp_cfg.get("max_velocity", 2.5))
        max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))
        yaw_tracking_gain = float(lp_cfg.get("yaw_tracking_gain", 3.0))
        yaw_speed_threshold = float(lp_cfg.get("yaw_speed_threshold", 0.10))

        # Keep labels inside the same hard limit used by execution and schema
        # validation. The spline can exceed it slightly around a measured
        # initial velocity even when its nominal samples were clamped.
        expert_speed = float(np.linalg.norm(expert_vel_world))
        if expert_speed > max_velocity:
            expert_vel_world *= max_velocity / max(expert_speed, 1e-9)

        # A new receding-horizon plan is generated every frame. Its first
        # open-loop yaw derivative repeatedly saturated in real captures and
        # made the vehicle spin. Close the loop against measured yaw instead.
        expert_yaw_rate = yaw_rate_for_world_velocity(
            current_yaw, expert_vel_world, yaw_tracking_gain,
            max_yaw_rate, yaw_speed_threshold)
        return expert_vel_world, expert_yaw_rate

    def _execute_trajectory_segment(self, result, cur_pos, cur_vel, cur_yaw,
                                     dt_sample, dt_ctrl, sample_index,
                                     max_velocity, max_acceleration, max_yaw_rate,
                                     command_lookahead_time=0.08,
                                     velocity_tracking_gain=2.0,
                                     yaw_tracking_gain=3.0,
                                     yaw_speed_threshold=0.10):
        """Execute dt_sample using sub-steps from the plan trajectory.

        Returns:
            next state, average measured yaw rate, and the final upper-level
            velocity/yaw-rate command actually sent during this segment.
        """
        pos = np.asarray(cur_pos, dtype=np.float64).copy()
        vel = np.asarray(cur_vel, dtype=np.float64).copy()
        yaw = float(cur_yaw)
        total_yaw_change = 0.0

        if result is None or len(result.trajectory) == 0:
            # Hover
            elapsed = 0.0
            epsilon = 1e-9
            while elapsed < dt_sample - epsilon:
                step_dt = min(dt_ctrl, dt_sample - elapsed)
                if not self._dynamics.step_velocity_command(
                        np.zeros(3, dtype=np.float64), 0.0, step_dt):
                    raise RuntimeError("Flightmare hover step failed")
                ds = self._dynamics.get_state()
                pos, vel = ds.position_world.copy(), ds.velocity_world.copy()
                qx, qy, qz, qw = ds.quaternion_world_body
                new_yaw = math.atan2(2.0 * (qw*qz + qx*qy),
                                     1.0 - 2.0 * (qy*qy + qz*qz))
                yr = ds.angular_velocity_body[2]
                yaw = new_yaw
                total_yaw_change += yr * step_dt
                elapsed += step_dt
            avg_yaw_rate = total_yaw_change / max(dt_sample, 1e-9)
            return (pos, vel, yaw, avg_yaw_rate,
                    np.zeros(3, dtype=np.float64), 0.0)

        traj = result.trajectory
        traj_duration = float(traj[-1].t)
        elapsed = 0.0
        epsilon = 1e-9
        last_command_flu = np.zeros(3, dtype=np.float64)
        last_command_yaw_rate = 0.0

        while elapsed < dt_sample - epsilon:
            step_dt = min(dt_ctrl, dt_sample - elapsed)
            # Find trajectory point closest to current elapsed time
            # Every result is a newly planned trajectory whose time origin is
            # the current state x_t.  Never carry the episode sample index into
            # this local trajectory's clock.
            # t=0 deliberately carries the current velocity for continuity.
            # Sampling only at elapsed=0 on every fresh replan therefore
            # perpetuates reset drift and never executes the forward plan.
            # Preview velocity, but track the position at the current local
            # trajectory time.  Future position is not a current error.
            exec_time = elapsed + max(0.0, command_lookahead_time)
            clamped_time = min(exec_time, traj_duration)
            preview_idx = min(
                range(len(traj)),
                key=lambda i: abs(
                    float(traj[i].t) - clamped_time))
            current_idx = min(
                range(len(traj)),
                key=lambda i: abs(float(traj[i].t) - elapsed))
            reference_pos = np.array(
                traj[current_idx].position, dtype=np.float64)
            feedforward_vel = np.array(
                traj[preview_idx].velocity, dtype=np.float64)
            desired_vel_world = (feedforward_vel +
                                 velocity_tracking_gain *
                                 (reference_pos - pos))
            desired_speed = float(np.linalg.norm(desired_vel_world))
            if desired_speed > max_velocity:
                desired_vel_world *= max_velocity / max(desired_speed, 1e-9)
            yaw_rate_cmd = yaw_rate_for_world_velocity(
                yaw, desired_vel_world, yaw_tracking_gain,
                max_yaw_rate, yaw_speed_threshold)
            command_state = self._dynamics.get_state()
            desired_vel_flu = world_vector_to_body_flu_quat(
                desired_vel_world,
                command_state.quaternion_world_body)
            last_command_flu = desired_vel_flu.copy()
            last_command_yaw_rate = float(yaw_rate_cmd)
            if not self._dynamics.step_velocity_command(
                    desired_vel_flu, yaw_rate_cmd, step_dt):
                raise RuntimeError("Flightmare trajectory step failed")
            ds = self._dynamics.get_state()
            pos, vel = ds.position_world.copy(), ds.velocity_world.copy()
            qx, qy, qz, qw = ds.quaternion_world_body
            yaw = math.atan2(2.0 * (qw*qz + qx*qy),
                             1.0 - 2.0 * (qy*qy + qz*qz))
            yr = float(ds.angular_velocity_body[2])
            total_yaw_change += yr * step_dt
            elapsed += step_dt

        avg_yaw_rate = total_yaw_change / max(dt_sample, 1e-9)
        return (pos, vel, yaw, avg_yaw_rate,
                last_command_flu, last_command_yaw_rate)

    def _execute_hover(self, cur_pos, cur_vel, cur_yaw,
                        dt_sample, dt_ctrl,
                        max_velocity, max_acceleration, max_yaw_rate):
        """Execute a zero-velocity hover for dt_sample.

        Returns:
            next state, average measured yaw rate, and the final zero hover
            command actually sent during this segment.
        """
        pos = np.asarray(cur_pos, dtype=np.float64).copy()
        vel = np.asarray(cur_vel, dtype=np.float64).copy()
        yaw = float(cur_yaw)
        total_yaw_change = 0.0
        elapsed = 0.0
        epsilon = 1e-9

        while elapsed < dt_sample - epsilon:
            step_dt = min(dt_ctrl, dt_sample - elapsed)
            if not self._dynamics.step_velocity_command(
                    np.zeros(3, dtype=np.float64), 0.0, step_dt):
                raise RuntimeError("Flightmare hover step failed")
            ds = self._dynamics.get_state()
            pos, vel = ds.position_world.copy(), ds.velocity_world.copy()
            qx, qy, qz, qw = ds.quaternion_world_body
            yaw = math.atan2(2.0 * (qw*qz + qx*qy),
                             1.0 - 2.0 * (qy*qy + qz*qz))
            yr = float(ds.angular_velocity_body[2])
            total_yaw_change += yr * step_dt
            elapsed += step_dt

        avg_yaw_rate = total_yaw_change / max(dt_sample, 1e-9)
        return (pos, vel, yaw, avg_yaw_rate,
                np.zeros(3, dtype=np.float64), 0.0)

    def _execute_velocity_command(self, velocity_flu, yaw_rate, duration_s):
        """Execute one selected learner/safety command through the backend."""
        if not self._dynamics.step_velocity_command(
                np.asarray(velocity_flu, dtype=np.float64),
                float(yaw_rate), float(duration_s)):
            raise RuntimeError("Selected velocity command failed in dynamics backend")
        state = self._dynamics.get_state()
        qx, qy, qz, qw = state.quaternion_world_body
        yaw = math.atan2(2.0 * (qw*qz + qx*qy),
                         1.0 - 2.0 * (qy*qy + qz*qz))
        return (state.position_world.copy(), state.velocity_world.copy(), yaw,
                float(state.angular_velocity_body[2]),
                np.asarray(velocity_flu, dtype=np.float64).copy(),
                float(yaw_rate))

    def _build_training_row_v16(
            self, cur_pos, cur_vel, cur_yaw,
            decision: RuntimeDecision,
            plan_success, planner_compute_ms,
            planner_status_str,
            goal_np, goal_pt, global_path_length,
            max_guide_range,
            png_name, collision,
            plan, sample_index, dt_sample,
            frame_id, recording_start_mono, recording_start_epoch_ns,
            t_request_mono, recv_time_mono, latency_ms,
            h_fov_rad, v_fov_rad,
            trend_normal_h_bins, trend_v_bins,
            trend_sigma_bins, normal_h_bin_edges, v_bin_edges,
            azi_soft_names, ele_soft_names,
            previous_progress_s,
            guide_sel=None, ref_segment=None,
            observed_map=None, observed_esdf=None,
            planner_used_obs=0, planner_used_fallback=0,
            current_quaternion_xyzw=None,
            # v11 extras
            plan_is_fresh=0, plan_cache_valid=0,
            plan_source_frame_id=-1, plan_age_s=0.0,
            planner_due=False, planner_attempted=0,
            planner_retry_count=0, terminal_scale_used=1.0,
            recovery_elapsed_s=0.0,
            guide_progress_index=-1,
            global_path=None,
    ):
        """Serialize one schema-v16 row from an immutable RuntimeDecision."""
        snapshot = decision.plan_snapshot
        row = {}

        # ── Time & matching ──────────────────────────────────────
        trajectory_time_s = sample_index * dt_sample
        row["timestamp_ns"] = recording_start_epoch_ns + int(
            (t_request_mono - recording_start_mono) * 1e9)
        row["receive_timestamp_ns"] = recording_start_epoch_ns + int(
            (recv_time_mono - recording_start_mono) * 1e9)
        task_id = plan.get(
            "task_id", "task_{:03d}".format(self.traj_idx))
        row["episode_id"] = "{}:{}".format(self.scene_label, task_id)
        row["frame_id"] = frame_id
        row["episode_frame_index"] = int(sample_index)
        row["sequence_reset"] = int(sample_index == 0)
        row["control_dt_s"] = round(float(dt_sample), 9)
        row["trajectory_time_s"] = round(trajectory_time_s, 6)
        row["latency_ms"] = round(latency_ms, 3)
        row["match_method"] = "frame_id_exact"
        row["guide_mode_changed"] = 0
        row["recovery_entered"] = 0
        row["recovery_exited"] = 0

        # ── v11: runtime mode enums ──────────────────────────────
        row["planner_mode"] = decision.planner_mode
        row["control_mode"] = decision.control_mode
        row["trend_mode"] = decision.trend_mode
        is_goal_hold = (
            decision.trend_mode == TrendMode.GOAL_HOLD.value)

        # ── Current state (x_t) ──────────────────────────────────
        row["x"] = round(float(cur_pos[0]), 4)
        row["y"] = round(float(cur_pos[1]), 4)
        row["z"] = round(float(cur_pos[2]), 4)
        if current_quaternion_xyzw is None:
            half = 0.5 * float(cur_yaw)
            current_quaternion_xyzw = np.array(
                [0.0, 0.0, math.sin(half), math.cos(half)])
        qx, qy, qz, qw = np.asarray(current_quaternion_xyzw)
        row["qx"] = round(float(qx), 6)
        row["qy"] = round(float(qy), 6)
        row["qz"] = round(float(qz), 6)
        row["qw"] = round(float(qw), 6)

        row["state_vx_world"] = round(float(cur_vel[0]), 4)
        row["state_vy_world"] = round(float(cur_vel[1]), 4)
        row["state_vz_world"] = round(float(cur_vel[2]), 4)

        state_vel_flu = world_vector_to_body_flu_quat(
            cur_vel, current_quaternion_xyzw)
        row["state_vx_flu"] = round(float(state_vel_flu[0]), 4)
        row["state_vy_flu"] = round(float(state_vel_flu[1]), 4)
        row["state_vz_flu"] = round(float(state_vel_flu[2]), 4)

        # ── Expert supervision (v13: from RuntimeDecision, NO recomputation) ──
        expert_vel_flu = decision.expert_velocity_flu.copy()
        expert_yaw_rate = decision.expert_yaw_rate

        row["expert_label_valid"] = (
            1 if (decision.plan_snapshot is not None and
                  decision.plan_snapshot.trajectory is not None and
                  len(decision.plan_snapshot.trajectory) > 0
                  and decision.control_mode != "EMERGENCY_STOP")
            else (1 if decision.control_mode in (
                ControlMode.ROTATE_IN_PLACE.value,
                ControlMode.HOLD_POSITION.value) else 0))

        # Convert FLU expert back to world for the world-velocity columns
        expert_vel_world = body_flu_to_world_quat(
            expert_vel_flu, current_quaternion_xyzw)
        row["expert_vx_world"] = round(float(expert_vel_world[0]), 6)
        row["expert_vy_world"] = round(float(expert_vel_world[1]), 6)
        row["expert_vz_world"] = round(float(expert_vel_world[2]), 6)

        row["expert_vx_flu"] = round(float(expert_vel_flu[0]), 6)
        row["expert_vy_flu"] = round(float(expert_vel_flu[1]), 6)
        row["expert_vz_flu"] = round(float(expert_vel_flu[2]), 6)
        row["expert_yaw_rate"] = round(float(expert_yaw_rate), 6)
        row["selected_command_vx_flu"] = round(
            float(decision.selected_velocity_flu[0]), 6)
        row["selected_command_vy_flu"] = round(
            float(decision.selected_velocity_flu[1]), 6)
        row["selected_command_vz_flu"] = round(
            float(decision.selected_velocity_flu[2]), 6)
        row["selected_command_yaw_rate"] = round(
            float(decision.selected_yaw_rate), 6)
        row["final_executed_actor"] = decision.selected_actor
        row["trajectory_reference_vx_flu"] = round(
            float(decision.trajectory_reference_velocity_flu[0]), 6)
        row["trajectory_reference_vy_flu"] = round(
            float(decision.trajectory_reference_velocity_flu[1]), 6)
        row["trajectory_reference_vz_flu"] = round(
            float(decision.trajectory_reference_velocity_flu[2]), 6)
        row["trajectory_feedback_vx_flu"] = round(
            float(decision.trajectory_feedback_velocity_flu[0]), 6)
        row["trajectory_feedback_vy_flu"] = round(
            float(decision.trajectory_feedback_velocity_flu[1]), 6)
        row["trajectory_feedback_vz_flu"] = round(
            float(decision.trajectory_feedback_velocity_flu[2]), 6)
        row["trajectory_sample_time_s"] = round(
            float(decision.trajectory_sample_time_s), 6)

        # ── Executed next state placeholders ─────────────────────
        row["executed_next_x"] = round(float(cur_pos[0]), 4)
        row["executed_next_y"] = round(float(cur_pos[1]), 4)
        row["executed_next_z"] = round(float(cur_pos[2]), 4)
        row["executed_next_vx_world"] = round(float(cur_vel[0]), 4)
        row["executed_next_vy_world"] = round(float(cur_vel[1]), 4)
        row["executed_next_vz_world"] = round(float(cur_vel[2]), 4)
        row["executed_next_vx_flu"] = round(float(state_vel_flu[0]), 4)
        row["executed_next_vy_flu"] = round(float(state_vel_flu[1]), 4)
        row["executed_next_vz_flu"] = round(float(state_vel_flu[2]), 4)
        row["actual_next_vx_flu"] = round(float(state_vel_flu[0]), 4)
        row["actual_next_vy_flu"] = round(float(state_vel_flu[1]), 4)
        row["actual_next_vz_flu"] = round(float(state_vel_flu[2]), 4)
        row["executed_next_yaw"] = round(float(cur_yaw), 6)
        row["executed_next_yaw_rate"] = 0.0

        # ── Global navigation labels ─────────────────────────────
        global_delta_world = goal_np - cur_pos
        global_distance_m = float(np.linalg.norm(global_delta_world))
        if global_distance_m < 1e-9:
            # The direction is mathematically undefined exactly at the goal,
            # but TERMINAL_HOLD is still a fully valid training sample.  Use
            # the canonical forward direction with zero distance.
            row["global_direction_valid"] = int(is_goal_hold)
            row["global_dir_x_flu"] = 1.0 if is_goal_hold else 0.0
            row["global_dir_y_flu"] = 0.0
            row["global_dir_z_flu"] = 0.0
        else:
            global_delta_flu = world_vector_to_body_flu_quat(
                global_delta_world, current_quaternion_xyzw)
            norm = float(np.linalg.norm(global_delta_flu))
            row["global_direction_valid"] = 1
            row["global_dir_x_flu"] = round(float(global_delta_flu[0]) / norm, 6)
            row["global_dir_y_flu"] = round(float(global_delta_flu[1]) / norm, 6)
            row["global_dir_z_flu"] = round(float(global_delta_flu[2]) / norm, 6)
        row["global_distance_m"] = round(global_distance_m, 4)
        row["global_distance_norm"] = round(
            min(global_distance_m, max_guide_range) / max(max_guide_range, 1e-9), 6)

        # ── v11: Trend labels (11 normal FOV bins + 2 recovery = 13 classes) ──
        # v11 FIX: guide source cascade
        # 1. Recovery mode → recovery_global_astar_target
        # 2. Valid cached Guide from last successful plan → latest_successful_plan_guide
        # 3. No valid cache → MUST plan or recovery; NEVER use unvalidated current GuideSelector
        guide_source = decision.guide_source
        guide_target_world = decision.guide_target_world
        guide_path_idx_target = decision.guide_target_path_index
        if decision.trend_mode in (
                TrendMode.RECOVERY.value, TrendMode.GOAL_HOLD.value):
            guide_cache_valid_int = 0
            guide_cache_age = 0.0
            guide_plan_id_val = -1
        elif (snapshot is not None and
              self._is_guide_cache_valid(
                  snapshot, trajectory_time_s,
                  guide_progress_index, global_path or [], cur_pos)):
            guide_cache_valid_int = 1
            guide_cache_age = trajectory_time_s - snapshot.plan_timestamp_s
            guide_plan_id_val = snapshot.plan_id
            self._max_guide_cache_age_s = max(
                self._max_guide_cache_age_s, guide_cache_age)
        else:
            # v11 FIX: no valid cached guide → invalid trend (must plan or recover)
            guide_cache_valid_int = 0
            guide_cache_age = 0.0
            guide_plan_id_val = -1

        row["guide_source"] = guide_source
        row["guide_plan_id"] = guide_plan_id_val
        row["guide_cache_age_s"] = round(guide_cache_age, 6)
        row["guide_cache_valid"] = guide_cache_valid_int
        row["guide_target_x_world"] = round(float(guide_target_world[0]), 4)
        row["guide_target_y_world"] = round(float(guide_target_world[1]), 4)
        row["guide_target_z_world"] = round(float(guide_target_world[2]), 4)
        row["guide_target_path_index"] = guide_path_idx_target

        # Compute FLU vector from current state to guide target
        guide_delta_world = np.asarray(guide_target_world) - cur_pos
        guide_distance_m = float(np.linalg.norm(guide_delta_world))
        row["guide_x_world"] = round(float(guide_target_world[0]), 4)
        row["guide_y_world"] = round(float(guide_target_world[1]), 4)
        row["guide_z_world"] = round(float(guide_target_world[2]), 4)

        # Always set 13-class count
        row["trend_horizontal_class_count"] = TREND_HORIZONTAL_CLASS_COUNT
        is_recovery = (
            decision.trend_mode == TrendMode.RECOVERY.value)
        if is_recovery:
            row["guide_mode"] = (
                "RECOVER_LEFT"
                if decision.recovery_direction == "left"
                else "RECOVER_RIGHT")
        else:
            row["guide_mode"] = "NORMAL"
        row["recovery_target_distance_m"] = round(
            float(np.linalg.norm(
                decision.recovery_target_world - cur_pos)), 6)
        row["recovery_target_distance_norm_debug"] = round(
            min(row["recovery_target_distance_m"], max_guide_range) /
            max(max_guide_range, 1e-9), 6)

        if is_goal_hold:
            (hold_h_class, hold_v_class,
             hold_h_soft, hold_v_soft) = goal_hold_guide_labels(
                 TREND_NORMAL_HORIZONTAL_BIN_COUNT, trend_v_bins)
            hold_h_bin = TREND_NORMAL_HORIZONTAL_BIN_COUNT // 2
            row["trend_label_valid"] = 1
            row["guide_dir_x_flu_exact"] = 1.0
            row["guide_dir_y_flu_exact"] = 0.0
            row["guide_dir_z_flu_exact"] = 0.0
            row["guide_distance_m"] = 0.0
            row["guide_distance_norm"] = 0.0
            row["guide_azimuth_rad"] = 0.0
            row["guide_elevation_rad"] = 0.0
            row["guide_azimuth_bin"] = hold_h_bin
            row["guide_elevation_bin"] = hold_v_class
            row["trend_horizontal_class_13"] = hold_h_class
            row["trend_normal_horizontal_class_11"] = hold_h_bin
            for i, name in enumerate(azi_soft_names):
                row[name] = round(float(hold_h_soft[i]), 6)
            for i, name in enumerate(ele_soft_names):
                row[name] = round(float(hold_v_soft[i]), 6)
        elif guide_distance_m < 1e-9 and not is_recovery:
            row["trend_label_valid"] = 0
            row["guide_dir_x_flu_exact"] = 0.0
            row["guide_dir_y_flu_exact"] = 0.0
            row["guide_dir_z_flu_exact"] = 0.0
            row["guide_distance_m"] = 0.0
            row["guide_distance_norm"] = 0.0
            row["guide_azimuth_rad"] = 0.0
            row["guide_elevation_rad"] = 0.0
            row["guide_azimuth_bin"] = -1
            row["guide_elevation_bin"] = -1
            row["trend_horizontal_class_13"] = -1
            row["trend_normal_horizontal_class_11"] = -1
            for name in azi_soft_names:
                row[name] = 0.0
            for name in ele_soft_names:
                row[name] = 0.0
        else:
            if is_recovery and guide_distance_m < 1e-9:
                gdx = math.cos(decision.recovery_azimuth_rad)
                gdy = math.sin(decision.recovery_azimuth_rad)
                gdz = 0.0
            else:
                guide_delta_flu = world_vector_to_body_flu_quat(
                    guide_delta_world, current_quaternion_xyzw)
                norm_v = float(np.linalg.norm(guide_delta_flu))
                gdx = float(guide_delta_flu[0]) / norm_v
                gdy = float(guide_delta_flu[1]) / norm_v
                gdz = float(guide_delta_flu[2]) / norm_v
            row["guide_dir_x_flu_exact"] = round(gdx, 6)
            row["guide_dir_y_flu_exact"] = round(gdy, 6)
            row["guide_dir_z_flu_exact"] = round(gdz, 6)
            row["guide_distance_m"] = round(guide_distance_m, 4)
            row["guide_distance_norm"] = (
                0.0 if is_recovery else round(
                    min(guide_distance_m, max_guide_range) /
                    max(max_guide_range, 1e-9), 6))

            azimuth = math.atan2(gdy, gdx)
            elevation = math.atan2(gdz, math.sqrt(gdx * gdx + gdy * gdy + 1e-12))
            row["guide_azimuth_rad"] = round(azimuth, 6)
            row["guide_elevation_rad"] = round(elevation, 6)

            # ── 13-class Trend horizontal label ──────────────────
            row["trend_label_valid"] = 1

            if is_recovery:
                # Recovery: horizontal class = 0 (LEFT) or 12 (RIGHT)
                # v11 FIX: use pre-computed recovery_direction and
                # recovery_azimuth_rad from the control tick.
                if decision.recovery_direction == "left":
                    h_class_13 = TREND_RECOVER_LEFT_CLASS
                else:
                    h_class_13 = TREND_RECOVER_RIGHT_CLASS
                row["trend_horizontal_class_13"] = h_class_13
                row["trend_normal_horizontal_class_11"] = -1
                row["guide_azimuth_bin"] = -1
                recovery_vertical_class = trend_v_bins // 2
                row["guide_elevation_bin"] = recovery_vertical_class

                # Recovery: one-hot, indices 0 or 12 only
                soft_13 = np.zeros(TREND_HORIZONTAL_CLASS_COUNT, dtype=np.float64)
                soft_13[h_class_13] = 1.0
                for i in range(TREND_HORIZONTAL_CLASS_COUNT):
                    if i < len(azi_soft_names):
                        row[azi_soft_names[i]] = round(float(soft_13[i]), 6)

                for i, name in enumerate(ele_soft_names):
                    row[name] = float(i == recovery_vertical_class)
            else:
                # Normal TRACK_GUIDE mode: 11 FOV bins → 13 classes
                # v11 FIX: FOV edges computed for 11 bins only
                normal_bin_centers = 0.5 * (
                    normal_h_bin_edges[:-1] + normal_h_bin_edges[1:])
                h_bin_11 = int(np.argmin(np.abs(normal_bin_centers - azimuth)))
                # Clamp to [0, 10]
                h_bin_11 = max(0, min(TREND_NORMAL_HORIZONTAL_BIN_COUNT - 1, h_bin_11))
                v_bin = int(np.argmin(np.abs(v_bin_edges - elevation)))

                row["guide_azimuth_bin"] = h_bin_11
                row["guide_elevation_bin"] = v_bin
                row["trend_normal_horizontal_class_11"] = h_bin_11

                # Map: old 0–10 → new 1–11
                h_class_13 = h_bin_11 + TREND_NORMAL_CLASS_OFFSET
                row["trend_horizontal_class_13"] = h_class_13

                # v11 FIX: 13-dim soft label
                # Index 0 and 12 = 0.0; indices 1-11 = Gaussian over 11 normal bins
                soft_13 = np.zeros(TREND_HORIZONTAL_CLASS_COUNT, dtype=np.float64)
                for i in range(TREND_NORMAL_HORIZONTAL_BIN_COUNT):
                    bin_center = normal_bin_centers[i]
                    bin_err = (azimuth - bin_center) / (
                        (h_fov_rad / (TREND_NORMAL_HORIZONTAL_BIN_COUNT - 1)) + 1e-12)
                    soft_13[i + TREND_NORMAL_CLASS_OFFSET] = math.exp(
                        -0.5 * (bin_err / trend_sigma_bins) ** 2)
                soft_sum = float(np.sum(soft_13))
                if soft_sum > 1e-12:
                    soft_13 /= soft_sum
                # Assertions
                assert len(soft_13) == TREND_HORIZONTAL_CLASS_COUNT
                assert np.isfinite(soft_13).all()
                assert soft_13[TREND_RECOVER_LEFT_CLASS] == 0.0
                assert soft_13[TREND_RECOVER_RIGHT_CLASS] == 0.0
                for i in range(TREND_HORIZONTAL_CLASS_COUNT):
                    if i < len(azi_soft_names):
                        row[azi_soft_names[i]] = round(float(soft_13[i]), 6)

                # Vertical soft labels
                v_soft = np.zeros(trend_v_bins, dtype=np.float64)
                for i in range(trend_v_bins):
                    bin_err = (elevation - v_bin_edges[i]) / (
                        (v_fov_rad / (trend_v_bins - 1)) + 1e-12)
                    v_soft[i] = math.exp(
                        -0.5 * (bin_err / trend_sigma_bins) ** 2)
                v_soft /= max(v_soft.sum(), 1e-12)
                for i, name in enumerate(ele_soft_names):
                    row[name] = round(float(v_soft[i]), 6)

        # ── v11: planner scheduling & retry fields ────────────────
        row["plan_source_frame_id"] = plan_source_frame_id
        row["plan_age_s"] = round(plan_age_s, 6)
        row["plan_is_fresh"] = plan_is_fresh
        row["plan_cache_valid"] = plan_cache_valid
        row["planner_due"] = 1 if planner_due else 0
        row["planner_attempted"] = planner_attempted
        row["planner_retry_count"] = planner_retry_count
        row["terminal_scale_used"] = round(terminal_scale_used, 4)

        # ── v11: recovery fields ─────────────────────────────────
        row["recovery_target_x_world"] = round(
            float(decision.recovery_target_world[0]), 4)
        row["recovery_target_y_world"] = round(
            float(decision.recovery_target_world[1]), 4)
        row["recovery_target_z_world"] = round(
            float(decision.recovery_target_world[2]), 4)
        row["recovery_target_path_index"] = (
            decision.recovery_target_path_index)
        row["recovery_elapsed_s"] = round(recovery_elapsed_s, 6)
        row["recovery_direction"] = decision.recovery_direction
        row["recovery_azimuth_rad"] = round(
            float(decision.recovery_azimuth_rad), 6)

        # ── Depth & collision ────────────────────────────────────
        row["depth_file"] = png_name
        row["collision"] = collision

        # ── Start / goal ─────────────────────────────────────────
        row["start_x"] = round(plan["start"][0], 4)
        row["start_y"] = round(plan["start"][1], 4)
        row["start_z"] = round(plan["start"][2], 4)
        row["goal_x"] = round(goal_pt[0], 4)
        row["goal_y"] = round(goal_pt[1], 4)
        row["goal_z"] = round(goal_pt[2], 4)

        # ── Planner & debug fields ───────────────────────────────
        if snapshot is not None:
            row["global_progress_s"] = round(
                float(snapshot.reference_path_start_index), 4)
            row["global_progress_index"] = snapshot.guide_path_index
            row["local_goal_index"] = snapshot.terminal_path_index
            row["plan_id"] = snapshot.plan_id
            row["plan_time_from_start_s"] = 0.0
            row["planner_min_clearance"] = round(
                snapshot.minimum_clearance_m, 4)
        else:
            row["global_progress_s"] = round(previous_progress_s, 4)
            row["global_progress_index"] = -1
            row["local_goal_index"] = -1
            row["plan_id"] = -1
            row["plan_time_from_start_s"] = 0.0
            row["planner_min_clearance"] = 0.0

        row["global_progress_ratio"] = round(
            row["global_progress_s"] / max(global_path_length, 1e-6), 6)
        row["planner_status"] = planner_status_str
        row["planner_success"] = plan_success
        row["planner_compute_ms"] = round(planner_compute_ms, 3)
        row["distance_to_final_goal"] = round(
            float(np.linalg.norm(cur_pos - goal_np)), 4)
        # ── Phase 2: observed map diagnostics ────────────────────
        if observed_map is not None:
            row["observed_map_revision"] = observed_map.get_revision()
            row["observed_esdf_revision"] = (
                self._observed_esdf_cache.stats_summary()["current_revision"]
                if self._observed_esdf_cache is not None else
                observed_map.get_revision())
            row["observed_known_voxel_count"] = observed_map.known_voxel_count()
            row["observed_occupied_voxel_count"] = observed_map.occupied_voxel_count()
            row["observed_free_voxel_count"] = observed_map.free_voxel_count()
        else:
            row["observed_map_revision"] = -1
            row["observed_esdf_revision"] = -1
            row["observed_known_voxel_count"] = 0
            row["observed_occupied_voxel_count"] = 0
            row["observed_free_voxel_count"] = 0

        # ── Phase 2: guide selection diagnostics ─────────────────
        if guide_sel is not None:
            row["guide_candidate_count"] = guide_sel.candidate_count
            row["guide_visible"] = 1 if guide_sel.visible else 0
            row["guide_depth_visible"] = 1 if guide_sel.depth_visible else 0
            row["guide_corridor_check_enabled"] = (
                1 if guide_sel.corridor_check_enabled else 0)
            row["guide_corridor_known_free_ratio"] = round(
                guide_sel.corridor_known_free_ratio, 4)
            row["guide_path_index"] = guide_sel.guide_path_index
            row["guide_rejection_reason"] = (
                guide_sel.rejection_reason[:80]
                if guide_sel.rejection_reason else "")
            row["terminal_path_index"] = guide_sel.terminal_path_index
            row["terminal_x_world"] = round(
                float(guide_sel.terminal_position_world[0]), 4)
            row["terminal_y_world"] = round(
                float(guide_sel.terminal_position_world[1]), 4)
            row["terminal_z_world"] = round(
                float(guide_sel.terminal_position_world[2]), 4)
            row["terminal_distance_m"] = round(
                guide_sel.terminal_distance_m, 4)
            row["terminal_path_arc_length_m"] = round(
                guide_sel.terminal_path_arc_length_m, 4)
        else:
            row["guide_candidate_count"] = 0
            row["guide_visible"] = 0
            row["guide_depth_visible"] = 0
            row["guide_corridor_check_enabled"] = 0
            row["guide_corridor_known_free_ratio"] = -1.0
            row["guide_path_index"] = -1
            row["guide_rejection_reason"] = "no_selector"
            row["terminal_path_index"] = -1
            row["terminal_x_world"] = 0.0
            row["terminal_y_world"] = 0.0
            row["terminal_z_world"] = 0.0
            row["terminal_distance_m"] = 0.0
            row["terminal_path_arc_length_m"] = 0.0

        # ── Phase 2: planner flags ───────────────────────────────
        row["planner_used_observed_esdf"] = planner_used_obs
        row["planner_map_source"] = (
            "observed_esdf" if planner_used_obs else "global_esdf")
        row["planner_unknown_is_free"] = 0
        row["planner_used_global_fallback"] = planner_used_fallback
        row["reference_segment_point_count"] = (
            len(ref_segment) if ref_segment else 0)
        row["scene_id"] = self.scene_label
        row["task_id"] = plan.get("task_id",
                                   "task_{:03d}".format(self.traj_idx))

        return row

    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════

    def _st_finish_recording(self):
        """v7: Called after online planning loop exits to finalize recording."""
        plan = self.current_planned[self.traj_idx]
        ctrl_hz = self.g["control"]["control_hz"]
        rec_hz = self.g["control"]["record_hz"]
        data_cfg = self.g["data"]
        schema_version = int(data_cfg["schema_version"])
        scene_vehicle_r, scene_safety_m, _, _ = (
            self._scene_geometry_contract())

        # Compute planner stats
        avg_plan_ms = (sum(self._planning_times_ms) / max(len(self._planning_times_ms), 1)
                       if self._planning_times_ms else 0.0)
        sorted_times = sorted(self._planning_times_ms) if self._planning_times_ms else [0]
        p95_idx = max(0, int(len(sorted_times) * 0.95) - 1)
        p95_plan_ms = sorted_times[p95_idx] if sorted_times else 0.0
        max_plan_ms = max(self._planning_times_ms) if self._planning_times_ms else 0.0
        min_exec_cl = min(self._executed_clearances) if self._executed_clearances else 0.0

        # The global path endpoint equals the requested goal by construction;
        # success must be measured from the final executed command instead.
        goal_np = np.array(plan["goal"], dtype=np.float64)
        final_pos = self._final_executed_position
        dist_to_goal = (float(np.linalg.norm(np.array(final_pos) - goal_np))
                        if final_pos is not None else float("inf"))
        reached_goal = bool(self._trajectory_reached_goal)

        # ── v7 metadata ──────────────────────────────────────────────
        collection_mode = data_cfg["collection_mode"]
        meta = {
            "scene": self.scene_label,
            "trajectory": "traj_{:03d}".format(self.traj_idx + 1),
            "start": plan["start"],
            "goal": plan["goal"],
            "sent_control_frames": self._rec_sent_control_frames,
            "raw_received_frames": self._rec_raw_received_frames,
            "written_rows": self._rec_written_rows,
            "discarded_extra_frames": self._rec_discarded_extra_frames,
            "missing_record_ticks": self._rec_missing_record_ticks,
            "unmatched_frames": self._rec_unmatched_frames,
            "exact_matches": self._rec_exact_matches,
            "fallback_matches": self._rec_fallback_matches,
            "control_hz": ctrl_hz,
            "record_hz": rec_hz,
            "max_inflight_frames": int(
                self.g.get("sync", {}).get("max_inflight_frames", 4)),
            "depth_w": self._depth_cfg["width"],
            "depth_h": self._depth_cfg["height"],
            "valid": plan.get("valid", False),
            "validation_report": plan.get("validation_report", {}),
            "esdf_stats": self.current_esdf_stats if hasattr(self, "current_esdf_stats") else {},
            # ── Schema v16 metadata ──
            "schema_version": schema_version,
            "terminal_label_semantics": "goal_hold_v1",
            "collection_mode": collection_mode,
            "sample_semantics": (
                "depth_t,state_t -> guide_t,expert_command_t; selected command "
                "is executed through Flightmare dynamics"),
            "sequence_semantics": {
                "episode_id": "stable within one recorded trajectory",
                "episode_frame_index": "zero-based contiguous row index",
                "sequence_reset": "1 only on the first row of an episode",
                "control_dt_s": "actual fixed record-control interval",
            },
            "image_time_angular_velocity_semantics": (
                "state_angular_velocity_*_body belongs to depth_t/state_t"),
            "previous_executed_command_semantics": (
                "last upper-level velocity_flu/yaw_rate command actually sent "
                "to dynamics before depth_t; unavailable only at episode start"),
            "episode_validity_policy": (
                "reject_trajectory_if_any_required_label_is_invalid_v1"),
            "label_lookahead_time_s": float(
                data_cfg.get("label_lookahead_time_s", 0.08)),
            "training_coordinate_frame": {
                "name": "CAMERA_FLU",
                "x": "forward",
                "y": "left",
                "z": "up",
            },
            "world_coordinate_frame": "ROS_WORLD_FLU",
            "guide_source": (
                "current_depth_goal_explorer_or_goal_tolerance_hold"),
            "guide_visibility_source": (
                "current_depth_camera_fov_and_relative_mission_goal"),
            "guide_selection_rule": (
                "direct_goal_ray_if_clear_else_minimum_obstacle_avoiding_"
                "azimuth_with_right_tie_break"),
            "guide_known_free_corridor_check_enabled": True,
            "guide_certified_range_m": float(
                self._guide_selector.explorer_usable_range_m),
            "guide_swept_radius_m": float(
                self._guide_selector.explorer_required_radius_m),
            "guide_esdf_validation_clearance_m": float(
                self._guide_selector.explorer_esdf_validation_clearance_m),
            "trajectory_terminal_rule": (
                "NORMAL: selected visible Guide is the hard optimization "
                "endpoint with automatically allocated feasible duration; "
                "RECOVERY: no local trajectory, rotate in place; "
                "GOAL_HOLD: owns braking after first goal-tolerance entry, "
                "uses center Guide with value=0 and zero four-dimensional "
                "Control, and releases only if stopped outside tolerance"),
            "unknown_space_policy": "not_applicable_to_complete_global_esdf",
            "global_map_fallback_enabled": False,
            "trend_config": {
                "normal_horizontal_bins": TREND_NORMAL_HORIZONTAL_BIN_COUNT,
                "horizontal_class_count": TREND_HORIZONTAL_CLASS_COUNT,
                "recover_left_class": TREND_RECOVER_LEFT_CLASS,
                "normal_class_offset": TREND_NORMAL_CLASS_OFFSET,
                "recover_right_class": TREND_RECOVER_RIGHT_CLASS,
                "guide_mode_mapping": {
                    "NORMAL": "horizontal=1..11,vertical=0..6",
                    "RECOVER_LEFT": "horizontal=0,vertical=3,value=0",
                    "RECOVER_RIGHT": "horizontal=12,vertical=3,value=0",
                },
                "terminal_hold_mapping": (
                    "runtime trend_mode=GOAL_HOLD remains network guide_mode="
                    "NORMAL with horizontal=6,vertical=3,value=0"),
            },
            "committed_label_validity": {
                "required_row_fields": [
                    "frame_valid",
                    "expert_label_valid",
                    "trend_label_valid",
                    "global_direction_valid",
                ],
                "required_value": 1,
                "failure_action": "reject_complete_trajectory",
                "partial_supervision_supported": False,
            },
            "trend_vertical_bins": int(
                data_cfg["trend"]["vertical_bins"]),
            "trend_soft_sigma_bins": float(
                data_cfg["trend"]["soft_sigma_bins"]),
            "max_guide_range_m": float(
                self._guide_selector.explorer_usable_range_m),
            "maximum_global_guidance_distance_m": 0.0,
            "depth_all_valid_in_simulation": True,
            # ── Phase 2: observed map metadata ──
            "local_planner_map_source": "global_esdf",
            "local_expert_map": "global_esdf",
            "observed_map_enabled": False,
            "observed_esdf_enabled": False,
            "local_trajectory_collision_check": "global_esdf",
            "trend_supervision_source": "guide_waypoint",
            "control_supervision_source": (
                "future_velocity_and_yaw_rate_sampled_from_local_bspline; "
                "zero_velocity_and_yaw_rate_in_goal_hold"),
            "global_map_usage": [
                "task_feasibility",
                "offline_audit",
                "local_bspline_optimization",
                "local_trajectory_collision_validation",
            ],
            "global_map_used_for_local_collision_optimization": True,
            "global_esdf_cache_key": self._current_esdf_cache_key,
            "global_esdf_cache_hit": self._current_esdf_cache_hit,
            "global_esdf_cache_stats": (
                self._global_esdf_cache.stats_summary()
                if self._global_esdf_cache is not None else {}),
            "observed_esdf_cache_stats": (
                self._observed_esdf_cache.stats_summary()
                if self._observed_esdf_cache is not None else {}),
            "dynamics_backend": (
                self._dynamics.backend_name if self._dynamics is not None else "unavailable"),
            "simulation_hz": float(self.g.get("dynamics", {}).get("simulation_hz", 0.0)),
            "dynamics_control_hz": float(self.g.get("dynamics", {}).get("control_hz", 0.0)),
            "render_hz": float(self.g.get("dynamics", {}).get("render_hz", 0.0)),
            "scene_manifest_path": self._current_scene_manifest_path,
            "task_manifest_path": (
                self._current_task_manifest_paths[self.traj_idx]
                if self.traj_idx < len(self._current_task_manifest_paths) else ""),
            "dagger": ({
                "round_id": self._dagger_ctrl.round_id,
                "beta": self._dagger_ctrl.current_beta,
                "model_path": self._policy_provider.model_path,
                "model_hash": self._policy_provider.model_hash(),
                "stats": self._dagger_ctrl.stats_summary(),
            } if self._dagger_ctrl is not None else {"enabled": False}),
            # ── v8 label quality stats ──
            "invalid_expert_label_count": getattr(
                self, "_invalid_expert_label_count", 0),
            "invalid_trend_label_count": getattr(
                self, "_invalid_trend_label_count", 0),
            "invalid_frame_count": getattr(
                self, "_episode_invalid_frame_count", 0),
            "invalid_frame_reason_counts": getattr(
                self, "_episode_invalid_reason_counts", {}),
            "consecutive_guide_failures": getattr(
                self, "_consecutive_guide_failures", 0),
            # ── Phase 3: scene & task metadata ──
            "scene_generation_enabled": self._use_scene_gen,
            "scene_generation_source": self.g.get("scene_generation", {}).get(
                "source", "density_driven"),
            "scene_profile_mode": self._use_profile_mode,
            "scene_profile_name": self._current_profile_name,
            "scene_density_tier": (
                getattr(self._current_profile, "density_tier", "")
                if self._current_profile is not None else ""),
            "scene_profile_index": self._scene_profile_index if self._use_profile_mode else -1,
            "scene_index_in_profile": self._scene_index_in_profile if self._use_profile_mode else -1,
            "scene_id": self.scene_label,
            "scene_target_density_mode": self._current_target_density_mode,
            "scene_target_density": (self._current_target_density
                                     if self._current_target_density is not None else -1.0),
            "scene_obstacle_count": len(getattr(self, "_current_scene_obstacles", [])),
            "scene_obstacle_radius_min_actual_m": (
                min(o.radius_m for o in self._current_scene_obstacles)
                if self._current_scene_obstacles else 0.0),
            "scene_obstacle_radius_max_actual_m": (
                max(o.radius_m for o in self._current_scene_obstacles)
                if self._current_scene_obstacles else 0.0),
            "scene_obstacle_radius_mean_actual_m": (
                sum(o.radius_m for o in self._current_scene_obstacles) /
                max(len(self._current_scene_obstacles), 1)
                if self._current_scene_obstacles else 0.0),
            "task_index_in_scene": (self._task_index_in_scene
                                    if self._use_profile_mode else -1),
            "scene_topology_valid": (
                self._current_scene_validation.valid
                if self._current_scene_validation is not None else None),
            "scene_rejection_reason": (
                self._current_scene_validation.rejection_reason
                if self._current_scene_validation is not None else ""),
            "scene_minimum_surface_gap_m": (
                self._current_scene_validation.minimum_surface_gap_m
                if self._current_scene_validation is not None else 0.0),
            "scene_u_shape_detected": (
                self._current_scene_validation.u_shape_detected
                if self._current_scene_validation is not None else False),
            "scene_dead_end_detected": (
                self._current_scene_validation.dead_end_detected
                if self._current_scene_validation is not None else False),
            # ── Phase 3: density metrics (v9) ──
            "scene_actual_raw_occupancy_ratio": (
                compute_raw_occupancy(
                    self._current_scene_obstacles,
                    compute_region_area(self._scene_generator.obstacle_region)
                    if self._scene_generator is not None else 1.0)
                if self._current_scene_obstacles else 0.0),
            "scene_actual_inflated_occupancy_ratio": (
                compute_inflated_occupancy(
                    self._current_scene_obstacles,
                    compute_region_area(self._scene_generator.obstacle_region)
                    if self._scene_generator is not None else 1.0,
                    scene_vehicle_r,
                    scene_safety_m)
                if self._current_scene_obstacles else 0.0),
            "scene_actual_obstacles_per_100m2": (
                compute_obstacles_per_100m2(
                    self._current_scene_obstacles,
                    compute_region_area(self._scene_generator.obstacle_region)
                    if self._scene_generator is not None else 1.0)
                if self._current_scene_obstacles else 0.0),
            "task_direct_path_blocked": (
                self._current_task_validation.direct_path_blocked
                if self._current_task_validation is not None else None),
            "task_direct_blocker_count": (
                self._current_task_validation.direct_blocker_count
                if self._current_task_validation is not None else 0),
            "task_nearest_direct_blocker_distance_m": (
                self._current_task_validation.nearest_direct_blocker_distance_m
                if self._current_task_validation is not None else 0.0),
            "task_coverage_target_task_type": (
                self._current_task_validation.coverage_target_task_type
                if self._current_task_validation is not None else ""),
            "task_coverage_actual_task_type": (
                self._current_task_validation.coverage_actual_task_type
                if self._current_task_validation is not None else ""),
            "task_coverage_target_blocker_distance_band": (
                self._current_task_validation.
                coverage_target_blocker_distance_band
                if self._current_task_validation is not None else ""),
            "task_coverage_actual_blocker_distance_band": (
                self._current_task_validation.
                coverage_actual_blocker_distance_band
                if self._current_task_validation is not None else ""),
            "task_coverage_target_height_band": (
                self._current_task_validation.coverage_target_height_band
                if self._current_task_validation is not None else ""),
            "task_coverage_actual_height_band": (
                self._current_task_validation.coverage_actual_height_band
                if self._current_task_validation is not None else ""),
            "task_coverage_region_pair_index": (
                self._current_task_validation.coverage_region_pair_index
                if self._current_task_validation is not None else -1),
            "task_detour_ratio": (
                self._current_task_validation.detour_ratio
                if self._current_task_validation is not None else 0.0),
            "task_dominant_obstacle_id": (
                self._current_task_validation.dominant_obstacle_id
                if self._current_task_validation is not None else ""),
            "task_lower_cost_side": (
                self._current_task_validation.lower_cost_side
                if self._current_task_validation is not None else ""),
            "task_side_cost_difference_ratio": (
                self._current_task_validation.side_cost_difference_ratio
                if self._current_task_validation is not None else 0.0),
            "task_global_side_choice_valid": (
                self._current_task_validation.global_side_choice_valid
                if self._current_task_validation is not None else None),
            "observability_invalid_frame_count": getattr(
                self, "_invalid_obs_frame_count", 0),
            "observability_trigger_count": self._observability_trigger_count,
            "observability_consistent_count": self._observability_consistent_count,
            # ── v5 planner metadata (retained) ──
            "planner_type": "receding_horizon_local",
            "planner_backend": self._planner_backend,
            "dynamics_config": self.g.get("dynamics", {}),
            "yaw_rate_execution_semantics": (
                "upper-level Control label/command is a yaw-rate target; "
                "Flightmare applies the configured yaw-rate acceleration "
                "limit before body-rate control"),
            "local_planner_config": self.g.get("planning", {}).get("local_planner", {}),
            "global_path_length": plan.get("global_path_length", 0.0),
            "raw_global_path_points": len(plan.get("raw_path", [])),
            "shortcut_global_path_points": len(plan.get("global_path", [])),
            "total_replans": self._total_replans,
            "successful_replans": self._successful_replans,
            "failed_replans": self._failed_replans,
            "emergency_hold_count": self._emergency_hold_count,
            "average_planning_ms": round(avg_plan_ms, 2),
            "p95_planning_ms": round(p95_plan_ms, 2),
            "max_planning_ms": round(max_plan_ms, 2),
            "minimum_executed_clearance": round(min_exec_cl, 4),
            "reached_goal": reached_goal,
            "final_goal_error": round(dist_to_goal, 4),
            "final_executed_position": final_pos,
            "final_executed_velocity": self._final_executed_velocity,
            "exit_reason": self._trajectory_exit_reason,
            # ── v11: online runtime stats ──
            "online_runtime_config": self.g.get("online_runtime", {}),
            "fresh_plan_control_frame_count": getattr(
                self, "_fresh_plan_frame_count", 0),
            "cached_plan_control_frame_count": getattr(
                self, "_cached_plan_frame_count", 0),
            "recovery_control_frame_count": getattr(
                self, "_recovery_frame_count", 0),
            "planner_attempt_count": getattr(
                self, "_planner_attempt_count", 0),
            "planner_success_count": getattr(
                self, "_planner_success_count", 0),
            "planner_failure_count": getattr(
                self, "_planner_failure_count", 0),
            "planner_retry_success_count": getattr(
                self, "_planner_retry_success_count", 0),
            "deadline_retime_success_count": getattr(
                self, "_deadline_retime_success_count", 0),
            "last_planner_rejection_status": getattr(
                self, "_last_planner_rejection_status", ""),
            "last_planner_rejection_message": getattr(
                self, "_last_planner_rejection_message", ""),
            "recovery_entry_count": getattr(
                self, "_recovery_entry_count", 0),
            "recovery_success_count": getattr(
                self, "_recovery_success_count", 0),
            "recovery_timeout_count": getattr(
                self, "_recovery_timeout_count", 0),
            "maximum_plan_age_s": getattr(
                self, "_max_plan_age_s", 0.0),
            "maximum_guide_cache_age_s": getattr(
                self, "_max_guide_cache_age_s", 0.0),
            "recover_left_frame_count": getattr(
                self, "_recover_left_frame_count", 0),
            "recover_right_frame_count": getattr(
                self, "_recover_right_frame_count", 0),
            "normal_guide_frame_count": getattr(
                self, "_normal_guide_frame_count", 0),
            "goal_hold_frame_count": getattr(
                self, "_goal_hold_frame_count", 0),
            "recovery_exit_count": getattr(
                self, "_recovery_exit_count", 0),
            "guide_value_zero_count": getattr(
                self, "_guide_value_zero_count", 0),
            "guide_value_saturated_count": getattr(
                self, "_guide_value_saturated_count", 0),
            "global_direction_invalid_count": getattr(
                self, "_global_direction_invalid_count", 0),
            "sequence_reset_count": getattr(
                self, "_sequence_reset_count", 0),
            "trend_horizontal_class_count": TREND_HORIZONTAL_CLASS_COUNT,
            "ESDF_clearance_semantics": (
                "ESDF values already have drone_radius subtracted by ESDFBuilder. "
                "Local planner min_clearance is additional safety margin on top."
            ),
            "status": "inprogress",
        }
        for class_index, count in enumerate(getattr(
                self, "_horizontal_class_counts", [])):
            meta["horizontal_class_count_{:02d}".format(
                class_index)] = int(count)
        for class_index, count in enumerate(getattr(
                self, "_vertical_class_counts", [])):
            meta["vertical_class_count_{:02d}".format(
                class_index)] = int(count)
        if self._dagger_ctrl is not None:
            meta["dagger"]["round_manifest_path"] = os.path.join(
                self._dagger_ctrl.get_output_dir(), "round_manifest.json")
        meta_path = os.path.join(self._inprogress_dir, "metadata.json")
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2, sort_keys=True)
            mf.flush()
            os.fsync(mf.fileno())

        self.total_trajectories += 1
        self.total_frames_sent += self._rec_sent_control_frames
        self.total_frames_received += self._rec_raw_received_frames

        # Debug: plot executed vs global path
        if self.debug and _MPL_AVAILABLE:
            executed_xy = []
            data_csv = os.path.join(self._inprogress_dir, "data.csv")
            if os.path.isfile(data_csv):
                try:
                    with open(data_csv, "r") as f:
                        header = f.readline().strip().split(",")
                        x_idx = header.index("x") if "x" in header else 1
                        y_idx = header.index("y") if "y" in header else 2
                        for line in f:
                            parts = line.strip().split(",")
                            if len(parts) > max(x_idx, y_idx):
                                try:
                                    executed_xy.append(
                                        (float(parts[x_idx]), float(parts[y_idx])))
                                except ValueError:
                                    pass
                except Exception as exc:
                    rospy.logwarn("[DEBUG] Could not read executed positions: %s", exc)

            self._debug_plot_executed(
                "traj_{:03d}".format(self.traj_idx + 1),
                plan["start"], plan["goal"],
                plan.get("raw_path"), plan.get("global_path"),
                executed_xy)

        rospy.loginfo("[ONLINE] Done: sent_ctrl=%d raw_rcvd=%d written=%d "
                      "replans=%d/%d/%d avg_plan=%.1fms → %s",
                      self._rec_sent_control_frames,
                      self._rec_raw_received_frames,
                      self._rec_written_rows,
                      self._successful_replans, self._failed_replans,
                      self._emergency_hold_count,
                      avg_plan_ms,
                      self._inprogress_dir)
        self._enter_state(State.STOP_RECORDING)

    def _st_stop_recording(self):
        self.traj_idx += 1
        self._enter_state(State.VALIDATE_AND_COMMIT)

    # ═══════════════════════════════════════════════════════════════
    #  VALIDATE_AND_COMMIT – real data integrity checks
    # ═══════════════════════════════════════════════════════════════

    def _st_validate_and_commit(self):
        """Validate the just-completed trajectory and atomically commit or reject."""
        if self._inprogress_dir is None or not os.path.isdir(self._inprogress_dir):
            self._route_next()
            return

        plan = self.current_planned[self.traj_idx - 1]
        validation_passed = True
        failure_reasons = []

        # ── Build column-name → index map from CSV header ────────
        data_path = os.path.join(self._inprogress_dir, "data.csv")
        csv_rows = 0
        col_map = {}
        if os.path.isfile(data_path):
            try:
                with open(data_path, "r") as f:
                    header_line = f.readline().strip()
                    headers = header_line.split(",")
                    col_map = {name.strip(): idx for idx, name in enumerate(headers)}
                    for _ in f:
                        csv_rows += 1
            except Exception as exc:
                validation_passed = False
                failure_reasons.append("csv_unparseable: {}".format(exc))
        else:
            validation_passed = False
            failure_reasons.append("csv_missing")

        validation_cfg = self.g.get("planning", {}).get("validation", {})
        minimum_rows = int(validation_cfg.get("minimum_rows", 10))
        if csv_rows < minimum_rows:
            validation_passed = False
            failure_reasons.append(
                "insufficient_rows: {} < {}".format(csv_rows, minimum_rows))

        # ── Resolve column indices by name ───────────────────────
        depth_file_col = col_map.get("depth_file", 15)
        collision_col = col_map.get("collision", 16)

        # ── Check PNG count vs CSV rows ──────────────────────────
        depth_dir = os.path.join(self._inprogress_dir, "depth")
        png_files = []
        none_depth_count = 0
        if os.path.isdir(depth_dir):
            png_files = [f for f in os.listdir(depth_dir) if f.endswith(".png")]

        if os.path.isfile(data_path) and col_map:
            with open(data_path, "r") as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) > depth_file_col and parts[depth_file_col] == "none":
                        none_depth_count += 1

        if none_depth_count > max(1, csv_rows * 0.2):
            validation_passed = False
            failure_reasons.append("too_many_none_depth: {}/{}".format(
                none_depth_count, csv_rows))

        # PNG count should match CSV rows with valid depth filenames
        valid_depth_rows = csv_rows - none_depth_count
        if len(png_files) != valid_depth_rows:
            failure_reasons.append(
                "png_count_mismatch: pngs={} valid_depth_rows={}".format(
                    len(png_files), valid_depth_rows))
            if abs(len(png_files) - valid_depth_rows) > max(1, valid_depth_rows * 0.1):
                validation_passed = False

        # ── No duplicate PNG filenames ───────────────────────────
        if len(png_files) != len(set(png_files)):
            validation_passed = False
            failure_reasons.append("duplicate_png_filenames")

        # ── Sync error threshold check ───────────────────────────
        sync_path = os.path.join(self._inprogress_dir, "sync.csv")
        max_latency_ms = 0.0
        max_match_error_ms = 0.0
        latency_sample_count = 0
        latency_violation_count = 0
        configured_latency_limit_ms = float(
            self.g.get("sync", {}).get("max_acceptable_latency_ms", 250.0))
        if os.path.isfile(sync_path):
            try:
                with open(sync_path, "r") as f:
                    f.readline()
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) >= 4:
                            try:
                                lat = float(parts[2])
                                merr = float(parts[3])
                                if lat >= 0.0:
                                    latency_sample_count += 1
                                    if lat > configured_latency_limit_ms:
                                        latency_violation_count += 1
                                if lat > max_latency_ms:
                                    max_latency_ms = lat
                                if merr > max_match_error_ms:
                                    max_match_error_ms = merr
                            except ValueError:
                                pass
            except Exception:
                pass

        sync_cfg = self.g.get("sync", {})
        max_acceptable_match_error_ms = float(
            sync_cfg.get("max_acceptable_sync_error_ms", 100.0))
        latency_violation_pct = (100.0 * latency_violation_count /
                                 max(1, latency_sample_count))
        max_latency_violation_pct = float(
            sync_cfg.get("max_latency_violation_pct", 1.0))
        catastrophic_latency_ms = float(
            sync_cfg.get("catastrophic_latency_ms", 5000.0))
        if (latency_violation_pct > max_latency_violation_pct or
            max_latency_ms > catastrophic_latency_ms):
            validation_passed = False
            failure_reasons.append(
                "latency_exceeded: max={:.1f}ms violations={}/{} ({:.2f}%)".format(
                    max_latency_ms, latency_violation_count,
                    latency_sample_count, latency_violation_pct))
        if max_match_error_ms > max_acceptable_match_error_ms:
            validation_passed = False
            failure_reasons.append("match_error_exceeded: {:.1f}ms > {:.0f}ms".format(
                max_match_error_ms, max_acceptable_match_error_ms))
        unmatched_pct = (100.0 * self._rec_unmatched_frames /
                         max(1, self._rec_raw_received_frames))
        max_unmatched_pct = float(sync_cfg.get("max_unmatched_pct", 1.0))
        if unmatched_pct > max_unmatched_pct:
            validation_passed = False
            failure_reasons.append(
                "too_many_unmatched_frames: {}/{} ({:.2f}%) > {:.2f}%".format(
                    self._rec_unmatched_frames, self._rec_raw_received_frames,
                    unmatched_pct, max_unmatched_pct))

        # ── Collision check (strict) ─────────────────────────────
        collision_count = 0
        if os.path.isfile(data_path) and col_map:
            with open(data_path, "r") as f:
                f.readline()
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) > collision_col and parts[collision_col] == "1":
                        collision_count += 1

        strict_collision_policy = True
        if strict_collision_policy and collision_count > 0:
            validation_passed = False
            failure_reasons.append("collisions_detected: {}".format(collision_count))

        # 6. Plan validation
        if not plan.get("valid", False):
            validation_passed = False
            rpt = plan.get("validation_report", {})
            reasons = rpt.get("invalid_reasons", ["plan_invalid"])
            failure_reasons.extend(reasons)

        if not self._trajectory_reached_goal:
            validation_passed = False
            failure_reasons.append(
                "goal_not_reached: reason={} error_m={:.3f}".format(
                    self._trajectory_exit_reason,
                    float(np.linalg.norm(np.array(self._final_executed_position) -
                                         np.array(plan["goal"])))
                    if self._final_executed_position is not None else float("inf")))

        invalid_frame_count = int(getattr(
            self, "_episode_invalid_frame_count", 0))
        if invalid_frame_count > 0:
            validation_passed = False
            reason_counts = getattr(
                self, "_episode_invalid_reason_counts", {})
            failure_reasons.append(
                "episode_contains_invalid_frames: count={} reasons={}".format(
                    invalid_frame_count,
                    json.dumps(reason_counts, sort_keys=True)))

        # Production acceptance invariants.  These checks use recorded fields,
        # not metadata claims.
        if os.path.isfile(data_path) and col_map:
            with open(data_path, "r") as f:
                for row_index, row in enumerate(csv.DictReader(f), 1):
                    if str(row.get("frame_valid", "0")) != "1":
                        validation_passed = False
                        failure_reasons.append(
                            "invalid_frame_at_row:{}:{}".format(
                                row_index,
                                row.get("frame_invalid_reason", "unspecified")))
                        break
                    if str(row.get("expert_label_valid", "0")) != "1":
                        validation_passed = False
                        failure_reasons.append(
                            "invalid_expert_label_at_row:{}".format(row_index))
                        break
                    formal_checks = (
                        # v11: trend_label_valid must be 1 in ALL modes (including RECOVERY)
                        (str(row.get("trend_label_valid", "0")) == "1",
                         "invalid_trend_label"),
                        (str(row.get("global_direction_valid", "0")) == "1",
                         "invalid_global_direction"),
                        (row.get("guide_source", "") in (
                         "farthest_visible_astar_waypoint",
                         "latest_successful_plan_guide",
                         "recovery_global_astar_target",
                         "no_valid_cached_guide",
                         "current_guide_selector",
                         "current_depth_visible_guide",
                         "current_depth_goal_explorer",
                          "recovery_goal_direction",
                          "recovery_explore_direction",
                          "goal_tolerance_hold",
                          "invalid_no_visible_guide",
                          ), "invalid_guide_source"),
                        # v11: guide_visible/depth_visible are NOT hard requirements.
                        # The drone can fly with a cached plan while the guide is
                        # temporarily occluded. What matters is valid Trend+Control output.
                        # v11: corridor check is optional
                        (str(row.get("guide_corridor_check_enabled", "1")) in ("0", "1"),
                         "unexpected_guide_corridor_check"),
                        # v11: planner_success may be 0 in RECOVERY mode
                        (str(row.get("planner_success", "")).lower() in ("1", "true")
                         or str(row.get("planner_mode", "")) in (
                             "RECOVERY", "GOAL_HOLD"),
                          "planner_unsuccessful"),
                        (row.get("planner_map_source", "") == "global_esdf",
                         "planner_map_not_global_esdf"),
                        (str(row.get("planner_used_observed_esdf", "1")) == "0",
                         "planner_used_observed_esdf"),
                        (str(row.get("planner_used_global_fallback", "1")) == "0",
                         "global_fallback"),
                    )
                    failed_check = next(
                        (name for passed, name in formal_checks if not passed),
                        None)
                    if failed_check is not None:
                        validation_passed = False
                        failure_reasons.append(
                            "{}_at_row:{}".format(failed_check, row_index))
                        break

        if self._dynamics is None or self._dynamics.backend_name != "flightmare":
            validation_passed = False
            failure_reasons.append("production_dynamics_backend_not_flightmare")
        if self.g.get("data", {}).get("collection_mode") != "deterministic_lockstep":
            validation_passed = False
            failure_reasons.append("production_collection_mode_not_lockstep")
        if not self._current_esdf_cache_key:
            validation_passed = False
            failure_reasons.append("global_esdf_cache_metadata_invalid")
        if self._current_scene_validation is not None and not self._current_scene_validation.valid:
            validation_passed = False
            failure_reasons.append("scene_topology_invalid")
        if self._current_task_validation is not None:
            if not self._current_task_validation.valid:
                validation_passed = False
                failure_reasons.append("task_invalid")
            side_cost_enabled = bool(self.g.get(
                "scene_generation", {}).get(
                    "side_cost", {}).get("enabled", False))
            if (side_cost_enabled and
                    not self._current_task_validation.global_side_choice_valid):
                validation_passed = False
                failure_reasons.append("side_cost_invalid")
        obs_cfg = self.g.get("scene_generation", {}).get("observability_audit", {})
        if self._use_observed_map and obs_cfg.get("enabled", False):
            if self._invalid_obs_frame_count >= int(obs_cfg.get(
                    "maximum_invalid_frames_before_reject", 5)):
                validation_passed = False
                failure_reasons.append("runtime_side_choice_unobservable")
            if (self._observability_trigger_count == 0 or
                    self._observability_consistent_count == 0):
                validation_passed = False
                failure_reasons.append("runtime_observability_audit_not_established")
        if self._dagger_ctrl is not None:
            if (self._dagger_ctrl.rollout_mode != "expert" and
                    not self._policy_provider.model_hash()):
                validation_passed = False
                failure_reasons.append("dagger_model_hash_missing")

        # 7. Metadata completeness
        meta_path = os.path.join(self._inprogress_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            validation_passed = False
            failure_reasons.append("metadata_missing")

        # 8. Schema v16 command and row-integrity checks
        data_cfg = self.g["data"]
        schema_version = int(data_cfg["schema_version"])
        if schema_version != 16:
            validation_passed = False
            failure_reasons.append(
                "unsupported_current_schema_version:{}".format(
                    schema_version))
        elif os.path.isfile(data_path) and col_map:
            v16_issues = self._validate_schema_v16(
                data_path, col_map, data_cfg)
            if v16_issues:
                rospy.logwarn("[Validate] Schema v16 issues: %s",
                              "; ".join(v16_issues[:10]))
                failure_reasons.extend(v16_issues)
                validation_passed = False

        # ── Commit or reject ──────────────────────────────────────
        if validation_passed:
            self._commit_trajectory()
        else:
            self._reject_trajectory(failure_reasons)

        self._inprogress_dir = None
        self._final_dir = None
        self._route_next()

    def _validate_schema_v16(self, data_path, col_map, data_cfg):
        """Validate strict schema-v16 data invariants. Returns issue strings."""
        issues = []
        trend_cfg_v = data_cfg["trend"]
        trend_normal_h_bins = int(
            trend_cfg_v["normal_horizontal_bins"])
        trend_v_bins = int(trend_cfg_v["vertical_bins"])
        lp_cfg = self.g.get("planning", {}).get("local_planner", {})
        max_velocity = float(lp_cfg.get("max_velocity", 2.5))
        max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))
        command_compare_atol = float(
            self.g.get("planning", {}).get("validation", {}).get(
                "command_compare_atol", 1.0e-5))
        required_fields = (
            "timestamp_ns", "receive_timestamp_ns",
            "episode_id", "frame_id", "episode_frame_index",
            "sequence_reset", "control_dt_s",
            "frame_valid", "frame_invalid_reason",
            "state_angular_velocity_x_body",
            "state_angular_velocity_y_body",
            "state_angular_velocity_z_body",
            "state_angular_velocity_x_flu",
            "state_angular_velocity_y_flu",
            "state_angular_velocity_z_flu",
            "gravity_direction_x_flu",
            "gravity_direction_y_flu",
            "gravity_direction_z_flu",
            "state_vx_flu", "state_vy_flu", "state_vz_flu",
            "previous_executed_command_valid",
            "previous_executed_command_frame_id", "previous_executed_actor",
            "previous_executed_command_vx_flu",
            "previous_executed_command_vy_flu",
            "previous_executed_command_vz_flu",
            "previous_executed_command_yaw_rate",
            "expert_label_valid",
            "expert_vx_flu", "expert_vy_flu", "expert_vz_flu",
            "expert_yaw_rate",
            "trend_label_valid", "match_method",
            "guide_mode",
            "guide_mode_changed", "recovery_entered", "recovery_exited",
            "learner_output_valid", "selected_command_vx_flu",
            "selected_command_vy_flu", "selected_command_vz_flu",
            "selected_command_yaw_rate", "final_executed_actor",
            "applied_command_vx_flu", "applied_command_vy_flu",
            "applied_command_vz_flu", "applied_command_yaw_rate",
            "applied_command_actor", "applied_command_valid",
            "trajectory_reference_vx_flu",
            "trajectory_reference_vy_flu",
            "trajectory_reference_vz_flu",
            "trajectory_feedback_vx_flu",
            "trajectory_feedback_vy_flu",
            "trajectory_feedback_vz_flu",
            "trajectory_sample_time_s",
            "recovery_azimuth_rad",
            "recovery_target_distance_m",
            "recovery_target_distance_norm_debug",
            "global_direction_valid",
            "global_dir_x_flu", "global_dir_y_flu",
            "global_dir_z_flu", "global_distance_m",
            "global_distance_norm",
            "actual_next_vx_flu", "actual_next_vy_flu",
            "actual_next_vz_flu", "scene_id", "task_id",
            "guide_source", "guide_visible", "guide_depth_visible",
            "guide_candidate_count", "guide_path_index",
            "terminal_path_index", "terminal_x_world", "terminal_y_world",
            "terminal_z_world", "guide_corridor_check_enabled",
            "guide_corridor_known_free_ratio", "planner_success",
            "planner_map_source", "planner_used_observed_esdf",
            "planner_used_global_fallback", "reference_segment_point_count")

        try:
            with open(data_path, "r") as f:
                reader = csv.DictReader(f)
                prev_frame_id = -1
                previous_applied_command = None
                previous_applied_frame_id = -1
                previous_episode_frame_index = -1
                previous_timestamp_ns = -1
                episode_id = None
                previous_schema_guide_mode = None
                row_idx = 0
                for row in reader:
                    row_idx += 1

                    for required in required_fields:
                        if required not in row or row.get(required, "") == "":
                            issues.append("CRITICAL: missing_or_empty {} at row {}".format(
                                required, row_idx))

                    for validity_field in (
                            "frame_valid", "expert_label_valid",
                            "trend_label_valid", "global_direction_valid"):
                        if str(row.get(validity_field, "0")) != "1":
                            issues.append(
                                "CRITICAL: {} must be 1 at row {}: {}"
                                .format(
                                    validity_field, row_idx,
                                    row.get(
                                        "frame_invalid_reason",
                                        "unspecified")))

                    try:
                        image_angular_velocity = np.array([
                            float(row.get("state_angular_velocity_x_body", "nan")),
                            float(row.get("state_angular_velocity_y_body", "nan")),
                            float(row.get("state_angular_velocity_z_body", "nan"))])
                        if not np.all(np.isfinite(image_angular_velocity)):
                            issues.append(
                                "CRITICAL: invalid image-time angular velocity at row {}".format(
                                    row_idx))
                        control_input_state = np.array([
                            float(row["gravity_direction_x_flu"]),
                            float(row["gravity_direction_y_flu"]),
                            float(row["gravity_direction_z_flu"]),
                            float(row["state_vx_flu"]),
                            float(row["state_vy_flu"]),
                            float(row["state_vz_flu"]),
                            float(row["state_angular_velocity_z_flu"])])
                        if not np.all(np.isfinite(control_input_state)):
                            issues.append(
                                "CRITICAL: invalid_control_input_state at row "
                                "{}".format(row_idx))
                        previous_command_valid = str(row.get(
                            "previous_executed_command_valid", "0")) == "1"
                        if row_idx > 1 and not previous_command_valid:
                            issues.append(
                                "CRITICAL: missing previous executed command at row {}".format(
                                    row_idx))
                        if previous_command_valid:
                            previous_command = np.array([
                                float(row.get("previous_executed_command_vx_flu", "nan")),
                                float(row.get("previous_executed_command_vy_flu", "nan")),
                                float(row.get("previous_executed_command_vz_flu", "nan")),
                                float(row.get("previous_executed_command_yaw_rate", "nan"))])
                            if not np.all(np.isfinite(previous_command)):
                                issues.append(
                                    "CRITICAL: invalid previous executed command at row {}".format(
                                        row_idx))
                    except (ValueError, TypeError):
                        issues.append(
                            "CRITICAL: unparseable image-time state/history at row {}".format(
                                row_idx))

                    # match_method must be exact frame_id
                    mm = row.get("match_method", "")
                    if mm and mm != "frame_id_exact":
                        issues.append("CRITICAL: non-exact match_method='{}' at row {}".format(mm, row_idx))

                    # v12: frame_id must be strictly consecutive (fid == prev_fid + 1)
                    try:
                        fid = int(row.get("frame_id", -1))
                        if fid <= prev_frame_id:
                            issues.append("CRITICAL: non-monotonic frame_id {} -> {} at row {}".format(
                                prev_frame_id, fid, row_idx))
                        elif prev_frame_id >= 0 and fid != prev_frame_id + 1:
                            issues.append("CRITICAL: non-consecutive frame_id {} -> {} at row {}".format(
                                prev_frame_id, fid, row_idx))
                        prev_frame_id = fid
                    except (KeyError, ValueError, TypeError):
                        pass

                    try:
                        current_episode_id = row["episode_id"]
                        episode_frame_index = int(row["episode_frame_index"])
                        sequence_reset = int(row["sequence_reset"])
                        control_dt_s = float(row["control_dt_s"])
                        timestamp_ns = int(row["timestamp_ns"])
                        expected_dt_s = 1.0 / float(self._record_hz)
                        if row_idx == 1:
                            episode_id = current_episode_id
                            if (episode_frame_index != 0 or
                                    sequence_reset != 1):
                                issues.append(
                                    "CRITICAL: invalid_sequence_reset at row "
                                    "{}".format(row_idx))
                        else:
                            if (current_episode_id != episode_id or
                                    episode_frame_index !=
                                    previous_episode_frame_index + 1):
                                issues.append(
                                    "CRITICAL: episode_frame_index_discontinuity "
                                    "at row {}".format(row_idx))
                            if sequence_reset != 0:
                                issues.append(
                                    "CRITICAL: invalid_sequence_reset at row "
                                    "{}".format(row_idx))
                            if timestamp_ns <= previous_timestamp_ns:
                                issues.append(
                                    "CRITICAL: timestamp_discontinuity at row "
                                    "{}".format(row_idx))
                        if (not np.isfinite(control_dt_s) or
                                control_dt_s <= 0.0 or
                                abs(control_dt_s - expected_dt_s) >
                                max(1.0e-6, 0.05 * expected_dt_s)):
                            issues.append(
                                "CRITICAL: invalid_control_dt at row {}".format(
                                    row_idx))
                        previous_episode_frame_index = episode_frame_index
                        previous_timestamp_ns = timestamp_ns
                    except (KeyError, ValueError, TypeError, ZeroDivisionError):
                        issues.append(
                            "CRITICAL: episode_frame_index_discontinuity at row "
                            "{}".format(row_idx))

                    # Check normalized distances in [0, 1]
                    for fname in ("global_distance_norm", "guide_distance_norm"):
                        val = row.get(fname)
                        if val is not None and val != "":
                            try:
                                v = float(val)
                                if v < -1e-6 or v > 1.0 + 1e-6:
                                    issues.append("{}={} out of [0,1] at row {}".format(fname, v, row_idx))
                            except ValueError:
                                pass

                    try:
                        expert = np.array([
                            float(row.get("expert_vx_flu", "nan")),
                            float(row.get("expert_vy_flu", "nan")),
                            float(row.get("expert_vz_flu", "nan"))])
                        expert_yaw = float(row.get("expert_yaw_rate", "nan"))
                        if (not np.all(np.isfinite(expert)) or
                                not np.isfinite(expert_yaw) or
                                np.linalg.norm(expert) > max_velocity + 1e-6 or
                                abs(expert_yaw) > max_yaw_rate + 1e-6):
                            issues.append("CRITICAL: invalid expert command at row {}".format(row_idx))
                    except (ValueError, TypeError):
                        issues.append("CRITICAL: unparseable expert command at row {}".format(row_idx))

                    try:
                        expert_command = np.array([
                            float(row["expert_vx_flu"]),
                            float(row["expert_vy_flu"]),
                            float(row["expert_vz_flu"]),
                            float(row["expert_yaw_rate"])])
                        selected_command = np.array([
                            float(row["selected_command_vx_flu"]),
                            float(row["selected_command_vy_flu"]),
                            float(row["selected_command_vz_flu"]),
                            float(row["selected_command_yaw_rate"])])
                        applied_command = np.array([
                            float(row["applied_command_vx_flu"]),
                            float(row["applied_command_vy_flu"]),
                            float(row["applied_command_vz_flu"]),
                            float(row["applied_command_yaw_rate"])])
                        current_previous_command = np.array([
                            float(row["previous_executed_command_vx_flu"]),
                            float(row["previous_executed_command_vy_flu"]),
                            float(row["previous_executed_command_vz_flu"]),
                            float(row["previous_executed_command_yaw_rate"])])

                        if str(row.get("expert_label_valid", "0")) == "1":
                            if not np.allclose(
                                    expert_command, selected_command,
                                    atol=command_compare_atol, rtol=0.0):
                                issues.append(
                                    "CRITICAL: expert_selected_command_mismatch "
                                    "at row {}".format(row_idx))
                        if (str(row.get("applied_command_valid", "0")) != "1" or
                                not np.allclose(
                                    selected_command, applied_command,
                                    atol=command_compare_atol, rtol=0.0)):
                            issues.append(
                                "CRITICAL: selected_applied_command_mismatch "
                                "at row {}".format(row_idx))
                        if (row.get("final_executed_actor", "") !=
                                row.get("applied_command_actor", "")):
                            issues.append(
                                "CRITICAL: selected_applied_actor_mismatch "
                                "at row {}".format(row_idx))

                        previous_valid = (
                            str(row["previous_executed_command_valid"]) == "1")
                        if row_idx == 1:
                            if previous_valid or not np.allclose(
                                    current_previous_command,
                                    np.zeros(4, dtype=np.float64),
                                    atol=command_compare_atol, rtol=0.0):
                                issues.append(
                                    "CRITICAL: invalid_episode_start_previous_command")
                        elif (not previous_valid or
                              previous_applied_command is None or
                              not np.allclose(
                                  current_previous_command,
                                  previous_applied_command,
                                  atol=command_compare_atol, rtol=0.0) or
                              int(row["previous_executed_command_frame_id"]) !=
                              previous_applied_frame_id):
                            issues.append(
                                "CRITICAL: previous_applied_command_discontinuity "
                                "at row {}".format(row_idx))

                        previous_applied_command = applied_command.copy()
                        previous_applied_frame_id = int(row["frame_id"])
                    except (KeyError, ValueError, TypeError):
                        issues.append(
                            "CRITICAL: unparseable command consistency fields "
                            "at row {}".format(row_idx))

                    if str(row.get("learner_output_valid", "0")) == "1":
                        try:
                            learner = np.array([
                                float(row.get("learner_vx_flu", "nan")),
                                float(row.get("learner_vy_flu", "nan")),
                                float(row.get("learner_vz_flu", "nan"))])
                            learner_yaw = float(row.get("learner_yaw_rate", "nan"))
                            out_of_bounds = (
                                not np.all(np.isfinite(learner)) or
                                not np.isfinite(learner_yaw) or
                                np.linalg.norm(learner) > max_velocity + 1e-6 or
                                abs(learner_yaw) > max_yaw_rate + 1e-6)
                            if out_of_bounds and str(row.get("safety_override", "0")) != "1":
                                issues.append("CRITICAL: unsafe learner command not overridden at row {}".format(row_idx))
                        except (ValueError, TypeError):
                            issues.append("CRITICAL: unparseable learner command at row {}".format(row_idx))

                    if row.get("planner_map_source", "") != "global_esdf":
                        issues.append(
                            "CRITICAL: planner map is not global ESDF at row {}".format(
                                row_idx))
                    if str(row.get("planner_used_observed_esdf", "1")) != "0":
                        issues.append(
                            "CRITICAL: planner used observed ESDF at row {}".format(
                                row_idx))
                    if str(row.get("planner_used_global_fallback", "1")) != "0":
                        issues.append(
                            "CRITICAL: planner used global fallback at row {}".format(
                                row_idx))

                    # Valid global direction should have norm ~1
                    gdv = row.get("global_direction_valid")
                    try:
                        gdx = float(row["global_dir_x_flu"])
                        gdy = float(row["global_dir_y_flu"])
                        gdz = float(row["global_dir_z_flu"])
                        global_distance_m = float(row["global_distance_m"])
                        global_distance_norm = float(
                            row["global_distance_norm"])
                        global_norm = math.sqrt(
                            gdx * gdx + gdy * gdy + gdz * gdz)
                        if (str(row.get("frame_valid", "0")) == "1" and
                                str(gdv) != "1"):
                            issues.append(
                                "CRITICAL: missing_global_guide_input at row "
                                "{}".format(row_idx))
                        if str(gdv) == "1" and (
                                not np.isfinite(global_norm) or
                                abs(global_norm - 1.0) > 1.0e-4):
                            issues.append(
                                "CRITICAL: invalid_global_direction_norm at row "
                                "{}".format(row_idx))
                        if (not np.isfinite(global_distance_m) or
                                global_distance_m < 0.0):
                            issues.append(
                                "CRITICAL: invalid_global_distance at row "
                                "{}".format(row_idx))
                        if (not np.isfinite(global_distance_norm) or
                                global_distance_norm < 0.0 or
                                global_distance_norm > 1.0):
                            issues.append(
                                "CRITICAL: invalid_global_distance_norm at row "
                                "{}".format(row_idx))
                    except (KeyError, ValueError, TypeError):
                        issues.append(
                            "CRITICAL: missing_global_guide_input at row "
                            "{}".format(row_idx))

                    try:
                        guide_mode = row["guide_mode"]
                        mode_changed = int(row["guide_mode_changed"])
                        recovery_entered = int(row["recovery_entered"])
                        recovery_exited = int(row["recovery_exited"])
                        horizontal_class = int(
                            row["trend_horizontal_class_13"])
                        vertical_class = int(row["guide_elevation_bin"])
                        guide_value = float(row["guide_distance_norm"])
                        horizontal_soft = np.array([
                            float(row[
                                "trend_horizontal_soft_{:02d}".format(i)])
                            for i in range(TREND_HORIZONTAL_CLASS_COUNT)])
                        vertical_soft = np.array([
                            float(row[
                                "guide_elevation_soft_{}".format(i)])
                            for i in range(trend_v_bins)])
                        expert_translation = np.array([
                            float(row["expert_vx_flu"]),
                            float(row["expert_vy_flu"]),
                            float(row["expert_vz_flu"])])
                        expert_yaw_rate = float(row["expert_yaw_rate"])
                        if previous_schema_guide_mode is None:
                            expected_transition = (0, 0, 0)
                        else:
                            changed = int(
                                guide_mode != previous_schema_guide_mode)
                            expected_transition = (
                                changed,
                                int(changed and
                                    previous_schema_guide_mode == "NORMAL" and
                                    guide_mode != "NORMAL"),
                                int(changed and
                                    previous_schema_guide_mode != "NORMAL" and
                                    guide_mode == "NORMAL"))
                        if ((mode_changed, recovery_entered, recovery_exited) !=
                                expected_transition):
                            issues.append(
                                "CRITICAL: invalid_guide_mode_transition at row "
                                "{}".format(row_idx))
                        previous_schema_guide_mode = guide_mode

                        if guide_mode in ("RECOVER_LEFT", "RECOVER_RIGHT"):
                            expected_class = (
                                TREND_RECOVER_LEFT_CLASS
                                if guide_mode == "RECOVER_LEFT"
                                else TREND_RECOVER_RIGHT_CLASS)
                            if abs(guide_value) > command_compare_atol:
                                issues.append(
                                    "CRITICAL: recovery_guide_value_not_zero "
                                    "at row {}".format(row_idx))
                            recovery_vertical_class = trend_v_bins // 2
                            if vertical_class != recovery_vertical_class:
                                issues.append(
                                    "CRITICAL: recovery_vertical_class_not_center "
                                    "at row {}".format(row_idx))
                            expected_vertical_soft = np.zeros(
                                trend_v_bins, dtype=np.float64)
                            expected_vertical_soft[
                                recovery_vertical_class] = 1.0
                            if not np.allclose(
                                    vertical_soft, expected_vertical_soft,
                                    atol=command_compare_atol, rtol=0.0):
                                issues.append(
                                    "CRITICAL: recovery_vertical_soft_mismatch "
                                    "at row {}".format(row_idx))
                            if (np.linalg.norm(expert_translation) >
                                    command_compare_atol):
                                issues.append(
                                    "CRITICAL: recovery_translation_command_nonzero "
                                    "at row {}".format(row_idx))
                            if horizontal_class != expected_class:
                                issues.append(
                                    "CRITICAL: recovery_horizontal_class_mismatch "
                                    "at row {}".format(row_idx))
                            expected_horizontal_soft = np.zeros(
                                TREND_HORIZONTAL_CLASS_COUNT,
                                dtype=np.float64)
                            expected_horizontal_soft[expected_class] = 1.0
                            if not np.allclose(
                                    horizontal_soft,
                                    expected_horizontal_soft,
                                    atol=command_compare_atol, rtol=0.0):
                                issues.append(
                                    "CRITICAL: recovery_horizontal_soft_mismatch "
                                    "at row {}".format(row_idx))
                            yaw_deadband = float(
                                self.g.get("online_runtime", {}).get(
                                    "recovery", {}).get(
                                    "yaw_deadband_rad", 0.05))
                            recovery_azimuth = float(
                                row.get("recovery_azimuth_rad", 0.0))
                            if abs(recovery_azimuth) <= yaw_deadband:
                                if (abs(expert_yaw_rate) >
                                        command_compare_atol):
                                    issues.append(
                                        "CRITICAL: recovery_yaw_nonzero_in_deadband "
                                        "at row {}".format(row_idx))
                            else:
                                expected_recovery_mode = (
                                    "RECOVER_LEFT"
                                    if recovery_azimuth > 0.0
                                    else "RECOVER_RIGHT")
                                if guide_mode != expected_recovery_mode:
                                    issues.append(
                                        "CRITICAL: recovery_direction_azimuth_mismatch "
                                        "at row {}".format(row_idx))
                                yaw_sign_valid = (
                                    expert_yaw_rate >= 0.0
                                    if recovery_azimuth > 0.0
                                    else expert_yaw_rate <= 0.0)
                                if (abs(expert_yaw_rate) >
                                        command_compare_atol and
                                        not yaw_sign_valid):
                                    issues.append(
                                        "CRITICAL: recovery_yaw_direction_mismatch "
                                        "at row {}".format(row_idx))
                        elif guide_mode == "NORMAL":
                            if horizontal_class < 1 or horizontal_class > 11:
                                issues.append(
                                    "CRITICAL: normal_horizontal_class_out_of_range "
                                    "at row {}".format(row_idx))
                            if (abs(horizontal_soft[0]) >
                                    command_compare_atol or
                                    abs(horizontal_soft[-1]) >
                                    command_compare_atol):
                                issues.append(
                                    "CRITICAL: normal_horizontal_soft_endpoint_nonzero "
                                    "at row {}".format(row_idx))
                            if (vertical_class < 0 or
                                    vertical_class >= trend_v_bins):
                                issues.append(
                                    "CRITICAL: normal_vertical_class_out_of_range "
                                    "at row {}".format(row_idx))
                            if (not np.isfinite(guide_value) or
                                    guide_value < 0.0 or guide_value > 1.0):
                                issues.append(
                                    "CRITICAL: normal_guide_value_out_of_range "
                                    "at row {}".format(row_idx))
                            if row.get("trend_mode", "") == "GOAL_HOLD":
                                (hold_h_class, hold_v_class,
                                 hold_h_soft, hold_v_soft) = \
                                    goal_hold_guide_labels(
                                        TREND_NORMAL_HORIZONTAL_BIN_COUNT,
                                        trend_v_bins)
                                hold_modes_valid = (
                                    row.get("planner_mode", "") ==
                                    "GOAL_HOLD" and
                                    row.get("control_mode", "") ==
                                    "HOLD_POSITION" and
                                    row.get("guide_source", "") ==
                                    "goal_tolerance_hold")
                                hold_targets_valid = (
                                    horizontal_class == hold_h_class and
                                    vertical_class == hold_v_class and
                                    abs(guide_value) <=
                                    command_compare_atol and
                                    np.allclose(
                                        horizontal_soft, hold_h_soft,
                                        atol=command_compare_atol, rtol=0.0) and
                                    np.allclose(
                                        vertical_soft, hold_v_soft,
                                        atol=command_compare_atol, rtol=0.0))
                                hold_command_valid = (
                                    np.linalg.norm(expert_translation) <=
                                    command_compare_atol and
                                    abs(expert_yaw_rate) <=
                                    command_compare_atol)
                                if not hold_modes_valid:
                                    issues.append(
                                        "CRITICAL: invalid_goal_hold_modes "
                                        "at row {}".format(row_idx))
                                if not hold_targets_valid:
                                    issues.append(
                                        "CRITICAL: invalid_goal_hold_targets "
                                        "at row {}".format(row_idx))
                                if not hold_command_valid:
                                    issues.append(
                                        "CRITICAL: nonzero_goal_hold_command "
                                        "at row {}".format(row_idx))
                        else:
                            issues.append(
                                "CRITICAL: invalid_guide_mode at row {}".format(
                                    row_idx))
                    except (KeyError, ValueError, TypeError):
                        issues.append(
                            "CRITICAL: invalid_guide_mode at row {}".format(
                                row_idx))

                    # Valid trend direction should have norm ~1
                    tlv = row.get("trend_label_valid")
                    if tlv is not None and str(tlv) == "1":
                        try:
                            gdx = float(row.get("guide_dir_x_flu_exact", 0))
                            gdy = float(row.get("guide_dir_y_flu_exact", 0))
                            gdz = float(row.get("guide_dir_z_flu_exact", 0))
                            norm = math.sqrt(gdx*gdx + gdy*gdy + gdz*gdz)
                            if abs(norm - 1.0) > 0.01:
                                issues.append("guide_dir norm={:.4f} != 1 at row {}".format(norm, row_idx))
                        except (ValueError, TypeError):
                            pass

                        # Valid trend: soft label sums should be ~1
                        # v11: 13-class horizontal soft labels (trend_horizontal_soft_00..12)
                        h_sum = 0.0
                        for i in range(TREND_HORIZONTAL_CLASS_COUNT):
                            key = "trend_horizontal_soft_{:02d}".format(i)
                            try:
                                h_sum += float(row.get(key, 0))
                            except (ValueError, TypeError):
                                pass
                        if abs(h_sum - 1.0) > 0.02:
                            issues.append("azimuth soft sum={:.4f} != 1 at row {}".format(h_sum, row_idx))

                        v_sum = 0.0
                        for i in range(trend_v_bins):
                            key = "guide_elevation_soft_{}".format(i)
                            try:
                                v_sum += float(row.get(key, 0))
                            except (ValueError, TypeError):
                                pass
                        if abs(v_sum - 1.0) > 0.02:
                            issues.append(
                                "elevation soft sum={:.4f} != 1 at row {}"
                                .format(v_sum, row_idx))

                    # Invalid trend: bin should be -1, soft labels all zero
                    if tlv is not None and str(tlv) == "0":
                        try:
                            ab = int(row.get("guide_azimuth_bin", -1))
                            eb = int(row.get("guide_elevation_bin", -1))
                            if ab != -1:
                                issues.append("invalid trend azimuth_bin={} != -1 at row {}".format(ab, row_idx))
                            if eb != -1:
                                issues.append("invalid trend elevation_bin={} != -1 at row {}".format(eb, row_idx))

                            for i in range(TREND_HORIZONTAL_CLASS_COUNT):
                                key = "trend_horizontal_soft_{:02d}".format(i)
                                try:
                                    v = float(row.get(key, 0))
                                    if abs(v) > 1e-9:
                                        issues.append("invalid trend horizontal_soft non-zero={} at row {}".format(v, row_idx))
                                        break
                                except ValueError:
                                    pass
                            for i in range(trend_v_bins):
                                key = "guide_elevation_soft_{}".format(i)
                                try:
                                    v = float(row.get(key, 0))
                                    if abs(v) > 1e-9:
                                        issues.append("invalid trend elevation_soft non-zero={} at row {}".format(v, row_idx))
                                        break
                                except ValueError:
                                    pass
                        except (ValueError, TypeError):
                            pass

        except Exception as exc:
            issues.append(
                "CRITICAL: schema_v16_validation_exception: {}".format(exc))

        return issues

    def _commit_trajectory(self):
        """Atomically rename .inprogress → final directory."""
        meta_path = os.path.join(self._inprogress_dir, "metadata.json")
        try:
            with open(meta_path, "r") as mf:
                meta = json.load(mf)
            meta["status"] = "committed"
            meta["committed_at_ns"] = int(time.time() * 1e9)
            with open(meta_path, "w") as mf:
                json.dump(meta, mf, indent=2, sort_keys=True)
                mf.flush()
                os.fsync(mf.fileno())
        except Exception as exc:
            rospy.logwarn("[Commit] Could not update metadata: %s", exc)

        if self._dagger_ctrl is not None:
            self._dagger_ctrl.update_round_manifest(
                self._policy_provider.model_path,
                self._policy_provider.model_hash(),
                "{}:traj_{:03d}".format(self.scene_label, self.traj_idx))

        # Atomic rename
        try:
            if os.path.exists(self._final_dir):
                rospy.logwarn("[Commit] Final dir already exists, removing: %s", self._final_dir)
                shutil.rmtree(self._final_dir, ignore_errors=True)
            os.rename(self._inprogress_dir, self._final_dir)
            self.total_committed += 1
            # committed frames = actual CSV data rows, NOT sent control frames
            self.total_frames_committed += self._rec_written_rows
            rospy.loginfo("[Commit] ✓ Committed: %s  (rows=%d)",
                          self._final_dir, self._rec_written_rows)
        except Exception as exc:
            rospy.logerr("[Commit] Rename failed: %s", exc)
            failure_reasons = ["atomic_rename_failed: {}".format(exc)]
            self._reject_trajectory(failure_reasons)

    def _reject_trajectory(self, reasons):
        """Move failed data to _failed/ directory."""
        fail_reason_path = os.path.join(self._inprogress_dir, "failure_reason.json")
        try:
            with open(fail_reason_path, "w") as ff:
                json.dump({"reasons": reasons, "failed_at_ns": int(time.time() * 1e9)},
                          ff, indent=2)
        except Exception:
            pass

        traj_name = os.path.basename(self._inprogress_dir).replace(".inprogress", "")
        failed_root = getattr(self, "_episode_failed_dir", self._failed_dir)
        failed_dest = os.path.join(failed_root,
                                   "{}_{}".format(self.scene_label, traj_name))
        try:
            if os.path.exists(failed_dest):
                shutil.rmtree(failed_dest, ignore_errors=True)
            shutil.move(self._inprogress_dir, failed_dest)
            rospy.logwarn("[Commit] ✗ Rejected: %s → %s  reasons=%s",
                          self._inprogress_dir, failed_dest, reasons)
        except Exception as exc:
            rospy.logerr("[Commit] Could not move failed data: %s", exc)

    def _route_next(self):
        """Route to next trajectory, next scene, next profile, or done."""
        if self.traj_idx >= len(self.current_planned):
            # All tasks for current scene are done
            if self._use_profile_mode and self._current_profile is not None:
                # Profile mode: advance to next scene in profile
                self._task_index_in_scene = 0
                self._scene_index_in_profile += 1
                self._scene_generation_retry_offset = 0
                self.seed_idx += 1
                rospy.loginfo("[FSM] Profile '%s': scene %d complete. Advancing to scene %d/%d.",
                              self._current_profile_name,
                              self._scene_index_in_profile - 1,
                              self._scene_index_in_profile,
                              self._current_profile.scene_count)
                self._enter_state(State.NEXT_CONFIG)
            else:
                # Fixed-scenario diagnostic mode
                self.seed_idx += 1
                self._enter_state(State.NEXT_CONFIG)
        else:
            # More tasks remain in current scene
            self._enter_state(State.RESET_DRONE)

    def _st_next_config(self):
        self._enter_state(State.GENERATE_OBSTACLE_CONFIG)

    # ═══════════════════════════════════════════════════════════════
    #  Backward-compatible state aliases  (no-op, route to main path)
    # ═══════════════════════════════════════════════════════════════

# ============================================================================
#  ROS entry point
# ============================================================================
def main():
    rospy.init_node("il_dataset_manager", anonymous=False)

    # Dry-run mode
    if rospy.get_param("~dry_run", False):
        cfg = load_config()
        g = cfg["global"]
        rospy.loginfo("=" * 60)
        rospy.loginfo("  DRY RUN — configuration summary")
        rospy.loginfo("=" * 60)
        rospy.loginfo("  Scene ID:       %d", g["scene_id"])
        rospy.loginfo("  Depth:          %dx%d", g["depth"]["width"], g["depth"]["height"])
        rospy.loginfo("  Control:        %.0f Hz  |  Record: %.0f Hz",
                      g["control"]["control_hz"], g["control"]["record_hz"])
        rospy.loginfo("  Nominal speed:  %.1f m/s  |  Max: %.1f m/s",
                      g["planning"]["time_param"]["nominal_speed"],
                      g["planning"]["time_param"]["max_velocity"])
        rospy.loginfo("  ESDF resolution: %.2f m", g["esdf"]["resolution"])
        rospy.loginfo("  Output dir:     %s", g["output_dir"])
        scene_cfg = g["scene_generation"]
        source = scene_cfg["source"]
        rospy.loginfo("  Scene source:   %s", source)
        if source == "density_driven":
            for profile in scene_cfg.get("profiles", []):
                if profile.get("enabled", True):
                    rospy.loginfo(
                        "  '%s': %d scenes, density=%.3f–%.3f",
                        profile["name"], profile["scene_count"],
                        profile["density_min"], profile["density_max"])
        else:
            rospy.loginfo(
                "  Fixed scene:     %s  |  obstacles=%d  |  tasks=%d",
                scene_cfg.get("fixed_scene_name", "unnamed"),
                len(scene_cfg.get("fixed_obstacles", [])),
                len(scene_cfg.get(
                    "common_task_generation", {}).get("fixed_tasks", [])))
        rospy.loginfo("=" * 60)
        return

    cfg = load_config()
    mgr = ILManager(cfg)
    mgr.run()


if __name__ == "__main__":
    main()
