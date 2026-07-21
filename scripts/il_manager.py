#!/usr/bin/env python3
"""
Imitation-Learning Dataset Manager  —  State-Machine Orchestrator  v5
===========================================================================

Controls the full data-collection workflow via a formal finite-state
machine.  Each state performs its action, waits for a condition (with
timeout), then transitions to the next state.

Key improvements over v4:
  - Receding-horizon local planner (C++ pybind11 backend)
  - Online re-planning with execute-prefix model
  - Global A* serves as reference path only
  - Planner worker thread with GIL release
  - Extended data.csv columns for planner metadata
  - global_path.csv and local_plans.csv sidecar files
  - Schema version 5

Usage:
    roslaunch il_dataset il_dataset_collect.launch
"""

from __future__ import print_function, division

import json, math, os, sys, time, random, copy, threading, shutil, traceback, csv
import numpy as np

import rospy
import rospkg

# Ensure this script's directory is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Also add flightmare_dataset_tools scripts dir (for SmartObstacleSampler)
_rp = rospkg.RosPack()
_ft_scripts = os.path.join(_rp.get_path("flightmare_dataset_tools"), "scripts")
if os.path.isdir(_ft_scripts) and _ft_scripts not in sys.path:
    sys.path.insert(0, _ft_scripts)

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
    matplotlib.use("Agg")
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
    UnityBridge, ESDFBuilder, SyncBuffer, ObstacleGenerator,
    StartGoalGenerator, make_depth_vehicle, make_dummy_vehicle,
    world_vel_to_body, load_ply, wait_for_stable_file,
    integrate_velocity_command,
    body_rfu_to_flu, body_flu_to_rfu, world_vector_to_body_flu,
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
        SceneManifestWriter, ObstacleVisibilityAuditor,
    )
    _SCENARIO_AVAILABLE = True
except ImportError:
    _SCENARIO_AVAILABLE = False
    CylinderObstacleSpec = None  # fallback

from il_config import load_config

# Import il_trajectory from THIS package (avoid shadowing by flightmare_dataset_tools)
import importlib.util as _importlib_util
_il_traj_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "il_trajectory.py")
_il_traj_spec = _importlib_util.spec_from_file_location("il_trajectory", _il_traj_path)
_il_traj = _importlib_util.module_from_spec(_il_traj_spec)
_il_traj_spec.loader.exec_module(_il_traj)
GlobalPathPlanner = _il_traj.GlobalPathPlanner
TrajectoryPlanner = _il_traj.TrajectoryPlanner  # for backward compat

# ── C++ local planner (optional — will be None if not available) ─────
_CPP_PLANNER_AVAILABLE = False
_LocalPlanner = None
_LocalPlannerConfig = None
_VehicleState = None
_TrajectoryPoint = None
_LocalPlanResult = None
_PlannerStatus = None

try:
    import _il_local_planner as _cpp_planner
    _LocalPlanner = _cpp_planner.LocalPlanner
    _LocalPlannerConfig = _cpp_planner.LocalPlannerConfig
    _VehicleState = _cpp_planner.VehicleState
    _TrajectoryPoint = _cpp_planner.TrajectoryPoint
    _LocalPlanResult = _cpp_planner.LocalPlanResult
    _PlannerStatus = _cpp_planner.PlannerStatus
    _CPP_PLANNER_AVAILABLE = True
    rospy.loginfo("[Manager] C++ local planner loaded successfully.")
except ImportError as e:
    rospy.logwarn("[Manager] C++ local planner not available: %s", e)
    rospy.logwarn("[Manager] Will use Python fallback (limited functionality).")


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

    # ── Schema v8 ordered field list for csv.DictWriter ───────────────
    # Phase 2 adds observed map, guide selection, and planner diagnostics.
    DATA_SCHEMA_V8_FIELDS = [
        # -- time & matching --
        "timestamp_ns", "receive_timestamp_ns", "frame_id",
        "trajectory_time_s", "latency_ms", "match_method",
        # -- current state (x_t, before executing expert command) --
        "x", "y", "z", "qx", "qy", "qz", "qw",
        "state_vx_world", "state_vy_world", "state_vz_world",
        "state_vx_flu", "state_vy_flu", "state_vz_flu",
        # -- expert supervision (sampled from plan generated from x_t) --
        "expert_label_valid",
        "expert_vx_world", "expert_vy_world", "expert_vz_world",
        "expert_vx_flu", "expert_vy_flu", "expert_vz_flu",
        "expert_yaw_rate",
        # -- executed next state (x_(t+1), after dt_sample integration) --
        "executed_next_x", "executed_next_y", "executed_next_z",
        "executed_next_vx_world", "executed_next_vy_world", "executed_next_vz_world",
        "executed_next_vx_flu", "executed_next_vy_flu", "executed_next_vz_flu",
        "executed_next_yaw", "executed_next_yaw_rate",
        # -- global navigation labels --
        "global_direction_valid",
        "global_dir_x_flu", "global_dir_y_flu", "global_dir_z_flu",
        "global_distance_m", "global_distance_norm",
        # -- temporary trend labels (stage 2: farthest_visible_astar_waypoint) --
        "trend_label_valid", "guide_source",
        "guide_x_world", "guide_y_world", "guide_z_world",
        "guide_dir_x_flu_exact", "guide_dir_y_flu_exact", "guide_dir_z_flu_exact",
        "guide_distance_m", "guide_distance_norm",
        "guide_azimuth_rad", "guide_elevation_rad",
        "guide_azimuth_bin", "guide_elevation_bin",
        # guide_azimuth_soft_0 ... guide_elevation_soft_{V-1} (appended dynamically)
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
        # -- legacy compatibility --
        "legacy_state_vx_rfu", "legacy_state_vy_rfu", "legacy_state_vz_rfu",
        "legacy_plan_age_ms",
        # -- Phase 2: observed map diagnostics --
        "observed_map_revision",
        "observed_known_voxel_count",
        "observed_occupied_voxel_count",
        "observed_free_voxel_count",
        # -- Phase 2: guide selection diagnostics --
        "guide_candidate_count",
        "guide_visible",
        "guide_depth_visible",
        "guide_corridor_known_free_ratio",
        "guide_path_index",
        "guide_rejection_reason",
        # -- Phase 2: terminal diagnostics --
        "terminal_path_index",
        "terminal_distance_m",
        "terminal_path_arc_length_m",
        # -- Phase 2: planner observed ESDF flags --
        "planner_used_observed_esdf",
        "planner_unknown_is_free",
        "planner_used_global_fallback",
        "reference_segment_point_count",
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

        if self._planner_backend == "cpp_pybind" and _CPP_PLANNER_AVAILABLE:
            self._init_cpp_planner(lp_cfg)
        else:
            rospy.logwarn("[Manager] Using Python-only mode (no C++ local planner).")
            self._planner_backend = "python_fallback"
            self._cpp_planner = None

        # ── Phase 2: observed map, ESDF, camera model, guide selector ──
        obs_cfg = self.g.get("observed_map", {})
        self._use_observed_map = bool(obs_cfg.get("enabled", False))
        self._observed_map = None
        self._observed_esdf = None
        self._camera_model = None
        self._guide_selector = None
        self._guide_progress_index = -1
        self._consecutive_guide_failures = 0

        if self._use_observed_map and _OBSERVED_MAP_AVAILABLE:
            self._observed_map = RollingObservedOccupancyMap(config)
            self._observed_esdf = ObservedESDF(config)
            self._camera_model = PinholeCameraModel(
                self._depth_cfg["width"], self._depth_cfg["height"],
                math.radians(self._depth_cfg["fov"]))
            if _GUIDE_SELECTOR_AVAILABLE:
                self._guide_selector = GuideSelector(config, self._camera_model)
            rospy.loginfo("[Manager] Phase 2: observed map + ESDF + guide selector enabled.")
        elif self._use_observed_map:
            rospy.logwarn("[Manager] Phase 2 modules not available; falling back to global ESDF.")
            self._use_observed_map = False

        # ── Phase 3: scene & task generation ────────────────────────
        sg_cfg = self.g.get("scene_generation", {})
        self._use_scene_gen = bool(sg_cfg.get("enabled", False))
        self._scene_generator = None
        self._scene_validator = None
        self._task_generator = None
        self._side_cost_eval = None
        self._manifest_writer = None
        self._obs_auditor = None

        if self._use_scene_gen and _SCENARIO_AVAILABLE:
            self._scene_generator = YamlCylinderSceneGenerator(config)
            self._scene_validator = CylinderSceneValidator(config)
            self._task_generator = StartGoalTaskGenerator(config)
            self._side_cost_eval = SideCostEvaluator(config)
            self._manifest_writer = SceneManifestWriter(self.output_root)
            self._obs_auditor = ObstacleVisibilityAuditor(config)
            rospy.loginfo("[Manager] Phase 3: scene + task generation + observability audit enabled.")
        elif self._use_scene_gen:
            rospy.logwarn("[Manager] Phase 3 modules not available.")
            self._use_scene_gen = False

        # Scene/task tracking
        self._current_scene_obstacles = []  # list of CylinderObstacleSpec
        self._current_scene_validation = None
        self._current_task_validation = None
        self._current_dominant_obstacle = None
        self._current_scene_subseed = 0
        self._current_scene_attempt = 0
        self._invalid_obs_frame_count = 0

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
        self.current_pairs = []
        self.current_planned = []
        self.scene_label = ""

        # Per-trajectory in-progress tracking
        self._inprogress_dir = None
        self._inprogress_file = None
        self._sync_file = None
        self._global_path_file = None
        self._local_plans_file = None

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
        cfg.execute_prefix_time = float(lp_cfg.get("execute_prefix_time", 0.60))
        cfg.max_plan_age = float(lp_cfg.get("max_plan_age", 0.75))
        cfg.planning_time_budget_ms = float(lp_cfg.get("planning_time_budget_ms", 40.0))
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
        cfg.max_iterations = int(lp_cfg.get("max_iterations", 60))
        cfg.convergence_tolerance = float(lp_cfg.get("convergence_tolerance", 1e-4))
        cfg.initial_step_size = float(lp_cfg.get("initial_step_size", 0.1))
        cfg.minimum_step_size = float(lp_cfg.get("minimum_step_size", 1e-4))
        cfg.max_cost_samples_per_segment = int(
            lp_cfg.get("max_cost_samples_per_segment", 64))

        cfg.min_clearance = float(lp_cfg.get("min_clearance", 0.10))
        cfg.target_clearance = float(lp_cfg.get("target_clearance", 0.20))
        cfg.collision_check_spacing = float(lp_cfg.get("collision_check_spacing", 0.05))

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
        rospy.loginfo("  IL Manager v3 – FSM starting")
        rospy.loginfo("  Output: %s", self.output_root)
        rospy.loginfo("  Scenes: %d", len(self.cfg["scenes"]))
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
    def _close_open_files(self):
        for attr in ("_inprogress_file", "_sync_file", "_global_path_file", "_local_plans_file"):
            f = getattr(self, attr, None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
                setattr(self, attr, None)

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
        # Phase 3: new scene generation pipeline
        if self._use_scene_gen and self._scene_generator is not None:
            if self.scene_idx >= int(self.g.get("scene_generation", {}).get(
                    "max_scene_generation_attempts", 200)):
                self._enter_state(State.DONE)
                return

            rospy.loginfo("=" * 60)
            rospy.loginfo("  PHASE 3 SCENE GENERATION: attempt %d", self.scene_idx + 1)
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

                    self.scene_label = "scene_{:04d}_sub{:04d}".format(
                        self.scene_idx, sub_seed)
                    rospy.loginfo("[SceneGen] ACCEPTED: %d obstacles, subseed=%d, attempt=%d/%d",
                                  len(obstacles), sub_seed, attempt + 1,
                                  self._scene_generator.max_scene_attempts)
                    break
                else:
                    rospy.logwarn("[SceneGen] Attempt %d REJECTED: %s",
                                  attempt + 1, validation.rejection_reason)

            if not obstacles or (validation is not None and not validation.valid):
                rospy.logerr("[SceneGen] Exhausted all %d attempts for scene %d.",
                             self._scene_generator.max_scene_attempts, self.scene_idx)
                self.scene_idx += 1
                self._enter_state(State.NEXT_CONFIG)
                return

            self.scene_idx += 1
            self._enter_state(State.WAIT_SCENE_READY,
                              self.g["fsm"]["scene_settle_timeout"])
            return

        # Legacy: original obstacle generation from scenes list
        if self.scene_idx >= len(self.cfg["scenes"]):
            self._enter_state(State.DONE)
            return

        self.current_scene_cfg = self.cfg["scenes"][self.scene_idx]
        seeds = self.current_scene_cfg.get("seeds", [0])
        if self.seed_idx >= len(seeds):
            self.scene_idx += 1
            self.seed_idx = 0
            self._enter_state(State.NEXT_CONFIG)
            return

        self.current_seed = seeds[self.seed_idx]
        name = self.current_scene_cfg["name"]
        self.scene_label = "{}_seed{:04d}".format(name, self.current_seed)
        rospy.loginfo("=" * 60)
        rospy.loginfo("  SCENE: %s", self.scene_label)
        rospy.loginfo("=" * 60)

        gen = ObstacleGenerator(self.g, self.current_scene_cfg)
        self.current_obstacles = gen.generate(self.current_seed)
        self.current_obj_list = ObstacleGenerator.to_unity_objects(self.current_obstacles)

        self._enter_state(State.WAIT_SCENE_READY,
                          self.g["fsm"]["scene_settle_timeout"])

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
            self.current_esdf, self.current_esdf_origin, self.current_esdf_stats = \
                self.esdf_builder.build(self.current_ply_path)
            self._enter_state(State.GENERATE_START_GOAL_PAIRS)
        except Exception as exc:
            rospy.logerr("ESDF build failed: %s", exc)
            traceback.print_exc()
            self._enter_state(State.ERROR)

    def _st_generate_start_goal_pairs(self):
        # Phase 3: new task generation pipeline
        if (self._use_scene_gen and self._task_generator is not None and
                self._scene_generator is not None and
                len(self._current_scene_obstacles) > 0):
            rospy.loginfo("[FSM] Phase 3: generating tasks via StartGoalTaskGenerator...")

            # Create A* planner for task validation
            from il_trajectory import AStarPlanner as _AStarPlanner

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

            tasks = self._task_generator.generate_tasks(
                self._current_scene_obstacles,
                self.current_esdf, self.current_esdf_origin,
                self.g["esdf"]["resolution"],
                astar_fn,
                seed=self._current_scene_subseed)

            # Convert to current_pairs format
            self.current_pairs = []
            for start, goal, task_val in tasks:
                pair = {"start": start, "goal": goal, "valid_endpoints": True,
                        "_task_validation": task_val}
                self.current_pairs.append(pair)

            self._desired_pair_count = len(self.current_pairs)
            self.traj_idx = 0
            self.current_planned = []
            rospy.loginfo("[FSM] Phase 3: Generated %d validated tasks.", len(self.current_pairs))

            # Write scene manifest
            if self._manifest_writer is not None and self._current_scene_validation is not None:
                scene_id = "scene_{:04d}".format(self.scene_idx - 1)
                task_results_for_manifest = [
                    pair.get("_task_validation") for pair in self.current_pairs
                    if pair.get("_task_validation") is not None]
                self._manifest_writer.write_scene_manifest(
                    scene_id,
                    self._scene_generator.base_seed,
                    self._current_scene_attempt,
                    self._current_scene_subseed,
                    self._current_scene_obstacles,
                    self._current_scene_validation,
                    task_results_for_manifest,
                    self._scene_generator.obstacle_region)
                # Write task manifests
                for ti, pair in enumerate(self.current_pairs):
                    tv = pair.get("_task_validation")
                    if tv is not None:
                        self._manifest_writer.write_task_manifest(
                            scene_id,
                            "task_{:03d}".format(ti),
                            pair["start"], pair["goal"], tv)

            self._enter_state(State.PLAN_GLOBAL_PATHS)
            return

        # Legacy: original start-goal pair generation
        sg_cfg = self.g["start_goal"]
        num = sg_cfg.get("num_pairs_per_config", 5)
        candidate_multiplier = max(
            1, int(sg_cfg.get("candidate_pair_multiplier", 2)))
        candidate_num = num * candidate_multiplier
        gen = StartGoalGenerator(sg_cfg)
        self.current_pairs = gen.generate_pairs(
            candidate_num, self.current_esdf, self.current_esdf_origin,
            self.g["esdf"]["resolution"], seed=self.current_seed)
        self._desired_pair_count = num
        self.traj_idx = 0
        self.current_planned = []
        rospy.loginfo("[FSM] Generated %d start→goal pairs.", len(self.current_pairs))
        self._enter_state(State.PLAN_GLOBAL_PATHS)

    def _st_plan_global_paths(self):
        """v5: Plan only global reference paths (A* + shortcut), not full trajectories."""
        planner = GlobalPathPlanner(
            self.current_esdf, self.current_esdf_origin,
            self.g["esdf"]["resolution"], self.g)

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

            plan = planner.plan_global(start, goal)
            if plan is None:
                rospy.logwarn("  A* FAILED for start→goal pair %d", pi + 1)
                plan = {
                    "start": list(start), "goal": list(goal), "valid": False,
                    "validation_report": {"total_violations": 999,
                                          "invalid_reasons": ["astar_failed"]},
                    "raw_path": [], "global_path": [],
                    "global_path_length": 0.0,
                    "sampled_traj": [], "controls": [],
                    "optimised_path": None, "total_time": 0.0,
                }
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
        """v5: Return initial yaw from global path direction."""
        plan = self.current_planned[self.traj_idx]
        global_path = plan.get("global_path", [])
        if len(global_path) >= 2:
            dx = global_path[1][0] - global_path[0][0]
            dy = global_path[1][1] - global_path[0][1]
            return math.atan2(dy, dx) - math.pi / 2.0
        return 0.0

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
        vehicle = make_depth_vehicle(start, init_yaw, self._depth_cfg)
        msg = {"scene_id": self.g["scene_id"], "frame_id": 0,
               "vehicles": [vehicle], "objects": self.current_obj_list}
        self.bridge.send_pose(msg)
        self._enter_state(State.WAIT_DRONE_STABLE,
                          self.g["fsm"]["drone_stable_timeout"])

    def _st_wait_drone_stable(self):
        if self._timed_out():
            self._enter_state(State.INIT_LOCAL_PLANNER)
            return
        plan = self.current_planned[self.traj_idx]
        start = plan["start"]
        init_yaw = self._get_current_initial_yaw()
        vehicle = make_depth_vehicle(start, init_yaw, self._depth_cfg)
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
        if (self._current_task_validation is not None and
                self._current_task_validation.dominant_obstacle_id):
            # Find the dominant obstacle object
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
                    rospy.loginfo("[FSM] Observed map reset at start position.")
                self._guide_progress_index = -1
                self._consecutive_guide_failures = 0

                # Set global ESDF (for legacy fallback / global path tracking)
                esdf_data = np.asarray(self.current_esdf, dtype=np.float32, order='C')
                origin = np.array(self.current_esdf_origin, dtype=np.float64)
                ok = self._cpp_planner.set_esdf(
                    esdf_data, origin, self.g["esdf"]["resolution"])
                if not ok:
                    rospy.logerr("[FSM] Failed to set ESDF in C++ planner")
                    self._enter_state(State.ERROR)
                    return

                # Set global path (copied once into C++)
                gp_np = np.array(global_path, dtype=np.float64, order='C')
                ok = self._cpp_planner.set_global_path(gp_np)
                if not ok:
                    rospy.logerr("[FSM] Failed to set global path in C++ planner")
                    self._enter_state(State.ERROR)
                    return

                # Reset planner state
                init_state = _VehicleState()
                init_state.position = (float(start[0]), float(start[1]), float(start[2]))
                init_state.velocity = (0.0, 0.0, 0.0)
                init_state.acceleration = (0.0, 0.0, 0.0)
                init_state.yaw = float(init_yaw)
                init_state.yaw_rate = 0.0
                self._cpp_planner.reset(init_state)

                rospy.loginfo("[FSM] C++ local planner initialized for trajectory %d.",
                              self.traj_idx + 1)
            except Exception as exc:
                rospy.logerr("[FSM] C++ planner init failed: %s", exc)
                traceback.print_exc()
                if self._planner_backend == "cpp_pybind":
                    self._enter_state(State.ERROR)
                    return
        else:
            rospy.loginfo("[FSM] No C++ planner; using kinematic pass-through mode.")

        self._enter_state(State.START_RECORDING)

    def _st_start_recording(self):
        """v7: Set up recording files.  Branches to lockstep or legacy async mode."""
        collection_mode = self.g.get("data", {}).get(
            "collection_mode", "deterministic_lockstep")

        if collection_mode == "deterministic_lockstep":
            self._enter_state(State.ONLINE_PLAN_AND_RECORD,
                              self.g["fsm"]["trajectory_timeout"])
        elif collection_mode == "legacy_async":
            # Route to legacy async handler (renamed)
            self._enter_state(State.ONLINE_PLAN_AND_RECORD,
                              self.g["fsm"]["trajectory_timeout"])
        else:
            rospy.logwarn("[FSM] Unknown collection_mode '%s', using lockstep.",
                          collection_mode)
            self._enter_state(State.ONLINE_PLAN_AND_RECORD,
                              self.g["fsm"]["trajectory_timeout"])

        # Create .inprogress directory
        traj_name = "traj_{:03d}".format(self.traj_idx + 1)
        self._inprogress_dir = os.path.join(
            self.output_root, self.scene_label, traj_name + ".inprogress")
        self._final_dir = os.path.join(
            self.output_root, self.scene_label, traj_name)

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
        collection_mode = self.g.get("data", {}).get(
            "collection_mode", "deterministic_lockstep")
        if collection_mode == "legacy_async":
            self._st_online_plan_and_record_legacy_async()
            return

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
        label_lookahead_time_s = float(data_cfg.get("label_lookahead_time_s", 0.08))
        max_guide_range = float(self._depth_cfg["max_m"])

        lp_cfg = self.g.get("planning", {}).get("local_planner", {})
        max_velocity = float(lp_cfg.get("max_velocity", 2.5))
        max_acceleration = float(lp_cfg.get("max_acceleration", 3.5))
        max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))
        goal_tolerance = float(lp_cfg.get("goal_tolerance", 0.30))
        goal_speed_tol = float(lp_cfg.get("goal_speed_tolerance", 0.20))
        goal_hold_ticks = int(lp_cfg.get("goal_hold_ticks", 3))
        configured_failure_limit = int(
            lp_cfg.get("max_consecutive_failures", 3))
        unity_response_timeout_s = max(
            0.25, float(self.g.get("sync", {}).get(
                "unity_response_timeout_s", 2.0)))

        # Trend bin config
        trend_h_bins = int(data_cfg.get("trend_horizontal_bins", 11))
        trend_v_bins = int(data_cfg.get("trend_vertical_bins", 7))
        trend_sigma_bins = float(data_cfg.get("trend_soft_sigma_bins", 0.75))
        h_fov_deg = float(self._depth_cfg["fov"])  # horizontal FOV
        h_fov_rad = math.radians(h_fov_deg)
        v_fov_rad = 2.0 * math.atan(
            math.tan(h_fov_rad / 2.0) * img_h / max(img_w, 1))
        h_bin_edges = np.linspace(-h_fov_rad / 2.0, h_fov_rad / 2.0, trend_h_bins)
        v_bin_edges = np.linspace(-v_fov_rad / 2.0, v_fov_rad / 2.0, trend_v_bins)

        # ── Build dynamic field list with soft label columns ──────────
        schema_fields = list(self.DATA_SCHEMA_V8_FIELDS)
        # Insert soft label columns before depth_file
        depth_idx = schema_fields.index("depth_file")
        azi_soft_names = []
        ele_soft_names = []
        for i in range(trend_h_bins):
            name = "guide_azimuth_soft_{}".format(i)
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

        # ── Local plans CSV ──────────────────────────────────────────
        self._local_plans_file = open(lp_path, "w")
        self._local_plans_file.write(
            "plan_id,request_timestamp_ns,"
            "state_x,state_y,state_z,state_vx,state_vy,state_vz,state_yaw,"
            "local_goal_x,local_goal_y,local_goal_z,"
            "progress_s,progress_index,local_goal_index,"
            "status,success,planning_time_ms,min_clearance,traj_point_count\n")

        # ── State variables ──────────────────────────────────────────
        goal_pt = plan["goal"]
        goal_np = np.array(goal_pt)
        global_path_length = plan.get("global_path_length", 0.0)
        cur_pos = np.array(plan["start"], dtype=np.float64)
        cur_vel = np.zeros(3, dtype=np.float64)
        cur_yaw = self._get_current_initial_yaw()

        consecutive_failures = 0
        sample_index = 0
        sent_frame_id = 0
        recording_start_mono = time.monotonic()
        recording_start_epoch_ns = int(time.time() * 1e9)
        last_valid_response_mono = time.monotonic()
        previous_progress_s = -1.0
        goal_hold_counter = 0

        rospy.loginfo("[ONLINE-LOCKSTEP] Starting. ctrl=%.0fHz rec=%.0fHz dt_sample=%.3fs",
                      ctrl_hz, rec_hz, dt_sample)

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

            vehicle = make_depth_vehicle(cur_pos.tolist(), float(cur_yaw), self._depth_cfg)
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
                # Convert uint16 PNG back to float depth for map integration
                depth_m = depth_u16.astype(np.float64) / 65535.0 * depth_max_m
                if Image is not None:
                    png_name = "{:06d}.png".format(sample_index)
                    Image.fromarray(depth_u16, mode="I;16").save(
                        os.path.join(depth_dir, png_name))

            # ── Phase 2: Integrate depth into observed map ─────────
            t_map_start = time.monotonic()
            guide_sel = GuideSelection()
            ref_segment = []
            if self._use_observed_map and self._observed_map is not None and depth_m is not None:
                timestamp_s = (recv_time_mono - recording_start_mono) + sample_index * dt_sample
                self._observed_map.integrate_depth(
                    depth_m, cur_pos.tolist(), float(cur_yaw), timestamp_s)
                self._observed_map.recenter_if_needed(cur_pos)

                # Rebuild observed ESDF
                if (self._observed_map.get_revision() %
                        self._observed_esdf.rebuild_every_n_frames == 0 or
                        not self._observed_esdf.is_built()):
                    t_esdf_start = time.monotonic()
                    occ = self._observed_map.get_occupancy()
                    known = self._observed_map.get_known_mask()
                    self._observed_esdf.rebuild(
                        occ, known,
                        self._observed_map.get_origin(),
                        self._observed_map.get_resolution())
                    t_esdf_ms = (time.monotonic() - t_esdf_start) * 1000.0

                # Select Guide and Terminal
                if self._guide_selector is not None:
                    t_guide_start = time.monotonic()
                    guide_sel = self._guide_selector.select(
                        global_path, self._guide_progress_index,
                        cur_pos, float(cur_yaw), cur_vel, depth_m,
                        self._observed_map, self._observed_esdf)
                    t_guide_ms = (time.monotonic() - t_guide_start) * 1000.0
                    if guide_sel.valid:
                        self._guide_progress_index = max(
                            self._guide_progress_index, guide_sel.guide_path_index)
                        self._consecutive_guide_failures = 0
                        # Get reference segment for planner
                        ref_segment = self._guide_selector.get_reference_segment(
                            global_path,
                            max(0, self._guide_progress_index - 10),
                            guide_sel.terminal_path_index)
                    else:
                        self._consecutive_guide_failures += 1
            t_map_ms = (time.monotonic() - t_map_start) * 1000.0

            # ── Step 3: Plan from current state (x_t) ───────────────
            # The expert command is sampled from a plan generated from x_t.
            # Phase 2: uses observed ESDF and explicit guide/terminal.
            result = None
            plan_success = False
            planner_compute_ms = 0.0
            planner_status_str = "NO_PLAN"
            planner_used_obs = 0
            planner_used_fallback = 0

            if self._cpp_planner is not None:
                # Set the appropriate ESDF on the planner
                if (self._use_observed_map and self._observed_esdf is not None and
                        self._observed_esdf.is_built()):
                    esdf_arr = self._observed_esdf.get_esdf()
                    known_arr = self._observed_esdf.get_known_mask()
                    origin = self._observed_esdf.get_origin()
                    res = self._observed_esdf.get_resolution()
                    # Ensure C-contiguous
                    esdf_arr = np.ascontiguousarray(esdf_arr, dtype=np.float32)
                    known_arr = np.ascontiguousarray(known_arr.astype(np.uint8))
                    self._cpp_planner.set_observed_esdf(
                        esdf_arr, known_arr,
                        np.array(origin, dtype=np.float64),
                        float(res), False)  # unknown_is_free = false
                    planner_used_obs = 1
                elif self.current_esdf is not None:
                    # Fallback: use global ESDF (only for legacy/debug)
                    rospy.logwarn("[ONLINE-LOCKSTEP] No observed ESDF; using global ESDF (legacy).")
                    esdf_data = np.asarray(self.current_esdf, dtype=np.float32, order='C')
                    origin = np.array(self.current_esdf_origin, dtype=np.float64)
                    self._cpp_planner.set_esdf(
                        esdf_data, origin,
                        self.g["esdf"]["resolution"])

                t_plan_start = time.monotonic()
                try:
                    state = _VehicleState()
                    state.position = (float(cur_pos[0]), float(cur_pos[1]),
                                      float(cur_pos[2]))
                    state.velocity = (float(cur_vel[0]), float(cur_vel[1]),
                                      float(cur_vel[2]))
                    state.yaw = float(cur_yaw)
                    state.yaw_rate = 0.0

                    lp_cfg_local = self.g.get("planning", {}).get("local_planner", {})
                    forbid_unknown = bool(lp_cfg_local.get("forbid_unknown_space", True))
                    allow_fb = bool(lp_cfg_local.get("allow_global_map_fallback", False))

                    if guide_sel.valid and ref_segment:
                        # Phase 2: use explicit request if C++ types are available
                        _has_new_types = ('_LocalPlanningRequest' in dir() and
                                          _LocalPlanningRequest is not None)
                        if _has_new_types:
                            req = _LocalPlanningRequest()
                            req.state = state
                            req.previous_progress_s = previous_progress_s
                            req.guide_waypoint = (
                                float(guide_sel.guide_position_world[0]),
                                float(guide_sel.guide_position_world[1]),
                                float(guide_sel.guide_position_world[2]))
                            req.guide_waypoint_index = guide_sel.guide_path_index
                            req.trajectory_terminal = (
                                float(guide_sel.terminal_position_world[0]),
                                float(guide_sel.terminal_position_world[1]),
                                float(guide_sel.terminal_position_world[2]))
                            req.trajectory_terminal_index = guide_sel.terminal_path_index
                            for pt in ref_segment:
                                req.reference_path_segment.append(
                                    (float(pt[0]), float(pt[1]), float(pt[2])))
                            req.forbid_unknown_space = forbid_unknown
                            req.allow_global_map_fallback = allow_fb
                            result = self._cpp_planner.plan_local_with_request(req)
                        else:
                            # Fallback to legacy plan_local if new types not available
                            result = self._cpp_planner.plan_local(state, previous_progress_s)
                    else:
                        # Guide not valid: use legacy plan_local for hover/low-speed
                        result = self._cpp_planner.plan_local(state, previous_progress_s)

                    t_plan_end = time.monotonic()
                    planner_compute_ms = (t_plan_end - t_plan_start) * 1000.0
                    self._total_replans += 1
                    self._planning_times_ms.append(
                        result.planning_time_ms if hasattr(result, 'planning_time_ms') else planner_compute_ms)

                    planner_status_str = str(result.status)
                    plan_success = result.success
                    # Check Phase 2 flags
                    if hasattr(result, 'used_global_fallback'):
                        planner_used_fallback = 1 if result.used_global_fallback else 0
                    if hasattr(result, 'used_observed_esdf'):
                        planner_used_obs = 1 if result.used_observed_esdf else planner_used_obs

                    if result.success:
                        self._successful_replans += 1
                        self._executed_clearances.append(result.min_clearance)
                        previous_progress_s = max(previous_progress_s,
                                                  result.progress_s)
                        consecutive_failures = 0
                        # Write local plan to CSV
                        self._local_plans_file.write(
                            "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},"
                            "{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{:.2f},{:.4f}\n"
                            .format(result.plan_id,
                                    int(t_request_mono * 1e9),
                                    state.position[0], state.position[1], state.position[2],
                                    state.velocity[0], state.velocity[1], state.velocity[2],
                                    state.yaw,
                                    result.local_goal[0], result.local_goal[1], result.local_goal[2],
                                    result.progress_s, result.progress_index, result.local_goal_index,
                                    int(result.status), result.success,
                                    result.planning_time_ms, result.min_clearance,
                                    len(result.trajectory)))
                        self._local_plans_file.flush()
                    else:
                        self._failed_replans += 1
                        consecutive_failures += 1
                        if (int(result.status) == 5 or  # COLLISION
                            (hasattr(_PlannerStatus, 'COLLISION') and
                             result.status == _PlannerStatus.COLLISION)):
                            self._emergency_hold_count += 1
                except Exception as exc:
                    rospy.logerr("[ONLINE-LOCKSTEP] Planner exception: %s", exc)
                    consecutive_failures += 1

            # ── Step 4: Build training row from x_t, plan result, depth ──
            row = self._build_training_row_v8(
                cur_pos, cur_vel, cur_yaw,
                result, plan_success, planner_compute_ms,
                planner_status_str,
                goal_np, goal_pt, global_path_length,
                label_lookahead_time_s, max_guide_range,
                png_name, collision,
                plan, sample_index, dt_sample,
                frame_id, recording_start_mono, recording_start_epoch_ns,
                t_request_mono, recv_time_mono, latency_ms,
                h_fov_rad, v_fov_rad, trend_h_bins, trend_v_bins,
                trend_sigma_bins, h_bin_edges, v_bin_edges,
                azi_soft_names, ele_soft_names,
                previous_progress_s,
                # Phase 2 extras
                guide_sel, ref_segment,
                self._observed_map, self._observed_esdf,
                planner_used_obs, planner_used_fallback,
            )

            # ── Step 5: Execute dt_sample to advance to x_(t+1) ──────
            # The executed_next fields belong to x_(t+1).
            if result is not None and result.success and len(result.trajectory) > 0:
                exec_next_pos, exec_next_vel, exec_next_yaw, exec_next_yaw_rate = \
                    self._execute_trajectory_segment(
                        result, cur_pos.copy(), cur_vel.copy(), float(cur_yaw),
                        dt_sample, dt_ctrl, sample_index,
                        max_velocity, max_acceleration, max_yaw_rate)
            else:
                # Plan failed: hover in place for dt_sample
                exec_next_pos, exec_next_vel, exec_next_yaw, exec_next_yaw_rate = \
                    self._execute_hover(
                        cur_pos.copy(), cur_vel.copy(), float(cur_yaw),
                        dt_sample, dt_ctrl,
                        max_velocity, max_acceleration, max_yaw_rate)

            # Patch executed_next fields with post-execution values
            row["executed_next_x"] = round(float(exec_next_pos[0]), 4)
            row["executed_next_y"] = round(float(exec_next_pos[1]), 4)
            row["executed_next_z"] = round(float(exec_next_pos[2]), 4)
            row["executed_next_vx_world"] = round(float(exec_next_vel[0]), 4)
            row["executed_next_vy_world"] = round(float(exec_next_vel[1]), 4)
            row["executed_next_vz_world"] = round(float(exec_next_vel[2]), 4)
            exec_next_vel_flu = world_vector_to_body_flu(
                exec_next_vel, float(exec_next_yaw))
            row["executed_next_vx_flu"] = round(float(exec_next_vel_flu[0]), 4)
            row["executed_next_vy_flu"] = round(float(exec_next_vel_flu[1]), 4)
            row["executed_next_vz_flu"] = round(float(exec_next_vel_flu[2]), 4)
            row["executed_next_yaw"] = round(float(exec_next_yaw), 6)
            row["executed_next_yaw_rate"] = round(float(exec_next_yaw_rate), 6)

            # track invalid counts for metadata
            if row.get("expert_label_valid", 0) == 0:
                self._invalid_expert_label_count += 1
            if row.get("trend_label_valid", 0) == 0:
                self._invalid_trend_label_count += 1

            # Write row (state_t, depth_t, nav_t, expert_cmd_t, state_(t+1))
            self._csv_writer.writerow(row)
            self._rec_written_rows += 1

            # ── Sync diagnostics ─────────────────────────────────────
            self._sync_file.write("{},{},{:.2f},0.00,frame_id_exact,0,0,{},{}\n".format(
                sample_index, frame_id, latency_ms,
                self._rec_exact_matches, self._rec_fallback_matches))

            # Advance state to x_(t+1)
            cur_pos = exec_next_pos
            cur_vel = exec_next_vel
            cur_yaw = exec_next_yaw

            # ── Goal check ───────────────────────────────────────────
            dist_to_goal = float(np.linalg.norm(cur_pos - goal_np))
            speed = float(np.linalg.norm(cur_vel))
            if dist_to_goal <= goal_tolerance and speed <= goal_speed_tol:
                goal_hold_counter += 1
                if goal_hold_counter >= goal_hold_ticks:
                    rospy.loginfo("[ONLINE-LOCKSTEP] Goal reached at sample %d.", sample_index)
                    self._trajectory_reached_goal = True
                    self._trajectory_exit_reason = "goal_reached"
                    break
            else:
                goal_hold_counter = 0

            # ── Consecutive failure check ───────────────────────────
            if consecutive_failures >= configured_failure_limit:
                self._trajectory_exit_reason = "consecutive_planner_failures"
                rospy.logerr("[ONLINE-LOCKSTEP] %d consecutive planner failures; aborting.",
                             consecutive_failures)
                break

            sample_index += 1
            self._rec_sent_control_frames += 1

        self._final_executed_position = cur_pos.tolist()
        self._final_executed_velocity = cur_vel.tolist()
        if self._trajectory_exit_reason == "running":
            self._trajectory_exit_reason = "shutdown" if rospy.is_shutdown() else "loop_exited"
        self._close_open_files()
        self._st_finish_recording()

    # ═══════════════════════════════════════════════════════════════
    #  Lockstep helper methods
    # ═══════════════════════════════════════════════════════════════

    def _wait_for_exact_depth_frame(self, target_frame_id, depth_float_len,
                                     img_w, img_h, depth_max_m,
                                     unity_response_timeout_s,
                                     last_valid_response_mono):
        """Wait for a Unity depth response with exact frame_id match.

        Returns:
            (depth_u16, collision, recv_frame_id, recv_time_mono)
            If timeout, returns (None, 0, None, None).
            If wrong frame_id, returns (None, 0, wrong_fid, recv_time_mono).
        """
        deadline = last_valid_response_mono + unity_response_timeout_s
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

            # Verify frame_id match
            if _fid != target_frame_id:
                rospy.logwarn("[LOCKSTEP] Discarding stale frame_id=%s (expected %d)",
                              _fid, target_frame_id)
                return None, collision, _fid, recv_time_mono

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

    def _sample_expert_command(self, result, label_lookahead_time_s):
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
        expert_yaw_rate = float(command_point.yaw_rate)
        return expert_vel_world, expert_yaw_rate

    def _execute_trajectory_segment(self, result, cur_pos, cur_vel, cur_yaw,
                                     dt_sample, dt_ctrl, sample_index,
                                     max_velocity, max_acceleration, max_yaw_rate):
        """Execute dt_sample using sub-steps from the plan trajectory.

        Returns:
            (next_pos, next_vel, next_yaw, avg_yaw_rate) — all numpy arrays/scalars.
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
                desired_vel = np.zeros(3, dtype=np.float64)
                pos, vel, yaw, yr = integrate_velocity_command(
                    pos, vel, yaw, desired_vel, step_dt,
                    max_velocity, max_acceleration, max_yaw_rate)
                total_yaw_change += yr * step_dt
                elapsed += step_dt
            avg_yaw_rate = total_yaw_change / max(dt_sample, 1e-9)
            return pos, vel, yaw, avg_yaw_rate

        traj = result.trajectory
        traj_duration = float(traj[-1].t)
        elapsed = 0.0
        epsilon = 1e-9

        while elapsed < dt_sample - epsilon:
            step_dt = min(dt_ctrl, dt_sample - elapsed)
            # Find trajectory point closest to current elapsed time
            exec_time = (sample_index * dt_sample) + elapsed
            clamped_time = min(exec_time, traj_duration)
            best_idx = min(range(len(traj)),
                           key=lambda i: abs(float(traj[i].t) - clamped_time))
            tp = traj[best_idx]
            desired_vel = np.array(tp.velocity, dtype=np.float64)

            pos, vel, yaw, yr = integrate_velocity_command(
                pos, vel, yaw, desired_vel, step_dt,
                max_velocity, max_acceleration, max_yaw_rate)
            total_yaw_change += yr * step_dt
            elapsed += step_dt

        avg_yaw_rate = total_yaw_change / max(dt_sample, 1e-9)
        return pos, vel, yaw, avg_yaw_rate

    def _execute_hover(self, cur_pos, cur_vel, cur_yaw,
                        dt_sample, dt_ctrl,
                        max_velocity, max_acceleration, max_yaw_rate):
        """Execute a zero-velocity hover for dt_sample.

        Returns:
            (next_pos, next_vel, next_yaw, avg_yaw_rate).
        """
        pos = np.asarray(cur_pos, dtype=np.float64).copy()
        vel = np.asarray(cur_vel, dtype=np.float64).copy()
        yaw = float(cur_yaw)
        total_yaw_change = 0.0
        elapsed = 0.0
        epsilon = 1e-9

        while elapsed < dt_sample - epsilon:
            step_dt = min(dt_ctrl, dt_sample - elapsed)
            desired_vel = np.zeros(3, dtype=np.float64)
            pos, vel, yaw, yr = integrate_velocity_command(
                pos, vel, yaw, desired_vel, step_dt,
                max_velocity, max_acceleration, max_yaw_rate)
            total_yaw_change += yr * step_dt
            elapsed += step_dt

        avg_yaw_rate = total_yaw_change / max(dt_sample, 1e-9)
        return pos, vel, yaw, avg_yaw_rate

    def _build_training_row_v8(self, cur_pos, cur_vel, cur_yaw,
                                result, plan_success, planner_compute_ms,
                                planner_status_str,
                                goal_np, goal_pt, global_path_length,
                                label_lookahead_time_s, max_guide_range,
                                png_name, collision,
                                plan, sample_index, dt_sample,
                                frame_id, recording_start_mono, recording_start_epoch_ns,
                                t_request_mono, recv_time_mono, latency_ms,
                                h_fov_rad, v_fov_rad, trend_h_bins, trend_v_bins,
                                trend_sigma_bins, h_bin_edges, v_bin_edges,
                                azi_soft_names, ele_soft_names,
                                previous_progress_s,
                                guide_sel=None, ref_segment=None,
                                observed_map=None, observed_esdf=None,
                                planner_used_obs=0, planner_used_fallback=0):
        """Build a schema v7 training row as a dict.

        IMPORTANT SEMANTICS (documented in comments):
        - The depth image and state fields belong to x_t.
        - The expert command is sampled from a plan generated from x_t.
        - The executed_next fields belong to x_(t+1).
        """
        row = {}

        # ── Time & matching ──────────────────────────────────────
        trajectory_time_s = sample_index * dt_sample
        row["timestamp_ns"] = recording_start_epoch_ns + int(
            (t_request_mono - recording_start_mono) * 1e9)
        row["receive_timestamp_ns"] = recording_start_epoch_ns + int(
            (recv_time_mono - recording_start_mono) * 1e9)
        row["frame_id"] = frame_id
        row["trajectory_time_s"] = round(trajectory_time_s, 6)
        row["latency_ms"] = round(latency_ms, 3)
        row["match_method"] = "frame_id_exact"

        # ── Current state (x_t) ──────────────────────────────────
        row["x"] = round(float(cur_pos[0]), 4)
        row["y"] = round(float(cur_pos[1]), 4)
        row["z"] = round(float(cur_pos[2]), 4)
        # Quaternion (yaw only for now)
        half = 0.5 * float(cur_yaw)
        row["qx"] = 0.0
        row["qy"] = 0.0
        row["qz"] = round(math.sin(half), 6)
        row["qw"] = round(math.cos(half), 6)

        # Current state velocity (world)
        row["state_vx_world"] = round(float(cur_vel[0]), 4)
        row["state_vy_world"] = round(float(cur_vel[1]), 4)
        row["state_vz_world"] = round(float(cur_vel[2]), 4)

        # Current state velocity (FLU)
        state_vel_flu = world_vector_to_body_flu(cur_vel, float(cur_yaw))
        row["state_vx_flu"] = round(float(state_vel_flu[0]), 4)
        row["state_vy_flu"] = round(float(state_vel_flu[1]), 4)
        row["state_vz_flu"] = round(float(state_vel_flu[2]), 4)

        # Legacy RFU body velocity (for debug/compat)
        vx_rfu, vy_rfu, vz_rfu = world_vel_to_body(
            float(cur_vel[0]), float(cur_vel[1]), float(cur_vel[2]),
            float(cur_yaw))
        row["legacy_state_vx_rfu"] = round(vx_rfu, 4)
        row["legacy_state_vy_rfu"] = round(vy_rfu, 4)
        row["legacy_state_vz_rfu"] = round(vz_rfu, 4)

        # ── Expert supervision ───────────────────────────────────
        expert_vel_world, expert_yaw_rate = self._sample_expert_command(
            result, label_lookahead_time_s)

        row["expert_label_valid"] = 1 if (result is not None and plan_success
                                           and len(result.trajectory) > 0) else 0

        row["expert_vx_world"] = round(float(expert_vel_world[0]), 4)
        row["expert_vy_world"] = round(float(expert_vel_world[1]), 4)
        row["expert_vz_world"] = round(float(expert_vel_world[2]), 4)

        expert_vel_flu = world_vector_to_body_flu(expert_vel_world, float(cur_yaw))
        row["expert_vx_flu"] = round(float(expert_vel_flu[0]), 4)
        row["expert_vy_flu"] = round(float(expert_vel_flu[1]), 4)
        row["expert_vz_flu"] = round(float(expert_vel_flu[2]), 4)
        row["expert_yaw_rate"] = round(float(expert_yaw_rate), 6)

        # ── Executed next state placeholders ─────────────────────
        # These are the state BEFORE executing; the AFTER values will
        # be set below after the execution step. For now, store the
        # current values as the "executed_next" baseline. The actual
        # executed_next values are the cur_pos/cur_vel AFTER
        # _execute_trajectory_segment. Since _build_training_row_v8
        # is called BEFORE execution, we store the pre-execution
        # state here as a placeholder which gets overwritten.
        #
        # NOTE: The caller (lockstep loop) must update these after
        # execution. We store them in row and the caller patches them.
        row["executed_next_x"] = round(float(cur_pos[0]), 4)
        row["executed_next_y"] = round(float(cur_pos[1]), 4)
        row["executed_next_z"] = round(float(cur_pos[2]), 4)
        row["executed_next_vx_world"] = round(float(cur_vel[0]), 4)
        row["executed_next_vy_world"] = round(float(cur_vel[1]), 4)
        row["executed_next_vz_world"] = round(float(cur_vel[2]), 4)
        row["executed_next_vx_flu"] = round(float(state_vel_flu[0]), 4)
        row["executed_next_vy_flu"] = round(float(state_vel_flu[1]), 4)
        row["executed_next_vz_flu"] = round(float(state_vel_flu[2]), 4)
        row["executed_next_yaw"] = round(float(cur_yaw), 6)
        row["executed_next_yaw_rate"] = 0.0

        # ── Global navigation labels ─────────────────────────────
        global_delta_world = goal_np - cur_pos
        global_distance_m = float(np.linalg.norm(global_delta_world))
        if global_distance_m < 1e-9:
            row["global_direction_valid"] = 0
            row["global_dir_x_flu"] = 0.0
            row["global_dir_y_flu"] = 0.0
            row["global_dir_z_flu"] = 0.0
        else:
            global_delta_flu = world_vector_to_body_flu(
                global_delta_world, float(cur_yaw))
            norm = float(np.linalg.norm(global_delta_flu))
            row["global_direction_valid"] = 1
            row["global_dir_x_flu"] = round(float(global_delta_flu[0]) / norm, 6)
            row["global_dir_y_flu"] = round(float(global_delta_flu[1]) / norm, 6)
            row["global_dir_z_flu"] = round(float(global_delta_flu[2]) / norm, 6)
        row["global_distance_m"] = round(global_distance_m, 4)
        row["global_distance_norm"] = round(
            min(global_distance_m, max_guide_range) / max(max_guide_range, 1e-9), 6)

        # ── Trend labels (Phase 2: farthest_visible_astar_waypoint) ──
        # If GuideSelector produced a valid guide, use it.
        # Otherwise fall back to legacy local_planner_goal.
        if guide_sel is not None and guide_sel.valid:
            row["guide_source"] = "farthest_visible_astar_waypoint"
            local_goal_world = guide_sel.guide_position_world.copy()
            guide_dist = guide_sel.guide_distance_m
            guide_dir_flu = guide_sel.guide_direction_flu.copy()
            azimuth = guide_sel.azimuth_rad
            elevation = guide_sel.elevation_rad
            guide_norm_val = guide_sel.guide_distance_norm
        else:
            row["guide_source"] = "legacy_local_planner_goal"
            # Temporary stage-1 label source.
            # This will be replaced by the farthest currently visible A* waypoint.
            if (result is not None and result.success and
                    hasattr(result, 'local_goal') and result.local_goal is not None):
                local_goal_world = np.array(result.local_goal, dtype=np.float64)
            else:
                local_goal_world = cur_pos.copy()
            guide_delta_world = local_goal_world - cur_pos
            guide_dist = float(np.linalg.norm(guide_delta_world))
            guide_dir_flu = world_vector_to_body_flu(
                guide_delta_world, float(cur_yaw))
            guide_n = float(np.linalg.norm(guide_dir_flu))
            if guide_n > 1e-9:
                guide_dir_flu = guide_dir_flu / guide_n
            azimuth = math.atan2(float(guide_dir_flu[1]), float(guide_dir_flu[0]))
            elevation = math.atan2(float(guide_dir_flu[2]),
                                   math.sqrt(float(guide_dir_flu[0])**2 +
                                            float(guide_dir_flu[1])**2 + 1e-12))
            guide_norm_val = min(guide_dist, max_guide_range) / max(max_guide_range, 1e-9)

        row["guide_x_world"] = round(float(local_goal_world[0]), 4)
        row["guide_y_world"] = round(float(local_goal_world[1]), 4)
        row["guide_z_world"] = round(float(local_goal_world[2]), 4)

        guide_delta_world = local_goal_world - cur_pos
        guide_distance_m = float(np.linalg.norm(guide_delta_world))
        if guide_distance_m < 1e-9:
            # Local goal coincides with current position
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
            # All soft labels = 0
            for name in azi_soft_names:
                row[name] = 0.0
            for name in ele_soft_names:
                row[name] = 0.0
        else:
            guide_delta_flu = world_vector_to_body_flu(
                guide_delta_world, float(cur_yaw))
            norm = float(np.linalg.norm(guide_delta_flu))
            gdx = float(guide_delta_flu[0]) / norm
            gdy = float(guide_delta_flu[1]) / norm
            gdz = float(guide_delta_flu[2]) / norm
            row["guide_dir_x_flu_exact"] = round(gdx, 6)
            row["guide_dir_y_flu_exact"] = round(gdy, 6)
            row["guide_dir_z_flu_exact"] = round(gdz, 6)

            row["guide_distance_m"] = round(guide_distance_m, 4)
            row["guide_distance_norm"] = round(
                min(guide_distance_m, max_guide_range) / max(max_guide_range, 1e-9), 6)

            # Azimuth / elevation in FLU
            azimuth = math.atan2(gdy, gdx)
            elevation = math.atan2(gdz, math.sqrt(gdx * gdx + gdy * gdy + 1e-12))
            row["guide_azimuth_rad"] = round(azimuth, 6)
            row["guide_elevation_rad"] = round(elevation, 6)

            # Check if guide is within FOV
            in_h_fov = (-h_fov_rad / 2.0 - 1e-9 <= azimuth <=
                         h_fov_rad / 2.0 + 1e-9)
            in_v_fov = (-v_fov_rad / 2.0 - 1e-9 <= elevation <=
                         v_fov_rad / 2.0 + 1e-9)

            if in_h_fov and in_v_fov:
                row["trend_label_valid"] = 1
                # Hard bin
                h_bin = int(np.argmin(np.abs(h_bin_edges - azimuth)))
                v_bin = int(np.argmin(np.abs(v_bin_edges - elevation)))
                row["guide_azimuth_bin"] = h_bin
                row["guide_elevation_bin"] = v_bin

                # Soft labels (Gaussian weights)
                h_soft = np.zeros(trend_h_bins, dtype=np.float64)
                for i in range(trend_h_bins):
                    bin_err = (azimuth - h_bin_edges[i]) / (
                        (h_fov_rad / (trend_h_bins - 1)) + 1e-12)
                    h_soft[i] = math.exp(-0.5 * (bin_err / trend_sigma_bins) ** 2)
                h_soft /= max(h_soft.sum(), 1e-12)
                for i, name in enumerate(azi_soft_names):
                    row[name] = round(float(h_soft[i]), 6)

                v_soft = np.zeros(trend_v_bins, dtype=np.float64)
                for i in range(trend_v_bins):
                    bin_err = (elevation - v_bin_edges[i]) / (
                        (v_fov_rad / (trend_v_bins - 1)) + 1e-12)
                    v_soft[i] = math.exp(-0.5 * (bin_err / trend_sigma_bins) ** 2)
                v_soft /= max(v_soft.sum(), 1e-12)
                for i, name in enumerate(ele_soft_names):
                    row[name] = round(float(v_soft[i]), 6)
            else:
                # Outside FOV
                row["trend_label_valid"] = 0
                row["guide_azimuth_bin"] = -1
                row["guide_elevation_bin"] = -1
                for name in azi_soft_names:
                    row[name] = 0.0
                for name in ele_soft_names:
                    row[name] = 0.0

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
        row["global_progress_s"] = round(
            result.progress_s if result is not None else previous_progress_s, 4)
        row["global_progress_ratio"] = round(
            (result.progress_s if result is not None else previous_progress_s)
            / max(global_path_length, 1e-6), 6)
        row["global_progress_index"] = (
            result.progress_index if result is not None else -1)
        row["local_goal_index"] = (
            result.local_goal_index if result is not None else -1)
        row["plan_id"] = (result.plan_id if result is not None else -1)
        row["plan_time_from_start_s"] = round(
            float(result.trajectory[0].t) if (result is not None
                and len(result.trajectory) > 0) else 0.0, 6)
        row["planner_status"] = planner_status_str
        row["planner_success"] = plan_success
        row["planner_compute_ms"] = round(planner_compute_ms, 3)
        row["planner_min_clearance"] = round(
            result.min_clearance if result is not None else 0.0, 4)
        row["distance_to_final_goal"] = round(
            float(np.linalg.norm(cur_pos - goal_np)), 4)
        row["legacy_plan_age_ms"] = 0.0

        # ── Phase 2: observed map diagnostics ────────────────────
        if observed_map is not None:
            row["observed_map_revision"] = observed_map.get_revision()
            row["observed_known_voxel_count"] = observed_map.known_voxel_count()
            row["observed_occupied_voxel_count"] = observed_map.occupied_voxel_count()
            row["observed_free_voxel_count"] = observed_map.free_voxel_count()
        else:
            row["observed_map_revision"] = -1
            row["observed_known_voxel_count"] = 0
            row["observed_occupied_voxel_count"] = 0
            row["observed_free_voxel_count"] = 0

        # ── Phase 2: guide selection diagnostics ─────────────────
        if guide_sel is not None:
            row["guide_candidate_count"] = guide_sel.candidate_count
            row["guide_visible"] = 1 if guide_sel.visible else 0
            row["guide_depth_visible"] = 1 if guide_sel.depth_visible else 0
            row["guide_corridor_known_free_ratio"] = round(
                guide_sel.corridor_known_free_ratio, 4)
            row["guide_path_index"] = guide_sel.guide_path_index
            row["guide_rejection_reason"] = guide_sel.rejection_reason[:80] if guide_sel.rejection_reason else ""
            row["terminal_path_index"] = guide_sel.terminal_path_index
            row["terminal_distance_m"] = round(
                float(np.linalg.norm(guide_sel.terminal_position_world - cur_pos)), 4)
            # Approximate terminal arc length
            row["terminal_path_arc_length_m"] = round(
                float(np.linalg.norm(
                    guide_sel.terminal_position_world - guide_sel.guide_position_world)) +
                guide_sel.guide_distance_m, 4)
        else:
            row["guide_candidate_count"] = 0
            row["guide_visible"] = 0
            row["guide_depth_visible"] = 0
            row["guide_corridor_known_free_ratio"] = 0.0
            row["guide_path_index"] = -1
            row["guide_rejection_reason"] = "no_selector"
            row["terminal_path_index"] = -1
            row["terminal_distance_m"] = 0.0
            row["terminal_path_arc_length_m"] = 0.0

        # ── Phase 2: planner flags ───────────────────────────────
        row["planner_used_observed_esdf"] = planner_used_obs
        row["planner_unknown_is_free"] = 0  # Phase 2: never treat unknown as free
        row["planner_used_global_fallback"] = planner_used_fallback
        row["reference_segment_point_count"] = (
            len(ref_segment) if ref_segment else 0)

        return row

    # ═══════════════════════════════════════════════════════════════
    #  v5 legacy: ONLINE_PLAN_AND_RECORD (async planner worker)
    #  Renamed from _st_online_plan_and_record for backward compat.
    # ═══════════════════════════════════════════════════════════════

    def _st_online_plan_and_record_legacy_async(self):
        """v5 legacy: Online receding-horizon planning with async planner worker.

        The planner updates a desired velocity asynchronously.  A fixed-rate
        controller limits acceleration/yaw-rate and integrates pose without
        replaying or resetting planner position samples.
        """
        plan = self.current_planned[self.traj_idx]
        global_path = plan.get("global_path", [])
        if not global_path:
            rospy.logwarn("[ONLINE] Empty global path — finishing.")
            self._st_finish_recording()
            return

        ctrl_hz = self._control_hz
        rec_hz = self._record_hz
        dt_ctrl = 1.0 / ctrl_hz
        dt_rec = 1.0 / rec_hz
        depth_max_m = self._depth_cfg["max_m"]
        img_w, img_h = self._depth_cfg["width"], self._depth_cfg["height"]
        depth_float_len = img_w * img_h * 4

        depth_dir = os.path.join(self._inprogress_dir, "depth")
        data_path = os.path.join(self._inprogress_dir, "data.csv")
        sync_path = os.path.join(self._inprogress_dir, "sync.csv")
        gp_path = os.path.join(self._inprogress_dir, "global_path.csv")
        lp_path = os.path.join(self._inprogress_dir, "local_plans.csv")

        sync_buffer = SyncBuffer(max_entries=256)

        # ── Open data files with v5 extended schema ──────────────
        self._inprogress_file = open(data_path, "w")
        self._inprogress_file.write(
            "timestamp_ns,x,y,z,qx,qy,qz,qw,"
            "vel_x_body,vel_y_body,vel_z_body,"
            "ctrl_vx_body,ctrl_vy_body,ctrl_vz_body,ctrl_yaw_rate,"
            "depth_file,collision,"
            "start_x,start_y,start_z,goal_x,goal_y,goal_z,"
            "schema_version,latency_ms,match_error_ms,frame_id,vel_source,"
            "trajectory_time_s,control_frame_id,send_timestamp_ns,"
            "local_goal_x_body,local_goal_y_body,local_goal_z_body,"
            "local_goal_x_world,local_goal_y_world,local_goal_z_world,"
            "global_progress_s,global_progress_ratio,global_progress_index,"
            "local_goal_index,plan_id,plan_time_from_start_s,plan_age_ms,"
            "planner_status,planner_success,planner_compute_ms,"
            "planner_min_clearance,distance_to_final_goal,state_source\n")

        self._sync_file = open(sync_path, "w")
        self._sync_file.write(
            "recv_step,frame_id_matched,latency_ms,match_error_ms,match_method,"
            "is_dropped,ctrl_queue_len,exact_matches,fallback_matches\n")

        # ── Global path CSV ──────────────────────────────────────
        self._global_path_file = open(gp_path, "w")
        self._global_path_file.write("index,x,y,z,s\n")
        cum_s = 0.0
        for idx, pt in enumerate(global_path):
            if idx > 0:
                cum_s += np.linalg.norm(np.array(pt) - np.array(global_path[idx-1]))
            self._global_path_file.write("{},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                idx, pt[0], pt[1], pt[2], cum_s))
        self._global_path_file.flush()

        # ── Local plans CSV ──────────────────────────────────────
        self._local_plans_file = open(lp_path, "w")
        self._local_plans_file.write(
            "plan_id,request_timestamp_ns,"
            "state_x,state_y,state_z,state_vx,state_vy,state_vz,state_yaw,"
            "local_goal_x,local_goal_y,local_goal_z,"
            "progress_s,progress_index,local_goal_index,"
            "status,success,planning_time_ms,min_clearance,traj_point_count\n")

        # ── State variables ──────────────────────────────────────
        goal_pt = plan["goal"]
        goal_np = np.array(goal_pt)
        global_path_length = plan.get("global_path_length", 0.0)
        cur_pos = np.array(plan["start"], dtype=np.float64)
        cur_vel = np.zeros(3, dtype=np.float64)
        cur_yaw = self._get_current_initial_yaw()

        latest_plan_result = None
        plan_generation_time = 0.0
        previous_progress_s = -1.0
        consecutive_failures = 0
        goal_hold_counter = 0

        sent_frame_id = 0
        ctrl_step = 0
        rec_step = 0
        t_next_ctrl = time.monotonic()
        t_next_rec = t_next_ctrl + dt_rec
        t_next_plan = t_next_ctrl
        recording_start_mono = time.monotonic()
        recording_start_epoch_ns = int(time.time() * 1e9)

        lp_cfg = self.g.get("planning", {}).get("local_planner", {})
        command_lookahead_time = float(
            lp_cfg.get("velocity_command_lookahead_time", 0.12))
        velocity_tracking_gain = float(
            lp_cfg.get("velocity_tracking_gain", 1.5))
        max_velocity = float(lp_cfg.get("max_velocity", 2.5))
        max_acceleration = float(lp_cfg.get("max_acceleration", 3.5))
        max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))
        max_inflight_frames = max(
            1, int(self.g.get("sync", {}).get("max_inflight_frames", 4)))
        unity_response_timeout_s = max(
            0.25, float(self.g.get("sync", {}).get(
                "unity_response_timeout_s", 2.0)))
        max_plan_age = float(lp_cfg.get("max_plan_age", 0.75))
        goal_tolerance = float(lp_cfg.get("goal_tolerance", 0.30))
        goal_speed_tol = float(lp_cfg.get("goal_speed_tolerance", 0.20))
        goal_hold_ticks = int(lp_cfg.get("goal_hold_ticks", 3))
        configured_failure_limit = int(
            lp_cfg.get("max_consecutive_failures", 3))
        failure_grace_time = float(lp_cfg.get("failure_grace_time", 1.0))
        max_consecutive_failures = max(
            configured_failure_limit,
            int(math.ceil(failure_grace_time * self._planner_hz)))
        goal_reached_pending = False
        last_valid_response_mono = time.monotonic()
        planner_worker = (_AsyncPlannerWorker(self._cpp_planner)
                          if self._cpp_planner is not None else None)
        stale_planner_results = 0

        rospy.loginfo("[ONLINE] Starting. ctrl=%.0fHz rec=%.0fHz plan=%.0fHz(async) inflight<=%d",
                      ctrl_hz, rec_hz, self._planner_hz, max_inflight_frames)

        while not rospy.is_shutdown():
            if self._timed_out():
                rospy.logwarn("[ONLINE] Timeout.")
                self._trajectory_exit_reason = "trajectory_timeout"
                break
            now_mono = time.monotonic()

            # Consume completed optimization without blocking control/render.
            completed = (planner_worker.take_completed()
                         if planner_worker is not None else None)
            if completed is not None:
                result = completed.get("result")
                request_mono = completed["request_mono"]
                request_age = max(0.0, now_mono - request_mono)
                state = completed["state"]
                self._total_replans += 1

                if completed.get("exception") is not None:
                    rospy.logerr("[ONLINE] Planner exception: %s",
                                 completed["exception"])
                    self._failed_replans += 1
                    consecutive_failures += 1
                elif result is not None:
                    self._planning_times_ms.append(result.planning_time_ms)
                    self._local_plans_file.write(
                        "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},"
                        "{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{:.2f},{:.4f}\n"
                        .format(result.plan_id, int(request_mono * 1e9),
                                state.position[0], state.position[1], state.position[2],
                                state.velocity[0], state.velocity[1], state.velocity[2],
                                state.yaw,
                                result.local_goal[0], result.local_goal[1], result.local_goal[2],
                                result.progress_s, result.progress_index, result.local_goal_index,
                                int(result.status), result.success,
                                result.planning_time_ms, result.min_clearance,
                                len(result.trajectory)))
                    self._local_plans_file.flush()

                    if result.success and request_age <= max_plan_age:
                        latest_plan_result = result
                        plan_generation_time = request_mono
                        previous_progress_s = max(previous_progress_s,
                                                  result.progress_s)
                        consecutive_failures = 0
                        self._successful_replans += 1
                        self._executed_clearances.append(result.min_clearance)
                    elif result.success:
                        stale_planner_results += 1
                        if stale_planner_results <= 3 or stale_planner_results % 25 == 0:
                            rospy.logwarn(
                                "[ONLINE] Discarding stale plan: age=%.1fms limit=%.1fms",
                                request_age * 1000.0, max_plan_age * 1000.0)
                    else:
                        self._failed_replans += 1
                        consecutive_failures += 1
                        if result.status == _PlannerStatus.COLLISION:
                            self._emergency_hold_count += 1

                if consecutive_failures >= max_consecutive_failures:
                    self._trajectory_exit_reason = "consecutive_planner_failures"
                    rospy.logerr("[ONLINE] %d consecutive planner failures; aborting.",
                                 consecutive_failures)
                    break

            # Submit only the newest state; never queue planner requests.
            if (not goal_reached_pending and now_mono >= t_next_plan and
                    planner_worker is not None):
                state = _VehicleState()
                state.position = (float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2]))
                state.velocity = (float(cur_vel[0]), float(cur_vel[1]), float(cur_vel[2]))
                state.yaw = float(cur_yaw)
                state.yaw_rate = 0.0
                planner_worker.submit({
                    "state": state,
                    "previous_progress_s": previous_progress_s,
                    "request_mono": now_mono,
                })
                t_next_plan = now_mono + 1.0 / self._planner_hz

            # ── PLAN ─────────────────────────────────────────────
            if False and (not goal_reached_pending and now_mono >= t_next_plan and
                self._cpp_planner is not None):
                try:
                    state = _VehicleState()
                    state.position = (float(cur_pos[0]), float(cur_pos[1]), float(cur_pos[2]))
                    state.velocity = (float(cur_vel[0]), float(cur_vel[1]), float(cur_vel[2]))
                    state.yaw = float(cur_yaw)
                    state.yaw_rate = 0.0
                    result = self._cpp_planner.plan_local(state, previous_progress_s)
                    self._total_replans += 1
                    self._planning_times_ms.append(result.planning_time_ms)
                    # Plan age starts when the result becomes executable.  With
                    # the request-start timestamp here, a slow but valid plan
                    # was immediately considered stale and replaced by a hold,
                    # leaving the vehicle frozen at the same state forever.
                    plan_generation_time = time.monotonic()

                    if result.success:
                        latest_plan_result = result
                        current_plan_sample_idx = 0
                        previous_progress_s = result.progress_s
                        consecutive_failures = 0
                        self._successful_replans += 1
                        self._executed_clearances.append(result.min_clearance)
                    else:
                        self._failed_replans += 1
                        consecutive_failures += 1
                        if result.status == _PlannerStatus.COLLISION:
                            self._emergency_hold_count += 1
                        if consecutive_failures >= max_consecutive_failures:
                            self._trajectory_exit_reason = "consecutive_planner_failures"
                            rospy.logerr("[ONLINE] %d consecutive failures — aborting.",
                                         consecutive_failures)
                            break
                        if latest_plan_result is not None:
                            plan_age = now_mono - plan_generation_time
                            if (plan_age <= max_plan_age and
                                current_plan_sample_idx < len(latest_plan_result.trajectory)):
                                pass  # continue with old plan
                            else:
                                latest_plan_result = None

                    self._local_plans_file.write(
                        "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},"
                        "{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{},{},{:.2f},{:.4f}\n"
                        .format(result.plan_id, int(now_mono * 1e9),
                                state.position[0], state.position[1], state.position[2],
                                state.velocity[0], state.velocity[1], state.velocity[2],
                                state.yaw,
                                result.local_goal[0], result.local_goal[1], result.local_goal[2],
                                result.progress_s, result.progress_index, result.local_goal_index,
                                int(result.status), result.success,
                                result.planning_time_ms, result.min_clearance,
                                len(result.trajectory)))
                    self._local_plans_file.flush()
                except Exception as exc:
                    rospy.logerr("[ONLINE] Planner exception: %s", exc)
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self._trajectory_exit_reason = "planner_exception"
                        break

                t_next_plan += 1.0 / self._planner_hz
                if t_next_plan <= now_mono:
                    t_next_plan = now_mono + 1.0 / self._planner_hz

            # ── SEND ─────────────────────────────────────────────
            # Bounded pipeline: allow enough in-flight frames to sustain the
            # requested frame rate across normal Unity render RTT, but stop
            # advancing before an unbounded pose backlog can form.
            if (not goal_reached_pending and now_mono >= t_next_ctrl and
                sync_buffer.size() < max_inflight_frames):
                desired_vel = np.zeros(3, dtype=np.float64)
                command_point = None
                pm = {}
                if (latest_plan_result is not None and
                    len(latest_plan_result.trajectory) > 0):
                    plan_age = now_mono - plan_generation_time
                    if plan_age <= max_plan_age:
                        command_time = max(0.0, plan_age) + command_lookahead_time
                        trajectory = latest_plan_result.trajectory
                        idx = min(range(len(trajectory)),
                                  key=lambda i: abs(
                                      float(trajectory[i].t) - command_time))
                        command_point = trajectory[idx]
                        reference_pos = np.array(command_point.position,
                                                 dtype=np.float64)
                        feedforward_vel = np.array(command_point.velocity,
                                                   dtype=np.float64)
                        desired_vel = (feedforward_vel + velocity_tracking_gain *
                                       (reference_pos - cur_pos))
                        pm = {
                            "local_goal_world": list(latest_plan_result.local_goal),
                            "global_progress_s": latest_plan_result.progress_s,
                            "global_progress_ratio": (
                                latest_plan_result.progress_s / max(global_path_length, 1e-6)),
                            "global_progress_index": latest_plan_result.progress_index,
                            "local_goal_index": latest_plan_result.local_goal_index,
                            "plan_id": latest_plan_result.plan_id,
                            "plan_sample_time_s": command_point.t,
                            "plan_age_ms": plan_age * 1000.0,
                            "planner_status": str(latest_plan_result.status),
                            "planner_success": latest_plan_result.success,
                            "planner_compute_ms": latest_plan_result.planning_time_ms,
                            "planner_min_clearance": latest_plan_result.min_clearance,
                            "distance_to_final_goal": float(
                                np.linalg.norm(cur_pos - goal_np)),
                            "state_source": "velocity_integrated",
                        }
                    else:
                        latest_plan_result = None

                if command_point is None:
                    pm = {"local_goal_world": cur_pos.tolist(),
                          "global_progress_s": previous_progress_s,
                          "global_progress_ratio": 0.0, "global_progress_index": -1,
                          "local_goal_index": -1, "plan_id": -1,
                          "plan_sample_time_s": 0.0, "plan_age_ms": -1.0,
                          "planner_status": "NO_PLAN", "planner_success": False,
                          "planner_compute_ms": 0.0, "planner_min_clearance": 0.0,
                          "distance_to_final_goal": float(np.linalg.norm(cur_pos - goal_np)),
                          "state_source": "velocity_integrated"}

                # Bound speed near the final goal so repeated local plans
                # cannot continually command cruise speed through it.
                dist_before = float(np.linalg.norm(cur_pos - goal_np))
                desired_speed = float(np.linalg.norm(desired_vel))
                stopping_distance = max(
                    0.0, dist_before - 0.5 * goal_tolerance)
                stopping_speed = math.sqrt(
                    2.0 * max_acceleration * stopping_distance)
                if desired_speed > stopping_speed:
                    desired_vel *= stopping_speed / max(desired_speed, 1e-9)

                cur_pos, cur_vel, cur_yaw, yr = integrate_velocity_command(
                    cur_pos, cur_vel, cur_yaw, desired_vel, dt_ctrl,
                    max_velocity, max_acceleration, max_yaw_rate)
                pos_wp = cur_pos.tolist()
                yaw = float(cur_yaw)
                vx_w, vy_w, vz_w = [float(v) for v in cur_vel]
                # Continuous execution time; the selected plan-local sample
                # time is stored separately in plan_time_from_start_s.
                traj_time = max(0.0, now_mono - recording_start_mono)

                vehicle = make_depth_vehicle(pos_wp, yaw, self._depth_cfg)
                msg = {"scene_id": self.g["scene_id"], "frame_id": sent_frame_id,
                       "vehicles": [vehicle], "objects": self.current_obj_list}
                self.bridge.send_pose(msg)

                cvx_b, cvy_b, cvz_b = world_vel_to_body(vx_w, vy_w, vz_w, yaw)
                t_sent_mono = time.monotonic()
                half = 0.5 * yaw
                sync_buffer.push({
                    "frame_id": sent_frame_id, "t_sent": t_sent_mono,
                    "t_sent_epoch_ns": int(time.time() * 1e9),
                    "pos": pos_wp,
                    "quat": [0.0, 0.0, math.sin(half), math.cos(half)],
                    "vel_body": [cvx_b, cvy_b, cvz_b],
                    "vel_world": [vx_w, vy_w, vz_w],
                    "ctrl_v_body": [cvx_b, cvy_b, cvz_b],
                    "ctrl_yr": yr, "yaw": yaw,
                    "vel_source": "velocity_controller_integrated",
                    "trajectory_time_s": traj_time,
                    "local_goal_world": pm.get("local_goal_world", [0,0,0]),
                    "global_progress_s": pm.get("global_progress_s", 0.0),
                    "global_progress_ratio": pm.get("global_progress_ratio", 0.0),
                    "global_progress_index": pm.get("global_progress_index", -1),
                    "local_goal_index": pm.get("local_goal_index", -1),
                    "plan_id": pm.get("plan_id", -1),
                    "plan_sample_time_s": pm.get("plan_sample_time_s", 0.0),
                    "plan_age_ms": pm.get("plan_age_ms", -1.0),
                    "planner_status": pm.get("planner_status", "UNKNOWN"),
                    "planner_success": pm.get("planner_success", False),
                    "planner_compute_ms": pm.get("planner_compute_ms", 0.0),
                    "planner_min_clearance": pm.get("planner_min_clearance", 0.0),
                    "distance_to_final_goal": pm.get("distance_to_final_goal", 0.0),
                    "state_source": pm.get("state_source", "velocity_integrated"),
                })
                ctrl_step += 1
                sent_frame_id += 1
                t_next_ctrl += dt_ctrl
                if t_next_ctrl <= now_mono:
                    t_next_ctrl = now_mono + dt_ctrl
                self._rec_sent_control_frames += 1

                dist_to_goal = float(np.linalg.norm(cur_pos - goal_np))
                speed = float(np.linalg.norm(cur_vel))
                if dist_to_goal <= goal_tolerance and speed <= goal_speed_tol:
                    goal_hold_counter += 1
                    if goal_hold_counter >= goal_hold_ticks:
                        rospy.loginfo("[ONLINE] Goal reached.")
                        self._trajectory_reached_goal = True
                        self._trajectory_exit_reason = "goal_reached"
                        goal_reached_pending = True
                else:
                    goal_hold_counter = 0
                if previous_progress_s > 0 and previous_progress_s >= global_path_length * 0.99:
                    if dist_to_goal <= goal_tolerance and speed <= goal_speed_tol:
                        self._trajectory_reached_goal = True
                        self._trajectory_exit_reason = "goal_reached_progress"
                        goal_reached_pending = True

            # ── RECEIVE ──────────────────────────────────────────
            if now_mono >= t_next_rec:
                # Consume at most one response per record tick.  Draining the
                # socket and retaining only the newest response discards valid
                # intermediate frame_ids and leaves their control states
                # permanently unmatched when multiple frames are in flight.
                latest_r = self.bridge.try_recv()
                drain_count = 1 if latest_r is not None else 0
                self._rec_raw_received_frames += drain_count
                self._rec_discarded_extra_frames += max(0, drain_count - 1)

                depth_u16 = None; collision = 0; recv_frame_id = None
                if latest_r is not None:
                    msg_dict, img_parts = latest_r
                    _fid = msg_dict.get("pub_frame_id")
                    if _fid is None: _fid = msg_dict.get("frame_id")
                    recv_frame_id = _fid
                    vehicles = msg_dict.get("pub_vehicles", [])
                    if vehicles and vehicles[0].get("collision", False): collision = 1
                    for part in img_parts:
                        if len(part) >= depth_float_len:
                            raw = part[:depth_float_len]
                            df32 = np.frombuffer(raw, dtype=np.float32).reshape((img_h, img_w))
                            dm = np.flipud(df32 * 100.0)
                            dm = np.nan_to_num(dm, nan=depth_max_m, posinf=depth_max_m, neginf=0)
                            depth_u16 = np.clip(dm / max(1e-6, depth_max_m) * 65535,
                                                0, 65535).astype(np.uint16)
                            break

                recv_time_mono = time.monotonic()
                match_entry = None; latency_ms = -1.0; match_error_ms = -1.0
                match_method = "none"
                if recv_frame_id is not None:
                    match_entry = sync_buffer.match_and_remove(recv_frame_id)
                    if match_entry is not None:
                        match_method = "frame_id"; match_error_ms = 0.0
                        latency_ms = (recv_time_mono - match_entry["t_sent"]) * 1000.0
                        self._rec_exact_matches += 1
                        last_valid_response_mono = recv_time_mono
                    else:
                        # An explicit but unknown frame_id is normally a stale
                        # initialization/reset reply. Do not attach that image
                        # to the newest command: retaining pending states lets
                        # the next correctly identified Unity frame catch up.
                        match_method = "stale_frame_id"
                        self._rec_unmatched_frames += 1
                if (match_entry is None and recv_frame_id is None and
                    drain_count > 0):
                    match_entry = sync_buffer.drain_to_latest()
                    if match_entry is not None:
                        match_method = "drain_to_latest"
                        latency_ms = (recv_time_mono - match_entry["t_sent"]) * 1000.0
                        match_error_ms = latency_ms
                        self._rec_fallback_matches += 1
                        last_valid_response_mono = recv_time_mono

                png_name = "none"
                if (match_entry is not None and depth_u16 is not None and
                    Image is not None):
                    png_name = "{:06d}.png".format(rec_step)
                    Image.fromarray(depth_u16, mode="I;16").save(os.path.join(depth_dir, png_name))

                if match_entry is not None:
                    matched_t_sent = match_entry["t_sent"]
                    ts = recording_start_epoch_ns + int((matched_t_sent - recording_start_mono) * 1e9)
                    p = match_entry["pos"]; q = match_entry["quat"]
                    v = match_entry["vel_body"]; cv = match_entry["ctrl_v_body"]
                    cyr = match_entry["ctrl_yr"]; vs = match_entry.get("vel_source", "unknown")
                    matched_fid = match_entry.get("frame_id", -1)
                    traj_t = match_entry.get("trajectory_time_s", 0.0)
                    ctrl_fid = match_entry.get("frame_id", -1)
                    send_epoch = match_entry.get("t_sent_epoch_ns", 0)
                    lg_w = match_entry.get("local_goal_world", [0,0,0])
                    lgx_b, lgy_b, lgz_b = world_vel_to_body(
                        lg_w[0]-p[0], lg_w[1]-p[1], lg_w[2]-p[2], match_entry.get("yaw", 0.0))
                    self._inprogress_file.write(
                        "{:d},{:.4f},{:.4f},{:.4f},{:.6f},{:.6f},{:.6f},{:.6f},"
                        "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.6f},"
                        "{},{:d},"
                        "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},"
                        "{:d},{:.2f},{:.2f},{:d},{},"
                        "{:.3f},{:d},{:d},"
                        "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},"
                        "{:.4f},{:.6f},{:d},{:d},{:d},{:.3f},{:.2f},"
                        "{},{},{:.2f},{:.4f},{:.4f},{}\n"
                        .format(ts, p[0], p[1], p[2], q[0], q[1], q[2], q[3],
                                v[0], v[1], v[2], cv[0], cv[1], cv[2], cyr,
                                png_name, collision,
                                plan["start"][0], plan["start"][1], plan["start"][2],
                                goal_pt[0], goal_pt[1], goal_pt[2],
                                6, latency_ms, match_error_ms, matched_fid, vs,
                                traj_t, ctrl_fid, send_epoch,
                                lgx_b, lgy_b, lgz_b, lg_w[0], lg_w[1], lg_w[2],
                                match_entry.get("global_progress_s", 0.0),
                                match_entry.get("global_progress_ratio", 0.0),
                                match_entry.get("global_progress_index", -1),
                                match_entry.get("local_goal_index", -1),
                                match_entry.get("plan_id", -1),
                                match_entry.get("plan_sample_time_s", 0.0),
                                match_entry.get("plan_age_ms", -1.0),
                                match_entry.get("planner_status", "UNKNOWN"),
                                match_entry.get("planner_success", False),
                                match_entry.get("planner_compute_ms", 0.0),
                                match_entry.get("planner_min_clearance", 0.0),
                                match_entry.get("distance_to_final_goal", 0.0),
                                match_entry.get("state_source", "unknown")))
                    self._rec_written_rows += 1

                matched_fid_out = match_entry.get("frame_id", -1) if match_entry else -1
                self._sync_file.write("{},{},{:.2f},{:.2f},{},{},{},{},{}\n".format(
                    rec_step, matched_fid_out, latency_ms, match_error_ms, match_method,
                    1 if (drain_count == 0 and latest_r is None) else 0,
                    sync_buffer.size(), self._rec_exact_matches, self._rec_fallback_matches))
                rec_step += 1
                t_next_rec += dt_rec
                if t_next_rec <= now_mono:
                    t_next_rec = now_mono + dt_rec

            # Do not close files until the final goal frame has returned and
            # been written to data.csv.
            if goal_reached_pending and sync_buffer.size() == 0:
                break

            if (sync_buffer.size() >= max_inflight_frames and
                time.monotonic() - last_valid_response_mono >
                    unity_response_timeout_s):
                self._trajectory_exit_reason = "unity_response_timeout"
                rospy.logerr(
                    "[ONLINE] Unity response timeout: %.2fs with %d pending frames.",
                    time.monotonic() - last_valid_response_mono,
                    sync_buffer.size())
                break

            time.sleep(min(dt_ctrl, dt_rec) * 0.05)

        if planner_worker is not None:
            planner_worker.stop()

        self._final_executed_position = cur_pos.tolist()
        self._final_executed_velocity = cur_vel.tolist()
        if self._trajectory_exit_reason == "running":
            self._trajectory_exit_reason = "shutdown" if rospy.is_shutdown() else "loop_exited"
        self._close_open_files()
        self._st_finish_recording()
    def _st_finish_recording(self):
        """v7: Called after online planning loop exits to finalize recording."""
        plan = self.current_planned[self.traj_idx]
        ctrl_hz = self.g["control"]["control_hz"]
        rec_hz = self.g["control"]["record_hz"]
        data_cfg = self.g.get("data", {})
        schema_version = int(data_cfg.get("schema_version", 7))

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
        collection_mode = data_cfg.get("collection_mode", "deterministic_lockstep")
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
            # ── Schema v8 metadata ──
            "schema_version": schema_version,
            "collection_mode": collection_mode,
            "sample_semantics": "depth_t,state_t,navigation_t -> expert_command_t",
            "label_lookahead_time_s": float(
                data_cfg.get("label_lookahead_time_s", 0.08)),
            "training_coordinate_frame": {
                "name": "CAMERA_FLU",
                "x": "forward",
                "y": "left",
                "z": "up",
            },
            "world_coordinate_frame": "ROS_WORLD_FLU",
            "legacy_internal_body_frame": {
                "name": "RFU",
                "x": "right",
                "y": "forward",
                "z": "up",
            },
            "guide_source": ("farthest_visible_astar_waypoint"
                             if self._guide_selector is not None
                             else "legacy_local_planner_goal"),
            "guide_selection_rule": ("maximum_forward_astar_path_index_satisfying_"
                                      "range_fov_depth_visibility_and_known_free_corridor"),
            "trajectory_terminal_rule": ("farthest_dynamically_reachable_path_point_"
                                          "not_beyond_guide"),
            "unknown_space_policy": "occupied_or_infeasible",
            "global_map_fallback_enabled": False,
            "trend_horizontal_bins": int(
                data_cfg.get("trend_horizontal_bins", 11)),
            "trend_vertical_bins": int(
                data_cfg.get("trend_vertical_bins", 7)),
            "trend_soft_sigma_bins": float(
                data_cfg.get("trend_soft_sigma_bins", 0.75)),
            "max_guide_range_m": float(self._depth_cfg["max_m"]),
            "depth_all_valid_in_simulation": True,
            # ── Phase 2: observed map metadata ──
            "local_expert_map": "observed_depth_history_esdf",
            "global_map_usage": [
                "weighted_astar",
                "task_feasibility",
                "offline_audit",
            ],
            "global_map_used_for_local_collision_optimization": False,
            # ── v8 label quality stats ──
            "invalid_expert_label_count": getattr(
                self, "_invalid_expert_label_count", 0),
            "invalid_trend_label_count": getattr(
                self, "_invalid_trend_label_count", 0),
            "consecutive_guide_failures": getattr(
                self, "_consecutive_guide_failures", 0),
            # ── Phase 3: scene & task metadata ──
            "scene_generation_enabled": self._use_scene_gen,
            "scene_obstacle_count": len(getattr(self, "_current_scene_obstacles", [])),
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
            "task_direct_path_blocked": (
                self._current_task_validation.direct_path_blocked
                if self._current_task_validation is not None else None),
            "task_direct_blocker_count": (
                self._current_task_validation.direct_blocker_count
                if self._current_task_validation is not None else 0),
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
            # ── v5 planner metadata (retained) ──
            "planner_type": "receding_horizon_local",
            "planner_backend": self._planner_backend,
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
            "ESDF_clearance_semantics": (
                "ESDF values already have drone_radius subtracted by ESDFBuilder. "
                "Local planner min_clearance is additional safety margin on top."
            ),
            "status": "inprogress",
        }
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

        # 7. Metadata completeness
        meta_path = os.path.join(self._inprogress_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            validation_passed = False
            failure_reasons.append("metadata_missing")

        # 8. Schema v7 lightweight data checks (non-fatal for some)
        data_cfg = self.g.get("data", {})
        schema_version = int(data_cfg.get("schema_version", 7))
        if schema_version >= 7 and os.path.isfile(data_path) and col_map:
            v7_issues = self._validate_schema_v7(
                data_path, col_map, data_cfg)
            if v7_issues:
                rospy.logwarn("[Validate] Schema v7 issues: %s",
                              "; ".join(v7_issues[:10]))
                # Do not reject episode for trend label issues in stage 1,
                # but track them in metadata.
                # Only reject for critical structural issues.
                critical = [i for i in v7_issues if i.startswith("CRITICAL:")]
                if critical:
                    failure_reasons.extend(critical)
                    validation_passed = False

        # ── Commit or reject ──────────────────────────────────────
        if validation_passed:
            self._commit_trajectory()
        else:
            self._reject_trajectory(failure_reasons)

        self._inprogress_dir = None
        self._final_dir = None
        self._route_next()

    def _validate_schema_v7(self, data_path, col_map, data_cfg):
        """Lightweight schema v7 data checks. Returns list of issue strings."""
        issues = []
        trend_h_bins = int(data_cfg.get("trend_horizontal_bins", 11))
        trend_v_bins = int(data_cfg.get("trend_vertical_bins", 7))

        try:
            with open(data_path, "r") as f:
                reader = csv.DictReader(f)
                prev_frame_id = -1
                row_idx = 0
                for row in reader:
                    row_idx += 1

                    # Check required v7 fields exist
                    if "expert_label_valid" not in row:
                        issues.append("CRITICAL: missing expert_label_valid at row {}".format(row_idx))
                    if "trend_label_valid" not in row:
                        issues.append("CRITICAL: missing trend_label_valid at row {}".format(row_idx))
                    if "match_method" not in row:
                        issues.append("CRITICAL: missing match_method at row {}".format(row_idx))

                    # match_method must be exact frame_id
                    mm = row.get("match_method", "")
                    if mm and mm != "frame_id_exact":
                        issues.append("CRITICAL: non-exact match_method='{}' at row {}".format(mm, row_idx))

                    # frame_id monotonically increasing
                    try:
                        fid = int(row.get("frame_id", -1))
                        if fid <= prev_frame_id:
                            issues.append("CRITICAL: non-monotonic frame_id {} -> {} at row {}".format(
                                prev_frame_id, fid, row_idx))
                        prev_frame_id = fid
                    except (ValueError, TypeError):
                        pass

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

                    # Valid global direction should have norm ~1
                    gdv = row.get("global_direction_valid")
                    if gdv is not None and str(gdv) == "1":
                        try:
                            gdx = float(row.get("global_dir_x_flu", 0))
                            gdy = float(row.get("global_dir_y_flu", 0))
                            gdz = float(row.get("global_dir_z_flu", 0))
                            norm = math.sqrt(gdx*gdx + gdy*gdy + gdz*gdz)
                            if abs(norm - 1.0) > 0.01:
                                issues.append("global_dir norm={:.4f} != 1 at row {}".format(norm, row_idx))
                        except (ValueError, TypeError):
                            pass

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
                        h_sum = 0.0
                        for i in range(trend_h_bins):
                            key = "guide_azimuth_soft_{}".format(i)
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
                            issues.append("elevation soft sum={:.4f} != 1 at row {}".format(v_sum, row_idx))

                    # Invalid trend: bin should be -1, soft labels all zero
                    if tlv is not None and str(tlv) == "0":
                        try:
                            ab = int(row.get("guide_azimuth_bin", -1))
                            eb = int(row.get("guide_elevation_bin", -1))
                            if ab != -1:
                                issues.append("invalid trend azimuth_bin={} != -1 at row {}".format(ab, row_idx))
                            if eb != -1:
                                issues.append("invalid trend elevation_bin={} != -1 at row {}".format(eb, row_idx))

                            for i in range(trend_h_bins):
                                key = "guide_azimuth_soft_{}".format(i)
                                try:
                                    v = float(row.get(key, 0))
                                    if abs(v) > 1e-9:
                                        issues.append("invalid trend azimuth_soft non-zero={} at row {}".format(v, row_idx))
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
            issues.append("CRITICAL: schema_v7_validation_exception: {}".format(exc))

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
        failed_dest = os.path.join(self._failed_dir,
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
        """Route to next trajectory, next config, or done."""
        if self.traj_idx >= len(self.current_planned):
            self.seed_idx += 1
            self._enter_state(State.NEXT_CONFIG)
        else:
            self._enter_state(State.RESET_DRONE)

    def _st_next_config(self):
        self._enter_state(State.GENERATE_OBSTACLE_CONFIG)

    # ═══════════════════════════════════════════════════════════════
    #  Backward-compatible state aliases  (no-op, route to main path)
    # ═══════════════════════════════════════════════════════════════

    def _st_next_trajectory(self):
        rospy.logwarn("[FSM] NEXT_TRAJECTORY is deprecated – routing to VALIDATE_AND_COMMIT")
        self._enter_state(State.VALIDATE_AND_COMMIT)


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
        for s in cfg["scenes"]:
            n_seeds = len(s.get("seeds", [0]))
            n_pairs = g["start_goal"]["num_pairs_per_config"]
            rospy.loginfo("  '%s': %d seeds × %d pairs  (r=%.2f–%.2f m, occ=%.2f)",
                          s["name"], n_seeds, n_pairs,
                          s["radius_min"], s["radius_max"], s["target_occupancy"])
        rospy.loginfo("=" * 60)
        return

    cfg = load_config()
    mgr = ILManager(cfg)
    mgr.run()


if __name__ == "__main__":
    main()
