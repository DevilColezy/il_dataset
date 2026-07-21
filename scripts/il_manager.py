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
)
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

        if not global_path:
            rospy.logerr("[FSM] No global path for trajectory %d", self.traj_idx + 1)
            self.traj_idx += 1
            self._enter_state(State.RESET_DRONE if self.traj_idx < len(self.current_planned)
                              else State.VALIDATE_AND_COMMIT)
            return

        if self._cpp_planner is not None:
            try:
                # Set ESDF (copied once into C++)
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
        """v5: Set up recording files for online planning mode."""
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

        # v5: Reset planner stats
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

    # ═══════════════════════════════════════════════════════════════
    #  v5: ONLINE_PLAN_AND_RECORD — receding-horizon control loop
    # ═══════════════════════════════════════════════════════════════

    def _st_online_plan_and_record(self):
        """v5: Online receding-horizon planning and data recording.

        Control loop at control_hz with C++ local planner at planner_hz.
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
        """v5: Called after online planning loop exits to finalize recording."""
        plan = self.current_planned[self.traj_idx]
        ctrl_hz = self.g["control"]["control_hz"]
        rec_hz = self.g["control"]["record_hz"]

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

        # Save metadata to .inprogress (v5)
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
            "schema_version": 6,
            "vel_source": "velocity_controller_integrated",
            "state_source": "velocity_integrated",
            "body_frame_convention": {"x": "right", "y": "forward", "z": "up"},
            # v5 planner metadata
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

        # ── Commit or reject ──────────────────────────────────────
        if validation_passed:
            self._commit_trajectory()
        else:
            self._reject_trajectory(failure_reasons)

        self._inprogress_dir = None
        self._final_dir = None
        self._route_next()

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
