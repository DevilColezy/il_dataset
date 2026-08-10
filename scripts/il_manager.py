#!/usr/bin/env python3
"""
il_manager.py  —  IL dataset collection manager (two-level navigation
expert, schema v24).

Automatic unattended lifecycle (section I/XX/XXIV-XXVIII):
    connect AvoidBench (10253/10254)
      -> procedurally generate Scene (cylinders, verified Object_t)
      -> send scene to Unity, wait ready
      -> export ONE scene-specific point cloud
      -> build the privileged SCENE map (C++)
      -> generate many start-goal tasks (C++ batch evaluation)
      -> per task: reset drone + expert, run the 5/30 Hz lockstep loop,
         record episode, write task manifest, NEXT_TASK
      -> scene complete -> NEXT_SCENE -> NEXT_PROFILE -> DONE
No external task manifests, manual start/goal or per-scene interaction is
required (section LXXVIII).

Inside an episode the existing two-level loop is unchanged:
    30 Hz: state + depth -> observed map (C++) -> 5 Hz macro expert ->
    30 Hz local planner (A* + B-spline) -> execution safety -> FLU
    controller -> dataset recording (labels match the executed command).

The manager is ROS entry `il_dataset_manager`.  Heavy loops run in C++
(`_il_local_planner`); Python owns lifecycle, scheduling and recording.
"""

from __future__ import print_function, division

import json
import math
import os
import shutil
import sys
import threading
import time
from enum import Enum

# The compiled pybind module (_il_local_planner.so) is built into the
# source scripts/ directory (CMake LIBRARY_OUTPUT_DIRECTORY = scripts/),
# together with the sibling il_* modules.  The manager can be launched
# directly from the source tree OR through the catkin-installed
# executable (devel/lib/il_dataset/il_manager.py, catkin_install_python),
# so add the script's own directory AND the rospack-resolved source
# scripts/ directory to sys.path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
try:
    import rospkg
    _PKG_SCRIPTS_DIR = os.path.join(
        rospkg.RosPack().get_path("il_dataset"), "scripts")
    if os.path.isdir(_PKG_SCRIPTS_DIR) and \
            _PKG_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, _PKG_SCRIPTS_DIR)
except Exception:  # noqa: BLE001 — rospkg optional outside a ROS shell
    pass

import numpy as np
import rospy

import _il_local_planner as module

from il_common import (
    UnityBridge,
    make_depth_vehicle,
    world_vector_to_body_flu_quat,
    body_flu_vector_to_world_quat,
    normalize_angle,
    quantize_bounded_vector,
    load_ply,
)
from il_config import (
    load_config,
    build_observed_map_config,
    build_goal_capture_config,
    build_oracle_config,
    build_macro_candidate_config,
    build_recoverability_config,
    build_planner_config,
    build_intervention_config,
    build_task_generation_config,
)
from il_dynamics import create_dynamics_backend
from il_macro_expert import MacroExpert
from il_dataset_writer import DatasetWriter
from il_scene_generator import (
    ProceduralSceneGenerator,
    SceneGeometryValidator,
    SceneProfile,
)
from il_task_generator import (
    MultiscaleTaskGenerator,
    DatasetQuota,
    runtime_classify,
)
from il_generation_manifest import (
    SceneManifestWriter,
    TaskManifestWriter,
    GenerationFailureWriter,
)


class State(Enum):
    BOOT = 0
    WAIT_UNITY = 1
    GENERATE_SCENE = 2
    SEND_SCENE = 3
    SETTLE_SCENE = 4
    EXPORT_POINTCLOUD = 5
    WAIT_POINTCLOUD = 6
    BUILD_PRIVILEGED_MAP = 7
    GENERATE_TASKS = 8
    RESET_DRONE = 9
    RUN_TASK = 10
    FINISH_TASK = 11
    FINISH_SCENE = 12
    NEXT_SCENE = 13
    NEXT_PROFILE = 14
    DONE = 15
    ERROR = 16


class TaskOutcome(Enum):
    """Final quota lifecycle outcome of one scheduled task (sections
    XI/XXXIV).  Every task resolves to exactly one of these."""
    COMMITTED = 0
    FAILED = 1
    CANCELLED = 2


class ILManager(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.g = cfg["global"]
        self._depth_cfg = self.g.get("depth", {})
        self._record_hz = float(self.g.get("control", {}).get("record_hz", 30.0))
        self._dt = 1.0 / self._record_hz
        self._macro_hz = float(self.g.get("macro_expert", {}).get("update_hz", 5.0))
        self._macro_interval = max(1, int(round(self._record_hz / self._macro_hz)))
        self._fsm = self.g.get("fsm", {})
        self._sync_cfg = self.g.get("sync", {})
        self._controller_cfg = self.g.get("trajectory_controller", {})
        self._safety_cfg = self.g.get("execution_safety", {})
        # Round 5: the UNIFIED navigation clearance and its speed-dependent
        # dynamic margin are computed by the C++ LocalPlanner — Python calls
        # `_local_planner.effective_clearance_for(state)` (e.g. the braking
        # check and cached-suffix safety) instead of re-deriving the hard
        # boundary in Python. Fresh plans add their C++ quality margin.
        # Scene settle + point-cloud completion config (sections VIII/XXIV).
        self._scene_runtime = self.g.get("scene_runtime", {})
        _pc_cfg = self.g.get("pointcloud", {})
        self._pc_stable_window_s = float(_pc_cfg.get("stable_window_s", 0.5))
        self._pc_min_file_bytes = int(_pc_cfg.get("min_file_bytes", 1024))
        self._pc_max_retries = int(_pc_cfg.get("max_retries", 3))

        # Outputs
        self.output_root = self.g.get("output_dir") or "dataset/il_data"
        if not os.path.isdir(self.output_root):
            os.makedirs(self.output_root)
        self._inprogress_root = os.path.join(self.output_root, "_inprogress")
        if not os.path.isdir(self._inprogress_root):
            os.makedirs(self._inprogress_root)

        # Bridge
        self._bridge = UnityBridge(self.g["pub_port"], self.g["sub_port"])
        self._frame_latencies = []
        self._frame_finite_ratios = []

        # C++ modules
        self._observed_map = module.ObservedMap(build_observed_map_config(self.g, module))
        self._local_planner = module.LocalPlanner(build_planner_config(self.g, module))
        self._local_planner.set_map(self._observed_map)
        self._goal_capture_controller = module.GoalCaptureController(
            build_goal_capture_config(self.g, module))
        self._recoverability = module.LocalRecoverability(
            build_recoverability_config(self.g, module))
        self._candidate_config = build_macro_candidate_config(self.g, module)
        self._candidate_search = module.MacroCandidateSearch(self._candidate_config)
        self._oracle = module.PrivilegedOracle()
        self._oracle_config = build_oracle_config(self.g, module)
        self._intervention_config = build_intervention_config(self.g, module)
        self._intervention_oracle = module.PrivilegedInterventionOracle(
            self._intervention_config)

        # Macro expert
        macro_cfg = dict(self.g.get("macro_expert", {}))
        macro_cfg.update(self.g.get("macro_candidates", {}))
        goal_capture_cfg = self._controller_cfg.get("goal_capture", {})
        macro_cfg["goal_approach_deceleration_mps2"] = float(
            goal_capture_cfg.get("approach_deceleration_mps2", 2.5))
        self._macro_expert = MacroExpert(
            macro_cfg, module, self._recoverability,
            self._candidate_search, self._candidate_config)

        # Dynamics
        self._dynamics = create_dynamics_backend(cfg)

        # FSM
        self.state = State.BOOT
        self._state_start = 0.0
        self._error = ""

        # ── Scene / task generation (sections XXII-XXVIII) ───────────
        sg = self.g.get("scene_generation", {})
        veh = self.g.get("vehicle", {})
        self._scene_generator = ProceduralSceneGenerator(
            sg, float(veh.get("radius_m", 0.30)),
            float(veh.get("safety_margin_m", 0.20)))
        self._scene_validator = SceneGeometryValidator(
            sg, float(veh.get("radius_m", 0.30)),
            float(veh.get("safety_margin_m", 0.20)))
        self._profiles = [SceneProfile(p) for p in sg.get("profiles", [])]
        self._profile_index = 0
        self._scene_index = 0
        self._scene_attempt = 0
        self._max_scene_attempts = int(sg.get("max_generation_attempts", 24))
        self._tasks_per_scene = int(sg.get("tasks_per_scene", 12))
        self._min_tasks_per_scene = int(
            sg.get("minimum_tasks_per_scene", 1))
        self._current_scene = None
        self._current_tasks = []
        self._current_task = None
        self._current_task_id = None
        self._current_task_quota_resolved = False
        self._task_index = 0
        self._episode_index = 0
        self._scene_dir = None
        self._pc_path = None
        self._pc_retries = 0
        self._pc_request_time = 0.0
        self._pc_ack_received = False
        self._pc_save_success_received = False
        self._pc_last_keepalive = 0.0
        self._last_ply_size = -1
        self._ply_stable_since = None
        self._settle_last_keepalive = 0.0
        self._scene_pose_msg = None
        self._keepalive_thread = None
        self._generation_rng = None

        tg = self.g.get("task_generation", {})
        self._task_gen_oracle = module.TaskGenerationOracle(
            build_task_generation_config(self.g, module))
        self._task_generator = MultiscaleTaskGenerator(
            {
                "flight_height_m": tg.get("flight_height_m", 5.0),
                "region_min": tg.get("region_min", [1.5, 16.0, 3.5]),
                "region_max": tg.get("region_max", [28.5, 60.0, 11.5]),
                "distance_bands": tg.get("distance_bands", {}),
                "sampling": tg.get("sampling", {}),
                "classification": tg.get("classification", {}),
                "initial_yaw": tg.get("initial_yaw", {}),
                "task_seed_base": tg.get("task_seed_base", 999983),
            },
            module, self.g.get("task_oracle", {}), self._task_gen_oracle)
        self._quota = DatasetQuota(tg.get("class_weights", {}))
        self._runtime_classify_enabled = bool(
            (tg.get("runtime_classification", {}) or {}).get(
                "enabled", True))
        # AvoidBench wire identifier (numeric, from config) vs dataset
        # scene key (string, procedural) are kept strictly separate
        # (sections VII-IX): `_unity_scene_id` is the only value ever sent
        # as AvoidBench "scene_id"; `_dataset_scene_key` only names
        # dataset dirs / manifests / PLY files.
        self._unity_scene_id = int(self.g.get("scene_id", 1))
        self._dataset_scene_key = None

        # Stats
        self._total_episodes = 0
        self._committed_episodes = 0
        self._total_scenes = 0
        self._failed_tasks = 0
        self._generation_failures_path = os.path.join(
            self.output_root, "generation_failures.jsonl")

    # ═════════════════════════════════════════════════════════════════
    #  FSM
    # ═════════════════════════════════════════════════════════════════
    def _cleanup_output(self):
        """Clear the whole output root before a fresh collection run.

        A previous crashed run can leave stale `_inprogress/` episode dirs,
        cached `maps/*.ply`, `scenes/`, `_failed/` and `_debug/` entries.
        If they are not removed, the next run re-creates an identical
        episode id (e.g. `scene_..._task_004_ep04`) and DatasetWriter's
        `os.makedirs()` raises FileExistsError.  This restores the
        pre-reshape behavior of clearing the selected output folder at
        startup (every run produces a fresh dataset).
        """
        if not os.path.isdir(self.output_root):
            os.makedirs(self.output_root)
            return
        removed = 0
        for entry in os.listdir(self.output_root):
            entry_path = os.path.join(self.output_root, entry)
            try:
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.unlink(entry_path)
                elif os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                removed += 1
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn("[Manager] cleanup failed for %s: %s",
                              entry_path, exc)
        rospy.loginfo("[Manager] Output dir cleaned: %s (%d entries)",
                      self.output_root, removed)
        # Recreate the subdirs the FSM relies on.
        for sub in ("_inprogress", "maps", "scenes", "_failed", "_debug"):
            sub_path = os.path.join(self.output_root, sub)
            if not os.path.isdir(sub_path):
                os.makedirs(sub_path)

    def run(self):
        # Fresh collection run: clear the whole output root first (stale
        # _inprogress episodes / maps / scenes), exactly like the
        # pre-reshape manager did.
        self._cleanup_output()
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            self._tick()
            if self.state == State.DONE:
                # Normal completion: no task may still be pending (section
                # XXXVII).  Any leftover is a lifecycle bug — cancel it so
                # the final summary is honest.
                leftover = self._quota.cancel_all_pending()
                if leftover:
                    rospy.logerr(
                        "[Manager] DONE with %d unresolved pending tasks "
                        "(auto-cancelled)", leftover)
                rospy.loginfo(
                    "[Manager] Collection complete. scenes=%d episodes=%d "
                    "committed=%d quota=%s", self._total_scenes,
                    self._total_episodes, self._committed_episodes,
                    self._quota.summary())
                return
            if self.state == State.ERROR:
                # Release every leftover pending on a fatal abort (section
                # XXXVIII): current/future tasks cannot run.
                leftover = self._quota.cancel_all_pending()
                rospy.logerr(
                    "[Manager] ERROR: %s (auto-cancelled %d pending tasks)",
                    self._error, leftover)
                return
            rate.sleep()

    def _enter_state(self, state):
        self.state = state
        self._state_start = time.monotonic()

    def _timed_out(self, timeout_s):
        return time.monotonic() - self._state_start > timeout_s

    def _tick(self):
        handlers = {
            State.BOOT: self._st_boot,
            State.WAIT_UNITY: self._st_wait_unity,
            State.GENERATE_SCENE: self._st_generate_scene,
            State.SEND_SCENE: self._st_send_scene,
            State.SETTLE_SCENE: self._st_settle_scene,
            State.EXPORT_POINTCLOUD: self._st_export_pointcloud,
            State.WAIT_POINTCLOUD: self._st_wait_pointcloud,
            State.BUILD_PRIVILEGED_MAP: self._st_build_privileged_map,
            State.GENERATE_TASKS: self._st_generate_tasks,
            State.RESET_DRONE: self._st_reset_drone,
            State.RUN_TASK: self._st_run_task,
            State.FINISH_TASK: self._st_finish_task,
            State.FINISH_SCENE: self._st_finish_scene,
            State.NEXT_SCENE: self._st_next_scene,
            State.NEXT_PROFILE: self._st_next_profile,
        }
        handler = handlers.get(self.state)
        if handler is not None:
            handler()

    # ── Boot / connection ────────────────────────────────────────────
    def _st_boot(self):
        if not self._profiles:
            self._error = "no_scene_profiles"
            self._enter_state(State.ERROR)
            return
        self._enter_state(State.WAIT_UNITY)

    def _st_wait_unity(self):
        # Auto-retry the ZMQ handshake until AvoidBench is ready (section
        # XXI).  No user input is required after launch.
        if not self._bridge._bound:
            self._bridge.bind()
            rospy.loginfo(
                "[Manager] Bound ZMQ in=%s / out=%s; waiting for AvoidBench "
                "ready handshake (scene id=%d)",
                self.g.get("pub_port"), self.g.get("sub_port"),
                self._unity_scene_id)
        if self._bridge.connect_handshake(
                self._unity_scene_id, self._depth_cfg,
                timeout=self._fsm.get("connect_timeout", 60.0)):
            rospy.loginfo("[Manager] Connected to AvoidBench "
                          "(unity scene id=%d)", self._unity_scene_id)
            self._enter_state(State.GENERATE_SCENE)
        elif self._timed_out(self._fsm.get("connect_timeout", 60.0)):
            self._error = "unity_connect_timeout"
            self._enter_state(State.ERROR)

    # ── Scene lifecycle (sections XX-XXVII) ──────────────────────────
    def _next_scene_seed_rng(self):
        import random as _rng_mod
        seed = (self._scene_generator.seed_base * 31 +
                self._profile_index * 104729 +
                self._scene_index * 15485863 +
                self._scene_attempt * 40503) & 0x7FFFFFFF
        return _rng_mod.Random(seed)

    def _st_generate_scene(self):
        if self._profile_index >= len(self._profiles):
            self._enter_state(State.DONE)
            return
        profile = self._profiles[self._profile_index]
        if self._scene_index >= profile.scene_count:
            self._profile_index += 1
            self._scene_index = 0
            self._scene_attempt = 0
            self._enter_state(State.NEXT_PROFILE)
            return
        try:
            scene = self._scene_generator.generate(
                profile, self._scene_index, self._scene_attempt)
            ok, reason, _ = self._scene_validator.validate(scene)
            if not ok:
                raise ValueError("scene invalid: %s" % reason)
        except Exception as exc:  # noqa: BLE001
            self._scene_attempt += 1
            GenerationFailureWriter.write(self._generation_failures_path, {
                "event": "scene_generation",
                "profile": profile.name,
                "scene_index": self._scene_index,
                "attempt": self._scene_attempt,
                "reason": str(exc),
            })
            if self._scene_attempt >= self._max_scene_attempts:
                self._error = "scene_generation_exhausted: %s" % exc
                self._enter_state(State.ERROR)
                return
            rospy.logwarn("[Manager] Scene retry %d: %s",
                          self._scene_attempt, exc)
            self._enter_state(State.GENERATE_SCENE)
            return
        self._current_scene = scene
        self._dataset_scene_key = scene.scene_key
        self._pc_retries = 0  # fresh scene -> fresh point-cloud retry budget
        self._generation_rng = self._next_scene_seed_rng()
        self._scene_dir = os.path.join(
            self.output_root, "scenes", scene.scene_key)
        if not os.path.isdir(self._scene_dir):
            os.makedirs(self._scene_dir)
        rospy.loginfo("[Manager] Profile %s Scene %d/%d: %s generated %d "
                      "obstacles (mode=%s)",
                      profile.name, self._scene_index + 1,
                      profile.scene_count, scene.scene_key,
                      len(scene.obstacles),
                      scene.metrics.get("mode"))
        self._enter_state(State.SEND_SCENE)

    def _drain_bridge_messages(self):
        """Drain EVERY pending AvoidBench message (returns the count).

        A single `try_recv()` would leave a backlog; scene switches and the
        settle / point-cloud phases must never mistake a stale reply from a
        previous scene for the current one (sections X-XI).  An empty queue
        (`try_recv() == None`) simply means no new message — it is NOT a
        connection failure (section XLVII).
        """
        drained = 0
        while True:
            r = self._bridge.try_recv()
            if r is None:
                break
            drained += 1
        return drained

    # ── Blocking-phase keep-alive ───────────────────────────────────
    # AvoidBench restarts its whole scene if it receives NO message for
    # `connection_timeout_seconds = 5` (CameraController.cs), which resets
    # readyToRender and re-runs its 5-step initializeObjects state machine
    # (one step per received Pose, no depth frames meanwhile).  The
    # blocking map-build / task-generation phases send nothing for many
    # seconds, tripping that timeout and burning the first ~6 depth
    # requests of the scene (depth_frame_timeout).  A background thread
    # re-sends the scene Pose every ~2 s so the silent gap never exceeds
    # 5 s.  send_pose() is mutex-protected (UnityBridge._send_lock) and
    # these phases never touch the bridge from the main thread, so the
    # background send is safe.
    def _start_keepalive(self):
        period = float(self._fsm.get("block_keepalive_period", 2.0))
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop, args=(period,), daemon=True)
        self._keepalive_thread.start()

    def _keepalive_loop(self, period):
        while not self._keepalive_stop.wait(period):
            try:
                if self._scene_pose_msg is not None:
                    self._bridge.send_pose(self._scene_pose_msg)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn("[Manager] keep-alive send failed: %s", exc)

    def _stop_keepalive(self):
        thread = self._keepalive_thread
        if thread is not None:
            self._keepalive_stop.set()
            thread.join(timeout=float(
                self._fsm.get("block_keepalive_period", 2.0)) + 1.0)
            self._keepalive_thread = None

    def _build_scene_pose_message(self):
        """One unified scene Pose message shared by SEND_SCENE and the
        SETTLE_SCENE / WAIT_POINTCLOUD keep-alives (section XLVI).  Uses
        the numeric `unity_scene_id` — never the dataset scene key."""
        vehicle = make_depth_vehicle([0.0, 0.0, 5.0], 0.0, self._depth_cfg)
        return {
            "scene_id": self._unity_scene_id,
            "frame_id": 0,
            "vehicles": [vehicle],
            "objects": self._current_scene.unity_objects,
        }

    def _st_send_scene(self):
        self._drain_bridge_messages()
        self._scene_pose_msg = self._build_scene_pose_message()
        self._bridge.send_pose(self._scene_pose_msg)
        self._settle_last_keepalive = 0.0
        rospy.loginfo("[Manager] Scene sent via Pose: %s (unity id=%d)",
                      self._dataset_scene_key, self._unity_scene_id)
        self._enter_state(State.SETTLE_SCENE)

    def _st_settle_scene(self):
        """Time-based scene settle (sections IV-IX).

        AvoidBench does NOT re-send `ready` after a procedural object
        update — `ready` is only an initial handshake condition (section
        II).  While settling we re-send the SAME scene Pose as keep-alive
        and drain the whole incoming queue; when `settle_time_s` elapses
        we proceed to the point cloud export regardless of whether any new
        `ready` arrived.  A missing second `ready` is never a scene
        generation failure (section IX).
        """
        sr = self._scene_runtime
        settle_time = float(sr.get("settle_time_s", 8.0))
        keep_interval = 1.0 / max(0.1, float(sr.get(
            "settle_keepalive_hz", 5.0)))
        self._drain_bridge_messages()
        now = time.monotonic()
        if now - self._settle_last_keepalive >= keep_interval:
            self._bridge.send_pose(self._scene_pose_msg)
            self._settle_last_keepalive = now
        if now - self._state_start >= settle_time:
            rospy.loginfo("[Manager] Scene settled: %s",
                          self._dataset_scene_key)
            self._enter_state(State.EXPORT_POINTCLOUD)

    def _st_export_pointcloud(self):
        self._drain_bridge_messages()
        pc_cfg = self.g.get("pointcloud", {})
        # Absolute directory with a trailing separator in the request
        # (sections XIX-XXI): AvoidBench joins `path` + `file_name`.
        pc_dir = os.path.abspath(os.path.join(self.output_root, "maps"))
        if not os.path.isdir(pc_dir):
            os.makedirs(pc_dir)
        # Scene-specific PLY name (section XXV/XXVI): use the DATASET scene
        # key, never the numeric Unity scene id.
        base = self._dataset_scene_key
        pc_request_path = pc_dir.rstrip("/\\") + os.sep
        req = {
            "range": pc_cfg.get("range", [27.0, 44.0, 8.0]),
            "origin": pc_cfg.get("origin", [1.5, 16.0, 3.5]),
            "resolution": pc_cfg.get("resolution", 0.10),
            "path": pc_request_path,
            "file_name": base,
        }
        self._pc_path = os.path.join(pc_dir, base + ".ply")
        # A stale PLY from an earlier identical scene key must never be
        # mistaken for this fresh export (section XXXI).
        if os.path.exists(self._pc_path):
            try:
                os.remove(self._pc_path)
            except OSError:
                pass
        # Per-request state.  The request is sent exactly ONCE per entry
        # (section XVIII); completion is decided by the PLY file itself.
        self._pc_request_time = time.monotonic()
        self._pc_ack_received = False
        self._pc_save_success_received = False
        self._pc_last_keepalive = 0.0
        self._last_ply_size = -1
        self._ply_stable_since = None
        self._bridge.send_pc_request(req)
        self._enter_state(State.WAIT_POINTCLOUD)

    def _st_wait_pointcloud(self):
        timeout = self._fsm.get("pc_export_timeout", 600.0)
        keep_interval = max(
            0.5, float(self._fsm.get("keep_alive_period", 3.0)))
        # Drain the WHOLE queue every tick; ACK / save_pc_success are only
        # diagnostics (sections XX-XXII).
        while True:
            r = self._bridge.try_recv()
            if r is None:
                break
            meta = r[0] if r else {}
            if meta.get("get_pc_msg"):
                self._pc_ack_received = True
            if meta.get("save_pc_success"):
                self._pc_save_success_received = True
        # Keep Unity communication alive without starting any episode
        # (sections XXV-XXVII): re-send the current scene Pose.
        now = time.monotonic()
        if now - self._pc_last_keepalive >= keep_interval:
            self._bridge.send_pose(self._scene_pose_msg)
            self._pc_last_keepalive = now
        # The PLY file itself is the final source of truth (sections
        # XXIII/XXVIII): existence + stability, independent of any ACK.
        if self._ply_is_stable():
            rospy.loginfo(
                "[Manager] point cloud stable: %s (ack=%s save_ok=%s)",
                self._pc_path, self._pc_ack_received,
                self._pc_save_success_received)
            self._enter_state(State.BUILD_PRIVILEGED_MAP)
            return
        # Real timeout only when the PLY never completed (section XXIX) —
        # a missing ACK / save_pc_success alone is never a failure.
        if now - self._pc_request_time > timeout:
            self._pc_retries += 1
            GenerationFailureWriter.write(self._generation_failures_path, {
                "event": "pointcloud_timeout",
                "scene_key": self._dataset_scene_key,
                "attempt": self._pc_retries,
                "ply_exists": os.path.isfile(self._pc_path),
            })
            if self._pc_retries >= self._pc_max_retries:
                # Give up on this scene: regenerate it (new seed).  Only
                # repeated global failures abort the whole collection
                # (section XXX).
                self._scene_attempt += 1
                if self._scene_attempt >= self._max_scene_attempts:
                    self._error = "pointcloud_export_exhausted"
                    self._enter_state(State.ERROR)
                    return
                self._enter_state(State.GENERATE_SCENE)
                return
            # Retry the SAME scene's point cloud export.
            rospy.logwarn("[Manager] point cloud timeout retry %d/%d",
                          self._pc_retries, self._pc_max_retries)
            self._enter_state(State.EXPORT_POINTCLOUD)

    def _ply_is_stable(self):
        """Non-blocking PLY stability check (section XXIV).  The file must
        exist, exceed the minimum size and keep an identical size for the
        stable window before it counts as complete."""
        try:
            if not os.path.isfile(self._pc_path):
                self._last_ply_size = -1
                self._ply_stable_since = None
                return False
            size = os.path.getsize(self._pc_path)
            now = time.monotonic()
            if size < self._pc_min_file_bytes:
                self._last_ply_size = size
                self._ply_stable_since = None
                return False
            if size == self._last_ply_size:
                if self._ply_stable_since is None:
                    self._ply_stable_since = now
                return (now - self._ply_stable_since) >= \
                    self._pc_stable_window_s
            self._last_ply_size = size
            self._ply_stable_since = None
            return False
        except OSError:
            return False

    def _st_build_privileged_map(self):
        # This blocking phase (PLY load + grid/ESDF/Dijkstra) can take
        # tens of seconds with no message on the wire; keep the scene Pose
        # flowing so AvoidBench's 5 s connection timeout never reloads the
        # scene (see _start_keepalive).
        self._start_keepalive()
        try:
            points = load_ply(self._pc_path)
            # Grid bounds = the AvoidBench point-cloud sampling box, NOT the
            # obstacle placement region.  SavePointCloud.cs interprets the
            # request origin as the CENTER and range as the FULL extent, so
            # the box is origin +/- range/2 (verified 'fixed' config).  It
            # is designed to cover the factory WALLS; the task start/goal
            # regions sit inside it.  Using the smaller obstacle region here
            # would clip the walls out of the privileged map and reject
            # start/goal near the factory edges.
            pc_cfg = self.g.get("pointcloud", {})
            pc_origin = np.asarray(
                pc_cfg.get("origin", [1.5, 16.0, 3.5]), dtype=np.float64)
            pc_range = np.asarray(
                pc_cfg.get("range", [27.0, 44.0, 8.0]), dtype=np.float64)
            rmin = pc_origin - 0.5 * pc_range
            rmax = pc_origin + 0.5 * pc_range
            if not self._oracle.build_scene(points, self._oracle_config,
                                            rmin, rmax):
                raise ValueError("privileged scene build failed")
            rospy.loginfo(
                "[Manager] privileged map built: %d points "
                "(grid x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f])",
                len(points), rmin[0], rmax[0], rmin[1], rmax[1], rmin[2],
                rmax[2])
        except Exception as exc:  # noqa: BLE001
            self._error = "privileged_map_build_failed: %s" % exc
            self._enter_state(State.ERROR)
            return
        finally:
            self._stop_keepalive()
        self._enter_state(State.GENERATE_TASKS)

    def _st_generate_tasks(self):
        rng = self._generation_rng
        # Same keep-alive reasoning as the map build: batch candidate
        # evaluation is blocking and can exceed AvoidBench's 5 s
        # connection timeout.
        self._start_keepalive()
        try:
            tasks, note = self._task_generator.generate_tasks(
                self._current_scene, self._oracle, self._quota, rng,
                self._tasks_per_scene)
        except Exception as exc:  # noqa: BLE001
            self._error = "task_generation_failed: %s" % exc
            self._enter_state(State.ERROR)
            return
        finally:
            self._stop_keepalive()
        if len(tasks) < self._min_tasks_per_scene:
            # This scene cannot produce enough valid tasks -> regenerate it
            # with a new seed (section XXXVIII/XLIX).  The generated tasks
            # were already note_scheduled()'d; since the scene is rejected
            # they will never run, so every one of them must be cancelled
            # (sections IX/XXXII) — otherwise phantom pending would make
            # these classes look over-satisfied.
            for t in tasks:
                try:
                    self._quota.note_cancelled(t.target_task_class)
                except RuntimeError as exc:
                    rospy.logerr("[Manager] quota cancel error for %s: %s",
                                 t.task_id, exc)
                t.quota_resolved = True
            self._scene_attempt += 1
            GenerationFailureWriter.write(self._generation_failures_path, {
                "event": "insufficient_tasks",
                "scene_key": self._current_scene.scene_key,
                "attempt": self._scene_attempt,
                "task_count": len(tasks),
            })
            if self._scene_attempt >= self._max_scene_attempts:
                self._error = "task_generation_exhausted"
                self._enter_state(State.ERROR)
                return
            rospy.logwarn("[Manager] Scene %s: only %d tasks; regenerating",
                          self._current_scene.scene_key, len(tasks))
            self._enter_state(State.GENERATE_SCENE)
            return
        self._current_tasks = tasks
        self._task_index = 0
        self._episode_index = 0
        self._total_scenes += 1
        self._write_scene_manifest()
        rospy.loginfo("[Manager] Scene %s: %s",
                      self._current_scene.scene_key, note)
        self._enter_state(State.RESET_DRONE)

    def _write_scene_manifest(self):
        path = os.path.join(self._scene_dir, "scene_manifest.json")
        tasks = [t.to_dict() for t in self._current_tasks]
        SceneManifestWriter.write(
            path, self._current_scene.to_dict(), tasks, self._pc_path,
            self._task_generator.task_seed_base, self._unity_scene_id)

    # ── Task lifecycle (sections XLIII/LII) ──────────────────────────
    def _record_task_failure(self, reason):
        self._failed_tasks += 1
        GenerationFailureWriter.write(self._generation_failures_path, {
            "event": "task_skipped",
            "scene_key": self._dataset_scene_key,
            "task_id": self._current_task_id,
            "reason": reason,
        })

    def _resolve_current_task_quota(self, outcome, committed_class=None):
        """Resolve the CURRENT task's quota exactly once (sections XII-XV).

        Every scheduled task must end in exactly one of COMMITTED / FAILED /
        CANCELLED; the per-task `_current_task_quota_resolved` guard makes
        double resolution impossible (sections XI/XLVI).  pending is always
        released under the task's TARGET (scheduled) class; only the
        committed credit may use a runtime class (sections VI-VII).  Marks
        `task.quota_resolved` so FINISH_SCENE can verify full resolution.
        """
        if self._current_task_quota_resolved:
            rospy.logerr("[Manager] quota double-resolve attempt for %s",
                         self._current_task_id)
            return
        self._current_task_quota_resolved = True
        task = self._current_tasks[self._task_index]
        task.quota_resolved = True
        scheduled_class = task.target_task_class
        try:
            if outcome == TaskOutcome.COMMITTED:
                cls = (committed_class if committed_class is not None
                       else scheduled_class)
                self._quota.note_committed(scheduled_class, cls)
            elif outcome == TaskOutcome.FAILED:
                self._quota.note_failed(scheduled_class)
            elif outcome == TaskOutcome.CANCELLED:
                self._quota.note_cancelled(scheduled_class)
        except RuntimeError as exc:
            # Pending underflow means a lifecycle bookkeeping bug; log it
            # loudly but never crash the unattended collection (section X).
            # The task stays marked resolved (no double attempts); any
            # leftover pending is auto-cancelled at scene end / DONE.
            rospy.logerr("[Manager] quota resolution error for %s: %s",
                         self._current_task_id, exc)

    def _st_reset_drone(self):
        if self._task_index >= len(self._current_tasks):
            self._enter_state(State.FINISH_SCENE)
            return
        task = self._current_tasks[self._task_index]
        self._current_task = {
            "task_id": task.task_id,
            "start": task.start,
            "goal": task.goal,
            "initial_yaw": task.initial_yaw,
            "task_class": task.target_task_class,
            "generation_metrics": task.metrics,
        }
        self._current_task_id = task.task_id
        self._current_task_quota_resolved = False
        # Goal-specific cost-to-go / connectivity for THIS task (the scene
        # map is already built once, section XLIV/LXXI).  A set_task failure
        # must still release the task's pending quota (section XVI).
        if not self._oracle.set_task(task.start, task.goal):
            self._record_task_failure("set_task_unreachable")
            self._resolve_current_task_quota(TaskOutcome.FAILED)
            self._task_index += 1
            self._enter_state(State.RESET_DRONE)
            return
        # A reset that the backend rejects is also a FAILED outcome — never
        # a silent skip that leaves phantom pending (section XVII).
        try:
            self._dynamics.reset(task.start, task.initial_yaw)
        except Exception as exc:  # noqa: BLE001
            self._record_task_failure("reset_failed: %s" % exc)
            self._resolve_current_task_quota(TaskOutcome.FAILED)
            self._task_index += 1
            self._enter_state(State.RESET_DRONE)
            return
        self._observed_map.reset(task.start)
        self._observed_map.force_rebuild_esdf()
        self._macro_expert.reset()
        self._intervention_oracle.reset()
        self._trajectory_reached_goal = False
        # A fast vehicle can pass through the goal tolerance disc between
        # 5 Hz macro ticks. This latch requests terminal recovery but is not
        # itself a success declaration.
        self._goal_capture_latched = False
        self._exit_reason = ""
        # P1/P5 per-episode safety + failure-taxonomy bookkeeping.
        self._last_plan_failure_status = None
        self._critical_plan_failure_occurred = False
        self._stale_plan_invalidations = 0
        self._committed_side_at_failure = int(self._macro_expert.committed_side)
        self._failure_taxonomy = ""
        self._consecutive_plan_failures = 0
        self._local_unrecoverable_pending = False
        self._brake_hold_time = 0.0
        self._emergency_stop_time = 0.0
        self._active_result = None
        self._active_plan_start_time = None
        self._active_plan_action = None
        self._total_plans = 0
        self._successful_plans = 0
        self._frame_latencies = []
        self._frame_finite_ratios = []
        self._unmatched_frames = 0
        self._exact_matches = 0
        self._last_velocity_world = None
        self._last_acceleration_world = None
        self._last_yaw_rate = 0.0
        self._last_execution_mode = None
        self._goal_capture_controller.reset()
        # 30 Hz -> 5 Hz macro-interval feedback accumulator (section XX).
        self._macro_feedback = self._new_macro_feedback()
        self._macro_feedback_log = None
        self._macro_feedback_is_new = 0
        # Per-episode writer.  A stale `_inprogress/<episode>.inprogress`
        # dir from a previous crashed run would make DatasetWriter's
        # os.makedirs() raise FileExistsError — remove it defensively even
        # though the startup output cleanup normally guarantees a clean
        # root.
        self._episode_id = "%s_ep%02d" % (task.task_id, self._episode_index)
        stale_episode_dir = os.path.join(
            self._inprogress_root, self._episode_id + ".inprogress")
        if os.path.isdir(stale_episode_dir):
            shutil.rmtree(stale_episode_dir, ignore_errors=True)
        self._writer = DatasetWriter(
            self.g.get("dataset_logging", {}), self._episode_id,
            self._inprogress_root, self._dataset_scene_key, task.task_id,
            task.start, task.goal, task.initial_yaw, self._depth_cfg)
        self._recording_start_mono = time.monotonic()
        self._matched_frames = 0
        self._unmatched_frames = 0
        self._exact_matches = 0
        # Initial observed recoverability cache (sections XXX-XLII): frozen
        # at the FIRST valid observed evaluation of this episode, never at
        # the end.  None means no valid initial evaluation was obtained.
        self._initial_observed_recoverable = None
        self._initial_observed_recoverability_status = None
        self._initial_observed_recoverability_tick = None
        # Quick settle before the lockstep loop.
        settle = time.monotonic() + 0.3
        while time.monotonic() < settle:
            ds = self._dynamics.get_state()
            if float(np.linalg.norm(ds.velocity_world)) < 0.05:
                break
            time.sleep(0.01)
        rospy.loginfo("[Manager] Task %d/%d %s [%s]",
                      self._task_index + 1, len(self._current_tasks),
                      task.task_id, task.target_task_class.value)
        self._enter_state(State.RUN_TASK)

    # ═════════════════════════════════════════════════════════════════
    #  30 Hz online loop (blocking)
    # ═════════════════════════════════════════════════════════════════
    def _st_run_task(self):
        dt = self._dt
        macro_interval = self._macro_interval
        goal = self._current_task["goal"]
        trajectory_timeout = self._fsm.get("trajectory_timeout", 120.0)
        # Independent WALL-CLOCK task watchdog (section "time"): the sim
        # trajectory timeout bounds SIMULATED flight time, but if the
        # simulation time stalls (dynamics / depth / planning degradation)
        # a task could occupy the process indefinitely.  This cap bounds
        # the PROCESS time per task; it is deliberately well above the
        # nominal wall duration and never affects navigation timing.
        trajectory_wall_timeout = float(
            self._fsm.get("trajectory_wall_timeout_s", 600.0))
        # P1 stale-plan invalidation: a fresh-plan failure with one of
        # these statuses means the ENVIRONMENT changed around the drone
        # (new collision / unknown / dead-end / invalid terminal).  After
        # such a failure the OLD cached trajectory may steer toward the
        # very region that just invalidated the fresh plan, so the cache is
        # dropped immediately and the episode falls through to brake /
        # emergency instead of re-executing the stale suffix.
        _critical_plan_failures = (
            module.PlannerStatus.COLLISION,
            module.PlannerStatus.UNKNOWN_SPACE,
            module.PlannerStatus.SEARCH_FAILED,
            module.PlannerStatus.LOCAL_TERMINAL_INVALID,
            module.PlannerStatus.DYNAMICS_VIOLATION,
            module.PlannerStatus.INVALID_INPUT,
        )
        sample_index = 0
        last_plan_time = None
        last_trajectory = []

        # The macro action is held in the world frame for macro_interval
        # frames (section IX).
        held_action = None

        # ── Depth warm-up ───────────────────────────────────────────
        # AvoidBench's depth renderer may stay silent for several seconds
        # right after a scene load / point-cloud export / connection-timeout
        # scene reload: while readyToRender is false it feeds received
        # Poses into its 5-step initializeObjects state machine and sends
        # NO depth frame, so the first depth request of such a task would
        # time out (2 s) and the episode would be rejected with 1 empty
        # row.  Wait for the FIRST frame here (capped by
        # fsm.depth_warmup_timeout) before the recorded loop.  Warm-up
        # probes are discarded and must not count toward the sync gates,
        # so the match counters are reset before recording starts.
        warmup_timeout = float(self._fsm.get("depth_warmup_timeout", 30.0))
        ds_warm = self._dynamics.get_state()
        warm_pos = ds_warm.position_world
        warm_quat = ds_warm.quaternion_world_body
        warmup_start = time.monotonic()
        while time.monotonic() - warmup_start < warmup_timeout \
                and not rospy.is_shutdown():
            if self._request_depth_frame(-1, warm_pos, warm_quat) is not None:
                break
        self._exact_matches = 0
        self._unmatched_frames = 0
        # Simulation-time origin for this episode (section "time"): the
        # dynamics backend advances `DynamicsState.simulation_time_s` by a
        # fixed 1/record_hz per executed `step_velocity_command()` —
        # regardless of the wall-clock rate.  ALL navigation / data time is
        # measured on the SIMULATION axis; the wall clock is used only for
        # pacing, communication timeouts and performance statistics.
        recording_sim_start = self._dynamics.get_state().simulation_time_s

        while not rospy.is_shutdown():
            t_loop = time.monotonic()
            # Wall-clock elapsed since the episode start: ONLY for pacing /
            # watchdog / performance statistics — never for navigation or
            # data time (section "time").
            wall_elapsed = t_loop - self._recording_start_mono
            # Independent WALL-CLOCK watchdog: if the simulation time axis
            # stalls or the loop degrades severely, never let one task
            # occupy the process indefinitely.  Sim time stays authoritative
            # for navigation; this only bounds the wall time per task.
            if wall_elapsed > trajectory_wall_timeout:
                self._exit_reason = "wall_clock_watchdog"
                break

            frame_id = sample_index

            # ── 1. State ────────────────────────────────────────────
            ds = self._dynamics.get_state()
            pos = ds.position_world
            quat = ds.quaternion_world_body
            vel_world = ds.velocity_world
            acc_world = ds.acceleration_world
            yaw = float(np.arctan2(
                2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2])))
            state = {
                "position": pos,
                "velocity": vel_world,
                "acceleration": acc_world,
                "yaw": yaw,
                "yaw_rate": float(ds.angular_velocity_body[2]) if
                len(ds.angular_velocity_body) > 2 else 0.0,
            }
            # Simulation time is the SINGLE navigation/data time axis.
            sim_elapsed = ds.simulation_time_s - recording_sim_start
            # The trajectory timeout measures SIMULATED flight time (the
            # wall clock may lag far behind the simulation).
            if sim_elapsed > trajectory_timeout:
                self._exit_reason = "trajectory_timeout"
                break
            timestamp = sim_elapsed
            if np.linalg.norm(np.asarray(pos, dtype=np.float64) - goal) <= \
                    float(self.g.get("macro_expert", {}).get(
                        "goal_tolerance_m", 0.30)):
                self._goal_capture_latched = True

            # ── 2. Depth ────────────────────────────────────────────
            depth_frame = self._request_depth_frame(frame_id, pos, quat)
            if depth_frame is None:
                self._exit_reason = "depth_frame_timeout"
                break
            depth_m_raw, depth_student, raw_finite_ratio, latency_ms = \
                depth_frame
            self._frame_latencies.append(latency_ms)
            self._frame_finite_ratios.append(raw_finite_ratio)

            # ── 3. Observed map integration (C++) ───────────────────
            # The RAW depth (with NaN/Inf preserved) is integrated; the C++
            # map classifies ray validity itself (section XII).  The
            # canonicalised student depth is ONLY recorded, never used for
            # ray integration.
            recentered = self._observed_map.recenter_if_needed(pos)
            self._observed_map.integrate_depth(
                depth_m_raw.astype(np.float32), pos, quat, timestamp)
            if recentered:
                # The map was wiped; integrate first, then rebuild so the
                # free bubble and rays are reflected.
                self._observed_map.force_rebuild_esdf()
            else:
                self._observed_map.rebuild_esdf()  # cadence-controlled

            # ── 4. Macro expert (5 Hz) ──────────────────────────────
            is_macro_tick = (sample_index % macro_interval) == 0
            if is_macro_tick:
                dt_macro = macro_interval * dt
                interval_feedback = self._consume_macro_interval_feedback()
                interval_feedback["goal_capture_latched"] = int(
                    self._goal_capture_latched)
                self._macro_feedback_log = interval_feedback
                self._macro_feedback_is_new = 1
                held_action = self._macro_expert.update(
                    goal, state, self._observed_map, dt_s=dt_macro,
                    now_s=sim_elapsed, interval_feedback=interval_feedback)
                # Freeze the initial OBSERVED recoverability at the FIRST
                # valid evaluation of this episode (sections XXX-XLII):
                # only once the observed map actually holds integrated data
                # and the macro expert has computed a current
                # recoverability.  Never refrozen later; never replaced at
                # FINISH_TASK by the last evaluation.
                if self._initial_observed_recoverable is None and \
                        self._observed_map.known_count() > 0 and \
                        self._macro_expert.last_recoverability is not None:
                    status = \
                        self._macro_expert.last_recoverability.status
                    self._initial_observed_recoverable = bool(
                        status == module.RecoverabilityStatus
                        .DIRECT_REJOIN_SUCCESS)
                    self._initial_observed_recoverability_status = status
                    self._initial_observed_recoverability_tick = sample_index
                # Debug trace (section "debug"): one JSON line per macro
                # tick with the expert's internal state for post-hoc review
                # via scripts/debug_viewer.py.
                if self.g.get("dataset_logging", {}).get(
                        "debug_trace", False):
                    trace = dict(self._macro_expert.last_trace or {})
                    trace["frame"] = sample_index
                    trace["trajectory_time_s"] = round(sim_elapsed, 4)
                    self._writer.write_trace(trace)
                if held_action.mode == module.MacroMode.GOAL_REACHED:
                    self._trajectory_reached_goal = True
                    self._exit_reason = "goal_reached"
                    break
                if held_action.mode == module.MacroMode.FAILED:
                    self._exit_reason = "macro_%s" % self._macro_expert.failed_reason
                    break
            elif held_action is not None:
                held_action.is_new_tick = False
            elif held_action is None:
                # Very first frames: build a direct guide until the first
                # macro tick.
                held_action = self._macro_expert.make_direct_action(goal, state)
            if not is_macro_tick:
                self._macro_feedback_log = None
                self._macro_feedback_is_new = 0

            # ── 5. Local planning (30 Hz, observed map only) ────────
            req = module.LocalPlanRequest()
            req.state.position = pos
            req.state.velocity = vel_world
            req.state.acceleration = acc_world
            req.state.yaw = yaw
            req.state.yaw_rate = state["yaw_rate"]
            req.macro_guide_world = held_action.guide_world
            req.has_macro_yaw = held_action.has_desired_yaw
            req.macro_yaw_world = held_action.desired_yaw_world
            req.goal_world = goal
            # Stop-at-goal is decided INSIDE the planner (section XVII):
            # full search arrival + goal_stop_tolerance + known goal.
            req.committed_side = held_action.committed_side
            req.previous_trajectory = last_trajectory
            if last_plan_time is not None:
                req.previous_trajectory_age_s = sim_elapsed - last_plan_time
            result = self._local_planner.plan(req)
            self._total_plans += 1

            # Classify a planning failure only after execution selection.
            # A cached suffix has its own current-map validation; camera
            # rotation and goal capture do not execute this spline result.
            # ── 6. Execution selection (section XIII) ───────────────
            execution_mode, trajectory, fresh_plan, cached_used, \
                sample_offset = self._select_execution(
                    result, last_trajectory, last_plan_time, sim_elapsed,
                    held_action, state, goal)
            planning_failure_executed = execution_mode in (
                module.ExecutionMode.BRAKE_HOLD,
                module.ExecutionMode.EMERGENCY_STOP)
            if planning_failure_executed and \
                    result.status in _critical_plan_failures:
                self._last_plan_failure_status = int(result.status)
                self._critical_plan_failure_occurred = True
            # A successful replan is a candidate replacement, not an order to
            # restart execution at trajectory t=0.  Only the trajectory that
            # is actually selected for execution becomes the active cache.
            if execution_mode == module.ExecutionMode.TRACK_FRESH:
                self._successful_plans += 1
                self._consecutive_plan_failures = 0
                self._local_unrecoverable_pending = False
                last_trajectory = result.trajectory
                last_plan_time = sim_elapsed
            elif result.success:
                self._successful_plans += 1
                self._consecutive_plan_failures = 0
                self._local_unrecoverable_pending = False

            # Executed-plan semantics (sections XVII/XVIII): the training
            # labels and plan metadata must describe the ACTIVE executed
            # trajectory, not the failed planning attempt of this frame.
            if execution_mode == module.ExecutionMode.TRACK_FRESH:
                self._active_result = result
                self._active_plan_start_time = sim_elapsed
                self._active_plan_action = self._action_signature(held_action)
                executed_result = result
            elif execution_mode == module.ExecutionMode.TRACK_CACHED and \
                    self._active_result is not None:
                executed_result = self._active_result
            else:
                executed_result = result
            if execution_mode == module.ExecutionMode.TRACK_CACHED:
                # Cached execution: a per-frame validated active suffix
                # remains selected, whether the replacement replan succeeded
                # or failed.  The controller therefore advances in simulated
                # trajectory time instead of restarting at t=0.
                pass
            elif execution_mode in (
                    module.ExecutionMode.ROTATE_ONLY,
                    module.ExecutionMode.GOAL_CAPTURE):
                self._consecutive_plan_failures = 0
                self._local_unrecoverable_pending = False
            elif execution_mode != module.ExecutionMode.TRACK_FRESH:
                self._consecutive_plan_failures += 1
                self._local_unrecoverable_pending = \
                    self._consecutive_plan_failures >= 2

            # Brake / emergency hold timeouts (section XIII).
            if execution_mode == module.ExecutionMode.BRAKE_HOLD:
                self._brake_hold_time += dt
            else:
                self._brake_hold_time = 0.0
            if execution_mode == module.ExecutionMode.EMERGENCY_STOP:
                # Round 5 (requirement 4): an emergency stop VOIDS the
                # executed trajectory.  The cached suffix and the active
                # plan must never be re-selected on a later frame to resume
                # forward motion while the drone is still decelerating —
                # dynamic braking keeps commanding deceleration from the
                # CURRENT velocity, and with the cache cleared every
                # subsequent frame can only hold/stop, never TRACK_CACHED.
                last_trajectory = []
                last_plan_time = None
                self._active_result = None
                self._active_plan_start_time = None
                self._active_plan_action = None
                self._emergency_stop_time += dt
                if self._emergency_stop_time > \
                        self._safety_cfg.get("max_emergency_stop_seconds", 2.0):
                    self._exit_reason = "emergency_stop_timeout"
                    break
            else:
                self._emergency_stop_time = 0.0
            # Accumulate this frame into the macro-interval feedback
            # (section XXII): the 5 Hz macro consumes the WHOLE interval,
            # not just the last frame before its tick.
            fb = self._macro_feedback
            fb["interval_frame_count"] += 1
            if not result.success and planning_failure_executed:
                fb["planning_failure_count"] += 1
            if execution_mode == module.ExecutionMode.TRACK_FRESH:
                fb["fresh_frame_count"] += 1
            elif execution_mode == module.ExecutionMode.TRACK_CACHED:
                fb["cached_frame_count"] += 1
            elif execution_mode == module.ExecutionMode.BRAKE_HOLD:
                fb["brake_frame_count"] += 1
            elif execution_mode == module.ExecutionMode.EMERGENCY_STOP:
                fb["emergency_frame_count"] += 1
            if self._local_unrecoverable_pending:
                fb["local_unrecoverable_count"] += 1

            # ── 7. Trajectory controller ────────────────────────────
            cmd = self._compute_command(
                trajectory, state, ds, execution_mode, held_action,
                goal, goal[2], sample_offset)
            if cmd is None:
                self._exit_reason = "controller_invalid"
                break

            # ── 8. Execute ──────────────────────────────────────────
            ok = self._dynamics.step_velocity_command(
                cmd["velocity_flu"], cmd["yaw_rate"], dt)
            if not ok:
                self._exit_reason = "dynamics_step_failed"
                break

            # ── 9. Record (labels match the executed command) ───────
            self._writer.write_sync(
                frame_id, latency_ms, "frame_id_exact", False,
                self._exact_matches, self._unmatched_frames)
            plan_age_s = sim_elapsed - self._active_plan_start_time \
                if self._active_plan_start_time is not None else 0.0
            self._write_row(frame_id, sample_index, sim_elapsed, state, ds,
                            depth_student, raw_finite_ratio, latency_ms,
                            held_action, executed_result, result,
                            execution_mode, fresh_plan, cached_used, cmd,
                            plan_age_s)
            self._matched_frames += 1

            sample_index += 1

            # Wall-clock pacing at record rate (wall clock ONLY — the loop
            # may legitimately run below 30 Hz; sim time is authoritative
            # for navigation / data).
            if wall_elapsed < sample_index * dt:
                time.sleep(sample_index * dt - wall_elapsed)

        self._trajectory_exit_reason = self._exit_reason
        # The committed side at the moment the loop stopped (for the
        # side_selection_error / unsafe_approach taxonomy).
        self._committed_side_at_failure = int(
            self._macro_expert.committed_side)
        # Classify BEFORE the final row is written so the recorded
        # failure_taxonomy reflects the actual exit (pure diagnostic).
        self._classify_failure()
        self._writer.write_row({
            "episode_frame_index": sample_index,
            "frame_valid": 0,
            "frame_invalid_reason": self._exit_reason,
            "failure_taxonomy": self._failure_taxonomy,
        })
        self._enter_state(State.FINISH_TASK)

    # ── P5 failure taxonomy (pure diagnostics, never student input) ──
    def _classify_failure(self):
        """Map the episode exit to one of the documented failure
        categories (diagnostic only — never a student input and never a
        mode hint):

          - goal_reached            terminal success (position + speed).
          - unsafe_approach         the drone had to emergency-stop /
                                    controller-invalidate while moving;
                                    no committed side was the cause.
          - stale_plan_after_failure a safety-critical fresh-plan failure
                                    occurred earlier and the episode then
                                    ended in an emergency/controller stop —
                                    the P1 cache invalidation is what keeps
                                    this from executing a stale plan.
          - side_selection_error    the episode failed while a LEFT/RIGHT
                                    side was committed (SIDE_GUIDE) or right
                                    after a critical plan failure under a
                                    committed side — the side led into an
                                    unsafe / dead-end approach.
          - observe_deadlock        OBSERVE could not resolve within its
                                    absolute budget / no valid side.
          - local_unknown_block     exit was caused by unknown-space /
                                    depth availability rather than a
                                    committed-side error.
          - trajectory_timeout      simulated flight-time budget exceeded.
        """
        r = self._trajectory_exit_reason or ""
        side = self._committed_side_at_failure
        side_committed = side in (int(module.Side.LEFT),
                                  int(module.Side.RIGHT))
        if self._trajectory_reached_goal:
            self._failure_taxonomy = "goal_reached"
            return
        if r == "emergency_stop_timeout":
            if self._critical_plan_failure_occurred:
                self._failure_taxonomy = "stale_plan_after_failure"
            elif side_committed:
                self._failure_taxonomy = "side_selection_error"
            else:
                self._failure_taxonomy = "unsafe_approach"
            return
        if r in ("controller_invalid", "dynamics_step_failed"):
            self._failure_taxonomy = "unsafe_approach"
            return
        if "observe" in r or r == "no_valid_side_no_route":
            self._failure_taxonomy = "observe_deadlock"
            return
        if r.startswith("macro_"):
            self._failure_taxonomy = ("side_selection_error"
                                      if side_committed else "macro_failed")
            return
        if r == "depth_frame_timeout":
            # No depth -> the observed map cannot grow -> unknown blocks.
            self._failure_taxonomy = "local_unknown_block"
            return
        if r == "trajectory_timeout":
            self._failure_taxonomy = "trajectory_timeout"
            return
        if self._critical_plan_failure_occurred:
            self._failure_taxonomy = ("side_selection_error"
                                      if side_committed
                                      else "local_unknown_block")
            return
        self._failure_taxonomy = "unknown"

    # ── Macro-interval feedback (sections XX-XXIII) ──────────────────
    def _new_macro_feedback(self):
        return {
            "interval_frame_count": 0,
            "planning_failure_count": 0,
            "fresh_frame_count": 0,
            "cached_frame_count": 0,
            "brake_frame_count": 0,
            "emergency_frame_count": 0,
            "local_unrecoverable_count": 0,
        }

    def _consume_macro_interval_feedback(self):
        """Consume the aggregated 30 Hz feedback of the last macro
        interval and reset the accumulator for the next one (section
        XXIII)."""
        feedback = dict(self._macro_feedback)
        self._macro_feedback = self._new_macro_feedback()
        return feedback

    # ── Depth request / match ────────────────────────────────────────
    def _request_depth_frame(self, frame_id, pos, quat):
        vehicle = make_depth_vehicle(pos, 0.0, self._depth_cfg,
                                     quaternion_xyzw=quat)
        msg = {
            "scene_id": self._unity_scene_id,
            "frame_id": frame_id,
            "vehicles": [vehicle],
            "objects": self._current_scene.unity_objects,
        }
        t_sent = time.monotonic()
        self._bridge.send_pose(msg)
        deadline = t_sent + \
            self._sync_cfg.get("unity_response_timeout_s", 2.0)
        while time.monotonic() < deadline and not rospy.is_shutdown():
            r = self._bridge.try_recv()
            if r is None:
                time.sleep(0.002)
                continue
            meta, parts = r
            reply_id = meta.get("pub_frame_id", meta.get("frame_id", -1))
            if reply_id != frame_id:
                self._unmatched_frames += 1
                continue
            latency_ms = (time.monotonic() - t_sent) * 1000.0
            for part in parts:
                expected = self._depth_cfg.get("width", 640) * \
                    self._depth_cfg.get("height", 480) * 4
                if len(part) >= expected:
                    raw = np.frombuffer(part[:expected], dtype=np.float32)
                    depth = raw.reshape(
                        (self._depth_cfg.get("height", 480),
                         self._depth_cfg.get("width", 640)))
                    # AvoidBench validated convention: raw * 100 = metres.
                    # Section XII: validity is decided on the RAW frame
                    # BEFORE any NaN/Inf -> max replacement.  NaN/Inf/<=0
                    # pixels are "no valid measurement" (they must NOT be
                    # interpreted as free-to-max rays).  The RAW metres are
                    # handed to the C++ observed map which performs the
                    # ray-validity classification itself.
                    depth_m_raw = np.flipud(depth * 100.0)
                    valid = np.isfinite(depth_m_raw) & (depth_m_raw > 0.0)
                    raw_finite_ratio = float(np.mean(valid))
                    # Canonicalised STUDENT depth (network input, single
                    # channel, no mask): invalid / no-return -> perception
                    # range, clipped.  Never re-used for map integration.
                    perception_range = self.g.get(
                        "dataset_logging", {}).get("perception_range_m",
                                                   self._depth_cfg.get(
                                                       "max_m", 5.0))
                    depth_student = np.nan_to_num(
                        depth_m_raw, nan=perception_range,
                        posinf=perception_range, neginf=0.0)
                    depth_student = np.clip(
                        depth_student, 0.0, perception_range)
                    self._exact_matches += 1
                    return depth_m_raw, depth_student, raw_finite_ratio, \
                        latency_ms
        self._unmatched_frames += 1
        return None

    # ── Execution selection (section XIII) ───────────────────────────
    @staticmethod
    def _action_signature(action):
        if action is None:
            return None
        return {
            "mode": int(action.mode),
            "guide": np.asarray(action.guide_world, dtype=np.float64).copy(),
            "has_yaw": bool(action.has_desired_yaw),
            "yaw": float(action.desired_yaw_world),
            "side": int(action.committed_side),
            "observe_subtype": int(getattr(action, "observe_subtype", 0)),
        }

    def _active_plan_matches_action(self, held_action):
        active = self._active_plan_action
        if active is None or held_action is None:
            return False
        current = self._action_signature(held_action)
        if (active["mode"] != current["mode"] or
                active["side"] != current["side"] or
                active["observe_subtype"] != current["observe_subtype"] or
                active["has_yaw"] != current["has_yaw"]):
            return False
        if np.linalg.norm(active["guide"] - current["guide"]) > \
                float(self._safety_cfg.get(
                    "active_guide_replan_distance_m", 0.50)):
            return False
        if active["has_yaw"] and abs(normalize_angle(
                active["yaw"] - current["yaw"])) > float(
                    self._safety_cfg.get("active_yaw_replan_delta_rad", 0.20)):
            return False
        return True

    def _select_execution(self, result, last_trajectory, last_plan_time,
                          sim_elapsed, held_action, state, goal_world):
        safety = self._safety_cfg
        # The terminal controller is allowed only over a complete segment
        # that is already known-free in the causal observed map.  The
        # ignored spline result therefore cannot leak hidden geometry.
        if held_action is not None and \
                held_action.mode == module.MacroMode.GOAL_APPROACH:
            vs = module.VehicleState()
            vs.position = state["position"]
            vs.velocity = state["velocity"]
            vs.acceleration = state["acceleration"]
            vs.yaw = state["yaw"]
            vs.yaw_rate = state["yaw_rate"]
            if self._observed_map.segment_known_and_clear(
                    state["position"], goal_world,
                    self._local_planner.effective_clearance_for(vs), 0.05):
                return (module.ExecutionMode.GOAL_CAPTURE, [],
                        False, False, 0.0)

        # A camera-only macro action preempts any old forward trajectory.
        if held_action is not None and \
                held_action.mode == module.MacroMode.OBSERVE and \
                getattr(held_action, "observe_subtype", 0) == 0:
            return (module.ExecutionMode.ROTATE_ONLY, [], False, False, 0.0)

        # Cached trajectory: the C++ suffix validator re-checks the
        # remaining segment against the CURRENT observed map (section
        # VIII).  The controller samples it with an offset equal to the
        # trajectory age (sim_elapsed - plan_start_time) — all on the
        # SIMULATION time axis.
        if (last_trajectory and last_plan_time is not None and
                self._active_plan_matches_action(held_action)):
            age = sim_elapsed - last_plan_time
            if age <= safety.get("max_plan_age_s", 0.70):
                remaining = last_trajectory[-1].t - age
                if remaining >= safety.get("min_remaining_trajectory_s", 0.10):
                    vs = module.VehicleState()
                    vs.position = state["position"]
                    vs.velocity = state["velocity"]
                    vs.acceleration = state["acceleration"]
                    vs.yaw = state["yaw"]
                    vs.yaw_rate = state["yaw_rate"]
                    valid = self._local_planner.validate_trajectory_suffix(
                        last_trajectory, last_plan_time, sim_elapsed, vs,
                        safety.get("max_position_error_m", 0.50),
                        safety.get("max_velocity_error_mps", 1.00))
                    if valid.all_clear:
                        return (module.ExecutionMode.TRACK_CACHED,
                                last_trajectory, False, True, age)

        if result.success:
            return (module.ExecutionMode.TRACK_FRESH, result.trajectory,
                    True, False, 0.0)

        # Emergency brake when the swept braking volume collides (C++).
        if self._collision_risk(state):
            return (module.ExecutionMode.EMERGENCY_STOP, [], False, False, 0.0)

        # Brake-hold for too long (still planning-failed) escalates to an
        # emergency stop (section XIII).
        if self._brake_hold_time > \
                safety.get("max_brake_hold_seconds", 1.0):
            return (module.ExecutionMode.EMERGENCY_STOP, [], False, False, 0.0)

        return (module.ExecutionMode.BRAKE_HOLD, [], False, False, 0.0)

    def _collision_risk(self, state):
        """Swept-volume braking + current-pose safety check computed in
        C++ (sections XVIII / round 5-6).

        ALWAYS evaluates the current pose, the current velocity and the
        predicted braking trajectory (reaction delay at constant velocity,
        then deceleration to rest).  Low speed ONLY shortens the predicted
        braking distance inside the C++ `swept_brake_risk` — it NEVER
        disables collision detection (the old `speed < 0.3` early-return is
        gone).  There is NO Python-side early-return either: when the ESDF
        is not built / the map is uninitialised / the state is invalid, the
        C++ `swept_brake_risk` itself returns risk=true (round 6), so
        unknown space is never silently treated as safe and the episode
        cannot enter BRAKE_HOLD and wait without a risk diagnosis.

        The hard boundary is the SINGLE C++ computation
        (`_local_planner.effective_clearance_for`), shared with cached
        suffix validation. Fresh planning adds its C++ quality margin; no
        Python-side safety formula is duplicated here."""
        vs = module.VehicleState()
        vs.position = state["position"]
        vs.velocity = state["velocity"]
        vs.acceleration = state["acceleration"]
        vs.yaw = state["yaw"]
        vs.yaw_rate = state["yaw_rate"]
        brake_result = self._observed_map.swept_brake_risk(
            vs,
            self._safety_cfg.get("brake_reaction_delay_s", 0.10),
            self._safety_cfg.get("emergency_deceleration_mps2", 5.0),
            self._local_planner.effective_clearance_for(vs),
            0.05)
        return bool(brake_result.risk)

    # ── Trajectory controller (section XII) ──────────────────────────
    def _compute_command(self, trajectory, state, ds, execution_mode,
                         held_action, goal_world, altitude_target_z,
                         sample_offset=0.0):
        tc = self._controller_cfg
        out = {"velocity_flu": np.zeros(3), "yaw_rate": 0.0}
        max_yaw_rate = tc.get("max_yaw_rate_rps", 2.0)
        max_vel = tc.get("max_velocity_mps", 2.5)
        max_accel = tc.get("max_acceleration_mps2", 3.5)
        max_jerk = tc.get("max_jerk_mps3", 25.0)
        accel_limit = max_accel * self._dt
        jerk_limit = max_jerk * self._dt
        max_yaw_accel = tc.get("max_yaw_accel_rps2", 8.0) * self._dt
        yaw_gain = tc.get("yaw_gain", 2.0)
        yaw = float(state["yaw"])
        quat = ds.quaternion_world_body

        if execution_mode == module.ExecutionMode.GOAL_CAPTURE and \
                self._last_execution_mode != module.ExecutionMode.GOAL_CAPTURE:
            self._goal_capture_controller.reset()

        if execution_mode in (module.ExecutionMode.TRACK_FRESH,
                              module.ExecutionMode.TRACK_CACHED):
            # Sample the trajectory at the velocity lookahead time.  For a
            # cached trajectory the sample is shifted by the trajectory age
            # (sim_elapsed - plan_start_time, SIMULATION time) so the drone
            # re-executes the REMAINING segment, not from t=0 (section
            # VIII).
            lookahead = tc.get("velocity_lookahead_time_s", 0.08) + \
                max(0.0, sample_offset)
            idx = 0
            best = 1e9
            for i, point in enumerate(trajectory):
                d = abs(point.t - lookahead)
                if d < best:
                    best = d
                    idx = i
            point = trajectory[idx]
            # p_ref / v_ref / a_ref
            vel_world = point.velocity + \
                tc.get("position_gain", 2.0) * (point.position - state["position"])
            # Acceleration feedforward: the commanded acceleration acting
            # over the feedforward horizon contributes to the required
            # velocity.
            vel_world = vel_world + \
                point.acceleration * tc.get("acceleration_feedforward_time_s", 0.30)

            # Limits: max velocity, real acceleration AND jerk limits
            # (section XIX).  The previous velocity gives the acceleration
            # limit; the previous acceleration gives the jerk limit.
            speed = float(np.linalg.norm(vel_world))
            if speed > max_vel and speed > 1e-9:
                vel_world = vel_world * (max_vel / speed)
            prev = getattr(self, "_last_velocity_world", None)
            prev_accel = getattr(self, "_last_acceleration_world", None)
            if self._last_execution_mode not in (
                    module.ExecutionMode.TRACK_FRESH,
                    module.ExecutionMode.TRACK_CACHED):
                # A fresh trajectory is initialized from the measured
                # physical state after rotate/brake/emergency, not from the
                # previous mode's zero/hold command.  This prevents the first
                # recovered plan from commanding near-zero velocity while the
                # vehicle is still moving.
                prev = np.asarray(state["velocity"], dtype=np.float64)
                prev_accel = np.asarray(state["acceleration"],
                                        dtype=np.float64)
            if prev is not None:
                delta = vel_world - prev
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > accel_limit and delta_norm > 1e-9:
                    vel_world = prev + delta * (accel_limit / delta_norm)
            if prev is not None and prev_accel is not None:
                new_accel = (vel_world - prev) / self._dt
                accel_change = new_accel - prev_accel
                change_norm = float(np.linalg.norm(accel_change))
                if change_norm > jerk_limit and change_norm > 1e-9:
                    desired_accel = prev_accel + \
                        accel_change * (jerk_limit / change_norm)
                    vel_world = prev + desired_accel * self._dt
            vel_flu = world_vector_to_body_flu_quat(vel_world, quat)
            out["velocity_flu"] = vel_flu

            # Yaw: reference yaw + closed-loop gain.
            yaw_ref = point.yaw
            yaw_rate_ref = point.yaw_rate
            yaw_err = normalize_angle(yaw_ref - yaw)
            yr = yaw_rate_ref + yaw_gain * yaw_err
            yr = max(-max_yaw_rate, min(max_yaw_rate, yr))
            prev_yr = getattr(self, "_last_yaw_rate", 0.0)
            yr = max(prev_yr - max_yaw_accel, min(prev_yr + max_yaw_accel, yr))
            self._last_yaw_rate = yr
            out["yaw_rate"] = yr
        elif execution_mode == module.ExecutionMode.GOAL_CAPTURE:
            vs = module.VehicleState()
            vs.position = state["position"]
            vs.velocity = state["velocity"]
            vs.acceleration = state["acceleration"]
            vs.yaw = state["yaw"]
            vs.yaw_rate = state["yaw_rate"]
            capture = self._goal_capture_controller.compute(
                vs, goal_world, self._goal_capture_latched, self._dt)
            if not capture.valid:
                return None
            vel_world = np.asarray(capture.velocity_world,
                                   dtype=np.float64)
            out["velocity_flu"] = world_vector_to_body_flu_quat(
                vel_world, quat)
            yaw_target = yaw
            if held_action is not None and held_action.has_desired_yaw:
                yaw_target = held_action.desired_yaw_world
            yaw_err = normalize_angle(yaw_target - yaw)
            yr = max(-max_yaw_rate, min(max_yaw_rate,
                                        yaw_gain * yaw_err))
            prev_yr = getattr(self, "_last_yaw_rate", 0.0)
            yr = max(prev_yr - max_yaw_accel,
                     min(prev_yr + max_yaw_accel, yr))
            self._last_yaw_rate = yr
            out["yaw_rate"] = yr
        elif execution_mode == module.ExecutionMode.ROTATE_ONLY:
            # Pure rotation toward the macro desired yaw.  Velocity is
            # exactly zero (section IX); no short forward advance is
            # emitted (the OBSERVE move is handled by the 5 Hz macro via a
            # SIDE/GUIDE re-plan, never by a hidden 0.3 m/s push).
            yaw_target = state["yaw"]
            if held_action is not None and held_action.has_desired_yaw:
                yaw_target = held_action.desired_yaw_world
            yaw_err = normalize_angle(yaw_target - yaw)
            yr = max(-max_yaw_rate, min(max_yaw_rate, yaw_gain * yaw_err))
            prev_yr = getattr(self, "_last_yaw_rate", 0.0)
            yr = max(prev_yr - max_yaw_accel, min(prev_yr + max_yaw_accel, yr))
            self._last_yaw_rate = yr
            out["yaw_rate"] = yr
            out["velocity_flu"] = np.zeros(3)
        elif execution_mode == module.ExecutionMode.BRAKE_HOLD:
            # Decelerate to hover.
            vel_flu = world_vector_to_body_flu_quat(state["velocity"], quat)
            decel = tc.get("brake_deceleration_mps2", 3.0) * self._dt
            speed = float(np.linalg.norm(vel_flu))
            if speed > decel:
                vel_flu = vel_flu * ((speed - decel) / speed)
            else:
                vel_flu = np.zeros(3)
            out["velocity_flu"] = vel_flu
            out["yaw_rate"] = 0.0
            self._last_yaw_rate = 0.0
        else:  # EMERGENCY_STOP
            # Emergency deceleration toward hover (stronger than brake).
            vel_flu = world_vector_to_body_flu_quat(state["velocity"], quat)
            decel = self._safety_cfg.get(
                "emergency_deceleration_mps2", 5.0) * self._dt
            speed = float(np.linalg.norm(vel_flu))
            if speed > decel:
                vel_flu = vel_flu * ((speed - decel) / speed)
            else:
                vel_flu = np.zeros(3)
            out["velocity_flu"] = vel_flu
            out["yaw_rate"] = 0.0
            self._last_yaw_rate = 0.0

        # The local search is intentionally planar at the task flight
        # slice.  Preserve that slice in every execution mode so a small
        # simulator drift during rotate/brake cannot become a permanent
        # lower planning height.
        altitude_cfg = tc.get("altitude_hold", {})
        if altitude_cfg.get("enabled", True):
            error = float(altitude_target_z - state["position"][2])
            if abs(error) <= float(altitude_cfg.get("deadband_m", 0.02)):
                error = 0.0
            max_vertical_speed = float(
                altitude_cfg.get("max_speed_mps", 0.6))
            vertical_world = float(np.clip(
                float(altitude_cfg.get("kp", 1.5)) * error -
                float(altitude_cfg.get("kd", 0.6)) *
                    float(state["velocity"][2]),
                -max_vertical_speed, max_vertical_speed))
            commanded_world = body_flu_vector_to_world_quat(
                out["velocity_flu"], quat)
            commanded_world[2] = vertical_world
            out["velocity_flu"] = world_vector_to_body_flu_quat(
                commanded_world, quat)

        # Final command clamp.
        speed = float(np.linalg.norm(out["velocity_flu"]))
        if speed > max_vel and speed > 1e-9:
            out["velocity_flu"] = out["velocity_flu"] * (max_vel / speed)
        out["velocity_flu"] = quantize_bounded_vector(
            out["velocity_flu"], max_vel)
        out["yaw_rate"] = float(np.clip(out["yaw_rate"], -max_yaw_rate,
                                        max_yaw_rate))
        final_velocity_world = body_flu_vector_to_world_quat(
            out["velocity_flu"], quat)
        previous_velocity_world = getattr(
            self, "_last_velocity_world", None)
        if execution_mode in (module.ExecutionMode.TRACK_FRESH,
                              module.ExecutionMode.TRACK_CACHED) and \
                self._last_execution_mode not in (
                    module.ExecutionMode.TRACK_FRESH,
                    module.ExecutionMode.TRACK_CACHED):
            previous_velocity_world = np.asarray(
                state["velocity"], dtype=np.float64)
        self._last_velocity_world = final_velocity_world
        self._last_acceleration_world = (
            np.zeros(3) if previous_velocity_world is None else
            (final_velocity_world - previous_velocity_world) / self._dt)
        self._last_execution_mode = execution_mode
        return out

    # ── Dataset row (section XIV) ────────────────────────────────────
    # `sim_elapsed` is the SIMULATION time axis (section "time"): recorded
    # as trajectory_time_s and used for plan_age_s.  The wall clock is NOT
    # written into any navigation time field.
    def _write_row(self, frame_id, sample_index, sim_elapsed, state, ds,
                   depth_student, raw_finite_ratio, latency_ms, held_action,
                   executed_result, planning_attempt, execution_mode,
                   fresh_plan, cached_used, cmd, plan_age_s):
        pos = state["position"]
        quat = ds.quaternion_world_body
        goal = self._current_task["goal"]
        goal_dist = float(np.linalg.norm(goal - pos))
        goal_dir_world = np.zeros(3)
        if goal_dist > 1e-6:
            goal_dir_world = (goal - pos) / goal_dist
        goal_dir_flu = world_vector_to_body_flu_quat(goal_dir_world, quat)
        gravity_flu = world_vector_to_body_flu_quat(
            np.array([0.0, 0.0, -1.0]), quat)

        guide_flu = np.zeros(3)
        guide_dir_flu = np.zeros(3)
        guide_dist = 0.0
        if held_action is not None and held_action.guide_world is not None:
            delta = held_action.guide_world - pos
            guide_dist = float(np.linalg.norm(delta))
            if guide_dist > 1e-6:
                guide_flu = world_vector_to_body_flu_quat(delta, quat)
                guide_dir_flu = guide_flu / guide_dist

        # Executed-plan semantics (section XVIII): local_terminal,
        # minimum_clearance, plan age and plan status all describe the
        # ACTIVE executed trajectory (fresh or cached), while the fresh
        # planning attempt status is kept as a pure diagnostic.
        executed_tracking = execution_mode in (
            module.ExecutionMode.TRACK_FRESH,
            module.ExecutionMode.TRACK_CACHED)
        executed_ok = executed_tracking and executed_result.success and \
            len(executed_result.trajectory) > 0
        local_terminal_flu = np.zeros(3)
        if executed_ok:
            terminal = executed_result.trajectory_terminal
            tdelta = terminal - pos
            if np.linalg.norm(tdelta) > 1e-9:
                local_terminal_flu = world_vector_to_body_flu_quat(tdelta, quat)

        # Privileged diagnostics (kept separate from student inputs).
        ds_cfg = self.g.get("dataset_logging", {})
        collect_priv = ds_cfg.get("collect_privileged_diagnostics", True)
        rec = self._macro_expert.last_recoverability
        blocker = self._macro_expert.last_blocker
        intervention = self._macro_expert.last_intervention
        local_recoverable = rec.status if rec is not None else -1
        blocker_signature = -1
        blocker_ray_depth = -1.0
        blocker_cell_count = 0
        blocker_track_id = -1
        left_edge = right_edge = left_corridor = right_corridor = 0
        if blocker is not None:
            blocker_signature = int(blocker.blocker_signature)
            blocker_ray_depth = float(blocker.blocking_ray_depth)
            blocker_cell_count = int(blocker.component_cell_count)
            left_edge = int(blocker.left_edge_visible)
            right_edge = int(blocker.right_edge_visible)
            left_corridor = int(blocker.left_corridor_known)
            right_corridor = int(blocker.right_corridor_known)
        blocker_track_id = self._macro_expert.blocker_track_id
        privileged_side, margin = self._oracle.privileged_best_side(
            self._macro_expert.last_candidates)
        # The decision margin is computed by the macro from the viable
        # candidate global costs (section VI).
        privileged_local_recoverable = \
            int(intervention.privileged_local_recoverable) \
            if intervention is not None else 1
        privileged_future_intervention_required = \
            int(intervention.privileged_future_intervention_required) \
            if intervention is not None else 0
        privileged_rejoin_reached = \
            int(intervention.privileged_rejoin_reached) \
            if intervention is not None else 0
        privileged_local_path_length = \
            float(intervention.privileged_local_path_length) \
            if intervention is not None else -1.0
        privileged_local_duration = \
            float(intervention.privileged_local_duration) \
            if intervention is not None else -1.0
        privileged_detour_ratio = \
            float(intervention.privileged_detour_ratio) \
            if intervention is not None else -1.0
        privileged_min_clearance = \
            float(intervention.privileged_min_clearance) \
            if intervention is not None else -1.0
        privileged_goal_progress = \
            float(intervention.privileged_goal_progress) \
            if intervention is not None else -1.0
        # Observed/privileged recoverability audit (section XXV): unified
        # rejoin capability bound + real-meter margins + terminal tangent.
        observed_rejoin_distance = \
            float(rec.rejoin_distance) if rec is not None else -1.0
        observed_path_length = \
            float(rec.path_length) if rec is not None else -1.0
        observed_detour_ratio = \
            float(rec.detour_ratio) if rec is not None else -1.0
        observed_terminal_alignment = \
            float(rec.terminal_guide_alignment) \
            if rec is not None else -1.0
        privileged_rejoin_distance = \
            float(intervention.privileged_rejoin_distance) \
            if intervention is not None else -1.0
        privileged_terminal_alignment = \
            float(intervention.privileged_terminal_alignment) \
            if intervention is not None else -1.0
        direct_no_progress_time = \
            float(self._macro_expert.direct_no_progress_time)
        observe_no_information_time = \
            float(self._macro_expert.observe_no_information_time)
        causal_intervention_evidence = \
            int(self._macro_expert.causal_intervention_evidence)
        macro_decision_observable = \
            int(self._macro_expert.macro_decision_observable)
        macro_decision_confidence = \
            float(self._macro_expert.macro_decision_confidence)
        global_ctg = self._oracle.cost_to_go(pos)
        global_clearance = self._oracle.clearance(pos)
        candidate_costs = json.dumps([
            {"type": int(c.type), "side": int(c.side),
             "score": round(c.privileged_score, 4),
             "ctg": round(c.global_cost_to_go, 3),
             "conn": int(c.connected_to_goal),
             "reach": int(c.known_reachable)}
            for c in self._macro_expert.last_candidates[:12]
        ])
        if not collect_priv:
            local_recoverable = -1
            blocker_signature = -1
            blocker_ray_depth = -1.0
            blocker_cell_count = 0
            blocker_track_id = -1
            left_edge = right_edge = left_corridor = right_corridor = 0
            privileged_side = 0
            margin = 0.0
            global_ctg = -1.0
            global_clearance = -1.0
            candidate_costs = ""
            privileged_local_recoverable = 1
            privileged_future_intervention_required = 0
            privileged_rejoin_reached = 0
            privileged_local_path_length = -1.0
            privileged_local_duration = -1.0
            privileged_detour_ratio = -1.0
            privileged_min_clearance = -1.0
            privileged_goal_progress = -1.0
            observed_rejoin_distance = -1.0
            observed_path_length = -1.0
            observed_detour_ratio = -1.0
            observed_terminal_alignment = -1.0
            privileged_rejoin_distance = -1.0
            privileged_terminal_alignment = -1.0
            direct_no_progress_time = 0.0
            observe_no_information_time = 0.0
            causal_intervention_evidence = 0
            macro_decision_observable = 1
            macro_decision_confidence = 0.0

        # Macro-interval feedback diagnostics (section XXVIII): recorded on
        # the macro tick frame with real aggregated values; other frames
        # record zeros with macro_feedback_is_new = 0 (section XXIX).
        fb = self._macro_feedback_log if self._macro_feedback_is_new else {}
        macro_feedback_is_new = int(self._macro_feedback_is_new)
        macro_interval_frame_count = int(fb.get("interval_frame_count", 0))
        macro_interval_planning_failures = int(
            fb.get("planning_failure_count", 0))
        macro_interval_cached_frames = int(fb.get("cached_frame_count", 0))
        macro_interval_brake_frames = int(fb.get("brake_frame_count", 0))
        macro_interval_emergency_frames = int(
            fb.get("emergency_frame_count", 0))
        macro_interval_local_unrecoverable_frames = int(
            fb.get("local_unrecoverable_count", 0))

        desired_yaw = held_action.desired_yaw_world if \
            held_action.has_desired_yaw else state["yaw"]
        yaw = state["yaw"]
        desired_yaw_delta = normalize_angle(desired_yaw - yaw)

        self._writer.write_depth(frame_id, depth_student, raw_finite_ratio)
        depth_file = self._writer.depth_file_for(frame_id)
        row = {
            "timestamp_ns": time.time_ns(),
            "receive_timestamp_ns": time.time_ns(),
            "episode_id": self._episode_id,
            "frame_id": frame_id,
            "episode_frame_index": sample_index,
            "control_dt_s": self._dt,
            "trajectory_time_s": sim_elapsed,
            "latency_ms": latency_ms,
            "match_method": "frame_id_exact",
            "frame_valid": 1,
            "frame_invalid_reason": "",
            "x": pos[0], "y": pos[1], "z": pos[2],
            "qx": quat[0], "qy": quat[1], "qz": quat[2], "qw": quat[3],
            "state_vx_world": state["velocity"][0],
            "state_vy_world": state["velocity"][1],
            "state_vz_world": state["velocity"][2],
            "state_vx_flu": ds.velocity_flu[0],
            "state_vy_flu": ds.velocity_flu[1],
            "state_vz_flu": ds.velocity_flu[2],
            "yaw": yaw,
            "yaw_rate": state["yaw_rate"],
            "gravity_dx_flu": gravity_flu[0],
            "gravity_dy_flu": gravity_flu[1],
            "gravity_dz_flu": gravity_flu[2],
            "depth_file": depth_file,
            "raw_depth_finite_ratio": raw_finite_ratio,
            "goal_direction_flu_x": goal_dir_flu[0],
            "goal_direction_flu_y": goal_dir_flu[1],
            "goal_direction_flu_z": goal_dir_flu[2],
            "goal_distance_m": goal_dist,
            "goal_distance_norm": min(1.0, goal_dist /
                                      ds_cfg.get("perception_range_m", 5.0)),
            "velocity_flu_x": ds.velocity_flu[0],
            "velocity_flu_y": ds.velocity_flu[1],
            "velocity_flu_z": ds.velocity_flu[2],
            # macro labels
            "macro_is_new_tick": int(held_action.is_new_tick),
            "macro_mode": int(held_action.mode),
            "macro_committed_side": int(held_action.committed_side),
            # P2 side-selection consistency diagnostics (pure diagnostics,
            # never student input): chosen vs committed vs privileged-best
            # side, per-side candidate FULL/connected counts and the last
            # side release reason.
            "macro_chosen_side": self._macro_expert.macro_chosen_side,
            "side_rejection_reason":
                self._macro_expert.side_rejection_reason,
            "side_candidate_full_left":
                self._macro_expert.side_candidate_full_left,
            "side_candidate_full_right":
                self._macro_expert.side_candidate_full_right,
            "side_candidate_connected_left":
                self._macro_expert.side_candidate_connected_left,
            "side_candidate_connected_right":
                self._macro_expert.side_candidate_connected_right,
            "macro_observe_side": int(
                getattr(held_action, "observe_side", 0)),
            "macro_confidence": held_action.confidence,
            "macro_decision_reason": held_action.reason,
            "macro_decision_observable": macro_decision_observable,
            "macro_decision_confidence": macro_decision_confidence,
            "macro_decision_margin": margin,
            "causal_intervention_evidence": causal_intervention_evidence,
            "macro_guide_world_x": held_action.guide_world[0] if
            held_action.guide_world is not None else 0.0,
            "macro_guide_world_y": held_action.guide_world[1] if
            held_action.guide_world is not None else 0.0,
            "macro_guide_world_z": held_action.guide_world[2] if
            held_action.guide_world is not None else 0.0,
            "macro_guide_flu_x": guide_flu[0],
            "macro_guide_flu_y": guide_flu[1],
            "macro_guide_flu_z": guide_flu[2],
            "macro_guide_direction_flu_x": guide_dir_flu[0],
            "macro_guide_direction_flu_y": guide_dir_flu[1],
            "macro_guide_direction_flu_z": guide_dir_flu[2],
            "macro_guide_distance_m": guide_dist,
            "desired_yaw_world": desired_yaw,
            "desired_yaw_delta": desired_yaw_delta,
            "desired_yaw_sin": math.sin(desired_yaw_delta),
            "desired_yaw_cos": math.cos(desired_yaw_delta),
            # local labels — EXECUTED plan semantics (section XVIII)
            "local_terminal_valid": int(executed_ok),
            "local_terminal_world_x": executed_result.trajectory_terminal[0]
            if executed_ok else 0.0,
            "local_terminal_world_y": executed_result.trajectory_terminal[1]
            if executed_ok else 0.0,
            "local_terminal_world_z": executed_result.trajectory_terminal[2]
            if executed_ok else 0.0,
            "local_terminal_flu_x": local_terminal_flu[0],
            "local_terminal_flu_y": local_terminal_flu[1],
            "local_terminal_flu_z": local_terminal_flu[2],
            "execution_mode": int(execution_mode),
            "velocity_command_flu_x": cmd["velocity_flu"][0],
            "velocity_command_flu_y": cmd["velocity_flu"][1],
            "velocity_command_flu_z": cmd["velocity_flu"][2],
            "yaw_rate_command": cmd["yaw_rate"],
            "fresh_plan": int(fresh_plan),
            "cached_plan_used": int(cached_used),
            "active_plan_is_fresh": int(fresh_plan),
            "active_plan_is_cached": int(cached_used),
            "planning_status": int(executed_result.status)
            if executed_tracking else -1,
            "minimum_clearance": executed_result.min_clearance
            if executed_tracking else -1.0,
            "trajectory_duration_s": executed_result.duration_s
            if executed_tracking else -1.0,
            "fresh_planning_status": int(planning_attempt.status),
            # privileged diagnostics
            "local_recoverable": local_recoverable,
            "blocker_signature": blocker_signature,
            "blocker_ray_depth": blocker_ray_depth,
            "blocker_cell_count": blocker_cell_count,
            "blocker_track_id": blocker_track_id,
            "left_edge_visible": left_edge,
            "right_edge_visible": right_edge,
            "left_corridor_known": left_corridor,
            "right_corridor_known": right_corridor,
            "privileged_best_side": int(privileged_side),
            "privileged_local_recoverable": privileged_local_recoverable,
            "privileged_rejoin_reached": privileged_rejoin_reached,
            "privileged_future_intervention_required":
                privileged_future_intervention_required,
            "privileged_local_path_length": privileged_local_path_length,
            "privileged_local_duration": privileged_local_duration,
            "privileged_detour_ratio": privileged_detour_ratio,
            "privileged_min_clearance": privileged_min_clearance,
            "privileged_goal_progress": privileged_goal_progress,
            "observed_rejoin_distance": observed_rejoin_distance,
            "observed_path_length": observed_path_length,
            "observed_detour_ratio": observed_detour_ratio,
            "observed_terminal_alignment": observed_terminal_alignment,
            "privileged_rejoin_distance": privileged_rejoin_distance,
            "privileged_terminal_alignment": privileged_terminal_alignment,
            "direct_no_progress_time": direct_no_progress_time,
            "observe_no_information_time": observe_no_information_time,
            # macro-interval feedback diagnostics (section XXVIII/XXIX)
            "macro_feedback_is_new": macro_feedback_is_new,
            "macro_interval_frame_count": macro_interval_frame_count,
            "macro_interval_planning_failures":
                macro_interval_planning_failures,
            "macro_interval_cached_frames": macro_interval_cached_frames,
            "macro_interval_brake_frames": macro_interval_brake_frames,
            "macro_interval_emergency_frames":
                macro_interval_emergency_frames,
            "macro_interval_local_unrecoverable_frames":
                macro_interval_local_unrecoverable_frames,
            "global_cost_to_go": global_ctg if np.isfinite(global_ctg) else -1.0,
            "global_clearance": global_clearance if
            np.isfinite(global_clearance) else -1.0,
            "global_candidate_costs": candidate_costs,
            # goal / plan bookkeeping — executed-plan semantics (XVII)
            "goal_world_x": goal[0], "goal_world_y": goal[1], "goal_world_z": goal[2],
            "distance_to_final_goal": goal_dist,
            "plan_id": executed_result.plan_id if executed_tracking else -1,
            "plan_age_s": plan_age_s if executed_tracking else -1.0,
            "plan_is_fresh": int(fresh_plan),
            "plan_status": int(executed_result.status)
            if executed_tracking else -1,
            "plan_compute_ms": executed_result.planning_time_ms
            if executed_tracking else -1.0,
            "scene_id": self._dataset_scene_key,
            "task_id": self._current_task_id,
            "episode_valid": 1,
            # P5 failure taxonomy + P1 stale-plan diagnostics (pure
            # diagnostics, never student input): the documented episode
            # failure category, the last safety-critical plan failure
            # status and how many times a stale cached plan was invalidated.
            "failure_taxonomy": self._failure_taxonomy,
            "critical_plan_failure_status": self._last_plan_failure_status,
            "stale_plan_invalidations": self._stale_plan_invalidations,
            # Active-observation diagnostics (section XLVII-XLIX): pure
            # diagnostics, never part of any student input.  Held from the
            # last 5 Hz macro tick.
            "observe_scan_side": self._macro_expert.observe_scan_side,
            "left_scan_exhausted": self._macro_expert.left_scan_exhausted,
            "right_scan_exhausted": self._macro_expert.right_scan_exhausted,
            "observe_rotation_exhausted":
                self._macro_expert.observe_rotation_exhausted,
            "observe_stagnant_rotate_time":
                self._macro_expert.observe_stagnant_rotate_time,
            "observe_raw_candidate_count":
                self._macro_expert.observe_raw_candidate_count,
            "observe_lattice_candidate_count":
                self._macro_expert.observe_lattice_candidate_count,
            "observe_frontier_candidate_count":
                self._macro_expert.observe_frontier_candidate_count,
            "observe_endpoint_known_free_count":
                self._macro_expert.observe_endpoint_known_free_count,
            "observe_local_full_count":
                self._macro_expert.observe_local_full_count,
            # P3 recovery diagnostics (pure diagnostics, never student
            # input): forward vs known-free-retreat FULL viewpoint counts
            # and whether the active OBSERVE move is a retreat.
            "observe_forward_full_count":
                self._macro_expert.observe_forward_full_count,
            "observe_retreat_full_count":
                self._macro_expert.observe_retreat_full_count,
            "observe_retreat_candidate_count":
                self._macro_expert.observe_retreat_candidate_count,
            "observe_recovery_active":
                self._macro_expert.observe_recovery_active,
            "observe_reject_unknown": self._macro_expert.observe_reject_unknown,
            "observe_reject_endpoint_clearance":
                self._macro_expert.observe_reject_endpoint_clearance,
            "observe_reject_min_distance":
                self._macro_expert.observe_reject_min_distance,
            "observe_reject_max_distance":
                self._macro_expert.observe_reject_max_distance,
            "observe_reject_partial": self._macro_expert.observe_reject_partial,
            "observe_reject_no_path": self._macro_expert.observe_reject_no_path,
            "observe_left_valid_count":
                self._macro_expert.observe_left_valid_count,
            "observe_right_valid_count":
                self._macro_expert.observe_right_valid_count,
            "observe_center_valid_count":
                self._macro_expert.observe_center_valid_count,
            "observe_selected_source":
                self._macro_expert.observe_selected_source,
            "observe_selected_side": self._macro_expert.observe_selected_side,
            "observe_selected_distance":
                self._macro_expert.observe_selected_distance,
            "observe_selected_path_length":
                self._macro_expert.observe_selected_path_length,
            "observe_selected_info_gain":
                self._macro_expert.observe_selected_info_gain,
            "observe_selected_clearance":
                self._macro_expert.observe_selected_clearance,
        }
        self._writer.write_row(row)
        if self.g.get("dataset_logging", {}).get(
                "write_local_plan_points", True) and executed_ok:
            self._writer.write_local_plan(
                executed_result.plan_id, frame_id,
                str(executed_result.status), executed_result.success,
                executed_result.planning_time_ms,
                executed_result.min_clearance, executed_result.duration_s,
                executed_result.guide_waypoint,
                executed_result.trajectory_terminal,
                executed_result.search_status, executed_result.trajectory)

    # ── End of episode / task / scene (sections L-LII) ───────────────
    def _st_finish_task(self):
        self._total_episodes += 1
        # Sync-quality gates for the commit decision (section "sync").
        sync = self._sync_cfg
        total = max(1, self._matched_frames + self._unmatched_frames)
        unmatched_pct = 100.0 * self._unmatched_frames / total
        latency_violation_pct = 100.0 * sum(
            1 for ms in self._frame_latencies
            if ms > sync.get("max_acceptable_latency_ms", 250.0)) / \
            max(1, len(self._frame_latencies))
        catastrophic = any(
            ms > sync.get("catastrophic_latency_ms", 5000.0)
            for ms in self._frame_latencies)
        mean_finite_ratio = float(np.mean(self._frame_finite_ratios)) if \
            self._frame_finite_ratios else 0.0
        none_depth_pct = 100.0 * (1.0 - mean_finite_ratio)

        reject_reason = ""
        if unmatched_pct > sync.get("max_unmatched_pct", 1.0):
            reject_reason = "excessive_unmatched_frames"
        elif latency_violation_pct > sync.get("max_latency_violation_pct", 1.0):
            reject_reason = "excessive_latency_violations"
        elif catastrophic:
            reject_reason = "catastrophic_latency"
        elif none_depth_pct > sync.get("max_none_depth_pct", 20.0):
            reject_reason = "excessive_none_depth"

        success = self._trajectory_reached_goal and not reject_reason
        if not success and not reject_reason:
            reject_reason = self._trajectory_exit_reason
        extra = {
            "exit_reason": self._trajectory_exit_reason,
            "reject_reason": reject_reason,
            "failure_taxonomy": self._failure_taxonomy,
            "critical_plan_failure_status": self._last_plan_failure_status,
            "stale_plan_invalidations": self._stale_plan_invalidations,
            "total_plans": self._total_plans,
            "successful_plans": self._successful_plans,
            "reached_goal": self._trajectory_reached_goal,
            "unmatched_pct": unmatched_pct,
            "latency_violation_pct": latency_violation_pct,
            "none_depth_pct": none_depth_pct,
            "quality_committed": success,
        }
        # The dataset commit must ACTUALLY succeed before an episode may
        # count as committed training data (sections XXII-XXIII): a write /
        # rename failure is a FAILED outcome, never COMMITTED.
        write_ok = False
        try:
            final_dir = self._writer.finish(success, reject_reason, extra)
            write_ok = final_dir is not None
        except Exception as exc:  # noqa: BLE001
            rospy.logerr("[Manager] dataset write failed: %s", exc)
            write_ok = False
        committed = bool(success and write_ok)
        if committed:
            self._committed_episodes += 1
        else:
            self._failed_tasks += 1

        # Task manifest output + runtime classification (section XLVII).
        # `initial_observed_recoverable` is the value FROZEN at the first
        # valid observed evaluation (sections XXXVI/LXV) — never the last
        # episode recoverability.  None is recorded as null (not False)
        # when no valid initial evaluation was ever obtained (section
        # XXXVII).
        task = self._current_tasks[self._task_index]
        runtime_info = {
            "episode_result": "success" if committed else "failed",
            "exit_reason": self._trajectory_exit_reason,
            "reject_reason": reject_reason,
            "failure_taxonomy": self._failure_taxonomy,
            "dataset_write_ok": write_ok,
        }
        runtime_cls = None
        if self._runtime_classify_enabled:
            initial_observed = self._initial_observed_recoverable
            runtime_info["initial_observed_recoverable"] = initial_observed
            runtime_info["initial_observed_recoverability_status"] = \
                int(self._initial_observed_recoverability_status) if \
                self._initial_observed_recoverability_status is not None \
                else None
            runtime_info["initial_observed_recoverability_tick"] = \
                self._initial_observed_recoverability_tick
            runtime_cls = runtime_classify(
                task.target_task_class,
                bool(task.metrics.get("privileged_local_recoverable")),
                initial_observed)
            runtime_info["runtime_task_class"] = \
                runtime_cls.value if runtime_cls is not None else None
            runtime_info["classification_mismatch"] = bool(
                runtime_cls is not None and
                runtime_cls != task.target_task_class)
        # Quota accounting (sections XXXIII-XLI) through the single task
        # resolver: COMMITTED only after the writer commit succeeded;
        # otherwise FAILED.  pending is released under the scheduled
        # (target) class; the committed credit uses the runtime class with
        # the target class as fallback (sections VI/VII/XXIV).
        if committed:
            committed_cls = (runtime_cls if runtime_cls is not None
                             else task.target_task_class)
            self._resolve_current_task_quota(
                TaskOutcome.COMMITTED, committed_class=committed_cls)
            runtime_info["committed_class"] = committed_cls.value
        else:
            self._resolve_current_task_quota(TaskOutcome.FAILED)
            runtime_info["committed_class"] = None
        task_dir = os.path.join(self._scene_dir, "tasks", task.task_id)
        TaskManifestWriter.write(
            os.path.join(task_dir, "task_manifest.json"),
            task.to_dict(), runtime_info)
        if not committed:
            GenerationFailureWriter.write(self._generation_failures_path, {
                "event": "task_failed",
                "task_id": task.task_id,
                "scene_key": self._dataset_scene_key,
                "exit_reason": self._trajectory_exit_reason,
                "reject_reason": reject_reason,
                "failure_taxonomy": self._failure_taxonomy,
                "dataset_write_ok": write_ok,
            })

        self._episode_index += 1
        self._task_index += 1
        if self._task_index < len(self._current_tasks):
            # Next task on the SAME scene (section LII): no scene re-send,
            # no point cloud re-export, no map rebuild.
            self._enter_state(State.RESET_DRONE)
        else:
            self._enter_state(State.FINISH_SCENE)

    def _st_finish_scene(self):
        rospy.loginfo("[Manager] Scene complete: %s (%d tasks)",
                      self._dataset_scene_key, len(self._current_tasks))
        # Scene invariant (section XXX): every task of this scene must be
        # resolved exactly once.  Any leftover is a lifecycle bug — cancel
        # it here so no phantom pending leaks into the next scene.
        unresolved = [t.task_id for t in self._current_tasks
                      if not t.quota_resolved]
        if unresolved:
            rospy.logerr(
                "[Manager] Scene %s: %d unresolved quota tasks: %s",
                self._dataset_scene_key, len(unresolved), unresolved)
            for t in self._current_tasks:
                if not t.quota_resolved:
                    try:
                        self._quota.note_cancelled(t.target_task_class)
                    except RuntimeError as exc:
                        rospy.logerr(
                            "[Manager] quota cancel error for %s: %s",
                            t.task_id, exc)
                    t.quota_resolved = True
        self._scene_index += 1
        self._scene_attempt = 0
        self._current_tasks = []
        self._current_scene = None
        self._episode_index = 0
        self._enter_state(State.NEXT_SCENE)

    def _st_next_scene(self):
        self._enter_state(State.GENERATE_SCENE)

    def _st_next_profile(self):
        # _st_generate_scene advanced the profile index; loop back to the
        # generation entry which re-checks completion.
        self._enter_state(State.GENERATE_SCENE)


def main():
    rospy.init_node("il_dataset_manager")
    # Dump the PYTHON stack on a native crash (SIGABRT/SIGSEGV, e.g. a C++
    # heap corruption like "double free or corruption").  This pinpoints the
    # exact Python call site that led into the aborting C++ module, which is
    # essential for native debugging of the local planner.
    try:
        import faulthandler
        faulthandler.enable()
    except Exception:  # noqa: BLE001
        pass
    cfg = load_config()
    # The launch file always sets the ~dry_run param (default false), so
    # we must check its VALUE, not mere existence (has_param would always
    # be true and the manager would exit in dry-run every time).
    if rospy.get_param("~dry_run", False):
        g = cfg.get("global", {})
        sg = g.get("scene_generation", {})
        tg = g.get("task_generation", {})
        total_scenes = sum(
            int(p.get("scene_count", 20)) for p in sg.get("profiles", []))
        weights = tg.get("class_weights", {})
        rospy.loginfo(
            "[Manager] Config loaded (dry-run): profiles=%d scenes=%d "
            "tasks_per_scene=%d class_weights=%s",
            len(sg.get("profiles", [])), total_scenes,
            sg.get("tasks_per_scene", 12), weights)
        return
    manager = ILManager(cfg)
    manager.run()


if __name__ == "__main__":
    main()
