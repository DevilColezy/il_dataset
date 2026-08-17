#!/usr/bin/env python3
"""Validate collection-only manifest loading (generation/collection split).

Mocks ROS / pybind, builds a synthetic blueprint manifest JSON (as written
by _write_blueprint_manifest), loads it via the REAL
JointV2Manager._load_collection_manifest and checks that the rebuilt tasks
/ scenes / preflighted pool expose the same fields used by the collection
loop and the committed top-up scoring.
"""
import json
import os
import sys
import tempfile
import types
from types import SimpleNamespace

# ── Mock ROS / project modules before importing il_manager ────────────
def _nolog(*a, **k):
    pass


rospy = types.ModuleType("rospy")
rospy.loginfo = _nolog
rospy.logwarn = _nolog
rospy.logerr = _nolog
rospy.logfatal = _nolog
rospy.init_node = _nolog
rospy.ROSInterruptException = type("ROSInterruptException", (Exception,), {})
rospy.is_shutdown = lambda: False
rospy.has_param = lambda *a: False
rospy.get_param = lambda *a, **k: None
sys.modules["rospy"] = rospy

il_config = types.ModuleType("il_config")
il_config.load_config = lambda *a, **k: {"global": {}}
sys.modules["il_config"] = il_config

il_common = types.ModuleType("il_common")


class UnityBridge(object):
    pass


il_common.UnityBridge = UnityBridge
il_common.world_vector_to_body_flu_quat = lambda *a, **k: (0.0, 0.0, 0.0)
sys.modules["il_common"] = il_common

il_expert_config = types.ModuleType("il_expert_config")
il_expert_config.build_params = lambda *a, **k: (None, [])
il_expert_config.build_scene_bounds = lambda *a, **k: ([0.0, 0.0], [1.0, 1.0])
sys.modules["il_expert_config"] = il_expert_config

il_dataset_writer = types.ModuleType("il_dataset_writer")


class DatasetWriter(object):
    pass


il_dataset_writer.DatasetWriter = DatasetWriter
sys.modules["il_dataset_writer"] = il_dataset_writer

expert_mod = types.ModuleType("_il_hierarchical_expert")


class HierarchicalExpert(object):
    pass


class TruthCylinderAudit(object):
    pass


class SceneTaskBlueprintGenerator(object):
    pass


expert_mod.HierarchicalExpert = HierarchicalExpert
expert_mod.TruthCylinderAudit = TruthCylinderAudit
expert_mod.SceneTaskBlueprintGenerator = SceneTaskBlueprintGenerator
sys.modules["_il_hierarchical_expert"] = expert_mod

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import il_manager  # noqa: E402

M = il_manager.JointV2Manager
passed = 0


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError("FAIL: " + name)
    passed += 1
    print("ok -", name)


# ── Build a synthetic manifest payload exactly like the writer's ──────
def make_task(tid, sid):
    return {
        "scene_id": sid, "task_id": tid, "seed": 1000 + tid,
        "start": [float(tid), 2.0], "goal": [float(tid + 5), 2.0],
        "initial_yaw": 0.0, "flight_height_m": 2.0,
        "behavior_class": "chicane", "density_class": "medium",
        "radius_class": "medium", "distance_class": "medium",
        "side_class": "both",
        "audit": {"accepted": True, "preflight_ticks": 400,
                  "min_truth_clearance_m": 0.62, "goal_distance_m": 5.0,
                  "preflight_status": "ok"},
        "summary": {
            "macro5hz": {
                "tick_total": 60, "pass_count": 40, "normal_count": 10,
                "turn_left_count": 5, "turn_right_count": 5,
                "correction_angle_hist": [0, 0, 0, 0, 0, 0, 0, 10, 0, 0],
                "correction_distance_hist": [0, 0, 10, 0, 0],
            },
            "local30hz": {
                "direct_count": 20, "avoidance_count": 40,
                "deflection_hist": [0, 0, 10, 10, 5, 0, 0],
                "yaw_rate_hist": [0, 0, 0, 0, 40, 0, 0, 0, 0],
                "speed_hist": [5, 5, 5, 5, 5, 5],
            },
        },
    }


payload = {
    "generation_ok": True,
    "scenes": [{"scene_id": 1, "obstacles": [
        {"id": 0, "x": 1.0, "y": 2.0, "radius": 0.5, "height_m": 8.0}]}],
    "tasks": [make_task(1, 1), make_task(2, 1)],
    "preflighted": [make_task(1, 1), make_task(2, 1), make_task(3, 1)],
}

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "joint_v2_blueprint_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    m = object.__new__(M)
    m._g = {}
    plan = m._load_collection_manifest(path)

    check("tasks loaded", len(plan.tasks) == 2)
    check("scenes loaded", len(plan.scenes) == 1)
    check("preflighted loaded", len(plan.preflighted) == 3)
    check("task fields", plan.tasks[0].start_x == 1.0 and
          plan.tasks[0].goal_x == 6.0 and plan.tasks[0].flight_height_m == 2.0)
    check("task audit", plan.tasks[0].audit.accepted is True and
          abs(plan.tasks[0].audit.min_truth_clearance_m - 0.62) < 1e-9)
    check("scene obstacles", plan.scenes[0].obstacles[0].radius == 0.5)
    check("summary macro counts",
          plan.tasks[0].summary.macro_pass_count == 40 and
          plan.tasks[0].summary.macro_turn_left_count == 5)
    check("summary hist total",
          plan.tasks[0].summary.macro_correction_angle_hist.total() == 10.0)
    check("summary hist counts",
          plan.tasks[0].summary.local_deflection_hist.counts[2] == 10)

    # top-up scoring against the loaded pool (uses summary histograms).
    cfg = {"mtc": 24,
           "correction_angle_edges": [-90, -60, -45, -30, -15, 0, 15, 30, 45,
                                      60, 90],
           "correction_distance_edges": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
           "deflection_edges": [-90, -60, -30, -10, 10, 30, 60, 90],
           "yaw_rate_edges": [-2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2],
           "speed_edges": [0, 0.5, 1, 1.5, 2, 2.5, 3],
           "min_deflection_speed_mps": 0.10}
    targets = m._gap_targets(cfg)
    stats = m._new_committed_stats(cfg)
    gaps = m._evaluate_gaps(stats, targets)
    check("gaps from empty stats", len(gaps) > 0)
    # The task that contributes to corr_angle bin7 should score > 0.
    score = m._score_task_for_gaps(plan.tasks[0], stats, targets)
    check("loaded task scores against gaps", score > 0.0)
    contrib = m._task_contribution(plan.tasks[0],
                                   "hist_bin:macro_correction_angle:7")
    check("loaded task hist contribution", contrib == 10.0)

print("\nALL %d CHECKS PASSED" % passed)
