#!/usr/bin/env python3
"""Unit tests for atomic scene/task quota acceptance in il_manager."""

import ast
import os
import unittest


_MANAGER_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "scripts", "il_manager.py"))


class _State:
    GENERATE_OBSTACLE_CONFIG = "GENERATE_OBSTACLE_CONFIG"
    ERROR = "ERROR"


class _RosLog:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def logwarn(self, *args):
        self.warnings.append(args)

    def logerr(self, *args):
        self.errors.append(args)


class _FailureManifestWriter:
    def __init__(self):
        self.calls = []

    def write_failure_manifest(self, *args):
        self.calls.append(args)


def _load_isolated_manager_method(rospy_stub):
    """Compile the real method without importing ROS/Flightmare modules."""
    with open(_MANAGER_PATH, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=_MANAGER_PATH)
    manager_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ILManager")
    method_node = next(
        node for node in manager_node.body
        if isinstance(node, ast.FunctionDef) and
        node.name == "_reject_current_scene_for_task_failure")
    harness = ast.ClassDef(
        name="Harness", bases=[], keywords=[], body=[method_node],
        decorator_list=[])
    module = ast.fix_missing_locations(
        ast.Module(body=[harness], type_ignores=[]))
    namespace = {"State": _State, "rospy": rospy_stub}
    exec(compile(module, _MANAGER_PATH, "exec"), namespace)
    return namespace["Harness"]


def _make_harness(profile_mode, current_attempt, max_attempts):
    ros_log = _RosLog()
    cls = _load_isolated_manager_method(ros_log)
    manager = cls()
    manager._use_profile_mode = profile_mode
    manager._current_scene_attempt = current_attempt
    manager._scene_generator = type(
        "SceneGenerator", (), {"max_scene_attempts": max_attempts})()
    manager._current_profile_name = "profile_a"
    manager._scene_index_in_profile = 2
    manager._scene_profile_index = 1
    manager._current_effective_scene_seed = 123
    manager._scene_generation_retry_offset = 0
    manager._failure_manifest_writer = _FailureManifestWriter()
    manager.g = {
        "scene_generation": {"fixed_scene_name": "fixed_a"}}
    manager.entered_states = []
    manager._enter_state = manager.entered_states.append
    return manager, ros_log


class ManagerTaskQuotaRetryTest(unittest.TestCase):
    def test_density_scene_retries_with_next_layout_attempt(self):
        manager, ros_log = _make_harness(True, 1, 3)
        manager._reject_current_scene_for_task_failure(
            "TASK_POSTFILTER_QUOTA_UNAVAILABLE", "retained 7 of 12")

        self.assertEqual(manager._scene_generation_retry_offset, 1)
        self.assertEqual(
            manager.entered_states, [_State.GENERATE_OBSTACLE_CONFIG])
        self.assertEqual(manager._failure_manifest_writer.calls, [])
        self.assertEqual(len(ros_log.warnings), 1)

    def test_exhausted_density_scene_writes_manifest_and_errors(self):
        manager, ros_log = _make_harness(True, 3, 3)
        manager._reject_current_scene_for_task_failure(
            "TASK_POSTFILTER_QUOTA_UNAVAILABLE", "retained 7 of 12")

        self.assertEqual(manager.entered_states, [_State.ERROR])
        self.assertEqual(len(manager._failure_manifest_writer.calls), 1)
        call = manager._failure_manifest_writer.calls[0]
        self.assertEqual(call[:4], ("profile_a", 1, 2, 123))
        self.assertEqual(
            call[4],
            "TASK_POSTFILTER_QUOTA_UNAVAILABLE:retained 7 of 12")
        self.assertEqual(call[5], 3)
        self.assertEqual(len(ros_log.errors), 1)

    def test_fixed_scene_fails_without_retry(self):
        manager, ros_log = _make_harness(False, 1, 3)
        manager._reject_current_scene_for_task_failure(
            "TASK_POSTFILTER_QUOTA_UNAVAILABLE", "retained 0 of 1")

        self.assertEqual(manager._scene_generation_retry_offset, 0)
        self.assertEqual(manager.entered_states, [_State.ERROR])
        self.assertEqual(manager._failure_manifest_writer.calls, [])
        self.assertEqual(len(ros_log.errors), 1)

    def test_postfilter_quota_check_uses_shared_failure_path(self):
        with open(_MANAGER_PATH, "r", encoding="utf-8") as stream:
            tree = ast.parse(stream.read(), filename=_MANAGER_PATH)
        manager_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ILManager")
        method_node = next(
            node for node in manager_node.body
            if isinstance(node, ast.FunctionDef) and
            node.name == "_st_generate_start_goal_pairs")
        reason_codes = []
        for node in ast.walk(method_node):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "_reject_current_scene_for_task_failure":
                continue
            if node.args and isinstance(node.args[0], (ast.Str, ast.Constant)):
                reason_codes.append(getattr(
                    node.args[0], "value", getattr(node.args[0], "s", None)))
        self.assertIn("TASK_COVERAGE_UNAVAILABLE", reason_codes)
        self.assertIn("TASK_POSTFILTER_QUOTA_UNAVAILABLE", reason_codes)


if __name__ == "__main__":
    unittest.main()
