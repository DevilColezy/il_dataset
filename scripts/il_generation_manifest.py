#!/usr/bin/env python3
"""il_generation_manifest.py  —  scene / task / failure manifest writers.

Manifests are OUTPUTS for reproducibility, debug and replay — they are
never required inputs for collection (sections VI/XLVI-XLVII/LXXIV).
"""

from __future__ import print_function, division

import json
import os


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d)


class SceneManifestWriter(object):
    @staticmethod
    def write(path, scene_dict, tasks_dicts, pointcloud_path, task_seed,
              unity_scene_id=None):
        record = dict(scene_dict)
        record["pointcloud_path"] = pointcloud_path
        record["task_seed_base"] = int(task_seed)
        # Both identifiers are recorded (section XI): `scene_key` is the
        # dataset-internal scene name; `unity_scene_id` is the numeric
        # AvoidBench wire identifier.  Only `scene_key` identifies the
        # scene inside the dataset.
        record["unity_scene_id"] = unity_scene_id
        record["tasks"] = tasks_dicts
        _ensure_dir(path)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)


class TaskManifestWriter(object):
    @staticmethod
    def write(path, task_dict, runtime_info=None):
        record = dict(task_dict)
        if runtime_info:
            record.update(runtime_info)
        _ensure_dir(path)
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)


class GenerationFailureWriter(object):
    """Appends one failure record per event (section XLIX/LI)."""

    @staticmethod
    def write(path, record):
        _ensure_dir(path)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
