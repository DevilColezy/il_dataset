#!/usr/bin/env python3
"""il_task_generator.py  —  multi-scale start-goal task generation.

Task generation runs AFTER the scene's real point cloud is exported and
the privileged scene map is built (section XXVII).  Every candidate
start-goal is validated in C++ (`TaskGenerationOracle.evaluate_candidates`)
against the REAL exported geometry — Python only samples candidates,
classifies them into behavioural TaskClasses and balances the dataset
quota (sections XXVIII/XXXVII/XXXIX).

TaskClass (section X) is NEVER a student input and NEVER forces the macro
expert's mode (section XVIII).  It is only used for generation targets,
dataset balancing and diagnostics.
"""

from __future__ import print_function, division

import math
import random
import zlib
from enum import Enum

import numpy as np


class TaskClass(Enum):
    OPEN_DIRECT = "open_direct"
    LOCAL_REACTIVE = "local_reactive"
    NEAR_OCCLUDED = "near_occluded"
    STRATEGIC_BLOCKER = "strategic_blocker"
    ASYMMETRIC_SIDE = "asymmetric_side"
    AMBIGUOUS_SIDE = "ambiguous_side"
    COMPOUND_BARRIER = "compound_barrier"


class GeneratedTask(object):
    def __init__(self, task_id, scene_id, start, goal, initial_yaw,
                 target_task_class, task_seed, metrics):
        self.task_id = str(task_id)
        self.scene_id = str(scene_id)
        self.start = np.asarray(start, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.initial_yaw = float(initial_yaw)
        self.target_task_class = target_task_class  # TaskClass
        self.task_seed = int(task_seed)
        self.metrics = metrics or {}

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "scene_id": self.scene_id,
            "start": self.start.tolist(),
            "goal": self.goal.tolist(),
            "initial_yaw": round(self.initial_yaw, 5),
            "target_task_class": self.target_task_class.value,
            "task_seed": self.task_seed,
            "metrics": self.metrics,
        }


class DatasetQuota(object):
    """Dataset-level balancing across TaskClasses (section LXII)."""

    def __init__(self, class_weights):
        self.weights = {}
        for name, w in (class_weights or {}).items():
            try:
                cls = TaskClass(name)
            except ValueError:
                continue
            self.weights[cls] = float(w)
        for cls in TaskClass:
            self.weights.setdefault(cls, 0.0)
        self.accepted = {cls: 0 for cls in TaskClass}
        self.attempted = {cls: 0 for cls in TaskClass}

    def note_attempted(self, cls):
        self.attempted[cls] += 1

    def note_accepted(self, cls):
        self.accepted[cls] += 1

    def need(self, cls):
        total = sum(self.accepted.values())
        expected = self.weights[cls] * max(1, total)
        return max(0.0, expected - self.accepted[cls])

    def needs_class(self, cls):
        return self.need(cls) > 0.0

    def summary(self):
        return {
            "accepted": {c.value: n for c, n in self.accepted.items()},
            "attempted": {c.value: n for c, n in self.attempted.items()},
        }


class MultiscaleTaskGenerator(object):
    def __init__(self, cfg, module, task_oracle_cfg, task_gen_oracle):
        self.cfg = cfg or {}
        self._module = module
        self._oracle_cfg = task_oracle_cfg or {}
        self._gen = task_gen_oracle  # C++ TaskGenerationOracle
        self.region_min = np.asarray(self.cfg.get("region_min",
                                                  [1.5, 16.0, 3.5]),
                                     dtype=np.float64)
        self.region_max = np.asarray(self.cfg.get("region_max",
                                                  [28.5, 60.0, 11.5]),
                                     dtype=np.float64)
        self.flight_height_m = float(self.cfg.get("flight_height_m", 5.0))
        bands = self.cfg.get("distance_bands", {}) or {}
        self.distance_bands = []
        for name, b in bands.items():
            self.distance_bands.append((name, float(b.get("min_m", 4.0)),
                                        float(b.get("max_m", 8.0))))
        sampling = self.cfg.get("sampling", {}) or {}
        self.candidate_batch_size = int(sampling.get("candidate_batch_size", 64))
        self.maximum_batches_per_scene = int(
            sampling.get("maximum_batches_per_scene", 6))
        cl = self.cfg.get("classification", {}) or {}
        self.near_occluded_max_blocker_m = float(
            cl.get("near_occluded_max_blocker_distance_m", 2.0))
        self.compound_min_blockers = int(cl.get("compound_min_blockers", 2))
        self.ambiguous_max_path_diff_m = float(
            cl.get("ambiguous_max_path_diff_m", 1.0))
        iy = self.cfg.get("initial_yaw", {}) or {}
        self.goal_aligned_probability = float(iy.get("goal_aligned_probability", 0.5))
        self.random_offset_deg = float(iy.get("random_offset_deg", 60.0))
        self.task_seed_base = int(self.cfg.get("task_seed_base", 999983))

    # ── Candidate sampling ───────────────────────────────────────────
    def _sample_point(self, rng, z):
        x = rng.uniform(self.region_min[0], self.region_max[0])
        y = rng.uniform(self.region_min[1], self.region_max[1])
        return np.array([x, y, z], dtype=np.float64)

    def _sample_batch(self, rng, band_min, band_max):
        n = self.candidate_batch_size
        starts = []
        goals = []
        z = self.flight_height_m
        for _ in range(n):
            start = self._sample_point(rng, z)
            # Rejection-sample a goal in the requested distance band.
            for _ in range(40):
                goal = self._sample_point(rng, z)
                d = float(np.linalg.norm(goal - start))
                if band_min <= d <= band_max:
                    break
            starts.append(start)
            goals.append(goal)
        return starts, goals

    # ── Classification (behavioural scale, section X-XVIII) ─────────
    def classify(self, r):
        """Map a C++ TaskCandidateResult to a TaskClass (or None)."""
        if not r.goal_reachable or not r.start_free or not r.goal_free:
            return None
        if r.reason != "ok":
            return None
        if not r.direct_blocked:
            return TaskClass.OPEN_DIRECT
        if r.privileged_local_recoverable:
            if r.nearest_blocker_distance_m >= 0.0 and \
                    r.nearest_blocker_distance_m <= self.near_occluded_max_blocker_m:
                return TaskClass.NEAR_OCCLUDED
            return TaskClass.LOCAL_REACTIVE
        # Not locally recoverable -> strategic scale.
        if r.direct_blocker_count >= self.compound_min_blockers:
            return TaskClass.COMPOUND_BARRIER
        left = r.left_global_feasible
        right = r.right_global_feasible
        if left and right:
            diff = abs(float(r.left_path_length) - float(r.right_path_length))
            if diff <= self.ambiguous_max_path_diff_m:
                return TaskClass.AMBIGUOUS_SIDE
            return TaskClass.ASYMMETRIC_SIDE
        if left != right:
            return TaskClass.ASYMMETRIC_SIDE
        return TaskClass.STRATEGIC_BLOCKER

    def _initial_yaw(self, start, goal, rng):
        """Goal-aligned with probability, else a bounded random offset
        (section XLVIII)."""
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        goal_yaw = math.atan2(dy, dx) - 0.5 * math.pi
        if rng.random() < self.goal_aligned_probability:
            return goal_yaw
        offset = math.radians(rng.uniform(-self.random_offset_deg,
                                          self.random_offset_deg))
        return goal_yaw + offset

    # ── Main entry ───────────────────────────────────────────────────
    def generate_tasks(self, scene, oracle, quota, rng, tasks_per_scene):
        """Generate up to `tasks_per_scene` tasks on the built scene map.

        Returns (tasks, generation_note).  A task is only accepted when its
        endpoints are free, globally reachable and classified; under-sampled
        classes are prioritised (dataset-level quota, section LXII).
        """
        tasks = []
        coord_candidates = []
        for batch_i in range(self.maximum_batches_per_scene):
            band = self.distance_bands[
                batch_i % len(self.distance_bands)] if self.distance_bands \
                else (None, 4.0, 28.0)
            _, bmin, bmax = band
            starts, goals = self._sample_batch(rng, bmin, bmax)
            starts_arr = np.asarray(starts, dtype=np.float64)
            goals_arr = np.asarray(goals, dtype=np.float64)
            results = self._gen.evaluate_candidates(oracle, starts_arr,
                                                    goals_arr)
            for (r, s, g) in zip(results, starts, goals):
                cls = self.classify(r)
                if cls is None:
                    continue
                quota.note_attempted(cls)
                coord_candidates.append((cls, r, s, g))
            if len(coord_candidates) >= 4 * tasks_per_scene:
                break

        # Priority: quota-deficient classes first, then lower detour and
        # higher clearance (section LXII).
        coord_candidates.sort(
            key=lambda t: self._candidate_quality(t[0], t[1], quota),
            reverse=True)

        used_rng = random.Random(
            (self.task_seed_base +
             zlib.crc32(scene.scene_id.encode("utf-8"))) & 0x7FFFFFFF)
        for cls, r, s, g in coord_candidates:
            if len(tasks) >= tasks_per_scene:
                break
            quota.note_accepted(cls)
            task_id = "%s_task_%03d" % (scene.scene_id, len(tasks))
            task_seed = (self.task_seed_base + len(tasks) * 131) & 0x7FFFFFFF
            tasks.append(GeneratedTask(
                task_id, scene.scene_id, s, g,
                self._initial_yaw(np.asarray(s), np.asarray(g), used_rng),
                cls, task_seed, {
                    "straight_distance": round(float(r.straight_distance), 4),
                    "global_path_length": round(float(r.global_path_length), 4),
                    "global_detour_ratio": round(float(r.global_detour_ratio), 4),
                    "global_min_clearance": round(
                        float(r.global_min_clearance), 4),
                    "direct_blocked": bool(r.direct_blocked),
                    "direct_blocker_count": int(r.direct_blocker_count),
                    "nearest_blocker_distance_m": round(
                        float(r.nearest_blocker_distance_m), 4),
                    "privileged_local_recoverable": bool(
                        r.privileged_local_recoverable),
                    "left_global_feasible": bool(r.left_global_feasible),
                    "right_global_feasible": bool(r.right_global_feasible),
                    "left_path_length": round(float(r.left_path_length), 4),
                    "right_path_length": round(float(r.right_path_length), 4),
                }))
        note = "generated %d tasks" % len(tasks)
        return tasks, note

    def _candidate_quality(self, cls, r, quota):
        q = 0.0
        if quota.needs_class(cls):
            q += 10.0
        q -= min(1.0, max(0.0, r.global_detour_ratio - 1.0)) * 0.5
        if r.global_min_clearance > 0.0:
            q += min(1.0, r.global_min_clearance) * 0.3
        return q


def runtime_classify(target_class, privileged_local_recoverable,
                     initial_observed_recoverable):
    """Runtime re-classification (sections XL/XLI/LXIII).  NEAR_OCCLUDED is
    only certain once the first observed frames confirm that the local
    layer cannot see a way out even though the privileged audit can."""
    if target_class is None:
        return None
    if target_class == TaskClass.NEAR_OCCLUDED:
        if initial_observed_recoverable:
            return TaskClass.LOCAL_REACTIVE
        return TaskClass.NEAR_OCCLUDED
    if privileged_local_recoverable and not initial_observed_recoverable:
        return TaskClass.NEAR_OCCLUDED
    return target_class
