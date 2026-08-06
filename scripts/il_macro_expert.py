#!/usr/bin/env python3
"""
il_macro_expert.py  —  5 Hz macro navigation expert.

States (section IV):
    DIRECT_GUIDE  — direct guide along the goal ray; the local layer
                    resolves small obstacles.
    SIDE_GUIDE    — committed lateral bypass toward a known side corridor.
    OBSERVE       — insufficient known corridor: rotate / short safe move to
                    expose the bypass edge.
    GOAL_REACHED  — final goal reached.
    FAILED        — observation / progress budgets exhausted.

All map work (blocker analysis, candidate generation, recoverability,
privileged scoring) runs in C++.  Python only carries the state machine,
hysteresis counters, the fixed-left tie-break and world-frame freeze.
"""

from __future__ import print_function, division

import math

import numpy as np

from il_common import normalize_angle


class MacroExpert(object):
    """5 Hz macro expert consuming C++ recoverability / candidate / oracle
    results."""

    def __init__(self, cfg, module, recoverability, candidate_search,
                 oracle, candidate_config):
        self.cfg = dict(cfg)
        self._module = module
        self._recoverability = recoverability      # C++ LocalRecoverability
        self._candidate_search = candidate_search  # C++ MacroCandidateSearch
        self._oracle = oracle                      # C++ PrivilegedOracle
        self._candidate_cfg = candidate_config     # C++ MacroCandidateConfig
        self._current_position = np.zeros(3, dtype=np.float64)
        self.reset()

    # ── Episode lifecycle ────────────────────────────────────────────
    def reset(self):
        self.mode = self._module.MacroMode.DIRECT_GUIDE
        self.committed_side = self._module.Side.NONE
        self._direct_counter = 0
        self._side_counter = 0
        self._observe_time_s = 0.0
        self._observe_yaw_delta = 0.0
        self._no_progress_time_s = 0.0
        self._prev_goal_distance = None
        self._prev_guide_world = None
        self._last_decision_margin = 0.0
        self._last_recoverability = None
        self._last_blocker = None
        self._last_candidates = []
        self._failed_reason = ""

    def _build_action(self, mode, guide_world, desired_yaw, has_yaw,
                      confidence, reason):
        action = self._module.MacroAction()
        action.mode = mode
        action.committed_side = self.committed_side
        action.guide_world = guide_world
        action.desired_yaw_world = float(desired_yaw)
        action.has_desired_yaw = has_yaw
        action.confidence = float(max(0.0, min(1.0, confidence)))
        action.is_new_tick = True
        action.reason = reason
        if guide_world is not None:
            delta = np.asarray(guide_world, dtype=np.float64) - \
                self._current_position
            action.guide_distance = float(np.linalg.norm(delta))
        return action

    def make_direct_action(self, goal_world, state):
        """Build a DIRECT_GUIDE action toward the goal (used before the
        first 5 Hz tick and by tests)."""
        goal = np.asarray(goal_world, dtype=np.float64)
        pos = np.asarray(state["position"], dtype=np.float64)
        dist = float(np.linalg.norm(goal - pos))
        goal_dir = np.zeros(3, dtype=np.float64)
        if dist > 1e-6:
            goal_dir = (goal - pos) / dist
        lookahead = min(dist, self.cfg["macro_lookahead_distance_m"])
        guide = pos + goal_dir * lookahead
        yaw_world = math.atan2(goal_dir[1], goal_dir[0]) - 0.5 * math.pi
        self._current_position = pos
        self._current_yaw = float(state["yaw"])
        return self._build_action(
            self._module.MacroMode.DIRECT_GUIDE, guide, yaw_world, True,
            0.9, "direct_initial")

    # ── Per-tick update (called at 5 Hz) ─────────────────────────────
    def update(self, goal_world, state, observed_map, dt_s=0.2):
        goal = np.asarray(goal_world, dtype=np.float64)
        pos = np.asarray(state["position"], dtype=np.float64)
        yaw = float(state["yaw"])
        speed = float(np.linalg.norm(np.asarray(state["velocity"])))
        self._current_position = pos
        self._current_yaw = yaw

        goal_dist = float(np.linalg.norm(goal - pos))
        goal_dir_world = np.zeros(3, dtype=np.float64)
        if goal_dist > 1e-6:
            goal_dir_world = (goal - pos) / goal_dist

        # ── GOAL_REACHED ─────────────────────────────────────────────
        if goal_dist <= self.cfg["goal_tolerance_m"] and \
                speed <= self.cfg["goal_speed_tolerance_mps"]:
            self.mode = self._module.MacroMode.GOAL_REACHED
            self._prev_goal_distance = goal_dist
            self._prev_guide_world = pos
            return self._build_action(
                self.mode, pos, None, False, 1.0, "goal_reached")

        # ── No-progress watchdog ─────────────────────────────────────
        if self._prev_goal_distance is not None:
            progress_rate = (self._prev_goal_distance - goal_dist) / dt_s
            if progress_rate < self.cfg["min_goal_progress_m_s"]:
                self._no_progress_time_s += dt_s
            else:
                self._no_progress_time_s = 0.0
        self._prev_goal_distance = goal_dist
        if self._no_progress_time_s > self.cfg["max_no_progress_seconds"]:
            self.mode = self._module.MacroMode.FAILED
            self._failed_reason = "no_progress_timeout"
            return self._build_action(
                self.mode, pos, None, False, 0.0, "no_progress_timeout")

        # ── Direct guide (navigation intent, not a collision-free
        #     guarantee; small obstacles may sit between here and it). ──
        lookahead = min(goal_dist, self.cfg["macro_lookahead_distance_m"])
        direct_guide = pos + goal_dir_world * lookahead

        # ── Local recoverability of the direct intent ────────────────
        vs = self._module.VehicleState()
        vs.position = pos
        vs.velocity = np.asarray(state["velocity"])
        vs.acceleration = np.asarray(state["acceleration"])
        vs.yaw = yaw
        vs.yaw_rate = float(state.get("yaw_rate", 0.0))
        rec = self._recoverability.test(observed_map, vs, direct_guide)
        self._last_recoverability = rec

        if rec.status == \
                self._module.RecoverabilityStatus.DIRECT_REJOIN_SUCCESS:
            return self._handle_direct_recoverable(
                goal, pos, yaw, goal_dir_world, direct_guide, rec)
        return self._handle_not_recoverable(
            goal, pos, yaw, goal_dir_world, direct_guide, goal_dist,
            observed_map, vs, rec, dt_s)

    # ── DIRECT intent is locally recoverable ─────────────────────────
    def _handle_direct_recoverable(self, goal, pos, yaw, goal_dir_world,
                                   direct_guide, rec):
        if self.mode == self._module.MacroMode.SIDE_GUIDE:
            # Hysteresis: SIDE -> DIRECT after N consecutive recoverable
            # ticks with normal goal progress.
            self._side_counter += 1
            if self._side_counter >= self.cfg["side_to_direct_frames"] and \
                    self._no_progress_time_s <= 0.0:
                self.mode = self._module.MacroMode.DIRECT_GUIDE
                self._side_counter = 0
        elif self.mode == self._module.MacroMode.OBSERVE:
            # OBSERVE -> DIRECT as soon as the direct intent recovers.
            self.mode = self._module.MacroMode.DIRECT_GUIDE
            self._observe_time_s = 0.0
            self._observe_yaw_delta = 0.0
        else:
            self.mode = self._module.MacroMode.DIRECT_GUIDE
        self._direct_counter = 0

        yaw_world = math.atan2(goal_dir_world[1], goal_dir_world[0]) - \
            0.5 * math.pi
        confidence = 0.9
        if rec.minimum_clearance < self.cfg.get("confidence_gate", 0.5):
            confidence = 0.7
        action = self._build_action(
            self.mode, direct_guide, yaw_world, True, confidence,
            "direct_recoverable")
        self._prev_guide_world = np.asarray(direct_guide)
        return action

    # ── DIRECT intent is NOT locally recoverable ─────────────────────
    def _handle_not_recoverable(self, goal, pos, yaw, goal_dir_world,
                                direct_guide, goal_dist, observed_map, vs,
                                rec, dt_s):
        self._last_decision_margin = 0.0
        if self.mode == self._module.MacroMode.DIRECT_GUIDE:
            # Hysteresis: N consecutive non-recoverable macro ticks before
            # leaving DIRECT_GUIDE.
            self._direct_counter += 1
            if self._direct_counter < self.cfg["direct_to_side_frames"]:
                yaw_world = math.atan2(goal_dir_world[1],
                                       goal_dir_world[0]) - 0.5 * math.pi
                return self._build_action(
                    self.mode, direct_guide, yaw_world, True, 0.6,
                    "direct_pending_hysteresis")
            self._direct_counter = 0

        # OBSERVE timeout -> FAILED.
        if self.mode == self._module.MacroMode.OBSERVE:
            self._observe_time_s += dt_s
            if self._observe_time_s > self.cfg["max_observe_seconds"]:
                self.mode = self._module.MacroMode.FAILED
                self._failed_reason = "observe_timeout"
                return self._build_action(
                    self.mode, pos, None, False, 0.0, "observe_timeout")

        # ── Blocker + candidates (C++) ───────────────────────────────
        blocker = self._module.analyze_goal_blocker(
            observed_map, vs, goal, self._candidate_cfg)
        self._last_blocker = blocker
        prev_ptr = None
        if self._prev_guide_world is not None:
            prev_ptr = self._prev_guide_world
        candidates = self._candidate_search.generate_candidates(
            observed_map, vs, goal, blocker, prev_ptr)
        self._oracle.score_candidates(
            candidates, vs, goal, self.committed_side, prev_ptr)
        self._last_candidates = candidates

        # ── Known side corridor exists -> SIDE_GUIDE ─────────────────
        left_known = blocker.left_corridor_known
        right_known = blocker.right_corridor_known
        if left_known or right_known:
            side = self._select_side(blocker, candidates)
            candidate = self._best_candidate_for_side(candidates, side)
            self.committed_side = side
            self.mode = self._module.MacroMode.SIDE_GUIDE
            self._side_counter = 0
            self._observe_time_s = 0.0
            self._observe_yaw_delta = 0.0
            guide = direct_guide if candidate is None else \
                np.asarray(candidate.position_world)
            yaw_world = math.atan2(guide[1] - pos[1], guide[0] - pos[0]) - \
                0.5 * math.pi
            side_name = "left" if side == self._module.Side.LEFT else "right"
            action = self._build_action(
                self.mode, guide, yaw_world, True, 0.75,
                "side_guide_%s" % side_name)
            self._prev_guide_world = guide
            return action

        # ── No known corridor -> OBSERVE ─────────────────────────────
        side = self.committed_side
        if side == self._module.Side.NONE:
            side = (self._module.Side.LEFT
                    if self.cfg.get("left_preferred_on_tie", True)
                    else self._module.Side.RIGHT)
        self.committed_side = side

        # Sweep the yaw toward the committed side (never a fixed
        # left/right 90/90 scan).  Each 5 Hz tick advances the delta.
        sign = 1.0 if side == self._module.Side.LEFT else -1.0
        fov_half = math.radians(self.cfg.get("observe_fov_half_deg", 45.0))
        sweep_step = self.cfg.get("observe_rotation_rate_rps", 1.5) * dt_s
        self._observe_yaw_delta += sign * sweep_step
        self._observe_yaw_delta = \
            sign * min(abs(self._observe_yaw_delta), fov_half)

        self.mode = self._module.MacroMode.OBSERVE
        desired_yaw_world = normalize_angle(yaw + self._observe_yaw_delta)

        # Short known-safe observe move (zero displacement allowed).
        obs_step = self.cfg.get("observe_step_m", 0.6)
        observe_dir = np.array(
            [-math.sin(desired_yaw_world), math.cos(desired_yaw_world), 0.0],
            dtype=np.float64)
        probe = pos + observe_dir * obs_step
        guide = probe
        if not observed_map.is_known_free(
                probe, self.cfg.get("observe_clearance_m", 0.20)):
            guide = pos
        side_name = "left" if side == self._module.Side.LEFT else "right"
        action = self._build_action(
            self.mode, guide, desired_yaw_world, True, 0.5,
            "observe_%s" % side_name)
        self._prev_guide_world = np.asarray(guide)
        return action

    # ── Side selection (fixed-left tie-break, never unconditional) ──
    def _select_side(self, blocker, candidates):
        left_known = blocker.left_corridor_known
        right_known = blocker.right_corridor_known
        if left_known and not right_known:
            return self._module.Side.LEFT
        if right_known and not left_known:
            return self._module.Side.RIGHT

        # Both feasible: use the privileged evaluation when the margin is
        # clear; otherwise keep the committed side; a fresh tie goes LEFT.
        privileged_side, margin = self._oracle.privileged_best_side(
            candidates)
        self._last_decision_margin = float(margin)
        if privileged_side != self._module.Side.NONE:
            return privileged_side
        if self.committed_side != self._module.Side.NONE:
            return self.committed_side
        if self.cfg.get("left_preferred_on_tie", True):
            return self._module.Side.LEFT
        return self._module.Side.RIGHT

    def _best_candidate_for_side(self, candidates, side):
        best = None
        for candidate in candidates:
            if candidate.side != side:
                continue
            if candidate.type != self._module.CandidateType.SIDE:
                continue
            if best is None or \
                    candidate.privileged_score < best.privileged_score:
                best = candidate
        return best

    # ── Diagnostics for the dataset writer ───────────────────────────
    @property
    def failed_reason(self):
        return self._failed_reason

    @property
    def last_decision_margin(self):
        return self._last_decision_margin

    @property
    def last_candidates(self):
        return self._last_candidates

    @property
    def last_recoverability(self):
        return self._last_recoverability

    @property
    def last_blocker(self):
        return self._last_blocker
