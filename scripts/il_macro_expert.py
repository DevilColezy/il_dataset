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

Decision rule (section II / XXIV): every 5 Hz tick computes BOTH
  - the OBSERVED local recoverability of the direct intent (C++), and
  - the PRIVILEGED global viability of the direct intent (C++).
Only when the observed intent is locally recoverable AND the privileged
evaluator reports the direct intent as globally viable do we stay DIRECT.
Otherwise the blocker + candidates are generated and an observable SIDE
candidate (full_goal_reached in the observed map) is required before
committing SIDE_GUIDE; otherwise the macro goes OBSERVE.

Side memory (section XIII): failed_left / failed_right record why a side
was abandoned so the drone never flips back and forth.  Strategic progress
(section XIV) is measured per mode: DIRECT uses goal-distance decrease,
SIDE uses global cost-to-go decrease, OBSERVE uses observed-map known-voxel
growth / edge visibility changes.

All map work (blocker analysis, candidate generation, recoverability,
privileged scoring, intervention evaluation) runs in C++.  Python only
carries the state machine, hysteresis counters, side memory and
world-frame freeze.
"""

from __future__ import print_function, division

import math

import numpy as np

from il_common import normalize_angle


class MacroExpert(object):
    """5 Hz macro expert consuming C++ recoverability / candidate / oracle
    results."""

    def __init__(self, cfg, module, recoverability, candidate_search,
                 oracle, intervention_oracle, candidate_config):
        self.cfg = dict(cfg)
        self._module = module
        self._recoverability = recoverability        # C++ LocalRecoverability
        self._candidate_search = candidate_search    # C++ MacroCandidateSearch
        self._oracle = oracle                        # C++ PrivilegedOracle
        self._intervention_oracle = intervention_oracle  # C++ PrivilegedInterventionOracle
        self._candidate_cfg = candidate_config       # C++ MacroCandidateConfig
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
        self._observe_reference_yaw_world = None
        self._observe_known_reference = None
        self._no_progress_time_s = 0.0
        self._prev_goal_distance = None
        self._prev_guide_world = None
        self._side_progress_reference = None
        self._side_commit_start_time = 0.0
        self._failed_left = None
        self._failed_right = None
        self._last_decision_margin = 0.0
        self._last_recoverability = None
        self._last_intervention = None
        self._last_blocker = None
        self._last_candidates = []
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.0
        self._failed_reason = ""

    def _build_action(self, mode, guide_world, desired_yaw, has_yaw,
                      confidence, reason):
        action = self._module.MacroAction()
        action.mode = mode
        action.committed_side = self.committed_side
        action.guide_world = guide_world
        # Section IV: never float(None).  has_desired_yaw guards the field.
        if has_yaw and desired_yaw is not None:
            action.has_desired_yaw = True
            action.desired_yaw_world = float(desired_yaw)
        else:
            action.has_desired_yaw = False
            action.desired_yaw_world = 0.0
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
    def update(self, goal_world, state, observed_map, dt_s=0.2,
               local_unrecoverable=False):
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

        # ── Direct guide (navigation intent, not a collision-free
        #     guarantee; small obstacles may sit between here and it). ──
        lookahead = min(goal_dist, self.cfg["macro_lookahead_distance_m"])
        direct_guide = pos + goal_dir_world * lookahead

        vs = self._module.VehicleState()
        vs.position = pos
        vs.velocity = np.asarray(state["velocity"])
        vs.acceleration = np.asarray(state["acceleration"])
        vs.yaw = yaw
        vs.yaw_rate = float(state.get("yaw_rate", 0.0))

        # ── OBSERVED recoverability + PRIVILEGED intervention ────────
        rec = self._recoverability.test(observed_map, vs, direct_guide)
        self._last_recoverability = rec
        priv = self._intervention_oracle.evaluate(
            self._oracle, vs, direct_guide, goal)
        self._last_intervention = priv
        self._last_decision_margin = float(priv.decision_margin)

        # Strategic no-progress watchdog (section XIV).
        self._update_no_progress(dt_s, goal_dist, priv, observed_map)
        if self._no_progress_time_s > self.cfg["max_no_progress_seconds"]:
            self.mode = self._module.MacroMode.FAILED
            self._failed_reason = "no_progress_timeout"
            self._macro_decision_observable = False
            self._macro_decision_confidence = 0.0
            return self._build_action(
                self.mode, pos, None, False, 0.0, "no_progress_timeout")

        # ── Decision table (section XXIV) ────────────────────────────
        # The local layer reports that it cannot recover (repeated planning
        # failures): never trust DIRECT_REJOIN_SUCCESS in that case.
        direct_ok = \
            rec.status == \
            self._module.RecoverabilityStatus.DIRECT_REJOIN_SUCCESS and \
            bool(priv.direct_viable) and not local_unrecoverable
        if direct_ok:
            return self._handle_direct_recoverable(
                goal_dir_world, direct_guide, rec)
        return self._handle_not_recoverable(
            goal, pos, yaw, goal_dir_world, direct_guide,
            observed_map, vs, priv, dt_s, local_unrecoverable)

    # ── Strategic no-progress tracking (section XIV) ─────────────────
    def _update_no_progress(self, dt_s, goal_dist, priv, observed_map):
        made_progress = False
        if self.mode == self._module.MacroMode.DIRECT_GUIDE:
            if self._prev_goal_distance is not None:
                made_progress = \
                    (self._prev_goal_distance - goal_dist) >= \
                    self.cfg["min_goal_progress_m_s"] * dt_s
        elif self.mode == self._module.MacroMode.SIDE_GUIDE:
            # Global cost-to-go decrease is the SIDE-mode progress signal.
            ctg = float(priv.direct_cost_to_go)
            if math.isfinite(ctg) and \
                    self._side_progress_reference is not None and \
                    math.isfinite(self._side_progress_reference):
                made_progress = \
                    (self._side_progress_reference - ctg) >= \
                    self.cfg["min_goal_progress_m_s"] * dt_s
            if self._side_progress_reference is None:
                made_progress = True
        elif self.mode == self._module.MacroMode.OBSERVE:
            # OBSERVE progress: observed known-voxel growth (or the very
            # first observation tick).
            known = int(observed_map.known_count())
            if self._observe_known_reference is not None:
                made_progress = known > self._observe_known_reference
            else:
                made_progress = True
        self._prev_goal_distance = goal_dist
        if made_progress:
            self._no_progress_time_s = 0.0
        else:
            self._no_progress_time_s += dt_s

    # ── DIRECT intent is locally+globally valid ──────────────────────
    def _handle_direct_recoverable(self, goal_dir_world, direct_guide, rec):
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
            self._observe_reference_yaw_world = None
        else:
            self.mode = self._module.MacroMode.DIRECT_GUIDE
        self._direct_counter = 0

        yaw_world = math.atan2(goal_dir_world[1], goal_dir_world[0]) - \
            0.5 * math.pi
        confidence = 0.9
        if rec.minimum_clearance < self.cfg.get("confidence_gate", 0.5):
            confidence = 0.7
        self._macro_decision_observable = True
        self._macro_decision_confidence = confidence
        action = self._build_action(
            self.mode, direct_guide, yaw_world, True, confidence,
            "direct_recoverable")
        self._prev_guide_world = np.asarray(direct_guide)
        return action

    # ── DIRECT intent is NOT fully valid ─────────────────────────────
    def _handle_not_recoverable(self, goal, pos, yaw, goal_dir_world,
                                direct_guide, observed_map, vs,
                                priv, dt_s, local_unrecoverable=False):
        if self.mode == self._module.MacroMode.DIRECT_GUIDE and \
                not local_unrecoverable:
            # Hysteresis: N consecutive non-recoverable macro ticks before
            # leaving DIRECT_GUIDE.  When the local layer already reported
            # it cannot recover, the hysteresis is skipped.
            self._direct_counter += 1
            if self._direct_counter < self.cfg["direct_to_side_frames"]:
                yaw_world = math.atan2(goal_dir_world[1],
                                       goal_dir_world[0]) - 0.5 * math.pi
                self._macro_decision_observable = True
                self._macro_decision_confidence = 0.6
                return self._build_action(
                    self.mode, direct_guide, yaw_world, True, 0.6,
                    "direct_pending_hysteresis")
            self._direct_counter = 0
        elif self.mode == self._module.MacroMode.DIRECT_GUIDE:
            # local_unrecoverable: skip straight to the strategic decision.
            self._direct_counter = 0

        # OBSERVE timeout -> FAILED.
        if self.mode == self._module.MacroMode.OBSERVE:
            self._observe_time_s += dt_s
            if self._observe_time_s > self.cfg["max_observe_seconds"]:
                self.mode = self._module.MacroMode.FAILED
                self._failed_reason = "observe_timeout"
                self._macro_decision_observable = False
                self._macro_decision_confidence = 0.0
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

        # ── Observable SIDE candidates: REAL reachability (full_goal) ─
        side_cands = {self._module.Side.LEFT: [],
                      self._module.Side.RIGHT: []}
        for candidate in candidates:
            if candidate.type != self._module.CandidateType.SIDE:
                continue
            if not candidate.full_goal_reached:
                continue
            if candidate.side in side_cands:
                side_cands[candidate.side].append(candidate)

        # Preferred side from the privileged evaluation (section II case
        # E) combined with side memory (section XIII).
        preferred = self._privileged_preferred_side(priv)
        side = self._select_observable_side(preferred, side_cands)

        if side is not None:
            # Commit SIDE_GUIDE.
            candidate = self._best_candidate_for_side(
                side_cands[side], side)
            self.committed_side = side
            self.mode = self._module.MacroMode.SIDE_GUIDE
            self._side_counter = 0
            self._observe_time_s = 0.0
            self._observe_yaw_delta = 0.0
            self._observe_reference_yaw_world = None
            self._side_commit_start_time = self._no_progress_time_s
            self._side_progress_reference = \
                float(priv.direct_cost_to_go) \
                if math.isfinite(float(priv.direct_cost_to_go)) else None
            guide = direct_guide if candidate is None else \
                np.asarray(candidate.position_world)
            yaw_world = math.atan2(guide[1] - pos[1], guide[0] - pos[0]) - \
                0.5 * math.pi
            side_name = "left" if side == self._module.Side.LEFT else "right"
            self._macro_decision_observable = True
            self._macro_decision_confidence = 0.75
            action = self._build_action(
                self.mode, guide, yaw_world, True, 0.75,
                "side_guide_%s" % side_name)
            self._prev_guide_world = guide
            return action

        # ── No usable observable SIDE candidate -> OBSERVE ───────────
        # Record why the attempted side failed (section XIII).  A side is
        # only marked failed after it was actually attempted: a committed
        # SIDE_GUIDE that lost its reachable corridor, a side whose sweep
        # saturated with no corridor, or a committed side the privileged
        # map now reports disconnected.
        fov_half_obs = math.radians(self.cfg.get("observe_fov_half_deg", 45.0))
        if self.mode == self._module.MacroMode.SIDE_GUIDE:
            self._mark_side_failed(self.committed_side,
                                   "LOCAL_PATH_FAILED")
        elif self.mode == self._module.MacroMode.OBSERVE and \
                abs(self._observe_yaw_delta) >= fov_half_obs - 1e-6:
            self._mark_side_failed(self.committed_side,
                                   "OBSERVED_NO_CORRIDOR")
        if self.mode == self._module.MacroMode.SIDE_GUIDE and \
                self.committed_side != self._module.Side.NONE:
            if self.committed_side == self._module.Side.LEFT and \
                    not bool(priv.left_globally_feasible):
                self._mark_side_failed(self._module.Side.LEFT,
                                       "PRIVILEGED_DISCONNECTED")
            if self.committed_side == self._module.Side.RIGHT and \
                    not bool(priv.right_globally_feasible):
                self._mark_side_failed(self._module.Side.RIGHT,
                                       "PRIVILEGED_DISCONNECTED")

        # OBSERVE side: committed (if not failed) or first non-failed or
        # LEFT (section XIII).
        side = self._observe_side()
        self.committed_side = side

        # Sweep the yaw from the FIXED reference yaw captured at OBSERVE
        # entry (section XVI).  Never current_yaw + accumulated delta.
        if self.mode != self._module.MacroMode.OBSERVE or \
                self._observe_reference_yaw_world is None:
            self._observe_reference_yaw_world = yaw
            self._observe_yaw_delta = 0.0
            self._observe_known_reference = int(observed_map.known_count())
        sign = 1.0 if side == self._module.Side.LEFT else -1.0
        sweep_step = self.cfg.get("observe_rotation_rate_rps", 1.5) * dt_s
        self._observe_yaw_delta += sign * sweep_step
        self._observe_yaw_delta = \
            sign * min(abs(self._observe_yaw_delta), fov_half_obs)

        self.mode = self._module.MacroMode.OBSERVE
        desired_yaw_world = normalize_angle(
            self._observe_reference_yaw_world + self._observe_yaw_delta)

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
        self._macro_decision_observable = False
        self._macro_decision_confidence = 0.5
        action = self._build_action(
            self.mode, guide, desired_yaw_world, True, 0.5,
            "observe_%s" % side_name)
        self._prev_guide_world = np.asarray(guide)
        return action

    # ── Side memory (section XIII) ───────────────────────────────────
    def _mark_side_failed(self, side, reason):
        if side == self._module.Side.LEFT:
            self._failed_left = reason
        elif side == self._module.Side.RIGHT:
            self._failed_right = reason

    def _observe_side(self):
        if self.committed_side == self._module.Side.LEFT and \
                self._failed_left is None:
            return self._module.Side.LEFT
        if self.committed_side == self._module.Side.RIGHT and \
                self._failed_right is None:
            return self._module.Side.RIGHT
        if self._failed_left is None:
            return self._module.Side.LEFT
        if self._failed_right is None:
            return self._module.Side.RIGHT
        # Both sides failed: prefer the one with a reason that is not a
        # hard structural failure (try the last committed first).
        if self.committed_side == self._module.Side.RIGHT:
            return self._module.Side.RIGHT
        return self._module.Side.LEFT

    # ── Side selection (section II case E) ───────────────────────────
    def _privileged_preferred_side(self, priv):
        """Return the privileged-preferred side (LEFT/RIGHT/NONE) using
        global feasibility and decision margin, respecting side memory and
        the committed side on genuine ties."""
        left_feasible = bool(priv.left_globally_feasible)
        right_feasible = bool(priv.right_globally_feasible)
        left_failed = self._failed_left is not None
        right_failed = self._failed_right is not None
        if left_feasible and not left_failed and not right_feasible:
            return self._module.Side.LEFT
        if right_feasible and not right_failed and not left_feasible:
            return self._module.Side.RIGHT
        if left_feasible and right_feasible:
            margin = float(priv.decision_margin)
            cost_margin = self.cfg.get("cost_margin_m", 2.0)
            if margin >= cost_margin:
                if float(priv.left_cost_to_go) <= \
                        float(priv.right_cost_to_go):
                    return self._module.Side.LEFT
                return self._module.Side.RIGHT
            # Genuine tie: keep the committed side if still feasible,
            # otherwise the fixed left preference (never unconditional).
            if self.committed_side == self._module.Side.LEFT and \
                    not left_failed:
                return self._module.Side.LEFT
            if self.committed_side == self._module.Side.RIGHT and \
                    not right_failed:
                return self._module.Side.RIGHT
            if self.cfg.get("left_preferred_on_tie", True):
                return self._module.Side.LEFT
            return self._module.Side.RIGHT
        if left_feasible and not left_failed:
            return self._module.Side.LEFT
        if right_feasible and not right_failed:
            return self._module.Side.RIGHT
        return self._module.Side.NONE

    def _select_observable_side(self, preferred, side_cands):
        """Choose the side to commit: the privileged-preferred side when an
        observable (full_goal_reached) candidate exists there, else the
        other side if observable, else None -> OBSERVE."""
        left_cands = side_cands[self._module.Side.LEFT]
        right_cands = side_cands[self._module.Side.RIGHT]
        if preferred == self._module.Side.LEFT and left_cands:
            return self._module.Side.LEFT
        if preferred == self._module.Side.RIGHT and right_cands:
            return self._module.Side.RIGHT
        if left_cands and not right_cands:
            return self._module.Side.LEFT
        if right_cands and not left_cands:
            return self._module.Side.RIGHT
        if left_cands and right_cands:
            # Both observable: prefer the privileged-preferred side (fall
            # back to the committed side, then left).
            if self.committed_side == self._module.Side.LEFT and left_cands:
                return self._module.Side.LEFT
            if self.committed_side == self._module.Side.RIGHT and right_cands:
                return self._module.Side.RIGHT
            if self.cfg.get("left_preferred_on_tie", True):
                return self._module.Side.LEFT
            return self._module.Side.RIGHT
        return None

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
    def last_intervention(self):
        return self._last_intervention

    @property
    def last_blocker(self):
        return self._last_blocker

    @property
    def macro_decision_observable(self):
        return bool(self._macro_decision_observable)

    @property
    def macro_decision_confidence(self):
        return float(self._macro_decision_confidence)
