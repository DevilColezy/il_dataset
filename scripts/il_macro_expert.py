#!/usr/bin/env python3
"""il_macro_expert.py  —  5 Hz macro navigation expert.

States (section IV):
    DIRECT_GUIDE  — direct guide along the goal ray; the 30 Hz local layer
                    resolves small obstacles (local scale).
    SIDE_GUIDE    — committed lateral bypass toward a known side corridor.
    OBSERVE       — insufficient known corridor: rotate / short safe move to
                    expose the bypass edge.
    GOAL_REACHED  — final goal reached.
    FAILED        — observation / progress budgets exhausted.

Scale rule (section I): scale is NEVER defined by obstacle size / bbox /
world length / straight-line blockers.  It is defined ONLY by
    current observation + local planner finite-horizon ability +
    whether the direct guide can be re-entered.
A far large obstacle with visible edges and a complete local bypass stays
DIRECT (the 30 Hz layer handles it); a tiny but close, edge-occluded
obstacle may require OBSERVE / SIDE_GUIDE.

Decision rule (sections III/IV/XXII): every 5 Hz tick computes BOTH
  - the OBSERVED local recoverability of the direct intent (C++), and
  - the PRIVILEGED local-scale audit (C++ short-range search on the full
    map, ALLOWING bypass — never a direct-ray collision check).
The privileged result is RECORDED (auxiliary labels / episode analysis)
but NEVER gates the main mode by itself.  The main mode is causal:

    if observed == DIRECT_REJOIN_SUCCESS and not causal_intervention_evidence:
        -> DIRECT_GUIDE
    else:
        -> proceed_to_macro_intervention()
            blocker -> candidates -> observed FULL reachability filter ->
            privileged global-connectivity filter -> SIDE_GUIDE else OBSERVE

causal_intervention_evidence is built ONLY from student-observable history
(repeated local replanning failures / cached / BRAKE_HOLD, repeated
near-zero progress) — privilege never upgrades DIRECT alone.

Side memory (section VIII): failed_left / failed_right store an enum
reason; NOT_YET_OBSERVED is not a failure; LEFT failed -> try RIGHT.
Strategic progress (section VII) uses ROLLING windows (not the entry
reference): SIDE uses global cost-to-go / arc-length decrease, OBSERVE
uses recent known-cell / edge / reachable-candidate growth.

All map work runs in C++.  Python carries the state machine, causal
evidence, side memory and world-frame freeze.
"""

from __future__ import print_function, division

import math
import time
from enum import Enum

import numpy as np

from il_common import normalize_angle


class SideFailure(Enum):
    """Why a side was abandoned (section VIII)."""
    NOT_YET_OBSERVED = 0
    OBSERVED_NO_CORRIDOR = 1
    LOCAL_PATH_FAILED = 2
    PRIVILEGED_DISCONNECTED = 3
    NO_PROGRESS = 4


# Human-readable names for the debug trace (scripts/debug_viewer.py).
_MACRO_MODE_NAMES = {
    0: "DIRECT_GUIDE", 1: "SIDE_GUIDE", 2: "OBSERVE",
    3: "GOAL_REACHED", 4: "FAILED",
}
_SIDE_NAMES = {0: "NONE", 1: "LEFT", -1: "RIGHT"}
_SIDE_FAILURE_NAMES = {
    0: "NOT_YET_OBSERVED", 1: "OBSERVED_NO_CORRIDOR",
    2: "LOCAL_PATH_FAILED", 3: "PRIVILEGED_DISCONNECTED", 4: "NO_PROGRESS",
}


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
        self._observe_time_s = 0.0
        self._observe_yaw_delta = 0.0
        self._observe_reference_yaw_world = None
        # OBSERVE information baselines (sections XVI-XVIII): monotonic
        # bests for known cells and per-category FULL-reachable movement
        # counts (SIDE / OBSERVE / GOAL_FRONTIER), state-change for edge
        # visibility.  A fresh session writes the CURRENT values — never
        # None — so no tick can ever hit `int > None`.
        self._observe_best_known = None
        self._observe_last_edge_mask = None
        self._observe_best_side_count = None
        self._observe_best_observe_count = None
        self._observe_best_frontier_count = None
        self._observe_last_info_time = None
        self._prev_guide_world = None
        # Active-observation scan lifecycle (sections XXI-XXXV): per-side
        # scan exhaustion is SEPARATE from side failure memory (failed_*).
        # Scan exhaustion means "rotation alone no longer yields new
        # information here" — never "this side is infeasible".
        self._observe_scan_side = None
        self._left_scan_exhausted = False
        self._right_scan_exhausted = False
        self._observe_stagnant_time = 0.0
        self._observe_last_target_yaw = None
        self._observe_last_move_guide = None
        self._prev_observe_viewpoint = None
        # World anchor of the current observation viewpoint (sections
        # XXI-XXIV): a real OBSERVE_MOVE beyond viewpoint_reset_distance_m
        # re-bases the whole scan session at the new position.
        self._observe_anchor_position_world = None
        # Per-tick OBSERVE diagnostics (pure diagnostics, never student
        # input).  Held between 5 Hz ticks so the 30 Hz recorder sees them.
        self.observe_scan_side = 0
        self.left_scan_exhausted = 0
        self.right_scan_exhausted = 0
        self.observe_rotation_exhausted = 0
        self.observe_stagnant_rotate_time = 0.0
        self.observe_raw_candidate_count = 0
        self.observe_lattice_candidate_count = 0
        self.observe_frontier_candidate_count = 0
        self.observe_endpoint_known_free_count = 0
        self.observe_local_full_count = 0
        self.observe_reject_unknown = 0
        self.observe_reject_endpoint_clearance = 0
        self.observe_reject_min_distance = 0
        self.observe_reject_max_distance = 0
        self.observe_reject_partial = 0
        self.observe_reject_no_path = 0
        self.observe_reject_failed_side = 0
        self.observe_left_valid_count = 0
        self.observe_right_valid_count = 0
        self.observe_center_valid_count = 0
        self.observe_selected_source = ""
        self.observe_selected_side = 0
        self.observe_selected_distance = 0.0
        self.observe_selected_path_length = 0.0
        self.observe_selected_info_gain = 0.0
        self.observe_selected_clearance = 0.0
        # OBSERVE session diagnostics (sections XXXV-XXXVI): whether a
        # session baseline has been established this tick and how far the
        # drone is from its observation anchor.
        self.observe_session_initialized = 0
        self.observe_anchor_distance = 0.0
        # DIRECT session (section XXV): history is initialised ONLY when
        # entering DIRECT from another mode; a DIRECT->DIRECT tick only
        # UPDATES the history so causal evidence accumulates across 5 Hz
        # cycles (never reset on plain DIRECT ticks).
        self._direct_session_active = False
        self._direct_best_goal_distance = None
        self._direct_last_progress_time = None
        self.direct_no_progress_time = 0.0
        self._direct_local_failure_ticks = 0
        self._direct_cached_ticks = 0
        self._direct_brake_ticks = 0
        self.causal_intervention_evidence = False
        self.observe_no_information_time = 0.0
        # SIDE session (sections XVI/XVII): strategic progress uses ONLY
        # global cost-to-go (or committed-candidate distance as fallback).
        # Travelled distance is a diagnostic, never strategic progress.
        self._side_session_active = False
        self._side_best_cost = None
        self._side_target_world = None
        self._side_best_target_dist = None
        self._side_last_progress_time = None
        self._side_last_pos = None
        self._side_path_progress = 0.0
        # Side memory (section VIII/X): bound to the ACTIVE blocker TRACK
        # (stable world-geometry association).  Side memory is reset only
        # when a CONFIRMED new blocker appears or the old blocker is
        # confirmed passed.
        self._blocker_track = None
        self._blocker_lost_since = None
        self._blocker_new_pending = None
        self._blocker_switch_ticks = 0
        self._blocker_matches_current = False
        self._next_blocker_id = 0
        # Stable DIRECT recovery (sections X-XV): consecutive macro ticks
        # with DIRECT genuinely stable -> release old blocker memory.
        self._direct_stable_ticks = 0
        self._blocker_released_this_tick = False
        self._failed_left = None
        self._failed_right = None
        self._side_local_fail_count = {
            self._module.Side.LEFT: 0,
            self._module.Side.RIGHT: 0,
        }
        self._last_recoverability = None
        self._last_intervention = None
        self._last_blocker = None
        self._last_candidates = []
        self._last_trace = None
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.0
        self._failed_reason = ""

    def _build_action(self, mode, guide_world, desired_yaw, has_yaw,
                      confidence, reason, observe_side=None):
        action = self._module.MacroAction()
        action.mode = mode
        # Interface contract (sections XII/XIV): ONLY SIDE_GUIDE exports a
        # LEFT/RIGHT committed side to the 30 Hz local planner.  DIRECT and
        # OBSERVE always export NONE so a stale side topology never biases
        # the local A*.  OBSERVE direction is carried in observe_side.
        if mode == self._module.MacroMode.SIDE_GUIDE:
            action.committed_side = self.committed_side
        else:
            action.committed_side = self._module.Side.NONE
        if observe_side is None:
            observe_side = self._module.Side.NONE
        # pybind's MacroAction.observe_side is a `Side` enum member — the
        # setter rejects an int (TypeError); assign the enum directly like
        # committed_side.  int() conversion only happens when READING (CSV /
        # trace), where pybind enum -> int is supported.
        action.observe_side = observe_side
        action.guide_world = guide_world
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
        first 5 Hz tick)."""
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
               interval_feedback=None):
        goal = np.asarray(goal_world, dtype=np.float64)
        pos = np.asarray(state["position"], dtype=np.float64)
        yaw = float(state["yaw"])
        speed = float(np.linalg.norm(np.asarray(state["velocity"])))
        self._current_position = pos
        self._current_yaw = yaw
        now_s = time.monotonic()
        # Per-tick diagnostic: cleared here, set True only when the stable
        # DIRECT release fires below (section XII).
        self._blocker_released_this_tick = False
        # Per-tick OBSERVE diagnostics are reset here and populated only in
        # the OBSERVE branch (section XLVII); non-OBSERVE ticks record zeros.
        self._reset_observe_diagnostics()

        goal_dist = float(np.linalg.norm(goal - pos))
        goal_dir_world = np.zeros(3, dtype=np.float64)
        if goal_dist > 1e-6:
            goal_dir_world = (goal - pos) / goal_dist

        # ── GOAL_REACHED ─────────────────────────────────────────────
        if goal_dist <= self.cfg["goal_tolerance_m"] and \
                speed <= self.cfg["goal_speed_tolerance_mps"]:
            self.mode = self._module.MacroMode.GOAL_REACHED
            self._prev_guide_world = pos
            action = self._build_action(
                self.mode, pos, None, False, 1.0, "goal_reached")
            self._capture_trace(action)
            return action

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

        # ── OBSERVED recoverability + PRIVILEGED local-scale audit ───
        rec = self._recoverability.test(observed_map, vs, direct_guide)
        self._last_recoverability = rec
        priv = self._intervention_oracle.evaluate(
            self._oracle, vs, direct_guide, goal, now_s)
        self._last_intervention = priv

        # Causal evidence (student-observable history only, sections
        # III/X/XI): DIRECT long no-progress and repeated local replanning
        # failures upgrade to macro intervention — never an immediate
        # episode FAILED.
        if self.mode == self._module.MacroMode.DIRECT_GUIDE:
            self._update_direct_causal(now_s, goal_dist, interval_feedback)
        elif self.mode == self._module.MacroMode.SIDE_GUIDE:
            self._update_side_progress(now_s, state)

        # ── Main decision table (section IV / XXII) ──────────────────
        observed_ok = rec.status == \
            self._module.RecoverabilityStatus.DIRECT_REJOIN_SUCCESS

        # Stable DIRECT recovery (sections X-XV): accumulate macro ticks
        # while DIRECT is genuinely stable (observed recoverable + no
        # causal evidence + no local-unrecoverable interval).  Reaching
        # `direct_release_ticks` confirms the old blocker is passed and
        # releases its topology memory (sections XI/XII).  A single DIRECT
        # tick (or a reappearing blocker) resets the counter (section XIV).
        feedback_now = interval_feedback or {}
        stable_local = int(feedback_now.get(
            "local_unrecoverable_count", 0)) == 0
        if observed_ok and not self.causal_intervention_evidence and \
                stable_local:
            self._direct_stable_ticks += 1
            if self._direct_stable_ticks >= self.cfg.get(
                    "direct_release_ticks", 3) and \
                    self._blocker_track is not None:
                self._blocker_released_this_tick = True
                self._release_blocker()
        else:
            self._direct_stable_ticks = 0

        if observed_ok and not self.causal_intervention_evidence:
            action = self._handle_direct(goal_dir_world, direct_guide, rec,
                                         goal_dist, now_s)
            self._capture_trace(action)
            return action
        action = self._proceed_to_macro_intervention(
            goal, pos, yaw, direct_guide, observed_map, vs, dt_s, now_s)
        self._capture_trace(action)
        return action

    # ── DIRECT session (sections II/III/XXV) ─────────────────────────
    def _enter_direct_session(self, goal_dist, now_s):
        """Initialise the DIRECT causal window ONCE when entering DIRECT
        from SIDE_GUIDE / OBSERVE.  Never called on a DIRECT->DIRECT tick,
        so evidence accumulates across 5 Hz cycles."""
        self._direct_session_active = True
        # Leaving SIDE closes its rolling strategic session so a later
        # re-entry to the same committed side re-initialises the metrics.
        self._side_session_active = False
        self._direct_best_goal_distance = goal_dist
        self._direct_last_progress_time = now_s
        self.direct_no_progress_time = 0.0
        self._direct_local_failure_ticks = 0
        self._direct_cached_ticks = 0
        self._direct_brake_ticks = 0
        self.causal_intervention_evidence = False

    def _exit_direct_session(self):
        """DIRECT session ends when the mode leaves DIRECT.  The causal
        evidence has served its purpose (trigger macro intervention) and
        must NOT keep blocking a later re-entry into DIRECT (sections
        II/III, Case A)."""
        self._direct_session_active = False
        self.causal_intervention_evidence = False
        self._direct_local_failure_ticks = 0
        self._direct_cached_ticks = 0
        self._direct_brake_ticks = 0
        self._direct_best_goal_distance = None
        self._direct_last_progress_time = None
        self.direct_no_progress_time = 0.0

    def _update_direct_causal(self, now_s, goal_dist, interval_feedback):
        """Rolling DIRECT-session update using 30 Hz -> 5 Hz INTERVAL
        feedback (sections XXIV-XXVI): per-interval planning-failure /
        cached / brake / emergency aggregates drive macro-tick counters
        with decay, so a single occasional bad frame never triggers
        intervention while consecutive unstable intervals do."""
        cf = self.cfg.get("causal_feedback", {})
        feedback = interval_feedback or {}
        interval_frames = max(1, int(feedback.get("interval_frame_count", 1)))
        if self._direct_best_goal_distance is None or \
                goal_dist < self._direct_best_goal_distance - \
                self.cfg.get("goal_progress_epsilon_m", 0.05):
            self._direct_best_goal_distance = goal_dist
            self._direct_last_progress_time = now_s
        if self._direct_last_progress_time is not None:
            self.direct_no_progress_time = \
                now_s - self._direct_last_progress_time

        # Local planning failures within the interval (section XXVI).
        bad_interval = int(feedback.get("planning_failure_count", 0)) >= \
            int(cf.get("interval_failure_threshold", 2))
        if bad_interval or int(feedback.get(
                "local_unrecoverable_count", 0)) > 0:
            self._direct_local_failure_ticks += 1
        else:
            self._direct_local_failure_ticks = max(
                0, self._direct_local_failure_ticks - 1)
        # Cached / brake ratio over the interval (Cases G/H).
        cached_ratio = int(feedback.get("cached_frame_count", 0)) / \
            interval_frames
        brake_ratio = int(feedback.get("brake_frame_count", 0)) / \
            interval_frames
        if cached_ratio >= float(cf.get("cached_ratio_threshold", 0.5)):
            self._direct_cached_ticks += 1
        else:
            self._direct_cached_ticks = max(
                0, self._direct_cached_ticks - 1)
        if brake_ratio >= float(cf.get("brake_ratio_threshold", 0.5)):
            self._direct_brake_ticks += 1
        else:
            self._direct_brake_ticks = max(
                0, self._direct_brake_ticks - 1)

        evidence = (
            self._direct_local_failure_ticks >=
            int(cf.get("local_failure_macro_ticks", 2))
            or self._direct_cached_ticks >=
            int(cf.get("cached_macro_ticks", 2))
            or self._direct_brake_ticks >=
            int(cf.get("brake_macro_ticks", 2))
            or int(feedback.get("emergency_frame_count", 0)) >=
            int(cf.get("emergency_macro_ticks", 1)))
        # DIRECT long no-progress is CAUSAL evidence for macro intervention
        # — never an immediate episode FAILED (section VII / Case C).
        if self._direct_last_progress_time is not None and \
                self.direct_no_progress_time > \
                self.cfg.get("direct_intervention_timeout", 5.0):
            evidence = True
        self.causal_intervention_evidence = evidence

    # ── SIDE rolling strategic progress (sections VII/XV/XVIII) ──────
    def _update_side_progress(self, now_s, state):
        """Strategic progress = monotonic improvement of a strategic metric
        ONLY: global cost-to-go decrease (metric A, preferred) or committed-
        candidate distance decrease (metric B, fallback when no CTG).
        Ordinary travelled distance is recorded as a diagnostic but NEVER
        counts as progress, so left/right oscillation (Case F) cannot keep
        a side alive."""
        pos = np.asarray(state["position"], dtype=np.float64)
        made_progress = False
        ctg = float(self._oracle.cost_to_go(pos)) \
            if self._oracle.built() else float("inf")
        if math.isfinite(ctg):
            if self._side_best_cost is None or \
                    ctg < self._side_best_cost - \
                    self.cfg.get("progress_cost_epsilon_m", 0.10):
                self._side_best_cost = ctg
                made_progress = True
        elif self._side_target_world is not None:
            # No global CTG available: fall back to the distance to the
            # committed world candidate (metric B).
            d = float(np.linalg.norm(self._side_target_world - pos))
            if self._side_best_target_dist is None or \
                    d < self._side_best_target_dist - \
                    self.cfg.get("progress_cost_epsilon_m", 0.10):
                self._side_best_target_dist = d
                made_progress = True
        # Diagnostic only: travelled distance is NOT strategic progress.
        if self._side_last_pos is not None:
            self._side_path_progress += float(
                np.linalg.norm(pos - self._side_last_pos))
        self._side_last_pos = pos
        if made_progress:
            self._side_last_progress_time = now_s

    # ── DIRECT intent is observed-recoverable (no causal evidence) ───
    def _handle_direct(self, goal_dir_world, direct_guide, rec,
                       goal_dist, now_s):
        entering = not self._direct_session_active
        if entering:
            # Entering DIRECT from SIDE_GUIDE / OBSERVE: initialise a fresh
            # DIRECT causal window (section XXV).  A DIRECT->DIRECT tick
            # keeps the accumulated history — evidence is never cleared on
            # a plain DIRECT tick.
            if self.mode == self._module.MacroMode.OBSERVE:
                # Leaving OBSERVE (section X): the whole session — yaw
                # reference, baselines, scan exhaustion, stagnation,
                # anchor — is cleared so a later re-entry starts fresh.
                self._observe_time_s = 0.0
                self._clear_observe_session()
            self._enter_direct_session(goal_dist, now_s)
        self.mode = self._module.MacroMode.DIRECT_GUIDE
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

    # ── Macro intervention (observed local intent NOT recoverable) ───
    def _proceed_to_macro_intervention(self, goal, pos, yaw, direct_guide,
                                       observed_map, vs, dt_s, now_s):
        # OBSERVE runs while it keeps gaining information.  The ONLY
        # absolute limit is the large episode-level anti-deadlock guard
        # (section XXII); a short fixed observe duration is never a
        # terminal failure (sections XIX/XX/G).
        if self.mode == self._module.MacroMode.OBSERVE:
            self._observe_time_s += dt_s
            if self._observe_time_s > self.cfg.get(
                    "macro_intervention_absolute_safety_timeout", 45.0):
                self.mode = self._module.MacroMode.FAILED
                self._failed_reason = "observe_absolute_timeout"
                self._macro_decision_observable = False
                self._macro_decision_confidence = 0.0
                return self._build_action(
                    self.mode, pos, None, False, 0.0,
                    "observe_absolute_timeout")

        # ── Blocker + candidates (C++) ───────────────────────────────
        blocker = self._module.analyze_goal_blocker(
            observed_map, vs, goal, self._candidate_cfg)
        self._last_blocker = blocker
        # Side memory is bound to the ACTIVE blocker TRACK (section X):
        # stable world-geometry association, lost-grace and release-on-pass.
        self._update_blocker_tracking(blocker, now_s)
        prev_ptr = None
        if self._prev_guide_world is not None:
            prev_ptr = self._prev_guide_world
        candidates = self._candidate_search.generate_candidates(
            observed_map, vs, goal, blocker, prev_ptr)
        self._oracle.score_candidates(
            candidates, vs, goal, self.committed_side, prev_ptr)
        self._last_candidates = candidates

        # ── Candidate filters (section IV/V) ─────────────────────────
        # 1) observed FULL reachability (never PARTIAL), 2) privileged
        # global connectivity (a side that is observed-path-able but a
        # global dead-end is EXCLUDED).
        viable = {self._module.Side.LEFT: [],
                  self._module.Side.RIGHT: []}
        for candidate in candidates:
            if candidate.type != self._module.CandidateType.SIDE:
                continue
            if not candidate.full_goal_reached:
                continue
            if not candidate.connected_to_goal:
                continue
            if candidate.side in viable:
                viable[candidate.side].append(candidate)

        # ── Side failure memory (section VIII/XIV) ───────────────────
        self._update_side_failures(viable, now_s)

        # ── SIDE selection (section VI) ──────────────────────────────
        side = self._select_side(viable)

        if side is not None:
            candidate = self._best_candidate_for_side(viable[side], side)
            entering_side = (not self._side_session_active) or \
                (side != self.committed_side)
            if entering_side:
                self._enter_side_session(pos, now_s)
            self.committed_side = side
            self.mode = self._module.MacroMode.SIDE_GUIDE
            guide = direct_guide if candidate is None else \
                np.asarray(candidate.position_world)
            # Re-baseline the strategic target on entry or when the chosen
            # candidate moves to a new location (new strategic segment).
            target_changed = (
                candidate is not None and
                (self._side_target_world is None or
                 float(np.linalg.norm(
                     np.asarray(candidate.position_world) -
                     self._side_target_world)) > 0.5))
            if entering_side or target_changed:
                self._set_side_target(guide, pos)
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

        # ── No viable candidate -> OBSERVE (section XV) ──────────────
        # OBSERVE side is a PREFERENCE (committed if not failed, else first
        # non-failed), NOT a hard topology commitment: only a PROVEN-failed
        # side is excluded from movement (sections XVIII-XX).  If both sides
        # are failed -> NO_VALID_SIDE (side == None).
        side = self._observe_side()
        self.committed_side = side if side is not None \
            else self._module.Side.NONE
        pref_side = side

        # Rolling OBSERVE information-gain window (sections XVI-XVIII):
        # known cells (monotonic best), edge visibility (state change) and
        # newly FULL-reachable movement candidates count as progress.  The
        # movement counts include SIDE + OBSERVE + GOAL_FRONTIER FULL
        # reachability — an OBSERVE/FRONTIER viewpoint becoming reachable
        # is itself decision information (sections XII-XIV/XXXIX).
        # Per-category BEST baselines are monotonic so candidate-count
        # jitter (2 -> 1 -> 2) never produces repeated false progress
        # (sections XVII/XLIV), while a fresh session always writes the
        # CURRENT observed values as baselines — never None (sections
        # II-III/XIX/XXXVII).
        known = int(observed_map.known_count())
        edge_mask = (int(blocker.left_edge_visible),
                     int(blocker.right_edge_visible))
        side_full = len([c for c in candidates
                         if c.type == self._module.CandidateType.SIDE and
                         c.full_goal_reached])
        observe_full = len([c for c in candidates
                            if c.type ==
                            self._module.CandidateType.OBSERVE and
                            c.full_goal_reached])
        frontier_full = len([c for c in candidates
                             if c.type ==
                             self._module.CandidateType.GOAL_FRONTIER and
                             c.full_goal_reached])
        # A real OBSERVE_MOVE that travelled beyond the viewpoint reset
        # distance establishes a NEW observation anchor: scan exhaustion,
        # yaw reference, stagnation and info baselines are all re-based at
        # the current position (sections XXI-XXIV/XLII).  Small drifts
        # never reset scan state (section XLIII).
        anchor_moved = (
            self._observe_anchor_position_world is not None and
            float(np.linalg.norm(
                pos - self._observe_anchor_position_world)) >=
            float(self.cfg.get("viewpoint_reset_distance_m", 0.35)))
        info_gained = False
        if self.mode != self._module.MacroMode.OBSERVE or \
                self._observe_reference_yaw_world is None or anchor_moved:
            # Entering OBSERVE (fresh session) or the drone reached a new
            # observation viewpoint: start a fresh session whose baselines
            # are the CURRENT observed values (never None, so the next
            # tick's comparison can never hit `int > None`).  First entry
            # keeps the "session start" semantics of info_gained = True.
            self._exit_direct_session()
            self._side_session_active = False
            self._start_observe_session(
                pref_side, yaw, now_s,
                known_count=known, edge_mask=edge_mask,
                side_full_count=side_full,
                observe_full_count=observe_full,
                frontier_full_count=frontier_full,
                pos=pos)
            info_gained = True
        else:
            info_gained = self._observe_information_progress(
                known, edge_mask, side_full, observe_full, frontier_full,
                now_s)
        if self._observe_last_info_time is not None:
            self.observe_no_information_time = \
                now_s - self._observe_last_info_time
        else:
            self.observe_no_information_time = 0.0

        self.mode = self._module.MacroMode.OBSERVE

        # OBSERVE no-information timeout (section XVI): change strategy
        # (OBSERVE_MOVE) or try the other side — never an immediate FAILED.
        no_new_info = self.observe_no_information_time > \
            self.cfg.get("observe_no_information_timeout", 4.0)

        fov_half_obs = math.radians(self.cfg.get("observe_fov_half_deg", 45.0))
        rotation_exhausted = \
            abs(self._observe_yaw_delta) >= fov_half_obs - 1e-6

        # ── FULL-reachable movement candidates (sections XV/XIX) ─────
        # OBSERVE + GOAL_FRONTIER candidates whose observed LocalPathSearch
        # FULLY reached the viewpoint (never a straight-corridor proxy).  A
        # viewpoint may need a detour around known obstacles — that is fine,
        # the 30 Hz planner can route it.  Only PROVEN-failed sides are
        # hard-excluded; the observe preference never blocks the other side.
        min_move = self.cfg.get("min_observe_move_distance_m", 0.15)
        move_cands = []
        for c in candidates:
            if c.type not in (self._module.CandidateType.OBSERVE,
                              self._module.CandidateType.GOAL_FRONTIER):
                continue
            if not c.full_goal_reached or not c.known_reachable:
                continue
            # Safety net (section XIII): never emit a zero/near-zero
            # displacement observation "move" (C++ already filters).
            if float(np.linalg.norm(
                    np.asarray(c.position_world) - pos)) < min_move:
                continue
            if c.side == self._module.Side.LEFT and \
                    self._failed_left is not None:
                self.observe_reject_failed_side += 1
                continue
            if c.side == self._module.Side.RIGHT and \
                    self._failed_right is not None:
                self.observe_reject_failed_side += 1
                continue
            move_cands.append(c)
        # Valid movement counts per side (diagnostics).
        self.observe_left_valid_count = len([c for c in move_cands
            if c.side == self._module.Side.LEFT])
        self.observe_right_valid_count = len([c for c in move_cands
            if c.side == self._module.Side.RIGHT])
        self.observe_center_valid_count = len([c for c in move_cands
            if c.side == self._module.Side.NONE])

        # OBSERVED_NO_CORRIDOR (section XIV/XLV): ONLY with strong evidence —
        # the preferred side's edge is ACTUALLY visible, it has been observed
        # for a while with no new info AND no usable movement candidate
        # remains.  Scan exhaustion alone NEVER marks a side failed.
        if no_new_info and pref_side is not None and not move_cands:
            edge_visible = (
                (pref_side == self._module.Side.LEFT and
                 bool(blocker.left_edge_visible)) or
                (pref_side == self._module.Side.RIGHT and
                 bool(blocker.right_edge_visible)))
            if edge_visible:
                self._mark_side_failed(pref_side,
                                       SideFailure.OBSERVED_NO_CORRIDOR)
                pref_side = self._observe_side()

        # ── NO_VALID_SIDE (section XVII) ─────────────────────────────
        if pref_side is None:
            priv = self._last_intervention
            route_ok = priv is not None and self._oracle.built() and \
                priv.reason != self._module.InterventionReason.NO_GLOBAL_ROUTE
            if not route_ok:
                # Privileged global map also confirms no route: FAILED.
                self.mode = self._module.MacroMode.FAILED
                self._failed_reason = "no_valid_side_no_route"
                self._macro_decision_observable = False
                self._macro_decision_confidence = 0.0
                return self._build_action(
                    self.mode, pos, None, False, 0.0, "no_valid_side_no_route")
            # Unknown space remains: keep OBSERVE facing forward, never
            # re-selecting a failed side.
            self.committed_side = self._module.Side.NONE

        # ── Active observation decision (section LXIX) ───────────────
        # Movement candidate with preference + fallback: preferred side,
        # then centre / neutral frontier, then the other non-failed side.
        # Once both scans are exhausted this is exactly the expanded search
        # over every non-failed viewpoint (sections XXX-XXXI).
        best_move = self._best_observe_move(move_cands, pos, pref_side)
        clearly_better = (best_move is not None and
                          float(best_move.unknown_information_gain) >=
                          self.cfg.get("observe_info_gain_move_threshold",
                                       0.05))
        stagnant = self._observe_stagnant_time > \
            self.cfg.get("max_stagnant_rotate_s", 2.0)

        if not self._both_scans_exhausted() and not stagnant and \
                ((not no_new_info) or (not rotation_exhausted)):
            # Rotation still yields information (or is still sweeping): keep
            # OBSERVE_ROTATE unless a clearly better viewpoint exists
            # (sections LXVIII/LXIX).
            if clearly_better:
                guide = np.asarray(best_move.position_world)
                desired_yaw_world = normalize_angle(
                    math.atan2(guide[1] - pos[1], guide[0] - pos[0]) -
                    0.5 * math.pi)
                observe_subtype = 1
            else:
                guide, desired_yaw_world, observe_subtype = \
                    self._observe_rotate_action(pos, yaw, dt_s, fov_half_obs)
        elif best_move is not None:
            # Active perception target: preferred side, then centre, then
            # other non-failed side (expanded search included).
            guide = np.asarray(best_move.position_world)
            desired_yaw_world = normalize_angle(
                math.atan2(guide[1] - pos[1], guide[0] - pos[0]) -
                0.5 * math.pi)
            observe_subtype = 1
        elif not self._both_scans_exhausted():
            # No movement candidate: advance the per-side scan lifecycle
            # (sections XXVI-XXXI).  Exhausted rotation + no new info marks
            # the current scan exhausted and hands over to the OTHER
            # non-failed side with a fresh sweep reference.
            if rotation_exhausted or no_new_info or stagnant:
                if not self._scan_side_is_exhausted():
                    self._mark_current_scan_exhausted()
                other = self._other_scan_side()
                if other is not None:
                    self._switch_observe_scan_side(other, yaw, now_s)
                    guide, desired_yaw_world, observe_subtype = \
                        self._observe_rotate_action(pos, yaw, dt_s,
                                                    fov_half_obs)
                else:
                    # No other usable side: run a RECONSIDERATION sweep from
                    # the current yaw so the yaw target keeps changing (the
                    # observed map can grow; never a fixed-yaw loop).
                    self._observe_reference_yaw_world = yaw
                    self._observe_yaw_delta = 0.0
                    self._observe_stagnant_time = 0.0
                    guide, desired_yaw_world, observe_subtype = \
                        self._observe_rotate_action(pos, yaw, dt_s,
                                                    fov_half_obs)
            else:
                guide, desired_yaw_world, observe_subtype = \
                    self._observe_rotate_action(pos, yaw, dt_s, fov_half_obs)
        else:
            # BOTH scans exhausted and no FULL-reachable viewpoint anywhere
            # (section XXIX/XXXII): NEVER repeat the same fixed yaw for tens
            # of seconds.  Expanded search already produced nothing, so run a
            # reconsideration sweep from the current yaw — a fresh sweep
            # phase keeps the yaw target changing while the map grows.
            self._observe_reference_yaw_world = yaw
            self._observe_yaw_delta = 0.0
            self._observe_stagnant_time = 0.0
            guide, desired_yaw_world, observe_subtype = \
                self._observe_rotate_action(pos, yaw, dt_s, fov_half_obs)

        # Stagnation bookkeeping for the NEXT tick (fixed-yaw + no-move +
        # no-info accumulation; section XXXIII/XXXV).
        self._update_observe_stagnation(
            desired_yaw_world, observe_subtype == 0, info_gained, dt_s)

        # Selected-viewpoint diagnostics (section XLIX).
        if observe_subtype == 1 and best_move is not None:
            self._prev_observe_viewpoint = np.asarray(guide)
            self.observe_selected_source = best_move.source
            self.observe_selected_side = int(best_move.side)
            self.observe_selected_distance = round(float(np.linalg.norm(
                np.asarray(best_move.position_world) - pos)), 3)
            self.observe_selected_path_length = round(
                float(best_move.observed_path_length), 3)
            self.observe_selected_info_gain = round(
                float(best_move.unknown_information_gain), 4)
            self.observe_selected_clearance = round(
                float(best_move.minimum_clearance), 3)
        else:
            self._prev_observe_viewpoint = None
            self.observe_selected_source = ""
            self.observe_selected_side = 0
            self.observe_selected_distance = 0.0
            self.observe_selected_path_length = 0.0
            self.observe_selected_info_gain = 0.0
            self.observe_selected_clearance = 0.0

        # ── OBSERVE per-tick diagnostics (section XLVII/XLVIII) ──────
        self.observe_scan_side = int(self._observe_scan_side) \
            if self._observe_scan_side is not None else 0
        self.left_scan_exhausted = int(self._left_scan_exhausted)
        self.right_scan_exhausted = int(self._right_scan_exhausted)
        self.observe_rotation_exhausted = int(rotation_exhausted)
        self.observe_stagnant_rotate_time = round(
            self._observe_stagnant_time, 3)
        self.observe_anchor_distance = round(float(
            np.linalg.norm(pos - self._observe_anchor_position_world)), 3) \
            if self._observe_anchor_position_world is not None else 0.0
        diag = self._candidate_search.last_observe_diagnostics()
        self.observe_raw_candidate_count = int(diag.raw_candidate_count)
        self.observe_lattice_candidate_count = int(diag.lattice_candidate_count)
        self.observe_frontier_candidate_count = int(diag.frontier_candidate_count)
        self.observe_endpoint_known_free_count = int(
            diag.endpoint_known_free_count)
        self.observe_local_full_count = int(diag.full_local_count)
        self.observe_reject_unknown = int(diag.reject_unknown)
        self.observe_reject_endpoint_clearance = int(
            diag.reject_endpoint_clearance)
        self.observe_reject_min_distance = int(diag.reject_min_distance)
        self.observe_reject_max_distance = int(diag.reject_max_distance)
        self.observe_reject_partial = int(diag.partial_count)
        self.observe_reject_no_path = int(diag.no_path_count)

        side_name = "left" if pref_side == self._module.Side.LEFT else \
            ("right" if pref_side == self._module.Side.RIGHT else "forward")
        action = self._build_action(
            self.mode, guide, desired_yaw_world, True, 0.5,
            "observe_%s" % side_name, observe_side=pref_side)
        action.observe_subtype = observe_subtype
        self._macro_decision_observable = False
        self._macro_decision_confidence = 0.5
        self._prev_guide_world = np.asarray(guide)
        return action

    # ── Session enter helpers (section XXV) ──────────────────────────
    def _enter_side_session(self, pos, now_s):
        """Initialise SIDE rolling strategic metrics once when entering
        SIDE (or switching committed side).  Never called on a plain
        SIDE->SIDE tick, so no-progress accumulation is preserved."""
        self._side_session_active = True
        self._exit_direct_session()
        # Leaving OBSERVE (section X): full session clear so a later
        # re-entry starts fresh.
        self._observe_time_s = 0.0
        self._clear_observe_session()
        pos_ctg = float(self._oracle.cost_to_go(pos)) \
            if self._oracle.built() else float("inf")
        self._side_best_cost = pos_ctg if math.isfinite(pos_ctg) else None
        self._side_best_target_dist = None
        self._side_last_progress_time = now_s
        self._side_last_pos = pos
        self._side_path_progress = 0.0
        self._side_target_world = None

    def _set_side_target(self, guide, pos):
        """Rebaseline the committed world candidate used by metric B."""
        self._side_target_world = np.asarray(guide)
        self._side_best_target_dist = float(
            np.linalg.norm(self._side_target_world - pos))

    # ── Blocker tracking (sections V-XI) ─────────────────────────────
    def _update_blocker_tracking(self, blocker, now_s):
        """Associate the current GoalBlocker with the ACTIVE blocker track
        using world geometry (centroid distance + expanded bbox overlap).
        - same blocker          -> keep side memory
        - confirmed new blocker -> fresh side state (section IX)
        - short loss            -> grace keep (section X / Case C)
        - confirmed disappearance -> release side memory (section XI)
        """
        if not blocker.found:
            self._blocker_matches_current = False
            if self._blocker_track is not None:
                if self._blocker_lost_since is None:
                    self._blocker_lost_since = now_s
                if now_s - self._blocker_lost_since > \
                        self.cfg.get("blocker_lost_grace_s", 2.0):
                    # Confirmed passed: release side memory (section XI).
                    self._release_blocker()
            return

        centroid = np.asarray(blocker.centroid, dtype=np.float64)
        bbox_min = np.asarray(blocker.bbox_min_world, dtype=np.float64)
        bbox_max = np.asarray(blocker.bbox_max_world, dtype=np.float64)

        if self._blocker_track is None:
            if self._blocker_lost_since is not None and \
                    now_s - self._blocker_lost_since <= \
                    self.cfg.get("blocker_lost_grace_s", 2.0):
                # Reappeared within grace: same blocker (Case C).
                self._blocker_lost_since = None
                self._refresh_blocker_track(centroid, bbox_min, bbox_max,
                                            blocker, now_s)
                self._blocker_matches_current = True
            else:
                self._create_blocker_track(centroid, bbox_min, bbox_max,
                                           blocker, now_s)
                self._blocker_matches_current = True
            return

        if self._blocker_lost_since is not None:
            # Back within grace: same blocker.
            self._blocker_lost_since = None
            self._refresh_blocker_track(centroid, bbox_min, bbox_max,
                                        blocker, now_s)
            self._blocker_matches_current = True
            return

        if self._blocker_associated(self._blocker_track, centroid,
                                    bbox_min, bbox_max):
            self._refresh_blocker_track(centroid, bbox_min, bbox_max,
                                        blocker, now_s)
            self._blocker_switch_ticks = 0
            self._blocker_new_pending = None
            self._blocker_matches_current = True
            return

        # Different blocker candidate: confirm with hysteresis before
        # switching side memory (section IX).
        pending = {
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "ray_depth": float(blocker.blocking_ray_depth),
        }
        if self._blocker_new_pending is not None and \
                self._blocker_associated(self._blocker_new_pending,
                                         centroid, bbox_min, bbox_max):
            self._blocker_switch_ticks += 1
        else:
            self._blocker_new_pending = pending
            self._blocker_switch_ticks = 1
        if self._blocker_switch_ticks >= self.cfg.get(
                "blocker_rebind_ticks", 2):
            self._create_blocker_track(centroid, bbox_min, bbox_max,
                                       blocker, now_s)
            self._blocker_new_pending = None
            self._blocker_switch_ticks = 0
            self._blocker_matches_current = True
        else:
            # Not yet confirmed: the current blocker is treated as NOT the
            # active one (a single off-track observation never wipes the
            # side memory or counts as a same-blocker consecutive failure).
            self._blocker_matches_current = False

    def _blocker_associated(self, track, centroid, bbox_min, bbox_max):
        """Association (section VII): centroid distance + expanded bbox
        overlap.  The observed component grows as the map expands, so a
        small overlap pad is used."""
        assoc_dist = self.cfg.get("blocker_association_distance_m", 1.5)
        centroid_dist = float(
            np.linalg.norm(track["centroid"] - centroid))
        if centroid_dist > assoc_dist:
            return False
        pad = self.cfg.get("blocker_overlap_pad_m", 0.5)
        a_min = track["bbox_min"][:2] - pad
        a_max = track["bbox_max"][:2] + pad
        b_min = bbox_min[:2]
        b_max = bbox_max[:2]
        ov_min = np.maximum(a_min, b_min)
        ov_max = np.minimum(a_max, b_max)
        overlap = float(np.prod(np.maximum(0.0, ov_max - ov_min)))
        if overlap <= 0.0:
            # No overlap: accept only if the centroids are very close.
            return centroid_dist <= 0.5 * assoc_dist
        return True

    def _create_blocker_track(self, centroid, bbox_min, bbox_max,
                              blocker, now_s):
        self._next_blocker_id += 1
        self._blocker_track = {
            "id": self._next_blocker_id,
            "centroid": centroid,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "ray_depth": float(blocker.blocking_ray_depth),
            "extent": float(blocker.extent),
            "last_seen": now_s,
        }
        self._blocker_lost_since = None
        self._blocker_new_pending = None
        self._blocker_switch_ticks = 0
        # New independent blocker: fresh side state (section IX/Case B).
        self._reset_side_memory()

    def _refresh_blocker_track(self, centroid, bbox_min, bbox_max,
                               blocker, now_s):
        if self._blocker_track is None:
            return
        self._blocker_track["centroid"] = centroid
        self._blocker_track["bbox_min"] = bbox_min
        self._blocker_track["bbox_max"] = bbox_max
        self._blocker_track["ray_depth"] = float(blocker.blocking_ray_depth)
        self._blocker_track["extent"] = float(blocker.extent)
        self._blocker_track["last_seen"] = now_s

    def _release_blocker(self):
        """Old blocker confirmed passed (sections XI/XII): release side
        memory and ALL macro-intervention topology state so they cannot
        bias navigation toward the next obstacle.  Does NOT touch episode
        statistics, goal, observed map or trajectory cache."""
        self._blocker_track = None
        self._blocker_lost_since = None
        self._blocker_new_pending = None
        self._blocker_switch_ticks = 0
        self._blocker_matches_current = False
        self._direct_stable_ticks = 0
        # Observe session (section XII): fresh absolute budget; the full
        # session clear runs inside _reset_side_memory (new topology).
        self._observe_time_s = 0.0
        self._reset_side_memory()

    def _reset_side_memory(self):
        self.committed_side = self._module.Side.NONE
        self._failed_left = None
        self._failed_right = None
        self._side_local_fail_count = {
            self._module.Side.LEFT: 0,
            self._module.Side.RIGHT: 0,
        }
        self._side_session_active = False
        self._side_best_cost = None
        self._side_target_world = None
        self._side_best_target_dist = None
        self._side_last_progress_time = None
        self._side_last_pos = None
        self._side_path_progress = 0.0
        # Fresh topology -> fresh active-observation session (section
        # XXVIII/XL): old blocker yaw reference, scan exhaustion, info
        # baselines and anchor must never leak into the new obstacle.
        self._clear_observe_session()

    # ── Side failure memory (section VIII/XIV) ───────────────────────
    def _mark_side_failed(self, side, reason):
        if side == self._module.Side.LEFT:
            self._failed_left = reason
        elif side == self._module.Side.RIGHT:
            self._failed_right = reason

    def _update_side_failures(self, viable, now_s):
        local_fail_threshold = self.cfg.get(
            "local_path_fail_threshold", 2)
        # PRIVILEGED_DISCONNECTED: an observed-FULL candidate exists on a
        # side but the global map says it is a dead end -> confirmed fail.
        for side in (self._module.Side.LEFT, self._module.Side.RIGHT):
            observed_full = any(
                c.type == self._module.CandidateType.SIDE and
                c.side == side and c.full_goal_reached
                for c in self._last_candidates)
            if observed_full and not viable[side]:
                self._mark_side_failed(side,
                                       SideFailure.PRIVILEGED_DISCONNECTED)
        # LOCAL_PATH_FAILED: only confirmed after CONSECUTIVE failures of
        # the SAME side with the ACTIVE blocker (section XIV) — a single
        # planning failure never marks a side failed.
        if self.mode == self._module.MacroMode.SIDE_GUIDE and \
                self.committed_side in (self._module.Side.LEFT,
                                        self._module.Side.RIGHT):
            committed = self.committed_side
            if not viable[committed]:
                same_blocker = bool(self._blocker_matches_current)
                if same_blocker:
                    self._side_local_fail_count[committed] += 1
                    if self._side_local_fail_count[committed] >= \
                            local_fail_threshold:
                        self._mark_side_failed(
                            committed, SideFailure.LOCAL_PATH_FAILED)
                else:
                    self._side_local_fail_count[committed] = 1
            else:
                self._side_local_fail_count[committed] = 0
        # NO_PROGRESS (section XVIII): the committed SIDE still HAS a
        # viable candidate but strategic rolling progress has stalled for
        # `side_no_progress_seconds`.  `_side_last_progress_time` is only
        # advanced by _update_side_progress on a REAL strategic metric
        # improvement, so oscillation eventually triggers this.
        if self.mode == self._module.MacroMode.SIDE_GUIDE and \
                self.committed_side in (self._module.Side.LEFT,
                                        self._module.Side.RIGHT) and \
                self._side_last_progress_time is not None and \
                viable[self.committed_side] and \
                now_s - self._side_last_progress_time > \
                self.cfg.get("side_no_progress_seconds", 6.0):
            self._mark_side_failed(self.committed_side,
                                   SideFailure.NO_PROGRESS)

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
        # Both sides failed -> NO_VALID_SIDE (never re-select a failed one).
        return None

    # ── Active-observation scan lifecycle (sections XXI-XXXV) ───────
    def _clear_observe_session(self):
        """Drop the ENTIRE active-observation session (leave OBSERVE /
        confirmed new blocker / episode reset).  Scan exhaustion, sweep
        reference, info baselines, stagnation and the observation anchor
        are all per-obstacle topology — never global — so a new blocker or
        a later re-entry into OBSERVE must start fully fresh (sections
        VII-XI/XL).  The absolute OBSERVE time budget `_observe_time_s` is
        deliberately NOT part of this clear: it is the final anti-deadlock
        guard and is reset only when the mode leaves OBSERVE or the blocker
        is released (section XLV)."""
        self._observe_scan_side = None
        self._left_scan_exhausted = False
        self._right_scan_exhausted = False
        self._observe_reference_yaw_world = None
        self._observe_yaw_delta = 0.0
        self._observe_best_known = None
        self._observe_last_edge_mask = None
        self._observe_best_side_count = None
        self._observe_best_observe_count = None
        self._observe_best_frontier_count = None
        self._observe_last_info_time = None
        self.observe_no_information_time = 0.0
        self._observe_stagnant_time = 0.0
        self._observe_last_target_yaw = None
        self._observe_last_move_guide = None
        self._prev_observe_viewpoint = None
        self._observe_anchor_position_world = None
        self.observe_session_initialized = 0
        self.observe_anchor_distance = 0.0

    def _reset_observe_diagnostics(self):
        """Zero the per-tick OBSERVE diagnostics (section XLVII-XLIX).  The
        OBSERVE branch repopulates them; every other mode records zeros."""
        self.observe_scan_side = 0
        self.left_scan_exhausted = 0
        self.right_scan_exhausted = 0
        self.observe_rotation_exhausted = 0
        self.observe_stagnant_rotate_time = 0.0
        self.observe_raw_candidate_count = 0
        self.observe_lattice_candidate_count = 0
        self.observe_frontier_candidate_count = 0
        self.observe_endpoint_known_free_count = 0
        self.observe_local_full_count = 0
        self.observe_reject_unknown = 0
        self.observe_reject_endpoint_clearance = 0
        self.observe_reject_min_distance = 0
        self.observe_reject_max_distance = 0
        self.observe_reject_partial = 0
        self.observe_reject_no_path = 0
        self.observe_reject_failed_side = 0
        self.observe_left_valid_count = 0
        self.observe_right_valid_count = 0
        self.observe_center_valid_count = 0
        self.observe_selected_source = ""
        self.observe_selected_side = 0
        self.observe_selected_distance = 0.0
        self.observe_selected_path_length = 0.0
        self.observe_selected_info_gain = 0.0
        self.observe_selected_clearance = 0.0
        self.observe_session_initialized = 0
        self.observe_anchor_distance = 0.0

    def _start_observe_session(self, preferred_side, yaw_world, now_s,
                               known_count, edge_mask, side_full_count,
                               observe_full_count, frontier_full_count, pos):
        """Establish a fresh OBSERVE session (OBSERVE entry / new blocker /
        confirmed OBSERVE_MOVE to a new observation anchor).  The session's
        baselines are the CURRENT observed values — never None — so the
        next tick's information comparison can never crash with
        `int > None` and a fresh session never reports fake progress
        (sections II-III/XIX/XXXVII-XXXVIII).  The preferred side becomes
        the scan side and the sweep reference is the current yaw."""
        self._clear_observe_session()
        self._observe_scan_side = preferred_side
        self._observe_reference_yaw_world = yaw_world
        self._observe_yaw_delta = 0.0
        self._observe_anchor_position_world = np.asarray(pos,
                                                         dtype=np.float64)
        self._observe_best_known = int(known_count)
        self._observe_last_edge_mask = edge_mask
        self._observe_best_side_count = int(side_full_count)
        self._observe_best_observe_count = int(observe_full_count)
        self._observe_best_frontier_count = int(frontier_full_count)
        self._observe_last_info_time = now_s
        self.observe_no_information_time = 0.0
        self._observe_stagnant_time = 0.0
        self._observe_last_target_yaw = None
        self._observe_last_move_guide = None
        self._prev_observe_viewpoint = None
        self.observe_session_initialized = 1
        self.observe_anchor_distance = 0.0

    def _observe_information_progress(self, known_count, edge_mask,
                                      side_full_count, observe_full_count,
                                      frontier_full_count, now_s):
        """Compare the CURRENT observed state against the session baseline
        and report genuine information progress (sections XVI-XVIII):
          - known cells: monotonic best (session-local; a fresh session
            re-bases it, so viewpoint changes never leave a stale best),
          - edge visibility: state change,
          - FULL-reachable movement counts: monotonic per-category best
            for SIDE + OBSERVE + GOAL_FRONTIER, so a newly reachable
            observation viewpoint counts as progress (section XXXIX) while
            candidate-count jitter never does (section XLIV).
        Defensive fallback (section V): a half-initialised session — which
        should be impossible after `_start_observe_session` — is healed by
        adopting the current values instead of raising `int > None`."""
        if self._observe_best_known is None:
            self._observe_best_known = int(known_count)
        if self._observe_last_edge_mask is None:
            self._observe_last_edge_mask = edge_mask
        if self._observe_best_side_count is None:
            self._observe_best_side_count = int(side_full_count)
        if self._observe_best_observe_count is None:
            self._observe_best_observe_count = int(observe_full_count)
        if self._observe_best_frontier_count is None:
            self._observe_best_frontier_count = int(frontier_full_count)
        if self._observe_last_info_time is None:
            self._observe_last_info_time = now_s

        progress = (
            int(known_count) > self._observe_best_known
            or edge_mask != self._observe_last_edge_mask
            or int(side_full_count) > self._observe_best_side_count
            or int(observe_full_count) > self._observe_best_observe_count
            or int(frontier_full_count) > self._observe_best_frontier_count
        )
        if progress:
            self._observe_best_known = max(
                self._observe_best_known, int(known_count))
            self._observe_last_edge_mask = edge_mask
            self._observe_best_side_count = max(
                self._observe_best_side_count, int(side_full_count))
            self._observe_best_observe_count = max(
                self._observe_best_observe_count, int(observe_full_count))
            self._observe_best_frontier_count = max(
                self._observe_best_frontier_count, int(frontier_full_count))
            self._observe_last_info_time = now_s
        return progress

    def _scan_side_is_exhausted(self):
        if self._observe_scan_side == self._module.Side.LEFT:
            return self._left_scan_exhausted
        if self._observe_scan_side == self._module.Side.RIGHT:
            return self._right_scan_exhausted
        return False

    def _both_scans_exhausted(self):
        return self._left_scan_exhausted and self._right_scan_exhausted

    def _mark_current_scan_exhausted(self):
        if self._observe_scan_side == self._module.Side.LEFT:
            self._left_scan_exhausted = True
        elif self._observe_scan_side == self._module.Side.RIGHT:
            self._right_scan_exhausted = True

    def _other_scan_side(self):
        """The other non-failed, non-exhausted side, or None."""
        if self._observe_scan_side == self._module.Side.LEFT:
            if not self._right_scan_exhausted and self._failed_right is None:
                return self._module.Side.RIGHT
            return None
        if self._observe_scan_side == self._module.Side.RIGHT:
            if not self._left_scan_exhausted and self._failed_left is None:
                return self._module.Side.LEFT
            return None
        # No scan side yet: first non-failed, non-exhausted side.
        if not self._left_scan_exhausted and self._failed_left is None:
            return self._module.Side.LEFT
        if not self._right_scan_exhausted and self._failed_right is None:
            return self._module.Side.RIGHT
        return None

    def _switch_observe_scan_side(self, side, yaw, now_s):
        """Switch the scan side LEFT <-> RIGHT within an ACTIVE OBSERVE
        session (section XXVI): only the sweep reference (current yaw),
        yaw delta, info timer and stagnation window are re-based.  The
        session baselines, blocker and failed-side memory are untouched."""
        self._observe_scan_side = side
        self._observe_reference_yaw_world = yaw
        self._observe_yaw_delta = 0.0
        self._observe_stagnant_time = 0.0
        self._observe_last_target_yaw = yaw
        self._observe_last_info_time = now_s
        self.observe_no_information_time = 0.0

    def _observe_rotate_action(self, pos, yaw, dt_s, fov_half_obs):
        """OBSERVE_ROTATE: pure rotation about the current position toward
        the sweep target of the CURRENT scan side (guide == position, zero
        translation).  Returns (guide, desired_yaw_world, subtype)."""
        scan_side = self._observe_scan_side
        if scan_side is not None:
            sign = 1.0 if scan_side == self._module.Side.LEFT else -1.0
            sweep_step = self.cfg.get("observe_rotation_rate_rps", 1.5) * dt_s
            self._observe_yaw_delta += sign * sweep_step
            self._observe_yaw_delta = \
                sign * min(abs(self._observe_yaw_delta), fov_half_obs)
            desired_yaw_world = normalize_angle(
                self._observe_reference_yaw_world + self._observe_yaw_delta)
        else:
            desired_yaw_world = yaw
        return pos, desired_yaw_world, 0

    def _update_observe_stagnation(self, target_yaw, rotating, info_gained,
                                   dt_s):
        """No-action deadlock detector (section XXXIII/XXXV): consecutive
        macro ticks with a FIXED yaw target, no translation and no new
        information accumulate `_observe_stagnant_time`; any change (target
        moved, a move was issued, information arrived) resets it."""
        fixed_yaw = (self._observe_last_target_yaw is not None and
                     abs(normalize_angle(
                         target_yaw - self._observe_last_target_yaw)) < 1e-3)
        if rotating and not info_gained and fixed_yaw:
            self._observe_stagnant_time += dt_s
        else:
            self._observe_stagnant_time = 0.0
        self._observe_last_target_yaw = target_yaw

    def _rank_observe_moves(self, cands, pos):
        """Score FULL-reachable OBSERVE/GOAL_FRONTIER movement candidates:
        information gain (primary) + goal progress + clearance - path cost,
        plus a current-scan-side preference bonus and a hysteresis bonus for
        the previously selected viewpoint (section XVII/XLI).  Returns the
        candidates best-first."""
        pref = self._observe_scan_side
        hyst_m = self.cfg.get("observe_viewpoint_hysteresis_m", 0.8)
        hyst_bonus = self.cfg.get("observe_viewpoint_hysteresis_bonus", 0.15)
        lookahead = max(1.0, self.cfg.get("macro_lookahead_distance_m", 4.5))
        scored = []
        for c in cands:
            cpos = np.asarray(c.position_world, dtype=np.float64)
            score = float(c.unknown_information_gain)
            score += 0.15 * float(c.goal_progress) / lookahead
            score += 0.05 * max(0.0, float(c.minimum_clearance))
            score -= 0.02 * float(c.observed_path_length)
            if pref is not None and c.side == pref:
                score += 0.10
            if self._prev_observe_viewpoint is not None and \
                    float(np.linalg.norm(
                        cpos - self._prev_observe_viewpoint)) <= hyst_m:
                score += hyst_bonus
            scored.append((score, c))
        scored.sort(key=lambda t: -t[0])
        return [c for _, c in scored]

    def _best_observe_move(self, move_cands, pos, prefer_side):
        """Pick the best movement candidate with preference + fallback
        (sections XVIII/XIX): preferred side first, then centre / neutral
        frontier, then the other non-failed side.  Proven-failed sides are
        already hard-excluded from `move_cands` (section XX)."""
        if not move_cands:
            return None
        ranked = self._rank_observe_moves(move_cands, pos)
        tier1 = [c for c in ranked if c.side == prefer_side]
        tier2 = [c for c in ranked
                 if c.side == self._module.Side.NONE]
        tier3 = [c for c in ranked
                 if c.side != prefer_side and
                 c.side != self._module.Side.NONE]
        for tier in (tier1, tier2, tier3):
            if tier:
                return tier[0]
        return ranked[0] if ranked else None

    # ── SIDE selection (section VI/VIII) ─────────────────────────────
    def _select_side(self, viable):
        """Choose the side to commit among observed-FULL + global-connected
        candidates.  Failed sides are MASKED FIRST (section VIII): a failed
        LEFT/RIGHT can never be re-selected, so LEFT NO_PROGRESS cannot be
        followed by a LEFT re-commit.  Keep the committed side if still
        valid; never switch on small cost fluctuations; fixed LEFT tie."""
        left = list(viable[self._module.Side.LEFT]) \
            if self._failed_left is None else []
        right = list(viable[self._module.Side.RIGHT]) \
            if self._failed_right is None else []
        if self.committed_side == self._module.Side.LEFT and left:
            return self._module.Side.LEFT
        if self.committed_side == self._module.Side.RIGHT and right:
            return self._module.Side.RIGHT
        if left and not right:
            return self._module.Side.LEFT
        if right and not left:
            return self._module.Side.RIGHT
        if left and right:
            left_cost = min(float(c.global_cost_to_go) for c in left)
            right_cost = min(float(c.global_cost_to_go) for c in right)
            margin = abs(left_cost - right_cost)
            cost_margin = self.cfg.get("cost_margin_m", 2.0)
            if margin >= cost_margin:
                return self._module.Side.LEFT \
                    if left_cost < right_cost else self._module.Side.RIGHT
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

    @property
    def blocker_track_id(self):
        """Stable id of the ACTIVE blocker track (diagnostic)."""
        if self._blocker_track is not None:
            return int(self._blocker_track["id"])
        return -1

    # ── Debug trace (section "debug") ────────────────────────────────
    @staticmethod
    def _round_opt(value, ndigits=4):
        if value is None:
            return None
        try:
            return round(float(value), ndigits)
        except (TypeError, ValueError):
            return None

    def _capture_trace(self, action):
        """Snapshot the macro expert's internal state for this tick.  The
        manager writes it to trace.jsonl when dataset_logging.debug_trace
        is enabled; it is NEVER part of the student input."""
        rec = self._last_recoverability
        priv = self._last_intervention
        trace = {
            "mode": int(action.mode),
            "mode_name": _MACRO_MODE_NAMES.get(int(action.mode), "UNKNOWN"),
            "reason": str(action.reason),
            "committed_side": _SIDE_NAMES.get(int(action.committed_side), "?"),
            "observe_side": _SIDE_NAMES.get(int(action.observe_side), "?"),
            "observe_subtype": int(action.observe_subtype),
            "confidence": float(action.confidence),
            "macro_decision_observable": bool(self._macro_decision_observable),
            "causal_intervention_evidence": bool(self.causal_intervention_evidence),
            "direct_no_progress_time": self._round_opt(self.direct_no_progress_time),
            "direct_stable_ticks": int(self._direct_stable_ticks),
            "direct_local_failure_ticks": int(self._direct_local_failure_ticks),
            "direct_cached_ticks": int(self._direct_cached_ticks),
            "direct_brake_ticks": int(self._direct_brake_ticks),
            "blocker_released_this_tick": bool(self._blocker_released_this_tick),
            "failed_left": _SIDE_FAILURE_NAMES.get(
                self._failed_left.value, "NONE") if self._failed_left is not None
            else "NONE",
            "failed_right": _SIDE_FAILURE_NAMES.get(
                self._failed_right.value, "NONE") if self._failed_right is not None
            else "NONE",
            "side_best_cost": self._round_opt(self._side_best_cost),
            "side_path_progress": self._round_opt(self._side_path_progress),
            "observe_time_s": self._round_opt(self._observe_time_s),
            "observe_yaw_delta_deg": self._round_opt(
                math.degrees(float(self._observe_yaw_delta)), 2),
            "observe_no_information_time": self._round_opt(
                self.observe_no_information_time),
            "observe_scan_side": int(self.observe_scan_side),
            "left_scan_exhausted": bool(self.left_scan_exhausted),
            "right_scan_exhausted": bool(self.right_scan_exhausted),
            "observe_rotation_exhausted": bool(self.observe_rotation_exhausted),
            "observe_stagnant_rotate_time": self._round_opt(
                self.observe_stagnant_rotate_time),
            "observe_raw_candidate_count": int(self.observe_raw_candidate_count),
            "observe_lattice_candidate_count": int(
                self.observe_lattice_candidate_count),
            "observe_frontier_candidate_count": int(
                self.observe_frontier_candidate_count),
            "observe_local_full_count": int(self.observe_local_full_count),
            "observe_reject_unknown": int(self.observe_reject_unknown),
            "observe_reject_endpoint_clearance": int(
                self.observe_reject_endpoint_clearance),
            "observe_reject_partial": int(self.observe_reject_partial),
            "observe_reject_no_path": int(self.observe_reject_no_path),
            "observe_reject_failed_side": int(self.observe_reject_failed_side),
            "observe_left_valid_count": int(self.observe_left_valid_count),
            "observe_right_valid_count": int(self.observe_right_valid_count),
            "observe_center_valid_count": int(self.observe_center_valid_count),
            "observe_selected_source": str(self.observe_selected_source),
            "observe_selected_side": int(self.observe_selected_side),
            "observe_selected_distance": self._round_opt(
                self.observe_selected_distance),
            "observe_selected_path_length": self._round_opt(
                self.observe_selected_path_length),
            "observe_selected_info_gain": self._round_opt(
                self.observe_selected_info_gain),
            "observe_selected_clearance": self._round_opt(
                self.observe_selected_clearance),
            "observe_session_initialized": int(self.observe_session_initialized),
            "observe_anchor_distance": self._round_opt(
                self.observe_anchor_distance),
            "observe_best_known": self._observe_best_known,
            "observe_best_side_count": self._observe_best_side_count,
            "observe_best_observe_count": self._observe_best_observe_count,
            "observe_best_frontier_count": self._observe_best_frontier_count,
            "blocker_track_id": self.blocker_track_id,
            "blocker_matches_current": bool(self._blocker_matches_current),
            "rec_status": int(rec.status) if rec is not None else -1,
            "rec_min_clearance": self._round_opt(
                rec.minimum_clearance) if rec is not None else -1.0,
            "rec_detour_ratio": self._round_opt(
                rec.detour_ratio) if rec is not None else -1.0,
            "rec_terminal_alignment": self._round_opt(
                rec.terminal_guide_alignment) if rec is not None else -1.0,
            "rec_rejoin_distance": self._round_opt(
                rec.rejoin_distance) if rec is not None else -1.0,
            "priv_local_recoverable": bool(priv.privileged_local_recoverable)
            if priv is not None else False,
            "priv_reason": str(priv.reason) if priv is not None else "",
        }
        cands = []
        if self._last_candidates:
            top = sorted(self._last_candidates,
                         key=lambda c: float(c.privileged_score))[:5]
            cands = [
                {"type": int(c.type), "side": _SIDE_NAMES.get(int(c.side), "?"),
                 "source": str(c.source),
                 "score": round(float(c.privileged_score), 4),
                 "full": bool(c.full_goal_reached),
                 "conn": bool(c.connected_to_goal),
                 # World position for debug_viewer.py candidate overlay.
                 "pos": [float(c.position_world[0]),
                         float(c.position_world[1]),
                         float(c.position_world[2])]}
                for c in top
            ]
        trace["candidates"] = cands
        self._last_trace = trace

    @property
    def last_trace(self):
        return self._last_trace
