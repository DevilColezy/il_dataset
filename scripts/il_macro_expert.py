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
        self._observe_last_known = None
        self._observe_last_edge_mask = None
        self._observe_last_reachable_count = None
        self._observe_last_info_time = None
        self._prev_guide_world = None
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
        # Side memory (section VIII/X): bound to the ACTIVE blocker
        # component with a light rebind hysteresis.
        self._active_blocker_id = None
        self._blocker_pending_id = None
        self._blocker_pending_ticks = 0
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
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.0
        self._failed_reason = ""

    def _build_action(self, mode, guide_world, desired_yaw, has_yaw,
                      confidence, reason):
        action = self._module.MacroAction()
        action.mode = mode
        action.committed_side = self.committed_side
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
               local_unrecoverable=False, cached=False, brake=False):
        goal = np.asarray(goal_world, dtype=np.float64)
        pos = np.asarray(state["position"], dtype=np.float64)
        yaw = float(state["yaw"])
        speed = float(np.linalg.norm(np.asarray(state["velocity"])))
        self._current_position = pos
        self._current_yaw = yaw
        now_s = time.monotonic()

        goal_dist = float(np.linalg.norm(goal - pos))
        goal_dir_world = np.zeros(3, dtype=np.float64)
        if goal_dist > 1e-6:
            goal_dir_world = (goal - pos) / goal_dist

        # ── GOAL_REACHED ─────────────────────────────────────────────
        if goal_dist <= self.cfg["goal_tolerance_m"] and \
                speed <= self.cfg["goal_speed_tolerance_mps"]:
            self.mode = self._module.MacroMode.GOAL_REACHED
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
            self._update_direct_causal(now_s, goal_dist, local_unrecoverable,
                                       cached, brake)
        elif self.mode == self._module.MacroMode.SIDE_GUIDE:
            self._update_side_progress(now_s, state)

        # ── Main decision table (section IV / XXII) ──────────────────
        observed_ok = rec.status == \
            self._module.RecoverabilityStatus.DIRECT_REJOIN_SUCCESS
        if observed_ok and not self.causal_intervention_evidence:
            return self._handle_direct(goal_dir_world, direct_guide, rec,
                                       goal_dist, now_s)
        return self._proceed_to_macro_intervention(
            goal, pos, yaw, direct_guide, observed_map, vs, dt_s, now_s)

    # ── DIRECT session (sections III/X/XII/XXV) ──────────────────────
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

    def _update_direct_causal(self, now_s, goal_dist, local_unrecoverable,
                              cached, brake):
        """Rolling DIRECT-session update: best goal distance + last
        improvement time, and consecutive local-failure / cached / brake
        counts.  None of these are reset on a plain DIRECT tick, so causal
        evidence accumulates across 5 Hz cycles (section V)."""
        if self._direct_best_goal_distance is None or \
                goal_dist < self._direct_best_goal_distance - \
                self.cfg.get("goal_progress_epsilon_m", 0.05):
            self._direct_best_goal_distance = goal_dist
            self._direct_last_progress_time = now_s
        if self._direct_last_progress_time is not None:
            self.direct_no_progress_time = \
                now_s - self._direct_last_progress_time
        # Consecutive 30 Hz local-layer feedback (manager flags): a single
        # occasional failure (Case A) resets the count; a run of failures
        # (Case B) accumulates.
        if local_unrecoverable:
            self._direct_local_failure_ticks += 1
        else:
            self._direct_local_failure_ticks = 0
        if cached:
            self._direct_cached_ticks += 1
        else:
            self._direct_cached_ticks = 0
        if brake:
            self._direct_brake_ticks += 1
        else:
            self._direct_brake_ticks = 0
        evidence = (
            self._direct_local_failure_ticks >=
            self.cfg.get("causal_evidence_frames", 2)
            or self._direct_cached_ticks >=
            self.cfg.get("causal_evidence_frames", 2)
            or self._direct_brake_ticks >=
            self.cfg.get("causal_evidence_frames", 2))
        # DIRECT long no-progress (goal distance not improving) is CAUSAL
        # EVIDENCE for macro intervention — never an immediate FAILED
        # (section VII / Case C).
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
                self._observe_time_s = 0.0
                self._observe_yaw_delta = 0.0
                self._observe_reference_yaw_world = None
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
        # Side memory is bound to the ACTIVE blocker component (section X):
        # a confirmed new blocker gets a fresh side state.
        self._update_blocker_binding(blocker)
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
        self._update_side_failures(viable, blocker, now_s)

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
        # OBSERVE side: committed (if not failed) or first non-failed or
        # LEFT.  If both sides are failed -> NO_VALID_SIDE (side == None).
        side = self._observe_side()
        self.committed_side = side if side is not None \
            else self._module.Side.NONE

        # Rolling OBSERVE information-gain window (section XVI): compare
        # against the LAST progress point (not the entry reference).
        known = int(observed_map.known_count())
        edge_mask = (int(blocker.left_edge_visible),
                     int(blocker.right_edge_visible))
        reachable = len([c for c in candidates
                         if c.type == self._module.CandidateType.SIDE and
                         c.full_goal_reached])
        if self.mode != self._module.MacroMode.OBSERVE or \
                self._observe_reference_yaw_world is None:
            # Entering OBSERVE: initialise the observation reference and
            # close the DIRECT / SIDE sessions (their history stays frozen
            # until a fresh re-entry).
            self._direct_session_active = False
            self._side_session_active = False
            self._observe_reference_yaw_world = yaw
            self._observe_yaw_delta = 0.0
            self._observe_last_known = known
            self._observe_last_edge_mask = edge_mask
            self._observe_last_reachable_count = reachable
            self._observe_last_info_time = now_s
        else:
            if known > self._observe_last_known or \
                    edge_mask != self._observe_last_edge_mask or \
                    reachable > self._observe_last_reachable_count:
                self._observe_last_known = known
                self._observe_last_edge_mask = edge_mask
                self._observe_last_reachable_count = reachable
                self._observe_last_info_time = now_s
        if self._observe_last_info_time is not None:
            self.observe_no_information_time = \
                now_s - self._observe_last_info_time
        else:
            self.observe_no_information_time = 0.0

        self.mode = self._module.MacroMode.OBSERVE

        # OBSERVE no-information timeout (section XVI): change strategy
        # (OBSERVE_MOVE) or try the other side — never an immediate FAILED.
        no_new_info = self._observe_last_info_time is not None and \
            self.observe_no_information_time > \
            self.cfg.get("observe_no_information_timeout", 4.0)

        fov_half_obs = math.radians(self.cfg.get("observe_fov_half_deg", 45.0))
        rotation_exhausted = \
            abs(self._observe_yaw_delta) >= fov_half_obs - 1e-6

        # FULL-reachable observation viewpoint candidates (OBSERVE_MOVE).
        observe_move_cands = [
            c for c in candidates
            if c.type in (self._module.CandidateType.OBSERVE,
                          self._module.CandidateType.GOAL_FRONTIER) and
            c.full_goal_reached and c.known_reachable
        ]

        # OBSERVED_NO_CORRIDOR (section XIV): only when the committed side's
        # edge is ACTUALLY visible AND it was observed for a while with no
        # new info AND no usable observation move remains.  A single 45°
        # yaw sweep never marks a side failed (section XIII).
        if no_new_info and side is not None:
            edge_visible = (
                (side == self._module.Side.LEFT and
                 bool(blocker.left_edge_visible)) or
                (side == self._module.Side.RIGHT and
                 bool(blocker.right_edge_visible)))
            if edge_visible and not observe_move_cands:
                self._mark_side_failed(side, SideFailure.OBSERVED_NO_CORRIDOR)

        # ── NO_VALID_SIDE (section XVII) ─────────────────────────────
        if side is None:
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
            side = None
            self.committed_side = self._module.Side.NONE
            self._observe_reference_yaw_world = yaw
            self._observe_yaw_delta = 0.0

        # ── OBSERVE_MOVE vs OBSERVE_ROTATE (section XV) ──────────────
        if no_new_info and observe_move_cands and \
                (rotation_exhausted or side is None):
            move_candidate = max(
                observe_move_cands,
                key=lambda c: c.unknown_information_gain)
            guide = np.asarray(move_candidate.position_world)
            desired_yaw_world = normalize_angle(
                math.atan2(guide[1] - pos[1], guide[0] - pos[0]) - 0.5 * math.pi)
            observe_subtype = 1
        else:
            # OBSERVE_ROTATE: pure rotation about the current position.
            # The actual execution is ROTATE_ONLY (zero velocity), so the
            # macro label must carry zero translation: guide == position
            # (section XXV, Case E).  The sweep advances from the FIXED
            # reference yaw captured at OBSERVE entry.
            if side is not None:
                sign = 1.0 if side == self._module.Side.LEFT else -1.0
                sweep_step = self.cfg.get("observe_rotation_rate_rps", 1.5) * \
                    dt_s
                self._observe_yaw_delta += sign * sweep_step
                self._observe_yaw_delta = \
                    sign * min(abs(self._observe_yaw_delta), fov_half_obs)
            desired_yaw_world = normalize_angle(
                self._observe_reference_yaw_world + self._observe_yaw_delta)
            guide = pos
            observe_subtype = 0

        side_name = "left" if side == self._module.Side.LEFT else \
            ("right" if side == self._module.Side.RIGHT else "forward")
        action = self._build_action(
            self.mode, guide, desired_yaw_world, True, 0.5,
            "observe_%s" % side_name)
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
        self._direct_session_active = False
        self._observe_time_s = 0.0
        self._observe_yaw_delta = 0.0
        self._observe_reference_yaw_world = None
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

    def _update_blocker_binding(self, blocker):
        """Bind side memory to the ACTIVE blocker component with a light
        hysteresis (section X): a short detection jitter never wipes side
        memory; only a CONFIRMED new blocker resets failed_left/right."""
        bid = blocker.component_id
        if bid < 0:
            # No blocking component: keep the current binding.
            self._blocker_pending_id = None
            self._blocker_pending_ticks = 0
            return
        if self._active_blocker_id is None:
            self._active_blocker_id = bid
            self._blocker_pending_id = None
            self._blocker_pending_ticks = 0
            return
        if bid == self._active_blocker_id:
            self._blocker_pending_id = None
            self._blocker_pending_ticks = 0
            return
        if self._blocker_pending_id == bid:
            self._blocker_pending_ticks += 1
        else:
            self._blocker_pending_id = bid
            self._blocker_pending_ticks = 1
        if self._blocker_pending_ticks >= self.cfg.get(
                "blocker_rebind_ticks", 2):
            # Confirmed new independent blocker: fresh side state.
            self._active_blocker_id = bid
            self._failed_left = None
            self._failed_right = None
            self._side_local_fail_count = {
                self._module.Side.LEFT: 0,
                self._module.Side.RIGHT: 0,
            }
            self.committed_side = self._module.Side.NONE
            self._blocker_pending_id = None
            self._blocker_pending_ticks = 0

    # ── Side failure memory (section VIII/XIV) ───────────────────────
    def _mark_side_failed(self, side, reason):
        if side == self._module.Side.LEFT:
            self._failed_left = reason
        elif side == self._module.Side.RIGHT:
            self._failed_right = reason

    def _update_side_failures(self, viable, blocker, now_s):
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
                same_blocker = (
                    self._active_blocker_id is not None and
                    blocker.component_id == self._active_blocker_id)
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
