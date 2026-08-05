#!/usr/bin/env python3
"""Causal 5 Hz macro expert for horizontal active-perception navigation.

The expert uses only goal-relative state, the current depth frame, a finite
history observed occupancy map, and local-planner feedback.  It never queries
the global path, global ESDF, or ground-truth obstacle geometry.
"""

from __future__ import division, print_function

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MacroState(Enum):
    GOAL_SEEK = "GOAL_SEEK"
    BYPASS_LEFT = "BYPASS_LEFT"
    BYPASS_RIGHT = "BYPASS_RIGHT"
    ACTIVE_SCAN_LEFT = "ACTIVE_SCAN_LEFT"
    ACTIVE_SCAN_RIGHT = "ACTIVE_SCAN_RIGHT"
    ACTIVE_PEEK_LEFT = "ACTIVE_PEEK_LEFT"
    ACTIVE_PEEK_RIGHT = "ACTIVE_PEEK_RIGHT"
    GOAL_HOLD = "GOAL_HOLD"


class CommittedSide(Enum):
    NONE = 0
    # FLU is [forward, left, up], so positive y means left.
    LEFT = 1
    RIGHT = -1


@dataclass
class MacroGuide:
    """Continuous macro intent expressed in the current FLU frame."""

    valid: bool = False
    move_direction_flu: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    move_distance_m: float = 0.0
    move_distance_norm: float = 0.0
    yaw_direction_flu_xy: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0]))
    move_target_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    look_target_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3))
    macro_state: str = ""
    committed_side: int = 0
    decision_reason: str = ""
    macro_map_revision: int = 0
    macro_known_free_count: int = 0
    macro_occupied_count: int = 0
    macro_unknown_count: int = 0
    # Deprecated: kept for backward-compat CSV readers.  Prefer the two
    # separate counters below.
    frontier_candidate_count: int = 0
    # Count of raw frontier clusters extracted from the rolling map before
    # any reachability check.
    extracted_frontier_candidate_count: int = 0
    # Count of candidates that passed the bounded-A* reachability screen.
    reachable_frontier_candidate_count: int = 0
    # Aggregate rejection reasons (compact key → count for CSV width).
    frontier_rejection_summary: str = ""


@dataclass
class MacroExpertConfig:
    update_hz: float = 5.0
    local_hz: float = 30.0
    effective_guide_range_m: float = 4.45
    map_resolution_m: float = 0.10
    map_history_seconds: float = 3.0
    guide_range_fraction: float = 0.85
    blocked_corridor_radius_m: float = 0.55
    guide_swept_radius_m: float = 0.30
    guide_obstacle_clearance_m: float = 0.20
    guide_corridor_min_ratio: float = 1.0
    minimum_corridor_score: float = 0.45
    symmetric_score_tolerance: float = 0.03
    preferred_symmetric_side: str = "left"
    bypass_goal_weight: float = 0.45
    minimum_guide_distance_m: float = 0.50
    frontier_candidate_limit: int = 8
    frontier_angular_bin_deg: float = 10.0
    frontier_standoff_m: float = 0.45
    frontier_max_backtrack_m: float = 0.20
    frontier_goal_progress_weight: float = 1.0
    frontier_information_weight: float = 0.25
    frontier_yaw_cost_weight: float = 0.10
    frontier_side_commitment_bonus: float = 0.15
    enter_blocked_frames: int = 3
    exit_clear_frames: int = 8
    minimum_progress_rate_m_s: float = 0.15
    minimum_commit_time_s: float = 0.8
    active_scan_yaw_rate_rps: float = 2.0
    active_scan_min_angle_deg: float = 15.0
    active_scan_max_duration_s: float = 2.0
    active_peek_distance_m: float = 2.0
    active_peek_safety_radius_m: float = 0.55
    vertical_avoidance_enabled: bool = False
    vertical_active_perception_enabled: bool = False

    def validate(self, depth_max_m):
        if self.update_hz <= 0.0 or self.local_hz <= 0.0:
            raise ValueError("macro/local update rates must be positive")
        ratio = self.local_hz / self.update_hz
        if abs(ratio - round(ratio)) > 1.0e-6:
            raise ValueError("local_hz must be an integer multiple of update_hz")
        if not 0.0 < self.guide_range_fraction <= 1.0:
            raise ValueError("guide_range_fraction must be in (0, 1]")
        if not 0.0 < self.effective_guide_range_m <= depth_max_m:
            raise ValueError("effective guide range must be inside depth range")
        if not 0.0 < self.active_peek_distance_m < self.effective_guide_range_m:
            raise ValueError("active peek distance must be inside guide range")
        if self.map_resolution_m <= 0.0:
            raise ValueError("map_resolution_m must be positive")
        if self.map_history_seconds <= 0.0:
            raise ValueError("map_history_seconds must be positive")
        if self.enter_blocked_frames <= 0 or self.exit_clear_frames <= 0:
            raise ValueError("macro hysteresis frame counts must be positive")
        if not 0.0 < self.minimum_corridor_score <= 1.0:
            raise ValueError("minimum_corridor_score must be in (0, 1]")
        if self.guide_swept_radius_m <= 0.0:
            raise ValueError("guide_swept_radius_m must be positive")
        if self.guide_obstacle_clearance_m < 0.0:
            raise ValueError("guide_obstacle_clearance_m must be non-negative")
        if not 0.0 < self.guide_corridor_min_ratio <= 1.0:
            raise ValueError("guide_corridor_min_ratio must be in (0, 1]")
        if not 0.0 <= self.symmetric_score_tolerance <= 1.0:
            raise ValueError("symmetric_score_tolerance must be in [0, 1]")
        if str(self.preferred_symmetric_side).lower() not in ("left", "right"):
            raise ValueError("preferred_symmetric_side must be left or right")
        if not 0.0 < self.bypass_goal_weight < 1.0:
            raise ValueError("bypass_goal_weight must be in (0, 1)")
        if not 0.0 < self.minimum_guide_distance_m < self.effective_guide_range_m:
            raise ValueError(
                "minimum_guide_distance_m must be inside guide range")
        if self.frontier_candidate_limit <= 0:
            raise ValueError("frontier_candidate_limit must be positive")
        if not 1.0 <= self.frontier_angular_bin_deg <= 45.0:
            raise ValueError("frontier_angular_bin_deg must be in [1, 45]")
        if not 0.0 < self.frontier_standoff_m < self.effective_guide_range_m:
            raise ValueError("frontier_standoff_m must be inside guide range")
        if self.frontier_max_backtrack_m < 0.0:
            raise ValueError("frontier_max_backtrack_m must be non-negative")
        if (self.frontier_goal_progress_weight < 0.0 or
                self.frontier_information_weight < 0.0 or
                self.frontier_yaw_cost_weight < 0.0 or
                self.frontier_side_commitment_bonus < 0.0):
            raise ValueError("frontier score weights must be non-negative")
        if (self.active_scan_yaw_rate_rps <= 0.0 or
                self.active_scan_min_angle_deg <= 0.0 or
                self.active_scan_max_duration_s <= 0.0):
            raise ValueError("active scan parameters must be positive")
        # Full scan: one continuous 360° rotation at the configured yaw
        # rate captures the complete surroundings.  At 30 Hz the 90° FOV
        # camera observes the full circle in ~7.9 s.
        full_scan_deg = 360.0
        full_scan_time = (
            math.radians(full_scan_deg) / self.active_scan_yaw_rate_rps
            + 0.5)  # yaw accel/decel margin (single direction, no switching)
        if self.active_scan_max_duration_s + 1.0e-9 < full_scan_time:
            raise ValueError(
                "active_scan_max_duration_s ({:.2f} s) is too short for "
                "a {:.0f}° full scan at {:.2f} rad/s "
                "(needs ≥ {:.2f} s)".format(
                    self.active_scan_max_duration_s,
                    full_scan_deg,
                    self.active_scan_yaw_rate_rps,
                    full_scan_time))
        if self.map_history_seconds + 1.0e-9 < self.active_scan_max_duration_s:
            raise ValueError(
                "map_history_seconds ({:.2f} s) must be ≥ "
                "active_scan_max_duration_s ({:.2f} s)".format(
                    self.map_history_seconds,
                    self.active_scan_max_duration_s))
        if self.vertical_avoidance_enabled:
            raise ValueError("vertical obstacle avoidance must remain disabled")
        if self.vertical_active_perception_enabled:
            raise ValueError("vertical active perception must remain disabled")


class MacroExpert:
    """Finite-state horizontal macro expert with side commitment."""

    def __init__(self, config):
        self.cfg = config
        self._reset()
        self._guide_reachability_checker = None

    def set_guide_reachability_checker(self, checker):
        """Install the expert-only C++ observed-ESDF reachability query."""
        self._guide_reachability_checker = checker

    def reset(self):
        self._reset()

    def force_active_scan(
        self,
        goal_direction_flu,
        goal_distance_m,
        observed_map,
        current_position_world,
        current_quaternion_xyzw,
    ):
        """Atomically replace a rejected MOVE Guide with safe perception.

        When a scan session is already in progress, the session clock is
        *not* reset — the drone has already observed one side and resetting
        the timer would allow plan→fail→scan cycles to accumulate unbounded
        time in the same direction without the macro-level timeout ever
        firing.

        When the planner repeatedly fails near the goal (OPTIMIZATION_FAILED
        or COLLISION), the committed side can lock the scan to one direction
        forever.  Track consecutive restarts and alternate sides so both
        flanks are tried.
        """
        committed = self._committed_side
        if committed != CommittedSide.NONE:
            self._consecutive_scan_restarts += 1
            if self._consecutive_scan_restarts >= 3:
                # Stuck on the committed side — try the opposite flank.
                scan_side = (CommittedSide.RIGHT if committed == CommittedSide.LEFT
                             else CommittedSide.LEFT)
                self._consecutive_scan_restarts = 0
            else:
                scan_side = committed
        else:
            scan_side = self._preferred_side()
            self._consecutive_scan_restarts = 0
        scan_state = self._scan_state(scan_side)
        self._state = scan_state
        # Preserve the scan session when one is already active.  Only start
        # a fresh session when entering ACTIVE_SCAN from a non-scan state.
        if self._scan_session_initial_yaw is None:
            self._begin_scan_session()
        else:
            # Sync the per-phase accumulator to the (possibly new) direction.
            self._scan_angle_accum = 0.0
            self._scan_signed_angle = 0.0
            self._scan_last_yaw = self._current_observed_yaw
        return self._build_guide(
            scan_state,
            self._unit3(goal_direction_flu),
            float(goal_distance_m),
            observed_map,
            np.asarray(current_position_world, dtype=np.float64),
            np.asarray(current_quaternion_xyzw, dtype=np.float64),
            source_state=scan_state)

    def update(
        self,
        goal_direction_flu,
        goal_distance_m,
        depth_m,
        observed_map,
        current_position_world,
        current_yaw,
        current_quaternion_xyzw,
        current_velocity_world,
        local_blocked=False,
        local_progress_rate=1.0,
        local_feasible=True,
        dt_since_last_macro=0.2,
    ):
        self._current_velocity_world = np.asarray(
            current_velocity_world, dtype=np.float64)
        dt = max(0.0, float(dt_since_last_macro))
        yaw = float(current_yaw)
        self._current_observed_yaw = yaw
        self._tick_time += dt
        goal_dir = self._unit3(goal_direction_flu)
        position_world = np.asarray(current_position_world, dtype=np.float64)
        quaternion = np.asarray(current_quaternion_xyzw, dtype=np.float64)
        self._last_frontier_candidate_count = 0
        scan_exhausted = self._scan_exhausted_this_tick
        self._scan_exhausted_this_tick = False

        if float(goal_distance_m) < 0.30:
            if self._state in (MacroState.ACTIVE_SCAN_LEFT,
                               MacroState.ACTIVE_SCAN_RIGHT):
                self._end_scan_session()
            self._state = MacroState.GOAL_HOLD
            return self._make_goal_hold_guide(position_world, quaternion)

        blocked_now = self._is_blocked(
            goal_dir, depth_m, observed_map, position_world, quaternion,
            local_blocked, local_progress_rate, local_feasible)
        if blocked_now:
            self._blocked_counter += 1
            self._clear_counter = 0
        else:
            self._blocked_counter = max(0, self._blocked_counter - 1)
            self._clear_counter += 1

        previous_state = self._state
        state = previous_state
        query_args = (
            goal_dir, depth_m, observed_map, position_world, quaternion)

        if state == MacroState.GOAL_SEEK:
            if self._blocked_counter >= self.cfg.enter_blocked_frames:
                state = self._choose_bypass_side(*query_args)
                self._tick_time = 0.0
                self._clear_counter = 0

        elif state in (MacroState.BYPASS_LEFT, MacroState.BYPASS_RIGHT):
            commit_ok = self._tick_time >= self.cfg.minimum_commit_time_s
            if (commit_ok and self._blocked_counter == 0 and
                    self._clear_counter >= self.cfg.exit_clear_frames):
                state = MacroState.GOAL_SEEK
                self._committed_side = CommittedSide.NONE
                self._consecutive_scan_restarts = 0
                self._tick_time = 0.0
            elif commit_ok and not self._bypass_side_feasible(
                    state, *query_args):
                # Preserve the committed homotopy while acquiring a better
                # view.  Immediately scanning the opposite side creates the
                # left/right switching that later appears as expert wobble.
                state = (MacroState.ACTIVE_SCAN_LEFT
                         if state == MacroState.BYPASS_LEFT
                         else MacroState.ACTIVE_SCAN_RIGHT)
                self._begin_scan_session()

        elif state in (MacroState.ACTIVE_SCAN_LEFT,
                       MacroState.ACTIVE_SCAN_RIGHT):
            # ── Scan-session state machine ────────────────────────────
            # At 30 Hz depth the camera captures the full 90° FOV in one
            # frame.  One continuous 360° rotation at 0.8 rad/s (~7.9 s)
            # observes the complete surroundings and fits within the 8 s
            # scan budget.  There is no reason to split this into two
            # alternating ±90° phases; a single sweep is both faster and
            # keeps the rolling map coherent.
            #
            # The session tracks cumulative unwrapped yaw from the initial
            # heading.  Once the camera has swept 360° the session is
            # exhausted; until then frontier candidates are extracted
            # continuously (after a short min_angle settle period).
            # ────────────────────────────────────────────────────────────
            if self._scan_session_initial_yaw is None:
                self._begin_scan_session()
            if self._scan_last_yaw is None:
                self._scan_last_yaw = yaw
            yaw_delta = self._wrap_angle(yaw - self._scan_last_yaw)
            self._scan_last_yaw = yaw
            self._scan_signed_angle += yaw_delta
            # Cumulative rotation (unwrapped, ≥ 0 when turning in the
            # scan direction).
            scan_sign = (1.0 if state == MacroState.ACTIVE_SCAN_LEFT
                         else -1.0)
            self._scan_angle_accum = max(
                0.0, scan_sign * self._scan_signed_angle)
            self._scan_time += dt
            scanned_enough = self._scan_angle_accum >= math.radians(
                self.cfg.active_scan_min_angle_deg)
            full_circle_rad = 2.0 * math.pi
            full_scan_timeout = (
                self._scan_time >= self.cfg.active_scan_max_duration_s)

            # Determine which side to search — the side the camera is
            # currently rotating toward.
            scan_side = (CommittedSide.LEFT
                         if state == MacroState.ACTIVE_SCAN_LEFT
                         else CommittedSide.RIGHT)

            candidate = (
                self._find_horizontal_corridor(
                    *query_args, required_side=scan_side)
                if scanned_enough else None)
            if candidate is not None:
                # Found a reachable frontier — commit the bypass.
                state = candidate
                self._set_committed_side(state)
                self._end_scan_session()
                self._tick_time = 0.0
                self._blocked_counter = 0
                self._clear_counter = 0
            elif (self._scan_angle_accum >= full_circle_rad or
                  full_scan_timeout):
                # Full 360° swept (or timeout).  The camera has seen
                # everything around the drone.  If no bypass was found,
                # try a peek on the committed side as a last resort.
                # If that also fails, fall back to GOAL_SEEK — staying in
                # ACTIVE_SCAN would just restart another 360° and loop
                # forever until the online_runtime timeout.
                side_val = (self._committed_side.value
                            if self._committed_side != CommittedSide.NONE
                            else scan_side.value)
                if self._peek_side_feasible(
                        side_val, goal_dir, depth_m, observed_map,
                        position_world, quaternion):
                    state = (MacroState.ACTIVE_PEEK_LEFT
                             if side_val > 0
                             else MacroState.ACTIVE_PEEK_RIGHT)
                else:
                    # Give up scanning — try GOAL_SEEK with whatever
                    # observations we have.  The bounded A* may still
                    # find a partial forward path.
                    state = MacroState.GOAL_SEEK
                    self._committed_side = CommittedSide.NONE
                    self._consecutive_scan_restarts = 0
                    self._scan_exhausted_this_tick = True
                    # Reset hysteresis: don't let _is_blocked immediately
                    # re-trigger scanning on the next tick.  Give GOAL_SEEK
                    # a few ticks to try moving before blocking again.
                    self._blocked_counter = 0
                    self._clear_counter = self.cfg.exit_clear_frames
                self._end_scan_session()
                self._tick_time = 0.0
            elif (scanned_enough and
                  self._extracted_frontier_count > 0 and
                  self._scan_peek_count < 2):
                # Frontiers exist but bounded A* rejected them — peek to
                # bridge the known-space gap (only if we haven't already
                # completed a full circle).
                if self._peek_side_feasible(
                        scan_side.value, goal_dir, depth_m,
                        observed_map, position_world, quaternion):
                    state = (MacroState.ACTIVE_PEEK_LEFT
                             if scan_side == CommittedSide.LEFT
                             else MacroState.ACTIVE_PEEK_RIGHT)
                    self._scan_peek_count += 1
                    self._tick_time = 0.0

        elif state in (MacroState.ACTIVE_PEEK_LEFT,
                       MacroState.ACTIVE_PEEK_RIGHT):
            peek_side = (CommittedSide.LEFT
                         if state == MacroState.ACTIVE_PEEK_LEFT
                         else CommittedSide.RIGHT)
            candidate = self._find_horizontal_corridor(
                *query_args, required_side=peek_side)
            if candidate is not None:
                # Peek succeeded — the new observations bridged the gap.
                state = candidate
                self._set_committed_side(state)
                self._end_scan_session()
                self._tick_time = 0.0
                self._clear_counter = 0
            elif self._clear_counter >= self.cfg.exit_clear_frames:
                state = MacroState.GOAL_SEEK
                self._committed_side = CommittedSide.NONE
                self._consecutive_scan_restarts = 0
                self._end_scan_session()
                self._tick_time = 0.0
            else:
                side = (CommittedSide.LEFT.value
                        if state == MacroState.ACTIVE_PEEK_LEFT
                        else CommittedSide.RIGHT.value)
                if not self._peek_side_feasible(
                        side, goal_dir, depth_m, observed_map,
                        position_world, quaternion):
                    # Peek corridor closed — resume scanning from where we
                    # left off.  Do NOT start a fresh session: the drone has
                    # already observed one side; restarting would discard
                    # that progress and risk oscillating left/right peeks.
                    state = (MacroState.ACTIVE_SCAN_LEFT
                             if side > 0 else MacroState.ACTIVE_SCAN_RIGHT)
                    if self._scan_session_initial_yaw is None:
                        self._begin_scan_session()
                    else:
                        # Resume: only reset the per-phase accumulator.
                        self._scan_angle_accum = 0.0
                        self._scan_signed_angle = 0.0
                        self._scan_last_yaw = yaw

        self._state = state
        return self._build_guide(
            state, goal_dir, float(goal_distance_m), observed_map,
            position_world, quaternion, source_state=previous_state)

    def _reset(self):
        self._state = MacroState.GOAL_SEEK
        self._committed_side = CommittedSide.NONE
        self._blocked_counter = 0
        self._clear_counter = 0
        self._tick_time = 0.0
        self._scan_angle_accum = 0.0
        self._scan_signed_angle = 0.0
        self._scan_last_yaw = None
        self._current_observed_yaw = None
        self._scan_time = 0.0
        # Scan-session state — survives side-switches within one scan episode.
        self._scan_session_initial_yaw = None
        self._scan_phase = 0  # 0 = left-first, 1 = crossing to right
        self._scan_left_angle_reached = False
        self._consecutive_scan_restarts = 0  # across-session counter
        self._scan_exhausted_this_tick = False  # prevent immediate re-scan
        self._frontier_rejection_reasons = {}  # candidate_index → reason
        self._extracted_frontier_count = 0
        self._reachable_frontier_count = 0
        self._selected_frontier_target_world = None
        self._selected_frontier_side = CommittedSide.NONE
        self._selected_frontier_score = -float("inf")
        self._selected_frontier_map_revision = -1
        self._last_frontier_candidate_count = 0
        self._held_move_target_world = np.zeros(3, dtype=np.float64)
        self._held_look_target_world = np.zeros(3, dtype=np.float64)
        self._held_move_direction_flu = np.array(
            [1.0, 0.0, 0.0], dtype=np.float64)
        self._held_move_distance_m = 0.0
        self._held_move_distance_norm = 0.0
        self._held_yaw_direction_flu_xy = np.array(
            [1.0, 0.0], dtype=np.float64)
        self._held_frontier_candidate_count = 0
        self._held_extracted_frontier_count = 0
        self._held_reachable_frontier_count = 0
        self._held_rejection_reasons = {}
        self._held_valid = False

    def _begin_scan_session(self):
        """Record the scan-session reference yaw (called once per episode).

        Always cleans up any stale session state first so that callers in
        _build_guide and _choose_bypass_side can safely start a fresh session
        without manually calling _end_scan_session.
        """
        self._end_scan_session()
        self._scan_session_initial_yaw = self._current_observed_yaw
        self._scan_angle_accum = 0.0
        self._scan_signed_angle = 0.0
        self._scan_relative_angle = 0.0
        self._scan_last_yaw = self._current_observed_yaw
        self._scan_time = 0.0
        self._scan_phase = 0
        self._scan_left_angle_reached = False
        self._scan_peek_count = 0  # peeks attempted in this session
        self._frontier_rejection_reasons = {}
        self._extracted_frontier_count = 0
        self._reachable_frontier_count = 0
        self._tick_time = 0.0
        self._selected_frontier_target_world = None
        self._selected_frontier_side = CommittedSide.NONE
        self._selected_frontier_score = -float("inf")
        self._selected_frontier_map_revision = -1
        self._last_frontier_candidate_count = 0

    def _end_scan_session(self):
        """Clean up scan-session state when leaving ACTIVE_SCAN."""
        self._scan_session_initial_yaw = None
        self._scan_phase = 0
        self._scan_left_angle_reached = False
        self._scan_angle_accum = 0.0
        self._scan_signed_angle = 0.0
        self._scan_last_yaw = None
        self._scan_time = 0.0
        self._frontier_rejection_reasons = {}
        self._extracted_frontier_count = 0
        self._reachable_frontier_count = 0

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _set_committed_side(self, state):
        self._committed_side = (
            CommittedSide.LEFT if state == MacroState.BYPASS_LEFT
            else CommittedSide.RIGHT)

    def _preferred_side(self):
        return (CommittedSide.LEFT
                if str(self.cfg.preferred_symmetric_side).lower() == "left"
                else CommittedSide.RIGHT)

    @staticmethod
    def _bypass_state(side):
        return (MacroState.BYPASS_LEFT
                if side == CommittedSide.LEFT else MacroState.BYPASS_RIGHT)

    @staticmethod
    def _scan_state(side):
        return (MacroState.ACTIVE_SCAN_LEFT
                if side == CommittedSide.LEFT else MacroState.ACTIVE_SCAN_RIGHT)

    def _goal_relative_lateral(self, goal_dir_flu, side):
        """Horizontal left/right relative to the goal ray, not body yaw."""
        goal = np.asarray(goal_dir_flu, dtype=np.float64).reshape(3)
        horizontal_norm = float(np.linalg.norm(goal[:2]))
        if horizontal_norm <= 1.0e-9:
            return np.array([0.0, float(side), 0.0], dtype=np.float64)
        gx, gy = goal[0] / horizontal_norm, goal[1] / horizontal_norm
        # In FLU, rotating [gx, gy] counter-clockwise gives goal-left.
        return np.array(
            [-float(side) * gy, float(side) * gx, 0.0],
            dtype=np.float64)

    @staticmethod
    def _unit3(vector, fallback=None):
        value = np.asarray(vector, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(value))
        if norm > 1.0e-9:
            return value / norm
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0])
        return np.asarray(fallback, dtype=np.float64).copy()

    @staticmethod
    def _unit2(vector):
        value = np.asarray(vector, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(value))
        if norm <= 1.0e-9:
            return np.array([1.0, 0.0])
        return value / norm

    @staticmethod
    def _flu_to_world(vector_flu, quaternion_xyzw):
        from il_common import body_flu_to_world_quat
        return body_flu_to_world_quat(
            np.asarray(vector_flu, dtype=np.float64), quaternion_xyzw)

    @staticmethod
    def _world_to_flu(vector_world, quaternion_xyzw):
        from il_common import world_vector_to_body_flu_quat
        return world_vector_to_body_flu_quat(
            np.asarray(vector_world, dtype=np.float64), quaternion_xyzw)

    def _corridor_ratio(
        self, observed_map, start_world, direction_flu, distance_m,
        radius_m, quaternion_xyzw,
    ):
        if observed_map is None or distance_m <= 1.0e-3:
            return None
        direction_world = self._flu_to_world(
            self._unit3(direction_flu), quaternion_xyzw)
        end_world = np.asarray(start_world) + direction_world * distance_m
        return float(observed_map.sample_known_free_ratio_along_corridor(
            start_world=np.asarray(start_world, dtype=np.float64),
            end_world=np.asarray(end_world, dtype=np.float64),
            radius_m=float(radius_m),
            spacing_m=max(0.05, self.cfg.map_resolution_m),
            min_clearance_m=0.0))

    @staticmethod
    def _depth_sector_score(depth_m, side):
        """Conservative current-frame fallback; invalid depth is blocked."""
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.ndim != 2 or depth.size == 0:
            return 0.0
        height, width = depth.shape
        y0, y1 = int(0.25 * height), max(
            int(0.75 * height), int(0.25 * height) + 1)
        if side > 0:  # image left is FLU left
            x0, x1 = 0, max(1, int(0.55 * width))
        elif side < 0:
            x0, x1 = min(width - 1, int(0.45 * width)), width
        else:
            x0, x1 = int(0.25 * width), max(
                int(0.75 * width), int(0.25 * width) + 1)
        sector = depth[y0:y1, x0:x1]
        valid = sector[np.isfinite(sector) & (sector > 0.0)]
        if valid.size < max(8, sector.size // 10):
            return 0.0
        return float(np.count_nonzero(valid >= 1.5)) / float(valid.size)

    def _is_blocked(
        self, goal_dir_flu, depth_m, observed_map,
        position_world, quaternion, local_blocked,
        local_progress_rate, local_feasible,
    ):
        if local_blocked or not local_feasible:
            return True
        if float(local_progress_rate) < self.cfg.minimum_progress_rate_m_s:
            return True
        ratio = self._corridor_ratio(
            observed_map, position_world, goal_dir_flu,
            self.cfg.effective_guide_range_m,
            self.cfg.blocked_corridor_radius_m, quaternion)
        if ratio is not None:
            return ratio < 0.60
        return self._depth_sector_score(depth_m, 0) < 0.55

    def _choose_bypass_side(
        self, goal_dir_flu, depth_m, observed_map,
        position_world, quaternion,
    ):
        if self._map_supports_frontiers(observed_map):
            selected = self._select_reachable_frontier(
                goal_dir_flu, observed_map, position_world, quaternion)
            if selected is not None:
                self._committed_side = selected
                return self._bypass_state(selected)

            # No safe frontier viewpoint is reachable yet.  Use the coarse
            # sector score only to choose where to look; never turn that score
            # directly into a translational Guide.
            left = self._score_horizontal_corridor(
                goal_dir_flu, depth_m, observed_map,
                CommittedSide.LEFT.value, position_world, quaternion)
            right = self._score_horizontal_corridor(
                goal_dir_flu, depth_m, observed_map,
                CommittedSide.RIGHT.value, position_world, quaternion)
            if abs(left - right) <= self.cfg.symmetric_score_tolerance:
                selected = self._preferred_side()
            else:
                selected = (CommittedSide.LEFT
                            if left > right else CommittedSide.RIGHT)
            self._committed_side = selected
            self._begin_scan_session()
            return self._scan_state(selected)

        # Static/offline compatibility when a real rolling voxel map is not
        # available. Formal collection always uses the frontier branch above.
        left = self._score_horizontal_corridor(
            goal_dir_flu, depth_m, observed_map, CommittedSide.LEFT.value,
            position_world, quaternion)
        right = self._score_horizontal_corridor(
            goal_dir_flu, depth_m, observed_map, CommittedSide.RIGHT.value,
            position_world, quaternion)
        preferred = self._preferred_side()
        if max(left, right) < self.cfg.minimum_corridor_score:
            # A wide occluder can make both sides unknown.  Do not translate
            # into either side merely because one weak score is numerically
            # larger: first turn the camera toward the deterministic side.
            self._committed_side = preferred
            self._begin_scan_session()
            return self._scan_state(preferred)
        if abs(left - right) <= self.cfg.symmetric_score_tolerance:
            selected = preferred
        else:
            selected = (CommittedSide.LEFT
                        if left > right else CommittedSide.RIGHT)
        self._committed_side = selected
        return self._bypass_state(selected)

    def _score_horizontal_corridor(
        self, goal_dir_flu, depth_m, observed_map, side,
        position_world, quaternion,
    ):
        lateral = self._goal_relative_lateral(goal_dir_flu, side)
        goal_weight = self.cfg.bypass_goal_weight
        direction = self._unit3(
            goal_weight * goal_dir_flu + (1.0 - goal_weight) * lateral)
        map_scores = []
        for fraction in (0.35, 0.65, 1.0):
            ratio = self._corridor_ratio(
                observed_map, position_world, direction,
                self.cfg.effective_guide_range_m * fraction,
                self.cfg.blocked_corridor_radius_m, quaternion)
            if ratio is not None:
                map_scores.append(ratio)
        depth_score = self._depth_sector_score(depth_m, side)
        if not map_scores:
            return depth_score
        return 0.8 * float(np.mean(map_scores)) + 0.2 * depth_score

    def _bypass_side_feasible(
        self, state, goal_dir_flu, depth_m, observed_map,
        position_world, quaternion,
    ):
        side_enum = (CommittedSide.LEFT
                     if state == MacroState.BYPASS_LEFT
                     else CommittedSide.RIGHT)
        if self._map_supports_frontiers(observed_map):
            return self._select_reachable_frontier(
                goal_dir_flu, observed_map, position_world, quaternion,
                required_side=side_enum) is not None
        side = side_enum.value
        return self._score_horizontal_corridor(
            goal_dir_flu, depth_m, observed_map, side,
            position_world, quaternion) >= self.cfg.minimum_corridor_score

    def _find_horizontal_corridor(
        self, goal_dir_flu, depth_m, observed_map,
        position_world, quaternion, required_side=None,
    ):
        if self._map_supports_frontiers(observed_map):
            selected = self._select_reachable_frontier(
                goal_dir_flu, observed_map, position_world, quaternion,
                required_side=required_side)
            return (None if selected is None
                    else self._bypass_state(selected))

        # Compatibility fallback for unit tests and static callers without a
        # voxel map. It is not used by formal dataset collection.
        left = self._score_horizontal_corridor(
            goal_dir_flu, depth_m, observed_map, CommittedSide.LEFT.value,
            position_world, quaternion)
        right = self._score_horizontal_corridor(
            goal_dir_flu, depth_m, observed_map, CommittedSide.RIGHT.value,
            position_world, quaternion)
        if max(left, right) < self.cfg.minimum_corridor_score:
            return None
        if required_side is not None:
            required_score = (left if required_side == CommittedSide.LEFT
                              else right)
            return (self._bypass_state(required_side)
                    if required_score >= self.cfg.minimum_corridor_score
                    else None)
        if (self._committed_side != CommittedSide.NONE and
                ((self._committed_side == CommittedSide.LEFT and
                  left >= self.cfg.minimum_corridor_score and
                  left + self.cfg.symmetric_score_tolerance >= right) or
                 (self._committed_side == CommittedSide.RIGHT and
                  right >= self.cfg.minimum_corridor_score and
                  right + self.cfg.symmetric_score_tolerance >= left))):
            selected = self._committed_side
        elif abs(left - right) <= self.cfg.symmetric_score_tolerance:
            selected = self._preferred_side()
        else:
            selected = (CommittedSide.LEFT
                        if left > right else CommittedSide.RIGHT)
        return self._bypass_state(selected)

    @staticmethod
    def _map_supports_frontiers(observed_map):
        return bool(
            observed_map is not None and
            hasattr(observed_map, "get_occupancy") and
            hasattr(observed_map, "get_origin") and
            hasattr(observed_map, "get_resolution"))

    def _extract_frontier_candidates(
        self, goal_dir_flu, observed_map, position_world, quaternion,
        required_side=None,
    ):
        """Extract goal-directed observation poses from the causal grid.

        A frontier cell is known-free and horizontally adjacent to UNKNOWN.
        The flight target is pulled back from that boundary into known space;
        the C++ bounded A* query below performs the authoritative reachability
        and clearance check.  This is intentionally a local navigation
        frontier, not an exploration tour and not a fixed left/right ray.
        """
        self._last_frontier_candidate_count = 0
        if not self._map_supports_frontiers(observed_map):
            return []
        try:
            occupancy = observed_map.get_occupancy(copy=False)
        except TypeError:
            occupancy = observed_map.get_occupancy()
        occupancy = np.asarray(occupancy, dtype=np.uint8)
        if occupancy.ndim != 3 or min(occupancy.shape) < 3:
            return []

        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        origin = np.asarray(observed_map.get_origin(), dtype=np.float64)
        resolution = float(observed_map.get_resolution())
        if resolution <= 0.0:
            return []
        grid = np.floor((position - origin) / resolution).astype(np.int32)
        iz = int(np.clip(grid[2], 0, occupancy.shape[2] - 1))
        # UNKNOWN=0, FREE=1, OCCUPIED=2. Macro planning is deliberately 2-D
        # at the current flight height. Vehicle inflation, 3-D clearance and
        # the final trajectory are checked by the observed-ESDF C++ planner.
        free_column = occupancy[:, :, iz] == 1
        unknown_slice = occupancy[:, :, iz] == 0
        padded_unknown = np.pad(
            unknown_slice, ((1, 1), (1, 1)),
            mode="constant", constant_values=True)
        unknown_neighbours = np.zeros(free_column.shape, dtype=np.uint8)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                unknown_neighbours += padded_unknown[
                    1 + dx:1 + dx + free_column.shape[0],
                    1 + dy:1 + dy + free_column.shape[1]]
        frontier_indices = np.argwhere(
            free_column & (unknown_neighbours > 0))
        if frontier_indices.size == 0:
            return []

        frontier_xy = origin[:2] + (
            frontier_indices.astype(np.float64) + 0.5) * resolution
        vectors_xy = frontier_xy - position[:2]
        frontier_ranges = np.linalg.norm(vectors_xy, axis=1)
        maximum_frontier_range = (
            self.cfg.effective_guide_range_m +
            self.cfg.frontier_standoff_m + 2.0 * resolution)
        valid = (
            np.isfinite(frontier_ranges) &
            (frontier_ranges >=
             self.cfg.minimum_guide_distance_m +
             self.cfg.frontier_standoff_m) &
            (frontier_ranges <= maximum_frontier_range))
        if not np.any(valid):
            return []
        frontier_indices = frontier_indices[valid]
        frontier_xy = frontier_xy[valid]
        vectors_xy = vectors_xy[valid]
        frontier_ranges = frontier_ranges[valid]

        goal_world = self._flu_to_world(goal_dir_flu, quaternion)
        goal_horizontal = np.asarray(goal_world[:2], dtype=np.float64)
        goal_horizontal_norm = float(np.linalg.norm(goal_horizontal))
        if goal_horizontal_norm <= 1.0e-9:
            body_forward = self._flu_to_world(
                np.array([1.0, 0.0, 0.0]), quaternion)
            goal_horizontal = body_forward[:2]
            goal_horizontal_norm = max(
                float(np.linalg.norm(goal_horizontal)), 1.0e-9)
        goal_horizontal /= goal_horizontal_norm

        unit_xy = vectors_xy / np.maximum(
            frontier_ranges[:, None], 1.0e-9)
        alignment = unit_xy.dot(goal_horizontal)
        cross = (goal_horizontal[0] * unit_xy[:, 1] -
                 goal_horizontal[1] * unit_xy[:, 0])
        azimuth = np.arctan2(cross, alignment)
        target_ranges = np.minimum(
            frontier_ranges - self.cfg.frontier_standoff_m,
            self.cfg.effective_guide_range_m *
            float(np.clip(self.cfg.guide_range_fraction, 0.1, 1.0)))
        goal_progress = target_ranges * alignment
        valid = (
            (target_ranges >= self.cfg.minimum_guide_distance_m) &
            (goal_progress >= -self.cfg.frontier_max_backtrack_m))
        if required_side is not None:
            # Near-zero cross products are assigned deterministically so a
            # symmetric scene cannot cause left/right label flicker.
            side_metric = cross * float(required_side.value)
            valid &= side_metric >= -math.sin(math.radians(2.0))
        if not np.any(valid):
            return []

        frontier_indices = frontier_indices[valid]
        frontier_xy = frontier_xy[valid]
        unit_xy = unit_xy[valid]
        target_ranges = target_ranges[valid]
        goal_progress = goal_progress[valid]
        azimuth = azimuth[valid]
        cross = cross[valid]
        info_gain = unknown_neighbours[
            frontier_indices[:, 0], frontier_indices[:, 1]
        ].astype(np.float64) / 8.0

        guide_range = max(
            self.cfg.effective_guide_range_m *
            float(np.clip(self.cfg.guide_range_fraction, 0.1, 1.0)),
            1.0e-6)
        bin_width = math.radians(self.cfg.frontier_angular_bin_deg)
        angular_bins = np.floor((azimuth + math.pi) / bin_width).astype(int)
        unique_bins, bin_counts = np.unique(
            angular_bins, return_counts=True)
        maximum_bin_count = max(int(np.max(bin_counts)), 1)
        bin_gain_by_id = {
            int(bin_id): float(count) / maximum_bin_count
            for bin_id, count in zip(unique_bins, bin_counts)
        }
        cluster_gain = np.array(
            [bin_gain_by_id[int(bin_id)] for bin_id in angular_bins],
            dtype=np.float64)
        scores = (
            self.cfg.frontier_goal_progress_weight *
            goal_progress / guide_range +
            self.cfg.frontier_information_weight *
            (0.5 * info_gain + 0.5 * cluster_gain) -
            self.cfg.frontier_yaw_cost_weight *
            np.abs(azimuth) / math.pi)
        sides = np.where(cross > math.sin(math.radians(2.0)), 1,
                         np.where(cross < -math.sin(math.radians(2.0)),
                                  -1, self._preferred_side().value))
        if required_side is not None:
            sides.fill(required_side.value)
        if self._committed_side != CommittedSide.NONE:
            scores += self.cfg.frontier_side_commitment_bonus * (
                sides == self._committed_side.value)

        # Retain only one representative per angular sector. This clusters
        # the dense voxel frontier without a Python connected-component walk
        # and keeps the bounded-A* query count deterministic.
        representatives = {}
        for index in range(len(scores)):
            key = int(angular_bins[index])
            previous = representatives.get(key)
            if (previous is None or scores[index] > scores[previous] + 1.0e-12 or
                    (abs(scores[index] - scores[previous]) <= 1.0e-12 and
                     target_ranges[index] > target_ranges[previous])):
                representatives[key] = index

        candidates = []
        for index in representatives.values():
            target = position.copy()
            target[:2] += unit_xy[index] * target_ranges[index]
            # Obstacle bypass is strictly horizontal. Goal altitude is
            # recovered by GOAL_SEEK before/after the bypass, never by using
            # vertical motion as an avoidance action.
            target[2] = position[2]
            direction = target - position
            distance = float(np.linalg.norm(direction))
            if distance < self.cfg.minimum_guide_distance_m:
                continue
            direction /= distance
            candidates.append({
                "target_world": target,
                "direction_world": direction,
                "distance_m": distance,
                "side": (CommittedSide.LEFT if sides[index] > 0
                         else CommittedSide.RIGHT),
                "score": float(scores[index]),
                "azimuth_abs": float(abs(azimuth[index])),
            })
        candidates.sort(key=lambda item: (
            -item["score"], item["azimuth_abs"],
            -item["distance_m"],
            -item["side"].value))
        candidates = candidates[:self.cfg.frontier_candidate_limit]
        self._last_frontier_candidate_count = len(candidates)
        return candidates

    def _select_reachable_frontier(
        self, goal_dir_flu, observed_map, position_world, quaternion,
        required_side=None,
    ):
        candidates = self._extract_frontier_candidates(
            goal_dir_flu, observed_map, position_world, quaternion,
            required_side=required_side)
        self._extracted_frontier_count = len(candidates)
        self._reachable_frontier_count = 0
        self._frontier_rejection_reasons = {}
        position = np.asarray(position_world, dtype=np.float64)
        for idx, candidate in enumerate(candidates):
            direction_flu = self._world_to_flu(
                candidate["direction_world"], quaternion)
            # Check the endpoint against the map.  When the C++
            # reachability checker is installed, it additionally runs
            # bounded A* and validates the start position.
            endpoint_known = (
                observed_map.is_known_free(candidate["target_world"])
                if hasattr(observed_map, "is_known_free") else True)
            if not endpoint_known:
                self._frontier_rejection_reasons[idx] = "endpoint_unknown"
                continue
            reachable = self._fit_known_free_guide_distance(
                observed_map, position, direction_flu, quaternion,
                candidate["distance_m"], allow_clipping=False)
            if reachable + 1.0e-6 < self.cfg.minimum_guide_distance_m:
                # Try to infer a more specific reason from the reachability
                # checker result.  When it returns exactly 0.0 and the
                # C++ checker is installed, the bounded A* likely failed.
                if reachable <= 0.0:
                    self._frontier_rejection_reasons[idx] = (
                        "astar_no_path_or_start_unknown")
                else:
                    self._frontier_rejection_reasons[idx] = (
                        "insufficient_reachable_distance")
                continue
            self._reachable_frontier_count += 1
            target = position + candidate["direction_world"] * reachable
            self._selected_frontier_target_world = target
            self._selected_frontier_side = candidate["side"]
            self._selected_frontier_score = candidate["score"]
            self._selected_frontier_map_revision = int(
                observed_map.get_revision())
            return candidate["side"]
        return None

    def _fit_known_free_guide_distance(
        self, observed_map, position_world, direction_flu,
        quaternion, desired_distance_m, allow_clipping=True,
    ):
        """Choose the farthest guide endpoint reachable in the causal map.

        Formal collection delegates this check to the C++ local planner. It
        requires a hard-safe endpoint and a bounded-A* path through known
        free space when the direct segment is blocked. Full B-spline and
        dynamics optimization remains the responsibility of the independent
        30 Hz trajectory planner, so Guide selection does not optimize the
        same trajectory twice. Goal rays may be shortened to a safe prefix;
        a frontier viewpoint is checked at its exact standoff pose.
        The loop below is only an offline/static-test fallback when the C++
        planner callback is unavailable.
        """
        desired = max(0.0, float(desired_distance_m))
        if desired <= 1.0e-6 or observed_map is None:
            return desired
        direction_world = self._flu_to_world(
            self._unit3(direction_flu), quaternion)
        step = max(0.20, 2.0 * self.cfg.map_resolution_m)
        minimum_distance = (
            min(self.cfg.minimum_guide_distance_m, desired)
            if allow_clipping else desired)
        if self._guide_reachability_checker is not None:
            reachable = float(self._guide_reachability_checker(
                np.asarray(position_world, dtype=np.float64),
                self._current_velocity_world,
                direction_world,
                desired,
                minimum_distance,
                step))
            if not np.isfinite(reachable):
                return 0.0
            return float(np.clip(reachable, 0.0, desired))

        # Offline/static-test fallback when the C++ local planner is absent.
        candidate = desired
        while candidate + 1.0e-9 >= minimum_distance:
            endpoint = np.asarray(position_world) + direction_world * candidate
            ratio = observed_map.sample_known_free_ratio_along_corridor(
                start_world=np.asarray(position_world, dtype=np.float64),
                end_world=endpoint,
                radius_m=self.cfg.guide_swept_radius_m,
                spacing_m=max(0.05, 0.5 * self.cfg.map_resolution_m),
                min_clearance_m=self.cfg.guide_obstacle_clearance_m)
            if ratio + 1.0e-9 >= self.cfg.guide_corridor_min_ratio:
                return candidate
            candidate -= step
        return 0.0

    def _peek_side_feasible(
        self, side, goal_dir_flu, depth_m, observed_map,
        position_world, quaternion,
    ):
        del depth_m
        lateral = self._goal_relative_lateral(goal_dir_flu, side)
        ratio = self._corridor_ratio(
            observed_map, position_world, lateral,
            self.cfg.active_peek_distance_m,
            self.cfg.active_peek_safety_radius_m, quaternion)
        # Peek is active translation. It requires map-confirmed swept-volume
        # clearance because the camera is not necessarily aligned with motion.
        return ratio is not None and ratio >= 0.95

    def _make_goal_hold_guide(self, position_world, quaternion):
        forward_world = self._flu_to_world(
            np.array([1.0, 0.0, 0.0]), quaternion)
        guide = MacroGuide(
            valid=True,
            move_direction_flu=np.array([1.0, 0.0, 0.0]),
            move_distance_m=0.0,
            move_distance_norm=0.0,
            yaw_direction_flu_xy=np.array([1.0, 0.0]),
            move_target_world=position_world.copy(),
            look_target_world=position_world + forward_world,
            macro_state=MacroState.GOAL_HOLD.value,
            committed_side=CommittedSide.NONE.value,
            decision_reason="goal_hold")
        self._hold_guide(guide)
        return guide

    def _build_guide(
        self, state, goal_dir_flu, goal_distance_m, observed_map,
        position_world, quaternion, source_state=None,
    ):
        del source_state
        guide_range = (
            self.cfg.effective_guide_range_m *
            float(np.clip(self.cfg.guide_range_fraction, 0.1, 1.0)))
        needs_reachability_check = False
        if state == MacroState.GOAL_SEEK:
            move_dir = goal_dir_flu.copy()
            move_dist = min(goal_distance_m, guide_range)
            yaw_dir = self._unit2(move_dir[:2])
            reason = "goal_seek"
            needs_reachability_check = True
        elif state in (MacroState.BYPASS_LEFT, MacroState.BYPASS_RIGHT):
            side = (CommittedSide.LEFT
                    if state == MacroState.BYPASS_LEFT
                    else CommittedSide.RIGHT)
            if not self._map_supports_frontiers(observed_map):
                # Offline/static compatibility only. Formal collection never
                # synthesizes bypass rays; it uses the frontier target below.
                lateral = self._goal_relative_lateral(
                    goal_dir_flu, side.value)
                move_dir = self._unit3(
                    self.cfg.bypass_goal_weight * goal_dir_flu +
                    (1.0 - self.cfg.bypass_goal_weight) * lateral)
                move_dist = min(goal_distance_m * 0.60, guide_range)
                yaw_dir = self._unit2(move_dir[:2])
                reason = ("offline_bypass_left"
                          if side == CommittedSide.LEFT
                          else "offline_bypass_right")
                needs_reachability_check = True
            else:
                if (self._selected_frontier_target_world is None or
                        self._selected_frontier_side != side):
                    selected = self._select_reachable_frontier(
                        goal_dir_flu, observed_map, position_world, quaternion,
                        required_side=side)
                    if selected is None:
                        scan_state = self._scan_state(side)
                        self._state = scan_state
                        self._committed_side = side
                        self._begin_scan_session()
                        return self._build_guide(
                            scan_state, goal_dir_flu, goal_distance_m,
                            observed_map, position_world, quaternion)
                target_delta_world = (
                    np.asarray(self._selected_frontier_target_world,
                               dtype=np.float64) - position_world)
                move_dist = float(np.linalg.norm(target_delta_world))
                if move_dist <= 1.0e-6:
                    scan_state = self._scan_state(side)
                    self._state = scan_state
                    self._begin_scan_session()
                    return self._build_guide(
                        scan_state, goal_dir_flu, goal_distance_m,
                        observed_map, position_world, quaternion)
                move_dir = self._unit3(self._world_to_flu(
                    target_delta_world / move_dist, quaternion))
                yaw_dir = self._unit2(move_dir[:2])
                reason = ("frontier_bypass_left"
                          if side == CommittedSide.LEFT
                          else "frontier_bypass_right")
        elif state in (MacroState.ACTIVE_SCAN_LEFT,
                       MacroState.ACTIVE_SCAN_RIGHT):
            move_dir = np.array([1.0, 0.0, 0.0])
            move_dist = 0.0
            scan_sign = (CommittedSide.LEFT.value
                         if state == MacroState.ACTIVE_SCAN_LEFT
                         else CommittedSide.RIGHT.value)
            # The FLU instruction is held until the next 5 Hz macro tick.
            # Command one bounded relative-yaw increment per tick; using the
            # accumulated scan angle as another relative command would make
            # the vehicle increasingly overshoot and oscillate.
            scan_angle = scan_sign * min(
                math.radians(self.cfg.active_scan_min_angle_deg),
                math.radians(30.0))
            yaw_dir = np.array(
                [math.cos(scan_angle), math.sin(scan_angle)])
            reason = "active_scan_left" if scan_sign > 0 else "active_scan_right"
        elif state in (MacroState.ACTIVE_PEEK_LEFT,
                       MacroState.ACTIVE_PEEK_RIGHT):
            side = (CommittedSide.LEFT.value
                    if state == MacroState.ACTIVE_PEEK_LEFT
                    else CommittedSide.RIGHT.value)
            lateral = self._goal_relative_lateral(goal_dir_flu, side)
            move_dir = self._unit3(0.25 * goal_dir_flu + 0.75 * lateral)
            move_dist = min(self.cfg.active_peek_distance_m, 0.5 * guide_range)
            # During an active translation the camera must substantially
            # face the swept direction.  A small goal component still makes
            # the newly revealed region useful for the subsequent decision.
            yaw_dir = self._unit2(
                (0.20 * goal_dir_flu + 0.80 * move_dir)[:2])
            reason = "active_peek_left" if side > 0 else "active_peek_right"
            needs_reachability_check = True
        else:
            return self._make_goal_hold_guide(position_world, quaternion)

        if move_dist > 0.0 and needs_reachability_check:
            fitted_distance = self._fit_known_free_guide_distance(
                observed_map, position_world, move_dir, quaternion, move_dist)
            if fitted_distance <= 0.0:
                # A blocked goal ray does not define a bypass direction.
                # First try actual goal-directed frontier viewpoints from the
                # accumulated causal map. Each candidate is screened by C++
                # bounded A* before becoming a Guide.
                if (state == MacroState.GOAL_SEEK and
                        self._map_supports_frontiers(observed_map)):
                    selected = self._select_reachable_frontier(
                        goal_dir_flu, observed_map, position_world,
                        quaternion)
                    if selected is not None:
                        self._committed_side = selected
                        bypass_state = self._bypass_state(selected)
                        self._state = bypass_state
                        self._tick_time = 0.0
                        return self._build_guide(
                            bypass_state, goal_dir_flu, goal_distance_m,
                            observed_map, position_world, quaternion)

                # Keep observing the attempted/committed side. A single
                # rejected candidate must not reverse the scan at 15 degrees;
                # the ACTIVE_SCAN state itself changes side only after the
                # configured sector duration/angle has been exhausted.
                if self._scan_exhausted_this_tick:
                    # Just came from a full 360° scan exhaustion → don't
                    # re-enter scan.  Accept zero distance; the next tick
                    # will re-evaluate with fresh observations.
                    self._scan_exhausted_this_tick = False
                    move_dist = 0.0
                    reason += "_scan_exhausted_hold"
                    # Skip the scan re-entry below; fall through to guide
                    # construction.
                else:
                    if state == MacroState.ACTIVE_PEEK_LEFT:
                        scan_side = CommittedSide.RIGHT
                    elif state == MacroState.ACTIVE_PEEK_RIGHT:
                        scan_side = CommittedSide.LEFT
                    elif self._committed_side != CommittedSide.NONE:
                        scan_side = self._committed_side
                    else:
                        scan_side = self._preferred_side()
                    scan_state = self._scan_state(scan_side)
                    self._state = scan_state
                    self._committed_side = scan_side
                    self._begin_scan_session()
                    return self._build_guide(
                        scan_state, goal_dir_flu, goal_distance_m,
                        observed_map, position_world, quaternion)
            if fitted_distance + 1.0e-6 < move_dist:
                reason += "_map_clipped"
            move_dist = fitted_distance

        move_world = self._flu_to_world(move_dir, quaternion)
        yaw_world = self._flu_to_world(
            np.array([yaw_dir[0], yaw_dir[1], 0.0]), quaternion)
        move_target_world = position_world + move_world * move_dist
        look_target_world = position_world + self._unit3(yaw_world)

        known_free = occupied = unknown = revision = 0
        if observed_map is not None:
            known_free = int(observed_map.free_voxel_count())
            occupied = int(observed_map.occupied_voxel_count())
            total = int(np.prod(observed_map.get_occupancy().shape))
            unknown = max(0, total - int(observed_map.known_voxel_count()))
            revision = int(observed_map.get_revision())

        # Summarise rejection reasons: deduplicate into "key:N" pairs.
        rejection_summary_parts = []
        reason_counts = {}
        for reason in self._frontier_rejection_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for key, count in sorted(reason_counts.items()):
            rejection_summary_parts.append("{}:{}".format(key, count))
        rejection_summary = ";".join(rejection_summary_parts)

        guide = MacroGuide(
            valid=True,
            move_direction_flu=move_dir.astype(np.float64),
            move_distance_m=float(move_dist),
            move_distance_norm=float(np.clip(
                move_dist / max(self.cfg.effective_guide_range_m, 1.0e-6),
                0.0, 1.0)),
            yaw_direction_flu_xy=yaw_dir.astype(np.float64),
            move_target_world=move_target_world.astype(np.float64),
            look_target_world=look_target_world.astype(np.float64),
            macro_state=state.value,
            committed_side=self._committed_side.value,
            decision_reason=reason,
            macro_map_revision=revision,
            macro_known_free_count=known_free,
            macro_occupied_count=occupied,
            macro_unknown_count=unknown,
            frontier_candidate_count=self._last_frontier_candidate_count,
            extracted_frontier_candidate_count=(
                self._extracted_frontier_count),
            reachable_frontier_candidate_count=(
                self._reachable_frontier_count),
            frontier_rejection_summary=rejection_summary)
        self._hold_guide(guide)
        return guide

    def _hold_guide(self, guide):
        self._held_move_target_world = guide.move_target_world.copy()
        self._held_look_target_world = guide.look_target_world.copy()
        self._held_move_direction_flu = \
            guide.move_direction_flu.copy()
        self._held_move_distance_m = float(guide.move_distance_m)
        self._held_move_distance_norm = float(guide.move_distance_norm)
        self._held_yaw_direction_flu_xy = \
            guide.yaw_direction_flu_xy.copy()
        self._held_frontier_candidate_count = int(
            guide.frontier_candidate_count)
        self._held_extracted_frontier_count = int(
            guide.extracted_frontier_candidate_count)
        self._held_reachable_frontier_count = int(
            guide.reachable_frontier_candidate_count)
        self._held_rejection_reasons = dict(
            self._frontier_rejection_reasons)
        self._held_valid = True

    def get_held_guide_flu(self, current_position_world, quaternion_xyzw):
        """Return a zero-order-held 5 Hz FLU instruction.

        World targets are reconstructed from the held body-relative values so
        the 30 Hz local expert consumes exactly what the deployed student can
        produce between macro updates.
        """
        if not self._held_valid:
            return None
        position_world = np.asarray(current_position_world, dtype=np.float64)
        move_direction = self._held_move_direction_flu.copy()
        move_distance = self._held_move_distance_m
        yaw_direction = self._held_yaw_direction_flu_xy.copy()
        move_world = self._flu_to_world(move_direction, quaternion_xyzw)
        yaw_world = self._flu_to_world(
            np.array([yaw_direction[0], yaw_direction[1], 0.0]),
            quaternion_xyzw)
        move_target_world = position_world + move_world * move_distance
        look_target_world = position_world + self._unit3(yaw_world)
        # Summarise rejection reasons from the held data.
        rejection_summary_parts = []
        reason_counts = {}
        for reason in self._held_rejection_reasons.values():
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for key, count in sorted(reason_counts.items()):
            rejection_summary_parts.append("{}:{}".format(key, count))

        return {
            "move_direction_flu": move_direction,
            "move_distance_m": move_distance,
            "move_distance_norm": self._held_move_distance_norm,
            "yaw_direction_flu_xy": yaw_direction,
            "move_target_world": move_target_world,
            "look_target_world": look_target_world,
            "macro_state": self._state.value,
            "committed_side": self._committed_side.value,
            "frontier_candidate_count": (
                self._held_frontier_candidate_count),
            "extracted_frontier_candidate_count": (
                self._held_extracted_frontier_count),
            "reachable_frontier_candidate_count": (
                self._held_reachable_frontier_count),
            "frontier_rejection_summary": ";".join(
                rejection_summary_parts),
        }
