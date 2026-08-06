#!/usr/bin/env python3
"""Causal 5 Hz macro expert for horizontal active-perception navigation.

The expert uses only goal-relative state, the current depth frame, a finite
history observed occupancy map, and local-planner feedback.  It never queries
the global path, global ESDF, or ground-truth obstacle geometry.
"""

from __future__ import division, print_function

import collections
import heapq
import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class MacroState(Enum):
    GOAL_SEEK = "GOAL_SEEK"
    BYPASS = "BYPASS"
    PROBE = "PROBE"


class CommittedSide(Enum):
    NONE = 0
    # FLU is [forward, left, up], so positive y means left.
    LEFT = 1
    RIGHT = -1


def _obstacle_distance_field(grid):
    """Distance (in cells) from every cell to the nearest OCCUPIED cell,
    via multi-source BFS.  Returns an int32 array (0 on occupied)."""
    rows, cols = grid.shape
    dist = np.full((rows, cols), 999999, dtype=np.int32)
    queue = collections.deque()
    occ = grid == 2
    for i in range(rows):
        for j in range(cols):
            if occ[i, j]:
                dist[i, j] = 0
                queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        nd = dist[i, j] + 1
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and dist[ni, nj] > nd:
                    dist[ni, nj] = nd
                    queue.append((ni, nj))
    return dist


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

    # ── Episode-level scan budget exhaustion (Task 2) ──
    # True when episode scan budget is exhausted; the Manager should
    # terminate the episode rather than continue scanning.
    scan_budget_exhausted: bool = False


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
    frontier_information_weight: float = 0.06
    frontier_yaw_cost_weight: float = 0.10
    frontier_side_commitment_bonus: float = 0.15
    enter_blocked_frames: int = 3
    exit_clear_frames: int = 8
    minimum_progress_rate_m_s: float = 0.15
    minimum_commit_time_s: float = 0.8
    # ── V15 guide-line navigation ──
    # 2-D A* on the observed map with "unknown = free + small penalty".
    # The resulting guide line is the optimal path to the goal given
    # current information; the drone flies its farthest reachable point
    # (receding horizon) and the line is recomputed every tick.
    guide_unknown_cost: float = 1.05
    # A* path is "straight" (→ GOAL_SEEK) when path_length / straight ≤ this.
    guide_straight_ratio: float = 1.20
    # Yaw blend fraction toward the goal during BYPASS (camera leads).
    bypass_yaw_goal_weight: float = 0.60
    # ── V15.2 A* obstacle-distance penalty (guide-line clearance) ──
    # The observed map stores RAW obstacle-surface voxels and the ESDF
    # clearance is dist_to_surface - vehicle_radius.  A penalty radius of
    # only ~vehicle_radius puts the guide line exactly at ESDF=0, so the
    # 30 Hz B-spline optimizer grazes the boundary and the planner rejects
    # (UNKNOWN_SPACE) → safety stop → stutter.  Keep the guide line at
    # >= guide_line_clearance_m from surfaces (cells = ceil(m / res)), so
    # the drone centre always has ESDF clearance >= this minus vehicle
    # radius and the B-spline fits comfortably.
    guide_line_clearance_m: float = 0.55
    # Obstacle-distance penalty gain applied to guide-line search steps
    # inside `penalty_radius` cells of an OCCUPIED voxel.
    guide_penalty_gain: float = 0.6
    # ── V15.3 temporal consistency (adjacent guide lines must not jump) ──
    # The guide-line search is anchored to the PREVIOUS tick's guide line:
    # each candidate cell's signed lateral offset from the goal ray must
    # stay within [prev_offset - hard, prev_offset + hard] (hard = blocked),
    # with a soft band that adds `guide_lateral_cost` per metre beyond
    # `guide_lateral_soft_m`.  This prevents the line from flipping from
    # one side of an obstacle to the other while rounding it (e.g. going
    # around on the left, then suddenly routing to the right).
    guide_lateral_soft_m: float = 0.30
    guide_lateral_hard_m: float = 0.70
    guide_lateral_cost: float = 3.0
    # ── V15.2 explore advance ──
    # When a guide line EXISTS but its known-free advance is exhausted
    # (obstacle ahead, far side unobserved), the drone makes a small
    # explore step along the line into the unknown edge, camera leading.
    # Observation turns unknown→free and the known-free advance resumes.
    # Only when A* has NO route at all does the drone enter PROBE (pan).
    guide_explore_step_m: float = 0.8
    # Guard against infinite explore: after this many consecutive explore
    # steps without recovering a known-free advance, fall back to PROBE.
    max_explore_steps: int = 10
    # ── V15.7 observed-corridor gating ──
    # GOAL_SEEK is only allowed when the corridor toward the goal is
    # actually OBSERVED known-free (ratio >= this over the guide range);
    # an unobserved "straight" line must be routed via BYPASS instead of
    # blind-flying.  The small explore step only advances when at least
    # this fraction of the short corridor ahead is observed known-free.
    goal_corridor_clear_ratio: float = 0.60
    explore_ahead_min_ratio: float = 0.30
    # ── PROBE (replaces ACTIVE_SCAN 360° sweep) ──
    # When A* finds no path, briefly pan left then right to gather
    # information, then re-evaluate.  No full-circle rotation.
    probe_side_duration_s: float = 1.2
    probe_max_duration_s: float = 3.0
    probe_side_angle_deg: float = 90.0
    active_scan_yaw_rate_rps: float = 2.0
    # ── 90° minimum ensures the drone observes enough lateral space
    # before committing to BYPASS.  A narrow scan (<45°) leaves the
    # bypass corridor unobserved → UNKNOWN_SPACE / OPTIMIZATION_FAILED.
    active_scan_min_angle_deg: float = 90.0
    active_scan_max_duration_s: float = 5.0
    vertical_avoidance_enabled: bool = False
    vertical_active_perception_enabled: bool = False

    # ── Episode-level scan budget ──
    # Completed full-circle scans that yielded NO reachable frontier.
    max_completed_scans: int = 2
    # Cumulative scan time across the entire episode (s).
    max_cumulative_scan_time_s: float = 10.0
    # Consecutive scans with zero increase in known-free or reachable candidates.
    max_unsuccessful_scans: int = 2

    # ── Goal tolerance unified with Manager/planner ──
    goal_tolerance_m: float = 0.30

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
        if self.guide_line_clearance_m <= 0.0:
            raise ValueError("guide_line_clearance_m must be positive")
        if self.guide_penalty_gain < 0.0:
            raise ValueError("guide_penalty_gain must be non-negative")
        if not 0.0 < self.guide_lateral_soft_m < self.guide_lateral_hard_m:
            raise ValueError(
                "guide_lateral_soft_m must be in (0, guide_lateral_hard_m)")
        if self.guide_lateral_cost < 0.0:
            raise ValueError("guide_lateral_cost must be non-negative")
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
        # V15 PROBE: brief left/right pan replaces the 360° sweep.
        if (self.probe_side_duration_s <= 0.0 or
                self.probe_max_duration_s <= 0.0 or
                self.probe_side_angle_deg <= 0.0):
            raise ValueError("probe parameters must be positive")
        if self.probe_max_duration_s + 1.0e-9 < 2.0 * self.probe_side_duration_s:
            raise ValueError(
                "probe_max_duration_s ({:.2f} s) must cover both sides "
                "(2 × probe_side_duration_s = {:.2f} s)".format(
                    self.probe_max_duration_s,
                    2.0 * self.probe_side_duration_s))
        if (self.guide_unknown_cost < 1.0 or
                self.guide_straight_ratio <= 1.0):
            raise ValueError(
                "guide_unknown_cost must be ≥ 1.0 and "
                "guide_straight_ratio must be > 1.0")
        if self.map_history_seconds + 1.0e-9 < self.probe_max_duration_s:
            raise ValueError(
                "map_history_seconds ({:.2f} s) must be ≥ "
                "probe_max_duration_s ({:.2f} s)".format(
                    self.map_history_seconds,
                    self.probe_max_duration_s))
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
        # Episode-level scan budget (reset per episode, NOT per scan session)
        self._episode_cumulative_scan_time_s = 0.0
        self._episode_completed_scans = 0
        self._episode_unsuccessful_scans = 0
        self._episode_last_known_free_before_scan = -1
        self._scan_exhausted_this_episode = False

    def set_guide_reachability_checker(self, checker):
        """Install the expert-only C++ observed-ESDF reachability query."""
        self._guide_reachability_checker = checker

    def reset(self):
        self._reset()
        # ── Episode-level scan budget resets per new episode ──
        self._episode_cumulative_scan_time_s = 0.0
        self._episode_completed_scans = 0
        self._episode_unsuccessful_scans = 0
        self._episode_last_known_free_before_scan = -1
        self._scan_exhausted_this_episode = False

    def force_active_scan(
        self,
        goal_direction_flu,
        goal_distance_m,
        observed_map,
        current_position_world,
        current_quaternion_xyzw,
    ):
        """Replace a rejected MOVE Guide with a brief left/right probe.

        V15: enter PROBE (pan left then right).  Termination after
        probing is an upper-layer decision — the planner's rejection
        merely asks for information; it does not change macro intent.
        """
        self._state = MacroState.PROBE
        if self._scan_session_initial_yaw is None:
            self._begin_scan_session()
        else:
            self._scan_last_yaw = self._current_observed_yaw
        return self._build_guide(
            MacroState.PROBE,
            self._unit3(goal_direction_flu),
            float(goal_distance_m),
            observed_map,
            np.asarray(current_position_world, dtype=np.float64),
            np.asarray(current_quaternion_xyzw, dtype=np.float64),
            source_state=MacroState.PROBE)

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
        """V15 guide-line 3-state machine: GOAL_SEEK / BYPASS / PROBE.

        A 2-D A* guide line (unknown=free+penalty) is recomputed every
        tick.  GOAL_SEEK when the line is straight to the goal, BYPASS
        when it routes around obstacles, PROBE when A* finds no path
        (brief left/right pan, then re-evaluate; upper-layer decision on
        termination, independent of the planner).
        """
        # ── V15.2 perf: cache the guide-line computation for THIS update ──
        # _compute_guide_path / _select_guide_target are invoked several
        # times per tick (update() + _build_guide + recursive fallbacks)
        # with identical inputs; without caching that is up to 6 A* runs
        # (~70 ms each) per macro tick, which stalled phase-2 to ~600 ms
        # and made the drone fly in ~1.1 s jerky steps.
        self._guide_cache = {}
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
        goal_dist = float(goal_distance_m)

        # ── At goal: hold position, face goal ──
        if goal_dist < self.cfg.goal_tolerance_m:
            if self._state == MacroState.PROBE:
                self._end_scan_session()
            self._state = MacroState.GOAL_SEEK
            # Zero-distance GOAL_SEEK = hold
            return self._build_guide(
                MacroState.GOAL_SEEK, goal_dir, goal_dist, observed_map,
                position_world, quaternion, at_goal=True)

        previous_state = self._state
        state = previous_state

        # ── V15: guide-line decision every tick ──
        # 2-D A* on the observed map (unknown = free + penalty).  The
        # DECISION driver is the farthest REACHABLE advance point on the
        # line (_select_guide_target): no advance → PROBE; a straight
        # line → GOAL_SEEK; a bent line → BYPASS.
        target, _ = self._select_guide_target(
            goal_dir, observed_map, position_world, quaternion,
            goal_dist=goal_dist)
        path = self._compute_guide_path(
            goal_dir, observed_map, position_world, quaternion,
            goal_dist=goal_dist)
        straight = self._guide_path_straight(
            path, min(goal_dist, self.cfg.effective_guide_range_m),
            self.cfg.guide_straight_ratio)
        # V15.7: GOAL_SEEK requires the goal corridor to be OBSERVED
        # known-free.  A line that is straight only because UNKNOWN is
        # treated as free must not trigger blind GOAL_SEEK flight.
        goal_corridor_clear = self._goal_corridor_observed_clear(
            goal_dir, observed_map, position_world, quaternion,
            goal_dist=goal_dist)

        if state == MacroState.GOAL_SEEK:
            if target is None:
                if path is not None:
                    # V15.7: a route exists but its known-free advance is
                    # exhausted — the corridor ahead is unobserved or
                    # blocked.  Do NOT blind-advance into the unknown:
                    # route via BYPASS so the guide line's known points
                    # are followed (BYPASS pans when even those are gone).
                    self._explore_count = 0
                    self._blocked_counter += 1
                    self._clear_counter = 0
                    if self._blocked_counter >= self.cfg.enter_blocked_frames:
                        state = MacroState.BYPASS
                        self._tick_time = 0.0
                else:
                    # No route at all — accumulate blockage, then probe.
                    self._explore_count = 0
                    self._blocked_counter += 1
                    self._clear_counter = 0
                    if self._blocked_counter >= self.cfg.enter_blocked_frames:
                        state = MacroState.PROBE
                        self._begin_scan_session()
                        self._tick_time = 0.0
            else:
                self._explore_count = 0
                if straight and goal_corridor_clear:
                    # Guide line is straight AND the corridor is observed
                    # clear → keep seeking the goal.
                    self._blocked_counter = 0
                    self._clear_counter = self.cfg.exit_clear_frames
                else:
                    # V15.7: straight-but-unobserved or bent line → route
                    # around via the guide line's known points (BYPASS).
                    self._blocked_counter += 1
                    self._clear_counter = 0
                    if self._blocked_counter >= self.cfg.enter_blocked_frames:
                        state = MacroState.BYPASS
                        self._tick_time = 0.0

        elif state == MacroState.BYPASS:
            # The guide line (recomputed above) routes around an obstacle.
            # Follow it — _build_guide consumes the farthest reachable
            # point every tick, so the target rolls forward as the map
            # updates.  Straight + observed-clear line → back to GOAL_SEEK;
            # no advance → explore only when the corridor ahead is actually
            # observed; only a truly route-less state probes.
            if target is None:
                if path is not None:
                    explore_dir_flu = self._guide_explore_direction_flu(
                        path, position_world, quaternion, goal_dir)
                    if self._known_free_ahead(
                            explore_dir_flu, observed_map, position_world,
                            quaternion):
                        # V15.2 explore advance along the existing route
                        # (a short observed-corridor step, camera leads).
                        self._explore_count += 1
                        if self._explore_count >= self.cfg.max_explore_steps:
                            self._explore_count = 0
                            self._blocked_counter += 1
                            self._clear_counter = 0
                            if self._blocked_counter >= self.cfg.enter_blocked_frames:
                                state = MacroState.PROBE
                                self._begin_scan_session()
                                self._tick_time = 0.0
                        else:
                            self._blocked_counter = 0
                            self._clear_counter = 0
                    else:
                        # V15.7: nothing observed ahead — pan to observe
                        # instead of blind-flying into the unknown.
                        self._explore_count = 0
                        self._blocked_counter += 1
                        self._clear_counter = 0
                        if self._blocked_counter >= self.cfg.enter_blocked_frames:
                            state = MacroState.PROBE
                            self._begin_scan_session()
                            self._tick_time = 0.0
                else:
                    self._explore_count = 0
                    self._blocked_counter += 1
                    self._clear_counter = 0
                    if self._blocked_counter >= self.cfg.enter_blocked_frames:
                        state = MacroState.PROBE
                        self._begin_scan_session()
                        self._tick_time = 0.0
            else:
                self._explore_count = 0
                if straight and goal_corridor_clear:
                    self._clear_counter += 1
                    if self._clear_counter >= self.cfg.exit_clear_frames:
                        state = MacroState.GOAL_SEEK
                        self._tick_time = 0.0
                else:
                    # Still routing (or corridor not yet observed) — keep
                    # following the guide line.
                    self._blocked_counter = 0
                    self._clear_counter = 0

        elif state == MacroState.PROBE:

            # ── Brief left/right pan to gather information ──
            # No 360° sweep.  Pan left/right (yaw from _build_guide),
            # then re-evaluate.  Early exit as soon as a guide line
            # appears.  Termination is an upper-layer decision based on
            # the map, independent of the planner.
            if self._scan_session_initial_yaw is None:
                self._begin_scan_session()
            self._scan_time += dt
            done = self._scan_time >= self.cfg.probe_max_duration_s
            if target is not None and self._scan_time >= 0.5:
                done = True  # path recovered during the pan
            if done:
                self._end_scan_session()
                if target is None:
                    # Still no reachable advance after probing — terminate.
                    self._episode_cumulative_scan_time_s += self._scan_time
                    self._episode_completed_scans += 1
                    self._scan_exhausted_this_episode = True
                    budget_exceeded = (
                        self._episode_completed_scans > self.cfg.max_completed_scans
                        or self._episode_cumulative_scan_time_s > self.cfg.max_cumulative_scan_time_s)
                    self._scan_exhausted_this_tick = budget_exceeded
                    state = MacroState.GOAL_SEEK
                    self._blocked_counter = 0
                    self._clear_counter = self.cfg.exit_clear_frames
                    self._tick_time = 0.0
                    self._state = state
                    return self._build_guide(
                        state, goal_dir, goal_dist, observed_map,
                        position_world, quaternion,
                        source_state=previous_state,
                        scan_budget_exhausted=True)
                elif straight and goal_corridor_clear:
                    state = MacroState.GOAL_SEEK
                else:
                    state = MacroState.BYPASS
                self._blocked_counter = 0
                self._clear_counter = self.cfg.exit_clear_frames
                self._tick_time = 0.0

        self._state = state
        return self._build_guide(
            state, goal_dir, goal_dist, observed_map,
            position_world, quaternion, source_state=previous_state)

    def _reset(self):
        # V15.7: start in BYPASS, not GOAL_SEEK.  At episode start the
        # goal corridor is usually unobserved; GOAL_SEEK is only entered
        # once the corridor is confirmed known-free (see update()).
        self._state = MacroState.BYPASS
        self._committed_side = CommittedSide.NONE
        self._blocked_counter = 0
        self._clear_counter = 0
        self._tick_time = 0.0
        self._scan_angle_accum = 0.0
        self._scan_signed_angle = 0.0
        self._scan_last_yaw = None
        self._current_observed_yaw = None
        self._scan_time = 0.0
        # Scan-session state.
        self._scan_session_initial_yaw = None
        self._scan_exhausted_this_tick = False  # prevent immediate re-scan
        self._frontier_rejection_reasons = {}  # candidate_index → reason
        self._extracted_frontier_count = 0
        self._reachable_frontier_count = 0
        self._selected_frontier_target_world = None
        self._selected_frontier_side = CommittedSide.NONE
        self._selected_frontier_score = -float("inf")
        self._selected_frontier_map_revision = -1
        self._last_frontier_candidate_count = 0
        self._explore_count = 0  # V15.2 consecutive explore steps
        # V15.3: previous tick's guide line, used as the lateral temporal
        # reference so adjacent guide lines cannot flip sides of an obstacle.
        self._prev_guide_line_world = None
        self._guide_cache = {}
        self._last_bypass_score = -1.0  # V14: enforce score improvement after scan
        # ── V14.2 active bypass target ──
        # What we are ACTUALLY flying toward, decoupled from the last
        # search result (_selected_frontier_*).  update() decides whether
        # to retarget; _build_guide consumes the active target.
        self._active_bypass_target_world = None
        self._active_bypass_score = -float("inf")
        self._active_bypass_side = CommittedSide.NONE
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
        """Record the scan-session reference yaw.

        Always cleans up any stale session state first.
        """
        self._end_scan_session()
        self._scan_session_initial_yaw = self._current_observed_yaw
        self._scan_angle_accum = 0.0
        self._scan_signed_angle = 0.0
        self._scan_last_yaw = self._current_observed_yaw
        self._scan_time = 0.0
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

    def _preferred_side(self):
        return (CommittedSide.LEFT
                if str(self.cfg.preferred_symmetric_side).lower() == "left"
                else CommittedSide.RIGHT)

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

    def _goal_direction_feasible(
        self, goal_dir_flu, observed_map, position_world,
        quaternion,
    ):
        """Check if the goal-direction corridor is actually clear.

        Prevents GOAL_SEEK from being abandoned due to transient
        planner failures that drop progress_rate below threshold.
        Only the observed-map corridor matters for the blockage
        decision — planner failures are resolved by the planner's
        own retry logic, not by switching macro state.
        """
        ratio = self._corridor_ratio(
            observed_map, position_world, goal_dir_flu,
            self.cfg.effective_guide_range_m,
            self.cfg.blocked_corridor_radius_m, quaternion)
        if ratio is not None:
            return ratio >= self.cfg.goal_corridor_clear_ratio
        # V15.7: no map data yet — never assume the corridor is open.
        # An unobserved goal corridor must be routed via BYPASS/probe.
        return False

    def _goal_corridor_observed_clear(
        self, goal_dir_flu, observed_map, position_world, quaternion,
        goal_dist=None,
    ):
        """True only when the goal corridor is OBSERVED known-free.

        V15.7: GOAL_SEEK gate.  Unlike _goal_direction_feasible, never
        assumes an unobserved corridor is open.  The corridor is sampled
        out to min(effective_guide_range, goal_dist) so a goal that is
        right in front only requires a short observed corridor.
        """
        if observed_map is None:
            return False
        distance = float(goal_dist) if (
            goal_dist is not None and goal_dist >= 0.0
        ) else self.cfg.effective_guide_range_m
        distance = min(distance, self.cfg.effective_guide_range_m)
        if distance <= 0.0:
            return True
        ratio = self._corridor_ratio(
            observed_map, position_world, goal_dir_flu, distance,
            self.cfg.blocked_corridor_radius_m, quaternion)
        if ratio is None:
            return False
        return ratio >= self.cfg.goal_corridor_clear_ratio

    def _known_free_ahead(
        self, direction_flu, observed_map, position_world, quaternion,
    ):
        """True when a short corridor ahead is observed known-free.

        V15.7: gates the small explore step so the drone only advances
        into space the camera has partially cleared.  At episode start
        (nothing observed toward the goal) this returns False, so the
        drone pans instead of blind-flying.
        """
        if observed_map is None:
            return False
        ratio = self._corridor_ratio(
            observed_map, position_world, direction_flu,
            self.cfg.guide_explore_step_m,
            self.cfg.guide_swept_radius_m, quaternion)
        if ratio is None:
            return False
        return ratio >= self.cfg.explore_ahead_min_ratio

    def _guide_explore_direction_flu(
        self, path, position_world, quaternion, fallback_dir_flu,
    ):
        """First guide-line segment direction (for explore gating)."""
        if path is not None and len(path) >= 2:
            seg = np.asarray(path[1], dtype=np.float64) - np.asarray(
                path[0], dtype=np.float64)
            seg_len = float(np.linalg.norm(seg))
            if seg_len > 1.0e-6:
                return self._unit3(self._world_to_flu(
                    seg / seg_len, quaternion))
        return self._unit3(fallback_dir_flu)

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
        # ── Layered architecture: Macro Expert is pure perception ──
        # Planner failures, safety stops, and low progress rate are
        # lower-layer issues that must NOT influence the macro-level
        # blockage decision.  Only the observed map determines whether
        # the goal corridor is physically obstructed.  If the planner
        # cannot execute the macro's intent, the episode should
        # terminate — not silently change the macro's decision.
        del local_blocked, local_progress_rate, local_feasible
        ratio = self._corridor_ratio(
            observed_map, position_world, goal_dir_flu,
            self.cfg.effective_guide_range_m,
            self.cfg.blocked_corridor_radius_m, quaternion)
        if ratio is not None:
            return ratio < 0.60
        # Depth-only fallback when map not yet populated (first frame).
        return self._depth_sector_score(depth_m, 0) < 0.55

    # ────────────────────────────────────────────────────────────────────
    # V15 guide-line navigation
    # ────────────────────────────────────────────────────────────────────
    # A guide line is the optimal path from the current position toward
    # the goal under the "unknown = free + small penalty" assumption,
    # computed by 2-D A* on the observed-map slice at flight height.
    # The macro flies the farthest reachable point of the line (receding
    # horizon); every tick the line is recomputed as the map updates.
    @staticmethod
    def _astar_2d(grid, start, target, unknown_cost,
                  penalty_radius=6, penalty_gain=0.6):
        """A* on a 2-D occupancy grid (UNKNOWN=0, FREE=1, OCCUPIED=2).

        penalty_radius is in CELLS: cells within this distance of an
        OCCUPIED cell accumulate extra step cost so the guide line keeps
        away from obstacle surfaces (caller converts a real-world
        clearance target to cells using the map resolution).

        Returns a list of (x, y) grid cells from start to target
        (inclusive), or None if no path exists.
        """
        rows, cols = grid.shape
        sx, sy = start
        tx, ty = target
        if not (0 <= sx < rows and 0 <= sy < cols):
            return None
        if not (0 <= tx < rows and 0 <= ty < cols):
            return None
        if grid[tx, ty] == 2:
            # Target occupied — backtrack toward start for nearest walkable.
            found = None
            steps = max(abs(tx - sx), abs(ty - sy))
            for s in range(1, steps + 1):
                cx = int(round(sx + (tx - sx) * (1.0 - s / (steps + 1))))
                cy = int(round(sy + (ty - sy) * (1.0 - s / (steps + 1))))
                cx = int(np.clip(cx, 0, rows - 1))
                cy = int(np.clip(cy, 0, cols - 1))
                if grid[cx, cy] != 2:
                    found = (cx, cy)
                    break
            if found is None:
                return None
            target = found
            tx, ty = target

        open_heap = [(0.0, 0, (sx, sy))]
        g_score = {(sx, sy): 0.0}
        came_from = {}
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        # ── V15.2: obstacle-distance penalty ──
        # Cells within `penalty_radius` cells of an OCCUPIED cell get extra
        # cost, so the guide line keeps away from obstacles instead of
        # hugging their inflated edge.  The B-spline optimizer keeps final
        # clearance, but the MACRO line must not aim through a 0.1 m gap.
        dist_field = _obstacle_distance_field(grid)
        counter = 0
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current == (tx, ty):
                break
            cx, cy = current
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue
                cell = grid[nx, ny]
                if cell == 2:
                    continue
                step_cost = (unknown_cost if cell == 0 else 1.0)
                if dx != 0 and dy != 0:
                    step_cost *= math.sqrt(2.0)
                # Push the line away from obstacles (soft constraint).
                d_obs = dist_field[nx, ny]
                if d_obs < penalty_radius:
                    step_cost += penalty_gain * (
                        penalty_radius - d_obs)
                tentative = g_score[current] + step_cost
                if tentative < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = tentative
                    came_from[(nx, ny)] = current
                    h = math.hypot(nx - tx, ny - ty)
                    counter += 1
                    heapq.heappush(
                        open_heap, (tentative + h, counter, (nx, ny)))

        if (tx, ty) not in came_from and (tx, ty) != (sx, sy):
            return None
        path = [(tx, ty)]
        cur = (tx, ty)
        while cur != (sx, sy):
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    def _compute_guide_path(
        self, goal_dir_flu, observed_map, position_world, quaternion,
        goal_dist=None,
    ):
        """Compute the 2-D guide line in world coordinates.

        Returns an (N, 3) array of world points (start → target) or None.
        The A* target is the point guide_range ahead along the goal ray,
        clamped into the rolling map; if the goal is inside the map, the
        goal itself is used.

        V15.2: result is cached per update() tick (inputs are identical
        for every call within one tick), eliminating redundant searches.

        V15.3: the guide line is a goal-directed DEPTH-FIRST search toward
        the terminal, implemented in C++ (`_il_local_planner.
        compute_guide_line_2d`) with a Python DFS fallback for offline
        tooling.  It is anchored to the previous tick's guide line so
        adjacent updates cannot jump to the opposite side of an obstacle.

        V15.5: `goal_dist` clamps the guide-line range to the remaining
        goal distance so the line (and the executed target) never extends
        PAST the goal — the debug view no longer shows the candidate
        overshooting the terminal in the latter half of the flight.
        goal_dist is constant within a tick, so the per-tick cache stays
        valid.
        """
        cache = getattr(self, "_guide_cache", None)
        if cache is not None and "path" in cache:
            return cache["path"]
        if not self._map_supports_frontiers(observed_map):
            return None
        try:
            occupancy = observed_map.get_occupancy(copy=False)
        except TypeError:
            occupancy = observed_map.get_occupancy()
        occupancy = np.asarray(occupancy, dtype=np.uint8)
        if occupancy.ndim != 3 or min(occupancy.shape) < 3:
            return None
        origin = np.asarray(observed_map.get_origin(), dtype=np.float64)
        res = float(observed_map.get_resolution())
        if res <= 0.0:
            return None
        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        grid = np.floor((position - origin) / res).astype(np.int32)
        iz = int(np.clip(grid[2], 0, occupancy.shape[2] - 1))
        grid_2d = occupancy[:, :, iz]

        goal_world = self._flu_to_world(goal_dir_flu, quaternion)
        guide_range = (
            self.cfg.effective_guide_range_m *
            float(np.clip(self.cfg.guide_range_fraction, 0.1, 1.0)))
        # V15.5: never extend the guide line past the goal.
        if goal_dist is not None and goal_dist >= 0.0:
            guide_range = min(guide_range, float(goal_dist))
        target_world = position + self._unit3(goal_world) * guide_range

        # V15.2: resolution-aware obstacle-distance penalty.  Convert the
        # real-world guide-line clearance target to cells; at 0.10 m this
        # is 6 cells (0.55 m from the raw surface = ESDF clearance 0.25).
        penalty_radius = max(
            2, int(math.ceil(self.cfg.guide_line_clearance_m / res)))
        path = self._guide_line_search(
            grid_2d, origin, res, position, target_world, penalty_radius)
        if path is None:
            cache = getattr(self, "_guide_cache", None)
            if cache is not None:
                cache["path"] = None
            return None
        cache = getattr(self, "_guide_cache", None)
        if cache is not None:
            cache["path"] = path
        return path

    def _guide_line_search(
        self, grid_2d, origin, res, position, target_world, penalty_radius,
    ):
        """Goal-directed DFS guide line (C++ preferred, Python fallback).

        Returns an (N, 3) world path (start → target) or None.  The search
        dives toward the terminal (remaining Chebyshev distance first),
        keeps ESDF clearance via the obstacle-distance penalty, and stays
        within a lateral band around the previous tick's guide line.
        """
        prev_line = getattr(self, "_prev_guide_line_world", None)
        if prev_line is not None and len(prev_line) >= 2:
            prev_line_xy = np.asarray(
                prev_line, dtype=np.float64)[:, :2].copy()
        else:
            prev_line_xy = np.zeros((0, 2), dtype=np.float64)
        try:
            import _il_local_planner as _cpp
            if hasattr(_cpp, "compute_guide_line_2d"):
                out = _cpp.compute_guide_line_2d(
                    np.ascontiguousarray(grid_2d, dtype=np.uint8),
                    np.ascontiguousarray(origin[:2], dtype=np.float64),
                    float(res),
                    np.ascontiguousarray(
                        position[:2], dtype=np.float64),
                    np.ascontiguousarray(
                        target_world[:2], dtype=np.float64),
                    float(self.cfg.guide_unknown_cost),
                    int(penalty_radius),
                    float(self.cfg.guide_penalty_gain),
                    np.ascontiguousarray(prev_line_xy, dtype=np.float64),
                    float(self.cfg.guide_lateral_soft_m),
                    float(self.cfg.guide_lateral_hard_m),
                    float(self.cfg.guide_lateral_cost))
                arr = np.asarray(out, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 1:
                    return None
                path = np.zeros((arr.shape[0], 3), dtype=np.float64)
                path[:, :2] = arr
                path[:, 2] = position[2]
                return path
        except Exception:
            pass
        # Python fallback (offline tooling / no C++ module).
        return self._guide_line_search_python(
            grid_2d, origin, res, position, target_world, penalty_radius)

    def _guide_line_search_python(
        self, grid_2d, origin, res, position, target_world, penalty_radius,
    ):
        """Python mirror of the C++ goal-directed guide-line search.

        Weighted A* (h_weight > 1) that dives depth-first toward the
        terminal while ACCUMULATING the obstacle-distance and lateral
        temporal penalties along the path (a raw DFS cannot accumulate
        penalty and therefore hugs obstacle boundaries).  Same lateral
        temporal band around the previous guide line.
        """
        rows, cols = grid_2d.shape
        sx = int(np.clip(
            int(np.floor((position[0] - origin[0]) / res)), 0, rows - 1))
        sy = int(np.clip(
            int(np.floor((position[1] - origin[1]) / res)), 0, cols - 1))
        tx = int(np.clip(
            int(np.floor((target_world[0] - origin[0]) / res)), 0, rows - 1))
        ty = int(np.clip(
            int(np.floor((target_world[1] - origin[1]) / res)), 0, cols - 1))
        if grid_2d[sx, sy] == 2:
            return None
        if grid_2d[tx, ty] == 2:
            found = False
            steps = max(abs(tx - sx), abs(ty - sy))
            for s in range(1, steps + 1):
                f = 1.0 - s / (steps + 1)
                cx = int(round(sx + (tx - sx) * f))
                cy = int(round(sy + (ty - sy) * f))
                cx = int(np.clip(cx, 0, rows - 1))
                cy = int(np.clip(cy, 0, cols - 1))
                if grid_2d[cx, cy] != 2:
                    tx, ty = cx, cy
                    found = True
                    break
            if not found:
                return None

        dist = _obstacle_distance_field(grid_2d)
        gd = target_world[:2] - position[:2]
        glen = float(np.linalg.norm(gd))
        if glen < 1.0e-9:
            return np.array(
                [[position[0], position[1], position[2]]],
                dtype=np.float64)
        gx, gy = gd[0] / glen, gd[1] / glen

        prev = getattr(self, "_prev_guide_line_world", None)
        ref = []
        if prev is not None and len(prev) >= 2:
            for p in prev:
                fx = p[0] - position[0]
                fy = p[1] - position[1]
                ref.append((gx * fx + gy * fy, gx * fy - gy * fx))
            ref.sort()

        def ref_lat(f):
            if not ref:
                return 0.0
            if len(ref) == 1:
                return ref[0][1]
            if f <= ref[0][0]:
                return ref[0][1]
            if f >= ref[-1][0]:
                return ref[-1][1]
            lo, hi = 0, len(ref) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if ref[mid][0] <= f:
                    lo = mid
                else:
                    hi = mid
            f0, f1 = ref[lo][0], ref[hi][0]
            t = (f - f0) / (f1 - f0) if f1 > f0 else 0.0
            return ref[lo][1] + t * (ref[hi][1] - ref[lo][1])

        soft = self.cfg.guide_lateral_soft_m
        hard = self.cfg.guide_lateral_hard_m
        lcost = self.cfg.guide_lateral_cost
        unknown_cost = self.cfg.guide_unknown_cost
        pgain = self.cfg.guide_penalty_gain
        sqrt2 = math.sqrt(2.0)
        h_weight = 1.6  # >1 -> greedy depth-first dive toward the goal
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        parent = {}
        inf = float("inf")
        g_score = {(sx, sy): 0.0}
        # heap entries (f, -g, i, j): on f ties prefer the DEEPER node
        import heapq as _heapq
        open_heap = [(h_weight * max(abs(sx - tx), abs(sy - ty)),
                      0.0, sx, sy)]
        found_t = False
        best = (sx, sy)
        best_h = max(abs(sx - tx), abs(sy - ty))
        max_exp = rows * cols * 8
        expansions = 0
        while open_heap:
            _, neg_g, ci, cj = _heapq.heappop(open_heap)
            if -neg_g > g_score.get((ci, cj), inf) + 1.0e-9:
                continue
            expansions += 1
            if expansions >= max_exp:
                break
            if ci == tx and cj == ty:
                found_t = True
                best = (ci, cj)
                break
            h = max(abs(ci - tx), abs(cj - ty))
            if h < best_h:
                best_h = h
                best = (ci, cj)
            cur_g = -neg_g
            for dx, dy in neighbors:
                ni, nj = ci + dx, cj + dy
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if grid_2d[ni, nj] == 2:
                    continue
                wx = origin[0] + (ni + 0.5) * res
                wy = origin[1] + (nj + 0.5) * res
                fx = wx - position[0]
                fy = wy - position[1]
                f = gx * fx + gy * fy
                lat = gx * fy - gy * fx
                dev = abs(lat - ref_lat(f))
                if dev > hard:
                    continue
                d_obs = int(dist[ni, nj])
                step = (unknown_cost if grid_2d[ni, nj] == 0 else 1.0)
                if dx != 0 and dy != 0:
                    step *= sqrt2
                if d_obs < penalty_radius:
                    step += pgain * (penalty_radius - d_obs)
                if dev > soft:
                    step += lcost * (dev - soft)
                ng = cur_g + step
                if ng >= g_score.get((ni, nj), inf) - 1.0e-9:
                    continue
                g_score[(ni, nj)] = ng
                parent[(ni, nj)] = (ci, cj)
                hh = max(abs(ni - tx), abs(nj - ty))
                _heapq.heappush(
                    open_heap, (ng + h_weight * hh, -ng, ni, nj))

        end = best
        cells = [end]
        while cells[-1] != (sx, sy):
            if cells[-1] not in parent:
                break
            cells.append(parent[cells[-1]])
        cells.reverse()
        path = np.zeros((len(cells), 3), dtype=np.float64)
        for k, (gi, gj) in enumerate(cells):
            path[k, 0] = origin[0] + (gi + 0.5) * res
            path[k, 1] = origin[1] + (gj + 0.5) * res
            path[k, 2] = position[2]
        return path

    @staticmethod
    def _guide_path_straight(path, straight_dist, ratio):
        """True when the guide line is essentially straight to the goal."""
        if path is None or len(path) < 2:
            return True
        total = 0.0
        for i in range(1, len(path)):
            total += float(np.linalg.norm(path[i] - path[i - 1]))
        direct = float(np.linalg.norm(path[-1] - path[0]))
        if direct <= 1.0e-6:
            return total <= ratio * straight_dist
        return total <= ratio * direct

    @staticmethod
    def _advance_along_path(path, position, distance_m):
        """Return the world point `distance_m` along the guide line from the
        drone position (walking the path segments).  Returns the path end if
        the line is shorter, or None when no path."""
        if path is None or len(path) < 2:
            return None
        remaining = max(0.0, float(distance_m))
        for i in range(1, len(path)):
            seg = path[i] - path[i - 1]
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1.0e-9:
                continue
            if remaining <= seg_len:
                return path[i - 1] + seg * (remaining / seg_len)
            remaining -= seg_len
        return path[-1].copy()

    def _select_guide_target(
        self, goal_dir_flu, observed_map, position_world, quaternion,
        goal_dist=None,
    ):
        """Choose the farthest reachable point along the guide line.

        Returns (target_world, score) with score = reachable/guide_range,
        or (None, -inf) when no reachable advance exists.

        V15.2: result is cached per update() tick (inputs are identical
        for every call within one tick), eliminating redundant reachable
        scans that dominated the phase-2 budget.

        V15.5: `goal_dist` (constant within a tick) clamps the walk so the
        executed target never extends past the goal.
        """
        cache = getattr(self, "_guide_cache", None)
        if cache is not None and "target" in cache:
            return cache["target"]
        path = self._compute_guide_path(
            goal_dir_flu, observed_map, position_world, quaternion,
            goal_dist=goal_dist)
        if path is None or len(path) < 2:
            result = (None, -float("inf"))
            if cache is not None:
                cache["target"] = result
            return result
        position = np.asarray(position_world, dtype=np.float64).reshape(3)
        guide_range = (
            self.cfg.effective_guide_range_m *
            float(np.clip(self.cfg.guide_range_fraction, 0.1, 1.0)))
        if goal_dist is not None and goal_dist >= 0.0:
            guide_range = min(guide_range, float(goal_dist))

        # V15.7 goal snap: when the goal is directly in front (within the
        # guide range) and its whole corridor is observed known-free,
        # target the goal EXACTLY.  Previously the target hovered at the
        # last path waypoint ~0.1 m short of the goal, so the debug view
        # showed the selected point floating around the terminal and
        # guide_is_final (1 mm tolerance) stayed False.
        if (goal_dist is not None and goal_dist >= 0.0 and
                goal_dist <= guide_range + 1.0e-3):
            goal_pt = position + self._unit3(self._flu_to_world(
                goal_dir_flu, quaternion)) * goal_dist
            goal_known = (
                observed_map.is_known_free(goal_pt)
                if hasattr(observed_map, "is_known_free") else True)
            if goal_known:
                reach_to_goal = self._fit_known_free_guide_distance(
                    observed_map, position, goal_dir_flu, quaternion,
                    goal_dist, allow_clipping=True)
                if reach_to_goal + 1.0e-6 >= goal_dist:
                    result = (goal_pt, 1.0)
                    cache = getattr(self, "_guide_cache", None)
                    if cache is not None:
                        cache["target"] = result
                    return result

        # Walk the path outward, measuring cumulative distance, and check
        # reachability at each waypoint.  Pick the farthest reachable.
        #
        # V15.1 SAFETY CONTRACT: the guide line may pass through UNKNOWN
        # space (it is a planning direction), but an EXECUTED advance point
        # must lie in OBSERVED known-free space.  The drone only flies to
        # places the camera has already cleared; the line then extends as
        # new observations turn unknown into free.  This is what makes
        # exploration serve navigation instead of blind-flying.
        best_target = None
        best_score = -float("inf")
        cumulative = 0.0
        for i in range(1, len(path)):
            cumulative += float(np.linalg.norm(path[i] - path[i - 1]))
            if cumulative > guide_range:
                break
            # Hard constraint: only observed free waypoints are flyable.
            if (hasattr(observed_map, "is_known_free") and
                    not observed_map.is_known_free(path[i])):
                # Stop at the first unobserved waypoint — anything beyond
                # is still unknown and must not be targeted.
                break
            delta = path[i] - position
            dist = float(np.linalg.norm(delta))
            if dist < 1.0e-6:
                continue
            direction_flu = self._world_to_flu(
                delta / dist, quaternion)
            reachable = self._fit_known_free_guide_distance(
                observed_map, position, direction_flu, quaternion, dist,
                allow_clipping=True)
            if reachable + 1.0e-6 < self.cfg.minimum_guide_distance_m:
                continue
            # Re-anchor the target to the actual reachable distance.
            target = position + (delta / dist) * reachable
            score = cumulative / guide_range
            if score > best_score:
                best_score = score
                best_target = target
        if best_target is None:
            result = (None, -float("inf"))
            cache = getattr(self, "_guide_cache", None)
            if cache is not None:
                cache["target"] = result
            return result
        result = (best_target, float(best_score))
        cache = getattr(self, "_guide_cache", None)
        if cache is not None:
            cache["target"] = result
        return result

    def _choose_bypass_side(
        self, goal_dir_flu, depth_m, observed_map,
        position_world, quaternion,
    ):
        # ── Goal-first: if the goal corridor is clear, stay in GOAL_SEEK ──
        # Frontiers are exploration targets for obstructed paths.  When the
        # observed map shows a clear corridor toward the goal, the goal IS
        # the target — not a frontier somewhere near it.  This prevents the
        # drone from orbiting the goal through successive frontier targets
        # when it could just fly straight there.
        if self._goal_direction_feasible(
                goal_dir_flu, observed_map, position_world, quaternion):
            self._state = MacroState.GOAL_SEEK
            self._committed_side = CommittedSide.NONE
            self._consecutive_scan_restarts = 0
            self._blocked_counter = 0
            self._clear_counter = self.cfg.exit_clear_frames
            return MacroState.GOAL_SEEK

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

    def _find_best_bypass_point(
        self, goal_dir_flu, observed_map, position_world, quaternion,
    ):
        """Try both left and right sides.  Return (best_CommittedSide, score)
        for the reachable frontier that maximises goal progress, or
        (None, -inf) if neither side has a reachable point.

        V14: replaces _choose_bypass_side's multi-step fallback with a
        simple max-over-sides comparison.  No scanning, no side commitment
        bias — just pick the single best exploration point toward the goal.
        """
        best_side = None
        best_score = -float("inf")
        best_target = None
        for side in (CommittedSide.LEFT, CommittedSide.RIGHT):
            selected = self._select_reachable_frontier(
                goal_dir_flu, observed_map, position_world, quaternion,
                required_side=side)
            if selected is not None:
                score = float(self._selected_frontier_score)
                if score > best_score + 1.0e-9:
                    best_score = score
                    best_side = selected
                    best_target = np.asarray(
                        self._selected_frontier_target_world,
                        dtype=np.float64).copy()
        if best_side is not None:
            # Restore the cache so it points at the BEST candidate, not
            # whatever the last side in the loop happened to find.
            self._selected_frontier_target_world = best_target
            self._selected_frontier_side = best_side
            self._selected_frontier_score = best_score
        return best_side, best_score

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
        scores = (
            self.cfg.frontier_goal_progress_weight *
            goal_progress / guide_range +
            self.cfg.frontier_information_weight * info_gain -
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

    def _build_guide(
        self, state, goal_dir_flu, goal_distance_m, observed_map,
        position_world, quaternion, source_state=None,
        scan_budget_exhausted=False, at_goal=False,
    ):
        """V15: GOAL_SEEK / BYPASS / PROBE only, driven by the guide line."""
        del source_state
        guide_range = (
            self.cfg.effective_guide_range_m *
            float(np.clip(self.cfg.guide_range_fraction, 0.1, 1.0)))
        needs_reachability_check = False
        goal_dist = float(goal_distance_m)

        if state == MacroState.GOAL_SEEK:
            if at_goal:
                move_dir = goal_dir_flu.copy()
                move_dist = 0.0
                reason = "goal_hold_at_target"
                yaw_dir = self._unit2(move_dir[:2])
            elif scan_budget_exhausted:
                move_dir = goal_dir_flu.copy()
                move_dist = 0.0
                reason = "scan_exhausted|goal_seek_zero_move"
                yaw_dir = self._unit2(move_dir[:2])
            else:
                # ── V15.1: follow the guide line, never blind-fly ──
                # GOAL_SEEK targets the farthest OBSERVED known-free
                # advance along the guide line toward the goal.  If the
                # line still passes through unknown space (not yet seen),
                # the drone flies the last known point and the camera
                # leads toward the goal to reveal the rest.
                target, _ = self._select_guide_target(
                    goal_dir_flu, observed_map, position_world, quaternion,
                    goal_dist=goal_distance_m)
                if target is not None:
                    target_delta_world = (
                        np.asarray(target, dtype=np.float64) - position_world)
                    move_dist = float(np.linalg.norm(target_delta_world))
                    move_dir = self._unit3(self._world_to_flu(
                        target_delta_world / max(move_dist, 1.0e-6),
                        quaternion))
                    reason = "goal_seek_guide"
                else:
                    # No known-free advance.  If a guide line still exists
                    # (obstacle ahead, far side unobserved), make a small
                    # EXPLORE step along it with the camera leading —
                    # observation turns unknown→free and advance resumes.
                    path = self._compute_guide_path(
                        goal_dir_flu, observed_map, position_world,
                        quaternion, goal_dist=goal_distance_m)
                    explore = self._advance_along_path(
                        path, position_world, self.cfg.guide_explore_step_m)
                    if explore is not None:
                        explore_delta = (
                            np.asarray(explore, dtype=np.float64)
                            - position_world)
                        explore_dist = float(np.linalg.norm(explore_delta))
                        explore_dir_flu = self._unit3(self._world_to_flu(
                            explore_delta / max(explore_dist, 1.0e-6),
                            quaternion))
                        # V15.7: only advance when the short corridor ahead
                        # is observed known-free; otherwise hold and yaw
                        # toward the goal so the camera reveals it.
                        if (explore_dist > 0.2 and self._known_free_ahead(
                                explore_dir_flu, observed_map,
                                position_world, quaternion)):
                            move_dist = explore_dist
                            move_dir = explore_dir_flu
                            reason = "goal_seek_explore"
                        else:
                            move_dist = 0.0
                            move_dir = goal_dir_flu.copy()
                            reason = "goal_seek_explore_hold"
                    else:
                        # No route at all — safe clipped prefix of the goal
                        # ray only; reachability clips to observed space.
                        move_dir = goal_dir_flu.copy()
                        move_dist = min(goal_dist, guide_range)
                        reason = "goal_seek_ray_clipped"
                        needs_reachability_check = True
                yaw_dir = self._unit2(move_dir[:2])

        elif state == MacroState.BYPASS:
            # ── V15: follow the guide line ──
            # Fly the farthest reachable point of the 2-D A* line.  The
            # line is recomputed every tick, so the target rolls forward
            # as the map updates.  Camera leads toward the goal.
            target, _ = self._select_guide_target(
                goal_dir_flu, observed_map, position_world, quaternion,
                goal_dist=goal_distance_m)
            if target is not None:
                target_delta_world = (
                    np.asarray(target, dtype=np.float64) - position_world)
                move_dist = float(np.linalg.norm(target_delta_world))
                move_dir = self._unit3(self._world_to_flu(
                    target_delta_world / move_dist, quaternion))
                # V15.7: the planned yaw is the direction to the selected
                # candidate point (also saved as the macro_yaw_dir label).
                # The nose points where the drone is flying — no more goal
                # blend that leaves the candidate off to the side.
                yaw_dir = self._unit2(move_dir[:2])
                reason = "guide_bypass"
            else:
                # No known-free advance.  If a route still exists, explore
                # forward along the line (camera leads); only when A* has
                # no route at all do we enter PROBE.
                path = self._compute_guide_path(
                    goal_dir_flu, observed_map, position_world, quaternion,
                    goal_dist=goal_distance_m)
                explore = self._advance_along_path(
                    path, position_world, self.cfg.guide_explore_step_m)
                if explore is not None:
                    explore_delta = (
                        np.asarray(explore, dtype=np.float64)
                        - position_world)
                    explore_dist = float(np.linalg.norm(explore_delta))
                    explore_dir_flu = self._unit3(self._world_to_flu(
                        explore_delta / max(explore_dist, 1.0e-6),
                        quaternion))
                    # V15.7: only advance the small explore step when the
                    # short corridor ahead is actually observed; otherwise
                    # hold and let the camera reveal it (PROBE pans).
                    if (explore_dist > 0.2 and self._known_free_ahead(
                            explore_dir_flu, observed_map, position_world,
                            quaternion)):
                        move_dist = explore_dist
                        move_dir = explore_dir_flu
                        yaw_dir = self._unit2(move_dir[:2])
                        reason = "guide_bypass_explore"
                    else:
                        move_dist = 0.0
                        move_dir = self._unit3(self._world_to_flu(
                            self._flu_to_world(goal_dir_flu, quaternion),
                            quaternion))
                        yaw_dir = self._unit2(goal_dir_flu[:2])
                        reason = "guide_bypass_explore_hold"
                else:
                    # No route at all → probe (brief left/right pan).
                    self._state = MacroState.PROBE
                    self._begin_scan_session()
                    return self._build_guide(
                        MacroState.PROBE, goal_dir_flu, goal_dist,
                        observed_map, position_world, quaternion)

        elif state == MacroState.PROBE:
            # ── Brief left/right pan, no translation ──
            move_dir = np.array([1.0, 0.0, 0.0])
            move_dist = 0.0
            phase = int(self._scan_time / self.cfg.probe_side_duration_s) % 2
            sign = 1.0 if phase == 0 else -1.0
            pan = sign * math.radians(self.cfg.probe_side_angle_deg)
            yaw_dir = np.array([math.cos(pan), math.sin(pan)])
            reason = "probe_left" if phase == 0 else "probe_right"

        else:
            # Unknown state — hold position safely.
            move_dir = np.array([1.0, 0.0, 0.0])
            move_dist = 0.0
            yaw_dir = np.array([1.0, 0.0])
            reason = "unknown_state_hold"

        # ── Reachability check: clip move distance to known-free space ──
        if move_dist > 0.0 and needs_reachability_check:
            fitted_distance = self._fit_known_free_guide_distance(
                observed_map, position_world, move_dir, quaternion,
                move_dist)
            if fitted_distance <= 0.0:
                # Goal ray blocked.  Follow the guide line instead; if no
                # reachable advance exists, probe (brief left/right pan).
                if state == MacroState.GOAL_SEEK:
                    target, _ = self._select_guide_target(
                        goal_dir_flu, observed_map, position_world,
                        quaternion, goal_dist=goal_distance_m)
                    if target is not None:
                        self._state = MacroState.BYPASS
                        self._tick_time = 0.0
                        return self._build_guide(
                            MacroState.BYPASS, goal_dir_flu, goal_dist,
                            observed_map, position_world, quaternion)
                if self._scan_exhausted_this_tick:
                    self._scan_exhausted_this_tick = False
                    move_dist = 0.0
                    reason += "_scan_exhausted_hold"
                else:
                    self._state = MacroState.PROBE
                    self._begin_scan_session()
                    return self._build_guide(
                        MacroState.PROBE, goal_dir_flu, goal_dist,
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

        rejection_summary_parts = []
        reason_counts = {}
        for r in self._frontier_rejection_reasons.values():
            reason_counts[r] = reason_counts.get(r, 0) + 1
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
            frontier_rejection_summary=rejection_summary,
            scan_budget_exhausted=scan_budget_exhausted)
        # V15.3: remember the guide line as the lateral temporal reference
        # for the NEXT tick's search.  Only real forward intent
        # (GOAL_SEEK/BYPASS with movement) becomes a reference — probes and
        # holds clear it so a pan cannot skew the next guide line.
        if (state in (MacroState.GOAL_SEEK, MacroState.BYPASS) and
                move_dist > 0.05):
            cache = getattr(self, "_guide_cache", None)
            self._prev_guide_line_world = (
                cache.get("path") if cache is not None else None)
        else:
            self._prev_guide_line_world = None
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
