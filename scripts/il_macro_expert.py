"""Causal 5 Hz macro expert based on an observed reference path.

The expert deliberately assumes the collection scenes contain no dead ends
or concave traps.  It therefore does not try to solve general exploration.
It maintains one simple intent: travel from the vehicle to the goal, repair
that intent around *observed occupied* cells, and use UNKNOWN only as a
continuation direction.  The 30 Hz planner remains the only component that
executes a trajectory, and never enters UNKNOWN space.
"""

from __future__ import print_function, division

import math

import numpy as np

from il_common import normalize_angle


_MODE_NAMES = {
    0: "DIRECT_GUIDE", 1: "SIDE_GUIDE", 2: "OBSERVE",
    3: "GOAL_REACHED", 4: "FAILED", 5: "GOAL_APPROACH",
}


class MacroExpert(object):
    """Observed-path macro policy.

    A known obstacle on the goal ray produces left/right C++ side candidates.
    A small visible bend remains DIRECT so the local planner owns the
    avoidance.  A large bend commits a SIDE waypoint until the goal ray is
    clear again.  UNKNOWN is not an obstacle and is never a flight endpoint:
    it only makes the policy keep a goal-directed continuation or rotate its
    camera when no safe prefix is available.
    """

    def __init__(self, cfg, module, recoverability, candidate_search,
                 candidate_config):
        self.cfg = dict(cfg)
        self._module = module
        self._recoverability = recoverability
        self._candidate_search = candidate_search
        self._candidate_cfg = candidate_config
        # Privileged information is diagnostics-only and never enters this
        # observable expert action path.
        self._current_position = np.zeros(3, dtype=np.float64)
        self._current_yaw = 0.0
        self.reset()

    def reset(self):
        self.mode = self._module.MacroMode.DIRECT_GUIDE
        self.committed_side = self._module.Side.NONE
        self._side_target_world = None
        self._side_entered_s = None
        self._observe_started_s = None
        self._observe_scan_phase = 0
        self._observe_scan_reached_s = None
        self._initial_alignment_pending = True
        self._initial_alignment_yaw_reached_s = None
        self._direct_clear_ticks = 0
        self._active_observe_target_world = None
        self._active_observe_target_yaw = None
        self._active_observe_target_side = self._module.Side.NONE
        self._active_observe_target_source = ""
        self._active_observe_target_failure_ticks = 0
        self._rejected_frontier_targets = []
        self._frontier_rejection_anchor = None
        self._best_goal_distance = None
        self._last_progress_s = None
        self.direct_no_progress_time = 0.0
        self.observe_no_information_time = 0.0
        self.causal_intervention_evidence = False
        self._failed_reason = ""
        self._last_recoverability = None
        self._last_intervention = None
        self._last_blocker = None
        self._last_candidates = []
        self._last_trace = None
        self._next_blocker_id = 0
        self._blocker_signature = None
        self._blocker_track_id = -1
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.0

        # Recorder diagnostics.  These are held between macro ticks; zero
        # means that the simplified policy has no corresponding old concept.
        self.macro_chosen_side = 0
        self.side_rejection_reason = ""
        self.side_candidate_full_left = 0
        self.side_candidate_full_right = 0
        self.side_candidate_connected_left = 0
        self.side_candidate_connected_right = 0
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
        self.observe_forward_full_count = 0
        self.observe_retreat_full_count = 0
        self.observe_retreat_candidate_count = 0
        self.observe_recovery_active = 0
        self.observe_reject_unknown = 0
        self.observe_reject_endpoint_clearance = 0
        self.observe_reject_min_distance = 0
        self.observe_reject_max_distance = 0
        self.observe_reject_partial = 0
        self.observe_reject_no_path = 0
        self.observe_left_valid_count = 0
        self.observe_right_valid_count = 0
        self.observe_center_valid_count = 0
        self.observe_selected_source = ""
        self.observe_selected_side = 0
        self.observe_selected_distance = 0.0
        self.observe_selected_path_length = 0.0
        self.observe_selected_info_gain = 0.0
        self.observe_selected_clearance = 0.0

    def _build_action(self, mode, guide, desired_yaw, has_yaw, confidence,
                      reason, observe_side=None, observe_subtype=0):
        action = self._module.MacroAction()
        action.mode = mode
        action.committed_side = (
            self.committed_side if mode == self._module.MacroMode.SIDE_GUIDE
            else self._module.Side.NONE)
        action.observe_side = (self._module.Side.NONE if observe_side is None
                               else observe_side)
        action.guide_world = np.asarray(guide, dtype=np.float64)
        action.has_desired_yaw = bool(has_yaw)
        action.desired_yaw_world = float(desired_yaw) if has_yaw else 0.0
        action.confidence = float(np.clip(confidence, 0.0, 1.0))
        action.is_new_tick = True
        action.observe_subtype = int(observe_subtype)
        action.reason = reason
        action.guide_distance = float(np.linalg.norm(action.guide_world -
                                                      self._current_position))
        return action

    @staticmethod
    def _camera_yaw(direction):
        """Flightmare body-yaw for a horizontal world direction."""
        d = np.asarray(direction, dtype=np.float64)
        if np.linalg.norm(d[:2]) < 1.0e-6:
            return 0.0
        return math.atan2(d[1], d[0]) - 0.5 * math.pi

    def _vehicle_state(self, state):
        vs = self._module.VehicleState()
        vs.position = np.asarray(state["position"], dtype=np.float64)
        vs.velocity = np.asarray(state["velocity"], dtype=np.float64)
        vs.acceleration = np.asarray(state["acceleration"], dtype=np.float64)
        vs.yaw = float(state["yaw"])
        vs.yaw_rate = float(state.get("yaw_rate", 0.0))
        return vs

    def _direct_guide(self, position, goal):
        delta = goal - position
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-6:
            return goal.copy(), np.zeros(3, dtype=np.float64), distance
        direction = delta / distance
        return (position + direction * min(
            distance, float(self.cfg.get("macro_lookahead_distance_m", 4.5))),
            direction, distance)

    def _update_progress(self, goal_distance, now_s, feedback):
        epsilon = float(self.cfg.get("goal_progress_epsilon_m", 0.05))
        if self._best_goal_distance is None or \
                goal_distance < self._best_goal_distance - epsilon:
            self._best_goal_distance = goal_distance
            self._last_progress_s = now_s
        self.direct_no_progress_time = (
            0.0 if self._last_progress_s is None
            else max(0.0, now_s - self._last_progress_s))

        f = feedback or {}
        failed = int(f.get("planning_failure_count", 0))
        emergency = int(f.get("emergency_frame_count", 0))
        self.causal_intervention_evidence = bool(
            emergency > 0 or
            failed >= int(self.cfg.get("local_failure_count_for_observe", 4)) or
            self.direct_no_progress_time >=
            float(self.cfg.get("direct_intervention_timeout", 5.0)))

    def _update_blocker_track(self, blocker):
        if blocker is None or not blocker.found:
            return
        signature = int(blocker.blocker_signature)
        if signature != self._blocker_signature:
            self._next_blocker_id += 1
            self._blocker_track_id = self._next_blocker_id
            self._blocker_signature = signature

    def _side_candidates(self, candidates):
        left, right = [], []
        side_type = self._module.CandidateType.SIDE
        for candidate in candidates:
            if candidate.type != side_type or \
                    not candidate.known_reachable or \
                    not candidate.full_goal_reached:
                continue
            if candidate.side == self._module.Side.LEFT:
                left.append(candidate)
            elif candidate.side == self._module.Side.RIGHT:
                right.append(candidate)
        self.side_candidate_full_left = len(left)
        self.side_candidate_full_right = len(right)
        self.side_candidate_connected_left = len(left)
        self.side_candidate_connected_right = len(right)
        return left, right

    @staticmethod
    def _best(candidates):
        return max(candidates, key=lambda c: (
            float(c.observed_score), float(c.goal_progress),
            float(c.minimum_clearance), -float(c.observed_path_length)))

    def _bend_metrics(self, position, goal_direction, candidate):
        offset = np.asarray(candidate.position_world, dtype=np.float64) - position
        horizontal = float(np.linalg.norm(offset[:2]))
        if horizontal < 1.0e-6:
            return math.pi, 1.0
        travel = offset[:2] / horizontal
        goal_xy = goal_direction[:2]
        goal_norm = float(np.linalg.norm(goal_xy))
        if goal_norm < 1.0e-6:
            return 0.0, 0.0
        goal_xy = goal_xy / goal_norm
        dot = float(np.clip(np.dot(goal_xy, travel), -1.0, 1.0))
        turn = math.acos(dot)
        lateral = abs(goal_xy[0] * offset[1] - goal_xy[1] * offset[0])
        return turn, lateral / max(1.0, horizontal)

    def _is_low_bend(self, position, goal_direction, candidate):
        turn, lateral_ratio = self._bend_metrics(position, goal_direction,
                                                  candidate)
        return (turn <= math.radians(float(self.cfg.get(
                    "local_bend_max_deg", 25.0))) and
                lateral_ratio <= float(self.cfg.get(
                    "local_lateral_ratio_max", 0.30)))

    def _side_is_still_safe(self, observed_map):
        if self._side_target_world is None:
            return False
        return bool(observed_map.is_known_free(
            self._side_target_world,
            float(self._candidate_search.required_clearance_m())))

    def _select_side(self, left, right):
        by_side = {
            self._module.Side.LEFT: left,
            self._module.Side.RIGHT: right,
        }
        if self.committed_side in by_side:
            # A committed topology is never replaced by the opposite side
            # because one depth frame lost its candidate.  The caller either
            # keeps the still-known target or releases into OBSERVE.
            if by_side[self.committed_side]:
                return self._best(by_side[self.committed_side])
            return None
        available = left + right
        if not available:
            return None
        return self._best(available)

    def _side_candidate_action(self, selected, position, goal,
                               goal_direction, guide):
        """Turn one observed FULL-reachable side candidate into an action.

        Candidate reachability is stronger executable evidence than the
        blocker's known/unknown classification.  A low bend remains DIRECT,
        but after real local failures its concrete reachable endpoint is used
        as the recovery guide instead of repeating the failed far guide.
        """
        selected_side = selected.side
        self.macro_chosen_side = int(selected_side)
        if self._is_low_bend(position, goal_direction, selected) and \
                self.committed_side == self._module.Side.NONE:
            self.mode = self._module.MacroMode.DIRECT_GUIDE
            recovery = bool(self.causal_intervention_evidence)
            target = (np.asarray(selected.position_world, dtype=np.float64)
                      if recovery else guide)
            self._observe_started_s = None
            self._observe_scan_reached_s = None
            self._macro_decision_observable = True
            self._macro_decision_confidence = 0.90 if recovery else 0.85
            return self._build_action(
                self.mode, target, self._camera_yaw(goal_direction), True,
                self._macro_decision_confidence,
                "direct_local_recovery" if recovery else "direct_local_bend")

        self.mode = self._module.MacroMode.SIDE_GUIDE
        self._observe_started_s = None
        self._observe_scan_reached_s = None
        self.committed_side = selected_side
        self._side_target_world = np.asarray(selected.position_world,
                                              dtype=np.float64)
        if self._side_entered_s is None:
            self._side_entered_s = self._last_now_s
        desired = self._camera_yaw(goal - self._side_target_world)
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.9
        return self._build_action(
            self.mode, self._side_target_world, desired, True, 0.9,
            "side_reference_bend_%s" % (
                "left" if selected_side == self._module.Side.LEFT
                else "right"))

    def _observe_action(self, position, goal_direction, blocker, side,
                        reason):
        """Last-resort, pure camera observation with no safe motion prefix."""
        starting_observe = self._observe_started_s is None
        if starting_observe:
            self._observe_scan_phase = 0
            self._observe_scan_reached_s = None
            self.observe_rotation_exhausted = 0
        direction = goal_direction.copy()
        if blocker is not None and blocker.found:
            if side == self._module.Side.NONE:
                if blocker.left_edge_visible and not blocker.right_edge_visible:
                    side = self._module.Side.LEFT
                elif blocker.right_edge_visible and not blocker.left_edge_visible:
                    side = self._module.Side.RIGHT
            edge = (blocker.left_edge_world if side != self._module.Side.RIGHT
                    else blocker.right_edge_world)
            edge_delta = np.asarray(edge, dtype=np.float64) - position
            if np.linalg.norm(edge_delta[:2]) > 1.0e-6:
                direction = edge_delta / np.linalg.norm(edge_delta)
            yaw = self._camera_yaw(direction)
            self._observe_scan_reached_s = None
        else:
            # A goal-facing camera can still miss a grazing obstacle or the
            # free continuation around it.  Sweep three deterministic,
            # learnable goal-relative headings instead of repeatedly
            # commanding the already reached goal yaw.
            scan_angle = math.radians(float(self.cfg.get(
                "observe_scan_half_angle_deg", 35.0)))
            scan_hold_s = float(self.cfg.get("observe_scan_hold_s", 0.20))
            scan_tolerance = math.radians(float(self.cfg.get(
                "observe_scan_yaw_tolerance_deg", 6.0)))
            offsets = (scan_angle, -scan_angle, 0.0)
            scan_sides = (self._module.Side.LEFT,
                          self._module.Side.RIGHT,
                          self._module.Side.NONE)
            phase = int(self._observe_scan_phase) % len(offsets)
            yaw = normalize_angle(self._camera_yaw(direction) + offsets[phase])
            if abs(normalize_angle(self._current_yaw - yaw)) <= scan_tolerance:
                if self._observe_scan_reached_s is None:
                    self._observe_scan_reached_s = self._last_now_s
                elif self._last_now_s - self._observe_scan_reached_s >= \
                        scan_hold_s:
                    next_phase = (phase + 1) % len(offsets)
                    if phase == len(offsets) - 1:
                        self.observe_rotation_exhausted = 1
                    self._observe_scan_phase = next_phase
                    self._observe_scan_reached_s = None
                    phase = int(self._observe_scan_phase)
                    yaw = normalize_angle(
                        self._camera_yaw(direction) + offsets[phase])
            else:
                self._observe_scan_reached_s = None
            side = scan_sides[phase]
        self.mode = self._module.MacroMode.OBSERVE
        if starting_observe:
            self._observe_started_s = self._last_now_s
        self.committed_side = self._module.Side.NONE
        self._side_target_world = None
        self._clear_active_observe_target()
        self.observe_scan_side = int(side)
        self.observe_selected_side = int(side)
        self.observe_selected_source = "path_frontier"
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.55
        return self._build_action(self.mode, position, yaw, True, 0.55,
                                   reason, observe_side=side,
                                   observe_subtype=0)

    def _frontier_candidates(self, candidates):
        """Known-free, FULL-reachable endpoints immediately before UNKNOWN."""
        frontier_type = self._module.CandidateType.GOAL_FRONTIER
        return [candidate for candidate in candidates
                if candidate.type == frontier_type and
                candidate.known_reachable and candidate.full_goal_reached]

    @staticmethod
    def _frontier_distance(position, candidate):
        return float(np.linalg.norm(
            np.asarray(candidate.position_world, dtype=np.float64) - position))

    def _usable_frontiers(self, position, candidates):
        """Drop targets already rejected by the executing local planner."""
        reject_radius = max(0.25, float(
            self.cfg.get("frontier_rejection_radius_m", 0.40)))
        usable = []
        for candidate in self._frontier_candidates(candidates):
            target = np.asarray(candidate.position_world, dtype=np.float64)
            if any(np.linalg.norm(target - rejected) < reject_radius
                   for rejected in self._rejected_frontier_targets):
                continue
            usable.append(candidate)
        return usable

    @staticmethod
    def _best_frontier(position, candidates):
        """Prefer the bounded straight known-free prefix over a far frontier."""
        prefixes = [candidate for candidate in candidates
                    if str(candidate.source) == "goal_safe_prefix"]
        pool = prefixes if prefixes else candidates
        return max(pool, key=lambda candidate: (
            float(candidate.goal_progress),
            float(candidate.minimum_clearance),
            -MacroExpert._frontier_distance(position, candidate)))

    def _observe_move_failed(self, feedback):
        f = feedback or {}
        executed_frames = (int(f.get("fresh_frame_count", 0)) +
                           int(f.get("cached_frame_count", 0)))
        hard_failure = (
            int(f.get("emergency_frame_count", 0)) > 0 or
            int(f.get("planning_failure_count", 0)) >= int(
                self.cfg.get("local_failure_count_for_observe", 4)))
        # A replacement plan may fail while a validated cached suffix is
        # still executing.  That is not evidence that the macro endpoint is
        # bad and must never cause target hopping.
        return hard_failure and executed_frames == 0

    def _retire_failed_observe_target(self, position, speed_mps, feedback):
        if self._active_observe_target_world is None:
            self._active_observe_target_failure_ticks = 0
            return
        if not self._observe_move_failed(feedback):
            self._active_observe_target_failure_ticks = 0
            return
        self._active_observe_target_failure_ticks += 1
        required_ticks = max(1, int(self.cfg.get(
            "observe_target_failure_ticks", 2)))
        retire_speed = max(0.0, float(self.cfg.get(
            "observe_target_retire_max_speed_mps", 0.30)))
        # Do not replace a waypoint while the vehicle is still carrying
        # momentum from it.  Wait for persistent failure and a nearly stopped
        # state so the next label has a clean causal boundary.
        if self._active_observe_target_failure_ticks < required_ticks or \
                speed_mps > retire_speed:
            return
        self._rejected_frontier_targets.append(
            self._active_observe_target_world.copy())
        self._frontier_rejection_anchor = position.copy()
        self._clear_active_observe_target()

    def _clear_active_observe_target(self):
        """Forget an OBSERVE motion target only after a terminal event."""
        self._active_observe_target_world = None
        self._active_observe_target_yaw = None
        self._active_observe_target_side = self._module.Side.NONE
        self._active_observe_target_source = ""
        self._active_observe_target_failure_ticks = 0

    def _hold_active_observe_target(self, position, observed_map):
        """Continue a successful observed-free prefix across macro ticks.

        Candidate generation is intentionally stateless.  It can therefore
        omit a prefix for one 5 Hz tick after the map changes, even though the
        local planner is safely executing that very prefix.  The macro label
        must not turn that harmless omission into a rotate command.  Holding
        this target is safe because its endpoint is rechecked in the observed
        map, and the 30 Hz suffix validator remains the final authority.
        """
        target = self._active_observe_target_world
        if target is None:
            return None

        arrival_radius = max(
            0.10, 0.5 * float(self._candidate_cfg.min_observe_move_distance_m))
        clearance = float(self._candidate_search.required_clearance_m())
        if np.linalg.norm(target - position) <= arrival_radius or \
                not observed_map.is_known_free(target, clearance):
            self._clear_active_observe_target()
            self._observe_started_s = None
            return None

        self.mode = self._module.MacroMode.OBSERVE
        if self._observe_started_s is None:
            self._observe_started_s = self._last_now_s
        self.committed_side = self._module.Side.NONE
        self._side_target_world = None
        self.observe_scan_side = int(self._active_observe_target_side)
        self.observe_selected_side = int(self._active_observe_target_side)
        self.observe_selected_source = self._active_observe_target_source
        self.observe_selected_distance = float(np.linalg.norm(target - position))
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.75
        return self._build_action(
            self.mode, target, float(self._active_observe_target_yaw), True,
            0.75, "observe_hold_safe_prefix",
            observe_side=self._active_observe_target_side,
            observe_subtype=1)

    def _observe_move_action(self, position, candidate, reason):
        """Move only through observed free space to expose the next frontier."""
        target = np.asarray(candidate.position_world, dtype=np.float64)
        self.mode = self._module.MacroMode.OBSERVE
        if self._observe_started_s is None:
            self._observe_started_s = self._last_now_s
        self.committed_side = self._module.Side.NONE
        self._side_target_world = None
        self._active_observe_target_world = target.copy()
        self._active_observe_target_yaw = float(candidate.observation_yaw_world)
        self._active_observe_target_side = candidate.side
        self._active_observe_target_source = str(candidate.source)
        self._active_observe_target_failure_ticks = 0
        self.observe_rotation_exhausted = 0
        reject_radius = max(0.25, float(
            self.cfg.get("frontier_rejection_radius_m", 0.40)))
        self._rejected_frontier_targets = [
            rejected for rejected in self._rejected_frontier_targets
            if np.linalg.norm(target - rejected) >= reject_radius]
        self.observe_scan_side = int(candidate.side)
        self.observe_selected_side = int(candidate.side)
        self.observe_selected_source = str(candidate.source)
        self.observe_selected_distance = float(np.linalg.norm(target - position))
        self.observe_selected_path_length = float(candidate.observed_path_length)
        self.observe_selected_info_gain = float(candidate.unknown_information_gain)
        self.observe_selected_clearance = float(candidate.minimum_clearance)
        self.observe_endpoint_known_free_count = 1
        self.observe_local_full_count = 1
        self.observe_forward_full_count = 1
        self._macro_decision_observable = True
        self._macro_decision_confidence = 0.75
        return self._build_action(
            self.mode, target, float(candidate.observation_yaw_world), True,
            0.75, reason, observe_side=candidate.side, observe_subtype=1)

    def make_direct_action(self, goal_world, state):
        goal = np.asarray(goal_world, dtype=np.float64)
        position = np.asarray(state["position"], dtype=np.float64)
        _, direction, _ = self._direct_guide(position, goal)
        self._current_position = position
        self._current_yaw = float(state["yaw"])
        # The goal ray is often outside the initial depth FOV.  Rotate in
        # place first so the first observed-map decision is based on a
        # goal-facing image, never on an unobserved long DIRECT guide.
        self.mode = self._module.MacroMode.OBSERVE
        return self._build_action(
            self.mode, position, self._camera_yaw(direction), True, 0.95,
            "observe_initial_goal_alignment",
            observe_side=self._module.Side.NONE, observe_subtype=0)

    def update(self, goal_world, state, observed_map, now_s, dt_s=0.2,
               interval_feedback=None):
        goal = np.asarray(goal_world, dtype=np.float64)
        position = np.asarray(state["position"], dtype=np.float64)
        self._current_position = position
        self._current_yaw = float(state["yaw"])
        self._last_now_s = float(now_s)
        guide, goal_direction, goal_distance = self._direct_guide(position, goal)
        speed = float(np.linalg.norm(np.asarray(state["velocity"],
                                                dtype=np.float64)))

        if goal_distance <= float(self.cfg.get("goal_tolerance_m", 0.30)) and \
                speed <= float(self.cfg.get("goal_speed_tolerance_mps", 0.20)):
            self.mode = self._module.MacroMode.GOAL_REACHED
            action = self._build_action(self.mode, goal,
                                        self._camera_yaw(goal_direction), True,
                                        1.0, "goal_reached")
            self._capture_trace(action)
            return action

        # A 5 Hz macro tick can miss the brief instant where a fast drone
        # crosses the goal tolerance circle. The manager records that 30 Hz
        # event as a persistent capture latch. Once captured (or once inside
        # the approach radius), keep requesting the actual goal until the
        # position-and-speed completion condition above is met.
        feedback = interval_feedback or {}
        goal_captured = bool(feedback.get("goal_capture_latched", 0))
        base_approach_radius = max(
            float(self.cfg.get("goal_tolerance_m", 0.30)),
            float(self.cfg.get("goal_approach_radius_m", 0.80)))
        capture_deceleration = max(0.10, float(self.cfg.get(
            "goal_approach_deceleration_mps2", 2.5)))
        macro_period = max(0.0, float(dt_s))
        stopping_distance = speed * speed / (2.0 * capture_deceleration)
        approach_radius = max(
            base_approach_radius,
            float(self.cfg.get("goal_tolerance_m", 0.30)) +
            stopping_distance + speed * macro_period)
        approach_segment_clear = bool(observed_map.segment_known_and_clear(
            position, goal,
            float(self._candidate_search.required_clearance_m()), 0.05))
        if goal_captured or self.mode == self._module.MacroMode.GOAL_APPROACH \
                or (goal_distance <= approach_radius and
                    approach_segment_clear):
            self.mode = self._module.MacroMode.GOAL_APPROACH
            self.committed_side = self._module.Side.NONE
            self._side_target_world = None
            self._side_entered_s = None
            self._observe_started_s = None
            self._clear_active_observe_target()
            self._macro_decision_observable = True
            self._macro_decision_confidence = 1.0
            action = self._build_action(
                self.mode, goal, self._camera_yaw(goal_direction), True,
                1.0, "goal_approach_capture" if goal_captured else
                "goal_approach")
            self._capture_trace(action)
            return action

        if self.mode == self._module.MacroMode.OBSERVE and \
                self._observe_started_s is not None and \
                now_s - self._observe_started_s > float(self.cfg.get(
                    "macro_intervention_absolute_safety_timeout", 45.0)):
            self.mode = self._module.MacroMode.FAILED
            self._failed_reason = "observe_no_safe_prefix"
            action = self._build_action(self.mode, position, None, False,
                                        0.0, self._failed_reason)
            self._capture_trace(action)
            return action

        # One-shot initial perception alignment.  This is deliberately not
        # the obstacle-recovery OBSERVE loop: it has no lateral sweep and
        # exits as soon as goal-facing depth has been integrated.
        if self._initial_alignment_pending:
            desired_yaw = self._camera_yaw(goal_direction)
            yaw_error = abs(normalize_angle(self._current_yaw - desired_yaw))
            yaw_tolerance = math.radians(float(self.cfg.get(
                "initial_goal_alignment_tolerance_deg", 8.0)))
            min_duration_s = float(self.cfg.get(
                "initial_goal_alignment_min_duration_s", 0.20))
            map_ready = bool(observed_map.esdf_built() and
                             observed_map.known_count() > 0)
            if yaw_error <= yaw_tolerance:
                if self._initial_alignment_yaw_reached_s is None:
                    self._initial_alignment_yaw_reached_s = float(now_s)
            else:
                self._initial_alignment_yaw_reached_s = None
            if map_ready and self._initial_alignment_yaw_reached_s is not None and \
                    now_s - self._initial_alignment_yaw_reached_s >= min_duration_s:
                self._initial_alignment_pending = False
            else:
                self.mode = self._module.MacroMode.OBSERVE
                self._observe_started_s = None
                self.committed_side = self._module.Side.NONE
                self._side_target_world = None
                self._clear_active_observe_target()
                self.observe_scan_side = int(self._module.Side.NONE)
                self.observe_selected_side = int(self._module.Side.NONE)
                self.observe_selected_source = "initial_goal_alignment"
                self._macro_decision_observable = True
                self._macro_decision_confidence = 0.95
                action = self._build_action(
                    self.mode, position, desired_yaw, True, 0.95,
                    "observe_initial_goal_alignment",
                    observe_side=self._module.Side.NONE,
                    observe_subtype=0)
                self._capture_trace(action)
                return action

        vs = self._vehicle_state(state)
        # Retained solely as an observable diagnostic recorded by the data
        # writer.  It is deliberately not a macro decision gate.
        self._last_recoverability = self._recoverability.test(
            observed_map, vs, guide)
        blocker = self._module.analyze_goal_blocker(
            observed_map, vs, goal, self._candidate_cfg)
        self._last_blocker = blocker
        self._update_blocker_track(blocker)
        candidates = self._candidate_search.generate_candidates(
            observed_map, vs, goal, blocker, self._side_target_world)
        self._last_candidates = list(candidates)
        left, right = self._side_candidates(self._last_candidates)
        self._update_progress(goal_distance, now_s, interval_feedback)
        if self._frontier_rejection_anchor is not None and \
                np.linalg.norm(position - self._frontier_rejection_anchor) > \
                float(self.cfg.get("frontier_rejection_reset_distance_m", 0.50)):
            self._rejected_frontier_targets = []
            self._frontier_rejection_anchor = None
        self._retire_failed_observe_target(
            position, speed, interval_feedback)
        frontiers = self._usable_frontiers(position, self._last_candidates)
        self.observe_frontier_candidate_count = len(frontiers)
        self.macro_chosen_side = 0
        self.side_rejection_reason = ""

        selected = self._select_side(left, right)
        if not blocker.found:
            self._direct_clear_ticks += 1
            action = self._hold_active_observe_target(position, observed_map)
            scan_pending = bool(
                self.mode == self._module.MacroMode.OBSERVE and
                self._observe_started_s is not None and
                self._active_observe_target_world is None and
                not self.observe_rotation_exhausted)
            scan_finished = bool(
                self.mode == self._module.MacroMode.OBSERVE and
                self._observe_started_s is not None and
                self.observe_rotation_exhausted)
            if action is not None:
                pass
            elif (self.causal_intervention_evidence or scan_pending) and \
                    not scan_finished:
                if frontiers:
                    action = self._observe_move_action(
                        position, self._best_frontier(position, frontiers),
                        "observe_local_failure_frontier")
                else:
                    action = self._observe_action(
                        position, goal_direction, blocker,
                        self._module.Side.NONE,
                        "observe_local_failure_scan")
            else:
                if self._direct_clear_ticks >= int(self.cfg.get(
                        "side_release_clear_ticks", 2)):
                    self.committed_side = self._module.Side.NONE
                    self._side_target_world = None
                    self._side_entered_s = None
                self._observe_started_s = None
                self._observe_scan_reached_s = None
                self.observe_rotation_exhausted = 0
                self._clear_active_observe_target()
                self._rejected_frontier_targets = []
                self._frontier_rejection_anchor = None
                self.mode = self._module.MacroMode.DIRECT_GUIDE
                self._macro_decision_observable = True
                self._macro_decision_confidence = 0.95
                action = self._build_action(
                    self.mode, guide, self._camera_yaw(goal_direction), True,
                    0.95, "direct_clear_path")
            self._capture_trace(action)
            return action

        self._direct_clear_ticks = 0

        # UNKNOWN on the reference ray is a frontier, not a collision.  Keep
        # direct guidance while the first depth frames build a usable map.
        # Afterwards, a failure can advance only to a known-free, FULL-
        # reachable frontier prefix.  Rotation is the final fallback.
        if not blocker.blocked_by_known:
            startup_warmup = float(self.cfg.get("initial_map_warmup_s", 1.0))
            action = self._hold_active_observe_target(position, observed_map)
            scan_pending = bool(
                self.mode == self._module.MacroMode.OBSERVE and
                self._observe_started_s is not None and
                self._active_observe_target_world is None and
                not self.observe_rotation_exhausted)
            scan_finished = bool(
                self.mode == self._module.MacroMode.OBSERVE and
                self._observe_started_s is not None and
                self.observe_rotation_exhausted)
            if action is not None:
                pass
            elif selected is not None and self.causal_intervention_evidence:
                action = self._side_candidate_action(
                    selected, position, goal, goal_direction, guide)
            elif now_s < startup_warmup or (
                    not self.causal_intervention_evidence and
                    not scan_pending) or scan_finished:
                self.mode = self._module.MacroMode.DIRECT_GUIDE
                self._observe_started_s = None
                self._observe_scan_reached_s = None
                self.observe_rotation_exhausted = 0
                self._clear_active_observe_target()
                self._macro_decision_observable = True
                self._macro_decision_confidence = 0.8
                action = self._build_action(
                    self.mode, guide, self._camera_yaw(goal_direction), True,
                    0.8, "direct_frontier_continuation")
            else:
                if frontiers:
                    action = self._observe_move_action(
                        position, self._best_frontier(position, frontiers),
                        "observe_frontier_safe_prefix")
                else:
                    action = self._observe_action(
                        position, goal_direction, blocker,
                        self._module.Side.NONE,
                        "observe_frontier_no_safe_prefix")
            self._capture_trace(action)
            return action

        # A known occupied component intersects the reference path.  C++
        # candidates are known-free and FULL reachable; the macro decides
        # only whether their bend is local or strategic.
        if selected is None:
            # If the previous strategic endpoint is still observed free,
            # keep that commitment instead of replacing it after one noisy
            # depth frame.  Otherwise obtain information toward the more
            # visible edge; no global map is consulted.
            if self.committed_side != self._module.Side.NONE and \
                    self._side_is_still_safe(observed_map):
                side = self.committed_side
                target = self._side_target_world
                direction = target - position
                self.mode = self._module.MacroMode.SIDE_GUIDE
                self.macro_chosen_side = int(side)
                self._macro_decision_observable = True
                self._macro_decision_confidence = 0.65
                action = self._build_action(
                    self.mode, target, self._camera_yaw(direction), True,
                    0.65, "side_hold_observed_target")
            else:
                side = (self.committed_side if
                        self.committed_side != self._module.Side.NONE else
                        self._module.Side.LEFT)
                action = self._observe_action(position, goal_direction,
                                               blocker, side,
                                               "observe_wait_for_side_prefix")
            self._capture_trace(action)
            return action

        action = self._side_candidate_action(
            selected, position, goal, goal_direction, guide)
        self._capture_trace(action)
        return action

    def _capture_trace(self, action):
        blocker = self._last_blocker
        self._last_trace = {
            "mode": _MODE_NAMES.get(int(action.mode), str(int(action.mode))),
            "reason": action.reason,
            "guide_world": [round(float(v), 4) for v in action.guide_world],
            "committed_side": int(action.committed_side),
            "observe_side": int(action.observe_side),
            "blocker_found": int(blocker.found) if blocker is not None else 0,
            "blocker_known": int(blocker.blocked_by_known)
            if blocker is not None else 0,
            "candidate_count": len(self._last_candidates),
            "local_failure_evidence": int(self.causal_intervention_evidence),
        }

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
        return self._macro_decision_observable

    @property
    def macro_decision_confidence(self):
        return self._macro_decision_confidence

    @property
    def blocker_track_id(self):
        return self._blocker_track_id

    @property
    def last_trace(self):
        return self._last_trace
