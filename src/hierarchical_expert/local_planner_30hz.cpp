#include "il_dataset/hierarchical_expert/local_planner_30hz.hpp"

#include "il_dataset/hierarchical_expert/kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>

namespace il_dataset {
namespace expert {

// ────────────────────────────────────────────────────────────────────
//  Candidate forward simulation — SAME shared kinematics as the
//  preflight simulator (prediction never diverges from execution).
// ────────────────────────────────────────────────────────────────────
static Trajectory2D simulateCandidate(const VehicleState2D& st,
                                      const BodyCommand2D& cmd,
                                      const Params2D& p) {
    Trajectory2D traj;
    traj.points.push_back(st.position);
    traj.yaw.push_back(st.yaw);
    traj.t.push_back(0.0);

    VehicleState2D s = st;
    for (double t = p.lp_dt; t <= p.lp_horizon_s + 1e-6; t += p.lp_dt) {
        s = integrateKinematicStep(s, cmd, p.lp_dt, p);
        traj.points.push_back(s.position);
        traj.yaw.push_back(s.yaw);
        traj.t.push_back(t);
    }
    traj.valid = true;
    return traj;
}

uint64_t LocalPlanner30Hz::commandTieHash(double vx, double vy, double yr) {
    uint64_t h = 1469598103934665603ULL;  // FNV offset basis
    auto mix = [&h](uint64_t word) {
        for (int i = 0; i < 8; ++i) {
            h ^= (word >> (8 * i)) & 0xFFULL;
            h *= 1099511628211ULL;
        }
    };
    uint64_t a = 0, b = 0, c = 0;
    std::memcpy(&a, &vx, sizeof(a));
    std::memcpy(&b, &vy, sizeof(b));
    std::memcpy(&c, &yr, sizeof(c));
    mix(a);
    mix(b);
    mix(c);
    return h;
}

void LocalPlanner30Hz::reset() {
    turn_hysteresis_active_ = false;
    current_trajectory_ = Trajectory2D{};
    last_command_ = BodyCommand2D{};
    has_last_command_ = false;
    last_mission_revision_ = 0;
    last_target_position_ = Vec2d(0.0, 0.0);
    last_target_valid_ = false;
    limit_cycle_window_.clear();
    limit_cycle_detected_ = false;
    last_cycle_mission_revision_ = 0;
}

// ────────────────────────────────────────────────────────────────────
//  Shared output semantic: one-control-period reachable command
// ────────────────────────────────────────────────────────────────────
BodyCommand2D LocalPlanner30Hz::reachableCommand(
    const VehicleState2D& state, const BodyCommand2D& intent) const {
    const Vec2d current_v_body = bodyVelocity(state);
    const double dt = (p_.lp_control_period_s > 0.0)
                          ? p_.lp_control_period_s
                          : (1.0 / 30.0);
    const double dv = p_.lp_max_accel * dt;
    const double dyr = p_.lp_max_yaw_accel * dt;
    BodyCommand2D out;
    out.vx_body = clamp(intent.vx_body, current_v_body.x() - dv,
                        current_v_body.x() + dv);
    out.vy_body = clamp(intent.vy_body, current_v_body.y() - dv,
                        current_v_body.y() + dv);
    const double spd = std::hypot(out.vx_body, out.vy_body);
    if (spd > p_.lp_max_speed && spd > 1e-9) {
        out.vx_body *= p_.lp_max_speed / spd;
        out.vy_body *= p_.lp_max_speed / spd;
    }
    out.yaw_rate = clamp(intent.yaw_rate, state.yaw_rate - dyr,
                         state.yaw_rate + dyr);
    out.yaw_rate = clamp(out.yaw_rate, -p_.lp_max_yaw_rate,
                         p_.lp_max_yaw_rate);
    return out;
}

Vec2d LocalPlanner30Hz::bodyVelocity(const VehicleState2D& state) const {
    const double c = std::cos(state.yaw), sn = std::sin(state.yaw);
    return Vec2d(c * state.velocity_world.x() + sn * state.velocity_world.y(),
                 -sn * state.velocity_world.x() + c * state.velocity_world.y());
}

// ────────────────────────────────────────────────────────────────────
//  Per-mission bookkeeping (v5)
// ────────────────────────────────────────────────────────────────────
bool LocalPlanner30Hz::updateMissionState(const LocalTarget& target,
                                          PlannerResult& res) {
    bool reset_memory = false;
    if (target.mission_revision != last_mission_revision_) reset_memory = true;
    if (target.valid && last_target_valid_ &&
        (target.position - last_target_position_).norm() >
            p_.lp_target_discontinuity_reset_m) {
        reset_memory = true;
        res.target_discontinuity_reset = true;
    }
    if (reset_memory) {
        last_command_ = BodyCommand2D{};
        has_last_command_ = false;
        limit_cycle_window_.clear();
        limit_cycle_detected_ = false;
        last_cycle_mission_revision_ = target.mission_revision;
    }
    last_mission_revision_ = target.mission_revision;
    last_target_position_ = target.position;
    last_target_valid_ = target.valid;
    return reset_memory;
}

BodyCommand2D LocalPlanner30Hz::terminalIntent(
    const VehicleState2D& state, const LocalTarget& target) const {
    BodyCommand2D cmd;
    const Vec2d to = target.position - state.position;
    const double dist = to.norm();
    if (dist <= 1e-9) return cmd;

    const double capture_radius = 0.5 * p_.task_goal_tolerance;
    const double remaining = std::max(0.0, dist - capture_radius);
    const double proportional = p_.lp_terminal_speed_gain * remaining;
    const double braking =
        std::sqrt(std::max(0.0, 2.0 * p_.lp_max_accel * remaining));
    const double desired_speed =
        std::min(p_.lp_terminal_max_speed, std::min(proportional, braking));

    const Vec2d desired_world = (to / dist) * desired_speed;
    const double c = std::cos(state.yaw), sn = std::sin(state.yaw);
    cmd.vx_body = c * desired_world.x() + sn * desired_world.y();
    cmd.vy_body = -sn * desired_world.x() + c * desired_world.y();

    if (dist > p_.task_goal_tolerance) {
        const double bearing =
            wrapAngle(std::atan2(to.y(), to.x()) - state.yaw);
        cmd.yaw_rate = clamp(p_.lp_turn_k * bearing,
                             -p_.lp_terminal_max_yaw_rate,
                             p_.lp_terminal_max_yaw_rate);
    }
    return cmd;
}

LocalPlannerCandidate LocalPlanner30Hz::makeTerminalCandidate(
    const VehicleState2D& state, const LocalTarget& target) const {
    LocalPlannerCandidate candidate;
    const BodyCommand2D intent = terminalIntent(state, target);
    const BodyCommand2D first = reachableCommand(state, intent);
    candidate.desired_vx_body = intent.vx_body;
    candidate.desired_vy_body = intent.vy_body;
    candidate.desired_yaw_rate = intent.yaw_rate;
    candidate.vx_body = first.vx_body;
    candidate.vy_body = first.vy_body;
    candidate.yaw_rate = first.yaw_rate;
    candidate.stable_index = -1;

    // Roll out the same feedback law that will be recomputed at 30 Hz.
    Trajectory2D traj;
    traj.points.push_back(state.position);
    traj.yaw.push_back(state.yaw);
    traj.t.push_back(0.0);
    VehicleState2D predicted = state;
    for (double t = p_.lp_dt; t <= p_.lp_horizon_s + 1e-6;
         t += p_.lp_dt) {
        predicted = integrateKinematicStep(
            predicted, terminalIntent(predicted, target), p_.lp_dt, p_);
        traj.points.push_back(predicted.position);
        traj.yaw.push_back(predicted.yaw);
        traj.t.push_back(t);
    }
    traj.valid = true;
    candidate.traj = std::move(traj);
    return candidate;
}

// ────────────────────────────────────────────────────────────────────
//  Core planning (mutating or preview)
// ────────────────────────────────────────────────────────────────────
PlannerResult LocalPlanner30Hz::computePlan(const VehicleState2D& state,
                                            const LocalObservation& obs,
                                            const LocalTarget& target,
                                            bool mutate) {
    PlannerResult res;
    res.failure_reason = FailureReason::NONE;
    res.target_mission_revision = target.mission_revision;
    res.handoff_clearance_m = handoffClearance();

    if (!target.valid) {
        res.failure_reason = FailureReason::TARGET_OUTSIDE_FOV;
        res.planner_status = PlannerStatus::NO_TARGET;
        if (mutate) {
            last_command_ = BodyCommand2D{};
            has_last_command_ = true;
            updateMissionState(target, res);
        }
        return res;
    }

    if (mutate) {
        updateMissionState(target, res);
    }

    const Vec2d to = target.position - state.position;
    const double bearing = wrapAngle(std::atan2(to.y(), to.x()) - state.yaw);
    res.target_bearing_error_deg = rad2deg(std::fabs(bearing));
    const double dist = to.norm();
    res.local_target_distance = dist;
    const double enter = (dist <= p_.lp_near_goal_heading_relax_distance)
                             ? deg2rad(p_.lp_near_goal_turn_enter_deg)
                             : deg2rad(p_.lp_turn_enter_deg);
    const double exit_ang = deg2rad(p_.lp_turn_exit_deg);

    // normalized_distance == 1 is the public direction+distance contract's
    // reserved pure-rotation command.
    const bool pure_rotation_target =
        target.normalized_distance >= 1.0 - 1e-9;
    if (pure_rotation_target) {
        const bool bearing_pending = std::fabs(bearing) > exit_ang;
        const bool yaw_rate_pending =
            std::fabs(state.yaw_rate) > p_.lp_turn_exit_max_yaw_rate;
        const bool rotation_pending = bearing_pending || yaw_rate_pending;
        turn_hysteresis_active_ = rotation_pending;

        const double yaw_intent = bearing_pending
            ? clamp(p_.lp_turn_k * bearing, -p_.lp_max_yaw_rate,
                    p_.lp_max_yaw_rate)
            : 0.0;
        const BodyCommand2D intent{0.0, 0.0, yaw_intent};
        const BodyCommand2D out = reachableCommand(state, intent);
        res.success = true;
        res.turn_mode = rotation_pending;
        res.intent_vx_body = intent.vx_body;
        res.intent_vy_body = intent.vy_body;
        res.intent_yaw_rate = intent.yaw_rate;
        res.vx_body = out.vx_body;
        res.vy_body = out.vy_body;
        res.yaw_rate = out.yaw_rate;
        res.selected_output_speed_mps = std::hypot(out.vx_body, out.vy_body);
        res.planner_status = rotation_pending ? PlannerStatus::TURNING
                                              : PlannerStatus::SAFE_HOLD;
        res.candidate_progress_qualified = false;
        res.output_progress_qualified = false;
        res.progress_qualified = false;
        res.stationary_candidate_selected =
            res.selected_output_speed_mps <= p_.lp_min_progress_speed_mps;
        res.stationary_selection_reason = rotation_pending
            ? "distance_one_pure_rotation"
            : "distance_one_waiting_for_5hz";
        res.failure_reason = FailureReason::NONE;
        if (mutate) {
            current_trajectory_ = Trajectory2D{};
            last_command_ = out;
            has_last_command_ = true;
            limit_cycle_window_.clear();
            limit_cycle_detected_ = false;
            last_cycle_mission_revision_ = target.mission_revision;
        }
        return res;
    }

    // TURN hysteresis.  Exit requires BOTH a small remaining bearing AND a
    // small ACTUAL yaw rate.
    if (dist <= p_.task_goal_tolerance) {
        turn_hysteresis_active_ = false;
    } else if (turn_hysteresis_active_) {
        const bool bearing_ok = std::fabs(bearing) <= exit_ang;
        const bool yaw_rate_ok =
            std::fabs(state.yaw_rate) <= p_.lp_turn_exit_max_yaw_rate;
        if (bearing_ok && yaw_rate_ok) turn_hysteresis_active_ = false;
    } else {
        if (std::fabs(bearing) > enter) turn_hysteresis_active_ = true;
    }

    if (turn_hysteresis_active_) {
        res.success = true;
        res.turn_mode = true;
        const BodyCommand2D intent{
            0.0, 0.0,
            clamp(p_.lp_turn_k * bearing, -p_.lp_max_yaw_rate,
                  p_.lp_max_yaw_rate)};
        const BodyCommand2D out = reachableCommand(state, intent);
        res.intent_vx_body = intent.vx_body;
        res.intent_vy_body = intent.vy_body;
        res.intent_yaw_rate = intent.yaw_rate;
        res.vx_body = out.vx_body;
        res.vy_body = out.vy_body;
        res.yaw_rate = out.yaw_rate;
        res.selected_output_speed_mps = std::hypot(out.vx_body, out.vy_body);
        res.planner_status = PlannerStatus::TURNING;
        res.candidate_progress_qualified = false;
        res.output_progress_qualified = false;
        res.progress_qualified = false;
        res.stationary_candidate_selected =
            res.selected_output_speed_mps <= p_.lp_min_progress_speed_mps;
        res.stationary_selection_reason =
            res.stationary_candidate_selected ? "turn_mode" : "";
        res.failure_reason = FailureReason::NONE;
        if (mutate) {
            current_trajectory_ = Trajectory2D{};
            last_command_ = out;
            has_last_command_ = true;
            limit_cycle_window_.clear();
            limit_cycle_detected_ = false;
            last_cycle_mission_revision_ = target.mission_revision;
        }
        return res;
    }

    res.turn_mode = false;

    // 1) Previously executed trajectory cut by a newly observed obstacle?
    if (mutate) {
        bool dynamic_violation = false;
        if (currentTrajectoryBlocked(state, obs, dynamic_violation)) {
            res.immediate_avoidance = true;
            res.blocked_observed = true;
            res.dynamic_clearance_blocked =
                res.dynamic_clearance_blocked || dynamic_violation;
        }
    }

    // ── Local corridor assessment (v7) ─────────────────────────────
    {
        bool corridor_blocked = false;
        double first_dist = std::numeric_limits<double>::quiet_NaN();
        assessLocalCorridor(state, obs, target, corridor_blocked, first_dist);
        res.local_corridor_blocked = corridor_blocked;
        res.first_blocking_obstacle_distance = first_dist;
    }

    auto countRejected = [&](CandidateRejectReason reject_enum) {
        switch (reject_enum) {
            case CandidateRejectReason::NOT_KNOWN_FREE:
                ++res.reject_not_known_free;
                break;
            case CandidateRejectReason::OUTSIDE_CURRENT_FOV:
                ++res.reject_outside_current_fov;
                break;
            case CandidateRejectReason::OBSERVED_CLEARANCE_TOO_SMALL:
                ++res.reject_observed_clearance_too_small;
                break;
            case CandidateRejectReason::NO_PROGRESS:
                ++res.reject_no_progress;
                break;
            case CandidateRejectReason::INSUFFICIENT_BRAKING_CLEARANCE:
                ++res.reject_insufficient_braking_clearance;
                break;
            default:
                ++res.reject_other;
                break;
        }
    };

    auto acceptCandidate = [&](const LocalPlannerCandidate& selected,
                               PlannerStatus status,
                               const std::string& stationary_reason) {
        res.success = true;
        res.planner_status = status;
        res.vx_body = selected.vx_body;
        res.vy_body = selected.vy_body;
        res.yaw_rate = selected.yaw_rate;
        res.intent_vx_body = selected.desired_vx_body;
        res.intent_vy_body = selected.desired_vy_body;
        res.intent_yaw_rate = selected.desired_yaw_rate;
        res.selected_output_speed_mps =
            std::hypot(selected.vx_body, selected.vy_body);
        res.selected = selected.traj;
        res.min_observed_clearance = selected.min_clearance;
        res.selected_soft_min_clearance_m = selected.soft_min_clearance;
        res.selected_dynamic_required_clearance_m =
            selected.max_dynamic_required_clearance;
        res.selected_closing_speed_mps = selected.max_closing_speed;
        res.nominal_progress_m = selected.nominal_progress_m;
        res.executable_progress_m = selected.executable_progress_m;
        res.safe_prefix_duration_s = selected.safe_prefix_duration_s;
        res.candidate_progress_qualified = selected.progress_qualified;
        res.output_progress_qualified = selected.progress_qualified;
        res.progress_qualified = res.output_progress_qualified;
        res.stationary_candidate_selected = selected.stationary;
        res.stationary_selection_reason =
            selected.stationary ? stationary_reason : "";
        res.failure_reason = FailureReason::NONE;
        res.selected_terminal_heading_error_deg =
            rad2deg(selected.terminal_heading_error_rad);
        res.selected_velocity_alignment_error_deg =
            rad2deg(selected.velocity_alignment_error_rad);
        res.selected_cross_track_error_m = selected.cross_track_error_m;
        res.selected_cost_total = selected.cost;
        res.selected_cost_progress = selected.cost_progress;
        res.selected_cost_clearance = selected.cost_clearance;
        res.selected_cost_smoothness = selected.cost_smoothness;
        res.selected_cost_speed_change = selected.cost_speed_change;
        res.selected_cost_yaw_rate_change = selected.cost_yaw_rate_change;
        res.selected_cost_terminal_heading = selected.cost_terminal_heading;
        res.selected_cost_velocity_alignment = selected.cost_velocity_alignment;
        res.selected_cost_cross_track = selected.cost_cross_track;
        res.selected_cost_obstacle_risk = selected.obstacle_risk_cost;
        res.predicted_closest_clearance = selected.predicted_closest_clearance;
        res.time_to_collision = selected.time_to_collision;
        res.obstacle_risk_cost = selected.obstacle_risk_cost;
        res.avoidance_strength = selected.avoidance_strength;
        res.avoidance_active = selected.avoidance_active;
        if (mutate) {
            current_trajectory_ = selected.traj;
            last_command_ = BodyCommand2D{selected.vx_body, selected.vy_body,
                                          selected.yaw_rate};
            has_last_command_ = true;
        }
    };

    // Collect the full spatial risk neighbourhood ONCE per 30 Hz tick.
    std::vector<Vec2d> risk_occ_cells;
    collectRiskOccupiedCells(state, obs, risk_occ_cells);

    // Final-goal feedback is attempted before the coarse velocity lattice.
    if (dist <= p_.lp_terminal_control_distance) {
        LocalPlannerCandidate terminal = makeTerminalCandidate(state, target);
        std::string reason;
        CandidateRejectReason reject_enum = CandidateRejectReason::NONE;
        if (evaluateCandidate(terminal, state, obs, target, risk_occ_cells, reason,
                              reject_enum)) {
            acceptCandidate(terminal, PlannerStatus::TERMINAL_SETTLING,
                            terminal.stationary ? "terminal_settling" : "");
            return res;
        }
        terminal.reject_reason = reason;
        countRejected(reject_enum);
        res.rejected_candidates.push_back(terminal.traj);
    }

    // ── Terminal convergence region (v5/v7) ────────────────────────
    const bool terminal_region = dist <= p_.lp_terminal_control_distance;

    // 2) Generate + evaluate candidates.
    auto candidates = generateCandidates(state);
    res.dynamic_window_candidate_count =
        static_cast<uint32_t>(candidates.size());
    const LocalPlannerCandidate* best_progressing = nullptr;
    const LocalPlannerCandidate* best_safe = nullptr;

    auto betterThan = [&](const LocalPlannerCandidate& a,
                          const LocalPlannerCandidate* b) {
        if (b == nullptr) return true;
        if (a.cost < b->cost - p_.lp_cost_tie_tolerance) return true;
        if (a.cost > b->cost + p_.lp_cost_tie_tolerance) return false;
        if (a.terminal_heading_error_rad < b->terminal_heading_error_rad)
            return true;
        if (a.terminal_heading_error_rad > b->terminal_heading_error_rad)
            return false;
        if (a.cross_track_error_m < b->cross_track_error_m) return true;
        if (a.cross_track_error_m > b->cross_track_error_m) return false;
        const double av = std::fabs(a.vy_body), bv = std::fabs(b->vy_body);
        if (av < bv) return true;
        if (av > bv) return false;
        const double ay = std::fabs(a.yaw_rate), by = std::fabs(b->yaw_rate);
        if (ay < by) return true;
        if (ay > by) return false;
        const double ac = std::fabs(a.yaw_rate - state.yaw_rate);
        const double bc = std::fabs(b->yaw_rate - state.yaw_rate);
        if (ac < bc) return true;
        if (ac > bc) return false;
        if (has_last_command_ &&
            last_mission_revision_ == target.mission_revision) {
            const Vec2d previous(last_command_.vx_body,
                                 last_command_.vy_body);
            const double alc =
                (Vec2d(a.vx_body, a.vy_body) - previous).norm();
            const double blc =
                (Vec2d(b->vx_body, b->vy_body) - previous).norm();
            if (alc < blc) return true;
            if (alc > blc) return false;
        }
        return a.tie_hash < b->tie_hash;
    };

    for (auto& c : candidates) {
        std::string reason;
        CandidateRejectReason reject_enum = CandidateRejectReason::NONE;
        const bool ok =
            evaluateCandidate(c, state, obs, target, risk_occ_cells, reason,
                              reject_enum);
        c.reject_reason = reason;
        if (!ok) {
            countRejected(reject_enum);
            res.rejected_candidates.push_back(c.traj);
            continue;
        }
        if (betterThan(c, best_safe)) best_safe = &c;
        if (c.progress_qualified && betterThan(c, best_progressing)) {
            best_progressing = &c;
        }
    }

    // ── Hierarchical selection (v5) ────────────────────────────────
    if (best_progressing || (terminal_region && best_safe)) {
        const LocalPlannerCandidate* sel =
            terminal_region ? (best_progressing ? best_progressing : best_safe)
                            : best_progressing;
        const PlannerStatus status =
            terminal_region ? PlannerStatus::TERMINAL_SETTLING
                            : PlannerStatus::SAFE_PROGRESSING;
        std::string stationary_reason;
        if (sel->stationary) {
            stationary_reason =
                terminal_region ? "terminal_settling" : "safe_progressing";
        }
        acceptCandidate(*sel, status, stationary_reason);
        if (mutate) {
            const bool cycle = updateLimitCycle(res, target, state);
            res.local_limit_cycle_detected = cycle;
            if (cycle) {
                res.success = false;
                const BodyCommand2D brake =
                    reachableCommand(state, BodyCommand2D{0.0, 0.0, 0.0});
                res.vx_body = brake.vx_body;
                res.vy_body = brake.vy_body;
                res.yaw_rate = brake.yaw_rate;
                res.selected_output_speed_mps =
                    std::hypot(brake.vx_body, brake.vy_body);
                res.output_progress_qualified = false;
                res.progress_qualified = false;
                res.selected = Trajectory2D{};
                res.selected_soft_min_clearance_m =
                    std::numeric_limits<double>::quiet_NaN();
                res.selected_dynamic_required_clearance_m =
                    std::numeric_limits<double>::quiet_NaN();
                res.selected_closing_speed_mps =
                    std::numeric_limits<double>::quiet_NaN();
                res.failure_reason =
                    FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE;
                res.blocked_observed = true;
                res.planner_status =
                    PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE;
                res.stationary_candidate_selected = true;
                res.stationary_selection_reason = "limit_cycle_override";
                current_trajectory_ = Trajectory2D{};
                last_command_ = brake;
                has_last_command_ = true;
            }
        }
        return res;
    }

    // 3) Failure: classify + safe braking.
    res.success = false;
    const bool observed_block = res.reject_observed_clearance_too_small > 0;
    const bool dynamic_block = res.reject_insufficient_braking_clearance > 0;
    const bool has_block_evidence =
        observed_block || dynamic_block ||
        corridorBlockedByObserved(state, obs, target);
    res.dynamic_clearance_blocked =
        res.dynamic_clearance_blocked || dynamic_block;

    if (best_safe) {
        if (has_block_evidence) {
            res.failure_reason = FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE;
            res.blocked_observed = true;
            res.planner_status = PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE;
        } else {
            res.failure_reason = FailureReason::STALLED_WITHOUT_PROGRESS;
            res.planner_status = PlannerStatus::STALLED_WITHOUT_PROGRESS;
        }
        res.vx_body = best_safe->vx_body;
        res.vy_body = best_safe->vy_body;
        res.yaw_rate = best_safe->yaw_rate;
        res.intent_vx_body = best_safe->desired_vx_body;
        res.intent_vy_body = best_safe->desired_vy_body;
        res.intent_yaw_rate = best_safe->desired_yaw_rate;
        res.selected_output_speed_mps =
            std::hypot(best_safe->vx_body, best_safe->vy_body);
        res.selected = best_safe->traj;
        res.min_observed_clearance = best_safe->min_clearance;
        res.selected_soft_min_clearance_m = best_safe->soft_min_clearance;
        res.selected_dynamic_required_clearance_m =
            best_safe->max_dynamic_required_clearance;
        res.selected_closing_speed_mps = best_safe->max_closing_speed;
        res.nominal_progress_m = best_safe->nominal_progress_m;
        res.executable_progress_m = best_safe->executable_progress_m;
        res.safe_prefix_duration_s = best_safe->safe_prefix_duration_s;
        res.candidate_progress_qualified = false;
        res.output_progress_qualified = false;
        res.progress_qualified = false;
        res.stationary_candidate_selected = best_safe->stationary;
        res.stationary_selection_reason =
            best_safe->stationary ? "stalled_without_progress" : "";
        res.selected_cost_obstacle_risk = best_safe->obstacle_risk_cost;
        res.predicted_closest_clearance = best_safe->predicted_closest_clearance;
        res.time_to_collision = best_safe->time_to_collision;
        res.obstacle_risk_cost = best_safe->obstacle_risk_cost;
        res.avoidance_strength = best_safe->avoidance_strength;
        res.avoidance_active = best_safe->avoidance_active;
    } else {
        // No feasible candidate at all.
        if (has_block_evidence) {
            res.planner_status = PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE;
            res.failure_reason = FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE;
            res.blocked_observed = true;
        } else {
            res.planner_status = PlannerStatus::NO_SAFE_CANDIDATE;
            res.failure_reason = FailureReason::NO_SAFE_CANDIDATE;
        }
        const BodyCommand2D brake =
            reachableCommand(state, BodyCommand2D{0.0, 0.0, 0.0});
        res.vx_body = brake.vx_body;
        res.vy_body = brake.vy_body;
        res.yaw_rate = brake.yaw_rate;
        res.selected_output_speed_mps =
            std::hypot(brake.vx_body, brake.vy_body);
        res.stationary_candidate_selected =
            res.selected_output_speed_mps <= p_.lp_min_progress_speed_mps;
        res.stationary_selection_reason = "no_safe_motion_candidate";
    }

    // Emergency brake: stopping distance not available.
    if (!spaceToStop(state, obs, stoppingDistance(state))) {
        res.emergency_brake = true;
        res.planner_status = PlannerStatus::EMERGENCY_BRAKE;
        const BodyCommand2D brake =
            reachableCommand(state, BodyCommand2D{0.0, 0.0, 0.0});
        res.vx_body = brake.vx_body;
        res.vy_body = brake.vy_body;
        res.yaw_rate = brake.yaw_rate;
        res.selected_output_speed_mps =
            std::hypot(brake.vx_body, brake.vy_body);
        res.stationary_candidate_selected =
            res.selected_output_speed_mps <= p_.lp_min_progress_speed_mps;
        res.stationary_selection_reason = "emergency_brake";
    }
    if (mutate) {
        current_trajectory_ = best_safe ? best_safe->traj : Trajectory2D{};
        last_command_ = BodyCommand2D{res.vx_body, res.vy_body, res.yaw_rate};
        has_last_command_ = true;
        if (updateLimitCycle(res, target, state)) {
            res.local_limit_cycle_detected = true;
            if (res.failure_reason !=
                FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE) {
                res.failure_reason = FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE;
                res.blocked_observed = true;
                res.planner_status = PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE;
            }
        }
    }
    return res;
}

PlannerResult LocalPlanner30Hz::plan(const VehicleState2D& state,
                                     const LocalObservation& obs,
                                     const LocalTarget& target) {
    return computePlan(state, obs, target, /*mutate=*/true);
}

PreviewResult LocalPlanner30Hz::previewPlan(const VehicleState2D& state,
                                            const LocalObservation& obs,
                                            const LocalTarget& target) {
    const bool saved_hyst = turn_hysteresis_active_;
    const auto res = computePlan(state, obs, target, /*mutate=*/false);
    turn_hysteresis_active_ = saved_hyst;
    PreviewResult pv;
    pv.success = res.success;
    pv.turn_mode = res.turn_mode;
    pv.emergency_brake = res.emergency_brake;
    pv.failure_reason = res.failure_reason;
    pv.planner_status = res.planner_status;
    pv.has_progressing_trajectory =
        res.success && res.progress_qualified &&
        (res.planner_status == PlannerStatus::SAFE_PROGRESSING ||
         res.planner_status == PlannerStatus::TERMINAL_SETTLING);
    pv.executable_progress_m = res.executable_progress_m;
    pv.safe_prefix_duration_s = res.safe_prefix_duration_s;
    pv.selected_output_speed_mps = res.selected_output_speed_mps;
    return pv;
}

bool LocalPlanner30Hz::canBrakeSafely(const VehicleState2D& state,
                                      const LocalObservation& obs) const {
    return spaceToStop(state, obs, stoppingDistance(state));
}

// ────────────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────────────
bool LocalPlanner30Hz::currentTrajectoryBlocked(
    const VehicleState2D& state, const LocalObservation& obs,
    bool& dynamic_violation) const {
    dynamic_violation = false;
    if (!current_trajectory_.valid || current_trajectory_.points.empty())
        return false;
    size_t start_i = 0;
    double best = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < current_trajectory_.points.size(); ++i) {
        const double d =
            (current_trajectory_.points[i] - state.position).squaredNorm();
        if (d < best) {
            best = d;
            start_i = i;
        }
    }
    const double step = std::max(1e-3, 0.5 * p_.obs_resolution);
    for (size_t i = start_i + 1; i < current_trajectory_.points.size(); ++i) {
        const Vec2d& a = current_trajectory_.points[i - 1];
        const Vec2d& b = current_trajectory_.points[i];
        double dt_seg = p_.lp_dt;
        if (i < current_trajectory_.t.size() &&
            i - 1 < current_trajectory_.t.size()) {
            const double dt = current_trajectory_.t[i] -
                              current_trajectory_.t[i - 1];
            if (dt > 1e-9) dt_seg = dt;
        }
        const double seg = (b - a).norm();
        const Vec2d v_seg =
            (seg > 1e-9) ? (b - a) / dt_seg : Vec2d(0.0, 0.0);
        const double search_r = clearanceSearchRadius(v_seg.norm());
        const int steps = std::max(1, static_cast<int>(std::ceil(seg / step)));
        for (int k = 0; k <= steps; ++k) {
            const Vec2d p = a + (b - a) * (static_cast<double>(k) / steps);
            bool hard_violation = false;
            bool envelope_violation = false;
            obs.forEachOccupiedWithin(
                p, search_r, [&](const Vec2d& centre, double distance) {
                    if (distance < p_.lp_min_clearance) {
                        hard_violation = true;
                    }
                    Vec2d dir = centre - p;
                    const double dl = dir.norm();
                    if (dl > 1e-9) dir /= dl;
                    const double closing = std::max(0.0, v_seg.dot(dir));
                    if (distance < requiredClearance(closing)) {
                        envelope_violation = true;
                    }
                });
            if (hard_violation) return true;
            if (envelope_violation) {
                dynamic_violation = true;
                return true;
            }
        }
    }
    return false;
}

std::vector<LocalPlannerCandidate> LocalPlanner30Hz::generateCandidates(
    const VehicleState2D& state) const {
    struct Raw {
        double desired_vx, desired_vy, desired_yr;  // intent
        double vx, vy, yr;                          // executable output
    };
    std::vector<Raw> raws;
    raws.reserve(p_.lp_speed_samples.size() *
                 p_.lp_lateral_ratio_samples.size() *
                 p_.lp_yaw_rate_samples.size());
    for (double vx : p_.lp_speed_samples) {
        for (double lr : p_.lp_lateral_ratio_samples) {
            const double raw_vy = lr * p_.lp_max_speed;
            for (double yr : p_.lp_yaw_rate_samples) {
                const BodyCommand2D out =
                    reachableCommand(state, BodyCommand2D{vx, raw_vy, yr});
                Raw r;
                r.desired_vx = vx;
                r.desired_vy = raw_vy;
                r.desired_yr = yr;
                r.vx = out.vx_body;
                r.vy = out.vy_body;
                r.yr = out.yaw_rate;
                raws.push_back(r);
            }
        }
    }

    // ── Deterministic de-duplication on the JOINT (intent + output) key ──
    // Plus a small, deterministic set of low-speed FOV-edge escape intents
    // (both sides symmetric, no side/truth state).
    double min_positive_speed = std::numeric_limits<double>::infinity();
    for (double speed : p_.lp_speed_samples) {
        if (speed > 1e-6) {
            min_positive_speed = std::min(min_positive_speed, speed);
        }
    }
    if (std::isfinite(min_positive_speed)) {
        const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
        const double escape_yaw_rate = std::min(0.25, p_.lp_max_yaw_rate);
        const double escape_speeds[2] = {
            0.5 * min_positive_speed, min_positive_speed};
        const double bearing_fractions[2] = {0.75, 0.90};
        for (double speed : escape_speeds) {
            if (speed <= 1e-6) continue;
            for (double fraction : bearing_fractions) {
                for (int side_sign : {-1, 1}) {
                    const double bearing =
                        static_cast<double>(side_sign) * fraction * fov_half;
                    const double vx = speed * std::cos(bearing);
                    const double vy = speed * std::sin(bearing);
                    const double yaw_options[2] = {
                        0.0,
                        static_cast<double>(side_sign) * escape_yaw_rate};
                    for (double yr : yaw_options) {
                        const BodyCommand2D out_cmd = reachableCommand(
                            state, BodyCommand2D{vx, vy, yr});
                        Raw r;
                        r.desired_vx = vx;
                        r.desired_vy = vy;
                        r.desired_yr = yr;
                        r.vx = out_cmd.vx_body;
                        r.vy = out_cmd.vy_body;
                        r.yr = out_cmd.yaw_rate;
                        raws.push_back(r);
                    }
                }
            }
        }
    }

    constexpr double kQ = 1e4;
    using JointKey =
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    auto keyOf = [](const Raw& r) {
        return JointKey(
            static_cast<int64_t>(std::llround(r.desired_vx * kQ)),
            static_cast<int64_t>(std::llround(r.desired_vy * kQ)),
            static_cast<int64_t>(std::llround(r.desired_yr * kQ)),
            static_cast<int64_t>(std::llround(r.vx * kQ)),
            static_cast<int64_t>(std::llround(r.vy * kQ)),
            static_cast<int64_t>(std::llround(r.yr * kQ)));
    };
    std::map<JointKey, Raw> unique;
    for (const Raw& r : raws) unique[keyOf(r)] = r;
    std::vector<Raw> ordered;
    ordered.reserve(unique.size());
    for (const auto& kv : unique) ordered.push_back(kv.second);

    // ── stable_index: deterministic AND mirror-symmetric ────────────
    std::map<std::tuple<int64_t, int64_t, int64_t>, int> mirror_index;
    int idx = 0;
    std::vector<LocalPlannerCandidate> out;
    out.reserve(ordered.size());
    for (const Raw& r : ordered) {
        const int64_t vxq = static_cast<int64_t>(std::llround(r.desired_vx * kQ));
        int64_t vyq = static_cast<int64_t>(std::llround(r.desired_vy * kQ));
        int64_t yrq = static_cast<int64_t>(std::llround(r.desired_yr * kQ));
        auto mirror = std::make_tuple(vxq, vyq, yrq);
        if (vyq < 0 || (vyq == 0 && yrq < 0)) {
            mirror = std::make_tuple(vxq, -vyq, -yrq);
        }
        int si = idx;
        const auto it = mirror_index.find(mirror);
        if (it != mirror_index.end()) {
            si = it->second;
        } else {
            mirror_index[mirror] = si;
            ++idx;
        }

        LocalPlannerCandidate cand;
        cand.desired_vx_body = r.desired_vx;
        cand.desired_vy_body = r.desired_vy;
        cand.desired_yaw_rate = r.desired_yr;
        cand.vx_body = r.vx;
        cand.vy_body = r.vy;
        cand.yaw_rate = r.yr;
        cand.stable_index = si;
        cand.tie_hash = commandTieHash(r.desired_vx, r.desired_vy, r.desired_yr);
        cand.traj = simulateCandidate(
            state, BodyCommand2D{r.desired_vx, r.desired_vy, r.desired_yr}, p_);
        out.push_back(std::move(cand));
    }
    return out;
}

bool LocalPlanner30Hz::evaluateCandidate(LocalPlannerCandidate& c,
                                         const VehicleState2D& state,
                                         const LocalObservation& obs,
                                         const LocalTarget& target,
                                         const std::vector<Vec2d>& risk_occ_cells,
                                         std::string& reject_reason,
                                         CandidateRejectReason& reject_enum) const {
    reject_enum = CandidateRejectReason::NONE;
    const double fov = deg2rad(p_.obs_fov_deg);
    const double dist_current = (target.position - state.position).norm();

    double min_clear = std::numeric_limits<double>::infinity();
    double max_dyn_req = 0.0;
    double max_closing = 0.0;
    const double seg_step = std::max(1e-3, 0.5 * p_.obs_resolution);

    // ── NOMINAL rollout (v5): limit at target capture / target-plane
    //    crossing / closest-approach-then-recede.  Obstacles behind a near
    //    target must never enter the prediction. ─────────────────────
    auto limitTrajectoryAtTarget = [&](const Trajectory2D& full,
                                       bool* captured_at_goal) {
        if (captured_at_goal) *captured_at_goal = false;
        if (!full.valid || full.points.size() < 2) return full;

        const double goal_tol = std::max(1e-6, p_.task_goal_tolerance);
        const Vec2d to_target = target.position - state.position;
        const double target_dist = to_target.norm();
        const Vec2d target_dir = target_dist > 1e-9
            ? to_target / target_dist
            : Vec2d(1.0, 0.0);

        size_t stop_index = full.points.size() - 1;
        size_t closest_index = 0;
        double closest_dist = target_dist;
        bool explicit_stop = false;
        for (size_t i = 1; i < full.points.size(); ++i) {
            const Vec2d rel = full.points[i] - state.position;
            const double d = (target.position - full.points[i]).norm();
            if (d < closest_dist) {
                closest_dist = d;
                closest_index = i;
            }
            if (d <= goal_tol) {
                stop_index = i;
                explicit_stop = true;
                break;
            }
            if (rel.dot(target_dir) >= target_dist + goal_tol) {
                stop_index = i;
                explicit_stop = true;
                break;
            }
        }

        if (!explicit_stop && closest_index > 0 &&
            closest_index + 1 < full.points.size()) {
            const double next_dist =
                (target.position - full.points[closest_index + 1]).norm();
            if (next_dist > closest_dist + 1e-6) {
                stop_index = closest_index;
            }
        }
        if (captured_at_goal) *captured_at_goal = explicit_stop;
        if (stop_index + 1 >= full.points.size()) return full;

        Trajectory2D limited;
        limited.valid = full.valid;
        limited.points.reserve(stop_index + 1);
        limited.yaw.reserve(std::min(stop_index + 1, full.yaw.size()));
        limited.t.reserve(std::min(stop_index + 1, full.t.size()));
        for (size_t i = 0; i <= stop_index; ++i) {
            limited.points.push_back(full.points[i]);
            if (i < full.yaw.size()) limited.yaw.push_back(full.yaw[i]);
            if (i < full.t.size()) limited.t.push_back(full.t[i]);
        }
        return limited;
    };

    bool captured_at_goal = false;
    c.traj = limitTrajectoryAtTarget(c.traj, &captured_at_goal);
    c.nominal_traj = c.traj;
    std::vector<Vec2d> occ_cells;
    collectOccupiedCells(c.traj, obs, occ_cells);

    // Soft obstacle risk is also target-bounded.
    std::vector<Vec2d> target_risk_cells;
    target_risk_cells.reserve(risk_occ_cells.size());
    const Vec2d risk_to_target = target.position - state.position;
    const double risk_target_dist = risk_to_target.norm();
    const Vec2d risk_target_dir = risk_target_dist > 1e-9
        ? risk_to_target / risk_target_dist
        : Vec2d(1.0, 0.0);
    const double target_risk_limit =
        risk_target_dist + std::max(0.0, p_.task_goal_tolerance);
    for (const Vec2d& cell : risk_occ_cells) {
        if ((cell - state.position).dot(risk_target_dir) <=
            target_risk_limit + 1e-9) {
            target_risk_cells.push_back(cell);
        }
    }

    auto pointOk = [&](const Vec2d& p, const Vec2d& v_seg,
                       bool is_vehicle_point, std::string& why,
                       CandidateRejectReason& why_enum) -> bool {
        if (!is_vehicle_point) {
            const Vec2d rel = p - state.position;
            if (rel.squaredNorm() > 1e-12) {
                if (rel.norm() > p_.obs_range_m + 1e-6) {
                    why = "outside_current_fov";
                    why_enum = CandidateRejectReason::OUTSIDE_CURRENT_FOV;
                    return false;
                }
                const double bearing =
                    wrapAngle(std::atan2(rel.y(), rel.x()) - state.yaw);
                if (std::fabs(bearing) > fov / 2.0 + 1e-6) {
                    why = "outside_current_fov";
                    why_enum = CandidateRejectReason::OUTSIDE_CURRENT_FOV;
                    return false;
                }
            }
        }
        if (!obs.isKnownFree(p.x(), p.y())) {
            why = "not_known_free";
            why_enum = CandidateRejectReason::NOT_KNOWN_FREE;
            return false;
        }
        const double search_r = clearanceSearchRadius(v_seg.norm());
        const double r2 = search_r * search_r;
        bool any_hard_violation = false;
        bool any_envelope_violation = false;
        for (const Vec2d& centre : occ_cells) {
            const Vec2d dc = centre - p;
            const double d2 = dc.squaredNorm();
            if (d2 > r2 + 1e-12) continue;
            const double distance = std::sqrt(std::max(0.0, d2));
            min_clear = std::min(min_clear, distance);
            if (distance < p_.lp_min_clearance) {
                any_hard_violation = true;
            }
            const double dl = dc.norm();
            Vec2d dir_all = dc;
            if (dl > 1e-9) dir_all /= dl;
            const double closing_all =
                std::max(0.0, v_seg.dot(dir_all));
            max_closing = std::max(max_closing, closing_all);
            const double required_all = requiredClearance(closing_all);
            max_dyn_req = std::max(max_dyn_req, required_all);
            if (distance < required_all) {
                any_envelope_violation = true;
            }
        }
        if (any_hard_violation) {
            why = "observed_clearance_too_small";
            why_enum = CandidateRejectReason::OBSERVED_CLEARANCE_TOO_SMALL;
            return false;
        }
        if (any_envelope_violation) {
            why = "insufficient_braking_clearance";
            why_enum =
                CandidateRejectReason::INSUFFICIENT_BRAKING_CLEARANCE;
            return false;
        }
        return true;
    };

    // ── EXECUTABLE SAFE PREFIX ─────────────────────────────────────
    size_t prefix_last = 0;
    std::string first_fail;
    CandidateRejectReason first_fail_enum = CandidateRejectReason::NONE;
    bool failed = false;
    for (size_t i = 0; i + 1 < c.traj.points.size() && !failed; ++i) {
        const Vec2d& a = c.traj.points[i];
        const Vec2d& b = c.traj.points[i + 1];
        double dt_seg = p_.lp_dt;
        if (i + 1 < c.traj.t.size() && i < c.traj.t.size()) {
            const double dt = c.traj.t[i + 1] - c.traj.t[i];
            if (dt > 1e-9) dt_seg = dt;
        }
        const double seg = (b - a).norm();
        const Vec2d v_seg =
            (seg > 1e-9) ? (b - a) / dt_seg : Vec2d(0.0, 0.0);
        const int steps =
            std::max(1, static_cast<int>(std::ceil(seg / seg_step)));
        for (int k = 1; k <= steps; ++k) {
            const Vec2d p = a + (b - a) * (static_cast<double>(k) / steps);
            std::string why;
            CandidateRejectReason why_enum = CandidateRejectReason::NONE;
            const double min_clear_before = min_clear;
            const double max_dyn_before = max_dyn_req;
            const double max_closing_before = max_closing;
            if (!pointOk(p, v_seg, /*is_vehicle_point=*/false, why,
                         why_enum)) {
                min_clear = min_clear_before;
                max_dyn_req = max_dyn_before;
                max_closing = max_closing_before;
                first_fail = why;
                first_fail_enum = why_enum;
                failed = true;
                break;
            }
        }
        if (!failed) prefix_last = i + 1;
    }

    // A visible occupied-clearance violation rejects the whole candidate.
    if (failed &&
        (first_fail_enum == CandidateRejectReason::OBSERVED_CLEARANCE_TOO_SMALL ||
         first_fail_enum ==
             CandidateRejectReason::INSUFFICIENT_BRAKING_CLEARANCE)) {
        reject_reason = first_fail;
        reject_enum = first_fail_enum;
        return false;
    }

    const size_t min_prefix_steps = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(
               p_.lp_min_executable_prefix_s / std::max(1e-6, p_.lp_dt))));
    // A candidate truncated by GOAL CAPTURE (the trajectory reaches the
    // goal-tolerance sphere) legitimately has a short prefix: the drone is
    // converging onto the target.  Without this exemption every forward
    // candidate near the goal collapses to one segment and is rejected as
    // "no_usable_prefix", leaving only the stationary candidate -> the
    // drone stalls a few cm short of the goal (blueprint stall=52 at
    // dgoal=0.41 vs goal_tolerance=0.40).
    if (prefix_last < min_prefix_steps && !captured_at_goal) {
        reject_reason = first_fail.empty() ? "no_usable_prefix" : first_fail;
        reject_enum = first_fail.empty() ? CandidateRejectReason::OTHER
                                         : first_fail_enum;
        return false;
    }

    Trajectory2D exec;
    exec.valid = true;
    for (size_t i = 0; i <= prefix_last && i < c.traj.points.size(); ++i) {
        exec.points.push_back(c.traj.points[i]);
        if (i < c.traj.yaw.size()) exec.yaw.push_back(c.traj.yaw[i]);
        if (i < c.traj.t.size()) exec.t.push_back(c.traj.t[i]);
    }
    if (exec.points.size() < 2) {
        reject_reason = "no_usable_prefix";
        reject_enum = CandidateRejectReason::OTHER;
        return false;
    }

    // ── Progress metrics (v5) ──────────────────────────────────────
    const Vec2d& end = exec.points.back();
    const double dist_end = (target.position - end).norm();
    const double executable_progress_m = dist_current - dist_end;
    double nominal_progress_m = executable_progress_m;
    if (!c.nominal_traj.points.empty()) {
        const Vec2d& nend = c.nominal_traj.points.back();
        nominal_progress_m =
            dist_current - (target.position - nend).norm();
    }
    if (dist_current > p_.task_goal_tolerance &&
        executable_progress_m < -p_.lp_max_allowed_regress_m) {
        reject_reason = "no_progress";
        reject_enum = CandidateRejectReason::NO_PROGRESS;
        return false;
    }

    const double exec_duration =
        exec.t.empty() ? 0.0 : std::max(0.0, exec.t.back());
    const double scoring_horizon = std::max(
        p_.lp_min_executable_prefix_s, p_.lp_scoring_horizon_s);
    const double achievable_progress_m = p_.lp_max_speed * scoring_horizon;

    // ── Progress qualification (v5) ────────────────────────────────
    const double output_speed = std::hypot(c.vx_body, c.vy_body);
    const bool moving = output_speed >= p_.lp_min_progress_speed_mps;
    c.stationary = !moving;
    c.safe_prefix_duration_s = exec_duration;
    c.nominal_progress_m = nominal_progress_m;
    c.executable_progress_m = executable_progress_m;
    c.achievable_progress_m = achievable_progress_m;
    c.progress_qualified =
        (dist_current > p_.task_goal_tolerance) &&
        (executable_progress_m >= p_.lp_min_progress_epsilon_m) && moving;

    // ── Target-alignment metrics ───────────────────────────────────
    const Vec2d ref = target.position - state.position;
    const double ref_len = ref.norm();
    Vec2d ref_dir(1.0, 0.0);
    if (ref_len > 1e-9) ref_dir = ref / ref_len;
    const Vec2d ref_lat(-ref_dir.y(), ref_dir.x());

    const bool target_direction_defined = ref_len > 1e-6;
    double target_dir = exec.yaw.back();
    if (target_direction_defined) {
        target_dir = std::atan2(ref.y(), ref.x());
    }
    const double terminal_yaw = exec.yaw.back();
    const double terminal_heading_error = target_direction_defined
        ? std::fabs(wrapAngle(target_dir - terminal_yaw)) / M_PI
        : 0.0;

    double velocity_alignment_error = terminal_heading_error;
    if (exec.points.size() >= 2) {
        const double dt_last = std::max(
            1e-6, exec.t.back() - exec.t[exec.t.size() - 2]);
        const Vec2d v_end =
            (exec.points.back() - exec.points[exec.points.size() - 2]) / dt_last;
        if (target_direction_defined &&
            v_end.norm() > p_.vehicle_stationary_speed_mps) {
            const double vdir = std::atan2(v_end.y(), v_end.x());
            velocity_alignment_error =
                std::fabs(wrapAngle(target_dir - vdir)) / M_PI;
        }
    }

    double cross_sum = 0.0;
    int cross_n = 0;
    for (const Vec2d& p : exec.points) {
        cross_sum += std::fabs((p - state.position).dot(ref_lat));
        ++cross_n;
    }
    const double cross_track_m = cross_n > 0 ? cross_sum / cross_n : 0.0;
    const double cost_cross_track = clamp(
        cross_track_m / std::max(1e-6, p_.lp_cross_track_normalize_m), 0.0,
        1.5);

    // ── Cost terms (all normalised) ────────────────────────────────
    const double progress_ratio = clamp(
        executable_progress_m / std::max(1e-6, achievable_progress_m),
        -1.0, 1.0);
    const double short_prefix_penalty = clamp(
        (scoring_horizon - exec_duration) /
            std::max(1e-6, scoring_horizon),
        0.0, 1.0);
    const double cost_progress =
        1.0 - progress_ratio + short_prefix_penalty;
    const double cost_clearance = std::isfinite(min_clear)
        ? clamp((p_.lp_soft_clearance_radius_m - min_clear) /
                    std::max(1e-6, p_.lp_soft_clearance_radius_m -
                                       p_.lp_min_clearance),
                0.0, 1.0)
        : 0.0;

    const double c_yaw = std::cos(state.yaw), s_yaw = std::sin(state.yaw);
    const Vec2d current_v_body(
        c_yaw * state.velocity_world.x() + s_yaw * state.velocity_world.y(),
        -s_yaw * state.velocity_world.x() + c_yaw * state.velocity_world.y());
    const Vec2d command_v(c.vx_body, c.vy_body);
    const bool comparable_previous_command =
        has_last_command_ &&
        last_mission_revision_ == target.mission_revision;
    const Vec2d previous_command_v = comparable_previous_command
        ? Vec2d(last_command_.vx_body, last_command_.vy_body)
        : current_v_body;
    auto commandedAcceleration = [&](const Vec2d& desired) {
        return Vec2d(
            clamp((desired.x() - current_v_body.x()) / p_.lp_dt,
                  -p_.lp_max_accel, p_.lp_max_accel),
            clamp((desired.y() - current_v_body.y()) / p_.lp_dt,
                  -p_.lp_max_accel, p_.lp_max_accel));
    };

    double cost_smoothness =
        (commandedAcceleration(command_v) -
         commandedAcceleration(previous_command_v)).norm() /
        std::max(1e-6, 2.0 * p_.lp_max_accel);
    int smooth_cnt = 1;
    for (size_t i = 3; i < exec.points.size(); ++i) {
        const Vec2d v0 =
            (exec.points[i - 2] - exec.points[i - 3]) / p_.lp_dt;
        const Vec2d v1 =
            (exec.points[i - 1] - exec.points[i - 2]) / p_.lp_dt;
        const Vec2d v2 =
            (exec.points[i] - exec.points[i - 1]) / p_.lp_dt;
        const Vec2d a0 = (v1 - v0) / p_.lp_dt;
        const Vec2d a1 = (v2 - v1) / p_.lp_dt;
        cost_smoothness +=
            (a1 - a0).norm() / std::max(1e-6, 2.0 * p_.lp_max_accel);
        ++smooth_cnt;
    }
    cost_smoothness /= std::max(1, smooth_cnt);

    const double output_command_change = comparable_previous_command
        ? (command_v - previous_command_v).norm()
        : (command_v - current_v_body).norm();
    const double cost_speed_change = clamp(
        output_command_change / std::max(1e-6, p_.lp_max_speed), 0.0, 1.5);

    const double previous_yaw_reference = comparable_previous_command
        ? last_command_.yaw_rate
        : state.yaw_rate;
    const double cmd_yaw_change =
        0.5 * (std::fabs(c.yaw_rate - state.yaw_rate) +
               std::fabs(c.yaw_rate - previous_yaw_reference)) /
        std::max(1e-6, p_.lp_max_yaw_rate);
    double yr_var = 0.0;
    int yr_n = 0;
    for (size_t i = 2; i < exec.yaw.size(); ++i) {
        const double y1 =
            wrapAngle(exec.yaw[i - 1] - exec.yaw[i - 2]) / p_.lp_dt;
        const double y2 =
            wrapAngle(exec.yaw[i] - exec.yaw[i - 1]) / p_.lp_dt;
        const double yaw_accel = std::fabs(y2 - y1) / p_.lp_dt;
        yr_var += yaw_accel / std::max(1e-6, p_.lp_max_yaw_accel);
        ++yr_n;
    }
    const double yr_var_cost = yr_n > 0 ? yr_var / yr_n : 0.0;
    const double cost_yaw_rate_change =
        clamp(cmd_yaw_change + yr_var_cost, 0.0, 1.5);

    const double cost_terminal_heading = terminal_heading_error;
    const double cost_velocity_alignment = velocity_alignment_error;

    // ── Continuous early-avoidance risk (v7) ───────────────────────
    computeObstacleRisk(c, state, target_risk_cells);
    const double cost_obstacle_risk = c.obstacle_risk_cost;

    const double total =
        p_.cost_w_progress * cost_progress +
        p_.cost_w_clearance * cost_clearance +
        p_.cost_w_smoothness * cost_smoothness +
        p_.cost_w_speed_change * cost_speed_change +
        p_.cost_w_yaw_rate_change * cost_yaw_rate_change +
        p_.cost_w_terminal_heading * cost_terminal_heading +
        p_.cost_w_velocity_alignment * cost_velocity_alignment +
        p_.cost_w_cross_track * cost_cross_track +
        p_.cost_w_obstacle_risk * cost_obstacle_risk;

    c.min_clearance = min_clear;
    c.soft_min_clearance = min_clear;
    c.max_dynamic_required_clearance = max_dyn_req;
    c.max_closing_speed = max_closing;
    c.cost = total;
    c.cost_progress = cost_progress;
    c.cost_clearance = cost_clearance;
    c.cost_smoothness = cost_smoothness;
    c.cost_speed_change = cost_speed_change;
    c.cost_yaw_rate_change = cost_yaw_rate_change;
    c.cost_terminal_heading = cost_terminal_heading;
    c.cost_velocity_alignment = cost_velocity_alignment;
    c.cost_cross_track = cost_cross_track;
    c.terminal_heading_error_rad = terminal_heading_error * M_PI;
    c.velocity_alignment_error_rad = velocity_alignment_error * M_PI;
    c.cross_track_error_m = cross_track_m;
    c.obstacle_risk_cost = cost_obstacle_risk;
    c.status = c.progress_qualified ? PlannerStatus::SAFE_PROGRESSING
                                    : PlannerStatus::SAFE_HOLD;
    c.traj = std::move(exec);
    c.feasible = true;
    return true;
}

void LocalPlanner30Hz::computeObstacleRisk(
    LocalPlannerCandidate& c, const VehicleState2D& state,
    const std::vector<Vec2d>& occ_cells) const {
    c.obstacle_risk_cost = 0.0;
    c.predicted_closest_clearance =
        std::numeric_limits<double>::infinity();
    c.time_to_collision = std::numeric_limits<double>::infinity();
    c.avoidance_strength = 0.0;
    c.avoidance_active = false;
    if (occ_cells.empty() || c.nominal_traj.points.size() < 2) return;

    const double cy = std::cos(state.yaw);
    const double sy = std::sin(state.yaw);
    Vec2d dir_cand(cy * c.desired_vx_body - sy * c.desired_vy_body,
                   sy * c.desired_vx_body + cy * c.desired_vy_body);
    const double dl = dir_cand.norm();
    if (dl > 1e-9) {
        dir_cand /= dl;
    } else {
        dir_cand = c.nominal_traj.points.back() - state.position;
        if (dir_cand.norm() > 1e-9) {
            dir_cand.normalize();
        } else {
            dir_cand = Vec2d(cy, sy);
        }
    }

    const double hard = p_.lp_min_clearance;
    const double lon_horizon = p_.lp_risk_distance_horizon_m;
    const double lon_shoulder = 1.0;
    const double clear_horizon = p_.lp_risk_distance_horizon_m;
    const double ttc_horizon = p_.lp_risk_ttc_horizon_s;
    const double corridor_hw = p_.lp_risk_corridor_half_width;
    const double corridor_shoulder = 0.5;
    const double traj_radius = p_.lp_risk_trajectory_radius_m;
    const auto& pts = c.nominal_traj.points;
    const auto& ts = c.nominal_traj.t;
    double best_risk = 0.0;
    double best_clear = std::numeric_limits<double>::infinity();
    double best_ttc = std::numeric_limits<double>::infinity();

    for (const Vec2d& cell : occ_cells) {
        const Vec2d to = cell - state.position;
        const double lon = to.dot(dir_cand);
        if (lon <= 0.0) continue;
        const double lat = std::fabs(cross2(dir_cand, to));
        const double lon_gate = clamp(
            (lon_horizon + lon_shoulder - lon) /
                std::max(1e-6, lon_shoulder),
            0.0, 1.0);
        if (lon_gate <= 0.0) continue;
        const double g_corridor = clamp(
            1.0 - (lat - corridor_hw) / std::max(1e-6, corridor_shoulder),
            0.0, 1.0);
        double d2_best = std::numeric_limits<double>::infinity();
        double first_conflict_time =
            std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < pts.size(); ++i) {
            const double d2 = (pts[i] - cell).squaredNorm();
            if (d2 < d2_best) {
                d2_best = d2;
            }
            if (!std::isfinite(first_conflict_time) &&
                d2 <= p_.lp_nominal_clearance_m *
                          p_.lp_nominal_clearance_m) {
                first_conflict_time =
                    (i < ts.size()) ? ts[i] : (i * p_.lp_dt);
            }
        }
        const double d_path = std::sqrt(std::max(0.0, d2_best));
        const double g_traj = clamp(
            1.0 - (d_path - hard) / std::max(1e-6, traj_radius), 0.0, 1.0);
        const double membership = std::max(g_corridor, g_traj);
        if (membership <= 0.0) continue;
        const double dist_factor = clamp(
            (lon_horizon - lon) / std::max(1e-6, lon_horizon - hard),
            0.0, 1.0);
        const double clear_factor = clamp(
            (clear_horizon - d_path) /
                std::max(1e-6, clear_horizon - hard),
            0.0, 1.0);
        const double ttc = first_conflict_time;
        const double ttc_factor = clamp(
            (ttc_horizon - ttc) / std::max(1e-6, ttc_horizon), 0.0, 1.0);
        const double risk_i =
            lon_gate * membership *
            std::max(dist_factor, std::max(clear_factor, ttc_factor));
        if (risk_i > best_risk) best_risk = risk_i;
        if (d_path < best_clear) best_clear = d_path;
        if (ttc < best_ttc) best_ttc = ttc;
    }

    c.predicted_closest_clearance = best_clear;
    c.time_to_collision = best_ttc;
    c.avoidance_strength = clamp(best_risk, 0.0, 1.5);
    c.obstacle_risk_cost = c.avoidance_strength;
    c.avoidance_active =
        c.avoidance_strength >= p_.lp_avoidance_active_threshold;
}

void LocalPlanner30Hz::assessLocalCorridor(
    const VehicleState2D& state, const LocalObservation& obs,
    const LocalTarget& target, bool& blocked,
    double& first_blocking_distance_m) const {
    blocked = false;
    first_blocking_distance_m = std::numeric_limits<double>::quiet_NaN();
    const Vec2d to = target.position - state.position;
    const double dist = to.norm();
    if (dist < 1e-3) return;
    const Vec2d dir = to / dist;
    const double range = std::min(dist, p_.obs_range_m);
    const double step = p_.obs_resolution * 0.5;
    const double corridor = p_.lp_risk_corridor_half_width;
    for (double d = step; d <= range; d += step) {
        const Vec2d p = state.position + dir * d;
        if (obs.minClearanceToOccupied(p, corridor) < corridor) {
            blocked = true;
            first_blocking_distance_m = d;
            return;
        }
    }
}

void LocalPlanner30Hz::collectOccupiedCells(const Trajectory2D& traj,
                                            const LocalObservation& obs,
                                            std::vector<Vec2d>& out) const {
    out.clear();
    if (!obs.valid() || traj.points.empty()) return;
    const double max_search_r = clearanceSearchRadius(p_.lp_max_speed);
    double bx0 = std::numeric_limits<double>::infinity();
    double by0 = std::numeric_limits<double>::infinity();
    double bx1 = -std::numeric_limits<double>::infinity();
    double by1 = -std::numeric_limits<double>::infinity();
    for (const Vec2d& pt : traj.points) {
        bx0 = std::min(bx0, pt.x());
        by0 = std::min(by0, pt.y());
        bx1 = std::max(bx1, pt.x());
        by1 = std::max(by1, pt.y());
    }
    bx0 -= max_search_r;
    by0 -= max_search_r;
    bx1 += max_search_r;
    by1 += max_search_r;
    const GridIndex2D g0 =
        worldToGrid(Vec2d(bx0, by0), obs.origin, obs.resolution);
    const GridIndex2D g1 =
        worldToGrid(Vec2d(bx1, by1), obs.origin, obs.resolution);
    out.reserve(64);
    for (int iy = std::max(0, g0.iy); iy <= std::min(obs.height - 1, g1.iy);
         ++iy) {
        for (int ix = std::max(0, g0.ix);
             ix <= std::min(obs.width - 1, g1.ix); ++ix) {
            if (obs.cells[obs.idx(ix, iy)] == CellState::OCCUPIED) {
                out.emplace_back(
                    obs.origin.x() + (ix + 0.5) * obs.resolution,
                    obs.origin.y() + (iy + 0.5) * obs.resolution);
            }
        }
    }
}

void LocalPlanner30Hz::collectRiskOccupiedCells(
    const VehicleState2D& state, const LocalObservation& obs,
    std::vector<Vec2d>& out) const {
    out.clear();
    if (!obs.valid()) return;
    const double radius =
        p_.lp_risk_distance_horizon_m +
        std::max(1.0, p_.lp_risk_trajectory_radius_m);
    obs.forEachOccupiedWithin(
        state.position, radius,
        [&](const Vec2d& centre, double) { out.push_back(centre); });
}

bool LocalPlanner30Hz::corridorBlockedByObserved(
    const VehicleState2D& state, const LocalObservation& obs,
    const LocalTarget& target) const {
    const Vec2d to = target.position - state.position;
    const double dist = to.norm();
    if (dist < 1e-3) return false;
    const Vec2d dir = to / dist;
    const double range = std::min(dist, p_.obs_range_m);
    const double step = p_.obs_resolution * 0.5;
    for (double d = step; d <= range; d += step) {
        const Vec2d p = state.position + dir * d;
        if (obs.minClearanceToOccupied(p, p_.lp_min_clearance) <
            p_.lp_min_clearance) {
            return true;
        }
    }
    if (obs.isOccupied(target.position.x(), target.position.y())) return true;
    return false;
}

double LocalPlanner30Hz::stoppingDistance(const VehicleState2D& state) const {
    const double v = state.velocity_world.norm();
    if (v <= 1e-3) return 0.0;
    const double t = v / p_.lp_max_accel;
    const double d = v * t - 0.5 * p_.lp_max_accel * t * t;
    return d + p_.lp_brake_stop_margin_m;
}

bool LocalPlanner30Hz::spaceToStop(const VehicleState2D& state,
                                   const LocalObservation& obs,
                                   double dist) const {
    const double v = state.velocity_world.norm();
    Vec2d dir;
    if (v > p_.vehicle_stationary_speed_mps) {
        dir = state.velocity_world / v;
    } else {
        dir = Vec2d(std::cos(state.yaw), std::sin(state.yaw));
    }
    const double step = p_.obs_resolution * 0.5;
    const double handoff = handoffClearance();
    const double hard_clearance = p_.scene_safety_clearance;
    auto centreSafe = [&](const Vec2d& p, double required_clearance) {
        if (!obs.isKnownFree(p.x(), p.y())) return false;
        const NearestOccupiedResult nr = obs.nearestOccupied(
            p, std::max(p_.lp_soft_clearance_radius_m, required_clearance));
        return !nr.found || nr.distance >= required_clearance;
    };
    if (!centreSafe(state.position, hard_clearance)) return false;
    for (double d = step; d <= std::max(0.0, dist) + 1e-6; d += step) {
        if (!centreSafe(state.position + dir * d, handoff)) return false;
    }
    return true;
}

// ────────────────────────────────────────────────────────────────────
//  Deterministic stall / control-oscillation detector (v5)
// ────────────────────────────────────────────────────────────────────
bool LocalPlanner30Hz::updateLimitCycle(const PlannerResult& res,
                                        const LocalTarget& target,
                                        const VehicleState2D& state) {
    if (limit_cycle_window_.empty()) {
        last_cycle_mission_revision_ = target.mission_revision;
    } else if (last_cycle_mission_revision_ != target.mission_revision ||
               res.target_discontinuity_reset) {
        limit_cycle_window_.clear();
        limit_cycle_detected_ = false;
        last_cycle_mission_revision_ = target.mission_revision;
    }

    LimitCycleSample s;
    s.target_update_event = target.update_event;
    s.mission_revision = target.mission_revision;
    s.dist_to_target = (target.position - state.position).norm();
    s.vx_body = res.vx_body;
    s.vy_body = res.vy_body;
    s.yaw_rate = res.yaw_rate;
    s.blocked = !res.success &&
                (res.blocked_observed || res.dynamic_clearance_blocked);
    s.position = state.position;
    s.target_position = target.position;
    limit_cycle_window_.push_back(s);
    const int win = std::max(1, p_.lp_limit_cycle_window_ticks);
    while (static_cast<int>(limit_cycle_window_.size()) > win) {
        limit_cycle_window_.erase(limit_cycle_window_.begin());
    }

    const int min_samples = std::max(3, win / 2);
    if (static_cast<int>(limit_cycle_window_.size()) < min_samples) {
        limit_cycle_detected_ = false;
        return false;
    }
    const auto& first = limit_cycle_window_.front();
    const auto& last = limit_cycle_window_.back();

    const double displacement = (last.position - first.position).norm();
    double projected_progress = 0.0;
    for (size_t i = 0; i + 1 < limit_cycle_window_.size(); ++i) {
        const auto& a = limit_cycle_window_[i];
        const auto& b = limit_cycle_window_[i + 1];
        const Vec2d step = b.position - a.position;
        const Vec2d to = a.target_position - a.position;
        const double len = to.norm();
        if (len > 1e-9) {
            projected_progress += std::max(0.0, step.dot(to / len));
        }
    }
    if (displacement >= p_.lp_limit_cycle_net_progress_m ||
        projected_progress >= p_.lp_limit_cycle_net_progress_m) {
        limit_cycle_detected_ = false;
        return false;
    }

    int blocked_ticks = 0;
    for (const auto& e : limit_cycle_window_) {
        if (e.blocked) ++blocked_ticks;
    }
    if (blocked_ticks >= std::max(1, p_.lp_limit_cycle_min_blocked_ticks)) {
        limit_cycle_detected_ = true;
        return true;
    }

    int sign_flips = 0;
    int stop_lateral_stop = 0;
    for (size_t i = 0; i < limit_cycle_window_.size(); ++i) {
        const auto& e = limit_cycle_window_[i];
        const double spd = std::hypot(e.vx_body, e.vy_body);
        const int sg = (std::fabs(e.vy_body) < 0.05)
                           ? 0
                           : (e.vy_body > 0.0 ? 1 : -1);
        if (i > 0) {
            const auto& p = limit_cycle_window_[i - 1];
            const int psg = (std::fabs(p.vy_body) < 0.05)
                                ? 0
                                : (p.vy_body > 0.0 ? 1 : -1);
            if (sg != 0 && psg != 0 && sg != psg) ++sign_flips;
        }
        const bool lateral_now = std::fabs(e.vy_body) > 0.3 && spd > 0.05;
        if (i >= 1 && i + 1 < limit_cycle_window_.size()) {
            const auto& pp = limit_cycle_window_[i - 1];
            const auto& nn = limit_cycle_window_[i + 1];
            const double pspd = std::hypot(pp.vx_body, pp.vy_body);
            const double nspd = std::hypot(nn.vx_body, nn.vy_body);
            const bool p_zero = pspd <= 0.05;
            const bool n_zero = nspd <= 0.05;
            if (p_zero && lateral_now && n_zero) ++stop_lateral_stop;
        }
    }
    if (blocked_ticks > 0 &&
        (sign_flips >= std::max(1, p_.lp_limit_cycle_lateral_flip_count) ||
         stop_lateral_stop >= 1)) {
        limit_cycle_detected_ = true;
        return true;
    }

    limit_cycle_detected_ = false;
    return false;
}

}  // namespace expert
}  // namespace il_dataset
