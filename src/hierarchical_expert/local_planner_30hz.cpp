#include "il_dataset/hierarchical_expert/local_planner_30hz.hpp"

#include "il_dataset/hierarchical_expert/kinematics.hpp"
#include "il_dataset/hierarchical_expert/ego_bspline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>

namespace il_dataset {
namespace expert {

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
//  Candidate forward simulation 鈥?SAME shared kinematics as the
//  preflight simulator (prediction never diverges from execution).
// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
void LocalPlanner30Hz::reset() {
    turn_hysteresis_active_ = false;
    turn_ticks_ = 0;
    turn_hold_until_ = 0;
    plan_ticks_ = 0;
    current_trajectory_ = PlanarTrajectory{};
    last_command_ = VelocityCommand3D{};
    has_last_command_ = false;
    last_mission_revision_ = 0;
    last_target_position_ = Vec2d(0.0, 0.0);
    last_target_valid_ = false;
    // R25 limit-cycle windows: cleared per task.
    limit_cycle_dist_window_.clear();
    limit_cycle_bearing_window_.clear();
    limit_cycle_blocked_window_.clear();
}

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
//  Shared output semantic: one-control-period reachable command
// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
VelocityCommand3D LocalPlanner30Hz::reachableCommand(
    const PlanarState& state, const VelocityCommand3D& intent) const {
    const Vec2d current_v_body = bodyVelocity(state);
    const double dt = (p_.lp_control_period_s > 0.0)
                          ? p_.lp_control_period_s
                          : (1.0 / 30.0);
    const double dv = p_.lp_max_accel * dt;
    const double dyr = p_.lp_max_yaw_accel * dt;
    // COMMAND RAMP (r20260819_cmd_ramp_feedforward): clamp the command
    // within lp_max_accel*dt of the PREVIOUS COMMAND (not the state), so
    // the command itself ramps at lp_max_accel = 2 m/s^2.  The backend
    // velocity controller FEED-FORWARDS that ramp (VelocityYawRateController
    // adds d(v_cmd)/dt), so the drone tracks ~2 m/s^2 with a modest P gain.
    // State-pinning made the feedforward mirror the state acceleration (it
    // self-cancelled) and the closed loop sat at kp*dv/kd ~= 0.45 m/s^2.
    //
    // Historical note (episode af4159ce): an earlier command-ramp build
    // made the command outrun the state, the P term saturated and the
    // drone entered a ~卤15 deg tilt limit cycle.  That build used kp=36,
    // where P saturates on ANY error > 0.11 m/s.  With the current
    // feedforward + kp=8/kd=1.2 the steady-state tracking error at a
    // 2 m/s^2 ramp is ~0.3 m/s (P = 2.4 < 4, tilt ~11.5 deg < 35), so the
    // P term no longer saturates.  Validated analytically; verify on the
    // attitude loop (roll/pitch in the CSV) before pushing further.
    //
    // Fallbacks: (1) no previous command yet (first tick after reset) 鈫?
    // state-pin; (2) the previous command diverged from the state by more
    // After that, every output is slew-limited from the previous output;
    // without a planner reset) 鈫?state-pin so we never chase a stale
    // command with a saturated P term.
    const Vec2d prev = has_last_command_
        ? Vec2d(last_command_.vx_body, last_command_.vy_body)
        : current_v_body;
    // Once emitted, a supervision command must always be the slew base.
    // Switching to measured velocity on tracking lag made the label itself
    // violate lp_max_accel.  New tasks/resets already seed this state from
    // the measured velocity in updateMissionState().
    const Vec2d base = prev;
    VelocityCommand3D out;
    // Limit the XY command as one vector.  Independent component clamps
    // permit a diagonal step of sqrt(2)*lp_max_accel*dt (2.83 m/s^2 in the
    // collected traces although the configured limit is 2.0 m/s^2).
    Vec2d next(intent.vx_body, intent.vy_body);
    Vec2d delta = next - base;
    const double delta_norm = delta.norm();
    if (delta_norm > dv && delta_norm > 1e-12) {
        delta *= dv / delta_norm;
        next = base + delta;
    }
    out.vx_body = next.x();
    out.vy_body = next.y();
    const double spd = std::hypot(out.vx_body, out.vy_body);
    if (spd > p_.lp_max_speed && spd > 1e-9) {
        out.vx_body *= p_.lp_max_speed / spd;
        out.vy_body *= p_.lp_max_speed / spd;
    }
    // The supervision command itself must be slew-rate limited.  Clamping
    // around ACTUAL yaw rate allowed target_yaw_rate to jump by 0.25--0.29
    // rad/s in one 30 Hz tick whenever tracking lag changed.  Use the last
    // emitted command as the reference; on the first tick it is seeded from
    // the measured state by reset/updateMissionState.
    const double yaw_base = has_last_command_
        ? last_command_.yaw_rate
        : state.yaw_rate;
    out.yaw_rate = clamp(intent.yaw_rate, yaw_base - dyr,
                         yaw_base + dyr);
    out.yaw_rate = clamp(out.yaw_rate, -p_.lp_max_yaw_rate,
                         p_.lp_max_yaw_rate);
    // The vertical channel is intentionally NOT reachable here 鈥?it is
    // owned by VerticalController / CommandComposer3D.
    return out;
}

Vec2d LocalPlanner30Hz::bodyVelocity(const PlanarState& state) const {
    const double c = std::cos(state.yaw), sn = std::sin(state.yaw);
    return Vec2d(c * state.velocity_world.x() + sn * state.velocity_world.y(),
                 -sn * state.velocity_world.x() + c * state.velocity_world.y());
}

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
//  Per-mission bookkeeping (v5)
// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
bool LocalPlanner30Hz::updateMissionState(const ResolvedPlanarTarget& target,
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
        last_command_ = VelocityCommand3D{};
        has_last_command_ = false;
    }
    last_mission_revision_ = target.mission_revision;
    last_target_position_ = target.position;
    last_target_valid_ = target.valid;
    return reset_memory;
}

VelocityCommand3D LocalPlanner30Hz::terminalIntent(
    const PlanarState& state, const ResolvedPlanarTarget& target) const {
    VelocityCommand3D cmd;
    const Vec2d to = target.position - state.position;
    const double dist = to.norm();
    if (dist <= 1e-9) return cmd;

    const double capture_radius = 0.5 * p_.task_goal_tolerance;
    const double remaining = std::max(0.0, dist - capture_radius);
    const double proportional = p_.lp_terminal_speed_gain * remaining;
    const double braking =
        std::sqrt(std::max(0.0, 2.0 * p_.lp_eff_accel_mps2 * remaining));
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

PlannerResult LocalPlanner30Hz::computePlan(const PlanarState& state,
                                            const LocalObservation& obs,
                                            const ResolvedPlanarTarget& target,
                                            bool mutate) {
    PlannerResult res;
    res.failure_reason = FailureReason::NONE;
    res.target_mission_revision = target.mission_revision;
    res.handoff_clearance_m = handoffClearance();

    if (!target.valid) {
        res.failure_reason = FailureReason::TARGET_OUTSIDE_FOV;
        res.planner_status = PlannerStatus::NO_TARGET;
        if (mutate) {
            last_command_ = VelocityCommand3D{};
            has_last_command_ = true;
            updateMissionState(target, res);
        }
        return res;
    }

    if (mutate) {
        const bool mission_changed =
            target.mission_revision != last_mission_revision_;
        // Preserve the current command: updateMissionState() clears it on any
        // reset, but a SAME-MISSION target discontinuity (macro correction
        // enter/exit) must NOT snap it to the state 鈥?that produced a
        // one-tick command jump (measured -0.48 m/s at a correction exit 鈫?
        // 14.5 m/s^2 feedforward pulse 鈫?tilt slam).  Keep the leading
        // command and let reachableCommand ramp toward the new plan at
        // lp_max_accel.  Fresh missions re-seed from state below.
        const VelocityCommand3D prev_cmd = last_command_;
        const bool had_cmd = has_last_command_;
        const bool reset_memory = updateMissionState(target, res);
        if (reset_memory) {
            if (mission_changed) {
                // Fresh task: seed the command-ramp base at the CURRENT
                // state velocity so the command does not jump.
                const Vec2d vb = bodyVelocity(state);
                last_command_ = VelocityCommand3D{vb.x(), vb.y(), 0.0,
                                                  state.yaw_rate};
                has_last_command_ = true;
            } else if (had_cmd) {
                // Same-mission target jump: keep the old command, ramp to
                // the new plan smoothly.
                last_command_ = prev_cmd;
                has_last_command_ = true;
            }
            // if !had_cmd 鈫?leave state-pinned base (first tick after reset).
        }
    }

    const Vec2d to = target.position - state.position;
    const double bearing = wrapAngle(std::atan2(to.y(), to.x()) - state.yaw);
    res.target_bearing_error_deg = rad2deg(std::fabs(bearing));
    const double dist = to.norm();
    const double actual_fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
    res.local_target_distance = dist;
    const double exit_ang = deg2rad(p_.lp_turn_exit_deg);

    // Obstacle-proximity signal used to label ACTIVE AVOIDANCE in every
    // live branch.  (The v9 risk-based avoidance_active never made it into
    // the new B-spline core 鈥?it only fired in the planned-trajectory
    // branch on a >3掳 bearing deviation, so the drone could be visibly
    // steering around an obstacle with avoidance_active=false, starving
    // the local:avoidance distribution targets.)
    const double avoid_proximity_m = 2.0;
    auto nearObstacle = [&]() {
        return obs.minClearanceToOccupied(state.position, avoid_proximity_m) <
               avoid_proximity_m;
    };
    auto observedClearanceAtState = [&]() {
        const double search_r = std::max(
            p_.obs_range_m,
            clearanceSearchRadius(state.velocity_world.norm()));
        return obs.minClearanceToOccupied(state.position, search_r);
    };

    // Rotation is a two-phase manoeuvre: first remove translational motion,
    // then rotate with an exactly-zero horizontal command.  Previously the
    // planner set turn_mode immediately while reachableCommand was still
    // ramping a 2 m/s command toward zero, so "turn_to_target" travelled
    // more than one metre in several collected turn segments.
    auto rotationBrakeRequired = [&]() {
        const double measured_speed = state.velocity_world.norm();
        const double commanded_speed = has_last_command_
            ? std::hypot(last_command_.vx_body, last_command_.vy_body)
            : measured_speed;
        return std::max(measured_speed, commanded_speed) >
               p_.vehicle_stationary_speed_mps;
    };
    auto brakeBeforeRotation = [&]() {
        const VelocityCommand3D intent{0.0, 0.0, 0.0, 0.0};
        const VelocityCommand3D out = reachableCommand(state, intent);
        PlannerResult brake = res;
        brake.success = true;
        brake.turn_mode = false;
        brake.avoidance_active = nearObstacle();
        brake.min_observed_clearance = observedClearanceAtState();
        brake.vx_body = out.vx_body;
        brake.vy_body = out.vy_body;
        brake.yaw_rate = out.yaw_rate;
        brake.intent_vx_body = 0.0;
        brake.intent_vy_body = 0.0;
        brake.intent_yaw_rate = 0.0;
        brake.selected_output_speed_mps =
            std::hypot(out.vx_body, out.vy_body);
        brake.planner_status = PlannerStatus::SAFE_HOLD;
        brake.candidate_progress_qualified = false;
        brake.output_progress_qualified = false;
        brake.progress_qualified = false;
        brake.stationary_candidate_selected =
            brake.selected_output_speed_mps <= p_.lp_min_progress_speed_mps;
        brake.stationary_selection_reason = "brake_before_rotation";
        brake.failure_reason = FailureReason::NONE;
        if (mutate) {
            current_trajectory_ = PlanarTrajectory{};
            last_command_ = out;
            has_last_command_ = true;
        }
        return brake;
    };

    // normalized_distance == 1 is the public direction+distance contract's
    // reserved pure-rotation command.
    const bool pure_rotation_target =
        target.normalized_distance >= 1.0 - 1e-9;
    if (pure_rotation_target) {
        if (rotationBrakeRequired()) {
            return brakeBeforeRotation();
        }
        const bool bearing_pending = std::fabs(bearing) > exit_ang;
        const bool yaw_rate_pending =
            std::fabs(state.yaw_rate) > p_.lp_turn_exit_max_yaw_rate;
        bool rotation_pending = bearing_pending || yaw_rate_pending;
        // Spin guard (pure-rotation TURN directives, shared constants with
        // the turn-hysteresis branch below): the corrector issues bounded
        // turn steps while the drone rotates in place with no translational
        // progress, so the yaw rate never drops and the correction can
        // never exit.  Force a hold after a bounded rotation so the yaw
        // rate decays and the 5 Hz corrector can exit or re-plan.
        const uint64_t kMaxTurnTicks = 70;        // 2.3 s
        const uint64_t kTurnCooldownTicks = 60;   // 2.0 s hard-brake hold
        bool turn_forced_hold = false;
        if (mutate) {
            if (rotation_pending) {
                ++turn_ticks_;
                if (turn_ticks_ > kMaxTurnTicks) {
                    turn_forced_hold = true;
                    rotation_pending = false;
                    turn_ticks_ = 0;
                    turn_hold_until_ = plan_ticks_ + kTurnCooldownTicks;
                }
            } else {
                turn_ticks_ = 0;
            }
            if (plan_ticks_ < turn_hold_until_) {
                // Cooldown after a spin-guard hold: stay stopped so the
                // yaw rate decays and the corrector can exit/re-plan.
                rotation_pending = false;
            }
        }
        turn_hysteresis_active_ = rotation_pending;

        const double yaw_intent =
            (bearing_pending && !turn_forced_hold &&
             plan_ticks_ >= turn_hold_until_)
                ? turnYawRate(state, bearing)
                : 0.0;
        const VelocityCommand3D intent{0.0, 0.0, 0.0, yaw_intent};
        VelocityCommand3D out = reachableCommand(state, intent);
        // reachableCommand now ramps relative to the previous yaw command,
        // so the spin-guard hold decelerates smoothly without a hard jump.
        res.success = true;
        res.turn_mode = rotation_pending;
        res.avoidance_active = rotation_pending && nearObstacle();
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
            current_trajectory_ = PlanarTrajectory{};
            last_command_ = out;
            has_last_command_ = true;
        }
        return res;
    }

    // TURN hysteresis.  Exit as soon as the target enters the physical FOV;
    // the regular command ramp then damps any residual yaw rate while local
    // navigation resumes.
    if (dist <= p_.task_goal_tolerance) {
        turn_hysteresis_active_ = false;
        turn_ticks_ = 0;
    } else if (turn_hysteresis_active_) {
        // Stop rotating as soon as the target enters the physical camera FOV.
        const bool bearing_ok = std::fabs(bearing) <= actual_fov_half;
        if (bearing_ok) {
            turn_hysteresis_active_ = false;
            turn_ticks_ = 0;
        }
    } else if (plan_ticks_ >= turn_hold_until_) {
        if (std::fabs(bearing) > actual_fov_half + 1e-9) {
            turn_hysteresis_active_ = true;
            turn_ticks_ = 0;
        }
    }

    // Braking prepares a turn but must not consume its bounded rotation
    // budget.  A slow physical stop should not trip the spin guard before
    // the first yaw command is emitted.
    if (turn_hysteresis_active_ && rotationBrakeRequired()) {
        return brakeBeforeRotation();
    }

    if (turn_hysteresis_active_ && mutate) {
        // Spin guard: never let the pure-rotation turn run on forever.
        // When the corrected target is unreachable the bearing can stay
        // large and the drone would spin in place at max yaw rate with no
        // way out; force-release after a bounded turn (~2.3 s at 30 Hz,
        // enough for any single <=200掳 turn at max yaw rate) into the
        // normal brake/hold path so the 5 Hz corrector can re-plan (stall
        // detection refreshes the target).
        const uint64_t kMaxTurnTicks = 70;        // 2.3 s
        const uint64_t kTurnCooldownTicks = 60;   // 2.0 s hard-brake hold
        ++turn_ticks_;
        if (turn_ticks_ > kMaxTurnTicks) {
            turn_hysteresis_active_ = false;
            turn_ticks_ = 0;
            turn_hold_until_ = plan_ticks_ + kTurnCooldownTicks;
        }
    }

    if (turn_hysteresis_active_) {
        res.success = true;
        res.turn_mode = true;
        res.avoidance_active = nearObstacle();
        const VelocityCommand3D intent{
            0.0, 0.0, 0.0, turnYawRate(state, bearing)};
        const VelocityCommand3D out = reachableCommand(state, intent);
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
            current_trajectory_ = PlanarTrajectory{};
            last_command_ = out;
            has_last_command_ = true;
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

    // 鈹€鈹€ Local corridor assessment (v7) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    {
        bool corridor_blocked = false;
        double first_dist = std::numeric_limits<double>::quiet_NaN();
        CorridorBlockReason cbr = CorridorBlockReason::CLEAR;
        Vec2d first_block(0.0, 0.0);
        uint32_t first_age = 0;
        bool risk_near = false;
        assessLocalCorridor(state, obs, target, corridor_blocked, first_dist,
                            cbr, first_block, first_age, risk_near);
        res.local_corridor_blocked = corridor_blocked;
        res.first_blocking_obstacle_distance = first_dist;
        res.corridor_block_reason = cbr;
        res.first_block_x = first_block.x();
        res.first_block_y = first_block.y();
        res.first_block_age_ticks = first_age;
        res.risk_corridor_near_obstacle = risk_near;
    }

    // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
    //  NEW CORE (user redesign): yaw-first + FOV-boundary B-spline planner
    // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
    //  1) TURN FIRST — pure-rotate ONLY when the target direction is outside
    //     the actual camera FOV (±45° for a 90° FOV): inside the FOV the
    //     target is visible, so the drone starts
    //     moving directly from standstill (no crabbing; the pure-pursuit
    //     yaw tracking aligns the nose while driving).
    //  2) SUBGOAL 鈥?the target when it is inside the FOV (endpoint speed 0,
    //     full stop), else a FOV-boundary point from the segmented FOV
    //     toward the target (endpoint speed = cruise 2 m/s; unknown beyond
    //     the FOV ASSUMED PASSABLE 鈥?no hard not-known-free gate).
    //  3) TIME-PARAMETERIZED CUBIC B-SPLINE from the current state to the
    //     subgoal, staying inside the FOV, clearance against the observed
    //     OCCUPIED ESDF (UNKNOWN passable), dynamics-feasible speed profile
    //     (command-ramp accel).  Re-planned every 30 Hz tick; the executed
    //     command goes through reachableCommand (smooth + dynamics-feasible).
    //  4) If no path exists, brake safely.  The 5 Hz planner independently
    //     previews this same local planning contract before deciding whether
    //     to pass through or correct the target.
    {
        const Vec2d to_t = target.position - state.position;
        const double tdist = to_t.norm();
        const double b_t =
            wrapAngle(std::atan2(to_t.y(), to_t.x()) - state.yaw);
        const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
        // The target visibility decision uses the actual camera FOV.  Keep a
        // smaller planning band below for path samples near the image edge,
        // but do not rotate a target merely because it is inside that band.
        const double target_fov_half = fov_half;
        const double planning_fov_half = std::max(
            0.0, fov_half - deg2rad(p_.te_turn_ray_margin_deg));
        // Inside the actual FOV the drone can start moving from standstill;
        // the
        // pure-pursuit yaw tracking aligns the nose while driving.
        const double range = p_.obs_range_m - 0.5 * p_.obs_resolution;

        // (1) turn first (only outside the actual camera FOV).  Skip when
        // already inside the goal tolerance: at the goal the bearing to it
        // flips with sub-tolerance position jitter, and chasing it would
        // keep the yaw rate above the terminal settle threshold forever
        // (terminal_yaw_rate episode failures) 鈥?settle the heading instead.
        if (std::fabs(b_t) > target_fov_half + 1e-9 &&
            dist > p_.task_goal_tolerance) {
            // The generic turn hysteresis above owns the spin-guard
            // cooldown.  This defensive duplicate branch must respect that
            // hold as well; otherwise it immediately resumes rotating and
            // makes the guard ineffective.
            if (plan_ticks_ < turn_hold_until_) {
                return brakeBeforeRotation();
            }
            if (rotationBrakeRequired()) {
                return brakeBeforeRotation();
            }
            const VelocityCommand3D intent{
                0.0, 0.0, 0.0, turnYawRate(state, b_t)};
            const VelocityCommand3D out = reachableCommand(state, intent);
            res.success = true;
            res.turn_mode = true;
            res.vx_body = out.vx_body;
            res.vy_body = out.vy_body;
            res.yaw_rate = out.yaw_rate;
            res.intent_vx_body = intent.vx_body;
            res.intent_vy_body = intent.vy_body;
            res.intent_yaw_rate = intent.yaw_rate;
            res.selected_output_speed_mps =
                std::hypot(out.vx_body, out.vy_body);
            res.planner_status = PlannerStatus::TURNING;
            res.candidate_progress_qualified = false;
            res.output_progress_qualified = false;
            res.progress_qualified = false;
            res.stationary_candidate_selected =
                res.selected_output_speed_mps <= p_.lp_min_progress_speed_mps;
            res.stationary_selection_reason =
                res.stationary_candidate_selected ? "yaw_first_turn" : "";
            res.failure_reason = FailureReason::NONE;
            if (mutate) {
                current_trajectory_ = PlanarTrajectory{};
                last_command_ = out;
                has_last_command_ = true;
            }
            return res;
        }

        // (2) subgoal selection + (3) trajectory plan
        Vec2d endpoint;
        double v_end =
            std::min(p_.lp_cruise_speed_mps, p_.lp_max_speed);
        bool terminal = false;
        PlanarTrajectory plan;
        double min_clear = std::numeric_limits<double>::infinity();
        bool planned = false;
        // The planner always receives the full world target.  The original
        // goal becomes terminal below the training clip distance.  A fixed
        // upper-planner waypoint is explicitly marked fly-through, so label
        // clipping never accidentally turns it into a final stop.
        // Visibility is an angular property.  Do not classify a far-away
        // but angle-visible mission goal as "out of FOV" merely because its
        // range exceeds the local depth window; that case is the ordinary
        // (>4.5 m) receding-horizon navigation contract.
        const bool target_angle_in_fov =
            std::fabs(b_t) <= target_fov_half + 1e-9;
        const bool target_in_fov =
            tdist <= range + 1e-9 && target_angle_in_fov;
        // Bearing of the SUBGOAL actually selected (used to decide whether
        // the planner is actively avoiding vs going straight to the goal).
        double chosen_b = b_t;
        // 鈹€鈹€ Goal-capture stop (R20e) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // Within the capture tolerance the drone must STOP and settle the
        // heading, never route away.  The goal is already inside capture
        // range — when it sits outside the actual FOV, the angular visibility
        // check is false and the A*/scan routing produced a CRUISE plan to a far
        // endpoint, flying the drone PAST the goal and into an orbit
        // (measured task174: 0.36 m from the goal, 67掳 off-nose, cruise
        // plan to a 4.95 m endpoint -> 36 s orbit, goal_not_reached).
        // Brake toward the goal; goalReached() (needs speed <
        // vehicle_goal_stop_speed and |yaw_rate| <= turn_exit_max_yaw_rate)
        // fires once settled.
        //
        // R26: micro-approach zone.  In 0.4-0.8 m the exact-goal B-spline
        // and its 0.2/0.35 m pullbacks fail (a cell ~0.9 m beyond the goal
        // and/or the tight terminal geometry), so the planner reported
        // BLOCKED at 0.48 m and the macro started an infinite
        // PASS/TURN_RIGHT loop (task 65, ~36 s, 2458° of yaw).  When the
        // terminal target is close, in FOV and the straight path is
        // HARD-clear, skip the spline and drive the proportional terminal
        // controller directly to settle inside the tolerance.
        const bool micro_approach =
            target.terminal && dist <= p_.lp_terminal_micro_approach_m &&
            target_in_fov && microApproachSafe(state, obs, target.position);
        if (target.terminal &&
            (dist <= p_.task_goal_tolerance || micro_approach)) {
            const VelocityCommand3D ti = terminalIntent(state, target);
            const VelocityCommand3D out = reachableCommand(state, ti);
            res.success = true;
            res.turn_mode = false;
            res.vx_body = out.vx_body;
            res.vy_body = out.vy_body;
            res.yaw_rate = out.yaw_rate;
            res.intent_vx_body = ti.vx_body;
            res.intent_vy_body = ti.vy_body;
            res.intent_yaw_rate = ti.yaw_rate;
            res.selected_output_speed_mps =
                std::hypot(out.vx_body, out.vy_body);
            res.planner_status = PlannerStatus::TERMINAL_SETTLING;
            res.plan_terminal = true;
            res.plan_end_speed_mps = 0.0;
            res.plan_executed_speed_mps = res.selected_output_speed_mps;
            res.avoidance_active = nearObstacle();
            res.candidate_progress_qualified = false;
            res.output_progress_qualified = false;
            res.progress_qualified = false;
            res.stationary_candidate_selected =
                res.selected_output_speed_mps <=
                p_.lp_min_progress_speed_mps;
            res.stationary_selection_reason =
                res.stationary_candidate_selected ? "goal_capture_stop" : "";
            res.failure_reason = FailureReason::NONE;
            if (mutate) {
                current_trajectory_ = PlanarTrajectory{};
                last_command_ = out;
                has_last_command_ = true;
            }
            return res;
        }
        if (!target.terminal) {
            // Non-terminal target: A*-first local reactive routing.  Route
            // from the current position to the target and initialise the
            // B-spline along that
            // route, ending AT the guide (route.back()) 鈥?NOT at a ~3 m
            // horizon truncation and NOT a straight chord.  The straight-
            // direct branch is skipped: the guide is a near-term waypoint
            // the 5 Hz corrector places around big-scale blockers, so the
            // local planner's job is to follow the obstacle-free corridor
            // toward it.  The time-parameterised B-spline bends around
            // observed obstacles (real curvature), validated against the
            // 0.5 m clearance floor.  A* / plan failures fall through to
            // the bearing scan below (architecture C) — never a hard stop
            // merely because a temporary waypoint needs a local reroute.
            std::vector<Vec2d> route =
                routeAStar(obs, state.position, target.position,
                           /*max_expansions=*/6000,
                           /*fov_front_m=*/macroGuideFrontFovM(),
                           state.yaw);
            if (route.size() >= 2) {
                endpoint = route.back();
                v_end = std::min(p_.lp_cruise_speed_mps, p_.lp_max_speed);
                terminal = false;
                planned = planEgoOrStraightWithPath(
                    state, obs, endpoint, v_end, terminal, route, plan,
                    min_clear);
                if (planned) {
                    // Active-avoidance label: the A* route deviated from the
                    // direct target bearing.
                    chosen_b = wrapAngle(
                        std::atan2(endpoint.y() - state.position.y(),
                                   endpoint.x() - state.position.x()) -
                        state.yaw);
                }
            }
            if (!planned) {
                // Pullback retry (defence-in-depth): the guide cell can sit
                // just inside the 0.5 m A* inflation (a stale / occluded
                // observed cell, or a guide placed at the edge of the free
                // corridor), so routeAStar returns EMPTY and the plan would
                // otherwise fall straight to the far FOV-edge scan.  Retry
                // with the goal pulled back along the drone->guide line so
                // the drone still ADVANCES toward the guide through the free
                // corridor instead of drifting to a far boundary point (the
                // 30 Hz layer continuously replans toward the same fixed
                // upper waypoint until its lifecycle completes).
                const Vec2d to_guide = target.position - state.position;
                const double gd = to_guide.norm();
                const Vec2d gdir =
                    gd > 1e-9 ? to_guide / gd : Vec2d(1.0, 0.0);
                const double pbs[2] = {0.5, 0.9};
                for (double pb : pbs) {
                    const Vec2d pull = target.position - gdir * pb;
                    route = routeAStar(obs, state.position, pull,
                                       /*max_expansions=*/6000,
                                       /*fov_front_m=*/macroGuideFrontFovM(),
                                       state.yaw);
                    if (route.size() < 2) continue;
                    endpoint = route.back();
                    planned = planEgoOrStraightWithPath(
                        state, obs, endpoint, v_end, terminal, route, plan,
                        min_clear);
                    if (planned) {
                        chosen_b = wrapAngle(
                            std::atan2(endpoint.y() - state.position.y(),
                                       endpoint.x() - state.position.x()) -
                            state.yaw);
                        break;
                    }
                }
            }
        } else {
            // Terminal target: it is visible and closer than the 4.5 m
            // training clip distance, so plan a zero-speed arrival.
            if (target_in_fov) {
                endpoint = target.position;
                v_end = 0.0;
                terminal = true;
                planned = planEgoOrStraight(state, obs, endpoint, v_end, terminal,
                                            plan, min_clear);
                if (!planned) {
                    // Pullback retry: a stale / spurious observed cell at (or
                    // within clearance of) the goal cell makes the exact-goal
                    // stop plan invalid, and the drone then OVERSHOOTS the goal
                    // at cruise (measured: flew PAST the goal at 1.8 m/s into
                    // the region boundary -> truth boundary-brake rejection).
                    // Retry the stop with the endpoint pulled back by up to the
                    // goal tolerance (0.4 m) so goal capture still triggers.
                    const Vec2d to_goal_dir =
                        dist > 1e-9 ? to / dist : Vec2d(1.0, 0.0);
                    const double pullbacks[2] = {0.2, 0.35};
                    for (double pb : pullbacks) {
                        const Vec2d stop_pt = target.position - to_goal_dir * pb;
                        if (planEgoOrStraight(state, obs, stop_pt, 0.0,
                                              /*terminal=*/true, plan,
                                              min_clear)) {
                            endpoint = stop_pt;
                            planned = true;
                            break;
                        }
                    }
                }
                if (!planned) {
                    // No safe terminal trajectory exists.  Do not turn this
                    // into a successful straight-line terminal approach: the
                    // upper planner must see the failed handoff and issue a
                    // temporary bypass/search target.  A reachable zero
                    // command is still emitted so the vehicle decelerates
                    // smoothly while the 5 Hz planner takes over.
                    const VelocityCommand3D out = reachableCommand(
                        state, VelocityCommand3D{0.0, 0.0, 0.0, 0.0});
                    res.success = false;
                    res.turn_mode = false;
                    res.vx_body = out.vx_body;
                    res.vy_body = out.vy_body;
                    res.yaw_rate = out.yaw_rate;
                    res.intent_vx_body = 0.0;
                    res.intent_vy_body = 0.0;
                    res.intent_yaw_rate = 0.0;
                    res.selected_output_speed_mps =
                        std::hypot(out.vx_body, out.vy_body);
                    res.planner_status =
                        res.local_corridor_blocked
                            ? PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE
                            : PlannerStatus::NO_SAFE_CANDIDATE;
                    res.plan_terminal = true;
                    res.plan_end_speed_mps = 0.0;
                    res.plan_executed_speed_mps =
                        res.selected_output_speed_mps;
                    res.avoidance_active = nearObstacle();
                    res.candidate_progress_qualified = false;
                    res.output_progress_qualified = false;
                    res.progress_qualified = false;
                    res.stationary_candidate_selected =
                        res.selected_output_speed_mps <=
                        p_.lp_min_progress_speed_mps;
                    res.stationary_selection_reason =
                        res.stationary_candidate_selected
                            ? "terminal_plan_unavailable"
                            : "terminal_brake_for_handoff";
                    res.failure_reason =
                        res.local_corridor_blocked
                            ? FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE
                            : FailureReason::NO_SAFE_CANDIDATE;
                    res.min_observed_clearance = observedClearanceAtState();
                    if (!spaceToStop(state, obs, stoppingDistance(state))) {
                        res.emergency_brake = true;
                        res.planner_status = PlannerStatus::EMERGENCY_BRAKE;
                    }
                    if (mutate) {
                        current_trajectory_ = PlanarTrajectory{};
                        last_command_ = out;
                        has_last_command_ = true;
                    }
                    return res;
                }
            }
            if (!planned) {
                // 鈹€鈹€ ENDPOINT ROUTING (EGO-style, R20d) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
                // The local planner plans to ONE fixed endpoint.  When the
                // direct plan failed, the endpoint comes from a lightweight
                // A* route over the observed grid (the "global" route): A*
                // finds the opening at ANY angle, and the endpoint is the
                // route exit at the visible range (or the goal when the route
                // reaches it).  The old 卤35掳 bearing scan could miss a lateral
                // opening and dead-ended the drone; A* routes around blockers
                // regardless of their bearing.  The 5 Hz corrector's waypoint
                // is already routed, so A* merely re-confirms it around any
                // nearby blocker.
                const Vec2d route_goal = target.position;
                const std::vector<Vec2d> route =
                    routeAStar(obs, state.position, route_goal,
                               /*max_expansions=*/6000);
                if (route.size() >= 2) {
                    endpoint = route.back();
                    // The route exit is a fly-through point unless it reaches
                    // the terminal target.
                    const bool reached_goal =
                        (endpoint - target.position).norm() <=
                        1.5 * obs.resolution;
                    if (target.terminal && reached_goal) {
                        terminal = true;
                        v_end = 0.0;
                    } else {
                        // Fly-through: the endpoint is the A* route point at
                        // the receding horizon (~3 m along the route), NOT the
                        // full route end (the far clipped target).  Using the
                        // full route end made the B-spline depart STRAIGHT
                        // toward a blocker directly ahead (the obstacle sits
                        // inside the front) and fail validation -> the drone
                        // stalled with NO_SAFE_CANDIDATE (measured task30:
                        // 0.81 m from obstacle (6.51,12.31)r1.19, no plan for
                        // 8 s, macro fell into TURN_RIGHT spin).  The drone
                        // follows the route in ~3 m hops, re-planning every
                        // tick.
                        terminal = false;
                        v_end =
                            std::min(p_.lp_cruise_speed_mps, p_.lp_max_speed);
                        const double horizon =
                            std::max(2.0, std::min(3.0, v_end * 1.5));
                        const size_t hi = static_cast<size_t>(
                            std::min<double>(horizon / obs.resolution,
                                             route.size() - 1));
                        endpoint = route[hi];
                    }
                    planned = planEgoOrStraight(state, obs, endpoint, v_end,
                                                terminal, plan, min_clear);
                    if (planned) {
                        // Active-avoidance label: the A* route deviated from
                        // the direct target bearing.
                        chosen_b = wrapAngle(
                            std::atan2(endpoint.y() - state.position.y(),
                                       endpoint.x() - state.position.x()) -
                            state.yaw);
                    }
                }
            }
        }
        if (!planned) {
            // A* gave no route (start/goal cell occupied or grid
            // invalid) 鈫?legacy segmented horizontal FOV scan fallback:
            // scan bearings from the target direction outward; the first
            // clear segment is the subgoal.
            const double step =
                deg2rad(p_.macro_local_candidate_bearing_step_deg);
            const double b0 = clamp(b_t, -planning_fov_half,
                                    planning_fov_half);
            // Scan the full planning band (both safe FOV edges), not just
            // around b0: the free opening may lie far from the
            // target bearing (e.g. a diagonal gap or a lateral detour) and
            // the drone must be able to steer to either edge regardless of
            // where the goal points.  kMax covers both sides; clamping
            // makes duplicated edge bearings harmless.
            const int kMax = static_cast<int>(
                std::ceil(2.0 * planning_fov_half / step));
            for (int k = 0; k <= kMax && !planned; ++k) {
                for (int side = 0; side < 2 && !planned; ++side) {
                    const double sgn = (side == 0) ? 1.0 : -1.0;
                    if (k == 0 && side == 1) continue;
                    const double b = clamp(
                        b0 + sgn * static_cast<double>(k) * step,
                        -planning_fov_half, planning_fov_half);
                    endpoint =
                        state.position +
                        Vec2d(std::cos(state.yaw + b),
                              std::sin(state.yaw + b)) *
                            range;
                    planned = planEgoOrStraight(state, obs, endpoint, v_end,
                                                /*terminal=*/false, plan,
                                                min_clear);
                    if (planned) chosen_b = b;
                }
            }
        }

        if (planned) {
            // (4) execute: pure-pursuit along the planned B-spline reference
            //     (smooth, time-parameterized, dynamics-feasible); the nose
            //     tracks the reference heading (yaw control).
            const Vec2d look =
                plan.points.size() > 1 ? plan.points[1] : endpoint;
            const Vec2d d_to = look - state.position;
            const double heading =
                d_to.squaredNorm() > 1e-12 ? std::atan2(d_to.y(), d_to.x())
                                           : state.yaw;
            const double remaining = (endpoint - state.position).norm();
            // The planned profile may have been reduced by the multi-cruise
            // retry to pass a tight gap 鈥?never command faster than the
            // cruise level at which the plan was actually validated
            // (plan.cruise_mps is that level, stored by the planner).
            // NOTE: the speed must NOT be estimated from the plan
            // trajectory's LAST SEGMENT 鈥?the integration stops with a
            // partial final step (ss clamped at Ls), so
            // (points.back()-points[n-2])/dt swings between ~0 and cruise
            // every few ticks, pulsing the command between full brake and
            // full acceleration (measured: v_des 0.02鈫?.00鈫?.14... and the
            // command saturating the 2 m/s^2 ramp back and forth) 鈫?
            // sustained pitch surging, the visible "attitude unstable".
            const double v_plan =
                (plan.cruise_mps > 1e-6)
                    ? std::min(v_end, plan.cruise_mps)
                    : v_end;
            double v_des;
            if (terminal) {
                // Approach speed: cruise capped by the braking profile so
                // the drone stops AT the target (v_end=0 is the ENDPOINT
                // speed, not the approach speed).  Brake 0.3 m EARLY so
                // the actual stop lands inside the goal tolerance
                // (task_goal_tolerance = 0.4 m) instead of past it 鈥?a
                // safety margin that also covers transient tracking lag.
                const double rem_eff =
                    std::max(0.0, remaining - 0.3);
                v_des = std::min(
                    p_.lp_terminal_max_speed,
                    std::sqrt(std::max(
                        0.0, 2.0 * p_.lp_eff_accel_mps2 * rem_eff)));
            } else {
                v_des = v_plan;
            }
            // Belt-and-braces cap at the validated cruise level.
            if (plan.cruise_mps > 1e-6) {
                v_des = std::min(v_des, plan.cruise_mps);
            }
            // At the goal (already inside the tolerance) do NOT chase the
            // plan heading 鈥?with sub-tolerance position jitter the heading
            // error flips sign and the yaw would oscillate above the
            // terminal settle threshold forever.  Settle: yaw intent 0 and
            // let the brake profile stop the drone; goalReached() only
            // needs |yaw_rate| <= lp_turn_exit_max_yaw_rate.
            const double yaw_limit = terminal
                ? p_.lp_terminal_max_yaw_rate
                : p_.lp_max_yaw_rate;
            const double yaw_intent =
                (terminal && dist <= p_.task_goal_tolerance)
                    ? 0.0
                    : clamp(p_.lp_turn_k * wrapAngle(heading - state.yaw),
                            -yaw_limit, yaw_limit);
            const VelocityCommand3D intent{
                v_des, 0.0, 0.0, yaw_intent};
            const VelocityCommand3D out = reachableCommand(state, intent);
            res.success = true;
            res.turn_mode = false;
            res.vx_body = out.vx_body;
            res.vy_body = out.vy_body;
            res.yaw_rate = out.yaw_rate;
            res.intent_vx_body = intent.vx_body;
            res.intent_vy_body = intent.vy_body;
            res.intent_yaw_rate = intent.yaw_rate;
            res.selected_output_speed_mps =
                std::hypot(out.vx_body, out.vy_body);
            res.planner_status =
                terminal ? PlannerStatus::TERMINAL_SETTLING
                         : PlannerStatus::SAFE_PROGRESSING;
            res.selected = plan;
            res.plan_terminal = terminal;
            res.plan_end_speed_mps = v_end;
            res.plan_executed_speed_mps = v_des;
            res.min_observed_clearance = min_clear;
            // Active avoidance: the planner is steering around an obstacle
            // when the chosen subgoal deviates from the direct target
            // direction (the scan had to leave b0), the direct corridor to
            // the target is blocked, the planned B-spline detours from the
            // straight line to its subgoal (mid-path weaving), or an
            // observed OCCUPIED cell is within avoid_proximity_m of the
            // vehicle.  Feeds hierarchical_mode "local_avoidance" and the
            // blueprint local:avoidance coverage.  (The plain bearing-devia
            // -tion rule alone barely fired: the scan stops at the first
            // clear bearing so chosen_b often equals b_t even while the
            // path genuinely detours.)
            double plan_cross_track = 0.0;
            if (plan.valid && plan.points.size() >= 2) {
                const Vec2d pa = plan.points.front();
                const Vec2d pb = plan.points.back();
                const Vec2d ab = pb - pa;
                const double L2 = ab.squaredNorm();
                if (L2 > 1e-9) {
                    for (const Vec2d& q : plan.points) {
                        const double u = std::max(
                            0.0, std::min(1.0, (q - pa).dot(ab) / L2));
                        plan_cross_track =
                            std::max(plan_cross_track, (q - (pa + ab * u)).norm());
                    }
                }
            }

            // A valid spline is not sufficient when it is still the direct
            // chord through a known blocked corridor.  That used to be
            // reported as SAFE_PROGRESSING/PASS_THROUGH and allowed the
            // vehicle to keep moving at cruise speed into the blocker.  Only
            // accept a blocked corridor here when the selected plan actually
            // leaves the direct bearing or has a meaningful lateral bend.
            const bool direct_blocked_without_detour =
                res.local_corridor_blocked &&
                std::fabs(wrapAngle(chosen_b - b_t)) <= deg2rad(3.0) &&
                plan_cross_track <= 0.35;
            if (direct_blocked_without_detour) {
                const VelocityCommand3D brake = reachableCommand(
                    state, VelocityCommand3D{0.0, 0.0, 0.0, 0.0});
                res.success = false;
                res.vx_body = brake.vx_body;
                res.vy_body = brake.vy_body;
                res.yaw_rate = brake.yaw_rate;
                res.intent_vx_body = 0.0;
                res.intent_vy_body = 0.0;
                res.intent_yaw_rate = 0.0;
                res.selected_output_speed_mps =
                    std::hypot(brake.vx_body, brake.vy_body);
                res.avoidance_active = true;
                res.planner_status =
                    PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE;
                res.failure_reason =
                    FailureReason::BLOCKED_BY_OBSERVED_OBSTACLE;
                res.selected = PlanarTrajectory{};
                res.plan_terminal = false;
                res.plan_end_speed_mps = 0.0;
                res.plan_executed_speed_mps =
                    res.selected_output_speed_mps;
                res.candidate_progress_qualified = false;
                res.output_progress_qualified = false;
                res.progress_qualified = false;
                res.stationary_candidate_selected = true;
                res.stationary_selection_reason =
                    "blocked_direct_corridor_without_detour";
                res.min_observed_clearance = observedClearanceAtState();
                if (!spaceToStop(state, obs, stoppingDistance(state))) {
                    res.emergency_brake = true;
                    res.planner_status = PlannerStatus::EMERGENCY_BRAKE;
                }
                if (mutate) {
                    current_trajectory_ = PlanarTrajectory{};
                    last_command_ = brake;
                    has_last_command_ = true;
                }
                return res;
            }
            res.avoidance_active =
                std::fabs(wrapAngle(chosen_b - b_t)) > deg2rad(3.0) ||
                res.local_corridor_blocked || plan_cross_track > 0.35 ||
                nearObstacle();
            res.nominal_progress_m = 0.0;
            res.executable_progress_m =
                plan.valid ? (dist - (plan.points.empty()
                                          ? dist
                                          : (target.position -
                                             plan.points.back())
                                                .norm()))
                           : 0.0;
            res.safe_prefix_duration_s = plan.valid ? plan.t.back() : 0.0;
            res.candidate_progress_qualified =
                res.selected_output_speed_mps > p_.lp_min_progress_speed_mps;
            res.output_progress_qualified =
                res.candidate_progress_qualified;
            res.progress_qualified = res.output_progress_qualified;
            res.stationary_candidate_selected =
                res.selected_output_speed_mps <= p_.lp_min_progress_speed_mps;
            res.stationary_selection_reason =
                res.stationary_candidate_selected ? "fov_plan_stationary" : "";
            res.failure_reason = FailureReason::NONE;
            if (mutate) {
                current_trajectory_ = plan;
                last_command_ = out;
                has_last_command_ = true;
            }
            return res;
        }
        // (2.5) Keep aligning while re-planning.  The target direction is
        // inside the usable band (|b_t| <= 35掳) but no trajectory was found
        // on this tick.  Instead of braking immediately, keep turning the
        // nose toward the target and let the 30 Hz replan retry every tick 鈥?
        // rotating changes the observation and can reveal a usable opening
        // that the previous body frame could not see.  Only when the nose
        // is actually ALIGNED (|b_t| <= lp_turn_exit_deg, the same
        // alignment threshold as the turn hysteresis) and planning STILL
        // fails do we hand over to the 5 Hz macro takeover (escape rotation
        // / brake / macro TURN or sub-target).  This removes the 30-35掳
        // "dead zone" where the drone braked and waited for the macro even
        // though a little more rotation would have produced a plan.
        // A visible target must not trigger a pure-rotation retry.  If the
        // local planner cannot find a safe trajectory while the target is in
        // the actual FOV, report NO_SAFE_CANDIDATE so the slower macro planner
        // can issue a temporary correction target.
        if (!target_angle_in_fov && std::fabs(b_t) > exit_ang + 1e-9 &&
            dist > p_.task_goal_tolerance) {
            if (rotationBrakeRequired()) {
                return brakeBeforeRotation();
            }
            const VelocityCommand3D intent{
                0.0, 0.0, 0.0, turnYawRate(state, b_t)};
            const VelocityCommand3D out = reachableCommand(state, intent);
            res.success = true;
            res.turn_mode = true;
            res.avoidance_active = nearObstacle();
            res.vx_body = out.vx_body;
            res.vy_body = out.vy_body;
            res.yaw_rate = out.yaw_rate;
            res.intent_vx_body = intent.vx_body;
            res.intent_vy_body = intent.vy_body;
            res.intent_yaw_rate = intent.yaw_rate;
            res.selected_output_speed_mps =
                std::hypot(out.vx_body, out.vy_body);
            res.planner_status = PlannerStatus::TURNING;
            res.candidate_progress_qualified = false;
            res.output_progress_qualified = false;
            res.progress_qualified = false;
            res.stationary_candidate_selected =
                res.selected_output_speed_mps <=
                p_.lp_min_progress_speed_mps;
            res.stationary_selection_reason =
                res.stationary_candidate_selected ? "align_while_planning"
                                                  : "";
            res.failure_reason = FailureReason::NONE;
            if (mutate) {
                current_trajectory_ = PlanarTrajectory{};
                last_command_ = out;
                has_last_command_ = true;
            }
            return res;
        }
        // (3) FOV planning failed 鈫?the drone may be facing a dead end
        // whose opening lies OUTSIDE the usable band (e.g. a diagonal
        // corridor at 50-60掳, beyond the 卤35掳 scan and even the 卤45掳 FOV
        // edge).  Rotating toward the most-open FOV edge brings that
        // opening into the scan band; braking forever (previously all 15
        // bearing candidates fail 鈫?permanent NO_SAFE_CANDIDATE stall) is
        // only correct when the drone is genuinely boxed in.
        {
            const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
            const double step =
                deg2rad(p_.macro_local_candidate_bearing_step_deg);
            // Clear range along a body bearing: march until the path comes
            // within lp_min_clearance of an observed OCCUPIED cell (the
            // SAME hard-clearance test the trajectory planner uses, so the
            // "opening" found here is genuinely passable, not just
            // surface-free).
            auto clearRangeBody = [&](double b) -> double {
                const Vec2d dir(std::cos(state.yaw + b),
                                std::sin(state.yaw + b));
                const double mstep = p_.obs_resolution * 0.5;
                double range = 0.0;
                for (double d = mstep; d <= p_.obs_range_m + 1e-9;
                     d += mstep) {
                    const Vec2d p = state.position + dir * d;
                    if (obs.minClearanceToOccupied(p, handoffClearance()) <
                        handoffClearance()) {
                        break;
                    }
                    range = d;
                }
                return range;
            };
            // Best (longest clearance-clear) bearing across the FULL FOV.
            double best_b = 0.0, best_r = -1.0;
            for (double b = -fov_half; b <= fov_half + 1e-9; b += step) {
                const double r = clearRangeBody(b);
                if (r > best_r) {
                    best_r = r;
                    best_b = b;
                }
            }
            // Escape-rotate ONLY when the opening is outside the safe
            // planning band (a bearing inside
            // the band was tested and failed, so rotating toward it would
            // change nothing) AND there is meaningful free space there.
            const bool opening_at_edge =
                std::fabs(best_b) > planning_fov_half + 1e-9;
            // Do not turn merely to search for a wider edge while the mission
            // target is visible; this is a macro-planner takeover condition.
            if (!target_angle_in_fov && opening_at_edge &&
                best_r >= std::max(p_.lp_min_clearance, 0.5)) {
                if (rotationBrakeRequired()) {
                    return brakeBeforeRotation();
                }
                const VelocityCommand3D intent{
                    0.0, 0.0, 0.0, turnYawRate(state, best_b)};
                const VelocityCommand3D out = reachableCommand(state, intent);
                res.success = true;
                res.turn_mode = true;
                res.avoidance_active = true;  // escape-rotate past a blocker
                res.vx_body = out.vx_body;
                res.vy_body = out.vy_body;
                res.yaw_rate = out.yaw_rate;
                res.intent_vx_body = intent.vx_body;
                res.intent_vy_body = intent.vy_body;
                res.intent_yaw_rate = intent.yaw_rate;
                res.selected_output_speed_mps =
                    std::hypot(out.vx_body, out.vy_body);
                res.planner_status = PlannerStatus::TURNING;
                res.candidate_progress_qualified = false;
                res.output_progress_qualified = false;
                res.progress_qualified = false;
                res.stationary_candidate_selected =
                    res.selected_output_speed_mps <=
                    p_.lp_min_progress_speed_mps;
                res.stationary_selection_reason =
                    res.stationary_candidate_selected ? "fov_escape_rotate"
                                                      : "";
                res.failure_reason = FailureReason::NONE;
                if (mutate) {
                    current_trajectory_ = PlanarTrajectory{};
                    last_command_ = out;
                    has_last_command_ = true;
                }
                return res;
            }
            // Genuinely boxed in: brake safely.  The next 5 Hz assessment
            // will observe this planning outcome through an independent
            // preview and may issue a corrected target.
            res.success = false;
            res.failure_reason = FailureReason::NO_SAFE_CANDIDATE;
            res.planner_status = PlannerStatus::NO_SAFE_CANDIDATE;
            res.avoidance_active = true;  // brake because boxed in by obstacles
            const VelocityCommand3D brake =
                reachableCommand(state, VelocityCommand3D{0.0, 0.0, 0.0, 0.0});
            res.vx_body = brake.vx_body;
            res.vy_body = brake.vy_body;
            res.yaw_rate = brake.yaw_rate;
            res.intent_vx_body = 0.0;
            res.intent_vy_body = 0.0;
            res.intent_yaw_rate = 0.0;
            res.selected_output_speed_mps =
                std::hypot(brake.vx_body, brake.vy_body);
            res.candidate_progress_qualified = false;
            res.output_progress_qualified = false;
            res.progress_qualified = false;
            res.stationary_candidate_selected = true;
            res.stationary_selection_reason = "fov_plan_blocked";
            res.min_observed_clearance = observedClearanceAtState();
            if (!spaceToStop(state, obs, stoppingDistance(state))) {
                res.emergency_brake = true;
                res.planner_status = PlannerStatus::EMERGENCY_BRAKE;
            }
            if (mutate) {
                current_trajectory_ = PlanarTrajectory{};
                last_command_ = brake;
                has_last_command_ = true;
            }
            return res;
        }
    }

    // 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
}
LocalPlanner30Hz::ResolvedPlanarTarget LocalPlanner30Hz::resolveTarget(
    const PlanarState& state, const PlanarTarget& target) const {
    ResolvedPlanarTarget resolved;
    resolved.update_event = target.update_event;
    resolved.mission_revision = target.mission_revision;
    resolved.normalized_distance = target.normalized_distance;
    resolved.valid = target.valid();
    if (!resolved.valid) return resolved;

    resolved.position = target.position_world;

    const double terminal_distance_m =
        std::max(0.0, p_.obs_range_m -
                          std::max(0.0, p_.te_normal_distance_reserve_m));
    const bool pure_rotation = target.normalized_distance >= 1.0 - 1e-9;
    const double actual_distance =
        (target.position_world - state.position).norm();
    resolved.terminal =
        !pure_rotation && !target.flythrough &&
        actual_distance < terminal_distance_m - 1e-9;
    return resolved;
}

PlannerResult LocalPlanner30Hz::plan(const PlanarState& state,
                                     const LocalObservation& obs,
                                     const PlanarTarget& target) {
    ++plan_ticks_;
    const ResolvedPlanarTarget resolved = resolveTarget(state, target);
    PlannerResult res = computePlan(state, obs, resolved, /*mutate=*/true);
    // R25: limit-cycle window must see EVERY branch's actual outcome, so it
    // is updated here (computePlan has many early returns).
    updateLimitCycleWindow(state, resolved, res);
    return res;
}

void LocalPlanner30Hz::updateLimitCycleWindow(const PlanarState& state,
                                              const ResolvedPlanarTarget& target,
                                              PlannerResult& res) {
    const Vec2d to = target.position - state.position;
    const double d = to.norm();
    const double brg = std::fabs(wrapAngle(
        std::atan2(to.y(), to.x()) - state.yaw));
    const bool blocked =
        res.local_corridor_blocked ||
        res.failure_reason != FailureReason::NONE ||
        res.planner_status == PlannerStatus::SAFE_HOLD ||
        res.planner_status == PlannerStatus::NO_SAFE_CANDIDATE ||
        res.planner_status == PlannerStatus::BLOCKED_BY_OBSERVED_OBSTACLE;
    limit_cycle_dist_window_.push_back(d);
    limit_cycle_bearing_window_.push_back(brg);
    limit_cycle_blocked_window_.push_back(blocked ? 1 : 0);
    while (limit_cycle_dist_window_.size() > kLimitCycleWindowTicks) {
        limit_cycle_dist_window_.pop_front();
        limit_cycle_bearing_window_.pop_front();
        limit_cycle_blocked_window_.pop_front();
    }
    res.local_limit_cycle_detected = false;
    if (limit_cycle_dist_window_.size() >= kLimitCycleWindowTicks) {
        const double d0 = limit_cycle_dist_window_.front();
        const double d1 = limit_cycle_dist_window_.back();
        const double b0 = limit_cycle_bearing_window_.front();
        const double b1 = limit_cycle_bearing_window_.back();
        const bool dist_progress = (d0 - d1) >= kLimitCycleMinProgressM;
        const bool bearing_progress =
            (b0 - b1) >= deg2rad(kLimitCycleMinTurnProgressDeg);
        const size_t nb = static_cast<size_t>(std::count(
            limit_cycle_blocked_window_.begin(),
            limit_cycle_blocked_window_.end(), 1));
        const double blocked_ratio =
            static_cast<double>(nb) /
            static_cast<double>(limit_cycle_blocked_window_.size());
        // A window is a LIMIT CYCLE when the drone neither approaches the
        // target nor re-rotates toward it AND the overwhelming majority of
        // frames are blocked / hold.  This makes the 5 Hz corrector able
        // to force a takeover (new waypoint / search strategy).
        res.local_limit_cycle_detected =
            !(dist_progress || bearing_progress) &&
            blocked_ratio > kLimitCycleBlockedRatio;
    }
}

PreviewResult LocalPlanner30Hz::previewPlan(const PlanarState& state,
                                            const LocalObservation& obs,
                                            const PlanarTarget& target) const {
    // A fresh probe prevents an active correction/turn hysteresis episode
    // from biasing the upper planner's question: "can local planning handle
    // the original target now?"  The probe receives exactly the same local
    // information and the same two target channels as the real planner.
    LocalPlanner30Hz probe(p_);
    const auto res = probe.plan(state, obs, target);
    PreviewResult pv;
    pv.success = res.success;
    pv.turn_mode = res.turn_mode;
    pv.emergency_brake = res.emergency_brake;
    pv.plan_valid = res.selected.valid && res.selected.points.size() >= 2;
    pv.plan_terminal = res.plan_terminal;
    pv.progress_qualified = res.progress_qualified;
    pv.local_corridor_blocked = res.local_corridor_blocked;
    pv.avoidance_active = res.avoidance_active;
    pv.selected_output_speed_mps = res.selected_output_speed_mps;
    pv.min_observed_clearance_m = res.min_observed_clearance;
    pv.failure_reason = res.failure_reason;
    pv.planner_status = res.planner_status;
    return pv;
}

bool LocalPlanner30Hz::canBrakeSafely(const PlanarState& state,
                                      const LocalObservation& obs) const {
    return spaceToStop(state, obs, stoppingDistance(state));
}

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
//  Helpers
// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
bool LocalPlanner30Hz::currentTrajectoryBlocked(
    const PlanarState& state, const LocalObservation& obs,
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
                    if (distance < handoffClearance()) {
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

//  FOV-boundary / target B-SPLINE trajectory planner (user redesign)
// 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
// A clamped cubic B-spline (cubic B茅zier 鈥?C虏 smooth) reference path from
// the current state to `endpoint` (the target when terminal, else a
// FOV-boundary point), TIME-PARAMETERIZED by a dynamics-feasible speed
// profile (command-ramp acceleration limit):
//   terminal 鈫?decelerate to a FULL STOP at the endpoint (endpoint speed 0);
//   boundary 鈫?cruise up to the desired speed (lp_cruise_speed_mps).
// The path stays inside the FOV and clears observed OCCUPIED cells (the
// local ESDF); UNKNOWN is PASSABLE (no hard not-known-free gate).  The
// trajectory is re-planned every 30 Hz tick; the executed command goes
// through reachableCommand (command-ramped at lp_max_accel), so the
// control output is smooth and dynamics-feasible.
static Vec2d cubicBSplinePoint(const Vec2d& p0, const Vec2d& p1,
                               const Vec2d& p2, const Vec2d& p3, double u) {
    const double u1 = 1.0 - u;
    const double w0 = u1 * u1 * u1;
    const double w1 = 3.0 * u1 * u1 * u;
    const double w2 = 3.0 * u1 * u * u;
    const double w3 = u * u * u;
    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
}

bool LocalPlanner30Hz::planFovTrajectory(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_end, bool terminal,
    PlanarTrajectory& out, double& min_clear) const {
    out = PlanarTrajectory{};
    out.valid = false;
    min_clear = std::numeric_limits<double>::infinity();
    const Vec2d delta = endpoint - state.position;
    const double L = delta.norm();
    if (L < 1e-6) {
        out.valid = terminal;
        return terminal;
    }
    const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
    const double cruise_nom =
        std::min(p_.lp_cruise_speed_mps, p_.lp_max_speed);
    const double v_start = state.velocity_world.norm();
    const double a = p_.lp_eff_accel_mps2;

    // B-spline control points: depart along the endpoint direction for
    // boundary subgoals 鈥?the segment is validated over a SHORT receding
    // horizon, so the drone must be able to turn tightly (e.g. thread a
    // diagonal gap) instead of lazily arcing along its current velocity.
    // Terminal stops keep departing along the current velocity for a
    // smooth, decelerating approach to the goal.
    const Vec2d dir = delta / L;
    const Vec2d dep_dir =
        (terminal && v_start > p_.vehicle_stationary_speed_mps)
            ? state.velocity_world / v_start
            : dir;
    const Vec2d p0 = state.position;
    const Vec2d p3 = endpoint;
    const Vec2d p1 = p0 + dep_dir * (L / 3.0);
    const Vec2d p2 = p3 - dir * (L / 3.0);

    // Dense arc-length table of the spline (geometry is cruise-independent).
    const int N = 64;
    std::vector<Vec2d> pts(N);
    std::vector<double> s(N, 0.0);
    pts[0] = cubicBSplinePoint(p0, p1, p2, p3, 0.0);
    for (int i = 1; i < N; ++i) {
        const double u = static_cast<double>(i) / (N - 1);
        const Vec2d p = cubicBSplinePoint(p0, p1, p2, p3, u);
        s[i] = s[i - 1] + (p - pts[i - 1]).norm();
        pts[i] = p;
    }
    const double Ls = std::max(s[N - 1], 1e-6);

    auto pointAt = [&](double ss) -> Vec2d {
        if (ss <= 0.0) return pts[0];
        if (ss >= Ls) return pts[N - 1];
        for (int i = 1; i < N; ++i) {
            if (s[i] >= ss) {
                const double w =
                    (ss - s[i - 1]) / std::max(1e-9, s[i] - s[i - 1]);
                return pts[i - 1] + (pts[i] - pts[i - 1]) * w;
            }
        }
        return pts[N - 1];
    };

    // Multi-cruise retry: a tight gap (e.g. the 1.94 m blocker4鈥搈edium
    // corridor) is only passable at reduced speed because the dynamic
    // braking clearance grows with speed.  Try cruise levels from nominal
    // down to 1/32; the first clearance-valid, time-parameterized path wins.
    // The speed profile respects the command-ramp acceleration limit.
    const double cruise_cands[6] = {
        cruise_nom,        0.5 * cruise_nom, 0.25 * cruise_nom,
        0.125 * cruise_nom, 0.0625 * cruise_nom, 0.03125 * cruise_nom};
    for (double cruise : cruise_cands) {
        if (cruise <= 1e-6) continue;
        const double ve = terminal ? 0.0 : std::min(v_end, cruise);
        // 鈹€鈹€ Receding horizon 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        // Terminal stops validate the WHOLE spline (the goal is inside the
        // visible range and the drone must stop there).  Boundary subgoals
        // (FOV-boundary waypoint; the real goal is beyond perception)
        // validate a SHORT lookahead only: the 30 Hz replan advances the
        // horizon, so the drone weaves around obstacles (gaps, diagonal
        // corridors) instead of requiring one fully-clear straight path to
        // the 4.95 m perception edge 鈥?which almost never exists in
        // clutter and previously caused permanent NO_SAFE_CANDIDATE brakes.
        const double lookahead =
            std::max(2.5, std::min(3.0, cruise * 1.5));
        const double L_check = terminal ? Ls : std::min(Ls, lookahead);
        // Time parameterization: v(s) = min(v_acc(s), v_dec(s)).
        auto speedAt = [&](double ss) -> double {
            const double v_acc = std::min(
                cruise,
                std::sqrt(std::max(0.0, v_start * v_start + 2.0 * a * ss)));
            const double rem = Ls - ss;
            const double v_dec = std::min(
                cruise, std::sqrt(std::max(0.0, ve * ve + 2.0 * a * rem)));
            return std::min(v_acc, v_dec);
        };

        PlanarTrajectory cand;
        cand.points.push_back(state.position);
        cand.yaw.push_back(state.yaw);
        cand.t.push_back(0.0);
        double ss = 0.0, cmin = std::numeric_limits<double>::infinity();
        Vec2d prev = state.position;
        bool reached = false, bad = false;
        for (double t_ctrl = p_.lp_dt;
             t_ctrl <= p_.lp_horizon_s + 1e-6 && !reached; t_ctrl += p_.lp_dt) {
            // Advance the arc by the distance covered over dt.  At
            // standstill (v_start = 0) speedAt(0) = 0, so a plain
            // `ss += speedAt(ss)*dt` Euler step would DEADLOCK: the drone
            // never leaves ss = 0, `reached` never fires and every cruise
            // level fails -> permanent NO_SAFE_CANDIDATE (every preflight
            // and every takeoff was stalling).  Include the constant-
            // acceleration travel term 0.5*a*dt^2 so the first interval
            // covers the ramp-up from rest; at cruise it is a negligible
            // (conservative, slightly-shorter) over-advance.
            const double v_cur = speedAt(ss);
            ss = std::min(Ls, ss + v_cur * p_.lp_dt +
                               0.5 * a * p_.lp_dt * p_.lp_dt);
            const Vec2d p = pointAt(ss);
            // 鈹€鈹€ FOV constraint: the trajectory must stay inside the FOV 鈹€鈹€
            const double b_p = wrapAngle(
                std::atan2(p.y() - state.position.y(),
                           p.x() - state.position.x()) -
                state.yaw);
            if (std::fabs(b_p) > fov_half + 1e-9) {
                bad = true;
                break;
            }
            // 鈹€鈹€ Clearance vs observed OCCUPIED (ESDF); UNKNOWN passable 鈹€鈹€
            const Vec2d tangent =
                (p - prev).squaredNorm() > 1e-12 ? (p - prev).normalized()
                                                 : dir;
            const double v = speedAt(ss);
            bool hard_violation = false;
            bool envelope_violation = false;
            const double search_r = clearanceSearchRadius(v);
            obs.forEachOccupiedWithin(
                p, search_r, [&](const Vec2d& centre, double distance) {
                    cmin = std::min(cmin, distance);
                    if (distance < handoffClearance())
                        hard_violation = true;
                    Vec2d dir_c = centre - p;
                    const double dl = dir_c.norm();
                    if (dl > 1e-9) dir_c /= dl;
                    const double closing =
                        std::max(0.0, dir_c.dot(tangent) * v);
                    if (distance < requiredClearance(closing))
                        envelope_violation = true;
                });
            if (hard_violation || envelope_violation) {
                bad = true;
                break;
            }
            prev = p;
            cand.points.push_back(p);
            cand.yaw.push_back(std::atan2(tangent.y(), tangent.x()));
            cand.t.push_back(t_ctrl);
            if (ss >= L_check - 1e-6) reached = true;
        }
        if (!bad && reached) {
            cand.valid = true;
            cand.cruise_mps = cruise;
            out = std::move(cand);
            min_clear = cmin;
            return true;
        }
    }
    return false;
}

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
//  EGO-style optimisation B-spline first, straight-ray fallback (R19)
// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
EgoBsplineOptimizer::Config LocalPlanner30Hz::egoConfig(const Params2D& p) {
    EgoBsplineOptimizer::Config c;
    c.lambda_smooth = p.ego_lambda_smooth;
    c.lambda_collision = p.ego_lambda_collision;
    c.lambda_feasibility = p.ego_lambda_feasibility;
    c.lambda_fitness = p.ego_lambda_fitness;
    c.lambda_fov = p.ego_lambda_fov;
    c.clearance_m = p.ego_clearance_m +
                    std::max(0.0, p.lp_clearance_discretization_margin_m);
    c.n_segments = std::max(4, p.ego_n_segments);
    c.ts = std::max(0.05, p.ego_ts);
    c.max_iter = std::max(1, p.ego_max_iter);
    c.max_vel = std::max(0.1, p.lp_cruise_speed_mps);
    c.max_acc = std::max(0.1, p.lp_max_accel);
    c.cruise_mps = std::max(0.1, p.lp_cruise_speed_mps);
    c.eff_accel_mps2 = std::max(0.1, p.lp_eff_accel_mps2);
    c.min_clearance = p.lp_min_clearance +
                      std::max(0.0, p.lp_clearance_discretization_margin_m);
    c.obstacle_reaction_time_s = p.lp_obstacle_reaction_time_s;
    c.soft_clearance_radius_m = p.lp_soft_clearance_radius_m;
    // Shared static handoff base; trajectory validation adds the reaction
    // and stopping terms using this value.
    c.handoff_clearance_m = p.lp_min_clearance +
                            std::max(0.0,
                                     p.lp_clearance_discretization_margin_m);
    c.obs_range_m = p.obs_range_m;
    c.obs_resolution = p.obs_resolution;
    c.obs_fov_deg = p.obs_fov_deg;
    c.lp_dt = p.lp_dt;
    c.horizon_s = p.lp_horizon_s;
    // The optimizer must use the physical camera FOV as its soft boundary.
    // The turn-ray margin is only a conservative sampling/certification
    // margin for the upper planner; using it here made a visible target at
    // 35--45 deg look unplannable and triggered unnecessary rotations.
    c.fov_half_rad = deg2rad(0.5 * p.obs_fov_deg);
    c.nearest_search_r = c.clearance_m + c.demarcation + 0.4;
    return c;
}

bool LocalPlanner30Hz::planEgoOrStraight(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_end, bool terminal,
    PlanarTrajectory& out, double& min_clear) const {
    if (p_.ego_enabled) {
        const EgoBsplineOptimizer::Config cfg = egoConfig(p_);
        PlanarTrajectory e =
            ego_bspline_.plan(state, obs, endpoint, v_end, terminal, cfg,
                              min_clear);
        if (e.valid) {
            out = std::move(e);
            return true;
        }
    }
    return planFovTrajectory(state, obs, endpoint, v_end, terminal, out,
                             min_clear);
}

bool LocalPlanner30Hz::planEgoOrStraightWithPath(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_end, bool terminal,
    const std::vector<Vec2d>& astar_path, PlanarTrajectory& out,
    double& min_clear) const {
    // USER ARCHITECTURE A+B: the EGO B-spline is initialised along the REAL
    // A* route (resampled), so the optimised curve follows the obstacle-free
    // corridor, curves around blockers and ends AT the endpoint (the guide).
    if (p_.ego_enabled) {
        const EgoBsplineOptimizer::Config cfg = egoConfig(p_);
        PlanarTrajectory e =
            ego_bspline_.plan(state, obs, endpoint, v_end, terminal, cfg,
                              min_clear, astar_path);
        if (e.valid) {
            out = std::move(e);
            return true;
        }
    }
    // Fallback (architecture C): straight validated plan toward the same
    // endpoint.  It will typically fail when the A* route was needed 鈥?
    // validation rejects a straight chord through a blocker 鈥?so the caller
    // then moves on to the bearing scan.
    return planFovTrajectory(state, obs, endpoint, v_end, terminal, out,
                             min_clear);
}

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
//  EGO-style A* routing (R20d)
// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
std::vector<Vec2d> LocalPlanner30Hz::routeAStar(
    const LocalObservation& obs, const Vec2d& start, const Vec2d& goal,
    int max_expansions, double fov_front_m, double yaw) const {
    std::vector<Vec2d> empty;
    if (!obs.valid() || obs.width <= 0 || obs.height <= 0) return empty;
    const double res = obs.resolution;
    const GridIndex2D s = worldToGrid(start, obs.origin, res);
    const GridIndex2D g = worldToGrid(goal, obs.origin, res);
    if (!obs.inGrid(s.ix, s.iy) || !obs.inGrid(g.ix, g.iy)) return empty;

    const size_t N = static_cast<size_t>(obs.width) * obs.height;
    const size_t W = static_cast<size_t>(obs.width);
    // 鈹€鈹€ FOV-front constraint (macro-guide routing) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    // The B-spline's front-3 m validation requires the executed front of
    // the curve to stay inside the FOV (obs_fov_deg/2).  A* ignores the
    // current heading, so for a guide placed BEHIND a blocker it routes
    // "over the top" (globally shortest but heading behind the drone,
    // e.g. -132掳 from the nose) 鈥?the B-spline then fails its FOV check
    // and the plan falls to the FOV-edge scan, which drags the drone into
    // the blocker's 0.5 m inflation band where no guide certifies anymore
    // and the corrector spins TURN_RIGHT (measured:
    // joint_v2_000000_316ed0e2, 6.5 s deadlock).  When enabled, cells
    // within `fov_front_m` of the start that lie outside the FOV cone get
    // a per-cell cost penalty so the route leaves through the VISIBLE
    // corridor (the B-spline front then passes validation and the drone
    // curves around the blocker at 0.5 m clearance).
    const double fov_half = 0.5 * deg2rad(p_.obs_fov_deg);
    const double fov_pen = 8.0 * res;
    auto fovFrontPenalty = [&](int ix, int iy) -> double {
        if (fov_front_m <= 0.0) return 0.0;
        const Vec2d c = gridCellCenter(ix, iy, obs.origin, res);
        const Vec2d rel = c - start;
        const double dist = rel.norm();
        if (dist > fov_front_m || dist < 1e-6) return 0.0;
        const double b = wrapAngle(
            std::atan2(rel.y(), rel.x()) - yaw);
        if (std::fabs(b) <= fov_half + 1e-9) return 0.0;
        return fov_pen;
    };
    // 鈹€鈹€ Clearance inflation (R20f) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    // The A* route must keep the hard clearance from OCCUPIED cells, not
    // just avoid the cells themselves: a route that threaded ~0.3 m from an
    // obstacle surface looked "free" to the A* but the B-spline front
    // validation then rejected it -> stall with NO_SAFE_CANDIDATE
    // (measured task30: 0.81 m from obstacle (6.51,12.31)r1.19, no plan
    // for 8 s).  Mark every cell within lp_min_clearance of an OCCUPIED
    // cell as blocked (a disk inflation).
    const double clr = std::max(handoffClearance(), 2.0 * res);
    const int cr = static_cast<int>(std::ceil(clr / res));
    std::vector<uint8_t> blocked_mask(N, 0);
    for (int iy = 0; iy < obs.height; ++iy) {
        for (int ix = 0; ix < obs.width; ++ix) {
            if (obs.at(ix, iy) != CellState::OCCUPIED) continue;
            const Vec2d oc = gridCellCenter(ix, iy, obs.origin, res);
            for (int dy = -cr; dy <= cr; ++dy) {
                for (int dx = -cr; dx <= cr; ++dx) {
                    const int nx = ix + dx, ny = iy + dy;
                    if (!obs.inGrid(nx, ny)) continue;
                    const Vec2d nc = gridCellCenter(nx, ny, obs.origin, res);
                    if ((nc - oc).norm() <= clr + 0.5 * res) {
                        blocked_mask[static_cast<size_t>(ny) * W + nx] = 1;
                    }
                }
            }
        }
    }
    auto blocked = [&](int ix, int iy) {
        return blocked_mask[static_cast<size_t>(iy) * W + ix] != 0;
    };
    if (blocked(s.ix, s.iy) || blocked(g.ix, g.iy)) return empty;

    // Sparse-ish arrays over the whole observed grid; reset only touched
    // cells via the parent/gcost write pattern (epoch not needed 鈥?we
    // allocate fresh and the A* is bounded, 30 Hz).
    std::vector<double> gcost(N, std::numeric_limits<double>::infinity());
    std::vector<int> parent(N, -1);
    std::vector<uint8_t> closed(N, 0);
    const double unknown_cost = 3.0 * res;  // UNKNOWN passable, penalised

    using Node = std::tuple<double, double, int, int>;  // f, g, ix, iy
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    const size_t sid = static_cast<size_t>(s.iy) * obs.width + s.ix;
    gcost[sid] = 0.0;
    open.emplace(0.0, 0.0, s.ix, s.iy);

    static const int kDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static const int kDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    static const double kCost[8] = {1.0, 1.0, 1.0, 1.0,
                                    std::sqrt(2.0), std::sqrt(2.0),
                                    std::sqrt(2.0), std::sqrt(2.0)};
    auto heuristic = [&](int ix, int iy) {
        const double dx = (ix - g.ix) * res;
        const double dy = (iy - g.iy) * res;
        return std::sqrt(dx * dx + dy * dy);
    };

    int best_ix = s.ix, best_iy = s.iy;
    double best_f = heuristic(s.ix, s.iy);
    int exp = 0;
    while (!open.empty() && exp < max_expansions) {
        const auto [f, gv, cx, cy] = open.top();
        open.pop();
        const size_t cid = static_cast<size_t>(cy) * obs.width + cx;
        if (closed[cid]) continue;  // stale heap entry
        closed[cid] = 1;
        ++exp;
        if (cx == g.ix && cy == g.iy) {
            best_ix = cx;
            best_iy = cy;
            break;
        }
        if (f < best_f) {
            best_f = f;
            best_ix = cx;
            best_iy = cy;
        }
        for (int k = 0; k < 8; ++k) {
            const int nx = cx + kDx[k], ny = cy + kDy[k];
            if (!obs.inGrid(nx, ny) || blocked(nx, ny)) continue;
            // No diagonal corner cutting.
            if (k >= 4 && (blocked(cx + kDx[k], cy) ||
                           blocked(cx, cy + kDy[k]))) {
                continue;
            }
            const size_t nid = static_cast<size_t>(ny) * obs.width + nx;
            if (closed[nid]) continue;
            const double extra =
                (obs.at(nx, ny) == CellState::UNKNOWN) ? unknown_cost : 0.0;
            const double fpen = fovFrontPenalty(nx, ny);
            const double ng = gv + kCost[k] * res + extra + fpen;
            if (ng < gcost[nid] - 1e-9) {
                gcost[nid] = ng;
                parent[nid] = static_cast<int>(cid);
                open.emplace(ng + heuristic(nx, ny), ng, nx, ny);
            }
        }
    }

    // Reconstruct from the goal cell when reached, else the best-reached
    // cell (the nearest opening the search found).
    std::vector<Vec2d> path;
    int ix = best_ix, iy = best_iy;
    while (ix >= 0 && iy >= 0) {
        path.push_back(gridCellCenter(ix, iy, obs.origin, res));
        const size_t id = static_cast<size_t>(iy) * obs.width + ix;
        const int p = parent[id];
        if (p < 0) break;
        ix = static_cast<int>(p % obs.width);
        iy = static_cast<int>(p / obs.width);
    }
    std::reverse(path.begin(), path.end());
    if (path.size() < 2) return empty;
    return path;
}

void LocalPlanner30Hz::assessLocalCorridor(
    const PlanarState& state, const LocalObservation& obs,
    const ResolvedPlanarTarget& target, bool& blocked,
    double& first_blocking_distance_m, CorridorBlockReason& reason,
    Vec2d& first_block, uint32_t& first_block_age_ticks,
    bool& risk_near_obstacle) const {
    blocked = false;
    first_blocking_distance_m = std::numeric_limits<double>::quiet_NaN();
    reason = CorridorBlockReason::CLEAR;
    first_block = Vec2d(0.0, 0.0);
    first_block_age_ticks = 0;
    risk_near_obstacle = false;
    const Vec2d to = target.position - state.position;
    const double dist = to.norm();
    if (dist < 1e-3) return;
    const Vec2d dir = to / dist;
    const double range = std::min(dist, p_.obs_range_m);
    const double step = p_.obs_resolution * 0.5;
    // R26: the HARD corridor uses handoffClearance() (the same collision
    // distance as the A* inflation / B-spline validation) and decides
    // whether the path is truly blocked.  The 1 m risk corridor is a SOFT
    // proximity signal only (risk_corridor_near_obstacle): it must NEVER
    // trigger macro takeover — a cell ~0.9 m BEYOND the goal was falsely
    // flagging the terminal corridor as blocked at 0.48 m (task 65) and
    // produced an infinite PASS/TURN_RIGHT loop.
    const double hard_corridor = handoffClearance();
    const double risk_corridor = p_.lp_risk_corridor_half_width;

    // R25: distinguish the blocking source.  A cell observed THIS tick
    // (age 0) is CURRENT-frame evidence; an older cell (age > 0) exists
    // only in the merged short-term history and may be stale — a phantom
    // blockage that the fresh FOV no longer confirms.
    const auto nearestOccupiedInCorridor = [&](const Vec2d& p, double radius,
                                               Vec2d& centre_out,
                                               uint32_t& age_out) {
        double best = radius;
        bool found = false;
        Vec2d bc(0.0, 0.0);
        uint32_t bage = 0;
        obs.forEachOccupiedWithin(
            p, radius, [&](const Vec2d& centre, double distance) {
                if (distance >= best) return;
                best = distance;
                found = true;
                bc = centre;
                const GridIndex2D g =
                    worldToGrid(centre, obs.origin, obs.resolution);
                if (obs.inGrid(g.ix, g.iy)) {
                    bage = obs.age_ticks[obs.idx(g.ix, g.iy)];
                }
            });
        if (found) {
            centre_out = bc;
            age_out = bage;
        }
        return found;
    };

    // Start cell must be known free (hard gate).
    if (!obs.isKnownFree(state.position.x(), state.position.y())) {
        blocked = true;
        reason = CorridorBlockReason::START_NOT_FREE;
        first_blocking_distance_m = 0.0;
        first_block = state.position;
        return;
    }
    // Target cell semantics: an OCCUPIED target is a hard block (the
    // planner cannot end a trajectory inside an obstacle).  An UNKNOWN
    // target cell is NOT a corridor block by itself — it is reported as a
    // diagnostic reason only when the rest of the corridor is clear, so
    // the CSV can distinguish "real obstacle at goal" from "goal cell
    // never observed free" without fabricating extra blockages.
    const CellState target_state =
        obs.atWorld(target.position.x(), target.position.y());
    if (target_state == CellState::OCCUPIED) {
        blocked = true;
        reason = CorridorBlockReason::TARGET_NOT_FREE;
        first_blocking_distance_m = dist;
        first_block = target.position;
        return;
    }
    for (double d = step; d <= range; d += step) {
        const Vec2d p = state.position + dir * d;
        Vec2d bc(0.0, 0.0);
        uint32_t bage = 0;
        if (nearestOccupiedInCorridor(p, hard_corridor, bc, bage)) {
            blocked = true;
            first_blocking_distance_m = d;
            first_block = bc;
            first_block_age_ticks = bage;
            reason = bage == 0 ? CorridorBlockReason::CURRENT_OCCUPIED
                               : CorridorBlockReason::HISTORY_OCCUPIED;
            return;
        }
        if (obs.minClearanceToOccupied(p, risk_corridor) < risk_corridor) {
            risk_near_obstacle = true;
        }
    }
    // Corridor clear but the target cell is unknown: report diagnostically
    // (blocked stays false).
    if (target_state == CellState::UNKNOWN &&
        dist <= p_.obs_range_m + 1e-9) {
        reason = CorridorBlockReason::UNKNOWN_TARGET;
    }
}

bool LocalPlanner30Hz::microApproachSafe(const PlanarState& state,
                                         const LocalObservation& obs,
                                         const Vec2d& goal) const {
    const Vec2d to = goal - state.position;
    const double dist = to.norm();
    if (dist < 1e-3) return true;
    const Vec2d dir = to / dist;
    const double step = p_.obs_resolution * 0.5;
    const double clr = handoffClearance();
    for (double d = step; d <= dist + 1e-6; d += step) {
        const Vec2d p = state.position + dir * d;
        if (obs.minClearanceToOccupied(p, clr) < clr) return false;
    }
    return true;
}

double LocalPlanner30Hz::stoppingDistance(const PlanarState& state) const {
    const double v = state.velocity_world.norm();
    if (v <= 1e-3) return 0.0;
    const double t = v / p_.lp_eff_accel_mps2;
    const double d = v * t - 0.5 * p_.lp_eff_accel_mps2 * t * t;
    return d + p_.lp_brake_stop_margin_m;
}

bool LocalPlanner30Hz::spaceToStop(const PlanarState& state,
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
    const double hard_clearance = handoffClearance();
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

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
//  Deterministic stall / control-oscillation detector (v5)
// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
}  // namespace expert
}  // namespace il_dataset
