#!/usr/bin/env python3
"""
il_dynamics.py  —  Flightmare Dynamics Backend  (Phase 4)

Provides:
  - DynamicsState dataclass
  - DynamicsBackend abstract interface
  - FlightmareDynamicsBackend: uses real Flightmare quadrotor dynamics
  - VelocityYawRateController: converts velocity/yaw-rate commands
    to low-level control inputs

Flightmare dynamics is the only supported execution backend.
"""

from __future__ import print_function, division

import math, time
import numpy as np
from dataclasses import dataclass, field

import rospy

from il_common import (world_vector_to_body_flu,
                       world_vector_to_body_flu_quat,
                       body_flu_to_flightlib_body)


# ============================================================================
#  DynamicsState
# ============================================================================

@dataclass
class DynamicsState:
    """Full dynamics state from the backend."""
    position_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    quaternion_world_body: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0],
                                         dtype=np.float64))

    velocity_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    velocity_flu: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))

    angular_velocity_body: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    acceleration_world: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    acceleration_flu: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))

    simulation_time_s: float = 0.0

    collision: bool = False


# ============================================================================
#  DynamicsBackend (abstract)
# ============================================================================

class DynamicsBackend:
    """Abstract interface for dynamics simulation."""

    def reset(self, initial_position, initial_yaw,
              initial_velocity=np.zeros(3), initial_angular_velocity=np.zeros(3)):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def step_velocity_command(self, velocity_command_flu, yaw_rate_command,
                               duration_s):
        """Apply velocity/yaw-rate command for duration_s seconds.

        Returns:
            True if dynamics stepped successfully.
        """
        raise NotImplementedError

    def restore_state_keep_motors(
            self, position, yaw, velocity=None, angular_velocity=None):
        """Restore the rigid-body state after actuator warm-up."""
        raise NotImplementedError

    def close(self):
        pass

    @property
    def backend_name(self):
        raise NotImplementedError


# ============================================================================
#  Flightmare Dynamics Backend
# ============================================================================

class FlightmareDynamicsBackend(DynamicsBackend):
    """Adapter for the repository's real ``flightlib::Quadrotor``.

    ``_flightmare_dynamics.FlightmareDynamics`` is a deliberately small
    standalone pybind wrapper around ``Quadrotor.reset(QuadState)``,
    ``Quadrotor.run(Command, dt)`` and ``Quadrotor.getState`` — it does
    NOT pull in the old expert stack.  Failure to import that compiled
    bridge is fatal; production collection never falls back to kinematics.
    """

    def __init__(self, config):
        dyn_cfg = config.get("global", {}).get("dynamics", {})
        vc_cfg = dyn_cfg.get("velocity_controller", {})

        self._sim_hz = float(dyn_cfg.get("simulation_hz", 200.0))
        self._control_hz = float(dyn_cfg.get("control_hz", 50.0))
        self._render_hz = float(dyn_cfg.get("render_hz", 20.0))
        self._control_mode = str(dyn_cfg.get("control_mode", "velocity_yaw_rate"))
        self._deterministic = bool(dyn_cfg.get("deterministic_time", True))
        if not bool(vc_cfg.get("use_existing_flightmare_controller", False)):
            raise RuntimeError(
                "The real Flightlib body-rate controller is required")

        self._settle_time_s = float(dyn_cfg.get("reset", {}).get("settle_time_s", 0.30))
        self._vel_noise_std = np.array(
            dyn_cfg.get("reset", {}).get("initial_velocity_noise_std_mps", [0, 0, 0]),
            dtype=np.float64)
        self._ang_vel_noise_std = np.array(
            dyn_cfg.get("reset", {}).get("initial_angular_velocity_noise_std_rps", [0, 0, 0]),
            dtype=np.float64)

        # Controller gains
        self._kp_vel = np.array(vc_cfg.get("kp_velocity", [3.0, 3.0, 3.0]),
                                dtype=np.float64)
        self._ki_vel = np.array(vc_cfg.get("ki_velocity", [0.0, 0.0, 0.0]),
                                dtype=np.float64)
        self._kd_vel = np.array(vc_cfg.get("kd_velocity", [0.2, 0.2, 0.2]),
                                dtype=np.float64)
        self._max_accel = np.array(
            vc_cfg.get("maximum_acceleration_mps2", [4.0, 4.0, 2.0]),
            dtype=np.float64)
        self._max_tilt_deg = float(vc_cfg.get("maximum_tilt_deg", 35.0))
        self._max_yaw_rate = float(vc_cfg.get("maximum_yaw_rate_rps", 1.5))
        self._max_yaw_acceleration = float(
            vc_cfg.get("maximum_yaw_acceleration_rps2", 4.0))
        self._integrator_limit = np.array(
            vc_cfg.get("integrator_limit", [1.0, 1.0, 0.5]),
            dtype=np.float64)

        try:
            from _flightmare_dynamics import FlightmareDynamics
            self._quad_dynamics = FlightmareDynamics()
        except Exception as e:
            raise RuntimeError(
                "Real flightlib dynamics bridge is unavailable: {}. Rebuild "
                "il_dataset with flightlib; no production fallback is allowed."
                .format(e))

        # Controller state
        self._deriv_tau = float(
            vc_cfg.get("derivative_filter_tau_s", 0.0))
        self._controller = VelocityYawRateController(
            self._kp_vel, self._ki_vel, self._kd_vel,
            self._max_accel, self._max_tilt_deg, self._max_yaw_rate,
            self._integrator_limit, self._deriv_tau)
        self._attitude_gain = float(vc_cfg.get("attitude_gain", 6.0))
        # R29g: angular-rate damping on the attitude loop (rotorS / RPG
        # style).  body_rates = -att_gain*att_error - ang_rate_gain*omega:
        # the D-on-body-rate term adds the missing rate damping that their
        # dedicated rate controllers (rotorS rate_controller, rpg
        # body_rates_p 0.1..0.52) provide — without it the attitude error
        # is mapped straight to a body-rate command and the loop rings at
        # ~2 Hz.  0 disables.
        self._angular_rate_gain = float(
            vc_cfg.get("angular_rate_gain", 0.5))
        self._max_body_rate = float(vc_cfg.get("maximum_body_rate_rps", 6.0))
        self._last_state = None
        self._commanded_yaw_rate = 0.0

        rospy.loginfo("[Dynamics] Flightmare backend initialized (%s). "
                      "sim=%.0fHz ctrl=%.0fHz render=%.0fHz",
                      "flightlib::Quadrotor",
                      self._sim_hz, self._control_hz, self._render_hz)

    @property
    def backend_name(self):
        return "flightmare"

    def reset(self, initial_position, initial_yaw,
              initial_velocity=None, initial_angular_velocity=None):
        """Reset quadrotor to initial state."""
        if initial_velocity is None:
            initial_velocity = np.zeros(3, dtype=np.float64)
        if initial_angular_velocity is None:
            initial_angular_velocity = np.zeros(3, dtype=np.float64)

        self._controller.reset()
        self._commanded_yaw_rate = 0.0

        # Apply noise
        if np.any(self._vel_noise_std > 0):
            noise = np.random.randn(3) * self._vel_noise_std
            initial_velocity = np.asarray(initial_velocity, dtype=np.float64) + noise
        initial_angular_velocity = np.asarray(
            initial_angular_velocity, dtype=np.float64).copy()
        if np.any(self._ang_vel_noise_std > 0):
            initial_angular_velocity += np.random.randn(3) * self._ang_vel_noise_std

        half = 0.5 * float(initial_yaw)
        quaternion_wxyz = np.array(
            [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
        ok = self._quad_dynamics.reset(
            np.asarray(initial_position, dtype=np.float64), quaternion_wxyz,
            np.asarray(initial_velocity, dtype=np.float64),
            initial_angular_velocity)
        if not ok:
            raise RuntimeError("flightlib::Quadrotor::reset rejected the initial state")
        self._last_state = self.get_state()

        # Settle
        if self._settle_time_s > 0.0 and not self.step_velocity_command(
                np.zeros(3, dtype=np.float64), 0.0, self._settle_time_s):
            raise RuntimeError("Flightmare settle step failed")
        # reset() initializes every motor at zero speed.  The velocity
        # controller spins them up during settle, but it cannot recover the
        # altitude lost before hover thrust is available.  Restore the exact
        # requested pose without resetting the now-spinning motors.
        self.restore_state_keep_motors(
            initial_position, initial_yaw,
            initial_velocity, initial_angular_velocity)

    def restore_state_keep_motors(
            self, position, yaw, velocity=None, angular_velocity=None):
        if velocity is None:
            velocity = np.zeros(3, dtype=np.float64)
        if angular_velocity is None:
            angular_velocity = np.zeros(3, dtype=np.float64)
        half = 0.5 * float(yaw)
        quaternion_wxyz = np.array(
            [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
        ok = self._quad_dynamics.set_state_preserve_motors(
            np.asarray(position, dtype=np.float64), quaternion_wxyz,
            np.asarray(velocity, dtype=np.float64),
            np.asarray(angular_velocity, dtype=np.float64))
        if not ok:
            raise RuntimeError(
                "flightlib::Quadrotor::setState rejected the warm state")
        self._controller.reset()
        self._commanded_yaw_rate = 0.0
        self._last_state = self.get_state()
        return True

    def get_state(self):
        """Return current DynamicsState."""
        raw = np.asarray(self._quad_dynamics.state(), dtype=np.float64)
        if raw.shape != (26,) or not np.all(np.isfinite(raw)):
            raise RuntimeError("Invalid state returned by flightlib::Quadrotor")
        # raw = [time, p(3), quaternion(wxyz), v(3), w(3), a(3), ...]
        sim_time = float(raw[0])
        pos = raw[1:4].copy()
        q_wxyz = raw[4:8].copy()
        vel = raw[8:11].copy()
        angular = raw[11:14].copy()
        accel = raw[14:17].copy()
        q_xyzw = np.array(
            [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
        return DynamicsState(
            position_world=pos,
            quaternion_world_body=q_xyzw,
            velocity_world=vel,
            velocity_flu=world_vector_to_body_flu_quat(vel, q_xyzw),
            angular_velocity_body=angular,
            acceleration_world=accel,
            acceleration_flu=world_vector_to_body_flu_quat(accel, q_xyzw),
            simulation_time_s=sim_time)

    def step_velocity_command(self, velocity_command_flu, yaw_rate_command,
                               duration_s):
        """Apply velocity command for duration_s, stepping dynamics internally."""
        if duration_s <= 0:
            return True

        command = np.asarray(velocity_command_flu, dtype=np.float64)
        if command.shape != (3,) or not np.all(np.isfinite(command)):
            return False
        if not np.isfinite(yaw_rate_command):
            return False

        elapsed = 0.0
        dt_sim = 1.0 / self._sim_hz
        dt_ctrl = 1.0 / self._control_hz
        epsilon = 1e-9

        # Feedforward for this command interval: the expert holds the
        # command constant for duration_s (30 Hz), so the feedforward is the
        # ramp rate Δcmd/duration_s applied for the whole interval (not a
        # 1/50-per-tick pulse, which caused the 3 Hz tilt limit cycle).
        feedforward_hold = self._controller.begin_velocity_command(
            command, duration_s)

        while elapsed < duration_s - epsilon:
            ctrl_dur = min(dt_ctrl, duration_s - elapsed)

            state = self.get_state()
            yaw = self._yaw_from_xyzw(state.quaternion_world_body)
            accel_command_flu = self._controller.update(
                command, state.velocity_world, yaw, ctrl_dur,
                state.quaternion_world_body, feedforward_hold,
                float(state.angular_velocity_body[2]))
            target_yaw_rate = float(np.clip(
                yaw_rate_command,
                -self._max_yaw_rate, self._max_yaw_rate))
            maximum_yaw_delta = (
                max(0.0, self._max_yaw_acceleration) * ctrl_dur)
            yaw_delta = float(np.clip(
                target_yaw_rate - self._commanded_yaw_rate,
                -maximum_yaw_delta, maximum_yaw_delta))
            self._commanded_yaw_rate += yaw_delta

            ctrl_elapsed = 0.0
            while ctrl_elapsed < ctrl_dur - epsilon:
                step_dt = min(dt_sim, ctrl_dur - ctrl_elapsed)
                if not self._run_acceleration_step(
                        accel_command_flu,
                        self._commanded_yaw_rate, step_dt):
                    return False
                ctrl_elapsed += step_dt

            elapsed += ctrl_dur

        return True

    @staticmethod
    def _yaw_from_wxyz(q):
        w, x, y, z = q
        return math.atan2(2.0 * (w * z + x * y),
                          1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _yaw_from_xyzw(q):
        return FlightmareDynamicsBackend._yaw_from_wxyz(
            np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))

    @staticmethod
    def _rotation_from_wxyz(q):
        w, x, y, z = q / max(np.linalg.norm(q), 1e-12)
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _run_acceleration_step(self, accel_flu, yaw_rate, dt):
        raw = np.asarray(self._quad_dynamics.state(), dtype=np.float64)
        q_wxyz = raw[4:8]
        yaw = self._yaw_from_wxyz(q_wxyz)
        # body angular velocity (QuadState OME at x[10:13] -> raw[11:14])
        omega_body = raw[11:14]

        rotation = self._rotation_from_wxyz(q_wxyz)
        accel_world = rotation.dot(body_flu_to_flightlib_body(accel_flu))
        thrust_world = accel_world + np.array([0.0, 0.0, 9.81])
        thrust_norm = float(np.linalg.norm(thrust_world))
        if thrust_norm < 1e-6:
            return False
        b3_des = thrust_world / thrust_norm
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        b2_des = np.cross(b3_des, heading)
        b2_norm = float(np.linalg.norm(b2_des))
        if b2_norm < 1e-6:
            return False
        b2_des /= b2_norm
        b1_des = np.cross(b2_des, b3_des)
        desired_rotation = np.column_stack((b1_des, b2_des, b3_des))
        error_matrix = 0.5 * (
            desired_rotation.T.dot(rotation) - rotation.T.dot(desired_rotation))
        attitude_error = np.array([
            error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
        # R29g: angular-rate damping (rotorS / RPG style rate loop).
        # body_rates = -att_gain*attitude_error - ang_rate_gain*omega:
        # the measured body-rate term damps the roll/pitch ring without a
        # separate rate controller in flightlib.
        body_rates = (-self._attitude_gain * attitude_error -
                      self._angular_rate_gain * omega_body)
        body_rates[2] += float(np.clip(
            yaw_rate, -self._max_yaw_rate, self._max_yaw_rate))
        body_rates = np.clip(body_rates, -self._max_body_rate, self._max_body_rate)
        collective = max(0.0, float(np.dot(thrust_world, rotation[:, 2])))
        return bool(self._quad_dynamics.run(collective, body_rates, float(dt)))

    def close(self):
        pass


# ============================================================================
#  VelocityYawRateController
# ============================================================================

class VelocityYawRateController:
    """PID velocity controller converting velocity commands to accelerations.

    Uses FLU frame: forward=x, left=y, up=z.
    """

    def __init__(self, kp, ki, kd, max_accel, max_tilt_deg, max_yaw_rate,
                 integrator_limit, deriv_tau=0.0):
        self._kp = np.asarray(kp, dtype=np.float64)
        self._ki = np.asarray(ki, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._max_accel = np.asarray(max_accel, dtype=np.float64)
        self._max_tilt_rad = math.radians(max_tilt_deg)
        self._max_yaw_rate = float(max_yaw_rate)
        self._int_limit = np.asarray(integrator_limit, dtype=np.float64)
        # R29f: first-order low-pass time constant (s) for the derivative
        # term.  0 disables filtering.  kd acts on a measured-velocity
        # difference (control_hz), which is noisy; the noise pumps the
        # attitude loop and contributes to the ~2.2 Hz roll/pitch limit
        # cycle.  A tau ~ 0.05 s (cut-off ~3 Hz) keeps real damping while
        # removing the high-frequency measurement noise.
        self._deriv_tau = float(deriv_tau)

        self._integrator = np.zeros(3, dtype=np.float64)
        self._prev_velocity_world = None
        self._prev_desired_vel_flu = None
        self._prev_command_flu = None
        self._deriv_filtered = None

    def reset(self):
        self._integrator = np.zeros(3, dtype=np.float64)
        self._prev_velocity_world = None
        self._prev_desired_vel_flu = None
        self._prev_command_flu = None
        self._deriv_filtered = None

    def begin_velocity_command(self, desired_vel_flu, duration_s):
        """Feedforward for one piecewise-constant velocity command interval.

        The expert updates the command at 30 Hz, so ``desired_vel_flu`` is
        held for ``duration_s`` (~1/30 s).  The correct velocity feedforward
        is the command ramp rate (Δcmd / duration_s) applied for the WHOLE
        interval.  Dividing Δcmd by the per-tick control dt (1/50) instead
        pulses the feedforward at 1.67x the true rate; combined with the P
        term it then clips at ``maximum_acceleration`` and drives a
        bang-bang tilt limit cycle (~3 Hz continuous "nodding", seen in the
        joint_v2 command-ramp build).

        Returns the feedforward vector to hold for this interval (already
        clipped to ``max_accel``).  Returns zero on the first call after a
        reset (no previous command to differentiate against).
        """
        desired = np.asarray(desired_vel_flu, dtype=np.float64)
        if self._prev_command_flu is None:
            self._prev_command_flu = desired.copy()
            return np.zeros(3, dtype=np.float64)
        feedforward = (desired - self._prev_command_flu) / max(
            float(duration_s), 1e-9)
        feedforward = np.clip(feedforward, -self._max_accel,
                              self._max_accel)
        self._prev_command_flu = desired.copy()
        return feedforward

    def update(self, desired_vel_flu, current_vel_world, current_yaw, dt,
               current_quaternion_xyzw=None, feedforward_hold=None,
               yaw_rate=None):
        """Compute acceleration command from velocity error.

        Args:
            desired_vel_flu: [vx_fwd, vy_left, vz_up] desired in FLU.
            current_vel_world: [vx, vy, vz] current velocity in world.
            current_yaw: current yaw angle.
            dt: control time step.
            current_quaternion_xyzw: current attitude quaternion (xyzw).
            feedforward_hold: optional precomputed feedforward vector to
                hold for the whole command interval (see
                ``begin_velocity_command``).  When provided it replaces the
                per-tick feedforward.
            yaw_rate: current body yaw rate (rad/s).  When provided, a
                coordinated-turn centripetal feedforward v_fwd * yaw_rate is
                added to the lateral acceleration so the velocity vector
                rotates with the body during turns (see the body below).

        Returns:
            accel_cmd_flu: [ax, ay, az] acceleration command in FLU.
        """
        # Convert current world velocity to FLU
        current_flu = (world_vector_to_body_flu_quat(
            current_vel_world, current_quaternion_xyzw)
            if current_quaternion_xyzw is not None else
            world_vector_to_body_flu(current_vel_world, current_yaw))

        desired_vel = np.asarray(desired_vel_flu, dtype=np.float64)
        error = desired_vel - current_flu

        # Velocity feedforward.  The expert updates the command at 30 Hz and
        # begin_velocity_command() computes the correct ramp rate
        # (Δcmd / command_period) ONCE per command interval; we hold it for
        # the whole interval so the vehicle tracks a 2 m/s^2 ramp without a
        # large P gain.  (The old per-tick form divided Δcmd by the 1/50
        # control dt, pulsing the feedforward at 1.67x the true rate and
        # clipping at max_accel -> bang-bang tilt limit cycle ~±16-20 deg.)
        if feedforward_hold is not None:
            feedforward = np.asarray(feedforward_hold, dtype=np.float64)
        elif self._prev_desired_vel_flu is None:
            feedforward = np.zeros(3, dtype=np.float64)
        else:
            feedforward = (desired_vel - self._prev_desired_vel_flu) / max(
                dt, 1e-9)
            feedforward = np.clip(feedforward, -self._max_accel,
                                  self._max_accel)
        self._prev_desired_vel_flu = desired_vel.copy()

        # Integral with anti-windup
        self._integrator += error * dt
        self._integrator = np.clip(self._integrator, -self._int_limit,
                                   self._int_limit)

        # Derivative on measurement.  Differentiating error makes an immediate
        # emergency command change (for example moving -> zero in RECOVERY)
        # appear as physical acceleration and creates a derivative kick.
        # Differentiate world velocity first so yaw/body-frame rotation is not
        # mistaken for translational acceleration, then express it in FLU.
        current_velocity_world = np.asarray(
            current_vel_world, dtype=np.float64)
        if self._prev_velocity_world is None:
            derivative_raw = np.zeros(3, dtype=np.float64)
        else:
            measured_acceleration_world = (
                current_velocity_world - self._prev_velocity_world) / max(
                    dt, 1e-9)
            derivative_raw = -(
                world_vector_to_body_flu_quat(
                    measured_acceleration_world,
                    current_quaternion_xyzw)
                if current_quaternion_xyzw is not None else
                world_vector_to_body_flu(
                    measured_acceleration_world, current_yaw))
        # R29f: first-order low-pass on the derivative term (alpha =
        # dt/(tau+dt)).  Without it the D term amplifies the noisy
        # measured-velocity difference and pumps the attitude loop.
        if self._deriv_tau > 0.0:
            alpha = max(dt, 1e-6) / (self._deriv_tau + max(dt, 1e-6))
            if self._deriv_filtered is None:
                derivative = derivative_raw
            else:
                derivative = (alpha * derivative_raw +
                              (1.0 - alpha) * self._deriv_filtered)
            self._deriv_filtered = derivative.copy()
        else:
            derivative = derivative_raw
        self._prev_velocity_world = current_velocity_world.copy()

        # PID + feedforward
        accel = (self._kp * error +
                 self._ki * self._integrator +
                 self._kd * derivative +
                 feedforward)

        # Coordinated-turn centripetal feedforward.  While the drone yaws at
        # rate w with forward speed v, its velocity vector must rotate WITH
        # the body, which requires a lateral (body-y) acceleration of v*w.
        # Without it the turn is uncoordinated: the body-frame side-slip vy
        # grows at rate -w*v, and the lateral velocity loop (vy_cmd = 0) then
        # over-banks to fight it, producing a ~2.5 Hz roll oscillation (roll
        # swings to ~19 deg in the joint_v2 episodes).  Feeding v*w (same
        # sign: right turn -> bank right) cancels the side-slip rate and the
        # loop only trims the residual.
        if yaw_rate is not None:
            accel[1] += current_flu[0] * float(yaw_rate)

        # Limit acceleration
        accel = np.clip(accel, -self._max_accel, self._max_accel)

        # Tilt limit: horizontal acceleration should respect tilt constraint
        # a_horiz_max = g * tan(max_tilt)
        g = 9.81
        max_horiz_accel = g * math.tan(self._max_tilt_rad)
        horiz = math.sqrt(accel[0]**2 + accel[1]**2)
        if horiz > max_horiz_accel:
            scale = max_horiz_accel / max(horiz, 1e-9)
            accel[0] *= scale
            accel[1] *= scale

        return accel


# ============================================================================
#  Factory function
# ============================================================================

def create_dynamics_backend(config):
    """Create the sole supported Flightmare dynamics backend."""
    dyn_cfg = config.get("global", {}).get("dynamics", {})
    backend_name = str(dyn_cfg.get("backend", "flightmare"))
    if backend_name != "flightmare":
        raise ValueError(
            "Only the flightmare dynamics backend is supported, got '{}'"
            .format(backend_name))
    return FlightmareDynamicsBackend(config)
