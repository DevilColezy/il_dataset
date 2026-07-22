#!/usr/bin/env python3
"""
il_dynamics.py  —  Flightmare Dynamics Backend  (Phase 4)

Provides:
  - DynamicsState dataclass
  - DynamicsBackend abstract interface
  - FlightmareDynamicsBackend: uses real Flightmare quadrotor dynamics
  - LegacyKinematicBackend: uses integrate_velocity_command() for debug only
  - VelocityYawRateController: converts velocity/yaw-rate commands
    to low-level control inputs

The default execution backend is Flightmare dynamics, not manual kinematic
integration.  The legacy kinematic backend is only available when explicitly
configured via backend: legacy_kinematic.
"""

from __future__ import print_function, division

import math, time
import numpy as np
from dataclasses import dataclass, field

import rospy

from il_common import (integrate_velocity_command, world_vector_to_body_flu,
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

    ``_il_local_planner.FlightmareDynamics`` is a deliberately small pybind
    wrapper around ``Quadrotor.reset(QuadState)``, ``Quadrotor.run(Command,
    dt)`` and ``Quadrotor.getState``.  Failure to import that compiled bridge
    is fatal; production collection never falls back to kinematics.
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
        self._integrator_limit = np.array(
            vc_cfg.get("integrator_limit", [1.0, 1.0, 0.5]),
            dtype=np.float64)

        try:
            from _il_local_planner import FlightmareDynamics
            self._quad_dynamics = FlightmareDynamics()
        except Exception as e:
            raise RuntimeError(
                "Real flightlib dynamics bridge is unavailable: {}. Rebuild "
                "il_dataset with flightlib; no production fallback is allowed."
                .format(e))

        # Controller state
        self._controller = VelocityYawRateController(
            self._kp_vel, self._ki_vel, self._kd_vel,
            self._max_accel, self._max_tilt_deg, self._max_yaw_rate,
            self._integrator_limit)
        self._attitude_gain = float(vc_cfg.get("attitude_gain", 6.0))
        self._max_body_rate = float(vc_cfg.get("maximum_body_rate_rps", 6.0))
        self._last_state = None

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

        while elapsed < duration_s - epsilon:
            ctrl_dur = min(dt_ctrl, duration_s - elapsed)

            state = self.get_state()
            yaw = self._yaw_from_xyzw(state.quaternion_world_body)
            accel_command_flu = self._controller.update(
                command, state.velocity_world, yaw, ctrl_dur,
                state.quaternion_world_body)

            ctrl_elapsed = 0.0
            while ctrl_elapsed < ctrl_dur - epsilon:
                step_dt = min(dt_sim, ctrl_dur - ctrl_elapsed)
                if not self._run_acceleration_step(
                        accel_command_flu, float(yaw_rate_command), step_dt):
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
        body_rates = -self._attitude_gain * attitude_error
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
                 integrator_limit):
        self._kp = np.asarray(kp, dtype=np.float64)
        self._ki = np.asarray(ki, dtype=np.float64)
        self._kd = np.asarray(kd, dtype=np.float64)
        self._max_accel = np.asarray(max_accel, dtype=np.float64)
        self._max_tilt_rad = math.radians(max_tilt_deg)
        self._max_yaw_rate = float(max_yaw_rate)
        self._int_limit = np.asarray(integrator_limit, dtype=np.float64)

        self._integrator = np.zeros(3, dtype=np.float64)
        self._prev_error = np.zeros(3, dtype=np.float64)

    def reset(self):
        self._integrator = np.zeros(3, dtype=np.float64)
        self._prev_error = np.zeros(3, dtype=np.float64)

    def update(self, desired_vel_flu, current_vel_world, current_yaw, dt,
               current_quaternion_xyzw=None):
        """Compute acceleration command from velocity error.

        Args:
            desired_vel_flu: [vx_fwd, vy_left, vz_up] desired in FLU.
            current_vel_world: [vx, vy, vz] current velocity in world.
            current_yaw: current yaw angle.
            dt: control time step.

        Returns:
            accel_cmd_flu: [ax, ay, az] acceleration command in FLU.
        """
        # Convert current world velocity to FLU
        current_flu = (world_vector_to_body_flu_quat(
            current_vel_world, current_quaternion_xyzw)
            if current_quaternion_xyzw is not None else
            world_vector_to_body_flu(current_vel_world, current_yaw))

        error = np.asarray(desired_vel_flu, dtype=np.float64) - current_flu

        # Integral with anti-windup
        self._integrator += error * dt
        self._integrator = np.clip(self._integrator, -self._int_limit,
                                   self._int_limit)

        # Derivative (prevent kick on first step)
        derivative = (error - self._prev_error) / max(dt, 1e-9)
        self._prev_error = error.copy()

        # PID
        accel = (self._kp * error +
                 self._ki * self._integrator +
                 self._kd * derivative)

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
#  Legacy Kinematic Backend (debug only)
# ============================================================================

class LegacyKinematicBackend(DynamicsBackend):
    """Kinematic velocity integrator (debug/CI only).

    Uses integrate_velocity_command() from il_common.  This backend
    MUST NOT be used for production data collection.  It is only
    available when explicitly configured via backend: legacy_kinematic.
    """

    def __init__(self, config):
        dyn_cfg = config.get("global", {}).get("dynamics", {})
        lp_cfg = config.get("global", {}).get("planning", {}).get(
            "local_planner", {})

        self._sim_hz = float(dyn_cfg.get("simulation_hz", 200.0))
        self._control_hz = float(dyn_cfg.get("control_hz", 50.0))
        self._render_hz = float(dyn_cfg.get("render_hz", 20.0))

        self._max_vel = float(lp_cfg.get("max_velocity", 2.5))
        self._max_accel = float(lp_cfg.get("max_acceleration", 3.5))
        self._max_yaw_rate = float(lp_cfg.get("max_yaw_rate", 2.0))

        self._pos = np.zeros(3, dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0
        self._sim_time = 0.0

    @property
    def backend_name(self):
        return "legacy_kinematic"

    def reset(self, initial_position, initial_yaw,
              initial_velocity=None, initial_angular_velocity=None):
        if initial_velocity is None:
            initial_velocity = np.zeros(3, dtype=np.float64)
        self._pos = np.asarray(initial_position, dtype=np.float64)
        self._yaw = float(initial_yaw)
        self._vel = np.asarray(initial_velocity, dtype=np.float64)
        self._sim_time = 0.0

    def get_state(self):
        flu_vel = world_vector_to_body_flu(self._vel, self._yaw)
        return DynamicsState(
            position_world=self._pos.copy(),
            quaternion_world_body=np.array(
                [0.0, 0.0, math.sin(self._yaw / 2.0), math.cos(self._yaw / 2.0)],
                dtype=np.float64),
            velocity_world=self._vel.copy(),
            velocity_flu=flu_vel,
            angular_velocity_body=np.zeros(3, dtype=np.float64),
            acceleration_world=np.zeros(3, dtype=np.float64),
            acceleration_flu=np.zeros(3, dtype=np.float64),
            simulation_time_s=self._sim_time,
        )

    def step_velocity_command(self, velocity_command_flu, yaw_rate_command,
                               duration_s):
        """Step using kinematic integration."""
        if duration_s <= 0:
            return True

        elapsed = 0.0
        dt_ctrl = 1.0 / self._control_hz
        epsilon = 1e-9

        cmd_flu = np.asarray(velocity_command_flu, dtype=np.float64)
        # Convert FLU desired to world frame for integrate_velocity_command
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        desired_world = np.array([
            cmd_flu[0] * cos_y - cmd_flu[1] * sin_y,
            cmd_flu[0] * sin_y + cmd_flu[1] * cos_y,
            cmd_flu[2]
        ], dtype=np.float64)

        while elapsed < duration_s - epsilon:
            step_dt = min(dt_ctrl, duration_s - elapsed)
            self._pos, self._vel, self._yaw, yr = integrate_velocity_command(
                self._pos, self._vel, self._yaw, desired_world, step_dt,
                self._max_vel, self._max_accel, self._max_yaw_rate)
            self._sim_time += step_dt
            elapsed += step_dt

        return True

    def close(self):
        pass


# ============================================================================
#  Factory function
# ============================================================================

def create_dynamics_backend(config):
    """Create the appropriate dynamics backend from configuration.

    Raises RuntimeError if flightmare is requested but unavailable.
    """
    dyn_cfg = config.get("global", {}).get("dynamics", {})
    backend_name = str(dyn_cfg.get("backend", "flightmare"))
    allow_legacy = bool(dyn_cfg.get("allow_legacy_kinematic_backend", True))
    legacy_explicit = bool(dyn_cfg.get("legacy_backend_explicit_only", True))

    if backend_name == "legacy_kinematic":
        if legacy_explicit:
            rospy.logwarn("[Dynamics] Using LEGACY kinematic backend (debug only). "
                          "NOT suitable for production data collection.")
        return LegacyKinematicBackend(config)

    if backend_name == "flightmare":
        try:
            return FlightmareDynamicsBackend(config)
        except RuntimeError:
            rospy.logerr(
                "[Dynamics] Flightmare backend failed to initialize. "
                "legacy_kinematic is available but NOT auto-selected. "
                "Set backend: legacy_kinematic explicitly for debug.")
            raise

    raise ValueError("Unknown dynamics backend: {}".format(backend_name))
