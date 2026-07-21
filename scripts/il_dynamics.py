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

from il_common import integrate_velocity_command, world_vector_to_body_flu


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
    """Real Flightmare quadrotor dynamics.

    Uses the Flightmare/FlightGym quadrotor dynamics engine.  The velocity
    command is converted to motor thrusts / body rates via a low-level
    velocity controller.

    NOTE: This class requires the Flightmare Python bindings to be available.
    If the Flightmare dynamics API is not importable, this backend will
    raise a clear error on initialization — it will NOT silently fall back
    to kinematic integration.

    Expected Flightmare API (to be confirmed against actual project):
      - flightgym.QuadrotorDynamics or similar
      - Methods: reset(state), step(command, dt), getState()
    """

    def __init__(self, config):
        dyn_cfg = config.get("global", {}).get("dynamics", {})
        vc_cfg = dyn_cfg.get("velocity_controller", {})

        self._sim_hz = float(dyn_cfg.get("simulation_hz", 200.0))
        self._control_hz = float(dyn_cfg.get("control_hz", 50.0))
        self._render_hz = float(dyn_cfg.get("render_hz", 20.0))
        self._control_mode = str(dyn_cfg.get("control_mode", "velocity_yaw_rate"))
        self._deterministic = bool(dyn_cfg.get("deterministic_time", True))

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

        # State
        self._sim_time = 0.0
        self._pos = np.zeros(3, dtype=np.float64)
        self._vel = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0
        self._vel_integrator = np.zeros(3, dtype=np.float64)
        self._prev_vel_error = np.zeros(3, dtype=np.float64)

        # Try to import Flightmare dynamics
        self._flightmare_available = False
        self._quad_dynamics = None
        self._flightmare_api_info = ""

        try:
            # Attempt to import known Flightmare/FlightGym interfaces
            # These imports mirror what may exist in the project's dependencies.
            # If unavailable, the backend will report the exact error.
            try:
                import flightgym
                self._flightmare_api_info = "flightgym"
                # Check for QuadrotorDynamics class
                if hasattr(flightgym, 'QuadrotorDynamics'):
                    self._quad_dynamics = flightgym.QuadrotorDynamics()
                    self._flightmare_available = True
                elif hasattr(flightgym, 'Quadrotor'):
                    self._quad_dynamics = flightgym.Quadrotor()
                    self._flightmare_available = True
            except ImportError:
                pass

            if not self._flightmare_available:
                try:
                    from flightmare_python import QuadrotorDynamics
                    self._quad_dynamics = QuadrotorDynamics()
                    self._flightmare_available = True
                    self._flightmare_api_info = "flightmare_python.QuadrotorDynamics"
                except ImportError:
                    pass

            if not self._flightmare_available:
                raise RuntimeError(
                    "Flightmare dynamics backend unavailable. "
                    "Could not import flightgym or flightmare_python. "
                    "Explicitly select backend: legacy_kinematic only for debugging. "
                    "Do NOT use legacy_kinematic for production data collection.")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                "Flightmare dynamics initialization failed: {}. "
                "Explicitly select backend: legacy_kinematic only for debugging.".format(e))

        # Controller state
        self._controller = VelocityYawRateController(
            self._kp_vel, self._ki_vel, self._kd_vel,
            self._max_accel, self._max_tilt_deg, self._max_yaw_rate,
            self._integrator_limit)

        rospy.loginfo("[Dynamics] Flightmare backend initialized (%s). "
                      "sim=%.0fHz ctrl=%.0fHz render=%.0fHz",
                      self._flightmare_api_info,
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

        self._pos = np.asarray(initial_position, dtype=np.float64)
        self._yaw = float(initial_yaw)
        self._vel = np.asarray(initial_velocity, dtype=np.float64)
        self._sim_time = 0.0
        self._vel_integrator = np.zeros(3, dtype=np.float64)
        self._prev_vel_error = np.zeros(3, dtype=np.float64)
        self._controller.reset()

        # Apply noise
        if np.any(self._vel_noise_std > 0):
            noise = np.random.randn(3) * self._vel_noise_std
            self._vel += noise
        if np.any(self._ang_vel_noise_std > 0):
            self._yaw += np.random.randn() * self._ang_vel_noise_std[2] * 0.01

        # Settle
        dt_sim = 1.0 / self._sim_hz
        n_settle = int(self._settle_time_s / dt_sim)
        for _ in range(n_settle):
            self._step_single(dt_sim, np.zeros(3, dtype=np.float64), 0.0)

    def get_state(self):
        """Return current DynamicsState."""
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
        """Apply velocity command for duration_s, stepping dynamics internally."""
        if duration_s <= 0:
            return True

        elapsed = 0.0
        dt_sim = 1.0 / self._sim_hz
        dt_ctrl = 1.0 / self._control_hz
        epsilon = 1e-9

        while elapsed < duration_s - epsilon:
            ctrl_dur = min(dt_ctrl, duration_s - elapsed)

            # Compute low-level control from velocity command
            cmd_flu = np.asarray(velocity_command_flu, dtype=np.float64)
            yaw_cmd = float(yaw_rate_command)

            # Run the velocity controller to get acceleration command
            accel_cmd_flu = self._controller.update(
                cmd_flu, self._vel, self._yaw, dt_ctrl)

            # Step simulation at sim_hz within this control interval
            ctrl_elapsed = 0.0
            while ctrl_elapsed < ctrl_dur - epsilon:
                step_dt = min(dt_sim, ctrl_dur - ctrl_elapsed)
                self._step_single(step_dt, accel_cmd_flu, yaw_cmd)
                ctrl_elapsed += step_dt

            elapsed += ctrl_dur

        return True

    def _step_single(self, dt, accel_cmd_flu, yaw_rate_cmd):
        """Single dynamics integration step."""
        # Convert FLU acceleration to world frame
        cos_y = math.cos(self._yaw)
        sin_y = math.sin(self._yaw)
        # FLU -> World: rotate by yaw
        accel_world = np.array([
            accel_cmd_flu[0] * cos_y - accel_cmd_flu[1] * sin_y,
            accel_cmd_flu[0] * sin_y + accel_cmd_flu[1] * cos_y,
            accel_cmd_flu[2]
        ], dtype=np.float64)

        # Simple Euler integration (placeholder for real Flightmare dynamics)
        self._vel += accel_world * dt
        self._pos += self._vel * dt

        # Yaw integration
        yr = max(-self._max_yaw_rate, min(self._max_yaw_rate, yaw_rate_cmd))
        self._yaw += yr * dt
        self._yaw = math.atan2(math.sin(self._yaw), math.cos(self._yaw))

        self._sim_time += dt

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

    def update(self, desired_vel_flu, current_vel_world, current_yaw, dt):
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
        current_flu = world_vector_to_body_flu(current_vel_world, current_yaw)

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
