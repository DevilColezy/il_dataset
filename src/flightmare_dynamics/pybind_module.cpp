/// @file   pybind_module.cpp
/// @brief  Standalone pybind11 module for the real flightlib dynamics
///         bridge (`_flightmare_dynamics`).
///
/// This is the ONLY consumer of flightlib from Python.  It is kept
/// deliberately small and independent of the (deprecated) old expert
/// stack so the collection manager can drive the real Flightmare
/// quadrotor dynamics without loading any old expert module.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "flightlib/common/command.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/objects/quadrotor.hpp"

namespace py = pybind11;

namespace {

class FlightmareDynamicsBridge {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FlightmareDynamicsBridge()
    : quadrotor_(std::make_unique<flightlib::Quadrotor>(
          flightlib::QuadrotorDynamics(1.0, 0.25))) {
    // The dynamics and allocation matrix are fixed for this backend.  Keep
    // the inverse out of the 200 Hz inner loop; recomputing it there only
    // adds latency and jitter to the outer 30 Hz benchmark loop.
    allocation_matrix_inv_ =
        quadrotor_->getDynamics().getAllocationMatrix().cast<double>().inverse();
    resetController();
  }

  void configureVelocityController(
      const Eigen::Vector3d& kp_velocity,
      const Eigen::Vector3d& ki_velocity,
      const Eigen::Vector3d& kd_velocity,
      const Eigen::Vector3d& maximum_acceleration,
      double maximum_tilt_deg,
      double maximum_yaw_rate,
      double maximum_yaw_acceleration,
      const Eigen::Vector3d& integrator_limit,
      double derivative_filter_tau,
      double attitude_gain,
      double angular_rate_gain,
      double maximum_body_rate,
      double simulation_hz,
      double control_hz,
      bool use_trajectory_acceleration,
      bool use_trajectory_yaw_rate,
      bool use_trajectory_velocity_feedforward) {
    kp_velocity_ = kp_velocity;
    ki_velocity_ = ki_velocity;
    kd_velocity_ = kd_velocity;
    maximum_acceleration_ = maximum_acceleration;
    integrator_limit_ = integrator_limit;
    maximum_tilt_rad_ = maximum_tilt_deg *
        3.14159265358979323846 / 180.0;
    maximum_yaw_rate_ = maximum_yaw_rate;
    maximum_yaw_acceleration_ = maximum_yaw_acceleration;
    derivative_filter_tau_ = derivative_filter_tau;
    attitude_gain_ = attitude_gain;
    angular_rate_gain_ = angular_rate_gain;
    maximum_body_rate_ = maximum_body_rate;
    simulation_hz_ = simulation_hz;
    control_hz_ = control_hz;
    use_trajectory_acceleration_ = use_trajectory_acceleration;
    use_trajectory_yaw_rate_ = use_trajectory_yaw_rate;
    use_trajectory_velocity_feedforward_ =
        use_trajectory_velocity_feedforward;
  }

  bool reset(const Eigen::Vector3d& position,
             const Eigen::Vector4d& quaternion_wxyz,
             const Eigen::Vector3d& velocity,
             const Eigen::Vector3d& angular_velocity) {
    flightlib::QuadState state;
    state.setZero();
    state.p = position.cast<float>();
    state.qx = quaternion_wxyz.cast<float>();
    state.v = velocity.cast<float>();
    state.w = angular_velocity.cast<float>();
    state.t = 0.0;
    const bool ok = quadrotor_->reset(state);
    if (ok) resetController();
    return ok;
  }

  bool setStatePreserveMotors(
      const Eigen::Vector3d& position,
      const Eigen::Vector4d& quaternion_wxyz,
      const Eigen::Vector3d& velocity,
      const Eigen::Vector3d& angular_velocity) {
    flightlib::QuadState state;
    if (!quadrotor_->getState(&state)) {
      return false;
    }
    state.p = position.cast<float>();
    state.qx = quaternion_wxyz.cast<float>();
    state.qx.normalize();
    state.v = velocity.cast<float>();
    state.w = angular_velocity.cast<float>();
    state.a.setZero();
    state.tau.setZero();
    const bool ok = quadrotor_->setState(state);
    if (ok) resetController();
    return ok;
  }

  bool run(double collective_thrust,
           const Eigen::Vector3d& body_rates,
           double dt) {
    if (!std::isfinite(collective_thrust) || !body_rates.allFinite() ||
        !std::isfinite(dt) || dt <= 0.0) {
      return false;
    }
    flightlib::Command command;
    command.t = stateTime();
    command.collective_thrust = collective_thrust;
    command.omega = body_rates.cast<float>();
    return quadrotor_->run(command, dt);
  }

  // Return the last native controller sample.  This is intentionally a
  // snapshot rather than a ROS topic: the benchmark writes it into its
  // per-tick JSONL debug record without adding another high-rate transport.
  py::dict controllerDebug() const {
    py::dict out;
    out["valid"] = telemetry_valid_;
    out["simulation_time_s"] = telemetry_simulation_time_s_;
    out["control_step_dt_s"] = telemetry_control_step_dt_s_;
    out["derivative_dt_s"] = telemetry_derivative_dt_s_;
    out["trajectory_command"] = telemetry_trajectory_command_;
    out["desired_velocity_flu"] = vector3List(telemetry_desired_velocity_flu_);
    out["current_velocity_flu"] = vector3List(telemetry_current_velocity_flu_);
    out["velocity_error_flu"] = vector3List(telemetry_velocity_error_flu_);
    out["integrator_flu"] = vector3List(telemetry_integrator_flu_);
    out["derivative_flu"] = vector3List(telemetry_derivative_flu_);
    out["command_feedforward_flu"] = vector3List(telemetry_feedforward_flu_);
    out["target_position_world"] = vector3List(telemetry_target_position_world_);
    out["position_error_world"] = vector3List(telemetry_position_error_world_);
    out["target_acceleration_world"] = vector3List(telemetry_target_acceleration_world_);
    out["acceleration_flu_raw"] = vector3List(telemetry_acceleration_flu_raw_);
    out["acceleration_flu"] = vector3List(telemetry_acceleration_flu_);
    out["acceleration_world"] = vector3List(telemetry_acceleration_world_);
    out["thrust_world"] = vector3List(telemetry_thrust_world_);
    out["collective_thrust"] = telemetry_collective_thrust_;
    out["current_yaw"] = telemetry_current_yaw_;
    out["desired_yaw"] = telemetry_desired_yaw_;
    out["yaw_error"] = telemetry_yaw_error_;
    out["body_rates_commanded"] = vector3List(telemetry_body_rates_commanded_);
    out["body_rates_applied"] = vector3List(telemetry_body_rates_applied_);
    out["body_rates_actual"] = vector3List(telemetry_body_rates_actual_);
    out["quaternion_wxyz"] = vector4List(telemetry_quaternion_wxyz_);
    out["acceleration_component_saturated"] =
        telemetry_acceleration_component_saturated_;
    out["horizontal_acceleration_saturated"] =
        telemetry_horizontal_acceleration_saturated_;
    out["yaw_rate_saturated"] = telemetry_yaw_rate_saturated_;
    out["allocation_limited"] = telemetry_allocation_limited_;
    out["use_trajectory_acceleration"] = use_trajectory_acceleration_;
    out["use_trajectory_yaw_rate"] = use_trajectory_yaw_rate_;
    out["use_trajectory_velocity_feedforward"] =
        use_trajectory_velocity_feedforward_;
    return out;
  }

  // Run the complete velocity controller and the 200 Hz rigid-body inner
  // loop in C++.  Python calls this once per benchmark tick rather than
  // crossing the pybind boundary for every inner simulation step.
  bool stepVelocityCommand(const Eigen::Vector3d& desired_velocity_flu,
                           double yaw_rate_command,
                           double duration_s,
                           const Eigen::Vector3d& target_position_world,
                           const Eigen::Vector3d& target_acceleration_world,
                           double desired_yaw,
                           const Eigen::Vector3d& position_gain,
                           const Eigen::Vector3d& velocity_gain,
                           double yaw_position_gain) {
    if (!desired_velocity_flu.allFinite() ||
        !std::isfinite(yaw_rate_command) || !std::isfinite(duration_s) ||
      duration_s <= 0.0 || simulation_hz_ <= 0.0 || control_hz_ <= 0.0) {
      return false;
    }

    const bool trajectory_command =
        target_position_world.allFinite() &&
        target_acceleration_world.allFinite() &&
        position_gain.allFinite() && velocity_gain.allFinite() &&
        std::isfinite(desired_yaw) && std::isfinite(yaw_position_gain);

    Eigen::Vector3d feedforward = Eigen::Vector3d::Zero();
    if (previous_command_valid_ &&
        (!trajectory_command || use_trajectory_velocity_feedforward_)) {
      feedforward = (desired_velocity_flu - previous_command_) / duration_s;
      feedforward = feedforward.cwiseMax(-maximum_acceleration_)
                                .cwiseMin(maximum_acceleration_);
    }
    previous_command_ = desired_velocity_flu;
    previous_command_valid_ = true;

    const double dt_sim = 1.0 / simulation_hz_;
    const double dt_control = 1.0 / control_hz_;
    double elapsed = 0.0;
    constexpr double kEpsilon = 1e-9;

    constexpr double kMinimumControlDuration = 1e-4;
    while (elapsed < duration_s - kEpsilon) {
      const double remaining = duration_s - elapsed;
      // Do not turn a floating-point remainder into a real controller update.
      // Such a few-microsecond interval makes the velocity derivative explode
      // even though the corresponding dynamics time is negligible.
      if (remaining < kMinimumControlDuration)
        break;
      const double control_duration =
          std::min(dt_control, remaining);

      flightlib::QuadState state;
      if (!quadrotor_->getState(&state)) return false;

      const Eigen::Quaterniond quaternion(
          static_cast<double>(state.qx[0]),
          static_cast<double>(state.qx[1]),
          static_cast<double>(state.qx[2]),
          static_cast<double>(state.qx[3]));
      const Eigen::Matrix3d rotation = quaternion.normalized().toRotationMatrix();
      const Eigen::Vector3d velocity_world = state.v.cast<double>();
      const Eigen::Vector3d body_velocity = rotation.transpose() * velocity_world;
      const Eigen::Vector3d current_velocity_flu(
          body_velocity[1], -body_velocity[0], body_velocity[2]);
      const double current_yaw = std::atan2(
          2.0 * (quaternion.w() * quaternion.z() +
                  quaternion.x() * quaternion.y()),
          1.0 - 2.0 * (quaternion.y() * quaternion.y() +
                        quaternion.z() * quaternion.z()));
      Eigen::Vector3d position_error_world = Eigen::Vector3d::Zero();
      if (trajectory_command)
        position_error_world = target_position_world - state.p.cast<double>();

      const Eigen::Vector3d velocity_error =
          desired_velocity_flu - current_velocity_flu;
      integrator_ += velocity_error * control_duration;
      integrator_ = integrator_.cwiseMax(-integrator_limit_)
                              .cwiseMin(integrator_limit_);

      Eigen::Vector3d derivative = Eigen::Vector3d::Zero();
      double derivative_dt = control_duration;
      if (previous_velocity_valid_ && previous_velocity_time_valid_) {
        derivative_dt = static_cast<double>(state.t) -
            previous_velocity_time_s_;
        if (!std::isfinite(derivative_dt) ||
            derivative_dt < kMinimumControlDuration)
          derivative_dt = 0.0;
      }
      if (previous_velocity_valid_ && derivative_dt > 0.0) {
        const Eigen::Vector3d measured_acceleration =
            (velocity_world - previous_velocity_world_) / derivative_dt;
        const Eigen::Vector3d measured_body =
            rotation.transpose() * measured_acceleration;
        const Eigen::Vector3d measured_flu(
            measured_body[1], -measured_body[0], measured_body[2]);
        const Eigen::Vector3d derivative_raw = -measured_flu;
        if (derivative_filter_tau_ > 0.0) {
          const double filter_dt = derivative_dt > 0.0
              ? derivative_dt : control_duration;
          const double alpha = filter_dt /
              (derivative_filter_tau_ + filter_dt);
          derivative = previous_derivative_valid_
              ? alpha * derivative_raw + (1.0 - alpha) * derivative_filtered_
              : derivative_raw;
          derivative_filtered_ = derivative;
          previous_derivative_valid_ = true;
        } else {
          derivative = derivative_raw;
        }
      }
      previous_velocity_world_ = velocity_world;
      previous_velocity_time_s_ = static_cast<double>(state.t);
      previous_velocity_valid_ = true;
      previous_velocity_time_valid_ = true;

      Eigen::Vector3d acceleration =
          kp_velocity_.cwiseProduct(velocity_error) +
          ki_velocity_.cwiseProduct(integrator_) +
          kd_velocity_.cwiseProduct(derivative) + feedforward;
      if (trajectory_command) {
        const Eigen::Vector3d position_error_body =
            rotation.transpose() * position_error_world;
        const Eigen::Vector3d position_error_flu(
            position_error_body[1], -position_error_body[0],
            position_error_body[2]);
        const Eigen::Vector3d acceleration_body =
            rotation.transpose() * target_acceleration_world;
        const Eigen::Vector3d acceleration_flu(
            acceleration_body[1], -acceleration_body[0],
            acceleration_body[2]);
        acceleration += position_gain.cwiseProduct(position_error_flu) +
                        velocity_gain.cwiseProduct(velocity_error);
        if (use_trajectory_acceleration_)
          acceleration += acceleration_flu;
      }
      // Match the existing Python controller: the optional term uses the
      // measured body yaw rate, not the requested yaw rate.
      acceleration[1] += current_velocity_flu[0] *
          static_cast<double>(state.w[2]);
      const Eigen::Vector3d acceleration_flu_raw = acceleration;
      bool acceleration_component_saturated = false;
      for (int i = 0; i < 3; ++i) {
        if (std::abs(acceleration[i]) > maximum_acceleration_[i])
          acceleration_component_saturated = true;
      }
      acceleration = acceleration.cwiseMax(-maximum_acceleration_)
                                  .cwiseMin(maximum_acceleration_);

      const double max_horizontal_acceleration =
          9.81 * std::tan(maximum_tilt_rad_);
      const double horizontal_norm =
          std::hypot(acceleration[0], acceleration[1]);
      const bool horizontal_acceleration_saturated =
          horizontal_norm > max_horizontal_acceleration;
      if (horizontal_norm > max_horizontal_acceleration) {
        acceleration[0] *= max_horizontal_acceleration / horizontal_norm;
        acceleration[1] *= max_horizontal_acceleration / horizontal_norm;
      }

      // FLU [forward,left,up] -> Flightlib body [right,forward,up].
      const Eigen::Vector3d acceleration_body(
          -acceleration[1], acceleration[0], acceleration[2]);
      const Eigen::Vector3d acceleration_world =
          rotation * acceleration_body;
      const Eigen::Vector3d thrust_world =
          acceleration_world + Eigen::Vector3d(0.0, 0.0, 9.81);
      const double thrust_norm = thrust_world.norm();
      if (thrust_norm < 1e-6) return false;
      const Eigen::Vector3d b3_des = thrust_world / thrust_norm;

      const double heading_yaw = trajectory_command ? desired_yaw : current_yaw;
      const Eigen::Vector3d heading(std::cos(heading_yaw),
                                    std::sin(heading_yaw), 0.0);
      Eigen::Vector3d b2_des = b3_des.cross(heading);
      const double b2_norm = b2_des.norm();
      if (b2_norm < 1e-6) return false;
      b2_des /= b2_norm;
      const Eigen::Vector3d b1_des = b2_des.cross(b3_des);

      Eigen::Matrix3d desired_rotation;
      desired_rotation.col(0) = b1_des;
      desired_rotation.col(1) = b2_des;
      desired_rotation.col(2) = b3_des;
      const Eigen::Matrix3d error_matrix = 0.5 * (
          desired_rotation.transpose() * rotation -
          rotation.transpose() * desired_rotation);
      Eigen::Vector3d body_rates(
          error_matrix(2, 1), error_matrix(0, 2), error_matrix(1, 0));
      body_rates = -attitude_gain_ * body_rates -
          angular_rate_gain_ * state.w.cast<double>();

      double yaw_error = 0.0;
      double target_yaw_rate = use_trajectory_yaw_rate_
          ? yaw_rate_command : 0.0;
      if (trajectory_command) {
        yaw_error = std::atan2(
            std::sin(desired_yaw - current_yaw),
            std::cos(desired_yaw - current_yaw));
        target_yaw_rate += yaw_position_gain * yaw_error;
      }
      const bool yaw_rate_saturated =
          target_yaw_rate > maximum_yaw_rate_ ||
          target_yaw_rate < -maximum_yaw_rate_;
      target_yaw_rate = std::max(
          -maximum_yaw_rate_, std::min(maximum_yaw_rate_, target_yaw_rate));
      const double max_yaw_delta =
          std::max(0.0, maximum_yaw_acceleration_) * control_duration;
      commanded_yaw_rate_ += std::max(
          -max_yaw_delta, std::min(max_yaw_delta,
              target_yaw_rate - commanded_yaw_rate_));
      body_rates[2] += commanded_yaw_rate_;
      body_rates = body_rates.cwiseMax(
          Eigen::Vector3d::Constant(-maximum_body_rate_)).cwiseMin(
          Eigen::Vector3d::Constant(maximum_body_rate_));
      const Eigen::Vector3d body_rates_commanded = body_rates;

      const double collective = std::max(
          0.0, thrust_world.dot(rotation.col(2)));
      double control_elapsed = 0.0;
      while (control_elapsed < control_duration - kEpsilon) {
        const double step_dt = std::min(
            dt_sim, control_duration - control_elapsed);
        // Keep the commanded body rates inside the thrust allocation's
        // feasible set (see limitRatesToAllocation).  Otherwise flightlib's
        // per-motor clamp corrupts the total thrust and the drone rockets up.
        flightlib::QuadState current_state;
        if (!quadrotor_->getState(&current_state)) return false;
        const Eigen::Vector3d feasible_rates = limitRatesToAllocation(
            body_rates, current_state, collective);
        const bool allocation_limited =
            (feasible_rates - body_rates).norm() > 1e-9;

        telemetry_valid_ = true;
        telemetry_simulation_time_s_ = static_cast<double>(state.t);
        telemetry_control_step_dt_s_ = control_duration;
        telemetry_derivative_dt_s_ = derivative_dt;
        telemetry_trajectory_command_ = trajectory_command;
        telemetry_desired_velocity_flu_ = desired_velocity_flu;
        telemetry_current_velocity_flu_ = current_velocity_flu;
        telemetry_velocity_error_flu_ = velocity_error;
        telemetry_integrator_flu_ = integrator_;
        telemetry_derivative_flu_ = derivative;
        telemetry_feedforward_flu_ = feedforward;
        telemetry_target_position_world_ = trajectory_command
            ? target_position_world : Eigen::Vector3d::Zero();
        telemetry_position_error_world_ = position_error_world;
        telemetry_target_acceleration_world_ = trajectory_command
            ? target_acceleration_world : Eigen::Vector3d::Zero();
        telemetry_acceleration_flu_raw_ = acceleration_flu_raw;
        telemetry_acceleration_flu_ = acceleration;
        telemetry_acceleration_world_ = acceleration_world;
        telemetry_thrust_world_ = thrust_world;
        telemetry_collective_thrust_ = collective;
        telemetry_current_yaw_ = current_yaw;
        telemetry_desired_yaw_ = trajectory_command ? desired_yaw : current_yaw;
        telemetry_yaw_error_ = yaw_error;
        telemetry_body_rates_commanded_ = body_rates_commanded;
        telemetry_body_rates_applied_ = feasible_rates;
        telemetry_body_rates_actual_ = state.w.cast<double>();
        telemetry_quaternion_wxyz_ = state.qx.cast<double>();
        telemetry_acceleration_component_saturated_ =
            acceleration_component_saturated;
        telemetry_horizontal_acceleration_saturated_ =
            horizontal_acceleration_saturated;
        telemetry_yaw_rate_saturated_ = yaw_rate_saturated;
        telemetry_allocation_limited_ = allocation_limited;
        flightlib::Command command;
        command.t = static_cast<double>(current_state.t);
        command.collective_thrust = collective;
        command.omega = feasible_rates.cast<float>();
        if (!quadrotor_->run(command, step_dt)) return false;
        control_elapsed += step_dt;
      }
      elapsed += control_duration;
    }
    return true;
  }

  // flightlib's low-level loop maps commanded body rates to torques
  // tau = J * Kinv_ang_vel_tau * (omega_des - omega) + omega x (J omega) and
  // allocates them to the four motors.  When that allocation leaves the
  // per-motor range [0, thrust_max], flightlib clamps each motor on its own
  // and the TOTAL thrust no longer matches the commanded collective (it can
  // reach ~2-3x and rocket the drone upward).  Scale the rate error down so
  // every motor stays inside its thrust limits; this only activates when the
  // requested rates exceed the actuator's feasible authority (e.g. a large
  // yaw error saturating the yaw body-rate loop).
  Eigen::Vector3d limitRatesToAllocation(
      const Eigen::Vector3d& omega_des,
      const flightlib::QuadState& state,
      double collective) {
    const flightlib::QuadrotorDynamics& dynamics = quadrotor_->getDynamics();
    const Eigen::Matrix3d J = dynamics.getJ().cast<double>();
    const Eigen::Vector3d omega = state.w.cast<double>();
    const Eigen::Vector3d tau_gyro = omega.cross(J * omega);
    // Same Kinv_ang_vel_tau_ diagonal as flightlib::Quadrotor::runFlightCtl.
    const Eigen::Vector3d tau_cmd =
        J * body_rate_gain_.cwiseProduct(omega_des - omega);
    const double force =
        static_cast<double>(dynamics.getMass()) * collective;
    const Eigen::Vector4d base =
        allocation_matrix_inv_ * Eigen::Vector4d(
            force, tau_gyro.x(), tau_gyro.y(), tau_gyro.z());
    const Eigen::Vector4d pert =
        allocation_matrix_inv_ * Eigen::Vector4d(
            0.0, tau_cmd.x(), tau_cmd.y(), tau_cmd.z());
    const double t_max =
        0.25 * static_cast<double>(dynamics.collective_thrust_max());
    const double t_min = 0.0;
    double scale = 1.0;
    for (int i = 0; i < 4; ++i) {
      if (pert[i] > 1e-12) {
        scale = std::min(scale, (t_max - base[i]) / pert[i]);
      } else if (pert[i] < -1e-12) {
        scale = std::min(scale, (t_min - base[i]) / pert[i]);
      }
    }
    scale = std::max(0.0, std::min(1.0, scale));
    return omega + scale * (omega_des - omega);
  }

  Eigen::VectorXd state() const {
    flightlib::QuadState state;
    if (!quadrotor_->getState(&state)) {
      throw std::runtime_error("flightlib::Quadrotor::getState failed");
    }
    Eigen::VectorXd output(flightlib::QuadState::SIZE + 1);
    output[0] = state.t;
    output.tail(flightlib::QuadState::SIZE) = state.x.cast<double>();
    return output;
  }

 private:
  void resetController() {
    integrator_.setZero();
    previous_velocity_world_.setZero();
    previous_command_.setZero();
    derivative_filtered_.setZero();
    previous_velocity_valid_ = false;
    previous_velocity_time_s_ = 0.0;
    previous_velocity_time_valid_ = false;
    previous_command_valid_ = false;
    previous_derivative_valid_ = false;
    commanded_yaw_rate_ = 0.0;
    resetTelemetry();
  }

  static py::list vector3List(const Eigen::Vector3d& value) {
    py::list result;
    for (int i = 0; i < 3; ++i) result.append(value[i]);
    return result;
  }

  static py::list vector4List(const Eigen::Vector4d& value) {
    py::list result;
    for (int i = 0; i < 4; ++i) result.append(value[i]);
    return result;
  }

  void resetTelemetry() {
    telemetry_valid_ = false;
    telemetry_simulation_time_s_ = 0.0;
    telemetry_control_step_dt_s_ = 0.0;
    telemetry_derivative_dt_s_ = 0.0;
    telemetry_trajectory_command_ = false;
    telemetry_desired_velocity_flu_.setZero();
    telemetry_current_velocity_flu_.setZero();
    telemetry_velocity_error_flu_.setZero();
    telemetry_integrator_flu_.setZero();
    telemetry_derivative_flu_.setZero();
    telemetry_feedforward_flu_.setZero();
    telemetry_target_position_world_.setZero();
    telemetry_position_error_world_.setZero();
    telemetry_target_acceleration_world_.setZero();
    telemetry_acceleration_flu_raw_.setZero();
    telemetry_acceleration_flu_.setZero();
    telemetry_acceleration_world_.setZero();
    telemetry_thrust_world_.setZero();
    telemetry_collective_thrust_ = 0.0;
    telemetry_current_yaw_ = 0.0;
    telemetry_desired_yaw_ = 0.0;
    telemetry_yaw_error_ = 0.0;
    telemetry_body_rates_commanded_.setZero();
    telemetry_body_rates_applied_.setZero();
    telemetry_body_rates_actual_.setZero();
    telemetry_quaternion_wxyz_.setZero();
    telemetry_quaternion_wxyz_[0] = 1.0;
    telemetry_acceleration_component_saturated_ = false;
    telemetry_horizontal_acceleration_saturated_ = false;
    telemetry_yaw_rate_saturated_ = false;
    telemetry_allocation_limited_ = false;
  }

  double stateTime() const {
    flightlib::QuadState state;
    return quadrotor_->getState(&state) ? state.t : 0.0;
  }

  std::unique_ptr<flightlib::Quadrotor> quadrotor_;
  Eigen::Vector3d kp_velocity_{3.0, 3.0, 3.0};
  Eigen::Vector3d ki_velocity_{0.0, 0.0, 0.0};
  Eigen::Vector3d kd_velocity_{0.2, 0.2, 0.2};
  Eigen::Vector3d maximum_acceleration_{4.0, 4.0, 2.0};
  Eigen::Vector3d integrator_limit_{1.0, 1.0, 0.5};
  Eigen::Vector3d integrator_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d previous_velocity_world_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d previous_command_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d derivative_filtered_{Eigen::Vector3d::Zero()};
  bool previous_velocity_valid_{false};
  double previous_velocity_time_s_{0.0};
  bool previous_velocity_time_valid_{false};
  bool previous_command_valid_{false};
  bool previous_derivative_valid_{false};
  double maximum_tilt_rad_{35.0 * 3.14159265358979323846 / 180.0};
  double maximum_yaw_rate_{3.14159265358979323846};
  double maximum_yaw_acceleration_{4.0};
  double derivative_filter_tau_{0.0};
  double attitude_gain_{6.0};
  double angular_rate_gain_{0.5};
  double maximum_body_rate_{6.0};
  double simulation_hz_{200.0};
  double control_hz_{50.0};
  bool use_trajectory_acceleration_{true};
  bool use_trajectory_yaw_rate_{true};
  bool use_trajectory_velocity_feedforward_{true};
  double commanded_yaw_rate_{0.0};
  // Mirrors flightlib::Quadrotor::Kinv_ang_vel_tau_ (diag 16.6, 16.6, 5.0)
  // used by runFlightCtl, needed for the allocation-feasibility limiter.
  Eigen::Vector3d body_rate_gain_{16.6, 16.6, 5.0};
  Eigen::Matrix4d allocation_matrix_inv_{Eigen::Matrix4d::Identity()};

  bool telemetry_valid_{false};
  double telemetry_simulation_time_s_{0.0};
  double telemetry_control_step_dt_s_{0.0};
  double telemetry_derivative_dt_s_{0.0};
  bool telemetry_trajectory_command_{false};
  Eigen::Vector3d telemetry_desired_velocity_flu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_current_velocity_flu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_velocity_error_flu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_integrator_flu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_derivative_flu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_feedforward_flu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_target_position_world_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_position_error_world_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_target_acceleration_world_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_acceleration_flu_raw_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_acceleration_flu_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_acceleration_world_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_thrust_world_{Eigen::Vector3d::Zero()};
  double telemetry_collective_thrust_{0.0};
  double telemetry_current_yaw_{0.0};
  double telemetry_desired_yaw_{0.0};
  double telemetry_yaw_error_{0.0};
  Eigen::Vector3d telemetry_body_rates_commanded_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_body_rates_applied_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d telemetry_body_rates_actual_{Eigen::Vector3d::Zero()};
  Eigen::Vector4d telemetry_quaternion_wxyz_{1.0, 0.0, 0.0, 0.0};
  bool telemetry_acceleration_component_saturated_{false};
  bool telemetry_horizontal_acceleration_saturated_{false};
  bool telemetry_yaw_rate_saturated_{false};
  bool telemetry_allocation_limited_{false};
};

}  // namespace

PYBIND11_MODULE(_flightmare_dynamics, m) {
    m.doc() = "Real flightlib Quadrotor dynamics bridge for Flightmare";

    py::class_<FlightmareDynamicsBridge>(m, "FlightmareDynamics")
        .def(py::init<>())
        .def("reset", &FlightmareDynamicsBridge::reset,
             py::arg("position"), py::arg("quaternion_wxyz"),
             py::arg("velocity"), py::arg("angular_velocity"))
        .def("set_state_preserve_motors",
             &FlightmareDynamicsBridge::setStatePreserveMotors,
             py::arg("position"), py::arg("quaternion_wxyz"),
             py::arg("velocity"), py::arg("angular_velocity"))
        .def("run", &FlightmareDynamicsBridge::run,
             py::arg("collective_thrust"), py::arg("body_rates"),
             py::arg("dt"))
        .def("configure_velocity_controller",
             &FlightmareDynamicsBridge::configureVelocityController,
             py::arg("kp_velocity"), py::arg("ki_velocity"),
             py::arg("kd_velocity"), py::arg("maximum_acceleration"),
             py::arg("maximum_tilt_deg"), py::arg("maximum_yaw_rate"),
             py::arg("maximum_yaw_acceleration"),
             py::arg("integrator_limit"),
             py::arg("derivative_filter_tau"),
             py::arg("attitude_gain"), py::arg("angular_rate_gain"),
             py::arg("maximum_body_rate"), py::arg("simulation_hz"),
             py::arg("control_hz"),
             py::arg("use_trajectory_acceleration"),
             py::arg("use_trajectory_yaw_rate"),
             py::arg("use_trajectory_velocity_feedforward"))
        .def("step_velocity_command",
             &FlightmareDynamicsBridge::stepVelocityCommand,
             py::arg("velocity_command_flu"), py::arg("yaw_rate"),
             py::arg("duration_s"), py::arg("target_position_world"),
             py::arg("target_acceleration_world"), py::arg("desired_yaw"),
             py::arg("position_gain"), py::arg("velocity_gain"),
             py::arg("yaw_position_gain"),
             py::call_guard<py::gil_scoped_release>())
        .def("controller_debug",
             &FlightmareDynamicsBridge::controllerDebug)
        .def("state", &FlightmareDynamicsBridge::state);
}
