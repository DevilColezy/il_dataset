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

#include <cmath>
#include <memory>
#include <stdexcept>

#include "flightlib/common/command.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/objects/quadrotor.hpp"

namespace py = pybind11;

namespace {

class FlightmareDynamicsBridge {
 public:
  FlightmareDynamicsBridge()
    : quadrotor_(std::make_unique<flightlib::Quadrotor>(
          flightlib::QuadrotorDynamics(1.0, 0.25))) {}

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
    return quadrotor_->reset(state);
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
    return quadrotor_->setState(state);
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
  double stateTime() const {
    flightlib::QuadState state;
    return quadrotor_->getState(&state) ? state.t : 0.0;
  }

  std::unique_ptr<flightlib::Quadrotor> quadrotor_;
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
        .def("state", &FlightmareDynamicsBridge::state);
}
