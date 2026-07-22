/// pybind11 bindings for the il_dataset C++ local planner.
///
/// Exposes:
///   - VehicleState, TrajectoryPoint, LocalPlanResult, ValidationResult
///   - LocalPlannerConfig, LocalPlanner, PlannerStatus
///
/// The planLocal() method releases the GIL so that Python control
/// and depth-receive threads can continue while optimization runs.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <memory>
#include <stdexcept>

#include "flightlib/common/command.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/objects/quadrotor.hpp"
#include <pybind11/eigen.h>

#include "il_dataset/local_planner/types.hpp"
#include "il_dataset/local_planner/esdf_grid.hpp"
#include "il_dataset/local_planner/local_planner.hpp"
#include "il_dataset/local_planner/depth_integrator.hpp"

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
using namespace il_dataset;

PYBIND11_MODULE(_il_local_planner, m) {
  py::class_<FlightmareDynamicsBridge>(m, "FlightmareDynamics")
    .def(py::init<>())
    .def("reset", &FlightmareDynamicsBridge::reset,
         py::arg("position"), py::arg("quaternion_wxyz"),
         py::arg("velocity"), py::arg("angular_velocity"))
    .def("run", &FlightmareDynamicsBridge::run,
         py::arg("collective_thrust"), py::arg("body_rates"), py::arg("dt"))
    .def("state", &FlightmareDynamicsBridge::state);
    m.doc() = "C++ receding-horizon local planner for IL dataset collection";

    // ── Enums ────────────────────────────────────────────────────
    py::enum_<PlannerStatus>(m, "PlannerStatus",
        "Status codes returned by the local planner.")
        .value("SUCCESS", PlannerStatus::SUCCESS)
        .value("INVALID_INPUT", PlannerStatus::INVALID_INPUT)
        .value("NO_GLOBAL_PATH", PlannerStatus::NO_GLOBAL_PATH)
        .value("LOCAL_GOAL_INVALID", PlannerStatus::LOCAL_GOAL_INVALID)
        .value("OPTIMIZATION_FAILED", PlannerStatus::OPTIMIZATION_FAILED)
        .value("COLLISION", PlannerStatus::COLLISION)
        .value("DYNAMICS_VIOLATION", PlannerStatus::DYNAMICS_VIOLATION)
        .value("OUTSIDE_MAP", PlannerStatus::OUTSIDE_MAP)
        .value("EMERGENCY_HOLD", PlannerStatus::EMERGENCY_HOLD)
        .value("UNKNOWN_SPACE", PlannerStatus::UNKNOWN_SPACE)
        .export_values();

    // ── Data structures ──────────────────────────────────────────
    py::class_<VehicleState>(m, "VehicleState",
        "Vehicle state in ROS world coordinates (x-fwd, y-left, z-up).")
        .def(py::init<>())
        .def_readwrite("position", &VehicleState::position)
        .def_readwrite("velocity", &VehicleState::velocity)
        .def_readwrite("acceleration", &VehicleState::acceleration)
        .def_readwrite("yaw", &VehicleState::yaw)
        .def_readwrite("yaw_rate", &VehicleState::yaw_rate);

    py::class_<LocalPlanningRequest>(m, "LocalPlanningRequest",
        "Phase 2: explicit planning request with guide/terminal separation.")
        .def(py::init<>())
        .def_readwrite("state", &LocalPlanningRequest::state)
        .def_readwrite("previous_progress_s",
                       &LocalPlanningRequest::previous_progress_s)
        .def_readwrite("guide_waypoint",
                       &LocalPlanningRequest::guide_waypoint)
        .def_readwrite("guide_waypoint_index",
                       &LocalPlanningRequest::guide_waypoint_index)
        .def_readwrite("trajectory_terminal",
                       &LocalPlanningRequest::trajectory_terminal)
        .def_readwrite("trajectory_terminal_index",
                       &LocalPlanningRequest::trajectory_terminal_index)
        .def_readwrite("reference_path_segment",
                       &LocalPlanningRequest::reference_path_segment)
        .def_readwrite("forbid_unknown_space",
                       &LocalPlanningRequest::forbid_unknown_space)
        .def_readwrite("allow_global_map_fallback",
                       &LocalPlanningRequest::allow_global_map_fallback);

    py::class_<TrajectoryPoint>(m, "TrajectoryPoint",
        "Single point on a dense time-sampled trajectory.")
        .def(py::init<>())
        .def_readwrite("t", &TrajectoryPoint::t)
        .def_readwrite("position", &TrajectoryPoint::position)
        .def_readwrite("velocity", &TrajectoryPoint::velocity)
        .def_readwrite("acceleration", &TrajectoryPoint::acceleration)
        .def_readwrite("yaw", &TrajectoryPoint::yaw)
        .def_readwrite("yaw_rate", &TrajectoryPoint::yaw_rate)
        .def_readwrite("clearance", &TrajectoryPoint::clearance);

    py::class_<LocalPlanResult>(m, "LocalPlanResult",
        "Complete result of a single local-planning invocation.")
        .def(py::init<>())
        .def_readwrite("success", &LocalPlanResult::success)
        .def_readwrite("status", &LocalPlanResult::status)
        .def_readwrite("message", &LocalPlanResult::message)
        .def_readwrite("trajectory", &LocalPlanResult::trajectory)
        .def_readwrite("planning_time_ms", &LocalPlanResult::planning_time_ms)
        .def_readwrite("min_clearance", &LocalPlanResult::min_clearance)
        .def_readwrite("progress_s", &LocalPlanResult::progress_s)
        .def_readwrite("progress_index", &LocalPlanResult::progress_index)
        .def_readwrite("local_goal_index", &LocalPlanResult::local_goal_index)
        .def_readwrite("local_goal", &LocalPlanResult::local_goal)
        .def_readwrite("plan_id", &LocalPlanResult::plan_id)
        // Phase 2 fields
        .def_readwrite("guide_waypoint", &LocalPlanResult::guide_waypoint)
        .def_readwrite("guide_waypoint_index",
                       &LocalPlanResult::guide_waypoint_index)
        .def_readwrite("trajectory_terminal",
                       &LocalPlanResult::trajectory_terminal)
        .def_readwrite("trajectory_terminal_index",
                       &LocalPlanResult::trajectory_terminal_index)
        .def_readwrite("used_global_fallback",
                       &LocalPlanResult::used_global_fallback)
        .def_readwrite("used_observed_esdf",
                       &LocalPlanResult::used_observed_esdf);

    py::class_<ValidationResult>(m, "ValidationResult",
        "Result of trajectory collision/clearance validation.")
        .def(py::init<>())
        .def_readwrite("all_clear", &ValidationResult::all_clear)
        .def_readwrite("any_collision", &ValidationResult::any_collision)
        .def_readwrite("min_clearance", &ValidationResult::min_clearance)
        .def_readwrite("clearance_violation_count",
                       &ValidationResult::clearance_violation_count)
        .def_readwrite("worst_position", &ValidationResult::worst_position)
        .def_readwrite("worst_time", &ValidationResult::worst_time)
        .def_readwrite("worst_clearance", &ValidationResult::worst_clearance);

    // ── LocalPlannerConfig ───────────────────────────────────────
    py::class_<LocalPlannerConfig>(m, "LocalPlannerConfig",
        "Configuration for the local receding-horizon planner.")
        .def(py::init<>())
        .def_readwrite("planner_hz", &LocalPlannerConfig::planner_hz)
        .def_readwrite("horizon_time", &LocalPlannerConfig::horizon_time)
        .def_readwrite("execute_prefix_time", &LocalPlannerConfig::execute_prefix_time)
        .def_readwrite("max_plan_age", &LocalPlannerConfig::max_plan_age)
        .def_readwrite("planning_time_budget_ms",
                       &LocalPlannerConfig::planning_time_budget_ms)
        .def_readwrite("optimizer", &LocalPlannerConfig::optimizer)
        .def_readwrite("trajectory_dt", &LocalPlannerConfig::trajectory_dt)
        .def_readwrite("lookahead_distance", &LocalPlannerConfig::lookahead_distance)
        .def_readwrite("min_lookahead_distance",
                       &LocalPlannerConfig::min_lookahead_distance)
        .def_readwrite("max_lookahead_distance",
                       &LocalPlannerConfig::max_lookahead_distance)
        .def_readwrite("lookahead_velocity_gain",
                       &LocalPlannerConfig::lookahead_velocity_gain)
        .def_readwrite("curvature_lookahead_gain",
                       &LocalPlannerConfig::curvature_lookahead_gain)
        .def_readwrite("local_map_radius", &LocalPlannerConfig::local_map_radius)
        .def_readwrite("max_reference_points",
                       &LocalPlannerConfig::max_reference_points)
        .def_readwrite("control_points", &LocalPlannerConfig::control_points)
        .def_readwrite("max_iterations", &LocalPlannerConfig::max_iterations)
        .def_readwrite("convergence_tolerance",
                       &LocalPlannerConfig::convergence_tolerance)
        .def_readwrite("initial_step_size", &LocalPlannerConfig::initial_step_size)
        .def_readwrite("minimum_step_size", &LocalPlannerConfig::minimum_step_size)
        .def_readwrite("max_cost_samples_per_segment",
                       &LocalPlannerConfig::max_cost_samples_per_segment)
        .def_readwrite("min_clearance", &LocalPlannerConfig::min_clearance)
        .def_readwrite("target_clearance", &LocalPlannerConfig::target_clearance)
        .def_readwrite("collision_check_spacing",
                       &LocalPlannerConfig::collision_check_spacing)
        .def_readwrite("weight_smooth", &LocalPlannerConfig::weight_smooth)
        .def_readwrite("weight_jerk", &LocalPlannerConfig::weight_jerk)
        .def_readwrite("weight_guide", &LocalPlannerConfig::weight_guide)
        .def_readwrite("weight_obstacle", &LocalPlannerConfig::weight_obstacle)
        .def_readwrite("weight_goal", &LocalPlannerConfig::weight_goal)
        .def_readwrite("weight_dynamics", &LocalPlannerConfig::weight_dynamics)
        .def_readwrite("nominal_speed", &LocalPlannerConfig::nominal_speed)
        .def_readwrite("max_velocity", &LocalPlannerConfig::max_velocity)
        .def_readwrite("max_acceleration", &LocalPlannerConfig::max_acceleration)
        .def_readwrite("max_jerk", &LocalPlannerConfig::max_jerk)
        .def_readwrite("max_yaw_rate", &LocalPlannerConfig::max_yaw_rate)
        .def_readwrite("goal_tolerance", &LocalPlannerConfig::goal_tolerance)
        .def_readwrite("goal_speed_tolerance", &LocalPlannerConfig::goal_speed_tolerance)
        .def_readwrite("goal_hold_ticks", &LocalPlannerConfig::goal_hold_ticks)
        .def_readwrite("max_consecutive_failures",
                       &LocalPlannerConfig::max_consecutive_failures)
        .def_readwrite("reduce_lookahead_on_failure",
                       &LocalPlannerConfig::reduce_lookahead_on_failure)
        .def_readwrite("emergency_hold_enabled",
                       &LocalPlannerConfig::emergency_hold_enabled);

    // ── LocalPlanner ─────────────────────────────────────────────
    py::class_<LocalPlanner>(m, "LocalPlanner",
        "Receding-horizon local trajectory planner.")
        .def(py::init<const LocalPlannerConfig&>(),
             py::arg("config"),
             "Create a planner with the given configuration.")

        // set_esdf: numpy float32 array, shape [gx, gy, gz]
        .def("set_esdf",
             [](LocalPlanner& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> esdf_data,
                const Eigen::Vector3d& origin,
                double resolution) -> bool {
                 py::buffer_info buf = esdf_data.request();
                 if (buf.ndim != 3) {
                     PyErr_SetString(PyExc_ValueError,
                         "ESDF array must be 3-dimensional [gx, gy, gz]");
                     throw py::error_already_set();
                 }
                 const float* ptr = static_cast<const float*>(buf.ptr);
                 int gx = static_cast<int>(buf.shape[0]);
                 int gy = static_cast<int>(buf.shape[1]);
                 int gz = static_cast<int>(buf.shape[2]);
                 return self.setESDF(ptr, gx, gy, gz,
                                     origin.x(), origin.y(), origin.z(),
                                     resolution);
             },
             py::arg("esdf_data"),
             py::arg("origin"),
             py::arg("resolution"),
             "Set the ESDF map. Data is COPIED once. "
             "esdf_data: float32 numpy array [gx, gy, gz]. "
             "ESDF values should already have drone_radius subtracted.")

        // set_observed_esdf: Phase 2 — with known mask
        .def("set_observed_esdf",
             [](LocalPlanner& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> esdf_data,
                py::array_t<uint8_t, py::array::c_style | py::array::forcecast> known_mask,
                const Eigen::Vector3d& origin,
                double resolution,
                bool unknown_is_free) -> bool {
                 py::buffer_info ebuf = esdf_data.request();
                 py::buffer_info kbuf = known_mask.request();
                 if (ebuf.ndim != 3 || kbuf.ndim != 3) {
                     PyErr_SetString(PyExc_ValueError,
                         "ESDF and mask arrays must be 3-dimensional [gx, gy, gz]");
                     throw py::error_already_set();
                 }
                 if (ebuf.shape[0] != kbuf.shape[0] ||
                     ebuf.shape[1] != kbuf.shape[1] ||
                     ebuf.shape[2] != kbuf.shape[2]) {
                     PyErr_SetString(PyExc_ValueError,
                         "ESDF and mask must have identical dimensions");
                     throw py::error_already_set();
                 }
                 const float* eptr = static_cast<const float*>(ebuf.ptr);
                 const uint8_t* kptr = static_cast<const uint8_t*>(kbuf.ptr);
                 int gx = static_cast<int>(ebuf.shape[0]);
                 int gy = static_cast<int>(ebuf.shape[1]);
                 int gz = static_cast<int>(ebuf.shape[2]);
                 return self.setObservedESDF(eptr, kptr, gx, gy, gz,
                                              origin.x(), origin.y(), origin.z(),
                                              resolution, unknown_is_free);
             },
             py::arg("esdf_data"),
             py::arg("known_mask"),
             py::arg("origin"),
             py::arg("resolution"),
             py::arg("unknown_is_free"),
             "Set observed ESDF with known mask (Phase 2). "
             "esdf_data: float32 [gx, gy, gz]. "
             "known_mask: uint8 [gx, gy, gz], 0=unknown, 1=known.")

        // set_global_path: numpy float64 array, shape [N, 3]
        .def("set_global_path",
             [](LocalPlanner& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> path) -> bool {
                 py::buffer_info buf = path.request();
                 if (buf.ndim != 2 || buf.shape[1] != 3) {
                     PyErr_SetString(PyExc_ValueError,
                         "Global path must be [N, 3] float64 array");
                     throw py::error_already_set();
                 }
                 const double* ptr = static_cast<const double*>(buf.ptr);
                 int n = static_cast<int>(buf.shape[0]);
                 return self.setGlobalPath(ptr, n);
             },
             py::arg("path"),
             "Set the global reference path (A* shortcut output). "
             "path: float64 numpy array [N, 3].")

        // reset
        .def("reset",
             &LocalPlanner::reset,
             py::arg("initial_state"),
             "Reset planner state for a new trajectory.")

        // plan_local – RELEASES GIL
        .def("plan_local",
             [](const LocalPlanner& self,
                const VehicleState& current_state,
                double previous_progress_s) -> LocalPlanResult {
                 py::gil_scoped_release release;
                 return self.planLocal(current_state, previous_progress_s);
             },
             py::arg("current_state"),
             py::arg("previous_progress_s"),
             "Plan a local trajectory from the current state. "
             "Releases the Python GIL during optimization.")

        // plan_local_with_request: Phase 2
        .def("plan_local_with_request",
             [](const LocalPlanner& self,
                const LocalPlanningRequest& request) -> LocalPlanResult {
                 py::gil_scoped_release release;
                 return self.planLocalWithRequest(request);
             },
             py::arg("request"),
             "Phase 2: Plan with explicit guide/terminal/reference segment. "
             "Releases the Python GIL during optimization.")

        // validate_trajectory
        .def("validate_trajectory",
             &LocalPlanner::validateTrajectory,
             py::arg("trajectory"),
             "Validate a trajectory for collisions and clearance.")

        // is_ready
        .def("is_ready", &LocalPlanner::isReady,
             "Return whether the planner has been initialized with ESDF and global path.")

        // current_plan_id
        .def("current_plan_id", &LocalPlanner::currentPlanId,
             "Get the current plan ID counter value.");

    // ── ESDFGrid (for testing / debugging) ───────────────────────
    py::class_<ESDFGrid>(m, "ESDFGrid",
        "Trilinear ESDF grid for collision checking.")
        .def(py::init<>())
        .def("get_value", &ESDFGrid::getValue,
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Query ESDF value at world position via trilinear interpolation.")
        .def("get_gradient",
             [](const ESDFGrid& self, double x, double y, double z) -> py::tuple {
                 double clearance = 0.0;
                 Eigen::Vector3d grad = self.getGradient(x, y, z, &clearance);
                 return py::make_tuple(grad, clearance);
             },
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Query ESDF gradient and clearance at world position.")
        .def("is_free", &ESDFGrid::isFree,
             py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("min_clearance") = 0.0,
             "Check if a point has at least the given clearance.")
        .def_property_readonly("gx", &ESDFGrid::gx)
        .def_property_readonly("gy", &ESDFGrid::gy)
        .def_property_readonly("gz", &ESDFGrid::gz)
        .def_property_readonly("origin_x", &ESDFGrid::originX)
        .def_property_readonly("origin_y", &ESDFGrid::originY)
        .def_property_readonly("origin_z", &ESDFGrid::originZ)
        .def_property_readonly("resolution", &ESDFGrid::resolution)
        .def_property_readonly("initialized", &ESDFGrid::initialized)
        .def_property_readonly("memory_bytes", &ESDFGrid::memoryBytes);

    // ── Depth integrator ─────────────────────────────────────────
    m.def("integrate_depth", &integrate_depth,
          py::arg("points_world"),
          py::arg("points_cam_z"),
          py::arg("cam_pos"),
          py::arg("occ_grid"),
          py::arg("last_obs_time"),
          py::arg("occ_endpoint_margin"),
          py::arg("free_space_spacing"),
          py::arg("resolution"),
          py::arg("max_depth_m"),
          py::arg("timestamp_s"),
          py::arg("origin"),
          py::arg("grid_dims"),
          "Integrate depth rays into occupancy grid (C++). "
          "Modifies occ_grid and last_obs_time in-place. "
          "Returns number of changed voxels.");
}
