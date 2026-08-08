/// pybind11 bindings for the il_dataset C++ two-level navigation stack.
///
/// Exposes:
///   - Types: VehicleState, TrajectoryPoint, MacroAction, all enums,
///     RecoverabilityResult, MacroCandidate, GoalBlocker,
///     LocalPlanRequest/Result, ControllerCommand, ValidationResult
///   - ObservedMap + ObservedMapConfig (depth integration + ESDF)
///   - ESDFGrid (debug queries)
///   - LocalRecoverability (5 Hz recoverability query)
///   - MacroCandidateSearch + analyzeGoalBlocker
///   - PrivilegedOracle (global map + cost-to-go + candidate scoring)
///   - PrivilegedInterventionOracle (macro-intervention necessity evaluator)
///   - LocalPlanner (A*-seeded B-spline, 30 Hz) + TrajectoryOptimizationConfig
///   - FlightmareDynamics (existing flightlib bridge)
///
/// The A* search (LocalPathSearch) and the yaw planner (YawPlanner) are
/// internal to the LocalPlanner / LocalRecoverability C++ implementations;
/// only their read-only result types (LocalPathResult / Status) are
/// exposed.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <cmath>
#include <memory>
#include <stdexcept>

#include "flightlib/common/command.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/objects/quadrotor.hpp"

#include "il_dataset/local_planner/types.hpp"
#include "il_dataset/local_planner/observed_map.hpp"
#include "il_dataset/local_planner/esdf_grid.hpp"
#include "il_dataset/local_planner/local_path_search.hpp"
#include "il_dataset/local_planner/local_recoverability.hpp"
#include "il_dataset/local_planner/macro_candidate_search.hpp"
#include "il_dataset/local_planner/privileged_oracle.hpp"
#include "il_dataset/local_planner/privileged_intervention_oracle.hpp"
#include "il_dataset/local_planner/task_generation_oracle.hpp"
#include "il_dataset/local_planner/spline_planner.hpp"
#include "il_dataset/local_planner/yaw_planner.hpp"

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

using namespace il_dataset;

PYBIND11_MODULE(_il_local_planner, m) {
    m.doc() = "C++ two-level navigation expert stack for IL dataset collection";

    // ── Flightmare dynamics bridge ─────────────────────────────────
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

    // ── Enums ───────────────────────────────────────────────────────
    py::enum_<PlannerStatus>(m, "PlannerStatus")
        .value("SUCCESS", PlannerStatus::SUCCESS)
        .value("INVALID_INPUT", PlannerStatus::INVALID_INPUT)
        .value("NO_SAFE_MOTION", PlannerStatus::NO_SAFE_MOTION)
        .value("LOCAL_TERMINAL_INVALID", PlannerStatus::LOCAL_TERMINAL_INVALID)
        .value("OPTIMIZATION_FAILED", PlannerStatus::OPTIMIZATION_FAILED)
        .value("COLLISION", PlannerStatus::COLLISION)
        .value("DYNAMICS_VIOLATION", PlannerStatus::DYNAMICS_VIOLATION)
        .value("UNKNOWN_SPACE", PlannerStatus::UNKNOWN_SPACE)
        .value("SEARCH_FAILED", PlannerStatus::SEARCH_FAILED)
        .value("EMERGENCY_HOLD", PlannerStatus::EMERGENCY_HOLD)
        .export_values();

    py::enum_<MacroMode>(m, "MacroMode")
        .value("DIRECT_GUIDE", MacroMode::DIRECT_GUIDE)
        .value("SIDE_GUIDE", MacroMode::SIDE_GUIDE)
        .value("OBSERVE", MacroMode::OBSERVE)
        .value("GOAL_REACHED", MacroMode::GOAL_REACHED)
        .value("FAILED", MacroMode::FAILED)
        .export_values();

    py::enum_<Side>(m, "Side")
        .value("NONE", Side::NONE)
        .value("LEFT", Side::LEFT)
        .value("RIGHT", Side::RIGHT)
        .export_values();

    py::enum_<RecoverabilityStatus>(m, "RecoverabilityStatus")
        .value("DIRECT_REJOIN_SUCCESS", RecoverabilityStatus::DIRECT_REJOIN_SUCCESS)
        .value("PARTIAL_PROGRESS_ONLY", RecoverabilityStatus::PARTIAL_PROGRESS_ONLY)
        .value("BLOCKED_BY_KNOWN", RecoverabilityStatus::BLOCKED_BY_KNOWN)
        .value("BLOCKED_BY_UNKNOWN", RecoverabilityStatus::BLOCKED_BY_UNKNOWN)
        .value("NO_SAFE_MOTION", RecoverabilityStatus::NO_SAFE_MOTION)
        .export_values();

    py::enum_<CandidateType>(m, "CandidateType")
        .value("DIRECT", CandidateType::DIRECT)
        .value("SIDE", CandidateType::SIDE)
        .value("OBSERVE", CandidateType::OBSERVE)
        .value("GOAL_FRONTIER", CandidateType::GOAL_FRONTIER)
        .value("PREVIOUS_CONTINUATION", CandidateType::PREVIOUS_CONTINUATION)
        .export_values();

    py::enum_<ExecutionMode>(m, "ExecutionMode")
        .value("TRACK_FRESH", ExecutionMode::TRACK_FRESH)
        .value("TRACK_CACHED", ExecutionMode::TRACK_CACHED)
        .value("ROTATE_ONLY", ExecutionMode::ROTATE_ONLY)
        .value("BRAKE_HOLD", ExecutionMode::BRAKE_HOLD)
        .value("EMERGENCY_STOP", ExecutionMode::EMERGENCY_STOP)
        .export_values();

    // ── Data structures ─────────────────────────────────────────────
    py::class_<VehicleState>(m, "VehicleState")
        .def(py::init<>())
        .def_readwrite("position", &VehicleState::position)
        .def_readwrite("velocity", &VehicleState::velocity)
        .def_readwrite("acceleration", &VehicleState::acceleration)
        .def_readwrite("yaw", &VehicleState::yaw)
        .def_readwrite("yaw_rate", &VehicleState::yaw_rate);

    py::class_<TrajectoryPoint>(m, "TrajectoryPoint")
        .def(py::init<>())
        .def_readwrite("t", &TrajectoryPoint::t)
        .def_readwrite("position", &TrajectoryPoint::position)
        .def_readwrite("velocity", &TrajectoryPoint::velocity)
        .def_readwrite("acceleration", &TrajectoryPoint::acceleration)
        .def_readwrite("yaw", &TrajectoryPoint::yaw)
        .def_readwrite("yaw_rate", &TrajectoryPoint::yaw_rate)
        .def_readwrite("clearance", &TrajectoryPoint::clearance);

    py::class_<MacroAction>(m, "MacroAction")
        .def(py::init<>())
        .def_readwrite("mode", &MacroAction::mode)
        .def_readwrite("committed_side", &MacroAction::committed_side)
        .def_readwrite("guide_world", &MacroAction::guide_world)
        .def_readwrite("guide_flu", &MacroAction::guide_flu)
        .def_readwrite("desired_yaw_world", &MacroAction::desired_yaw_world)
        .def_readwrite("has_desired_yaw", &MacroAction::has_desired_yaw)
        .def_readwrite("confidence", &MacroAction::confidence)
        .def_readwrite("guide_distance", &MacroAction::guide_distance)
        .def_readwrite("is_new_tick", &MacroAction::is_new_tick)
        .def_readwrite("observe_subtype", &MacroAction::observe_subtype)
        .def_readwrite("observe_side", &MacroAction::observe_side)
        .def_readwrite("reason", &MacroAction::reason);

    py::class_<RecoverabilityResult>(m, "RecoverabilityResult")
        .def(py::init<>())
        .def_readwrite("status", &RecoverabilityResult::status)
        .def_readwrite("feasible", &RecoverabilityResult::feasible)
        .def_readwrite("known_free", &RecoverabilityResult::known_free)
        .def_readwrite("minimum_clearance", &RecoverabilityResult::minimum_clearance)
        .def_readwrite("estimated_duration", &RecoverabilityResult::estimated_duration)
        .def_readwrite("goal_progress", &RecoverabilityResult::goal_progress)
        .def_readwrite("terminal_guide_alignment",
                       &RecoverabilityResult::terminal_guide_alignment)
        .def_readwrite("path_length", &RecoverabilityResult::path_length)
        .def_readwrite("rejoin_distance", &RecoverabilityResult::rejoin_distance)
        .def_readwrite("detour_ratio", &RecoverabilityResult::detour_ratio)
        .def_readwrite("rejoin_point", &RecoverabilityResult::rejoin_point)
        .def_readwrite("blocker_signature",
                       &RecoverabilityResult::blocker_signature)
        .def_readwrite("left_edge_visible", &RecoverabilityResult::left_edge_visible)
        .def_readwrite("right_edge_visible", &RecoverabilityResult::right_edge_visible)
        .def_readwrite("left_corridor_known", &RecoverabilityResult::left_corridor_known)
        .def_readwrite("right_corridor_known", &RecoverabilityResult::right_corridor_known)
        .def_readwrite("reason", &RecoverabilityResult::reason);

    py::class_<MacroCandidate>(m, "MacroCandidate")
        .def(py::init<>())
        .def_readwrite("type", &MacroCandidate::type)
        .def_readwrite("side", &MacroCandidate::side)
        .def_readwrite("position_world", &MacroCandidate::position_world)
        .def_readwrite("position_flu", &MacroCandidate::position_flu)
        .def_readwrite("known_reachable", &MacroCandidate::known_reachable)
        .def_readwrite("full_goal_reached", &MacroCandidate::full_goal_reached)
        .def_readwrite("found_partial", &MacroCandidate::found_partial)
        .def_readwrite("observed_path_cost", &MacroCandidate::observed_path_cost)
        .def_readwrite("observed_path_length", &MacroCandidate::observed_path_length)
        .def_readwrite("minimum_clearance", &MacroCandidate::minimum_clearance)
        .def_readwrite("goal_progress", &MacroCandidate::goal_progress)
        .def_readwrite("left_edge_visible", &MacroCandidate::left_edge_visible)
        .def_readwrite("right_edge_visible", &MacroCandidate::right_edge_visible)
        .def_readwrite("unknown_information_gain",
                       &MacroCandidate::unknown_information_gain)
        .def_readwrite("connected_to_goal", &MacroCandidate::connected_to_goal)
        .def_readwrite("global_cost_to_go", &MacroCandidate::global_cost_to_go)
        .def_readwrite("global_clearance", &MacroCandidate::global_clearance)
        .def_readwrite("global_path_length", &MacroCandidate::global_path_length)
        .def_readwrite("privileged_score", &MacroCandidate::privileged_score)
        .def_readwrite("source", &MacroCandidate::source);

    py::class_<GoalBlocker>(m, "GoalBlocker")
        .def(py::init<>())
        .def_readwrite("found", &GoalBlocker::found)
        .def_readwrite("blocker_signature", &GoalBlocker::blocker_signature)
        .def_readwrite("centroid", &GoalBlocker::centroid)
        .def_readwrite("bbox_min_world", &GoalBlocker::bbox_min_world)
        .def_readwrite("bbox_max_world", &GoalBlocker::bbox_max_world)
        .def_readwrite("extent", &GoalBlocker::extent)
        .def_readwrite("blocking_ray_depth", &GoalBlocker::blocking_ray_depth)
        .def_readwrite("component_cell_count", &GoalBlocker::component_cell_count)
        .def_readwrite("blocked_by_known", &GoalBlocker::blocked_by_known)
        .def_readwrite("left_edge_world", &GoalBlocker::left_edge_world)
        .def_readwrite("right_edge_world", &GoalBlocker::right_edge_world)
        .def_readwrite("left_edge_visible", &GoalBlocker::left_edge_visible)
        .def_readwrite("right_edge_visible", &GoalBlocker::right_edge_visible)
        .def_readwrite("left_corridor_known", &GoalBlocker::left_corridor_known)
        .def_readwrite("right_corridor_known", &GoalBlocker::right_corridor_known)
        .def_readwrite("left_corridor_point", &GoalBlocker::left_corridor_point)
        .def_readwrite("right_corridor_point", &GoalBlocker::right_corridor_point);

    py::class_<LocalPlanRequest>(m, "LocalPlanRequest")
        .def(py::init<>())
        .def_readwrite("state", &LocalPlanRequest::state)
        .def_readwrite("macro_guide_world", &LocalPlanRequest::macro_guide_world)
        .def_readwrite("has_macro_yaw", &LocalPlanRequest::has_macro_yaw)
        .def_readwrite("macro_yaw_world", &LocalPlanRequest::macro_yaw_world)
        .def_readwrite("goal_world", &LocalPlanRequest::goal_world)
        .def_readwrite("previous_trajectory", &LocalPlanRequest::previous_trajectory)
        .def_readwrite("previous_trajectory_age_s",
                       &LocalPlanRequest::previous_trajectory_age_s)
        .def_readwrite("committed_side", &LocalPlanRequest::committed_side)
        .def_readwrite("forbid_unknown_space", &LocalPlanRequest::forbid_unknown_space);

    py::class_<LocalPlanResult>(m, "LocalPlanResult")
        .def(py::init<>())
        .def_readwrite("success", &LocalPlanResult::success)
        .def_readwrite("status", &LocalPlanResult::status)
        .def_readwrite("message", &LocalPlanResult::message)
        .def_readwrite("trajectory", &LocalPlanResult::trajectory)
        .def_readwrite("guide_waypoint", &LocalPlanResult::guide_waypoint)
        .def_readwrite("trajectory_terminal", &LocalPlanResult::trajectory_terminal)
        .def_readwrite("min_clearance", &LocalPlanResult::min_clearance)
        .def_readwrite("duration_s", &LocalPlanResult::duration_s)
        .def_readwrite("planning_time_ms", &LocalPlanResult::planning_time_ms)
        .def_readwrite("plan_id", &LocalPlanResult::plan_id)
        .def_readwrite("search_status", &LocalPlanResult::search_status);

    py::class_<ControllerCommand>(m, "ControllerCommand")
        .def(py::init<>())
        .def_readwrite("velocity_flu", &ControllerCommand::velocity_flu)
        .def_readwrite("yaw_rate", &ControllerCommand::yaw_rate)
        .def_readwrite("valid", &ControllerCommand::valid);

    py::class_<ValidationResult>(m, "ValidationResult")
        .def(py::init<>())
        .def_readwrite("all_clear", &ValidationResult::all_clear)
        .def_readwrite("any_collision", &ValidationResult::any_collision)
        .def_readwrite("any_unknown", &ValidationResult::any_unknown)
        .def_readwrite("min_clearance", &ValidationResult::min_clearance)
        .def_readwrite("clearance_violation_count",
                       &ValidationResult::clearance_violation_count)
        .def_readwrite("worst_position", &ValidationResult::worst_position)
        .def_readwrite("worst_time", &ValidationResult::worst_time)
        .def_readwrite("worst_clearance", &ValidationResult::worst_clearance);

    // ── Brake-risk + privileged intervention result types ────────────
    py::class_<BrakeRiskResult>(m, "BrakeRiskResult")
        .def(py::init<>())
        .def_readwrite("risk", &BrakeRiskResult::risk)
        .def_readwrite("min_clearance", &BrakeRiskResult::min_clearance)
        .def_readwrite("first_risk_time", &BrakeRiskResult::first_risk_time)
        .def_readwrite("braking_distance", &BrakeRiskResult::braking_distance);

    py::enum_<InterventionReason>(m, "InterventionReason")
        .value("DIRECT_GLOBALLY_VALID", InterventionReason::DIRECT_GLOBALLY_VALID)
        .value("DIRECT_LONG_WALL_BLOCKED", InterventionReason::DIRECT_LONG_WALL_BLOCKED)
        .value("DIRECT_GLOBAL_DISCONNECTED", InterventionReason::DIRECT_GLOBAL_DISCONNECTED)
        .value("DIRECT_EXCESSIVE_DETOUR", InterventionReason::DIRECT_EXCESSIVE_DETOUR)
        .value("DIRECT_LOOP_RISK", InterventionReason::DIRECT_LOOP_RISK)
        .value("NO_GLOBAL_ROUTE", InterventionReason::NO_GLOBAL_ROUTE)
        .export_values();

    py::enum_<PrivilegedRecoverabilityFailure>(m, "PrivilegedRecoverabilityFailure")
        .value("NONE", PrivilegedRecoverabilityFailure::NONE)
        .value("NO_REJOIN_PATH", PrivilegedRecoverabilityFailure::NO_REJOIN_PATH)
        .value("EXCESSIVE_PATH_LENGTH", PrivilegedRecoverabilityFailure::EXCESSIVE_PATH_LENGTH)
        .value("EXCESSIVE_DURATION", PrivilegedRecoverabilityFailure::EXCESSIVE_DURATION)
        .value("EXCESSIVE_DETOUR", PrivilegedRecoverabilityFailure::EXCESSIVE_DETOUR)
        .value("LOW_CLEARANCE", PrivilegedRecoverabilityFailure::LOW_CLEARANCE)
        .value("LOW_GOAL_PROGRESS", PrivilegedRecoverabilityFailure::LOW_GOAL_PROGRESS)
        .value("BAD_TERMINAL_ALIGNMENT", PrivilegedRecoverabilityFailure::BAD_TERMINAL_ALIGNMENT)
        .export_values();

    py::class_<PrivilegedInterventionResult>(m, "PrivilegedInterventionResult")
        .def(py::init<>())
        .def_readwrite("privileged_local_recoverable",
                       &PrivilegedInterventionResult::privileged_local_recoverable)
        .def_readwrite("privileged_rejoin_reached",
                       &PrivilegedInterventionResult::privileged_rejoin_reached)
        .def_readwrite("privileged_rejoin_distance",
                       &PrivilegedInterventionResult::privileged_rejoin_distance)
        .def_readwrite("privileged_local_path_length",
                       &PrivilegedInterventionResult::privileged_local_path_length)
        .def_readwrite("privileged_local_duration",
                       &PrivilegedInterventionResult::privileged_local_duration)
        .def_readwrite("privileged_detour_ratio",
                       &PrivilegedInterventionResult::privileged_detour_ratio)
        .def_readwrite("privileged_min_clearance",
                       &PrivilegedInterventionResult::privileged_min_clearance)
        .def_readwrite("privileged_goal_progress",
                       &PrivilegedInterventionResult::privileged_goal_progress)
        .def_readwrite("privileged_terminal_alignment",
                       &PrivilegedInterventionResult::privileged_terminal_alignment)
        .def_readwrite("privileged_future_intervention_required",
                       &PrivilegedInterventionResult::privileged_future_intervention_required)
        .def_readwrite("failure_reason",
                       &PrivilegedInterventionResult::failure_reason)
        .def_readwrite("current_cost_to_go",
                       &PrivilegedInterventionResult::current_cost_to_go)
        .def_readwrite("direct_cost_to_go",
                       &PrivilegedInterventionResult::direct_cost_to_go)
        .def_readwrite("loop_risk", &PrivilegedInterventionResult::loop_risk)
        .def_readwrite("reason", &PrivilegedInterventionResult::reason);

    py::class_<PrivilegedInterventionConfig>(m, "PrivilegedInterventionConfig")
        .def(py::init<>())
        .def_readwrite("search_clearance_m",
                       &PrivilegedInterventionConfig::search_clearance_m)
        .def_readwrite("search_max_time_ms",
                       &PrivilegedInterventionConfig::search_max_time_ms)
        .def_readwrite("rejoin_distance_m",
                       &PrivilegedInterventionConfig::rejoin_distance_m)
        .def_readwrite("search_lateral_margin_m",
                       &PrivilegedInterventionConfig::search_lateral_margin_m)
        .def_readwrite("search_longitudinal_margin_m",
                       &PrivilegedInterventionConfig::search_longitudinal_margin_m)
        .def_readwrite("max_duration_s",
                       &PrivilegedInterventionConfig::max_duration_s)
        .def_readwrite("max_path_length_m",
                       &PrivilegedInterventionConfig::max_path_length_m)
        .def_readwrite("nominal_speed_mps",
                       &PrivilegedInterventionConfig::nominal_speed_mps)
        .def_readwrite("max_detour_ratio",
                       &PrivilegedInterventionConfig::max_detour_ratio)
        .def_readwrite("min_goal_progress_m",
                       &PrivilegedInterventionConfig::min_goal_progress_m)
        .def_readwrite("min_terminal_alignment",
                       &PrivilegedInterventionConfig::min_terminal_alignment)
        .def_readwrite("terminal_tangent_min_baseline",
                       &PrivilegedInterventionConfig::terminal_tangent_min_baseline)
        .def_readwrite("loop_ignore_recent_s",
                       &PrivilegedInterventionConfig::loop_ignore_recent_s)
        .def_readwrite("loop_leave_radius_m",
                       &PrivilegedInterventionConfig::loop_leave_radius_m)
        .def_readwrite("loop_revisit_radius_m",
                       &PrivilegedInterventionConfig::loop_revisit_radius_m)
        .def_readwrite("loop_min_speed_mps",
                       &PrivilegedInterventionConfig::loop_min_speed_mps)
        .def_readwrite("loop_min_revisits",
                       &PrivilegedInterventionConfig::loop_min_revisits)
        .def_readwrite("loop_history_size",
                       &PrivilegedInterventionConfig::loop_history_size);

    py::class_<PrivilegedInterventionOracle>(m, "PrivilegedInterventionOracle")
        .def(py::init<const PrivilegedInterventionConfig&>())
        .def("evaluate",
             [](PrivilegedInterventionOracle& self,
                const PrivilegedOracle& oracle,
                const VehicleState& state,
                const Eigen::Vector3d& direct_guide_world,
                const Eigen::Vector3d& goal_world,
                double current_time_s) {
                 py::gil_scoped_release release;
                 return self.evaluate(oracle, state, direct_guide_world,
                                      goal_world, current_time_s);
             },
             py::arg("oracle"), py::arg("state"),
             py::arg("direct_guide_world"), py::arg("goal_world"),
             py::arg("current_time_s"))
        .def("reset", &PrivilegedInterventionOracle::reset);

    // ── Local path search result (exposed read-only) ─────────────────
    py::enum_<LocalPathResult::Status>(m, "LocalPathStatus")
        .value("FULL_GOAL_REACHED", LocalPathResult::Status::FULL_GOAL_REACHED)
        .value("PARTIAL_TERMINAL_REACHED",
               LocalPathResult::Status::PARTIAL_TERMINAL_REACHED)
        .value("NO_PATH", LocalPathResult::Status::NO_PATH)
        .export_values();

    py::class_<LocalPathResult>(m, "LocalPathResult")
        .def(py::init<>())
        .def_readwrite("status", &LocalPathResult::status)
        .def_readwrite("full_goal_reached", &LocalPathResult::full_goal_reached)
        .def_readwrite("found_partial", &LocalPathResult::found_partial)
        .def_readwrite("terminal", &LocalPathResult::terminal)
        .def_readwrite("path", &LocalPathResult::path)
        .def_readwrite("path_cost", &LocalPathResult::path_cost)
        .def_readwrite("minimum_clearance", &LocalPathResult::minimum_clearance)
        .def_readwrite("failure_reason", &LocalPathResult::failure_reason)
        .def_readwrite("expanded_nodes", &LocalPathResult::expanded_nodes)
        .def_readwrite("compute_ms", &LocalPathResult::compute_ms);

    // ── ObservedMap ─────────────────────────────────────────────────
    py::class_<ObservedMapConfig>(m, "ObservedMapConfig")
        .def(py::init<>())
        .def_readwrite("resolution", &ObservedMapConfig::resolution)
        .def_readwrite("size_x_m", &ObservedMapConfig::size_x_m)
        .def_readwrite("size_y_m", &ObservedMapConfig::size_y_m)
        .def_readwrite("size_z_m", &ObservedMapConfig::size_z_m)
        .def_readwrite("history_seconds", &ObservedMapConfig::history_seconds)
        .def_readwrite("occupied_endpoint_margin_m",
                       &ObservedMapConfig::occupied_endpoint_margin_m)
        .def_readwrite("vehicle_radius_m", &ObservedMapConfig::vehicle_radius_m)
        .def_readwrite("max_depth_m", &ObservedMapConfig::max_depth_m)
        .def_readwrite("horizontal_fov_deg", &ObservedMapConfig::horizontal_fov_deg)
        .def_readwrite("esdf_max_distance_m", &ObservedMapConfig::esdf_max_distance_m)
        .def_readwrite("free_space_spacing_m", &ObservedMapConfig::free_space_spacing_m)
        .def_readwrite("depth_integration_step",
                       &ObservedMapConfig::depth_integration_step)
        .def_readwrite("rebuild_every_n_frames",
                       &ObservedMapConfig::rebuild_every_n_frames)
        .def_readwrite("recenter_threshold_m",
                       &ObservedMapConfig::recenter_threshold_m);

    py::class_<ObservedMap>(m, "ObservedMap")
        .def(py::init<const ObservedMapConfig&>(), py::arg("config"))
        .def("reset", &ObservedMap::reset, py::arg("center_world"))
        .def("recenter_if_needed", &ObservedMap::recenterIfNeeded,
             py::arg("center_world"))
        .def("integrate_depth",
             [](ObservedMap& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> depth,
                const Eigen::Vector3d& cam_pos,
                const Eigen::Vector4d& quat_xyzw,
                double timestamp_s) {
                 py::buffer_info buf = depth.request();
                 if (buf.ndim != 2) {
                     throw std::invalid_argument(
                         "depth must be a 2-D (H, W) float array in metres");
                 }
                 const int height = static_cast<int>(buf.shape[0]);
                 const int width = static_cast<int>(buf.shape[1]);
                 self.integrateDepth(static_cast<const float*>(buf.ptr),
                                     height, width, cam_pos, quat_xyzw,
                                     timestamp_s);
             },
             py::arg("depth_m"), py::arg("cam_pos_world"),
             py::arg("quat_xyzw"), py::arg("timestamp_s"))
        .def("mark_vehicle_free_bubble", &ObservedMap::markVehicleFreeBubble,
             py::arg("center_world"), py::arg("timestamp_s"))
        .def("purge_expired", &ObservedMap::purgeExpired, py::arg("now_s"))
        .def("rebuild_esdf", &ObservedMap::rebuildEsdf)
        .def("force_rebuild_esdf", &ObservedMap::forceRebuildEsdf)
        .def("is_known",
             [](const ObservedMap& self, const Eigen::Vector3d& p) {
                 return self.isKnown(p.x(), p.y(), p.z());
             })
        .def("is_known_free",
             [](const ObservedMap& self, const Eigen::Vector3d& p,
                double min_clearance) {
                 return self.isKnownFree(p.x(), p.y(), p.z(), min_clearance);
             },
             py::arg("point_world"), py::arg("min_clearance"))
        .def("esdf_value",
             [](const ObservedMap& self, const Eigen::Vector3d& p) {
                 return self.esdfValue(p.x(), p.y(), p.z());
             })
        .def("occupancy_at",
             [](const ObservedMap& self, const Eigen::Vector3d& p) {
                 return self.occupancyAt(p.x(), p.y(), p.z());
             })
        .def("get_occupancy",
             [](const ObservedMap& self) {
                 const auto& occ = self.occupancy();
                 return py::array_t<std::uint8_t>(
                     {self.gx(), self.gy(), self.gz()}, occ.data());
             })
        .def("get_known_mask",
             [](const ObservedMap& self) {
                 const auto& mask = self.knownMask();
                 return py::array_t<std::uint8_t>(
                     {self.gx(), self.gy(), self.gz()}, mask.data());
             })
        .def("get_esdf",
             [](const ObservedMap& self) {
                 const auto& esdf = self.esdf();
                 return py::array_t<float>(
                     {self.gx(), self.gy(), self.gz()}, esdf.data());
             })
        .def("origin", &ObservedMap::origin)
        .def("resolution", &ObservedMap::resolution)
        .def("gx", &ObservedMap::gx)
        .def("gy", &ObservedMap::gy)
        .def("gz", &ObservedMap::gz)
        .def("revision", &ObservedMap::revision)
        .def("esdf_revision", &ObservedMap::esdfRevision)
        .def("esdf_built", &ObservedMap::esdfBuilt)
        .def("known_count", &ObservedMap::knownCount)
        .def("occupied_count", &ObservedMap::occupiedCount)
        .def("free_count", &ObservedMap::freeCount)
        .def("unknown_count", &ObservedMap::unknownCount)
        .def("swept_brake_risk",
             [](const ObservedMap& self, const VehicleState& state,
                double reaction_delay_s, double deceleration_mps2,
                double safety_margin_m, double sample_spacing_m) {
                 py::gil_scoped_release release;
                 return self.sweptBrakeRisk(
                     state, reaction_delay_s, deceleration_mps2,
                     safety_margin_m, sample_spacing_m);
             },
             py::arg("state"), py::arg("reaction_delay_s"),
             py::arg("deceleration_mps2"), py::arg("safety_margin_m"),
             py::arg("sample_spacing_m"));

    // ── ESDFGrid (debug) ────────────────────────────────────────────
    py::class_<ESDFGrid>(m, "ESDFGrid")
        .def(py::init<>())
        .def("set_map", &ESDFGrid::setMap, py::arg("map"))
        .def("get_value", &ESDFGrid::getValue,
             py::arg("x"), py::arg("y"), py::arg("z"))
        .def("is_known_free", &ESDFGrid::isKnownFree,
             py::arg("x"), py::arg("y"), py::arg("z"),
             py::arg("min_clearance"))
        .def("initialized", &ESDFGrid::initialized);

    // ── LocalRecoverability ─────────────────────────────────────────
    py::class_<RecoverabilityConfig>(m, "RecoverabilityConfig")
        .def(py::init<>())
        .def_readwrite("rejoin_distance_m", &RecoverabilityConfig::rejoin_distance_m)
        .def_readwrite("search_clearance_m", &RecoverabilityConfig::search_clearance_m)
        .def_readwrite("max_duration_s", &RecoverabilityConfig::max_duration_s)
        .def_readwrite("max_path_length_m", &RecoverabilityConfig::max_path_length_m)
        .def_readwrite("min_goal_progress_m", &RecoverabilityConfig::min_goal_progress_m)
        .def_readwrite("min_terminal_alignment",
                       &RecoverabilityConfig::min_terminal_alignment)
        .def_readwrite("max_detour_ratio", &RecoverabilityConfig::max_detour_ratio)
        .def_readwrite("nominal_speed_mps", &RecoverabilityConfig::nominal_speed_mps)
        .def_readwrite("terminal_tangent_min_baseline",
                       &RecoverabilityConfig::terminal_tangent_min_baseline)
        .def_readwrite("side_corridor_length_m",
                       &RecoverabilityConfig::side_corridor_length_m)
        .def_readwrite("side_corridor_radius_m",
                       &RecoverabilityConfig::side_corridor_radius_m)
        .def_readwrite("edge_search_radius_m",
                       &RecoverabilityConfig::edge_search_radius_m);

    py::class_<LocalRecoverability>(m, "LocalRecoverability")
        .def(py::init<const RecoverabilityConfig&>(), py::arg("config"))
        .def("test",
             [](const LocalRecoverability& self, const ObservedMap& map,
                const VehicleState& state,
                const Eigen::Vector3d& direct_guide_world) {
                 py::gil_scoped_release release;
                 return self.test(map, state, direct_guide_world);
             },
             py::arg("map"), py::arg("state"), py::arg("direct_guide_world"));

    // ── MacroCandidateSearch ────────────────────────────────────────
    py::class_<MacroCandidateConfig>(m, "MacroCandidateConfig")
        .def(py::init<>())
        .def_readwrite("lookahead_distance_m",
                       &MacroCandidateConfig::lookahead_distance_m)
        .def_readwrite("side_corridor_length_m",
                       &MacroCandidateConfig::side_corridor_length_m)
        .def_readwrite("side_corridor_radius_m",
                       &MacroCandidateConfig::side_corridor_radius_m)
        .def_readwrite("edge_search_radius_m",
                       &MacroCandidateConfig::edge_search_radius_m)
        .def_readwrite("min_candidate_clearance_m",
                       &MacroCandidateConfig::min_candidate_clearance_m)
        .def_readwrite("candidate_spacing_m",
                       &MacroCandidateConfig::candidate_spacing_m)
        .def_readwrite("observe_step_m", &MacroCandidateConfig::observe_step_m)
        .def_readwrite("min_observe_move_distance_m",
                       &MacroCandidateConfig::min_observe_move_distance_m)
        .def_readwrite("observe_lateral_distances_m",
                       &MacroCandidateConfig::observe_lateral_distances_m)
        .def_readwrite("observe_forward_distances_m",
                       &MacroCandidateConfig::observe_forward_distances_m)
        .def_readwrite("max_viewpoint_candidates",
                       &MacroCandidateConfig::max_viewpoint_candidates)
        .def_readwrite("max_viewpoint_searches_per_tick",
                       &MacroCandidateConfig::max_viewpoint_searches_per_tick)
        .def_readwrite("min_frontier_searches_per_tick",
                       &MacroCandidateConfig::min_frontier_searches_per_tick)
        .def_readwrite("max_observe_move_distance_m",
                       &MacroCandidateConfig::max_observe_move_distance_m)
        .def_readwrite("observe_info_gain_radius_m",
                       &MacroCandidateConfig::observe_info_gain_radius_m)
        .def_readwrite("max_frontier_candidates",
                       &MacroCandidateConfig::max_frontier_candidates)
        .def_readwrite("frontier_standoff_m",
                       &MacroCandidateConfig::frontier_standoff_m)
        .def_readwrite("goal_frontier_cone_deg",
                       &MacroCandidateConfig::goal_frontier_cone_deg)
        .def_readwrite("corridor_check_spacing_m",
                       &MacroCandidateConfig::corridor_check_spacing_m)
        .def_readwrite("search_clearance_m",
                       &MacroCandidateConfig::search_clearance_m)
        .def_readwrite("search_max_time_ms",
                       &MacroCandidateConfig::search_max_time_ms)
        .def_readwrite("search_region_margin_m",
                       &MacroCandidateConfig::search_region_margin_m)
        .def_readwrite("side_bias_gain",
                       &MacroCandidateConfig::side_bias_gain);

    m.def("analyze_goal_blocker", &analyzeGoalBlocker,
          py::arg("map"), py::arg("state"), py::arg("goal_world"),
          py::arg("config"),
          "Identify the goal blocker and its visible edges/corridors.");

    py::class_<MacroCandidateSearch>(m, "MacroCandidateSearch")
        .def(py::init<const MacroCandidateConfig&>(), py::arg("config"))
        .def("generate_candidates",
             [](const MacroCandidateSearch& self, const ObservedMap& map,
                const VehicleState& state, const Eigen::Vector3d& goal_world,
                const GoalBlocker& blocker,
                const py::object& prev_candidate_world) {
                 // Convert the optional previous candidate to plain C++
                 // data BEFORE releasing the GIL (section X): touching
                 // py::object while the GIL is released is a data race.
                 Eigen::Vector3d* prev_ptr = nullptr;
                 Eigen::Vector3d prev_val;
                 if (!prev_candidate_world.is_none()) {
                     prev_val =
                         prev_candidate_world.cast<Eigen::Vector3d>();
                     prev_ptr = &prev_val;
                 }
                 py::gil_scoped_release release;
                 return self.generateCandidates(map, state, goal_world,
                                                blocker, prev_ptr);
             },
             py::arg("map"), py::arg("state"), py::arg("goal_world"),
             py::arg("blocker"), py::arg("prev_candidate_world") = py::none())
        .def("last_observe_diagnostics",
             &MacroCandidateSearch::lastObserveDiagnostics);

    py::class_<ObserveDiagnostics>(m, "ObserveDiagnostics")
        .def(py::init<>())
        .def_readonly("raw_candidate_count",
                      &ObserveDiagnostics::raw_candidate_count)
        .def_readonly("lattice_candidate_count",
                      &ObserveDiagnostics::lattice_candidate_count)
        .def_readonly("frontier_candidate_count",
                      &ObserveDiagnostics::frontier_candidate_count)
        .def_readonly("endpoint_known_free_count",
                      &ObserveDiagnostics::endpoint_known_free_count)
        .def_readonly("full_local_count",
                      &ObserveDiagnostics::full_local_count)
        .def_readonly("partial_count", &ObserveDiagnostics::partial_count)
        .def_readonly("no_path_count", &ObserveDiagnostics::no_path_count)
        .def_readonly("reject_unknown", &ObserveDiagnostics::reject_unknown)
        .def_readonly("reject_endpoint_clearance",
                      &ObserveDiagnostics::reject_endpoint_clearance)
        .def_readonly("reject_min_distance",
                      &ObserveDiagnostics::reject_min_distance)
        .def_readonly("reject_max_distance",
                      &ObserveDiagnostics::reject_max_distance);

    // ── PrivilegedOracle ────────────────────────────────────────────
    py::class_<PrivilegedOracleConfig::Scoring>(m, "PrivilegedOracleScoring")
        .def(py::init<>())
        .def_readwrite("weight_observed_cost",
                       &PrivilegedOracleConfig::Scoring::weight_observed_cost)
        .def_readwrite("weight_cost_to_go",
                       &PrivilegedOracleConfig::Scoring::weight_cost_to_go)
        .def_readwrite("weight_connectivity",
                       &PrivilegedOracleConfig::Scoring::weight_connectivity)
        .def_readwrite("weight_clearance",
                       &PrivilegedOracleConfig::Scoring::weight_clearance)
        .def_readwrite("weight_goal_progress",
                       &PrivilegedOracleConfig::Scoring::weight_goal_progress)
        .def_readwrite("weight_information",
                       &PrivilegedOracleConfig::Scoring::weight_information)
        .def_readwrite("weight_yaw_cost",
                       &PrivilegedOracleConfig::Scoring::weight_yaw_cost)
        .def_readwrite("weight_side_switch",
                       &PrivilegedOracleConfig::Scoring::weight_side_switch)
        .def_readwrite("weight_repeat",
                       &PrivilegedOracleConfig::Scoring::weight_repeat)
        .def_readwrite("side_switch_penalty",
                       &PrivilegedOracleConfig::Scoring::side_switch_penalty)
        .def_readwrite("repeat_penalty",
                       &PrivilegedOracleConfig::Scoring::repeat_penalty)
        .def_readwrite("clearance_target_m",
                       &PrivilegedOracleConfig::Scoring::clearance_target_m)
        .def_readwrite("yaw_cost_scale_rad",
                       &PrivilegedOracleConfig::Scoring::yaw_cost_scale_rad);

    py::class_<PrivilegedOracleConfig>(m, "PrivilegedOracleConfig")
        .def(py::init<>())
        .def_readwrite("resolution", &PrivilegedOracleConfig::resolution)
        .def_readwrite("vehicle_radius_m", &PrivilegedOracleConfig::vehicle_radius_m)
        .def_readwrite("inflation_m", &PrivilegedOracleConfig::inflation_m)
        .def_readwrite("max_esdf_distance_m",
                       &PrivilegedOracleConfig::max_esdf_distance_m)
        .def_readwrite("map_margin_m", &PrivilegedOracleConfig::map_margin_m)
        .def_readwrite("min_z_m", &PrivilegedOracleConfig::min_z_m)
        .def_readwrite("max_z_m", &PrivilegedOracleConfig::max_z_m)
        .def_readwrite("cost_to_go_cap_m", &PrivilegedOracleConfig::cost_to_go_cap_m)
        .def_readwrite("scoring", &PrivilegedOracleConfig::scoring);

    py::class_<PrivilegedOracle>(m, "PrivilegedOracle")
        .def(py::init<>())
        .def("build",
             [](PrivilegedOracle& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> points,
                const Eigen::Vector3d& start, const Eigen::Vector3d& goal,
                const PrivilegedOracleConfig& config) {
                 py::buffer_info buf = points.request();
                 if (buf.ndim != 2 || buf.shape[1] != 3) {
                     throw std::invalid_argument(
                         "points must be an (N, 3) float64 array");
                 }
                 const int n = static_cast<int>(buf.shape[0]);
                 const double* ptr = static_cast<const double*>(buf.ptr);
                 std::vector<Eigen::Vector3d> cloud;
                 cloud.reserve(static_cast<size_t>(n));
                 for (int i = 0; i < n; ++i) {
                     cloud.emplace_back(ptr[3 * i + 0], ptr[3 * i + 1],
                                        ptr[3 * i + 2]);
                 }
                 py::gil_scoped_release release;
                 return self.build(cloud, start, goal, config);
             },
             py::arg("points_world"), py::arg("start"), py::arg("goal"),
             py::arg("config"))
        .def("build_scene",
             [](PrivilegedOracle& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> points,
                const PrivilegedOracleConfig& config,
                const py::object& region_min, const py::object& region_max) {
                 py::buffer_info buf = points.request();
                 if (buf.ndim != 2 || buf.shape[1] != 3) {
                     throw std::invalid_argument(
                         "points must be an (N, 3) float64 array");
                 }
                 const int n = static_cast<int>(buf.shape[0]);
                 const double* ptr = static_cast<const double*>(buf.ptr);
                 std::vector<Eigen::Vector3d> cloud;
                 cloud.reserve(static_cast<size_t>(n));
                 for (int i = 0; i < n; ++i) {
                     cloud.emplace_back(ptr[3 * i + 0], ptr[3 * i + 1],
                                        ptr[3 * i + 2]);
                 }
                 Eigen::Vector3d vmin;
                 Eigen::Vector3d vmax;
                 Eigen::Vector3d* pmin = nullptr;
                 Eigen::Vector3d* pmax = nullptr;
                 if (!region_min.is_none()) {
                     vmin = region_min.cast<Eigen::Vector3d>();
                     pmin = &vmin;
                 }
                 if (!region_max.is_none()) {
                     vmax = region_max.cast<Eigen::Vector3d>();
                     pmax = &vmax;
                 }
                 py::gil_scoped_release release;
                 return self.buildScene(cloud, config, pmin, pmax);
             },
             py::arg("points_world"), py::arg("config"),
             py::arg("region_min") = py::none(),
             py::arg("region_max") = py::none())
        .def("set_task",
             [](PrivilegedOracle& self, const Eigen::Vector3d& start,
                const Eigen::Vector3d& goal) {
                 return self.setTask(start, goal);
             },
             py::arg("start"), py::arg("goal"))
        .def("built", &PrivilegedOracle::built)
        .def("task_reachable", &PrivilegedOracle::taskReachable)
        .def("start_goal_distance", &PrivilegedOracle::startGoalDistance)
        .def("is_occupied",
             [](const PrivilegedOracle& self, const Eigen::Vector3d& p) {
                 return self.isOccupied(p.x(), p.y(), p.z());
             })
        .def("clearance",
             [](const PrivilegedOracle& self, const Eigen::Vector3d& p) {
                 return self.clearance(p.x(), p.y(), p.z());
             })
        .def("is_free",
             [](const PrivilegedOracle& self, const Eigen::Vector3d& p,
                double required_clearance) {
                 return self.isFree(p.x(), p.y(), p.z(), required_clearance);
             },
             py::arg("point_world"), py::arg("required_clearance"))
        .def("cost_to_go",
             [](const PrivilegedOracle& self, const Eigen::Vector3d& p) {
                 return self.costToGo(p.x(), p.y(), p.z());
             })
        .def("connected_to_goal",
             [](const PrivilegedOracle& self, const Eigen::Vector3d& p) {
                 return self.connectedToGoal(p.x(), p.y(), p.z());
             })
        .def("direction_cost_to_go",
             [](const PrivilegedOracle& self, const Eigen::Vector3d& pos,
                const Eigen::Vector3d& dir, double distance) {
                 return self.directionCostToGo(pos, dir, distance);
             })
        .def("score_candidates",
             [](const PrivilegedOracle& self,
                std::vector<MacroCandidate>& candidates,
                const VehicleState& state, const Eigen::Vector3d& goal,
                Side committed_side,
                const py::object& previous_guide_world) {
                 Eigen::Vector3d* prev_ptr = nullptr;
                 Eigen::Vector3d prev_val;
                 if (!previous_guide_world.is_none()) {
                     prev_val = previous_guide_world.cast<Eigen::Vector3d>();
                     prev_ptr = &prev_val;
                 }
                 self.scoreCandidates(&candidates, state, goal,
                                      committed_side, prev_ptr);
             },
             py::arg("candidates"), py::arg("state"), py::arg("goal"),
             py::arg("committed_side"),
             py::arg("previous_guide_world") = py::none())
        .def("privileged_best_side",
             [](const PrivilegedOracle& self,
                const std::vector<MacroCandidate>& candidates) {
                 double margin = 0.0;
                 const Side side = self.privilegedBestSide(candidates, &margin);
                 return py::make_tuple(side, margin);
             },
             py::arg("candidates"))
        .def("gx", &PrivilegedOracle::gx)
        .def("gy", &PrivilegedOracle::gy)
        .def("gz", &PrivilegedOracle::gz)
        .def("origin", &PrivilegedOracle::origin)
        .def("resolution", &PrivilegedOracle::resolution)
        .def("get_esdf",
             [](const PrivilegedOracle& self) {
                 const auto& esdf = self.esdf();
                 return py::array_t<float>(
                     {self.gx(), self.gy(), self.gz()}, esdf.data());
             })
        .def("get_cost_to_go",
             [](const PrivilegedOracle& self) {
                 const auto& ctg = self.costToGoGrid();
                 return py::array_t<float>({self.gx(), self.gy()},
                                           ctg.data());
             });

    // ── LocalPlanner (B-spline) ─────────────────────────────────────
    py::class_<TrajectoryOptimizationConfig>(m, "TrajectoryOptimizationConfig")
        .def(py::init<>())
        .def_readwrite("planning_time_budget_ms",
                       &TrajectoryOptimizationConfig::planning_time_budget_ms)
        .def_readwrite("trajectory_dt", &TrajectoryOptimizationConfig::trajectory_dt)
        .def_readwrite("horizon_time", &TrajectoryOptimizationConfig::horizon_time)
        .def_readwrite("optimizer", &TrajectoryOptimizationConfig::optimizer)
        .def_readwrite("control_points", &TrajectoryOptimizationConfig::control_points)
        .def_readwrite("max_iterations", &TrajectoryOptimizationConfig::max_iterations)
        .def_readwrite("convergence_tolerance",
                       &TrajectoryOptimizationConfig::convergence_tolerance)
        .def_readwrite("initial_step_size",
                       &TrajectoryOptimizationConfig::initial_step_size)
        .def_readwrite("minimum_step_size",
                       &TrajectoryOptimizationConfig::minimum_step_size)
        .def_readwrite("seed_trust_radius",
                       &TrajectoryOptimizationConfig::seed_trust_radius)
        .def_readwrite("horizontal_avoidance_only",
                       &TrajectoryOptimizationConfig::horizontal_avoidance_only)
        .def_readwrite("min_clearance", &TrajectoryOptimizationConfig::min_clearance)
        .def_readwrite("target_clearance",
                       &TrajectoryOptimizationConfig::target_clearance)
        .def_readwrite("collision_check_spacing",
                       &TrajectoryOptimizationConfig::collision_check_spacing)
        .def_readwrite("weight_path_length",
                       &TrajectoryOptimizationConfig::weight_path_length)
        .def_readwrite("weight_smooth", &TrajectoryOptimizationConfig::weight_smooth)
        .def_readwrite("weight_jerk", &TrajectoryOptimizationConfig::weight_jerk)
        .def_readwrite("weight_obstacle",
                       &TrajectoryOptimizationConfig::weight_obstacle)
        .def_readwrite("weight_dynamics",
                       &TrajectoryOptimizationConfig::weight_dynamics)
        .def_readwrite("nominal_speed", &TrajectoryOptimizationConfig::nominal_speed)
        .def_readwrite("max_velocity", &TrajectoryOptimizationConfig::max_velocity)
        .def_readwrite("max_acceleration",
                       &TrajectoryOptimizationConfig::max_acceleration)
        .def_readwrite("max_jerk", &TrajectoryOptimizationConfig::max_jerk)
        .def_readwrite("lookahead_distance",
                       &TrajectoryOptimizationConfig::lookahead_distance)
        .def_readwrite("terminal_speed_ratio",
                       &TrajectoryOptimizationConfig::terminal_speed_ratio)
        .def_readwrite("goal_stop_tolerance_m",
                       &TrajectoryOptimizationConfig::goal_stop_tolerance_m)
        .def_readwrite("warm_start_max_age_s",
                       &TrajectoryOptimizationConfig::warm_start_max_age_s)
        .def_readwrite("warm_start_max_terminal_deviation_m",
                       &TrajectoryOptimizationConfig::warm_start_max_terminal_deviation_m)
        .def_readwrite("search_clearance_m",
                       &TrajectoryOptimizationConfig::search_clearance_m)
        .def_readwrite("search_max_time_ms",
                       &TrajectoryOptimizationConfig::search_max_time_ms)
        .def_readwrite("search_region_margin_m",
                       &TrajectoryOptimizationConfig::search_region_margin_m)
        .def_readwrite("search_side_bias_gain",
                       &TrajectoryOptimizationConfig::search_side_bias_gain)
        .def_readwrite("yaw_max_rate", &TrajectoryOptimizationConfig::yaw_max_rate)
        .def_readwrite("yaw_max_accel", &TrajectoryOptimizationConfig::yaw_max_accel)
        .def_readwrite("yaw_fov_half_deg",
                       &TrajectoryOptimizationConfig::yaw_fov_half_deg)
        .def_readwrite("yaw_fov_margin_deg",
                       &TrajectoryOptimizationConfig::yaw_fov_margin_deg)
        .def_readwrite("yaw_speed_threshold_mps",
                       &TrajectoryOptimizationConfig::yaw_speed_threshold_mps);

    py::class_<LocalPlanner>(m, "LocalPlanner")
        .def(py::init<const TrajectoryOptimizationConfig&>(), py::arg("config"))
        .def("set_map", &LocalPlanner::setMap, py::arg("map"))
        .def("plan",
             [](const LocalPlanner& self, const LocalPlanRequest& request) {
                 py::gil_scoped_release release;
                 return self.plan(request);
             },
             py::arg("request"),
             "Plan a 30 Hz local trajectory. Releases the GIL.")
        .def("validate_trajectory", &LocalPlanner::validateTrajectory,
             py::arg("trajectory"))
        .def("validate_trajectory_suffix",
             [](const LocalPlanner& self,
                const std::vector<TrajectoryPoint>& trajectory,
                double plan_start_time, double current_time,
                const VehicleState& state,
                double min_clearance, double max_position_error,
                double max_velocity_error) {
                 py::gil_scoped_release release;
                 return self.validateTrajectorySuffix(
                     trajectory, plan_start_time, current_time, state,
                     min_clearance, max_position_error, max_velocity_error);
             },
             py::arg("trajectory"), py::arg("plan_start_time"),
             py::arg("current_time"), py::arg("state"),
             py::arg("min_clearance"), py::arg("max_position_error"),
             py::arg("max_velocity_error"))
        .def("current_plan_id", &LocalPlanner::currentPlanId);

    // ── TaskGenerationOracle (scene/task generation time only) ──────
    py::class_<TaskCandidateResult>(m, "TaskCandidateResult")
        .def(py::init<>())
        .def_readwrite("start_free", &TaskCandidateResult::start_free)
        .def_readwrite("goal_free", &TaskCandidateResult::goal_free)
        .def_readwrite("goal_reachable", &TaskCandidateResult::goal_reachable)
        .def_readwrite("straight_distance",
                       &TaskCandidateResult::straight_distance)
        .def_readwrite("global_path_length",
                       &TaskCandidateResult::global_path_length)
        .def_readwrite("global_detour_ratio",
                       &TaskCandidateResult::global_detour_ratio)
        .def_readwrite("global_min_clearance",
                       &TaskCandidateResult::global_min_clearance)
        .def_readwrite("direct_blocked", &TaskCandidateResult::direct_blocked)
        .def_readwrite("direct_blocker_count",
                       &TaskCandidateResult::direct_blocker_count)
        .def_readwrite("nearest_blocker_distance_m",
                       &TaskCandidateResult::nearest_blocker_distance_m)
        .def_readwrite("privileged_local_recoverable",
                       &TaskCandidateResult::privileged_local_recoverable)
        .def_readwrite("left_global_feasible",
                       &TaskCandidateResult::left_global_feasible)
        .def_readwrite("right_global_feasible",
                       &TaskCandidateResult::right_global_feasible)
        .def_readwrite("left_path_length",
                       &TaskCandidateResult::left_path_length)
        .def_readwrite("right_path_length",
                       &TaskCandidateResult::right_path_length)
        .def_readwrite("reason", &TaskCandidateResult::reason);

    py::class_<TaskGenerationConfig>(m, "TaskGenerationConfig")
        .def(py::init<>())
        .def_readwrite("start_clearance_m",
                       &TaskGenerationConfig::start_clearance_m)
        .def_readwrite("goal_clearance_m",
                       &TaskGenerationConfig::goal_clearance_m)
        .def_readwrite("direct_corridor_clearance_m",
                       &TaskGenerationConfig::direct_corridor_clearance_m)
        .def_readwrite("min_task_distance_m",
                       &TaskGenerationConfig::min_task_distance_m)
        .def_readwrite("max_task_distance_m",
                       &TaskGenerationConfig::max_task_distance_m)
        .def_readwrite("lateral_probe_offset_m",
                       &TaskGenerationConfig::lateral_probe_offset_m)
        .def_readwrite("lateral_probe_spacing_m",
                       &TaskGenerationConfig::lateral_probe_spacing_m)
        .def_readwrite("lateral_probe_count",
                       &TaskGenerationConfig::lateral_probe_count)
        .def_readwrite("lateral_path_clearance_m",
                       &TaskGenerationConfig::lateral_path_clearance_m)
        .def_readwrite("search_clearance_m",
                       &TaskGenerationConfig::search_clearance_m)
        .def_readwrite("search_max_time_ms",
                       &TaskGenerationConfig::search_max_time_ms)
        .def_readwrite("rejoin_distance_m",
                       &TaskGenerationConfig::rejoin_distance_m)
        .def_readwrite("max_duration_s",
                       &TaskGenerationConfig::max_duration_s)
        .def_readwrite("max_path_length_m",
                       &TaskGenerationConfig::max_path_length_m)
        .def_readwrite("nominal_speed_mps",
                       &TaskGenerationConfig::nominal_speed_mps)
        .def_readwrite("max_detour_ratio",
                       &TaskGenerationConfig::max_detour_ratio)
        .def_readwrite("min_goal_progress_m",
                       &TaskGenerationConfig::min_goal_progress_m)
        .def_readwrite("min_terminal_alignment",
                       &TaskGenerationConfig::min_terminal_alignment)
        .def_readwrite("terminal_tangent_min_baseline",
                       &TaskGenerationConfig::terminal_tangent_min_baseline)
        .def_readwrite("search_lateral_margin_m",
                       &TaskGenerationConfig::search_lateral_margin_m)
        .def_readwrite("search_longitudinal_margin_m",
                       &TaskGenerationConfig::search_longitudinal_margin_m);

    py::class_<TaskGenerationOracle>(m, "TaskGenerationOracle")
        .def(py::init<const TaskGenerationConfig&>(), py::arg("config"))
        .def("evaluate",
             [](const TaskGenerationOracle& self,
                const PrivilegedOracle& oracle,
                const Eigen::Vector3d& start,
                const Eigen::Vector3d& goal) {
                 return self.evaluate(oracle, start, goal);
             },
             py::arg("oracle"), py::arg("start"), py::arg("goal"))
        .def("evaluate_candidates",
             [](const TaskGenerationOracle& self,
                const PrivilegedOracle& oracle,
                py::array_t<double, py::array::c_style | py::array::forcecast> starts,
                py::array_t<double, py::array::c_style | py::array::forcecast> goals) {
                 auto starts_buf = starts.request();
                 auto goals_buf = goals.request();
                 if (starts_buf.ndim != 2 || starts_buf.shape[1] != 3 ||
                     goals_buf.ndim != 2 || goals_buf.shape[1] != 3) {
                     throw std::invalid_argument(
                         "starts/goals must be (N, 3) float64 arrays");
                 }
                 const int n = static_cast<int>(starts_buf.shape[0]);
                 const double* sp = static_cast<const double*>(starts_buf.ptr);
                 const double* gp = static_cast<const double*>(goals_buf.ptr);
                 std::vector<Eigen::Vector3d> sv;
                 std::vector<Eigen::Vector3d> gv;
                 sv.reserve(static_cast<size_t>(n));
                 gv.reserve(static_cast<size_t>(n));
                 for (int i = 0; i < n; ++i) {
                     sv.emplace_back(sp[3 * i + 0], sp[3 * i + 1],
                                     sp[3 * i + 2]);
                     gv.emplace_back(gp[3 * i + 0], gp[3 * i + 1],
                                     gp[3 * i + 2]);
                 }
                 py::gil_scoped_release release;
                 return self.evaluateCandidates(oracle, sv, gv);
             },
             py::arg("oracle"), py::arg("starts"), py::arg("goals"));
}
