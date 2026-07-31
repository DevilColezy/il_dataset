#include "il_dataset/local_planner/local_planner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {

using il_dataset::LocalPlanner;
using il_dataset::LocalPlannerConfig;
using il_dataset::LocalPlanningRequest;
using il_dataset::PlannerStatus;

bool checkKinematicConsistency(
    const std::vector<il_dataset::TrajectoryPoint>& trajectory,
    double* max_velocity_residual) {
    *max_velocity_residual = 0.0;
    for (size_t i = 1; i < trajectory.size(); ++i) {
        const double dt = trajectory[i].t - trajectory[i - 1].t;
        if (!(dt > 0.0)) return false;
        const Eigen::Vector3d interval_velocity =
            (trajectory[i].position - trajectory[i - 1].position) / dt;
        const Eigen::Vector3d integrated_velocity =
            0.5 * (trajectory[i].velocity +
                   trajectory[i - 1].velocity);
        *max_velocity_residual = std::max(
            *max_velocity_residual,
            (interval_velocity - integrated_velocity).norm());
    }
    return *max_velocity_residual <= 0.15;
}

double trajectoryLength(
    const std::vector<il_dataset::TrajectoryPoint>& trajectory) {
    double length = 0.0;
    for (size_t i = 1; i < trajectory.size(); ++i) {
        length +=
            (trajectory[i].position -
             trajectory[i - 1].position).norm();
    }
    return length;
}

struct DynamicPeaks {
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
    double yaw_rate = 0.0;
};

DynamicPeaks measureDynamicPeaks(
    const std::vector<il_dataset::TrajectoryPoint>& trajectory) {
    DynamicPeaks peaks;
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& point = trajectory[i];
        peaks.velocity = std::max(
            peaks.velocity, point.velocity.norm());
        peaks.acceleration = std::max(
            peaks.acceleration, point.acceleration.norm());
        peaks.yaw_rate = std::max(
            peaks.yaw_rate, std::abs(point.yaw_rate));
        if (i == 0) continue;
        const double dt = point.t - trajectory[i - 1].t;
        if (dt > 0.0) {
            peaks.jerk = std::max(
                peaks.jerk,
                (point.acceleration -
                 trajectory[i - 1].acceleration).norm() / dt);
        }
    }
    return peaks;
}

std::unique_ptr<LocalPlanner> makePlanner(
    const std::vector<Eigen::Vector3d>& path,
    bool with_cylinder,
    double planning_time_budget_ms = 30.0,
    int control_points = 12) {
    LocalPlannerConfig config;
    config.trajectory_dt = 0.02;
    config.max_iterations = 10000;
    config.planning_time_budget_ms = planning_time_budget_ms;
    config.control_points = control_points;
    auto planner = std::make_unique<LocalPlanner>(config);

    constexpr int gx = 101;
    constexpr int gy = 81;
    constexpr int gz = 61;
    constexpr double resolution = 0.1;
    constexpr double origin_x = -2.0;
    constexpr double origin_y = -4.0;
    constexpr double origin_z = 0.0;
    std::vector<float> esdf(gx * gy * gz, 5.0f);
    for (int ix = 0; ix < gx; ++ix) {
        for (int iy = 0; iy < gy; ++iy) {
            // ESDFGrid origin denotes the corner of voxel (0,0,0); stored
            // distance samples therefore belong to voxel centres.
            const double x =
                origin_x + (static_cast<double>(ix) + 0.5) * resolution;
            const double y =
                origin_y + (static_cast<double>(iy) + 0.5) * resolution;
            const float clearance = with_cylinder
                ? static_cast<float>(
                    std::hypot(x - 2.5, y) - 0.45)
                : 5.0f;
            for (int iz = 0; iz < gz; ++iz) {
                esdf[(ix * gy + iy) * gz + iz] = clearance;
            }
        }
    }
    if (!planner->setESDF(
            esdf.data(), gx, gy, gz,
            origin_x, origin_y, origin_z, resolution)) {
        throw std::runtime_error("setESDF failed");
    }

    std::vector<double> flat_path;
    flat_path.reserve(path.size() * 3);
    for (const auto& point : path) {
        flat_path.push_back(point.x());
        flat_path.push_back(point.y());
        flat_path.push_back(point.z());
    }
    if (!planner->setGlobalPath(
            flat_path.data(), static_cast<int>(path.size()))) {
        throw std::runtime_error("setGlobalPath failed");
    }
    return planner;
}

LocalPlanningRequest makeRequest(
    const std::vector<Eigen::Vector3d>& path,
    const Eigen::Vector3d& velocity) {
    LocalPlanningRequest request;
    request.state.position = path.front();
    request.state.velocity = velocity;
    request.state.yaw = -0.4;
    request.guide_waypoint = path.back();
    request.guide_waypoint_index =
        static_cast<int>(path.size()) - 1;
    request.trajectory_terminal = path.back();
    request.trajectory_terminal_index =
        static_cast<int>(path.size()) - 1;
    // The explicit-Guide planner must not use the global path as a local
    // tracking reference.  The field remains empty for compatibility.
    request.reference_path_segment.clear();
    request.forbid_unknown_space = false;
    return request;
}

}  // namespace

int main() {
    const std::vector<Eigen::Vector3d> straight{
        {0.0, 0.0, 1.0}, {5.0, 0.0, 1.0}};
    auto direct_planner = makePlanner(straight, false);
    const auto direct = direct_planner->planLocalWithRequest(
        makeRequest(straight, {1.0, 0.8, 0.0}));
    double direct_residual = 0.0;
    const double direct_path_length =
        trajectoryLength(direct.trajectory);
    if (!direct.success ||
        direct.status != PlannerStatus::SUCCESS ||
        direct.trajectory.empty() ||
        (direct.trajectory.front().position - straight.front()).norm() >
            1.0e-9 ||
        (direct.trajectory.back().position - straight.back()).norm() >
            1.0e-9 ||
        direct_path_length >
            1.25 * (straight.back() - straight.front()).norm() ||
        std::abs(direct.trajectory.front().yaw + 0.4) > 1.0e-9 ||
        !checkKinematicConsistency(
            direct.trajectory, &direct_residual)) {
        std::cerr << "direct trajectory failed: " << direct.message
                  << ", residual=" << direct_residual << '\n';
        return 1;
    }

    // Regression for a lockstep failure recorded in scale_transition.  The
    // optimizer had less than four milliseconds left, so the old code skipped
    // time allocation and rejected a collision-free trajectory at
    // a=3.6437 m/s^2.  Under this short budget either the regular time
    // allocation or its bounded final fallback must satisfy the same data
    // contract.  Guide and terminal remain exact because they supervise Trend
    // while this trajectory supervises Control.
    // Translate the recorded world coordinates into the compact unit-test
    // ESDF.  Translation preserves every relative geometric and dynamic
    // quantity involved in this regression.
    const Eigen::Vector3d recorded_position{
        4.000000, 0.000000, 3.334509};
    const Eigen::Vector3d recorded_guide{
        0.063178, 2.035582, 2.934087};
    const std::vector<Eigen::Vector3d> recorded_global_path{
        recorded_position, recorded_guide, {0.0, 3.0, 2.0}};
    auto deadline_retime_planner = makePlanner(
        recorded_global_path, false, 10.0, 16);
    LocalPlanningRequest deadline_retime_request;
    deadline_retime_request.state.position = recorded_position;
    deadline_retime_request.state.velocity =
        {0.021141, 0.008783, 1.754899};
    deadline_retime_request.state.acceleration =
        {-1.352858, 0.651665, 0.177886};
    deadline_retime_request.state.yaw = 0.439039;
    deadline_retime_request.guide_waypoint = recorded_guide;
    deadline_retime_request.guide_waypoint_index = 1;
    deadline_retime_request.trajectory_terminal = recorded_guide;
    deadline_retime_request.trajectory_terminal_index = 1;
    deadline_retime_request.forbid_unknown_space = false;
    const auto deadline_retime =
        deadline_retime_planner->planLocalWithRequest(
            deadline_retime_request);
    const DynamicPeaks deadline_retime_peaks =
        measureDynamicPeaks(deadline_retime.trajectory);
    double deadline_retime_residual = 0.0;
    constexpr double dynamics_tolerance = 1.02;
    if (!deadline_retime.success ||
        deadline_retime.status != PlannerStatus::SUCCESS ||
        deadline_retime.trajectory.empty() ||
        (deadline_retime.trajectory.front().position -
         recorded_position).norm() > 1.0e-9 ||
        (deadline_retime.trajectory.front().velocity -
         deadline_retime_request.state.velocity).norm() > 1.0e-8 ||
        (deadline_retime.trajectory.front().acceleration -
         deadline_retime_request.state.acceleration).norm() > 1.0e-8 ||
        (deadline_retime.trajectory.back().position -
         recorded_guide).norm() > 1.0e-9 ||
        deadline_retime.trajectory.back().acceleration.norm() >
            1.0e-8 ||
        !checkKinematicConsistency(
            deadline_retime.trajectory,
            &deadline_retime_residual) ||
        deadline_retime_peaks.velocity >
            2.5 * dynamics_tolerance ||
        deadline_retime_peaks.acceleration >
            3.5 * dynamics_tolerance ||
        deadline_retime_peaks.jerk >
            15.0 * dynamics_tolerance ||
        deadline_retime_peaks.yaw_rate >
            2.0 * dynamics_tolerance) {
        std::cerr
            << "deadline retime regression failed: "
            << deadline_retime.message
            << ", planning_ms=" << deadline_retime.planning_time_ms
            << ", residual=" << deadline_retime_residual
            << ", v=" << deadline_retime_peaks.velocity
            << ", a=" << deadline_retime_peaks.acceleration
            << ", jerk=" << deadline_retime_peaks.jerk
            << ", yaw_rate=" << deadline_retime_peaks.yaw_rate
            << '\n';
        return 10;
    }

    // The next recorded frame exposed a second failure mode.  Regular
    // B-spline time allocation had compounded 2.76 s to 9.82 s, and the
    // analytic fallback inherited that runaway horizon.  With a 1.95 m/s
    // upward boundary velocity it left the finite ESDF before returning to
    // Guide.  The fallback must restart from the seed-derived horizon.
    const Eigen::Vector3d next_recorded_position{
        4.000000, 0.000000, 3.396392};
    const Eigen::Vector3d next_recorded_guide{
        0.063794, 2.035004, 2.987073};
    LocalPlanningRequest next_deadline_request;
    next_deadline_request.state.position = next_recorded_position;
    next_deadline_request.state.velocity =
        {0.026094, 0.018600, 1.951199};
    next_deadline_request.state.acceleration =
        {-1.038654, 0.496527, 0.187559};
    next_deadline_request.state.yaw = 0.448804;
    next_deadline_request.guide_waypoint = next_recorded_guide;
    next_deadline_request.guide_waypoint_index = 1;
    next_deadline_request.trajectory_terminal = next_recorded_guide;
    next_deadline_request.trajectory_terminal_index = 1;
    next_deadline_request.forbid_unknown_space = false;
    const auto next_deadline_retime =
        deadline_retime_planner->planLocalWithRequest(
            next_deadline_request);
    const DynamicPeaks next_deadline_peaks =
        measureDynamicPeaks(next_deadline_retime.trajectory);
    double next_deadline_residual = 0.0;
    if (!next_deadline_retime.success ||
        next_deadline_retime.status != PlannerStatus::SUCCESS ||
        next_deadline_retime.trajectory.empty() ||
        (next_deadline_retime.trajectory.front().position -
         next_recorded_position).norm() > 1.0e-9 ||
        (next_deadline_retime.trajectory.front().velocity -
         next_deadline_request.state.velocity).norm() > 1.0e-8 ||
        (next_deadline_retime.trajectory.front().acceleration -
         next_deadline_request.state.acceleration).norm() > 1.0e-8 ||
        (next_deadline_retime.trajectory.back().position -
         next_recorded_guide).norm() > 1.0e-9 ||
        next_deadline_retime.trajectory.back().acceleration.norm() >
            1.0e-8 ||
        !checkKinematicConsistency(
            next_deadline_retime.trajectory,
            &next_deadline_residual) ||
        next_deadline_peaks.velocity >
            2.5 * dynamics_tolerance ||
        next_deadline_peaks.acceleration >
            3.5 * dynamics_tolerance ||
        next_deadline_peaks.jerk >
            15.0 * dynamics_tolerance ||
        next_deadline_peaks.yaw_rate >
            2.0 * dynamics_tolerance) {
        std::cerr
            << "next deadline retime regression failed: "
            << next_deadline_retime.message
            << ", planning_ms="
            << next_deadline_retime.planning_time_ms
            << ", residual=" << next_deadline_residual
            << ", v=" << next_deadline_peaks.velocity
            << ", a=" << next_deadline_peaks.acceleration
            << ", jerk=" << next_deadline_peaks.jerk
            << ", yaw_rate=" << next_deadline_peaks.yaw_rate
            << '\n';
        return 11;
    }

    // A fresh plan is generated every 1/30 s in lockstep collection.  Model
    // the Flightmare velocity controller instead of assigning the preview
    // velocity instantaneously, and preserve the preceding plan's preview
    // acceleration as the next spline boundary.  The vehicle must reach a
    // useful fraction of cruise speed instead of replaying a near-zero
    // acceleration prefix indefinitely.
    auto warm_replan_planner = makePlanner(straight, false);
    auto warm_request = makeRequest(straight, {0.0, 0.0, 0.0});
    Eigen::Vector3d measured_acceleration = Eigen::Vector3d::Zero();
    std::vector<il_dataset::TrajectoryPoint> previous_trajectory;
    constexpr double replan_dt = 1.0 / 30.0;
    for (int iteration = 0; iteration < 60; ++iteration) {
        Eigen::Vector3d continuation =
            (Eigen::Vector3d(1.8, 0.0, 0.0) -
             warm_request.state.velocity);
        if (!previous_trajectory.empty()) {
            const double sample_time = 0.08 + replan_dt;
            const auto point = std::min_element(
                previous_trajectory.begin(), previous_trajectory.end(),
                [sample_time](const auto& lhs, const auto& rhs) {
                    return std::abs(lhs.t - sample_time) <
                           std::abs(rhs.t - sample_time);
                });
            continuation = point->acceleration;
        }
        warm_request.state.acceleration =
            0.20 * measured_acceleration + 0.80 * continuation;
        const auto plan =
            warm_replan_planner->planLocalWithRequest(warm_request);
        if (!plan.success || plan.trajectory.size() < 5) {
            std::cerr << "warm replan failed at iteration "
                      << iteration << ": " << plan.message << '\n';
            return 8;
        }
        const Eigen::Vector3d command_velocity =
            plan.trajectory[4].velocity +
            0.30 * plan.trajectory[4].acceleration;
        Eigen::Vector3d controller_acceleration =
            3.0 * (command_velocity - warm_request.state.velocity);
        if (controller_acceleration.norm() > 3.5) {
            controller_acceleration *=
                3.5 / controller_acceleration.norm();
        }
        warm_request.state.position +=
            warm_request.state.velocity * replan_dt +
            0.5 * controller_acceleration * replan_dt * replan_dt;
        warm_request.state.velocity +=
            controller_acceleration * replan_dt;
        measured_acceleration = controller_acceleration;
        previous_trajectory = plan.trajectory;
    }
    const double warm_replan_speed =
        warm_request.state.velocity.norm();
    if (warm_replan_speed < 1.05 ||
        warm_request.state.position.x() < 1.40) {
        std::cerr << "warm replanning still replays the slow prefix: speed="
                  << warm_replan_speed << ", x="
                  << warm_request.state.position.x() << '\n';
        return 9;
    }

    const std::vector<Eigen::Vector3d> around_cylinder{
        {0.0, 0.0, 1.0},
        {1.5, 0.0, 1.0},
        {1.8, 0.9, 1.0},
        {3.2, 0.9, 1.0},
        {3.5, 0.0, 1.0},
        {5.0, 0.0, 1.0}};
    auto curved_planner = makePlanner(around_cylinder, true);
    const auto curved = curved_planner->planLocalWithRequest(
        makeRequest(around_cylinder, {0.8, 0.0, 0.0}));
    double curved_residual = 0.0;
    double curved_min_clearance =
        std::numeric_limits<double>::infinity();
    double curved_max_deviation = 0.0;
    double curved_avoidance_onset_fraction =
        std::numeric_limits<double>::infinity();
    double curved_path_length = 0.0;
    const Eigen::Vector3d straight_direction =
        around_cylinder.back() - around_cylinder.front();
    const double straight_length_sq =
        straight_direction.squaredNorm();
    for (size_t point_index = 0;
         point_index < curved.trajectory.size();
         ++point_index) {
        const auto& point = curved.trajectory[point_index];
        if (point_index > 0) {
            curved_path_length +=
                (point.position -
                 curved.trajectory[point_index - 1].position).norm();
        }
        curved_min_clearance =
            std::min(curved_min_clearance, point.clearance);
        const double alpha = std::max(
            0.0, std::min(
                1.0,
                (point.position - around_cylinder.front()).dot(
                    straight_direction) / straight_length_sq));
        const Eigen::Vector3d straight_point =
            around_cylinder.front() + alpha * straight_direction;
        curved_max_deviation = std::max(
            curved_max_deviation,
            (point.position - straight_point).norm());
        if ((point.position - straight_point).norm() >= 0.02) {
            curved_avoidance_onset_fraction = std::min(
                curved_avoidance_onset_fraction, alpha);
        }
    }
    if (!curved.success ||
        curved.status != PlannerStatus::SUCCESS ||
        curved.planning_time_ms >= 33.3 ||
        // The optimized spline should retain useful buffer above the 0.02 m
        // hard boundary without any post-sampling trajectory repair.
        curved_min_clearance < 0.05 ||
        curved_max_deviation < 0.10 ||
        curved_path_length >
            1.35 * (around_cylinder.back() -
                    around_cylinder.front()).norm() ||
        curved_avoidance_onset_fraction > 0.35 ||
        !checkKinematicConsistency(
            curved.trajectory, &curved_residual)) {
        std::cerr << "curved trajectory failed: " << curved.message
                  << ", residual=" << curved_residual
                  << ", planning_ms=" << curved.planning_time_ms
                  << ", min_clearance=" << curved_min_clearance
                  << ", max_deviation=" << curved_max_deviation
                  << ", path_length=" << curved_path_length
                  << ", onset_fraction="
                  << curved_avoidance_onset_fraction << '\n';
        return 2;
    }

    // Replanning at 30 Hz must execute the avoidance instead of repeatedly
    // resetting a zero-slope trajectory prefix.  Advance the state with the
    // same 80 ms trajectory sample used by the online manager.
    auto receding_avoidance_planner =
        makePlanner(around_cylinder, true);
    auto receding_avoidance_request =
        makeRequest(around_cylinder, {0.8, 0.0, 0.0});
    double receding_avoidance_min_clearance =
        std::numeric_limits<double>::infinity();
    double receding_avoidance_max_lateral = 0.0;
    for (int iteration = 0; iteration < 120; ++iteration) {
        const auto plan =
            receding_avoidance_planner->planLocalWithRequest(
                receding_avoidance_request);
        if (!plan.success || plan.trajectory.size() < 5) {
            std::cerr << "receding avoidance plan failed at iteration "
                      << iteration << ": " << plan.message
                      << ", planning_ms=" << plan.planning_time_ms
                      << ", state="
                      << receding_avoidance_request.state.position.transpose()
                      << ", velocity="
                      << receding_avoidance_request.state.velocity.transpose()
                      << '\n';
            return 3;
        }
        const auto& command = plan.trajectory[4];  // t = 0.08 s
        constexpr double control_dt = 1.0 / 30.0;
        receding_avoidance_request.state.position +=
            command.velocity * control_dt;
        receding_avoidance_request.state.velocity =
            command.velocity;
        receding_avoidance_request.state.acceleration =
            command.acceleration;
        receding_avoidance_request.state.yaw = command.yaw;
        const double analytic_clearance =
            std::hypot(
                receding_avoidance_request.state.position.x() - 2.5,
                receding_avoidance_request.state.position.y()) -
            0.45;
        receding_avoidance_min_clearance = std::min(
            receding_avoidance_min_clearance,
            analytic_clearance);
        receding_avoidance_max_lateral = std::max(
            receding_avoidance_max_lateral,
            std::abs(
                receding_avoidance_request.state.position.y()));
        if (analytic_clearance < 0.02) {
            std::cerr << "receding avoidance entered hard clearance at "
                      << "iteration " << iteration
                      << ": clearance=" << analytic_clearance << '\n';
            return 4;
        }
        if (receding_avoidance_request.state.position.x() > 3.5) {
            break;
        }
    }
    if (receding_avoidance_request.state.position.x() < 3.0 ||
        receding_avoidance_max_lateral < 0.30) {
        std::cerr << "receding avoidance kept postponing the bend: x="
                  << receding_avoidance_request.state.position.x()
                  << ", max_lateral="
                  << receding_avoidance_max_lateral << '\n';
        return 5;
    }

    // Replanning must not indefinitely replay a zero-acceleration trajectory
    // prefix. Start above the path with upward velocity and execute only the
    // first 80 ms of each newly generated plan, as the online manager does.
    const std::vector<Eigen::Vector3d> vertical_correction{
        {0.0, 0.0, 1.0}, {5.0, 0.0, 1.0}};
    auto receding_planner = makePlanner(vertical_correction, false);
    auto receding_request = makeRequest(
        vertical_correction, {0.0, 0.0, 1.0});
    receding_request.state.position = {0.0, 0.0, 3.0};
    const double initial_height = receding_request.state.position.z();
    for (int iteration = 0; iteration < 120; ++iteration) {
        const auto plan =
            receding_planner->planLocalWithRequest(receding_request);
        if (!plan.success || plan.trajectory.size() < 5) {
            std::cerr << "receding-horizon plan failed: "
                      << plan.message << '\n';
            return 6;
        }
        const auto& command = plan.trajectory[4];  // t = 0.08 s
        constexpr double control_dt = 1.0 / 30.0;
        receding_request.state.position +=
            command.velocity * control_dt;
        receding_request.state.velocity = command.velocity;
        receding_request.state.acceleration =
            command.acceleration;
        receding_request.state.yaw = command.yaw;
    }
    if (receding_request.state.position.x() <= 1.0 ||
        receding_request.state.position.z() >= initial_height - 0.75 ||
        receding_request.state.velocity.z() > 0.10) {
        std::cerr << "receding-horizon prefix did not start/correct state: x="
                  << receding_request.state.position.x() << ", z="
                  << receding_request.state.position.z()
                  << ", vz=" << receding_request.state.velocity.z() << '\n';
        return 7;
    }

    std::cout << "direct_points=" << direct.trajectory.size()
              << " direct_residual=" << direct_residual
              << " direct_path_length=" << direct_path_length
              << " deadline_retime_ms="
              << deadline_retime.planning_time_ms
              << " deadline_retime_a="
              << deadline_retime_peaks.acceleration
              << " deadline_retime_residual="
              << deadline_retime_residual
              << " next_deadline_retime_ms="
              << next_deadline_retime.planning_time_ms
              << " next_deadline_retime_a="
              << next_deadline_peaks.acceleration
              << " next_deadline_retime_residual="
              << next_deadline_residual
              << " warm_replan_speed=" << warm_replan_speed
              << " warm_replan_x=" << warm_request.state.position.x()
              << " curved_points=" << curved.trajectory.size()
              << " curved_residual=" << curved_residual
              << " curved_planning_ms=" << curved.planning_time_ms
              << " curved_min_clearance=" << curved_min_clearance
              << " curved_max_deviation=" << curved_max_deviation
              << " curved_path_length=" << curved_path_length
              << " curved_onset_fraction="
              << curved_avoidance_onset_fraction
              << " avoidance_x="
              << receding_avoidance_request.state.position.x()
              << " avoidance_max_lateral="
              << receding_avoidance_max_lateral
              << " avoidance_min_clearance="
              << receding_avoidance_min_clearance
              << " receding_x="
              << receding_request.state.position.x()
              << " receding_z="
              << receding_request.state.position.z()
              << " receding_vz="
              << receding_request.state.velocity.z() << '\n';
    return 0;
}
