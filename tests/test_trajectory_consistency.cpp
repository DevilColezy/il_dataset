#include "il_dataset/local_planner/local_planner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
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

std::unique_ptr<LocalPlanner> makePlanner(
    const std::vector<Eigen::Vector3d>& path,
    bool with_cylinder) {
    LocalPlannerConfig config;
    config.trajectory_dt = 0.02;
    config.max_iterations = 40;
    config.planning_time_budget_ms = 200.0;
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
            const double x = origin_x + ix * resolution;
            const double y = origin_y + iy * resolution;
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
    request.reference_path_segment = path;
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
    if (!direct.success ||
        direct.status != PlannerStatus::SUCCESS ||
        direct.trajectory.empty() ||
        (direct.trajectory.front().position - straight.front()).norm() >
            1.0e-9 ||
        (direct.trajectory.back().position - straight.back()).norm() >
            1.0e-9 ||
        std::abs(direct.trajectory.front().yaw + 0.4) > 1.0e-9 ||
        !checkKinematicConsistency(
            direct.trajectory, &direct_residual)) {
        std::cerr << "direct trajectory failed: " << direct.message
                  << ", residual=" << direct_residual << '\n';
        return 1;
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
    if (!curved.success ||
        curved.status != PlannerStatus::SUCCESS ||
        !checkKinematicConsistency(
            curved.trajectory, &curved_residual)) {
        std::cerr << "curved trajectory failed: " << curved.message
                  << ", residual=" << curved_residual << '\n';
        return 2;
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
            return 3;
        }
        const auto& command = plan.trajectory[4];  // t = 0.08 s
        constexpr double control_dt = 1.0 / 30.0;
        receding_request.state.position +=
            command.velocity * control_dt;
        receding_request.state.velocity = command.velocity;
        receding_request.state.yaw = command.yaw;
    }
    if (receding_request.state.position.x() <= 1.0 ||
        receding_request.state.position.z() >= initial_height - 0.75 ||
        receding_request.state.velocity.z() > 0.10) {
        std::cerr << "receding-horizon prefix did not start/correct state: x="
                  << receding_request.state.position.x() << ", z="
                  << receding_request.state.position.z()
                  << ", vz=" << receding_request.state.velocity.z() << '\n';
        return 4;
    }

    std::cout << "direct_points=" << direct.trajectory.size()
              << " direct_residual=" << direct_residual
              << " curved_points=" << curved.trajectory.size()
              << " curved_residual=" << curved_residual
              << " receding_x="
              << receding_request.state.position.x()
              << " receding_z="
              << receding_request.state.position.z()
              << " receding_vz="
              << receding_request.state.velocity.z() << '\n';
    return 0;
}
