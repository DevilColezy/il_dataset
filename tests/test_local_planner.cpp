/// Unit tests for LocalPlanner: progress tracking, local goal, plan, validate.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "il_dataset/local_planner/types.hpp"
#include "il_dataset/local_planner/esdf_grid.hpp"
#include "il_dataset/local_planner/local_planner.hpp"

using namespace il_dataset;

class LocalPlannerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple open ESDF (no obstacles) — 20x20x20 grid
        int gx = 20, gy = 20, gz = 10;
        std::vector<float> data(gx * gy * gz, 3.0f);  // all free, clearance = 3m

        cfg_.min_clearance = 0.35;
        cfg_.target_clearance = 0.55;
        cfg_.collision_check_spacing = 0.05;
        cfg_.lookahead_distance = 4.0;
        cfg_.horizon_time = 2.5;
        cfg_.trajectory_dt = 0.04;
        cfg_.control_points = 12;
        cfg_.max_iterations = 20;
        cfg_.nominal_speed = 1.8;
        cfg_.max_velocity = 2.5;
        cfg_.max_acceleration = 3.5;
        cfg_.max_yaw_rate = 2.0;

        planner_ = std::make_unique<LocalPlanner>(cfg_);

        // Set ESDF with origin at (0, 0, 0), resolution 0.15
        bool ok = planner_->setESDF(data.data(), gx, gy, gz,
                                     0.0, 0.0, 0.0, 0.15);
        ASSERT_TRUE(ok);

        // Set a straight global path from (1, 1, 2) to (1, 16, 2) — 15m long
        std::vector<double> path;
        for (int i = 0; i <= 15; ++i) {
            path.push_back(1.0);       // x
            path.push_back(1.0 + i);   // y
            path.push_back(2.0);       // z
        }
        ok = planner_->setGlobalPath(path.data(), 16);
        ASSERT_TRUE(ok);
    }

    LocalPlannerConfig cfg_;
    std::unique_ptr<LocalPlanner> planner_;
};

// ── Progress tracking tests ─────────────────────────────────────────

TEST_F(LocalPlannerTest, ProgressOnPath) {
    // Drone exactly on the path at y=5.0
    VehicleState state;
    state.position = Eigen::Vector3d(1.0, 5.0, 2.0);
    state.velocity = Eigen::Vector3d(0.0, 1.0, 0.0);
    state.yaw = 0.0;

    auto result = planner_->planLocal(state, -1.0);
    EXPECT_TRUE(result.success);
    // Progress should be ~4.0 (start at y=1, current at y=5 → 4m)
    EXPECT_NEAR(result.progress_s, 4.0, 1.0);
}

TEST_F(LocalPlannerTest, ProgressMonotonic) {
    VehicleState s1;
    s1.position = Eigen::Vector3d(1.0, 3.0, 2.0);
    s1.velocity = Eigen::Vector3d(0.0, 1.0, 0.0);
    auto r1 = planner_->planLocal(s1, -1.0);

    VehicleState s2;
    s2.position = Eigen::Vector3d(1.0, 5.0, 2.0);
    s2.velocity = Eigen::Vector3d(0.0, 1.0, 0.0);
    auto r2 = planner_->planLocal(s2, r1.progress_s);

    EXPECT_GT(r2.progress_s, r1.progress_s);  // progress should increase
}

// ── Straight-line planning (no obstacles) ───────────────────────────

TEST_F(LocalPlannerTest, StraightPathNoObstacles) {
    VehicleState state;
    state.position = Eigen::Vector3d(1.0, 1.0, 2.0);
    state.velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
    state.yaw = 0.0;

    auto result = planner_->planLocal(state, -1.0);
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.status, PlannerStatus::SUCCESS);
    EXPECT_GT(result.trajectory.size(), 10u);

    // Trajectory should be roughly straight (max lateral deviation < 1.0m for this simple case)
    for (const auto& pt : result.trajectory) {
        EXPECT_NEAR(pt.position.x(), 1.0, 1.0);  // should stay near x=1
        EXPECT_NEAR(pt.position.z(), 2.0, 0.5);  // should stay near z=2
    }
}

// ── Validation tests ─────────────────────────────────────────────────

TEST_F(LocalPlannerTest, ValidateClearTrajectory) {
    std::vector<TrajectoryPoint> traj;
    for (int i = 0; i < 10; ++i) {
        TrajectoryPoint pt;
        pt.t = i * 0.04;
        pt.position = Eigen::Vector3d(1.0, 1.0 + i * 0.5, 2.0);
        traj.push_back(pt);
    }

    auto val = planner_->validateTrajectory(traj);
    EXPECT_TRUE(val.all_clear);
    EXPECT_FALSE(val.any_collision);
    EXPECT_EQ(val.clearance_violation_count, 0);
}

TEST_F(LocalPlannerTest, RejectCollisionTrajectory) {
    // Create a trajectory that goes outside the map
    std::vector<TrajectoryPoint> traj;
    for (int i = 0; i < 10; ++i) {
        TrajectoryPoint pt;
        pt.t = i * 0.04;
        pt.position = Eigen::Vector3d(-100.0, -100.0, -100.0);  // outside map
        traj.push_back(pt);
    }

    auto val = planner_->validateTrajectory(traj);
    EXPECT_FALSE(val.all_clear);
    EXPECT_TRUE(val.any_collision);
}

// ── Local goal tests ─────────────────────────────────────────────────

TEST_F(LocalPlannerTest, LocalGoalNearStart) {
    VehicleState state;
    state.position = Eigen::Vector3d(1.0, 1.0, 2.0);
    state.velocity = Eigen::Vector3d::Zero();

    auto result = planner_->planLocal(state, -1.0);
    // Local goal should be ahead of start
    EXPECT_GT(result.local_goal.y(), state.position.y());
}

TEST_F(LocalPlannerTest, LocalGoalNearEnd) {
    VehicleState state;
    state.position = Eigen::Vector3d(1.0, 14.0, 2.0);  // near goal at y=16
    state.velocity = Eigen::Vector3d(0.0, 1.0, 0.0);

    auto result = planner_->planLocal(state, 13.0);
    // Local goal should be close to or at the final goal
    EXPECT_NEAR(result.local_goal.y(), 16.0, 3.0);
}

// ── Performance: planning time ───────────────────────────────────────

TEST_F(LocalPlannerTest, PlanningTimeReasonable) {
    VehicleState state;
    state.position = Eigen::Vector3d(1.0, 3.0, 2.0);
    state.velocity = Eigen::Vector3d(0.0, 1.0, 0.0);

    auto result = planner_->planLocal(state, -1.0);
    // Planning should take less than 100ms
    EXPECT_LT(result.planning_time_ms, 100.0);
}

// ── Plan ID increments ───────────────────────────────────────────────

TEST_F(LocalPlannerTest, PlanIdMonotonic) {
    VehicleState state;
    state.position = Eigen::Vector3d(1.0, 3.0, 2.0);
    state.velocity = Eigen::Vector3d::Zero();

    auto r1 = planner_->planLocal(state, -1.0);
    auto r2 = planner_->planLocal(state, r1.progress_s);
    EXPECT_GT(r2.plan_id, r1.plan_id);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
