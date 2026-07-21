/// Unit tests for ESDFGrid: trilinear interpolation, gradient, out-of-bounds.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "il_dataset/local_planner/esdf_grid.hpp"

using namespace il_dataset;

class ESDFGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 4x4x4 ESDF with a spherical obstacle
        int gx = 10, gy = 10, gz = 10;
        std::vector<float> data(gx * gy * gz, 2.0f);  // all free by default

        // Place a spherical obstacle at center (radius ~2 voxels)
        for (int ix = 0; ix < gx; ++ix) {
            for (int iy = 0; iy < gy; ++iy) {
                for (int iz = 0; iz < gz; ++iz) {
                    double cx = (ix - 5) * 0.15;
                    double cy = (iy - 5) * 0.15;
                    double cz = (iz - 5) * 0.15;
                    double dist = std::sqrt(cx*cx + cy*cy + cz*cz) - 0.3;  // 0.3m radius obstacle
                    // ESDF with drone_radius already subtracted
                    data[ix * gy * gz + iy * gz + iz] = static_cast<float>(dist);
                }
            }
        }

        grid_.setData(data.data(), gx, gy, gz, -1.0, -1.0, -1.0, 0.15);
    }

    ESDFGrid grid_;
};

TEST_F(ESDFGridTest, Initialized) {
    EXPECT_TRUE(grid_.initialized());
    EXPECT_EQ(grid_.gx(), 10);
    EXPECT_EQ(grid_.gy(), 10);
    EXPECT_EQ(grid_.gz(), 10);
}

TEST_F(ESDFGridTest, OriginCorrect) {
    EXPECT_NEAR(grid_.originX(), -1.0, 1e-6);
    EXPECT_NEAR(grid_.originY(), -1.0, 1e-6);
    EXPECT_NEAR(grid_.originZ(), -1.0, 1e-6);
    EXPECT_NEAR(grid_.resolution(), 0.15, 1e-6);
}

TEST_F(ESDFGridTest, GetValueInsideObstacle) {
    // Center of grid is at (5+0.5)*0.15 - 1.0 = -0.175 in each axis
    double val = grid_.getValue(-0.175, -0.175, -0.175);
    // This is inside the obstacle — should be negative
    EXPECT_LT(val, 0.0);
}

TEST_F(ESDFGridTest, GetValueFreeSpace) {
    // Far from center: (0+0.5)*0.15 - 1.0 = -0.925
    double val = grid_.getValue(-0.925, -0.925, -0.925);
    // This is far from the obstacle — should be positive
    EXPECT_GT(val, 1.0);
}

TEST_F(ESDFGridTest, GetValueOutsideMap) {
    // Outside the grid bounds
    double val = grid_.getValue(-100.0, -100.0, -100.0);
    EXPECT_LT(val, 0.0);  // sentinel = collision
}

TEST_F(ESDFGridTest, TrilinearInterpolationSmooth) {
    // Test that trilinear interpolation is continuous
    double v1 = grid_.getValue(-0.4, -0.4, -0.4);
    double v2 = grid_.getValue(-0.41, -0.4, -0.4);
    // Values should be close for small perturbations
    EXPECT_NEAR(v1, v2, std::abs(v1) * 0.5 + 0.01);
}

TEST_F(ESDFGridTest, GradientDirection) {
    // Gradient should point away from obstacle (toward increasing ESDF)
    double clearance = 0.0;
    // Point near obstacle boundary, gradient should point outward
    Eigen::Vector3d grad = grid_.getGradient(-0.2, -0.175, -0.175, &clearance);
    // At position x=-0.2 (closer to obstacle at center ~ -0.175), gradient should point negative x
    // (toward more negative x = away from obstacle center which is at -0.175)
    // Actually obstacle center is at approx (-0.175, -0.175, -0.175)
    // Point at (-0.2, ...) is left of center, gradient should point left (more negative x)
    EXPECT_LT(grad.x(), 0.0);  // gradient points away from obstacle
}

TEST_F(ESDFGridTest, IsFree) {
    EXPECT_FALSE(grid_.isFree(-0.175, -0.175, -0.175, 0.0));   // inside obstacle
    EXPECT_TRUE(grid_.isFree(-0.925, -0.925, -0.925, 0.0));     // free space
    EXPECT_FALSE(grid_.isFree(-0.175, -0.175, -0.175, 0.5));    // clearance too low
}

TEST_F(ESDFGridTest, EmptyGridNotInitialized) {
    ESDFGrid empty;
    EXPECT_FALSE(empty.initialized());
    EXPECT_LT(empty.getValue(0, 0, 0), 0.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
