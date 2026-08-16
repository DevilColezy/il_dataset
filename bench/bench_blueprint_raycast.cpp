// ═══════════════════════════════════════════════════════════════════
//  bench_blueprint_raycast.cpp
//  Micro-benchmark of the ANALYTIC ray cast (rayNearestObstacleHit)
//  against the OLD spatial-marching loop it replaces.  Both are executed
//  over the same synthetic scenes (obstacle counts 5 / 30 / 60) for the
//  same number of rays; wall-clock is measured with std::chrono.
//
//  Build with the same toolchain as test_blueprint_geometry.cpp
//  (IL_DATASET_BUILD_TESTS=ON).  Run it on the Linux collection box and
//  report the ratio; the analytic helper should be orders of magnitude
//  faster for dense scenes at the preflight patch resolution.
// ═══════════════════════════════════════════════════════════════════

#include "il_dataset/hierarchical_expert/ray_cast_2d.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace il_dataset::expert;

namespace {
using Clock = std::chrono::steady_clock;

/// Legacy reference: march along the ray at `step` and test every sample
/// against every circle (what synthesizePatch() did before the fix).
double marchHit(const Vec2d& o, const Vec2d& d,
                const std::vector<Vec2d>& centers,
                const std::vector<double>& radii, double range,
                double step) {
    double hit = std::numeric_limits<double>::infinity();
    for (double t = step; t <= range + 1e-9; t += step) {
        const Vec2d p = o + d * t;
        for (size_t i = 0; i < centers.size(); ++i) {
            if ((p - centers[i]).norm() < radii[i]) {
                return t;
            }
        }
    }
    return hit;
}

double runAnalytic(const std::vector<Vec2d>& centers,
                   const std::vector<double>& radii, const Vec2d& wmin,
                   const Vec2d& wmax, int n_rays, double range,
                   const std::vector<Vec2d>& dirs, const Vec2d& origin) {
    const auto t0 = Clock::now();
    for (int i = 0; i < n_rays; ++i) {
        const double hit = rayNearestObstacleHit(origin, dirs[i], centers,
                                                 radii, true, wmin, wmax);
        (void)hit;
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - t0)
        .count();
}

double runMarch(const std::vector<Vec2d>& centers,
                const std::vector<double>& radii, const Vec2d& wmin,
                const Vec2d& wmax, int n_rays, double range,
                const std::vector<Vec2d>& dirs, const Vec2d& origin) {
    const double step = 0.05;  // obs_resolution * 0.5 (preflight default)
    const auto t0 = Clock::now();
    for (int i = 0; i < n_rays; ++i) {
        const double hit = marchHit(origin, dirs[i], centers, radii, range,
                                    step);
        (void)hit;
    }
    return std::chrono::duration<double, std::milli>(Clock::now() - t0)
        .count();
}
}  // namespace

int main() {
    const int n_rays = 181;  // 90 deg FOV at 0.5 deg angular resolution
    const double range = 5.0;
    const Vec2d wmin(-7.0, -1.0), wmax(10.0, 31.0);
    const Vec2d origin(0.0, 15.0);

    std::vector<Vec2d> dirs;
    for (int i = 0; i < n_rays; ++i) {
        const double a = -M_PI / 4.0 + M_PI / 2.0 * (static_cast<double>(i) /
                                                     (n_rays - 1));
        dirs.emplace_back(std::cos(a), std::sin(a));
    }

    std::mt19937_64 rng(42);
    std::fprintf(stderr,
                 "n_obs   analytic_ms   march_ms   speedup\n");
    for (const int n : {5, 30, 60}) {
        std::vector<Vec2d> centers;
        std::vector<double> radii;
        for (int i = 0; i < n; ++i) {
            centers.emplace_back(rng() % 1500 / 100.0 - 4.0,
                                 rng() % 2500 / 100.0 + 2.0);
            radii.push_back(0.1 + (rng() % 50) / 100.0);
        }
        // Warm up.
        runAnalytic(centers, radii, wmin, wmax, n_rays, range, dirs, origin);
        runMarch(centers, radii, wmin, wmax, n_rays, range, dirs, origin);
        const double t_an = runAnalytic(centers, radii, wmin, wmax, n_rays,
                                        range, dirs, origin);
        const double t_ma = runMarch(centers, radii, wmin, wmax, n_rays,
                                     range, dirs, origin);
        std::fprintf(stderr, "%6d   %10.3f   %10.3f   %7.1fx\n", n, t_an,
                     t_ma, t_ma / std::max(1e-9, t_an));
    }
    return 0;
}
