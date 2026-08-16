// ═══════════════════════════════════════════════════════════════════
//  test_blueprint_geometry.cpp
//  Pure C++17 static sanity tests for the OFFLINE Scene/Task Blueprint
//  pipeline (il_dataset-feature-v4 round 2 fixes):
//    * analytic ray-circle / ray-rect / rayNearestObstacleHit;
//    * SceneProfileGenerator realizations (empty / r=6 blocker / 30 tiny
//      / clustered / corridor / bottleneck / chicane);
//    * TaskCandidateGenerator layered initial-yaw coverage (±45 / ±170);
//    * BlueprintGenerationController PREFLIGHT BUDGET gate: with
//      max_total_preflight_tasks = 10 the total ATTEMPT counter (success
//      + failure) must never exceed 10 (this is the P0 bug: the old loop
//      only counted SUCCESSES and could burn unlimited failed attempts).
//
//  NOTE: this is a build-time / CI entry.  It is compiled against
//  il_hierarchical_expert_lib exactly like the pybind module.  It is NOT
//  run on the Windows editing box (no catkin / flightlib / Eigen toolchain
//  here) — it is meant for `catkin build il_dataset --cmake-args
//  -DIL_DATASET_BUILD_TESTS=ON` followed by `catkin run_tests` or running
//  the produced executable directly.
// ═══════════════════════════════════════════════════════════════════

#include "il_dataset/hierarchical_expert/ray_cast_2d.hpp"
#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"
#include "il_dataset/hierarchical_expert/scene_profile_generator.hpp"
#include "il_dataset/hierarchical_expert/scene_geometry_cache.hpp"
#include "il_dataset/hierarchical_expert/task_candidate_generator.hpp"
#include "il_dataset/hierarchical_expert/blueprint_generation_controller.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace il_dataset::expert;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "  [FAIL] %s\n", what);
    }
}

void checkNear(double a, double b, double tol, const char* what) {
    ++g_checks;
    if (!(std::fabs(a - b) <= tol)) {
        ++g_failures;
        std::fprintf(stderr, "  [FAIL] %s: got %.6f want %.6f\n", what, a, b);
    }
}

BlueprintGenerationConfig makeCfg() {
    BlueprintGenerationConfig c;
    c.warehouse = WarehouseGeometry{-7.0, 10.0, 0.0, 30.0, 1.0};
    c.base_seed = 260812;
    c.use_profile_catalog = true;
    return c;
}

}  // namespace

// ── 1. analytic ray helpers ───────────────────────────────────────
static void testRayHelpers() {
    std::fprintf(stderr, "test: ray helpers\n");
    const Vec2d ox(0.0, 0.0), dx(1.0, 0.0), dy(0.0, 1.0);

    // rayCircleHit: hit / miss / behind / inside.
    checkNear(rayCircleHit(ox, dx, Vec2d(5.0, 0.0), 1.0), 4.0, 1e-9,
              "circle hit at t=4");
    checkNear(rayCircleHit(ox, dx, Vec2d(5.0, 2.0), 1.0),
              std::numeric_limits<double>::infinity(), 0.0,
              "circle miss (off-axis)");
    checkNear(rayCircleHit(ox, dx, Vec2d(-3.0, 0.0), 1.0),
              std::numeric_limits<double>::infinity(), 0.0,
              "circle behind origin -> miss");
    checkNear(rayCircleHit(Vec2d(5.0, 0.0), dx, Vec2d(5.0, 0.0), 1.0), 1.0,
              1e-9, "origin inside circle -> exit hit");
    checkNear(rayCircleHit(ox, dy, Vec2d(0.0, 4.0), 1.0), 3.0, 1e-9,
              "circle hit along +y");

    // rayRectHit: envelope [-7,10] x [-1,31] (1 m wall shell around the
    // free region [-7,10] x [0,30]).
    const Vec2d wmin(-7.0, -1.0), wmax(10.0, 31.0);
    // Origin INSIDE the envelope (the normal camera case): the WALL is the
    // EXIT point of the ray, never t=0.
    checkNear(rayRectHit(Vec2d(0.0, 15.0), dx, wmin, wmax), 10.0, 1e-9,
              "inside envelope, +x -> exit at x=10");
    checkNear(rayRectHit(Vec2d(0.0, 15.0), Vec2d(-1.0, 0.0), wmin, wmax),
              7.0, 1e-9, "inside envelope, -x -> exit at x=-7");
    checkNear(rayRectHit(Vec2d(0.0, 15.0), dy, wmin, wmax), 16.0, 1e-9,
              "inside envelope, +y -> exit at y=31");
    checkNear(rayRectHit(Vec2d(0.0, 15.0), Vec2d(0.0, -1.0), wmin, wmax),
              16.0, 1e-9, "inside envelope, -y -> exit at y=-1");
    // Origin OUTSIDE, pointing at the envelope: entry point.
    checkNear(rayRectHit(Vec2d(20.0, 15.0), Vec2d(-1.0, 0.0), wmin, wmax),
              10.0, 1e-9, "outside envelope, -x -> entry at x=10");
    // Ray pointing away from the envelope from outside: miss.
    checkNear(rayRectHit(Vec2d(20.0, 15.0), dx, wmin, wmax),
              std::numeric_limits<double>::infinity(), 0.0,
              "outside envelope, +x -> miss");

    // rayNearestObstacleHit: circle + wall combined, nearest wins.
    {
        std::vector<Vec2d> centers{Vec2d(4.0, 0.0), Vec2d(8.0, 0.0)};
        std::vector<double> radii{0.5, 1.0};
        checkNear(rayNearestObstacleHit(Vec2d(0.0, 15.0), dx, centers, radii,
                                        true, wmin, wmax),
                  4.0 - 0.5, 1e-9,
                  "nearest obstacle (circle at 4) wins over wall at 10");
        checkNear(rayNearestObstacleHit(Vec2d(0.0, 15.0), dx,
                                        std::vector<Vec2d>{},
                                        std::vector<double>{}, true, wmin,
                                        wmax),
                  10.0, 1e-9, "no circles -> wall exit at x=10");
        checkNear(rayNearestObstacleHit(Vec2d(0.0, 15.0), dx,
                                        std::vector<Vec2d>{},
                                        std::vector<double>{}, false, wmin,
                                        wmax),
                  std::numeric_limits<double>::infinity(), 0.0,
                  "no circles, no wall -> no hit");
    }
}

// ── 2. scene profile realizations ─────────────────────────────────
static void testScenes() {
    std::fprintf(stderr, "test: scene profile realizations\n");
    const BlueprintGenerationConfig cfg = makeCfg();
    SceneProfileGenerator gen(cfg);

    struct { const char* name; bool want_empty; int lo; int hi; } cases[] = {
        {"empty", true, 0, 0},
        {"large_single", false, 1, 1},        // r in [4,6] central blocker
        {"dense_tiny", false, 20, 30},        // up to 30 tiny obstacles
        {"clustered", false, 10, 20},
        {"corridor", false, 6, 12},
        {"bottleneck", false, 4, 8},
        {"chicane", false, 6, 12},
    };
    for (const auto& c : cases) {
        const SceneProfile* p = gen.findProfile(c.name);
        check(p != nullptr, (std::string("profile exists: ") + c.name).c_str());
        if (!p) continue;
        // Both horizontal and vertical orientation paths are seeded by the
        // scene seed; realize with two different seeds to exercise both.
        bool any_ok = false;
        for (uint64_t seed : {1001ull, 1002ull, 1003ull}) {
            SceneGenerationOutcome out = gen.generate(*p, 0, seed);
            if (!out.success) continue;
            any_ok = true;
            check(out.scene.is_empty == c.want_empty,
                  (std::string("empty flag matches profile: ") + c.name)
                      .c_str());
            const int n = out.scene.actual_obstacle_count;
            check(n >= c.lo && n <= c.hi,
                  (std::string("count in band: ") + c.name).c_str());
            // Every obstacle must be inside the free region with margin.
            const double m = cfg.boundary_margin_m;
            bool inside = true;
            for (const auto& o : out.scene.obstacles) {
                if (o.x - o.radius < cfg.warehouse.free_min_x + m - 1e-6 ||
                    o.x + o.radius > cfg.warehouse.free_max_x - m + 1e-6 ||
                    o.y - o.radius < cfg.warehouse.free_min_y + m - 1e-6 ||
                    o.y + o.radius > cfg.warehouse.free_max_y - m + 1e-6) {
                    inside = false;
                }
            }
            check(inside, (std::string("all obstacles in-region: ") + c.name)
                              .c_str());
            // The planner-required passage must be respected between all
            // obstacle pairs.
            const double req = cfg.plannerRequiredPassage();
            bool gap_ok = true;
            for (size_t i = 0; i < out.scene.obstacles.size(); ++i) {
                for (size_t j = i + 1; j < out.scene.obstacles.size(); ++j) {
                    const auto& a = out.scene.obstacles[i];
                    const auto& b = out.scene.obstacles[j];
                    const double d =
                        std::hypot(a.x - b.x, a.y - b.y) - a.radius - b.radius;
                    if (d < req - 1e-6) gap_ok = false;
                }
            }
            check(gap_ok, (std::string("min surface gap >= passage: ") + c.name)
                              .c_str());
            // Geometry cache must build (planning validity) and expose the
            // obstacles it was given.
            SceneMetadata meta = out.metadata;
            SceneGeometryCache geo;
            if (geo.build(out.scene, cfg, meta)) {
                check(geo.obstacleCenters().size() ==
                          static_cast<size_t>(n),
                      (std::string("cache center count matches: ") + c.name)
                          .c_str());
                check(geo.obstacleRadii().size() ==
                          static_cast<size_t>(n),
                      (std::string("cache radius count matches: ") + c.name)
                          .c_str());
            }
            break;  // one successful realization is enough per profile
        }
        check(any_ok, (std::string("at least one successful realization: ") +
                       c.name)
                          .c_str());
    }

    // r = 6 obstacle (max radius in the region): large_single profile.
    {
        const SceneProfile* p = gen.findProfile("large_single");
        bool saw_large = false;
        for (uint64_t seed : {7ull, 8ull, 9ull}) {
            SceneGenerationOutcome out = gen.generate(*p, 0, seed);
            if (!out.success) continue;
            for (const auto& o : out.scene.obstacles) {
                if (o.radius >= 5.5) saw_large = true;
            }
        }
        check(saw_large, "large_single produces an r>=5.5 blocker");
    }
}

// ── 3. layered initial yaw coverage ───────────────────────────────
static void testInitialYaw() {
    std::fprintf(stderr, "test: layered initial yaw\n");
    const BlueprintGenerationConfig cfg = makeCfg();
    TaskCandidateGenerator gen(cfg);
    const std::vector<double> w = cfg.yaw_weights;

    // Goal exactly +45 deg off the expert +X axis -> sampleInitialYaw must
    // be able to produce BOTH +45 and -45 signed errors (left/right).
    int plus = 0, minus = 0;
    for (uint64_t s = 0; s < 4000; ++s) {
        Rng rng(s * 0x9E3779B97F4A7C15ull + 12345);
        const double yaw = gen.sampleInitialYaw(deg2rad(45.0), w, rng);
        const double err = wrapAngle(deg2rad(45.0) -
                                     CoordinateAdapter::flightmareYawToExpert(yaw));
        if (err > 1e-9) ++plus;
        if (err < -1e-9) ++minus;
    }
    check(plus > 0 && minus > 0, "yaw sampling covers both left and right");

    // ±170 deg rear goals: the outermost strata must be reachable.
    int rear_left = 0, rear_right = 0;
    for (uint64_t s = 0; s < 8000; ++s) {
        Rng rng(s * 0x9E3779B97F4A7C15ull + 999);
        const double yaw = gen.sampleInitialYaw(deg2rad(170.0), w, rng);
        const double err = wrapAngle(deg2rad(170.0) -
                                     CoordinateAdapter::flightmareYawToExpert(yaw));
        if (err > deg2rad(150.0)) ++rear_left;
        if (err < -deg2rad(150.0)) ++rear_right;
    }
    check(rear_left > 0 && rear_right > 0,
          "yaw sampling reaches the ±150..180 strata");
}

// ── 4. preflight budget gate (P0 bug regression) ──────────────────
static void testPreflightBudget() {
    std::fprintf(stderr, "test: preflight budget gate\n");
    BlueprintGenerationConfig cfg = makeCfg();
    // Tiny per-task tick budget => every preflight FAILS (goal never
    // reached).  The old code counted only SUCCESSES, so this test would
    // have burned an unbounded number of failures; the fix caps attempts.
    cfg.max_total_preflight_tasks = 10;
    cfg.max_total_preflight_ticks = 50000;
    cfg.max_preflight_ticks_per_task = 5;
    cfg.max_generation_rounds = 2;
    cfg.max_task_candidates_per_scene = 20;
    cfg.max_scene_candidates = 64;
    cfg.min_scenes = 1;
    cfg.min_selected_scenes = 1;
    cfg.min_tasks = 1;

    Params2D p;
    p.region_min_x = cfg.warehouse.free_min_x;
    p.region_max_x = cfg.warehouse.free_max_x;
    p.region_min_y = cfg.warehouse.free_min_y;
    p.region_max_y = cfg.warehouse.free_max_y;

    BlueprintGenerationController ctl(p, cfg);
    const BlueprintResult r = ctl.generate();

    check(r.preflight_attempt_count <= 10,
          "preflight attempts never exceed max_total_preflight_tasks=10");
    check(r.preflight_attempt_count ==
              r.preflight_success_count + r.preflight_failure_count,
          "attempt == success + failure (budget counts attempts, not just "
          "successes)");
    check(r.tasks_preflighted == r.preflight_attempt_count,
          "tasks_preflighted matches the attempt counter");
    check(r.full_preflight_attempted <= r.preflight_attempt_count,
          "full (non-early-terminated) attempts <= total attempts");
    check(r.preflight_failure_count > 0,
          "the tiny tick budget forces preflight failures (test validity)");
    std::fprintf(stderr,
                 "  budget result: attempts=%llu success=%llu failure=%llu "
                 "budget_reason=%s\n",
                 static_cast<unsigned long long>(r.preflight_attempt_count),
                 static_cast<unsigned long long>(r.preflight_success_count),
                 static_cast<unsigned long long>(r.preflight_failure_count),
                 r.budget_exhausted_reason.c_str());
}

int main() {
    std::fprintf(stderr, "test_blueprint_geometry: starting\n");
    testRayHelpers();
    testScenes();
    testInitialYaw();
    testPreflightBudget();
    std::fprintf(stderr, "checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
