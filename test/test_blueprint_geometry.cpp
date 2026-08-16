// ═══════════════════════════════════════════════════════════════════
//  test_blueprint_geometry.cpp
//  Pure C++17 static sanity tests for the OFFLINE Scene/Task Blueprint
//  pipeline (il_dataset-feature-v4 round 3 fixes):
//    * analytic ray-circle / ray-rect / rayNearestObstacleHit (wall
//      envelope derived from WarehouseGeometry — the SINGLE source);
//    * SceneProfileGenerator realizations (empty / r=6 blocker / 30 tiny
//      / clustered / corridor / bottleneck / chicane) + recorded
//      orientation + VERTICAL chicane alternation;
//    * TaskCandidateGenerator layered initial-yaw coverage (±45 / ±170);
//    * StallDetector regression (normal motion / stationary / pure TURN);
//    * scene consolidation never drops below min_selected_scenes;
//    * NORMAL_CORRECTION-only correction histograms (TURN excluded);
//    * HARD global tick budget (total never exceeds max);
//    * use_profile_catalog=false with no profiles => clear failure;
//    * preflight attempt budget (success + failure counted).
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
#include "il_dataset/hierarchical_expert/distribution_analyzer.hpp"
#include "il_dataset/hierarchical_expert/scene_profile_generator.hpp"
#include "il_dataset/hierarchical_expert/scene_geometry_cache.hpp"
#include "il_dataset/hierarchical_expert/stall_detector.hpp"
#include "il_dataset/hierarchical_expert/task_candidate_generator.hpp"
#include "il_dataset/hierarchical_expert/blueprint_generation_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
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

// Numeric comparison that handles +inf == +inf (fabs(inf-inf) is NaN) and
// treats NaN as NEVER equal (unless the expected value is NaN itself).
void checkNear(double a, double b, double tol, const char* what) {
    ++g_checks;
    if (std::isnan(a) || std::isnan(b)) {
        if (!(std::isnan(a) && std::isnan(b))) {
            ++g_failures;
            std::fprintf(stderr, "  [FAIL] %s: NaN mismatch (got %.6f want %.6f)\n",
                         what, a, b);
        }
        return;
    }
    if (std::isinf(a) && std::isinf(b)) {
        if (std::signbit(a) != std::signbit(b)) {
            ++g_failures;
            std::fprintf(stderr, "  [FAIL] %s: +inf/-inf mismatch\n", what);
        }
        return;
    }
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

    // ── WALL ENVELOPE FROM THE SINGLE SOURCE (WarehouseGeometry). ──
    // free region x[-7,10] y[0,30], wall_extension 1 m =>
    // envelope x[-8,11] y[-1,31].  Never hand-write the envelope here.
    WarehouseGeometry wh;
    wh.free_min_x = -7.0;
    wh.free_max_x = 10.0;
    wh.free_min_y = 0.0;
    wh.free_max_y = 30.0;
    wh.wall_extension_m = 1.0;
    const Vec2d wmin = wh.envelopeMin();  // (-8,-1)
    const Vec2d wmax = wh.envelopeMax();  // (11,31)
    checkNear(wmin.x(), -8.0, 1e-9, "envelopeMin.x == -8");
    checkNear(wmax.x(), 11.0, 1e-9, "envelopeMax.x == 11");
    checkNear(wmin.y(), -1.0, 1e-9, "envelopeMin.y == -1");
    checkNear(wmax.y(), 31.0, 1e-9, "envelopeMax.y == 31");

    // Origin INSIDE the envelope (the normal camera case): the WALL is the
    // EXIT point of the ray, never t=0.
    checkNear(rayRectHit(Vec2d(0.0, 15.0), dx, wmin, wmax), 11.0, 1e-9,
              "inside envelope, +x -> exit at x=11");
    checkNear(rayRectHit(Vec2d(0.0, 15.0), Vec2d(-1.0, 0.0), wmin, wmax),
              8.0, 1e-9, "inside envelope, -x -> exit at x=-8");
    checkNear(rayRectHit(Vec2d(0.0, 15.0), dy, wmin, wmax), 16.0, 1e-9,
              "inside envelope, +y -> exit at y=31");
    checkNear(rayRectHit(Vec2d(0.0, 15.0), Vec2d(0.0, -1.0), wmin, wmax),
              16.0, 1e-9, "inside envelope, -y -> exit at y=-1");
    // Origin OUTSIDE, pointing at the envelope: entry point.
    checkNear(rayRectHit(Vec2d(20.0, 15.0), Vec2d(-1.0, 0.0), wmin, wmax),
              9.0, 1e-9, "outside envelope, -x -> entry at x=11");
    // Ray pointing away from the envelope from outside: miss.
    checkNear(rayRectHit(Vec2d(20.0, 15.0), dx, wmin, wmax),
              std::numeric_limits<double>::infinity(), 0.0,
              "outside envelope, +x -> miss");

    // ── Wall-ray test with a concrete origin (0,10) ────────────────
    // +x wall distance = 11-0 = 11; -x = 0-(-8) = 8; +y = 31-10 = 21;
    // -y = 10-(-1) = 11.
    const Vec2d origin(0.0, 10.0);
    checkNear(rayRectHit(origin, dx, wmin, wmax), 11.0, 1e-9,
              "origin(0,10) +x wall at 11");
    checkNear(rayRectHit(origin, Vec2d(-1.0, 0.0), wmin, wmax), 8.0, 1e-9,
              "origin(0,10) -x wall at 8");
    checkNear(rayRectHit(origin, dy, wmin, wmax), 21.0, 1e-9,
              "origin(0,10) +y wall at 21");
    checkNear(rayRectHit(origin, Vec2d(0.0, -1.0), wmin, wmax), 11.0, 1e-9,
              "origin(0,10) -y wall at 11");

    // rayNearestObstacleHit: circle + wall combined, nearest wins.
    {
        std::vector<Vec2d> centers{Vec2d(4.0, 0.0), Vec2d(8.0, 0.0)};
        std::vector<double> radii{0.5, 1.0};
        // A circle closer than the wall must win.
        checkNear(rayNearestObstacleHit(Vec2d(0.0, 15.0), dx, centers, radii,
                                        true, wmin, wmax),
                  4.0 - 0.5, 1e-9,
                  "nearest obstacle (circle at 4) wins over wall at 11");
        checkNear(rayNearestObstacleHit(Vec2d(0.0, 15.0), dx,
                                        std::vector<Vec2d>{},
                                        std::vector<double>{}, true, wmin,
                                        wmax),
                  11.0, 1e-9, "no circles -> wall exit at x=11");
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
            // Directional structures must record a concrete orientation
            // and mirror it into the metadata (single source, no guessing).
            const bool directional =
                std::string(c.name) == "corridor" ||
                std::string(c.name) == "bottleneck" ||
                std::string(c.name) == "chicane";
            if (directional) {
                check(out.scene.structure_orientation ==
                              StructureOrientation::HORIZONTAL ||
                          out.scene.structure_orientation ==
                              StructureOrientation::VERTICAL,
                      (std::string("directional profile records "
                                   "orientation: ") + c.name).c_str());
                check(out.metadata.structure_orientation ==
                          structureOrientationName(
                              out.scene.structure_orientation),
                      (std::string("metadata orientation matches recorded: ") +
                       c.name).c_str());
            } else {
                check(out.scene.structure_orientation ==
                          StructureOrientation::NONE,
                      (std::string("non-directional profile has NONE "
                                   "orientation: ") + c.name).c_str());
            }
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

// ── 5. stall detector regression (P0 fix) ─────────────────────────
static void testStallDetector() {
    std::fprintf(stderr, "test: stall detector\n");
    const double dt = 1.0 / 30.0;
    const double threshold = 0.02 * dt;  // 0.02 m/s => m/tick

    // Case A — normal motion: step displacement 0.05 m/tick (1.5 m/s),
    // 120 consecutive ticks => never a stall.
    {
        StallDetector sd;
        sd.disp_threshold = threshold;
        sd.window_ticks = 90;
        bool triggered = false;
        for (int i = 0; i < 120; ++i) {
            triggered = sd.update(0.05, false);
        }
        check(!triggered, "A: 120 moving ticks never trigger a stall");
        check(sd.consecutive == 0, "A: moving ticks reset the counter");
    }

    // Case B — stationary non-TURN: step displacement 0 for >= window
    // ticks => MUST trigger a stall.
    {
        StallDetector sd;
        sd.disp_threshold = threshold;
        sd.window_ticks = 90;
        bool triggered = false;
        int tick = 0;
        for (; tick < 300; ++tick) {
            triggered = sd.update(0.0, false);
            if (triggered) break;
        }
        check(triggered, "B: stationary non-TURN triggers a stall");
        check(tick + 1 == 90,
              "B: stall triggers exactly at window_ticks=90");
    }

    // Case C — pure TURN: position ~constant but in_turn=true for > 90
    // ticks => NEVER triggers.
    {
        StallDetector sd;
        sd.disp_threshold = threshold;
        sd.window_ticks = 90;
        bool triggered = false;
        for (int i = 0; i < 200; ++i) {
            triggered = sd.update(0.0, true);  // turning in place
        }
        check(!triggered, "C: pure TURN never triggers a stall");
    }

    // Regression: the P0 bug made `disp` ALWAYS 0 (prev was updated before
    // the displacement was computed) — a moving drone accumulated a stall
    // count on EVERY tick.  Case A proves a moving drone resets each tick.
}

// ── 6. vertical chicane alternation ───────────────────────────────
static void testVerticalChicane() {
    std::fprintf(stderr, "test: vertical chicane\n");
    const BlueprintGenerationConfig cfg = makeCfg();
    SceneProfileGenerator gen(cfg);
    const SceneProfile* p = gen.findProfile("chicane");
    check(p != nullptr, "chicane profile exists");
    if (!p) return;

    // Find BOTH orientations across seeds.
    bool saw_h = false, saw_v = false;
    for (uint64_t seed = 1; seed < 200; ++seed) {
        SceneGenerationOutcome out = gen.generate(*p, 0, seed);
        if (!out.success) continue;
        check(out.scene.generation_valid, "generated chicane is valid");
        // The recorded orientation must be present and match metadata.
        const StructureOrientation o = out.scene.structure_orientation;
        check(o == StructureOrientation::HORIZONTAL ||
                  o == StructureOrientation::VERTICAL,
              "chicane records a concrete orientation");
        check(out.metadata.structure_orientation ==
                  structureOrientationName(o),
              "metadata orientation matches the recorded one");
        const int flips =
            SceneProfileGenerator::countChicaneFlips(out.scene, cfg.warehouse, o);
        if (o == StructureOrientation::HORIZONTAL) {
            saw_h = true;
            // sorted by x; sign of (y-cy) alternates
            check(flips >= cfg.min_chicane_alternations,
                  "horizontal chicane has enough alternations");
        } else {
            saw_v = true;
            check(flips >= cfg.min_chicane_alternations,
                  "vertical chicane has enough x-alternations");
        }
        if (saw_h && saw_v) break;
    }
    check(saw_h, "at least one horizontal chicane realized");
    check(saw_v, "at least one vertical chicane realized");

    // A vertical-but-NOT-alternating obstacle set must score 0 flips (the
    // "left and right both exist" check would wrongly pass it).
    {
        BlueprintScene bad;
        // Obstacles all on the +x side, spread along y (vertical layout).
        for (int i = 0; i < 6; ++i) {
            BlueprintObstacle o;
            o.x = cfg.warehouse.free_min_x + 2.0 + (i % 2);  // all right of cx
            o.y = cfg.warehouse.free_min_y + 3.0 + 4.0 * i;
            o.radius = 0.5;
            o.id = i;
            bad.obstacles.push_back(o);
        }
        const int flips = SceneProfileGenerator::countChicaneFlips(
            bad, cfg.warehouse, StructureOrientation::VERTICAL);
        check(flips == 0,
              "vertical non-alternating obstacle set has 0 flips (must "
              "fail sanity)");
    }
}

// ── 7. selected scene consolidation keeps min_selected_scenes ─────
namespace {
/// Build a "rich" summary that covers every hard distribution minimum
/// (5 Hz ticks, correction groups, deflection groups, depth, yaw, path).
TaskDistributionSummary richSummary(double yaw_signed_deg,
                                    double path_len_m, int tag) {
    TaskDistributionSummary s;
    s.initial_yaw_error_signed_deg = yaw_signed_deg;
    s.initial_yaw_error_abs_deg = std::fabs(yaw_signed_deg);
    s.preflight_path_length_m = path_len_m;
    s.path_stretch_ratio = 1.2;
    s.reached_goal = true;
    s.macro_tick_total = 60;
    s.macro_pass_count = 20;
    s.macro_normal_count = 20;
    s.macro_turn_left_count = (tag % 2 == 0) ? 10 : 5;
    s.macro_turn_right_count = (tag % 2 == 0) ? 5 : 10;
    s.local_direct_count = 30;
    s.local_avoidance_count = 30;
    s.macro_correction_angle_hist.configure(
        {-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90});
    s.macro_correction_distance_hist.configure({0.0, 0.2, 0.4, 0.6, 0.8, 1.0});
    s.local_deflection_hist.configure(
        {-90, -60, -30, -10, 10, 30, 60, 90});
    s.local_yaw_rate_hist.configure({-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0});
    s.local_speed_hist.configure({0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0});
    // NORMAL correction left / near / right samples.
    s.macro_correction_angle_hist.add(-50.0);
    s.macro_correction_angle_hist.add(-20.0);
    s.macro_correction_angle_hist.add(0.0);
    s.macro_correction_angle_hist.add(20.0);
    s.macro_correction_angle_hist.add(50.0);
    s.macro_correction_distance_hist.add(0.5);
    s.macro_correction_distance_hist.add(0.5);
    // Deflection strong-right / right / near-direct / left / strong-left.
    s.local_deflection_hist.add(-70.0);
    s.local_deflection_hist.add(-40.0);
    s.local_deflection_hist.add(0.0);
    s.local_deflection_hist.add(40.0);
    s.local_deflection_hist.add(70.0);
    // Depth near/mid/far/free.
    s.depth_near_count = 30;
    s.depth_mid_count = 30;
    s.depth_far_count = 30;
    s.depth_free_count = 30;
    s.depth_samples = 120;
    return s;
}
}  // namespace

static void testConsolidation() {
    std::fprintf(stderr, "test: scene consolidation\n");
    BlueprintGenerationConfig cfg = makeCfg();
    // Relax every hard minimum so 2 scenes' tasks alone satisfy coverage.
    cfg.min_macro_ticks_per_class = 1;
    cfg.min_depth_samples_per_band = 1;
    cfg.min_yaw_samples_per_bin = 1;
    cfg.min_path_samples_per_class = 1;
    cfg.min_grouped_deflection_samples = 1;
    cfg.min_grouped_correction_samples = 1;
    cfg.min_tasks_per_scene = 1;
    cfg.max_tasks_per_scene = 4;
    cfg.min_tasks = 4;
    cfg.min_selected_scenes = 4;

    // 6 scenes, each with 4 tasks that each richly cover the minimums.
    std::vector<BlueprintTask> candidates;
    for (uint64_t sid = 0; sid < 6; ++sid) {
        for (uint64_t tid = 0; tid < 4; ++tid) {
            BlueprintTask t;
            t.scene_id = sid;
            t.task_id = sid * 100 + tid;
            t.summary = richSummary((tid % 2 == 0) ? 30.0 : -30.0,
                                    15.0 + static_cast<double>(tid), 
                                    static_cast<int>(tid));
            candidates.push_back(t);
        }
    }

    DistributionAnalyzer analyzer(cfg);
    std::vector<uint64_t> per_scene;
    const std::vector<BlueprintTask> selected =
        analyzer.select(candidates, per_scene);

    // With min_selected_scenes=4 the consolidation must NOT drop below 4.
    std::set<uint64_t> scenes;
    for (const auto& t : selected) scenes.insert(t.scene_id);
    check(scenes.size() >= 4,
          "consolidation keeps >= min_selected_scenes=4 distinct scenes");

    // With min_selected_scenes=2 the same pool may consolidate to 2.
    cfg.min_selected_scenes = 2;
    DistributionAnalyzer analyzer2(cfg);
    std::vector<uint64_t> per_scene2;
    const std::vector<BlueprintTask> selected2 =
        analyzer2.select(candidates, per_scene2);
    std::set<uint64_t> scenes2;
    for (const auto& t : selected2) scenes2.insert(t.scene_id);
    check(scenes2.size() >= 2,
          "consolidation with min_selected_scenes=2 keeps >= 2 scenes");
    check(scenes2.size() < scenes.size(),
          "looser min_selected_scenes allows more consolidation");
}

// ── 8. NORMAL_CORRECTION-only correction histograms ───────────────
static void testCorrectionHistogram() {
    std::fprintf(stderr, "test: correction histogram (NORMAL only)\n");
    BlueprintGenerationConfig cfg = makeCfg();
    cfg.min_macro_ticks_per_class = 1;
    cfg.min_depth_samples_per_band = 1;
    cfg.min_yaw_samples_per_bin = 1;
    cfg.min_path_samples_per_class = 1;
    cfg.min_grouped_deflection_samples = 1;
    cfg.min_grouped_correction_samples = 4;  // must NOT be silently lowered

    // Task 1: 10 TURN_LEFT + 10 TURN_RIGHT, ZERO NORMAL_CORRECTION, and an
    // EMPTY correction-angle histogram (the controller now only fills it
    // on NORMAL_CORRECTION).  TURN counts are present; correction groups
    // must stay 0.
    {
        DistributionAccumulator acc;
        acc.configure(cfg);
        TaskDistributionSummary s = richSummary(30.0, 15.0, 0);
        s.macro_turn_left_count = 10;
        s.macro_turn_right_count = 10;
        s.macro_normal_count = 0;
        s.macro_correction_angle_hist.clear();  // no NORMAL samples
        acc.addTask(s);
        check(acc.count("macro:turn_left") == 10,
              "TURN_LEFT count recorded");
        check(acc.count("macro:turn_right") == 10,
              "TURN_RIGHT count recorded");
        const Histogram1D* h = acc.histogram("macro_correction_angle");
        check(h != nullptr && h->total() == 0,
              "TURN does NOT populate the correction-angle histogram");
    }

    // Task 2: NORMAL left / near / right samples land in the right bins.
    {
        DistributionAccumulator acc;
        acc.configure(cfg);
        TaskDistributionSummary s = richSummary(-20.0, 15.0, 1);
        // richSummary already adds -50 (right group), -20/0/20 (near), 50
        // (left).  Verify the grouped coverage counts.
        acc.addTask(s);
        const Histogram1D* h = acc.histogram("macro_correction_angle");
        check(h != nullptr && h->total() >= 5,
              "NORMAL correction samples populate the histogram");
        const double right = static_cast<double>(h->at(0) + h->at(1) +
                                                 h->at(2) + h->at(3) + h->at(4));
        const double left = static_cast<double>(h->at(5) + h->at(6) +
                                                h->at(7) + h->at(8) + h->at(9));
        check(right >= 1.0 && left >= 1.0,
              "NORMAL right and left groups both populated");
    }
}

// ── 9. HARD global tick budget ────────────────────────────────────
static void testHardTickBudget() {
    std::fprintf(stderr, "test: hard global tick budget\n");
    BlueprintGenerationConfig cfg = makeCfg();
    cfg.max_total_preflight_tasks = 100;    // not the binding constraint
    cfg.max_total_preflight_ticks = 100;    // the binding constraint
    cfg.max_preflight_ticks_per_task = 90;  // per-task cap (> global)
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

    // THE hard invariant: the effective per-task budget is capped by the
    // remaining global ticks, so the total can NEVER exceed the cap.
    check(r.total_preflight_ticks <= 100,
          "total_preflight_ticks never exceeds max_total_preflight_ticks=100");
    // When the tick budget was actually the binding stop, the reason must
    // be preflight_tick_budget (never a misleading ordinary timeout).
    if (r.total_preflight_ticks >= 100) {
        check(r.budget_exhausted_reason == "preflight_tick_budget",
              "the global tick cap is reported as preflight_tick_budget");
    }
    std::fprintf(stderr,
                 "  tick budget result: total_ticks=%llu attempts=%llu "
                 "reason=%s\n",
                 static_cast<unsigned long long>(r.total_preflight_ticks),
                 static_cast<unsigned long long>(r.preflight_attempt_count),
                 r.budget_exhausted_reason.c_str());
}

// ── 10. use_profile_catalog config semantics ──────────────────────
static void testProfileCatalogConfig() {
    std::fprintf(stderr, "test: profile catalog config\n");
    // use_profile_catalog=false + empty profiles => clear failure (no
    // silent empty generation).
    {
        BlueprintGenerationConfig cfg = makeCfg();
        cfg.use_profile_catalog = false;
        cfg.profiles.clear();
        Params2D p;
        BlueprintGenerationController ctl(p, cfg);
        const BlueprintResult r = ctl.generate();
        check(!r.generation_ok &&
                  r.failure_reason.find("use_profile_catalog=false") !=
                      std::string::npos,
              "use_profile_catalog=false with no profiles fails clearly");
    }
    // use_profile_catalog=false + a user profile => that profile is used.
    {
        BlueprintGenerationConfig cfg = makeCfg();
        cfg.use_profile_catalog = false;
        cfg.profiles.clear();
        SceneProfile sp;
        sp.name = "custom_empty";
        sp.count_min = 0;
        sp.count_max = 0;
        sp.structure = SceneStructure::EMPTY;
        cfg.profiles.push_back(sp);
        cfg.profile_sequence = {"custom_empty"};
        cfg.min_scenes = 1;
        cfg.min_selected_scenes = 1;
        cfg.min_tasks = 1;
        Params2D p;
        BlueprintGenerationController ctl(p, cfg);
        const BlueprintResult r = ctl.generate();
        check(r.scenes_generated >= 1,
              "use_profile_catalog=false with a custom profile still "
              "generates scenes");
    }
}

int main() {
    std::fprintf(stderr, "test_blueprint_geometry: starting\n");
    testRayHelpers();
    testScenes();
    testInitialYaw();
    testPreflightBudget();
    testStallDetector();
    testVerticalChicane();
    testConsolidation();
    testCorrectionHistogram();
    testHardTickBudget();
    testProfileCatalogConfig();
    std::fprintf(stderr, "checks=%d failures=%d\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
