#include "il_dataset/hierarchical_expert/blueprint_generation_controller.hpp"

#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"
#include "il_dataset/hierarchical_expert/preflight_simulator.hpp"
#include "il_dataset/hierarchical_expert/stall_detector.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <deque>
#include <future>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <thread>

namespace il_dataset {
namespace expert {

namespace {

using Clock = std::chrono::steady_clock;
inline double msSince(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

/// Deterministic 64-bit seed mixing (splitmix-inspired).
inline uint64_t mixSeed(uint64_t a, uint64_t b) {
    uint64_t x = a ^ (b + 0x9E3779B97F4A7C15ULL +
                      (a << 6) + (a >> 2));
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// Legacy density class from obstacle count (manifest compat).
inline const char* legacyDensityClass(double count,
                                      const BlueprintGenerationConfig& cfg) {
    if (count <= cfg.density_sparse_max) return "sparse";
    if (count >= cfg.density_dense_min) return "dense";
    return "medium";
}

/// Legacy radius class from max obstacle radius (manifest compat).
inline const char* legacyRadiusClass(double max_radius, bool is_empty,
                                     const BlueprintGenerationConfig& cfg) {
    if (is_empty || max_radius <= 0.0) return "none";
    if (max_radius <= cfg.radius_small_max_m) return "small";
    if (max_radius >= cfg.radius_large_min_m) return "large";
    return "medium";
}

/// Geometry classes that can plausibly produce a macro takeover.  CLEAR
/// candidates are deliberately not used for a turn probe: forcing a yaw on a
/// clear task would manufacture an unrepresentative spin sample.
inline bool turnProbeGeometry(const std::string& geom_type) {
    return geom_type == "LARGE_OCCLUSION" ||
           geom_type == "LONG_DETOUR" ||
           geom_type == "CHICANE" ||
           geom_type == "MULTI_OBSTACLE" ||
           geom_type == "LOCAL_AVOIDANCE" ||
           geom_type == "OFFSET_AVOIDANCE" ||
           geom_type == "NARROW_BUT_PLANNABLE";
}

/// Return the configured target for a count metric.  Explicit YAML targets
/// are preferred; the fallback keeps the probe useful with legacy/default
/// configurations.
inline uint64_t countTarget(const DistributionAnalyzer& analyzer,
                            const std::string& key, uint64_t fallback) {
    const std::string metric = "count:" + key;
    for (const auto& target : analyzer.targets()) {
        if (target.metric == metric) {
            return static_cast<uint64_t>(std::max(1.0, std::ceil(target.target)));
        }
    }
    return fallback;
}

/// Force the initial heading to put the goal outside the configured FOV on a
/// requested side.  This changes only the sampled task heading; the label is
/// still accepted only when the real expert emits the requested TURN.
inline void forceTurnProbeYaw(BlueprintTask& task, int side,
                              double magnitude_deg,
                              double& yaw_error_signed_deg) {
    const double goal_bearing =
        std::atan2(task.goal_y - task.start_y, task.goal_x - task.start_x);
    const double magnitude = std::max(45.0 + 1.0, std::fabs(magnitude_deg));
    yaw_error_signed_deg = side < 0 ? -magnitude : magnitude;
    const double expert_yaw = wrapAngle(
        goal_bearing - deg2rad(yaw_error_signed_deg));
    task.initial_yaw = CoordinateAdapter::expertYawToFlightmare(expert_yaw);
}

/// Merge a per-candidate / per-round QualificationCounters into a total.
inline void accumulateQual(QualificationCounters& dst,
                           const QualificationCounters& src) {
    dst.candidates_checked += src.candidates_checked;
    dst.endpoint_pass += src.endpoint_pass;
    dst.connectivity_pass += src.connectivity_pass;
    dst.straight_clear += src.straight_clear;
    dst.blocked += src.blocked;
    dst.side_qualification_attempt += src.side_qualification_attempt;
    dst.both_sides_feasible += src.both_sides_feasible;
    dst.accepted += src.accepted;
    dst.reject_endpoint += src.reject_endpoint;
    dst.reject_clearance += src.reject_clearance;
    dst.reject_different_component += src.reject_different_component;
    dst.reject_global_route += src.reject_global_route;
    dst.reject_global_astar_budget += src.reject_global_astar_budget;
    dst.reject_left_infeasible += src.reject_left_infeasible;
    dst.reject_right_infeasible += src.reject_right_infeasible;
    dst.reject_both_sides_required += src.reject_both_sides_required;
    dst.reject_side_search_budget += src.reject_side_search_budget;
    dst.reject_geom_mismatch += src.reject_geom_mismatch;
    dst.total_astar_expansions += src.total_astar_expansions;
}

/// One enqueued closed-loop preflight (parallel batch item).  `task` and
/// `scene` are owned COPIES so each worker thread reads/writes only its
/// own item — preflightOne is a pure function of (task, scene, budgets)
/// and touches no shared state besides the caller-owned counters, which
/// the worker returns locally.
struct PendingPreflight {
    BlueprintTask task;
    BlueprintScene scene;      // value copy (obstacles are small)
    uint64_t tick_base = 0;
    double yaw_err = 0.0;
    uint64_t task_tick_budget = 0;
    int probe_side = 0;        // +1 LEFT / -1 RIGHT / 0 ordinary
};

/// Worker-thread result of one preflight.  `ticks_used` replaces the
/// serial code's `total_preflight_ticks` out-param; the main thread merges
/// it into the global budget after joining.
struct PreflightOutcome {
    bool accepted = false;
    TaskDistributionSummary summary;
    bool early_terminated = false;
    bool global_tick_truncated = false;
    std::string reject_reason = "accepted";
    double depth_proxy_ms = 0.0;
    uint64_t ticks_used = 0;
};

// ═══════════════════════════════════════════════════════════════════
//  scene-level parallel pipeline helpers (new architecture)
// ═══════════════════════════════════════════════════════════════════

/// One scene spec of the new pipeline: level (0..scene_levels-1), index
/// within the level (sparse -> dense), radius band and target cylinder
/// count.
struct SceneSpec {
    int level = 0;
    int level_index = 0;
    uint64_t scene_id = 0;
    uint64_t seed = 0;
    int target_count = 0;
    double rmin = 0.15;
    double rmax = 0.5;
    double dmin = 0.0;  // adaptive start-goal distance floor
    double dmax = 0.0;  // adaptive start-goal distance ceiling
};

/// Derive a scene spec from (level, index).  Sparse -> dense within a
/// level; the target cylinder count is scaled by the radius band (bigger
/// cylinders => fewer of them; "相对密集但总体稀疏").
inline SceneSpec makeSceneSpec(const BlueprintGenerationConfig& cfg, int level,
                               int level_index, uint64_t scene_id,
                               uint64_t seed) {
    SceneSpec s;
    s.level = level;
    s.level_index = level_index;
    s.scene_id = scene_id;
    s.seed = seed;
    s.rmin = (level >= 0 && level < static_cast<int>(cfg.level_radius_min_m.size()))
                 ? cfg.level_radius_min_m[level]
                 : 0.15;
    s.rmax = (level >= 0 && level < static_cast<int>(cfg.level_radius_max_m.size()))
                 ? cfg.level_radius_max_m[level]
                 : 1.5;
    // ── 尺度自适应距离 ──────────────────────────────────────────
    // medium(1)/large(2) 层抬升 start-goal 距离下限:大尺度障碍的短路径
    // 连不通(dmin = max(min_task, scale×rmax));small(0)/mixed(3) 保持全
    // 范围(大障碍任务由 A* 连通性自然过滤)。
    s.dmin = cfg.min_task_distance_m;
    s.dmax = cfg.max_task_distance_m;
    if (level == 1 || level == 2) {
        s.dmin = std::max(cfg.min_task_distance_m,
                          cfg.distance_min_radius_scale * s.rmax);
    }
    // ── Occupancy-driven obstacle count (2026-08-26) ───────────────
    // target_count = occupancy_target × free-area / E[π r²], where the
    // target occupancy ramps sparse→dense across the level's scenes.
    // Replaces the old per-level count table (small 5..23 was too empty
    // for tiny cylinders, large 2..11 too crammed for big ones).
    // E[r²] for a log-uniform radius on [rmin,rmax] =
    // (rmax²-rmin²)/(2·ln(rmax/rmin)); placeCylinders caps the actual
    // count by the surface-gap constraint anyway.
    const int n_occ_min = static_cast<int>(cfg.level_occupancy_min.size());
    const int n_occ_max = static_cast<int>(cfg.level_occupancy_max.size());
    const double occ_min =
        (level < n_occ_min) ? cfg.level_occupancy_min[level] : 0.05;
    const double occ_max =
        (level < n_occ_max) ? cfg.level_occupancy_max[level] : 0.10;
    const int per_level =
        (level >= 0 &&
         level < static_cast<int>(cfg.scenes_per_level_list.size()))
            ? std::max(1, cfg.scenes_per_level_list[level])
            : std::max(1, cfg.scenes_per_level);
    const double t = per_level > 1
                         ? static_cast<double>(level_index) / (per_level - 1)
                         : 0.0;
    const double occupancy = occ_min + (occ_max - occ_min) * t;
    const double lr = std::log(
        std::max(1e-6, s.rmax / std::max(1e-6, s.rmin)));
    const double er2 = lr > 1e-9
                           ? (s.rmax * s.rmax - s.rmin * s.rmin) / (2.0 * lr)
                           : 0.5 * (s.rmin * s.rmin + s.rmax * s.rmax);
    const double area = cfg.warehouse.area();
    const double kPi = 3.14159265358979323846;
    s.target_count = std::max(
        2, static_cast<int>(std::lround(occupancy * area / (kPi * er2))));
    return s;
}

/// Place `target_count` cylinders in the warehouse FREE region.  Radius is
/// log-uniform in [rmin, rmax]; every obstacle keeps the pairwise SURFACE
/// gap >= cfg.obstacle_surface_gap_min_m (traversable) and its centre at
/// least (radius + cfg.obstacle_boundary_min_m) from the free-region
/// border.  Stops early (returns the placed count) when a cylinder cannot
/// be placed within the attempt budget — the scene is then sparser than
/// requested instead of violating the constraints.
inline int placeCylinders(const BlueprintGenerationConfig& cfg,
                          const WarehouseGeometry& wh, int target_count,
                          double rmin, double rmax, uint64_t seed,
                          std::vector<BlueprintObstacle>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(std::max(0, target_count)));
    std::mt19937_64 rng(mixSeed(seed, 0xC0BBA0EULL));
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const Vec2d fmin = wh.freeMin(), fmax = wh.freeMax();
    const double gap = std::max(0.0, cfg.obstacle_surface_gap_min_m);
    const double bmin = std::max(0.0, cfg.obstacle_boundary_min_m);
    const double lo = std::log(std::max(1e-3, rmin));
    const double hi = std::log(std::max(1e-3, rmax));
    // centre -> AABB surface distance (0 inside).
    auto rectDist = [](double x, double y, const KnownRect& r) {
        const double dx = std::max({r.min_x - x, 0.0, x - r.max_x});
        const double dy = std::max({r.min_y - y, 0.0, y - r.max_y});
        return std::hypot(dx, dy);
    };
    for (int i = 0; i < target_count; ++i) {
        const double r = std::exp(lo + u01(rng) * (hi - lo));
        bool placed = false;
        for (int att = 0; att < 400 && !placed; ++att) {
            const double x = fmin.x() + u01(rng) * (fmax.x() - fmin.x());
            const double y = fmin.y() + u01(rng) * (fmax.y() - fmin.y());
            // border: centre >= (r + boundary) from the border
            if (x - r < fmin.x() + bmin || x + r > fmax.x() - bmin ||
                y - r < fmin.y() + bmin || y + r > fmax.y() - bmin) {
                continue;
            }
            bool ok = true;
            for (const auto& o : out) {
                if (std::hypot(x - o.x, y - o.y) < r + o.radius + gap) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                for (const auto& kr : cfg.known_rects) {
                    if (rectDist(x, y, kr) < r + gap) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) {
                BlueprintObstacle ob;
                ob.x = x;
                ob.y = y;
                ob.radius = r;
                ob.height_m = cfg.obstacle_height_min_m +
                    u01(rng) * (cfg.obstacle_height_max_m -
                                cfg.obstacle_height_min_m);
                // 1/2 of the obstacles are square BOXES (axis-aligned
                // cuboids).  The half-extent r/sqrt(2) keeps the box inside
                // the same circumscribed circle of radius r used for
                // generation / spacing, so the layout constraints are
                // unchanged; the truth audit, preflight depth and runtime
                // rendering all use the exact AABB.
                if (u01(rng) < 0.5) {
                    const double half = r / std::sqrt(2.0);
                    ob.half_w = half;
                    ob.half_h = half;
                }
                ob.id = i;
                out.push_back(ob);
                placed = true;
            }
        }
        if (!placed) break;  // cannot place more — keep what we have
    }
    return static_cast<int>(out.size());
}

/// Distance from point p to segment a-b (Euclidean).
inline double distToSegment(const Vec2d& p, const Vec2d& a, const Vec2d& b) {
    const Vec2d ab = b - a;
    const double len2 = ab.squaredNorm();
    if (len2 < 1e-12) return (p - a).norm();
    const double t = std::max(0.0, std::min(1.0, (p - a).dot(ab) / len2));
    return (p - (a + t * ab)).norm();
}

/// Blocked-line test used to steer the label balance.  Two strictness
/// levels:
///   * strict_core=false (small / medium scenes): a line whose swept disk
///     passes within `clearance` of an obstacle SURFACE counts as blocked.
///     These are mostly solvable by the 30 Hz local avoidance, so they
///     mainly feed the avoidance labels.
///   * strict_core=true (large / mixed scenes): the line must cut through
///     the obstacle CORE (distance to centre < 0.6*radius).  The local
///     FOV cannot bypass it directly, so the 5 Hz macro corrector takes
///     over → real detour / long_detour segments.  Long detours are only
///     PHYSICALLY producible in large-obstacle scenes.
inline bool lineBlocked(const BlueprintScene& scene, const Vec2d& start,
                        const Vec2d& goal, double clearance,
                        bool strict_core) {
    for (const auto& o : scene.obstacles) {
        const double d = distToSegment(Vec2d(o.x, o.y), start, goal);
        if (strict_core) {
            if (d < o.radius * 0.6) return true;
        } else {
            if (d < o.radius + clearance) return true;
        }
    }
    return false;
}

/// Direct-line (start-goal Euclidean) distance class:
///   0 = short (< task_distance_short_max_m), 1 = medium, 2 = long.
inline int distClass(const BlueprintGenerationConfig& cfg, double dist_m) {
    if (dist_m < cfg.task_distance_short_max_m) return 0;
    if (dist_m < cfg.task_distance_medium_max_m) return 1;
    return 2;
}

struct GreedyAStarResult {
    bool reachable = false;
    double path_len_m = 0.0;
    int expansions = 0;
};

/// Greedy toward-goal A* on the SceneGeometryCache grid: a standard A*
/// whose heuristic is the Euclidean distance to the goal (so the search
/// rushes toward the goal).  Used as a FAST connectivity check for the
/// scene-level pipeline (it does not plan an optimal path, it only decides
/// reachability under `clearance` and returns a rough path length).
/// `max_expansions` bounds the search so a genuinely blocked pair is
/// rejected cheaply.
inline GreedyAStarResult greedyAStar(const SceneGeometryCache& geo,
                                     const Vec2d& start, const Vec2d& goal,
                                     double clearance, int max_expansions) {
    GreedyAStarResult res;
    const int w = geo.w(), h = geo.h();
    const double r = geo.res();
    const Vec2d minb = geo.minBounds();
    auto toGrid = [&](const Vec2d& p) -> std::pair<int, int> {
        return {static_cast<int>(std::lround((p.x() - minb.x()) / r)),
                static_cast<int>(std::lround((p.y() - minb.y()) / r))};
    };
    const auto [sx, sy] = toGrid(start);
    const auto [gx, gy] = toGrid(goal);
    if (!geo.inGrid(sx, sy) || !geo.inGrid(gx, gy)) return res;
    if (!geo.cellFree(gx, gy, clearance)) return res;
    if (!geo.cellFree(sx, sy, clearance)) return res;
    const size_t N = static_cast<size_t>(w) * h;
    auto idx = [&](int ix, int iy) { return static_cast<size_t>(iy) * w + ix; };
    auto hcost = [&](int ix, int iy) {
        return std::hypot((ix - gx) * r, (iy - gy) * r);
    };
    std::vector<double> g(N, std::numeric_limits<double>::infinity());
    std::vector<int> came(N, -1);
    std::vector<char> closed(N, 0);
    using QNode = std::pair<double, size_t>;
    std::priority_queue<QNode, std::vector<QNode>, std::greater<QNode>> open;
    const size_t sid = idx(sx, sy);
    const size_t gid = idx(gx, gy);
    g[sid] = 0.0;
    open.push({hcost(sx, sy), sid});
    static const int dirs[8][2] = {{1, 0},  {-1, 0}, {0, 1},  {0, -1},
                                   {1, 1},  {1, -1}, {-1, 1}, {-1, -1}};
    const double d8 = r * 1.4142135623730951;
    while (!open.empty() && res.expansions < max_expansions) {
        const auto top = open.top();
        open.pop();
        const size_t id = top.second;
        if (closed[id]) continue;
        closed[id] = 1;
        ++res.expansions;
        if (id == gid) {
            double len = 0.0;
            int cur = static_cast<int>(id);
            while (came[cur] != -1) {
                const int px = came[cur] % w, py = came[cur] / w;
                const int cx = cur % w, cy = cur / w;
                const bool diag = (cx != px) && (cy != py);
                len += diag ? d8 : r;
                cur = came[cur];
            }
            res.reachable = true;
            res.path_len_m = len;
            return res;
        }
        const int ix = static_cast<int>(id % w), iy = static_cast<int>(id / w);
        for (const auto& d : dirs) {
            const int nx = ix + d[0], ny = iy + d[1];
            if (!geo.inGrid(nx, ny)) continue;
            const size_t nid = idx(nx, ny);
            if (closed[nid]) continue;
            if (!geo.cellFree(nx, ny, clearance)) continue;
            const bool diag = (d[0] != 0) && (d[1] != 0);
            const double ng = g[id] + (diag ? d8 : r);
            if (ng < g[nid]) {
                g[nid] = ng;
                came[nid] = static_cast<int>(id);
                open.push({ng + hcost(nx, ny), nid});
            }
        }
    }
    return res;
}

}  // namespace

BlueprintGenerationController::BlueprintGenerationController(
    const Params2D& params, const BlueprintGenerationConfig& cfg)
    : p_(params),
      cfg_(cfg),
      profile_gen_(cfg),
      analyzer_(cfg),
      task_gen_(cfg) {}

bool BlueprintGenerationController::cheapFilterPass(
    const BlueprintScene& scene, const SceneGeometryCache& geo,
    const BlueprintTask& task) const {
    (void)scene;  // the filter uses the cached geometry (geo) only
    const double r = cfg_.vehicle_radius_m;
    const Vec2d start(task.start_x, task.start_y);
    const Vec2d goal(task.goal_x, task.goal_y);

    // 1. Bounds: start/goal disks stay inside the warehouse FREE region.
    if (!cfg_.warehouse.inFree(start.x(), start.y(), r) ||
        !cfg_.warehouse.inFree(goal.x(), goal.y(), r)) {
        return false;
    }
    // 2. Clearance: both endpoints lie on a free cell of the MAIN
    //    component with the configured centre->surface clearance.
    if (!geo.pointFreeMain(start, cfg_.free_cell_surface_clearance_m) ||
        !geo.pointFreeMain(goal, cfg_.free_cell_surface_clearance_m)) {
        return false;
    }
    // 3. Distance band + zero-length.
    const double d = (goal - start).norm();
    if (d < cfg_.min_task_distance_m - 1e-9 || d > cfg_.max_task_distance_m + 1e-9) {
        return false;
    }
    // 4. The straight segment swept by the drone disk stays inside the
    //    free region.
    if (!segmentDiskInsideBounds(start.x(), start.y(), goal.x(), goal.y(), r,
                                 cfg_.warehouse.freeMin(),
                                 cfg_.warehouse.freeMax())) {
        return false;
    }
    return true;
}

bool BlueprintGenerationController::preflightOne(
    BlueprintTask& task, const BlueprintScene& scene, uint64_t tick_base,
    TaskDistributionSummary& summary, double yaw_error_signed_deg,
    uint64_t task_tick_budget, uint64_t& total_preflight_ticks,
    bool& early_terminated, bool& global_tick_truncated,
    std::string& reject_reason, double& depth_proxy_ms,
    SegmentLabeler* segmenter) const {
    PreflightSimulator sim(p_);
    Scene2D s2d;
    s2d.min_bounds = cfg_.warehouse.freeMin();
    s2d.max_bounds = cfg_.warehouse.freeMax();
    s2d.valid = true;
    std::vector<Vec2d> known_min, known_max;
    known_min.reserve(cfg_.known_rects.size() + scene.obstacles.size());
    known_max.reserve(cfg_.known_rects.size() + scene.obstacles.size());
    for (const auto& o : scene.obstacles) {
        if (o.isBox()) {
            // Box obstacles enter the synthetic depth as AABB rects (the
            // runtime renders Transparen_Cube AABBs), NOT as circles — the
            // ray-cast uses rayRectHit for known rects.
            known_min.emplace_back(o.x - o.half_w, o.y - o.half_h);
            known_max.emplace_back(o.x + o.half_w, o.y + o.half_h);
            continue;
        }
        Obstacle2D ob;
        ob.center = Vec2d(o.x, o.y);
        ob.radius = o.radius;
        ob.id = o.id;
        s2d.obstacles.push_back(ob);
    }
    early_terminated = false;
    global_tick_truncated = false;
    reject_reason = "accepted";
    const Vec2d wall_envelope_min = cfg_.warehouse.envelopeMin();
    const Vec2d wall_envelope_max = cfg_.warehouse.envelopeMax();
    const Vec2d* wall_min = nullptr;
    const Vec2d* wall_max = nullptr;
    if (cfg_.walls_visible_in_observation) {
        wall_min = &wall_envelope_min;
        wall_max = &wall_envelope_max;
    }
    // Fixed known-obstacle AABBs (real scene point-cloud structures) must
    // enter the synthetic patch too, otherwise the preflight sees a MUCH
    // emptier world than the runtime depth and wrongly certifies tasks the
    // real expert will fail.
    for (const auto& kr : cfg_.known_rects) {
        known_min.emplace_back(kr.min_x, kr.min_y);
        known_max.emplace_back(kr.max_x, kr.max_y);
    }
    sim.configure(s2d, s2d.min_bounds, s2d.max_bounds, wall_min, wall_max,
                  known_min, known_max);
    // Coarse-step quick preflight: a larger dynamics step (dt_scale/30 s)
    // makes the vehicle travel further per expert decision, so a FULL
    // start->goal trajectory needs fewer ticks (same expert decision
    // stream, same 5 Hz macro cadence per 6 ticks).  Default scale 1.0 =
    // exact real-time 30 Hz behaviour.
    sim.setStepDt((1.0 / 30.0) * std::max(1.0, cfg_.quick_preflight_dt_scale));

    TruthCylinderAudit truth;
    truth.configure(scene.obstacles, p_.drone_radius, s2d.min_bounds,
                    s2d.max_bounds);

    sim.resetTask(Vec2d(task.start_x, task.start_y),
                  Vec2d(task.goal_x, task.goal_y), task.initial_yaw,
                  tick_base, task.flight_height_m);

    // ── summary init ──────────────────────────────────────────────
    summary.task_id = task.task_id;
    summary.scene_id = task.scene_id;
    summary.scene_profile = scene.profile;
    summary.task_geom_type = task.geom_type;
    summary.straight_distance_m = task.audit.straight_distance_m;
    summary.initial_yaw_error_signed_deg = yaw_error_signed_deg;
    summary.initial_yaw_error_abs_deg = std::fabs(yaw_error_signed_deg);
    summary.macro_correction_angle_hist.configure(cfg_.correction_angle_edges_deg);
    summary.macro_correction_distance_hist.configure(cfg_.correction_distance_edges);
    summary.local_deflection_hist.configure(cfg_.deflection_edges_deg);
    summary.local_yaw_rate_hist.configure(cfg_.yaw_rate_edges);
    summary.local_speed_hist.configure(cfg_.speed_edges);

    DepthProxyEvaluator depth_proxy(cfg_, p_);
    // P2: cache the scene-static circle geometry once per preflight so the
    // stride samples never rebuild centres/radii (150+ samples per task).
    depth_proxy.configure(scene.obstacles);
    const bool depth_walls = cfg_.walls_visible_in_observation;

    // ── closed-loop preflight ──────────────────────────────────────
    // The EFFECTIVE per-task budget is passed in (already capped by the
    // remaining GLOBAL tick budget — see generate()).  `per_task_budget`
    // is the raw per-task cap, kept to detect global-cap truncation.
    const uint64_t budget = std::max<uint64_t>(1, task_tick_budget);
    const uint64_t per_task_budget =
        std::max<uint64_t>(1, cfg_.max_preflight_ticks_per_task);
    const double dt = (1.0 / std::max(1e-6, cfg_.control_rate_hz)) *
                      std::max(1.0, cfg_.quick_preflight_dt_scale);
    const uint64_t stride = std::max<uint64_t>(1, cfg_.depth_proxy_sample_stride_ticks);
    uint64_t ticks = 0;
    bool reached = false, collision = false, out_of_bounds = false;
    bool truth_brake_triggered = false;
    bool macro_label_invalid = false, qual_exceeded = false;
    double min_clearance = std::numeric_limits<double>::infinity();
    double clear_sum = 0.0, clear_count = 0.0;
    double obs_clear_sum = 0.0, obs_clear_count = 0.0;
    uint64_t max_consecutive_active = 0, cur_consecutive_active = 0;
    uint64_t turn_update_frames = 0, normal_update_frames = 0;
    bool saw_turn_left = false, saw_turn_right = false;
    // P0 stall-fix regression: `prev` (previous-position latch) and
    // `path_len` (accumulated preflight path length) must be declared
    // BEFORE the tick loop.  `prev` starts at the task start so the FIRST
    // step displacement is start->firstState (the stall detector and the
    // swept collision audit both rely on it).
    double path_len = 0.0;
    Vec2d prev = HorizontalProjection::position(sim.state().position);

    // Early-termination bookkeeping (blueprint-only; never changes expert
    // labels): no-progress over a rolling window (OFF by default; when
    // enabled uses a COMBINED criterion — see the loop) and a stall
    // detector.
    const Vec2d orig_goal(task.goal_x, task.goal_y);
    std::deque<double> goal_dist_window;
    std::deque<double> motion_window;
    // Raw window (0 = no-progress disabled).  A pure "must approach the
    // goal" rule would kill legitimate long detours, so we only use it
    // when the user explicitly enables it and combine it with the window
    // motion floor.
    const int no_prog_win = cfg_.no_progress_window_ticks;
    bool no_progress_triggered = false;
    bool stall_triggered = false;
    // PURE stall detector (shared with the regression tests).  Threshold:
    // speed [m/s] * dt [s/tick] = m/tick — derived from the explicit
    // control rate (no magic 30.0).  The WINDOW is rescaled by the dt
    // scale so the PHYSICAL stall duration stays the same under a coarse
    // quick-preflight step (90 ticks @ 1/30 s == 15 ticks @ 1/5 s).
    StallDetector stall;
    stall.disp_threshold = cfg_.stall_speed_mps * dt;
    stall.window_ticks = std::max(
        1, static_cast<int>(
               static_cast<double>(std::max(1, cfg_.stall_window_ticks)) /
               std::max(1.0, cfg_.quick_preflight_dt_scale)));

    for (uint64_t t = 0; t < budget; ++t) {
        // Depth proxy at stride BEFORE the step (matches runtime: the
        // depth used by the expert is the one at the CURRENT pose).
        if (t % stride == 0) {
            const auto t_depth = Clock::now();
            const auto sample = depth_proxy.castAt(
                HorizontalProjection::position(sim.state().position),
                sim.state().yaw, scene.obstacles, depth_walls,
                wall_envelope_min, wall_envelope_max);
            depth_proxy_ms += msSince(t_depth);
            depth_proxy.accumulate(sample, summary);
        }

        const auto res = sim.step(tick_base + t, false);
        ticks = t + 1;
        const ExpertStepOutput& out = res.output;

        // ── behaviour SEGMENT labeling (optional; the new preflight
        //    planner feeds this with the per-tick expert output so the
        //    trajectory is split into straight / avoidance / detour
        //    segments). ─────────────────────────────────────────────
        if (segmenter) segmenter->onTick(out);

        // ── P0 FIX: capture the step displacement BEFORE updating `prev`.
        //    The old code updated `prev` first, so the stall detector's
        //    `(position - prev)` was ALWAYS 0 and every non-TURN tick
        //    accumulated a stall count — systematically killing normal
        //    long / detour / chicane tasks after ~stall_window_ticks. ──
        const double step_disp =
            (HorizontalProjection::position(res.state.position) - prev)
                .norm();  // m per tick
        const double prev_x = prev.x(), prev_y = prev.y();

        // Continuous swept truth clearance / collision over prev->new.
        const double seg_clr = truth.segmentMinClearance(
            prev_x, prev_y, res.state.position.x(), res.state.position.y());
        min_clearance = std::min(min_clearance, seg_clr);
        if (std::isfinite(seg_clr)) {
            clear_sum += seg_clr;
            clear_count += 1.0;
        }
        if (res.truth_collision ||
            truth.segmentCollision(prev_x, prev_y,
                                   res.state.position.x(),
                                   res.state.position.y())) {
            collision = true;
            break;
        }
        // Apply the same obstacle edge-clearance floor as the runtime truth
        // judge.  Previously a preflight with 0.375 m centre-to-surface
        // clearance was accepted, then the identical collected task failed
        // the 0.4 m runtime floor.
        const double truth_brake_floor =
            p_.drone_radius + std::max(0.0, p_.lp_brake_stop_margin_m);
        double truth_brake_risk = 0.0;
        bool runtime_truth_would_trigger = false;
        truth.brakeRisk(
            res.state.position.x(), res.state.position.y(),
            res.state.velocity_world.x(), res.state.velocity_world.y(),
            p_.lp_eff_accel_mps2, p_.lp_brake_stop_margin_m,
            truth_brake_risk, runtime_truth_would_trigger);
        if ((std::isfinite(seg_clr) && seg_clr < truth_brake_floor) ||
            runtime_truth_would_trigger) {
            truth_brake_triggered = true;
            break;
        }
        if (res.out_of_bounds ||
            truth.segmentCrossesBounds(prev_x, prev_y,
                                       res.state.position.x(),
                                       res.state.position.y(),
                                       p_.drone_radius)) {
            out_of_bounds = true;
            break;
        }
        path_len += step_disp;
        prev = HorizontalProjection::position(res.state.position);

        // ── 30 Hz behaviour stats ──────────────────────────────────
        if (out.hierarchical_mode == "direct") ++summary.local_direct_count;
        if (out.avoidance_active) ++summary.local_avoidance_count;
        const double speed =
            std::hypot(out.target_velocity_flu_x, out.target_velocity_flu_y);
        summary.local_speed_hist.add(speed);
        summary.local_yaw_rate_hist.add(out.target_yaw_rate);
        if (speed >= cfg_.min_deflection_speed_mps) {
            const double gx = out.goal_direction_flu_x, gy = out.goal_direction_flu_y;
            const double gl = std::hypot(gx, gy);
            if (gl > 1e-6) {
                const double vx = out.target_velocity_flu_x,
                             vy = out.target_velocity_flu_y;
                const double ang = rad2deg(wrapAngle(std::atan2(
                    gx * vy - gy * vx, gx * vx + gy * vy)));
                summary.local_deflection_hist.add(ang);
            }
        }
        if (std::isfinite(out.min_observed_clearance_m)) {
            obs_clear_sum += out.min_observed_clearance_m;
            obs_clear_count += 1.0;
        }

        // ── 5 Hz tick-level macro stats ────────────────────────────
        if (out.macro_update_mask) {
            ++summary.macro_tick_total;
            const std::string& ct = out.macro_correction_type;
            if (ct == "PASS_THROUGH") {
                ++summary.macro_pass_count;
            } else if (ct == "NORMAL_CORRECTION") {
                ++summary.macro_normal_count;
                ++normal_update_frames;
            } else if (ct == "TURN_LEFT") {
                ++summary.macro_turn_left_count;
                ++turn_update_frames;
                saw_turn_left = true;
            } else if (ct == "TURN_RIGHT") {
                ++summary.macro_turn_right_count;
                ++turn_update_frames;
                saw_turn_right = true;
            }
            if (out.macro_label_valid != 1) {
                macro_label_invalid = true;
                break;
            }
            // ── P1 FIX: the correction-angle / correction-distance
            //    histograms must ONLY accumulate NORMAL_CORRECTION.
            //    The training definition of "correction angle" is the
            //    LOCAL effective-target direction relative to the ORIGINAL
            //    navigation-goal direction while the 5 Hz corrector is
            //    doing a NORMAL correction.  TURN_LEFT / TURN_RIGHT have
            //    their own counts (macro_turn_left/right_count) and must
            //    NOT pollute the NORMAL-correction grouped coverage. ──
            if (ct == "NORMAL_CORRECTION" && out.target_correction_active) {
                const double gx = out.navigation_goal_direction_flu_x,
                             gy = out.navigation_goal_direction_flu_y;
                const double ex = out.goal_direction_flu_x,
                             ey = out.goal_direction_flu_y;
                const double gl = std::hypot(gx, gy), el = std::hypot(ex, ey);
                if (gl > 1e-6 && el > 1e-6) {
                    const double ang = rad2deg(wrapAngle(std::atan2(
                        gx * ey - gy * ex, gx * ex + gy * ey)));
                    summary.macro_correction_angle_hist.add(ang);
                }
                const double dn = out.macro_distance_norm;
                if (std::isfinite(dn)) {
                    summary.macro_correction_distance_hist.add(clamp(dn, 0.0, 1.0));
                }
            }
        }
        if (out.target_correction_active) {
            ++cur_consecutive_active;
            max_consecutive_active =
                std::max(max_consecutive_active, cur_consecutive_active);
        } else {
            cur_consecutive_active = 0;
        }

        if (res.goal_reached) {
            reached = true;
            break;
        }

        // ── Early termination (budget-only, no label impact) ───────
        // No-progress (ONLY when explicitly enabled, no_prog_win > 0):
        // over a rolling window BOTH conditions must hold —
        //   (a) the distance to the ORIGINAL goal shrunk by less than
        //       `no_progress_min_progress_m`, AND
        //   (b) the drone actually travelled less than
        //       `no_progress_window_min_motion_m` over the window.
        // The motion floor protects legitimate long detours that move
        // laterally or briefly away from the goal.  When disabled, only
        // the stall detector + global tick budget guard the preflight.
        if (no_prog_win > 0) {
            const double d_to_goal =
                (orig_goal - HorizontalProjection::position(res.state.position))
                    .norm();
            goal_dist_window.push_back(d_to_goal);
            motion_window.push_back(step_disp);
            if (static_cast<int>(goal_dist_window.size()) > no_prog_win + 1) {
                goal_dist_window.pop_front();
                motion_window.pop_front();
            }
            if (static_cast<int>(goal_dist_window.size()) == no_prog_win + 1) {
                const double shrink =
                    goal_dist_window.front() - goal_dist_window.back();
                double motion = 0.0;
                for (const double d : motion_window) motion += d;
                if (shrink < cfg_.no_progress_min_progress_m &&
                    motion < cfg_.no_progress_window_min_motion_m) {
                    no_progress_triggered = true;
                    early_terminated = true;
                    break;
                }
            }
        }
        // Stall: the drone is physically stationary (per-tick displacement
        // below `stall_speed_mps * dt`) for `stall_window_ticks` while NOT
        // in a legitimate TURN update.  `step_disp` is the displacement of
        // THIS tick (captured before `prev` was updated above — the P0
        // fix); a pure TURN keeps position ~constant but is exempt via
        // `in_turn`.  A moving drone (step_disp >= threshold) resets the
        // counter every tick, so normal tasks can never accumulate a stall.
        {
            // Legitimate reorientation is NOT a stall.  The new planner
            // does yaw-first pure rotation at 30 Hz whenever the target is
            // outside the usable FOV band (hierarchical_mode
            // "turn_to_target" / planner_status "TURNING"), WITHOUT a 5 Hz
            // macro TURN directive — so the old macro-only exemption would
            // count every such rotation as a stall and reject the task
            // after stall_window_ticks.  The FSM spin guard still bounds
            // pathological rotation, and the tick budget catches endless
            // turning, so exempting local turns is safe.
            const bool in_turn =
                out.macro_correction_type == "TURN_LEFT" ||
                out.macro_correction_type == "TURN_RIGHT" ||
                out.hierarchical_mode == "turn_to_target" ||
                out.planner_status == "TURNING";
            if (stall.update(step_disp, in_turn)) {
                stall_triggered = true;
                early_terminated = true;
                break;
            }
        }
    }
    if (early_terminated) {
        // The episode was cut by the detector, not by the tick budget.
        qual_exceeded = false;
    }
    if (ticks >= budget && !reached && !early_terminated) qual_exceeded = true;
    // Global-cap truncation: the task consumed its ENTIRE effective budget
    // and that budget was SMALLER than the raw per-task cap => it was cut
    // by the remaining GLOBAL tick budget, not by a normal task timeout.
    // This only drives diagnostics (budget_exhausted_reason), never the
    // training labels.
    if (!reached && !early_terminated && ticks >= budget &&
        budget < per_task_budget) {
        global_tick_truncated = true;
    }

    // ── summary quality fields ─────────────────────────────────────
    summary.preflight_ticks = ticks;
    summary.preflight_duration_s = static_cast<double>(ticks) / cfg_.control_rate_hz;
    summary.preflight_path_length_m = path_len;
    summary.path_stretch_ratio =
        summary.straight_distance_m > 1e-6
            ? path_len / summary.straight_distance_m
            : 1.0;
    summary.reached_goal = reached;
    summary.collision = collision;
    summary.out_of_bounds = out_of_bounds;
    summary.minimum_clearance_m = min_clearance;
    summary.min_observed_clearance_m =
        obs_clear_count > 0.0 ? obs_clear_sum / obs_clear_count
                              : std::numeric_limits<double>::infinity();
    summary.mean_observed_clearance_m =
        clear_count > 0.0 ? clear_sum / clear_count
                          : std::numeric_limits<double>::infinity();

    // ── audit (backward-compatible BlueprintTaskAudit) ─────────────
    task.audit.preflight_ticks = ticks;
    task.audit.min_truth_clearance_m = min_clearance;
    task.audit.reached_goal = reached;
    task.audit.truth_collision = collision;
    task.audit.truth_brake_triggered = truth_brake_triggered;
    task.audit.out_of_bounds = out_of_bounds;
    task.audit.macro_label_ok = !macro_label_invalid;
    task.audit.qualification_exceeded = qual_exceeded;
    task.audit.path_length_m = path_len;
    task.audit.path_stretch_ratio = summary.path_stretch_ratio;
    task.audit.preflight_duration_s = summary.preflight_duration_s;
    task.audit.accepted =
        reached && !collision && !truth_brake_triggered && !out_of_bounds &&
        !macro_label_invalid && !qual_exceeded;

    total_preflight_ticks += ticks;
    if (!task.audit.accepted) {
        task.behavior_class = "rejected";
        task.side_class = "none";
        // Rejection category for the per-round breakdown (also reflected
        // in the detailed preflight_status string).
        if (no_progress_triggered) {
            reject_reason = "no_progress";
        } else if (stall_triggered) {
            reject_reason = "stall";
        } else if (collision) {
            reject_reason = "collision";
        } else if (truth_brake_triggered) {
            reject_reason = "truth_brake_would_trigger";
        } else if (out_of_bounds) {
            reject_reason = "out_of_bounds";
        } else if (qual_exceeded) {
            reject_reason = "timeout";
        } else if (!reached) {
            reject_reason = "goal_not_reached";
        } else {
            reject_reason = "macro_label";
        }
        task.audit.preflight_status =
            early_terminated
                ? (no_progress_triggered
                       ? "preflight_rejected:early_termination:no_progress"
                       : "preflight_rejected:early_termination:stall")
                : (collision
                       ? "preflight_rejected:truth_collision"
                       : (truth_brake_triggered
                              ? "preflight_rejected:truth_brake_would_trigger"
                              : (out_of_bounds
                                     ? "preflight_rejected:out_of_bounds"
                                     : (qual_exceeded
                                            ? (global_tick_truncated
                                                   ? "preflight_rejected:global_tick_budget"
                                                   : "preflight_rejected:qualification_budget")
                                            : (!reached
                                                   ? "preflight_rejected:goal_not_reached"
                                                   : "preflight_rejected:macro_label_invalid")))));
        if (segmenter) segmenter->finish();
        return false;
    }
    task.audit.preflight_status = "preflight_accepted";

    // ── behaviour class (kept from the previous classifier) ────────
    const uint64_t turn_l = summary.macro_turn_left_count;
    const uint64_t turn_r = summary.macro_turn_right_count;
    const uint64_t normal = summary.macro_normal_count;
    if (saw_turn_left && normal > 0) {
        task.behavior_class = "turn_normal";
    } else if (turn_l > 0 && turn_r == 0) {
        task.behavior_class = "turn_left";
    } else if (turn_r > 0 && turn_l == 0) {
        task.behavior_class = "turn_right";
    } else if (turn_l > 0 && turn_r > 0) {
        task.behavior_class = "turn_both";
    } else if (max_consecutive_active >=
               static_cast<uint64_t>(std::max(1, cfg_.min_macro_ticks_per_class))) {
        task.behavior_class = "long_takeover";
    } else if (normal > 0) {
        task.behavior_class = "normal";
    } else if (summary.local_avoidance_count > 0) {
        task.behavior_class = "local_avoidance";
    } else {
        task.behavior_class = "clear";
    }
    summary.behavior_class = task.behavior_class;

    task.saw_turn_left = saw_turn_left;
    task.saw_turn_right = saw_turn_right;
    task.saw_normal_correction = normal > 0;
    task.turn_update_count = turn_update_frames;
    task.normal_update_count = normal_update_frames;
    if (saw_turn_left && saw_turn_right) {
        task.side_class = "both";
    } else if (saw_turn_left) {
        task.side_class = "left";
    } else if (saw_turn_right) {
        task.side_class = "right";
    } else {
        task.side_class = "none";
    }

    // ── distance class from the ACTUAL preflight path length ───────
    if (path_len <= cfg_.path_short_max_m) {
        task.distance_class = "short";
    } else if (path_len >= cfg_.path_long_min_m) {
        task.distance_class = "long";
    } else {
        task.distance_class = "medium";
    }
    if (segmenter) segmenter->finish();
    return true;
}

void BlueprintGenerationController::updateLegacyStrata(
    BlueprintResult& result, const BlueprintScene& scene) const {
    if (scene.is_empty) return;
    const std::string dc =
        legacyDensityClass(static_cast<double>(scene.actual_obstacle_count), cfg_);
    const std::string rc = legacyRadiusClass(scene.actual_max_radius_m,
                                             scene.is_empty, cfg_);
    int density_idx = 0, radius_idx = 0;
    if (dc == "medium") density_idx = 1;
    else if (dc == "dense") density_idx = 2;
    if (rc == "medium") radius_idx = 1;
    else if (rc == "large") radius_idx = 2;
    const int sid = radius_idx * 3 + density_idx;  // matches legacy schedule
    if (sid >= 0 && sid < 9) {
        result.strata_covered_flags[static_cast<size_t>(sid)] = 1;
    }
}

void BlueprintGenerationController::fillCategoryCounts(
    BlueprintResult& result) const {
    result.category_counts.clear();
    std::map<std::string, uint64_t> density, radius, distance;
    for (const auto& t : result.tasks) {
        result.category_counts["behavior:" + t.behavior_class] += 1;
        density[t.density_class] += 1;
        radius[t.radius_class] += 1;
        distance[t.distance_class] += 1;
    }
    for (const char* lvl : {"sparse", "medium", "dense"}) {
        result.category_counts["density:" + std::string(lvl)] = density[lvl];
    }
    for (const char* lvl : {"small", "medium", "large"}) {
        result.category_counts["radius:" + std::string(lvl)] = radius[lvl];
    }
    for (const char* lvl : {"short", "medium", "long"}) {
        result.category_counts["distance:" + std::string(lvl)] = distance[lvl];
    }
    result.category_counts["turn_left"] = 0;
    result.category_counts["turn_right"] = 0;
    for (const auto& t : result.tasks) {
        if (t.saw_turn_left) ++result.category_counts["turn_left"];
        if (t.saw_turn_right) ++result.category_counts["turn_right"];
    }
}

void BlueprintGenerationController::fillDistributionReport(
    BlueprintResult& result, const DistributionAccumulator& acc) const {
    result.distribution_counts.clear();
    for (const auto& kv : acc.counts) {
        result.distribution_counts[kv.first] = kv.second;
    }
    result.distribution_histograms.clear();
    for (const auto& kv : acc.histograms) {
        result.distribution_histograms[kv.first] = kv.second.counts;
    }
}

BlueprintResult BlueprintGenerationController::generate() {
    BlueprintResult result;
    GenerationTiming timing;
    const auto t_total = Clock::now();

    result.base_seed = cfg_.base_seed;
    result.requested_scenes = std::max(1, cfg_.min_scenes);
    result.requested_tasks_per_scene = std::max(1, cfg_.max_tasks_per_scene);
    result.strata_required = 9;
    result.strata_covered_flags.assign(9, 0);

    analyzer_.reset();
    std::vector<BlueprintTask> global_pool;
    std::map<uint64_t, BlueprintScene> scenes;

    // ── scene-level parallel pipeline (new architecture) ─────────
    // The main thread pre-generates scene_levels x scenes_per_level scene
    // specs (each level's cylinder radius band, sparse -> dense), then
    // scene_parallel_threads workers run whole scenes concurrently; the
    // main thread merges and balances afterwards.
    if (cfg_.scene_level_parallel) {
        return generateSceneParallel();
    }

    // Parallel task preflight: cfg_.parallel_tasks worker threads for the
    // expensive closed-loop simulations (0 or 1 = serial).  preflightOne is
    // a pure function of (task, scene, budgets), so candidates enqueued in
    // a batch run concurrently and their outcomes are merged back in
    // SUBMISSION order (the analyzer / pool / counters see the same
    // sequence a serial run would, per batch).
    const int parallel_workers = std::max(1, cfg_.parallel_tasks);
    // use_profile_catalog=false with NO user-provided profiles is a config
    // error: there would be nothing to generate and the run would silently
    // produce zero scenes.  Fail with a clear reason instead.
    if (!cfg_.use_profile_catalog && cfg_.profiles.empty()) {
        result.failure_reason =
            "use_profile_catalog=false requires explicit profiles "
            "(profile catalog disabled with an empty profiles list)";
        return result;
    }

    const int max_rounds = std::max(1, cfg_.max_generation_rounds);
    const int max_scene_candidates = std::max(1, cfg_.max_scene_candidates);
    const uint64_t max_preflights =
        std::max<uint64_t>(1, cfg_.max_total_preflight_tasks);
    const uint64_t max_preflight_ticks =
        std::max<uint64_t>(1, cfg_.max_total_preflight_ticks);
    const int max_tasks_per_scene = std::max(1, cfg_.max_task_candidates_per_scene);

    uint64_t global_task_id = 0;
    uint64_t scene_counter = 0;
    uint64_t pool_target = 0;
    // Budget counters: the PREFLIGHT budgets are enforced on ATTEMPTS and
    // TICKS (success + failure), never on the accepted pool size alone.
    uint64_t total_preflight_attempts = 0;
    uint64_t total_preflight_ticks = 0;
    uint64_t full_preflight_attempted = 0;  // not early-terminated
    uint64_t full_preflight_success = 0;    // accepted AND ran to completion
    // max_scene_candidates limits actual SCENE GENERATION ATTEMPTS (each
    // scene-loop iteration is one attempt, success OR failure), never just
    // the number of successfully stored scenes.
    uint64_t scene_generation_attempts = 0;
    // Privileged task-qualification aggregates + preflight-after-qual
    // efficiency counters.
    QualificationCounters qual_total;
    // REAL-TIME generation-wide qualification expansion counter: updated
    // immediately after EVERY qualify() call (not at round end), and used
    // by budgetExceeded() so a round can never overshoot by one task.
    uint64_t total_qualification_expansions = 0;
    uint64_t full_preflight_after_qual = 0;
    uint64_t full_preflight_success_after_qual = 0;
    BudgetExhaustion budget_exhausted = BudgetExhaustion::NONE;
    std::vector<RoundStats> round_logs;

    auto budgetExceeded = [&]() {
        if (total_preflight_attempts >= max_preflights) {
            budget_exhausted = BudgetExhaustion::PREFLIGHT_ATTEMPT_BUDGET;
            return true;
        }
        if (total_preflight_ticks >= max_preflight_ticks) {
            budget_exhausted = BudgetExhaustion::PREFLIGHT_TICK_BUDGET;
            return true;
        }
        // Generation-wide privileged-qualification A* expansion hard bound:
        // the REAL-TIME counter is checked before every task, so a task is
        // never started (nor a round continued) after the budget is spent.
        if (cfg_.qualification.enabled &&
            total_qualification_expansions >=
                cfg_.qualification.max_total_qualification_expansions) {
            budget_exhausted = BudgetExhaustion::QUALIFICATION_EXPANSION_BUDGET;
            return true;
        }
        return false;
    };

    for (int round = 1; round <= max_rounds; ++round) {
        if (budgetExceeded()) break;
        if (scenes.size() >= static_cast<size_t>(max_scene_candidates)) {
            budget_exhausted = BudgetExhaustion::SCENE_BUDGET;
            break;
        }
        const auto t_round = Clock::now();
        RoundStats rs;
        rs.round = static_cast<uint64_t>(round);
        QualificationCounters qc_round;  // per-round qualification counts
        // Per-round timing snapshots (for the round log breakdown of the
        // serial qualification gate vs the parallel preflight).
        const double qual_ms_start = timing.task_qualification_ms;
        const double preflight_ms_start = timing.preflight_total_ms;

        const int scene_budget = max_scene_candidates - static_cast<int>(scenes.size());
        const int rounds_left = max_rounds - round + 1;
        const int round_scenes =
            std::max(1, static_cast<int>(std::ceil(
                            static_cast<double>(scene_budget) /
                            static_cast<double>(rounds_left))));

        std::mt19937_64 round_rng(mixSeed(cfg_.base_seed, 0x0000F00DULL + round));

        for (int s = 0; s < round_scenes; ++s) {
            if (budgetExceeded()) break;
            if (scene_generation_attempts >=
                static_cast<uint64_t>(max_scene_candidates)) {
                budget_exhausted = BudgetExhaustion::SCENE_BUDGET;
                break;
            }

            // ── profile pick (deficit-driven or explicit sequence) ──
            const SceneProfile* prof = nullptr;
            const bool exploration_round =
                cfg_.pool_first_exploration &&
                (round <= std::max(0, cfg_.exploration_rounds) ||
                 (cfg_.exploration_min_pool_tasks > 0 &&
                  global_pool.size() < cfg_.exploration_min_pool_tasks));
            if (!cfg_.profile_sequence.empty()) {
                const size_t idx =
                    static_cast<size_t>(scene_counter) % cfg_.profile_sequence.size();
                prof = profile_gen_.findProfile(cfg_.profile_sequence[idx]);
                if (!prof) break;  // unknown profile name: stop (config error)
            } else {
                prof = profile_gen_.pickProfile(
                    round_rng, exploration_round
                        ? std::map<std::string, double>()
                        : analyzer_.profileTagWeights());
            }
            if (!prof) break;

            // ── scene realization (one ATTEMPT, success or failure) ──
            ++scene_generation_attempts;
            const uint64_t scene_id = scene_counter++;
            const uint64_t scene_seed = mixSeed(cfg_.base_seed, scene_id + 1);
            const auto t_scene = Clock::now();
            SceneGenerationOutcome out = profile_gen_.generate(
                *prof, scene_id, scene_seed);
            timing.scene_generation_ms += msSince(t_scene);
            ++result.scenes_generated;
            ++rs.scenes_generated;
            if (!out.success) {
                out.scene.metadata = out.metadata;
                result.scenes.push_back(out.scene);
                continue;
            }
            ++result.scenes_valid;
            ++rs.scenes_valid;
            scenes[scene_id] = out.scene;

            // ── one-time geometry cache (planning validity) ─────────
            const auto t_geo = Clock::now();
            SceneGeometryCache geo;
            SceneMetadata meta = out.metadata;
            const bool geo_ok = geo.build(out.scene, cfg_, meta);
            timing.scene_geometry_cache_ms += msSince(t_geo);
            out.scene.metadata = meta;
            result.scenes.push_back(out.scene);
            updateLegacyStrata(result, out.scene);
            if (!geo_ok) {
                // scene is invalid for task planning: keep it in the
                // manifest (metadata says why) but generate no tasks.
                continue;
            }

            // ── privileged task qualifier: one-time truth-ESDF grid for
            //    this scene (endpoint / connectivity / straight blocker /
            //    LEFT-RIGHT side routes).  Scene-static, never rebuilt per
            //    task. ───────────────────────────────────────────────
            qualifier_.configure(out.scene, geo, cfg_);

            // ── task candidates for this scene ──────────────────────
            const uint64_t remaining_attempts =
                max_preflights > total_preflight_attempts
                    ? max_preflights - total_preflight_attempts
                    : 0;
            const int remaining_preflight_cap = static_cast<int>(
                std::min<uint64_t>(remaining_attempts,
                                   std::numeric_limits<int>::max()));
            const int task_target =
                std::min(max_tasks_per_scene, remaining_preflight_cap);
            pool_target += static_cast<uint64_t>(task_target);
            std::vector<BlueprintTask> scene_pool;
            uint64_t scene_qualification_expansions = 0;
            bool scene_qualification_budget_exhausted = false;
            int attempts = 0;
            // ── parallel preflight batch ─────────────────────────────
            // Sampling / cheap filter / qualification stay serial (they
            // are cheap and feed the budget checks); the expensive
            // closed-loop preflight is deferred into batches of up to
            // `parallel_workers` candidates and executed concurrently.
            std::vector<PendingPreflight> batch;
            batch.reserve(static_cast<size_t>(parallel_workers));
            // Run every pending preflight on its own worker thread, then
            // merge outcomes back on THIS thread in submission order so
            // analyzer_ / pools / counters / failure_breakdown see exactly
            // the serial sequence (per batch).
            auto flushBatch = [&](std::vector<PendingPreflight>& b) {
                if (b.empty()) return;
                const auto t_flush = Clock::now();
                std::vector<std::future<PreflightOutcome>> futures;
                futures.reserve(b.size());
                for (auto& p : b) {
                    futures.push_back(std::async(
                        std::launch::async, [this, &p]() {
                            PreflightOutcome o;
                            uint64_t local_ticks = 0;
                            o.accepted = preflightOne(
                                p.task, p.scene, p.tick_base, o.summary,
                                p.yaw_err, p.task_tick_budget, local_ticks,
                                o.early_terminated, o.global_tick_truncated,
                                o.reject_reason, o.depth_proxy_ms);
                            o.ticks_used = local_ticks;
                            return o;
                        }));
                }
                for (size_t i = 0; i < b.size(); ++i) {
                    PendingPreflight& p = b[i];
                    PreflightOutcome o = futures[i].get();
                    BlueprintTask& task = p.task;
                    const TaskDistributionSummary& summary = o.summary;
                    const int probe_side = p.probe_side;

                    // Merge the worker's local tick counter into the
                    // global budget (serial section, no contention).
                    timing.depth_proxy_total_ms += o.depth_proxy_ms;
                    ++timing.preflight_count;
                    timing.preflight_ticks += o.ticks_used;
                    total_preflight_ticks += o.ticks_used;
                    ++result.tasks_preflighted;
                    ++rs.preflight_attempted;
                    task.summary = summary;

                    // A probe is only a sampling hint by default.  Keep
                    // the actual preflight label in the candidate pool so
                    // local yaw-first behaviour is not mistaken for a
                    // failed task.  Strict matching remains available for
                    // focused diagnostics.
                    if (o.accepted && probe_side != 0 &&
                        cfg_.macro_probe_require_match) {
                        const bool matched =
                            probe_side > 0
                                ? summary.macro_turn_left_count > 0
                                : summary.macro_turn_right_count > 0;
                        if (!matched) {
                            o.accepted = false;
                            o.reject_reason = "macro_turn_probe_mismatch";
                        }
                    }
                    ++rs.failure_breakdown[o.reject_reason];

                    if (o.global_tick_truncated) {
                        // The GLOBAL remaining tick budget cut this task
                        // short (not a normal task timeout): report it and
                        // stop the whole generation — no further preflight
                        // can run.
                        budget_exhausted =
                            BudgetExhaustion::PREFLIGHT_TICK_BUDGET;
                    }
                    if (!o.early_terminated) ++full_preflight_attempted;
                    if (o.accepted) {
                        ++timing.preflight_success_count;
                        ++result.preflight_success_tasks;
                        ++rs.preflight_success;
                        if (!o.early_terminated) ++full_preflight_success;
                        ++full_preflight_success_after_qual;
                        scene_pool.push_back(task);
                        global_pool.push_back(task);
                        analyzer_.addTask(summary);
                        // Keep deficit-driven task/yaw weights current
                        // within a round; recomputation is negligible next
                        // to preflight.
                        analyzer_.recompute();
                    } else {
                        ++timing.preflight_failure_count;
                        ++result.preflight_failure_count;
                    }
                }
                timing.preflight_total_ms += msSince(t_flush);
                b.clear();
            };
            while (static_cast<int>(scene_pool.size()) < task_target &&
                   attempts < cfg_.max_task_generation_attempts &&
                   !budgetExceeded() && !scene_qualification_budget_exhausted) {
                ++attempts;
                const uint64_t task_seed =
                    mixSeed(scene_seed, 0x5EEDF157ULL +
                                            static_cast<uint64_t>(attempts));
                BlueprintTask task;
                TaskGeomType geom = TaskGeomType::CLEAR;
                double yaw_err = 0.0;
                // P2: scene feasibility mask x global deficit weights.  A
                // class the scene cannot produce (LARGE_OCCLUSION in an
                // empty scene, ...) is zeroed BEFORE sampling, so the
                // sampler never burns attempts on an impossible request.
                const auto feas = task_gen_.feasibilityFor(out.scene, geo);
                std::vector<double> type_weights = exploration_round
                    ? std::vector<double>(static_cast<size_t>(kNumTaskGeomTypes), 1.0)
                    : analyzer_.taskTypeWeights();
                const std::vector<double> yaw_weights = exploration_round
                    ? cfg_.yaw_weights : analyzer_.yawWeights();
                bool any_feasible = false;
                for (size_t i = 0; i < type_weights.size(); ++i) {
                    if (i < feas.size() && !feas[i]) {
                        type_weights[i] = 0.0;
                    } else if (type_weights[i] > 0.0) {
                        any_feasible = true;
                    }
                }
                if (!any_feasible) {
                    // Defensive: CLEAR is always feasible by construction;
                    // if the mask somehow zeroed everything keep CLEAR.
                    type_weights.assign(type_weights.size(), 0.0);
                    type_weights[static_cast<size_t>(TaskGeomType::CLEAR)] = 1.0;
                }
                const auto t_samp = Clock::now();
                const bool sampled = task_gen_.sample(
                    out.scene, geo, type_weights, yaw_weights,
                    task_seed, global_task_id, scene_id, task, geom, yaw_err);
                timing.task_candidate_generation_ms += msSince(t_samp);
                if (!sampled) break;
                ++result.tasks_sampled;
                ++global_task_id;

                // ── cheap staged filter before preflight ────────────
                // Probe rare macro-turn branches while their configured
                // tick targets are unmet.  This is a sampling hint only;
                // preflight must observe the requested TURN label.
                int probe_side = 0;  // +1 LEFT, -1 RIGHT, 0 = ordinary
                if (cfg_.macro_probe_enabled && !exploration_round) {
                    const uint64_t left_target = countTarget(
                        analyzer_, "macro:turn_left", 24);
                    const uint64_t right_target = countTarget(
                        analyzer_, "macro:turn_right", 24);
                    const uint64_t left_now =
                        analyzer_.accumulator().count("macro:turn_left");
                    const uint64_t right_now =
                        analyzer_.accumulator().count("macro:turn_right");
                    if (left_now < left_target || right_now < right_target) {
                        if (left_now < left_target &&
                            (right_now >= right_target || left_now <= right_now)) {
                            probe_side = 1;
                        } else {
                            probe_side = -1;
                        }
                        if (!turnProbeGeometry(task.geom_type)) {
                            // A clear task cannot provide a meaningful
                            // takeover label; resample before preflight.
                            continue;
                        }
                        forceTurnProbeYaw(task, probe_side,
                                           cfg_.macro_probe_yaw_error_deg,
                                           yaw_err);
                    }
                }
                const auto t_filter = Clock::now();
                if (!cheapFilterPass(out.scene, geo, task)) {
                    timing.cheap_filter_ms += msSince(t_filter);
                    ++timing.cheap_filter_rejected;
                    ++result.cheap_filter_rejected;
                    ++rs.cheap_rejected;
                    ++rs.task_candidates;
                    continue;
                }
                timing.cheap_filter_ms += msSince(t_filter);
                ++rs.task_candidates;

                // ── PRIVILEGED task qualification (port of the 2D causal
                //    qualification): endpoint safety -> connectivity ->
                //    straight-corridor blocker -> (blocked only) LEFT /
                //    RIGHT side-constrained A*.  Clear tasks skip the side
                //    search.  Rejects here are CHEAP (no full preflight);
                //    only geometrically fair, both-sides-feasible tasks
                //    reach the expensive expert. ─────────────────────
                if (cfg_.qualification.enabled) {
                    const Vec2d t_start(task.start_x, task.start_y);
                    const Vec2d t_goal(task.goal_x, task.goal_y);
                    TaskQualificationSummary q;
                    // Generation-wide HARD budget: only the remaining
                    // expansion budget is handed to this task; every node
                    // it expands is deducted from it, so the task can
                    // NEVER overshoot the global cap (even mid-round).
                    const uint64_t max_global =
                        cfg_.qualification.max_total_qualification_expansions;
                    uint64_t remaining_global =
                        max_global > total_qualification_expansions
                            ? max_global - total_qualification_expansions
                            : 0;
                    if (remaining_global == 0) {
                        budget_exhausted =
                            BudgetExhaustion::QUALIFICATION_EXPANSION_BUDGET;
                        break;
                    }
                    if (cfg_.qualification.max_expansions_per_scene > 0 &&
                        scene_qualification_expansions >=
                            cfg_.qualification.max_expansions_per_scene) {
                        // Abandon only this scene.  The outer generator can
                        // continue with another random scene while the
                        // generation-wide budget remains available.
                        scene_qualification_budget_exhausted = true;
                        break;
                    }
                    if (cfg_.qualification.max_expansions_per_scene > 0) {
                        remaining_global = std::min<uint64_t>(
                            remaining_global,
                            cfg_.qualification.max_expansions_per_scene -
                                scene_qualification_expansions);
                    }
                    const uint64_t remaining_before = remaining_global;
                    const auto t_qual = Clock::now();
                    qualifier_.qualify(t_start, t_goal, q, qc_round,
                                       remaining_global);
                    timing.task_qualification_ms += msSince(t_qual);
                    // Real-time accounting (immediate, not round-end).
                    const uint64_t used = remaining_before - remaining_global;
                    total_qualification_expansions += used;
                    scene_qualification_expansions += used;
                    task.qualification = q;
                    if (!q.accepted) {
                        ++result.qualification_rejected;
                        continue;  // try another candidate (never the scene)
                    }
                    // Realized geometric class from the qualification
                    // geometry (not the scene profile alone).  Argument
                    // order: (geo, scene, start, goal, q).
                    const TaskGeomType realized = task_gen_.classifyQualified(
                        geo, out.scene, t_start, t_goal, q);
                    task.geom_type = taskGeomTypeName(realized);
                    task.qualification.realized_geom_type = task.geom_type;
                    task.qualification.qualification_class =
                        q.qualification_class;
                    if (probe_side != 0 && q.straight_corridor_clear) {
                        // A turn probe must exercise the upper-layer
                        // takeover case, not merely rotate toward a clear
                        // goal.  Reject clear corridors before the expensive
                        // expert preflight and resample the geometry.
                        continue;
                    }
                }

                // ── full preflight + distribution summary ───────────
                // Deferred to the parallel batch: enqueue now, run the
                // whole batch on worker threads, merge in order.  The
                // per-task budget is snapshotted from the remaining global
                // tick budget at enqueue time (serial section, so no
                // contention); the batch is flushed before new candidates
                // are enqueued, keeping any overshoot bounded.
                ++full_preflight_after_qual;
                // P2 HARD tick budget: the effective per-task budget is
                // capped by the REMAINING global tick budget, so the
                // global tick total can never overshoot by more than
                // (parallel_workers-1) x max_preflight_ticks_per_task.
                const uint64_t remaining_global_ticks =
                    max_preflight_ticks > total_preflight_ticks
                        ? max_preflight_ticks - total_preflight_ticks
                        : 0;
                if (remaining_global_ticks == 0) {
                    budget_exhausted = BudgetExhaustion::PREFLIGHT_TICK_BUDGET;
                    break;
                }
                // Enqueuing a candidate commits one preflight attempt
                // (matches the serial accounting: each preflight consumes
                // exactly one attempt unit).
                if (total_preflight_attempts >= max_preflights) {
                    budget_exhausted = BudgetExhaustion::PREFLIGHT_ATTEMPT_BUDGET;
                    break;
                }
                ++total_preflight_attempts;
                const uint64_t task_tick_budget = std::min<uint64_t>(
                    static_cast<uint64_t>(std::max(1, cfg_.max_preflight_ticks_per_task)),
                    remaining_global_ticks);
                const uint64_t tick_base = task.task_id * 600000ull;
                batch.push_back(PendingPreflight{
                    std::move(task), out.scene, tick_base, yaw_err,
                    task_tick_budget, probe_side});
                if (static_cast<int>(batch.size()) >= parallel_workers) {
                    flushBatch(batch);
                    if (budget_exhausted != BudgetExhaustion::NONE) break;
                }
            }
            // Flush any remaining candidates before moving on (next scene
            // or final selection).
            flushBatch(batch);
            if (timing.preflight_count > 0) {
                timing.preflight_average_ms =
                    timing.preflight_total_ms /
                    static_cast<double>(timing.preflight_count);
            }
            result.total_task_candidates +=
                static_cast<uint64_t>(scene_pool.size());
            if (scene_qualification_budget_exhausted && cfg_.log_rounds) {
                std::fprintf(stderr,
                             "[blueprint] scene %llu qualification cap reached; "
                             "continuing with another scene (pool=%llu)\n",
                             static_cast<unsigned long long>(scene_id),
                             static_cast<unsigned long long>(scene_pool.size()));
            }
        }

        // ── end-of-round stats + sanity log ─────────────────────────
        result.generation_rounds = static_cast<uint64_t>(round);
        analyzer_.recompute();
        const CoverageResult& cov = analyzer_.coverage();
        rs.selected_pool = global_pool.size();
        rs.elapsed_ms = msSince(t_round);
        rs.preflight_avg_ms =
            rs.preflight_attempted > 0
                ? (timing.preflight_total_ms - timing.depth_proxy_total_ms) /
                      static_cast<double>(rs.preflight_attempted)
                : 0.0;
        for (const auto& d : analyzer_.deficits()) {
            if (d.deficit > 1e-9 || d.excess > 1e-9 || d.below_minimum) {
                rs.remaining_deficits.push_back(d.summary());
            }
        }
        // Aggregate this round's privileged qualification counters.
        rs.qualification = qc_round;
        accumulateQual(qual_total, qc_round);
        if (cfg_.qualification.log_qualification_stats && cfg_.log_rounds) {
            const auto& q = rs.qualification;
            std::fprintf(stderr,
                         "[blueprint]   round %llu qualification: checked=%llu "
                         "endpoint=%llu conn=%llu straight_clear=%llu "
                         "blocked=%llu side_attempt=%llu both=%llu "
                         "accept=%llu reject[endpoint=%llu comp=%llu "
                         "global_route=%llu global_astar_budget=%llu "
                         "side_budget=%llu both_required=%llu] "
                         "astar_exp=%llu\n",
                         static_cast<unsigned long long>(round),
                         static_cast<unsigned long long>(q.candidates_checked),
                         static_cast<unsigned long long>(q.endpoint_pass),
                         static_cast<unsigned long long>(q.connectivity_pass),
                         static_cast<unsigned long long>(q.straight_clear),
                         static_cast<unsigned long long>(q.blocked),
                         static_cast<unsigned long long>(q.side_qualification_attempt),
                         static_cast<unsigned long long>(q.both_sides_feasible),
                         static_cast<unsigned long long>(q.accepted),
                         static_cast<unsigned long long>(q.reject_endpoint),
                         static_cast<unsigned long long>(q.reject_different_component),
                         static_cast<unsigned long long>(q.reject_global_route),
                         static_cast<unsigned long long>(q.reject_global_astar_budget),
                         static_cast<unsigned long long>(q.reject_side_search_budget),
                         static_cast<unsigned long long>(q.reject_both_sides_required),
                         static_cast<unsigned long long>(q.total_astar_expansions));
        }
        if (cfg_.log_rounds) {
            std::fprintf(stderr,
                         "[blueprint] round %llu: scenes=%llu/%llu "
                         "candidates=%llu cheap_rej=%llu preflight=%llu "
                         "success=%llu pool=%llu explore=%s hard=%s soft=%s "
                         "elapsed_ms=%.1f qual_ms=%.1f preflight_ms=%.1f\n",
                         static_cast<unsigned long long>(round),
                         static_cast<unsigned long long>(rs.scenes_valid),
                         static_cast<unsigned long long>(rs.scenes_generated),
                         static_cast<unsigned long long>(rs.task_candidates),
                         static_cast<unsigned long long>(rs.cheap_rejected),
                         static_cast<unsigned long long>(rs.preflight_attempted),
                         static_cast<unsigned long long>(rs.preflight_success),
                         static_cast<unsigned long long>(rs.selected_pool),
                         (cfg_.pool_first_exploration &&
                          (static_cast<int>(round) <=
                               std::max(0, cfg_.exploration_rounds) ||
                           (cfg_.exploration_min_pool_tasks > 0 &&
                            global_pool.size() <
                                cfg_.exploration_min_pool_tasks)))
                             ? "1"
                             : "0",
                         cov.hard_minimums_met ? "1" : "0",
                         cov.soft_targets_met ? "1" : "0", rs.elapsed_ms,
                         timing.task_qualification_ms - qual_ms_start,
                         timing.preflight_total_ms - preflight_ms_start);
            // Failure breakdown (stall / no_progress are the key signals to
            // watch after the stall-displacement fix).
            if (!rs.failure_breakdown.empty()) {
                std::string breakdown;
                for (const auto& kv : rs.failure_breakdown) {
                    if (!breakdown.empty()) breakdown += " ";
                    breakdown += kv.first + "=" +
                                 std::to_string(kv.second);
                }
                std::fprintf(stderr, "[blueprint]   round %llu rejections: %s\n",
                             static_cast<unsigned long long>(round),
                             breakdown.c_str());
            }
        }
        round_logs.push_back(rs);
        if (cov.hard_minimums_met && cov.soft_targets_met) {
            // Normal early satisfaction — NOT a budget exhaustion.
            budget_exhausted = BudgetExhaustion::NONE;
            break;
        }
        if (budgetExceeded()) break;
        // Exhausted the generation-round budget while coverage is unmet
        // (only if no more specific budget already stopped us).
        if (round >= max_rounds &&
            budget_exhausted == BudgetExhaustion::NONE) {
            budget_exhausted = BudgetExhaustion::GENERATION_ROUND_BUDGET;
        }
    }

    // ── final greedy selection ─────────────────────────────────────
    const auto t_sel = Clock::now();
    std::vector<uint64_t> per_scene;
    result.tasks = analyzer_.select(global_pool, per_scene);
    timing.selection_ms += msSince(t_sel);
    result.tasks_quota_accepted = result.tasks.size();
    result.tasks_pool_accepted = global_pool.size();
    result.preflight_success_tasks = global_pool.size();
    result.preflighted = global_pool;  // candidate pool (manifest-compatible)
    result.tasks_pool_target = pool_target;
    result.pool_budget_exhausted =
        (total_preflight_attempts >= max_preflights ||
         total_preflight_ticks >= max_preflight_ticks) &&
        analyzer_.coverage().hard_minimums_met == false;

    // ── efficiency + budget diagnostics ────────────────────────────
    result.preflight_attempt_count = total_preflight_attempts;
    result.preflight_success_count = global_pool.size();
    result.preflight_failure_count =
        total_preflight_attempts > global_pool.size()
            ? total_preflight_attempts - global_pool.size()
            : 0;
    result.total_preflight_ticks = total_preflight_ticks;
    result.full_preflight_attempted = full_preflight_attempted;
    result.full_preflight_success = full_preflight_success;
    result.preflight_acceptance_ratio =
        total_preflight_attempts > 0
            ? static_cast<double>(global_pool.size()) /
                  static_cast<double>(total_preflight_attempts)
            : 0.0;
    result.selected_per_preflight_ratio =
        total_preflight_attempts > 0
            ? static_cast<double>(result.tasks.size()) /
                  static_cast<double>(total_preflight_attempts)
            : 0.0;
    result.budget_exhausted_reason = budgetExhaustionName(budget_exhausted);
    result.round_logs = round_logs;

    // ── privileged task-qualification efficiency (aggregate) ───────
    result.qualification = qual_total;
    result.task_candidates_generated = result.tasks_sampled;
    result.endpoint_pass_count = qual_total.endpoint_pass;
    result.connectivity_pass_count = qual_total.connectivity_pass;
    result.straight_clear_count = qual_total.straight_clear;
    result.blocked_count = qual_total.blocked;
    result.side_qualification_attempt_count =
        qual_total.side_qualification_attempt;
    result.both_sides_feasible_count = qual_total.both_sides_feasible;
    result.qualification_accept_count = qual_total.accepted;
    result.total_astar_expansions = qual_total.total_astar_expansions;
    result.qualification_pass_ratio =
        qual_total.candidates_checked > 0
            ? static_cast<double>(qual_total.accepted) /
                  static_cast<double>(qual_total.candidates_checked)
            : 0.0;
    result.full_preflight_success_after_qualification_ratio =
        full_preflight_after_qual > 0
            ? static_cast<double>(full_preflight_success_after_qual) /
                  static_cast<double>(full_preflight_after_qual)
            : 0.0;

    // per-scene selected counts (indexed by scene_id, like the old quota
    // selector) + selected scene ids in order of first appearance.
    result.per_scene_accepted = per_scene;
    {
        std::set<uint64_t> seen;
        for (const auto& t : result.tasks) {
            if (seen.insert(t.scene_id).second) {
                result.selected_scene_ids.push_back(t.scene_id);
            }
        }
        result.selected_scene_count = seen.size();
    }

    // ── coverage on the SELECTED subset ────────────────────────────
    // NOTE: the EFFECTIVE targets come from the analyzer (its copy is
    // populated by buildDefaultTargets() when the config omits them); the
    // raw cfg_.targets may be EMPTY and would silently pass coverage.
    const std::vector<DistributionTarget>& effective_targets =
        analyzer_.targets();
    DistributionAccumulator sel_acc;
    sel_acc.configure(cfg_);
    for (const auto& t : result.tasks) sel_acc.addTask(t.summary);
    const CoverageResult sel_cov =
        evaluateCoverage(sel_acc, effective_targets, cfg_);
    result.hard_minimums_met = sel_cov.hard_minimums_met;
    result.soft_targets_met = sel_cov.soft_targets_met;
    result.warnings = sel_cov.warnings;
    result.remaining_deficits.clear();
    for (const auto& d : sel_cov.deficits) {
        if (d.deficit > 1e-9 || d.excess > 1e-9 || d.below_minimum) {
            result.remaining_deficits.push_back(d.summary());
        }
    }
    // Legacy hard-quota report (unmet_quotas) from the hard deficits.
    result.unmet_quotas.clear();
    for (const auto& w : sel_cov.warnings) {
        if (w.find("[HARD]") != std::string::npos) {
            result.unmet_quotas.push_back(w);
        }
    }

    fillDistributionReport(result, sel_acc);
    fillCategoryCounts(result);

    // ── per-scene floor (SOFT warning: a selected scene with fewer than
    //    min_tasks_per_scene tasks is a warning, never a hard failure —
    //    scene count is a means to distribution balance, not the goal) ──
    for (size_t i = 0; i < result.per_scene_accepted.size(); ++i) {
        const uint64_t c = result.per_scene_accepted[i];
        if (c > 0 && c < static_cast<uint64_t>(cfg_.min_tasks_per_scene)) {
            result.warnings.push_back(
                "scene " + std::to_string(i) + " selected " +
                std::to_string(c) + " < min_tasks_per_scene=" +
                std::to_string(cfg_.min_tasks_per_scene) + " [soft]");
        }
    }

    // ── strata / legacy counts ─────────────────────────────────────
    uint64_t covered = 0;
    for (uint64_t f : result.strata_covered_flags) covered += f;
    result.strata_covered = covered;

    // ── generation_ok (new semantics) ──────────────────────────────
    result.generation_ok =
        result.hard_minimums_met &&
        result.selected_scene_count >=
            static_cast<uint64_t>(std::max(1, cfg_.min_selected_scenes)) &&
        result.tasks.size() >= static_cast<size_t>(cfg_.min_tasks);
    if (!result.generation_ok) {
        if (!result.hard_minimums_met) {
            result.failure_reason = "distribution minimum coverage unmet";
        } else if (result.selected_scene_count <
                   static_cast<uint64_t>(std::max(1, cfg_.min_selected_scenes))) {
            result.failure_reason =
                "insufficient selected scenes (" +
                std::to_string(result.selected_scene_count) + "<" +
                std::to_string(cfg_.min_selected_scenes) + ")";
        } else if (result.tasks.size() < static_cast<size_t>(cfg_.min_tasks)) {
            result.failure_reason = "insufficient selected tasks (" +
                                    std::to_string(result.tasks.size()) + "<" +
                                    std::to_string(cfg_.min_tasks) + ")";
        }
    }
    if (result.generation_ok && budget_exhausted != BudgetExhaustion::NONE) {
        // Reached the requirements but also exhausted a budget: keep the
        // success but record the exhaustion reason for diagnostics.
        if (result.budget_exhausted_reason == "none") {
            result.budget_exhausted_reason =
                budgetExhaustionName(budget_exhausted);
        }
    }

    // ── timing ─────────────────────────────────────────────────────
    timing.total_ms = msSince(t_total);
    result.timing_ms = timing.asMap();
    return result;
}

// ═══════════════════════════════════════════════════════════════════
//  Scene-level parallel pipeline (new architecture)
// ═══════════════════════════════════════════════════════════════════

BlueprintGenerationController::SceneWorkResult
BlueprintGenerationController::runOneScene(int level, int level_index,
                                           uint64_t scene_id,
                                           uint64_t seed) const {
    SceneWorkResult wr;
    wr.scene_id = scene_id;
    wr.level = level;
    const auto t0 = Clock::now();
    double place_ms = 0.0, grid_ms = 0.0, astar_ms = 0.0, preflight_ms = 0.0;
    const WarehouseGeometry& wh = cfg_.warehouse;
    const SceneSpec spec = makeSceneSpec(cfg_, level, level_index, scene_id,
                                         seed);

    // ── 1. scene: cylinders (surface gap >= 1.2, border >= 0.6) ──
    BlueprintScene scene;
    scene.scene_id = scene_id;
    scene.seed = seed;
    scene.profile = (level == 0) ? "small"
                    : (level == 1) ? "medium"
                    : (level == 2) ? "large" : "mixed";
    {
        const auto t1 = Clock::now();
        scene.actual_obstacle_count = placeCylinders(
            cfg_, wh, spec.target_count, spec.rmin, spec.rmax, seed,
            scene.obstacles);
        place_ms = msSince(t1);
    }
    if (scene.actual_obstacle_count == 0) {
        wr.reason = "no cylinders placed";
        return wr;
    }
    scene.is_empty = false;
    scene.generation_valid = true;
    scene.density_class =
        scene.actual_obstacle_count <= 6 ? "sparse"
        : scene.actual_obstacle_count <= 14 ? "medium" : "dense";
    scene.actual_density_class = scene.density_class;
    scene.actual_radius_class = (spec.rmax >= 2.9) ? "large"
                                : (spec.rmax >= 1.4) ? "medium" : "small";
    double rsum = 0.0, rmin_o = 1e9, rmax_o = 0.0;
    for (const auto& o : scene.obstacles) {
        rsum += o.radius;
        rmin_o = std::min(rmin_o, o.radius);
        rmax_o = std::max(rmax_o, o.radius);
    }
    scene.actual_min_radius_m = rmin_o;
    scene.actual_max_radius_m = rmax_o;
    scene.metadata.profile = scene.profile;
    scene.metadata.obstacle_count = scene.actual_obstacle_count;
    scene.metadata.radius_min = rmin_o;
    scene.metadata.radius_max = rmax_o;
    scene.metadata.radius_mean =
        rsum / static_cast<double>(scene.actual_obstacle_count);
    scene.metadata.scene_seed = seed;
    scene.metadata.structure_orientation = "none";

    // ── 2. build the 2D grid ───────────────────────────────────────
    SceneGeometryCache geo;
    SceneMetadata meta;
    {
        const auto t2 = Clock::now();
        const bool grid_ok = geo.build(scene, cfg_, meta);
        grid_ms = msSince(t2);
        if (!grid_ok) {
            wr.reason = "grid build failed: " + meta.geometry_failure_reason;
            wr.scene = scene;
            return wr;
        }
    }
    scene.metadata = meta;
    wr.scene = scene;

    // ── 3. sample start/goal pairs, greedy A*, distance balance,
    //      quick-expert preflight, labels ───────────────────────────
    // scene_target = expected / total scenes, using the ACTUAL per-level
    // scene count (scenes_per_level_list) — the old formula used the fixed
    // scenes_per_level default and silently shrank every scene's quota
    // (e.g. 16 scenes x 160 expected => 4 tasks/scene instead of 10).
    const int per_level_scenes =
        (level >= 0 &&
         level < static_cast<int>(cfg_.scenes_per_level_list.size()))
            ? std::max(1, cfg_.scenes_per_level_list[level])
            : std::max(1, cfg_.scenes_per_level);
    const int scene_target = std::max(
        1, cfg_.expected_collect_tasks /
               (std::max(1, cfg_.scene_levels) * per_level_scenes));
    // 中间多两端短:short : medium : long = 1 : 2 : 1 — the bulk of each
    // scene sits in the MID distance band (richer avoidance behaviour),
    // while the extreme short/long bands stay covered but sparse.
    const int unit = std::max(1, scene_target / 4);
    int dist_targets[3] = {unit, scene_target - 2 * unit, unit};
    int dist_counts[3] = {0, 0, 0};
    // ── 尺度自适应距离分层:在 [spec.dmin, spec.dmax] 内均分三档 ──
    // 大尺度层的 dmin 已抬升 → 整层都是较长距离, 不再分配短路径。
    const double d_lo = spec.dmin, d_hi = spec.dmax;
    const double d_band = std::max(1e-6, (d_hi - d_lo) / 3.0);
    auto dcls = [&](double d) {
        if (d < d_lo + d_band) return 0;
        if (d < d_lo + 2.0 * d_band) return 1;
        return 2;
    };

    const std::vector<size_t>& cells = geo.validCells();
    if (cells.size() < 2) {
        wr.reason = "too few valid free cells";
        return wr;
    }
    const int w = geo.w();
    auto cellCenterOf = [&](size_t id) {
        return geo.cellCenter(static_cast<int>(id % w),
                              static_cast<int>(id / w));
    };
    const double astar_clr = cfg_.free_cell_surface_clearance_m;  // == scene valid-cell clearance
    const uint64_t quick_ticks =
        static_cast<uint64_t>(std::max(1, cfg_.quick_preflight_max_ticks));
    std::mt19937_64 rng(mixSeed(spec.seed, 0x7A57A77ULL));
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    // New preflight planner: behaviour SEGMENT labeler (splits each
    // trajectory into straight / light|large avoidance / detour /
    // medium|long detour segments).  Same expert + same Params2D as the
    // real collection, so the clearance parameters are identical.  The
    // detour-duration thresholds are rescaled by the coarse dt_scale so
    // the PHYSICAL durations stay fixed.
    SegmentLabeler segmenter(30.0, 30, 90, cfg_.quick_preflight_dt_scale);

    uint64_t attempts = 0;
    const uint64_t max_attempts = 2000;
    uint64_t task_seq = 0;
    // ── blocked / unblocked balance ─────────────────────────────────
    // A macro detour is triggered by the GEOMETRY (the direct start->goal
    // line is blocked by an obstacle, so the 5 Hz corrector takes over) —
    // NOT by the initial yaw.  The blocked share is scaled by the LEVEL:
    // large / mixed scenes (big cylinders) can actually produce long
    // detours, small / medium scenes cannot (obstacles are small, the
    // detour is short) — so small/medium target mostly straight +
    // avoidance labels, large/mixed target more detour labels.
    double blocked_ratio = 0.5;
    if (level >= 2) {
        // 放宽(large/mixed 0.75→0.50):blocked 直线穿障在短距离+高密度下
        // 难构造, 是每场景任务量的瓶颈; 降低占比让更多任务能生成。
        blocked_ratio = 0.50;
    } else if (level == 1) {
        blocked_ratio = 0.35;
    } else {
        blocked_ratio = 0.30;
    }
    const int blocked_target =
        std::max(1, static_cast<int>(std::round(scene_target * blocked_ratio)));
    const int unblocked_target = std::max(1, scene_target - blocked_target);
    int blocked_accept = 0, unblocked_accept = 0;
    // ── 步骤3+4: macro-turn probe counters (per-scene) ────────────
    // 序列路径的 macro_probe 只作用于 taskTypeWeights 采样（task_candidate_
    // generator）；scene_parallel 路径的 geom_type 恒为 CLEAR，该探针从未
    // 生效——这是宏观 TURN 任务供应不足的根因。这里对 BLOCKED 任务直接
    // 强制 ±90° 初始 yaw 偏移，使 5Hz 专家起步即触发 TURN 接管。
    int probe_left_issued = 0, probe_right_issued = 0;
    while ((dist_counts[0] + dist_counts[1] + dist_counts[2] < scene_target ||
            blocked_accept < blocked_target ||
            unblocked_accept < unblocked_target) &&
           attempts < max_attempts && wr.preflights_run < 300) {
        ++attempts;
        // pick the most-deficient distance class
        int cls = 0;
        if (dist_counts[0] >= dist_targets[0] &&
            dist_counts[1] >= dist_targets[1]) {
            cls = 2;
        } else if (dist_counts[0] >= dist_targets[0]) {
            cls = 1;
        }
        const bool need_blocked = blocked_accept < blocked_target;
        const bool need_unblocked = unblocked_accept < unblocked_target;
        // sample start
        const Vec2d start = cellCenterOf(cells[rng() % cells.size()]);
        bool found = false;
        Vec2d goal;
        bool line_blocked = false;
        // ── large / mixed scenes: CONSTRUCT a "goal behind a big
        //    obstacle" pair when a blocked task is needed.  Random
        //    sampling rarely hits a CORE-blocked straight line when a
        //    scene holds only a few big cylinders, so we place the goal
        //    on the far side of a random obstacle — the straight line then
        //    provably cuts through its core and forces a macro detour. ──
        const bool construct_blocked =
            (level >= 2) && need_blocked && !need_unblocked &&
            !scene.obstacles.empty();
        if (construct_blocked) {
            const double band_center = (cls == 0) ? (d_lo + 0.5 * d_band)
                                       : (cls == 1) ? (d_lo + 1.5 * d_band)
                                       : (d_lo + 2.5 * d_band);
            for (int gatt = 0; gatt < 250 && !found; ++gatt) {
                const auto& o = scene.obstacles[rng() % scene.obstacles.size()];
                const Vec2d oc(o.x, o.y);
                const Vec2d dvec = start - oc;
                const double ds = dvec.norm();
                if (ds < 1e-6) continue;
                const Vec2d dir = dvec / ds;  // obstacle -> start
                // goal on the far side: start->goal distance = ds + r_goal
                double r_goal = band_center - ds;
                if (r_goal < o.radius + astar_clr + 0.5) {
                    r_goal = o.radius + astar_clr + 0.5;  // clear the far side
                }
                const Vec2d g = oc - dir * r_goal;
                const double d = (g - start).norm();
                if (dcls(d) != cls) continue;
                if (!geo.pointFreeMain(g, astar_clr)) continue;
                goal = g;
                line_blocked = true;  // straight line cuts the core
                found = true;
            }
        }
        // ordinary sampling (small/medium scenes, or when the needed
        // blocked/unblocked class can be met by the random draw)
        if (!construct_blocked && !found) {
            for (int gatt = 0; gatt < 400 && !found; ++gatt) {
                const Vec2d g = cellCenterOf(cells[rng() % cells.size()]);
                const double d = (g - start).norm();
                if (dcls(d) != cls ||
                    d < spec.dmin - 1e-9 ||
                    d > spec.dmax + 1e-9) {
                    continue;
                }
                const bool blk = lineBlocked(scene, start, g, astar_clr,
                                             /*strict_core=*/level >= 2);
                if (need_blocked && !need_unblocked && !blk) continue;
                if (need_unblocked && !need_blocked && blk) continue;
                goal = g;
                line_blocked = blk;
                found = true;
            }
        }
        if (!found) continue;
        // greedy toward-goal A* connectivity check (fast)
        const auto t_astar = Clock::now();
        GreedyAStarResult astar =
            greedyAStar(geo, start, goal, astar_clr, 10000);
        astar_ms += msSince(t_astar);
        ++wr.astar_calls;
        wr.astar_expansions += static_cast<uint64_t>(astar.expansions);
        if (!astar.reachable) continue;
        // NEW: reject absurdly long detours.  A start/goal pair on OPPOSITE
        // sides of a wall / big building is A*-connected only by walking all
        // the way around the structure — the resulting episode is a useless
        // "super-detour" for the 5 Hz student.  Cap the route length at
        // stretch_ratio x straight + slack.
        {
            const double straight = (goal - start).norm();
            const double max_route =
                straight * cfg_.max_route_stretch_ratio +
                cfg_.max_route_stretch_slack_m;
            if (astar.path_len_m > max_route) {
                continue;
            }
        }

        // build the task
        BlueprintTask task;
        task.scene_id = scene_id;
        task.task_id = scene_id * 100000ULL + task_seq;
        task.seed = mixSeed(spec.seed, 0x7A57A77ULL + attempts);
        task.start_x = start.x();
        task.start_y = start.y();
        task.goal_x = goal.x();
        task.goal_y = goal.y();
        task.flight_height_m = cfg_.flight_height_min_m +
            u01(rng) * (cfg_.flight_height_max_m - cfg_.flight_height_min_m);
        // ── initial yaw: aim at the goal with a MODEST random offset
        //    (0..35°, mirror-balanced).  A macro detour is decided by the
        //    blocked-geometry, NOT by a large initial yaw error, so we no
        //    longer force huge offsets. ─────────────────────────────
        const double goal_bearing =
            std::atan2(goal.y() - start.y(), goal.x() - start.x());
        const double yaw_offset =
            (u01(rng) < 0.5 ? -1.0 : 1.0) * (35.0 * u01(rng)) * deg2rad(1.0);
        task.initial_yaw =
            CoordinateAdapter::expertYawToFlightmare(goal_bearing + yaw_offset);
        double expert_yaw =
            CoordinateAdapter::flightmareYawToExpert(task.initial_yaw);
        double yaw_err = wrapAngle(goal_bearing - expert_yaw);
        task.geom_type = "CLEAR";
        const double dline = (goal - start).norm();
        const int dcls_line = dcls(dline);
        task.distance_class = dcls_line == 0 ? "short"
                              : (dcls_line == 1 ? "medium" : "long");
        // ── 步骤3+4: macro-turn probe for BLOCKED tasks ────────────
        // 对 blocked 任务（直线穿障）强制 ±yaw_error 初始朝向，使 5Hz 专家
        // 在起步阶段即进入搜索旋转 → 产生真实的 TURN_LEFT / TURN_RIGHT 标签。
        // 每场景最多 probe blocked_target 个任务；probe 计数在发出时递增（两侧
        // 交替），匹配失败只是拒绝该候选，blocked_target 仍由普通 blocked 任务
        // 达成，保证不烧穿 preflight 预算。
        int probe_side = 0;  // +1 LEFT, -1 RIGHT, 0 = 不强制
        if (cfg_.macro_probe_enabled && line_blocked &&
            blocked_accept < blocked_target &&
            (probe_left_issued + probe_right_issued) < blocked_target) {
            probe_side =
                (probe_left_issued <= probe_right_issued) ? 1 : -1;
            if (probe_side > 0) {
                ++probe_left_issued;
            } else {
                ++probe_right_issued;
            }
            forceTurnProbeYaw(task, probe_side,
                              cfg_.macro_probe_yaw_error_deg, yaw_err);
        }
        ++wr.candidates_sampled;
        ++task_seq;

        // ── quick expert preflight (simplified, with behaviour SEGMENT
        //    labelling — the new preflight planner) ────────────────
        TaskDistributionSummary summary;
        uint64_t local_ticks = 0;
        bool early = false, gtrunc = false;
        std::string reason = "accepted";
        double depth_ms = 0.0;
        const uint64_t tick_base = task.task_id * 600000ull;
        segmenter.reset();
        const auto t_pre = Clock::now();
        const bool accepted =
            preflightOne(task, scene, tick_base, summary, yaw_err, quick_ticks,
                         local_ticks, early, gtrunc, reason, depth_ms,
                         &segmenter);
        preflight_ms += msSince(t_pre);
        ++wr.preflights_run;
        task.summary = summary;
        // ── 步骤3+4: macro-turn probe match check ─────────────────
        // require_match=true 时 probe 任务必须实际产生所请求的 TURN 标签，
        // 否则拒绝该候选（不进 pool，也不计入 blocked_accept），继续采样补足。
        if (probe_side != 0 && cfg_.macro_probe_require_match) {
            const bool matched =
                probe_side > 0 ? summary.macro_turn_left_count > 0
                               : summary.macro_turn_right_count > 0;
            if (!matched) {
                task.behavior_class = "rejected";
                task.side_class = "none";
                task.audit.preflight_status =
                    "preflight_rejected:macro_turn_probe_mismatch";
                wr.rejected.push_back(task);
                continue;  // 不匹配：跳过 quick_accept，继续采样
            }
        }
        // This task's behaviour SEGMENT labels (all behaviours that
        // occurred anywhere in the trajectory).
        std::map<std::string, uint64_t> task_seg;
        for (const auto& kv : segmenter.labelCounts()) {
            task_seg[kv.first] += kv.second;
        }
        task.segment_label_counts = task_seg;
        // Quick preflight acceptance is RELAXED: the reduced tick budget is
        // far too short to physically reach the goal (tasks span 4..28 m),
        // so "reached the goal" would reject everything.  A candidate is
        // accepted when the quick flight was SAFE (no collision / no
        // out-of-bounds) and the macro label is valid — the expert's
        // behaviour label is still meaningful for the balance statistics.
        // Full-arrival tasks (accepted=true) keep the original label.
        const bool safe = !summary.collision && !summary.out_of_bounds;
        const bool macro_ok = task.audit.macro_label_ok;
        const bool quick_accept = accepted || (safe && macro_ok);
        if (quick_accept) {
            if (!accepted && task.behavior_class == "rejected") {
                // Safe flight that did not reach the goal: re-derive the
                // behaviour label from the summary (mirrors preflightOne).
                const uint64_t tl = summary.macro_turn_left_count;
                const uint64_t tr = summary.macro_turn_right_count;
                const uint64_t nm = summary.macro_normal_count;
                if (task.saw_turn_left && nm > 0) {
                    task.behavior_class = "turn_normal";
                } else if (tl > 0 && tr == 0) {
                    task.behavior_class = "turn_left";
                } else if (tr > 0 && tl == 0) {
                    task.behavior_class = "turn_right";
                } else if (tl > 0 && tr > 0) {
                    task.behavior_class = "turn_both";
                } else if (task.turn_update_count > 0 ||
                           task.normal_update_count > 0) {
                    task.behavior_class = "long_takeover";
                } else if (nm > 0) {
                    task.behavior_class = "normal";
                } else if (summary.local_avoidance_count > 0) {
                    task.behavior_class = "local_avoidance";
                } else {
                    task.behavior_class = "clear";
                }
                task.summary.behavior_class = task.behavior_class;
            }
            wr.tasks.push_back(task);
            wr.task_segment_counts.push_back(task_seg);
            for (const auto& kv : task_seg) {
                wr.segment_label_counts[kv.first] += kv.second;
            }
            ++wr.label_counts[task.behavior_class];
            ++wr.dist_counts[task.distance_class];
            ++dist_counts[cls];
            if (line_blocked) {
                ++blocked_accept;
            } else {
                ++unblocked_accept;
            }
        } else {
            task.behavior_class = "rejected";
            task.side_class = "none";
            wr.rejected.push_back(task);
        }
    }
    wr.ok = true;
    wr.wall_ms = msSince(t0);
    std::fprintf(stderr,
                 "[blueprint-scene] level %d idx %d scene %llu: cylinders=%d "
                 "candidates=%d astar_calls=%llu astar_exp=%llu "
                 "preflights=%d accepted=%zu wall_ms=%.0f "
                 "[place=%.0f grid=%.0f astar=%.0f preflight=%.0f]",
                 level, level_index,
                 static_cast<unsigned long long>(scene_id),
                 scene.actual_obstacle_count, wr.candidates_sampled,
                 static_cast<unsigned long long>(wr.astar_calls),
                 static_cast<unsigned long long>(wr.astar_expansions),
                 wr.preflights_run, wr.tasks.size(), wr.wall_ms,
                 place_ms, grid_ms, astar_ms, preflight_ms);
    if (!wr.segment_label_counts.empty()) {
        std::fprintf(stderr, " seg=");
        for (const auto& kv : wr.segment_label_counts) {
            std::fprintf(stderr, "%s:%llu", kv.first.c_str(),
                         static_cast<unsigned long long>(kv.second));
        }
    }
    std::fprintf(stderr, "\n");
    return wr;
}

BlueprintResult BlueprintGenerationController::generateSceneParallel() {
    BlueprintResult result;
    const auto t0 = Clock::now();
    result.base_seed = cfg_.base_seed;

    const int levels = std::max(1, cfg_.scene_levels);
    // Per-level scene counts (sparse -> dense), from scenes_per_level_list
    // when it matches the level count, else the uniform scenes_per_level.
    std::vector<int> per_level_counts(
        levels, std::max(1, cfg_.scenes_per_level));
    if (static_cast<int>(cfg_.scenes_per_level_list.size()) == levels) {
        for (int L = 0; L < levels; ++L) {
            per_level_counts[L] =
                std::max(1, cfg_.scenes_per_level_list[L]);
        }
    }
    const int n_threads = std::max(1, cfg_.scene_parallel_threads);
    uint64_t n_scenes = 0;
    for (int L = 0; L < levels; ++L) n_scenes += per_level_counts[L];
    const int total_target = std::max(1, cfg_.expected_collect_tasks);

    // ── Phase 1: main thread pre-generates the scene specs ────────
    struct Spec {
        int level;
        int idx;
        uint64_t scene_id;
        uint64_t seed;
    };
    std::vector<Spec> specs;
    specs.reserve(n_scenes);
    uint64_t scene_id = 0;
    for (int L = 0; L < levels; ++L) {
        for (int j = 0; j < per_level_counts[L]; ++j) {
            const uint64_t seed =
                mixSeed(cfg_.base_seed, 0x5CEA5001ULL + scene_id);
            specs.push_back({L, j, scene_id, seed});
            ++scene_id;
        }
    }

    // ── Phase 2: scene-level parallel (fixed worker pool) ─────────
    std::vector<SceneWorkResult> results(n_scenes);
    std::atomic<size_t> next{0};
    {
        std::vector<std::thread> pool;
        pool.reserve(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            pool.emplace_back([this, &next, &results, &specs]() {
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= specs.size()) break;
                    const Spec& s = specs[i];
                    results[i] =
                        runOneScene(s.level, s.idx, s.scene_id, s.seed);
                }
            });
        }
        for (auto& th : pool) th.join();
    }

    // ── Phase 3: merge + label balance + reasonableness ───────────
    std::vector<std::vector<BlueprintTask>> by_level(
        static_cast<size_t>(levels));
    std::vector<BlueprintTask> pool_all;
    uint64_t total_candidates = 0, total_preflights = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        const SceneWorkResult& r = results[i];
        if (!r.ok) {
            if (cfg_.log_rounds) {
                std::fprintf(stderr,
                             "[blueprint-scene] scene %llu (level %d) skipped: %s\n",
                             static_cast<unsigned long long>(r.scene_id),
                             r.level, r.reason.c_str());
            }
            continue;
        }
        result.scenes.push_back(r.scene);
        total_candidates += static_cast<uint64_t>(r.candidates_sampled);
        total_preflights += static_cast<uint64_t>(r.preflights_run);
        if (r.level >= 0 && r.level < levels) {
            for (const auto& t : r.tasks) by_level[static_cast<size_t>(r.level)].push_back(t);
        }
        for (const auto& t : r.tasks) pool_all.push_back(t);
    }
    result.scenes_generated = result.scenes.size();
    result.scenes_valid = result.scenes.size();
    result.tasks_sampled = total_candidates;
    result.tasks_preflighted = static_cast<uint64_t>(pool_all.size());

    // label overview
    std::map<std::string, uint64_t> total_labels;
    for (const auto& t : pool_all) ++total_labels[t.behavior_class];

    // behaviour SEGMENT label overview (new preflight planner: each
    // trajectory is split into straight / light|large avoidance / detour
    // / medium|long detour segments).
    std::map<std::string, uint64_t> total_segment_labels;
    for (const auto& r : results) {
        for (const auto& kv : r.segment_label_counts) {
            total_segment_labels[kv.first] += kv.second;
        }
    }

    // ── balance pick: per level, approximate label balance ─────────
    const int per_level_target = std::max(1, total_target / std::max(1, levels));
    std::vector<BlueprintTask> selected;
    std::vector<uint64_t> selected_per_level(static_cast<size_t>(levels), 0);
    for (int L = 0; L < levels; ++L) {
        const auto& ltasks = by_level[static_cast<size_t>(L)];
        if (ltasks.empty()) continue;
        // group by behaviour label
        std::map<std::string, std::vector<const BlueprintTask*>> by_label;
        for (const auto& t : ltasks) by_label[t.behavior_class].push_back(&t);
        const int n_labels = std::max(1, static_cast<int>(by_label.size()));
        int per_label = std::max(1, per_level_target / n_labels);
        std::vector<const BlueprintTask*> picked;
        std::map<std::string, size_t> next_idx;
        // first pass: per_label from each label group
        for (auto& kv : by_label) {
            const int take = std::min(per_label, static_cast<int>(kv.second.size()));
            for (int k = 0; k < take; ++k) picked.push_back(kv.second[k]);
            next_idx[kv.first] = static_cast<size_t>(take);
        }
        // top-up: keep the least-represented label ahead, but only from
        // groups that still have un-picked candidates (a group that ran
        // out must not block filling from the remaining groups).
        int guard = 0;
        while (static_cast<int>(picked.size()) < per_level_target &&
               static_cast<int>(picked.size()) < static_cast<int>(ltasks.size()) &&
               guard < 100000) {
            ++guard;
            std::map<std::string, int> cnt;
            for (const auto* p : picked) ++cnt[p->behavior_class];
            // find the least-represented label that still has candidates
            const BlueprintTask* best = nullptr;
            int best_cnt = INT_MAX;
            std::string best_key;
            for (auto& kv : by_label) {
                if (next_idx[kv.first] >= kv.second.size()) continue;  // drained
                if (cnt[kv.first] < best_cnt) {
                    best_cnt = cnt[kv.first];
                    best_key = kv.first;
                }
            }
            if (best_key.empty()) break;
            best = by_label[best_key][next_idx[best_key]++];
            picked.push_back(best);
        }
        for (const auto* p : picked) {
            selected.push_back(*p);
            ++selected_per_level[static_cast<size_t>(L)];
        }
    }
    result.tasks = selected;
    result.tasks_pool_accepted = static_cast<uint64_t>(pool_all.size());
    result.tasks_quota_accepted = static_cast<uint64_t>(selected.size());
    result.preflighted = pool_all;

    // ── reasonableness report ──────────────────────────────────────
    const double avg = selected.empty()
                           ? 0.0
                           : static_cast<double>(selected.size()) /
                                 std::max<size_t>(1, result.scenes.size());
    std::fprintf(stderr,
                 "\n[blueprint-scene] ===== scene-level parallel summary =====\n"
                 "  scenes ok/planned = %zu / %llu   threads = %d\n"
                 "  candidates = %llu  quick-preflights = %llu  accepted = %zu\n"
                 "  selected = %zu  (expected %d)   avg/scene = %.2f\n",
                 result.scenes.size(), static_cast<unsigned long long>(n_scenes),
                 n_threads, static_cast<unsigned long long>(total_candidates),
                 static_cast<unsigned long long>(total_preflights),
                 pool_all.size(), selected.size(), total_target, avg);
    for (int L = 0; L < levels; ++L) {
        std::fprintf(stderr, "  level %d: tasks(accepted)=%zu selected=%llu\n",
                     L, by_level[static_cast<size_t>(L)].size(),
                     static_cast<unsigned long long>(selected_per_level[static_cast<size_t>(L)]));
    }
    std::fprintf(stderr, "  label distribution (pool):");
    for (const auto& kv : total_labels) {
        std::fprintf(stderr, " %s=%llu", kv.first.c_str(),
                     static_cast<unsigned long long>(kv.second));
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "  segment-label distribution (pool):");
    for (const auto& kv : total_segment_labels) {
        std::fprintf(stderr, " %s=%llu", kv.first.c_str(),
                     static_cast<unsigned long long>(kv.second));
    }
    std::fprintf(stderr, "\n");
    // per-level segment-label distribution (long detours are only
    // PHYSICALLY producible in scenes with large obstacles).
    std::vector<std::map<std::string, uint64_t>> seg_by_level(
        static_cast<size_t>(levels));
    for (const auto& r : results) {
        if (r.level >= 0 && r.level < levels) {
            for (const auto& kv : r.segment_label_counts) {
                seg_by_level[static_cast<size_t>(r.level)][kv.first] += kv.second;
            }
        }
    }
    for (int L = 0; L < levels; ++L) {
        std::fprintf(stderr, "  level %d segment-labels:", L);
        for (const auto& kv : seg_by_level[static_cast<size_t>(L)]) {
            std::fprintf(stderr, " %s=%llu", kv.first.c_str(),
                         static_cast<unsigned long long>(kv.second));
        }
        std::fprintf(stderr, "\n");
    }
    const bool reached = static_cast<int>(selected.size()) >=
                         static_cast<int>(0.8 * total_target);
    std::fprintf(stderr,
                 "  reasonableness: expected %d -> got %zu (%s)\n",
                 total_target, selected.size(),
                 reached ? "reaches the target (>=80%)" : "below 80% of target");
    if (!reached) {
        result.warnings.push_back(
            "scene-level parallel: selected " +
            std::to_string(selected.size()) + " < 80% of expected " +
            std::to_string(total_target) +
            "; increase scenes_per_level / expected_collect_tasks or relax "
            "placement constraints");
    }

    result.generation_ok =
        !selected.empty() &&
        static_cast<int>(selected.size()) >= std::max(1, cfg_.min_tasks);
    if (!result.generation_ok) {
        result.failure_reason = "insufficient selected tasks";
    }
    // ── coverage on the SELECTED subset (same semantics as the serial
    //    path): the scene-parallel path previously never evaluated the
    //    distribution targets, so hard minimums were silently ignored and
    //    generation_ok stayed true even when e.g. the behavior:turn_both /
    //    behavior:long_takeover floors were unmet.  Evaluate now and gate
    //    generation_ok on hard_minimums_met exactly like the serial path. ─
    const std::vector<DistributionTarget>& effective_targets =
        analyzer_.targets();
    DistributionAccumulator sel_acc;
    sel_acc.configure(cfg_);
    for (const auto& t : selected) sel_acc.addTask(t.summary);
    const CoverageResult sel_cov =
        evaluateCoverage(sel_acc, effective_targets, cfg_);
    result.hard_minimums_met = sel_cov.hard_minimums_met;
    result.soft_targets_met = sel_cov.soft_targets_met;
    for (const auto& w : sel_cov.warnings) result.warnings.push_back(w);
    result.remaining_deficits.clear();
    for (const auto& d : sel_cov.deficits) {
        if (d.deficit > 1e-9 || d.excess > 1e-9 || d.below_minimum) {
            result.remaining_deficits.push_back(d.summary());
        }
    }
    result.unmet_quotas.clear();
    for (const auto& w : sel_cov.warnings) {
        if (w.find("[HARD]") != std::string::npos) {
            result.unmet_quotas.push_back(w);
        }
    }
    fillDistributionReport(result, sel_acc);
    result.generation_ok =
        result.hard_minimums_met &&
        !selected.empty() &&
        static_cast<int>(selected.size()) >= std::max(1, cfg_.min_tasks);
    if (!result.generation_ok) {
        result.failure_reason = result.hard_minimums_met
                                    ? "insufficient selected tasks"
                                    : "distribution minimum coverage unmet";
    }
    result.generation_rounds = 1;
    result.timing_ms["scene_level_total_ms"] = msSince(t0);
    result.timing_ms["total_ms"] = msSince(t0);
    return result;
}

}  // namespace expert
}  // namespace il_dataset
