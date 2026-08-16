#include "il_dataset/hierarchical_expert/blueprint_generation_controller.hpp"

#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"
#include "il_dataset/hierarchical_expert/preflight_simulator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <map>
#include <set>

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
    uint64_t& total_preflight_ticks, bool& early_terminated,
    double& depth_proxy_ms) const {
    PreflightSimulator sim(p_);
    Scene2D s2d;
    s2d.min_bounds = cfg_.warehouse.freeMin();
    s2d.max_bounds = cfg_.warehouse.freeMax();
    s2d.valid = true;
    for (const auto& o : scene.obstacles) {
        Obstacle2D ob;
        ob.center = Vec2d(o.x, o.y);
        ob.radius = o.radius;
        ob.id = o.id;
        s2d.obstacles.push_back(ob);
    }
    early_terminated = false;
    const Vec2d wall_envelope_min = cfg_.warehouse.envelopeMin();
    const Vec2d wall_envelope_max = cfg_.warehouse.envelopeMax();
    const Vec2d* wall_min = nullptr;
    const Vec2d* wall_max = nullptr;
    if (cfg_.walls_visible_in_observation) {
        wall_min = &wall_envelope_min;
        wall_max = &wall_envelope_max;
    }
    sim.configure(s2d, s2d.min_bounds, s2d.max_bounds, wall_min, wall_max);

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
    const bool depth_walls = cfg_.walls_visible_in_observation;

    // ── closed-loop preflight ──────────────────────────────────────
    Vec2d prev(task.start_x, task.start_y);
    double path_len = 0.0;
    const uint64_t budget = std::max<uint64_t>(1, cfg_.max_preflight_ticks_per_task);
    const uint64_t stride = std::max<uint64_t>(1, cfg_.depth_proxy_sample_stride_ticks);
    uint64_t ticks = 0;
    bool reached = false, collision = false, out_of_bounds = false;
    bool macro_label_invalid = false, qual_exceeded = false;
    double min_clearance = std::numeric_limits<double>::infinity();
    double clear_sum = 0.0, clear_count = 0.0;
    double obs_clear_sum = 0.0, obs_clear_count = 0.0;
    uint64_t max_consecutive_active = 0, cur_consecutive_active = 0;
    uint64_t turn_update_frames = 0, normal_update_frames = 0;
    bool saw_turn_left = false, saw_turn_right = false;

    // Early-termination bookkeeping (blueprint-only; never changes expert
    // labels): no-progress over a rolling window, and a stall detector.
    const Vec2d orig_goal(task.goal_x, task.goal_y);
    std::deque<double> goal_dist_window;
    const int no_prog_win = std::max(1, cfg_.no_progress_window_ticks);
    int stall_consecutive = 0;
    const int stall_win = std::max(1, cfg_.stall_window_ticks);

    for (uint64_t t = 0; t < budget; ++t) {
        // Depth proxy at stride BEFORE the step (matches runtime: the
        // depth used by the expert is the one at the CURRENT pose).
        if (t % stride == 0) {
            const auto t_depth = Clock::now();
            const auto sample = depth_proxy.castAt(
                sim.state().position, sim.state().yaw, scene.obstacles,
                depth_walls, wall_envelope_min, wall_envelope_max);
            depth_proxy_ms += msSince(t_depth);
            depth_proxy.accumulate(sample, summary);
        }

        const auto res = sim.step(tick_base + t, false);
        ticks = t + 1;
        const ExpertStepOutput& out = res.output;

        // Continuous swept truth clearance / collision over prev->new.
        const double seg_clr = truth.segmentMinClearance(
            prev.x(), prev.y(), res.state.position.x(), res.state.position.y());
        min_clearance = std::min(min_clearance, seg_clr);
        if (std::isfinite(seg_clr)) {
            clear_sum += seg_clr;
            clear_count += 1.0;
        }
        if (res.truth_collision ||
            truth.segmentCollision(prev.x(), prev.y(),
                                   res.state.position.x(),
                                   res.state.position.y())) {
            collision = true;
            break;
        }
        if (res.out_of_bounds ||
            truth.segmentCrossesBounds(prev.x(), prev.y(),
                                       res.state.position.x(),
                                       res.state.position.y(),
                                       p_.drone_radius)) {
            out_of_bounds = true;
            break;
        }
        path_len += (res.state.position - prev).norm();
        prev = res.state.position;

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
            // Correction-angle / distance histograms when the corrector
            // is actively redirecting (NORMAL / TURN).
            if (out.target_correction_active) {
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
        // No-progress: over a rolling window of `no_progress_window_ticks`
        // the distance to the ORIGINAL goal must have shrunk by at least
        // `no_progress_min_progress_m`.  Stalled / circling drones burn
        // the whole preflight budget for nothing; cut them short.
        {
            const double d_to_goal = (orig_goal - res.state.position).norm();
            goal_dist_window.push_back(d_to_goal);
            if (static_cast<int>(goal_dist_window.size()) > no_prog_win + 1) {
                goal_dist_window.pop_front();
            }
            if (static_cast<int>(goal_dist_window.size()) == no_prog_win + 1) {
                const double shrink =
                    goal_dist_window.front() - goal_dist_window.back();
                if (shrink < cfg_.no_progress_min_progress_m) {
                    early_terminated = true;
                    break;
                }
            }
        }
        // Stall: |velocity| (per-tick displacement proxy) below the stall
        // threshold for `stall_window_ticks` while not in a legitimate
        // TURN update.
        {
            const double disp =
                (res.state.position - prev).norm();  // per 1/30 s step
            const bool in_turn =
                out.macro_correction_type == "TURN_LEFT" ||
                out.macro_correction_type == "TURN_RIGHT";
            if (!in_turn && disp < cfg_.stall_speed_mps / 30.0) {
                ++stall_consecutive;
            } else {
                stall_consecutive = 0;
            }
            if (stall_consecutive >= stall_win) {
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

    // ── summary quality fields ─────────────────────────────────────
    summary.preflight_ticks = ticks;
    summary.preflight_duration_s = static_cast<double>(ticks) / 30.0;
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
    task.audit.out_of_bounds = out_of_bounds;
    task.audit.macro_label_ok = !macro_label_invalid;
    task.audit.qualification_exceeded = qual_exceeded;
    task.audit.path_length_m = path_len;
    task.audit.path_stretch_ratio = summary.path_stretch_ratio;
    task.audit.preflight_duration_s = summary.preflight_duration_s;
    task.audit.accepted =
        reached && !collision && !out_of_bounds && !macro_label_invalid &&
        !qual_exceeded;

    total_preflight_ticks += ticks;
    if (!task.audit.accepted) {
        task.behavior_class = "rejected";
        task.side_class = "none";
        task.audit.preflight_status =
            early_terminated
                ? "preflight_rejected:early_termination"
                : (collision
                       ? "preflight_rejected:truth_collision"
                       : (out_of_bounds
                              ? "preflight_rejected:out_of_bounds"
                              : (qual_exceeded
                                     ? "preflight_rejected:qualification_budget"
                                     : (!reached
                                            ? "preflight_rejected:goal_not_reached"
                                            : "preflight_rejected:macro_label_invalid"))));
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

    // parallel_tasks is NOT implemented: fail fast instead of silently
    // ignoring the requested concurrency (the il_config validator also
    // rejects it, this is the C++ guard for direct API users).
    if (cfg_.parallel_tasks) {
        result.failure_reason =
            "parallel_tasks is not implemented (must be false)";
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

        const int scene_budget = max_scene_candidates - static_cast<int>(scenes.size());
        const int rounds_left = max_rounds - round + 1;
        const int round_scenes =
            std::max(1, static_cast<int>(std::ceil(
                            static_cast<double>(scene_budget) /
                            static_cast<double>(rounds_left))));

        std::mt19937_64 round_rng(mixSeed(cfg_.base_seed, 0x0000F00DULL + round));

        for (int s = 0; s < round_scenes; ++s) {
            if (budgetExceeded()) break;
            if (scenes.size() >= static_cast<size_t>(max_scene_candidates)) {
                budget_exhausted = BudgetExhaustion::SCENE_BUDGET;
                break;
            }

            // ── profile pick (deficit-driven or explicit sequence) ──
            const SceneProfile* prof = nullptr;
            if (!cfg_.profile_sequence.empty()) {
                const size_t idx =
                    static_cast<size_t>(scene_counter) % cfg_.profile_sequence.size();
                prof = profile_gen_.findProfile(cfg_.profile_sequence[idx]);
                if (!prof) break;  // unknown profile name: stop (config error)
            } else {
                prof = profile_gen_.pickProfile(round_rng,
                                                analyzer_.profileTagWeights());
            }
            if (!prof) break;

            // ── scene realization ───────────────────────────────────
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
            const auto t_task = Clock::now();
            std::vector<BlueprintTask> scene_pool;
            int attempts = 0;
            while (static_cast<int>(scene_pool.size()) < task_target &&
                   attempts < cfg_.max_task_generation_attempts &&
                   !budgetExceeded()) {
                ++attempts;
                const uint64_t task_seed =
                    mixSeed(scene_seed, 0x5EEDF157ULL +
                                            static_cast<uint64_t>(attempts));
                BlueprintTask task;
                TaskGeomType geom = TaskGeomType::CLEAR;
                double yaw_err = 0.0;
                const auto t_samp = Clock::now();
                const bool sampled = task_gen_.sample(
                    out.scene, geo, analyzer_.taskTypeWeights(),
                    analyzer_.yawWeights(), task_seed, global_task_id,
                    scene_id, task, geom, yaw_err);
                timing.task_candidate_generation_ms += msSince(t_samp);
                if (!sampled) break;
                ++result.tasks_sampled;
                ++global_task_id;

                // ── cheap staged filter before preflight ────────────
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

                // ── full preflight + distribution summary ───────────
                TaskDistributionSummary summary;
                const uint64_t tick_base = task.task_id * 600000ull;
                const auto t_pre = Clock::now();
                bool early_terminated = false;
                double depth_proxy_ms = 0.0;
                const uint64_t ticks_before = total_preflight_ticks;
                const bool accepted = preflightOne(
                    task, out.scene, tick_base, summary, yaw_err,
                    total_preflight_ticks, early_terminated, depth_proxy_ms);
                timing.preflight_total_ms += msSince(t_pre);
                timing.depth_proxy_total_ms += depth_proxy_ms;
                ++timing.preflight_count;
                timing.preflight_ticks += (total_preflight_ticks - ticks_before);
                ++result.tasks_preflighted;
                ++total_preflight_attempts;
                ++rs.preflight_attempted;
                task.summary = summary;

                if (!early_terminated) ++full_preflight_attempted;
                if (accepted) {
                    ++timing.preflight_success_count;
                    ++result.preflight_success_tasks;
                    ++rs.preflight_success;
                    if (!early_terminated) ++full_preflight_success;
                    scene_pool.push_back(task);
                    global_pool.push_back(task);
                    analyzer_.addTask(summary);
                } else {
                    ++timing.preflight_failure_count;
                    ++result.preflight_failure_count;
                }
            }
            if (timing.preflight_count > 0) {
                timing.preflight_average_ms =
                    timing.preflight_total_ms /
                    static_cast<double>(timing.preflight_count);
            }
            result.total_task_candidates +=
                static_cast<uint64_t>(scene_pool.size());
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
        if (cfg_.log_rounds) {
            std::fprintf(stderr,
                         "[blueprint] round %llu: scenes=%llu/%llu "
                         "candidates=%llu cheap_rej=%llu preflight=%llu "
                         "success=%llu pool=%llu hard=%s soft=%s "
                         "elapsed_ms=%.1f\n",
                         static_cast<unsigned long long>(round),
                         static_cast<unsigned long long>(rs.scenes_valid),
                         static_cast<unsigned long long>(rs.scenes_generated),
                         static_cast<unsigned long long>(rs.task_candidates),
                         static_cast<unsigned long long>(rs.cheap_rejected),
                         static_cast<unsigned long long>(rs.preflight_attempted),
                         static_cast<unsigned long long>(rs.preflight_success),
                         static_cast<unsigned long long>(rs.selected_pool),
                         cov.hard_minimums_met ? "1" : "0",
                         cov.soft_targets_met ? "1" : "0", rs.elapsed_ms);
        }
        round_logs.push_back(rs);
        if (cov.hard_minimums_met && cov.soft_targets_met) {
            budget_exhausted = BudgetExhaustion::NONE;
            break;
        }
        if (budgetExceeded()) break;
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

}  // namespace expert
}  // namespace il_dataset
