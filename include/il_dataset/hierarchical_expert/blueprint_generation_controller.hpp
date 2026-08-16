#pragma once
/// @file   blueprint_generation_controller.hpp
/// @brief  Deficit-driven, budgeted, deterministic Blueprint generation.
///
/// Pipeline (replaces the fixed "strata schedule + hard quota" path):
///   round 1..max_generation_rounds:
///     weighted SceneProfile pick (deficit weights)
///       -> scene realization (SceneProfileGenerator)
///       -> one-time geometry cache (SceneGeometryCache)
///       -> fast task-candidate sampling (TaskCandidateGenerator)
///       -> cheap filter (bounds/clearance/distance/component)
///       -> C++ preflight with the SAME expert (PreflightSimulator)
///          + TaskDistributionSummary (path / depth proxy / 5 Hz ticks /
///                                     30 Hz deflection / yaw / stretch)
///     -> global DistributionAnalyzer recomputes deficits
///   final greedy selection (DistributionAnalyzer::select) with a
///   scene_switch_penalty (fewer scenes, more complementary tasks each)
///   -> BlueprintResult (manifest-ready, backward compatible).
///
/// All randomness derives from base_seed / scene_seed / task_seed; every
/// loop has a strict budget; the result is deterministic for a fixed
/// config + seed.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/distribution_analyzer.hpp"
#include "il_dataset/hierarchical_expert/depth_proxy_evaluator.hpp"
#include "il_dataset/hierarchical_expert/scene_geometry_cache.hpp"
#include "il_dataset/hierarchical_expert/scene_profile_generator.hpp"
#include "il_dataset/hierarchical_expert/task_candidate_generator.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

class BlueprintGenerationController {
public:
    BlueprintGenerationController(const Params2D& params,
                                  const BlueprintGenerationConfig& cfg);

    BlueprintResult generate();

private:
    /// Cheap staged filter: bounds / clearance / distance / component /
    /// zero-length.  Runs BEFORE any preflight; rejects count as
    /// cheap_filter_rejected.
    bool cheapFilterPass(const BlueprintScene& scene,
                         const SceneGeometryCache& geo,
                         const BlueprintTask& task) const;

    /// Full closed-loop preflight of ONE task with the SAME expert, plus
    /// the TaskDistributionSummary.  `yaw_error_signed_deg` is the signed
    /// goal-bearing error sampled with the task.  `task_tick_budget` is the
    /// EFFECTIVE per-task tick budget (already capped by the remaining
    /// global tick budget; 0 must never reach here).  Returns the audit
    /// `accepted` flag.  Out-params: `total_preflight_ticks` accumulates
    /// the ticks consumed (success + failure), `early_terminated` is set
    /// when the no-progress / stall detector cut the episode short,
    /// `global_tick_truncated` is set when the task was cut by the GLOBAL
    /// remaining-tick cap (not a normal task timeout), `reject_reason`
    /// receives one of: accepted / collision / timeout / no_progress /
    /// stall / out_of_bounds / macro_label / goal_not_reached, and
    /// `depth_proxy_ms` accumulates the DepthProxyEvaluator wall time.
    bool preflightOne(BlueprintTask& task, const BlueprintScene& scene,
                      uint64_t tick_base, TaskDistributionSummary& summary,
                      double yaw_error_signed_deg, uint64_t task_tick_budget,
                      uint64_t& total_preflight_ticks, bool& early_terminated,
                      bool& global_tick_truncated, std::string& reject_reason,
                      double& depth_proxy_ms) const;

    /// Legacy 3x3 density x radius strata coverage (manifest compat).
    void updateLegacyStrata(BlueprintResult& result,
                            const BlueprintScene& scene) const;
    /// Legacy category_counts (manifest compat) from selected tasks.
    void fillCategoryCounts(BlueprintResult& result) const;
    /// Build the distribution report (counts + histograms) for the result.
    void fillDistributionReport(BlueprintResult& result,
                                const DistributionAccumulator& acc) const;

    Params2D p_;
    BlueprintGenerationConfig cfg_;
    SceneProfileGenerator profile_gen_;
    DistributionAnalyzer analyzer_;
    TaskCandidateGenerator task_gen_;
};

}  // namespace expert
}  // namespace il_dataset
