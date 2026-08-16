#pragma once
/// @file   distribution_analyzer.hpp
/// @brief  The quota replacement: distribution targets, a global
///         accumulator over task summaries, deficit computation and the
///         deterministic greedy final selector.
///
/// A candidate task's value is approximated by:
///   score = Σ_targets weight · (contribution to bins still below target)
///           − Σ_targets weight · (contribution beyond the soft maximum)
/// plus a scene_switch_penalty when accepting a task from a NEW scene
/// (so the final blueprint prefers fewer scenes with more complementary
/// tasks per scene).  Soft targets that stay below the ideal only emit
/// warnings; only hard minimums can fail generation_ok.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

// ═══════════════════════════════════════════════════════════════════
//  DistributionAccumulator — merges every preflighted task summary
// ═══════════════════════════════════════════════════════════════════
struct DistributionAccumulator {
    // Tick-level / count keys:
    //   "macro:total", "macro:pass", "macro:normal",
    //   "macro:turn_left", "macro:turn_right",
    //   "local:direct", "local:avoidance",
    //   "depth:near", "depth:mid", "depth:far", "depth:free",
    //   "task:count", "profile:<name>", "geom:<type>"
    std::map<std::string, uint64_t> counts;
    // Histogram keys:
    //   "macro_correction_angle", "macro_correction_distance",
    //   "local_deflection", "local_yaw_rate", "local_speed",
    //   "yaw_error_abs", "path_length", "stretch"
    std::map<std::string, Histogram1D> histograms;

    /// Pre-create the task-level histograms (yaw_error_abs / path_length /
    /// stretch) with the edges configured by the analyzer, so every
    /// addTask() and every metric evaluation shares ONE edge definition.
    void configure(const BlueprintGenerationConfig& cfg);

    void clear() {
        counts.clear();
        histograms.clear();
    }
    uint64_t count(const std::string& key) const {
        const auto it = counts.find(key);
        return it == counts.end() ? 0 : it->second;
    }
    Histogram1D* histogram(const std::string& name) {
        auto it = histograms.find(name);
        return it == histograms.end() ? nullptr : &it->second;
    }
    const Histogram1D* histogram(const std::string& name) const {
        auto it = histograms.find(name);
        return it == histograms.end() ? nullptr : &it->second;
    }
    /// Merge one preflighted task summary (all tick counts, histograms and
    /// one task-level sample for yaw error / path length / stretch).
    void addTask(const TaskDistributionSummary& s);
};

// ═══════════════════════════════════════════════════════════════════
//  Target evaluation helpers
// ═══════════════════════════════════════════════════════════════════
/// Current achieved value of a target metric in the accumulator.
double distributionAchieved(const DistributionAccumulator& acc,
                            const DistributionTarget& t);

/// Marginal contribution of one task summary to a target metric.  The
/// accumulator supplies the authoritative histogram edges (so single-value
/// metrics like yaw_error_abs / path_length share ONE edge definition).
double distributionContribution(const TaskDistributionSummary& s,
                                const DistributionAccumulator& acc,
                                const DistributionTarget& t);

// ═══════════════════════════════════════════════════════════════════
//  DistributionDeficit + deficit computation
// ═══════════════════════════════════════════════════════════════════
struct DistributionDeficit {
    std::string key;
    double achieved = 0.0;
    double target = 0.0;
    double minimum = 0.0;
    double maximum = 1e18;
    double deficit = 0.0;  // >0 : below target
    double excess = 0.0;   // >0 : above maximum
    bool below_minimum = false;

    std::string summary() const;
};

/// Compute per-target deficits from the accumulator.
std::vector<DistributionDeficit> computeDeficits(
    const DistributionAccumulator& acc,
    const std::vector<DistributionTarget>& targets);

/// Hard-minimum coverage (generation_ok gate) and soft coverage flags.
struct CoverageResult {
    bool hard_minimums_met = true;
    bool soft_targets_met = true;
    std::vector<std::string> warnings;
    std::vector<DistributionDeficit> deficits;
};

/// Evaluate hard/soft coverage; emits warnings for unmet soft targets.
CoverageResult evaluateCoverage(
    const DistributionAccumulator& acc,
    const std::vector<DistributionTarget>& targets,
    const BlueprintGenerationConfig& cfg);

/// Greedy score of one candidate given the current accumulator.
double scoreTask(const TaskDistributionSummary& s,
                 const DistributionAccumulator& acc,
                 const std::vector<DistributionTarget>& targets);

// ═══════════════════════════════════════════════════════════════════
//  DistributionAnalyzer — owns accumulator + targets + deficit-driven
//  sampling weights + the final greedy selector
// ═══════════════════════════════════════════════════════════════════
class DistributionAnalyzer {
public:
    explicit DistributionAnalyzer(const BlueprintGenerationConfig& cfg);

    void reset();
    void addTask(const TaskDistributionSummary& s);
    void recompute();

    const DistributionAccumulator& accumulator() const { return acc_; }
    const std::vector<DistributionDeficit>& deficits() const {
        return deficits_;
    }
    const std::vector<DistributionTarget>& targets() const {
        return cfg_.targets;
    }
    std::vector<std::string> deficitStrings() const;
    const CoverageResult& coverage() const { return coverage_; }

    /// Deficit-driven profile tag multipliers (empty / sparse / dense /
    /// tiny / large / blocker / clustered / corridor / narrow ...).
    std::map<std::string, double> profileTagWeights() const;
    /// Deficit-driven task-geometry type weights (length == 8).
    std::vector<double> taskTypeWeights() const;
    /// Deficit-driven initial-yaw stratum weights (length == edges-1).
    std::vector<double> yawWeights() const;
    /// True when the long-path class is deficient (controller widens the
    /// goal-distance band to produce LONG_DETOUR candidates).
    bool longPathDeficit() const;

    /// Deterministic greedy final selection over the full preflighted
    /// candidate pool.  Fills per_scene_accepted (indexed by scene_id).
    std::vector<BlueprintTask> select(
        const std::vector<BlueprintTask>& candidates,
        std::vector<uint64_t>& per_scene_accepted) const;

private:
    void buildDefaultTargets();

    BlueprintGenerationConfig cfg_;
    DistributionAccumulator acc_;
    std::vector<DistributionDeficit> deficits_;
    CoverageResult coverage_;
};

}  // namespace expert
}  // namespace il_dataset
