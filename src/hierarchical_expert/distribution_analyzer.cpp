#include "il_dataset/hierarchical_expert/distribution_analyzer.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace il_dataset {
namespace expert {

// ═══════════════════════════════════════════════════════════════════
//  DistributionAccumulator
// ═══════════════════════════════════════════════════════════════════
void DistributionAccumulator::configure(const BlueprintGenerationConfig& cfg) {
    histograms["yaw_error_abs"].configure(cfg.yaw_edges_deg);
    histograms["path_length"].configure(
        {0.0, cfg.path_short_max_m, cfg.path_long_min_m, 1e6});
    histograms["stretch"].configure({1.0, 1.1, 1.5, 1e6});
}

void DistributionAccumulator::addTask(const TaskDistributionSummary& s) {
    // tick-level counts
    counts["macro:total"] += s.macro_tick_total;
    counts["macro:pass"] += s.macro_pass_count;
    counts["macro:normal"] += s.macro_normal_count;
    counts["macro:turn_left"] += s.macro_turn_left_count;
    counts["macro:turn_right"] += s.macro_turn_right_count;
    counts["local:direct"] += s.local_direct_count;
    counts["local:avoidance"] += s.local_avoidance_count;
    // depth proxy sample counts (rays)
    counts["depth:near"] += s.depth_near_count;
    counts["depth:mid"] += s.depth_mid_count;
    counts["depth:far"] += s.depth_far_count;
    counts["depth:free"] += s.depth_free_count;
    // task-level counts
    counts["task:count"] += 1;
    counts["task:accepted"] += (s.reached_goal && !s.collision && !s.out_of_bounds)
                                   ? 1
                                   : 0;
    counts["profile:" + s.scene_profile] += 1;
    counts["geom:" + s.task_geom_type] += 1;
    if (s.initial_yaw_error_signed_deg >= 0.0) {
        counts["yaw:left"] += 1;
    } else {
        counts["yaw:right"] += 1;
    }
    if (s.path_stretch_ratio >= 1.1) {
        counts["stretch:detour"] += 1;
    } else {
        counts["stretch:direct"] += 1;
    }

    // tick-level histograms (merge)
    if (s.macro_correction_angle_hist.valid()) {
        auto& h = histograms["macro_correction_angle"];
        if (!h.valid()) h.configure(s.macro_correction_angle_hist.edges);
        for (size_t i = 0; i < h.counts.size() && i < s.macro_correction_angle_hist.counts.size();
             ++i) {
            h.counts[i] += s.macro_correction_angle_hist.counts[i];
        }
    }
    if (s.macro_correction_distance_hist.valid()) {
        auto& h = histograms["macro_correction_distance"];
        if (!h.valid()) h.configure(s.macro_correction_distance_hist.edges);
        for (size_t i = 0; i < h.counts.size() && i < s.macro_correction_distance_hist.counts.size();
             ++i) {
            h.counts[i] += s.macro_correction_distance_hist.counts[i];
        }
    }
    if (s.local_deflection_hist.valid()) {
        auto& h = histograms["local_deflection"];
        if (!h.valid()) h.configure(s.local_deflection_hist.edges);
        for (size_t i = 0; i < h.counts.size() && i < s.local_deflection_hist.counts.size();
             ++i) {
            h.counts[i] += s.local_deflection_hist.counts[i];
        }
    }
    if (s.local_yaw_rate_hist.valid()) {
        auto& h = histograms["local_yaw_rate"];
        if (!h.valid()) h.configure(s.local_yaw_rate_hist.edges);
        for (size_t i = 0; i < h.counts.size() && i < s.local_yaw_rate_hist.counts.size();
             ++i) {
            h.counts[i] += s.local_yaw_rate_hist.counts[i];
        }
    }
    if (s.local_speed_hist.valid()) {
        auto& h = histograms["local_speed"];
        if (!h.valid()) h.configure(s.local_speed_hist.edges);
        for (size_t i = 0; i < h.counts.size() && i < s.local_speed_hist.counts.size();
             ++i) {
            h.counts[i] += s.local_speed_hist.counts[i];
        }
    }
    // task-level histograms (one sample per task; edges pre-configured by
    // configure(cfg), so every addTask shares the SAME edge definition)
    auto& yaw_h = histograms["yaw_error_abs"];
    if (yaw_h.valid()) yaw_h.add(s.initial_yaw_error_abs_deg);
    auto& path_h = histograms["path_length"];
    if (path_h.valid()) path_h.add(s.preflight_path_length_m);
    auto& stretch_h = histograms["stretch"];
    if (stretch_h.valid()) stretch_h.add(s.path_stretch_ratio);
}

// ═══════════════════════════════════════════════════════════════════
//  Target metric evaluation
// ═══════════════════════════════════════════════════════════════════
namespace {

/// Task-level count for a "count:<key>" metric (0 for unknown keys).
inline double taskCountForKey(const TaskDistributionSummary& s,
                              const std::string& key) {
    if (key == "macro:total") return static_cast<double>(s.macro_tick_total);
    if (key == "macro:pass") return static_cast<double>(s.macro_pass_count);
    if (key == "macro:normal") return static_cast<double>(s.macro_normal_count);
    if (key == "macro:turn_left") return static_cast<double>(s.macro_turn_left_count);
    if (key == "macro:turn_right") return static_cast<double>(s.macro_turn_right_count);
    if (key == "local:direct") return static_cast<double>(s.local_direct_count);
    if (key == "local:avoidance") return static_cast<double>(s.local_avoidance_count);
    if (key == "depth:near") return static_cast<double>(s.depth_near_count);
    if (key == "depth:mid") return static_cast<double>(s.depth_mid_count);
    if (key == "depth:far") return static_cast<double>(s.depth_far_count);
    if (key == "depth:free") return static_cast<double>(s.depth_free_count);
    if (key == "task:count" || key == "task:accepted") return 1.0;
    if (key.rfind("profile:", 0) == 0) {
        return s.scene_profile == key.substr(8) ? 1.0 : 0.0;
    }
    if (key.rfind("geom:", 0) == 0) {
        return s.task_geom_type == key.substr(5) ? 1.0 : 0.0;
    }
    return 0.0;
}

/// Single-value metric value for a task summary (yaw_error_abs /
/// path_length / stretch), or false when the metric is a real histogram.
inline bool taskSingleValue(const TaskDistributionSummary& s,
                            const std::string& name, double& val) {
    if (name == "yaw_error_abs") {
        val = s.initial_yaw_error_abs_deg;
        return true;
    }
    if (name == "path_length") {
        val = s.preflight_path_length_m;
        return true;
    }
    if (name == "stretch") {
        val = s.path_stretch_ratio;
        return true;
    }
    return false;
}

/// Histogram from a task summary for tick-level histogram metrics.
inline const Histogram1D* taskHistogramFor(const TaskDistributionSummary& s,
                                           const std::string& name) {
    if (name == "macro_correction_angle") return &s.macro_correction_angle_hist;
    if (name == "macro_correction_distance") return &s.macro_correction_distance_hist;
    if (name == "local_deflection") return &s.local_deflection_hist;
    if (name == "local_yaw_rate") return &s.local_yaw_rate_hist;
    if (name == "local_speed") return &s.local_speed_hist;
    return nullptr;
}

}  // namespace

double distributionContribution(const TaskDistributionSummary& s,
                                const DistributionAccumulator& acc,
                                const DistributionTarget& t) {
    if (t.metric.rfind("count:", 0) == 0) {
        return taskCountForKey(s, t.metric.substr(6));
    }
    if (t.metric.rfind("hist:", 0) == 0) {
        const std::string rest = t.metric.substr(5);
        const size_t colon = rest.find(':');
        const std::string name =
            colon == std::string::npos ? rest : rest.substr(0, colon);
        if (colon == std::string::npos) {
            double single_val = 0.0;
            if (taskSingleValue(s, name, single_val)) {
                return 1.0;  // one task sample
            }
            const Histogram1D* h = taskHistogramFor(s, name);
            return h ? static_cast<double>(h->total()) : 0.0;
        }
        const int bin = std::atoi(rest.substr(colon + 1).c_str());
        double single_val = 0.0;
        if (taskSingleValue(s, name, single_val)) {
            // one sample: contributes 1 when it falls in the bin; the
            // ACCUMULATOR histogram owns the authoritative edges.
            const Histogram1D* acc_h = acc.histogram(name);
            if (!acc_h || !acc_h->valid()) return 0.0;
            if (bin < 0 || bin >= static_cast<int>(acc_h->edges.size()) - 1) {
                return 0.0;
            }
            return (single_val >= acc_h->edges[static_cast<size_t>(bin)] &&
                    single_val < acc_h->edges[static_cast<size_t>(bin + 1)])
                       ? 1.0
                       : 0.0;
        }
        const Histogram1D* h = taskHistogramFor(s, name);
        return h ? static_cast<double>(h->at(bin)) : 0.0;
    }
    // frac / balance metrics have no per-task marginal contribution
    // (evaluated on the whole accumulator only).
    return 0.0;
}

double distributionAchieved(const DistributionAccumulator& acc,
                            const DistributionTarget& t) {
    if (t.metric.rfind("count:", 0) == 0) {
        return static_cast<double>(acc.count(t.metric.substr(6)));
    }
    if (t.metric.rfind("hist:", 0) == 0) {
        const std::string rest = t.metric.substr(5);
        const size_t colon = rest.find(':');
        const std::string name = colon == std::string::npos ? rest : rest.substr(0, colon);
        const Histogram1D* h = acc.histogram(name);
        if (!h) return 0.0;
        if (colon == std::string::npos) return static_cast<double>(h->total());
        const int bin = std::atoi(rest.substr(colon + 1).c_str());
        return static_cast<double>(h->at(bin));
    }
    if (t.metric.rfind("frac:", 0) == 0) {
        const std::string rest = t.metric.substr(5);
        const size_t slash = rest.find('/');
        if (slash == std::string::npos) return 0.0;
        const double num = static_cast<double>(acc.count(rest.substr(0, slash)));
        const double den = static_cast<double>(acc.count(rest.substr(slash + 1)));
        return den > 0.0 ? num / den : 0.0;
    }
    if (t.metric.rfind("balance:", 0) == 0) {
        const std::string rest = t.metric.substr(8);
        const size_t slash = rest.find('/');
        if (slash == std::string::npos) return 1.0;
        const double a = static_cast<double>(acc.count(rest.substr(0, slash)));
        const double b = static_cast<double>(acc.count(rest.substr(slash + 1)));
        return std::abs(a - b) / (a + b + 1.0);
    }
    return 0.0;
}

// ═══════════════════════════════════════════════════════════════════
//  Deficits + coverage
// ═══════════════════════════════════════════════════════════════════
std::string DistributionDeficit::summary() const {
    std::string out = key + " achieved=" + std::to_string(achieved) +
                      " target=" + std::to_string(target) +
                      " minimum=" + std::to_string(minimum);
    if (deficit > 1e-9) out += " deficit=" + std::to_string(deficit);
    if (excess > 1e-9) out += " excess=" + std::to_string(excess);
    if (below_minimum) out += " BELOW_MINIMUM";
    return out;
}

std::vector<DistributionDeficit> computeDeficits(
    const DistributionAccumulator& acc,
    const std::vector<DistributionTarget>& targets) {
    std::vector<DistributionDeficit> out;
    out.reserve(targets.size());
    for (const auto& t : targets) {
        const double achieved = distributionAchieved(acc, t);
        DistributionDeficit d;
        d.key = t.key;
        d.achieved = achieved;
        d.target = t.target;
        d.minimum = t.minimum;
        d.maximum = t.maximum;
        d.deficit = std::max(0.0, t.target - achieved);
        d.excess = std::max(0.0, achieved - t.maximum);
        d.below_minimum = achieved < t.minimum - 1e-9;
        out.push_back(d);
    }
    return out;
}

CoverageResult evaluateCoverage(const DistributionAccumulator& acc,
                                const std::vector<DistributionTarget>& targets,
                                const BlueprintGenerationConfig& cfg) {
    CoverageResult res;
    res.deficits = computeDeficits(acc, targets);
    for (const auto& d : res.deficits) {
        if (d.below_minimum) {
            res.hard_minimums_met = false;
            res.warnings.push_back(d.summary() + " [HARD]");
        } else if (d.deficit > 1e-9) {
            res.warnings.push_back(d.summary() + " [soft]");
        }
        if (d.excess > 1e-9) {
            res.warnings.push_back(d.key + " exceeded max by " +
                                   std::to_string(d.excess));
        }
    }

    // ── Structural hard checks (left/right balance, path classes) ──
    const double tl = static_cast<double>(acc.count("macro:turn_left"));
    const double tr = static_cast<double>(acc.count("macro:turn_right"));
    const double turn_ratio = std::abs(tl - tr) / (tl + tr + 1.0);
    if (turn_ratio > cfg.max_turn_imbalance_ratio) {
        res.hard_minimums_met = false;
        res.warnings.push_back(
            "turn left/right imbalance ratio=" + std::to_string(turn_ratio) +
            " > " + std::to_string(cfg.max_turn_imbalance_ratio) + " [HARD]");
    }
    const double yl = static_cast<double>(acc.count("yaw:left"));
    const double yr = static_cast<double>(acc.count("yaw:right"));
    const double yaw_ratio = std::abs(yl - yr) / (yl + yr + 1.0);
    if (yaw_ratio > cfg.max_yaw_imbalance_ratio) {
        res.hard_minimums_met = false;
        res.warnings.push_back(
            "yaw left/right imbalance ratio=" + std::to_string(yaw_ratio) +
            " > " + std::to_string(cfg.max_yaw_imbalance_ratio) + " [HARD]");
    }
    // short/medium/long path classes (path_length histogram, 3 bins)
    const Histogram1D* path_h = acc.histogram("path_length");
    if (path_h && path_h->valid()) {
        for (int i = 0; i < 3; ++i) {
            if (path_h->at(i) < static_cast<uint64_t>(cfg.min_path_samples_per_class)) {
                res.hard_minimums_met = false;
                res.warnings.push_back(
                    std::string("path class bin ") + std::to_string(i) +
                    "=" + std::to_string(path_h->at(i)) + " < " +
                    std::to_string(cfg.min_path_samples_per_class) +
                    " [HARD]");
            }
        }
    }

    // ── Grouped local-deflection coverage (avoid the degenerate case
    // where bin-level soft targets are met but ALL samples pile into one
    // side; each coarse behavioural group must have a hard minimum). ──
    // deflection_edges: [-90,-60,-30,-10,10,30,60,90] -> 7 bins:
    //   bin0 strong_right, bin1 right, bin2 slight_right, bin3 near_direct,
    //   bin4 slight_left, bin5 left, bin6 strong_left
    const Histogram1D* def_h = acc.histogram("local_deflection");
    if (def_h && def_h->valid() && def_h->edges.size() >= 8) {
        auto group = [&](int lo, int hi) {
            uint64_t total = 0;
            for (int i = lo; i <= hi; ++i) {
                total += def_h->at(i);
            }
            return total;
        };
        struct { const char* name; int lo; int hi; uint64_t min; } def_groups[] = {
            {"deflection_group:strong_right", 0, 1,
             static_cast<uint64_t>(cfg.min_grouped_deflection_samples)},
            {"deflection_group:right", 2, 2,
             static_cast<uint64_t>(cfg.min_grouped_deflection_samples)},
            {"deflection_group:near_direct", 3, 3,
             static_cast<uint64_t>(cfg.min_grouped_deflection_samples)},
            {"deflection_group:left", 4, 4,
             static_cast<uint64_t>(cfg.min_grouped_deflection_samples)},
            {"deflection_group:strong_left", 5, 6,
             static_cast<uint64_t>(cfg.min_grouped_deflection_samples)},
        };
        for (const auto& g : def_groups) {
            const uint64_t got = group(g.lo, g.hi);
            if (got < g.min) {
                res.hard_minimums_met = false;
                res.warnings.push_back(
                    std::string(g.name) + "=" + std::to_string(got) +
                    " < " + std::to_string(g.min) + " [HARD]");
            }
        }
    }

    // ── Grouped macro-correction coverage (the same anti-degeneracy
    // requirement for the 5 Hz steering-correction distribution). ──
    // correction_angle_edges: [-90,-60,-45,-30,-15,0,15,30,45,60,90] ->
    // 10 bins; groups: right=bins0-4, near=bins4-5, left=bins5-9.
    const Histogram1D* corr_h = acc.histogram("macro_correction_angle");
    if (corr_h && corr_h->valid()) {
        auto group = [&](int lo, int hi) {
            uint64_t total = 0;
            for (int i = lo; i <= hi; ++i) {
                total += corr_h->at(i);
            }
            return total;
        };
        struct { const char* name; int lo; int hi; uint64_t min; } corr_groups[] = {
            {"correction_group:right", 0, 4,
             static_cast<uint64_t>(cfg.min_grouped_correction_samples)},
            {"correction_group:near", 4, 5,
             static_cast<uint64_t>(cfg.min_grouped_correction_samples)},
            {"correction_group:left", 5, 9,
             static_cast<uint64_t>(cfg.min_grouped_correction_samples)},
        };
        for (const auto& g : corr_groups) {
            const uint64_t got = group(g.lo, g.hi);
            if (got < g.min) {
                res.hard_minimums_met = false;
                res.warnings.push_back(
                    std::string(g.name) + "=" + std::to_string(got) +
                    " < " + std::to_string(g.min) + " [HARD]");
            }
        }
    }
    res.soft_targets_met =
        res.hard_minimums_met &&
        std::none_of(res.warnings.begin(), res.warnings.end(),
                     [](const std::string& w) {
                         return w.find("[soft]") != std::string::npos;
                     });
    return res;
}

double scoreTask(const TaskDistributionSummary& s,
                 const DistributionAccumulator& acc,
                 const std::vector<DistributionTarget>& targets) {
    double score = 0.0;
    for (const auto& t : targets) {
        const double c = distributionContribution(s, acc, t);
        if (c <= 0.0) continue;
        const double cur = distributionAchieved(acc, t);
        const double after = cur + c;
        const double ideal = t.target * (1.0 + t.tolerance);
        double gain = 0.0;
        if (after <= ideal + 1e-9) {
            gain = c;
        } else if (cur < ideal) {
            gain = std::max(0.0, ideal - cur);
        }
        score += t.weight * gain;
        if (after > t.maximum) {
            score -= t.weight * (after - t.maximum);
        }
    }
    return score;
}

// ═══════════════════════════════════════════════════════════════════
//  DistributionAnalyzer
// ═══════════════════════════════════════════════════════════════════
void DistributionAnalyzer::buildDefaultTargets() {
    cfg_.targets.clear();
    const double big = 1e18;

    auto addCount = [&](const std::string& key, double target, double minimum,
                        double weight, bool hard) {
        DistributionTarget t;
        t.key = key;
        t.metric = "count:" + key;
        t.target = target;
        t.minimum = minimum;
        t.maximum = big;
        t.weight = weight;
        t.minimum_hard = hard;
        cfg_.targets.push_back(t);
    };
    auto addHistTotal = [&](const std::string& name, const std::string& key,
                            double target, double minimum, double weight,
                            bool hard) {
        DistributionTarget t;
        t.key = key;
        t.metric = "hist:" + name;
        t.target = target;
        t.minimum = minimum;
        t.maximum = big;
        t.weight = weight;
        t.minimum_hard = hard;
        cfg_.targets.push_back(t);
    };
    auto addHistBin = [&](const std::string& name, const std::string& key,
                          int bin, double target, double minimum, double weight,
                          bool hard) {
        DistributionTarget t;
        t.key = key;
        t.metric = "hist:" + name + ":" + std::to_string(bin);
        t.bin_index = bin;
        t.target = target;
        t.minimum = minimum;
        t.maximum = big;
        t.weight = weight;
        t.minimum_hard = hard;
        cfg_.targets.push_back(t);
    };
    auto addBalance = [&](const std::string& key, const std::string& a,
                          const std::string& b, double target,
                          double minimum, double weight) {
        DistributionTarget t;
        t.key = key;
        t.metric = "balance:" + a + "/" + b;
        t.target = target;
        t.minimum = minimum;
        t.maximum = big;
        t.weight = weight;
        t.minimum_hard = false;  // verified structurally in evaluateCoverage
        cfg_.targets.push_back(t);
    };

    const int mtc = std::max(1, cfg_.min_macro_ticks_per_class);
    const int mds = std::max(1, cfg_.min_depth_samples_per_band);
    const int mys = std::max(1, cfg_.min_yaw_samples_per_bin);
    const int mps = std::max(1, cfg_.min_path_samples_per_class);

    // ── 5 Hz tick-level macro coverage (priority 2) ────────────────
    addCount("macro:total", 4.0 * mtc, mtc, 0.5, false);
    addCount("macro:pass", 3.0 * mtc, mtc, 1.0, true);
    addCount("macro:normal", 3.0 * mtc, mtc, 1.0, true);
    addCount("macro:turn_left", 3.0 * mtc, mtc, 2.0, true);
    addCount("macro:turn_right", 3.0 * mtc, mtc, 2.0, true);
    // correction-angle coverage (soft per-bin, hard total)
    addHistTotal("macro_correction_angle", "corr_angle_total", 4.0 * mtc, mtc,
                 1.0, true);
    const int n_corr_bins = static_cast<int>(cfg_.correction_angle_edges_deg.size()) - 1;
    for (int i = 0; i < n_corr_bins; ++i) {
        addHistBin("macro_correction_angle",
                   "corr_angle:bin" + std::to_string(i), i, 2.0, 0.0, 0.8, false);
    }
    const int n_corr_dist = static_cast<int>(cfg_.correction_distance_edges.size()) - 1;
    for (int i = 0; i < n_corr_dist; ++i) {
        addHistBin("macro_correction_distance",
                   "corr_dist:bin" + std::to_string(i), i, 2.0, 0.0, 0.6, false);
    }

    // ── 30 Hz avoidance coverage (priority 3) ──────────────────────
    addCount("local:avoidance", 3.0 * mtc, mtc, 1.2, true);
    addCount("local:direct", 3.0 * mtc, mtc, 0.4, false);
    addHistTotal("local_deflection", "deflection_total", 4.0 * mtc, mtc, 1.0,
                 true);
    // 7 symmetric bins: strong_right..strong_left (deflection_edges)
    const int n_def = static_cast<int>(cfg_.deflection_edges_deg.size()) - 1;
    for (int i = 0; i < n_def; ++i) {
        const bool strong = (i == 0 || i == n_def - 1);
        const bool medium = (i == 1 || i == n_def - 2);
        const double target = strong ? 6.0 : (medium ? 5.0 : 4.0);
        addHistBin("local_deflection",
                   std::string("deflection:bin") + std::to_string(i), i,
                   target, 0.0, strong ? 1.4 : 1.0, false);
    }

    // ── depth proxy coverage (priority 4) ──────────────────────────
    addCount("depth:near", 3.0 * mds, mds, 1.0, true);
    addCount("depth:mid", 3.0 * mds, mds, 1.0, true);
    addCount("depth:far", 3.0 * mds, mds, 1.2, true);
    addCount("depth:free", 3.0 * mds, mds, 1.0, true);

    // ── initial yaw coverage (priority 5, hard per bin) ────────────
    const int n_yaw = static_cast<int>(cfg_.yaw_edges_deg.size()) - 1;
    for (int i = 0; i < n_yaw; ++i) {
        addHistBin("yaw_error_abs", "yaw:bin" + std::to_string(i), i,
                   2.0 * mys, mys, 1.0, true);
    }

    // ── path length coverage (priority 6, hard per class) ──────────
    addHistBin("path_length", "path:short", 0, 2.0 * mps, mps, 0.8, true);
    addHistBin("path_length", "path:medium", 1, 2.0 * mps, mps, 0.8, true);
    addHistBin("path_length", "path:long", 2, 2.0 * mps, mps, 1.2, true);

    // ── stretch (soft) ─────────────────────────────────────────────
    addCount("stretch:detour", std::max(4, mps * 2), 0, 0.5, false);

    // ── balances (verified structurally; soft entries) ─────────────
    addBalance("balance:turn", "macro:turn_left", "macro:turn_right",
               cfg_.max_turn_imbalance_ratio, cfg_.max_turn_imbalance_ratio,
               2.0);
    addBalance("balance:yaw", "yaw:left", "yaw:right",
               cfg_.max_yaw_imbalance_ratio, cfg_.max_yaw_imbalance_ratio,
               1.0);
}

DistributionAnalyzer::DistributionAnalyzer(
    const BlueprintGenerationConfig& cfg)
    : cfg_(cfg) {
    if (cfg_.targets.empty()) buildDefaultTargets();
    reset();
}

void DistributionAnalyzer::reset() {
    acc_.clear();
    acc_.configure(cfg_);
    deficits_.clear();
    coverage_ = CoverageResult{};
}

void DistributionAnalyzer::addTask(const TaskDistributionSummary& s) {
    acc_.addTask(s);
}

void DistributionAnalyzer::recompute() {
    deficits_ = computeDeficits(acc_, cfg_.targets);
    coverage_ = evaluateCoverage(acc_, cfg_.targets, cfg_);
}

std::vector<std::string> DistributionAnalyzer::deficitStrings() const {
    std::vector<std::string> out;
    out.reserve(deficits_.size());
    for (const auto& d : deficits_) {
        if (d.deficit > 1e-9 || d.excess > 1e-9) {
            out.push_back(d.summary());
        }
    }
    return out;
}

namespace {
const DistributionDeficit* findDeficit(
    const std::vector<DistributionDeficit>& deficits,
    const std::string& key) {
    for (const auto& d : deficits) {
        if (d.key == key) return &d;
    }
    return nullptr;
}
bool deficitBelow(const std::vector<DistributionDeficit>& deficits,
                  const std::string& key, double threshold = 1e-9) {
    const DistributionDeficit* d = findDeficit(deficits, key);
    return d && d->deficit > threshold;
}
}  // namespace

std::map<std::string, double> DistributionAnalyzer::profileTagWeights() const {
    std::map<std::string, double> w;
    auto bump = [&](const std::string& tag, double mult) {
        auto it = w.find(tag);
        w[tag] = it == w.end() ? mult : it->second * mult;
    };
    const bool turn_short =
        deficitBelow(deficits_, "macro:turn_left") ||
        deficitBelow(deficits_, "macro:turn_right") ||
        deficitBelow(deficits_, "yaw:bin4") ||
        deficitBelow(deficits_, "yaw:bin5");
    const bool far_short =
        deficitBelow(deficits_, "depth:far") ||
        deficitBelow(deficits_, "depth:free");
    const bool near_short =
        deficitBelow(deficits_, "depth:near") ||
        deficitBelow(deficits_, "depth:mid");
    const bool avoid_short =
        deficitBelow(deficits_, "local:avoidance") ||
        deficitBelow(deficits_, "deflection:bin0") ||
        deficitBelow(deficits_, "deflection:bin6");
    const bool long_short = deficitBelow(deficits_, "path:long");

    if (turn_short) {
        bump("blocker", 1.8);
        bump("large", 1.6);
        bump("chicane", 1.5);
        bump("narrow", 1.3);
    }
    if (far_short) {
        bump("sparse", 1.8);
        bump("clear", 1.5);
    }
    if (near_short) {
        bump("dense", 1.8);
        bump("tiny", 1.5);
    }
    if (avoid_short) {
        bump("dense", 1.4);
        bump("narrow", 1.5);
        bump("corridor", 1.4);
        bump("chicane", 1.3);
    }
    if (long_short) {
        bump("large", 1.4);
        bump("blocker", 1.3);
        bump("sparse", 1.2);
    }
    return w;
}

std::vector<double> DistributionAnalyzer::taskTypeWeights() const {
    std::vector<double> w(static_cast<size_t>(kNumTaskGeomTypes), 1.0);
    const bool turn_short =
        deficitBelow(deficits_, "macro:turn_left") ||
        deficitBelow(deficits_, "macro:turn_right") ||
        deficitBelow(deficits_, "yaw:bin4") ||
        deficitBelow(deficits_, "yaw:bin5");
    const bool avoid_short =
        deficitBelow(deficits_, "local:avoidance") ||
        deficitBelow(deficits_, "deflection:bin0") ||
        deficitBelow(deficits_, "deflection:bin6");
    const bool long_short = deficitBelow(deficits_, "path:long");
    const bool far_short =
        deficitBelow(deficits_, "depth:far") ||
        deficitBelow(deficits_, "depth:free");

    if (turn_short) {
        w[static_cast<size_t>(TaskGeomType::LARGE_OCCLUSION)] *= 2.0;
        w[static_cast<size_t>(TaskGeomType::CHICANE)] *= 1.8;
        w[static_cast<size_t>(TaskGeomType::LONG_DETOUR)] *= 1.6;
        w[static_cast<size_t>(TaskGeomType::OFFSET_AVOIDANCE)] *= 1.3;
    }
    if (avoid_short) {
        w[static_cast<size_t>(TaskGeomType::LOCAL_AVOIDANCE)] *= 1.8;
        w[static_cast<size_t>(TaskGeomType::OFFSET_AVOIDANCE)] *= 1.6;
        w[static_cast<size_t>(TaskGeomType::NARROW_BUT_PLANNABLE)] *= 1.5;
        w[static_cast<size_t>(TaskGeomType::MULTI_OBSTACLE)] *= 1.3;
    }
    if (long_short) {
        w[static_cast<size_t>(TaskGeomType::LONG_DETOUR)] *= 2.0;
    }
    if (far_short) {
        w[static_cast<size_t>(TaskGeomType::CLEAR)] *= 1.4;
    }
    return w;
}

std::vector<double> DistributionAnalyzer::yawWeights() const {
    std::vector<double> w = cfg_.yaw_weights;
    const int n = static_cast<int>(w.size());
    for (int i = 0; i < n; ++i) {
        if (deficitBelow(deficits_, "yaw:bin" + std::to_string(i))) {
            w[static_cast<size_t>(i)] *= 1.6;
        }
    }
    if (deficitBelow(deficits_, "macro:turn_left") ||
        deficitBelow(deficits_, "macro:turn_right")) {
        if (n > 4) w[4] *= 1.6;
        if (n > 5) w[5] *= 1.6;
    }
    return w;
}

bool DistributionAnalyzer::longPathDeficit() const {
    return deficitBelow(deficits_, "path:long");
}

std::vector<BlueprintTask> DistributionAnalyzer::select(
    const std::vector<BlueprintTask>& candidates,
    std::vector<uint64_t>& per_scene_accepted) const {
    per_scene_accepted.clear();
    const auto& targets = cfg_.targets;

    std::map<uint64_t, std::vector<const BlueprintTask*>> by_scene;
    uint64_t max_scene_id = 0;
    for (const auto& c : candidates) {
        by_scene[c.scene_id].push_back(&c);
        max_scene_id = std::max(max_scene_id, c.scene_id);
    }

    DistributionAccumulator acc;
    acc.configure(cfg_);  // same histogram edges as the analyzer's accumulator
    std::map<uint64_t, uint64_t> scene_count;
    std::map<uint64_t, bool> scene_opened;
    std::set<uint64_t> used_task_ids;

    const int per_scene_cap = std::max(1, cfg_.max_tasks_per_scene);
    const size_t pool_size = candidates.size();
    const int max_iter = static_cast<int>(pool_size * 2 + 8);
    std::vector<BlueprintTask> selected;
    selected.reserve(pool_size);

    // Balance-aware marginal bonus: a task that feeds the MINORITY side of
    // the turn / yaw balance shrinks the imbalance and is rewarded; one
    // feeding the MAJORITY side is penalised.  Pure delta-of-imbalance, so
    // it is zero while both sides are empty and never distorts the start.
    const double turn_balance_weight = 1.2;
    const double yaw_balance_weight = 1.0;
    auto balanceDelta = [](double a, double b, double ca, double cb) {
        const double cur = std::abs(a - b) / (a + b + 1.0);
        const double an = a + ca, bn = b + cb;
        const double after = std::abs(an - bn) / (an + bn + 1.0);
        return after - cur;  // <0 means the task improved the balance
    };

    auto balanceAdjustedScore = [&](const BlueprintTask& c) {
        double s = scoreTask(c.summary, acc, targets);
        const auto& sm = c.summary;
        // turn side
        {
            const double tl = static_cast<double>(acc.count("macro:turn_left"));
            const double tr = static_cast<double>(acc.count("macro:turn_right"));
            const double ca = sm.macro_turn_left_count > 0 ? 1.0 : 0.0;
            const double cb = sm.macro_turn_right_count > 0 ? 1.0 : 0.0;
            const double delta = balanceDelta(tl, tr, ca, cb);
            s -= turn_balance_weight * delta;
            const double an = tl + ca, bn = tr + cb;
            const double after = std::abs(an - bn) / (an + bn + 1.0);
            if (after > cfg_.max_turn_imbalance_ratio) {
                s -= turn_balance_weight *
                     (after - cfg_.max_turn_imbalance_ratio) * 4.0;
            }
        }
        // yaw side (initial yaw error sign)
        {
            const double yl = static_cast<double>(acc.count("yaw:left"));
            const double yr = static_cast<double>(acc.count("yaw:right"));
            const double ca = sm.initial_yaw_error_signed_deg >= 0.0 ? 1.0 : 0.0;
            const double cb = sm.initial_yaw_error_signed_deg < 0.0 ? 1.0 : 0.0;
            const double delta = balanceDelta(yl, yr, ca, cb);
            s -= yaw_balance_weight * delta;
            const double an = yl + ca, bn = yr + cb;
            const double after = std::abs(an - bn) / (an + bn + 1.0);
            if (after > cfg_.max_yaw_imbalance_ratio) {
                s -= yaw_balance_weight *
                     (after - cfg_.max_yaw_imbalance_ratio) * 4.0;
            }
        }
        return s;
    };

    for (int iter = 0; iter < max_iter && selected.size() < pool_size; ++iter) {
        const BlueprintTask* best = nullptr;
        double best_score = -1e18;
        for (const auto& kv : by_scene) {
            const uint64_t sid = kv.first;
            if (scene_count[sid] >= static_cast<uint64_t>(per_scene_cap)) {
                continue;
            }
            const bool opened = scene_opened[sid];
            for (const BlueprintTask* c : kv.second) {
                if (used_task_ids.count(c->task_id)) continue;
                double score = balanceAdjustedScore(*c);
                if (!opened) score -= cfg_.scene_switch_penalty;
                if (score > best_score) {
                    best_score = score;
                    best = c;
                }
            }
        }
        if (best == nullptr) break;
        if (best_score <= 0.0) {
            // No positive marginal contribution left.  Stop only when the
            // hard minimums are already covered; otherwise keep accepting
            // the LEAST-BAD candidate (bounded by max_iter / pool size) so
            // the selector genuinely attempts hard coverage from the pool.
            const CoverageResult cov = evaluateCoverage(acc, cfg_.targets, cfg_);
            if (cov.hard_minimums_met) break;
        }

        selected.push_back(*best);
        selected.back().selection_score = best_score;
        used_task_ids.insert(best->task_id);
        scene_count[best->scene_id] += 1;
        scene_opened[best->scene_id] = true;
        acc.addTask(best->summary);
    }

    // ── Scene consolidation ────────────────────────────────────────
    // After greedy coverage, drop entire scenes whose tasks are NOT needed
    // to satisfy the hard minimums.  Greedily try to remove scenes in
    // ascending order of total marginal value (least valuable first) and
    // keep a removal whenever the hard coverage still holds afterwards.
    {
        // Group the currently-selected tasks by scene.
        std::map<uint64_t, std::vector<size_t>> sel_by_scene;
        for (size_t i = 0; i < selected.size(); ++i) {
            sel_by_scene[selected[i].scene_id].push_back(i);
        }
        std::vector<uint64_t> scene_order;
        for (const auto& kv : sel_by_scene) scene_order.push_back(kv.first);
        std::sort(scene_order.begin(), scene_order.end(),
                  [&](uint64_t a, uint64_t b) {
                      const auto& va = sel_by_scene[a];
                      const auto& vb = sel_by_scene[b];
                      double sa = 0.0, sb = 0.0;
                      for (size_t i : va) sa += selected[i].selection_score;
                      for (size_t i : vb) sb += selected[i].selection_score;
                      if (sa != sb) return sa < sb;
                      return a < b;
                  });

        std::vector<bool> removed(selected.size(), false);
        for (const uint64_t sid : scene_order) {
            // Rebuild the accumulator from every kept task except this
            // scene's; if the hard minimums still hold, drop the scene.
            DistributionAccumulator trial;
            trial.configure(cfg_);
            for (size_t i = 0; i < selected.size(); ++i) {
                if (removed[i]) continue;
                if (selected[i].scene_id == sid) continue;
                trial.addTask(selected[i].summary);
            }
            const CoverageResult cov = evaluateCoverage(trial, cfg_.targets, cfg_);
            if (cov.hard_minimums_met) {
                for (const size_t i : sel_by_scene[sid]) removed[i] = true;
            }
        }
        // Rebuild the selected list without the dropped scenes.
        std::vector<BlueprintTask> consolidated;
        consolidated.reserve(selected.size());
        for (size_t i = 0; i < selected.size(); ++i) {
            if (!removed[i]) consolidated.push_back(selected[i]);
        }
        selected.swap(consolidated);
    }

    per_scene_accepted.assign(static_cast<size_t>(max_scene_id) + 1, 0);
    scene_count.clear();
    for (const auto& t : selected) {
        per_scene_accepted[t.scene_id] += 1;
        scene_count[t.scene_id] += 1;
    }
    return selected;
}

}  // namespace expert
}  // namespace il_dataset
