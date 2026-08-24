#pragma once
/// @file   segment_labeler.hpp
/// @brief  Trajectory-internal BEHAVIOUR SEGMENT labeler for the new
///         preflight planner.
///
/// A single episode (start -> goal) is rarely one behaviour: the same
/// flight can contain straight cruising, a light/large local avoidance,
/// and a macro detour in different portions.  This class consumes the
/// per-tick ExpertStepOutput stream of the 30 Hz HierarchicalExpert and
/// splits the trajectory into CONTIGUOUS behaviour SEGMENTS, each labelled
/// with one of six classes:
///
///   straight        — direct flight, no local avoidance, no macro target
///                     correction (hierarchical_mode "direct").
///   light_avoidance — 30 Hz local avoidance active, |target bearing
///                     error| < light_avoid_threshold_deg.
///   large_avoidance — 30 Hz local avoidance active, |target bearing
///                     error| >= light_avoid_threshold_deg.
///   detour          — 5 Hz macro target correction active (NORMAL_CORRECTION
///                     / TURN_LEFT / TURN_RIGHT), for a SHORT segment
///                     (< detour_short_max_ticks).
///   medium_detour   — macro correction segment of medium duration
///                     [detour_short_max_ticks, detour_medium_max_ticks).
///   long_detour     — macro correction segment of long duration
///                     (>= detour_medium_max_ticks).
///
/// Classification priority per tick: macro correction (detour) >
/// local avoidance (light/large) > straight.  A detour segment's final
/// class is only decided when the segment ENDS (duration-dependent), so
/// segments are flushed lazily by finish().

#include <map>
#include <string>
#include <vector>

#include "il_dataset/hierarchical_expert/hierarchical_expert.hpp"  // ExpertStepOutput

namespace il_dataset {
namespace expert {

/// One contiguous behaviour segment of a trajectory.
struct BehaviorSegment {
    std::string label;       // one of the six classes above
    uint64_t start_tick = 0; // first 30 Hz tick of the segment
    uint64_t tick_count = 0; // duration in 30 Hz ticks
    // Rolling diagnostics of the segment (averaged over its ticks).
    double avg_target_bearing_error_deg = 0.0;
    double min_observed_clearance_m =
        std::numeric_limits<double>::infinity();
};

/// Consumes the per-tick ExpertStepOutput stream and produces segments.
/// Stateful per episode; call reset() before a new task.
class SegmentLabeler {
public:
    /// `detour_short_max_ticks` / `detour_medium_max_ticks` are expressed
    /// for the REAL 30 Hz rate (dt_scale = 1.0).  When the quick preflight
    /// uses a coarse dynamics step (dt_scale > 1, each tick travels
    /// dt_scale further), the same PHYSICAL detour durations need fewer
    /// ticks, so the thresholds are divided by dt_scale here (keeps the
    /// physical semantics: short < 1 s, medium 1..3 s, long >= 3 s).
    explicit SegmentLabeler(double light_avoid_threshold_deg = 30.0,
                            uint64_t detour_short_max_ticks = 30,
                            uint64_t detour_medium_max_ticks = 90,
                            double dt_scale = 1.0)
        : light_avoid_threshold_deg_(light_avoid_threshold_deg),
          detour_short_max_ticks_(
              dt_scale > 1.0
                  ? std::max<uint64_t>(
                        1, static_cast<uint64_t>(detour_short_max_ticks /
                                                 dt_scale))
                  : detour_short_max_ticks),
          detour_medium_max_ticks_(
              dt_scale > 1.0
                  ? std::max<uint64_t>(
                        1, static_cast<uint64_t>(detour_medium_max_ticks /
                                                 dt_scale))
                  : detour_medium_max_ticks) {}

    /// Clear all state (call once per new episode / task).
    void reset() {
        segments_.clear();
        counts_.clear();
        cur_label_.clear();
        cur_start_ = 0;
        cur_ticks_ = 0;
        tick_ = 0;
    }

    /// Feed one tick's expert output.  Must be called for every tick in
    /// order.
    void onTick(const ExpertStepOutput& out) {
        const std::string c = classifyTick(out);
        const double bear = std::fabs(out.target_bearing_error_deg);
        if (c == cur_label_) {
            ++cur_ticks_;
            // rolling averages
            const double n = static_cast<double>(cur_ticks_);
            cur_bearing_sum_ += bear;
            cur_min_clr_ = std::min(cur_min_clr_, out.min_observed_clearance_m);
            (void)n;
            ++tick_;
            return;
        }
        flushCurrent();
        cur_label_ = c;
        cur_start_ = tick_;
        cur_ticks_ = 1;
        cur_bearing_sum_ = bear;
        cur_min_clr_ = out.min_observed_clearance_m;
        ++tick_;
    }

    /// End the episode: flush the final segment.  Must be called once
    /// after the last onTick().
    void finish() { flushCurrent(); }

    const std::vector<BehaviorSegment>& segments() const { return segments_; }
    /// Label -> number of SEGMENTS with that label.
    const std::map<std::string, uint64_t>& labelCounts() const {
        return counts_;
    }
    uint64_t totalTicks() const { return tick_; }
    /// Number of segments bearing a label (0 when none).
    uint64_t count(const std::string& label) const {
        const auto it = counts_.find(label);
        return it == counts_.end() ? 0 : it->second;
    }

private:
    /// Per-tick base class (macro correction > local avoidance > straight).
    std::string classifyTick(const ExpertStepOutput& out) const {
        if (out.target_correction_active) {
            return "detour";
        }
        if (out.avoidance_active) {
            const double be = std::fabs(out.target_bearing_error_deg);
            return be < light_avoid_threshold_deg_ ? "light_avoidance"
                                                   : "large_avoidance";
        }
        return "straight";
    }

    /// Commit the current segment (resolving detour durations).
    void flushCurrent() {
        if (cur_ticks_ == 0) return;
        std::string label = cur_label_;
        if (label == "detour") {
            if (cur_ticks_ < detour_short_max_ticks_) {
                label = "detour";
            } else if (cur_ticks_ < detour_medium_max_ticks_) {
                label = "medium_detour";
            } else {
                label = "long_detour";
            }
        }
        BehaviorSegment s;
        s.label = label;
        s.start_tick = cur_start_;
        s.tick_count = cur_ticks_;
        if (cur_ticks_ > 0) {
            s.avg_target_bearing_error_deg =
                cur_bearing_sum_ / static_cast<double>(cur_ticks_);
        }
        s.min_observed_clearance_m = cur_min_clr_;
        segments_.push_back(s);
        ++counts_[label];
        cur_ticks_ = 0;
        cur_label_.clear();
        cur_bearing_sum_ = 0.0;
        cur_min_clr_ = std::numeric_limits<double>::infinity();
    }

    double light_avoid_threshold_deg_;
    uint64_t detour_short_max_ticks_;
    uint64_t detour_medium_max_ticks_;

    std::vector<BehaviorSegment> segments_;
    std::map<std::string, uint64_t> counts_;
    std::string cur_label_;
    uint64_t cur_start_ = 0;
    uint64_t cur_ticks_ = 0;
    double cur_bearing_sum_ = 0.0;
    double cur_min_clr_ = std::numeric_limits<double>::infinity();
    uint64_t tick_ = 0;
};

}  // namespace expert
}  // namespace il_dataset
