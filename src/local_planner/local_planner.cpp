#include "il_dataset/local_planner/local_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <chrono>

namespace il_dataset {

// ─────────────────────────────────────────────────────────────────────
//  LocalPlanner implementation
// ─────────────────────────────────────────────────────────────────────

LocalPlanner::LocalPlanner(const LocalPlannerConfig& config)
    : config_(config) {}

LocalPlanner::~LocalPlanner() = default;

bool LocalPlanner::setESDF(const float* data,
                           int gx, int gy, int gz,
                           double origin_x, double origin_y, double origin_z,
                           double resolution) {
    return esdf_.setData(data, gx, gy, gz, origin_x, origin_y, origin_z, resolution);
}

bool LocalPlanner::setObservedESDF(const float* data,
                                    const uint8_t* known_mask,
                                    int gx, int gy, int gz,
                                    double origin_x, double origin_y, double origin_z,
                                    double resolution,
                                    bool unknown_is_free) {
    return esdf_.setDataWithMask(data, known_mask, gx, gy, gz,
                                  origin_x, origin_y, origin_z,
                                  resolution, unknown_is_free);
}

bool LocalPlanner::setGlobalPath(const double* path, int n_points) {
    if (path == nullptr || n_points < 2) return false;

    global_path_.clear();
    global_path_.reserve(n_points);
    for (int i = 0; i < n_points; ++i) {
        global_path_.emplace_back(path[3 * i], path[3 * i + 1], path[3 * i + 2]);
    }
    recomputeArcLengths();
    return true;
}

void LocalPlanner::reset(const VehicleState& initial_state) {
    initial_state_ = initial_state;
}

bool LocalPlanner::isReady() const {
    // The action-generating planner is conditioned only on the request's
    // complete guide and the causal observed ESDF. A global path, when set
    // by a legacy/debug caller, is diagnostic and must not gate planning.
    return esdf_.initialized();
}

// ─────────────────────────────────────────────────────────────────────
//  Arc-length precomputation
// ─────────────────────────────────────────────────────────────────────

void LocalPlanner::recomputeArcLengths() {
    arc_lengths_.clear();
    if (global_path_.empty()) return;

    arc_lengths_.reserve(global_path_.size());
    arc_lengths_.push_back(0.0);
    for (size_t i = 1; i < global_path_.size(); ++i) {
        double seg = (global_path_[i] - global_path_[i - 1]).norm();
        arc_lengths_.push_back(arc_lengths_.back() + seg);
    }
}

// ─────────────────────────────────────────────────────────────────────
//  Progress tracking
// ─────────────────────────────────────────────────────────────────────

LocalPlanner::ProgressResult
LocalPlanner::computeProgress(const Eigen::Vector3d& position,
                              double previous_progress_s) const {
    ProgressResult result;
    if (global_path_.size() < 2) return result;

    double total_length = arc_lengths_.back();
    if (total_length < 1e-6) return result;

    // Search range: allow modest backward search, mainly forward
    double search_start = std::max(0.0, previous_progress_s - 3.0);
    double search_end = std::min(total_length, previous_progress_s + 20.0);

    // If previous_progress_s is invalid (first call), search entire path
    if (previous_progress_s < 0.0) {
        search_start = 0.0;
        search_end = total_length;
    }

    double best_dist = std::numeric_limits<double>::max();
    int best_seg = -1;
    double best_t = 0.0;
    double best_s = 0.0;

    for (size_t i = 0; i + 1 < global_path_.size(); ++i) {
        double seg_start_s = arc_lengths_[i];
        double seg_end_s = arc_lengths_[i + 1];

        // Skip segments outside search range
        if (seg_end_s < search_start - 1e-6) continue;
        if (seg_start_s > search_end + 1e-6) break;

        const Eigen::Vector3d& a = global_path_[i];
        const Eigen::Vector3d& b = global_path_[i + 1];
        Eigen::Vector3d ab = b - a;
        double ab_len_sq = ab.squaredNorm();
        if (ab_len_sq < 1e-12) continue;

        double t = (position - a).dot(ab) / ab_len_sq;
        t = std::max(0.0, std::min(1.0, t));

        Eigen::Vector3d proj = a + t * ab;
        double dist = (position - proj).squaredNorm();

        if (dist < best_dist) {
            best_dist = dist;
            best_seg = static_cast<int>(i);
            best_t = t;
            best_s = seg_start_s + t * (seg_end_s - seg_start_s);
        }
    }

    // Verify progress is reasonably monotonic (prevent jumping to wrong branch)
    if (best_seg >= 0) {
        double progress_delta = best_s - previous_progress_s;
        // If this is not the first call and progress jumped backwards by a lot,
        // clamp to the forward-search minimum
        if (previous_progress_s > 1e-6 && progress_delta < -5.0) {
            // Fallback: use the nearest segment in the forward direction only
            best_dist = std::numeric_limits<double>::max();
            for (size_t i = 0; i + 1 < global_path_.size(); ++i) {
                double seg_start_s = arc_lengths_[i];
                double seg_end_s = arc_lengths_[i + 1];
                if (seg_start_s < search_start - 1e-6) continue;

                const Eigen::Vector3d& a = global_path_[i];
                const Eigen::Vector3d& b = global_path_[i + 1];
                Eigen::Vector3d ab = b - a;
                double ab_len_sq = ab.squaredNorm();
                if (ab_len_sq < 1e-12) continue;

                double t = (position - a).dot(ab) / ab_len_sq;
                t = std::max(0.0, std::min(1.0, t));
                Eigen::Vector3d proj = a + t * ab;
                double dist = (position - proj).squaredNorm();

                if (dist < best_dist) {
                    best_dist = dist;
                    best_seg = static_cast<int>(i);
                    best_t = t;
                    best_s = seg_start_s + t * (seg_end_s - seg_start_s);
                }
            }
        }
    }

    if (best_seg >= 0) {
        result.valid = true;
        result.progress_s = best_s;
        result.segment_index = best_seg;
        result.t = best_t;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  Continuous collision validation
// ─────────────────────────────────────────────────────────────────────

ValidationResult LocalPlanner::validateTrajectory(
    const std::vector<TrajectoryPoint>& trajectory) const {

    ValidationResult result;
    result.all_clear = true;
    result.min_clearance = std::numeric_limits<double>::max();
    result.worst_clearance = std::numeric_limits<double>::max();

    if (trajectory.size() < 2) {
        result.all_clear = false;
        result.any_collision = true;
        return result;
    }

    double spacing = config_.collision_check_spacing;
    // Ensure spacing <= esdf_resolution / 2
    spacing = std::min(spacing, esdf_.resolution() * 0.5);
    const double initial_clearance = esdf_.getValue(
        trajectory.front().position.x(), trajectory.front().position.y(),
        trajectory.front().position.z());
    const bool recovery_candidate =
        initial_clearance > 0.0 &&
        initial_clearance < config_.min_clearance - 1e-3;
    bool recovery_non_degrading = true;

    for (size_t i = 0; i + 1 < trajectory.size(); ++i) {
        const auto& p0 = trajectory[i];
        const auto& p1 = trajectory[i + 1];
        if (!p0.position.allFinite() || !p1.position.allFinite()) {
            result.all_clear = false;
            result.any_collision = true;
            result.min_clearance = -1.0;
            result.worst_clearance = -1.0;
            result.worst_position = p0.position;
            result.worst_time = p0.t;
            return result;
        }
        Eigen::Vector3d seg = p1.position - p0.position;
        double seg_len = seg.norm();
        // A dynamically feasible sample cannot jump this far in one control
        // interval.  Reject immediately instead of turning a runaway optimizer
        // point into millions of collision-check samples.
        const double max_sample_step = std::max(
            1.0, config_.max_velocity * config_.trajectory_dt * 4.0);
        if (!std::isfinite(seg_len) || seg_len > max_sample_step) {
            result.all_clear = false;
            result.any_collision = true;
            result.min_clearance = -1.0;
            result.worst_clearance = -1.0;
            result.worst_position = p0.position;
            result.worst_time = p0.t;
            return result;
        }
        int steps = std::max(2, static_cast<int>(seg_len / spacing) + 1);

        for (int s = 0; s <= steps; ++s) {
            double alpha = static_cast<double>(s) / steps;
            Eigen::Vector3d pt = p0.position * (1.0 - alpha) + p1.position * alpha;

            double clearance = esdf_.getValue(pt.x(), pt.y(), pt.z());

            if (recovery_candidate && clearance < initial_clearance - 1e-3) {
                recovery_non_degrading = false;
            }

            // Track the actual minimum even when it is still above the hard
            // limit.  Final-trajectory repair deliberately builds a buffer
            // above min_clearance, so it needs the location of the soft
            // minimum instead of only the first hard violation.
            if (clearance < result.min_clearance) {
                result.min_clearance = clearance;
                result.worst_clearance = clearance;
                result.worst_position = pt;
                result.worst_time =
                    p0.t + alpha * (p1.t - p0.t);
            }

            // ESDF interpolation around the threshold can differ by a few
            // floating-point ulps.  Keep a 1 mm numerical tolerance; this is
            // not a reduction of the configured physical safety margin.
            if (clearance < config_.min_clearance - 1e-3) {
                result.clearance_violation_count++;
                result.all_clear = false;
            }
            if (clearance <= 0.0) {
                result.any_collision = true;
            }
        }
    }

    // A tracking controller can enter the configured safety margin without
    // entering the physical obstacle.  Rejecting the start sample forever
    // makes every subsequent fallback fail from the identical state.  Accept
    // only a non-colliding trajectory that never reduces the initial
    // clearance and exits back into the configured safe set.
    if (recovery_candidate && !result.any_collision &&
        recovery_non_degrading) {
        const auto& end = trajectory.back().position;
        const double end_clearance = esdf_.getValue(end.x(), end.y(), end.z());
        if (end_clearance >= config_.min_clearance - 1e-3) {
            result.all_clear = true;
            result.clearance_violation_count = 0;
        }
    }

    return result;
}


// ─────────────────────────────────────────────────────────────────────
//  Phase 2: planLocalWithRequest — explicit guide/terminal, observed ESDF
// ─────────────────────────────────────────────────────────────────────

LocalPlanResult LocalPlanner::planLocalWithRequest(
    const LocalPlanningRequest& request) const {

    // Delegates to the compact production spline core.
    return planSplineWithRequest(request);
}

}  // namespace il_dataset

