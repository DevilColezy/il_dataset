#include "il_dataset/local_planner/local_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <functional>
#include <utility>

#ifdef IL_DATASET_HAS_NLOPT
#include <nlopt.h>
#endif

namespace il_dataset {

// ─────────────────────────────────────────────────────────────────────
//  LocalPlannerConfig helpers
// ─────────────────────────────────────────────────────────────────────

namespace {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kTwoPi = 2.0 * kPi;
}  // anonymous

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
    return esdf_.initialized() && global_path_.size() >= 2;
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
//  Local goal selection
// ─────────────────────────────────────────────────────────────────────

LocalPlanner::LocalGoalResult
LocalPlanner::selectLocalGoal(double progress_s,
                              const Eigen::Vector3d& /*current_position*/,
                              double current_speed) const {
    LocalGoalResult result;
    if (global_path_.empty() || arc_lengths_.empty()) return result;

    double total_length = arc_lengths_.back();

    // Adaptive lookahead
    double speed = std::max(0.1, current_speed);
    double lookahead = config_.lookahead_distance +
                       config_.lookahead_velocity_gain * speed;
    lookahead = std::max(config_.min_lookahead_distance, lookahead);
    lookahead = std::min(config_.max_lookahead_distance, lookahead);
    lookahead = std::min(config_.local_map_radius, lookahead);

    // Check if near final goal
    double dist_to_end = total_length - progress_s;
    if (dist_to_end <= lookahead + config_.goal_tolerance) {
        // Use final goal directly
        result.position = global_path_.back();
        result.waypoint_index = static_cast<int>(global_path_.size()) - 1;
        result.arc_length_from_start = total_length;
        result.is_final_goal = true;
        result.valid = true;
        return result;
    }

    // Walk forward along path to find point at lookahead distance
    double target_s = progress_s + lookahead;
    target_s = std::min(target_s, total_length - 1e-6);

    // Binary search on arc_lengths_
    auto it = std::lower_bound(arc_lengths_.begin(), arc_lengths_.end(), target_s);
    if (it == arc_lengths_.end()) {
        result.position = global_path_.back();
        result.waypoint_index = static_cast<int>(global_path_.size()) - 1;
        result.arc_length_from_start = total_length;
        result.is_final_goal = true;
        result.valid = true;
        return result;
    }

    int idx = static_cast<int>(it - arc_lengths_.begin());
    if (idx == 0) {
        result.position = global_path_[0];
        result.waypoint_index = 0;
        result.arc_length_from_start = 0.0;
    } else {
        double s0 = arc_lengths_[idx - 1];
        double s1 = arc_lengths_[idx];
        double alpha = (s1 - s0 > 1e-12) ? (target_s - s0) / (s1 - s0) : 0.0;
        alpha = std::max(0.0, std::min(1.0, alpha));
        result.position = global_path_[idx - 1] * (1.0 - alpha) + global_path_[idx] * alpha;
        result.waypoint_index = idx;
        result.arc_length_from_start = target_s;
    }

    result.valid = true;
    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  Yaw utilities
// ─────────────────────────────────────────────────────────────────────

double LocalPlanner::yawFromTangent(const Eigen::Vector3d& direction) {
    // Project convention: yaw=0 means the vehicle nose faces world +Y.
    return std::atan2(direction.y(), direction.x()) - kPi / 2.0;
}

double LocalPlanner::wrapAngle(double angle) {
    angle = std::fmod(angle, kTwoPi);
    if (angle > kPi) angle -= kTwoPi;
    if (angle < -kPi) angle += kTwoPi;
    return angle;
}

// ─────────────────────────────────────────────────────────────────────
//  Trajectory sampling from control points (cubic B-spline style)
// ─────────────────────────────────────────────────────────────────────

std::vector<TrajectoryPoint>
LocalPlanner::sampleTrajectory(
    const std::vector<Eigen::Vector3d>& control_points,
    const Eigen::Vector3d& start_pos,
    const Eigen::Vector3d& start_vel,
    const Eigen::Vector3d& start_acc,
    const Eigen::Vector3d& goal_pos,
    double start_yaw,
    double dt,
    int num_samples) const {

    std::vector<TrajectoryPoint> traj;
    traj.reserve(num_samples);
    // A receding-horizon planner repeatedly executes only the beginning of
    // each trajectory. Therefore braking/steering toward the goal must be a
    // boundary condition at t=0; postponing it to the middle of the horizon
    // makes every replan reset before the correction is reached.
    (void)start_acc;
    auto desired_initial_acceleration =
        [&](double total_time) -> Eigen::Vector3d {
        const double safe_time = std::max(dt, total_time);
        Eigen::Vector3d acceleration =
            6.0 * (goal_pos - start_pos) /
                (safe_time * safe_time) -
            4.0 * start_vel / safe_time;
        const double limit =
            0.85 * std::max(0.1, config_.max_acceleration);
        const double magnitude = acceleration.norm();
        if (magnitude > limit) {
            acceleration *= limit / magnitude;
        }
        return acceleration;
    };

    // Use a minimum-jerk / quintic polynomial approach from start to goal
    // with control points shaping the interior of the trajectory.
    // For simplicity and determinism, we use a cubic spline through control points
    // with the first control point being the start position and the last being
    // the goal (or near it).

    if (control_points.size() == 2 && num_samples >= 2) {
        const Eigen::Vector3d delta = goal_pos - start_pos;
        const double distance = delta.norm();
        const double total_time = std::max(dt, (num_samples - 1) * dt);
        Eigen::Vector3d tangent = Eigen::Vector3d::UnitX();
        if (distance > 1.0e-9) {
            tangent = delta / distance;
        }
        // Full vector boundary conditions are important here.  Projecting the
        // current velocity onto the new line silently discarded lateral
        // inertia, so the reported velocity was not the derivative of the
        // reported position during replanning.
        const Eigen::Vector3d displacement = goal_pos - start_pos;
        const Eigen::Vector3d initial_acceleration =
            desired_initial_acceleration(total_time);
        const Eigen::Vector3d a3 =
            10.0 * displacement / std::pow(total_time, 3) -
            6.0 * start_vel / std::pow(total_time, 2) -
            1.5 * initial_acceleration / total_time;
        const Eigen::Vector3d a4 =
            -15.0 * displacement / std::pow(total_time, 4) +
            8.0 * start_vel / std::pow(total_time, 3) +
            1.5 * initial_acceleration /
                std::pow(total_time, 2);
        const Eigen::Vector3d a5 =
            6.0 * displacement / std::pow(total_time, 5) -
            3.0 * start_vel / std::pow(total_time, 4) -
            0.5 * initial_acceleration /
                std::pow(total_time, 3);

        const double initial_yaw = start_yaw;
        const double target_yaw = yawFromTangent(tangent);
        const double yaw_delta = wrapAngle(target_yaw - initial_yaw);

        for (int i = 0; i < num_samples; ++i) {
            const double t = i * dt;
            const double u = std::min(1.0, t / total_time);
            const double t2 = t * t;
            const double t3 = t2 * t;
            const double t4 = t3 * t;
            const double t5 = t4 * t;
            const double progress =
                10.0 * u * u * u - 15.0 * std::pow(u, 4) +
                6.0 * std::pow(u, 5);
            const double progress_rate =
                (30.0 * u * u - 60.0 * u * u * u +
                 30.0 * std::pow(u, 4)) / total_time;

            TrajectoryPoint point;
            point.t = t;
            point.position =
                start_pos + start_vel * t +
                0.5 * initial_acceleration * t2 +
                a3 * t3 + a4 * t4 + a5 * t5;
            point.velocity =
                start_vel + initial_acceleration * t +
                3.0 * a3 * t2 + 4.0 * a4 * t3 + 5.0 * a5 * t4;
            point.acceleration =
                initial_acceleration +
                6.0 * a3 * t + 12.0 * a4 * t2 + 20.0 * a5 * t3;
            point.yaw = wrapAngle(initial_yaw + progress * yaw_delta);
            point.yaw_rate = progress_rate * yaw_delta;
            point.clearance = esdf_.getValue(
                point.position.x(), point.position.y(), point.position.z());
            traj.push_back(point);
        }
        traj.front().position = start_pos;
        traj.back().position = goal_pos;
        traj.back().velocity.setZero();
        traj.back().acceleration.setZero();
        return traj;
    }

    if (control_points.size() < 2 || num_samples < 2) {
        // Fallback: straight line with velocity profile
        Eigen::Vector3d dir = goal_pos - start_pos;
        double total_dist = dir.norm();
        if (total_dist < 1e-6) {
            for (int i = 0; i < num_samples; ++i) {
                TrajectoryPoint pt;
                pt.t = i * dt;
                pt.position = start_pos;
                pt.velocity = Eigen::Vector3d::Zero();
                pt.acceleration = Eigen::Vector3d::Zero();
                pt.yaw = 0.0;
                pt.yaw_rate = 0.0;
                pt.clearance = esdf_.getValue(start_pos.x(), start_pos.y(), start_pos.z());
                traj.push_back(pt);
            }
            return traj;
        }

        Eigen::Vector3d unit_dir = dir / total_dist;
        double total_time = num_samples * dt;
        // Replanning starts from the current velocity.  Resetting each direct
        // path to zero made a velocity executor brake at planner_hz.
        double v_cruise = std::min(config_.nominal_speed,
                                   total_dist / (total_time * 0.7));
        const double start_speed = std::max(
            0.0, std::min(config_.max_velocity, start_vel.dot(unit_dir)));
        double t_accel = std::abs(v_cruise - start_speed) /
                         config_.max_acceleration;
        t_accel = std::min(t_accel, total_time * 0.3);

        Eigen::Vector3d prev_pos = start_pos;
        Eigen::Vector3d prev_vel = start_vel;
        double prev_yaw = start_yaw;

        for (int i = 0; i < num_samples; ++i) {
            double t = i * dt;
            double s;  // fraction along path [0, 1]
            double v;  // speed

            if (t < t_accel) {
                // Acceleration phase
                const double sign = (v_cruise >= start_speed) ? 1.0 : -1.0;
                v = start_speed + sign * config_.max_acceleration * t;
                s = (start_speed * t +
                     0.5 * sign * config_.max_acceleration * t * t) / total_dist;
            } else if (t > total_time - t_accel) {
                // Deceleration phase
                double t_rem = total_time - t;
                v = config_.max_acceleration * t_rem;
                s = 1.0 - 0.5 * config_.max_acceleration * t_rem * t_rem / total_dist;
            } else {
                v = v_cruise;
                const double sign = (v_cruise >= start_speed) ? 1.0 : -1.0;
                double s_accel = (start_speed * t_accel +
                    0.5 * sign * config_.max_acceleration * t_accel * t_accel) /
                    total_dist;
                s = s_accel + v_cruise * (t - t_accel) / total_dist;
            }
            s = std::max(0.0, std::min(1.0, s));
            v = std::max(0.0, std::min(config_.max_velocity, v));

            Eigen::Vector3d pos = start_pos + s * dir;
            Eigen::Vector3d vel = unit_dir * v;

            // Yaw from velocity direction
            double yaw;
            if (vel.norm() > 0.01) {
                yaw = yawFromTangent(vel.normalized());
            } else {
                yaw = prev_yaw;
            }
            double yaw_rate = 0.0;
            if (i > 0 && dt > 1e-9) {
                yaw_rate = wrapAngle(yaw - prev_yaw) / dt;
            }

            TrajectoryPoint pt;
            pt.t = t;
            pt.position = pos;
            pt.velocity = vel;
            if (i == 0) {
                pt.acceleration.setZero();
            } else {
                pt.acceleration = (vel - prev_vel) / dt;
            }
            pt.yaw = yaw;
            pt.yaw_rate = yaw_rate;
            pt.clearance = esdf_.getValue(pos.x(), pos.y(), pos.z());
            traj.push_back(pt);

            prev_pos = pos;
            prev_vel = vel;
            prev_yaw = yaw;
        }
        return traj;
    }

    // Use control points to define a path via piecewise cubic spline
    // and then apply a velocity profile.
    int n_cp = static_cast<int>(control_points.size());

    // Build cumulative chord-length parameterization of control points
    std::vector<double> cp_s(n_cp, 0.0);
    for (int i = 1; i < n_cp; ++i) {
        cp_s[i] = cp_s[i - 1] + (control_points[i] - control_points[i - 1]).norm();
    }
    double total_cp_len = cp_s.back();

    if (total_cp_len < 1e-6) {
        // Degenerate: straight line
        return sampleTrajectory({start_pos, goal_pos}, start_pos, start_vel,
                                start_acc, goal_pos, start_yaw, dt, num_samples);
    }

    double total_time = std::max(dt, (num_samples - 1) * dt);
    double prev_yaw = start_yaw;

    // Clamped uniform cubic B-spline.  Clamping makes the first and last
    // control points exact curve endpoints; de Boor evaluation preserves all
    // bends supplied by the observed-known-free A* reference segment.
    const int degree = std::min(3, n_cp - 1);
    std::vector<double> knots(n_cp + degree + 1, 0.0);
    const int internal_spans = n_cp - degree;
    for (int i = degree + 1; i < n_cp; ++i) {
        knots[i] = static_cast<double>(i - degree) / internal_spans;
    }
    for (int i = n_cp; i < static_cast<int>(knots.size()); ++i) {
        knots[i] = 1.0;
    }
    auto evaluate_bspline = [&](double u) {
        u = std::max(0.0, std::min(1.0, u));
        int span = n_cp - 1;
        if (u < 1.0) {
            auto upper = std::upper_bound(knots.begin() + degree,
                                          knots.begin() + n_cp + 1, u);
            span = std::max(degree,
                std::min(n_cp - 1, static_cast<int>(upper - knots.begin()) - 1));
        }
        std::vector<Eigen::Vector3d> d(degree + 1);
        for (int j = 0; j <= degree; ++j) d[j] = control_points[span - degree + j];
        for (int r = 1; r <= degree; ++r) {
            for (int j = degree; j >= r; --j) {
                const int idx = span - degree + j;
                const double denom = knots[idx + degree - r + 1] - knots[idx];
                const double alpha = denom > 1e-12 ? (u - knots[idx]) / denom : 0.0;
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
            }
        }
        return d[degree];
    };

    // Traverse the reference geometry with a non-zero initial progress rate.
    // A zero-slope minimum-jerk law makes the first executable prefix almost
    // independent of the curved B-spline.  At 30 Hz every replan then resets
    // that zero-slope prefix, indefinitely postponing avoidance even though
    // the complete displayed trajectory bends around the obstacle.
    std::vector<Eigen::Vector3d> positions(num_samples);
    constexpr double kCurveDerivativeEpsilon = 1.0e-4;
    const Eigen::Vector3d curve_start =
        evaluate_bspline(0.0);
    const Eigen::Vector3d curve_epsilon =
        evaluate_bspline(kCurveDerivativeEpsilon);
    const Eigen::Vector3d curve_two_epsilon =
        evaluate_bspline(2.0 * kCurveDerivativeEpsilon);
    const Eigen::Vector3d curve_three_epsilon =
        evaluate_bspline(3.0 * kCurveDerivativeEpsilon);
    const Eigen::Vector3d curve_derivative =
        (curve_epsilon - curve_start) /
        kCurveDerivativeEpsilon;
    const Eigen::Vector3d curve_second_derivative =
        (curve_two_epsilon -
         2.0 * curve_epsilon +
         curve_start) /
        (kCurveDerivativeEpsilon *
         kCurveDerivativeEpsilon);
    const Eigen::Vector3d curve_third_derivative =
        (curve_three_epsilon -
         3.0 * curve_two_epsilon +
         3.0 * curve_epsilon -
         curve_start) /
        std::pow(kCurveDerivativeEpsilon, 3);
    const double curve_derivative_norm = curve_derivative.norm();
    const Eigen::Vector3d endpoint_delta = goal_pos - start_pos;
    const Eigen::Vector3d initial_curve_tangent =
        curve_derivative_norm > 1.0e-9
            ? curve_derivative / curve_derivative_norm
            : (endpoint_delta.norm() > 1.0e-9
                   ? endpoint_delta.normalized()
                   : Eigen::Vector3d::UnitX());
    const double forward_speed = std::max(
        0.0, start_vel.dot(initial_curve_tangent));
    // path_u is dimensionless and time is normalized to [0, 1].
    // Clamp only to preserve a monotone quintic for unusually large incoming
    // speeds; normal flight uses a rate close to one.
    const double initial_progress_rate = std::max(
        0.0, std::min(
            2.5,
            curve_derivative_norm > 1.0e-9
                ? forward_speed * total_time / curve_derivative_norm
                : 0.0));
    const double progress_a3 =
        10.0 - 6.0 * initial_progress_rate;
    const double progress_a4 =
        -15.0 + 8.0 * initial_progress_rate;
    const double progress_a5 =
        6.0 - 3.0 * initial_progress_rate;
    const Eigen::Vector3d initial_curve_velocity =
        curve_derivative *
        (initial_progress_rate / total_time);
    const Eigen::Vector3d initial_curve_acceleration =
        curve_second_derivative *
        std::pow(
            initial_progress_rate / total_time, 2);
    const Eigen::Vector3d initial_curve_jerk =
        curve_third_derivative *
            std::pow(
                initial_progress_rate / total_time, 3) +
        curve_derivative *
            (6.0 * progress_a3 /
             std::pow(total_time, 3));
    const Eigen::Vector3d residual_velocity =
        start_vel - initial_curve_velocity;
    // Preserve the measured velocity exactly at t=0, then release only the
    // residual component quickly enough that B-spline curvature affects the
    // 80 ms control label.  The lower bound keeps the implied acceleration
    // finite.  It also grows with an auto-timed trajectory, so extending the
    // duration can actually reduce boundary jerk instead of repeatedly
    // reproducing the same fixed transient.
    const double boundary_correction_time = std::max(
        0.50, std::min(
            1.20,
            std::max(
                2.2 * residual_velocity.norm() /
                    std::max(
                        0.1,
                        config_.max_acceleration),
                0.12 * total_time)));
    const Eigen::Vector3d bounded_initial_acceleration =
        desired_initial_acceleration(total_time);
    const double correction_decay_rate =
        1.0 / boundary_correction_time +
        3.0 / total_time;
    // For correction(t) = (v_r*t + 0.5*a_r*t^2)*decay(t),
    // correction''(0) = a_r - 2*decay_rate*v_r.  Choose a_r so the
    // complete B-spline trajectory starts at the bounded acceleration
    // instead of producing a fixed >5 m/s^2 transient that cannot be
    // removed by stretching the total trajectory duration.
    const Eigen::Vector3d residual_acceleration =
        bounded_initial_acceleration -
        initial_curve_acceleration +
        2.0 * correction_decay_rate *
            residual_velocity;
    const double correction_second_decay =
        correction_decay_rate * correction_decay_rate -
        3.0 / (total_time * total_time);
    // Extend the same boundary identity by one derivative.  For
    // P(t)=v_r*t+0.5*a_r*t^2+j_r*t^3/6 and decay g(t):
    // (P*g)'''(0)=j_r-3*k*a_r+3*g''(0)*v_r.
    // A zero initial jerk leaves ample margin below max_jerk while the
    // subsequent curve still begins inside the executable prefix.
    const Eigen::Vector3d residual_jerk =
        -initial_curve_jerk +
        3.0 * correction_decay_rate *
            residual_acceleration -
        3.0 * correction_second_decay *
            residual_velocity;
    for (int i = 0; i < num_samples; ++i) {
        const double t = i * dt;
        const double u = std::min(1.0, t / total_time);
        const double u2 = u * u;
        const double u3 = u2 * u;
        const double u4 = u3 * u;
        const double u5 = u4 * u;
        const double path_u = std::max(
            0.0, std::min(
                1.0,
                initial_progress_rate * u +
                progress_a3 * u3 +
                progress_a4 * u4 +
                progress_a5 * u5));
        const double end_taper = std::pow(1.0 - u, 3);
        const Eigen::Vector3d inertia_correction =
            (residual_velocity * t +
             0.5 * residual_acceleration * t * t +
             (1.0 / 6.0) * residual_jerk *
                 t * t * t) *
            std::exp(-t / boundary_correction_time) *
            end_taper;
        positions[i] = evaluate_bspline(path_u) + inertia_correction;
    }
    positions.front() = start_pos;
    positions.back() = goal_pos;

    std::vector<Eigen::Vector3d> velocities(
        num_samples, Eigen::Vector3d::Zero());
    for (int i = 0; i < num_samples - 1; ++i) {
        velocities[i] = (positions[i + 1] - positions[i]) / dt;
    }
    velocities.back().setZero();

    std::vector<Eigen::Vector3d> accelerations(
        num_samples, Eigen::Vector3d::Zero());
    for (int i = 0; i < num_samples - 1; ++i) {
        accelerations[i] = (velocities[i + 1] - velocities[i]) / dt;
    }
    if (num_samples > 1) {
        accelerations.back() = accelerations[num_samples - 2];
    }

    for (int i = 0; i < num_samples; ++i) {
        const double t = i * dt;
        double yaw = prev_yaw;
        double yaw_rate = 0.0;
        if (i > 0 && velocities[i].norm() > 0.01) {
            const double desired_yaw =
                yawFromTangent(velocities[i].normalized());
            const double max_step =
                std::max(0.0, config_.max_yaw_rate) * dt * 0.98;
            const double desired_step = wrapAngle(desired_yaw - prev_yaw);
            const double step = std::max(
                -max_step, std::min(max_step, desired_step));
            yaw = wrapAngle(prev_yaw + step);
            yaw_rate = step / dt;
        }

        TrajectoryPoint pt;
        pt.t = t;
        pt.position = positions[i];
        pt.velocity = velocities[i];
        pt.acceleration = accelerations[i];
        pt.yaw = yaw;
        pt.yaw_rate = yaw_rate;
        pt.clearance = esdf_.getValue(
            pt.position.x(), pt.position.y(), pt.position.z());
        traj.push_back(pt);
        prev_yaw = yaw;
    }

    return traj;
}

std::vector<TrajectoryPoint>
LocalPlanner::sampleTrajectoryFeasible(
    const std::vector<Eigen::Vector3d>& control_points,
    const Eigen::Vector3d& start_pos,
    const Eigen::Vector3d& start_vel,
    const Eigen::Vector3d& start_acc,
    const Eigen::Vector3d& goal_pos,
    double start_yaw) const {

    double path_length = 0.0;
    for (size_t i = 1; i < control_points.size(); ++i) {
        path_length += (control_points[i] - control_points[i - 1]).norm();
    }
    path_length = std::max(path_length, (goal_pos - start_pos).norm());

    const double dt = config_.trajectory_dt;
    const double nominal = std::max(0.1,
        std::min(config_.nominal_speed, config_.max_velocity));
    const double accel = std::max(0.1, config_.max_acceleration);
    double duration = std::max({
        4.0 * dt,
        path_length / nominal,
        2.0 * std::sqrt(std::max(0.0, path_length) / accel),
        start_vel.norm() / accel
    });
    // Local guides are range-limited. Six nominal travel times leaves room
    // for jerk-limited start/stop transitions; a larger value indicates an
    // invalid sampled shape and must not become a 200 s "successful" plan.
    const double duration_cap = std::max(
        6.0, 6.0 * path_length / nominal + start_vel.norm() / accel);
    duration = std::min(duration, duration_cap);

    std::vector<TrajectoryPoint> trajectory;
    double sampled_duration = -1.0;
    for (int iteration = 0; iteration < 8; ++iteration) {
        const int samples = std::max(
            2, static_cast<int>(std::ceil(duration / dt)) + 1);
        trajectory = sampleTrajectory(
            control_points, start_pos, start_vel, start_acc, goal_pos,
            start_yaw, dt, samples);
        sampled_duration = (samples - 1) * dt;

        double max_speed = 0.0;
        double max_acceleration = 0.0;
        double max_jerk = 0.0;
        double max_yaw_rate = 0.0;
        for (size_t i = 0; i < trajectory.size(); ++i) {
            max_speed = std::max(max_speed, trajectory[i].velocity.norm());
            max_acceleration = std::max(
                max_acceleration, trajectory[i].acceleration.norm());
            max_yaw_rate = std::max(
                max_yaw_rate, std::abs(trajectory[i].yaw_rate));
            if (i > 0) {
                const double positional_speed =
                    (trajectory[i].position -
                     trajectory[i - 1].position).norm() / dt;
                max_speed = std::max(max_speed, positional_speed);
                const double finite_difference_acceleration =
                    (trajectory[i].velocity -
                     trajectory[i - 1].velocity).norm() / dt;
                max_acceleration = std::max(
                    max_acceleration, finite_difference_acceleration);
                const double finite_difference_jerk =
                    (trajectory[i].acceleration -
                     trajectory[i - 1].acceleration).norm() / dt;
                max_jerk = std::max(max_jerk, finite_difference_jerk);
            }
        }

        const double speed_scale =
            max_speed / std::max(0.1, config_.max_velocity);
        const double acceleration_scale = std::sqrt(
            max_acceleration / std::max(0.1, config_.max_acceleration));
        // Jerk scales as ~1/T³ for position-sampled trajectories, so the
        // required time extension is linear in the violation ratio (not cbrt).
        const double jerk_scale =
            max_jerk / std::max(0.1, config_.max_jerk);
        const double yaw_scale =
            max_yaw_rate / std::max(0.1, config_.max_yaw_rate);
        const double required_scale = std::max({
            1.0, speed_scale, acceleration_scale, jerk_scale, yaw_scale});
        // Match the validator's kRequestDynamicsTolerance (2 %) so the
        // auto-timed result passes final validation.
        if (required_scale <= 1.02) break;
        const double next_duration = std::min(
            duration_cap,
            duration * std::min(3.0, required_scale * 1.05));
        if (next_duration <= duration + 1.0e-9) break;
        duration = next_duration;
    }
    if (duration > sampled_duration + 0.5 * dt) {
        const int samples = std::max(
            2, static_cast<int>(std::ceil(duration / dt)) + 1);
        trajectory = sampleTrajectory(
            control_points, start_pos, start_vel, start_acc, goal_pos,
            start_yaw, dt, samples);
    }
    return trajectory;
}

// ─────────────────────────────────────────────────────────────────────
//  Cost computation
// ─────────────────────────────────────────────────────────────────────

double LocalPlanner::computeCost(
    const std::vector<Eigen::Vector3d>& control_points,
    const Eigen::Vector3d& /*start_pos*/,
    const Eigen::Vector3d& /*start_vel*/,
    const Eigen::Vector3d& local_goal,
    const std::vector<Eigen::Vector3d>& global_ref_segment,
    bool near_final_goal) const {

    double cost = 0.0;
    int n = static_cast<int>(control_points.size());
    if (n < 2) return 0.0;

    // J_smooth: second-order difference (acceleration squared proxy)
    double smooth_cost = 0.0;
    for (int i = 1; i < n - 1; ++i) {
        Eigen::Vector3d lap = control_points[i - 1] + control_points[i + 1] -
                              2.0 * control_points[i];
        smooth_cost += lap.squaredNorm();
    }
    cost += config_.weight_smooth * smooth_cost;

    // J_jerk: third-order difference
    double jerk_cost = 0.0;
    for (int i = 2; i < n - 2; ++i) {
        Eigen::Vector3d j = control_points[i - 2] - 3.0 * control_points[i - 1] +
                            3.0 * control_points[i + 1] - control_points[i + 2];
        jerk_cost += j.squaredNorm();
    }
    cost += config_.weight_jerk * jerk_cost;

    // J_guide: optional straight-chord proximity.  Production sets this to
    // zero because the exact Guide endpoint already constrains the macro
    // direction; intermediate control points must remain free to avoid early.
    if (config_.weight_guide > 0.0) {
        double guide_cost = 0.0;
        for (const auto& cp : control_points) {
            double min_dist_sq = std::numeric_limits<double>::max();
            for (size_t j = 0; j + 1 < global_ref_segment.size(); ++j) {
                const Eigen::Vector3d& a = global_ref_segment[j];
                const Eigen::Vector3d& b = global_ref_segment[j + 1];
                Eigen::Vector3d ab = b - a;
                double ab_len_sq = ab.squaredNorm();
                double t = ab_len_sq > 1e-12
                    ? (cp - a).dot(ab) / ab_len_sq
                    : 0.0;
                t = std::max(0.0, std::min(1.0, t));
                Eigen::Vector3d proj = a + t * ab;
                double dist_sq = (cp - proj).squaredNorm();
                min_dist_sq = std::min(min_dist_sq, dist_sq);
            }
            guide_cost += min_dist_sq;
        }
        cost += config_.weight_guide * guide_cost;
    }

    // J_obstacle: ESDF-based penalty
    double obs_cost = 0.0;
    // Evaluate the same clamped B-spline geometry used by sampleTrajectory().
    // Penalizing the control polygon allowed the smoothed curve to cut inside
    // a safe-looking corner and fail only during final trajectory validation.
    const int obstacle_degree = std::min(3, n - 1);
    std::vector<double> obstacle_knots(
        n + obstacle_degree + 1, 0.0);
    const int obstacle_internal_spans = n - obstacle_degree;
    for (int i = obstacle_degree + 1; i < n; ++i) {
        obstacle_knots[i] =
            static_cast<double>(i - obstacle_degree) /
            obstacle_internal_spans;
    }
    for (int i = n;
         i < static_cast<int>(obstacle_knots.size()); ++i) {
        obstacle_knots[i] = 1.0;
    }
    auto evaluate_obstacle_bspline = [&](double u) {
        u = std::max(0.0, std::min(1.0, u));
        int span = n - 1;
        if (u < 1.0) {
            auto upper = std::upper_bound(
                obstacle_knots.begin() + obstacle_degree,
                obstacle_knots.begin() + n + 1, u);
            span = std::max(
                obstacle_degree,
                std::min(
                    n - 1,
                    static_cast<int>(
                        upper - obstacle_knots.begin()) - 1));
        }
        std::vector<Eigen::Vector3d> values(
            obstacle_degree + 1);
        for (int j = 0; j <= obstacle_degree; ++j) {
            values[j] =
                control_points[span - obstacle_degree + j];
        }
        for (int r = 1; r <= obstacle_degree; ++r) {
            for (int j = obstacle_degree; j >= r; --j) {
                const int idx =
                    span - obstacle_degree + j;
                const double denominator =
                    obstacle_knots[
                        idx + obstacle_degree - r + 1] -
                    obstacle_knots[idx];
                const double alpha =
                    denominator > 1.0e-12
                        ? (u - obstacle_knots[idx]) / denominator
                        : 0.0;
                values[j] =
                    (1.0 - alpha) * values[j - 1] +
                    alpha * values[j];
            }
        }
        return values[obstacle_degree];
    };
    double obstacle_polygon_length = 0.0;
    for (int i = 1; i < n; ++i) {
        obstacle_polygon_length +=
            (control_points[i] -
             control_points[i - 1]).norm();
    }
    const int obstacle_samples = std::max(
        2, std::min(
            std::max(
                2,
                (n - 1) *
                    std::max(
                        2,
                        config_.max_cost_samples_per_segment)),
            static_cast<int>(
                obstacle_polygon_length /
                std::max(
                    0.01,
                    config_.collision_check_spacing)) +
                1));
    for (int sample = 0; sample <= obstacle_samples; ++sample) {
        const double u =
            static_cast<double>(sample) / obstacle_samples;
        const Eigen::Vector3d point =
            evaluate_obstacle_bspline(u);
        const double clearance = esdf_.getValue(
            point.x(), point.y(), point.z());
        if (clearance < config_.target_clearance) {
            const double deficit =
                config_.target_clearance - clearance;
            // Normalize by the available soft-clearance band.  In metre
            // units deficit^2 is numerically tiny (0.15 m -> 0.0225), so
            // guide/smoothness costs previously pulled a safe seed back
            // toward the straight chord.  The hard threshold is unchanged.
            const double clearance_band = std::max(
                0.05,
                config_.target_clearance -
                    config_.min_clearance);
            const double normalized_deficit =
                deficit / clearance_band;
            obs_cost +=
                normalized_deficit * normalized_deficit;
        }
        if (clearance <= config_.min_clearance) {
            const double hard_deficit =
                config_.min_clearance - clearance;
            const double resolution_scale =
                std::max(0.01, esdf_.resolution());
            // This is a hard feasibility boundary, not another soft
            // preference.  Without a fixed barrier an only-slightly unsafe
            // sample (for example 0.013 m with a 0.02 m limit) contributes a
            // surprisingly small metre-squared penalty.  SBPLX can then trade
            // it for guide/smoothness improvement and return an invalid
            // incumbent when its time budget expires.
            obs_cost += 1000.0 + 1000.0 *
                (hard_deficit * hard_deficit +
                 resolution_scale * hard_deficit);
        }
    }
    cost += config_.weight_obstacle * obs_cost;

    // J_goal: endpoint near local_goal
    double goal_dist_sq = (control_points.back() - local_goal).squaredNorm();
    double goal_weight = near_final_goal ? config_.weight_goal * 3.0 : config_.weight_goal;
    cost += goal_weight * goal_dist_sq;

    // J_dynamics: velocity / acceleration limits on sampled trajectory
    // (sampled from control points to approximate). Quick check: distance between
    // consecutive control points should not exceed max_velocity * dt_control_points
    double control_polygon_length = 0.0;
    for (int i = 1; i < n; ++i) {
        control_polygon_length +=
            (control_points[i] - control_points[i - 1]).norm();
    }
    const double optimization_time = std::max(
        config_.horizon_time,
        control_polygon_length /
            std::max(0.1, std::min(config_.nominal_speed,
                                   config_.max_velocity)));
    double cp_dt = optimization_time / (n - 1);
    double dyn_cost = 0.0;
    for (int i = 1; i < n; ++i) {
        double dist = (control_points[i] - control_points[i - 1]).norm();
        double v = dist / cp_dt;
        if (v > config_.max_velocity) {
            dyn_cost += (v - config_.max_velocity) * (v - config_.max_velocity);
        }
    }
    cost += config_.weight_dynamics * dyn_cost;

    return cost;
}

// ─────────────────────────────────────────────────────────────────────
//  Control-point optimization (gradient descent + line search)
// ─────────────────────────────────────────────────────────────────────

bool LocalPlanner::optimizeControlPoints(
    std::vector<Eigen::Vector3d>& control_points,
    const Eigen::Vector3d& start_pos,
    const Eigen::Vector3d& start_vel,
    const Eigen::Vector3d& local_goal,
    const std::vector<Eigen::Vector3d>& global_ref_segment,
    bool near_final_goal) const {

    int n = static_cast<int>(control_points.size());
    if (n < 2) return false;

    // Fix first control point to start_pos
    control_points[0] = start_pos;

    const auto optimization_start = std::chrono::steady_clock::now();
    const double budget_ms = std::max(1.0, config_.planning_time_budget_ms);
    const double optimization_budget_ms =
        std::max(1.0, budget_ms - 2.0);
    const auto budget_exhausted = [&]() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() -
                   optimization_start).count() >= optimization_budget_ms;
    };

    // A straight path through an obstacle is a poor initial condition for a
    // high-dimensional derivative-free optimizer: ESDF gradients on opposite
    // sides of the obstacle cancel and hundreds of evaluations can be spent
    // before a coherent bypass direction appears.  First form an elastic-band
    // seed using one consistent ESDF ascent direction.  This is deterministic,
    // bounded, and normally costs only a few hundred ESDF queries.
    if (n > 2) {
        Eigen::Vector3d bypass_direction = Eigen::Vector3d::Zero();
        const Eigen::Vector3d guide_delta = local_goal - start_pos;
        const Eigen::Vector3d guide_direction =
            guide_delta.norm() > 1.0e-9
                ? guide_delta.normalized()
                : Eigen::Vector3d::UnitX();
        constexpr int kClearanceSeedIterations = 24;
        for (int iteration = 0;
             iteration < kClearanceSeedIterations && !budget_exhausted();
             ++iteration) {
            double worst_clearance =
                std::numeric_limits<double>::infinity();
            Eigen::Vector3d worst_point = start_pos;
            bool has_clearance_deficit = false;

            for (int i = 0; i < n - 1 && !budget_exhausted(); ++i) {
                const Eigen::Vector3d segment =
                    control_points[i + 1] - control_points[i];
                const int steps = std::max(
                    2, std::min(
                        std::max(2, config_.max_cost_samples_per_segment),
                        static_cast<int>(
                            segment.norm() /
                            std::max(0.01, config_.collision_check_spacing)) +
                            1));
                for (int sample = 0; sample <= steps; ++sample) {
                    const double alpha =
                        static_cast<double>(sample) / steps;
                    const Eigen::Vector3d point =
                        (1.0 - alpha) * control_points[i] +
                        alpha * control_points[i + 1];
                    const double clearance = esdf_.getValue(
                        point.x(), point.y(), point.z());
                    if (clearance < worst_clearance) {
                        worst_clearance = clearance;
                        worst_point = point;
                    }
                    if (clearance >= config_.target_clearance) continue;

                    has_clearance_deficit = true;
                }
            }

            if (!has_clearance_deficit) break;

            if (bypass_direction.norm() <= 0.5) {
                double ignored_clearance = worst_clearance;
                Eigen::Vector3d gradient = esdf_.getGradient(
                    worst_point.x(), worst_point.y(), worst_point.z(),
                    &ignored_clearance);
                // Only the component normal to travel produces an actual
                // bend; a tangential push merely bunches control points.
                gradient -= gradient.dot(guide_direction) * guide_direction;
                if (gradient.norm() > 1.0e-6) {
                    bypass_direction = gradient.normalized();
                } else {
                    // At the exact ESDF medial axis the gradient may be zero.
                    // Probe a deterministic orthogonal basis and choose the
                    // side whose nearby clearance increases most.
                    Eigen::Vector3d axis =
                        std::abs(guide_direction.z()) < 0.8
                            ? Eigen::Vector3d::UnitZ()
                            : Eigen::Vector3d::UnitY();
                    Eigen::Vector3d side =
                        guide_direction.cross(axis).normalized();
                    Eigen::Vector3d vertical =
                        guide_direction.cross(side).normalized();
                    const double probe =
                        std::max(0.10, 2.0 * esdf_.resolution());
                    const std::array<Eigen::Vector3d, 4> candidates{
                        side, -side, vertical, -vertical};
                    double best_clearance =
                        -std::numeric_limits<double>::infinity();
                    for (const auto& candidate : candidates) {
                        const Eigen::Vector3d probe_point =
                            worst_point + probe * candidate;
                        const double candidate_clearance = esdf_.getValue(
                            probe_point.x(), probe_point.y(), probe_point.z());
                        if (candidate_clearance > best_clearance) {
                            best_clearance = candidate_clearance;
                            bypass_direction = candidate;
                        }
                    }
                }

                // The first pass only located the worst point.  Re-run it
                // with the selected direction so forces cannot cancel.
                continue;
            }

            // Apply one broad, smooth bump rather than moving only the
            // already-dangerous control points.  The displacement ramps up
            // from zero at the current state, reaches its maximum at the
            // worst ESDF location, then smoothly rejoins the exact Guide.
            // This creates lateral velocity before reaching the obstacle and
            // avoids the former "follow the chord, then make a sharp hook"
            // behaviour.
            const double guide_length_sq = guide_delta.squaredNorm();
            double obstacle_fraction =
                guide_length_sq > 1.0e-9
                    ? (worst_point - start_pos).dot(guide_delta) /
                          guide_length_sq
                    : 0.5;
            obstacle_fraction = std::max(
                0.10, std::min(0.90, obstacle_fraction));
            const double peak_update = std::min(
                0.12,
                0.70 * std::max(
                    0.0, config_.target_clearance - worst_clearance));
            auto smooth_step = [](double value) {
                value = std::max(0.0, std::min(1.0, value));
                return value * value * (3.0 - 2.0 * value);
            };
            double maximum_update = 0.0;
            for (int i = 1; i < n - 1; ++i) {
                const double fraction =
                    static_cast<double>(i) / (n - 1);
                const double phase =
                    fraction <= obstacle_fraction
                        ? fraction / obstacle_fraction
                        : (1.0 - fraction) /
                              (1.0 - obstacle_fraction);
                const Eigen::Vector3d update =
                    peak_update * smooth_step(phase) *
                    bypass_direction;
                control_points[i] += update;
                maximum_update = std::max(maximum_update, update.norm());
            }
            control_points.front() = start_pos;
            control_points.back() = local_goal;
            if (maximum_update < 1.0e-4) break;
        }
    }

#ifdef IL_DATASET_HAS_NLOPT
    if (config_.optimizer == "auto" || config_.optimizer == "nlopt") {
        if (budget_exhausted()) return true;
        // Both endpoints are boundary conditions.  Optimizing the terminal
        // and then overwriting it with local_goal wastes three dimensions and
        // can invalidate the optimizer's reported optimum.
        const int variable_count = std::max(0, n - 2) * 3;
        if (variable_count == 0) return true;
        std::vector<double> x(variable_count);
        for (int i = 1; i < n - 1; ++i)
            for (int d = 0; d < 3; ++d)
                x[(i - 1) * 3 + d] = control_points[i][d];

        auto unpack = [&](const std::vector<double>& values,
                          std::vector<Eigen::Vector3d>& points) {
            points[0] = start_pos;
            points[n - 1] = local_goal;
            for (int i = 1; i < n - 1; ++i)
                for (int d = 0; d < 3; ++d)
                    points[i][d] = values[(i - 1) * 3 + d];
        };

        struct ObjectiveContext {
            std::function<double(unsigned, const double*, double*)> evaluate;
            nlopt_opt optimizer = nullptr;
        } context;

        context.evaluate = [&](unsigned dimension, const double* values, double* gradient) {
            if (budget_exhausted()) {
                if (gradient != nullptr) std::fill(gradient, gradient + dimension, 0.0);
                if (context.optimizer != nullptr) nlopt_force_stop(context.optimizer);
                return 1e30;
            }
            std::vector<double> values_vec(values, values + dimension);
            std::vector<Eigen::Vector3d> points = control_points;
            unpack(values_vec, points);
            double base = computeCost(points, start_pos, start_vel,
                                      local_goal, global_ref_segment,
                                      near_final_goal);
            if (!std::isfinite(base)) base = 1e30;
            if (gradient != nullptr) {
                const double eps = 1e-4;
                std::vector<double> perturbed = values_vec;
                for (unsigned j = 0; j < dimension; ++j) {
                    if (budget_exhausted()) {
                        std::fill(gradient + j, gradient + dimension, 0.0);
                        if (context.optimizer != nullptr)
                            nlopt_force_stop(context.optimizer);
                        break;
                    }
                    perturbed[j] += eps;
                    unpack(perturbed, points);
                    const double perturbed_cost = computeCost(
                        points, start_pos, start_vel, local_goal,
                        global_ref_segment, near_final_goal);
                    gradient[j] = std::isfinite(perturbed_cost)
                                      ? (perturbed_cost - base) / eps
                                      : 0.0;
                    perturbed[j] = values_vec[j];
                }
            }
            return base;
        };

        const auto objective_trampoline =
            [](unsigned dimension, const double* values, double* gradient,
               void* opaque) -> double {
                return static_cast<ObjectiveContext*>(opaque)->evaluate(
                    dimension, values, gradient);
            };

        // A numerical gradient costs O(3*N) complete objective evaluations
        // and can overrun the real-time budget inside one callback.  SBPLX is
        // derivative-free, bounded and checks maxtime between fixed-size
        // objective calls, giving much more deterministic latency here.
        nlopt_opt opt = nlopt_create(NLOPT_LN_SBPLX,
                                     static_cast<unsigned>(variable_count));
        if (opt != nullptr) {
            context.optimizer = opt;
            // Keep every free control point inside both the ESDF map and a
            // bounded tube around the relevant global-path segment.  An
            // unconstrained L-BFGS step can otherwise create kilometre-scale
            // points, making subsequent dense collision validation take
            // seconds even though optimization itself was time-bounded.
            Eigen::Vector3d corridor_min = start_pos.cwiseMin(local_goal);
            Eigen::Vector3d corridor_max = start_pos.cwiseMax(local_goal);
            for (const auto& point : global_ref_segment) {
                corridor_min = corridor_min.cwiseMin(point);
                corridor_max = corridor_max.cwiseMax(point);
            }
            corridor_min.array() -= 1.5;
            corridor_max.array() += 1.5;
            const Eigen::Vector3d local_min =
                (start_pos.array() - config_.local_map_radius).matrix();
            const Eigen::Vector3d local_max =
                (start_pos.array() + config_.local_map_radius).matrix();
            corridor_min = corridor_min.cwiseMax(local_min);
            corridor_max = corridor_max.cwiseMin(local_max);
            const double map_margin = std::max(esdf_.resolution(), 0.05);
            const Eigen::Vector3d map_min(esdf_.originX() + map_margin,
                                          esdf_.originY() + map_margin,
                                          esdf_.originZ() + map_margin);
            const Eigen::Vector3d map_max(
                esdf_.originX() + esdf_.gx() * esdf_.resolution() - map_margin,
                esdf_.originY() + esdf_.gy() * esdf_.resolution() - map_margin,
                esdf_.originZ() + esdf_.gz() * esdf_.resolution() - map_margin);
            corridor_min = corridor_min.cwiseMax(map_min);
            corridor_max = corridor_max.cwiseMin(map_max);
            std::vector<double> lower(variable_count), upper(variable_count);
            for (int i = 1; i < n - 1; ++i) {
                for (int d = 0; d < 3; ++d) {
                    const int j = (i - 1) * 3 + d;
                    lower[j] = corridor_min[d];
                    upper[j] = corridor_max[d];
                    x[j] = std::max(lower[j], std::min(upper[j], x[j]));
                }
            }
            nlopt_set_lower_bounds(opt, lower.data());
            nlopt_set_upper_bounds(opt, upper.data());
            nlopt_set_min_objective(opt, objective_trampoline, &context);
            nlopt_set_maxeval(opt, std::max(1, config_.max_iterations));
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() -
                    optimization_start).count();
            // Preserve time for unpacking and dense final validation inside
            // the caller's strict 30 Hz planning tick.
            const double remaining_ms =
                std::max(0.5, optimization_budget_ms - elapsed_ms);
            nlopt_set_maxtime(opt, remaining_ms * 0.001);
            nlopt_set_initial_step1(
                opt, std::max(config_.minimum_step_size,
                              config_.initial_step_size));
            nlopt_set_ftol_rel(opt,
                              std::max(1e-12, config_.convergence_tolerance));
            double optimum = std::numeric_limits<double>::infinity();
            const nlopt_result status = nlopt_optimize(opt, x.data(), &optimum);
            context.optimizer = nullptr;
            nlopt_destroy(opt);
            unpack(x, control_points);

            // Positive values are normal convergence.  MAXTIME/MAXEVAL are
            // also usable bounded exits as long as the returned point is finite.
            if (status > 0 || status == NLOPT_MAXTIME_REACHED ||
                status == NLOPT_MAXEVAL_REACHED || budget_exhausted()) {
                return std::all_of(x.begin(), x.end(),
                                   [](double value) { return std::isfinite(value); });
            }
            if (config_.optimizer == "nlopt") return false;
        } else if (config_.optimizer == "nlopt") {
            return false;
        }
    }
#endif

    double step_size = config_.initial_step_size;
    double prev_cost = computeCost(control_points, start_pos, start_vel,
                                   local_goal, global_ref_segment, near_final_goal);

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        if (budget_exhausted()) break;
        // Numerical gradient for each free interior control point.
        std::vector<Eigen::Vector3d> grad(n, Eigen::Vector3d::Zero());
        double eps = 1e-4;

        for (int i = 1; i < n - 1; ++i) {
            for (int d = 0; d < 3; ++d) {
                if (budget_exhausted()) return true;
                double orig = control_points[i][d];
                control_points[i][d] = orig + eps;
                double cost_plus = computeCost(control_points, start_pos, start_vel,
                                               local_goal, global_ref_segment,
                                               near_final_goal);
                control_points[i][d] = orig - eps;
                double cost_minus = computeCost(control_points, start_pos, start_vel,
                                                local_goal, global_ref_segment,
                                                near_final_goal);
                control_points[i][d] = orig;

                grad[i][d] = (cost_plus - cost_minus) / (2.0 * eps);
            }
        }

        // Backtracking line search
        std::vector<Eigen::Vector3d> candidate = control_points;
        bool accepted = false;
        double trial_step = step_size;

        for (int bt = 0; bt < 10; ++bt) {
            if (budget_exhausted()) return true;
            for (int i = 1; i < n - 1; ++i) {
                candidate[i] = control_points[i] - trial_step * grad[i];
            }
            candidate.front() = start_pos;
            candidate.back() = local_goal;

            double new_cost = computeCost(candidate, start_pos, start_vel,
                                          local_goal, global_ref_segment,
                                          near_final_goal);

            if (new_cost < prev_cost - 1e-12) {
                control_points = candidate;
                accepted = true;
                step_size = std::min(step_size * 1.1, config_.initial_step_size * 2.0);
                break;
            }
            trial_step *= 0.5;
        }

        if (!accepted) {
            step_size = std::max(step_size * 0.5, config_.minimum_step_size);
            if (step_size <= config_.minimum_step_size + 1e-12) {
                break;  // converged
            }
            continue;
        }

        double new_cost = computeCost(control_points, start_pos, start_vel,
                                      local_goal, global_ref_segment, near_final_goal);

        if (std::abs(prev_cost - new_cost) < config_.convergence_tolerance &&
            step_size <= config_.minimum_step_size * 2.0) {
            break;
        }

        prev_cost = new_cost;
    }

    return true;
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
//  Emergency hold trajectory
// ─────────────────────────────────────────────────────────────────────

std::vector<TrajectoryPoint>
LocalPlanner::generateEmergencyHold(const VehicleState& current_state) const {
    std::vector<TrajectoryPoint> traj;
    double dt = config_.trajectory_dt;
    int n = static_cast<int>(config_.horizon_time / dt);

    Eigen::Vector3d pos = current_state.position;
    Eigen::Vector3d vel = current_state.velocity;

    double hold_yaw = current_state.yaw;

    for (int i = 0; i < n; ++i) {
        double t = i * dt;
        // Exponential deceleration
        double decay = std::exp(-3.0 * t);
        Eigen::Vector3d v = vel * decay;
        pos = pos + v * dt;

        TrajectoryPoint pt;
        pt.t = t;
        pt.position = pos;
        pt.velocity = v;
        pt.acceleration = -3.0 * v;  // approx
        pt.yaw = hold_yaw;
        pt.yaw_rate = 0.0;
        pt.clearance = esdf_.getValue(pos.x(), pos.y(), pos.z());
        traj.push_back(pt);
    }

    return traj;
}

// ─────────────────────────────────────────────────────────────────────
//  Global-path fallback
// ─────────────────────────────────────────────────────────────────────

std::vector<TrajectoryPoint>
LocalPlanner::generateGlobalPathFallback(const VehicleState& current_state,
                                         double progress_s,
                                         double lookahead) const {
    double total_length = arc_lengths_.back();
    double end_s = std::min(progress_s + lookahead, total_length);

    auto point_at_s = [&](double query_s) -> Eigen::Vector3d {
        query_s = std::max(0.0, std::min(total_length, query_s));
        auto it = std::lower_bound(arc_lengths_.begin(), arc_lengths_.end(), query_s);
        if (it == arc_lengths_.begin()) return global_path_.front();
        if (it == arc_lengths_.end()) return global_path_.back();
        const size_t idx = static_cast<size_t>(it - arc_lengths_.begin());
        const double s0 = arc_lengths_[idx - 1];
        const double s1 = arc_lengths_[idx];
        const double alpha = (s1 > s0 + 1e-12) ? (query_s - s0) / (s1 - s0) : 0.0;
        return (global_path_[idx - 1] * (1.0 - alpha) +
                global_path_[idx] * alpha).eval();
    };

    // Preserve all bends of the validated global path.  The old fallback used
    // a direct current->final-goal line when no sparse shortcut waypoint fell
    // inside the horizon, which could cut directly through an obstacle.
    std::vector<Eigen::Vector3d> segment{current_state.position};
    const Eigen::Vector3d projected_start = point_at_s(progress_s);
    if ((projected_start - segment.back()).norm() > 1e-3)
        segment.push_back(projected_start);
    for (size_t i = 0; i < global_path_.size(); ++i) {
        if (arc_lengths_[i] > progress_s + 1e-6 &&
            arc_lengths_[i] < end_s - 1e-6) {
            segment.push_back(global_path_[i]);
        }
    }
    const Eigen::Vector3d fallback_goal = point_at_s(end_s);
    if ((fallback_goal - segment.back()).norm() > 1e-3)
        segment.push_back(fallback_goal);
    if (segment.size() < 2) segment.push_back(fallback_goal);

    int n_samples = static_cast<int>(config_.horizon_time / config_.trajectory_dt);
    return sampleTrajectory(segment, current_state.position, current_state.velocity,
                            current_state.acceleration, segment.back(),
                            current_state.yaw,
                            config_.trajectory_dt, n_samples);
}

// ─────────────────────────────────────────────────────────────────────
//  Main planning entry point
// ─────────────────────────────────────────────────────────────────────

LocalPlanResult LocalPlanner::planLocal(
    const VehicleState& current_state,
    double previous_progress_s) const {

    auto t_start = std::chrono::steady_clock::now();

    LocalPlanResult result;
    result.plan_id = plan_id_counter_.fetch_add(1);

    // Check readiness
    if (!isReady()) {
        result.success = false;
        result.status = PlannerStatus::NO_GLOBAL_PATH;
        result.message = "Planner not initialized (no ESDF or global path)";
        return result;
    }

    // Check input validity
    if (!std::isfinite(current_state.position.x()) ||
        !std::isfinite(current_state.position.y()) ||
        !std::isfinite(current_state.position.z())) {
        result.success = false;
        result.status = PlannerStatus::INVALID_INPUT;
        result.message = "Current state contains NaN/Inf position";
        return result;
    }

    // ── Progress tracking ──────────────────────────────────────
    auto progress = computeProgress(current_state.position, previous_progress_s);
    if (!progress.valid) {
        result.success = false;
        result.status = PlannerStatus::LOCAL_GOAL_INVALID;
        result.message = "Could not project position onto global path";
        return result;
    }
    result.progress_s = progress.progress_s;
    result.progress_index = progress.segment_index;

    // ── Local goal selection ───────────────────────────────────
    double current_speed = current_state.velocity.norm();
    auto local_goal = selectLocalGoal(progress.progress_s,
                                      current_state.position, current_speed);
    if (!local_goal.valid) {
        result.success = false;
        result.status = PlannerStatus::LOCAL_GOAL_INVALID;
        result.message = "Could not select local goal";
        return result;
    }
    result.local_goal = local_goal.position;
    result.local_goal_index = local_goal.waypoint_index;

    // ── Straight-line check ────────────────────────────────────
    // If the direct line from current position to local goal is entirely clear,
    // use a simple trajectory without heavy optimization.
    bool direct_clear = true;
    {
        Eigen::Vector3d dir = local_goal.position - current_state.position;
        double dist = dir.norm();
        if (dist > 1e-6) {
            int steps = std::max(10, static_cast<int>(dist / config_.collision_check_spacing));
            for (int s = 0; s <= steps && direct_clear; ++s) {
                double a = static_cast<double>(s) / steps;
                Eigen::Vector3d pt = current_state.position + a * dir;
                if (!esdf_.isFree(
                        pt.x(), pt.y(), pt.z(),
                        config_.target_clearance)) {
                    direct_clear = false;
                }
            }
        }
    }

    std::vector<TrajectoryPoint> traj;
    bool optimized = false;
    bool used_global_fallback = false;

    if (direct_clear) {
        // Simple straight-line trajectory
        int n_samples = static_cast<int>(config_.horizon_time / config_.trajectory_dt);
        std::vector<Eigen::Vector3d> cp = {current_state.position, local_goal.position};
        traj = sampleTrajectory(cp, current_state.position, current_state.velocity,
                                current_state.acceleration, local_goal.position,
                                current_state.yaw,
                                config_.trajectory_dt, n_samples);
        optimized = true;
    } else {
        // ── Initialize control points ──────────────────────────
        int n_cp = config_.control_points;
        std::vector<Eigen::Vector3d> control_points(n_cp);

        // Linearly interpolate between start and local_goal for initial guess
        control_points[0] = current_state.position;
        for (int i = 1; i < n_cp; ++i) {
            double alpha = static_cast<double>(i) / (n_cp - 1);
            control_points[i] = current_state.position * (1.0 - alpha) +
                                local_goal.position * alpha;
        }
        // Bias initial guess toward global path
        for (int i = 1; i < n_cp - 1; ++i) {
            double alpha = static_cast<double>(i) / (n_cp - 1);
            double s = progress.progress_s + alpha *
                       (local_goal.arc_length_from_start - progress.progress_s);
            // Find nearest global path point at this arc-length
            auto it = std::lower_bound(arc_lengths_.begin(), arc_lengths_.end(), s);
            int idx = std::min(static_cast<int>(it - arc_lengths_.begin()),
                               static_cast<int>(global_path_.size()) - 1);
            if (idx > 0 && idx < static_cast<int>(global_path_.size())) {
                double s0 = arc_lengths_[idx - 1];
                double s1 = arc_lengths_[idx];
                double blend = (s1 > s0 + 1e-12) ? (s - s0) / (s1 - s0) : 0.0;
                blend = std::max(0.0, std::min(1.0, blend));
                Eigen::Vector3d gp = global_path_[idx - 1] * (1.0 - blend) +
                                     global_path_[idx] * blend;
                // Weighted blend: 70% global path, 30% straight line
                control_points[i] = gp * 0.7 + control_points[i] * 0.3;
            }
        }

        // Build local reference segment from global path
        std::vector<Eigen::Vector3d> ref_segment;
        double ref_start_s = std::max(0.0, progress.progress_s - 1.0);
        double ref_end_s = std::min(arc_lengths_.back(),
                                    local_goal.arc_length_from_start + 2.0);
        for (size_t i = 0; i < global_path_.size(); ++i) {
            if (arc_lengths_[i] >= ref_start_s && arc_lengths_[i] <= ref_end_s) {
                ref_segment.push_back(global_path_[i]);
            }
        }
        if (ref_segment.size() >
            static_cast<size_t>(std::max(2, config_.max_reference_points))) {
            const size_t keep = static_cast<size_t>(
                std::max(2, config_.max_reference_points));
            std::vector<Eigen::Vector3d> bounded_ref;
            bounded_ref.reserve(keep);
            for (size_t k = 0; k < keep; ++k) {
                const size_t idx = k * (ref_segment.size() - 1) / (keep - 1);
                bounded_ref.push_back(ref_segment[idx]);
            }
            ref_segment.swap(bounded_ref);
        }
        if (ref_segment.size() < 2) {
            ref_segment = {current_state.position, local_goal.position};
        }

        // ── Optimize ───────────────────────────────────────────
        // Guide reference: uniformly-spaced points on the straight
        // line from current position to local_goal.  One point per
        // control point ensures even CP distribution, preventing
        // endpoint collapse and giving natural deceleration.
        std::vector<Eigen::Vector3d> straight_guide(n_cp);
        for (int i = 0; i < n_cp; ++i) {
            double frac = static_cast<double>(i) / (n_cp - 1);
            straight_guide[i] = current_state.position * (1.0 - frac) +
                                local_goal.position * frac;
        }
        optimized = optimizeControlPoints(control_points,
                                          current_state.position,
                                          current_state.velocity,
                                          local_goal.position,
                                          straight_guide,
                                          local_goal.is_final_goal);

        // ── Sample trajectory from control points ──────────────
        int n_samples = static_cast<int>(config_.horizon_time / config_.trajectory_dt);
        traj = sampleTrajectory(control_points,
                                current_state.position,
                                current_state.velocity,
                                current_state.acceleration,
                                local_goal.position,
                                current_state.yaw,
                                config_.trajectory_dt,
                                n_samples);

        if (!optimized) {
            // Optimization failed – check if we can still use the trajectory
            auto val = validateTrajectory(traj);
            if (val.any_collision || !val.all_clear) {
                // Try fallback
                traj = generateGlobalPathFallback(current_state,
                                                  progress.progress_s,
                                                  config_.lookahead_distance * 0.5);
            }
        }
    }

    // ── Final validation ───────────────────────────────────────
    auto validation = validateTrajectory(traj);

    // A converged optimizer can still return a trajectory below the clearance
    // threshold. Recover along the already validated global path instead of
    // returning the same COLLISION result three times from an unchanged state.
    if (validation.any_collision || !validation.all_clear) {
        auto fallback = generateGlobalPathFallback(
            current_state, progress.progress_s,
            std::max(config_.min_lookahead_distance,
                     config_.lookahead_distance * 0.5));
        auto fallback_validation = validateTrajectory(fallback);
        if (!fallback_validation.any_collision && fallback_validation.all_clear) {
            traj = std::move(fallback);
            validation = fallback_validation;
            used_global_fallback = true;
            optimized = true;
        }
    }

    // ── Populate result ────────────────────────────────────────
    // Verify the time-scaled trajectory itself, not just the nominal-horizon
    // optimizer cost. This prevents a geometrically valid trajectory from
    // being reported successful when its timing still exceeds a vehicle limit.
    bool dynamics_feasible = !traj.empty();
    constexpr double kDynamicsTolerance = 1.02;
    for (size_t i = 0; dynamics_feasible && i < traj.size(); ++i) {
        const auto& point = traj[i];
        dynamics_feasible =
            point.velocity.norm() <= config_.max_velocity * kDynamicsTolerance &&
            point.acceleration.norm() <=
                config_.max_acceleration * kDynamicsTolerance &&
            std::abs(point.yaw_rate) <=
                config_.max_yaw_rate * kDynamicsTolerance;
        if (!dynamics_feasible || i == 0) {
            continue;
        }

        const double dt = point.t - traj[i - 1].t;
        if (!(dt > 1.0e-9)) {
            dynamics_feasible = false;
            break;
        }
        const double position_speed =
            (point.position - traj[i - 1].position).norm() / dt;
        const double jerk =
            (point.acceleration - traj[i - 1].acceleration).norm() / dt;
        dynamics_feasible =
            position_speed <= config_.max_velocity * kDynamicsTolerance &&
            jerk <= config_.max_jerk * kDynamicsTolerance;
    }

    result.trajectory = std::move(traj);
    result.min_clearance = validation.min_clearance;

    auto t_end = std::chrono::steady_clock::now();
    result.planning_time_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    if (validation.any_collision) {
        result.success = false;
        result.status = PlannerStatus::COLLISION;
        result.message = "Trajectory has collision (clearance <= 0)";
    } else if (!validation.all_clear) {
        result.success = false;
        result.status = PlannerStatus::COLLISION;
        result.message = "Trajectory has " +
                         std::to_string(validation.clearance_violation_count) +
                         " clearance violations";
    } else if (!dynamics_feasible) {
        result.success = false;
        result.status = PlannerStatus::DYNAMICS_VIOLATION;
        result.message = "Time-scaled trajectory exceeds a dynamics limit";
    } else if (!optimized && !direct_clear) {
        result.success = false;
        result.status = PlannerStatus::OPTIMIZATION_FAILED;
        result.message = "Optimization failed to converge";
    } else {
        result.success = true;
        result.status = PlannerStatus::SUCCESS;
        result.message = used_global_fallback ? "OK (global-path fallback)" : "OK";
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────
//  Phase 2: planLocalWithRequest — explicit guide/terminal, observed ESDF
// ─────────────────────────────────────────────────────────────────────

LocalPlanResult LocalPlanner::planLocalWithRequest(
    const LocalPlanningRequest& request) const {

    // The explicit-Guide data-collection path uses the compact production
    // spline core.  The implementation below is retained only as historical
    // legacy code while planLocal() still serves the old asynchronous API.
    return planSplineWithRequest(request);

    auto t_start = std::chrono::steady_clock::now();

    LocalPlanResult result;
    result.plan_id = plan_id_counter_.fetch_add(1);
    result.guide_waypoint = request.guide_waypoint;
    result.guide_waypoint_index = request.guide_waypoint_index;
    result.trajectory_terminal = request.trajectory_terminal;
    result.trajectory_terminal_index = request.trajectory_terminal_index;
    result.used_observed_esdf = esdf_.hasKnownMask();

    // Check readiness
    if (!isReady()) {
        result.success = false;
        result.status = PlannerStatus::NO_GLOBAL_PATH;
        result.message = "Planner not initialized (no ESDF or global path)";
        return result;
    }

    // Check input validity
    const auto& cs = request.state;
    if (!std::isfinite(cs.position.x()) ||
        !std::isfinite(cs.position.y()) ||
        !std::isfinite(cs.position.z())) {
        result.success = false;
        result.status = PlannerStatus::INVALID_INPUT;
        result.message = "Current state contains NaN/Inf position";
        return result;
    }

    // ── Progress tracking ──────────────────────────────────────
    auto progress = computeProgress(cs.position, request.previous_progress_s);
    result.progress_s = progress.valid ? progress.progress_s : 0.0;
    result.progress_index = progress.valid ? progress.segment_index : -1;

    // Use externally-provided terminal as the optimization target
    Eigen::Vector3d terminal = request.trajectory_terminal;
    const bool near_final_goal =
        request.guide_waypoint_index >= 0 &&
        !global_path_.empty() &&
        request.guide_waypoint_index >=
            static_cast<int>(global_path_.size()) - 1;

    // If no explicit terminal, fall back to horizon-distance forward point
    if ((terminal - cs.position).norm() < 1e-6) {
        // Emergency: use current position + velocity direction
        terminal = cs.position + cs.velocity.normalized() *
                   std::max(0.5, cs.velocity.norm() * 0.5);
    }

    result.local_goal = terminal;  // compatibility alias
    result.local_goal_index = request.trajectory_terminal_index;

    // Determine collision check function
    bool forbid_unknown = request.forbid_unknown_space;
    auto isFreeCheck = [&](double x, double y, double z, double cl) -> bool {
        if (forbid_unknown && esdf_.hasKnownMask()) {
            return esdf_.isKnownFree(x, y, z, cl);
        }
        return esdf_.isFree(x, y, z, cl);
    };

    // ── Straight-line check ────────────────────────────────────
    bool direct_clear = true;
    {
        Eigen::Vector3d dir = terminal - cs.position;
        double dist = dir.norm();
        if (dist > 1e-6) {
            int steps = std::max(10,
                static_cast<int>(dist / config_.collision_check_spacing));
            for (int s = 0; s <= steps && direct_clear; ++s) {
                double a = static_cast<double>(s) / steps;
                Eigen::Vector3d pt = cs.position + a * dir;
                if (!isFreeCheck(
                        pt.x(), pt.y(), pt.z(),
                        config_.target_clearance)) {
                    direct_clear = false;
                }
            }
        }
    }

    std::vector<TrajectoryPoint> traj;
    bool optimized = false;
    bool used_global_fallback = false;
    int final_clearance_repairs = 0;
    std::vector<Eigen::Vector3d> reference_control_points;

    if (direct_clear) {
        std::vector<Eigen::Vector3d> cp = {cs.position, terminal};
        traj = sampleTrajectoryFeasible(
            cp, cs.position, cs.velocity, cs.acceleration, terminal,
            cs.yaw);
        optimized = true;
        const auto direct_validation = validateTrajectory(traj);
        if (direct_validation.any_collision ||
            !direct_validation.all_clear) {
            direct_clear = false;
            optimized = false;
            traj.clear();
        }
    }
    if (!direct_clear) {
        // ── Initialize control points as STRAIGHT LINE from start to terminal ──
        // NO reference_path_segment — the global path only provides the visible
        // waypoint; the local planner optimizes its own shape via ESDF.
        const double desired_cp_spacing = std::max(
            2.0 * config_.collision_check_spacing,
            config_.control_point_spacing);
        const double straight_dist = (terminal - cs.position).norm();
        const int density_cp =
            static_cast<int>(std::ceil(straight_dist / desired_cp_spacing)) + 1;
        const int n_cp = std::max(
            2, std::min(
                std::max(2, config_.max_reference_points),
                std::max(config_.control_points, density_cp)));
        std::vector<Eigen::Vector3d> control_points(n_cp);

        // Straight-line initialization: current position → terminal
        control_points[0] = cs.position;
        for (int i = 1; i < n_cp; ++i) {
            double alpha = static_cast<double>(i) / (n_cp - 1);
            control_points[i] = cs.position * (1.0 - alpha) +
                                terminal * alpha;
        }
        control_points.front() = cs.position;
        control_points.back() = terminal;
        reference_control_points = control_points;

        // ── Optimize ───────────────────────────────────────────
        // Guide reference: uniformly-spaced points on the straight
        // line (one per control point).  Prevents CP bunching at
        // endpoints and gives natural velocity taper near goal.
        std::vector<Eigen::Vector3d> straight_guide(n_cp);
        for (int i = 0; i < n_cp; ++i) {
            double frac = static_cast<double>(i) / (n_cp - 1);
            straight_guide[i] = cs.position * (1.0 - frac) +
                                terminal * frac;
        }
        optimized = optimizeControlPoints(control_points,
                                          cs.position, cs.velocity,
                                          terminal, straight_guide,
                                          near_final_goal);
        control_points.front() = cs.position;
        control_points.back() = terminal;

        // ── Sample ─────────────────────────────────────────────
        traj = sampleTrajectoryFeasible(
            control_points, cs.position, cs.velocity,
            cs.acceleration, terminal, cs.yaw);

        // computeCost() shapes the clamped B-spline geometry, whereas the
        // executable trajectory additionally contains the current-state
        // continuity correction and an automatically selected duration.
        // Close to an obstacle that final transformation can move an
        // otherwise acceptable curve back across the configured hard margin.
        // Close the loop on the object that is actually executed: use its
        // worst ESDF sample to apply a broad, coherent bump, then re-time and
        // revalidate.  This is not a global-path fallback and the exact Guide
        // remains the fixed endpoint.
        auto repaired_validation = validateTrajectory(traj);
        Eigen::Vector3d repair_direction = Eigen::Vector3d::Zero();
        const double desired_repair_clearance = std::min(
            config_.target_clearance,
            config_.min_clearance + 0.05);
        const double trajectory_start_clearance = esdf_.getValue(
            cs.position.x(), cs.position.y(), cs.position.z());
        // The current state is an immutable endpoint.  If tracking has
        // already brought it inside the desired buffer, require the repaired
        // suffix to be non-degrading instead of chasing an impossible minimum
        // above the fixed start clearance.
        const double repair_clearance = std::min(
            desired_repair_clearance,
            trajectory_start_clearance);
        constexpr int kMaximumFinalClearanceRepairs = 10;
        for (int repair_iteration = 0;
             repair_iteration < kMaximumFinalClearanceRepairs &&
             (repaired_validation.any_collision ||
              !repaired_validation.all_clear ||
              repaired_validation.min_clearance < repair_clearance);
             ++repair_iteration) {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t_start).count();
            if (elapsed_ms >=
                std::max(1.0, config_.planning_time_budget_ms - 3.0)) {
                break;
            }
            final_clearance_repairs = repair_iteration + 1;

            const Eigen::Vector3d guide_delta = terminal - cs.position;
            const double guide_length_sq = guide_delta.squaredNorm();
            if (guide_length_sq <= 1.0e-9) break;
            const Eigen::Vector3d guide_direction =
                guide_delta.normalized();

            double ignored_clearance =
                repaired_validation.worst_clearance;
            Eigen::Vector3d current_direction = esdf_.getGradient(
                repaired_validation.worst_position.x(),
                repaired_validation.worst_position.y(),
                repaired_validation.worst_position.z(),
                &ignored_clearance);
            current_direction -=
                current_direction.dot(guide_direction) *
                guide_direction;
            if (current_direction.norm() <= 1.0e-6) {
                if (repair_direction.norm() <= 0.5) {
                    Eigen::Vector3d axis =
                        std::abs(guide_direction.z()) < 0.8
                            ? Eigen::Vector3d::UnitZ()
                            : Eigen::Vector3d::UnitY();
                    Eigen::Vector3d side =
                        guide_direction.cross(axis).normalized();
                    Eigen::Vector3d vertical =
                        guide_direction.cross(side).normalized();
                    const std::array<Eigen::Vector3d, 4> candidates{
                        side, -side, vertical, -vertical};
                    const double probe =
                        std::max(0.10, 2.0 * esdf_.resolution());
                    double best_clearance =
                        -std::numeric_limits<double>::infinity();
                    for (const auto& candidate : candidates) {
                        const Eigen::Vector3d probe_point =
                            repaired_validation.worst_position +
                            probe * candidate;
                        const double candidate_clearance = esdf_.getValue(
                            probe_point.x(), probe_point.y(),
                            probe_point.z());
                        if (candidate_clearance > best_clearance) {
                            best_clearance = candidate_clearance;
                            current_direction = candidate;
                        }
                    }
                }
            } else {
                current_direction.normalize();
            }
            if (current_direction.norm() > 0.5) {
                // Keep one homotopy side, but update the direction as the
                // minimum moves around the obstacle surface.  Holding the
                // first gradient forever becomes nearly tangential after a
                // few repairs and wastes the remaining iterations.
                if (repair_direction.norm() > 0.5 &&
                    current_direction.dot(repair_direction) < 0.0) {
                    current_direction = -current_direction;
                }
                repair_direction = current_direction.normalized();
            }
            if (repair_direction.norm() <= 0.5) break;

            // Locate the dangerous portion with the exact time -> path_u
            // progress law used by sampleTrajectory().  Neither projection
            // onto the Guide chord nor the nearest control point is a valid
            // inverse for a clamped B-spline with non-linear auto timing; both
            // can place the repair peak several spans after the violation.
            const double total_time =
                !traj.empty()
                    ? std::max(config_.trajectory_dt, traj.back().t)
                    : config_.horizon_time;
            const int repair_degree = std::min(3, n_cp - 1);
            const int repair_internal_spans =
                n_cp - repair_degree;
            Eigen::Vector3d initial_curve_derivative =
                Eigen::Vector3d::Zero();
            if (n_cp >= 2 && repair_internal_spans > 0) {
                initial_curve_derivative =
                    static_cast<double>(
                        repair_degree * repair_internal_spans) *
                    (control_points[1] - control_points[0]);
            }
            const double derivative_norm =
                initial_curve_derivative.norm();
            const Eigen::Vector3d initial_tangent =
                derivative_norm > 1.0e-9
                    ? initial_curve_derivative / derivative_norm
                    : guide_direction;
            const double forward_speed = std::max(
                0.0, cs.velocity.dot(initial_tangent));
            const double initial_progress_rate = std::max(
                0.0, std::min(
                    2.5,
                    derivative_norm > 1.0e-9
                        ? forward_speed * total_time /
                              derivative_norm
                        : 0.0));
            const double normalized_time = std::max(
                0.0, std::min(
                    1.0,
                    repaired_validation.worst_time / total_time));
            const double time2 =
                normalized_time * normalized_time;
            const double time3 = time2 * normalized_time;
            const double time4 = time3 * normalized_time;
            const double time5 = time4 * normalized_time;
            const double progress_a3 =
                10.0 - 6.0 * initial_progress_rate;
            const double progress_a4 =
                -15.0 + 8.0 * initial_progress_rate;
            const double progress_a5 =
                6.0 - 3.0 * initial_progress_rate;
            double obstacle_fraction =
                initial_progress_rate * normalized_time +
                progress_a3 * time3 +
                progress_a4 * time4 +
                progress_a5 * time5;
            obstacle_fraction = std::max(
                0.0, std::min(1.0, obstacle_fraction));

            // Evaluate the non-zero B-spline basis functions at path_u.
            // Moving a broad range of control points by an index-space bump
            // is not a local repair: path_u is a knot parameter, not
            // control_point_index/(N-1).  Repeated bumps therefore stretched
            // the far half of the trajectory into a large hook while barely
            // moving the actual low-clearance sample.  The minimum-norm
            // projection below updates only the at-most degree+1 control
            // points that truly influence this curve parameter.
            std::vector<double> repair_knots(
                n_cp + repair_degree + 1, 0.0);
            for (int i = repair_degree + 1; i < n_cp; ++i) {
                repair_knots[i] =
                    static_cast<double>(i - repair_degree) /
                    repair_internal_spans;
            }
            for (int i = n_cp;
                 i < static_cast<int>(repair_knots.size()); ++i) {
                repair_knots[i] = 1.0;
            }
            // Repair toward min_clearance + 0.05 m so interpolation and the
            // next 30 Hz replan do not immediately fall back onto the hard
            // boundary; target_clearance remains only the soft preference.
            const double desired_curve_displacement = std::min(
                0.12,
                std::max(
                    0.01,
                    repair_clearance -
                        repaired_validation.worst_clearance));

            auto apply_local_basis_displacement =
                [&](double parameter, double gain) {
                parameter = std::max(
                    0.0, std::min(1.0, parameter));
                int span = n_cp - 1;
                if (parameter < 1.0) {
                    auto upper = std::upper_bound(
                        repair_knots.begin() + repair_degree,
                        repair_knots.begin() + n_cp + 1,
                        parameter);
                    span = std::max(
                        repair_degree,
                        std::min(
                            n_cp - 1,
                            static_cast<int>(
                                upper - repair_knots.begin()) - 1));
                }
                std::vector<double> basis(
                    repair_degree + 1, 0.0);
                std::vector<double> left(
                    repair_degree + 1, 0.0);
                std::vector<double> right(
                    repair_degree + 1, 0.0);
                basis[0] = 1.0;
                for (int j = 1; j <= repair_degree; ++j) {
                    left[j] =
                        parameter -
                        repair_knots[span + 1 - j];
                    right[j] =
                        repair_knots[span + j] -
                        parameter;
                    double saved = 0.0;
                    for (int r = 0; r < j; ++r) {
                        const double denominator =
                            right[r + 1] + left[j - r];
                        const double term =
                            denominator > 1.0e-12
                                ? basis[r] / denominator
                                : 0.0;
                        basis[r] =
                            saved + right[r + 1] * term;
                        saved = left[j - r] * term;
                    }
                    basis[j] = saved;
                }
                double movable_norm_sq = 0.0;
                for (int j = 0; j <= repair_degree; ++j) {
                    const int control_index =
                        span - repair_degree + j;
                    if (control_index <= 0 ||
                        control_index >= n_cp - 1) {
                        continue;
                    }
                    movable_norm_sq += basis[j] * basis[j];
                }
                if (movable_norm_sq <= 1.0e-12) return;
                for (int j = 0; j <= repair_degree; ++j) {
                    const int control_index =
                        span - repair_degree + j;
                    if (control_index <= 0 ||
                        control_index >= n_cp - 1) {
                        continue;
                    }
                    control_points[control_index] +=
                        (basis[j] / movable_norm_sq) *
                        gain * desired_curve_displacement *
                        repair_direction;
                }
            };

            // A single point projection produces a narrow last-moment kink.
            // Use a finite five-anchor support: deviation starts several knot
            // spans before the obstacle, peaks at the dangerous parameter,
            // then rejoins more quickly after it.  The support remains local
            // and cannot accumulate into the former whole-trajectory hook.
            constexpr std::array<double, 5> kSupportOffsets{
                -0.18, -0.10, 0.0, 0.08, 0.16};
            constexpr std::array<double, 5> kSupportGains{
                0.30, 0.60, 0.80, 0.45, 0.20};
            for (size_t support_index = 0;
                 support_index < kSupportOffsets.size();
                 ++support_index) {
                apply_local_basis_displacement(
                    obstacle_fraction +
                        kSupportOffsets[support_index],
                    kSupportGains[support_index]);
            }
            control_points.front() = cs.position;
            control_points.back() = terminal;
            traj = sampleTrajectoryFeasible(
                control_points, cs.position, cs.velocity,
                cs.acceleration, terminal, cs.yaw);
            repaired_validation = validateTrajectory(traj);
        }

        if (!optimized) {
            auto val = validateTrajectory(traj);
            if (val.any_collision || !val.all_clear) {
                // NO fallback — optimization failed, return failure
                // The caller handles failure via hover/abort, NOT via
                // an alternative planner or global path bypass.
            }
        }
    }

    // ── Final validation ───────────────────────────────────────
    auto validation = validateTrajectory(traj);

    // Retry with unmodified straight-line reference shape.
    if ((validation.any_collision || !validation.all_clear) &&
        !reference_control_points.empty()) {
        auto reference_trajectory = sampleTrajectoryFeasible(
            reference_control_points, cs.position, cs.velocity,
            cs.acceleration, terminal, cs.yaw);
        auto reference_validation =
            validateTrajectory(reference_trajectory);
        if (!reference_validation.any_collision &&
            reference_validation.all_clear) {
            traj = std::move(reference_trajectory);
            validation = reference_validation;
            optimized = true;
        }
    }

    // NO global_map_fallback — straight-line only, return failure otherwise.

    // Check for unknown space violations
    // Validate the actual time-scaled samples. The optimizer uses a nominal
    // horizon for shaping, while sampleTrajectoryFeasible() assigns the final
    // arrival time from the configured vehicle limits.
    bool request_dynamics_feasible = !traj.empty();
    constexpr double kRequestDynamicsTolerance = 1.02;
    double request_max_velocity = 0.0;
    double request_max_acceleration = 0.0;
    double request_max_jerk = 0.0;
    double request_max_yaw_rate = 0.0;
    double request_max_velocity_residual = 0.0;
    double request_max_acceleration_residual = 0.0;
    for (size_t i = 0; i < traj.size(); ++i) {
        const auto& point = traj[i];
        request_max_velocity =
            std::max(request_max_velocity, point.velocity.norm());
        request_max_acceleration =
            std::max(request_max_acceleration, point.acceleration.norm());
        request_max_yaw_rate =
            std::max(request_max_yaw_rate, std::abs(point.yaw_rate));
        const bool point_feasible =
            point.position.allFinite() &&
            point.velocity.allFinite() &&
            point.acceleration.allFinite() &&
            point.velocity.norm() <=
                config_.max_velocity * kRequestDynamicsTolerance &&
            point.acceleration.norm() <=
                config_.max_acceleration * kRequestDynamicsTolerance &&
            std::abs(point.yaw_rate) <=
                config_.max_yaw_rate * kRequestDynamicsTolerance;
        request_dynamics_feasible =
            request_dynamics_feasible && point_feasible;
        if (i == 0) {
            continue;
        }

        const double dt = point.t - traj[i - 1].t;
        if (!(dt > 1.0e-9)) {
            request_dynamics_feasible = false;
            continue;
        }
        const Eigen::Vector3d interval_velocity =
            (point.position - traj[i - 1].position) / dt;
        const double position_speed = interval_velocity.norm();
        const Eigen::Vector3d interval_acceleration =
            (point.velocity - traj[i - 1].velocity) / dt;
        const double jerk =
            (point.acceleration - traj[i - 1].acceleration).norm() / dt;
        request_max_velocity =
            std::max(request_max_velocity, position_speed);
        request_max_jerk = std::max(request_max_jerk, jerk);

        // Trapezoidal integration residuals catch trajectories whose reported
        // velocity/acceleration were generated independently of position.
        const double velocity_residual =
            (interval_velocity -
             0.5 * (point.velocity + traj[i - 1].velocity)).norm();
        const double acceleration_residual =
            (interval_acceleration -
             0.5 * (point.acceleration +
                    traj[i - 1].acceleration)).norm();
        request_max_velocity_residual =
            std::max(request_max_velocity_residual, velocity_residual);
        request_max_acceleration_residual =
            std::max(request_max_acceleration_residual,
                     acceleration_residual);

        const double velocity_residual_limit = std::max(
            0.15, 2.0 * config_.max_acceleration * dt);
        const double acceleration_residual_limit = std::max(
            0.50, 2.0 * config_.max_jerk * dt);
        request_dynamics_feasible =
            request_dynamics_feasible &&
            position_speed <=
                config_.max_velocity * kRequestDynamicsTolerance &&
            jerk <= config_.max_jerk * kRequestDynamicsTolerance &&
            velocity_residual <= velocity_residual_limit &&
            acceleration_residual <= acceleration_residual_limit;
    }

    if (forbid_unknown && esdf_.hasKnownMask()) {
        bool has_unknown = false;
        for (const auto& tp : traj) {
            if (!esdf_.isKnown(tp.position.x(), tp.position.y(), tp.position.z())) {
                has_unknown = true;
                break;
            }
        }
        if (has_unknown) {
            result.success = false;
            result.status = PlannerStatus::UNKNOWN_SPACE;
            result.message = "Trajectory enters unknown space";
            result.trajectory = std::move(traj);
            result.min_clearance = validation.min_clearance;
            auto t_end = std::chrono::steady_clock::now();
            result.planning_time_ms = std::chrono::duration<double, std::milli>(
                t_end - t_start).count();
            return result;
        }
    }

    // ── Populate result ────────────────────────────────────────
    result.trajectory = std::move(traj);
    result.min_clearance = validation.min_clearance;

    auto t_end = std::chrono::steady_clock::now();
    result.planning_time_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();

    if (validation.any_collision) {
        result.success = false;
        result.status = PlannerStatus::COLLISION;
        result.message =
            "Trajectory has collision (clearance <= 0) after " +
            std::to_string(final_clearance_repairs) +
            " final-clearance repairs, min=" +
            std::to_string(validation.min_clearance);
    } else if (!validation.all_clear) {
        result.success = false;
        result.status = PlannerStatus::COLLISION;
        result.message = "Trajectory has " +
                         std::to_string(validation.clearance_violation_count) +
                         " clearance violations after " +
                         std::to_string(final_clearance_repairs) +
                         " final-clearance repairs, min=" +
                         std::to_string(validation.min_clearance);
    } else if (!request_dynamics_feasible) {
        result.success = false;
        result.status = PlannerStatus::DYNAMICS_VIOLATION;
        result.message =
            "Auto-timed trajectory exceeds a dynamics limit: v=" +
            std::to_string(request_max_velocity) +
            ", a=" + std::to_string(request_max_acceleration) +
            ", jerk=" + std::to_string(request_max_jerk) +
            ", yaw_rate=" + std::to_string(request_max_yaw_rate) +
            ", v_residual=" +
            std::to_string(request_max_velocity_residual) +
            ", a_residual=" +
            std::to_string(request_max_acceleration_residual);
    } else if (!optimized && !direct_clear) {
        result.success = false;
        result.status = PlannerStatus::OPTIMIZATION_FAILED;
        result.message = "Optimization failed to converge";
    } else {
        result.success = true;
        result.status = PlannerStatus::SUCCESS;
        result.message =
            final_clearance_repairs > 0
                ? "OK (final clearance repaired " +
                      std::to_string(final_clearance_repairs) + "x)"
                : (used_global_fallback
                       ? "OK (global-path fallback)"
                       : "OK");
    }

    result.used_global_fallback = used_global_fallback;

    return result;
}

}  // namespace il_dataset
