#include "il_dataset/local_planner/local_planner.hpp"

#include <algorithm>
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

    // Use a minimum-jerk progress law for the reference geometry.  A
    // short-lived boundary correction represents the vehicle's current
    // inertia without overwriting velocity independently of position.
    std::vector<Eigen::Vector3d> positions(num_samples);
    const double braking_time = std::max(
        0.45, std::min(2.00,
            2.50 * start_vel.norm() /
            std::max(0.1, config_.max_acceleration)));
    const Eigen::Vector3d initial_acceleration =
        desired_initial_acceleration(total_time);
    const Eigen::Vector3d correction_quadratic =
        initial_acceleration +
        2.0 * start_vel / braking_time +
        6.0 * start_vel / total_time;
    for (int i = 0; i < num_samples; ++i) {
        const double t = i * dt;
        const double u = std::min(1.0, t / total_time);
        const double u2 = u * u;
        const double u3 = u2 * u;
        const double path_u =
            10.0 * u3 - 15.0 * u3 * u + 6.0 * u3 * u2;
        const double end_taper = std::pow(1.0 - u, 3);
        const Eigen::Vector3d inertia_correction =
            (start_vel * t +
             0.5 * correction_quadratic * t * t) *
            std::exp(-t / braking_time) * end_taper;
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

    // J_guide: distance of control points to global reference segment
    double guide_cost = 0.0;
    for (const auto& cp : control_points) {
        double min_dist_sq = std::numeric_limits<double>::max();
        for (size_t j = 0; j + 1 < global_ref_segment.size(); ++j) {
            const Eigen::Vector3d& a = global_ref_segment[j];
            const Eigen::Vector3d& b = global_ref_segment[j + 1];
            Eigen::Vector3d ab = b - a;
            double ab_len_sq = ab.squaredNorm();
            double t = ab_len_sq > 1e-12 ? (cp - a).dot(ab) / ab_len_sq : 0.0;
            t = std::max(0.0, std::min(1.0, t));
            Eigen::Vector3d proj = a + t * ab;
            double dist_sq = (cp - proj).squaredNorm();
            min_dist_sq = std::min(min_dist_sq, dist_sq);
        }
        guide_cost += min_dist_sq;
    }
    cost += config_.weight_guide * guide_cost;

    // J_obstacle: ESDF-based penalty
    double obs_cost = 0.0;
    double spacing = config_.collision_check_spacing;
    for (int i = 0; i < n - 1; ++i) {
        Eigen::Vector3d seg = control_points[i + 1] - control_points[i];
        double seg_len = seg.norm();
        int steps = std::max(2, static_cast<int>(seg_len / spacing) + 1);
        steps = std::min(steps,
                         std::max(2, config_.max_cost_samples_per_segment));
        for (int s = 0; s <= steps; ++s) {
            double alpha = static_cast<double>(s) / steps;
            Eigen::Vector3d pt = control_points[i] * (1.0 - alpha) + control_points[i + 1] * alpha;
            double clearance = esdf_.getValue(pt.x(), pt.y(), pt.z());
            if (clearance < config_.target_clearance) {
                double deficit = config_.target_clearance - clearance;
                obs_cost += deficit * deficit;
            }
            // Hard constraint penalty
            if (clearance <= config_.min_clearance) {
                obs_cost += 1000.0;  // large penalty for violation
            }
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
    const auto budget_exhausted = [&]() {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - optimization_start).count() >= budget_ms;
    };

#ifdef IL_DATASET_HAS_NLOPT
    if (config_.optimizer == "auto" || config_.optimizer == "nlopt") {
        const int variable_count = (n - 1) * 3;
        std::vector<double> x(variable_count);
        for (int i = 1; i < n; ++i)
            for (int d = 0; d < 3; ++d)
                x[(i - 1) * 3 + d] = control_points[i][d];

        auto unpack = [&](const std::vector<double>& values,
                          std::vector<Eigen::Vector3d>& points) {
            points[0] = start_pos;
            for (int i = 1; i < n; ++i)
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
            for (int i = 1; i < n; ++i) {
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
            nlopt_set_maxtime(opt, budget_ms * 0.001);
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
        // Numerical gradient for each free control point (indices 1 .. n-1)
        std::vector<Eigen::Vector3d> grad(n, Eigen::Vector3d::Zero());
        double eps = 1e-4;

        for (int i = 1; i < n; ++i) {
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
            for (int i = 1; i < n; ++i) {
                candidate[i] = control_points[i] - trial_step * grad[i];
            }

            // Ensure endpoint moves toward local_goal
            const Eigen::Vector3d endpoint_error = control_points.back() - local_goal;
            const double endpoint_distance = endpoint_error.norm();
            if (endpoint_distance > 1e-9) {
                candidate.back() = control_points.back() -
                                   trial_step * (endpoint_error / endpoint_distance) *
                                   std::min(trial_step * 10.0, endpoint_distance * 0.5);
            }

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

            result.min_clearance = std::min(result.min_clearance, clearance);

            // ESDF interpolation around the threshold can differ by a few
            // floating-point ulps.  Keep a 1 mm numerical tolerance; this is
            // not a reduction of the configured physical safety margin.
            if (clearance < config_.min_clearance - 1e-3) {
                result.clearance_violation_count++;
                result.all_clear = false;
                if (clearance < result.worst_clearance) {
                    result.worst_clearance = clearance;
                    result.worst_position = pt;
                    result.worst_time = p0.t + alpha * (p1.t - p0.t);
                }
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
                if (!esdf_.isFree(pt.x(), pt.y(), pt.z(), config_.min_clearance)) {
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
    bool near_final_goal = false;

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
                if (!isFreeCheck(pt.x(), pt.y(), pt.z(), config_.min_clearance)) {
                    direct_clear = false;
                }
            }
        }
    }

    std::vector<TrajectoryPoint> traj;
    bool optimized = false;
    bool used_global_fallback = false;
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
        // ── Initialize control points from reference path segment ──
        const auto& ref_seg = request.reference_path_segment;
        std::vector<double> ref_arc;
        double ref_total = (terminal - cs.position).norm();
        if (ref_seg.size() >= 2) {
            // Compute cumulative arc-length on reference segment
            ref_arc.push_back(0.0);
            for (size_t i = 1; i < ref_seg.size(); ++i) {
                ref_arc.push_back(ref_arc.back() +
                    (ref_seg[i] - ref_seg[i - 1]).norm());
            }
            ref_total = std::max(ref_total, ref_arc.back());
        }

        const double desired_cp_spacing = std::max(
            2.0 * config_.collision_check_spacing,
            config_.control_point_spacing);
        const int density_cp =
            static_cast<int>(std::ceil(ref_total / desired_cp_spacing)) + 1;
        const int n_cp = std::max(
            2, std::min(
                std::max(2, config_.max_reference_points),
                std::max(config_.control_points, density_cp)));
        std::vector<Eigen::Vector3d> control_points(n_cp);
        const double sampling_ref_total =
            ref_arc.empty() ? 0.0 : ref_arc.back();

        if (ref_seg.size() >= 2) {
            // Fix first control point to start position
            control_points[0] = cs.position;

            if (sampling_ref_total > 1e-6) {
                // Sample control points at uniform arc-length along reference
                for (int i = 1; i < n_cp; ++i) {
                    double frac = static_cast<double>(i) / (n_cp - 1);
                    double target_s = frac * sampling_ref_total;
                    auto it = std::lower_bound(ref_arc.begin(), ref_arc.end(), target_s);
                    int idx = std::min(static_cast<int>(it - ref_arc.begin()),
                                       static_cast<int>(ref_seg.size()) - 1);
                    if (idx == 0) {
                        control_points[i] = ref_seg[0];
                    } else {
                        double s0 = ref_arc[idx - 1];
                        double s1 = ref_arc[idx];
                        double alpha = (s1 > s0 + 1e-12) ?
                            (target_s - s0) / (s1 - s0) : 0.0;
                        alpha = std::max(0.0, std::min(1.0, alpha));
                        control_points[i] = ref_seg[idx - 1] * (1.0 - alpha) +
                                           ref_seg[idx] * alpha;
                    }
                }
            } else {
                // Degenerate: linear interpolation
                for (int i = 1; i < n_cp; ++i) {
                    double alpha = static_cast<double>(i) / (n_cp - 1);
                    control_points[i] = cs.position * (1.0 - alpha) +
                                        terminal * alpha;
                }
            }
        } else {
            // No reference segment: linear from start to terminal
            control_points[0] = cs.position;
            for (int i = 1; i < n_cp; ++i) {
                double alpha = static_cast<double>(i) / (n_cp - 1);
                control_points[i] = cs.position * (1.0 - alpha) +
                                    terminal * alpha;
            }
        }
        control_points.front() = cs.position;
        control_points.back() = terminal;
        reference_control_points = control_points;

        // Build reference segment from A* sub-path (or fallback)
        std::vector<Eigen::Vector3d> ref_for_cost = ref_seg;
        if (ref_for_cost.size() < 2) {
            ref_for_cost = {cs.position, terminal};
        }
        if (ref_for_cost.size() > static_cast<size_t>(
                std::max(2, config_.max_reference_points))) {
            const size_t keep = static_cast<size_t>(
                std::max(2, config_.max_reference_points));
            std::vector<Eigen::Vector3d> bounded;
            bounded.reserve(keep);
            for (size_t k = 0; k < keep; ++k) {
                const size_t idx = k * (ref_for_cost.size() - 1) / (keep - 1);
                bounded.push_back(ref_for_cost[idx]);
            }
            ref_for_cost.swap(bounded);
        }

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

        if (!optimized) {
            auto val = validateTrajectory(traj);
            if (val.any_collision || !val.all_clear) {
                // Phase 2: NO global map fallback.
                // Only allow known-free A* sub-path fallback.
                if (request.allow_global_map_fallback) {
                    traj = generateGlobalPathFallback(
                        cs, progress.progress_s,
                        config_.lookahead_distance * 0.5);
                    used_global_fallback = true;
                }
                // else: return the (possibly colliding) trajectory;
                // the caller handles failure via hover/abort.
            }
        }
    }

    // ── Final validation ───────────────────────────────────────
    auto validation = validateTrajectory(traj);

    // Smoothing can cut a collision-free reference corner. In that case,
    // retry the unmodified reference shape while preserving the exact Guide
    // terminal and its automatically assigned arrival time.
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

    if ((validation.any_collision || !validation.all_clear) &&
        request.allow_global_map_fallback) {
        auto fallback = generateGlobalPathFallback(
            cs, progress.progress_s,
            std::max(config_.min_lookahead_distance,
                     config_.lookahead_distance * 0.5));
        auto fb_val = validateTrajectory(fallback);
        if (!fb_val.any_collision && fb_val.all_clear) {
            traj = std::move(fallback);
            validation = fb_val;
            used_global_fallback = true;
            optimized = true;
        }
    }

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
        result.message = "Trajectory has collision (clearance <= 0)";
    } else if (!validation.all_clear) {
        result.success = false;
        result.status = PlannerStatus::COLLISION;
        result.message = "Trajectory has " +
                         std::to_string(validation.clearance_violation_count) +
                         " clearance violations";
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
        result.message = used_global_fallback ? "OK (global-path fallback)" : "OK";
    }

    result.used_global_fallback = used_global_fallback;

    return result;
}

}  // namespace il_dataset
