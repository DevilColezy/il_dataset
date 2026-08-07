#include "il_dataset/local_planner/spline_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#ifdef IL_DATASET_HAS_NLOPT
#include <nlopt.h>
#endif

#include "il_dataset/local_planner/esdf_grid.hpp"
#include "il_dataset/local_planner/local_path_search.hpp"
#include "il_dataset/local_planner/observed_map.hpp"
#include "il_dataset/local_planner/yaw_planner.hpp"

namespace il_dataset {

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kDegree = 3;
constexpr double kEpsilon = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;
constexpr double kDynamicsTolerance = 1.02;

double clamp(double value, double lo, double hi) {
    return std::max(lo, std::min(hi, value));
}

double wrapAngleLocal(double angle) {
    constexpr double kTwoPi = 2.0 * kPi;
    angle = std::fmod(angle, kTwoPi);
    if (angle > kPi) angle -= kTwoPi;
    if (angle < -kPi) angle += kTwoPi;
    return angle;
}

double yawFromVelocity(const Eigen::Vector3d& velocity) {
    return std::atan2(velocity.y(), velocity.x()) - 0.5 * kPi;
}

std::vector<double> makeClampedKnots(int control_point_count, int degree) {
    std::vector<double> knots(
        static_cast<size_t>(control_point_count + degree + 1), 0.0);
    const int spans = control_point_count - degree;
    for (int i = degree + 1; i < control_point_count; ++i) {
        knots[static_cast<size_t>(i)] =
            static_cast<double>(i - degree) / spans;
    }
    for (int i = control_point_count;
         i < static_cast<int>(knots.size()); ++i) {
        knots[static_cast<size_t>(i)] = 1.0;
    }
    return knots;
}

int findSpan(const std::vector<double>& knots,
             int control_point_count,
             int degree,
             double parameter) {
    if (parameter >= 1.0) return control_point_count - 1;
    const auto upper = std::upper_bound(
        knots.begin() + degree,
        knots.begin() + control_point_count + 1,
        std::max(0.0, parameter));
    return std::max(
        degree,
        std::min(
            control_point_count - 1,
            static_cast<int>(upper - knots.begin()) - 1));
}

struct LocalBasis {
    int first = 0;
    std::vector<double> weights;
};

LocalBasis evaluateBasis(const std::vector<double>& knots,
                         int control_point_count,
                         int degree,
                         double parameter) {
    parameter = clamp(parameter, 0.0, 1.0);
    const int span =
        findSpan(knots, control_point_count, degree, parameter);
    LocalBasis result;
    result.first = span - degree;
    result.weights.assign(static_cast<size_t>(degree + 1), 0.0);
    std::vector<double> left(static_cast<size_t>(degree + 1), 0.0);
    std::vector<double> right(static_cast<size_t>(degree + 1), 0.0);
    result.weights[0] = 1.0;
    for (int j = 1; j <= degree; ++j) {
        left[static_cast<size_t>(j)] =
            parameter - knots[static_cast<size_t>(span + 1 - j)];
        right[static_cast<size_t>(j)] =
            knots[static_cast<size_t>(span + j)] - parameter;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            const double denominator =
                right[static_cast<size_t>(r + 1)] +
                left[static_cast<size_t>(j - r)];
            const double term =
                denominator > kEpsilon
                    ? result.weights[static_cast<size_t>(r)] / denominator
                    : 0.0;
            result.weights[static_cast<size_t>(r)] =
                saved + right[static_cast<size_t>(r + 1)] * term;
            saved = left[static_cast<size_t>(j - r)] * term;
        }
        result.weights[static_cast<size_t>(j)] = saved;
    }
    return result;
}

Eigen::Vector3d evaluateSpline(
    const std::vector<Eigen::Vector3d>& control_points,
    const std::vector<double>& knots,
    int degree,
    double parameter) {
    if (control_points.empty()) return Eigen::Vector3d::Zero();
    const LocalBasis basis = evaluateBasis(
        knots, static_cast<int>(control_points.size()), degree, parameter);
    Eigen::Vector3d value = Eigen::Vector3d::Zero();
    for (int j = 0; j <= degree; ++j) {
        value += basis.weights[static_cast<size_t>(j)] *
                 control_points[static_cast<size_t>(basis.first + j)];
    }
    return value;
}

struct SplineDerivatives {
    std::vector<Eigen::Vector3d> first;
    std::vector<Eigen::Vector3d> second;
    std::vector<Eigen::Vector3d> third;
    std::vector<double> first_knots;
    std::vector<double> second_knots;
    std::vector<double> third_knots;
    std::vector<double> first_scales;
    std::vector<double> second_scales;
    std::vector<double> third_scales;
};

SplineDerivatives buildDerivatives(
    const std::vector<Eigen::Vector3d>& control_points,
    const std::vector<double>& knots) {
    SplineDerivatives result;
    const int count = static_cast<int>(control_points.size());
    result.first.reserve(static_cast<size_t>(count - 1));
    result.first_scales.reserve(static_cast<size_t>(count - 1));
    for (int i = 0; i + 1 < count; ++i) {
        const double denominator =
            knots[static_cast<size_t>(i + kDegree + 1)] -
            knots[static_cast<size_t>(i + 1)];
        const double scale =
            denominator > kEpsilon ? kDegree / denominator : 0.0;
        result.first_scales.push_back(scale);
        result.first.push_back(
            scale * (control_points[static_cast<size_t>(i + 1)] -
                     control_points[static_cast<size_t>(i)]));
    }
    result.first_knots.assign(knots.begin() + 1, knots.end() - 1);

    const int first_degree = kDegree - 1;
    result.second.reserve(
        result.first.size() > 1 ? result.first.size() - 1 : 0);
    result.second_scales.reserve(result.second.capacity());
    for (int i = 0; i + 1 < static_cast<int>(result.first.size()); ++i) {
        const double denominator =
            result.first_knots[static_cast<size_t>(i + first_degree + 1)] -
            result.first_knots[static_cast<size_t>(i + 1)];
        const double scale =
            denominator > kEpsilon ? first_degree / denominator : 0.0;
        result.second_scales.push_back(scale);
        result.second.push_back(
            scale * (result.first[static_cast<size_t>(i + 1)] -
                     result.first[static_cast<size_t>(i)]));
    }
    result.second_knots.assign(
        result.first_knots.begin() + 1,
        result.first_knots.end() - 1);

    const int second_degree = kDegree - 2;
    result.third.reserve(
        result.second.size() > 1 ? result.second.size() - 1 : 0);
    result.third_scales.reserve(result.third.capacity());
    for (int i = 0; i + 1 < static_cast<int>(result.second.size()); ++i) {
        const double denominator =
            result.second_knots[static_cast<size_t>(i + second_degree + 1)] -
            result.second_knots[static_cast<size_t>(i + 1)];
        const double scale =
            denominator > kEpsilon ? second_degree / denominator : 0.0;
        result.third_scales.push_back(scale);
        result.third.push_back(
            scale * (result.second[static_cast<size_t>(i + 1)] -
                     result.second[static_cast<size_t>(i)]));
    }
    result.third_knots.assign(
        result.second_knots.begin() + 1,
        result.second_knots.end() - 1);
    return result;
}

void imposeBoundaryState(
    std::vector<Eigen::Vector3d>* control_points,
    const std::vector<double>& knots,
    double duration,
    const Eigen::Vector3d& start_position,
    const Eigen::Vector3d& start_velocity,
    const Eigen::Vector3d& start_acceleration,
    const Eigen::Vector3d& end_position,
    const Eigen::Vector3d& end_velocity) {
    auto& points = *control_points;
    if (points.size() < 6) return;
    const int count = static_cast<int>(points.size());
    points.front() = start_position;
    points.back() = end_position;

    const double d0_denominator =
        knots[static_cast<size_t>(kDegree + 1)] - knots[1];
    const Eigen::Vector3d desired_d0 = duration * start_velocity;
    points[1] = points[0] +
                (d0_denominator / kDegree) * desired_d0;

    std::vector<double> first_knots(knots.begin() + 1, knots.end() - 1);
    const double e0_denominator =
        first_knots[static_cast<size_t>(kDegree)] - first_knots[1];
    const Eigen::Vector3d desired_e0 =
        duration * duration * start_acceleration;
    const Eigen::Vector3d desired_d1 =
        desired_d0 +
        (e0_denominator / static_cast<double>(kDegree - 1)) * desired_e0;
    const double d1_denominator =
        knots[static_cast<size_t>(kDegree + 2)] - knots[2];
    points[2] = points[1] +
                (d1_denominator / kDegree) * desired_d1;

    const int derivative_index = count - 2;
    const double end_denominator =
        knots[static_cast<size_t>(derivative_index + kDegree + 1)] -
        knots[static_cast<size_t>(derivative_index + 1)];
    points[static_cast<size_t>(count - 2)] =
        end_position -
        (end_denominator / kDegree) * duration * end_velocity;

    const int previous_derivative_index = count - 3;
    const double previous_end_denominator =
        knots[static_cast<size_t>(previous_derivative_index + kDegree + 1)] -
        knots[static_cast<size_t>(previous_derivative_index + 1)];
    points[static_cast<size_t>(count - 3)] =
        points[static_cast<size_t>(count - 2)] -
        (previous_end_denominator / kDegree) * duration * end_velocity;
}

double polylineLength(const std::vector<Eigen::Vector3d>& points) {
    double length = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        length += (points[i] - points[i - 1]).norm();
    }
    return length;
}

std::vector<Eigen::Vector3d> resamplePolyline(
    const std::vector<Eigen::Vector3d>& path,
    int output_count) {
    std::vector<Eigen::Vector3d> result;
    if (path.empty() || output_count <= 0) return result;
    if (path.size() == 1 || output_count == 1) {
        result.assign(static_cast<size_t>(output_count), path.front());
        return result;
    }
    std::vector<double> arc(path.size(), 0.0);
    for (size_t i = 1; i < path.size(); ++i) {
        arc[i] = arc[i - 1] + (path[i] - path[i - 1]).norm();
    }
    const double total = arc.back();
    result.reserve(static_cast<size_t>(output_count));
    if (total <= kEpsilon) {
        result.assign(static_cast<size_t>(output_count), path.front());
        return result;
    }
    size_t segment = 1;
    for (int i = 0; i < output_count; ++i) {
        const double target =
            total * static_cast<double>(i) / (output_count - 1);
        while (segment + 1 < arc.size() && arc[segment] < target) {
            ++segment;
        }
        const double segment_length = arc[segment] - arc[segment - 1];
        const double alpha =
            segment_length > kEpsilon
                ? (target - arc[segment - 1]) / segment_length
                : 0.0;
        result.push_back(
            (1.0 - alpha) * path[segment - 1] + alpha * path[segment]);
    }
    return result;
}

std::vector<Eigen::Vector3d> initializeSplineControlPoints(
    const std::vector<Eigen::Vector3d>& path,
    int control_point_count) {
    std::vector<Eigen::Vector3d> points(
        static_cast<size_t>(control_point_count), path.front());
    const std::vector<Eigen::Vector3d> geometry =
        resamplePolyline(path, control_point_count - 2);
    for (int i = 3; i < control_point_count - 3; ++i) {
        points[static_cast<size_t>(i)] =
            geometry[static_cast<size_t>(i - 1)];
    }
    points.back() = path.back();
    points[static_cast<size_t>(control_point_count - 2)] = path.back();
    return points;
}

struct SplineObjective {
    const ESDFGrid& esdf;
    const TrajectoryOptimizationConfig& config;
    std::vector<Eigen::Vector3d> points;
    std::vector<double> knots;
    int free_begin = 3;
    int free_end = 0;
    double duration = 1.0;

    double evaluate(const double* values, double* gradient) {
        for (int i = free_begin; i < free_end; ++i) {
            points[static_cast<size_t>(i)] =
                Eigen::Vector3d(
                    values[3 * (i - free_begin) + 0],
                    values[3 * (i - free_begin) + 1],
                    values[3 * (i - free_begin) + 2]);
        }
        std::vector<Eigen::Vector3d> point_gradient(
            points.size(), Eigen::Vector3d::Zero());
        double cost = 0.0;

        // Weak first-difference term (removes the affine null-space).
        for (size_t i = 0; i + 1 < points.size(); ++i) {
            const Eigen::Vector3d edge = points[i + 1] - points[i];
            cost += config.weight_path_length * edge.squaredNorm();
            const Eigen::Vector3d g =
                2.0 * config.weight_path_length * edge;
            point_gradient[i] -= g;
            point_gradient[i + 1] += g;
        }
        // Smoothness (acceleration) cost.
        for (size_t i = 0; i + 2 < points.size(); ++i) {
            const Eigen::Vector3d acceleration =
                points[i + 2] - 2.0 * points[i + 1] + points[i];
            cost += config.weight_smooth * acceleration.squaredNorm();
            const Eigen::Vector3d g =
                2.0 * config.weight_smooth * acceleration;
            point_gradient[i] += g;
            point_gradient[i + 1] -= 2.0 * g;
            point_gradient[i + 2] += g;
        }
        // Jerk cost.
        for (size_t i = 0; i + 3 < points.size(); ++i) {
            const Eigen::Vector3d jerk =
                points[i + 3] - 3.0 * points[i + 2] +
                3.0 * points[i + 1] - points[i];
            cost += config.weight_jerk * jerk.squaredNorm();
            const Eigen::Vector3d g = 2.0 * config.weight_jerk * jerk;
            point_gradient[i] -= g;
            point_gradient[i + 1] += 3.0 * g;
            point_gradient[i + 2] -= 3.0 * g;
            point_gradient[i + 3] += g;
        }

        // ESDF obstacle cost sampled on the curve.
        const double polygon_length = polylineLength(points);
        const int sample_count = std::max(
            4 * (static_cast<int>(points.size()) - kDegree),
            std::min(
                512,
                static_cast<int>(std::ceil(
                    2.0 * polygon_length /
                    std::max(0.05, config.collision_check_spacing)))));
        const double soft_band = std::max(
            0.05, config.target_clearance - config.min_clearance);
        const double resolution = std::max(0.02, esdf.resolution());
        const double optimization_floor = std::min(
            config.target_clearance, config.min_clearance + 0.05);
        for (int sample = 0; sample <= sample_count; ++sample) {
            const double parameter =
                static_cast<double>(sample) / sample_count;
            const LocalBasis basis = evaluateBasis(
                knots, static_cast<int>(points.size()),
                kDegree, parameter);
            Eigen::Vector3d position = Eigen::Vector3d::Zero();
            for (int j = 0; j <= kDegree; ++j) {
                position += basis.weights[static_cast<size_t>(j)] *
                            points[static_cast<size_t>(basis.first + j)];
            }
            double clearance = 0.0;
            const Eigen::Vector3d esdf_gradient = esdf.getGradient(
                position.x(), position.y(), position.z(), &clearance);
            const double normalization = 1.0 / (sample_count + 1);
            Eigen::Vector3d position_gradient = Eigen::Vector3d::Zero();
            if (std::isfinite(clearance) &&
                clearance < config.target_clearance) {
                const double residual =
                    (config.target_clearance - clearance) / soft_band;
                cost += config.weight_obstacle *
                        residual * residual * normalization;
                position_gradient +=
                    config.weight_obstacle *
                    (-2.0 * residual / soft_band) *
                    esdf_gradient * normalization;
            }
            if (std::isfinite(clearance) &&
                clearance < optimization_floor) {
                const double residual =
                    (optimization_floor - clearance + resolution) /
                    resolution;
                cost += config.weight_obstacle * 1000.0 *
                        residual * residual * normalization;
                position_gradient +=
                    config.weight_obstacle *
                    (-2000.0 * residual / resolution) *
                    esdf_gradient * normalization;
            }
            for (int j = 0; j <= kDegree; ++j) {
                point_gradient[
                    static_cast<size_t>(basis.first + j)] +=
                    basis.weights[static_cast<size_t>(j)] *
                    position_gradient;
            }
        }

        // Analytic dynamics costs from derivative control points.
        const SplineDerivatives derivatives =
            buildDerivatives(points, knots);
        std::vector<Eigen::Vector3d> first_gradient(
            derivatives.first.size(), Eigen::Vector3d::Zero());
        std::vector<Eigen::Vector3d> second_gradient(
            derivatives.second.size(), Eigen::Vector3d::Zero());
        std::vector<Eigen::Vector3d> third_gradient(
            derivatives.third.size(), Eigen::Vector3d::Zero());
        auto addLimitCost = [&](const Eigen::Vector3d& value,
                                double limit,
                                double time_scale,
                                Eigen::Vector3d* derivative_gradient) {
            const Eigen::Vector3d physical = value / time_scale;
            const double magnitude = physical.norm();
            if (magnitude <= limit || magnitude <= kEpsilon) return;
            const double residual =
                (magnitude - limit) / std::max(0.1, limit);
            const double dynamic_weight =
                100.0 * config.weight_dynamics;
            cost += dynamic_weight * residual * residual;
            *derivative_gradient +=
                2.0 * dynamic_weight * residual * physical /
                (magnitude * time_scale * std::max(0.1, limit));
        };
        for (size_t i = 0; i < derivatives.first.size(); ++i) {
            addLimitCost(derivatives.first[i], config.max_velocity,
                         duration, &first_gradient[i]);
        }
        for (size_t i = 0; i < derivatives.second.size(); ++i) {
            addLimitCost(derivatives.second[i], config.max_acceleration,
                         duration * duration, &second_gradient[i]);
        }
        for (size_t i = 0; i < derivatives.third.size(); ++i) {
            addLimitCost(derivatives.third[i], config.max_jerk,
                         duration * duration * duration,
                         &third_gradient[i]);
        }
        for (size_t i = 0; i < third_gradient.size(); ++i) {
            const Eigen::Vector3d contribution =
                derivatives.third_scales[i] * third_gradient[i];
            second_gradient[i] -= contribution;
            second_gradient[i + 1] += contribution;
        }
        for (size_t i = 0; i < second_gradient.size(); ++i) {
            const Eigen::Vector3d contribution =
                derivatives.second_scales[i] * second_gradient[i];
            first_gradient[i] -= contribution;
            first_gradient[i + 1] += contribution;
        }
        for (size_t i = 0; i < first_gradient.size(); ++i) {
            const Eigen::Vector3d contribution =
                derivatives.first_scales[i] * first_gradient[i];
            point_gradient[i] -= contribution;
            point_gradient[i + 1] += contribution;
        }

        if (gradient != nullptr) {
            for (int i = free_begin; i < free_end; ++i) {
                const Eigen::Vector3d& value =
                    point_gradient[static_cast<size_t>(i)];
                gradient[3 * (i - free_begin) + 0] = value.x();
                gradient[3 * (i - free_begin) + 1] = value.y();
                gradient[3 * (i - free_begin) + 2] = value.z();
            }
        }
        return cost;
    }
};

#ifdef IL_DATASET_HAS_NLOPT
double nloptObjective(unsigned dimension,
                      const double* values,
                      double* gradient,
                      void* opaque) {
    auto* objective = static_cast<SplineObjective*>(opaque);
    (void)dimension;
    return objective->evaluate(values, gradient);
}
#endif

bool optimizeSpline(std::vector<Eigen::Vector3d>* points,
                    const std::vector<double>& knots,
                    double duration,
                    const ESDFGrid& esdf,
                    const TrajectoryOptimizationConfig& config,
                    const std::vector<Eigen::Vector3d>& seed_anchors,
                    const Eigen::Vector3d& corridor_min,
                    const Eigen::Vector3d& corridor_max,
                    Clock::time_point deadline) {
    const int free_begin = 3;
    const int free_end = static_cast<int>(points->size()) - 3;
    const int variable_count = 3 * (free_end - free_begin);
    if (variable_count <= 0 || seed_anchors.size() != points->size()) {
        return false;
    }
    SplineObjective objective{
        esdf, config, *points, knots, free_begin, free_end, duration};
    std::vector<double> values(static_cast<size_t>(variable_count), 0.0);
    std::vector<double> lower(static_cast<size_t>(variable_count), 0.0);
    std::vector<double> upper(static_cast<size_t>(variable_count), 0.0);
    for (int i = free_begin; i < free_end; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            const int index = 3 * (i - free_begin) + axis;
            const double anchor =
                seed_anchors[static_cast<size_t>(i)][axis];
            if (config.horizontal_avoidance_only && axis == 2) {
                lower[static_cast<size_t>(index)] = anchor;
                upper[static_cast<size_t>(index)] = anchor;
            } else {
                lower[static_cast<size_t>(index)] = std::max(
                    corridor_min[axis], anchor - config.seed_trust_radius);
                upper[static_cast<size_t>(index)] = std::min(
                    corridor_max[axis], anchor + config.seed_trust_radius);
            }
            if (lower[static_cast<size_t>(index)] >
                upper[static_cast<size_t>(index)]) {
                return false;
            }
            values[static_cast<size_t>(index)] = clamp(
                (*points)[static_cast<size_t>(i)][axis],
                lower[static_cast<size_t>(index)],
                upper[static_cast<size_t>(index)]);
        }
    }

    bool accepted = false;
#ifdef IL_DATASET_HAS_NLOPT
    if (config.optimizer == "auto" || config.optimizer == "nlopt") {
        nlopt_opt optimizer = nlopt_create(
            NLOPT_LD_LBFGS, static_cast<unsigned>(variable_count));
        if (optimizer != nullptr) {
            nlopt_set_min_objective(optimizer, nloptObjective, &objective);
            nlopt_set_lower_bounds(optimizer, lower.data());
            nlopt_set_upper_bounds(optimizer, upper.data());
            nlopt_set_ftol_rel(
                optimizer, std::max(1.0e-7, config.convergence_tolerance));
            nlopt_set_maxeval(
                optimizer, std::max(20, std::min(250, config.max_iterations)));
            const double remaining =
                std::chrono::duration<double>(deadline - Clock::now()).count();
            nlopt_set_maxtime(optimizer, std::max(0.001, remaining));
            double minimum = std::numeric_limits<double>::infinity();
            const nlopt_result status =
                nlopt_optimize(optimizer, values.data(), &minimum);
            accepted = status > 0 || status == NLOPT_MAXTIME_REACHED ||
                       status == NLOPT_MAXEVAL_REACHED;
            nlopt_destroy(optimizer);
        }
    }
#endif
    if (!accepted && Clock::now() < deadline &&
        (config.optimizer == "auto" || config.optimizer == "native")) {
        std::vector<double> gradient(
            static_cast<size_t>(variable_count), 0.0);
        double cost = objective.evaluate(values.data(), gradient.data());
        accepted = std::isfinite(cost);
        constexpr size_t kHistorySize = 8;
        std::deque<std::vector<double>> step_history;
        std::deque<std::vector<double>> gradient_history;
        std::deque<double> inverse_curvature_history;
        const int iterations =
            std::max(20, std::min(250, config.max_iterations));
        for (int iteration = 0;
             iteration < iterations && Clock::now() < deadline;
             ++iteration) {
            double norm_sq = 0.0;
            for (double component : gradient) {
                norm_sq += component * component;
            }
            if (norm_sq <= config.convergence_tolerance *
                               config.convergence_tolerance) {
                accepted = true;
                break;
            }
            std::vector<double> quasi_gradient = gradient;
            std::vector<double> alpha(step_history.size(), 0.0);
            for (size_t reverse = step_history.size();
                 reverse > 0; --reverse) {
                const size_t history_index = reverse - 1;
                double dot = 0.0;
                for (int i = 0; i < variable_count; ++i) {
                    dot += step_history[history_index][static_cast<size_t>(i)] *
                           quasi_gradient[static_cast<size_t>(i)];
                }
                alpha[history_index] =
                    inverse_curvature_history[history_index] * dot;
                for (int i = 0; i < variable_count; ++i) {
                    quasi_gradient[static_cast<size_t>(i)] -=
                        alpha[history_index] *
                        gradient_history[history_index][static_cast<size_t>(i)];
                }
            }
            double initial_hessian_scale = 1.0;
            if (!step_history.empty()) {
                double sy = 0.0;
                double yy = 0.0;
                const auto& last_step = step_history.back();
                const auto& last_gradient = gradient_history.back();
                for (int i = 0; i < variable_count; ++i) {
                    sy += last_step[static_cast<size_t>(i)] *
                          last_gradient[static_cast<size_t>(i)];
                    yy += last_gradient[static_cast<size_t>(i)] *
                          last_gradient[static_cast<size_t>(i)];
                }
                if (sy > 1.0e-12 && yy > 1.0e-12) {
                    initial_hessian_scale = sy / yy;
                }
            }
            std::vector<double> direction(
                static_cast<size_t>(variable_count), 0.0);
            for (int i = 0; i < variable_count; ++i) {
                direction[static_cast<size_t>(i)] =
                    initial_hessian_scale * quasi_gradient[static_cast<size_t>(i)];
            }
            for (size_t history_index = 0;
                 history_index < step_history.size(); ++history_index) {
                double dot = 0.0;
                for (int i = 0; i < variable_count; ++i) {
                    dot += gradient_history[history_index][static_cast<size_t>(i)] *
                           direction[static_cast<size_t>(i)];
                }
                const double beta =
                    inverse_curvature_history[history_index] * dot;
                for (int i = 0; i < variable_count; ++i) {
                    direction[static_cast<size_t>(i)] +=
                        step_history[history_index][static_cast<size_t>(i)] *
                        (alpha[history_index] - beta);
                }
            }
            double directional_derivative = 0.0;
            for (int i = 0; i < variable_count; ++i) {
                direction[static_cast<size_t>(i)] *= -1.0;
                directional_derivative +=
                    gradient[static_cast<size_t>(i)] *
                    direction[static_cast<size_t>(i)];
            }
            if (!(directional_derivative < -1.0e-12)) {
                directional_derivative = -norm_sq;
                for (int i = 0; i < variable_count; ++i) {
                    direction[static_cast<size_t>(i)] = -gradient[static_cast<size_t>(i)];
                }
            }

            bool improved = false;
            double maximum_direction = 0.0;
            for (double component : direction) {
                maximum_direction =
                    std::max(maximum_direction, std::abs(component));
            }
            double step_length =
                maximum_direction > kEpsilon
                    ? std::min(
                          1.0,
                          std::max(config.minimum_step_size,
                                   config.initial_step_size) /
                              maximum_direction)
                    : 1.0;
            for (int line_search = 0;
                 line_search < 12 && Clock::now() < deadline;
                 ++line_search) {
                std::vector<double> candidate = values;
                for (int i = 0; i < variable_count; ++i) {
                    candidate[static_cast<size_t>(i)] = clamp(
                        candidate[static_cast<size_t>(i)] +
                            step_length * direction[static_cast<size_t>(i)],
                        lower[static_cast<size_t>(i)],
                        upper[static_cast<size_t>(i)]);
                }
                std::vector<double> candidate_gradient(
                    static_cast<size_t>(variable_count), 0.0);
                const double candidate_cost = objective.evaluate(
                    candidate.data(), candidate_gradient.data());
                double projected_slope = 0.0;
                for (int i = 0; i < variable_count; ++i) {
                    projected_slope +=
                        gradient[static_cast<size_t>(i)] *
                        (candidate[static_cast<size_t>(i)] -
                         values[static_cast<size_t>(i)]);
                }
                if (std::isfinite(candidate_cost) &&
                    projected_slope < 0.0 &&
                    candidate_cost <= cost + 1.0e-4 * projected_slope) {
                    std::vector<double> actual_step(
                        static_cast<size_t>(variable_count), 0.0);
                    std::vector<double> gradient_change(
                        static_cast<size_t>(variable_count), 0.0);
                    double curvature = 0.0;
                    for (int i = 0; i < variable_count; ++i) {
                        actual_step[static_cast<size_t>(i)] =
                            candidate[static_cast<size_t>(i)] -
                            values[static_cast<size_t>(i)];
                        gradient_change[static_cast<size_t>(i)] =
                            candidate_gradient[static_cast<size_t>(i)] -
                            gradient[static_cast<size_t>(i)];
                        curvature += actual_step[static_cast<size_t>(i)] *
                                     gradient_change[static_cast<size_t>(i)];
                    }
                    if (curvature > 1.0e-10) {
                        if (step_history.size() == kHistorySize) {
                            step_history.pop_front();
                            gradient_history.pop_front();
                            inverse_curvature_history.pop_front();
                        }
                        step_history.push_back(std::move(actual_step));
                        gradient_history.push_back(std::move(gradient_change));
                        inverse_curvature_history.push_back(1.0 / curvature);
                    }
                    values.swap(candidate);
                    gradient.swap(candidate_gradient);
                    cost = candidate_cost;
                    improved = true;
                    accepted = true;
                    break;
                }
                step_length *= 0.5;
            }
            if (!improved || step_length < config.minimum_step_size) {
                break;
            }
        }
    }
    if (!accepted) return false;
    for (int i = free_begin; i < free_end; ++i) {
        (*points)[static_cast<size_t>(i)] =
            Eigen::Vector3d(
                values[3 * (i - free_begin) + 0],
                values[3 * (i - free_begin) + 1],
                values[3 * (i - free_begin) + 2]);
    }
    return true;
}

struct DynamicMetrics {
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
    double yaw_rate = 0.0;
    bool finite = true;
};

DynamicMetrics measureDynamics(
    const std::vector<TrajectoryPoint>& trajectory) {
    DynamicMetrics result;
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& point = trajectory[i];
        result.finite =
            result.finite && point.position.allFinite() &&
            point.velocity.allFinite() && point.acceleration.allFinite();
        result.velocity = std::max(result.velocity, point.velocity.norm());
        result.acceleration =
            std::max(result.acceleration, point.acceleration.norm());
        result.yaw_rate = std::max(result.yaw_rate, std::abs(point.yaw_rate));
        if (i == 0) continue;
        const double dt = point.t - trajectory[i - 1].t;
        if (dt <= kEpsilon) {
            result.finite = false;
            continue;
        }
        result.jerk = std::max(
            result.jerk,
            (point.acceleration - trajectory[i - 1].acceleration).norm() / dt);
    }
    return result;
}

bool dynamicsFeasible(const DynamicMetrics& dynamics,
                      const TrajectoryOptimizationConfig& config) {
    return dynamics.finite &&
           dynamics.velocity <= config.max_velocity * kDynamicsTolerance &&
           dynamics.acceleration <=
               config.max_acceleration * kDynamicsTolerance &&
           dynamics.jerk <= config.max_jerk * kDynamicsTolerance &&
           dynamics.yaw_rate <= config.yaw_max_rate * kDynamicsTolerance;
}

double dynamicsViolationRatio(const DynamicMetrics& dynamics,
                              const TrajectoryOptimizationConfig& config) {
    if (!dynamics.finite) return std::numeric_limits<double>::infinity();
    return std::max({
        dynamics.velocity / std::max(0.1, config.max_velocity),
        dynamics.acceleration / std::max(0.1, config.max_acceleration),
        dynamics.jerk / std::max(0.1, config.max_jerk),
        dynamics.yaw_rate / std::max(0.1, config.yaw_max_rate)
    });
}

double requiredDurationScale(const DynamicMetrics& dynamics,
                             const TrajectoryOptimizationConfig& config) {
    if (!dynamics.finite) return std::numeric_limits<double>::infinity();
    const double velocity_ratio =
        dynamics.velocity / std::max(0.1, config.max_velocity);
    const double acceleration_ratio =
        dynamics.acceleration / std::max(0.1, config.max_acceleration);
    const double jerk_ratio = dynamics.jerk / std::max(0.1, config.max_jerk);
    const double yaw_ratio =
        dynamics.yaw_rate / std::max(0.1, config.yaw_max_rate);
    return std::max({
        1.0,
        velocity_ratio,
        std::sqrt(std::max(0.0, acceleration_ratio)),
        std::cbrt(std::max(0.0, jerk_ratio)),
        yaw_ratio
    });
}

}  // namespace

LocalPlanner::LocalPlanner(const TrajectoryOptimizationConfig& config)
    : config_(config) {}

void LocalPlanner::setMap(const ObservedMap* map) { map_ = map; }

ValidationResult LocalPlanner::validateTrajectorySegmentSpatially(
    const std::vector<TrajectoryPoint>& trajectory,
    double start_t,
    double min_clearance) const {
    ValidationResult result;
    result.all_clear = true;
    if (trajectory.empty()) {
        result.all_clear = false;
        result.any_collision = true;
        return result;
    }
    if (map_ == nullptr || !map_->esdfBuilt()) {
        result.all_clear = false;
        result.any_collision = true;
        result.any_unknown = true;
        return result;
    }
    double min_clear = std::numeric_limits<double>::infinity();

    auto check_point = [&](const TrajectoryPoint& point, bool* clear_ok) {
        if (!point.position.allFinite() || !point.velocity.allFinite() ||
            !point.acceleration.allFinite()) {
            result.any_collision = true;
            result.all_clear = false;
            result.worst_position = point.position;
            result.worst_time = point.t;
            *clear_ok = false;
            return;
        }
        const bool known =
            map_->isKnown(point.position.x(), point.position.y(),
                          point.position.z());
        const double clearance =
            map_->esdfValue(point.position.x(), point.position.y(),
                            point.position.z());
        if (std::isfinite(clearance)) {
            min_clear = std::min(min_clear, clearance);
        }
        if (!known || !std::isfinite(clearance) ||
            clearance <= min_clearance) {
            result.any_collision = true;
            result.all_clear = false;
            result.clearance_violation_count++;
            if (clearance < result.worst_clearance) {
                result.worst_clearance = clearance;
                result.worst_position = point.position;
                result.worst_time = point.t;
            }
            if (!known) result.any_unknown = true;
            *clear_ok = false;
        }
    };

    for (size_t i = 0; i < trajectory.size(); ++i) {
        if (trajectory[i].t < start_t - 1.0e-6) continue;
        bool ok = true;
        check_point(trajectory[i], &ok);
        if (!ok) return result;
        // Spatial interpolation (sections XVI/XIX): the maximum
        // collision-check spacing along the trajectory must be <=
        // collision_check_spacing, regardless of the output time step (dt).
        if (i + 1 < trajectory.size()) {
            const Eigen::Vector3d seg =
                trajectory[i + 1].position - trajectory[i].position;
            const double seg_len = seg.norm();
            const double spacing =
                std::max(0.02, config_.collision_check_spacing);
            if (seg_len > spacing) {
                const int steps = static_cast<int>(std::ceil(seg_len / spacing));
                for (int k = 1; k < steps; ++k) {
                    TrajectoryPoint interp;
                    const double alpha = static_cast<double>(k) / steps;
                    interp.t = trajectory[i].t +
                               alpha * (trajectory[i + 1].t - trajectory[i].t);
                    interp.position =
                        trajectory[i].position + alpha * seg;
                    bool interp_ok = true;
                    check_point(interp, &interp_ok);
                    if (!interp_ok) return result;
                }
            }
        }
    }
    result.min_clearance =
        std::isfinite(min_clear) ? min_clear : 0.0;
    return result;
}

ValidationResult LocalPlanner::validateTrajectory(
    const std::vector<TrajectoryPoint>& trajectory) const {
    if (trajectory.empty()) {
        ValidationResult result;
        result.all_clear = false;
        result.any_collision = true;
        return result;
    }
    return validateTrajectorySegmentSpatially(
        trajectory, trajectory.front().t, config_.min_clearance);
}

ValidationResult LocalPlanner::validateTrajectorySuffix(
    const std::vector<TrajectoryPoint>& trajectory,
    double plan_start_time,
    double current_time,
    const VehicleState& state,
    double min_clearance,
    double max_position_error,
    double max_velocity_error) const {
    ValidationResult result;
    result.all_clear = true;
    if (map_ == nullptr || !map_->esdfBuilt()) {
        result.all_clear = false;
        result.any_collision = true;
        result.any_unknown = true;
        return result;
    }
    if (trajectory.empty() || !std::isfinite(plan_start_time) ||
        !std::isfinite(current_time)) {
        result.all_clear = false;
        result.any_collision = true;
        return result;
    }
    const double elapsed = current_time - plan_start_time;
    const double age = std::max(0.0, elapsed);

    // Interpolate the reference position / velocity AT the current age
    // (section X) — the drone is re-executing the REMAINING segment, so
    // the state must match the trajectory sampled at `age`, not at t=0.
    auto sample_at = [&](double t, Eigen::Vector3d* pos,
                         Eigen::Vector3d* vel) {
        if (t <= trajectory.front().t) {
            *pos = trajectory.front().position;
            *vel = trajectory.front().velocity;
            return;
        }
        if (t >= trajectory.back().t) {
            *pos = trajectory.back().position;
            *vel = trajectory.back().velocity;
            return;
        }
        for (size_t i = 0; i + 1 < trajectory.size(); ++i) {
            if (trajectory[i + 1].t < t) continue;
            const TrajectoryPoint& a = trajectory[i];
            const TrajectoryPoint& b = trajectory[i + 1];
            const double dt = std::max(1.0e-6, b.t - a.t);
            const double alpha = std::max(0.0, std::min(1.0, (t - a.t) / dt));
            *pos = (1.0 - alpha) * a.position + alpha * b.position;
            *vel = (1.0 - alpha) * a.velocity + alpha * b.velocity;
            return;
        }
    };
    Eigen::Vector3d ref_pos, ref_vel;
    sample_at(age, &ref_pos, &ref_vel);
    const double position_error = (state.position - ref_pos).norm();
    const double velocity_error = (state.velocity - ref_vel).norm();
    if (position_error > max_position_error ||
        velocity_error > max_velocity_error) {
        result.all_clear = false;
        result.any_collision = true;
        result.worst_position = state.position;
        result.worst_time = age;
        return result;
    }

    // Safety validation from the current age (section XX), with the SAME
    // spatial interpolation strictness as the fresh trajectory.
    return validateTrajectorySegmentSpatially(trajectory, age, min_clearance);
}

LocalPlanResult LocalPlanner::plan(const LocalPlanRequest& request) const {
    const auto planning_start = Clock::now();
    const auto planning_deadline =
        planning_start + std::chrono::duration_cast<Clock::duration>(
                             std::chrono::duration<double, std::milli>(
                                 std::max(1.0, config_.planning_time_budget_ms)));
    LocalPlanResult result;
    result.plan_id = plan_id_counter_++;
    result.guide_waypoint = request.macro_guide_world;

    auto finish = [&]() {
        result.planning_time_ms =
            std::chrono::duration<double, std::milli>(Clock::now() -
                                                      planning_start)
                .count();
        return result;
    };

    if (map_ == nullptr || !map_->esdfBuilt()) {
        result.status = PlannerStatus::NO_SAFE_MOTION;
        result.message = "observed_map_not_built";
        return finish();
    }
    const VehicleState& state = request.state;
    if (!state.position.allFinite() || !state.velocity.allFinite() ||
        !state.acceleration.allFinite() ||
        !request.macro_guide_world.allFinite()) {
        result.status = PlannerStatus::INVALID_INPUT;
        result.message = "planning_request_contains_nan_or_inf";
        return finish();
    }

    // ── 1. Observed-map A* seed path ──────────────────────────────
    LocalSearchConfig search_config;
    search_config.search_clearance_m = config_.search_clearance_m;
    search_config.committed_side = request.committed_side;
    search_config.side_bias_gain = config_.search_side_bias_gain;
    search_config.max_time_ms = config_.search_max_time_ms;
    search_config.region_margin_m = config_.search_region_margin_m;
    search_config.forbid_unknown = request.forbid_unknown_space;
    LocalPathSearch search;
    const LocalPathResult search_result =
        search.search(*map_, state, request.macro_guide_world, search_config);
    if (!search_result.found_partial) {
        result.status = PlannerStatus::SEARCH_FAILED;
        result.message = search_result.failure_reason.empty()
                             ? "no_local_path"
                             : search_result.failure_reason;
        result.search_status = 2;
        return finish();
    }
    result.search_status = search_result.full_goal_reached ? 0 : 1;
    Eigen::Vector3d terminal = search_result.terminal;

    // ── 2. Seed path (A* + warm start) ────────────────────────────
    std::vector<Eigen::Vector3d> seed_path = search_result.path;
    std::string seed_source = "astar";
    if (!request.previous_trajectory.empty() &&
        request.previous_trajectory_age_s <= config_.warm_start_max_age_s) {
        // Extract the known-free suffix of the previous trajectory and use
        // it as a warm start when it still ends near the new terminal.
        std::vector<Eigen::Vector3d> suffix;
        bool suffix_valid = true;
        for (const TrajectoryPoint& point : request.previous_trajectory) {
            if (point.t < request.previous_trajectory_age_s) continue;
            if (!map_->isKnownFree(point.position.x(), point.position.y(),
                                   point.position.z(),
                                   config_.search_clearance_m)) {
                suffix_valid = false;
                break;
            }
            suffix.push_back(point.position);
        }
        if (suffix_valid && suffix.size() >= 2) {
            const double deviation =
                (suffix.back() - terminal).head<2>().norm();
            if (deviation <= config_.warm_start_max_terminal_deviation_m) {
                // Replace the start with the current state and prepend the
                // A* start segment so the seed is continuous.
                suffix.front() = state.position;
                seed_path = std::move(suffix);
                seed_source = "warm_start";
            }
        }
    }

    // ── 3. Truncate the local terminal to the planning horizon ────
    // The local terminal must be executable within the fixed local
    // horizon (section III / X.3): walk the seed path and keep only the
    // farthest point within horizon_time * cruise speed.
    {
        const double horizon_budget =
            std::max(1.0, config_.horizon_time * config_.nominal_speed);
        double accumulated = 0.0;
        Eigen::Vector3d trunc = state.position;
        for (size_t i = 1; i < seed_path.size(); ++i) {
            const double segment = (seed_path[i] - seed_path[i - 1]).norm();
            if (accumulated + segment > horizon_budget) {
                const double alpha =
                    segment > kEpsilon
                        ? (horizon_budget - accumulated) / segment
                        : 1.0;
                trunc = (1.0 - alpha) * seed_path[i - 1] +
                        alpha * seed_path[i];
                break;
            }
            accumulated += segment;
            trunc = seed_path[i];
        }
        if ((trunc - seed_path.front()).norm() < 0.3) {
            // The horizon cannot even fit a short move; keep the A*
            // terminal (already the farthest known-safe point).
            trunc = terminal;
        }
        terminal = trunc;
        // Rebuild the seed path by arc length, ending at the truncated
        // terminal.
        std::vector<Eigen::Vector3d> trimmed;
        trimmed.reserve(seed_path.size());
        trimmed.push_back(state.position);
        double arc = 0.0;
        for (size_t i = 1; i < seed_path.size(); ++i) {
            const double segment = (seed_path[i] - seed_path[i - 1]).norm();
            if (arc + segment > horizon_budget + 1.0e-3) break;
            arc += segment;
            trimmed.push_back(seed_path[i]);
        }
        if (trimmed.size() < 2 ||
            (trimmed.back() - terminal).norm() > 0.3) {
            trimmed.push_back(terminal);
        }
        seed_path = std::move(trimmed);
    }
    result.trajectory_terminal = terminal;

    // ── 4. B-spline geometry ──────────────────────────────────────
    const int control_point_count =
        std::max(8, config_.control_points);
    const std::vector<double> knots =
        makeClampedKnots(control_point_count, kDegree);
    const Eigen::Vector3d displacement = terminal - state.position;
    const double distance = displacement.norm();
    if (distance < 1.0e-3) {
        result.status = PlannerStatus::LOCAL_TERMINAL_INVALID;
        result.message = "local_terminal_indistinguishable_from_position";
        return finish();
    }
    const Eigen::Vector3d direction = displacement / distance;

    double terminal_speed = 0.0;
    Eigen::Vector3d terminal_velocity = Eigen::Vector3d::Zero();
    // Stop-at-goal is decided INTERNALLY (section XVII): the search fully
    // reached the goal (not just a partial terminal), the terminal is
    // within `goal_stop_tolerance_m` of the goal, and the goal is known
    // free in the current map.
    const bool stop_at_goal =
        search_result.full_goal_reached &&
        (terminal - request.goal_world).head<2>().norm() <=
            config_.goal_stop_tolerance_m &&
        map_->isKnownFree(request.goal_world.x(), request.goal_world.y(),
                          request.goal_world.z(), config_.min_clearance);
    if (!stop_at_goal) {
        const double distance_ratio = clamp(
            distance / std::max(0.5, config_.lookahead_distance), 0.55, 1.0);
        terminal_speed =
            config_.nominal_speed * distance_ratio *
            config_.terminal_speed_ratio;
        terminal_velocity = terminal_speed * direction;
    }
    Eigen::Vector3d start_acceleration = state.acceleration;
    if (start_acceleration.norm() > config_.max_acceleration) {
        start_acceleration *=
            config_.max_acceleration / start_acceleration.norm();
    }
    const double travel_speed =
        std::max(0.5, config_.nominal_speed *
                          clamp(distance / std::max(0.5, config_.lookahead_distance),
                                0.55, 1.0));
    double duration = std::max({
        1.0,
        distance / travel_speed,
        std::sqrt(12.0 * distance / std::max(0.1, config_.max_acceleration))
    });
    const bool has_macro_yaw = request.has_macro_yaw;
    const double macro_yaw = request.macro_yaw_world;
    const double planned_target_yaw =
        has_macro_yaw ? wrapAngleLocal(macro_yaw)
                      : (displacement.head<2>().norm() > 1.0e-6
                             ? yawFromVelocity(displacement)
                             : state.yaw);
    duration = std::max(
        duration,
        1.05 * std::abs(wrapAngleLocal(planned_target_yaw - state.yaw)) /
            std::max(0.1, config_.yaw_max_rate));
    duration = std::max(duration,
                        polylineLength(seed_path) /
                            std::max(0.5, config_.nominal_speed));

    const std::vector<Eigen::Vector3d> seed_control_points =
        initializeSplineControlPoints(seed_path, control_point_count);
    std::vector<Eigen::Vector3d> control_points = seed_control_points;
    imposeBoundaryState(&control_points, knots, duration, state.position,
                        state.velocity, start_acceleration, terminal,
                        terminal_velocity);

    Eigen::Vector3d corridor_min = state.position.cwiseMin(terminal);
    Eigen::Vector3d corridor_max = state.position.cwiseMax(terminal);
    for (const Eigen::Vector3d& point : seed_path) {
        corridor_min = corridor_min.cwiseMin(point);
        corridor_max = corridor_max.cwiseMax(point);
    }
    corridor_min.array() -= config_.seed_trust_radius;
    corridor_max.array() += config_.seed_trust_radius;
    corridor_min.x() = std::max(
        corridor_min.x(), map_->origin().x() + 0.5 * map_->resolution());
    corridor_min.y() = std::max(
        corridor_min.y(), map_->origin().y() + 0.5 * map_->resolution());
    corridor_min.z() = std::max(
        corridor_min.z(), map_->origin().z() + 0.5 * map_->resolution());
    corridor_max.x() = std::min(
        corridor_max.x(),
        map_->origin().x() + (map_->gx() - 0.5) * map_->resolution());
    corridor_max.y() = std::min(
        corridor_max.y(),
        map_->origin().y() + (map_->gy() - 0.5) * map_->resolution());
    corridor_max.z() = std::min(
        corridor_max.z(),
        map_->origin().z() + (map_->gz() - 0.5) * map_->resolution());

    ESDFGrid esdf;
    esdf.setMap(map_);

    bool optimized = optimizeSpline(
        &control_points, knots, duration, esdf, config_,
        seed_control_points, corridor_min, corridor_max,
        planning_deadline - std::chrono::milliseconds(3));
    imposeBoundaryState(&control_points, knots, duration, state.position,
                        state.velocity, start_acceleration, terminal,
                        terminal_velocity);
    if (!optimized) {
        result.status = PlannerStatus::OPTIMIZATION_FAILED;
        result.message = "gradient_optimization_no_incumbent";
        return finish();
    }

    auto sampleTrajectory = [&](const std::vector<Eigen::Vector3d>& points,
                                double traj_duration) {
        std::vector<TrajectoryPoint> trajectory;
        const SplineDerivatives derivatives = buildDerivatives(points, knots);
        const int sample_count = std::max(
            2, static_cast<int>(std::ceil(traj_duration / config_.trajectory_dt)) +
                   1);
        trajectory.reserve(static_cast<size_t>(sample_count));
        for (int i = 0; i < sample_count; ++i) {
            const double time = i + 1 == sample_count
                                    ? traj_duration
                                    : std::min(traj_duration, i * config_.trajectory_dt);
            const double parameter = clamp(time / traj_duration, 0.0, 1.0);
            TrajectoryPoint point;
            point.t = time;
            point.position = evaluateSpline(points, knots, kDegree, parameter);
            point.velocity =
                evaluateSpline(derivatives.first, derivatives.first_knots,
                               kDegree - 1, parameter) /
                traj_duration;
            point.acceleration =
                evaluateSpline(derivatives.second, derivatives.second_knots,
                               kDegree - 2, parameter) /
                (traj_duration * traj_duration);
            point.clearance = esdf.getValue(
                point.position.x(), point.position.y(), point.position.z());
            trajectory.push_back(point);
        }
        return trajectory;
    };

    std::vector<TrajectoryPoint> trajectory =
        sampleTrajectory(control_points, duration);
    DynamicMetrics dynamics = measureDynamics(trajectory);

    // ── Iterative dynamic retiming ────────────────────────────────
    for (int time_iteration = 0; time_iteration < 6; ++time_iteration) {
        const double violation_ratio =
            dynamicsViolationRatio(dynamics, config_);
        const double dynamic_scale = requiredDurationScale(dynamics, config_);
        if (violation_ratio <= kDynamicsTolerance ||
            Clock::now() + std::chrono::milliseconds(4) >= planning_deadline) {
            break;
        }
        duration *= std::max(1.03, 1.03 * dynamic_scale);
        terminal_velocity =
            std::min(terminal_speed,
                     polylineLength(seed_path) / std::max(0.1, duration)) *
            direction;
        control_points = seed_control_points;
        imposeBoundaryState(&control_points, knots, duration, state.position,
                            state.velocity, start_acceleration, terminal,
                            terminal_velocity);
        optimizeSpline(&control_points, knots, duration, esdf, config_,
                       seed_control_points, corridor_min, corridor_max,
                       planning_deadline - std::chrono::milliseconds(1));
        imposeBoundaryState(&control_points, knots, duration, state.position,
                            state.velocity, start_acceleration, terminal,
                            terminal_velocity);
        trajectory = sampleTrajectory(control_points, duration);
        const ValidationResult time_validation =
            validateTrajectory(trajectory);
        if (time_validation.any_collision) {
            optimizeSpline(&control_points, knots, duration, esdf, config_,
                           seed_control_points, corridor_min, corridor_max,
                           planning_deadline - std::chrono::milliseconds(1));
            imposeBoundaryState(&control_points, knots, duration,
                                state.position, state.velocity,
                                start_acceleration, terminal,
                                terminal_velocity);
            trajectory = sampleTrajectory(control_points, duration);
            if (validateTrajectory(trajectory).any_collision) break;
        }
        dynamics = measureDynamics(trajectory);
    }

    // ── Yaw planning (FOV-constrained) ────────────────────────────
    YawPlannerConfig yaw_config;
    yaw_config.max_yaw_rate = config_.yaw_max_rate;
    yaw_config.max_yaw_accel = config_.yaw_max_accel;
    yaw_config.fov_half_deg = config_.yaw_fov_half_deg;
    yaw_config.fov_margin_deg = config_.yaw_fov_margin_deg;
    yaw_config.speed_threshold_mps = config_.yaw_speed_threshold_mps;
    YawPlanner yaw_planner(yaw_config);
    yaw_planner.planYaw(&trajectory, state.yaw, planned_target_yaw,
                        has_macro_yaw);

    // ── Strict validation ─────────────────────────────────────────
    const ValidationResult validation = validateTrajectory(trajectory);
    result.min_clearance = validation.min_clearance;
    result.duration_s = duration;
    if (validation.any_unknown || validation.any_collision) {
        result.status = validation.any_unknown ? PlannerStatus::UNKNOWN_SPACE
                                               : PlannerStatus::COLLISION;
        std::ostringstream message;
        message << (validation.any_unknown ? "trajectory_enters_unknown"
                                           : "trajectory_collides")
                << ": min_clearance=" << validation.min_clearance
                << " violations=" << validation.clearance_violation_count;
        result.message = message.str();
        return finish();
    }
    dynamics = measureDynamics(trajectory);
    if (!dynamicsFeasible(dynamics, config_)) {
        result.status = PlannerStatus::DYNAMICS_VIOLATION;
        std::ostringstream message;
        message << "dynamics_violation: v=" << dynamics.velocity
                << " a=" << dynamics.acceleration
                << " jerk=" << dynamics.jerk
                << " yaw_rate=" << dynamics.yaw_rate;
        result.message = message.str();
        return finish();
    }
    result.success = true;
    result.status = PlannerStatus::SUCCESS;
    result.trajectory = std::move(trajectory);
    result.message = seed_source == "warm_start" ? "ok_warm_start" : "ok_astar_seed";
    return finish();
}

}  // namespace il_dataset
