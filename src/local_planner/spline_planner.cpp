#include "il_dataset/local_planner/local_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <queue>
#include <sstream>
#include <utility>
#include <vector>

#ifdef IL_DATASET_HAS_NLOPT
#include <nlopt.h>
#endif

namespace il_dataset {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int kDegree = 3;
constexpr double kEpsilon = 1.0e-9;

double clamp(double value, double lo, double hi) {
    return std::max(lo, std::min(hi, value));
}

double wrapAngleLocal(double angle) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kTwoPi = 2.0 * kPi;
    angle = std::fmod(angle, kTwoPi);
    if (angle > kPi) angle -= kTwoPi;
    if (angle < -kPi) angle += kTwoPi;
    return angle;
}

double yawFromVelocity(const Eigen::Vector3d& velocity) {
    constexpr double kPi = 3.14159265358979323846;
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
    if (degree <= 0 || control_points.size() == 1) {
        const int index = std::max(
            0, std::min(
                static_cast<int>(control_points.size()) - 1,
                findSpan(
                    knots, static_cast<int>(control_points.size()),
                    0, parameter)));
        return control_points[static_cast<size_t>(index)];
    }
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
            result.first_knots[
                static_cast<size_t>(i + first_degree + 1)] -
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
            result.second_knots[
                static_cast<size_t>(i + second_degree + 1)] -
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
        (e0_denominator / static_cast<double>(kDegree - 1)) *
            desired_e0;
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
        knots[static_cast<size_t>(
            previous_derivative_index + kDegree + 1)] -
        knots[static_cast<size_t>(previous_derivative_index + 1)];
    // Zero terminal acceleration: the final two first-derivative control
    // points are equal.  This removes the endpoint acceleration spike while
    // preserving a non-zero velocity at a moving Guide.
    points[static_cast<size_t>(count - 3)] =
        points[static_cast<size_t>(count - 2)] -
        (previous_end_denominator / kDegree) *
            duration * end_velocity;
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
    // P0/P1/P2 and P[n-3]/P[n-2]/P[n-1] are boundary-state control points.
    // With S=N-3 spans, P2 already lies near path fraction 1/S.  P3 must
    // therefore continue near 2/S.  Reusing 1/S duplicates P2, creates an
    // artificial acceleration spike and makes time allocation over-stretch.
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

bool segmentClear(const ESDFGrid& esdf,
                  const Eigen::Vector3d& from,
                  const Eigen::Vector3d& to,
                  double clearance,
                  bool forbid_unknown,
                  double spacing) {
    const double distance = (to - from).norm();
    const int samples = std::max(
        1, static_cast<int>(std::ceil(distance / std::max(0.02, spacing))));
    for (int i = 0; i <= samples; ++i) {
        const double alpha = static_cast<double>(i) / samples;
        const Eigen::Vector3d point = (1.0 - alpha) * from + alpha * to;
        const bool free = forbid_unknown && esdf.hasKnownMask()
            ? esdf.isKnownFree(
                  point.x(), point.y(), point.z(), clearance)
            : esdf.isFree(
                  point.x(), point.y(), point.z(), clearance);
        if (!free) return false;
    }
    return true;
}

struct SearchEntry {
    double score = 0.0;
    int index = -1;
    bool operator<(const SearchEntry& other) const {
        return score > other.score;
    }
};

std::vector<Eigen::Vector3d> searchLocalSeed(
    const ESDFGrid& esdf,
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& start_velocity,
    const Eigen::Vector3d& goal,
    const LocalPlannerConfig& config,
    bool forbid_unknown,
    Clock::time_point deadline) {
    const double search_resolution =
        std::max(0.20, esdf.resolution());
    const double expansion =
        std::min(2.0, std::max(1.0, 0.4 * config.local_map_radius));
    Eigen::Vector3d minimum = start.cwiseMin(goal);
    Eigen::Vector3d maximum = start.cwiseMax(goal);
    minimum.array() -= expansion;
    maximum.array() += expansion;
    minimum.z() = std::max(
        minimum.z(),
        esdf.originZ() + 0.5 * esdf.resolution());
    maximum.x() = std::min(
        maximum.x(),
        esdf.originX() + (esdf.gx() - 0.5) * esdf.resolution());
    maximum.y() = std::min(
        maximum.y(),
        esdf.originY() + (esdf.gy() - 0.5) * esdf.resolution());
    maximum.z() = std::min(
        maximum.z(),
        esdf.originZ() + (esdf.gz() - 0.5) * esdf.resolution());
    minimum.x() = std::max(
        minimum.x(), esdf.originX() + 0.5 * esdf.resolution());
    minimum.y() = std::max(
        minimum.y(), esdf.originY() + 0.5 * esdf.resolution());

    const Eigen::Array3i dimensions =
        ((maximum - minimum).array() / search_resolution)
            .floor().cast<int>() + 1;
    if ((dimensions <= 1).any()) return {};
    const int nx = dimensions.x();
    const int ny = dimensions.y();
    const int nz = dimensions.z();
    const size_t total =
        static_cast<size_t>(nx) * ny * nz;
    if (total > 250000) return {};

    auto encode = [ny, nz](int ix, int iy, int iz) {
        return (ix * ny + iy) * nz + iz;
    };
    auto decode = [ny, nz](int index) {
        Eigen::Vector3i value;
        value.z() = index % nz;
        index /= nz;
        value.y() = index % ny;
        value.x() = index / ny;
        return value;
    };
    auto position = [&](const Eigen::Vector3i& index) {
        return minimum +
               search_resolution * index.cast<double>();
    };
    auto nearestIndex = [&](const Eigen::Vector3d& point) {
        Eigen::Array3i index =
            ((point - minimum).array() / search_resolution)
                .round().cast<int>();
        index = index.max(Eigen::Array3i::Zero());
        index = index.min(dimensions - 1);
        return Eigen::Vector3i(index.matrix());
    };

    const Eigen::Vector3i start_grid = nearestIndex(start);
    const Eigen::Vector3i goal_grid = nearestIndex(goal);
    const int start_index =
        encode(start_grid.x(), start_grid.y(), start_grid.z());
    const int goal_index =
        encode(goal_grid.x(), goal_grid.y(), goal_grid.z());
    const double search_clearance =
        config.min_clearance +
        std::min(0.15, 0.75 * std::max(
            0.0, config.target_clearance - config.min_clearance));
    const Eigen::Vector3d travel = goal - start;
    const double travel_length_sq = travel.squaredNorm();
    Eigen::Vector3d travel_direction = Eigen::Vector3d::UnitX();
    if (travel_length_sq > kEpsilon) {
        travel_direction = travel / std::sqrt(travel_length_sq);
    }
    Eigen::Vector3d preferred_lateral =
        start_velocity -
        start_velocity.dot(travel_direction) * travel_direction;
    if (preferred_lateral.norm() > 0.05) {
        preferred_lateral.normalize();
    } else {
        preferred_lateral.setZero();
    }

    std::vector<double> cost(
        total, std::numeric_limits<double>::infinity());
    std::vector<int> parent(total, -1);
    std::vector<uint8_t> closed(total, 0);
    std::priority_queue<SearchEntry> open;
    cost[static_cast<size_t>(start_index)] = 0.0;
    open.push({(goal - start).norm(), start_index});

    bool found = false;
    int expansions = 0;
    while (!open.empty() && Clock::now() < deadline &&
           expansions < 120000) {
        const SearchEntry current = open.top();
        open.pop();
        if (closed[static_cast<size_t>(current.index)] != 0) continue;
        closed[static_cast<size_t>(current.index)] = 1;
        ++expansions;
        if (current.index == goal_index) {
            found = true;
            break;
        }
        const Eigen::Vector3i grid = decode(current.index);
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const Eigen::Vector3i next =
                        grid + Eigen::Vector3i(dx, dy, dz);
                    if (next.x() < 0 || next.x() >= nx ||
                        next.y() < 0 || next.y() >= ny ||
                        next.z() < 0 || next.z() >= nz) {
                        continue;
                    }
                    const int next_index =
                        encode(next.x(), next.y(), next.z());
                    if (closed[static_cast<size_t>(next_index)] != 0) {
                        continue;
                    }
                    const Eigen::Vector3d next_position = position(next);
                    double clearance = esdf.getValue(
                        next_position.x(), next_position.y(),
                        next_position.z());
                    const bool endpoint =
                        next_index == start_index ||
                        next_index == goal_index;
                    const bool known =
                        !forbid_unknown || !esdf.hasKnownMask() ||
                        esdf.isKnown(
                            next_position.x(), next_position.y(),
                            next_position.z());
                    if (!known ||
                        (!endpoint && clearance <= search_clearance)) {
                        continue;
                    }
                    const double step =
                        search_resolution *
                        std::sqrt(
                            static_cast<double>(
                                dx * dx + dy * dy + dz * dz));
                    const double normalized_deficit =
                        std::max(
                            0.0,
                            config.target_clearance - clearance) /
                        std::max(0.05, config.target_clearance);
                    double side_switch_penalty = 0.0;
                    if (preferred_lateral.squaredNorm() > 0.5 &&
                        travel_length_sq > kEpsilon) {
                        const double progress = clamp(
                            (next_position - start).dot(travel) /
                                travel_length_sq,
                            0.0, 1.0);
                        const Eigen::Vector3d chord_point =
                            start + progress * travel;
                        const double signed_lateral =
                            (next_position - chord_point).dot(
                                preferred_lateral);
                        side_switch_penalty =
                            4.0 * std::max(0.0, -signed_lateral) /
                            std::max(search_resolution, expansion);
                    }
                    const double tentative =
                        cost[static_cast<size_t>(current.index)] +
                        step * (1.0 +
                                1.5 * normalized_deficit *
                                    normalized_deficit +
                                side_switch_penalty);
                    if (tentative >=
                        cost[static_cast<size_t>(next_index)]) {
                        continue;
                    }
                    cost[static_cast<size_t>(next_index)] = tentative;
                    parent[static_cast<size_t>(next_index)] =
                        current.index;
                    const double heuristic =
                        (goal - next_position).norm();
                    open.push({tentative + heuristic, next_index});
                }
            }
        }
    }
    if (!found) return {};

    std::vector<Eigen::Vector3d> reverse_path;
    bool reached_start = false;
    for (int index = goal_index; index >= 0;
         index = parent[static_cast<size_t>(index)]) {
        reverse_path.push_back(position(decode(index)));
        if (index == start_index) {
            reached_start = true;
            break;
        }
    }
    if (reverse_path.empty() || !reached_start) return {};
    std::reverse(reverse_path.begin(), reverse_path.end());
    reverse_path.front() = start;
    reverse_path.back() = goal;

    std::vector<Eigen::Vector3d> shortcut_path;
    shortcut_path.push_back(start);
    size_t anchor = 0;
    while (anchor + 1 < reverse_path.size()) {
        size_t next = reverse_path.size() - 1;
        while (next > anchor + 1 &&
               !segmentClear(
                   esdf, reverse_path[anchor], reverse_path[next],
                   search_clearance, forbid_unknown,
                   0.5 * search_resolution)) {
            --next;
        }
        shortcut_path.push_back(reverse_path[next]);
        anchor = next;
    }
    std::vector<Eigen::Vector3d> dense_path;
    dense_path.push_back(start);
    for (size_t segment_index = 1;
         segment_index < shortcut_path.size(); ++segment_index) {
        const Eigen::Vector3d from = shortcut_path[segment_index - 1];
        const Eigen::Vector3d to = shortcut_path[segment_index];
        const int steps = std::max(
            1, static_cast<int>(std::ceil(
                (to - from).norm() / search_resolution)));
        for (int step = 1; step <= steps; ++step) {
            dense_path.push_back(
                from +
                (static_cast<double>(step) / steps) * (to - from));
        }
    }
    reverse_path.swap(dense_path);

    // Remove voxel-grid stair steps without changing homotopy.  Every
    // Laplacian update is accepted only when both adjacent segments retain
    // the search clearance, so this cannot recreate the corner-cutting
    // shortcut that the A* seed is meant to prevent.
    for (int smoothing_iteration = 0;
         smoothing_iteration < 20; ++smoothing_iteration) {
        bool changed = false;
        std::vector<Eigen::Vector3d> smoothed = reverse_path;
        for (size_t i = 1; i + 1 < reverse_path.size(); ++i) {
            const Eigen::Vector3d candidate =
                0.25 * reverse_path[i - 1] +
                0.50 * reverse_path[i] +
                0.25 * reverse_path[i + 1];
            if (segmentClear(
                    esdf, reverse_path[i - 1], candidate,
                    search_clearance, forbid_unknown,
                    0.5 * search_resolution) &&
                segmentClear(
                    esdf, candidate, reverse_path[i + 1],
                    search_clearance, forbid_unknown,
                    0.5 * search_resolution)) {
                smoothed[i] = candidate;
                changed = changed ||
                    (candidate - reverse_path[i]).norm() > 1.0e-4;
            }
        }
        reverse_path.swap(smoothed);
        if (!changed) break;
    }
    return reverse_path;
}

struct SplineObjective {
    const ESDFGrid& esdf;
    const LocalPlannerConfig& config;
    std::vector<Eigen::Vector3d> points;
    std::vector<double> knots;
    int free_begin = 3;
    int free_end = 0;
    double duration = 1.0;
    bool forbid_unknown = false;

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

        // A weak first-difference term removes the affine null-space of the
        // second/third-difference costs.  It discourages needless detours and
        // loops without attracting the curve to the straight Guide chord.
        for (size_t i = 0; i + 1 < points.size(); ++i) {
            const Eigen::Vector3d edge = points[i + 1] - points[i];
            cost += config.weight_path_length * edge.squaredNorm();
            const Eigen::Vector3d g =
                2.0 * config.weight_path_length * edge;
            point_gradient[i] -= g;
            point_gradient[i + 1] += g;
        }

        // Correct second- and third-order B-spline control-point differences.
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
        for (size_t i = 0; i + 3 < points.size(); ++i) {
            const Eigen::Vector3d jerk =
                points[i + 3] - 3.0 * points[i + 2] +
                3.0 * points[i + 1] - points[i];
            cost += config.weight_jerk * jerk.squaredNorm();
            const Eigen::Vector3d g =
                2.0 * config.weight_jerk * jerk;
            point_gradient[i] -= g;
            point_gradient[i + 1] += 3.0 * g;
            point_gradient[i + 2] -= 3.0 * g;
            point_gradient[i + 3] += g;
        }

        // ESDF collision objective evaluated on the exact curve being sampled.
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
            if (clearance < config.target_clearance) {
                const double residual =
                    (config.target_clearance - clearance) / soft_band;
                cost += config.weight_obstacle *
                        residual * residual * normalization;
                position_gradient +=
                    config.weight_obstacle *
                    (-2.0 * residual / soft_band) *
                    esdf_gradient * normalization;
            }
            if (clearance < optimization_floor) {
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
            if (forbid_unknown && esdf.hasKnownMask() &&
                !esdf.isKnown(position.x(), position.y(), position.z())) {
                cost += 1.0e4 * normalization;
            }
            for (int j = 0; j <= kDegree; ++j) {
                point_gradient[
                    static_cast<size_t>(basis.first + j)] +=
                    basis.weights[static_cast<size_t>(j)] *
                    position_gradient;
            }
        }

        // B-spline derivative control points provide analytic dynamic costs.
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
                2.0 * dynamic_weight * residual *
                physical /
                (magnitude * time_scale * std::max(0.1, limit));
        };
        for (size_t i = 0; i < derivatives.first.size(); ++i) {
            addLimitCost(
                derivatives.first[i], config.max_velocity,
                duration, &first_gradient[i]);
        }
        for (size_t i = 0; i < derivatives.second.size(); ++i) {
            addLimitCost(
                derivatives.second[i], config.max_acceleration,
                duration * duration, &second_gradient[i]);
        }
        for (size_t i = 0; i < derivatives.third.size(); ++i) {
            addLimitCost(
                derivatives.third[i], config.max_jerk,
                duration * duration * duration, &third_gradient[i]);
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
                    const LocalPlannerConfig& config,
                    bool forbid_unknown,
                    const std::vector<Eigen::Vector3d>& seed_anchors,
                    const Eigen::Vector3d& corridor_min,
                    const Eigen::Vector3d& corridor_max,
                    Clock::time_point deadline) {
    const int free_begin = 3;
    const int free_end = static_cast<int>(points->size()) - 3;
    const int variable_count = 3 * (free_end - free_begin);
    if (variable_count <= 0 ||
        seed_anchors.size() != points->size()) {
        return false;
    }
    SplineObjective objective{
        esdf, config, *points, knots,
        free_begin, free_end, duration, forbid_unknown};
    std::vector<double> values(static_cast<size_t>(variable_count), 0.0);
    std::vector<double> lower(static_cast<size_t>(variable_count), 0.0);
    std::vector<double> upper(static_cast<size_t>(variable_count), 0.0);
    for (int i = free_begin; i < free_end; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            const int index = 3 * (i - free_begin) + axis;
            const double anchor =
                seed_anchors[static_cast<size_t>(i)][axis];
            lower[static_cast<size_t>(index)] = std::max(
                corridor_min[axis],
                anchor - config.seed_trust_radius);
            upper[static_cast<size_t>(index)] = std::min(
                corridor_max[axis],
                anchor + config.seed_trust_radius);
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
            nlopt_set_min_objective(
                optimizer, nloptObjective, &objective);
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
            if (norm_sq <=
                config.convergence_tolerance *
                config.convergence_tolerance) {
                accepted = true;
                break;
            }

            // Standard limited-memory BFGS two-loop recursion.
            std::vector<double> quasi_gradient = gradient;
            std::vector<double> alpha(step_history.size(), 0.0);
            for (size_t reverse = step_history.size();
                 reverse > 0; --reverse) {
                const size_t history_index = reverse - 1;
                double dot = 0.0;
                for (int i = 0; i < variable_count; ++i) {
                    dot += step_history[history_index][
                               static_cast<size_t>(i)] *
                           quasi_gradient[static_cast<size_t>(i)];
                }
                alpha[history_index] =
                    inverse_curvature_history[history_index] * dot;
                for (int i = 0; i < variable_count; ++i) {
                    quasi_gradient[static_cast<size_t>(i)] -=
                        alpha[history_index] *
                        gradient_history[history_index][
                            static_cast<size_t>(i)];
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
                    initial_hessian_scale *
                    quasi_gradient[static_cast<size_t>(i)];
            }
            for (size_t history_index = 0;
                 history_index < step_history.size(); ++history_index) {
                double dot = 0.0;
                for (int i = 0; i < variable_count; ++i) {
                    dot += gradient_history[history_index][
                               static_cast<size_t>(i)] *
                           direction[static_cast<size_t>(i)];
                }
                const double beta =
                    inverse_curvature_history[history_index] * dot;
                for (int i = 0; i < variable_count; ++i) {
                    direction[static_cast<size_t>(i)] +=
                        step_history[history_index][
                            static_cast<size_t>(i)] *
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
                    direction[static_cast<size_t>(i)] =
                        -gradient[static_cast<size_t>(i)];
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
                          std::max(
                              config.minimum_step_size,
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
                            step_length *
                                direction[static_cast<size_t>(i)],
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
                        curvature +=
                            actual_step[static_cast<size_t>(i)] *
                            gradient_change[static_cast<size_t>(i)];
                    }
                    if (curvature > 1.0e-10) {
                        if (step_history.size() == kHistorySize) {
                            step_history.pop_front();
                            gradient_history.pop_front();
                            inverse_curvature_history.pop_front();
                        }
                        step_history.push_back(std::move(actual_step));
                        gradient_history.push_back(
                            std::move(gradient_change));
                        inverse_curvature_history.push_back(
                            1.0 / curvature);
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
            if (!improved ||
                step_length < config.minimum_step_size) {
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

std::vector<TrajectoryPoint> sampleSpline(
    const std::vector<Eigen::Vector3d>& points,
    const std::vector<double>& knots,
    double duration,
    double dt,
    double initial_yaw,
    double maximum_yaw_rate,
    const ESDFGrid& esdf) {
    std::vector<TrajectoryPoint> trajectory;
    if (points.size() < 4 || duration <= 0.0 || dt <= 0.0) {
        return trajectory;
    }
    const SplineDerivatives derivatives =
        buildDerivatives(points, knots);
    const int sample_count =
        std::max(2, static_cast<int>(std::ceil(duration / dt)) + 1);
    trajectory.reserve(static_cast<size_t>(sample_count));
    double previous_yaw = initial_yaw;
    double previous_time = 0.0;
    for (int i = 0; i < sample_count; ++i) {
        const double time =
            i + 1 == sample_count ? duration : std::min(duration, i * dt);
        const double parameter = clamp(time / duration, 0.0, 1.0);
        TrajectoryPoint point;
        point.t = time;
        point.position =
            evaluateSpline(points, knots, kDegree, parameter);
        point.velocity =
            evaluateSpline(
                derivatives.first, derivatives.first_knots,
                kDegree - 1, parameter) / duration;
        point.acceleration =
            evaluateSpline(
                derivatives.second, derivatives.second_knots,
                kDegree - 2, parameter) /
            (duration * duration);
        const double interval =
            i == 0 ? 0.0 : time - previous_time;
        if (i == 0) {
            point.yaw = initial_yaw;
            point.yaw_rate = 0.0;
        } else {
            double target_yaw = previous_yaw;
            if (point.velocity.head<2>().norm() > 1.0e-5) {
                target_yaw = yawFromVelocity(point.velocity);
            }
            const double difference =
                wrapAngleLocal(target_yaw - previous_yaw);
            const double applied = clamp(
                difference,
                -maximum_yaw_rate * interval,
                maximum_yaw_rate * interval);
            point.yaw = wrapAngleLocal(previous_yaw + applied);
            point.yaw_rate =
                interval > kEpsilon ? applied / interval : 0.0;
        }
        point.clearance = esdf.getValue(
            point.position.x(), point.position.y(), point.position.z());
        trajectory.push_back(point);
        previous_yaw = point.yaw;
        previous_time = time;
    }
    return trajectory;
}

std::vector<TrajectoryPoint> sampleQuinticBoundaryTrajectory(
    const Eigen::Vector3d& start_position,
    const Eigen::Vector3d& start_velocity,
    const Eigen::Vector3d& start_acceleration,
    const Eigen::Vector3d& end_position,
    const Eigen::Vector3d& end_velocity,
    const Eigen::Vector3d& end_acceleration,
    double duration,
    double dt,
    double initial_yaw,
    double maximum_yaw_rate,
    const ESDFGrid& esdf) {
    std::vector<TrajectoryPoint> trajectory;
    if (duration <= 0.0 || dt <= 0.0) {
        return trajectory;
    }

    const double duration2 = duration * duration;
    const double duration3 = duration2 * duration;
    const double duration4 = duration3 * duration;
    const double duration5 = duration4 * duration;
    const Eigen::Vector3d c0 = start_position;
    const Eigen::Vector3d c1 = start_velocity;
    const Eigen::Vector3d c2 = 0.5 * start_acceleration;
    const Eigen::Vector3d displacement =
        end_position - start_position;
    const Eigen::Vector3d c3 =
        (20.0 * displacement -
         (8.0 * end_velocity + 12.0 * start_velocity) * duration -
         (3.0 * start_acceleration - end_acceleration) * duration2) /
        (2.0 * duration3);
    const Eigen::Vector3d c4 =
        (-30.0 * displacement +
         (14.0 * end_velocity + 16.0 * start_velocity) * duration +
         (3.0 * start_acceleration - 2.0 * end_acceleration) *
             duration2) /
        (2.0 * duration4);
    const Eigen::Vector3d c5 =
        (12.0 * displacement -
         (6.0 * end_velocity + 6.0 * start_velocity) * duration -
         (start_acceleration - end_acceleration) * duration2) /
        (2.0 * duration5);

    const int sample_count =
        std::max(
            2,
            static_cast<int>(
                std::ceil(duration / dt)) + 1);
    trajectory.reserve(static_cast<size_t>(sample_count));
    double previous_yaw = initial_yaw;
    double previous_time = 0.0;
    for (int i = 0; i < sample_count; ++i) {
        const double time =
            i + 1 == sample_count
                ? duration
                : std::min(duration, i * dt);
        const double time2 = time * time;
        const double time3 = time2 * time;
        const double time4 = time3 * time;
        const double time5 = time4 * time;
        TrajectoryPoint point;
        point.t = time;
        point.position =
            c0 + c1 * time + c2 * time2 + c3 * time3 +
            c4 * time4 + c5 * time5;
        point.velocity =
            c1 + 2.0 * c2 * time + 3.0 * c3 * time2 +
            4.0 * c4 * time3 + 5.0 * c5 * time4;
        point.acceleration =
            2.0 * c2 + 6.0 * c3 * time + 12.0 * c4 * time2 +
            20.0 * c5 * time3;

        const double interval =
            i == 0 ? 0.0 : time - previous_time;
        if (i == 0) {
            point.yaw = initial_yaw;
            point.yaw_rate = 0.0;
        } else {
            double target_yaw = previous_yaw;
            if (point.velocity.head<2>().norm() > 1.0e-5) {
                target_yaw = yawFromVelocity(point.velocity);
            }
            const double difference =
                wrapAngleLocal(target_yaw - previous_yaw);
            const double applied = clamp(
                difference,
                -maximum_yaw_rate * interval,
                maximum_yaw_rate * interval);
            point.yaw = wrapAngleLocal(previous_yaw + applied);
            point.yaw_rate =
                interval > kEpsilon ? applied / interval : 0.0;
        }
        point.clearance = esdf.getValue(
            point.position.x(), point.position.y(), point.position.z());
        trajectory.push_back(point);
        previous_yaw = point.yaw;
        previous_time = time;
    }
    return trajectory;
}

struct DynamicMetrics {
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
    double yaw_rate = 0.0;
    bool finite = true;
};

constexpr double kDynamicsTolerance = 1.02;

DynamicMetrics measureDynamics(
    const std::vector<TrajectoryPoint>& trajectory) {
    DynamicMetrics result;
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& point = trajectory[i];
        result.finite =
            result.finite &&
            point.position.allFinite() &&
            point.velocity.allFinite() &&
            point.acceleration.allFinite();
        result.velocity =
            std::max(result.velocity, point.velocity.norm());
        result.acceleration =
            std::max(result.acceleration, point.acceleration.norm());
        result.yaw_rate =
            std::max(result.yaw_rate, std::abs(point.yaw_rate));
        if (i == 0) continue;
        const double dt = point.t - trajectory[i - 1].t;
        if (dt <= kEpsilon) {
            result.finite = false;
            continue;
        }
        result.jerk = std::max(
            result.jerk,
            (point.acceleration -
             trajectory[i - 1].acceleration).norm() / dt);
    }
    return result;
}

bool dynamicsFeasible(
    const DynamicMetrics& dynamics,
    const LocalPlannerConfig& config) {
    return dynamics.finite &&
           dynamics.velocity <=
               config.max_velocity * kDynamicsTolerance &&
           dynamics.acceleration <=
               config.max_acceleration * kDynamicsTolerance &&
           dynamics.jerk <=
               config.max_jerk * kDynamicsTolerance &&
           dynamics.yaw_rate <=
               config.max_yaw_rate * kDynamicsTolerance;
}

double dynamicsViolationRatio(
    const DynamicMetrics& dynamics,
    const LocalPlannerConfig& config) {
    if (!dynamics.finite) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max({
        dynamics.velocity / std::max(0.1, config.max_velocity),
        dynamics.acceleration / std::max(0.1, config.max_acceleration),
        dynamics.jerk / std::max(0.1, config.max_jerk),
        dynamics.yaw_rate / std::max(0.1, config.max_yaw_rate)
    });
}

double requiredDurationScale(
    const DynamicMetrics& dynamics,
    const LocalPlannerConfig& config) {
    if (!dynamics.finite) {
        return std::numeric_limits<double>::infinity();
    }
    const double velocity_ratio =
        dynamics.velocity / std::max(0.1, config.max_velocity);
    const double acceleration_ratio =
        dynamics.acceleration / std::max(0.1, config.max_acceleration);
    const double jerk_ratio =
        dynamics.jerk / std::max(0.1, config.max_jerk);
    const double yaw_ratio =
        dynamics.yaw_rate / std::max(0.1, config.max_yaw_rate);
    return std::max({
        1.0,
        velocity_ratio,
        std::sqrt(std::max(0.0, acceleration_ratio)),
        std::cbrt(std::max(0.0, jerk_ratio)),
        yaw_ratio
    });
}

}  // namespace

LocalPlanResult LocalPlanner::planSplineWithRequest(
    const LocalPlanningRequest& request) const {
    const auto planning_start = Clock::now();
    const auto planning_deadline =
        planning_start +
        std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(
                std::max(1.0, config_.planning_time_budget_ms)));
    LocalPlanResult result;
    result.plan_id = plan_id_counter_.fetch_add(1);
    result.guide_waypoint = request.guide_waypoint;
    result.guide_waypoint_index = request.guide_waypoint_index;
    result.trajectory_terminal = request.trajectory_terminal;
    result.trajectory_terminal_index = request.trajectory_terminal_index;
    result.local_goal = request.trajectory_terminal;
    result.local_goal_index = request.trajectory_terminal_index;
    result.used_global_fallback = false;
    result.used_observed_esdf = esdf_.hasKnownMask();

    auto finish = [&]() {
        result.planning_time_ms =
            std::chrono::duration<double, std::milli>(
                Clock::now() - planning_start).count();
        return result;
    };
    if (!isReady()) {
        result.status = PlannerStatus::NO_GLOBAL_PATH;
        result.message = "Planner not initialized (no ESDF or global path)";
        return finish();
    }
    const VehicleState& state = request.state;
    if (!state.position.allFinite() ||
        !state.velocity.allFinite() ||
        !state.acceleration.allFinite() ||
        !request.trajectory_terminal.allFinite()) {
        result.status = PlannerStatus::INVALID_INPUT;
        result.message = "Planning request contains NaN/Inf";
        return finish();
    }
    const auto progress =
        computeProgress(state.position, request.previous_progress_s);
    result.progress_s = progress.valid ? progress.progress_s : 0.0;
    result.progress_index = progress.valid ? progress.segment_index : -1;

    const Eigen::Vector3d terminal = request.trajectory_terminal;
    const Eigen::Vector3d displacement = terminal - state.position;
    const double distance = displacement.norm();
    if (distance < 1.0e-3) {
        result.status = PlannerStatus::LOCAL_GOAL_INVALID;
        result.message = "Guide/terminal is indistinguishable from current position";
        return finish();
    }
    const Eigen::Vector3d direction = displacement / distance;
    const bool final_guide =
        request.guide_waypoint_index >= 0 &&
        !global_path_.empty() &&
        request.guide_waypoint_index >=
            static_cast<int>(global_path_.size()) - 1;

    const double distance_ratio = clamp(
        distance / std::max(0.5, config_.lookahead_distance),
        0.25, 1.0);
    double terminal_speed = config_.nominal_speed * distance_ratio;
    if (final_guide) {
        const double braking_distance = std::max(
            0.0, distance - 0.5 * config_.goal_tolerance);
        terminal_speed = std::min(
            config_.nominal_speed,
            std::sqrt(
                2.0 * std::max(0.1, config_.max_acceleration) *
                braking_distance));
    }
    Eigen::Vector3d terminal_velocity =
        terminal_speed * direction;
    Eigen::Vector3d start_acceleration = state.acceleration;
    if (start_acceleration.norm() > config_.max_acceleration) {
        start_acceleration *=
            config_.max_acceleration / start_acceleration.norm();
    }
    // Duration follows Guide distance and cruise speed.  A non-zero local
    // terminal velocity prevents braking at every moving Guide; the final
    // global goal still has an exact zero-velocity boundary without making
    // the whole segment crawl at half the cruise speed.
    const double travel_speed = std::max(
        0.5, config_.nominal_speed * distance_ratio);
    double duration = std::max({
        0.6,
        distance / travel_speed,
        std::sqrt(
            6.0 * distance /
            std::max(0.1, config_.max_acceleration))
    });

    // Keep the configured optimization dimension fixed.  Collision checking
    // already samples the continuous curve at ESDF-scale spacing; increasing
    // the number of control points to distance/control_point_spacing makes
    // the knot interval very short and cubically amplifies harmless A* grid
    // ripples into large jerk.
    const int control_point_count = std::max(
        8,
        std::min(
            std::max(8, config_.max_reference_points),
            config_.control_points));
    const std::vector<double> knots =
        makeClampedKnots(control_point_count, kDegree);
    const bool forbid_unknown = request.forbid_unknown_space;
    const bool direct_soft_clear = segmentClear(
        esdf_, state.position, terminal,
        config_.target_clearance, forbid_unknown,
        config_.collision_check_spacing);

    std::vector<Eigen::Vector3d> seed_path;
    bool used_search_seed = false;
    if (direct_soft_clear) {
        seed_path = {state.position, terminal};
    } else {
        const auto search_deadline = std::min(
            planning_deadline,
            planning_start + std::chrono::milliseconds(7));
        seed_path = searchLocalSeed(
            esdf_, state.position, state.velocity, terminal, config_,
            forbid_unknown, search_deadline);
        used_search_seed = !seed_path.empty();
        if (seed_path.empty()) {
            result.status = PlannerStatus::OPTIMIZATION_FAILED;
            result.message =
                "Bounded local A* could not initialize a collision-free homotopy";
            return finish();
        }
    }
    const std::vector<Eigen::Vector3d> seed_control_points =
        initializeSplineControlPoints(seed_path, control_point_count);
    std::vector<Eigen::Vector3d> control_points = seed_control_points;
    duration = std::max(
        duration,
        polylineLength(seed_path) /
            std::max(0.5, config_.nominal_speed));
    imposeBoundaryState(
        &control_points, knots, duration,
        state.position, state.velocity, start_acceleration,
        terminal, terminal_velocity);
    // Keep the seed-derived horizon for the independent analytic fallback.
    // Iterative B-spline boundary rebuilding can occasionally increase a
    // residual derivative peak and compound duration on every iteration.
    // Starting the fallback from that runaway duration makes an upward
    // initial velocity overshoot the finite ESDF before returning to Guide.
    const double boundary_fallback_duration = duration;

    Eigen::Vector3d corridor_min = state.position.cwiseMin(terminal);
    Eigen::Vector3d corridor_max = state.position.cwiseMax(terminal);
    for (const auto& point : seed_path) {
        corridor_min = corridor_min.cwiseMin(point);
        corridor_max = corridor_max.cwiseMax(point);
    }
    const double corridor_padding = config_.seed_trust_radius;
    corridor_min.array() -= corridor_padding;
    corridor_max.array() += corridor_padding;
    corridor_min.x() = std::max(
        corridor_min.x(),
        esdf_.originX() + 0.5 * esdf_.resolution());
    corridor_min.y() = std::max(
        corridor_min.y(),
        esdf_.originY() + 0.5 * esdf_.resolution());
    corridor_min.z() = std::max(
        corridor_min.z(),
        esdf_.originZ() + 0.5 * esdf_.resolution());
    corridor_max.x() = std::min(
        corridor_max.x(),
        esdf_.originX() + (esdf_.gx() - 0.5) * esdf_.resolution());
    corridor_max.y() = std::min(
        corridor_max.y(),
        esdf_.originY() + (esdf_.gy() - 0.5) * esdf_.resolution());
    corridor_max.z() = std::min(
        corridor_max.z(),
        esdf_.originZ() + (esdf_.gz() - 0.5) * esdf_.resolution());

    bool optimized = optimizeSpline(
        &control_points, knots, duration, esdf_, config_,
        forbid_unknown, seed_control_points, corridor_min, corridor_max,
        planning_deadline - std::chrono::milliseconds(2));
    imposeBoundaryState(
        &control_points, knots, duration,
        state.position, state.velocity, start_acceleration,
        terminal, terminal_velocity);
    if (!optimized) {
        result.status = PlannerStatus::OPTIMIZATION_FAILED;
        result.message =
            "Gradient B-spline optimization did not produce an incumbent";
        return finish();
    }

    std::vector<TrajectoryPoint> trajectory = sampleSpline(
        control_points, knots, duration, config_.trajectory_dt,
        state.yaw, config_.max_yaw_rate, esdf_);
    DynamicMetrics dynamics = measureDynamics(trajectory);

    // Iterative time reallocation follows the standard B-spline workflow.
    // Boundary control points are rebuilt and the same objective is optimized
    // again; geometry is never altered after the final sampling pass.
    // Six inexpensive rescaling opportunities cover modest residual
    // violations after geometry optimization.  The absolute planning
    // deadline below remains authoritative, so this cannot exceed the
    // configured 30 ms budget.
    for (int time_iteration = 0; time_iteration < 6; ++time_iteration) {
        const double violation_ratio =
            dynamicsViolationRatio(dynamics, config_);
        const double dynamic_scale =
            requiredDurationScale(dynamics, config_);
        if (violation_ratio <= kDynamicsTolerance ||
            Clock::now() + std::chrono::milliseconds(4) >=
                planning_deadline) {
            break;
        }
        // Scale only as much as the measured derivative violation requires.
        // A large fixed reserve compounds over repeated 30 Hz replans.
        duration *= std::max(1.03, 1.03 * dynamic_scale);
        // The Guide is a receding local endpoint, not a state that the
        // vehicle must reach at cruise speed.  When duration is enlarged,
        // keeping the old physical terminal speed moves the final boundary
        // control points farther backwards and can create a new acceleration
        // peak.  Bound it by the average speed of the original safe seed.
        // Rebuilding from that seed also prevents an abnormal optimized loop
        // from becoming the next time-allocation iteration's geometry.
        const double safe_length = polylineLength(seed_path);
        terminal_velocity =
            std::min(
                terminal_speed,
                safe_length / std::max(0.1, duration)) * direction;
        control_points = seed_control_points;
        imposeBoundaryState(
            &control_points, knots, duration,
            state.position, state.velocity, start_acceleration,
            terminal, terminal_velocity);
        optimizeSpline(
            &control_points, knots, duration, esdf_, config_,
            forbid_unknown, seed_control_points, corridor_min, corridor_max,
            planning_deadline - std::chrono::milliseconds(1));
        imposeBoundaryState(
            &control_points, knots, duration,
            state.position, state.velocity, start_acceleration,
            terminal, terminal_velocity);
        trajectory = sampleSpline(
            control_points, knots, duration, config_.trajectory_dt,
            state.yaw, config_.max_yaw_rate, esdf_);
        const ValidationResult time_validation =
            validateTrajectory(trajectory);
        if (time_validation.any_collision ||
            !time_validation.all_clear) {
            optimizeSpline(
                &control_points, knots, duration, esdf_, config_,
                forbid_unknown, seed_control_points,
                corridor_min, corridor_max,
                planning_deadline - std::chrono::milliseconds(1));
            imposeBoundaryState(
                &control_points, knots, duration,
                state.position, state.velocity, start_acceleration,
                terminal, terminal_velocity);
            trajectory = sampleSpline(
                control_points, knots, duration, config_.trajectory_dt,
                state.yaw, config_.max_yaw_rate, esdf_);
        }
        dynamics = measureDynamics(trajectory);
    }

    ValidationResult validation = validateTrajectory(trajectory);
    bool used_deadline_retime = false;
    int cheap_retime_attempts = 0;
    DynamicMetrics last_cheap_retime_dynamics;
    ValidationResult last_cheap_retime_validation;

    // A small residual derivative violation must not discard an otherwise
    // valid Guide/Control training frame.  Do not time-warp the final spline:
    // preserving both endpoint derivatives with a nonlinear clock creates
    // extra u''(t) velocity terms and can increase acceleration/jerk in the
    // middle of the trajectory (the recorded scale_transition failure did
    // exactly that).  Instead, enlarge the duration and sample an analytic
    // quintic trajectory from the exact measured start state to the Guide.
    //
    // This is optimizer-free and bounded.  The Guide endpoint remains exact,
    // while terminal velocity is reduced consistently with the longer
    // receding horizon.  Every candidate undergoes complete ESDF and dynamics
    // validation, so an analytic curve that cuts an obstacle is rejected
    // rather than becoming a Control label.
    // This only gates whether a fully revalidated fallback is attempted; it
    // does not relax the accepted Control-label limits.  A 1.5 ratio covers
    // short-budget NLopt incumbents whose jerk is noisier than the production
    // 30 ms result while remaining cheap to correct through time allocation.
    constexpr double kMaximumCheapRetimeViolationRatio = 1.50;
    constexpr int kMaximumCheapRetimeAttempts = 4;
    const auto cheap_retime_deadline =
        planning_deadline + std::chrono::milliseconds(2);
    if (!validation.any_collision &&
        validation.all_clear &&
        !dynamicsFeasible(dynamics, config_) &&
        dynamicsViolationRatio(dynamics, config_) <=
            kMaximumCheapRetimeViolationRatio) {
        double candidate_duration = boundary_fallback_duration;
        DynamicMetrics candidate_dynamics = dynamics;
        for (int attempt = 0;
             attempt < kMaximumCheapRetimeAttempts;
             ++attempt) {
            if (Clock::now() >= cheap_retime_deadline) break;
            cheap_retime_attempts = attempt + 1;
            const double scale =
                requiredDurationScale(candidate_dynamics, config_);
            if (!std::isfinite(scale)) break;

            // Five percent reserve keeps a trajectory that only barely
            // crossed the tolerance from failing again on the next 30 Hz
            // replan due to sampling or state-estimation noise.
            const double next_duration =
                candidate_duration *
                std::max(1.05, 1.05 * scale);
            if (next_duration / boundary_fallback_duration >= 1.95) break;
            candidate_duration = next_duration;
            const Eigen::Vector3d candidate_terminal_velocity =
                std::min(
                    terminal_speed,
                    polylineLength(seed_path) /
                        std::max(0.1, candidate_duration)) *
                direction;
            std::vector<TrajectoryPoint> candidate_trajectory =
                sampleQuinticBoundaryTrajectory(
                    state.position, state.velocity, start_acceleration,
                    terminal, candidate_terminal_velocity,
                    Eigen::Vector3d::Zero(), candidate_duration,
                    config_.trajectory_dt, state.yaw,
                    config_.max_yaw_rate, esdf_);
            if (candidate_trajectory.empty()) break;
            const ValidationResult candidate_validation =
                validateTrajectory(candidate_trajectory);
            candidate_dynamics =
                measureDynamics(candidate_trajectory);
            last_cheap_retime_dynamics = candidate_dynamics;
            last_cheap_retime_validation = candidate_validation;
            if (candidate_validation.any_collision ||
                !candidate_validation.all_clear) {
                continue;
            }
            if (!dynamicsFeasible(candidate_dynamics, config_)) {
                continue;
            }

            duration = candidate_duration;
            trajectory = std::move(candidate_trajectory);
            validation = candidate_validation;
            dynamics = candidate_dynamics;
            used_deadline_retime = true;
            break;
        }
    }

    result.trajectory = std::move(trajectory);
    result.min_clearance = validation.min_clearance;
    if (forbid_unknown && esdf_.hasKnownMask()) {
        for (const auto& point : result.trajectory) {
            if (!esdf_.isKnown(
                    point.position.x(), point.position.y(),
                    point.position.z())) {
                result.status = PlannerStatus::UNKNOWN_SPACE;
                result.message = "Optimized B-spline enters unknown space";
                return finish();
            }
        }
    }
    if (validation.any_collision || !validation.all_clear) {
        result.status = PlannerStatus::COLLISION;
        std::ostringstream message;
        message << "Optimized B-spline violates clearance: min="
                << validation.min_clearance
                << ", start=" << result.trajectory.front().clearance
                << ", worst_t=" << validation.worst_time
                << ", samples=" << validation.clearance_violation_count;
        result.message = message.str();
        return finish();
    }
    if (!dynamicsFeasible(dynamics, config_)) {
        result.status = PlannerStatus::DYNAMICS_VIOLATION;
        std::ostringstream message;
        message << "B-spline dynamics violation: v=" << dynamics.velocity
                << ", a=" << dynamics.acceleration
                << ", jerk=" << dynamics.jerk
                << ", yaw_rate=" << dynamics.yaw_rate;
        if (cheap_retime_attempts > 0) {
            message << ", final_retime_attempts="
                    << cheap_retime_attempts
                    << ", final_retime_v="
                    << last_cheap_retime_dynamics.velocity
                    << ", final_retime_a="
                    << last_cheap_retime_dynamics.acceleration
                    << ", final_retime_jerk="
                    << last_cheap_retime_dynamics.jerk
                    << ", final_retime_yaw_rate="
                    << last_cheap_retime_dynamics.yaw_rate
                    << ", final_retime_clearance="
                    << last_cheap_retime_validation.min_clearance;
        }
        result.message = message.str();
        return finish();
    }
    result.success = true;
    result.status = PlannerStatus::SUCCESS;
    if (used_deadline_retime) {
        result.message =
            used_search_seed
                ? "OK (local A* seed + gradient B-spline + final retime)"
                : "OK (direct gradient B-spline + final retime)";
    } else {
        result.message =
            used_search_seed ? "OK (local A* seed + gradient B-spline)" :
                               "OK (direct gradient B-spline)";
    }
    return finish();
}

}  // namespace il_dataset
