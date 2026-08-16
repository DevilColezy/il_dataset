#pragma once
/// @file   ray_cast_2d.hpp
/// @brief  Shared ANALYTIC 2D ray-cast helpers (no spatial marching).
///
/// Used by BOTH the DepthProxyEvaluator (synthetic depth proxy) and the
/// PreflightSimulator (synthetic observation patch).  Every ray is cast
/// against the truth cylinders via the closed-form ray-circle
/// intersection and against the warehouse walls via the slab
/// ray-rectangle intersection — complexity O(rays x obstacles) instead of
/// O(rays x spatial_samples x obstacles).
///
/// Sign / frame convention: ray origin `o` and direction `d` are in the
/// EXPERT 2D world frame (X right, Y up).  `t` is the distance along the
/// (unit) direction; the returned hit is the nearest positive `t`.

#include "il_dataset/hierarchical_expert/types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

/// Nearest positive intersection distance `t` of the ray `o + t*d` with a
/// circle (centre `c`, radius `r`).  Returns +inf when the ray misses or
/// the circle is behind the origin.  `d` must be a unit vector.
inline double rayCircleHit(const Vec2d& o, const Vec2d& d, const Vec2d& c,
                           double r) {
    const Vec2d oc = o - c;
    const double b = 2.0 * d.dot(oc);
    const double cc = oc.squaredNorm() - r * r;
    const double disc = b * b - 4.0 * cc;
    if (disc < 0.0) return std::numeric_limits<double>::infinity();
    const double sq = std::sqrt(disc);
    const double t0 = (-b - sq) * 0.5;
    const double t1 = (-b + sq) * 0.5;
    if (t0 > 1e-9) return t0;
    if (t1 > 1e-9) return t1;
    return std::numeric_limits<double>::infinity();
}

/// Nearest positive intersection distance `t` of the ray `o + t*d` with an
/// axis-aligned rectangle [min_b, max_b] (the warehouse wall envelope).
/// Uses the slab method (exact).  Returns +inf when the ray misses.
inline double rayRectHit(const Vec2d& o, const Vec2d& d, const Vec2d& min_b,
                         const Vec2d& max_b) {
    double tmin = 0.0, tmax = std::numeric_limits<double>::infinity();
    auto slab = [&](double origin, double dir, double lo, double hi) {
        if (std::fabs(dir) < 1e-12) {
            if (origin < lo || origin > hi) return false;
            return true;
        }
        const double inv = 1.0 / dir;
        double t1 = (lo - origin) * inv;
        double t2 = (hi - origin) * inv;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1);
        tmax = std::min(tmax, t2);
        return tmin <= tmax;
    };
    if (!slab(o.x(), d.x(), min_b.x(), max_b.x())) {
        return std::numeric_limits<double>::infinity();
    }
    if (!slab(o.y(), d.y(), min_b.y(), max_b.y())) {
        return std::numeric_limits<double>::infinity();
    }
    // tmax <= 0 -> the rectangle is behind the ray origin.
    if (tmax <= 1e-9) return std::numeric_limits<double>::infinity();
    // tmin <= 0 -> the origin is INSIDE the rectangle (the camera is always
    // inside the warehouse envelope): the WALL is hit at the EXIT point,
    // not at distance 0.  Only a ray that never crosses the boundary has
    // tmin<=0 AND tmax<=0, and that case is already rejected above.
    if (tmin <= 0.0) return tmax;
    return tmin;
}

/// Nearest hit of a ray against a set of circles PLUS an optional
/// axis-aligned rectangle (walls).  `has_wall` enables the rectangle.
/// Returns the nearest positive `t` (+inf if nothing is hit).
inline double rayNearestObstacleHit(
    const Vec2d& o, const Vec2d& d,
    const std::vector<Vec2d>& centers, const std::vector<double>& radii,
    bool has_wall, const Vec2d& wall_min, const Vec2d& wall_max) {
    double best = std::numeric_limits<double>::infinity();
    const size_t n = std::min(centers.size(), radii.size());
    for (size_t i = 0; i < n; ++i) {
        best = std::min(best, rayCircleHit(o, d, centers[i], radii[i]));
    }
    if (has_wall) {
        best = std::min(best, rayRectHit(o, d, wall_min, wall_max));
    }
    return best;
}

}  // namespace expert
}  // namespace il_dataset
