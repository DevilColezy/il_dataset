#include "il_dataset/hierarchical_expert/preflight_simulator.hpp"

#include "il_dataset/hierarchical_expert/coordinate_adapter.hpp"
#include "il_dataset/hierarchical_expert/ray_cast_2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

namespace {

/// Distance from a point to a finite segment (exact, for swept audit).
inline double distToSegment(const Vec2d& p, const Vec2d& a, const Vec2d& b) {
    const Vec2d ab = b - a;
    const double len2 = ab.squaredNorm();
    if (len2 <= 1e-12) return (p - a).norm();
    const double t = clamp((p - a).dot(ab) / len2, 0.0, 1.0);
    return (p - (a + ab * t)).norm();
}

}  // namespace

void PreflightSimulator::configure(const Scene2D& scene,
                                   const Vec2d& min_bounds,
                                   const Vec2d& max_bounds,
                                   const Vec2d* wall_min,
                                   const Vec2d* wall_max) {
    scene_ = scene;
    min_bounds_ = min_bounds;
    max_bounds_ = max_bounds;
    has_wall_ = false;
    if (wall_min && wall_max) {
        has_wall_ = true;
        wall_min_ = *wall_min;
        wall_max_ = *wall_max;
    }
    expert_.configure(p_, min_bounds_, max_bounds_);
    configured_ = true;
}

void PreflightSimulator::resetTask(const Vec2d& start, const Vec2d& goal,
                                   double initial_yaw_fm, uint64_t tick,
                                   double flight_z) {
    task_.start = start;
    task_.goal = goal;
    task_.initial_yaw = CoordinateAdapter::flightmareYawToExpert(initial_yaw_fm);
    task_.valid = true;
    flight_z_ = flight_z;
    state_ = VehicleState2D{};
    state_.position = start;
    state_.yaw = task_.initial_yaw;
    state_.velocity_world = Vec2d(0.0, 0.0);
    state_.yaw_rate = 0.0;
    expert_.resetTask(start, goal, initial_yaw_fm, tick, flight_z);
}

LocalObservation PreflightSimulator::synthesizePatch(uint64_t tick) const {
    // ── The SAME camera rig as the runtime path. ────────────────────
    const double pos[3] = {state_.position.x(), state_.position.y(),
                           flight_z_};
    double q[4];
    CameraRig2D::quatFromExpertYaw(state_.yaw, q);
    CameraRig2D rig(p_, pos, q);

    const double range = p_.obs_range_m;
    const double fov = deg2rad(p_.obs_fov_deg);
    const double ray_da = deg2rad(p_.obs_ray_angular_res_deg);
    const int n_rays = std::max(
        1, static_cast<int>(std::ceil(fov / std::max(1e-9, ray_da))));

    // Pre-extract the circle geometry (avoid per-ray Scene2D dereferences).
    std::vector<Vec2d> centers;
    std::vector<double> radii;
    centers.reserve(scene_.obstacles.size());
    radii.reserve(scene_.obstacles.size());
    for (const auto& o : scene_.obstacles) {
        centers.push_back(o.center);
        radii.push_back(o.radius);
    }

    std::vector<double> ray_hit(n_rays + 1,
                                std::numeric_limits<double>::infinity());
    std::vector<bool> ray_seen(n_rays + 1, false);

    const Vec2d cam2(rig.worldX(), rig.worldY());
    for (int i = 0; i <= n_rays; ++i) {
        const double bearing =
            -fov / 2.0 + fov * (static_cast<double>(i) / n_rays);
        double dir[2];
        rig.rayWorldDirXY(bearing, dir);
        // ANALYTIC ray-circle + ray-wall intersection: O(obstacles) per
        // ray, no spatial marching (was O(range/steps x obstacles)).
        const double hit = rayNearestObstacleHit(
            cam2, Vec2d(dir[0], dir[1]), centers, radii, has_wall_,
            wall_min_, wall_max_);
        ray_hit[i] = (hit > range) ? std::numeric_limits<double>::infinity()
                                   : hit;
        ray_seen[i] = true;  // valid return (hit or no-hit to R)
    }

    return obs_builder_.buildFromRays(ray_hit, ray_seen, pos, q, min_bounds_,
                                      tick);
}

bool PreflightSimulator::segmentCrossesBounds(double x0, double y0, double x1,
                                              double y1, double r) const {
    // The SAME shared helper as TruthCylinderAudit (types.hpp): convex
    // safe rectangle — both endpoints inside the r-shrunk rectangle is
    // exact for a straight segment.  No per-class copy, no segSegDist.
    return !segmentDiskInsideBounds(x0, y0, x1, y1, r, min_bounds_,
                                    max_bounds_);
}

PreflightSimulator::SimStepResult PreflightSimulator::step(
    uint64_t tick, bool collision_override) {
    SimStepResult result;
    result.state = state_;
    const Vec2d prev_pos = state_.position;

    // Truth collision audit (privileged, judge-only):
    //  * POINT collision at the current state;
    //  * CONTINUOUS SWEPT collision of the executed segment prev→new;
    //  * out-of-bounds of the drone disk.
    bool point_collision = false;
    for (const auto& o : scene_.obstacles) {
        if ((state_.position - o.center).norm() < o.radius + p_.drone_radius) {
            point_collision = true;
            break;
        }
    }

    const LocalObservation patch = synthesizePatch(tick);
    result.output = expert_.stepFromPatch(
        state_, patch, flight_z_, tick, point_collision || collision_override);

    // Integrate the executable command (shared kinematics).
    if (!result.output.terminal) {
        state_ = integrateKinematicStep(
            state_,
            BodyCommand2D{result.output.target_velocity_flu_x,
                          result.output.target_velocity_flu_y,
                          result.output.target_yaw_rate},
            1.0 / 30.0, p_);
    }

    // Swept collision of the drone disk along prev→new.
    bool swept = false;
    for (const auto& o : scene_.obstacles) {
        if (distToSegment(o.center, prev_pos, state_.position) <
            o.radius + p_.drone_radius - 1e-9) {
            swept = true;
            break;
        }
    }
    // Point collision at the NEW state.
    bool new_point = false;
    for (const auto& o : scene_.obstacles) {
        if ((state_.position - o.center).norm() <
            o.radius + p_.drone_radius) {
            new_point = true;
            break;
        }
    }
    result.out_of_bounds = segmentCrossesBounds(
        prev_pos.x(), prev_pos.y(), state_.position.x(), state_.position.y(),
        p_.drone_radius);
    result.truth_collision =
        collision_override || point_collision || swept || new_point;

    result.goal_reached =
        result.output.fsm_state == "GOAL_REACHED" ||
        (task_.goal - state_.position).norm() <= p_.task_goal_tolerance;
    result.state = state_;
    return result;
}

}  // namespace expert
}  // namespace il_dataset
