#include "il_dataset/hierarchical_expert/depth_proxy_evaluator.hpp"

#include "il_dataset/hierarchical_expert/ray_cast_2d.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

DepthProxySample DepthProxyEvaluator::castAt(
    const Vec2d& pos_expert, double yaw_expert,
    const std::vector<BlueprintObstacle>& obstacles, bool has_wall,
    const Vec2d& wall_min, const Vec2d& wall_max) const {
    DepthProxySample s;
    const double range = cfg_.depth_far_max_m;
    const double near_max = cfg_.depth_near_max_m;
    const double mid_max = cfg_.depth_mid_max_m;
    const int n_rays = std::max(8, cfg_.depth_proxy_num_rays);
    const double fov = deg2rad(p_.obs_fov_deg);

    // Camera rig: the same camera origin / orientation as the runtime.
    const double pos[3] = {pos_expert.x(), pos_expert.y(), 0.0};
    double q[4];
    CameraRig2D::quatFromExpertYaw(yaw_expert, q);
    CameraRig2D rig(p_, pos, q);
    const Vec2d cam(rig.worldX(), rig.worldY());

    // Pre-extract circle geometry for the shared analytic ray helper.
    std::vector<Vec2d> centers;
    std::vector<double> radii;
    centers.reserve(obstacles.size());
    radii.reserve(obstacles.size());
    for (const auto& o : obstacles) {
        centers.emplace_back(o.x, o.y);
        radii.push_back(o.radius);
    }

    s.total_rays = static_cast<uint64_t>(n_rays);
    int consecutive = 0, max_consecutive = 0;

    for (int i = 0; i < n_rays; ++i) {
        // Bearing in the CAMERA frame (matches runtime ray semantics);
        // the world direction is the SAME rig helper the runtime uses.
        const double bearing =
            -fov * 0.5 + fov * (static_cast<double>(i) / std::max(1, n_rays - 1));
        double dir_world[2] = {0.0, 0.0};
        rig.rayWorldDirXY(bearing, dir_world);
        const Vec2d d(dir_world[0], dir_world[1]);

        const double hit = rayNearestObstacleHit(
            cam, d, centers, radii, has_wall, wall_min, wall_max);
        if (hit > range) {
            ++s.free_count;
            consecutive = 0;
        } else {
            ++s.occupied_rays;
            ++consecutive;
            max_consecutive = std::max(max_consecutive, consecutive);
            if (hit <= near_max) {
                ++s.near_count;
            } else if (hit <= mid_max) {
                ++s.mid_count;
            } else {
                ++s.far_count;
            }
            s.min_visible = std::min(s.min_visible, hit);
            s.sum_visible += hit;
            ++s.visible_count;
        }
    }
    if (n_rays > 0) {
        // angular span of the longest consecutive occupied run
        const double ray_span = fov / std::max(1, n_rays - 1);
        s.max_angular_occlusion_deg =
            rad2deg(ray_span * static_cast<double>(max_consecutive));
    }
    return s;
}

void DepthProxyEvaluator::accumulate(const DepthProxySample& s,
                                     TaskDistributionSummary& out) const {
    out.depth_samples += s.total_rays;
    out.depth_near_count += s.near_count;
    out.depth_mid_count += s.mid_count;
    out.depth_far_count += s.far_count;
    out.depth_free_count += s.free_count;
    out.depth_min_visible_m =
        std::min(out.depth_min_visible_m, s.min_visible);
    if (s.visible_count > 0) {
        // Running mean over all visible rays across samples.
        const double prev_sum =
            out.depth_mean_visible_m *
            static_cast<double>(out.depth_visible_count);
        out.depth_visible_count += s.visible_count;
        out.depth_mean_visible_m =
            (prev_sum + s.sum_visible) /
            static_cast<double>(out.depth_visible_count);
    }
    out.depth_max_angular_occlusion_deg =
        std::max(out.depth_max_angular_occlusion_deg,
                 s.max_angular_occlusion_deg);
    out.depth_occupied_ray_ratio =
        out.depth_samples > 0
            ? static_cast<double>(out.depth_near_count + out.depth_mid_count +
                                  out.depth_far_count) /
                  static_cast<double>(out.depth_samples)
            : 0.0;
}

}  // namespace expert
}  // namespace il_dataset
