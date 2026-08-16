#pragma once
/// @file   depth_proxy_evaluator.hpp
/// @brief  Cheap 2D synthetic raycast depth-distribution proxy.
///
/// NOT a student input and NOT a Unity render: it casts a fixed number of
/// 2D rays from the CAMERA (CameraRig2D at the vehicle pose, same rig as
/// the runtime) against the truth cylinders with ANALYTIC ray-circle
/// intersection, at a configured temporal stride along the preflight
/// trajectory.  Output feeds the near/mid/far/free depth distribution
/// targets used to balance multi-scale depth avoidance training data.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/flightmare_2d_observation.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <vector>

namespace il_dataset {
namespace expert {

class DepthProxyEvaluator {
public:
    explicit DepthProxyEvaluator(const BlueprintGenerationConfig& cfg,
                                 const Params2D& params)
        : cfg_(cfg), p_(params) {}

    /// Cache the scene-static circle geometry (centres + radii) so the
    /// many castAt() calls along a preflight never rebuild it.  Call once
    /// per scene (per preflight episode).  When not called, castAt() still
    /// falls back to building from `obstacles` on every sample.
    void configure(const std::vector<BlueprintObstacle>& obstacles);

    /// One synthetic raycast sample from a vehicle pose (expert frame).
    /// `pos_expert`/`yaw_expert` are the expert-2D position and yaw
    /// (world XY identity with the Flightmare frame).  `has_wall` /
    /// `wall_min` / `wall_max` add the warehouse wall envelope to the ray
    /// hits (analytic slab intersection, same helper as the preflight).
    /// The `obstacles` argument is only used as a fallback when the
    /// scene-static geometry was not cached via configure().
    DepthProxySample castAt(const Vec2d& pos_expert, double yaw_expert,
                            const std::vector<BlueprintObstacle>& obstacles,
                            bool has_wall = false,
                            const Vec2d& wall_min = Vec2d(0, 0),
                            const Vec2d& wall_max = Vec2d(0, 0)) const;

    /// Merge one sample into a TaskDistributionSummary (raw counts).
    void accumulate(const DepthProxySample& s,
                    TaskDistributionSummary& out) const;

    int numRays() const { return cfg_.depth_proxy_num_rays; }
    int strideTicks() const { return cfg_.depth_proxy_sample_stride_ticks; }

private:
    BlueprintGenerationConfig cfg_;
    Params2D p_;
    // Scene-static cached geometry (filled by configure(); cleared when
    // the obstacles change).
    std::vector<Vec2d> centers_;
    std::vector<double> radii_;
    bool geometry_cached_ = false;
};

}  // namespace expert
}  // namespace il_dataset
