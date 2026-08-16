#pragma once
/// @file   task_candidate_generator.hpp
/// @brief  Fast task-candidate generation for one scene.
///
/// * start / goal are sampled from the pre-computed VALID task cells of
///   the MAIN connected component (the SceneGeometryCache is built once);
/// * cheap analytic re-checks (surface clearance + boundary + distance
///   band) reject invalid pairs BEFORE any preflight;
/// * geometric PROXY classes (CLEAR / LOCAL_AVOIDANCE / ... / LONG_DETOUR)
///   bias which goal is sampled (the real behaviour class is decided by
///   preflight afterwards);
/// * initial yaw uses LAYERED sampling over the absolute goal-bearing
///   error (0-15 / 15-35 / 35-55 / 55-90 / 90-150 / 150-180 deg) with a
///   random sign, so the ±180° range (including the ±45° FOV boundary
///   and rear-goal turns) is covered and left/right stays mirror-balanced.
///
/// All randomness comes from one explicit seed; results are deterministic.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/scene_geometry_cache.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

class TaskCandidateGenerator {
public:
    explicit TaskCandidateGenerator(const BlueprintGenerationConfig& cfg)
        : cfg_(cfg) {}

    /// Geometric PROXY class of a start/goal pair (uses only scene truth;
    /// privileged, offline, never a student input).
    TaskGeomType classifyGeometry(const BlueprintScene& scene,
                                  const Vec2d& start,
                                  const Vec2d& goal) const;

    /// Sample one task candidate.  `task_type_weights` biases the desired
    /// proxy class (length == kNumTaskGeomTypes), `yaw_weights` biases the
    /// initial-yaw strata.  Returns true on success; on budget exhaustion
    /// returns the best (most recently classified) candidate it found so
    /// the caller can still fill its pool (never silently invalid).
    bool sample(const BlueprintScene& scene, const SceneGeometryCache& geo,
                const std::vector<double>& task_type_weights,
                const std::vector<double>& yaw_weights, uint64_t seed,
                uint64_t task_id, uint64_t scene_id, BlueprintTask& out,
                TaskGeomType& geom_out, double& yaw_error_signed_deg) const;

    /// Sample an initial Flightmare yaw (convention B) from the layered
    /// distribution for a given goal bearing (expert frame).
    double sampleInitialYaw(double goal_bearing_expert,
                            const std::vector<double>& yaw_weights,
                            Rng& rng) const;

private:
    /// The straight corridor half-width (m) used by the proxy classifier:
    /// vehicle radius + navigation clearance + discretisation margin.
    double corridorHalfWidth() const {
        return cfg_.vehicle_radius_m + cfg_.navigation_clearance_m +
               cfg_.clearance_discretization_margin_m;
    }
    /// "Large" obstacle radius threshold for LARGE_OCCLUSION.
    double largeRadiusThreshold() const { return 1.5; }

    BlueprintGenerationConfig cfg_;
};

}  // namespace expert
}  // namespace il_dataset
