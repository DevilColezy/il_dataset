#pragma once
/// @file   scene_profile_generator.hpp
/// @brief  Scene Profile catalog + deterministic random realization.
///
/// Replaces the fixed "empty + 3x3 strata" scene schedule.  A profile
/// describes a DISTRIBUTION (count band, radius band/mode, structure,
/// cluster/passage parameters); a scene is a random realization of one
/// profile.  Every placement must satisfy the planner-compatible rules:
///   * centre inside the warehouse FREE region with boundary margin;
///   * pairwise surface gap >= min_surface_gap_m (>= planner-required
///     traversable passage, validated by the config);
///   * structured profiles (corridor / bottleneck / chicane) keep a
///     central passage of width >= planner-required passage.
/// All randomness is derived from one explicit uint64_t seed.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <random>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

struct SceneGenerationOutcome {
    bool success = false;
    BlueprintScene scene;
    SceneMetadata metadata;
    int attempts = 0;
    std::string reason;
};

class SceneProfileGenerator {
public:
    explicit SceneProfileGenerator(const BlueprintGenerationConfig& cfg);

    /// The full profile catalog (defaults + any user-provided profiles).
    const std::vector<SceneProfile>& profiles() const { return profiles_; }
    /// Look up a profile by name (nullptr when absent).
    const SceneProfile* findProfile(const std::string& name) const;

    /// Weighted profile pick from the catalog (weights are the product of
    /// the base profile weight and the deficit multipliers).
    const SceneProfile* pickProfile(std::mt19937_64& rng,
                                    const std::map<std::string, double>&
                                        deficit_weights) const;

    /// Realize ONE scene from a profile + seed.  Never degrades silently:
    /// returns success=false (with reason) when the placement budget is
    /// exhausted.
    SceneGenerationOutcome generate(const SceneProfile& profile,
                                    uint64_t scene_id, uint64_t seed) const;

private:
    /// Fill the static default profile catalog (see .cpp).
    void buildDefaultCatalog();

    bool placeOne(const BlueprintScene& scene, const SceneProfile& profile,
                  const BlueprintGenerationConfig& cfg, Rng& rng,
                  BlueprintObstacle& out) const;
    /// Structured placement (clustered / corridor / bottleneck / chicane /
    /// central_blocker / edge_clutter).  Orientation and cluster centres
    /// are drawn ONCE per scene realization and reused for every obstacle;
    /// per-side along-spacing is enforced so the pairwise surface gap is
    /// guaranteed by construction (rejection rarely triggers).
    bool realizeStructured(const SceneProfile& profile, const BlueprintGenerationConfig& cfg,
                           Rng& rng, int desired, std::vector<BlueprintObstacle>& out,
                           int& placed) const;
    /// Post-realization sanity validation: a realization must actually
    /// match its own profile structure (clustered groups, corridor free
    /// channel, bottleneck narrowing, chicane alternation).  Returns false
    /// (with reason) when the realization should be rejected and retried.
    bool validateProfileStructure(const SceneProfile& profile,
                                  const BlueprintGenerationConfig& cfg,
                                  const BlueprintScene& scene,
                                  std::string& reason) const;
    /// Compute the geometric metadata (radius bands, density proxy) from
    /// the realized obstacle set.
    SceneMetadata computeMetadata(const SceneProfile& profile,
                                  const BlueprintScene& scene,
                                  uint64_t seed, int attempts) const;

    BlueprintGenerationConfig cfg_;
    std::vector<SceneProfile> profiles_;
};

}  // namespace expert
}  // namespace il_dataset
