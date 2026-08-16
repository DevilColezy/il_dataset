#pragma once
/// @file   scene_geometry_cache.hpp
/// @brief  One-time-per-scene geometry cache: truth ESDF over the
///         warehouse FREE region, 8-connected free components, main
///         component area and the pre-computed list of valid task cells.
///
/// Built ONCE per scene and reused by every task candidate of that scene
/// (never rebuilt per task — the pipeline must avoid
/// "scene × task × full ESDF rebuild").  It also computes the
/// planning-validity flags (main component area >= min area, narrowest
/// corridor proxy) and updates SceneMetadata.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <cstdint>
#include <vector>

namespace il_dataset {
namespace expert {

class SceneGeometryCache {
public:
    /// Build the cache for one scene.  Returns false when the scene is
    /// geometrically unusable (no valid free cells / no main component).
    bool build(const BlueprintScene& scene,
               const BlueprintGenerationConfig& cfg, SceneMetadata& meta);

    // ── grid access ────────────────────────────────────────────────
    bool inGrid(int ix, int iy) const {
        return ix >= 0 && iy >= 0 && ix < w_ && iy < h_;
    }
    size_t idOf(int ix, int iy) const {
        return static_cast<size_t>(iy) * w_ + ix;
    }
    Vec2d cellCenter(int ix, int iy) const {
        return gridCellCenter(ix, iy, min_bounds_, res_);
    }
    double distAt(size_t id) const { return dist_[id]; }
    int compAt(size_t id) const { return comp_[id]; }
    int w() const { return w_; }
    int h() const { return h_; }
    double res() const { return res_; }
    const Vec2d& minBounds() const { return min_bounds_; }
    int mainComponent() const { return main_component_; }
    uint64_t mainComponentAreaCells() const { return main_area_cells_; }
    bool cellFree(int ix, int iy, double clearance) const {
        if (!inGrid(ix, iy)) return false;
        return dist_[idOf(ix, iy)] > clearance + 1e-9;
    }
    /// True when the world point lies on a free cell of the MAIN component
    /// with the given centre->surface clearance.
    bool pointFreeMain(const Vec2d& p, double clearance) const {
        const GridIndex2D g = worldToGrid(p, min_bounds_, res_);
        if (!inGrid(g.ix, g.iy)) return false;
        const size_t id = idOf(g.ix, g.iy);
        return comp_[id] == main_component_ && dist_[id] > clearance + 1e-9;
    }
    /// All free cells of the MAIN component (flat cell index list).
    const std::vector<size_t>& validCells() const { return valid_cells_; }
    /// Pre-computed centre->surface distance for each valid cell.
    const std::vector<double>& validCellClearances() const {
        return valid_clearances_;
    }
    /// Narrowest obstacle-pair surface gap in the scene (0 when <2).
    double estimatedCorridorWidth() const { return estimated_corridor_width_; }

private:
    double res_ = 0.1;
    Vec2d min_bounds_{0.0, 0.0};
    int w_ = 0, h_ = 0;
    std::vector<double> dist_;   // centre->surface distance (free region)
    std::vector<int> comp_;      // component id (-1 = not free)
    int main_component_ = -1;
    uint64_t main_area_cells_ = 0;
    std::vector<size_t> valid_cells_;
    std::vector<double> valid_clearances_;
    double estimated_corridor_width_ = 0.0;
    BlueprintGenerationConfig cfg_;
};

}  // namespace expert
}  // namespace il_dataset
