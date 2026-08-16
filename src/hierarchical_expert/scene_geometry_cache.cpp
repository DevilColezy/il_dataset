#include "il_dataset/hierarchical_expert/scene_geometry_cache.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

namespace il_dataset {
namespace expert {

bool SceneGeometryCache::build(const BlueprintScene& scene,
                               const BlueprintGenerationConfig& cfg,
                               SceneMetadata& meta) {
    cfg_ = cfg;
    const WarehouseGeometry& wh = cfg.warehouse;
    res_ = cfg.esdf_resolution_m;
    min_bounds_ = wh.freeMin();
    const Vec2d max_bounds = wh.freeMax();
    w_ = std::max(1, static_cast<int>(std::ceil(
                         (max_bounds.x() - min_bounds_.x()) / res_)));
    h_ = std::max(1, static_cast<int>(std::ceil(
                         (max_bounds.y() - min_bounds_.y()) / res_)));
    dist_.assign(static_cast<size_t>(w_) * h_,
                 std::numeric_limits<double>::infinity());
    comp_.assign(static_cast<size_t>(w_) * h_, -1);
    valid_cells_.clear();
    valid_clearances_.clear();

    const double margin = cfg.boundary_margin_m;
    const double free_min = cfg.free_cell_surface_clearance_m;
    // ESDF over the free region: analytic centre->surface distance.
    for (int iy = 0; iy < h_; ++iy) {
        for (int ix = 0; ix < w_; ++ix) {
            const size_t id = static_cast<size_t>(iy) * w_ + ix;
            const Vec2d cw = cellCenter(ix, iy);
            if (!wh.inFree(cw.x(), cw.y(), margin)) {
                dist_[id] = -1.0;  // not a candidate cell
                continue;
            }
            double best = std::numeric_limits<double>::infinity();
            for (const auto& o : scene.obstacles) {
                const double d = (cw - Vec2d(o.x, o.y)).norm() - o.radius;
                best = std::min(best, d);
            }
            dist_[id] = best;
            comp_[id] = best > free_min + 1e-9 ? 0 : -1;
        }
    }

    // ── 8-connected components (no diagonal corner-cutting) ────────
    const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    auto isFree = [&](int ix, int iy) -> bool {
        if (ix < 0 || iy < 0 || ix >= w_ || iy >= h_) return false;
        return comp_[static_cast<size_t>(iy) * w_ + ix] != -1;
    };
    int next = 1;
    uint64_t best_area = 0;
    int best_comp = -1;
    for (int iy = 0; iy < h_; ++iy) {
        for (int ix = 0; ix < w_; ++ix) {
            const size_t id = static_cast<size_t>(iy) * w_ + ix;
            if (comp_[id] != 0) continue;
            std::queue<int> qx, qy;
            qx.push(ix);
            qy.push(iy);
            comp_[id] = next;
            uint64_t area = 0;
            while (!qx.empty()) {
                const int cx = qx.front();
                const int cy = qy.front();
                qx.pop();
                qy.pop();
                ++area;
                for (int k = 0; k < 8; ++k) {
                    const int nx = cx + dx[k], ny = cy + dy[k];
                    if (nx < 0 || ny < 0 || nx >= w_ || ny >= h_) continue;
                    if (dx[k] != 0 && dy[k] != 0 &&
                        (!isFree(nx, cy) || !isFree(cx, ny))) {
                        continue;
                    }
                    const size_t nid = static_cast<size_t>(ny) * w_ + nx;
                    if (comp_[nid] != 0) continue;
                    comp_[nid] = next;
                    qx.push(nx);
                    qy.push(ny);
                }
            }
            if (area > best_area) {
                best_area = area;
                best_comp = next;
            }
            ++next;
        }
    }
    main_component_ = best_comp;
    main_area_cells_ = best_area;

    // ── Collect valid task cells of the MAIN component ─────────────
    for (size_t id = 0; id < comp_.size(); ++id) {
        if (comp_[id] == main_component_) {
            valid_cells_.push_back(id);
            valid_clearances_.push_back(dist_[id]);
        }
    }

    // ── Scene-level immutable geometry (computed ONCE, reused by every
    //    task candidate — never recomputed per start/goal pair) ──────
    obs_centers_.clear();
    obs_radii_.clear();
    large_obs_.clear();
    narrow_passages_.clear();
    for (const auto& o : scene.obstacles) {
        obs_centers_.emplace_back(o.x, o.y);
        obs_radii_.push_back(o.radius);
    }
    const double large_r = 1.5;  // same threshold as the task proxy
    const double narrow_lo = cfg.plannerRequiredPassage();
    const double narrow_hi = 2.0 * narrow_lo;
    estimated_corridor_width_ = 0.0;
    if (scene.obstacles.size() >= 2) {
        double best = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < scene.obstacles.size(); ++i) {
            if (obs_radii_[i] >= large_r) large_obs_.push_back(static_cast<int>(i));
            for (size_t j = i + 1; j < scene.obstacles.size(); ++j) {
                const double gap = std::hypot(obs_centers_[i].x() - obs_centers_[j].x(),
                                              obs_centers_[i].y() - obs_centers_[j].y()) -
                                   obs_radii_[i] - obs_radii_[j];
                best = std::min(best, gap);
                // Narrow-but-planable passage candidate.
                if (gap >= narrow_lo - 1e-9 && gap <= narrow_hi + 1e-9) {
                    NarrowPassage np;
                    np.a_id = static_cast<int>(i);
                    np.b_id = static_cast<int>(j);
                    np.a_center = obs_centers_[i];
                    np.b_center = obs_centers_[j];
                    np.center = (obs_centers_[i] + obs_centers_[j]) * 0.5;
                    np.width = gap;
                    Vec2d ab = obs_centers_[j] - obs_centers_[i];
                    const double abn = ab.norm();
                    np.axis = abn > 1e-9 ? ab / abn : Vec2d(1.0, 0.0);
                    narrow_passages_.push_back(np);
                }
            }
        }
        if (std::isfinite(best)) estimated_corridor_width_ = best;
    } else if (scene.obstacles.size() == 1) {
        if (obs_radii_[0] >= large_r) large_obs_.push_back(0);
    }

    // ── Planning validity ──────────────────────────────────────────
    const double free_area = wh.area();
    meta.free_space_ratio = free_area > 1e-9
                                ? static_cast<double>(main_area_cells_) *
                                      res_ * res_ / free_area
                                : 0.0;
    meta.estimated_corridor_width = estimated_corridor_width_;
    const double min_area = cfg.min_main_component_area_m2;
    meta.planning_valid = main_area_cells_ > 0 &&
                          static_cast<double>(main_area_cells_) * res_ * res_ >=
                              min_area - 1e-9;
    meta.planning_failure_reason =
        meta.planning_valid
            ? ""
            : "main free component area below " +
                  std::to_string(min_area) + " m^2";

    return !valid_cells_.empty() && meta.planning_valid;
}

}  // namespace expert
}  // namespace il_dataset
