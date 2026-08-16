#pragma once
/// @file   route_qualifier.hpp
/// @brief  PRIVILEGED task qualification (port of the 2D causal
///         qualification from il_2d_multiscale_debug).
///
/// Runs BEFORE the expensive Full HierarchicalExpert preflight and answers
/// three questions for a candidate start/goal pair:
///   1. endpoint safety      — both endpoints have >= endpoint clearance
///                              and lie on the main traversable component;
///   2. global connectivity  — same connected component (cached lookup +
///                              optional bounded A* confirmation);
///   3. straight-corridor blocker analysis + causal LEFT/RIGHT routes —
///      when the direct start->goal corridor is blocked at the ROUTE
///      clearance, side-constrained A* routes are planned around the
///      PRIMARY blocker on the LEFT and RIGHT side; with
///      require_both_sides_feasible=true the task is accepted only when
///      BOTH homotopy branches are globally feasible.
///
/// EVERYTHING here is privileged truth: the A* routes, the blocker and the
/// side feasibility are used ONLY for task fairness / geometry
/// classification / generation statistics.  They are NEVER fed to the
/// 5 Hz or 30 Hz expert and never stored as student inputs.
///
/// Side definition (rotation invariant, ported from the 2D reference):
///   forward = normalize(goal - start)
///   left    = (-forward.y, forward.x)
///   right   = -left
///   LEFT  ⇔  cross(forward, p - blocker_center) > 0
///   RIGHT ⇔  cross(forward, p - blocker_center) < 0
/// The start->goal axis and the blocker centre are FIXED for the whole
/// qualification, so LEFT/RIGHT never depend on world axes.
///
/// Performance: the analytic truth-ESDF grid (obstacle distance AND region
/// boundary distance) is built ONCE per scene in configure(); A* reuses
/// generation-marked buffers (no per-query W*H allocation) and every
/// search has a hard node-expansion cap.  Clear tasks never run a search.

#include "il_dataset/hierarchical_expert/blueprint_types.hpp"
#include "il_dataset/hierarchical_expert/scene_geometry_cache.hpp"
#include "il_dataset/hierarchical_expert/scene_task_blueprint.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace il_dataset {
namespace expert {

class TaskRouteQualifier {
public:
    TaskRouteQualifier() = default;

    /// Build the scene-static analytic truth-ESDF grid (obstacle surface
    /// distance AND region-boundary distance, 2D reference semantics) and
    /// cache the obstacle geometry.  Must be called once per scene before
    /// any qualify().
    void configure(const BlueprintScene& scene, const SceneGeometryCache& geo,
                   const BlueprintGenerationConfig& cfg);

    /// Run the full qualification pipeline for one candidate.  Fills
    /// `out` (accepted + reasons + side routes) and updates `counters`
    /// (aggregate).  Never throws; always deterministic for (start,goal).
    void qualify(const Vec2d& start, const Vec2d& goal,
                 TaskQualificationSummary& out,
                 QualificationCounters& counters) const;

    /// True when the ESDF grid is built and usable.
    bool valid() const { return w_ > 0 && h_ > 0; }

    // ── exposed analytic primitives (shared with the tests) ────────
    /// True signed clearance at a world point:
    /// min(obstacle surface distance, distance to the region boundary).
    double esdfAt(const Vec2d& p) const;
    /// Strict selectable-space test: esdf > clearance.
    bool isFree(const Vec2d& p, double clearance) const;
    /// Straight-corridor blocker detector (route-clearance margin).
    /// Fills the nearest forward blocker (ties by penetration then id).
    bool findStraightBlocker(const Vec2d& start, const Vec2d& goal,
                             int& blocker_id, Vec2d& blocker_center,
                             double& blocker_radius,
                             std::vector<int>& blocking_ids) const;

private:
    // ── grid ───────────────────────────────────────────────────────
    int ixOf(double x) const {
        return static_cast<int>(std::floor((x - min_.x()) / res_));
    }
    int iyOf(double y) const {
        return static_cast<int>(std::floor((y - min_.y()) / res_));
    }
    bool inGrid(int ix, int iy) const {
        return ix >= 0 && iy >= 0 && ix < w_ && iy < h_;
    }
    size_t idOf(int ix, int iy) const {
        return static_cast<size_t>(iy) * w_ + ix;
    }
    Vec2d cellCenter(int ix, int iy) const {
        return Vec2d(min_.x() + (static_cast<double>(ix) + 0.5) * res_,
                     min_.y() + (static_cast<double>(iy) + 0.5) * res_);
    }
    double cellEsdf(int ix, int iy) const {
        return grid_[idOf(ix, iy)];
    }
    bool cellFreeStrict(int ix, int iy, double clearance) const {
        return inGrid(ix, iy) && cellEsdf(ix, iy) > clearance + 1e-9;
    }
    /// Distance from p to the inside of the region rectangle.
    double boundaryDist(const Vec2d& p) const;

    // ── side-constrained A* (2D port) ──────────────────────────────
    /// A* on the grid with esdf > clearance, optional lateral side-bias
    /// relative to the FIXED (side_axis, side_center) reference.
    /// side_sign: +1 LEFT, -1 RIGHT.  Bounded by max_expansions; reuses
    /// generation-marked buffers.  Returns the world waypoint path.
    std::vector<Vec2d> astarPath(const Vec2d& a, const Vec2d& b,
                                 double clearance, const Vec2d& side_axis,
                                 const Vec2d& side_center, double side_bias,
                                 int side_sign, int max_expansions,
                                 uint32_t& expansions_out) const;
    Vec2d sideTangent(const Vec2d& p, const Vec2d& center, double R,
                      const Vec2d& axis, bool left_side) const;
    Vec2d projectToLegalCell(const Vec2d& pt, double clearance) const;
    bool straightSafe(const Vec2d& a, const Vec2d& b, double clearance) const;
    void losShortcut(std::vector<Vec2d>& path, double clearance) const;
    /// Continuous polyline verification at `clearance` (sampled <= res/2).
    bool routeSafe(const std::vector<Vec2d>& path, double clearance) const;
    bool passesBlockerOnSide(const Vec2d& side_axis, const Vec2d& side_center,
                             const Vec2d& blocker_center,
                             double blocker_radius, bool left_side,
                             const std::vector<Vec2d>& path) const;
    /// Build one side route (tangent gateways + 3-segment side-A* + LOS +
    /// homotopy).  Fills `out` result.
    void planSideRoute(const Vec2d& start, const Vec2d& goal,
                       const Vec2d& blocker_center, double blocker_radius,
                       const Vec2d& axis_u, bool left_side,
                       SideRouteResult& out) const;

    // ── scene-static state ─────────────────────────────────────────
    BlueprintGenerationConfig cfg_;
    Vec2d min_{0.0, 0.0};
    Vec2d max_{0.0, 0.0};
    double res_ = 0.1;
    int w_ = 0, h_ = 0;
    std::vector<double> grid_;  // analytic ESDF (obstacle + boundary)
    std::vector<Vec2d> centers_;
    std::vector<double> radii_;
    std::vector<int> large_obs_;
    // Connectivity components of the qualifier's own ESDF grid at the
    // endpoint clearance (exact 2D semantics; built once per scene).
    std::vector<int> comp_;       // -1 = not selectable, else component id
    int main_comp_ = -1;
    std::vector<int> comp_areas_;
    /// Component label of a grid cell (built in configure()).
    int componentAt(int ix, int iy) const {
        return inGrid(ix, iy) ? comp_[idOf(ix, iy)] : -1;
    }
    /// Component of a world point at the endpoint clearance.
    int componentAtPoint(const Vec2d& p) const {
        return componentAt(ixOf(p.x()), iyOf(p.y()));
    }

    // Reusable A* buffers (generation-marked, never reallocated per query).
    mutable std::vector<uint32_t> gen_;      // visited (open or closed) epoch
    mutable std::vector<uint32_t> closed_;   // expanded epoch
    mutable std::vector<double> gcost_;
    mutable std::vector<int> parent_;
    mutable uint32_t epoch_ = 1;
};

}  // namespace expert
}  // namespace il_dataset
