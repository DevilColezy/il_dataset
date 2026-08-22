#pragma once
/// @file   ego_bspline.hpp
/// @brief  EGO-style optimisation-based cubic B-spline local path planner.
///
/// STRUCTURE COPIED FROM THE OPEN-SOURCE EGO-PLANNER
/// (ZJU-FAST-Lab/ego-planner, https://github.com/ZJU-FAST-Lab/ego-planner):
///   * uniform cubic B-spline over control points;
///   * cost terms  calcSmoothnessCost (jerk elastic band),
///     calcDistanceCostRebound (collision push from the distance field),
///     calcFeasibilityCost (velocity / acceleration limits),
///     calcFitnessCost (guidance toward a straight reference);
///   * combined cost f = λ1·smooth + λ2·collision + λ3·feasibility (+ fitness);
///   * L-BFGS over the FREE control points (solver: lbfgs.hpp, the same
///     open-source L-BFGS EGO bundles), then time re-allocation.
///
/// ADAPTATIONS to this package's sensor / dynamics reality:
///   * the signed-distance field is replaced by the sparse OBSERVED grid:
///     the collision distance/gradient come from `nearestOccupied` (the
///     nearest OCCUPIED cell centre — the observed obstacle surface ring);
///   * a soft FOV penalty keeps the optimised path inside the current
///     visible wedge (planning blind outside the FOV is never allowed);
///   * the optimised geometry is re-parameterised with the SAME
///     accel-limited speed profile and validated with the SAME hard
///     clearance + dynamic envelope as the straight-line plan
///     (planFovTrajectory), so it is a drop-in replacement, not a riskier
///     path.
///
/// The straight-ray degeneracy of the previous "B-spline" (4 collinear
/// control points ⇒ zero curvature) is the exact failure this fixes: the
/// optimiser BENDS the control points away from observed obstacles while
/// smoothness + guidance keep it short and goal-directed.

#include "il_dataset/hierarchical_expert/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace il_dataset {
namespace expert {

class EgoBsplineOptimizer {
public:
    /// Tuning surface (defaults match the planner params; exposed through
    /// Params2D.ego_* + the yaml `local_planner.ego_*` section).
    struct Config {
        // EGO cost weights.
        double lambda_smooth = 0.5;
        double lambda_collision = 2.0;  // sample-based gradient is thinner
        double lambda_feasibility = 0.2;
        double lambda_fitness = 0.8;    // toward the detour guide
        double lambda_fov = 0.3;
        // EGO collision margin: CURVE samples are pushed out until they are
        // >= clearance_m from an OCCUPIED cell centre (USER DIRECTIVE: 4
        // cells = 0.4 m = drone radius 0.3 + cell 0.1).  demarcation =
        // smooth band of the cubic -> quadratic cost transition (EGO dist0).
        double clearance_m = 0.4;
        double demarcation = 0.4;
        // Detour guide (our substitute for EGO's A* guide): the straight
        // reference is pushed laterally around the first blocking obstacle
        // cluster; the curve is then attracted toward the guide.
        double guide_clearance_m = 0.85;
        // Fitness cross-track denominator (EGO b2).  With a STRAIGHT
        // reference b2=1 fights the collision bend; with the DETOUR guide a
        // moderate value keeps the curve on the guide while still smoothing
        // its corners.
        double fitness_cross_b2 = 5.0;
        // Geometry.
        int n_segments = 8;     // B-spline segments (>= 4)
        double ts = 0.4;        // initial knot span (s); re-allocated later
        // Feasibility limits (EGO max_vel / max_acc).
        double max_vel = 2.0;
        double max_acc = 2.0;
        // Optimizer.
        int max_iter = 60;
        double nearest_search_r = 1.6;  // nearest-OCCUPIED search radius (m)
        // Soft FOV bound (rad) for the FOV penalty term.
        double fov_half_rad = 0.785398; // physical 90 deg camera FOV / 2
        // Speed profile / validation (mirrors local_planner_30hz params).
        double cruise_mps = 2.0;
        double eff_accel_mps2 = 2.0;
        double min_clearance = 0.4;     // 4 cells = collision distance
        double obstacle_reaction_time_s = 0.2;
        double soft_clearance_radius_m = 1.0;
        double handoff_clearance_m = 0.4;  // 4 cells = collision distance
        double obs_range_m = 5.0;
        double obs_resolution = 0.1;
        double obs_fov_deg = 90.0;  // hard FOV validation bound (±fov/2)
        double lp_dt = 0.1;
        double horizon_s = 4.0;
    };

    /// Optimise a cubic B-spline from `state` toward `endpoint` and return
    /// a validated, time-parameterised PlanarTrajectory (invalid when no
    /// safe path exists).  `terminal == true` → full stop at the endpoint;
    /// otherwise the endpoint is a FOV-boundary waypoint (cruise end speed).
    /// `min_clear` receives the smallest observed OCCUPIED-centre distance
    /// along the validated path (inf when no path).
    PlanarTrajectory plan(const PlanarState& state,
                          const LocalObservation& obs,
                          const Vec2d& endpoint, double v_end, bool terminal,
                          const Config& cfg, double& min_clear) const;

    /// Same as plan(), but the B-spline free control points are INITIALISED
    /// along a REAL A* route (`astar_path`, resampled to control-point
    /// resolution) instead of the straight-line detour guide — so the
    /// optimised curve follows the obstacle-free corridor, curves around
    /// blockers and ends AT the endpoint (USER ARCHITECTURE: A* →
    /// time-parameterised dynamics-feasible B-spline).  Falls back to the
    /// plain plan() when the path is unusable.
    PlanarTrajectory plan(const PlanarState& state,
                          const LocalObservation& obs,
                          const Vec2d& endpoint, double v_end, bool terminal,
                          const Config& cfg, double& min_clear,
                          const std::vector<Vec2d>& astar_path) const;

private:
    /// Uniform cubic B-spline: control points Q[0..M], M = n_segments + 3,
    /// knot span ts.  The curve passes through Q[1] at the start (departing
    /// along Q[2]-Q[0]) and Q[M-2] at the end (arriving along Q[M]-Q[M-2]).
    struct CtrlPts {
        std::vector<Vec2d> q;  // size M + 1 = n_segments + 4
        double ts = 0.4;
    };

    static Vec2d evalSpline(const CtrlPts& c, int seg, double u);
    static Vec2d evalSplineTangent(const CtrlPts& c, int seg, double u);

    // EGO cost terms (structure from ego_planner bspline_optimizer.cpp).
    // Each writes the FULL control-point gradient (grad.size() = q.size()).
    static double smoothnessCost(const CtrlPts& c, std::vector<Vec2d>& grad);
    /// SAMPLE-based collision: dense curve samples are pushed away from the
    /// nearest OCCUPIED cell and the gradient is distributed to the four
    /// segment control points through the B-spline basis weights (EGO's
    /// calcDistanceCostRebound applied on the curve, not the control points).
    double collisionCost(const CtrlPts& c, const LocalObservation& obs,
                         const Config& cfg, std::vector<Vec2d>& grad) const;
    static double feasibilityCost(const CtrlPts& c, const Config& cfg,
                                  std::vector<Vec2d>& grad);
    /// Fitness toward the DETOUR guide (EGO calcFitnessCost): keeps the
    /// curve on the guide (which already avoids obstacles) and advancing
    /// toward the subgoal.
    double fitnessCost(const CtrlPts& c, const std::vector<Vec2d>& guide,
                       const Config& cfg, std::vector<Vec2d>& grad) const;
    double fovCost(const CtrlPts& c, const PlanarState& state,
                   const Config& cfg, std::vector<Vec2d>& grad) const;

    /// Build a lateral detour guide around the first blocking obstacle
    /// cluster along the straight line start->endpoint (our substitute for
    /// EGO's A* guide path).  Returns M+1 guide points at control-point
    /// resolution.
    static std::vector<Vec2d> buildDetourGuide(
        const PlanarState& state, const LocalObservation& obs,
        const Vec2d& endpoint, int M, const Config& cfg);

    /// Resample an A* grid polyline to M+1 guide points, distributed by
    /// arc length (control-point resolution).  Empty when the path is too
    /// short.
    static std::vector<Vec2d> resamplePathToGuide(
        const std::vector<Vec2d>& path, int M);

    /// Build the clamped control points (initialised along the detour
    /// guide or an A* path when `guide_override` is non-null) and optimise
    /// the free ones with L-BFGS.  `dep_dir` is the start departure
    /// direction, `dir` the end approach direction, L the
    /// start->endpoint distance.  Non-static (const) because it calls the
    /// non-static const cost terms collisionCost / fitnessCost / fovCost.
    bool optimizeControlPoints(const PlanarState& state,
                               const LocalObservation& obs,
                               const Vec2d& endpoint, bool terminal,
                               const Config& cfg, const Vec2d& dep_dir,
                               const Vec2d& dir, double L,
                               const std::vector<Vec2d>* guide_override,
                               CtrlPts& out) const;

    /// Shared plan() body; `guide_override` selects the fitness/init guide
    /// (a resampled A* path) instead of the straight-line detour guide.
    PlanarTrajectory planImpl(const PlanarState& state,
                              const LocalObservation& obs,
                              const Vec2d& endpoint, double v_end,
                              bool terminal, const Config& cfg,
                              double& min_clear,
                              const std::vector<Vec2d>* guide_override) const;

    /// Sample + time-parameterise (accel-limited speed profile) + validate
    /// the optimised spline at one cruise level.  `allow_fov_exit == true`
    /// (A*-path plans) skips the HARD front-3 m FOV bearing gate — the A*
    /// route around a blocker can legitimately leave the instantaneous
    /// FOV (real EGO has no such gate; collision is enforced by the
    /// sample-based cost + 30 Hz replan).  The front clearance check stays.
    static bool buildAndValidate(const CtrlPts& c, const PlanarState& state,
                                 const LocalObservation& obs,
                                 const Vec2d& endpoint, double v_start,
                                 double v_end, bool terminal,
                                 const Config& cfg, double cruise,
                                 bool allow_fov_exit, PlanarTrajectory& out,
                                 double& min_clear);
};

}  // namespace expert
}  // namespace il_dataset
