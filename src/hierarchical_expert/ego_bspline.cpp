#include "il_dataset/hierarchical_expert/ego_bspline.hpp"

#include "il_dataset/hierarchical_expert/lbfgs.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace il_dataset {
namespace expert {

// ═══════════════════════════════════════════════════════════════════
//  Uniform cubic B-spline evaluation (EGO UniformBspline segment form)
// ═══════════════════════════════════════════════════════════════════
namespace {

/// Segment formula of a uniform cubic B-spline (standard basis matrix):
///   p(u) = 1/6 [ (1-u)^3 Qi + (3u^3-6u^2+4) Q(i+1)
///               + (-3u^3+3u^2+3u+1) Q(i+2) + u^3 Q(i+3) ],  u in [0,1].
inline Vec2d segmentPoint(const Vec2d& q0, const Vec2d& q1, const Vec2d& q2,
                          const Vec2d& q3, double u) {
    const double a = (1.0 - u) * (1.0 - u) * (1.0 - u);
    const double b = 3.0 * u * u * u - 6.0 * u * u + 4.0;
    const double c = -3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0;
    const double d = u * u * u;
    return (q0 * a + q1 * b + q2 * c + q3 * d) / 6.0;
}

/// Derivative w.r.t. u (world-unit tangent is /ts, irrelevant here since we
/// only need the DIRECTION for yaw and closing-speed).
inline Vec2d segmentTangent(const Vec2d& q0, const Vec2d& q1, const Vec2d& q2,
                            const Vec2d& q3, double u) {
    const double a = -3.0 * (1.0 - u) * (1.0 - u);
    const double b = 9.0 * u * u - 12.0 * u;
    const double c = -9.0 * u * u + 6.0 * u + 3.0;
    const double d = 3.0 * u * u;
    return (q0 * a + q1 * b + q2 * c + q3 * d) / 6.0;
}

// The optimiser and every fallback share the same physical stopping
// envelope; a low collision cost is not by itself a safety certificate.
double requiredClearance(double closing_speed,
                         const EgoBsplineOptimizer::Config& cfg) {
    const double s = std::max(0.0, closing_speed);
    const double a = std::max(1e-6, cfg.eff_accel_mps2);
    return cfg.handoff_clearance_m +
           s * std::max(0.0, cfg.obstacle_reaction_time_s) +
           s * s / (2.0 * a);
}

double clearanceSearchRadius(double total_speed,
                             const EgoBsplineOptimizer::Config& cfg) {
    return std::max(cfg.soft_clearance_radius_m,
                    requiredClearance(total_speed, cfg));
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════
//  Spline evaluation
// ═══════════════════════════════════════════════════════════════════
Vec2d EgoBsplineOptimizer::evalSpline(const CtrlPts& c, int seg, double u) {
    const int M = static_cast<int>(c.q.size()) - 1;
    seg = std::max(0, std::min(seg, M - 3));
    return segmentPoint(c.q[seg], c.q[seg + 1], c.q[seg + 2], c.q[seg + 3],
                        u);
}

Vec2d EgoBsplineOptimizer::evalSplineTangent(const CtrlPts& c, int seg,
                                             double u) {
    const int M = static_cast<int>(c.q.size()) - 1;
    seg = std::max(0, std::min(seg, M - 3));
    return segmentTangent(c.q[seg], c.q[seg + 1], c.q[seg + 2], c.q[seg + 3],
                          u);
}

// ═══════════════════════════════════════════════════════════════════
//  EGO cost terms (structure from ego_planner bspline_optimizer.cpp)
// ═══════════════════════════════════════════════════════════════════
double EgoBsplineOptimizer::smoothnessCost(const CtrlPts& c,
                                           std::vector<Vec2d>& grad) {
    const int M = static_cast<int>(c.q.size()) - 1;
    double cost = 0.0;
    for (int i = 0; i <= M - 3; ++i) {
        // EGO calcSmoothnessCost (jerk elastic band): third difference of
        // the control points is proportional to jerk for a uniform cubic
        // B-spline.
        const Vec2d jerk = c.q[i + 3] - 3.0 * c.q[i + 2] + 3.0 * c.q[i + 1] -
                           c.q[i];
        cost += jerk.squaredNorm();
        const Vec2d gj = 2.0 * jerk;
        grad[i] += -gj;
        grad[i + 1] += 3.0 * gj;
        grad[i + 2] += -3.0 * gj;
        grad[i + 3] += gj;
    }
    return cost;
}

double EgoBsplineOptimizer::collisionCost(const CtrlPts& c,
                                          const LocalObservation& obs,
                                          const Config& cfg,
                                          std::vector<Vec2d>& grad) const {
    double cost = 0.0;
    const int M = static_cast<int>(c.q.size()) - 1;
    const int n_seg = M - 3;
    // EGO calcDistanceCostRebound: cubic below `demarcation`, quadratic
    // above, gradient along the obstacle-to-sample direction.  Applied on
    // DENSE CURVE SAMPLES (not only the control points): the collision
    // gradient is distributed to the four segment control points through
    // the B-spline basis weights, so it has a real LATERAL component even
    // for a centred obstacle (control-point-only pushes were axial and the
    // curve stayed on the straight ray through the obstacle).
    const double a = 3.0 * cfg.demarcation;
    const double b = -3.0 * cfg.demarcation * cfg.demarcation;
    const double ccc =
        cfg.demarcation * cfg.demarcation * cfg.demarcation;
    constexpr int kSamplesPerSeg = 12;
    for (int si = 0; si < n_seg; ++si) {
        for (int k = 0; k <= kSamplesPerSeg; ++k) {
            const double u =
                static_cast<double>(k) / static_cast<double>(kSamplesPerSeg);
            const Vec2d p =
                segmentPoint(c.q[si], c.q[si + 1], c.q[si + 2], c.q[si + 3],
                             u);
            const NearestOccupiedResult nr =
                obs.nearestOccupied(p, cfg.nearest_search_r);
            if (!nr.found) continue;
            const double dist = nr.distance;
            const double dist_err = cfg.clearance_m - dist;
            if (dist_err < 0.0) continue;
            Vec2d dir_grad(0.0, 0.0);
            if (dist > 1e-9) dir_grad = (p - nr.cell_center) / dist;
            Vec2d gp(0.0, 0.0);
            if (dist_err < cfg.demarcation) {
                cost += dist_err * dist_err * dist_err;
                gp = -3.0 * dist_err * dist_err * dir_grad;
            } else {
                cost += a * dist_err * dist_err + b * dist_err + ccc;
                gp = -(2.0 * a * dist_err + b) * dir_grad;
            }
            // Basis weights of the cubic B-spline segment (p = Σ w_k Q).
            const double w0 = (1.0 - u) * (1.0 - u) * (1.0 - u) / 6.0;
            const double w1 = (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0;
            const double w2 =
                (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0;
            const double w3 = u * u * u / 6.0;
            grad[si] += w0 * gp;
            grad[si + 1] += w1 * gp;
            grad[si + 2] += w2 * gp;
            grad[si + 3] += w3 * gp;
        }
    }
    return cost;
}

double EgoBsplineOptimizer::feasibilityCost(const CtrlPts& c,
                                            const Config& cfg,
                                            std::vector<Vec2d>& grad) {
    double cost = 0.0;
    const int M = static_cast<int>(c.q.size()) - 1;
    const double ts = c.ts;
    const double ts_inv2 = 1.0 / (ts * ts);
    // Velocity feasibility: vi = (Q(i+1)-Q(i))/ts (EGO calcFeasibilityCost).
    for (int i = 0; i < M; ++i) {
        const Vec2d vi = (c.q[i + 1] - c.q[i]) / ts;
        const double vn = vi.norm();
        if (vn > cfg.max_vel) {
            const double diff = vn - cfg.max_vel;
            cost += diff * diff * ts_inv2;
            const Vec2d dv = (vn > 1e-9) ? (vi / vn) : Vec2d(1.0, 0.0);
            const Vec2d gv = 2.0 * diff * ts_inv2 * dv / ts;
            grad[i] -= gv;
            grad[i + 1] += gv;
        }
    }
    // Acceleration feasibility: ai = (Q(i+2)-2Q(i+1)+Q(i))/ts^2.
    for (int i = 0; i < M - 1; ++i) {
        const Vec2d ai = (c.q[i + 2] - 2.0 * c.q[i + 1] + c.q[i]) * ts_inv2;
        const double an = ai.norm();
        if (an > cfg.max_acc) {
            const double diff = an - cfg.max_acc;
            cost += diff * diff;
            const Vec2d da = (an > 1e-9) ? (ai / an) : Vec2d(1.0, 0.0);
            const Vec2d ga = 2.0 * diff * ts_inv2 * da;
            grad[i] += ga;
            grad[i + 1] -= 2.0 * ga;
            grad[i + 2] += ga;
        }
    }
    return cost;
}

double EgoBsplineOptimizer::fitnessCost(const CtrlPts& c,
                                        const std::vector<Vec2d>& guide,
                                        const Config& cfg,
                                        std::vector<Vec2d>& grad) const {
    double cost = 0.0;
    const int M = static_cast<int>(c.q.size()) - 1;
    if (static_cast<int>(guide.size()) != M + 1) return 0.0;
    // EGO calcFitnessCost toward the DETOUR guide: x = spline point near
    // control point i minus the guide reference; v = local guide direction;
    // f = (x·v)^2/25 + |x×v|^2 / fitness_cross_b2.  The cross-track term is
    // moderated (fitness_cross_b2 = 5, not EGO's 1) because our guide is a
    // coarse polyline: a strong b2 would hug the guide's corners.
    const double a2 = 25.0, b2 = cfg.fitness_cross_b2;
    for (int i = 1; i < M; ++i) {
        const Vec2d x = (c.q[i - 1] + 4.0 * c.q[i] + c.q[i + 1]) / 6.0 -
                        guide[i];
        Vec2d v = guide[std::min(M, i + 1)] - guide[std::max(0, i - 1)];
        const double vn = v.norm();
        if (vn > 1e-9) v /= vn;
        const double xdotv = x.dot(v);
        const double xcrossv = x.x() * v.y() - x.y() * v.x();  // 2D cross
        cost += (xdotv * xdotv) / a2 + (xcrossv * xcrossv) / b2;
        const Vec2d dfdx = 2.0 * xdotv / a2 * v +
                           2.0 / b2 * xcrossv * Vec2d(v.y(), -v.x());
        grad[i - 1] += dfdx / 6.0;
        grad[i] += 4.0 * dfdx / 6.0;
        grad[i + 1] += dfdx / 6.0;
    }
    return cost;
}

double EgoBsplineOptimizer::referenceCost(const CtrlPts& c,
                                          const std::vector<Vec2d>& ref,
                                          const Config& cfg,
                                          std::vector<Vec2d>& grad) {
    (void)cfg;
    double cost = 0.0;
    const int M = static_cast<int>(c.q.size()) - 1;
    if (static_cast<int>(ref.size()) != M + 1) return 0.0;
    // Soft temporal anchor: pull the FREE control points toward the previous
    // plan's curve (resampled to control-point resolution).  Control-point
    // space is used for a simple gradient; the reference guide was built
    // from the previous CURVE, so this keeps consecutive replans within ~a
    // segment of the committed path without hard-constraining the detour.
    for (int i = 3; i <= M - 4; ++i) {
        const Vec2d d = c.q[i] - ref[i];
        cost += d.squaredNorm();
        grad[i] += 2.0 * d;
    }
    return cost;
}

double EgoBsplineOptimizer::fovCost(const CtrlPts& c,
                                    const PlanarState& state,
                                    const Config& cfg,
                                    std::vector<Vec2d>& grad) const {
    // Soft penalty keeping FREE control points inside the usable FOV band.
    double cost = 0.0;
    const int M = static_cast<int>(c.q.size()) - 1;
    for (int i = 3; i <= M - 4; ++i) {
        const Vec2d rel = c.q[i] - state.position;
        const double r = rel.norm();
        if (r < 1e-9) continue;
        const double b = wrapAngle(std::atan2(rel.y(), rel.x()) - state.yaw);
        const double over = std::fabs(b) - cfg.fov_half_rad;
        if (over <= 0.0) continue;
        cost += over * over;
        // d|b|/dQ = sign(b) * [-rel.y/r^2, rel.x/r^2]
        const Vec2d db(-rel.y() / (r * r), rel.x() / (r * r));
        grad[i] += 2.0 * over * (b > 0.0 ? 1.0 : -1.0) * db;
    }
    return cost;
}

// ═══════════════════════════════════════════════════════════════════
//  Detour guide construction + control points + L-BFGS optimisation
// ═══════════════════════════════════════════════════════════════════
std::vector<Vec2d> EgoBsplineOptimizer::buildDetourGuide(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, int M, const Config& cfg) {
    // EGO's BsplineOptimizer is initialised from an A* guide path that
    // already detours around obstacles.  We have no A*: build a coarse
    // guide that pushes the straight reference laterally around the first
    // blocking obstacle cluster, using the OBSERVED OCCUPIED cells.
    std::vector<Vec2d> guide;
    const Vec2d to = endpoint - state.position;
    const double L = to.norm();
    if (L < 1e-9) {
        guide.assign(M + 1, state.position);
        return guide;
    }
    const Vec2d dir = to / L;
    const Vec2d perp(-dir.y(), dir.x());
    guide.resize(M + 1);
    for (int i = 0; i <= M; ++i) {
        guide[i] = state.position + dir * (L * static_cast<double>(i) /
                                           static_cast<double>(M));
    }
    // Determine a CONSISTENT detour side from the mean lateral component of
    // the radial push of all blocked guide points.
    double side_sum = 0.0;
    for (int i = 0; i <= M; ++i) {
        const NearestOccupiedResult nr =
            obs.nearestOccupied(guide[i], cfg.nearest_search_r);
        if (!nr.found) continue;
        if (nr.distance < cfg.guide_clearance_m) {
            const Vec2d radial =
                (guide[i] - nr.cell_center) / std::max(nr.distance, 1e-9);
            side_sum += radial.dot(perp);
        }
    }
    const double sgn = side_sum >= 0.0 ? 1.0 : -1.0;
    // Per-point lateral offset, then smooth it (box filter) so the guide
    // does not have sharp corners.
    std::vector<double> offs(M + 1, 0.0);
    for (int i = 0; i <= M; ++i) {
        const NearestOccupiedResult nr =
            obs.nearestOccupied(guide[i], cfg.nearest_search_r);
        if (nr.found && nr.distance < cfg.guide_clearance_m) {
            offs[i] = cfg.guide_clearance_m - nr.distance;
        }
    }
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<double> no(M + 1, 0.0);
        no[0] = offs[0];
        no[M] = offs[M];
        for (int i = 1; i < M; ++i) {
            no[i] = (offs[i - 1] + offs[i] + offs[i + 1]) / 3.0;
        }
        offs.swap(no);
    }
    for (int i = 0; i <= M; ++i) {
        guide[i] += perp * (sgn * offs[i]);
    }
    return guide;
}

// ═══════════════════════════════════════════════════════════════════
//  A* path resampling (USER ARCHITECTURE A+B)
// ═══════════════════════════════════════════════════════════════════
std::vector<Vec2d> EgoBsplineOptimizer::resamplePathToGuide(
    const std::vector<Vec2d>& path, int M) {
    std::vector<Vec2d> guide;
    if (path.size() < 2 || M < 1) return guide;
    std::vector<double> acc(path.size(), 0.0);
    for (size_t i = 1; i < path.size(); ++i) {
        acc[i] = acc[i - 1] + (path[i] - path[i - 1]).norm();
    }
    const double total = acc.back();
    if (total < 1e-9) {
        guide.assign(M + 1, path.front());
        return guide;
    }
    guide.resize(M + 1);
    guide[0] = path.front();
    guide[M] = path.back();
    size_t j = 0;
    for (int i = 1; i < M; ++i) {
        const double target = total * static_cast<double>(i) /
                              static_cast<double>(M);
        while (j + 1 < path.size() && acc[j + 1] < target) ++j;
        if (j + 1 >= path.size()) {
            guide[i] = path.back();
            continue;
        }
        const double seg = acc[j + 1] - acc[j];
        const double w = seg > 1e-9 ? (target - acc[j]) / seg : 0.0;
        guide[i] = path[j] + (path[j + 1] - path[j]) * w;
    }
    return guide;
}

bool EgoBsplineOptimizer::optimizeControlPoints(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, bool terminal, const Config& cfg,
    const Vec2d& dep_dir, const Vec2d& dir, double L,
    const std::vector<Vec2d>* guide_override,
    const std::vector<Vec2d>* ref_guide, CtrlPts& out) const {
    (void)terminal;
    const int M = cfg.n_segments + 3;
    const int n_free = M - 6;  // free control points Q[3..M-4]
    if (n_free < 1) return false;
    const double d = L / 6.0;
    const double speed = state.velocity_world.norm();
    const double departure_d =
        speed > 0.05 ? std::min(d, std::max(0.15, 0.35 * speed)) : d;

    // Guide reference used for the fitness term: a REAL A* path when
    // provided (USER A+B — the curve follows the obstacle-free corridor),
    // otherwise the lateral detour guide (straight line + obstacle push).
    const std::vector<Vec2d> guide =
        guide_override ? *guide_override
                       : buildDetourGuide(state, obs, endpoint, M, cfg);
    // R27 temporal anchor: a compatible previous plan (resampled to M+1
    // control-point positions).  When available it WARM-STARTS the free
    // control points (temporal continuity) and, if lambda_ref > 0, adds the
    // anchoring cost — the guide still drives the obstacle-fitness term.
    const bool use_ref = ref_guide != nullptr &&
                         static_cast<int>(ref_guide->size()) == M + 1;

    CtrlPts c;
    c.q.assign(M + 1, Vec2d(0.0, 0.0));
    c.ts = cfg.ts;
    // Clamped start: curve passes through Q[1] = start, departing along
    // dep_dir (tangent ∝ Q2 - Q0).
    c.q[0] = state.position - dep_dir * departure_d;
    c.q[1] = state.position;
    c.q[2] = state.position + dep_dir * departure_d;
    // Clamped end: curve passes through Q[M-2] = endpoint, arriving along
    // dir (tangent ∝ Q[M] - Q[M-2]).
    c.q[M - 3] = endpoint - dir * d;
    c.q[M - 2] = endpoint;
    c.q[M - 1] = endpoint + dir * d;
    c.q[M] = endpoint + dir * (2.0 * d);
    // Initialise the free control points ALONG the temporal reference when
    // present (previous plan), else along the detour guide / A* route.
    for (int i = 3; i <= M - 4; ++i) {
        c.q[i] = use_ref ? (*ref_guide)[i] : guide[i];
    }

    // Pack the free control points into the L-BFGS variable vector.
    Eigen::VectorXd x(2 * n_free);
    for (int j = 0; j < n_free; ++j) {
        x[2 * j] = c.q[3 + j].x();
        x[2 * j + 1] = c.q[3 + j].y();
    }

    // Objective: f = λ1*smooth + λ2*collision + λ3*feasibility +
    //             λ4*fitness + λfov*fov + λref*reference  (EGO combined).
    const auto obj = [&](const Eigen::VectorXd& xv,
                         Eigen::VectorXd& g) -> double {
        for (int j = 0; j < n_free; ++j) {
            c.q[3 + j] = Vec2d(xv[2 * j], xv[2 * j + 1]);
        }
        std::vector<Vec2d> gs(M + 1, Vec2d(0.0, 0.0));
        std::vector<Vec2d> gc(M + 1, Vec2d(0.0, 0.0));
        std::vector<Vec2d> gf(M + 1, Vec2d(0.0, 0.0));
        std::vector<Vec2d> gfit(M + 1, Vec2d(0.0, 0.0));
        std::vector<Vec2d> gfo(M + 1, Vec2d(0.0, 0.0));
        std::vector<Vec2d> gref(M + 1, Vec2d(0.0, 0.0));
        const double cs = smoothnessCost(c, gs);
        const double cc = collisionCost(c, obs, cfg, gc);
        const double cf = feasibilityCost(c, cfg, gf);
        const double cfit = fitnessCost(c, guide, cfg, gfit);
        const double cfo = fovCost(c, state, cfg, gfo);
        const double cref =
            (cfg.lambda_ref > 0.0 && use_ref)
                ? referenceCost(c, *ref_guide, cfg, gref)
                : 0.0;
        const double cost = cfg.lambda_smooth * cs + cfg.lambda_collision * cc +
                            cfg.lambda_feasibility * cf +
                            cfg.lambda_fitness * cfit + cfg.lambda_fov * cfo +
                            cfg.lambda_ref * cref;
        for (int j = 0; j < n_free; ++j) {
            const int i = 3 + j;
            g[2 * j] = cfg.lambda_smooth * gs[i].x() +
                       cfg.lambda_collision * gc[i].x() +
                       cfg.lambda_feasibility * gf[i].x() +
                       cfg.lambda_fitness * gfit[i].x() +
                       cfg.lambda_fov * gfo[i].x() +
                       cfg.lambda_ref * gref[i].x();
            g[2 * j + 1] = cfg.lambda_smooth * gs[i].y() +
                           cfg.lambda_collision * gc[i].y() +
                           cfg.lambda_feasibility * gf[i].y() +
                           cfg.lambda_fitness * gfit[i].y() +
                           cfg.lambda_fov * gfo[i].y() +
                           cfg.lambda_ref * gref[i].y();
        }
        return cost;
    };

    LbfgsParams lp;
    lp.max_iterations = cfg.max_iter;
    lp.g_epsilon = 1e-5;
    lbfgsMinimize(x, obj, lp);

    // Copy the optimised control points back.
    for (int j = 0; j < n_free; ++j) {
        c.q[3 + j] = Vec2d(x[2 * j], x[2 * j + 1]);
    }
    out = std::move(c);
    return true;
}

// ═══════════════════════════════════════════════════════════════════
//  Time parameterisation + validation (mirrors planFovTrajectory)
// ═══════════════════════════════════════════════════════════════════
bool EgoBsplineOptimizer::buildAndValidate(
    const CtrlPts& c, const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_start, double v_end, bool terminal,
    const Config& cfg, double cruise, bool allow_fov_exit,
    PlanarTrajectory& out, double& min_clear) {
    (void)endpoint;
    (void)terminal;
    out = PlanarTrajectory{};
    out.valid = false;
    min_clear = std::numeric_limits<double>::infinity();
    const int n_seg = static_cast<int>(c.q.size()) - 4;
    if (n_seg < 1) return false;
    const double fov_half = 0.5 * deg2rad(cfg.obs_fov_deg);
    const double a = cfg.eff_accel_mps2;

    // Dense arc-length table of the optimised spline.
    const int N = 64;
    std::vector<Vec2d> pts(N);
    std::vector<double> s(N, 0.0);
    pts[0] = evalSpline(c, 0, 0.0);
    for (int i = 1; i < N; ++i) {
        const double u = static_cast<double>(i) / (N - 1);
        const double seg = u * n_seg;
        const int si = std::max(0, std::min(n_seg - 1, static_cast<int>(seg)));
        const Vec2d p = evalSpline(c, si, seg - si);
        s[i] = s[i - 1] + (p - pts[i - 1]).norm();
        pts[i] = p;
    }
    const double Ls = std::max(s[N - 1], 1e-6);

    auto pointAt = [&](double ss) -> Vec2d {
        if (ss <= 0.0) return pts[0];
        if (ss >= Ls) return pts[N - 1];
        for (int i = 1; i < N; ++i) {
            if (s[i] >= ss) {
                const double w = (ss - s[i - 1]) /
                                 std::max(1e-9, s[i] - s[i - 1]);
                return pts[i - 1] + (pts[i] - pts[i - 1]) * w;
            }
        }
        return pts[N - 1];
    };
    auto speedAt = [&](double ss) -> double {
        const double v_acc = std::min(
            cruise, std::sqrt(std::max(0.0, v_start * v_start + 2.0 * a * ss)));
        const double v_dec = std::min(
            cruise, std::sqrt(std::max(
                        0.0, v_end * v_end + 2.0 * a * (Ls - ss))));
        return std::min(v_acc, v_dec);
    };

    // ── Receding-horizon validation (EGO-style, R20d) ────────────
    // Only the EXECUTED FRONT of the trajectory must be collision-free:
    // the drone re-plans every control tick (30 Hz), so the tail beyond
    // the front is never executed as-is — it is re-routed as new
    // observation arrives.  Hard-gating the whole path to the far endpoint
    // made the planner dead-end (NO_SAFE_CANDIDATE) whenever a blocker sat
    // near the endpoint even though the front was clear.  The front covers
    // the braking + reaction distance so the drone can always stop safely
    // within the validated portion.
    const double front_dist =
        std::max(2.0, std::min(3.0, cruise * 1.5));
    // R28: when a corridor block forces validation beyond the front window
    // (validate_front_m > 0), extend the CLEARANCE check out to the block.
    // The FOV gate stays at the base front — the drone turns into the curve
    // as it advances, and A*-route plans skip the FOV gate anyway.  This
    // makes a straight chord through a blocked corridor FAIL validation
    // (instead of "passing" because the blocker sits beyond the ~3 m front)
    // so the bearing scan continues to a genuine detour (task 440: BLOCKED
    // + unnecessary macro takeover on first sight of a small obstacle).
    const double front_clear =
        cfg.validate_front_m > 0.0
            ? std::max(front_dist, std::min(cfg.validate_front_m, Ls))
            : front_dist;

    out.points.push_back(state.position);
    out.yaw.push_back(state.yaw);
    out.t.push_back(0.0);
    double ss = 0.0, cmin = std::numeric_limits<double>::infinity();
    Vec2d prev = state.position;
    bool reached = false, bad = false;
    for (double t_ctrl = cfg.lp_dt;
         t_ctrl <= cfg.horizon_s + 1e-6 && !reached; t_ctrl += cfg.lp_dt) {
        const double v_cur = speedAt(ss);
        ss = std::min(Ls, ss + v_cur * cfg.lp_dt +
                              0.5 * a * cfg.lp_dt * cfg.lp_dt);
        const Vec2d p = pointAt(ss);
        const Vec2d tangent = (p - prev).squaredNorm() > 1e-12
                                  ? (p - prev).normalized()
                                  : Vec2d(1.0, 0.0);
        const bool in_front_fov = ss <= front_dist + 1e-9;
        const bool in_front_clr = ss <= front_clear + 1e-9;
        if (in_front_fov || in_front_clr) {
            // FOV constraint (front only — the tail is re-planned).
            // USER DIRECTIVE (2026-08-20): EGO-style local planning from
            // the current state to the target.  The A*-route plan may
            // legitimately LEAVE the instantaneous FOV — the shortest
            // route around a blocker can head over/around it (measured:
            // task254 id2 at (4.07,15.05) r1.07; the south bypass needs a
            // ~+48° turn, the north bypass ~-132°, both outside the 45°
            // hard cone).  A hard front FOV gate here made every such plan
            // fail validation -> FOV-edge scan -> the drone crept to the
            // blocker edge and deadlocked (TURN_RIGHT spin).  Real EGO has
            // no hard FOV gate: the sample-based collision cost keeps the
            // executed front clear and the 30 Hz replan re-validates as the
            // drone turns.  Skipped ONLY for A*-path plans (allow_fov_exit);
            // the detour-guide path keeps the gate.
            if (!allow_fov_exit && in_front_fov) {
                const double b_p = wrapAngle(
                    std::atan2(p.y() - state.position.y(),
                               p.x() - state.position.x()) -
                    state.yaw);
                if (std::fabs(b_p) > fov_half + 1e-9) {
                    bad = true;
                    break;
                }
            }
            // Clearance vs observed OCCUPIED (sparse ESDF); UNKNOWN
            // passable.  Front only (extended to the corridor block when
            // validate_front_m > 0).
            if (!in_front_clr) continue;
            const double v = speedAt(ss);
            const double search_r = clearanceSearchRadius(v, cfg);
            bool hard_violation = false, env_violation = false;
            obs.forEachOccupiedWithin(
                p, search_r, cfg.max_history_age_ticks,
                [&](const Vec2d& centre, double distance) {
                    cmin = std::min(cmin, distance);
                    if (distance < cfg.min_clearance) hard_violation = true;
                    Vec2d dir_c = centre - p;
                    const double dl = dir_c.norm();
                    if (dl > 1e-9) dir_c /= dl;
                    const double closing =
                        std::max(0.0, dir_c.dot(tangent) * v);
                    if (distance < requiredClearance(closing, cfg))
                        env_violation = true;
                });
            if (hard_violation || env_violation) {
                bad = true;
                break;
            }
        }
        prev = p;
        out.points.push_back(p);
        out.yaw.push_back(std::atan2(tangent.y(), tangent.x()));
        out.t.push_back(t_ctrl);
        if (ss >= Ls - 1e-6) reached = true;
    }
    if (!bad && reached) {
        out.valid = true;
        out.cruise_mps = cruise;
        min_clear = cmin;
        return true;
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════
//  Public entry points
// ═══════════════════════════════════════════════════════════════════
PlanarTrajectory EgoBsplineOptimizer::plan(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_end, bool terminal,
    const Config& cfg, double& min_clear) const {
    return planImpl(state, obs, endpoint, v_end, terminal, cfg, min_clear,
                    /*guide_override=*/nullptr, /*ref_guide=*/nullptr);
}

PlanarTrajectory EgoBsplineOptimizer::plan(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_end, bool terminal,
    const Config& cfg, double& min_clear,
    const std::vector<Vec2d>& astar_path) const {
    const int M = cfg.n_segments + 3;
    std::vector<Vec2d> guide = resamplePathToGuide(astar_path, M);
    if (guide.size() != static_cast<size_t>(M + 1)) {
        return planImpl(state, obs, endpoint, v_end, terminal, cfg, min_clear,
                        /*guide_override=*/nullptr, /*ref_guide=*/nullptr);
    }
    return planImpl(state, obs, endpoint, v_end, terminal, cfg, min_clear,
                    &guide, /*ref_guide=*/nullptr);
}

PlanarTrajectory EgoBsplineOptimizer::planAnchored(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_end, bool terminal,
    const Config& cfg, double& min_clear,
    const std::vector<Vec2d>* astar_path,
    const std::vector<Vec2d>* ref_traj) const {
    const int M = cfg.n_segments + 3;
    // Guide (fitness + fallback init): A* route when provided.
    std::vector<Vec2d> guide;
    const std::vector<Vec2d>* guide_ptr = nullptr;
    if (astar_path != nullptr) {
        guide = resamplePathToGuide(*astar_path, M);
        if (guide.size() == static_cast<size_t>(M + 1)) guide_ptr = &guide;
    }
    // Temporal reference (warm-start + anchoring cost): previous plan.
    std::vector<Vec2d> ref_guide;
    const std::vector<Vec2d>* ref_ptr = nullptr;
    if (ref_traj != nullptr && !ref_traj->empty()) {
        ref_guide = resamplePathToGuide(*ref_traj, M);
        if (ref_guide.size() == static_cast<size_t>(M + 1)) ref_ptr = &ref_guide;
    }
    return planImpl(state, obs, endpoint, v_end, terminal, cfg, min_clear,
                    guide_ptr, ref_ptr);
}

PlanarTrajectory EgoBsplineOptimizer::planImpl(
    const PlanarState& state, const LocalObservation& obs,
    const Vec2d& endpoint, double v_end, bool terminal,
    const Config& cfg, double& min_clear,
    const std::vector<Vec2d>* guide_override,
    const std::vector<Vec2d>* ref_guide) const {
    PlanarTrajectory out;
    out.valid = false;
    min_clear = std::numeric_limits<double>::infinity();
    const Vec2d to = endpoint - state.position;
    const double L = to.norm();
    if (L < 1e-6) {
        if (terminal) {
            out.valid = true;
            out.points.push_back(state.position);
            out.yaw.push_back(state.yaw);
            out.t.push_back(0.0);
        }
        return out;
    }
    const Vec2d dir = to / L;
    const double v_start = state.velocity_world.norm();
    // Departure direction is a physical boundary condition.  Preserve the
    // measured velocity while moving, but keep the handle short so the
    // spline turns promptly toward the route/endpoint.  At standstill, use
    // the A* route's first segment (when available), otherwise the endpoint.
    const Vec2d dep_dir = [&]() -> Vec2d {
        // The departure tangent is a physical boundary condition.  Keep the
        // measured velocity direction while moving; route/endpoint direction
        // is used only after the vehicle is effectively stationary.
        if (v_start > 0.05) return state.velocity_world / v_start;
        if (guide_override) {
            if (guide_override->size() >= 2) {
                const Vec2d gd = (*guide_override)[1] - (*guide_override)[0];
                if (gd.squaredNorm() > 1e-12) return gd.normalized();
            }
        }
        return dir;
    }();

    CtrlPts c;
    if (!optimizeControlPoints(state, obs, endpoint, terminal, cfg, dep_dir,
                               dir, L, guide_override, ref_guide, c)) {
        return out;
    }

    // Multi-cruise retry: a tight gap is only passable at reduced speed
    // (the dynamic braking envelope grows with speed) — same as the
    // straight-line planner.  The GEOMETRY is optimised once; each cruise
    // level only re-times + re-validates.  Down to 1/32 of nominal so a
    // route whose clearance barely clears the hard floor (e.g. the
    // boxed-in pocket in task30, 0.82 m after the EGO bend) is still
    // flyable — the drone creeps out of the pocket, then re-plans faster
    // once clear.
    const double cruise_nom = std::min(cfg.cruise_mps, cfg.max_vel);
    const double cruises[6] = {cruise_nom,       0.5 * cruise_nom,
                               0.25 * cruise_nom, 0.125 * cruise_nom,
                               0.0625 * cruise_nom, 0.03125 * cruise_nom};
    // R28c: prefer the plan with the SMALLEST lateral bend across the cruise
    // retries (a slow straight thread through a gap beats a fast wide
    // detour).  The straight/small-bend candidate has cross-track ~0 and
    // wins whenever it validates at some cruise level.
    double mc = std::numeric_limits<double>::infinity();
    PlanarTrajectory best;
    double best_ct = std::numeric_limits<double>::infinity();
    double best_mc = std::numeric_limits<double>::infinity();
    for (double cruise : cruises) {
        if (cruise <= 1e-6) continue;
        PlanarTrajectory cand;
        if (buildAndValidate(c, state, obs, endpoint, v_start, v_end, terminal,
                             cfg, cruise,
                             /*allow_fov_exit=*/guide_override != nullptr,
                             cand, mc)) {
            double ct = 0.0;
            if (cand.points.size() >= 2) {
                const Vec2d ab = endpoint - state.position;
                const double L2 = ab.squaredNorm();
                if (L2 > 1e-12) {
                    for (const Vec2d& q : cand.points) {
                        const double u = std::max(
                            0.0, std::min(1.0, (q - state.position).dot(ab) / L2));
                        const Vec2d proj = state.position + ab * u;
                        ct = std::max(ct, (q - proj).norm());
                    }
                }
            }
            if (ct < best_ct) {
                best_ct = ct;
                best = std::move(cand);
                best_mc = mc;
            }
        }
    }
    if (best.valid) {
        min_clear = best_mc;
        return best;
    }
    return out;
}

}  // namespace expert
}  // namespace il_dataset
