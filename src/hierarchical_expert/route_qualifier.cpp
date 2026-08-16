#include "il_dataset/hierarchical_expert/route_qualifier.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>

namespace il_dataset {
namespace expert {

namespace {

// 8-neighbour offsets: even k = cardinal (cost 1), odd k = diagonal (cost
// sqrt2).  (di[k], dj[k]) order matches the 2D reference so the
// no-corner-cutting check below is identical.
const int kDi[8] = {1, 1, 0, -1, -1, -1, 0, 1};
const int kDj[8] = {0, 1, 1, 1, 0, -1, -1, -1};
const double kW[8] = {1.0, M_SQRT2, 1.0, M_SQRT2,
                      1.0, M_SQRT2, 1.0, M_SQRT2};

inline double cross2(const Vec2d& a, const Vec2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}
inline Vec2d rot2(const Vec2d& v, double a) {
    const double c = std::cos(a), s = std::sin(a);
    return Vec2d(c * v.x() - s * v.y(), s * v.x() + c * v.y());
}

}  // namespace

// ────────────────────────────────────────────────────────────────────
//  configure() — one-time scene-static truth-ESDF grid
// ────────────────────────────────────────────────────────────────────
void TaskRouteQualifier::configure(const BlueprintScene& scene,
                                   const SceneGeometryCache& geo,
                                   const BlueprintGenerationConfig& cfg) {
    cfg_ = cfg;
    min_ = cfg.warehouse.freeMin();
    max_ = cfg.warehouse.freeMax();
    res_ = std::max(1e-3, cfg.esdf_resolution_m);
    w_ = std::max(1, static_cast<int>(
                         std::ceil((max_.x() - min_.x()) / res_)));
    h_ = std::max(1, static_cast<int>(
                         std::ceil((max_.y() - min_.y()) / res_)));
    grid_.assign(static_cast<size_t>(w_) * h_,
                 std::numeric_limits<double>::infinity());

    // Analytic signed ESDF with the 2D reference semantics:
    //   esdf(p) = min( min over cylinders (||p-c|| - r),
    //                  distance from p to the inside of the rectangle )
    const auto& centers = geo.obstacleCenters();
    const auto& radii = geo.obstacleRadii();
    const size_t n_obs = std::min(centers.size(), radii.size());
    for (int iy = 0; iy < h_; ++iy) {
        for (int ix = 0; ix < w_; ++ix) {
            const Vec2d p = cellCenter(ix, iy);
            double best = boundaryDist(p);
            for (size_t i = 0; i < n_obs; ++i) {
                best = std::min(best, (p - centers[i]).norm() - radii[i]);
            }
            grid_[idOf(ix, iy)] = best;
        }
    }

    // Cache the scene-static obstacle geometry (single source: the cache).
    centers_ = geo.obstacleCenters();
    radii_ = geo.obstacleRadii();
    large_obs_ = geo.largeObstacles();

    // ── Connectivity components on the qualifier's OWN grid at the
    //    endpoint clearance (2D reference semantics: 8-neighbour flood
    //    fill, STRICT esdf > clearance, no diagonal corner cutting). ──
    const double endpoint_clr = cfg_.endpointRequiredClearance();
    const size_t n = static_cast<size_t>(w_) * h_;
    comp_.assign(n, -1);
    comp_areas_.clear();
    int label = 0;
    for (int iy = 0; iy < h_; ++iy) {
        for (int ix = 0; ix < w_; ++ix) {
            const size_t id = idOf(ix, iy);
            if (comp_[id] != -1 || !cellFreeStrict(ix, iy, endpoint_clr)) {
                continue;
            }
            int area = 0;
            std::queue<int> qx, qy;
            qx.push(ix);
            qy.push(iy);
            comp_[id] = label;
            while (!qx.empty()) {
                const int cx = qx.front(), cy = qy.front();
                qx.pop();
                qy.pop();
                ++area;
                for (int k = 0; k < 8; ++k) {
                    const int nx = cx + kDj[k], ny = cy + kDi[k];
                    if (!inGrid(nx, ny)) continue;
                    if (!cellFreeStrict(nx, ny, endpoint_clr)) continue;
                    // No diagonal corner cutting.
                    if ((k % 2 == 1) &&
                        (!cellFreeStrict(cx + kDj[k], cy, endpoint_clr) ||
                         !cellFreeStrict(cx, cy + kDi[k], endpoint_clr))) {
                        continue;
                    }
                    const size_t nid = idOf(nx, ny);
                    if (comp_[nid] != -1) continue;
                    comp_[nid] = label;
                    qx.push(nx);
                    qy.push(ny);
                }
            }
            comp_areas_.push_back(area);
            ++label;
        }
    }
    main_comp_ = -1;
    int best_area = -1;
    for (size_t c = 0; c < comp_areas_.size(); ++c) {
        if (comp_areas_[c] > best_area) {
            best_area = comp_areas_[c];
            main_comp_ = static_cast<int>(c);
        }
    }

    // Prepare the reusable A* buffers (never reallocated per query).
    gen_.assign(n, 0);
    closed_.assign(n, 0);
    gcost_.assign(n, 0.0);
    parent_.assign(n, -1);
    epoch_ = 1;
}

double TaskRouteQualifier::boundaryDist(const Vec2d& p) const {
    // Distance from p to the inside of the region rectangle.
    const double dx = std::min(p.x() - min_.x(), max_.x() - p.x());
    const double dy = std::min(p.y() - min_.y(), max_.y() - p.y());
    return std::min(dx, dy);
}

double TaskRouteQualifier::esdfAt(const Vec2d& p) const {
    if (p.x() < min_.x() || p.x() > max_.x() || p.y() < min_.y() ||
        p.y() > max_.y()) {
        // Outside the region: signed distance is negative (outside).
        const double dx = std::max(min_.x() - p.x(), p.x() - max_.x());
        const double dy = std::max(min_.y() - p.y(), p.y() - max_.y());
        return -std::sqrt(dx * dx + dy * dy);
    }
    double best = boundaryDist(p);
    for (size_t i = 0; i < centers_.size() && i < radii_.size(); ++i) {
        best = std::min(best, (p - centers_[i]).norm() - radii_[i]);
    }
    return best;
}

bool TaskRouteQualifier::isFree(const Vec2d& p, double clearance) const {
    return esdfAt(p) > clearance;
}

// ────────────────────────────────────────────────────────────────────
//  Straight-corridor blocker detection (2D reference semantics)
// ────────────────────────────────────────────────────────────────────
bool TaskRouteQualifier::findStraightBlocker(
    const Vec2d& start, const Vec2d& goal, int& blocker_id,
    Vec2d& blocker_center, double& blocker_radius,
    std::vector<int>& blocking_ids) const {
    const Vec2d axis = goal - start;
    const double axis_len = std::max(1e-6, axis.norm());
    // The straight corridor is blocked when an obstacle surface comes
    // within the ROUTE QUALIFICATION clearance of the segment (mirrors
    // the 2D margin = safety + route_margin + discretisation_margin).
    const double margin = cfg_.routeQualificationClearance();
    blocking_ids.clear();
    bool blocked = false;
    double best_along = std::numeric_limits<double>::infinity();
    double best_pen = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < centers_.size() && i < radii_.size(); ++i) {
        const double t = clamp(((centers_[i] - start).dot(axis)) /
                                   (axis_len * axis_len),
                               0.0, 1.0);
        const Vec2d proj = start + axis * t;
        const double d = (centers_[i] - proj).norm();
        const double pen = radii_[i] + margin - d;
        if (pen > 0.0) {
            blocked = true;
            const double along = t * axis_len;
            if (along < best_along - 1e-9 ||
                (std::fabs(along - best_along) <= 1e-9 &&
                 (pen > best_pen + 1e-9 ||
                  (std::fabs(pen - best_pen) <= 1e-9 &&
                   (blocker_id < 0 ||
                    static_cast<int>(i) < blocker_id))))) {
                best_along = along;
                best_pen = pen;
                blocker_id = static_cast<int>(i);
                blocker_center = centers_[i];
                blocker_radius = radii_[i];
            }
        }
    }
    if (blocked) {
        // All obstacles blocking the straight corridor (for diagnostics).
        for (size_t i = 0; i < centers_.size() && i < radii_.size(); ++i) {
            const double t =
                clamp(((centers_[i] - start).dot(axis)) / (axis_len * axis_len),
                      0.0, 1.0);
            const Vec2d proj = start + axis * t;
            const double d = (centers_[i] - proj).norm();
            if (radii_[i] + margin - d > 0.0) {
                blocking_ids.push_back(static_cast<int>(i));
            }
        }
    }
    return blocked;
}

// ────────────────────────────────────────────────────────────────────
//  Side-constrained A* (2D port, bounded + buffer-reusing)
// ────────────────────────────────────────────────────────────────────
std::vector<Vec2d> TaskRouteQualifier::astarPath(
    const Vec2d& a, const Vec2d& b, double clearance, const Vec2d& side_axis,
    const Vec2d& side_center, double side_bias, int side_sign,
    int max_expansions, uint32_t& expansions_out) const {
    expansions_out = 0;
    std::vector<Vec2d> empty;
    if (w_ == 0 || h_ == 0) return empty;
    const int sx = ixOf(a.x()), sy = iyOf(a.y());
    const int gx = ixOf(b.x()), gy = iyOf(b.y());
    if (!cellFreeStrict(sx, sy, clearance) ||
        !cellFreeStrict(gx, gy, clearance)) {
        return empty;
    }
    const size_t sid = idOf(sx, sy);

    // Advance the generation epoch (reset when it wraps).
    ++epoch_;
    if (epoch_ == 0) {
        std::fill(gen_.begin(), gen_.end(), 0u);
        std::fill(closed_.begin(), closed_.end(), 0u);
        epoch_ = 1;
    }
    const uint32_t epoch = epoch_;
    const double axis_len = std::max(1e-6, side_axis.norm());

    auto h = [&](int ix, int iy) {
        const double dx = (ix - gx) * res_, dy = (iy - gy) * res_;
        return std::sqrt(dx * dx + dy * dy);
    };
    auto sidePenalty = [&](int ix, int iy) -> double {
        if (side_bias <= 0.0) return 0.0;
        const Vec2d p = cellCenter(ix, iy);
        // Unified convention: cross(axis, p - side_center) > 0  ⇔  LEFT.
        const double lateral = cross2(side_axis, p - side_center) / axis_len;
        // LEFT(side_sign=+1): penalize the RIGHT side (lateral < 0);
        // RIGHT(side_sign=-1): penalize the LEFT side (lateral > 0).
        return side_bias * std::max(0.0, -static_cast<double>(side_sign) * lateral);
    };

    using Node = std::tuple<double, double, int, int>;  // f, g, ix, iy
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    gcost_[sid] = 0.0;
    parent_[sid] = -1;
    gen_[sid] = epoch;
    closed_[sid] = 0;
    open.emplace(h(sx, sy) + sidePenalty(sx, sy), 0.0, sx, sy);

    uint32_t exp = 0;
    bool found = false;
    int final_ix = -1, final_iy = -1;
    while (!open.empty() && exp < static_cast<uint32_t>(max_expansions)) {
        const auto [f, g, cx, cy] = open.top();
        open.pop();
        const size_t cid = idOf(cx, cy);
        if (closed_[cid] == epoch) continue;  // stale heap entry
        closed_[cid] = epoch;                 // expand now (settle)
        ++exp;
        if (cx == gx && cy == gy) {
            found = true;
            final_ix = cx;
            final_iy = cy;
            break;
        }
        for (int k = 0; k < 8; ++k) {
            const int nx = cx + kDj[k], ny = cy + kDi[k];
            if (!inGrid(nx, ny)) continue;
            if (!cellFreeStrict(nx, ny, clearance)) continue;
            // No diagonal corner cutting (consistent with connectivity).
            if ((k % 2 == 1) &&
                (!cellFreeStrict(cx + kDj[k], cy, clearance) ||
                 !cellFreeStrict(cx, cy + kDi[k], clearance))) {
                continue;
            }
            const size_t nid = idOf(nx, ny);
            if (closed_[nid] == epoch) continue;
            const double ng = g + kW[k] * res_ + sidePenalty(nx, ny) * res_;
            if (gen_[nid] != epoch || ng < gcost_[nid] - 1e-9) {
                gcost_[nid] = ng;
                parent_[nid] = static_cast<int>(cid);
                gen_[nid] = epoch;
                open.emplace(ng + h(nx, ny), ng, nx, ny);
            }
        }
    }
    expansions_out = exp;
    if (!found) return empty;

    std::vector<Vec2d> path;
    int ix = final_ix, iy = final_iy;
    while (ix >= 0 && iy >= 0) {
        path.push_back(cellCenter(ix, iy));
        const size_t id = idOf(ix, iy);
        const int p = parent_[id];
        if (p < 0) break;
        ix = p % w_;
        iy = p / w_;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

// ────────────────────────────────────────────────────────────────────
//  Tangent gateway + legal-cell projection (2D port)
// ────────────────────────────────────────────────────────────────────
Vec2d TaskRouteQualifier::sideTangent(const Vec2d& p, const Vec2d& center,
                                      double R, const Vec2d& axis,
                                      bool left_side) const {
    const Vec2d n(-axis.y(), axis.x());  // left normal
    const Vec2d u = p - center;
    const double d = u.norm();
    if (d <= R) {
        const Vec2d nn = left_side ? n : -n;
        return center + nn * R;
    }
    const Vec2d dir = u / d;
    const double alpha = std::acos(clamp(R / d, -1.0, 1.0));
    const Vec2d t1 = center + rot2(dir, +alpha) * R;
    const Vec2d t2 = center + rot2(dir, -alpha) * R;
    const double l1 = cross2(axis, t1 - center);
    const double l2 = cross2(axis, t2 - center);
    if (left_side) return (l1 > l2) ? t1 : t2;
    return (l1 < l2) ? t1 : t2;
}

Vec2d TaskRouteQualifier::projectToLegalCell(const Vec2d& pt,
                                             double clearance) const {
    const double eps = 0.5 * res_ * std::sqrt(2.0) + 1e-3;
    const int max_ring = std::max(
        1, static_cast<int>(std::ceil(
               cfg_.qualification.gateway_projection_radius_m /
               std::max(1e-6, res_))));
    for (int ring = 1; ring <= max_ring; ++ring) {
        for (int dy = -ring; dy <= ring; ++dy) {
            for (int dx = -ring; dx <= ring; ++dx) {
                if (std::max(std::abs(dx), std::abs(dy)) != ring) continue;
                const Vec2d c(pt.x() + dx * res_, pt.y() + dy * res_);
                if (isFree(c, clearance + eps)) return c;
            }
        }
    }
    return pt;
}

// ────────────────────────────────────────────────────────────────────
//  Path safety / shortcut (2D port)
// ────────────────────────────────────────────────────────────────────
bool TaskRouteQualifier::straightSafe(const Vec2d& a, const Vec2d& b,
                                      double clearance) const {
    const double dist = (b - a).norm();
    const int steps = std::max(2, static_cast<int>(
                                     std::ceil(dist / (0.5 * res_))));
    for (int i = 0; i <= steps; ++i) {
        const Vec2d p = a + (b - a) * (static_cast<double>(i) / steps);
        if (!isFree(p, clearance)) return false;
    }
    return true;
}

void TaskRouteQualifier::losShortcut(std::vector<Vec2d>& path,
                                     double clearance) const {
    bool improved = true;
    while (improved && path.size() > 2) {
        improved = false;
        for (size_t i = 0; i + 2 < path.size(); ++i) {
            for (size_t j = path.size() - 1; j > i + 1; --j) {
                if (straightSafe(path[i], path[j], clearance)) {
                    path.erase(path.begin() + static_cast<long>(i) + 1,
                               path.begin() + static_cast<long>(j));
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }
    }
}

bool TaskRouteQualifier::routeSafe(const std::vector<Vec2d>& path,
                                   double clearance) const {
    if (path.empty()) return false;
    for (size_t i = 1; i < path.size(); ++i) {
        const Vec2d& a = path[i - 1];
        const Vec2d& b = path[i];
        const double seg = (b - a).norm();
        const int steps = std::max(2, static_cast<int>(
                                         std::ceil(seg / (0.5 * res_))));
        for (int k = 0; k <= steps; ++k) {
            const Vec2d p = a + (b - a) * (static_cast<double>(k) / steps);
            if (!isFree(p, clearance)) return false;
        }
    }
    return true;
}

// ────────────────────────────────────────────────────────────────────
//  Homotopy verification (2D port, continuous sampling)
// ────────────────────────────────────────────────────────────────────
bool TaskRouteQualifier::passesBlockerOnSide(
    const Vec2d& side_axis, const Vec2d& side_center,
    const Vec2d& blocker_center, double blocker_radius, bool left_side,
    const std::vector<Vec2d>& path) const {
    if (path.empty()) return true;
    const double axis_len = std::max(1e-6, side_axis.norm());
    const double tol = cfg_.qualification.homotopy_side_tolerance_m;
    const double infl = blocker_radius + cfg_.routeQualificationClearance();
    const double step = std::max(1e-3, 0.5 * res_);
    double nearest_lat = 0.0;
    double nearest_d = std::numeric_limits<double>::infinity();
    for (size_t i = 1; i < path.size(); ++i) {
        const Vec2d& a = path[i - 1];
        const Vec2d& b = path[i];
        const double seg = (b - a).norm();
        const int steps = std::max(1, static_cast<int>(std::ceil(seg / step)));
        for (int k = 0; k <= steps; ++k) {
            const Vec2d p = a + (b - a) * (static_cast<double>(k) / steps);
            const double d = (p - side_center).norm();
            const double lat = cross2(side_axis, p - side_center) / axis_len;
            if (d < nearest_d) {
                nearest_d = d;
                nearest_lat = lat;
            }
            if (d < infl) {
                if (left_side && lat < -tol) return false;
                if (!left_side && lat > tol) return false;
            }
        }
    }
    const bool passes = left_side ? nearest_lat > tol : nearest_lat < -tol;
    return passes;
}

// ────────────────────────────────────────────────────────────────────
//  One side route (tangent gateways + 3-segment A* + LOS + homotopy)
// ────────────────────────────────────────────────────────────────────
void TaskRouteQualifier::planSideRoute(
    const Vec2d& start, const Vec2d& goal, const Vec2d& blocker_center,
    double blocker_radius, const Vec2d& axis_u, bool left_side,
    SideRouteResult& out) const {
    out = SideRouteResult{};
    out.checked = true;
    out.reject_reason = "";
    const double clearance = cfg_.routeQualificationClearance();
    const int side_sign = left_side ? 1 : -1;
    const double bias = cfg_.qualification.side_bias;
    const Vec2d axis = axis_u * (goal - start).norm();
    const int cap = cfg_.qualification.max_side_route_expansions;

    // The blocker is always inflated by the route clearance (the tangent
    // points must keep that much room from the blocker surface).
    const double R = blocker_radius + clearance;

    // Tangent gateways on the requested GLOBAL side (never the same
    // rotation sign for both endpoints), projected into strictly-legal
    // cells.
    const Vec2d gP = sideTangent(start, blocker_center, R, axis_u, left_side);
    const Vec2d gG = sideTangent(goal, blocker_center, R, axis_u, left_side);
    const Vec2d gP_legal = projectToLegalCell(gP, clearance);
    const Vec2d gG_legal = projectToLegalCell(gG, clearance);

    std::vector<Vec2d> path;
    uint32_t exp_total = 0;
    auto s1 = astarPath(start, gP_legal, clearance, axis, blocker_center, bias,
                        side_sign, cap, exp_total);
    uint32_t e2 = 0, e3 = 0;
    auto s2 = astarPath(gP_legal, gG_legal, clearance, axis, blocker_center,
                        bias, side_sign, cap, e2);
    auto s3 = astarPath(gG_legal, goal, clearance, axis, blocker_center, bias,
                        side_sign, cap, e3);
    exp_total += e2 + e3;
    out.expanded_nodes = exp_total;
    if (!s1.empty() && !s2.empty() && !s3.empty()) {
        // Per-segment shortcut so the forced gateways are never shortcut
        // away and the homotopy side cannot flip.
        losShortcut(s1, clearance);
        losShortcut(s2, clearance);
        losShortcut(s3, clearance);
        if (routeSafe(s1, clearance) && routeSafe(s2, clearance) &&
            routeSafe(s3, clearance)) {
            path = s1;
            path.insert(path.end(), s2.begin() + 1, s2.end());
            path.insert(path.end(), s3.begin() + 1, s3.end());
        }
    }
    if (path.empty()) {
        // Fallback: single side-constrained A* (no forced gateways).
        uint32_t e0 = 0;
        path = astarPath(start, goal, clearance, axis, blocker_center, bias,
                         side_sign, cap, e0);
        out.expanded_nodes = exp_total + e0;
        if (!path.empty()) {
            losShortcut(path, clearance);
            if (!routeSafe(path, clearance)) path.clear();
        }
    }
    if (path.empty()) {
        out.feasible = false;
        // Distinguish "search hit its node budget" from "genuinely no
        // route" so the diagnostics can tell a timeout from a dead end.
        out.reject_reason =
            (out.expanded_nodes >= static_cast<uint32_t>(cap))
                ? "side_search_budget_exceeded"
                : "side_route_not_found";
        return;
    }

    // Homotopy: the path must actually pass the blocker on the requested
    // side (continuous check against the FIXED start->goal reference).
    if (!passesBlockerOnSide(axis, blocker_center, blocker_center,
                             blocker_radius, left_side, path)) {
        out.feasible = false;
        out.reject_reason = "homotopy_side_mismatch";
        return;
    }

    // Path length + minimum route clearance along the (densified) path.
    out.path_length_m = 0.0;
    out.min_clearance_m = std::numeric_limits<double>::infinity();
    for (size_t i = 1; i < path.size(); ++i) {
        out.path_length_m += (path[i] - path[i - 1]).norm();
    }
    const double step = std::max(1e-3, 0.5 * res_);
    for (size_t i = 1; i < path.size(); ++i) {
        const Vec2d& a = path[i - 1];
        const Vec2d& b = path[i];
        const double seg = (b - a).norm();
        const int steps = std::max(1, static_cast<int>(std::ceil(seg / step)));
        for (int k = 0; k <= steps; ++k) {
            out.min_clearance_m =
                std::min(out.min_clearance_m,
                         esdfAt(a + (b - a) * (static_cast<double>(k) / steps)));
        }
    }
    if (!std::isfinite(out.min_clearance_m)) out.min_clearance_m = 0.0;
    out.feasible = true;
}

// ────────────────────────────────────────────────────────────────────
//  qualify() — the whole pipeline
// ────────────────────────────────────────────────────────────────────
void TaskRouteQualifier::qualify(const Vec2d& start, const Vec2d& goal,
                                 TaskQualificationSummary& out,
                                 QualificationCounters& counters) const {
    out = TaskQualificationSummary{};
    const auto& qcfg = cfg_.qualification;
    const double endpoint_clr = cfg_.endpointRequiredClearance();

    // ── 1. endpoint safety ─────────────────────────────────────────
    ++counters.candidates_checked;
    out.endpoint_valid =
        (start.x() >= min_.x() && start.x() <= max_.x() &&
         start.y() >= min_.y() && start.y() <= max_.y() &&
         goal.x() >= min_.x() && goal.x() <= max_.x() &&
         goal.y() >= min_.y() && goal.y() <= max_.y()) &&
        isFree(start, endpoint_clr) && isFree(goal, endpoint_clr);
    if (!out.endpoint_valid) {
        out.qualification_class = "endpoint_invalid";
        out.reject_reason = "endpoint_clearance_invalid";
        ++counters.reject_endpoint;
        return;
    }
    ++counters.endpoint_pass;

    // ── 2. connectivity (qualifier's own component map, exact) ─────
    // Built once per scene on the boundary-aware ESDF at the endpoint
    // clearance (8-conn, no diagonal corner-cutting — same as the 2D
    // ConnectivityAnalyzer).  Same main component is an exact test.
    const int c_start = componentAtPoint(start);
    const int c_goal = componentAtPoint(goal);
    out.connectivity_valid =
        c_start == main_comp_ && c_goal == main_comp_ && main_comp_ >= 0;
    if (!out.connectivity_valid) {
        out.qualification_class = "different_component";
        out.reject_reason = "start_goal_different_component";
        ++counters.reject_different_component;
        return;
    }
    ++counters.connectivity_pass;

    // ── 3. straight-corridor blocker analysis ──────────────────────
    int blocker_id = -1;
    Vec2d blocker_center(0.0, 0.0);
    double blocker_radius = 0.0;
    std::vector<int> blocking_ids;
    const bool blocked =
        findStraightBlocker(start, goal, blocker_id, blocker_center,
                            blocker_radius, blocking_ids);
    out.straight_corridor_clear = !blocked;
    if (!blocked) {
        // Direct route-clear corridor: no side search needed.
        out.qualification_class = "clear";
        out.accepted = true;
        ++counters.straight_clear;
        ++counters.accepted;
        return;
    }
    ++counters.blocked;
    out.primary_blocker_id = blocker_id;
    out.primary_blocker_x = blocker_center.x();
    out.primary_blocker_y = blocker_center.y();
    out.primary_blocker_radius = blocker_radius;
    out.blocking_obstacle_ids = std::move(blocking_ids);

    // ── 4. optional global-route confirmation (bounded A* at the ROUTE
    //    clearance — a same-component pair at the base clearance is not a
    //    guarantee of a planner-compatible route). ──────────────────
    if (qcfg.run_astar_confirmation) {
        uint32_t e = 0;
        const Vec2d axis = goal - start;
        const std::vector<Vec2d> route =
            astarPath(start, goal, cfg_.routeQualificationClearance(), axis,
                      blocker_center, 0.0, 1, qcfg.max_astar_expansions, e);
        counters.total_astar_expansions += e;
        if (route.empty()) {
            out.qualification_class = "blocked_no_global_route";
            out.reject_reason = "global_route_missing";
            ++counters.reject_global_route;
            return;
        }
    }

    // ── 5. causal LEFT / RIGHT route qualification ─────────────────
    ++counters.side_qualification_attempt;
    const Vec2d axis_u = (goal - start).normalized();
    planSideRoute(start, goal, blocker_center, blocker_radius, axis_u, true,
                  out.left);
    planSideRoute(start, goal, blocker_center, blocker_radius, axis_u, false,
                  out.right);
    counters.total_astar_expansions +=
        out.left.expanded_nodes + out.right.expanded_nodes;

    const bool left_ok = out.left.feasible;
    const bool right_ok = out.right.feasible;
    if (out.left.reject_reason.find("budget") != std::string::npos ||
        out.right.reject_reason.find("budget") != std::string::npos) {
        ++counters.reject_side_search_budget;
    }
    if (left_ok && right_ok) {
        ++counters.both_sides_feasible;
        out.qualification_class = "blocked_both_feasible";
        out.accepted = true;
    } else if (!left_ok && !right_ok) {
        out.qualification_class = "blocked_no_side";
        out.reject_reason = "no_feasible_side_route";
        ++counters.reject_left_infeasible;
        ++counters.reject_right_infeasible;
        if (!left_ok) out.left.reject_reason = out.left.reject_reason.empty()
                                                   ? "infeasible"
                                                   : out.left.reject_reason;
        if (!right_ok)
            out.right.reject_reason = out.right.reject_reason.empty()
                                          ? "infeasible"
                                          : out.right.reject_reason;
        return;
    } else {
        // Single side feasible.
        out.qualification_class = "blocked_single_side";
        ++counters.reject_left_infeasible;
        ++counters.reject_right_infeasible;
        if (!left_ok)
            out.left.reject_reason = out.left.reject_reason.empty()
                                         ? "infeasible"
                                         : out.left.reject_reason;
        if (!right_ok)
            out.right.reject_reason = out.right.reject_reason.empty()
                                          ? "infeasible"
                                          : out.right.reject_reason;
        if (qcfg.require_both_sides_feasible) {
            out.reject_reason = "require_both_sides_feasible";
            ++counters.reject_both_sides_required;
            return;
        }
        // Relaxed mode: accept a single feasible side (right preferred by
        // the deterministic runtime default).
        out.accepted = right_ok ? true : left_ok;
        out.qualification_class = "blocked_single_side_accepted";
    }

    // ── 6. privileged route stretch ────────────────────────────────
    const double straight = (goal - start).norm();
    double min_route_len = std::numeric_limits<double>::infinity();
    if (left_ok) min_route_len = std::min(min_route_len, out.left.path_length_m);
    if (right_ok)
        min_route_len = std::min(min_route_len, out.right.path_length_m);
    out.privileged_min_route_stretch =
        (std::isfinite(min_route_len) && straight > 1e-6)
            ? min_route_len / straight
            : 0.0;

    if (out.accepted) ++counters.accepted;
}

}  // namespace expert
}  // namespace il_dataset
