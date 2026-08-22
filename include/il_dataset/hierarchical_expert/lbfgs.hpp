#pragma once
/// @file   lbfgs.hpp
/// @brief  Compact Limited-memory BFGS (L-BFGS) unconstrained minimizer.
///
/// STRUCTURE COPIED FROM THE OPEN-SOURCE LIBRARY:
///   * Naoaki Okazaki, "libLBFGS: Library of Limited-memory
///     Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)", https://github.com/chokkan/liblbfgs
///     (MIT license) — the two-loop recursion, H0 scaling and the
///     sufficient-decrease line search follow its structure.
///   * The same library is the solver bundled by EGO-Planner
///     (ZJU-FAST-Lab/ego-planner, src/planner/bspline_opt/lbfgs.hpp) for its
///     B-spline optimisation.
/// This is a self-contained C++17 re-implementation over Eigen vectors
/// (structure copied, code written for this package).  It minimizes a
/// smooth differentiable objective with an analytic gradient:
///     min_x  f(x),   g(x) = df/dx supplied by the caller.
///
/// L-BFGS keeps the last `mem` (s, y) pairs and builds the search
/// direction with the two-loop recursion + a scalar H0 = s^T y / y^T y,
/// then performs a backtracking Armijo line search.

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace il_dataset {
namespace expert {

struct LbfgsParams {
    /// Number of (s, y) pairs kept in the history (the "m" of L-BFGS).
    int mem = 6;
    /// Max outer iterations.
    int max_iterations = 100;
    /// Converge when the gradient norm drops below this.
    double g_epsilon = 1e-4;
    /// Converge when the relative cost change is below this.
    double rel_cost_tol = 1e-10;
    /// Armijo sufficient-decrease factor (c1 in f <= f0 + c1*alpha*g·d).
    double ftol = 1e-4;
    /// Minimum line-search step (give up if no decrease found above it).
    double min_step = 1e-12;
    /// Step reduction factor for the backtracking line search.
    double step_scale = 0.5;
};

struct LbfgsResult {
    bool converged = false;
    int iterations = 0;
    int evaluations = 0;
    double cost = 0.0;
};

/// Minimize `obj` starting from `x` (in-place).  `obj(x, grad)` must fill
/// `grad` with the gradient at `x` and return the cost.  Returns the final
/// cost in `result.cost`.
inline LbfgsResult lbfgsMinimize(
    Eigen::VectorXd& x,
    const std::function<double(const Eigen::VectorXd&, Eigen::VectorXd&)>& obj,
    const LbfgsParams& params = LbfgsParams{}) {
    LbfgsResult result;
    const int n = static_cast<int>(x.size());
    if (n <= 0) {
        result.converged = true;
        return result;
    }
    const int mem = std::max(1, params.mem);
    std::vector<Eigen::VectorXd> s_hist, y_hist;
    s_hist.reserve(mem);
    y_hist.reserve(mem);

    Eigen::VectorXd g(n), g_new(n), d(n), x_new(n);
    double f = obj(x, g);
    result.evaluations = 1;
    result.cost = f;
    if (g.norm() < params.g_epsilon) {
        result.converged = true;
        return result;
    }

    for (int iter = 0; iter < params.max_iterations; ++iter) {
        result.iterations = iter + 1;

        // ── Two-loop recursion (Nocedal & Wright Algo 7.4) ─────────
        Eigen::VectorXd q = g;
        const int k = static_cast<int>(s_hist.size());
        std::vector<double> alpha(k);
        for (int i = k - 1; i >= 0; --i) {
            const double rho =
                1.0 / std::max(1e-12, s_hist[i].dot(y_hist[i]));
            alpha[i] = rho * s_hist[i].dot(q);
            q -= alpha[i] * y_hist[i];
        }
        // H0 = gamma * I with gamma = s_k^T y_k / y_k^T y_k.
        double gamma = 1.0;
        if (k > 0) {
            const double sy = s_hist.back().dot(y_hist.back());
            const double yy = y_hist.back().squaredNorm();
            if (yy > 1e-12) gamma = sy / yy;
        }
        Eigen::VectorXd r = gamma * q;
        for (int i = 0; i < k; ++i) {
            const double rho =
                1.0 / std::max(1e-12, s_hist[i].dot(y_hist[i]));
            const double beta = rho * y_hist[i].dot(r);
            r += s_hist[i] * (alpha[i] - beta);
        }
        d = -r;
        const double dg = d.dot(g);
        if (dg >= 0.0) d = -g;  // not a descent direction → steepest descent

        // ── Backtracking Armijo line search ────────────────────────
        double step = 1.0;
        bool found = false;
        double f_new = f;
        while (step > params.min_step) {
            x_new = x + step * d;
            f_new = obj(x_new, g_new);
            ++result.evaluations;
            if (f_new <= f + params.ftol * step * dg) {
                found = true;
                break;
            }
            step *= params.step_scale;
        }
        if (!found) break;  // no sufficient decrease → stop

        // ── History update (FIFO of size mem) ──────────────────────
        s_hist.push_back(x_new - x);
        y_hist.push_back(g_new - g);
        if (static_cast<int>(s_hist.size()) > mem) {
            s_hist.erase(s_hist.begin());
            y_hist.erase(y_hist.begin());
        }

        x = x_new;
        g = g_new;
        result.cost = f_new;

        const double rel_change =
            std::fabs(f_new - f) / std::max(1.0, std::fabs(f));
        f = f_new;
        if (g.norm() < params.g_epsilon || rel_change < params.rel_cost_tol) {
            result.converged = true;
            break;
        }
    }
    return result;
}

}  // namespace expert
}  // namespace il_dataset
