#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The generic, mass-free VarPro fitting core: fit the nonlinear
/// parameters of one target by Levenberg-Marquardt, eliminating the linear
/// coefficients in closed form at every trial point.
///
/// This header knows nothing about LG modes, ellipsoids, or mass matrices.
/// Everything problem-specific arrives either as an already-whitened array or
/// as the basis functor, which is exactly the point of the layering: this is
/// the piece that stays identical across targets, across choices of extra
/// basis, and across problems that are not point-spread functions at all.
///
/// **Everything crossing this boundary is hatted** -- `z_hat`, `y_hat`,
/// `e_hat`, `theta_hat` -- because whitening.hpp and ellipsoid_transform.hpp
/// have already transformed the problem into the coordinates in which it is an
/// ordinary Euclidean least-squares fit.
///
/// ## The shape of the problem
///
/// Stack the k probes so the basis at a trial theta_hat gives a design matrix
/// A(theta_hat) = Z_hat^T values(theta_hat), the theta-independent extra basis
/// a constant block B = Z_hat^T E_hat, and the fit is
///
///     min over theta_hat, c, s  of  || y_hat - A(theta_hat) c - B s ||^2 .
///
/// VarPro is three ideas, each a visible piece here:
///
///  1. **Projection.** At fixed theta_hat the optimal (c, s) is a linear
///     solve, so the outer loop only ever searches over theta_hat
///     (`inner_solve`).
///  2. **The constant block goes once, not per trial point.**
///     Frisch-Waugh-Lovell: residualize y_hat and every A against an
///     orthonormal basis of range(B) up front, fit the smooth block alone, and
///     recover s by one small back-solve at the end.
///  3. **The reduced residual's Jacobian** comes from differentiating the
///     projector -- Golub-Pereyra, with Kaufman's simplification as the
///     default, built in a single batched reverse sweep.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/detail/levenberg_marquardt.hpp"
#include "lgpsf/exceptions.hpp"

namespace lgpsf {

/// Which Jacobian of the reduced residual the outer loop is given.
enum class JacobianVariant
{
    /// One reverse sweep through the basis. The default. Gives the EXACT
    /// gradient; only the quadratic model differs from the exact Jacobian.
    Kaufman,
    /// The exact Jacobian of the reduced residual. Needs a basis evaluation
    /// that provides jac(). Used for strict testing and as a fallback.
    GolubPereyra
};

struct VarProOptions
{
    /// Ridge added to the inner least-squares problem after per-column
    /// equilibration -- guards a smooth feature the probes barely resolve.
    /// It regularizes the COEFFICIENTS only: the returned residual is the
    /// plain model misfit, and the range basis is cut by numerical rank alone.
    double ridge = 1e-8;

    JacobianVariant jacobian = JacobianVariant::Kaufman;

    LMScaling scaling = LMScaling::Jacobian;
    Eigen::VectorXd fixed_scale;

    int max_evaluations = 50;
    double ftol = 1e-8;
    double xtol = 1e-8;
    double gtol = 1e-8;
};

struct VarProResult
{
    Eigen::VectorXd theta_hat;  ///< (P,) fitted nonlinear parameters.
    Eigen::VectorXd c;          ///< (num_modes,) smooth coefficients there.
    Eigen::VectorXd s;          ///< (num_extra,) extra-basis coefficients.
    Eigen::VectorXd residual;   ///< (k,) whitened residual at the optimum.
    double cost = 0.0;          ///< 0.5 * ||residual||^2.

    bool success = false;
    LMStatus status = LMStatus::MaxEvaluations;
    int num_iterations = 0;
    int num_residual_evaluations = 0;
    std::string message;

    /// (k, P) reduced-residual Jacobian at the returned point -- a diagnostic
    /// (its singular values say how well determined the fit is), not needed
    /// for the fit itself.
    Eigen::MatrixXd jacobian;
};

namespace detail {

/// (I - Q Q^T) V: V minus its projection onto Q's orthonormal columns. Q may
/// have zero columns, in which case this is the identity.
///
/// One overload, not two: a vector binds here as a single-column matrix, and a
/// separate `Ref<const VectorXd>` version would be ambiguous with this one for
/// every vector argument.
///
/// @param Q Orthonormal columns to project out of; may have zero columns.
/// @param V What to project, one column each.
/// @return  V with its Q-component removed, same shape as @p V.
inline Eigen::MatrixXd project_out( const Eigen::MatrixXd& Q,
                                    const Eigen::Ref<const Eigen::MatrixXd>& V )
{
    if ( Q.cols() == 0 )
    {
        return V;
    }
    return V - Q * (Q.transpose() * V);
}

/// Orthonormal basis for range(B), as a (k, rank) matrix.
///
/// Via SVD with a rank cutoff rather than a plain QR, so exactly dependent
/// columns -- a caller passing the same extra function twice, say -- collapse
/// instead of contaminating the basis.
///
/// @param B The matrix whose range is wanted, (k, num_cols).
/// @return  (k, rank) with orthonormal columns; (k, 0) when @p B has none.
inline Eigen::MatrixXd orthonormal_range( const Eigen::Ref<const Eigen::MatrixXd>& B )
{
    if ( B.cols() == 0 )
    {
        return Eigen::MatrixXd::Zero(B.rows(), 0);
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU);
    const Eigen::VectorXd& sigma = svd.singularValues();
    if ( sigma.size() == 0 || sigma(0) == 0.0 )
    {
        return Eigen::MatrixXd::Zero(B.rows(), 0);
    }
    const double tol = static_cast<double>(std::max(B.rows(), B.cols()))
                       * std::numeric_limits<double>::epsilon() * sigma(0);
    Eigen::Index rank = 0;
    while ( rank < sigma.size() && sigma(rank) > tol )
    {
        ++rank;
    }
    return svd.matrixU().leftCols(rank);
}

/// Everything one trial point's linear solve produced: the coefficients and
/// residual the outer loop needs, plus the factorization pieces the Jacobian
/// reuses, so nothing is factored twice per trial point.
///
/// `sigma`/`V` are present only when the solve was asked for `InnerFactors::Full`
/// -- see `inner_solve`. Everything else is always populated.
struct InnerSolve
{
    Eigen::VectorXd c;         ///< (num_modes,)
    Eigen::VectorXd residual;  ///< (k,) y_tilde - A_tilde c
    Eigen::MatrixXd U;         ///< (k, rank) orthonormal basis of range(A_tilde)
    Eigen::VectorXd sigma;     ///< (rank,) singular values of the EQUILIBRATED matrix
    Eigen::MatrixXd V;         ///< (num_modes, rank) right singular vectors
    Eigen::VectorXd col_scale; ///< (num_modes,) column norms divided out
};

/// How much of the factorization the caller will actually use.
///
/// The distinction is worth a parameter because it is worth a lot of time: at
/// field scale with 21 modes the SVD was 61% of the whole fit, and `RangeOnly`
/// is served by a QR that measures 6-9x faster at these sizes. See
/// experiments/inner-solve-profile.md.
enum class InnerFactors
{
    /// `c`, `residual`, `U` and `col_scale`. What the reduced residual and the
    /// Kaufman Jacobian need, and all they need.
    RangeOnly,
    /// Additionally `sigma` and `V`, which only the Golub-Pereyra Jacobian uses.
    Full
};

/// The inner least-squares solve -- the "projection" in variable projection.
///
/// Columns are equilibrated first, so `ridge` acts on singular values of a
/// matrix whose scale is known rather than on the caller's units, and so the
/// rank cutoff means something.
///
/// **Two factorizations, one answer.** The ridge filter
/// `sigma / (sigma^2 + ridge)` IS Tikhonov -- it computes
/// `(A^T A + ridge I)^-1 A^T y` -- so when the caller needs no singular values
/// a column-pivoted QR gets there too: `Q`'s leading columns are an orthonormal
/// basis of `range(A~)` (the projector built from them is basis-independent, so
/// the Jacobian is unchanged), and `c` follows from a small stacked QR that
/// applies the ridge without ever forming `R^T R`. The SVD still runs when
/// `Full` is asked for, and when the QR reports RANK DEFICIENCY -- there the two
/// genuinely differ, since a pivoted QR returns a basic solution where the SVD
/// returns the minimum-norm one, and the minimum-norm one is what a rank cutoff
/// is for.
///
/// @param A_tilde The design matrix, extra block already projected out.
/// @param y_tilde The data, likewise projected.
/// @param ridge   Tikhonov damping on the coefficients; 0 disables it.
/// @param factors `RangeOnly` when only the projector and coefficients are
///                needed (the fast path); `Full` when the singular values and
///                right vectors are, as Golub-Pereyra requires.
/// @return        The coefficients, the residual, the range basis, and -- for
///                `Full` -- the singular values and right singular vectors.
inline InnerSolve inner_solve( const Eigen::MatrixXd& A_tilde,
                               const Eigen::VectorXd& y_tilde, double ridge,
                               InnerFactors factors = InnerFactors::Full )
{
    InnerSolve out;
    const Eigen::Index num_modes = A_tilde.cols();

    out.col_scale.resize(num_modes);
    for ( Eigen::Index j = 0; j < num_modes; ++j )
    {
        const double norm = A_tilde.col(j).norm();
        out.col_scale(j) = ( norm == 0.0 ) ? 1.0 : norm;
    }
    Eigen::MatrixXd equilibrated = A_tilde;
    for ( Eigen::Index j = 0; j < num_modes; ++j )
    {
        equilibrated.col(j) /= out.col_scale(j);
    }

    if ( num_modes == 0 )
    {
        out.c = Eigen::VectorXd::Zero(0);
        out.residual = y_tilde;
        out.U = Eigen::MatrixXd::Zero(y_tilde.size(), 0);
        out.sigma = Eigen::VectorXd::Zero(0);
        out.V = Eigen::MatrixXd::Zero(0, 0);
        return out;
    }

    if ( factors == InnerFactors::RangeOnly )
    {
        // Threshold chosen to mirror the SVD branch's rank cutoff: Eigen tests
        // |R(i,i)| against maxpivot * threshold(), and maxpivot plays the role
        // of sigma_max, so max(rows, cols) * eps makes the two agree in form.
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(equilibrated);
        qr.setThreshold(static_cast<double>(std::max(equilibrated.rows(), num_modes))
                        * std::numeric_limits<double>::epsilon());
        if ( qr.rank() == num_modes )
        {
            const Eigen::MatrixXd R = qr.matrixR()
                                          .topLeftCorner(num_modes, num_modes)
                                          .triangularView<Eigen::Upper>();
            out.U = qr.householderQ()
                    * Eigen::MatrixXd::Identity(equilibrated.rows(), num_modes);

            // The ridge, applied WITHOUT squaring the conditioning: solving
            // [R; sqrt(ridge) I] d = [Q^T y; 0] in least squares is the same
            // Tikhonov problem as (R^T R + ridge I) d = R^T Q^T y, and its
            // condition number is the square root of that system's.
            Eigen::MatrixXd stacked(2 * num_modes, num_modes);
            stacked.topRows(num_modes) = R;
            stacked.bottomRows(num_modes) =
                std::sqrt(std::max(ridge, 0.0))
                * Eigen::MatrixXd::Identity(num_modes, num_modes);
            Eigen::VectorXd target = Eigen::VectorXd::Zero(2 * num_modes);
            target.head(num_modes) = out.U.transpose() * y_tilde;

            const Eigen::VectorXd permuted = stacked.householderQr().solve(target);
            const Eigen::VectorXd scaled = qr.colsPermutation() * permuted;

            out.sigma = Eigen::VectorXd::Zero(0);   // not computed on this path
            out.V = Eigen::MatrixXd::Zero(num_modes, 0);
            out.c = scaled.array() / out.col_scale.array();
            out.residual = y_tilde - A_tilde * out.c;
            return out;
        }
        // rank deficient: fall through to the SVD, which is what the rank
        // cutoff and the minimum-norm solution exist for
    }

    Eigen::BDCSVD<Eigen::MatrixXd> svd(equilibrated,
                                       Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& sigma = svd.singularValues();
    const double sigma_max = ( sigma.size() > 0 ) ? sigma(0) : 0.0;
    const double tol = static_cast<double>(std::max(equilibrated.rows(), num_modes))
                       * std::numeric_limits<double>::epsilon() * sigma_max;
    Eigen::Index rank = 0;
    while ( rank < sigma.size() && sigma(rank) > tol )
    {
        ++rank;
    }

    out.U = svd.matrixU().leftCols(rank);
    out.sigma = sigma.head(rank);
    out.V = svd.matrixV().leftCols(rank);

    const Eigen::VectorXd projected = out.U.transpose() * y_tilde;
    Eigen::VectorXd filtered(rank);
    for ( Eigen::Index i = 0; i < rank; ++i )
    {
        filtered(i) = out.sigma(i) / (out.sigma(i) * out.sigma(i) + ridge)
                      * projected(i);
    }
    out.c = (out.V * filtered).array() / out.col_scale.array();
    out.residual = y_tilde - A_tilde * out.c;
    return out;
}

/// Does this basis evaluation offer the uncontracted parameter Jacobian?
template <typename T, typename = void>
struct has_jacobian : std::false_type
{
};
template <typename T>
struct has_jacobian<T, std::void_t<decltype(std::declval<T&>().jac())>>
    : std::true_type
{
};

/// The reduced (theta_hat-only) problem handed to the outer loop:
/// r(theta_hat) = y_tilde - A_tilde(theta_hat) c*(theta_hat), plus its
/// Jacobian by differentiating the projector.
///
/// Holds what is fixed across trial points -- the probes, the extra block and
/// its orthonormal range, the residualized data -- plus a ONE-ENTRY CACHE of
/// the latest point's basis evaluation and inner solve. The outer loop asks
/// for the residual and the Jacobian separately but back to back at the same
/// point, and both need the same factorization and the same evaluation.
///
/// Jacobian formulas, with A~ the residualized design matrix, P_perp the
/// projector onto range(A~)'s orthogonal complement, r the reduced residual,
/// and dA~_q the q-th parameter derivative:
///
///     Golub-Pereyra (exact):  dr_q = -P_perp (dA~_q c) - pinv(A~)^T dA~_q^T r
///     Kaufman:                dr_q = -P_perp (dA~_q c)
///
/// The Kaufman term comes from ONE reverse sweep rather than P forward ones:
/// dA_q c contracts the mode axis against the fixed vector c before anything
/// else, and sum_i c_i phi_hat_i is a single scalar-per-point function, so all
/// P components of its parameter gradient come from one batched vjp with
/// cotangent w_hat[j, i] = c_i. The (num_modes, P, K) tensor never
/// materializes and num_modes and P never meet in any product.
///
/// The Golub-Pereyra second term has no such collapse -- it keeps both the
/// mode and parameter axes alive -- so that variant needs jac().
///
/// How they relate: the dropped term is O(||r||), so the two coincide exactly
/// at a zero-residual fit; its columns lie in range(A~) while the kept term's
/// lie in the complement, so it is orthogonal to r and BOTH VARIANTS GIVE THE
/// SAME EXACT GRADIENT. Only the quadratic model differs, by a PSD O(||r||^2)
/// term: Kaufman's J^T J is the Schur complement of the full-space
/// Gauss-Newton Hessian (linearize, then eliminate), Golub-Pereyra's is the
/// Gauss-Newton Hessian of the reduced problem (eliminate, then linearize).
/// The operations do not commute, by exactly that term.
template <typename Basis>
class ReducedProblem
{
public:
    using Evaluation =
        std::decay_t<decltype(std::declval<const Basis&>()(
            std::declval<const Eigen::VectorXd&>()))>;

    ReducedProblem( Eigen::MatrixXd z_hat, Eigen::VectorXd y_hat,
                    const Eigen::Ref<const Eigen::MatrixXd>& e_hat,
                    const Basis& basis, double ridge,
                    InnerFactors factors = InnerFactors::Full )
        : z_hat_(std::move(z_hat)), y_hat_(std::move(y_hat)), basis_(basis),
          ridge_(ridge), factors_(factors)
    {
        B_ = z_hat_.transpose() * e_hat;
        Q_B_ = orthonormal_range(B_);
        y_tilde_ = project_out(Q_B_, y_hat_);
    }

    Eigen::Index num_probes() const { return z_hat_.cols(); }
    const Eigen::MatrixXd& extra_block() const { return B_; }
    const Eigen::VectorXd& data() const { return y_hat_; }

    /// A(theta_hat) before residualization -- the back-solve for s needs it.
    Eigen::MatrixXd design_matrix( const Eigen::VectorXd& theta_hat )
    {
        solve_at(theta_hat, factors_);
        return z_hat_.transpose() * evaluation_->values();
    }

    /// @param factors what this caller needs. A cache entry computed for
    ///        `RangeOnly` does not satisfy a later `Full` request, so the
    ///        point is re-solved -- which keeps a caller that mixes variants
    ///        correct rather than quietly handing it an empty `sigma`.
    const InnerSolve& solve_at( const Eigen::VectorXd& theta_hat,
                                InnerFactors factors )
    {
        if ( have_cache_ && theta_hat.size() == cached_point_.size()
             && theta_hat == cached_point_
             && !( factors == InnerFactors::Full
                   && cached_factors_ == InnerFactors::RangeOnly ) )
        {
            return cache_;
        }

        bool usable = true;
        Eigen::MatrixXd A_tilde;
        try
        {
            evaluation_.reset();
            evaluation_.emplace(basis_(theta_hat));
            A_tilde = project_out(Q_B_, Eigen::MatrixXd(z_hat_.transpose()
                                                        * evaluation_->values()));
            usable = A_tilde.allFinite();
            if ( usable )
            {
                num_modes_ = A_tilde.cols();
            }
        }
        catch ( const InfeasibleParameters& )
        {
            // NOT a failure: this point in parameter space simply has no
            // basis. Score it as "the smooth model contributes nothing" --
            // residual = y_tilde, the worst finite cost any point can produce
            // -- so the outer loop rejects the step and backs off. Only the
            // Jacobian path would be troubled by this sentinel, and the outer
            // loop evaluates the Jacobian only at accepted points, which beat
            // this cost, and at a start point validated on entry.
            evaluation_.reset();
            usable = false;
        }

        if ( usable )
        {
            cache_ = inner_solve(A_tilde, y_tilde_, ridge_, factors);
        }
        else
        {
            cache_ = InnerSolve{};
            cache_.c = Eigen::VectorXd::Zero(num_modes_);
            cache_.residual = y_tilde_;
            cache_.U = Eigen::MatrixXd::Zero(z_hat_.cols(), 0);
            cache_.sigma = Eigen::VectorXd::Zero(0);
            cache_.V = Eigen::MatrixXd::Zero(num_modes_, 0);
            cache_.col_scale = Eigen::VectorXd::Ones(num_modes_);
        }
        cached_point_ = theta_hat;
        cached_factors_ = factors;
        have_cache_ = true;
        return cache_;
    }

    Eigen::VectorXd residual( const Eigen::VectorXd& theta_hat )
    {
        return solve_at(theta_hat, factors_).residual;
    }

    Eigen::MatrixXd jacobian( const Eigen::VectorXd& theta_hat,
                              JacobianVariant variant, int num_params )
    {
        // Golub-Pereyra reads sigma and V, so it needs the full factorization
        // whatever this problem was configured for.
        const InnerSolve& solution =
            solve_at(theta_hat, variant == JacobianVariant::GolubPereyra
                                    ? InnerFactors::Full
                                    : factors_);
        if ( !evaluation_.has_value() )
        {
            return Eigen::MatrixXd::Zero(z_hat_.cols(), num_params);
        }

        // Kaufman, in one reverse sweep: cotangent w_hat[j, i] = c_i.
        const Eigen::MatrixXd cotangent =
            Eigen::VectorXd::Ones(z_hat_.rows()) * solution.c.transpose();
        const Eigen::MatrixXd G = evaluation_->vjp(cotangent);   // (K, P)
        const Eigen::MatrixXd columns = z_hat_.transpose() * G;  // (k, P)
        Eigen::MatrixXd J =
            -project_out(solution.U, project_out(Q_B_, columns));

        if ( variant == JacobianVariant::GolubPereyra )
        {
            if constexpr ( has_jacobian<Evaluation>::value )
            {
                const auto& dPhi = evaluation_->jac();
                for ( std::size_t q = 0; q < dPhi.size(); ++q )
                {
                    const Eigen::MatrixXd dA = project_out(
                        Q_B_, Eigen::MatrixXd(z_hat_.transpose() * dPhi[q]));
                    // -pinv(A~)^T dA~_q^T r, from the stored SVD pieces:
                    // pinv(A~)^T w = U diag(1/sigma) V^T (w / col_scale).
                    const Eigen::VectorXd w = dA.transpose() * solution.residual;
                    const Eigen::VectorXd scaled =
                        (w.array() / solution.col_scale.array()).matrix();
                    const Eigen::VectorXd inner =
                        ((solution.V.transpose() * scaled).array()
                         / solution.sigma.array()).matrix();
                    J.col(static_cast<Eigen::Index>(q)) -= solution.U * inner;
                }
            }
            else
            {
                throw std::invalid_argument(
                    "lgpsf::fit_varpro: the golub-pereyra Jacobian needs a basis "
                    "whose evaluation provides jac(); this one offers only "
                    "values() and vjp()");
            }
        }
        return J;
    }

    Evaluation* evaluation() { return evaluation_ ? &*evaluation_ : nullptr; }

private:
    Eigen::MatrixXd z_hat_;  ///< (K, k)
    Eigen::VectorXd y_hat_;  ///< (k,)
    const Basis& basis_;
    double ridge_;
    InnerFactors factors_;

    Eigen::MatrixXd B_;        ///< (k, num_extra)
    Eigen::MatrixXd Q_B_;      ///< (k, rank)
    Eigen::VectorXd y_tilde_;  ///< (k,)

    std::optional<Evaluation> evaluation_;
    Eigen::VectorXd cached_point_;
    InnerSolve cache_;
    bool have_cache_ = false;
    InnerFactors cached_factors_ = InnerFactors::Full;
    Eigen::Index num_modes_ = 0;
};

} // end namespace detail

/// Fit the nonlinear parameters theta_hat, and the linear coefficients (c, s),
/// for one target.
///
/// @param z_hat  (K, k) whitened probe fields over the target's support batch.
/// @param y_hat  (k,) the target's whitened response to each probe.
/// @param basis  theta_hat -> an evaluation offering `values()` -> (K,
///               num_modes), `vjp(w_hat)` -> (K, P), and optionally `jac()` ->
///               P of (K, num_modes). ONE callable rather than a triple,
///               because the values and the derivative are needed at the SAME
///               point with the inner solve in between -- the Kaufman
///               cotangent is built from the values -- so they cannot be fused
///               into one value-and-gradient call, but they can and should
///               share one evaluation. Golub-Pereyra support is a CAPABILITY
///               of the evaluation, detected at compile time.
/// @param theta_hat_init  (P,) starting point. Domain-specific, so producing
///               it is the caller's job.
/// @param e_hat  (K, num_extra) whitened theta-independent basis, e.g. the
///               diagonal spike. Pass a (K, 0) matrix for none.
/// @param options   tolerances, the coefficient ridge, and which Jacobian
///               variant to use.
/// @param callback  optional; receives (theta_hat, c, residual) at the start
///               point and at each accepted step. Monitoring only -- it cannot
///               stop the fit.
/// @return       the fitted parameters and coefficients, the residual, the
///               cost, and whether the loop converged.
/// @throws std::invalid_argument if the shapes disagree, or if
///         `JacobianVariant::GolubPereyra` is requested of a basis whose
///         evaluation offers no `jac()`.
///
/// Deliberately out of scope, being a caller's decision rather than VarPro's:
/// whether to prefer this fit over some baseline (that needs held-out data
/// this function never sees), and post-hoc bound checking on the result.
template <typename Basis>
VarProResult fit_varpro(
    const Eigen::Ref<const Eigen::MatrixXd>& z_hat,
    const Eigen::Ref<const Eigen::VectorXd>& y_hat, const Basis& basis,
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat_init,
    const Eigen::Ref<const Eigen::MatrixXd>& e_hat,
    const VarProOptions& options = VarProOptions(),
    const std::function<void( const Eigen::VectorXd&, const Eigen::VectorXd&,
                              const Eigen::VectorXd& )>& callback = {} )
{
    const Eigen::Index num_points = z_hat.rows();
    const Eigen::Index num_probes = z_hat.cols();
    const Eigen::Index num_params = theta_hat_init.size();

    if ( y_hat.size() != num_probes )
    {
        throw std::invalid_argument(
            "lgpsf::fit_varpro: y_hat has " + std::to_string(y_hat.size())
            + " entries but z_hat has " + std::to_string(num_probes) + " probes");
    }
    if ( e_hat.rows() != num_points )
    {
        throw std::invalid_argument(
            "lgpsf::fit_varpro: e_hat has " + std::to_string(e_hat.rows())
            + " rows but z_hat has " + std::to_string(num_points) + " points");
    }
    if ( num_probes < num_params )
    {
        throw std::invalid_argument(
            "lgpsf::fit_varpro: Levenberg-Marquardt needs at least as many "
            "probes as parameters; got " + std::to_string(num_probes)
            + " probes < " + std::to_string(num_params) + " parameters");
    }

    // Kaufman reads only the range basis, so its solves take the QR path; the
    // exact Jacobian needs sigma and V and pays for the SVD.
    detail::ReducedProblem<Basis> reduced(
        z_hat, y_hat, e_hat, basis, options.ridge,
        options.jacobian == JacobianVariant::GolubPereyra
            ? detail::InnerFactors::Full
            : detail::InnerFactors::RangeOnly);

    // Validate the start point before the loop, so a caller's bad scaling is a
    // loud error rather than a fit that silently never moves.
    {
        bool startable = true;
        try
        {
            auto at = basis(Eigen::VectorXd(theta_hat_init));
            startable = at.values().allFinite();
            if ( options.jacobian == JacobianVariant::GolubPereyra
                 && !detail::has_jacobian<decltype(at)>::value )
            {
                throw std::invalid_argument(
                    "lgpsf::fit_varpro: the golub-pereyra Jacobian needs a basis "
                    "whose evaluation provides jac()");
            }
        }
        catch ( const InfeasibleParameters& )
        {
            startable = false;
        }
        if ( !startable )
        {
            throw std::invalid_argument(
                "lgpsf::fit_varpro: the basis is not evaluable at theta_hat_init "
                "-- check its scaling (a log-Cholesky diagonal far outside the "
                "geometry of the point batch is the usual cause)");
        }
    }

    const auto residual_fn = [&]( const Eigen::VectorXd& point ) {
        return reduced.residual(point);
    };
    // The Jacobian is evaluated exactly once per outer iteration -- at the
    // start point and after each accepted step -- so the callback traces the
    // iteration path directly, with no de-duplication needed.
    //
    // One call still has to be added at the end, though: the loop can satisfy
    // its convergence test IMMEDIATELY after accepting a step, and then it
    // returns without ever evaluating the Jacobian at that final point. See
    // below.
    Eigen::VectorXd last_callback_point;
    const auto jacobian_fn = [&]( const Eigen::VectorXd& point ) {
        Eigen::MatrixXd J = reduced.jacobian(point, options.jacobian,
                                             static_cast<int>(num_params));
        if ( callback )
        {
            const detail::InnerSolve& solution =
                reduced.solve_at(point, detail::InnerFactors::RangeOnly);
            callback(point, solution.c, solution.residual);
            last_callback_point = point;
        }
        return J;
    };

    LevenbergMarquardtOptions lm_options;
    lm_options.ftol = options.ftol;
    lm_options.xtol = options.xtol;
    lm_options.gtol = options.gtol;
    lm_options.max_evaluations = options.max_evaluations;
    lm_options.scaling = options.scaling;
    lm_options.fixed_scale = options.fixed_scale;

    const LevenbergMarquardtResult lm =
        levenberg_marquardt(residual_fn, jacobian_fn, theta_hat_init, lm_options);

    VarProResult result;
    result.theta_hat = lm.parameters;
    const detail::InnerSolve& final_solve =
        reduced.solve_at(result.theta_hat, detail::InnerFactors::RangeOnly);
    result.c = final_solve.c;

    // Close the trace on the returned point, so the callback always ends where
    // the fit did. Needed whenever the loop converged right after accepting a
    // step, since no Jacobian -- and hence no callback -- happened there.
    if ( callback
         && (last_callback_point.size() != result.theta_hat.size()
             || last_callback_point != result.theta_hat) )
    {
        callback(result.theta_hat, final_solve.c, final_solve.residual);
    }

    // Frisch-Waugh-Lovell back-solve: given (theta_hat, c), the optimal extra
    // coefficients solve the small problem B s ~ y_hat - A c, and the joint
    // residual that results equals the reduced residual exactly.
    const Eigen::MatrixXd A = reduced.design_matrix(result.theta_hat);
    const Eigen::VectorXd target = y_hat - A * result.c;
    if ( reduced.extra_block().cols() == 0 )
    {
        result.s = Eigen::VectorXd::Zero(0);
    }
    else
    {
        result.s = reduced.extra_block()
                       .bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                       .solve(target);
    }
    result.residual = target - reduced.extra_block() * result.s;
    result.cost = 0.5 * result.residual.squaredNorm();

    result.success = lm.success;
    result.status = lm.status;
    result.num_iterations = lm.num_iterations;
    result.num_residual_evaluations = lm.num_residual_evaluations;
    result.message = lm.message;
    result.jacobian = lm.jacobian;
    return result;
}

/// Overload for a fit with no extra basis at all.
template <typename Basis>
VarProResult fit_varpro(
    const Eigen::Ref<const Eigen::MatrixXd>& z_hat,
    const Eigen::Ref<const Eigen::VectorXd>& y_hat, const Basis& basis,
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat_init,
    const VarProOptions& options = VarProOptions(),
    const std::function<void( const Eigen::VectorXd&, const Eigen::VectorXd&,
                              const Eigen::VectorXd& )>& callback = {} )
{
    return fit_varpro(z_hat, y_hat, basis, theta_hat_init,
                      Eigen::MatrixXd::Zero(z_hat.rows(), 0), options, callback);
}

} // end namespace lgpsf
