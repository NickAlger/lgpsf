#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The two deflation constructions: free-residual Rayleigh--Ritz
/// (zero extra operator access) and the value pass (spend m true applies on
/// eigenvalue ESTIMATES, where estimation --- not directions --- is the
/// limiting factor).
///
/// Both target the error H_d - (B + E) of the corrected operator, measured
/// in the build metric M0 = B + E + a0 H_r (the whitened error
/// D = M0^{-1/2} (H_d - B - E) M0^{-1/2}). Both start from the same free
/// ingredient: the archive residuals R = Y - (B + E) Z are exact samples of
/// the error action, already paid for when the fit was probed. Their
/// M0-orthonormalized span is the RESIDUAL BASIS, and on the Pine Island
/// Glacier study it captures the certified error directions to principal
/// cosines of ~0.99 --- the directions are nearly free; the VALUES are
/// where the two constructions differ:
///
/// - `deflate_free` estimates the values from the same residuals by a
///   regularized Rayleigh--Ritz quotient (a truncated pseudoinverse; the
///   truncation is essential --- the raw pseudoinverse produced values of
///   27.7 against a certified extreme of 3.10 and lost definiteness).
///   Helped at large probe counts (34 -> 22 iterations at the best
///   setting), and helped not at all below k ~ 20. Ship-with caveats.
/// - `value_pass` spends m TRUE applications of H_d on the top of the
///   residual basis and forms the exact Rayleigh matrix of the whitened
///   error there. On PIG this was decisive: a k=100 fit plus a 50-apply
///   value pass reached 5/12/18/24 iterations versus 12/26/36/45
///   undeflated. The new pairs (Q, H_d Q) enter the archive as secant
///   information, reusable by any later rebuild.
///
/// Both require a certified struct (run `make_pd` first): the inner
/// M0-solves use the two-level path at a0, which the certificate
/// guarantees. Estimated eigenvalues are clamped below at
/// `A.clamp_floor` (default -0.95): a whitened-error eigenvalue at -1
/// would make the corrected operator singular at a0, so the clamp is what
/// keeps the a >= a0 zone guaranteed after deflation.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/corrections/solve.hpp"

namespace lgpsf::corrections {

/// Knobs shared by both constructions.
struct DeflateOptions
{
    double rcond = 3e-2;      ///< free mode: pseudoinverse truncation. The
                              ///< PIG sweep was U-shaped in this, optimum
                              ///< drifting toward stronger truncation as the
                              ///< probe count shrinks; 3e-2 was best at
                              ///< k = 200.
    int rank = -1;            ///< keep the top-r estimated modes by |theta|;
                              ///< -1 keeps all
    double solve_rtol = 1e-8; ///< inner two-level tolerance for M0^{-1}
    double oracle_tol = 1e-10;
    int max_iters = 500;      ///< inner PCG cap
    double gram_drop_tol = 1e-12;  ///< residual-basis Gram drop, relative
};

/// What a deflation pass did.
struct DeflateReport
{
    int residuals = 0;   ///< archive pairs consumed
    int basis = 0;       ///< M0-orthonormal residual-basis size
    int applies = 0;     ///< TRUE H_d applications spent (0 for free mode)
    int kept = 0;        ///< estimated modes merged into the block
    int clamped = 0;     ///< eigenvalue estimates clamped at clamp_floor
    int added = 0;       ///< block columns actually appended
    double theta_min = std::numeric_limits<double>::quiet_NaN();
    double theta_max = std::numeric_limits<double>::quiet_NaN();
};

/// V1: exact Rayleigh matrix on the top of the residual basis (m applies).
/// V2: one power step --- half the budget refines the basis through the
/// true error before the Rayleigh matrix is formed; keeps improving where
/// V1 saturates (V1 cannot see outside the residual span).
enum class ValuePassMode
{
    V1,
    V2
};

namespace detail {

/// The M0-orthonormal residual basis, ordered by decreasing whitened
/// residual energy, plus the pieces both constructions reuse.
struct ResidualBasis
{
    Eigen::MatrixXd R;      ///< (N, k) residuals Y - (B + E) Z
    Eigen::MatrixXd Qt;     ///< (N, r) M0-orthonormal basis of M0^{-1} R
    Eigen::VectorXd energy; ///< kept Gram eigenvalues, descending
};

inline ResidualBasis residual_basis( const ShiftedOperator& A,
                                     const DeflateOptions& opts )
{
    if ( !A.lambda_floor.has_value() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections deflation: the struct is uncertified -- run "
            "make_pd first (the inner M0-solves need the a0 guarantee)");
    }
    if ( A.archive.Z.size() == 0 || A.archive.Y.size() == 0 )
    {
        throw std::invalid_argument(
            "lgpsf::corrections deflation: the probe archive is empty -- "
            "deflation is built from the archived pairs (Z, Y = Hd Z)");
    }

    ResidualBasis basis;
    basis.R = A.archive.Y - apply(A, A.archive.Z, 0.0);

    SolveOpts inner;
    inner.mode = SolveMode::TwoLevel;
    inner.rtol = opts.solve_rtol;
    inner.oracle_tol = opts.oracle_tol;
    inner.max_iters = opts.max_iters;
    const Eigen::MatrixXd X = solve(A, basis.R, A.a0, inner).X;

    // Gram of the whitened residuals: X^T M0 X = X^T R up to the solve
    // tolerance; symmetrized to keep the eigensolver honest
    Eigen::MatrixXd G = X.transpose() * basis.R;
    G = 0.5 * (G + G.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(G);
    const double scale = eigen.eigenvalues().cwiseAbs().maxCoeff();
    std::vector<Eigen::Index> kept;
    for ( Eigen::Index i = eigen.eigenvalues().size() - 1; i >= 0; --i )
    {
        if ( eigen.eigenvalues()(i) > opts.gram_drop_tol * scale )
        {
            kept.push_back(i);  // descending order
        }
    }
    basis.Qt.resize(A.dim(), static_cast<Eigen::Index>(kept.size()));
    basis.energy.resize(static_cast<Eigen::Index>(kept.size()));
    for ( std::size_t j = 0; j < kept.size(); ++j )
    {
        const Eigen::Index at = static_cast<Eigen::Index>(j);
        basis.Qt.col(at) = X * eigen.eigenvectors().col(kept[j])
                           / std::sqrt(eigen.eigenvalues()(kept[j]));
        basis.energy(at) = eigen.eigenvalues()(kept[j]);
    }
    return basis;
}

/// Eigendecompose a symmetric estimate T on an M0-orthonormal basis Qb,
/// keep the top-r by |theta|, clamp, and merge the correction
/// (M0 Qb) W Theta W^T (M0 Qb)^T into the block. Shared tail of both
/// constructions.
inline void keep_clamp_merge( ShiftedOperator& A, const Eigen::MatrixXd& Qb,
                              const Eigen::MatrixXd& T, Provenance tag,
                              const DeflateOptions& opts,
                              DeflateReport& report )
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
        0.5 * (T + T.transpose()));
    std::vector<Eigen::Index> order(
        static_cast<std::size_t>(eigen.eigenvalues().size()));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&]( Eigen::Index a, Eigen::Index b ) {
                  return std::abs(eigen.eigenvalues()(a))
                         > std::abs(eigen.eigenvalues()(b));
              });
    Eigen::Index keep = static_cast<Eigen::Index>(order.size());
    if ( opts.rank >= 0 )
    {
        keep = std::min<Eigen::Index>(keep, opts.rank);
    }
    if ( keep == 0 )
    {
        return;
    }

    Eigen::MatrixXd W(eigen.eigenvectors().rows(), keep);
    Eigen::MatrixXd Theta = Eigen::MatrixXd::Zero(keep, keep);
    for ( Eigen::Index j = 0; j < keep; ++j )
    {
        W.col(j) = eigen.eigenvectors().col(order[static_cast<std::size_t>(j)]);
        double theta = eigen.eigenvalues()(order[static_cast<std::size_t>(j)]);
        if ( theta < A.clamp_floor )
        {
            theta = A.clamp_floor;
            ++report.clamped;
        }
        Theta(j, j) = theta;
        report.theta_min = std::isnan(report.theta_min)
                               ? theta
                               : std::min(report.theta_min, theta);
        report.theta_max = std::isnan(report.theta_max)
                               ? theta
                               : std::max(report.theta_max, theta);
    }
    report.kept = static_cast<int>(keep);

    // the correction lives on the image directions M0 Qb W; the block wants
    // V with Hr V = image, i.e. one oracle solve per kept mode
    const Eigen::MatrixXd images = apply(A, Qb, A.a0) * W;
    const Eigen::MatrixXd V_new = A.hr.solve(images, opts.oracle_tol);
    const MergeReport merged =
        merge(A.block, A.hr, V_new, Theta, Theta, tag);
    report.added = merged.added;
}

} // end namespace detail

/// Free-residual Rayleigh--Ritz deflation: zero extra operator access. The
/// value estimate on the residual basis is
///   T = sym( (Qt^T R) (Qt^T M0 Z)^+_rcond ),
/// the regularized quotient of "what the error did" by "where it was
/// probed"; the rcond truncation is what keeps interpolatory noise out of
/// the eigenvalue estimates. Caveats from the PIG study ship with this
/// function: it helped at large probe counts and not at all at k <= 20 ---
/// if fresh operator applies are affordable, `value_pass` dominates.
inline DeflateReport deflate_free( ShiftedOperator& A,
                                   DeflateOptions opts = {} )
{
    if ( !(opts.rcond > 0.0 && opts.rcond < 1.0) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::deflate_free: rcond must lie in (0, 1)");
    }
    const detail::ResidualBasis basis = detail::residual_basis(A, opts);
    DeflateReport report;
    report.residuals = static_cast<int>(A.archive.Z.cols());
    report.basis = static_cast<int>(basis.Qt.cols());
    if ( basis.Qt.cols() == 0 )
    {
        return report;
    }

    const Eigen::MatrixXd QtR = basis.Qt.transpose() * basis.R;
    const Eigen::MatrixXd QtM0Z =
        basis.Qt.transpose() * apply(A, A.archive.Z, A.a0);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        QtM0Z, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double cutoff = opts.rcond * svd.singularValues()(0);
    Eigen::VectorXd inverted =
        Eigen::VectorXd::Zero(svd.singularValues().size());
    for ( Eigen::Index i = 0; i < svd.singularValues().size(); ++i )
    {
        if ( svd.singularValues()(i) > cutoff )
        {
            inverted(i) = 1.0 / svd.singularValues()(i);
        }
    }
    const Eigen::MatrixXd pinv = svd.matrixV() * inverted.asDiagonal()
                                 * svd.matrixU().transpose();
    const Eigen::MatrixXd T = QtR * pinv;

    detail::keep_clamp_merge(A, basis.Qt, T, Provenance::Deflation, opts,
                             report);
    return report;
}

/// The value pass: spend `m` true applications of H_d on eigenvalue
/// ESTIMATES. The basis is free (the top of the residual basis); the
/// applies buy the exact Rayleigh matrix of the whitened error there,
///   T = Qb^T (H_d - B - E) Qb,
/// which is what free deflation could only estimate. V1 spends all m on
/// the residual basis and saturates when m reaches its dimension; V2
/// spends half refining the basis through the true error first (one power
/// step) and keeps improving past that point. The new pairs (Q, H_d Q)
/// are appended to the archive --- secant information any later rebuild
/// can reuse without touching H_d again.
inline DeflateReport value_pass( ShiftedOperator& A, const SymmetricOp& Hd,
                                 int m, ValuePassMode mode = ValuePassMode::V1,
                                 DeflateOptions opts = {} )
{
    if ( m <= 0 )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::value_pass: m must be positive");
    }
    if ( Hd.dim() != A.dim() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::value_pass: Hd dim "
            + std::to_string(Hd.dim()) + " != struct dim "
            + std::to_string(A.dim()));
    }
    const detail::ResidualBasis basis = detail::residual_basis(A, opts);
    DeflateReport report;
    report.residuals = static_cast<int>(A.archive.Z.cols());
    report.basis = static_cast<int>(basis.Qt.cols());
    if ( basis.Qt.cols() == 0 )
    {
        return report;
    }

    const Eigen::Index budget1 =
        ( mode == ValuePassMode::V1 )
            ? std::min<Eigen::Index>(m, basis.Qt.cols())
            : std::min<Eigen::Index>(std::max(1, m / 2), basis.Qt.cols());
    Eigen::MatrixXd Qb = basis.Qt.leftCols(budget1);
    Eigen::MatrixXd HdQ = Hd.apply(Qb);

    if ( mode == ValuePassMode::V2 )
    {
        // one power step: push the basis through the true whitened error
        // and M0-orthonormalize the new directions against it
        SolveOpts inner;
        inner.mode = SolveMode::TwoLevel;
        inner.rtol = opts.solve_rtol;
        inner.oracle_tol = opts.oracle_tol;
        inner.max_iters = opts.max_iters;
        const Eigen::MatrixXd error_image = HdQ - apply(A, Qb, 0.0);
        const Eigen::MatrixXd X2 = solve(A, error_image, A.a0, inner).X;

        const Eigen::MatrixXd M0X2 = apply(A, X2, A.a0);
        const Eigen::MatrixXd M0Q1 = apply(A, Qb, A.a0);
        const Eigen::MatrixXd cross = Qb.transpose() * M0X2;
        Eigen::MatrixXd W2 = X2 - Qb * cross;
        Eigen::MatrixXd M0W2 = M0X2 - M0Q1 * cross;
        Eigen::MatrixXd G2 = W2.transpose() * M0W2;
        G2 = 0.5 * (G2 + G2.transpose()).eval();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(G2);
        const double scale =
            std::max(eigen.eigenvalues().cwiseAbs().maxCoeff(), 1e-300);
        std::vector<Eigen::Index> fresh;
        for ( Eigen::Index i = eigen.eigenvalues().size() - 1; i >= 0; --i )
        {
            if ( eigen.eigenvalues()(i) > opts.gram_drop_tol * scale )
            {
                fresh.push_back(i);
            }
        }
        const Eigen::Index budget2 = std::min<Eigen::Index>(
            static_cast<Eigen::Index>(fresh.size()), m - budget1);
        if ( budget2 > 0 )
        {
            Eigen::MatrixXd Q2(A.dim(), budget2);
            for ( Eigen::Index j = 0; j < budget2; ++j )
            {
                Q2.col(j) =
                    W2 * eigen.eigenvectors().col(fresh[static_cast<std::size_t>(j)])
                    / std::sqrt(
                          eigen.eigenvalues()(fresh[static_cast<std::size_t>(j)]));
            }
            const Eigen::MatrixXd HdQ2 = Hd.apply(Q2);
            Eigen::MatrixXd Qb_ext(A.dim(), Qb.cols() + budget2);
            Qb_ext << Qb, Q2;
            Eigen::MatrixXd HdQ_ext(A.dim(), Qb.cols() + budget2);
            HdQ_ext << HdQ, HdQ2;
            Qb = std::move(Qb_ext);
            HdQ = std::move(HdQ_ext);
        }
    }
    report.applies = static_cast<int>(Qb.cols());

    // the exact Rayleigh matrix of the whitened error on the basis
    const Eigen::MatrixXd T =
        Qb.transpose() * (HdQ - apply(A, Qb, 0.0));
    detail::keep_clamp_merge(A, Qb, T, Provenance::ValuePass, opts, report);

    // archive growth: these pairs are the only new trace of H_d, and they
    // outlive this call
    const Eigen::Index old = A.archive.Q_vp.cols();
    Eigen::MatrixXd Q_all(A.dim(), old + Qb.cols());
    Eigen::MatrixXd HdQ_all(A.dim(), old + Qb.cols());
    if ( old > 0 )
    {
        Q_all.leftCols(old) = A.archive.Q_vp;
        HdQ_all.leftCols(old) = A.archive.HdQ_vp;
    }
    Q_all.rightCols(Qb.cols()) = Qb;
    HdQ_all.rightCols(Qb.cols()) = HdQ;
    A.archive.Q_vp = std::move(Q_all);
    A.archive.HdQ_vp = std::move(HdQ_all);
    return report;
}

/// Fold the ARCHIVED value-pass pairs (Q, H_d Q) into the block at the
/// current shift and corrections — zero new H_d applies. The archived
/// basis was M0-orthonormal in the metric it was bought in; here it is
/// re-orthonormalized in the CURRENT metric by linear combination, and
/// H_d of the combined columns follows exactly from the stored images
/// (linearity — this is what makes the pairs secant information rather
/// than a stale cache). Used by `rebuild_at`; callable on its own.
inline DeflateReport fold_value_pairs( ShiftedOperator& A,
                                       DeflateOptions opts = {} )
{
    if ( !A.lambda_floor.has_value() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::fold_value_pairs: the struct is "
            "uncertified -- run make_pd first");
    }
    if ( A.archive.Q_vp.size() == 0 )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::fold_value_pairs: no archived value-pass "
            "pairs");
    }
    DeflateReport report;
    report.residuals = static_cast<int>(A.archive.Q_vp.cols());

    const Eigen::MatrixXd& Q = A.archive.Q_vp;
    const Eigen::MatrixXd& HdQ = A.archive.HdQ_vp;
    Eigen::MatrixXd G = Q.transpose() * apply(A, Q, A.a0);
    G = 0.5 * (G + G.transpose()).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(G);
    const double scale = eigen.eigenvalues().cwiseAbs().maxCoeff();
    std::vector<Eigen::Index> kept;
    for ( Eigen::Index i = eigen.eigenvalues().size() - 1; i >= 0; --i )
    {
        if ( eigen.eigenvalues()(i) > opts.gram_drop_tol * scale )
        {
            kept.push_back(i);
        }
    }
    if ( kept.empty() )
    {
        return report;
    }
    Eigen::MatrixXd S(Q.cols(), static_cast<Eigen::Index>(kept.size()));
    for ( std::size_t j = 0; j < kept.size(); ++j )
    {
        S.col(static_cast<Eigen::Index>(j)) =
            eigen.eigenvectors().col(kept[j])
            / std::sqrt(eigen.eigenvalues()(kept[j]));
    }
    const Eigen::MatrixXd Qb = Q * S;      // M0-orthonormal, current metric
    const Eigen::MatrixXd HdQb = HdQ * S;  // exact, by linearity
    report.basis = static_cast<int>(Qb.cols());
    report.applies = 0;  // the whole point

    const Eigen::MatrixXd T = Qb.transpose() * (HdQb - apply(A, Qb, 0.0));
    detail::keep_clamp_merge(A, Qb, T, Provenance::ValuePass, opts, report);
    return report;
}

} // end namespace lgpsf::corrections
