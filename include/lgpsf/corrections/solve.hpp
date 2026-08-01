#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Varying-shift semantics (the three zones) and the solve paths.
///
/// The struct carries the BUILD shift a0; every solve takes its own `a`.
/// Three zones, checked in this order:
///
///   a <= -lambda_floor          REFUSED   the sparse part is indefinite
///                                         there; rebuild_at is the remedy
///   -lambda_floor < a < a0      WARNED    the flip certificate still holds
///                                         analytically, but the clamp
///                                         guarantee was set in the a0
///                                         metric — certify at runtime
///   a >= a0                     GUARANTEED  the build contract (flip
///                                         certificate + clamp discipline)
///
/// The runtime certificate in the warning zone is analytic and exact about
/// what it bounds: make_pd certified  B + E_cert >= lambda_floor * H_r  (in
/// the pencil sense), and everything added since is
/// D = C_corr - C_corr_certified on the block's H_r-orthonormal columns, so
///
///   P(a) = B + E + a H_r  is PD  if  a + lambda_floor + min(0, eig_min(D)) > 0.
///
/// This is sufficient, not necessary — corrections built with the clamp
/// discipline are PD at a0 by a different (M0-metric) argument that does not
/// transfer below a0, which is exactly why this zone warns.
///
/// Two solve modes:
///
/// - GLR (default): the deployment object IS M(a) = a H_r + S, applied by
///   the closed-form Woodbury inverse — one oracle solve per application,
///   re-shiftable for free. This is the architecture validated end-to-end
///   at PIG scale as a preconditioner for the consumer's Krylov iteration.
/// - Two-level: solve P(a) = B + E + a H_r itself by an inner PCG
///   preconditioned by M(a)^{-1}. This uses the sparse apply's
///   beyond-rank-k accuracy, which GLR mode discards; inner conditioning is
///   governed by 1 + lambda_{k+1}/a with lambda_{k+1} the first pencil
///   value NOT in the block — pin it with extend_modes, which costs no
///   H_d access. A consumer wrapping THIS mode as a preconditioner must
///   use a flexible outer method (FCG/FGMRES): the inner iteration makes
///   the preconditioner slightly nonlinear.

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>

#include "lgpsf/corrections/shifted_operator.hpp"

namespace lgpsf::corrections {

/// Where a shift `a` stands relative to the contracts.
enum class Zone
{
    Guaranteed,
    Warned,
    Refused
};

/// The zone, and what is actually known about definiteness there.
struct ZoneReport
{
    Zone zone = Zone::Warned;
    /// The analytic sufficient condition (file comment), evaluated in the
    /// warning zone; true by contract in the guaranteed zone; false when
    /// nothing is certified or the condition fails (failure does NOT prove
    /// indefiniteness — the condition is sufficient, not necessary).
    bool analytic_pd = false;
    /// eig_min of the post-certification correction (0 if none, or if
    /// nothing is certified). The margin the warning-zone condition uses.
    double post_cert_min = 0.0;
    std::string detail;  ///< human-readable reason
};

/// Classify a shift against the struct's contracts. Cheap: at most one
/// rank-sized eigendecomposition.
inline ZoneReport classify_shift( const ShiftedOperator& A, double a )
{
    ZoneReport report;
    if ( !(a > 0.0) )
    {
        report.zone = Zone::Refused;
        report.detail = "a must be positive (H_r carries the shift)";
        return report;
    }
    if ( !A.lambda_floor.has_value() )
    {
        report.zone = Zone::Warned;
        report.detail =
            "no certified floor: make_pd has not certified, so definiteness "
            "is unknown at any a";
        return report;
    }
    const double floor = -*A.lambda_floor;
    if ( a <= floor )
    {
        report.zone = Zone::Refused;
        report.detail =
            "a = " + std::to_string(a) + " is at or below the certified "
            "floor " + std::to_string(floor)
            + " (B + E + a H_r is indefinite there); rebuild_at(a) restores "
              "the contracts from the archive";
        return report;
    }

    // the post-certification correction, exactly (merges only extend and
    // add — see C_corr_certified)
    const Eigen::Index now = A.block.rank();
    const Eigen::Index then = A.C_corr_certified.rows();
    Eigen::MatrixXd delta = A.block.C_corr;
    delta.topLeftCorner(then, then) -= A.C_corr_certified;
    if ( now > 0 && !delta.isZero(0.0) )
    {
        report.post_cert_min = std::min(
            0.0, Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(delta)
                     .eigenvalues()
                     .minCoeff());
    }

    if ( a >= A.a0 )
    {
        report.zone = Zone::Guaranteed;
        report.analytic_pd = true;
        report.detail = "a >= a0: the build contract applies";
        return report;
    }
    report.zone = Zone::Warned;
    report.analytic_pd = (a + *A.lambda_floor + report.post_cert_min) > 0.0;
    report.detail =
        report.analytic_pd
            ? "below a0 but above the floor; PD certified analytically "
              "(flip certificate + post-certification correction bound)"
            : "below a0 and the analytic sufficient condition fails -- not "
              "proof of indefiniteness, but nothing here certifies PD; an "
              "indefinite operator will reveal itself in CG";
    return report;
}

/// GLR or two-level (file comment).
enum class SolveMode
{
    Glr,
    TwoLevel
};

struct SolveOpts
{
    SolveMode mode = SolveMode::Glr;
    double oracle_tol = 1e-10;  ///< passed to every oracle solve
    double rtol = 1e-8;         ///< two-level: relative residual target
    int max_iters = 500;        ///< two-level: inner PCG cap
};

struct SolveResult
{
    Eigen::MatrixXd X;
    ZoneReport zone;
    int iterations = 0;          ///< inner PCG iterations (0 in GLR mode)
    double relative_residual = 0.0;  ///< of the system the mode solves:
                                     ///< M(a) X = B (GLR) or P(a) X = B
                                     ///< (two-level), max over columns
};

namespace detail {

/// PCG on P(a) = B + E + a H_r, preconditioned by the GLR Woodbury
/// M(a)^{-1}, one right-hand side. P(a) must be SPD for CG's theory; on a
/// warned shift an indefinite operator diagnoses itself (breakdown or
/// stagnation) rather than converging to a wrong answer.
inline Eigen::VectorXd pcg_two_level( const ShiftedOperator& A,
                                      const Eigen::VectorXd& b, double a,
                                      const SolveOpts& opts, int& iters,
                                      double& rel_res )
{
    const double b_norm = b.norm();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(A.dim());
    if ( b_norm == 0.0 )
    {
        iters = 0;
        rel_res = 0.0;
        return x;
    }
    Eigen::VectorXd r = b;
    Eigen::VectorXd z = glr_solve(A, r, a, opts.oracle_tol);
    Eigen::VectorXd p = z;
    double rz = r.dot(z);
    iters = 0;
    rel_res = 1.0;
    for ( int it = 0; it < opts.max_iters; ++it )
    {
        const Eigen::VectorXd Pp = apply(A, p, a);
        const double pAp = p.dot(Pp);
        if ( !(pAp > 0.0) )
        {
            throw std::runtime_error(
                "lgpsf::corrections::solve (two-level): CG met a direction "
                "of nonpositive curvature (p^T P(a) p = "
                + std::to_string(pAp) + ") -- P(a) is not positive definite "
                "at a = " + std::to_string(a));
        }
        const double alpha = rz / pAp;
        x += alpha * p;
        r -= alpha * Pp;
        ++iters;
        rel_res = r.norm() / b_norm;
        if ( rel_res <= opts.rtol )
        {
            break;
        }
        z = glr_solve(A, r, a, opts.oracle_tol);
        const double rz_next = r.dot(z);
        p = z + (rz_next / rz) * p;
        rz = rz_next;
    }
    return x;
}

} // end namespace detail

/// Solve at shift `a`, zone-checked. GLR mode applies M(a)^{-1} in closed
/// form; two-level mode solves P(a) itself by inner PCG preconditioned by
/// M(a)^{-1}. Refused shifts throw; warned shifts proceed and say so in
/// the returned ZoneReport.
inline SolveResult solve( const ShiftedOperator& A,
                          const Eigen::Ref<const Eigen::MatrixXd>& B_rhs,
                          double a, SolveOpts opts = {} )
{
    SolveResult result;
    result.zone = classify_shift(A, a);
    if ( result.zone.zone == Zone::Refused )
    {
        throw std::domain_error("lgpsf::corrections::solve: "
                                + result.zone.detail);
    }
    if ( opts.mode == SolveMode::Glr )
    {
        result.X = glr_solve(A, B_rhs, a, opts.oracle_tol);
        double worst = 0.0;
        const Eigen::MatrixXd residual = glr_apply(A, result.X, a) - B_rhs;
        for ( Eigen::Index j = 0; j < B_rhs.cols(); ++j )
        {
            const double scale = B_rhs.col(j).norm();
            if ( scale > 0.0 )
            {
                worst = std::max(worst, residual.col(j).norm() / scale);
            }
        }
        result.relative_residual = worst;
        return result;
    }
    result.X.resize(A.dim(), B_rhs.cols());
    for ( Eigen::Index j = 0; j < B_rhs.cols(); ++j )
    {
        int iters = 0;
        double rel_res = 0.0;
        result.X.col(j) =
            detail::pcg_two_level(A, B_rhs.col(j), a, opts, iters, rel_res);
        result.iterations = std::max(result.iterations, iters);
        result.relative_residual = std::max(result.relative_residual, rel_res);
    }
    return result;
}

} // end namespace lgpsf::corrections
