#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The durable struct of the corrections layer, and its GLR-mode
/// deployment: apply the corrected operator, and apply the Woodbury inverse
/// of its low-rank + shift part at ANY shift `a` with no refactorization.
///
/// A `ShiftedOperator` represents
///
///   P(a)  =  B + E + a H_r,     E = (H_r V) C_corr (H_r V)^T  (the block's
///                               correction content),
///
/// for a symmetric operator B, a consumer-supplied SPD H_r, and a
/// caller-supplied shift `a` — the struct carries the BUILD shift `a0` but
/// every operation takes `a` as an argument (varying regularization is the
/// normal case, not the exception). The GLR deployment object is
///
///   M(a)  =  a H_r + S,         S = (H_r V) C_surr (H_r V)^T  (the block's
///                               surrogate content: the known spectral
///                               content of B + E),
///
/// whose inverse closes over the H_r oracle analytically BECAUSE V is
/// H_r-orthonormal: with C_surr = U diag(theta) U^T,
///
///   M(a)^{-1} x = (1/a) H_r^{-1} x
///                 - V U diag(theta_i / (a (a + theta_i))) U^T V^T x,
///
/// one oracle solve plus O(N rho) per application, exact for every symmetric
/// C, and positive definite IFF a > max(0, -min_i theta_i) — an analytic
/// certificate, not an iterative one. Changing `a` is scalar arithmetic; the
/// rank-sized eigendecomposition of C is recomputed per call on purpose
/// (microseconds against an N-sized oracle solve, and it can never go stale
/// against the caller-visible block).
///
/// What this header does NOT do: pencil Lanczos / flip (they fill the block
/// and lambda_floor), zone semantics for a < a0, the two-level solve, and
/// deflation — later slices, layered on exactly these pieces.

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/corrections/hr_oracle.hpp"
#include "lgpsf/corrections/mode_block.hpp"
#include "lgpsf/corrections/symmetric_op.hpp"

namespace lgpsf::corrections {

/// Every trace of the true operator H_d the consumer ever paid for: probe
/// pairs, held-out QC pairs, value-pass pairs. Irreplaceable — the fit and
/// every deflation are derived from this, and rebuilds at a new shift re-read
/// it instead of touching H_d again. Plain data, columns convention `(N, k)`;
/// empty `(0, 0)` members mean "absent".
struct ProbeArchive
{
    Eigen::MatrixXd Z;      ///< probes, (N, k)
    Eigen::MatrixXd Y;      ///< H_d Z, (N, k)
    Eigen::MatrixXd Z_qc;   ///< held-out probes, (N, k_qc)
    Eigen::MatrixXd Y_qc;   ///< their true responses, (N, k_qc)
    Eigen::MatrixXd Q_vp;   ///< value-pass directions, (N, m)
    Eigen::MatrixXd HdQ_vp; ///< their true responses, (N, m)
};

/// Build-time options for `make_shifted_operator`.
struct BuildOptions
{
    /// Refuse an operator whose measured `symmetry_defect` exceeds this.
    /// Symmetric operators measure ~1e-15; an unsymmetrized row fit measures
    /// orders of magnitude above — the gap is wide, the default generous.
    double symmetry_tol = 1e-8;
    int symmetry_pairs = 8;        ///< Gaussian pairs for the check
    unsigned symmetry_seed = 0;    ///< its seed (deterministic)
};

/// The durable struct: the operator handle, the oracle, the archive, ONE
/// mode block, and the contract scalars. Plain public members — everything
/// but `op` and `hr` persists as arrays (library convention); the operator
/// and oracle are re-supplied on load, never serialized.
struct ShiftedOperator
{
    SymmetricOp op;         ///< B, the symmetric operator being corrected
    HrOracle hr;            ///< the consumer's H_r
    ProbeArchive archive;   ///< every trace of H_d
    ModeBlock block;        ///< all spectral corrections, one object

    double a0;              ///< the build shift; operations take their own `a`
    double gamma = 0.5;     ///< flip safety factor (meaningful after make_pd)
    double clamp_floor = -0.95;  ///< deflation eigenvalue clamp
    /// Leftmost surviving pencil eigenvalue of B_pd against H_r; set by
    /// make_pd. The exact PD floor of the sparse part: B_pd + a H_r > 0
    /// iff a > -lambda_floor. Absent until a flip pass has run.
    std::optional<double> lambda_floor;

    Eigen::Index dim() const { return op.dim(); }
};

/// Structural consistency across the members, reported rather than thrown.
inline std::vector<std::string> validate( const ShiftedOperator& A )
{
    std::vector<std::string> issues = validate(A.block);
    const Eigen::Index n = A.op.dim();
    if ( A.hr.dim() != n )
    {
        issues.push_back("H_r oracle dim does not match the operator");
    }
    if ( A.block.dim() != n )
    {
        issues.push_back("mode block dim does not match the operator");
    }
    if ( !(A.a0 > 0.0) )
    {
        issues.push_back("a0 must be positive");
    }
    if ( !(A.gamma > 0.0 && A.gamma < 1.0) )
    {
        issues.push_back("gamma must lie in (0, 1)");
    }
    const auto pair_check = [&]( const Eigen::MatrixXd& L,
                                 const Eigen::MatrixXd& R, const char* name ) {
        if ( L.size() == 0 && R.size() == 0 )
        {
            return;  // absent is fine
        }
        if ( L.rows() != n || R.rows() != n || L.cols() != R.cols() )
        {
            issues.push_back(std::string("archive ") + name
                             + " pair is inconsistent with the operator dim");
        }
    };
    pair_check(A.archive.Z, A.archive.Y, "probe");
    pair_check(A.archive.Z_qc, A.archive.Y_qc, "QC");
    pair_check(A.archive.Q_vp, A.archive.HdQ_vp, "value-pass");
    return issues;
}

/// Build the struct. Verifies the operator's symmetry (seeded stochastic
/// check, matvecs only) — the one misuse the producing side cannot warn
/// about is an unsymmetrized operator handed across, and it is caught here.
/// The block starts empty; later operations (Lanczos, flip, deflation) fill
/// it.
inline ShiftedOperator make_shifted_operator( SymmetricOp op,
                                              ProbeArchive archive,
                                              HrOracle hr, double a0,
                                              BuildOptions opts = {} )
{
    if ( !(a0 > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::make_shifted_operator: a0 must be positive, "
            "got " + std::to_string(a0));
    }
    if ( hr.dim() != op.dim() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::make_shifted_operator: operator dim "
            + std::to_string(op.dim()) + " != oracle dim "
            + std::to_string(hr.dim()));
    }
    const double defect =
        symmetry_defect(op, opts.symmetry_pairs, opts.symmetry_seed);
    if ( !(defect <= opts.symmetry_tol) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::make_shifted_operator: the operator is not "
            "symmetric (measured defect " + std::to_string(defect)
            + " > tol " + std::to_string(opts.symmetry_tol)
            + "). For an lgpsf fit, assemble with Symmetrize::Weighted.");
    }
    const Eigen::Index n = op.dim();
    // full aggregate init (the trailing values restate the declared member
    // defaults) so partial-initialization warnings stay meaningful elsewhere
    ShiftedOperator A{std::move(op),   std::move(hr), std::move(archive),
                      empty_block(n),  a0,            0.5,
                      -0.95,           std::nullopt};
    const std::vector<std::string> issues = validate(A);
    if ( !issues.empty() )
    {
        std::string what =
            "lgpsf::corrections::make_shifted_operator: inconsistent inputs:";
        for ( const std::string& issue : issues )
        {
            what += "\n  - " + issue;
        }
        throw std::invalid_argument(what);
    }
    return A;
}

/// The corrected, shifted operator's action: (B + E + a H_r) X.
/// `a = 0` is legal here (it is just B + E); definiteness is a property of
/// the SOLVE paths, not of applying.
inline Eigen::MatrixXd apply( const ShiftedOperator& A,
                              const Eigen::Ref<const Eigen::MatrixXd>& X,
                              double a )
{
    Eigen::MatrixXd result = A.op.apply(X) + apply_correction(A.block, X);
    if ( a != 0.0 )
    {
        result += a * A.hr.apply(X);
    }
    return result;
}

/// The GLR deployment operator's action: M(a) X = a H_r X + S X, with S the
/// block's SURROGATE content (the known spectral content of B + E).
inline Eigen::MatrixXd glr_apply( const ShiftedOperator& A,
                                  const Eigen::Ref<const Eigen::MatrixXd>& X,
                                  double a )
{
    return a * A.hr.apply(X) + apply_surrogate(A.block, X);
}

/// The exact PD floor of M(a): M(a) > 0 iff a > glr_pd_floor(A). Analytic —
/// the surrogate's pencil eigenvalues against H_r are eig(C_surr) exactly.
inline double glr_pd_floor( const ShiftedOperator& A )
{
    if ( A.block.rank() == 0 )
    {
        return 0.0;
    }
    return std::max(0.0, -surrogate_eigenvalues(A.block).minCoeff());
}

/// M(a)^{-1} B_rhs by the diagonal-capacitance Woodbury formula (file
/// comment): one oracle solve at `oracle_tol` plus O(N rho) per column.
/// Valid at EVERY a above the analytic floor with zero refactorization —
/// an L-curve sweep re-calls this with different `a` and pays nothing new.
/// @throws std::domain_error if a is at or below `glr_pd_floor` — the
///         message carries the floor, which is the number the caller needs.
inline Eigen::MatrixXd glr_solve( const ShiftedOperator& A,
                                  const Eigen::Ref<const Eigen::MatrixXd>& B_rhs,
                                  double a, double oracle_tol = 1e-10 )
{
    const double floor = glr_pd_floor(A);
    if ( !(a > floor) )
    {
        throw std::domain_error(
            "lgpsf::corrections::glr_solve: M(a) is not positive definite at "
            "a = " + std::to_string(a) + "; the certified floor is "
            + std::to_string(floor)
            + " (a must exceed it). Analytic certificate from the block's "
              "pencil eigenvalues.");
    }
    Eigen::MatrixXd result = A.hr.solve(B_rhs, oracle_tol) / a;
    if ( A.block.rank() > 0 )
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(A.block.C_surr);
        const Eigen::VectorXd& theta = eigen.eigenvalues();
        const Eigen::MatrixXd& U = eigen.eigenvectors();
        Eigen::MatrixXd P = U.transpose() * (A.block.V.transpose() * B_rhs);
        for ( Eigen::Index i = 0; i < theta.size(); ++i )
        {
            P.row(i) *= theta(i) / (a * (a + theta(i)));
        }
        result.noalias() -= A.block.V * (U * P);
    }
    return result;
}

} // end namespace lgpsf::corrections
