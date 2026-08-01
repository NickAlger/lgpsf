#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The H_r-orthonormal mode block: ONE low-rank object holding every
/// spectral correction the layer makes, in two coefficient matrices over one
/// shared basis.
///
/// The block holds an H_r-orthonormal basis V (V^T H_r V = I, a provenance
/// tag per column) and TWO symmetric coefficient matrices over it, because
/// the layer deploys two different operators:
///
///   correction   E      = (H_r V) C_corr (H_r V)^T   — added to B, so the
///                          struct represents  P0 = B + E;
///   surrogate    S      = (H_r V) C_surr (H_r V)^T   — the KNOWN spectral
///                          content of P0 on span(V), deployed as
///                          M(a) = a H_r + S, the GLR preconditioner.
///
/// One matrix cannot serve both roles. Two concrete failures force the
/// split: caching P0's rightmost pencil modes (for M) must NOT change P0 —
/// their correction coefficient is zero while their surrogate coefficient is
/// their eigenvalue; and a flipped mode needs -c*lambda in the correction
/// (that is the surgery on B) but its CORRECTED eigenvalue
/// (1-c)*lambda in the surrogate (that is what M must present).
///
/// The consistency rule callers maintain: whenever a merge changes the
/// operator (C_corr += Delta on span(V)), the same Delta is known content of
/// P0 and belongs in C_surr too — plus whatever content of B itself the
/// caller has learned (a flip pass knows the mode's B-eigenvalue from the
/// same Lanczos run; a deflation pass knows only its correction, and a later
/// cache extension can refine the surrogate there). `merge` therefore takes
/// both increments explicitly.
///
/// Why H_r-orthonormal: with W = H_r V, the shifted operator
/// M(a) = a H_r + W C_surr W^T inverts by Woodbury with W^T H_r^{-1} W = I,
/// so the capacitance is diagonal in the eigenbasis of C_surr and changing
/// `a` costs scalar arithmetic (the Woodbury lives with the deployable
/// struct, not here). The block's pencil eigenvalues against H_r are exactly
/// the eigenvalues of its coefficient matrices — which is what makes the
/// layer's PD certificates analytic instead of iterative.
///
/// Both V and H_r V are stored. Every application and every Gram needs
/// H_r V; storing it keeps those exactly consistent with merge-time
/// arithmetic and makes the struct self-contained without an oracle at hand.
///
/// Persistence follows the library convention: the struct is plain public
/// arrays (inspect, gather, save from Python with numpy), no bespoke format.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/corrections/hr_oracle.hpp"

namespace lgpsf::corrections {

/// Where a block column came from. `PencilCache` and `Flip` are CACHE —
/// recomputable from the operator and the oracle; `Deflation` and
/// `ValuePass` carry information whose only ground truth is the probe
/// archive. Correctness never depends on cache freshness.
enum class Provenance
{
    PencilCache,
    Flip,
    Deflation,
    ValuePass
};

/// The block. Plain data, free functions below; see the file comment.
struct ModeBlock
{
    Eigen::MatrixXd V;             ///< (N, rho), H_r-orthonormal columns
    Eigen::MatrixXd HrV;           ///< (N, rho), H_r V
    Eigen::MatrixXd C_corr;        ///< (rho, rho) sym: correction to B
    Eigen::MatrixXd C_surr;        ///< (rho, rho) sym: known content of B + E
    std::vector<Provenance> tags;  ///< per column of V

    Eigen::Index dim() const { return V.rows(); }
    Eigen::Index rank() const { return V.cols(); }
};

/// A block with no modes yet, over vectors of the given dimension.
inline ModeBlock empty_block( Eigen::Index dim )
{
    if ( dim <= 0 )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::empty_block: dim must be positive, got "
            + std::to_string(dim));
    }
    ModeBlock block;
    block.V = Eigen::MatrixXd(dim, 0);
    block.HrV = Eigen::MatrixXd(dim, 0);
    block.C_corr = Eigen::MatrixXd(0, 0);
    block.C_surr = Eigen::MatrixXd(0, 0);
    return block;
}

/// Structural consistency, reported rather than thrown (mirrors
/// `lgpsf::validate`). H_r-orthonormality is a numerical property of the
/// data, not a structural one — the tests own it.
inline std::vector<std::string> validate( const ModeBlock& block )
{
    std::vector<std::string> issues;
    if ( block.V.rows() <= 0 )
    {
        issues.push_back("V has no rows; build with empty_block(dim)");
    }
    if ( block.HrV.rows() != block.V.rows()
         || block.HrV.cols() != block.V.cols() )
    {
        issues.push_back("HrV shape does not match V");
    }
    for ( const auto* C : {&block.C_corr, &block.C_surr} )
    {
        const char* name = ( C == &block.C_corr ) ? "C_corr" : "C_surr";
        if ( C->rows() != block.V.cols() || C->cols() != block.V.cols() )
        {
            issues.push_back(std::string(name) + " is not (rank, rank)");
        }
        else if ( *C != C->transpose() )
        {
            issues.push_back(std::string(name)
                             + " is not exactly symmetric (merge keeps it so)");
        }
    }
    if ( static_cast<Eigen::Index>(block.tags.size()) != block.V.cols() )
    {
        issues.push_back("tags does not have one entry per column of V");
    }
    if ( !block.V.allFinite() || !block.HrV.allFinite()
         || !block.C_corr.allFinite() || !block.C_surr.allFinite() )
    {
        issues.push_back("non-finite values present");
    }
    return issues;
}

namespace detail {

inline Eigen::MatrixXd apply_block_matrix(
    const ModeBlock& block, const Eigen::MatrixXd& C,
    const Eigen::Ref<const Eigen::MatrixXd>& X, const char* where )
{
    if ( X.rows() != block.dim() )
    {
        throw std::invalid_argument(
            std::string("lgpsf::corrections::") + where + ": block has "
            + std::to_string(X.rows()) + " rows, block dim is "
            + std::to_string(block.dim()) + " (vectors are COLUMNS here)");
    }
    if ( block.rank() == 0 )
    {
        return Eigen::MatrixXd::Zero(X.rows(), X.cols());
    }
    return block.HrV * (C * (block.HrV.transpose() * X));
}

inline Eigen::VectorXd coefficient_eigenvalues( const Eigen::MatrixXd& C )
{
    if ( C.rows() == 0 )
    {
        return Eigen::VectorXd(0);
    }
    return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(C).eigenvalues();
}

} // end namespace detail

/// The correction's action: E X = (H_r V) C_corr (H_r V)^T X for `(N, m)`
/// columns X. O(N rho m); no oracle involved.
inline Eigen::MatrixXd apply_correction(
    const ModeBlock& block, const Eigen::Ref<const Eigen::MatrixXd>& X )
{
    return detail::apply_block_matrix(block, block.C_corr, X,
                                      "apply_correction");
}

/// The surrogate's action: S X = (H_r V) C_surr (H_r V)^T X — the low-rank
/// part of the GLR deployment operator M(a) = a H_r + S.
inline Eigen::MatrixXd apply_surrogate(
    const ModeBlock& block, const Eigen::Ref<const Eigen::MatrixXd>& X )
{
    return detail::apply_block_matrix(block, block.C_surr, X,
                                      "apply_surrogate");
}

/// Pencil eigenvalues of the CORRECTION against H_r — exactly
/// `eig(C_corr)`, ascending. Feeds the PD certificates for B + E + a H_r.
inline Eigen::VectorXd correction_eigenvalues( const ModeBlock& block )
{
    return detail::coefficient_eigenvalues(block.C_corr);
}

/// Pencil eigenvalues of the SURROGATE against H_r — exactly
/// `eig(C_surr)`, ascending. Feeds the PD certificate for M(a).
inline Eigen::VectorXd surrogate_eigenvalues( const ModeBlock& block )
{
    return detail::coefficient_eigenvalues(block.C_surr);
}

/// What `merge` did.
struct MergeReport
{
    int requested = 0;        ///< candidate directions offered
    int added = 0;            ///< new orthonormal columns appended
    double largest_dropped = 0.0;  ///< largest residual Gram eigenvalue among
                                   ///< dropped directions (0 if none dropped)
};

/// Fold a contribution into the block: on the candidate directions V_new,
/// add `Cc_new` to the operator correction and `Cs_new` to the surrogate
/// content (see the file comment for who passes what — a pure cache
/// extension passes Cc_new = 0; a pure correction with no learned content
/// passes Cs_new = Cc_new; a flip passes -c*lambda and (1-c)*lambda).
///
/// Existing columns are NEVER modified — new directions are orthonormalized
/// AGAINST the block (two Gram-Schmidt passes, rank-sized Grams, H_r applies
/// only), so provenance stays per-column and previously computed quantities
/// stay bitwise valid. The part of a candidate lying inside the current span
/// folds into the coefficients of the existing columns instead — merging the
/// same directions twice adds no columns, it adds coefficients. Candidates
/// whose out-of-block component has squared H_r-norm below
/// `drop_tol * max_j ||v_j||_{H_r}^2` contribute coefficients only (with the
/// default, directions below ~1e-7 of the largest candidate are folded).
///
/// Cost: q oracle APPLIES (no solves) + rank-sized dense algebra.
/// @param block    Updated in place.
/// @param hr       The H_r oracle (applies only).
/// @param V_new    (N, q) candidate directions; need not be orthonormal.
/// @param Cc_new   (q, q) correction coefficients; symmetrized on entry.
/// @param Cs_new   (q, q) surrogate coefficients; symmetrized on entry.
/// @param tag      Provenance for every column this merge appends.
/// @param drop_tol Relative SQUARED-norm floor for keeping a direction.
inline MergeReport merge( ModeBlock& block, const HrOracle& hr,
                          const Eigen::Ref<const Eigen::MatrixXd>& V_new,
                          const Eigen::Ref<const Eigen::MatrixXd>& Cc_new,
                          const Eigen::Ref<const Eigen::MatrixXd>& Cs_new,
                          Provenance tag, double drop_tol = 1e-14 )
{
    if ( V_new.rows() != block.dim() || hr.dim() != block.dim() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::merge: dimension mismatch between block ("
            + std::to_string(block.dim()) + "), candidates ("
            + std::to_string(V_new.rows()) + " rows) and oracle ("
            + std::to_string(hr.dim()) + ")");
    }
    if ( Cc_new.rows() != V_new.cols() || Cc_new.cols() != V_new.cols()
         || Cs_new.rows() != V_new.cols() || Cs_new.cols() != V_new.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::merge: Cc_new and Cs_new must be (q, q) for "
            "q candidate columns");
    }
    if ( !(drop_tol > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::merge: drop_tol must be positive");
    }
    MergeReport report;
    report.requested = static_cast<int>(V_new.cols());
    if ( V_new.cols() == 0 )
    {
        return report;
    }

    const Eigen::MatrixXd Cc = 0.5 * (Cc_new + Cc_new.transpose());
    const Eigen::MatrixXd Cs = 0.5 * (Cs_new + Cs_new.transpose());
    const Eigen::Index p = block.rank();
    const Eigen::Index q = V_new.cols();

    const Eigen::MatrixXd HrVnew = hr.apply(V_new);
    // scale for the drop decision: the largest candidate's squared H_r-norm
    const double scale =
        V_new.cwiseProduct(HrVnew).colwise().sum().maxCoeff();

    // coefficients of the candidates on the EXISTING columns (which do not
    // change below, so these stay exact)
    const Eigen::MatrixXd A = block.V.transpose() * HrVnew;  // (p, q)

    // first pass: project out the block, orthonormalize the residual by its
    // H_r-Gram eigendecomposition, dropping near-in-span directions
    Eigen::MatrixXd W = V_new - block.V * A;
    Eigen::MatrixXd HrW = HrVnew - block.HrV * A;
    Eigen::MatrixXd V_add(block.dim(), 0);
    Eigen::MatrixXd HrV_add(block.dim(), 0);
    if ( scale > 0.0 )
    {
        Eigen::MatrixXd G = W.transpose() * HrW;
        G = 0.5 * (G + G.transpose()).eval();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> first(G);
        std::vector<Eigen::Index> kept;
        for ( Eigen::Index i = 0; i < q; ++i )
        {
            if ( first.eigenvalues()(i) > drop_tol * scale )
            {
                kept.push_back(i);
            }
            else
            {
                report.largest_dropped =
                    std::max(report.largest_dropped,
                             std::max(first.eigenvalues()(i), 0.0));
            }
        }
        if ( !kept.empty() )
        {
            Eigen::MatrixXd X(q, static_cast<Eigen::Index>(kept.size()));
            for ( std::size_t k = 0; k < kept.size(); ++k )
            {
                X.col(static_cast<Eigen::Index>(k)) =
                    first.eigenvectors().col(kept[k])
                    / std::sqrt(first.eigenvalues()(kept[k]));
            }
            V_add = W * X;
            HrV_add = HrW * X;

            // second pass: twice is enough — re-project against the block,
            // then renormalize among themselves
            if ( p > 0 )
            {
                const Eigen::MatrixXd A2 = block.V.transpose() * HrV_add;
                V_add -= block.V * A2;
                HrV_add -= block.HrV * A2;
            }
            Eigen::MatrixXd G2 = V_add.transpose() * HrV_add;
            G2 = 0.5 * (G2 + G2.transpose()).eval();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> second(G2);
            // eigenvalues here are ~1 by construction; a nonpositive one can
            // only be roundoff on an already-dropped-scale direction
            std::vector<Eigen::Index> sane;
            for ( Eigen::Index i = 0; i < second.eigenvalues().size(); ++i )
            {
                if ( second.eigenvalues()(i) > 0.0 )
                {
                    sane.push_back(i);
                }
            }
            Eigen::MatrixXd X2(V_add.cols(),
                               static_cast<Eigen::Index>(sane.size()));
            for ( std::size_t k = 0; k < sane.size(); ++k )
            {
                X2.col(static_cast<Eigen::Index>(k)) =
                    second.eigenvectors().col(sane[k])
                    / std::sqrt(second.eigenvalues()(sane[k]));
            }
            V_add = (V_add * X2).eval();
            HrV_add = (HrV_add * X2).eval();
        }
    }
    const Eigen::Index r = V_add.cols();
    report.added = static_cast<int>(r);

    // exact coefficients of the candidates on the FINAL new columns
    const Eigen::MatrixXd B = V_add.transpose() * HrVnew;  // (r, q)

    // extended coefficient matrices: the candidates' contribution lands on
    // [existing, added] through M = [A; B], identically for both matrices
    Eigen::MatrixXd M(p + r, q);
    M.topRows(p) = A;
    M.bottomRows(r) = B;
    const auto extend = [&]( const Eigen::MatrixXd& C_old,
                             const Eigen::MatrixXd& C_inc ) {
        Eigen::MatrixXd C_ext = Eigen::MatrixXd::Zero(p + r, p + r);
        C_ext.topLeftCorner(p, p) = C_old;
        C_ext += M * C_inc * M.transpose();
        C_ext = 0.5 * (C_ext + C_ext.transpose()).eval();
        return C_ext;
    };
    Eigen::MatrixXd Cc_ext = extend(block.C_corr, Cc);
    Eigen::MatrixXd Cs_ext = extend(block.C_surr, Cs);

    Eigen::MatrixXd V_ext(block.dim(), p + r);
    V_ext.leftCols(p) = block.V;
    V_ext.rightCols(r) = V_add;
    Eigen::MatrixXd HrV_ext(block.dim(), p + r);
    HrV_ext.leftCols(p) = block.HrV;
    HrV_ext.rightCols(r) = HrV_add;

    block.V = std::move(V_ext);
    block.HrV = std::move(HrV_ext);
    block.C_corr = std::move(Cc_ext);
    block.C_surr = std::move(Cs_ext);
    block.tags.insert(block.tags.end(), static_cast<std::size_t>(r), tag);
    return report;
}

} // end namespace lgpsf::corrections
