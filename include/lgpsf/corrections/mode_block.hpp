#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The H_r-orthonormal mode block: ONE low-rank object holding every
/// spectral correction the layer makes.
///
/// The block represents the symmetric low-rank correction
///
///   E  =  (H_r V) C (H_r V)^T,        V^T H_r V = I,
///
/// with a provenance tag per column of V. Flip modes, cached pencil modes,
/// deflation modes and value-pass modes are all THE SAME KIND of object —
/// H_r-orthonormal directions with a symmetric coefficient matrix — so they
/// live in one block and are merged by H_r-Gram orthonormalization rather
/// than accumulating as separate terms.
///
/// Why H_r-orthonormal: with W = H_r V, the shifted operator
/// M(a) = a H_r + W C W^T inverts by Woodbury with W^T H_r^{-1} W = I, so the
/// capacitance is diagonal in the eigenbasis of C and changing `a` costs
/// scalar arithmetic (see the plan, §3; the Woodbury lives with the
/// deployable struct, not here). The block's pencil eigenvalues against H_r
/// are exactly `eig(C)` — which is what makes the layer's PD certificates
/// analytic instead of iterative.
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
    Eigen::MatrixXd C;             ///< (rho, rho), symmetric coefficients
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
    block.C = Eigen::MatrixXd(0, 0);
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
    if ( block.C.rows() != block.V.cols() || block.C.cols() != block.V.cols() )
    {
        issues.push_back("C is not (rank, rank)");
    }
    if ( block.C != block.C.transpose() )
    {
        issues.push_back("C is not exactly symmetric (merge keeps it so)");
    }
    if ( static_cast<Eigen::Index>(block.tags.size()) != block.V.cols() )
    {
        issues.push_back("tags does not have one entry per column of V");
    }
    if ( !block.V.allFinite() || !block.HrV.allFinite() || !block.C.allFinite() )
    {
        issues.push_back("non-finite values present");
    }
    return issues;
}

/// The correction's action: E X = (H_r V) C (H_r V)^T X for `(N, m)` columns
/// X. O(N rho m); no oracle involved.
inline Eigen::MatrixXd apply_correction(
    const ModeBlock& block, const Eigen::Ref<const Eigen::MatrixXd>& X )
{
    if ( X.rows() != block.dim() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::apply_correction: block has "
            + std::to_string(X.rows()) + " rows, block dim is "
            + std::to_string(block.dim()) + " (vectors are COLUMNS here)");
    }
    if ( block.rank() == 0 )
    {
        return Eigen::MatrixXd::Zero(X.rows(), X.cols());
    }
    return block.HrV * (block.C * (block.HrV.transpose() * X));
}

/// The block's pencil eigenvalues against H_r — exactly `eig(C)`, ascending,
/// because V is H_r-orthonormal. The analytic ingredient of every PD
/// certificate in the layer.
inline Eigen::VectorXd pencil_eigenvalues( const ModeBlock& block )
{
    if ( block.rank() == 0 )
    {
        return Eigen::VectorXd(0);
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(block.C);
    return eigen.eigenvalues();
}

/// What `merge` did.
struct MergeReport
{
    int requested = 0;        ///< candidate directions offered
    int added = 0;            ///< new orthonormal columns appended
    double largest_dropped = 0.0;  ///< largest residual Gram eigenvalue among
                                   ///< dropped directions (0 if none dropped)
};

/// Fold the contribution  (H_r V_new) C_new (H_r V_new)^T  into the block.
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
/// @param C_new    (q, q) coefficients; symmetrized on entry.
/// @param tag      Provenance for every column this merge appends.
/// @param drop_tol Relative SQUARED-norm floor for keeping a direction.
inline MergeReport merge( ModeBlock& block, const HrOracle& hr,
                          const Eigen::Ref<const Eigen::MatrixXd>& V_new,
                          const Eigen::Ref<const Eigen::MatrixXd>& C_new,
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
    if ( C_new.rows() != V_new.cols() || C_new.cols() != V_new.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::merge: C_new must be (q, q) for q candidate "
            "columns");
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

    const Eigen::MatrixXd Cn = 0.5 * (C_new + C_new.transpose());
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

    // extended coefficient matrix: the candidates' contribution lands on
    // [existing, added] through M = [A; B]
    Eigen::MatrixXd M(p + r, q);
    M.topRows(p) = A;
    M.bottomRows(r) = B;
    Eigen::MatrixXd C_ext = Eigen::MatrixXd::Zero(p + r, p + r);
    C_ext.topLeftCorner(p, p) = block.C;
    C_ext += M * Cn * M.transpose();
    C_ext = 0.5 * (C_ext + C_ext.transpose()).eval();

    Eigen::MatrixXd V_ext(block.dim(), p + r);
    V_ext.leftCols(p) = block.V;
    V_ext.rightCols(r) = V_add;
    Eigen::MatrixXd HrV_ext(block.dim(), p + r);
    HrV_ext.leftCols(p) = block.HrV;
    HrV_ext.rightCols(r) = HrV_add;

    block.V = std::move(V_ext);
    block.HrV = std::move(HrV_ext);
    block.C = std::move(C_ext);
    block.tags.insert(block.tags.end(), static_cast<std::size_t>(r), tag);
    return report;
}

} // end namespace lgpsf::corrections
