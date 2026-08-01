#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The operator boundary of the corrections layer: a symmetric linear
/// operator known only through its matvec.
///
/// The corrections layer (`lgpsf::corrections`, this directory) is
/// OPERATOR-BLIND: nothing in it reads a matrix entry. It consumes the
/// operator to be corrected through this one type — a dimension plus a block
/// matvec — so the same machinery serves a sparse lgpsf assembly, a
/// block-low-rank format, a dense test matrix, or a shell around someone
/// else's code, and porting to distributed memory is a change of vector type,
/// not of algorithm. See `dev/pencil-corrections-plan.md`, principle P4.
///
/// Two contracts a wrapped operator must honor:
///
/// - **Symmetry.** Everything crossing this boundary is symmetric; no
///   transpose matvec exists here on purpose. Symmetrization is the
///   PRODUCER's job (for lgpsf, `assemble_sparse` with
///   `Symmetrize::Weighted` — the weighted formula is per-entry and cannot
///   be composed from matvecs). `symmetry_defect` measures compliance from
///   matvecs alone; the layer checks it before trusting an operator.
///
/// - **Blocks are columns.** A block of m vectors is `(dim, m)`, one vector
///   per COLUMN — this layer speaks linear algebra. Note the fit layer's
///   probe convention is the opposite (records as rows); adapters between
///   the two worlds are where the transposition happens.

#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace lgpsf::corrections {

/// A symmetric linear operator, type-erased to dimension + block matvec.
///
/// Type-erased rather than a template parameter on purpose: one indirect call
/// amortized over an (N x m) block application is free, the durable structs
/// of this layer need an OWNED, storable handle (a wrapped callable keeps
/// whatever it captured alive), and the layer compiles once instead of once
/// per operator type.
///
/// The handle itself cannot verify symmetry cheaply, so construction accepts
/// any callable of the right shape; `symmetry_defect` is the measurement, and
/// the entry points that build durable objects from an operator run it.
class SymmetricOp
{
public:
    /// The wrapped matvec: given `(dim, m)` columns, return `(dim, m)` columns.
    using ApplyFn =
        std::function<Eigen::MatrixXd( const Eigen::Ref<const Eigen::MatrixXd>& )>;

    SymmetricOp( Eigen::Index dim, ApplyFn apply )
        : dim_(dim), apply_(std::move(apply))
    {
        if ( dim_ <= 0 )
        {
            throw std::invalid_argument(
                "lgpsf::corrections::SymmetricOp: dim must be positive, got "
                + std::to_string(dim_));
        }
        if ( !apply_ )
        {
            throw std::invalid_argument(
                "lgpsf::corrections::SymmetricOp: apply callable is empty");
        }
    }

    Eigen::Index dim() const { return dim_; }

    /// Apply to a block of column vectors. @p X is `(dim, m)`; so is the result.
    /// @throws std::invalid_argument on a wrong-shaped input;
    ///         std::runtime_error if the wrapped callable breaks its shape
    ///         contract (that is the callable's bug, not the caller's).
    Eigen::MatrixXd apply( const Eigen::Ref<const Eigen::MatrixXd>& X ) const
    {
        if ( X.rows() != dim_ )
        {
            throw std::invalid_argument(
                "lgpsf::corrections::SymmetricOp::apply: block has "
                + std::to_string(X.rows()) + " rows, operator dim is "
                + std::to_string(dim_) + " (vectors are COLUMNS here)");
        }
        Eigen::MatrixXd result = apply_(X);
        if ( result.rows() != dim_ || result.cols() != X.cols() )
        {
            throw std::runtime_error(
                "lgpsf::corrections::SymmetricOp::apply: the wrapped callable "
                "returned (" + std::to_string(result.rows()) + ", "
                + std::to_string(result.cols()) + ") for a ("
                + std::to_string(dim_) + ", " + std::to_string(X.cols())
                + ") input");
        }
        return result;
    }

private:
    Eigen::Index dim_;
    ApplyFn apply_;
};

/// Wrap a sparse matrix. Taken BY VALUE: the handle owns its copy, so it can
/// outlive the caller's matrix. Symmetry is the caller's responsibility —
/// for an lgpsf fit, assemble with `Symmetrize::Weighted` first.
inline SymmetricOp sparse_op( Eigen::SparseMatrix<double> A )
{
    if ( A.rows() != A.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::sparse_op: matrix is "
            + std::to_string(A.rows()) + " x " + std::to_string(A.cols())
            + "; a SymmetricOp must be square");
    }
    auto held = std::make_shared<Eigen::SparseMatrix<double>>(std::move(A));
    held->makeCompressed();
    return SymmetricOp(
        held->rows(),
        [held]( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
        { return (*held) * X; });
}

/// Wrap a dense matrix, same ownership rule. Mainly for tests — including the
/// layer's own format-independence test (dense vs sparse wrapping of the same
/// matrix must give identical downstream results).
inline SymmetricOp dense_op( Eigen::MatrixXd A )
{
    if ( A.rows() != A.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::dense_op: matrix is "
            + std::to_string(A.rows()) + " x " + std::to_string(A.cols())
            + "; a SymmetricOp must be square");
    }
    auto held = std::make_shared<Eigen::MatrixXd>(std::move(A));
    return SymmetricOp(
        held->rows(),
        [held]( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
        { return (*held) * X; });
}

/// Measured symmetry defect, from matvecs alone.
///
/// Draws `pairs` Gaussian pairs (x, y) from the given seed and returns the
/// largest relative defect
///   |x^T(By) - y^T(Bx)| / (||Bx|| ||y|| + ||By|| ||x||),
/// i.e. the asymmetric part of the bilinear form relative to its natural
/// scale. A genuinely symmetric operator measures at rounding level (~1e-16,
/// growing mildly with dim); an lgpsf fit assembled WITHOUT symmetrization
/// measures at its actual relative asymmetry, orders of magnitude above that
/// — the gap is wide, so the test needs no delicate threshold.
///
/// Cost: two block applies of `pairs` columns. Deterministic in `seed`.
inline double symmetry_defect( const SymmetricOp& B, int pairs = 8,
                               unsigned seed = 0 )
{
    if ( pairs <= 0 )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::symmetry_defect: pairs must be positive, got "
            + std::to_string(pairs));
    }
    const Eigen::Index n = B.dim();
    std::mt19937 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::MatrixXd X(n, pairs);
    Eigen::MatrixXd Y(n, pairs);
    for ( Eigen::Index j = 0; j < pairs; ++j )
    {
        for ( Eigen::Index i = 0; i < n; ++i )
        {
            X(i, j) = normal(gen);
            Y(i, j) = normal(gen);
        }
    }
    const Eigen::MatrixXd BX = B.apply(X);
    const Eigen::MatrixXd BY = B.apply(Y);

    double worst = 0.0;
    for ( Eigen::Index j = 0; j < pairs; ++j )
    {
        const double forward = X.col(j).dot(BY.col(j));
        const double backward = Y.col(j).dot(BX.col(j));
        const double scale = BX.col(j).norm() * Y.col(j).norm()
                             + BY.col(j).norm() * X.col(j).norm();
        if ( scale > 0.0 )
        {
            worst = std::max(worst, std::abs(forward - backward) / scale);
        }
    }
    return worst;
}

} // end namespace lgpsf::corrections
