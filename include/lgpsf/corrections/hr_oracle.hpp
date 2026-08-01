#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The regularization-operator boundary of the corrections layer:
/// apply H_r, and solve with it to a requested tolerance.
///
/// H_r is the consumer's SPD regularization operator — the right-hand side of
/// every pencil in this layer. The layer needs exactly two things from it:
/// `apply` (cheap, used for H_r-inner products and Gram matrices) and `solve`
/// (the expensive primitive — every spectral operation costs O(#modes) of
/// these and nothing bigger). In production `solve` is the consumer's own
/// machinery, typically Krylov preconditioned by multigrid, wrapped from C++
/// or Python; nothing in this layer ever factors anything N-sized.
///
/// The `tol` argument to `solve` is a RELATIVE accuracy request
/// (a Krylov implementation would read it as a relative residual tolerance).
/// It is advisory: an oracle may solve more accurately than asked — the
/// reference adapter below is direct and exact — but must not solve less.
///
/// Blocks are columns, `(dim, m)`, as everywhere in this layer.

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace lgpsf::corrections {

/// The H_r boundary, type-erased for the same reasons as `SymmetricOp`:
/// dispatch cost is nil at block granularity, the durable structs need an
/// owned handle, and the layer compiles once.
class HrOracle
{
public:
    /// Given `(dim, m)` columns X, return H_r X.
    using ApplyFn =
        std::function<Eigen::MatrixXd( const Eigen::Ref<const Eigen::MatrixXd>& )>;
    /// Given `(dim, m)` columns B and a relative tolerance, return H_r^{-1} B
    /// to (at least) that accuracy.
    using SolveFn = std::function<Eigen::MatrixXd(
        const Eigen::Ref<const Eigen::MatrixXd>&, double )>;

    HrOracle( Eigen::Index dim, ApplyFn apply, SolveFn solve )
        : dim_(dim), apply_(std::move(apply)), solve_(std::move(solve))
    {
        if ( dim_ <= 0 )
        {
            throw std::invalid_argument(
                "lgpsf::corrections::HrOracle: dim must be positive, got "
                + std::to_string(dim_));
        }
        if ( !apply_ || !solve_ )
        {
            throw std::invalid_argument(
                "lgpsf::corrections::HrOracle: apply and solve callables must "
                "both be non-empty");
        }
    }

    Eigen::Index dim() const { return dim_; }

    /// H_r X for a `(dim, m)` block X.
    Eigen::MatrixXd apply( const Eigen::Ref<const Eigen::MatrixXd>& X ) const
    {
        check_block(X, "apply");
        Eigen::MatrixXd result = apply_(X);
        check_result(result, X.cols(), "apply");
        return result;
    }

    /// H_r^{-1} B for a `(dim, m)` block B, to relative tolerance @p tol.
    Eigen::MatrixXd solve( const Eigen::Ref<const Eigen::MatrixXd>& B,
                           double tol ) const
    {
        if ( !(tol > 0.0) )
        {
            throw std::invalid_argument(
                "lgpsf::corrections::HrOracle::solve: tol must be positive, "
                "got " + std::to_string(tol));
        }
        check_block(B, "solve");
        Eigen::MatrixXd result = solve_(B, tol);
        check_result(result, B.cols(), "solve");
        return result;
    }

private:
    void check_block( const Eigen::Ref<const Eigen::MatrixXd>& X,
                      const char* where ) const
    {
        if ( X.rows() != dim_ )
        {
            throw std::invalid_argument(
                std::string("lgpsf::corrections::HrOracle::") + where
                + ": block has " + std::to_string(X.rows())
                + " rows, oracle dim is " + std::to_string(dim_)
                + " (vectors are COLUMNS here)");
        }
    }
    void check_result( const Eigen::MatrixXd& result, Eigen::Index cols,
                       const char* where ) const
    {
        if ( result.rows() != dim_ || result.cols() != cols )
        {
            throw std::runtime_error(
                std::string("lgpsf::corrections::HrOracle::") + where
                + ": the wrapped callable returned ("
                + std::to_string(result.rows()) + ", "
                + std::to_string(result.cols()) + ") for a ("
                + std::to_string(dim_) + ", " + std::to_string(cols)
                + ") input");
        }
    }

    Eigen::Index dim_;
    ApplyFn apply_;
    SolveFn solve_;
};

/// The reference adapter: wrap a sparse SPD H_r, factoring it once with
/// SimplicialLLT. Solves are direct and exact (`tol` is honored trivially).
///
/// This is the testing / small-N path, NOT the scalable one — an N-sized
/// factorization is exactly what the layer's required path avoids (plan, P2).
/// It exists so the layer can be exercised hermetically, and so a consumer
/// with a small 2-D problem has a zero-effort oracle.
///
/// Taken BY VALUE; the handle owns the matrix and the factorization.
/// @throws std::invalid_argument if the matrix is not square or the
///         factorization reports it is not positive definite.
inline HrOracle sparse_hr_oracle( Eigen::SparseMatrix<double> Hr )
{
    if ( Hr.rows() != Hr.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::sparse_hr_oracle: matrix is "
            + std::to_string(Hr.rows()) + " x " + std::to_string(Hr.cols())
            + "; H_r must be square (and SPD)");
    }
    auto held = std::make_shared<Eigen::SparseMatrix<double>>(std::move(Hr));
    held->makeCompressed();
    auto factor =
        std::make_shared<Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>>();
    factor->compute(*held);
    if ( factor->info() != Eigen::Success )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::sparse_hr_oracle: SimplicialLLT failed -- "
            "the matrix is not positive definite (or is structurally "
            "deficient)");
    }
    return HrOracle(
        held->rows(),
        [held]( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
        { return (*held) * X; },
        [factor]( const Eigen::Ref<const Eigen::MatrixXd>& B,
                  double /*tol: direct solve, trivially honored*/ )
            -> Eigen::MatrixXd { return factor->solve(B); });
}

} // end namespace lgpsf::corrections
