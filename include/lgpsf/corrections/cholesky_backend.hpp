#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The optional, capability-gated Cholesky backend — the ONE file in
/// the corrections layer allowed to see matrix entries.
///
/// The required path never factors anything N-sized (plan, P2): sparse
/// factorization of B + a H_r is only scalable in 2-D, and the fit's
/// interesting regime — large ellipsoids, wide stencils — is its worst
/// case. But where it IS viable (small N, 2-D), a direct factorization is
/// both a fast exact solve and an independent check on the whole iterative
/// stack. This backend activates only when the caller ADDITIONALLY
/// supplies sparse forms of B and H_r; construction verifies them
/// stochastically against the struct's own operator handles, so a
/// mismatched matrix is refused rather than silently trusted.
///
/// The solve factors S(a) = B_sparse + a H_r_sparse once per shift (keyed
/// cache) and wraps the block correction around the factorization by
/// Woodbury, in the form valid for ANY symmetric coefficient matrix:
///
///   P(a)^{-1} = S^{-1} - S^{-1} W (I + C M)^{-1} C W^T S^{-1},
///       W = H_r V,   M = W^T S^{-1} W,   C = C_corr.
///
/// `sparse_part_pd` is the alternative exact PD certificate the plan
/// promised: LDLT inertia of S(a), i.e. positive definiteness of the RAW
/// sparse pencil at shift a — the entry-level cross-check on the layer's
/// analytic certificates.

#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include "lgpsf/corrections/shifted_operator.hpp"

namespace lgpsf::corrections {

namespace detail {

struct CholeskyFactors
{
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::MatrixXd SinvW;               // S^{-1} W, (N, rho)
    Eigen::PartialPivLU<Eigen::MatrixXd> capacitance;  // I + C M
    Eigen::Index block_rank = 0;         // rank the cache entry was built at
};

} // end namespace detail

/// The backend: the two sparse matrices plus a per-shift factorization
/// cache. Copyable (the cache is shared); invalidate nothing by hand —
/// a cache entry rebuilds automatically when the block has grown past it.
struct CholeskyBackend
{
    Eigen::SparseMatrix<double> B_sparse;
    Eigen::SparseMatrix<double> Hr_sparse;
    std::shared_ptr<std::map<double, std::shared_ptr<detail::CholeskyFactors>>>
        cache = std::make_shared<
            std::map<double, std::shared_ptr<detail::CholeskyFactors>>>();
};

/// Build the backend, verifying both matrices against the struct's operator
/// handles on seeded random blocks — the capability gate with its guard.
inline CholeskyBackend make_cholesky_backend(
    Eigen::SparseMatrix<double> B_sparse,
    Eigen::SparseMatrix<double> Hr_sparse, const ShiftedOperator& A,
    int check_cols = 4, double check_tol = 1e-10, unsigned seed = 0 )
{
    if ( B_sparse.rows() != A.dim() || B_sparse.cols() != A.dim()
         || Hr_sparse.rows() != A.dim() || Hr_sparse.cols() != A.dim() )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::make_cholesky_backend: matrix dims do not "
            "match the struct (" + std::to_string(A.dim()) + ")");
    }
    std::mt19937 gen(seed ^ 0x9E3779B9u);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::MatrixXd X(A.dim(), check_cols);
    for ( Eigen::Index j = 0; j < check_cols; ++j )
    {
        for ( Eigen::Index i = 0; i < A.dim(); ++i )
        {
            X(i, j) = normal(gen);
        }
    }
    const auto mismatch = [&]( const Eigen::MatrixXd& via_matrix,
                               const Eigen::MatrixXd& via_handle ) {
        const double scale = via_handle.cwiseAbs().maxCoeff();
        return (via_matrix - via_handle).cwiseAbs().maxCoeff()
               > check_tol * std::max(scale, 1e-300);
    };
    if ( mismatch(B_sparse * X, A.op.apply(X)) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::make_cholesky_backend: B_sparse disagrees "
            "with the struct's operator handle -- refusing a matrix that is "
            "not the operator");
    }
    if ( mismatch(Hr_sparse * X, A.hr.apply(X)) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::make_cholesky_backend: Hr_sparse disagrees "
            "with the struct's H_r oracle");
    }
    CholeskyBackend backend;
    backend.B_sparse = std::move(B_sparse);
    backend.Hr_sparse = std::move(Hr_sparse);
    backend.B_sparse.makeCompressed();
    backend.Hr_sparse.makeCompressed();
    return backend;
}

namespace detail {

inline std::shared_ptr<CholeskyFactors> factors_at(
    const CholeskyBackend& backend, const ShiftedOperator& A, double a )
{
    auto found = backend.cache->find(a);
    if ( found != backend.cache->end()
         && found->second->block_rank == A.block.rank() )
    {
        return found->second;
    }
    auto factors = std::make_shared<CholeskyFactors>();
    const Eigen::SparseMatrix<double> S =
        backend.B_sparse + a * backend.Hr_sparse;
    factors->solver.compute(S);
    if ( factors->solver.info() != Eigen::Success )
    {
        throw std::runtime_error(
            "lgpsf::corrections cholesky backend: LDLT factorization of "
            "B + a Hr failed at a = " + std::to_string(a));
    }
    factors->block_rank = A.block.rank();
    if ( A.block.rank() > 0 )
    {
        factors->SinvW = factors->solver.solve(A.block.HrV);
        const Eigen::MatrixXd M =
            A.block.HrV.transpose() * factors->SinvW;
        factors->capacitance.compute(
            Eigen::MatrixXd(Eigen::MatrixXd::Identity(A.block.rank(),
                                                      A.block.rank())
                            + A.block.C_corr * M));
    }
    (*backend.cache)[a] = factors;
    return factors;
}

} // end namespace detail

/// Direct solve of P(a) = B + E + a H_r through the sparse factorization,
/// block correction wrapped by Woodbury. Exact to factorization accuracy;
/// the reference against which the iterative paths can be checked.
inline Eigen::MatrixXd cholesky_solve(
    const CholeskyBackend& backend, const ShiftedOperator& A,
    const Eigen::Ref<const Eigen::MatrixXd>& B_rhs, double a )
{
    const auto factors = detail::factors_at(backend, A, a);
    Eigen::MatrixXd X = factors->solver.solve(B_rhs);
    if ( A.block.rank() > 0 )
    {
        X.noalias() -= factors->SinvW
                       * factors->capacitance.solve(
                             A.block.C_corr
                             * (A.block.HrV.transpose() * X));
    }
    return X;
}

/// The entry-level exact PD certificate: is the RAW sparse pencil
/// B + a H_r positive definite at this shift? Read off the LDLT inertia.
/// Cross-checks the layer's analytic certificates (for the raw operator,
/// this flips exactly at minus the leftmost raw pencil eigenvalue).
inline bool sparse_part_pd( const CholeskyBackend& backend, double a )
{
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(Eigen::SparseMatrix<double>(backend.B_sparse
                                               + a * backend.Hr_sparse));
    if ( solver.info() != Eigen::Success )
    {
        return false;
    }
    return (solver.vectorD().array() > 0.0).all();
}

} // end namespace lgpsf::corrections
