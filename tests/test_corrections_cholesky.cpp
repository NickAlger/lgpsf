// SPDX-License-Identifier: MIT
//
// The capability-gated Cholesky backend: refused when the matrices are not
// the operator, exact when they are, agreeing three ways with the iterative
// stack, and providing the entry-level PD certificate for the raw pencil.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/cholesky_backend.hpp"
#include "lgpsf/corrections/pencil_lanczos.hpp"
#include "lgpsf/corrections/solve.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::FlipMode;
using lgpsf::corrections::LanczosOptions;
using lgpsf::corrections::ProbeArchive;
using lgpsf::corrections::ShiftedOperator;
using lgpsf::corrections::SolveMode;
using lgpsf::corrections::SolveOpts;
using lgpsf::corrections::cholesky_solve;
using lgpsf::corrections::extend_modes;
using lgpsf::corrections::make_cholesky_backend;
using lgpsf::corrections::make_pd;
using lgpsf::corrections::make_shifted_operator;
using lgpsf::corrections::solve;
using lgpsf::corrections::sparse_hr_oracle;
using lgpsf::corrections::sparse_op;
using lgpsf::corrections::sparse_part_pd;

namespace {

Eigen::SparseMatrix<double> spd_sparse( int n )
{
    std::vector<Eigen::Triplet<double>> entries;
    for ( int i = 0; i < n; ++i )
    {
        entries.emplace_back(i, i, 2.5 + 0.01 * i);
        if ( i + 1 < n )
        {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> Hr(n, n);
    Hr.setFromTriplets(entries.begin(), entries.end());
    return Hr;
}

} // namespace

TEST_CASE("the backend gate refuses matrices that are not the operator")
{
    std::mt19937 gen(0);
    const int n = 30;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(Hr);
    const Eigen::MatrixXd G = test_helpers::randn_points(n, n, gen);
    Eigen::MatrixXd Bd = 0.5 * (G + G.transpose());
    const Eigen::SparseMatrix<double> Bs = Bd.sparseView();

    const ShiftedOperator A = make_shifted_operator(
        sparse_op(Bs), ProbeArchive{}, sparse_hr_oracle(Hr), 1e-2);
    CHECK_NOTHROW(make_cholesky_backend(Bs, Hr, A));

    Eigen::MatrixXd wrong = Bd;
    wrong(3, 7) += 0.1;
    wrong(7, 3) += 0.1;
    CHECK_THROWS_AS(
        make_cholesky_backend(wrong.sparseView(), Hr, A),
        std::invalid_argument);
    CHECK_THROWS_AS(make_cholesky_backend(Bs, spd_sparse(n + 1), A),
                    std::invalid_argument);
}

TEST_CASE("three-way agreement, and the cache follows the block")
{
    std::mt19937 gen(1);
    const int n = 40;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(Hr);
    const Eigen::MatrixXd G = test_helpers::randn_points(n, n, gen);
    Eigen::MatrixXd Bd = 0.5 * (G + G.transpose());
    for ( int i = 0; i < n; ++i )
    {
        Bd(i, i) += 1.0;
    }
    const Eigen::SparseMatrix<double> Bs = Bd.sparseView();

    ShiftedOperator A = make_shifted_operator(
        sparse_op(Bs), ProbeArchive{}, sparse_hr_oracle(Hr), 1e-1);
    LanczosOptions lopts;
    lopts.max_iters = 300;
    REQUIRE(make_pd(A, 0.5, FlipMode::Flip, lopts).certified);
    const auto backend = make_cholesky_backend(Bs, Hr, A);

    const Eigen::MatrixXd B_rhs = test_helpers::randn_points(n, 2, gen);
    const double a = 0.2;

    // dense truth, iterative two-level, and the backend must all agree
    const Eigen::MatrixXd Ecorr =
        A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    const Eigen::MatrixXd dense_truth =
        (Bd + Ecorr + a * Hrd).ldlt().solve(Eigen::MatrixXd(B_rhs));

    SolveOpts opts;
    opts.mode = SolveMode::TwoLevel;
    opts.rtol = 1e-12;
    const Eigen::MatrixXd iterative = solve(A, B_rhs, a, opts).X;
    const Eigen::MatrixXd direct = cholesky_solve(backend, A, B_rhs, a);

    const double scale = dense_truth.cwiseAbs().maxCoeff();
    CHECK((direct - dense_truth).cwiseAbs().maxCoeff() < 1e-9 * scale);
    CHECK((iterative - dense_truth).cwiseAbs().maxCoeff() < 1e-8 * scale);

    // growing the block invalidates the cached factors automatically
    extend_modes(A, 3, 0.0, lopts);
    const Eigen::MatrixXd Ecorr2 =
        A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    const Eigen::MatrixXd dense2 =
        (Bd + Ecorr2 + a * Hrd).ldlt().solve(Eigen::MatrixXd(B_rhs));
    CHECK((cholesky_solve(backend, A, B_rhs, a) - dense2)
              .cwiseAbs()
              .maxCoeff()
          < 1e-9 * dense2.cwiseAbs().maxCoeff());
}

TEST_CASE("the inertia certificate flips at the raw pencil floor")
{
    std::mt19937 gen(2);
    const int n = 30;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(Hr);
    const Eigen::MatrixXd G = test_helpers::randn_points(n, n, gen);
    Eigen::MatrixXd Bd = 0.5 * (G + G.transpose());
    const Eigen::SparseMatrix<double> Bs = Bd.sparseView();

    const ShiftedOperator A = make_shifted_operator(
        sparse_op(Bs), ProbeArchive{}, sparse_hr_oracle(Hr), 1e-2);
    const auto backend = make_cholesky_backend(Bs, Hr, A);

    // dense truth for the raw pencil's leftmost eigenvalue
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigen(Bd, Hrd);
    const double raw_floor = -eigen.eigenvalues()(0);
    REQUIRE(raw_floor > 0.0);  // a random symmetric fit is indefinite

    CHECK(sparse_part_pd(backend, 1.01 * raw_floor));
    CHECK(!sparse_part_pd(backend, 0.99 * raw_floor));
}
