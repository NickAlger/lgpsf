// SPDX-License-Identifier: MIT
//
// Deflation on problems where the error H_d - B is KNOWN and exactly
// low-rank: free deflation must recover it exactly when the probes span it,
// the value pass must recover it exactly from true applies, and V2 must
// reach past the residual span where V1 saturates.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/deflation.hpp"
#include "lgpsf/corrections/pencil_lanczos.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::DeflateOptions;
using lgpsf::corrections::LanczosOptions;
using lgpsf::corrections::ProbeArchive;
using lgpsf::corrections::ShiftedOperator;
using lgpsf::corrections::ValuePassMode;
using lgpsf::corrections::deflate_free;
using lgpsf::corrections::dense_op;
using lgpsf::corrections::make_pd;
using lgpsf::corrections::make_shifted_operator;
using lgpsf::corrections::sparse_hr_oracle;
using lgpsf::corrections::value_pass;

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

/// H_d SPD with a decaying pencil spectrum, and B = H_d - E_true with
/// E_true of the given rank on mid-spectrum eigendirections -- so the
/// error is exactly low-rank and every recovery claim has dense truth.
struct DeflationProblem
{
    Eigen::MatrixXd Hd;
    Eigen::MatrixXd B;
    Eigen::MatrixXd Hrd;
    Eigen::SparseMatrix<double> Hr;
};

DeflationProblem known_error_problem( int n, int error_rank,
                                      std::mt19937& gen )
{
    DeflationProblem problem;
    problem.Hr = spd_sparse(n);
    problem.Hrd = Eigen::MatrixXd(problem.Hr);

    const Eigen::MatrixXd R = test_helpers::randn_points(n, n, gen);
    const Eigen::MatrixXd G = R.transpose() * problem.Hrd * R;
    const Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (G + G.transpose()));
    const Eigen::MatrixXd U =
        llt.matrixL().solve(R.transpose()).transpose();

    Eigen::VectorXd mu(n);
    for ( int i = 0; i < n; ++i )
    {
        mu(i) = 3.0 * std::pow(10.0, -3.0 * i / (n - 1));  // 3.0 .. 3e-3
    }
    Eigen::MatrixXd Hd =
        problem.Hrd * U * mu.asDiagonal() * U.transpose() * problem.Hrd;
    problem.Hd = 0.5 * (Hd + Hd.transpose());

    // the error lives on mid-spectrum directions, small enough that B's
    // pencil stays positive (no flips needed; make_pd just certifies)
    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(n, n);
    for ( int j = 0; j < error_rank; ++j )
    {
        const int at = 8 + 2 * j;
        const double sign = ( j % 3 == 2 ) ? -1.0 : 1.0;
        delta += sign * 0.4 * mu(at) * (problem.Hrd * U.col(at))
                 * (problem.Hrd * U.col(at)).transpose();
    }
    problem.B = problem.Hd - delta;
    return problem;
}

double relative_error( const ShiftedOperator& A,
                       const DeflationProblem& problem )
{
    const Eigen::MatrixXd P0 =
        problem.B + A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    return (problem.Hd - P0).norm() / problem.Hd.norm();
}

ShiftedOperator certified( const DeflationProblem& problem,
                           const Eigen::MatrixXd& Z )
{
    ProbeArchive archive;
    archive.Z = Z;
    archive.Y = problem.Hd * Z;
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.B), archive,
                              sparse_hr_oracle(problem.Hr), 1e-2);
    LanczosOptions opts;
    opts.max_iters = 300;
    REQUIRE(make_pd(A, 0.5, lgpsf::corrections::FlipMode::Flip, opts)
                .certified);
    return A;
}

} // namespace

TEST_CASE("free deflation recovers an exactly low-rank error from residuals")
{
    std::mt19937 gen(0);
    const int n = 40;
    const DeflationProblem problem = known_error_problem(n, 3, gen);
    ShiftedOperator A =
        certified(problem, test_helpers::randn_points(n, 12, gen));

    const double before = relative_error(A, problem);
    REQUIRE(before > 1e-3);  // there is something to recover

    DeflateOptions opts;
    opts.rcond = 1e-6;  // exact data: no interpolatory noise to regularize
    const auto report = deflate_free(A, opts);
    CHECK(report.residuals == 12);
    CHECK(report.basis == 3);   // the residual span IS the error span
    CHECK(report.kept == 3);
    CHECK(report.clamped == 0);
    CHECK(report.applies == 0);  // free means free

    // with the probes spanning the rank-3 error, recovery is exact to
    // solver tolerance
    CHECK(relative_error(A, problem) < 1e-6 * before);
}

TEST_CASE("the value pass recovers the same error from true applies")
{
    std::mt19937 gen(1);
    const int n = 40;
    const DeflationProblem problem = known_error_problem(n, 3, gen);
    ShiftedOperator A =
        certified(problem, test_helpers::randn_points(n, 12, gen));
    const double before = relative_error(A, problem);

    const auto report =
        value_pass(A, dense_op(problem.Hd), 3, ValuePassMode::V1);
    CHECK(report.applies == 3);
    CHECK(report.kept == 3);
    CHECK(relative_error(A, problem) < 1e-6 * before);

    // the pairs the pass paid for are archived as secant information
    CHECK(A.archive.Q_vp.cols() == 3);
    CHECK((A.archive.HdQ_vp - problem.Hd * A.archive.Q_vp)
              .cwiseAbs()
              .maxCoeff()
          < 1e-10 * problem.Hd.cwiseAbs().maxCoeff());
}

TEST_CASE("V2 reaches past the residual span where V1 saturates")
{
    // Six error modes but only three probes: the residual basis is
    // 3-dimensional, so V1 can never see the other half of the error. V2
    // spends half its budget pushing through the true operator first.
    std::mt19937 gen(2);
    const int n = 40;
    const DeflationProblem problem = known_error_problem(n, 6, gen);
    const Eigen::MatrixXd Z = test_helpers::randn_points(n, 3, gen);

    ShiftedOperator v1 = certified(problem, Z);
    const auto r1 = value_pass(v1, dense_op(problem.Hd), 6,
                               ValuePassMode::V1);
    CHECK(r1.applies == 3);  // saturated at the basis dimension

    ShiftedOperator v2 = certified(problem, Z);
    const auto r2 = value_pass(v2, dense_op(problem.Hd), 6,
                               ValuePassMode::V2);
    CHECK(r2.applies > r1.applies);
    CHECK(relative_error(v2, problem) < relative_error(v1, problem));
}

TEST_CASE("deflation refuses an uncertified struct and an empty archive")
{
    std::mt19937 gen(3);
    const int n = 30;
    const DeflationProblem problem = known_error_problem(n, 2, gen);

    ProbeArchive with_pairs;
    with_pairs.Z = test_helpers::randn_points(n, 4, gen);
    with_pairs.Y = problem.Hd * with_pairs.Z;
    ShiftedOperator uncertified =
        make_shifted_operator(dense_op(problem.B), with_pairs,
                              sparse_hr_oracle(problem.Hr), 1e-2);
    CHECK_THROWS_AS(deflate_free(uncertified), std::invalid_argument);

    ShiftedOperator empty =
        make_shifted_operator(dense_op(problem.B), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);
    LanczosOptions opts;
    opts.max_iters = 200;
    REQUIRE(make_pd(empty, 0.5, lgpsf::corrections::FlipMode::Flip, opts)
                .certified);
    CHECK_THROWS_AS(deflate_free(empty), std::invalid_argument);
    CHECK_THROWS_AS(
        value_pass(empty, dense_op(problem.Hd), 0, ValuePassMode::V1),
        std::invalid_argument);
}
