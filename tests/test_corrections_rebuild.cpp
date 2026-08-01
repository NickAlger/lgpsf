// SPDX-License-Identifier: MIT
//
// rebuild_at: re-anchoring the contracts at a smaller shift touches no new
// H_d information -- new flips come from the newly opened window, deflation
// re-derives from archived residuals, and value pairs fold back in exactly
// by linearity.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/rebuild.hpp"
#include "lgpsf/corrections/solve.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::FlipMode;
using lgpsf::corrections::LanczosOptions;
using lgpsf::corrections::ProbeArchive;
using lgpsf::corrections::RebuildOptions;
using lgpsf::corrections::ShiftedOperator;
using lgpsf::corrections::ValuePassMode;
using lgpsf::corrections::Zone;
using lgpsf::corrections::classify_shift;
using lgpsf::corrections::dense_op;
using lgpsf::corrections::make_pd;
using lgpsf::corrections::make_shifted_operator;
using lgpsf::corrections::rebuild_at;
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

struct PencilProblem
{
    Eigen::MatrixXd Bd;
    Eigen::MatrixXd Hrd;
    Eigen::SparseMatrix<double> Hr;
    Eigen::VectorXd lambda;
};

PencilProblem with_spectrum( const Eigen::VectorXd& lambda,
                             std::mt19937& gen )
{
    const int n = static_cast<int>(lambda.size());
    PencilProblem problem;
    problem.Hr = spd_sparse(n);
    problem.Hrd = Eigen::MatrixXd(problem.Hr);
    problem.lambda = lambda;
    const Eigen::MatrixXd R = test_helpers::randn_points(n, n, gen);
    const Eigen::MatrixXd G = R.transpose() * problem.Hrd * R;
    const Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (G + G.transpose()));
    const Eigen::MatrixXd U =
        llt.matrixL().solve(R.transpose()).transpose();
    Eigen::MatrixXd B = problem.Hrd * U * lambda.asDiagonal() * U.transpose()
                        * problem.Hrd;
    problem.Bd = 0.5 * (B + B.transpose());
    return problem;
}

} // namespace

TEST_CASE("rebuild_at flips the newly opened window and renews the contracts")
{
    // a mode at -1e-3 survives the a0 = 1e-2 build (threshold -5e-3) but is
    // below the a1 = 1e-3 threshold (-5e-4): rebuild must flip exactly it
    std::mt19937 gen(0);
    const int n = 40;
    Eigen::VectorXd lambda(n);
    lambda(0) = -0.5;
    lambda(1) = -1e-3;
    lambda(2) = -2e-6;
    for ( int i = 3; i < n; ++i )
    {
        lambda(i) = 3.0 * std::pow(10.0, -3.5 * (i - 3) / (n - 4));
    }
    const PencilProblem problem = with_spectrum(lambda, gen);
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);
    LanczosOptions lopts;
    lopts.max_iters = 400;
    const auto first = make_pd(A, 0.5, FlipMode::Flip, lopts);
    REQUIRE(first.certified);
    CHECK(first.flipped == 1);  // only -0.5 is below -5e-3
    CHECK(*A.lambda_floor == doctest::Approx(-1e-3).epsilon(1e-6));
    // a = 1e-3 sits exactly ON the floor (refused: the contract is strict);
    // strictly inside the window it is warned
    CHECK(classify_shift(A, 1e-3).zone == Zone::Refused);
    CHECK(classify_shift(A, 2e-3).zone == Zone::Warned);

    RebuildOptions ropts;
    ropts.lanczos = lopts;
    const auto rebuilt = rebuild_at(A, 1e-3, ropts);
    REQUIRE(rebuilt.flip.certified);
    CHECK(rebuilt.flip.flipped == 1);  // exactly the -1e-3 mode
    CHECK(*A.lambda_floor == doctest::Approx(-2e-6).epsilon(1e-3));
    CHECK(A.a0 == 1e-3);
    CHECK(classify_shift(A, 1e-3).zone == Zone::Guaranteed);

    // dense truth: the corrected pencil spectrum has both flips reflected
    const Eigen::MatrixXd P0d =
        problem.Bd + A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
        P0d, problem.Hrd);
    Eigen::VectorXd expected = lambda;
    expected(0) = 0.5;
    expected(1) = 1e-3;
    std::sort(expected.data(), expected.data() + expected.size());
    CHECK((eigen.eigenvalues() - expected).cwiseAbs().maxCoeff() < 1e-7);
}

TEST_CASE("archived value pairs fold back in with zero new applies")
{
    // plant a rank-3 error, pay for it once with a value pass at a0, then
    // rebuild at a smaller shift: the correction must survive the move
    // without touching H_d again
    std::mt19937 gen(1);
    const int n = 40;
    Eigen::VectorXd lambda(n);
    for ( int i = 0; i < n; ++i )
    {
        lambda(i) = 3.0 * std::pow(10.0, -3.0 * i / (n - 1));
    }
    const PencilProblem problem = with_spectrum(lambda, gen);

    Eigen::MatrixXd U = test_helpers::randn_points(n, 3, gen);
    for ( int j = 0; j < 3; ++j )
    {
        U.col(j) /= std::sqrt(U.col(j).dot(problem.Hrd * U.col(j)));
    }
    Eigen::MatrixXd values = Eigen::MatrixXd::Zero(3, 3);
    values.diagonal() << 0.3, 0.2, 0.1;
    const Eigen::MatrixXd delta =
        (problem.Hrd * U) * values * (problem.Hrd * U).transpose();
    const Eigen::MatrixXd Hd = problem.Bd + delta;

    ProbeArchive archive;
    archive.Z = test_helpers::randn_points(n, 8, gen);
    archive.Y = Hd * archive.Z;
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.Bd), archive,
                              sparse_hr_oracle(problem.Hr), 1e-2);
    LanczosOptions lopts;
    lopts.max_iters = 300;
    REQUIRE(make_pd(A, 0.5, FlipMode::Flip, lopts).certified);
    value_pass(A, dense_op(Hd), 3, ValuePassMode::V1);

    const auto err = [&]( const ShiftedOperator& S ) {
        const Eigen::MatrixXd P0 =
            problem.Bd
            + S.block.HrV * S.block.C_corr * S.block.HrV.transpose();
        return (Hd - P0).norm() / Hd.norm();
    };
    const double corrected = err(A);
    REQUIRE(corrected < 1e-6);
    const Eigen::Index pairs_before = A.archive.Q_vp.cols();

    RebuildOptions ropts;
    ropts.lanczos = lopts;
    const auto rebuilt = rebuild_at(A, 1e-3, ropts);
    REQUIRE(rebuilt.flip.certified);
    CHECK(rebuilt.refolded);
    CHECK(rebuilt.value_fold.applies == 0);        // the whole point
    CHECK(A.archive.Q_vp.cols() == pairs_before);  // folding is not paying
    CHECK(err(A) < 1e-5);                          // the correction survived
    CHECK(A.a0 == 1e-3);
    CHECK(classify_shift(A, 1e-3).zone == Zone::Guaranteed);

    CHECK_THROWS_AS(rebuild_at(A, 0.0), std::invalid_argument);
}
