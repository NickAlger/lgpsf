// SPDX-License-Identifier: MIT
//
// Pencil Lanczos, the cache extension, and the a0-tied flip, on operators
// with a CONSTRUCTED pencil spectrum: B = Hr U diag(lambda) U^T Hr with
// U^T Hr U = I, so every assertion has exact truth to compare against.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/pencil_lanczos.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::FlipMode;
using lgpsf::corrections::LanczosOptions;
using lgpsf::corrections::ProbeArchive;
using lgpsf::corrections::ShiftedOperator;
using lgpsf::corrections::apply;
using lgpsf::corrections::correction_eigenvalues;
using lgpsf::corrections::dense_op;
using lgpsf::corrections::extend_modes;
using lgpsf::corrections::make_pd;
using lgpsf::corrections::make_shifted_operator;
using lgpsf::corrections::pencil_sweep;
using lgpsf::corrections::sparse_hr_oracle;
using lgpsf::corrections::surrogate_eigenvalues;

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
    Eigen::VectorXd lambda;  ///< the exact pencil spectrum, as given
};

/// B with the requested pencil spectrum: U = R chol(R^T Hr R)^{-T} is
/// H_r-orthonormal, and B = Hr U diag(lambda) U^T Hr.
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
        llt.matrixL().solve(R.transpose()).transpose();  // R L^{-T}

    Eigen::MatrixXd B = problem.Hrd * U * lambda.asDiagonal() * U.transpose()
                        * problem.Hrd;
    problem.Bd = 0.5 * (B + B.transpose());
    return problem;
}

/// The test spectrum: two flippable negatives, two tiny surviving negatives,
/// and a decaying positive tail (the physics-like shape).
Eigen::VectorXd standard_spectrum( int n )
{
    Eigen::VectorXd lambda(n);
    lambda(0) = -0.5;
    lambda(1) = -0.2;
    lambda(2) = -2e-6;
    lambda(3) = -1e-6;
    for ( int i = 4; i < n; ++i )
    {
        // 3.0 down to ~1e-3, log-spaced
        lambda(i) = 3.0 * std::pow(10.0, -3.5 * (i - 4) / (n - 5));
    }
    return lambda;
}

Eigen::VectorXd dense_pencil_spectrum( const Eigen::MatrixXd& Bd,
                                       const Eigen::MatrixXd& Hrd )
{
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigen(Bd, Hrd);
    return eigen.eigenvalues();
}

} // namespace

TEST_CASE("extend_modes caches true rightmost modes and leaves the operator alone")
{
    std::mt19937 gen(0);
    const int n = 40;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);

    const Eigen::MatrixXd X = test_helpers::randn_points(n, 2, gen);
    const Eigen::MatrixXd before = apply(A, X, 0.1);

    LanczosOptions opts;
    opts.max_iters = 120;
    const auto first = extend_modes(A, 5, 0.0, opts);
    CHECK(first.added == 5);
    CHECK(A.block.rank() == 5);

    // the cached surrogate content IS the top of the constructed spectrum
    Eigen::VectorXd sorted = problem.lambda;
    std::sort(sorted.data(), sorted.data() + sorted.size());
    const Eigen::VectorXd cached = surrogate_eigenvalues(A.block);
    for ( int k = 0; k < 5; ++k )
    {
        CHECK(cached(4 - k)
              == doctest::Approx(sorted(n - 1 - k)).epsilon(1e-8));
    }
    // and the next (uncached) value is reported -- the lambda_{k+1} that
    // governs two-level inner conditioning
    CHECK(first.next_value == doctest::Approx(sorted(n - 6)).epsilon(1e-6));

    // the operator itself is untouched by cache growth
    CHECK(apply(A, X, 0.1) == before);
    CHECK(correction_eigenvalues(A.block).cwiseAbs().maxCoeff() < 1e-12);

    // incremental: a second call deepens the cache instead of repeating it
    const auto second = extend_modes(A, 3, 0.0, opts);
    CHECK(second.added == 3);
    CHECK(A.block.rank() == 8);
    const Eigen::VectorXd deeper = surrogate_eigenvalues(A.block);
    CHECK(deeper(0) == doctest::Approx(sorted(n - 8)).epsilon(1e-8));

    // a value target stops the sweep down the spectrum
    ShiftedOperator A2 =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);
    const auto targeted = extend_modes(A2, 30, 1.0, opts);
    int expected = 0;
    for ( int i = 0; i < n; ++i )
    {
        if ( problem.lambda(i) > 1.0 )
        {
            ++expected;
        }
    }
    CHECK(targeted.added == expected);
}

TEST_CASE("make_pd flips exactly the sub-threshold modes and certifies the floor")
{
    std::mt19937 gen(1);
    const int n = 40;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);

    LanczosOptions opts;
    opts.max_iters = 400;
    const auto report = make_pd(A, 0.5, FlipMode::Flip, opts);

    // threshold is -gamma a0 = -5e-3: exactly {-0.5, -0.2} are below it
    CHECK(report.flipped == 2);
    CHECK(report.certified);
    CHECK(report.leftmost_before == doctest::Approx(-0.5).epsilon(1e-7));
    // the floor is the leftmost SURVIVOR: -2e-6 (the noise tail stays)
    CHECK(report.lambda_floor == doctest::Approx(-2e-6).epsilon(1e-3));
    CHECK(A.lambda_floor.has_value());
    CHECK(A.gamma == 0.5);

    // dense truth: the corrected operator's pencil spectrum is the original
    // with -0.5 -> +0.5 and -0.2 -> +0.2
    const Eigen::MatrixXd P0d =
        problem.Bd
        + A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    const Eigen::VectorXd corrected =
        dense_pencil_spectrum(P0d, problem.Hrd);
    Eigen::VectorXd expected = problem.lambda;
    expected(0) = 0.5;
    expected(1) = 0.2;
    std::sort(expected.data(), expected.data() + expected.size());
    CHECK((corrected - expected).cwiseAbs().maxCoeff() < 1e-7);

    // the exact contract: B_pd + a Hr > 0 iff a > -lambda_floor
    const auto min_eig = [&]( double a ) {
        return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(
                   P0d + a * problem.Hrd)
            .eigenvalues()
            .minCoeff();
    };
    CHECK(min_eig(2.0 * -report.lambda_floor) > 0.0);
    CHECK(min_eig(0.5 * -report.lambda_floor) < 0.0);

    // and the GLR surrogate presents the flipped modes at their CORRECTED
    // values (+0.5, +0.2), not their raw or doubled ones
    const Eigen::VectorXd surr = surrogate_eigenvalues(A.block);
    CHECK(surr.maxCoeff() == doctest::Approx(0.5).epsilon(1e-7));
    CHECK(surr.minCoeff() == doctest::Approx(0.2).epsilon(1e-7));
}

TEST_CASE("relu leaves corrected modes at zero instead of reflecting them")
{
    std::mt19937 gen(2);
    const int n = 30;
    Eigen::VectorXd lambda = standard_spectrum(n);
    const PencilProblem problem = with_spectrum(lambda, gen);
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);

    LanczosOptions opts;
    opts.max_iters = 300;
    const auto report = make_pd(A, 0.5, FlipMode::Relu, opts);
    CHECK(report.flipped == 2);
    CHECK(report.certified);

    const Eigen::MatrixXd P0d =
        problem.Bd
        + A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    const Eigen::VectorXd corrected =
        dense_pencil_spectrum(P0d, problem.Hrd);
    // the two former negatives now sit at ~0, between the tiny survivors
    // and the positive tail
    CHECK(std::abs(corrected(2)) < 1e-7);
    CHECK(std::abs(corrected(3)) < 1e-7);
}

TEST_CASE("make_pd on an exhausted budget reports uncertified and resumes")
{
    std::mt19937 gen(3);
    const int n = 40;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);

    // enough budget to complete the flip round (the whole 40-dim sweep) but
    // not the certification sweep that follows it
    LanczosOptions partial;
    partial.max_iters = 45;
    auto report = make_pd(A, 0.5, FlipMode::Flip, partial);
    CHECK(!report.certified);
    CHECK(!A.lambda_floor.has_value());
    CHECK(report.flipped == 2);  // progress happened and persists

    // a second call continues from the block instead of starting over
    int total_flipped = report.flipped;
    for ( int round = 0; round < 10 && !report.certified; ++round )
    {
        partial.seed = static_cast<unsigned>(100 + round);
        report = make_pd(A, 0.5, FlipMode::Flip, partial);
        total_flipped += report.flipped;
    }
    CHECK(report.certified);
    CHECK(total_flipped == 2);  // nothing was flipped twice
    CHECK(report.lambda_floor == doctest::Approx(-2e-6).epsilon(1e-3));

    CHECK_THROWS_AS(make_pd(A, 1.5, FlipMode::Flip, partial),
                    std::invalid_argument);
}
