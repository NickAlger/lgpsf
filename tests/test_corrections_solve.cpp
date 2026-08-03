// SPDX-License-Identifier: MIT
//
// Zone semantics and the two solve paths. The zones are contracts about
// definiteness at a caller-supplied shift; the tests pin each boundary and
// each mechanism against dense truth on constructed-spectrum problems.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/pencil_lanczos.hpp"
#include "lgpsf/corrections/solve.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::FlipMode;
using lgpsf::corrections::LanczosOptions;
using lgpsf::corrections::ProbeArchive;
using lgpsf::corrections::Provenance;
using lgpsf::corrections::ShiftedOperator;
using lgpsf::corrections::SolveMode;
using lgpsf::corrections::SolveOpts;
using lgpsf::corrections::Zone;
using lgpsf::corrections::classify_shift;
using lgpsf::corrections::dense_op;
using lgpsf::corrections::extend_modes;
using lgpsf::corrections::make_pd;
using lgpsf::corrections::make_shifted_operator;
using lgpsf::corrections::merge;
using lgpsf::corrections::solve;
using lgpsf::corrections::sparse_hr_oracle;

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

Eigen::VectorXd standard_spectrum( int n )
{
    Eigen::VectorXd lambda(n);
    lambda(0) = -0.5;
    lambda(1) = -0.2;
    lambda(2) = -2e-6;
    lambda(3) = -1e-6;
    for ( int i = 4; i < n; ++i )
    {
        lambda(i) = 3.0 * std::pow(10.0, -3.5 * (i - 4) / (n - 5));
    }
    return lambda;
}

/// A certified struct over the standard spectrum: a0 = 1e-2, flip done,
/// floor at -2e-6.
ShiftedOperator certified_example( const PencilProblem& problem )
{
    ShiftedOperator A =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);
    LanczosOptions opts;
    opts.max_iters = 400;
    const auto report = make_pd(A, 0.5, FlipMode::Flip, opts);
    REQUIRE(report.certified);
    return A;
}

} // namespace

TEST_CASE("the three zones sit exactly where the contracts say")
{
    std::mt19937 gen(0);
    const int n = 40;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);

    // uncertified: warned everywhere, nothing claimed
    ShiftedOperator raw =
        make_shifted_operator(dense_op(problem.Bd), ProbeArchive{},
                              sparse_hr_oracle(problem.Hr), 1e-2);
    CHECK(classify_shift(raw, 1.0).zone == Zone::Warned);
    CHECK(!classify_shift(raw, 1.0).analytic_pd);

    ShiftedOperator A = certified_example(problem);
    const double floor = -*A.lambda_floor;  // ~2e-6

    CHECK(classify_shift(A, 1e-2).zone == Zone::Guaranteed);   // a = a0
    CHECK(classify_shift(A, 5.0).zone == Zone::Guaranteed);
    const auto warned = classify_shift(A, 1e-4);               // in between
    CHECK(warned.zone == Zone::Warned);
    CHECK(warned.analytic_pd);  // flip certificate, no later corrections
    CHECK(classify_shift(A, 0.5 * floor).zone == Zone::Refused);
    CHECK(classify_shift(A, 0.0).zone == Zone::Refused);
    CHECK(classify_shift(A, -1.0).zone == Zone::Refused);

    // refused shifts throw from solve, with the floor in the message
    CHECK_THROWS_AS(solve(A, Eigen::MatrixXd::Ones(n, 1), 0.5 * floor),
                    std::domain_error);
}

TEST_CASE("post-certification corrections move the analytic boundary")
{
    std::mt19937 gen(1);
    const int n = 40;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A = certified_example(problem);

    // cache growth is NOT a correction: the certificate must not degrade
    LanczosOptions opts;
    opts.max_iters = 100;
    extend_modes(A, 4, 0.0, opts);
    CHECK(classify_shift(A, 1e-4).analytic_pd);
    CHECK(classify_shift(A, 1e-4).post_cert_min == 0.0);

    // a correction with eig_min = -d shifts the certified region to
    // a > d - lambda_floor, exactly. Coefficients are stated on the RAW
    // candidate directions, so Hr-normalize the direction to make -d the
    // pencil value itself.
    const double d = 1e-3;
    Eigen::VectorXd v = test_helpers::randn_points(n, 1, gen).col(0);
    v /= std::sqrt(v.dot(problem.Hrd * v));
    Eigen::MatrixXd Cc = Eigen::MatrixXd::Zero(1, 1);
    Cc(0, 0) = -d;
    merge(A.block, A.hr, Eigen::MatrixXd(v), Cc, Cc,
          Provenance::Deflation);
    const auto above = classify_shift(A, 2e-3);   // 2e-3 + (-2e-6) + (-1e-3) > 0
    const auto below = classify_shift(A, 5e-4);   // 5e-4 + (-2e-6) + (-1e-3) < 0
    CHECK(above.zone == Zone::Warned);
    CHECK(above.analytic_pd);
    CHECK(above.post_cert_min == doctest::Approx(-d).epsilon(1e-9));
    CHECK(below.zone == Zone::Warned);
    CHECK(!below.analytic_pd);

    // the guaranteed zone is untouched by the warning-zone bookkeeping
    CHECK(classify_shift(A, 1e-2).zone == Zone::Guaranteed);
}

TEST_CASE("GLR-mode solve reports the residual of the system it solves")
{
    std::mt19937 gen(2);
    const int n = 35;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A = certified_example(problem);
    LanczosOptions opts;
    opts.max_iters = 100;
    extend_modes(A, 5, 0.0, opts);

    const Eigen::MatrixXd B = test_helpers::randn_points(n, 3, gen);
    const auto result = solve(A, B, 0.3);
    CHECK(result.zone.zone == Zone::Guaranteed);
    CHECK(result.iterations == 0);
    CHECK(result.relative_residual < 1e-9);  // Woodbury is exact to oracle
}

TEST_CASE("two-level solve matches dense truth in both live zones")
{
    std::mt19937 gen(3);
    const int n = 40;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A = certified_example(problem);
    LanczosOptions lopts;
    lopts.max_iters = 150;
    extend_modes(A, 8, 0.0, lopts);

    const Eigen::MatrixXd Ecorr =
        A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    const Eigen::MatrixXd B = test_helpers::randn_points(n, 2, gen);

    SolveOpts opts;
    opts.mode = SolveMode::TwoLevel;
    opts.rtol = 1e-10;
    for ( double a : {2e-2 /* guaranteed */, 1e-4 /* warned */} )
    {
        const auto result = solve(A, B, a, opts);
        CHECK(result.relative_residual <= opts.rtol);
        const Eigen::MatrixXd truth =
            (problem.Bd + Ecorr + a * problem.Hrd)
                .ldlt()
                .solve(Eigen::MatrixXd(B));
        CHECK((result.X - truth).cwiseAbs().maxCoeff()
              < 1e-7 * truth.cwiseAbs().maxCoeff());
    }
}

TEST_CASE("the cached modes are what make the inner iteration fast")
{
    // Inner conditioning is 1 + lambda_{k+1}/a: with the top of the spectrum
    // cached, PCG needs visibly fewer iterations than with an empty cache.
    std::mt19937 gen(4);
    const int n = 40;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    const double a = 2e-2;
    const Eigen::MatrixXd b = test_helpers::randn_points(n, 1, gen);

    SolveOpts opts;
    opts.mode = SolveMode::TwoLevel;
    opts.rtol = 1e-8;

    ShiftedOperator bare = certified_example(problem);
    const auto slow = solve(bare, b, a, opts);

    ShiftedOperator cached = certified_example(problem);
    LanczosOptions lopts;
    lopts.max_iters = 150;
    extend_modes(cached, 8, 0.0, lopts);
    const auto fast = solve(cached, b, a, opts);

    CHECK(fast.iterations < slow.iterations);
    CHECK(fast.relative_residual <= opts.rtol);
}

TEST_CASE("an indefinite operator reveals itself in the inner iteration")
{
    std::mt19937 gen(5);
    const int n = 30;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A = certified_example(problem);

    // an (unclamped, surrogate-invisible) correction that makes P(a)
    // genuinely indefinite below a0
    Eigen::MatrixXd Cc = Eigen::MatrixXd::Zero(1, 1);
    Cc(0, 0) = -5.0;
    merge(A.block, A.hr, test_helpers::randn_points(n, 1, gen), Cc,
          Eigen::MatrixXd::Zero(1, 1), Provenance::Deflation);

    const auto zone = classify_shift(A, 5e-3);
    CHECK(zone.zone == Zone::Warned);
    CHECK(!zone.analytic_pd);  // the certificate correctly refuses to vouch

    SolveOpts opts;
    opts.mode = SolveMode::TwoLevel;
    CHECK_THROWS_AS(solve(A, Eigen::MatrixXd::Ones(n, 1), 5e-3, opts),
                    std::runtime_error);
}

TEST_CASE("glr_precondition serves where M(a) is indefinite")
{
    std::mt19937 gen(6);
    const int n = 30;
    const PencilProblem problem = with_spectrum(standard_spectrum(n), gen);
    ShiftedOperator A = certified_example(problem);

    // a trustworthy NEGATIVE error mode, as a deflation would store it
    Eigen::VectorXd v = test_helpers::randn_points(n, 1, gen).col(0);
    v /= std::sqrt(v.dot(problem.Hrd * v));
    Eigen::MatrixXd Cc = Eigen::MatrixXd::Zero(1, 1);
    Cc(0, 0) = -2e-3;
    merge(A.block, A.hr, Eigen::MatrixXd(v), Cc, Cc,
          lgpsf::corrections::Provenance::ValuePass);

    // below the surrogate floor: the exact solve refuses, the
    // preconditioner does not, and the two coincide when the shift is
    // above the floor... first, the refusal boundary is the surrogate's:
    // the merge folds part of the direction onto the flip columns, so the
    // surrogate's min eigenvalue lands NEAR -2e-3 rather than exactly on it
    const double glr_floor = lgpsf::corrections::glr_pd_floor(A);
    CHECK(glr_floor > 1e-3);
    CHECK(glr_floor <= 2e-3 * (1.0 + 1e-9));
    const Eigen::MatrixXd x = Eigen::MatrixXd::Ones(n, 1);
    CHECK_THROWS_AS(lgpsf::corrections::glr_solve(A, x, 0.5 * glr_floor),
                    std::domain_error);
    CHECK(lgpsf::corrections::glr_precondition(A, x, 0.5 * glr_floor)
              .allFinite());

    // the |theta| operator really is applied: against dense truth
    const double a = 0.5 * glr_floor;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(A.block.C_surr);
    const Eigen::MatrixXd S_abs =
        (A.block.HrV * eig.eigenvectors())
        * eig.eigenvalues().cwiseAbs().asDiagonal()
        * (A.block.HrV * eig.eigenvectors()).transpose();
    const Eigen::MatrixXd expected =
        (a * problem.Hrd + S_abs).ldlt().solve(Eigen::MatrixXd(x));
    CHECK((lgpsf::corrections::glr_precondition(A, x, a) - expected)
              .cwiseAbs()
              .maxCoeff()
          < 1e-9 * expected.cwiseAbs().maxCoeff());

    // and the two-level solve now converges below the surrogate floor
    // (the OPERATOR is still PD there: the correction is small)
    SolveOpts opts;
    opts.mode = SolveMode::TwoLevel;
    opts.rtol = 1e-10;
    const Eigen::MatrixXd b = test_helpers::randn_points(n, 1, gen);
    const auto result = solve(A, b, a, opts);
    const Eigen::MatrixXd Ecorr =
        A.block.HrV * A.block.C_corr * A.block.HrV.transpose();
    const Eigen::MatrixXd truth =
        (problem.Bd + Ecorr + a * problem.Hrd).ldlt().solve(
            Eigen::MatrixXd(b));
    CHECK((result.X - truth).cwiseAbs().maxCoeff()
          < 1e-7 * truth.cwiseAbs().maxCoeff());

    // when the surrogate is PSD, precondition == solve exactly
    ShiftedOperator clean = certified_example(problem);
    LanczosOptions lopts;
    lopts.max_iters = 100;
    extend_modes(clean, 4, 0.0, lopts);
    const Eigen::MatrixXd via_solve =
        lgpsf::corrections::glr_solve(clean, x, 0.7);
    const Eigen::MatrixXd via_precondition =
        lgpsf::corrections::glr_precondition(clean, x, 0.7);
    CHECK(via_solve == via_precondition);
}
