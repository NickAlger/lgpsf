// SPDX-License-Identifier: MIT
//
// Checks on the VarPro fitting core: the inner linear algebra against
// brute-force references, the Frisch-Waugh-Lovell identity that licenses
// projecting the extra block out once, the reduced-residual Jacobian against
// finite differences, the structural relationship between the two Jacobian
// variants, and an end-to-end recovery of a known answer.
//
// All self-contained: nothing here is compared against a stored reference, so
// the suite cannot drift out of step with the code it tests.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/varpro.hpp"
#include "lgpsf/whitening.hpp"
#include "test_helpers.hpp"

using lgpsf::InfeasibleParameters;
using lgpsf::JacobianVariant;
using lgpsf::Mode;
using lgpsf::MuMode;
using lgpsf::VarProOptions;
using lgpsf::VarProResult;
using lgpsf::WhitenedBasis;
using lgpsf::WhitenedBasisAt;
using lgpsf::detail::InnerFactors;
using lgpsf::detail::InnerSolve;
using lgpsf::detail::ReducedProblem;
using lgpsf::detail::inner_solve;
using lgpsf::detail::orthonormal_range;
using lgpsf::detail::project_out;
using lgpsf::fit_varpro;
using lgpsf::num_harmonics;
using lgpsf::theta_hat_size;
using lgpsf::whiten_extra;

namespace {

Eigen::VectorXd random_vector( int n, std::mt19937& gen, double lo = -1.0,
                               double hi = 1.0 )
{
    std::uniform_real_distribution<double> dist(lo, hi);
    Eigen::VectorXd v(n);
    for ( int i = 0; i < n; ++i )
    {
        v(i) = dist(gen);
    }
    return v;
}

std::vector<Mode> some_modes( int dim, int max_ell = 2, int max_p = 1 )
{
    std::vector<Mode> modes;
    for ( int ell = 0; ell <= max_ell; ++ell )
    {
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            for ( int p = 0; p <= max_p; ++p )
            {
                modes.push_back(Mode{p, ell, m});
            }
        }
    }
    return modes;
}

/// One synthetic target, wired exactly the way a caller wires it: whitened
/// probes and extra basis as arrays, the whitened smooth basis as the functor.
struct Problem
{
    Eigen::VectorXd theta_hat;
    Eigen::VectorXd mu0;
    MuMode mu_mode = MuMode::Fitted;
    int num_params = 0;
    Eigen::MatrixXd z_hat;  // (K, k)
    Eigen::MatrixXd e_hat;  // (K, num_extra)
    std::vector<Mode> modes;
    WhitenedBasis basis;
    Eigen::Index num_probes = 0;
};

Problem make_problem( std::mt19937& gen, int dim = 2,
                      MuMode mu_mode = MuMode::Fitted, int num_extra = 1,
                      int num_probes = 24, int num_points = 16 )
{
    const std::vector<Mode> modes = some_modes(dim);
    const Eigen::VectorXd mu0 = random_vector(dim, gen);
    const Eigen::VectorXd theta_hat =
        random_vector(theta_hat_size(dim, mu_mode), gen, -0.3, 0.3);
    const double target_mass = 1.3;
    const Eigen::MatrixXd x =
        test_helpers::uniform_points(num_points, dim, gen, -1.5, 1.5);
    const Eigen::VectorXd m2 =
        test_helpers::uniform_points(num_points, 1, gen, 0.5, 2.0).col(0);

    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(num_points, num_extra);
    for ( int d = 0; d < num_extra; ++d )
    {
        E(d, d) = 1.0;  // one-hot spikes at the first few batch points
    }

    return Problem{theta_hat,
                   mu0,
                   mu_mode,
                   theta_hat_size(dim, mu_mode),
                   test_helpers::randn_points(num_points, num_probes, gen),
                   whiten_extra(E, target_mass, m2),
                   modes,
                   WhitenedBasis(x, target_mass, m2, modes, mu0, mu_mode),
                   num_probes};
}

/// A basis evaluation exposing only values() and vjp() -- the minimal contract
/// the default (Kaufman) path is allowed to need.
class ValuesAndVjpOnly
{
public:
    explicit ValuesAndVjpOnly( WhitenedBasisAt at ) : at_(std::move(at)) {}
    const Eigen::MatrixXd& values() { return at_.values(); }
    Eigen::MatrixXd vjp( const Eigen::Ref<const Eigen::MatrixXd>& w )
    {
        return at_.vjp(w);
    }

private:
    WhitenedBasisAt at_;
};

struct BasisWithoutJacobian
{
    const WhitenedBasis* inner = nullptr;
    ValuesAndVjpOnly operator()( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        return ValuesAndVjpOnly((*inner)(p));
    }
};

struct CountingBasis
{
    const WhitenedBasis* inner = nullptr;
    int* count = nullptr;
    WhitenedBasisAt operator()( const Eigen::Ref<const Eigen::VectorXd>& p ) const
    {
        ++(*count);
        return (*inner)(p);
    }
};

} // namespace

TEST_CASE("projecting out an orthonormal block is an idempotent projector")
{
    std::mt19937 gen(0);
    const Eigen::MatrixXd raw = test_helpers::randn_points(12, 3, gen);
    const Eigen::MatrixXd Q = orthonormal_range(raw);
    const Eigen::MatrixXd V = test_helpers::randn_points(12, 4, gen);

    const Eigen::MatrixXd once = project_out(Q, V);
    CHECK((project_out(Q, Eigen::MatrixXd(once)) - once).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((Q.transpose() * once).cwiseAbs().maxCoeff() < 1e-12);

    // no block at all is the identity
    const Eigen::MatrixXd empty = Eigen::MatrixXd::Zero(12, 0);
    CHECK(project_out(empty, V) == V);
}

TEST_CASE("the range basis collapses dependent columns instead of spanning them")
{
    std::mt19937 gen(1);
    Eigen::MatrixXd B(10, 3);
    B.col(0) = test_helpers::randn_points(10, 1, gen).col(0);
    B.col(1) = test_helpers::randn_points(10, 1, gen).col(0);
    B.col(2) = B.col(0);  // a caller passing the same extra function twice

    const Eigen::MatrixXd Q = orthonormal_range(B);
    CHECK(Q.cols() == 2);
    CHECK((Q.transpose() * Q - Eigen::MatrixXd::Identity(2, 2))
              .cwiseAbs().maxCoeff() < 1e-12);
    // and it still spans everything B spans
    CHECK(project_out(Q, B).cwiseAbs().maxCoeff() < 1e-12);

    CHECK(orthonormal_range(Eigen::MatrixXd::Zero(10, 0)).cols() == 0);
    CHECK(orthonormal_range(Eigen::MatrixXd::Zero(10, 2)).cols() == 0);
}

TEST_CASE("the inner solve reproduces least squares, and its ridge the damped system")
{
    std::mt19937 gen(2);
    for ( int num_modes : {3, 6} )
    {
        const Eigen::MatrixXd A = test_helpers::randn_points(20, num_modes, gen);
        const Eigen::VectorXd y = test_helpers::randn_points(20, 1, gen).col(0);

        const InnerSolve plain = inner_solve(A, y, 0.0);
        const Eigen::VectorXd reference =
            A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
        CHECK((plain.c - reference).cwiseAbs().maxCoeff() < 1e-9);
        CHECK((plain.residual - (y - A * reference)).cwiseAbs().maxCoeff() < 1e-9);
        // U is an orthonormal basis of range(A), which the Jacobian's projector needs
        CHECK(plain.U.cols() == num_modes);
        CHECK((plain.U.transpose() * plain.U
               - Eigen::MatrixXd::Identity(num_modes, num_modes))
                  .cwiseAbs().maxCoeff() < 1e-12);

        // The ridge acts on the EQUILIBRATED matrix, so the independent
        // formula is the damped normal equations in the scaled variables.
        const double ridge = 1e-3;
        const InnerSolve damped = inner_solve(A, y, ridge);
        Eigen::MatrixXd scaled = A;
        for ( Eigen::Index j = 0; j < num_modes; ++j )
        {
            scaled.col(j) /= damped.col_scale(j);
        }
        const Eigen::MatrixXd normal =
            scaled.transpose() * scaled
            + ridge * Eigen::MatrixXd::Identity(num_modes, num_modes);
        const Eigen::VectorXd expected_scaled =
            normal.ldlt().solve(scaled.transpose() * y);
        const Eigen::VectorXd expected =
            (expected_scaled.array() / damped.col_scale.array()).matrix();
        CHECK((damped.c - expected).cwiseAbs().maxCoeff() < 1e-9);
    }
}

TEST_CASE("the inner solve detects rank deficiency")
{
    std::mt19937 gen(3);
    Eigen::MatrixXd A = test_helpers::randn_points(15, 4, gen);
    A.col(3) = A.col(1);  // a feature the probes cannot distinguish

    const InnerSolve solution = inner_solve(A, test_helpers::randn_points(15, 1, gen).col(0),
                                            0.0);
    CHECK(solution.U.cols() == 3);
    CHECK(solution.sigma.size() == 3);
    CHECK(solution.c.allFinite());
}

TEST_CASE("RangeOnly takes the QR and answers the same as the SVD")
{
    // The fast path is not an approximation: the SVD's sigma/(sigma^2 + ridge)
    // filter IS Tikhonov, so a QR reaches the same coefficients. What legitimately
    // differs is U -- a different orthonormal basis of the SAME subspace -- which
    // is why the check is on the PROJECTOR, the only way U is ever used.
    std::mt19937 gen(31);
    for ( int num_modes : {3, 6, 12} )
    {
        for ( double ridge : {0.0, 1e-8, 1e-3} )
        {
            const Eigen::MatrixXd A = test_helpers::randn_points(30, num_modes, gen);
            const Eigen::VectorXd y = test_helpers::randn_points(30, 1, gen).col(0);

            const InnerSolve full = inner_solve(A, y, ridge, InnerFactors::Full);
            const InnerSolve range = inner_solve(A, y, ridge, InnerFactors::RangeOnly);

            CHECK((range.c - full.c).cwiseAbs().maxCoeff() < 1e-10);
            CHECK((range.residual - full.residual).cwiseAbs().maxCoeff() < 1e-10);
            CHECK((range.col_scale - full.col_scale).cwiseAbs().maxCoeff() < 1e-12);

            // the QR path really did skip the SVD
            CHECK(range.sigma.size() == 0);
            CHECK(full.sigma.size() == num_modes);

            // U orthonormal, and the same projector -- what the Jacobian uses
            CHECK(range.U.cols() == num_modes);
            CHECK((range.U.transpose() * range.U
                   - Eigen::MatrixXd::Identity(num_modes, num_modes))
                      .cwiseAbs().maxCoeff() < 1e-12);
            const Eigen::MatrixXd probe = test_helpers::randn_points(30, 4, gen);
            CHECK((project_out(range.U, probe) - project_out(full.U, probe))
                      .cwiseAbs().maxCoeff() < 1e-10);
        }
    }
}

TEST_CASE("RangeOnly falls back to the SVD when the QR reports rank deficiency")
{
    // The one case where QR and SVD genuinely disagree: a pivoted QR returns a
    // basic solution, the SVD the minimum-norm one, and the minimum-norm one is
    // the point of having a rank cutoff. So the fast path must decline.
    std::mt19937 gen(32);
    Eigen::MatrixXd A = test_helpers::randn_points(15, 4, gen);
    A.col(3) = A.col(1);
    const Eigen::VectorXd y = test_helpers::randn_points(15, 1, gen).col(0);

    const InnerSolve range = inner_solve(A, y, 0.0, InnerFactors::RangeOnly);
    const InnerSolve full = inner_solve(A, y, 0.0, InnerFactors::Full);

    // sigma present at all is the evidence the fallback fired
    CHECK(range.sigma.size() == 3);
    CHECK(range.U.cols() == 3);
    CHECK((range.c - full.c).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((range.residual - full.residual).cwiseAbs().maxCoeff() < 1e-12);

    // and the minimum-norm property is what a basic solution would have lost
    CHECK(std::abs(range.c(1) - range.c(3)) < 1e-9);
}

TEST_CASE("a Golub-Pereyra request re-solves a point cached range-only")
{
    // fit_varpro configures the problem once from options.jacobian, but the
    // per-call variant is what decides what the factorization must contain.
    // Asking for the exact Jacobian at a point whose cache holds no sigma has
    // to recompute rather than divide by an empty vector.
    std::mt19937 gen(33);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);
    ReducedProblem<WhitenedBasis> reduced(problem.z_hat, y_hat, problem.e_hat,
                                          problem.basis, 0.0,
                                          InnerFactors::RangeOnly);

    const Eigen::VectorXd residual = reduced.residual(problem.theta_hat);
    CHECK(reduced.solve_at(problem.theta_hat, InnerFactors::RangeOnly).sigma.size() == 0);

    const Eigen::MatrixXd exact = reduced.jacobian(
        problem.theta_hat, JacobianVariant::GolubPereyra, problem.num_params);
    CHECK(exact.allFinite());

    ReducedProblem<WhitenedBasis> reference(problem.z_hat, y_hat, problem.e_hat,
                                            problem.basis, 0.0, InnerFactors::Full);
    const Eigen::MatrixXd expected = reference.jacobian(
        problem.theta_hat, JacobianVariant::GolubPereyra, problem.num_params);
    CHECK((exact - expected).cwiseAbs().maxCoeff() < 1e-9);
    CHECK((residual - reference.residual(problem.theta_hat)).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("Frisch-Waugh-Lovell: residualize then fit equals the joint fit")
{
    // This identity is what licenses projecting the extra block out ONCE up
    // front rather than at every trial point -- the whole reason the constant
    // block never enters the outer loop.
    std::mt19937 gen(4);
    for ( int num_extra : {1, 3} )
    {
        const int num_modes = 5;
        const Eigen::MatrixXd A = test_helpers::randn_points(20, num_modes, gen);
        const Eigen::MatrixXd B = test_helpers::randn_points(20, num_extra, gen);
        const Eigen::VectorXd y = test_helpers::randn_points(20, 1, gen).col(0);

        Eigen::MatrixXd joint(20, num_modes + num_extra);
        joint << A, B;
        const Eigen::VectorXd all =
            joint.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
        const Eigen::VectorXd c_joint = all.head(num_modes);
        const Eigen::VectorXd s_joint = all.tail(num_extra);
        const Eigen::VectorXd r_joint = y - A * c_joint - B * s_joint;

        const Eigen::MatrixXd Q = orthonormal_range(B);
        const InnerSolve solution =
            inner_solve(project_out(Q, A), project_out(Q, y), 0.0);
        const Eigen::VectorXd s =
            B.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                .solve(Eigen::VectorXd(y - A * solution.c));

        CHECK((solution.c - c_joint).cwiseAbs().maxCoeff() < 1e-8);
        CHECK((s - s_joint).cwiseAbs().maxCoeff() < 1e-8);
        CHECK((solution.residual - r_joint).cwiseAbs().maxCoeff() < 1e-8);
    }
}

TEST_CASE("the Golub-Pereyra Jacobian matches finite differences")
{
    // The strict exactness test. The reduced residual's parameter dependence
    // runs through c*(theta_hat) and the projector too, and y_hat is random so
    // the residual is large -- which means the second, residual-proportional
    // term genuinely participates. Kaufman alone could not pass this.
    std::mt19937 gen(5);
    constexpr double fd_step = 1e-6;
    for ( auto setup : {std::make_pair(1, MuMode::Fitted),
                        std::make_pair(2, MuMode::Fitted),
                        std::make_pair(2, MuMode::Pinned)} )
    {
        const Problem problem = make_problem(gen, setup.first, setup.second);
        const Eigen::VectorXd y_hat =
            test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen)
                .col(0);
        ReducedProblem<WhitenedBasis> reduced(problem.z_hat, y_hat, problem.e_hat,
                                              problem.basis, 0.0);

        const Eigen::MatrixXd J = reduced.jacobian(
            problem.theta_hat, JacobianVariant::GolubPereyra, problem.num_params);
        const double scale = std::max(1.0, J.cwiseAbs().maxCoeff());
        for ( int q = 0; q < problem.num_params; ++q )
        {
            Eigen::VectorXd plus = problem.theta_hat, minus = problem.theta_hat;
            plus(q) += fd_step;
            minus(q) -= fd_step;
            const Eigen::VectorXd fd =
                (reduced.residual(plus) - reduced.residual(minus)) / (2 * fd_step);
            CHECK((J.col(q) - fd).cwiseAbs().maxCoeff() / scale < 1e-5);
        }
    }
}

TEST_CASE("Kaufman and Golub-Pereyra differ only in the quadratic model")
{
    // The structural identities: Kaufman's columns lie in range(A~)'s
    // orthogonal complement and the dropped term's lie in range(A~), so the
    // two are orthogonal -- and therefore BOTH VARIANTS GIVE THE SAME
    // GRADIENT. Kaufman never changes first-order information.
    std::mt19937 gen(6);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);
    ReducedProblem<WhitenedBasis> reduced(problem.z_hat, y_hat, problem.e_hat,
                                          problem.basis, 0.0);

    const Eigen::MatrixXd kaufman = reduced.jacobian(
        problem.theta_hat, JacobianVariant::Kaufman, problem.num_params);
    const Eigen::MatrixXd exact = reduced.jacobian(
        problem.theta_hat, JacobianVariant::GolubPereyra, problem.num_params);
    const InnerSolve& solution =
        reduced.solve_at(problem.theta_hat, InnerFactors::Full);

    // the dropped term is genuinely present
    const Eigen::MatrixXd dropped = exact - kaufman;
    CHECK(dropped.cwiseAbs().maxCoeff() > 1e-6);

    // Kaufman's columns are orthogonal to range(A~), the dropped term's lie in it
    CHECK((solution.U.transpose() * kaufman).cwiseAbs().maxCoeff() < 1e-10);
    CHECK(project_out(solution.U, dropped).cwiseAbs().maxCoeff() < 1e-10);

    // ...hence the same exact gradient
    const Eigen::VectorXd g_kaufman = kaufman.transpose() * solution.residual;
    const Eigen::VectorXd g_exact = exact.transpose() * solution.residual;
    CHECK((g_kaufman - g_exact).cwiseAbs().maxCoeff()
          < 1e-10 * std::max(1.0, g_exact.cwiseAbs().maxCoeff()));
}

TEST_CASE("the two Jacobian variants coincide at a zero-residual fit")
{
    // The dropped term is O(||r||), so where the model fits exactly there is
    // nothing to drop.
    std::mt19937 gen(7);
    const Problem problem = make_problem(gen);
    const Eigen::MatrixXd values = problem.basis(problem.theta_hat).values();
    const Eigen::MatrixXd A = problem.z_hat.transpose() * values;
    const Eigen::MatrixXd B = problem.z_hat.transpose() * problem.e_hat;
    const Eigen::VectorXd c = random_vector(static_cast<int>(A.cols()), gen);
    const Eigen::VectorXd s = random_vector(static_cast<int>(B.cols()), gen);
    const Eigen::VectorXd y_hat = A * c + B * s;

    ReducedProblem<WhitenedBasis> reduced(problem.z_hat, y_hat, problem.e_hat,
                                          problem.basis, 0.0);
    REQUIRE(reduced.residual(problem.theta_hat).norm() < 1e-9);

    const Eigen::MatrixXd kaufman = reduced.jacobian(
        problem.theta_hat, JacobianVariant::Kaufman, problem.num_params);
    const Eigen::MatrixXd exact = reduced.jacobian(
        problem.theta_hat, JacobianVariant::GolubPereyra, problem.num_params);
    CHECK((kaufman - exact).cwiseAbs().maxCoeff() < 1e-8);
}

TEST_CASE("the basis is evaluated once per point, not once per question")
{
    // The outer loop asks for the residual and the Jacobian separately but
    // back to back at the same point, and both need the same evaluation and
    // the same factorization. Anything else doubles the dominant cost.
    std::mt19937 gen(8);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);

    int evaluations = 0;
    const CountingBasis counting{&problem.basis, &evaluations};
    ReducedProblem<CountingBasis> reduced(problem.z_hat, y_hat, problem.e_hat,
                                          counting, 0.0);

    reduced.residual(problem.theta_hat);
    CHECK(evaluations == 1);
    reduced.jacobian(problem.theta_hat, JacobianVariant::Kaufman, problem.num_params);
    CHECK(evaluations == 1);
    reduced.residual(problem.theta_hat);
    CHECK(evaluations == 1);

    Eigen::VectorXd elsewhere = problem.theta_hat;
    elsewhere(0) += 0.1;
    reduced.residual(elsewhere);
    CHECK(evaluations == 2);
}

TEST_CASE("a fit recovers a synthetic target it was built from")
{
    std::mt19937 gen(9);
    for ( MuMode mu_mode : {MuMode::Pinned, MuMode::Fitted} )
    {
        const Problem problem = make_problem(gen, 2, mu_mode);
        const Eigen::MatrixXd A =
            problem.z_hat.transpose() * problem.basis(problem.theta_hat).values();
        const Eigen::MatrixXd B = problem.z_hat.transpose() * problem.e_hat;
        const Eigen::VectorXd c_true = random_vector(static_cast<int>(A.cols()), gen);
        const Eigen::VectorXd s_true = random_vector(static_cast<int>(B.cols()), gen);
        const Eigen::VectorXd y_hat = A * c_true + B * s_true;

        Eigen::VectorXd start = problem.theta_hat;
        start += 0.05 * random_vector(problem.num_params, gen);

        VarProOptions options;
        options.ridge = 0.0;
        options.max_evaluations = 200;
        const VarProResult result = fit_varpro(problem.z_hat, y_hat, problem.basis,
                                               start, problem.e_hat, options);

        MESSAGE("recovery: " << result.num_iterations << " steps, cost " << result.cost
                             << ", |dtheta| "
                             << (result.theta_hat - problem.theta_hat).norm());
        CHECK(result.success);
        CHECK(result.cost < 1e-16);
        CHECK((result.theta_hat - problem.theta_hat).cwiseAbs().maxCoeff() < 1e-5);
        CHECK((result.c - c_true).cwiseAbs().maxCoeff() < 1e-5);
        CHECK((result.s - s_true).cwiseAbs().maxCoeff() < 1e-5);
    }
}

TEST_CASE("the returned pieces are consistent with each other on a noisy problem")
{
    std::mt19937 gen(10);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);

    const VarProResult result =
        fit_varpro(problem.z_hat, y_hat, problem.basis, problem.theta_hat,
                   problem.e_hat);

    const Eigen::MatrixXd A =
        problem.z_hat.transpose() * problem.basis(result.theta_hat).values();
    const Eigen::MatrixXd B = problem.z_hat.transpose() * problem.e_hat;
    const Eigen::VectorXd rebuilt = y_hat - A * result.c - B * result.s;

    CHECK((result.residual - rebuilt).cwiseAbs().maxCoeff() < 1e-10);
    CHECK(result.cost == doctest::Approx(0.5 * result.residual.squaredNorm()));
    CHECK(result.residual.size() == problem.num_probes);
    CHECK(result.c.size() == static_cast<Eigen::Index>(problem.modes.size()));
    CHECK(result.s.size() == problem.e_hat.cols());
    CHECK(result.jacobian.rows() == problem.num_probes);
    CHECK(result.jacobian.cols() == problem.num_params);
}

TEST_CASE("the callback traces the outer iteration path")
{
    std::mt19937 gen(11);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);

    std::vector<Eigen::VectorXd> visited;
    const auto callback = [&]( const Eigen::VectorXd& point, const Eigen::VectorXd&,
                               const Eigen::VectorXd& ) { visited.push_back(point); };

    VarProOptions options;
    options.max_evaluations = 100;
    const VarProResult result = fit_varpro(problem.z_hat, y_hat, problem.basis,
                                           problem.theta_hat, problem.e_hat, options,
                                           callback);

    REQUIRE(visited.size() >= 1u);
    CHECK(visited.front() == problem.theta_hat);
    CHECK(visited.back() == result.theta_hat);
    // The Jacobian is evaluated exactly once per outer iteration -- at the
    // start and after each accepted step -- so no de-duplication is needed.
    CHECK(visited.size() == static_cast<std::size_t>(result.num_iterations) + 1u);
}

TEST_CASE("an infeasible trial point is scored, not thrown")
{
    // A log-Cholesky diagonal far enough out overflows the ellipsoid. That is
    // an extreme trial step, not a bug: the point must score as the worst
    // finite cost so the outer loop backs off.
    std::mt19937 gen(12);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);
    ReducedProblem<WhitenedBasis> reduced(problem.z_hat, y_hat, problem.e_hat,
                                          problem.basis, 1e-8);

    Eigen::VectorXd runaway = problem.theta_hat;
    runaway(problem.num_params - 3) = -900.0;  // into the log-diagonal block
    CHECK_THROWS_AS(problem.basis(runaway), InfeasibleParameters);

    Eigen::VectorXd sentinel;
    CHECK_NOTHROW(sentinel = reduced.residual(runaway));
    CHECK(sentinel.allFinite());
    // "the smooth model contributes nothing": the residual is the data itself,
    // residualized against the extra block
    const Eigen::MatrixXd Q =
        orthonormal_range(Eigen::MatrixXd(problem.z_hat.transpose() * problem.e_hat));
    CHECK((sentinel - Eigen::VectorXd(project_out(Q, y_hat))).cwiseAbs().maxCoeff() < 1e-12);
    // and it is worse than an ordinary point, so a fit rejects the step
    CHECK(sentinel.norm() > reduced.residual(problem.theta_hat).norm());
}

TEST_CASE("Golub-Pereyra is refused when the basis cannot supply it")
{
    std::mt19937 gen(13);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);
    const BasisWithoutJacobian limited{&problem.basis};

    // the default variant is happy with values() and vjp() alone
    VarProOptions options;
    CHECK_NOTHROW(fit_varpro(problem.z_hat, y_hat, limited, problem.theta_hat,
                             problem.e_hat, options));

    options.jacobian = JacobianVariant::GolubPereyra;
    CHECK_THROWS_AS(fit_varpro(problem.z_hat, y_hat, limited, problem.theta_hat,
                               problem.e_hat, options),
                    std::invalid_argument);
}

TEST_CASE("malformed inputs are rejected eagerly")
{
    std::mt19937 gen(14);
    const Problem problem = make_problem(gen);
    const Eigen::VectorXd y_hat =
        test_helpers::randn_points(static_cast<int>(problem.num_probes), 1, gen).col(0);

    CHECK_THROWS_AS(fit_varpro(problem.z_hat, Eigen::VectorXd::Zero(3),
                               problem.basis, problem.theta_hat, problem.e_hat),
                    std::invalid_argument);
    CHECK_THROWS_AS(fit_varpro(problem.z_hat, y_hat, problem.basis,
                               problem.theta_hat,
                               Eigen::MatrixXd::Zero(problem.z_hat.rows() + 1, 1)),
                    std::invalid_argument);

    // fewer probes than parameters
    const Eigen::MatrixXd few = problem.z_hat.leftCols(2);
    CHECK_THROWS_AS(fit_varpro(few, Eigen::VectorXd::Zero(2), problem.basis,
                               problem.theta_hat, problem.e_hat),
                    std::invalid_argument);

    // a start point the basis cannot be evaluated at
    Eigen::VectorXd runaway = problem.theta_hat;
    runaway(problem.num_params - 3) = -900.0;
    CHECK_THROWS_AS(fit_varpro(problem.z_hat, y_hat, problem.basis, runaway,
                               problem.e_hat),
                    std::invalid_argument);
}

TEST_CASE("a fit with no extra basis at all")
{
    std::mt19937 gen(15);
    const Problem problem = make_problem(gen, 2, MuMode::Fitted, 0);
    const Eigen::MatrixXd A =
        problem.z_hat.transpose() * problem.basis(problem.theta_hat).values();
    const Eigen::VectorXd c_true = random_vector(static_cast<int>(A.cols()), gen);
    const Eigen::VectorXd y_hat = A * c_true;

    Eigen::VectorXd start = problem.theta_hat;
    start += 0.05 * random_vector(problem.num_params, gen);

    VarProOptions options;
    options.ridge = 0.0;
    options.max_evaluations = 200;
    const VarProResult result =
        fit_varpro(problem.z_hat, y_hat, problem.basis, start, options);

    CHECK(result.s.size() == 0);
    CHECK(result.cost < 1e-16);
    CHECK((result.c - c_true).cwiseAbs().maxCoeff() < 1e-5);
}
