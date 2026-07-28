// SPDX-License-Identifier: MIT
//
// Checks on the hand-rolled Levenberg-Marquardt loop.
//
// Tested in isolation, on problems whose answers are known independently of
// anything else in this repository: a linear least-squares problem whose
// minimizer is
// computable in closed form, and two classical nonlinear least-squares test
// problems with published solutions. The trust-region subproblem solver, being
// the part that is not a transcription of MINPACK, is additionally checked
// directly against the equations it claims to solve.
//
// All self-contained: nothing here is compared against a stored reference, so
// the suite cannot drift out of step with the code it tests.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/detail/levenberg_marquardt.hpp"
#include "test_helpers.hpp"

using lgpsf::LMScaling;
using lgpsf::LMStatus;
using lgpsf::LevenbergMarquardtOptions;
using lgpsf::LevenbergMarquardtResult;
using lgpsf::detail::TrustRegionStep;
using lgpsf::detail::solve_trust_region;
using lgpsf::levenberg_marquardt;

namespace {

/// Rosenbrock as a least-squares problem: r = (10(x2 - x1^2), 1 - x1), whose
/// unique zero is (1, 1). The classical start is (-1.2, 1).
Eigen::VectorXd rosenbrock( const Eigen::VectorXd& x )
{
    Eigen::VectorXd r(2);
    r(0) = 10.0 * (x(1) - x(0) * x(0));
    r(1) = 1.0 - x(0);
    return r;
}

Eigen::MatrixXd rosenbrock_jacobian( const Eigen::VectorXd& x )
{
    Eigen::MatrixXd J(2, 2);
    J << -20.0 * x(0), 10.0,
         -1.0,          0.0;
    return J;
}

/// Powell's singular function: four residuals in four unknowns whose unique
/// zero is the origin, and whose Jacobian is SINGULAR there -- the case that
/// separates a trust-region solver that handles rank deficiency from one that
/// only appears to. The classical start is (3, -1, 0, 1).
Eigen::VectorXd powell_singular( const Eigen::VectorXd& x )
{
    Eigen::VectorXd r(4);
    r(0) = x(0) + 10.0 * x(1);
    r(1) = std::sqrt(5.0) * (x(2) - x(3));
    r(2) = (x(1) - 2.0 * x(2)) * (x(1) - 2.0 * x(2));
    r(3) = std::sqrt(10.0) * (x(0) - x(3)) * (x(0) - x(3));
    return r;
}

Eigen::MatrixXd powell_singular_jacobian( const Eigen::VectorXd& x )
{
    const double a = x(1) - 2.0 * x(2);
    const double b = x(0) - x(3);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(4, 4);
    J(0, 0) = 1.0;
    J(0, 1) = 10.0;
    J(1, 2) = std::sqrt(5.0);
    J(1, 3) = -std::sqrt(5.0);
    J(2, 1) = 2.0 * a;
    J(2, 2) = -4.0 * a;
    J(3, 0) = 2.0 * std::sqrt(10.0) * b;
    J(3, 3) = -2.0 * std::sqrt(10.0) * b;
    return J;
}

} // namespace

TEST_CASE("the trust-region subproblem solves the equations it claims to")
{
    // The hook step is the part of this loop that is NOT a transcription of
    // MINPACK, so it is pinned against its own defining conditions rather than
    // only through the outcomes of a fit: the step must sit on the trust-region
    // boundary and satisfy the damped normal equations for the lambda reported.
    std::mt19937 gen(0);
    for ( int rows : {6, 12} )
    {
        for ( int cols : {2, 4, 5} )
        {
            const Eigen::MatrixXd J = test_helpers::randn_points(rows, cols, gen);
            const Eigen::VectorXd f =
                test_helpers::randn_points(rows, 1, gen).col(0);
            const Eigen::VectorXd scale =
                test_helpers::uniform_points(cols, 1, gen, 0.5, 2.0).col(0);

            // A bound far beyond the Gauss-Newton step: lambda must be zero and
            // the step must be the plain least-squares solution.
            const TrustRegionStep loose = solve_trust_region(J, f, scale, 1e6, 0.0);
            CHECK(loose.parameter == 0.0);
            const Eigen::VectorXd gauss_newton =
                J.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(-f);
            CHECK((loose.step - gauss_newton).cwiseAbs().maxCoeff() < 1e-10);

            // A bound well inside it: the step lands on the boundary (within
            // MINPACK's 10% acceptance band) and solves
            // (J^T J + lambda D^2) p = -J^T f.
            const double bound = 0.1 * scale.cwiseProduct(gauss_newton).norm();
            const TrustRegionStep tight =
                solve_trust_region(J, f, scale, bound, 0.0);
            CHECK(tight.parameter > 0.0);
            CHECK(std::abs(scale.cwiseProduct(tight.step).norm() - bound)
                  <= 0.1 * bound);

            const Eigen::MatrixXd damped =
                J.transpose() * J
                + tight.parameter
                      * scale.cwiseProduct(scale).asDiagonal().toDenseMatrix();
            const Eigen::VectorXd normal_equations =
                damped * tight.step + J.transpose() * f;
            CHECK(normal_equations.cwiseAbs().maxCoeff()
                  < 1e-9 * std::max(1.0, (J.transpose() * f).cwiseAbs().maxCoeff()));
        }
    }
}

TEST_CASE("a linear problem converges to the exact least-squares solution")
{
    // r(x) = A x - b has a closed-form minimizer and a NONZERO residual there,
    // so this exercises the ftol path rather than the zero-residual one, and
    // the answer comes from an independent decomposition rather than from this
    // loop.
    std::mt19937 gen(1);
    for ( int rows : {8, 20} )
    {
        for ( int cols : {2, 5} )
        {
            const Eigen::MatrixXd A = test_helpers::randn_points(rows, cols, gen);
            const Eigen::VectorXd b =
                test_helpers::randn_points(rows, 1, gen).col(0);
            const Eigen::VectorXd exact =
                A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

            const auto residual = [&]( const Eigen::VectorXd& x ) {
                return Eigen::VectorXd(A * x - b);
            };
            const auto jacobian = [&]( const Eigen::VectorXd& ) { return A; };

            LevenbergMarquardtOptions options;
            options.max_evaluations = 200;
            const LevenbergMarquardtResult result = levenberg_marquardt(
                residual, jacobian,
                test_helpers::randn_points(cols, 1, gen).col(0), options);

            CHECK(result.success);
            CHECK((result.parameters - exact).cwiseAbs().maxCoeff() < 1e-8);
            CHECK(result.cost
                  == doctest::Approx(0.5 * (A * exact - b).squaredNorm()));
        }
    }
}

TEST_CASE("starting at the solution stops immediately on gtol")
{
    std::mt19937 gen(2);
    const Eigen::MatrixXd A = test_helpers::randn_points(10, 3, gen);
    const Eigen::VectorXd b = test_helpers::randn_points(10, 1, gen).col(0);
    const Eigen::VectorXd exact =
        A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    const auto residual = [&]( const Eigen::VectorXd& x ) {
        return Eigen::VectorXd(A * x - b);
    };
    const auto jacobian = [&]( const Eigen::VectorXd& ) { return A; };

    const LevenbergMarquardtResult result =
        levenberg_marquardt(residual, jacobian, exact);
    CHECK(result.status == LMStatus::GradientTolerance);
    CHECK(result.success);
    CHECK(result.num_iterations == 0);
    CHECK(result.num_residual_evaluations == 1);
    CHECK(result.parameters == exact);
}

TEST_CASE("Rosenbrock converges to its known minimizer")
{
    Eigen::VectorXd start(2);
    start << -1.2, 1.0;

    LevenbergMarquardtOptions options;
    options.max_evaluations = 200;
    const LevenbergMarquardtResult result =
        levenberg_marquardt(rosenbrock, rosenbrock_jacobian, start, options);

    MESSAGE("Rosenbrock: " << result.num_iterations << " accepted steps, "
                           << result.num_residual_evaluations << " evaluations, cost "
                           << result.cost);
    CHECK(result.success);
    CHECK(result.cost < 1e-20);
    CHECK(std::abs(result.parameters(0) - 1.0) < 1e-8);
    CHECK(std::abs(result.parameters(1) - 1.0) < 1e-8);
}

TEST_CASE("Powell's singular function converges despite a singular Jacobian")
{
    Eigen::VectorXd start(4);
    start << 3.0, -1.0, 0.0, 1.0;

    LevenbergMarquardtOptions options;
    options.max_evaluations = 500;
    options.ftol = 1e-14;
    options.xtol = 1e-14;
    options.gtol = 1e-14;
    const LevenbergMarquardtResult result = levenberg_marquardt(
        powell_singular, powell_singular_jacobian, start, options);

    MESSAGE("Powell singular: " << result.num_iterations << " accepted steps, "
                                << result.num_residual_evaluations
                                << " evaluations, cost " << result.cost
                                << ", ||x|| " << result.parameters.norm());
    // Far tighter than the published MINPACK behavior on this problem, and not
    // an accident: the minimum-norm Gauss-Newton step handles the Jacobian's
    // null directions cleanly instead of damping its way past them. Bounds are
    // left with headroom against compiler and platform variation -- the
    // observed values are cost ~1e-58 and ||x|| ~4e-15.
    CHECK(result.cost < 1e-20);
    CHECK(result.parameters.norm() < 1e-6);
}

TEST_CASE("the converged answer does not depend on how the parameters are scaled")
{
    // What x_scale='jac' is for: rescaling a parameter changes the trajectory
    // but must not change where the loop lands. Rosenbrock's second parameter
    // is stretched by a thousand and the answer mapped back.
    Eigen::VectorXd stretch(2);
    stretch << 1.0, 1000.0;

    const auto residual = [&]( const Eigen::VectorXd& y ) {
        return rosenbrock(stretch.cwiseProduct(y));
    };
    const auto jacobian = [&]( const Eigen::VectorXd& y ) {
        return Eigen::MatrixXd(rosenbrock_jacobian(stretch.cwiseProduct(y))
                               * stretch.asDiagonal());
    };

    Eigen::VectorXd start(2);
    start << -1.2, 1.0 / 1000.0;

    LevenbergMarquardtOptions options;
    options.max_evaluations = 400;
    const LevenbergMarquardtResult result =
        levenberg_marquardt(residual, jacobian, start, options);

    MESSAGE("stretched Rosenbrock: " << result.num_iterations << " accepted steps, cost "
                                     << result.cost);
    CHECK(result.success);
    const Eigen::VectorXd recovered = stretch.cwiseProduct(result.parameters);
    CHECK(std::abs(recovered(0) - 1.0) < 1e-6);
    CHECK(std::abs(recovered(1) - 1.0) < 1e-6);
}

TEST_CASE("a single permitted evaluation returns the starting point untouched")
{
    // The operator layer's baseline guard depends on this exactly: a fit given
    // no room to move must hand back its initial parameters, not an
    // almost-initial approximation to them.
    Eigen::VectorXd start(2);
    start << -1.2, 1.0;

    LevenbergMarquardtOptions options;
    options.max_evaluations = 1;
    const LevenbergMarquardtResult result =
        levenberg_marquardt(rosenbrock, rosenbrock_jacobian, start, options);

    CHECK(result.parameters == start);
    CHECK(result.num_iterations == 0);
    CHECK(result.num_residual_evaluations == 1);
    CHECK(result.status == LMStatus::MaxEvaluations);
    CHECK_FALSE(result.success);
    CHECK(result.residual == rosenbrock(start));
    CHECK(result.jacobian == rosenbrock_jacobian(start));
    CHECK(result.cost == doctest::Approx(0.5 * rosenbrock(start).squaredNorm()));
}

TEST_CASE("a non-finite trial residual is rejected, not propagated")
{
    // A basis that cannot be evaluated at an extreme trial point reports the
    // worst finite cost upstream, but the loop must also survive an outright
    // non-finite value: the step is rejected and the trust region contracts.
    const auto residual = [&]( const Eigen::VectorXd& x ) {
        Eigen::VectorXd r(2);
        if ( x(0) > 0.5 )
        {
            r.setConstant(std::numeric_limits<double>::quiet_NaN());
            return r;
        }
        r(0) = x(0) - 0.25;
        r(1) = x(1) + 0.5;
        return r;
    };
    const auto jacobian = [&]( const Eigen::VectorXd& ) {
        return Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2));
    };

    Eigen::VectorXd start(2);
    start << 0.0, 0.0;

    LevenbergMarquardtOptions options;
    options.max_evaluations = 100;
    const LevenbergMarquardtResult result =
        levenberg_marquardt(residual, jacobian, start, options);

    CHECK(result.residual.allFinite());
    CHECK(result.cost < 1e-20);
    CHECK(std::abs(result.parameters(0) - 0.25) < 1e-9);
    CHECK(std::abs(result.parameters(1) + 0.5) < 1e-9);
}

TEST_CASE("a fixed scaling is honored and validated")
{
    Eigen::VectorXd start(2);
    start << -1.2, 1.0;

    LevenbergMarquardtOptions options;
    options.scaling = LMScaling::Fixed;
    options.max_evaluations = 400;
    options.fixed_scale = Eigen::VectorXd::Ones(2);
    const LevenbergMarquardtResult result =
        levenberg_marquardt(rosenbrock, rosenbrock_jacobian, start, options);
    CHECK(result.success);
    CHECK(std::abs(result.parameters(0) - 1.0) < 1e-6);

    options.fixed_scale = Eigen::VectorXd::Ones(3);
    CHECK_THROWS_AS(
        levenberg_marquardt(rosenbrock, rosenbrock_jacobian, start, options),
        std::invalid_argument);
}
