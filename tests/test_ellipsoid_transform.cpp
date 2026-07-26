// SPDX-License-Identifier: MIT
//
// Checks on the ellipsoid pullback and its parameter encodings: the pullback
// against an independent triangular solve, both derivative pairs against
// finite differences AND against adjoint consistency, the forward and reverse
// Jacobian sweeps against each other, and the encoding conversions as exact
// round-trips.
//
// All self-contained -- nothing is compared against a stored reference or
// against the Python prototype.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/ellipsoid_transform.hpp"
#include "test_helpers.hpp"

using lgpsf::EllipsoidFrame;
using lgpsf::FrameTangent;
using lgpsf::MuMode;
using lgpsf::PullbackCotangent;
using lgpsf::dim_from_theta_size;
using lgpsf::eval_T;
using lgpsf::freeze_mu;
using lgpsf::jacobian_tensor_forward;
using lgpsf::jacobian_tensor_reverse;
using lgpsf::jvp_T;
using lgpsf::jvp_unpack_theta_hat;
using lgpsf::make_frame;
using lgpsf::pullback;
using lgpsf::pullback_jvp;
using lgpsf::pullback_vjp;
using lgpsf::release_mu;
using lgpsf::theta_hat_size;
using lgpsf::theta_size;
using lgpsf::to_theta;
using lgpsf::to_theta_hat;
using lgpsf::unpack_theta;
using lgpsf::unpack_theta_hat;
using lgpsf::vjp_T;
using lgpsf::vjp_unpack_theta_hat;

namespace {

// The pullback divides by L's diagonal, so its finite-difference error is a
// little worse conditioned than the LG basis's; 1e-5 on a 1e-6 central step.
constexpr double kFdStep = 1e-6;
constexpr double kFdTol  = 1e-5;

const std::vector<MuMode> kModes = {MuMode::Pinned, MuMode::Fitted};

/// A lower-triangular factor with a well-separated positive diagonal, so
/// nothing under test is near-singular for reasons unrelated to the test.
Eigen::MatrixXd random_cholesky( int dim, std::mt19937& gen )
{
    std::uniform_real_distribution<double> diag(0.5, 2.0);
    std::uniform_real_distribution<double> off(-0.5, 0.5);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(dim, dim);
    for ( int i = 0; i < dim; ++i )
    {
        L(i, i) = diag(gen);
        for ( int j = 0; j < i; ++j )
        {
            L(i, j) = off(gen);
        }
    }
    return L;
}

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

/// A theta_hat with a modest log-diagonal, so L stays away from singular.
Eigen::VectorXd random_theta_hat( int dim, MuMode mode, std::mt19937& gen )
{
    return random_vector(theta_hat_size(dim, mode), gen, -0.3, 0.3);
}

} // namespace

TEST_CASE("theta and theta_hat sizes follow their closed forms")
{
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const int n_lower = dim * (dim - 1) / 2;
        CHECK(theta_size(dim) == 2 * dim + n_lower);
        CHECK(theta_hat_size(dim, MuMode::Fitted) == theta_size(dim));
        CHECK(theta_hat_size(dim, MuMode::Pinned) == dim + n_lower);

        // Pinning removes exactly the center parameters, nothing else.
        CHECK(theta_hat_size(dim, MuMode::Fitted)
              - theta_hat_size(dim, MuMode::Pinned) == dim);

        // The public encoding is self-describing.
        CHECK(dim_from_theta_size(theta_size(dim)) == dim);
    }
    // 2, 5, 9, 14, ... -- lengths between them decode nothing.
    CHECK_THROWS_AS(dim_from_theta_size(4), std::invalid_argument);
    CHECK_THROWS_AS(dim_from_theta_size(0), std::invalid_argument);
}

TEST_CASE("the pullback inverts L against an independent triangular solve")
{
    // u is DEFINED by L u = x - mu. pullback computes it through an explicit
    // inverse instead; this pins that shortcut against the defining equation,
    // which no other check in this file does.
    std::mt19937 gen(11);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const EllipsoidFrame frame =
            make_frame(random_vector(dim, gen), random_cholesky(dim, gen));

        CHECK((frame.L * frame.L_inv - Eigen::MatrixXd::Identity(dim, dim))
                  .cwiseAbs().maxCoeff() < 1e-12);

        const Eigen::MatrixXd x = test_helpers::uniform_points(20, dim, gen, -1.5, 1.5);
        const Eigen::MatrixXd u = pullback(frame, x);

        for ( Eigen::Index k = 0; k < x.rows(); ++k )
        {
            const Eigen::VectorXd residual =
                frame.L * u.row(k).transpose()
                - (x.row(k).transpose() - frame.mu);
            CHECK(residual.cwiseAbs().maxCoeff() < 1e-12);
        }
    }
}

TEST_CASE("pullback JVP and VJP are adjoint")
{
    std::mt19937 gen(0);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const EllipsoidFrame frame =
            make_frame(random_vector(dim, gen), random_cholesky(dim, gen));
        const Eigen::MatrixXd x = test_helpers::uniform_points(20, dim, gen, -1.5, 1.5);
        const Eigen::MatrixXd u = pullback(frame, x);

        FrameTangent tangent;
        tangent.dmu = random_vector(dim, gen);
        tangent.dL = test_helpers::uniform_points(dim, dim, gen, -1.0, 1.0);
        const Eigen::MatrixXd w =
            test_helpers::uniform_points(20, dim, gen, -1.5, 1.5);

        const Eigen::MatrixXd du = pullback_jvp(frame, tangent, u);
        const PullbackCotangent cotangent = pullback_vjp(frame, u, w);

        for ( Eigen::Index k = 0; k < x.rows(); ++k )
        {
            const double lhs = w.row(k).dot(du.row(k));
            double rhs = cotangent.w_mu.row(k).dot(tangent.dmu);
            for ( int i = 0; i < dim; ++i )
            {
                for ( int j = 0; j < dim; ++j )
                {
                    rhs += cotangent.component(i, j)(k) * tangent.dL(i, j);
                }
            }
            CHECK(std::abs(lhs - rhs) < 1e-10);
        }
    }
}

TEST_CASE("pullback JVP matches finite differences")
{
    std::mt19937 gen(1);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const Eigen::VectorXd mu = random_vector(dim, gen);
        const Eigen::MatrixXd L = random_cholesky(dim, gen);
        const EllipsoidFrame frame = make_frame(mu, L);
        const Eigen::MatrixXd x = test_helpers::uniform_points(10, dim, gen, -1.5, 1.5);
        const Eigen::MatrixXd u = pullback(frame, x);

        // perturb the center
        for ( int k = 0; k < dim; ++k )
        {
            FrameTangent tangent;
            tangent.dmu = Eigen::VectorXd::Zero(dim);
            tangent.dmu(k) = 1.0;
            tangent.dL = Eigen::MatrixXd::Zero(dim, dim);
            const Eigen::MatrixXd analytic = pullback_jvp(frame, tangent, u);

            Eigen::VectorXd mu_p = mu, mu_m = mu;
            mu_p(k) += kFdStep;
            mu_m(k) -= kFdStep;
            const Eigen::MatrixXd fd =
                (pullback(make_frame(mu_p, L), x) - pullback(make_frame(mu_m, L), x))
                / (2 * kFdStep);
            CHECK((analytic - fd).cwiseAbs().maxCoeff() < kFdTol);
        }

        // perturb each lower-triangular entry, diagonal included
        for ( int a = 0; a < dim; ++a )
        {
            for ( int b = 0; b <= a; ++b )
            {
                FrameTangent tangent;
                tangent.dmu = Eigen::VectorXd::Zero(dim);
                tangent.dL = Eigen::MatrixXd::Zero(dim, dim);
                tangent.dL(a, b) = 1.0;
                const Eigen::MatrixXd analytic = pullback_jvp(frame, tangent, u);

                Eigen::MatrixXd L_p = L, L_m = L;
                L_p(a, b) += kFdStep;
                L_m(a, b) -= kFdStep;
                const Eigen::MatrixXd fd =
                    (pullback(make_frame(mu, L_p), x) - pullback(make_frame(mu, L_m), x))
                    / (2 * kFdStep);
                CHECK((analytic - fd).cwiseAbs().maxCoeff() < kFdTol);
            }
        }
    }
}

TEST_CASE("T's JVP and VJP are adjoint in theta_hat")
{
    std::mt19937 gen(2);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( MuMode mode : kModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mode, gen);
            const int n_params = theta_hat_size(dim, mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(15, dim, gen, -1.5, 1.5);
            const Eigen::VectorXd dtheta_hat = random_vector(n_params, gen);
            const Eigen::MatrixXd w =
                test_helpers::uniform_points(15, dim, gen, -1.5, 1.5);

            const Eigen::MatrixXd du = jvp_T(theta_hat, dtheta_hat, x, mu0, mode);
            const Eigen::MatrixXd rows = vjp_T(theta_hat, x, w, mu0, mode);
            REQUIRE(rows.cols() == n_params);

            for ( Eigen::Index k = 0; k < x.rows(); ++k )
            {
                const double lhs = w.row(k).dot(du.row(k));
                const double rhs = rows.row(k).dot(dtheta_hat);
                CHECK(std::abs(lhs - rhs) < 1e-10);
            }
        }
    }
}

TEST_CASE("T's JVP matches finite differences in theta_hat")
{
    std::mt19937 gen(3);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( MuMode mode : kModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mode, gen);
            const int n_params = theta_hat_size(dim, mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(10, dim, gen, -1.5, 1.5);

            for ( int q = 0; q < n_params; ++q )
            {
                Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_params);
                direction(q) = 1.0;
                const Eigen::MatrixXd analytic =
                    jvp_T(theta_hat, direction, x, mu0, mode);

                Eigen::VectorXd plus = theta_hat, minus = theta_hat;
                plus(q) += kFdStep;
                minus(q) -= kFdStep;
                const Eigen::MatrixXd fd =
                    (eval_T(plus, x, mu0, mode) - eval_T(minus, x, mu0, mode))
                    / (2 * kFdStep);
                CHECK((analytic - fd).cwiseAbs().maxCoeff() < kFdTol);
            }
        }
    }
}

TEST_CASE("the forward and reverse Jacobian sweeps agree")
{
    std::mt19937 gen(4);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( MuMode mode : kModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mode, gen);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(12, dim, gen, -1.5, 1.5);

            const std::vector<Eigen::MatrixXd> forward =
                jacobian_tensor_forward(theta_hat, x, mu0, mode);
            const std::vector<Eigen::MatrixXd> reverse =
                jacobian_tensor_reverse(theta_hat, x, mu0, mode);

            REQUIRE(forward.size()
                    == static_cast<std::size_t>(theta_hat_size(dim, mode)));
            REQUIRE(reverse.size() == forward.size());
            for ( std::size_t q = 0; q < forward.size(); ++q )
            {
                CHECK((forward[q] - reverse[q]).cwiseAbs().maxCoeff() < 1e-10);
            }
        }
    }
}

TEST_CASE("the Jacobian tensor's columns are the directional derivatives")
{
    // The tensor shares one frame and one pullback across all P directions;
    // this pins that sharing as a pure restructuring -- column q must be
    // exactly what a separate jvp_T call in direction q produces.
    std::mt19937 gen(5);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( MuMode mode : kModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mode, gen);
            const int n_params = theta_hat_size(dim, mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(9, dim, gen, -1.5, 1.5);

            const std::vector<Eigen::MatrixXd> tensor =
                jacobian_tensor_forward(theta_hat, x, mu0, mode);
            for ( int q = 0; q < n_params; ++q )
            {
                Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_params);
                direction(q) = 1.0;
                CHECK(tensor[static_cast<std::size_t>(q)]
                      == jvp_T(theta_hat, direction, x, mu0, mode));
            }
        }
    }
}

TEST_CASE("the Jacobian tensor matches finite differences")
{
    std::mt19937 gen(6);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( MuMode mode : kModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mode, gen);
            const int n_params = theta_hat_size(dim, mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(8, dim, gen, -1.5, 1.5);

            const std::vector<Eigen::MatrixXd> tensor =
                jacobian_tensor_forward(theta_hat, x, mu0, mode);
            for ( int q = 0; q < n_params; ++q )
            {
                Eigen::VectorXd plus = theta_hat, minus = theta_hat;
                plus(q) += kFdStep;
                minus(q) -= kFdStep;
                const Eigen::MatrixXd fd =
                    (eval_T(plus, x, mu0, mode) - eval_T(minus, x, mu0, mode))
                    / (2 * kFdStep);
                CHECK((tensor[static_cast<std::size_t>(q)] - fd)
                          .cwiseAbs().maxCoeff() < kFdTol);
            }
        }
    }
}

TEST_CASE("the two encodings describe the same ellipsoid")
{
    std::mt19937 gen(7);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( MuMode mode : kModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mode, gen);

            const Eigen::VectorXd theta = to_theta(theta_hat, mu0, mode);
            REQUIRE(theta.size() == theta_size(dim));

            // Round-tripping is exact in every component except the center,
            // where it cannot be: the displacement encoding stores mu as
            // mu0 + delta, and (mu0 + delta) - mu0 recovers delta only to the
            // resolution of mu itself. That is the same resolution the
            // absolute encoding stores mu at in the first place, so the
            // displacement encoding loses nothing the public one didn't --
            // which is the property worth pinning, and it is what licenses
            // converting freely at the fit's boundaries.
            const double resolution =
                std::numeric_limits<double>::epsilon()
                * std::max(1.0, unpack_theta(theta).mu.cwiseAbs().maxCoeff());
            CHECK((to_theta_hat(theta, mu0, mode) - theta_hat)
                      .cwiseAbs().maxCoeff() <= resolution);
            CHECK((to_theta(to_theta_hat(theta, mu0, mode), mu0, mode) - theta)
                      .cwiseAbs().maxCoeff() <= resolution);

            // The L block, which has no such cancellation, round-trips exactly.
            CHECK(to_theta_hat(theta, mu0, mode).tail(dim * (dim + 1) / 2)
                  == theta_hat.tail(dim * (dim + 1) / 2));

            // and both decode to the same frame
            const EllipsoidFrame from_hat = unpack_theta_hat(theta_hat, mu0, mode);
            const EllipsoidFrame from_theta = unpack_theta(theta);
            CHECK((from_hat.mu - from_theta.mu).cwiseAbs().maxCoeff() < 1e-14);
            CHECK((from_hat.L - from_theta.L).cwiseAbs().maxCoeff() < 1e-14);

            // the public encoding carries the center explicitly
            CHECK((from_theta.mu - theta.head(dim)).cwiseAbs().maxCoeff() == 0.0);
        }
    }
}

TEST_CASE("pinning and releasing mu round-trip and preserve the pullback")
{
    // Releasing must warm-start a fitted stage at exactly the pinned stage's
    // ellipsoid -- that equivalence is what licenses the two-stage mu policy.
    std::mt19937 gen(8);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const Eigen::VectorXd mu0 = random_vector(dim, gen);
        const Eigen::VectorXd pinned = random_theta_hat(dim, MuMode::Pinned, gen);
        const Eigen::MatrixXd x = test_helpers::uniform_points(7, dim, gen, -1.5, 1.5);

        const Eigen::VectorXd released = release_mu(pinned, dim);
        REQUIRE(released.size() == theta_hat_size(dim, MuMode::Fitted));
        CHECK(released.head(dim).cwiseAbs().maxCoeff() == 0.0);
        CHECK(eval_T(released, x, mu0, MuMode::Fitted)
              == eval_T(pinned, x, mu0, MuMode::Pinned));

        // freeze_mu re-pins at whatever center theta describes, and the
        // re-pinned pair reproduces it exactly.
        const Eigen::VectorXd fitted = random_theta_hat(dim, MuMode::Fitted, gen);
        const Eigen::VectorXd theta = to_theta(fitted, mu0, MuMode::Fitted);
        const std::pair<Eigen::VectorXd, Eigen::VectorXd> frozen = freeze_mu(theta);
        CHECK(frozen.second == theta.head(dim));
        CHECK(eval_T(frozen.first, x, frozen.second, MuMode::Pinned)
              == eval_T(fitted, x, mu0, MuMode::Fitted));
        CHECK(release_mu(frozen.first, dim) == to_theta_hat(theta, frozen.second,
                                                            MuMode::Fitted));
    }
}

TEST_CASE("pinned mode ignores mu0's role in the parameters, not in the geometry")
{
    // A pinned theta_hat has no center parameters at all, so its Jacobian has
    // exactly N fewer columns -- not N zero columns, which is what would break
    // the LM's column-norm scaling.
    std::mt19937 gen(9);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const Eigen::VectorXd mu0 = random_vector(dim, gen);
        const Eigen::VectorXd pinned = random_theta_hat(dim, MuMode::Pinned, gen);
        const Eigen::MatrixXd x = test_helpers::uniform_points(10, dim, gen, -1.5, 1.5);
        const Eigen::MatrixXd w = test_helpers::uniform_points(10, dim, gen, -1.0, 1.0);

        const Eigen::MatrixXd rows = vjp_T(pinned, x, w, mu0, MuMode::Pinned);
        CHECK(rows.cols() == theta_hat_size(dim, MuMode::Pinned));

        // Moving mu0 still moves the geometry: the pullback shifts by exactly
        // L^{-1} times the shift.
        Eigen::VectorXd shifted_mu0 = mu0;
        shifted_mu0(0) += 0.25;
        const EllipsoidFrame frame = unpack_theta_hat(pinned, mu0, MuMode::Pinned);
        const Eigen::MatrixXd before = eval_T(pinned, x, mu0, MuMode::Pinned);
        const Eigen::MatrixXd after = eval_T(pinned, x, shifted_mu0, MuMode::Pinned);
        const Eigen::VectorXd expected = frame.L_inv * (mu0 - shifted_mu0);
        for ( Eigen::Index k = 0; k < x.rows(); ++k )
        {
            CHECK(((after.row(k) - before.row(k)).transpose() - expected)
                      .cwiseAbs().maxCoeff() < 1e-12);
        }
    }
}

TEST_CASE("malformed inputs are rejected eagerly")
{
    const Eigen::VectorXd mu0 = Eigen::VectorXd::Zero(2);
    const Eigen::VectorXd theta_hat = Eigen::VectorXd::Zero(5);  // fitted, dim 2

    CHECK_THROWS_AS(unpack_theta_hat(theta_hat, mu0, MuMode::Pinned),
                    std::invalid_argument);
    CHECK_NOTHROW(unpack_theta_hat(theta_hat, mu0, MuMode::Fitted));

    // a Sigma passed where its Cholesky factor belongs
    Eigen::MatrixXd symmetric(2, 2);
    symmetric << 2.0, 0.5, 0.5, 3.0;
    CHECK_THROWS_AS(make_frame(mu0, symmetric), std::invalid_argument);

    // a non-positive diagonal
    Eigen::MatrixXd degenerate(2, 2);
    degenerate << 1.0, 0.0, 0.5, 0.0;
    CHECK_THROWS_AS(make_frame(mu0, degenerate), std::invalid_argument);

    // points of the wrong dimension
    const EllipsoidFrame frame = make_frame(mu0, Eigen::MatrixXd::Identity(2, 2));
    CHECK_THROWS_AS(pullback(frame, Eigen::MatrixXd::Zero(4, 3)),
                    std::invalid_argument);
}
