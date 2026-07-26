// SPDX-License-Identifier: MIT
//
// Checks on the composed feature phi_i(x) = psi_i(T(x)): the composition
// against its own definition, and the parameter-derivative chain against
// finite differences AND adjoint consistency.
//
// All self-contained -- nothing is compared against a stored reference or
// against the Python prototype.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/lg_ellipsoid_feature.hpp"
#include "test_helpers.hpp"

using lgpsf::FeatureAt;
using lgpsf::Mode;
using lgpsf::MuMode;
using lgpsf::eval_T;
using lgpsf::eval_feature;
using lgpsf::eval_lg_basis;
using lgpsf::jac_feature;
using lgpsf::jvp_feature;
using lgpsf::num_harmonics;
using lgpsf::theta_hat_size;
using lgpsf::vjp_feature;

namespace {

constexpr double kFdStep = 1e-6;

const std::vector<MuMode> kMuModes = {MuMode::Pinned, MuMode::Fitted};

/// A mode set broad enough to exercise shared harmonics (several p per shell)
/// and shared radial profiles (several m per degree) at once.
std::vector<Mode> some_modes( int dim, int max_ell = 3, int max_p = 2 )
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

Eigen::VectorXd random_theta_hat( int dim, MuMode mode, std::mt19937& gen )
{
    return random_vector(theta_hat_size(dim, mode), gen, -0.3, 0.3);
}

} // namespace

TEST_CASE("the feature is the LG basis composed with the pullback")
{
    // phi_i(x) := psi_i(T(x)) is a definition, so it is checked as one: the
    // pullback computed separately, then the LG basis on it. Exact, not to a
    // tolerance -- FeatureAt shares work between the two but must not change
    // a single value by doing so.
    std::mt19937 gen(0);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim);
        for ( MuMode mu_mode : kMuModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(12, dim, gen, -1.5, 1.5);

            const Eigen::MatrixXd expected =
                eval_lg_basis(modes, eval_T(theta_hat, x, mu0, mu_mode));
            CHECK(eval_feature(theta_hat, x, modes, mu0, mu_mode) == expected);
            CHECK(expected.rows() == x.rows());
            CHECK(expected.cols() == static_cast<Eigen::Index>(modes.size()));
        }
    }
}

TEST_CASE("the feature's JVP and VJP are adjoint")
{
    std::mt19937 gen(1);
    double worst = 0.0;
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim);
        for ( MuMode mu_mode : kMuModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
            const int n_params = theta_hat_size(dim, mu_mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(10, dim, gen, -1.5, 1.5);
            const Eigen::VectorXd dtheta_hat = random_vector(n_params, gen);
            const Eigen::MatrixXd w = test_helpers::uniform_points(
                10, static_cast<int>(modes.size()), gen, -1.0, 1.0);

            const Eigen::MatrixXd dphi =
                jvp_feature(theta_hat, dtheta_hat, x, modes, mu0, mu_mode);
            const Eigen::MatrixXd rows =
                vjp_feature(theta_hat, x, w, modes, mu0, mu_mode);
            REQUIRE(rows.cols() == n_params);

            for ( Eigen::Index k = 0; k < x.rows(); ++k )
            {
                const double lhs = w.row(k).dot(dphi.row(k));
                const double rhs = rows.row(k).dot(dtheta_hat);
                const double gap =
                    std::abs(lhs - rhs) / std::max(1.0, std::abs(lhs));
                worst = std::max(worst, gap);
                CHECK(gap < 1e-11);
            }
        }
    }
    MESSAGE("worst relative adjoint-consistency gap: " << worst);
}

TEST_CASE("the feature's JVP matches finite differences")
{
    std::mt19937 gen(2);
    double worst = 0.0;
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim);
        for ( MuMode mu_mode : kMuModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
            const int n_params = theta_hat_size(dim, mu_mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(8, dim, gen, -1.5, 1.5);

            for ( int q = 0; q < n_params; ++q )
            {
                Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_params);
                direction(q) = 1.0;
                const Eigen::MatrixXd analytic =
                    jvp_feature(theta_hat, direction, x, modes, mu0, mu_mode);

                Eigen::VectorXd plus = theta_hat, minus = theta_hat;
                plus(q) += kFdStep;
                minus(q) -= kFdStep;
                const Eigen::MatrixXd fd =
                    (eval_feature(plus, x, modes, mu0, mu_mode)
                     - eval_feature(minus, x, modes, mu0, mu_mode))
                    / (2 * kFdStep);

                const double err =
                    (analytic - fd).cwiseAbs().maxCoeff()
                    / std::max(1.0, analytic.cwiseAbs().maxCoeff());
                worst = std::max(worst, err);
                CHECK(err < 1e-5);
            }
        }
    }
    MESSAGE("worst relative finite-difference error: " << worst);
}

TEST_CASE("the feature Jacobian's columns are the directional derivatives")
{
    // jac() shares the pullback and the per-mode spatial gradients across all
    // P directions; this pins that sharing as a pure restructuring.
    std::mt19937 gen(3);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim);
        for ( MuMode mu_mode : kMuModes )
        {
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
            const int n_params = theta_hat_size(dim, mu_mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(10, dim, gen, -1.5, 1.5);

            const std::vector<Eigen::MatrixXd> jac =
                jac_feature(theta_hat, x, modes, mu0, mu_mode);
            REQUIRE(jac.size() == static_cast<std::size_t>(n_params));
            for ( int q = 0; q < n_params; ++q )
            {
                REQUIRE(jac[static_cast<std::size_t>(q)].rows() == x.rows());
                REQUIRE(jac[static_cast<std::size_t>(q)].cols()
                        == static_cast<Eigen::Index>(modes.size()));
                Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_params);
                direction(q) = 1.0;
                CHECK(jac[static_cast<std::size_t>(q)]
                      == jvp_feature(theta_hat, direction, x, modes, mu0, mu_mode));
            }
        }
    }
}

TEST_CASE("one FeatureAt answers repeatedly without disturbing itself")
{
    // The VarPro order is values -> linear solve -> derivative at the SAME
    // parameters, so asking for a derivative after the values must leave the
    // values untouched, and a second ask must return the same numbers.
    std::mt19937 gen(4);
    const int dim = 3;
    const std::vector<Mode> modes = some_modes(dim, 2, 1);
    const Eigen::VectorXd mu0 = random_vector(dim, gen);
    const Eigen::MatrixXd x = test_helpers::uniform_points(9, dim, gen, -1.5, 1.5);

    for ( MuMode mu_mode : kMuModes )
    {
        const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
        const int n_params = theta_hat_size(dim, mu_mode);
        const Eigen::VectorXd dtheta_hat = random_vector(n_params, gen);
        const Eigen::MatrixXd w = test_helpers::uniform_points(
            9, static_cast<int>(modes.size()), gen, -1.0, 1.0);

        FeatureAt at(theta_hat, x, modes, mu0, mu_mode);
        const Eigen::MatrixXd values = at.values();
        const Eigen::MatrixXd from_vjp = at.vjp(w);

        CHECK(at.values() == values);
        CHECK(at.vjp(w) == from_vjp);
        CHECK(at.jvp(dtheta_hat) == at.jvp(dtheta_hat));
        CHECK(at.values() == values);

        // and it agrees with the throwaway-object wrappers
        CHECK(values == eval_feature(theta_hat, x, modes, mu0, mu_mode));
        CHECK(from_vjp == vjp_feature(theta_hat, x, w, modes, mu0, mu_mode));

        // the geometry it was built on is available without decoding again
        CHECK(at.u() == eval_T(theta_hat, x, mu0, mu_mode));
        CHECK(at.num_params() == n_params);
        CHECK(at.num_modes() == modes.size());
    }
}

TEST_CASE("an empty mode set is a well-formed feature basis")
{
    std::mt19937 gen(5);
    const int dim = 2;
    const std::vector<Mode> none;
    const Eigen::VectorXd mu0 = random_vector(dim, gen);
    const Eigen::MatrixXd x = test_helpers::uniform_points(6, dim, gen, -1.0, 1.0);

    for ( MuMode mu_mode : kMuModes )
    {
        const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
        const int n_params = theta_hat_size(dim, mu_mode);
        FeatureAt at(theta_hat, x, none, mu0, mu_mode);

        CHECK(at.values().rows() == x.rows());
        CHECK(at.values().cols() == 0);

        const Eigen::MatrixXd rows = at.vjp(Eigen::MatrixXd::Zero(x.rows(), 0));
        CHECK(rows.rows() == x.rows());
        CHECK(rows.cols() == n_params);
        CHECK(rows.cwiseAbs().maxCoeff() == 0.0);

        CHECK(at.jac().size() == static_cast<std::size_t>(n_params));
        CHECK(at.jac()[0].cols() == 0);
    }
}
