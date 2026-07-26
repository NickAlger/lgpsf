// SPDX-License-Identifier: MIT
//
// Checks on the whitening layer -- the only place the mass matrices appear.
//
// Two kinds. The usual finite-difference and adjoint-consistency checks on the
// whitened derivative chain, and the end-to-end one: build a row of H
// explicitly from the row model (docs/varpro-whitening-notes.tex eq. 1), apply
// it to a random probe, and check that the whitened quantities satisfy the
// whitened regression (eq. 7) to machine precision. That second one is the
// only check here that involves M1 or the extra basis at all, so it is the
// only one that can see a pure scaling error in either -- exactly the class of
// bug (a missing sqrt(m_rho) on the extra whitening) that the derivation
// turned up in the prototype.
//
// All self-contained -- nothing is compared against a stored reference or
// against the Python prototype.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/whitening.hpp"
#include "test_helpers.hpp"

using lgpsf::Mode;
using lgpsf::MuMode;
using lgpsf::WhitenedBasis;
using lgpsf::WhitenedBasisAt;
using lgpsf::eval_feature;
using lgpsf::num_harmonics;
using lgpsf::theta_hat_size;
using lgpsf::whiten_data;
using lgpsf::whiten_extra;
using lgpsf::whiten_probes;
using lgpsf::whitened_eval_feature;
using lgpsf::whitened_jac_feature;
using lgpsf::whitened_jvp_feature;
using lgpsf::whitened_vjp_feature;

namespace {

constexpr double kFdStep = 1e-6;

const std::vector<MuMode> kMuModes = {MuMode::Pinned, MuMode::Fitted};

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

/// Nonuniform masses throughout, so a mistaken power of M2 cannot hide.
Eigen::VectorXd random_masses( int n, std::mt19937& gen )
{
    return random_vector(n, gen, 0.5, 2.0);
}

} // namespace

TEST_CASE("the whitening transforms are what the derivation says")
{
    std::mt19937 gen(0);
    const int num_points = 9;
    const Eigen::VectorXd m2 = random_masses(num_points, gen);
    const double target_mass = 1.7;

    const Eigen::MatrixXd z = test_helpers::randn_points(num_points, 4, gen);
    const Eigen::MatrixXd z_hat = whiten_probes(z, m2);
    const Eigen::MatrixXd E = test_helpers::randn_points(num_points, 2, gen);
    const Eigen::MatrixXd E_hat = whiten_extra(E, target_mass, m2);
    const Eigen::VectorXd y = random_vector(5, gen);

    for ( Eigen::Index k = 0; k < num_points; ++k )
    {
        for ( Eigen::Index j = 0; j < z.cols(); ++j )
        {
            CHECK(z_hat(k, j) == doctest::Approx(std::sqrt(m2(k)) * z(k, j)));
        }
        // the extra basis takes the INVERSE power of M2 -- it is a discrete
        // correction, not a quadrature object
        for ( Eigen::Index d = 0; d < E.cols(); ++d )
        {
            CHECK(E_hat(k, d)
                  == doctest::Approx(std::sqrt(target_mass) * E(k, d) / std::sqrt(m2(k))));
        }
    }
    CHECK((whiten_data(y, target_mass) - y / std::sqrt(target_mass))
              .cwiseAbs().maxCoeff() == 0.0);

    CHECK_THROWS_AS(whiten_data(y, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(whiten_extra(E, -1.0, m2), std::invalid_argument);
    Eigen::VectorXd bad = m2;
    bad(3) = 0.0;
    CHECK_THROWS_AS(whiten_extra(E, target_mass, bad), std::invalid_argument);
}

TEST_CASE("the whitened JVP and VJP are adjoint")
{
    std::mt19937 gen(1);
    double worst = 0.0;
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim);
        for ( MuMode mu_mode : kMuModes )
        {
            const int num_points = 10;
            const double target_mass = 0.5 + 1.5 * (dim / 4.0);
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
            const int n_params = theta_hat_size(dim, mu_mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(num_points, dim, gen, -1.5, 1.5);
            const Eigen::VectorXd m2 = random_masses(num_points, gen);
            const Eigen::VectorXd dtheta_hat = random_vector(n_params, gen);
            const Eigen::MatrixXd w_hat = test_helpers::uniform_points(
                num_points, static_cast<int>(modes.size()), gen, -1.0, 1.0);

            const Eigen::MatrixXd dphi_hat = whitened_jvp_feature(
                theta_hat, dtheta_hat, x, target_mass, m2, modes, mu0, mu_mode);
            const Eigen::MatrixXd rows = whitened_vjp_feature(
                theta_hat, x, target_mass, m2, w_hat, modes, mu0, mu_mode);
            REQUIRE(rows.cols() == n_params);

            for ( Eigen::Index k = 0; k < num_points; ++k )
            {
                const double lhs = w_hat.row(k).dot(dphi_hat.row(k));
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

TEST_CASE("the whitened JVP matches finite differences")
{
    std::mt19937 gen(2);
    double worst = 0.0;
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim);
        for ( MuMode mu_mode : kMuModes )
        {
            const int num_points = 8;
            const double target_mass = 1.3;
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
            const int n_params = theta_hat_size(dim, mu_mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(num_points, dim, gen, -1.5, 1.5);
            const Eigen::VectorXd m2 = random_masses(num_points, gen);

            for ( int q = 0; q < n_params; ++q )
            {
                Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_params);
                direction(q) = 1.0;
                const Eigen::MatrixXd analytic = whitened_jvp_feature(
                    theta_hat, direction, x, target_mass, m2, modes, mu0, mu_mode);

                Eigen::VectorXd plus = theta_hat, minus = theta_hat;
                plus(q) += kFdStep;
                minus(q) -= kFdStep;
                const Eigen::MatrixXd fd =
                    (whitened_eval_feature(plus, x, target_mass, m2, modes, mu0, mu_mode)
                     - whitened_eval_feature(minus, x, target_mass, m2, modes, mu0,
                                             mu_mode))
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

TEST_CASE("the whitened Jacobian's columns are the directional derivatives")
{
    std::mt19937 gen(3);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim);
        for ( MuMode mu_mode : kMuModes )
        {
            const int num_points = 10;
            const double target_mass = 0.8;
            const Eigen::VectorXd mu0 = random_vector(dim, gen);
            const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
            const int n_params = theta_hat_size(dim, mu_mode);
            const Eigen::MatrixXd x =
                test_helpers::uniform_points(num_points, dim, gen, -1.5, 1.5);
            const Eigen::VectorXd m2 = random_masses(num_points, gen);

            const std::vector<Eigen::MatrixXd> jac = whitened_jac_feature(
                theta_hat, x, target_mass, m2, modes, mu0, mu_mode);
            REQUIRE(jac.size() == static_cast<std::size_t>(n_params));
            for ( int q = 0; q < n_params; ++q )
            {
                Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_params);
                direction(q) = 1.0;
                CHECK(jac[static_cast<std::size_t>(q)]
                      == whitened_jvp_feature(theta_hat, direction, x, target_mass,
                                              m2, modes, mu0, mu_mode));
            }
        }
    }
}

TEST_CASE("WhitenedBasisAt shares work without changing any number")
{
    // One object serves values, jvp, jac and vjp from a single pullback and a
    // single LG evaluation. Sharing work is not licence to change a value, so
    // this is bit equality against the throwaway-object wrappers, not a
    // tolerance.
    std::mt19937 gen(4);
    const int dim = 2;
    const int num_points = 8;
    const std::vector<Mode> modes = some_modes(dim, 1, 1);
    const Eigen::VectorXd mu0 = random_vector(dim, gen);
    const Eigen::MatrixXd x =
        test_helpers::uniform_points(num_points, dim, gen, -1.5, 1.5);
    const Eigen::VectorXd m2 = random_masses(num_points, gen);
    const double target_mass = 1.7;
    const Eigen::MatrixXd w_hat = test_helpers::randn_points(
        num_points, static_cast<int>(modes.size()), gen);

    for ( MuMode mu_mode : kMuModes )
    {
        const Eigen::VectorXd theta_hat = random_theta_hat(dim, mu_mode, gen);
        const Eigen::VectorXd dtheta_hat =
            random_vector(theta_hat_size(dim, mu_mode), gen);

        const WhitenedBasis basis(x, target_mass, m2, modes, mu0, mu_mode);
        WhitenedBasisAt at = basis(theta_hat);

        CHECK(at.values()
              == whitened_eval_feature(theta_hat, x, target_mass, m2, modes, mu0,
                                       mu_mode));
        CHECK(at.vjp(w_hat)
              == whitened_vjp_feature(theta_hat, x, target_mass, m2, w_hat, modes,
                                      mu0, mu_mode));
        CHECK(at.jvp(dtheta_hat)
              == whitened_jvp_feature(theta_hat, dtheta_hat, x, target_mass, m2,
                                      modes, mu0, mu_mode));
        const std::vector<Eigen::MatrixXd> jac = at.jac();
        const std::vector<Eigen::MatrixXd> reference =
            whitened_jac_feature(theta_hat, x, target_mass, m2, modes, mu0, mu_mode);
        REQUIRE(jac.size() == reference.size());
        for ( std::size_t q = 0; q < jac.size(); ++q )
        {
            CHECK(jac[q] == reference[q]);
        }

        // asking twice must give identical contents, and asking for values
        // after a derivative must not disturb them
        const Eigen::MatrixXd values = at.values();
        CHECK(at.vjp(w_hat) == at.vjp(w_hat));
        CHECK(at.values() == values);

        CHECK(basis.num_params() == theta_hat_size(dim, mu_mode));
        CHECK(basis.num_modes() == modes.size());
    }
}

TEST_CASE("the whitened regression reproduces the row model")
{
    // H[rho, j] = m_rho m_j sum_i c_i phi_i(x_j) + m_rho sum_d s_d E[j, d],
    // y = H[rho, :] . z for one probe realization. The whitened quantities
    // must then satisfy y_hat = c . (phi_hat^T z_hat) + s . (E_hat^T z_hat)
    // to machine precision -- with M1 != M2 throughout, so the two different
    // masses cannot be conflated.
    std::mt19937 gen(5);
    double worst = 0.0;
    for ( int dim = 1; dim <= 3; ++dim )
    {
        const std::vector<Mode> modes = some_modes(dim, 2, 1);
        const int n_modes = static_cast<int>(modes.size());
        for ( MuMode mu_mode : kMuModes )
        {
            for ( const std::vector<int>& extra_positions :
                  {std::vector<int>{0}, std::vector<int>{0, 3}} )
            {
                const int num_points = 12;
                const int n_extra = static_cast<int>(extra_positions.size());
                const double target_mass = 0.5 + 1.5 * (dim / 3.0);
                const Eigen::VectorXd mu0 = random_vector(dim, gen);
                const Eigen::VectorXd theta_hat =
                    random_theta_hat(dim, mu_mode, gen);
                const Eigen::MatrixXd x =
                    test_helpers::uniform_points(num_points, dim, gen, -1.5, 1.5);
                const Eigen::VectorXd m2 = random_masses(num_points, gen);

                // extra basis: one-hot indicators at the given points
                Eigen::MatrixXd E = Eigen::MatrixXd::Zero(num_points, n_extra);
                for ( int d = 0; d < n_extra; ++d )
                {
                    E(extra_positions[static_cast<std::size_t>(d)], d) = 1.0;
                }

                const Eigen::VectorXd c = random_vector(n_modes, gen);
                const Eigen::VectorXd s = random_vector(n_extra, gen);

                const Eigen::MatrixXd phi =
                    eval_feature(theta_hat, x, modes, mu0, mu_mode);
                const Eigen::VectorXd smooth =
                    (target_mass * m2.array() * (phi * c).array()).matrix();
                const Eigen::VectorXd extra = target_mass * (E * s);
                const Eigen::VectorXd h_row = smooth + extra;

                const Eigen::VectorXd z = test_helpers::randn_points(num_points, 1, gen);
                const double y = h_row.dot(z);

                const double y_hat = whiten_data(
                    Eigen::VectorXd::Constant(1, y), target_mass)(0);
                const Eigen::VectorXd z_hat = whiten_probes(z, m2).col(0);
                const Eigen::MatrixXd phi_hat = whitened_eval_feature(
                    theta_hat, x, target_mass, m2, modes, mu0, mu_mode);
                const Eigen::MatrixXd e_hat = whiten_extra(E, target_mass, m2);

                const double predicted = c.dot(phi_hat.transpose() * z_hat)
                                         + s.dot(e_hat.transpose() * z_hat);

                const double err =
                    std::abs(predicted - y_hat) / std::max(1.0, std::abs(y_hat));
                worst = std::max(worst, err);
                CHECK(err < 1e-12);
            }
        }
    }
    MESSAGE("worst relative whitened-regression error: " << worst);
}
