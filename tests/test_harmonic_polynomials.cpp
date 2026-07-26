// SPDX-License-Identifier: MIT
//
// Self-contained checks on the harmonic polynomials and their generated
// table. Every assertion is intrinsic: a property the objects must have as
// mathematics, checkable from this header alone. Nothing is compared against
// a stored reference or against the Python prototype, so these survive
// further C++-side development.

#include <cmath>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/harmonic_polynomials.hpp"
#include "test_helpers.hpp"

using lgpsf::HarmonicPolynomial;
using lgpsf::Shell;
using lgpsf::eval_harmonic;
using lgpsf::grad_harmonic;
using lgpsf::harmonic_basis;
using lgpsf::harmonic_terms;
using lgpsf::max_degree;
using lgpsf::max_dimension;
using lgpsf::num_harmonics;

namespace {

/// Every (N, ell) with at least one mode.
std::vector<std::pair<int, int>> nonempty_shells()
{
    std::vector<std::pair<int, int>> out;
    for ( int dim = 1; dim <= max_dimension(); ++dim )
    {
        for ( int ell = 0; ell <= max_degree(); ++ell )
        {
            if ( num_harmonics(dim, ell) > 0 )
            {
                out.emplace_back(dim, ell);
            }
        }
    }
    return out;
}

std::vector<int> exponent_vector( const HarmonicPolynomial& poly, int term )
{
    return std::vector<int>(poly.exponents + static_cast<std::ptrdiff_t>(term) * poly.dim,
                            poly.exponents + static_cast<std::ptrdiff_t>(term + 1) * poly.dim);
}

double binomial( int n, int k )
{
    if ( k < 0 || n < 0 || k > n )
    {
        return 0.0;
    }
    double out = 1.0;
    for ( int i = 0; i < k; ++i )
    {
        out = out * (n - i) / (i + 1);
    }
    return std::round(out);
}

} // end anonymous namespace

TEST_CASE("harmonic term lists are well formed")
{
    for ( auto [dim, ell] : nonempty_shells() )
    {
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            const HarmonicPolynomial poly = harmonic_terms(dim, ell, m);
            CAPTURE(dim);
            CAPTURE(ell);
            CAPTURE(m);
            REQUIRE(poly.dim == dim);
            REQUIRE(poly.degree == ell);
            // never empty: a harmonic basis polynomial is never the zero
            // polynomial, and the evaluator's output shape relies on it
            REQUIRE(poly.num_terms > 0);

            std::set<std::vector<int>> seen;
            std::vector<int> previous;
            for ( int t = 0; t < poly.num_terms; ++t )
            {
                const std::vector<int> alpha = exponent_vector(poly, t);
                int degree_sum = 0;
                for ( int a : alpha )
                {
                    REQUIRE(a >= 0);
                    degree_sum += a;
                }
                REQUIRE(degree_sum == ell);
                // stored term <=> nonzero coefficient is the storage invariant
                REQUIRE(poly.coefficients[t] != 0.0);
                REQUIRE(seen.insert(alpha).second);
                if ( t > 0 )
                {
                    REQUIRE(previous > alpha); // lexicographically descending
                }
                previous = alpha;
            }
        }
    }
}

TEST_CASE("num_harmonics matches the closed-form shell dimension")
{
    // dim H_ell(R^N) = C(N+ell-1, N-1) - C(N+ell-3, N-1): an independent check
    // that the generator's Gram-Schmidt lost no basis vector and double-counted
    // none.
    for ( int dim = 1; dim <= max_dimension(); ++dim )
    {
        for ( int ell = 0; ell <= max_degree(); ++ell )
        {
            const double expected = binomial(dim + ell - 1, dim - 1)
                                    - binomial(dim + ell - 3, dim - 1);
            CAPTURE(dim);
            CAPTURE(ell);
            REQUIRE(num_harmonics(dim, ell) == static_cast<int>(expected));
        }
    }
}

TEST_CASE("each polynomial's support lies in one exponent parity class")
{
    // Why the stored form is sparse: harmonic projection only shifts exponents
    // by +-2, and the Gaussian moment matrix is block diagonal across parity
    // classes, so Gram-Schmidt cannot mix them.
    for ( auto [dim, ell] : nonempty_shells() )
    {
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            const HarmonicPolynomial poly = harmonic_terms(dim, ell, m);
            std::set<std::vector<int>> classes;
            for ( int t = 0; t < poly.num_terms; ++t )
            {
                std::vector<int> parity = exponent_vector(poly, t);
                for ( int& a : parity )
                {
                    a %= 2;
                }
                classes.insert(parity);
            }
            CAPTURE(dim);
            CAPTURE(ell);
            CAPTURE(m);
            REQUIRE(classes.size() == 1u);
        }
    }
}

TEST_CASE("the basis is a Gram-Schmidt staircase within each parity class")
{
    for ( auto [dim, ell] : nonempty_shells() )
    {
        std::map<std::vector<int>, std::vector<std::vector<int>>> pivots_by_class;
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            const HarmonicPolynomial poly = harmonic_terms(dim, ell, m);
            const std::vector<int> pivot = exponent_vector(poly, 0);

            std::set<std::vector<int>> support;
            for ( int t = 0; t < poly.num_terms; ++t )
            {
                support.insert(exponent_vector(poly, t));
            }

            std::vector<int> parity = pivot;
            for ( int& a : parity )
            {
                a %= 2;
            }
            for ( const std::vector<int>& earlier : pivots_by_class[parity] )
            {
                CAPTURE(dim);
                CAPTURE(ell);
                CAPTURE(m);
                // vanishes on the leading monomial of every earlier polynomial
                REQUIRE(support.count(earlier) == 0u);
            }
            pivots_by_class[parity].push_back(pivot);
        }
    }
}

TEST_CASE("every polynomial is harmonic")
{
    // Delta Y = 0, checked symbolically on the stored terms.
    double worst = 0.0;
    for ( auto [dim, ell] : nonempty_shells() )
    {
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            const HarmonicPolynomial poly = harmonic_terms(dim, ell, m);
            std::map<std::vector<int>, double> laplacian;
            double max_coeff = 0.0;
            for ( int t = 0; t < poly.num_terms; ++t )
            {
                const std::vector<int> alpha = exponent_vector(poly, t);
                const double coeff = poly.coefficients[t];
                max_coeff = std::max(max_coeff, std::abs(coeff));
                for ( int k = 0; k < dim; ++k )
                {
                    if ( alpha[k] >= 2 )
                    {
                        std::vector<int> beta = alpha;
                        beta[k] -= 2;
                        laplacian[beta] += coeff * alpha[k] * (alpha[k] - 1);
                    }
                }
            }
            if ( laplacian.empty() )
            {
                continue; // ell < 2: the Laplacian vanishes term by term
            }
            double max_residual = 0.0;
            for ( const auto& [beta, value] : laplacian )
            {
                (void)beta;
                max_residual = std::max(max_residual, std::abs(value));
            }
            const double scale = max_coeff * std::max(1, ell * (ell - 1));
            CAPTURE(dim);
            CAPTURE(ell);
            CAPTURE(m);
            worst = std::max(worst, max_residual / scale);
            REQUIRE(max_residual / scale < 1e-14);
        }
    }
    MESSAGE("worst relative |Delta Y| over the table: " << worst);
}

TEST_CASE("the polynomials are homogeneous of degree ell")
{
    std::mt19937 gen(0);
    for ( auto [dim, ell] : nonempty_shells() )
    {
        const Eigen::MatrixXd u = test_helpers::randn_points(5, dim, gen);
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            const Eigen::VectorXd base = eval_harmonic(ell, m, u);
            for ( double t : {0.5, 1.7, -1.3} )
            {
                const Eigen::VectorXd scaled = eval_harmonic(ell, m, t * u);
                const Eigen::VectorXd expected = std::pow(t, ell) * base;
                const double scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
                CAPTURE(dim);
                CAPTURE(ell);
                CAPTURE(m);
                CAPTURE(t);
                REQUIRE((scaled - expected).cwiseAbs().maxCoeff() / scale < 1e-12);
            }
        }
    }
}

TEST_CASE("each shell is orthonormal on the sphere")
{
    // For homogeneous degree-ell P, Q:
    //   int_{R^N} P Q exp(-r^2) dx = <P, Q>_{S^(N-1)} Gamma(ell + N/2) / 2,
    // and the left side is a finite sum of Gaussian monomial moments -- so the
    // Gram matrix is computed exactly (up to roundoff), not sampled.
    double worst_diag = 0.0;
    double worst_off = 0.0;
    for ( auto [dim, ell] : nonempty_shells() )
    {
        const int count = num_harmonics(dim, ell);
        const double radial = std::tgamma(ell + dim / 2.0) / 2.0;
        std::vector<HarmonicPolynomial> polys;
        for ( int m = 0; m < count; ++m )
        {
            polys.push_back(harmonic_terms(dim, ell, m));
        }

        for ( int i = 0; i < count; ++i )
        {
            for ( int j = i; j < count; ++j )
            {
                double total = 0.0;
                for ( int s = 0; s < polys[i].num_terms; ++s )
                {
                    const std::vector<int> a = exponent_vector(polys[i], s);
                    for ( int t = 0; t < polys[j].num_terms; ++t )
                    {
                        const std::vector<int> b = exponent_vector(polys[j], t);
                        double moment = 1.0;
                        for ( int k = 0; k < dim; ++k )
                        {
                            moment *= test_helpers::gaussian_moment_1d(a[k] + b[k]);
                            if ( moment == 0.0 )
                            {
                                break;
                            }
                        }
                        if ( moment != 0.0 )
                        {
                            total += polys[i].coefficients[s]
                                     * polys[j].coefficients[t] * moment;
                        }
                    }
                }
                const double value = total / radial;
                CAPTURE(dim);
                CAPTURE(ell);
                CAPTURE(i);
                CAPTURE(j);
                if ( i == j )
                {
                    worst_diag = std::max(worst_diag, std::abs(value - 1.0));
                    REQUIRE(std::abs(value - 1.0) < 1e-11);
                }
                else
                {
                    worst_off = std::max(worst_off, std::abs(value));
                    REQUIRE(std::abs(value) < 1e-11);
                }
            }
        }
    }
    MESSAGE("sphere Gram: max |diag - 1| = " << worst_diag
            << ", max |off-diag| = " << worst_off);
}

TEST_CASE("gradients match central finite differences")
{
    const double step = 1e-5;
    std::mt19937 gen(1);
    for ( auto [dim, ell] : nonempty_shells() )
    {
        if ( ell > 6 )
        {
            continue; // deeper shells are covered by the identities above; FD
                      // there is limited by monomial-basis cancellation
        }
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            for ( int trial = 0; trial < 2; ++trial )
            {
                Eigen::MatrixXd u = ( trial == 0 )
                    ? Eigen::MatrixXd(Eigen::MatrixXd::Zero(1, dim))
                    : test_helpers::uniform_points(1, dim, gen, -1.5, 1.5);

                const auto [value, analytic] = grad_harmonic(ell, m, u);
                (void)value;
                Eigen::VectorXd fd(dim);
                for ( int k = 0; k < dim; ++k )
                {
                    Eigen::MatrixXd up = u;
                    Eigen::MatrixXd um = u;
                    up(0, k) += step;
                    um(0, k) -= step;
                    fd(k) = (eval_harmonic(ell, m, up)(0)
                             - eval_harmonic(ell, m, um)(0)) / (2 * step);
                }
                const double scale = std::max(1.0, fd.cwiseAbs().maxCoeff());
                CAPTURE(dim);
                CAPTURE(ell);
                CAPTURE(m);
                REQUIRE((analytic.row(0).transpose() - fd).cwiseAbs().maxCoeff()
                        / scale < 1e-6);
            }
        }
    }
}

TEST_CASE("grad_harmonic's value agrees with eval_harmonic exactly")
{
    std::mt19937 gen(2);
    for ( auto [dim, ell] : nonempty_shells() )
    {
        const Eigen::MatrixXd u = test_helpers::randn_points(6, dim, gen);
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            const Eigen::VectorXd only = eval_harmonic(ell, m, u);
            const auto [with_grad, unused] = grad_harmonic(ell, m, u);
            (void)unused;
            CAPTURE(dim);
            CAPTURE(ell);
            CAPTURE(m);
            REQUIRE(only == with_grad);
        }
    }
}

TEST_CASE("batched harmonic_basis matches the one-at-a-time path exactly")
{
    std::mt19937 gen(3);
    for ( int dim = 1; dim <= max_dimension(); ++dim )
    {
        std::vector<Shell> all;
        for ( int ell = 0; ell <= max_degree(); ++ell )
        {
            for ( int m = 0; m < num_harmonics(dim, ell); ++m )
            {
                all.push_back(Shell{ell, m});
            }
        }
        std::vector<std::vector<Shell>> lists = {
            {all.front()},
            {all.front(), all.front()},                  // duplicates
            std::vector<Shell>(all.rbegin(), all.rbegin() + std::min<size_t>(8, all.size())),
            all,
        };
        const Eigen::MatrixXd u = test_helpers::randn_points(9, dim, gen);
        for ( const std::vector<Shell>& shells : lists )
        {
            const lgpsf::HarmonicBasis basis = harmonic_basis(shells, u, true);
            for ( size_t i = 0; i < shells.size(); ++i )
            {
                const auto [value, gradient] =
                    grad_harmonic(shells[i].ell, shells[i].m, u);
                CAPTURE(dim);
                CAPTURE(i);
                REQUIRE(basis.values.col(static_cast<int>(i)) == value);
                REQUIRE(basis.gradients[i] == gradient);
            }
        }
    }
}

TEST_CASE("known_values skips the value pass without changing the gradient")
{
    std::mt19937 gen(4);
    const int dim = 3;
    std::vector<Shell> shells;
    for ( int ell = 0; ell <= 3; ++ell )
    {
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            shells.push_back(Shell{ell, m});
        }
    }
    const Eigen::MatrixXd u = test_helpers::randn_points(11, dim, gen);

    const lgpsf::HarmonicBasis full = harmonic_basis(shells, u, true);
    const lgpsf::HarmonicBasis values_only = harmonic_basis(shells, u, false);
    const lgpsf::HarmonicBasis reused =
        harmonic_basis(shells, u, true, &values_only.values);

    REQUIRE(values_only.values == full.values);
    REQUIRE(reused.values == full.values);
    for ( size_t i = 0; i < shells.size(); ++i )
    {
        REQUIRE(reused.gradients[i] == full.gradients[i]);
    }
    REQUIRE(values_only.gradients.empty());
}

TEST_CASE("out-of-range requests throw")
{
    CHECK_THROWS_AS(num_harmonics(max_dimension() + 1, 0), std::invalid_argument);
    CHECK_THROWS_AS(num_harmonics(1, max_degree() + 1), std::invalid_argument);
    CHECK_THROWS_AS(num_harmonics(0, 0), std::invalid_argument);
    CHECK_THROWS_AS(num_harmonics(2, -1), std::invalid_argument);
    CHECK_THROWS_AS(harmonic_terms(3, 2, -1), std::invalid_argument);
    CHECK_THROWS_AS(harmonic_terms(3, 2, num_harmonics(3, 2)), std::invalid_argument);

    // N = 1, ell >= 2 is empty but legal to ask about
    CHECK(num_harmonics(1, 5) == 0);
}
