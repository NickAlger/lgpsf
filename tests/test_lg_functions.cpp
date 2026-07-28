// SPDX-License-Identifier: MIT
//
// Checks on the Laguerre-Gaussian modes: L^2(R^N) orthonormality by exact
// quadrature, the Laguerre recurrence against closed forms, the batched path
// against the one-at-a-time reference, and the derivative pair against finite
// differences and adjoint consistency.
//
// All self-contained: nothing here is compared against a stored reference, so
// the suite cannot drift out of step with the code it tests.

#include <cmath>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/lg_functions.hpp"
#include "test_helpers.hpp"

using lgpsf::LGBasisAt;
using lgpsf::Mode;
using lgpsf::eval_lg;
using lgpsf::eval_lg_nd;
using lgpsf::genlaguerre;
using lgpsf::grad_eval_lg_nd;
using lgpsf::grad_lg_basis;
using lgpsf::lg_norm;
using lgpsf::max_degree;
using lgpsf::modes_up_to_level;
using lgpsf::num_harmonics;
using lgpsf::vjp_lg_basis;

namespace {

/// Mode sets spanning the shapes the factorized evaluator must handle: a
/// singleton, a pure-radial family (one shell, many p), a pure-angular family
/// (one p, many shells), complete shells, and a list with duplicates in
/// non-canonical order.
std::vector<std::vector<Mode>> mode_sets( int dim )
{
    std::vector<std::vector<Mode>> sets;
    sets.push_back({Mode{0, 0, 0}});

    std::vector<Mode> radial;
    for ( int p = 0; p < 6; ++p )
    {
        radial.push_back(Mode{p, 0, 0});
    }
    sets.push_back(radial);

    std::vector<Mode> angular;
    for ( int ell = 0; ell < 4; ++ell )
    {
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            angular.push_back(Mode{0, ell, m});
        }
    }
    sets.push_back(angular);

    sets.push_back(modes_up_to_level(dim, 4));
    sets.push_back(modes_up_to_level(dim, 6));

    std::vector<Mode> scrambled = modes_up_to_level(dim, 3);
    std::reverse(scrambled.begin(), scrambled.end());
    std::vector<Mode> with_duplicates = scrambled;
    with_duplicates.insert(with_duplicates.end(), scrambled.begin(),
                           scrambled.begin() + std::min<std::size_t>(3, scrambled.size()));
    sets.push_back(with_duplicates);
    return sets;
}

} // end anonymous namespace

TEST_CASE("genlaguerre matches the textbook low-order closed forms")
{
    std::mt19937 gen(0);
    const Eigen::VectorXd x =
        test_helpers::uniform_points(50, 1, gen, 0.0, 6.0).col(0);

    for ( double alpha : {0.0, 0.5, 1.0, 2.5, 4.0} )
    {
        std::vector<Eigen::VectorXd> closed;
        closed.push_back(Eigen::VectorXd::Ones(x.size()));
        closed.push_back(Eigen::VectorXd((1.0 + alpha) - x.array()));
        closed.push_back(Eigen::VectorXd(
            x.array().square() / 2.0 - (alpha + 2.0) * x.array()
            + (alpha + 1.0) * (alpha + 2.0) / 2.0));
        closed.push_back(Eigen::VectorXd(
            -x.array().cube() / 6.0
            + (alpha + 3.0) * x.array().square() / 2.0
            - (alpha + 2.0) * (alpha + 3.0) * x.array() / 2.0
            + (alpha + 1.0) * (alpha + 2.0) * (alpha + 3.0) / 6.0));

        for ( int p = 0; p < static_cast<int>(closed.size()); ++p )
        {
            const Eigen::VectorXd got = genlaguerre(p, alpha, x);
            const double scale = std::max(1.0, closed[p].cwiseAbs().maxCoeff());
            CAPTURE(alpha);
            CAPTURE(p);
            REQUIRE((got - closed[p]).cwiseAbs().maxCoeff() / scale < 1e-12);
        }
    }
}

TEST_CASE("the Gauss-Hermite rule is exact on monomials")
{
    // Characterizes the quadrature the orthonormality test below rests on, so
    // that test's tolerance is grounded rather than guessed. The rule is exact
    // in exact arithmetic for degree <= 2n-1; in floating point its accuracy is
    // limited by the eigensolver's nodes, and high-degree monomials at the
    // outer nodes amplify any node perturbation.
    for ( int n : {8, 12, 16} )
    {
        const auto [nodes, weights] = test_helpers::gauss_hermite(n);
        double worst = 0.0;
        for ( int a = 0; a <= 2 * n - 1; ++a )
        {
            double got = 0.0;
            double magnitude = 0.0;
            for ( int i = 0; i < n; ++i )
            {
                const double term = weights(i) * std::pow(nodes(i), a);
                got += term;
                magnitude += std::abs(term);
            }
            const double want = test_helpers::gaussian_moment_1d(a);
            // Scale by the magnitude actually being cancelled, not by the
            // answer: every ODD moment is exactly zero, reached by cancelling
            // terms as large as ~1e7 at n = 16, so dividing by |want| (or by
            // 1) would measure the cancellation depth rather than the rule.
            const double scale = std::max({std::abs(want), magnitude, 1e-300});
            worst = std::max(worst, std::abs(got - want) / scale);
        }
        MESSAGE("n=" << n << " nodes: worst moment error, relative to the "
                << "magnitude cancelled: " << worst);
        CAPTURE(n);
        // Observed: 2.4e-14 at n=8, 3.1e-13 at n=12, 5.5e-12 at n=16 -- the
        // rule degrades roughly an order of magnitude per 4 nodes, as the
        // Golub-Welsch eigenproblem conditions worse and the outer nodes reach
        // further. Bound tightly at the n the orthonormality test below
        // actually uses, loosely beyond it.
        REQUIRE(worst < ( n <= 12 ? 1e-12 : 1e-10 ));
    }
}

TEST_CASE("the modes are orthonormal in L^2(R^N)")
{
    // int psi_i psi_j du = delta_ij: the defining property of the LG basis,
    // and the one check that exercises the harmonic table, the Laguerre
    // recurrence, the normalization constant and the alpha = ell + N/2 - 1
    // bookkeeping all at once.
    //
    // psi_i psi_j exp(r^2) is a polynomial of total degree level_i + level_j,
    // so tensor-product Gauss-Hermite with enough nodes evaluates the integral
    // EXACTLY (to roundoff) rather than approximately.
    const int n_nodes = 12;
    const std::vector<std::pair<int, int>> cases = {{1, 10}, {2, 8}, {3, 6}};

    for ( auto [dim, max_level] : cases )
    {
        REQUIRE(2 * n_nodes - 1 >= 2 * max_level); // exactness precondition

        auto [grid, weights] = test_helpers::gauss_hermite_grid(n_nodes, dim);
        // psi_i psi_j carries exp(-r^2) and the quadrature weight supplies it
        // too, so divide it back out to leave the polynomial the rule integrates
        const Eigen::VectorXd scaled_weights =
            weights.array() * grid.rowwise().squaredNorm().array().exp();

        const std::vector<Mode> modes = modes_up_to_level(dim, max_level);
        const Eigen::MatrixXd values = LGBasisAt(modes, grid).values();

        const Eigen::MatrixXd gram =
            values.transpose()
            * (values.array().colwise() * scaled_weights.array()).matrix();
        const Eigen::MatrixXd error =
            gram - Eigen::MatrixXd::Identity(modes.size(), modes.size());

        CAPTURE(dim);
        CAPTURE(modes.size());
        const double worst = error.cwiseAbs().maxCoeff();
        MESSAGE("N=" << dim << ": " << modes.size()
                << " modes orthonormal to " << worst);
        // ~1e-13 observed. The floor here is the quadrature nodes, not the
        // basis: the rule itself is only accurate to ~1e-14 relative to the
        // magnitudes it cancels (see the test above), and the deepest modes
        // reach ~1e3-1e4 at the outer nodes before cancelling to delta_ij.
        REQUIRE(worst < 1e-9);
    }
}

TEST_CASE("the batched path matches the one-at-a-time reference exactly")
{
    // Every shared quantity is the same expression evaluated once instead of
    // many times, with no reassociation, so equality is the contract -- a
    // tolerance here would hide exactly the drift this test exists to catch.
    std::mt19937 gen(4);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        std::vector<Eigen::MatrixXd> batches = {
            Eigen::MatrixXd::Zero(1, dim),
            test_helpers::randn_points(1, dim, gen),
            test_helpers::randn_points(9, dim, gen),
            test_helpers::randn_points(50, dim, gen, 3.0),
        };
        for ( const std::vector<Mode>& modes : mode_sets(dim) )
        {
            for ( const Eigen::MatrixXd& u : batches )
            {
                LGBasisAt at(modes, u);
                const Eigen::MatrixXd values = at.values();
                const std::vector<Eigen::MatrixXd> gradients = at.grad();

                REQUIRE(values.cols() == static_cast<Eigen::Index>(modes.size()));
                REQUIRE(gradients.size() == modes.size());
                for ( std::size_t i = 0; i < modes.size(); ++i )
                {
                    CAPTURE(dim);
                    CAPTURE(i);
                    REQUIRE(values.col(static_cast<Eigen::Index>(i))
                            == eval_lg_nd(modes[i].p, modes[i].ell, modes[i].m, u));
                    REQUIRE(gradients[i]
                            == grad_eval_lg_nd(modes[i].p, modes[i].ell, modes[i].m, u));
                }
            }
        }
    }
}

TEST_CASE("asking for values before or after the gradient gives the same numbers")
{
    // The object shares one harmonic evaluation between the two; the
    // values-then-derivative order is the VarPro pattern, and must not change
    // a single bit relative to asking the other way round.
    std::mt19937 gen(5);
    for ( int dim = 2; dim <= 3; ++dim )
    {
        const Eigen::MatrixXd u = test_helpers::randn_points(17, dim, gen);
        const std::vector<Mode> modes = modes_up_to_level(dim, 5);

        LGBasisAt values_first(modes, u);
        const Eigen::MatrixXd v1 = values_first.values();
        const std::vector<Eigen::MatrixXd> g1 = values_first.grad();

        LGBasisAt grad_first(modes, u);
        const std::vector<Eigen::MatrixXd> g2 = grad_first.grad();
        const Eigen::MatrixXd v2 = grad_first.values();

        REQUIRE(v1 == v2);
        for ( std::size_t i = 0; i < modes.size(); ++i )
        {
            CAPTURE(dim);
            CAPTURE(i);
            REQUIRE(g1[i] == g2[i]);
        }
    }
}

TEST_CASE("the vjp matches the contracted gradient")
{
    // vjp_lg_basis regroups the sum by shell, which reassociates it -- so
    // unlike the other batched paths this one is checked to roundoff.
    std::mt19937 gen(6);
    double worst = 0.0;
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( const std::vector<Mode>& modes : mode_sets(dim) )
        {
            for ( int n_points : {1, 13, 200} )
            {
                const Eigen::MatrixXd u = test_helpers::randn_points(n_points, dim, gen);
                const Eigen::MatrixXd w = test_helpers::randn_points(
                    n_points, static_cast<int>(modes.size()), gen);

                const Eigen::MatrixXd got = vjp_lg_basis(modes, u, w);
                const std::vector<Eigen::MatrixXd> gradients = grad_lg_basis(modes, u);
                Eigen::MatrixXd want = Eigen::MatrixXd::Zero(n_points, dim);
                for ( std::size_t i = 0; i < modes.size(); ++i )
                {
                    want.array() += gradients[i].array().colwise()
                                    * w.col(static_cast<Eigen::Index>(i)).array();
                }

                const double scale = std::max(1e-300, want.cwiseAbs().maxCoeff());
                const double err = (got - want).cwiseAbs().maxCoeff() / scale;
                worst = std::max(worst, err);
                CAPTURE(dim);
                CAPTURE(modes.size());
                CAPTURE(n_points);
                REQUIRE(err < 1e-12);
            }
        }
    }
    MESSAGE("worst relative disagreement vs the per-mode sum: " << worst);
}

TEST_CASE("the vjp is adjoint to the gradient")
{
    // <w, J du> == <vjp(w), du>: the adjoint-consistency half of this
    // project's derivative-testing convention.
    std::mt19937 gen(7);
    double worst = 0.0;
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( const std::vector<Mode>& modes : mode_sets(dim) )
        {
            const int n_points = 17;
            const Eigen::MatrixXd u = test_helpers::randn_points(n_points, dim, gen);
            const Eigen::MatrixXd w = test_helpers::randn_points(
                n_points, static_cast<int>(modes.size()), gen);
            const Eigen::MatrixXd du = test_helpers::randn_points(n_points, dim, gen);

            const std::vector<Eigen::MatrixXd> gradients = grad_lg_basis(modes, u);
            double lhs = 0.0;
            for ( std::size_t i = 0; i < modes.size(); ++i )
            {
                lhs += (gradients[i].cwiseProduct(du)).rowwise().sum().dot(
                    w.col(static_cast<Eigen::Index>(i)));
            }
            const double rhs = vjp_lg_basis(modes, u, w).cwiseProduct(du).sum();

            const double scale = std::max({std::abs(lhs), std::abs(rhs), 1e-300});
            const double err = std::abs(lhs - rhs) / scale;
            worst = std::max(worst, err);
            CAPTURE(dim);
            CAPTURE(modes.size());
            REQUIRE(err < 1e-11);
        }
    }
    MESSAGE("worst adjoint-consistency gap: " << worst);
}

TEST_CASE("mode gradients match central finite differences")
{
    const double step = 1e-5;
    std::mt19937 gen(8);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( int ell = 0; ell < 4; ++ell )
        {
            for ( int p = 0; p < 4; ++p )
            {
                for ( int m = 0; m < num_harmonics(dim, ell); ++m )
                {
                    for ( int trial = 0; trial < 3; ++trial )
                    {
                        Eigen::MatrixXd u = ( trial == 0 )
                            ? Eigen::MatrixXd(Eigen::MatrixXd::Zero(1, dim))
                            : test_helpers::uniform_points(1, dim, gen, -1.5, 1.5);

                        const Eigen::MatrixXd analytic = grad_eval_lg_nd(p, ell, m, u);
                        Eigen::VectorXd fd(dim);
                        for ( int k = 0; k < dim; ++k )
                        {
                            Eigen::MatrixXd up = u;
                            Eigen::MatrixXd um = u;
                            up(0, k) += step;
                            um(0, k) -= step;
                            fd(k) = (eval_lg_nd(p, ell, m, up)(0)
                                     - eval_lg_nd(p, ell, m, um)(0)) / (2 * step);
                        }
                        const double scale = std::max(1.0, fd.cwiseAbs().maxCoeff());
                        CAPTURE(dim);
                        CAPTURE(p);
                        CAPTURE(ell);
                        CAPTURE(m);
                        REQUIRE((analytic.row(0).transpose() - fd).cwiseAbs().maxCoeff()
                                / scale < 1e-6);
                    }
                }
            }
        }
    }
}

TEST_CASE("the N-dimensional path reproduces the 2D reference convention")
{
    // eval_lg is written independently, in the original research note's
    // signed-ell cos/sin convention. For each (p, ell) the N-dimensional pair
    // {m = 0, 1} must span the same 2-space as {+ell, -ell}, with the change
    // of basis an orthogonal matrix that is the SAME at every point -- that is
    // what "indexes the same space, up to which index is which" means.
    std::mt19937 gen(9);
    const Eigen::MatrixXd u = test_helpers::randn_points(64, 2, gen);
    const Eigen::VectorXd u1 = u.col(0);
    const Eigen::VectorXd u2 = u.col(1);

    for ( int p = 0; p < 4; ++p )
    {
        // ell = 0: a single mode, so the two paths must agree outright
        const Eigen::VectorXd nd0 = eval_lg_nd(p, 0, 0, u);
        const Eigen::VectorXd ref0 = eval_lg(p, 0, u1, u2);
        const double scale0 = std::max(1.0, ref0.cwiseAbs().maxCoeff());
        CAPTURE(p);
        REQUIRE((nd0 - ref0).cwiseAbs().maxCoeff() / scale0 < 1e-12);

        for ( int ell = 1; ell < 5; ++ell )
        {
            REQUIRE(num_harmonics(2, ell) == 2);
            Eigen::MatrixXd nd(u.rows(), 2);
            nd.col(0) = eval_lg_nd(p, ell, 0, u);
            nd.col(1) = eval_lg_nd(p, ell, 1, u);

            Eigen::MatrixXd ref(u.rows(), 2);
            ref.col(0) = eval_lg(p, ell, u1, u2);
            ref.col(1) = eval_lg(p, -ell, u1, u2);

            // least-squares change of basis nd -> ref, then check it is
            // orthogonal and that it reproduces ref everywhere
            const Eigen::MatrixXd basis_change =
                nd.colPivHouseholderQr().solve(ref);
            const Eigen::MatrixXd residual = nd * basis_change - ref;
            const double scale = std::max(1.0, ref.cwiseAbs().maxCoeff());

            CAPTURE(p);
            CAPTURE(ell);
            REQUIRE(residual.cwiseAbs().maxCoeff() / scale < 1e-10);
            REQUIRE((basis_change.transpose() * basis_change
                     - Eigen::Matrix2d::Identity()).cwiseAbs().maxCoeff() < 1e-10);
        }
    }
}

TEST_CASE("modes_up_to_level enumerates and validates")
{
    // level ordering and the ell cap
    for ( const Mode& mode : modes_up_to_level(3, 6) )
    {
        CHECK(2 * mode.p + mode.ell <= 6);
    }
    for ( const Mode& mode : modes_up_to_level(3, 10, 2) )
    {
        CHECK(2 * mode.p + mode.ell <= 10);
        CHECK(mode.ell <= 2);
    }
    // nested: each level's set contains the previous one
    for ( int level = 1; level <= 6; ++level )
    {
        const std::vector<Mode> smaller = modes_up_to_level(2, level - 1);
        const std::vector<Mode> larger = modes_up_to_level(2, level);
        CHECK(larger.size() > smaller.size());
        for ( const Mode& mode : smaller )
        {
            CHECK(std::find(larger.begin(), larger.end(), mode) != larger.end());
        }
    }
    // N=1 has no ell >= 2 modes at all
    for ( const Mode& mode : modes_up_to_level(1, 8) )
    {
        CHECK(mode.ell <= 1);
    }
    // past the table it raises rather than truncating silently
    CHECK_THROWS_AS(modes_up_to_level(2, max_degree() + 1), std::invalid_argument);
    CHECK_NOTHROW(modes_up_to_level(2, 100, 2));
}

TEST_CASE("empty mode sets are handled")
{
    const Eigen::MatrixXd u = Eigen::MatrixXd::Zero(7, 2);
    LGBasisAt at({}, u);
    CHECK(at.values().rows() == 7);
    CHECK(at.values().cols() == 0);
    CHECK(at.grad().empty());
    CHECK(at.vjp(Eigen::MatrixXd::Zero(7, 0)).rows() == 7);
    CHECK(at.vjp(Eigen::MatrixXd::Zero(7, 0)).cols() == 2);
}

TEST_CASE("lg_norm is the constant that normalizes the radial factor")
{
    // Independent of the modes: C^2 int_0^inf L_p^alpha(t)^2 t^alpha e^-t dt
    // = 2 ... checked instead via the Laguerre orthogonality relation
    // int_0^inf t^alpha e^-t L_p L_q = Gamma(p+alpha+1)/p! delta_pq, so
    // C_{p} = sqrt(2 p! / Gamma(p+alpha+1)) makes C^2 * that ratio equal 2.
    for ( int dim = 1; dim <= 4; ++dim )
    {
        for ( int ell = 0; ell < 4; ++ell )
        {
            const double alpha = ell + dim / 2.0 - 1.0;
            for ( int p = 0; p < 5; ++p )
            {
                const double c = lg_norm(p, ell, dim);
                const double ratio = std::tgamma(p + alpha + 1.0) / std::tgamma(p + 1.0);
                CAPTURE(dim);
                CAPTURE(ell);
                CAPTURE(p);
                CHECK(std::abs(c * c * ratio - 2.0) < 1e-12 * 2.0);
            }
        }
    }
}
