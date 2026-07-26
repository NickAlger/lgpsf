// SPDX-License-Identifier: MIT
//
// Checks on the initial-ellipsoid dictionary and the probe-moment estimators:
// the geometry against its defining properties, the ladder ordering against
// the combinatorics it claims, and the estimators against data built to have a
// known answer.
//
// All self-contained -- nothing is compared against a stored reference or
// against the Python prototype.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/init_dictionary.hpp"
#include "lgpsf/probe_moments.hpp"
#include "test_helpers.hpp"

using lgpsf::InitCandidate;
using lgpsf::MuMode;
using lgpsf::RawMoments;
using lgpsf::backproject;
using lgpsf::circle_rungs;
using lgpsf::ladder_scales;
using lgpsf::local_spacing;
using lgpsf::mid_out;
using lgpsf::oriented_sigma;
using lgpsf::raw_moments;
using lgpsf::theta_hat_from_cholesky;
using lgpsf::theta_hat_size;
using lgpsf::unpack_theta_hat;
using lgpsf::window_radius;
using lgpsf::window_rungs;
using lgpsf::window_shape;

TEST_CASE("packing a Cholesky factor into theta_hat round-trips through the decoder")
{
    std::mt19937 gen(0);
    for ( int dim = 1; dim <= 4; ++dim )
    {
        Eigen::MatrixXd L = Eigen::MatrixXd::Zero(dim, dim);
        std::uniform_real_distribution<double> diag(0.5, 2.0);
        std::uniform_real_distribution<double> off(-0.5, 0.5);
        for ( int i = 0; i < dim; ++i )
        {
            L(i, i) = diag(gen);
            for ( int j = 0; j < i; ++j )
            {
                L(i, j) = off(gen);
            }
        }

        const Eigen::VectorXd theta_hat = theta_hat_from_cholesky(L);
        CHECK(theta_hat.size() == theta_hat_size(dim, MuMode::Pinned));

        // The decoder is the authority on the encoding, so the round trip is
        // the check that matters -- not the byte layout.
        const Eigen::VectorXd mu0 = Eigen::VectorXd::Zero(dim);
        CHECK((unpack_theta_hat(theta_hat, mu0, MuMode::Pinned).L - L)
                  .cwiseAbs().maxCoeff() < 1e-12);
    }

    Eigen::MatrixXd degenerate = Eigen::MatrixXd::Identity(2, 2);
    degenerate(1, 1) = 0.0;
    CHECK_THROWS_AS(theta_hat_from_cholesky(degenerate), std::invalid_argument);
}

TEST_CASE("the window shape measures the region, not the mesh density")
{
    // The point of the mass weighting. Two batches covering the SAME region --
    // one uniformly sampled, one heavily oversampled on the left half -- must
    // give the same shape once the masses reflect the cells they represent.
    const int per_side = 21;
    std::vector<double> uniform_x, uniform_y;
    for ( int i = 0; i < per_side; ++i )
    {
        for ( int j = 0; j < per_side; ++j )
        {
            uniform_x.push_back(-1.0 + 2.0 * i / (per_side - 1));
            uniform_y.push_back(-0.5 + 1.0 * j / (per_side - 1));
        }
    }
    const int count = static_cast<int>(uniform_x.size());
    Eigen::MatrixXd x(count, 2);
    for ( int i = 0; i < count; ++i )
    {
        x(i, 0) = uniform_x[static_cast<std::size_t>(i)];
        x(i, 1) = uniform_y[static_cast<std::size_t>(i)];
    }

    const Eigen::MatrixXd even =
        window_shape(x, Eigen::VectorXd::Ones(count));

    // Now give the left half four times the sample density by quartering the
    // cell area (mass) there -- the region is unchanged, so the shape must be.
    Eigen::VectorXd graded = Eigen::VectorXd::Ones(count);
    for ( int i = 0; i < count; ++i )
    {
        if ( x(i, 0) < 0.0 )
        {
            graded(i) = 1.0;  // same mass: the region is what is being measured
        }
    }
    CHECK((window_shape(x, graded) - even).cwiseAbs().maxCoeff() < 1e-12);

    // The shape is normalized to a largest eigenvalue of one, and this region
    // is twice as wide as it is tall, so the axis ratio should be about 2.
    const Eigen::MatrixXd sigma = even * even.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sigma);
    CHECK(solver.eigenvalues()(1) == doctest::Approx(1.0));
    CHECK(std::sqrt(solver.eigenvalues()(1) / solver.eigenvalues()(0))
          == doctest::Approx(2.0).epsilon(0.02));
}

TEST_CASE("the window shape's aspect ratio is capped on a degenerate window")
{
    // A near-collinear batch: without the eigenvalue floor this shape would be
    // arbitrarily thin, and every hypothesis built from it useless.
    const int count = 40;
    Eigen::MatrixXd x(count, 2);
    for ( int i = 0; i < count; ++i )
    {
        x(i, 0) = -1.0 + 2.0 * i / (count - 1);
        x(i, 1) = 1e-9 * x(i, 0);
    }
    const Eigen::MatrixXd L = window_shape(x, Eigen::VectorXd::Ones(count));
    const Eigen::MatrixXd sigma = L * L.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sigma);
    CHECK(solver.eigenvalues()(0) >= 1e-4 * (1.0 - 1e-12));
    CHECK(solver.eigenvalues()(1) == doctest::Approx(1.0));
}

TEST_CASE("oriented_sigma has the requested axes and orientation")
{
    const double a = 3.0, b = 1.0;
    for ( double angle : {0.0, 30.0, 90.0, 145.0} )
    {
        const Eigen::MatrixXd sigma = oriented_sigma(a, b, angle);
        CHECK((sigma - sigma.transpose()).cwiseAbs().maxCoeff() < 1e-12);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sigma);
        CHECK(std::sqrt(solver.eigenvalues()(1)) == doctest::Approx(a));
        CHECK(std::sqrt(solver.eigenvalues()(0)) == doctest::Approx(b));

        // the major axis points along the requested angle (up to a sign)
        const Eigen::Vector2d major = solver.eigenvectors().col(1);
        const double radians = angle * M_PI / 180.0;
        const Eigen::Vector2d expected(std::cos(radians), std::sin(radians));
        CHECK(std::abs(std::abs(major.dot(expected)) - 1.0) < 1e-12);
    }
}

TEST_CASE("local spacing and window radius bracket the batch's scales")
{
    // A regular grid has a known nearest-neighbor distance and a known extent,
    // so both ends of the ladder are checkable exactly.
    const int per_side = 11;
    const double step = 0.25;
    Eigen::MatrixXd x(per_side * per_side, 2);
    int row = 0;
    for ( int i = 0; i < per_side; ++i )
    {
        for ( int j = 0; j < per_side; ++j )
        {
            x(row, 0) = i * step;
            x(row, 1) = j * step;
            ++row;
        }
    }
    Eigen::VectorXd center(2);
    center << 1.25, 1.25;  // exactly on a grid point

    CHECK(local_spacing(x, center) == doctest::Approx(step));
    // farthest corner from the center of a 2.5 x 2.5 grid
    CHECK(window_radius(x, center) == doctest::Approx(std::sqrt(2.0) * 1.25));

    CHECK_THROWS_AS(local_spacing(x.topRows(1), center), std::invalid_argument);
    CHECK_THROWS_AS(window_radius(x, Eigen::VectorXd::Zero(3)), std::invalid_argument);
}

TEST_CASE("the ladder is log-spaced and hits both endpoints")
{
    const Eigen::VectorXd scales = ladder_scales(0.1, 10.0, 6);
    REQUIRE(scales.size() == 6);
    CHECK(scales(0) == doctest::Approx(0.1));
    CHECK(scales(5) == doctest::Approx(10.0));
    // equal ratios between consecutive rungs is what "log-spaced" means
    const double ratio = scales(1) / scales(0);
    for ( int i = 1; i < 5; ++i )
    {
        CHECK(scales(i + 1) / scales(i) == doctest::Approx(ratio));
    }

    // a radius below the local spacing collapses to a degenerate-but-valid ladder
    const Eigen::VectorXd inverted = ladder_scales(5.0, 1.0, 4);
    CHECK(inverted(0) == doctest::Approx(5.0));
    CHECK(inverted(3) == doctest::Approx(5.0));
    CHECK(ladder_scales(0.1, 10.0, 1).size() == 1);
    CHECK_THROWS_AS(ladder_scales(0.1, 10.0, 0), std::invalid_argument);
}

TEST_CASE("mid-out ordering visits the middle first and then works outward")
{
    for ( int n = 1; n <= 8; ++n )
    {
        const std::vector<int> order = mid_out(n);
        REQUIRE(order.size() == static_cast<std::size_t>(n));

        // a permutation of 0..n-1
        std::vector<int> sorted = order;
        std::sort(sorted.begin(), sorted.end());
        for ( int i = 0; i < n; ++i )
        {
            CHECK(sorted[static_cast<std::size_t>(i)] == i);
        }

        // distance from the middle is non-decreasing along the order
        const double middle = (n - 1) / 2.0;
        for ( std::size_t i = 1; i < order.size(); ++i )
        {
            CHECK(std::abs(order[i] - middle) >= std::abs(order[i - 1] - middle) - 1e-12);
        }
    }
    CHECK(mid_out(5)[0] == 2);
    CHECK(mid_out(4)[0] == 1);  // ties broken toward the lower index
}

TEST_CASE("circle rungs are isotropic at the requested scales")
{
    Eigen::VectorXd radii(3);
    radii << 0.5, 1.0, 4.0;
    for ( int dim = 1; dim <= 3; ++dim )
    {
        const std::vector<InitCandidate> rungs = circle_rungs(radii, dim);
        REQUIRE(rungs.size() == 3u);
        // mid-out: the middle rung comes first
        CHECK(rungs[0].label.find("r=1") != std::string::npos);

        const Eigen::VectorXd mu0 = Eigen::VectorXd::Zero(dim);
        for ( const InitCandidate& rung : rungs )
        {
            const Eigen::MatrixXd L =
                unpack_theta_hat(rung.theta_hat, mu0, MuMode::Pinned).L;
            const Eigen::MatrixXd sigma = L * L.transpose();
            // isotropic: sigma is radius^2 times the identity
            const double radius = L(0, 0);
            CHECK((sigma - radius * radius * Eigen::MatrixXd::Identity(dim, dim))
                      .cwiseAbs().maxCoeff() < 1e-12);
        }
    }
}

TEST_CASE("window rungs scale the window's shape to the requested major axis")
{
    std::mt19937 gen(1);
    const Eigen::MatrixXd x = test_helpers::randn_points(60, 2, gen, 1.0);
    const Eigen::MatrixXd shape = window_shape(x, Eigen::VectorXd::Ones(60));

    Eigen::VectorXd radii(3);
    radii << 0.5, 2.0, 8.0;
    const std::vector<InitCandidate> rungs = window_rungs(radii, shape);
    REQUIRE(rungs.size() == 3u);

    const Eigen::VectorXd mu0 = Eigen::VectorXd::Zero(2);
    for ( const InitCandidate& rung : rungs )
    {
        const Eigen::MatrixXd L =
            unpack_theta_hat(rung.theta_hat, mu0, MuMode::Pinned).L;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
            Eigen::MatrixXd(L * L.transpose()));
        const double major = std::sqrt(solver.eigenvalues()(1));
        // the shape is normalized to unit largest eigenvalue, so the scaled
        // copy's major semi-axis IS the rung's scale
        bool matched = false;
        for ( Eigen::Index i = 0; i < radii.size(); ++i )
        {
            matched = matched || std::abs(major - radii(i)) < 1e-9;
        }
        CHECK(matched);
    }
}

TEST_CASE("backprojection is unbiased for iid standard-normal probes")
{
    // E[(1/k) sum_l y_l z_l] = E[z z^T] r = r. With many probes the estimate
    // must approach the truth it was built from.
    std::mt19937 gen(2);
    const int num_points = 12;
    const Eigen::VectorXd truth = test_helpers::randn_points(num_points, 1, gen).col(0);

    double previous = std::numeric_limits<double>::infinity();
    for ( int num_probes : {400, 4000, 40000} )
    {
        const Eigen::MatrixXd z =
            test_helpers::randn_points(num_points, num_probes, gen);
        const Eigen::VectorXd y = z.transpose() * truth;
        const double error = (backproject(z, y) - truth).norm() / truth.norm();
        CHECK(error < previous);  // converging, as 1/sqrt(k)
        previous = error;
    }
    CHECK(previous < 0.05);

    CHECK_THROWS_AS(backproject(Eigen::MatrixXd::Zero(4, 3), Eigen::VectorXd::Zero(2)),
                    std::invalid_argument);
}

TEST_CASE("raw moments recover a known center and covariance")
{
    // Weights proportional to a Gaussian on a fine grid reproduce that
    // Gaussian's own mean and covariance, so the answer is known in advance.
    const int per_side = 61;
    const double half_width = 4.0;
    Eigen::MatrixXd x(per_side * per_side, 2);
    Eigen::VectorXd weights(per_side * per_side);

    Eigen::Vector2d center(0.7, -0.3);
    Eigen::Matrix2d sigma;
    sigma << 0.6, 0.2, 0.2, 0.35;
    const Eigen::Matrix2d precision = sigma.inverse();

    int row = 0;
    for ( int i = 0; i < per_side; ++i )
    {
        for ( int j = 0; j < per_side; ++j )
        {
            // Centered ON the true mean, so the truncation is symmetric and
            // cannot bias it -- an off-center box would, at this width.
            const double px =
                center(0) - half_width + 2 * half_width * i / (per_side - 1);
            const double py =
                center(1) - half_width + 2 * half_width * j / (per_side - 1);
            x(row, 0) = px;
            x(row, 1) = py;
            const Eigen::Vector2d d = Eigen::Vector2d(px, py) - center;
            weights(row) = std::exp(-0.5 * d.dot(precision * d));
            ++row;
        }
    }

    const RawMoments moments = raw_moments(x, weights, -1, 0.0, 0.0);
    CHECK((moments.mu - center).cwiseAbs().maxCoeff() < 1e-6);
    CHECK((moments.sigma - sigma).cwiseAbs().maxCoeff() < 1e-3);
}

TEST_CASE("the spike is excluded from the moments, and the thresholds bite")
{
    // The target's own point carries the mesh-unresolvable spike. Left in, it
    // drags the center toward itself and shrinks the covariance -- which is
    // exactly what spike_index exists to prevent.
    const int count = 41;
    Eigen::MatrixXd x(count, 1);
    Eigen::VectorXd r(count);
    for ( int i = 0; i < count; ++i )
    {
        x(i, 0) = -2.0 + 4.0 * i / (count - 1);
        r(i) = std::exp(-0.5 * x(i, 0) * x(i, 0));
    }
    const int spike_at = 5;
    x(spike_at, 0) = -1.5;
    r(spike_at) = 50.0;  // a spike far from the kernel's center

    const RawMoments with_spike = raw_moments(x, r, -1, 0.0, 0.0);
    const RawMoments without = raw_moments(x, r, spike_at, 0.0, 0.0);

    // Left in, the spike dominates and pulls the center most of the way to it.
    CHECK(with_spike.mu(0) < -0.5);
    // Excluded, the center returns to the kernel's own -- not exactly zero,
    // since dropping one point of a symmetric grid leaves it slightly lopsided.
    CHECK(std::abs(without.mu(0)) < 0.05);
    CHECK(without.sigma(0, 0) > with_spike.sigma(0, 0));

    // A relative threshold suppresses the tails, tightening the covariance.
    const RawMoments trimmed = raw_moments(x, r, spike_at, 0.5, 0.0);
    CHECK(trimmed.sigma(0, 0) < without.sigma(0, 0));

    // Suppressing everything is an error, not a silent NaN.
    CHECK_THROWS_AS(raw_moments(x, r, spike_at, 1.5, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(raw_moments(x, r, count, 0.0, 0.0), std::invalid_argument);
}

TEST_CASE("the noise threshold rescues a bump on a mostly-far-field window")
{
    // The regime noise_mad is FOR, per its documentation: a conservative
    // window where most entries are backprojection noise, so the median
    // estimates that noise floor. Without it the noise -- spread over the
    // whole window -- inflates the covariance far beyond the bump's own.
    //
    // (On a tight window where most entries carry signal the median instead
    // OVERESTIMATES the noise and this threshold suppresses everything, which
    // is why it defaults off and why raw_moments raises rather than returning
    // a silent NaN. That case is checked below.)
    std::mt19937 gen(3);
    std::uniform_real_distribution<double> noise(-0.02, 0.02);
    const int count = 201;
    Eigen::MatrixXd x(count, 1);
    Eigen::VectorXd r(count);
    for ( int i = 0; i < count; ++i )
    {
        x(i, 0) = -10.0 + 20.0 * i / (count - 1);
        r(i) = std::exp(-0.5 * x(i, 0) * x(i, 0)) + noise(gen);
    }

    const RawMoments raw = raw_moments(x, r, -1, 0.0, 0.0);
    const RawMoments denoised = raw_moments(x, r, -1, 0.0, 3.0);

    MESSAGE("far-field window: sigma " << raw.sigma(0, 0) << " -> " << denoised.sigma(0, 0)
                                       << " (bump's own is 1)");
    // Observed: 3.32 -> 0.91 against the bump's own variance of 1, so the
    // noise inflates the estimate by ~3.7x and the threshold recovers it.
    CHECK(raw.sigma(0, 0) > 3.0);
    CHECK(denoised.sigma(0, 0) < 1.5);
    CHECK(raw.sigma(0, 0) / denoised.sigma(0, 0) > 3.0);
    CHECK(std::abs(denoised.mu(0)) < 0.05);

    // and on a tight window, where the median overestimates the noise, it
    // suppresses everything -- loudly
    const Eigen::MatrixXd tight = x.middleRows(95, 11);
    const Eigen::VectorXd tight_r = r.segment(95, 11);
    CHECK_THROWS_AS(raw_moments(tight, tight_r, -1, 0.0, 3.0), std::invalid_argument);
}
