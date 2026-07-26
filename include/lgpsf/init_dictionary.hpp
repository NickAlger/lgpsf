#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Initial-ellipsoid dictionary generation: the hypothesis side of the
/// probe fit, as a standalone library.
///
/// Everything here is geometry and policy over the caller's batch -- no
/// probes, no fitting, no randomness. Parameter vectors come out in the PINNED
/// encoding (`theta_hat` with no center block, shape only); callers move them
/// to the fitted encoding with `release_mu` when they want it. See
/// ellipsoid_transform.hpp for both.
///
/// Point batches are (K, N) coordinate-major.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <ellipsoid_tree/kd_tree.hpp>

#include "lgpsf/ellipsoid_transform.hpp"

namespace lgpsf {

/// One labelled starting hypothesis. The label is carried through the
/// candidate stream and out into diagnostics, so a fit can say which family
/// and which rung produced its answer.
struct InitCandidate
{
    std::string label;
    Eigen::VectorXd theta_hat;  ///< Pinned encoding: log-diagonal, then strict lower.
};

/// Pack a lower-triangular Cholesky factor into the pinned `theta_hat`
/// encoding: N log-diagonals, then the strictly-lower entries row-major.
inline Eigen::VectorXd theta_hat_from_cholesky(
    const Eigen::Ref<const Eigen::MatrixXd>& L )
{
    const Eigen::Index n = L.rows();
    if ( L.cols() != n )
    {
        throw std::invalid_argument("lgpsf::theta_hat_from_cholesky: L must be square");
    }
    Eigen::VectorXd theta_hat(theta_hat_size(static_cast<int>(n), MuMode::Pinned));
    Eigen::Index idx = 0;
    for ( Eigen::Index i = 0; i < n; ++i )
    {
        if ( !(L(i, i) > 0.0) )
        {
            throw std::invalid_argument(
                "lgpsf::theta_hat_from_cholesky: L's diagonal must be positive; "
                "entry " + std::to_string(i) + " is " + std::to_string(L(i, i)));
        }
        theta_hat(idx++) = std::log(L(i, i));
    }
    for ( Eigen::Index i = 1; i < n; ++i )
    {
        for ( Eigen::Index j = 0; j < i; ++j )
        {
            theta_hat(idx++) = L(i, j);
        }
    }
    return theta_hat;
}

/// The window's own ellipsoid shape: the mass-weighted covariance of the batch
/// geometry, normalized to largest eigenvalue 1, as a lower Cholesky factor.
///
/// The mass weighting is the point. Lumped masses are cell areas, so weighting
/// by them measures the window REGION under the uniform measure, independent of
/// mesh grading -- an unweighted point covariance would measure where the mesh
/// happens to be dense instead. Eigenvalues are floored at 1e-4 of the largest,
/// which caps the aspect ratio against degenerate windows (near-collinear, or
/// clipped by a boundary).
inline Eigen::MatrixXd window_shape( const Eigen::Ref<const Eigen::MatrixXd>& x,
                                     const Eigen::Ref<const Eigen::VectorXd>& m2_diag )
{
    if ( x.rows() != m2_diag.size() )
    {
        throw std::invalid_argument(
            "lgpsf::window_shape: x has " + std::to_string(x.rows())
            + " points but m2_diag has " + std::to_string(m2_diag.size()) + " entries");
    }
    const double total = m2_diag.sum();
    if ( !(total > 0.0) )
    {
        throw std::invalid_argument("lgpsf::window_shape: the masses must sum to a positive value");
    }
    const Eigen::VectorXd w = m2_diag / total;
    const Eigen::VectorXd center = x.transpose() * w;
    const Eigen::MatrixXd centered = x.rowwise() - center.transpose();
    const Eigen::MatrixXd covariance = centered.transpose() * w.asDiagonal() * centered;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    Eigen::VectorXd values = solver.eigenvalues();  // ascending
    const double largest = values(values.size() - 1);
    if ( !(largest > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::window_shape: the batch has no spatial extent (its "
            "mass-weighted covariance is singular)");
    }
    for ( Eigen::Index i = 0; i < values.size(); ++i )
    {
        values(i) = std::max(values(i), 1e-4 * largest) / largest;
    }
    const Eigen::MatrixXd normalized =
        solver.eigenvectors() * values.asDiagonal() * solver.eigenvectors().transpose();
    return Eigen::MatrixXd(normalized.llt().matrixL());
}

/// A 2D SPD covariance with 1-sigma semi-axes (a, b), the a-axis rotated
/// `angle_degrees` from horizontal -- the building block for
/// orientation-covering dictionaries, which is the circle family's blind spot.
inline Eigen::MatrixXd oriented_sigma( double a, double b, double angle_degrees )
{
    const double angle = angle_degrees * M_PI / 180.0;
    Eigen::Matrix2d rotation;
    rotation << std::cos(angle), -std::sin(angle),
                std::sin(angle),  std::cos(angle);
    Eigen::Matrix2d axes = Eigen::Matrix2d::Zero();
    axes(0, 0) = a * a;
    axes(1, 1) = b * b;
    return rotation * axes * rotation.transpose();
}

/// Mesh spacing at mu0: the nearest-neighbor distance of the batch point
/// closest to mu0. The ladder's bottom scale.
///
/// Note the layout flip at the tree boundary: `ellipsoid_tree` indexes points
/// as COLUMNS of a (dim, n) matrix, the opposite of this library's
/// coordinate-major convention, so the batch is transposed on the way in.
inline double local_spacing( const Eigen::Ref<const Eigen::MatrixXd>& x,
                             const Eigen::Ref<const Eigen::VectorXd>& mu0 )
{
    if ( x.cols() != mu0.size() )
    {
        throw std::invalid_argument(
            "lgpsf::local_spacing: x has " + std::to_string(x.cols())
            + " coordinate columns but mu0 has " + std::to_string(mu0.size())
            + " entries");
    }
    if ( x.rows() < 2 )
    {
        throw std::invalid_argument(
            "lgpsf::local_spacing: needs at least two points to have a spacing, got "
            + std::to_string(x.rows()));
    }

    const Eigen::MatrixXd points = x.transpose();
    const ellipsoid_tree::KDTree tree(points);

    const auto nearest_to_center = tree.query(mu0, 1);
    const int nearest = nearest_to_center.first(0, 0);

    const auto neighbors = tree.query(points.col(nearest), 2);
    return std::sqrt(neighbors.second(1, 0));
}

/// Radius of the batch around mu0: the ladder's top scale, and the
/// window-containment admissibility bound. The window is conservative, so the
/// true kernel fits inside it by construction.
inline double window_radius( const Eigen::Ref<const Eigen::MatrixXd>& x,
                             const Eigen::Ref<const Eigen::VectorXd>& mu0 )
{
    if ( x.cols() != mu0.size() )
    {
        throw std::invalid_argument(
            "lgpsf::window_radius: x has " + std::to_string(x.cols())
            + " coordinate columns but mu0 has " + std::to_string(mu0.size())
            + " entries");
    }
    if ( x.rows() == 0 )
    {
        throw std::invalid_argument("lgpsf::window_radius: the batch is empty");
    }
    return (x.rowwise() - mu0.transpose()).rowwise().norm().maxCoeff();
}

/// Log-spaced scales from the local mesh spacing to the window radius.
///
/// Too-small and too-large initial ellipsoids both fail, and differently; a log
/// ladder brackets every case observed so far.
inline Eigen::VectorXd ladder_scales( double local, double radius, int num_rungs )
{
    if ( num_rungs < 1 )
    {
        throw std::invalid_argument(
            "lgpsf::ladder_scales: num_rungs must be >= 1, got "
            + std::to_string(num_rungs));
    }
    const double low = std::max(local, 1e-12 * std::max(radius, 1.0));
    const double high = std::max(radius, local);
    Eigen::VectorXd scales(num_rungs);
    if ( num_rungs == 1 )
    {
        scales(0) = low;
        return scales;
    }
    const double step = (std::log(high) - std::log(low)) / (num_rungs - 1);
    for ( int i = 0; i < num_rungs; ++i )
    {
        scales(i) = std::exp(std::log(low) + i * step);
    }
    scales(num_rungs - 1) = high;  // exact at the endpoint, as geomspace is
    return scales;
}

/// Index order for a ladder of length n: middle first, then outward.
///
/// Ordering is load-bearing, not cosmetic: mid scales win most often and the
/// extremes are insurance, while the absolute early-exit certificate fires on
/// the FIRST good-enough candidate -- so a good ordering is what makes early
/// stopping cheap rather than lucky.
inline std::vector<int> mid_out( int n )
{
    std::vector<int> order(static_cast<std::size_t>(std::max(0, n)));
    for ( int i = 0; i < n; ++i )
    {
        order[static_cast<std::size_t>(i)] = i;
    }
    const double middle = (n - 1) / 2.0;
    std::stable_sort(order.begin(), order.end(), [middle]( int a, int b ) {
        const double da = std::abs(a - middle);
        const double db = std::abs(b - middle);
        if ( da != db )
        {
            return da < db;
        }
        return a < b;
    });
    return order;
}

namespace detail {

inline std::string rung_label( const char* family, double radius )
{
    // "%.3g", matching the prototype's labels so diagnostics read the same.
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%s r=%.3g", family, radius);
    return std::string(buffer);
}

} // end namespace detail

/// Labelled isotropic starting hypotheses at the given scales, in mid-out order.
inline std::vector<InitCandidate> circle_rungs(
    const Eigen::Ref<const Eigen::VectorXd>& radii, int dim )
{
    std::vector<InitCandidate> out;
    const int num_lower = dim * (dim - 1) / 2;
    for ( int i : mid_out(static_cast<int>(radii.size())) )
    {
        Eigen::VectorXd theta_hat(theta_hat_size(dim, MuMode::Pinned));
        theta_hat.head(dim).setConstant(std::log(radii(i)));
        theta_hat.tail(num_lower).setZero();
        out.push_back(InitCandidate{detail::rung_label("circle", radii(i)),
                                    std::move(theta_hat)});
    }
    return out;
}

/// Labelled scaled copies of the window's own shape (major semi-axis = the
/// rung's scale), in mid-out order.
inline std::vector<InitCandidate> window_rungs(
    const Eigen::Ref<const Eigen::VectorXd>& radii,
    const Eigen::Ref<const Eigen::MatrixXd>& shape )
{
    std::vector<InitCandidate> out;
    for ( int i : mid_out(static_cast<int>(radii.size())) )
    {
        out.push_back(InitCandidate{
            detail::rung_label("window", radii(i)),
            theta_hat_from_cholesky(Eigen::MatrixXd(radii(i) * shape))});
    }
    return out;
}

} // end namespace lgpsf
