#pragma once
// SPDX-License-Identifier: MIT

/// @file
/// @brief Shared test utilities: seeded generators and quadrature rules.

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>

namespace test_helpers {

/// A (K, dim) batch of standard-normal points -- the (points as rows)
/// convention every lgpsf entry point takes.
inline Eigen::MatrixXd randn_points( int num_points, int dim, std::mt19937& gen,
                                     double scale = 1.0 )
{
    std::normal_distribution<double> dist(0.0, scale);
    Eigen::MatrixXd points(num_points, dim);
    for ( int jj = 0; jj < dim; ++jj )
    {
        for ( int ii = 0; ii < num_points; ++ii )
        {
            points(ii, jj) = dist(gen);
        }
    }
    return points;
}

inline Eigen::MatrixXd uniform_points( int num_points, int dim, std::mt19937& gen,
                                       double lo, double hi )
{
    std::uniform_real_distribution<double> dist(lo, hi);
    Eigen::MatrixXd points(num_points, dim);
    for ( int jj = 0; jj < dim; ++jj )
    {
        for ( int ii = 0; ii < num_points; ++ii )
        {
            points(ii, jj) = dist(gen);
        }
    }
    return points;
}

/// Points drawn uniformly from the unit sphere S^(dim-1), as rows.
inline Eigen::MatrixXd random_sphere_points( int num_points, int dim,
                                             std::mt19937& gen )
{
    Eigen::MatrixXd points = randn_points(num_points, dim, gen);
    for ( int ii = 0; ii < num_points; ++ii )
    {
        points.row(ii) /= points.row(ii).norm();
    }
    return points;
}

/// Gauss-Hermite nodes and weights for int_R f(t) exp(-t^2) dt, exact for
/// polynomial f of degree <= 2n-1.
///
/// Golub-Welsch on the Hermite Jacobi matrix (diagonal 0, off-diagonal
/// sqrt(k/2)): nodes are its eigenvalues, weights sqrt(pi) times the squared
/// first component of each eigenvector. A symmetric eigenvalue solve is the
/// only machinery needed -- no special-function library, matching this
/// project's no-scipy-equivalent constraint.
inline std::pair<Eigen::VectorXd, Eigen::VectorXd> gauss_hermite( int n )
{
    Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(n, n);
    for ( int k = 1; k < n; ++k )
    {
        const double beta = std::sqrt(k / 2.0);
        jacobi(k, k - 1) = beta;
        jacobi(k - 1, k) = beta;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(jacobi);
    Eigen::VectorXd nodes = solver.eigenvalues();
    Eigen::VectorXd weights(n);
    for ( int ii = 0; ii < n; ++ii )
    {
        const double first = solver.eigenvectors()(0, ii);
        weights(ii) = std::sqrt(M_PI) * first * first;
    }
    return {std::move(nodes), std::move(weights)};
}

/// Tensor-product Gauss-Hermite over R^dim: points as rows (n^dim, dim) with
/// the matching product weights.
inline std::pair<Eigen::MatrixXd, Eigen::VectorXd> gauss_hermite_grid(
    int n, int dim )
{
    const auto [nodes, weights] = gauss_hermite(n);
    Eigen::Index total = 1;
    for ( int d = 0; d < dim; ++d )
    {
        total *= n;
    }

    Eigen::MatrixXd points(total, dim);
    Eigen::VectorXd product_weights = Eigen::VectorXd::Ones(total);
    for ( Eigen::Index flat = 0; flat < total; ++flat )
    {
        Eigen::Index rest = flat;
        for ( int d = 0; d < dim; ++d )
        {
            const Eigen::Index idx = rest % n;
            rest /= n;
            points(flat, d) = nodes(idx);
            product_weights(flat) *= weights(idx);
        }
    }
    return {std::move(points), std::move(product_weights)};
}

/// int_R t^a exp(-t^2) dt: zero for odd a, Gamma((a+1)/2) otherwise.
inline double gaussian_moment_1d( int a )
{
    return ( a % 2 != 0 ) ? 0.0 : std::tgamma((a + 1) / 2.0);
}

} // end namespace test_helpers
