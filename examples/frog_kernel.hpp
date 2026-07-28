// SPDX-License-Identifier: MIT
//
// The rotating frog kernel: the shared test problem for the C++ examples.
//
// Entry for entry the same operator as `frog_kernel.py`, so the C++ and Python
// example paths fit the same thing and can be compared directly. Not a lesson
// in itself -- a target to fit.
//
// The kernel is a Gaussian whose covariance ROTATES with position, modulated
// by a cos-sin product. The rotation means a stationary convolution cannot
// represent it, while a fitted per-row ellipsoid can; the modulation makes
// each row strongly non-Gaussian -- a bright core with a notch through it --
// so the LG modes above level 0 have to earn their place.
//
// With kModulation = 1 the factor (1 + kModulation * modulation) stays in
// [0, 2], so the kernel is non-negative. The fitted approximation is signed
// either way, since LG modes are.
//
//     H(i, j) = m1(i) * m2(j) * phi(x_i, x_j)
//
// on a uniform grid over the unit square, with m the lumped mass h^2.
//
// The kernel is anchored at the TARGET, so a ROW of the operator is a
// point-spread function -- the object lgpsf models, and therefore the object
// worth plotting.

#pragma once

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Dense>

namespace frog
{

// Variances along the unrotated axes, and the modulation strength.
inline const Eigen::Vector2d kSigma0Diag(0.01, 0.0025);
inline constexpr double kModulation = 1.0;

/// The local rotation angle at x.
inline double angle_at( const Eigen::Vector2d& x )
{
    return 0.5 * M_PI * (x(0) + x(1));
}

/// Vanishes on the boundary of the unit square, so the kernel is compact.
inline double bump_at( const Eigen::Vector2d& x )
{
    return x(0) * (1.0 - x(0)) * x(1) * (1.0 - x(1));
}

/// Row `target` of the kernel: its point-spread function over every source.
///
/// `sources` is (K, 2) -- points as ROWS, the C++ convention. Returns (K).
inline Eigen::VectorXd frog_row( const Eigen::Vector2d& target,
                                 const Eigen::MatrixXd& sources )
{
    const Eigen::Vector2d& sd = kSigma0Diag;
    const double theta = angle_at(target);
    const double c = std::cos(theta), s = std::sin(theta);

    const Eigen::Index num_sources = sources.rows();
    Eigen::VectorXd row(num_sources);
    for ( Eigen::Index j = 0; j < num_sources; ++j )
    {
        const double d0 = sources(j, 0) - target(0);
        const double d1 = sources(j, 1) - target(1);
        // p = R(target) * d, with R = [[c, -s], [s, c]]
        const double p0 = c * d0 - s * d1;
        const double p1 = s * d0 + c * d1;

        const double maha2 = p0 * p0 / sd(0) + p1 * p1 / sd(1);
        const double gaussian = std::exp(-0.5 * maha2)
                                / (2.0 * M_PI * std::sqrt(sd(0) * sd(1)));
        const double modulation = std::cos(p0 / (std::sqrt(sd(0)) / 2.0))
                                  * std::sin(p1 / (std::sqrt(sd(1)) / 2.0));
        row(j) = bump_at(target) * (1.0 + kModulation * modulation) * gaussian;
    }
    return row;
}

/// The kernel's local covariance at x -- the a-priori ellipsoid field.
///
/// Sigma = R^T diag(sd) R at the row's own point: the right shape and
/// orientation, and nothing about the modulation, which is what the LG modes
/// have to discover.
inline Eigen::Matrix2d frog_covariance( const Eigen::Vector2d& x )
{
    const double theta = angle_at(x);
    const double c = std::cos(theta), s = std::sin(theta);
    Eigen::Matrix2d R;
    R << c, -s, s, c;
    return R.transpose() * kSigma0Diag.asDiagonal() * R;
}

/// The grid, the masses, the dense truth, and the prior ellipsoid field.
struct Problem
{
    Eigen::MatrixXd x;                      ///< (K, 2), points as rows
    Eigen::VectorXd mass;                   ///< (K), lumped mass h^2
    Eigen::MatrixXd H;                      ///< (K, K), the dense truth
    std::vector<Eigen::MatrixXd> sigma;     ///< per-row prior covariance
    double spacing = 0.0;

    Eigen::Index size() const { return mass.size(); }
};

inline Problem build_problem( int grid )
{
    Problem problem;
    const Eigen::Index count = static_cast<Eigen::Index>(grid) * grid;
    problem.spacing = 1.0 / grid;

    problem.x.resize(count, 2);
    for ( int i = 0; i < grid; ++i )
    {
        for ( int j = 0; j < grid; ++j )
        {
            // Cell centers, off the boundary; index order matches numpy's
            // meshgrid(indexing="ij") so the two languages agree row for row.
            problem.x(i * grid + j, 0) = (i + 0.5) / grid;
            problem.x(i * grid + j, 1) = (j + 0.5) / grid;
        }
    }

    problem.mass = Eigen::VectorXd::Constant(count, problem.spacing * problem.spacing);

    problem.H.resize(count, count);
    for ( Eigen::Index i = 0; i < count; ++i )
    {
        const Eigen::Vector2d target = problem.x.row(i).transpose();
        problem.H.row(i) = problem.mass(i)
                           * frog_row(target, problem.x).array()
                           * problem.mass.array();
    }

    problem.sigma.reserve(static_cast<std::size_t>(count));
    for ( Eigen::Index i = 0; i < count; ++i )
    {
        problem.sigma.push_back(frog_covariance(problem.x.row(i).transpose()));
    }
    return problem;
}

/// `num_probes` random probes and their responses -- all the fitter ever sees.
///
/// Returns (V, HV), each (K, num_probes) with probes as COLUMNS. The generator
/// is fixed, but note that C++ and numpy draw different numbers from the same
/// seed, so the two languages fit the same operator from different probes.
inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
probes( const Problem& problem, int num_probes, unsigned seed = 0 )
{
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::MatrixXd V(problem.size(), num_probes);
    for ( Eigen::Index j = 0; j < V.cols(); ++j )
    {
        for ( Eigen::Index i = 0; i < V.rows(); ++i ) { V(i, j) = normal(generator); }
    }
    return {V, problem.H * V};
}

} // namespace frog
