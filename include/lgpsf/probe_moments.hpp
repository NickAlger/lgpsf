#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Estimators of a target from its probe data alone -- no fitting.
///
/// The target is any function on the batch known only through inner products
/// with random probe fields (one row of an implicitly available operator being
/// the motivating case). These are the zero-extra-matvec initial-guess and
/// quality-control companions to the fitting layer.
///
/// Point batches are (K, N) coordinate-major; probe fields are (K, k), one
/// column per probe.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace lgpsf {

/// Unbiased estimate of the target's raw coefficient vector over the batch,
/// from probe pairs: r_hat = (1/k) sum_l y_l z_l.
///
/// Requires the probe entries to be iid standard normal in the RAW dof basis
/// (E[z z^T] = I), which is the convention of the intended pipeline. Noise per
/// entry is O(||target|| / sqrt(k)) -- see raw_moments' thresholds, which
/// exist to deal with exactly that floor.
///
/// @param z (K, k) raw probe fields on the batch.
/// @param y (k,) raw responses.
/// @return  (K,) the unbiased estimate of the raw target on the batch.
/// @throws std::invalid_argument if the shapes disagree or there are no probes.
inline Eigen::VectorXd backproject( const Eigen::Ref<const Eigen::MatrixXd>& z,
                                    const Eigen::Ref<const Eigen::VectorXd>& y )
{
    if ( z.cols() != y.size() )
    {
        throw std::invalid_argument(
            "lgpsf::backproject: z has " + std::to_string(z.cols())
            + " probes but y has " + std::to_string(y.size()) + " responses");
    }
    if ( z.cols() == 0 )
    {
        throw std::invalid_argument("lgpsf::backproject: needs at least one probe");
    }
    return (z * y) / static_cast<double>(z.cols());
}

/// A center and covariance estimated from raw target values.
struct RawMoments
{
    Eigen::VectorXd mu;     ///< (N,)
    Eigen::MatrixXd sigma;  ///< (N, N)
};

/// Weighted mean and covariance of the target kernel's magnitude, from RAW
/// values -- measured or backprojected.
///
/// **No mass vector, deliberately.** r_raw[j] = m_target * m_j * phi(x_j), so
/// |r_raw| already carries the lumped-mass quadrature weight: summing
/// |r_raw|-weighted point statistics IS the mesh approximation of the
/// continuum moments of |phi|. That is the whole mass subtlety here, resolved
/// by using raw values rather than kernel values.
///
/// @param spike_index excludes the target's own point, if it has one (pass a
///   negative value for none). That point's raw value contains the
///   mesh-unresolvable spike, which belongs to the discrete correction rather
///   than to the smooth kernel whose moments are wanted; left in, it drags the
///   center toward that point and shrinks the covariance.
/// @param rel_threshold zeroes weights below this fraction of the largest.
/// @param noise_mad zeroes weights below this many robust noise sigmas, with
///   sigma estimated as 1.4826 * median|r_raw|. Around 3-4 suits backprojected
///   targets on CONSERVATIVE support windows, where most entries are far-field
///   noise and the median therefore estimates it; on tight windows most
///   entries carry signal, the median overestimates the noise, and
///   rel_threshold alone is safer.
///
/// Set both thresholds to zero for exact (measured) targets.
///
/// @param x     (K, N) batch points.
/// @param r_raw (K,) the raw target on the batch.
/// @return      The mass-free centre and covariance of |r_raw|.
/// @throws std::invalid_argument on shape mismatch, or if the thresholds
///         suppress every weight -- a silent NaN would otherwise reach the fit.
inline RawMoments raw_moments( const Eigen::Ref<const Eigen::MatrixXd>& x,
                               const Eigen::Ref<const Eigen::VectorXd>& r_raw,
                               int spike_index = -1,
                               double rel_threshold = 0.05,
                               double noise_mad = 0.0 )
{
    const Eigen::Index num_points = x.rows();
    if ( r_raw.size() != num_points )
    {
        throw std::invalid_argument(
            "lgpsf::raw_moments: x has " + std::to_string(num_points)
            + " points but r_raw has " + std::to_string(r_raw.size()) + " entries");
    }
    if ( spike_index >= num_points )
    {
        throw std::invalid_argument(
            "lgpsf::raw_moments: spike_index " + std::to_string(spike_index)
            + " is outside the batch of " + std::to_string(num_points) + " points");
    }

    Eigen::VectorXd w = r_raw.cwiseAbs();
    if ( spike_index >= 0 )
    {
        w(spike_index) = 0.0;
    }
    if ( rel_threshold > 0.0 )
    {
        const double cutoff = rel_threshold * w.maxCoeff();
        for ( Eigen::Index j = 0; j < num_points; ++j )
        {
            if ( w(j) < cutoff )
            {
                w(j) = 0.0;
            }
        }
    }
    if ( noise_mad > 0.0 )
    {
        // The median is of the ORIGINAL magnitudes, spike included: it is
        // estimating the noise floor of the measurement, not of what survived
        // the thresholds above.
        std::vector<double> magnitudes(static_cast<std::size_t>(num_points));
        for ( Eigen::Index j = 0; j < num_points; ++j )
        {
            magnitudes[static_cast<std::size_t>(j)] = std::abs(r_raw(j));
        }
        std::sort(magnitudes.begin(), magnitudes.end());
        const std::size_t half = magnitudes.size() / 2;
        const double median = ( magnitudes.size() % 2 == 1 )
                                  ? magnitudes[half]
                                  : 0.5 * (magnitudes[half - 1] + magnitudes[half]);
        const double cutoff = noise_mad * 1.4826 * median;
        for ( Eigen::Index j = 0; j < num_points; ++j )
        {
            if ( w(j) < cutoff )
            {
                w(j) = 0.0;
            }
        }
    }

    const double total = w.sum();
    if ( !(total > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::raw_moments: every weight was suppressed -- the thresholds "
            "(rel_threshold, noise_mad) are too aggressive for this data, or "
            "r_raw is identically zero away from the spike");
    }
    w /= total;

    RawMoments out;
    out.mu = x.transpose() * w;
    const Eigen::MatrixXd centered = x.rowwise() - out.mu.transpose();
    out.sigma = centered.transpose() * w.asDiagonal() * centered;
    return out;
}

} // end namespace lgpsf
