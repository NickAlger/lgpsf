// SPDX-License-Identifier: MIT
#pragma once

/// The one SVD the library uses: Eigen's one-sided Jacobi.
///
/// Every SVD in lgpsf is of a per-row design matrix or Jacobian -- at most a
/// few hundred rows (probes) by a few dozen columns (modes) -- and most of them
/// are the rank-deficient fallback of a pivoted QR.  Eigen 3.4.0's BDCSVD, the
/// divide-and-conquer SVD used here before 2026-10-03, aborts inside
/// `perturbCol0` on some rank-deficient inputs with repeated or zero singular
/// values (an index assertion on its internal permutation; Eigen issue 2663,
/// fixed after 3.4.0).  The uncapped wedge ladder reaches such matrices at 100
/// probes.  At these sizes JacobiSVD is as fast as the alternative and has no
/// such path, so it is used everywhere, through this helper.
#include <Eigen/Dense>

namespace lgpsf::detail {

/// Thin SVD of a dense matrix (U and V computed) by one-sided Jacobi.
inline Eigen::JacobiSVD<Eigen::MatrixXd> thin_svd( const Eigen::Ref<const Eigen::MatrixXd>& M )
{
    return Eigen::JacobiSVD<Eigen::MatrixXd>(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
}

/// Thin SVD with U only (the range), by one-sided Jacobi.
inline Eigen::JacobiSVD<Eigen::MatrixXd> thin_svd_u( const Eigen::Ref<const Eigen::MatrixXd>& M )
{
    return Eigen::JacobiSVD<Eigen::MatrixXd>(M, Eigen::ComputeThinU);
}

}  // namespace lgpsf::detail
