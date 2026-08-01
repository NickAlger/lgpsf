#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Generalized Lanczos on the pencil (B + E, H_r), and the two
/// operations built on it: caching rightmost modes for the GLR surrogate
/// (`extend_modes`) and the a0-tied eigenvalue flip (`make_pd`).
///
/// One machine serves both ends of the spectrum: Lanczos on H_r^{-1} (B + E)
/// in the H_r inner product — every iteration costs one operator apply, one
/// oracle SOLVE, and one H_r apply — with full reorthogonalization against
/// both the Krylov basis and the existing mode block. Deflating against the
/// block is what makes every operation here INCREMENTAL: modes already in
/// the block are invisible, so a second call continues where the first
/// stopped, and nothing is ever found twice.
///
/// The well-posedness rule this file lives by (plan, P1): never enumerate
/// modes near an accumulation point. The pencil's spectrum accumulates at 0
/// with only the physics-determined informed modes away from it, so
/// "rightmost n modes" and "every mode below -gamma*a0" are both
/// mesh-independent requests; their Euclidean counterparts are not.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/corrections/shifted_operator.hpp"

namespace lgpsf::corrections {

/// Budget and tolerances for one Lanczos-based operation.
struct LanczosOptions
{
    int max_iters = 150;       ///< oracle-solve budget (the expensive count)
    double oracle_tol = 1e-10; ///< passed through to every oracle solve
    double ritz_tol = 1e-9;    ///< a Ritz pair converges when its residual
                               ///< bound is below ritz_tol * spectral scale
    unsigned seed = 0;         ///< start-vector seed (deterministic)
};

/// Everything one deflated Lanczos run learned: ALL Ritz pairs (ascending),
/// their residual bounds, and how far the run got.
struct PencilSweep
{
    Eigen::VectorXd values;     ///< Ritz values, ascending
    Eigen::MatrixXd vectors;    ///< (N, m) H_r-orthonormal Ritz vectors
    Eigen::VectorXd residuals;  ///< pencil residual bound per pair
    double scale = 0.0;         ///< spectral scale the bounds compare against
    int iterations = 0;         ///< oracle solves spent
    bool exhausted = false;     ///< invariant subspace reached (all exact)

    bool converged( Eigen::Index i, double ritz_tol ) const
    {
        return exhausted || residuals(i) <= ritz_tol * scale;
    }
};

/// One deflated H_r-Lanczos run on the pencil (B + E, H_r). Does not touch
/// the struct: the caller decides what to merge. Both ends of the spectrum
/// converge first, which is exactly what the layer ever asks for.
inline PencilSweep pencil_sweep( const ShiftedOperator& A,
                                 LanczosOptions opts = {} )
{
    if ( opts.max_iters <= 0 || !(opts.oracle_tol > 0.0)
         || !(opts.ritz_tol > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::pencil_sweep: max_iters, oracle_tol and "
            "ritz_tol must all be positive");
    }
    const Eigen::Index n = A.dim();
    const Eigen::Index room = n - A.block.rank();
    const int m_cap = static_cast<int>(std::min<Eigen::Index>(
        room, static_cast<Eigen::Index>(opts.max_iters)));
    PencilSweep sweep;
    if ( m_cap <= 0 )
    {
        sweep.exhausted = true;
        sweep.values = Eigen::VectorXd(0);
        sweep.vectors = Eigen::MatrixXd(n, 0);
        sweep.residuals = Eigen::VectorXd(0);
        return sweep;
    }

    Eigen::MatrixXd Q(n, m_cap);
    Eigen::MatrixXd HrQ(n, m_cap);
    std::vector<double> alpha;
    std::vector<double> beta;  // beta[j] couples q_j and q_{j+1}

    const auto deflate = [&]( Eigen::VectorXd& u, Eigen::Index basis_size ) {
        // classical Gram-Schmidt in the H_r inner product, twice is enough;
        // stored H_r-images make every projection oracle-free
        for ( int pass = 0; pass < 2; ++pass )
        {
            if ( A.block.rank() > 0 )
            {
                u.noalias() -= A.block.V * (A.block.HrV.transpose() * u);
            }
            if ( basis_size > 0 )
            {
                u.noalias() -= Q.leftCols(basis_size)
                               * (HrQ.leftCols(basis_size).transpose() * u);
            }
        }
    };

    // Deflated, H_r-normalized random vectors; one generator serves the
    // start and every restart. The seed is SALTED: a caller who built the
    // operator's own data from the same innocently-chosen seed (everyone
    // uses 0) must not hand the engine start vectors correlated with the
    // operator — an exact stream collision can make every start vector an
    // eigenvector and every emptiness test lie.
    std::mt19937 gen(opts.seed ^ 0x9E3779B9u);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto fresh_direction = [&]( Eigen::Index basis_size,
                                      Eigen::VectorXd& q_out,
                                      Eigen::VectorXd& hq_out ) {
        // several attempts before concluding emptiness: losing one random
        // vector to deflation is evidence, losing three is proof enough
        for ( int attempt = 0; attempt < 3; ++attempt )
        {
            Eigen::VectorXd v(n);
            for ( Eigen::Index i = 0; i < n; ++i )
            {
                v(i) = normal(gen);
            }
            const double pre_norm = std::sqrt(v.dot(A.hr.apply(v).col(0)));
            deflate(v, basis_size);
            const Eigen::VectorXd hv = A.hr.apply(v);
            const double v_norm = std::sqrt(std::max(v.dot(hv), 0.0));
            if ( v_norm > 1e-10 * pre_norm )
            {
                q_out = v / v_norm;
                hq_out = hv / v_norm;
                return true;
            }
        }
        return false;
    };
    {
        Eigen::VectorXd q0, hq0;
        if ( !fresh_direction(0, q0, hq0) )
        {
            throw std::runtime_error(
                "lgpsf::corrections::pencil_sweep: the start vector vanished "
                "under deflation -- the block already spans the space");
        }
        Q.col(0) = q0;
        HrQ.col(0) = hq0;
    }

    double magnitude = 0.0;  // running spectral scale for breakdown tests
    int m = 0;
    for ( int j = 0; j < m_cap; ++j )
    {
        // u = H_r^{-1} (B + E) q_j : one operator apply + one oracle solve
        Eigen::VectorXd u = A.hr.solve(
            A.op.apply(Q.col(j)) + apply_correction(A.block, Q.col(j)),
            opts.oracle_tol);
        alpha.push_back(HrQ.col(j).dot(u));
        magnitude = std::max(magnitude, std::abs(alpha.back()));
        deflate(u, j + 1);
        const Eigen::VectorXd hu = A.hr.apply(u);
        double b = std::sqrt(std::max(u.dot(hu), 0.0));
        m = j + 1;
        const bool breakdown = b <= 1e-12 * std::max(magnitude, 1e-300);
        if ( j + 1 < m_cap )
        {
            if ( breakdown )
            {
                // span(Q) is invariant; for a self-adjoint operator so is
                // its H_r-orthogonal complement, so a fresh start there
                // continues the sweep — a ZERO coupling in T decouples the
                // finished block correctly
                Eigen::VectorXd q_next, hq_next;
                if ( !fresh_direction(j + 1, q_next, hq_next) )
                {
                    sweep.exhausted = true;  // complement is empty
                    break;
                }
                beta.push_back(0.0);
                Q.col(j + 1) = q_next;
                HrQ.col(j + 1) = hq_next;
            }
            else
            {
                beta.push_back(b);
                magnitude = std::max(magnitude, b);
                Q.col(j + 1) = u / b;
                HrQ.col(j + 1) = hu / b;
            }
        }
        else
        {
            if ( breakdown )
            {
                sweep.exhausted = true;  // swept the whole deflated space
            }
            beta.push_back(b);  // the exit residual coupling
        }
    }
    sweep.iterations = m;

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for ( int j = 0; j < m; ++j )
    {
        T(j, j) = alpha[static_cast<std::size_t>(j)];
        if ( j + 1 < m )
        {
            T(j, j + 1) = beta[static_cast<std::size_t>(j)];
            T(j + 1, j) = beta[static_cast<std::size_t>(j)];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(T);
    const double exit_beta =
        ( sweep.exhausted || beta.empty() )
            ? 0.0
            : beta[static_cast<std::size_t>(m - 1)];

    sweep.values = eigen.eigenvalues();
    sweep.vectors = Q.leftCols(m) * eigen.eigenvectors();
    sweep.residuals = exit_beta
                      * eigen.eigenvectors().row(m - 1).transpose().cwiseAbs();
    sweep.scale = std::max(magnitude, sweep.values.cwiseAbs().maxCoeff());
    return sweep;
}

/// What `extend_modes` did.
struct ExtendReport
{
    int added = 0;              ///< cache columns appended
    double next_value =
        std::numeric_limits<double>::quiet_NaN();  ///< first value NOT cached
                                                   ///< (the lambda_{k+1} that
                                                   ///< governs two-level
                                                   ///< conditioning)
    double leftmost_estimate =
        std::numeric_limits<double>::quiet_NaN();  ///< sweep's left edge
    int iterations = 0;
};

/// Cache rightmost pencil modes of B + E into the block (tag `PencilCache`,
/// SURROGATE content only — extending the cache never changes the operator).
/// Stops at `n_right` modes or at the first converged value at or below
/// `lambda_min_target`, whichever comes first. Incremental: already-cached
/// modes are deflated away, so repeated calls deepen the cache.
inline ExtendReport extend_modes( ShiftedOperator& A, int n_right,
                                  double lambda_min_target = 0.0,
                                  LanczosOptions opts = {} )
{
    if ( n_right <= 0 )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::extend_modes: n_right must be positive");
    }
    const PencilSweep sweep = pencil_sweep(A, opts);
    ExtendReport report;
    report.iterations = sweep.iterations;
    if ( sweep.values.size() == 0 )
    {
        return report;
    }
    report.leftmost_estimate = sweep.values(0);

    std::vector<Eigen::Index> take;
    for ( Eigen::Index i = sweep.values.size() - 1; i >= 0; --i )
    {
        if ( !sweep.converged(i, opts.ritz_tol)
             || sweep.values(i) <= lambda_min_target
             || static_cast<int>(take.size()) >= n_right )
        {
            report.next_value = sweep.values(i);
            break;
        }
        take.push_back(i);
    }
    if ( !take.empty() )
    {
        Eigen::MatrixXd V_new(A.dim(), static_cast<Eigen::Index>(take.size()));
        Eigen::MatrixXd lambdas = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(take.size()),
            static_cast<Eigen::Index>(take.size()));
        for ( std::size_t k = 0; k < take.size(); ++k )
        {
            const Eigen::Index at = static_cast<Eigen::Index>(k);
            V_new.col(at) = sweep.vectors.col(take[k]);
            lambdas(at, at) = sweep.values(take[k]);
        }
        const MergeReport merged = merge(
            A.block, A.hr, V_new,
            Eigen::MatrixXd::Zero(V_new.cols(), V_new.cols()), lambdas,
            Provenance::PencilCache);
        report.added = merged.added;
    }
    return report;
}

/// Flip (c = 2) or relu (c = 1).
enum class FlipMode
{
    Flip,
    Relu
};

/// What `make_pd` did, and whether the contract is certified.
struct FlipReport
{
    int flipped = 0;            ///< modes corrected (this call)
    bool certified = false;     ///< leftmost surviving mode resolved
    double lambda_floor =
        std::numeric_limits<double>::quiet_NaN();  ///< the exact PD floor:
                                                   ///< B+E+aHr > 0 iff
                                                   ///< a > -lambda_floor
    double leftmost_before =
        std::numeric_limits<double>::quiet_NaN();  ///< most negative flipped
    int iterations = 0;         ///< oracle solves spent (this call)
    Eigen::VectorXd flipped_values;  ///< pencil values corrected (this call)
};

/// Correct every pencil mode of B + E below the a0-tied threshold
/// -gamma * a0, by flip or relu, as block entries (tag `Flip`; the operator
/// handle is never modified). The slightly-negative noise tail above the
/// threshold is deliberately left in place — the deployment shift absorbs
/// it, and in the N -> infinity limit enumerating it is exactly the
/// ill-posed operation this layer exists to avoid.
///
/// On success (`certified`), records the EXACT data-dependent contract in
/// the struct:  B + E + a H_r > 0  iff  a > -lambda_floor,  with
/// lambda_floor the leftmost surviving pencil value (a byproduct of the
/// same Lanczos run, no extra cost). If the budget runs out first, returns
/// `certified = false` with progress kept — corrected modes stay in the
/// block — so calling again simply continues.
///
/// Call this BEFORE caching leftmost modes with `extend_modes`: make_pd
/// only corrects modes its own (block-deflated) Lanczos discovers.
inline FlipReport make_pd( ShiftedOperator& A, double gamma = 0.5,
                           FlipMode mode = FlipMode::Flip,
                           LanczosOptions opts = {} )
{
    if ( !(gamma > 0.0 && gamma < 1.0) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::make_pd: gamma must lie in (0, 1), got "
            + std::to_string(gamma));
    }
    const double threshold = -gamma * A.a0;
    const double c = ( mode == FlipMode::Flip ) ? 2.0 : 1.0;

    FlipReport report;
    std::vector<double> flipped_values;
    double corrected_min = std::numeric_limits<double>::infinity();
    int budget = opts.max_iters;
    unsigned round_seed = opts.seed;

    while ( budget > 0 )
    {
        LanczosOptions round = opts;
        round.max_iters = std::min(budget, std::max(40, opts.max_iters / 4));
        round.seed = round_seed++;
        const PencilSweep sweep = pencil_sweep(A, round);
        budget -= sweep.iterations;
        report.iterations += sweep.iterations;
        if ( sweep.values.size() == 0 )
        {
            break;  // nothing left outside the block
        }

        // flip every converged mode below the threshold
        std::vector<Eigen::Index> below;
        for ( Eigen::Index i = 0; i < sweep.values.size(); ++i )
        {
            if ( sweep.values(i) < threshold
                 && sweep.converged(i, opts.ritz_tol) )
            {
                below.push_back(i);
            }
        }
        if ( !below.empty() )
        {
            Eigen::MatrixXd V_new(A.dim(),
                                  static_cast<Eigen::Index>(below.size()));
            Eigen::MatrixXd Cc = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(below.size()),
                static_cast<Eigen::Index>(below.size()));
            Eigen::MatrixXd Cs = Cc;
            for ( std::size_t k = 0; k < below.size(); ++k )
            {
                const double lambda = sweep.values(below[k]);
                const Eigen::Index at = static_cast<Eigen::Index>(k);
                V_new.col(at) = sweep.vectors.col(below[k]);
                Cc(at, at) = -c * lambda;          // the surgery on B
                Cs(at, at) = (1.0 - c) * lambda;   // the corrected content
                flipped_values.push_back(lambda);
                corrected_min = std::min(corrected_min, (1.0 - c) * lambda);
            }
            merge(A.block, A.hr, V_new, Cc, Cs, Provenance::Flip);
            continue;  // re-sweep the now-deflated operator
        }

        // nothing (left) to flip: certified when the leftmost pair resolved
        if ( sweep.converged(0, opts.ritz_tol) )
        {
            report.certified = true;
            report.lambda_floor = std::min(sweep.values(0), corrected_min);
            A.lambda_floor = report.lambda_floor;
            A.gamma = gamma;
            // baseline for the warning-zone certificate: corrections added
            // AFTER this moment are exactly C_corr minus this snapshot
            A.C_corr_certified = A.block.C_corr;
            break;
        }
        // leftmost not resolved: spend more budget from a fresh start
    }

    report.flipped = static_cast<int>(flipped_values.size());
    if ( !flipped_values.empty() )
    {
        report.leftmost_before =
            *std::min_element(flipped_values.begin(), flipped_values.end());
        report.flipped_values = Eigen::Map<Eigen::VectorXd>(
            flipped_values.data(),
            static_cast<Eigen::Index>(flipped_values.size()));
    }
    return report;
}

} // end namespace lgpsf::corrections
