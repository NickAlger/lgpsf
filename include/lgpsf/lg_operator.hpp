#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief `LGOperator`: an operator represented in Laguerre-Gaussian PSF form,
/// and everything that can be done with one.
///
/// **The represented object is not a matrix.** It is
///
///     H~  =  M1 Phi~ M2  +  M1 S
///
/// a sum of two objects of DIFFERENT TYPES -- the same distinction the
/// whitening derivation is built on:
///
///  - **Phi~, a semi-discrete continuum kernel**: for each row a genuine
///    function of the source coordinate, `sum_i c_i phi_i(x; theta_rho)`.
///    Rectangular by nature -- evaluable between arbitrary point sets,
///    meaningful on other meshes.
///  - **S, a sparse dof-tied discrete correction**: today a diagonal. Square by
///    nature -- it was DEFINED as the part of the point-spread function the mesh
///    cannot resolve, so it has no off-grid meaning.
///
/// **This header knows nothing about fitting.** An `LGOperator` is a data
/// structure: `operator_fit.hpp` is one producer of them, but a caller with a
/// physics-based approximation can fill one in directly, merge several with
/// `concatenate_rows`, or load one from disk, and everything here works
/// unchanged. `validate` is what makes that path checkable.
///
/// ## Deployed support == the row's window
///
/// Every evaluation is restricted to the row's window and is zero outside it,
/// `eval_kernel` included. For an operator that came from a fit this is not a
/// convenience: the fit's objective evaluates the basis ONLY on the window, so
/// out-of-window model mass is not merely unverified -- it is UNPENALIZED, and
/// the optimizer will spend it to chase in-window noise. At PIG field scale one
/// such row carried 94% of a whole-operator test error. `eval_kernel_unrestricted`
/// is the named opt-out, so asking for extrapolation requires saying so.
///
/// One rule covers every helper:
///
///     support  =  the row's window  intersect  the fitted tau-ellipsoid
///
/// with `truncation_tau` defaulting to infinity. `eval_kernel` applies the
/// window as a REGION (it answers at points that need not be mesh columns);
/// the dof-context helpers apply it by index; `assemble_sparse(op, tau)` is the
/// same rule with a finite tau.
///
/// ## Free functions, deliberately
///
/// Methods for "what am I" -- `num_rows`, `has_model`, `row_window`,
/// `row_modes`. Free functions for "what can be done with me". Nothing here
/// needs privileged access, the type is a transparent flat-array structure
/// meant to be inspected, gathered across MPI ranks and serialized, and a
/// caller's own `foo(const LGOperator&)` should be a peer of ours rather than
/// a second-class citizen.

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ellipsoid_tree/detail/parallel_for.hpp>
#include <ellipsoid_tree/geometry.hpp>
#include <ellipsoid_tree/object_tree.hpp>

#include "lgpsf/ellipsoid_transform.hpp"
#include "lgpsf/lg_ellipsoid_feature.hpp"
#include "lgpsf/lg_functions.hpp"

namespace lgpsf {

/// An operator in Laguerre-Gaussian PSF form: the compressed, self-contained
/// representation of `H~ = M1 Phi~ M2 + M1 S`, and the only thing any consumer
/// of the approximation needs.
///
/// About `(P + m + 1)` doubles per row -- the parametric form IS the compressed
/// operator, and every matrix format is a decompression of it. Rows with no
/// shipped model hold NaN and `mode_set_id == -1`.
///
/// Deliberately independent of how it was produced. Nothing here is about a
/// fit; `operator_fit.hpp` is one producer, but a physics-based approximation
/// can fill one in directly. It carries its own copies of the coordinates and
/// masses, so it is self-contained and serializable on its own.
struct LGOperator
{
    int dim = 0;
    Eigen::MatrixXd x_cols;                 ///< (K_all, N)
    std::optional<Eigen::MatrixXd> x_rows;  ///< (R_all, N), or unset when square
    Eigen::VectorXd m1_diag;                ///< (R_all,)
    Eigen::VectorXd m2_diag;                ///< (K_all,)
    bool spike = true;

    /// (R_all, P) fitted parameters in the PUBLIC absolute encoding, so every
    /// row decodes with `unpack_theta` alone -- no mu0, no mode.
    Eigen::MatrixXd theta;
    Eigen::MatrixXd mu;  ///< (R_all, N)
    Eigen::MatrixXd L;   ///< (R_all, N*N), row-major (i, j) per row

    Eigen::MatrixXd c;                        ///< (R_all, m_max), zero-padded
    std::vector<int> mode_set_id;             ///< (R_all,), -1 = no model
    std::vector<std::vector<Mode>> mode_sets; ///< The distinct mode lists in use
    Eigen::VectorXd s;                        ///< (R_all,) additive spike coefficients

    /// The fit window as a REGION, one ellipsoid per row, already scaled so
    /// membership is `(x - center)^T covariance^-1 (x - center) <= 1`.
    ///
    /// Stored because the window has to be answerable at points that are not
    /// mesh columns -- `eval_kernel` takes arbitrary coordinates, and the index
    /// list below cannot restrict those. On mesh columns the two agree exactly,
    /// by construction. NaN for rows with no window.
    Eigen::MatrixXd window_center;      ///< (R_all, N)
    Eigen::MatrixXd window_covariance;  ///< (R_all, N*N), row-major (i, j)

    /// The same window as CSR-style column indices -- a derived cache of the
    /// dual descent, and the fast path for the dof-context helpers.
    std::vector<int> window_indptr;
    std::vector<int> window_indices;

    Eigen::Index num_rows() const { return m1_diag.size(); }
    Eigen::Index num_cols() const { return m2_diag.size(); }

    /// Does row rho carry a shipped model?
    bool has_model( int rho ) const
    {
        return mode_set_id[static_cast<std::size_t>(rho)] >= 0;
    }

    /// Row rho's fit-window column indices, sorted.
    std::vector<int> row_window( int rho ) const
    {
        return std::vector<int>(window_indices.begin() + window_indptr[rho],
                                window_indices.begin() + window_indptr[rho + 1]);
    }

    /// The mode list row rho's coefficient prefix corresponds to.
    const std::vector<Mode>& row_modes( int rho ) const
    {
        if ( !has_model(rho) )
        {
            throw std::invalid_argument(
                "lgpsf::LGOperator::row_modes: row " + std::to_string(rho)
                + " has no fitted model");
        }
        return mode_sets[static_cast<std::size_t>(
            mode_set_id[static_cast<std::size_t>(rho)])];
    }
};


// ---------------------------------------------------------------------------
// Helpers, each typed to the component(s) it touches.
//
// DEPLOYED SUPPORT == FIT WINDOW. Every dof-context helper below restricts row
// rho to `fit.row_window(rho)` and is zero outside it. Windowed
// cross-validation is blind to out-of-window model energy, and
// polynomial-times-Gaussian modes extrapolate violently past the data, so the
// object that was scored and the object that ships have to be the same one.
// `eval_kernel` is the exception by design: it is component access to the raw
// smooth kernel at arbitrary points, not the deployed operator.
// ---------------------------------------------------------------------------

/// Rows carrying a shipped model.
///
/// Defined by `mode_set_id >= 0` rather than by the fit status, which lives in
/// the diagnostics: an operator has to know its own modeled rows without being
/// told how the fit went. The two agree exactly -- a test pins it.
inline std::vector<int> model_rows( const LGOperator& fit )
{
    std::vector<int> rows;
    for ( std::size_t rho = 0; rho < fit.mode_set_id.size(); ++rho )
    {
        if ( fit.mode_set_id[rho] >= 0 )
        {
            rows.push_back(static_cast<int>(rho));
        }
    }
    return rows;
}

namespace detail {

/// The raw smooth component of one row, together with the fitted pullback the
/// truncation tests need -- both from a single evaluation.
struct KernelAt
{
    Eigen::VectorXd values;  ///< (Q,)
    Eigen::MatrixXd u;       ///< (Q, N), so ||u_q|| is the fitted Mahalanobis radius
};

inline KernelAt kernel_at( const LGOperator& fit, int rho,
                           const Eigen::Ref<const Eigen::MatrixXd>& x_query )
{
    const std::vector<Mode>& modes = fit.row_modes(rho);
    // The stored theta is the public absolute encoding, so splitting off its
    // center gives exactly the pinned pair the feature layer wants.
    const std::pair<Eigen::VectorXd, Eigen::VectorXd> split =
        freeze_mu(fit.theta.row(rho).transpose());
    FeatureAt at(split.first, x_query, modes, split.second, MuMode::Pinned);
    KernelAt out;
    out.values = at.values()
                 * fit.c.row(rho).head(static_cast<Eigen::Index>(modes.size()))
                       .transpose();
    out.u = at.u();
    return out;
}

/// Squared Mahalanobis radius of each query point in row rho's WINDOW
/// ellipsoid; a point is inside the window when this is at most 1.
inline Eigen::VectorXd window_radius2( const LGOperator& fit, int rho,
                                       const Eigen::Ref<const Eigen::MatrixXd>& x_query )
{
    Eigen::MatrixXd shape(fit.dim, fit.dim);
    for ( int i = 0; i < fit.dim; ++i )
    {
        for ( int j = 0; j < fit.dim; ++j )
        {
            shape(i, j) = fit.window_covariance(rho, i * fit.dim + j);
        }
    }
    const Eigen::MatrixXd centered =
        x_query.rowwise() - fit.window_center.row(rho);
    const Eigen::MatrixXd whitened =
        shape.llt().matrixL().solve(centered.transpose());
    return whitened.colwise().squaredNorm().transpose();
}


/// Is `column` in this row's (sorted) window?
inline bool in_window( const LGOperator& fit, int rho, int column )
{
    const auto begin = fit.window_indices.begin() + fit.window_indptr[static_cast<std::size_t>(rho)];
    const auto end = fit.window_indices.begin() + fit.window_indptr[static_cast<std::size_t>(rho) + 1];
    return std::binary_search(begin, end, column);
}

/// The deployed smooth row on a given set of columns:
/// m1[rho] * m2[j] * Phi~(rho, x_j). No window restriction and no spike --
/// callers apply those.
inline Eigen::VectorXd deployed_smooth(
    const LGOperator& fit, int rho, const std::vector<int>& columns,
    double truncation_tau = std::numeric_limits<double>::infinity() )
{
    Eigen::MatrixXd points(static_cast<Eigen::Index>(columns.size()), fit.dim);
    for ( std::size_t i = 0; i < columns.size(); ++i )
    {
        points.row(static_cast<Eigen::Index>(i)) = fit.x_cols.row(columns[i]);
    }
    const KernelAt at = kernel_at(fit, rho, points);
    Eigen::VectorXd values = at.values;
    const bool trim = !std::isinf(truncation_tau);
    for ( std::size_t i = 0; i < columns.size(); ++i )
    {
        const Eigen::Index q = static_cast<Eigen::Index>(i);
        if ( trim && at.u.row(q).squaredNorm() > truncation_tau * truncation_tau )
        {
            values(q) = 0.0;
        }
        values(q) *= fit.m1_diag(rho) * fit.m2_diag(columns[i]);
    }
    return values;
}

} // end namespace detail

/// [smooth] Phi~(rho, x) at arbitrary query points, RESTRICTED TO THE FIT
/// WINDOW: rectangular by nature, no masses, no spike, zero outside the window.
///
/// The window restriction is the default here, not a convenience. The fit's
/// objective evaluates the basis only on the window, so out-of-window model
/// mass is not merely unverified -- it is UNPENALIZED, and the optimizer will
/// spend it to chase in-window noise. The extension of the fitted form beyond
/// the window is therefore an artifact of an objective that could not see it.
/// Use `eval_kernel_unrestricted` to get it anyway, deliberately.
///
/// Since the query points need not be mesh columns, the window is applied as
/// the stored REGION rather than the column-index list; the two agree exactly
/// on mesh columns.
///
/// `truncation_tau` additionally trims to the fitted kernel's own
/// tau-ellipsoid, `||u|| <= tau` in the pullback coordinate -- tau standard
/// deviations of the fitted kernel, the same tau `assemble_sparse` takes.
/// Infinity (the default) adds nothing.
///
/// `x_query` is (Q, N); the result is (Q, rows.size()), one column per
/// requested row. Throws for a row with no shipped model.
inline Eigen::MatrixXd eval_kernel(
    const LGOperator& fit, const std::vector<int>& rows,
    const Eigen::Ref<const Eigen::MatrixXd>& x_query,
    double truncation_tau = std::numeric_limits<double>::infinity() )
{
    Eigen::MatrixXd out(x_query.rows(), static_cast<Eigen::Index>(rows.size()));
    const bool trim = !std::isinf(truncation_tau);
    for ( std::size_t i = 0; i < rows.size(); ++i )
    {
        const int rho = rows[i];
        const detail::KernelAt at = detail::kernel_at(fit, rho, x_query);
        const Eigen::VectorXd radius2 = detail::window_radius2(fit, rho, x_query);
        Eigen::VectorXd values = at.values;
        for ( Eigen::Index q = 0; q < values.size(); ++q )
        {
            const bool inside =
                radius2(q) <= 1.0
                && (!trim || at.u.row(q).squaredNorm()
                                 <= truncation_tau * truncation_tau);
            if ( !inside )
            {
                values(q) = 0.0;
            }
        }
        out.col(static_cast<Eigen::Index>(i)) = values;
    }
    return out;
}

/// [smooth] The raw parametric smooth component, with NO window restriction.
///
/// Component access, deliberately named so that asking for extrapolation
/// requires saying so. Beyond the fit window this is an extension of the
/// fitted form into a region the fit's objective never evaluated, so it
/// carries no evidence and can be arbitrarily large; see `eval_kernel`.
inline Eigen::MatrixXd eval_kernel_unrestricted(
    const LGOperator& fit, const std::vector<int>& rows,
    const Eigen::Ref<const Eigen::MatrixXd>& x_query )
{
    Eigen::MatrixXd out(x_query.rows(), static_cast<Eigen::Index>(rows.size()));
    for ( std::size_t i = 0; i < rows.size(); ++i )
    {
        out.col(static_cast<Eigen::Index>(i)) =
            detail::kernel_at(fit, rows[i], x_query).values;
    }
    return out;
}

/// [both] Paired entries of the DEPLOYED operator in the dof context.
///
/// `H~[rho, j] = m1[rho] m2[j] Phi~(rho, x_j)` for j in row rho's fit window
/// and zero outside it, plus `m1[rho] s[rho]` when `j == rho` (the spike dof,
/// square context only). `rows` and `cols` are equal-length index arrays;
/// the result holds their values. Rows with no model give zero.
inline Eigen::VectorXd eval_entries(
    const LGOperator& fit, const std::vector<int>& rows,
    const std::vector<int>& cols,
    double truncation_tau = std::numeric_limits<double>::infinity() )
{
    if ( rows.size() != cols.size() )
    {
        throw std::invalid_argument(
            "lgpsf::eval_entries: rows and cols must have the same length");
    }
    Eigen::VectorXd out = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(rows.size()));

    std::map<int, std::vector<std::size_t>> by_row;
    for ( std::size_t i = 0; i < rows.size(); ++i )
    {
        by_row[rows[i]].push_back(i);
    }
    for ( const auto& entry : by_row )
    {
        const int rho = entry.first;
        if ( !fit.has_model(rho) )
        {
            continue;  // no shipped model: the row is zero
        }
        std::vector<int> columns;
        for ( std::size_t i : entry.second )
        {
            columns.push_back(cols[i]);
        }
        const Eigen::VectorXd values =
            detail::deployed_smooth(fit, rho, columns, truncation_tau);
        for ( std::size_t k = 0; k < entry.second.size(); ++k )
        {
            const std::size_t slot = entry.second[k];
            double value = detail::in_window(fit, rho, cols[slot])
                               ? values(static_cast<Eigen::Index>(k))
                               : 0.0;
            if ( fit.spike && cols[slot] == rho )
            {
                value += fit.m1_diag(rho) * fit.s(rho);
            }
            out(static_cast<Eigen::Index>(slot)) = value;
        }
    }
    return out;
}

/// [both] The DEPLOYED operator applied to `v`, with zero assembly.
///
/// Each row's kernel is evaluated on its FIT WINDOW only, which is both the
/// deployed-support invariant and the fast path -- O(sum of window sizes)
/// rather than O(R K). `v` is (K_all, q); rows with no model give zero rows.
inline Eigen::MatrixXd matvec(
    const LGOperator& fit, const Eigen::Ref<const Eigen::MatrixXd>& v,
    double truncation_tau = std::numeric_limits<double>::infinity(),
    int num_threads = 0 )
{
    if ( v.rows() != fit.num_cols() )
    {
        throw std::invalid_argument(
            "lgpsf::matvec: v has " + std::to_string(v.rows()) + " rows but the "
            "operator has " + std::to_string(fit.num_cols()) + " columns");
    }
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(fit.num_rows(), v.cols());
    const std::vector<int> rows = model_rows(fit);
    // Rows write disjoint output rows, so this is bit-identical across thread
    // counts without any reduction.
    ellipsoid_tree::detail::parallel_for(
        0, static_cast<std::ptrdiff_t>(rows.size()),
        [&]( std::ptrdiff_t begin, std::ptrdiff_t end ) {
            for ( std::ptrdiff_t r = begin; r < end; ++r )
            {
                const int rho = rows[static_cast<std::size_t>(r)];
                const std::vector<int> window = fit.row_window(rho);
                if ( window.empty() )
                {
                    continue;
                }
                const Eigen::VectorXd weights =
                    detail::deployed_smooth(fit, rho, window, truncation_tau);
                for ( std::size_t i = 0; i < window.size(); ++i )
                {
                    out.row(rho) +=
                        weights(static_cast<Eigen::Index>(i)) * v.row(window[i]);
                }
                if ( fit.spike )
                {
                    out.row(rho) += fit.m1_diag(rho) * fit.s(rho) * v.row(rho);
                }
            }
        },
        num_threads);
    return out;
}

/// The fitted ellipsoids, as the operator layer's geometry summary.
struct EllipsoidField
{
    Eigen::MatrixXd mu;                   ///< (R_all, N); NaN where no model
    std::vector<Eigen::MatrixXd> sigma;   ///< R_all covariances L L^T
};

/// [smooth] The fitted (mu, Sigma) field -- what `ellipsoid_tree` consumes.
inline EllipsoidField ellipsoid_field( const LGOperator& fit )
{
    EllipsoidField field;
    field.mu = fit.mu;
    field.sigma.reserve(static_cast<std::size_t>(fit.num_rows()));
    for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
    {
        Eigen::MatrixXd L(fit.dim, fit.dim);
        for ( int i = 0; i < fit.dim; ++i )
        {
            for ( int j = 0; j < fit.dim; ++j )
            {
                L(i, j) = fit.L(rho, i * fit.dim + j);
            }
        }
        field.sigma.push_back(L * L.transpose());
    }
    return field;
}

/// What, if anything, `assemble_sparse` does about symmetry.
///
/// An ASSEMBLY POLICY, not a fit property: row fits do not produce a symmetric
/// operator, and rows-as-is versus averaging versus column-consistent
/// reconciliation are consumer decisions with consumer-specific right answers.
enum class Symmetrize
{
    None,    ///< Rows exactly as fitted.
    Average  ///< (A + A^T) / 2; square dof context only.
};

/// [both] Sparse decompression of the DEPLOYED operator.
///
/// Per modeled row: smooth entries on the tau-ellipsoid support of the FITTED
/// kernel, INTERSECTED with that row's fit window -- deployed support == fit
/// support, so the tau truncation only trims the Gaussian tail inside the
/// window it was scored on -- plus the additive spike on the diagonal.
///
/// The support pattern for every row comes from ONE dual-tree descent of the
/// column-point tree against the fitted-ellipsoid tree, the same machinery
/// `fit_operator` uses to derive the windows themselves.
inline Eigen::SparseMatrix<double> assemble_sparse(
    const LGOperator& fit, double tau, Symmetrize symmetrize = Symmetrize::None,
    int num_threads = 0 )
{
    if ( !(tau > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::assemble_sparse: tau must be positive, got " + std::to_string(tau));
    }
    if ( symmetrize == Symmetrize::Average
         && (fit.x_rows || fit.num_rows() != fit.num_cols()) )
    {
        throw std::invalid_argument(
            "lgpsf::assemble_sparse: averaging needs the square dof context");
    }

    Eigen::SparseMatrix<double> assembled(fit.num_rows(), fit.num_cols());
    const std::vector<int> rows = model_rows(fit);
    if ( rows.empty() )
    {
        assembled.makeCompressed();
        return assembled;
    }

    const EllipsoidField field = ellipsoid_field(fit);
    std::vector<ellipsoid_tree::Ellipsoid> kernels;
    kernels.reserve(rows.size());
    for ( int rho : rows )
    {
        kernels.push_back(ellipsoid_tree::Ellipsoid{
            field.mu.row(rho).transpose(), field.sigma[static_cast<std::size_t>(rho)]});
    }
    const ellipsoid_tree::EllipsoidTree kernel_tree(std::move(kernels), tau);

    std::vector<ellipsoid_tree::Ball> points;
    points.reserve(static_cast<std::size_t>(fit.num_cols()));
    for ( Eigen::Index j = 0; j < fit.num_cols(); ++j )
    {
        points.push_back(ellipsoid_tree::Ball{fit.x_cols.row(j).transpose(), 0.0});
    }
    const ellipsoid_tree::BallTree column_tree(std::move(points));

    std::vector<std::vector<int>> support(rows.size());
    for ( const std::pair<int, int>& hit :
          ellipsoid_tree::collision_pairs(column_tree, kernel_tree) )
    {
        support[static_cast<std::size_t>(hit.second)].push_back(hit.first);
    }

    // Per-row triplet buffers, concatenated in ROW ORDER afterwards: the
    // pattern and the summation order are then independent of scheduling, so
    // the assembled matrix is bit-identical across thread counts.
    std::vector<std::vector<Eigen::Triplet<double>>> per_row(rows.size());
    ellipsoid_tree::detail::parallel_for(
        0, static_cast<std::ptrdiff_t>(rows.size()),
        [&]( std::ptrdiff_t begin, std::ptrdiff_t end ) {
            for ( std::ptrdiff_t e = begin; e < end; ++e )
            {
                const std::size_t slot = static_cast<std::size_t>(e);
                const int rho = rows[slot];
                std::vector<int>& columns = support[slot];
                std::sort(columns.begin(), columns.end());

                const std::vector<int> window = fit.row_window(rho);
                std::vector<int> deployed;
                std::set_intersection(columns.begin(), columns.end(),
                                      window.begin(), window.end(),
                                      std::back_inserter(deployed));

                if ( !deployed.empty() )
                {
                    const Eigen::VectorXd values =
                        detail::deployed_smooth(fit, rho, deployed, tau);
                    for ( std::size_t i = 0; i < deployed.size(); ++i )
                    {
                        per_row[slot].emplace_back(
                            rho, deployed[i], values(static_cast<Eigen::Index>(i)));
                    }
                }
                if ( fit.spike )
                {
                    // Additive on top of the unmodified smooth part at the
                    // diagonal; duplicate triplets sum, which is that convention.
                    per_row[slot].emplace_back(rho, rho,
                                               fit.m1_diag(rho) * fit.s(rho));
                }
            }
        },
        num_threads);

    std::vector<Eigen::Triplet<double>> triplets;
    for ( const std::vector<Eigen::Triplet<double>>& block : per_row )
    {
        triplets.insert(triplets.end(), block.begin(), block.end());
    }
    assembled.setFromTriplets(triplets.begin(), triplets.end());
    if ( symmetrize == Symmetrize::Average )
    {
        assembled = 0.5 * (Eigen::SparseMatrix<double>(assembled)
                           + Eigen::SparseMatrix<double>(assembled.transpose()));
    }
    assembled.makeCompressed();
    return assembled;
}

/// [both] Per-row relative residual against held-out probes -- the scorecard.
///
/// `||H~[rho] V_qc - HV_qc[rho]|| / ||HV_qc[rho]||`, NaN for rows with no model.
inline Eigen::VectorXd qc_map( const LGOperator& fit,
                               const Eigen::Ref<const Eigen::MatrixXd>& V_qc,
                               const Eigen::Ref<const Eigen::MatrixXd>& HV_qc,
                               int num_threads = 0 )
{
    if ( HV_qc.rows() != fit.num_rows() || HV_qc.cols() != V_qc.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::qc_map: HV_qc must be (num_rows, num_probes) to match V_qc");
    }
    const Eigen::MatrixXd predicted =
        matvec(fit, V_qc, std::numeric_limits<double>::infinity(), num_threads);
    Eigen::VectorXd out = Eigen::VectorXd::Constant(
        fit.num_rows(), std::numeric_limits<double>::quiet_NaN());
    for ( int rho : model_rows(fit) )
    {
        const double scale = std::max(HV_qc.row(rho).norm(), 1e-300);
        out(rho) = (predicted.row(rho) - HV_qc.row(rho)).norm() / scale;
    }
    return out;
}

/// Structural problems found in an operator, one message each. Empty means the
/// structure is self-consistent.
///
/// This checks SHAPE and INTERNAL CONSISTENCY, not whether the operator is a
/// good approximation of anything -- that is what `qc_map` is for.
inline std::vector<std::string> validate( const LGOperator& fit )
{
    std::vector<std::string> problems;
    const auto complain = [&problems]( const std::string& message ) {
        problems.push_back(message);
    };

    const Eigen::Index rows = fit.m1_diag.size();
    const Eigen::Index cols = fit.m2_diag.size();
    if ( fit.dim < 1 )
    {
        complain("dim must be at least 1");
        return problems;  // nothing below is meaningful without it
    }
    if ( fit.x_cols.rows() != cols || fit.x_cols.cols() != fit.dim )
    {
        complain("x_cols must be (num_cols, dim)");
    }
    if ( fit.x_rows && (fit.x_rows->rows() != rows || fit.x_rows->cols() != fit.dim) )
    {
        complain("x_rows must be (num_rows, dim) when present");
    }
    if ( fit.spike && fit.x_rows )
    {
        complain("the spike needs the square dof context, so it cannot coexist "
                 "with separate row coordinates");
    }

    const auto expect_rows = [&]( Eigen::Index got, const char* what ) {
        if ( got != rows )
        {
            complain(std::string(what) + " must have one entry per row");
        }
    };
    expect_rows(fit.theta.rows(), "theta");
    expect_rows(fit.mu.rows(), "mu");
    expect_rows(fit.L.rows(), "L");
    expect_rows(fit.c.rows(), "c");
    expect_rows(fit.s.size(), "s");
    expect_rows(static_cast<Eigen::Index>(fit.mode_set_id.size()), "mode_set_id");
    expect_rows(fit.window_center.rows(), "window_center");
    expect_rows(fit.window_covariance.rows(), "window_covariance");

    if ( fit.theta.cols() != theta_size(fit.dim) )
    {
        complain("theta must be (num_rows, N(N+3)/2) -- the public absolute encoding");
    }
    if ( fit.mu.cols() != fit.dim || fit.window_center.cols() != fit.dim )
    {
        complain("mu and window_center must have dim columns");
    }
    if ( fit.L.cols() != fit.dim * fit.dim
         || fit.window_covariance.cols() != fit.dim * fit.dim )
    {
        complain("L and window_covariance must have dim*dim columns");
    }

    if ( static_cast<Eigen::Index>(fit.window_indptr.size()) != rows + 1 )
    {
        complain("window_indptr must have num_rows + 1 entries");
    }
    else
    {
        if ( fit.window_indptr.front() != 0
             || fit.window_indptr.back()
                    != static_cast<int>(fit.window_indices.size()) )
        {
            complain("window_indptr must start at 0 and end at window_indices.size()");
        }
        for ( std::size_t i = 1; i < fit.window_indptr.size(); ++i )
        {
            if ( fit.window_indptr[i] < fit.window_indptr[i - 1] )
            {
                complain("window_indptr must be non-decreasing");
                break;
            }
        }
    }
    for ( int column : fit.window_indices )
    {
        if ( column < 0 || column >= cols )
        {
            complain("window_indices holds an out-of-range column");
            break;
        }
    }

    for ( Eigen::Index rho = 0; rho < rows; ++rho )
    {
        const int id = fit.mode_set_id[static_cast<std::size_t>(rho)];
        if ( id < -1 || id >= static_cast<int>(fit.mode_sets.size()) )
        {
            complain("mode_set_id " + std::to_string(id) + " on row "
                     + std::to_string(rho) + " does not name a mode set");
            break;
        }
        if ( id >= 0
             && fit.c.cols()
                    < static_cast<Eigen::Index>(
                          fit.mode_sets[static_cast<std::size_t>(id)].size()) )
        {
            complain("c is narrower than row " + std::to_string(rho)
                     + "'s mode set, so its coefficients cannot be stored");
            break;
        }
    }
    return problems;
}

/// Rows of several operators over the SAME columns, end to end.
///
/// The reason this exists rather than being open-coded: merging chunk fits
/// means remapping every row-indexed array at once -- `mode_set_id` into a
/// combined mode-set table, and `window_indptr` by a running offset -- and
/// getting either wrong is silent. The column geometry, masses and spike flag
/// must agree across the parts, since they describe the space being mapped
/// from rather than the rows being mapped to.
inline LGOperator concatenate_rows( const std::vector<LGOperator>& parts )
{
    if ( parts.empty() )
    {
        throw std::invalid_argument("lgpsf::concatenate_rows: nothing to concatenate");
    }
    const LGOperator& first = parts.front();
    Eigen::Index total_rows = 0;
    Eigen::Index widest = 0;
    for ( const LGOperator& part : parts )
    {
        if ( part.dim != first.dim || part.m2_diag.size() != first.m2_diag.size()
             || part.x_cols != first.x_cols || part.m2_diag != first.m2_diag
             || part.spike != first.spike )
        {
            throw std::invalid_argument(
                "lgpsf::concatenate_rows: the parts must share the same column "
                "geometry, column masses and spike setting");
        }
        if ( part.x_rows.has_value() != first.x_rows.has_value() )
        {
            throw std::invalid_argument(
                "lgpsf::concatenate_rows: the parts must agree on whether row "
                "coordinates are separate from column coordinates");
        }
        total_rows += part.num_rows();
        widest = std::max(widest, part.c.cols());
    }

    LGOperator out;
    out.dim = first.dim;
    out.x_cols = first.x_cols;
    out.m2_diag = first.m2_diag;
    out.spike = first.spike;
    out.m1_diag.resize(total_rows);
    out.theta.resize(total_rows, theta_size(first.dim));
    out.mu.resize(total_rows, first.dim);
    out.L.resize(total_rows, first.dim * first.dim);
    out.c = Eigen::MatrixXd::Zero(total_rows, widest);
    out.s.resize(total_rows);
    out.window_center.resize(total_rows, first.dim);
    out.window_covariance.resize(total_rows, first.dim * first.dim);
    out.mode_set_id.resize(static_cast<std::size_t>(total_rows));
    out.window_indptr.assign(static_cast<std::size_t>(total_rows) + 1, 0);
    if ( first.x_rows )
    {
        out.x_rows = Eigen::MatrixXd(total_rows, first.dim);
    }

    Eigen::Index row = 0;
    for ( const LGOperator& part : parts )
    {
        // Mode sets are de-duplicated across parts, so ids stay dense and a
        // merged operator is no larger than it needs to be.
        std::vector<int> remap(part.mode_sets.size(), -1);
        for ( std::size_t i = 0; i < part.mode_sets.size(); ++i )
        {
            const auto found = std::find(out.mode_sets.begin(), out.mode_sets.end(),
                                         part.mode_sets[i]);
            if ( found == out.mode_sets.end() )
            {
                remap[i] = static_cast<int>(out.mode_sets.size());
                out.mode_sets.push_back(part.mode_sets[i]);
            }
            else
            {
                remap[i] = static_cast<int>(found - out.mode_sets.begin());
            }
        }

        for ( Eigen::Index local = 0; local < part.num_rows(); ++local, ++row )
        {
            out.m1_diag(row) = part.m1_diag(local);
            out.theta.row(row) = part.theta.row(local);
            out.mu.row(row) = part.mu.row(local);
            out.L.row(row) = part.L.row(local);
            out.c.row(row).head(part.c.cols()) = part.c.row(local);
            out.s(row) = part.s(local);
            out.window_center.row(row) = part.window_center.row(local);
            out.window_covariance.row(row) = part.window_covariance.row(local);
            if ( first.x_rows )
            {
                out.x_rows->row(row) = part.x_rows->row(local);
            }
            const int id = part.mode_set_id[static_cast<std::size_t>(local)];
            out.mode_set_id[static_cast<std::size_t>(row)] =
                ( id < 0 ) ? -1 : remap[static_cast<std::size_t>(id)];

            const std::vector<int> window =
                part.row_window(static_cast<int>(local));
            out.window_indptr[static_cast<std::size_t>(row) + 1] =
                out.window_indptr[static_cast<std::size_t>(row)]
                + static_cast<int>(window.size());
            out.window_indices.insert(out.window_indices.end(), window.begin(),
                                      window.end());
        }
    }
    return out;
}

/// [spike] The Dirac mass field `m_rho * s_rho` -- the mesh-independent
/// content of the S component. A map of it is a resolution diagnostic: where
/// it is large, the mesh is starving.
inline Eigen::VectorXd spike_measure( const LGOperator& fit )
{
    return fit.m1_diag.cwiseProduct(fit.s);
}

} // end namespace lgpsf
