#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Whole-operator fitting: the per-target probe fit run over the rows
/// of an implicitly available operator.
///
/// **The fitted object is not a matrix.** It is
///
///     H~  =  M1 Phi~ M2  +  M1 S
///
/// a sum of two objects of DIFFERENT TYPES -- the same distinction the
/// whitening derivation is built on:
///
///  - **Phi~, a semi-discrete continuum kernel**: for each fitted row a
///    genuine function of the source coordinate. Rectangular by nature --
///    evaluable between arbitrary point sets, meaningful on other meshes.
///  - **S, a sparse dof-tied discrete correction**: today a diagonal. Square
///    by nature -- it was DEFINED as the part of the point-spread function the
///    mesh cannot resolve, so it has no off-grid meaning.
///
/// **The per-row protocol**: gate -> window -> fit -> baseline guard -> status.
///
/// The BASELINE GUARD is always on. A plain linear fit at the caller's
/// `sigma[rho]`, pinned at `mu0[rho]` -- one least-squares solve, no outer
/// loop -- is scored on the same folds as the search, and the searched fit
/// ships only if it STRICTLY beats it. So the method is never worse than the
/// a-priori-Gaussian status quo it replaces, by construction.
///
/// **The deployed operator is window-restricted.** Windowed cross-validation
/// is blind to out-of-window model energy, and polynomial-times-Gaussian modes
/// extrapolate violently beyond the data -- at PIG field scale one rogue row
/// carried 94% of a whole-operator test error. Every dof-context helper
/// restricts a row to its FIT WINDOW, stored here as CSR-style index arrays:
/// fitted object == deployed object, which is what makes the per-row scores
/// honest for deployment.
///
/// ## The window: an ellipsoid by default, a ball on request
///
/// The window is `{x : (x - mu0)^T sigma^-1 (x - mu0) <= tau_window^2}`, the
/// caller's best-guess ellipsoid inflated by `tau_window`. Inflating an
/// ellipsoid preserves its aspect and orientation exactly, which is what the
/// row layer's window-shape initialization family is built to exploit.
///
/// How much of that anisotropy to keep is ONE CONTINUOUS KNOB,
/// `window_aspect_cap`, which caps the window's axis ratio by flooring the
/// eigenvalues of `sigma` at `lambda_max / cap^2`:
///
///     cap = 1          every axis becomes lambda_max: an isotropic window,
///                      i.e. the ellipsoid's bounding sphere at the same tau
///     cap = infinity   the floor is zero: the caller's ellipsoid, untouched
///     cap = kappa      the axis ratio is min(the prior's, kappa)
///
/// so the two endpoints are the ball and the ellipsoid and everything between
/// is reachable. The query is always an ellipsoid,
/// `{x : (x - mu0)^T A^-1 (x - mu0) <= tau_window^2}` -- only `A` changes --
/// so the cap moves the SHAPE and never the scale, and `tau_window` means the
/// same thing at every setting.
///
/// What the knob really trades is how much trust is placed in the prior's
/// ORIENTATION. A sphere is conservative in every direction regardless of
/// whether the prior points the right way; the caller's ellipsoid is
/// conservative only along the axes the prior nominates. At cap kappa the
/// window still extends `tau_window * a_max / kappa` in its narrowest
/// direction, which bounds the damage from a badly rotated prior while taking
/// most of the point-count saving -- and the saving is the product of the
/// capped axis ratios, so it compounds with dimension.
///
/// (Flooring eigenvalues to bound an aspect ratio is the same device
/// `window_shape` already uses on degenerate windows.)
///
/// Windows for ALL rows come from ONE dual-tree descent -- the column-point
/// tree against a tree of every row's query ellipsoid -- rather than a query
/// per row, the same way the deployed sparsity pattern is derived.
///
/// **History, because it matters for reading the evidence.** The design
/// intent was always the ellipsoid (`8e64243`: "in the intended pipeline it IS
/// a conservatively inflated ellipsoid, whose aspect and orientation survive
/// the inflation"). The Python prototype's operator layer nevertheless shipped
/// a ball, and every PIG field-scale experiment that validated the method ran
/// with the ball -- and with `window_shape_rungs` switched off, which is the
/// right thing to do when the window is a sphere and that family can recover
/// nothing from it. So: **the ball is known to perform well and the ellipsoid
/// is not yet measured at field scale.** The C++ side restores the intended
/// default and reaches the ball by setting `window_aspect_cap = 1`, so the
/// comparison -- and everything between the two -- can be run cheaply once the
/// port is finished. Until then, treat neither endpoint as settled.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iterator>
#include <map>
#include <optional>
#include <tuple>
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
#include "lgpsf/init_dictionary.hpp"
#include "lgpsf/lg_ellipsoid_feature.hpp"
#include "lgpsf/mode_policy.hpp"
#include "lgpsf/probe_fit.hpp"
#include "lgpsf/whitening.hpp"

namespace lgpsf {

/// `sigma` with its axis ratio capped at `aspect_cap`, by flooring the
/// eigenvalues at `lambda_max / aspect_cap^2`.
///
/// The eigenvectors are untouched, so the orientation always survives; only
/// how elongated the shape may be changes. `aspect_cap == 1` returns
/// `lambda_max * I` (isotropic), `aspect_cap == infinity` returns `sigma`
/// unchanged, and the resulting axis ratio is `min(the input's, aspect_cap)`.
inline Eigen::MatrixXd capped_covariance( const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                                          double aspect_cap )
{
    if ( !(aspect_cap >= 1.0) )
    {
        throw std::invalid_argument(
            "lgpsf::capped_covariance: aspect_cap must be >= 1 (1 is an isotropic "
            "window); got " + std::to_string(aspect_cap));
    }
    if ( std::isinf(aspect_cap) )
    {
        return sigma;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sigma);
    Eigen::VectorXd values = solver.eigenvalues();  // ascending
    const double floor = values(values.size() - 1) / (aspect_cap * aspect_cap);
    values = values.cwiseMax(floor);
    return solver.eigenvectors() * values.asDiagonal()
           * solver.eigenvectors().transpose();
}

/// What happened to a row.
enum class RowStatus
{
    Fit,               ///< The searched fit beat the baseline and shipped.
    GatedOut,          ///< The caller's gate excluded this row.
    FallbackBaseline,  ///< The baseline shipped; the search did not beat it.
    Failed             ///< The row raised; see `failures`.
};

/// Why a row's search stopped. Extends the row layer's reasons with the case
/// where no search was possible at all.
enum class RowStop
{
    None,
    Target,
    ModePatience,
    Exhausted,
    SearchInfeasible  ///< No mode set passed the counting rule for a search.
};

inline const char* to_string( RowStatus status )
{
    switch ( status )
    {
        case RowStatus::Fit:              return "fit";
        case RowStatus::GatedOut:         return "gated_out";
        case RowStatus::FallbackBaseline: return "fallback_baseline";
        case RowStatus::Failed:           return "failed";
    }
    return "unknown";
}

struct OperatorFitConfig
{
    /// Window inflation, in standard deviations of the caller's `sigma`.
    ///
    /// This is where conservatism comes from, and the row layer's
    /// window-containment admissibility guard is only as valid as this is
    /// conservative: at the default 10, a best-guess sigma underestimating by
    /// 3x still leaves ~3 true sigmas of margin. That justification travels
    /// with the parameter.
    double tau_window = 10.0;

    /// Caps the window's axis ratio: 1 gives an isotropic window (a ball),
    /// infinity (the default) gives the caller's ellipsoid untouched, and any
    /// value between trades point count against robustness to a badly oriented
    /// prior. See the file comment.
    double window_aspect_cap = std::numeric_limits<double>::infinity();

    /// Model the diagonal spike. Requires the square dof context (no separate
    /// row coordinates), since the spike is tied to the row's own dof.
    bool spike = true;

    /// The per-row candidate-stream policy. Its `split` and `jitter` are
    /// OVERWRITTEN here: the operator layer owns them, so every row and the
    /// baseline guard score on identical folds.
    ProbeFitConfig row;

    /// Unset (the default) gives deterministic round-robin folds and a
    /// deterministic jitter table -- the whole fit is then reproducible with
    /// nothing to remember. Setting it permutes the folds, as an explicit
    /// check that a result is not an artifact of one partition.
    std::optional<std::uint32_t> seed;

    /// 0 lets the implementation choose. Results are bit-identical across
    /// thread counts by construction: rows write disjoint slots, and every
    /// shared registry is built serially in row order afterwards.
    int num_threads = 0;
};

/// The fitted operator itself: the compressed, self-contained representation
/// of `H~ = M1 Phi~ M2 + M1 S`, and the only thing any consumer of the
/// approximation needs.
///
/// About `(P + m + 1)` doubles per row -- the parametric form IS the compressed
/// operator, and every matrix format is a decompression of it. Rows with no
/// shipped model hold NaN and `mode_set_id == -1`.
///
/// Deliberately separate from the diagnostics of the fit that produced it.
/// Nothing here is about HOW the fit went; someone merging chunk fits, loading
/// one from disk, or building one from a different method should not have to
/// invent a stop reason. It carries its own copies of the coordinates and
/// masses, so it is self-contained and serializable on its own.
struct FittedOperator
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
                "lgpsf::FittedOperator::row_modes: row " + std::to_string(rho)
                + " has no fitted model");
        }
        return mode_sets[static_cast<std::size_t>(
            mode_set_id[static_cast<std::size_t>(rho)])];
    }
};

/// Per-row provenance for the fit that produced a FittedOperator: how each row
/// went, not what it produced.
///
/// Every array is indexed by row, aligned with the operator's. Nothing here is
/// read by any evaluation -- that separation is what the split is for, and a
/// test pins it.
struct FitDiagnostics
{
    Eigen::VectorXd score;           ///< (R_all,) CV score of the shipped model
    Eigen::VectorXd baseline_score;  ///< (R_all,) CV score of the baseline
    std::vector<RowStop> stop_reason;
    std::vector<char> released;      ///< Where the shipped model's center was fitted
    std::vector<RowStatus> status;
    std::map<int, std::string> failures;  ///< Row -> message, for failed rows

    OperatorFitConfig config;  ///< Provenance echo.
};

/// What `fit_operator` returns: the operator, and how it went.
struct OperatorFit
{
    FittedOperator model;
    FitDiagnostics diagnostics;
};

namespace detail {

/// Full-data linear coefficients at fixed parameters -- the baseline's single
/// least-squares solve. Mirrors linear_cv_score's design matrix exactly.
template <typename Basis>
inline std::pair<Eigen::VectorXd, Eigen::VectorXd> linear_fit(
    const Eigen::MatrixXd& z_hat, const Eigen::VectorXd& y_hat,
    const Basis& basis, const Eigen::VectorXd& theta_hat,
    const Eigen::MatrixXd& e_hat )
{
    const Eigen::MatrixXd values = basis(theta_hat).values();
    Eigen::MatrixXd design(z_hat.cols(), values.cols() + e_hat.cols());
    design.leftCols(values.cols()) = z_hat.transpose() * values;
    if ( e_hat.cols() > 0 )
    {
        design.rightCols(e_hat.cols()) = z_hat.transpose() * e_hat;
    }
    const Eigen::VectorXd all =
        design.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y_hat);
    return {all.head(values.cols()), all.tail(e_hat.cols())};
}

/// Everything one row produced, before the serial gather. Kept per row so the
/// parallel phase writes only disjoint slots.
struct RowOutcome
{
    RowStatus status = RowStatus::GatedOut;
    RowStop stop = RowStop::None;
    std::vector<int> window;
    Eigen::VectorXd theta;
    Eigen::VectorXd mu;
    Eigen::MatrixXd L;
    Eigen::VectorXd c;
    std::vector<Mode> modes;
    double s = 0.0;
    double score = std::numeric_limits<double>::quiet_NaN();
    double baseline_score = std::numeric_limits<double>::quiet_NaN();
    bool released = false;
    std::string failure;
};

} // end namespace detail

/// Fit the parametric approximation from raw probes and responses.
///
/// @param x_cols  (K_all, N) column-dof coordinates.
/// @param m1_diag (R_all,) row masses. @param m2_diag (K_all,) column masses.
/// @param V       (K_all, k) raw random probes.
/// @param HV      (R_all, k) raw responses, `HV(rho, l) = (H V.col(l))(rho)`.
/// @param sigma   R_all covariances: the caller's BEST GUESS at each bump's
///                shape, NOT required to be conservative. Three consumers --
///                the sigma0 initialization rung, the baseline fit, and the
///                window (made conservative by `tau_window`).
/// @param mu0     (R_all, N) reference centers; unset defaults to the row
///                dofs' own coordinates.
/// @param x_rows  (R_all, N) row-dof coordinates; unset means the square
///                context, where row dof rho IS column dof rho.
/// @param gate    (R_all,) which rows to attempt; unset means all. Gated rows
///                get a status, not silence.
/// @param window_ellipsoids Per-row overrides of the derived window, as
///                ellipsoids scaled so membership is Mahalanobis <= 1. An
///                unset entry means "derive this row's window as usual".
///                `tau_window` and `window_aspect_cap` do NOT apply to an
///                override -- they exist to derive a window from a best guess,
///                and an override already is the answer.
///
///                A window must be a REGION, not a set of indices, because
///                `eval_kernel` has to answer at points that are not mesh
///                columns. `init_dictionary.hpp`'s `ellipsoid_from_points`
///                converts a hand-picked index set into an admissible one.
inline OperatorFit fit_operator(
    const Eigen::Ref<const Eigen::MatrixXd>& x_cols,
    const Eigen::Ref<const Eigen::VectorXd>& m1_diag,
    const Eigen::Ref<const Eigen::VectorXd>& m2_diag,
    const Eigen::Ref<const Eigen::MatrixXd>& V,
    const Eigen::Ref<const Eigen::MatrixXd>& HV,
    const std::vector<Eigen::MatrixXd>& sigma,
    const OperatorFitConfig& config = OperatorFitConfig(),
    const std::optional<Eigen::MatrixXd>& mu0 = std::nullopt,
    const std::optional<Eigen::MatrixXd>& x_rows = std::nullopt,
    const std::vector<char>& gate = {},
    const std::vector<std::optional<ellipsoid_tree::Ellipsoid>>& window_ellipsoids = {} )
{
    const int dim = static_cast<int>(x_cols.cols());
    const Eigen::Index num_cols = x_cols.rows();
    const Eigen::Index num_rows = m1_diag.size();
    const Eigen::Index num_probes = V.cols();

    if ( m2_diag.size() != num_cols || V.rows() != num_cols )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: x_cols, m2_diag and V must agree on the column count");
    }
    if ( HV.rows() != num_rows || HV.cols() != num_probes )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: HV must be (num_rows, num_probes)");
    }
    if ( static_cast<Eigen::Index>(sigma.size()) != num_rows )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: sigma must have one covariance per row");
    }
    if ( !config.row.mode_policy )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: config.row.mode_policy is required");
    }
    if ( !(config.window_aspect_cap >= 1.0) )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: window_aspect_cap must be >= 1 (1 is an "
            "isotropic window, infinity the caller's ellipsoid untouched)");
    }
    if ( x_rows && x_rows->rows() != num_rows )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: x_rows must have one coordinate per row");
    }
    if ( config.spike && x_rows )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: the spike needs the square dof context, so it "
            "cannot be combined with separate row coordinates; set spike = false");
    }
    if ( !gate.empty() && static_cast<Eigen::Index>(gate.size()) != num_rows )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: gate must have one entry per row");
    }
    if ( !window_ellipsoids.empty()
         && static_cast<Eigen::Index>(window_ellipsoids.size()) != num_rows )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: window_ellipsoids must have one entry per row");
    }

    const Eigen::MatrixXd centers =
        mu0 ? *mu0 : ( x_rows ? *x_rows : Eigen::MatrixXd(x_cols) );
    if ( centers.rows() != num_rows || centers.cols() != dim )
    {
        throw std::invalid_argument("lgpsf::fit_operator: mu0 must be (num_rows, N)");
    }

    const int num_extra_config = config.spike ? 1 : 0;
    const MuMode ladder_mode =
        ( config.row.mu == MuPolicy::Free ) ? MuMode::Fitted : MuMode::Pinned;

    // Each counting rule counts the parameters ACTUALLY BEING FIT. The
    // baseline is a linear fit in the pinned encoding, so it counts N(N+1)/2;
    // the search counts its own stream encoding, which is N(N+3)/2 when the
    // center is fitted. (The Python prototype used the free count for both --
    // an unintended slip that went uncaught because it is conservative,
    // skipping more levels rather than fewer.)
    const int baseline_params = theta_hat_size(dim, MuMode::Pinned);
    const int search_params = theta_hat_size(dim, ladder_mode);

    // Randomness lives HERE and only here: one split and one jitter table for
    // the whole fit, built before any row is touched, so every row and the
    // baseline score on identical folds and the result cannot depend on
    // scheduling. See the plan's randomness section.
    ProbeFitConfig row_config = config.row;
    row_config.split =
        config.seed
            ? kfold_split(static_cast<int>(num_probes), config.row.cv_folds, *config.seed)
            : kfold_split(static_cast<int>(num_probes), config.row.cv_folds);
    row_config.jitter = jitter_table(theta_hat_size(dim, ladder_mode),
                                     kMaxModeProposals,
                                     config.seed ? *config.seed : 0u);

    // The a-priori baseline may never depend on an adaptive trajectory, so the
    // policy is asked for its feedback-blind sets.
    ModeSearchContext baseline_ctx;
    baseline_ctx.dim = dim;
    baseline_ctx.num_probes = static_cast<int>(num_probes);
    baseline_ctx.num_extra = num_extra_config;
    baseline_ctx.num_params = baseline_params;
    const std::vector<std::vector<Mode>> baseline_sets =
        config.row.mode_policy->baseline_sets(baseline_ctx);

    std::vector<ellipsoid_tree::Ball> points;
    points.reserve(static_cast<std::size_t>(num_cols));
    for ( Eigen::Index j = 0; j < num_cols; ++j )
    {
        points.push_back(ellipsoid_tree::Ball{x_cols.row(j).transpose(), 0.0});
    }
    const ellipsoid_tree::BallTree column_tree(std::move(points));

    std::vector<detail::RowOutcome> outcomes(static_cast<std::size_t>(num_rows));
    std::vector<Eigen::MatrixXd> prior(static_cast<std::size_t>(num_rows));
    std::vector<char> attempt(static_cast<std::size_t>(num_rows), 0);
    Eigen::MatrixXd window_center = Eigen::MatrixXd::Constant(
        num_rows, dim, std::numeric_limits<double>::quiet_NaN());
    Eigen::MatrixXd window_covariance = Eigen::MatrixXd::Constant(
        num_rows, dim * dim, std::numeric_limits<double>::quiet_NaN());

    // --- every window at once, by one dual-tree descent ---------------------
    //
    // Not a query per row: the column-point tree is descended against a tree of
    // every row's query ellipsoid, so the whole window field costs one
    // traversal -- the same way the deployed sparsity pattern is derived.
    {
        std::vector<ellipsoid_tree::Ellipsoid> queries;
        std::vector<int> query_row;
        for ( Eigen::Index rho = 0; rho < num_rows; ++rho )
        {
            detail::RowOutcome& outcome = outcomes[static_cast<std::size_t>(rho)];
            if ( !gate.empty() && !gate[static_cast<std::size_t>(rho)] )
            {
                continue;  // stays GatedOut
            }
            const Eigen::MatrixXd& covariance = sigma[static_cast<std::size_t>(rho)];
            const Eigen::LLT<Eigen::MatrixXd> chol(covariance);
            if ( chol.info() != Eigen::Success )
            {
                outcome.status = RowStatus::Failed;
                outcome.failure = "sigma is not positive definite";
                continue;
            }
            prior[static_cast<std::size_t>(rho)] = chol.matrixL();
            attempt[static_cast<std::size_t>(rho)] = 1;

            // The window is stored PRE-SCALED, so membership is Mahalanobis
            // <= 1 whether it was derived or supplied. That makes the two paths
            // mean the same thing and lets the tree be queried at tau = 1.
            ellipsoid_tree::Ellipsoid region;
            if ( !window_ellipsoids.empty()
                 && window_ellipsoids[static_cast<std::size_t>(rho)] )
            {
                region = *window_ellipsoids[static_cast<std::size_t>(rho)];
            }
            else
            {
                region.mu = centers.row(rho).transpose();
                region.Sigma = config.tau_window * config.tau_window
                               * capped_covariance(covariance,
                                                   config.window_aspect_cap);
            }
            window_center.row(rho) = region.mu.transpose();
            for ( int i = 0; i < dim; ++i )
            {
                for ( int j = 0; j < dim; ++j )
                {
                    window_covariance(rho, i * dim + j) = region.Sigma(i, j);
                }
            }
            queries.push_back(region);
            query_row.push_back(static_cast<int>(rho));
        }

        if ( !queries.empty() )
        {
            const ellipsoid_tree::EllipsoidTree window_tree(
                std::move(queries), 1.0, config.num_threads);
            const std::vector<std::pair<int, int>> pairs =
                ellipsoid_tree::collision_pairs(column_tree, window_tree);
            for ( const std::pair<int, int>& hit : pairs )
            {
                outcomes[static_cast<std::size_t>(query_row[static_cast<std::size_t>(hit.second)])]
                    .window.push_back(hit.first);
            }
            for ( int rho : query_row )
            {
                std::vector<int>& window = outcomes[static_cast<std::size_t>(rho)].window;
                std::sort(window.begin(), window.end());
            }
        }
    }

    ellipsoid_tree::detail::parallel_for(
        0, static_cast<std::ptrdiff_t>(num_rows),
        [&]( std::ptrdiff_t begin, std::ptrdiff_t end ) {
            for ( std::ptrdiff_t rho = begin; rho < end; ++rho )
            {
                detail::RowOutcome& outcome = outcomes[static_cast<std::size_t>(rho)];
                if ( !attempt[static_cast<std::size_t>(rho)] )
                {
                    continue;  // gated out, or its window could not be formed
                }
                try
                {
                    const Eigen::VectorXd center = centers.row(rho).transpose();
                    const Eigen::MatrixXd& covariance =
                        sigma[static_cast<std::size_t>(rho)];
                    const Eigen::MatrixXd& prior_L = prior[static_cast<std::size_t>(rho)];
                    const std::vector<int>& window = outcome.window;
                    if ( window.size() < 2u )
                    {
                        throw std::invalid_argument(
                            "window has " + std::to_string(window.size())
                            + " points (sigma too small, or tau_window too tight?)");
                    }

                    int spike_position = -1;
                    if ( config.spike )
                    {
                        const auto found = std::lower_bound(
                            window.begin(), window.end(), static_cast<int>(rho));
                        if ( found != window.end() && *found == static_cast<int>(rho) )
                        {
                            spike_position =
                                static_cast<int>(found - window.begin());
                        }
                        // else the row dof fell outside its own window, only
                        // possible with a far-off explicit mu0; the shipped
                        // model then simply has no spike.
                    }

                    const Eigen::Index window_size =
                        static_cast<Eigen::Index>(window.size());
                    Eigen::MatrixXd x_window(window_size, dim);
                    Eigen::VectorXd m2_window(window_size);
                    Eigen::MatrixXd z(window_size, num_probes);
                    for ( Eigen::Index i = 0; i < window_size; ++i )
                    {
                        const int column = window[static_cast<std::size_t>(i)];
                        x_window.row(i) = x_cols.row(column);
                        m2_window(i) = m2_diag(column);
                        z.row(i) = V.row(column);
                    }
                    const Eigen::VectorXd y = HV.row(rho).transpose();
                    const double target_mass = m1_diag(rho);
                    const int num_extra = ( spike_position >= 0 ) ? 1 : 0;

                    // --- baseline: a linear fit at sigma[rho], pinned -------
                    const Eigen::MatrixXd z_hat = whiten_probes(z, m2_window);
                    const Eigen::VectorXd y_hat = whiten_data(y, target_mass);
                    Eigen::MatrixXd extra = Eigen::MatrixXd::Zero(window_size, num_extra);
                    if ( num_extra > 0 )
                    {
                        extra(spike_position, 0) = 1.0;
                    }
                    const Eigen::MatrixXd e_hat =
                        whiten_extra(extra, target_mass, m2_window);
                    const Eigen::VectorXd theta_baseline =
                        theta_hat_from_cholesky(prior_L);

                    double baseline_score = std::numeric_limits<double>::infinity();
                    Eigen::VectorXd baseline_c, baseline_s;
                    const std::vector<Mode>* baseline_modes = nullptr;
                    for ( const std::vector<Mode>& modes : baseline_sets )
                    {
                        if ( static_cast<int>(num_probes)
                             < 2 * (static_cast<int>(modes.size()) + num_extra
                                    + baseline_params) )
                        {
                            continue;
                        }
                        const WhitenedBasis basis(x_window, target_mass, m2_window,
                                                  modes, center, MuMode::Pinned);
                        const double score =
                            linear_cv_score(z_hat, y_hat, basis, theta_baseline,
                                            e_hat, row_config.split);
                        if ( score < baseline_score )
                        {
                            baseline_score = score;
                            std::tie(baseline_c, baseline_s) = detail::linear_fit(
                                z_hat, y_hat, basis, theta_baseline, e_hat);
                            baseline_modes = &modes;
                        }
                    }
                    if ( baseline_modes == nullptr )
                    {
                        throw std::invalid_argument(
                            "no mode set passed the counting rule at k="
                            + std::to_string(num_probes));
                    }

                    // --- the searched fit -----------------------------------
                    bool searchable = false;
                    for ( const std::vector<Mode>& modes : baseline_sets )
                    {
                        searchable =
                            searchable
                            || static_cast<int>(num_probes)
                                   >= 2 * (static_cast<int>(modes.size()) + num_extra
                                           + search_params);
                    }
                    std::optional<ProbeFitResult> searched;
                    if ( searchable )
                    {
                        searched = fit_from_probes(
                            x_window, m2_window, z, y, center, spike_position,
                            row_config, covariance, target_mass);
                    }

                    // --- the guard ------------------------------------------
                    outcome.baseline_score = baseline_score;
                    if ( searched && searched->score < baseline_score )
                    {
                        outcome.status = RowStatus::Fit;
                        outcome.theta = searched->theta;
                        outcome.mu = searched->mu;
                        outcome.L = searched->L;
                        outcome.c = searched->c;
                        outcome.modes = searched->modes;
                        outcome.s = searched->s.size() ? searched->s(0) : 0.0;
                        outcome.score = searched->score;
                        outcome.released = searched->released;
                    }
                    else
                    {
                        outcome.status = RowStatus::FallbackBaseline;
                        outcome.theta =
                            to_theta(theta_baseline, center, MuMode::Pinned);
                        outcome.mu = center;
                        outcome.L = prior_L;
                        outcome.c = baseline_c;
                        outcome.modes = *baseline_modes;
                        outcome.s = baseline_s.size() ? baseline_s(0) : 0.0;
                        outcome.score = baseline_score;
                        outcome.released = false;
                    }
                    if ( !searched )
                    {
                        outcome.stop = RowStop::SearchInfeasible;
                    }
                    else
                    {
                        switch ( searched->stop_reason )
                        {
                            case StopReason::Target:
                                outcome.stop = RowStop::Target; break;
                            case StopReason::ModePatience:
                                outcome.stop = RowStop::ModePatience; break;
                            case StopReason::Exhausted:
                                outcome.stop = RowStop::Exhausted; break;
                        }
                    }
                }
                catch ( const std::exception& error )
                {
                    outcome.status = RowStatus::Failed;
                    outcome.failure = error.what();
                    outcome.window.clear();
                }
            }
        },
        config.num_threads);

    // --- serial gather, in row order ---------------------------------------
    //
    // Everything shared is built HERE rather than inside the parallel phase.
    // The mode-set registry is the reason: assigning ids as rows finish would
    // make the ids depend on thread scheduling, and results would stop being
    // bit-identical across thread counts.
    OperatorFit result;
    FittedOperator& fit = result.model;
    FitDiagnostics& diagnostics = result.diagnostics;
    fit.dim = dim;
    fit.x_cols = x_cols;
    fit.x_rows = x_rows;
    fit.m1_diag = m1_diag;
    fit.m2_diag = m2_diag;
    fit.spike = config.spike;
    diagnostics.config = config;
    fit.window_center = std::move(window_center);
    fit.window_covariance = std::move(window_covariance);

    const int public_params = theta_size(dim);
    fit.theta = Eigen::MatrixXd::Constant(num_rows, public_params,
                                          std::numeric_limits<double>::quiet_NaN());
    fit.mu = Eigen::MatrixXd::Constant(num_rows, dim,
                                       std::numeric_limits<double>::quiet_NaN());
    fit.L = Eigen::MatrixXd::Constant(num_rows, dim * dim,
                                      std::numeric_limits<double>::quiet_NaN());
    fit.s = Eigen::VectorXd::Zero(num_rows);
    diagnostics.score = Eigen::VectorXd::Constant(num_rows,
                                          std::numeric_limits<double>::quiet_NaN());
    diagnostics.baseline_score = Eigen::VectorXd::Constant(
        num_rows, std::numeric_limits<double>::quiet_NaN());
    fit.mode_set_id.assign(static_cast<std::size_t>(num_rows), -1);
    diagnostics.stop_reason.assign(static_cast<std::size_t>(num_rows), RowStop::None);
    diagnostics.released.assign(static_cast<std::size_t>(num_rows), 0);
    diagnostics.status.assign(static_cast<std::size_t>(num_rows), RowStatus::GatedOut);
    fit.window_indptr.assign(static_cast<std::size_t>(num_rows) + 1, 0);

    std::map<std::vector<Mode>, int> registry;
    std::size_t widest = 0;
    for ( Eigen::Index rho = 0; rho < num_rows; ++rho )
    {
        const detail::RowOutcome& outcome = outcomes[static_cast<std::size_t>(rho)];
        diagnostics.status[static_cast<std::size_t>(rho)] = outcome.status;
        diagnostics.stop_reason[static_cast<std::size_t>(rho)] = outcome.stop;
        diagnostics.released[static_cast<std::size_t>(rho)] = outcome.released ? 1 : 0;
        diagnostics.baseline_score(rho) = outcome.baseline_score;

        fit.window_indptr[static_cast<std::size_t>(rho) + 1] =
            fit.window_indptr[static_cast<std::size_t>(rho)]
            + static_cast<int>(outcome.window.size());
        fit.window_indices.insert(fit.window_indices.end(), outcome.window.begin(),
                                  outcome.window.end());

        if ( !outcome.failure.empty() )
        {
            diagnostics.failures[static_cast<int>(rho)] = outcome.failure;
        }
        if ( outcome.status != RowStatus::Fit
             && outcome.status != RowStatus::FallbackBaseline )
        {
            continue;
        }

        fit.theta.row(rho) = outcome.theta.transpose();
        fit.mu.row(rho) = outcome.mu.transpose();
        for ( int i = 0; i < dim; ++i )
        {
            for ( int j = 0; j < dim; ++j )
            {
                fit.L(rho, i * dim + j) = outcome.L(i, j);
            }
        }
        diagnostics.score(rho) = outcome.score;
        fit.s(rho) = outcome.s;

        auto found = registry.find(outcome.modes);
        if ( found == registry.end() )
        {
            found = registry.emplace(outcome.modes,
                                     static_cast<int>(fit.mode_sets.size())).first;
            fit.mode_sets.push_back(outcome.modes);
        }
        fit.mode_set_id[static_cast<std::size_t>(rho)] = found->second;
        widest = std::max(widest, static_cast<std::size_t>(outcome.c.size()));
    }

    fit.c = Eigen::MatrixXd::Zero(num_rows, static_cast<Eigen::Index>(widest));
    for ( Eigen::Index rho = 0; rho < num_rows; ++rho )
    {
        const Eigen::VectorXd& coefficients =
            outcomes[static_cast<std::size_t>(rho)].c;
        if ( coefficients.size() > 0 )
        {
            fit.c.row(rho).head(coefficients.size()) = coefficients.transpose();
        }
    }
    return result;
}

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
inline std::vector<int> model_rows( const FittedOperator& fit )
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

inline KernelAt kernel_at( const FittedOperator& fit, int rho,
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
inline Eigen::VectorXd window_radius2( const FittedOperator& fit, int rho,
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
inline bool in_window( const FittedOperator& fit, int rho, int column )
{
    const auto begin = fit.window_indices.begin() + fit.window_indptr[static_cast<std::size_t>(rho)];
    const auto end = fit.window_indices.begin() + fit.window_indptr[static_cast<std::size_t>(rho) + 1];
    return std::binary_search(begin, end, column);
}

/// The deployed smooth row on a given set of columns:
/// m1[rho] * m2[j] * Phi~(rho, x_j). No window restriction and no spike --
/// callers apply those.
inline Eigen::VectorXd deployed_smooth(
    const FittedOperator& fit, int rho, const std::vector<int>& columns,
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
    const FittedOperator& fit, const std::vector<int>& rows,
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
    const FittedOperator& fit, const std::vector<int>& rows,
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
    const FittedOperator& fit, const std::vector<int>& rows,
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
    const FittedOperator& fit, const Eigen::Ref<const Eigen::MatrixXd>& v,
    double truncation_tau = std::numeric_limits<double>::infinity() )
{
    if ( v.rows() != fit.num_cols() )
    {
        throw std::invalid_argument(
            "lgpsf::matvec: v has " + std::to_string(v.rows()) + " rows but the "
            "operator has " + std::to_string(fit.num_cols()) + " columns");
    }
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(fit.num_rows(), v.cols());
    for ( int rho : model_rows(fit) )
    {
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
    return out;
}

/// The fitted ellipsoids, as the operator layer's geometry summary.
struct EllipsoidField
{
    Eigen::MatrixXd mu;                   ///< (R_all, N); NaN where no model
    std::vector<Eigen::MatrixXd> sigma;   ///< R_all covariances L L^T
};

/// [smooth] The fitted (mu, Sigma) field -- what `ellipsoid_tree` consumes.
inline EllipsoidField ellipsoid_field( const FittedOperator& fit )
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
    const FittedOperator& fit, double tau, Symmetrize symmetrize = Symmetrize::None )
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

    std::vector<Eigen::Triplet<double>> triplets;
    for ( std::size_t e = 0; e < rows.size(); ++e )
    {
        const int rho = rows[e];
        std::vector<int>& columns = support[e];
        std::sort(columns.begin(), columns.end());

        const std::vector<int> window = fit.row_window(rho);
        std::vector<int> deployed;
        std::set_intersection(columns.begin(), columns.end(), window.begin(),
                              window.end(), std::back_inserter(deployed));

        if ( !deployed.empty() )
        {
            const Eigen::VectorXd values =
                detail::deployed_smooth(fit, rho, deployed, tau);
            for ( std::size_t i = 0; i < deployed.size(); ++i )
            {
                triplets.emplace_back(rho, deployed[i],
                                      values(static_cast<Eigen::Index>(i)));
            }
        }
        if ( fit.spike )
        {
            // Additive on top of the unmodified smooth part at the diagonal;
            // duplicate triplets sum, which is exactly that convention.
            triplets.emplace_back(rho, rho, fit.m1_diag(rho) * fit.s(rho));
        }
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
inline Eigen::VectorXd qc_map( const FittedOperator& fit,
                               const Eigen::Ref<const Eigen::MatrixXd>& V_qc,
                               const Eigen::Ref<const Eigen::MatrixXd>& HV_qc )
{
    if ( HV_qc.rows() != fit.num_rows() || HV_qc.cols() != V_qc.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::qc_map: HV_qc must be (num_rows, num_probes) to match V_qc");
    }
    const Eigen::MatrixXd predicted = matvec(fit, V_qc);
    Eigen::VectorXd out = Eigen::VectorXd::Constant(
        fit.num_rows(), std::numeric_limits<double>::quiet_NaN());
    for ( int rho : model_rows(fit) )
    {
        const double scale = std::max(HV_qc.row(rho).norm(), 1e-300);
        out(rho) = (predicted.row(rho) - HV_qc.row(rho)).norm() / scale;
    }
    return out;
}

/// [spike] The Dirac mass field `m_rho * s_rho` -- the mesh-independent
/// content of the S component. A map of it is a resolution diagnostic: where
/// it is large, the mesh is starving.
inline Eigen::VectorXd spike_measure( const FittedOperator& fit )
{
    return fit.m1_diag.cwiseProduct(fit.s);
}

} // end namespace lgpsf
