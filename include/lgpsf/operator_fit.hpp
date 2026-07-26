#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Whole-operator fitting: the per-target probe fit run over the rows of
/// an implicitly available operator, producing an `LGOperator`.
///
/// This header is a PRODUCER of `lg_operator.hpp`'s data structure, not its
/// definition. Everything about representing, evaluating and assembling an
/// operator lives there and depends on none of the fitting machinery below.
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
/// ## Do not gate dead rows -- fitting them is already free and correct
///
/// `gate` defaults to unset (attempt every row) and that is the RECOMMENDED
/// setting, including when the operator is known to contain rows whose
/// response is identically zero. A dead row costs ONE candidate: its data is
/// zero, so the inner solve returns zero coefficients at a CV score of exactly
/// 0, the baseline guard ties it, and the row ships the baseline -- a valid
/// model that predicts exactly zero. It cannot fail, it cannot produce NaN,
/// and it cannot poison a neighbour, because a row is fitted only from its own
/// window and its own data.
///
/// Measured on the full 6557-row PIG operator (1481 dead rows): ungated
/// fitting took 159.9 s against 162.4 s gated -- the gate saved nothing -- and
/// every live row's prediction was BIT-IDENTICAL with and without it. That
/// last part is structural rather than lucky: each row's window comes from its
/// own ellipsoid, and the CV folds and the warm-start jitter table are global,
/// so no row can observe which other rows were attempted.
///
/// So the gate is for rows the CALLER does not want modeled -- a subdomain, a
/// boundary layer, a two-pass workflow -- and not for rows the caller expects
/// the fitter to struggle with. Gated rows get `RowStatus::GatedOut`, never
/// silence, so a gate is always visible in the diagnostics.
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
#include "lgpsf/lg_operator.hpp"
#include "lgpsf/lg_ellipsoid_feature.hpp"
#include "lgpsf/mode_policy.hpp"
#include "lgpsf/probe_fit.hpp"
#include "lgpsf/whitening.hpp"

namespace lgpsf {

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

/// Per-row provenance for the fit that produced a LGOperator: how each row
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
    LGOperator model;
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
/// @param gate    (R_all,) which rows to attempt; unset means all, which is
///                the recommended setting -- see "Do not gate dead rows"
///                above. Gated rows get a status, not silence.
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
                        const EllipsoidFrame shipped = searched->model.frame();
                        outcome.theta = searched->model.theta;
                        outcome.mu = shipped.mu;
                        outcome.L = shipped.L;
                        outcome.c = searched->model.c;
                        outcome.modes = searched->model.modes;
                        outcome.s =
                            searched->model.s.size() ? searched->model.s(0) : 0.0;
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
    LGOperator& fit = result.model;
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

} // end namespace lgpsf
