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
/// the deployed support is the fit window, which is what makes the per-row
/// scores honest for deployment. The fit's QUADRATURE on that window may be
/// coarsened (`coarsen_above`, ON by default at 3,000 points): cells graded so
/// that their error is controlled for every kernel width at once, singletons near the
/// centre, the deployed support untouched (coarsen_window.hpp) -- and the
/// reported scores are then re-evaluated on the full window, so the baseline
/// guard and the diagnostics still speak about the deployed object.
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
/// **Which shape to use.** Measured at field scale, neither dominates. With a
/// small mode set the two are indistinguishable in accuracy (0.0442 against
/// 0.0443) and the ellipsoid is ~20% faster on 18% fewer window points; with a
/// large one the ball is 2% better (0.0147 against 0.0150) and the time
/// advantage is gone, because mode count rather than window size then
/// dominates the cost. So the cheap-window argument for the ellipsoid weakens
/// exactly where the modes get expensive.
///
/// The default is the caller's ellipsoid untouched. Set `window_aspect_cap =
/// 1` for a ball, or anything between to cap the ratio. One caveat if you
/// choose the ball: turn `window_shape_rungs` off with it, since that
/// initialization family can recover nothing from a sphere.
///
/// Intermediate caps remain untried, and this was one operator.

#include <algorithm>
#include <chrono>
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

#include "lgpsf/coarsen_window.hpp"
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

    /// Coarsen the FIT's quadrature on any window with more points than this;
    /// 3,000 by default; 0 never coarsens. A per-row work bound that needs no
    /// estimate of the kernel's width: for the fit only, the window is
    /// replaced by graded cells of size at most `coarsen_eps` times their
    /// distance from the row's centre (in the window ellipsoid's own
    /// coordinates), each reduced to its mass-weighted centroid, summed mass
    /// and mean probe fields, with the spike and the farthest point kept as
    /// singletons. The cell count is about `3 pi / eps^2` per dyadic annulus
    /// in 2D -- logarithmic in the window's size -- so a 90,000-point window
    /// at eps 0.1 costs what a 7,000-point one does. The deployed support
    /// (`window_indptr` / `window_indices`, `window_center` /
    /// `window_covariance`) is the full window regardless, and `score` /
    /// `baseline_score` are re-evaluated on it. A row whose protected set
    /// (today: the spike) would exceed this trigger is left uncoarsened
    /// rather than refused; `FitDiagnostics::fit_points` says what each
    /// row's fit actually ran on. See coarsen_window.hpp.
    ///
    /// DEFAULT CHANGED 2026-09-08, 0 -> 3000: fits are no longer bitwise
    /// identical to 0.2.x on rows whose window exceeds the trigger.  Field
    /// validation on a continental ice-sheet Hessian took the worst rank's
    /// fit from 2,015 s to 588 s at the same prior with the operator, the
    /// QC ladder and the deployed solve unchanged; below ~2,000 points the
    /// singleton core near the centre leaves nothing to merge, so a smaller
    /// trigger only adds bookkeeping.  Set 0 to restore the old behaviour.
    int coarsen_above = 3000;

    /// Grading ratio of the coarsened quadrature: cell size over distance
    /// from the centre. A mode of radial degree p and angular order ell has
    /// its finest structure at scale `sigma / (p + ell)` near its own radius,
    /// so the uniform resolution condition is `eps * (p + ell)_max <~ 0.5`
    /// -- at the ladder's top level of 5, `eps <= 0.1` -- and the centroid
    /// rule makes the aggregation error second order, `(eps * ell)^2`. A
    /// coarser quadrature aliases the high modes into noise, which the
    /// ladder's cross-validation treats as the conservative failure (it stops
    /// lower). Also arms the released-centre resolution rule
    /// (`ProbeFitConfig::resolution_eps`) at this value on coarsened rows.
    /// Finite and positive; read only when `coarsen_above > 0`.
    double coarsen_eps = 0.1;

    /// Model the diagonal spike. The spike is tied to the row's own column
    /// dof: in the square context (no separate row coordinates) that is the
    /// row index itself; with separate row coordinates the caller must name
    /// it explicitly via `fit_operator`'s `row_own_col` (subset/rectangular
    /// contexts, e.g. a distributed rank fitting its rows against own + halo
    /// columns), or set spike = false.
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
    ///
    /// That is a WITHIN-BUILD guarantee. Across builds compiled with different
    /// flags a few rows can land on a different local minimum, because one ULP
    /// is enough to choose a basin where the objective is flat. See
    /// `docs/reproducibility.md` -- measured at 0.08% of rows, with no effect
    /// on the field.
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

    /// (R_all,) how many quadrature points each row's fit ran on: the window
    /// size, or the coarse cell count when `coarsen_above` triggered; 0 for
    /// gated and failed rows. Whether the work bound is binding is readable
    /// from this and nowhere else.
    Eigen::VectorXi fit_points;

    /// (R_all,) basis evaluations the row's search spent, summed over its
    /// candidates (`ProbeFitResult::evaluations_total`); 0 where the search
    /// did not run (gated, failed, or no mode set was searchable).
    Eigen::VectorXi evaluations;
    /// (R_all,) candidates the row's search tried; 0 where it did not run.
    Eigen::VectorXi candidates;
    /// (R_all,) a dimensionless proxy for the row's fit cost:
    ///
    ///     work = fit_points * evaluations * modes,
    ///
    /// with `modes` the size of the largest mode set the search tried. Every
    /// basis evaluation is O(fit_points * modes), and the evaluations are
    /// where a row's time goes; the per-row floor (window gather, whitening,
    /// the baseline's CV score) is not in it. Deterministic, like its
    /// factors. 0 where the search did not run.
    Eigen::VectorXd work;
    /// (R_all,) TELEMETRY, NOT DETERMINISTIC: the phase split of
    /// `row_seconds`, `std::chrono::steady_clock`, for the load-balance
    /// study.  `coarsen_seconds` is the graded coarsening alone,
    /// `search_seconds` the mode-set ladder and its Levenberg-Marquardt
    /// searches (everything that reads only the coarse quadrature), and
    /// `rescore_seconds` the full-window re-score of the finalists with its
    /// guard.  What `row_seconds` holds beyond the three is the window
    /// gather, the whitening and the bookkeeping.  Only `search_seconds`
    /// could move to another rank under a fitting-only redistribution
    /// (`dev/row-balance-plan.md`), so the resident share is
    /// `1 - search_seconds / row_seconds`.  Like `row_seconds` these vary run
    /// to run and across thread counts, are read by no decision and no
    /// output, and are excluded from the bit-identity tests.  0 where the
    /// phase did not run.
    Eigen::VectorXd coarsen_seconds;
    Eigen::VectorXd search_seconds;
    Eigen::VectorXd rescore_seconds;
    /// (R_all,) TELEMETRY, NOT DETERMINISTIC: wall-clock seconds of the row's
    /// fit block inside the parallel loop -- from the window gather to the
    /// guard, `std::chrono::steady_clock` -- for the load-balance study.
    /// It varies run to run and across thread counts, it is read by no
    /// decision and no output, and the bit-identity tests exclude it. 0 for
    /// gated rows; a failed row records the time it spent before failing.
    Eigen::VectorXd row_seconds;

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
    int fit_points = 0;
    int evaluations = 0;
    int candidates = 0;
    int max_modes = 0;      ///< largest mode set the search tried
    double row_seconds = 0.0;
    double coarsen_seconds = 0.0;
    double search_seconds = 0.0;
    double rescore_seconds = 0.0;
    std::string failure;
};

/// PHASE A's product and PHASE B's input: everything the mode-set ladder and
/// the LM search read, and nothing else. A row's fit is a pure function of
/// this, which is what makes a fitting-only redistribution bitwise safe
/// (`dev/row-balance-plan.md`, sections 3 and 6): no rank-local state is read
/// below phase A.
///
/// The quadrature is held as NON-OWNING pointers. On an uncoarsened row
/// `x_fit` IS the full window, and no row may copy its window; the pointees
/// are the caller's `x_window` / `m2_window` / `z`, or the `CoarseWindow` it
/// built from them, and both outlive the search. Everything else is small
/// enough to own (dim x dim, or one value per probe).
///
/// `sigma` is the row's RAW a-priori covariance and is NOT redundant with
/// `prior_L`. It seeds the LM stream -- `InitialGuess::sigma` reaches
/// `theta_hat_from_sigma`, which is `theta_hat_from_cholesky(chol(sigma))` --
/// and rebuilding `L L^T` and factoring it again does not return `L` bit for
/// bit. A package that shipped only `prior_L` would fit a different row.
struct RowFitProblem
{
    const Eigen::MatrixXd* x_fit = nullptr;   ///< (fit_size, dim)
    const Eigen::VectorXd* m2_fit = nullptr;  ///< (fit_size,)
    const Eigen::MatrixXd* z_fit = nullptr;   ///< (fit_size, num_probes)
    int spike_fit = -1;       ///< The spike's cell in the quadrature, or -1
    int num_extra = 0;        ///< 1 with a spike, 0 without
    Eigen::VectorXd y;        ///< (num_probes,) the row's responses
    double target_mass = 0.0;
    Eigen::MatrixXd prior_L;  ///< (dim, dim) Cholesky factor of sigma
    Eigen::MatrixXd sigma;    ///< (dim, dim) the RAW covariance -- see above
    Eigen::VectorXd center;   ///< (dim,)
    bool coarsened = false;   ///< Whether the quadrature is the coarsening

    /// The config the search runs under: the shared `row_config`, or the
    /// coarsened row's own copy of it with the released-centre resolution
    /// rule armed. Non-owning like the quadrature; `coarsened` is the flag a
    /// package would carry, the host rebuilding the copy from it.
    const ProbeFitConfig* fit_config = nullptr;
};

/// PHASE B's product and PHASE C's input: the two finalists, plus the counters
/// only the search knows. The counters must cross back with the finalists --
/// they become `FitDiagnostics::evaluations` / `candidates` / `work` and,
/// circularly, the next rung's scheduling weight -- and nothing downstream can
/// recompute them.
struct RowFitCandidates
{
    /// Index into the globally identical `baseline_sets`, so this identifies
    /// the same mode set on any rank. -1 never leaves phase B: it throws.
    int baseline_index = -1;
    Eigen::VectorXd theta_baseline;
    Eigen::VectorXd baseline_c;
    Eigen::VectorXd baseline_s;
    double baseline_score = std::numeric_limits<double>::infinity();

    /// The searched fit; unset when no mode set was searchable.
    std::optional<ProbeFitResult> searched;

    int evaluations = 0;
    int candidates = 0;
    int max_modes = 0;  ///< largest mode set the search tried

    /// (num_probes,) the whitened responses. Recomputable from the problem's
    /// `y` and `target_mass`, so a migration would refill it on arrival rather
    /// than send it; it rides here so the fused path whitens exactly once,
    /// inside the search timer, as before the phase split.
    Eigen::VectorXd y_hat;
};

/// PHASE B: the pinned baseline over the mode-set ladder, then the searched
/// fit. Reads only the fit's quadrature -- never the full window -- and
/// touches no clock, no `RowOutcome` and no caller state beyond the globals
/// every rank holds identically. This is the phase a fitting-only
/// redistribution moves (`dev/row-balance-plan.md`).
///
/// @throws std::invalid_argument if no mode set passes the counting rule.
inline RowFitCandidates fit_row_candidates(
    const RowFitProblem& problem,
    const std::vector<std::vector<Mode>>& baseline_sets,
    const ProbeFitConfig& row_config,
    Eigen::Index num_probes,
    int baseline_params,
    int search_params )
{
    const Eigen::MatrixXd& x_fit = *problem.x_fit;
    const Eigen::VectorXd& m2_fit = *problem.m2_fit;
    const Eigen::MatrixXd& z_fit = *problem.z_fit;
    const Eigen::VectorXd& center = problem.center;
    const double target_mass = problem.target_mass;
    const int num_extra = problem.num_extra;
    const int spike_fit = problem.spike_fit;
    const Eigen::Index fit_size = x_fit.rows();

    RowFitCandidates out;

    // --- baseline: a linear fit at sigma[rho], pinned -----------------------
    const Eigen::MatrixXd z_hat = whiten_probes(z_fit, m2_fit);
    Eigen::VectorXd y_hat = whiten_data(problem.y, target_mass);
    Eigen::MatrixXd extra = Eigen::MatrixXd::Zero(fit_size, num_extra);
    if ( num_extra > 0 )
    {
        extra(spike_fit, 0) = 1.0;
    }
    const Eigen::MatrixXd e_hat =
        whiten_extra(extra, target_mass, m2_fit);
    Eigen::VectorXd theta_baseline =
        theta_hat_from_cholesky(problem.prior_L);

    double baseline_score = std::numeric_limits<double>::infinity();
    Eigen::VectorXd baseline_c, baseline_s;
    int baseline_index = -1;
    for ( std::size_t set = 0; set < baseline_sets.size(); ++set )
    {
        const std::vector<Mode>& modes = baseline_sets[set];
        if ( static_cast<int>(num_probes)
             < 2 * (static_cast<int>(modes.size()) + num_extra
                    + baseline_params) )
        {
            continue;
        }
        const WhitenedBasis basis(x_fit, target_mass, m2_fit,
                                  modes, center, MuMode::Pinned);
        const double score =
            linear_cv_score(z_hat, y_hat, basis, theta_baseline,
                            e_hat, row_config.split);
        if ( score < baseline_score )
        {
            baseline_score = score;
            std::tie(baseline_c, baseline_s) = detail::linear_fit(
                z_hat, y_hat, basis, theta_baseline, e_hat);
            baseline_index = static_cast<int>(set);
        }
    }
    if ( baseline_index < 0 )
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
        // The caller's a-priori shape is passed as the FIRST
        // guess, so it is tried before the default rungs --
        // the dictionary the row fit assembles is unchanged
        // from when `sigma0` was its own parameter.
        InitialGuess prior;
        prior.sigma = problem.sigma;
        prior.label = "sigma0";
        searched = fit_from_probes(
            x_fit, m2_fit, z_fit, problem.y, center, spike_fit,
            *problem.fit_config, {prior}, target_mass);
    }
    if ( searched )
    {
        out.evaluations = searched->evaluations_total;
        out.candidates = searched->candidates_tried;
        for ( const CandidateFit& candidate : searched->candidates )
        {
            out.max_modes = std::max(
                out.max_modes,
                static_cast<int>(candidate.num_modes()));
        }
    }

    // The finalists, and the counters that must travel with them.
    out.baseline_index = baseline_index;
    out.baseline_score = baseline_score;
    out.theta_baseline = std::move(theta_baseline);
    out.baseline_c = std::move(baseline_c);
    out.baseline_s = std::move(baseline_s);
    out.y_hat = std::move(y_hat);
    out.searched = std::move(searched);
    return out;
}

/// PHASE C: the full-window re-score of the finalists, the baseline guard, and
/// the selection into `outcome`. Runs where the window is -- the deployed
/// support is the full window, never the coarse one, so this cannot move off
/// the row's owner -- and reads the full-window arrays phase A already built
/// (@p x_window, @p m2_window, @p z), never re-gathering them here.
///
/// Of @p problem it reads only the owned members (center, prior_L,
/// target_mass, num_extra, coarsened); the quadrature pointers are phase B's
/// business and may be dangling by now.
inline void select_row_fit(
    const RowFitProblem& problem,
    const RowFitCandidates& finalists,
    const Eigen::MatrixXd& x_window,
    const Eigen::VectorXd& m2_window,
    const Eigen::MatrixXd& z,
    int spike_position,
    const std::vector<std::vector<Mode>>& baseline_sets,
    const ProbeFitConfig& row_config,
    RowOutcome& outcome )
{
    const Eigen::Index window_size = x_window.rows();
    const Eigen::VectorXd& center = problem.center;
    const double target_mass = problem.target_mass;
    const int num_extra = problem.num_extra;
    const std::vector<Mode>& baseline_modes =
        baseline_sets[static_cast<std::size_t>(finalists.baseline_index)];
    const std::optional<ProbeFitResult>& searched = finalists.searched;
    const Eigen::VectorXd& y_hat = finalists.y_hat;
    const Eigen::VectorXd& theta_baseline = finalists.theta_baseline;
    double baseline_score = finalists.baseline_score;

    // --- full-window re-score of the finalists --------------
    //
    // On a coarsened row the scores so far are coarse-
    // quadrature scores: the search's internal currency. The
    // guard is taken, and the diagnostics report, the same two
    // models scored on the FULL window -- one linear_cv_score
    // each, O(K m), negligible against a fit -- so `score` and
    // `baseline_score` keep their literal meaning, and an
    // aliasing artifact (good on the cells, bad on the points)
    // cannot ship. The coefficients are not refit.
    double searched_score =
        searched ? searched->score
                 : std::numeric_limits<double>::infinity();
    if ( problem.coarsened )
    {
        const Eigen::MatrixXd z_hat_full = whiten_probes(z, m2_window);
        Eigen::MatrixXd extra_full =
            Eigen::MatrixXd::Zero(window_size, num_extra);
        if ( num_extra > 0 )
        {
            extra_full(spike_position, 0) = 1.0;
        }
        const Eigen::MatrixXd e_hat_full =
            whiten_extra(extra_full, target_mass, m2_window);

        const WhitenedBasis full_baseline(
            x_window, target_mass, m2_window, baseline_modes,
            center, MuMode::Pinned);
        baseline_score = linear_cv_score(
            z_hat_full, y_hat, full_baseline, theta_baseline,
            e_hat_full, row_config.split);

        if ( searched )
        {
            // The winner is evaluated in the encoding
            // fit_from_probes fitted it in: about `center`,
            // the only centre this row's dictionary uses (the
            // prior guess carries none; rungs, warm starts and
            // the release stage all sit at default_mu), with
            // the centre fitted when the candidate's was --
            // released, or under MuPolicy::Free, where
            // `released` stays false by convention. A pinned
            // re-encoding would drop a released displacement
            // and score the wrong model.
            const MuMode winner_mode =
                ( searched->released
                  || row_config.mu == MuPolicy::Free )
                    ? MuMode::Fitted
                    : MuMode::Pinned;
            const WhitenedBasis full_winner(
                x_window, target_mass, m2_window,
                searched->model.modes, center, winner_mode);
            searched_score = linear_cv_score(
                z_hat_full, y_hat, full_winner,
                to_theta_hat(searched->model.theta, center,
                             winner_mode),
                e_hat_full, row_config.split);
        }
    }

    // --- the guard ------------------------------------------
    outcome.baseline_score = baseline_score;
    if ( searched && searched_score < baseline_score )
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
        outcome.score = searched_score;
        outcome.released = searched->released;
    }
    else
    {
        outcome.status = RowStatus::FallbackBaseline;
        outcome.theta =
            to_theta(theta_baseline, center, MuMode::Pinned);
        outcome.mu = center;
        outcome.L = problem.prior_L;
        outcome.c = finalists.baseline_c;
        outcome.modes = baseline_modes;
        outcome.s = finalists.baseline_s.size() ? finalists.baseline_s(0) : 0.0;
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

} // end namespace detail

/// Fit the parametric approximation from raw probes and responses.
///
/// @param x_cols  (K_all, N) column-dof coordinates.
/// @param m1_diag (R_all,) row masses. @param m2_diag (K_all,) column masses.
/// @param V       (K_all, k) raw random probes.
/// @param HV      (R_all, k) raw responses, `HV(rho, l) = (H V.col(l))(rho)`.
/// @param sigma   R_all covariances: the caller's BEST GUESS at each bump's
///                shape, NOT required to be conservative. Three consumers --
///                the a-priori initialization guess, the baseline fit, and the
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
/// @return        `{model, diagnostics}` -- the operator, and per-row
///                provenance that nothing in evaluation reads.
/// @throws std::invalid_argument if the shapes disagree, if
///         `config.row.mode_policy` is unset (required -- no growth order is
///         defensible as a silent default), or if `config.tau_window` or
///         `config.window_aspect_cap` is not positive.
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
    const std::vector<std::optional<ellipsoid_tree::Ellipsoid>>& window_ellipsoids = {},
    const std::vector<int>& row_own_col = {} )
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
    if ( config.coarsen_above < 0 )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: coarsen_above must be >= 0 (0 never coarsens), got "
            + std::to_string(config.coarsen_above));
    }
    if ( !(config.coarsen_eps > 0.0) || !std::isfinite(config.coarsen_eps) )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: coarsen_eps must be finite and positive, got "
            + std::to_string(config.coarsen_eps));
    }
    if ( x_rows && x_rows->rows() != num_rows )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: x_rows must have one coordinate per row");
    }
    if ( config.spike && x_rows && row_own_col.empty() )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: the spike needs to know each row's own column "
            "dof; with separate row coordinates pass row_own_col (the column "
            "index of each row's own dof, -1 for rows without one) or set "
            "spike = false");
    }
    if ( !row_own_col.empty()
         && static_cast<Eigen::Index>(row_own_col.size()) != num_rows )
    {
        throw std::invalid_argument(
            "lgpsf::fit_operator: row_own_col must have one entry per row");
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

    // Each counting rule counts the parameters ACTUALLY BEING FIT, and the two
    // differ. The baseline is a linear fit in the pinned encoding, so it counts
    // N(N+1)/2; the search counts its own stream encoding, which is N(N+3)/2
    // when the center is fitted. Using the larger count for both is safe but
    // wrong: it skips levels the baseline could afford, which makes the
    // a-priori model look worse than it is and the guard fire less often.
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
    // The window ellipsoid as a frame, for the graded coarsening, which
    // grades in the window's own coordinates. Built only when coarsening can
    // trigger, so the default path does no extra work and gains no failure
    // mode; kept per row rather than un-flattened from window_covariance.
    std::vector<EllipsoidFrame> window_frame(
        config.coarsen_above > 0 ? static_cast<std::size_t>(num_rows) : 0u);
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
            if ( config.coarsen_above > 0 )
            {
                // Sigma_w = L_w L_w^T; a derived window is SPD by construction
                // (sigma passed its own Cholesky above), a supplied one is
                // the caller's word, checked here.
                const Eigen::LLT<Eigen::MatrixXd> window_chol(region.Sigma);
                if ( window_chol.info() != Eigen::Success )
                {
                    outcome.status = RowStatus::Failed;
                    outcome.failure = "window covariance is not positive definite";
                    attempt[static_cast<std::size_t>(rho)] = 0;
                    continue;
                }
                window_frame[static_cast<std::size_t>(rho)] =
                    make_frame(region.mu, Eigen::MatrixXd(window_chol.matrixL()));
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
                // Telemetry only (FitDiagnostics::row_seconds): nothing below
                // reads the clock.
                const auto row_start = std::chrono::steady_clock::now();
                try
                {
                    // --- PHASE A: window, quadrature, and the fit package ---
                    //
                    // Resolves this row's geometry and gathers its data.  Runs
                    // where the row's columns are; a redistribution moves what
                    // comes out of it, not this.
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
                        // The row's own column dof: identity in the square
                        // context, explicit via row_own_col in subset/
                        // rectangular contexts (e.g., a distributed rank
                        // fitting its rows against own + halo columns).
                        const int own =
                            row_own_col.empty()
                                ? static_cast<int>(rho)
                                : row_own_col[static_cast<std::size_t>(rho)];
                        if ( own >= 0 )
                        {
                            const auto found = std::lower_bound(
                                window.begin(), window.end(), own);
                            if ( found != window.end() && *found == own )
                            {
                                spike_position =
                                    static_cast<int>(found - window.begin());
                            }
                        }
                        // else the row dof fell outside its own window (far-
                        // off explicit mu0) or has no own column (own < 0);
                        // the shipped model then simply has no spike.
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

                    // --- the fit's quadrature: the window, or its graded
                    //     coarsening ----------------------------------------
                    //
                    // From here to the guard everything reads x_fit / m2_fit /
                    // z_fit / spike_fit. The window itself -- outcome.window,
                    // window_center / window_covariance -- is the deployed
                    // support and is never touched. A row whose protected set
                    // (today: just the spike) would exceed the trigger is left
                    // uncoarsened rather than refused, so with one protected
                    // position the third condition is vacuous; it is the
                    // general rule written down.
                    const bool coarsened =
                        config.coarsen_above > 0
                        && window_size > config.coarsen_above
                        && num_extra <= config.coarsen_above;
                    CoarseWindow coarse;
                    if ( coarsened )
                    {
                        std::vector<int> protected_positions;
                        if ( spike_position >= 0 )
                        {
                            protected_positions.push_back(spike_position);
                        }
                        const auto coarsen_start =
                            std::chrono::steady_clock::now();
                        coarse = coarsen_window(
                            x_window, m2_window, z, protected_positions, center,
                            window_frame[static_cast<std::size_t>(rho)],
                            config.coarsen_eps);
                        outcome.coarsen_seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now()
                                - coarsen_start).count();
                        if ( coarse.x.rows() < 2 )
                        {
                            // Only coincident points can do this (the spike
                            // and the farthest point are distinct singletons
                            // otherwise); fail attributably rather than let
                            // local_spacing throw its own message.
                            throw std::invalid_argument(
                                "lgpsf::fit_operator: coarsened window has "
                                + std::to_string(coarse.x.rows()) + " points (from "
                                + std::to_string(window_size) + " at coarsen_eps="
                                + std::to_string(config.coarsen_eps)
                                + "; are the window's points coincident?)");
                        }
                    }
                    const Eigen::MatrixXd& x_fit = coarsened ? coarse.x : x_window;
                    const Eigen::VectorXd& m2_fit = coarsened ? coarse.m2 : m2_window;
                    const Eigen::MatrixXd& z_fit = coarsened ? coarse.z : z;
                    const int spike_fit =
                        ( coarsened && spike_position >= 0 )
                            ? coarse.protected_cells.front()
                            : spike_position;
                    const Eigen::Index fit_size = x_fit.rows();
                    outcome.fit_points = static_cast<int>(fit_size);

                    // The released-centre resolution rule (probe_fit.hpp) is
                    // armed on coarsened rows only: it rejects a needle sitting
                    // on cells the grading made too coarse for it, and there
                    // are no such cells otherwise. row_config is shared across
                    // rows, so a coarsened row takes its own copy.
                    std::optional<ProbeFitConfig> coarse_config;
                    if ( coarsened )
                    {
                        coarse_config = row_config;
                        coarse_config->resolution_eps = config.coarsen_eps;
                    }
                    const ProbeFitConfig& fit_config =
                        coarse_config ? *coarse_config : row_config;

                    // PHASE A ends here. Everything the search needs is now in
                    // one self-contained package -- the quadrature by pointer,
                    // so an uncoarsened row still fits on its own window with no
                    // copy -- and nothing below it reads rank-local state.
                    detail::RowFitProblem problem;
                    problem.x_fit = &x_fit;
                    problem.m2_fit = &m2_fit;
                    problem.z_fit = &z_fit;
                    problem.spike_fit = spike_fit;
                    problem.num_extra = num_extra;
                    problem.y = y;
                    problem.target_mass = target_mass;
                    problem.prior_L = prior_L;
                    problem.sigma = covariance;
                    problem.center = center;
                    problem.coarsened = coarsened;
                    problem.fit_config = &fit_config;

                    // --- PHASE B: the baseline ladder and the searched fit --
                    //
                    // From here to the counter harvest nothing reads the full
                    // window, only the fit's quadrature.  It is the part a
                    // fitting-only redistribution could move, and it is a pure
                    // function of `problem` (dev/row-balance-plan.md).
                    const auto search_start = std::chrono::steady_clock::now();
                    const detail::RowFitCandidates finalists =
                        detail::fit_row_candidates(problem, baseline_sets,
                                                   row_config, num_probes,
                                                   baseline_params, search_params);
                    outcome.evaluations = finalists.evaluations;
                    outcome.candidates = finalists.candidates;
                    outcome.max_modes = finalists.max_modes;
                    outcome.search_seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now()
                            - search_start).count();

                    // --- PHASE C: full-window re-score, guard, selection ----
                    //
                    // Back on the full window phase A gathered: the deployed
                    // support is the window, never the coarsening, so this
                    // phase stays with the row's owner.
                    const auto rescore_start = std::chrono::steady_clock::now();
                    detail::select_row_fit(problem, finalists, x_window, m2_window,
                                           z, spike_position, baseline_sets,
                                           row_config, outcome);
                    outcome.rescore_seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now()
                            - rescore_start).count();
                }
                catch ( const std::exception& error )
                {
                    outcome.status = RowStatus::Failed;
                    outcome.failure = error.what();
                    outcome.window.clear();
                    outcome.fit_points = 0;
                    outcome.evaluations = 0;
                    outcome.candidates = 0;
                    outcome.max_modes = 0;
                    outcome.coarsen_seconds = 0.0;
                    outcome.search_seconds = 0.0;
                    outcome.rescore_seconds = 0.0;
                }
                outcome.row_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - row_start).count();
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
    fit.spike_col = row_own_col;
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
    diagnostics.fit_points = Eigen::VectorXi::Zero(num_rows);
    diagnostics.evaluations = Eigen::VectorXi::Zero(num_rows);
    diagnostics.candidates = Eigen::VectorXi::Zero(num_rows);
    diagnostics.work = Eigen::VectorXd::Zero(num_rows);
    diagnostics.row_seconds = Eigen::VectorXd::Zero(num_rows);
    diagnostics.coarsen_seconds = Eigen::VectorXd::Zero(num_rows);
    diagnostics.search_seconds = Eigen::VectorXd::Zero(num_rows);
    diagnostics.rescore_seconds = Eigen::VectorXd::Zero(num_rows);
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
        diagnostics.fit_points(rho) = outcome.fit_points;
        diagnostics.evaluations(rho) = outcome.evaluations;
        diagnostics.candidates(rho) = outcome.candidates;
        diagnostics.work(rho) = static_cast<double>(outcome.fit_points)
                                * static_cast<double>(outcome.evaluations)
                                * static_cast<double>(outcome.max_modes);
        diagnostics.row_seconds(rho) = outcome.row_seconds;
        diagnostics.coarsen_seconds(rho) = outcome.coarsen_seconds;
        diagnostics.search_seconds(rho) = outcome.search_seconds;
        diagnostics.rescore_seconds(rho) = outcome.rescore_seconds;

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
