#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Fitting a target known only through random probe measurements: the
/// general-purpose top layer over one target, assembling everything below it
/// behind one raw-data interface.
///
/// **The problem.** A function on a batch of points is known only through
/// inner products with random probe fields, y_l = <target, z_l>. One row of an
/// implicitly available operator H = M1 Phi M2 is the motivating case, but
/// nothing here requires that. The caller supplies the batch, the lumped
/// masses on it, the raw probes and responses, a reference center mu0, and a
/// mode-growth policy. This layer whitens internally -- masses are ROUTED here
/// but all mass math stays in whitening.hpp -- runs an ORDERED STREAM OF
/// CANDIDATE FITS under one selection rule with early-stopping certificates,
/// and returns the winning model plus the full scored candidate table.
///
/// The three-way classification behind the design:
///  - **Structural facts** the caller declares and this layer never
///    adjudicates: the window, probes, masses, mu0, the spike index, the mode
///    dictionary.
///  - **Competing hypotheses** adjudicated by data, all through one mechanism
///    (a candidate list and one selection rule): initial shape and scale,
///    mode-set size, pinned versus released center.
///  - **Numerical hygiene** with fixed defaults, in VarProOptions.
///
/// **Selection = admissibility, then score, then simplicity.**
///  - ADMISSIBILITY: the caller's window is conservative, so the true kernel
///    fits inside it by construction. Fits whose major semi-axis exceeds the
///    window radius, or whose released center leaves the window, are excluded.
///    Few-equation validation cannot reliably reject such degenerate fits --
///    observed live, a 3000:1 needle three times the window won a 4-equation
///    holdout by 0.02 -- and this also kills the center runaway structurally.
///  - SCORE: linear-stage K-fold cross-validation. The parameters are fit once
///    per candidate on all k equations; the LINEAR coefficients are then refit
///    leave-fold-out at those parameters, so every equation is scored
///    out-of-sample for the linear stage at the price of one nonlinear fit.
///    Never the in-sample cost: a degenerate fit can lower that while ruining
///    the model.
///  - SIMPLICITY TIE-BREAK: among admissible candidates within `tie_delta` of
///    the best score, prefer fewer modes, then pinned over released.
///
/// **Early stopping is one-sided.** Certificates fire only when the data shows
/// the target is easy; a hard target fails them and buys the full grid.
/// `target_score` stops everything once an admissible candidate is certifiably
/// good; `mode_patience` stops growing the ladder. There is deliberately NO
/// patience on the initialization axis: score-versus-scale is not unimodal, and
/// two initializations can agree on the wrong minimum, so those prune only via
/// the absolute certificate.
///
/// **No randomness.** The cross-validation split and the warm-start jitter
/// arrive as DATA (see the plan's randomness section), so this function and
/// everything beneath it are pure functions of their inputs. Defaults are
/// deterministic; the operator layer overrides them.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/ellipsoid_transform.hpp"
#include "lgpsf/exceptions.hpp"
#include "lgpsf/init_dictionary.hpp"
#include "lgpsf/lg_expansion.hpp"
#include "lgpsf/mode_policy.hpp"
#include "lgpsf/varpro.hpp"
#include "lgpsf/whitening.hpp"

namespace lgpsf {

/// One fold of a cross-validation split.
///
/// Train and validation indices are both stored, rather than "validation, and
/// train is the complement", so the type can express splits that are not
/// partitions: overlapping validation sets, subsampled training sets, a single
/// designed holdout.
struct CvFold
{
    Eigen::VectorXi train;
    Eigen::VectorXi validation;
};

using CrossValidationSplit = std::vector<CvFold>;

namespace detail {

/// A uniform in (0, 1) from raw generator output. The half prevents an exact
/// zero, and no `std::*_distribution` appears anywhere: `mt19937` is specified
/// by the standard but the distributions' algorithms are not, so they differ
/// across standard libraries.
inline double uniform_unit( std::mt19937& gen )
{
    return (static_cast<double>(gen()) + 0.5) / 4294967296.0;
}

/// An unbiased integer in [0, bound), by rejection.
inline std::uint32_t uniform_below( std::mt19937& gen, std::uint32_t bound )
{
    const std::uint32_t threshold = (0u - bound) % bound;
    while ( true )
    {
        const std::uint32_t draw = gen();
        if ( draw >= threshold )
        {
            return draw % bound;
        }
    }
}

inline int resolved_fold_count( int num_probes, int requested )
{
    return std::max(2, std::min(requested, num_probes / 2));
}

/// Build a split by dealing the given order round-robin into folds.
inline CrossValidationSplit deal_folds( const std::vector<int>& order, int num_folds )
{
    const int total = static_cast<int>(order.size());
    CrossValidationSplit split;
    split.reserve(static_cast<std::size_t>(num_folds));
    for ( int f = 0; f < num_folds; ++f )
    {
        std::vector<int> validation, train;
        for ( int i = 0; i < total; ++i )
        {
            ( i % num_folds == f ? validation : train ).push_back(order[static_cast<std::size_t>(i)]);
        }
        CvFold fold;
        fold.validation = Eigen::Map<const Eigen::VectorXi>(
            validation.data(), static_cast<Eigen::Index>(validation.size()));
        fold.train = Eigen::Map<const Eigen::VectorXi>(
            train.data(), static_cast<Eigen::Index>(train.size()));
        split.push_back(std::move(fold));
    }
    return split;
}

} // end namespace detail

/// A deterministic round-robin split: fold f holds the probes with index
/// congruent to f.
///
/// No generator at all. The probes are iid by construction, so probe index
/// carries no information and the split is a nuisance parameter doing no
/// statistical work -- see the plan's randomness section, including when that
/// stops being true (structured probes).
inline CrossValidationSplit kfold_split( int num_probes, int num_folds )
{
    const int folds = detail::resolved_fold_count(num_probes, num_folds);
    std::vector<int> order(static_cast<std::size_t>(num_probes));
    for ( int i = 0; i < num_probes; ++i )
    {
        order[static_cast<std::size_t>(i)] = i;
    }
    return detail::deal_folds(order, folds);
}

/// A permuted split: the same round-robin deal over a shuffled order.
///
/// An explicit opt-in, for checking that a result is not an artifact of one
/// particular partition.
inline CrossValidationSplit kfold_split( int num_probes, int num_folds,
                                         std::uint32_t seed )
{
    const int folds = detail::resolved_fold_count(num_probes, num_folds);
    std::vector<int> order(static_cast<std::size_t>(num_probes));
    for ( int i = 0; i < num_probes; ++i )
    {
        order[static_cast<std::size_t>(i)] = i;
    }
    std::mt19937 gen(seed);
    for ( std::size_t i = order.size(); i > 1; --i )
    {
        const std::size_t j = detail::uniform_below(gen, static_cast<std::uint32_t>(i));
        std::swap(order[i - 1], order[j]);
    }
    return detail::deal_folds(order, folds);
}

/// Warm-start perturbations, one per ladder level: 0.05 * U(-1, 1).
///
/// A table rather than draws taken during the fit, so the fit stays a pure
/// function of its inputs. Exact warm starts sit on the enrichment saddle, so
/// the perturbation matters; its distribution does not.
inline std::vector<Eigen::VectorXd> jitter_table( int num_params, int num_levels,
                                                  std::uint32_t seed = 0 )
{
    std::mt19937 gen(seed);
    std::vector<Eigen::VectorXd> table;
    table.reserve(static_cast<std::size_t>(num_levels));
    for ( int level = 0; level < num_levels; ++level )
    {
        Eigen::VectorXd row(num_params);
        for ( int q = 0; q < num_params; ++q )
        {
            row(q) = 0.05 * (2.0 * detail::uniform_unit(gen) - 1.0);
        }
        table.push_back(std::move(row));
    }
    return table;
}

/// Whether the ellipsoid center is a competing hypothesis, and when.
enum class MuPolicy
{
    /// Pin the center at mu0 throughout. **The default.** At operator scale
    /// releasing shipped on ~91% of PIG rows while buying nothing on held-out
    /// data, guarded only by a far-too-loose window-radius bound; release
    /// stays available but is no longer the default until a basin-scale
    /// ||mu - mu0|| bound re-arms it.
    Pinned,
    /// Fit the center from the start, seeded at mu0.
    Free,
    /// Pin it for the stream, then release the winning level's fits and accept
    /// a released one only under the simplicity tie-break.
    PinnedThenRelease
};

/// Why the search stopped.
enum class StopReason
{
    Target,        ///< An admissible candidate met `target_score`.
    ModePatience,  ///< The mode ladder stopped improving.
    Exhausted      ///< The policy ran out of proposals.
};

inline const char* to_string( StopReason reason )
{
    switch ( reason )
    {
        case StopReason::Target:       return "target";
        case StopReason::ModePatience: return "mode_patience";
        case StopReason::Exhausted:    return "exhausted";
    }
    return "unknown";
}

struct ProbeFitConfig
{
    MuPolicy mu = MuPolicy::Pinned;

    /// The ladder axis. Required -- there is one mechanism, so an explicit
    /// mode list is `FixedSet` and a level ladder is `ShellLadder`.
    std::shared_ptr<const ModePolicy> mode_policy;

    /// How many default circle rungs to add, at log-spaced scales from the
    /// local mesh spacing to the batch radius -- `circle_ladder`'s dictionary.
    ///
    /// **0 means "only the guesses I passed"**, and is an error if none were.
    /// Anything else appends that many rungs AFTER the caller's guesses, so a
    /// supplied a-priori shape is still tried first.
    int num_rungs = 3;

    /// Absolute early-exit certificate: stop once an admissible candidate
    /// scores at or below this. Unset disables it.
    std::optional<double> target_score = 0.05;

    /// Stop growing the ladder after this many consecutive levels without
    /// improving the best score. Two or more, because a single-step worsening
    /// on a nested family is usually noise.
    int mode_patience = 2;

    /// Simplicity tie-break margin.
    double tie_delta = 0.0;

    int cv_folds = 5;

    /// The split, as data. Empty means "build the deterministic round-robin
    /// split from cv_folds".
    CrossValidationSplit split;

    /// Warm-start perturbations, one per level. Empty means "build the
    /// deterministic default".
    std::vector<Eigen::VectorXd> jitter;

    VarProOptions varpro = [] {
        VarProOptions options;
        options.max_evaluations = 100;  // top-of-ladder rungs may wander
        return options;
    }();
};

/// One scored entry of the candidate table: the model it found, and how it
/// went. The same separation the operator layer makes -- nothing about the
/// search is needed to evaluate the model, and `model` decodes on its own.
struct CandidateFit
{
    LGExpansion model;

    std::string label;        ///< "circle r=..", "window r=..", "sigma0", "warm(..)", "release(..)".
    std::string modes_label;

    /// Whether this candidate's center was fitted. Pure provenance: `model`
    /// stores absolute parameters, so nothing needs this to decode it.
    bool released = false;

    /// Where this candidate started, in the PUBLIC absolute encoding -- the
    /// same one `model.theta` uses, so `theta_init.head(N)` is the center it
    /// was seeded at and the pair reads as "started here, finished there".
    ///
    /// Absolute rather than the internal `theta_hat` because candidates no
    /// longer share one origin: a guess may carry its own center, and a
    /// displacement against an unstated reference cannot be decoded.
    Eigen::VectorXd theta_init;

    /// In-sample whitened cost -- a diagnostic, NEVER used for selection.
    double cost = 0.0;
    /// Linear-stage cross-validation score at the fitted parameters.
    double score = std::numeric_limits<double>::infinity();

    Eigen::VectorXd axes;  ///< Fitted 1-sigma semi-axes.
    bool success = false;
    int num_iterations = 0;

    /// False if the fit violates window containment -- impossible or runaway,
    /// by the conservativeness of the window.
    bool admissible = true;

    std::size_t num_modes() const { return model.modes.size(); }
};

/// The winning model, plus the full audit trail.
struct ProbeFitResult
{
    /// The winner. Its `theta` is the PUBLIC absolute encoding, so the ellipsoid
    /// comes from `model.frame()` -- no mu0, no mode.
    LGExpansion model;

    bool released = false;  ///< Whether the winner's center was fitted.

    double score = std::numeric_limits<double>::infinity();
    StopReason stop_reason = StopReason::Exhausted;
    int winner = -1;  ///< Index into `candidates`.

    std::vector<CandidateFit> candidates;
    std::vector<std::string> skipped;  ///< Levels the counting rule rejected.
};

/// Linear-stage K-fold cross-validation score of a model at given parameters.
///
/// The linear coefficients are refit leave-fold-out at those parameters and
/// scored on the held folds; the result is the relative whitened residual over
/// all scored equations.
///
/// Public because it scores ANY model -- fitted here or supplied a priori --
/// at zero additional nonlinear fits, which makes a field-wide quality map
/// affordable before fitting anything.
///
/// @param z_hat     (K, k) whitened probe fields.
/// @param y_hat     (k,) whitened responses.
/// @param basis     The basis functor to evaluate at @p theta_hat.
/// @param theta_hat The parameters to score. NOT fitted here.
/// @param e_hat     (K, num_extra) whitened extra basis; (K, 0) for none.
/// @param split     The folds. Must partition the probes.
/// @return          Relative whitened residual over all held-out equations;
///                  infinity if the model cannot be evaluated there.
template <typename Basis>
double linear_cv_score( const Eigen::Ref<const Eigen::MatrixXd>& z_hat,
                        const Eigen::Ref<const Eigen::VectorXd>& y_hat,
                        const Basis& basis,
                        const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
                        const Eigen::Ref<const Eigen::MatrixXd>& e_hat,
                        const CrossValidationSplit& split )
{
    const double infinite = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd values;
    try
    {
        values = basis(theta_hat).values();
    }
    catch ( const InfeasibleParameters& )
    {
        return infinite;
    }
    if ( !values.allFinite() )
    {
        return infinite;
    }

    Eigen::MatrixXd design(z_hat.cols(), values.cols() + e_hat.cols());
    design.leftCols(values.cols()) = z_hat.transpose() * values;
    if ( e_hat.cols() > 0 )
    {
        design.rightCols(e_hat.cols()) = z_hat.transpose() * e_hat;
    }
    if ( !design.allFinite() )
    {
        return infinite;
    }

    double squared = 0.0;
    for ( const CvFold& fold : split )
    {
        Eigen::MatrixXd train(fold.train.size(), design.cols());
        Eigen::VectorXd target(fold.train.size());
        for ( Eigen::Index i = 0; i < fold.train.size(); ++i )
        {
            train.row(i) = design.row(fold.train(i));
            target(i) = y_hat(fold.train(i));
        }
        // A pivoted QR where the fold is full rank, the SVD where it is not:
        // the two agree there, and only the rank-deficient case needs the
        // minimum-norm solution the SVD gives. Same trade as `inner_solve`.
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(train);
        const Eigen::VectorXd coefficients =
            ( qr.rank() == train.cols() )
                ? Eigen::VectorXd(qr.solve(target))
                : Eigen::VectorXd(
                      train.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV)
                          .solve(target));
        for ( Eigen::Index i = 0; i < fold.validation.size(); ++i )
        {
            const Eigen::Index row = fold.validation(i);
            const double residual = y_hat(row) - design.row(row).dot(coefficients);
            squared += residual * residual;
        }
    }
    return std::sqrt(squared) / std::max(y_hat.norm(), 1e-300);
}

namespace detail {

/// Fitted 1-sigma semi-axes: sqrt(eig(L L^T)).
inline Eigen::VectorXd axes_of( const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
                                const Eigen::Ref<const Eigen::VectorXd>& mu0,
                                MuMode mode )
{
    const EllipsoidFrame frame = unpack_theta_hat(theta_hat, mu0, mode);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        Eigen::MatrixXd(frame.L * frame.L.transpose()));
    return solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
}

} // end namespace detail

/// Fit one target from raw probe data.
///
/// Runs the ordered candidate stream -- initial ellipsoids x scales x mode-set
/// rungs -- scoring each on held-out probes, and returns the best along with
/// the whole search. See the file comment for the architecture, and
/// docs/defaults.md for what the policy knobs do.
///
/// @param x        (K, N) batch coordinates, selected by the caller.
/// @param m2_diag  (K,) lumped masses on the batch.
/// @param z        (K, k) raw probe fields restricted to the batch.
/// @param y        (k,) the target's raw responses.
/// @param default_mu  (N,) the row's center, e.g. the target's own dof
///                 location. Three roles: it carries the spatial dimension
///                 (`N = default_mu.size()`; there is no separate dimension
///                 parameter), it centers the default rungs and any guess that
///                 does not carry its own `mu`, and it sets the batch radius
///                 the admissibility guard uses.
/// @param spike_index  Index of the target's own point within the batch, which
///                 builds the one-hot spike; negative disables the spike,
///                 justified only for targets known not to be spike-dominated.
/// @param config   Candidate axes and search policy.
/// @param guesses  Starting ellipsoids, tried in order and BEFORE the default
///                 rungs. Where to seed a nonlinear search is problem-specific,
///                 so this is data rather than a flag selecting among
///                 dictionaries chosen in advance; `circle_ladder`,
///                 `window_shape_ladder` and `oriented_ladder` build the common
///                 ones. Pass the a-priori covariance here to try it first,
///                 which is what `fit_operator` does.
/// @param target_mass  The target's own mass. Unset infers it from
///                 `m2_diag[spike_index]`, exact for the square equal-mass
///                 case and 1 with no spike. The whitened design is
///                 column-equilibrated inside the inner solve, so this
///                 rescales ONLY the returned coefficients -- the parameters,
///                 scores and whole selection are invariant -- but callers
///                 with M1 != M2 must pass it to get correctly scaled ones.
/// @return         The winning model, its held-out score, why the ladder
///                 stopped, and every candidate that was tried.
/// @throws std::invalid_argument if the shapes disagree, if
///         `config.mode_policy` is unset (it is required -- no order is
///         defensible as a silent default), if `spike_index` is out of range,
///         or if `config.num_rungs == 0` leaves no guesses at all.
inline ProbeFitResult fit_from_probes(
    const Eigen::Ref<const Eigen::MatrixXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& m2_diag,
    const Eigen::Ref<const Eigen::MatrixXd>& z,
    const Eigen::Ref<const Eigen::VectorXd>& y,
    const Eigen::Ref<const Eigen::VectorXd>& default_mu, int spike_index = -1,
    const ProbeFitConfig& config = ProbeFitConfig(),
    const std::vector<InitialGuess>& guesses = {},
    std::optional<double> target_mass = std::nullopt )
{
    const Eigen::Ref<const Eigen::VectorXd>& mu0 = default_mu;
    const int dim = static_cast<int>(mu0.size());
    const Eigen::Index num_points = x.rows();
    const Eigen::Index num_probes = z.cols();

    if ( x.cols() != dim )
    {
        throw std::invalid_argument(
            "lgpsf::fit_from_probes: x has " + std::to_string(x.cols())
            + " coordinate columns but mu0 has " + std::to_string(dim));
    }
    if ( m2_diag.size() != num_points || z.rows() != num_points )
    {
        throw std::invalid_argument(
            "lgpsf::fit_from_probes: x, m2_diag and z must agree on the batch size");
    }
    if ( y.size() != num_probes )
    {
        throw std::invalid_argument(
            "lgpsf::fit_from_probes: y has " + std::to_string(y.size())
            + " entries but z has " + std::to_string(num_probes) + " probes");
    }
    if ( spike_index >= num_points )
    {
        throw std::invalid_argument("lgpsf::fit_from_probes: spike_index is out of range");
    }
    if ( !config.mode_policy )
    {
        throw std::invalid_argument(
            "lgpsf::fit_from_probes: config.mode_policy is required (use FixedSet "
            "for an explicit mode list, ShellLadder for a level ladder)");
    }

    const int num_extra = ( spike_index >= 0 ) ? 1 : 0;
    const double mass =
        target_mass ? *target_mass
                    : (spike_index >= 0 ? m2_diag(spike_index) : 1.0);

    Eigen::MatrixXd extra = Eigen::MatrixXd::Zero(num_points, num_extra);
    if ( num_extra > 0 )
    {
        extra(spike_index, 0) = 1.0;
    }
    const Eigen::MatrixXd e_hat = whiten_extra(extra, mass, m2_diag);
    const Eigen::MatrixXd z_hat = whiten_probes(z, m2_diag);
    const Eigen::VectorXd y_hat = whiten_data(y, mass);

    // The counting rule uses the PINNED parameter count regardless of the mu
    // policy, so the set of admissible levels does not move when the center is
    // released -- which keeps pinned and released candidates comparable on the
    // same mode sets.
    const int counting_params = theta_hat_size(dim, MuMode::Pinned);
    const MuMode ladder_mode =
        ( config.mu == MuPolicy::Free ) ? MuMode::Fitted : MuMode::Pinned;

    const CrossValidationSplit split =
        config.split.empty()
            ? kfold_split(static_cast<int>(num_probes), config.cv_folds)
            : config.split;

    // The admissibility guard's bound, needed whether or not any default rung
    // is generated.
    const double radius = window_radius(x, mu0);

    // A guess resolved into what the fit needs: the shape in the pinned
    // encoding, and the center it is anchored at.
    struct ResolvedInit
    {
        std::string label;
        Eigen::VectorXd theta_hat;
        Eigen::VectorXd center;
    };

    std::vector<ResolvedInit> inits;
    for ( std::size_t i = 0; i < guesses.size(); ++i )
    {
        const InitialGuess& guess = guesses[i];
        if ( guess.sigma.rows() != dim || guess.sigma.cols() != dim )
        {
            throw std::invalid_argument(
                "lgpsf::fit_from_probes: guess " + std::to_string(i)
                + " has a " + std::to_string(guess.sigma.rows()) + "x"
                + std::to_string(guess.sigma.cols()) + " sigma but N is "
                + std::to_string(dim));
        }
        if ( guess.mu && guess.mu->size() != dim )
        {
            throw std::invalid_argument(
                "lgpsf::fit_from_probes: guess " + std::to_string(i)
                + " has a mu of size " + std::to_string(guess.mu->size())
                + " but N is " + std::to_string(dim));
        }
        inits.push_back(ResolvedInit{
            guess.label.empty() ? "guess[" + std::to_string(i) + "]"
                                : guess.label,
            theta_hat_from_sigma(guess.sigma),
            guess.mu ? *guess.mu : Eigen::VectorXd(mu0)});
    }
    // The default dictionary: isotropic rungs spanning the batch's own scales.
    // They are the fallback for a prior that misleads -- see circle_ladder --
    // and `num_rungs = 0` is how a caller says "only my guesses".
    if ( config.num_rungs > 0 )
    {
        for ( const InitialGuess& rung :
              circle_ladder(x, mu0, config.num_rungs) )
        {
            inits.push_back(ResolvedInit{rung.label,
                                         theta_hat_from_sigma(rung.sigma),
                                         Eigen::VectorXd(mu0)});
        }
    }
    if ( inits.empty() )
    {
        throw std::invalid_argument(
            "lgpsf::fit_from_probes: nothing to start from -- num_rungs is 0 "
            "and no guesses were supplied");
    }

    const std::vector<Eigen::VectorXd> jitter =
        config.jitter.empty()
            ? jitter_table(theta_hat_size(dim, ladder_mode), kMaxModeProposals)
            : config.jitter;

    // Each candidate is fitted about its OWN center: a guess may carry one,
    // and under MuPolicy::Pinned that is where the center stays. The basis and
    // every decode of theta_hat therefore take the candidate's center rather
    // than a single row-wide origin.
    const auto make_basis = [&]( const std::vector<Mode>& modes, MuMode mode,
                                 const Eigen::VectorXd& center ) {
        return WhitenedBasis(x, mass, m2_diag, modes, center, mode);
    };

    // Bases are keyed on (modes, center) and shared: when every guess sits at
    // default_mu -- the usual case -- this builds exactly one per level, as it
    // did when the center was row-wide.
    struct BasisCache
    {
        std::vector<std::pair<Eigen::VectorXd, WhitenedBasis>> entries;

        const WhitenedBasis& get( const Eigen::VectorXd& center,
                                  const std::vector<Mode>& modes, MuMode mode,
                                  const std::function<WhitenedBasis(
                                      const std::vector<Mode>&, MuMode,
                                      const Eigen::VectorXd&)>& build )
        {
            for ( const auto& entry : entries )
            {
                if ( entry.first.size() == center.size()
                     && entry.first == center )
                {
                    return entry.second;
                }
            }
            entries.emplace_back(center, build(modes, mode, center));
            return entries.back().second;
        }
    };

    std::vector<CandidateFit> candidates;
    std::vector<std::string> skipped;

    const auto run_candidate = [&]( const std::string& label,
                                    const std::string& modes_label,
                                    const std::vector<Mode>& modes,
                                    const Eigen::VectorXd& start, MuMode mode,
                                    bool released, const WhitenedBasis& basis,
                                    const Eigen::VectorXd& center ) {
        CandidateFit candidate;
        candidate.label = label;
        candidate.modes_label = modes_label;
        candidate.released = released;
        candidate.theta_init = to_theta(start, center, mode);
        candidate.model.modes = modes;

        VarProResult fit;
        try
        {
            fit = fit_varpro(z_hat, y_hat, basis, start, e_hat, config.varpro);
        }
        catch ( const std::invalid_argument& )
        {
            // An unusable starting point -- a ladder rung far outside the
            // batch geometry. Expected and survivable: the candidate simply
            // scores as unusable and the stream moves on.
            candidate.model.theta = to_theta(start, center, mode);
            candidate.model.c =
                Eigen::VectorXd::Zero(static_cast<Eigen::Index>(modes.size()));
            candidate.model.s = Eigen::VectorXd::Zero(e_hat.cols());
            candidate.axes = Eigen::VectorXd::Zero(dim);
            candidate.admissible = false;
            return candidate;
        }

        candidate.model.theta = to_theta(fit.theta_hat, center, mode);
        candidate.model.c = fit.c;
        candidate.model.s = fit.s;
        candidate.cost = fit.cost;
        candidate.success = fit.success;
        candidate.num_iterations = fit.num_iterations;
        candidate.score =
            linear_cv_score(z_hat, y_hat, basis, fit.theta_hat, e_hat, split);
        candidate.axes = detail::axes_of(fit.theta_hat, center, mode);

        bool admissible = candidate.axes.maxCoeff() <= radius;
        if ( mode == MuMode::Fitted )
        {
            // The displacement encoding makes this the bound it always meant
            // to be: theta_hat's leading block IS mu - center.
            admissible = admissible && fit.theta_hat.head(dim).norm() <= radius;
        }
        candidate.admissible = admissible;
        return candidate;
    };

    // Adaptive feedback for policies that want it: the EXACT one-step
    // sum-of-squares reduction each candidate mode would buy at the current
    // winner's parameters. A projection against the active whitened design,
    // never a refit, so it costs one linear solve and one range basis per
    // level -- nothing beside the fits themselves.
    //
    // No policy shipped here consumes it (MarginGreedy is parked), but the
    // engine supplies it regardless, so an adaptive policy dropped in needs no
    // change on this side.
    //
    // Meaningful only for modes NOT already active: an active mode's column
    // projects to roundoff, leaving a ratio of two near-zeros that the
    // relative floor cannot discriminate. Margin groups are never active, so
    // the intended callers never meet that case.
    struct MarginScorer
    {
        std::function<Eigen::VectorXd( const std::vector<Mode>& )> profit;
        double residual_norm_squared = 0.0;
    };
    // Takes the center the theta_hat is encoded against: the winner's, which
    // need not be default_mu.
    const auto make_margin_scorer = [&]( const Eigen::VectorXd& theta_hat,
                                         const std::vector<Mode>& active,
                                         const Eigen::VectorXd& center ) {
        MarginScorer scorer;
        const WhitenedBasis active_basis =
            make_basis(active, ladder_mode, center);
        Eigen::MatrixXd values;
        try
        {
            values = active_basis(theta_hat).values();
        }
        catch ( const InfeasibleParameters& )
        {
            return scorer;
        }
        Eigen::MatrixXd design(num_probes, values.cols() + e_hat.cols());
        design.leftCols(values.cols()) = z_hat.transpose() * values;
        if ( e_hat.cols() > 0 )
        {
            design.rightCols(e_hat.cols()) = z_hat.transpose() * e_hat;
        }
        if ( !design.allFinite() )
        {
            return scorer;
        }

        const Eigen::VectorXd coefficients =
            design.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y_hat);
        const Eigen::VectorXd residual = y_hat - design * coefficients;
        const Eigen::MatrixXd range = detail::orthonormal_range(design);

        scorer.residual_norm_squared = residual.squaredNorm();
        scorer.profit = [&, theta_hat, residual, range, center](
                            const std::vector<Mode>& candidate_modes ) {
            const WhitenedBasis candidate_basis =
                make_basis(candidate_modes, ladder_mode, center);
            Eigen::MatrixXd columns;
            try
            {
                columns = z_hat.transpose() * candidate_basis(theta_hat).values();
            }
            catch ( const InfeasibleParameters& )
            {
                return Eigen::VectorXd(Eigen::VectorXd::Zero(
                    static_cast<Eigen::Index>(candidate_modes.size())));
            }
            if ( !columns.allFinite() )
            {
                return Eigen::VectorXd(Eigen::VectorXd::Zero(columns.cols()));
            }
            columns = detail::project_out(range, columns);

            Eigen::VectorXd out = Eigen::VectorXd::Zero(columns.cols());
            Eigen::VectorXd denominator(columns.cols());
            for ( Eigen::Index j = 0; j < columns.cols(); ++j )
            {
                denominator(j) = columns.col(j).squaredNorm();
            }
            const double floor =
                1e-30 * std::max(denominator.size() ? denominator.maxCoeff() : 0.0,
                                 1e-300);
            for ( Eigen::Index j = 0; j < columns.cols(); ++j )
            {
                if ( denominator(j) > floor )
                {
                    const double inner = columns.col(j).dot(residual);
                    out(j) = inner * inner / denominator(j);
                }
            }
            return out;
        };
        return scorer;
    };

    const auto select = [&]( const std::vector<CandidateFit>& pool ) {
        std::vector<int> allowed;
        for ( std::size_t i = 0; i < pool.size(); ++i )
        {
            if ( pool[i].admissible )
            {
                allowed.push_back(static_cast<int>(i));
            }
        }
        if ( allowed.empty() )
        {
            for ( std::size_t i = 0; i < pool.size(); ++i )
            {
                allowed.push_back(static_cast<int>(i));
            }
        }
        double best = std::numeric_limits<double>::infinity();
        for ( int i : allowed )
        {
            best = std::min(best, pool[static_cast<std::size_t>(i)].score);
        }
        int winner = allowed.front();
        bool have = false;
        for ( int i : allowed )
        {
            const CandidateFit& candidate = pool[static_cast<std::size_t>(i)];
            if ( !(candidate.score <= best + config.tie_delta) )
            {
                continue;
            }
            const CandidateFit& incumbent = pool[static_cast<std::size_t>(winner)];
            const bool better =
                !have
                || std::make_tuple(candidate.num_modes(), candidate.released, i)
                       < std::make_tuple(incumbent.num_modes(), incumbent.released,
                                         winner);
            if ( better )
            {
                winner = i;
                have = true;
            }
        }
        return winner;
    };

    const auto hit_target = [&]( const CandidateFit& candidate ) {
        return config.target_score && candidate.admissible
               && candidate.score <= *config.target_score;
    };

    // --- the candidate stream ---------------------------------------------
    std::vector<LevelRecord> history;
    std::vector<std::pair<std::string, std::vector<Mode>>> modes_of;
    StopReason stop_reason = StopReason::Exhausted;
    double best_score = std::numeric_limits<double>::infinity();
    int patience_left = config.mode_patience;
    bool have_warm_start = false;
    Eigen::VectorXd warm_start;
    Eigen::VectorXd warm_center = mu0;
    // One center per candidate, parallel to `candidates`: the decode origin is
    // per-candidate now, and the winner's has to survive into the warm start.
    std::vector<Eigen::VectorXd> level_centers;
    std::vector<Mode> last_fit_modes;
    bool last_fit_valid = false;
    bool stopped = false;
    MarginScorer scorer;

    while ( static_cast<int>(history.size()) < kMaxModeProposals )
    {
        ModeSearchContext ctx;
        ctx.dim = dim;
        ctx.num_probes = static_cast<int>(num_probes);
        ctx.num_extra = num_extra;
        ctx.num_params = counting_params;
        ctx.history = history;
        ctx.margin_profit = scorer.profit;
        ctx.residual_norm_squared = scorer.residual_norm_squared;

        std::optional<ModeProposal> proposal = config.mode_policy->propose(ctx);
        if ( !proposal )
        {
            break;
        }
        for ( const LevelRecord& record : history )
        {
            if ( record.label == proposal->label )
            {
                throw std::invalid_argument(
                    "lgpsf::fit_from_probes: the mode policy reused the label '"
                    + proposal->label + "'");
            }
        }
        if ( last_fit_valid )
        {
            const std::set<Mode> offered(proposal->modes.begin(), proposal->modes.end());
            for ( const Mode& mode : last_fit_modes )
            {
                if ( offered.find(mode) == offered.end() )
                {
                    throw std::invalid_argument(
                        "lgpsf::fit_from_probes: the mode policy's proposal '"
                        + proposal->label + "' does not contain the previously "
                        "fitted set (nested growth is the contract)");
                }
            }
        }

        if ( static_cast<int>(num_probes)
             < 2 * (static_cast<int>(proposal->modes.size()) + num_extra
                    + counting_params) )
        {
            skipped.push_back(proposal->label);
            history.push_back(
                LevelRecord{proposal->label, proposal->modes, true, false});
            continue;
        }
        if ( patience_left <= 0 )
        {
            stop_reason = StopReason::ModePatience;
            break;
        }

        modes_of.emplace_back(proposal->label, proposal->modes);
        BasisCache level_bases;

        struct LevelInit
        {
            std::string label;
            Eigen::VectorXd theta_hat;
            Eigen::VectorXd center;
        };
        std::vector<LevelInit> level_inits;
        if ( have_warm_start )
        {
            const Eigen::VectorXd& kick =
                jitter[std::min(history.size(), jitter.size() - 1)];
            level_inits.push_back(
                LevelInit{"warm(" + candidates.back().modes_label + ")",
                          Eigen::VectorXd(warm_start + kick), warm_center});
        }
        for ( const ResolvedInit& init : inits )
        {
            level_inits.push_back(
                LevelInit{init.label,
                          ladder_mode == MuMode::Pinned
                              ? init.theta_hat
                              : release_mu(init.theta_hat, dim),
                          init.center});
        }

        const std::size_t level_start = candidates.size();
        for ( const LevelInit& entry : level_inits )
        {
            const WhitenedBasis& basis = level_bases.get(
                entry.center, proposal->modes, ladder_mode, make_basis);
            candidates.push_back(run_candidate(entry.label, proposal->label,
                                               proposal->modes, entry.theta_hat,
                                               ladder_mode, false, basis,
                                               entry.center));
            level_centers.push_back(entry.center);
            if ( hit_target(candidates.back()) )
            {
                stop_reason = StopReason::Target;
                stopped = true;
                break;
            }
        }

        int level_winner = -1;
        for ( std::size_t i = level_start; i < candidates.size(); ++i )
        {
            if ( candidates[i].admissible
                 && (level_winner < 0
                     || candidates[i].score
                            < candidates[static_cast<std::size_t>(level_winner)].score) )
            {
                level_winner = static_cast<int>(i);
            }
        }
        if ( level_winner >= 0
             && candidates[static_cast<std::size_t>(level_winner)].score < best_score )
        {
            best_score = candidates[static_cast<std::size_t>(level_winner)].score;
            patience_left = config.mode_patience;
            warm_center = level_centers[static_cast<std::size_t>(level_winner)];
            warm_start = to_theta_hat(
                candidates[static_cast<std::size_t>(level_winner)].model.theta,
                warm_center, ladder_mode);
            have_warm_start = true;
        }
        else
        {
            --patience_left;
        }

        history.push_back(LevelRecord{proposal->label, proposal->modes, false,
                                      level_winner >= 0});
        if ( level_winner >= 0 )
        {
            const Eigen::VectorXd& winner_center =
                level_centers[static_cast<std::size_t>(level_winner)];
            scorer = make_margin_scorer(
                to_theta_hat(
                    candidates[static_cast<std::size_t>(level_winner)].model.theta,
                    winner_center, ladder_mode),
                proposal->modes, winner_center);
        }
        last_fit_modes = proposal->modes;
        last_fit_valid = true;
        if ( stopped )
        {
            break;
        }
    }

    if ( candidates.empty() )
    {
        throw std::invalid_argument(
            "lgpsf::fit_from_probes: no mode set passed the counting rule "
            "k >= 2*(m + " + std::to_string(num_extra) + " + "
            + std::to_string(counting_params) + ") at k="
            + std::to_string(num_probes)
            + " (skipped " + std::to_string(skipped.size()) + " proposal(s))");
    }

    int winner = select(candidates);

    // --- guarded release stage --------------------------------------------
    if ( config.mu == MuPolicy::PinnedThenRelease && stop_reason != StopReason::Target )
    {
        const std::string winning_label = candidates[static_cast<std::size_t>(winner)].modes_label;
        std::vector<Mode> winning_modes;
        for ( const auto& entry : modes_of )
        {
            if ( entry.first == winning_label )
            {
                winning_modes = entry.second;
            }
        }
        BasisCache free_bases;

        std::vector<int> sources;
        for ( std::size_t i = 0; i < candidates.size(); ++i )
        {
            if ( candidates[i].modes_label == winning_label && candidates[i].admissible
                 && !candidates[i].released )
            {
                sources.push_back(static_cast<int>(i));
            }
        }
        std::stable_sort(sources.begin(), sources.end(), [&]( int a, int b ) {
            return candidates[static_cast<std::size_t>(a)].score
                   < candidates[static_cast<std::size_t>(b)].score;
        });

        std::vector<Eigen::VectorXd> seen;
        for ( int index : sources )
        {
            const CandidateFit& source = candidates[static_cast<std::size_t>(index)];
            bool duplicate = false;
            for ( const Eigen::VectorXd& previous : seen )
            {
                duplicate = duplicate
                            || (previous - source.model.theta).cwiseAbs().maxCoeff() < 1e-8;
            }
            if ( duplicate )
            {
                continue;
            }
            seen.push_back(source.model.theta);

            // Re-encode against the SOURCE's own center, not a row-wide one:
            // that is what makes this exactly release_mu of the pinned
            // candidate, so the released fit starts where the pinned one
            // finished with a zero displacement. Against any other origin it
            // would silently start displaced -- and still converge to
            // something, which is why this is spelled out.
            const Eigen::VectorXd& source_center =
                level_centers[static_cast<std::size_t>(index)];
            const WhitenedBasis& free_basis = free_bases.get(
                source_center, winning_modes, MuMode::Fitted, make_basis);
            candidates.push_back(run_candidate(
                "release(" + source.label + ")", winning_label, winning_modes,
                to_theta_hat(source.model.theta, source_center, MuMode::Fitted),
                MuMode::Fitted, true, free_basis, source_center));
            level_centers.push_back(source_center);
            if ( hit_target(candidates.back()) )
            {
                stop_reason = StopReason::Target;
                break;
            }
        }
        winner = select(candidates);
    }

    const CandidateFit& champion = candidates[static_cast<std::size_t>(winner)];

    ProbeFitResult result;
    result.model = champion.model;
    result.released = champion.released;
    result.score = champion.score;
    result.stop_reason = stop_reason;
    result.winner = winner;
    result.candidates = std::move(candidates);
    result.skipped = std::move(skipped);
    return result;
}

} // end namespace lgpsf
