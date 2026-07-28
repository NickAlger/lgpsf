#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Mode-growth policies: the extensible ladder axis of the probe fit.
///
/// See dev/archive/mode-policy-plan.md for the design and the PIG slice-38 evidence
/// behind it. The short version: the best growth ORDER is budget- and
/// row-dependent -- complete shells win at k = 20, an ell-capped radial wedge
/// ties them at a sixth of the cost at k = 100 -- so the ladder is a pluggable
/// POLICY rather than a decree.
///
/// **Division of labour.** A policy PROPOSES mode sets, one at a time. The
/// engine keeps every structural guard and all selection semantics: the
/// counting rule, window-containment admissibility, cross-validation scoring,
/// the patience and target certificates, warm starts, the simplicity
/// tie-break. Policies never see masses or whitening.
///
/// **Policies are STATELESS**, and the interface enforces it -- `propose` is
/// const and derives its position entirely from `ctx.history`. That makes the
/// feedback-blind baseline replay trivial, keeps resume and debugging simple,
/// and means a policy can be shared across threads and rows without a thought.
///
/// Engine-enforced contracts: each proposal must contain the previously FITTED
/// set (patience and warm-start semantics assume a growing set); labels must
/// be unique; oversized proposals are recorded as skipped and the policy is
/// polled again; a hard proposal cap guards non-terminating policies.
///
/// `MarginGreedy` is deliberately NOT ported: it is parked pending the
/// novelty-floor refinement (see the plan's addendum), and the Python
/// prototype remains the working reference for that follow-up.

#include <algorithm>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/harmonic_polynomials.hpp"
#include "lgpsf/lg_functions.hpp"

namespace lgpsf {

/// Engine-side cap on policy proposals per fit -- a runaway guard, not a
/// tuning knob.
inline constexpr int kMaxModeProposals = 64;

/// One proposed rung of the ladder.
struct ModeProposal
{
    std::string label;
    std::vector<Mode> modes;
};

/// One entry of the ladder history handed back to policies.
struct LevelRecord
{
    std::string label;
    std::vector<Mode> modes;

    /// True if the counting rule rejected this proposal, so it was never fit.
    bool skipped = false;

    /// Whether the level produced a winning admissible candidate. A BOOLEAN,
    /// not the candidate: policies only ever ask whether feedback exists, so
    /// carrying the fit itself would make this header depend on the engine
    /// that includes it.
    bool has_winner = false;
};

/// Everything a policy may condition on. Engine-built.
struct ModeSearchContext
{
    int dim = 0;
    int num_probes = 0;   ///< k, the number of probe equations.
    int num_extra = 0;
    int num_params = 0;   ///< P, the parameter count of the stream's encoding.

    std::vector<LevelRecord> history;

    /// Per-mode EXACT one-step sum-of-squares reductions for a candidate mode
    /// list, against the current winner's residual at its fitted parameters --
    /// a projection, not a refit. Empty before any successful fit.
    ///
    /// No policy shipped here consumes it; it exists so that an adaptive
    /// policy (MarginGreedy, when unparked) needs no change on the engine
    /// side. The engine supplies it after every successful level, at the cost
    /// of one linear solve and one range basis -- nothing beside the fits.
    ///
    /// Ask only about modes NOT already active. An active mode's column
    /// projects to roundoff, so its "profit" is a 0/0 and means nothing.
    std::function<Eigen::VectorXd( const std::vector<Mode>& )> margin_profit;

    /// Squared norm of that residual -- the noise-gate denominator.
    double residual_norm_squared = 0.0;

    /// Largest mode count passing the counting rule k >= 2 (m + n_extra + P).
    int max_modes() const
    {
        return num_probes / 2 - num_extra - num_params;
    }

    /// The most recent non-skipped record, or nullptr.
    const LevelRecord* last_fit() const
    {
        for ( auto it = history.rbegin(); it != history.rend(); ++it )
        {
            if ( !it->skipped )
            {
                return &*it;
            }
        }
        return nullptr;
    }
};

/// Base class for mode-growth policies.
class ModePolicy
{
public:
    virtual ~ModePolicy() = default;

    /// The next set to fit, or nothing to end the ladder.
    virtual std::optional<ModeProposal> propose( const ModeSearchContext& ctx ) const = 0;

    /// Mode sets the operator layer's a-priori baseline guard may score.
    ///
    /// The baseline must not depend on any adaptive trajectory, so the default
    /// replays `propose` against a context with empty feedback -- which is
    /// exact for every non-adaptive policy, and the reason adaptive ones must
    /// override it.
    virtual std::vector<std::vector<Mode>> baseline_sets(
        const ModeSearchContext& ctx ) const
    {
        ModeSearchContext blind;
        blind.dim = ctx.dim;
        blind.num_probes = ctx.num_probes;
        blind.num_extra = ctx.num_extra;
        blind.num_params = ctx.num_params;

        std::vector<std::vector<Mode>> sets;
        while ( static_cast<int>(blind.history.size()) < kMaxModeProposals )
        {
            std::optional<ModeProposal> proposal = propose(blind);
            if ( !proposal )
            {
                break;
            }
            blind.history.push_back(
                LevelRecord{proposal->label, proposal->modes, false, false});
            sets.push_back(std::move(proposal->modes));
        }
        return sets;
    }
};

namespace detail {

inline std::string formatted( const char* format, int a, int b = 0 )
{
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), format, a, b);
    return std::string(buffer);
}

/// All modes of one (p, ell) group: the harmonic multiplet that always travels
/// together (the cos/sin pair in 2D). Empty past the generated table, so a
/// policy can walk ell upward without knowing where the table stops.
inline std::vector<Mode> harmonic_group( int dim, int p, int ell )
{
    if ( ell > max_degree() )
    {
        return {};
    }
    std::vector<Mode> group;
    const int count = num_harmonics(dim, ell);
    group.reserve(static_cast<std::size_t>(count));
    for ( int m = 0; m < count; ++m )
    {
        group.push_back(Mode{p, ell, m});
    }
    return group;
}

} // end namespace detail

/// A feedback-blind policy: a fixed sequence indexed by how many proposals
/// have already been made.
class SequencePolicy : public ModePolicy
{
public:
    std::optional<ModeProposal> propose( const ModeSearchContext& ctx ) const override
    {
        const std::vector<ModeProposal> sequence = build_sequence(ctx);
        const std::size_t position = ctx.history.size();
        if ( position >= sequence.size() )
        {
            return std::nullopt;
        }
        return sequence[position];
    }

    /// The whole ladder this policy would walk, in order.
    virtual std::vector<ModeProposal> build_sequence(
        const ModeSearchContext& ctx ) const = 0;
};

/// A single explicit mode set -- the `modes` argument expressed as a policy.
class FixedSet : public SequencePolicy
{
public:
    explicit FixedSet( std::vector<Mode> modes, std::string label = "explicit" )
        : modes_(std::move(modes)), label_(std::move(label))
    {
    }

    std::vector<ModeProposal> build_sequence( const ModeSearchContext& ) const override
    {
        return {ModeProposal{label_, modes_}};
    }

private:
    std::vector<Mode> modes_;
    std::string label_;
};

/// Complete oscillator shells up to each listed level, ascending -- the
/// classic mode-levels ladder.
class ShellLadder : public SequencePolicy
{
public:
    explicit ShellLadder( std::vector<int> levels ) : levels_(std::move(levels))
    {
        std::sort(levels_.begin(), levels_.end());
    }

    std::vector<ModeProposal> build_sequence( const ModeSearchContext& ctx ) const override
    {
        std::vector<ModeProposal> sequence;
        sequence.reserve(levels_.size());
        for ( int level : levels_ )
        {
            sequence.push_back(ModeProposal{detail::formatted("level<=%d", level),
                                            modes_up_to_level(ctx.dim, level)});
        }
        return sequence;
    }

private:
    std::vector<int> levels_;
};

/// A caller-supplied nested list of mode sets.
class ExplicitLadder : public SequencePolicy
{
public:
    explicit ExplicitLadder( std::vector<std::vector<Mode>> sets )
        : sets_(std::move(sets))
    {
    }

    std::vector<ModeProposal> build_sequence( const ModeSearchContext& ) const override
    {
        std::vector<ModeProposal> sequence;
        sequence.reserve(sets_.size());
        for ( std::size_t i = 0; i < sets_.size(); ++i )
        {
            sequence.push_back(ModeProposal{
                detail::formatted("set%d(m=%d)", static_cast<int>(i),
                                  static_cast<int>(sets_[i].size())),
                sets_[i]});
        }
        return sequence;
    }

private:
    std::vector<std::vector<Mode>> sets_;
};

/// Level-ordered ell-capped wedges {2p + ell <= L, ell <= ell_max}, L
/// ascending -- the strongest fixed policy at k >= 40 in the PIG slice-38
/// study, and the recommended starting point for rows that are roughly
/// elliptical. There is no default policy: `fit_operator` requires one,
/// because the best growth order depends on the operator (a strongly angular
/// kernel is served badly by a small `ell_max`).
///
/// Consecutive equal sets are deduplicated: past the cap, raising L adds only
/// modes with ell > ell_max, so the wedge stops growing while L keeps going.
class WedgeLadder : public SequencePolicy
{
public:
    explicit WedgeLadder( int max_level = 10, int ell_max = 2 )
        : max_level_(max_level), ell_max_(ell_max)
    {
    }

    std::vector<ModeProposal> build_sequence( const ModeSearchContext& ctx ) const override
    {
        std::vector<ModeProposal> sequence;
        for ( int level = 0; level <= max_level_; ++level )
        {
            std::vector<Mode> modes = modes_up_to_level(ctx.dim, level, ell_max_);
            if ( sequence.empty() || modes.size() > sequence.back().modes.size() )
            {
                sequence.push_back(
                    ModeProposal{detail::formatted("wedge<=%d", level),
                                 std::move(modes)});
            }
        }
        return sequence;
    }

private:
    int max_level_;
    int ell_max_;
};

/// Pure-radial-FIRST prefixes: all radial groups (p, 0) ascending in p, then
/// the ell = 1 groups, then ell = 2, and so on.
///
/// The tight-budget hypothesis -- spend on radial depth before angular
/// structure. Note that at k = 20 on PIG complete shells beat it; it is kept
/// because the ordering question is exactly what this axis exists to explore.
/// Each rung admits `groups_per_rung` harmonic groups, the first rung being
/// the seed group alone.
class RadialFirstLadder : public SequencePolicy
{
public:
    explicit RadialFirstLadder( int max_level = 10, int ell_max = 2,
                                int groups_per_rung = 2 )
        : max_level_(max_level), ell_max_(ell_max),
          groups_per_rung_(std::max(1, groups_per_rung))
    {
    }

    std::vector<ModeProposal> build_sequence( const ModeSearchContext& ctx ) const override
    {
        std::vector<std::vector<Mode>> groups;
        for ( int p = 0; p <= max_level_ / 2; ++p )
        {
            std::vector<Mode> group = detail::harmonic_group(ctx.dim, p, 0);
            if ( !group.empty() )
            {
                groups.push_back(std::move(group));
            }
        }
        for ( int ell = 1; ell <= ell_max_; ++ell )
        {
            for ( int p = 0; p <= (max_level_ - ell) / 2; ++p )
            {
                std::vector<Mode> group = detail::harmonic_group(ctx.dim, p, ell);
                if ( !group.empty() )
                {
                    groups.push_back(std::move(group));
                }
            }
        }

        std::size_t total = 0;
        for ( const std::vector<Mode>& group : groups )
        {
            total += group.size();
        }

        std::vector<ModeProposal> sequence;
        for ( std::size_t stop = 1; stop <= groups.size();
              stop += static_cast<std::size_t>(groups_per_rung_) )
        {
            std::vector<Mode> modes;
            for ( std::size_t g = 0; g < stop; ++g )
            {
                modes.insert(modes.end(), groups[g].begin(), groups[g].end());
            }
            sequence.push_back(ModeProposal{
                detail::formatted("radial#%d(m=%d)", static_cast<int>(sequence.size()),
                                  static_cast<int>(modes.size())),
                std::move(modes)});
        }
        // The stride can overshoot the last group, so close the ladder on the
        // full set rather than stopping short of it.
        if ( !sequence.empty() && sequence.back().modes.size() < total )
        {
            std::vector<Mode> modes;
            for ( const std::vector<Mode>& group : groups )
            {
                modes.insert(modes.end(), group.begin(), group.end());
            }
            sequence.push_back(ModeProposal{
                detail::formatted("radial#%d(m=%d)", static_cast<int>(sequence.size()),
                                  static_cast<int>(modes.size())),
                std::move(modes)});
        }
        return sequence;
    }

private:
    int max_level_;
    int ell_max_;
    int groups_per_rung_;
};

} // end namespace lgpsf
