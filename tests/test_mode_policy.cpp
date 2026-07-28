// SPDX-License-Identifier: MIT
//
// Checks on the mode-growth policies: the properties the engine relies on
// (nestedness, statelessness, position from history alone, termination), the
// combinatorics each ladder claims, and the baseline-replay contract.
//
// All self-contained: nothing here is compared against a stored reference, so
// the suite cannot drift out of step with the code it tests.

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/mode_policy.hpp"

using lgpsf::ExplicitLadder;
using lgpsf::FixedSet;
using lgpsf::LevelRecord;
using lgpsf::Mode;
using lgpsf::ModePolicy;
using lgpsf::ModeProposal;
using lgpsf::ModeSearchContext;
using lgpsf::RadialFirstLadder;
using lgpsf::ShellLadder;
using lgpsf::WedgeLadder;
using lgpsf::kMaxModeProposals;
using lgpsf::modes_up_to_level;
using lgpsf::num_harmonics;

namespace {

ModeSearchContext context( int dim, int num_probes = 200, int num_extra = 1,
                           int num_params = 5 )
{
    ModeSearchContext ctx;
    ctx.dim = dim;
    ctx.num_probes = num_probes;
    ctx.num_extra = num_extra;
    ctx.num_params = num_params;
    return ctx;
}

/// Walk a policy to exhaustion the way the engine does: propose, record, poll
/// again. Every proposal is treated as fit and successful.
std::vector<ModeProposal> walk( const ModePolicy& policy, ModeSearchContext ctx )
{
    std::vector<ModeProposal> rungs;
    while ( static_cast<int>(ctx.history.size()) < kMaxModeProposals )
    {
        std::optional<ModeProposal> proposal = policy.propose(ctx);
        if ( !proposal )
        {
            break;
        }
        ctx.history.push_back(LevelRecord{proposal->label, proposal->modes, false, true});
        rungs.push_back(*proposal);
    }
    return rungs;
}

bool contains_all( const std::vector<Mode>& outer, const std::vector<Mode>& inner )
{
    const std::set<Mode> haystack(outer.begin(), outer.end());
    for ( const Mode& mode : inner )
    {
        if ( haystack.find(mode) == haystack.end() )
        {
            return false;
        }
    }
    return true;
}

/// A policy that never stops -- to check the runaway guard is real.
class NeverEnds : public ModePolicy
{
public:
    std::optional<ModeProposal> propose( const ModeSearchContext& ctx ) const override
    {
        return ModeProposal{"forever" + std::to_string(ctx.history.size()),
                            {Mode{0, 0, 0}}};
    }
};

} // namespace

TEST_CASE("every ladder grows, and every rung contains the one before it")
{
    // The engine's patience and warm-start semantics both assume a GROWING
    // mode set, so this is a contract every shipped policy owes it.
    const std::vector<Mode> explicit_modes = modes_up_to_level(2, 4);
    ShellLadder shells({0, 2, 4, 6});
    WedgeLadder wedge(10, 2);
    RadialFirstLadder radial(10, 2, 2);
    ExplicitLadder explicit_ladder({modes_up_to_level(2, 2), modes_up_to_level(2, 4)});

    const std::vector<const ModePolicy*> policies = {&shells, &wedge, &radial,
                                                     &explicit_ladder};
    for ( const ModePolicy* policy : policies )
    {
        for ( int dim = 1; dim <= 3; ++dim )
        {
            const std::vector<ModeProposal> rungs = walk(*policy, context(dim));
            REQUIRE(rungs.size() >= 1u);
            for ( std::size_t i = 1; i < rungs.size(); ++i )
            {
                CHECK(contains_all(rungs[i].modes, rungs[i - 1].modes));
                CHECK(rungs[i].modes.size() > rungs[i - 1].modes.size());
            }
            // labels are unique, which the engine relies on to key levels
            std::set<std::string> labels;
            for ( const ModeProposal& rung : rungs )
            {
                CHECK(labels.insert(rung.label).second);
            }
            // no duplicate modes within a rung
            for ( const ModeProposal& rung : rungs )
            {
                const std::set<Mode> unique(rung.modes.begin(), rung.modes.end());
                CHECK(unique.size() == rung.modes.size());
            }
        }
    }
}

TEST_CASE("policies are stateless: the answer depends only on the history")
{
    WedgeLadder wedge(10, 2);
    ModeSearchContext ctx = context(2);

    // asking twice at the same position gives the same answer
    const std::optional<ModeProposal> first = wedge.propose(ctx);
    const std::optional<ModeProposal> again = wedge.propose(ctx);
    REQUIRE(first.has_value());
    REQUIRE(again.has_value());
    CHECK(first->label == again->label);
    CHECK(first->modes == again->modes);

    // and a history rebuilt from scratch reaches the identical rung
    ModeSearchContext replayed = context(2);
    replayed.history.push_back(LevelRecord{"whatever", {}, false, true});
    ModeSearchContext advanced = context(2);
    advanced.history.push_back(LevelRecord{first->label, first->modes, false, true});
    CHECK(wedge.propose(replayed)->modes == wedge.propose(advanced)->modes);
}

TEST_CASE("the shell ladder is exactly the complete shells, in ascending order")
{
    for ( int dim = 1; dim <= 4; ++dim )
    {
        // deliberately unsorted input: the ladder must ascend regardless
        ShellLadder ladder({6, 0, 4, 2});
        const std::vector<ModeProposal> rungs = walk(ladder, context(dim));
        REQUIRE(rungs.size() == 4u);
        const std::vector<int> expected = {0, 2, 4, 6};
        for ( std::size_t i = 0; i < rungs.size(); ++i )
        {
            CHECK(rungs[i].modes == modes_up_to_level(dim, expected[i]));
            CHECK(rungs[i].label
                  == "level<=" + std::to_string(expected[i]));
        }
    }
}

TEST_CASE("the wedge caps the angular order and deduplicates stalled levels")
{
    const int dim = 2;
    const int ell_max = 2;
    WedgeLadder ladder(10, ell_max);
    const std::vector<ModeProposal> rungs = walk(ladder, context(dim));

    for ( const ModeProposal& rung : rungs )
    {
        for ( const Mode& mode : rung.modes )
        {
            CHECK(mode.ell <= ell_max);
        }
    }
    // strictly growing, because equal consecutive wedges are dropped
    for ( std::size_t i = 1; i < rungs.size(); ++i )
    {
        CHECK(rungs[i].modes.size() > rungs[i - 1].modes.size());
    }
    // the last rung is the full capped wedge at max_level
    CHECK(rungs.back().modes.size() == modes_up_to_level(dim, 10, ell_max).size());

    // an uncapped wedge is just the shell ladder over every level
    WedgeLadder uncapped(6, -1);
    const std::vector<ModeProposal> plain = walk(uncapped, context(dim));
    CHECK(plain.back().modes == modes_up_to_level(dim, 6));

    // a wedge capped at ell = 0 is pure radial.
    // (Note the named local: iterating `walk(...).back().modes` directly
    // dangles -- lifetime extension does not reach through the member
    // accesses, so the temporary dies before the loop body runs.)
    WedgeLadder radial_only(8, 0);
    const std::vector<ModeProposal> pure_radial = walk(radial_only, context(dim));
    for ( const Mode& mode : pure_radial.back().modes )
    {
        CHECK(mode.ell == 0);
    }
}

TEST_CASE("the radial-first ladder spends on radial depth before angular structure")
{
    const int dim = 2;
    RadialFirstLadder ladder(10, 2, 2);
    const std::vector<ModeProposal> rungs = walk(ladder, context(dim));
    REQUIRE(rungs.size() >= 2u);

    // Its defining property: every purely radial mode the ladder will ever
    // admit is present before the first angular one appears.
    std::size_t first_angular = rungs.size();
    for ( std::size_t i = 0; i < rungs.size(); ++i )
    {
        for ( const Mode& mode : rungs[i].modes )
        {
            if ( mode.ell > 0 )
            {
                first_angular = std::min(first_angular, i);
            }
        }
    }
    REQUIRE(first_angular < rungs.size());
    const std::vector<Mode>& before = rungs[first_angular].modes;
    const int radial_total = 10 / 2 + 1;  // (p, 0) for p = 0..5
    int radial_present = 0;
    for ( const Mode& mode : before )
    {
        if ( mode.ell == 0 )
        {
            ++radial_present;
        }
    }
    CHECK(radial_present == radial_total);

    // the ladder ends on the full group set, whatever the stride
    for ( int per_rung : {1, 2, 3, 5, 100 } )
    {
        RadialFirstLadder strided(10, 2, per_rung);
        const std::vector<ModeProposal> walked = walk(strided, context(dim));
        CHECK(walked.back().modes.size()
              == walk(RadialFirstLadder(10, 2, 1), context(dim)).back().modes.size());
    }
}

TEST_CASE("a fixed set is a one-rung ladder")
{
    const std::vector<Mode> modes = modes_up_to_level(2, 4);
    FixedSet policy(modes, "mine");
    const std::vector<ModeProposal> rungs = walk(policy, context(2));
    REQUIRE(rungs.size() == 1u);
    CHECK(rungs[0].modes == modes);
    CHECK(rungs[0].label == "mine");

    FixedSet unlabelled(modes);
    CHECK(walk(unlabelled, context(2))[0].label == "explicit");
}

TEST_CASE("an explicit ladder returns its sets unchanged, in order")
{
    const std::vector<std::vector<Mode>> sets = {
        modes_up_to_level(3, 1), modes_up_to_level(3, 2), modes_up_to_level(3, 4)};
    ExplicitLadder ladder(sets);
    const std::vector<ModeProposal> rungs = walk(ladder, context(3));
    REQUIRE(rungs.size() == sets.size());
    for ( std::size_t i = 0; i < sets.size(); ++i )
    {
        CHECK(rungs[i].modes == sets[i]);
        CHECK(rungs[i].label
              == "set" + std::to_string(i) + "(m=" + std::to_string(sets[i].size()) + ")");
    }
}

TEST_CASE("baseline sets replay the ladder a policy would walk unaided")
{
    // The operator layer's baseline guard must not depend on any adaptive
    // trajectory. For a feedback-blind policy that means the baseline sets are
    // exactly the rungs -- which is what makes the default replay correct.
    ShellLadder shells({0, 2, 4});
    WedgeLadder wedge(8, 2);
    RadialFirstLadder radial(8, 1, 2);

    for ( const ModePolicy* policy :
          std::vector<const ModePolicy*>{&shells, &wedge, &radial} )
    {
        const ModeSearchContext ctx = context(2);
        const std::vector<ModeProposal> rungs = walk(*policy, ctx);
        const std::vector<std::vector<Mode>> baseline = policy->baseline_sets(ctx);
        REQUIRE(baseline.size() == rungs.size());
        for ( std::size_t i = 0; i < rungs.size(); ++i )
        {
            CHECK(baseline[i] == rungs[i].modes);
        }
    }
}

TEST_CASE("the proposal cap stops a policy that never would")
{
    NeverEnds policy;
    const ModeSearchContext ctx = context(2);
    CHECK(policy.baseline_sets(ctx).size()
          == static_cast<std::size_t>(kMaxModeProposals));
    CHECK(walk(policy, ctx).size() == static_cast<std::size_t>(kMaxModeProposals));
}

TEST_CASE("the counting rule's budget is what the engine will enforce")
{
    // k >= 2 (m + n_extra + P). The policies do not apply it -- the engine
    // does -- but the context computes it, and getting it wrong would silently
    // change every skip decision.
    ModeSearchContext ctx = context(2, 100, 1, 5);
    CHECK(ctx.max_modes() == 100 / 2 - 1 - 5);

    // a budget can go negative when the probes barely cover the parameters,
    // which the engine reads as "skip everything"
    ModeSearchContext starved = context(2, 4, 1, 5);
    CHECK(starved.max_modes() < 0);
}

TEST_CASE("history position ignores skipped rungs for the winner, not for the count")
{
    // A sequence policy advances on EVERY proposal, skipped ones included --
    // that is what stops the engine re-offering a set the counting rule
    // already rejected.
    ShellLadder ladder({0, 2, 4});
    ModeSearchContext ctx = context(2);

    const ModeProposal first = *ladder.propose(ctx);
    ctx.history.push_back(LevelRecord{first.label, first.modes, true, false});
    const ModeProposal second = *ladder.propose(ctx);
    CHECK(second.label == "level<=2");
    CHECK(ctx.last_fit() == nullptr);  // nothing has been fit yet

    ctx.history.push_back(LevelRecord{second.label, second.modes, false, true});
    REQUIRE(ctx.last_fit() != nullptr);
    CHECK(ctx.last_fit()->label == "level<=2");
    CHECK(ctx.last_fit()->has_winner);
}
