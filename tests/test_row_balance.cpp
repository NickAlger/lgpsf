// SPDX-License-Identifier: MIT
//
// Checks on the row-balancing rule: the structural invariants every plan must
// satisfy, the no-op on a balanced problem, the two places where the target
// `T` is NOT achieved (both of them regressions against properties an earlier
// draft of dev/row-balance-plan.md asserted and that are false), the honest
// makespan bound on randomized instances, purity, and the edge cases.
//
// All self-contained: nothing here is compared against a stored reference, so
// the suite cannot drift out of step with the code it tests.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/row_balance.hpp"

using lgpsf::BalancePlan;
using lgpsf::balance_rows;

namespace {

struct Instance
{
    std::vector<double> weights;
    std::vector<int>    owners;
    int                 num_ranks = 1;
    double              tolerance = 0.0;
};

/// The total, accumulated in ascending row index -- the same order the rule
/// uses, so `T` can be predicted here to the last bit.
double total_weight( const std::vector<double>& weights )
{
    double total = 0.0;
    for ( double w : weights )
    {
        total += w;
    }
    return total;
}

double heaviest_row( const std::vector<double>& weights )
{
    double heaviest = 0.0;
    for ( double w : weights )
    {
        heaviest = std::max(heaviest, w);
    }
    return heaviest;
}

/// The load of every rank under an assignment, ascending row index.
std::vector<double> loads_under( const std::vector<double>& weights,
                                 const std::vector<int>&    host,
                                 int                        num_ranks )
{
    std::vector<double> loads(static_cast<std::size_t>(num_ranks), 0.0);
    for ( std::size_t i = 0; i < weights.size(); ++i )
    {
        loads[static_cast<std::size_t>(host[i])] += weights[i];
    }
    return loads;
}

/// The heaviest row step 1 selected, zero when it selected nothing. This --
/// not the heaviest row that CHANGES rank -- is the quantity the makespan
/// bound is stated over: a selected row placed back on its own owner was still
/// placed on the least loaded rank and still counts against that rank's load.
double heaviest_moved( const Instance& in, const BalancePlan& plan )
{
    double heaviest = 0.0;
    for ( int i : plan.moved )
    {
        heaviest = std::max(heaviest, in.weights[static_cast<std::size_t>(i)]);
    }
    return heaviest;
}

/// Everything a plan must satisfy on ANY instance: a well-formed assignment,
/// self-consistent reported numbers, and the two invariants the rule's own
/// construction guarantees (nothing is selected on a rank that was under `T`,
/// and no rank is emptied by shedding).
void check_structure( const Instance& in, const BalancePlan& plan )
{
    const std::size_t nrows = in.weights.size();
    const std::size_t nranks = static_cast<std::size_t>(in.num_ranks);

    REQUIRE(plan.host.size() == nrows);
    REQUIRE(plan.predicted_load.size() == nranks);

    // The assignment lands somewhere legal, and only a selected row may have
    // been reassigned at all.
    std::vector<bool> selected(nrows, false);
    for ( int i : plan.moved )
    {
        REQUIRE(i >= 0);
        REQUIRE(static_cast<std::size_t>(i) < nrows);
        selected[static_cast<std::size_t>(i)] = true;
    }
    for ( std::size_t i = 0; i < nrows; ++i )
    {
        CHECK(plan.host[i] >= 0);
        CHECK(plan.host[i] < in.num_ranks);
        if ( !selected[i] )
        {
            CHECK(plan.host[i] == in.owners[i]);
        }
    }

    // `moved` is ascending and duplicate-free; `migrated` is exactly its
    // host-changing subset, in the same order.
    CHECK(std::is_sorted(plan.moved.begin(), plan.moved.end()));
    CHECK(std::adjacent_find(plan.moved.begin(), plan.moved.end())
          == plan.moved.end());
    std::vector<int> expect_migrated;
    for ( int i : plan.moved )
    {
        if ( plan.host[static_cast<std::size_t>(i)]
             != in.owners[static_cast<std::size_t>(i)] )
        {
            expect_migrated.push_back(i);
        }
    }
    CHECK(plan.migrated == expect_migrated);

    // The reported arithmetic is the arithmetic.
    const double total = total_weight(in.weights);
    const double target =
        std::max((1.0 + in.tolerance) * total / in.num_ranks,
                 heaviest_row(in.weights));
    CHECK(plan.target == target);
    CHECK(plan.predicted_load == loads_under(in.weights, plan.host, in.num_ranks));
    CHECK(plan.predicted_makespan
          == *std::max_element(plan.predicted_load.begin(),
                               plan.predicted_load.end()));
    const double mean = total / in.num_ranks;
    if ( mean > 0.0 )
    {
        CHECK(plan.predicted_imbalance == plan.predicted_makespan / mean);
    }
    else
    {
        CHECK(plan.predicted_imbalance == 1.0);
    }

    // Nothing is shed from a rank whose identity load -- computed exactly as
    // the rule computes it, in ascending row index -- is at or under the
    // target, and no rank is emptied. The first of these is an EXACT
    // comparison on purpose: the rule judges a rank by the same number it
    // reports, so a rank within an ulp of `T` cannot shed a row that the
    // published loads say it should have kept.
    const std::vector<double> identity =
        loads_under(in.weights, in.owners, in.num_ranks);
    std::vector<int> owned(nranks, 0);
    std::vector<int> shed(nranks, 0);
    for ( std::size_t i = 0; i < nrows; ++i )
    {
        ++owned[static_cast<std::size_t>(in.owners[i])];
        if ( selected[i] )
        {
            ++shed[static_cast<std::size_t>(in.owners[i])];
        }
    }
    for ( std::size_t r = 0; r < nranks; ++r )
    {
        if ( !(identity[r] > plan.target) )
        {
            CHECK(shed[r] == 0);
        }
        if ( owned[r] > 0 )
        {
            CHECK(shed[r] < owned[r]);
        }
    }
}

/// Byte-for-byte equality of two plans, doubles included.
bool identical( const BalancePlan& a, const BalancePlan& b )
{
    if ( a.host != b.host || a.moved != b.moved || a.migrated != b.migrated )
    {
        return false;
    }
    if ( a.predicted_load.size() != b.predicted_load.size() )
    {
        return false;
    }
    if ( !a.predicted_load.empty()
         && std::memcmp(a.predicted_load.data(), b.predicted_load.data(),
                        a.predicted_load.size() * sizeof(double)) != 0 )
    {
        return false;
    }
    const double lhs[3] = {a.target, a.predicted_makespan, a.predicted_imbalance};
    const double rhs[3] = {b.target, b.predicted_makespan, b.predicted_imbalance};
    return std::memcmp(lhs, rhs, sizeof(lhs)) == 0;
}

/// A spread of instances: rank counts from 1 to 64, four weight laws (the
/// heavy-tailed one is the case the scheme exists for), spatially clustered
/// ownership as well as uniform, and tolerances from 0 up to values large
/// enough to switch migration off.
std::vector<Instance> random_instances( unsigned seed, int how_many )
{
    std::mt19937 gen(seed);
    const std::vector<int>    rank_counts = {1, 2, 3, 5, 8, 16, 64};
    const std::vector<double> tolerances  = {0.0, 0.01, 0.1, 0.5, 3.0};

    std::vector<Instance> out;
    for ( int t = 0; t < how_many; ++t )
    {
        Instance in;
        in.num_ranks = rank_counts[gen() % rank_counts.size()];
        in.tolerance = tolerances[gen() % tolerances.size()];
        const int nrows = 1 + static_cast<int>(gen() % 400);

        const int law = static_cast<int>(gen() % 4);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        for ( int i = 0; i < nrows; ++i )
        {
            double w = 0.0;
            switch ( law )
            {
                case 0:  // uniform
                    w = unit(gen);
                    break;
                case 1:  // all equal
                    w = 2.5;
                    break;
                case 2:  // heavy-tailed: log-normal, three decades of spread
                    w = std::exp(std::normal_distribution<double>(0.0, 3.5)(gen));
                    break;
                default:  // mostly zero, a few spikes
                    w = ( unit(gen) < 0.1 ) ? 100.0 * unit(gen) : 0.0;
                    break;
            }
            in.weights.push_back(w);
        }

        // Half the instances get clustered ownership -- contiguous blocks of
        // rows to one rank, which is what makes an expensive REGION overload
        // one rank rather than spreading the tail evenly.
        if ( gen() % 2 == 0 )
        {
            std::uniform_int_distribution<int> pick(0, in.num_ranks - 1);
            for ( int i = 0; i < nrows; ++i )
            {
                in.owners.push_back(pick(gen));
            }
        }
        else
        {
            const int block = 1 + nrows / in.num_ranks;
            for ( int i = 0; i < nrows; ++i )
            {
                in.owners.push_back(std::min(i / block, in.num_ranks - 1));
            }
        }
        out.push_back(std::move(in));
    }
    return out;
}

} // end namespace

TEST_CASE( "a balanced problem is left completely alone" )
{
    // Every rank under T: the scheme must be a no-op, which is the property
    // that makes it safe to default on.
    std::vector<Instance> cases;

    Instance even;                             // a perfect split already
    even.weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    even.owners  = {0, 0, 1, 1, 2, 2};
    even.num_ranks = 3;
    even.tolerance = 0.1;
    cases.push_back(even);

    Instance lumpy;                            // uneven, but inside tolerance
    lumpy.weights = {3.0, 2.0, 2.5, 2.5, 2.0, 3.0};
    lumpy.owners  = {0, 0, 1, 1, 2, 2};
    lumpy.num_ranks = 3;
    lumpy.tolerance = 0.2;
    cases.push_back(lumpy);

    Instance slack;                            // wildly uneven, huge tolerance
    slack.weights = {10.0, 1.0, 1.0, 1.0};
    slack.owners  = {0, 0, 1, 2};
    slack.num_ranks = 3;
    slack.tolerance = 5.0;
    cases.push_back(slack);

    for ( const Instance& in : cases )
    {
        const BalancePlan plan =
            balance_rows(in.weights, in.owners, in.num_ranks, in.tolerance);
        check_structure(in, plan);
        CHECK(plan.moved.empty());
        CHECK(plan.migrated.empty());
        CHECK(plan.host == in.owners);
        CHECK(plan.predicted_load
              == loads_under(in.weights, in.owners, in.num_ranks));
        CHECK(plan.predicted_makespan <= plan.target);
    }
}

TEST_CASE( "the target is not always reachable, and the plan reports that" )
{
    // The counterexample from dev/row-balance-plan.md section 4. Step 1 sheds
    // WHOLE rows, so it sheds more than the overflow, and what is left over
    // there cannot absorb it. An earlier draft asserted feasibility; this is
    // the regression against that.
    Instance in;
    in.weights   = {0.9, 0.9, 0.9, 0.1};
    in.owners    = {0, 0, 0, 1};
    in.num_ranks = 2;
    in.tolerance = 0.1;

    const BalancePlan plan =
        balance_rows(in.weights, in.owners, in.num_ranks, in.tolerance);
    check_structure(in, plan);

    // T = max(1.1 * 2.8 / 2, 0.9) = 1.54.
    CHECK(plan.target == doctest::Approx(1.54));

    // Rank 0 is at 2.7 and has to get to 1.54, so it sheds two of its three
    // 0.9s -- the lowest indices, the weights being tied.
    CHECK(plan.moved == std::vector<int>{0, 1});

    // Row 0 goes to rank 1 (load 0.1, the emptiest); row 1 then finds rank 0
    // at 0.9 against rank 1's 1.0 and goes back home, which is allowed.
    CHECK(plan.host == std::vector<int>{1, 0, 0, 1});
    CHECK(plan.migrated == std::vector<int>{0});

    // The point of the test: 1.8 > 1.54, reported rather than claimed away.
    CHECK(plan.predicted_makespan == doctest::Approx(1.8));
    CHECK(plan.predicted_makespan > plan.target);
    CHECK(plan.predicted_load[0] == doctest::Approx(1.8));
    CHECK(plan.predicted_load[1] == doctest::Approx(1.0));
}

TEST_CASE( "a shed row may be placed back on its own owner" )
{
    // Rank 0 sheds its 10 to get under T = 10.45, then turns out to be the
    // emptiest rank and takes it straight back. So `moved` is non-empty while
    // `migrated` is empty and nothing travels -- the distinction the exchange
    // in the MPI layer sizes itself from.
    Instance in;
    in.weights   = {10.0, 1.0, 8.0};
    in.owners    = {0, 0, 1};
    in.num_ranks = 2;
    in.tolerance = 0.1;

    const BalancePlan plan =
        balance_rows(in.weights, in.owners, in.num_ranks, in.tolerance);
    check_structure(in, plan);

    CHECK(plan.target == doctest::Approx(10.45));
    CHECK(plan.moved == std::vector<int>{0});
    CHECK(plan.migrated.empty());
    CHECK(plan.host == in.owners);
    CHECK(plan.predicted_makespan == doctest::Approx(11.0));
    CHECK(plan.predicted_makespan > plan.target);
}

TEST_CASE( "the makespan bound is max(T, mean + heaviest moved), not the second term alone" )
{
    // dev/row-balance-plan.md section 4 states the list-scheduling bound
    // `makespan <= sum w / m + max_{moved} w` unconditionally. It is false,
    // for the same reason the feasibility claim was: a rank UNDER the target
    // is never touched, so its load walks into the makespan whatever the
    // moved rows weigh. Here rank 0 holds one row of 10 and never moves,
    // while rank 1 sheds light rows onto an empty rank 2.
    Instance in;
    in.num_ranks = 3;
    in.tolerance = 0.1;
    in.weights.push_back(10.0);
    in.owners.push_back(0);
    for ( int i = 0; i < 200; ++i )
    {
        in.weights.push_back(0.06);
        in.owners.push_back(1);
    }

    const BalancePlan plan =
        balance_rows(in.weights, in.owners, in.num_ranks, in.tolerance);
    check_structure(in, plan);

    const double total = total_weight(in.weights);
    const double mean  = total / in.num_ranks;
    CHECK(plan.target == doctest::Approx(10.0));       // pinned by the heavy row
    CHECK(heaviest_moved(in, plan) == doctest::Approx(0.06));
    CHECK(plan.predicted_makespan == doctest::Approx(10.0));

    // The plan's stated bound, violated.
    CHECK(plan.predicted_makespan > mean + heaviest_moved(in, plan));
    // The bound that actually holds.
    CHECK(plan.predicted_makespan
          <= std::max(plan.target, mean + heaviest_moved(in, plan)));

    // Rank 1 did get down to the target, and rank 2 absorbed the overflow.
    CHECK(plan.predicted_load[1] <= plan.target);
    CHECK(plan.predicted_load[2] == doctest::Approx(2.04));
}

TEST_CASE( "the makespan bound holds on randomized instances" )
{
    for ( const Instance& in : random_instances(20260908u, 240) )
    {
        const BalancePlan plan =
            balance_rows(in.weights, in.owners, in.num_ranks, in.tolerance);
        check_structure(in, plan);

        const double total = total_weight(in.weights);
        const double mean  = total / in.num_ranks;
        const double moved = heaviest_moved(in, plan);
        // Room for the last bits of a few hundred accumulated additions.
        const double slack = 1e-9 * std::max(1.0, total);

        CHECK(plan.predicted_makespan <= std::max(plan.target, mean + moved) + slack);

        // The sharp half of the bound: a rank that RECEIVED a row was the
        // least loaded when it did, and the least load never exceeds the
        // mean, so it ends at most one moved row above the mean.
        std::vector<bool> received(static_cast<std::size_t>(in.num_ranks), false);
        for ( int i : plan.moved )
        {
            received[static_cast<std::size_t>(
                plan.host[static_cast<std::size_t>(i)])] = true;
        }
        for ( std::size_t r = 0; r < received.size(); ++r )
        {
            if ( received[r] )
            {
                CHECK(plan.predicted_load[r] <= mean + moved + slack);
            }
        }

        // Migration happens exactly when some rank is over the target.
        const std::vector<double> identity =
            loads_under(in.weights, in.owners, in.num_ranks);
        const bool overloaded =
            std::any_of(identity.begin(), identity.end(),
                        [&plan]( double load ) { return load > plan.target; });
        CHECK(plan.moved.empty() == !overloaded);

        // Balancing never makes the prediction worse than doing nothing.
        // OBSERVED, not proved: it holds over these instances and over a
        // quarter of a million randomized ones checked against an independent
        // reference, but the only proof to hand covers the case where the
        // shedding rank has itself received nothing yet.
        const double identity_makespan =
            *std::max_element(identity.begin(), identity.end());
        CHECK(plan.predicted_makespan <= identity_makespan + slack);
    }
}

TEST_CASE( "the plan is a pure function of its four arguments" )
{
    for ( const Instance& in : random_instances(11u, 40) )
    {
        const BalancePlan first =
            balance_rows(in.weights, in.owners, in.num_ranks, in.tolerance);

        // Same call again, same vectors.
        CHECK(identical(first, balance_rows(in.weights, in.owners,
                                            in.num_ranks, in.tolerance)));

        // Disturb the heap and the allocator's free lists, and run unrelated
        // instances in between, then rebuild the inputs at fresh addresses by
        // a different construction path.
        std::mt19937 gen(99u);
        for ( int t = 0; t < 4; ++t )
        {
            std::vector<double> junk(1 + gen() % 5000, 1.0);
            junk.push_back(static_cast<double>(gen() % 7));
            CHECK(junk.size() > 0);
        }
        for ( const Instance& other : random_instances(7u, 3) )
        {
            const BalancePlan ignored = balance_rows(
                other.weights, other.owners, other.num_ranks, other.tolerance);
            CHECK(ignored.host.size() == other.weights.size());
        }

        std::vector<double> weights_again(in.weights.size());
        std::vector<int>    owners_again(in.owners.size());
        for ( std::size_t i = in.weights.size(); i-- > 0; )
        {
            weights_again[i] = in.weights[i];
            owners_again[i]  = in.owners[i];
        }
        CHECK(identical(first, balance_rows(weights_again, owners_again,
                                            in.num_ranks, in.tolerance)));
    }
}

TEST_CASE( "ties go to the lowest row index and the lowest rank index" )
{
    SUBCASE( "equal weights: the lowest-indexed rows are the ones shed" )
    {
        // Rank 0 holds three unit rows and must get to T = 2.2, so exactly one
        // goes, and it is row 0.
        const std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};
        const std::vector<int>    owners  = {0, 0, 0, 1};
        const BalancePlan plan = balance_rows(weights, owners, 2, 0.1);

        CHECK(plan.target == doctest::Approx(2.2));
        CHECK(plan.moved == std::vector<int>{0});
        CHECK(plan.migrated == std::vector<int>{0});
        CHECK(plan.host == std::vector<int>{1, 0, 0, 1});
        CHECK(plan.predicted_makespan == doctest::Approx(2.0));
    }

    SUBCASE( "equal capacity: the lowest-indexed rank takes the row" )
    {
        // Ranks 1 and 2 are both empty; row 0 goes to rank 1.
        const std::vector<double> weights = {2.0, 2.0};
        const std::vector<int>    owners  = {0, 0};
        const BalancePlan plan = balance_rows(weights, owners, 3, 0.1);

        CHECK(plan.target == doctest::Approx(2.0));
        CHECK(plan.moved == std::vector<int>{0});
        CHECK(plan.host == std::vector<int>{1, 0});
    }
}

TEST_CASE( "the edge cases" )
{
    SUBCASE( "one rank has nowhere to send anything" )
    {
        const std::vector<double> weights = {3.0, 1.0, 4.0, 1.0, 5.0};
        const std::vector<int>    owners  = {0, 0, 0, 0, 0};
        for ( double tolerance : {0.0, 0.1, 10.0} )
        {
            const Instance in{weights, owners, 1, tolerance};
            const BalancePlan plan = balance_rows(weights, owners, 1, tolerance);
            check_structure(in, plan);
            CHECK(plan.moved.empty());
            CHECK(plan.migrated.empty());
            CHECK(plan.host == owners);
            CHECK(plan.predicted_makespan == doctest::Approx(14.0));
            CHECK(plan.predicted_imbalance == doctest::Approx(1.0));
        }
    }

    SUBCASE( "a rank that owns no rows is an ordinary host" )
    {
        const std::vector<double> weights = {5.0, 5.0, 1.0};
        const std::vector<int>    owners  = {0, 0, 1};
        const Instance in{weights, owners, 3, 0.1};
        const BalancePlan plan = balance_rows(weights, owners, 3, 0.1);
        check_structure(in, plan);

        CHECK(plan.target == doctest::Approx(5.0));
        CHECK(plan.moved == std::vector<int>{0});
        CHECK(plan.host == std::vector<int>{2, 0, 1});
        CHECK(plan.predicted_makespan == doctest::Approx(5.0));
        CHECK(plan.predicted_makespan <= plan.target);
    }

    SUBCASE( "a single row heavier than the mean pins the target" )
    {
        const std::vector<double> weights = {100.0, 1.0, 1.0, 1.0};
        const std::vector<int>    owners  = {0, 1, 2, 3};
        const Instance in{weights, owners, 4, 0.1};
        const BalancePlan plan = balance_rows(weights, owners, 4, 0.1);
        check_structure(in, plan);

        // (1.1 * 103 / 4) = 28.3 loses to the heavy row.
        CHECK(plan.target == doctest::Approx(100.0));
        CHECK(plan.moved.empty());
        CHECK(plan.host == owners);
        // The granularity floor: no rule that moves whole rows can beat 100.
        CHECK(plan.predicted_makespan == doctest::Approx(100.0));
    }

    SUBCASE( "all weights equal" )
    {
        std::vector<double> weights(12, 4.0);
        std::vector<int>    owners(12, 0);
        for ( int i = 0; i < 12; ++i )
        {
            owners[static_cast<std::size_t>(i)] = ( i < 8 ) ? 0 : 1;
        }
        const Instance in{weights, owners, 4, 0.0};
        const BalancePlan plan = balance_rows(weights, owners, 4, 0.0);
        check_structure(in, plan);

        CHECK(plan.target == doctest::Approx(12.0));
        CHECK(plan.predicted_makespan == doctest::Approx(12.0));
        CHECK(plan.predicted_imbalance == doctest::Approx(1.0));
        for ( int r = 0; r < 4; ++r )
        {
            CHECK(plan.predicted_load[static_cast<std::size_t>(r)]
                  == doctest::Approx(12.0));
        }
    }

    SUBCASE( "all weights zero" )
    {
        const std::vector<double> weights(9, 0.0);
        const std::vector<int>    owners  = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        const Instance in{weights, owners, 3, 0.1};
        const BalancePlan plan = balance_rows(weights, owners, 3, 0.1);
        check_structure(in, plan);

        CHECK(plan.target == 0.0);
        CHECK(plan.moved.empty());
        CHECK(plan.host == owners);
        CHECK(plan.predicted_makespan == 0.0);
        // Nothing to balance is perfect balance, not a division by zero.
        CHECK(plan.predicted_imbalance == 1.0);
    }

    SUBCASE( "no rows at all" )
    {
        const std::vector<double> weights;
        const std::vector<int>    owners;
        const Instance in{weights, owners, 4, 0.1};
        const BalancePlan plan = balance_rows(weights, owners, 4, 0.1);
        check_structure(in, plan);

        CHECK(plan.host.empty());
        CHECK(plan.moved.empty());
        CHECK(plan.migrated.empty());
        CHECK(plan.predicted_load == std::vector<double>(4, 0.0));
        CHECK(plan.target == 0.0);
        CHECK(plan.predicted_makespan == 0.0);
        CHECK(plan.predicted_imbalance == 1.0);
    }

    SUBCASE( "tolerance zero targets the mean, still floored by the heaviest row" )
    {
        const std::vector<double> weights = {1.0, 1.0, 1.0, 1.0};
        const std::vector<int>    owners  = {0, 0, 0, 0};
        const Instance in{weights, owners, 2, 0.0};
        const BalancePlan plan = balance_rows(weights, owners, 2, 0.0);
        check_structure(in, plan);

        CHECK(plan.target == doctest::Approx(2.0));
        CHECK(plan.moved == std::vector<int>{0, 1});
        CHECK(plan.host == std::vector<int>{1, 1, 0, 0});
        CHECK(plan.predicted_makespan == doctest::Approx(2.0));

        // The floor: with one dominant row, tolerance 0 cannot get below it.
        const std::vector<double> lopsided = {7.0, 1.0, 1.0};
        const BalancePlan floored = balance_rows(lopsided, {0, 0, 0}, 3, 0.0);
        CHECK(floored.target == doctest::Approx(7.0));
        CHECK(floored.predicted_makespan == doctest::Approx(7.0));
    }
}

TEST_CASE( "invalid input is rejected by name" )
{
    const std::vector<double> weights = {1.0, 2.0};
    const std::vector<int>    owners  = {0, 1};

    CHECK_THROWS_AS(balance_rows(weights, owners, 0, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(balance_rows(weights, owners, -3, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(balance_rows(weights, {0}, 2, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(balance_rows(weights, {0, 2}, 2, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(balance_rows(weights, {0, -1}, 2, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(balance_rows({1.0, -1.0}, owners, 2, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(
        balance_rows({1.0, std::numeric_limits<double>::quiet_NaN()}, owners, 2, 0.1),
        std::invalid_argument);
    CHECK_THROWS_AS(
        balance_rows({1.0, std::numeric_limits<double>::infinity()}, owners, 2, 0.1),
        std::invalid_argument);
    CHECK_THROWS_AS(balance_rows(weights, owners, 2, -0.1), std::invalid_argument);
    CHECK_THROWS_AS(
        balance_rows(weights, owners, 2, std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);

    // The message names the function, so a throw from deep in a fit is
    // traceable without a debugger.
    try
    {
        balance_rows(weights, owners, 0, 0.1);
        CHECK(false);
    }
    catch ( const std::invalid_argument& err )
    {
        CHECK(std::string(err.what()).find("lgpsf::balance_rows")
              != std::string::npos);
    }
}
