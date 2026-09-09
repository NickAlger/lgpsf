#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Where each row should be FITTED: a water-filling rule that moves the
/// overflow of the busiest ranks onto the ranks with room, and reports the
/// makespan it actually achieves rather than the one it aimed at.
///
/// A distributed fit gives every rank the rows whose column dof it owns, which
/// is right for the halo and wrong for the clock: fit cost per row spans two
/// orders of magnitude and the expensive rows cluster spatially, so one rank
/// can spend twenty times the mean on a rung. Fitting is the one phase that
/// does not care where it runs -- a row's fit is a pure function of the
/// package handed to it -- so the fix is to fit some rows somewhere else and
/// send the answers home. This header is the decision half of that: given a
/// predicted work per row and the rank that owns each row, which rows should
/// be fitted elsewhere and where.
///
/// It is deliberately a pure function of four arguments. No MPI, no I/O, no
/// randomness, no dependence on allocation or iteration order -- the same
/// (weights, owners, num_ranks, tolerance) gives the same plan bit for bit on
/// every rank, so every rank can compute the whole plan and none of it has to
/// be communicated. That is also why the header sits here rather than under
/// `mpi/`: the rule has no MPI dependency and is unit-tested from the serial
/// suite. The exchange that carries the packages lives in the MPI layer and
/// includes this.
///
/// **The rule.** With `w_i` the predicted work of row `i`, `m` the rank count
/// and `eps` the imbalance tolerance,
///
///     T = max( (1 + eps) * sum_i w_i / m ,  max_i w_i )
///
/// 1. *Select the overflow.* Every rank whose load exceeds `T` sheds its
///    largest rows, in decreasing `w`, until what is left is at most `T`.
///    Nothing is shed when the problem is already balanced, so a defaulted-on
///    feature is a no-op on a balanced problem.
/// 2. *Place it.* In decreasing `w`, each shed row goes to the rank with the
///    most remaining capacity below `T` -- the least loaded rank. A row may be
///    placed back on its own owner: the owner is an ordinary candidate with an
///    ordinary remaining capacity, and a row that lands back home simply never
///    travels.
///
/// Ties are broken by ascending row index among rows and by ascending rank
/// index among hosts, which is what makes the result a function of the
/// arguments alone. A rank's load is its own rows summed in ascending row
/// index -- the same number `BalancePlan::predicted_load` reports -- so which
/// ranks shed is reconstructible from the plan's own numbers rather than from
/// an internal accumulation that differs from them in the last bits.
///
/// **`T` is a target, not a guarantee, and the plan says so.** Step 1 sheds
/// WHOLE rows, so it sheds more than the overflow, and the remaining capacity
/// can be unable to absorb what was shed. Two ranks with rows
/// `{0.9, 0.9, 0.9}` and `{0.1}` at `eps = 0.1` give `T = 1.54`; the first
/// rank must shed two rows and the second can hold only one, so the achieved
/// makespan is 1.8. `max_i w_i <= T` does not rule this out and no feasibility
/// test is attempted -- `BalancePlan::predicted_makespan` is reported so the
/// caller can see the shortfall.
///
/// **What is guaranteed.** This is list scheduling onto machines with
/// pre-existing loads, so
///
///     makespan <= max( T ,  sum_i w_i / m + max_{moved} w_i )
///
/// with the second term sharp for every rank that actually receives a row:
/// a receiving rank was the least loaded when it received, and the least load
/// never exceeds the mean. The `T` in the outer max is not slack -- a rank
/// under `T` is never touched, so its load survives untouched into the
/// makespan even when it is above the mean plus the heaviest moved row.
/// (`dev/row-balance-plan.md` section 4 states the second term alone; that
/// form is false whenever an untouched rank is the bottleneck, e.g. one rank
/// holding a single heavy row while another sheds many light ones.)
///
/// **The granularity floor.** `T >= max_i w_i` means the achievable balance is
/// floored by the single most expensive row, and at large rank counts that
/// floor, not the mean, is what binds. Getting past it means splitting one
/// row's fit across ranks, which this rule does not do.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lgpsf {

/// The output of `balance_rows`: where every row is to be fitted, plus enough
/// of the arithmetic for the caller to judge the plan without redoing it.
struct BalancePlan
{
    /// (nrows) the rank that is to FIT each row, equal to the owner for a row
    /// that stays. This is the whole assignment; everything else is derived.
    std::vector<int> host;

    /// Ascending row indices of the rows step 1 selected for rehosting.
    /// A few of them may have been placed back on their own owner, so this is
    /// NOT the set that travels -- see `migrated`. It is the set the bound
    /// above takes its `max_{moved} w_i` over, and it is empty exactly when no
    /// rank exceeded `T`.
    std::vector<int> moved;

    /// Ascending row indices of the subset of `moved` with `host != owner`:
    /// the rows that actually cross a rank boundary, and therefore the rows an
    /// exchange has to pack. Empty whenever `moved` is.
    std::vector<int> migrated;

    /// (num_ranks) predicted load of each rank under `host`: the sum of the
    /// weights of the rows hosted there, accumulated in ascending row index.
    /// For a rank that neither shed nor received, this is bit for bit the
    /// load that was compared against `target` -- so `moved` is empty exactly
    /// when no rank's identity load, computed this way, exceeds `target`.
    std::vector<double> predicted_load;

    /// `T`, the target load the rule aimed at.
    double target = 0.0;

    /// `max_r predicted_load[r]`: the makespan the plan ACHIEVES. Compare it
    /// against `target` -- it can exceed it, and that is not a bug.
    double predicted_makespan = 0.0;

    /// `predicted_makespan` divided by the mean rank load `sum_i w_i / m`;
    /// 1.0 when the total weight is zero. The number a report wants.
    double predicted_imbalance = 1.0;
};

/// Decide where every row is to be fitted.
///
/// @param weights    (nrows) predicted work per row; finite and >= 0. Any
///                   monotone proxy for seconds will do -- the fit path uses
///                   the coarse point count times the previous rung's
///                   evaluation count -- because only the ORDER and the ratios
///                   of the weights enter the rule.
/// @param owners     (nrows) the rank owning each row, in `[0, num_ranks)`.
/// @param num_ranks  the number of ranks, >= 1.
/// @param tolerance  `eps` in the rule above; finite and >= 0. Zero targets a
///                   perfectly even split (still floored by the heaviest row),
///                   0.1 is a sane value, and a tolerance large enough to put
///                   `T` above every rank's load disables migration.
///
/// @returns the plan; see `BalancePlan`. With no rank over `T` the plan is the
///          identity: `host == owners`, `moved` and `migrated` empty.
///
/// @throws std::invalid_argument on `num_ranks < 1`, mismatched array lengths,
///         an owner out of range, a negative or non-finite weight, a total
///         weight that is not finite, or a negative or non-finite tolerance.
inline BalancePlan balance_rows( const std::vector<double>& weights,
                                 const std::vector<int>&    owners,
                                 int                        num_ranks,
                                 double                     tolerance )
{
    // ---- validation ------------------------------------------------------
    if ( num_ranks < 1 )
    {
        throw std::invalid_argument(
            "lgpsf::balance_rows: num_ranks must be >= 1, got "
            + std::to_string(num_ranks));
    }
    if ( owners.size() != weights.size() )
    {
        throw std::invalid_argument(
            "lgpsf::balance_rows: weights has " + std::to_string(weights.size())
            + " entries but owners has " + std::to_string(owners.size()));
    }
    if ( weights.size() > static_cast<std::size_t>(
                              std::numeric_limits<int>::max()) )
    {
        throw std::invalid_argument(
            "lgpsf::balance_rows: rows are indexed by int, so there can be at "
            "most " + std::to_string(std::numeric_limits<int>::max())
            + " of them; got " + std::to_string(weights.size()));
    }
    if ( !(tolerance >= 0.0) || !std::isfinite(tolerance) )
    {
        throw std::invalid_argument(
            "lgpsf::balance_rows: tolerance must be finite and >= 0, got "
            + std::to_string(tolerance));
    }

    const int nrows = static_cast<int>(weights.size());
    for ( int i = 0; i < nrows; ++i )
    {
        if ( !(weights[static_cast<std::size_t>(i)] >= 0.0)
             || !std::isfinite(weights[static_cast<std::size_t>(i)]) )
        {
            throw std::invalid_argument(
                "lgpsf::balance_rows: weights must be finite and >= 0; entry "
                + std::to_string(i) + " is "
                + std::to_string(weights[static_cast<std::size_t>(i)]));
        }
        const int owner = owners[static_cast<std::size_t>(i)];
        if ( owner < 0 || owner >= num_ranks )
        {
            throw std::invalid_argument(
                "lgpsf::balance_rows: owner " + std::to_string(owner)
                + " of row " + std::to_string(i) + " is out of range for "
                + std::to_string(num_ranks) + " ranks");
        }
    }

    // ---- the target ------------------------------------------------------
    // Accumulated in ascending row index so the total, and therefore T, is a
    // reproducible function of the arrays and not of any traversal here.
    double total = 0.0;
    double heaviest = 0.0;
    for ( int i = 0; i < nrows; ++i )
    {
        total += weights[static_cast<std::size_t>(i)];
        heaviest = std::max(heaviest, weights[static_cast<std::size_t>(i)]);
    }
    if ( !std::isfinite(total) )
    {
        throw std::invalid_argument(
            "lgpsf::balance_rows: the weights sum to something that is not "
            "finite, so no target load exists");
    }
    const double target =
        std::max((1.0 + tolerance) * total / num_ranks, heaviest);

    BalancePlan plan;
    plan.host = owners;
    plan.target = target;

    // The rows of each rank, ascending -- the order the tie-break needs.
    std::vector<std::vector<int>> rows_of(static_cast<std::size_t>(num_ranks));
    for ( int i = 0; i < nrows; ++i )
    {
        rows_of[static_cast<std::size_t>(owners[static_cast<std::size_t>(i)])]
            .push_back(i);
    }

    // Decreasing weight, ties by ascending row index: a stable sort of an
    // ascending list, so the tie-break is the input order rather than
    // whatever an unstable sort happens to do.
    const auto heavier_first = [&weights]( int a, int b )
    {
        return weights[static_cast<std::size_t>(a)]
               > weights[static_cast<std::size_t>(b)];
    };

    // ---- step 1: select the overflow -------------------------------------
    // The load a rank is judged against is its own rows summed in ASCENDING
    // ROW INDEX -- bit for bit the number `predicted_load` reports for a rank
    // nothing touches. That matters more than it looks: summing the same rows
    // in some other order differs in the last bits, and a rank sitting within
    // an ulp of `T` would then shed a row that the published numbers say it
    // should have kept. Tying the two together is what makes "who sheds"
    // reconstructible from the plan itself.
    //
    // The residual then comes off by subtraction, which is monotone in
    // floating point because weights are non-negative. The loop stops one row
    // short of emptying a rank. That cap is not a policy choice: `T` is at
    // least the heaviest row, so in exact arithmetic a single row is always
    // under the target and the last row is never shed anyway. It is written
    // down so that rounding cannot make it false.
    std::vector<double> load(static_cast<std::size_t>(num_ranks), 0.0);
    for ( int r = 0; r < num_ranks; ++r )
    {
        const std::vector<int>& owned = rows_of[static_cast<std::size_t>(r)];
        double residual = 0.0;
        for ( int i : owned )
        {
            residual += weights[static_cast<std::size_t>(i)];
        }

        std::vector<int> order = owned;
        std::stable_sort(order.begin(), order.end(), heavier_first);

        const std::size_t count = order.size();
        std::size_t shed = 0;
        while ( shed + 1 < count && residual > target )
        {
            residual -= weights[static_cast<std::size_t>(order[shed])];
            ++shed;
        }
        load[static_cast<std::size_t>(r)] = residual;
        for ( std::size_t j = 0; j < shed; ++j )
        {
            plan.moved.push_back(order[j]);
        }
    }
    std::sort(plan.moved.begin(), plan.moved.end());

    // ---- step 2: place it ------------------------------------------------
    // The least loaded rank is the one with the most remaining capacity below
    // T, `T` being the same for all of them; comparing loads rather than
    // capacities avoids a subtraction that could round two different loads to
    // the same capacity. `(load, rank)` in a set orders by load and then by
    // ascending rank, which is exactly the tie-break.
    {
        std::vector<int> order = plan.moved;
        std::stable_sort(order.begin(), order.end(), heavier_first);

        std::set<std::pair<double, int>> by_load;
        for ( int r = 0; r < num_ranks; ++r )
        {
            by_load.insert({load[static_cast<std::size_t>(r)], r});
        }
        for ( int i : order )
        {
            const auto lightest = by_load.begin();
            const int    host   = lightest->second;
            const double filled =
                lightest->first + weights[static_cast<std::size_t>(i)];
            by_load.erase(lightest);
            by_load.insert({filled, host});
            plan.host[static_cast<std::size_t>(i)] = host;
        }
    }

    // ---- what the plan achieves ------------------------------------------
    // Recomputed from `host` in ascending row index rather than carried out of
    // the placement loop, so the reported load is the number the caller would
    // get by summing the assignment themselves.
    plan.predicted_load.assign(static_cast<std::size_t>(num_ranks), 0.0);
    for ( int i = 0; i < nrows; ++i )
    {
        plan.predicted_load[static_cast<std::size_t>(
            plan.host[static_cast<std::size_t>(i)])] +=
            weights[static_cast<std::size_t>(i)];
    }
    plan.predicted_makespan = *std::max_element(plan.predicted_load.begin(),
                                                plan.predicted_load.end());

    const double mean = total / num_ranks;
    plan.predicted_imbalance =
        ( mean > 0.0 ) ? plan.predicted_makespan / mean : 1.0;

    for ( int i : plan.moved )
    {
        if ( plan.host[static_cast<std::size_t>(i)]
             != owners[static_cast<std::size_t>(i)] )
        {
            plan.migrated.push_back(i);
        }
    }

    return plan;
}

} // end namespace lgpsf
