# Load-balanced row assignment for the distributed fit

Plan for removing the wall-clock imbalance of `dist_fit` when a few rows are
much more expensive than the rest. Drafted 2026-09-08 from a joint session;
the architecture was settled in discussion and the code facts below were
established by a full read of the fit path, with file:line references so a
fresh session can implement from this document alone. Nothing here is
implemented yet.

**One-paragraph version.** Every rank fits exactly the rows it owns. On the
maintainer's continental ice-sheet Hessian the expensive rows cluster
spatially, so one rank of 192 spends 482 s on a rung whose mean is 22 s, and
the fit is the dominant term in the build. The fix is a *fitting-only*
redistribution: the row's coarsened quadrature is packaged on its owner, the
LM search runs wherever there is capacity, and the candidate parameters come
back to the owner, which re-scores them on the true window, applies the guard
and assembles. The halo is untouched, the deployed operator is untouched, and
`dist_fit`'s input and output contract is unchanged, so the redistribution is
invisible above this library. Rows to move and where they go come from one
water-filling rule with a single knob, an imbalance tolerance, which
degenerates to no migration when the problem is already balanced. How big the
win is turns almost entirely on how much of a row's time stays with its owner
(the re-score and the assembly), which is today an estimate rather than a
measurement: see slice 0.

---

## 1. The measurement

From one build's fit dump on the continental problem, 409,545 rows over 192
ranks, last rung (30 probes), with the per-row `row_seconds` telemetry
(`operator_fit.hpp:303-309`) as ground truth. The table is the per-rank
max/mean of *actual* fit seconds under the water-filling rule of section 4
(an earlier version of this table simulated a different algorithm -- "move the
top X% globally" -- and charged the whole of a row's time to the move; both
were wrong).

| resident share `r` | rows moved | resulting max/mean |
|---|---|---|
| 0 (nothing stays home) | 13,490 | 1.9 |
| 0.10 | 13,490 | 4.0 |
| 0.15 | 13,490 | 5.0 |
| 0.25 | 13,490 | 7.0 |
| 0.40 | 13,490 | 10.1 |

Weights are `fit_points x` the previous rung's `evaluations`, tolerance 0.1,
and `r` is the share of a migrated row's seconds that stays with its owner
because phases A and C do not move (section 2). With `fit_points` alone the
rule moves more rows (17,150) and lands at 4.1 at `r = 0.15`.

**MEASURED 2026-09-08: `r = 0.084`, so the outcome is 3.6.** Slice 0 is done.
Real continental windows were replayed on a laptop with the phase timers on
(the replayed coarse point count matches the recorded one exactly for every
row, which is the check that the geometry is faithful); the search alone was
rescaled to the recorded evaluation count, since the synthetic target
converges in about 6% of them while the coarsening, the re-score and the
gather are value-independent and measured directly. Over the rows that would
migrate, `r` is 0.05 / 0.09 / 0.14 at p10 / p50 / p90 and 0.084 weighted by
seconds. The bias is conservative: if the field reached larger mode sets than
the replay, `r` is overstated and the payoff is bigger. So the fit's per-rank
max/mean goes 22.2 -> 3.6, a rebuild goes from 1,938 s to about 1,110
(Gauss-Newton) and from 3,554 s to about 1,400 (full Hessian). Method and
numbers: the maintainer's research repo, `row_balance/`.

**The result is about `1 + 22 r`, so the entire payoff is the residual.** Once
the search is balanced, what is left sits on the same ranks that were
overloaded, and it alone sets the makespan. The structural estimate is
`r ~ 0.12` (the full-window re-score is about 9% of a wide row's fit, assembly
about 3%), which puts the outcome near 4 and a full-Hessian rebuild at roughly
1,600 s instead of 3,554. But `r` is an ESTIMATE, and it is worse for exactly
the rows that move: phases A and C scale with the full window while phase B
scales with the coarse cells, so a heavily coarsened row has a larger `r` than
the average. Measuring `r` was slice 0; it is done, and the number above is what the rest
of this document is sized by.

Three facts drive the design.

1. **The tail is thin.** The top 1% of rows hold 21% of the fit points, the
   top 2% hold 32%, the top 5% hold 49%. A few percent of the rows carry a
   third of the work.
2. **The weight is known before phase A, and must be.** REVISED for slice 3.
   Assigning after phase A would mean every row's package exists at once,
   which on the busiest rank is gigabytes -- phase A is what BUILDS the
   package. And phase A cannot run twice for free: the coarsening is about 4%
   of a wide row's time, so a throwaway counting pass would push the resident
   share from 0.084 to about 0.12 and cost a third of the payoff. So the
   assignment is computed right after the window pre-pass, which already runs
   for every row in one dual-tree descent before the fit loop
   (`operator_fit.hpp:542-643`), from quantities that are exact and free
   there:
   - **rung 1**: the WINDOW size, `outcome.window.size()`. Exact, already
     computed, no coarsening needed.
   - **rungs 2 and up**: the previous rung's `fit_points x evaluations` for
     the same row. The cell structure depends only on geometry and
     `coarsen_eps`, so `fit_points` is IDENTICAL across the rungs of one
     build -- after rung 1 the coarse count is known exactly, for free, and
     only the evaluation count is a prediction.
   The local proxy (window ellipsoid area times node density) is not needed
   and is dropped; it was an estimate OF the window size, which we simply
   have.
3. **History supplies the other half.** Points alone correlate 0.62 in log
   with row seconds; `fit_points x evaluations` correlates 0.94. `dist_fit` is
   called once per rung and the halo is not rebuilt for this scheme, so the
   assignment is recomputed at every rung from the previous rung's exact
   evaluation counts. Rung 1 has points only; rungs 2 and up have both.

For scale: on that problem the fit is 976 s of a 1938 s Gauss-Newton build and
2504 s of a 3554 s full-Hessian build, so this is the dominant cost of a
rebuild.

## 2. The architecture: D1 and D2

Call the existing spatially localized distribution **D1** (a rank owns the
rows whose column dof it owns) and the rebalanced fitting distribution **D2**.
The redistribution is a performance hack that puts rows where they do not
belong, so it is isolated to the one phase that needs it:

```
D1: resolve windows -> coarsen -> package the fit problem
    |
    | migrate (only the rows the rule selects)
    v
D2: the mode-set ladder and the LM search -> candidate parameters
    |
    | return (a few dozen doubles per row)
    v
D1: re-score the candidates on the FULL window -> guard -> select
    -> assemble_sparse -> everything downstream, unchanged
```

**Assembly on D1 is forced, not chosen.** The deployed operator's support is
the full window, never the coarse one (`operator_fit.hpp:717-726`, and the
coarsening note in `docs/`). D2 never receives the full window, so D2 cannot
assemble. The same argument puts the full-window re-score on D1. The two
questions have one answer.

**Consequences, all good:**

- `halo_plan` is called with exactly the same argument as today
  (`halo_exchange.hpp:88-92`, the caller's own rows), so halo volume, halo
  memory and the plan's collectives are untouched. An earlier design that
  shipped raw windows to foreign ranks would have grown each rank's halo
  toward the whole mesh; that problem does not arise here.
- `dist_fit`'s `DistFitInput` and `DistFitResult` (`mpi/dist_fit.hpp:67-100`)
  are unchanged in meaning: `B_local` still comes back in local-row by
  combined-column indexing. The three downstream expressions that assume a
  rank holds exactly its own rows (`mpi/dist_wsym.hpp:70`, and in the
  consumer the QC residual and the CSR row counts) keep working untouched,
  because the rank does still hold exactly its own rows.
- The migration payload is the coarse cells, not the window -- PROVIDED the
  row was coarsened. `coarsen_above` defaults to 3,000 as of 2026-09-08, so
  this holds for the wide rows by default; but a row below the trigger has
  `x_fit == x_window` (`:755-757`) and migrating it ships the raw window. The
  byte cap in section 9 is what makes that safe rather than a footnote. At the
  last rung the moved rows are about 1.9e7 cells; at 3 + 30 doubles per cell
  that is about 5 GB across the whole job. Note that is 26 MB per rank ON
  AVERAGE and the senders are few by construction (see section 9).

**The cost of the clean seam.** The re-score stays on the overloaded rank. For
a wide row the search runs a few hundred evaluations over a few thousand
cells while the re-score is two evaluations over the full window, so it is
around 9% of the row's fit; assembly adds about 3% more (it is one pass over a
deployed support that is roughly 70% of the window, against the re-score's two
passes plus a probe contraction). Section 1 shows that this residual, not the
search, is what sets the achievable makespan. Worth it for the isolation -- and
NOT optional: the re-score is the only thing that scores the finalists on
points the fit never saw, so it is the honesty check on a coarsening the fit
was optimized against. That is why the coarse score could be found to be
optimistic above eps 0.15 at all.

## 3. Where the seams already are

The row body is already delimited almost exactly where the phases need to cut.

**The window pre-pass is already separate.** `fit_operator` resolves every
row's window in ONE dual-tree descent before the fit loop
(`operator_fit.hpp:542-643`), storing per row: `outcome.window` (sorted
combined column indices, `:629-641`), `window_center` / `window_covariance`
(`:614-622`), `window_frame` when coarsening is armed (`:604-613`), `prior`
(the Cholesky factor of sigma, `:579`) and `attempt`. **These persist for the
whole call**, so the re-score in phase C can rebuild the full-window arrays
with no new storage and no second descent. (An earlier concern that the split
would force either hundreds of megabytes of window indices or a re-descent
was unfounded: the indices are already kept.)

**The fit loop's body** (`operator_fit.hpp:645-980`, per row `rho`) then runs:

| lines | what | phase |
|---|---|---|
| 660-694 | center, covariance, `prior_L`, the window, `spike_position` by binary search into the sorted window | A |
| 696-708 | gather `x_window`, `m2_window`, `z` from the combined columns; `y`, `target_mass` | A |
| 716-757 | the coarsening decision and `coarsen_window`; `x_fit` / `m2_fit` / `z_fit` / `spike_fit`; `outcome.fit_points` | A |
| 765-775 | `coarse_config` (the released-centre resolution rule) | A |
| 777-857 | whitening, the pinned baseline at the prior, the mode-set ladder, `fit_from_probes`, and then `:847-857` reading `evaluations_total` / `candidates_tried` / `max_modes` off the result | **B** |
| 859-980 | the full-window re-score of the finalists, the guard, selection into `outcome` | C |

Phase B ends at `:857`, not at the `fit_from_probes` call: `:847-857` harvest
the search's own counters, and those become `FitDiagnostics::evaluations`,
`candidates` and `work` (`:1035-1039`), `DistFitResult`'s totals
(`dist_fit.hpp:203-210`), the consumer's report and dump columns 28/29 -- and,
circularly, section 4's preferred weight for the NEXT rung. They must cross
back with the finalists. Nothing would catch it if they did not: the MPI gate
compares dense row images, not diagnostics.

The comment at `:717-726` already states the invariant the split relies on:
"From here to the guard everything reads `x_fit` / `m2_fit` / `z_fit` /
`spike_fit`. The window itself is the deployed support and is never touched."
Phase B is exactly the region that comment describes.

**What crosses A to B** (all dense, small, trivially serializable): `x_fit`
(fit_size, dim), `m2_fit` (fit_size), `z_fit` (fit_size, num_probes),
`spike_fit`, `y` (num_probes), `target_mass`, `prior_L` (dim, dim),
**`sigma` (dim, dim)**, `center` (dim), and the one flag that makes
`coarse_config` (whether the row was coarsened).

`sigma` is on that list for a reason that is easy to miss. `:841` sets
`prior.sigma = covariance` -- the row's RAW a-priori covariance, bound at
`:661-662` from the caller's `sigma[rho]`, not `prior_L` -- and that
`InitialGuess` is the first seed of the LM stream (`probe_fit.hpp:631`,
`theta_hat_from_sigma(guess.sigma)`). It is the only read of rank-local state
left inside phase B. It LOOKS recoverable, because `theta_hat_from_sigma(S)`
is defined as `theta_hat_from_cholesky(chol(S).matrixL())`
(`init_dictionary.hpp:444-460`) and `prior_L` is exactly that factor
(`:572-579`). It is not: rebuilding `sigma = L L^T` on the host and factoring
it again does not return `L` bit for bit. Ship the `dim x dim` block; at
`dim = 2` that is four doubles.

**What crosses B to A** for phase C: the finalists, plus the counters. The
pinned baseline (`baseline_modes` as an index into the globally identical
`baseline_sets` at `:531-532`, `theta_baseline`, `baseline_c`, `baseline_s`,
`baseline_score`), the searched fit from `fit_from_probes` (`:843`), and
`evaluations_total`, `candidates_tried`, `max_modes`. Two shape notes: `c` has
length `num_modes + num_extra`, so this is not a fixed-stride record; and a
failed row carries a variable-length message (see section 9). `y_hat` is
recomputable on D1 from `y` and `target_mass`, so it does not travel.

## 4. The assignment rule

Makespan minimization with a fixed partial assignment. Let `w_i` be the
predicted work of row `i`, `L_r` the predicted load of rank `r` under the
identity assignment, `m` the rank count and `eps` the caller's imbalance
tolerance. Set

    T = max( (1 + eps) * sum_i w_i / m ,  max_i w_i )

Then:

1. **Select the overflow.** For each rank whose load exceeds `T`, move its
   largest rows, in decreasing `w`, until its residual load is at most `T`.
   This is the smallest overflow that reaches the target; it is empty when the
   problem is already balanced.
2. **Place it.** In decreasing `w`, put each moved row on the rank with the
   most remaining capacity below `T`.

A moved row MAY be placed back on its own owner; the owner is an ordinary
candidate host with an ordinary remaining capacity.

Properties, all worth a unit test:

- **No migration when balanced.** If every `L_r <= T` the set is empty and the
  scheme is a no-op, which is what a defaulted-on library feature must do.
- **Bound.** `makespan <= max( T , sum_i w_i / m + max_{moved} w_i )`.
  CORRECTED 2026-09-08 (slice 2): an earlier version of this document claimed
  the second term alone, which is false, because a rank already under `T` is
  never touched and its load walks into the makespan whatever moved. One row
  of 10 on rank 0, two hundred rows of 0.06 on rank 1 and an empty rank 2 at
  `eps = 0.1` gives a makespan of 10 against a claimed 7.393. It is also NOT
  Graham's 4/3, which is for an empty schedule.
- **Determinism.** Ties between rows go to the lower row index; ties between
  candidate HOSTS go to the lower rank index (compare loads, not `T - load`:
  the subtraction can round two distinct loads to one capacity). The rule is a
  pure function of the weight vector, the ownership map, `m` and `eps`.
- **One summation order.** A rank must be judged against exactly the load the
  plan reports for it. Summing a rank's rows in one order internally and
  reporting another differs in the last bits, and that let a rank within an ulp
  of `T` shed a row the published numbers say it should have kept, in 5 of
  260,000 random instances.
- **Two moved sets, not one.** "A row may be placed back on its own owner"
  leaves "moved" ambiguous, and only one reading makes the bound hold. The plan
  reports both: the rows step 1 SELECTED, and the subset whose host differs
  from its owner, which is what the exchange must size itself from.

**`T` is a target, not a guarantee.** Step 1 sheds WHOLE rows until a rank is
under `T`, so it sheds more than the overflow, and the remaining capacity can
be unable to absorb what it shed. Counterexample: two ranks, rows
`{0.9, 0.9, 0.9}` on A and `{0.1}` on B, `eps = 0.1`; then
`T = max(1.1 x 1.4, 0.9) = 1.54`, A must shed two rows (1.8 of mass) and B can
take only one of them. `max_i w_i <= T` does not rule this out. Do not write a
feasibility test; write the bound above, and let the rule return its predicted
makespan so the caller can see when the target was not reached.

**The granularity floor.** `T >= max_i w_i` means the achievable balance is
floored by the single most expensive row. At 192 ranks with `fit_points x
evaluations` this is ALREADY binding: the target comes out at the largest row's
weight rather than at the mean. As allocations grow the floor gets worse (the
mean rank load falls, the largest row does not), so whole-row granularity is
the eventual limit of this scheme, and getting past it means splitting a row's
search across ranks -- out of scope here, noted in section 9.

**The caller may override the rule entirely.** `DistFitInput::balance_assign`
takes a local-row-to-host map and is used as given. The rule here is a
heuristic over the one thing the library can see, a predicted cost per row; a
caller often knows more (which ranks share a node, where the columns a row
needs already live, a partition it computed for its own reasons), and that
knowledge has nowhere else to enter. Decided 2026-09-08: this is a supported
extension point, not merely the gate's test hook. The override's hosts are
allgathered, since unlike the library's own rule they cannot be recomputed
identically on every rank.

**The locality refinement.** A row worth moving is a wide row, and a wide row's
window already spans several ranks' territories. Restrict the candidate hosts
of a row to the ranks whose own columns intersect its window (cheaply: whose
territory bounding box meets the window ellipsoid's box) and pick the one with
the most capacity among those. Makespan is primary, halo and locality are the
tie-break. Defer to a later slice if it complicates the first one.

**The weights.** The assignment is computed right after the window pre-pass
and before phase A (see section 1, fact 2). The weight is the window size on
rung 1, and the previous rung's `fit_points x evaluations` for the same row on
rungs 2 and up. No proxy, no state carried between builds. Anchor each rank's
initial load on the previous rung's MEASURED per-rank seconds rather than on
the sum of predicted weights, when a previous rung exists.

A misprediction costs wall time. It does NOT cost correctness or bit-identity,
because a row's fit is a pure function of the package that travels with it.
It can, however, cost memory: see the byte cap in section 9.

## 5. The API

New header `include/lgpsf/row_balance.hpp`, namespace `lgpsf`. NOT under
`mpi/` (as an earlier version of this document said): the rule has no MPI
dependency and has to be unit-testable from the serial suite, which does not
link MPI. The exchange, in slice 3, does live in the MPI layer and includes
this.

```cpp
namespace lgpsf::mpi {

/// The assignment rule, pure: no MPI, no randomness, no allocation order
/// dependence.  weights and owners are the GLOBAL row arrays as seen after an
/// allgather of the local counts; the return is the destination rank of every
/// row (== owner for the rows that stay).
struct BalancePlan
{
    std::vector<int> host;        ///< (nrows_global) destination rank
    std::vector<int> moved;       ///< global indices of the moved rows
    double predicted_imbalance;   ///< max/mean of the resulting predicted load
};

BalancePlan balance_rows(const std::vector<double>& weights,
                         const std::vector<int>&    owners,
                         int                        num_ranks,
                         double                     tolerance);

/// The exchange: pack the phase-A packages of the moved rows, alltoall, and
/// return the candidates.  Both directions are one Alltoall of counts plus
/// Isend/Irecv of payload, as in dist_wsym's triplet exchange.
}
```

In `operator_fit.hpp`, the three phases become `detail` functions over two
`detail` structs (`RowFitProblem`, `RowFitCandidates`). They do NOT enter the
public surface: only `dist_fit`, in the same library, needs them.

The knob goes on `DistFitInput`, NOT on `OperatorFitConfig`: a serial user who
set it there would get silence, since `fit_operator` has no ranks to balance
over.

```cpp
    /// Fitting-only row redistribution: the imbalance tolerance of the
    /// water-filling rule (0 = disabled, the default; 0.1 is a sane value).
    /// Affects wall time only; the result is bitwise identical either way.
    double balance_tolerance = 0.0;

    /// (nrows) the previous rung's per-row evaluation counts, or empty.
    /// Purely a scheduling input; see section 4.
    Eigen::VectorXd prev_evaluations;
```

`fitted_on_rank` goes on `DistFitResult` for the same reason, not on
`FitDiagnostics`.

**How the phases are exposed matters.** Materializing every row's package at
once is gigabytes on the busiest rank, and phase A is what builds a package.
So `fit_operator` takes an optional TWO-CALLBACK delegate, and only delegated
rows ever have a package:

```cpp
struct RowDelegate
{
    /// Called ONCE after the window pre-pass, with every row's window size
    /// (0 for a row that will not be fitted).  Returns the host of each row;
    /// an entry equal to this rank's own index means "fit it here".  This is
    /// where the caller runs balance_rows.  Absent => every row is local.
    std::function<std::vector<int>(const std::vector<int>& window_sizes)> assign;

    /// Called ONCE after phase A of the delegated rows, with their problems
    /// in row order and an output slot each.  The implementation is free to
    /// solve them anywhere -- dist_fit ships them to their host, runs
    /// detail::fit_row_candidates there, and ships the candidates back.
    std::function<void(const std::vector<int>& rows,
                       const std::vector<RowFitProblem>& problems,
                       std::vector<RowFitCandidates>& out)> solve;
};
```

The loop becomes: pre-pass; `assign`; then per row, phase A followed
immediately by B and C fused when the row is local, or phase A and a packed
problem when it is not; then `solve`; then phase C for the delegated rows.
Local rows keep exactly today's fused path and never materialize anything.
This also keeps `operator_fit.hpp` free of MPI: the exchange lives entirely
behind `solve`.

**AS BUILT (slice 3 first half, `4085adc`), with three corrections to the
sketch above.** `RowDelegate` also carries `self`, the caller's own host index,
which "a host equal to this rank's own index" needs and the sketch omitted.
`solve` takes a `detail::RowFitContext` first -- the baseline sets, the row
config carrying the hoisted CV split and jitter table, the probe count and the
two parameter counts -- because without them the search cannot run at all, and
rebuilding that hoisting outside `fit_operator` would duplicate the one piece
of logic the determinism argument rests on; a receiving rank uses its own
context, which SPMD makes identical. And an output slot needs an UNSET state,
or a delegate that fails to solve a row ships a silent zero: `solved()` is
false when `baseline_index < 0`, and `failure` carries a foreign throw's
message home. `solve` is called even with zero rows, so a collective
implementation is entered on every rank; an empty return from `assign` means
"no delegation at all" and skips it, so a collective `assign` must return
full-length on every rank or on none.

**AS BUILT (slice 3 second half), `include/lgpsf/mpi/row_delegate.hpp`.**
`lgpsf::mpi::RowExchange` is the delegate: construct it on a communicator, hand
`delegate()` to `fit_operator`, read `fitted_on_rank()` afterwards. `dist_fit`
builds one when `DistFitInput::balance_tolerance > 0` and otherwise passes no
delegate at all. What travels, and why, is documented at the top of that
header; what the exchange does about failure is section 9 below, mechanized.
Six things this sketch got wrong or left out, all found by building it:

1. **A rank with NO ROWS could not participate.** Its full-length assignment IS
   the empty vector, so "empty means all mine" made it the one rank that skips
   `solve` -- which is where the collectives are. That is a hang, not a missed
   optimization, and an idle rank is an ordinary thing at high rank counts.
   `fit_operator` now reads an empty return from a zero-row caller as
   full-length and enters `solve` anyway (given a `solve` to enter).
2. **A rank with no rows could not be fitted at all**, redistribution or none:
   `fit_operator` refuses a spiked fit with separate row coordinates unless
   `row_own_col` is non-empty, and with no rows the empty vector is the
   full-length answer again. That check now reads the length rather than the
   emptiness. It is a pre-existing hole in `dist_fit` that the gate's
   zero-row pass found on its first run.
3. **`prev_evaluations` carries the WEIGHT, not the evaluation count.** The
   weight the rule wants is `fit_points x evaluations`, and `dist_fit` cannot
   form that product: it has the window sizes, not the previous rung's coarse
   point counts, and with coarsening on the two differ by an order of magnitude
   for exactly the rows that migrate. The caller has both factors for free in
   the previous rung's diagnostics, so it passes the product. The field keeps
   the name this document gave it; its documentation says what it holds.
4. **The host's phase-B seconds needed somewhere to ride**, so
   `RowFitCandidates::search_seconds` exists. The owner reports it as the row's
   `search_seconds` and ADDS it into `row_seconds`, which is the attribution
   section 9 decided on.
5. **`assign` needs the failure discipline too**, not just `solve`. Its inputs
   are rank-local (the weight vector's length, an override's length), so a
   throw there is a throw on one rank inside a sequence of collectives. It
   carries the same flag and reduces it after its last collective.
6. **The decision to delegate is itself collective.** A rank that leaves
   `balance_tolerance` at 0 while its peers set it does not produce a different
   answer, it hangs the job, so `dist_fit` reduces the decision (and the
   tolerance) before building anything. That reduction runs even when the
   feature is off, because "off here, on there" is exactly the case it exists
   to catch.

## 6. Determinism

Reassignment is bitwise-safe, and the reasons are already in the code:

- **The randomness is hoisted before any row is touched.**
  `operator_fit.hpp:511-522` builds the CV split and the jitter table from
  `num_probes`, `dim` and the optional global seed only, never from a row
  index or a rank, precisely so "the result cannot depend on scheduling".
- **A row's fit is a pure function of its package.** Every input crosses as
  `MPI_DOUBLE`; nothing is recomputed from rank-local state. This holds ONLY
  if the package is the full list in section 3, `sigma` included. The seed of
  the LM stream reads the raw covariance, and reconstructing it from its
  Cholesky factor on the host is not bit-identical. The perverse-assignment
  gate pass is what catches an omission here, which is the strongest reason to
  write that test before the migration rather than after.
- **Window content and order are already partition-independent.** Membership is
  an exact predicate at the leaf, the window is sorted (`:634-640`), and
  `spike_position` is a binary search into that sorted window (`:685-691`), so
  it is the rank of the own gid among window gids wherever the row is fitted.
  The existing n = 1/2/4 gate already relies on this, since the combined
  column set differs at every rank count today.
- **No cross-row accumulation** anywhere in the fit; `assemble_sparse`
  concatenates per-row triplet blocks in row order.
- **The mode-set registry** (`:1019-1077`) assigns ids serially in local row
  order *after* the loop, and rows stay in D1 order, so ids are unaffected.

**The gate.** `tests/mpi/test_dist_fit_mpi.cpp` already builds a serial
reference on rank 0 and compares dense row images bitwise. Add a pass with
`balance_tolerance` set so low that a large fraction of rows migrate, plus a
deliberately perverse assignment (every row hosted by owner + 1 mod size), and
require bit-identical output against the same reference. That is the whole
correctness argument, mechanized.

## 7. What changes where

| repo | change | size |
|---|---|---|
| **lgpsf** | phase split in `operator_fit.hpp` + the solve hook, two detail structs | the bulk |
| | `mpi/row_balance.hpp`: the rule and the exchange | ~300-400 lines |
| | `dist_fit.hpp`: the knob, `prev_evaluations`, `fitted_on_rank`, the failure flag | ~120 lines |
| | tests: balanced / perverse / failure gate passes, unit tests of the rule | ~200 lines |
| | Python bindings: the two new `DistFitInput` fields and `fitted_on_rank`, mirrored field by field, plus their pytest assertions | ~30 lines |
| | `docs/` (the coarsening note gains a paragraph) + this plan | |
| **consumer (lgpsf-hessian)** | pass the tolerance through to the public header | ~10 lines |
| | keep the previous rung's `evaluations` and pass them back in | ~20 lines |
| | migrated count and achieved imbalance in the report | ~10 lines |
| | dump column for the fitting rank: bump `hdr[1]` to 5 and `P`, and update the offline readers | ~20 lines |
| **application (ymir)** | one option, the numbers on the existing fit log line | ~15 lines |

The consumer keeps its row-gid identity, its triplet handling, its
symmetrization and its assembly exactly as they are. That is the payoff of
putting the seam where it is.

**An unstated precondition, now stated:** every rank's `config.row.mode_policy`
must be the same policy. It is a virtual, un-serializable object
(`mode_policy.hpp:126-152`) and does not travel with a package. SPMD makes this
true today; migration makes it load-bearing.

## 8. Slices

0. **Measure `r`, the resident share.** DONE 2026-09-08 (lgpsf `2e4e1cd`): three
   `steady_clock` pairs around the coarsening, the search and the re-score,
   reported alongside `row_seconds` and exposed to Python; then real
   continental windows replayed locally against them. `r = 0.084`. No cluster
   job was needed and none is: once the timers reach the consumer's dump, the
   next production run confirms them for free.
1. **Phase split, no migration.** `operator_fit.hpp` only; the three phases
   run back to back on D1. Acceptance: every existing test passes bitwise,
   including the MPI gate, with no API change.
2. **The assignment rule.** `balance_rows` as a pure function with serial unit
   tests for the properties in section 4, INCLUDING the counterexample there as
   a regression (the rule must report a predicted makespan above `T`, not
   claim success). Acceptance also includes re-running section 1's study
   through the real rule on the recorded dump. Independent of slice 1.
3. **Migration.** The exchange plus the wiring in `dist_fit`, default off.
   Acceptance: the perverse-assignment gate pass is bit-identical, and the
   balanced pass at `tolerance = 0.1` is bit-identical to the unbalanced one.
4. **Consumer and application pass-through**, then one continental run with it
   on. Acceptance: the operator, the QC ladder decisions and the CG iteration
   count are unchanged; the fit wall max/mean drops.
5. **The locality tie-break and the byte cap**, if the field numbers say they
   are needed.

Everything through slice 4 keeps the default behaviour bit-identical, so the
risk sits in one continental run rather than in the library.

## 9. Risks, decisions and open questions

**Failure paths (unresolved, and the way this hangs a job).** Today the try
block spans A+B+C as one unit (`:658-974`) and the catch clears
`outcome.window`, zeroes the counters and stores a message (`:965-974`). Three
things follow, and the first is the dangerous one.

- **Exchange counts must be derived from data, never from the balance plan.**
  Which rows are packageable is only known AFTER phase A: `attempt[rho] == 0`
  rows never enter the loop (gated at `:567-570`, `sigma` not SPD at
  `:573-578`, window covariance not SPD at `:603-610`), and the
  `window.size() < 2` throw (`:665-670`) and the degenerate-coarsening throw
  (`:741-753`) both fire inside phase A. A receiver that sized its receives
  from the globally known plan would deadlock the job on one row failing on
  one rank.
- **A throw on a foreign host must travel back** as a status plus a
  variable-length message, and the owner must then execute the catch semantics
  locally, including clearing its own window.
- **A throw that is not per-row scoped** -- `bad_alloc` while packing, above
  all -- diverges ranks inside a collective sequence and hangs rather than
  crashes. `fit_operator` contains no MPI today and cannot do this. Wrap the
  whole migration region so every throw becomes a rank-local flag, `Allreduce`
  it, and throw after the exchanges complete.

**Sender-side memory, and the byte cap (SUPERSEDED 2026-09-09 -- read this,
not what the paragraph used to say).** The average bytes per rank is a
misleading number: the senders are few by construction, since being overloaded
is what makes them senders, so a plan fine for the clock can be an
out-of-memory. That much stands. Everything this document said about HOW to
bound it was wrong, and a continental run proved it:

- It said the ASSIGNMENT needs the cap, enforced before phase A materializes
  anything, on the window size as an upper bound of the payload, reverting the
  lightest migrations first, as a pure function of the global arrays so every
  rank reverts the same rows. Every clause of that is now false.
- The closing claim -- "`fit_points_rank_max` was not needed: the window size
  is already an upper bound and is already known where the decision is made"
  -- is the defect stated as a conclusion. The window size IS an upper bound,
  but a ten- to fiftyfold one on exactly the rows the rule wants to move,
  because a package carries coarsened CELLS. So a cap built on it rejects the
  expensive rows and keeps the cheap ones. Measured: 6,763 of ~6,900 planned
  migrations reverted, predicted imbalance 20.62 against an unbalanced 20.7.
  The earlier instinct, size it from `fit_points_rank_max`, named the right
  quantity; the error was insisting the decision be made in the one place
  where only window sizes exist.
- Neither did this document list the option that turned out to be right, and
  which is cheaper than both the ones it did list (a cap in the rule, or
  sending in waves): **do not send it, fit it at home.** The owner is holding
  the package; solving it locally is bit-identical by the same argument that
  licenses sending it away.

As built (`040a23c`): `assign` produces the makespan plan and nothing else;
the cap lives in `solve`, on exactly the bytes `pack_problem` will write; a
dropped row is fitted by its owner in the same `parallel_for` as the foreign
rows; the receive-side budget rides the acknowledgement round, now tri-state
(take it / refused for budget / refused because this rank has failed). Drops
are rank-local and need no agreement -- a drop moves WORK, never an answer --
so the sender drops heaviest first, ties by ascending row, and the receiver
refuses whole peers heaviest first, ties by ascending rank. Default 512 MiB,
about 2C of transient peak on the few ranks that send or host.

**Section 3 and this section contradicted each other, and the code followed
this one.** Section 3 already said the payload is the coarse cells, one
paragraph before pointing at a cap that measured the window. A cross-reference
would have caught it before a 192-rank run did.

**The counters are rank-local.** `rows_migrated`, `rows_capped` and
`bytes_sent` are per rank; only `predicted_imbalance` is the plan's and global.
A consumer that reports them must reduce them, which lgpsf-hessian did not do
at first, so the first continental run printed rank 0's share beside a global
capped count.

**Cell-structure reuse across rungs (open, profiling-gated -- maintainer).**
The cells depend only on geometry (`x`, `m2`, centre, frame, eps), not on the
probes, so the structure could be computed once per build and only the new
probe columns shipped per rung. Today `dist_fit` runs per rung and recoarsens
each time. The trade is not compute against compute but MEMORY against
compute: the cell assignment is one integer per window point, the same size as
the window index list already kept, so caching it across rungs roughly doubles
that storage -- a few hundred megabytes on the busiest rank. Slice 0's
coarsening timer settles whether it is worth anything.

**The re-score stays on D1 (DECIDED, maintainer).** Not merely because that is
where the window is. One purpose of the re-score is to assess how well the
coarsened fit does on the UNCOARSENED points, and the fit was optimized
against the coarse cells, so the coarse score is biased by selection rather
than merely approximate. Scoring on anything the fit was fitted to defeats the
purpose. If the residual ever binds hard, the two options that PRESERVE that
purpose are: a distributed re-score (the score is a mass-weighted quadrature
over the window and is therefore reducible across the ranks holding it -- note
this is still a FULL-window score, not a coarse one); or scoring both
finalists on a fresh random subsample of the full window, mass weighted, both
on the same draw. A fresh subsample is unbiased for the full-window score
precisely because the fit was not optimized against it, and scoring both
candidates on one draw cancels most of the noise in what is only ever a
comparison. Either would need validating against the recorded eps 0.15 case,
where the coarse score is known to lie.

**Threads (AGREED, maintainer).** The migrated rows join the host's own rows in
one `parallel_for`; rows write disjoint slots, so nothing new is needed.

**`row_seconds` attribution (DECIDED, for the MPI half).** `row_seconds` keeps
its meaning: everything attributable to the row, wherever it ran. So the host
returns its phase-B seconds on `RowFitCandidates` and the owner ADDS them to
the row's total, leaving `r = 1 - search_seconds / row_seconds` well defined
and comparable across migrated and resident rows. As built, a delegated row
has `search_seconds = 0` and `row_seconds` counts only the owner's phases,
which is what the field must fix. Per-rank wall time is a different quantity:
the scheduler wants the sum over rows FITTED on a rank, so
`DistFitResult::seconds_total` must be documented as that, not as "this rank's
rows".

**Nothing in the interface enforces the failure discipline (open, and it is
the one that hangs).** A collective `solve` is entered once per `fit_operator`
call and must run to completion inside it. The "wrap the migration region,
`Allreduce` a flag, throw only after the exchanges complete" rule of this
section lives entirely in the delegate's body; a rank that throws out of
`solve` while its peers are still in an exchange will hang the job. The
interface cannot prevent that, so the MPI half must be written with it in
front of mind, and the gate must include a rank that fails while others do
not.

**HOW THE DISCIPLINE WAS ENFORCED (slice 3 second half).** Both callbacks are
written as one region with a rank-local `failed` flag and one `MPI_Allreduce`
of it AFTER the last exchange; nothing throws before that reduction except a
condition every rank decides identically. Three things beyond the plan were
needed:

- **Exchange counts come from the buffers**, in both directions and in rows AND
  in doubles. A rank whose packing threw sends nothing at all and says so, and
  a receiver that could not unpack still answers one record per row it was
  sent, so a short answer degrades to failed rows rather than to a wait.
- **An acknowledgement round after the counts.** Counts alone make a
  RECEIVE-side allocation failure unsurvivable: the senders have already been
  told how much is coming, so a receiver that threw would leave them in an
  `Isend` nobody matches. Each direction is therefore counts -> allocate ->
  "can you take it?" -> payload, and a refusing receiver's senders skip it. The
  rows then come home unsolved, which is a failed row, and the flag makes every
  rank throw anyway.
- **A cap on the MPI count itself.** `bytes_cap` is a knob, and MPI counts are
  `int`, so an oversized buffer is turned into the collective failure rather
  than into a truncated message.

**THE BYTE CAP AS BUILT.** Enforced inside `assign`, before phase A
materializes anything, on an upper bound of each row's payload -- its WINDOW
size, since coarsening only ever shrinks the quadrature -- with the lightest
migrations reverted first, as a pure function of the same global arrays the
plan is, so every rank reverts the same rows. The default is 64 MiB per rank
per direction. It is sized by MEMORY HEADROOM rather than by the 26 MB average
of section 2, because the average is exactly the misleading number: the senders
are few, so the cap is meant to bind on them. A sender holds the packages phase
A built plus its copy of them in the pack buffer, a receiver its receive buffer
plus the unpacked problems, so 64 MiB is about 128 MiB of peak per rank per
direction -- roughly 6% of a rank's memory at 48 ranks on a 192 GB node, and
above the whole job's average traffic, so it does not bind in the intended
regime. `fit_points_rank_max` was not needed: the window size is already an
upper bound and is already known where the decision is made.

**The gate must cover more than the perverse pass.** The perverse assignment
(`owner + 1 mod size`) is the identity at n = 1, so that pass is only
meaningful at n >= 2. Add: a row that throws while on a foreign host, a rank
that owns zero rows, and a row gated out by `attempt`.
