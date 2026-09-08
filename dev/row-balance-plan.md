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
degenerates to no migration when the problem is already balanced.

---

## 1. The measurement

From one build's fit dump on the continental problem, 409,545 rows over 192
ranks, last rung (30 probes), with the per-row `row_seconds` telemetry
(`operator_fit.hpp:303-309`) as ground truth. The table is the per-rank
max/mean of *actual* fit seconds after keeping every row on its owner except
the listed fraction, which is placed longest-first onto the least-loaded rank.

| rows moved | local proxy | previous rung's work | oracle (true seconds) |
|---|---|---|---|
| none (today) | 22.2 | 22.2 | 22.2 |
| top 1% | 4.5 | 3.8 | 3.0 |
| top 2% | 3.0 | 2.4 | 1.9 |
| top 5% | 2.0 | 1.3 | 1.3 |
| top 10% | 1.0 | 1.0 | 1.0 |

Three facts drive the design.

1. **The tail is thin.** The top 1% of rows hold 21% of the fit points, the
   top 2% hold 32%, the top 5% hold 49%. Moving a couple of percent of the
   rows moves a third of the work.
2. **Nothing needs history to start.** A purely local proxy, the window
   ellipsoid's area times the local node density from a nearest-neighbour
   query, predicts the true coarse point count with log-correlation 0.96 and
   balances nearly as well as the count itself. Both inputs are available
   before anything expensive happens, so the first rung of the first build is
   balanced too.
3. **History is better when it exists.** `fit_points x evaluations` from the
   *previous rung* correlates 0.94 in log with row seconds, against 0.62 for
   points alone. Because `dist_fit` is called once per rung
   (`fit_impl.hpp`, the ladder loop) and the halo is not rebuilt for this
   scheme, the assignment can be recomputed at every rung from the previous
   rung's exact measurements. Rung 1 uses the proxy or a caller hint; rungs 2
   and up use measurements.

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
- The migration payload is the coarse cells, not the window. At the last rung
  the moved 2% is about 1.9e7 cells; at 3 + 30 doubles per cell that is about
  5 GB across the whole job, 26 MB per rank.

**The cost of the clean seam.** The re-score stays on the overloaded rank. For
a wide row the search runs a few hundred evaluations over a few thousand
cells while the re-score is two evaluations over the full window, so it is
around 9% of the row's fit. That caps the achievable gain near tenfold rather
than twentyfold: the busiest rank in the measurement above would land near 60
s rather than near 22. Worth it for the isolation.

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
| 777-843 | whitening, the pinned baseline at the prior, the mode-set ladder, `fit_from_probes` | **B** |
| 859-980 | the full-window re-score of the finalists, the guard, selection into `outcome` | C |

The comment at `:717-726` already states the invariant the split relies on:
"From here to the guard everything reads `x_fit` / `m2_fit` / `z_fit` /
`spike_fit`. The window itself is the deployed support and is never touched."
Phase B is exactly the region that comment describes.

**What crosses A to B** (all dense, small, trivially serializable): `x_fit`
(fit_size, dim), `m2_fit` (fit_size), `z_fit` (fit_size, num_probes),
`spike_fit`, `y` (num_probes), `target_mass`, `prior_L` (dim, dim), `center`
(dim), and the one flag that makes `coarse_config` (whether the row was
coarsened).

**What crosses B to A** for phase C: the finalists. Today those are the pinned
baseline (`baseline_modes`, `theta_baseline`, `baseline_c`, `baseline_s`,
`baseline_score`) and the searched fit returned by `fit_from_probes`
(`:843`). A few dozen doubles plus a mode list.

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

Properties, all worth a unit test:

- **No migration when balanced.** If every `L_r <= T` the set is empty and the
  scheme is a no-op, which is what a defaulted-on library feature must do.
- **Feasibility.** `T` is at least the average, so total slack at or below the
  waterline is at least the total overflow; greedy packing can only fail
  through indivisibility, which the second term of `T` rules out.
- **Bound.** The placement is longest-processing-time-first into the least
  loaded, so it carries the standard Graham guarantee against the optimum of
  the residual problem.
- **Determinism.** Ties broken by row global id; the rule is a pure function of
  the weight vector, the ownership map and `eps`.

**The locality refinement.** A row worth moving is a wide row, and a wide row's
window already spans several ranks' territories. Restrict the candidate hosts
of a row to the ranks whose own columns intersect its window (cheaply: whose
territory bounding box meets the window ellipsoid's box) and pick the one with
the most capacity among those. Makespan is primary, halo and locality are the
tie-break. Defer to a later slice if it complicates the first one.

**The weights.** In order of preference, whichever is available:
`fit_points x evaluations` from the previous rung of the same build (exact,
log-correlation 0.94 with seconds); a caller-supplied hint from the previous
build; the local proxy (window area times local node density, log-correlation
0.96 with the coarse point count). A misprediction costs wall time and
nothing else: the fit of a row is a pure function of the package that travels
with it, so correctness and bit-identity do not depend on the weights.

## 5. The API

New header `include/lgpsf/mpi/row_balance.hpp`, above `dist_fit.hpp` in the
layering and below nothing else in the MPI layer.

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

In `OperatorFitConfig`: one knob.

```cpp
    /// Fitting-only row redistribution: the imbalance tolerance of the
    /// water-filling rule (0 = disabled, the default; 0.1 is a sane value).
    /// Affects wall time only; the result is bitwise identical either way.
    double balance_tolerance = 0.0;
```

In `DistFitInput`: one optional field.

```cpp
    /// (nrows) predicted per-row work from a previous build, or empty.  Only
    /// ever a scheduling hint.
    Eigen::VectorXd row_weight_hint;
```

In `FitDiagnostics`: one field, `fitted_on_rank`, for diagnosis.

## 6. Determinism

Reassignment is bitwise-safe, and the reasons are already in the code:

- **The randomness is hoisted before any row is touched.**
  `operator_fit.hpp:511-522` builds the CV split and the jitter table from
  `num_probes`, `dim` and the optional global seed only, never from a row
  index or a rank, precisely so "the result cannot depend on scheduling".
- **A row's fit is a pure function of its package.** Every input crosses as
  `MPI_DOUBLE`; nothing is recomputed from rank-local state.
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
| **lgpsf** | phase split in `operator_fit.hpp`, two detail structs | the bulk |
| | `mpi/row_balance.hpp`: the rule and the exchange | ~300-400 lines |
| | `dist_fit.hpp`: the knob, the hint, the proxy fallback | ~80 lines |
| | tests: the balanced/perverse gate pass, unit tests of the rule | ~150 lines |
| | docs note + this plan | |
| **consumer (lgpsf-hessian)** | pass the tolerance through the config to the public header | ~10 lines |
| | persist the previous build's per-row work, pass as the hint (optional) | ~30 lines |
| | migrated count and achieved imbalance in the report | ~10 lines |
| | optional dump column for the fitting rank | ~10 lines |
| **application (ymir)** | one option, the numbers on the existing fit log line | ~15 lines |

The consumer keeps its row-gid identity, its triplet handling, its
symmetrization and its assembly exactly as they are. That is the payoff of
putting the seam where it is.

## 8. Slices

1. **Phase split, no migration.** `operator_fit.hpp` only; the three phases
   run back to back on D1. Acceptance: every existing test passes bitwise,
   including the MPI gate, with no API change.
2. **The assignment rule.** `balance_rows` as a pure function with serial unit
   tests for the four properties in section 4. Independent of slice 1; can run
   in parallel.
3. **Migration.** The exchange plus the wiring in `dist_fit`, default off.
   Acceptance: the perverse-assignment gate pass is bit-identical, and the
   balanced pass at `tolerance = 0.1` is bit-identical to the unbalanced one.
4. **Consumer and application pass-through**, then one continental run with it
   on. Acceptance: the operator, the QC ladder decisions and the CG iteration
   count are unchanged; the fit wall max/mean drops.
5. **The weight feedback and the locality tie-break**, if the field numbers say
   the proxy and the previous rung leave anything on the table.

Everything through slice 4 keeps the default behaviour bit-identical, so the
risk sits in one continental run rather than in the library.

## 9. Open questions

- **Cell-structure reuse across rungs.** The cells depend only on geometry
  (`x`, `m2`, centre, frame, eps), not on the probes, so the structure could be
  computed once per build and only the new probe columns shipped per rung.
  Today `dist_fit` runs per rung and would recoarsen each time. Worth doing
  only if profiling says the coarsening itself matters; it is far cheaper than
  the search.
- **How much of the re-score to keep on D1.** The 9% residual is what caps the
  gain. If it ever binds, the alternative is a distributed re-score (the score
  is a mass-weighted quadrature over the window and is therefore reducible),
  but that is a much larger change and should not be attempted first.
- **Threads.** The migrated rows join the host's own rows in one `parallel_for`;
  rows write disjoint slots, so nothing new is needed. Confirm the row-seconds
  telemetry still attributes sensibly when a rank's row set is heterogeneous.
