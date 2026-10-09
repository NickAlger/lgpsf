# Column-owner phases for the distributed fit -- plan, v2 after review (2026-10-09)

> v1 was reviewed by an agent without project context (logical correctness and simplicity); v2 takes its
> corrections: the baseline is not an owner-side phase, the re-score needs design blocks, the symmetrization
> variant weights by the A-row id, stage 1's nnz claim was wrong, the coarsening is done slice-wise with today's
> code instead of a new tessellation, cross-rank reproducibility needs reproducible summation, the assembly's
> row-level verdict needs a reduce, and the distributed `LGOperator` loses its column arrays.  Verified against
> the code: `fit_row_candidates` "reads only the fit's quadrature -- never the full window" (operator_fit.hpp
> ~589), `select_row_fit` re-scores with one `linear_cv_score` per finalist on the full window (~739-791),
> `dist_wsym.hpp` weights by `w2[t.row]` (132) and indexes `energy[t.row - row_start]` (70), `assemble_sparse`
> ships a row's smooth part and spike "TOGETHER or not at all" (lg_operator.hpp 632).

**The problem.** In the SPMD fit (`mpi/dist_fit.hpp` -> `fit_operator`) everything a row needs is brought to
the row's owner: the halo pushes every window column's probes and mass to it (`halo_plan` / `halo_push`), and
the owner resolves the window, coarsens it, searches (or delegates the search, `row_delegate.hpp`), re-scores
the finalists on the full window and assembles the row's entries.  Only the search migrates.  The rest scales
with the RAW window and stays home.  On the ice-sheet application at 1.19 M rows the rank owning the widest
windows holds ~100x the mean window nodes, the fit wall tracks that rank at ~1e-5 s per window node, and the
build's wall is 13-17x the mean work while the balance's plan reports 1.10.  A node-count cap on the window is
not acceptable: the window is the a-priori ellipsoid, i.e. physics.  So the work moves, not the window.

**The idea (Nick, 2026-10-09).** Ship the MODEL, not the data.  Every owner-side phase of a row is a sum over
the window's columns, and the columns have owners.  Give the column owners the row's geometry and later its
candidate models (theta, coefficients: ~50-70 doubles) and let each do its share on its own columns, returning
partials.  The owner keeps the row (status, model, diagnostics, the dump); the search keeps running wherever
the balance puts it; the operator comes out the same on the same windows.

## The phases, as sums over the window's columns

| phase | today | with column owners: what a column owner does on its own columns in the window |
|---|---|---|
| window resolution | owner's dual tree over the halo'd points | its own dual tree over its own columns against the row's window ellipsoid (exact; nothing sent) |
| coarsening (phase A) | `coarsen_window` on the gathered window | `coarsen_window` on ITS SLICE (same frame, centre, eps); returns the slice's cells: mass, mass-weighted centroid and probe means, each cell's smallest gid; singletons are bitwise copies |
| baseline + search (phase B) | on the coarse batch, owner or delegate | UNCHANGED: on the coarse batch assembled from the slices, at the owner or wherever the row delegate puts it |
| re-score of the finalists (phase C) | one `linear_cv_score` per finalist on the full window (refits the linear coefficients per fold) | the finalists' (theta, mode set) come down; it returns each finalist's DESIGN block on its columns, k x (modes + extra); the owner sums the blocks and scores the folds (`linear_cv_score` split into design assembly and fold scoring) |
| assembly | `assemble_sparse` on the window at the owner | the winner comes down; it evaluates `m_i k_i(x_j) m_j` at its own columns inside the tau-ellipsoid: the entries STAY with it (column layout of A) |

Not an owner-side phase: the baseline (it runs on the coarse batch inside phase B).

## Slice-wise coarsening (the reviewer's simplification, adopted)

Coarsen each (row, rank) slice independently with today's `coarsen_window` and concatenate the slices' cells
at the owner in canonical order (slices by the rank's gid block, cells within a slice by smallest member gid, as
they are emitted).  The `eps * d` bound is per cell, so the union of graded quadratures of disjoint pieces is a
graded quadrature of the union; each slice protects its own farthest point (physical distance, ties to the
smallest gid, the gid shipped), so the global farthest is a singleton for free; the spike is protected on the
owner's own slice only; singletons remain bitwise copies of their points; a slice smaller than its would-be
cell count ships as points automatically (its cells are singletons).  No new tessellation, no `h_min`, no leaf
identity, no merge of partial cells.  Cost: extra cells where one coarsening's leaf would straddle a rank
boundary, O(ranks touched x boundary cells) per row -- measured offline before building (see "Validate the
thesis first"); fallback if it matters: the dyadic root aligned on the window's unit box, still without merging.

## What has to change in lgpsf

**0. Instrumentation first** (`dist_fit`, `fit_operator`, a day): timers around the phases outside the row loop
(halo push, window resolution pre-pass, assembly) and per-rank maxima surfaced in `DistFitResult`.  The dump's
per-row telemetry says the in-row phases (coarsen, re-score) are a small share of row seconds; the laptop
split says the outside-the-loop phases are the larger part of a giant row's cost; a flop count says the
re-score should dominate.  Stage 0 settles which it is on the cluster and fixes the order of the rest.

**1. The reverse plan** (`mpi/halo_exchange.hpp`, new `ReversePlan`): allgather each rank's column box (or its
box-forest cut); every row owner pushes the windows that touch a remote column box to that rank: row gid,
centre, `L_w` (the receiver factors nothing), the coarsening eps, once per build (windows are geometry; rungs
reuse the pattern).  Each rank resolves membership exactly on its own columns, prunes pairs with empty slices,
and sends the slice counts to the owner (data-derived, as `row_delegate.hpp`'s discipline wants); the
`coarsened` flag follows from the counts.  The pair list (row gid, covering rank) is the pattern for every
exchange below: three lockstep sparse exchanges per rung (cells up; finalists down / designs up; winner down)
plus the energy reduce, so the column-side work cannot overlap the search -- expected, stated.

**2. Column-side kernels** (pure): `slice_cells` (= `coarsen_window` on the slice), `design_block(frame, modes,
own columns, m, z)`, `assemble_slice(model, tau, own columns)`.  Loops over the rank's own columns inside the
window with `eval_lg_basis`.

**3. The new `dist_fit` flow**, driving `detail::fit_row_candidates` and `detail::select_row_fit` directly
instead of adding callbacks to `fit_operator` (the seams exist; `parallel-architecture-revisit.md` warned
against more hooks): (a) once per build, windows down, counts up; (b) per rung, slice cells up, the owner
concatenates in canonical order -> the coarse batch; `RowDelegate::assign` takes window sizes from the counts
before phase A and `fit_points` after (b), otherwise unchanged; (c) phase B as today; (d) finalists down, design
blocks up, the owner scores the folds and decides (the guard); (e) the winner down, assembled at the column
owners.  The per-row diagnostics, status and model stay with the owner; the dump is unchanged.

**4. Symmetrization and layout** (`mpi/dist_wsym.hpp`): the row energies `||A_i,:||^2` are spread over column
owners -> one reduce of partial energies to the row owners (one double per pair), then a variant of
`dist_weighted_symmetrize` that takes w^2 by GLOBAL id and weights each entry by its A-ROW id (the entry (i, j)
held by owner(j) gets `w_i^2`), with the stay/ship rule mirrored (the untransposed product travels to
owner(i), the transposed stays).  NOT "feed A^T": that pairs the wrong weight.  B is symmetric, so the column
layout of A yields B's rows at the row owners as today; `lgh_fit_get_mat`'s CSR, the held-out QC and the GLR
matvec take it unchanged.  `LGH_WSYM_NONE` is incompatible with column phases (forbid it, or add a transpose
exchange).  The row-level assembly verdict rides the same reduce: a per-row OR of "any non-finite entry" over
covering ranks, encoded in the w^2 allgather (NaN = dropped row), so column owners drop a whole row's entries
and the spike together, as `assemble_sparse` does; the `on_window` fallback (kernel covariance not finite) is a
model-level flag the owner ships with the winner.

**5. The distributed `LGOperator` contract**: the owner no longer holds the window's points.  Kept: theta, mu,
L, c, s, status, window centre and covariance, `kernel_at` at the row's own x, `spike_measure`, diagnostics
(the bridge's dump and `finish_fit` read only these).  Dropped: `x_cols` (own + halo), `window_indptr /
indices`, `row_window`, `assemble_sparse`, `qc_map`, column `eval`; `DistFitResult::B_local / col_gids /
own_to_comb`; `window_candidates` comes from the counts.  The Python bindings mirror `DistFitResult`.

## What has to change in lgpsf-hessian (`impl/fit_impl.hpp`)

Build the reverse plan (the halo plan goes once the halo of raw probes is retired); call the new `dist_fit`;
the energy reduce before the symmetrization variant; the CSR from B's rows (unchanged); print the stage-0
timers; `--pencil-column-phases 0|1` (default 0 until validated).  The fit dump (v6) is unchanged.

## Reproducibility

Partial sums reassociate: slice cells, design blocks and row energies are sums of per-rank pieces, so the
cross-rank-count bitwise gate (G-L2) is lost unless the partials use association-independent summation; the
partials are a trivial share of the cost, so use reproducible (binned) summation for them.  Independently, a
row's kernel evaluated as per-rank slices moves points between Eigen's vectorized packets and the scalar tail
(`lg_functions.hpp`, the Gaussian via `.array().exp()`), which can differ in the last bit from the batch
evaluation: test `kernel_at` on a batch against an arbitrary split, bitwise, before claiming bitwise anything;
if it fails, evaluate the Gaussian with a scalar loop in the slice kernels.  State in the acceptance which
comparisons are bitwise and which are to tolerance.

## Stages

| stage | content | gain | effort (a guess from code size, not a measured velocity) |
|---|---|---|---|
| 0 | phase timers outside the row loop, per-rank maxima; the thesis measured offline | the attribution and the order of the rest | 1 day |
| 1 | reverse plan + model-down / partials-up machinery; the re-score by design blocks AND the assembly at the column owners; the energy reduce + the symmetrization variant + the verdict | the two owner-side phases that touch every window node per finalist / per entry; the wsym exchange reverses direction | 1 week (both share the machinery; the re-score first) |
| 2 | slice-wise coarsening; the raw-probe halo retired | the last O(window) owner phase; the halo traffic (~200 GB per build on albmap_40) -> cells x k | 3-4 days (no new tessellation) |

Stage 1 does NOT change B's nnz per rank: the symmetric B's row layout is a property of B and the dof
partition, and the symmetrization returns B's rows to the row owners today as well.

## Validate the thesis first, then test

Measured offline from the fit dump (coverage_balance.py in the maintainer's research notes): window nodes per
rank by row ownership against by column ownership (the plan's load), coverage per column, ranks touched per
window, and the share of a row's window in slices smaller than a coarsening's cell count.  Tests: `kernel_at`
batch vs split (bitwise); the symmetrization variant vs serial `weighted_symmetrize` on a random A; design from
partials vs `linear_cv_score` (tolerance); n = 1 column phases vs n = 1 old path on the same partition (the
association noise floor); a perverse pattern (every row covered by every rank) and a poison row that throws in
phase A, with column phases on, for hangs; lgpsf's MPI gate at n = 1 / 2 / 4 and lgpsf-hessian's tests; then a
continental build on skx-dev (16-24 nodes, albmap_40, production flags): fit wall per build, bytes on the wire,
qcE unchanged within noise.

## Not in this plan

The matrix-free application of the LG operator (Nick, 2026-10-09: every component of the input vector
evaluated at coarse bins for all ellipsoids that intersect it, the integrals with each row evaluated coarsely;
no sparse assembly, no sparse halo) -- the same column-side pattern taken all the way, a future paper; the
weighted symmetrization is its open question.  See `dev/parallel-architecture-revisit.md` for the row-and-column
ownership both would eventually want.
