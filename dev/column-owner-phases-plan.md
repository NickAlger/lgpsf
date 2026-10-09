# Column-owner phases for the distributed fit -- plan (draft for Nick's review, 2026-10-09)

**The problem.** In the SPMD fit (`mpi/dist_fit.hpp` -> `fit_operator`) everything a row needs is brought to
the row's owner: the halo pushes every window column's probes and mass to it (`halo_plan` / `halo_push`), and
the owner resolves the window, coarsens it, runs the baseline, searches (or delegates the search,
`row_delegate.hpp`), re-scores the finalists on the full window and assembles the row's entries.  Only the
search migrates under the row balance.  Everything else scales with the RAW window and stays home.  On the
ice-sheet application at 1.19 M rows the rank owning the widest windows holds ~100x the mean window nodes, the
fit wall tracks that rank at ~1e-5 s per window node, and the build's wall is 13-17x the mean work while the
balance's plan reports 1.10.  A node-count cap on the window is not acceptable: the window is the a-priori
ellipsoid, i.e. physics.  So the work has to move, not the window.

**The idea (Nick, 2026-10-09).** Ship the MODEL, not the data.  Every owner-side phase of a row is a sum over
the window's columns, and the columns have owners.  Give the column owners the row's geometry (centre, window
frame, coarsening frame and eps) and later its candidate models (theta, coefficients: ~50-70 doubles), and let
each of them do its share on its own columns, returning partials.  The owner keeps the row (status, model,
diagnostics, the dump), the search keeps running wherever the balance puts it, and the operator comes out the
same on the same windows.  Work then spreads by column coverage (how many windows contain a column), which is
far more uniform than row ownership; the raw-probe halo disappears.

## The phases, per row, as sums over columns

| phase today (owner) | sum over the window's columns j | what the column owner returns per (row, rank) |
|---|---|---|
| window resolution | membership `(x_j - c)^T W^-1 (x_j - c) <= 1` | nothing to send; resolved locally (the owner's dual tree on ITS columns) |
| coarsening | per cell C: `m_C = sum m_j`, `m_C x_C = sum m_j x_j`, `m_C z_C = sum m_j z_j` | cell partials: cells x (1 + N + k) doubles; plus the rank's farthest window point |
| baseline (pinned linear fit at the a-priori) | the design `Phi_qa = sum_j m_j z_qj psi_a(x_j)` | k x modes doubles (the baseline's mode set is global) |
| re-score of the finalists | `sum_j k_f(x_j) m_j z_qj` per probe q, finalist f | k x finalists doubles |
| assembly | `B_ij = m_i k_i(x_j) m_j` on window intersect the tau-ellipsoid | NOTHING: the entries stay with the column owner (column layout of B) |

## What has to change in lgpsf

**0. Instrumentation first** (`dist_fit`, `fit_operator`, a day): timers around the phases outside the row loop
(halo push, window resolution, assembly) and per-rank maxima surfaced in `DistFitResult`, so the attribution
above is measured before and after each stage.  (The bridge prints them as `[PENCIL-FIT]` lines.)

**1. The reverse plan** (`mpi/halo_exchange.hpp`, new `ReversePlan`): today's plan allgathers box-forest cuts of
each rank's FOOTPRINTS and pushes column points to the footprint owners.  The reverse: allgather each rank's
COLUMN box (one box per rank, or its box-forest cut), and let every row owner push the window ellipsoids
(centre, W: 1 + N + N(N+1)/2 doubles + the row gid) that touch a remote column box to that rank.  Each rank then
holds the foreign windows that may contain its columns and resolves membership exactly with its own dual tree
over its own columns.  The pair list (row gid, covering rank) is the communication pattern for every phase
below; it is built once per build (the windows are geometry; the rungs reuse it).

**2. A data-independent coarsening** (`coarsen_window.hpp`, new `coarsen_window_fixed`): the current 2^N-tree is
rooted on the bounding box of the points present and splits while a node holds MORE THAN ONE point, so partial
point sets on different ranks would tessellate differently.  The fixed variant: root = the window's unit ball
box `[-1, 1]^N` in whitened coordinates; split while diagonal > `eps * d` (d the whitened distance from the box
to the origin), down to a floor `h_min` passed with the frame (the row's local spacing, so the core still splits
to single points); empty children dropped.  A leaf's identity is then its path in the tree, the same on every
rank, and partials from different ranks for the same leaf add.  The protected positions (the spike = the owner's
own column) and the farthest point (the owner picks it from the ranks' farthest-point reports) are singletons
added by the owner.  Same error bound as today (the `eps * r` rule), not bit-identical to today's cells.
Hybrid rule per (row, rank) pair: if the rank's columns in the window are fewer than the cells they would fall
into, ship the points themselves (they ARE singleton cells) -- never worse than today in bytes.

**3. Column-side kernels** (small, pure): `cell_partials(frame, eps, h_min, my_window_columns, m, z)`,
`design_partials(frame, mode set, ...)`, `rescore_partials(models, ...)`, `assemble_partial(model, tau, ...)`.
All are loops over the rank's own columns inside the window with `eval_lg_basis`; the first three return small
dense blocks, the last appends to the rank's column-layout block of B.

**4. The new `dist_fit` flow**, per rung: (a) windows down the reverse plan; (b) partials up: cells (or points),
farthest points, baseline design blocks; owner sums in a FIXED order (by sending rank, then leaf) -> the coarse
batch and the baseline, bit-reproducible at a fixed partition; (c) the search as today (`fit_operator`'s phase B
on the coarse batch, the row delegate unchanged: its package is the coarse batch); (d) finalist models down,
re-score partials up, the owner decides (phase C's guard); (e) the winner down, assembly at the column owners
into the column layout.  The per-row diagnostics, status and model stay with the owner as now; the dump is
unchanged.

**5. Symmetrization and layout** (`mpi/dist_wsym.hpp`): the weighted symmetrization needs the ROW energies
`||A_i,:||^2`, which are now spread over column owners -> one reduce of partial row energies to the row owners
(one double per pair), then `dist_weighted_symmetrize` fed with the rows of A^T (= the column layout) and the
weights supplied by global id instead of computed from the local rows.  The result is B^T = B: the column layout
of the symmetric B is its row layout, so the consumers (the GLR's B matvec, the held-out QC, `lgh_fit_get_mat`)
take it as they do today, with nnz per rank now following column coverage (the GLR matvec's "nnz per rank
max/mean 41" today).

## What has to change in lgpsf-hessian (`impl/fit_impl.hpp`)

Build the reverse plan next to the halo plan (or instead of it once stage 3 lands); pass `h_min` (the row's
local spacing, which the bridge has) with the window; call the new `dist_fit`; the row-energy reduce before
`dist_weighted_symmetrize`; the column layout into `b_rowptr / b_colgids` (unchanged code, since B is
symmetric); print the new phase timers.  Options: `--pencil-column-phases 0|1` (default 0 until validated), the
hybrid threshold.  The fit dump (v6) is unchanged.

## Stages, each useful on its own

| stage | content | gain | effort |
|---|---|---|---|
| 0 | phase timers outside the row loop, per-rank maxima | the attribution, measured | 1 day |
| 1 | reverse plan + assembly at the column owners + row-energy reduce + wsym with external weights | assembly off the owner; B's nnz balanced by coverage (the GLR matvec) | ~1 week |
| 2 | baseline and re-score partials | the two remaining O(window) owner phases gone except coarsening | 2-3 days |
| 3 | `coarsen_window_fixed` + cell partials + the hybrid rule; the raw-probe halo retired | the last O(window) owner phase gone; the halo traffic (200 GB per build today) -> cells x k | ~1 week |

Expected on albmap_40 at 24 nodes: the fit wall from ~4,800 s per build (the busiest rank's 4.4e8 window
nodes x 1e-5) toward the balanced search plus coverage-balanced phases, a few hundred seconds; the halo from
~200 GB to ~20 GB per build.

## Tests and acceptance

lgpsf's MPI gate (n = 1 / 2 / 4): the new path must give the same operator as the old one up to summation
order (stage 1-2 bitwise except the row-energy reduce's order; stage 3 differs by the coarsening rule, so the
comparison is qcE and the deployed entries against the truth on the frog / heat examples); lgpsf-hessian's
tests at n = 1 / 2 / 4; then a continental build on skx-dev (16-24 nodes, albmap_40, the production flags):
fit wall per build, bytes on the wire, nnz per rank max/mean, qcE unchanged within noise.

## Not in this plan

The matrix-free application of the LG operator (Nick, 2026-10-09: every component of the input vector
evaluated at coarse bins for all ellipsoids that intersect it, the integrals with each row evaluated coarsely;
no sparse assembly, no sparse halo) -- the same column-side pattern, a future paper; the weighted
symmetrization is its open question.  See `dev/parallel-architecture-revisit.md` for the row-and-column
ownership that both would eventually want.
