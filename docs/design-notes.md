# Design notes

Running log of design decisions relevant to the eventual C++ port, recorded
as they come up during prototyping so they don't have to be re-derived later.

## Point-batch memory layout: Eigen is transposed relative to numpy

**Decision:** when a batch of `K` points in `N` dimensions is represented as
a single matrix, use shape `(K, N)` (points as rows, dimensions as columns)
in Eigen, but shape `(N, K)` (dimensions as rows, points as columns) in
numpy. These are transposes of each other -- deliberately, not a mistake to
reconcile later.

**Why:** the LG evaluation code (`eval_lg_nd`, `grad_eval_lg_nd` in
`prototype/lg_functions.py`) vectorizes over points in a
structure-of-arrays (SoA) pattern: every elementwise operation runs on "one
coordinate's value across all `K` points" as a single contiguous array. That
access pattern -- contiguous per-dimension slices -- is what should be free
under each library's *default* storage order, and the two libraries default
oppositely:

- **numpy defaults to row-major (C order).** A `(N, K)` array has each row
  (fixed dimension, all points) contiguous.
- **Eigen defaults to column-major.** A `(K, N)` matrix has each *column*
  (fixed dimension, all points) contiguous, since Eigen stores
  column-by-column. So the natural, no-copy-needed shape in Eigen is the
  transpose of numpy's: `(K, N)`, with `points.col(k)` giving a contiguous,
  zero-copy `VectorXd` of dimension `k` across all points.

Getting this backwards in either language (e.g. an `(N, K)` Eigen matrix, or
a Fortran-order numpy array to fake column contiguity) still produces
correct results -- it just means every per-dimension slice is a strided,
non-contiguous view, which is both slower and blocks SIMD vectorization of
the elementwise arithmetic.

**Consequence for the core eval/gradient functions:** they take a single
`(N, *batch_shape)` array (non-batch axes first, batch axes last -- see
"Vectorize over the batch only, arrays not tuples" below), which is exactly
the free-contiguity layout described above: slicing out one coordinate's
values across the whole batch (`u[k]`) needs no copy. An earlier version of
this note recommended `N` separate vector arguments instead of a single
array, specifically to avoid baking in a row/column convention -- that
concern turned out not to apply once the array shape itself *is* the
numpy-natural convention above, so there was nothing to protect against by
avoiding a real array. See the tuple-removal entry for the fuller reasoning
(ergonomics and internal-consistency, not a layout concern).

**Forward-looking flag, not yet a problem:** if points ever cross the
Python/C++ boundary via pybind11 (a numpy `(N, K)` array in, an Eigen
`(K, N)` matrix expected), that boundary is a transpose. pybind11's
automatic Eigen &lt;-&gt; numpy conversion needs to be told about this
explicitly, or it will silently insert a copy. Nothing to do about this yet
-- there are no bindings -- but worth remembering when that work starts.

## Vectorize over the batch only, arrays not tuples

**Decision (2026-07-23):** every point-indexed quantity (`x`, `u`, `w`, `du`,
...) is a genuine numpy array of shape `(N, *batch_shape)` -- non-batch axes
first, batch axes last, per the layout entry above. Earlier versions of
`eval_lg_nd`/`grad_eval_lg_nd` took `u` as `*u` (variadic, N separate
positional args) and `ellipsoid_transform.py` took `x` as a plain Python
tuple of N arrays -- two different, inconsistent, non-array conventions that
had crept in without ever being deliberately chosen.

**Why the switch:** checked for an actual justification and didn't find
one. Memory contiguity is identical either way (a real `(N, K)` C-order
array's rows are exactly as contiguous as N separate arrays). Every
internal loop was already written as `for ui in u: ...` / `zip(u, mono)`,
and iterating a real ndarray's leading axis gives the identical sequence of
sub-arrays that iterating a tuple does -- so switching required zero changes
to any loop body, only to the outer data structure. The tuple convention's
only cost was inconsistency across files and handing callers a Python
tuple instead of a real array when they likely have one already.

**Governing principle going forward (agreed 2026-07-23):** vectorize over
the point batch always; loops over every other axis -- monomial count,
spatial dimension `N`, theta's parameter count `P`, the Laguerre recurrence
depth `p` -- are fine, and often preferable, since they keep memory at
`O(batch)` instead of `O(axis * batch)`. Don't reach for `einsum`/broadcast
tricks over those other axes by default; a small Python loop over a
small, fixed axis is the *correct* choice here, not a compromise. See
`prototype/lg_functions.py` (loop over the monomial table) and
`prototype/ellipsoid_transform.py` (loops over `N`/`P` in the theta
pack/unpack functions) for the pattern.

## Triangular solves via explicit inverse, not substitution

**Decision (2026-07-23):** `ellipsoid_transform.py`'s pullback
(`u = L^{-1}(x - mu)`) and its derivatives never do forward/backward
substitution. Instead `L` (or `L^{-1}` where needed) is formed explicitly
once per call via `np.linalg.inv` and applied to the whole point batch as a
single `einsum` contraction (`_apply_matrix` in that file).

**Why this is free, not a tradeoff:** theta -- and hence `L` -- is never
batched (see the vectorization principle above), so `L` is always a single
`N x N` matrix with `N <= 4`. Forming its inverse is `O(N^3)`, negligible,
and done once regardless of how many points are in the batch. This replaces
a sequential N-step substitution loop with one contraction, and it made the
VJP's outer-product term (`w_L = -outer(z, u)`) collapse from a nested
Python double-loop into a single `einsum` call too.

## Dense Jacobian at the fitting-core interface, not matrix-free jvp/vjp

**Decision (2026-07-24):** the VarPro fitting core takes a full-Jacobian
callable `basis_jac(theta) -> (n_modes, P, K)` and builds its dense
`(k, P)` Levenberg-Marquardt Jacobian explicitly. It does *not* take
jvp/vjp callables, and there is no matrix-free / inexact-CG path.

**Why dense:** P (theta's parameter count) is 3..14. A matrix-free
Gauss-Newton/CG solve costs two derivative sweeps per CG iteration (one
forward, one reverse) and up to P iterations, i.e. up to ~2P sweeps per
outer step versus P for the dense build -- and gives up things LM wants:
near-free re-solves as the damping lambda is retried (dense: factor J
once, each new lambda is O(P^2); matrix-free: a fresh CG solve each
time), QR/SVD solves on J itself instead of the squared-conditioning
J^T J, and free access to J's column norms / singular values for trust-
region scaling and well-determinedness diagnostics. Matrix-free becomes
competitive when P reaches the hundreds and CG converges far short of P
iterations; the *scale* in this problem lives in the number of rows
(embarrassingly parallel), never inside one row's fit.

**Why the jac (not the jvp) is the interface:** the LM Jacobian needs all
P directions at every trial theta, and the direction-independent dominant
cost -- each mode's spatial LG gradient (Laguerre recurrence + harmonic
table) -- can only be shared *inside* the feature layer. A jvp called in
a loop over P directions recomputes all of it P times; `jac_feature`
computes it once (a ~P-fold saving on the Jacobian build). The jvp/vjp
pair remains the primitive layer underneath: `jac_feature` is tested
column-by-column against `jvp_feature`, which is itself FD- and
adjoint-verified.

**Update (2026-07-24, later the same day):** the dense-Jacobian
conclusion stands, but the "vjp is not in the fitting-core signature"
part is superseded: the default (Kaufman) Jacobian build collapses to a
single batched reverse-mode call, so `basis_vjp` is back as the *primary*
derivative callable and `basis_jac` is optional (Golub-Pereyra variant
only). See the next entry.

## Kaufman Jacobian in one reverse sweep

**Decision (2026-07-24, revising part of the previous entry):** the
default (Kaufman) reduced-residual Jacobian is built by ONE batched
reverse-mode call, not by looping the P forward directions:

    G[q, j] = sum_i c_i dPhi_hat[i, q, j] = basis_vjp(theta, w_hat)[q, j]
    with cotangent w_hat[i, j] = c_i;
    J_K = -P_perp_A ( P_perp_B ( Z_hat G^T ) ).

The (n_modes, P, K) feature-Jacobian tensor never materializes, and
n_modes and P never meet in any product.

**Why this works:** Kaufman's column q consumes dA_q c -- the mode axis
contracts against the *fixed* coefficient vector c before anything else.
sum_i c_i phi_hat_i is a single scalar-per-point function (the fitted
smooth model with c frozen), so all P components of its theta-gradient
come from one reverse sweep. The general rule: reverse mode wins when
the output side is contracted to few covectors, forward mode when the
input side is contracted to few directions; Kaufman's c-contraction puts
the Jacobian build in the gradient regime. The exact (Golub-Pereyra)
Jacobian's second term W[i, q] = sum_j dPhi_hat[i, q, j] (Z_hat^T r)_j
has no such collapse -- both the mode and theta axes stay alive, a
genuine (n_modes x P) bilinear block, for which forward mode (P sweeps,
P < n_modes) is the natural choice. Hence the fitting-core interface:
`basis_vjp` (batched, per-point -- exactly why vjp_T/vjp_feature return
unsummed (P, *batch) covectors) is required; `basis_jac` is optional,
used only by variant="golub-pereyra".

**Cost hierarchy worth remembering** (per LM iteration): exact Jacobian
= P forward sweeps; Kaufman = 1 reverse sweep + dense algebra; the cost
gradient J^T r = that same sweep contracted once more (g = -G Z_hat^T r).
Kaufman delivers full-width (approximate) Jacobian columns at gradient
price -- an AD-complexity argument for it as the default, independent of
the usual small-residual argument. Verified in
test_varpro.py::test_kaufman_reverse_mode_matches_jac_built (reverse
build vs an independent forward-built reference, machine precision).

## Probe vocabulary and the module split (2026-07-24)

**Decision:** `row_fit.py` became `probe_fit.py` (`fit_from_probes`,
`ProbeFitConfig/Result`, `spike_index`), with two pieces split out:
`init_dictionary.py` (hypothesis/init generation: ladders, window
shape, oriented sigmas, geometry, ordering) and `probe_moments.py`
(`backproject` + `raw_moments`). Two helpers were promoted downward:
`lg_functions.modes_up_to_level` (was duplicated 4x across the repo and
examples) and `whitening.whitened_basis` (the closure builder every
consumer was hand-rolling).

**Why the rename:** "row fit" misnamed the method twice over -- the
windowing is done by the caller, and the method fits ANY function known
only through inner products with random probe vectors; an operator row
is the motivating example, not the definition. "Probe" was already the
codebase's own vocabulary. Scope limit: `whitening.py`/`varpro.py` keep
their internal row_mass/"row" language (the whitening derivation in
docs/varpro-whitening-notes.tex is written in row terms); whitening.py
carries a clarifying vocabulary note instead. **Why the split:** code
outside fit_row kept re-implementing its pieces (the duplication test
for a real boundary), and each module now maps 1:1 onto a future C++
header.

## The row-fit orchestration layer: raw interface, holdout selection

**Decisions (2026-07-24, `prototype/row_fit.py`; evidence: the
frog-kernel robustness study in `docs/robust-init-notes.md` and the PIG
slice-37 refits in the research repo):**

- **Raw-data interface**: the caller supplies coordinates, lumped
  masses, raw probes/responses, `mu0`, modes; `fit_row` whitens
  internally. Masses are *routed* through this layer but all mass math
  stays in `whitening.py` -- the confinement convention gains a second
  boundary, not a second implementation.
- **One candidate grid, one selection rule (2026-07-24 refactor)**: all
  data-adjudicated choices -- init shape x scale, mode-set size, fixed
  vs released mu -- are entries of a single ordered candidate stream
  under one rule: admissibility, then score, then simplicity tie-break
  (within tie_delta of the best, prefer fewer modes, then pinned mu --
  the one-standard-error shape; subsumes the release accept-guard).
  Structural facts (window, probes, masses, diag_index) are never
  adjudicated; numerical hygiene stays in VarProOptions.
- **Score = linear-stage K-fold CV, never in-sample cost**: a
  degenerate theta can lower the in-sample cost while ruining the fit
  (observed live on the PIG release runaway: mu 3000 km off-domain with
  LOWER probe cost). theta is fit once per candidate on all equations;
  the linear coefficients are refit leave-fold-out at that theta, so
  every equation is scored out-of-sample for the linear stage at the
  price of ONE nonlinear fit per candidate (the theta stage is mildly
  optimistic -- P <= 5 on k equations, bounded by the counting rule,
  which also guarantees the folds are well-posed).
- **Early stopping = adaptive effort allocation, one-sided by design**:
  certificates fire only when the data shows the row is easy, so hard
  rows automatically buy the full grid. target_score exits everything
  once an admissible candidate is good enough (nothing else can beat a
  tie); mode_patience (>= 2: single-step worsening is noise) stops the
  nested mode ladder; candidate ORDER (sigma0 first, then window and
  circle rungs middle-out) makes the target fire fast. NO patience or
  consensus stopping on the init axis: score-vs-scale is non-unimodal
  and two inits can agree on the wrong minimum (both observed at 8:1).
  Each new mode level seeds a JITTERED warm start from the previous
  level's best (exact warm starts sit on the enrichment saddle).
- **Initial-Sigma ladder**: log-spaced circles from the local mesh
  spacing at `mu0` to a whole-batch circle, plus an optional caller
  `sigma0` rung. Too-small and too-large both fail in different ways;
  best-of over the ladder brackets every observed case.
- **Mixed ladder (window-shape rungs)**: scaled copies of the window's
  own shape join the circles -- the mass-weighted covariance of the
  batch geometry (masses ~ cell areas, so this measures the window
  REGION, not mesh density) recovers the aspect/orientation of the
  caller's conservative ellipsoid, which the circle family is blind to.
  Circles stay as shape-agnostic insurance (the window shape inherits
  the caller's prior errors; boundary-clipped windows lie).
- **Window-containment admissibility**: the conservative window bounds
  the true kernel by construction, so fits whose major semi-axis
  exceeds the window radius -- or released centers leaving the window
  -- are excluded from selection. Necessary because few-equation
  holdout scores cannot reliably reject degenerate fits (observed: a
  3000:1 needle 3x the window winning a 4-equation holdout by 0.02).
- **Fixed-mu ladder, guarded release once per rung**: pinning mu
  collapses outcome variance; a wrong `mu0` makes every pinned rung
  compensate with a distorted shape, so no single rung's release basin
  is trustworthy -- release them all, accept only if a released fit
  beats every pinned one on the holdout. The free-mu basin is narrow
  (empirically ~half the smaller kernel sigma); centers further off
  than that are the backprojection workflow's job, not LM's.
- **Backprojection init estimation is separate and caller-invoked**
  (`backproject_row` + `row_moments`): `r_hat = Z^T y / k` is unbiased
  for iid-standard-normal probes, and the RAW row values already carry
  the lumped-mass quadrature weight (`r_j = m_rho m_j phi_j`), so the
  moment estimator takes NO mass vector -- only the diagonal spike
  excluded (`diag_index`) and noise thresholds (the backprojection has
  a flat `||row||/sqrt(k)` noise floor; far-field entries otherwise
  drag the mean to the window centroid and inflate the covariance,
  while over-aggressive hard thresholds truncate the tails and shrink
  it -- `noise_mad ~ 3` on conservative windows).

## Mass matrices confined to a single layer, via noise whitening

**Decision (2026-07-24):** `M1` (row mass) and `M2` (column mass) appear in
exactly one file, `prototype/whitening.py`. Everything below it
(`lg_functions.py`, `ellipsoid_transform.py`, `lg_ellipsoid_feature.py`) is
mass-free; everything above it (the VarPro fitting core) only ever sees
already-whitened arrays and never imports or references a mass matrix.

**Why:** combining theta-dependent smooth basis functions (which live in
the discretized function space $X$, weighted by $M_2$) with theta-
independent "extra" basis functions like the diagonal spike (which live
natively in the dual space $X'$, weighted by $M_2^{-1}$) naively -- e.g.
orthogonalizing them under a plain Euclidean inner product -- happens to
work for a single one-hot spike but is not correct in general, and more
importantly requires the fitting code to reach into $M_2$ (and its
inverse) directly. Noise whitening (rescaling every basis function,
derivative, and datum once by $\sqrt{m_\rho}\,M_2^{\pm 1/2}$) converts the
whole per-row fit into an ordinary Euclidean least-squares problem, where
plain-Euclidean orthogonalization is *provably* the correct operation --
not a different, coincidentally-similar approximation to the dual-space
projection, the identical thing, verified algebraically and numerically
(`test_whitening.py`). Full derivation:
`docs/varpro-whitening-notes.tex`.

**Consequence for the derivative machinery:** the whitening operator is a
fixed (theta-independent), symmetric linear map, so it composes with the
existing JVP/VJP chain the same way Stage 1/Stage 2 composition already
does -- apply it to the raw JVP output; apply it to the incoming cotangent
before feeding the existing VJP chain. No new derivative formulas needed
anywhere for whitening itself.

## The operator layer: parametric output, free-mu storage, target-mass routing

**Decision (2026-07-24):** `prototype/operator_fit.py` implements the
agreed whole-operator design (`docs/operator-api-plan.md`): the output
is the parametric two-component object `H~ = M1 Phi~ M2 + M1 S` as
padded flat arrays, never a matrix; every matrix format is a
decompression helper (`eval_kernel`, `eval_entries`, `matvec` /
`to_linear_operator`, `assemble_sparse`, `ellipsoid_field`, `qc_map`,
`spike_measure`), each typed to the component(s) it touches. Three
implementation choices worth recording for the C++ port:

- **theta is stored in the free-mu encoding for every row**, pinned-mu
  winners included (via `release_mu`, which is exact by construction of
  the encoding). One uniform decode path (`mu0=None`) for all
  downstream consumers, at zero cost; whether mu was actually optimized
  lives in the separate `released` flag.
- **`target_mass` kwarg on `fit_from_probes`:** the row layer used to
  infer the target's mass as `m2_diag[spike_index]` -- exact only for
  the square equal-mass case. The whitened design matrix is column-
  equilibrated inside the inner solve, so the target mass rescales ONLY
  the returned `(c, s)`; theta, CV scores, and selection are invariant.
  The operator layer passes `m1_diag[rho]`, making `M1 != M2` operators
  exact with no post-correction (pinned by
  `test_fit_from_probes_target_mass_scales_coefficients` and by the
  operator recovery test, which uses `M1 != M2` throughout).
- **Baseline guard comparability:** the baseline (linear LG fit at
  `sigma[rho]`, pinned mu0, one lstsq per counting-rule-admissible mode
  set) is scored with `linear_cv_score` on the SAME folds
  (`cfg.row.cv_folds`/`seed`) and the same whitened data as the
  search's own scores, so "searched ships only if strictly better" is
  an apples-to-apples comparison; ties (e.g. a crippled search stuck at
  the baseline's own theta) ship the simpler baseline.

## ellipsoid_tree for the operator layer's geometry queries

**Decision (2026-07-24, Nick):** `operator_fit.py` uses the
`ellipsoid_tree` library (pip: `ellipsoid-tree`; the same library the
C++ port links) for its spatial queries, confined to that layer the
way scipy is -- indexing/infrastructure, never reference math; the
rest of the prototype stays pure numpy(+scipy).

- **Window derivation** is a ball query on a zero-radius `BallTree`
  over the columns (a point cloud, in ellipsoid_tree's vocabulary).
- **`assemble_sparse`** gets its whole sparsity pattern from ONE
  `collision_pairs(points_tree, ellipsoid_tree)` dual tree descent:
  every (column point, fitted tau-ellipsoid) incidence, for all rows
  at once, exactly (the library's tree pruning is conservative, the
  member tests exact -- verified index-for-index against the
  ball-prefilter + Mahalanobis reference it replaced, which
  over-fetched by ~the aspect ratio before its exact test). The
  `EllipsoidTree(mus, Sigmas, tau)` array overload consumes exactly
  what `ellipsoid_field` emits -- the integration seam the plan doc
  promised is now exercised in-repo.
- **Boundary layout:** ellipsoid_tree takes points as rows, `(K, N)`
  -- the transpose of this repo's `(N, K)` batch convention, and the
  same flip already prescribed for the Eigen side; the `.T` happens at
  the call boundary only.
- `init_dictionary.local_spacing` still uses scipy's cKDTree for its
  k-NN query (ellipsoid_tree's KDTree could take it; not worth churn
  now).
