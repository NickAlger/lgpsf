# Design notes

Running log of design decisions relevant to the eventual C++ port, recorded
as they come up during prototyping so they don't have to be re-derived later.

## Point-batch memory layout: coordinate-major, spelled natively in each language

**Decision:** when a batch of `K` points in `N` dimensions is represented as
a single matrix, use shape `(K, N)` (points as rows, dimensions as columns)
in Eigen, but shape `(N, K)` (dimensions as rows, points as columns) in
numpy. These are transposes of each other -- deliberately, not a mistake to
reconcile later.

**Why:** the LG evaluation code (`eval_lg_nd`, `grad_eval_lg_nd` in
`archive/python-prototype/lg_functions.py`) vectorizes over points in a
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
  transpose of numpy's: `(K, N)`, with `points.col(d)` giving a contiguous,
  zero-copy `VectorXd` of coordinate `d` across all points.

**Say it as one layout, not two shapes** (clarified 2026-07-25, after this
entry's "transpose" framing caused real confusion). The invariant is
**coordinate-major**: all `K` values of coordinate 0, then all `K` of
coordinate 1, and so on. numpy `(N,K)[d][k]` and Eigen `(K,N)(k,d)` are both
at offset `d*K + k` -- *the same bytes in the same order*, one layout spelled
natively in each language. Nothing is transposed anywhere; only the naming of
which index is called "row" differs, because the two libraries default
oppositely.

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

**Measured, 2026-07-25** (C++, `-O2 -march=native`, Eigen 3.4, K = 2000;
holds across K = 200..20000). Coordinate-major against point-major (each
point's `N` coordinates contiguous), best formulation of each:

| kernel | N=2 | N=3 | N=4 |
|---|---|---|---|
| `r^2` per point | 3.2x | 3.2x | 2.1x |
| one harmonic term | 1.9x | 1.8x | 2.0x |
| pullback GEMM | 1.3x | 1.5x | 1.4x |

Coordinate-major wins every kernel at every `N`. The gap is widest on `r^2`
because point-major turns it into `K` separate length-`N` horizontal
reductions with no SIMD across points, which is exactly the failure mode the
"vectorize over the batch only" principle is about. Also checked, and it is
NOT an inefficiency: `u.rowwise().squaredNorm()` versus an explicit loop
accumulating over coordinate columns is a wash (the loop wins 1.5x at N=2 but
loses ~1.2x at N=3, K=20000), so the readable form stays.

**The pybind11 boundary is free, not a transpose** (corrects an earlier note
here, and decision (d) of the port plan). Since the layouts are identical
bytes, a C-contiguous numpy `(N, K)` array `Map`s directly onto the Eigen
`(K, N)` matrix with **zero copies**. Bindings should therefore expose `(N,
K)` to numpy, matching the frozen prototype exactly -- which also makes the
PIG replay a drop-in. It does mean taking `py::array_t<double,
py::array::c_style>` and building the `Eigen::Map` by hand rather than
relying on pybind11's automatic Eigen caster, which would try to match shapes
rather than layouts.

**`ellipsoid_tree` uses the opposite convention:** its KDTree/BallTree take
`(dim, n)`, one point per *column* -- point-major. So M4 needs one transpose
of the mesh coordinates when building the trees. That is once per operator
fit, not per row, over `O(R*N)` doubles, so it is nothing; it just needs to
be written down rather than discovered.

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
`archive/python-prototype/lg_functions.py` (loop over the monomial table) and
`archive/python-prototype/ellipsoid_transform.py` (loops over `N`/`P` in the theta
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

**Decisions (2026-07-24, `archive/python-prototype/probe_fit.py`; evidence: the
frog-kernel robustness study in `robust-init-notes.md` and the PIG
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

> **SUPERSEDED (2026-07-29)** by "Initial guesses are data, not flags" at the
> end of this log. The two families and the `sigma0` rung are no longer
> selected by config flags: the caller passes `InitialGuess` values and the fit
> appends `num_rungs` circles. The window-shape family was measured on a real
> Hessian, never helped, and is now the opt-in `window_shape_ladder`. The
> reasoning above about what each family is blind to survives intact -- it is
> why the circles are still the default and why `oriented_ladder` exists.
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
exactly one file, `archive/python-prototype/whitening.py`. Everything below it
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

**Decision (2026-07-24):** `archive/python-prototype/operator_fit.py` implements the
agreed whole-operator design (`archive/operator-api-plan.md`): the output
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

## The mode axis is a policy (ModePolicy), not a config list

**Decision (2026-07-25, Nick):** mode growth is a pluggable strategy --
`mode_policy.ModePolicy.propose(ctx)` -- because the PIG slice-38
field runs showed the best growth ORDER is budget-dependent (complete
shells win at k=20 where the counting rule caps m at 6; the ell-capped
radial wedge ties shells at one sixth the fit cost at k=100). Points
worth recording for the C++ port:

- **Policies are stateless**: position and feedback come entirely from
  `ctx.history` -- the port is a virtual interface with no lifecycle,
  and the operator layer's baseline guard gets its a-priori sets from
  a feedback-blind replay (`baseline_sets`), never from an adaptive
  trajectory.
- **The engine keeps every guard** (counting rule, admissibility,
  patience/target, warm starts, selection); a policy only proposes.
  Contracts engine-enforced: nested growth, unique labels, a hard
  proposal cap; oversized proposals are recorded as skipped and the
  policy is re-polled.
- **Adaptive feedback is an exact projection, not a refit**: VarPro's
  inner problem is linear, so the engine hands policies a margin
  scorer computing exact one-step SSE reductions for candidate modes
  at the current winner's theta (whitened feature eval + QR-projection
  against the active design). MarginGreedy grows a downward-closed
  (p, ell) frontier on these profits behind a noise gate
  (profit > noise_gate * q/nu * ||r||^2) -- the guard against greedy
  selection amplifying small-k validation noise (the released-mu
  lesson at field scale).
- Legacy fields (`mode_levels`/`mode_sets`/`modes`) resolve to
  ShellLadder/ExplicitLadder/FixedSet; equivalence is pinned by
  fingerprint tests, so the refactor is a provable no-op for existing
  callers.

## Deployed support == fit window (the slice-38 invariant)

**Decision (2026-07-25, Nick):** the operator layer's dof-context
helpers (`eval_entries`, `matvec`, `assemble_sparse`, hence `qc_map`)
restrict each row to its FIT WINDOW, stored on `OperatorFit` as flat
CSR-style arrays (`window_indptr`/`window_indices` -- exact under
user-overridden windows, C++/MPI-friendly). ~~`eval_kernel` remains the
raw parametric smooth component at arbitrary points.~~

> **SUPERSEDED** by "The fit window is a region; truncation is the default
> everywhere" below. `eval_kernel` truncates too; the named opt-out is
> `eval_kernel_unrestricted`. The exemption recorded here caused a rogue-row
> failure at field scale, which is what forced the change.

**Why:** the windowed CV score is blind to model energy outside the
window, and the polynomial x Gaussian LG modes extrapolate violently
beyond the data support -- at PIG field scale one rogue row (level-2
coefficients cancelling in-window, 99% of predicted energy outside)
carried 94% of the whole-operator test error, and released-mu noise
chases were the same blindness. Restricting deployment to the window
makes fitted object == deployed object -- psfladder's structural
invariant -- so the per-row CV is honest for deployment by
construction. Rejected alternative: an analytic energy-containment
guard via |det L| sum c^2 (a continuum identity that fails exactly at
the coarse-mesh/boundary rows in question). Also the fast path: matvec
now costs O(sum window) instead of O(R K). assemble_sparse's tau
truncation now only trims the Gaussian tail INSIDE the window.

## Operator-layer default policy: WedgeLadder(10, 2); adaptive parked

**Decision (2026-07-25, Nick):** fit_operator with no mode source
defaults to the level-ordered ell-capped wedge -- best or tied at
every probe budget in the PIG smooth- and rough-beta benchmarks,
cheapest at large k. MarginGreedy is parked with its evidence and the
queued conditioning refinements recorded in archive/mode-policy-plan.md's
addendum (the noise gate was half its problem; the profit score's
near-redundancy normalization is the other half -- it can prefer
ill-conditioned additions, the opposite of the selection-by-
informedness motivation).

## Harmonic polynomials get their own module; the table ships sparse

**Decision (2026-07-25, Nick):** `harmonic_polynomials.py` is a new
module holding the harmonic polynomials $Y_{\ell,m}$ -- evaluation,
gradient, term list, shell dimensions -- and is **the only place the
generated table's storage format appears**. The table itself
(`lg_harmonics_table.py`) now stores each polynomial as its list of
NONZERO terms, `(exponent tuple, coefficient)` pairs, instead of a
shared monomial basis plus a dense coefficient row.

**Why a module (the structural argument).** The table stores
polynomials, but the code had no polynomial: evaluating one was an
inlined double loop, written twice, because `grad_eval_lg_nd` needs
$Y$ as well as $\nabla Y$. ~22 of that function's 39 body lines were
verbatim copies of `eval_lg_nd` (table lookup, bounds check, `r2`, the
whole `Y` accumulation, `alpha`, `norm`, `genlaguerre`, the Gaussian).
With `eval_harmonic`/`grad_harmonic` extracted -- the latter returning
`(Y, dY)` from one pass -- both functions collapse to statements of the
factorization $\psi = C_{p,\ell,N}\,Y_{\ell,m}(u)\,L_p^\alpha(r^2)
e^{-r^2/2}$ and its product rule, ~7 and ~10 lines. The confinement is
the same discipline `whitening.py` applies to $M_1/M_2$, and it scopes
every future change to the polynomial representation (term ordering,
precomputed power tables, Horner, a reduced-precision path) to one
file with its own intrinsic test suite.

**Why sparse (the representation argument).** 91.5% of the dense
coefficients (82,171 of 89,794) are exactly zero. That is not decay or
rounding: the smallest surviving |coefficient| in the whole table is
**0.225**, with zero entries anywhere below 1e-12 -- a ~15-decade gap,
so no tolerance exists to choose or get wrong. The generator's
`!= 0` drop test runs on exact `Fraction`s, and re-running the
exact-rational Gram-Schmidt confirmed zero cases of a nonzero rational
rounding to `0.0`. Two exact mechanisms produce the pattern, both
pinned as tests:

- **Exponent parity classes.** `harmonic_project` builds
  $h=\sum_k c_k |x|^{2k}\Delta^k x^\alpha$; both $\Delta$ and $|x|^2$
  shift one exponent by $\pm2$, so every term keeps $\alpha$'s parity
  vector mod 2. The Gaussian moment matrix is block diagonal across
  those classes, so Gram-Schmidt never mixes them. Measured: all 650
  basis rows have support in exactly one class, 0 violations.
- **Gram-Schmidt staircase.** Within a class the accepted vectors are
  orthogonalized in a fixed monomial order, so each vanishes on the
  leading monomial of every earlier one -- QR triangularity. Holds for
  every shell.

The root cause of the mismatch: dense-over-a-shared-basis is the
*generator's* natural representation (it is doing linear algebra on a
coefficient vector space), while a sparse term list is the
*consumer's*. The file shipped the producer's shape.

**Cost.** Evaluation speedups (500 points, all modes to level 6): 1.65x
at N=2, 3.31x at N=3, 7.51x at N=4; gradients 1.57x/3.69x/9.13x. The
benefit grows with N because the dense monomial count grows
combinatorially while the nonzero count does not.

Size: the Python file goes 592,677 -> 277,889 bytes, only 53%, which
does NOT contradict the 91.5%-of-coefficients-are-zero figure -- the
two measure different things, and the gap is worth understanding
because it is an artifact of Python source text that will not carry to
C++:

- 91.5% counts *entries*; the file counts *characters*. A dropped zero
  is `0.0, ` (5 chars); a surviving coefficient is
  `-319.817453476103, ` (20 chars). So 91.5% of entries is only 73% of
  the coefficient text: 563,029 -> 152,174 bytes.
- Per-row storage *duplicates* exponent tuples the dense form shared
  once per shell: 1,364 tuples (17,764 bytes) -> 7,623 tuples (103,830
  bytes), adding 86 KB back. Plus ~10 KB of extra per-row list
  formatting.

In binary every coefficient is 8 bytes whatever its value, so the C++
footprint tracks the entry count instead: **723 KB dense -> 90 KB
sparse, 8.0x** (parallel `double coeffs[nnz]` / `int8 exponents[nnz*N]`
arrays; the shared-exponent + int16-index variant is 81 KB, only 11%
better, and costs a gather per term). Do not reason about the C++
footprint from the Python file size.

**Verification.** Bit-for-bit, not approximate: all 89,794 densified
entries identical to the pre-change table, and `eval_lg_nd`/
`grad_eval_lg_nd` identical across 1,820 (mode, batch-shape) pairs for
N=1..4, plus `modes_up_to_level` identical over 220 configurations.
That criterion earned its keep -- the first attempt differed by 1 ULP
because the rewrite multiplied `genlaguerre` and the Gaussian
separately where the original grouped them into a `radial` intermediate
first. Floating-point association is part of the contract when the
acceptance test is exactness; the grouping was restored.

**Health of the table, checked while investigating** (all now permanent
tests): the committed float rows are harmonic to $|\Delta Y|\sim10^{-16}$
relative, orthonormal on $S^{N-1}$ to $10^{-14}$, and `num_harmonics`
matches the closed form $\dim\mathcal{H}_\ell = \binom{N+\ell-1}{N-1} -
\binom{N+\ell-3}{N-1}$ for every shell. Coefficient magnitudes *grow*
with $\ell$ rather than decaying -- for $N=2$ they are exactly the
binomial coefficients of $\mathrm{Re}(x+iy)^\ell$ over $\sqrt\pi$
(max at $\ell=10$: $252/\sqrt\pi = 142.18$) -- which is why there is no
small-magnitude tail to confuse with the structural zeros.

**The one real numerical caveat** (unchanged by this work, worth
recording): evaluating a degree-$\ell$ harmonic *in the monomial basis*
is cancellation-prone, like evaluating Chebyshev polynomials from
monomial coefficients. Measured on the unit sphere, $\sum|\text{terms}|
/ |Y|$ reaches ~$3\times10^2$ at $\ell=2$ and ~$6\times10^3$ at
$\ell=10$: about 2.5 to 3.8 digits, leaving ~12. Fine for this
application, and the module boundary is where a better-conditioned
scheme would go if a lower-precision path ever needs one.

**Behavior change:** `modes_up_to_level` used to stop silently at the
table's edge (`if (N, ell) not in TABLE: break`), quietly returning a
level-10 mode set for `max_level=15`. It now raises. Safe for every
built-in policy (all cap at `max_level=10`, and `mode_levels` is
caller-supplied with no default), but it newly rejects e.g.
`ShellLadder([15])`.

## The unit of LG evaluation is a mode SET, not a mode

**Decision (2026-07-25, Nick):** `eval_lg_basis(modes, u)` /
`grad_lg_basis(modes, u)` (and `eval_harmonic_basis` /
`grad_harmonic_basis` under them) are the production entry points;
`eval_lg_nd` / `grad_eval_lg_nd` remain as the one-at-a-time readable
reference and as the exactness oracle the batched path is pinned
against.

**Why, from the structure of the object.** The mode index factorizes:

$$\psi_{p,\ell,m} = C_{p,\ell,N}\cdot Y_{\ell,m}(u)\cdot L_p^{\alpha(\ell)}(r^2)\cdot e^{-r^2/2}$$

with the angular factor independent of $p$, the radial factor
independent of $m$, and the Gaussian independent of both. A mode set is
a product-like family drawn from two small factor sets. Passing a
FLATTENED index re-derives every factor per mode, so this is not a
caching opportunity bolted onto the evaluator -- it is the evaluator
being handed the wrong unit of work. Measured redundancy on the mode
sets a real row fit actually evaluates (not the top rung -- the ladder
spends most calls at 1-13 modes):

| factor | per-mode | factorized | |
|---|---|---|---|
| Gaussian + r^2 | 21,560 | 2,656 | 8.1x |
| harmonic evaluations | 21,560 | 10,920 | 2.0x |
| monomial terms | 24,826 | 12,849 | 1.9x |
| radial products | 21,560 | 13,741 | 1.6x |
| Laguerre steps | 13,391 | 6,953 | 1.9x |

The Gaussian dominates: its saving factor IS the average mode-set size,
and `exp` is the most expensive per-point primitive in either language.

**A structural bonus.** $\alpha(\ell)+1 = \alpha(\ell+1)$ exactly (in
floating point too -- all the intermediates are integers or halves), and
$\frac{d}{dt}L_p^\alpha = -L_{p-1}^{\alpha+1}$, so the derivative of the
radial factor at $\ell$ is an entry of the table for $\ell+1$. ONE
triangular family of Laguerre tables supplies every value and every
derivative in the mode set; the one-at-a-time path ran a second
recurrence per mode.

**No new object in any signature.** A batched function is stateless: it
computes shared intermediates in locals and discards them. There is no
prepared-point-batch to construct, thread through calls, keep consistent
with `u`, or invalidate -- points stay plain arrays everywhere, and the
change is confined to `harmonic_polynomials.py`, `lg_functions.py` and
the four `lg_ellipsoid_feature.py` wrappers.

**Gradients are returned per-mode and uncontracted.** The three
consumers contract them differently (`vjp_feature` against a cotangent,
`jvp_feature` against `du`, `jac_feature` against the theta Jacobian),
and -- decisive -- the exact Golub-Pereyra VarPro variant needs the
uncontracted $(n_{modes}, P, K)$ tensor, since its second term has no
reverse-mode collapse. Contracting inside the LG layer would have
foreclosed the exact reduced Gauss-Newton Hessian. The LG layer knows
nothing about theta or cotangents; that is `lg_ellipsoid_feature`'s job.

**Where the time was, measured before deciding.** cProfile on a
representative row fit (K=400 window points, k=100 probes, the
`WedgeLadder(10, 2)` default): `eval_lg_nd` + `grad_eval_lg_nd` were
~47% of cumulative time, dense linear algebra only ~7.5% (`svd` +
`inv` + `lstsq`), the pullback `eval_T` just 2.4%. The fit is
basis-evaluation-bound, not SVD-bound. Also found: `u[k]**0`
materialized an array of ones 26,490 times, 3.2% of the whole fit, pure
waste -- the batched path skips zero exponents instead.

**Result:** end-to-end row fit 2.55s -> 1.85s (**1.38x**), LG share
47% -> 35%, Python function calls 1.09M -> 0.90M, with the fit returning
bit-identical theta, c, s and score.

**Verification.** Bit-for-bit again: 168 (mode set, batch) pairs for
`eval_lg_basis`/`grad_lg_basis` vs the one-at-a-time path, and 224
feature-layer calls across ALL FOUR functions (eval/jvp/jac/vjp, free
and fixed mu) vs the snapshotted per-mode implementations. jvp_feature
and jac_feature get zero calls in a real fit, so without that they would
have been covered only by tolerance-based tests. The exactness claim is
kept as a permanent test (batched == one-at-a-time), not just
scaffolding: a tolerance there would hide precisely the drift the test
exists to catch.

**Not done here, taken up separately:** `basis_eval` and `basis_vjp` are
called at the SAME theta each LM iteration (1,528 and 1,128 calls in the
profiled fit) and share the pullback, r^2, the Gaussian, every harmonic
and the whole Laguerre table. Capturing that requires changing the
fitting core's basis-callable contract.

## The VarPro VJP regroups by shell, not by mode

**Decision (2026-07-25, Nick):** `vjp_lg_basis(modes, u, w)` -- the
u-gradient of the pointwise-weighted sum `sum_i w_i psi_i` -- computes
that sum by collecting terms per SHELL rather than forming each mode's
gradient and adding them. Substituting the product rule into the sum and
collecting by shell $s$:

$$\sum_i w_i\nabla\psi_i = g\Big[\sum_s A_s\nabla Y_s - u\sum_s Y_s B_s\Big],\quad
A_s=\sum_{i\in s} C_i w_i L_{p_i},\ \ B_s=\sum_{i\in s} C_i w_i(L_{p_i}-2L'_{p_i})$$

$A_s$ and $B_s$ are scalar fields, so only the harmonic gradients carry
the leading $N$ axis: one $(N, K)$ multiply-add per SHELL instead of
several per MODE, the $u$ term collapses to a single multiply for the
whole sum, and the $(n_{modes}, N, K)$ tensor is never materialized.

This is the natural form of the object: the gradient of a weighted sum
of modes IS a per-shell quantity. `grad_lg_basis` stays for consumers
needing the per-mode tensor -- `jvp_feature`, `jac_feature`, and
decisively the exact Golub-Pereyra VarPro variant, whose second term has
no reverse-mode collapse. Only `vjp_feature` changed, so the entire
Golub-Pereyra path is provably untouched (verified bit-identical over 60
calls).

**Two claims that did NOT survive measurement, recorded so they are not
re-argued:**

- **"Better numerical properties" is false.** Against an independently
  written 80-bit extended-precision reference, the regrouped and
  per-mode forms sit at the same ~1.5 ulp: median relative error
  3.26e-16 vs 3.22e-16, and neither wins consistently (11/8/1 over a
  sweep of mode sets and batch sizes). The regrouping does not remove
  roundings, it relocates them into the $A_s$/$B_s$ accumulation. The
  reference was written from the formulas rather than by running
  `grad_lg_basis` at higher precision, so it cannot share a mistake with
  either path under test.
- **The end-to-end Python win is ~nothing: 1.02x.** In isolation the VJP
  is 1.55-2.35x faster (best when shells << modes: N=3 wedge, 46 modes /
  9 shells -> 2.35x), but a real fit lives at 1-13 modes with ~5 shells,
  and Python is dispatch-bound (see the overhead entry). The
  microbenchmark's top rungs are not where the fit spends its time.

**Kept anyway, on C++ grounds**: C++ is arithmetic-bound, where the
1.55-2.35x is real rather than diluted; the avoided
$(n_{modes}, N, K)$ allocation matters more in a threaded per-row loop;
and the prototype's job is to be the reference implementation for the
port. The price is the bit-identity certification -- this path is now
pinned by a ~1e-15 tolerance test plus adjoint consistency
(`<w, J du> == <vjp(w), du>`, worst gap 2.5e-14) rather than by exact
equality.

## The fitting core takes ONE basis callable, per theta, not three

**Decision (2026-07-25, Nick):** `fit_varpro(z_hat, y_hat, basis,
theta_init, e_hat=None, options=...)` -- where `basis(theta)` returns an
object exposing `values()`, `vjp(w_hat)` and optionally `jac()` --
replaces the previous `basis_eval` / `basis_vjp` / `basis_jac` triple.
The object is built once per trial theta and held in the same one-entry
cache as the inner solve, so the values and whatever derivative the step
needs share one evaluation. Three layers mirror the three modules:
`LGBasisAt` (r^2, the Gaussian, the shell index, harmonics, the Laguerre
tables), `FeatureAt` (adds the pullback and the theta chain rule), and
`WhitenedBasisAt` (adds the mass scaling, keeping masses confined). The
free functions become thin wrappers over them, so each formula exists
once.

**Why an object and not a value-and-gradient call.** The usual
`value_and_grad` pattern does not apply: the Kaufman cotangent is
`w_hat[i, j] = c_i`, and `c` comes from the inner linear solve, which
needs the values. So the sequence is values -> solve -> derivative, with
caller work in between; there is nothing to fuse. What IS shared is a
partial evaluation at a fixed theta, and an object with an explicit
lifetime is the honest way to name that. A hidden one-entry cache behind
the old triple would also have worked and needed no signature change --
rejected because the C++ port hand-rolls LM (M2), where the loop
naturally evaluates the residual at trial thetas and residual+Jacobian
at accepted ones; the contract should be shaped for that call pattern,
with scipy's separate `fun`/`jac` the thing that bends.

**Capability, not optionality.** "Golub-Pereyra needs `basis_jac`"
became "Golub-Pereyra needs an evaluation providing `jac()`", checked
once at `fit_varpro` entry. `test_varpro.py` now builds a deliberately
minimal `values()`-and-`vjp()`-only basis to pin that the default
Kaufman path never reaches for more -- a sharper test than passing
`basis_jac=None` was.

**Also closed while here:** the gradient pass used to recompute a
bit-identical `Y` when `values()` had already run -- i.e. exactly the
VarPro order. `harmonic_basis(..., values=Y)` skips the value
accumulation in that case.

**Result: 1.04x on an end-to-end row fit** (2.49s -> 2.40s, stable over
repeats), with theta, c, s and score **bit-for-bit identical** -- the
acceptance criterion for a pure restructuring, verified by running the
same fit in the pre- and post-change trees in separate subprocesses, plus
all four feature functions across N and both mu encodings.

The small win is the expected one, not a disappointment: this layer is
dispatch-bound (see the overhead entry), the LG evaluation is ~35% of the
fit, and B removes duplicated *arithmetic* -- one pullback, r^2, the
Gaussian, the harmonics and the value-side Laguerre tables per accepted
step -- rather than duplicated numpy calls. In C++, where arithmetic is
what costs, the same removal is worth proportionally more. B was taken
for the contract, not the clock; it must be settled before `varpro.hpp`
freezes its interface, which is why it was done in the prototype now
rather than discovered during M2.

## Two parameter encodings: public `theta`, internal `theta_hat` (2026-07-25, C++)

**Decision** (`include/lgpsf/ellipsoid_transform.hpp`, M1). The ellipsoid
parameters exist in two encodings, and the C++ side names them apart:

| | contents | length | decodes standalone? |
|---|---|---|---|
| `theta` -- public | `[mu, log-diag(L), strict-lower(L)]` | always `N(N+3)/2` | **yes** |
| `theta_hat` -- internal | `[delta, log-diag, strict-lower]`, or `[log-diag, strict-lower]` when mu is pinned | mode-dependent | no (needs `mu0`, `MuMode`) |

with `mu = mu0 + delta` when fitted and `mu = mu0` when pinned. `to_theta` /
`to_theta_hat` convert; `unpack_theta(theta)` needs nothing but the vector.

**Why the public one must not depend on `mu0`:** `fit_operator(mu0=None)`
defaults each row's reference center to its own source point, and that is the
common case. If the returned `theta` were mu0-relative, an `OperatorFit`'s
theta array would be meaningless without also carrying the centers, and rows
would not decode uniformly. So everything handed back to a caller --
`OperatorFit.theta`, `ProbeFitResult.theta` -- is absolute, exactly as the
prototype already did.

**Why the internal one is a displacement:** absolute centers carry the mesh's
physical coordinates (order 1e6 m for the ice-sheet problem this came from)
against a log-diagonal of order 1 -- six orders of magnitude in one vector
that the trust region and `xtol` both act on. `x_scale='jac'` mitigates that
but does not remove it. Displacements are of order the local ellipsoid, like
every other entry.

**Why pinning drops the block rather than freezing it:** the alternative --
one always-fitted encoding plus an optimizer active set -- was considered and
rejected. Pinned mu is the operator-layer default, so it would compute `N`
useless Jacobian columns per LM iteration, and those columns are *identically
zero*, which is what MINPACK's `x_scale='jac'` column-norm scaling divides
by. The mask would have to be real anyway, so two encodings is strictly
simpler.

**What it costs: nothing in derivative math.** The center enters as a
translation, so `d(mu)/d(theta)` is the identity in both encodings and
`jvp_unpack_theta_hat` / `vjp_unpack_theta_hat` are the same code either way.
No new rules, no new derivative tests.

**What it costs numerically: nothing the absolute encoding didn't already.**
The round trip is not bit-exact -- `(mu0 + delta) - mu0` recovers `delta` only
to the resolution of `mu` -- but that is precisely the resolution at which the
absolute encoding stores `mu` in the first place. Pinned as a test
(`test_ellipsoid_transform.cpp`, "the two encodings describe the same
ellipsoid"): the center round-trips to within `eps * |mu|` and the L block
round-trips exactly.

**Naming.** `_hat` already marks "in the coordinates the numerical core works
in" (`z_hat`, `y_hat`, `e_hat`, `phi_hat` -- see the whitening entry), so
`varpro.hpp` gets a clean invariant: everything crossing into the fitting core
is hatted, and the conversions back out are confined to its boundary. The
transform differs -- mass whitening there, an mu0-shift and mode reduction
here -- but the role is identical.

**Consequence for signatures:** there is no `N` parameter and no
`std::optional<VectorXd> mu0` anywhere in the module (the port plan had
proposed the latter). `mu0` is required and always present -- the fitting
layers already require it, since it sets the init ladder's scales and seeds
every candidate -- so `N` is `mu0.size()`, and the fit/pin switch is a
two-valued `enum class MuMode` rather than a sentinel smuggled through an
array slot. `release_mu` and `freeze_mu` survive but lose their `mu0`
arguments: releasing is "prepend `N` zeros", freezing is "split off the mu
block".

## The hand-rolled LM: an SVD hook step, not a MINPACK transcription (2026-07-25, C++)

**Decision** (`include/lgpsf/detail/levenberg_marquardt.hpp`, M2). The outer
loop is the one numeric the prototype delegates, so it is the only piece with
no reference to port. It reproduces MINPACK's *semantics* -- the trust-region
update rule and constants, `x_scale='jac'` column scaling with the monotone-max
update, the exact ftol/xtol/gtol test forms, `factor = 100`, and the
one-evaluation-returns-x0 rule -- but solves the trust-region subproblem
differently.

MINPACK's `lmpar` repeatedly QR-factorizes `[J; sqrt(par) D]` inside a root
find. That machinery exists because MINPACK targets large parameter counts.
Here `P <= 14` and the residual length is tens, so ONE SVD of the scaled
Jacobian serves every trial value of the Levenberg parameter: with
`J D^-1 = U S V^T` and `g = U^T f`,

    p(lambda) = -D^-1 V diag( s_i / (s_i^2 + lambda) ) g,
    ||D p(lambda)||^2 = sum_i ( s_i g_i / (s_i^2 + lambda) )^2,

so evaluating the constraint at any lambda costs O(P) and its derivative is
closed-form. Newton on More's nearly-linear `1/||Dp|| - 1/delta`, safeguarded
by bisection, then lands on the boundary in two or three iterations.

This is the same EXACT subproblem solution lmpar computes, not an
approximation to it, and rank deficiency needs no special case: zero singular
values contribute nothing to `p(lambda)` by construction.

**Measured against MINPACK** (scipy `least_squares(method='lm', x_scale='jac')`,
same problems, same starts):

| | MINPACK | this loop |
|---|---|---|
| Rosenbrock, from (-1.2, 1) | 21 evaluations, cost 0 | 17 evaluations, cost 0 |
| Powell singular, from (3,-1,0,1) | 53 evaluations, cost 2.0e-61 | 51 evaluations, cost 1.3e-58 |

Same regime, marginally fewer evaluations, identical answers. Powell's singular
function is the one that matters: its Jacobian is singular AT the solution, and
the minimum-norm Gauss-Newton step walks into that null space cleanly rather
than damping its way around it.

**Tested in isolation, not through VarPro.** A failure in a fit could otherwise
not be told apart from a failure in the Jacobian, and the Jacobian is the
subtler of the two. So the permanent tests use problems whose answers come from
outside this repository entirely -- a linear least-squares problem whose
minimizer an independent decomposition supplies (and whose residual is nonzero,
so the ftol path is exercised), plus the two classical problems above -- and
the hook step, being the part that is not a transcription, is additionally
pinned against its own defining conditions: the step lands on the trust-region
boundary and satisfies `(J^T J + lambda D^2) p = -J^T f` for the lambda
reported.

**One thing the tests caught.** The claim that "the Jacobian is evaluated once
per outer iteration, so the callback needs no de-duplication" is true but
insufficient: the loop can satisfy its convergence test IMMEDIATELY after
accepting a step and return without ever evaluating the Jacobian there, so the
final point would never be reported. `fit_varpro` closes the trace explicitly
at the returned point. The prototype's own de-duplication workaround -- against
the extra Jacobian evaluations scipy inserts at arbitrary points -- is indeed
unnecessary here; this is a different gap.

## The operator window: one aspect-ratio cap spanning ball to ellipsoid (2026-07-26, C++)

**Decision** (Nick, `include/lgpsf/operator_fit.hpp`, M4). The fit window is
the caller's `sigma[rho]` inflated by `tau_window`, with its axis ratio capped
by ONE CONTINUOUS KNOB, `window_aspect_cap`, which floors sigma's eigenvalues
at `lambda_max / cap^2`:

| cap | window |
|---|---|
| 1 | isotropic -- the ellipsoid's bounding sphere at the same tau (the ball) |
| kappa | axis ratio `min(the prior's, kappa)` |
| infinity (**default**) | the caller's ellipsoid, untouched |

The two endpoints are exactly the ball and the ellipsoid, so this REPLACES what
began as a boolean and strictly extends it. The query is always an ellipsoid
and only `A` changes, so the cap moves the SHAPE and never the scale --
`tau_window` means the same thing at every setting, which is what makes the
ball-vs-ellipsoid comparison a one-variable experiment. The eigenvectors are
untouched, so orientation always survives (undefined at cap = 1, where the
result is isotropic -- a fact a test asserted before it was true).

What the knob trades is how much trust goes on the prior's ORIENTATION. A
sphere is conservative in every direction whether or not the prior points the
right way; the caller's ellipsoid is conservative only along the axes the prior
nominates. At cap kappa the window still extends `tau * a_max / kappa` in its
narrowest direction, bounding the damage from a badly rotated prior while
taking most of the point-count saving. (Flooring eigenvalues to bound an aspect
ratio is the same device `window_shape` already uses on degenerate windows.)

Measured, 3:1 prior, tau = 1.5, window points at cap 1 / 1.5 / 3 / infinity:
**363 / 331 / 205 / 205** -- monotone, and capping AT the prior's own aspect is
correctly a no-op.

**The archaeology, because the evidence is split.** The row layer was designed
on the ellipsoid premise -- `8e64243` justifies the window-shape init family
with "the caller's window selection is information already paid for (**in the
intended pipeline it IS a conservatively inflated ellipsoid, whose aspect and
orientation survive the inflation**), and orientation is precisely the circle
family's blind spot (the 8:1 frog study)". Four hours later `cfcc6f1`, the
operator-API design doc, specified "window = kd-ball of radius tau_window *
(largest 1-sigma axis)", and `1ea8d25` implemented it. The operator doc never
notes that it contradicts the premise the row layer's init family rests on.
Different sessions, different contexts, nobody caught it at the time.

**Why it matters, measured.** A ball window's mass-weighted covariance is
isotropic by construction, so `window_shape` recovers nothing from it: on a
uniform grid with an 8:1 prior, a ball window gives aspect **1.000** while the
prior's own ellipse gives **8.016** (true 8.0), with 8x fewer points. The
window-shape rungs therefore duplicate the circle rungs under a ball window --
and indeed every PIG operator run disables them explicitly
(the slice-38 and slice-39 operator drivers in the glaciology repo).

**Why it is not a simple fix.** The ball is what every field-scale experiment
validating this method actually ran, and it is the convention inherited from
the pre-lgpsf research pipeline (`slice37`: "Window and validation conventions
are slice 36's, verbatim"). It is also the orientation-agnostic choice:
conservatism enters as one scalar applied isotropically, so the admissibility
guard's validity does not additionally depend on the prior's orientation being
right. And since deployed support == fit window, the window is also the
deployed operator's truncation -- `slice38` measured that error floor directly.
So: **the ball is known to perform well, and the ellipsoid is not yet measured
at field scale.**

**The resolution.** Default to the intended ellipsoid (`cap = infinity`), reach
the validated ball at `cap = 1`, and settle it by experiment once the C++ port
makes the PIG replay cheap -- with the intermediate settings available, which a
boolean would not have given. Neither endpoint is settled until then.

**Implementation point.** Windows for ALL rows come from ONE dual-tree descent
(column-point tree against a tree of every row's query ellipsoid), not a query
per row, matching how the deployed sparsity pattern is already derived.

**Testing note worth keeping.** The "bit-identical across thread counts" check
first failed for a reason that had nothing to do with threads: the output
arrays are NaN-padded for rows with no shipped model, and `NaN != NaN`, so
Eigen's `operator==` reported a difference between a run and ITSELF. The
integer and status arrays passed precisely because they carry no NaN. Compare
NaN-padded arrays with a NaN-aware helper.

## The counting rule counts what is actually fit (2026-07-26, C++)

**Decision** (Nick). `k >= 2 (m + n_extra + P)` uses the parameter count of the
fit it guards: `N(N+1)/2` in the pinned encoding, `N(N+3)/2` with the center
fitted. The baseline guard is a linear fit in the pinned encoding, so it counts
the former; the search counts its own stream encoding.

**The prototype had this wrong**, and it is an unintended slip rather than a
choice. `operator_fit.py:330` reads

    P_fix = N + N * (N - 1) // 2 + N            # == theta_size(N, mu0=any)

whose comment asserts the expression is encoding-independent. It is not -- it
is the FREE count (5 vs 3 at N=2, 9 vs 6 at N=3, 14 vs 10 at N=4), so
`P_fix == P_free`, `P_stream` was a no-op, and the baseline guard was held to a
stricter budget than the search it guards. Uncaught because the error is
conservative: a too-large P only ever skips more mode levels, never admits one
it should not -- the same reason the ball window went unnoticed.

Worth noting for the C++ side too: an earlier version of `probe_fit.hpp` copied
the prototype's pinned-count-regardless-of-policy behavior and justified it as
keeping pinned and released candidates comparable. That justification was
post-hoc rationalization of a bug, which is a good reminder that a plausible
reason can always be found for whatever the code already does.

## The fit window is a region, and truncation is the default everywhere (2026-07-26, C++)

**Decision** (Nick, M4). Three connected changes to the operator layer.

**1. Truncation to the fit window is the DEFAULT for every evaluation**,
`eval_kernel` included; the untruncated parametric form is reachable only
through `eval_kernel_unrestricted`.

The reason is stronger than "out-of-window values are unverified". The fit's
objective evaluates the LG basis ONLY on the window, so out-of-window model
mass is **unpenalized**: the optimizer will spend it to chase in-window noise,
and on problematic rows (small or noisy entries) it does. That is the slice-38
failure -- one rogue row carried 94% of a whole-operator test error, and
truncating deployment to the window fixed it completely. So the extension of
the fitted form beyond the window is an artifact of an objective that could not
see it, not a kernel that merely happens to be known best near the window.
Calling it "component access" relabels the hazard rather than removing it.

**2. The window is stored as a REGION** (`window_center`,
`window_covariance`, pre-scaled so membership is Mahalanobis <= 1), because
`eval_kernel` answers at arbitrary coordinates and an index list cannot
restrict a point that is not a mesh column. The CSR index list stays as the
derived cache and the fast path. A test pins that the two agree on every mesh
column.

Caveat worth keeping: the ellipsoid is a proxy for the window POINT SET. The
fit saw points; the ellipsoid is the region containing exactly those mesh
points, so the two agree on mesh columns exactly, but off-mesh on a coarse mesh
the ellipsoid extends past the outermost window point and a query there is
still extrapolation -- bounded, but extrapolation.

**3. The `windows=` index-list override is gone**, replaced by an optional
per-row window ELLIPSOID. It was used nowhere in the PIG experiments and only
in one prototype test. Removing it makes the region invariant unconditional --
no NaN rows, no throw-versus-fallback decision, no special case threaded
through the helpers. `tau_window` and `window_aspect_cap` do not apply to an
override: they derive a window from a best guess, and an override already is
the answer.

`init_dictionary.hpp` gained `ellipsoid_from_points`, which converts a
hand-picked index set into an admissible window: the mass-weighted mean and
covariance, scaled so the farthest point sits exactly on the boundary, giving
exact containment in closed form. Not the minimum-volume enclosing ellipsoid,
which is tighter but iterative.

What that costs: an exactly non-ellipsoidal window (a mesh-connectivity patch,
a boundary-clipped region) can no longer be expressed. Through the helper it
becomes a SUPERSET -- other columns inside the ellipsoid come along. Measured
on a 40-point hand-picked set: 42 columns realized. That is conservative in the
direction that matters, since the hazard is mass outside the window, but a
caller who wanted a tight irregular region gets a rounder, larger one.

**One rule, four helpers.** `support = fit window ∩ fitted tau-ellipsoid`, with
`truncation_tau` defaulting to infinity. `eval_kernel` applies the window as a
region, `eval_entries` and `matvec` by index, and `assemble_sparse(fit, tau)`
is that rule with a finite tau -- now pinned by a test asserting its stored
entries equal `eval_entries` at the same tau, where before nothing connected
them. Measured trim on one row: nonzeros 1 / 7 / 26 / 37 / 37 at tau
0.5 / 1 / 2 / 4 / infinity.

## FittedOperator and FitDiagnostics are separate types (2026-07-26, C++)

**Decision** (Nick, M4). `fit_operator` returns `OperatorFit { FittedOperator
model; FitDiagnostics diagnostics; }` instead of one struct holding both.

**The cut was already there; the type just failed to express it.** Counting
field accesses across the evaluation helpers:

| touched by evaluation | never touched |
|---|---|
| `dim`, `s`, `m1_diag`, `spike`, `x_cols`, `x_rows`, `m2_diag`, `theta`, `mu`, `L`, `c`, `mode_set_id`, `mode_sets`, `window_*` | `score`, `baseline_score`, `stop_reason`, `released`, `failures`, `config` -- **zero readers each** |

The only apparent straddler was `status`, read solely by `model_rows` as the
"does this row have a model" predicate. `mode_set_id >= 0` encodes exactly
that -- the gather sets it only for `Fit` and `FallbackBaseline` rows -- and
`eval_entries` already used that form. `model_rows` now does too, and a test
pins the equivalence so the two cannot drift.

**Why.** The strongest reason is constructibility, and it was already being
paid: the slice-38 driver builds an OperatorFit by hand to merge
per-chunk fits and must invent `stop_reason=[""]`, `status=["gated_out"]`,
`score=nan`, `baseline_score=nan`, `failures={}`. Someone merging chunks,
loading from disk, or building an operator from a different method should not
have to fabricate a stop reason. Beyond that: the design doc defines the fitted
object as `H~ = M1 Phi~ M2 + M1 S`, a mathematical object that scores and stop
reasons are not part of; and the two have different lifetimes, the model being
the thing worth persisting while diagnostics are session artifacts.

**Shape.** A named struct with two members rather than a tuple (tuples lose the
names) and rather than model-as-a-slot-with-diagnostics-loose (which leaves the
container half-encapsulated, the asymmetry being removed). Structured bindings
still work. `num_rows`, `num_cols`, `row_window`, `row_modes` and the new
`has_model` moved onto `FittedOperator`, being model queries.

`FittedOperator` keeps its own copies of `x_cols`, `m1_diag`, `m2_diag` so it
is self-contained and serializable alone -- ~24 MB duplicated at PIG scale
(K ~ 1e6, N = 2), which buys not having to keep caller arrays alive alongside a
saved operator.

Behaviour preservation was the acceptance criterion: every pre-existing
assertion passed unchanged after the split.

## LGOperator is a data structure, not a fit result (2026-07-26, C++)

**Decision** (Nick, M4). `FittedOperator` became **`LGOperator`** and moved to
its own header, `lg_operator.hpp`; `operator_fit.hpp` is now one PRODUCER of
that structure rather than its definition.

**The name.** "FittedOperator" said how it was made rather than what it is, and
implied a provenance that is not required: a caller with a physics-based
approximation can fill one in directly, bypassing the fitter entirely.
`LGOperator` says what it is, and `LG` matches the house casing (`LGBasisAt`,
`eval_lg_basis`).

**The header split, measured before deciding.** The evaluation helpers
reference NOTHING from the fitting stack -- zero mentions of whitening, varpro,
mode_policy, probe_fit or init_dictionary across the whole helper section. So
`lg_operator.hpp` needs only `lg_ellipsoid_feature`, `ellipsoid_transform`,
`lg_functions`, Eigen and ellipsoid_tree. A consumer who wants to APPLY or
ASSEMBLE an operator no longer includes the machinery that fits one, and the
claim "you can build one without our fitter" is structurally true rather than
documented. Behaviour preservation was the acceptance criterion and it held:
every assertion passed unchanged across the move.

**Free functions, not methods.** Considered and rejected. Nothing in the helper
set needs privileged access; the type is a transparent flat-array structure
meant to be inspected, gathered across MPI ranks, merged and serialized, so
private fields would fight the property the split was for; and a user's own
`foo(const LGOperator&)` should be a peer of ours. The line drawn, and written
into the header: **methods for "what am I"** (`num_rows`, `has_model`,
`row_window`, `row_modes`), **free functions for "what can be done with me"**.

**Three additions the split made natural.**

`num_threads` on the row-parallel helpers, honouring the plan's uniform
trailing-parameter convention that `matvec`, `assemble_sparse` and `qc_map`
had been ignoring. `matvec` needs no reduction, since rows write disjoint
output rows; `assemble_sparse` fills per-row triplet buffers and concatenates
them in ROW ORDER, so the pattern and the summation order are independent of
scheduling and the result is bit-identical across thread counts.

`validate`, returning one message per structural problem. Hand-construction is
now a supported path, so it needs to be checkable: mode-set ids naming nothing,
coefficients too narrow for a row's mode set, non-monotone window offsets,
out-of-range window indices, a wrong theta encoding width, a spike alongside
separate row coordinates. It checks SHAPE, not approximation quality -- that is
what `qc_map` is for.

`concatenate_rows`, the operation that driver open-codes. Its
reason to exist is that merging means remapping every row-indexed array at once
-- `mode_set_id` into a combined mode-set table (de-duplicated, so ids stay
dense) and `window_indptr` by a running offset -- and getting either wrong is
silent.

**Testing note.** `test_lg_operator.cpp` never includes `operator_fit.hpp` and
never calls the fitter: every operator in it is written down by hand. That is
what makes the independence a property of the build rather than a claim in a
comment.

## LGExpansion: the row model as its own type (2026-07-26, C++)

**Decision** (Nick, M4). `CandidateFit` and `ProbeFitResult` were the same
model-plus-diagnostics mixture `OperatorFit` had been, one layer down. The
model half is now `LGExpansion` (`lg_expansion.hpp`).

**Two things the split closed that were not on the list.** `released` was not a
diagnostic: it told you whether `theta_hat` was Pinned- or Fitted-encoded, so
it was needed to DECODE the candidate. Storing the absolute `theta`, as
`LGOperator` does, makes the model self-decoding and drops `released` to
provenance. And a `CandidateFit` was not self-describing -- it stored
`num_modes` and `modes_label` but not the mode list, which lived in the
engine's `modes_of` map, so a candidate's model could not be evaluated from the
candidate.

**The spike.** Held inside the expansion, not beside it. The tempting argument
for excluding it -- `s` weights a discrete dof-tied correction, not a function
of `x` -- loses to the fact that `c` and `s` are FITTED JOINTLY: the projection
in variable projection couples them, so separating them at the type level would
imply an independence the mathematics denies. `LGOperator` holds both for the
same reason. The distinction is carried by the documentation rather than the
type system.

One asymmetry recorded so nobody "fixes" it: `c` is self-describing because its
mode list travels with it, while `s` is meaningless without the caller's extra
basis, which the expansion does NOT carry. A mode list is tens of integer
triples; an extra basis is a `(K, num_extra)` array the caller owns, and
storing it would give every candidate of every row a copy of the batch.

**The name.** Not `LGFunction`, which reads as a single mode. "Expansion" says
the informative thing -- a sum over a basis, not a basis element -- and does
not over-commit the way `LGKernel` would, since `probe_fit` is deliberately
general about what is being fit.

**What it bought beyond tidiness.** An `LGOperator` row IS an `LGExpansion`
plus a window plus the masses, so that relationship became expressible:
`build_operator` assembles an operator from per-row expansions, doing the
bookkeeping that makes filling the flat arrays by hand error-prone, and
`row_expansion` extracts one back out. That is the ergonomic form of the
physics-based construction path -- previously it meant filling twelve parallel
arrays consistently, which is why `validate` had to exist.

**A bug the round-trip test caught immediately.** The first `build_operator`
skipped unmodeled rows without carrying the CSR window offset forward, leaving
holes in the running total, and the patch loop written to compensate was
convoluted rather than correct. Carrying the offset forward before the skip is
the fix. This is precisely the bookkeeping the function exists to own, which is
an argument for it existing rather than against.

## The C++ port is 8.4x the prototype single-threaded, 25x on four cores (2026-07-26)

Measured at PIG scale on identical data -- 6561 columns, 100 probes, 100 rows,
slice38's configuration, both running the BALL window so the work is the same
(mean window 283 points, 100/100 rows shipped, either way). Full numbers in the
maintainer-local `dev/timing-2026-07-26.md`.

| | per row | vs prototype |
|---|---|---|
| Python prototype | 241.3 ms | 1x |
| C++, 1 thread | 28.6 ms | 8.4x |
| C++, 4 threads | 9.6 ms | 25.3x |

Threading scales 1.92x on 2 and 3.00x on 4. The ellipsoid window (the C++
default) is a further 1.6x, being roughly half the points -- less work rather
than faster work.

For the replay this is the difference between ~26 min and ~63 s for a
whole-field fit, so a five-setting `window_aspect_cap` sweep costs about five
minutes.

**Where the time goes.** One row, 314-point window, 17 candidates: 25.6 ms.
`fit_varpro` is 840 us per candidate and `linear_cv_score` 219 us, so ~70% of a
row is the outer LM loop and ~15% is cross-validation. Within the LM, one
iteration costs `values()` (13.0 us) plus `vjp()` (25.5 us), and ~20 iterations
of that is ~770 us of the 840 -- ~~**the LM is essentially all basis
evaluation**~~, which is what the design predicted and what the prototype
measured in Python (47% there).

> **SUPERSEDED.** That held only at the SIX modes this run used. At
> production mode counts the fit is SVD-bound -- 61% of instructions at
> k = 100 -- which is what the QR-first inner solve came from. See "The fit
> is SVD-bound, not basis-bound" below, and
> `experiments/inner-solve-profile.md`. A profile taken at the wrong size
> answers the wrong question.

Note this vindicates the standing warning at the top of the port plan in a
direction worth recording: the Python measurement said the fit was
basis-bound, and it still is in C++, but the SHARE went up rather than down
once numpy's per-call dispatch overhead disappeared.

## Whole-PIG fit in C++: the recorded 0.0147 reproduces exactly (2026-07-26)

Real PIG data, slice38's protocol verbatim (6557 nodes, white-500 archive,
k=100 fit probes, held-out pool 250..500, tau_window=5, mu pinned, n_rungs=4,
window-shape rungs off), scored with slice38's metric. Detail in the
maintainer-local `dev/pig-cxx-2026-07-26.md`.

**Match the deployment variant before comparing anything.** slice38 reports
three numbers and its recorded table is the third: `assemble_sparse` at
`TAU_SPARSE = 6`, clipped to the fit window, then averaged with its transpose.
An un-symmetrized parametric matvec at `tau = infinity` is a different number
and was never going to line up.

| levels | window | parametric (tau=inf) | clipped (tau=6) | +sym |
|---|---|---|---|---|
| [0,1,2] | ball | 0.0521 | 0.0523 | 0.0442 |
| [0,1,2] | ellipsoid | 0.0522 | 0.0524 | 0.0443 |
| [0..6] | ball | 0.0189 | 0.0189 | **0.0147** |
| [0..6] | ellipsoid | 0.0191 | 0.0191 | 0.0150 |

**The shells-L6 ball number is 0.0147, the recorded prototype figure to all
four digits** -- and 0.0189 clipped, also matching, once the saved prototype
fit is re-scored (see the probe-budget section below; the k=100 npz stores no
clipped field, so the research README's figure had to be recomputed rather than
read back). The whole C++ stack -- LM, VarPro, mode ladder, baseline guard,
windows, assembly -- lands on the reference answer at field scale. The tau=6
clip costs essentially nothing (the window is the binding truncation);
symmetrization is worth ~15%, matching the recorded "~20%".

**The window question, now with two configurations.** At levels [0,1,2] the two
are indistinguishable (0.0442 / 0.0443) and the ellipsoid is ~20% faster on 18%
fewer window points; at levels [0..6] the ball is 2% better (0.0147 / 0.0150)
and the time advantage is gone (786 s / 777 s), because mode count rather than
window size then dominates the cost. So neither shape dominates, and the
cheap-window argument for the ellipsoid weakens exactly where the modes get
expensive. Still one basal-friction state, and intermediate caps remain untried.

**Faithfulness.** The prototype on the same bytes and configuration agrees per
row to a median 3.1% relative difference in CV score (worst 11.6%). Not
bit-identical, and should not be -- the C++ uses deterministic round-robin folds
where the prototype permutes, so the folds themselves differ, and the counting
rule was corrected. The global metric agreeing to four digits despite that is
the stronger statement of the two.

## The probe-budget sweep: where the counting-rule fix is worth 16% (2026-07-26)

Same protocol at k = 20, 40, 100, ball window, shells to level 6, ungated,
scored +sym. The prototype column is not the research README: it is the SAVED
prototype fits re-scored by `dev/pig_rescore_prototype.py`, which reproduces
the recorded `test_err_clip`/`test_err_clip_sym` for k=20 and k=40 to four
digits and therefore also supplies k=100, whose npz predates the
window-clipping code and stores no such field.

| k | prototype +sym | C++ +sym | prototype fit time | C++ fit time |
|---|---|---|---|---|
| 20 | 0.0608 | 0.0603 | 471.7 s (6 proc) | 42.5 s (4 thr) |
| 40 | 0.0320 | **0.0269** | 1315.9 s | 185.3 s |
| 100 | 0.0147 | 0.0147 | 3478.7 s | 785.9 s |

**The C++ is 16% better at k=40, and identical at k=100.** That pattern is not
noise -- it is the counting-rule fix, and it is fully explained by which mode
sets the BASELINE GUARD is allowed to use.

The baseline is a linear fit at a PINNED theta, so its admissibility should
count the pinned parameter size. The prototype counted the free-mu size
(`P_fix = N + N(N-1)/2 + N`, i.e. 5 in 2D, the bug this port corrected); the
C++ counts `theta_hat_size(dim, Pinned)`, i.e. 3. Under `k >= 2(m + extra + P)`
that moves the cap on the baseline's mode count:

| k | prototype baseline cap | C++ baseline cap |
|---|---|---|
| 20 | 3 modes | 6 modes |
| 40 | 10 modes | 15 modes |
| 100 | 28 modes (ladder top) | 28 modes (ladder top) |

- **k=100: the caps coincide**, both saturate the level-6 ladder, and the two
  implementations agree to four digits on both deployment variants (0.0189
  clipped, 0.0147 +sym) with 4168 vs 4187 searched fits shipping. The exact
  match is not luck; it is the budget at which the deliberate divergences have
  nothing left to bite on.
- **k=40: the largest cap gap**, and the largest win. A 15-mode baseline beats
  the searched fit on 617 live rows where the 10-mode one managed 192 -- so the
  guard ships the a-priori sigma three times as often, and the result improves
  by 16%. Worth sitting with: the searched fit "winning" less often is what
  made the operator better, which is the windowed-CV optimism already on record
  from the released-mu finding, caught here by a stronger status quo.
- **k=20: the caps differ but the baseline barely ships** (51 rows vs 11), so
  the effect is 0.8%.

So the fix is not cosmetic, and its value is concentrated at intermediate
probe budgets -- exactly where a too-strict baseline cap leaves accuracy on the
table without the ladder's top rung being reachable to compensate.

(Timing is 6 worker processes against 4 threads, so the per-worker ratio is
~11x / ~7x / ~4.4x rather than the raw column ratio, and the prototype runs
may not have been on this machine. The direction is safe, the constant is not.)

**A trap in reading the saved fits:** slice38's `modes_tab` is a
`(num_sets, max_modes, 3)` array PADDED WITH -1, so `len(tab[i])` is the
padding width for every set, not that set's mode count -- which reads as "every
row used the same number of modes" and hides exactly the effect above. Count
`(tab[i][:, 0] >= 0).sum()`.

## The fit is SVD-bound, not basis-bound, at production mode counts (2026-07-26)

Profiled on smooth-beta PIG with the ellipsoid window and `WedgeLadder(10,2)`,
k = 20/40/100, by callgrind. Detail in `dev/perf-2026-07-26.md`.

| share of all instructions | k=20 | k=40 | k=100 |
|---|---|---|---|
| **all SVD** | **23.9%** | **49.6%** | **60.9%** |
| basis vjp + values + harmonics | 40.2% | 23.1% | 11.6% |

**This overturns the earlier timing note's "the LM is essentially all basis
evaluation".** That was measured at 6 modes, where it holds. The inner solve is
O(k m^2) with a large constant and the basis is O(K m), so the crossover
arrives early: by k=40 the SVD is already half the run, and at k=100 it is
three fifths of everything.

**The SVD is replaceable exactly.** `inner_solve`'s filter
`sigma/(sigma^2 + ridge)` IS Tikhonov -- it computes
`(A^T A + ridge I)^-1 A^T y` -- so a thin QR gives both an orthonormal basis of
range(A~), which is all the default Kaufman Jacobian needs, and an R whose
`m x m` Cholesky yields the same `c`. Eigen's SVD is 6-9x slower than a QR at
these sizes regardless of conditioning.

**LANDED**, as `detail::InnerFactors` on `inner_solve`:

- `RangeOnly` -- what the reduced residual and the Kaufman Jacobian need -- runs
  a `ColPivHouseholderQR` whose threshold mirrors the SVD branch's rank cutoff.
  `U` is `Q`'s leading columns; the projector built from it is basis-independent,
  so the Jacobian is unchanged.
- The ridge is applied by a small stacked least-squares solve,
  `[R; sqrt(ridge) I] d = [Q^T y; 0]`, rather than by forming `R^T R + ridge I`.
  Same Tikhonov answer, but the condition number is the square root of the
  normal-equation one, so the speed does not come out of the conditioning.
- **The SVD still runs in the two cases that need it**: when `Full` is asked for
  (Golub-Pereyra reads `sigma` and `V`) and when the QR reports RANK DEFICIENCY.
  There the two genuinely differ -- a pivoted QR returns a basic solution where
  the SVD returns the minimum-norm one -- and the minimum-norm one is the point
  of having a rank cutoff.
- `ReducedProblem` records what its cache holds, so a Golub-Pereyra request at a
  point cached range-only re-solves rather than dividing by an empty `sigma`.
- `linear_cv_score`'s folds take the same QR-with-SVD-fallback route.

Measured, whole field, 4 threads:

| | SVD | landed | speedup | clipped | +sym |
|---|---|---|---|---|---|
| shells-L6 k=100, ball | 785.9 s | **376.8 s** | **2.09x** | 0.0189 both | **0.0147 both** |
| shells-L6 k=100, ellipsoid | 777.0 s | 401.4 s | 1.94x | 0.0191 both | 0.0150 both |
| shells-L6 k=40, ball | 185.3 s | 123.2 s | 1.50x | 0.0354 both | 0.0269 both |
| shells-L6 k=40, ellipsoid | 140.8 s | 103.0 s | 1.37x | 0.0358 both | 0.0275 both |

**Every headline field number is unchanged to four digits**, including the
0.0147 that matches the prototype. One or two rows in 6557 change their
searched/baseline decision. Per row single-threaded the gain is 1.13x / 1.37x /
1.54x at k = 20/40/100; the whole-field figure is better because the SVD's
allocator traffic was also costing memory bandwidth across threads.

Suite: 145 cases / 105,699 assertions, with three new tests pinning that
`RangeOnly` and `Full` agree on coefficients, residual and PROJECTOR (not `U`
itself, which is legitimately a different basis of the same subspace); that the
fallback fires on rank deficiency and recovers the minimum-norm solution; and
that a Golub-Pereyra request re-solves a range-only cache.

Also established, so it need not be re-checked: **there is no redundant
computation.** `solve_at`'s one-entry cache hits ~48% of calls, which is the
residual/Jacobian pairing working as designed, and basis evaluations equal
inner solves plus the CV and startability checks.

## Rough beta: the baseline-cap lever cuts the other way (2026-07-26)

Slice 39 replayed in C++ -- the alpha=0.01 MAP state, where the pointwise
a-priori sigma is 3-5x too wide at the channel. slice39's configuration
verbatim (`WedgeLadder(10,2)`, mu pinned, tau_window 5, ungated, ball window),
scored on its 30 held-out pairs. Detail in `dev/rb-cxx-2026-07-26.md`.

| cell | prototype +sym | C++ +sym | delta |
|---|---|---|---|
| smoothed k=20 | 0.1864 | 0.1998 | +7.2% |
| smoothed k=50 | 0.0899 | 0.0920 | +2.3% |
| smoothed k=100 | 0.0567 | **0.0561** | -1.1% |
| pointwise k=20 | 0.1917 | 0.2041 | +6.5% |
| pointwise k=100 | 0.0549 | **0.0545** | -0.7% |

**The C++ is worse at small k here, where at smooth beta it was 16% better.**
Same knob, opposite sign, and the mechanism is confirmed off the shipped
models: the baseline guard's counting-rule fix (pinned parameter size 3, not
the free-mu 5) lets the BASELINE use 6 wedge modes instead of 3 at k=20 and 21
instead of 18 at k=50, and nothing extra at k=100 where both reach the ladder's
top. The cap gap tracks the result gap monotonically and vanishes exactly where
it should.

All the fix does is let the a-priori-sigma model win the CV comparison more
often. At smooth beta the prior is nearly right and that was worth +16% at
k=40; at rough beta the prior is wrong precisely at the channel and it costs
~7% at k=20. **The counting rule is correct -- it counts the parameters
actually being fit -- but it is a lever on how often the PRIOR ships, and the
prior's quality is state-dependent.** It does not break the guard's guarantee
(every shipped row still beats the status quo on its own CV); the loss is
windowed CV being optimistic about a wrong prior, the same optimism on record
from the released-mu finding. The effect is bounded and shrinking with k, and
lgpsf still beats psfladder by >= 1.5x in every cell.

**The column forensics reproduce nearly exactly**, which locates the gap. Ten
impulse columns, 60-km off-diagonal sqrt(mL)-weighted relative L2 from the
deployed operator: at k=100, EIGHT OF TEN match the prototype to three
decimals, all 8 channel nodes land in 0.081-0.136 against 0.079-0.136, and node
1757 -- advected 9.6 km, fitted with mu PINNED -- matches at 0.117 to the
digit. At k=20 it is again 8 of 10 identical. So the field is not drifting; the
gap lives entirely in the rows where the two make a different ship decision.

**Sigma-insensitivity survives the port.** At k=100 the raw pointwise prior is
marginally BETTER than the hand-smoothed one in both implementations (0.0545 vs
0.0561 in C++, 0.0549 vs 0.0567 in the prototype), and at k=20 both show the
same 2-3% penalty. Fitted theta absorbing ordinary prior error is a property of
the method, not of the implementation.

## Dead rows need no gate in C++ either (2026-07-26)

slice38 ran ungated on the argument that a dead row (response identically zero)
costs one candidate and ships zeros. Measured on the C++ port, all 1481 dead
PIG rows: **CV score exactly 0.0, baseline exactly 0.0, all finite, status
`fallback_baseline`, one mode, zero coefficients, predicted response exactly
0.0** -- 0 failures. Ungated fitting costs the same as gated (159.9 s vs 162.4 s
for the whole field), so the gate buys nothing at this scale.

Two things this pins that were previously only argued:

- **Live-row predictions are bit-identical with and without the gate** (max
  difference exactly 0). Gating is row selection and nothing else: windows come
  from each row's own ellipsoid, and the CV folds and jitter table are global,
  so no live row can see which other rows were attempted.
- Under `Symmetrize::Average` the dead rows do carry values -- the transpose of
  live rows' columns, 0.13-0.31% of the signal energy -- with or without the
  gate. That is a symmetrization artifact, not a gating one.

The `fallback_baseline` share is configuration-dependent in a way worth noting:
61 live rows at levels [0,1,2], but **908 at levels [0..6]**. With 28 modes
available the searched theta fails to beat the a-priori sigma on ~18% of live
rows, and the always-on baseline guard is what keeps those rows at the status
quo instead of worse.

**A bridge bug worth recording**, since anyone moving arrays between numpy and
Eigen will meet it: `numpy.ndarray.tofile()` always writes C-order and silently
IGNORES `asfortranarray`, so a Fortran-ordered array does not land as
column-major. The first run gave CV ~0.97 and infinite errors; running the
PROTOTYPE on the same dumped bytes reproduced the ~0.98, which located the fault
in the bridge rather than in either implementation.

## `-march=native` changes the answer on a few rows, and that is the problem's fault (2026-07-27)

> The public write-up of this entry and the one after it is
> [`docs/reproducibility.md`](../docs/reproducibility.md), which is the
> version to cite and to keep current. What follows is the working record.

Replaying the whole PIG fit through the Python bindings and comparing against
the C++ raw-binary bridge was expected to be bit-identical -- same core, two
callers. It was not, until the two were compiled with the SAME FLAGS. The
bridge had `-march=native`; the bindings module has plain `-O3`.

The disagreement is almost entirely last-bit:

| |Δscore| over the 5076 live rows | rows |
|---|---|
| > 0 | 4634 (91%) |
| > 1e-15 | 1380 (27%) |
| > 1e-9 | 279 (5.5%) |
| > 1e-3 | **4 (0.08%)** |

Median `|Δscore|` is **1.1e-16**, one ULP. But the maximum is 5.1e-2, a
relative 0.62 on one row.

**That tail is not a bug and not a marshalling error.** `-march=native` enables
FMA and wider vectors, which reorders floating-point reductions; a 1-ULP
difference early in a row's fit is occasionally enough to send the
Levenberg-Marquardt loop into a DIFFERENT LOCAL MINIMUM. The VarPro landscape
is multimodal, so two nearby basins can both be admissible and the row lands in
whichever the trajectory reaches. Every row still made the same ship decision --
`status` was identical even before the flags matched -- so nothing downstream
noticed.

With matched flags, everything is bit-identical: score, baseline_score, status
and all three deployment variants, for both window shapes.

**Consequences worth remembering.** Bit-identity is a within-build property
here, not a cross-build one, so any acceptance criterion phrased as
"bit-for-bit" has to say against which build. Comparing results across
machines, compilers or `-march` settings should expect a handful of marginal
rows to differ materially while the field statistics do not move. And the
reverse inference is available too: if a comparison shows MANY rows differing,
that is not floating point -- 91% differing by one ULP is, but 5% differing by
1e-3 would not be.

### Which rows, and does it matter (investigated 2026-07-27)

A 2x2 -- {QR, SVD} inner solve x {plain -O3, -march=native} -- built as four
binding modules and compared row by row, scored against the HELD-OUT probe
pool that no fit saw.

**It is not the QR's doing.** At matched flags, QR and SVD agree to a median
|dscore| of 5.6e-17 with ZERO rows differing by more than 1e-3, and the global
metric matches to eight digits (0.05209035 both). Under `-march=native` the SVD
flips **exactly the same four rows** as the QR: 1326, 1627, 5083, 5819. The
inner solve is irrelevant to this; the reduction order is what matters.

**The four rows are knife-edge rows at the ice margin.**

| row | \|Yt\| / median | dist. to nearest dead node | field median dist. | modes, plain -> native | score percentile |
|---|---|---|---|---|---|
| 1326 | 2.35 | **3.0 km** | 84.7 km | 6 -> 6 | 32 |
| 1627 | 1.01 | 44.0 km | 84.7 km | 3 -> 6 | 90 |
| 5083 | 6.17 | **6.6 km** | 84.7 km | 3 -> 6 | 97 |
| 5819 | 0.19 | **4.8 km** | 84.7 km | 6 -> 6 | 55 |

Three of the four sit 3-7 km from a dead node against a field median of 85 km,
so they are at the edge of the data. They are **not small** -- one carries 6x
the median row energy. Two of the four are in the worst 10% and worst 3% of the
field by fit quality, which is the point: a row the method already struggles
with is a row whose objective is flat or multimodal, and that is exactly where
one ULP decides the basin. In two of the four the MODE LADDER stopped at a
different rung (3 modes versus 6), so the difference is discrete rather than a
small perturbation of one answer.

**It does not matter globally.** Mean held-out qc over all 5076 live rows is
0.1702 either way, with `-march=native` better on 2300 of them -- 45%, a coin
flip. The global parametric error moves in the fifth decimal:

    plain 0.05209035    native 0.05208739

On the four rows themselves native happened to win 3-1 (mean qc 0.2188 vs
0.2375), which at n = 4 is not evidence of anything.

So: a real effect, confined to rows that are hard for independent reasons, with
no systematic direction and no measurable consequence for the operator.

## Initial guesses are data, not flags (2026-07-29)

**Decision:** `fit_from_probes` takes a collection of `InitialGuess{sigma, mu?,
label}` instead of choosing among dictionaries selected by config flags.
`sigma0`, `window_shape_rungs` and `circle_rungs_above_aspect` are gone;
`num_rungs` gains `0` = "only my guesses"; `mu0` is `default_mu`. Plan and
rationale: [`archive/initial-guess-api-plan.md`](archive/initial-guess-api-plan.md).

**Why:** where to seed a nonlinear search is problem-specific, and the library
had already taken this position once -- `mode_policy` is required rather than
defaulted because "no single choice is defensible as a silent default".
Initialization has exactly that character.

It is also where a flag had already gone wrong. `circle_rungs_above_aspect = 3`
read as an economy and was in fact an off switch removing the only safety net,
on a knob whose failure the cross-validation score does not report -- median CV
moved 1.0% while held-out error moved 106%. `guesses = []` cannot hide that way.

**Three things worth keeping:**

- **`MuPolicy::Pinned` means "the optimizer does not move mu from its initial
  guess", not "mu is `default_mu`".** Each candidate is now fitted about its own
  center. This needed no change to `ellipsoid_transform.hpp`: the reference mu
  was already a parameter of `to_theta` and friends.
- **The release path's zero-displacement invariant broke.** It assumed a pinned
  candidate's center IS `mu0`; with per-candidate centers it must re-encode
  against the candidate's own, or a released fit silently starts displaced and
  still converges to something.
- **`CandidateFit::theta_hat_init` became `theta_init`, in the public absolute
  encoding.** Candidates no longer share an origin, so a displacement against an
  unstated reference cannot be decoded -- and provenance in the public encoding
  is more useful anyway.

**What it cost:** nothing. `fit_operator` is bit-identical across the change,
because it passes the caller's `sigma` as the first guess.

## The frog understates what it cannot construct (2026-07-29)

**Observation, recorded because it changed a shipped default.** The synthetic
sweep concluded the circle rungs were worthless -- 1.00x error under every
damaged prior it tested. A real Hessian said removing them doubles the held-out
error. Two separate reasons, and both generalize:

1. **It never removed the backup.** Its "circles never" variant left the
   window-shape rungs in place, and those are multi-scale starts too. Remove
   both and the frog does show a loss.
2. **It cannot build the failure.** `damaged_priors` rotates the prior, or
   scales it *and the window with it*, so it never produces a prior wrong in
   SCALE while the window stays usable -- which is exactly what the circles
   rescue, and exactly what the glaciology pointwise prior is.

Even after fixing (1) the frog reports 1.14x where the real operator reports
2x. **A synthetic problem is a good check on whether something is broken and a
bad one on whether something is unnecessary.**

## Weighted symmetrization: the weak row owns the disagreement (2026-07-31)

First slice of [`pencil-corrections-plan.md`](pencil-corrections-plan.md).
`Symmetrize::Weighted` is the per-entry convex average with inverse-row-energy
weights, `w_i^2 = 1 / (||A_i,:||^2 + (0.01 * median ||A_r,:||)^2)`.

**Why not plain averaging.** Each row is fitted independently, so where two
rows' supports overlap they hold two independent opinions of one entry.
Averaging weights those opinions equally, which is right only when the rows
have comparable scale: on the Pine Island Glacier Hessian, strong rows'
truncation tails overwrote weak rows' entire signal, and the weighted form
fixed it — it dominated both plain averaging and rows-as-is on every
operator metric measured there. On the frog
kernel, whose row scales are comparable, `Weighted` and `Average` differ by ~2%
— consistent with the claim that the weights only matter when scales differ.

**Decisions.**
- The `0.01` relative floor is hardcoded, not a parameter: the result was flat
  over two decades of the floor in the PIG study, so a knob would be noise.
- The median is over rows with any stored entries (unmodeled rows are skipped),
  midpoint-of-two convention, matching the `np.median` reference the method was
  validated with. The binding test pins C++-vs-scipy agreement to 1e-14.
- Exact symmetry is BY CONSTRUCTION, not by tolerance: both orientations of an
  entry are the same two products summed, so they round identically, and the
  C++ test checks `B == B^T` bitwise.
- Implemented as a post-pass on the assembled matrix (`detail::
  weighted_symmetrize`), not inside the row loop: the weights need every row's
  norm, so it is inherently a whole-matrix operation.

## The corrections layer is operator-blind, and a sibling (2026-08-01)

The pencil corrections layer (`include/lgpsf/corrections/`) consumes the
operator it corrects through a type-erased boundary — `SymmetricOp`, a
dimension plus a block matvec — and never reads a matrix entry. Decided after
auditing every planned operation: nothing in the required path needs entries
(the one entry-level algorithm, weighted symmetrization, is an ASSEMBLY
operation and stays format-side). Consequences worth remembering:

- **No transpose matvec exists at the boundary, on purpose.** Everything
  crossing it is symmetric; the layer measures compliance with
  `symmetry_defect` (matvecs only) instead of trusting the producer — which
  catches the one misuse assembly cannot warn about, an as-fitted operator
  handed across by mistake.
- **Sibling, not downstream:** `corrections/*` includes no fit header and no
  fit header includes `corrections/*`; `check_dependencies.py` enforces both
  directions (and a planted violation was verified to be caught). The
  corrections umbrella position also means it is NOT in `lgpsf.hpp`.
- **Type erasure over templates** for `SymmetricOp`/`HrOracle`: dispatch is
  amortized over (N x m) blocks, the durable structs need owned handles that
  cannot dangle (adapters take their matrix BY VALUE), and the layer compiles
  once — relevant in a repo whose heaviest TU peaks at 2.8 GB.
- **Blocks are columns** `(N, m)` in this layer, the linear-algebra
  convention; the fit layers keep probes-as-rows. Adapters are where the
  transposition belongs, and both boundary classes validate shapes on BOTH
  sides (caller blocks and callable results), so a rows-convention block
  fails loudly instead of transposing silently.

Full reasoning: [`pencil-corrections-plan.md`](pencil-corrections-plan.md),
P4 and §7.
