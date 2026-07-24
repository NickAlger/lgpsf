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
