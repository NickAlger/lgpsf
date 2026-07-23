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
  (fixed dimension, all points) contiguous. This is already how
  `eval_lg_nd(p, ell, m, *u)` works -- `u` is `N` separate arrays, i.e. an
  `(N, K)` SoA layout pre-split into `N` objects. No change needed on the
  Python side.
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

**Consequence for the core eval/gradient functions:** keep them
layout-agnostic. They should take `N` separate contiguous vectors (as
`eval_lg_nd`/`grad_eval_lg_nd` already do), not a single 2D matrix with a
baked-in row/column convention. That pushes the "how are points actually
stored" decision to whatever assembles those `N` vectors (mesh point
gathers, `ellipsoid_tree` neighbor lists, a pybind11 binding, ...), keeping
the numerical kernel decoupled from it.

**Forward-looking flag, not yet a problem:** if points ever cross the
Python/C++ boundary via pybind11 (a numpy `(N, K)` array in, an Eigen
`(K, N)` matrix expected), that boundary is a transpose. pybind11's
automatic Eigen &lt;-&gt; numpy conversion needs to be told about this
explicitly, or it will silently insert a copy. Nothing to do about this yet
-- there are no bindings -- but worth remembering when that work starts.
