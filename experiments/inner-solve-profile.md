# Where the fit spends its time, 2026-07-26

Smooth basal friction, ELLIPSOID window (`window_aspect_cap = infinity`),
`WedgeLadder(10, 2)`, mu pinned, `tau_window = 5`, `n_rungs = 4`, single
threaded. k = 20, 40, 100 weighted equally. Measured on the glaciology
validation data (see `docs/validation.md`) with a callgrind driver kept in
that repo; attribution by instruction count.

Baseline, 200 rows each:

| k | modes/row | ms/row |
|---|---|---|
| 20 | 4.0 | 18.7 |
| 40 | 12.1 | 84.6 |
| 100 | 21.2 | 301.2 |

## Where the time goes (share of total instructions)

| | k=20 | k=40 | k=100 |
|---|---|---|---|
| **all SVD** | **23.9%** | **49.6%** | **60.9%** |
| -- `inner_solve`'s SVD | 14.3% | 34.9% | 46.6% |
| -- `linear_cv_score`'s 5 folds | 5.3% | 11.3% | 12.8% |
| `LGBasisAt::vjp` | 17.4% | 10.4% | 5.5% |
| `harmonic_basis` | 13.2% | 6.6% | 2.8% |
| `WhitenedBasis::operator()` (build the evaluation) | 10.9% | 4.7% | 1.9% |
| `LGBasisAt::values` | 9.6% | 6.1% | 3.3% |
| `pullback_vjp` | 3.9% | 1.6% | 0.6% |
| `scale_rows` | 2.5% | 1.9% | 1.2% |

**At k=100, 61% of all work is the SVD.** This overturns the earlier timing
note, which concluded "the LM is essentially all basis evaluation" -- that was
measured at 6 modes, where it is true. The SVD is O(k m^2) with a large
constant while the basis is O(K m), so the crossover arrives early: by k=40 the
SVD is already half the run.

Allocator churn is the other visible overhead: malloc/free/memcpy/memset is
~21% of instructions at k=20, much of it inside the SVD, and its share falls as
k grows.

**No redundant computation found.** `solve_at`'s one-entry cache hits ~48% of
calls (17,208 calls / 9,001 solves at k=20), which is the residual/Jacobian
pairing working as designed. Basis evaluations equal inner solves plus the CV
and startability checks. Nothing is computed twice.

## The SVD is replaceable by a QR, exactly

`inner_solve` filters singular values by `sigma / (sigma^2 + ridge)` with
`ridge = 1e-8`. That filter IS Tikhonov: it computes
`c = (A^T A + ridge I)^-1 A^T y`. A thin QR gives an orthonormal basis of
range(A~) -- all the default Kaufman Jacobian needs -- and an R from which the
same `c` falls out of an `m x m` Cholesky. Same vector, no singular values.

Measured at the sizes the profile actually sees (`experiments/bench_solve.cpp`;
conditioning made no difference, tested at collinearity 0, 1e-2 and 1e-6):

| size | BDCSVD | JacobiSVD | ColPivQR | HouseholderQR | LLT-normal |
|---|---|---|---|---|---|
| 20 x 4 | 4.96 us | 4.42 us | 1.90 us | 2.17 us | 0.59 us |
| 40 x 12 | 58.2 us | 59.9 us | 7.42 us | 8.85 us | 2.33 us |
| 100 x 21 | 219 us | 271 us | 25.3 us | 26.3 us | 8.30 us |
| 100 x 28 | 372 us | 546 us | 39.4 us | 41.7 us | 13.7 us |

6-9x at the sizes that matter, and Eigen's SVD does not get better with
conditioning -- it is simply slow on small matrices.

## Prototype: measured, not estimated

Patched copy of the headers (`LGPSF_INNER_SOLVE_QR`), both SVD sites replaced.

| k | SVD ms/row | QR ms/row | speedup |
|---|---|---|---|
| 20 | 19.7 | 17.2 | **1.15x** |
| 40 | 84.1 | 59.7 | **1.41x** |
| 100 | 301.9 | 192.5 | **1.57x** |

Whole-field, shells-L6 at k=40, both windows:

| | SVD | QR | speedup | clipped | +sym |
|---|---|---|---|---|---|
| ball | 185.3 s | 117.6 s | 1.58x | 0.0354 both | 0.0269 both |
| ellipsoid | 140.8 s | 95.0 s | 1.48x | 0.0358 both | 0.0275 both |

**The field error is identical to four digits**, CV median 0.0872 vs 0.0871,
and the searched/baseline split moves on 3 rows out of 6557. So on the real
problem the shortcut costs nothing measurable.

Test suite against the patched headers: **124 of 126 cases pass, 105,271 of
105,273 assertions.** The two failures are exactly the two the shortcut
predicts, and no others:

- `the inner solve detects rank deficiency` -- the QR path returns no `sigma`
  and does not truncate.
- `the Golub-Pereyra Jacobian matches finite differences` -- GP needs `sigma`
  and `V`, which a QR does not produce.

## What a landable version would need

Keep the SVD, use it less:

- `ColPivHouseholderQR` (rank-revealing, and measured no slower than plain
  Householder here) for the common full-rank case; take `rank()` from it, `U`
  from `Q`'s leading columns, and `c` from the `m x m` Cholesky of
  `R^T R + ridge I`.
- Fall back to the existing SVD when the QR reports rank deficiency (rare) or
  when the Golub-Pereyra Jacobian is requested (a testing/fallback path).
- `linear_cv_score`'s five folds are a plain least-squares solve with no
  singular values needed downstream; `colPivHouseholderQr().solve()` is a
  one-line swap.

Roughly 30 lines in one function plus a flag. It does change results in the
genuinely rank-deficient case, where a basic solution replaces the minimum-norm
one -- which is why the fallback matters and why the existing rank-deficiency
test should be kept pointed at the SVD path.

## Looked at, not worth it

- **Allocator churn** (~21% of instructions at k=20). Eigen temporaries, one
  set per basis evaluation. Removing it means the evaluation objects carry
  reusable buffers instead of being constructed per trial theta -- a
  restructuring of the layer whose confinement is a design property. Share
  shrinks as k grows, so it is worst exactly where absolute cost is least.
- **gemm packing.** After the SVD fix the profile is gemm-dominated (39% of
  instructions), and `gemm_pack_lhs` is 17% of that -- Eigen re-packs the SAME
  fixed `z_hat` on every evaluation because only the skinny rhs changes. There
  is no portable way to hand Eigen a pre-packed operand.
- **`linear_cv_score` re-evaluates the basis** at the winner's theta, which
  `fit_varpro` just evaluated (5.6% at k=100). Avoiding it means threading a
  design matrix through the candidate stream, and the win is small.
- **`scale_rows`** allocates and copies a (K, m) matrix per evaluation to apply
  a fixed diagonal that could be folded into `z_hat` once per row. 1.2-2.5%,
  and it would move mass-matrix knowledge out of `whitening.hpp`.

## LANDED (same day)

`detail::InnerFactors` on `inner_solve`, plus the same treatment in
`linear_cv_score`. Shape of the change:

- `RangeOnly` runs a `ColPivHouseholderQR` with `setThreshold(max(rows,cols) *
  eps)`, chosen so the cutoff `|R(i,i)| > maxpivot * threshold` mirrors the SVD
  branch's `sigma > max(rows,cols) * eps * sigma_max`.
- `U` is `Q`'s leading columns. Legitimately a different basis from the SVD's,
  but the only use is `project_out(U, .)`, which is basis-independent.
- The ridge goes through a small stacked least-squares solve,
  `[R; sqrt(ridge) I] d = [Q^T y; 0]`, NOT `R^T R + ridge I`. Same Tikhonov
  answer with half the condition number, so the speedup is not paid for out of
  the conditioning. Costs ~2% against the normal-equation prototype.
- SVD retained for `Full` (Golub-Pereyra) and for QR-reported rank deficiency.
- `ReducedProblem` records which factors its cache holds and re-solves when a
  Golub-Pereyra request finds a range-only entry.

### Measured

Per row, single threaded:

| k | SVD | landed | speedup |
|---|---|---|---|
| 20 | 19.7 ms | 17.4 ms | 1.13x |
| 40 | 84.1 ms | 61.2 ms | 1.37x |
| 100 | 301.9 ms | 196.5 ms | 1.54x |

Whole field, 4 threads -- better than per-row, because the SVD's allocator
traffic was also costing memory bandwidth across threads:

| | SVD | landed | speedup | clipped | +sym |
|---|---|---|---|---|---|
| smooth L6 k=100 ball | 785.9 s | **376.8 s** | **2.09x** | 0.0189 both | **0.0147 both** |
| smooth L6 k=100 ellipsoid | 777.0 s | 401.4 s | 1.94x | 0.0191 both | 0.0150 both |
| smooth L6 k=40 ball | 185.3 s | 123.2 s | 1.50x | 0.0354 both | 0.0269 both |
| smooth L6 k=40 ellipsoid | 140.8 s | 103.0 s | 1.37x | 0.0358 both | 0.0275 both |
| rough beta k=100 smoothed | 532 s | 373 s | 1.43x | 0.0760 both | 0.0561 both |

**No field number moved at four digits, at either basal-friction state**, the
0.0147 prototype match included. The rough-beta run's ten impulse-column
forensics are identical to three decimals (0.114, 0.136, 0.081, 0.118, 0.117,
0.120, 0.096, 0.118, 0.064, 0.065). One or two rows in 6557 change their
searched/baseline decision per cell.

Suite: 145 cases / 105,699 assertions. Three new tests: `RangeOnly` vs `Full`
agreement on `c`, residual and the PROJECTOR across three ridges and three mode
counts; the rank-deficiency fallback firing and recovering the minimum-norm
solution (`c(1) == c(3)` on duplicated columns, which a basic solution loses);
and a Golub-Pereyra request re-solving a range-only cache.
