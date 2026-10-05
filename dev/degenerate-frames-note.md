# Degenerate fitted frames can ship (known problem, 2026-10-04; not fixed)

A searched row fit can pass the baseline guard with an ellipsoid frame that has collapsed, exploded,
or overflowed. Usually that is harmless. Rarely it puts an entry of order 1e11 on the operator's
diagonal. Found in a production operator fit (409,545 rows, an ice-sheet Gauss-Newton Hessian, 10
fit probes + 5 held out); the per-row dumps of older fits show the same rows, so the problem is old.

**Status: recorded, deliberately not fixed.** Any change here changes which rows ship the searched
fit, and with it the probe counts a quality-controlled ladder chooses and everything downstream; a
campaign of runs was in flight on the present behaviour. Fix after it, and re-baseline then.

## What was seen

One row carried the whole held-out error of a fit: energy-ratio QC 3.9e8 where 0.39 was the rest of
the operator. The same row, with the same probes, at two different operators (the probe set decides
it, the operator hardly). Its record:

| | |
|---|---|
| a-priori frame | isotropic, sigma 10.4 km; the node sits in a coarse patch of the mesh, nearest neighbours 24 to 39 km away, so its window (3.5 sigma) holds few nodes |
| status | `Fit` (the searched fit beat the baseline), one mode, centre at the node |
| fitted `L` | `[[6.1e306, 0], [1.1e3, 9.7e-16]]`: one axis at the edge of overflow, the other at round-off |
| `L L^T` | `[inf, inf, 1.1e6]` |
| coefficient `c` | -8.2e6 |
| spike `s` | 2.0e9, i.e. `m1 * s` = +8.6014e11 |
| kernel at the row's own node | `m1^2 * k0` = -8.6014e11 |

With five more probes the same row shipped the baseline and was ordinary.

## The mechanism, in three steps

1. **The search ends at a needle.** With few probes and a window of few nodes, the search ends at a
   frame whose one axis has shrunk until the Gaussian is nonzero only at the row's own node. On the
   mesh the mode is then a delta: its column of the design matrix is the spike's column. The other
   axis runs to 1e306: the search itself is unbounded, and the admissibility rule that should have
   excluded the result has an escape hatch (next section).
2. **The coefficients become a cancelling pair.** VarPro projects the extra block (the spike) out of
   the mode columns; a mode that lies in the spike's span leaves a reduced column of round-off, and
   the linear solve returns a huge `c` with the spike `s` compensating (the Frisch-Waugh-Lovell
   back-solve in `varpro.hpp`). The sum at the diagonal is right to one unit in the last place of
   8.6e11, i.e. to 1e-4. Probably the equilibration of the reduced matrix is what hides the
   collapse (a noise column scaled to unit norm looks well conditioned); this step was reasoned
   from the code's structure and the dumped coefficients, not traced line by line.
3. **The assembly breaks the pair.** `assemble_sparse` finds a row's stored columns by colliding the
   kernel ellipsoid `(mu, L L^T)` with the column points. A non-finite covariance evidently
   collides with nothing, not even the row's own node, so the smooth part of the row is dropped;
   the spike is added unconditionally. The diagonal is then `m1 * s` = 8.6e11 alone. (Inferred
   from the numbers, not stepped through the tree: at the two operators the row's held-out
   residuals stand in the ratio of the two spike entries, 1.872, to four digits, probe by probe.)

Step 3 needs `L L^T` to overflow, which is why one row in 400,000 blew up while about a thousand
shipped a degenerate frame.

## The rule that already exists, and its escape hatch (found 2026-10-04, later the same day)

`fit_from_probes` already states the principle: *"Fits whose major semi-axis exceeds the window
radius, or whose released center leaves the window, are excluded. Few-equation validation cannot
reliably reject such degenerate fits -- observed live, a 3000:1 needle three times the window won a
4-equation holdout by 0.02."* Each candidate gets `admissible = axes.maxCoeff() <= radius`, with
`radius = window_radius(x, mu0)` the distance to the farthest point of the batch, and selection is
admissibility first, then score.

The hole is in `select`: **when no candidate is admissible, every candidate is allowed**
(`if ( allowed.empty() ) { allow all }`), and the best-scoring inadmissible one goes on to the
baseline guard. Every shipped frame larger than its window came through there, the 6e306 needle
included: at least 1.4 to 2.4% of the searched rows at the final rung (the "above 1" row of the
table further down), 6.3% at 10 probes. There is no rule at all on the small side: a candidate
with a collapsed minor axis is admissible.

So the question is not whether to add a ceiling. One exists, at one window radius. The question
is what the fit does when nothing passes it.

## Why the baseline guard did not catch it

The guard (`select_row_fit`) ships the searched fit when its cross-validation score strictly beats
the baseline's. Both scores come from `linear_cv_score`: the basis at fixed parameters, the linear
coefficients REFIT on each training fold. In that arithmetic the degenerate model is simply "a
diagonal entry", and for a node whose window holds few nodes a diagonal entry predicted the held-out
folds better than the a-priori Gaussian plus spike. The guard did what it is defined to do. What it
never looks at is the object that ships:

- the shipped coefficients are the search's (`model.c`, `model.s`), the cancelling pair, not a refit;
- the shipped frame is `model.frame()`, and nothing checks that `L` and `L L^T` are finite, or that
  the frame can be resolved by the points it is evaluated on;
- the score is taken in the fit's own encoding of the frame, the assembly goes through `L L^T`.

## How common

Per-row dumps of the fit, rows with a model:

| fit | probes | exploded frame (non-finite, or an axis above 1e6 km) | collapsed (minor axis under 1 m) |
|---|---|---|---|
| this operator | 10 | 1,066 to 1,646 of 409,497 (0.3 to 0.4%) | 760 to 900 |
| this operator | 25 | 110 to 120 | 70 to 90 |
| an older fit of the same problem | 50 | 61 | 66 |
| a smaller domain, an older version | 10 | 108 of 27,660 | 130 |

All of them have status `Fit`. Coefficients reach 1e155 times the typical one. 53 to 77% of the
exploded rows start from an isotropic a-priori frame, against 15 to 21% of all rows (an isotropic
start has no preferred axis to keep). The wider family, rows whose spike cancels a large kernel
diagonal (about 7,000 at 10 probes), held 16% of the held-out error numerator of one fit at 10
probes and 0.2% at 25.

So far the consequence in a quality-controlled ladder has been one wasted rung (the QC sees the
row, the ladder draws more probes, and with more probes the row ships the baseline). A fit at a
fixed probe count has no such net.

## What a fix could be (to decide, not decided)

In rising order of how much they change the method:

1. **Refuse an unrepresentable frame at the guard.** A searched fit whose `L` or `L L^T` is not
   finite, or whose axes leave `[a fraction of the local point spacing, a multiple of the window]`,
   does not ship; the baseline does. Cheap, local to `select_row_fit`, and it states a principle:
   a frame must be resolvable by the points it is evaluated on and must live inside its window.
2. **Bound the frame in the search** (box constraints or a penalty on the log-axes in the LM
   parametrization), so the search cannot reach the needle at all.
3. **Detect the absorbed mode in VarPro**: compare each reduced column's norm with the unreduced
   one before equilibrating, and drop a mode that the extra block already spans. The row then
   ships "spike only", honestly, with `c = 0`.
4. **Make the assembly robust on its own**: treat a non-finite kernel ellipsoid as an error for
   that row (or evaluate the smooth part on the window when the collision query is empty), so a
   cancelling pair can never be broken silently. Worth doing whatever else is chosen.

Tests to add with the fix: a row whose window holds one or two points; a search started from an
isotropic frame with fewer probes than a well-posed fit needs; the assembled diagonal against the
dense evaluation for every shipped row.

## How many rows a bound on the frame would touch (measured 2026-10-04)

Final rung of two fits of the production operator (20 and 25 probes; about 380,000 searched rows
each), and the first rung (10 probes). The window is the a-priori ellipsoid inflated by 3.5; the local
point spacing is taken as the square root of the row's cell area.

| fitted frame | final rung | 10 probes |
|---|---|---|
| largest axis (1 sigma) / window's largest semi-axis: median | 0.31 to 0.34 | 0.33 |
| ... above 1 (the ellipsoid is larger than its window) | 1.4 to 2.4% (5,200 to 9,200 rows) | 6.3% |
| ... above 2 | 3,800 to 6,800 rows | 18,200 |
| ... above 10 | 2,100 to 3,900 rows | 11,100 |
| smallest axis / local spacing: median | 0.75 to 0.82 | 0.77 |
| ... below 0.5 (the a-priori floor) | 16 to 21% | 21% |
| ... below 0.1 | 1,100 to 1,400 rows | 9,100 |
| ... below 0.01 | 190 rows | 1,800 |

So a ceiling at the scale of the window is not a patch for a hundred overflowed rows: it changes one
to two percent of the fitted rows, the ones whose kernel is flat across the window and which the
unbounded fit represents by a frame far larger than the window with enormous coefficients. A floor
at half the spacing would change a fifth of the rows (the kernels of this problem are under-resolved
on its mesh over much of the domain), which makes the floor a separate modelling question.

## Mechanisms for a ceiling (discussion 2026-10-04, nothing decided)

*(Written before the escape hatch was found. What follows still describes the two mechanisms; where
they attach changes: the natural place is the empty-pool branch of `select`, and the natural
ceiling is the admissibility bound that already exists, one window radius. Options there: ship the
baseline when nothing is admissible; or clamp the best inadmissible candidate's axes into the
admissible range, re-solve its linear coefficients, re-score it, and let the baseline guard
decide. A floor would enter the same way, as a lower admissibility bound on the minor axis.)*

**No constrained optimizer is needed for a wall.** The search already has a feasibility test:
`make_frame` throws `InfeasibleParameters` when the log-Cholesky diagonal overflows, VarPro scores
such a trial point as "the smooth model contributes nothing", and the trust-region loop rejects the
step and contracts. A ceiling inside the search is that predicate made tighter (the frame's largest
axis, measured in the window's coordinates, above `c` window radii is infeasible). A few lines plus
passing the window scale to the basis. The iterate then creeps along the wall with shrinking steps
and stops on the step tolerance near a constrained stationary point: workable, and it ends runaways
early, but with a ceiling that thousands of rows reach it spends many small steps there.

**Clamp after the search, then one linear solve (the maintainer's proposal).** Take the search's
frame, clamp the eigenvalues of its covariance in the window's coordinates to `(c R)^2`, re-solve
the linear coefficients at the clamped frame (`detail::linear_fit`, what the baseline already
does), re-score it with `linear_cv_score` (what the coarsened rows already do for both finalists),
and let the guard decide against the baseline as now. Rows that are not clamped are untouched, bit
for bit. The shipped object is then the scored object, its coefficients are a least-squares
solution at a frame the points can resolve, and `L L^T` cannot overflow. What it does not do: stop
the search from running away (the wasted iterations stay), fix a collapsed minor axis (that needs
the floor, or dropping a mode the spike already spans), or make the clamped frame an optimum of
anything (its orientation and minor axis are those of a runaway search).

**Why a ceiling at two or three window radii should cost almost no accuracy.** Over its window a
Gaussian of sigma = 3 R is flat to 5% (2 R: 12%), and the higher modes carry the remaining
variation with coefficients of order one. The unbounded fit represents the same content with a
frame at 1e3 to 1e300 window radii and coefficients up to 1e155 times the typical one.

**Not recommended:** a bounded reparametrization (sigmoid of the log-axes) changes every fit, needs a
rotation parametrization in general dimension, and has flat directions at the bounds; a penalty adds
a weight and touches the Golub-Pereyra Jacobian without giving a guarantee.

Before any of it ships: an A/B at fixed probe counts (held-out QC, baseline and fallback counts,
fit time) with `c` in {1, 2, 3}.

## For readers of fitted-ellipsoid statistics

Until this is fixed, leave rows with a non-finite or extreme frame out of any statistic or picture
of fitted ellipsoids (the dump's `L` is in the record; status alone does not mark them).
