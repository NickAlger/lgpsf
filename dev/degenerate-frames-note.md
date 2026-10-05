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
   mesh the mode is then a delta: its column of the design matrix is the spike's column. Nothing
   bounds the frame, so the other axis is free to run to 1e306.
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

## For readers of fitted-ellipsoid statistics

Until this is fixed, leave rows with a non-finite or extreme frame out of any statistic or picture
of fitted ellipsoids (the dump's `L` is in the record; status alone does not mark them).
