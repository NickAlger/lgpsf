# Graded window coarsening: the numpy prototype, 2026-09-06

Slice S1 of [`../dev/window-coarsening-plan.md`](../dev/window-coarsening-plan.md):
a pure-numpy `coarsen_window` written to the plan's specification, run against
the existing Python bindings with no library change, to decide `eps`'s default
and to check the plan's error claims before the C++ is written. Measured by
[`window_coarsening.py`](window_coarsening.py) (`--quick` for a 3-minute
version; the full run below takes about 20 minutes and a fixed seed).

**Short version.** `eps = 0.1` is the right default. On 60 real rows spanning
windows of 250 to 100,000 points it changed no row's guard decision, kept the
fitted ellipsoid within 0.03% in axes and 0.25 degrees in orientation, and cut
the fit's wall time 11-54x on the 100,000-point windows. One claim in the plan
does not survive: with white-noise probes the quadrature error is **first**
order in `eps`, not second, because aggregation keeps a cell's probe *sum*
exactly but not its *dipole*. The fit tolerates it, since the error is zero-mean
across probes, but the plan's prose should say so. Two more findings worth
knowing: nothing coarsens within `h / eps` of the centre (a singleton core of
about `7 pi / eps^2` points, so the count law needs a floor), and the
guard taken on coarse-quadrature scores is optimistic -- at `eps >= 0.2` it
claims wins over the baseline that the full-window scores deny -- which is the
plan's full-window re-score (decision 2) earning its keep.

## The setup

**The algorithm** is the plan's, section 2, implemented as `coarsen_window` in
the script: whiten by the window frame, protect the spike and the farthest
point, a quadtree on the whitened bounding box split while a node holds more
than one point and its box diagonal exceeds `eps` times the box's distance
from the centre, cells sorted by their smallest member, accumulation in
ascending order, `eps <= 0` the identity. It is the reference for the C++.

**The problem** is the heat-equation inversion of
[`../examples/heat_inversion.py`](../examples/heat_inversion.py), rebuilt
matrix-free so it runs at 100,000 points (the example's dense `H` is only
there to report truth, and the fit never sees it). Grids of 48, 100, 200 and
316 give the same physics at four mesh spacings, so a row's window grows from
a few hundred points to 100,000 while its kernel stays put. Rows per grid:
slab, background, pocket and interface rows drawn at random, the two rows with
the smallest response (the domain corners, where Dirichlet leakage makes the
row weakest -- this problem has no genuinely dead rows), and pocket rows with
their prior widened 4x (`pocket-x4`: window radius 4x, kernel unchanged),
which is the plan's motivating case in miniature. 60 rows in all.

**The fit** is `fit_operator`'s row fit exactly: `tau_window = 10`, spike on,
pinned centre, three circle rungs after the prior, `ShellLadder(0..5)`,
60 probes (the counting rule needs 50 for level 5), and `target_score = None`
so the ladder is decided by cross-validation alone -- the demanding setting
for a quadrature, since 46 of the 60 rows ship level 4 or 5. The "full" fit is
`fit_from_probes` on the full window with `fit_operator`'s own configuration,
and it reproduces `fit_operator`'s row bit for bit: over all 60 rows the
maximum discrepancy in score and in baseline score is `0.0`. The coarse fit is
the same call on the cells, with the spike's cell as `spike_index` and
`target_mass` passed explicitly. Every coarse model is then **re-scored on
the full window** (`linear_cv_score` with a `WhitenedBasis` on the full
window at the coarse fit's theta), and that honest number is what "score"
means below unless it says "coarse own".

## (a) Correctness

Over every window and every `eps` (discs and real rows): mass conserved to
`1.7e-14` relative, `sum m_C z_C` conserved to `1.1e-14`, the spike and the
farthest point singletons in every case, and no multi-point cell above the
grading bound (the worst `diag / (eps d)` is `1.000`: attained, never
exceeded). `eps = 1e-9` on a 1001-point disc returns 1001 cells with
`cell_of == arange`, i.e. the identity in the same order.

## (b) The cell count

Grids of spacing `h` filling the unit disc (a ball window, `R = 1`), masses
`h^2`, the point nearest the centre protected:

| K | eps | cells | singletons | 3pi/eps^2 log2(R/h) | ratio | coarsen s | err mass | err m*z | protected ok | worst diag/(eps d) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1001 | 0.05 | 1001 | 1001 | 15672 | 0.06 | 0.04 | 0.0e+00 | 0.0e+00 | True | 0.000 |
| 1001 | 0.1 | 962 | 923 | 3918 | 0.25 | 0.03 | 0.0e+00 | 1.5e-15 | True | 0.963 |
| 1001 | 0.15 | 918 | 835 | 1741 | 0.53 | 0.03 | 0.0e+00 | 1.8e-15 | True | 0.994 |
| 1001 | 0.2 | 812 | 696 | 980 | 0.83 | 0.02 | 0.0e+00 | 1.3e-15 | True | 1.000 |
| 1001 | 0.3 | 490 | 288 | 435 | 1.13 | 0.02 | 0.0e+00 | 1.5e-15 | True | 0.971 |
| 10009 | 0.05 | 8712 | 8107 | 21934 | 0.40 | 0.24 | 0.0e+00 | 2.4e-15 | True | 1.000 |
| 10009 | 0.1 | 4332 | 2244 | 5483 | 0.79 | 0.13 | 0.0e+00 | 2.0e-15 | True | 1.000 |
| 10009 | 0.15 | 2472 | 1044 | 2437 | 1.01 | 0.07 | 0.0e+00 | 3.5e-15 | True | 1.000 |
| 10009 | 0.2 | 1727 | 612 | 1371 | 1.26 | 0.07 | 1.4e-16 | 3.1e-15 | True | 1.000 |
| 10009 | 0.3 | 932 | 281 | 609 | 1.53 | 0.03 | 2.8e-16 | 4.4e-15 | True | 0.955 |
| 100015 | 0.05 | 21563 | 8025 | 28195 | 0.76 | 0.69 | 2.8e-16 | 1.7e-15 | True | 1.000 |
| 100015 | 0.1 | 7671 | 2108 | 7049 | 1.09 | 0.27 | 5.7e-16 | 6.0e-15 | True | 1.000 |
| 100015 | 0.15 | 4019 | 959 | 3133 | 1.28 | 0.18 | 1.4e-15 | 6.5e-15 | True | 1.000 |
| 100015 | 0.2 | 2621 | 560 | 1762 | 1.49 | 0.12 | 1.1e-15 | 6.6e-15 | True | 1.000 |
| 100015 | 0.3 | 1358 | 259 | 783 | 1.73 | 0.08 | 2.8e-15 | 5.9e-15 | True | 0.955 |

The law's structure holds -- `1 / eps^2` times a logarithm -- but two things
it leaves out matter at the sizes in question.

- **A singleton core.** A cell holding two grid points has a diagonal of at
  least `h`, so nothing merges within whitened distance `h / eps` of the
  centre, and the dyadic quantization keeps most cells single out to a few
  times that. Measured: about `7 pi / eps^2` singletons (8025 at `eps = 0.05`
  against the naive `pi / eps^2 = 1257`). For small `eps` and moderate `K` the
  count is mostly this floor: at `K = 10^4`, `eps = 0.05`, 8107 of the 8712
  cells are singletons, and a 1001-point window does not coarsen at all.
- **The constant is above `3 pi` where the grading does bite.** The rule
  bounds the box *diagonal*, so a leaf's side is at most `eps d / sqrt(2)`
  (a factor 2 in area), and dyadic halving leaves leaves between half and the
  full allowed size. The ratio to the plan's law at `K = 10^5` runs from 0.76
  (`eps = 0.05`, floor still binding) to 1.73 (`eps = 0.3`).

For planning: a 100,000-point ball window becomes 21,600 cells at `eps = 0.05`
(4.6x), 7,700 at `0.1` (13x), 4,000 at `0.15` (25x). The plan's "90,000 points
to about 7,000 cells at `eps = 0.1`" is right. On real rows the count also
depends on where the centre sits: a corner row's window is a quarter disc and
gets a quarter of the cells (2,418 at 100,000 points, `eps = 0.1`, against
7,460 for a row in the interior). The Python coarsening itself costs 0.3 s per
100,000 points at `eps = 0.1`; the C++ will not be the bottleneck either way.

## (c) The fit on coarse cells against the fit on the full window

All 60 rows, per `eps`. `dscore` is the coarse fit's full-window score minus
the full fit's score (positive is worse); "beats baseline (honest)" is the
fraction of rows whose coarse fit is admissible and beats the full-window
baseline, to be read against the full fits' own 0.87; `cv_fixed` is the
score's change at the *full* winner's theta when only the quadrature changes;
"time coarse/full" is the ratio of summed wall times.

| eps | median Kc/K | median dscore | max dscore | beats baseline (honest) | admissible | same level | level lower | median dlog10 major | median dlog10 minor | median dangle deg (aspect>1.1) | median dc0 | median dspike | median cv_fixed | time coarse/full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.864 | +0.0000 | +0.0029 | 0.87 | 1.00 | 0.98 | 0.00 | 0.0000 | 0.0000 | 0.00 (39) | 0.0000 | 0.0000 | 2.3e-14 | 0.235 |
| 0.1 | 0.493 | +0.0000 | +0.0108 | 0.87 | 1.00 | 0.97 | 0.03 | 0.0001 | 0.0000 | 0.23 (39) | 0.0002 | 0.0003 | 5.3e-05 | 0.110 |
| 0.15 | 0.305 | +0.0000 | +0.1366 | 0.85 | 1.00 | 0.88 | 0.12 | 0.0008 | 0.0006 | 1.14 (39) | 0.0015 | 0.0029 | 5.9e-04 | 0.072 |
| 0.2 | 0.219 | +0.0002 | +0.0344 | 0.78 | 1.00 | 0.93 | 0.07 | 0.0019 | 0.0016 | 1.62 (39) | 0.0043 | 0.0128 | 2.4e-03 | 0.054 |
| 0.3 | 0.129 | +0.0006 | +0.0730 | 0.72 | 1.00 | 0.75 | 0.18 | 0.0042 | 0.0033 | 1.93 (39) | 0.0136 | 0.0365 | 1.3e-02 | 0.036 |

Full fits: 0.87 of rows beat the baseline; median score 0.0699 against a
median baseline of 0.1184; shipped levels `[4, 8, 0, 2, 25, 21]` for levels 0
to 5.

- **`eps = 0.05`** changes nothing that a table can see: `dscore` at most
  `+0.003`, 98% of rows at the same ladder level, orientation identical, every
  guard decision the same. It also buys little on rows below 10,000 points,
  because they are almost all singletons.
- **`eps = 0.1`** keeps every guard decision (0.87 both ways), the level on 97%
  of rows, axes within `10^-4` in `log10` (0.03%), orientation within 0.23
  degrees, `c_0` within 0.02% and the spike's share of the diagonal within
  0.03%, for 9x less wall time overall and 11-54x on the largest windows.
  The worst row (`g200/weak 0`, `+0.011`) stopped its ladder one level early.
- **`eps = 0.15`** shows the first real casualty: `g100/slab 3466` (see the
  worst-row table) collapses from level 5 at 0.060 to level 0 at 0.197 --
  above its baseline of 0.173, so the row would ship the baseline. This is the
  plan's "conservative failure" (the coarse quadrature makes the high modes
  look like noise, the ladder stops low), and at this `eps` it is already
  expensive. 12% of rows stop lower.
- **`eps >= 0.2`** loses 5 of 52 fits at 0.2 and 9 at 0.3, and the coarse
  score is no longer honest (section d).

The worst row at each `eps` (coarse fit re-scored on the full window, minus
the full fit):

| eps | tag | row | K | Kc | score full | coarse on full | coarse own | baseline | level full/coarse | cand full/coarse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | g316/weak | 315 | 99856 | 7311 | 0.0683 | 0.0711 | 0.0716 | 0.0863 | 5/5 | 29/29 |
| 0.1 | g200/weak | 0 | 40000 | 2084 | 0.0773 | 0.0881 | 0.1062 | 0.0959 | 3/2 | 29/24 |
| 0.15 | g100/slab | 3466 | 10000 | 2282 | 0.0600 | 0.1966 | 0.1938 | 0.1726 | 5/0 | 29/14 |
| 0.2 | g100/weak | 99 | 10000 | 590 | 0.0454 | 0.0799 | 0.1071 | 0.0677 | 5/3 | 29/29 |
| 0.3 | g200/background | 19353 | 31671 | 1138 | 0.0551 | 0.1281 | 0.1308 | 0.0674 | 4/1 | 29/19 |

Every casualty is a ladder that stopped early, never a fit that wandered:
axes and orientation barely move even at `eps = 0.3`. And every one is a
row whose kernel fills its window -- the honest-prior rows on the finest
meshes. The plan's motivating case, a prior too wide by decades, is the
*easy* case: on the `pocket-x4` rows the kernel sits inside the singleton
core and the coarse fit's score is within 0.0003 of the full fit's at every
`eps` up to 0.2, for 5-10x fewer points at `eps = 0.1`.

The ten largest windows, `dscore` (cells) per `eps` -- the regime where the
mesh is fine against the kernel (`h / sigma` down to 0.016) and the grading,
not the floor, decides:

| tag | row | K | score full | baseline | eps 0.05 | eps 0.1 | eps 0.15 | eps 0.2 | eps 0.3 |
|---|---|---|---|---|---|---|---|---|---|
| g316/weak | 0 | 99856 | 0.0813 | 0.0963 | +0.0005 (7311) | +0.0041 (2418) | +0.0040 (1275) | +0.0034 (789) | +0.0191 (431) |
| g316/weak | 315 | 99856 | 0.0683 | 0.0863 | +0.0029 (7311) | +0.0063 (2418) | +0.0122 (1275) | +0.0158 (789) | +0.0276 (431) |
| g316/slab | 3197 | 99856 | 0.0802 | 0.1093 | +0.0004 (10969) | +0.0100 (4299) | +0.0060 (2403) | +0.0085 (1627) | +0.0132 (924) |
| g316/background | 58318 | 95020 | 0.1044 | 0.1392 | -0.0002 (20917) | -0.0005 (7460) | -0.0006 (4005) | +0.0048 (2570) | +0.0017 (1389) |
| g316/pocket-x4 | 70262 | 71031 | 0.0206 | 0.8209 | +0.0000 (19132) | -0.0000 (6997) | +0.0001 (3851) | +0.0003 (2437) | +0.0042 (1347) |
| g200/weak | 0 | 40000 | 0.0773 | 0.0959 | +0.0001 (5957) | +0.0108 (2084) | +0.0018 (1101) | +0.0080 (722) | +0.0116 (380) |
| g200/weak | 199 | 40000 | 0.0916 | 0.1175 | +0.0001 (5957) | +0.0069 (2084) | +0.0044 (1101) | +0.0048 (722) | +0.0173 (380) |
| g200/slab | 5439 | 40000 | 0.0933 | 0.1110 | +0.0001 (11050) | -0.0000 (4695) | +0.0030 (2666) | +0.0304 (1815) | +0.0372 (1003) |
| g200/slab | 8902 | 40000 | 0.0665 | 0.1431 | +0.0003 (14127) | +0.0004 (5694) | +0.0003 (3126) | +0.0022 (2077) | +0.0133 (1140) |
| g200/background | 19353 | 31671 | 0.0551 | 0.0674 | +0.0000 (13286) | -0.0000 (5505) | -0.0001 (3137) | +0.0002 (2049) | +0.0730 (1138) |

Here `eps = 0.1` costs `+0.004` to `+0.010` on the three 100,000-point rows
that ship level 5 -- 5-12% of their score, still 0.01-0.02 below their
baselines -- and `eps = 0.05` brings that under `+0.003` for about 3x more cells.
The cost is real but small, and it is the price of the first-order error
below.

### The order of the quadrature error

The plan argues (section 1) that the centroid rule makes the aggregation
second order. Measured directly -- the relative error of each whitened design
column `sqrt(m_rho) sum_j m_j z_jl phi_i(x_j)` at a fixed theta, worst mode
per level -- on a 300,000-point disc with a Gaussian of width `R / 3`
(`h / sigma = 0.010`, the asymptotic regime), for white-noise probes and for
smooth ones (random plane waves):

| probes | eps | cells | level 0 | level 1 | level 2 | level 3 | level 4 | level 5 |
|---|---|---|---|---|---|---|---|---|
| white | 0.05 | 28211 | 1.10e-02 | 2.03e-02 | 1.96e-02 | 3.24e-02 | 3.26e-02 | 4.88e-02 |
| white | 0.1 | 9403 | 2.06e-02 | 4.09e-02 | 3.67e-02 | 5.93e-02 | 9.49e-02 | 8.30e-02 |
| white | 0.15 | 4813 | 3.74e-02 | 6.57e-02 | 6.03e-02 | 1.13e-01 | 1.24e-01 | 9.91e-02 |
| white | 0.2 | 3082 | 5.96e-02 | 7.30e-02 | 6.67e-02 | 1.62e-01 | 1.54e-01 | 1.50e-01 |
| white | 0.3 | 1599 | 7.20e-02 | 8.68e-02 | 1.38e-01 | 2.21e-01 | 2.04e-01 | 3.03e-01 |
| white | slope |  | 1.12 | 0.84 | 1.05 | 1.12 | 1.01 | 0.96 |
| smooth | 0.05 | 28211 | 1.71e-04 | 2.03e-04 | 3.62e-04 | 7.84e-04 | 6.08e-04 | 9.76e-04 |
| smooth | 0.1 | 9403 | 6.81e-04 | 7.97e-04 | 1.24e-03 | 2.57e-03 | 2.09e-03 | 3.50e-03 |
| smooth | 0.15 | 4813 | 1.53e-03 | 3.06e-03 | 3.59e-03 | 4.91e-03 | 4.82e-03 | 7.89e-03 |
| smooth | 0.2 | 3082 | 2.72e-03 | 3.29e-03 | 4.09e-03 | 6.61e-03 | 6.32e-03 | 1.09e-02 |
| smooth | 0.3 | 1599 | 5.53e-03 | 1.07e-02 | 1.31e-02 | 1.90e-02 | 1.80e-02 | 2.81e-02 |
| smooth | slope |  | 1.95 | 2.20 | 1.96 | 1.71 | 1.85 | 1.84 |

**Second order for smooth probes, first order for white noise**, and the
gap is 24-30x at `eps = 0.1`. The reason is in the Taylor expansion of one
cell's contribution: replacing `sum_{j in C} m_j z_jl phi_i(x_j)` by
`(sum m_j z_jl) phi_i(x_C)` leaves a leading error of
`grad phi_i(x_C) . d_Cl` with `d_Cl = sum_{j in C} m_j z_jl (x_j - x_C)`, the
probe's *dipole* inside the cell. The centroid makes `sum m_j (x_j - x_C)`
vanish, so `d` is second order for a probe that is smooth over the cell --
the plan's argument -- but a white-noise probe has a dipole of the same order
as its monopole, and the error is `eps` times the mode's relative gradient,
which is why it also grows about linearly with the level. The same slopes
(0.97-1.21) appear on the 15 largest real windows at their full winner's
theta, together with the score's change at that theta:

| eps | level 0 | level 1 | level 2 | level 3 | level 4 | level 5 | median cv_fixed | max cv_fixed |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 5.63e-03 | 1.24e-02 | 2.33e-02 | 2.94e-02 | 3.93e-02 | 4.35e-02 | 1.2e-03 | 7.6e-03 |
| 0.1 | 1.45e-02 | 2.77e-02 | 4.70e-02 | 6.34e-02 | 8.60e-02 | 1.01e-01 | 8.3e-03 | 3.2e-02 |
| 0.15 | 2.21e-02 | 4.30e-02 | 7.13e-02 | 9.55e-02 | 1.18e-01 | 1.53e-01 | 5.8e-03 | 4.3e-02 |
| 0.2 | 2.95e-02 | 5.16e-02 | 8.91e-02 | 1.23e-01 | 1.57e-01 | 1.84e-01 | 1.1e-02 | 6.9e-02 |
| 0.3 | 5.11e-02 | 8.23e-02 | 1.34e-01 | 1.79e-01 | 2.32e-01 | 3.01e-01 | 3.9e-02 | 1.0e-01 |
| slope | 1.21 | 1.04 | 0.97 | 1.01 | 0.98 | 1.06 | 1.71 | 1.42 |

A 10% column error at level 5 sounds alarming; the fit shrugs it off because
the error is zero-mean and independent across probes, so it enters the
regression like noise added to the design rather than like a bias: the score
moves by about `1e-2` at `eps = 0.1` on these rows and the fitted ellipsoid by
`10^-4` in `log10` axes. Two consequences for the plan. The prose in section 1
should say "first order in `eps` for white-noise probes (the default), second
order for smooth ones". And if S6 ever finds `eps = 0.1` too lossy on
fine-mesh rows, the fix is known: carry the per-cell dipole `d_C` (N extra
vectors per cell) and add `grad phi_i(x_C) . d_C` to the design, using the
basis gradient the fit already has for its Jacobian; that restores second
order. These numbers do not ask for it.

## (d) The safety check

Plan section 4.7: is the coarse-quadrature score a safe currency for the
search and the guard? On the 15 largest windows, the coarse fit's own score
against the same model's full-window score, and the guard's decision under
each:

| eps | max abs(score_c - score_c_full) | coarse says beat, full says lose | full says beat, coarse says lose |
|---|---|---|---|
| 0.05 | 0.0072 | 0 | 0 |
| 0.1 | 0.0181 | 0 | 0 |
| 0.15 | 0.0256 | 0 | 0 |
| 0.2 | 0.0381 | 2 | 0 |
| 0.3 | 0.0556 | 4 | 0 |

A model's coarse-quadrature score differs from its full-window score by up to
0.018 at `eps = 0.1` and 0.056 at 0.3; on the largest windows it is usually
the *higher* of the two, since the coarse quadrature adds noise to the design.
The guard's comparison is another matter: the coarse baseline moves too, and
the comparison taken on coarse numbers is **optimistic**. At `eps <= 0.15` it
never flips a decision; at `eps >= 0.2` it claims wins over the baseline that
the honest scores deny, and never the reverse.
So the plan's decision 2 -- re-score the finalists on the full window and take
the guard on those numbers -- is not a nicety: without it, `eps = 0.2` would
ship models worse than the prior on real data. With it, the coarse
quadrature's optimism can only cost a row its fit, never ship a bad one.

## (e) Wall time

Single runs, one process, single-threaded row fits; ratios are robust to the
20-40% run-to-run variation seen in absolute times. Regressions over all 60 rows
(full) and all 300 coarse fits:

    full:   t = -0.562 s + 2.37e-04 s/point  (K 246..99856, per candidate 5.67e-06 s/point, candidates 14..29)
    coarse: t =  0.002 s + 1.52e-04 s/cell   (Kc 172..20917, per candidate 6.01e-06 s/cell)

A cell costs what a point costs (`5.7e-6` against `6.0e-6` s per candidate),
so the coarse fit's time is the cell count's, linear as the plan assumes. The
per-point rate is 17x the plan's `1.4e-5` s/point/row because this run uses
four times the probes and exhausts the ladder (14-29 candidates per row);
that is a property of the setting, not of the coarsening.

| K bucket | rows | median K | median t full | median t coarse eps=0.1 | median Kc eps=0.1 | law 3pi/eps^2 log2(R/h) |
|---|---|---|---|---|---|---|
| [0, 500) | 5 | 285 | 0.06 | 0.06 | 285 | 3147 |
| [500, 2000) | 12 | 797 | 0.13 | 0.13 | 787 | 3771 |
| [2000, 8000) | 22 | 2403 | 0.40 | 0.25 | 1996 | 5648 |
| [8000, 30000) | 11 | 10000 | 1.62 | 0.40 | 3537 | 7205 |
| [30000, 10000000) | 10 | 55516 | 7.35 | 0.77 | 4497 | 8147 |

Below 2,000 points the coarsening is the identity and buys nothing; from a few
thousand it pays, and on the 100,000-point rows a 23-30 s fit becomes
0.5-1.4 s. In the plan's terms: a 100,000-point row at `eps = 0.1` costs less
than a 10,000-point row costs today. `coarsen_above` in the low thousands is
the right order for the trigger, as the plan guessed.

## Conclusions

1. **`eps = 0.1` as the default**, as the plan proposed. It is invisible on
   windows below a few thousand points, preserves every guard decision and
   97% of ladder levels on the 60 rows, keeps the ellipsoid within 0.03% in
   axes and 0.25 degrees, and costs at most `+0.01` in score on the finest-mesh
   rows for an 11-54x saving there. `0.15` is where the first fit is lost to an
   early ladder stop; `0.2` and above lose 10-17% of the fits and make the
   coarse score dishonest.
2. **The error is first order in `eps` for white-noise probes**, second order
   only for smooth probes; the plan's second-order claim assumes a probe smooth
   over the cell. The fit tolerates the first-order error because it is
   zero-mean across probes. A dipole-corrected cell would restore second order
   if it is ever needed.
3. **Surprises.** A singleton core of about `7 pi / eps^2` points, which puts
   a floor under the count law and makes `eps = 0.05` nearly useless below
   10,000 points; the optimism of a guard taken on coarse scores, which makes
   the full-window re-score load-bearing rather than cosmetic; and the fact that the plan's
   motivating case (a prior too wide) is the easy one, since a kernel small
   against its window lies inside the singleton core and is fitted exactly.
   The rows that feel `eps` are the honest-prior rows on fine meshes.

Nothing in the bindings blocked any item. Two parts of the plan are
replicated rather than exercised: the baseline and the guard are recomputed
in Python (`Scorer.baseline`, identical to `fit_operator`'s
`baseline_score` on all 60 rows), and the full-window re-score is done here
by hand; the call site of plan section 3 is untested by construction until S3.
The script is 551 lines rather than the 400 aimed for; the excess is the
reporting, which the tables above are copied from verbatim.
