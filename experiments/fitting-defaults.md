# What the initial-guess ladder buys, 2026-07-28

**PROVISIONAL.** One synthetic kernel, two dimensions, smooth everywhere. The
numbers below are consistent and one-sided, which is suspicious enough on its
own; nothing here should change a default until it survives a real operator.

## The question

The per-row search is conservative by construction. At every rung of the mode
ladder it tries `1 + num_rungs` window-shaped starts and `num_rungs` circular
ones, plus a warm start — 14 nonlinear fits per rung at the defaults, roughly
70 per row over a five-rung ladder. How much of that is load-bearing?

Measured by [`fitting_defaults.py`](fitting_defaults.py): whole-operator fits
on a 24×24 grid, 45 probes, shells to level 6, over five prior qualities and
two kernel anisotropies. Prior quality is the axis that should decide this — a
setting that only works when `sigma0` is right is not a default but a trap for
whoever supplies a mediocre prior, which is everyone on a real problem.

Full output: [`fitting-defaults.txt`](fitting-defaults.txt). Read timing
ratios rather than seconds; each configuration is fitted twice and the faster
run reported, because the first fit in a process pays warm-up costs larger
than most of the effects here.

## Nothing in the ladder changed an answer

Sixty whole-operator fits. Every relative-error ratio came out **1.00**, with
one 0.99 and two 1.01. Representative, on the 4:1 kernel:

| setting | rel. error | speedup |
|---|---|---|
| `num_rungs = 6`, circles always **(default)** | 0.1343 | — |
| `num_rungs = 3` | 0.1342 | 1.7× |
| `num_rungs = 2` | 0.1342 | 2.2× |
| `window_shape_rungs` off | 0.1342 | 1.6× |
| circles never | 0.1343 | 1.8× |

That holds under a 90°-rotated prior, a 4× too wide prior, a 4× too narrow
prior, and a prior with no shape information at all. The ladder is not what
determines accuracy on this problem, at any prior quality tested.

`circle_rungs_above_aspect = 3` behaves as designed: on the 4:1 kernel with a
correct prior it keeps the circles and costs nothing (1.00×), and on the
isotropic prior it drops them for 1.74×. The gate fires where it should.

## Why: the prior binds through the WINDOW, not the initial guess

A damaged prior does cost accuracy — a 90° rotation takes the 4:1 kernel from
0.134 to 0.631. But no amount of extra searching recovers it, which is not what
an initialization failure looks like.

It is not one. The window is derived from the same `sigma`, so a rotated prior
produces a rotated window, and deployed support is the fit window:

| prior | window | rel. error | mean window |
|---|---|---|---|
| correct | ellipsoid | 0.1343 | 111 pts |
| rotated 90° | ellipsoid (rotated too) | **0.6305** | 111 pts |
| rotated 90° | **ball** (`window_aspect_cap = 1`) | **0.1386** | 359 pts |
| correct | ball | 0.1328 | 359 pts |

**Switching to an orientation-free window recovers essentially all of it**:
0.63 → 0.14, against 0.13 for a correct prior. The fit was never failing to
find the shape; it was being asked to represent a row on the wrong region.

Two consequences worth carrying forward:

- **If you are unsure of your prior's orientation, use a ball window.** The
  ellipsoid window is worth 3× fewer points per row when the prior is right,
  and worth 4.7× the error when it is wrong.
- This experiment **cannot separate** "circle rungs are useless" from "circle
  rungs were never tested, because the window masked the effect they exist to
  fix". `fit_operator` derives both the window and the initial guess from one
  `sigma`, so damaging the prior damages both. Separating them would need the
  window's shape and the initialization's shape to be specifiable
  independently — which the API does not currently allow, and which may be
  worth adding precisely so this question can be answered.

## Provisional recommendations

Not yet applied. Pending a real operator.

| knob | now | proposed | basis |
|---|---|---|---|
| `num_rungs` | 6 | 3 | 1.7×, no measured accuracy cost; 2 is also clean but leaves no margin |
| `circle_rungs_above_aspect` | 1.0 | 3.0 | never changed an answer; gate fires as designed |
| `window_shape_rungs` | on | on | 1.6× available, but it is the family that survives when the window is the caller's ellipsoid, and this experiment cannot see that |

Together the first two would be roughly 3× fewer nonlinear fits per row.

## What this does not measure

- One kernel, smooth and two-dimensional, with a prior that is analytically
  exact before it is deliberately damaged. Real priors are wrong in
  correlated, spatially varying ways rather than by a clean global transform.
- Rows are all comparably well conditioned. There are no dead rows, no
  boundary-starved rows, and no rows where the counting rule binds hard.
- Only the default `MuPolicy::Pinned`. Released-mu fitting has more parameters
  to get wrong and might depend on initialization more than this does.
- Accuracy against a known truth. On a real problem the selection would be
  driven by held-out score alone, and the two can disagree.
