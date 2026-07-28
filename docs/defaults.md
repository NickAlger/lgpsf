# Defaults, and the evidence for them

Most of these were chosen from measurement rather than taste. Where the
evidence is a field-scale experiment, [validation.md](validation.md) says what
the problem was.

## `OperatorFitConfig`

| | default | |
|---|---|---|
| `tau_window` | `10.0` | Window radius, in standard deviations of the `sigma` you supplied. Deliberately generous — the admissibility guard assumes the window contains the row. |
| `window_aspect_cap` | `inf` | Window shape. `1` gives a ball, `inf` your ellipsoid untouched, anything between caps the axis ratio. |
| `spike` | `true` | Fit a diagonal correction for the part of the PSF the mesh cannot resolve. Turn it off when the kernel is mesh-resolved, and it must be off for rectangular operators, where no diagonal exists. |
| `seed` | unset | Unset means **no generator is consulted at all**: folds are round-robin and the jitter table is fixed. Runs are reproducible by construction, not by seeding. |
| `num_threads` | `0` | Implementation chooses. Results are bit-identical for any value. |
| `row.mode_policy` | **none — required** | See below. |

## `ProbeFitConfig`

| | default | |
|---|---|---|
| `mu` | `MuPolicy.Pinned` | The center stays where you put it. |
| `num_rungs` | `6` | Log-spaced circle scales in the initial-guess ladder. |
| `window_shape_rungs` | `true` | Also try scaled copies of the window's own shape. |
| `target_score` | `0.05` | Stop early once a candidate is certifiably good. `None` sweeps the whole ladder. |
| `mode_patience` | `2` | Stop growing modes after this many rungs without improvement. |
| `cv_folds` | `5` | Held-out folds for the selection score. |
| `varpro.ridge` | `1e-8` | Damps the linear coefficients only — the ellipsoid is never regularized. |
| `varpro.jacobian` | `Kaufman` | Drops a term that vanishes at the solution; one reverse sweep instead of a full Jacobian tensor, same answer. |
| `varpro.ftol` | `1e-8` | Stops one nonlinear fit on relative cost reduction. **This is the one that binds** — see below. |
| `varpro.xtol` | `1e-8` | Stops it on relative step size. |
| `varpro.gtol` | `1e-8` | Stops it on gradient orthogonality. |
| `varpro.max_evaluations` | `100` | Runaway guard, not a performance knob: the loop stops on `ftol` long before it reaches this. |

## The solver tolerance is loosenable, and `ftol` is the knob

The three tolerances are OR-ed and stop on different quantities, but `ftol` is
the only one that fires in practice: loosening it alone to `1e-4` gives 1.81× of
the 1.84× available from loosening all three. `xtol` needs the trust region to
shrink on rejected steps, and `gtol` tests a cosine that never approaches zero
here, because the reduced residual at the optimum is data noise rather than
zero.

Loosening it is nearly free in accuracy. With the mode ladder held at fixed
depth, `1e-8` → `1e-2` is 3.25× fewer LM iterations and leaves the error
unchanged to four significant figures.

What it does affect is **selection**, because the mode ladder reads the
cross-validation score: at `1e-3` a score perturbation of 7.5 × 10⁻⁵ was enough
to change how many modes shipped. `1e-4` is the loosest setting measured at
which every ladder decision matched `1e-8`, so **`1e-4` is the recommended value
if you want the fit cheaper** — worth about 1.6× — and further than that should
be checked against your own held-out scores rather than assumed.

Measurements in [`experiments/lm-tolerance.md`](../experiments/lm-tolerance.md).

## The mode policy is required, not defaulted

`fit_operator` throws if `config.row.mode_policy` is unset. That is deliberate:
the best growth order is **budget-dependent and problem-dependent**, and no
single choice is defensible as a silent default.

At field scale, complete shells won at k = 20 while `WedgeLadder(10, 2)`
matched them at k = 100 for a sixth of the modes. On the rotating frog kernel
in the examples, whose modulation is strongly angular, the wedge loses at every
budget because capping `|ell| ≤ 2` discards exactly what that problem needs.

`WedgeLadder(10, 2)` is the recommended starting point for operators whose rows
are roughly elliptical. If yours have angular structure, say so.

## The center is pinned by default

`MuPolicy.Pinned` rather than releasing it after a first pass. Releasing
*shipped* on about 91% of rows in a field-scale fit while buying nothing on
held-out score, guarded only by the window radius — far too loose a bound on
how far a center may wander. Release remains available on request; it will
become automatic again if a basin-scale bound on `||mu − mu0||` is added.

## Deployed support is the fit window

Every evaluation — `matvec`, `assemble_sparse`, `eval_entries`, `eval_kernel` —
restricts each row to the window it was fitted on. `eval_kernel_unrestricted`
is the named opt-out.

This is not a performance shortcut. The cross-validation score is blind to
model energy outside the window, so out-of-window mass is *unpenalized*, not
merely unverified — and Laguerre-Gaussian modes, being polynomials times a
Gaussian, extrapolate violently. At field scale a single rogue row with an
unconstrained tail was enough to dominate the global error. Fitted object and
deployed object are now the same object by construction.

## Do not gate dead rows

`fit_operator` attempts every row by default, and that is the recommended
setting **even when you know some rows carry no signal**. A zero-response row
costs one candidate: the coefficients come out zero, the CV score is exactly
zero, the baseline ties, and it ships a model predicting exactly zero. It
cannot fail and cannot produce NaN.

Measured over 6557 rows: ungated 159.9 s versus gated 162.4 s — gating was
*slower* — and every live row's prediction was bit-identical either way. The
gate is for rows you do not want modeled, not for rows you expect to be hard.

## The counting rule

A mode set of size `m` is admissible only when `k ≥ 2(m + P)`, with `P` the
number of ellipsoid parameters actually being fitted — 3 in 2-D pinned, 5 free.
Over-large sets are skipped rather than fitted.

The margin is conservative on purpose. The first inadmissible rung often still
scores well; the rule has to hold for every row of an operator, including those
with the least signal. [`examples/counting_rule.py`](../examples/counting_rule.py)
shows what happens past it: the residual collapses toward zero while the
held-out score gets worse.

## The always-on baseline guard

Every row is also fitted the cheap a-priori way — an LG fit at your `sigma`,
pinned, no search — and scored on the same folds. The searched fit ships only
if it is strictly better. **A fit is therefore never worse than the prior you
supplied**, and a row reporting `FallbackBaseline` is telling you the search
found nothing the prior did not already have.
