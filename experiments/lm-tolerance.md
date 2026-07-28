# How tight the Levenberg–Marquardt tolerance needs to be, 2026-07-28

**PROVISIONAL**, on the same terms as [`fitting-defaults.md`](fitting-defaults.md):
one smooth synthetic two-dimensional kernel. Measured by
[`lm_tolerance.py`](lm_tolerance.py); full output in
[`lm-tolerance.txt`](lm-tolerance.txt).

## The question

Every candidate in the per-row search is a nonlinear least-squares fit, run to
`ftol = xtol = gtol = 1e-8`. Nothing downstream reads a fit to eight digits:
candidates are selected on a held-out cross-validation score, and on a
well-fitted row that score sits around `1e-1`. So the outer loop may be
polishing digits the selection rule cannot see.

The tolerance was already a user option — `config.row.varpro.ftol` and friends,
at both the row and operator layers. What was open is the default.

## `ftol` is the only test that binds

The three tolerances stop on different quantities, so before sweeping them
together it is worth knowing which one fires. Loosened one at a time, on 25
interior rows fitted individually:

| loosened to 1e-4 | LM iterations/row | fewer |
|---|---|---|
| nothing (all at `1e-8`) | 1184.8 | — |
| `ftol` only | 653.0 | **1.81×** |
| `xtol` only | 1030.1 | 1.15× |
| `gtol` only | 1046.1 | 1.13× |
| all three | 643.0 | 1.84× |

`ftol` accounts for essentially the whole effect, and MINPACK's semantics
predict that. Its test is on the **relative** cost reduction, `1 − (‖f₊‖/‖f‖)²`,
which decays smoothly to zero as any optimum is approached. `xtol` needs the
trust region to shrink below `xtol · ‖x‖`, which happens only on rejected steps.
`gtol` tests the largest cosine between the residual and a Jacobian column — a
quantity that never approaches zero here, because the reduced residual at the
optimum is data noise rather than zero.

The three are OR-ed, so once `ftol` is loose the others add only 1.02× on top of
it. They are moved together anyway, so that "the tolerance" is one number, but
**`ftol` is the one to reach for** when tuning further.

## Accuracy does not move, across five decades

With the mode ladder pinned to full depth — which is the only configuration in
which the error column is a statement about the solver, for reasons below — the
result is as flat as it gets. 25 rows, `num_rungs = 6`:

| tolerance | iterations/row | rel. error | score |
|---|---|---|---|
| `1e-8` | 1207.6 | 0.122000 | 0.117308 |
| `1e-6` | 917.5 | 0.122001 | 0.117305 |
| `1e-4` | 656.0 | 0.122001 | 0.117281 |
| `1e-3` | 517.8 | 0.121998 | 0.117233 |
| `1e-2` | 371.0 | 0.121960 | 0.117054 |

**3.25× fewer iterations, and the error is unchanged to four significant
figures.**

Whole operators agree. 576 rows, correct prior, relative Frobenius error against
the dense truth:

| tolerance | rel. error | modes | speedup, `num_rungs = 6` | `num_rungs = 3` |
|---|---|---|---|---|
| `1e-8` | 0.127124 | 14.8 | — | — |
| `1e-6` | 0.127124 | 14.8 | 1.26× | 1.21× |
| `1e-4` | 0.127126 | 14.8 | **1.58×** | **1.46×** |
| `1e-3` | 0.127129 | 14.8 | 1.72× | 1.74× |
| `1e-2` | 0.127068 | 14.8 | 2.09× | 1.89× |

The same number of modes ships at every setting, and what movement there is sits
in the sixth digit and is not monotone — `1e-2` came out *lower* than `1e-8`.
That is perturbation, not degradation.

**The two knobs are separable.** The tolerance behaves the same at both rung
counts, and `num_rungs` 6 → 3 is worth 1.5–1.7× at every tolerance. Taking both,
9.1 s → 3.8 s at identical error. Disabling the early exit and damaging the
prior to carry no shape information both leave the picture unchanged.

## The confound: the ladder reads the score

Run the row layer the ordinary way, ladder free, and the error appears to
*improve* by 11% at `1e-3` and looser. That is not the solver getting better.

Both of the mode ladder's stopping rules — `target_score` and `mode_patience` —
read the cross-validation score. A tolerance change that moves a score in its
fifth decimal can flip whether a level counted as an improvement, and ship a row
with a different number of modes. Modes are worth far more than solver digits.

That is exactly what happened, and the two blocks line up entry for entry. At
`1e-3` and `1e-2` the free-ladder run reports 69.0 candidates, 14.8 modes,
517.8 and 371.0 iterations, and errors 0.121998 and 0.121960 — **every one of
those identical to what the pinned-ladder run reports at the same tolerance**.
The loose solve did not fit better. It stopped stopping early, and ran the same
full ladder the pinned configuration always runs.

So: **an error column that moves with the tolerance is more likely reporting a
ladder-depth flip than solver quality.** It is why the tables above are the
pinned-ladder and whole-operator ones, where the mode count is constant.

## Where the cliff is not

A tolerance loose enough to stop the loop before it reaches the basin would not
degrade accuracy gradually — it would fall off a cliff. Nothing in the range
tested is near one. That is the useful finding, but it also means **this
experiment does not locate the cliff**: it says only that `1e-2` is not past it
on this problem. A default should not be placed at the loosest value that
happened not to break.

There is a measured threshold to use instead, and it comes from the confound
above rather than from the accuracy tables. The thing a loose tolerance can
actually corrupt is not the fit — it is the **selection**, because the ladder
reads the score. So the question is how big a score perturbation each tolerance
induces, against how big a perturbation it takes to change a decision:

| `ftol` | score perturbation | ladder decisions |
|---|---|---|
| `1e-6` | 3.0 × 10⁻⁶ | unchanged |
| `1e-4` | 2.7 × 10⁻⁵ | unchanged |
| `1e-3` | 7.5 × 10⁻⁵ | **flipped** — 67.9 → 69.0 candidates, 14.2 → 14.8 modes |
| `1e-2` | 2.5 × 10⁻⁴ | flipped |

(Perturbations read off the pinned-ladder block, which is the only one where a
flip cannot contaminate them; the flip column from the free-ladder block beside
it. The whole-operator block gives the same perturbations to one digit.)

**`1e-4` is the last tolerance at which the ladder makes every decision it makes
at `1e-8`.** That is a directly measured margin rather than a theorized one, and
it is the reason for the recommendation below.

A scaling argument agrees in spirit but should not be leaned on. `ftol` limits
the relative cost reduction of a single accepted step, and the CV score is a
residual-norm ratio, which suggests a score perturbation around `t/2` in
relative terms. The perturbation does scale with `t` roughly as that predicts,
but `t/2` is **not a bound** — the observed values exceed it by 51×, 4.6× and
1.3× at `1e-6`, `1e-4` and `1e-3`, and only fall below it at `1e-2`. That is
expected: stopping when one step buys less than `t` says nothing about how much
total progress remains. Treat it as an order-of-magnitude guide.

One further datum sits where the argument says the first crack should be. Under
the shapeless prior — the weakest condition tested — `1e-2` is the only
condition-and-tolerance combination anywhere in the sweep where the error moved
the *wrong way* by more than noise: 0.136543 → 0.136612, 5 × 10⁻⁴ relative.
Everywhere else `1e-2` happened to land slightly better.

## `max_evaluations` is not what binds

Dropping the evaluation cap from 100 to 25 is worth 1.23× at the row layer and
**nothing at all** on whole operators, at either end of the tolerance range,
with no change in error anywhere. The loop stops on `ftol` long before it runs
out of evaluations, so the cap is a runaway guard rather than a performance
knob. Left alone.

## Read the wall-clock ratios loosely

The baseline configuration appears in two blocks and measured 9.1 s in one and
10.1 s in the other, so **ratios below about 1.15× here are noise**, even though
each configuration is fitted twice and the faster run reported. The
`max_evaluations` block shows the same thing from the other side: 0.97× and
0.99× "speedups" for strictly cheaper settings.

The effects this rests on are larger than that, and the iteration counts are not
timings at all — they are exact, deterministic, and reproduce run to run. Those
are the numbers to weigh.

## Recommendation

| knob | now | proposed |
|---|---|---|
| `varpro.ftol` | `1e-8` | **`1e-4`** |
| `varpro.xtol` | `1e-8` | **`1e-4`** |
| `varpro.gtol` | `1e-8` | **`1e-4`** |
| `varpro.max_evaluations` | `100` / `50` | unchanged — not the binding constraint |

`1e-4` rather than the faster `1e-2` because it is **the last tolerance at which
the mode ladder makes every decision it makes at `1e-8`**. Past it the fits
themselves are still fine — the accuracy tables stay flat all the way to `1e-2`
— but the *selection* starts moving, and a default should not sit where a knob
nominally about solver precision has begun to change which model ships. That it
moved in a helpful direction on this problem is luck, not a reason.

It takes 1.58× of the 2.09× available in wall time, and 1.84× of the 3.25×
reduction in iterations.

## What this does not measure

- One smooth synthetic two-dimensional kernel, as with the ladder sweep. The
  same caveat applies and for the same reason.
- Only `MuPolicy.Pinned`. Released-mu fitting has two more parameters and a
  flatter objective near the optimum, where a loose `ftol` has more room to stop
  early.
- Where the cliff is. The tested range does not reach it.
