# Robust initialization for the per-row VarPro fit

**Status: PARKED (2026-07-24), re-measured 2026-07-28 — see the update at the
top before reading on.** Design note capturing the initial-guess robustness
experiments and the recipe they support. The evidence below is the PROTOTYPE's,
reproducible at `archive/python-prototype/examples/varpro_frog_robust_init.py`
with the sweep figures committed alongside; paths in the body are as they were
when this was written.

---

## UPDATE 2026-07-28: the oriented family was never built, and at present it is
## not needed

**The gap.** The recipe below prescribes an oriented-ellipse family as the
member that fixes the circles' orientation blind spot.
`lgpsf::oriented_sigma(a, b, angle_degrees)` was ported to
`include/lgpsf/init_dictionary.hpp` for it and is tested — its docstring calls
orientation "the circle family's blind spot" — but **no rung family is built
from it**. `probe_fit.hpp` assembles `sigma0` + window rungs + circle rungs +
warm start, and nothing calls `oriented_sigma`. No note recorded that as a
decision; it appears to be an omission.

**Re-measured against the live library** by `experiments/anisotropy_hardening.py`
(write-up: `experiments/anisotropy-hardening.md`), on the same 8:1 frog:

- The blind spot is **real under free mu** — oriented starts reach the good
  basin 3/8 and 5/8 of the time against circles' 1/6 and 2/6.
- It **does not appear under pinned mu**, which is now the default (`MuPolicy`
  changed after this note was written). There circles are at least as reliable,
  4/6 and 5/6 against 4/8 and 6/8.
- The **shipping default reaches the best start of any family** in every
  condition, 1.00–1.01×. Individual circles still fail at both ends of the
  ladder exactly as described below; what saves them is that the ladder
  brackets the scale rather than guessing it.
- A circular start does **not** bias the fitted answer toward roundness: every
  successful start reports 7.7–12.5:1 against a true 8:1.

**Therefore: still parked, and now with a trigger.** Build the oriented family
if mu release is ever re-armed as a default, if a problem with genuinely
anisotropic priors appears (every PIG row is below 1.7:1 by construction, so
the field-scale validation cannot see this), or if N = 3 arrives — where
orientation is SO(3) and the dictionary-size caveat at the end of this note
bites hardest.

Doing it properly means committing to a small research project, which is why it
is parked rather than half-built.

---

## The problem

The reduced (theta-only) VarPro landscape is benign when the LG basis is
adequate and the target is mildly anisotropic -- on the frog kernel at
its native 2:1 aspect, any circular init within a 5x radius range
converges to the same optimum. Under stress it becomes genuinely
multimodal. At 8:1 aspect (the same 81-degree rotation), observed on
both a probe-starved 3-mode fit and a comfortable 21-mode fit:

- **Circles fail in both directions.** The good basin narrows to a
  sliver in radius; larger circles land in a "displaced dipole" local
  minimum or diverge, and *smaller* circles converge to a
  right-scale-wrong-orientation minimum (thin but vertical instead of
  tilted) -- matching the ellipsoid_psf_pig experience that too-small
  inits also break. A circle carries no orientation information, and
  orientation is exactly what the landscape punishes.
- **The exact a-priori moment-ellipsoid shape is not safe either**: it
  landed in (near-miss) local minima on *both* targets. A-priori moments
  are a valuable init, not a sufficient one.
- **A level-0 (single-Gaussian) pilot fit is convergence-robust but not
  shape-reliable.** It never runs away (with one even positive mode,
  inflating the ellipsoid fits localized data badly, so the cost itself
  pushes back), which makes it cheap insurance -- but at high anisotropy
  its own 1-mode landscape is multimodal and a single positive Gaussian
  cannot lock onto a strongly modulated target, so its fitted shape is
  untrustworthy. Warm-starting an enriched fit at the pilot optimum has
  an additional structural trap, the enrichment saddle (below).

## The parked recipe: portfolio multi-start over an init dictionary

Run the target fit from every member of a small, deliberately *diverse*
dictionary of initial ellipsoids and keep the best final cost:

- circles at a few scales (log-spaced around the expected size);
- oriented ellipses at a coarse angle grid x a moderate aspect (e.g.
  4 angles x 4:1 x 1-2 scales in 2D) -- this is the member family that
  fixes the circles' orientation blind spot, and the correctly-oriented
  entries found the best known solution on every broken case;
- the a-priori moment-ellipsoid shape, possibly at 2-3 scales;
- optionally the level-0 pilot's shape at 2-3 scales (cheap: 1-mode
  fits cost almost nothing).

In the head-to-head at 8:1, the union portfolio was the only strategy
that never lost; each individual family lost somewhere.

**Why this is affordable:** every fit in the portfolio reuses the same
probe data -- matvecs, the method's expensive currency, are spent once
per row regardless of portfolio size -- and the fits are tiny
(P <= 14-dimensional LM on a k x n_modes problem) and embarrassingly
parallel. Selection is by final whitened cost; if overfitting across
portfolio members ever becomes a concern, held-out probes can arbitrate
(same mechanism the mode-selection plan calls for).

**Dimension scaling caveat:** the angle grid is 1-D only in 2D. In 3D,
orientation space is SO(3); a coarse covering (e.g. the rotation group's
~24-element octahedral subset x a couple of aspect patterns) is still
plausible but the dictionary grows -- revisit when N=3 matters.

## Related structural facts (established on the way)

- **Enrichment saddle:** growing the basis by one energy level and
  warm-starting at the previous optimum with free mu always starts at a
  point where the mu-optimality conditions kill part of the new modes'
  first-order signal -- *all* of it for level 0 -> 1, since the dipole
  modes are exactly the mu-derivatives of the Gaussian, so the pilot
  optimum is a stationary point (saddle) of the enriched problem.
  Second-moment (L) optimality similarly kills the level-2 signal.
  Confirmed empirically both ways: exact-warm-start LM stops after one
  iteration on a level<=1 target, but escapes on a level<=5 target
  (levels 3+ still carry gradient). Full argument in
  `examples/varpro_frog_fit.py`'s module docstring.
- **Overflow sentinel:** wild trial thetas are survivable
  (`varpro._ReducedProblem._solve_at` scores them as worst-finite-cost),
  which multi-start relies on.

## Fixed-mu stage + release (explored 2026-07-24; strong)

In the real pipeline mu is the known node location, so pinning it is
free information. Encoding hooks existed at every layer already (the
`mu0=` switch); `release_mu`/`freeze_mu` in `ellipsoid_transform.py`
convert between the encodings (the free-mu theta is `[mu, fixed-mu
theta]` by construction, tested).

Results at 8:1: pinning mu collapses the outcome variance -- no
runaways, inits coalesce into a few basins. Comfortable target: with mu
pinned even at a *wrong* (offset) center, every tested radius converged
to the same pin-limited solution and every release recovered the global
best; this was the strongest single strategy observed. Probe-starved
3-mode target: the shape landscape stays multimodal under the pin, but
the smallest-radius member + release reached the global best, and every
release improved on its fixed stage (releases are cheap and never
hurt; the release warm start has no enrichment saddle since the
fixed-stage optimum is not mu-stationary). Recommended role in the
portfolio: primary strategy whenever mu is known a priori, with circles
across scales as its init family.

## Open ideas, not yet explored

- **Staged isotropic-constrained encoding**: fit (mu, log r) only
  (P = N+1), then release anisotropy -- a new small stage-2 theta
  encoding, which `ellipsoid_transform.py`'s two-stage design was built
  to accommodate.
