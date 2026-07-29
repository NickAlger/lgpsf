# Does the initial-guess ladder have an anisotropy blind spot? 2026-07-28

**Answer, at 8:1 and under the shipping defaults: no.** This experiment was
built to confirm a suspected weakness and did not confirm it. It is kept
because it is the only problem in the repo anisotropic enough to ask the
question, and because the answer is conditional on a default that could change.

Measured by [`anisotropy_hardening.py`](anisotropy_hardening.py); full output
in [`anisotropy-hardening.txt`](anisotropy-hardening.txt).

## The worry

The initial-guess dictionary offers `sigma0`, circles at `num_rungs` scales,
and optionally scaled copies of the window's shape. Every entry is either the
prior's own shape or **round** — nothing covers **orientation** independently
of the prior. If a target is strongly anisotropic and the prior's orientation
is wrong, radius coverage cannot help, because the axis the search must
discover is not the one being swept.

This is not a new suspicion. The Python prototype's
`varpro_frog_robust_init.py` found it at 8:1 and
[`dev/robust-init-notes.md`](../dev/robust-init-notes.md) parked a recipe for
an oriented-ellipse family. `lgpsf::oriented_sigma(a, b, angle_degrees)` was
ported for it and is tested — its docstring calls orientation "the circle
family's blind spot" — but **no rung family is built from it**, so nothing in
the library reaches it.

## The setup

The frog kernel hardened to 8:1 (1-σ axes 0.100 × 0.0125), grid 81 so the thin
axis and its modulation resolve, one row at (0.549, 0.352) where the kernel's
local rotation is 99°, fitted on a ball of 2537 points.

A **single-start** fit is expressible through the shipping API: set `sigma0` to
the guess and turn both rung families off, and the dictionary reduces to that
one entry; a one-level mode policy removes the warm start too. Each row of the
tables is therefore one fit from one starting ellipsoid.

Two mode budgets, from the prototype: **hard** (20 probes, 3 modes) and
**comfortable** (60 probes, 21 modes). Two center policies, because the
prototype's catastrophic cases were all free-mu and its own FIX C found that
pinning "collapses the outcome variance — no runaways anywhere". **Pinning is
now the library default.**

## Reaching the good basin

Outcomes are strongly bimodal: starts that succeed land within a few parts in
ten thousand of each other, failures are ~2× worse. "Reached the good basin"
means within 1.1× of the best start of any family — nothing sits near that line.

| target | mu policy | circles | oriented ellipses |
|---|---|---|---|
| hard | **pinned (default)** | **4 / 6** | 4 / 8 |
| hard | free | 1 / 6 | **3 / 8** |
| comfortable | **pinned (default)** | **5 / 6** | 6 / 8 |
| comfortable | free | 2 / 6 | **5 / 8** |

**With mu free, the prototype's finding reproduces**: most single starts fail,
and the oriented family is the more reliable of the two (3/8 against 1/6, 5/8
against 2/6). The blind spot is real.

**With mu pinned it does not.** The landscape is far better behaved, and the
circles are at least as reliable as the oriented entries. Pinning removes two
degrees of freedom from the search, and with them most of the multimodality.

Circles do fail individually at both ends of the ladder, exactly as the
prototype reported — `r = 0.15` runs away to a fitted 43.75:1 against a true
8:1, and `r = 0.025` stalls at 3.39:1. What saves them is that the ladder
**brackets** the scale rather than guessing it.

## What the shipping default reaches

| target | mu policy | rel. error | vs best start of any family |
|---|---|---|---|
| hard | pinned | 0.5727 | 1.00× |
| hard | free | 0.5607 | 1.00× |
| comfortable | pinned | 0.0406 | 1.01× |
| comfortable | free | 0.0399 | 1.01× |

`sigma0` + 3 circles reaches, in every condition, what the best member of a
15-entry hand-built dictionary reaches. The multi-start is doing its job: no
individual start is reliable, and it does not have to be.

(The hard target's floor of 0.57 is a **capacity** limit, not an initialization
one — 3 modes cannot represent a modulated 8:1 PSF, and every successful start
agrees to four digits.)

## A round start does not bias the fitted answer toward roundness

This was the specific concern, and the `fit aspect` column settles it. Every
start that reaches the good basin — circular or oriented — reports the same
strongly anisotropic ellipsoid:

| | fitted aspect | fitted angle | true |
|---|---|---|---|
| hard, pinned | 12.44 – 12.46 : 1 | 98° | 8:1 at 99° |
| comfortable, pinned | 7.66 – 7.67 : 1 | 99° | 8:1 at 99° |

A circular start is a **scaffold the solver discards**. Levenberg–Marquardt
optimizes the full log-Cholesky parameterization, so the fitted ellipsoid is
free to become as anisotropic as the data wants, and it does — at the hard
budget it *overshoots* to 12.4:1 rather than staying round. Whatever is wrong
with circle rungs, imposing an isotropy bias on the answer is not it.

## Conclusion

| | |
|---|---|
| Is there an orientation blind spot? | Yes, under **free mu** — measurable here. |
| Does it affect the shipping defaults? | **No.** Pinned mu plus a scale-bracketing ladder reaches the best available start in every condition tested. |
| Do circular starts bias the answer round? | **No.** Fitted aspect ratios are 8–12:1 regardless of start. |
| Should the oriented family be built? | Not on this evidence. Revisit if mu release is ever re-armed. |

## What this does not measure

- **One row, one kernel, 2-D.** The prototype's conclusion came from the same
  kernel, so this is a re-measurement under new defaults, not independent
  confirmation.
- **A correct center.** `mu0` is the true center here; the prototype also
  tested a deliberately offset one, which is the case that most stresses a
  pinned fit. Not run.
- **Anything above 8:1**, and nothing in 3-D, where orientation space is SO(3)
  and a covering dictionary would be much larger — see the dimension-scaling
  caveat in `dev/robust-init-notes.md`.
