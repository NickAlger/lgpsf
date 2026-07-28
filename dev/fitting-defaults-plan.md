# Plan: less conservative fitting defaults

Written 2026-07-28, mid-investigation, so a fresh session can continue without
the conversation. Two experiments remain. Everything here is checked against
the code as it stands at commit `82c813d`.

## Why

The per-row search is deliberately over-provisioned. At every rung of the mode
ladder it runs `1 + num_rungs` window-shaped starts plus `num_rungs` circular
ones, plus a warm start — 14 nonlinear fits per rung at the defaults, ~70 per
row over a five-rung ladder. The goal is to find defaults that cost less
without costing accuracy or robustness, and to give the user knobs where the
right answer depends on their problem.

## Where things stand

**Landed.**

- `ProbeFitConfig::circle_rungs_above_aspect`, shipped at **1.0 = current
  behaviour** (an aspect ratio is always ≥ 1, so 1.0 always adds the circles).
  Deliberately a no-op on landing; the default is what the experiments decide.
  Named to be unmistakable against `OperatorFitConfig::window_aspect_cap`,
  which shapes the WINDOW and is a different thing entirely.
- `experiments/fitting_defaults.py` and `fitting-defaults.md` — the frog sweep,
  60 whole-operator fits over five prior qualities and two anisotropies.

**Found, and it reframes the whole question.** Nothing in the initial-guess
ladder changed any answer: `num_rungs` 6→2 is 2.2× faster, dropping the
window-shape family 1.6×, never generating circles 1.8× — all at identical
error, under every damaged prior tested.

The reason is that **a bad prior binds through the WINDOW, not the initial
guess.** `fit_operator` derives both from the same `sigma`, so a rotated prior
gives a rotated window, and deployed support is the fit window:

| prior | window | rel. error |
|---|---|---|
| correct | ellipsoid | 0.1343 |
| rotated 90° | ellipsoid | 0.6305 |
| rotated 90° | ball (`window_aspect_cap = 1`) | 0.1386 |

So the frog experiment **cannot** distinguish "circle rungs are useless" from
"circle rungs were never tested, because the window masked the effect they
exist to fix". Keep that caveat attached to any conclusion.

**Recommended but NOT applied**, pending the PIG run below:

| knob | now | proposed |
|---|---|---|
| `ProbeFitConfig::num_rungs` | 6 | 3 |
| `ProbeFitConfig::circle_rungs_above_aspect` | 1.0 | 3.0 |
| `ProbeFitConfig::window_shape_rungs` | on | unchanged — the frog cannot see what it is for |

---

## Task A: the Levenberg–Marquardt tolerance

**It is already a user-settable option.** Do not add one. Verified 2026-07-28:

```python
config = lgpsf.OperatorFitConfig()
config.row.varpro.ftol = 1e-4        # works; pybind11 returns a live reference
config.row.varpro.max_evaluations = 40
```

`VarProOptions` exposes `ftol`, `xtol`, `gtol` (all `1e-8`), `ridge` (`1e-8`),
`jacobian`, and `max_evaluations`. `ProbeFitConfig::varpro` overrides
`max_evaluations` to 100. `probe_fit.hpp` passes `config.varpro` straight into
`fit_varpro`, so the value reaches the solver at both layers.

What remains is choosing a default. `1e-8` on a fit whose selection criterion
is a cross-validation score in the 1e-1 range is almost certainly far tighter
than the problem justifies — the outer loop is polishing digits the selection
rule cannot see.

**The experiment.** Extend `experiments/fitting_defaults.py`, or add a sibling:

- Sweep `ftol = xtol = gtol` over `1e-8, 1e-6, 1e-4, 1e-3, 1e-2`, and
  `max_evaluations` over `100, 50, 25`.
- Report, as the existing sweep does: relative Frobenius error, mean held-out
  score, and wall time (faster of two runs).
- Also report **mean LM iterations per candidate**, which is the quantity the
  tolerance actually controls. `VarProResult` carries `num_iterations` and
  `num_residual_evaluations`, and `ProbeFitResult.candidates[i].num_iterations`
  exposes it per candidate at the row layer. The operator layer does not
  aggregate it, so either do a row-layer sweep for that number or add it up
  from a handful of representative rows.
- Watch for the failure mode that matters: a loose tolerance should cost
  accuracy *gradually*. If the error is flat and then jumps, the loop is
  terminating before it reaches the basin, and the last safe value is not the
  one just before the jump.

**Interaction to check.** A looser tolerance may make the initial-guess ladder
matter *more* — a sloppy solve from a bad start may not reach the same
optimum. So sweep tolerance at both `num_rungs = 6` and `num_rungs = 3`; if
they disagree, the two knobs are not separable and the recommendation must
name a pair.

---

## Task B: rough-beta Pine Island Glacier

The frog is smooth, synthetic, 2-D, and its prior is analytically exact before
being deliberately damaged. Sixty unanimous results from it are suspicious on
their own. The real test is a PDE-derived Hessian with a prior that is wrong in
correlated, spatially varying ways.

**Where.** The `lgpsf_cxx_replay` directory of the glaciology research repo —
see `docs/validation.md` for what that is. This work belongs THERE, not in
lgpsf: it needs private data, and `tools/check_dependencies.py` enforces that
lgpsf never points into it. (Which is why this section names no path: the
check refuses one, correctly, even here.)

**Why rough beta specifically.** Slice 39, the α=0.01 MAP state. Its a-priori
sigma is known to be **3–5× too wide at the channel** — a genuinely bad prior
of the kind the frog can only imitate. It is also where the baseline-guard
counting-rule change cut the other way (−7% at k=20 versus +16% at smooth-beta
k=40), so it is the state that has historically disagreed with the easy one.

**Setup.**

```sh
export LGPSF_ROOT=<this checkout>
cd <the replay directory named above>
python rb_dump.py                    # regenerates data/, ~450 MB, gitignored
g++ -O3 -march=native -std=c++17 -pthread -I $LGPSF_ROOT/include \
    -I $LGPSF_ROOT/../ellipsoid_tree/include \
    -isystem $LGPSF_ROOT/build/_deps/eigen3_src-src -o rb_fit rb_fit.cpp
./rb_fit smoothed 50 4               # <sigma variant> <k> <threads>
```

`rb_fit.cpp` takes an aspect cap as its fourth argument, which is how to run
the ball-versus-ellipsoid window comparison the frog result now makes
interesting. It reports QC against 30 held-out pairs plus impulse-column
forensics.

**What to measure.** The same knobs as the frog sweep — `num_rungs`,
`circle_rungs_above_aspect`, `window_shape_rungs`, and the LM tolerance from
Task A — at k = 20 and k = 50, against the held-out QC score. There is no dense
truth here; held-out score is the metric, which is the honest situation anyway.

**The question that matters most.** On the frog, circle rungs never helped, but
the window confound means that is not evidence they are useless. Rough beta has
a genuinely bad prior *and* rows of varying anisotropy, so if circles ever earn
their keep it is here. If they do not, `circle_rungs_above_aspect = 3` is safe
to adopt. If they do, the gate needs to key on something other than the prior's
aspect ratio.

---

## Gotchas that will otherwise be rediscovered

- **Build with `-j2` on this machine, `-j1` for sanitizers.** The heaviest
  translation unit peaks at 2.8 GB (7.9 GB under ASan). `-j3` is permitted by
  the hook but locked the desktop on 2026-07-28. Symptom is a frozen machine,
  not an OOM message — it swaps and the build completes anyway.
- **Build Release.** Debug is ~33× slower, enough to look hung.
- **Rebuild the bindings after touching a header.** `build-py` does not track
  header changes into an already-imported module; a stale `.so` shows up as
  `AttributeError` on a new config field.
- **Do not infer that a background job died** from a quiet log and an empty
  `pgrep`. That was wrong twice in one session. `pgrep -x` also truncates
  names at 15 characters, and `pgrep -f <pattern>` matches the shell running
  it. Wait for the completion notification.
- **`ndarray.tofile()` always writes C-order** and silently ignores
  `asfortranarray`. Write `A.T` when you want column-major `A`. This is what
  makes the raw-binary bridge work; `pig_check.py` guards it.
- **A PIG score is meaningless without naming the deployment variant.**
  Slice 38 has three: parametric, window-clipped, and clipped+symmetrized.
- **Results are bit-identical within a build, not across builds.**
  `-march=native` reorders reductions and ~0.08% of rows land in a different
  local minimum. Compare field statistics, not rows, unless the flags match.

## When the answers land

Update in this order, all in one commit per decision:

1. The default in `include/lgpsf/probe_fit.hpp`.
2. `docs/defaults.md` — the table AND the evidence sentence beneath it.
3. `experiments/fitting-defaults.md` — mark the recommendation applied or
   overturned, with the PIG numbers.
4. `CHANGELOG.md` under Unreleased, since these are behaviour changes.
