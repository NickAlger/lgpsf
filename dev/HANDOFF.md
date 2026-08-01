# Developer handoff — current state and open threads

Continuity between working sessions: what is in flight, what is owed, and
what is parked. For the stable project overview — the math, the architecture,
the conventions — read [`CLAUDE.md`](../CLAUDE.md) instead. This file is
narrative and open items only; when a thread closes, its record moves to
[`archive/`](archive/).

## Where things stand

| | state |
|---|---|
| C++ core | complete — 149 cases / 105,699 assertions |
| QR-first inner solve | landed — 2.1× on a whole-field fit, every number unchanged to four digits |
| Field-scale validation | smooth and rough basal friction both reproduced; the recorded 0.0147 matched exactly |
| Python bindings | complete — 51 pytest cases |
| Examples | 16, covering every exported name; the frog example is the public gate |
| Fitting defaults | **changed 2026-07-28** — prior + 3 circles, `ftol` 1e-4. See below; fits are not bit-identical to 0.1.0 |
| Initial-guess API | **changed 2026-07-29** — guesses are data, not flags. [`archive/initial-guess-api-plan.md`](archive/initial-guess-api-plan.md); `fit_operator` bit-identical across it |
| Anisotropy blind spot | **parked, measurable, and now reachable** — `oriented_ladder` ships opt-in; `experiments/anisotropy_hardening.py`. See below |
| User documentation | written — README, installation, quickstart, api-guide, defaults |
| Direction of dependence | enforced by `tools/check_dependencies.py` over docs, dev and experiments too |
| Repo cleanup | 5 of 6 slices done — see below |
| Fitting speed | **~4.5× faster than 0.1.0** at unchanged accuracy — see below |
| Pencil corrections layer | **slices 1–5 of 12 landed** (`Symmetrize::Weighted` 2026-07-31; boundary, `ModeBlock`, `ShiftedOperator` + GLR Woodbury, and pencil Lanczos + `make_pd` 2026-08-01) — full implementation plan in [`pencil-corrections-plan.md`](pencil-corrections-plan.md); the layer is operator-blind (matvec-only) in `include/lgpsf/corrections/`, a sibling of the fit stack |
| **Not done** | **CI, cibuildwheel, M6 release infra** |

## Fitting defaults changed, 2026-07-28

The initial-guess dictionary is now **`sigma0` + 3 circles** (plus a warm
start), where it was `sigma0` + 6 circles + 6 window-shaped starts, and the LM
tolerance is `1e-4` rather than `1e-8`. About 4.4× fewer nonlinear fits per row.

Evidence: `experiments/lm-tolerance.md` and `experiments/fitting-defaults.md`
(the synthetic sweeps), and the rough-beta Pine Island Glacier sweep in the
research repo's replay directory — 9 variants × {smoothed, pointwise} ×
{k=20, k=50} × {ball, ellipsoid window}, held-out QC between −2.7% and +2.3%
with the worst row 12× better at k=20.

Three things worth carrying forward:

- **The circle rungs are load-bearing and must stay on by default.** They are
  the only family besides the caller's own guess. Dropping them on a prior that
  was 3–5× too wide doubled held-out error (0.20 → 0.50) and took failed rows
  from 190 to 816; on the frog the same comparison is 1.14×. `num_rungs = 0` is
  how a caller opts out, and the default must not move without redoing that
  measurement.
- **The CV score cannot see that failure** — median CV moved 1.0% while
  held-out error moved 106%. Any future auto-tuning driven by the CV score
  would walk straight into it.
- **Every PIG prior row is ≤ 1.67:1 by construction** (`build_sigma` caps the
  anisotropy at `r = 1.67`), so the field-scale validation cannot calibrate an
  aspect-ratio threshold. Nothing in the repo can.

## The anisotropy blind spot: parked with a trigger

The DEFAULT dictionary covers scale but not **orientation** — every entry is
the caller's prior or round. `oriented_ladder` now builds the missing family
(2026-07-29), so the question is answerable without a library change, but it is
not on by default. `dev/robust-init-notes.md` has the parked recipe and a
2026-07-28 update.

Re-measured by `experiments/anisotropy_hardening.py` at 8:1: the blind spot is
real under **free** mu but does not appear under the pinned default, where the
scale-bracketing ladder reaches the best start of any family. Circular starts
do not bias the fitted answer toward roundness (7.7–12.5:1 on a true 8:1).

**Parked deliberately** — deciding whether it should be default is a small
research project, not a patch. Pick it up if mu release is re-armed as a
default, if a problem with genuinely anisotropic priors appears, or if N = 3
arrives, where orientation is SO(3) and the covering dictionary grows.

## Initial guesses became data, 2026-07-29

`fit_from_probes` takes `guesses` — `InitialGuess{sigma, mu?, label}` — instead
of choosing among dictionaries selected by flags. `sigma0`,
`window_shape_rungs` and `circle_rungs_above_aspect` are gone;
`num_rungs = 0` means "only my guesses"; `mu0` is `default_mu`; and
`MuPolicy::Pinned` honours a guess's own `mu`. Plan and rationale in
[`archive/initial-guess-api-plan.md`](archive/initial-guess-api-plan.md),
migration table in the
CHANGELOG.

`fit_operator` is bit-identical across the change — it passes the caller's
`sigma` as the first guess. Two traps the plan flagged were real and are fixed:
the release path re-encodes against the candidate's own center (the old
zero-displacement invariant breaks once centers differ), and
`CandidateFit::theta_hat_init` became `theta_init` in the public absolute
encoding, because a displacement against an unstated origin cannot be decoded.

## How much faster, and at what cost

Measured three ways, all agreeing, after the defaults change and the API change
(the latter is bit-identical, so it contributes nothing):

| | speedup |
|---|---|
| Rough-beta PIG, 6557 rows, five conditions | **4.3–4.8×** |
| Frog row layer, 0.1.0's dictionary reconstructed exactly | **4.8×** |
| Frog operator layer, rungs + tolerance only | 2.4× |

The last is lower because `fit_operator` can no longer assemble the
window-shape family, so it omits half of what 0.1.0 did. The row-layer figure
is the like-for-like: the new API can rebuild 0.1.0's dictionary exactly as
`guesses = [sigma] + window_shape_ladder(x, m2, mu0, 6)` with `num_rungs = 6`.

**69.0 → 24.0 nonlinear fits per row.** Wall clock falls further than the fit
count, so roughly a third of the gain is the looser `ftol` making each surviving
fit cheaper.

Held-out QC across the five field-scale conditions: 0.974×, 0.982×, 0.983×,
0.997×, 1.003× — three of them better than 0.1.0.

## The cleanup: done

Six slices, agreed 2026-07-27, five complete.

1. **Prototype archived.** `archive/python-prototype/`, out of the sdist, out
   of `pytest` collection, out of `check_dependencies`' shipping list. Its
   harmonic-table generator was rescued into `tools/` first, since it emits a
   live header; the regenerated header is byte-identical.
2. **`dev/` purged and redistributed.** 651 MB to 100 KB. The PIG replay
   harness moved to the research repo, where its imports already lived;
   library benchmarks moved to `experiments/`; `dev/` is now TRACKED, and in
   the dependency check.
3. **`docs/` split by audience.** User-facing pages stay; the plans and the
   design log moved here, with `architecture.md` salvaged out of the port plan.
4. **User docs written.** README (was two lines, and is the PyPI description),
   installation, quickstart, api-guide, defaults.
5. **Doxygen sweep**, three phases. 192 @param / 61 @return / 25 @throws, from
   20/0/0. No reference to the prototype, to `dev/`, or to a slice number
   remains in `include/`, `bindings/` or `tests/`.

Plus an agreed detour: the example set, now 16 examples covering every exported
name, and the mesh-scalability study in `experiments/`.

### 6. Release infra -- the only slice left

Doxyfile, CONTRIBUTING, CHANGELOG, CITATION, and the CI workflow. The headers
are now in a state where Doxygen will produce something worth publishing, which
was not true before slice 5.

## Slice 6 in detail: what release still owes

1. **CI.** Nothing runs automatically. Should wire the C++ suite (g++/clang,
   sanitizers at `-j2`), the binding pytest, the frog example as the
   integration gate, `tools/check_dependencies.py`, and version consistency
   between `lgpsf.hpp`, `pyproject.toml` and CITATION.

   **What CI builds vs runs — decided 2026-07-27, do not re-litigate:**

   | | build | run |
   |---|---|---|
   | tests | yes | yes |
   | bindings | yes | yes |
   | Python examples | — | fast ones + `--quick` |
   | C++ examples | yes | `hello_world` only |
   | `experiments/` | **yes** (`-DLGPSF_BUILD_EXPERIMENTS=ON`) | **no** |

   Experiments are built but never run. Two reasons, and the second is the
   one that decides it. They take tens of minutes. And more importantly,
   `docs/reproducibility.md` establishes that results legitimately differ
   across builds with different compiler flags — so any tight numeric
   assertion would be flaky *by construction*, and loosening it until stable
   would leave it too weak to catch anything.

   But they must still COMPILE in CI, because the failure mode an experiment
   actually has is bit-rot, not wrong numbers. Two of the `dev/` scaffolds
   deleted in the cleanup had been broken for months — they subprocessed into
   a directory that no longer existed — and nobody noticed because nothing
   ever built them.

   Pipeline coverage is already served by `examples/operator_fit_frog.py
   --quick` and `bindings/tests/test_frog_integration.py`: the whole stack,
   on a public problem, in about ten seconds. The slow C++ examples are in the
   build-only bucket for the same reason — their numerics are covered by the
   Python integration test on the identical problem.
2. **cibuildwheel, actually exercised.** `pyproject.toml` is written but no
   wheel has been built through it. The `sdist.exclude` added in the cleanup
   is likewise unverified — `scikit_build_core` is not installed here and pip
   cannot reach the network from the sandbox. Expect the first real run to
   find something.
3. **The two portable archived examples.** `plot_lg_modes.py` and
   `lg_expansion_convergence.py` use only primitives that Tier B binds, so
   they can be rebuilt against `import lgpsf` and returned to `examples/`.
   The two `varpro_frog_*` sweeps are superseded by `operator_fit_frog.py`
   and should stay archived.

## Standing lessons

- **Python timings do not predict C++ relative performance.** ~80% of the
  prototype's LG-evaluation time was numpy per-call dispatch, measured by
  scaling the point count and reading the intercept. Re-measure on the C++
  side; never port a performance conclusion.
- **Use bit-identity as the acceptance criterion** for any change that is
  supposed to be a pure restructuring. It caught a 1-ULP regression that
  `allclose` would have passed. Use a tolerance only where the math genuinely
  reassociates — and say so.
- **A profile taken at the wrong size answers the wrong question.** The
  "basis-bound, not SVD-bound" conclusion held only at six modes; at
  production mode counts the fit is SVD-bound, which is what the QR change
  came from. See [`../experiments/inner-solve-profile.md`](../experiments/inner-solve-profile.md).

## Building on this machine

- `cmake` is not on `PATH` — it lives in a hand-extracted tarball under the
  home directory.
- A fresh configure needs `-DFETCHCONTENT_SOURCE_DIR_ELLIPSOID_TREE=<path to
  the local checkout>`, or CMake silently downloads the pinned v0.2.0 tarball
  instead of using it.
- **NEVER build with a bare `-j`.** On THIS machine (13 GB, 8 cores) use `-j2`
  normally and `-j1` for sanitizers. `-j3` is permitted by the hook but is
  measurably too much: the heaviest translation unit peaks at 2.8 GB, so three
  at once plus a desktop session pushes the machine into swap and locks it up
  for the duration. That happened on 2026-07-28. The hook's cap predates the
  measurement; treat it as a ceiling, not a target.
- Timing work wants `-march=native`; the test suite is not built with it.
  That difference is not cosmetic — see
  [`../docs/reproducibility.md`](../docs/reproducibility.md).
- matplotlib is not in the primary conda env and cannot be installed from the
  sandbox. `examples/operator_fit_frog.py` degrades gracefully without it.

## Closed: less conservative fitting defaults

Landed 2026-07-28; see the section near the top of this file for what shipped,
and [`archive/fitting-defaults-plan.md`](archive/fitting-defaults-plan.md) for
the plan it came from, annotated with which of its recommendations the real
operator overturned.

The lesson worth keeping: **the frog kernel damages a prior's shape and
orientation but not its scale**, and scale is what the circle rungs actually
rescue. That is why sixty unanimous frog results pointed the wrong way on one
of the four knobs, and why the field-scale check was not a formality.

## Parked

- **Released-mu re-arming.** Needs a basin-scale `||mu - mu0||` bound before
  it can be trusted at operator scale; release currently ships on ~91% of
  rows while buying nothing on held-out score.
- **MarginGreedy.** Benchmarked, parked, deliberately not ported. Needs the
  novelty-floor refinement first. The working implementation is in
  `archive/python-prototype/mode_policy.py` — the one thing there with no C++
  counterpart.
- **Level-2 cross-row amortization.** Neighbour warm starts and
  smoothed-theta seeds as candidate injection.
- **`LGOperator` serialization** (M6).
- **`fit_operator`'s `window_ellipsoids` override is unbound** in Python: it
  needs an `ellipsoid_tree::Ellipsoid` at the boundary, and that library's
  Python convention is the transpose of ours. Wants a decision, not a cast.
