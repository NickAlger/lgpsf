# Developer handoff — current state and open threads

Continuity between working sessions: what is in flight, what is owed, and
what is parked. For the stable project overview — the math, the architecture,
the conventions — read [`CLAUDE.md`](../CLAUDE.md) instead. This file is
narrative and open items only; when a thread closes, its record moves to
[`archive/`](archive/).

## Where things stand

| | state |
|---|---|
| C++ core, M0–M4 | complete — 145 cases / 105,699 assertions |
| QR-first inner solve | landed — 2.1× on a whole-field fit, every number unchanged to four digits |
| Field-scale validation | smooth and rough basal friction both reproduced; the recorded 0.0147 matched exactly |
| Python bindings, M5 | all layers bound — 49 pytest cases |
| Public integration gate | `examples/operator_fit_frog.py`, end to end, no private data |
| Direction of dependence | enforced by `tools/check_dependencies.py` |
| Repo cleanup | **in progress** — see below |

## In flight: the cleanup

Six slices, agreed 2026-07-27. Done: (1) the Python prototype archived and
its table generator rescued into `tools/`; (2) `dev/` purged and
redistributed; (3) `docs/` split by audience. Plus an agreed detour: the
example set, now 15 examples covering every exported name, and the mesh
scalability study in `experiments/`. Remaining:

4. **Write the user docs.** A real `README.md` — today it is two lines, and
   `pyproject.toml` names it as the PyPI long description. Then installation,
   a quickstart from the frog example, an API tour of the three layers, and a
   defaults page gathering what is currently scattered across four
   non-adjacent `design-notes` entries.
5. **Doxygen sweep**, in three sub-slices by layer. ~60–70 lines of genuine
   archaeology across the headers, concentrated in five files, plus missing
   parameter/shape/unit documentation on public entities.
6. **Release infra.** Doxyfile, CONTRIBUTING, CHANGELOG, CITATION, CI.

## What M5 still owes

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
- **NEVER build with a bare `-j`.** `-j3` normal, `-j2` for sanitizers. This
  machine has been OOM-crashed by large parallel builds and a PreToolUse hook
  enforces the limit; if a build is blocked, lower `-j` rather than routing
  around it.
- Timing work wants `-march=native`; the test suite is not built with it.
  That difference is not cosmetic — see
  [`../docs/reproducibility.md`](../docs/reproducibility.md).
- matplotlib is not in the primary conda env and cannot be installed from the
  sandbox. `examples/operator_fit_frog.py` degrades gracefully without it.

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
