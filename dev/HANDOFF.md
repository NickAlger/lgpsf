# Developer handoff — current state and open threads

Continuity between working sessions: what is in flight, what is owed, and
what is parked. For the stable project overview — the math, the architecture,
the conventions — read [`CLAUDE.md`](../CLAUDE.md) instead. This file is
narrative and open items only; when a thread closes, its record moves to
[`archive/`](archive/).

## Where things stand

| | state |
|---|---|
| C++ core | complete — 145 cases / 105,699 assertions |
| QR-first inner solve | landed — 2.1× on a whole-field fit, every number unchanged to four digits |
| Field-scale validation | smooth and rough basal friction both reproduced; the recorded 0.0147 matched exactly |
| Python bindings | complete — 49 pytest cases |
| Examples | 15, covering every exported name; the frog example is the public gate |
| User documentation | written — README, installation, quickstart, api-guide, defaults |
| Direction of dependence | enforced by `tools/check_dependencies.py` over docs, dev and experiments too |
| Repo cleanup | 5 of 6 slices done — see below |
| **Not done** | **CI, cibuildwheel, M6 release infra** |

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

Plus an agreed detour: the example set, now 15 examples covering every exported
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
