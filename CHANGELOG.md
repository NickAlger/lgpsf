# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Less conservative fitting defaults.** The initial-guess dictionary is now
  `sigma0` + 3 circles (plus a warm start), where it was `sigma0` + 6 circles +
  6 window-shaped starts. Roughly 4.4× fewer nonlinear fits per row, measured
  at −2.7% to +2.3% on held-out score across eight conditions of a field-scale
  PDE Hessian, and *better* in the tail (worst row 12× better at k = 20).
  - `ProbeFitConfig::num_rungs` `6` → `3`.
  - `ProbeFitConfig::window_shape_rungs` `true` → `false`. The window derives
    from the same `sigma` as the initial guess, so this family carried no shape
    information `sigma0` did not already have; when the window is a ball it
    degenerated into a second copy of the circle rungs.
  - `VarProOptions::ftol`, `xtol`, `gtol` `1e-8` → `1e-4`. `ftol` is the only
    one that binds. `1e-4` is the loosest setting at which the mode ladder
    still made every decision it makes at `1e-8`.
  - `ProbeFitConfig::circle_rungs_above_aspect` stays at `1.0` — circles
    always on. Raising it is now load-bearing rather than an economy: with the
    window rungs off it leaves `sigma0` as the only start, which doubled
    held-out error on a prior that was wrong in scale.

  Fits are **not** bit-identical to 0.1.0. Selection can change where two
  candidates tie, and looser tolerances shift which local minimum a row lands
  in. Set the four knobs back to their 0.1.0 values to reproduce old results.

### Added

- `experiments/lm_tolerance.py`, `experiments/anisotropy_hardening.py` and
  their write-ups, plus a `circle_rungs_above_aspect` gate on the circle rungs
  (default `1.0`, i.e. no behaviour change from that knob alone).

## [0.1.0] — 2026-07-28

Initial release. Alpha: the method is validated at field scale, the API is not
yet stable.

### Added

- **Operator layer.** `fit_operator` fits every row of an operator from random
  probes and their responses, threaded over rows and deterministic for any
  thread count. Returns an `LGOperator` (the model) and `FitDiagnostics`
  (per-row provenance) as separate objects.
- **`LGOperator` and its helpers**, in a header that depends on none of the
  fitting stack: `matvec`, `assemble_sparse`, `eval_entries`, `eval_kernel`,
  `qc_map`, `spike_measure`, `ellipsoid_field`, `validate`, `model_rows`,
  `row_expansion`. An operator can be built directly from a physics model with
  `build_operator` and merged across ranks with `concatenate_rows`, without the
  fitter being involved.
- **Row layer.** `fit_from_probes` fits one target known only through inner
  products with random fields, returning the winning model and the whole
  candidate search. `linear_cv_score` scores any model — fitted or a priori —
  at zero nonlinear fits.
- **Fitting core.** `fit_varpro`, a generic variable-projection solver over any
  separable basis, with Kaufman and Golub-Pereyra Jacobians and a QR-first
  inner solve. A hand-rolled trust-region Levenberg-Marquardt reproducing
  MINPACK's semantics, solving its subproblem by an SVD hook step.
- **Mode policies.** `FixedSet`, `ShellLadder`, `ExplicitLadder`,
  `WedgeLadder`, `RadialFirstLadder`. The policy is required rather than
  defaulted: the best growth order is budget- and problem-dependent.
- **Primitives.** The N-dimensional Laguerre-Gaussian basis over an exact-
  rational harmonic-polynomial table, the ellipsoid pullback and its
  derivatives, and the whitening transform that keeps mass matrices out of the
  fitting core.
- **Python bindings** covering the whole API, with a zero-copy `(N, K)` point
  boundary.
- **Fifteen examples**, each teaching one thing, and the frog-kernel example as
  a self-contained public integration test.
- **Documentation**: installation, quickstart, API guide, defaults with their
  evidence, validation, and reproducibility.

### Validated

- Reproduced against a real PDE-derived Gauss-Newton Hessian at 6557 degrees of
  freedom, at two basal-friction states, beating the prior reference method at
  every probe budget. See [`docs/validation.md`](docs/validation.md).
- The probe budget does not grow under mesh refinement: a 16× increase in
  degrees of freedom moved the budget for 5% accuracy from 63.9 probes to 64.3.
  See [`experiments/mesh-scalability.md`](experiments/mesh-scalability.md).

### Known limitations

- The API is not stable and no compatibility is promised before 1.0.
- Released-mu fitting is available but not the default; it needs a basin-scale
  bound on how far a center may move before it can be trusted at operator
  scale.
- Results are bit-identical within a build but not across builds with different
  compiler flags, where a small fraction of rows may reach a different local
  minimum. See [`docs/reproducibility.md`](docs/reproducibility.md).

[Unreleased]: https://github.com/NickAlger/lgpsf/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NickAlger/lgpsf/releases/tag/v0.1.0
