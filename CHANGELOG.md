# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
