# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere
to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Initial guesses are passed as data, not selected by flags.** Where to seed
  a nonlinear search is problem-specific, so `fit_from_probes` now takes a
  `guesses` collection of `InitialGuess{sigma, mu?, label}` instead of choosing
  among dictionaries the library imagined in advance.
  - **Removed** `fit_from_probes`'s `sigma0` parameter, and
    `ProbeFitConfig::window_shape_rungs` / `circle_rungs_above_aspect`.
  - `mu0` is renamed **`default_mu`** — it centers the default rungs and any
    guess that omits its own `mu`, and still carries the spatial dimension.
  - `MuPolicy::Pinned` now honours a **guess's own `mu`**: pinning means the
    optimizer does not move the center from its initial guess, not that the
    center is `default_mu`.
  - `ProbeFitConfig::num_rungs` gains **`0` = "only my guesses"**; it is an
    error if none were passed.
  - `CandidateFit::theta_hat_init` becomes **`theta_init`, in the public
    absolute encoding** — candidates no longer share one origin, so a
    displacement against an unstated reference could not be decoded.
  - **New**: `InitialGuess`, `circle_ladder`, `window_shape_ladder`,
    `oriented_ladder`, `theta_hat_from_sigma`. `oriented_ladder` makes the
    orientation-covering family reachable for the first time — it is not on by
    default, but the question is now answerable without a library change.
  - **New example**: [`initial_guesses.py`](examples/initial_guesses.py), on a
    deliberately multimodal row (8:1 anisotropy, signed modulation, released
    center) where one good guess is never worse, wins on 3 of 4 rows by up to
    2.1×, and costs under half the fits. `frog_row` gained an `a_mod`
    parameter so the row can be made signed.

  `fit_operator` is **bit-identical** across this change: it passes the
  caller's `sigma` as the first guess, so the dictionary it assembles is
  unchanged. Migration:

  | before | after |
  |---|---|
  | `fit_from_probes(..., sigma0=S)` | `fit_from_probes(..., guesses=[InitialGuess(S)])` |
  | `config.window_shape_rungs = True` | `guesses += window_shape_ladder(...)` |
  | `config.circle_rungs_above_aspect = inf` | `config.num_rungs = 0` |

- **Less conservative fitting defaults.** The initial-guess dictionary is now
  `sigma0` + 3 circles (plus a warm start), where it was `sigma0` + 6 circles +
  6 window-shaped starts. Roughly 4.4× fewer nonlinear fits per row, measured
  at −2.7% to +2.3% on held-out score across eight conditions of a field-scale
  PDE Hessian, and *better* in the tail (worst row 12× better at k = 20).
  - `ProbeFitConfig::num_rungs` `6` → `3`.
  - The window-shape rung family stopped being a default (and is now
    `window_shape_ladder`, opt-in). The window derives from the same `sigma`
    as the initial guess, so it carried no shape information the prior did not
    already have; when the window is a ball it degenerated into a second copy
    of the circle rungs.
  - `VarProOptions::ftol`, `xtol`, `gtol` `1e-8` → `1e-4`. `ftol` is the only
    one that binds. `1e-4` is the loosest setting at which the mode ladder
    still made every decision it makes at `1e-8`.
  - The circle rungs stay on by default, reversing the synthetic sweep's
    provisional recommendation to gate them. Dropping them leaves the prior as
    the only start, which doubled held-out error on a prior wrong in scale.

  Fits are **not** bit-identical to 0.1.0. Selection can change where two
  candidates tie, and looser tolerances shift which local minimum a row lands
  in. Set the four knobs back to their 0.1.0 values to reproduce old results.

### Added

- **`Symmetrize::Weighted`**: a scale-aware symmetrization for
  `assemble_sparse` — a per-entry convex average with inverse-row-energy
  weights, so where two row fits disagree about a shared entry the weaker
  row's value wins instead of the mean. Exactly symmetric by construction,
  and the recommendation over `Average` when the underlying operator is
  symmetric; on rows of comparable scale the two nearly coincide. Validated
  on a field-scale glaciology Hessian, where plain averaging let strong
  rows' tails contaminate weak rows' signal. The rectangular-operator
  refusal message now says "symmetrizing" rather than "averaging", since it
  covers both policies.
- `experiments/lm_tolerance.py` and `experiments/anisotropy_hardening.py`, with
  their write-ups.

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
