# Plan: initial guesses as data, not flags

> **IMPLEMENTED 2026-07-29.** Kept as the record of what was agreed and why,
> including the two traps it flagged, both of which were real. For the API as
> shipped read [`../docs/defaults.md`](../docs/defaults.md); for the migration,
> the `Unreleased` entry in the CHANGELOG.
>
> Everything below landed as planned. Two things worth recording that the plan
> did not predict:
>
> - **`circle_ladder` reproduces the old `circle_rungs` bit-for-bit.** The
>   worry was that `sigma = r^2 I` round-tripping through a Cholesky would shift
>   `log r` by an ULP and quietly move every default fit. It does not; a test
>   pins it.
> - **`experiments/fitting_defaults.py` got a measurement it could not express
>   before.** `fit_operator` always passes the caller's sigma as the first
>   guess, so `num_rungs = 0` is exactly "the prior alone" -- the
>   circles-without-prior comparison this note had to make on the frog at the
>   row layer is now available at the operator layer, and it agrees.

**Status: AGREED (2026-07-29).** Design settled in discussion;
this was the note reviewed before any code moved. Checked against `ba263e4`.

## Why

The per-row fit currently decides its own initial guesses from three flags —
`num_rungs`, `window_shape_rungs`, `circle_rungs_above_aspect` — which between
them can express about six dictionaries, all of them ones we thought of in
advance. Where and how to seed a nonlinear search is problem-specific, and the
library has already taken the opposite position once, for the same reason:

> `fit_operator` throws if `config.row.mode_policy` is unset. That is
> deliberate: the best growth order is **budget-dependent and
> problem-dependent**, and no single choice is defensible as a silent default.

Initialization has exactly that character. It is also where the flags have
already gone wrong once: `circle_rungs_above_aspect = 3` looks like an economy
and is in fact an off switch that removes the only safety net, and **nothing in
the cross-validation score reports the resulting failure** (median CV moves
1.0% while held-out error moves 106%). `guesses = []` cannot hide that way.

Three further wins:

- It **subsumes the parked oriented-ellipse work** (`dev/robust-init-notes.md`)
  at zero design risk. That research stops being "change the library" and
  becomes "what should the helper emit", answerable from outside.
- The primitives already exist. `init_dictionary.hpp` has `circle_rungs`,
  `window_rungs`, `oriented_sigma`, `ladder_scales`, `local_spacing`,
  `window_radius`, `theta_hat_from_cholesky` and `InitCandidate{label,
  theta_hat}`. The helpers are largely a `(mu, Sigma)`-faced re-export.
- `sigma0` stops being a special case and becomes one guess among others.

## The evidence for the new default

The default becomes **`num_rungs` circles and nothing else** — `sigma0` is no
longer in the dictionary unless the caller puts it there. That was measured
before agreeing to it, because the always-on baseline guard does **not** cover
it: `operator_fit.hpp:606` scores the prior with `linear_cv_score` at *fixed*
theta, a coefficients-only solve, not an LM fit started from the prior.

Frog, row layer, 16 interior rows x 5 prior qualities (no window confound — at
the row layer the batch is the support):

| prior | sigma0 + 3 circles | 3 circles only | sigma0 only |
|---|---|---|---|
| correct / rotated 90 / 4x narrow / isotropic | 0.1176 | **0.1176** | 0.1176 |
| 4x too wide | 0.1176 | **0.1176** | 0.1462 (**1.24x**, worst row 0.5516) |

Identical to four decimals in mean *and* worst row. The asymmetry is total:
dropping `sigma0` costs nothing, dropping the circles costs 1.24x and a 3.6x
worse tail.

8:1 anisotropy-hardened frog, the hardest basin available — circles-only is
*better* in all four conditions (0.5724 / 0.5465 / 0.0403 / 0.0396 against
0.5727 / 0.5607 / 0.0406 / 0.0399), recovering the same 7.7-12.5:1 shape.

**Why it holds, and when it would stop.** The ladder spans `local_spacing` to
`window_radius`, which on both problems brackets the true scale within ~1.5x,
and LM closes that. It would degrade if that span were very wide *and* the
basin narrow — three rungs over a 10^4 range would land far from the truth.
A caller in that situation should pass their own ladder, which is the point.

Not tested at operator scale on real data, because expressing "no sigma0" needs
this change. That path is unaffected anyway: `fit_operator` will pass the
caller's `sigma` as a guess, so its dictionary is unchanged.

## The API

### Row layer

```cpp
struct InitialGuess
{
    Eigen::MatrixXd sigma;                  ///< (N, N) SPD. Required.
    std::optional<Eigen::VectorXd> mu;      ///< (N,) center. Defaults to default_mu.
    std::string label;                      ///< Provenance. Auto-filled if empty.
};

ProbeFitResult fit_from_probes(
    x, m2_diag, z, y,
    const Eigen::Ref<const Eigen::VectorXd>& default_mu,
    int spike_index = -1,
    const ProbeFitConfig& config = ProbeFitConfig(),
    const std::vector<InitialGuess>& guesses = {},
    std::optional<double> target_mass = std::nullopt );
```

`sigma0` is **removed**. It was used for exactly two things (`probe_fit.hpp:584`
and the aspect gate at `:600`), both of which this replaces. A caller wanting
today's behaviour passes `{{sigma}}`.

Guesses are a **function argument, not a config field**: they are per-target
data, like `z`, `y` and `default_mu`. `config` stays policy.

### Helpers, in `init_dictionary.hpp`

Public, callable from outside, returning `std::vector<InitialGuess>`:

```cpp
circle_ladder(x, default_mu, num_rungs)              // what the default does
window_shape_ladder(x, m2_diag, default_mu, num_rungs)
oriented_ladder(x, default_mu, num_rungs, angles, aspect)   // the parked family
```

`oriented_ladder` is the one that makes `oriented_sigma` reachable for the
first time. Shipping it is not a commitment to it being a good idea; it makes
the question answerable without another library change.

### Config

`ProbeFitConfig` loses `window_shape_rungs` and `circle_rungs_above_aspect`,
keeps `num_rungs` — now meaning *how many default circle rungs*, with **`0`
meaning "only my guesses"**. This replaces the boolean discussed earlier: one
knob, self-documenting, no boolean-and-collection interaction.

Throws if `num_rungs == 0` and `guesses` is empty — zero candidates is a
programming error, not a silent fallback to the baseline. Note the guard moves
up a level: `ladder_scales` currently throws on `num_rungs < 1`.

## mu semantics

**`mu0` is renamed `default_mu`** and stays **required**. An earlier draft
proposed making it optional when no default rung needs centering; that was
wrong. It does four structural jobs beyond centering the default guess:

| | |
|---|---|
| `probe_fit.hpp:519` | **carries the spatial dimension** — `dim = mu0.size()`, the documented convention that there is no separate `N` parameter |
| `:653, :662, :670` | encoding origin for each candidate's public `theta` |
| `:942, :958` | the warm start round-trips through it between ladder levels |
| `:1030` | the mu-release path re-encodes against it |

The rename is the fix for the confusion it was meant to solve: once guesses
carry their own centers, `default_mu` says exactly what its centering role is,
and the docstring covers the other three.

**Pinned honours each guess's own mu.** `MuPolicy::Pinned` means "the optimizer
does not move mu from its initial guess", not "mu is `default_mu`". This needs
no change to `ellipsoid_transform.hpp`: the reference mu is *already* a
parameter of `to_theta` / `to_theta_hat` / `unpack_theta_hat`
(`ellipsoid_transform.hpp:286-304`), and `WhitenedBasis` already takes a center
per construction. Displacement is measured against the candidate's own initial
mu; `default_mu` is that reference only for guesses that omit one.

Externally there is no ambiguity: `LGExpansion::theta` is the public absolute
encoding, so a shipped model is self-describing whatever it was fitted against.

## Two traps

**1. The release path's zero-displacement invariant breaks.**
`probe_fit.hpp:1025-1027` says:

> Re-encoding the absolute parameters against `mu0` in the fitted mode is
> exactly `release_mu` of the pinned ones: a pinned candidate's center **IS**
> `mu0`, so the displacement comes out zero.

Once a pinned candidate can sit at its own mu, that displacement is not zero,
and the release path silently starts from a *displaced* center rather than
releasing in place. It will still converge to something, so this will not
announce itself. Fix: re-encode against the candidate's own center. Test:
release from an off-`default_mu` pinned candidate and assert the released fit
starts at that candidate's optimum.

**2. `CandidateFit::theta_hat_init` becomes undecodable.** It is publicly
readable and documented as "the starting point, in the stream's internal
encoding". Today every candidate shares one origin so a user can decode it with
`mu0`; with per-candidate origins it is a displacement from an unstated
reference. **Fix: store the start as absolute `theta`**, matching
`CandidateFit::model.theta`. The field exists for provenance, and provenance in
the public encoding is strictly more useful — it also removes the last place a
user must know the internal encoding exists.

## Implementation order

Each step should leave the suite green.

1. **`InitialGuess` + helpers** in `init_dictionary.hpp`, with tests. Nothing
   consumes them yet. `circle_ladder` must reproduce today's `circle_rungs`
   output bit-for-bit — that is the acceptance criterion, per the house rule.
2. **Thread per-candidate centers** through the row fit: `WhitenedBasis`,
   `to_theta`, `axes_of`, the warm start, the release path. With every guess
   still centered at `default_mu` this is a pure restructuring, so **bit-identity
   against `ba263e4` is the test**.
3. **Switch `theta_hat_init` to absolute `theta`.** Binding + test update.
4. **Replace the flags with `guesses`**: remove `sigma0`,
   `window_shape_rungs`, `circle_rungs_above_aspect`; `num_rungs` gains the `0`
   case. `fit_operator` passes `{{sigma}}` plus the default rungs, so operator
   behaviour is unchanged — again bit-identical, and the field-scale numbers in
   `experiments/` should not move.
5. **Bindings**, then **examples and docs**.

## Touch points

- `include/lgpsf/`: `init_dictionary.hpp`, `probe_fit.hpp`, `operator_fit.hpp`
- `bindings/lgpsf_bindings.cpp` — `InitialGuess` needs a binding; `(N,)`/`(N,N)`
  at the Python boundary, and note the points convention flip
- `tests/test_init_dictionary.cpp`, `test_probe_fit.cpp`, `test_operator_fit.cpp`
- `bindings/tests/test_lgpsf_py.py`
- Examples that set the disappearing knobs: `reading_a_row_fit.py:63-64`,
  `fit_one_psf.cpp:127-128`; and every example passing `sigma0=`
- `experiments/`: `fitting_defaults.py`, `lm_tolerance.py`,
  `anisotropy_hardening.py` (which builds single-start fits out of exactly the
  flags being removed — it gets *simpler*)
- `docs/defaults.md`, `docs/api-guide.md`, `CHANGELOG.md`, `CLAUDE.md`

## Migration

Breaking, pre-1.0, and worth a CHANGELOG entry that shows the rewrite:

| before | after |
|---|---|
| `fit_from_probes(..., sigma0=S)` | `fit_from_probes(..., guesses=[InitialGuess(S)])` |
| `config.window_shape_rungs = True` | `guesses += window_shape_ladder(...)` |
| `config.circle_rungs_above_aspect = inf` | `config.num_rungs = 0` |

`fit_operator` callers see no change: it keeps passing the caller's `sigma`
plus the default rungs.

## Settled

- **`fit_operator` does not expose guesses.** Agreed 2026-07-29. Per-row
  guesses would need a callback or a per-row vector, and nothing has asked for
  one. It passes `{{sigma}}` plus `num_rungs` circles; revisit when an
  application wants more.
- **`anisotropy_hardening.py` is the smoke test for the design.** It currently
  builds its single-start fits out of exactly the flags being removed
  (`circle_rungs_above_aspect = inf` with `window_shape_rungs = False`); under
  the new API that is `num_rungs = 0` plus one guess. If that does not read
  better afterwards, the design is wrong.

## Open questions

- `ProbeFitConfig` after this holds `mu`, `mode_policy`, `num_rungs`,
  `target_score`, `mode_patience`, `tie_delta`, `cv_folds`, `split`, `jitter`,
  `varpro` — ten fields, coherently *policy* now that the data has moved out.
  Worth keeping as a struct; revisit only if it grows again.
- `jitter` is warm-start perturbation, arguably the same concern as guesses.
  Left alone for now; folding it in is a separate simplification.
