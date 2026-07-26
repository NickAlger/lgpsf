# C++17/Eigen port plan -- agreed design

**Status: IN PROGRESS. The prototype is FROZEN as of `e5c36c9`
(2026-07-25) -- see CLAUDE.md for what that permits. Implementation
proceeds against this plan, M0 first.**

Before implementation began, four prototype changes settled design
questions this plan had left open or got wrong; each is recorded in
`docs/design-notes.md` with its measurements, and the relevant sections
below have been updated:

| | settled |
|---|---|
| `aaddae6` | sparse harmonic table + `harmonic_polynomials` module |
| `9e4af69` | evaluate a mode SET, not one mode at a time |
| `bc006a5` | the VarPro VJP regroups by shell |
| `e5c36c9` | the fitting core takes ONE per-theta basis object |

**Two standing warnings from that work.** (1) *Python timings do not
predict C++ relative performance here*: ~80% of the prototype's LG
evaluation time is numpy per-call dispatch overhead, which does not
exist in C++, so a change worth 1.02x-1.38x in Python may be worth much
more or much less there. Re-measure on the C++ side; do not port a
performance conclusion. (2) *Claims get measured before they get
believed*: two plausible arguments (the shell regrouping being more
accurate; the harmonic table's zeros being a tolerance question) were
falsified by measurement, and one 1-ULP regression was caught only
because bit-identity was the acceptance criterion rather than
`allclose`.

Grounded in two full-repo surveys (lgpsf itself + the ellipsoid_tree
engineering pattern, session 2026-07-25); the pattern to mirror is
ellipsoid_tree's, substituting `lgpsf` / `lgpsf::` / `LGPSF_` for the
name / namespace / macro prefix, PyPI distribution name `lgpsf`
(availability to be checked; module import name `lgpsf`).

## !! COMPILE MEMORY SAFETY -- READ BEFORE ANY BUILD !!

**This machine has been OOM-crashed more than once building
ellipsoid_tree/ellipsoid_psf with large `-j`.** It has ~13 GB RAM and
8 hardware threads; every lgpsf TU includes Eigen through the umbrella
header (~180 MB-1 GB+ peak per compile job; sanitizer builds heavier).
Hard rules for every session working on the C++ side:

- **NEVER run a bare `make -j` / `cmake --build . -j` / `-j$(nproc)`.**
- Default to **`-j3`** for normal builds; **`-j2` for sanitizer
  builds** (matches the ellipsoid_tree CI cap); drop to `-j2`/serial
  if anything else heavy is running (Python fits, browsers).
- Rule of thumb (from ellipsoid_tree's README): keep **>= 1-2 GB RAM
  per compile job**.
- Prefer building specific targets (`cmake --build build --target
  lgpsf_tests -j3`) over full rebuilds.
- Set up a **precompiled header early** (M0-M1, not "later"): the
  ellipsoid_tree measurements are ~1.5 s / ~180 MB per Eigen TU
  without PCH vs ~0.2 s / ~125 MB with. The test suite is one
  executable built from many TUs -- PCH is where most of the compile
  time and memory goes away. (ellipsoid_tree itself never shipped the
  PCH; lgpsf, with more TUs including BDCSVD/optimization code, should.)
- Keep test TUs lean: include the specific header under test, not the
  umbrella, wherever possible.

## Decisions (Nick, 2026-07-25)

- **(a) Dependencies: Eigen + ellipsoid_tree only.** Already wired:
  the existing CMake scaffold finds ellipsoid_tree (installed or
  pinned FetchContent v0.2.0) which brings Eigen + Threads
  transitively. No scipy equivalent is needed anywhere (see the
  numerics map below).
- **(b) Multithreading: yes** -- `ellipsoid_tree::detail::parallel_for`
  over operator rows only; row fits stay single-threaded. Uniform
  trailing `int num_threads = 0` convention. Results must be
  bit-deterministic across thread counts (disjoint output slots +
  per-row RNG seeds; see the randomness policy).
- **(c) Header-only: yes.** `include/lgpsf/*.hpp`, INTERFACE target
  (already scaffolded), `detail/` for internals, umbrella header
  carries the version macros (single source of truth; CMake regexes
  it, CI cross-checks pyproject/CITATION -- the ellipsoid_tree
  discipline).
- **(d) Python bindings: yes** -- pybind11 (>=2.12 found-or-fetched),
  one `bindings/lgpsf_bindings.cpp`, **`(N, K)` at the numpy boundary
  with ZERO copies** (corrected 2026-07-25: the Eigen `(K, N)` and numpy
  `(N, K)` forms are the *same bytes*, so a C-contiguous numpy array
  `Map`s straight onto the Eigen matrix -- take
  `py::array_t<double, py::array::c_style>` and build the `Eigen::Map`
  by hand rather than relying on the automatic Eigen caster, which
  matches shapes rather than layouts. Exposing `(N, K)` also matches the
  frozen prototype, making the PIG replay a drop-in),
  `gil_scoped_release` + `num_threads` on batched calls,
  scikit-build-core + cibuildwheel + Trusted Publishing, exactly per
  the ellipsoid_tree pyproject/workflow pattern.
- **(e) Prototype: frozen historical reference** (see Status header).
  It remains the generator of golden fixtures and the tolerance
  reference; its tests keep running in CI.

## File -> header map (1:1, dependency order)

| prototype file | header | port notes |
|---|---|---|
| generate_lg_harmonics_table.py | (offline codegen, NOT ported) | gains a mode emitting `lg_harmonics_table.hpp` from the same exact-rational source; `%.17g` literals, bit-identical doubles; never hand-translated |
| lg_harmonics_table.py | lg_harmonics_table.hpp | generated static data, SPARSE per-row term lists (see below); ~278 KB Python |
| harmonic_polynomials.py | harmonic_polynomials.hpp | the ONLY consumer of the table's format, in both languages; eval/grad/terms/`num_harmonics`/`max_degree`, plus the BATCHED `eval_harmonic_basis`/`grad_harmonic_basis` |
| lg_functions.py | lg_functions.hpp | pure recurrences; `std::tgamma` for half-integer gammas; `struct Mode{int p, ell, m;}`; the batched `eval_lg_basis`/`grad_lg_basis` are the production path (below) |
| ellipsoid_transform.py | ellipsoid_transform.hpp | **DONE (M1).** No `N` parameter and no optional `mu0`: `mu0` is required (the fitting layers already require it), so `N = mu0.size()`, and the sentinel becomes `enum class MuMode {Pinned, Fitted}`. Public `theta` vs internal `theta_hat` -- see design-notes |
| lg_ellipsoid_feature.py | lg_ellipsoid_feature.hpp | **DONE (M1).** `FeatureAt` + free wrappers; `jac()` is indexed by PARAMETER (num_params of (K, num_modes)), which is how Golub-Pereyra slices it. Does not retain `x` -- the pullback's derivatives need only `u` |
| whitening.py | whitening.hpp | **DONE (M1).** The ONLY masses layer (invariant preserved). `whitened_basis` became `WhitenedBasis`, a functor with `operator()(theta_hat) -> WhitenedBasisAt` -- a concrete type, not `std::function`, so `varpro.hpp` templates on it |
| varpro.py | varpro.hpp | **DONE (M2).** BDCSVD inner solve, FWL, Kaufman one-sweep, structs for options/results, `enum class JacobianVariant`. The basis is a TEMPLATE parameter, so there is no indirect call per trial point and Golub-Pereyra availability is a compile-time trait rather than `hasattr`. **The LM lives in `detail/levenberg_marquardt.hpp`**, not here -- it is the one piece with no reference to port, so it is generic and tested in isolation |
| init_dictionary.py | init_dictionary.hpp | cKDTree k-NN -> ellipsoid_tree KDTree; labelled rungs as `std::pair<std::string, VectorXd>` |
| probe_moments.py | probe_moments.hpp | trivial |
| mode_policy.py | mode_policy.hpp | **DONE (M3).** Virtual `ModePolicy` base with `propose` **const**, so statelessness is enforced by the type rather than by convention; `SequencePolicy` + FixedSet/ShellLadder/ExplicitLadder/WedgeLadder/RadialFirstLadder. **MarginGreedy stays Python-side while parked.** `LevelRecord` carries a `bool has_winner` rather than the winning fit -- policies only ever test it for existence, and the boolean breaks what would be a circular dependency on the engine's header |
| probe_fit.py | probe_fit.hpp | configs/results as structs; status/stop_reason as enums from day one; CV folds + jitter per the randomness policy |
| operator_fit.py | operator_fit.hpp | native ellipsoid_tree calls (BallTree/Ball/EllipsoidTree/collision_pairs); parallel_for over rows; OperatorFit flat arrays port as-is (windows already CSR); `to_linear_operator` is NOT ported (Python-side binding convenience) |

Not ported: `examples/*.py` plotting/research scripts, MarginGreedy
(until the novelty-floor iteration un-parks it), the parked docs ideas
(robust-init portfolio, scheme-C, BLR converter).

### The harmonic table's layout (settled 2026-07-25, prototype-first)

91.5% of the dense coefficients are structurally, exactly zero, so the
table stores each polynomial as its nonzero terms only -- see
docs/design-notes.md for the parity-class / Gram-Schmidt-staircase
argument and the evidence that this is exact rather than a tolerance.
C++ layout: **nonzero-major, two parallel arrays per shell** --
`double coeffs[nnz]` and `int8 exponents[nnz * N]`, walked in lockstep,
one linear pass per polynomial with no indirection. Deliberately NOT
interleaved into a `{double; int8[N];}` struct: that gives a 12-byte
stride at N=4 and misaligns every double after the first, for no
locality gain over the parallel form. Also cheaper than a shared
exponent block plus an index gather, which is only ~11% smaller
(81 KB vs 90 KB, measured) and adds a random access per term.

Sizes, table-wide (7,623 nonzero terms): **723 KB dense -> 90 KB
sparse, 8.0x**. Note this ratio tracks the 91.5%-of-coefficients-are-
zero figure, while the Python source file only shrank 53% -- decimal
literals are variable-width (a dropped `0.0` is 5 characters, a
surviving coefficient 20) and the per-row form duplicates exponent
tuples that the dense form shared. Neither effect exists in binary:
every double is 8 bytes regardless of value. Do not use the Python
file size to reason about the C++ footprint.

Emitted as flat `inline constexpr` blobs plus per-shell offset tables,
not nested aggregate initializers: a single large initializer of
scalars is what g++ compiles cheaply, which matters under the
compile-memory rules above. Shell lookup is `(N-1) * (MAX_ELL+1) + ell`,
no map.

`harmonic_polynomials.hpp` owns that layout; `lg_functions.hpp` never
sees it. Port `grad_harmonic` returning `(Y, dY)` from a single pass --
the LG product rule needs both, and that sharing is what removes the
duplication the Python refactor removed.

### Evaluate a mode SET, not a mode (settled 2026-07-25, prototype-first)

The mode index factorizes -- angular independent of `p`, radial
independent of `m`, Gaussian independent of both -- so `eval_lg_basis`/
`grad_lg_basis` (and `eval_harmonic_basis`/`grad_harmonic_basis` under
them) are the production entry points and must be ported as such;
one-at-a-time `eval_lg_nd`/`grad_eval_lg_nd` stay as the readable
reference and the tier-2 exactness oracle. Port notes:

- Share `r2` and the Gaussian once per call, each harmonic once per
  distinct `(ell, m)`, each radial profile once per distinct `(p, ell)`.
- ONE Laguerre table per angular order, run to the deepest `p` that
  order needs; `alpha(ell) + 1 == alpha(ell+1)`, so the `ell+1` table
  supplies the derivatives too -- no separate derivative recurrences.
- Per-dimension integer power tables, computed by REPEATED
  MULTIPLICATION in C++ (faster and no less accurate than `std::pow` for
  small exponents; this is the deliberate break from Python
  bit-identity). Skip `a == 0` factors rather than materializing ones.
- Return gradients per-mode and UNCONTRACTED from `grad_lg_basis`: the
  Golub-Pereyra variant needs the `(n_modes, P, K)` tensor, and
  `jvp_feature`/`jac_feature` contract differently.
- ALSO port `vjp_lg_basis(modes, u, w)`, which computes
  `sum_i w_i grad psi_i` by regrouping per SHELL (see design-notes): one
  `(N, K)` multiply-add per shell instead of several per mode, and the
  `(n_modes, N, K)` tensor never allocated. Worth 1.55-2.35x on the VJP
  in isolation -- only 1.02x end-to-end in Python because that layer is
  dispatch-bound, but C++ is arithmetic-bound so this is one of the
  places the port should beat the prototype by more than the constant
  factor. Accuracy is a wash, NOT an improvement; do not re-derive that.
- No prepared-point-batch object: batched free functions are stateless,
  so points stay plain matrices in every signature.

Measured in Python: 1.38x on an end-to-end row fit, LG share of the fit
47% -> 35%. C++ should do relatively better on the arithmetic (the
Gaussian is 8.1x redundant and `exp` is the dominant per-point
primitive) and worse on the call-overhead component, which does not
exist there.

### The fitting core's basis contract (settled 2026-07-25, prototype-first)

`fit_varpro` takes ONE callable, `basis(theta) -> evaluation object` with
`values()`, `vjp(w_hat)` and optionally `jac()`, not the old
eval/vjp/jac triple. Port it that way: `varpro.hpp` should take a basis
functor and hold its per-theta evaluation alongside the cached inner
solve. Three thin layers mirror the headers -- `LGBasisAt` in
lg_functions.hpp, `FeatureAt` in lg_ellipsoid_feature.hpp,
`WhitenedBasisAt` in whitening.hpp -- with the free functions as
wrappers.

Why not a `value_and_grad`: the Kaufman cotangent `w_hat[i,j] = c_i`
comes from the inner solve, which needs the values, so the order is
values -> solve -> derivative with caller work in between. Nothing to
fuse; what is shared is a partial evaluation at a fixed theta.

The parameter the basis consumes is **`theta_hat`**, not `theta` (M1; see
design-notes). That gives `varpro.hpp` a clean invariant: every input
crossing into the fitting core is hatted -- `z_hat`, `y_hat`, `e_hat`,
`theta_hat` -- and the conversions back to the public, mu0-independent
`theta` are confined to its boundary.

This matters more in C++ than it did in Python (1.04x there, being
dispatch-bound), and the HAND-ROLLED LM is why: its loop evaluates the
residual at trial thetas and residual+Jacobian at accepted ones, so the
object's lifetime falls out naturally instead of being forced by scipy's
separate `fun`/`jac`. Golub-Pereyra support is a CAPABILITY of the
evaluation (`jac()` present), checked once at entry -- not an optional
constructor argument.

## The numerics map (what replaces scipy)

- `np.linalg.svd` -> `Eigen::BDCSVD` (inner solve `(k, m)`, range
  basis `(k, n_extra)`; rank cutoff `max(shape)*eps*sigma_max` kept).
- `lstsq` -> `ColPivHouseholderQR` (or BDCSVD) solve; `qr` ->
  `HouseholderQR`; `eigh` -> `SelfAdjointEigenSolver`; `cholesky` ->
  `LLT` (throw -> the per-row "failed" path); `inv` on N<=4 ->
  fixed-size inverse; spectral norm of L -> N x N SVD.
- `scipy.spatial.cKDTree` -> `ellipsoid_tree::KDTree`. **Note the layout
  flip:** ellipsoid_tree's trees take `(dim, n)`, one point per COLUMN
  (point-major), which is the opposite of lgpsf's coordinate-major
  convention. M4 transposes the mesh coordinates once when building the
  trees -- once per operator fit, not per row.
- `scipy.sparse` -> plain CSR triplet arrays (match OperatorFit's
  flat-array style) or `Eigen::SparseMatrix` -- decide at M4; the
  deployed-support invariant (window-clipped helpers) ports verbatim.
- `scipy.optimize.least_squares(method="lm")` -> **the hand-rolled
  LM**, the one genuinely new piece.

## The LM contract (semantics, not trajectory)

Reproduce scipy/MINPACK *semantics*; do NOT chase bitwise trajectory
parity (fits are certified by CV scores and recovery tolerances, not
iterate paths):

1. Trust-region Levenberg-Marquardt on the reduced problem, analytic
   Jacobian callable, no bounds.
2. `x_scale='jac'` semantics: per-column Jacobian-norm scaling,
   updated as MINPACK does (monotone max).
3. Tolerance semantics: ftol (relative cost reduction), xtol (relative
   step), gtol (gradient-angle/orthogonality), defaults 1e-8;
   max_nfev cap.
4. **Zero accepted steps returns theta_init unchanged** (max_nfev=1
   must behave like MINPACK: evaluate once, return x0) -- the operator
   baseline-guard tie test depends on this exactly.
5. The overflow sentinel contract is upstream of the LM (non-finite
   design -> residual = y_tilde) and ports as-is.
6. Result fields: theta, success, message, n_iterations (accepted
   steps), n_function_evals, final jacobian.

Validation: the prototype's synthetic recovery suite at tolerance
(theta/c/s to 1e-6, cost floors), plus M5's PIG replay.

## Randomness + determinism (current design, 2026-07-25 -- revisitable)

The whole library needs randomness in exactly two places, both found by
survey rather than assumption: the **CV fold assignment**
(`probe_fit.py:168`) and the **warm-start jitter** (`probe_fit.py:501`).
Probe fields themselves are caller-supplied, so generating them is not
this library's job.

**Randomness is hoisted to `operator_fit`.** It builds the split and the
jitter table once, before any row is touched, and passes both DOWN as
data; `fit_from_probes` and everything beneath it are pure functions of
their inputs. The invariant is mechanically checkable: **no `<random>`
include and no generator anywhere below `operator_fit.hpp`.**

Two reasons, the first of which holds even single-threaded:

- The split is genuinely **field-level data**. `V` is `(K_all, k)` and
  every row uses all `k` probes -- a window restricts which POINTS enter,
  never which probes -- so there is one split for the whole fit. And
  candidate CV scores are only comparable on identical folds, which
  becomes a structural fact when there is one split object rather than
  something maintained by re-seeding with a fixed value at every call
  (which is how the prototype gets it).
- It makes M4's "bit-identical across `num_threads`" acceptance
  criterion true by construction. (Note the prototype does NOT share a
  generator across rows within a chunk, as an earlier version of this
  plan claimed -- `default_rng(cfg.seed)` is constructed inside
  `fit_from_probes`, so it is already per-row.)

Concretely:

- `struct CvFold { VectorXi train, validation; }`, and the split is a
  `std::vector<CvFold>`. Train and validation are stored EXPLICITLY
  rather than "validation, train = complement", so the type can express
  splits that are not partitions -- overlapping validation sets,
  subsampled training sets, a single designed holdout.
- **Default: round-robin, `fold(i) = i mod n_folds`, no generator at
  all.** `fit_operator(..., seed = nullopt)` is fully deterministic;
  passing a seed is an explicit opt-in to a permuted split, for checking
  that a result is not an artifact of one particular partition.
- Jitter is a shared `max_levels x P` table (`P` is field-level too),
  `0.05 * U(-1, 1)`, sized generously (64 levels) and reused cyclically
  beyond that. A per-row table is out at field scale (~1 GB at 10^6
  rows), and a shared one is exactly what the prototype already does,
  since every row re-seeds identically.
- Where a generator IS used (a seed was passed): `std::mt19937` **raw
  output only**, uniforms via `(gen() + 0.5) / 2^32`. Never
  `std::*_distribution` -- `mt19937` is specified by the standard but
  the distributions' algorithms are not, so they differ across
  libstdc++/libc++/MSVC. No Box-Muller is needed, the jitter being
  uniform.
- Numpy PCG64 streams are deliberately NOT reproduced.

**What this rests on, and when to revisit it.** Round-robin folds are
justified by the probes being EXCHANGEABLE -- i.i.d. random by
construction, so probe index carries no information and the split is a
nuisance parameter doing no statistical work. Structured probes
(sinusoids, polynomials, structured random ensembles -- all hypothetical
today) would break that: probe index would then correlate with
frequency or degree, and the split would become a modelling choice
rather than a nuisance, with round-robin stratifying across the
structure and contiguous blocks testing extrapolation instead. This is
a design fitted to the current probe model, not a rule.

The hoisting is what makes revisiting cheap: the split is an INPUT, so a
structured-probe experiment changes one call site in `operator_fit` and
nothing in the row fitter.

## Test doctrine (three tiers)

1. **Identity tests port as-is** (the real safety net): all FD checks,
   JVP/VJP adjoint consistency, whitened-regression identity, VarPro
   inner-algebra vs brute-force references, ladder/wedge
   combinatorics, encoding round-trips, overflow sentinel. RNG only
   generates inputs there; doctest + a `test_helpers.hpp` with seeded
   generators (ellipsoid_tree pattern).
2. **Internal-equivalence tests** (legacy-config == policy
   fingerprints; FWL == joint fit): assert path-A == path-B within the
   C++ implementation; never against Python golden values.
3. **Cross-language tolerance tests**: synthetic recovery outcomes vs
   the frozen prototype (fixture problems dumped to files), and the
   acceptance test -- **replay PIG slices 38/39 through the bindings**
   and match the committed numbers at tolerance (smooth beta ~0.015 @
   k=100 clipped+sym; rough beta ~0.057; forensic col-resids
   0.06-0.14). Caveat from the survey: end-to-end Python numbers are
   entangled with MINPACK + PCG64, so "tolerance" means statistically
   indistinguishable quality, not matching digits.

## Milestones (each a reviewable unit, riskiest early)

- **M0 -- DONE.** Shipped: the generator's C++ emission (both outputs in
  one run, so the Python table and the header cannot drift),
  `detail/lg_harmonics_table.hpp` (264 KB source, ~90 KB binary),
  `harmonic_polynomials.hpp`, `lg_functions.hpp`, `tests/pch.hpp`,
  `tests/test_helpers.hpp` (seeded generators + Golub-Welsch
  Gauss-Hermite), and 24 test cases / 97k assertions across
  `test_harmonic_polynomials.cpp` and `test_lg_functions.cpp`.
  Regeneration is byte-stable. Two calibration notes for later
  milestones: the C++ invariants land on the same numbers as the frozen
  prototype computed independently (harmonicity 1.32225e-16 both sides),
  and the LG orthonormality floor is the QUADRATURE, not the basis --
  the Golub-Welsch rule is itself only good to ~3e-13 at n=12 relative
  to the magnitudes it cancels, which is exactly the observed
  orthonormality, so do not read ~1e-13 there as basis error.
- **M0 (original spec)** table codegen + harmonic_polynomials.hpp +
  lg_functions.hpp
  (+ PCH scaffolding). Accept: the prototype's intrinsic suites port
  and pass -- table well-formedness, parity/staircase structure,
  `dim H_ell` closed form, `Delta Y = 0`, homogeneity, sphere
  orthonormality, `genlaguerre` vs closed forms, FD gradients, and the
  keystone **L^2(R^N) orthonormality of the LG modes by tensor-product
  Gauss-Hermite** (Golub-Welsch nodes from `SelfAdjointEigenSolver` on
  the Hermite Jacobi matrix; exact to roundoff, so `int psi_i psi_j =
  delta_ij` is a hard assertion covering table + recurrence +
  normalization + alpha bookkeeping at once -- 3e-15 in Python).
  Generated table byte-stable across reruns.
- **M1 -- DONE.** `ellipsoid_transform.hpp`, `lg_ellipsoid_feature.hpp`,
  `whitening.hpp`, and 25 test cases across their three test files; the
  suite is at 49 cases / 98,496 assertions. Settled here: the
  `theta`/`theta_hat` encoding split (design-notes), `mu0` required with
  `enum class MuMode` in place of the planned optional, and no `N`
  parameter anywhere. Two calibration notes: the whitened chain lands at
  ~1e-15 adjoint consistency and ~2e-10 relative FD, so M2 can hold the
  fitting core to tolerances that tight; and the row-model keystone was
  verified to FAIL when `sqrt(m_rho)` is dropped from `whiten_extra`
  (28 assertions), while the adjoint and FD checks pass right through
  that bug -- so do not weaken it into a derivative check.
- **M1 (original spec)** ellipsoid_transform + lg_ellipsoid_feature +
  whitening. Accept: full tier-1 identity suite green in C++.
- **M2 -- DONE.** `detail/levenberg_marquardt.hpp`, `varpro.hpp`, and
  `exceptions.hpp`, with 25 test cases across two test files; the suite
  is at 74 cases / 98,652 assertions. The LM went first and STANDALONE,
  so a failure in a fit can be told apart from a failure in a Jacobian.
  Settled here: the SVD hook step in place of `lmpar` (design-notes,
  with the MINPACK comparison), and `InfeasibleParameters` as a distinct
  exception so the core can catch "no basis exists at this point"
  without swallowing caller errors. Calibration: synthetic recovery
  converges in 4-5 accepted steps to cost ~1e-30 and |dtheta| ~1e-15,
  so M3 can hold `probe_fit` to tight recovery tolerances. Two claims
  were verified by making the tests FAIL: the Golub-Pereyra vs finite
  differences test does reject Kaufman (all 10 assertions), and the
  callback contract did need a final call at the returned point.
- **M2 (original spec)** varpro.hpp incl. the hand-rolled LM. Accept:
  inner-LA tests vs brute-force references; synthetic recovery at
  tolerance; the max_nfev=1 semantic; sentinel behavior.
- **M3** init_dictionary + probe_moments + mode_policy (fixed
  policies) + probe_fit. Accept: contract tests (counting rule, skip,
  tie-break, certificates), equivalence fingerprints, synthetic
  recovery suite.
- **M4** operator_fit with parallel_for + native ellipsoid_tree.
  Accept: M1!=M2 synthetic operator suite; deployed-support invariant;
  bit-identical results across num_threads in {1, 4}.
- **M5** bindings + pyproject + wheels config. Accept: bindings pytest
  (boundary/marshalling, rows-in/rows-out); **the PIG slice-38/39
  replay through the bindings**.
- **M6** infra: CI (g++/clang matrix + sanitizers at -j2 +
  version-consistency + a prototype-tests job), docs generation +
  Doxygen, README/CHANGELOG/CITATION/CONTRIBUTING per the
  ellipsoid_tree conventions (incl. the compile-memory section).

## Conventions to transcribe from ellipsoid_tree (survey highlights)

`#pragma once` + SPDX + "Part of lgpsf" header preamble; `///`
doc-comments only, detail/ Doxygen-excluded; error messages prefixed
`"lgpsf::Thing: lowercase message"`, eager validation with NaN-safe
`if (!(x > 0))` forms; vendored single-header deps under thirdparty/
with LICENSE; version single-sourced in umbrella macros with CI sync
check; tests one-file-per-header into a single doctest executable;
examples built with OUTPUT_NAME for the docs generator; `.gitignore`
keeps `dev/` and any `original/` out.
