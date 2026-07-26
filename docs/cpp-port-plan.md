# C++17/Eigen port plan -- agreed design

**Status: DESIGN AGREED (Nick + session 2026-07-25). Implementation in
a NEW session against this plan. The Python prototype is the reference
implementation; on port start it is FROZEN (CLAUDE.md gets a "FROZEN as
of <commit>" note, its tests stay green in CI, all new development
lands C++-first).**

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
  one `bindings/lgpsf_bindings.cpp`, points-as-rows `(K, N)` at the
  numpy boundary with ONE transpose (matches the design-notes layout
  entry), `gil_scoped_release` + `num_threads` on batched calls,
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
| harmonic_polynomials.py | harmonic_polynomials.hpp | the ONLY consumer of the table's format, in both languages; eval/grad/terms/`num_harmonics`/`max_degree` |
| lg_functions.py | lg_functions.hpp | pure recurrences; `std::tgamma` for half-integer gammas; `struct Mode{int p, ell, m;}` |
| ellipsoid_transform.py | ellipsoid_transform.hpp | `std::optional<VectorXd>` replaces the mu0=None fit-mu/fixed-mu sentinel; N<=4 fixed-size `.inverse()` per design-notes |
| lg_ellipsoid_feature.py | lg_ellipsoid_feature.hpp | eval/jvp/vjp/jac; jac is the fitting-core interface |
| whitening.py | whitening.hpp | the ONLY masses layer (invariant preserved); `whitened_basis` closure-builder becomes a small basis object with eval/vjp/jac methods (or `std::function` triple) |
| varpro.py | varpro.hpp | BDCSVD inner solve, FWL, Kaufman one-sweep; **the hand-rolled LM lives here** (contract below); options/results as structs |
| init_dictionary.py | init_dictionary.hpp | cKDTree k-NN -> ellipsoid_tree KDTree; labelled rungs as `std::pair<std::string, VectorXd>` |
| probe_moments.py | probe_moments.hpp | trivial |
| mode_policy.py | mode_policy.hpp | virtual `ModePolicy` base (stateless, per design); fixed policies only -- **MarginGreedy stays Python-side while parked** |
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

## The numerics map (what replaces scipy)

- `np.linalg.svd` -> `Eigen::BDCSVD` (inner solve `(k, m)`, range
  basis `(k, n_extra)`; rank cutoff `max(shape)*eps*sigma_max` kept).
- `lstsq` -> `ColPivHouseholderQR` (or BDCSVD) solve; `qr` ->
  `HouseholderQR`; `eigh` -> `SelfAdjointEigenSolver`; `cholesky` ->
  `LLT` (throw -> the per-row "failed" path); `inv` on N<=4 ->
  fixed-size inverse; spectral norm of L -> N x N SVD.
- `scipy.spatial.cKDTree` -> `ellipsoid_tree::KDTree`.
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

## Randomness + determinism policy

Deviation from ellipsoid_tree's "no randomness in library headers"
(lgpsf's CV folds and warm-start jitter are load-bearing), resolved
the ellipsoid_psf way:

- `std::mt19937` **raw output only** -- no `std::*_distribution`
  (implementation-defined across stdlibs). Uniforms via `gen() /
  2^32`; normals via Box-Muller on raw output (the ellipsoid_psf
  low_rank.hpp helpers are the model).
- **Per-row seeds: `seed + rho`** at the operator layer, so results
  are bit-deterministic regardless of scheduling/num_threads. (A
  documented improvement over the prototype, which shares one
  generator across rows within a chunk -- cross-language equivalence
  is tolerance-based anyway, see test doctrine.)
- Numpy PCG64 streams (fold permutations, jitter) are deliberately NOT
  reproduced.

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

- **M0** table codegen + harmonic_polynomials.hpp + lg_functions.hpp
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
- **M1** ellipsoid_transform + lg_ellipsoid_feature + whitening.
  Accept: full tier-1 identity suite green in C++.
- **M2** varpro.hpp incl. the hand-rolled LM. Accept: inner-LA tests
  vs brute-force references; synthetic recovery at tolerance; the
  max_nfev=1 semantic; sentinel behavior.
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
