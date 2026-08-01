# Architecture: what is where

A map of the headers for someone about to change one. For what the library
*does*, read [`../docs/`](../docs/); for the mathematics, read
[`../docs/varpro-whitening-notes.pdf`](../docs/varpro-whitening-notes.pdf).

Everything is header-only C++17 under `include/lgpsf/`, depending on Eigen
and `ellipsoid_tree`. The Python bindings in `bindings/` are a thin marshalling
layer over exactly these headers and add no logic of their own.

## Layering

Each level depends only downward. That is a real invariant, not an
aspiration — `test_lg_operator.cpp` asserts that `lg_operator.hpp` includes
none of the fitting stack.

```
lg_functions / harmonic_polynomials        pure basis math
  -> ellipsoid_transform                   the pullback T(theta, x)
  -> lg_ellipsoid_feature                  their composition
  -> whitening                             the ONLY layer touching masses
  -> varpro (+ detail/levenberg_marquardt) the fitting core
  -> init_dictionary, probe_moments, mode_policy
  -> probe_fit                             one target
  -> lg_expansion -> lg_operator           the data structures and their operations
  -> operator_fit                          one producer of them
```

The split at the bottom right matters: **`lg_operator.hpp` is not part of the
fitter.** An operator can be built from a physics-based approximation, merged,
validated, applied and assembled without ever including `operator_fit.hpp`.

## The headers

| header | what it provides |
|---|---|
| `detail/lg_harmonics_table.hpp` | Generated static data: the harmonic polynomials in exact-rational-derived literals, stored as per-row NONZERO terms (91.5% of dense coefficients vanish structurally). Regenerated only by `tools/generate_lg_harmonics_table.py`. |
| `harmonic_polynomials.hpp` | The only consumer of the table's storage format, so a change to how polynomials are represented is scoped to one file. Eval/grad/terms, `num_harmonics`, `max_degree`, plus the batched `eval_harmonic_basis`/`grad_harmonic_basis`. |
| `lg_functions.hpp` | `Mode{int p, ell, m;}`, `genlaguerre` (three-term recurrence — no special-function library exists in C++), `lg_norm`, `modes_up_to_level`, and `LGBasisAt`. The batched `eval_lg_basis`/`grad_lg_basis` are the production path; the one-at-a-time forms are kept as the readable reference. |
| `ellipsoid_transform.hpp` | The pullback and its JVP/VJP, in two composable stages so theta encodings never touch the geometry. `mu0` is required, so `N = mu0.size()`; the fixed/fitted switch is `enum class MuMode`. **Two encodings** — public absolute `theta`, internal `theta_hat`. |
| `lg_ellipsoid_feature.hpp` | `FeatureAt`: the composition phi_i(x; theta) = psi_i(T(theta, x)) with its derivatives. `jac()` is indexed by PARAMETER, which is how Golub-Pereyra slices it. Does not retain `x`. |
| `whitening.hpp` | `WhitenedBasis` -> `WhitenedBasisAt`. **The only place M1/M2 appear anywhere.** A concrete functor rather than `std::function`, so `varpro.hpp` can template on it. |
| `varpro.hpp` | `fit_varpro`, templated on the basis functor — no indirect call per trial point, and Golub-Pereyra availability is a compile-time trait. QR-first inner solve (`detail::InnerFactors`), Frisch-Waugh-Lovell preprocessing, Kaufman one-sweep Jacobian. |
| `detail/levenberg_marquardt.hpp` | A generic trust-region LM reproducing MINPACK's *semantics* (not its trajectory), solving the subproblem by an SVD hook step. Tested in isolation against published answers, never through VarPro. |
| `init_dictionary.hpp` | Initial-ellipsoid hypothesis generation: geometry and policy, no probes and no fitting. Home of `InitialGuess` — the public `(mu, sigma)` starting ellipsoid — and the `*_ladder` helpers that build dictionaries of them, which callers may use directly. First place `ellipsoid_tree`'s layout flip bites — its KDTree wants points as columns. |
| `probe_moments.hpp` | Zero-matvec estimators from probe data. `spike_index < 0` is the "absent" sentinel. |
| `mode_policy.hpp` | Virtual `ModePolicy` with `propose` **const**, so statelessness is enforced by the type rather than by convention. `LevelRecord` carries `bool has_winner` rather than the winning fit — that boolean breaks what would be a circular dependency on the engine's header. |
| `probe_fit.hpp` | One target's fit from probes: the ordered candidate stream, admissibility, K-fold CV scoring, early stopping. The CV split and jitter table arrive as DATA, so this and everything below it are pure functions of their inputs. |
| `lg_expansion.hpp` | `LGExpansion` — one target's model (absolute theta, its modes, `c`, `s`). Self-decoding, evaluable, checkable. Both the per-row content of an `LGOperator` and the model half of a fit result. |
| `lg_operator.hpp` | `LGOperator` and every evaluation/assembly helper. Free functions rather than methods: methods answer "what am I", free functions answer "what can be done with me". |
| `operator_fit.hpp` | `fit_operator`, row-parallel, with the always-on baseline guard and CSR fit windows. Window shape is one continuous `window_aspect_cap`. All windows from ONE dual-tree descent. |
| `exceptions.hpp` | `InfeasibleParameters` means "no basis exists at this point in parameter space" and the fitting core catches it, scoring the worst finite cost. Caller errors stay `std::invalid_argument`. |

## The corrections layer (`corrections/`)

A **sibling** of everything above, not a level of it. `include/lgpsf/corrections/`
is operator-blind machinery for turning a symmetric operator approximation plus
a regularization operator into a deployable SPD preconditioner — pencil
operations, PSD-ification, deflation
([`pencil-corrections-plan.md`](pencil-corrections-plan.md) is the layer's
plan of record). Nothing in it reads a matrix entry: it builds against Eigen
and its own headers only, and no fit header may include it —
`tools/check_dependencies.py` enforces both directions mechanically.
Namespace `lgpsf::corrections`; Python `lgpsf.corrections` submodule; not in
the `lgpsf.hpp` umbrella. **Blocks are columns**, `(N, m)` — the
linear-algebra convention, deliberately not the fit layers' probes-as-rows.

| header | what it provides |
|---|---|
| `corrections/symmetric_op.hpp` | `SymmetricOp`, the operator boundary: dimension + block matvec, type-erased so the durable structs can OWN their handle. `sparse_op` / `dense_op` adapters (by value — a handle cannot dangle), and `symmetry_defect`: a seeded stochastic symmetry measurement from matvecs alone, which catches an unsymmetrized operator handed across the boundary. |
| `corrections/hr_oracle.hpp` | `HrOracle`: apply $H_r$, and solve with it to a requested relative tolerance. The consumer's scalable solver (Krylov + multigrid) wraps in from C++ or Python; `sparse_hr_oracle` (SimplicialLLT, exact) is the reference adapter for tests and small $N$. |
| `corrections/mode_block.hpp` | `ModeBlock`: the ONE low-rank object holding every spectral correction — $E = (H_r V) C (H_r V)^T$ with $V^T H_r V = I$ and a provenance tag per column. `merge` folds a new contribution in by two-pass $H_r$-Gram orthonormalization ($q$ oracle APPLIES, no solves): existing columns are never modified, the in-span part of a candidate lands in coefficients. `pencil_eigenvalues` = `eig(C)` exactly, the analytic ingredient of the PD certificates. Plain data; persist from Python with numpy. |
| `corrections/shifted_operator.hpp` | `ShiftedOperator` — the durable struct (operator handle, oracle, `ProbeArchive`, one block, contract scalars) representing $B + E + aH_r$ for a CALLER-supplied $a$. `make_shifted_operator` is the checked build (symmetry gate + consistency). GLR deployment: `glr_solve` is the diagonal-capacitance Woodbury inverse of $M(a) = aH_r + E$ — one oracle solve + $O(N\rho)$ per column, exact at every $a$ above `glr_pd_floor` (analytic certificate) with zero refactorization; `eig(C)` is recomputed per call on purpose, so nothing can go stale against the caller-visible block. |

Further headers land per the plan's slice list.

## House rules

- `#pragma once`, SPDX, and the "Part of lgpsf" preamble on every header.
- `///` doc comments; `detail/` is excluded from Doxygen.
- Error messages read `"lgpsf::Thing: lowercase message"`, validated eagerly
  with NaN-safe `if (!(x > 0))` forms.
- Points are `(K, N)` — **points as ROWS**, the transpose of the archived
  Python. A C-contiguous numpy `(N, K)` and a column-major Eigen `(K, N)` are
  the same bytes, which is what makes the Python boundary zero-copy.
- Vectorize the point batch only. Loops over mode count, dimension, parameter
  count and recurrence depth are fine and often preferred.
- **Every JVP/VJP pair gets two tests**: finite differences *and* adjoint
  consistency. The second has caught sign and transpose bugs the first missed.
- Version is single-sourced in `lgpsf.hpp` and read by CMake.
- Tests are one file per header, compiled into a single doctest executable.

## Building here

Read the memory rules in [`HANDOFF.md`](HANDOFF.md) before your first build —
**never a bare `-j`**. This machine has been OOM-crashed by large parallel
builds, every translation unit pulls in Eigen, and a hook enforces the cap.
