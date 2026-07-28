# lgpsf — project guide

This file is auto-loaded into context for any Claude Code session working
in this repo. It's meant to get a fresh session oriented quickly, without
needing to read prior conversation transcripts. Keep it current as the
project evolves; it describes present state, not history.

Where things live: `docs/` is for people USING the library, `dev/` for
people CHANGING it (start at `dev/HANDOFF.md` for open threads,
`dev/architecture.md` for the header map), `experiments/` for measurements
about the library, and `archive/` for superseded code kept only as
provenance.

## What this project is

`lgpsf`: Laguerre-Gaussian point-spread-function (LG-PSF) Hessian
approximation with VarPro ellipsoid fitting. A method for cheaply
approximating a large, dense, PDE-derived operator $H$ (originally a
Gauss-Newton Hessian from a glaciology inverse problem, but the method
itself is general) as a sparse, locally-supported operator: per mesh row, a
smooth Laguerre-Gaussian kernel expansion on a fitted local ellipsoid, plus
a discrete "spike" correction for the part of the point-spread function the
mesh can't resolve. Fit via random probing (matvecs are the expensive
currency) and nonlinear least squares.

**The implementation is the C++.** `lgpsf` is a header-only C++17 library
(`include/lgpsf/`, depending on Eigen + `ellipsoid_tree`) with Python
bindings (`bindings/`). That is the library; `import lgpsf` gives you the
bindings.

> ### THE PYTHON PROTOTYPE IS ARCHIVED — do not treat it as the library
>
> The method was developed in Python first, then ported. That Python is at
> **`archive/python-prototype/`**, frozen at `e5c36c9` (2026-07-25). It is
> not built, not packaged, not in the sdist, and not collected by `pytest`.
> Do not read it to learn how the library works and do not copy code from
> it — everything it does, the C++ does, and in several places differently
> and on purpose (see its README for the divergence table).
>
> **The C++ is the GROUND TRUTH.** Nothing in the C++ or binding test
> suites measures against the archive, and a disagreement between the two
> is a fact about the archive rather than a bug report against the C++.
> Never constrain the C++ to match it. Its own tests still pass (82,
> `pytest` from that directory) and that is the only property it is kept
> for.

**Derived from, but diverges from, prior research.** The original method
was developed in `~/repos/nicks_research_experiments/ellipsoid_psf_pig`
(`lg-split-method-notes.tex`, `varpro-ellipsoid-notes.tex`,
`varpro-rung-plan.md`) -- read those for the original motivation and
experimental history, but **do not treat that plan as this project's
spec**. Concrete divergences: VarPro derivatives here are analytic
(forward- and reverse-mode), not finite-differenced; the LG basis is
generalized to arbitrary spatial dimension $N$ (the research repo was
2D-only); the smooth+spike combination is handled via a general
orthogonal-projection mechanism (noise whitening, below) rather than
hand-zeroing specific matrix entries.

## The mathematical framework

**Discretization structure.** $H = M_1 \Phi M_2$: $\Phi$ is the continuum
kernel, $M_1$ (row/target) and $M_2$ (column/source) are diagonal lumped
mass matrices. Functional-analytically, $M_2: X \to X'$ and $M_1: Y \to
Y'$ are the Riesz maps of the discretized $L^2$ inner products, $\Phi:
X'\to Y$, $H: X\to Y'$. Full treatment in
`docs/varpro-whitening-notes.tex`.

**The row model.** Fix row $\rho$, mass $m_\rho$, and a local
neighbor/support batch of $K$ column points with masses $m_j$:
$$H[\rho,j] \approx \underbrace{m_\rho m_j \textstyle\sum_i c_i\,\phi_i(x_j;\theta)}_{\text{smooth}} + \underbrace{m_\rho \textstyle\sum_d s_d\, e_d(j)}_{\text{extra (spike, ...)}}.$$
The smooth features $\phi_i$ are theta-dependent Laguerre-Gaussian modes on
a per-row ellipsoid; they carry the column mass $m_j$ because they're
continuum-kernel evaluations at a quadrature point. The "extra" basis
$e_d$ is theta-independent (a one-hot vector for the diagonal spike;
generalizes to e.g. a ring-neighbor correction) and carries no column
mass, because it's a direct discrete correction, not a quadrature object.

**The Laguerre-Gaussian basis.** Real eigenfunctions of the $N$-D quantum
harmonic oscillator: a harmonic polynomial (angular part, generalizes
2D's $\cos/\sin$) times a radial generalized-Laguerre polynomial times a
Gaussian, $L^2(\mathbb{R}^N)$-orthonormal. The harmonic-polynomial part is
generated offline in exact rational arithmetic (`generate_lg_harmonics_table.py`)
for $N=1..4$, oscillator level $\le 10$, and committed as a literal table.

**The ellipsoid pullback.** $T(\theta,x) = L(\theta)^{-1}(x-\mu(\theta))$
maps a physical point into the LG basis's natural round coordinates.
$\theta$ is a log-Cholesky encoding of the local covariance ellipsoid,
with $\mu$ either free (part of $\theta$) or fixed at a given constant.
**Notation: $T$ denotes the pullback**, not a forward map -- if a forward
map is ever needed, call it $T^{-1}$.

**VarPro.** The ellipsoid parameters $\theta$ are nonlinear; the
coefficients ($c$, $s$) are linear given $\theta$. Variable projection
eliminates the linear coefficients in closed form at every trial $\theta$
(an inner least-squares solve), so the outer Levenberg-Marquardt loop only
ever sees the small, well-conditioned reduced problem in $\theta$.

**Noise whitening** (`docs/varpro-whitening-notes.tex`, session
2026-07-24) is the mechanism that lets the smooth ($X$-valued, $M_2$-
weighted) and extra ($X'$-valued, $M_2^{-1}$-weighted) bases combine
without the fitting code ever touching a mass matrix: rescale every basis
function, derivative, and datum once, by $\sqrt{m_\rho}\,M_2^{\pm1/2}$, so
the whole per-row fit becomes an ordinary Euclidean least-squares problem.
Proven (not just assumed) that plain-Euclidean orthogonalization of the
whitened bases is exactly the correct (dual-space-respecting) projection,
and that the whitening operator being fixed/symmetric means every
existing JVP/VJP composes with it for free -- no new derivative math.


## Code architecture

**[`dev/architecture.md`](dev/architecture.md) is the map**: the layering
invariant, a per-header summary, and the house rules. It is written against
`include/lgpsf/` and kept current; do not re-derive the structure from
elsewhere.

The one-line version, each level depending only downward:

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

`lg_operator.hpp` depends on none of the fitting stack, and a test asserts it:
an operator can be built from a physics model, merged, validated, applied and
assembled without `operator_fit.hpp` being included at all.

## Conventions

- **Points are `(K, N)` in C++** -- points as ROWS -- and `(N, K)` at the
  Python boundary. These are the same bytes, which is what makes the binding
  zero-copy. Per-row records are `(R, ...)`, row first. Both differ from
  `ellipsoid_tree`'s Python convention. See `docs/api-guide.md`.
- **Vectorize the point batch only.** Loops over mode count, spatial dimension
  $N$, parameter count $P$ and recurrence depth are fine and often preferred:
  they keep memory at `O(batch)` rather than `O(axis * batch)`.
- **`MuMode` is the fit-mu/pin-mu switch**, threaded through
  `ellipsoid_transform.hpp` and everything above it. `mu0` is always required,
  so `N = mu0.size()` and there is no separate dimension parameter.
- **No special-function library in the core math.** `genlaguerre` and the
  harmonic table are explicit recurrences and generated literals, because C++
  has nothing to call. Ordinary linear algebra through Eigen is fine.
- **Every JVP/VJP pair gets two kinds of test**: finite differences *and*
  adjoint consistency (`sum(w * jvp(v)) == sum(vjp(w) * v)` for random `w`,
  `v`). The second has caught real sign and transpose bugs the first missed --
  do not skip it for new derivative code.
- **Bit-identity is the acceptance criterion** for any change meant to be a
  pure restructuring. It has caught a 1-ULP regression that `allclose` would
  have passed.

## Current state

The C++ library is complete and validated; the Python bindings cover the whole
API. What remains is packaging.

| | |
|---|---|
| C++ core | complete -- 145 cases / 105,699 assertions |
| Python bindings | complete -- 49 pytest cases |
| Examples | 15, covering every exported name |
| Docs | user-facing set written; Doxygen not yet configured |
| Field-scale validation | reproduced at both basal-friction states; see `docs/validation.md` |
| **Outstanding** | **CI, an exercised cibuildwheel run, and M6 release infra** |

Read [`dev/HANDOFF.md`](dev/HANDOFF.md) for open threads and what is parked.
It is tracked, not gitignored.

Two facts worth knowing before touching anything:

- **Build with optimization.** A Debug build is ~33x slower, which is enough to
  make an example look hung. If something seems stalled, check
  `CMAKE_BUILD_TYPE` first.
- **Never a bare `-j`.** `-j3` normal, `-j2` for sanitizers. This machine has
  been OOM-crashed by large parallel builds and a hook enforces the cap.

## Where to look for more

| | |
|---|---|
| `docs/` | For USING the library: installation, quickstart, API guide, defaults, validation, reproducibility. |
| `dev/architecture.md` | The header map and the layering invariant. |
| `dev/HANDOFF.md` | Open threads, parked work, machine-specific build notes. |
| `dev/design-notes.md` | The running decision log, oldest first. Some early entries predate the C++ and are marked where superseded. |
| `dev/archive/` | Closed threads: the executed port plan, the API plans, session records. |
| `experiments/` | Measurements about the library -- where the time goes, and whether refining the mesh costs more probes (it does not). |
| `examples/` | Fifteen examples, each teaching one thing. |
| `archive/python-prototype/` | The frozen Python the method was developed in. History, not a reference. |
