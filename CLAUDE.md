# lgpsf — project guide

This file is auto-loaded into context for any Claude Code session working
in this repo. It's meant to get a fresh session oriented quickly, without
needing to read prior conversation transcripts. Keep it current as the
project evolves; it describes present state, not history (for
session-by-session narrative, see `dev/HANDOFF.md` -- gitignored,
maintainer-local, not visible here if you're reading this fresh).

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

**Two-phase plan.** Build and verify the whole method in Python first
(`prototype/`), as the reference implementation; port to a header-only
C++17 library (`include/`, depends on Eigen + `ellipsoid_tree`) once the
design is settled.

> ### PROTOTYPE FROZEN as of `e5c36c9` (2026-07-25)
>
> The C++ port has started (`docs/cpp-port-plan.md`, milestones M0-M6).
> From that commit, **`prototype/` is the frozen reference
> implementation**: its tests keep running and stay green, but new
> development lands C++-first. Change it only for (a) fixes needed to
> keep its own tests passing, (b) the harmonic-table generator's C++
> emission mode, and (c) deliberate, recorded re-openings of a design
> question -- in which case the C++ side follows, and both get the same
> bit-identity or tolerance certification the frozen work carried.
>
> Do not "improve" the prototype to match something learned in C++
> without saying so explicitly: the two are supposed to be comparable,
> and the prototype is what the cross-language tolerance tests measure
> against.

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

## Code architecture (`prototype/`, bottom-up)

1. **`lg_functions.py`** -- pure LG basis math, no ellipsoid/theta.
   `genlaguerre` (3-term recurrence, no scipy -- ports directly to C++,
   which has no special-function library). `lg_norm` (the combining
   constant). `eval_lg` (2D-only reference, matches the original
   research note's cos/sin convention exactly).
   `eval_lg_nd`/`grad_eval_lg_nd` (general $N$), each a direct statement
   of $\psi = C\,Y_{\ell,m}(u)\,L_p^\alpha(r^2)e^{-r^2/2}$ and its
   product rule over (1b)'s harmonic polynomials -- this module knows
   nothing of the harmonic table. **`eval_lg_basis`/`grad_lg_basis` are
   the PRODUCTION entry points**: the mode index factorizes (angular
   independent of $p$, radial independent of $m$, Gaussian independent
   of both), so a mode SET is the natural unit -- $r^2$ and the Gaussian
   once, each harmonic once per distinct $(\ell,m)$, each radial profile
   once per distinct $(p,\ell)$ from one Laguerre recurrence per $\ell$
   (which also yields the derivatives, since
   $\alpha(\ell)+1 = \alpha(\ell+1)$). Bit-identical to the
   one-at-a-time path, which is retained as the readable reference and
   the exactness oracle. Gradients come back per-mode and UNCONTRACTED
   -- the exact Golub-Pereyra VarPro variant needs that tensor.
   `vjp_lg_basis(modes, u, w)` gives $\nabla_u\sum_i w_i\psi_i$ by
   regrouping the sum per SHELL (one $(N,K)$ op per shell, not per mode;
   the $(n_{modes},N,K)$ tensor is never built) -- the only path that is
   tolerance-certified rather than bit-identical.
   `modes_up_to_level` (raises rather than truncating past the generated
   table).
1b. **`harmonic_polynomials.py`** -- the harmonic polynomials
   $Y_{\ell,m}$ on $\mathbb{R}^N$: `eval_harmonic`, `grad_harmonic`
   (returns `(Y, dY)` from one pass), `harmonic_terms`,
   `num_harmonics`, `max_degree`/`max_dimension`. **The only place the
   generated table's storage format appears** -- the same confinement
   discipline `whitening.py` applies to the mass matrices, so a change
   to how the polynomials are represented or evaluated (term ordering,
   precomputed power tables, a reduced-precision path) is scoped to one
   file with its own intrinsic test suite.
2. **`generate_lg_harmonics_table.py`** -> **`lg_harmonics_table.py`** --
   offline generator (exact rational: monomial harmonic projection +
   Gram-Schmidt via Gaussian moments) for the $N$-D harmonic-polynomial
   table, committed as literal data (~278KB), not computed at runtime.
   Each polynomial is stored as its NONZERO terms only: 91.5% of the
   dense coefficients are exactly zero for structural reasons (exponent
   parity classes; Gram-Schmidt staircase), the drop test is an exact
   `Fraction != 0` and never a tolerance, and the change was certified
   bit-for-bit value-preserving. See `docs/design-notes.md`.
3. **`ellipsoid_transform.py`** -- the pullback $T(\theta,x)$ and its
   JVP/VJP, built as two composable stages (theta -> (mu, L), then the
   pullback geometry itself) so different theta encodings (`mu0=None` =
   fit mu, `mu0=<array>` = fixed mu) never touch the geometry code.
   Triangular solves via explicit `L^{-1}` + `einsum` (theta is never
   batched, so this is negligible cost and avoids a substitution loop).
4. **`lg_ellipsoid_feature.py`** -- composes (1) and (3):
   $\phi_i(x;\theta) := \psi_i(T(\theta,x))$ and its JVP/VJP/Jacobian,
   for a list of modes at once. Thin: one `eval_T` plus one
   `eval_lg_basis`/`grad_lg_basis` call, then the contraction that
   distinguishes the four functions (against `du`, against the theta
   Jacobian, against a cotangent). All mode-level sharing lives in (1);
   all theta knowledge lives here. Mass-free.
5. **`whitening.py`** -- **the only place $M_1$/$M_2$ appear anywhere in
   this codebase.** `whiten_probes`/`whiten_data`/`whiten_extra` for
   user-supplied raw arrays, plus whitened wrappers around (4)'s
   eval/JVP/VJP/Jacobian.
6. **`varpro.py`** -- the generic, mass-free VarPro fitting core,
   **implemented**: `fit_varpro(z_hat, y_hat, basis, theta_init,
   e_hat=None, options)` -> `VarProResult`, where `basis(theta)` returns
   a per-theta evaluation object (`values()`, `vjp(w_hat)`, optionally
   `jac()`) held in the same one-entry cache as the inner solve -- ONE
   callable, not an eval/vjp/jac triple, because the values and the
   derivative are needed at the same theta with the linear solve in
   between (the Kaufman cotangent is built from the values), so they
   cannot be fused but can share one evaluation. Golub-Pereyra is a
   capability of the evaluation (`jac()` present), not an optional
   argument.
   Internally: Frisch-Waugh-Lovell preprocessing of the theta-independent
   extra block (project it out once, back-solve its coefficients `s` at
   the end); an SVD-based inner solve (`_inner_solve` -- the "projection"
   in variable projection: equilibrated, ridge on the coefficients only,
   rank-truncated); `_ReducedProblem` with the reduced residual and two
   Jacobian variants -- Kaufman (default; built in ONE batched
   reverse-mode `basis_vjp` call with cotangent $w[i,j]=c_i$, the
   $(n_{modes},P,K)$ tensor never materializes) and exact Golub-Pereyra
   (needs `basis_jac`; used for strict FD testing and as a fallback);
   outer LM loop via `scipy.optimize.least_squares(method="lm")` (the one
   delegated piece -- the C++ port will need a hand-rolled LM, to be
   validated against this). Both variants share the exact gradient; see
   `_ReducedProblem`'s docstring for the Kaufman<->GP<->Schur-complement
   relationships.

7. **`init_dictionary.py`** -- initial-ellipsoid hypothesis generation
   as a standalone library (geometry + policy, no probes, no fitting):
   `theta_from_L`, `window_shape` (mass-weighted covariance of the
   batch geometry -- measures the window REGION, not mesh density),
   `oriented_sigma`, `local_spacing`/`window_radius`, `ladder_scales`,
   `mid_out` ordering, `circle_rungs`/`window_rungs`.
8. **`probe_moments.py`** -- zero-matvec estimators from probe data:
   `backproject` (unbiased raw-target estimate for iid-normal probes)
   + `raw_moments` ((mu, Sigma) from raw values, which already carry
   the lumped-mass quadrature weight so NO mass vector is needed; spike
   excluded via `spike_index`; `rel_threshold`/`noise_mad` noise
   thresholds). Future home of the parked conservative-field estimator
   (`docs/probe-moment-ellipsoids.md`).
9. **`probe_fit.py`** -- the general-purpose top layer: fit a target
   function known only through inner products with random probe fields
   (an operator row is the motivating example, not the definition).
   `fit_from_probes(x, m2_diag, z, y, mu0, modes, spike_index=,
   sigma0=, config=ProbeFitConfig(...))` -> `ProbeFitResult`. Whitens
   internally (masses *routed* here, mass math stays in
   `whitening.py`). One ORDERED CANDIDATE STREAM over the
   data-adjudicated axes -- init family x scale (from
   `init_dictionary`, `sigma0` first, rungs middle-out), nested
   mode-set ladder (`config.mode_levels`, counting-rule
   `k >= 2(m+extra+P)` admissibility, jittered warm starts across
   levels), fixed vs released mu -- under ONE selection rule:
   window-containment admissibility, then linear-stage K-fold CV score
   (`linear_cv_score`, public -- can score a priori models with zero
   fits; never in-sample cost), then a simplicity tie-break (fewer
   modes, then pinned mu, within `tie_delta`). Early stopping is
   one-sided adaptive effort: `target_score` exits when a candidate is
   certifiably good, `mode_patience` stops the mode ladder; hard
   targets fail the certificates and buy the full grid. Structural
   facts (window, probes, masses, `spike_index`) are caller-declared,
   never adjudicated; numerics stay in `VarProOptions`. THE MODE AXIS
   IS A POLICY (`mode_policy.py`, entry 9b): the stream polls
   `ModePolicy.propose()` for nested mode sets and supplies adaptive
   feedback (an exact margin-profit scorer built from the current
   winner's residual); `mode_levels`/`mode_sets`/`modes` resolve to
   ShellLadder/ExplicitLadder/FixedSet, so legacy callers are
   fingerprint-identical. `target_mass=`
   overrides the default target-mass inference (`m2_diag[spike_index]`,
   exact only square-equal-mass) -- rescales only the returned `(c, s)`;
   theta/scores/selection are invariant.
9b. **`mode_policy.py`** -- the mode-growth policy axis
    (docs/mode-policy-plan.md): stateless policies proposing nested
    mode sets from `ctx.history`; engine keeps every guard. Built-ins:
    `FixedSet`, `ShellLadder`, `ExplicitLadder`,
    `WedgeLadder(max_level, ell_max)` (level-ordered ell-capped --
    strongest fixed policy at k >= 40 on PIG), `RadialFirstLadder`,
    and `MarginGreedy` (adaptive downward-closed frontier; PARKED
    per the PIG benchmark -- gate-free it ties the best fixed policy
    at k=20 but trails the wedge ~2x at k>=40; needs the
    novelty-floor/conditioning refinement, see the plan doc's
    addendum). `modes_up_to_level` gained `ell_max` (wedges).
    PIG evidence: best growth order is budget-dependent (shells win at
    k=20, wedge ties shells at 1/6 cost at k=100); WedgeLadder(10, 2)
    is the OPERATOR-LAYER DEFAULT when no mode source is given.
10. **`operator_fit.py`** -- the whole-operator layer over (9), per
    `docs/operator-api-plan.md`: `fit_operator(x_cols, m1_diag,
    m2_diag, V, HV, sigma, mu0=, modes=, x_rows=, rows=, windows=,
    config=OperatorFitConfig(tau_window=10, spike=True,
    row=ProbeFitConfig(...)))` -> `OperatorFit`. With NO mode source
    given, defaults to `mode_policy=WedgeLadder(10, 2)`. The fitted object is
    the parametric two-component sum `H~ = M1 Phi~ M2 + M1 S` (smooth
    semi-discrete continuum kernel, rectangular by nature + sparse
    dof-tied spike, square by nature), NEVER a matrix. Geometry queries
    go through the `ellipsoid_tree` library (pip: `ellipsoid-tree`;
    the C++ port links it anyway), confined to this layer like scipy --
    indexing, never reference math. Per row: gate -> window (BallTree
    ball query; radius `tau_window * `largest 1-sigma axis of the
    user's best-guess `sigma[rho]`) ->
    `fit_from_probes(sigma0=sigma[rho], target_mass=m1[rho])` ->
    always-on BASELINE GUARD (linear LG fit at `sigma[rho]` pinned at
    `mu0[rho]`, CV-scored on the same folds; the searched fit ships
    only if strictly better -- never worse than the a-priori status
    quo, by construction) -> status `fit | gated_out |
    fallback_baseline | failed`. Output = padded flat arrays (theta
    ALWAYS stored free-mu-encoded, mu, L, c + `mode_set_id` decoder, s
    (additive convention), score, baseline_score, stop_reason,
    released, status, failures, and the per-row FIT WINDOWS as
    CSR-style `window_indptr`/`window_indices`). DEPLOYED SUPPORT ==
    FIT WINDOW: every dof-context helper restricts row rho to its
    window (the slice-38 invariant -- windowed CV is blind to
    out-of-window model energy and LG modes extrapolate violently, so
    fitted object == deployed object by construction).
    Component-typed helpers: `eval_kernel` (the RAW smooth component,
    arbitrary points, unrestricted), `eval_entries`/`matvec`/
    `to_linear_operator`/`assemble_sparse(tau, symmetrize=)` (the
    deployed operator; symmetry is an assembly policy; tau-support
    pattern for ALL rows from one `collision_pairs` points-tree x
    ellipsoid-tree dual descent, intersected with the windows),
    `ellipsoid_field` (the (mu, Sigma) stack `EllipsoidTree` consumes
    directly), `qc_map`, `spike_measure`.

Tests mirror this file-by-file (`test_harmonic_polynomials.py`, whose
checks are all intrinsic -- term-list well-formedness, the parity and
staircase structure, `num_harmonics` against the closed-form
$\dim\mathcal{H}_\ell$, $\Delta Y = 0$, homogeneity, and orthonormality
on $S^{N-1}$ by exact Gaussian moments; `test_lg_functions.py`, whose
keystone is $\int\psi_i\psi_j = \delta_{ij}$ over $\mathbb{R}^N$ by
tensor-product Gauss-Hermite -- exact to roundoff, so it pins the
table, the Laguerre recurrence, the normalization and the
$\alpha$ bookkeeping at once;
`test_ellipsoid_transform.py`, `test_lg_ellipsoid_feature.py`,
`test_whitening.py`, `test_varpro.py` -- the latter's end-to-end check
recovers a known $(\theta^*, c^*, s^*)$ from perturbed initialization --
`test_probe_fit.py`, whose synthetic targets exercise the raw-data
contract end to end, including nonuniform masses and the
backprojection-rescues-bad-center workflow; `test_mode_policy.py`,
whose fingerprint tests pin the legacy-config equivalence and whose
MarginGreedy tests pin true-support recovery, the noise-gate stop, and
the counting budget; and `test_operator_fit.py`, whose synthetic
operator has $M_1 \ne M_2$ throughout so exact coefficient recovery
also pins the target-mass routing).
Examples: `examples/plot_lg_modes.py` (2D mode grid),
`examples/lg_expansion_convergence.py` ($N=1,2,3$ convergence study).

## Conventions (see `docs/design-notes.md` for the full reasoning on each)

- **Point-batched arrays are real numpy arrays, never tuples.** Shape
  `(N, *batch_shape)` -- non-batch axes first, batch axes last, matching
  numpy's row-major-contiguous default (Eigen's column-major default
  means the *transpose* convention, `(K, N)`, is the right one there --
  don't carry the numpy convention over to the C++ port unchanged).
- **Vectorize the point batch only.** Loops over monomial count, spatial
  dimension $N$, theta's parameter count $P$, Laguerre recurrence depth
  are fine and often preferred (keeps memory at `O(batch)` not
  `O(axis * batch)`); don't reach for `einsum`/broadcast tricks over those
  other axes by default.
- **`mu0=None` vs `mu0=<array>`** is the fit-mu/fixed-mu switch, threaded
  through `ellipsoid_transform.py` and everything built on top of it.
- **No scipy in the core reference math** (`genlaguerre`, the harmonic
  table generator) -- these need to port to C++/Eigen, which has no
  special-function library, so the Python reference is written as
  explicit recurrences/formulas that translate directly. Ordinary linear
  algebra (matrix inverse, etc.) is fine via plain numpy, since Eigen has
  native equivalents there -- the scipy avoidance is specifically about
  special functions, not linear algebra in general.
- **Every JVP/VJP pair gets two kinds of test**: finite differences
  *and* adjoint-consistency (`sum(w * jvp(...)) == sum(vjp(...) * v)` for
  random `w`, `v`). The second has caught real sign/transpose bugs that
  the first alone missed -- don't skip it for new derivative code.

## Current status / what's not built yet

- **The C++ port: M0-M3 COMPLETE, M4 next** -- `docs/cpp-port-plan.md`
  (deps, threading, header-only, bindings, milestones M0-M6, the
  hand-rolled LM contract, and the COMPILE MEMORY SAFETY rules: this
  machine has been OOM-crashed by large `-j` builds -- **read that
  section before building any C++**). Shipped in M0:
  `include/lgpsf/detail/lg_harmonics_table.hpp` (generated alongside
  the Python table by the same run of
  `prototype/generate_lg_harmonics_table.py`, so they cannot drift),
  `include/lgpsf/harmonic_polynomials.hpp`,
  `include/lgpsf/lg_functions.hpp` (`Mode`, `genlaguerre`, `lg_norm`,
  `modes_up_to_level`, `LGBasisAt` with values/grad/vjp, the
  one-at-a-time reference functions, and the 2D `eval_lg` convention
  oracle), plus `tests/pch.hpp`, `tests/test_helpers.hpp` and the two
  intrinsic test files. Build with
  `cmake --build build --target lgpsf_tests -j3` (NEVER a bare `-j`).
  Shipped in M1: `ellipsoid_transform.hpp` (`EllipsoidFrame`,
  `MuMode`, the two encodings, both stages and their JVP/VJP),
  `lg_ellipsoid_feature.hpp` (`FeatureAt`), `whitening.hpp`
  (`WhitenedBasis` -> `WhitenedBasisAt`, the masses layer). Suite:
  49 cases / 98,496 assertions. **THE PARAMETER VECTOR HAS TWO
  ENCODINGS in C++** (`docs/design-notes.md`): public `theta` is
  absolute `[mu, log-diag, strict-lower]`, always `N(N+3)/2` long, and
  `unpack_theta(theta)` needs nothing else -- that is what keeps
  `fit_operator(mu0=None)` meaningful. Internal `theta_hat` is what the
  fitting core sees: the center as a DISPLACEMENT from `mu0` when
  fitted, absent when pinned. `mu0` is required everywhere (so `N =
  mu0.size()`; there is no `N` parameter and no optional `mu0`), and
  the fit/pin switch is `enum class MuMode`.
  Shipped in M2: `detail/levenberg_marquardt.hpp` (generic, tested on
  problems with published answers -- NOT through VarPro), `varpro.hpp`
  (`fit_varpro`, templated on the basis functor), and `exceptions.hpp`
  (`InfeasibleParameters`: "no basis exists at this point in parameter
  space", which the fitting core catches and scores as the worst finite
  cost -- caller errors stay `std::invalid_argument`). Suite: 74 cases
  / 98,652 assertions. The LM solves its trust-region subproblem by an
  SVD hook step rather than MINPACK's `lmpar`; measured against scipy
  it is marginally faster to converge with identical answers.
  Shipped in M3: `init_dictionary.hpp`, `probe_moments.hpp`,
  `mode_policy.hpp` (virtual `ModePolicy` with `propose` **const**;
  MarginGreedy stays parked Python-side) and `probe_fit.hpp`. Suite:
  116 cases / 100,365 assertions. **TWO POLICY CHANGES from the
  prototype, both deliberate.** (1) `MuPolicy::Pinned` is the DEFAULT --
  the slice-38 evidence is that release ships on ~91% of rows while
  buying nothing, so it stays available on request but is no longer
  automatic until a basin-scale `||mu - mu0||` bound re-arms it.
  (2) **RANDOMNESS IS HOISTED TO `operator_fit`**: the CV split and the
  warm-start jitter arrive at `fit_from_probes` as DATA, so it and
  everything beneath it are pure functions of their inputs -- checkable
  as "no `<random>` below `operator_fit.hpp`". Folds default to
  round-robin with no generator at all. See the plan's randomness
  section for what that rests on (probe exchangeability) and when to
  revisit it. M4 is `operator_fit.hpp` with `parallel_for` over rows
  and native ellipsoid_tree geometry.
- A hand-rolled Levenberg-Marquardt loop (port milestone M2): the
  prototype's outer loop delegates to scipy/MINPACK -- the one
  delegated numeric; everything else in `varpro.py` is library-free.
- Released-mu re-arming: `ProbeFitConfig`'s default
  `mu="fixed_then_release"` is safe at single-row scale, but PIG
  field-scale evidence (research-repo slice 38) showed release
  shipping on ~91% of rows while buying nothing, guarded only by the
  far-too-loose window-radius bound on the center. It needs a
  basin-scale `||mu - mu0||` bound before being trusted at operator
  scale; operator-level experiments pin mu meanwhile.
- MarginGreedy adaptive mode policy: PARKED with evidence and queued
  refinements (novelty floor first) -- see `docs/mode-policy-plan.md`'s
  addendum. `WedgeLadder(10, 2)` is the operator-layer default.
- Level-2 cross-row amortization (neighbor warm starts /
  smoothed-theta seeds as candidate injection) and field-level QC maps
  over status/stop_reason/spike-measure -- build on `fit_operator`.
- Python bindings (`bindings/` doesn't exist yet; port milestone M5;
  the hook location is marked by a comment in `CMakeLists.txt`).

(The library HAS been validated at field scale against real
PDE-Hessian probe data -- PIG slices 38/39 in the separate
maintainer-local research repo: it beats the prior psfladder reference
at every probe budget, at both smooth and rough basal-friction states;
the in-repo tests remain synthetic by design.)

## Where to look for more

- `docs/design-notes.md` -- terse, running log of C++-port-relevant
  decisions (layout, vectorization principle, etc.).
- `docs/varpro-whitening-notes.tex`/`.pdf` -- the whitening derivation in
  full, with the row-model and reconciliation-with-earlier-analysis
  arguments spelled out.
- `dev/HANDOFF.md` -- gitignored, maintainer-local session notes; not
  visible if you're reading only what's checked into the repo.
