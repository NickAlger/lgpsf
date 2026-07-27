# Issues encountered, session 2026-07-25/26 (M1–M4)

A record of the problems, discrepancies and defects that came up while porting
M1–M4, with what was considered and what was done. Descriptive only.

Sections A–F were written partway through; section G continues from there.

---

## A. Design questions reopened during the port

### A1. `mu0` and `N` were both redundant in the low-level signatures

The prototype threads `(theta, N, x, mu0=None)` through twelve functions, with
`mu0=None` doubling as the fit-mu/fixed-mu switch. The first C++ proposal was a
`ThetaEncoding` value type bundling `(dim, mu0)` with named constructors.

Nick rejected that as heavy-handed and pointed out that `mu0` is required at
the row-fit layer and above regardless of the mu mode, and that `N` is
recoverable from `mu0.size()`. Checking the prototype confirmed it:
`fit_from_probes` takes `mu0` positionally and uses it for the ladder scales,
the init seeds and the containment reference; the `None` that reaches the low
layer is manufactured at `probe_fit.py:386`.

Done: no `N` parameter and no optional `mu0` anywhere in the module. `mu0` is
required, `N = mu0.size()`, and the switch is `enum class MuMode`.

### A2. Two parameter encodings

Following from A1, the question was whether the fitted center should be stored
absolutely or as a displacement from `mu0`. Considered: absolute only (matching
the prototype); displacement only; one always-fitted encoding with an optimizer
active set masking the pinned components.

The active-set option was dropped because pinned mu is the operator-layer
default, so it would compute `N` identically-zero Jacobian columns per
iteration, and MINPACK's `x_scale='jac'` column-norm scaling divides by exactly
those norms.

Nick specified that the operator layer's returned `theta` must not depend on
`mu0`, since `fit_operator(mu0=None)` defaults each row's center to its own
source point, and accepted the displacement form internally under a different
name.

Done: public `theta` (absolute, always `N(N+3)/2` long, decodes with
`unpack_theta` alone) and internal `theta_hat` (displacement when fitted,
absent when pinned). The round trip is not bit-exact — `(mu0 + delta) - mu0`
recovers `delta` only to the resolution of `mu` — which a test asserts against
that bound rather than against equality.

### A3. Point-batch layout

Nick asked whether the C++ should use points as columns, recalling an earlier
intent that the two languages each use their natural layout. The existing
design note was headed "Eigen is transposed relative to numpy", which framed it
as a transpose.

Measurement: numpy `(N,K)[d][k]` and Eigen `(K,N)(k,d)` are both at offset
`d*K + k` — the same bytes, one coordinate-major layout spelled natively in
each language. Benchmarked coordinate-major against point-major at K=2000:
`r^2` 2.1–3.2x, one harmonic term 1.8–2.0x, the pullback GEMM 1.3–1.5x, all in
favour of coordinate-major.

Done: layout unchanged; the note re-headed "coordinate-major, spelled natively
in each language" with the measurements.

### A4. Randomness

Survey found randomness in exactly two places: the CV fold assignment
(`probe_fit.py:168`) and the warm-start jitter (`probe_fit.py:501`). Probe
fields are caller-supplied.

Considered: per-row seeds `seed + rho` inside the row fit (the plan's original
policy); deterministic round-robin folds; hoisting both to the operator layer.

Nick proposed hoisting. The argument that decided it was not thread safety but
that the split is field-level data: `V` is `(K_all, k)` and every row uses all
`k` probes, so there is one split for the whole fit, and "candidates are only
comparable on identical folds" becomes structural rather than maintained by
re-seeding with a fixed value at every call.

Nick asked that this not be written as a hard policy, noting that structured
probes (sinusoid, polynomial) would break the exchangeability that round-robin
folds rest on.

Done: split and jitter are built in `fit_operator` and passed down as data;
folds default to round-robin with no generator; the plan's randomness section
records the exchangeability dependency and that the hoisting is what makes
revisiting cheap.

### A5. Default mu policy

The prototype defaults to `fixed_then_release`. `CLAUDE.md` records that
release ships on ~91% of PIG rows while buying nothing, guarded only by a
window-radius bound on the center.

Nick: default to `Pinned`, allow override.

Done: `MuPolicy::Pinned` default; `Free` and `PinnedThenRelease` available.

### A6. The operator window: ball vs ellipsoid

The longest thread of the session. Sequence:

1. Reading `operator_fit.py` for M4, I proposed replacing the ball window with
   an ellipsoid query, with a capped aspect ratio, having measured that a ball
   window makes `window_shape` recover aspect 1.000 where the prior's own
   ellipse gives 8.016.
2. Nick pushed back that this was drifting from work already settled.
3. Reading `operator-api-plan.md` showed the ball stated as a decision
   with a rationale ("conservatism is supplied by tau_window inflation ... this
   justification travels with the parameter"). I withdrew the proposal.
4. Nick said the ellipsoid was the design intent all along and asked when the
   ball was chosen and why.
5. Git archaeology: `8e64243` (17:34) justifies the `window_shape_rungs`
   family with "in the intended pipeline it IS a conservatively inflated
   ellipsoid, whose aspect and orientation survive the inflation". `cfcc6f1`
   (21:47, same day) specifies a kd-ball. `1ea8d25` implements it. The operator
   doc does not note the contradiction. The research repo shows the ball
   predates lgpsf (`slice37`: "conventions are slice 36's, verbatim"), and
   every PIG operator run sets `window_shape_rungs=False`
   (`slice38:116`, `slice39:214`).
6. Nick's reading: a miscommunication between sessions, cause unknown; the ball
   is validated at field scale and the ellipsoid is not; the C++ speed makes
   re-running the comparison affordable. Decision: default to the ellipsoid,
   offer a boolean for the ball, settle it later by experiment.
7. Nick then asked that the ball be an isotropic ellipsoid rather than a
   separate query type, and that all windows come from one tree × tree dual
   descent.
8. Nick subsequently replaced the boolean with the continuous aspect cap.

Done: `window_aspect_cap` floors sigma's eigenvalues at `lambda_max / cap^2`.
`cap = 1` is isotropic (the ball), `cap = infinity` is the caller's ellipsoid
(the default), intermediate values give axis ratio `min(prior's, cap)`. One
ellipsoid query throughout, so only the shape matrix changes. All rows'
windows from one `collision_pairs` descent. Measured window points at a 3:1
prior, tau=1.5, cap 1 / 1.5 / 3 / infinity: 363 / 331 / 205 / 205.

---

## B. Discrepancies between documents, code, and experiments

### B1. `P_fix` is the free-mu count

`operator_fit.py:330` reads `P_fix = N + N*(N-1)//2 + N  # == theta_size(N,
mu0=any)`. The expression equals `theta_size(N, None)` — the free count (5 vs 3
at N=2, 9 vs 6 at N=3, 14 vs 10 at N=4) — so `P_fix == P_free`, `P_stream` is a
no-op, and the baseline guard used a stricter budget than the search.

Raised twice before Nick responded; his reading is an unintended bug, uncaught
because it is conservative. Instruction: the guards should count the parameters
actually being fit.

Done: baseline counts `N(N+1)/2` (pinned), search counts its stream encoding.
Note the instruction's formulas `N(N-1)` and `N + N(N-1)` coincide with these
at N=3 but not at other N; `theta_hat_size` computes the general ones.

An earlier version of `probe_fit.hpp` had copied the prototype's behaviour and
justified it in a comment as keeping pinned and released candidates comparable
on the same mode sets — a rationalization written after the fact.

### B2. The plan's claim about per-row generators

`cpp-port-plan.md` stated the prototype "shares one generator across rows
within a chunk". It does not: `default_rng(cfg.seed)` is constructed inside
`fit_from_probes`, so it is already per-row. Corrected.

### B3. The plan's binding-boundary claim

Decision (d) said "points-as-rows `(K, N)` at the numpy boundary with ONE
transpose". Given A3 this is wrong in both halves: bindings should expose
`(N, K)` to match the frozen prototype, and it costs zero copies. Corrected.

### B4. `ellipsoid_tree`'s two conventions

`operator_fit.py`'s docstring says ellipsoid_tree takes points as rows `(K, N)`;
the C++ `KDTree`/`BallTree` take `(dim, n)`. Both are correct — the Python
bindings transpose. The C++ side therefore transposes at the tree boundary,
which first appears in `local_spacing`.

### B5. `window_shape` under ball windows

Measured: a ball window's mass-weighted covariance is isotropic, so
`window_shape` returns ~identity and `window_rungs` duplicates `circle_rungs`.
This is consistent with the PIG runs disabling that family. Recorded in the
prototype README as a stale-comment trap. No change made to the row layer.

---

## C. Defects in the C++ written this session

### C1. `margin_profit` declared but never supplied

`mode_policy.hpp` documented that the engine builds the adaptive-feedback hook;
`probe_fit.hpp` never populated it. Found while writing the tests. Considered
leaving it unpopulated (no shipped policy consumes it, MarginGreedy being
parked) or implementing it. Implemented, built after each successful level, and
tested through a recording policy.

### C2. The LM callback never fired at the returned point

The loop can satisfy its convergence test immediately after accepting a step
and return without evaluating the Jacobian there, so the final point was never
reported. The prototype's own de-duplication workaround (against extra
evaluations scipy inserts) is a different issue and is genuinely unnecessary
here. `fit_varpro` now closes the trace explicitly at the returned point.

### C3. `std::vector` reallocation in `LGBasisAt` (carried from M0, noted here
for completeness): callers held references into a vector that could be extended
in place; switched to `std::deque`.

### C4. Compile-level items

- `Eigen::Block` without `LinearAccessBit` cannot be indexed with one argument;
  `PullbackCotangent::component` now returns `ConstColXpr`.
- Two `project_out` overloads taking `Ref<const MatrixXd>` and
  `Ref<const VectorXd>` were ambiguous for every vector argument; reduced to
  one.

---

## D. Test defects found

### D1. Range-for over a temporary's member (`test_mode_policy.cpp`)

`for (const Mode& m : walk(...).back().modes)` — lifetime extension does not
reach through the member accesses, so the temporary died before the loop body.
Surfaced as a wrong value rather than a crash.

### D2. `NaN != NaN` read as a threading failure (`test_operator_fit.cpp`)

The bit-identical-across-thread-counts check failed on `theta`, `mu`, `L`,
`score`, `baseline_score` while the integer and status arrays passed. Those are
exactly the NaN-padded arrays, and Eigen's `operator==` is all-coefficients-
equal. Diagnosed by comparing a run against a second run at the *same* thread
count, which reported the same "difference" — so neither a race nor a
thread-count dependence. Replaced with a NaN-aware comparison.

### D3. Assertions that were wrong as stated

- **Already-active margin profit.** Asserted an already-active mode buys ~0.
  Its column projects to roundoff, so the ratio is a 0/0 the relative floor
  cannot discriminate; measured 1.55. Replaced with the Cauchy-Schwarz bound
  (a one-step reduction cannot exceed the residual energy) and the case
  documented as outside the contract on both sides.
- **Orientation under the aspect cap.** Asserted orientation survives at every
  cap. At `cap = 1` the result is isotropic and has no orientation — any
  orthonormal basis diagonalizes it. Now asked only where defined.
- **Spike exclusion.** Asserted the recovered center is exactly zero after
  excluding the spike; dropping one point of a symmetric grid leaves it
  lopsided. Changed to "near".
- **Gaussian moment recovery.** The grid was not centered on the true mean, so
  truncation asymmetry biased it. Centered.
- **`noise_mad` threshold.** Tested in the regime the docstring warns against
  (a tight window, where the median overestimates the noise), where
  `raw_moments` correctly suppressed everything and raised. Rewritten in the
  far-field regime it is for: sigma 3.32 -> 0.91 against a bump variance of 1.
- **`raw_moments` inflation bound.** Guessed >5x; measured 3.7x.

### D4. Vacuous tests

- **Admissibility.** Only checked the flag agreed with the rule, which says
  nothing if nothing ever violates it. Rewritten to build a target from an
  ellipsoid 3x the window radius: 12 of 12 candidates ruled inadmissible, and
  the all-inadmissible fallback still returns a model.
- **Window geometry.** Compared window sizes at tau=10 on a mesh of half-width
  1, where every setting swallows the grid (363 = 363). Re-run at a tau where
  the shapes discriminate.

### D5. Tests verified by making them fail

- The whitening keystone: `sqrt(m_rho)` deliberately dropped from
  `whiten_extra`; 28 assertions fail in two cases, while the adjoint and
  finite-difference tests pass straight through the bug.
- The Golub-Pereyra Jacobian test: switched to Kaufman; all 10 assertions fail.

### D6. Claims checked and not confirmed

- `u.rowwise().squaredNorm()` was suspected of costing performance against an
  explicit loop over coordinate columns. Measured a wash (the loop wins 1.5x at
  N=2, loses ~1.2x at N=3, K=20000). Left as written.

---

## E. Raised, and since resolved

- **`window_shape_rungs` default at the operator layer.** Proposed defaulting
  it to `false` in C++. Nick: keep `true`. Note the argument had partly
  dissolved anyway -- it rested on ball windows making the family duplicative,
  and the default window is now the uncapped ellipsoid.
- **`assemble_sparse` output type.** Nick: `Eigen::SparseMatrix<double>` with
  `makeCompressed()`. Done.
- **Reviewing the M0 headers before M1.** Offered, not answered; M1 proceeded.
  Still unanswered, now moot.

---

## F. Measurements taken this session

| what | result |
|---|---|
| coordinate-major vs point-major, K=2000 | `r^2` 2.1–3.2x, harmonic term 1.8–2.0x, pullback 1.3–1.5x |
| `rowwise().squaredNorm()` vs explicit loop | a wash |
| `window_shape` on a ball window vs an 8:1 ellipse | aspect 1.000 vs 8.016; 8x point count |
| hand-rolled LM vs MINPACK, Rosenbrock | 17 evaluations vs 21, both cost 0 |
| hand-rolled LM vs MINPACK, Powell singular | 51 vs 53 evaluations, cost 1.3e-58 vs 2.0e-61 |
| feature layer adjoint / FD | 6.7e-16 / 2.4e-10 |
| whitened layer adjoint / FD | 1.0e-15 / 1.6e-10 |
| whitened regression keystone | 1.2e-15 |
| VarPro synthetic recovery | 4–5 accepted steps, cost ~1e-30, \|dtheta\| ~1e-15 |
| probe-fit synthetic recovery | CV score ~7e-16 |
| operator synthetic recovery, M1 != M2 | \|dc\| 6.8e-8, \|ds\| 2.6e-8 |
| window points by aspect cap (3:1 prior, tau=1.5) | 363 / 331 / 205 / 205 at cap 1 / 1.5 / 3 / inf |


---

## G. After the first record was written (same session, continued)

### G1. `eval_kernel` was untruncated by design, and the design was wrong

The helper table had `eval_kernel` as "component access, not the deployed
operator" -- the raw smooth kernel at arbitrary points, no window restriction.

Nick's argument for changing it: the fit's objective evaluates the LG functions
ONLY on the window, so out-of-window mass is not merely unverified, it is
UNPENALIZED. The optimizer will place mass outside the window to chase noise
inside it, and on rows with small or noisy entries it does. That is the
slice-38 failure (one rogue row, 94% of a whole-operator test error), which
window-truncating deployment fixed completely.

Considered: leave `eval_kernel` untruncated and rely on the dof-context helpers
(rejected -- calling it component access relabels the hazard); truncate to the
fitted ellipsoid at some tau (rejected -- does not bound out-of-window mass,
since a pathological fit has a large fitted ellipsoid); truncate to the window.

Done: truncation to the fit window is the default for every evaluation, with
`eval_kernel_unrestricted` as the named opt-out. One rule for all helpers:
`support = window ∩ fitted tau-ellipsoid`, `truncation_tau` defaulting to
infinity. `assemble_sparse(op, tau)` is that rule with a finite tau and is now
pinned by a test against `eval_entries` at the same tau -- previously nothing
connected them.

### G2. The window could not be applied at non-mesh points

Consequence of G1: `eval_kernel` takes arbitrary coordinates, and the window
was stored only as a CSR index list, which cannot restrict a point that is not
a mesh column. The caller's `sigma` is not stored on the fit, so the window
ellipsoid was not reconstructible either.

Done: the window is stored as a REGION (`window_center`, `window_covariance`,
pre-scaled so membership is Mahalanobis <= 1). The index list stays as the
derived cache and fast path; a test pins that the two agree on every mesh
column. Recorded caveat: the ellipsoid is a proxy for the window POINT SET --
exact on mesh columns, generous off-mesh on a coarse mesh.

### G3. The `windows=` index override had no region

An index list cannot be turned into a region without more information. Options
considered: throw when a region query is made; fall back to untruncated (would
reintroduce the hazard); fit a bounding ellipsoid at construction.

Nick's resolution: remove the `windows=` override entirely and require a window
ELLIPSOID, with a helper to convert an index set. Checked first -- `windows=`
is used nowhere in the PIG experiments and only in one prototype test.

Done: `window_ellipsoids` override; `ellipsoid_from_points` in
`init_dictionary.hpp` (mass-weighted mean and covariance scaled so the farthest
point is on the boundary; exact containment, closed form). `tau_window` and
`window_aspect_cap` do not apply to an override. Recorded cost: an exactly
non-ellipsoidal window can no longer be expressed; through the helper it
becomes a SUPERSET. Measured, 40 hand-picked points -> 42 columns realized.

### G4. `OperatorFit` was two things in one

Nick observed the struct held both a compressed operator representation and
diagnostics about the fit that produced it.

Measured before deciding: counting field accesses across the evaluation
helpers, `score`, `baseline_score`, `stop_reason`, `released`, `failures` and
`config` have ZERO readers each. The only apparent straddler was `status`, read
solely by `model_rows`; `mode_set_id >= 0` encodes that predicate exactly.

Options considered: model as a slot with diagnostics loose alongside (rejected
-- leaves the container half-encapsulated); a tuple return (rejected -- loses
the names); a named struct with two members.

Done: `OperatorFit { FittedOperator model; FitDiagnostics diagnostics; }`.
`model_rows` redefined via `mode_set_id`, with a test pinning the equivalence.
The model keeps its own copies of coordinates and masses (~24 MB at PIG scale)
so it is self-contained. Behaviour preservation was the acceptance criterion:
every pre-existing assertion passed unchanged.

### G5. The name and the file said "fit" when the structure does not require one

Nick observed the structure can be built by other means -- a physics-based
approximation, bypassing the fitter.

Measured: the evaluation helpers reference NOTHING from the fitting stack --
zero mentions of whitening, varpro, mode_policy, probe_fit, init_dictionary.

Done: `FittedOperator` -> `LGOperator`, moved to `lg_operator.hpp`;
`operator_fit.hpp` is a producer of the structure rather than its definition.
`test_lg_operator.cpp` never includes `operator_fit.hpp` and builds every
operator by hand, so the independence is a build property.

Also considered and rejected: turning the helpers into methods. Nothing needs
privileged access, and the type is a transparent flat-array structure meant to
be inspected, MPI-gathered, merged and serialized, so private fields would
fight the property the split was for. Line drawn and written into the header:
methods for "what am I", free functions for "what can be done with me".

Three additions the split made natural: `num_threads` on the row-parallel
helpers (the plan's uniform convention, which `matvec`, `assemble_sparse` and
`qc_map` had been ignoring); `validate`, since hand-construction is now a
supported path and needs to be checkable; `concatenate_rows`, the operation
the slice-38 driver open-codes.

### G6. Further measurements

| what | result |
|---|---|
| window points by aspect cap (3:1 prior, tau=1.5) | 363 / 331 / 205 / 205 at cap 1 / 1.5 / 3 / inf |
| `eval_entries` nonzeros by truncation_tau | 1 / 7 / 26 / 37 / 37 at tau 0.5 / 1 / 2 / 4 / inf |
| `assemble_sparse` nonzeros, tau=1 / tau=40 | 29 / 123 of 14641 |
| `qc_map` on held-out probes, synthetic M1 != M2 | 1.37e-8 |
| `ellipsoid_from_points`, 40 points | 42 columns realized |

### G7. `CandidateFit` was the same mixture, one layer down

Nick asked whether to split it the way `OperatorFit` was split.

Two things the classification turned up that were not anticipated. `released`
was not a diagnostic: it said whether `theta_hat` was Pinned- or Fitted-encoded,
so it was needed to DECODE the candidate. And a `CandidateFit` was not
self-describing -- it held `num_modes` and `modes_label` but not the mode list,
which lived in the engine's `modes_of` map.

Two subtleties Nick flagged in advance. The SPIKE: held inside the expansion
rather than beside it, because `c` and `s` are fitted jointly and the
projection couples them, so separating them at the type level would imply an
independence the mathematics denies -- the same reason `LGOperator` holds both.
The NAME: not `LGFunction`, which reads as a single mode; `LGExpansion` says
"a sum over a basis" without over-committing the way `LGKernel` would.

Done: `LGExpansion` in `lg_expansion.hpp`, holding absolute `theta`, the mode
list, `c` and `s`. `CandidateFit`/`ProbeFitResult` hold one.
`build_operator` / `row_expansion` express the fact that an `LGOperator` row IS
an expansion plus a window plus masses. Recorded asymmetry: `c` carries its
mode list, `s` does not carry the caller's extra basis, because a mode list is
tens of triples and an extra basis is a `(K, num_extra)` array.

One bug, caught by the build/extract round trip: `build_operator` skipped
unmodeled rows without carrying the CSR window offset forward, and the patch
loop written to compensate was convoluted rather than correct.

### G8. Still open

- **`ProbeFitResult` carries a full `candidates` table that `fit_operator`
  never reads.** Each row builds ~12 `CandidateFit`s (two `std::string`s and
  five Eigen vectors each) and discards them. Suspected allocation traffic at
  field scale; NOT measured, and this session produced two cases where a
  plausible performance claim did not survive measurement. To be profiled
  during the M5 replay, where the row count is realistic.
- **Serialization of `LGOperator`.** Now meaningful, since the model is
  self-contained. Probably M6.
