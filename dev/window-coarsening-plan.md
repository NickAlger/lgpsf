# Graded window coarsening — a per-row work bound for the operator fit

Plan for bounding the cost of one row's fit independently of how large its
window is. Drafted 2026-09-06 from a joint session; the design questions
were settled in discussion, the code facts below were established by a full
read of the fit path and carry file:line references so a fresh session can
implement from this document alone. Nothing here is implemented yet.

**One-paragraph version.** A row is fitted on its *window*, every column dof
inside `tau_window` × the caller's a-priori ellipsoid, and the fit's cost is
linear in the window's point count (basis evaluation is about half of a row
fit; see `design-notes.md`, the K = 400 profile). The window is only as good
as the prior: on the maintainer's continental ice-sheet Hessian a few hundred
rows whose prior was too wide by decades had windows of ~90,000 points, they
sat on one rank, and that rank's fit took 5,900 s against a mean of 75 s. The
prior is being repaired on the caller's side, but the library should not
depend on that. The fix here is a quadrature coarsening of the window that is
adequate for *every* kernel width at once, so it needs no estimate of the
true width: cells graded by distance from the row's centre, size at most
`eps` times that distance, floored at single points; each cell becomes its
mass-weighted centroid with the summed mass and the mass-weighted mean of
each probe field. The fit runs on the cells; the deployed operator, whose
support is the full window, is unchanged.

---

## 1. Why this is the right object to bound

The fit never regresses pointwise. Its unknowns are the coefficients of the
kernel on the probe *equations*: with `z_hat = M2^{1/2} z`, `y_hat =
y / sqrt(m_rho)` and the whitened basis `sqrt(m_rho) M2^{1/2} phi`, the design
matrix is contracted over the point axis (`probe_fit.hpp:394-399`,
`operator_fit.hpp:264-269`):

    design(l, i) = sqrt(m_rho) * sum_j  m2_j * z_{j l} * phi_i(x_j)

i.e. a mass-weighted quadrature of the model's action on probe `l`. Two
consequences drive everything below.

1. **Aggregation is exact for the probe's fine structure.** Replace a cell
   `C` by mass `m_C = sum_{j in C} m2_j`, centroid `x_C = sum m2_j x_j / m_C`
   and probe mean `z_C = sum m2_j z_j / m_C`. Then
   `m_C * z_{C l} * phi_i(x_C) = (sum_{j in C} m2_j z_{j l}) * phi_i(x_C)`:
   the probe sum inside the cell is kept exactly (white-noise probes lose
   nothing), and only the *basis function's* variation over the cell is
   approximated, by the midpoint rule.
2. **The spike column is invariant if the spike stays a singleton.** With
   `e_hat = sqrt(m_rho) M2^{-1/2} E` and `E` one-hot at the spike,
   `design(l, extra) = sqrt(m_rho) * z_{spike, l}`, independent of the masses
   (`whitening.hpp:142-150`). A singleton spike cell reproduces that column
   bit for bit and keeps the meaning of `s`, `spike_measure` and the assembled
   diagonal entry (`lg_operator.hpp:716-724`). Merging it would silently
   rescale `s`. This is load-bearing, not cosmetic.

So "subsample the window" is really "choose a coarser quadrature for the
model's action on the probes". Three candidate rules, judged as quadratures:

- *Nearest neighbours to the centre* keep the near field exact but silently
  shrink the window: 2,000 of 90,000 points is a radius 0.15 of the window's,
  about 0.6 prior sigmas. A kernel wider than that is truncated, and
  truncation biases `theta` and `c` rather than adding noise. Fine when the
  prior is good; wrong in exactly the case that motivates the change.
- *Uniform random with mass reweighting* is unbiased but its variance is set
  by how many samples land inside the true support: 500 true nodes in a
  90,000-point window at N = 2,000 gives eleven. Stratification does not
  change the order of magnitude.
- *Radially graded coarsening* is scale-free. For a Gaussian of width `sigma`
  the midpoint error from a cell of size `h` at distance `r` is about
  `(h r / sigma^2) exp(-r^2 / 2 sigma^2)`; maximizing over `sigma` at fixed
  `r` gives `0.74 h / r`. So `h <= eps * r` bounds the error at `eps` for every
  width simultaneously; the centroid rule makes it second order,
  `(eps * ell)^2` at `r ~ sigma` for an angular mode of order `ell`, for a
  probe that is smooth over the cell. **S1 measured (2026-09-06): for
  white-noise probes the design-column error is FIRST order in `eps`** (slope
  1.0 on real windows, 2.0 for smooth probes): a cell keeps the in-cell probe
  sum `sum_j m_j z_j` but drops its dipole `sum_j m_j z_j (x_j - x_C)`. The fit
  tolerates it because the error is zero-mean across probes (at `eps = 0.1`
  the column error is 1.5% at level 0 to 10% at level 5; the score changes by
  ~8e-3 at fixed theta on the largest windows). A dipole-corrected cell would
  restore second order if ever needed; out of scope here. The cell
  count is about `3 pi / eps^2` per dyadic annulus in 2D, so
  `N ~ 3 pi / eps^2 * log2(R / h_mesh)`: logarithmic in the window size, which
  is what makes the fit's cost scale under mesh refinement, and finite in 3D
  (`~4 pi / eps^3` per shell, an octree). At `eps = 0.15` a 90,000-point
  window becomes about 3,000 cells; at `eps = 0.1` about 7,000.

The one real limitation is angular resolution: an `ell = 4` mode wants
`eps <= 0.1` to be resolved at its own radius. Cross-validation is on our
side there: a coarse quadrature makes the high modes look like noise, so the
ladder stops lower, which is the conservative failure. And the deployed
operator is unaffected either way (§3).

**Anisotropy.** Build the cells in the coordinates of the *window* ellipsoid,
`u = L_w^{-1} (x - centre)` with `Sigma_w = L_w L_w^T` the window's covariance
as recorded in `window_center` / `window_covariance` (`operator_fit.hpp:
507-514`; the pullback idiom is `ellipsoid_transform.hpp:408-419`). The window
is already the object that encodes how much of the prior's shape the library
trusts: `window_aspect_cap` floors its minor axes (`init_dictionary.hpp:
236-255`), and a ball window (cap 1, the maintainer's production setting)
makes the grading Euclidean. A user who trusts anisotropic priors lifts the
cap and gets the grading in whitened coordinates for free. For a kernel
rotated against the window, resolution along its major axis degrades by the
window's aspect ratio — the same loss the fit already accepts through the cap.
One tree, no new notion of trust.

---

## 2. The function

New header `include/lgpsf/coarsen_window.hpp`, placed in the layering between
`init_dictionary.hpp` and `probe_fit.hpp` (`architecture.md`, the ladder): it
needs `ellipsoid_transform.hpp` for `EllipsoidFrame` / `pullback` and nothing
above. Not in `init_dictionary.hpp`, whose contract is "no probes" (`init_
dictionary.hpp:9-10`); this touches `z`.

```cpp
namespace lgpsf {

struct CoarseWindow
{
    Eigen::MatrixXd  x;                  ///< (Kc, N) cell centroids, mass-weighted
    Eigen::VectorXd  m2;                 ///< (Kc,)   summed cell masses
    Eigen::MatrixXd  z;                  ///< (Kc, k) mass-weighted mean probe fields
    std::vector<int> protected_cells;    ///< coarse positions of the protected points, in input order
    std::vector<int> cell_of;            ///< (K,) provenance: which cell each point joined
};

/// Pure: no threads, no randomness, no dependence on allocation or rank layout.
inline CoarseWindow coarsen_window(
    const Eigen::Ref<const Eigen::MatrixXd>& x_window,   // (K, N)
    const Eigen::Ref<const Eigen::VectorXd>& m2_window,  // (K,)
    const Eigen::Ref<const Eigen::MatrixXd>& z,          // (K, k)
    const std::vector<int>& protected_positions,         // kept as singleton cells: the spike,
                                                         // the support of every extra column
    const Eigen::Ref<const Eigen::VectorXd>& centre,     // (N,) the row's centre
    const EllipsoidFrame& window_frame,                  // the window's (mu, L, L_inv)
    double eps );

}
```

Algorithm, all in the whitened coordinates `u_j = pullback(window_frame, x_j)`:

1. **Singletons that must survive.** Every protected position (the spike, §1
   item 2, and the support of any extra column, §4 item 8) and the
   point farthest from the centre in *physical* distance — `window_radius`
   (`init_dictionary.hpp:313-328`) is the admissibility bound and the top
   ladder rung (`probe_fit.hpp:563`, `:605`), and a centroid lies strictly
   inside its cell's hull, so without this the guard silently tightens.
   Ties on the farthest point break on the smallest window position.
2. **The tree.** A 2^N-tree on the whitened bounding box of the window
   (midpoint splits on every axis; a quadtree in 2D, an octree in 3D). Split
   a node while it holds more than one point *and* its diagonal exceeds
   `eps` times the distance from the centre to the node's nearest corner
   (zero if the node contains the centre). Nodes containing a protected
   singleton are split until that point is alone. Every leaf is a cell.
   Deterministic by construction; the same idiom as the halo's
   `ellipsoid_tree::tree_cut` (`box_forest.hpp`), which is already a
   dependency, with a graded key instead of "largest box first".
3. **Aggregation and order.** Emit cells sorted by the smallest window
   position they contain; inside a cell accumulate in ascending position.
   Under `dist_fit`'s ascending-gid merge (`mpi/dist_fit.hpp:129-165`) the
   window array is rank-independent, so a coarsening that is a pure
   function of that array in that order is rank-independent too — the
   G-L2 gate's own mechanism, reused (§4.1).
4. **Diagnostics.** `cell_of` for provenance and tests; the count `Kc` is
   the work.

Points of order the report pinned down: no consumer of the batch assumes the
points are distinct, on a mesh, or ordered beyond row-for-row alignment of
`x`, `m2`, `z` and the spike position (`probe_fit.hpp:506-526`,
`varpro.hpp:592-614`, `lg_ellipsoid_feature.hpp:47-55`); masses must be
strictly positive (`whitening.hpp:61-76`, `:105-109`), which summed masses
are; and `local_spacing` (`init_dictionary.hpp:283-308`) throws below two
points, so the coarse set must have at least two — it always will, since the
spike and the farthest point are distinct singletons unless the window has
one point, which `operator_fit.hpp:555-560` already rejects.

---

## 3. The single call site, and what stays on the full window

Insert at `operator_fit.hpp:603`, after the per-row arrays are gathered
(`:588-602`) and before the whitening (`:605`). Everything from `:605` to
`:671` then reads the coarse arrays; nothing above line 600 and nothing after
line 673 changes.

```
:588-602  x_window / m2_window / z / y / target_mass / num_extra   unchanged
:603      cw = (K > n_max) ? coarsen_window(.., {spike_position}, ..) : identity   NEW
:605      whiten_probes(cw.z, cw.m2)                                  coarse
:607-610  extra = Zero(Kc, num_extra); extra(cw.protected_cells[0]) = 1   coarse
:613      whiten_extra(extra, target_mass, cw.m2)                     coarse
:628-629  WhitenedBasis(cw.x, target_mass, cw.m2, modes, center, ..)  coarse
:668-670  fit_from_probes(cw.x, cw.m2, cw.z, y, center, cw.protected_cells[0], ..) coarse
```

`y` and `target_mass` are not window-indexed at all (`:600-601`), and
`fit_operator` always passes `target_mass` explicitly (`:670`), so the
inference from `m2_diag[spike]` in `fit_from_probes` (`probe_fit.hpp:534-537`)
is dead on this path; the singleton spike keeps it harmless for direct and
Python callers too.

The window frame is needed inside the parallel loop: keep a per-row
`std::vector<ellipsoid_tree::Ellipsoid>` next to `prior` (`:458`), filled in
the same serial pre-pass that records the region (`:494-514`), rather than
un-flattening `window_covariance`.

| consumer | full window or coarse cells |
|---|---|
| `outcome.window` and the CSR `window_indptr` / `window_indices` (`:554`, `:777-781`) | **full** — the deployed support; never written by the coarsening |
| `window_center` / `window_covariance` (`:507-514`) | **full** — written before any fit |
| `spike_position` binary search (`:562-586`) | on the full window, then remapped by `coarsen_window` |
| `window.size() < 2` (`:555`) | full; add a coarse-side guard with its own message so a bad `eps` fails attributably (both land in the per-row catch at `:719-724` → `RowStatus::Failed`) |
| whitening, baseline, `linear_cv_score`, `detail::linear_fit`, `fit_from_probes` | coarse |
| baseline and search counting rules (`:622-627`, `probe_fit.hpp:917-925`) | probes only; unaffected |
| `window_radius` / admissibility (`probe_fit.hpp:563`, `:704-711`) | preserved by the farthest-point singleton |
| `circle_ladder` bottom rung via `local_spacing` (`probe_fit.hpp:605`) | preserved because near-centre cells are singletons under the grading; assert it in a test |
| `assemble_sparse`, `matvec`, `eval_entries`, `eval_kernel` (`lg_operator.hpp:637-746`, `:525-567`, `:466-511`, `:400-427`) | **full** — they read `row_window`; untouched |
| `DistFitResult::window_candidates` (`mpi/dist_fit.hpp:191-195`) | full; add `fit_points_total` beside it |

**There is no separate assembly support.** The deployed row is the fitted
kernel's `tau`-ellipsoid intersected with the full window
(`lg_operator.hpp:700-704`), so coarsening the fit leaves deployment on the
fine quadrature by construction. Two things follow. First, the honest-
deployment prose ("fitted object == deployed object", `operator_fit.hpp:
35-41`, `lg_operator.hpp:30-48`, `docs/defaults.md:124-135`) becomes "the
deployed support is the fit window; the fit's quadrature on it may be
coarsened, graded so that its error is controlled", and the grading is what
makes that defensible: near the centre the two agree exactly, far out the
kernel is small. Second, and the reason for the re-score below: the CV score
that decides the ladder and the guard is a coarse-quadrature score.

**Re-score the finalists on the full window.** After the search, one
`linear_cv_score` for the winner and one for the baseline on the full
`x_window` / `m2_window` / `z` (O(K m) each, negligible against a fit) gives
`diagnostics.score` and `baseline_score` their literal meaning back and lets
the guard's decision (`:673`) be taken on the honest numbers; the coarse
scores stay the search's internal currency. Recommended; it costs two basis
evaluations per coarsened row.

---

## 4. Invariants, and how each survives

1. **Bit-identity across thread counts, runs and rank counts**
   (`operator_fit.hpp:216-225`, `probe_fit.hpp:52-56`, `docs/reproducibility.md`,
   `mpi/dist_fit.hpp:15-19`; tested at `tests/test_operator_fit.cpp:240-278`,
   `tests/test_probe_fit.cpp:491-517`, `bindings/tests/test_lgpsf_py.py:773`,
   `tests/mpi/test_dist_fit_mpi.cpp`). Preserved iff `coarsen_window` is a
   pure function of the window array with no threads, a canonical split, a
   canonical cell order and ascending in-cell accumulation (§2, item 3). Every
   one of those tests is a live tripwire; none has golden numbers to update
   (`tests/test_operator_fit.cpp:6-8`).
2. **The baseline guard** (`operator_fit.hpp:29-33`, `:673`; `docs/defaults.md:
   161-167`). Both branches read the same coarse arrays, so the comparison
   stays apples to apples; the full-window re-score restores the external
   claim "never worse than the prior on the real data".
3. **Admissibility** (`probe_fit.hpp:30-35`, `:704-711`). The farthest-point
   singleton keeps `window_radius` exact; the centre-displacement bound uses
   the same radius.
4. **Do not gate dead rows** (`operator_fit.hpp:43-64`). Preserved iff the
   coarsening is per row and reads nothing outside the row's window: no shared
   tree across rows, no field-adaptive `n_max`. A dead row's coarse design is
   well formed (`z` is not zero, only `y` is), the inner solve returns zeros at
   score 0, the guard ties, the baseline ships, as now.
5. **Deployed support == fit window** (`tests/test_operator_fit.cpp:669-694`,
   `:831-898`). The CSR arrays are never written by the coarsening; add a test
   that they are identical with coarsening on and off.
6. **The counting rules** depend on probes and parameters only; unaffected.
7. **Extrapolation safety inside the window.** The pathology that forced
   "deployment ⊂ window" (fits chasing noise on near-zero rows, with
   polynomial-times-Gaussian modes taking large values where nothing
   constrained them) has a mechanism that coarsening can recreate *inside*
   the window: a basis function that oscillates within a cell is aliased by
   the quadrature, its coarse column is small or wrong, its coefficient can
   inflate, and the fine evaluation at deployment sees the full oscillation.
   Every deployed point lies in a cell the fit saw, so the "outside" case
   cannot recur; the "aliased" case is excluded by resolution: at distance
   `r` the cell is `eps * r`, a mode of radial degree `p` and angular order
   `ell` centred at the node has its finest structure at scale
   `sigma / (p + ell)` near `r ~ sigma` and is exponentially small beyond,
   and near the centre the cells are singletons, so the uniform condition is
   `eps * (p + ell)_max <~ 0.5` — at the ladder's top level of 5, `eps <= 0.1`,
   which the error analysis of §1 asks for anyway. The gap is a *released*
   centre displaced by `d`: a kernel there sits on cells of size `eps * d`
   and a needle with `sigma < eps * d` is under-resolved. Two guards close
   it: the full-window re-score of the finalists (§3) is a consistency
   check — an aliasing artifact scores well on the coarse quadrature and
   badly on the full window, and the guard taken on the full-window scores
   rejects it — and, structurally, a resolution rule in admissibility for
   released candidates, `min axis >= eps * ||mu - mu0|| * (p + ell)_max`.
   Dead rows stay safe by themselves: zero data give zero coefficients
   regardless of the design. A test should construct the failure (a needle
   candidate at a displaced centre on a coarsened window) and see it
   rejected.
8. **General extra bases.** Extra columns are not quadrature objects:
   `e_hat = sqrt(m_rho) M2^{-1/2} E` makes the design column the plain sum
   `sum_j z_{jl} E_{ji}`, without masses, so aggregation (which preserves
   `sum_j m_j z_j`) does not coarsen a general `E` consistently. The spike
   works because its support is one point kept as a singleton, and that
   generalizes with no special case: `coarsen_window` takes the *protected
   positions* and keeps each a singleton; `fit_operator` passes the spike, a
   direct caller with extras passes the union of their supports. Sparse
   extras are then exact, bit for bit, like the spike. If the protected set
   exceeds the trigger `coarsen_above`, the row is not coarsened and
   `fit_points` says so — a graceful guard, not a refusal.
9. **`window_shape` / `window_shape_ladder`** (opt-in) lose the within-cell
   second moments under aggregation (parallel-axis theorem) and read smaller.
   One-line caveat in the header's doc comment; not a default-path issue.

---

## 5. Configuration, reporting, documentation

- `OperatorFitConfig` (`operator_fit.hpp:180-226`) gains `int coarsen_above =
  0` (windows with more points than this are coarsened; 0 = off) and `double
  coarsen_eps = 0.1`. Off by default, so the change is a strict no-op for
  every existing test and for `docs/reproducibility.md`'s guarantees as
  stated. Two knobs rather than one: the trigger says when the bound applies,
  `eps` says how fine the quadrature is, and neither can be derived from the
  other without an iteration that would itself need a tolerance. The
  defaults decision for turning it on lives in §7.
- `FitDiagnostics` (`:234-244`) gains per-row `fit_points` (the coarse count;
  equals the window size when not coarsened). `DistFitResult` (`mpi/dist_fit.
  hpp:87-94`) gains `fit_points_total` beside `window_candidates`. Without
  these there is no way to see whether the bound is binding.
- Bindings: `OperatorFitConfig` fields, the diagnostics field, and
  `coarsen_window` itself (`bindings/lgpsf_bindings.cpp`, the config block and
  `:922-951`), so the prototype and the library can be compared from Python.
- Docs: rows in `docs/defaults.md` (`OperatorFitConfig` table `:7-16`; the
  "deployed support" paragraph `:124-135`), the header table and ladder in
  `dev/architecture.md`, the umbrella `include/lgpsf/lgpsf.hpp` if public,
  `CHANGELOG.md` `[Unreleased]`, and this plan's status line in `HANDOFF.md`.
- House rules apply as usual: `#pragma once` + SPDX + preamble, `///` docs,
  `"lgpsf::coarsen_window: lowercase message"` errors validated eagerly with
  NaN-safe forms, points as rows, one test file per header, `-j3` builds,
  `tools/check_dependencies.py` before committing.

---

## 6. Validation plan

**S1 — Python prototype, zero library changes. DONE 2026-09-06
(`experiments/window_coarsening.py`, `experiments/window-coarsening.md`).**
Sixty real rows of the heat problem, windows of 246 to 99,856 points, k = 60,
ladder 0..5. Findings: (i) `eps = 0.1` keeps every guard decision, 97% of
ladder levels, the axes within 1e-4 decades and 0.25 degrees, `c` within 0.02%,
for 9x less wall overall and 11-54x on the 1e5-point windows (23-30 s ->
0.5-1.4 s); `0.15` loses one fit to an early ladder stop; `>= 0.2` loses
10-17% of the fits. (ii) The quadrature error is first order for white-noise
probes (above), second for smooth ones. (iii) The coarse CV score is
optimistic: at `eps >= 0.2` it claims wins over the baseline that the
full-window score denies, never the reverse; none at `eps <= 0.15`. The
full-window re-score (decision 2) is load-bearing. (iv) A singleton core of
about `7 pi / eps^2` points sits under the count law (nothing merges within
`h / eps` of the centre), so `eps = 0.05` buys nothing below ~1e4 points and
there is no gain below ~2,000 points: `coarsen_above` belongs in the low
thousands. (v) A cell costs what a point costs (5.7e-6 vs 6.0e-6 s per
candidate), so the saving is exactly the cell ratio. (vi) The motivating case
(a prior too wide) is the easy one: the kernel then lies in the singleton core
and is fitted exactly; the rows that feel `eps` are honest-prior rows on fine
meshes. Defaults from this: `coarsen_eps = 0.1`, `coarsen_above` a few
thousand when it is turned on.

Original brief: `fit_from_probes` is fully
bound (`bindings/lgpsf_bindings.cpp:795-815`), and `LGOperator.row_window`,
`window_indices`, `x_cols`, `m2_diag` are readable (`:869-921`), so a real
row's window can be pulled from a completed `fit_operator`, coarsened in
numpy, and `fit_from_probes(full)` compared with `fit_from_probes(coarse)` on
score, `theta`, `c`, `s` and wall time. Deliverables: an `eps` sweep on a few
hundred rows of the maintainer's sub-mesh problem (window sizes 100-1,000)
and on synthetic windows scaled up to 100,000 points; the "`eps -> 0`
reproduces the full fit" check; the cell-count law against `3 pi / eps^2 *
log2(R/h)`. Templates: `examples/reading_a_row_fit.py`, `examples/fit_one_psf.
py`. This decides `eps`'s default and whether the second-order argument holds
for the ladder's top modes.

**S2 — the header and its tests** (`tests/test_coarsen_window.cpp`):
mass conservation `sum m_C == sum m2` and `sum m_C z_C == sum m2 z` to
round-off; the spike and the farthest point are singletons; every cell obeys
the grading rule; the count bound; identity when `eps -> 0`; purity (two calls
agree bit for bit); a permutation of the input that preserves the sorted
order changes nothing.

**S3 — the call site, config, diagnostics, docs.** Existing suite passes
unmodified (coarsening off). New cases in `test_operator_fit.cpp`: with
coarsening on, bit-identity across thread counts; the guard's strict `<` /
exact `==` property; CSR windows identical on and off; the re-scored
`score` / `baseline_score` are full-window numbers; a row whose coarse set
degenerates fails alone with the new message.

**S4 — bindings and the Python tests** (`bindings/tests/test_lgpsf_py.py`):
the config fields round-trip, the diagnostics field is exposed,
`coarsen_window` agrees with the S1 numpy version on the same window.

**S5 — the MPI gate.** `tests/mpi/test_dist_fit_mpi.cpp` with coarsening on
at `-n 1/2/4` (it is manual; see its header for the build line). This is the
rank-independence claim of §2, item 3, made concrete.

**S6 — field validation, then the default.** On the maintainer's problems,
with the same probes: (a) the sub-mesh, where windows are small and the
result must be unchanged in operator quality (the held-out QC map and the
fitted ellipsoids, both readable today) with a mild change in wall time;
(b) the continental run whose bad rank motivated this, where the fit wall on
that rank should fall from thousands of seconds to the order of its row
count times a bounded per-row cost, with the fitted operator's QC and the
Newton solve counts unchanged. Then decide whether `coarsen_above` gets a
non-zero default (a few thousand) or stays opt-in.

**S7 — the user-facing note** `docs/window-coarsening-notes.tex` / `.pdf`,
listed in `docs/README.md`'s table next to `varpro-whitening-notes`: the
quadrature view of the fit (§1), the aggregation rule and what it keeps
exact, the graded cell rule and its error bound in terms of the smoothness of
the Laguerre-Gaussian modes (the midpoint-rule error `(eps (p + ell))^2` at a
mode's own radius, the Gaussian envelope beyond it, the singleton near field),
the resolution condition `eps (p + ell)_max <~ 0.5`, the count law, the
window-frame grading and what the aspect cap means for it, the protected
positions (spike, extra bases), the full-window re-score, and the resulting
guarantee ("the deployed support is the fit window; the fit's quadrature on it
is coarsened with controlled error"). Formulas and the few numbers from S1/S6,
not narrative.

**S8 — a narrated example** `examples/coarsened_fit.py` (a page generated by
`docs/generate_examples.py`, so it is source-checked in CI): fit ONE function
with `fit_from_probes` on its full window and on the same window coarsened
with `coarsen_window` at two or three `eps` values, print the two fits'
ellipsoids, coefficients, full-window cross-validation scores, cell counts and
wall times side by side, and draw the kernel with the cells overlaid. It shows
the user how to call the feature and that it is accurate, on a single target
rather than a whole operator. A C++ twin is optional.

Cost expectation from the measured rate (about 1.4e-5 s per window point per
row at k = 15 when the prior is sane; several times that when the search runs
its full grid): a 90,000-point row at `eps = 0.15` costs what a 3,000-point
row costs today.

---

## 7. Decisions (maintainer, 2026-09-06)

1. **Default off.** `coarsen_above = 0`; a strict no-op until S6 says
   otherwise.
2. **Full-window re-score of the finalists**, and the guard is taken on the
   full-window scores. It is also the extrapolation-safety check of §4.7.
3. **Grading in the window frame** (§1); Euclidean falls out for ball windows.
4. **`coarsen_window` is public API**: bound, documented, in the umbrella
   header, with protected positions as the generalization of the spike
   (§4.8).

Also settled in discussion: the released-centre resolution rule (§4.7) ships
with the coarsening rather than later, and the S2 tests include the needle
construction.

## 8. Explicitly out of scope

- **Bounding the deployment cost.** `assemble_sparse`, `matvec` and the
  entry evaluators are O(|tau-support ∩ window|) per row and a row that ships
  its (wide) baseline assembles wide. This plan bounds the fit only; a
  deployment bound would be a separate change with its own trade-off (a
  truncated support changes the operator, a coarsened fit does not).
- **Rebalancing rows across ranks.** Even a perfect prior leaves clusters of
  legitimately wide windows on fine meshes; if S6 shows the fit is still
  wall-time significant after this bound and the caller's prior repair, that
  is the next lever, and it lives in `dist_fit`.
- **Adaptive `eps`** (e.g. from the prior's aspect or the ladder level).
  One fixed grading is the point; everything adaptive reintroduces a width
  estimate.
