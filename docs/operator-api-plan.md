# Operator-level fitting API -- agreed design (2026-07-24)

**Status: IMPLEMENTED (session 2026-07-24) as
`archive/python-prototype/operator_fit.py`** (`fit_operator` -> `OperatorFit` + the
helper table below), tests in `archive/python-prototype/test_operator_fit.py`. This
doc remains the design record; implementation-level decisions made
during the port (free-mu theta storage, the `target_mass` kwarg on
`fit_from_probes`, baseline-guard fold sharing) are logged in
`docs/design-notes.md`.

## The fitted object (the central decision)

The operator fit does NOT produce a matrix. It produces

    H~  =  M1 Phi~ M2  +  M1 S

a sum of TWO OBJECTS OF DIFFERENT TYPES -- the same distinction the
whitening derivation is built on (X-valued quadrature objects vs
X'-valued discrete corrections):

- **Phi~, a semi-discrete continuum kernel**: for each fitted row rho, a
  genuine function of the source coordinate,
  Phi~(rho, x) = sum_i c_i(rho) phi_i(x; theta_rho). Rectangular by
  nature: evaluable between arbitrary point sets, interpolable,
  exportable to ellipsoid_tree, meaningful on other meshes.
- **S, a sparse dof-indexed discrete correction**: today diag(s_rho);
  extensible to e.g. ring corrections but always explicitly sparse and
  mesh-tied. Square by nature: it exists only in the dof world, because
  it was DEFINED as the part of the PSF the mesh cannot resolve.

Every helper is typed to the component(s) it touches. The spike has no
off-grid meaning by construction; the API exposes its Dirac mass,
`spike_measure(rho) = m_rho * s_rho`, plus documentation -- a consumer
who must regularize it (delta on a finer grid, narrow Gaussian of width
h, drop it) makes that modeling decision themselves, visibly. On a
genuinely finer target mesh the honest answer is "refit".

## Spike coefficient convention (settled; already implemented)

`s` is ADDITIVE on top of the UNMODIFIED LG functions at the diagonal:

    H~[rho, rho] = m_rho^2 sum_i c_i phi_i(x_rho; theta) + m_rho s_rho.

This is what probe_fit already produces and what the recovery test pins
(`_make_row` constructs truth as smooth-including-diagonal plus additive
spike and asserts s == s_true). The FWL projection inside the fit is an
algorithm, not a semantics change: the returned (c, s) equal the joint
least squares against the unmodified basis (tested).

Span-invariance fact: the with-diagonal and excluded-diagonal
parameterizations span the same probe-space column space, so
predictions, residuals, and the whole theta landscape are IDENTICAL
under either convention; c is the same; only s shifts, by the smooth
diagonal contribution:

    s_total = s_additive + m_rho sum_i c_i phi_i(x_rho; theta).

(This is also why the lgpsf slice-37 refits reproduced the research
repo's slice-36 numbers digit-for-digit despite different diagonal
handling.) The additive convention is the physically right standard:
s -> 0 under mesh refinement (a field map of s is a resolution
diagnostic -- where the mesh is starving), and m_rho s is the honest
Dirac mass of what the continuum kernel could not carry.

## Inputs

- `x_cols` (N, K_all) column-dof coordinates; optionally distinct
  row-dof coordinates (keep rows != cols and M1 != M2 in the signature
  even though the motivating problem is square -- the smooth component
  is rectangular by nature and it costs nothing).
- `m1_diag`, `m2_diag`: lumped masses.
- `V` (K_all, k) raw random probes, `HV` (R_all, k) raw responses.
- `mu0` field (R, N), default = the row dofs' own coordinates (an
  a-priori mean estimate may differ, e.g. advected kernels).
- `sigma` field (R, N, N): the user's BEST GUESS at the actual bump
  shape -- NOT required to be conservative. Three consumers:
  (1) the sigma0 candidate rung, (2) the baseline linear fit (below),
  (3) the window, made conservative by inflation: window = kd-ball of
  radius tau_window * (largest 1-sigma axis of sigma[rho]).
- `tau_window` (default ~10, user-overridable). NOTE THE DEPENDENCY:
  the window-containment admissibility guard is only as valid as
  tau_window is conservative -- with default 10, a best-guess sigma
  underestimating by 3x still leaves ~3 true sigmas of margin. This
  justification travels with the parameter.
- `rows` subset / gate mask (dead rows, boundary junk -- e.g. the PIG
  V-gate). Ungated rows get status, not silence.
- `config`: ProbeFitConfig (per-row search) + the operator-level knobs
  above.

Windows are derived INSIDE the operator layer (it owns one kd-tree,
built once); explicit per-row index lists as an override. The row
layer's caller-windows contract is unchanged -- the operator layer IS
that caller.

## Per-row protocol

gate -> window -> fit_from_probes(sigma0=sigma[rho], spike_index where
the row dof appears in the batch) -> BASELINE GUARD -> status.

Baseline guard (always on): a plain linear LG fit at sigma[rho], pinned
at mu0[rho] -- one lstsq, no LM -- is always computed and CV-scored; the
searched fit ships only if it beats the baseline, else the baseline
ships. Consequence: the method is never worse than the a-priori-Gaussian
status quo it replaces, by construction. Per-row status enum:
`fit | gated_out | fallback_baseline | failed`.

## Output: OperatorFit (padded flat arrays -- C++/MPI-friendly)

theta (R, P), mu (R, N), L (R, N, N), c (R, m_max) padded with a
level/mode-set id array (R,) as decoder, s (R,), score (R,),
stop_reason (R,), released (R,), status (R,), masses kept for
convenience, config echo for provenance. Size: ~(P + m + 1) floats per
row -- the parametric form IS the compressed operator; every matrix
format is a decompression of it.

## Helpers (each typed to a component)

**Amendment (2026-07-25, PIG slice-38): deployed support == fit
window.** The dof-context helpers below restrict each row to its fit
window (stored on OperatorFit); eval_kernel stays the raw parametric
component. Rationale in docs/design-notes.md ("Deployed support == fit
window").

| helper | consumes | notes |
|---|---|---|
| eval_kernel(fit, rows, x_query) | smooth | rectangular, arbitrary points |
| eval_entries(fit, rows, cols) | both | square dof context; spike iff col == spike dof |
| matvec(fit, v) / LinearOperator | both | QC vs held-out probes with zero assembly |
| assemble_sparse(fit, tau, symmetrize=...) | both | tau-ellipsoid supports; spike on diagonal |
| ellipsoid_field(fit) -> (mu, Sigma) arrays | smooth (theta) | feeds ellipsoid_tree; lgpsf stays dependency-free |
| qc_map(fit, V_qc, HV_qc) | both | per-row held-out relative residual -- the scorecard |
| spike_measure(fit) | spike | m_rho s_rho, the Dirac mass field |

## Symmetry and SPD are ASSEMBLY POLICIES, not fit properties

Row fits do not produce H~ = H~^T. rows-as-is vs (A + A^T)/2 vs
column-consistent averaging (the ellipsoid_psf "cols vs sym" race), and
clamp/shift for SPD preconditioner factorizations, are consumer
decisions with consumer-specific right answers -- a second, independent
argument for the parametric output.

## Shape of the computation

fit_operator is deliberately boring: gate -> window -> fit_from_probes
per row -> baseline guard -> gather. Rows are independent by
construction (each touches HV[rho, :] and V[window, :] only): the
Python loop is the C++/MPI row-block port reference. Future field-level
ideas (neighbor warm starts, smoothed-theta seeds) enter as CANDIDATE
INJECTION into the per-row stream -- a sweep-order policy, not an API
change.

## Deferred / adjacent

- Scheme-C conservative ellipsoid fields from probes alone
  (docs/probe-moment-ellipsoids.md, parked): the alternative sigma
  source when no physics prior exists.
- The portfolio/init-dictionary background: docs/robust-init-notes.md.
- Level-2 cross-row amortization and QC maps over status/stop_reason/
  s-fields: after fit_operator exists.
