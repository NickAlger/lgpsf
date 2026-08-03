# Plan: the heat-inversion mini-paper

A tutorial mini-paper (`.tex` → committed `.pdf`) that walks the COMPLETE
lgpsf pipeline through one honest inverse problem, start to finish: pose
the PDE inverse problem, look at the Hessian's point-spread functions, fit
them, correct the fit, pick the regularization parameter with the fitted
operator doing the preconditioning, and show what the preconditioner does
to the reconstruction dynamics. Real code snippets throughout — every
snippet is executed code, not prose that resembles code.

Drafted 2026-08-03 from the maintainer's ten-item outline; this document
is the plan to confirm before implementation.

## What it is, and where it lives

- **Audience and register:** a user who wants to apply lgpsf to their own
  inverse problem. Tutorial register — the expository voice, with
  claim-carrying section headings allowed. It complements the four
  corrections examples (which teach the API one lesson at a time) by
  telling the one continuous story those examples deliberately slice up.
- **Location:** `docs/heat-pipeline/` — `heat-pipeline.tex`, `figures/`,
  `make_figures.py` (the single driver that computes everything and writes
  every figure), `snippets/` (extracted below). PDF committed, like the
  other docs notes. Public repo: the problem is fully synthetic and
  self-contained, no private references.
- **The problem instance:** `examples/heat_inversion.py`, reused as a
  module — same kappa field with sharp jumps, same discretization — at a
  finer grid for figure quality (proposal: `grid = 32`, n = 1024; dense
  ground-truth operations are still trivial there, fits take ~1 min).
  New ingredients the examples did not need:
  - a TRUE initial condition with features at two scales (one broad smooth
    blob spanning the slab boundary + several sharp small spots, some in
    the insulating pocket) — chosen so item 10's scale-separation story
    has something to separate;
  - synthetic data `d = A u0_true + noise`, noise level chosen so the
    L-curve corner is visibly interior (proposal: 1–2% relative; tune).

## The sections, item by item

Numbers refer to the maintainer's outline.

1. **The PDE and the inverse problem.** Half a page: the heat equation
   with discontinuous kappa, observe-everywhere-at-final-time data, the
   Tikhonov functional `J(u0) = 1/2 ||A u0 - d||^2 + a/2 |u0|_Hr^2`, the
   normal equations `(Hd + a Hr) u0 = A^T d` with `Hd = A^T A` — and the
   point that one application of `Hd` costs two PDE solves, which is why
   the whole paper is about never forming it. Snippet: the twenty lines
   that define the operator (from `heat_inversion.py`).
2. **The true initial condition** (figure: one panel, the two-scale
   field).
3. **The true conductivity** (figure: kappa on a log color scale; the
   jumps are the story). Items 2–3 can share a row.
4. **Sampled impulse responses** (figure: 4–6 PSFs as small images with
   their locations marked on a kappa inset): slab interior (broad), pocket
   (spike), background (narrow), and ON each interface — the skewed,
   two-sided, visibly non-Gaussian ones. Caption carries the mechanism:
   width ~ sqrt(t kappa), and an interface row floods one way and stalls
   the other. This is the "local but not Gaussian" claim made visual.
5. **One PSF vs its fit, as k grows** (figure: truth + fits at
   k = 10 / 20 / 50 / 150, one row, shared color scale; pick an interface
   row — the hard one). Snippet: `fit_operator` + `assemble_sparse(...,
   Weighted)`. Report the held-out QC score per k alongside.
6. **The L-curve, preconditioned by the fit** (figure: log-log
   ||A u0 - d|| vs |u0|_Hr with the corner marked). The computational
   point: ~15 values of `a`, ONE build — `make_pd` once at the smallest
   `a` in the sweep, then `solve(A_struct, rhs, a)` per point with zero
   refactorization; `classify_shift` zones printed for the sweep, and if
   we deliberately start the sweep below the build shift, `rebuild_at`
   makes a cameo. Snippet: the sweep loop (it is ~8 lines, which IS the
   argument).
7. **Reconstructions at three a's** (figure: too small / L-corner /
   too large; noise-overfit vs right vs oversmoothed). Same solves as
   item 6 — no new computation.
8. **Spectra at the optimal a — a three-figure progression** (maintainer
   design, 2026-08-03). Everything is the preconditioned system at the
   L-corner `a`; sorted spectrum on a log axis, one curve per variant,
   flat-at-1 is the goal the eye should learn to want.
   - **Fig 8a, the k-ladder:** regularization preconditioning first
     (`1 + lambda_i/a`, a huge range — the problem statement), then the
     certified wsym fit at k = 10 / 20 / 50 / 150: the spectrum tightens
     around 1 and compresses further as k grows. One axes, five curves.
   - **Fig 8b, the symmetrization ladder** at one fixed k (proposal:
     k = 50): as-fitted (no symmetrization), `Average`, `Weighted`. For
     the nonsymmetric as-fitted preconditioner the spectrum must be
     SINGULAR VALUES of the one-sided preconditioned operator
     `(B_asis + a Hr)^{-1} (Hd + a Hr)`; for comparability the proposal
     is to plot singular values of the one-sided operator for ALL THREE
     curves in this panel, with a caption note that for the two
     symmetric variants these tell the same story as the eigenvalues in
     the neighboring panels (they agree up to symmetric similarity, not
     entry-for-entry) — TO CONFIRM with the maintainer; the alternative
     is mixed sigma/eig curves in one panel. Policy note: the as-fitted
     variant admits no pencil flip (certification requires symmetry), so
     its curve is the raw LU-solved operator — which is part of the
     lesson.
   - **Fig 8c, the deflation ladder** at the same fixed k: flip-only,
     + free deflation, + value pass (m = 30): deflation tamps the TAILS
     of the spectrum while the bulk stays put.
   - *Empirical risk, flagged honestly:* on this problem the examples
     measured `Average` and `Weighted` near parity in aggregate error
     (the weighted win is on the weak-row tail), so Fig 8b's compression
     may be subtle at n = 1024; if the bulk spectra tie, the honest
     telling is the tail + the per-variant flip counts, and the caption
     says so. The free-deflation curve may also WORSEN the tail at lean
     k (the examples' finding) — also worth showing rather than hiding,
     with the value-pass curve as the resolution.
9. **PCG convergence curves** (figure: relative residual vs iteration,
   prior-preconditioned vs the fitted preconditioners of item 8; the
   spectra and the curves should visibly correspond — close the loop in
   the caption).
10. **PCG iterates as images** (figure, the closer: two rows of
    reconstruction snapshots at matched iterations, prior-preconditioned
    on top — large scales first, then medium, then small, in order — and
    the lgpsf-preconditioned run below, reconstructing all scales
    simultaneously). Caption reads the figure for the reader; the true
    IC's two-scale design exists for exactly this panel.

## Code snippets: extracted, not transcribed

The docgen lesson applies: prose that quotes code drifts. `make_figures.py`
carries `# snippet:name` / `# end` markers; a small extraction step writes
`snippets/name.tex` (verbatim blocks) at figure-build time, and the tex
`\input`s them. The committed PDF is therefore built from the exact code
that made the figures. `make_figures.py` runs with the plain bindings
build (compiler-flag reproducibility, as with docgen).

## Slices (each reviewable)

1. Driver skeleton: problem instance at grid 32, true IC, data, noise;
   figures for items 2–4. (Decides the visual language early.)
2. Fit ladder + item 5; QC table.
3. The build (make_pd + value_pass + cache) and the a-sweep: items 6–7.
4. Spectra + convergence + iterate snapshots: items 8–10.
5. The tex: sections around the figures, snippets wired in, PDF built.
6. Polish pass against the writing-style notes; regenerate everything
   from scratch once, commit.

## Decisions (maintainer, 2026-08-03)

- **Grid 32** to start; revisit upward only if the figures need it.
- **Show the noisy data** as part of item 1 (alongside the true IC).
- **Item 8 is the three-figure progression above** (k-ladder /
  symmetrization ladder / deflation ladder); no two-level curve.
- **Title: "LGPSF Heat Inverse Problem Tutorial".**
- **Closing cost table**: one row per build, columns = number of probes
  used (fit k, plus value-pass applies where applicable) and the
  preconditioned CONDITION NUMBER achieved at the L-corner `a` — the
  same variants as the item-8 curves, so the table and the figures
  cross-reference.

- **Fig 8b convention (decided 2026-08-03): singular values of the
  one-sided preconditioned operator for ALL THREE curves**, with the
  caption note about symmetric similarity. No questions remain — the
  plan is fully specified; implementation starts at slice 1.

- **Spike OFF for this problem (decided 2026-08-03, after slice 2).**
  The smooth LG expansion is the main lgpsf fit; the additive spike is
  a trick for undermeshed regimes, and this problem's narrowest PSFs
  (pocket, sigma ≈ 0.9 cells at grid 32) are still resolvable. Measured
  basis for the call: the spike coefficient is a free L2 basis vector,
  weakly determined at small probe budgets — negative on ~half the rows
  at k = 10 (122 assembled diagonals negative), still negative on ~1/3
  of rows at k = 50, digging anchor-cell holes that row-level L2 QC
  barely notices; aggregate held-out QC ties or improves without it
  (overall 0.26 vs 0.30 at k = 10, 0.070 vs 0.072 at k = 50; only the
  pocket prefers the spike, 0.006 vs 0.012 at k = 50). Possible library
  follow-up, not this paper's work: an optional nonnegativity constraint
  on the spike, or anchoring the diagonal to the Hutchinson estimate
  `E[z_i (Hz)_i]` the probes already contain.
