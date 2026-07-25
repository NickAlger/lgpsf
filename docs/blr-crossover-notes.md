# Block low rank as an intermediate format: crossover analysis and plan

**Status: DECISION RECORD (Nick + session 2026-07-24). No BLR work
now; the decision point is a measurable scaling crossover, with the
diagnostic and the build route (NOT ACA+) specified below.**

## The pipeline this serves

The ultimate goal is a global low-rank (GLR) approximation of the
prior-preconditioned misfit Hessian for the continental Antarctic ice
sheet on a relatively fine mesh. Conventional route: randomized
SVD/Lanczos directly on the Hessian, needing O(rank) matvecs -- rank
5000-10000+, each matvec an expensive PDE (Stokes) solve. This
project's route inserts the PSF fit so the expensive solves happen
only ~20-100 times:

    Hessian matvecs (few, expensive)
      -> PSF representation (operator_fit.OperatorFit -- parametric,
         ~(P + m + 1) floats/row, discretization-independent)
      -> intermediate matrix format (fast matvec, no PDE solves)
      -> global low rank via matvec-driven RSVD/Lanczos.

The intermediate format's job description: fast to BUILD from the
parametric fit, fast to APPLY, at scale (full Antarctica, fine mesh,
distributed). Current prototyping scale: Pine Island Glacier, coarse
mesh, one node.

## The two candidate decompressions, per row

- **Sparse (assemble_sparse(tau))**: kappa entries per row, kappa =
  points per tau-ellipsoid ~ pi tau^2 sigma_1 sigma_2 / h^2 in 2D --
  GROWS LIKE h^-2 under refinement. Entries exact at the support,
  build embarrassingly parallel (closed-form LG evals, O(m) flops
  each), matvec O(N kappa).
- **Block low rank (BLR, ellipsoid_psf's source-partitioned format)**:
  ~r (1 + c) stored entries per source, where r is the block rank and
  c = |T|/|S| the target-halo ratio (~2-4 for ellipsoid-sized blocks).
  r is set by the kernel's PHYSICAL variation across the block
  (mode count x effective number of distinct thetas), so it is
  MESH-INDEPENDENT: fixed physical blocks, refining h grows |S| but
  not r.

**Crossover: BLR wins when kappa >> r (1 + c).** Both sides are
measurable from a fitted OperatorFit:
kappa from the mesh and the fitted ellipsoids; r from per-block SVDs
computed on the coarse mesh (cheap -- entries are closed-form), then
extrapolated (r stays, kappa scales as (h_coarse/h_fine)^2).

- Coarse PIG scale: kappa ~ a few hundred, r(1+c) comparable -- no
  practical win, matching the ellipsoid_psf experimental experience
  (frog_compression example, 1681x1681, 256-source blocks: median
  block rank 83 at rtol 1e-2 even under OPTIMAL per-block SVD;
  storage saturates at the block-sparsity floor by rtol 1e-3). The
  observed "ACA+ exhausts most of the block" is mostly not an ACA
  failure: the blocks are genuinely not very low rank at that scale,
  because the theta field rotates across them.
- Antarctica scale (h ~ 0.5-1 km, sigma ~ 5-20 km): kappa ~ 1e3-1e4
  while r stays put -- an expected ~10-30x on storage and matvec,
  flat under further refinement instead of h^-2.

## The headroom bound (why sparse carries the prototype)

At R ~ 5-10k and N ~ 1e7, the GLR builder's OWN dense algebra --
O(N R^2) orthogonalization ~ 1e15 flops, N R ~ 1e11-entry factors --
is comparable to or above the total sparse-matvec cost (~1e14). The
pipeline's 1000x win is killing the Stokes solves, and the PSF fit
already banks that; the intermediate format only needs to not be the
bottleneck. This bounds what BLR can buy and says sparse likely
suffices through the first at-scale runs.

## If/when BLR is built: the route (never ACA+)

ACA exists because ellipsoid_psf's evaluator entries are expensive
(RBF interpolation over transported impulses), so few may be sampled;
its practical pains (pivot-chain fragility, disconnected-support
restarts, cross-compiler nondeterminism) are the price of that
constraint. lgpsf's entries are closed-form LG evaluations at O(m)
flops, so the constraint -- and the price -- vanish:

1. **Per-block dense/randomized SVD on closed-form entries**:
   deterministic, parallel, robust. Build cost O(sum |S||T|) cheap
   local flops (h-dependent sweep but no communication, no solves).
2. **Better -- shared-basis re-expansion (the analytic route)**:
   partition ROWS; re-expand each row's LG kernel into one shared
   per-block LG basis (Gaussian-times-polynomial against
   Gaussian-times-polynomial -- FMM-style M2L-like re-expansion; the
   center-offset absorption is exactly the enrichment/dipole math in
   docs/robust-init-notes.md). Block factors come out analytically:
   row factor = re-expanded coefficients, column factor = shared
   modes at the halo points. Build O(N m r), NO h-dependent entry
   sweep, and "r is bounded" becomes a truncation-error statement
   instead of an empirical hope.

**Composition, not duplication**: ellipsoid_psf::BlockLowRank is
deliberately pure linear algebra (ids + factors, no geometry, no
evaluator reference) with the distributed design already worked out
(deal blocks to ranks + two scatters). The eventual work item is a
CONVERTER -- OperatorFit -> blocks -> ellipsoid_psf.BlockLowRank --
reusing its apply/applyT, its RSVD-from-applies, and its MPI story.
lgpsf contributes only the new piece (parametric block
factorization); each package stays independently useful.

A property worth noting: the per-block-relative-tolerance locality
advantage BLR demonstrated over equal-storage GLR (frog_compression's
boundary-ring figure) is already native to lgpsf -- every row is fit
and CV-scored relative to its own scale.

## Plan of record

1. Close the pipeline end-to-end at PIG scale with what exists:
   fit_operator -> assemble_sparse -> RSVD -> compare against the
   true Hessian's spectrum/GLR. This tests the ACCURACY risk (does
   PSF -> GLR survive the two-stage approximation?), which no format
   work can de-risk.
2. From the same fit, compute the crossover table: per-row kappa,
   per-block r at the target rtol, r(1+c) vs kappa extrapolated to
   Antarctica h. Data-driven BLR go/no-go.
3. Only if (2) says go: the converter to ellipsoid_psf.BlockLowRank,
   via shared-basis re-expansion (preferred) or per-block RSVD.
