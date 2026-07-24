# Conservative ellipsoid fields from probe data alone (parked)

**Status: PARKED (2026-07-24).** Idea sketch + feasibility analysis for
estimating a *conservative* per-row ellipsoid field directly from stored
random-probe pairs, with no physics prior -- either to replace the
physics-based estimate (`ell = sqrt(4 H eta / beta)` on PIG, which is
wrong exactly where beta is rough) or to serve problems where no physics
estimate exists. No new operator matvecs anywhere below.

## Rejected on evidence

- **Moment probing of H itself** (apply H to `{1, x, x x^T}`; 6 exact
  matvecs, all rows at once): tried hard in the GPSF work and failed --
  the signed integrand cancels on oscillatory rows, and near Pine Island
  Bay (small regularization) approaching half the estimated Sigmas came
  out indefinite. This is a *bias* of the signed weights; exactness
  cannot fix it.
- **Estimating the same signed functionals from stored probes**
  (`H_hat v = Y (Z^T v)/k`): same bias, plus noise that grows with the
  field's global norm; localizing by patches raises a patch-size
  question that is chicken-and-egg with the ellipsoid being estimated.

## The candidate: squared-kernel moments ("Scheme C"), iterated

For iid standard-normal probes, `E[z_j z_j' (z_m^2 - 1)] = 2 d_jm d_j'm`
gives, per row rho and any weight field f:

    T_f(rho) = sum_m H[rho,m]^2 f_m
             = (1/2) E[ y(rho)^2 * sum_m f_m (z_m^2 - 1) ]

so six weight fields `{1, x1, x2, x1^2, x1 x2, x2^2}` yield phi^2-weighted
moments of EVERY row at once, O(N k) per pass, from stored (Z, Y) only.
Ingredients that matter:

- weights `H^2` are positive by construction -- no cancellation bias; any
  indefiniteness in Sigma_hat is finite-k *noise* (shrinks with k,
  clampable), a qualitatively different failure mode from the above;
- mass correction: `H^2` carries `m^2`, so use fields `f/m` to get the
  continuum quadrature right on graded meshes;
- spike removal: subtract `diag(H)^2 f/m` using the split-half Hutchinson
  product `d1 o d2` (two disjoint probe halves; unbiased for `diag^2`);
- Gaussian calibration `Sigma = 2 * Sigma_{phi^2}` (phi^2 of a Gaussian is
  a Gaussian with half the covariance).

**The crux (single-shot is hopeless):** noise in `T_f(rho)` scales as
`||H_rho||^2 ||f||_2 / sqrt(k)`; against the local signal this is
`~ (D/sigma)^2 sqrt(N/k)` for whole-domain second-moment fields --
thousands of percent at PIG-like scales. Two rescues, both zero-matvec
post-processing:

1. **Iterated window shrinkage, conservative by induction.** Start with
   window = whole mesh (trivially conservative: pure noise inflates
   Sigma_hat and drags mu_hat to the domain centroid, i.e. returns "the
   whole mesh"). Shrink each row's window to its current estimate
   inflated x2-3; re-run the localized contraction; iterate ~log(D/sigma)
   passes. The window contains the truth at every step by induction --
   the property the physics prior provided, bootstrapped from data.
2. **Cross-row field smoothing / upper envelope.** Neighboring rows have
   similar kernels (the premise of the whole PSF-interpolation method);
   smoothing the (mu, Sigma) fields across rows averages per-row noise
   down by the neighborhood row count. For conservatism, combine as an
   upper envelope (smallest ellipsoid containing the neighbors') rather
   than a mean.

Guards for the conservative role: deliberate inflation factor (the
physics pipeline also inflates: sigma_bar = 1.3 ell, r = 1.67, tau = 3),
SPD clamp, axis floor at the local mesh spacing. Note the failure
directions are inherently friendly: noise errs toward "too big", the
only acceptable direction here.

**Honest alternative:** once windows are iterated, the plain
|backprojection| moment estimator (`row_fit.row_moments`) can run the
identical shrinking-window loop; its rectification bias (E|noise| > 0)
also errs conservative. Scheme C's edges: unbiasedness, principled spike
subtraction, never forming rows. Which wins is empirical.

## The decisive (free) experiment, when picked up

Run the full iterate-and-smooth loop on the stored PIG probe set
(k = 130) while pretending the physics prior does not exist; score the
resulting ellipsoid field against the smoothed-beta field everywhere and
against the slice-34 measured moments at the ten forensic nodes; count
containment failures (the metric that matters) and indefinite-after-
clamp rows (the metric moment probing failed).
