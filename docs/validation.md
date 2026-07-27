# Field-scale validation: what "PIG" means

The headers and design notes cite **PIG** as the evidence behind several
defaults and warnings. This page says what that is, so the citations resolve.

## The problem

**Pine Island Glacier**, a marine-terminating outlet glacier of the West
Antarctic Ice Sheet. The operator is the **Gauss-Newton Hessian of a
basal-friction inverse problem**: infer the friction coefficient at the
ice-bed interface from observed surface velocities, with the ice-flow forward
model a nonlinear Stokes system.

That Hessian is what this library exists to approximate. It is:

- **dense** -- every dof influences every other through the elliptic solve;
- **expensive** -- one matvec costs a linearized forward and adjoint solve,
  which is why the method's entire budget is counted in matvecs;
- **not a convolution** -- the point-spread function's width, orientation and
  shape all vary across the domain, which is why a fitted per-row ellipsoid is
  the right object and a single stationary kernel is not.

The discretization is a 2-D triangulated mesh with **6557 nodes**, and the
lumped mass matrix supplies the `M1`/`M2` of the `H = M1 Phi M2` row model.

Two basal-friction states are used, because they stress the method very
differently:

| | state | why it matters |
|---|---|---|
| **smooth beta** | the friction field is smooth | the a-priori ellipsoid field is close to right |
| **rough beta** | the alpha = 0.01 MAP estimate | the pointwise a-priori sigma is **3-5x too wide at the ice-stream channel**, so the prior is badly wrong exactly where the physics is interesting |

## What was measured

Whole-operator fits of all 6557 rows, scored against **held-out probe pairs**
the fit never saw. Headline numbers, window-clipped and symmetrized:

| | k=20 | k=40 | k=100 |
|---|---|---|---|
| smooth beta | 0.0603 | 0.0269 | **0.0147** |
| rough beta | 0.1998 | -- | 0.0561 |

For scale: the prior reference method (`psfladder`, a frozen-theta ladder)
reaches 0.0238 at k=100 on the smooth state and 0.106 on the rough one, so
lgpsf is roughly 1.5-1.9x better at every probe budget, and better at rough
beta -- the state frozen-theta struggled with most -- than at smooth.

Column-level forensics at the eight worst channel nodes land at 0.079-0.136
against an idealized full-column VarPro reference of 0.16-0.22.

## What it changed in the library

These are the findings the header comments point at:

- **Deployed support == fit window.** Windowed cross-validation is blind to
  out-of-window model energy, and polynomial-times-Gaussian modes extrapolate
  violently past the data. One rogue row once carried **94% of a whole-operator
  test error**. Every evaluation is now window-restricted, with
  `eval_kernel_unrestricted` as the named opt-out.
- **Released mu is off by default.** Releasing the center shipped on ~91% of
  rows while buying nothing on held-out data, guarded only by a far-too-loose
  window-radius bound.
- **Dead rows need no gate.** A row whose response is identically zero costs
  one candidate and ships zero coefficients; gating them saves no measurable
  time and changes no other row's fit by a single bit.
- **The baseline guard is always on**, and it earns its keep: at the largest
  mode budgets it ships the a-priori model on ~18% of live rows.
- **`WedgeLadder(10, 2)`** is the strongest fixed mode-growth policy at
  k >= 40; complete shells win at k = 20.
- **The inner solve is QR-first.** At 21 modes the SVD was 61% of the whole
  fit; replacing it where the Kaufman Jacobian allows was worth 2.1x on a
  whole-field fit with every field number unchanged to four digits.

## What is and is not in this repository

**The PIG data is not public and this library does not depend on it.** No
build, test, example or acceptance criterion here reads it; the in-repo test
suites are synthetic by design, and the end-to-end example
(`examples/operator_fit_frog.py`) uses a self-contained analytic operator.

lgpsf is the general-purpose library; the glaciology work is a downstream
consumer of it. The dependency runs one way, and
`tools/check_dependencies.py` enforces that mechanically.

The detailed run records -- per-configuration tables, timings, forensics --
live in the maintainer's private notes rather than here. What is public is the
summary above and the design consequences, which is what a reader of the code
needs in order to understand why the defaults are what they are.
