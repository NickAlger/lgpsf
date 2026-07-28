# SPDX-License-Identifier: MIT
"""The rotating frog kernel: the shared test problem for these examples.

Not a lesson in itself -- a target to fit. Several examples need the same
operator, and restating it in each would bury what each one is actually
teaching. `frog_kernel.hpp` is the same problem in C++, entry for entry, so
the two language paths fit the same thing.

**Why a formula, and not a real operator.** Every example here builds the
dense matrix `H`, which is precisely what the method exists to avoid. That is
deliberate: knowing the kernel in closed form is what lets these examples
report an error against a ground truth, which no real application can do.

lgpsf is for operators that are available MATRIX-FREE -- you can apply them to
a vector, but you cannot inspect entries, because an application runs an
expensive computation rather than a memory read. The Gauss-Newton Hessian of a
PDE-constrained inverse problem is the motivating case: one application costs
a linearized forward solve plus an adjoint solve. If you can afford to form
the dense matrix, you can also afford simpler ways to sparsify it, and should
use them.

So read `H` here as a stand-in for something you could only probe. What the
fitter is given is never the matrix -- it is `V` and `HV`, which is exactly
what a matrix-free operator can supply.

The kernel (eq. 7.4 of the localpsf paper) is a Gaussian whose covariance
ROTATES with position, modulated by a cos-sin product. Two things make it a
fair test. The rotation means a stationary convolution cannot represent it,
while a fitted per-row ellipsoid can. The modulation means each row is
strongly non-Gaussian -- a bright core with a notch through it and side
lobes -- so the LG modes above level 0 have to earn their place.

With `A_MOD = 1` the factor `(1 + A_MOD * modulation)` stays in [0, 2], so
the kernel is non-negative; raise `A_MOD` past 1 if you want a signed target.
The fitted approximation is signed either way, since LG modes are.

    H[i, j] = m1[i] * m2[j] * phi(x_i, x_j)

on a uniform grid over the unit square, with `m` the lumped mass.

**The kernel is anchored at the TARGET**, i.e. transposed relative to how it
is usually written, so that a ROW of the operator is a point-spread function.
That is the object lgpsf models, so it is also the object worth plotting; see
`frog_row` for what goes wrong otherwise.
"""
import numpy as np

SIGMA0_DIAG = np.array([0.01, 0.0025])   # variances along the unrotated axes
A_MOD = 1.0                              # modulation strength


def _angle(x):
    """The local rotation angle at x, of shape (2, ...) -> (...)."""
    return 0.5 * np.pi * (x[0] + x[1])


def _bump(x):
    """Vanishes on the boundary of the unit square, so the kernel is compact."""
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])


def frog_row(target, sources, sigma0_diag=SIGMA0_DIAG):
    """Row `target` of the kernel: its point-spread function, over all sources.

    `target` is (2,), `sources` is (2, K); returns (K,).

    **The shape is anchored at the TARGET**, which is the transpose of how the
    frog kernel is usually written. That is deliberate and it matters: the
    fitter models a ROW as a smooth function of the source coordinate, so
    anchoring here makes the object the method represents and the object you
    plot the same thing. Anchor it at the source instead and each row becomes a
    transversal of many different local shapes -- which the LG expansion cannot
    represent smoothly, and which shows up as speckle.
    """
    sd = np.asarray(sigma0_diag)
    angle = _angle(target)                       # scalar: the local rotation
    cos, sin = np.cos(angle), np.sin(angle)
    d = sources - target[:, None]                # (2, K)
    # p = R(target) @ d, with R = [[cos, -sin], [sin, cos]]
    p0 = cos * d[0] - sin * d[1]
    p1 = sin * d[0] + cos * d[1]

    maha2 = p0**2 / sd[0] + p1**2 / sd[1]
    gaussian = np.exp(-0.5 * maha2) / (2.0 * np.pi * np.sqrt(sd.prod()))
    modulation = (np.cos(p0 / (np.sqrt(sd[0]) / 2.0))
                  * np.sin(p1 / (np.sqrt(sd[1]) / 2.0)))
    return _bump(target) * (1.0 + A_MOD * modulation) * gaussian


def frog_covariance(x, sigma0_diag=SIGMA0_DIAG):
    """The kernel's local covariance at x -- the a-priori ellipsoid field.

    From maha2 = d^T R^T diag(1/sd) R d, so Sigma = R^T diag(sd) R, evaluated
    at the row's own point. This is the honest "best guess a physicist would
    supply": the right shape and orientation, but nothing about the modulation,
    which is what the LG modes have to discover.
    """
    angle = _angle(x)
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([[cos, -sin], [sin, cos]])
    return R.T @ np.diag(np.asarray(sigma0_diag)) @ R


def build_problem(grid=40, seed=0, sigma0_diag=SIGMA0_DIAG):
    """The grid, the masses, the dense truth, and the prior ellipsoid field.

    `sigma0_diag` sets the kernel's anisotropy -- the variances along its
    unrotated axes. The default is 2:1; experiments vary it to ask what the
    fitter's initial-guess ladder is actually buying.
    """
    axis = (np.arange(grid) + 0.5) / grid        # cell centers, off the boundary
    mesh = np.meshgrid(axis, axis, indexing="ij")
    x = np.vstack([mesh[0].ravel(), mesh[1].ravel()])       # (2, K)
    count = x.shape[1]
    spacing = 1.0 / grid

    # Lumped mass of a uniform 2-D cell. NOTE: this is h^2, the quadrature
    # weight that makes H = M1 Phi M2 approximate the integral operator. (The
    # examples work with any positive diagonal so long as the same one is
    # handed to the fitter, so flip this if your convention differs.)
    mass = np.full(count, spacing**2)

    kernel = np.empty((count, count))
    for i in range(count):
        kernel[i] = frog_row(x[:, i], x, sigma0_diag)
    H = mass[:, None] * kernel * mass[None, :]

    sigma = np.stack([frog_covariance(x[:, i], sigma0_diag)
                      for i in range(count)])
    return dict(x=x, mass=mass, H=H, sigma=sigma, count=count, spacing=spacing,
                seed=seed)


def probes(problem, num_probes, seed=None):
    """`num_probes` random probes and their responses -- all the fitter sees."""
    rng = np.random.default_rng(problem["seed"] if seed is None else seed)
    V = rng.normal(size=(num_probes, problem["count"]))       # (num_probes, K)
    return V, V @ problem["H"].T                              # (num_probes, R)
