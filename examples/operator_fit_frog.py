# SPDX-License-Identifier: MIT
"""Fit a whole operator from random matvecs, and watch the error fall with k.

This is the end-to-end example: build a known dense operator, hand `fit_operator`
nothing but random probes and their responses, and compare. It is also the
library's integration test -- everything from the harmonic table to the sparse
assembly runs, on a problem that is entirely self-contained.

The operator is the **rotating frog kernel** (eq. 7.4 of the localpsf paper):
a Gaussian whose covariance ROTATES with position, modulated by a cos-sin
product that makes it mildly negative in places. That rotation is the point --
a stationary convolution cannot represent it, and it is exactly what a fitted
per-row ellipsoid can.

The kernel is anchored at the TARGET, i.e. transposed relative to how it is
usually written, so that a ROW of the operator is a point-spread function.
That is the object `fit_operator` models, so it is also the object worth
plotting; see `frog_row` for why the other orientation goes wrong.

    H[i, j] = m1[i] * m2[j] * phi(x_i, x_j)

on a uniform grid over the unit square, with `m` the lumped mass. The fitter
never sees H; it sees `V` (random probes) and `HV = H @ V`.

Run:

    python examples/operator_fit_frog.py            # figures + convergence table
    python examples/operator_fit_frog.py --quick    # small, no figures

Needs the `lgpsf` bindings importable (build with `-DLGPSF_BUILD_PYTHON=ON`)
and scipy for the sparse assembly. matplotlib is optional -- without it the
convergence table still prints and the figures are skipped.
"""
import argparse

import numpy as np

import lgpsf

# ---------------------------------------------------------------------------
# The frog kernel. Ported from the ellipsoid_psf examples and vectorized over
# the SOURCE, which is the direction an operator row needs: one target point,
# every source at once.
# ---------------------------------------------------------------------------

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


def build_problem(grid=40, seed=0):
    """The grid, the masses, the dense truth, and the prior ellipsoid field."""
    axis = (np.arange(grid) + 0.5) / grid        # cell centers, off the boundary
    mesh = np.meshgrid(axis, axis, indexing="ij")
    x = np.vstack([mesh[0].ravel(), mesh[1].ravel()])       # (2, K)
    count = x.shape[1]
    spacing = 1.0 / grid

    # Lumped mass of a uniform 2-D cell. NOTE: this is h^2, the quadrature
    # weight that makes H = M1 Phi M2 approximate the integral operator. (The
    # example works with any positive diagonal so long as the same one is
    # handed to the fitter, so flip this if your convention differs.)
    mass = np.full(count, spacing**2)

    kernel = np.empty((count, count))
    for i in range(count):
        kernel[i] = frog_row(x[:, i], x)
    H = mass[:, None] * kernel * mass[None, :]

    sigma = np.stack([frog_covariance(x[:, i]) for i in range(count)])
    return dict(x=x, mass=mass, H=H, sigma=sigma, count=count, spacing=spacing,
                seed=seed)


def fit_at(problem, num_probes, tau_window=3.0, max_level=8, seed=None):
    """Fit the operator from `num_probes` random matvecs and assemble it."""
    rng = np.random.default_rng(problem["seed"] if seed is None else seed)
    V = rng.normal(size=(num_probes, problem["count"]))       # (num_probes, K)
    HV = V @ problem["H"].T                                   # (num_probes, R)

    config = lgpsf.OperatorFitConfig()
    config.tau_window = tau_window
    config.spike = False          # the kernel is mesh-resolved; no spike needed
    config.row.mode_policy = lgpsf.ShellLadder(list(range(max_level + 1)))
    config.row.target_score = None

    fit = lgpsf.fit_operator(problem["x"], problem["mass"], problem["mass"],
                             V, HV, problem["sigma"], config=config)
    approx = np.asarray(lgpsf.assemble_sparse(fit.model, np.inf).todense())
    return fit, approx


def relative_frobenius(approx, H):
    return np.linalg.norm(approx - H, "fro") / np.linalg.norm(H, "fro")


# ---------------------------------------------------------------------------
# The counting rule sets a floor on k. With the center pinned in 2-D the row
# fit spends P = 3 parameters, and a mode set of size m is admissible only when
# k >= 2 (m + P). So k = 5 cannot fit even ONE mode, k = 10 affords one, k = 20
# affords six. That is why the small budgets below start at 10 rather than 5.
# ---------------------------------------------------------------------------

# `max_level = 8` above leaves 45 modes reachable, so at the top of this range
# the ladder is not the binding constraint -- the probe budget is, which is what
# the curve is meant to show.
CONVERGENCE_K = [10, 14, 20, 30, 45, 70, 110]
IMPULSE_K = [10, 20, 45]
IMPULSE_TARGETS = [(0.30, 0.30), (0.55, 0.35), (0.70, 0.65)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="small grid, two budgets, no figures")
    parser.add_argument("--grid", type=int, default=40)
    args = parser.parse_args()

    grid = 20 if args.quick else args.grid
    budgets = [10, 45] if args.quick else CONVERGENCE_K

    print(f"building the frog operator on a {grid} x {grid} grid "
          f"({grid**2} dofs) ...", flush=True)
    problem = build_problem(grid=grid)

    print(f"\n{'k':>5}  {'rel. Frobenius':>15}  {'modes/row':>10}  {'rows fit':>9}")
    errors, keep = [], {}
    for k in budgets:
        fit, approx = fit_at(problem, k)
        error = relative_frobenius(approx, problem["H"])
        errors.append(error)
        if k in IMPULSE_K:
            keep[k] = approx          # reused below rather than refitted
        modeled = [rho for rho in range(problem["count"])
                   if fit.model.has_model(rho)]
        modes = np.mean([len(fit.model.row_modes(rho)) for rho in modeled])
        shipped = int((fit.diagnostics.status == int(lgpsf.RowStatus.Fit)).sum())
        print(f"{k:>5}  {error:>15.4f}  {modes:>10.1f}  {shipped:>9}", flush=True)

    assert errors[-1] < errors[0], "more probes should not make it worse"
    if args.quick:
        print("\nquick check passed")
        return

    try:
        import matplotlib
    except ImportError:
        print("\n(matplotlib not installed -- numbers above are the whole "
              "result; install it for the figures)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.loglog(budgets, errors, "o-")
    ax.set_xlabel("number of random matvecs, k")
    ax.set_ylabel(r"$\|\tilde{H} - H\|_F / \|H\|_F$")
    ax.set_title("Frog operator: accuracy against probe budget")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("examples/operator_fit_frog_convergence.png", dpi=130)
    print("\nwrote examples/operator_fit_frog_convergence.png")

    # A ROW of H is the point-spread function at that target -- one model,
    # evaluated over every source. It is what the fitter represents, so it is
    # what the eye should judge.
    fig, axes = plt.subplots(len(IMPULSE_TARGETS), len(IMPULSE_K) + 1,
                             figsize=(3.1 * (len(IMPULSE_K) + 1),
                                      2.9 * len(IMPULSE_TARGETS)))
    for row, target in enumerate(IMPULSE_TARGETS):
        i = int(np.argmin(np.linalg.norm(
            problem["x"] - np.array(target)[:, None], axis=0)))
        truth = problem["H"][i, :].reshape(grid, grid)
        limit = np.abs(truth).max()
        for col, (title, field) in enumerate(
                [("true", truth)]
                + [(f"k = {k}", keep[k][i, :].reshape(grid, grid))
                   for k in IMPULSE_K]):
            ax = axes[row, col]
            ax.imshow(field.T, origin="lower", extent=(0, 1, 0, 1),
                      cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"target {target}")
    fig.suptitle("Point-spread functions (rows of H): truth, and fits at "
                 "three probe budgets")
    fig.tight_layout()
    fig.savefig("examples/operator_fit_frog_impulses.png", dpi=130)
    print("wrote examples/operator_fit_frog_impulses.png")


if __name__ == "__main__":
    main()
