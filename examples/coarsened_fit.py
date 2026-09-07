# SPDX-License-Identifier: MIT
"""Fit ONE target on its full window, then on the same window coarsened.

A row is fitted on its WINDOW -- every point inside `tau_window` times the
caller's a-priori ellipsoid -- and the fit's cost is linear in how many points
that is. So a prior that is too wide does not only start the search in the
wrong place; it makes the row expensive, because the window's point count
grows with the square of the prior's width and the fit wades through all of
them.

`coarsen_window` bounds that cost without knowing the kernel's width. The fit
never regresses pointwise. Its design matrix is a mass-weighted quadrature of
the model's action on the probes,

    design(l, i) = sqrt(m_rho) * sum_j  m_j * z[l, j] * phi_i(x_j)

so a coarser window is a coarser QUADRATURE, and a quadrature can be graded:
cells of size at most `eps` times their distance from the centre, single
points near it. Each cell becomes its mass-weighted centroid, its summed mass
and the mass-weighted mean of every probe field, which keeps the probe sums
inside it exactly; only the basis function's variation over the cell is
approximated. The spike and the farthest point stay singleton cells, so the
spike's column and the admissibility radius are unchanged, and the cell count
grows like log(R / h) instead of (R / h)^2.

Below, one target is fitted on its full window and on the window coarsened at
three `eps`, and every fit is RE-SCORED on the full window. A coarse fit's own
held-out score is measured on the cells and carries their quadrature error,
so it is not comparable with the full fit's -- both numbers are printed.
`fit_operator` does all of this for you on any row with more than
`coarsen_above` points, re-score included; this is the call it makes, on one
target, with the truth known.

    python examples/coarsened_fit.py
    python examples/coarsened_fit.py --outdir /tmp    # figure elsewhere
"""
import argparse
import os
import time

import numpy as np

import lgpsf

TAU = 5.0                       # window = TAU x prior; fit_operator's default is 10
TRUE_AXES, TRUE_ANGLE = (0.10, 0.04), 30.0         # 1-sigma axes, orientation
PRIOR_AXES = (0.10, 0.08)       # the guess: right along the major axis, 2x too
                                # wide across it, so the window has 2x the points
C_TRUE = np.array([1.0, 0.25, -0.30, 0.20, 0.15, -0.10])   # level <= 2 modes
SPIKE_SHARE = 0.6               # the spike's share of the diagonal entry
SPACING = 0.0045                # mesh spacing: about 30,000 points in the window
NUM_PROBES = 60
NOISE = 0.05                    # relative noise on y: the score's honest floor
LEVELS = [0, 1, 2, 3, 4]
EPS_LIST = [0.05, 0.1, 0.2]


def covariance(axes, angle):
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    R = np.array([[c, -s], [s, c]])
    return R @ np.diag(np.square(axes)) @ R.T


def pack_theta(mu, sigma):
    """The public theta: [mu | log diag L | strict lower L], Sigma = L L^T."""
    L = np.linalg.cholesky(sigma)
    return np.concatenate([mu, np.log(np.diag(L)), L[np.tril_indices(2, -1)]])


def geometry(model):
    """1-sigma axes (major, minor) and the major axis's angle, in degrees."""
    frame = model.frame()
    w, V = np.linalg.eigh(frame.L @ frame.L.T)
    return np.sqrt(w[::-1]), np.degrees(np.arctan2(V[1, -1], V[0, -1])) % 180.0


def build_window(rng):
    """A jittered grid clipped to the window ellipse, masses = cell areas."""
    window = lgpsf.make_frame(np.zeros(2),
                              TAU * np.linalg.cholesky(covariance(PRIOR_AXES, TRUE_ANGLE)))
    half = np.linalg.norm(window.L, axis=1)               # the ellipse's bounding box
    grid = np.meshgrid(np.arange(-half[0], half[0], SPACING),
                       np.arange(-half[1], half[1], SPACING), indexing="ij")
    x = np.vstack([grid[0].ravel(), grid[1].ravel()])     # (N, K): points as columns
    x += 0.3 * SPACING * rng.uniform(-1.0, 1.0, size=x.shape)
    x = x[:, np.linalg.norm(lgpsf.pullback(window, x), axis=0) <= 1.0]
    spike = int(np.argmin(np.linalg.norm(x, axis=0)))     # the row's own dof
    mu0 = x[:, spike]
    return x, np.full(x.shape[1], SPACING ** 2), spike, mu0, lgpsf.make_frame(mu0, window.L)


def score_on(x, mass, z, y, spike, mu0, target_mass, model, split):
    """Held-out score of `model` at ITS theta on a given quadrature (the full
    window or the cells) with no fitting at all: the whitening and the
    WhitenedBasis the fit uses internally, then linear_cv_score."""
    E = np.zeros((1, mass.size))
    E[0, spike] = 1.0
    basis = lgpsf.WhitenedBasis(x, target_mass, mass, model.modes, mu0, lgpsf.MuMode.Pinned)
    return lgpsf.linear_cv_score(
        lgpsf.whiten_probes(z, mass), lgpsf.whiten_data(y, target_mass), basis,
        lgpsf.to_theta_hat(model.theta, mu0, lgpsf.MuMode.Pinned),
        lgpsf.whiten_extra(E, target_mass, mass), split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="examples", help="where to write the figure")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    x, mass, spike, mu0, window = build_window(rng)

    # The target: a level-2 mode mix on an ellipsoid the prior gets wrong, plus
    # a spike. The same data model as fit_one_psf: probes z, responses y.
    target = lgpsf.LGExpansion(pack_theta(mu0, covariance(TRUE_AXES, TRUE_ANGLE)),
                               lgpsf.modes_up_to_level(2, 2), C_TRUE)
    smooth = lgpsf.eval_expansion(target, x)
    m_rho = mass[spike]
    s_true = SPIKE_SHARE / (1 - SPIKE_SHARE) * m_rho * smooth[spike]
    truth = m_rho * smooth * mass                     # the discrete row ...
    truth[spike] += m_rho * s_true                    # ... plus its spike
    z = rng.normal(size=(NUM_PROBES, x.shape[1]))     # (k, K) probe fields
    y = z @ truth                                     # all the fitter sees ...
    y += NOISE * np.linalg.norm(y) / np.sqrt(NUM_PROBES) * rng.normal(size=y.shape)
    # ... plus noise, standing in for the mismatch a real row always has
    # between its kernel and any finite expansion: the score's honest floor.

    config = lgpsf.ProbeFitConfig()
    config.mode_policy = lgpsf.ShellLadder(LEVELS)
    config.target_score = None
    split = lgpsf.kfold_split(NUM_PROBES, config.cv_folds)
    config.split = split                              # the same folds everywhere
    guesses = [lgpsf.InitialGuess(covariance(PRIOR_AXES, TRUE_ANGLE), label="prior")]

    def fit(x_fit, mass_fit, z_fit, spike_fit):
        return lgpsf.fit_from_probes(x_fit, mass_fit, z_fit, y, mu0, spike_index=spike_fit,
                                     config=config, guesses=guesses, target_mass=m_rho)

    tic = time.perf_counter()
    full = fit(x, mass, z, spike)
    t_full = time.perf_counter() - tic
    rows = [("full window", x.shape[1], t_full, full)]
    coarse = {}
    for eps in EPS_LIST:
        tic = time.perf_counter()
        # The window frame is TAU x the prior: the object that says how much of
        # the prior's shape is trusted, and the metric the cells are graded in.
        cw = lgpsf.coarsen_window(x, mass, z, [spike], mu0, window, eps)
        result = fit(cw.x, cw.m2, cw.z, cw.protected_cells[0])
        rows.append((f"eps {eps:g}", cw.num_cells, time.perf_counter() - tic, result))
        coarse[eps] = cw

    print(f"window: {x.shape[1]} points at spacing {SPACING:g}; truth 1-sigma axes "
          f"{TRUE_AXES[0]:.2f} x {TRUE_AXES[1]:.2f} ({TRUE_AXES[0] / SPACING:.0f} x "
          f"{TRUE_AXES[1] / SPACING:.0f} spacings),\nprior {PRIOR_AXES[0]:.2f} x "
          f"{PRIOR_AXES[1]:.2f}; {NUM_PROBES} probes. Scores: 'own' on the quadrature "
          f"the fit ran on,\n'on full' the same model re-scored on the full window.\n")
    print(f"{'fit on':<11} {'cells':>6} {'wall':>6} {'speed':>6}  {'1-sigma axes':<15} "
          f"{'angle':>5}  {'c[0]':>6} {'c[1]':>6} {'c[2]':>6} {'s/true':>6}  "
          f"{'own':>6} {'on full':>7} modes")
    axes, angle = geometry(target)
    print(f"{'truth':<11} {'':>6} {'':>6} {'':>6}  {axes[0]:.4f} x {axes[1]:.4f} {angle:>5.1f}  "
          f"{C_TRUE[0]:>6.3f} {C_TRUE[1]:>6.3f} {C_TRUE[2]:>6.3f} {1.0:>6.3f}")
    honest = {}
    for name, cells, t, result in rows:
        model = result.model
        axes, angle = geometry(model)
        honest[name] = score_on(x, mass, z, y, spike, mu0, m_rho, model, split)
        print(f"{name:<11} {cells:>6} {t:>5.2f}s {t_full / t:>5.1f}x  "
              f"{axes[0]:.4f} x {axes[1]:.4f} {angle:>5.1f}  "
              f"{model.c[0]:>6.3f} {model.c[1]:>6.3f} {model.c[2]:>6.3f} "
              f"{model.s[0] / s_true:>6.3f}  {result.score:>6.4f} {honest[name]:>7.4f} "
              f"{model.num_modes:>5}")

    # Re-scoring the full fit on its own window reproduces its score exactly;
    # the coarse fits, re-scored on the full window, land within a few percent.
    assert abs(honest["full window"] - full.score) < 1e-12
    worst = max(honest[f"eps {eps:g}"] for eps in EPS_LIST) / full.score - 1.0
    assert worst < 0.05, "a coarse fit should reproduce the full one closely"
    own = {name: result.score / honest[name] - 1.0 for name, _, _, result in rows}
    print(f"\nRe-scored on the full window, the coarse fits are within {worst * 100:.1f}% "
          f"of the full one's\nscore, with the same ellipsoid to about 1% and the same "
          f"modes, for {t_full / rows[2][2]:.0f}x less wall at\neps 0.1 (the coarsening "
          f"itself takes milliseconds). Their OWN scores are another\nmatter: the cells' "
          f"quadrature error inflates them, by {own['eps 0.1'] * 100:.0f}% at eps 0.1 and "
          f"{own['eps 0.2'] * 100:.0f}% at eps 0.2,\nfor models that are nearly as good. "
          f"A score measured on the cells is a search's\ninternal currency, not a "
          f"verdict -- which is why fit_operator re-scores its finalists\non the full "
          f"window before the baseline guard reads them.")

    try:
        import matplotlib
    except ImportError:
        print("\n(matplotlib not installed -- skipping the figure)")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Ellipse

    def ellipse(ax, sigma, **style):
        w, V = np.linalg.eigh(sigma)
        ax.add_patch(Ellipse(mu0, 2 * np.sqrt(w[1]), 2 * np.sqrt(w[0]),
                             angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                             facecolor="none", **style))

    cw, fitted = coarse[0.1], rows[2][3].model
    points_per_cell = np.bincount(cw.cell_of)
    limit = np.abs(smooth).max()
    fig, panels = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, field, title in ((panels[0], smooth, "the target on its full window"),
                             (panels[2], lgpsf.eval_expansion(fitted, x),
                              f"fitted on the {cw.num_cells} cells")):
        ax.scatter(x[0], x[1], c=field, s=1.5, cmap="RdBu_r", vmin=-limit, vmax=limit,
                   linewidths=0)
        ax.set_title(title, fontsize=11)
    ax = panels[1]
    sc = ax.scatter(cw.x[0], cw.x[1], c=points_per_cell, s=1.2 * points_per_cell,
                    cmap="viridis", norm=LogNorm(vmin=1), linewidths=0, alpha=0.85)
    farthest = int(np.argmax(np.linalg.norm(x - mu0[:, None], axis=0)))
    for cell, marker, label in ((cw.protected_cells[0], "x", "spike"),
                                (cw.cell_of[farthest], "+", "farthest point")):
        ax.plot(*cw.x[:, cell], marker, color="#d62728", markersize=8, markeredgewidth=1.8,
                label=f"{label}: a singleton")
    ticks = [2 ** i for i in range(int(np.log2(points_per_cell.max())) + 1)]
    bar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, ticks=ticks)
    bar.set_ticklabels([str(t) for t in ticks])
    bar.minorticks_off()
    bar.set_label("points per cell", fontsize=9)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"eps 0.1: {cw.num_cells} cells for {x.shape[1]} points", fontsize=11)
    for ax in panels:
        ellipse(ax, covariance(TRUE_AXES, TRUE_ANGLE), edgecolor="#333333", linewidth=1.2)
        ellipse(ax, covariance(PRIOR_AXES, TRUE_ANGLE), edgecolor="#333333", linewidth=1.0,
                linestyle=":")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    ellipse(panels[2], fitted.frame().L @ fitted.frame().L.T, edgecolor="#d62728",
            linewidth=1.2, linestyle="--")
    panels[0].set_xlabel("solid: the true 1-sigma ellipse; dotted: the prior", fontsize=9)
    panels[1].set_xlabel("single points near the centre; cells of size eps x distance beyond",
                         fontsize=9)
    panels[2].set_xlabel("dashed: the fitted 1-sigma ellipse", fontsize=9)
    fig.tight_layout()
    path = os.path.join(args.outdir, "coarsened_fit.png")
    fig.savefig(path, dpi=130)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
