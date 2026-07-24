"""Visualize the VarPro ellipsoid fit, iteration by iteration, on the
"frog" kernel.

The target is a single impulse response of the spatially varying "frog"
blur kernel (eq. 7.4 of the localpsf paper; the definition below is
copied from the frog examples in the ellipsoid_psf repo): at source point
x0, the response phi_x0(y) is an anisotropic Gaussian blob (1-sigma axes
0.1 x 0.05, rotated by (pi/2)(x0_1 + x0_2)) carrying a cos * sin
modulation that makes it bumpy and mildly negative -- genuinely
non-Gaussian, so the LG modes beyond (p, ell) = (0, 0) have real work to
do, and the fitted ellipsoid is NOT just a moment ellipse.

Fit setup, mirroring the method's actual regime: the batch is a uniform
grid patch around x0 (unit masses, so the whitening layer is exercised
but is numerically the identity), the data are k random-probe responses
y = Z phi -- probing, not direct function values -- there is no
extra/spike basis, and theta starts from a deliberately bad guess: an
isotropic circle somewhat too large, with the center offset.

Mode choice is a single user-set knob, deliberately: every mode with
oscillator level 2p + ell <= max_level, i.e. complete shells in energy
order ((L+1)(L+2)/2 modes in 2D). This is the energy-ordered family the
research repo's "single-knob" selection walks through; the data-driven
stopping rule (held-out probes) is intentionally NOT reimplemented here
-- each figure fixes the knob by hand. The script first runs a suite of
(num_probes, max_level) configurations, one trajectory figure each:

  suite A -- modes scaled with probes (n_modes a factor ~2-4 below k):
      ( 20, level 3 = 10 modes)   (60, level 5 = 21 modes)
      (200, level 8 = 45 modes)
  suite B -- probe-starved regime, k = 20 fixed, sweeping the mode knob:
      (20, level 1 = 3 modes)     (20, level 3 = 10 modes, shared with A)
      (20, level 6 = 28 modes)    <- MORE modes than probes: the 20 x 28
      design matrix has a nullspace, the probe data can be interpolated
      exactly at ANY theta, and the reduced residual VarPro minimizes is
      ~0 everywhere -- so there is nothing left to steer the ellipsoid,
      and the rendered "fit" is unconstrained off the probe subspace.

Trajectory figures: fit_varpro's callback records theta_init, every
accepted Levenberg-Marquardt iterate, and the final solution. Rows are
then subsampled to the informative part of the trajectory: a row is kept
only if the cost improved by more than 1% since the last kept row (plus
always the final iterate) -- LM spends many late iterations polishing
theta with visually imperceptible change, and those rows would all look
identical. Columns:

  left    the true function, with that iterate's 1-sigma (solid) and
          2-sigma (dashed) ellipses overlaid;
  middle  the smooth LG approximation sum_i c_i phi_i(y; theta) at that
          iterate, on the same symmetric color scale;
  right   |c_i| for every LG mode against its oscillator energy
          E = 2(2p + ell) + N, log scale, fixed axes across rows -- the
          spectral decay of the fit as the ellipsoid locks in.

The script then produces one INITIAL-ELLIPSOID SWEEP figure per entry in
INIT_SWEEPS: the same problem (identical probes) fit from several
theta_init choices, one row per init, with the true function overlaid by
the init (gray) and final (black) ellipses, the final LG approximation,
the final coefficient spectrum, and the cost trajectory. Motivating
questions: can a better initial guess -- the right radius, or the
a-priori moment-ellipsoid shape Sigma(x0) = R^T Sigma0 R that the real
pipeline supplies -- rescue the probe-starved 3-mode fit, whose
ellipsoid runs away from a bad circular init? And how large is the basin
of attraction, in initial radius, for a configuration that fits
comfortably (60 probes, 21 modes)?

A third sweep restricts the basis to ZEROTH order -- the single level-0
Gaussian mode -- asking whether the pure-Gaussian ellipsoid fit is more
robust to bad inits than the 3-mode fit (its sweep includes a WORSE init,
r = 0.25, than the 3-mode failure case). Empirically it is, in the
convergence sense: it never runs away (with one even, positive mode,
inflating the ellipsoid makes the model nearly constant, which fits the
localized data badly -- the cost itself pushes back; there are no odd
modes to conspire with huge canceling coefficients).

The script closes with the two-stage experiment that robustness
suggests: fit the Gaussian from a bad init, restart the 3-mode fit from
the Gaussian's fitted theta. Two findings, both visible in the printed
output:

  1. The exact warm start makes ZERO progress -- structurally, not
     numerically. The level-1 (dipole) modes are precisely the
     mu-derivatives of the level-0 Gaussian, so first-order optimality
     of the pilot fit (mu free) says the pilot residual is orthogonal to
     exactly the dipole columns: (c*, 0, 0) already solves the enriched
     inner problem, the enriched reduced gradient vanishes, and the
     pilot optimum is a stationary point (a saddle) of the 3-mode
     problem. LM stops after one iteration with theta unchanged. (This
     generalizes: growing the basis by one energy level and warm-starting
     at the previous optimum with mu free ALWAYS starts at a point where
     the mu-optimality conditions kill part -- for level 0 -> 1, all --
     of the new modes' first-order signal.)
  2. A small random jitter escapes the saddle, but only into whichever
     basin the pilot theta happens to sit in: the Gaussian-optimal
     ellipsoid is systematically larger than the best 3-mode one (it
     inflates to cover the modulation lobes), and from there the 3-mode
     fit reaches the secondary 0.287 minimum (r=0.15 pilot) or falls
     back into the runaway (r=0.25 pilot). On this problem, simply
     initializing SMALL (r <= 0.07) beats the Gaussian pilot.

Needs matplotlib, so run with the `tttt` conda env rather than `t3toolbox`:
    /home/nick/miniconda3/envs/tttt/bin/python examples/varpro_frog_fit.py
"""
import math
import os
import sys
from functools import partial

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))

from ellipsoid_transform import unpack_theta
from lg_ellipsoid_feature import eval_feature
from lg_harmonics_table import TABLE
from varpro import VarProOptions, fit_varpro
from whitening import (
    whiten_data,
    whiten_probes,
    whitened_eval_feature,
    whitened_vjp_feature,
)

# ---------------------------------------------------------------------------
# The frog kernel (localpsf paper eq. 7.4), copied from
# ellipsoid_psf/examples/frog_kernel.cpp and vectorized over the y batch.
# ---------------------------------------------------------------------------

SIGMA0_DIAG = np.array([0.01, 0.0025])  # variances along the unrotated axes
A_MOD = 1.0                             # modulation strength (mildly negative kernel)


def _rotation(x):
    ang = 0.5 * np.pi * (x[0] + x[1])
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s], [s, c]])


def _bump(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])


def frog(y, x, sigma0_diag=None):
    """phi_x(y): the impulse response at source x, evaluated at a batch of
    points y of shape (2, *batch_shape); returns (*batch_shape,).
    sigma0_diag overrides the unrotated-axis variances (default
    SIGMA0_DIAG) -- the anisotropy knob for the robustness experiments;
    the modulation frequencies scale along, so the kernel stays
    frog-like at any aspect ratio."""
    sd = SIGMA0_DIAG if sigma0_diag is None else np.asarray(sigma0_diag)
    R = _rotation(x)
    d = y - x.reshape(2, *([1] * (y.ndim - 1)))
    p = np.einsum("ij,j...->i...", R, d)
    maha2 = p[0] ** 2 / sd[0] + p[1] ** 2 / sd[1]
    G = np.exp(-0.5 * maha2) / (2.0 * np.pi * np.sqrt(sd.prod()))
    modulation = (np.cos(p[0] / (np.sqrt(sd[0]) / 2.0))
                  * np.sin(p[1] / (np.sqrt(sd[1]) / 2.0)))
    return _bump(x) * (1.0 + A_MOD * modulation) * G


# ---------------------------------------------------------------------------
# Fit configuration
# ---------------------------------------------------------------------------

X0 = np.array([0.55, 0.35])   # source point; rotation there is 81 degrees
HALF_WIDTH = 0.3              # half-width of the square window around X0
FIT_GRID = 41                 # fit batch is FIT_GRID^2 points on the window
DISPLAY_GRID = 121            # display resolution (independent of the fit)
MU_OFFSET = np.array([0.03, -0.02])  # every init's center error, held fixed

CONFIGS = [
    # (num_probes, max_level); level L keeps all 2p + ell <= L, (L+1)(L+2)/2 modes
    # suite A: modes scaled with probes, ~2-4x more probes than modes
    (20, 3),     # small:   20 probes, 10 modes
    (60, 5),     # medium:  60 probes, 21 modes
    (200, 8),    # large:  200 probes, 45 modes
    # suite B: probe-starved (k = 20 fixed), sweeping the mode knob;
    # (20, 3) above doubles as this sweep's middle entry
    (20, 1),     # few modes:        3 modes, crude but well-determined
    (20, 6),     # too many modes:  28 modes > 20 probes, underdetermined
]


def modes_up_to_level(N, max_level):
    modes = []
    for ell in range(max_level + 1):
        _, rows = TABLE[(N, ell)]
        for m in range(len(rows)):
            for p in range((max_level - ell) // 2 + 1):
                modes.append((p, ell, m))
    return modes


def window_grid(n):
    xs = np.linspace(X0[0] - HALF_WIDTH, X0[0] + HALF_WIDTH, n)
    ys = np.linspace(X0[1] - HALF_WIDTH, X0[1] + HALF_WIDTH, n)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=0)  # (2, n, n)


def theta_circle(radius):
    """Isotropic init of the given 1-sigma radius, center offset by MU_OFFSET."""
    return np.array([X0[0] + MU_OFFSET[0], X0[1] + MU_OFFSET[1],
                     math.log(radius), math.log(radius), 0.0])


def theta_oriented(a, b, angle_deg):
    """An anisotropic init: 1-sigma semi-axes (a, b) with the a-axis
    rotated angle_deg from horizontal, center offset by MU_OFFSET as
    always. Building block for orientation-covering init dictionaries."""
    ang = math.radians(angle_deg)
    R = np.array([[math.cos(ang), -math.sin(ang)],
                  [math.sin(ang), math.cos(ang)]])
    sigma = R @ np.diag([a**2, b**2]) @ R.T
    L = np.linalg.cholesky(sigma)
    return np.array([X0[0] + MU_OFFSET[0], X0[1] + MU_OFFSET[1],
                     math.log(L[0, 0]), math.log(L[1, 1]), L[1, 0]])


def theta_true_shape(scale=1.0, sigma0_diag=None):
    """The a-priori moment-ellipsoid init the real pipeline would supply:
    Sigma(x0) = R(x0)^T Sigma0 R(x0), log-Cholesky-encoded, axes scaled by
    `scale`; the center still carries the same MU_OFFSET error."""
    sd = SIGMA0_DIAG if sigma0_diag is None else np.asarray(sigma0_diag)
    R = _rotation(X0)
    sigma = R.T @ np.diag(sd) @ R
    L = scale * np.linalg.cholesky(sigma)
    return np.array([X0[0] + MU_OFFSET[0], X0[1] + MU_OFFSET[1],
                     math.log(L[0, 0]), math.log(L[1, 1]), L[1, 0]])


INIT_SWEEPS = [
    # hard case: 20 probes, 3 modes, where the circle-r=0.15 init runs away.
    # Can a better radius rescue it? If not, does the right *shape*?
    ("p020_L1", 20, 1, [
        ("circle r=0.15", theta_circle(0.15)),
        ("circle r=0.10", theta_circle(0.10)),
        ("circle r=0.07", theta_circle(0.07)),
        ("circle r=0.05", theta_circle(0.05)),
        ("true shape", theta_true_shape()),
        ("true shape x1.5", theta_true_shape(1.5)),
    ]),
    # easy case: 60 probes, 21 modes -- how wide is the basin in radius?
    ("p060_L5", 60, 5, [
        ("circle r=0.25", theta_circle(0.25)),
        ("circle r=0.15", theta_circle(0.15)),
        ("circle r=0.10", theta_circle(0.10)),
        ("circle r=0.05", theta_circle(0.05)),
    ]),
    # zeroth order: the single level-0 Gaussian mode, 20 probes -- is the
    # pure-Gaussian ellipsoid fit robust enough to bad inits to serve as a
    # cheap pilot stage? Includes a worse init (r=0.25) than the 3-mode
    # failure case.
    ("p020_L0", 20, 0, [
        ("circle r=0.25", theta_circle(0.25)),
        ("circle r=0.15", theta_circle(0.15)),
        ("circle r=0.10", theta_circle(0.10)),
        ("circle r=0.07", theta_circle(0.07)),
        ("circle r=0.05", theta_circle(0.05)),
        ("true shape", theta_true_shape()),
    ]),
]


# ---------------------------------------------------------------------------
# Shared fitting machinery
# ---------------------------------------------------------------------------

def build_problem(num_probes, max_level, sigma0_diag=None, fit_grid=None,
                  mu0=None):
    """One probing problem: the whitened arrays and basis closures for
    fitting the frog response at X0, exactly as fit_varpro consumes them.
    Fixed seed, so every fit of the same configuration sees the same
    probes (essential for comparing inits against each other). mu0 is the
    library-wide fit-mu/fixed-mu switch: None fits the center as part of
    theta; an array pins it there (theta then has 2 fewer entries)."""
    rng = np.random.default_rng(0)
    modes = modes_up_to_level(2, max_level)

    n_grid = FIT_GRID if fit_grid is None else fit_grid
    x_batch = window_grid(n_grid).reshape(2, -1)       # (2, K)
    K = x_batch.shape[1]
    f_true_batch = frog(x_batch, X0, sigma0_diag)      # (K,)

    row_mass = 1.0
    m2_diag = np.ones(K)                               # whitening == identity here
    z = rng.standard_normal((num_probes, K))
    z_hat = whiten_probes(z, m2_diag)
    y_hat = whiten_data(z @ f_true_batch, row_mass)

    common = dict(N=2, x=x_batch, row_mass=row_mass, m2_diag=m2_diag,
                  modes=modes, mu0=mu0)
    basis_eval = partial(whitened_eval_feature, **common)

    def basis_vjp(theta, w_hat):
        return whitened_vjp_feature(theta, w_hat=w_hat, **common)

    return dict(num_probes=num_probes, max_level=max_level, modes=modes,
                energies=np.array([2 * (2 * p + ell) + 2 for (p, ell, m) in modes]),
                z_hat=z_hat, y_hat=y_hat,
                basis_eval=basis_eval, basis_vjp=basis_vjp)


def fit_with_history(prob, theta_init):
    """Run fit_varpro on a built problem, recording the iterate trajectory
    via the callback. Returns (result, history, rel_misfit); history
    entries are (theta, c, cost)."""
    history = []
    result = fit_varpro(
        prob["z_hat"], prob["y_hat"],
        prob["basis_eval"], prob["basis_vjp"], theta_init,
        options=VarProOptions(max_nfev=100),
        callback=lambda th, c, r: history.append((th, c, 0.5 * float(r @ r))),
    )
    # relative misfit is comparable across configs; raw cost is not (it
    # sums over k probe entries, and k differs between configurations)
    rel_misfit = np.linalg.norm(result.residual) / np.linalg.norm(prob["y_hat"])
    return result, history, rel_misfit


def render_model(theta, c, X_disp, modes, mu0=None):
    """The fitted smooth model on the display grid. The fitted c applies to
    the whitened features, which equal the raw ones here (unit masses), so
    raw eval_feature renders it."""
    feats = eval_feature(theta, 2, X_disp, modes, mu0=mu0)  # (n_modes, n, n)
    return np.tensordot(c, feats, axes=([0], [0]))


def ellipse_path(theta, radius, tt, mu0=None):
    """Points of the |u| = radius ellipse of theta, shape (2, len(tt))."""
    mu, L = unpack_theta(theta, 2, mu0)
    return mu[:, None] + L @ (radius * np.stack([np.cos(tt), np.sin(tt)]))


def _coef_axis_limits(coef_arrays):
    """Shared log-scale limits for coefficient panels."""
    abs_c = np.concatenate([np.abs(c) for c in coef_arrays])
    c_top = max(float(abs_c.max()), 1e-12)
    positive = abs_c[abs_c > 0]
    c_bottom = max(float(positive.min()) if positive.size else c_top * 1e-8,
                   c_top * 1e-8)
    return c_bottom, c_top


EXTENT = (X0[0] - HALF_WIDTH, X0[0] + HALF_WIDTH,
          X0[1] - HALF_WIDTH, X0[1] + HALF_WIDTH)
TT = np.linspace(0.0, 2.0 * np.pi, 200)


# ---------------------------------------------------------------------------
# Figure 1 flavor: one row per (subsampled) LM iterate
# ---------------------------------------------------------------------------

def run_trajectory(num_probes, max_level):
    prob = build_problem(num_probes, max_level)
    modes = prob["modes"]
    print(f"--- {num_probes} probes, {len(modes)} modes (level <= {max_level}) ---")

    # deliberately bad start: offset center, isotropic circle a bit too big
    # (true 1-sigma axes are 0.1 and 0.05)
    result, history, rel_misfit = fit_with_history(prob, theta_circle(0.15))
    print(f"fit: success={result.success}, {result.n_iterations} iterations, "
          f"{result.n_function_evals} residual evals, cost={result.cost:.4e}, "
          f"relative misfit {rel_misfit:.3f}")

    # --- subsample the trajectory to the rows where something happened ---
    kept = [0]
    for i in range(1, len(history)):
        if history[i][2] < (1.0 - 0.01) * history[kept[-1]][2]:
            kept.append(i)
    if kept[-1] != len(history) - 1:
        kept.append(len(history) - 1)
    print(f"recorded {len(history)} iterates, keeping rows {kept}")

    # --- the figure ------------------------------------------------------
    X_disp = window_grid(DISPLAY_GRID)                 # (2, n, n)
    f_true = frog(X_disp, X0)
    vmax = np.max(np.abs(f_true))

    # fixed axes for the coefficient column, so decay is comparable across rows
    c_bottom, c_top = _coef_axis_limits([history[i][1] for i in kept])
    energies = prob["energies"]

    n_rows = len(kept)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10.8, 3.3 * n_rows))
    axes = np.atleast_2d(axes)

    for row, i in enumerate(kept):
        theta, c, cost = history[i]
        ax_true, ax_fit, ax_coef = axes[row]

        ax_true.imshow(f_true.T, origin="lower", extent=EXTENT,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for radius, style in [(1.0, "solid"), (2.0, "dashed")]:
            e = ellipse_path(theta, radius, TT)
            ax_true.plot(e[0], e[1], color="black", linestyle=style, linewidth=1.3)
        # pin every row to the window frame: an oversized early ellipse must
        # get clipped, not stretch this row's axes relative to the others
        ax_true.set_xlim(EXTENT[0], EXTENT[1])
        ax_true.set_ylim(EXTENT[2], EXTENT[3])

        approx = render_model(theta, c, X_disp, modes)
        ax_fit.imshow(approx.T, origin="lower", extent=EXTENT,
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax_coef.semilogy(energies, np.maximum(np.abs(c), c_bottom), "o",
                         markersize=4, alpha=0.75)
        ax_coef.set_ylim(c_bottom * 0.5, c_top * 2.0)
        ax_coef.set_xlim(energies.min() - 1, energies.max() + 1)
        ax_coef.grid(True, alpha=0.3)
        if row == n_rows - 1:
            ax_coef.set_xlabel(r"mode energy $2(2p+\ell)+N$", fontsize=9)

        label = "init" if i == 0 else ("final" if i == len(history) - 1 else f"iter {i}")
        ax_true.set_ylabel(f"{label}\ncost {cost:.2e}", fontsize=9)
        for ax in (ax_true, ax_fit, ax_coef):
            ax.tick_params(labelsize=7)

    axes[0, 0].set_title(r"true $\varphi_{x_0}$ + fitted ellipsoid", fontsize=11)
    axes[0, 1].set_title("LG approximation at this iterate", fontsize=11)
    axes[0, 2].set_title(r"$|c_i|$ vs mode energy", fontsize=11)
    fig.suptitle(f"VarPro frog fit -- {num_probes} probes, "
                 f"{len(modes)} modes (level $\\leq$ {max_level})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out_name = f"varpro_frog_fit_p{num_probes:03d}_L{max_level}.png"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")
    return num_probes, len(modes), max_level, result.cost, rel_misfit


# ---------------------------------------------------------------------------
# Figure 2 flavor: one row per initial ellipsoid, final state + cost curve
# ---------------------------------------------------------------------------

def run_init_sweep(tag, num_probes, max_level, inits,
                   sigma0_diag=None, fit_grid=None, title_note="", mu0=None):
    prob = build_problem(num_probes, max_level, sigma0_diag, fit_grid, mu0=mu0)
    modes = prob["modes"]
    print(f"=== init sweep {tag}: {num_probes} probes, {len(modes)} modes"
          f"{title_note} ===")

    rows = []
    for label, theta0 in inits:
        result, history, rel_misfit = fit_with_history(prob, theta0)
        costs = [h[2] for h in history]
        rows.append((label, theta0, result, costs, rel_misfit))
        print(f"  {label:16s} success={result.success} "
              f"iters={result.n_iterations:3d} rel_misfit={rel_misfit:.3f}")

    X_disp = window_grid(DISPLAY_GRID)
    f_true = frog(X_disp, X0, sigma0_diag)
    vmax = np.max(np.abs(f_true))

    c_bottom, c_top = _coef_axis_limits([r.c for _, _, r, _, _ in rows])
    energies = prob["energies"]
    all_costs = np.concatenate([np.asarray(costs) for _, _, _, costs, _ in rows])
    cost_lo, cost_hi = float(all_costs.min()), float(all_costs.max())

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14.0, 3.3 * n_rows))
    axes = np.atleast_2d(axes)

    for row, (label, theta0, result, costs, rel_misfit) in enumerate(rows):
        ax_true, ax_fit, ax_coef, ax_cost = axes[row]

        ax_true.imshow(f_true.T, origin="lower", extent=EXTENT,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        e0 = ellipse_path(theta0, 1.0, TT, mu0=mu0)
        ax_true.plot(e0[0], e0[1], color="gray", linestyle="solid", linewidth=1.2)
        for radius, style in [(1.0, "solid"), (2.0, "dashed")]:
            e = ellipse_path(result.theta, radius, TT, mu0=mu0)
            ax_true.plot(e[0], e[1], color="black", linestyle=style, linewidth=1.3)
        ax_true.set_xlim(EXTENT[0], EXTENT[1])
        ax_true.set_ylim(EXTENT[2], EXTENT[3])

        approx = render_model(result.theta, result.c, X_disp, modes, mu0=mu0)
        ax_fit.imshow(approx.T, origin="lower", extent=EXTENT,
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax_coef.semilogy(energies, np.maximum(np.abs(result.c), c_bottom), "o",
                         markersize=4, alpha=0.75)
        ax_coef.set_ylim(c_bottom * 0.5, c_top * 2.0)
        ax_coef.set_xlim(energies.min() - 1, energies.max() + 1)
        ax_coef.grid(True, alpha=0.3)

        ax_cost.semilogy(range(len(costs)), costs, "o-", markersize=3)
        ax_cost.set_ylim(cost_lo * 0.5, cost_hi * 2.0)
        ax_cost.grid(True, alpha=0.3)
        if row == n_rows - 1:
            ax_coef.set_xlabel(r"mode energy $2(2p+\ell)+N$", fontsize=9)
            ax_cost.set_xlabel("accepted iterate", fontsize=9)

        conv = "" if result.success else "  (no conv.)"
        ax_true.set_ylabel(f"{label}\nmisfit {rel_misfit:.3f}{conv}", fontsize=9)
        for ax in (ax_true, ax_fit, ax_coef, ax_cost):
            ax.tick_params(labelsize=7)

    axes[0, 0].set_title("true + init (gray) / final (black)", fontsize=11)
    axes[0, 1].set_title("final LG approximation", fontsize=11)
    axes[0, 2].set_title(r"final $|c_i|$ vs mode energy", fontsize=11)
    axes[0, 3].set_title("cost vs iterate", fontsize=11)
    fig.suptitle(f"Initial-ellipsoid sweep -- {num_probes} probes, "
                 f"{len(modes)} modes (level $\\leq$ {max_level}){title_note}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    out_name = f"varpro_frog_init_sweep_{tag}.png"
    out_path = os.path.join(os.path.dirname(__file__), out_name)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved {out_path}")
    return [(label, res.success, res.n_iterations, rel, res.theta)
            for label, _, res, _, rel in rows]


def run_two_stage(num_probes, bad_inits):
    """The pilot-stage experiment: from a bad init, fit the single
    level-0 Gaussian mode first, then restart the 3-mode fit from the
    Gaussian stage's fitted theta (exactly, and with a small jitter) --
    versus fitting the 3 modes directly from the same bad init. Same seed
    everywhere, so all fits see identical probe data. See the module
    docstring for why the exact warm start is a saddle of the enriched
    problem (dipoles = mu-derivatives of the Gaussian) and makes zero
    progress by construction."""
    prob_gauss = build_problem(num_probes, 0)
    prob_3mode = build_problem(num_probes, 1)
    rng = np.random.default_rng(42)
    print(f"=== two-stage (Gaussian pilot -> 3 modes), {num_probes} probes ===")
    for label, theta0 in bad_inits:
        res_g, _, mis_g = fit_with_history(prob_gauss, theta0)
        res_d, _, mis_d = fit_with_history(prob_3mode, theta0)
        res_2, _, mis_2 = fit_with_history(prob_3mode, res_g.theta)
        theta_jit = res_g.theta + 1e-2 * rng.standard_normal(res_g.theta.shape)
        res_j, _, mis_j = fit_with_history(prob_3mode, theta_jit)

        def _s(res):
            return "ok" if res.success else "NO CONV"

        print(f"  {label:16s} gaussian pilot {mis_g:.3f} ({_s(res_g)})  |  "
              f"direct 3-mode {mis_d:.3f} ({_s(res_d)})  |  "
              f"warm start {mis_2:.3f} ({_s(res_2)}, "
              f"{res_2.n_iterations} it, saddle)  |  "
              f"jittered {mis_j:.3f} ({_s(res_j)})")


def main():
    summary = [run_trajectory(num_probes, max_level)
               for num_probes, max_level in CONFIGS]
    print("\nprobes  modes  level  final cost  rel misfit")
    for num_probes, n_modes, max_level, cost, rel in summary:
        print(f"{num_probes:6d} {n_modes:6d} {max_level:6d}  {cost:.4e}  {rel:10.3f}")
    print()
    for tag, num_probes, max_level, inits in INIT_SWEEPS:
        run_init_sweep(tag, num_probes, max_level, inits)
    print()
    run_two_stage(20, [("circle r=0.25", theta_circle(0.25)),
                       ("circle r=0.15", theta_circle(0.15))])


if __name__ == "__main__":
    main()
