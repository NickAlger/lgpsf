"""The general-purpose single-row fitting layer: everything from this
project assembled behind one raw-data interface.

The caller supplies the row's support batch (coordinates already selected
by their own means, e.g. a conservative a-priori ellipsoid), the lumped
masses on that batch, the raw probe fields and the row's raw responses,
an initial center mu0, and the LG mode list. This module whitens
internally (masses are *routed* here but all mass math stays in
whitening.py), multi-starts the VarPro fit over a ladder of initial
ellipsoids, selects the winner on internally held-out probe equations,
optionally releases mu under an accept-only-if-better guard, and returns
the fitted row model plus per-rung diagnostics.

Design decisions (2026-07-24, from the frog-kernel robustness study and
the PIG slice-37 experiments -- see docs/robust-init-notes.md and the
research repo's README):

  - The initial-Sigma ladder is circles, log-spaced from the local mesh
    spacing at mu0 up to a circle containing the whole batch: too-small
    and too-large inits both fail, in different ways, and best-of over a
    log ladder brackets every case observed. An optional caller-supplied
    sigma0 (e.g. an a-priori or backprojection estimate) joins the
    ladder as one extra rung -- on the real PIG columns the a-priori
    init was frequently the winner, and it is free.
  - The ladder is MIXED (window_shape_rungs=True): alongside the
    circles, scaled copies of the window's own shape -- the
    mass-weighted covariance of the batch geometry. The caller's window
    selection is information already paid for (in the intended pipeline
    it IS a conservatively inflated ellipsoid, whose aspect and
    orientation survive the inflation), and orientation is precisely
    the circle family's blind spot (the 8:1 frog study). Mass weighting
    matters: lumped masses ~ cell areas, so the weighted covariance
    measures the window REGION's shape, not where the mesh happens to
    be dense. Circles are retained as shape-agnostic insurance -- the
    window shape inherits the caller's prior errors exactly where the
    prior is wrong, and near a domain boundary the clipped window's
    shape is an artifact; holdout selection arbitrates.
  - Selection is by held-out probe equations (an internal random split),
    NOT by in-sample cost: a degenerate theta can lower the in-sample
    cost while ruining the fit (observed on the PIG release runaway).
    With too few probes for a split, selection falls back to in-sample
    cost, flagged in the result.
  - Selection additionally enforces WINDOW-CONTAINMENT admissibility:
    the caller's window is conservative, so the true kernel fits inside
    it by construction -- a fitted major semi-axis exceeding the window
    radius, or a released center leaving the window, is excluded from
    selection (with few holdout equations the score alone cannot
    reliably reject such degenerate fits; observed live: a 3000:1
    needle 3x the window winning a 4-equation holdout by 0.02).
  - The ladder runs in the fixed-mu encoding (fewer parameters, no
    center drift -- pinning mu collapsed the outcome variance in every
    experiment); mu="fixed_then_release" (default) then releases mu via
    ellipsoid_transform.release_mu -- once per rung, since a wrong mu0
    makes every pinned rung compensate with a distorted shape and no
    single rung's basin is trustworthy -- and accepts a released fit
    only if it beats every pinned one on the holdout score.
  - The diagonal spike enters as diag_index (the row's own point within
    the batch). Evidence says keep it always: on diagonal-dominated rows
    its absence poisons theta unrecoverably, and where unneeded its
    coefficient is ~0 at no cost. A general extra basis can be added
    later as a kwarg without breaking this API.

The backprojection initial-guess estimator (backproject_row +
row_moments) is deliberately a separate, caller-invoked utility: its
output feeds fit_row via sigma0/mu0 when wanted. The mass subtlety it
resolves: the RAW row values are m_rho * m_j * phi(x_j), so weighting by
|r| already IS the mesh quadrature of the continuum kernel's moments --
no mass vector needed, only the diagonal spike excluded (it is the
mesh-unresolvable part, not the smooth kernel).
"""
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from ellipsoid_transform import release_mu, theta_size, unpack_theta
from varpro import VarProOptions, fit_varpro
from whitening import (
    whiten_data,
    whiten_extra,
    whiten_probes,
    whitened_eval_feature,
    whitened_vjp_feature,
)


# ---------------------------------------------------------------------------
# Backprojection initial-guess estimation (separate, caller-invoked)
# ---------------------------------------------------------------------------

def backproject_row(z, y):
    """Unbiased estimate of the raw row H[rho, :] restricted to the batch,
    from probe pairs: r_hat = (1/k) sum_l y_l z_l. Requires the probe
    entries to be iid standard normal in the raw dof basis (E[z z^T] = I),
    which is the convention of the intended pipeline. z: (k, K) raw probe
    fields on the batch; y: (k,) raw row responses. Noise per entry is
    O(||row|| / sqrt(k)) -- see row_moments' rel_threshold."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    return z.T @ y / z.shape[0]


def row_moments(x, r_raw, diag_index=None, rel_threshold=0.05,
                noise_mad=0.0):
    """Weighted mean and covariance of a row's kernel magnitude, from RAW
    row values (measured or backprojected). Returns (mu_hat, Sigma_hat).

    No mass vector: r_raw[j] = m_rho * m_j * phi(x_j), so |r_raw| already
    carries the lumped-mass quadrature weight -- summing |r_raw|-weighted
    point statistics IS the mesh approximation of the continuum moments
    of |phi|. That is the whole mass subtlety, resolved by using raw
    values rather than kernel values.

    diag_index excludes the row's own point: its raw value contains the
    mesh-unresolvable spike, which belongs to the discrete correction,
    not to the smooth kernel whose moments we want -- left in, it drags
    mu_hat toward the node and shrinks Sigma_hat.

    Two thresholds zero out small weights (a backprojected row has a
    flat O(||row||/sqrt(k)) noise floor; on a support window that is
    mostly far field, those noise entries outnumber the signal and drag
    mu_hat toward the window centroid and inflate Sigma_hat):
      - rel_threshold: relative to max|r|;
      - noise_mad: in robust noise sigmas, sigma_noise estimated as
        1.4826 * median|r|. Recommended ~3-4 for backprojected rows on
        CONSERVATIVE support windows, where most entries are far-field
        noise and the median estimates it; on tight windows where most
        entries carry signal, the median overestimates the noise and
        rel_threshold alone is safer.
    Set both to 0.0 for exact (measured) rows.
    """
    x = np.asarray(x, dtype=float)
    w = np.abs(np.asarray(r_raw, dtype=float)).copy()
    if diag_index is not None:
        w[diag_index] = 0.0
    if rel_threshold > 0.0:
        w[w < rel_threshold * w.max()] = 0.0
    if noise_mad > 0.0:
        sigma_noise = 1.4826 * np.median(np.abs(np.asarray(r_raw)))
        w[w < noise_mad * sigma_noise] = 0.0
    w = w / w.sum()
    mu_hat = x @ w
    d = x - mu_hat[:, None]
    sigma_hat = np.einsum("j,ij,kj->ik", w, d, d)
    return mu_hat, sigma_hat


# ---------------------------------------------------------------------------
# The row fit
# ---------------------------------------------------------------------------

@dataclass
class RungFit:
    """Diagnostics for one ladder rung (or the release stage)."""

    label: str
    theta_init: np.ndarray
    theta: np.ndarray
    cost: float
    """Final in-sample (fit-split) whitened cost."""
    score: float
    """Selection score: relative whitened residual on the held-out probe
    equations (or the in-sample relative residual if no holdout)."""
    axes: np.ndarray
    """(N,) fitted 1-sigma semi-axes, sqrt(eig(L L^T))."""
    success: bool
    n_iterations: int
    admissible: bool = True
    """False if the fit violates the window-containment bound: the
    caller's window is conservative, so the true kernel fits inside it
    BY CONSTRUCTION -- a fitted major semi-axis exceeding the window
    radius, or a released center leaving the window, is impossible-or-
    runaway and is excluded from selection (holdout scores on few
    equations are too noisy to reject such fits reliably on their own)."""


@dataclass
class RowFitResult:
    """Winner of the ladder (post release decision and full-data refit)."""

    mu: np.ndarray
    """(N,) fitted (or pinned) center."""
    L: np.ndarray
    """(N, N) lower-triangular Cholesky factor of the fitted Sigma."""
    c: np.ndarray
    """(n_modes,) smooth kernel coefficients: kernel(x) = sum c_i phi_i."""
    s: np.ndarray
    """(num_extra,) spike coefficient(s); empty if diag_index was None."""
    theta: np.ndarray
    """Winning theta in its own encoding (see mu_fixed)."""
    mu_fixed: bool
    """True if the returned fit pins mu at mu0 (fixed encoding)."""
    released: bool
    """True if the free-mu release stage ran AND passed the guard."""
    score: float
    """The winner's selection score (see RungFit.score)."""
    used_holdout: bool
    """False if probes were too few for a split and selection fell back
    to in-sample cost."""
    winner: int
    """Index into rungs of the winning entry."""
    rungs: List[RungFit] = field(default_factory=list)


def _window_shape(x, m2_diag):
    """The window's own ellipsoid shape: mass-weighted covariance of the
    batch geometry, normalized to largest eigenvalue 1, as a lower
    Cholesky factor. Lumped masses ~ cell areas, so the mass weighting
    makes this the uniform-measure covariance of the window REGION,
    independent of mesh grading (an unweighted point covariance would
    measure where the mesh is dense instead). Eigenvalues are floored at
    1e-4 of the largest, capping the shape's aspect against degenerate
    (near-collinear or boundary-clipped) windows."""
    w = m2_diag / m2_diag.sum()
    xbar = x @ w
    d = x - xbar[:, None]
    S = np.einsum("j,ij,kj->ik", w, d, d)
    evals, evecs = np.linalg.eigh(S)
    evals = np.maximum(evals, 1e-4 * evals[-1]) / evals[-1]
    return np.linalg.cholesky((evecs * evals) @ evecs.T)


def _quiet_fit(*args, **kwargs):
    """fit_varpro with overflow warnings silenced: wild ladder rungs are
    expected and deliberately survivable (varpro's overflow sentinel);
    their intermediate warnings are noise to the caller."""
    with np.errstate(over="ignore", invalid="ignore"):
        return fit_varpro(*args, **kwargs)


def _theta_fixed_from_L(L):
    N = L.shape[0]
    th = [np.log(L[i, i]) for i in range(N)]
    for i in range(1, N):
        for j in range(i):
            th.append(L[i, j])
    return np.array(th)


def _axes_of(theta, N, mu0):
    _, L = unpack_theta(theta, N, mu0)
    return np.sqrt(np.linalg.eigvalsh(L @ L.T))


def fit_row(x, m2_diag, z, y, mu0, modes,
            diag_index=None,
            mu="fixed_then_release",
            sigma0=None,
            n_rungs=6,
            window_shape_rungs=True,
            holdout=0.2,
            refit_full=True,
            seed=0,
            options=None,
            verbose=False):
    """Fit one operator row from raw probe data. See the module docstring
    for the strategy; parameters:

    x : (N, K) raw batch coordinates (caller-selected support).
    m2_diag : (K,) lumped masses on the batch.
    z : (k, K) raw probe fields restricted to the batch.
    y : (k,) the row's raw responses, y_l = (H z_l)[rho].
    mu0 : (N,) initial center (e.g. the row's node coordinates).
    modes : list of (p, ell, m) LG modes for the smooth part.
    diag_index : index of the row's own point within the batch; builds
        the one-hot spike extra basis (and supplies the row mass). None
        disables the spike -- justified only for rows known to not be
        diagonal-dominated.
    mu : "fixed" | "free" | "fixed_then_release" (default). "fixed" pins
        mu at mu0 throughout; "free" fits it from the start (ladder
        seeds mu at mu0); the default pins it for the ladder and then
        releases it for the winner under the holdout guard.
    sigma0 : optional (N, N) SPD initial covariance appended to the
        ladder as one extra rung (a-priori estimate, or
        row_moments(backproject_row(z, y), ...)).
    n_rungs : number of log-spaced circle rungs from the local mesh
        spacing at mu0 to a circle containing the whole batch.
    window_shape_rungs : also add n_rungs scaled copies of the window's
        own shape (_window_shape: mass-weighted covariance of the batch
        geometry, normalized so the MAJOR semi-axis equals the rung
        radius) over the same radius range -- the mixed ladder. See the
        module docstring for why, and for the boundary-clipping caveat.
    holdout : fraction of probe equations held out for selection.
    refit_full : refit the winning configuration on ALL probe pairs
        (warm-started) after selection.
    seed : rng seed for the internal holdout split.
    options : VarProOptions for the individual fits (default caps
        max_nfev at 100 -- top-of-ladder rungs may wander).
    verbose : print the rung table.
    """
    x = np.asarray(x, dtype=float)
    m2_diag = np.asarray(m2_diag, dtype=float)
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    N, K = x.shape
    k = z.shape[0]
    if z.shape != (k, K) or y.shape != (k,) or m2_diag.shape != (K,):
        raise ValueError(f"shape mismatch: x {x.shape}, m2_diag "
                         f"{m2_diag.shape}, z {z.shape}, y {y.shape}")
    if mu not in ("fixed", "free", "fixed_then_release"):
        raise ValueError(f"unknown mu mode: {mu!r}")
    options = options if options is not None else VarProOptions(max_nfev=100)

    row_mass = float(m2_diag[diag_index]) if diag_index is not None else 1.0
    if diag_index is not None:
        E = np.zeros((1, K))
        E[0, diag_index] = 1.0
        e_hat = whiten_extra(E, row_mass, m2_diag)
    else:
        e_hat = None

    # internal holdout split (selection data the fits never see)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(k)
    n_hold = int(round(holdout * k))
    P_fix = theta_size(N, mu0)
    used_holdout = n_hold >= 1 and (k - n_hold) >= max(P_fix + 1, 4)
    if not used_holdout:
        n_hold = 0
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]

    n_lin = len(modes) + (1 if diag_index is not None else 0)
    if len(fit_idx) < 2 * (n_lin + P_fix):
        warnings.warn(
            f"only {len(fit_idx)} fit equations for ~{n_lin} linear + "
            f"{P_fix} nonlinear parameters (guideline k >= 2*(m+P)): "
            f"expect overfitting", stacklevel=2)

    z_hat_all = whiten_probes(z, m2_diag)
    y_hat_all = whiten_data(y, row_mass)
    z_fit, y_fit = z_hat_all[fit_idx], y_hat_all[fit_idx]
    z_hold, y_hold = z_hat_all[hold_idx], y_hat_all[hold_idx]

    def closures(mu0_or_none):
        common = dict(N=N, x=x, row_mass=row_mass, m2_diag=m2_diag,
                      modes=modes, mu0=mu0_or_none)
        b_eval = partial(whitened_eval_feature, **common)

        def b_vjp(theta, w_hat, _c=common):
            return whitened_vjp_feature(theta, w_hat=w_hat, **_c)

        return b_eval, b_vjp

    def score(theta, c, s, b_eval, zh, yh):
        """Relative whitened residual of the fitted model on probe rows."""
        pred = (zh @ b_eval(theta).T) @ c
        if e_hat is not None:
            pred = pred + (zh @ e_hat.T) @ s
        denom = max(float(np.linalg.norm(yh)), 1e-300)
        return float(np.linalg.norm(yh - pred) / denom)

    # --- the initial-Sigma ladder -----------------------------------------
    pts = x.T
    tree = cKDTree(pts)
    _, i_near = tree.query(mu0, k=1)
    h_local = tree.query(pts[i_near], k=2)[0][1]
    r_max = float(np.linalg.norm(x - mu0[:, None], axis=0).max())
    radii = np.geomspace(max(h_local, 1e-12 * max(r_max, 1.0)),
                         max(r_max, h_local), n_rungs)
    inits = [(f"circle r={r:.3g}",
              np.concatenate([np.full(N, np.log(r)),
                              np.zeros(N * (N - 1) // 2)]))
             for r in radii]
    if window_shape_rungs:
        L_shape = _window_shape(x, m2_diag)
        inits += [(f"window r={r:.3g}", _theta_fixed_from_L(r * L_shape))
                  for r in radii]
    if sigma0 is not None:
        inits.append(("sigma0",
                      _theta_fixed_from_L(np.linalg.cholesky(sigma0))))

    ladder_fixed = mu in ("fixed", "fixed_then_release")
    enc_mu0 = mu0 if ladder_fixed else None
    b_eval, b_vjp = closures(enc_mu0)

    def admissible_of(theta, enc_mu0_):
        axes = _axes_of(theta, N, enc_mu0_)
        ok = bool(axes.max() <= r_max)
        if enc_mu0_ is None:
            ok = ok and bool(np.linalg.norm(theta[:N] - mu0) <= r_max)
        return ok

    def select(rung_list):
        """Best score among admissible rungs; all rungs if none are."""
        cands = [i for i, r in enumerate(rung_list) if r.admissible]
        if not cands:
            cands = list(range(len(rung_list)))
        return min(cands, key=lambda i: rung_list[i].score)

    rungs = []
    for label, th_fixed in inits:
        th0 = th_fixed if ladder_fixed else release_mu(th_fixed, mu0)
        res = _quiet_fit(z_fit, y_fit, b_eval, b_vjp, th0,
                         e_hat=e_hat, options=options)
        sc = score(res.theta, res.c, res.s, b_eval,
                   z_hold if used_holdout else z_fit,
                   y_hold if used_holdout else y_fit)
        rungs.append(RungFit(label=label, theta_init=th0, theta=res.theta,
                             cost=res.cost, score=sc,
                             axes=_axes_of(res.theta, N, enc_mu0),
                             success=res.success,
                             n_iterations=res.n_iterations,
                             admissible=admissible_of(res.theta, enc_mu0)))

    winner = select(rungs)
    win_theta = rungs[winner].theta
    win_enc_fixed = ladder_fixed

    # --- guarded free-mu release, one per ladder rung ---------------------
    # Releasing only the winner is not enough: when mu0 is off, every
    # pinned rung compensates with a distorted shape, and the winner's
    # basin need not contain the true free-mu optimum -- releasing each
    # rung keeps the portfolio's basin diversity through the release.
    released = False
    if mu == "fixed_then_release":
        b_eval_f, b_vjp_f = closures(None)
        n_ladder = len(rungs)
        best_fixed_score = rungs[winner].score
        for i in range(n_ladder):
            th0_f = release_mu(rungs[i].theta, mu0)
            res_f = _quiet_fit(z_fit, y_fit, b_eval_f, b_vjp_f, th0_f,
                               e_hat=e_hat, options=options)
            sc_f = score(res_f.theta, res_f.c, res_f.s, b_eval_f,
                         z_hold if used_holdout else z_fit,
                         y_hold if used_holdout else y_fit)
            rungs.append(RungFit(label=f"release({rungs[i].label})",
                                 theta_init=th0_f, theta=res_f.theta,
                                 cost=res_f.cost, score=sc_f,
                                 axes=_axes_of(res_f.theta, N, None),
                                 success=res_f.success,
                                 n_iterations=res_f.n_iterations,
                                 admissible=admissible_of(res_f.theta, None)))
        i_best_rel = select(rungs[n_ladder:]) + n_ladder
        # the guard: accept a released fit only if it is admissible AND
        # beats every pinned rung on the holdout
        if rungs[i_best_rel].admissible and \
                rungs[i_best_rel].score < best_fixed_score:
            released = True
            winner = i_best_rel
            win_theta = rungs[i_best_rel].theta
            win_enc_fixed = False

    # --- final refit of the winning configuration on all pairs ------------
    final_enc_mu0 = mu0 if win_enc_fixed else None
    b_eval_w, b_vjp_w = closures(final_enc_mu0)
    if refit_full and used_holdout:
        res_w = _quiet_fit(z_hat_all, y_hat_all, b_eval_w, b_vjp_w,
                           win_theta, e_hat=e_hat, options=options)
    else:
        res_w = _quiet_fit(z_fit, y_fit, b_eval_w, b_vjp_w,
                           win_theta, e_hat=e_hat, options=options)
    mu_out, L_out = unpack_theta(res_w.theta, N, final_enc_mu0)

    if verbose:
        for i, r in enumerate(rungs):
            mark = " <-- winner" if i == winner else ""
            adm = "" if r.admissible else "  INADMISSIBLE"
            print(f"  {r.label:16s} score {r.score:.4f}  cost {r.cost:.3e}  "
                  f"axes ({', '.join(f'{a:.3g}' for a in r.axes)})  "
                  f"{'ok' if r.success else 'NO CONV'}{adm}{mark}")

    return RowFitResult(mu=np.array(mu_out, dtype=float, copy=True),
                        L=L_out, c=res_w.c, s=res_w.s,
                        theta=res_w.theta, mu_fixed=win_enc_fixed,
                        released=released,
                        score=rungs[winner].score,
                        used_holdout=used_holdout,
                        winner=winner, rungs=rungs)
