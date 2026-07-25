"""The general-purpose single-row fitting layer: everything from this
project assembled behind one raw-data interface.

The caller supplies the row's support batch (coordinates already selected
by their own means, e.g. a conservative a-priori ellipsoid), the lumped
masses on that batch, the raw probe fields and the row's raw responses,
an initial center mu0, and either an explicit LG mode list or a ladder of
oscillator levels. This module whitens internally (masses are *routed*
here but all mass math stays in whitening.py), runs an ORDERED STREAM OF
CANDIDATE FITS -- initial-ellipsoid family x scale x mode set x mu
encoding -- under one selection rule with early-stopping certificates,
and returns the winning row model plus the full scored candidate table.

The three-way classification behind the design (2026-07-24):
  - STRUCTURAL FACTS the caller declares and we never adjudicate:
    window, probes, masses, mu0, diag_index (spike), the mode dictionary.
  - COMPETING HYPOTHESES adjudicated by data, all through one mechanism
    (a candidate list + one selection rule): init shape/scale, mode-set
    size, fixed vs released mu.
  - NUMERICAL HYGIENE with fixed defaults (VarProOptions): ridge,
    Jacobian variant, LM tolerances, overflow sentinel.

Candidate axes and the evidence behind them (frog robustness study,
docs/robust-init-notes.md; PIG slice-37 refits in the research repo):

  - Initial-Sigma ladder: log-spaced circles (local mesh spacing at mu0
    -> whole-batch radius) PLUS scaled copies of the window's own shape
    (mass-weighted covariance of the batch geometry: measures the window
    REGION, not mesh density; recovers the caller's conservative-
    ellipsoid orientation, the circle family's blind spot) PLUS an
    optional caller sigma0 rung (a-priori or backprojection estimate --
    frequently the winner on real PIG columns).
  - Mode-set ladder (config.mode_levels): nested sets of complete
    oscillator shells. A level is fit only if the counting rule
    k >= 2*(n_modes + n_extra + P) admits it (slice-36's overfitting
    rule), which also guarantees the CV folds below are well-posed.
    Each new level seeds a jittered warm start from the previous level's
    best fit IN ADDITION to the raw inits -- jittered because the exact
    warm start starts on the enrichment saddle (level L optimality kills
    part of level L+1's first-order signal; see robust-init-notes).
  - mu encoding: the stream runs pinned at mu0 (pinning collapsed
    outcome variance in every experiment); mu="fixed_then_release"
    (default) then releases the winning level's admissible fits (one
    per distinct theta -- a wrong mu0 distorts every pinned fit's
    shape, so no single basin is trustworthy) and accepts a released
    fit only if it beats the pinned ones under the tie-break below.

Selection = admissibility, then score, then simplicity:
  - ADMISSIBILITY: the caller's window is conservative, so the true
    kernel fits inside it by construction -- fits whose major semi-axis
    exceeds the window radius, or released centers leaving the window,
    are excluded (few-equation validation scores cannot reliably reject
    such degenerate fits; observed live: a 3000:1 needle 3x the window
    winning a 4-equation holdout by 0.02. This also kills the PIG-style
    mu runaway structurally).
  - SCORE: linear-stage K-fold cross-validation. theta is fit once per
    candidate on all k equations; the LINEAR coefficients are then
    refit leave-fold-out at that theta and scored on the held folds, so
    every equation is scored out-of-sample for the linear stage. (The
    theta stage is therefore mildly optimistic -- P <= 5 parameters on
    k equations, bounded by the counting rule -- the price of one
    nonlinear fit per candidate instead of folds+1.) Never in-sample
    cost: a degenerate theta can lower it while ruining the fit.
  - SIMPLICITY TIE-BREAK: among admissible candidates within tie_delta
    of the best score, prefer fewer modes, then pinned over released
    (the one-standard-error rule's shape; subsumes the old release
    guard).

Early stopping = adaptive effort allocation (its safety is one-sided:
certificates only fire when the data shows the row is easy; hard rows
fail them and buy the full grid):
  - target_score: stop everything once an admissible candidate's CV
    score is this good -- nothing else can do more than tie.
  - mode_patience: stop growing the mode ladder after this many
    consecutive levels without improving the best score (validation
    curves on nested families are U-shaped plus noise; patience >= 2
    because single-step worsening is often noise).
  - Candidate ORDER makes the target certificate fire fast: sigma0
    first, then window-shape rungs middle-out, then circles middle-out
    (mid scales won most in practice; extreme scales are insurance).
    NO patience on the init axis: score-vs-scale is non-unimodal (8:1
    frog: 0.65, 0.65, 0.60, 0.96, 0.58) and two inits can agree on the
    wrong minimum, so inits prune only via the absolute certificate.

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
from lg_harmonics_table import TABLE
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
# Small helpers
# ---------------------------------------------------------------------------

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


def _modes_up_to_level(N, max_level):
    """All (p, ell, m) with oscillator level 2p + ell <= max_level --
    complete shells in energy order, the single-knob mode family."""
    modes = []
    for ell in range(max_level + 1):
        if (N, ell) not in TABLE:
            break
        _, rows = TABLE[(N, ell)]
        for m_i in range(len(rows)):
            for p in range((max_level - ell) // 2 + 1):
                modes.append((p, ell, m_i))
    return modes


def _mid_out(n):
    """Index order for a length-n ladder: middle first, then outward --
    mid scales win most often, extremes are insurance."""
    return sorted(range(n), key=lambda i: (abs(i - (n - 1) / 2), i))


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class RowFitConfig:
    """Declares the candidate axes and the search policy for fit_row.
    Numerical hygiene stays in VarProOptions (the `varpro` field)."""

    mu: str = "fixed_then_release"
    """"fixed" pins mu at mu0 throughout; "free" fits it from the start
    (inits seeded at mu0); "fixed_then_release" (default) pins it for
    the stream, then releases the winning level's fits under the
    tie-break guard."""

    mode_levels: Optional[List[int]] = None
    """Nested mode ladder: fit complete shells up to each listed
    oscillator level, ascending, with counting-rule admissibility and
    mode_patience stopping. None = use fit_row's explicit `modes`
    argument as the single mode set."""

    n_rungs: int = 6
    """Log-spaced circle scales from the local mesh spacing at mu0 to a
    circle containing the whole batch."""

    window_shape_rungs: bool = True
    """Also ladder scaled copies of the window's own shape
    (_window_shape) over the same scale range."""

    target_score: Optional[float] = 0.05
    """Absolute early-exit certificate: stop the whole search once an
    admissible candidate's CV score is at or below this. None disables."""

    mode_patience: int = 2
    """Stop growing the mode ladder after this many consecutive levels
    without improving the best score."""

    tie_delta: float = 0.0
    """Simplicity tie-break margin: among admissible candidates within
    tie_delta of the best score, prefer fewer modes, then pinned mu."""

    cv_folds: int = 5
    """Folds for the linear-stage CV score."""

    seed: int = 0
    """Rng seed for the CV fold split and warm-start jitter."""

    varpro: VarProOptions = field(
        default_factory=lambda: VarProOptions(max_nfev=100))
    """Per-fit numerics (category C); max_nfev capped because
    top-of-ladder rungs may wander."""


@dataclass
class CandidateFit:
    """One scored entry of the candidate table (also release-stage fits)."""

    label: str
    """Init label: "circle r=..", "window r=..", "sigma0",
    "warm(level<=L)", or "release(<label>)"."""
    modes_label: str
    """"level<=L" for ladder sets, "explicit" for a caller mode list."""
    n_modes: int
    released: bool
    """True for release-stage (free-mu) fits."""
    theta_init: np.ndarray
    theta: np.ndarray
    c: np.ndarray
    s: np.ndarray
    cost: float
    """In-sample whitened cost of the theta fit (diagnostic only --
    never used for selection)."""
    score: float
    """Linear-stage CV score: relative whitened residual with the linear
    coefficients refit leave-fold-out at the fitted theta."""
    axes: np.ndarray
    """(N,) fitted 1-sigma semi-axes, sqrt(eig(L L^T))."""
    success: bool
    n_iterations: int
    admissible: bool = True
    """False if the fit violates window containment (major semi-axis
    beyond the window radius, or a released center outside the window):
    impossible-or-runaway by the conservativeness of the window."""


@dataclass
class RowFitResult:
    """The winning candidate, plus the full audit trail."""

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
    modes: List
    """The winning mode set -- the (p, ell, m) list c corresponds to."""
    mu_fixed: bool
    released: bool
    """True if a release-stage fit won."""
    score: float
    score_kind: str
    """"cv", or "insample" when probes were too few for folds."""
    stop_reason: str
    """"target" | "mode_patience" | "exhausted"."""
    winner: int
    """Index into candidates."""
    candidates: List[CandidateFit] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    """Mode levels skipped by the counting rule."""


# ---------------------------------------------------------------------------
# The row fit
# ---------------------------------------------------------------------------

def fit_row(x, m2_diag, z, y, mu0, modes=None,
            diag_index=None, sigma0=None,
            config=None, verbose=False):
    """Fit one operator row from raw probe data. See the module docstring
    for the architecture. Parameters:

    x : (N, K) raw batch coordinates (caller-selected support).
    m2_diag : (K,) lumped masses on the batch.
    z : (k, K) raw probe fields restricted to the batch.
    y : (k,) the row's raw responses, y_l = (H z_l)[rho].
    mu0 : (N,) initial center (e.g. the row's node coordinates).
    modes : explicit (p, ell, m) list -- required iff config.mode_levels
        is None.
    diag_index : index of the row's own point within the batch; builds
        the one-hot spike extra basis (and supplies the row mass). None
        disables the spike -- justified only for rows known to not be
        diagonal-dominated.
    sigma0 : optional (N, N) SPD initial covariance, tried FIRST (e.g.
        an a-priori estimate, or row_moments(backproject_row(z, y), ...)).
    config : RowFitConfig.
    verbose : print the candidate table.
    """
    cfg = config if config is not None else RowFitConfig()
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
    if cfg.mu not in ("fixed", "free", "fixed_then_release"):
        raise ValueError(f"unknown mu mode: {cfg.mu!r}")
    if (cfg.mode_levels is None) == (modes is None):
        raise ValueError("provide exactly one of `modes` (explicit list) "
                         "or config.mode_levels (nested ladder)")

    row_mass = float(m2_diag[diag_index]) if diag_index is not None else 1.0
    n_extra = 1 if diag_index is not None else 0
    if diag_index is not None:
        E = np.zeros((1, K))
        E[0, diag_index] = 1.0
        e_hat = whiten_extra(E, row_mass, m2_diag)
    else:
        e_hat = None

    z_hat = whiten_probes(z, m2_diag)
    y_hat = whiten_data(y, row_mass)
    y_norm = max(float(np.linalg.norm(y_hat)), 1e-300)
    P_fix = theta_size(N, mu0)

    # CV folds (the counting rule below guarantees every admitted level
    # leaves each training complement overdetermined for the linear stage)
    rng = np.random.default_rng(cfg.seed)
    n_folds = max(2, min(cfg.cv_folds, k // 2))
    folds = np.array_split(rng.permutation(k), n_folds)
    score_kind = "cv"

    def cv_score(theta, b_eval):
        """Linear-stage CV: linear coefficients refit leave-fold-out at
        the given theta, scored on the held folds."""
        with np.errstate(over="ignore", invalid="ignore"):
            Phi = b_eval(theta)
        if not np.all(np.isfinite(Phi)):
            return np.inf
        A = z_hat @ Phi.T
        X = np.hstack([A, z_hat @ e_hat.T]) if e_hat is not None else A
        resids = []
        for f in folds:
            train = np.setdiff1d(np.arange(k), f)
            coef, *_ = np.linalg.lstsq(X[train], y_hat[train], rcond=None)
            resids.append(y_hat[f] - X[f] @ coef)
        return float(np.linalg.norm(np.concatenate(resids)) / y_norm)

    # --- geometry: ladder scales and the window shape ---------------------
    pts = x.T
    tree = cKDTree(pts)
    _, i_near = tree.query(mu0, k=1)
    h_local = tree.query(pts[i_near], k=2)[0][1]
    r_max = float(np.linalg.norm(x - mu0[:, None], axis=0).max())
    radii = np.geomspace(max(h_local, 1e-12 * max(r_max, 1.0)),
                         max(r_max, h_local), cfg.n_rungs)

    inits = []
    if sigma0 is not None:
        inits.append(("sigma0",
                      _theta_fixed_from_L(np.linalg.cholesky(sigma0))))
    if cfg.window_shape_rungs:
        L_shape = _window_shape(x, m2_diag)
        inits += [(f"window r={radii[i]:.3g}",
                   _theta_fixed_from_L(radii[i] * L_shape))
                  for i in _mid_out(len(radii))]
    inits += [(f"circle r={radii[i]:.3g}",
               np.concatenate([np.full(N, np.log(radii[i])),
                               np.zeros(N * (N - 1) // 2)]))
              for i in _mid_out(len(radii))]

    # --- mode sets ---------------------------------------------------------
    if cfg.mode_levels is not None:
        levels = sorted(cfg.mode_levels)
        mode_sets = [(f"level<={L}", _modes_up_to_level(N, L)) for L in levels]
    else:
        mode_sets = [("explicit", list(modes))]
    modes_of = dict(mode_sets)

    ladder_fixed = cfg.mu in ("fixed", "fixed_then_release")
    enc_mu0 = mu0 if ladder_fixed else None

    def closures(mlist, mu0_or_none):
        common = dict(N=N, x=x, row_mass=row_mass, m2_diag=m2_diag,
                      modes=mlist, mu0=mu0_or_none)
        b_eval = partial(whitened_eval_feature, **common)

        def b_vjp(theta, w_hat, _c=common):
            return whitened_vjp_feature(theta, w_hat=w_hat, **_c)

        return b_eval, b_vjp

    def admissible_of(theta, enc):
        axes = _axes_of(theta, N, enc)
        ok = bool(axes.max() <= r_max)
        if enc is None:
            ok = ok and bool(np.linalg.norm(theta[:N] - mu0) <= r_max)
        return ok

    def run_candidate(label, mlabel, mlist, th0, enc, released,
                      b_eval, b_vjp):
        res = _quiet_fit(z_hat, y_hat, b_eval, b_vjp, th0,
                         e_hat=e_hat, options=cfg.varpro)
        return CandidateFit(
            label=label, modes_label=mlabel, n_modes=len(mlist),
            released=released, theta_init=np.asarray(th0, dtype=float),
            theta=res.theta, c=res.c, s=res.s, cost=res.cost,
            score=cv_score(res.theta, b_eval),
            axes=_axes_of(res.theta, N, enc), success=res.success,
            n_iterations=res.n_iterations,
            admissible=admissible_of(res.theta, enc))

    def select(cands):
        adm = [i for i, c in enumerate(cands) if c.admissible]
        if not adm:
            adm = list(range(len(cands)))
        best = min(cands[i].score for i in adm)
        window = [i for i in adm if cands[i].score <= best + cfg.tie_delta]
        return min(window, key=lambda i: (cands[i].n_modes,
                                          cands[i].released, i))

    def hit_target(c):
        return (cfg.target_score is not None and c.admissible
                and c.score <= cfg.target_score)

    # --- the fixed-encoding (or free, if mu="free") candidate stream ------
    candidates = []
    skipped = []
    stop_reason = "exhausted"
    best_score = np.inf
    patience_left = cfg.mode_patience
    prev_level_best_theta = None
    stopped = False

    for mlabel, mlist in mode_sets:
        if k < 2 * (len(mlist) + n_extra + P_fix):
            skipped.append(mlabel)
            continue
        b_eval, b_vjp = closures(mlist, enc_mu0)
        level_inits = list(inits)
        if prev_level_best_theta is not None:
            jitter = 0.05 * rng.standard_normal(prev_level_best_theta.shape)
            level_inits.insert(0, (f"warm({candidates[-1].modes_label})",
                                   prev_level_best_theta + jitter))
        level_start = len(candidates)
        for label, th_fixed in level_inits:
            th0 = th_fixed if ladder_fixed else release_mu(th_fixed, mu0)
            cand = run_candidate(label, mlabel, mlist, th0, enc_mu0,
                                 False, b_eval, b_vjp)
            candidates.append(cand)
            if hit_target(cand):
                stop_reason = "target"
                stopped = True
                break
        level_cands = candidates[level_start:]
        level_adm = [c for c in level_cands if c.admissible]
        if level_adm:
            level_best = min(c.score for c in level_adm)
            if level_best < best_score:
                best_score = level_best
                patience_left = cfg.mode_patience
                i_best = min((i for i in range(level_start, len(candidates))
                              if candidates[i].admissible),
                             key=lambda i: candidates[i].score)
                prev_level_best_theta = candidates[i_best].theta
            else:
                patience_left -= 1
        else:
            patience_left -= 1
        if stopped:
            break
        if patience_left <= 0 and mlabel != mode_sets[-1][0]:
            stop_reason = "mode_patience"
            break

    if not candidates:
        raise ValueError(
            f"every mode set failed the counting rule k >= 2*(m + "
            f"{n_extra} + {P_fix}) at k={k}; smallest set has "
            f"{min(len(m) for _, m in mode_sets)} modes")

    winner = select(candidates)

    # --- guarded release stage --------------------------------------------
    if cfg.mu == "fixed_then_release" and stop_reason != "target":
        win_mlabel = candidates[winner].modes_label
        mlist = modes_of[win_mlabel]
        b_eval_f, b_vjp_f = closures(mlist, None)
        seen = set()
        sources = sorted(
            (c for c in candidates
             if c.modes_label == win_mlabel and c.admissible
             and not c.released),
            key=lambda c: c.score)
        for c in sources:
            key = tuple(np.round(c.theta, 8))
            if key in seen:
                continue
            seen.add(key)
            cand = run_candidate(f"release({c.label})", win_mlabel, mlist,
                                 release_mu(c.theta, mu0), None, True,
                                 b_eval_f, b_vjp_f)
            candidates.append(cand)
            if hit_target(cand):
                stop_reason = "target"
                break
        winner = select(candidates)

    win = candidates[winner]
    final_enc = mu0 if (ladder_fixed and not win.released) else None
    mu_out, L_out = unpack_theta(win.theta, N, final_enc)

    if verbose:
        for i, c in enumerate(candidates):
            mark = " <-- winner" if i == winner else ""
            adm = "" if c.admissible else "  INADMISSIBLE"
            print(f"  [{c.modes_label:>9s}] {c.label:20s} "
                  f"score {c.score:.4f}  "
                  f"axes ({', '.join(f'{a:.3g}' for a in c.axes)})  "
                  f"{'ok' if c.success else 'NO CONV'}{adm}{mark}")
        print(f"  stop: {stop_reason}; skipped: {skipped or 'none'}")

    return RowFitResult(mu=np.array(mu_out, dtype=float, copy=True),
                        L=L_out, c=win.c, s=win.s, theta=win.theta,
                        modes=modes_of[win.modes_label],
                        mu_fixed=not win.released and ladder_fixed,
                        released=win.released,
                        score=win.score, score_kind=score_kind,
                        stop_reason=stop_reason, winner=winner,
                        candidates=candidates, skipped=skipped)
