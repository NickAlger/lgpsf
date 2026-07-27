"""Fitting a function from random probe measurements: the general-purpose
top layer, assembling everything in this project behind one raw-data
interface.

THE PROBLEM. A target function on a batch of points is known only
through inner products with random probe fields: y_l = <target, z_l>.
The motivating example is one row of an implicitly available operator
H = M1 Phi M2 (y_l is then (H z_l) evaluated at the target's dof), but
nothing here requires that -- any function measured against random
vectors fits the same frame. The caller supplies the batch (coordinates
already selected by their own means, e.g. a conservative a-priori
ellipsoid), the lumped masses on it, the raw probes and responses, an
initial center mu0, and either an explicit LG mode list or a ladder of
oscillator levels. This module whitens internally (masses are *routed*
here but all mass math stays in whitening.py), runs an ORDERED STREAM OF
CANDIDATE FITS -- initial-ellipsoid family x scale x mode set x mu
encoding -- under one selection rule with early-stopping certificates,
and returns the winning model plus the full scored candidate table.

The three-way classification behind the design (2026-07-24):
  - STRUCTURAL FACTS the caller declares and we never adjudicate:
    window, probes, masses, mu0, spike_index, the mode dictionary.
  - COMPETING HYPOTHESES adjudicated by data, all through one mechanism
    (a candidate list + one selection rule): init shape/scale, mode-set
    size, fixed vs released mu.
  - NUMERICAL HYGIENE with fixed defaults (VarProOptions): ridge,
    Jacobian variant, LM tolerances, overflow sentinel.

Candidate axes (generation lives in init_dictionary.py; evidence: frog
robustness study, docs/robust-init-notes.md; PIG slice-37 refits):
  - Initial-Sigma ladder: log-spaced circles (local mesh spacing at mu0
    -> whole-batch radius) PLUS scaled copies of the window's own shape
    (mass-weighted covariance of the batch geometry -- recovers the
    caller's conservative-ellipsoid orientation, the circle family's
    blind spot) PLUS an optional caller sigma0 rung (a-priori or
    probe_moments backprojection estimate).
  - Mode-set ladder (config.mode_levels): nested complete oscillator
    shells (lg_functions.modes_up_to_level). A level is fit only if the
    counting rule k >= 2*(n_modes + n_extra + P) admits it, which also
    guarantees the CV folds below are well-posed. Each new level seeds
    a jittered warm start from the previous level's best (exact warm
    starts sit on the enrichment saddle; see robust-init-notes).
  - mu encoding: the stream runs pinned at mu0 (pinning collapsed
    outcome variance in every experiment); mu="fixed_then_release"
    (default) then releases the winning level's admissible fits (one
    per distinct theta) and accepts a released fit only under the
    tie-break below. CAVEAT (PIG field scale, slice 38): the release
    guard's window-radius bound on the center is far too loose -- at
    operator scale release shipped on ~91% of rows while buying
    nothing on held-out data; pin mu for operator runs until release
    is re-armed with a basin-scale ||mu - mu0|| bound.

Selection = admissibility, then score, then simplicity:
  - ADMISSIBILITY: the caller's window is conservative, so the true
    kernel fits inside it by construction -- fits whose major semi-axis
    exceeds the window radius, or released centers leaving the window,
    are excluded (few-equation validation scores cannot reliably reject
    such degenerate fits; observed live: a 3000:1 needle 3x the window
    winning a 4-equation holdout by 0.02. This also kills the mu
    runaway structurally).
  - SCORE: linear-stage K-fold cross-validation (linear_cv_score,
    public -- it can also score a NON-fitted model, e.g. an a-priori
    guess per target, a zero-fit field diagnostic). theta is fit once
    per candidate on all k equations; the LINEAR coefficients are then
    refit leave-fold-out at that theta, so every equation is scored
    out-of-sample for the linear stage at the price of ONE nonlinear
    fit per candidate. (The theta stage is therefore mildly optimistic
    -- P <= 5 parameters on k equations, bounded by the counting rule.)
    Never in-sample cost: a degenerate theta can lower it while ruining
    the fit.
  - SIMPLICITY TIE-BREAK: among admissible candidates within tie_delta
    of the best score, prefer fewer modes, then pinned over released
    (the one-standard-error rule's shape; subsumes the old release
    guard).

Early stopping = adaptive effort allocation (its safety is one-sided:
certificates only fire when the data shows the target is easy; hard
targets fail them and buy the full grid):
  - target_score: stop everything once an admissible candidate's CV
    score is this good -- nothing else can do more than tie.
  - mode_patience: stop growing the mode ladder after this many
    consecutive levels without improving the best score (validation
    curves on nested families are U-shaped plus noise; patience >= 2
    because single-step worsening is often noise).
  - Candidate ORDER makes the target certificate fire fast: sigma0
    first, then window-shape rungs middle-out, then circles middle-out.
    NO patience on the init axis: score-vs-scale is non-unimodal (8:1
    frog: 0.65, 0.65, 0.60, 0.96, 0.58) and two inits can agree on the
    wrong minimum, so inits prune only via the absolute certificate.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ellipsoid_transform import release_mu, theta_size, unpack_theta
from init_dictionary import (
    circle_rungs,
    ladder_scales,
    local_spacing,
    theta_from_L,
    window_radius,
    window_rungs,
    window_shape,
)
from mode_policy import (
    MAX_PROPOSALS,
    ExplicitLadder,
    FixedSet,
    LevelRecord,
    ModeSearchContext,
    ShellLadder,
)
from varpro import VarProOptions, fit_varpro
from whitening import whiten_data, whiten_extra, whiten_probes, whitened_basis


def _resolve_mode_policy(cfg, modes):
    """Exactly-one-of validation + resolution of the legacy config
    fields into mode_policy instances."""
    n_sources = sum(x is not None for x in
                    (modes, cfg.mode_levels, cfg.mode_sets,
                     cfg.mode_policy))
    if n_sources != 1:
        raise ValueError("provide exactly one of `modes` (explicit "
                         "list), config.mode_levels (shell ladder), "
                         "config.mode_sets (explicit nested ladder), "
                         "or config.mode_policy")
    if cfg.mode_policy is not None:
        return cfg.mode_policy
    if cfg.mode_levels is not None:
        return ShellLadder(cfg.mode_levels)
    if cfg.mode_sets is not None:
        return ExplicitLadder(cfg.mode_sets)
    return FixedSet(modes)


def _quiet_fit(*args, **kwargs):
    """fit_varpro with overflow warnings silenced: wild ladder rungs are
    expected and deliberately survivable (varpro's overflow sentinel);
    their intermediate warnings are noise to the caller."""
    with np.errstate(over="ignore", invalid="ignore"):
        return fit_varpro(*args, **kwargs)


def _axes_of(theta, N, mu0):
    _, L = unpack_theta(theta, N, mu0)
    return np.sqrt(np.linalg.eigvalsh(L @ L.T))


def linear_cv_score(z_hat, y_hat, basis, theta, e_hat=None,
                    n_folds=5, seed=0):
    """Linear-stage K-fold CV score of a model at a given theta: the
    linear coefficients are refit leave-fold-out at that theta and
    scored on the held folds; returns the relative whitened residual
    over all equations. Public because it scores ANY model -- fitted
    here or supplied a priori -- against the probe data at zero
    additional fits (e.g. score the a-priori ellipsoid per target for a
    field-wide QC map before fitting anything)."""
    k = len(y_hat)
    y_norm = max(float(np.linalg.norm(y_hat)), 1e-300)
    with np.errstate(over="ignore", invalid="ignore"):
        Phi = basis(theta).values()
    if not np.all(np.isfinite(Phi)):
        return np.inf
    A = z_hat @ Phi.T
    X = np.hstack([A, z_hat @ e_hat.T]) if e_hat is not None else A
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(k), max(2, min(n_folds, k // 2)))
    resids = []
    for f in folds:
        train = np.setdiff1d(np.arange(k), f)
        coef, *_ = np.linalg.lstsq(X[train], y_hat[train], rcond=None)
        resids.append(y_hat[f] - X[f] @ coef)
    return float(np.linalg.norm(np.concatenate(resids)) / y_norm)


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class ProbeFitConfig:
    """Declares the candidate axes and the search policy for
    fit_from_probes. Numerical hygiene stays in VarProOptions (the
    `varpro` field)."""

    mu: str = "fixed_then_release"
    """"fixed" pins mu at mu0 throughout; "free" fits it from the start
    (inits seeded at mu0); "fixed_then_release" (default) pins it for
    the stream, then releases the winning level's fits under the
    tie-break guard."""

    mode_levels: Optional[List[int]] = None
    """Nested mode ladder: fit complete shells up to each listed
    oscillator level, ascending, with counting-rule admissibility and
    mode_patience stopping. Exactly one of `mode_levels`, `mode_sets`,
    or fit_from_probes' explicit `modes` argument must be given."""

    mode_sets: Optional[List[List[tuple]]] = None
    """Explicit nested mode-set ladder: a list of (p, ell, m) lists,
    ascending in size, each a superset of the previous (resolved to
    mode_policy.ExplicitLadder)."""

    mode_policy: Optional[object] = None
    """A mode_policy.ModePolicy instance -- the general ladder axis
    (docs/mode-policy-plan.md): shells, ell-capped wedges,
    radial-first prefixes, adaptive MarginGreedy, or any user policy.
    The policy PROPOSES mode sets; every structural guard and all
    selection semantics stay in this engine. Exactly one of
    {mode_policy, mode_levels, mode_sets, the `modes` argument}."""

    n_rungs: int = 6
    """Log-spaced circle scales from the local mesh spacing at mu0 to a
    circle containing the whole batch."""

    window_shape_rungs: bool = True
    """Also ladder scaled copies of the window's own shape
    (init_dictionary.window_shape) over the same scale range."""

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
    """Linear-stage CV score (linear_cv_score at the fitted theta)."""
    axes: np.ndarray
    """(N,) fitted 1-sigma semi-axes, sqrt(eig(L L^T))."""
    success: bool
    n_iterations: int
    admissible: bool = True
    """False if the fit violates window containment (major semi-axis
    beyond the window radius, or a released center outside the window):
    impossible-or-runaway by the conservativeness of the window."""


@dataclass
class ProbeFitResult:
    """The winning candidate, plus the full audit trail."""

    mu: np.ndarray
    """(N,) fitted (or pinned) center."""
    L: np.ndarray
    """(N, N) lower-triangular Cholesky factor of the fitted Sigma."""
    c: np.ndarray
    """(n_modes,) smooth kernel coefficients: kernel(x) = sum c_i phi_i."""
    s: np.ndarray
    """(num_extra,) spike coefficient(s); empty if spike_index was None."""
    theta: np.ndarray
    """Winning theta in its own encoding (see mu_fixed)."""
    modes: List
    """The winning mode set -- the (p, ell, m) list c corresponds to."""
    mu_fixed: bool
    released: bool
    """True if a release-stage fit won."""
    score: float
    score_kind: str
    """"cv" (currently always; field kept for future scoring variants)."""
    stop_reason: str
    """"target" | "mode_patience" | "exhausted"."""
    winner: int
    """Index into candidates."""
    candidates: List[CandidateFit] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    """Mode levels skipped by the counting rule."""


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_from_probes(x, m2_diag, z, y, mu0, modes=None,
                    spike_index=None, sigma0=None,
                    config=None, verbose=False, target_mass=None):
    """Fit a target function from raw probe data. See the module
    docstring for the architecture. Parameters:

    x : (N, K) raw batch coordinates (caller-selected support).
    m2_diag : (K,) lumped masses on the batch.
    z : (k, K) raw probe fields restricted to the batch.
    y : (k,) the target's raw responses, y_l = <target, z_l>.
    mu0 : (N,) initial center (e.g. the target's own dof location).
    modes : explicit (p, ell, m) list -- required iff config.mode_levels
        is None.
    spike_index : index of the target's own point within the batch;
        builds the one-hot spike extra basis. None disables the spike --
        justified only for targets known to not be spike-dominated.
    sigma0 : optional (N, N) SPD initial covariance, tried FIRST (e.g.
        an a-priori estimate, or
        probe_moments.raw_moments(probe_moments.backproject(z, y), ...)).
    config : ProbeFitConfig.
    verbose : print the candidate table.
    target_mass : the target's own lumped mass (M1)_rho,rho in the
        operator-row application. Default None infers it as
        m2_diag[spike_index] (exact for the square equal-mass case; 1.0
        with no spike). The whitened design matrix is column-equilibrated
        inside the inner solve, so this choice rescales ONLY the returned
        (c, s) -- theta, CV scores, and the whole selection are invariant
        -- but callers with M1 != M2 must pass it to get correctly scaled
        coefficients.
    """
    cfg = config if config is not None else ProbeFitConfig()
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
    policy = _resolve_mode_policy(cfg, modes)

    if target_mass is None:
        target_mass = (float(m2_diag[spike_index])
                       if spike_index is not None else 1.0)
    else:
        target_mass = float(target_mass)
    n_extra = 1 if spike_index is not None else 0
    if spike_index is not None:
        E = np.zeros((1, K))
        E[0, spike_index] = 1.0
        e_hat = whiten_extra(E, target_mass, m2_diag)
    else:
        e_hat = None

    z_hat = whiten_probes(z, m2_diag)
    y_hat = whiten_data(y, target_mass)
    P_fix = theta_size(N, mu0)

    # --- geometry: ladder scales and the window shape ---------------------
    h_local = local_spacing(x, mu0)
    r_max = window_radius(x, mu0)
    radii = ladder_scales(h_local, r_max, cfg.n_rungs)

    inits = []
    if sigma0 is not None:
        inits.append(("sigma0", theta_from_L(np.linalg.cholesky(sigma0))))
    if cfg.window_shape_rungs:
        inits += window_rungs(radii, window_shape(x, m2_diag))
    inits += circle_rungs(radii, N)

    ladder_fixed = cfg.mu in ("fixed", "fixed_then_release")
    enc_mu0 = mu0 if ladder_fixed else None
    rng = np.random.default_rng(cfg.seed)

    def basis(mlist, mu0_or_none):
        return whitened_basis(N, x, target_mass, m2_diag,
                              mlist, mu0=mu0_or_none)

    def admissible_of(theta, enc):
        axes = _axes_of(theta, N, enc)
        ok = bool(axes.max() <= r_max)
        if enc is None:
            ok = ok and bool(np.linalg.norm(theta[:N] - mu0) <= r_max)
        return ok

    def run_candidate(label, mlabel, mlist, th0, enc, released, b):
        res = _quiet_fit(z_hat, y_hat, b, th0,
                         e_hat=e_hat, options=cfg.varpro)
        return CandidateFit(
            label=label, modes_label=mlabel, n_modes=len(mlist),
            released=released, theta_init=np.asarray(th0, dtype=float),
            theta=res.theta, c=res.c, s=res.s, cost=res.cost,
            score=linear_cv_score(z_hat, y_hat, b, res.theta,
                                  e_hat=e_hat, n_folds=cfg.cv_folds,
                                  seed=cfg.seed),
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

    def make_margin_scorer(theta, active_modes):
        """Exact one-step SSE reductions for candidate modes at a fixed
        theta -- the adaptive policies' feedback. A projection against
        the active whitened design, never a refit."""
        basis_a = whitened_basis(N, x, target_mass, m2_diag,
                                 active_modes, mu0=enc_mu0)
        with np.errstate(over="ignore", invalid="ignore"):
            Phi = basis_a(theta).values()
        X = z_hat @ Phi.T
        if e_hat is not None:
            X = np.hstack([X, z_hat @ e_hat.T])
        if not np.all(np.isfinite(X)):
            return None, 0.0
        coef, *_ = np.linalg.lstsq(X, y_hat, rcond=None)
        r = y_hat - X @ coef
        Q = np.linalg.qr(X)[0]

        def margin_profit(cand_modes):
            basis_c = whitened_basis(N, x, target_mass, m2_diag,
                                     [tuple(m) for m in cand_modes],
                                     mu0=enc_mu0)
            with np.errstate(over="ignore", invalid="ignore"):
                Ac = z_hat @ basis_c(theta).values().T
            if not np.all(np.isfinite(Ac)):
                return np.zeros(Ac.shape[1])
            Ac = Ac - Q @ (Q.T @ Ac)
            num = (Ac.T @ r) ** 2
            den = np.einsum("ij,ij->j", Ac, Ac)
            out = np.zeros(len(num))
            good = den > 1e-30 * max(float(den.max()), 1e-300)
            out[good] = num[good] / den[good]
            return out

        return margin_profit, float(r @ r)

    # --- the fixed-encoding (or free, if mu="free") candidate stream ------
    candidates = []
    skipped = []
    history = []
    modes_of = {}
    stop_reason = "exhausted"
    best_score = np.inf
    patience_left = cfg.mode_patience
    prev_level_best_theta = None
    stopped = False
    margin_profit, resid_norm2 = None, 0.0
    last_fit_modes = None

    while len(history) < MAX_PROPOSALS:
        prop = policy.propose(ModeSearchContext(
            N=N, k=k, n_extra=n_extra, P=P_fix, history=history,
            margin_profit=margin_profit, resid_norm2=resid_norm2))
        if prop is None:
            break
        mlabel, mlist = prop
        mlist = [tuple(m) for m in mlist]
        if any(rec.label == mlabel for rec in history):
            raise ValueError(f"mode policy reused label {mlabel!r}")
        if (last_fit_modes is not None
                and not set(last_fit_modes) <= set(mlist)):
            raise ValueError(
                f"mode policy proposal {mlabel!r} does not contain the "
                f"previously fitted set (nested growth is the contract)")
        if k < 2 * (len(mlist) + n_extra + P_fix):
            skipped.append(mlabel)
            history.append(LevelRecord(mlabel, mlist, True, None))
            continue
        if patience_left <= 0:
            stop_reason = "mode_patience"
            break
        modes_of[mlabel] = mlist
        b = basis(mlist, enc_mu0)
        level_inits = list(inits)
        if prev_level_best_theta is not None:
            jitter = 0.05 * rng.standard_normal(prev_level_best_theta.shape)
            level_inits.insert(0, (f"warm({candidates[-1].modes_label})",
                                   prev_level_best_theta + jitter))
        level_start = len(candidates)
        for label, th_init in level_inits:
            if label.startswith("warm"):
                th0 = th_init          # already in the stream's encoding
            else:
                th0 = th_init if ladder_fixed else release_mu(th_init, mu0)
            cand = run_candidate(label, mlabel, mlist, th0, enc_mu0,
                                 False, b)
            candidates.append(cand)
            if hit_target(cand):
                stop_reason = "target"
                stopped = True
                break
        level_cands = candidates[level_start:]
        level_adm = [c for c in level_cands if c.admissible]
        level_winner = None
        if level_adm:
            i_best = min((i for i in range(level_start, len(candidates))
                          if candidates[i].admissible),
                         key=lambda i: candidates[i].score)
            level_winner = candidates[i_best]
            if level_winner.score < best_score:
                best_score = level_winner.score
                patience_left = cfg.mode_patience
                prev_level_best_theta = level_winner.theta
            else:
                patience_left -= 1
        else:
            patience_left -= 1
        history.append(LevelRecord(mlabel, mlist, False, level_winner))
        last_fit_modes = mlist
        if level_winner is not None:
            margin_profit, resid_norm2 = make_margin_scorer(
                level_winner.theta, mlist)
        if stopped:
            break

    if not candidates:
        raise ValueError(
            f"no mode set passed the counting rule k >= 2*(m + "
            f"{n_extra} + {P_fix}) at k={k} "
            f"(skipped proposals: {skipped or 'none'})")

    winner = select(candidates)

    # --- guarded release stage --------------------------------------------
    if cfg.mu == "fixed_then_release" and stop_reason != "target":
        win_mlabel = candidates[winner].modes_label
        mlist = modes_of[win_mlabel]
        b_free = basis(mlist, None)
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
                                 b_free)
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

    return ProbeFitResult(mu=np.array(mu_out, dtype=float, copy=True),
                          L=L_out, c=win.c, s=win.s, theta=win.theta,
                          modes=modes_of[win.modes_label],
                          mu_fixed=not win.released and ladder_fixed,
                          released=win.released,
                          score=win.score, score_kind="cv",
                          stop_reason=stop_reason, winner=winner,
                          candidates=candidates, skipped=skipped)
