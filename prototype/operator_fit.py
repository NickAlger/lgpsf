"""Whole-operator fitting: the per-target probe fit (probe_fit.py) run
over the rows of an implicitly available operator, per the agreed design
in docs/operator-api-plan.md.

THE FITTED OBJECT is not a matrix. It is

    H~  =  M1 Phi~ M2  +  M1 S

a sum of two objects of DIFFERENT TYPES -- the same distinction the
whitening derivation is built on (X-valued quadrature objects vs
X'-valued discrete corrections):

  - Phi~, a SEMI-DISCRETE CONTINUUM KERNEL: for each fitted row rho a
    genuine function of the source coordinate,
    Phi~(rho, x) = sum_i c_i(rho) phi_i(x; theta_rho). Rectangular by
    nature -- evaluable between arbitrary point sets, exportable to
    ellipsoid_tree, meaningful on other meshes.
  - S, a SPARSE DOF-TIED DISCRETE CORRECTION: today diag(s). Square by
    nature -- it was DEFINED as the part of the PSF the mesh cannot
    resolve, so it has no off-grid meaning. The API exposes its Dirac
    mass, spike_measure(rho) = m_rho * s_rho; a consumer who must
    regularize it (delta on a finer grid, narrow Gaussian of width h,
    drop it) makes that modeling decision themselves, visibly.

Every helper below is typed to the component(s) it touches (the helper
table in the plan doc). Symmetry and SPD are ASSEMBLY POLICIES
(assemble_sparse's symmetrize option), not fit properties: row fits do
not produce H~ = H~^T, and rows-as-is vs (A + A^T)/2 vs clamp/shift for
an SPD preconditioner are consumer decisions with consumer-specific
right answers.

THE PER-ROW PROTOCOL: gate -> window -> fit_from_probes -> baseline
guard -> status.

  - Windows are derived here (the operator layer owns one kd-tree,
    built once): the ball of radius tau_window * (largest 1-sigma axis
    of sigma[rho]) around mu0[rho]; explicit per-row index lists as an
    override. The user's sigma field is their BEST GUESS at the actual
    bump shape (NOT required to be conservative; it also seeds the
    sigma0 candidate rung and the baseline fit below); conservatism is
    supplied by tau_window inflation, and fit_from_probes' window-
    containment admissibility guard is only as valid as tau_window is
    conservative (default 10: a sigma underestimating by 3x still
    leaves ~3 true sigmas of margin).
  - The row fit receives the true target mass m1_diag[rho] (probe_fit's
    target_mass), so M1 != M2 operators get correctly scaled
    coefficients.
  - BASELINE GUARD (always on): a plain linear LG fit at sigma[rho],
    pinned at mu0[rho] -- one lstsq per counting-rule-admissible mode
    set, no LM -- is CV-scored with the same folds as the search, and
    the searched fit ships only if it STRICTLY beats the best baseline;
    else the baseline ships (status "fallback_baseline"). Consequence:
    the method is never worse than the a-priori-Gaussian status quo it
    replaces, by construction.
  - Per-row status: "fit" | "gated_out" | "fallback_baseline" |
    "failed" (ungated rows get status, not silence).

Rows are independent by construction (row rho touches HV[rho, :] and
V[window, :] only): the plain Python loop here is the C++/MPI row-block
port reference. Future field-level ideas (neighbor warm starts,
smoothed-theta seeds) enter as candidate injection into the per-row
stream -- a sweep-order policy, not an API change.

GEOMETRY QUERIES go through ellipsoid_tree (`pip install
ellipsoid-tree` -- the same library the C++ port links), confined to
this layer the way scipy is: spatial indexing, never reference math.
Window derivation is a BallTree ball query; assemble_sparse gets its
whole sparsity pattern -- which columns fall in which rows' fitted
tau-ellipsoids, for ALL rows at once -- from ONE collision_pairs
dual tree descent (points-tree x ellipsoid-tree). ellipsoid_tree
takes points as rows, (K, N), so the boundary transposes -- the same
layout flip docs/design-notes.md already prescribes for the Eigen side.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
import ellipsoid_tree as et

from ellipsoid_transform import release_mu, theta_size
from init_dictionary import theta_from_L
from lg_ellipsoid_feature import eval_feature
from lg_functions import modes_up_to_level
from probe_fit import ProbeFitConfig, fit_from_probes, linear_cv_score
from whitening import whiten_data, whiten_extra, whiten_probes, whitened_basis


# ---------------------------------------------------------------------------
# Config and result types
# ---------------------------------------------------------------------------

@dataclass
class OperatorFitConfig:
    """Operator-level knobs, plus the per-row search policy."""

    tau_window: float = 10.0
    """Window inflation: window radius = tau_window * (largest 1-sigma
    axis of sigma[rho]). This is where conservatism comes from -- the
    admissibility guard's validity travels with it (see module
    docstring)."""

    spike: bool = True
    """Model the diagonal spike (requires the square dof context:
    x_rows is None, so row dof rho IS column dof rho). Set False for
    rectangular fits or operators known to not be spike-dominated."""

    row: ProbeFitConfig = field(default_factory=ProbeFitConfig)
    """The per-row candidate-stream policy (probe_fit.ProbeFitConfig).
    Its cv_folds/seed also drive the baseline guard's CV score, so the
    two scores are computed on identical folds."""


@dataclass
class OperatorFit:
    """The parametric two-component operator approximation, as padded
    flat arrays (C++/MPI-friendly). ~(P + m + 1) floats per row: the
    parametric form IS the compressed operator; every matrix format is
    a decompression of it (see the helpers below).

    Rows without a shipped model (status "gated_out"/"failed") hold NaN
    in theta/mu/L/s/score and mode_set_id == -1."""

    N: int
    x_cols: np.ndarray
    """(N, K_all) column-dof coordinates (kept for the helpers)."""
    x_rows: Optional[np.ndarray]
    """(N, R_all) row-dof coordinates, or None for the square context
    (row dofs are the column dofs)."""
    m1_diag: np.ndarray
    """(R_all,) row masses (kept for convenience/helpers)."""
    m2_diag: np.ndarray
    """(K_all,) column masses."""
    spike: bool
    """Echo of config.spike: whether the S component exists."""
    theta: np.ndarray
    """(R_all, P) fitted theta, ALWAYS in the free-mu encoding
    (pinned-mu winners are stored via release_mu, which is exact), so
    every row decodes uniformly with mu0=None."""
    mu: np.ndarray
    """(R_all, N) fitted (or pinned) centers."""
    L: np.ndarray
    """(R_all, N, N) lower-triangular Cholesky factors; Sigma = L L^T."""
    c: np.ndarray
    """(R_all, m_max) smooth coefficients, zero-padded; decode the
    active prefix via mode_set_id/mode_sets."""
    mode_set_id: np.ndarray
    """(R_all,) index into mode_sets; -1 = no model."""
    mode_sets: List[List[Tuple[int, int, int]]]
    """The distinct (p, ell, m) mode lists appearing across rows."""
    s: np.ndarray
    """(R_all,) additive spike coefficients (0.0 where the spike is
    disabled; the convention is s ON TOP of the unmodified LG smooth
    part at the diagonal -- see the plan doc)."""
    score: np.ndarray
    """(R_all,) CV score of the shipped model."""
    baseline_score: np.ndarray
    """(R_all,) CV score of the baseline (linear fit at sigma[rho],
    pinned mu0) -- the searched-vs-status-quo margin is
    baseline_score - score."""
    stop_reason: np.ndarray
    """(R_all,) the row search's stop reason ("target" |
    "mode_patience" | "exhausted" | "search_infeasible" | "")."""
    released: np.ndarray
    """(R_all,) bool: True where the shipped model's mu was optimized."""
    status: np.ndarray
    """(R_all,) "fit" | "gated_out" | "fallback_baseline" | "failed"."""
    config: OperatorFitConfig
    """Provenance echo."""
    failures: Dict[int, str] = field(default_factory=dict)
    """Row index -> error message, for status "failed" rows."""

    def row_modes(self, rho):
        """The (p, ell, m) list row rho's c prefix corresponds to."""
        sid = int(self.mode_set_id[rho])
        if sid < 0:
            raise ValueError(f"row {rho} has no fitted model "
                             f"(status={self.status[rho]!r})")
        return self.mode_sets[sid]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def _linear_fit(z_hat, y_hat, b_eval, theta, e_hat):
    """Full-data linear coefficients at a fixed theta (the baseline's
    one lstsq): returns (c, s) with s empty if e_hat is None. Mirrors
    linear_cv_score's design matrix exactly."""
    Phi = b_eval(theta)
    X = z_hat @ Phi.T
    if e_hat is not None:
        X = np.hstack([X, z_hat @ e_hat.T])
    coef, *_ = np.linalg.lstsq(X, y_hat, rcond=None)
    n = Phi.shape[0]
    return coef[:n], coef[n:]


def fit_operator(x_cols, m1_diag, m2_diag, V, HV, sigma, mu0=None,
                 modes=None, x_rows=None, rows=None, windows=None,
                 config=None, verbose=False):
    """Fit the parametric approximation H~ = M1 Phi~ M2 + M1 S from raw
    probe data, one probe_fit search per row. See the module docstring
    for the protocol. Parameters:

    x_cols : (N, K_all) column-dof coordinates.
    m1_diag : (R_all,) row masses; m2_diag : (K_all,) column masses.
    V : (K_all, k) raw random probes; HV : (R_all, k) raw responses.
    sigma : (R_all, N, N) best-guess bump covariances (a single (N, N)
        broadcasts to all rows). NOT required to be conservative --
        tau_window supplies the conservatism.
    mu0 : (R_all, N) initial centers; default = the row dofs' own
        coordinates (pass explicitly for e.g. advected kernels).
    modes : explicit (p, ell, m) list -- required iff
        config.row.mode_levels is None (same contract as
        fit_from_probes).
    x_rows : (N, R_all) row-dof coordinates for a rectangular fit;
        None (default) = square context, row dofs are the column dofs
        (required for the spike).
    rows : row gate -- an index array or boolean mask of rows to fit;
        None = all. Ungated rows get status "gated_out".
    windows : optional length-R_all sequence overriding window
        derivation: entry rho is an index array into the columns, or
        None to derive that row's window as usual.
    config : OperatorFitConfig.
    verbose : print one line per fitted row.
    """
    cfg = config if config is not None else OperatorFitConfig()
    x_cols = np.asarray(x_cols, dtype=float)
    m1_diag = np.asarray(m1_diag, dtype=float)
    m2_diag = np.asarray(m2_diag, dtype=float)
    V = np.asarray(V, dtype=float)
    HV = np.asarray(HV, dtype=float)

    N, K_all = x_cols.shape
    R_all = HV.shape[0]
    k = V.shape[1]
    if m2_diag.shape != (K_all,) or V.shape != (K_all, k):
        raise ValueError(f"column-side shape mismatch: x_cols {x_cols.shape},"
                         f" m2_diag {m2_diag.shape}, V {V.shape}")
    if m1_diag.shape != (R_all,) or HV.shape != (R_all, k):
        raise ValueError(f"row-side shape mismatch: m1_diag {m1_diag.shape},"
                         f" HV {HV.shape}, k={k}")
    if x_rows is not None:
        x_rows = np.asarray(x_rows, dtype=float)
        if x_rows.shape != (N, R_all):
            raise ValueError(f"x_rows {x_rows.shape} != (N={N}, R={R_all})")
        if cfg.spike:
            raise ValueError("the spike requires the square dof context "
                             "(x_rows is None); pass "
                             "OperatorFitConfig(spike=False) for a "
                             "rectangular fit")
    elif R_all != K_all:
        raise ValueError(f"square context (x_rows=None) needs R == K, got "
                         f"R={R_all}, K={K_all}; pass x_rows for a "
                         f"rectangular fit")
    if mu0 is None:
        mu0 = (x_rows if x_rows is not None else x_cols).T.copy()
    else:
        mu0 = np.asarray(mu0, dtype=float)
        if mu0.shape != (R_all, N):
            raise ValueError(f"mu0 {mu0.shape} != (R={R_all}, N={N})")
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape == (N, N):
        sigma = np.broadcast_to(sigma, (R_all, N, N))
    if sigma.shape != (R_all, N, N):
        raise ValueError(f"sigma {sigma.shape} != (R={R_all}, N={N}, N={N})")
    if (cfg.row.mode_levels is None) == (modes is None):
        raise ValueError("provide exactly one of `modes` (explicit list) "
                         "or config.row.mode_levels (nested ladder)")
    if windows is not None and len(windows) != R_all:
        raise ValueError(f"windows must have one entry per row "
                         f"({R_all}), got {len(windows)}")

    gate = np.zeros(R_all, dtype=bool)
    if rows is None:
        gate[:] = True
    else:
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if rows.shape != (R_all,):
                raise ValueError(f"boolean rows mask {rows.shape} != "
                                 f"({R_all},)")
            gate = rows.copy()
        else:
            gate[rows] = True

    # the stream's fixed-mu theta size drives both counting rules below
    P_fix = N + N * (N - 1) // 2 + N            # == theta_size(N, mu0=any)
    P_free = theta_size(N, None)
    if cfg.row.mode_levels is not None:
        base_sets = [modes_up_to_level(N, lev)
                     for lev in sorted(cfg.row.mode_levels)]
    else:
        base_sets = [list(modes)]
    n_extra_cfg = 1 if cfg.spike else 0
    P_stream = P_free if cfg.row.mu == "free" else P_fix

    col_tree = et.BallTree(x_cols.T, np.zeros(K_all))

    theta_arr = np.full((R_all, P_free), np.nan)
    mu_arr = np.full((R_all, N), np.nan)
    L_arr = np.full((R_all, N, N), np.nan)
    s_arr = np.full(R_all, np.nan)
    score_arr = np.full(R_all, np.nan)
    base_arr = np.full(R_all, np.nan)
    stop_arr = np.array([""] * R_all, dtype=object)
    released_arr = np.zeros(R_all, dtype=bool)
    status_arr = np.array(["gated_out"] * R_all, dtype=object)
    c_by_row = [None] * R_all
    set_id_arr = np.full(R_all, -1, dtype=int)
    mode_sets: List[List[Tuple[int, int, int]]] = []
    set_ids: Dict[tuple, int] = {}
    failures: Dict[int, str] = {}

    def register_modes(mlist):
        key = tuple(tuple(m) for m in mlist)
        if key not in set_ids:
            set_ids[key] = len(mode_sets)
            mode_sets.append(list(mlist))
        return set_ids[key]

    for rho in range(R_all):
        if not gate[rho]:
            continue
        try:
            mu_rho = mu0[rho]
            L_prior = np.linalg.cholesky(sigma[rho])   # validates SPD

            # --- window --------------------------------------------------
            if windows is not None and windows[rho] is not None:
                idx = np.sort(np.asarray(windows[rho], dtype=int))
            else:
                radius = cfg.tau_window * float(np.linalg.norm(L_prior, 2))
                idx = np.sort(np.array(
                    col_tree.collisions(et.Ball(mu_rho, radius)), dtype=int))
            if idx.size < 2:
                raise ValueError(f"window has {idx.size} points "
                                 f"(mu0={mu_rho}, sigma too small or "
                                 f"tau_window too tight?)")

            spike_pos = None
            if cfg.spike:
                pos = int(np.searchsorted(idx, rho))
                if pos < idx.size and idx[pos] == rho:
                    spike_pos = pos
                # else: the row dof fell outside its own window (only
                # possible with a far-off explicit mu0); the shipped
                # model then simply has no spike (s stays 0).

            x_w = x_cols[:, idx]
            m2_w = m2_diag[idx]
            z = V[idx, :].T                       # (k, K_w)
            y = HV[rho]
            m1_rho = float(m1_diag[rho])
            n_extra = 1 if spike_pos is not None else 0

            # --- baseline: linear LG fit at sigma[rho], pinned mu0 --------
            z_hat = whiten_probes(z, m2_w)
            y_hat = whiten_data(y, m1_rho)
            if spike_pos is not None:
                E = np.zeros((1, idx.size))
                E[0, spike_pos] = 1.0
                e_hat = whiten_extra(E, m1_rho, m2_w)
            else:
                e_hat = None
            theta_base = theta_from_L(L_prior)
            base_score, base_c, base_s, base_modes = np.inf, None, None, None
            for mlist in base_sets:
                if k < 2 * (len(mlist) + n_extra + P_fix):
                    continue
                b_eval = whitened_basis(N, x_w, m1_rho, m2_w, mlist,
                                        mu0=mu_rho)[0]
                sc = linear_cv_score(z_hat, y_hat, b_eval, theta_base,
                                     e_hat=e_hat, n_folds=cfg.row.cv_folds,
                                     seed=cfg.row.seed)
                if sc < base_score:
                    base_score = sc
                    base_c, base_s = _linear_fit(z_hat, y_hat, b_eval,
                                                 theta_base, e_hat)
                    base_modes = mlist
            if base_modes is None:
                raise ValueError(
                    f"no mode set passes the counting rule k >= 2*(m + "
                    f"{n_extra} + {P_fix}) at k={k} (smallest set: "
                    f"{min(len(m) for m in base_sets)} modes)")

            # --- the searched fit ------------------------------------------
            searchable = any(k >= 2 * (len(m) + n_extra + P_stream)
                             for m in base_sets)
            res = None
            if searchable:
                res = fit_from_probes(x_w, m2_w, z, y, mu_rho, modes=modes,
                                      spike_index=spike_pos,
                                      sigma0=sigma[rho], config=cfg.row,
                                      target_mass=m1_rho)

            # --- baseline guard --------------------------------------------
            if res is not None and res.score < base_score:
                status_arr[rho] = "fit"
                theta_free = (res.theta if not res.mu_fixed
                              else release_mu(res.theta, res.mu))
                mu_arr[rho] = res.mu
                L_arr[rho] = res.L
                c_by_row[rho] = res.c
                s_arr[rho] = float(res.s[0]) if res.s.size else 0.0
                score_arr[rho] = res.score
                set_id_arr[rho] = register_modes(res.modes)
                released_arr[rho] = not res.mu_fixed
            else:
                status_arr[rho] = "fallback_baseline"
                theta_free = release_mu(theta_base, mu_rho)
                mu_arr[rho] = mu_rho
                L_arr[rho] = L_prior
                c_by_row[rho] = base_c
                s_arr[rho] = float(base_s[0]) if base_s.size else 0.0
                score_arr[rho] = base_score
                set_id_arr[rho] = register_modes(base_modes)
            theta_arr[rho] = theta_free
            base_arr[rho] = base_score
            stop_arr[rho] = (res.stop_reason if res is not None
                             else "search_infeasible")
            if verbose:
                sc_s = f"{res.score:.4g}" if res is not None else "---"
                print(f"row {rho}: window {idx.size} pts, searched {sc_s} "
                      f"({stop_arr[rho]}), baseline {base_score:.4g} "
                      f"-> {status_arr[rho]}")
        except (ValueError, np.linalg.LinAlgError) as exc:
            status_arr[rho] = "failed"
            failures[rho] = str(exc)
            if verbose:
                print(f"row {rho}: FAILED ({exc})")

    m_max = max((len(m) for m in mode_sets), default=0)
    c_arr = np.zeros((R_all, m_max))
    for rho, c in enumerate(c_by_row):
        if c is not None:
            c_arr[rho, :len(c)] = c

    return OperatorFit(N=N, x_cols=x_cols, x_rows=x_rows,
                       m1_diag=m1_diag, m2_diag=m2_diag, spike=cfg.spike,
                       theta=theta_arr, mu=mu_arr, L=L_arr, c=c_arr,
                       mode_set_id=set_id_arr, mode_sets=mode_sets,
                       s=s_arr, score=score_arr, baseline_score=base_arr,
                       stop_reason=stop_arr, released=released_arr,
                       status=status_arr, config=cfg, failures=failures)


# ---------------------------------------------------------------------------
# Helpers, each typed to the component(s) it touches
# ---------------------------------------------------------------------------

def _model_rows(fit):
    """Rows carrying a shipped model ("fit" or "fallback_baseline")."""
    return np.nonzero((fit.status == "fit")
                      | (fit.status == "fallback_baseline"))[0]


def eval_kernel(fit, rows, x_query):
    """[smooth component] Phi~(rho, x) at arbitrary query points:
    rectangular by nature, no masses, no spike. x_query: (N, *batch);
    returns (len(rows), *batch). Raises on rows without a model."""
    x_query = np.asarray(x_query, dtype=float)
    out = []
    for rho in np.asarray(rows, dtype=int):
        modes = fit.row_modes(rho)
        c = fit.c[rho, :len(modes)]
        vals = eval_feature(fit.theta[rho], fit.N, x_query, modes)
        out.append(np.tensordot(c, vals, axes=(0, 0)))
    return np.stack(out, axis=0)


def eval_entries(fit, rows, cols):
    """[both components] Paired entries of H~ in the dof context:
    H~[rho, j] = m1[rho] m2[j] Phi~(rho, x_cols[:, j]),
    plus m1[rho] s[rho] iff j == rho (the spike dof; square context).
    rows, cols: equal-length index arrays; returns their (n,) values."""
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.shape != cols.shape:
        raise ValueError(f"rows {rows.shape} and cols {cols.shape} differ")
    out = np.empty(rows.shape, dtype=float)
    for rho in np.unique(rows):
        mask = rows == rho
        j = cols[mask]
        kern = eval_kernel(fit, [rho], fit.x_cols[:, j])[0]
        vals = fit.m1_diag[rho] * fit.m2_diag[j] * kern
        if fit.spike:
            vals[j == rho] += fit.m1_diag[rho] * fit.s[rho]
        out[mask] = vals
    return out


def matvec(fit, v):
    """[both components] H~ @ v with zero assembly: exact application
    of the parametric operator (dense per-row kernel evaluation; the
    Gaussian tail supplies the decay -- assemble_sparse is the
    truncated alternative). v: (K_all,) or (K_all, q); rows without a
    model contribute zero rows. Returns (R_all,) or (R_all, q)."""
    v = np.asarray(v, dtype=float)
    R_all = fit.m1_diag.shape[0]
    out = np.zeros((R_all,) + v.shape[1:])
    for rho in _model_rows(fit):
        kern = eval_kernel(fit, [rho], fit.x_cols)[0]
        out[rho] = (fit.m1_diag[rho] * (fit.m2_diag * kern)) @ v
        if fit.spike:
            out[rho] += fit.m1_diag[rho] * fit.s[rho] * v[rho]
    return out


def to_linear_operator(fit):
    """[both components] scipy LinearOperator view of matvec (e.g. for
    QC against held-out probes with zero assembly)."""
    from scipy.sparse.linalg import LinearOperator
    shape = (fit.m1_diag.shape[0], fit.m2_diag.shape[0])
    return LinearOperator(shape, matvec=lambda v: matvec(fit, v),
                          matmat=lambda B: matvec(fit, B), dtype=float)


def assemble_sparse(fit, tau, symmetrize=None):
    """[both components] Sparse matrix decompression: per modeled row,
    smooth entries on the columns within Mahalanobis distance tau of
    the fitted ellipsoid (||L^{-1}(x_j - mu)|| <= tau), plus the spike
    on the diagonal. The whole sparsity pattern -- which columns fall
    in which rows' tau-ellipsoids, all rows at once -- comes from ONE
    ellipsoid_tree collision_pairs dual descent of the column-point
    tree against the fitted-ellipsoid tree. symmetrize: None (rows
    as-is) or "average" ((A + A^T)/2; square context only) -- an
    assembly POLICY, chosen by the consumer, not a fit property.
    Returns scipy CSR."""
    R_all = fit.m1_diag.shape[0]
    K_all = fit.m2_diag.shape[0]
    model = _model_rows(fit)
    rows_i, cols_i, vals = [], [], []
    if model.size:
        _, Sigma = ellipsoid_field(fit)
        etree = et.EllipsoidTree(fit.mu[model], Sigma[model], tau)
        ptree = et.BallTree(fit.x_cols.T, np.zeros(K_all))
        pairs = np.asarray(et.collision_pairs(ptree, etree),
                           dtype=int).reshape(-1, 2)
        order = np.argsort(pairs[:, 1], kind="stable")
        pcols, pells = pairs[order, 0], pairs[order, 1]
        starts = np.searchsorted(pells, np.arange(model.size))
        ends = np.searchsorted(pells, np.arange(model.size) + 1)
        for e, rho in enumerate(model):
            j = np.sort(pcols[starts[e]:ends[e]])
            if j.size:
                kern = eval_kernel(fit, [rho], fit.x_cols[:, j])[0]
                rows_i.append(np.full(j.size, rho))
                cols_i.append(j)
                vals.append(fit.m1_diag[rho] * fit.m2_diag[j] * kern)
            if fit.spike:
                rows_i.append(np.array([rho]))
                cols_i.append(np.array([rho]))
                vals.append(np.array([fit.m1_diag[rho] * fit.s[rho]]))
    if rows_i:
        A = scipy.sparse.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows_i), np.concatenate(cols_i))),
            shape=(R_all, K_all)).tocsr()
    else:
        A = scipy.sparse.csr_matrix((R_all, K_all))
    if symmetrize == "average":
        if fit.x_rows is not None or R_all != K_all:
            raise ValueError("symmetrize needs the square dof context")
        A = (A + A.T) * 0.5
    elif symmetrize is not None:
        raise ValueError(f"unknown symmetrize policy: {symmetrize!r}")
    return A


def ellipsoid_field(fit):
    """[smooth component] (mu (R, N), Sigma (R, N, N)) arrays of the
    fitted ellipsoids (Sigma = L L^T; NaN where no model) -- the
    ellipsoid_tree feed, keeping lgpsf dependency-free."""
    Sigma = np.einsum("rij,rkj->rik", fit.L, fit.L)
    return fit.mu.copy(), Sigma


def qc_map(fit, V_qc, HV_qc):
    """[both components] Per-row relative residual against held-out
    probes -- the scorecard: ||H~[rho] V_qc - HV_qc[rho]|| /
    ||HV_qc[rho]||. NaN for rows without a model."""
    pred = matvec(fit, np.asarray(V_qc, dtype=float))
    HV_qc = np.asarray(HV_qc, dtype=float)
    out = np.full(fit.m1_diag.shape[0], np.nan)
    for rho in _model_rows(fit):
        denom = max(float(np.linalg.norm(HV_qc[rho])), 1e-300)
        out[rho] = float(np.linalg.norm(pred[rho] - HV_qc[rho])) / denom
    return out


def spike_measure(fit):
    """[spike component] The Dirac mass field m_rho * s_rho -- the
    mesh-independent content of S (a map of it is a resolution
    diagnostic: where the mesh is starving). NaN where no model."""
    return fit.m1_diag * fit.s
