# SPDX-License-Identifier: MIT
"""Graded window coarsening: a numpy prototype and its measurements.

Slice S1 of `dev/window-coarsening-plan.md`. `coarsen_window` below is the
reference for the C++ that follows it: cells graded by whitened distance from
the row's centre (size at most `eps` times that distance), floored at single
points, the spike and the farthest point kept as singletons, each cell replaced
by its mass-weighted centroid with the summed mass and the mass-weighted mean
of every probe field. The fit then runs on the cells through the EXISTING
bindings -- `fit_from_probes` neither knows nor cares that the points are
centroids -- and the plan's questions are answered by measurement:

  (a) the aggregation is exact for the probe sums and every cell obeys the rule
  (b) the cell count against 3 pi / eps^2 * log2(R / h) on synthetic discs
  (c) on real rows of the heat problem, the coarse fit against the full one,
      scored on the FULL window so the comparison is honest; and the order of
      the quadrature error at fixed theta, on the rows and on a fine disc
  (d) the plan's safety check: the coarse-quadrature score against the
      full-window score of the same model
  (e) wall time against point count, full and coarse

Results and discussion: `window-coarsening.md`.

    python experiments/window_coarsening.py            # ~20 min
    python experiments/window_coarsening.py --quick    # ~3 min
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import lgpsf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))
import heat_inversion as heat  # noqa: E402  the public heat inverse problem

EPS_LIST = (0.05, 0.1, 0.15, 0.2, 0.3)
TOP_LEVEL = 5
NUM_PROBES = 60          # level 5 (21 modes) + spike + 3 params needs k >= 50
DIM = 2


# ---------------------------------------------------------------------------
# The algorithm (plan section 2), exactly as the C++ will implement it.
# ---------------------------------------------------------------------------

def box_distance(lo, hi):
    """Whitened distance from the origin to the box [lo, hi]; 0 if inside."""
    return np.linalg.norm(np.maximum(np.maximum(lo, -hi), 0.0))


def coarsen_window(x, m, z, protected, centre, L_w, eps):
    """Graded coarsening of one window. x (K, N) physical points, m (K,)
    positive masses, z (K, k) probe fields, `protected` window positions that
    stay singleton cells, `centre` (N,), L_w the Cholesky factor of the WINDOW
    ellipsoid (Sigma_w = L_w L_w^T, what fit_operator records as
    window_covariance). eps <= 0 is the identity.

    Returns x (Kc, N), m (Kc,), z (Kc, k), protected_cells (input order),
    cell_of (K,), and for the checks: the leaf boxes (whitened) and the
    farthest point's position. Pure: no randomness, no threads."""
    x, m, z = np.asarray(x, float), np.asarray(m, float), np.asarray(z, float)
    K, N = x.shape
    if eps <= 0.0:
        return dict(x=x.copy(), m=m.copy(), z=z.copy(), cell_of=np.arange(K),
                    protected_cells=list(protected), boxes=None, farthest=None)
    u = np.linalg.solve(L_w, (x - centre).T).T                     # 1. whiten
    farthest = int(np.argmax(np.linalg.norm(x - centre, axis=1)))  # 2. ties -> first
    keep = np.zeros(K, bool)
    keep[list(protected)] = True
    keep[farthest] = True
    lo0, hi0 = u.min(axis=0), u.max(axis=0)                        # 3. root box
    root_diag = np.linalg.norm(hi0 - lo0)
    upper_of = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(bool)
    leaves, boxes = [], []
    stack = [(np.arange(K), lo0, hi0, 0)]
    while stack:                                                   # 4. the tree
        members, lo, hi, depth = stack.pop()
        diag = np.linalg.norm(hi - lo)
        if (len(members) <= 1 or diag < 1e-12 * root_diag or depth > 60
                or (not keep[members].any() and diag <= eps * box_distance(lo, hi))):
            leaves.append(members)
            boxes.append((lo, hi))
            continue
        mid = 0.5 * (lo + hi)
        code = ((u[members] >= mid).astype(int) << np.arange(N)).sum(axis=1)
        for c in range(2 ** N):                # a point on the plane goes UP
            sel = members[code == c]           # ascending order is preserved
            if sel.size:
                up = upper_of[c]
                stack.append((sel, np.where(up, mid, lo), np.where(up, hi, mid),
                              depth + 1))
    order = np.argsort([mem[0] for mem in leaves], kind="stable")  # 6. order
    leaves = [leaves[i] for i in order]
    boxes = [boxes[i] for i in order]
    cell_of = np.empty(K, int)
    for c, mem in enumerate(leaves):
        cell_of[mem] = c
    # 5. aggregate; the CSR rows hold each cell's members in ascending order
    P = sp.csr_matrix((m, (cell_of, np.arange(K))), shape=(len(leaves), K))
    mc = P @ np.ones(K)
    return dict(x=(P @ x) / mc[:, None], m=mc, z=(P @ z) / mc[:, None],
                cell_of=cell_of, protected_cells=[int(cell_of[p]) for p in protected],
                boxes=boxes, farthest=farthest)


def check_coarsening(m, z, protected, eps, cw):
    """Item (a): mass and probe-sum conservation, protected singletons, and the
    grading rule on every multi-point cell (worst diag / (eps d); <= 1 is ok)."""
    sizes = np.bincount(cw["cell_of"])
    worst = 0.0
    for c, (lo, hi) in enumerate(cw["boxes"]):
        if sizes[c] > 1:
            d = box_distance(lo, hi)
            worst = max(worst, np.linalg.norm(hi - lo) / (eps * d) if d > 0 else np.inf)
    return dict(
        err_m=abs(cw["m"].sum() - m.sum()) / m.sum(),
        err_mz=np.abs(cw["m"] @ cw["z"] - m @ z).max() / np.abs(m @ z).max(),
        singletons=all(sizes[cw["cell_of"][p]] == 1
                       for p in list(protected) + [cw["farthest"]]),
        worst_ratio=worst)


# ---------------------------------------------------------------------------
# Small helpers: tables, the heat problem, scoring at given theta
# ---------------------------------------------------------------------------

def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for cells in rows:
        print("| " + " | ".join(str(c) for c in cells) + " |")


def med(v):
    return np.median(np.asarray(v, float))


def disc_points(K):
    """About K grid points (spacing h, masses h^2) filling the unit disc."""
    h = np.sqrt(np.pi / K)
    g = np.arange(-1.0 + h / 2, 1.0, h)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return pts[np.linalg.norm(pts, axis=1) <= 1.0], h


def build_heat(grid):
    """The heat problem of the examples, matrix-free: its dense H would not
    fit at 1e5 points, and the fit only ever sees probes."""
    axis = (np.arange(grid) + 0.5) / grid
    mesh = np.meshgrid(axis, axis, indexing="ij")
    x = np.vstack([mesh[0].ravel(), mesh[1].ravel()])
    count, h = x.shape[1], 1.0 / grid
    kappa = heat.conductivity(x)
    stepper = spla.splu(sp.eye(count, format="csc")
                        + (heat.TIME / heat.STEPS) * heat.diffusion_operator(grid, kappa))

    def propagate(U):
        for _ in range(heat.STEPS):
            U = stepper.solve(U)
        return U

    sig2 = 2.0 * (2.0 * heat.TIME) * kappa + (0.75 * h) ** 2
    return dict(x=x, mass=np.full(count, h * h), count=count, spacing=h, kappa=kappa,
                grid=grid, sigma=np.stack([s * np.eye(2) for s in sig2]),
                apply_Hd=lambda U: h * h * propagate(propagate(U)))


def row_config():
    config = lgpsf.OperatorFitConfig()          # tau_window 10, spike on, pinned
    config.row.mode_policy = lgpsf.ShellLadder(list(range(TOP_LEVEL + 1)))
    config.row.target_score = None              # let mode_patience decide
    return config


def level_of(model):
    return max(2 * mode.p + mode.ell for mode in model.modes)


def public_theta(centre, sigma):
    L = np.linalg.cholesky(sigma)
    return np.concatenate([centre, np.log(np.diag(L)), L[np.tril_indices(DIM, -1)]])


def geometry(model):
    """(major, minor) axes and the major axis's angle in degrees."""
    frame = model.frame()
    w, V = np.linalg.eigh(frame.L @ frame.L.T)
    return np.sqrt(w[::-1]), np.degrees(np.arctan2(V[1, -1], V[0, -1]))


class Scorer:
    """CV score, design matrix and baseline at GIVEN theta on ONE quadrature
    (x (N, K), m, z (k, K)) -- the full window or the coarse cells."""

    def __init__(self, x, m, z, y, centre, spike, target_mass, split=None):
        self.x, self.m, self.centre, self.split, self.mass = x, m, centre, split, target_mass
        self.z_hat = lgpsf.whiten_probes(z, m)
        self.y_hat = lgpsf.whiten_data(y, target_mass)
        E = np.zeros((1, len(m)))
        E[0, spike] = 1.0
        self.e_hat = lgpsf.whiten_extra(E, target_mass, m)

    def basis(self, modes):
        return lgpsf.WhitenedBasis(self.x, self.mass, self.m, modes, self.centre,
                                   lgpsf.MuMode.Pinned)

    def theta_hat(self, theta):
        return lgpsf.to_theta_hat(theta, self.centre, lgpsf.MuMode.Pinned)

    def score(self, theta, modes):
        return lgpsf.linear_cv_score(self.z_hat, self.y_hat, self.basis(modes),
                                     self.theta_hat(theta), self.e_hat, self.split)

    def design(self, theta, modes):
        """(k, num_modes): sqrt(m_rho) sum_j m_j z_jl phi_i(x_j) -- the object
        the coarsening approximates."""
        return self.z_hat @ self.basis(modes).values(self.theta_hat(theta)).T

    def baseline(self, theta):
        """fit_operator's baseline: the best level at the prior, pinned."""
        return min(self.score(theta, lgpsf.modes_up_to_level(DIM, lv))
                   for lv in range(TOP_LEVEL + 1)
                   if NUM_PROBES >= 2 * (len(lgpsf.modes_up_to_level(DIM, lv)) + 1 + 3))


MODES_TOP = lgpsf.modes_up_to_level(DIM, TOP_LEVEL)
LEVELS_TOP = np.array([2 * md.p + md.ell for md in MODES_TOP])


def column_error_by_level(D_coarse, D_full):
    err = np.linalg.norm(D_coarse - D_full, axis=0) / np.linalg.norm(D_full, axis=0)
    return [err[LEVELS_TOP == lv].max() for lv in range(TOP_LEVEL + 1)]


def slope(errors):
    """log-log slope over the eps whose error is above round-off."""
    ok = np.array([e > 1e-10 for e in errors])
    return np.polyfit(np.log(np.array(EPS_LIST)[ok]), np.log(np.array(errors)[ok]), 1)[0]


# ---------------------------------------------------------------------------
# (b) the cell-count law, and (c2) the quadrature order, on synthetic discs
# ---------------------------------------------------------------------------

def cell_count_law(sizes):
    print("\n## (b) cell count on synthetic discs (unit ball window)\n")
    rows = []
    for K in sizes:
        pts, h = disc_points(K)
        m = np.full(len(pts), h * h)
        z = np.random.default_rng(1).normal(size=(len(pts), 3))
        near = int(np.argmin(np.linalg.norm(pts, axis=1)))
        for eps in EPS_LIST:
            t = time.perf_counter()
            cw = coarsen_window(pts, m, z, [near], np.zeros(2), np.eye(2), eps)
            t = time.perf_counter() - t
            chk = check_coarsening(m, z, [near], eps, cw)
            law = 3 * np.pi / eps ** 2 * np.log2(1.0 / h)
            rows.append([len(pts), eps, len(cw["m"]), np.sum(np.bincount(cw["cell_of"]) == 1),
                         f"{law:.0f}", f"{len(cw['m']) / law:.2f}", f"{t:.2f}",
                         f"{chk['err_m']:.1e}", f"{chk['err_mz']:.1e}", chk["singletons"],
                         f"{chk['worst_ratio']:.3f}"])
    table(["K", "eps", "cells", "singletons", "3pi/eps^2 log2(R/h)", "ratio", "coarsen s",
           "err mass", "err m*z", "protected ok", "worst diag/(eps d)"], rows)
    pts, h = disc_points(sizes[0])     # eps -> 0 is the identity, in the same order
    cw = coarsen_window(pts, np.full(len(pts), h * h), np.zeros((len(pts), 1)),
                        [0], np.zeros(2), np.eye(2), 1e-9)
    assert len(cw["m"]) == len(pts) and np.array_equal(cw["cell_of"], np.arange(len(pts)))
    print(f"\neps = 1e-9 on K = {len(pts)}: identity, same order -- ok")


def quadrature_order(K):
    """(c2) in the asymptotic regime h << eps sigma: a Gaussian of width R/3
    on a fine disc, relative design-column error per level for white-noise
    probes and for smooth ones. The centroid rule is second order only when
    the in-cell probe dipole sum m_j z_j (x_j - x_C) vanishes."""
    pts, h = disc_points(K)
    m = np.full(len(pts), h * h)
    near = int(np.argmin(np.linalg.norm(pts, axis=1)))
    rng = np.random.default_rng(2)
    white = rng.normal(size=(8, len(pts)))
    smooth = np.cos(rng.normal(size=(8, 2)) * 6.0 @ pts.T + rng.uniform(0, 7, size=(8, 1)))
    theta = public_theta(np.zeros(2), (1.0 / 3.0) ** 2 * np.eye(2))
    print(f"\n## (c2) design-column error at fixed theta on a disc, K = {len(pts)}, "
          f"h/sigma = {3 * h:.3f}\n")
    rows = []
    for probes, name in ((white, "white"), (smooth, "smooth")):
        D = Scorer(pts.T, m, probes, np.zeros(8), np.zeros(2), near, h * h).design(theta, MODES_TOP)
        errs = []
        for eps in EPS_LIST:
            cw = coarsen_window(pts, m, probes.T, [near], np.zeros(2), np.eye(2), eps)
            Dc = Scorer(cw["x"].T, cw["m"], cw["z"].T, np.zeros(8), np.zeros(2),
                        cw["protected_cells"][0], h * h).design(theta, MODES_TOP)
            errs.append(column_error_by_level(Dc, D))
            rows.append([name, eps, len(cw["m"])] + [f"{e:.2e}" for e in errs[-1]])
        rows.append([name, "slope", ""] + [f"{slope([e[lv] for e in errs]):.2f}"
                                           for lv in range(TOP_LEVEL + 1)])
    table(["probes", "eps", "cells"] + [f"level {lv}" for lv in range(TOP_LEVEL + 1)], rows)


# ---------------------------------------------------------------------------
# (c), (d), (e): one real row, full window against coarse cells at every eps
# ---------------------------------------------------------------------------

def timed_fit(x, m, z, y, centre, spike, config, sigma, target_mass):
    t = time.perf_counter()
    result = lgpsf.fit_from_probes(x, m, z, y, centre, spike_index=spike, config=config,
                                   guesses=[lgpsf.InitialGuess(sigma, label="sigma0")],
                                   target_mass=target_mass)
    return result, time.perf_counter() - t


def spike_share(model, centre, mass):
    """The spike's share of the diagonal entry, s / (s + m_rho * smooth(centre))."""
    return model.s[0] / (model.s[0] + mass * lgpsf.eval_expansion(model, centre[:, None])[0])


def compare_row(problem, fit, V, HV, sigma, rho, config, split, tag):
    model = fit.model
    w = np.asarray(model.row_window(rho))
    x_w, m_w, z_w, y = problem["x"][:, w], problem["mass"][w], V[:, w], HV[:, rho]
    spike = int(np.searchsorted(w, rho))
    assert w[spike] == rho, "the row's own dof must be in its window"
    centre, mass = model.window_center[rho], problem["mass"][rho]
    L_w = np.linalg.cholesky(model.window_covariance[rho])
    prior = public_theta(centre, sigma[rho])

    full, t_full = timed_fit(x_w, m_w, z_w, y, centre, spike, config, sigma[rho], mass)
    S_full = Scorer(x_w, m_w, z_w, y, centre, spike, mass, split)
    D_full = S_full.design(full.model.theta, MODES_TOP)
    axes_f, angle_f = geometry(full.model)
    rec = dict(tag=tag, rho=rho, K=len(w), t_full=t_full, score_full=full.score,
               base_full=S_full.baseline(prior), level_full=level_of(full.model),
               ncand_full=len(full.candidates), aspect_full=axes_f[0] / axes_f[1],
               spike_full=spike_share(full.model, centre, mass),
               r_over_h=L_w[0, 0] / problem["spacing"], per_eps={},
               # consistency: the operator's own row and this call agree bit for bit
               op_dscore=abs(full.score - fit.diagnostics.score[rho])
               if fit.diagnostics.status[rho] == int(lgpsf.RowStatus.Fit) else 0.0,
               op_dbase=abs(S_full.baseline(prior) - fit.diagnostics.baseline_score[rho]),
               self_dscore=abs(S_full.score(full.model.theta, full.model.modes) - full.score))
    for eps in EPS_LIST:
        cw = coarsen_window(x_w.T, m_w, z_w.T, [spike], centre, L_w, eps)
        xc, zc, sc = cw["x"].T, cw["z"].T, cw["protected_cells"][0]
        coarse, t_c = timed_fit(xc, cw["m"], zc, y, centre, sc, config, sigma[rho], mass)
        S_c = Scorer(xc, cw["m"], zc, y, centre, sc, mass, split)
        axes_c, angle_c = geometry(coarse.model)
        rec["per_eps"][eps] = dict(
            Kc=len(cw["m"]), t_c=t_c, chk=check_coarsening(m_w, z_w.T, [spike], eps, cw),
            score_c=coarse.score,                                   # coarse quadrature
            score_c_full=S_full.score(coarse.model.theta, coarse.model.modes),
            base_c=S_c.baseline(prior),
            level_c=level_of(coarse.model), ncand_c=len(coarse.candidates),
            admissible=coarse.candidates[coarse.winner].admissible,
            dlog_axes=np.log10(axes_c / axes_f),
            dangle=((angle_c - angle_f + 90.0) % 180.0) - 90.0,
            dc0=abs(coarse.model.c[0] - full.model.c[0]) / abs(full.model.c[0]),
            dspike=abs(spike_share(coarse.model, centre, mass) - rec["spike_full"]),
            col_err=column_error_by_level(S_c.design(full.model.theta, MODES_TOP), D_full),
            cv_fixed=abs(S_c.score(full.model.theta, full.model.modes) - full.score))
    return rec


def pick_rows(problem, per_region, HV, rng):
    reg = heat.regions(problem)
    rows, region = [], {}
    for name in ("slab", "background", "pocket", "interface"):
        idx = np.flatnonzero(reg[name])
        for r in rng.choice(idx, size=min(per_region, len(idx)), replace=False):
            rows.append(int(r))
            region[int(r)] = name
    for r in np.argsort(np.linalg.norm(HV, axis=0))[:2]:     # the weakest rows
        rows.append(int(r))
        region[int(r)] = "weak"
    return sorted(set(rows)), region


def run_grid(grid, per_region, wide_rows, records):
    problem = build_heat(grid)
    rng = np.random.default_rng(grid)
    V = rng.normal(size=(NUM_PROBES, problem["count"]))
    HV = problem["apply_Hd"](V.T).T
    config = row_config()
    split = lgpsf.kfold_split(NUM_PROBES, config.row.cv_folds)
    rows, region = pick_rows(problem, per_region, HV, rng)
    # the motivating case: a prior too wide by decades -- pocket rows with
    # sigma x4 (window radius x4, the kernel unchanged)
    pocket = [r for r in rows if region[r] == "pocket"][:wide_rows]
    sigma_wide = problem["sigma"].copy()
    sigma_wide[pocket] *= 16.0
    for sigma, subset, label in ((problem["sigma"], rows, None), (sigma_wide, pocket, "pocket-x4")):
        if not subset:
            continue
        t = time.perf_counter()
        fit = lgpsf.fit_operator(problem["x"], problem["mass"], problem["mass"], V, HV,
                                 sigma, config=config, rows=np.array(subset))
        print(f"\ngrid {grid} ({problem['count']} points): fit_operator on {len(subset)} "
              f"rows in {time.perf_counter() - t:.1f} s", flush=True)
        for rho in subset:
            tag = label or region[rho]
            records.append(compare_row(problem, fit, V, HV, sigma, rho, config.row, split,
                                       f"g{grid}/{tag}"))
            print(f"  row {rho:>6} {tag:<10} K={records[-1]['K']:>6} "
                  f"t_full={records[-1]['t_full']:6.2f}s", flush=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(records):
    print(f"\nconsistency: max |fit_from_probes - fit_operator| score "
          f"{max(r['op_dscore'] for r in records):.1e}, baseline "
          f"{max(r['op_dbase'] for r in records):.1e}; re-scored winner "
          f"{max(r['self_dscore'] for r in records):.1e}")
    chks = [p["chk"] for r in records for p in r["per_eps"].values()]
    print(f"\n## (a) on the real windows, worst over rows and eps: err mass "
          f"{max(c['err_m'] for c in chks):.1e}, err m*z {max(c['err_mz'] for c in chks):.1e}, "
          f"protected singletons {all(c['singletons'] for c in chks)}, worst diag/(eps d) "
          f"{max(c['worst_ratio'] for c in chks):.3f}")

    print(f"\n## (c) coarse fit against the full fit, per eps ({len(records)} rows)\n")
    rows = []
    for eps in EPS_LIST:
        P = [(r, r["per_eps"][eps]) for r in records]
        ds = [p["score_c_full"] - r["score_full"] for r, p in P]
        aniso = [abs(p["dangle"]) for r, p in P if r["aspect_full"] > 1.1]
        rows.append([eps, f"{med([p['Kc'] / r['K'] for r, p in P]):.3f}", f"{med(ds):+.4f}",
                     f"{max(ds):+.4f}",
                     f"{np.mean([p['score_c_full'] < r['base_full'] for r, p in P]):.2f}",
                     f"{np.mean([p['admissible'] for r, p in P]):.2f}",
                     f"{np.mean([p['level_c'] == r['level_full'] for r, p in P]):.2f}",
                     f"{np.mean([p['level_c'] < r['level_full'] for r, p in P]):.2f}",
                     f"{med([abs(p['dlog_axes'][0]) for r, p in P]):.4f}",
                     f"{med([abs(p['dlog_axes'][1]) for r, p in P]):.4f}",
                     f"{med(aniso) if aniso else float('nan'):.2f} ({len(aniso)})",
                     f"{med([p['dc0'] for r, p in P]):.4f}", f"{med([p['dspike'] for r, p in P]):.4f}",
                     f"{med([p['cv_fixed'] for r, p in P]):.1e}",
                     f"{sum(p['t_c'] for r, p in P) / sum(r['t_full'] for r in records):.3f}"])
    table(["eps", "median Kc/K", "median dscore", "max dscore", "beats baseline (honest)",
           "admissible", "same level", "level lower", "median dlog10 major",
           "median dlog10 minor", "median dangle deg (aspect>1.1)", "median dc0",
           "median dspike", "median cv_fixed", "time coarse/full"], rows)
    print(f"\nfull fits: beat baseline on "
          f"{np.mean([r['score_full'] < r['base_full'] for r in records]):.2f} of rows; "
          f"median score {med([r['score_full'] for r in records]):.4f}, median baseline "
          f"{med([r['base_full'] for r in records]):.4f}; levels "
          f"{np.bincount([r['level_full'] for r in records], minlength=TOP_LEVEL + 1).tolist()}")

    print("\n## (c) worst row per eps: coarse fit re-scored on the full window, minus the full fit\n")
    rows = []
    for eps in EPS_LIST:
        r = max(records, key=lambda r: r["per_eps"][eps]["score_c_full"] - r["score_full"])
        p = r["per_eps"][eps]
        rows.append([eps, r["tag"], r["rho"], r["K"], p["Kc"], f"{r['score_full']:.4f}",
                     f"{p['score_c_full']:.4f}", f"{p['score_c']:.4f}", f"{r['base_full']:.4f}",
                     f"{r['level_full']}/{p['level_c']}", f"{r['ncand_full']}/{p['ncand_c']}"])
    table(["eps", "tag", "row", "K", "Kc", "score full", "coarse on full", "coarse own",
           "baseline", "level full/coarse", "cand full/coarse"], rows)

    big = sorted(records, key=lambda r: -r["K"])[:max(6, len(records) // 4)]
    print(f"\n## (c) the {min(10, len(big))} largest windows: dscore (cells) per eps\n")
    table(["tag", "row", "K", "score full", "baseline"] + [f"eps {eps}" for eps in EPS_LIST],
          [[r["tag"], r["rho"], r["K"], f"{r['score_full']:.4f}", f"{r['base_full']:.4f}"]
           + [f"{r['per_eps'][eps]['score_c_full'] - r['score_full']:+.4f} ({r['per_eps'][eps]['Kc']})"
              for eps in EPS_LIST] for r in big[:10]])
    print(f"\n## (c2) on the {len(big)} largest real windows (K >= {min(r['K'] for r in big)}): "
          "median worst relative design-column error per level, and the CV score's change, "
          "both at the full winner's theta\n")
    errs = [[med([r["per_eps"][eps]["col_err"][lv] for r in big]) for lv in range(TOP_LEVEL + 1)]
            for eps in EPS_LIST]
    cvf = [[f([r["per_eps"][eps]["cv_fixed"] for r in big]) for f in (med, max)] for eps in EPS_LIST]
    table(["eps"] + [f"level {lv}" for lv in range(TOP_LEVEL + 1)]
          + ["median cv_fixed", "max cv_fixed"],
          [[eps] + [f"{e:.2e}" for e in row] + [f"{v:.1e}" for v in cv]
           for eps, row, cv in zip(EPS_LIST, errs, cvf)]
          + [["slope"] + [f"{slope([e[lv] for e in errs]):.2f}" for lv in range(TOP_LEVEL + 1)]
             + [f"{slope([c[i] for c in cvf]):.2f}" for i in range(2)]])

    print(f"\n## (d) safety check on the {len(big)} largest windows\n")
    rows = []
    for eps in EPS_LIST:
        P = [(r, r["per_eps"][eps]) for r in big]
        rows.append([eps, f"{max(abs(p['score_c'] - p['score_c_full']) for r, p in P):.4f}",
                     sum((p["score_c"] < p["base_c"]) and not (p["score_c_full"] < r["base_full"])
                         for r, p in P),
                     sum((p["score_c_full"] < r["base_full"]) and not (p["score_c"] < p["base_c"])
                         for r, p in P)])
    table(["eps", "max abs(score_c - score_c_full)", "coarse says beat, full says lose",
           "full says beat, coarse says lose"], rows)

    print("\n## per-row detail on the largest windows (eps = 0.1)\n")
    table(["tag", "row", "K", "Kc", "level full/coarse", "cand full/coarse", "score full",
           "coarse on full", "coarse own", "baseline", "t full", "t coarse"],
          [[r["tag"], r["rho"], r["K"], p["Kc"], f"{r['level_full']}/{p['level_c']}",
            f"{r['ncand_full']}/{p['ncand_c']}", f"{r['score_full']:.4f}",
            f"{p['score_c_full']:.4f}", f"{p['score_c']:.4f}", f"{r['base_full']:.4f}",
            f"{r['t_full']:.2f}", f"{p['t_c']:.2f}"]
           for r in big for p in [r["per_eps"][0.1]]])

    print("\n## (e) wall time against point count\n")
    K = np.array([r["K"] for r in records], float)
    t = np.array([r["t_full"] for r in records])
    n = np.array([r["ncand_full"] for r in records], float)
    a, b = np.polyfit(K, t, 1)
    print(f"full:   t = {b:.3f} s + {a:.2e} s/point  (K {K.min():.0f}..{K.max():.0f}, "
          f"per candidate {np.median(t / n / K):.2e} s/point, candidates {n.min():.0f}..{n.max():.0f})")
    Kc = np.array([p["Kc"] for r in records for p in r["per_eps"].values()], float)
    tc = np.array([p["t_c"] for r in records for p in r["per_eps"].values()])
    nc = np.array([p["ncand_c"] for r in records for p in r["per_eps"].values()], float)
    a, b = np.polyfit(Kc, tc, 1)
    print(f"coarse: t = {b:.3f} s + {a:.2e} s/cell   (Kc {Kc.min():.0f}..{Kc.max():.0f}, "
          f"per candidate {np.median(tc / nc / Kc):.2e} s/cell)\n")
    edges = [0, 500, 2000, 8000, 30000, 10 ** 7]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        R = [r for r in records if lo <= r["K"] < hi]
        if R:
            rows.append([f"[{lo}, {hi})", len(R), f"{med([r['K'] for r in R]):.0f}",
                         f"{med([r['t_full'] for r in R]):.2f}",
                         f"{med([r['per_eps'][0.1]['t_c'] for r in R]):.2f}",
                         f"{med([r['per_eps'][0.1]['Kc'] for r in R]):.0f}",
                         f"{med([3 * np.pi / 0.01 * np.log2(r['r_over_h']) for r in R]):.0f}"])
    table(["K bucket", "rows", "median K", "median t full", "median t coarse eps=0.1",
           "median Kc eps=0.1", "law 3pi/eps^2 log2(R/h)"], rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    print(f"lgpsf {lgpsf.__version__}, probes {NUM_PROBES}, ladder 0..{TOP_LEVEL}, "
          f"eps {EPS_LIST}, {'quick' if args.quick else 'full'} mode")
    cell_count_law([1000, 10000] if args.quick else [1000, 10000, 100000])
    quadrature_order(30000 if args.quick else 300000)
    plan = [(48, 2, 1), (100, 1, 1)] if args.quick else \
        [(48, 4, 2), (100, 4, 3), (200, 2, 2), (316, 1, 1)]
    records = []
    for grid, per_region, wide in plan:
        run_grid(grid, per_region, wide, records)
    report(records)


if __name__ == "__main__":
    main()
