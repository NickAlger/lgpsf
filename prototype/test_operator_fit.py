"""Tests for operator_fit.py (the whole-operator layer).

The synthetic operator is built row-by-row from the exact row model
(H[rho, j] = m1_rho m2_j sum_i c_i phi_i(x_j; theta_rho) + m1_rho s
delta_{j,rho}) with smoothly varying per-row ellipsoids, scattered
points, and DELIBERATELY DIFFERENT row and column masses (M1 != M2) --
so exact coefficient recovery also pins the target_mass routing. The
tests exercise the per-row protocol (gate -> window -> search ->
baseline guard -> status), the padded-array output contract, and every
helper in the plan doc's table.

Run directly (`python test_operator_fit.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from ellipsoid_transform import release_mu
from init_dictionary import theta_from_L
from lg_ellipsoid_feature import eval_feature
from operator_fit import (
    OperatorFit,
    OperatorFitConfig,
    assemble_sparse,
    ellipsoid_field,
    eval_entries,
    eval_kernel,
    fit_operator,
    matvec,
    qc_map,
    spike_measure,
    to_linear_operator,
)
from probe_fit import ProbeFitConfig
from varpro import VarProOptions

MODES = [(0, 0, 0), (0, 1, 0), (0, 1, 1)]
C_TRUE = np.array([1.0, 0.35, -0.25])
S_TRUE = 0.7
FIT_ROWS = [3, 17, 42, 77, 101]


def _rot(ang):
    return np.array([[np.cos(ang), -np.sin(ang)],
                     [np.sin(ang), np.cos(ang)]])


def _true_L(x_rho):
    """Smoothly varying anisotropic ellipsoid: axes 0.45/0.25, angle a
    function of position."""
    ang = 0.4 * (x_rho[0] + 0.7 * x_rho[1])
    R = _rot(ang)
    return np.linalg.cholesky(R @ np.diag([0.45 ** 2, 0.25 ** 2]) @ R.T)


def _make_operator(rng, K=120, k=50):
    """Square operator on scattered 2D points, M1 != M2, every row
    exactly representable with MODES + spike."""
    x = rng.uniform(-1.5, 1.5, size=(2, K))
    m1 = rng.uniform(0.5, 2.0, size=K)
    m2 = rng.uniform(0.5, 2.0, size=K)
    H = np.zeros((K, K))
    Sig_true = np.zeros((K, 2, 2))
    for rho in range(K):
        L = _true_L(x[:, rho])
        Sig_true[rho] = L @ L.T
        theta = release_mu(theta_from_L(L), x[:, rho])
        phi = C_TRUE @ eval_feature(theta, 2, x, MODES)
        H[rho] = m1[rho] * m2 * phi
        H[rho, rho] += m1[rho] * S_TRUE
    V = rng.standard_normal((K, k))
    return dict(x=x, m1=m1, m2=m2, H=H, Sig_true=Sig_true,
                V=V, HV=H @ V)


_CACHE = {}


def _square_case():
    """One fitted square operator, shared across tests (the fit is the
    expensive part; the helper tests only read it)."""
    if "square" not in _CACHE:
        rng = np.random.default_rng(0)
        prob = _make_operator(rng)
        # deliberately imperfect prior: an isotropic circle (truth is
        # 0.45/0.25 rotated), so the searched fit must beat the baseline
        fit = fit_operator(prob["x"], prob["m1"], prob["m2"],
                           prob["V"], prob["HV"],
                           sigma=0.35 ** 2 * np.eye(2),
                           modes=MODES, rows=FIT_ROWS)
        _CACHE["square"] = (prob, fit)
    return _CACHE["square"]


def test_fit_operator_recovers_synthetic_square():
    """Gated rows are fit exactly (theta, c, s -- the latter two pin the
    m1-based target-mass routing, since M1 != M2 here); ungated rows get
    status, not silence; the output arrays decode coherently."""
    prob, fit = _square_case()
    assert isinstance(fit, OperatorFit)
    R = prob["x"].shape[1]
    for rho in range(R):
        expected = "fit" if rho in FIT_ROWS else "gated_out"
        assert fit.status[rho] == expected, f"row {rho}"

    mu_field, Sigma_field = ellipsoid_field(fit)
    for rho in FIT_ROWS:
        assert fit.score[rho] < 1e-5
        # searched fit strictly beat the (isotropic-prior) baseline
        assert fit.score[rho] < fit.baseline_score[rho]
        assert not fit.released[rho]            # exact mu0: no release ran
        assert fit.stop_reason[rho] == "target"
        # decoder: mode_set_id -> the explicit mode list
        assert fit.row_modes(rho) == MODES
        assert np.allclose(fit.c[rho, :3], C_TRUE, atol=1e-4)
        assert abs(fit.s[rho] - S_TRUE) < 1e-4
        assert np.allclose(mu_field[rho], prob["x"][:, rho], atol=1e-6)
        assert np.linalg.norm(Sigma_field[rho] - prob["Sig_true"][rho]) \
            < 1e-3 * np.linalg.norm(prob["Sig_true"][rho])
        # L is the Cholesky factor of the same Sigma
        assert np.allclose(fit.L[rho] @ fit.L[rho].T, Sigma_field[rho])
    # no-model rows: NaN parameters, -1 decoder
    ungated = [r for r in range(R) if r not in FIT_ROWS]
    assert np.all(np.isnan(fit.theta[ungated]))
    assert np.all(fit.mode_set_id[ungated] == -1)
    assert fit.failures == {}


def test_operator_helpers_entries_matvec_assemble_qc():
    """Every helper in the plan doc's table, against the explicit H."""
    prob, fit = _square_case()
    H = prob["H"]
    K = H.shape[1]
    rng = np.random.default_rng(11)

    # eval_entries: paired dof entries, spike iff col == row dof
    for rho in FIT_ROWS[:2]:
        vals = eval_entries(fit, np.full(K, rho), np.arange(K))
        assert np.linalg.norm(vals - H[rho]) < 1e-4 * np.linalg.norm(H[rho])

    # matvec: exact application of the parametric operator; gated rows
    # contribute zero rows
    v = rng.standard_normal(K)
    mv = matvec(fit, v)
    for rho in range(K):
        if rho in FIT_ROWS:
            assert abs(mv[rho] - H[rho] @ v) < 1e-4 * abs(H[rho] @ v)
        else:
            assert mv[rho] == 0.0
    B = rng.standard_normal((K, 3))
    assert matvec(fit, B).shape == (K, 3)
    op = to_linear_operator(fit)
    assert np.allclose(op @ v, mv)

    # assemble_sparse: tau-ellipsoid supports + spike on the diagonal
    A = assemble_sparse(fit, tau=6.0)
    assert A.shape == (K, K)
    for rho in FIT_ROWS:
        row = A[rho].toarray().ravel()
        assert np.linalg.norm(row - H[rho]) < 1e-3 * np.linalg.norm(H[rho])
    ungated = [r for r in range(K) if r not in FIT_ROWS]
    assert A[ungated].nnz == 0
    S = assemble_sparse(fit, tau=6.0, symmetrize="average")
    assert abs(S - S.T).max() < 1e-14

    # qc_map against held-out probes
    V_qc = rng.standard_normal((K, 30))
    qc = qc_map(fit, V_qc, H @ V_qc)
    assert np.all(qc[FIT_ROWS] < 1e-4)
    assert np.all(np.isnan(qc[ungated]))

    # spike_measure: the Dirac mass field m_rho * s_rho
    sm = spike_measure(fit)
    assert np.allclose(sm[FIT_ROWS], prob["m1"][FIT_ROWS] * S_TRUE,
                       rtol=1e-4)
    assert np.all(np.isnan(sm[ungated]))


def test_deployed_support_is_fit_window():
    """The slice-38 invariant: deployed support == fit window. Windows
    are stored (CSR-style, sorted, self-inclusive in the square
    context) and every dof-context helper returns exactly zero outside
    them -- the windowed CV score is honest for the deployed operator
    by construction."""
    prob, fit = _square_case()
    K = prob["x"].shape[1]
    assert fit.window_indptr is not None
    for rho in FIT_ROWS:
        win = fit.row_window(rho)
        assert win.size > 0 and np.all(np.diff(win) > 0)
        assert rho in win                      # own dof inside (spike)

    # the default fixture's windows cover the whole domain; refit two
    # rows with a TIGHT tau_window so a genuine exterior exists
    rho = FIT_ROWS[0]
    tight = fit_operator(prob["x"], prob["m1"], prob["m2"],
                         prob["V"], prob["HV"],
                         sigma=0.35 ** 2 * np.eye(2), modes=MODES,
                         rows=[rho],
                         config=OperatorFitConfig(tau_window=3.0))
    win = tight.row_window(rho)
    outside = np.setdiff1d(np.arange(K), win)
    assert outside.size > 0, "tight window still covers the mesh"
    j_out = int(outside[0])

    # eval_entries: zero outside the window
    vals = eval_entries(tight, np.full(outside.size, rho), outside)
    assert np.all(vals == 0.0)
    # assemble_sparse: no stored entries outside the window
    A = assemble_sparse(tight, tau=6.0).tocsr()
    assert np.all(np.isin(A[rho].indices, win))
    # matvec: a vector supported outside the window (and zero at rho,
    # so no spike term) produces exactly zero in that row
    v = np.zeros(K)
    v[j_out] = 1.0
    assert matvec(tight, v)[rho] == 0.0


def test_baseline_guard_ships_baseline():
    """With an ORACLE sigma prior and the LM search crippled
    (max_nfev=1: MINPACK returns theta_init unchanged), the searched
    score exactly ties the baseline's -- computed at the identical
    theta on identical folds -- and the strict-inequality guard ships
    the baseline: status "fallback_baseline", exact linear recovery."""
    prob, _ = _square_case()
    rho = 7
    cfg = OperatorFitConfig(
        row=ProbeFitConfig(varpro=VarProOptions(max_nfev=1)))
    fit = fit_operator(prob["x"], prob["m1"], prob["m2"],
                       prob["V"], prob["HV"], sigma=prob["Sig_true"],
                       modes=MODES, rows=[rho], config=cfg)
    assert fit.status[rho] == "fallback_baseline"
    assert fit.score[rho] == fit.baseline_score[rho]
    assert fit.score[rho] < 1e-8            # oracle theta, exact model
    assert not fit.released[rho]
    assert np.allclose(fit.c[rho, :3], C_TRUE, atol=1e-6)
    assert abs(fit.s[rho] - S_TRUE) < 1e-6
    assert np.allclose(fit.L[rho],
                       np.linalg.cholesky(prob["Sig_true"][rho]))
    assert np.allclose(fit.mu[rho], prob["x"][:, rho])


def test_rectangular_smooth_only():
    """Distinct row dofs (x_rows), smooth-only truth: the spike must be
    explicitly disabled (validated), and eval_kernel evaluates the
    fitted continuum kernel at arbitrary off-grid query points."""
    rng = np.random.default_rng(2)
    K, k, R = 100, 40, 3
    x_cols = rng.uniform(-1.5, 1.5, size=(2, K))
    m2 = rng.uniform(0.5, 2.0, size=K)
    x_rows = rng.uniform(-0.8, 0.8, size=(2, R))
    m1 = rng.uniform(0.5, 2.0, size=R)
    H = np.zeros((R, K))
    thetas = []
    for r in range(R):
        L = _true_L(x_rows[:, r])
        theta = release_mu(theta_from_L(L), x_rows[:, r])
        thetas.append(theta)
        H[r] = m1[r] * m2 * (C_TRUE @ eval_feature(theta, 2, x_cols, MODES))
    V = rng.standard_normal((K, k))
    HV = H @ V

    try:
        fit_operator(x_cols, m1, m2, V, HV, sigma=0.35 ** 2 * np.eye(2),
                     modes=MODES, x_rows=x_rows)
        assert False, "spike=True with x_rows must be rejected"
    except ValueError:
        pass

    fit = fit_operator(x_cols, m1, m2, V, HV,
                       sigma=0.35 ** 2 * np.eye(2), modes=MODES,
                       x_rows=x_rows,
                       config=OperatorFitConfig(spike=False))
    assert all(fit.status[r] == "fit" for r in range(R))
    assert np.all(fit.score < 1e-5)
    assert np.all(fit.s == 0.0)
    assert np.all(spike_measure(fit) == 0.0)

    xq = rng.uniform(-1.2, 1.2, size=(2, 37))     # off-grid queries
    kern = eval_kernel(fit, range(R), xq)
    for r in range(R):
        truth = C_TRUE @ eval_feature(thetas[r], 2, xq, MODES)
        assert np.linalg.norm(kern[r] - truth) \
            < 1e-4 * np.linalg.norm(truth)


def test_windows_override_and_failed_status():
    """An explicit per-row window is honored (entry rho overrides
    derivation; None entries derive as usual), and a non-SPD sigma row
    fails cleanly with a recorded message instead of raising."""
    prob, _ = _square_case()
    K = prob["x"].shape[1]
    rho_win, rho_bad = 42, 5

    windows = [None] * K
    order = np.argsort(np.linalg.norm(
        prob["x"] - prob["x"][:, [rho_win]], axis=0))
    windows[rho_win] = np.sort(order[:90])
    sigma = prob["Sig_true"].copy()
    sigma[rho_bad] = -np.eye(2)

    fit = fit_operator(prob["x"], prob["m1"], prob["m2"],
                       prob["V"], prob["HV"], sigma=sigma, modes=MODES,
                       rows=[rho_win, rho_bad], windows=windows)
    assert fit.status[rho_win] == "fit"
    # y carries contributions from ALL columns but the override window
    # sees only 90 of 120, so the excluded Gaussian tail acts as model
    # noise: a good-but-not-exact score is the correct outcome
    assert fit.score[rho_win] < 1e-2
    assert fit.status[rho_bad] == "failed"
    assert rho_bad in fit.failures
    assert np.all(np.isnan(fit.theta[rho_bad]))


if __name__ == "__main__":
    test_fit_operator_recovers_synthetic_square()
    test_operator_helpers_entries_matvec_assemble_qc()
    test_deployed_support_is_fit_window()
    test_baseline_guard_ships_baseline()
    test_rectangular_smooth_only()
    test_windows_override_and_failed_status()
    print("all operator_fit checks passed")
