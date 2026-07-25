"""Tests for row_fit.py: the candidate-stream single-row fitting layer
and the backprojection initial-guess estimator.

The synthetic rows are built from the exact row model (raw values
H[rho,j] = m_rho m_j sum c phi + m_rho s delta) on scattered points with
NONUNIFORM masses, probed with iid standard-normal fields -- i.e. the
raw-data contract fit_row and backproject_row/row_moments document,
exercised end to end including the whitening routing, the candidate
grid (init x mode-level x encoding), the linear-stage CV score, the
admissibility and counting-rule guards, and the early-stop certificates.

Run directly (`python test_row_fit.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from ellipsoid_transform import release_mu, unpack_theta
from lg_ellipsoid_feature import eval_feature
from row_fit import (
    RowFitConfig,
    RowFitResult,
    _window_shape,
    backproject_row,
    fit_row,
    row_moments,
)

MODES = [(0, 0, 0), (0, 1, 0), (0, 1, 1)]


def _make_row(rng, K=90, k=60, mu_true=None, s_true=0.7):
    """Scattered batch, nonuniform masses, an exactly-representable row."""
    x = rng.uniform(-1.5, 1.5, size=(2, K))
    m2 = rng.uniform(0.5, 2.0, size=K)
    diag_index = int(np.argmin(np.linalg.norm(x, axis=0)))
    if mu_true is None:
        mu_true = x[:, diag_index].copy()
    theta_true = np.concatenate([mu_true,
                                 [np.log(0.45), np.log(0.25), 0.12]])
    c_true = np.array([1.0, 0.35, -0.25])
    phi = c_true @ eval_feature(theta_true, 2, x, MODES)      # kernel values
    row_mass = m2[diag_index]
    raw_row = row_mass * m2 * phi
    raw_row[diag_index] += row_mass * s_true
    z = rng.standard_normal((k, K))
    y = z @ raw_row
    return dict(x=x, m2=m2, z=z, y=y, diag_index=diag_index,
                mu_true=mu_true, theta_true=theta_true, c_true=c_true,
                s_true=s_true, phi=phi)


def _kernel_err(result, prob):
    pred = result.c @ eval_feature(
        result.theta, 2, prob["x"], result.modes,
        mu0=prob["mu_true"] if result.mu_fixed else None)
    return np.linalg.norm(pred - prob["phi"]) / np.linalg.norm(prob["phi"])


def test_fit_row_recovers_synthetic_fixed_mu():
    """mu0 = the true center, mu='fixed': the stream must recover the
    exactly-representable row, fire the target certificate early, and
    return coherent audit fields."""
    rng = np.random.default_rng(0)
    prob = _make_row(rng)
    result = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"],
                     prob["mu_true"], MODES,
                     diag_index=prob["diag_index"],
                     config=RowFitConfig(mu="fixed"))
    assert isinstance(result, RowFitResult)
    assert result.score_kind == "cv"
    assert result.mu_fixed and not result.released
    assert result.score < 1e-6
    assert result.stop_reason == "target"
    assert len(result.candidates) <= 3   # the certificate fired early
    assert _kernel_err(result, prob) < 1e-5
    assert abs(result.s[0] - prob["s_true"]) < 1e-5
    assert np.allclose(result.mu, prob["mu_true"])
    assert result.modes == MODES
    assert result.candidates[result.winner].score == result.score


def test_fit_row_release_recovers_offset_center():
    """The row's true center is off the given mu0 by ~0.1 (within the
    free-mu basin -- empirically ~half the smaller kernel sigma; larger
    offsets land in a genuine local minimum and are the backprojection
    workflow's job, tested below). fixed_then_release must accept the
    release and recover center and kernel. target_score is disabled so
    the dipole modes' first-order absorption of the center offset cannot
    end the search before the release stage runs."""
    rng = np.random.default_rng(1)
    mu_true = np.array([0.35, -0.2])
    prob = _make_row(rng, mu_true=mu_true)
    mu0 = mu_true + 0.1 * np.array([0.78, -0.62])

    res_rel = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"], mu0,
                      MODES, diag_index=prob["diag_index"],
                      config=RowFitConfig(mu="fixed_then_release",
                                          target_score=None))
    assert res_rel.released and not res_rel.mu_fixed
    assert np.linalg.norm(res_rel.mu - mu_true) < 1e-3
    assert _kernel_err(res_rel, prob) < 1e-4

    res_fix = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"], mu0,
                      MODES, diag_index=prob["diag_index"],
                      config=RowFitConfig(mu="fixed", target_score=None))
    assert res_rel.score < res_fix.score   # releasing genuinely helped


def test_fit_row_backprojection_workflow_rescues_bad_center():
    """mu0 badly wrong (~0.3, outside the release basin): the plain fit
    cannot reach an exact fit. The intended remedy -- caller backprojects
    the row, estimates (mu, Sigma), and passes them as mu0/sigma0 --
    recovers the row exactly (the sigma0 rung's release carries the fit
    into the true basin even though the backprojected center estimate is
    itself rough at this probe count)."""
    rng = np.random.default_rng(1)
    mu_true = np.array([0.35, -0.2])
    prob = _make_row(rng, mu_true=mu_true)
    mu0_bad = mu_true + np.array([0.25, -0.2])

    res_bad = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"], mu0_bad,
                      MODES, diag_index=prob["diag_index"])
    assert res_bad.score > 1e-3   # stuck short of exact

    r_hat = backproject_row(prob["z"], prob["y"])
    mu_bp, sigma_bp = row_moments(prob["x"], r_hat,
                                  diag_index=prob["diag_index"],
                                  rel_threshold=0.05, noise_mad=3.0)
    res_wf = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"], mu_bp,
                     MODES, diag_index=prob["diag_index"], sigma0=sigma_bp)
    assert res_wf.released
    assert res_wf.score < 1e-6
    assert np.linalg.norm(res_wf.mu - mu_true) < 1e-3
    assert _kernel_err(res_wf, prob) < 1e-4


def test_fit_row_sigma0_tried_first():
    """A caller-supplied sigma0 heads the candidate order; a correct one
    fires the target certificate immediately."""
    rng = np.random.default_rng(2)
    prob = _make_row(rng)
    _, L_true = unpack_theta(prob["theta_true"], 2)
    result = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"],
                     prob["mu_true"], MODES,
                     diag_index=prob["diag_index"],
                     sigma0=L_true @ L_true.T,
                     config=RowFitConfig(mu="fixed"))
    assert result.candidates[0].label == "sigma0"
    assert result.stop_reason == "target"
    assert len(result.candidates) == 1
    assert result.score < 1e-6


def test_fit_row_mixed_ladder_enumeration():
    """With the target certificate disabled, the full mixed ladder runs:
    one window rung per circle rung; window_shape_rungs=False removes
    them."""
    rng = np.random.default_rng(7)
    prob = _make_row(rng)
    res = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"],
                  prob["mu_true"], MODES, diag_index=prob["diag_index"],
                  config=RowFitConfig(mu="fixed", target_score=None))
    labels = [c.label for c in res.candidates]
    n_circle = sum(l.startswith("circle") for l in labels)
    n_window = sum(l.startswith("window") for l in labels)
    assert n_circle == n_window > 0
    assert res.score < 1e-6

    res2 = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"],
                   prob["mu_true"], MODES, diag_index=prob["diag_index"],
                   config=RowFitConfig(mu="fixed", target_score=None,
                                       window_shape_rungs=False))
    assert not any(c.label.startswith("window") for c in res2.candidates)


def test_fit_row_mode_ladder_selects_simplest():
    """A row built from level<=1 modes, fit with mode_levels=[0, 1, 2]:
    level 0 cannot represent it, levels 1 and 2 both can (nested), and
    the simplicity tie-break must pick level 1. The warm-start candidate
    appears at the later levels."""
    rng = np.random.default_rng(3)
    prob = _make_row(rng)
    res = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"],
                  prob["mu_true"], modes=None,
                  diag_index=prob["diag_index"],
                  config=RowFitConfig(mu="fixed", target_score=None,
                                      mode_levels=[0, 1, 2],
                                      tie_delta=1e-3))
    assert res.modes == MODES                     # level<=1, same order
    assert res.candidates[res.winner].modes_label == "level<=1"
    assert res.skipped == []
    assert any(c.label.startswith("warm") for c in res.candidates)
    # level<=2 candidates fit at least as well in-sample, but simplicity
    # wins the tie
    assert res.score < 1e-3


def test_fit_row_counting_rule_skips_levels():
    """A mode level the probe budget cannot support (k < 2*(m + extra +
    P)) is skipped, not fit."""
    rng = np.random.default_rng(4)
    prob = _make_row(rng, k=14)   # level<=1 needs 2*(3+1+3)=14 <= 14; ok
    res = fit_row(prob["x"], prob["m2"], prob["z"], prob["y"],
                  prob["mu_true"], modes=None,
                  diag_index=prob["diag_index"],
                  config=RowFitConfig(mu="fixed", target_score=None,
                                      mode_levels=[1, 3]))
    assert res.skipped == ["level<=3"]
    assert all(c.modes_label == "level<=1" for c in res.candidates)


def test_window_shape_mass_weighted_geometry():
    """_window_shape must measure the window REGION's shape, independent
    of mesh grading: a 4:1 rectangle sampled uniformly and sampled with a
    4x-denser left half (masses = local cell areas) must give the same
    shape, matching the rectangle's aspect and orientation -- while the
    unweighted point covariance of the graded cloud gets the shape wrong."""
    rng = np.random.default_rng(6)
    a, b = 2.0, 0.5   # half-widths: aspect 4:1 along x

    K1 = 4000
    x1 = np.stack([rng.uniform(-a, a, K1), rng.uniform(-b, b, K1)])
    m1 = np.full(K1, 4 * a * b / K1)

    KL, KR = 3200, 800
    xl = np.stack([rng.uniform(-a, 0, KL), rng.uniform(-b, b, KL)])
    xr = np.stack([rng.uniform(0, a, KR), rng.uniform(-b, b, KR)])
    x2 = np.concatenate([xl, xr], axis=1)
    m2 = np.concatenate([np.full(KL, a * 2 * b / KL),
                         np.full(KR, a * 2 * b / KR)])

    def aspect_and_dir(L):
        S = L @ L.T
        w, v = np.linalg.eigh(S)
        return np.sqrt(w[-1] / w[0]), v[:, -1]

    for x, m in [(x1, m1), (x2, m2)]:
        asp, vmax = aspect_and_dir(_window_shape(x, m))
        assert abs(asp - a / b) < 0.15 * (a / b)
        assert abs(vmax[0]) > 0.99   # major axis along x

    # the unweighted covariance of the graded cloud is meaningfully wrong
    S_unw = np.cov(x2)
    w_unw = np.linalg.eigvalsh(S_unw)
    assert abs(np.sqrt(w_unw[-1] / w_unw[0]) - a / b) > 0.4


def test_row_moments_mass_quadrature():
    """The mass subtlety: raw row values already carry the lumped-mass
    quadrature weight, so row_moments needs no mass vector -- moments
    computed on two very different point densities (masses = local cell
    areas) must agree with each other and with the continuum answer."""
    mu_true = np.array([0.2, -0.1])
    ang = 0.5
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    Sig_true = R @ np.diag([0.30 ** 2, 0.15 ** 2]) @ R.T
    Sinv = np.linalg.inv(Sig_true)

    def kernel(x):
        d = x - mu_true[:, None]
        return np.exp(-0.5 * np.einsum("i...,ij,j...->...", d, Sinv, d))

    def moments_on(x, m):
        raw = m * kernel(x)                      # m_rho = 1
        return row_moments(x, raw, rel_threshold=0.0)

    rng = np.random.default_rng(3)
    # uniform density
    K1 = 4000
    x1 = rng.uniform(-1.2, 1.2, size=(2, K1))
    m1 = np.full(K1, (2.4 * 2.4) / K1)
    mu1, S1 = moments_on(x1, m1)
    # strongly nonuniform density: left half 4x denser, masses = cell areas
    KL, KR = 3200, 800
    xl = np.stack([rng.uniform(-1.2, 0.0, KL), rng.uniform(-1.2, 1.2, KL)])
    xr = np.stack([rng.uniform(0.0, 1.2, KR), rng.uniform(-1.2, 1.2, KR)])
    x2 = np.concatenate([xl, xr], axis=1)
    m2 = np.concatenate([np.full(KL, (1.2 * 2.4) / KL),
                         np.full(KR, (1.2 * 2.4) / KR)])
    mu2, S2 = moments_on(x2, m2)

    for mu_hat, S_hat in [(mu1, S1), (mu2, S2)]:
        assert np.linalg.norm(mu_hat - mu_true) < 0.03
        assert np.linalg.norm(S_hat - Sig_true) < 0.15 * np.linalg.norm(Sig_true)
    # and the two discretizations agree with each other
    assert np.linalg.norm(S1 - S2) < 0.15 * np.linalg.norm(Sig_true)


def test_row_moments_diag_exclusion():
    """A diagonal spike left inside the weights drags mu toward the node
    and shrinks Sigma; diag_index must neutralize it."""
    rng = np.random.default_rng(4)
    mu_true = np.array([0.3, 0.1])
    K = 2000
    x = rng.uniform(-1.2, 1.2, size=(2, K))
    m = np.full(K, (2.4 * 2.4) / K)
    d = x - mu_true[:, None]
    raw = m * np.exp(-0.5 * (d[0] ** 2 + d[1] ** 2) / 0.2 ** 2)
    diag_index = int(np.argmin(np.linalg.norm(x - 0.9, axis=0)))
    raw_spiked = raw.copy()
    raw_spiked[diag_index] += 50.0 * np.abs(raw).sum()

    mu_bad, S_bad = row_moments(x, raw_spiked, rel_threshold=0.0)
    mu_ok, S_ok = row_moments(x, raw_spiked, diag_index=diag_index,
                              rel_threshold=0.0)
    assert np.linalg.norm(mu_bad - mu_true) > 0.5        # poisoned
    assert np.linalg.norm(mu_ok - mu_true) < 0.02        # cured
    assert abs(np.trace(S_ok) - 2 * 0.2 ** 2) < 0.1 * 2 * 0.2 ** 2


def test_backprojection_end_to_end():
    """backproject_row + row_moments from k random probes approximates the
    measured-row moments: the intended sigma0 workflow."""
    rng = np.random.default_rng(5)
    prob = _make_row(rng, K=1500, k=600)
    r_hat = backproject_row(prob["z"], prob["y"])

    # reference: moments of the exact raw row (spike excluded)
    row_mass = prob["m2"][prob["diag_index"]]
    raw_row = row_mass * prob["m2"] * prob["phi"]
    mu_ref, S_ref = row_moments(prob["x"], raw_row, rel_threshold=0.0)

    mu_hat, S_hat = row_moments(prob["x"], r_hat,
                                diag_index=prob["diag_index"],
                                rel_threshold=0.05, noise_mad=3.0)
    # an INIT-quality estimate, not a fit: coarse tolerances by design
    assert np.linalg.norm(mu_hat - mu_ref) < 0.25
    assert np.linalg.norm(S_hat - S_ref) < 0.5 * np.linalg.norm(S_ref)


if __name__ == "__main__":
    test_fit_row_recovers_synthetic_fixed_mu()
    test_fit_row_release_recovers_offset_center()
    test_fit_row_backprojection_workflow_rescues_bad_center()
    test_fit_row_sigma0_tried_first()
    test_fit_row_mixed_ladder_enumeration()
    test_fit_row_mode_ladder_selects_simplest()
    test_fit_row_counting_rule_skips_levels()
    test_window_shape_mass_weighted_geometry()
    test_row_moments_mass_quadrature()
    test_row_moments_diag_exclusion()
    test_backprojection_end_to_end()
    print("all row_fit checks passed")
