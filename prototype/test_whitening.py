"""Tests for whitening.py.

Two kinds of check:
  1. The usual JVP/VJP adjoint-consistency and finite-difference checks for
     the whitened feature wrappers (mirroring test_lg_ellipsoid_feature.py).
  2. An end-to-end check (test_whitened_regression_reproduces_row_model):
     build an explicit H from the row model in docs/varpro-whitening-notes.tex
     eq. (1), apply it to a random probe to get y, and check that the
     whitened quantities (z_hat, y_hat, phi_hat, E_hat) satisfy the whitened
     regression eq. (7) to machine precision. This is the test that would
     have caught the missing sqrt(row_mass) factor on the extra-basis
     whitening -- a pure scaling bug that adjoint-consistency and FD checks
     on the feature alone can't see, since they never involve M1 or the
     extra basis at all.

Run directly (`python test_whitening.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from ellipsoid_transform import theta_size
from lg_ellipsoid_feature import eval_feature
from lg_harmonics_table import TABLE
from whitening import (
    whiten_data,
    whiten_extra,
    whiten_probes,
    whitened_eval_feature,
    whitened_jac_feature,
    whitened_jvp_feature,
    whitened_vjp_feature,
)

FD_STEP = 1e-6
FD_TOL = 1e-5


def _random_theta(N, mu0, rng):
    P = theta_size(N, mu0)
    theta = rng.uniform(-0.3, 0.3, size=P)
    idx = N if mu0 is None else 0
    theta[idx:idx + N] = rng.uniform(-0.3, 0.3, size=N)
    return theta


def _some_modes(N, max_ell=3, max_p=2):
    modes = []
    for ell in range(max_ell + 1):
        _, rows = TABLE[(N, ell)]
        for m in range(len(rows)):
            for p in range(max_p + 1):
                modes.append((p, ell, m))
    return modes


def test_whitened_jvp_vjp_adjoint_consistency():
    rng = np.random.default_rng(0)
    for N in [1, 2, 3]:
        modes = _some_modes(N)
        row_mass = rng.uniform(0.5, 2.0)
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            K = 10
            x = rng.uniform(-1.5, 1.5, size=(N, K))
            m2_diag = rng.uniform(0.5, 2.0, size=K)
            dtheta = rng.uniform(-1, 1, size=P)
            w_hat = rng.uniform(-1, 1, size=(len(modes), K))

            dphi_hat = whitened_jvp_feature(theta, dtheta, N, x, row_mass, m2_diag, modes, mu0=mu0)
            lhs = np.sum(w_hat * dphi_hat, axis=0)

            dtheta_batched = whitened_vjp_feature(theta, N, x, row_mass, m2_diag, w_hat, modes, mu0=mu0)
            rhs = np.sum(dtheta_batched * dtheta.reshape(P, 1), axis=0)

            err = np.max(np.abs(lhs - rhs))
            assert err < 1e-9, f"N={N} mu0={mu0}: adjoint mismatch, max err={err:.3e}"


def test_whitened_jvp_matches_finite_differences():
    rng = np.random.default_rng(1)
    for N in [1, 2, 3]:
        modes = _some_modes(N)
        row_mass = rng.uniform(0.5, 2.0)
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            K = 8
            x = rng.uniform(-1.5, 1.5, size=(N, K))
            m2_diag = rng.uniform(0.5, 2.0, size=K)

            for p in range(P):
                dtheta = np.zeros(P)
                dtheta[p] = 1.0
                analytic = whitened_jvp_feature(theta, dtheta, N, x, row_mass, m2_diag, modes, mu0=mu0)

                theta_p, theta_m = theta.copy(), theta.copy()
                theta_p[p] += FD_STEP
                theta_m[p] -= FD_STEP
                fd = (
                    whitened_eval_feature(theta_p, N, x, row_mass, m2_diag, modes, mu0=mu0)
                    - whitened_eval_feature(theta_m, N, x, row_mass, m2_diag, modes, mu0=mu0)
                ) / (2 * FD_STEP)
                err = np.max(np.abs(analytic - fd))
                assert err < FD_TOL, f"N={N} mu0={mu0} p={p}: err={err:.3e}"


def test_whitened_jac_matches_whitened_jvp_columns():
    """Column-by-column agreement of the whitened Jacobian tensor with
    per-direction whitened JVP calls (themselves FD-verified above)."""
    rng = np.random.default_rng(3)
    for N in [1, 2, 3]:
        modes = _some_modes(N)
        row_mass = rng.uniform(0.5, 2.0)
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            K = 10
            x = rng.uniform(-1.5, 1.5, size=(N, K))
            m2_diag = rng.uniform(0.5, 2.0, size=K)

            jac = whitened_jac_feature(theta, N, x, row_mass, m2_diag, modes, mu0=mu0)
            assert jac.shape == (len(modes), P, K)
            for q in range(P):
                dtheta = np.zeros(P)
                dtheta[q] = 1.0
                col = whitened_jvp_feature(theta, dtheta, N, x, row_mass, m2_diag, modes, mu0=mu0)
                err = np.max(np.abs(jac[:, q] - col))
                assert err < 1e-12, f"N={N} mu0={mu0} q={q}: err={err:.3e}"


def test_whitened_regression_reproduces_row_model():
    """Build H[rho, :] explicitly from the row model (varpro-whitening-notes
    eq. 1), apply it to a random probe to get y, and check the whitened
    quantities satisfy the whitened regression (eq. 7) to machine precision."""
    rng = np.random.default_rng(2)
    for N in [1, 2, 3]:
        modes = _some_modes(N, max_ell=2, max_p=1)
        n_modes = len(modes)
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            for num_extra, extra_positions in [(1, [0]), (2, [0, 3])]:
                theta = _random_theta(N, mu0, rng)
                row_mass = rng.uniform(0.5, 2.0)
                K = 12
                x = rng.uniform(-1.5, 1.5, size=(N, K))
                m2_diag = rng.uniform(0.5, 2.0, size=K)

                # extra basis: one-hot indicators at the given positions
                E = np.zeros((num_extra, K))
                for d, pos in enumerate(extra_positions):
                    E[d, pos] = 1.0

                c = rng.uniform(-1, 1, size=n_modes)
                s = rng.uniform(-1, 1, size=num_extra)

                # H[rho, j] = row_mass * m2_diag[j] * sum_i c_i phi_i(x_j)
                #             + row_mass * sum_d s_d E[d, j]
                phi_raw = eval_feature(theta, N, x, modes, mu0=mu0)  # (n_modes, K)
                smooth_row = row_mass * m2_diag * (c @ phi_raw)      # (K,)
                extra_row = row_mass * (s @ E)                       # (K,)
                H_row = smooth_row + extra_row                       # (K,)

                z = rng.standard_normal(K)
                y = H_row @ z  # y_ell(rho), a single probe realization

                y_hat = whiten_data(y, row_mass)
                z_hat = whiten_probes(z, m2_diag)
                phi_hat = whitened_eval_feature(theta, N, x, row_mass, m2_diag, modes, mu0=mu0)
                E_hat = whiten_extra(E, row_mass, m2_diag)

                y_hat_pred = c @ (phi_hat @ z_hat) + s @ (E_hat @ z_hat)

                err = abs(y_hat_pred - y_hat)
                scale = max(1.0, abs(y_hat))
                assert err / scale < 1e-10, (
                    f"N={N} mu0={mu0} num_extra={num_extra}: "
                    f"whitened regression mismatch, y_hat={y_hat:.6e} "
                    f"pred={y_hat_pred:.6e} err={err:.3e}"
                )


if __name__ == "__main__":
    test_whitened_jvp_vjp_adjoint_consistency()
    test_whitened_jvp_matches_finite_differences()
    test_whitened_jac_matches_whitened_jvp_columns()
    test_whitened_regression_reproduces_row_model()
    print("all whitening checks passed")
