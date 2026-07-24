"""Tests for lg_ellipsoid_feature.py: JVP/VJP adjoint-consistency and
finite-difference checks for the composed (LG mode) . (pullback) feature.

Run directly (`python test_lg_ellipsoid_feature.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from ellipsoid_transform import theta_size
from lg_ellipsoid_feature import eval_feature, jvp_feature, vjp_feature
from lg_harmonics_table import TABLE

FD_STEP = 1e-6
FD_TOL = 1e-5  # matches test_ellipsoid_transform.py's tolerance (pullback-based)


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


def test_jvp_vjp_adjoint_consistency():
    rng = np.random.default_rng(0)
    for N in [1, 2, 3, 4]:
        modes = _some_modes(N)
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            x = rng.uniform(-1.5, 1.5, size=(N, 10))
            dtheta = rng.uniform(-1, 1, size=P)
            w = rng.uniform(-1, 1, size=(len(modes), 10))

            dphi = jvp_feature(theta, dtheta, N, x, modes, mu0=mu0)  # (n_modes, K)
            lhs = np.sum(w * dphi, axis=0)  # (K,)

            dtheta_batched = vjp_feature(theta, N, x, w, modes, mu0=mu0)  # (P, K)
            rhs = np.sum(dtheta_batched * dtheta.reshape(P, 1), axis=0)

            err = np.max(np.abs(lhs - rhs))
            assert err < 1e-9, f"N={N} mu0={mu0}: adjoint mismatch, max err={err:.3e}"


def test_jvp_matches_finite_differences():
    rng = np.random.default_rng(1)
    for N in [1, 2, 3, 4]:
        modes = _some_modes(N)
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            x = rng.uniform(-1.5, 1.5, size=(N, 8))

            for p in range(P):
                dtheta = np.zeros(P)
                dtheta[p] = 1.0
                analytic = jvp_feature(theta, dtheta, N, x, modes, mu0=mu0)

                theta_p, theta_m = theta.copy(), theta.copy()
                theta_p[p] += FD_STEP
                theta_m[p] -= FD_STEP
                fd = (
                    eval_feature(theta_p, N, x, modes, mu0=mu0)
                    - eval_feature(theta_m, N, x, modes, mu0=mu0)
                ) / (2 * FD_STEP)
                err = np.max(np.abs(analytic - fd))
                assert err < FD_TOL, f"N={N} mu0={mu0} p={p}: err={err:.3e}"


if __name__ == "__main__":
    test_jvp_vjp_adjoint_consistency()
    test_jvp_matches_finite_differences()
    print("all lg_ellipsoid_feature checks passed")
