"""Tests for ellipsoid_transform.py: adjoint (JVP/VJP transpose) consistency
and finite-difference checks, at both the stage-1 (pullback) level and the
full theta-composed level.

Run directly (`python test_ellipsoid_transform.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from ellipsoid_transform import (
    pullback,
    pullback_jvp,
    pullback_vjp,
    theta_size,
    unpack_theta,
    eval_T,
    jvp_T,
    vjp_T,
    jacobian_tensor_forward,
    jacobian_tensor_reverse,
)

FD_STEP = 1e-6
FD_TOL = 1e-5  # loosened vs lg_functions' 1e-6: the pullback divides by L_ii,
                # so relative FD error is a bit worse conditioned in general


def _random_L(N, rng):
    diag = rng.uniform(0.5, 2.0, size=N)
    L = np.diag(diag)
    for i in range(1, N):
        for j in range(i):
            L[i, j] = rng.uniform(-0.5, 0.5)
    return L


def _random_batch(N, K, rng):
    """Array of shape (N, K): K random points in N dimensions."""
    return rng.uniform(-1.5, 1.5, size=(N, K))


# --------------------------------------------------------------------------
# Stage 1 (pullback) checks, bypassing theta entirely.
# --------------------------------------------------------------------------

def test_pullback_jvp_vjp_adjoint_consistency():
    rng = np.random.default_rng(0)
    for N in [1, 2, 3, 4]:
        mu = rng.uniform(-1, 1, size=N)
        L = _random_L(N, rng)
        x = _random_batch(N, 20, rng)
        dmu = rng.uniform(-1, 1, size=N)
        dL = rng.uniform(-1, 1, size=(N, N))
        w = _random_batch(N, 20, rng)

        du = pullback_jvp(mu, L, dmu, dL, x)
        lhs = np.sum(w * du, axis=0)  # pointwise, shape (20,)

        w_mu, w_L = pullback_vjp(mu, L, x, w)
        rhs = np.sum(w_mu * dmu.reshape(N, 1), axis=0)
        rhs = rhs + np.sum(w_L * dL.reshape(N, N, 1), axis=(0, 1))

        err = np.max(np.abs(lhs - rhs))
        assert err < 1e-10, f"N={N}: adjoint mismatch, max err={err:.3e}"


def test_pullback_jvp_matches_finite_differences():
    rng = np.random.default_rng(1)
    for N in [1, 2, 3, 4]:
        mu = rng.uniform(-1, 1, size=N)
        L = _random_L(N, rng)
        x = _random_batch(N, 10, rng)

        # perturb mu
        for k in range(N):
            dmu = np.zeros(N)
            dmu[k] = 1.0
            dL = np.zeros((N, N))
            analytic = pullback_jvp(mu, L, dmu, dL, x)

            mu_p, mu_m = mu.copy(), mu.copy()
            mu_p[k] += FD_STEP
            mu_m[k] -= FD_STEP
            fd = (pullback(mu_p, L, x) - pullback(mu_m, L, x)) / (2 * FD_STEP)
            err = np.max(np.abs(analytic - fd))
            assert err < FD_TOL, f"N={N} dmu[{k}]: err={err:.3e}"

        # perturb each lower-triangular entry of L (including diagonal)
        for a in range(N):
            for b in range(a + 1):
                dmu = np.zeros(N)
                dL = np.zeros((N, N))
                dL[a, b] = 1.0
                analytic = pullback_jvp(mu, L, dmu, dL, x)

                L_p, L_m = L.copy(), L.copy()
                L_p[a, b] += FD_STEP
                L_m[a, b] -= FD_STEP
                fd = (pullback(mu, L_p, x) - pullback(mu, L_m, x)) / (2 * FD_STEP)
                err = np.max(np.abs(analytic - fd))
                assert err < FD_TOL, f"N={N} dL[{a},{b}]: err={err:.3e}"


# --------------------------------------------------------------------------
# Full composition (theta-level) checks.
# --------------------------------------------------------------------------

def _random_theta(N, mu0, rng):
    P = theta_size(N, mu0)
    theta = rng.uniform(-0.3, 0.3, size=P)
    # keep log-diagonal entries modest so L doesn't get near-singular
    idx = N if mu0 is None else 0
    theta[idx:idx + N] = rng.uniform(-0.3, 0.3, size=N)
    return theta


def test_theta_jvp_vjp_adjoint_consistency():
    rng = np.random.default_rng(2)
    for N in [1, 2, 3, 4]:
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            x = _random_batch(N, 15, rng)
            dtheta = rng.uniform(-1, 1, size=P)
            w = _random_batch(N, 15, rng)

            du = jvp_T(theta, dtheta, N, x, mu0=mu0)
            lhs = np.sum(w * du, axis=0)

            dtheta_batched = vjp_T(theta, N, x, w, mu0=mu0)  # (P, K)
            rhs = np.sum(dtheta_batched * dtheta.reshape(P, 1), axis=0)

            err = np.max(np.abs(lhs - rhs))
            assert err < 1e-10, f"N={N} mu0={mu0}: adjoint mismatch, max err={err:.3e}"


def test_theta_jvp_matches_finite_differences():
    rng = np.random.default_rng(3)
    for N in [1, 2, 3, 4]:
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            x = _random_batch(N, 10, rng)

            for p in range(P):
                dtheta = np.zeros(P)
                dtheta[p] = 1.0
                analytic = jvp_T(theta, dtheta, N, x, mu0=mu0)

                theta_p, theta_m = theta.copy(), theta.copy()
                theta_p[p] += FD_STEP
                theta_m[p] -= FD_STEP
                fd = (eval_T(theta_p, N, x, mu0=mu0) - eval_T(theta_m, N, x, mu0=mu0)) / (2 * FD_STEP)
                err = np.max(np.abs(analytic - fd))
                assert err < FD_TOL, f"N={N} mu0={mu0} p={p}: err={err:.3e}"


def test_jacobian_tensor_forward_matches_reverse():
    rng = np.random.default_rng(4)
    for N in [1, 2, 3, 4]:
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            x = _random_batch(N, 12, rng)

            fwd = jacobian_tensor_forward(theta, N, x, mu0=mu0)
            rev = jacobian_tensor_reverse(theta, N, x, mu0=mu0)
            err = np.max(np.abs(fwd - rev))
            assert err < 1e-10, f"N={N} mu0={mu0}: forward/reverse tensor mismatch {err:.3e}"


def test_jacobian_tensor_matches_finite_differences():
    rng = np.random.default_rng(5)
    for N in [1, 2, 3, 4]:
        for mu0 in [None, rng.uniform(-1, 1, size=N)]:
            theta = _random_theta(N, mu0, rng)
            P = theta_size(N, mu0)
            x = _random_batch(N, 8, rng)

            tensor = jacobian_tensor_forward(theta, N, x, mu0=mu0)  # (N, P, K)
            for p in range(P):
                theta_p, theta_m = theta.copy(), theta.copy()
                theta_p[p] += FD_STEP
                theta_m[p] -= FD_STEP
                up = eval_T(theta_p, N, x, mu0=mu0)
                um = eval_T(theta_m, N, x, mu0=mu0)
                fd = (up - um) / (2 * FD_STEP)  # (N, K)
                err = np.max(np.abs(tensor[:, p, :] - fd))
                assert err < FD_TOL, f"N={N} mu0={mu0} p={p}: err={err:.3e}"


if __name__ == "__main__":
    test_pullback_jvp_vjp_adjoint_consistency()
    test_pullback_jvp_matches_finite_differences()
    test_theta_jvp_vjp_adjoint_consistency()
    test_theta_jvp_matches_finite_differences()
    test_jacobian_tensor_forward_matches_reverse()
    test_jacobian_tensor_matches_finite_differences()
    print("all ellipsoid_transform checks passed")
