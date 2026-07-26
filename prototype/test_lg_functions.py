"""Checks on the Laguerre-Gaussian modes: L^2(R^N) orthonormality by exact
quadrature, the Laguerre recurrence against closed forms, and
finite-difference checks for grad_eval_lg_nd against eval_lg_nd.

All self-contained -- no comparison against a stored reference.

Run directly (`python test_lg_functions.py`) or via pytest
(`pytest prototype/`) -- functions are plain asserts, no pytest dependency.
"""
import itertools
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from harmonic_polynomials import num_harmonics
from lg_functions import eval_lg_nd, genlaguerre, grad_eval_lg_nd, modes_up_to_level

FD_STEP = 1e-5
FD_TOL = 1e-6  # well above the ~1e-10 central-difference floor measured at FD_STEP


def _central_diff_grad(p, ell, m, u, h=FD_STEP):
    N = u.shape[0]
    grad = np.zeros(N)
    for k in range(N):
        u_plus = u.copy()
        u_plus[k] += h
        u_minus = u.copy()
        u_minus[k] -= h
        f_plus = eval_lg_nd(p, ell, m, u_plus)
        f_minus = eval_lg_nd(p, ell, m, u_minus)
        grad[k] = (f_plus - f_minus) / (2 * h)
    return grad


def _assert_gradient_matches_fd(p, ell, m, u):
    u = np.asarray(u, dtype=float)
    analytic = grad_eval_lg_nd(p, ell, m, u)
    fd = _central_diff_grad(p, ell, m, u)
    err = np.max(np.abs(analytic - fd))
    scale = max(1.0, np.max(np.abs(fd)))
    assert err / scale < FD_TOL, (
        f"N={u.shape[0]} p={p} ell={ell} m={m} u={u}: "
        f"analytic={analytic} fd={fd} rel_err={err / scale:.3e}"
    )


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    for N in [1, 2, 3, 4]:
        for ell in range(4):
            for p in range(4):
                for m in range(num_harmonics(N, ell)):
                    _assert_gradient_matches_fd(p, ell, m, np.zeros(N))  # origin
                    for _ in range(3):
                        u = rng.uniform(-1.5, 1.5, size=N)
                        _assert_gradient_matches_fd(p, ell, m, u)


def test_genlaguerre_matches_closed_forms():
    """The three-term recurrence against the textbook low-order formulas --
    an independent check that the recurrence is transcribed correctly."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 6.0, size=50)
    for alpha in [0.0, 0.5, 1.0, 2.5, 4.0]:
        closed = [
            np.ones_like(x),
            1.0 + alpha - x,
            (x**2 / 2.0) - (alpha + 2.0) * x + (alpha + 1.0) * (alpha + 2.0) / 2.0,
            (-x**3 / 6.0
             + (alpha + 3.0) * x**2 / 2.0
             - (alpha + 2.0) * (alpha + 3.0) * x / 2.0
             + (alpha + 1.0) * (alpha + 2.0) * (alpha + 3.0) / 6.0),
        ]
        for p, expected in enumerate(closed):
            got = genlaguerre(p, alpha, x)
            err = np.max(np.abs(got - expected)) / max(1.0, np.max(np.abs(expected)))
            assert err < 1e-12, f"L_{p}^{alpha}: relative error {err:.3e}"


def _gauss_hermite(n):
    """Nodes and weights for int_R f(t) exp(-t^2) dt, exact for polynomial f
    of degree <= 2n-1, via Golub-Welsch on the Hermite Jacobi matrix
    (diagonal 0, off-diagonal sqrt(k/2)). Uses only a symmetric eigenvalue
    solve, which is what the C++ port would do with Eigen -- no special
    function library."""
    beta = np.sqrt(np.arange(1, n) / 2.0)
    J = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, vectors = np.linalg.eigh(J)
    weights = math.sqrt(math.pi) * vectors[0, :] ** 2
    return nodes, weights


def test_modes_are_orthonormal_in_L2():
    """int psi_i psi_j du = delta_ij over R^N -- the defining property of
    the LG basis, and the one check that exercises the harmonic table, the
    Laguerre recurrence, the normalization constant and the
    alpha = ell + N/2 - 1 bookkeeping all at once.

    The integrand psi_i psi_j exp(r^2) is a polynomial of total degree
    level_i + level_j, so tensor-product Gauss-Hermite with enough nodes
    evaluates the integral exactly (to roundoff) rather than approximately.
    """
    n_nodes = 12
    cases = {1: 10, 2: 8, 3: 6}  # N -> max oscillator level

    nodes, weights = _gauss_hermite(n_nodes)
    for N, max_level in cases.items():
        assert 2 * n_nodes - 1 >= 2 * max_level, (
            f"N={N}: {n_nodes} nodes are not enough for exactness at "
            f"level {max_level}"
        )
        grid = np.array(list(itertools.product(nodes, repeat=N))).T  # (N, n^N)
        w = np.prod(np.array(list(itertools.product(weights, repeat=N))), axis=1)
        # psi_i psi_j carries exp(-r^2); the quadrature weight supplies it too,
        # so divide it back out to leave the polynomial the rule integrates.
        w = w * np.exp(np.sum(grid * grid, axis=0))

        modes = modes_up_to_level(N, max_level)
        values = np.stack([eval_lg_nd(p, ell, m, grid) for (p, ell, m) in modes])
        gram = (values * w) @ values.T

        err = np.max(np.abs(gram - np.eye(len(modes))))
        assert err < 1e-9, (
            f"N={N}, {len(modes)} modes up to level {max_level}: "
            f"max |Gram - I| = {err:.3e}"
        )
        print(f"  N={N}: {len(modes)} modes orthonormal to {err:.2e}")


if __name__ == "__main__":
    test_genlaguerre_matches_closed_forms()
    test_modes_are_orthonormal_in_L2()
    test_gradient_matches_finite_differences()
    print("all LG mode checks passed")
