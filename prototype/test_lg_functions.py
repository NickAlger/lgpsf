"""Finite-difference checks for grad_eval_lg_nd against eval_lg_nd.

Run directly (`python test_lg_functions.py`) or via pytest
(`pytest prototype/`) -- functions are plain asserts, no pytest dependency.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from lg_functions import eval_lg_nd, grad_eval_lg_nd
from lg_harmonics_table import TABLE

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
            _, rows = TABLE[(N, ell)]
            if not rows:
                continue
            for p in range(4):
                for m in range(len(rows)):
                    _assert_gradient_matches_fd(p, ell, m, np.zeros(N))  # origin
                    for _ in range(3):
                        u = rng.uniform(-1.5, 1.5, size=N)
                        _assert_gradient_matches_fd(p, ell, m, u)


if __name__ == "__main__":
    test_gradient_matches_finite_differences()
    print("all gradient finite-difference checks passed")
