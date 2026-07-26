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
from lg_functions import (
    eval_lg_basis, eval_lg_nd, genlaguerre, grad_eval_lg_nd, grad_lg_basis,
    modes_up_to_level, vjp_lg_basis,
)

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


def _mode_sets(N):
    """Mode sets spanning the shapes the factorized evaluator must handle:
    a singleton, a pure-radial family (one shell, many p), a pure-angular
    family (one p, many shells), complete shells, and a list with
    duplicates in non-canonical order."""
    scrambled = list(reversed(modes_up_to_level(N, 3)))
    return [
        [(0, 0, 0)],
        [(p, 0, 0) for p in range(6)],
        [(0, ell, m) for ell in range(4) for m in range(num_harmonics(N, ell))],
        modes_up_to_level(N, 4),
        modes_up_to_level(N, 6),
        scrambled + scrambled[:3],
    ]


def test_batched_matches_one_at_a_time_exactly():
    """eval_lg_basis / grad_lg_basis share r^2, the Gaussian, each harmonic
    across p and each Laguerre recurrence across m and p. All of that is
    reuse of identical expressions, never reassociation, so the batched
    results must be EXACTLY equal to the one-at-a-time reference -- not
    merely close. Equality is the contract; a tolerance here would hide
    exactly the kind of drift this test exists to catch."""
    rng = np.random.default_rng(4)
    checked = 0
    for N in [1, 2, 3, 4]:
        batches = [np.zeros(N), rng.normal(size=N), rng.normal(size=(N, 9)),
                   rng.normal(size=(N, 3, 4)), rng.normal(size=(N, 50)) * 3.0]
        for modes in _mode_sets(N):
            for u in batches:
                got = eval_lg_basis(modes, u)
                want = np.stack([eval_lg_nd(p, ell, m, u)
                                 for (p, ell, m) in modes], axis=0)
                assert got.shape == want.shape
                assert np.array_equal(got, want), (
                    f"eval N={N}, {len(modes)} modes, batch {np.shape(u)[1:]}: "
                    f"max diff {np.max(np.abs(got - want)):.3e}")

                got = grad_lg_basis(modes, u)
                want = np.stack([grad_eval_lg_nd(p, ell, m, u)
                                 for (p, ell, m) in modes], axis=0)
                assert got.shape == want.shape
                assert np.array_equal(got, want), (
                    f"grad N={N}, {len(modes)} modes, batch {np.shape(u)[1:]}: "
                    f"max diff {np.max(np.abs(got - want)):.3e}")
                checked += 1
    print(f"  {checked} (mode set, batch) pairs exactly equal")


def test_batched_handles_empty_mode_set():
    u = np.zeros((2, 7))
    assert eval_lg_basis([], u).shape == (0, 7)
    assert grad_lg_basis([], u).shape == (0, 2, 7)
    assert vjp_lg_basis([], u, np.zeros((0, 7))).shape == (2, 7)


def test_vjp_matches_the_contracted_gradient():
    """vjp_lg_basis regroups sum_i w_i grad psi_i by shell instead of
    accumulating per mode. That reassociates the sum, so -- unlike the
    other batched paths -- this one is checked to roundoff rather than
    exactly. Accuracy is a wash between the two forms (measured against an
    extended-precision reference, both sit at ~1.5 ulp); the regrouping is
    for operation count and memory, not numerics."""
    rng = np.random.default_rng(6)
    worst = 0.0
    for N in [1, 2, 3, 4]:
        for modes in _mode_sets(N):
            for K in [1, 13, 200]:
                u = rng.normal(size=(N, K))
                w = rng.normal(size=(len(modes), K))

                got = vjp_lg_basis(modes, u, w)
                grad = grad_lg_basis(modes, u)
                want = np.zeros_like(u)
                for i in range(len(modes)):
                    want = want + w[i] * grad[i]

                assert got.shape == want.shape
                scale = max(float(np.max(np.abs(want))), 1e-300)
                err = float(np.max(np.abs(got - want))) / scale
                worst = max(worst, err)
                assert err < 1e-12, (
                    f"N={N}, {len(modes)} modes, K={K}: relative "
                    f"disagreement {err:.3e}")
    print(f"  worst relative disagreement vs the per-mode sum: {worst:.3e}")


def test_vjp_is_adjoint_to_the_gradient():
    """<w, J du> == <vjp(w), du> for random w, du -- the adjoint-consistency
    half of this repo's derivative-testing convention (the finite-difference
    half reaches vjp_lg_basis through grad_eval_lg_nd, which is FD-checked
    and which grad_lg_basis reproduces exactly)."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for N in [1, 2, 3, 4]:
        for modes in _mode_sets(N):
            K = 17
            u = rng.normal(size=(N, K))
            w = rng.normal(size=(len(modes), K))
            du = rng.normal(size=(N, K))

            grad = grad_lg_basis(modes, u)                   # (m, N, K)
            directional = np.sum(grad * du, axis=1)          # (m, K)
            lhs = float(np.sum(w * directional))
            rhs = float(np.sum(vjp_lg_basis(modes, u, w) * du))

            scale = max(abs(lhs), abs(rhs), 1e-300)
            err = abs(lhs - rhs) / scale
            worst = max(worst, err)
            assert err < 1e-11, (
                f"N={N}, {len(modes)} modes: <w, J du>={lhs!r} vs "
                f"<vjp(w), du>={rhs!r}, relative gap {err:.3e}")
    print(f"  worst adjoint-consistency gap: {worst:.3e}")


if __name__ == "__main__":
    test_genlaguerre_matches_closed_forms()
    test_batched_matches_one_at_a_time_exactly()
    test_batched_handles_empty_mode_set()
    test_vjp_matches_the_contracted_gradient()
    test_vjp_is_adjoint_to_the_gradient()
    test_modes_are_orthonormal_in_L2()
    test_gradient_matches_finite_differences()
    print("all LG mode checks passed")
