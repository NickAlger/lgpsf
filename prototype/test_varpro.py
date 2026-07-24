"""Tests for varpro.py's inner linear-algebra layer (step 1: the
"projection" in variable projection).

All pure linear algebra on random matrices -- no LG modes, no whitening,
matching the layer's own ignorance of them. The reference implementations
are brute force (numpy lstsq on the full/joint/augmented systems), so each
test checks our structured path against an unstructured one that computes
the same object a different way:

  - _inner_solve at ridge=0 vs plain lstsq (coefficients, residual,
    residual-orthogonal-to-range, range basis).
  - _inner_solve at ridge>0 vs the augmented least-squares system
    [A_s; sqrt(ridge) I] -- an independent formula for the same ridge
    solution.
  - Rank deficiency: SVD cutoff finds the right rank and still produces
    the exact least-squares residual.
  - The Frisch-Waugh-Lovell path (orthonormalize the extra block, project
    it out of everything, fit the smooth block alone, back-solve for s)
    vs the joint fit over [A, B] -- coefficient-by-coefficient and
    residual-by-residual. This is the identity that lets fit_varpro handle
    the extra basis entirely in preprocessing.

Run directly (`python test_varpro.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from varpro import _inner_solve, _orthonormal_range, _project_out


def test_project_out():
    rng = np.random.default_rng(0)
    k = 12
    Q = _orthonormal_range(rng.standard_normal((k, 3)))
    V = rng.standard_normal((k, 5))

    W = _project_out(Q, V)
    assert np.max(np.abs(Q.T @ W)) < 1e-12          # lands in range(Q)^perp
    assert np.allclose(_project_out(Q, W), W)        # idempotent

    v = rng.standard_normal(k)                       # plain vectors too
    assert np.max(np.abs(Q.T @ _project_out(Q, v))) < 1e-12

    Q_empty = np.zeros((k, 0))                       # no extra basis at all
    assert np.allclose(_project_out(Q_empty, V), V)


def test_orthonormal_range():
    rng = np.random.default_rng(1)
    k = 10
    B = rng.standard_normal((k, 3))

    Q = _orthonormal_range(B)
    assert Q.shape == (k, 3)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert np.allclose(Q @ (Q.T @ B), B)             # range preserved

    # exactly-dependent columns collapse to the true rank
    B_deficient = np.column_stack([B[:, 0], 2.0 * B[:, 0], B[:, 1]])
    Q2 = _orthonormal_range(B_deficient)
    assert Q2.shape == (k, 2)
    assert np.allclose(Q2 @ (Q2.T @ B_deficient), B_deficient)

    assert _orthonormal_range(np.zeros((k, 0))).shape == (k, 0)


def test_inner_solve_matches_lstsq():
    rng = np.random.default_rng(2)
    k, n = 30, 8
    # uneven column scales, to make sure equilibration is exercised (and
    # undone) rather than a no-op
    A = rng.standard_normal((k, n)) * rng.uniform(0.05, 20.0, size=n)
    y = rng.standard_normal(k)

    sol = _inner_solve(A, y, ridge=0.0)
    c_ref = np.linalg.lstsq(A, y, rcond=None)[0]

    assert np.allclose(sol.c, c_ref, rtol=1e-9, atol=1e-12)
    assert np.allclose(sol.residual, y - A @ c_ref, rtol=1e-9, atol=1e-12)
    # the defining property of the projection: residual orthogonal to
    # every column of A
    assert np.max(np.abs(A.T @ sol.residual)) < 1e-9
    # range basis: orthonormal, full rank here, spans range(A)
    assert sol.U.shape == (k, n)
    assert np.allclose(sol.U.T @ sol.U, np.eye(n), atol=1e-12)
    assert np.allclose(sol.U @ (sol.U.T @ A), A)


def test_inner_solve_ridge_matches_augmented_system():
    """Independent formula for the same object: the ridge solution on the
    equilibrated matrix solves the augmented least-squares problem
    [A_s; sqrt(ridge) I] c_s ~ [y; 0]."""
    rng = np.random.default_rng(3)
    k, n = 25, 6
    A = rng.standard_normal((k, n)) * rng.uniform(0.05, 20.0, size=n)
    y = rng.standard_normal(k)
    ridge = 1e-3  # big enough that forgetting it would fail loudly

    sol = _inner_solve(A, y, ridge=ridge)

    col_scale = np.linalg.norm(A, axis=0)
    A_aug = np.vstack([A / col_scale, np.sqrt(ridge) * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])
    c_ref = np.linalg.lstsq(A_aug, y_aug, rcond=None)[0] / col_scale

    assert np.allclose(sol.c, c_ref, rtol=1e-9, atol=1e-12)
    # residual is the plain model misfit at the ridge coefficients
    assert np.allclose(sol.residual, y - A @ sol.c, rtol=1e-9, atol=1e-12)


def test_inner_solve_rank_deficient():
    rng = np.random.default_rng(4)
    k, n = 20, 5
    A = rng.standard_normal((k, n))
    A[:, 3] = 2.0 * A[:, 1]  # exact rank deficiency
    y = rng.standard_normal(k)

    sol = _inner_solve(A, y, ridge=0.0)
    assert sol.U.shape[1] == n - 1  # numerical rank found

    # the residual is still the exact least-squares residual: projection
    # doesn't care which of the dependent columns "gets" the coefficient
    c_ref = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(sol.residual, y - A @ c_ref, rtol=1e-9, atol=1e-12)
    assert np.max(np.abs(A.T @ sol.residual)) < 1e-9


def test_fwl_residualize_then_fit_matches_joint_fit():
    """The Frisch-Waugh-Lovell identity, which is what lets fit_varpro
    handle the theta-independent extra block entirely in preprocessing:
    project range(B) out of y and A once, fit the smooth block alone,
    back-solve for s -- and get exactly the joint minimizer over [A, B],
    same c, same s, same residual."""
    rng = np.random.default_rng(5)
    k, n = 40, 7
    for num_extra in [0, 1, 2]:
        A = rng.standard_normal((k, n))
        B = rng.standard_normal((k, num_extra))
        y = rng.standard_normal(k)

        # brute-force joint fit
        coeff = np.linalg.lstsq(np.column_stack([A, B]), y, rcond=None)[0]
        c_joint, s_joint = coeff[:n], coeff[n:]
        r_joint = y - A @ c_joint - B @ s_joint

        # FWL path, as fit_varpro will do it
        Q_B = _orthonormal_range(B)
        y_tilde = _project_out(Q_B, y)
        A_tilde = _project_out(Q_B, A)
        sol = _inner_solve(A_tilde, y_tilde, ridge=0.0)
        s = np.linalg.lstsq(B, y - A @ sol.c, rcond=None)[0]

        assert np.allclose(sol.c, c_joint, rtol=1e-8, atol=1e-10), num_extra
        assert np.allclose(s, s_joint, rtol=1e-8, atol=1e-10), num_extra
        assert np.allclose(sol.residual, r_joint, rtol=1e-8, atol=1e-10), num_extra


if __name__ == "__main__":
    test_project_out()
    test_orthonormal_range()
    test_inner_solve_matches_lstsq()
    test_inner_solve_ridge_matches_augmented_system()
    test_inner_solve_rank_deficient()
    test_fwl_residualize_then_fit_matches_joint_fit()
    print("all varpro inner-layer checks passed")
