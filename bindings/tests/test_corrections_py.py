# SPDX-License-Identifier: MIT
"""Tests for the lgpsf.corrections boundary bindings.

The corrections layer is operator-blind: it consumes operators through
SymmetricOp (dimension + block matvec) and HrOracle (apply + tolerance
solve). Blocks are (N, m), one vector per COLUMN -- note this differs from
the point-batch convention of the main module.
"""
import numpy as np
import pytest

import lgpsf
from test_lgpsf_py import fit_synthetic, synthetic_operator

corr = lgpsf.corrections


def symmetric_matrix(n, rng):
    G = rng.normal(size=(n, n))
    return 0.5 * (G + G.T) + n * np.eye(n)


def test_a_python_callable_a_dense_wrap_and_a_sparse_wrap_agree():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    A = symmetric_matrix(30, rng)
    X = rng.normal(size=(30, 4))

    from_callable = corr.SymmetricOp(30, lambda block: A @ block)
    from_dense = corr.dense_op(A)
    from_sparse = corr.sparse_op(scipy_sparse.csr_matrix(A))

    assert from_callable.dim == 30
    # the callable round-trips numpy's own product, exactly; the dense and
    # sparse wraps compute in Eigen, so cross-library agreement is to
    # rounding, not bitwise (the C++ suite pins the in-Eigen exactness)
    np.testing.assert_array_equal(from_callable.apply(X), A @ X)
    scale = np.abs(A @ X).max()
    np.testing.assert_allclose(from_dense.apply(X), A @ X, atol=1e-13 * scale)
    np.testing.assert_allclose(from_sparse.apply(X), A @ X, atol=1e-13 * scale)


def test_symmetry_defect_flags_the_unsymmetrized_fit():
    # The check the assembly layer cannot do: an as-fitted operator handed
    # across the boundary by mistake measures orders of magnitude above a
    # weighted-symmetrized one.
    scipy_sparse = pytest.importorskip("scipy.sparse")
    op = synthetic_operator()
    fit = fit_synthetic(op)

    as_fitted = corr.sparse_op(
        scipy_sparse.csr_matrix(lgpsf.assemble_sparse(fit.model, np.inf)))
    symmetrized = corr.sparse_op(scipy_sparse.csr_matrix(
        lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Weighted)))

    clean = corr.symmetry_defect(symmetrized)
    dirty = corr.symmetry_defect(as_fitted)
    assert clean < 1e-13
    assert dirty > 1e-3

    # deterministic in the seed
    assert corr.symmetry_defect(as_fitted, pairs=4, seed=7) \
        == corr.symmetry_defect(as_fitted, pairs=4, seed=7)


def test_shape_contracts_surface_as_python_exceptions():
    op = corr.dense_op(np.eye(5))

    # caller-side: a wrong-height block (rows convention, say)
    with pytest.raises(ValueError):
        op.apply(np.zeros((3, 2)))

    # callable-side: a wrapped function that breaks its shape promise
    broken = corr.SymmetricOp(4, lambda X: np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        broken.apply(np.zeros((4, 1)))

    with pytest.raises(ValueError):
        corr.dense_op(np.zeros((3, 4)))


def test_hr_oracle_roundtrips_and_refuses_indefiniteness():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    n = 25
    rng = np.random.default_rng(1)
    main = 2.5 + 0.01 * np.arange(n)
    Hr = scipy_sparse.diags([-np.ones(n - 1), main, -np.ones(n - 1)],
                            [-1, 0, 1], format="csc")

    oracle = corr.sparse_hr_oracle(Hr)
    assert oracle.dim == n
    X = rng.normal(size=(n, 3))
    np.testing.assert_allclose(oracle.apply(X), Hr @ X,
                               atol=1e-13 * np.abs(Hr @ X).max())
    np.testing.assert_allclose(oracle.solve(oracle.apply(X), 1e-8), X,
                               atol=1e-10)

    with pytest.raises(ValueError):
        oracle.solve(X, 0.0)

    indefinite = Hr.tolil()
    indefinite[3, 3] = -50.0
    with pytest.raises(ValueError):
        corr.sparse_hr_oracle(indefinite.tocsc())


def test_an_oracle_from_python_callables_is_a_first_class_citizen():
    # The production shape: apply and solve are the consumer's own solver.
    n = 12
    d = 1.0 + np.arange(n)
    oracle = corr.HrOracle(n,
                           lambda X: d[:, None] * X,
                           lambda B, tol: B / d[:, None])
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n, 2))
    np.testing.assert_allclose(oracle.solve(oracle.apply(X), 1e-8), X,
                               atol=1e-14)
