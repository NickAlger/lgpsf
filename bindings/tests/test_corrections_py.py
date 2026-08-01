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


def tridiag_spd(n):
    scipy_sparse = pytest.importorskip("scipy.sparse")
    main = 2.5 + 0.01 * np.arange(n)
    return scipy_sparse.diags([-np.ones(n - 1), main, -np.ones(n - 1)],
                              [-1, 0, 1], format="csc")


def test_mode_block_merge_represents_the_contribution():
    n = 30
    rng = np.random.default_rng(3)
    Hr = tridiag_spd(n)
    Hrd = Hr.toarray()
    oracle = corr.sparse_hr_oracle(Hr)

    block = corr.empty_block(n)
    assert block.rank == 0 and block.dim == n

    V_new = rng.normal(size=(n, 4))
    C_new = rng.normal(size=(4, 4))
    C_new = 0.5 * (C_new + C_new.T)
    report = corr.merge(block, oracle, V_new, C_new, corr.Provenance.Flip)
    assert (report.requested, report.added) == (4, 4)
    assert corr.validate(block) == []
    assert list(block.tags) == [corr.Provenance.Flip] * 4

    # the invariant and the represented correction, against dense truth
    np.testing.assert_allclose(block.V.T @ Hrd @ block.V, np.eye(4),
                               atol=1e-12)
    expected = Hrd @ V_new @ C_new @ V_new.T @ Hrd
    X = rng.normal(size=(n, 3))
    np.testing.assert_allclose(corr.apply_correction(block, X), expected @ X,
                               atol=1e-11 * np.abs(expected @ X).max())

    # pencil eigenvalues are eig(C), ascending
    np.testing.assert_allclose(corr.pencil_eigenvalues(block),
                               np.sort(np.linalg.eigvalsh(block.C)),
                               atol=1e-13)


def test_mode_block_round_trips_through_numpy(tmp_path):
    # Persistence is the library convention: the struct is plain arrays.
    n = 25
    rng = np.random.default_rng(4)
    oracle = corr.sparse_hr_oracle(tridiag_spd(n))
    block = corr.empty_block(n)
    corr.merge(block, oracle, rng.normal(size=(n, 3)), np.eye(3),
               corr.Provenance.Deflation)

    path = tmp_path / "block.npz"
    np.savez(path, V=block.V, HrV=block.HrV, C=block.C,
             tags=np.array([int(t) for t in block.tags]))

    data = np.load(path)
    loaded = corr.ModeBlock()
    loaded.V = data["V"]
    loaded.HrV = data["HrV"]
    loaded.C = data["C"]
    loaded.tags = [corr.Provenance(int(t)) for t in data["tags"]]
    assert corr.validate(loaded) == []
    assert list(loaded.tags) == list(block.tags)

    X = rng.normal(size=(n, 2))
    np.testing.assert_array_equal(corr.apply_correction(loaded, X),
                                  corr.apply_correction(block, X))


def test_the_full_pipeline_reaches_a_working_woodbury_solve():
    # fit -> weighted assembly -> boundary -> struct -> exact shifted inverse,
    # checked against scipy dense truth at several shifts with one build.
    scipy_sparse = pytest.importorskip("scipy.sparse")
    op = synthetic_operator()
    fit = fit_synthetic(op)
    n = op["count"]
    rng = np.random.default_rng(5)

    B = corr.sparse_op(scipy_sparse.csr_matrix(
        lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Weighted)))
    Hr = tridiag_spd(n)
    A = corr.make_shifted_operator(B, corr.ProbeArchive(),
                                   corr.sparse_hr_oracle(Hr), 1e-4)
    assert corr.validate(A) == []
    assert A.a0 == 1e-4 and A.lambda_floor is None

    # an as-fitted operator is refused at the door
    with pytest.raises(ValueError, match="not symmetric"):
        corr.make_shifted_operator(
            corr.sparse_op(scipy_sparse.csr_matrix(
                lgpsf.assemble_sparse(fit.model, np.inf))),
            corr.ProbeArchive(), corr.sparse_hr_oracle(Hr), 1e-4)

    # fill the block with mixed-sign modes and sweep the shift: one build,
    # every a above the certified floor, no refactorization anywhere
    corr.merge(A.block, A.hr, rng.normal(size=(n, 3)),
               np.diag([-0.2, 0.4, 1.0]), corr.Provenance.PencilCache)
    floor = corr.glr_pd_floor(A)
    Hrd = Hr.toarray()
    Ed = A.block.HrV @ A.block.C @ A.block.HrV.T
    X = rng.normal(size=(n, 2))
    for a in [1.1 * floor, 3.0 * floor, 50.0 * floor]:
        expected = np.linalg.solve(a * Hrd + Ed, X)
        np.testing.assert_allclose(corr.glr_solve(A, X, a), expected,
                                   atol=1e-9 * np.abs(expected).max())

    # below the floor the certificate refuses, with the floor in the message
    with pytest.raises(ValueError):
        corr.glr_solve(A, X, 0.9 * floor)
