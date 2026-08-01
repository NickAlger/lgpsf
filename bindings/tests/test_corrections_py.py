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
    report = corr.merge(block, oracle, V_new, C_new, C_new,
                        corr.Provenance.Flip)
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
    np.testing.assert_allclose(corr.correction_eigenvalues(block),
                               np.sort(np.linalg.eigvalsh(block.C_corr)),
                               atol=1e-13)


def test_mode_block_round_trips_through_numpy(tmp_path):
    # Persistence is the library convention: the struct is plain arrays.
    n = 25
    rng = np.random.default_rng(4)
    oracle = corr.sparse_hr_oracle(tridiag_spd(n))
    block = corr.empty_block(n)
    corr.merge(block, oracle, rng.normal(size=(n, 3)), np.eye(3), np.eye(3),
               corr.Provenance.Deflation)

    path = tmp_path / "block.npz"
    np.savez(path, V=block.V, HrV=block.HrV, C_corr=block.C_corr,
             C_surr=block.C_surr,
             tags=np.array([int(t) for t in block.tags]))

    data = np.load(path)
    loaded = corr.ModeBlock()
    loaded.V = data["V"]
    loaded.HrV = data["HrV"]
    loaded.C_corr = data["C_corr"]
    loaded.C_surr = data["C_surr"]
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
               np.diag([-0.2, 0.4, 1.0]), np.diag([-0.2, 0.4, 1.0]),
               corr.Provenance.PencilCache)
    floor = corr.glr_pd_floor(A)
    Hrd = Hr.toarray()
    Ed = A.block.HrV @ A.block.C_surr @ A.block.HrV.T
    X = rng.normal(size=(n, 2))
    for a in [1.1 * floor, 3.0 * floor, 50.0 * floor]:
        expected = np.linalg.solve(a * Hrd + Ed, X)
        np.testing.assert_allclose(corr.glr_solve(A, X, a), expected,
                                   atol=1e-9 * np.abs(expected).max())

    # below the floor the certificate refuses, with the floor in the message
    with pytest.raises(ValueError):
        corr.glr_solve(A, X, 0.9 * floor)


def test_make_pd_certifies_the_real_fit_against_dense_truth():
    # The whole point of the layer, end to end on a REAL fitted operator:
    # find the fit's negative pencil modes, flip the ones below -gamma*a0,
    # and certify the exact PD floor -- all checked against scipy's dense
    # generalized eigensolver.
    scipy = pytest.importorskip("scipy")
    import scipy.linalg
    import scipy.sparse

    op = synthetic_operator()
    fit = fit_synthetic(op)
    n = op["count"]
    Bs = scipy.sparse.csr_matrix(
        lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Weighted))
    Hr = tridiag_spd(n)
    Bd, Hrd = Bs.toarray(), Hr.toarray()

    pencil = scipy.linalg.eigh(Bd, Hrd, eigvals_only=True)
    assert pencil.min() < 0  # independently-fitted rows do go negative

    # choose a0 so the threshold -gamma*a0 bisects the negative range:
    # everything below half the most negative value gets flipped
    gamma = 0.5
    a0 = -pencil.min() / gamma * 0.5
    threshold = -gamma * a0
    expected_flips = int((pencil < threshold).sum())
    assert expected_flips >= 1

    A = corr.make_shifted_operator(corr.sparse_op(Bs), corr.ProbeArchive(),
                                   corr.sparse_hr_oracle(Hr), a0)
    report = corr.make_pd(A, gamma=gamma, max_iters=4 * n)
    assert report.certified
    assert report.flipped == expected_flips
    assert A.lambda_floor is not None

    # the floor is the leftmost SURVIVING pencil value, exactly
    survivors = np.concatenate([pencil[pencil >= threshold],
                                -report.flipped_values])
    np.testing.assert_allclose(report.lambda_floor, survivors.min(),
                               rtol=1e-6)

    # dense truth for the corrected operator: flipped modes reflected,
    # everything else in place
    P0d = Bd + A.block.HrV @ A.block.C_corr @ A.block.HrV.T
    corrected = scipy.linalg.eigh(P0d, Hrd, eigvals_only=True)
    expected = np.sort(np.where(pencil < threshold, -pencil, pencil))
    np.testing.assert_allclose(corrected, expected,
                               atol=1e-8 * np.abs(pencil).max())

    # and the certified contract holds on both sides of the floor
    def min_eig(a):
        return scipy.linalg.eigh(P0d + a * Hrd, Hrd,
                                 eigvals_only=True).min()
    assert min_eig(2.0 * -report.lambda_floor) > 0
    assert min_eig(0.5 * -report.lambda_floor) < 0


def test_zones_and_both_solve_modes_on_the_real_fit():
    # Certify the fit, then exercise the zone boundaries and both solve
    # paths against scipy dense truth.
    scipy = pytest.importorskip("scipy")
    import scipy.linalg
    import scipy.sparse

    op = synthetic_operator()
    fit = fit_synthetic(op)
    n = op["count"]
    Bs = scipy.sparse.csr_matrix(
        lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Weighted))
    Hr = tridiag_spd(n)
    Bd, Hrd = Bs.toarray(), Hr.toarray()
    pencil = scipy.linalg.eigh(Bd, Hrd, eigvals_only=True)
    a0 = -pencil.min() / 0.5 * 0.5
    A = corr.make_shifted_operator(corr.sparse_op(Bs), corr.ProbeArchive(),
                                   corr.sparse_hr_oracle(Hr), a0)

    # before certification: warned, nothing claimed
    assert corr.classify_shift(A, a0).zone == corr.Zone.Warned

    report = corr.make_pd(A, max_iters=4 * n)
    assert report.certified
    floor = -A.lambda_floor
    assert corr.classify_shift(A, a0).zone == corr.Zone.Guaranteed
    mid = 0.5 * (floor + a0)
    assert corr.classify_shift(A, mid).zone == corr.Zone.Warned
    assert corr.classify_shift(A, mid).analytic_pd
    assert corr.classify_shift(A, 0.5 * floor).zone == corr.Zone.Refused
    with pytest.raises(ValueError):
        corr.solve(A, np.ones((n, 1)), 0.5 * floor)

    # cache the top modes, then solve both ways at a warned shift
    corr.extend_modes(A, 6, max_iters=3 * n)
    rng = np.random.default_rng(6)
    b = rng.normal(size=(n, 1))
    P0d = Bd + A.block.HrV @ A.block.C_corr @ A.block.HrV.T

    glr = corr.solve(A, b, mid)
    Sd = A.block.HrV @ A.block.C_surr @ A.block.HrV.T
    np.testing.assert_allclose(glr.X, np.linalg.solve(mid * Hrd + Sd, b),
                               atol=1e-8 * np.abs(glr.X).max())

    two = corr.solve(A, b, mid, mode=corr.SolveMode.TwoLevel, rtol=1e-10)
    assert two.zone.zone == corr.Zone.Warned
    assert two.iterations > 0
    np.testing.assert_allclose(two.X, np.linalg.solve(P0d + mid * Hrd, b),
                               atol=1e-6 * np.abs(two.X).max())


def test_deflation_closes_the_error_the_fit_left_behind():
    # Pipeline test with EXACT truth: certify the real fit, then declare the
    # true operator to be the certified fit plus a planted low-rank error.
    # Free deflation must recover the plant from archived residuals alone;
    # the value pass must recover it from true applies.
    scipy = pytest.importorskip("scipy")
    import scipy.sparse

    op = synthetic_operator()
    fit = fit_synthetic(op)
    n = op["count"]
    rng = np.random.default_rng(7)

    Bs = scipy.sparse.csr_matrix(
        lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Weighted))
    Hr = tridiag_spd(n)
    Hrd = Hr.toarray()
    A = corr.make_shifted_operator(corr.sparse_op(Bs), corr.ProbeArchive(),
                                   corr.sparse_hr_oracle(Hr), 1e-2)
    assert corr.make_pd(A, max_iters=4 * n).certified

    def corrected(struct):
        return Bs.toarray() + struct.block.HrV @ struct.block.C_corr \
            @ struct.block.HrV.T

    # the plant: rank-3, positive, on Hr-normalized random directions
    U = rng.normal(size=(n, 3))
    U /= np.sqrt(np.sum(U * (Hrd @ U), axis=0))
    delta = (Hrd @ U) @ np.diag([0.3, 0.2, 0.1]) @ (Hrd @ U).T
    Hd = corrected(A) + delta

    Z = rng.normal(size=(n, 12))
    archive = corr.ProbeArchive()
    archive.Z = Z
    archive.Y = Hd @ Z
    A.archive = archive

    def err(struct):
        return np.linalg.norm(Hd - corrected(struct)) / np.linalg.norm(Hd)

    before = err(A)
    assert before > 1e-4

    free = corr.deflate_free(A, rcond=1e-6)
    assert (free.applies, free.kept, free.clamped) == (0, 3, 0)
    assert err(A) < 1e-5 * before  # exact data, exact recovery

    # value pass on a fresh struct over the same plant
    B = corr.make_shifted_operator(corr.sparse_op(Bs), archive,
                                   corr.sparse_hr_oracle(Hr), 1e-2)
    assert corr.make_pd(B, max_iters=4 * n).certified
    Hd_B = corrected(B) + delta
    archive_B = corr.ProbeArchive()
    archive_B.Z = Z
    archive_B.Y = Hd_B @ Z
    B.archive = archive_B
    vp = corr.value_pass(B, corr.dense_op(Hd_B), 3, corr.ValuePassMode.V1)
    assert vp.applies == 3 and vp.kept == 3
    assert B.archive.Q_vp.shape == (n, 3)
    # the remaining error is tiny compared with the plant it removed
    assert np.linalg.norm(Hd_B - corrected(B)) < 1e-5 * np.linalg.norm(delta)


def test_the_whole_struct_round_trips_through_numpy_and_rebuilds(tmp_path):
    # Persistence, the library way: everything but the operator handles is
    # plain arrays. Save, load into a fresh struct with re-supplied
    # handles, verify identical behavior, then rebuild_at on the loaded
    # struct -- the archive is what makes that possible.
    scipy = pytest.importorskip("scipy")
    import scipy.sparse

    op = synthetic_operator()
    fit = fit_synthetic(op)
    n = op["count"]
    rng = np.random.default_rng(8)
    Bs = scipy.sparse.csr_matrix(
        lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Weighted))
    Hr = tridiag_spd(n)

    A = corr.make_shifted_operator(corr.sparse_op(Bs), corr.ProbeArchive(),
                                   corr.sparse_hr_oracle(Hr), 1e-2)
    assert corr.make_pd(A, max_iters=4 * n).certified
    Hd = Bs.toarray() + A.block.HrV @ A.block.C_corr @ A.block.HrV.T
    U = rng.normal(size=(n, 2))
    U /= np.sqrt(np.sum(U * (Hr.toarray() @ U), axis=0))
    Hd = Hd + (Hr.toarray() @ U) @ np.diag([0.3, 0.15]) @ (Hr.toarray() @ U).T
    archive = corr.ProbeArchive()
    archive.Z = rng.normal(size=(n, 8))
    archive.Y = Hd @ archive.Z
    A.archive = archive
    corr.value_pass(A, corr.dense_op(Hd), 2, corr.ValuePassMode.V1)

    path = tmp_path / "struct.npz"
    np.savez(path, V=A.block.V, HrV=A.block.HrV, C_corr=A.block.C_corr,
             C_surr=A.block.C_surr,
             tags=np.array([int(t) for t in A.block.tags]),
             Z=A.archive.Z, Y=A.archive.Y, Q_vp=A.archive.Q_vp,
             HdQ_vp=A.archive.HdQ_vp, a0=A.a0, gamma=A.gamma,
             clamp_floor=A.clamp_floor, lambda_floor=A.lambda_floor,
             C_corr_certified=A.C_corr_certified)

    data = np.load(path)
    loaded_archive = corr.ProbeArchive()
    loaded_archive.Z = data["Z"]
    loaded_archive.Y = data["Y"]
    B = corr.make_shifted_operator(corr.sparse_op(Bs), loaded_archive,
                                   corr.sparse_hr_oracle(Hr),
                                   float(data["a0"]))
    B.archive.Q_vp = data["Q_vp"]
    B.archive.HdQ_vp = data["HdQ_vp"]
    B.block.V = data["V"]
    B.block.HrV = data["HrV"]
    B.block.C_corr = data["C_corr"]
    B.block.C_surr = data["C_surr"]
    B.block.tags = [corr.Provenance(int(t)) for t in data["tags"]]
    B.gamma = float(data["gamma"])
    B.clamp_floor = float(data["clamp_floor"])
    B.lambda_floor = float(data["lambda_floor"])
    B.C_corr_certified = data["C_corr_certified"]
    assert corr.validate(B) == []

    # identical behavior, and the loaded contracts are live
    X = rng.normal(size=(n, 2))
    np.testing.assert_array_equal(corr.apply(B, X, 0.3), corr.apply(A, X, 0.3))
    assert corr.classify_shift(B, 1e-2).zone == corr.Zone.Guaranteed

    # and the loaded struct can re-anchor, using only the archive
    report = corr.rebuild_at(B, 1e-3, max_iters=4 * n)
    assert report.flip.certified
    assert report.refolded and report.value_fold.applies == 0
    assert B.a0 == 1e-3
    assert corr.classify_shift(B, 1e-3).zone == corr.Zone.Guaranteed


def test_the_cholesky_backend_agrees_with_the_iterative_stack():
    scipy = pytest.importorskip("scipy")
    import scipy.sparse

    op = synthetic_operator()
    fit = fit_synthetic(op)
    n = op["count"]
    rng = np.random.default_rng(9)
    Bs = scipy.sparse.csr_matrix(
        lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Weighted))
    Hr = tridiag_spd(n)
    A = corr.make_shifted_operator(corr.sparse_op(Bs), corr.ProbeArchive(),
                                   corr.sparse_hr_oracle(Hr), 1e-2)
    assert corr.make_pd(A, max_iters=4 * n).certified

    backend = corr.make_cholesky_backend(Bs, Hr, A)
    with pytest.raises(ValueError):
        corr.make_cholesky_backend(2.0 * Bs, Hr, A)  # not the operator

    b = rng.normal(size=(n, 1))
    a = 5e-2
    direct = corr.cholesky_solve(backend, A, b, a)
    iterative = corr.solve(A, b, a, mode=corr.SolveMode.TwoLevel,
                           rtol=1e-12).X
    P0 = Bs.toarray() + A.block.HrV @ A.block.C_corr @ A.block.HrV.T
    dense = np.linalg.solve(P0 + a * Hr.toarray(), b)
    np.testing.assert_allclose(direct, dense, atol=1e-9 * np.abs(dense).max())
    np.testing.assert_allclose(iterative, dense,
                               atol=1e-7 * np.abs(dense).max())

    # the raw pencil's exact PD certificate, against the certified floor of
    # the CORRECTED operator: the raw floor can only be worse (more
    # negative), so PD of the raw pencil implies a above the raw floor
    assert not corr.sparse_part_pd(backend, 1e-12)
