# SPDX-License-Identifier: MIT
"""End-to-end integration test: fit a known operator from random matvecs.

This drives the WHOLE pipeline -- harmonic table, LG basis, pullback,
whitening, VarPro, the mode ladder, the baseline guard, the dual-tree windows,
sparse assembly -- on a self-contained analytic operator. Nothing private is
involved, which is the point: the library's end-to-end gate must be runnable by
anyone who checks out the repo.

It reuses `examples/operator_fit_frog.py` rather than restating the kernel, so
the example a reader learns from is the same code the test exercises.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "examples"))

import lgpsf
from frog_kernel import build_problem, frog_covariance, frog_row
from operator_fit_frog import fit_at, relative_frobenius

pytest.importorskip("scipy")


@pytest.fixture(scope="module")
def problem():
    return build_problem(grid=20)


def test_the_operator_is_recovered_better_with_more_probes(problem):
    coarse_fit, coarse = fit_at(problem, 10)
    fine_fit, fine = fit_at(problem, 45)

    coarse_error = relative_frobenius(coarse, problem["H"])
    fine_error = relative_frobenius(fine, problem["H"])

    # the headline claim of the whole library, on a problem with a known answer
    assert fine_error < 0.5 * coarse_error, (
        f"error did not fall with the probe budget: "
        f"{coarse_error:.4f} at k=10 vs {fine_error:.4f} at k=45")
    assert fine_error < 0.30

    # and it spent the extra budget on modes, which is the mechanism
    def mean_modes(fit):
        rows = [r for r in range(problem["count"]) if fit.model.has_model(r)]
        return np.mean([len(fit.model.row_modes(r)) for r in rows])

    assert mean_modes(fine_fit) > mean_modes(coarse_fit)


def test_matvec_and_the_assembled_matrix_agree_on_the_real_operator(problem):
    fit, assembled = fit_at(problem, 45)
    probes = np.random.default_rng(3).normal(size=(4, problem["count"]))
    np.testing.assert_allclose(lgpsf.matvec(fit.model, probes),
                               (assembled @ probes.T).T, atol=1e-14)


def test_the_row_envelope_is_the_targets_own_covariance(problem):
    """Pins the anchoring, which is the example's whole setup.

    The kernel is anchored at the TARGET, so a row is one point-spread function
    and `frog_covariance(target)` is exactly its envelope. Anchor it at the
    source instead and a row becomes a transversal of many different local
    shapes -- a perfectly good operator, but not one a per-row smooth expansion
    can represent, and the prior handed to the fitter would be the wrong shape.

    Derived here from `frog_covariance` independently of `frog_row`, so the two
    cannot drift apart silently.
    """
    x = problem["x"]
    target = x[:, 137]
    row = frog_row(target, x)

    sigma = frog_covariance(target)
    d = x - target[:, None]
    maha2 = np.einsum("ik,ij,jk->k", d, np.linalg.inv(sigma), d)
    envelope = np.exp(-0.5 * maha2) / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))

    # row = bump(target) * (1 + A_MOD * modulation) * envelope, and the
    # modulation is a product of a cos and a sin, so the ratio lies in [0, 2].
    from frog_kernel import A_MOD, _bump
    near = envelope > 1e-6 * envelope.max()
    ratio = row[near] / (envelope[near] * _bump(target))
    assert ratio.min() > -1e-9
    assert ratio.max() < 1.0 + A_MOD + 1e-9


def test_a_fitted_row_is_no_rougher_than_the_row_it_fits(problem):
    """The smooth basis, made checkable.

    Note the TRUTH is smooth in both directions -- it is a smooth function of
    both arguments -- so this is not inherited from the operator. Rows are
    smooth because one smooth model produces each; columns carry no such
    guarantee, since a column takes one value from each of many independently
    fitted rows.
    """
    fit, approx = fit_at(problem, 45)
    scale = np.abs(problem["H"]).max()
    truth = np.abs(np.diff(problem["H"], n=2, axis=1)).mean() / scale
    fitted = np.abs(np.diff(approx, n=2, axis=1)).mean() / scale
    assert fitted <= truth * 1.05


def test_the_prior_covariance_is_the_kernels_own_shape():
    # Sigma = R^T diag(sd) R must invert the Mahalanobis form the kernel uses,
    # or every window in the fit is the wrong shape.
    from frog_kernel import SIGMA0_DIAG, _angle
    x = np.array([0.37, 0.62])
    sigma = frog_covariance(x)
    angle = _angle(x)
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([[cos, -sin], [sin, cos]])
    np.testing.assert_allclose(np.linalg.inv(sigma),
                               R.T @ np.diag(1.0 / SIGMA0_DIAG) @ R, atol=1e-10)
