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
from operator_fit_frog import (build_problem, fit_at, frog_covariance,
                               frog_row, relative_frobenius)

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


def test_the_kernel_row_matches_a_pointwise_evaluation(problem):
    # Guards the vectorized-over-source rewrite in the example: the operator
    # row is phi(target, .) with the rotation taken at the SOURCE, and getting
    # that backwards would silently build a different operator to fit.
    x = problem["x"]
    target = x[:, 137]
    row = frog_row(target, x)
    for j in (0, 55, 199, 380):
        single = frog_row(target, x[:, j:j + 1])[0]
        assert row[j] == pytest.approx(single)


def test_the_prior_covariance_is_the_kernels_own_shape():
    # Sigma = R^T diag(sd) R must invert the Mahalanobis form the kernel uses,
    # or every window in the fit is the wrong shape.
    from operator_fit_frog import SIGMA0_DIAG, _angle
    x = np.array([0.37, 0.62])
    sigma = frog_covariance(x)
    angle = _angle(x)
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([[cos, -sin], [sin, cos]])
    np.testing.assert_allclose(np.linalg.inv(sigma),
                               R.T @ np.diag(1.0 / SIGMA0_DIAG) @ R, atol=1e-10)
