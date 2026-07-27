# SPDX-License-Identifier: MIT
"""Tests for the lgpsf Python bindings.

Array convention at the Python boundary: point batches are (N, K) --
coordinates DOWN, points ACROSS. That is the layout which maps onto the C++
core with no copy, and the transpose of ellipsoid_tree's Python one; see the
module docstring in `bindings/lgpsf_bindings.cpp` for why.

These tests check the BINDINGS -- marshalling, layout, error mapping -- not the
mathematics, which the C++ suite owns. Nothing here compares against
`prototype/`: the C++ is the ground truth and that directory is history.
"""

import pathlib
import re

import numpy as np
import pytest

import lgpsf


REPO = pathlib.Path(__file__).resolve().parents[2]


def test_version_matches_the_umbrella_header():
    # The header is the single source of truth; CMake parses it for the project
    # version and pyproject repeats it. A drift here means a wheel that lies
    # about which core it contains.
    text = (REPO / "include" / "lgpsf" / "lgpsf.hpp").read_text()
    parts = [
        re.search(rf"#define LGPSF_VERSION_{field} (\d+)", text).group(1)
        for field in ("MAJOR", "MINOR", "PATCH")
    ]
    assert lgpsf.__version__ == ".".join(parts)

    pyproject = (REPO / "pyproject.toml").read_text()
    assert re.search(r'^version = "(.*)"', pyproject, re.M).group(1) == lgpsf.__version__


# --------------------------------------------------------------------------
# The layout. This is the one bug this boundary invites and the reason the
# probe exists: a transposed read is SILENT -- it yields a plausible array of
# the wrong thing. Earlier this week the same mistake, made in a raw-binary
# bridge, showed up only as a cross-validation score of 0.97.
# --------------------------------------------------------------------------

def test_layout_probe_reads_coordinates_down_and_points_across():
    # Four points in 3-D, chosen so every coordinate and every point is
    # distinguishable and no symmetry could hide a transpose.
    x = np.array([[0.0, 1.0, 2.0, 3.0],      # coordinate 0 of each point
                  [10.0, 20.0, 30.0, 40.0],  # coordinate 1
                  [100.0, 200.0, 300.0, 400.0]])  # coordinate 2
    probe = lgpsf._layout_probe(x)

    assert probe["num_points"] == 4
    assert probe["dim"] == 3
    # lengths differ (3 vs 4), so a transposed read cannot even produce these
    assert probe["centroid"].shape == (3,)
    assert probe["norms"].shape == (4,)

    np.testing.assert_allclose(probe["first_point"], x[:, 0])
    np.testing.assert_allclose(probe["centroid"], x.mean(axis=1))
    np.testing.assert_allclose(probe["norms"], np.linalg.norm(x, axis=0))


def test_layout_probe_accepts_awkward_inputs_by_converting():
    # A non-contiguous slice and an integer array both have to work; the
    # zero-copy path is an optimization for clean input, not a precondition.
    base = np.arange(60.0).reshape(3, 20)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(lgpsf._layout_probe(view)["centroid"],
                               view.mean(axis=1))

    integers = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(lgpsf._layout_probe(integers)["centroid"],
                               integers.mean(axis=1))


def test_layout_probe_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"batch \(point\) axis LAST"):
        lgpsf._layout_probe(np.zeros(5))


def test_a_clean_array_is_not_copied():
    # The zero-copy claim, checked rather than asserted. Comparing values before
    # and after a mutation would NOT show this -- a copying implementation
    # re-reads the array on the next call too and gives the same answer. The
    # only honest check is the address C++ read from.
    x = np.ascontiguousarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert x.flags["C_CONTIGUOUS"] and x.dtype == np.float64
    assert lgpsf._layout_probe(x)["data_ptr"] == x.__array_interface__["data"][0]


def test_awkward_input_is_copied_and_says_so():
    # The flip side, so the boundary between the two paths is documented rather
    # than incidental: input that cannot be mapped is converted, which means a
    # different buffer.
    base = np.arange(60.0).reshape(3, 20)
    view = base[:, ::2]
    assert lgpsf._layout_probe(view)["data_ptr"] != view.__array_interface__["data"][0]

    integers = np.array([[1, 2, 3], [4, 5, 6]])
    assert (lgpsf._layout_probe(integers)["data_ptr"]
            != integers.__array_interface__["data"][0])


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def test_mode_is_a_value_type_that_behaves_like_its_tuple():
    mode = lgpsf.Mode(1, 2, -1)
    assert (mode.p, mode.ell, mode.m) == (1, 2, -1)
    assert tuple(mode) == (1, 2, -1)
    assert mode == lgpsf.Mode(1, 2, -1)
    assert mode != lgpsf.Mode(1, 2, 1)
    assert len({lgpsf.Mode(0, 0, 0), lgpsf.Mode(0, 0, 0)}) == 1
    assert "p=1" in repr(mode)


def test_modes_up_to_level_counts_and_caps():
    # 2-D shells: level n contributes n + 1 modes, so levels 0..L give
    # (L + 1)(L + 2) / 2.
    for level in range(5):
        expected = (level + 1) * (level + 2) // 2
        assert len(lgpsf.modes_up_to_level(2, level)) == expected

    # the wedge: capping ell keeps the level-ordered prefix the operator layer
    # defaults to
    capped = lgpsf.modes_up_to_level(2, 6, ell_max=2)
    assert all(mode.ell <= 2 for mode in capped)
    assert len(capped) < len(lgpsf.modes_up_to_level(2, 6))
    assert all(mode.p >= 0 for mode in capped)


def test_modes_up_to_level_raises_past_the_generated_table():
    assert lgpsf.max_oscillator_level() >= 10
    assert lgpsf.max_dimension() >= 4
    with pytest.raises(ValueError):
        lgpsf.modes_up_to_level(2, lgpsf.max_oscillator_level() + 1)


# --------------------------------------------------------------------------
# Tier B: the primitives, for day-to-day work from Python.
#
# These check the BINDING -- shapes, the batch-last convention, error mapping,
# and agreement between paths that the C++ already guarantees agree. The
# mathematics itself is the C++ suite's job; re-asserting it here would only
# test doctest twice.
# --------------------------------------------------------------------------

def unit_points(dim, count, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(dim, count))  # (N, K): coordinates down


def test_batched_results_put_the_batch_axis_last():
    # The convention that makes every array in and out the same bytes as the
    # C++ side. Shapes alone pin it, provided num_modes != K and N != K.
    u = unit_points(3, 7)
    modes = lgpsf.modes_up_to_level(3, 1)   # 1 + 3 = 4 modes in 3-D
    assert len(modes) == 4

    values = lgpsf.eval_lg_basis(modes, u)
    assert values.shape == (4, 7)

    gradients = lgpsf.grad_lg_basis(modes, u)
    assert gradients.shape == (4, 3, 7)

    w = np.ones((4, 7))
    assert lgpsf.vjp_lg_basis(modes, u, w).shape == (3, 7)

    assert lgpsf.eval_lg_nd(0, 0, 0, u).shape == (7,)
    assert lgpsf.grad_eval_lg_nd(0, 0, 0, u).shape == (3, 7)


def test_the_mode_set_path_and_the_one_at_a_time_path_agree():
    # Internal equivalence: the batched path is the production one and the
    # single-mode path is the readable reference, so a binding that got the
    # mode ordering or the stacking wrong shows up here.
    u = unit_points(2, 11, seed=1)
    modes = lgpsf.modes_up_to_level(2, 2)
    values = lgpsf.eval_lg_basis(modes, u)
    gradients = lgpsf.grad_lg_basis(modes, u)
    for i, mode in enumerate(modes):
        np.testing.assert_allclose(
            values[i], lgpsf.eval_lg_nd(mode.p, mode.ell, mode.m, u))
        np.testing.assert_allclose(
            gradients[i], lgpsf.grad_eval_lg_nd(mode.p, mode.ell, mode.m, u))


def test_vjp_contracts_the_gradients_it_never_builds():
    # vjp regroups per shell rather than materializing (num_modes, N, K); the
    # binding has to hand it a cotangent in the same layout the values use.
    u = unit_points(2, 9, seed=2)
    modes = lgpsf.modes_up_to_level(2, 2)
    rng = np.random.default_rng(3)
    w = rng.normal(size=(len(modes), 9))
    np.testing.assert_allclose(
        lgpsf.vjp_lg_basis(modes, u, w),
        np.einsum("ik,ink->nk", w, lgpsf.grad_lg_basis(modes, u)),
        rtol=1e-12, atol=1e-12)


def test_genlaguerre_matches_its_low_order_closed_forms():
    x = np.linspace(0.0, 3.0, 9)
    np.testing.assert_allclose(lgpsf.genlaguerre(0, 1.5, x), np.ones_like(x))
    np.testing.assert_allclose(lgpsf.genlaguerre(1, 1.5, x), 1.5 + 1.0 - x)
    a = 0.75
    np.testing.assert_allclose(
        lgpsf.genlaguerre(2, a, x),
        x**2 / 2 - (a + 2) * x + (a + 1) * (a + 2) / 2)


def test_harmonic_terms_are_a_term_table_not_a_point_batch():
    poly = lgpsf.harmonic_terms(3, 2, 0)
    assert poly["dim"] == 3 and poly["degree"] == 2
    num_terms = poly["coefficients"].shape[0]
    # one ROW per term here -- the batch-last rule is for point batches
    assert poly["exponents"].shape == (num_terms, 3)
    # a degree-2 harmonic's terms are all degree 2
    assert np.all(poly["exponents"].sum(axis=1) == 2)


def test_harmonic_evaluation_agrees_with_its_own_term_list():
    # Reconstructing Y from the exported terms is the check that the two
    # exports describe the same polynomial.
    u = unit_points(3, 6, seed=4)
    for degree in range(3):
        for m in range(lgpsf.num_harmonics(3, degree)):
            poly = lgpsf.harmonic_terms(3, degree, m)
            expected = sum(
                c * np.prod(u ** e[:, None], axis=0)
                for c, e in zip(poly["coefficients"], poly["exponents"]))
            np.testing.assert_allclose(lgpsf.eval_harmonic(degree, m, u),
                                       expected, rtol=1e-12, atol=1e-12)


def test_grad_harmonic_returns_value_and_gradient_together():
    u = unit_points(2, 5, seed=5)
    value, gradient = lgpsf.grad_harmonic(2, 0, u)
    assert value.shape == (5,) and gradient.shape == (2, 5)
    np.testing.assert_allclose(value, lgpsf.eval_harmonic(2, 0, u))


# ---- the pullback and its encodings ---------------------------------------

def test_theta_sizes_and_the_two_encodings_round_trip():
    for dim in (1, 2, 3):
        assert lgpsf.theta_size(dim) == dim * (dim + 3) // 2
        assert lgpsf.dim_from_theta_size(lgpsf.theta_size(dim)) == dim
        assert (lgpsf.theta_hat_size(dim, lgpsf.MuMode.Fitted)
                == lgpsf.theta_size(dim))
        assert (lgpsf.theta_hat_size(dim, lgpsf.MuMode.Pinned)
                == lgpsf.theta_size(dim) - dim)

    dim = 2
    mu0 = np.array([0.5, -1.25])
    theta = np.array([1.0, 2.0, np.log(0.3), np.log(0.7), 0.15])
    for mode in (lgpsf.MuMode.Pinned, lgpsf.MuMode.Fitted):
        theta_hat = lgpsf.to_theta_hat(theta, mu0, mode)
        assert theta_hat.shape == (lgpsf.theta_hat_size(dim, mode),)
        back = lgpsf.to_theta(theta_hat, mu0, mode)
        if mode is lgpsf.MuMode.Fitted:
            np.testing.assert_allclose(back, theta)
        else:
            # pinning discards the center by construction; the L block survives
            np.testing.assert_allclose(back[dim:], theta[dim:])
            np.testing.assert_allclose(back[:dim], mu0)


def test_unpack_theta_needs_nothing_else():
    # Self-decoding is what makes a fitted operator readable without its fit.
    theta = np.array([1.0, 2.0, np.log(0.3), np.log(0.7), 0.15])
    frame = lgpsf.unpack_theta(theta)
    assert frame.dim == 2
    np.testing.assert_allclose(frame.mu, [1.0, 2.0])
    np.testing.assert_allclose(np.diag(frame.L), [0.3, 0.7])
    assert frame.L[0, 1] == 0.0            # lower triangular, not upper --
    np.testing.assert_allclose(frame.L[1, 0], 0.15)   # a silent transpose would flip these
    np.testing.assert_allclose(frame.L @ frame.L_inv, np.eye(2), atol=1e-14)


def test_pullback_maps_the_ellipsoid_to_the_unit_sphere():
    frame = lgpsf.make_frame(np.array([1.0, -2.0]),
                             np.array([[0.5, 0.0], [0.2, 0.8]]))
    # points at Mahalanobis radius 1 must land on the unit circle
    angles = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    unit = np.vstack([np.cos(angles), np.sin(angles)])
    x = frame.mu[:, None] + frame.L @ unit
    u = lgpsf.pullback(frame, x)
    assert u.shape == (2, 12)
    np.testing.assert_allclose(np.linalg.norm(u, axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(u, unit, atol=1e-12)


def test_release_and_freeze_mu():
    theta = np.array([1.0, 2.0, np.log(0.3), np.log(0.7), 0.15])
    block, center = lgpsf.freeze_mu(theta)
    np.testing.assert_allclose(center, [1.0, 2.0])
    np.testing.assert_allclose(block, theta[2:])
    np.testing.assert_allclose(lgpsf.release_mu(block, 2)[2:], block)
    np.testing.assert_allclose(lgpsf.release_mu(block, 2)[:2], 0.0)


def test_infeasible_frame_is_an_error_not_a_nan():
    with pytest.raises(Exception):
        lgpsf.make_frame(np.zeros(2), np.array([[0.0, 0.0], [0.0, 1.0]]))


# ---- whitening -------------------------------------------------------------

def test_whitening_applies_opposite_powers_of_the_column_mass():
    # The asymmetry is the whole mechanism: the smooth basis carries M2^(1/2)
    # because it is a quadrature object, the extra basis M2^(-1/2) because it
    # is a discrete correction.
    m2 = np.array([0.5, 2.0, 4.0])
    target_mass = 9.0
    np.testing.assert_allclose(lgpsf.whitening_scale(target_mass, m2),
                               3.0 * np.sqrt(m2))

    z = np.arange(6.0).reshape(2, 3)          # (num_probes, K)
    np.testing.assert_allclose(lgpsf.whiten_probes(z, m2), z * np.sqrt(m2))

    E = np.ones((1, 3))                        # (num_extra, K)
    np.testing.assert_allclose(lgpsf.whiten_extra(E, target_mass, m2),
                               3.0 / np.sqrt(m2)[None, :])

    y = np.array([1.0, 2.0])
    np.testing.assert_allclose(lgpsf.whiten_data(y, target_mass), y / 3.0)


def test_whitening_rejects_a_nonpositive_mass():
    # All three entry points, because whiten_probes used not to check and
    # silently returned NaN -- which survives to a cost and reads there as a
    # bad fit rather than a bad input. Found by writing this test.
    with pytest.raises(ValueError):
        lgpsf.whiten_data(np.ones(2), 0.0)
    with pytest.raises(ValueError):
        lgpsf.whiten_probes(np.ones((1, 2)), np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        lgpsf.whiten_extra(np.ones((1, 2)), 1.0, np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        lgpsf.whitening_scale(-1.0, np.ones(2))
