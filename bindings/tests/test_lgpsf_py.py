# SPDX-License-Identifier: MIT
"""Tests for the lgpsf Python bindings.

Array convention at the Python boundary: point batches are (N, K) --
coordinates DOWN, points ACROSS. That is the layout which maps onto the C++
core with no copy, and the transpose of ellipsoid_tree's Python one; see the
module docstring in `bindings/lgpsf_bindings.cpp` for why.

These tests check the BINDINGS -- marshalling, layout, error mapping -- not the
mathematics, which the C++ suite owns. Nothing here compares against
the archived Python prototype: the C++ is the ground truth and that directory is history.
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


# --------------------------------------------------------------------------
# The row layer: fit one target known only through probe inner products.
# --------------------------------------------------------------------------

def synthetic_target(per_side=11, num_probes=40, level=2, seed=7):
    """A known (theta, c, s) built through the whitened identity.

    Mirrors the C++ suite's construction, which is the point: y is assembled
    from the same whitening the fitter inverts, so an exact recovery is the
    expected outcome and any layout slip at the boundary breaks it.
    """
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, per_side)
    grid = np.meshgrid(axis, axis, indexing="ij")
    x = np.vstack([grid[0].ravel(), grid[1].ravel()])   # (2, K)
    count = x.shape[1]

    spike_index = count // 2
    mu0 = x[:, spike_index].copy()
    mass = 0.8
    m2_diag = rng.uniform(0.4, 1.1, size=count)
    modes = lgpsf.modes_up_to_level(2, level)

    theta_hat_true = np.array([np.log(0.30), np.log(0.22), 0.05])
    basis = lgpsf.WhitenedBasis(x, mass, m2_diag, modes, mu0, lgpsf.MuMode.Pinned)
    phi_hat = basis.values(theta_hat_true)              # (num_modes, K)

    c_true = rng.normal(size=len(modes))
    s_true = rng.normal(size=1)

    z = rng.normal(size=(num_probes, count))            # (num_probes, K)
    extra = np.zeros((1, count))
    extra[0, spike_index] = 1.0
    e_hat = lgpsf.whiten_extra(extra, mass, m2_diag)
    z_hat = lgpsf.whiten_probes(z, m2_diag)
    y_hat = z_hat @ (c_true @ phi_hat) + z_hat @ (s_true @ e_hat)
    y = np.sqrt(mass) * y_hat

    return dict(x=x, m2_diag=m2_diag, z=z, y=y, mu0=mu0, mass=mass,
                spike_index=spike_index, modes=modes, c_true=c_true,
                s_true=s_true, theta_hat_true=theta_hat_true, e_hat=e_hat,
                z_hat=z_hat, y_hat=y_hat, basis=basis)


def fixed_config(target, **kwargs):
    config = lgpsf.ProbeFitConfig()
    config.mode_policy = lgpsf.FixedSet(target["modes"], "truth")
    config.mu = lgpsf.MuPolicy.Pinned
    config.target_score = None          # walk the whole stream, no early exit
    config.num_rungs = 3
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def test_fit_from_probes_recovers_a_known_target():
    target = synthetic_target()
    result = lgpsf.fit_from_probes(
        target["x"], target["m2_diag"], target["z"], target["y"], target["mu0"],
        spike_index=target["spike_index"], config=fixed_config(target),
        target_mass=target["mass"])

    expected = lgpsf.to_theta(target["theta_hat_true"], target["mu0"],
                              lgpsf.MuMode.Pinned)
    np.testing.assert_allclose(result.model.theta, expected, atol=1e-6)
    np.testing.assert_allclose(result.model.c, target["c_true"], atol=1e-6)
    np.testing.assert_allclose(result.model.s, target["s_true"], atol=1e-6)
    assert result.score < 1e-6
    assert list(result.model.modes) == list(target["modes"])
    assert not result.released                      # mu was pinned


def test_the_fitted_model_evaluates_to_the_target_it_was_built_from():
    # End to end through the OTHER direction: the model must reproduce the
    # smooth field, not merely the parameters that generated it.
    target = synthetic_target()
    result = lgpsf.fit_from_probes(
        target["x"], target["m2_diag"], target["z"], target["y"], target["mu0"],
        spike_index=target["spike_index"], config=fixed_config(target),
        target_mass=target["mass"])

    truth = target["c_true"] @ lgpsf.eval_lg_basis(
        target["modes"],
        lgpsf.pullback(lgpsf.unpack_theta(
            lgpsf.to_theta(target["theta_hat_true"], target["mu0"],
                           lgpsf.MuMode.Pinned)), target["x"]))
    got = lgpsf.eval_expansion(result.model, target["x"])
    assert got.shape == (target["x"].shape[1],)
    np.testing.assert_allclose(got, truth, atol=1e-6)


def test_the_audit_trail_comes_back():
    target = synthetic_target()
    result = lgpsf.fit_from_probes(
        target["x"], target["m2_diag"], target["z"], target["y"], target["mu0"],
        spike_index=target["spike_index"], config=fixed_config(target),
        target_mass=target["mass"])

    assert len(result.candidates) > 1
    assert 0 <= result.winner < len(result.candidates)
    winner = result.candidates[result.winner]
    assert winner.score == result.score
    assert winner.admissible and winner.success
    assert winner.num_modes == len(target["modes"])
    assert winner.axes.shape == (2,)
    assert isinstance(winner.label, str) and winner.label
    assert result.stop_reason in (lgpsf.StopReason.Target,
                                  lgpsf.StopReason.ModePatience,
                                  lgpsf.StopReason.Exhausted)
    # the in-sample cost is a diagnostic and must never be the selector
    assert all(cand.score >= result.score for cand in result.candidates
               if cand.admissible)


def test_every_built_in_mode_policy_drives_a_fit():
    target = synthetic_target()
    for policy in (lgpsf.FixedSet(target["modes"]),
                   lgpsf.ShellLadder([0, 1, 2]),
                   lgpsf.ExplicitLadder([target["modes"][:1], target["modes"]]),
                   lgpsf.WedgeLadder(4, 2),
                   lgpsf.RadialFirstLadder(4, 2)):
        config = fixed_config(target)
        config.mode_policy = policy
        result = lgpsf.fit_from_probes(
            target["x"], target["m2_diag"], target["z"], target["y"],
            target["mu0"], spike_index=target["spike_index"], config=config,
            target_mass=target["mass"])
        assert np.isfinite(result.score)
        assert result.model.num_modes > 0


def test_a_missing_mode_policy_is_a_loud_error():
    target = synthetic_target()
    config = lgpsf.ProbeFitConfig()          # mode_policy left unset
    with pytest.raises(ValueError, match="mode_policy"):
        lgpsf.fit_from_probes(target["x"], target["m2_diag"], target["z"],
                              target["y"], target["mu0"], config=config)


def test_linear_cv_score_needs_no_nonlinear_fit():
    # The property that makes an a-priori quality map affordable: score any
    # model at given parameters, zero fits.
    target = synthetic_target()
    split = lgpsf.kfold_split(target["z"].shape[0], 5)
    at_truth = lgpsf.linear_cv_score(
        target["z_hat"], target["y_hat"], target["basis"],
        target["theta_hat_true"], target["e_hat"], split)
    wrong = lgpsf.linear_cv_score(
        target["z_hat"], target["y_hat"], target["basis"],
        target["theta_hat_true"] + np.array([1.5, -1.2, 0.0]),
        target["e_hat"], split)
    assert at_truth < 1e-8
    assert wrong > 100 * max(at_truth, 1e-12)


def test_kfold_split_partitions_the_probes():
    for folds in (2, 5):
        split = lgpsf.kfold_split(23, folds)
        assert len(split) == folds
        seen = np.concatenate([np.asarray(f.validation) for f in split])
        np.testing.assert_array_equal(np.sort(seen), np.arange(23))
        for fold in split:
            assert len(fold.train) + len(fold.validation) == 23
            assert not set(np.asarray(fold.train)) & set(np.asarray(fold.validation))

    # no seed means no generator at all, so it is reproducible with nothing
    # to remember; a seed permutes
    a = lgpsf.kfold_split(20, 4)
    b = lgpsf.kfold_split(20, 4)
    np.testing.assert_array_equal(np.asarray(a[0].validation),
                                  np.asarray(b[0].validation))
    seeded = lgpsf.kfold_split(20, 4, seed=12345)
    assert len(seeded) == 4


def test_fit_varpro_directly():
    target = synthetic_target()
    result = lgpsf.fit_varpro(target["z_hat"], target["y_hat"], target["basis"],
                              target["theta_hat_true"] + 0.15,
                              e_hat=target["e_hat"])
    assert result.success
    np.testing.assert_allclose(result.theta_hat, target["theta_hat_true"],
                               atol=1e-6)
    np.testing.assert_allclose(result.c, target["c_true"], atol=1e-6)
    assert result.cost < 1e-12
    assert result.num_iterations > 0


def test_lg_expansion_is_self_decoding_and_checkable():
    theta = np.array([0.5, -0.25, np.log(0.4), np.log(0.6), 0.1])
    modes = lgpsf.modes_up_to_level(2, 1)
    expansion = lgpsf.LGExpansion(theta, modes, np.ones(len(modes)))
    assert expansion.dim == 2 and expansion.num_modes == len(modes)
    np.testing.assert_allclose(expansion.frame().mu, [0.5, -0.25])
    assert lgpsf.validate_expansion(expansion) == []

    mismatched = lgpsf.LGExpansion(theta, modes, np.ones(len(modes) + 1))
    assert lgpsf.validate_expansion(mismatched)


def test_infeasible_parameters_is_its_own_exception():
    assert issubclass(lgpsf.InfeasibleParameters, RuntimeError)


# --------------------------------------------------------------------------
# The operator layer.
#
# Two conventions meet here. A POINT BATCH is (N, K) in both directions; a
# PER-ROW RECORD is row-first, (R, ...), because `fit.mu[rho]` is how anyone
# reads one. R != N in any real problem, so mixing them is a shape error.
# --------------------------------------------------------------------------

def synthetic_operator(per_side=9, num_probes=30, fitted_rows=4, prior_scale=2.0,
                       seed=11):
    """A synthetic operator with a known model on a few interior rows.

    Every other row's response is identically ZERO, which makes this the
    dead-row situation in miniature as well.
    """
    rng = np.random.default_rng(seed)
    axis = np.linspace(-1.0, 1.0, per_side)
    grid = np.meshgrid(axis, axis, indexing="ij")
    x_cols = np.vstack([grid[0].ravel(), grid[1].ravel()])     # (2, K)
    count = x_cols.shape[1]

    m1 = rng.uniform(0.7, 1.5, size=count)
    m2 = rng.uniform(0.4, 1.1, size=count)          # M1 != M2 throughout
    modes = lgpsf.modes_up_to_level(2, 1)
    V = rng.normal(size=(num_probes, count))        # (num_probes, K)
    HV = np.zeros((num_probes, count))              # (num_probes, R)

    stride = count // (fitted_rows + 1)
    chosen = [stride * (i + 1) for i in range(fitted_rows)]
    theta_hat = np.array([np.log(0.26 * 1.6), np.log(0.26), 0.03])
    sigma = np.tile(np.eye(2), (count, 1, 1))
    truth = {}
    for rho in chosen:
        center = x_cols[:, rho].copy()
        frame = lgpsf.unpack_theta(lgpsf.to_theta(theta_hat, center,
                                                  lgpsf.MuMode.Pinned))
        # the caller's prior is deliberately wrong by a scale factor, so the
        # baseline is beatable and the search has something to do
        sigma[rho] = prior_scale**2 * frame.L @ frame.L.T

        basis = lgpsf.WhitenedBasis(x_cols, m1[rho], m2, modes, center,
                                    lgpsf.MuMode.Pinned)
        c = rng.normal(size=len(modes))
        s = 0.4
        h_row = np.sqrt(m1[rho]) * np.sqrt(m2) * (c @ basis.values(theta_hat))
        h_row[rho] += m1[rho] * s
        HV[:, rho] = V @ h_row
        truth[rho] = dict(c=c, s=s, theta=lgpsf.to_theta(theta_hat, center,
                                                         lgpsf.MuMode.Pinned))

    return dict(x_cols=x_cols, m1=m1, m2=m2, V=V, HV=HV, sigma=sigma,
                modes=modes, chosen=chosen, truth=truth, count=count)


def operator_config(op, **kwargs):
    config = lgpsf.OperatorFitConfig()
    config.tau_window = 10.0
    config.spike = True
    config.row.mode_policy = lgpsf.FixedSet(op["modes"], "truth")
    config.row.target_score = None
    config.row.num_rungs = 3
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def fit_synthetic(op, **kwargs):
    rows = kwargs.pop("rows", np.array(op["chosen"]))
    return lgpsf.fit_operator(op["x_cols"], op["m1"], op["m2"], op["V"],
                              op["HV"], op["sigma"],
                              config=operator_config(op, **kwargs), rows=rows)


def test_fit_operator_recovers_a_synthetic_operator_row_by_row():
    op = synthetic_operator()
    fit = fit_synthetic(op)

    for rho, truth in op["truth"].items():
        assert fit.model.has_model(rho)
        np.testing.assert_allclose(fit.model.theta[rho], truth["theta"], atol=1e-6)
        np.testing.assert_allclose(fit.model.c[rho, :len(truth["c"])],
                                   truth["c"], atol=1e-6)
        np.testing.assert_allclose(fit.model.s[rho], truth["s"], atol=1e-6)
        assert fit.diagnostics.score[rho] < 1e-6


def test_point_batches_round_trip_but_per_row_records_are_row_first():
    op = synthetic_operator()
    fit = fit_synthetic(op)
    dim, count = op["x_cols"].shape

    # what went in comes back out, same shape
    np.testing.assert_allclose(fit.model.x_cols, op["x_cols"])
    assert fit.model.x_rows is None                 # square dof context

    # per-row records lead with the row
    assert fit.model.num_rows == count
    assert fit.model.mu.shape == (count, dim)
    assert fit.model.L.shape == (count, dim, dim)
    assert fit.model.theta.shape == (count, lgpsf.theta_size(dim))
    assert fit.model.window_center.shape == (count, dim)
    assert fit.model.window_covariance.shape == (count, dim, dim)
    assert fit.diagnostics.score.shape == (count,)

    # L really is the lower-triangular factor, not its transpose
    for rho in op["chosen"]:
        L = fit.model.L[rho]
        assert L[0, 1] == 0.0
        np.testing.assert_allclose(L @ L.T,
                                   lgpsf.unpack_theta(fit.model.theta[rho]).L
                                   @ lgpsf.unpack_theta(fit.model.theta[rho]).L.T,
                                   atol=1e-12)


def test_rows_accepts_a_mask_or_indices_and_they_agree():
    op = synthetic_operator()
    mask = np.zeros(op["count"], dtype=bool)
    mask[op["chosen"]] = True
    by_mask = fit_synthetic(op, rows=mask)
    by_index = fit_synthetic(op, rows=np.array(op["chosen"]))
    np.testing.assert_array_equal(by_mask.diagnostics.status,
                                  by_index.diagnostics.status)
    np.testing.assert_allclose(by_mask.model.c, by_index.model.c)


def test_status_is_maskable_and_names_its_enum():
    op = synthetic_operator()
    fit = fit_synthetic(op)
    status = fit.diagnostics.status
    assert status.dtype == np.int8
    fitted = status == int(lgpsf.RowStatus.Fit)
    gated = status == int(lgpsf.RowStatus.GatedOut)
    assert fitted.sum() + gated.sum() == op["count"]
    assert set(np.flatnonzero(fitted)) == set(op["chosen"])
    assert fit.diagnostics.failures == {}
    assert fit.diagnostics.released.dtype == np.bool_


def test_dead_rows_need_no_gate():
    # Every ungated row of this problem has HV == 0, so dropping the gate makes
    # it the PIG situation in miniature: a dead row must cost one candidate and
    # ship zeros, and must not disturb any live row.
    op = synthetic_operator()
    gated = fit_synthetic(op)
    ungated = fit_synthetic(op, rows=None)

    assert ungated.diagnostics.failures == {}
    for rho in op["chosen"]:
        np.testing.assert_array_equal(ungated.model.c[rho], gated.model.c[rho])
        assert ungated.model.s[rho] == gated.model.s[rho]
        assert ungated.diagnostics.score[rho] == gated.diagnostics.score[rho]

    dead = [r for r in range(op["count"]) if r not in op["chosen"]]
    assert np.all(ungated.diagnostics.score[dead] == 0.0)
    assert np.all(ungated.model.c[dead] == 0.0)
    assert np.all(ungated.model.s[dead] == 0.0)

    probes = np.random.default_rng(0).normal(size=(3, op["count"]))
    applied = lgpsf.matvec(ungated.model, probes)
    assert np.all(applied[:, dead] == 0.0)


def test_results_are_identical_across_thread_counts():
    op = synthetic_operator()
    one = fit_synthetic(op, num_threads=1)
    four = fit_synthetic(op, num_threads=4)
    for name in ("theta", "mu", "c", "s"):
        a, b = getattr(one.model, name), getattr(four.model, name)
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.array_equal(np.where(both_nan, 0.0, a),
                              np.where(both_nan, 0.0, b))
    np.testing.assert_array_equal(one.diagnostics.status, four.diagnostics.status)


def test_matvec_agrees_with_the_assembled_sparse_operator():
    pytest.importorskip("scipy")
    op = synthetic_operator()
    fit = fit_synthetic(op)
    probes = np.random.default_rng(1).normal(size=(4, op["count"]))

    A = lgpsf.assemble_sparse(fit.model, np.inf)
    np.testing.assert_allclose(lgpsf.matvec(fit.model, probes),
                               (A @ probes.T).T, atol=1e-12)

    As = lgpsf.assemble_sparse(fit.model, np.inf, lgpsf.Symmetrize.Average)
    dense, dense_sym = np.asarray(A.todense()), np.asarray(As.todense())
    np.testing.assert_allclose(dense_sym, 0.5 * (dense + dense.T), atol=1e-14)


def test_the_deployed_operator_is_window_restricted():
    op = synthetic_operator()
    fit = fit_synthetic(op)
    for rho in op["chosen"]:
        window = set(fit.model.row_window(rho))
        outside = [j for j in range(op["count"]) if j not in window]
        if not outside:
            continue
        values = lgpsf.eval_entries(fit.model, [rho] * len(outside), outside)
        np.testing.assert_array_equal(values, 0.0)
        # ...and the opt-out really does extend past it
        extended = lgpsf.eval_kernel_unrestricted(
            fit.model, [rho], op["x_cols"][:, outside])
        assert np.any(extended != 0.0)


def test_qc_map_spike_measure_and_the_ellipsoid_field():
    op = synthetic_operator()
    fit = fit_synthetic(op)

    qc = lgpsf.qc_map(fit.model, op["V"], op["HV"])
    assert qc.shape == (op["count"],)
    assert np.all(qc[op["chosen"]] < 1e-6)          # exact fit, exact recovery

    np.testing.assert_allclose(lgpsf.spike_measure(fit.model),
                               op["m1"] * fit.model.s, equal_nan=True)

    mu, sigma = lgpsf.ellipsoid_field(fit.model)
    assert mu.shape == (op["count"], 2) and sigma.shape == (op["count"], 2, 2)
    for rho in op["chosen"]:
        np.testing.assert_allclose(sigma[rho], sigma[rho].T, atol=1e-14)


def test_build_operator_round_trips_through_row_expansion():
    # The physics-based construction path: an operator can be built without the
    # fitter ever being involved, and a fitted one can be taken apart.
    op = synthetic_operator()
    fit = fit_synthetic(op)

    rows = []
    for rho in range(op["count"]):
        if not fit.model.has_model(rho):
            rows.append(None)
            continue
        rows.append(lgpsf.OperatorRow(lgpsf.row_expansion(fit.model, rho),
                                      fit.model.window_center[rho],
                                      fit.model.window_covariance[rho],
                                      fit.model.row_window(rho)))

    rebuilt = lgpsf.build_operator(op["x_cols"], op["m1"], op["m2"], True, rows)
    assert lgpsf.validate_operator(rebuilt) == []
    probes = np.random.default_rng(2).normal(size=(3, op["count"]))
    np.testing.assert_allclose(lgpsf.matvec(rebuilt, probes),
                               lgpsf.matvec(fit.model, probes), atol=1e-12)


def test_the_window_aspect_cap_moves_the_shape():
    op = synthetic_operator()
    ball = fit_synthetic(op, tau_window=1.5, window_aspect_cap=1.0)
    ellipse = fit_synthetic(op, tau_window=1.5,
                            window_aspect_cap=float("inf"))
    for rho in op["chosen"]:
        cap1 = ball.model.window_covariance[rho]
        capinf = ellipse.model.window_covariance[rho]
        # cap = 1 is isotropic; the untouched prior is not
        np.testing.assert_allclose(cap1, cap1[0, 0] * np.eye(2), atol=1e-12)
        assert abs(capinf[0, 0] - capinf[1, 1]) > 1e-12
        # the ball contains the ellipsoid, so it can only have more points
        assert len(ball.model.row_window(rho)) >= len(ellipse.model.row_window(rho))
