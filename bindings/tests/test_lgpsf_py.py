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
    with pytest.raises(ValueError, match=r"\(N, K\)"):
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
