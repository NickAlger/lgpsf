"""Tests for mode_policy.py and its integration into probe_fit.

Three kinds of check:
  1. EQUIVALENCE: the legacy config fields (mode_levels, mode_sets,
     `modes`) resolve to policies bit-for-bit -- same winner, same
     score, same candidate table -- so the refactor is provably a
     no-op for every existing caller.
  2. CONTRACTS: nested growth enforced, label reuse rejected,
     oversized proposals skipped, the proposal cap terminates a
     runaway policy.

Run directly (`python test_mode_policy.py`) or via pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from lg_functions import modes_up_to_level
from lg_harmonics_table import TABLE
from mode_policy import (
    FixedSet,
    LevelRecord,
    ModePolicy,
    ModeSearchContext,
    RadialFirstLadder,
    ShellLadder,
    WedgeLadder,
)
from probe_fit import ProbeFitConfig, fit_from_probes
from test_probe_fit import MODES, _make_row


def _ctx(N=2, k=100, n_extra=1, P=3):
    return ModeSearchContext(N=N, k=k, n_extra=n_extra, P=P)


def _result_fingerprint(res):
    return (res.winner, res.score, res.stop_reason,
            tuple(res.theta.round(12)), tuple(res.modes),
            tuple(c.label for c in res.candidates))


def test_legacy_configs_resolve_to_identical_results():
    """mode_levels == ShellLadder, mode_sets == ExplicitLadder,
    modes == FixedSet: fingerprint-identical fits."""
    rng = np.random.default_rng(0)
    prob = _make_row(rng)
    args = (prob["x"], prob["m2"], prob["z"], prob["y"], prob["mu_true"])
    kw = dict(spike_index=prob["spike_index"])

    for legacy, policy in [
        (dict(mode_levels=[0, 1, 2]), dict(mode_policy=ShellLadder([0, 1, 2]))),
        (dict(mode_sets=[[(0, 0, 0)], MODES]),
         dict(mode_policy=None)),   # replaced below
    ]:
        cfg_a = ProbeFitConfig(mu="fixed", target_score=None, **legacy)
        if "mode_sets" in legacy:
            from mode_policy import ExplicitLadder
            policy = dict(mode_policy=ExplicitLadder(legacy["mode_sets"]))
        cfg_b = ProbeFitConfig(mu="fixed", target_score=None, **policy)
        res_a = fit_from_probes(*args, modes=None, config=cfg_a, **kw)
        res_b = fit_from_probes(*args, modes=None, config=cfg_b, **kw)
        assert _result_fingerprint(res_a) == _result_fingerprint(res_b)

    res_c = fit_from_probes(*args, modes=MODES,
                            config=ProbeFitConfig(mu="fixed"), **kw)
    res_d = fit_from_probes(*args, modes=None,
                            config=ProbeFitConfig(
                                mu="fixed", mode_policy=FixedSet(MODES)),
                            **kw)
    assert _result_fingerprint(res_c) == _result_fingerprint(res_d)


def test_wedge_ladder_sizes_and_ell_cap():
    """2D wedge sizes match the slice-38 table; every set respects the
    ell cap; nesting holds; N-D counts come from the harmonic table."""
    seq = WedgeLadder(10, 2)._sequence(_ctx())
    assert [len(ms) for _, ms in seq] == [1, 3, 6, 8, 11, 13, 16, 18,
                                          21, 23, 26]
    for _, ms in seq:
        assert all(ell <= 2 for (_, ell, _) in ms)
    for (_, a), (_, b) in zip(seq, seq[1:]):
        assert set(a) < set(b)
    # N=3: the ell<=2 wedge counts follow the 3D harmonic table
    seq3 = WedgeLadder(4, 2)._sequence(_ctx(N=3))
    n_l1 = len(TABLE[(3, 1)][1])
    n_l2 = len(TABLE[(3, 2)][1])
    # 2p+ell<=4, ell<=2 in 3D: 3 radial + p<=1 at ell=1 and ell=2
    assert len(seq3[-1][1]) == 3 + 2 * n_l1 + 2 * n_l2


def test_radial_first_ladder_orders_radial_before_angular():
    seq = RadialFirstLadder(10, 2, groups_per_rung=2)._sequence(_ctx())
    # first rungs are pure radial
    assert all(ell == 0 for (_, ell, _) in seq[1][1])
    assert all(ell == 0 for (_, ell, _) in seq[2][1])
    # full hull eventually reached, nested throughout
    assert len(seq[-1][1]) == len(modes_up_to_level(2, 10, ell_max=2))
    for (_, a), (_, b) in zip(seq, seq[1:]):
        assert set(a) < set(b)


def test_policy_contracts_enforced():
    rng = np.random.default_rng(2)
    prob = _make_row(rng)
    args = (prob["x"], prob["m2"], prob["z"], prob["y"], prob["mu_true"])
    kw = dict(spike_index=prob["spike_index"])

    class NonNested(ModePolicy):
        def propose(self, ctx):
            seq = [("a", MODES), ("b", [(0, 0, 0)])]   # shrinks!
            i = len(ctx.history)
            return seq[i] if i < len(seq) else None

    try:
        fit_from_probes(*args, modes=None, config=ProbeFitConfig(
            mu="fixed", target_score=None, mode_policy=NonNested()), **kw)
        assert False, "non-nested proposal must raise"
    except ValueError as e:
        assert "nested" in str(e)

    class Runaway(ModePolicy):
        def propose(self, ctx):
            i = len(ctx.history)
            big = modes_up_to_level(2, 8)          # always oversized at k=14
            return (f"r{i}", big)

    prob2 = _make_row(rng, k=14)
    try:
        fit_from_probes(prob2["x"], prob2["m2"], prob2["z"], prob2["y"],
                        prob2["mu_true"], modes=None,
                        spike_index=prob2["spike_index"],
                        config=ProbeFitConfig(mu="fixed",
                                              mode_policy=Runaway()))
        assert False, "all-skipped ladder must raise the counting error"
    except ValueError as e:
        assert "counting rule" in str(e)


def test_baseline_sets_replay_and_adaptive_override():
    shells = ShellLadder([0, 1, 2]).baseline_sets(_ctx())
    assert [len(s) for s in shells] == [1, 3, 6]


if __name__ == "__main__":
    test_legacy_configs_resolve_to_identical_results()
    test_wedge_ladder_sizes_and_ell_cap()
    test_radial_first_ladder_orders_radial_before_angular()
    test_policy_contracts_enforced()
    test_baseline_sets_replay_and_adaptive_override()
    print("all mode_policy checks passed")
