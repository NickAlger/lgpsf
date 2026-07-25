"""Mode-growth policies: the extensible ladder axis of the probe fit.

See docs/mode-policy-plan.md for the agreed design and the PIG
slice-38 evidence behind it (the best growth ORDER is budget- and
row-dependent: complete shells win at k=20, an ell-capped radial wedge
ties them at 1/6 the cost at k=100 -- so the ladder is a pluggable
POLICY, not a decree).

Division of labor: a ModePolicy PROPOSES mode sets, one at a time; the
engine (probe_fit.fit_from_probes) keeps every structural guard and
all selection semantics -- the counting rule, window-containment
admissibility, CV scoring, patience/target certificates, warm starts,
the simplicity tie-break. Policies never see masses or whitening;
adaptive feedback reaches them only through the engine-built
margin-profit scorer in the context object.

Policies are STATELESS: propose() derives its position from
ctx.history (every prior proposal, including counting-rule-skipped
ones, with the fitted level's winning candidate or None). This makes
the feedback-blind baseline replay trivial, keeps resume/debugging
simple, and ports to C++ as a virtual interface with no lifecycle.

Contracts (engine-enforced): each proposal must contain the previously
FITTED set (patience/warm-start semantics assume a growing mode set);
labels must be unique; oversized proposals are recorded as skipped and
the policy is polled again; a hard proposal cap guards against
non-terminating policies.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from lg_functions import modes_up_to_level
from lg_harmonics_table import TABLE

MAX_PROPOSALS = 64
"""Engine-side cap on policy proposals per fit (runaway guard)."""


@dataclass
class LevelRecord:
    """One entry of the ladder history handed back to policies."""

    label: str
    modes: List[Tuple[int, int, int]]
    skipped: bool
    """True if the counting rule rejected the proposal (never fit)."""
    winner: Optional[object]
    """The level's best admissible CandidateFit (probe_fit), or None if
    the level was skipped or produced no admissible candidate."""


@dataclass
class ModeSearchContext:
    """Everything a policy may condition on. Engine-built."""

    N: int
    k: int
    """Number of probe equations."""
    n_extra: int
    P: int
    """Theta parameter count of the stream's encoding."""
    history: List[LevelRecord] = field(default_factory=list)
    margin_profit: Optional[Callable] = None
    """None before any successful fit; afterwards maps a candidate mode
    list -> per-mode EXACT one-step SSE reductions against the current
    winner's residual at its fitted theta (a projection, not a refit)."""
    resid_norm2: float = 0.0
    """Squared norm of that residual (the noise-gate denominator)."""

    @property
    def m_max(self):
        """Largest mode count passing the counting rule
        k >= 2 (m + n_extra + P)."""
        return self.k // 2 - self.n_extra - self.P

    def last_fit(self):
        """The most recent non-skipped record, or None."""
        for rec in reversed(self.history):
            if not rec.skipped:
                return rec
        return None


class ModePolicy:
    """Base class. Subclasses implement propose(); baseline_sets has a
    feedback-blind default (replay propose() against an empty-feedback
    context) that is exact for every non-adaptive policy."""

    def propose(self, ctx):
        """Return (label, modes) for the next set to fit, or None to
        end the ladder."""
        raise NotImplementedError

    def baseline_sets(self, ctx):
        """Mode sets the operator layer's a-priori baseline guard may
        score (the baseline must not depend on any adaptive
        trajectory)."""
        blind = ModeSearchContext(N=ctx.N, k=ctx.k, n_extra=ctx.n_extra,
                                  P=ctx.P)
        sets = []
        while len(blind.history) < MAX_PROPOSALS:
            prop = self.propose(blind)
            if prop is None:
                break
            label, ms = prop
            blind.history.append(
                LevelRecord(label, [tuple(m) for m in ms], False, None))
            sets.append([tuple(m) for m in ms])
        return sets


class _SequencePolicy(ModePolicy):
    """Feedback-blind policies: a fixed sequence indexed by how many
    proposals have been made."""

    def _sequence(self, ctx):
        raise NotImplementedError

    def propose(self, ctx):
        seq = self._sequence(ctx)
        i = len(ctx.history)
        return seq[i] if i < len(seq) else None


class FixedSet(_SequencePolicy):
    """A single explicit mode set (the `modes` argument as a policy)."""

    def __init__(self, modes, label="explicit"):
        self.modes = [tuple(m) for m in modes]
        self.label = label

    def _sequence(self, ctx):
        return [(self.label, list(self.modes))]


class ShellLadder(_SequencePolicy):
    """Complete oscillator shells up to each listed level, ascending
    (the classic config.mode_levels ladder)."""

    def __init__(self, levels):
        self.levels = sorted(levels)

    def _sequence(self, ctx):
        return [(f"level<={L}", modes_up_to_level(ctx.N, L))
                for L in self.levels]


class ExplicitLadder(_SequencePolicy):
    """A caller-supplied nested list of mode sets (config.mode_sets)."""

    def __init__(self, sets):
        self.sets = [[tuple(m) for m in ms] for ms in sets]

    def _sequence(self, ctx):
        return [(f"set{i}(m={len(ms)})", list(ms))
                for i, ms in enumerate(self.sets)]


class WedgeLadder(_SequencePolicy):
    """Level-ordered ell-capped wedges W_L = {2p + ell <= L,
    ell <= ell_max}, L ascending (psfladder order="level"; the
    strongest fixed policy at k >= 40 in the PIG slice-38 study).
    Consecutive equal sets (levels adding only ell > ell_max modes)
    are deduplicated."""

    def __init__(self, max_level=10, ell_max=2):
        self.max_level = max_level
        self.ell_max = ell_max

    def _sequence(self, ctx):
        seq = []
        for L in range(self.max_level + 1):
            ms = modes_up_to_level(ctx.N, L, ell_max=self.ell_max)
            if not seq or len(ms) > len(seq[-1][1]):
                seq.append((f"wedge<={L}", ms))
        return seq


def _harmonic_group(N, p, ell):
    """All modes of one (p, ell) group: the harmonic multiplet that
    always travels together (cos/sin pairs in 2D)."""
    if (N, ell) not in TABLE:
        return []
    _, rows = TABLE[(N, ell)]
    return [(p, ell, m) for m in range(len(rows))]


class RadialFirstLadder(_SequencePolicy):
    """Pure-radial-FIRST prefixes (psfladder order="radial-first"):
    all radial groups (p, 0) ascending in p, then ell = 1 groups, then
    ell = 2, ... -- the tight-budget hypothesis of spending on radial
    depth before angular structure (note: at k=20 on PIG, shells beat
    this; kept as a policy because the ordering question is exactly
    what the axis exists to explore). Rungs add groups_per_rung
    harmonic groups at a time (the first rung is the seed group
    alone)."""

    def __init__(self, max_level=10, ell_max=2, groups_per_rung=2):
        self.max_level = max_level
        self.ell_max = ell_max
        self.groups_per_rung = max(1, groups_per_rung)

    def _sequence(self, ctx):
        groups = [_harmonic_group(ctx.N, p, 0)
                  for p in range(self.max_level // 2 + 1)]
        for ell in range(1, self.ell_max + 1):
            for p in range((self.max_level - ell) // 2 + 1):
                g = _harmonic_group(ctx.N, p, ell)
                if g:
                    groups.append(g)
        seq = []
        stop = 1
        while stop <= len(groups):
            ms = [m for g in groups[:stop] for m in g]
            seq.append((f"radial#{len(seq)}(m={len(ms)})", ms))
            stop += self.groups_per_rung
        if seq and len(seq[-1][1]) < sum(len(g) for g in groups):
            ms = [m for g in groups for m in g]
            seq.append((f"radial#{len(seq)}(m={len(ms)})", ms))
        return seq
