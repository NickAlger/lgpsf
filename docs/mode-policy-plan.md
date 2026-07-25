# Mode-growth policies: the extensible ladder axis -- agreed design

**Status: DESIGN AGREED (Nick + session 2026-07-25). Implementation in
two reviewable steps: (1) the policy axis with feedback-blind built-ins
(a pure refactor, provably a no-op against existing results), (2) the
adaptive MarginGreedy policy behind a noise gate.**

## Why

PIG slice-38 evidence (real Hessian probes, 6557-row operator fits):
the best mode-growth ORDER is budget- and row-dependent --

- k=20 (counting rule caps m at 6): complete shells (G + dipoles +
  quadrupoles) beat pure-radial-first 0.0608 vs 0.0808 -- at tiny
  budgets the REPAIR modes win (dipoles absorb center error, quads
  absorb shape error; the enrichment-derivative structure).
- k=100: the radial wedge (2p+ell <= 10, ell <= 2, 26 modes) ties the
  full shell ladder (0.0150 vs 0.0147) at ONE SIXTH the fit cost --
  at large budgets these PSFs want radial depth, and full shells waste
  most of their high-ell modes.

No fixed ordering dominates, so the ladder must become a pluggable
POLICY -- shells, wedges, radial-first, explicit lists, and a
data-driven adaptive frontier as instances of one mechanism -- without
touching the engine's guards or selection semantics.

## The principled structure

The module's three-way classification (probe_fit docstring) already
separates: STRUCTURAL guards (engine), COMPETING HYPOTHESES (candidate
generation), NUMERICS (VarProOptions). The mode axis moves fully into
the second category: a ModePolicy PROPOSES mode sets; the engine keeps
the counting rule, admissibility, CV scoring, patience/target
certificates, warm starts, and the selection rule. The policy never
sees masses or whitening -- adaptive feedback reaches it only through
an engine-built scorer closure.

    class ModePolicy:
        def propose(self, ctx) -> Optional[(label, modes)]
            # next set to fit, or None to end the ladder
        def baseline_sets(self, ctx) -> List[modes]
            # the sets the operator layer's a-priori baseline guard
            # may score; default = feedback-blind replay of propose()

**Policies are stateless**: propose() derives its position from
ctx.history (all prior proposals, including counting-rule-skipped
ones, with each level's winning CandidateFit or None). Statelessness
makes baseline_sets a trivial default (replay with empty feedback),
makes the C++ port a virtual interface with no lifecycle, and keeps
resume/debug simple.

**ModeSearchContext** (engine-built, passed to every propose call):
  - N, k, n_extra, P (stream encoding size);
  - m_max: the counting-rule budget, m_max = k//2 - n_extra - P, so
    well-behaved policies self-censor (the engine still enforces);
  - history: list of LevelRecord(label, modes, skipped, winner);
  - margin_profit: None before any fit; afterwards a callable mapping
    candidate modes -> their EXACT one-step SSE reductions against the
    current winner's reduced residual at its fitted theta (VarPro's
    inner problem is linear, so this is a projection, not a refit:
    whitened feature eval + small QR against the active design --
    thousands of times cheaper than an LM fit).

**Contracts** (engine-enforced): proposals must be supersets of the
last fitted set (patience and warm-start semantics assume a growing
Lambda; violation raises); a hard cap on total proposals guards
against non-terminating policies; oversized proposals are recorded as
skipped and the policy is polled again.

## Built-in policies

| policy | replaces / adds |
|---|---|
| FixedSet(modes) | the `modes` argument |
| ShellLadder(levels) | config.mode_levels |
| ExplicitLadder(sets) | config.mode_sets |
| WedgeLadder(max_level, ell_max=2) | level-ordered ell-capped wedges W_L (psfladder order="level"); N-D via the harmonic table counts |
| RadialFirstLadder(max_level, ell_max=2, groups_per_rung=2) | pure-radial-first prefixes (psfladder order="radial-first") |
| MarginGreedy(...) | NEW: the adaptive frontier (below) |

Config: ProbeFitConfig.mode_policy joins the exactly-one-of family
{mode_policy, mode_levels, mode_sets, `modes` argument}; the legacy
three resolve to the corresponding policies internally, so existing
callers and results are untouched.

## MarginGreedy (step 2)

Downward-closed active set Lambda in the (p, ell) lattice (trig/
harmonic partners grouped); candidates = the margin (indices whose
parents are all in Lambda). Per rung: score the margin's groups with
ctx.margin_profit, admit the best group -- or a Doerfler bulk: the
smallest margin subset capturing a `bulk` fraction of total margin
profit -- subject to a NOISE GATE: a q-dof group must beat
noise_gate * q/nu * ||r||^2 (nu = residual dof), the expected profit
of pure noise, else the policy returns None (a principled stop on top
of patience). Grounding: Gerstner-Griebel dimension-adaptive growth,
Blatman-Sudret adaptive sparse PCE, Cohen-DeVore-Schwab downward-
closed approximation; the noise gate guards the small-k per-row
selection-noise regime that slice-38 exposed (the released-mu lesson:
greedy optimizers amplify validation noise).

Deliberately NOT in scope yet: cross-row stabilization (aggregate
margin profits over neighborhoods / a shared field-level Lambda) --
the psfladder lesson that one global decision from 1e5 equations
beats 6557 decisions from 20 each. That enters later as candidate
injection at the operator layer, per the operator-api plan.

## Implementation plan

1. `prototype/mode_policy.py`: protocol + context + the five fixed
   policies. `lg_functions.modes_up_to_level` gains an `ell_max`
   kwarg (shared by the wedge builders).
2. `probe_fit.fit_from_probes`: the static `for mlabel, mlist in
   mode_sets` loop becomes a policy poll loop; counting-rule skip,
   patience, target, warm-start bodies unchanged. Config resolution +
   validation. `operator_fit`: base_sets from policy.baseline_sets.
3. Tests: equivalence pins (ShellLadder == mode_levels result,
   ExplicitLadder == mode_sets result, bit-for-bit on the existing
   synthetic problems); contract tests (non-nested proposal raises,
   oversized proposals skipped, proposal cap); N-D wedge counts
   against the harmonic table.
4. Step 2: MarginGreedy + margin_profit closure + noise-gate tests
   (pure-noise rows must stop at the seed; exactly-representable rows
   must recover the truth's support). Benchmark on PIG k=20/40/100
   against the fixed ladders before it can become any default.

Default policy: unchanged behavior (shells via mode_levels) until the
PIG benchmarks justify a new default; the slice-38 numbers suggest
WedgeLadder(10, 2) as the strongest fixed default at k >= 40.
