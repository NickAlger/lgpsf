# `prototype/` — the frozen Python reference implementation

**This is not the current implementation.** It is old code, kept for
reference. Active development is the header-only C++17 library in
[`include/lgpsf/`](../include/lgpsf/).

Frozen as of **`e5c36c9` (2026-07-25)**, at the point the C++ port began.

## What it is still good for

- **The design record.** Most of this project's reasoning was worked out here
  first, and the module docstrings carry it. The headers in `include/lgpsf/`
  reproduce that reasoning where it still applies.
- **A don't-rot check.** Its own tests still pass and are expected to keep
  passing, which is what keeps the design record above honest. Run them with
  `pytest` from this directory. They test the prototype against ITSELF; no
  test anywhere measures the C++ against this directory.

## What it is not

It is **not a ground truth and not a test oracle.** The C++ is the ground
truth; this is history. Where the two disagree that is a fact about the
prototype, not a bug report against the C++ -- and the C++ must never be
constrained to match it. Several
design questions were reopened during the port, and in every such case the
C++ side is the decision and this directory is the history. When the two
disagree, read [`docs/design-notes.md`](../docs/design-notes.md) — every
divergence is recorded there with its evidence.

Deliberate divergences so far:

| | prototype | C++ |
|---|---|---|
| parameter encoding | one `theta`, with a `mu0=None` sentinel selecting fixed/free | public absolute `theta` **and** internal `theta_hat` (center as a displacement); `mu0` always required, `MuMode` enum |
| default mu policy | `fixed_then_release` | `MuPolicy::Pinned` — release ships on ~91% of PIG rows while buying nothing, so it is now opt-in |
| randomness | CV folds and warm-start jitter drawn inside the row fit | hoisted to the operator layer and passed **down as data**; the row fit is a pure function of its inputs |
| operator window | a kd-ball of radius `tau * largest 1-sigma axis` | one continuous `window_aspect_cap` (1 = that ball, infinity = the caller's ellipsoid, the default) |
| window derivation | one ball query per row | all rows from ONE dual-tree descent |
| counting rule | the free-mu parameter count everywhere, including for pinned fits | the count of the parameters **actually being fit** |
| inner solve | one BDCSVD per trial point | QR-first (`InnerFactors::RangeOnly`), the SVD kept for Golub-Pereyra and for rank deficiency — 2.1x on a whole-field fit, same answers |

## Known-stale comments

Left as-is rather than edited, since the freeze permits only fixes needed to
keep the prototype's own tests passing. They are recorded here so nobody
re-derives a design decision from them:

- **`probe_fit.py` / `row_fit.py` history (`8e64243`)** — "in the intended
  pipeline it IS a conservatively inflated ellipsoid, whose aspect and
  orientation survive the inflation", justifying the `window_shape_rungs`
  initialization family. The intended pipeline was never that: the operator
  layer supplies a **ball**, and so did the pre-lgpsf research pipeline it
  inherited the convention from. Under a ball window those rungs recover
  nothing and duplicate the circle rungs, which is why every PIG operator run
  disables them explicitly. The C++ makes the window shape a knob so the
  question can finally be settled by measurement.
- **`operator_fit.py:330`** — `P_fix = N + N*(N-1)//2 + N  # == theta_size(N, mu0=any)`.
  That expression is not encoding-independent; it is the **free-mu** count, so
  `P_fix == P_free` and the baseline guard was held to a stricter budget than
  the search. Conservative, hence uncaught. Fixed in the C++.

## Layout

Bottom-up: `lg_functions.py` and `harmonic_polynomials.py` (pure basis math)
-> `ellipsoid_transform.py` (the pullback) -> `lg_ellipsoid_feature.py`
(their composition) -> `whitening.py` (the only masses layer) ->
`varpro.py` (the fitting core) -> `init_dictionary.py`, `probe_moments.py`,
`mode_policy.py` -> `probe_fit.py` (one target) -> `operator_fit.py` (all
rows). [`../CLAUDE.md`](../CLAUDE.md) describes each in detail.
