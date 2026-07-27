# The frozen Python prototype — HISTORY, NOT THE IMPLEMENTATION

> **Do not read this directory to learn how lgpsf works, and do not copy code
> out of it.** It is a superseded first draft, kept only so the project's
> reasoning can be traced. The library is the header-only C++17 code in
> [`include/lgpsf/`](../../include/lgpsf/) and its Python bindings, which are
> what `import lgpsf` gives you.

Frozen at **`e5c36c9` (2026-07-25)**, the commit where the C++ port began.
Nothing here has been developed since, and nothing here will be.

## What it is

The whole method was built and verified in Python first, then ported. This is
that Python. It is complete and its own tests pass (82 of them, `pytest` from
this directory), which is the only property it is maintained for: a design
record that still runs is harder to misremember than one that doesn't.

It is **not** a specification, a reference implementation, a test oracle, or a
source of golden values. **The C++ is the ground truth.** Where the two
disagree, that is a fact about this directory — not a bug report against the
C++, and never a reason to change the C++ to match.

## Deliberate divergences

Design questions reopened during the port. In every case the C++ is the
decision and this directory is the discarded alternative.

| | prototype | C++ |
|---|---|---|
| parameter encoding | one `theta`, with a `mu0=None` sentinel selecting fixed/free | public absolute `theta` **and** internal `theta_hat` (center as a displacement); `mu0` always required, `MuMode` enum |
| default mu policy | `fixed_then_release` | `MuPolicy::Pinned` — release ships on ~91% of PIG rows while buying nothing, so it is now opt-in |
| randomness | CV folds and warm-start jitter drawn inside the row fit | hoisted to the operator layer and passed **down as data**; the row fit is a pure function of its inputs |
| operator window | a kd-ball of radius `tau * largest 1-sigma axis` | one continuous `window_aspect_cap` (1 = that ball, infinity = the caller's ellipsoid, the default) |
| window derivation | one ball query per row | all rows from ONE dual-tree descent |
| counting rule | the free-mu parameter count everywhere, including for pinned fits | the count of the parameters **actually being fit** |
| inner solve | one BDCSVD per trial point | QR-first (`InnerFactors::RangeOnly`), the SVD kept for Golub-Pereyra and for rank deficiency — 2.1x on a whole-field fit, same answers |

Two comments in the code are known to be wrong and were left unedited, since
the freeze permits only changes that keep its own tests green:

- **`probe_fit.py`** claims the operator layer supplies "a conservatively
  inflated ellipsoid, whose aspect and orientation survive the inflation",
  justifying the `window_shape_rungs` family. It never did — it supplied a
  ball, under which those rungs duplicate the circle rungs. The C++ makes the
  window shape a knob instead.
- **`operator_fit.py:330`** — `P_fix = N + N*(N-1)//2 + N` is the *free-mu*
  count, not an encoding-independent one, so the baseline guard was held to a
  stricter budget than the search. Conservative, hence uncaught. Fixed in the
  C++.

## The harmonic table

`lg_harmonics_table.py` is generated data, and **its generator is not here.**
It lives at [`tools/generate_lg_harmonics_table.py`](../../tools/generate_lg_harmonics_table.py)
because it also emits the shipping C++ header, which is live code. That tool
no longer writes the Python table: this copy is frozen at its 2026-07-25
contents and regenerating it is not supported. Recover the original
dual-output generator from git at `e5c36c9` if you ever need it.

## Layout

Bottom-up: `lg_functions.py` and `harmonic_polynomials.py` (pure basis math)
→ `ellipsoid_transform.py` (the pullback) → `lg_ellipsoid_feature.py` (their
composition) → `whitening.py` (the only layer that touches mass matrices) →
`varpro.py` (the fitting core) → `init_dictionary.py`, `probe_moments.py`,
`mode_policy.py` → `probe_fit.py` (one target) → `operator_fit.py` (all rows).

`examples/` holds the four prototype-era example scripts and their figures.
They ran against this code, not against the library.

`MarginGreedy` in `mode_policy.py` is the one thing here with no C++
counterpart: it was benchmarked, parked, and deliberately not ported. If it is
ever revived, this is the working implementation to start from.

## Running it

Needs numpy and scipy; `operator_fit.py` also needs `ellipsoid_tree`. None of
this is declared in the project's `pyproject.toml`, because the prototype is
not packaged and is not installable — it runs from a checkout or not at all.

```
cd archive/python-prototype && pytest
```
