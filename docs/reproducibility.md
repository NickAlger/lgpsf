# Reproducibility: what is guaranteed, and what is not

lgpsf is deterministic, but "deterministic" needs a scope. This page says
exactly which comparisons are bit-exact, which are not, and why a small number
of rows can legitimately land on different answers when the same input is
fitted by two different builds of the same code.

## What is guaranteed

**Within one build, on one machine, the fit is a pure function of its inputs.**

- **Across thread counts.** `fit_operator` produces bit-identical output for any
  `num_threads`. Rows write disjoint output slots and every shared registry is
  assembled serially in row order afterwards, so nothing depends on scheduling.
  A test asserts this.
- **Across runs.** With `seed` unset the cross-validation folds are round-robin
  and the warm-start jitter table is fixed, so no generator is consulted at all.
  Re-running gives the same bytes.
- **Across callers.** The Python bindings and a C++ program calling the same
  headers produce identical results -- verified on a whole 6557-row field fit,
  on scores, statuses and every deployment variant.
- **Across the inner-solve path.** The QR fast path and the SVD fallback agree
  to a median of one ULP, with no row differing by more than 1e-3 and the global
  metric identical to eight digits.

## What is not guaranteed

**Across builds compiled with different flags, a few rows may differ
materially.** Not by a small amount -- by choosing a different local minimum.

The measured example: the same field fit, compiled `-O3` versus
`-O3 -march=native`.

| difference in a row's CV score | rows (of 5076 live) |
|---|---|
| > 0 | 4634 (91%) |
| > 1e-15 | 1380 (27%) |
| > 1e-9 | 279 (5.5%) |
| > 1e-3 | **4 (0.08%)** |

Median difference: **1.1e-16, one ULP**. Maximum: 5.1e-2, a relative 0.62 on
one row.

## Why

`-march=native` enables FMA and wider vector registers, which changes the
ORDER in which floating-point reductions are accumulated. That is a last-bit
effect on any single arithmetic result -- hence the 1-ULP median.

The per-row fit, however, is a **nonlinear iteration on a multimodal
objective**. Levenberg-Marquardt is a deterministic map, but not a continuous
one in the relevant sense: near a flat or bimodal region of the VarPro
landscape, two starting trajectories that differ in the last bit can descend
into different basins. Both are legitimate local minima of the same problem;
the fit reports whichever it reached.

Two consequences follow, and both are visible in the data:

- The effect is **discrete, not a perturbation**. In half the affected rows the
  mode ladder stopped at a different rung -- three modes rather than six -- so
  the two answers are different models, not the same model shifted slightly.
- The affected rows are **rows the fit already finds hard**. Two of the four
  were in the worst 10% and worst 3% of the field by fit quality. That is the
  mechanism rather than a coincidence: a poorly-determined row is one whose
  objective is flat, and a flat objective is where one ULP decides the outcome.

The inner solve is *not* the cause. Forcing the SVD path instead of the QR
flips exactly the same four rows.

## Does it matter

No, on the evidence available. Over all 5076 live rows of the field fit:

| | mean held-out score | global operator error |
|---|---|---|
| `-O3` | 0.1702 | 0.05209035 |
| `-O3 -march=native` | 0.1702 | 0.05208739 |

`-march=native` was better on 2300 rows of 5076 -- 45%, a coin flip. There is no
systematic direction, and the global error moves in the fifth decimal place.

The affected rows are also not the negligible ones: one carried six times the
median row energy, and three of the four sat within 3-7 km of the edge of the
data region against a field median of 85 km. So this is not "only tiny rows
wobble". It is "a few genuinely hard rows are on a knife edge", which is a
property of those rows rather than of the arithmetic.

## Practical guidance

**Comparing two results.** Expect ~90% of rows to differ by roughly one ULP and
a handful to differ materially. Compare field statistics, not individual rows,
unless the builds are identical.

**Reading a failed comparison.** The shape of the disagreement tells you which
kind of problem you have:

| what you see | what it means |
|---|---|
| 90% of rows differ by ~1 ULP, a few by more | floating point; expected across builds |
| a few percent of rows differ by 1e-3 or more | NOT floating point -- something is wrong |
| whole arrays differ, or shapes disagree | a marshalling or layout bug |
| results change with `num_threads` | a genuine bug; this is tested and must not happen |

That middle row is the useful one. One ULP amplifying on 0.08% of rows is the
signature of this phenomenon; 5% of rows differing by 1e-3 is not, and should
be investigated rather than attributed to the compiler.

**If you need cross-build reproducibility** -- for a regression test, or to
compare against a stored reference -- pin the compiler flags. Bit-identity is a
within-build property, so any acceptance criterion phrased as "bit-for-bit" has
to say against which build.

**If you need the fit itself to be less basin-sensitive**, that is a different
request and this document is not the answer to it: raise `num_rungs` so more
initializations are tried, or lower `target_score` so the search does not exit
early on a merely-adequate candidate. Neither is currently believed necessary --
the affected rows cost nothing measurable.
