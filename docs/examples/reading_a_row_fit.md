# What the fitter tried, what it kept, and why it stopped

`fit_from_probes` does not perform one fit. It runs an ordered stream of
CANDIDATES -- several initial ellipsoids at several scales, at each rung of the
mode ladder, optionally with the center released -- scores each one on held-out
probes, and ships the best. `ProbeFitResult` carries that whole audit trail,
which is the first place to look when a row disappoints.

The fields worth knowing:

    result.model        the winner: an LGExpansion you can evaluate
    result.score        its held-out cross-validation score (lower is better)
    result.winner       which candidate that was
    result.candidates   every candidate, with label, cost, score, axes,
                        iteration count and whether it was admissible
    result.stop_reason  Target | ModePatience | Exhausted
    result.skipped      mode sets the counting rule refused

Two things trip people up. **`cost` is in-sample and is never the selector** --
`score` is, and they disagree exactly when it matters. And a fit that stopped
with `Target` did not exhaust the ladder; it found something good enough and
saved you the rest.

## Output

```text

--- target_score = None: every candidate is scored ---
  #                           candidate  modes        cost     score  iters
  0                              sigma0      3    5.78e-07    0.3277      6   
  1                     circle r=0.0417      3    5.78e-07    0.3277      6   
  2                      circle r=0.827      3    3.79e-06    0.8275      2   
  3                      warm(level<=1)     10    8.27e-08    0.1642      6   
  4                              sigma0     10    8.27e-08    0.1643      9   
  5                     circle r=0.0417     10    8.27e-08    0.1642      6 <-
  6                      circle r=0.827     10    2.60e-06    0.8146      4   
     shipped 10 modes, score 0.1642, stopped: Exhausted

--- target_score = 0.2: stop once a candidate clears the bar ---
  #                           candidate  modes        cost     score  iters
  0                              sigma0      3    5.78e-07    0.3277      6   
  1                     circle r=0.0417      3    5.78e-07    0.3277      6   
  2                      circle r=0.827      3    3.79e-06    0.8275      2   
  3                      warm(level<=1)     10    8.27e-08    0.1642      6 <-
     shipped 10 modes, score 0.1642, stopped: Target

Early stopping skipped 3 candidate(s) for a score penalty of +0.0000.
On this row the remaining candidates were duplicates of the winner, so it
cost nothing at all. That is the usual case: easy rows exit early, and hard
rows fail the certificate and buy the full search anyway.

 modes   best in-sample cost   best held-out score
     3              5.78e-07                0.3277
    10              8.27e-08                0.1642
Cost falls with every mode added, whether or not the model got better.
The score is what notices; see counting_rule.py for a case where they
part company completely.
```

## Program

```python
import numpy as np

import lgpsf
from frog_kernel import build_problem

TARGET = (0.55, 0.35)
NUM_PROBES = 45


def show(result, note):
    print(f"\n--- {note} ---")
    print(f"{'#':>3}  {'candidate':>34}  {'modes':>5}  {'cost':>10}  "
          f"{'score':>8}  {'iters':>5}")
    for i, c in enumerate(result.candidates):
        mark = "<-" if i == result.winner else "  "
        score = "     -" if not np.isfinite(c.score) else f"{c.score:6.4f}"
        print(f"{i:>3}  {c.label[:34]:>34}  {c.num_modes:>5}  {c.cost:>10.2e}  "
              f"{score:>8}  {c.num_iterations:>5} {mark}")
    print(f"     shipped {result.model.num_modes} modes, score "
          f"{result.score:.4f}, stopped: "
          f"{lgpsf.StopReason(result.stop_reason).name}")


def main():
    problem = build_problem(grid=24)
    x, mass = problem["x"], problem["mass"]
    rho = int(np.argmin(np.linalg.norm(x - np.array(TARGET)[:, None], axis=0)))
    mu0, sigma0 = x[:, rho], problem["sigma"][rho]

    rng = np.random.default_rng(0)
    z = rng.normal(size=(NUM_PROBES, problem["count"]))
    y = z @ problem["H"][rho, :]

    # Two rungs and two initial scales, so the table stays readable.
    config = lgpsf.ProbeFitConfig()
    config.mode_policy = lgpsf.ShellLadder([1, 3])
    config.num_rungs = 2
    config.window_shape_rungs = False
    config.target_score = None            # exhaust the ladder
    exhaustive = lgpsf.fit_from_probes(x, mass, z, y, mu0, config=config,
                                       sigma0=sigma0, target_mass=mass[rho])
    show(exhaustive, "target_score = None: every candidate is scored")

    # The same search, allowed to stop as soon as a candidate is good enough.
    config.target_score = 0.2
    early = lgpsf.fit_from_probes(x, mass, z, y, mu0, config=config,
                                  sigma0=sigma0, target_mass=mass[rho])
    show(early, "target_score = 0.2: stop once a candidate clears the bar")

    saved = len(exhaustive.candidates) - len(early.candidates)
    penalty = early.score - exhaustive.score
    print(f"\nEarly stopping skipped {saved} candidate(s) for a score penalty "
          f"of {penalty:+.4f}.\nOn this row the remaining candidates were "
          "duplicates of the winner, so it\ncost nothing at all. That is the "
          "usual case: easy rows exit early, and hard\nrows fail the "
          "certificate and buy the full search anyway.")

    # In-sample cost and held-out score are different quantities, and the
    # asymmetry is the point: cost can only improve as modes are added, since
    # more parameters always fit the SAME data better.
    print(f"\n{'modes':>6}  {'best in-sample cost':>20}  {'best held-out score':>20}")
    finite = [c for c in exhaustive.candidates if np.isfinite(c.score)]
    for count in sorted({c.num_modes for c in finite}):
        same = [c for c in finite if c.num_modes == count]
        print(f"{count:>6}  {min(c.cost for c in same):>20.2e}  "
              f"{min(c.score for c in same):>20.4f}")
    print("Cost falls with every mode added, whether or not the model got "
          "better.\nThe score is what notices; see counting_rule.py for a case "
          "where they\npart company completely.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/reading_a_row_fit.py`](../../examples/reading_a_row_fit.py); the output and figures above come from actually running it.*
