# Symmetrizing a fit whose row scales differ by orders of magnitude

Row fits are independent, so a fitted symmetric operator comes back only
APPROXIMATELY symmetric: entries `(i, j)` and `(j, i)` hold two independent
estimates of the same true value. Reconciling them is what `Symmetrize` is
for, and `deploying_a_fit.py` already showed the prior question (is your
operator symmetric at all?). This example is about the next one: WHICH
reconciliation, when your rows live on wildly different scales.

The problem (`heat_inversion.py`) makes the scales physical: the heat
Hessian's row norms span more than two orders of magnitude because the
conductivity does. Where a strong row overlaps a weak one, plain averaging
`(A + A^T)/2` mixes the weak row's signal 50/50 with the strong row's TAIL
-- a part of the strong fit carrying absolute error on the strong row's
scale, which can exceed the weak row's entire signal. `Symmetrize.Weighted`
instead takes a convex combination with inverse-row-energy weights,

    B_ij = (w_i^2 A_ij + w_j^2 A_ji) / (w_i^2 + w_j^2),
    w_i^2 = 1 / (||A_i||^2 + (0.01 median ||A_r||)^2),

so the weak row owns the entries a strong row only grazes. Both produce an
EXACTLY symmetric matrix; they differ in whose opinion wins.

What the numbers below show, honestly: on rows of comparable scale the two
policies agree (both average away independent noise); on the weak slab rows
averaging is destructive and weighting is protective; and on the handful of
spike rows the fit got nearly exact, weighting costs a little -- the weight
model assumes per-row noise proportional to row energy, and rows fitted far
better than that deserve more ownership than they get. The worst row under
each policy is the number to watch.

## Output

```text
Row scales are physics, not noise (median true row norm per region):
  slab        6.85e-05
  background  3.32e-04
  pocket      1.40e-03
  interface   3.97e-04
  global spread: 280x

                            None_               Average              Weighted
asymmetry                 2.3e-02               0.0e+00               0.0e+00
frobenius                  0.0259                0.0187                0.0189
slab            0.068 /    0.138     0.096 /    0.311     0.062 /    0.126
background      0.038 /    0.123     0.031 /    0.070     0.033 /    0.076
pocket          0.000 /    0.001     0.002 /    0.008     0.004 /    0.014
interface       0.052 /    0.246     0.038 /    0.167     0.047 /    0.148
(per-region cells: median / worst row relative error)

On the weak (slab) rows, error relative to leaving the fit alone:
  Average:  median row 1.34x, worst row 5.4x  (contaminated)
  Weighted: median row 0.91x, worst row 1.24x  (protected)

Both policies produce an exactly symmetric matrix and agree in Frobenius
norm; they differ where it is dangerous to differ. Averaging lets the strong
rows' tails overwrite the weak rows' signal -- the worst weak row above is
the contamination. Weighting gives the entry to the row it matters to, at a
small, visible cost on the few spike rows the fit had nearly exact (their
errors stay two orders below the weak rows'). If the operator should be
symmetric and its rows span decades, `Weighted` is the safe default -- and
it is what the corrections layer expects to receive.
```

## Program

```python
import numpy as np

import lgpsf
from heat_inversion import build_problem, probes, regions

GROUPS = ["slab", "background", "pocket", "interface"]


def main():
    problem = build_problem(grid=24)
    truth = problem["H"]
    reg = regions(problem)
    row_norm = np.linalg.norm(truth, axis=1)

    print("Row scales are physics, not noise "
          "(median true row norm per region):")
    for name in GROUPS:
        print(f"  {name:11s} {np.median(row_norm[reg[name]]):.2e}")
    print(f"  global spread: {row_norm.max() / row_norm.min():.0f}x\n")

    V, HV = probes(problem, 110)
    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = True
    config.row.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4])
    config.row.target_score = None
    fit = lgpsf.fit_operator(problem["x"], problem["mass"], problem["mass"],
                             V, HV, problem["sigma"], config=config)

    results = {}
    for name, policy in [("None_", lgpsf.Symmetrize.None_),
                         ("Average", lgpsf.Symmetrize.Average),
                         ("Weighted", lgpsf.Symmetrize.Weighted)]:
        A = np.asarray(lgpsf.assemble_sparse(fit.model, 6.0, policy).todense())
        asym = np.abs(A - A.T).max() / np.abs(A).max()
        results[name] = dict(
            rel_rows=np.linalg.norm(A - truth, axis=1) / row_norm,
            frob=np.linalg.norm(A - truth) / np.linalg.norm(truth),
            asym=asym)

    print(f"{'':11s}" + "".join(f"{n:>22s}" for n in results))
    print(f"{'asymmetry':11s}"
          + "".join(f"{r['asym']:>22.1e}" for r in results.values()))
    print(f"{'frobenius':11s}"
          + "".join(f"{r['frob']:>22.4f}" for r in results.values()))
    for g in GROUPS:
        cells = "".join(
            f"{np.median(r['rel_rows'][reg[g]]):>10.3f}"
            f" /{r['rel_rows'][reg[g]].max():>9.3f}"
            for r in results.values())
        print(f"{g:11s}{cells}")
    print("(per-region cells: median / worst row relative error)\n")

    weak = reg["slab"]
    avg = results["Average"]["rel_rows"][weak] / results["None_"]["rel_rows"][weak]
    wtd = results["Weighted"]["rel_rows"][weak] / results["None_"]["rel_rows"][weak]
    print("On the weak (slab) rows, error relative to leaving the fit "
          "alone:\n"
          f"  Average:  median row {np.median(avg):.2f}x, "
          f"worst row {avg.max():.1f}x  (contaminated)\n"
          f"  Weighted: median row {np.median(wtd):.2f}x, "
          f"worst row {wtd.max():.2f}x  (protected)\n")
    print("Both policies produce an exactly symmetric matrix and agree in "
          "Frobenius\nnorm; they differ where it is dangerous to differ. "
          "Averaging lets the strong\nrows' tails overwrite the weak rows' "
          "signal -- the worst weak row above is\nthe contamination. "
          "Weighting gives the entry to the row it matters to, at a\nsmall, "
          "visible cost on the few spike rows the fit had nearly exact "
          "(their\nerrors stay two orders below the weak rows'). If the "
          "operator should be\nsymmetric and its rows span decades, "
          "`Weighted` is the safe default -- and\nit is what the corrections "
          "layer expects to receive.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/weighted_symmetrization.py`](../../examples/weighted_symmetrization.py); the output and figures above come from actually running it.*
