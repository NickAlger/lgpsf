# Deploying the corrected operator: one build, every shift, matrix-free

The struct built by the previous examples represents `B + E + a Hr` for a
CALLER-supplied shift `a` -- the build shift `a0` appears only in the
contracts. That is the shape a regularization sweep needs: one Hessian,
many `a`, and nothing refactored in between. This example is the consumer's
side of that bargain.

**Two deployment shapes, one fit.** The GLR deployment (the architecture
validated at field scale) is `(a Hr + |S|)^{-1}` applied in closed form
by `glr_precondition` -- one Hr-solve plus rank-sized arithmetic per
application, any positive shift, changing `a` costs scalar arithmetic.
Its quality is governed by `lambda_{k+1}/a`, the first pencil value NOT
in its cache over the shift -- so it wants a DEEP mode cache
(`extend_modes`: Hr-solves only, and by design it never changes the
operator), and it discards the sparse fit at deployment time. The
two-level deployment (`solve(..., TwoLevel)`) keeps the sparse fit in
the loop -- an inner PCG on the full `B + E + a Hr` -- and so profits
from deflation instead of cache depth. (The inner solve is applied
tightly below so plain outer CG stays valid; a production wrapper that
loosens it must use a FLEXIBLE outer method, FCG or FGMRES -- the
contract documented on the solver. `glr_solve` is `glr_precondition`'s
exact-solve sibling, defined only where `a Hr + S` itself is positive
definite; deflation legitimately stores negative error modes, and the
|S| variant is what preconditions below their magnitude.)

**Zones.** Every solve is checked against the certified contracts:
`a >= a0` is guaranteed, shifts between the floor and `a0` proceed with a
warning and an analytic runtime certificate, and shifts at or below the
certified floor are REFUSED -- not because the library is timid, but
because `B + E + a Hr` is provably indefinite there.

**`rebuild_at`.** When the sweep needs a shift below the floor, the
archive is what saves the day: re-certification discovers only the newly
opened flip window, and the value-pass pairs bought in `deflation.py` fold
back in exactly, by linearity, with ZERO new heat solves. Build at the
smallest shift you anticipate; `rebuild_at` covers the shift you did not.

## Output

```text
One fit, two deployments, certified floor 1.3e-05:
  two-level struct: 33 modes (flips + value-pass corrections)
  GLR struct:       123 modes (flips + cached top pencil modes;
                    first uncached value lambda_k+1 = 1.6e-04 -- GLR's
                    conditioning at shift a is roughly 1 + lambda_k+1/a)

The regularization sweep, matrix-free, ZERO refactorization
(outer PCG on the true system, iterations to 1e-6 / 1e-10):
        a         zone   prior only   GLR mode  two-level
    3e-05   Guaranteed    89/ 142     22/ 35     10/ 17
    3e-04   Guaranteed    33/  52      8/ 13      6/  9
    3e-03   Guaranteed    14/  21      5/  7      4/  7
Both columns reuse the same build at every shift. GLR pays one Hr-solve per
application and improves with cache depth (watch lambda_k+1 fall as you
extend); two-level pays a few inner iterations per application and profits
from the deflation instead. At shifts below lambda_k+1 the cache is too
shallow and GLR fades toward the prior -- deepen it, or switch shapes.

Asking for a = 3.8e-06, below the certified floor:
  refused: lgpsf::corrections::solve: a = 0.000004 is at or below the certified floor 0.0...

rebuild_at(3.8e-06): 21 new flips, new floor -1.8e-06,
  value pairs refolded with 0 new heat solves (linearity: the
  archived pairs re-anchor in the new metric exactly).
  the refused shift, after rebuild: Guaranteed, two-level PCG 29 / 44

The whole pipeline, for reference (fit -> symmetrize -> certify -> deflate
-> deploy), is the code of these four examples in sequence:
  B, Z, Y = assemble_sparse(fit_operator(...), tau, Weighted)
  A = make_shifted_operator(sparse_op(B), archive(Z, Y), hr_oracle, a0)
  make_pd(A)                    # certified PD contract
  value_pass(A, Hd, m)          # price the fit's remaining error
  solve(A, b, a) / glr_solve    # any shift above the floor,
  rebuild_at(A, a1)             # ...and below it, from the archive.
```

## Program

```python
import numpy as np

import lgpsf
from heat_inversion import build_problem, iterations_to, lean_fit, pcg

corr = lgpsf.corrections

A0 = 3e-5
SWEEP = [3e-5, 3e-4, 3e-3]


def main():
    problem = build_problem(grid=24)
    B_sparse, Z, Y = lean_fit(problem, num_probes=20)
    Hrd = problem["Hr"].toarray()
    truth = problem["H"]
    n = problem["count"]
    Hd_op = corr.SymmetricOp(n, problem["apply_Hd"])

    archive = corr.ProbeArchive()
    archive.Z = Z
    archive.Y = Y

    # the two-level deployment: certify, then spend 30 heat solves on values
    A = corr.make_shifted_operator(corr.sparse_op(B_sparse), archive,
                                   corr.sparse_hr_oracle(problem["Hr"]), A0)
    assert corr.make_pd(A, max_iters=3000).certified
    corr.value_pass(A, Hd_op, 30, corr.ValuePassMode.V2)
    floor = -A.lambda_floor

    # the GLR deployment: certify, then spend Hr-solves on cache DEPTH
    G = corr.make_shifted_operator(corr.sparse_op(B_sparse),
                                   corr.ProbeArchive(),
                                   corr.sparse_hr_oracle(problem["Hr"]), A0)
    assert corr.make_pd(G, max_iters=3000).certified
    corr.extend_modes(G, 60, max_iters=1500)
    cache = corr.extend_modes(G, 60, max_iters=1500)
    print(f"One fit, two deployments, certified floor {floor:.1e}:\n"
          f"  two-level struct: {A.block.rank} modes "
          "(flips + value-pass corrections)\n"
          f"  GLR struct:       {G.block.rank} modes (flips + cached top "
          "pencil modes;\n                    first uncached value "
          f"lambda_k+1 = {cache.next_value:.1e} -- GLR's\n"
          "                    conditioning at shift a is roughly "
          "1 + lambda_k+1/a)\n")

    def outer_counts(a, solve_P):
        history = pcg(lambda v: (truth + a * Hrd) @ v, solve_P,
                      np.random.default_rng(3).normal(size=n))
        return iterations_to(history, [1e-6, 1e-10])

    print("The regularization sweep, matrix-free, ZERO refactorization\n"
          "(outer PCG on the true system, iterations to 1e-6 / 1e-10):")
    print(f"{'a':>9s} {'zone':>12s} {'prior only':>12s} {'GLR mode':>10s}"
          f" {'two-level':>10s}")
    for a in SWEEP:
        zone = corr.classify_shift(A, a).zone
        prior = outer_counts(a, lambda r: A.hr.solve(
            r[:, None], 1e-10).ravel() / a)
        glr = outer_counts(a, lambda r: corr.glr_precondition(
            G, r[:, None], a).ravel())
        two = outer_counts(a, lambda r: corr.solve(
            A, r[:, None], a, mode=corr.SolveMode.TwoLevel,
            rtol=1e-10).X.ravel())
        print(f"{a:>9.0e} {str(zone).split('.')[-1]:>12s} "
              f"{prior[0]:>5d}/{prior[1]:>4d} {glr[0]:>6d}/{glr[1]:>3d} "
              f"{two[0]:>6d}/{two[1]:>3d}")
    print("Both columns reuse the same build at every shift. GLR pays one "
          "Hr-solve per\napplication and improves with cache depth (watch "
          "lambda_k+1 fall as you\nextend); two-level pays a few inner "
          "iterations per application and profits\nfrom the deflation "
          "instead. At shifts below lambda_k+1 the cache is too\nshallow "
          "and GLR fades toward the prior -- deepen it, or switch shapes."
          "\n")

    # -- below the floor: refusal, then rebuild ---------------------------
    a_low = 0.3 * floor
    print(f"Asking for a = {a_low:.1e}, below the certified floor:")
    try:
        corr.solve(A, np.ones((n, 1)), a_low)
    except ValueError as refusal:
        print(f"  refused: {str(refusal)[:78]}...")

    report = corr.rebuild_at(A, a_low, max_iters=3000)
    print(f"\nrebuild_at({a_low:.1e}): {report.flip.flipped} new flips, "
          f"new floor {report.flip.lambda_floor:.1e},\n"
          f"  value pairs refolded with {report.value_fold.applies} new "
          "heat solves (linearity: the\n  archived pairs re-anchor in the "
          "new metric exactly).")
    its = outer_counts(a_low, lambda r: corr.solve(
        A, r[:, None], a_low, mode=corr.SolveMode.TwoLevel,
        rtol=1e-10).X.ravel())
    print(f"  the refused shift, after rebuild: "
          f"{str(corr.classify_shift(A, a_low).zone).split('.')[-1]}, "
          f"two-level PCG {its[0]} / {its[1]}\n")

    # -- the whole pipeline, start to finish ------------------------------
    print("The whole pipeline, for reference (fit -> symmetrize -> certify "
          "-> deflate\n-> deploy), is the code of these four examples in "
          "sequence:\n"
          "  B, Z, Y = assemble_sparse(fit_operator(...), tau, Weighted)\n"
          "  A = make_shifted_operator(sparse_op(B), archive(Z, Y), "
          "hr_oracle, a0)\n"
          "  make_pd(A)                    # certified PD contract\n"
          "  value_pass(A, Hd, m)          # price the fit's remaining "
          "error\n"
          "  solve(A, b, a) / glr_solve    # any shift above the floor,\n"
          "  rebuild_at(A, a1)             # ...and below it, from the "
          "archive.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/shifted_deployment.py`](../../examples/shifted_deployment.py); the output and figures above come from actually running it.*
