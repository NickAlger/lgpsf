# SPDX-License-Identifier: MIT
"""Certifying positive definiteness: the pencil, the threshold, and why the
threshold anchors to a0.

A fitted symmetric operator is INDEFINITE -- independent row fits leave a
tail of small spurious negative eigenvalues even when the true operator is
positive semidefinite. Before the fit can serve as a preconditioner or a
proposal precision for systems `B + a Hr`, that tail needs a policy. The
corrections layer's policy (`lgpsf.corrections.make_pd`) rests on two
decisions this example demonstrates on the heat-inversion problem:

**Count in the pencil, not in Euclidean coordinates.** The right question
is not "which eigenvalues of B are negative" but "which eigenvalues of the
PENCIL B v = lambda Hr v are negative" -- dimensionless data-to-prior
ratios, measured against the regularization operator the deployment will
actually add. Below, the fit has over a hundred negative EUCLIDEAN
eigenvalues, but the pencil view shows them for what they are: noise
hugging zero, three orders of magnitude below the informed modes. (On a
field-scale glaciology Hessian the same comparison gave 1438 Euclidean
negatives against 14 pencil modes below threshold -- and the 14-mode flip
preconditioned within one iteration of the 1438-mode one.)

**Flip below `-gamma * a0`, and leave the noise tail alone.** `make_pd`
corrects only the pencil modes below `-gamma * a0` (default gamma = 1/2)
and records the EXACT contract that results:

    B + E + a Hr  is positive definite  <=>  a > -lambda_floor,

with `lambda_floor` the leftmost SURVIVING pencil value -- certified, not
hoped. On this problem the weighted-symmetrized fit is already clean
enough that at any sensible build shift NOTHING needs flipping, and the
certificate itself is the product: a proof, from matvecs and Hr-solves
alone, that the raw fit is safely positive definite for every shift you
would deploy at. The a0 sweep at the end shows what the threshold is
protecting you from: push a0 toward zero and the flip count and the
Lanczos bill both grow into the noise tail, with nothing gained -- the
spectrum accumulates at zero, and enumerating an accumulation point is
the one thing the layer refuses to do.

    python examples/certified_positive_definiteness.py
"""
import numpy as np
import scipy.linalg as sla

import lgpsf
from heat_inversion import build_problem, lean_fit

corr = lgpsf.corrections


def main():
    problem = build_problem(grid=24)
    B_sparse, _, _ = lean_fit(problem)
    B = np.asarray(B_sparse.todense())
    Hr = problem["Hr"]
    Hrd = Hr.toarray()

    # -- the two ways of counting ----------------------------------------
    euclid = np.linalg.eigvalsh(B)
    pencil = sla.eigh(B, Hrd, eigvals_only=True)
    print("The same fit, counted two ways:")
    print(f"  Euclidean eigenvalues below zero : {int((euclid < 0).sum())}")
    print(f"  pencil eigenvalues below zero    : {int((pencil < 0).sum())}"
          f"   (deepest {pencil.min():.1e}; informed modes reach "
          f"{pencil.max():.3f})")
    a0 = 1e-3
    print(f"  pencil below -a0/2 (a0 = {a0:g})    : "
          f"{int((pencil < -0.5 * a0).sum())}")
    print("The Euclidean count is discretization noise; the pencil view "
          "shows the\ntail hugging zero, orders below both the informed "
          "modes and the threshold.\n")

    # -- the certified struct --------------------------------------------
    hr_oracle = corr.sparse_hr_oracle(Hr)
    A = corr.make_shifted_operator(corr.sparse_op(B_sparse),
                                   corr.ProbeArchive(), hr_oracle, a0)
    report = corr.make_pd(A, max_iters=3000)
    print(f"make_pd at a0 = {a0:g}: flipped {report.flipped}, certified "
          f"lambda_floor = {report.lambda_floor:.2e}\n"
          f"({report.iterations} Hr-solves; matvecs and Hr-solves are ALL "
          "it used --\nno entries, no dense eigensolver)")

    # the contract, checked against dense truth on both sides of the floor
    for factor, side in [(2.0, "just above"), (0.5, "just below")]:
        a = factor * -report.lambda_floor
        least = np.linalg.eigvalsh(B + a * Hrd).min()
        verdict = "positive definite" if least > 0 else "INDEFINITE"
        print(f"  dense check, a {side} the floor ({a:.1e}): "
          f"min eig {least:+.1e} -> {verdict}")
    print("The certificate is exact: definiteness changes exactly where "
          "lambda_floor\nsays it does, and every deployment shift above the "
          "floor is safe.\n")

    # -- what the a0-anchored threshold protects you from ----------------
    print("Shrinking a0 toward zero (fresh build each row):")
    print(f"{'a0':>9s} {'threshold':>11s} {'flipped':>8s} {'Hr-solves':>10s}"
          f" {'certified':>10s}")
    for a0_try in [1e-3, 3e-5, 1e-5, 3e-6]:
        A_try = corr.make_shifted_operator(corr.sparse_op(B_sparse),
                                           corr.ProbeArchive(), hr_oracle,
                                           a0_try)
        rep = corr.make_pd(A_try, max_iters=3000)
        print(f"{a0_try:>9.0e} {-0.5 * a0_try:>11.1e} {rep.flipped:>8d} "
              f"{rep.iterations:>10d} {str(rep.certified):>10s}")
    print("\nThe deeper the threshold reaches into the noise tail, the more "
          "modes it\ntouches and the more the certification costs -- with "
          "nothing to show for\nit, because the deployment shift absorbs "
          "that tail anyway. Exact PSD of\nthe fit alone (a0 = 0) would "
          "mean enumerating the accumulation point at\nzero: mode counts "
          "and cost growing without bound under mesh refinement.\n"
          "Anchor a0 at the smallest shift you will deploy, and the flip "
          "set stays\nsmall, mesh-independent, and certified.")


if __name__ == "__main__":
    main()
