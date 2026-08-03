# SPDX-License-Identifier: MIT
"""Deflation: pricing the error the fit left behind, free or paid.

A lean fit compresses k probe applications into a sparse operator, and what
it could not represent remains as error -- here, concentrated in the smooth
long-range structure near the conductivity jumps. The corrections layer
offers two ways to build a low-rank estimate of that error and fold it into
the operator, both starting from the same free ingredient: the archived
residuals `R = Y - B Z` are exact samples of the error acting on the
probes, already paid for when the fit was probed.

**`deflate_free`** spends nothing more: it estimates the error's
eigenvalues from the residuals themselves by a regularized Rayleigh--Ritz
quotient. The catch, measured at field scale and reproduced below, is
that the VALUES are the hard part: the residual subspace is nearly right,
but in-sample value estimates from few probes are noise, and at this lean
budget the free pass makes the preconditioner WORSE. The report says so
before the iteration counts do -- clamps firing, and value estimates far
outside the range the true values turn out to occupy.

**`value_pass`** spends `m` TRUE operator applications -- here, real heat
solves through the problem's matrix-free `apply_Hd` -- on the exact
Rayleigh matrix of the error over the same free subspace. V1 puts every
apply into values and therefore SATURATES at the residual basis dimension
(ask for 30, get 20 -- `report.applies` says which). V2 spends half the
budget on one power step through the true error first, reaching
directions the archive never saw; it is the variant that keeps improving
past the free basis, and the only one that pays here. The new pairs
`(Q, Hd Q)` join the archive as secant information any later rebuild can
reuse (see `shifted_deployment.py`).

The referee is preconditioned CG on the TRUE system `(Hd + a Hr) x = b`:
each candidate operator serves as the preconditioner, applied exactly by
the layer's Cholesky backend (viable at this small N; the matrix-free
deployment is the next example's subject). Fewer iterations = the
correction closed more of the gap the fit left.

    python examples/deflation.py
"""
import numpy as np

import lgpsf
from heat_inversion import build_problem, iterations_to, lean_fit, pcg

corr = lgpsf.corrections

A0 = 3e-5
TOLS = [1e-6, 1e-10]


def certified_struct(problem, B_sparse, Z, Y):
    archive = corr.ProbeArchive()
    archive.Z = Z
    archive.Y = Y
    A = corr.make_shifted_operator(corr.sparse_op(B_sparse), archive,
                                   corr.sparse_hr_oracle(problem["Hr"]), A0)
    assert corr.make_pd(A, max_iters=3000).certified
    return A


def main():
    problem = build_problem(grid=24)
    B_sparse, Z, Y = lean_fit(problem, num_probes=20)
    Hrd = problem["Hr"].toarray()
    truth = problem["H"]
    n = problem["count"]

    system = truth + A0 * Hrd
    b = np.random.default_rng(3).normal(size=n)
    Hd_op = corr.SymmetricOp(n, problem["apply_Hd"])   # the real heat solver

    def score(A_struct, tag, note=""):
        backend = corr.make_cholesky_backend(B_sparse, problem["Hr"],
                                             A_struct)
        history = pcg(lambda v: system @ v,
                      lambda r: corr.cholesky_solve(backend, A_struct,
                                                    r[:, None], A0).ravel(),
                      b)
        its = iterations_to(history, TOLS)
        print(f"  {tag:42s} PCG {its[0]:>3d} / {its[1]:>3d}   {note}")
        return its

    print(f"PCG on the TRUE system (Hd + a Hr), a = {A0:g}; iterations to "
          f"1e-6 / 1e-10.\nEach preconditioner is the corrected fit, applied "
          "exactly (Cholesky backend).\n")

    # prior only: how hard the problem is with no fit at all
    empty = corr.make_shifted_operator(
        corr.SymmetricOp(n, lambda X: np.zeros_like(np.asarray(X, float))),
        corr.ProbeArchive(), corr.sparse_hr_oracle(problem["Hr"]), A0)
    history = pcg(lambda v: system @ v,
                  lambda r: corr.glr_solve(empty, r[:, None], A0).ravel(), b)
    its = iterations_to(history, TOLS)
    print(f"  {'prior only (a Hr)':42s} PCG {its[0]:>3d} / {its[1]:>3d}")

    A_flip = certified_struct(problem, B_sparse, Z, Y)
    score(A_flip, "certified fit, no deflation")

    A_free = certified_struct(problem, B_sparse, Z, Y)
    rep = corr.deflate_free(A_free, rcond=3e-2)
    score(A_free, "+ deflate_free (rcond 3e-2)",
          f"[kept {rep.kept}, clamped {rep.clamped}]")

    A_vp = certified_struct(problem, B_sparse, Z, Y)
    rep = corr.value_pass(A_vp, Hd_op, 30, corr.ValuePassMode.V1)
    score(A_vp, "+ value_pass V1, m=30 true heat solves",
          f"[kept {rep.kept}, clamped {rep.clamped}, "
          f"theta in [{rep.theta_min:.2f}, {rep.theta_max:.2f}]]")

    A_v2 = certified_struct(problem, B_sparse, Z, Y)
    rep = corr.value_pass(A_v2, Hd_op, 30, corr.ValuePassMode.V2)
    score(A_v2, "+ value_pass V2 (power step), m=30",
          f"[applies {rep.applies}, kept {rep.kept}]")

    print("\nThe free pass is worse than no deflation at all: its "
          "values come from\nthe same 20 residuals that defined the "
          "directions, and pricing 20\ndirections from 20 in-sample "
          "equations is interpolation, not estimation.\nThe reports said "
          "so before the counts did -- clamps, and estimates far\noutside "
          "the true range the value pass measures. V1 buys exact values "
          "but\nsaturates at the 20-dimensional free basis; V2 reaches "
          "past it and is\nthe only construction that beats the "
          "uncorrected fit here. This is the\nfield-scale allocation "
          "lesson in miniature: residual DIRECTIONS are\nnearly free, "
          "trustworthy VALUES cost true applies -- spend there. The\n"
          "pairs V2 bought are now in `archive.Q_vp`, kept because a "
          "rebuild at a\nnew shift can reuse them without touching the "
          "operator again.")


if __name__ == "__main__":
    main()
