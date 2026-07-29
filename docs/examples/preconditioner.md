# What a fit is FOR: preconditioning a regularized inverse problem

Every other example stops at "here is the error". This one spends the fit.

The setting is the standard one. A large, dense operator `H` is expensive to
apply, and you want to recover `x` from a noisy observation `b`:

    minimize  1/2 ||H x - b||^2  +  alpha/2 ||x||^2

whose normal equations are

    (H^T H + alpha I) x = H^T b.

That system is symmetric positive definite for any `alpha > 0`, so conjugate
gradients applies with no symmetrization and no artificial shift. Its
condition number grows as `alpha` shrinks, so CG gets slower exactly where the
reconstruction gets interesting.

**Fit H once, precondition every alpha.** The regularization parameter is
almost never known in advance -- it is swept, or chosen by discrepancy, or
continued downward -- and the fit does not depend on it. One batch of probes
buys the whole sweep.

**This example is synthetic, and it is worth being plain about that.** It
holds `H` as a dense matrix, which is the one thing the method exists to
avoid. If you really do have the dense matrix in memory, you do not need any
of this -- thresholding its small entries is simpler, exact, and cheaper. The
frog kernel is here because it is known by FORMULA, so every number below can
be checked against a ground truth that a real problem would never hand you.

The setting the method is actually for is MATRIX-FREE: you can apply the
operator to a vector, but you cannot look at its entries, because each
application runs an expensive computation rather than a memory read. The
motivating case is the Gauss-Newton Hessian of a PDE-constrained inverse
problem, where one application costs a linearized forward solve plus an
adjoint solve -- and the same shape appears whenever applying an operator
hides a subproblem.

There the probe count IS the cost, and everything else is bookkeeping. Notice
what `fit_operator` consumes below: `V` and `HV`, and nothing else. That is
exactly what a matrix-free operator can give you. The dense `H` in this file
is used only to score the result. It is also why the mesh-scalability finding
matters -- see `experiments/mesh-scalability.md`: the probe budget stays flat
under refinement, so the expensive part does not grow with the mesh.

Note what is being compressed. `H` is local, so its fit is sparse; but `H^T H`
is DENSE, because two rows of `H` overlap whenever their supports do. The
fitted `H^T H` is sparse anyway, and that sparse matrix is the preconditioner.

`preconditioner.cpp` is the same study in C++, where CG is written out so the
line the preconditioner enters on is visible. Here it is scipy's, which is
what you would actually reach for.

## Output

```text
fit of H from 80 probes: relative error 0.0299, 11.5% dense
H^T H is 100.0% dense; the fitted H^T H is 34.9% dense, with relative error 0.0215

    alpha       cond(A)        CG plain     CG with fit   speedup
    1e-02     4.741e+03            294               15     19.6x
    1e-03     4.740e+04            883               29     30.4x
    1e-04     4.740e+05           2619               78     33.6x
    1e-05     4.740e+06           5000+             259     19.3x

One fit, 80 operator applications, reused at every alpha -- the fit does
not depend on the regularization, so a sweep costs one batch of probes plus
a sparse factorization per value.

Both curves still grow as alpha shrinks; the fit lowers the whole curve
rather than flattening it. The speedup peaks in the middle and falls off at
the smallest alpha, where the approximation's own error -- 2% here -- becomes
what limits how well it can stand in for the true operator.
```

## Program

```python
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

import lgpsf
from frog_kernel import build_problem, probes

NUM_PROBES = 80
ALPHAS = [1e-2, 1e-3, 1e-4, 1e-5]
MAX_ITERATIONS = 5000


def count_iterations(A, b, M=None):
    """Solve with scipy's CG and report how many iterations it took."""
    calls = [0]
    kwargs = dict(maxiter=MAX_ITERATIONS, M=M,
                  callback=lambda _: calls.__setitem__(0, calls[0] + 1))
    try:                                    # scipy >= 1.12
        spla.cg(A, b, rtol=1e-8, **kwargs)
    except TypeError:                       # older scipy spells it `tol`
        spla.cg(A, b, tol=1e-8, **kwargs)
    return calls[0]


def main():
    problem = build_problem(grid=20)
    x, mass, H = problem["x"], problem["mass"], problem["H"]
    n = problem["count"]

    # ---- fit H, once ----------------------------------------------------
    V, HV = probes(problem, NUM_PROBES)

    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False
    config.row.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4, 5, 6])
    config.row.target_score = None
    fit = lgpsf.fit_operator(x, mass, mass, V, HV, problem["sigma"],
                             config=config)

    H_approx = lgpsf.assemble_sparse(fit.model, np.inf)
    print(f"fit of H from {NUM_PROBES} probes: relative error "
          f"{np.linalg.norm(np.asarray(H_approx.todense()) - H) / np.linalg.norm(H):.4f}, "
          f"{100.0 * H_approx.nnz / n**2:.1f}% dense")

    # ---- the normal-equations operator, true and approximate ------------
    G = H.T @ H
    G_approx = (H_approx.T @ H_approx).tocsc()
    print(f"H^T H is {100.0 * (np.abs(G) > 0).mean():.1f}% dense; the fitted "
          f"H^T H is {100.0 * G_approx.nnz / n**2:.1f}% dense, with relative "
          f"error "
          f"{np.linalg.norm(np.asarray(G_approx.todense()) - G) / np.linalg.norm(G):.4f}")

    # ---- sweep alpha, reusing the SAME fit -------------------------------
    scale = np.trace(G) / n
    b = np.random.default_rng(7).normal(size=n)
    identity = sparse.identity(n, format="csc")

    print(f"\n{'alpha':>9}  {'cond(A)':>12}  {'CG plain':>14}  "
          f"{'CG with fit':>14}  {'speedup':>8}")
    for relative_alpha in ALPHAS:
        alpha = relative_alpha * scale
        A = G + alpha * np.eye(n)

        # The preconditioner: the same shift on the approximation. The fit is
        # NOT recomputed -- only this sparse factorization is. Handing scipy a
        # LinearOperator is all it takes.
        solve = spla.factorized((G_approx + alpha * identity).tocsc())
        M = spla.LinearOperator((n, n), matvec=solve)

        condition = np.linalg.cond(A)
        plain = count_iterations(A, b)
        preconditioned = count_iterations(A, b, M)
        capped = "+" if plain >= MAX_ITERATIONS else " "
        print(f"{relative_alpha:>9.0e}  {condition:>12.3e}  "
              f"{plain:>13d}{capped}  {preconditioned:>14d}  "
              f"{plain / preconditioned:>7.1f}x")

    print(f"\nOne fit, {NUM_PROBES} operator applications, reused at every "
          "alpha -- the fit does\nnot depend on the regularization, so a sweep "
          "costs one batch of probes plus\na sparse factorization per value.")
    print("\nBoth curves still grow as alpha shrinks; the fit lowers the whole "
          "curve\nrather than flattening it. The speedup peaks in the middle "
          "and falls off at\nthe smallest alpha, where the approximation's own "
          "error -- 2% here -- becomes\nwhat limits how well it can stand in "
          "for the true operator.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/preconditioner.py`](../../examples/preconditioner.py); the output and figures above come from actually running it.*
