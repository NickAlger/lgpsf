# SPDX-License-Identifier: MIT
"""Using a fitted operator: the four ways, and what truncation does.

`fit_operator` returns a parametric object, not a matrix -- about `(P + m + 1)`
doubles per row. Every matrix you might want is a decompression of it, and
which one you want depends on what you are doing:

    matvec(fit, v)               apply it, assembling nothing
    assemble_sparse(fit, tau)    a scipy sparse matrix, for a solver
    eval_entries(fit, rows, cols)  specific entries, by index
    eval_kernel(fit, rows, x)    the underlying continuum kernel, at ARBITRARY
                                 points -- not just mesh nodes

The first three are the DEPLOYED operator and agree with each other exactly.
`eval_kernel` is different in kind: it is the smooth component `Phi~` without
the mass weights, which is what you want for plotting a PSF or evaluating on a
finer mesh.

**Every one of them truncates to the row's fit window by default.** That is
not a performance shortcut. The fit only ever measured the model inside its
window, so model mass outside it is unverified -- and LG modes, being
polynomials times a Gaussian, can grow violently once extrapolated.
`eval_kernel_unrestricted` is the named opt-out; use it knowing that.

    python examples/deploying_a_fit.py
"""
import numpy as np

import lgpsf
from frog_kernel import build_problem, probes

TARGET_ROW = 13 * 24 + 8


def main():
    problem = build_problem(grid=24)
    x, mass = problem["x"], problem["mass"]
    V, HV = probes(problem, 110)

    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False
    config.row.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4, 5, 6])
    config.row.target_score = None
    fit = lgpsf.fit_operator(x, mass, mass, V, HV, problem["sigma"],
                             config=config)
    model = fit.model                       # helpers take the MODEL, not the pair

    print(f"{model.num_rows} rows x {model.num_cols} columns, "
          f"{len(lgpsf.model_rows(model))} carrying a model")

    # ---- 1. the three deployed views agree ------------------------------
    A = lgpsf.assemble_sparse(model, np.inf)
    v = np.random.default_rng(1).normal(size=(3, model.num_cols))
    by_matvec = lgpsf.matvec(model, v)
    by_matrix = (A @ v.T).T
    print(f"\nmatvec vs assemble_sparse : max |difference| "
          f"{np.abs(by_matvec - by_matrix).max():.2e}")

    rows = [TARGET_ROW] * 5
    cols = list(range(TARGET_ROW - 2, TARGET_ROW + 3))
    by_entries = lgpsf.eval_entries(model, rows, cols)
    print(f"eval_entries vs the matrix: max |difference| "
          f"{np.abs(by_entries - np.asarray(A[rows, cols]).ravel()).max():.2e}")

    # ---- 2. truncation is a real restriction ----------------------------
    print(f"\n{'tau':>6}  {'nnz':>8}  {'% dense':>8}  {'rel. error vs truth':>19}")
    truth_norm = np.linalg.norm(problem["H"])
    for tau in [1.0, 2.0, 3.0, 6.0, np.inf]:
        At = lgpsf.assemble_sparse(model, tau)
        error = np.linalg.norm(np.asarray(At.todense()) - problem["H"]) / truth_norm
        print(f"{tau:>6}  {At.nnz:>8}  "
              f"{100.0 * At.nnz / (model.num_rows * model.num_cols):>7.1f}%  "
              f"{error:>19.4f}")

    # ---- 3. the window, and what lies outside it ------------------------
    window = model.row_window(TARGET_ROW)
    print(f"\nrow {TARGET_ROW}: window holds {len(window)} of "
          f"{model.num_cols} columns")

    # A line of query points marching away from the row's center. eval_kernel
    # zeroes past the window; the unrestricted form keeps extrapolating.
    center = model.mu[TARGET_ROW]
    direction = np.array([1.0, 0.0])
    distances = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    query = center[:, None] + direction[:, None] * distances

    # NOTE the shape: query points go in batch-last as (N, Q), but the result
    # comes back ROW-FIRST as (len(rows), Q), because per-row records are
    # indexed by row everywhere in this API.
    restricted = lgpsf.eval_kernel(model, [TARGET_ROW], query)[0, :]
    unrestricted = lgpsf.eval_kernel_unrestricted(model, [TARGET_ROW], query)[0, :]
    print(f"\n{'distance':>9}  {'eval_kernel':>14}  {'unrestricted':>14}")
    for d, r, u in zip(distances, restricted, unrestricted):
        print(f"{d:>9.2f}  {r:>14.4e}  {u:>14.4e}")
    print("The restricted form is exactly zero outside the window. The "
          "unrestricted\none keeps evaluating a model nothing ever checked "
          "there.")

    # ---- 4. symmetry is an assembly POLICY, and it is yours to get right --
    plain = lgpsf.assemble_sparse(model, 6.0, lgpsf.Symmetrize.None_)
    averaged = lgpsf.assemble_sparse(model, 6.0, lgpsf.Symmetrize.Average)
    weighted = lgpsf.assemble_sparse(model, 6.0, lgpsf.Symmetrize.Weighted)
    error_of = lambda M: (np.linalg.norm(np.asarray(M.todense()) - problem["H"])
                          / truth_norm)
    truth_asymmetry = (np.abs(problem["H"] - problem["H"].T).max()
                       / np.abs(problem["H"]).max())
    print(f"\nSymmetrize.None_    error {error_of(plain):.4f}")
    print(f"Symmetrize.Average  error {error_of(averaged):.4f}")
    print(f"Symmetrize.Weighted error {error_of(weighted):.4f}")
    print(f"\nSymmetrizing made it {error_of(averaged) / error_of(plain):.0f}x "
          "WORSE, and correctly so: this operator is\nnot symmetric "
          f"(max|H - H^T| / max|H| = {truth_asymmetry:.2f}), because the frog "
          "kernel's shape\nis anchored at the target. Both symmetrizers are "
          "for operators you KNOW\nare symmetric -- a Gauss-Newton Hessian, "
          "say -- where reconciling the two\nindependent row fits cancels "
          "their noise. `Weighted` is the recommendation\nthere: where the "
          "rows disagree about a shared entry, it sides with the row\nwhose "
          "scale the entry actually matters to, instead of letting a strong "
          "row's\ntail overwrite a weak row's signal. When row scales are "
          "comparable, as on\nthis kernel, the two differ little. Either is "
          "an assembly-time choice that\ndoes not touch the fit, so neither "
          "can warn you when symmetry itself is\nwrong.")


if __name__ == "__main__":
    main()
