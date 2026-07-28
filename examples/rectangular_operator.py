# SPDX-License-Identifier: MIT
"""When rows and columns live on different meshes.

Nothing about the method requires a square operator. `H = M1 Phi M2` maps one
discretized space to the dual of another, and those spaces can have different
meshes, different sizes and different mass matrices. Two arguments switch it
on:

    x_rows   (N, R)  where the TARGETS live, if not the same as x_cols
    mu0      (N, R)  where each row's PSF is centered, if not at x_rows

Everything else is unchanged. The shapes to keep straight:

    x_cols  (N, K)     m2_diag  (K,)     V   (k, K)   probes live on columns
    x_rows  (N, R)     m1_diag  (R,)     HV  (k, R)   responses live on rows

Here the columns are a fine 30 x 30 grid and the rows a coarse 15 x 15 one --
an operator that reads a fine field and produces a coarse one. Only the smooth
component is fitted: a diagonal SPIKE has no meaning when rows and columns are
not the same degrees of freedom, so `spike` must be off.

    python examples/rectangular_operator.py
"""
import numpy as np

import lgpsf
from frog_kernel import frog_covariance, frog_row

FINE, COARSE = 30, 15
NUM_PROBES = 60


def grid(n):
    axis = (np.arange(n) + 0.5) / n
    mesh = np.meshgrid(axis, axis, indexing="ij")
    return np.vstack([mesh[0].ravel(), mesh[1].ravel()])


def main():
    x_cols, x_rows = grid(FINE), grid(COARSE)
    num_cols, num_rows = x_cols.shape[1], x_rows.shape[1]
    m2 = np.full(num_cols, (1.0 / FINE) ** 2)     # fine cell area
    m1 = np.full(num_rows, (1.0 / COARSE) ** 2)   # coarse cell area

    # The truth: the same frog kernel, evaluated from coarse targets over fine
    # sources. Rectangular by construction.
    H = np.empty((num_rows, num_cols))
    for i in range(num_rows):
        H[i] = m1[i] * frog_row(x_rows[:, i], x_cols) * m2
    print(f"truth is {H.shape[0]} x {H.shape[1]} -- "
          f"{COARSE}x{COARSE} rows, {FINE}x{FINE} columns")

    sigma = np.stack([frog_covariance(x_rows[:, i]) for i in range(num_rows)])

    rng = np.random.default_rng(0)
    V = rng.normal(size=(NUM_PROBES, num_cols))   # probes on the COLUMN space
    HV = V @ H.T                                  # responses on the ROW space

    config = lgpsf.OperatorFitConfig()
    config.tau_window = 3.0
    config.spike = False        # required: no diagonal exists here
    config.row.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4, 5])
    config.row.target_score = None

    fit = lgpsf.fit_operator(x_cols, m1, m2, V, HV, sigma,
                             config=config, x_rows=x_rows)
    model = fit.model

    print(f"fitted operator is {model.num_rows} x {model.num_cols}, "
          f"{len(lgpsf.model_rows(model))} rows carrying a model")

    A = lgpsf.assemble_sparse(model, np.inf)
    error = np.linalg.norm(np.asarray(A.todense()) - H) / np.linalg.norm(H)
    print(f"relative Frobenius error: {error:.4f}  "
          f"({A.nnz} nonzeros, {100.0 * A.nnz / H.size:.1f}% dense)")

    v = rng.normal(size=(3, num_cols))
    print(f"matvec maps ({v.shape}) -> ({lgpsf.matvec(model, v).shape}) "
          "-- column space in, row space out")

    assert error < 0.15, "60 probes should fit this well"

    # Symmetry is not even expressible here, and the library says so.
    try:
        lgpsf.assemble_sparse(model, 6.0, lgpsf.Symmetrize.Average)
    except Exception as failure:                          # noqa: BLE001
        print(f"\nSymmetrize.Average is refused: {failure}")

    print("\nA square problem is the special case where x_rows is omitted and "
          "the two\nmass vectors happen to be equal. Nothing in the fit "
          "assumes it.")


if __name__ == "__main__":
    main()
