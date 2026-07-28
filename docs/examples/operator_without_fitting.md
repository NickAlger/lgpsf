# An LGOperator you build yourself, with no fitting at all

`LGOperator` is a data structure, not a fit result. `lg_operator.hpp` depends
on none of the fitting stack -- a test asserts it -- so if you can write down a
point-spread function analytically, you can hand-build the operator and use
every deployment helper without `fit_operator` ever being involved.

Reasons you might:

  * you have a physics-based PSF model and want it in the same sparse-assembly
    and matvec machinery;
  * you want a starting point, or a reference to measure a fit against;
  * you fitted on several machines and need to merge the pieces
    (`concatenate_rows`).

A row is an `LGExpansion` -- absolute `theta`, its mode list, `c` and `s` --
plus the window it is deployed on. Here we build a Gaussian PSF field
directly: one mode, level 0, with the coefficient chosen analytically.

## Output

```text
built 400 rows with no fitting
structural check: clean
assemble_sparse: 18430 nonzeros (11.5% dense)
matvec agrees with the matrix: 0.00e+00
row 200: 1 mode(s), c = 1.000

concatenate_rows: 200 + 200 -> 400 rows
merged matches the whole operator: 0.00e+00

relative Frobenius error of this hand-built operator: 0.7014
Not good -- one unfitted mode at the prior covariance cannot represent
the modulation. That is what the fit is for; see operator_fit_frog.py.
```

## Program

```python
import numpy as np

import lgpsf
from frog_kernel import build_problem, frog_covariance


def gaussian_row(center, sigma, amplitude):
    """One row as an LGExpansion: a single level-0 LG mode on this ellipsoid.

    theta is the PUBLIC absolute encoding, [mu, log-diag, strict-lower], so it
    decodes with `unpack_theta` alone. The Cholesky factor of Sigma gives the
    ellipsoid, and its log-diagonal is what theta stores.
    """
    L = np.linalg.cholesky(sigma)
    theta = np.concatenate([center, np.log(np.diag(L)), L[np.tril_indices(2, -1)]])
    return lgpsf.LGExpansion(theta, [lgpsf.Mode(0, 0, 0)],
                             np.array([amplitude]), np.array([]))


def main():
    problem = build_problem(grid=20)
    x, mass = problem["x"], problem["mass"]
    count = problem["count"]

    # A window per row: everything within 3 sigma of the center. Deployed
    # support IS the window, so this is the region the model is trusted on.
    rows = []
    for rho in range(count):
        center = x[:, rho]
        sigma = frog_covariance(center)
        offset = x - center[:, None]
        mahalanobis = np.einsum("in,ij,jn->n", offset, np.linalg.inv(sigma), offset)
        window = np.flatnonzero(mahalanobis <= 3.0**2)

        model = gaussian_row(center, sigma, amplitude=1.0)
        rows.append(lgpsf.OperatorRow(model, center, 3.0**2 * sigma,
                                      [int(j) for j in window]))

    operator = lgpsf.build_operator(x, mass, mass, False, rows)
    print(f"built {operator.num_rows} rows with no fitting")
    print("structural check:", lgpsf.validate_operator(operator) or "clean")

    # Every deployment helper works on it, unchanged.
    A = lgpsf.assemble_sparse(operator, np.inf)
    print(f"assemble_sparse: {A.nnz} nonzeros "
          f"({100.0 * A.nnz / (count * count):.1f}% dense)")

    v = np.random.default_rng(0).normal(size=(2, count))
    print(f"matvec agrees with the matrix: "
          f"{np.abs(lgpsf.matvec(operator, v) - (A @ v.T).T).max():.2e}")

    # `row_expansion` gives a row back as a standalone model.
    expansion = lgpsf.row_expansion(operator, count // 2)
    print(f"row {count // 2}: {expansion.num_modes} mode(s), "
          f"c = {np.asarray(expansion.c).ravel()[0]:.3f}")

    # ---- merging pieces --------------------------------------------------
    # Build (or fit) disjoint ROW RANGES independently, then concatenate.
    # Useful across MPI ranks, and the reason LGOperator carries its own
    # coordinates. Each part holds ONLY its own rows -- so each is rectangular,
    # with its own `x_rows` and `m1_diag`, and concatenation stacks them.
    half = count // 2
    lower = lgpsf.build_operator(x, mass[:half], mass, False, rows[:half],
                                 x_rows=x[:, :half])
    upper = lgpsf.build_operator(x, mass[half:], mass, False, rows[half:],
                                 x_rows=x[:, half:])
    merged = lgpsf.concatenate_rows([lower, upper])
    print(f"\nconcatenate_rows: {lower.num_rows} + {upper.num_rows} -> "
          f"{merged.num_rows} rows")

    merged_matrix = lgpsf.assemble_sparse(merged, np.inf)
    print(f"merged matches the whole operator: "
          f"{abs(merged_matrix - A).max():.2e}")

    # ---- and it is a real approximation ----------------------------------
    # A pure Gaussian at the a-priori covariance, no modulation and no fitted
    # amplitude, so it should be mediocre -- that is the point of comparison a
    # fit has to beat.
    error = (np.linalg.norm(np.asarray(A.todense()) - problem["H"])
             / np.linalg.norm(problem["H"]))
    print(f"\nrelative Frobenius error of this hand-built operator: {error:.4f}")
    print("Not good -- one unfitted mode at the prior covariance cannot "
          "represent\nthe modulation. That is what the fit is for; see "
          "operator_fit_frog.py.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/operator_without_fitting.py`](../../examples/operator_without_fitting.py); the output and figures above come from actually running it.*
