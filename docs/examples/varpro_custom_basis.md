# The fitting core on its own: VarPro, whitening, and the two Jacobians

Below the operator and row layers sits a general-purpose separable
least-squares solver. It has nothing to do with operators and only a little to
do with Laguerre-Gaussians:

    minimize over (theta, c, s)   || Z (B(theta) c + E s) - y ||^2

`theta` is nonlinear, `c` and `s` are linear. Variable projection eliminates
the linear coefficients in closed form at every trial `theta`, so the outer
Levenberg-Marquardt loop only ever sees a small, well-conditioned problem in
`theta` alone. That is why fitting a 28-mode expansion still only searches
3 parameters.

This example uses the pieces directly:

    whiten_probes / whiten_data   fold the mass matrices in, once
    WhitenedBasis                 B(theta): the basis at a trial ellipsoid
    fit_varpro                    the solve
    VarProOptions                 ridge, tolerances, Jacobian variant

**Whitening is what keeps mass matrices out of the fitting core.** Rescale the
probes by M2^(1/2) and the data by 1/sqrt(target_mass) once, and the whole
problem becomes ordinary Euclidean least squares -- see
docs/varpro-whitening-notes.pdf.

## Output

```text
10 modes in 2-D, but only 3 nonlinear parameters
(log-diagonal and strict-lower of the Cholesky factor; the center is pinned)

        jacobian         cost   iters  residual evals
         Kaufman   1.1094e-07       8               9
    GolubPereyra   1.1094e-07       5               6

same answer to 8.75e-04 in theta.
Kaufman is the default: it drops a term that vanishes at the solution, so
it costs one reverse sweep instead of a full Jacobian tensor and converges
to the same place.

prior 1-sigma axes  0.0500 x 0.1000
fitted 1-sigma axes 0.0342 x 0.0726
10 linear coefficients, never searched over

     ridge         cost       ||c||
     1e-12   1.1094e-07      5.0775
     1e-08   1.1094e-07      5.0775
     1e-04   1.1094e-07      5.0769
     1e-01   1.8789e-07      4.5509
The ridge damps the LINEAR coefficients only -- the ellipsoid is never
regularized, because it is what you are trying to learn.
```

## Program

```python
import numpy as np

import lgpsf
from frog_kernel import build_problem

TARGET = (0.55, 0.35)
NUM_PROBES = 60


def main():
    problem = build_problem(grid=24)
    x, mass = problem["x"], problem["mass"]
    rho = int(np.argmin(np.linalg.norm(x - np.array(TARGET)[:, None], axis=0)))
    mu0, sigma0 = x[:, rho], problem["sigma"][rho]

    rng = np.random.default_rng(0)
    z = rng.normal(size=(NUM_PROBES, problem["count"]))
    y = z @ problem["H"][rho, :]

    # ---- 1. whiten once --------------------------------------------------
    z_hat = lgpsf.whiten_probes(z, mass)          # M2^(1/2) z
    y_hat = lgpsf.whiten_data(y, mass[rho])       # y / sqrt(m1[rho])

    # ---- 2. the basis at a trial ellipsoid -------------------------------
    modes = lgpsf.modes_up_to_level(2, 3)
    basis = lgpsf.WhitenedBasis(x, mass[rho], mass, modes, mu0,
                                lgpsf.MuMode.Pinned)
    # basis.dim is the SPATIAL dimension; the parameter count is separate.
    num_parameters = lgpsf.theta_hat_size(basis.dim, lgpsf.MuMode.Pinned)
    print(f"{len(modes)} modes in {basis.dim}-D, but only {num_parameters} "
          "nonlinear parameters\n(log-diagonal and strict-lower of the "
          "Cholesky factor; the center is pinned)")

    # theta_hat is the INTERNAL encoding: no center, since it is pinned. Start
    # from the a-priori shape.
    axes = np.sqrt(np.linalg.eigvalsh(sigma0))
    theta0 = lgpsf.to_theta_hat(
        np.concatenate([mu0, np.log(axes), [0.0]]), mu0, lgpsf.MuMode.Pinned)
    empty = np.zeros((0, problem["count"]))

    # ---- 3. solve, both Jacobian variants --------------------------------
    print(f"\n{'jacobian':>16}  {'cost':>11}  {'iters':>6}  {'residual evals':>14}")
    solutions = {}
    for variant in [lgpsf.JacobianVariant.Kaufman,
                    lgpsf.JacobianVariant.GolubPereyra]:
        options = lgpsf.VarProOptions()
        options.jacobian = variant
        result = lgpsf.fit_varpro(z_hat, y_hat, basis, theta0, empty, options)
        solutions[variant.name] = result
        print(f"{variant.name:>16}  {result.cost:>11.4e}  "
              f"{result.num_iterations:>6}  {result.num_residual_evaluations:>14}")

    a, b = solutions["Kaufman"], solutions["GolubPereyra"]
    print(f"\nsame answer to {np.abs(a.theta_hat - b.theta_hat).max():.2e} in "
          "theta.\nKaufman is the default: it drops a term that vanishes at "
          "the solution, so\nit costs one reverse sweep instead of a full "
          "Jacobian tensor and converges\nto the same place.")

    # ---- 4. what came back ----------------------------------------------
    result = a
    frame = lgpsf.unpack_theta(
        lgpsf.to_theta(result.theta_hat, mu0, lgpsf.MuMode.Pinned))
    fitted_axes = np.sqrt(np.linalg.eigvalsh(frame.L @ frame.L.T))
    print(f"\nprior 1-sigma axes  {axes[0]:.4f} x {axes[1]:.4f}")
    print(f"fitted 1-sigma axes {fitted_axes[0]:.4f} x {fitted_axes[1]:.4f}")
    print(f"{len(result.c)} linear coefficients, never searched over")

    # ---- 5. the ridge is a real knob -------------------------------------
    print(f"\n{'ridge':>10}  {'cost':>11}  {'||c||':>10}")
    for ridge in [1e-12, 1e-8, 1e-4, 1e-1]:
        options = lgpsf.VarProOptions()
        options.ridge = ridge
        r = lgpsf.fit_varpro(z_hat, y_hat, basis, theta0, empty, options)
        print(f"{ridge:>10.0e}  {r.cost:>11.4e}  {np.linalg.norm(r.c):>10.4f}")
    print("The ridge damps the LINEAR coefficients only -- the ellipsoid is "
          "never\nregularized, because it is what you are trying to learn.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/varpro_custom_basis.py`](../../examples/varpro_custom_basis.py); the output and figures above come from actually running it.*
