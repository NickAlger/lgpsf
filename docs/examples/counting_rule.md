# The counting rule: why a cost near zero is not good news

A mode set of size `m` is only worth fitting when

    k >= 2 (m + P)

probes are available, where `P` is the number of ellipsoid parameters actually
being fitted -- 3 in 2-D with the center pinned, 5 with it free. Below that
threshold the residual can be driven to zero by the coefficients alone, so
every ellipsoid fits equally well and the nonlinear parameters are decided by
noise.

**lgpsf enforces this for you.** The candidate stream marks over-large mode
sets inadmissible and skips them; that is what `ProbeFitResult.skipped`
records. This example shows the guard doing its job, and then deliberately
steps around it -- calling `fit_varpro` directly, which has no guard -- so you
can see what the guard is preventing.

The failure is worth recognizing: **in-sample cost collapses toward zero while
the held-out score gets worse, and the fitted ellipse drifts.** If you ever
build a basis by hand and see a suspiciously perfect fit, count first.

Note that the rule is a safety margin, not a sharp cliff -- the first
inadmissible rung often still scores well, and the collapse arrives a rung or
two later. It is set where it is because it has to hold for every row of an
operator, including the ones with the least signal.

## Output

```text
k = 20 probes, P = 3 pinned ellipsoid parameters
=> admissible mode sets have m <= k/2 - P = 7

                 candidate  modes  admissible     score
                  level<=0      1        True    0.3352
                  level<=1      3        True    0.3604
                  level<=2      6        True    0.2024

shipped 6 modes, held-out score 0.2024
skipped: level<=3; level<=4; level<=5

 modes  admissible   in-sample cost   held-out        1-sigma axes
     3        True        1.864e-07     0.3604     0.045    0.105
     6        True        4.278e-08     0.2024     0.036    0.061
    10       False        8.884e-09     0.1775     0.035    0.071
    15       False        1.546e-09     0.2178     0.049    0.101
    21       False        3.306e-21     0.4657     0.046    0.132

Past the threshold the cost collapses toward zero and the score does not
follow it -- at m = 21 with k = 20 the residual is 1e-21 and the held-out
score is the WORST of the table. The fit is interpolating the probes.

Note the guard is conservative: m = 10 is already inadmissible and still
scores well. It is a margin that has to hold for every row of an operator,
including the ones with the least signal.
```

## Program

```python
import numpy as np

import lgpsf
from frog_kernel import build_problem

TARGET = (0.55, 0.35)
NUM_PROBES = 20


def main():
    problem = build_problem(grid=24)
    x, mass = problem["x"], problem["mass"]
    rho = int(np.argmin(np.linalg.norm(x - np.array(TARGET)[:, None], axis=0)))
    mu0, sigma0 = x[:, rho], problem["sigma"][rho]

    rng = np.random.default_rng(0)
    z = rng.normal(size=(NUM_PROBES, problem["count"]))
    y = z @ problem["H"][rho, :]

    parameters = lgpsf.theta_hat_size(2, lgpsf.MuMode.Pinned)
    budget = NUM_PROBES // 2 - parameters
    print(f"k = {NUM_PROBES} probes, P = {parameters} pinned ellipsoid "
          f"parameters\n=> admissible mode sets have m <= k/2 - P = {budget}\n")

    # ---- 1. the guard, doing its job ------------------------------------
    config = lgpsf.ProbeFitConfig()
    config.mode_policy = lgpsf.ShellLadder([0, 1, 2, 3, 4, 5, 6])
    config.target_score = None
    result = lgpsf.fit_from_probes(x, mass, z, y, mu0, config=config,
                                   sigma0=sigma0, target_mass=mass[rho])

    print(f"{'candidate':>26}  {'modes':>5}  {'admissible':>10}  {'score':>8}")
    seen = set()
    for candidate in result.candidates:
        key = (candidate.modes_label, candidate.num_modes)
        if key in seen:
            continue
        seen.add(key)
        score = "-" if not np.isfinite(candidate.score) else f"{candidate.score:.4f}"
        print(f"{candidate.modes_label:>26}  {candidate.num_modes:>5}  "
              f"{str(bool(candidate.admissible)):>10}  {score:>8}")
    print(f"\nshipped {result.model.num_modes} modes, held-out score "
          f"{result.score:.4f}")
    if result.skipped:
        print("skipped: " + "; ".join(result.skipped[:3]))

    # ---- 2. stepping around it ------------------------------------------
    # fit_varpro is the raw nonlinear solve. It fits what you give it.
    print(f"\n{'modes':>6}  {'admissible':>10}  {'in-sample cost':>15}  "
          f"{'held-out':>9}  {'1-sigma axes':>18}")
    z_hat = lgpsf.whiten_probes(z, mass)
    y_hat = lgpsf.whiten_data(y, mass[rho])
    folds = lgpsf.kfold_split(NUM_PROBES, 5)
    theta0 = lgpsf.to_theta_hat(
        np.concatenate([mu0, np.log(np.sqrt(np.linalg.eigvalsh(sigma0))), [0.0]]),
        mu0, lgpsf.MuMode.Pinned)
    empty = np.zeros((0, problem["count"]))

    for level in [1, 2, 3, 4, 5]:
        modes = lgpsf.modes_up_to_level(2, level)
        basis = lgpsf.WhitenedBasis(x, mass[rho], mass, modes, mu0,
                                    lgpsf.MuMode.Pinned)
        fit = lgpsf.fit_varpro(z_hat, y_hat, basis, theta0, empty)
        score = lgpsf.linear_cv_score(z_hat, y_hat, basis, fit.theta_hat,
                                      empty, folds)
        # theta_hat is the internal encoding; go through the public one to read
        # the ellipsoid off. unpack_theta returns an EllipsoidFrame whose L is
        # the Cholesky factor, so Sigma = L L^T.
        frame = lgpsf.unpack_theta(
            lgpsf.to_theta(fit.theta_hat, mu0, lgpsf.MuMode.Pinned))
        axes = np.sqrt(np.linalg.eigvalsh(frame.L @ frame.L.T))
        ok = len(modes) <= budget
        print(f"{len(modes):>6}  {str(ok):>10}  {fit.cost:>15.3e}  "
              f"{score:>9.4f}  {axes[0]:>8.3f} {axes[1]:>8.3f}")

    print("\nPast the threshold the cost collapses toward zero and the score "
          "does not\nfollow it -- at m = 21 with k = 20 the residual is 1e-21 "
          "and the held-out\nscore is the WORST of the table. The fit is "
          "interpolating the probes.\n\nNote the guard is conservative: m = 10 "
          "is already inadmissible and still\nscores well. It is a margin that "
          "has to hold for every row of an operator,\nincluding the ones with "
          "the least signal.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/counting_rule.py`](../../examples/counting_rule.py); the output and figures above come from actually running it.*
