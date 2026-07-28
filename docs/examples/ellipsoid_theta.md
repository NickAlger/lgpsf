# What `theta` is, and the two encodings that catch people out

Every fitted row is an ellipsoid plus coefficients, and the ellipsoid is
stored as `theta`. It is worth ten minutes because two different encodings
exist on purpose, and reading one as the other is silent.

**The public encoding**, which is what `LGOperator.theta` and
`LGExpansion.theta` hold, is ABSOLUTE and always the same length:

    theta = [ mu | log(diag L) | strict lower triangle of L ]

with `L` the lower Cholesky factor of the covariance, `Sigma = L L^T`. The
diagonal is stored as its LOGARITHM, which is what keeps the ellipsoid
positive-definite under an unconstrained optimizer. `unpack_theta(theta)`
needs nothing else -- no center, no mode flag -- which is what makes a fitted
operator self-describing.

**The internal encoding**, `theta_hat`, is what the fitting core actually
searches. It omits the center when the center is pinned, and stores it as a
DISPLACEMENT from `mu0` when it is fitted. That is why it is shorter, and why
it is meaningless without knowing `mu0` and the `MuMode`.

The pullback `T(theta, x) = L^-1 (x - mu)` maps a physical point into the
round coordinates the LG modes live in. Note the direction: T is the PULLBACK,
not a forward map.

## Output

```text
dim 2: theta_size = 5 = 2 centre + 2 log-diagonal + 1 strict-lower
theta = [ 0.3    -0.7    -1.6094 -2.5257  0.06  ]

unpack_theta -> mu [ 0.3 -0.7]
                Sigma recovered to 6.94e-18
                L_inv is cached, L @ L_inv = I to 8.33e-19

    MuMode  theta_hat_size  theta_hat
    Pinned               3  [-1.6094 -2.5257  0.06  ]
    Fitted               5  [ 0.      0.     -1.6094 -2.5257  0.06  ]
Pinned drops the centre entirely. Fitted keeps it as a DISPLACEMENT from
mu0, which is why it reads as zeros here -- we passed mu itself as mu0.

same ellipsoid, mu0 moved by (0.05, -0.02): theta_hat[:2] = [-0.05  0.02]

pullback of the 1-sigma ellipse -> radii [1. 1. 1. 1. 1.]
Exactly 1: the pullback turns the fitted ellipsoid into the unit circle,
which is the whole reason the LG basis can be defined once and reused.

release_mu: 3 parameters -> 5, same ellipsoid, centre now free

On a fitted operator, `theta` is ALWAYS the public encoding, so reading a
row needs nothing but unpack_theta -- and `mu` and `L` are also stored
directly, if you only want the geometry.
```

## Program

```python
import numpy as np

import lgpsf


def main():
    dim = 2
    mu = np.array([0.3, -0.7])
    Sigma = np.array([[0.04, 0.012], [0.012, 0.01]])
    L = np.linalg.cholesky(Sigma)

    theta = np.concatenate([mu, np.log(np.diag(L)), L[np.tril_indices(dim, -1)]])
    print(f"dim {dim}: theta_size = {lgpsf.theta_size(dim)} "
          f"= {dim} centre + {dim} log-diagonal + "
          f"{dim * (dim - 1) // 2} strict-lower")
    print(f"theta = {np.round(theta, 4)}")

    # ---- round trip ------------------------------------------------------
    frame = lgpsf.unpack_theta(theta)
    print(f"\nunpack_theta -> mu {np.round(frame.mu, 4)}")
    print(f"                Sigma recovered to "
          f"{np.abs(frame.L @ frame.L.T - Sigma).max():.2e}")
    print(f"                L_inv is cached, L @ L_inv = I to "
          f"{np.abs(frame.L @ frame.L_inv - np.eye(dim)).max():.2e}")

    # ---- the two encodings ----------------------------------------------
    print(f"\n{'MuMode':>10}  {'theta_hat_size':>14}  theta_hat")
    for mode in [lgpsf.MuMode.Pinned, lgpsf.MuMode.Fitted]:
        theta_hat = lgpsf.to_theta_hat(theta, mu, mode)
        size = lgpsf.theta_hat_size(dim, mode)
        print(f"{mode.name:>10}  {size:>14}  {np.round(theta_hat, 4)}")
        # ...and back again, given the same mu0 and mode.
        assert np.allclose(lgpsf.to_theta(theta_hat, mu, mode), theta)
    print("Pinned drops the centre entirely. Fitted keeps it as a "
          "DISPLACEMENT from\nmu0, which is why it reads as zeros here -- "
          "we passed mu itself as mu0.")

    # A displacement really is a displacement:
    elsewhere = mu + np.array([0.05, -0.02])
    theta_hat = lgpsf.to_theta_hat(theta, elsewhere, lgpsf.MuMode.Fitted)
    print(f"\nsame ellipsoid, mu0 moved by (0.05, -0.02): "
          f"theta_hat[:2] = {np.round(theta_hat[:2], 4)}")

    # ---- the pullback ----------------------------------------------------
    # T maps physical points into the coordinates the LG modes are defined in.
    # A point one Mahalanobis unit from the centre lands on the unit circle.
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    on_ellipse = mu[:, None] + L @ np.vstack([np.cos(angles), np.sin(angles)])
    pulled = lgpsf.pullback(frame, on_ellipse)
    print(f"\npullback of the 1-sigma ellipse -> radii "
          f"{np.round(np.linalg.norm(pulled, axis=0), 6)}")
    print("Exactly 1: the pullback turns the fitted ellipsoid into the unit "
          "circle,\nwhich is the whole reason the LG basis can be defined once "
          "and reused.")

    # ---- release_mu ------------------------------------------------------
    # Promote a pinned parameter vector to a fitted one, to let the centre move
    # after a first pass has settled the shape.
    pinned = lgpsf.to_theta_hat(theta, mu, lgpsf.MuMode.Pinned)
    released = lgpsf.release_mu(pinned, dim)
    print(f"\nrelease_mu: {len(pinned)} parameters -> {len(released)}, "
          "same ellipsoid, centre now free")
    assert np.allclose(lgpsf.to_theta(released, mu, lgpsf.MuMode.Fitted), theta)

    # ---- reading a fitted operator --------------------------------------
    print("\nOn a fitted operator, `theta` is ALWAYS the public encoding, so "
          "reading a\nrow needs nothing but unpack_theta -- and `mu` and `L` "
          "are also stored\ndirectly, if you only want the geometry.")


if __name__ == "__main__":
    main()
```

---

*Generated by `docs/generate_examples.py` from [`examples/ellipsoid_theta.py`](../../examples/ellipsoid_theta.py); the output and figures above come from actually running it.*
