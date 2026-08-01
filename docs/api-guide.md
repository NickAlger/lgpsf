# API guide

Three layers, each usable on its own. Most people need only the top one.

| layer | entry point | fits |
|---|---|---|
| operator | `fit_operator` | every row of an operator |
| target | `fit_from_probes` | one point-spread function |
| core | `fit_varpro` | any separable least-squares problem |

Below them sit the primitives — the Laguerre-Gaussian basis, the ellipsoid
pullback, the whitening transform — which you need only if you are building
something of your own.

## Array conventions

Two conventions coexist, deliberately, and mixing them up is the most common
mistake.

**Point batches are `(N, K)`** — coordinates down, points across. In C++ they
are `(K, N)`, points as rows. These are the *same bytes*: a C-contiguous numpy
`(N, K)` and a column-major Eigen `(K, N)` have identical layout, which is what
makes the Python boundary zero-copy.

**Per-row records are `(R, …)`** — row first, because `fit.mu[rho]` is how
anyone reads one.

So the point array `x` goes *in* as `(N, R)` and `LGOperator.mu` comes *out*
as `(R, N)`.
Similarly `eval_kernel` takes query points `(N, Q)` and returns `(len(rows), Q)`.

> Both differ from `ellipsoid_tree`'s Python convention, which takes points as
> rows. If you pass geometry between the two libraries, transpose.

Square matrices — covariances, Cholesky factors — are **not** mapped this way.
They go through pybind11's Eigen caster, because a silent transpose would turn
a lower-triangular factor into an upper one with no error.

## The operator layer

`fit_operator` returns an `OperatorFit`, which is two things that are worth
keeping apart:

- **`fit.model`** — an `LGOperator`. This is the operator: about `(P + m + 1)`
  doubles per row, and every matrix form is a decompression of it. Pass *this*
  to the helpers, not the pair.
- **`fit.diagnostics`** — how each row went. Read by nothing in evaluation, so
  you can throw it away once you trust the fit.

`LGOperator` deliberately depends on none of the fitting code. You can build
one directly from a physics model with `build_operator`, merge pieces across
ranks with `concatenate_rows`, and use every helper without the fitter being
involved — see
[`examples/operator_without_fitting.py`](../examples/operator_without_fitting.py).

**Helpers**, all taking the model: `matvec`, `assemble_sparse`, `eval_entries`,
`eval_kernel`, `qc_map`, `spike_measure`, `ellipsoid_field`, `validate`,
`model_rows`, `row_expansion`. Covered in
[`examples/deploying_a_fit.py`](../examples/deploying_a_fit.py) and
[`examples/operator_diagnostics.py`](../examples/operator_diagnostics.py).

`assemble_sparse` takes a `Symmetrize` policy: `None_` (rows exactly as
fitted), `Average`, or `Weighted` — the recommendation when the underlying
operator is symmetric, because it reconciles disagreeing rows by inverse row
energy instead of letting a strong row overwrite a weak one. Symmetrizing an
operator that is not symmetric makes it worse, and the library cannot tell —
see the example.

Rows and columns need not be the same points: pass `x_rows` for a rectangular
operator ([`examples/rectangular_operator.py`](../examples/rectangular_operator.py)).

## The target layer

`fit_from_probes` fits one function known only through inner products. An
operator row is the motivating case, not the definition — if you can produce
`y_i = <z_i, phi>` for random `z_i`, you can fit `phi`.

It is a **multi-start**: you pass the starting ellipsoids as
`guesses=[InitialGuess(sigma, mu=None, label="")]`, and the fit appends
`config.num_rungs` circles of its own (`0` to suppress them). `circle_ladder`,
`window_shape_ladder` and `oriented_ladder` build the common dictionaries. See
[`examples/initial_guesses.py`](../examples/initial_guesses.py) and
[defaults.md](defaults.md).

It returns the winning model plus the **whole search**: every candidate with
its label, cost, held-out score, fitted axes and iteration count. That table is
the first place to look when a row disappoints
([`examples/reading_a_row_fit.py`](../examples/reading_a_row_fit.py)).

**`cost` is in-sample and is never the selector.** `score` is. They disagree
exactly when it matters — see
[`examples/counting_rule.py`](../examples/counting_rule.py).

## The core

`fit_varpro` solves

    minimize over (theta, c, s)   || Z (B(theta) c + E s) - y ||²

by variable projection: the linear coefficients are eliminated in closed form
at every trial `theta`, so the outer Levenberg-Marquardt loop only ever sees
the small nonlinear problem. Fitting a 28-mode expansion still searches 3
parameters.

Mass matrices never reach it. `whiten_probes` and `whiten_data` fold them in
once, after which the problem is ordinary Euclidean least squares — the
derivation is in [varpro-whitening-notes.pdf](varpro-whitening-notes.pdf) and
the usage in
[`examples/varpro_custom_basis.py`](../examples/varpro_custom_basis.py).

## Two encodings of `theta`

The **public** encoding is absolute and always `N(N+3)/2` long:

    theta = [ mu | log(diag L) | strict lower triangle of L ],   Sigma = L Lᵀ

`unpack_theta(theta)` needs nothing else, which is what makes a fitted operator
self-describing. The diagonal is stored as a logarithm so an unconstrained
optimizer cannot produce an indefinite covariance.

The **internal** encoding, `theta_hat`, is what the fitting core searches. It
omits the center when pinned and stores it as a *displacement* from a reference
center when fitted, so it is shorter and meaningless without knowing both that
center and the `MuMode`. Candidates do not share one reference — a guess may
carry its own `mu` — which is why everything a fit hands back, including
`CandidateFit.theta_init`, is in the public encoding.

See [`examples/ellipsoid_theta.py`](../examples/ellipsoid_theta.py).

## Mode policies

A `ModePolicy` decides which mode sets the ladder proposes. `ShellLadder` is
complete but grows fast; `WedgeLadder(max_level, ell_max)` is level-ordered but
angularly capped, and much cheaper per rung. There is **no default** — the
policy is required, because the right order depends on your operator
([`examples/mode_policies.py`](../examples/mode_policies.py) shows them
disagreeing).

Policies are built-ins only in Python: `fit_operator` releases the GIL and
calls `propose` from worker threads, so a Python subclass would serialize the
fit.

## Errors

`std::invalid_argument` (Python `ValueError`) means you passed something
wrong. `InfeasibleParameters` means "no basis exists at this point in parameter
space" — the fitting core catches it internally and scores that candidate as
the worst finite cost, so you should only see it if you call the core yourself.
