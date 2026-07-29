# Quickstart

## What you need before you start

lgpsf fits an operator it never sees. You supply:

| | | |
|---|---|---|
| `x_cols` | `(N, K)` | where the columns live — coordinates down, points across |
| `m2_diag` | `(K,)` | their quadrature weights (lumped mass) |
| `m1_diag` | `(R,)` | the same for rows |
| `V` | `(k, K)` | random probe vectors |
| `HV` | `(k, R)` | `H @ V.T` — **the only thing your operator is asked for** |
| `sigma` | `(R, N, N)` | a rough guess at each row's shape |

`k` is the number of operator applications and therefore your entire cost.
Everything else is bookkeeping.

**`sigma` is not optional and is not fitted from nothing.** It is the a-priori
ellipsoid field: your best guess at how wide each point-spread function is and
which way it points. It sets the search window and the first initial guess --
the fit adds circles of its own -- and then refines it. Being wrong by a factor of two is fine; being wrong by a factor
of ten is not. On a PDE problem this usually comes from a length scale you
already know.

## The fit

```python
import lgpsf

config = lgpsf.OperatorFitConfig()
config.row.mode_policy = lgpsf.WedgeLadder(10, 2)   # required, no default

fit = lgpsf.fit_operator(x_cols, m1_diag, m2_diag, V, HV, sigma, config=config)
```

That returns `fit.model` (the operator) and `fit.diagnostics` (per-row
provenance). The fit is threaded over rows and deterministic.

## Using it

```python
y = lgpsf.matvec(fit.model, v)                     # apply, assembling nothing
A = lgpsf.assemble_sparse(fit.model, tau=6.0)      # a scipy sparse matrix
e = lgpsf.eval_entries(fit.model, rows, cols)      # specific entries
k = lgpsf.eval_kernel(fit.model, rows, x_query)    # the kernel, at any points
```

`tau` truncates each row at that many standard deviations of its fitted
ellipsoid — smaller is sparser and less accurate. Every one of these also
restricts to the row's **fit window**, the region the fit was actually measured
on; see [defaults.md](defaults.md).

## Checking it worked

You will not have a truth to compare against. Use held-out probes:

```python
qc = lgpsf.qc_map(fit.model, V_qc, HV_qc)          # per-row relative residual
status = fit.diagnostics.status                    # Fit | FallbackBaseline | ...
```

A row that shipped `FallbackBaseline` means the search never beat the a-priori
model, so the prior shipped instead — a fit is never worse than the `sigma` you
supplied.

## How many probes?

A mode set of size `m` needs `k ≥ 2(m + P)` probes, where `P` is the number of
ellipsoid parameters being fitted (3 in 2-D with the center pinned). Below that
the coefficients alone can drive the residual to zero and the ellipsoid is
decided by noise. The library enforces this and will refuse over-large mode
sets rather than fit them.

In practice: start around 40–60, look at `qc_map`, and add probes if the tail
is bad. Refining the mesh does **not** require more.

## Where to go next

- [`examples/fit_one_psf.py`](../examples/fit_one_psf.py) — the same thing for
  a single target, which is the clearest place to see the data model.
- [`examples/operator_fit_frog.py`](../examples/operator_fit_frog.py) — this
  quickstart, complete and runnable, with error against a known truth.
- [`examples/preconditioner.py`](../examples/preconditioner.py) — what to do
  with a fit once you have one.
- [`examples/counting_rule.py`](../examples/counting_rule.py) — what
  over-fitting looks like, so you recognize it.
- [api-guide.md](api-guide.md) — the layers, and the array conventions.
