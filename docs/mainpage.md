# lgpsf API reference {#mainpage}

**Compress a large, dense, matrix-free operator into a sparse one, using only
its action on random vectors.** Header-only C++17, depending on Eigen and
[ellipsoid_tree](https://github.com/NickAlger/ellipsoid_tree).

This site is the **API reference**, generated from the headers. For the project
overview, installation, worked examples and the evidence behind the defaults,
see the main documentation:

- [Project README](https://github.com/NickAlger/lgpsf#readme) — what it is for, and when not to use it
- [Quickstart](https://github.com/NickAlger/lgpsf/blob/main/docs/quickstart.md) — what you must supply, and how to check the fit worked
- [API guide](https://github.com/NickAlger/lgpsf/blob/main/docs/api-guide.md) — the three layers and the array conventions
- [Defaults](https://github.com/NickAlger/lgpsf/blob/main/docs/defaults.md) — every knob and the measurement behind it
- [Examples](https://github.com/NickAlger/lgpsf/tree/main/examples) — fifteen, each teaching one thing

## Where to start

Include the umbrella header `lgpsf/lgpsf.hpp`, or just the layer you need.

**Fitting a whole operator** — `lgpsf::fit_operator`, configured by
`lgpsf::OperatorFitConfig`, returning `lgpsf::OperatorFit`.

**Using the result** — `lgpsf::LGOperator` and its free functions:
`lgpsf::matvec`, `lgpsf::assemble_sparse`, `lgpsf::eval_entries`,
`lgpsf::eval_kernel`, `lgpsf::qc_map`, `lgpsf::ellipsoid_field`. This header
depends on none of the fitting code, so an operator can also be built directly
from a physics model with `lgpsf::build_operator`.

**Fitting one target** — `lgpsf::fit_from_probes`, for a single point-spread
function or anything else known through inner products with random fields.

**The fitting core** — `lgpsf::fit_varpro`, a general separable least-squares
solver that knows nothing about point-spread functions.

**The basis** — `lgpsf::Mode`, `lgpsf::LGBasisAt`, `lgpsf::modes_up_to_level`,
and the ellipsoid pullback in `lgpsf::EllipsoidFrame`.

## Two conventions worth knowing before you read further

**Points are `(K, N)` here — points as ROWS.** The Python bindings take
`(N, K)`; the two are the same bytes, which is what makes that boundary
zero-copy. Per-row records are `(R, …)`, row first.

**`theta` has two encodings.** The public one is absolute,
`[mu, log-diag(L), strict-lower(L)]`, and decodes with `lgpsf::unpack_theta`
alone. The internal `theta_hat` that the fitting core searches omits the center
when it is pinned and stores it as a displacement when it is not. Anything read
off a fitted operator is the public one.

Symbols under `lgpsf::detail` are implementation and are excluded here.
