# Examples

Each one teaches a single thing and prints its result. Run them from the
repository root:

    PYTHONPATH=<build>/bindings python examples/lg_modes.py

They need the Python bindings (build with `-DLGPSF_BUILD_PYTHON=ON`), numpy,
and scipy for anything that assembles a sparse matrix. **matplotlib is
optional** — the numbers always print; figures are skipped without it.

## Start here

| | |
|---|---|
| [`lg_modes.py`](lg_modes.py) | What a Laguerre-Gaussian mode is, before it is a config knob. Angular order across, radial order down. |
| [`lg_expansion_convergence.py`](lg_expansion_convergence.py) | The premise: a short LG expansion approximates a smooth bump well. Also the cost of dimension — 11 modes at N=1 reach what 286 need at N=3. |

## Fitting one target

| | |
|---|---|
| [`fit_one_psf.py`](fit_one_psf.py) | `fit_from_probes`: recover one point-spread function from probe inner products. **The data model to understand first** — you never hand the fitter function values. |
| [`fit_one_psf.cpp`](fit_one_psf.cpp) | The same in C++, where points are `(K, N)` — rows, not columns — the policy is a `shared_ptr`, and optional inputs are `std::optional`. |
| [`counting_rule.py`](counting_rule.py) | Why `k >= 2(m + P)`, and what going past it looks like: in-sample cost collapses to 1e-21 while the held-out score gets *worse*. |
| [`reading_a_row_fit.py`](reading_a_row_fit.py) | The candidate table — what the search tried, what it kept, why it stopped. The first place to look when a row disappoints. |
| [`mode_policies.py`](mode_policies.py) | Shells vs wedges vs radial-first, head to head. No order wins everywhere, which is why it is a policy. |

## Fitting a whole operator

| | |
|---|---|
| [`operator_fit_frog.py`](operator_fit_frog.py) | The end-to-end example: a dense operator, random matvecs, and the error falling with the probe budget. Also the integration test (`--quick`). |
| [`operator_fit_frog.cpp`](operator_fit_frog.cpp) | The same pipeline in C++, on the same problem, with figures from `ellipsoid_tree`'s `plot2d` — no plotting dependency to install. |
| [`deploying_a_fit.py`](deploying_a_fit.py) | `matvec`, `assemble_sparse`, `eval_entries`, `eval_kernel`: four views of a fit, what truncation costs, and why symmetrizing can make things worse. |
| [`operator_diagnostics.py`](operator_diagnostics.py) | Reading a fit with no truth available: statuses, the baseline guard, `qc_map`, `spike_measure`, `ellipsoid_field`. |
| [`operator_without_fitting.py`](operator_without_fitting.py) | `LGOperator` is a data structure. Build one from a physics model, merge pieces with `concatenate_rows`, never touch the fitter. |
| [`rectangular_operator.py`](rectangular_operator.py) | Rows and columns on different meshes. Square is the special case. |
| [`preconditioner.py`](preconditioner.py) | **What a fit is for.** Solve `(H^T H + alpha I) x = H^T b` with the fit as a preconditioner: 19-34x fewer CG iterations across four decades of alpha, from one batch of probes. Uses scipy's CG and sparse factorization. |
| [`preconditioner.cpp`](preconditioner.cpp) | The same study in C++, with CG written out so the one line the preconditioner enters on is visible. |

## Underneath

| | |
|---|---|
| [`varpro_custom_basis.py`](varpro_custom_basis.py) | The fitting core alone: whitening, `WhitenedBasis`, `fit_varpro`, the two Jacobian variants and the ridge. |
| [`ellipsoid_theta.py`](ellipsoid_theta.py) | What `theta` is, the public/internal encodings, and the pullback that turns a fitted ellipsoid into the unit circle. |

## Shared

[`frog_kernel.py`](frog_kernel.py) and [`frog_kernel.hpp`](frog_kernel.hpp) are
the test problem, not a lesson: a Gaussian whose covariance rotates with
position, so a stationary convolution cannot represent it. The two are
entry-for-entry identical, so the C++ and Python examples fit the same
operator.

## Building the C++ examples

**Configure Release.** The same run takes 79 seconds at `-O3` and 43 minutes
unoptimized, because every row fit is dense linear algebra through Eigen
expression templates:

    cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
    cmake --build build-release --target operator_fit_frog -j3
    ./build-release/examples/operator_fit_frog

Never build with a bare `-j` — every translation unit pulls in Eigen.
