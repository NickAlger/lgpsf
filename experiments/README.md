# `experiments/` — measurements about the library

Benchmarks and their write-ups: work whose subject is lgpsf itself. Not part
of the build, not in the sdist, not run by CI.

Work that merely *uses* lgpsf to study something else does not belong here.
The glaciology validation described in [`docs/validation.md`](../docs/validation.md)
is that kind of work, and its harness lives with the science rather than with
the library — lgpsf never depends on a private problem.

| | |
|---|---|
| [`mesh_scalability.cpp`](mesh_scalability.cpp) | **Does refinement cost more probes?** The claim the method rests on is that a point-spread function belongs to the continuum operator, not the mesh — so `k` should be O(1) under refinement. Sweeps grid × probe budget, and reports the compression and the CG iteration counts that follow. Results in [`mesh-scalability.md`](mesh-scalability.md). |
| [`bench_solve.cpp`](bench_solve.cpp) | Five factorizations × four sizes, at the shapes the inner solve actually sees. Eigen only, no lgpsf headers, no inputs — the most portable thing here. |
| [`bench_field_scale.cpp`](bench_field_scale.cpp) | A whole-field fit on a seeded synthetic problem at validation scale (6561 columns, 100 probes). Self-contained; uses no private data. |
| [`inner-solve-profile.md`](inner-solve-profile.md) | Where the fit spends its time. The finding that motivated the QR-first inner solve: **SVD-bound, not basis-bound** — 61% of instructions at k = 100. Also records what was looked at and rejected. |
| [`cxx-vs-python-timing.md`](cxx-vs-python-timing.md) | The C++ against the Python prototype it replaced: 8.4× single-threaded, 25× on four cores. Historical — the Python half is now frozen in `archive/`. Carries one superseded conclusion, flagged in place. |

The C++ studies build off by default, since they are measurements rather than
tests and some run for minutes:

    cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
          -DLGPSF_BUILD_EXPERIMENTS=ON
    cmake --build build-release --target mesh_scalability -j2

Build with optimization or the numbers are meaningless — an unoptimized build
of this code is ~33× slower, not marginally slower. Timing runs additionally
want `-march=native`, which is *not* what the test suite is built with; see
[`docs/reproducibility.md`](../docs/reproducibility.md) for why that
distinction matters when comparing results. Never build with a bare `-j`.
