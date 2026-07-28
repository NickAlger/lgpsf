# Contributing to lgpsf

Bug reports, fixes and focused improvements are welcome. This is a header-only
C++17 library (Eigen and [ellipsoid_tree](https://github.com/NickAlger/ellipsoid_tree))
with Python bindings.

If you are proposing a change to the *method* rather than the code — a new mode
policy, a different selection rule, another window shape — open an issue first.
Those decisions are usually settled by measurement here, and there is probably
evidence one way or the other already in [`docs/validation.md`](docs/validation.md)
or [`experiments/`](experiments/).

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2      # see the memory note below
ctest --test-dir build --output-on-failure
```

Two things that will otherwise cost you an afternoon — or your session:

- **Choose the job count by FREE memory, not by cores.** Eigen template
  instantiation dominates compiler memory: the heaviest translation unit peaks
  at **2.8 GB** in a Release build and **7.9 GB** under ASan+UBSan. Budget
  `(available GB - 4) / 3`, leaving room for whatever else you are running,
  and a single job for sanitizers. On a 13 GB workstation with an editor open
  that means 2. The failure mode arrives before the OOM killer: the machine
  starts swapping and the desktop locks up while the build carries on. Prefer
  building a specific target over everything.
- **Build with optimization.** A Debug build is not marginally slower but ~33×
  slower — enough to make an example look hung. If something seems stalled,
  check `CMAKE_BUILD_TYPE` first.

Tests use [doctest](https://github.com/doctest/doctest) (vendored), one file per
header, in `tests/`.

### Sanitizers

CI runs ASan+UBSan and TSan. To reproduce locally — note the `-j1`, since a
single sanitizer translation unit here peaks near 8 GB:

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLGPSF_BUILD_EXAMPLES=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j1
ctest --test-dir build-asan --output-on-failure
```

### Python bindings

```sh
cmake -S . -B build-py -DLGPSF_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-py --target lgpsf_python -j2
PYTHONPATH=build-py/bindings python3 -m pytest bindings/tests
```

## Testing rules specific to this library

**Every JVP/VJP pair gets two kinds of test**: finite differences *and*
adjoint consistency, `sum(w * jvp(v)) == sum(vjp(w) * v)` for random `w`, `v`.
Not either — both. The second has repeatedly caught sign and transpose bugs
that finite differences passed, because an FD check compares a derivative
against itself in the wrong basis and cannot see a consistent flip.

**Bit-identity is the acceptance criterion for any change meant to be a pure
restructuring.** If you rewrite an evaluation path for speed, the answers
should be identical to the last bit, and a test should say so. This caught a
1-ULP regression that `allclose` would have accepted. Use a tolerance only
where the math genuinely reassociates — and say in the comment that it does,
and why.

**Nothing compares against a stored reference.** The suites are self-contained
so they cannot drift out of step with the code. Please keep it that way: a
golden-file test that nobody can regenerate is a test that eventually gets
deleted rather than fixed.

## Invariants a change can silently break

- **`lg_operator.hpp` depends on none of the fitting stack.** An operator can
  be built from a physics model and deployed without `operator_fit.hpp` ever
  being included. A test asserts this; do not "simplify" it away.
- **Mass matrices appear in exactly one layer**, `whitening.hpp`. Everything
  above it works in whitened coordinates and everything below is mass-free.
- **Deployed support is the fit window.** Every evaluation truncates to the
  region the fit was scored on, because out-of-window model mass is
  unpenalized rather than merely unverified, and LG modes extrapolate
  violently. `eval_kernel_unrestricted` is the named opt-out.
- **lgpsf may not depend on any private problem.** Naming one in prose is
  fine — `docs/validation.md` explains what the glaciology data is — but a
  path, an import, or a filename nobody can open is not.
  `python3 tools/check_dependencies.py` enforces this and runs in CI.

## Where things go

| | |
|---|---|
| `docs/` | For people USING the library. |
| `dev/` | For people CHANGING it — notes, decisions, the architecture map. |
| `experiments/` | Measurements *about* the library. Built in CI, never run. |
| `examples/` | One idea each, printing its result, ending on an assertion. |
| `archive/` | Superseded code kept for provenance. Not built, not packaged. |

Work that merely *uses* lgpsf to study something else does not belong in this
repository at all.

## Documentation

Public API prose lives in the headers as Doxygen `///` comments and is
published to GitHub Pages. Prefer stating what a function does, its shapes and
units, and what it throws. Design history belongs in `dev/`, not in a header —
a reader should not need to know what the code used to be.

## Pull requests

- Branch from `main`, keep each PR focused, and match the surrounding style.
- Add a test for any behavior you change or add.
- All CI must be green: g++ and clang++, the sanitizers, the Python bindings,
  the dependency check, and version consistency.

## Versioning

The version is single-sourced in `include/lgpsf/lgpsf.hpp` (the
`LGPSF_VERSION_*` macros). `CMakeLists.txt` parses it and a CI check keeps
`pyproject.toml` and `CITATION.cff` in sync — bump all three together.
