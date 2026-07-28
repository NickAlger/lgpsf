# Installation

## Python

```
pip install lgpsf
```

Needs numpy. `assemble_sparse` returns a `scipy.sparse` matrix, so install
scipy too if you want assembly:

```
pip install lgpsf[sparse]
```

Wheels are built for CPython 3.9–3.13 on Linux and macOS. On other platforms
pip builds from source, which needs a C++17 compiler and CMake ≥ 3.16 —
everything else is fetched automatically.

## C++

Header-only. It needs a C++17 compiler, [Eigen](https://eigen.tuxfamily.org),
and [ellipsoid_tree](https://github.com/NickAlger/ellipsoid_tree), which
supplies the spatial queries and brings Eigen with it.

**FetchContent**, if you have neither dependency installed:

```cmake
include(FetchContent)
FetchContent_Declare(lgpsf
    GIT_REPOSITORY https://github.com/NickAlger/lgpsf.git
    GIT_TAG        v0.1.0)
FetchContent_MakeAvailable(lgpsf)

target_link_libraries(your_target PRIVATE lgpsf::lgpsf)
```

`ellipsoid_tree` is pulled in the same way if CMake cannot find an installed
copy.

**find_package**, against an installed copy:

```cmake
find_package(lgpsf 0.1 REQUIRED)
target_link_libraries(your_target PRIVATE lgpsf::lgpsf)
```

Then include what you need — the headers are layered, so you can take the
operator layer without the fitter, or the basis without either:

```cpp
#include "lgpsf/operator_fit.hpp"   // fitting a whole operator
#include "lgpsf/lg_operator.hpp"    // just the fitted object and its helpers
#include "lgpsf/probe_fit.hpp"      // one target
#include "lgpsf/lg_functions.hpp"   // the basis alone
```

## Building this repository

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2      # see the memory note below
ctest --test-dir build
```

Options: `LGPSF_BUILD_TESTS`, `LGPSF_BUILD_EXAMPLES` (both ON when standalone),
`LGPSF_BUILD_PYTHON`, `LGPSF_BUILD_EXPERIMENTS` (both OFF).

### ⚠️ Compiling needs a lot of memory

**Choose `-j` by available RAM, not by core count.** Every translation unit
instantiates Eigen expression templates heavily, and the compiler's peak
resident set scales with that rather than with the size of the source file.
Measured here on a Release build with GCC:

| translation unit | build | peak RSS |
|---|---|---|
| `test_operator_fit.cpp` | Release | **2.8 GB** |
| `examples/operator_fit_frog.cpp` | Release | 3.0 GB |
| `test_varpro.cpp` | Release | 1.9 GB |
| `test_lg_functions.cpp` | Release | 0.8 GB |
| `test_operator_fit.cpp` | **ASan+UBSan** | **7.9 GB** |

So building with one job per core is actively dangerous. On a 16-core machine
that is up to ~48 GB of compiler for an ordinary build.

**Budget against FREE memory, not total, and leave room for whatever else you
are running.** A workstation with an editor and a browser open has several
gigabytes already spoken for, and the compiler will take what is left. A
workable rule is

    jobs = (available GB - 4) / 3

capped at the core count. On a 13 GB machine with a desktop session that comes
out at 1 or 2 — not the 4 a naive total-memory rule would suggest.

**The failure mode arrives before the OOM killer does.** Long before anything
is killed, the machine starts swapping and the desktop becomes unresponsive
while the build grinds on. If your machine locks up mid-build, that is this,
and the fix is fewer jobs.

**Sanitizer builds need their own budget: ~8 GB per job.** Use one job unless
you have a lot of room. Continuous integration can be more aggressive than a
workstation, because the runner is dedicated and idle — this repository's CI
uses 3 jobs for ordinary builds and 1 for sanitizers.

Building specific targets rather than everything also helps:

```sh
cmake --build build --target lgpsf_tests -j2
```

### ⚠️ Build with optimization

A Debug build is not marginally slower but **~33× slower** — enough to make an
example look hung. The same whole-operator example run takes 79 seconds at
`-O3` and 43 minutes unoptimized. If something seems stalled, check
`CMAKE_BUILD_TYPE` before anything else.

## Python bindings from source

```
cmake -S . -B build-py -DLGPSF_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-py --target lgpsf_python -j2
PYTHONPATH=build-py/bindings python -c "import lgpsf; print(lgpsf.__version__)"
```
