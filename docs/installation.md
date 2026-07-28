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
cmake --build build -j3
ctest --test-dir build
```

Options: `LGPSF_BUILD_TESTS`, `LGPSF_BUILD_EXAMPLES` (both ON when standalone),
`LGPSF_BUILD_PYTHON`, `LGPSF_BUILD_EXPERIMENTS` (both OFF).

Two warnings worth heeding:

**Build with optimization.** Every translation unit goes through Eigen's
expression templates, and a Debug build is not marginally slower but ~33×
slower — enough to make an example look hung. If a run seems to have stalled,
check `CMAKE_BUILD_TYPE` first.

**Do not use a bare `-j`.** Each compile job can peak above a gigabyte. `-j3`
is a reasonable default and `-j2` for sanitizer builds.

## Python bindings from source

```
cmake -S . -B build-py -DLGPSF_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-py --target lgpsf_python -j3
PYTHONPATH=build-py/bindings python -c "import lgpsf; print(lgpsf.__version__)"
```
