// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Python bindings for lgpsf.
///
/// ## The array convention, and why it is not ellipsoid_tree's
///
/// **Point batches are `(N, K)` here: coordinates down, points across.**
///
/// Two reasons, neither of them deference to the frozen prototype. First, it
/// costs nothing: a C-contiguous numpy `(N, K)` array and a column-major Eigen
/// `(K, N)` matrix ARE THE SAME BYTES, so `map_points` builds an `Eigen::Map`
/// straight onto the caller's buffer with no copy. At field scale that matters
/// -- a full-Antarctica probe block is tens of gigabytes, and a marshalling
/// copy of it is not a rounding error. Second, the live research scripts that
/// consume this library already speak `(N, K)`, so porting them is a change of
/// import rather than a rewrite.
///
/// Note this DIFFERS from `ellipsoid_tree`'s Python bindings, which take points
/// as ROWS, `(m, d)`, and transpose into their own storage. Someone using both
/// libraries meets opposite conventions; that is a deliberate trade of
/// cross-library uniformity for zero-copy plus prototype compatibility, and it
/// is why every point-taking function below says its shape in its docstring.

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lgpsf/lg_functions.hpp"
#include "lgpsf/lgpsf.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace lgpsf;

namespace {

/// A point batch as the caller passed it: `(N, K)`, C-contiguous.
///
/// `forcecast` keeps the ergonomics honest -- a non-contiguous slice or an
/// integer array is converted rather than rejected -- while a clean float64
/// C-contiguous array, which is what any caller doing this at scale will have,
/// passes through untouched and gets mapped with no copy.
using PointsIn = py::array_t<double, py::array::c_style | py::array::forcecast>;

/// Zero-copy view of an `(N, K)` numpy array as the `(K, N)` Eigen matrix the
/// C++ API takes. The returned Map borrows `points`, so it must not outlive it.
Eigen::Map<const Eigen::MatrixXd> map_points( const PointsIn& points,
                                              const char* name )
{
    if ( points.ndim() != 2 )
    {
        throw std::invalid_argument(std::string("lgpsf: ") + name
                                    + " must be 2-D, (N, K) with coordinates "
                                      "down and points across; got ndim "
                                    + std::to_string(points.ndim()));
    }
    const Eigen::Index dim = points.shape(0);
    const Eigen::Index count = points.shape(1);
    return Eigen::Map<const Eigen::MatrixXd>(points.data(), count, dim);
}

} // end anonymous namespace

PYBIND11_MODULE(lgpsf, m)
{
    m.doc() = "Laguerre-Gaussian PSF operator approximation with VarPro "
              "ellipsoid fitting.\n\n"
              "Point batches are (N, K): coordinates down, points across -- the "
              "layout that maps onto the C++ core with no copy. Note this is "
              "the transpose of ellipsoid_tree's Python convention.";
    m.attr("__version__") = LGPSF_VERSION;

    py::class_<Mode>(m, "Mode",
                     "One Laguerre-Gaussian mode index: radial p, angular "
                     "degree ell, and the harmonic index m within that shell.")
        .def(py::init([]( int p, int ell, int mm ) { return Mode{p, ell, mm}; }),
             "p"_a, "ell"_a, "m"_a)
        .def_readonly("p", &Mode::p)
        .def_readonly("ell", &Mode::ell)
        .def_readonly("m", &Mode::m)
        .def("__eq__", []( const Mode& a, const Mode& b ) { return a == b; },
             py::is_operator())
        .def("__hash__", []( const Mode& a )
             { return py::hash(py::make_tuple(a.p, a.ell, a.m)); })
        .def("__len__", []( const Mode& ) { return 3; })
        .def("__getitem__", []( const Mode& a, int i ) {
                 if ( i == 0 || i == -3 ) { return a.p; }
                 if ( i == 1 || i == -2 ) { return a.ell; }
                 if ( i == 2 || i == -1 ) { return a.m; }
                 throw py::index_error("Mode index out of range");
             })
        .def("__repr__", []( const Mode& a ) {
                 std::ostringstream out;
                 out << "Mode(p=" << a.p << ", ell=" << a.ell << ", m=" << a.m << ")";
                 return out.str();
             });

    m.def("modes_up_to_level",
          []( int dim, int max_level, int ell_max )
          { return modes_up_to_level(dim, max_level, ell_max); },
          "dim"_a, "max_level"_a, "ell_max"_a = -1,
          "Every mode with oscillator level <= max_level, optionally capped at "
          "angular degree ell_max (-1 = no cap). Raises past the generated "
          "harmonic table rather than truncating.");

    m.def("max_oscillator_level", &max_degree,
          "Highest oscillator level the generated harmonic table supports.");
    m.def("max_dimension", &max_dimension,
          "Highest spatial dimension the generated harmonic table supports.");

    // ---- layout probe -----------------------------------------------------
    // Not part of the API; a permanent regression hook for the one bug this
    // boundary invites. A transposed read is silent -- it produces a plausible
    // array of the wrong thing -- so the check returns quantities of DIFFERENT
    // LENGTHS (N and K), which a transposed read cannot fake.
    m.def("_layout_probe",
          []( const PointsIn& points ) {
              const Eigen::Map<const Eigen::MatrixXd> x = map_points(points, "points");
              return py::dict(
                  "num_points"_a = x.rows(), "dim"_a = x.cols(),
                  "centroid"_a = Eigen::VectorXd(x.colwise().mean()),
                  "norms"_a = Eigen::VectorXd(x.rowwise().norm()),
                  "first_point"_a = Eigen::VectorXd(x.row(0).transpose()),
                  // the address C++ actually read from -- equal to the numpy
                  // array's own buffer exactly when no copy was made
                  "data_ptr"_a = reinterpret_cast<std::uintptr_t>(x.data()));
          },
          "points"_a,
          "Internal: read an (N, K) batch through the C++ mapping and report "
          "what the C++ side saw. Used by the test suite to pin the layout.");
}
