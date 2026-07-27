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
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lgpsf/ellipsoid_transform.hpp"
#include "lgpsf/harmonic_polynomials.hpp"
#include "lgpsf/lg_functions.hpp"
#include "lgpsf/lgpsf.hpp"
#include "lgpsf/whitening.hpp"

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
///
/// This generalizes past coordinates: ANY batched array is `(X, K)` in numpy
/// and `(K, X)` in Eigen, same bytes, so probe blocks `(num_probes, K)`, extra
/// bases `(num_extra, K)` and basis values `(num_modes, K)` all come through
/// here. Square non-batch matrices -- a Cholesky factor, a covariance -- do NOT:
/// they go through pybind11's Eigen caster, which handles strides, and where a
/// silent transpose would turn lower-triangular into upper.
Eigen::Map<const Eigen::MatrixXd> map_batch( const PointsIn& array,
                                             const char* name )
{
    if ( array.ndim() != 2 )
    {
        throw std::invalid_argument(std::string("lgpsf: ") + name
                                    + " must be 2-D, (X, K) with the batch "
                                      "(point) axis LAST; got ndim "
                                    + std::to_string(array.ndim()));
    }
    const Eigen::Index other = array.shape(0);
    const Eigen::Index count = array.shape(1);
    return Eigen::Map<const Eigen::MatrixXd>(array.data(), count, other);
}

/// Alias that reads better where the batch really is points.
Eigen::Map<const Eigen::MatrixXd> map_points( const PointsIn& points,
                                              const char* name )
{
    return map_batch(points, name);
}

/// The return direction: an Eigen `(K, X)` becomes a numpy `(X, K)`.
///
/// A plain memcpy, not a strided transpose -- column-major `(K, X)` and C-order
/// `(X, K)` are byte-for-byte the same, which is the whole point of the
/// convention.
py::array_t<double> batch_last( const Eigen::MatrixXd& m )
{
    py::array_t<double> out({m.cols(), m.rows()});
    std::memcpy(out.mutable_data(), m.data(),
                sizeof(double) * static_cast<std::size_t>(m.size()));
    return out;
}

/// A stack of per-mode `(K, N)` gradients as one `(num_modes, N, K)` array.
py::array_t<double> stack_batch_last( const std::vector<Eigen::MatrixXd>& mats )
{
    const py::ssize_t count = static_cast<py::ssize_t>(mats.size());
    const py::ssize_t other = count ? mats.front().cols() : 0;
    const py::ssize_t batch = count ? mats.front().rows() : 0;
    py::array_t<double> out({count, other, batch});
    for ( py::ssize_t i = 0; i < count; ++i )
    {
        std::memcpy(out.mutable_data(i, 0, 0), mats[static_cast<std::size_t>(i)].data(),
                    sizeof(double) * static_cast<std::size_t>(other * batch));
    }
    return out;
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

    // ---- Laguerre-Gaussian basis ------------------------------------------
    // Point batches are (N, K) in, and every batched result comes back with the
    // batch axis LAST: values (num_modes, K), gradients (num_modes, N, K).

    m.def("genlaguerre",
          []( int p, double alpha, const Eigen::VectorXd& x )
          { return genlaguerre(p, alpha, x); },
          "p"_a, "alpha"_a, "x"_a,
          "Generalized Laguerre polynomial L_p^alpha at each x, by the "
          "three-term recurrence. Plain 1-D array in, 1-D out.");

    m.def("lg_norm", &lg_norm, "p"_a, "ell"_a, "dim"_a,
          "The L2(R^dim) normalization constant combining the harmonic, "
          "radial and Gaussian factors of one mode.");

    m.def("eval_lg_basis",
          []( const std::vector<Mode>& modes, const PointsIn& u )
          { return batch_last(eval_lg_basis(modes, map_points(u, "u"))); },
          "modes"_a, "u"_a,
          "Values of a whole mode set at points u (N, K). Returns "
          "(num_modes, K). This is the production path: the mode index "
          "factorizes, so the set is the natural unit.");

    m.def("grad_lg_basis",
          []( const std::vector<Mode>& modes, const PointsIn& u )
          { return stack_batch_last(grad_lg_basis(modes, map_points(u, "u"))); },
          "modes"_a, "u"_a,
          "Spatial gradients of a whole mode set at points u (N, K), "
          "UNCONTRACTED: returns (num_modes, N, K).");

    m.def("vjp_lg_basis",
          []( const std::vector<Mode>& modes, const PointsIn& u, const PointsIn& w )
          { return batch_last(vjp_lg_basis(modes, map_points(u, "u"),
                                           map_batch(w, "w"))); },
          "modes"_a, "u"_a, "w"_a,
          "grad_u sum_i w_i psi_i(u) for cotangent w of shape (num_modes, K); "
          "returns (N, K). Regrouped per shell, so the (num_modes, N, K) "
          "tensor is never built.");

    m.def("eval_lg_nd",
          []( int p, int ell, int mm, const PointsIn& u )
          { return Eigen::VectorXd(eval_lg_nd(p, ell, mm, map_points(u, "u"))); },
          "p"_a, "ell"_a, "m"_a, "u"_a,
          "One mode at every point: (K,). The readable reference the batched "
          "path is pinned against.");

    m.def("grad_eval_lg_nd",
          []( int p, int ell, int mm, const PointsIn& u )
          { return batch_last(grad_eval_lg_nd(p, ell, mm, map_points(u, "u"))); },
          "p"_a, "ell"_a, "m"_a, "u"_a,
          "Spatial gradient of one mode at every point: (N, K).");

    m.def("eval_lg",
          []( int p, int ell, const Eigen::VectorXd& u1, const Eigen::VectorXd& u2 )
          { return eval_lg(p, ell, u1, u2); },
          "p"_a, "ell"_a, "u1"_a, "u2"_a,
          "The 2-D reference convention: sign of ell picks cos (ell > 0) or "
          "sin (ell < 0). Kept as the independent definition of the N = 2 "
          "case; the N-dimensional path indexes the same space by (ell >= 0, m) "
          "with no canonical cos/sin labelling.");

    // ---- harmonic polynomials ---------------------------------------------

    m.def("num_harmonics", &num_harmonics, "dim"_a, "degree"_a,
          "dim of the space of degree-`degree` harmonic polynomials on R^dim.");

    m.def("harmonic_terms",
          []( int dim, int degree, int mm ) {
              const HarmonicPolynomial poly = harmonic_terms(dim, degree, mm);
              // A TERM TABLE, not a point batch: one row per term reads
              // naturally and the batch-last convention does not apply.
              py::array_t<double> coefficients(poly.num_terms);
              py::array_t<int> exponents({poly.num_terms, poly.dim});
              for ( int t = 0; t < poly.num_terms; ++t )
              {
                  coefficients.mutable_at(t) = poly.coefficients[t];
                  for ( int k = 0; k < poly.dim; ++k )
                  {
                      exponents.mutable_at(t, k) = poly.exponents[t * poly.dim + k];
                  }
              }
              return py::dict("dim"_a = poly.dim, "degree"_a = poly.degree,
                              "coefficients"_a = coefficients,
                              "exponents"_a = exponents);
          },
          "dim"_a, "degree"_a, "m"_a,
          "The NONZERO terms of Y_{ell,m} on R^dim: coefficients (num_terms,) "
          "and exponents (num_terms, dim). The table stores only nonzeros -- "
          "91.5% of the dense coefficients vanish for structural reasons.");

    m.def("eval_harmonic",
          []( int degree, int mm, const PointsIn& u )
          { return Eigen::VectorXd(eval_harmonic(degree, mm, map_points(u, "u"))); },
          "degree"_a, "m"_a, "u"_a,
          "Y_{ell,m} at points u (N, K); the dimension comes from u. "
          "Returns (K,).");

    m.def("grad_harmonic",
          []( int degree, int mm, const PointsIn& u ) {
              std::pair<Eigen::VectorXd, Eigen::MatrixXd> both =
                  grad_harmonic(degree, mm, map_points(u, "u"));
              return py::make_tuple(both.first, batch_last(both.second));
          },
          "degree"_a, "m"_a, "u"_a,
          "Y_{ell,m} and its spatial gradient from one pass: "
          "((K,), (N, K)).");

    // ---- the ellipsoid pullback -------------------------------------------

    py::enum_<MuMode>(m, "MuMode",
                      "Whether a theta_hat carries the center as a fitted "
                      "displacement from mu0, or pins it there.")
        .value("Pinned", MuMode::Pinned)
        .value("Fitted", MuMode::Fitted);

    m.def("theta_size", &theta_size, "dim"_a,
          "Length of the PUBLIC absolute theta: [mu, log-diag, strict-lower], "
          "always dim(dim+3)/2, and self-decoding.");
    m.def("theta_hat_size", &theta_hat_size, "dim"_a, "mode"_a,
          "Length of the INTERNAL theta_hat, which omits the center when it "
          "is pinned.");
    m.def("dim_from_theta_size", &dim_from_theta_size, "size"_a);

    py::class_<EllipsoidFrame>(m, "EllipsoidFrame",
                               "A decoded ellipsoid: center mu and Cholesky "
                               "factor L, with Sigma = L L^T.")
        .def_readonly("mu", &EllipsoidFrame::mu)
        .def_readonly("L", &EllipsoidFrame::L)
        .def_readonly("L_inv", &EllipsoidFrame::L_inv)
        .def_property_readonly("dim", &EllipsoidFrame::dim)
        .def("__repr__", []( const EllipsoidFrame& f ) {
                 std::ostringstream out;
                 out << "EllipsoidFrame(dim=" << f.dim() << ")";
                 return out.str();
             });

    m.def("make_frame",
          []( Eigen::VectorXd mu, Eigen::MatrixXd L )
          { return make_frame(std::move(mu), std::move(L)); },
          "mu"_a, "L"_a,
          "Build a frame from an explicit center and Cholesky factor. Raises "
          "InfeasibleParameters if L's diagonal is not usable.");

    m.def("unpack_theta",
          []( const Eigen::VectorXd& theta ) { return unpack_theta(theta); },
          "theta"_a,
          "Decode the public absolute theta. Needs nothing else -- that is "
          "what makes a fitted operator self-describing.");
    m.def("unpack_theta_hat",
          []( const Eigen::VectorXd& theta_hat, const Eigen::VectorXd& mu0,
              MuMode mode ) { return unpack_theta_hat(theta_hat, mu0, mode); },
          "theta_hat"_a, "mu0"_a, "mode"_a);
    m.def("to_theta",
          []( const Eigen::VectorXd& theta_hat, const Eigen::VectorXd& mu0,
              MuMode mode ) { return to_theta(theta_hat, mu0, mode); },
          "theta_hat"_a, "mu0"_a, "mode"_a,
          "Internal encoding -> public absolute encoding.");
    m.def("to_theta_hat",
          []( const Eigen::VectorXd& theta, const Eigen::VectorXd& mu0,
              MuMode mode ) { return to_theta_hat(theta, mu0, mode); },
          "theta"_a, "mu0"_a, "mode"_a,
          "Public absolute encoding -> internal encoding.");
    m.def("release_mu",
          []( const Eigen::VectorXd& theta_hat, int dim )
          { return release_mu(theta_hat, dim); },
          "theta_hat"_a, "dim"_a,
          "A pinned theta_hat as a fitted one, by prepending a zero "
          "displacement.");
    m.def("freeze_mu",
          []( const Eigen::VectorXd& theta ) {
              std::pair<Eigen::VectorXd, Eigen::VectorXd> both = freeze_mu(theta);
              return py::make_tuple(both.first, both.second);
          },
          "theta"_a,
          "Split a public theta into (its L block, its center).");

    m.def("pullback",
          []( const EllipsoidFrame& frame, const PointsIn& x )
          { return batch_last(pullback(frame, map_points(x, "x"))); },
          "frame"_a, "x"_a,
          "T(theta, x) = L^-1 (x - mu): physical points (N, K) into the LG "
          "basis's round coordinates, (N, K) out. Note T is the PULLBACK, not "
          "a forward map.");

    // ---- whitening ---------------------------------------------------------
    // The only place the mass matrices appear. Every array here is batch-last
    // like the rest: probes (num_probes, K), extra basis (num_extra, K).

    m.def("whitening_scale",
          []( double target_mass, const Eigen::VectorXd& m2_diag )
          { return detail::whitening_scale(target_mass, m2_diag); },
          "target_mass"_a, "m2_diag"_a,
          "sqrt(target_mass) * sqrt(m2_diag): the scaling applied to the "
          "smooth basis.");
    m.def("whiten_probes",
          []( const PointsIn& z, const Eigen::VectorXd& m2_diag )
          { return batch_last(whiten_probes(map_batch(z, "z"), m2_diag)); },
          "z"_a, "m2_diag"_a,
          "z_hat = M2^(1/2) z for probe fields of shape (num_probes, K).");
    m.def("whiten_data",
          []( const Eigen::VectorXd& y, double target_mass )
          { return whiten_data(y, target_mass); },
          "y"_a, "target_mass"_a,
          "y_hat = y / sqrt(target_mass) for the target's raw responses (k,).");
    m.def("whiten_extra",
          []( const PointsIn& E, double target_mass, const Eigen::VectorXd& m2_diag )
          { return batch_last(whiten_extra(map_batch(E, "E"), target_mass, m2_diag)); },
          "E"_a, "target_mass"_a, "m2_diag"_a,
          "E_hat = sqrt(target_mass) M2^(-1/2) E for an extra basis of shape "
          "(num_extra, K). The INVERSE power of M2, unlike the smooth basis: "
          "extra functions are discrete corrections, not quadrature objects.");

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
