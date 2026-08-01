// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Python bindings for lgpsf.
///
/// ## The array convention, and why it is not ellipsoid_tree's
///
/// **Point batches are `(N, K)` here: coordinates down, points across.**
///
/// It costs nothing: a C-contiguous numpy `(N, K)` array and a column-major
/// Eigen `(K, N)` matrix ARE THE SAME BYTES, so `map_points` builds an
/// `Eigen::Map` straight onto the caller's buffer with no copy. At field scale
/// that matters -- a probe block for a continental mesh is tens of gigabytes,
/// and a marshalling copy of it is not a rounding error.
///
/// Note this DIFFERS from `ellipsoid_tree`'s Python bindings, which take points
/// as ROWS, `(m, d)`, and transpose into their own storage. Someone using both
/// libraries meets opposite conventions; that is a deliberate trade of
/// cross-library uniformity for zero copies, and it is why every point-taking
/// function below states its shape in its docstring.

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
#include "lgpsf/exceptions.hpp"
#include "lgpsf/harmonic_polynomials.hpp"
#include "lgpsf/lg_functions.hpp"
#include "lgpsf/lg_expansion.hpp"
#include "lgpsf/lg_operator.hpp"
#include "lgpsf/lgpsf.hpp"
#include "lgpsf/mode_policy.hpp"
#include "lgpsf/operator_fit.hpp"
#include "lgpsf/probe_fit.hpp"
#include "lgpsf/varpro.hpp"
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

/// A (R, N, N) numpy stack as the per-row covariances the C++ API takes.
///
/// Built elementwise rather than mapped. A covariance is SYMMETRIC, so a
/// transposed read of it is completely invisible -- there is no shape error and
/// no wrong answer until someone passes a non-symmetric matrix. Explicit
/// indexing costs nothing at R x 2 x 2 and removes the trap.
std::vector<Eigen::MatrixXd> sigma_from_array( const py::array_t<double>& sigma,
                                               Eigen::Index num_rows, int dim )
{
    if ( sigma.ndim() != 3 || sigma.shape(0) != num_rows || sigma.shape(1) != dim
         || sigma.shape(2) != dim )
    {
        throw std::invalid_argument(
            "lgpsf: sigma must have shape (num_rows, N, N) = ("
            + std::to_string(num_rows) + ", " + std::to_string(dim) + ", "
            + std::to_string(dim) + ")");
    }
    auto view = sigma.unchecked<3>();
    std::vector<Eigen::MatrixXd> out(static_cast<std::size_t>(num_rows));
    for ( Eigen::Index r = 0; r < num_rows; ++r )
    {
        Eigen::MatrixXd block(dim, dim);
        for ( int i = 0; i < dim; ++i )
        {
            for ( int j = 0; j < dim; ++j ) { block(i, j) = view(r, i, j); }
        }
        out[static_cast<std::size_t>(r)] = std::move(block);
    }
    return out;
}

/// A (R, N*N) row-major-per-row block as a (R, N, N) numpy stack.
py::array_t<double> unflatten_blocks( const Eigen::MatrixXd& flat, int dim )
{
    const py::ssize_t rows = flat.rows();
    py::array_t<double> out({rows, static_cast<py::ssize_t>(dim),
                             static_cast<py::ssize_t>(dim)});
    auto view = out.mutable_unchecked<3>();
    for ( py::ssize_t r = 0; r < rows; ++r )
    {
        for ( int i = 0; i < dim; ++i )
        {
            for ( int j = 0; j < dim; ++j ) { view(r, i, j) = flat(r, i * dim + j); }
        }
    }
    return out;
}

/// An enum-valued per-row diagnostic as an int8 array, so masking works.
template <typename Enum>
py::array_t<std::int8_t> codes_of( const std::vector<Enum>& values )
{
    py::array_t<std::int8_t> out(static_cast<py::ssize_t>(values.size()));
    for ( std::size_t i = 0; i < values.size(); ++i )
    {
        out.mutable_at(static_cast<py::ssize_t>(i)) =
            static_cast<std::int8_t>(values[i]);
    }
    return out;
}

/// `rows` as the C++ gate: a boolean mask OR an index array, both of which a
/// caller reaches for naturally, and neither of which the C++ takes.
std::vector<char> gate_from_rows( const py::object& rows, Eigen::Index num_rows )
{
    if ( rows.is_none() ) { return {}; }
    const py::array array = py::array::ensure(rows);
    if ( !array ) { throw std::invalid_argument("lgpsf: rows must be array-like"); }
    std::vector<char> gate(static_cast<std::size_t>(num_rows), 0);
    if ( py::isinstance<py::array_t<bool>>(array) )
    {
        const auto mask = array.cast<py::array_t<bool>>();
        if ( mask.size() != num_rows )
        {
            throw std::invalid_argument(
                "lgpsf: a boolean rows mask must have one entry per row");
        }
        for ( py::ssize_t i = 0; i < mask.size(); ++i )
        {
            gate[static_cast<std::size_t>(i)] = mask.at(i) ? 1 : 0;
        }
        return gate;
    }
    const auto indices = array.cast<py::array_t<py::ssize_t>>();
    for ( py::ssize_t i = 0; i < indices.size(); ++i )
    {
        const py::ssize_t rho = indices.at(i);
        if ( rho < 0 || rho >= num_rows )
        {
            throw std::invalid_argument("lgpsf: rows index out of range");
        }
        gate[static_cast<std::size_t>(rho)] = 1;
    }
    return gate;
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

    // ---- VarPro core -------------------------------------------------------

    py::register_exception<InfeasibleParameters>(m, "InfeasibleParameters",
                                                 PyExc_RuntimeError);

    py::enum_<JacobianVariant>(m, "JacobianVariant",
                               "Which Jacobian of the reduced residual the "
                               "outer loop is given. Both give the SAME exact "
                               "gradient; only the quadratic model differs.")
        .value("Kaufman", JacobianVariant::Kaufman,
               "One reverse sweep through the basis. The default.")
        .value("GolubPereyra", JacobianVariant::GolubPereyra,
               "The exact Jacobian; needs a basis offering jac().");

    py::class_<VarProOptions>(m, "VarProOptions",
                              "Numerics of one nonlinear fit. Nothing here "
                              "adjudicates a modelling question.")
        .def(py::init<>())
        .def_readwrite("ridge", &VarProOptions::ridge,
                       "Tikhonov ridge on the COEFFICIENTS, after per-column "
                       "equilibration. The returned residual is the plain "
                       "model misfit.")
        .def_readwrite("jacobian", &VarProOptions::jacobian)
        .def_readwrite("max_evaluations", &VarProOptions::max_evaluations)
        .def_readwrite("ftol", &VarProOptions::ftol)
        .def_readwrite("xtol", &VarProOptions::xtol)
        .def_readwrite("gtol", &VarProOptions::gtol);

    py::class_<VarProResult>(m, "VarProResult")
        .def_readonly("theta_hat", &VarProResult::theta_hat)
        .def_readonly("c", &VarProResult::c)
        .def_readonly("s", &VarProResult::s)
        .def_readonly("residual", &VarProResult::residual)
        .def_readonly("cost", &VarProResult::cost)
        .def_readonly("success", &VarProResult::success)
        .def_readonly("num_iterations", &VarProResult::num_iterations)
        .def_readonly("num_residual_evaluations",
                      &VarProResult::num_residual_evaluations)
        .def_readonly("message", &VarProResult::message);

    py::class_<WhitenedBasis>(m, "WhitenedBasis",
                              "The LG feature basis on an ellipsoid, scaled by "
                              "the masses. The only object in the fitting core "
                              "that knows a mass matrix exists.")
        .def(py::init([]( const PointsIn& x, double target_mass,
                          const Eigen::VectorXd& m2_diag,
                          std::vector<Mode> modes, const Eigen::VectorXd& mu0,
                          MuMode mu_mode ) {
                 return WhitenedBasis(Eigen::MatrixXd(map_points(x, "x")),
                                      target_mass, m2_diag, std::move(modes),
                                      mu0, mu_mode);
             }),
             "x"_a, "target_mass"_a, "m2_diag"_a, "modes"_a, "mu0"_a, "mu_mode"_a,
             "x is (N, K).")
        .def_property_readonly("dim", &WhitenedBasis::dim)
        .def("values",
             []( const WhitenedBasis& basis, const Eigen::VectorXd& theta_hat )
             { return batch_last(basis(theta_hat).values()); },
             "theta_hat"_a, "Whitened basis values at theta_hat: (num_modes, K).");

    m.def("fit_varpro",
          []( const PointsIn& z_hat, const Eigen::VectorXd& y_hat,
              const WhitenedBasis& basis, const Eigen::VectorXd& theta_hat_init,
              const PointsIn& e_hat, const VarProOptions& options ) {
              const Eigen::MatrixXd extra(map_batch(e_hat, "e_hat"));
              py::gil_scoped_release unlock;
              return fit_varpro(map_batch(z_hat, "z_hat"), y_hat, basis,
                                theta_hat_init, extra, options);
          },
          "z_hat"_a, "y_hat"_a, "basis"_a, "theta_hat_init"_a,
          "e_hat"_a = Eigen::MatrixXd(), "options"_a = VarProOptions(),
          "Fit one target's nonlinear parameters, eliminating the linear "
          "coefficients in closed form at every trial point. z_hat is "
          "(num_probes, K), e_hat is (num_extra, K); both already whitened.");

    // ---- mode policies -----------------------------------------------------
    // Built-ins only. `fit_operator` releases the GIL and calls propose() from
    // worker threads, so a Python-defined policy would have to re-acquire on
    // every call -- serializing the fit and inviting deadlock. The extension
    // point is deliberately closed.

    py::class_<ModePolicy, std::shared_ptr<ModePolicy>>(
        m, "ModePolicy", "Base class of the mode-growth policies. Not "
                         "subclassable from Python -- see the module notes.");

    py::class_<FixedSet, ModePolicy, std::shared_ptr<FixedSet>>(m, "FixedSet")
        .def(py::init<std::vector<Mode>, std::string>(),
             "modes"_a, "label"_a = "explicit",
             "One explicit mode list, proposed once.");

    py::class_<ShellLadder, ModePolicy, std::shared_ptr<ShellLadder>>(m, "ShellLadder")
        .def(py::init<std::vector<int>>(), "levels"_a,
             "Nested shells: rung i is every mode up to oscillator level "
             "levels[i].");

    py::class_<ExplicitLadder, ModePolicy, std::shared_ptr<ExplicitLadder>>(
        m, "ExplicitLadder")
        .def(py::init<std::vector<std::vector<Mode>>>(), "sets"_a,
             "A caller-supplied nested list of mode sets.");

    py::class_<WedgeLadder, ModePolicy, std::shared_ptr<WedgeLadder>>(m, "WedgeLadder")
        .def(py::init<int, int>(), "max_level"_a = 10, "ell_max"_a = 2,
             "Level-ordered with the angular degree capped -- the strongest "
             "fixed policy on PIG at k >= 40, and the operator layer's usual "
             "choice.");

    py::class_<RadialFirstLadder, ModePolicy, std::shared_ptr<RadialFirstLadder>>(
        m, "RadialFirstLadder")
        .def(py::init<int, int, int>(),
             "max_level"_a = 10, "ell_max"_a = 2, "groups_per_rung"_a = 2);

    // ---- the row fit --------------------------------------------------------

    py::enum_<MuPolicy>(m, "MuPolicy")
        .value("Pinned", MuPolicy::Pinned, "Pin the center at mu0. The default.")
        .value("Free", MuPolicy::Free, "Fit the center from the start.")
        .value("PinnedThenRelease", MuPolicy::PinnedThenRelease);

    py::enum_<StopReason>(m, "StopReason")
        .value("Target", StopReason::Target)
        .value("ModePatience", StopReason::ModePatience)
        .value("Exhausted", StopReason::Exhausted);

    py::class_<CvFold>(m, "CvFold")
        .def_readonly("train", &CvFold::train)
        .def_readonly("validation", &CvFold::validation);

    m.def("kfold_split",
          []( int num_probes, int num_folds, std::optional<std::uint32_t> seed ) {
              return seed ? kfold_split(num_probes, num_folds, *seed)
                          : kfold_split(num_probes, num_folds);
          },
          "num_probes"_a, "num_folds"_a, "seed"_a = std::nullopt,
          "Round-robin folds with no generator at all when seed is None, which "
          "is what makes a fit reproducible with nothing to remember.");

    m.def("jitter_table",
          []( int num_params, int num_levels, std::uint32_t seed )
          { return jitter_table(num_params, num_levels, seed); },
          "num_params"_a, "num_levels"_a, "seed"_a = 0u,
          "Warm-start perturbations, one per ladder level. Exact warm starts "
          "sit on the enrichment saddle, so the perturbation matters; its "
          "distribution does not.");

    py::class_<InitialGuess>(m, "InitialGuess",
                             "A starting ellipsoid for the per-row search, in "
                             "the public (mu, sigma) form.")
        .def(py::init([]( const Eigen::MatrixXd& sigma,
                          const std::optional<Eigen::VectorXd>& mu,
                          const std::string& label ) {
                 InitialGuess guess;
                 guess.sigma = sigma;
                 guess.mu = mu;
                 guess.label = label;
                 return guess;
             }),
             "sigma"_a, py::kw_only(), "mu"_a = std::nullopt, "label"_a = "",
             "sigma is (N, N) SPD. mu is (N,); omit it to start at the fit's "
             "default_mu. Under MuPolicy.Pinned mu is where the center STAYS.")
        .def_readwrite("sigma", &InitialGuess::sigma)
        .def_readwrite("mu", &InitialGuess::mu)
        .def_readwrite("label", &InitialGuess::label)
        .def("__repr__", []( const InitialGuess& g ) {
            return "<InitialGuess " + (g.label.empty() ? std::string("(unlabelled)")
                                                       : g.label)
                   + (g.mu ? ", own mu" : ", at default_mu") + ">";
        });

    m.def("circle_ladder",
          []( const PointsIn& x, const Eigen::VectorXd& default_mu,
              int num_rungs ) {
              return circle_ladder(Eigen::MatrixXd(map_points(x, "x")),
                                   default_mu, num_rungs);
          },
          "x"_a, "default_mu"_a, "num_rungs"_a,
          "Isotropic guesses at log-spaced scales -- the library's default "
          "dictionary. x is (N, K).");

    m.def("window_shape_ladder",
          []( const PointsIn& x, const Eigen::VectorXd& m2_diag,
              const Eigen::VectorXd& default_mu, int num_rungs ) {
              return window_shape_ladder(Eigen::MatrixXd(map_points(x, "x")),
                                         m2_diag, default_mu, num_rungs);
          },
          "x"_a, "m2_diag"_a, "default_mu"_a, "num_rungs"_a,
          "Scaled copies of the batch's own empirical shape. Carries no shape "
          "information your sigma did not already have when the batch is a "
          "window derived from it.");

    m.def("oriented_ladder",
          []( const PointsIn& x, const Eigen::VectorXd& default_mu,
              int num_rungs, const std::vector<double>& angles_degrees,
              double aspect ) {
              return oriented_ladder(Eigen::MatrixXd(map_points(x, "x")),
                                     default_mu, num_rungs, angles_degrees,
                                     aspect);
          },
          "x"_a, "default_mu"_a, "num_rungs"_a, "angles_degrees"_a, "aspect"_a,
          "Anisotropic guesses over a grid of orientations -- the circle "
          "family's blind spot. Two-dimensional only; not on by default "
          "because it has not earned it (see experiments/).");

    py::class_<ProbeFitConfig>(m, "ProbeFitConfig",
                               "The row fit's policy and numerics. Structural "
                               "facts -- window, probes, masses, spike index -- "
                               "are caller-declared, never adjudicated here.")
        .def(py::init<>())
        .def_readwrite("mu", &ProbeFitConfig::mu)
        .def_readwrite("mode_policy", &ProbeFitConfig::mode_policy,
                       "REQUIRED. One mechanism: an explicit list is FixedSet, "
                       "a level ladder is ShellLadder.")
        .def_readwrite("num_rungs", &ProbeFitConfig::num_rungs,
                       "How many default circle rungs to append after the "
                       "caller's guesses. 0 means 'only my guesses', and is an "
                       "error if none were passed.")
        .def_readwrite("target_score", &ProbeFitConfig::target_score,
                       "Absolute early-exit certificate; None disables it.")
        .def_readwrite("mode_patience", &ProbeFitConfig::mode_patience)
        .def_readwrite("tie_delta", &ProbeFitConfig::tie_delta)
        .def_readwrite("cv_folds", &ProbeFitConfig::cv_folds)
        .def_readwrite("split", &ProbeFitConfig::split,
                       "The CV split AS DATA. Empty means the deterministic "
                       "round-robin split from cv_folds.")
        .def_readwrite("jitter", &ProbeFitConfig::jitter)
        .def_readwrite("varpro", &ProbeFitConfig::varpro);

    py::class_<LGExpansion>(m, "LGExpansion",
                            "One target's model: absolute theta, its mode "
                            "list, and the coefficients. Self-decoding -- the "
                            "ellipsoid comes from frame() with no mu0 and no "
                            "mode.")
        .def(py::init([]( Eigen::VectorXd theta, std::vector<Mode> modes,
                          Eigen::VectorXd c, Eigen::VectorXd s ) {
                 return LGExpansion{std::move(theta), std::move(modes),
                                    std::move(c), std::move(s)};
             }),
             "theta"_a, "modes"_a, "c"_a, "s"_a = Eigen::VectorXd())
        .def_readonly("theta", &LGExpansion::theta)
        .def_readonly("modes", &LGExpansion::modes)
        .def_readonly("c", &LGExpansion::c)
        .def_readonly("s", &LGExpansion::s)
        .def_property_readonly("dim", &LGExpansion::dim)
        .def_property_readonly("num_modes", &LGExpansion::num_modes)
        .def("frame", &LGExpansion::frame)
        .def("__repr__", []( const LGExpansion& e ) {
                 std::ostringstream out;
                 out << "LGExpansion(dim=" << e.dim() << ", num_modes="
                     << e.num_modes() << ")";
                 return out.str();
             });

    m.def("eval_expansion",
          []( const LGExpansion& expansion, const PointsIn& x )
          { return Eigen::VectorXd(
                eval_expansion(expansion, map_points(x, "x"))); },
          "expansion"_a, "x"_a,
          "The smooth part at arbitrary query points (N, K); returns (K,). The "
          "spike has no off-grid meaning and is not included.");

    m.def("validate_expansion",
          []( const LGExpansion& expansion ) { return validate(expansion); },
          "expansion"_a,
          "Structural complaints about a hand-built expansion; empty is fine.");

    py::class_<CandidateFit>(m, "CandidateFit",
                             "One scored entry of the candidate table: the "
                             "model it found, and how it went.")
        .def_readonly("model", &CandidateFit::model)
        .def_readonly("label", &CandidateFit::label)
        .def_readonly("modes_label", &CandidateFit::modes_label)
        .def_readonly("released", &CandidateFit::released)
        .def_readonly("theta_init", &CandidateFit::theta_init,
                      "Where this candidate started, in the PUBLIC absolute\n"
                      "encoding -- so theta_init[:N] is the center it was\n"
                      "seeded at, and it pairs directly with model.theta.")
        .def_readonly("cost", &CandidateFit::cost,
                      "In-sample whitened cost -- a diagnostic, NEVER used for "
                      "selection.")
        .def_readonly("score", &CandidateFit::score)
        .def_readonly("axes", &CandidateFit::axes)
        .def_readonly("success", &CandidateFit::success)
        .def_readonly("num_iterations", &CandidateFit::num_iterations)
        .def_readonly("admissible", &CandidateFit::admissible)
        .def_property_readonly("num_modes", &CandidateFit::num_modes);

    py::class_<ProbeFitResult>(m, "ProbeFitResult",
                               "The winning model, plus the full audit trail.")
        .def_readonly("model", &ProbeFitResult::model)
        .def_readonly("released", &ProbeFitResult::released)
        .def_readonly("score", &ProbeFitResult::score)
        .def_readonly("stop_reason", &ProbeFitResult::stop_reason)
        .def_readonly("winner", &ProbeFitResult::winner)
        .def_readonly("candidates", &ProbeFitResult::candidates)
        .def_readonly("skipped", &ProbeFitResult::skipped);

    m.def("linear_cv_score",
          []( const PointsIn& z_hat, const Eigen::VectorXd& y_hat,
              const WhitenedBasis& basis, const Eigen::VectorXd& theta_hat,
              const PointsIn& e_hat, const CrossValidationSplit& split ) {
              const Eigen::MatrixXd extra(map_batch(e_hat, "e_hat"));
              py::gil_scoped_release unlock;
              return linear_cv_score(map_batch(z_hat, "z_hat"), y_hat, basis,
                                     theta_hat, extra, split);
          },
          "z_hat"_a, "y_hat"_a, "basis"_a, "theta_hat"_a, "e_hat"_a, "split"_a,
          "Cross-validation score of a model at GIVEN parameters, with no "
          "nonlinear fit at all -- which is what makes scoring an a-priori "
          "field affordable before fitting anything.");

    m.def("fit_from_probes",
          []( const PointsIn& x, const Eigen::VectorXd& m2_diag,
              const PointsIn& z, const Eigen::VectorXd& y,
              const Eigen::VectorXd& default_mu, int spike_index,
              const ProbeFitConfig& config,
              const std::vector<InitialGuess>& guesses,
              std::optional<double> target_mass ) {
              const Eigen::MatrixXd points(map_points(x, "x"));
              const Eigen::MatrixXd probes(map_batch(z, "z"));
              py::gil_scoped_release unlock;
              return fit_from_probes(points, m2_diag, probes, y, default_mu,
                                     spike_index, config, guesses, target_mass);
          },
          "x"_a, "m2_diag"_a, "z"_a, "y"_a, "default_mu"_a, py::kw_only(),
          "spike_index"_a = -1, "config"_a = ProbeFitConfig(),
          "guesses"_a = std::vector<InitialGuess>(),
          "target_mass"_a = std::nullopt,
          "Fit a target known only through inner products with probe fields. "
          "x is (N, K), z is (num_probes, K), y is (num_probes,). Starting "
          "ellipsoids go in `guesses`, tried before the default circle rungs; "
          "pass your a-priori covariance there to try it first.");

    // ---- the operator ------------------------------------------------------
    //
    // TWO conventions meet here, and the difference is worth stating plainly.
    // A POINT BATCH is (N, K) in both directions: what you pass as `x_cols`
    // comes back as `x_cols`. A PER-ROW RECORD is row-first, (R, ...), because
    // the leading axis is "which row" and `fit.mu[rho]` is how anyone reads
    // one. So `x_cols` is (N, K) while `mu` is (R, N); R != N in any real
    // problem, so a mix-up is a shape error rather than a silent wrong answer.

    py::enum_<RowStatus>(m, "RowStatus", "What happened to a row.")
        .value("Fit", RowStatus::Fit, "The searched fit beat the baseline.")
        .value("GatedOut", RowStatus::GatedOut, "The caller's gate excluded it.")
        .value("FallbackBaseline", RowStatus::FallbackBaseline,
               "The baseline shipped; the search did not beat it.")
        .value("Failed", RowStatus::Failed, "The row raised; see failures.");

    py::enum_<RowStop>(m, "RowStop")
        .value("None_", RowStop::None)
        .value("Target", RowStop::Target)
        .value("ModePatience", RowStop::ModePatience)
        .value("Exhausted", RowStop::Exhausted)
        .value("SearchInfeasible", RowStop::SearchInfeasible);

    py::enum_<Symmetrize>(m, "Symmetrize",
                          "What assemble_sparse does about symmetry. An "
                          "ASSEMBLY POLICY, not a fit property.")
        .value("None_", Symmetrize::None, "Rows exactly as fitted.")
        .value("Average", Symmetrize::Average, "(A + A^T) / 2; square context only.")
        .value("Weighted", Symmetrize::Weighted,
               "Convex average with inverse-row-energy weights -- the weak row "
               "owns the entries a strong row only grazes. The recommendation "
               "when a symmetric operator is wanted; square context only.");

    py::class_<OperatorFitConfig>(m, "OperatorFitConfig")
        .def(py::init<>())
        .def_readwrite("tau_window", &OperatorFitConfig::tau_window)
        .def_readwrite("window_aspect_cap", &OperatorFitConfig::window_aspect_cap,
                       "1 = an isotropic window (a ball), inf = the caller's "
                       "ellipsoid untouched, kappa = cap the axis ratio there. "
                       "Moves the window's SHAPE, never its scale.")
        .def_readwrite("spike", &OperatorFitConfig::spike)
        .def_readwrite("row", &OperatorFitConfig::row,
                       "The per-row config. Its split and jitter are "
                       "OVERWRITTEN here: the operator layer owns them, so "
                       "every row and the baseline guard score on identical "
                       "folds.")
        .def_readwrite("seed", &OperatorFitConfig::seed,
                       "None gives deterministic round-robin folds and a "
                       "deterministic jitter table -- the whole fit is then "
                       "reproducible with nothing to remember.")
        .def_readwrite("num_threads", &OperatorFitConfig::num_threads);

    py::class_<LGOperator>(m, "LGOperator",
                           "The fitted operator as a data structure: "
                           "H~ = M1 Phi~ M2 + M1 S, never a matrix. Knows "
                           "nothing about fitting -- operator_fit is one "
                           "producer of these, not their definition.")
        .def_readonly("dim", &LGOperator::dim)
        .def_property_readonly("x_cols",
                               []( const LGOperator& f ) { return batch_last(f.x_cols); },
                               "(N, K) column-dof coordinates -- a point batch.")
        .def_property_readonly("x_rows",
                               []( const LGOperator& f ) -> py::object {
                                   if ( !f.x_rows ) { return py::none(); }
                                   return batch_last(*f.x_rows);
                               },
                               "(N, R), or None in the square dof context.")
        .def_readonly("m1_diag", &LGOperator::m1_diag)
        .def_readonly("m2_diag", &LGOperator::m2_diag)
        .def_readonly("spike", &LGOperator::spike)
        .def_readonly("theta", &LGOperator::theta,
                      "(R, P) in the PUBLIC absolute encoding, so every row "
                      "decodes with unpack_theta alone.")
        .def_readonly("mu", &LGOperator::mu, "(R, N)")
        .def_property_readonly("L",
                               []( const LGOperator& f )
                               { return unflatten_blocks(f.L, f.dim); },
                               "(R, N, N) Cholesky factors; Sigma = L L^T.")
        .def_readonly("c", &LGOperator::c, "(R, m_max), zero-padded.")
        .def_readonly("s", &LGOperator::s, "(R,) additive spike coefficients.")
        .def_readonly("mode_set_id", &LGOperator::mode_set_id, "(R,), -1 = no model.")
        .def_readonly("mode_sets", &LGOperator::mode_sets)
        .def_property_readonly("window_center", []( const LGOperator& f )
                               { return f.window_center; }, "(R, N)")
        .def_property_readonly("window_covariance",
                               []( const LGOperator& f )
                               { return unflatten_blocks(f.window_covariance, f.dim); },
                               "(R, N, N), pre-scaled so membership is "
                               "Mahalanobis <= 1.")
        .def_readonly("window_indptr", &LGOperator::window_indptr)
        .def_readonly("window_indices", &LGOperator::window_indices)
        .def_property_readonly("num_rows", &LGOperator::num_rows)
        .def_property_readonly("num_cols", &LGOperator::num_cols)
        .def("has_model", &LGOperator::has_model, "rho"_a)
        .def("row_window", &LGOperator::row_window, "rho"_a,
             "Row rho's fit-window column indices, sorted. The DEPLOYED "
             "support: every evaluation is zero outside it.")
        .def("row_modes", &LGOperator::row_modes, "rho"_a)
        .def("__repr__", []( const LGOperator& f ) {
                 std::ostringstream out;
                 out << "LGOperator(dim=" << f.dim << ", num_rows=" << f.num_rows()
                     << ", num_cols=" << f.num_cols() << ")";
                 return out.str();
             });

    py::class_<FitDiagnostics>(m, "FitDiagnostics",
                               "Per-row provenance: how each row went, not "
                               "what it produced. Nothing here is read by any "
                               "evaluation.")
        .def_readonly("score", &FitDiagnostics::score)
        .def_readonly("baseline_score", &FitDiagnostics::baseline_score)
        .def_property_readonly("status",
                               []( const FitDiagnostics& d )
                               { return codes_of(d.status); },
                               "(R,) int8 of RowStatus, so masking works: "
                               "`status == int(RowStatus.Fit)`.")
        .def_property_readonly("stop_reason",
                               []( const FitDiagnostics& d )
                               { return codes_of(d.stop_reason); },
                               "(R,) int8 of RowStop.")
        .def_property_readonly("released",
                               []( const FitDiagnostics& d ) {
                                   py::array_t<bool> out(
                                       static_cast<py::ssize_t>(d.released.size()));
                                   for ( std::size_t i = 0; i < d.released.size(); ++i )
                                   {
                                       out.mutable_at(static_cast<py::ssize_t>(i)) =
                                           d.released[i] != 0;
                                   }
                                   return out;
                               })
        .def_readonly("failures", &FitDiagnostics::failures,
                      "Row -> message, for rows that raised.")
        .def_readonly("config", &FitDiagnostics::config, "Provenance echo.");

    py::class_<OperatorFit>(m, "OperatorFit",
                            "What fit_operator returns: the operator, and how "
                            "it went. Separate halves on purpose.")
        .def_readonly("model", &OperatorFit::model)
        .def_readonly("diagnostics", &OperatorFit::diagnostics);

    m.def("fit_operator",
          []( const PointsIn& x_cols, const Eigen::VectorXd& m1_diag,
              const Eigen::VectorXd& m2_diag, const PointsIn& V,
              const PointsIn& HV, const py::array_t<double>& sigma,
              const OperatorFitConfig& config,
              const std::optional<PointsIn>& mu0,
              const std::optional<PointsIn>& x_rows, const py::object& rows ) {
              const Eigen::MatrixXd columns(map_points(x_cols, "x_cols"));
              const Eigen::MatrixXd probes(map_batch(V, "V"));
              const Eigen::MatrixXd responses(map_batch(HV, "HV"));
              const int dim = static_cast<int>(columns.cols());
              std::vector<Eigen::MatrixXd> covariances =
                  sigma_from_array(sigma, m1_diag.size(), dim);
              std::optional<Eigen::MatrixXd> centers;
              if ( mu0 ) { centers = Eigen::MatrixXd(map_points(*mu0, "mu0")); }
              std::optional<Eigen::MatrixXd> row_points;
              if ( x_rows )
              {
                  row_points = Eigen::MatrixXd(map_points(*x_rows, "x_rows"));
              }
              const std::vector<char> gate = gate_from_rows(rows, m1_diag.size());

              py::gil_scoped_release unlock;
              return fit_operator(columns, m1_diag, m2_diag, probes, responses,
                                  covariances, config, centers, row_points, gate);
          },
          "x_cols"_a, "m1_diag"_a, "m2_diag"_a, "V"_a, "HV"_a, "sigma"_a,
          py::kw_only(), "config"_a = OperatorFitConfig(),
          "mu0"_a = std::nullopt, "x_rows"_a = std::nullopt,
          "rows"_a = py::none(),
          "Fit the whole operator, one row at a time.\n\n"
          "x_cols (N, K), V (num_probes, K), HV (num_probes, R), sigma "
          "(R, N, N), mu0 (N, R). `rows` accepts a boolean mask OR an index "
          "array and selects which rows to attempt; None (the default) "
          "attempts every row, WHICH IS THE RECOMMENDED SETTING -- a dead row "
          "costs one candidate and ships zeros, and gating buys nothing.");

    // ---- evaluating a fitted operator --------------------------------------

    m.def("model_rows", &model_rows, "fit"_a,
          "Indices of the rows carrying a shipped model.");

    m.def("eval_kernel",
          []( const LGOperator& fit, const std::vector<int>& rows,
              const PointsIn& x_query, double truncation_tau )
          { return batch_last(eval_kernel(fit, rows, map_points(x_query, "x_query"),
                                          truncation_tau)); },
          "fit"_a, "rows"_a, "x_query"_a,
          "truncation_tau"_a = std::numeric_limits<double>::infinity(),
          "The smooth component at arbitrary query points (N, Q), RESTRICTED "
          "TO THE FIT WINDOW and zero outside it. Returns (num_rows, Q).");

    m.def("eval_kernel_unrestricted",
          []( const LGOperator& fit, const std::vector<int>& rows,
              const PointsIn& x_query )
          { return batch_last(eval_kernel_unrestricted(
                fit, rows, map_points(x_query, "x_query"))); },
          "fit"_a, "rows"_a, "x_query"_a,
          "The fitted form extended BEYOND its window -- the named opt-out. "
          "Out-of-window model mass is unpenalized by the fit, not merely "
          "unverified, so asking for it requires saying so.");

    m.def("eval_entries",
          []( const LGOperator& fit, const std::vector<int>& rows,
              const std::vector<int>& cols, double truncation_tau )
          { return eval_entries(fit, rows, cols, truncation_tau); },
          "fit"_a, "rows"_a, "cols"_a,
          "truncation_tau"_a = std::numeric_limits<double>::infinity(),
          "Paired entries of the deployed operator; rows and cols are "
          "equal-length index arrays.");

    m.def("matvec",
          []( const LGOperator& fit, const PointsIn& v, double truncation_tau,
              int num_threads ) {
              const Eigen::MatrixXd vectors(map_batch(v, "v"));
              Eigen::MatrixXd result;
              {
                  // The release must END before the numpy array is built:
                  // `batch_last` allocates a PYTHON OBJECT, and doing that
                  // without the GIL is a segfault, not an error. Every other
                  // released call here returns a C++ value and lets pybind11
                  // cast it after the scope closes.
                  py::gil_scoped_release unlock;
                  result = matvec(fit, vectors, truncation_tau, num_threads);
              }
              return batch_last(result);
          },
          "fit"_a, "v"_a,
          "truncation_tau"_a = std::numeric_limits<double>::infinity(),
          "num_threads"_a = 0,
          "Apply the deployed operator with zero assembly. v is "
          "(num_vectors, K); returns (num_vectors, R).");

    m.def("assemble_sparse",
          []( const LGOperator& fit, double tau, Symmetrize symmetrize,
              int num_threads ) {
              py::gil_scoped_release unlock;
              return assemble_sparse(fit, tau, symmetrize, num_threads);
          },
          "fit"_a, "tau"_a, "symmetrize"_a = Symmetrize::None,
          "num_threads"_a = 0,
          "Sparse decompression of the deployed operator: the tau-ellipsoid "
          "support INTERSECTED with each row's fit window, plus the spike. "
          "Returns a scipy.sparse matrix -- the one function here that needs "
          "scipy.");

    m.def("qc_map",
          []( const LGOperator& fit, const PointsIn& V_qc, const PointsIn& HV_qc,
              int num_threads ) {
              const Eigen::MatrixXd probes(map_batch(V_qc, "V_qc"));
              const Eigen::MatrixXd responses(map_batch(HV_qc, "HV_qc"));
              py::gil_scoped_release unlock;
              return Eigen::VectorXd(qc_map(fit, probes, responses, num_threads));
          },
          "fit"_a, "V_qc"_a, "HV_qc"_a, "num_threads"_a = 0,
          "Per-row relative residual against held-out probes -- the "
          "scorecard. V_qc (num_qc, K), HV_qc (num_qc, R); NaN where no model.");

    m.def("spike_measure", &spike_measure, "fit"_a,
          "m1 * s, the mass-weighted spike content. Where it is large, the "
          "mesh is starving.");

    m.def("ellipsoid_field",
          []( const LGOperator& fit ) {
              const EllipsoidField field = ellipsoid_field(fit);
              py::array_t<double> sigma({static_cast<py::ssize_t>(field.sigma.size()),
                                         static_cast<py::ssize_t>(fit.dim),
                                         static_cast<py::ssize_t>(fit.dim)});
              auto view = sigma.mutable_unchecked<3>();
              for ( std::size_t r = 0; r < field.sigma.size(); ++r )
              {
                  for ( int i = 0; i < fit.dim; ++i )
                  {
                      for ( int j = 0; j < fit.dim; ++j )
                      {
                          view(static_cast<py::ssize_t>(r), i, j) =
                              field.sigma[r](i, j);
                      }
                  }
              }
              return py::make_tuple(field.mu, sigma);
          },
          "fit"_a,
          "The fitted (mu, Sigma) field as ((R, N), (R, N, N)) -- what "
          "ellipsoid_tree consumes.");

    m.def("validate_operator",
          []( const LGOperator& fit ) { return validate(fit); }, "fit"_a,
          "Structural complaints about a hand-built operator; empty is fine.");

    m.def("row_expansion", &row_expansion, "fit"_a, "rho"_a,
          "Extract one row's model as a standalone LGExpansion.");

    py::class_<OperatorRow>(m, "OperatorRow",
                            "One row of a hand-built operator: its model and "
                            "its window.")
        .def(py::init([]( LGExpansion model, Eigen::VectorXd window_center,
                          Eigen::MatrixXd window_covariance,
                          std::vector<int> window_indices ) {
                 return OperatorRow{std::move(model), std::move(window_center),
                                    std::move(window_covariance),
                                    std::move(window_indices)};
             }),
             "model"_a, "window_center"_a, "window_covariance"_a,
             "window_indices"_a)
        .def_readonly("model", &OperatorRow::model);

    m.def("build_operator",
          []( const PointsIn& x_cols, const Eigen::VectorXd& m1_diag,
              const Eigen::VectorXd& m2_diag, bool spike,
              const std::vector<std::optional<OperatorRow>>& rows,
              const std::optional<PointsIn>& x_rows ) {
              std::optional<Eigen::MatrixXd> row_points;
              if ( x_rows )
              {
                  row_points = Eigen::MatrixXd(map_points(*x_rows, "x_rows"));
              }
              return build_operator(Eigen::MatrixXd(map_points(x_cols, "x_cols")),
                                    m1_diag, m2_diag, spike, rows, row_points);
          },
          "x_cols"_a, "m1_diag"_a, "m2_diag"_a, "spike"_a, "rows"_a,
          "x_rows"_a = std::nullopt,
          "Assemble an operator from per-row expansions -- the path for a "
          "physics-based approximation, with no fitter involved. Unmodeled "
          "rows are None.");

    m.def("concatenate_rows", &concatenate_rows, "parts"_a,
          "Merge operators covering disjoint row ranges.");

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
