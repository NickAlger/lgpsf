#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Real harmonic polynomials Y_{ell,m} on R^N: evaluation and gradient.
///
/// Y_{ell,m} is a homogeneous polynomial of degree ell in N variables with
/// Delta Y = 0. For each (N, ell) the modes m = 0 .. num_harmonics(N, ell) - 1
/// form an orthonormal basis of that space under the surface measure on
/// S^(N-1); restricted to the sphere they generalize the spherical harmonics
/// (for N = 2 they are r^ell {cos, sin}(ell theta), the angular part of the
/// classical Laguerre-Gaussian modes).
///
/// **This header is the only place the generated table's storage format
/// appears.** Everything else goes through the functions below, the same
/// confinement whitening.hpp applies to the mass matrices. Changing how
/// harmonic polynomials are represented or evaluated -- a different term
/// ordering, a Horner scheme, a reduced-precision path -- is then a change to
/// this file alone.
///
/// Point batches are (K, N): points as ROWS, dimensions as COLUMNS, so
/// `u.col(k)` is one coordinate across the whole batch, contiguous under
/// Eigen's column-major default. (The Python bindings take (N, K), which is
/// the same bytes; see docs/api-guide.md.)

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/detail/lg_harmonics_table.hpp"

namespace lgpsf {

/// One harmonic shell index: degree `ell`, mode `m` within that degree.
struct Shell
{
    int ell = 0;
    int m   = 0;
};

/// A view of one stored polynomial: its nonzero terms, borrowed from the
/// generated table (no ownership, no allocation).
///
/// Term t has coefficient `coefficients[t]` and exponent vector
/// `exponents[t * dim + k]`, k = 0..dim-1. Terms absent from the list have an
/// exactly zero coefficient. Exponent vectors appear in lexicographically
/// descending order and the list is never empty.
struct HarmonicPolynomial
{
    int dim       = 0;
    int degree    = 0;
    int num_terms = 0;
    const double*      coefficients = nullptr;
    const signed char* exponents    = nullptr;
};

/// Largest spatial dimension N the generated table covers.
inline int max_dimension() { return detail::kHarmonicMaxDimension; }

/// Largest harmonic degree ell the generated table covers.
inline int max_degree() { return detail::kHarmonicMaxDegree; }

namespace detail {

/// Shell index for (N, ell): plain arithmetic, no map.
inline int harmonic_shell_index( int dim, int degree )
{
    if ( dim < 1 || dim > kHarmonicMaxDimension
         || degree < 0 || degree > kHarmonicMaxDegree )
    {
        throw std::invalid_argument(
            "lgpsf::harmonic_polynomials: table covers N in 1.."
            + std::to_string(kHarmonicMaxDimension) + " and ell in 0.."
            + std::to_string(kHarmonicMaxDegree) + "; (N=" + std::to_string(dim)
            + ", ell=" + std::to_string(degree) + ") is outside it");
    }
    return (dim - 1) * (kHarmonicMaxDegree + 1) + degree;
}

} // end namespace detail

/// Number of degree-ell harmonic modes in N dimensions -- dim H_ell(R^N), i.e.
/// the number of valid m values.
///
/// Zero only for N == 1, ell >= 2: there is no harmonic function of one
/// variable of degree >= 2.
///
/// @param dim    Spatial dimension N, 1 to max_dimension().
/// @param degree Harmonic degree ell, 0 to max_degree().
/// @return       The number of valid m values for that (N, ell).
/// @throws std::invalid_argument if (N, ell) is outside the generated table
///         -- a caller error, not an empty answer.
inline int num_harmonics( int dim, int degree )
{
    const int s = detail::harmonic_shell_index(dim, degree);
    return detail::kHarmonicShellRowBegin[s + 1] - detail::kHarmonicShellRowBegin[s];
}

/// The stored terms of Y_{ell,m} on R^N.
///
/// @param dim    Spatial dimension N.
/// @param degree Harmonic degree ell.
/// @param m      Which harmonic of that degree, 0 to num_harmonics()-1.
/// @return       Its nonzero terms; zero coefficients are absent by
///               construction, not by threshold.
/// @throws std::invalid_argument if (N, ell) is outside the table or @p m is
///         out of range.
inline HarmonicPolynomial harmonic_terms( int dim, int degree, int m )
{
    const int s = detail::harmonic_shell_index(dim, degree);
    const int count = num_harmonics(dim, degree);
    if ( m < 0 || m >= count )
    {
        throw std::invalid_argument(
            "lgpsf::harmonic_terms: N=" + std::to_string(dim) + " ell="
            + std::to_string(degree) + " has " + std::to_string(count)
            + " harmonic mode(s); m=" + std::to_string(m) + " is out of range");
    }
    const int row = detail::kHarmonicShellRowBegin[s] + m;

    HarmonicPolynomial poly;
    poly.dim       = dim;
    poly.degree    = degree;
    poly.num_terms = detail::kHarmonicRowTermBegin[row + 1]
                     - detail::kHarmonicRowTermBegin[row];
    poly.coefficients = detail::kHarmonicCoefficient + detail::kHarmonicRowTermBegin[row];
    poly.exponents    = detail::kHarmonicExponent + detail::kHarmonicRowExponentBegin[row];
    return poly;
}

/// A whole shell list evaluated at one point batch.
struct HarmonicBasis
{
    Eigen::MatrixXd values;                  ///< (K, num_shells), column per shell
    std::vector<Eigen::MatrixXd> gradients;  ///< num_shells of (K, dim); empty unless requested
};

namespace detail {

/// powers[k].col(a) = u.col(k)^a, a = 1..max_exponent (column 0 unused: an
/// a == 0 factor is exactly 1.0, so callers skip it rather than multiplying
/// by a vector of ones).
///
/// Built by repeated multiplication rather than std::pow -- faster, and no
/// less accurate for these small integer exponents.
inline std::vector<Eigen::MatrixXd> power_tables(
    const Eigen::Ref<const Eigen::MatrixXd>& u, int max_exponent )
{
    const Eigen::Index n_points = u.rows();
    const int dim = static_cast<int>(u.cols());
    std::vector<Eigen::MatrixXd> powers;
    powers.reserve(dim);
    for ( int k = 0; k < dim; ++k )
    {
        Eigen::MatrixXd column_powers(n_points, std::max(2, max_exponent + 1));
        column_powers.col(0).setOnes();
        column_powers.col(1) = u.col(k);
        for ( int a = 2; a <= max_exponent; ++a )
        {
            column_powers.col(a) =
                column_powers.col(a - 1).cwiseProduct(u.col(k));
        }
        powers.push_back(std::move(column_powers));
    }
    return powers;
}

} // end namespace detail

/// Evaluate a list of shells at one point batch, optionally with gradients.
///
/// `u` is (K, N); values comes back (K, num_shells) and, when requested,
/// gradients as num_shells matrices of shape (K, N).
///
/// `known_values` lets a caller that already holds the values for these shells
/// skip the value accumulation and get only the gradients -- the case that
/// arises whenever a basis object is asked for values first and a derivative
/// afterwards, which is exactly the VarPro pattern.
inline HarmonicBasis harmonic_basis(
    const std::vector<Shell>& shells,
    const Eigen::Ref<const Eigen::MatrixXd>& u,
    bool with_gradient = false,
    const Eigen::MatrixXd* known_values = nullptr )
{
    const Eigen::Index n_points = u.rows();
    const int dim = static_cast<int>(u.cols());
    const int n_shells = static_cast<int>(shells.size());

    int max_exponent = 0;
    std::vector<HarmonicPolynomial> polys;
    polys.reserve(shells.size());
    for ( const Shell& shell : shells )
    {
        polys.push_back(harmonic_terms(dim, shell.ell, shell.m));
        max_exponent = std::max(max_exponent, shell.ell);
    }
    const std::vector<Eigen::MatrixXd> powers = detail::power_tables(u, max_exponent);

    HarmonicBasis out;
    const bool want_values = ( known_values == nullptr );
    if ( want_values )
    {
        out.values = Eigen::MatrixXd::Zero(n_points, n_shells);
    }
    else
    {
        out.values = *known_values;
    }
    if ( with_gradient )
    {
        out.gradients.assign(n_shells, Eigen::MatrixXd::Zero(n_points, dim));
    }

    Eigen::VectorXd scratch(n_points);
    for ( int i = 0; i < n_shells; ++i )
    {
        const HarmonicPolynomial& poly = polys[i];
        for ( int t = 0; t < poly.num_terms; ++t )
        {
            const signed char* alpha = poly.exponents + static_cast<std::ptrdiff_t>(t) * dim;
            const double coeff = poly.coefficients[t];

            if ( want_values )
            {
                scratch.setConstant(coeff);
                for ( int k = 0; k < dim; ++k )
                {
                    if ( alpha[k] != 0 )
                    {
                        scratch = scratch.cwiseProduct(powers[k].col(alpha[k]));
                    }
                }
                out.values.col(i) += scratch;
            }
            if ( !with_gradient )
            {
                continue;
            }

            for ( int k = 0; k < dim; ++k )
            {
                const int a_k = alpha[k];
                if ( a_k == 0 )
                {
                    continue;
                }
                scratch.setConstant(coeff * a_k);
                for ( int j = 0; j < dim; ++j )
                {
                    const int power = ( j == k ) ? alpha[j] - 1 : alpha[j];
                    if ( power != 0 )
                    {
                        scratch = scratch.cwiseProduct(powers[j].col(power));
                    }
                }
                out.gradients[i].col(k) += scratch;
            }
        }
    }
    return out;
}

/// One harmonic polynomial at every point.
///
/// @param degree Harmonic degree ell.
/// @param m      Which harmonic of that degree.
/// @param u      Points, (K, N).
/// @return       Y_{ell,m}(u), shape (K,).
inline Eigen::VectorXd eval_harmonic(
    int degree, int m, const Eigen::Ref<const Eigen::MatrixXd>& u )
{
    return harmonic_basis({Shell{degree, m}}, u).values.col(0);
}

/// One harmonic polynomial and its spatial gradient, from a single pass over
/// the term list.
///
/// @param degree Harmonic degree ell.
/// @param m      Which harmonic of that degree.
/// @param u      Points, (K, N).
/// @return       {values (K,), gradient (K, N)}.
inline std::pair<Eigen::VectorXd, Eigen::MatrixXd> grad_harmonic(
    int degree, int m, const Eigen::Ref<const Eigen::MatrixXd>& u )
{
    HarmonicBasis basis = harmonic_basis({Shell{degree, m}}, u, true);
    return {basis.values.col(0), std::move(basis.gradients[0])};
}

} // end namespace lgpsf
