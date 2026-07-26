#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The noise-whitening layer: **the only place M1 (the target's own
/// mass) and M2 (the column masses) appear anywhere in this library.**
///
/// See docs/varpro-whitening-notes.tex for the derivation. Everything here is
/// per target: `target_mass` is the single scalar (M1)_rho,rho for whichever
/// dof is being fit (the derivation writes it m_rho), and `m2_diag` holds the
/// column masses (M2)_jj over that target's support batch, one per point.
///
///     z_hat = whiten_probes(z, m2_diag)               = M2^(1/2) z
///     y_hat = whiten_data(y, target_mass)             = y / sqrt(m_rho)
///     E_hat = whiten_extra(E, target_mass, m2_diag)   = sqrt(m_rho) M2^(-1/2) E
///     phi_hat = WhitenedBasisAt::values()             = sqrt(m_rho) M2^(1/2) phi
///
/// so that
///
///     y_hat = sum_i c_i (phi_hat_i . z_hat) + sum_d s_d (E_hat_d . z_hat)
///
/// is an ordinary mass-free linear model, and orthogonalizing phi_hat against
/// E_hat under the plain Euclidean inner product is exactly the correct,
/// dual-space-respecting projection. The smooth basis carries the column mass
/// (it is a continuum-kernel evaluation at a quadrature point) while the extra
/// basis does not (it is a direct discrete correction) -- which is why the two
/// are whitened by opposite powers of M2, and why that asymmetry has to be
/// stated exactly once, here.
///
/// The scaling is a fixed, parameter-independent, symmetric operator, so it
/// composes with the existing derivative chain with NO new derivative math: a
/// forward derivative gets the operator applied to the raw result, and a
/// reverse one gets it applied to the incoming cotangent first.
///
/// Hats and the fitting core. Everything crossing into varpro.hpp is hatted --
/// `z_hat`, `y_hat`, `e_hat`, and the `theta_hat` the basis consumes -- so the
/// core is mass-free and geometry-free by construction. The transforms differ
/// (mass whitening here, an mu0-shift and mode reduction for the parameters;
/// see ellipsoid_transform.hpp) but the role is the same: the coordinates the
/// numerics work in, with the conversions confined to this boundary.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/ellipsoid_transform.hpp"
#include "lgpsf/lg_ellipsoid_feature.hpp"
#include "lgpsf/lg_functions.hpp"

namespace lgpsf {

namespace detail {

/// sqrt(target_mass) * sqrt(m2_diag), the scaling applied to the smooth basis.
inline Eigen::VectorXd whitening_scale( double target_mass,
                                        const Eigen::Ref<const Eigen::VectorXd>& m2_diag )
{
    if ( !(target_mass > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::whitening: target_mass must be positive, got "
            + std::to_string(target_mass));
    }
    if ( (m2_diag.array() <= 0.0).any() )
    {
        throw std::invalid_argument(
            "lgpsf::whitening: every entry of m2_diag must be positive");
    }
    return std::sqrt(target_mass) * m2_diag.cwiseSqrt();
}

/// Scale every column of a point-indexed matrix by a per-point vector.
inline Eigen::MatrixXd scale_rows( const Eigen::Ref<const Eigen::MatrixXd>& a,
                                   const Eigen::Ref<const Eigen::VectorXd>& scale )
{
    if ( a.rows() != scale.size() )
    {
        throw std::invalid_argument(
            "lgpsf::whitening: expected " + std::to_string(scale.size())
            + " rows (one per point), got " + std::to_string(a.rows()));
    }
    return (a.array().colwise() * scale.array()).matrix();
}

} // end namespace detail

/// z_hat = M2^(1/2) z, for probe fields z of shape (K, num_probes).
inline Eigen::MatrixXd whiten_probes( const Eigen::Ref<const Eigen::MatrixXd>& z,
                                      const Eigen::Ref<const Eigen::VectorXd>& m2_diag )
{
    return detail::scale_rows(z, m2_diag.cwiseSqrt());
}

/// y_hat = y / sqrt(target_mass), for the target's raw probe responses (k,).
inline Eigen::VectorXd whiten_data( const Eigen::Ref<const Eigen::VectorXd>& y,
                                    double target_mass )
{
    if ( !(target_mass > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::whiten_data: target_mass must be positive, got "
            + std::to_string(target_mass));
    }
    return y / std::sqrt(target_mass);
}

/// E_hat = sqrt(target_mass) M2^(-1/2) E, for an extra basis of shape
/// (K, num_extra). The INVERSE power of M2, unlike the smooth basis: the extra
/// functions are discrete corrections, not quadrature objects, so they carry
/// no column mass of their own.
inline Eigen::MatrixXd whiten_extra( const Eigen::Ref<const Eigen::MatrixXd>& E,
                                     double target_mass,
                                     const Eigen::Ref<const Eigen::VectorXd>& m2_diag )
{
    const Eigen::VectorXd scale =
        (detail::whitening_scale(target_mass, m2_diag).array()
         / m2_diag.array()).matrix();
    return detail::scale_rows(E, scale);
}

/// The whitened basis evaluated at one theta_hat: a FeatureAt plus the fixed
/// scaling sqrt(target_mass) M2^(1/2).
///
/// This is the object the VarPro fitting core holds for the duration of one
/// trial theta_hat, so that the values and the derivative it needs afterwards
/// share the pullback and the whole LG evaluation underneath.
class WhitenedBasisAt
{
public:
    WhitenedBasisAt( FeatureAt at, Eigen::VectorXd scale )
        : at_(std::move(at)), scale_(std::move(scale))
    {
    }

    int num_params() const { return at_.num_params(); }
    std::size_t num_modes() const { return at_.num_modes(); }

    /// The raw, unwhitened evaluation underneath -- the geometry (frame, u) a
    /// caller may want without decoding theta_hat again.
    const FeatureAt& feature() const { return at_; }

    /// phi_hat: (K, num_modes).
    const Eigen::MatrixXd& values()
    {
        if ( !have_values_ )
        {
            have_values_ = true;
            values_ = detail::scale_rows(at_.values(), scale_);
        }
        return values_;
    }

    /// Directional derivative: (K, num_modes). The scaling does not depend on
    /// the parameters, so it simply passes through.
    Eigen::MatrixXd jvp( const Eigen::Ref<const Eigen::VectorXd>& dtheta_hat )
    {
        return detail::scale_rows(at_.jvp(dtheta_hat), scale_);
    }

    /// The full parameter Jacobian: num_params of (K, num_modes).
    std::vector<Eigen::MatrixXd> jac()
    {
        std::vector<Eigen::MatrixXd> out;
        out.reserve(at_.jac().size());
        for ( const Eigen::MatrixXd& column : at_.jac() )
        {
            out.push_back(detail::scale_rows(column, scale_));
        }
        return out;
    }

    /// Reverse mode: (K, num_params), for a cotangent w_hat matching values().
    /// The scaling is symmetric, so it applies to the incoming cotangent and
    /// the raw chain then runs unchanged.
    Eigen::MatrixXd vjp( const Eigen::Ref<const Eigen::MatrixXd>& w_hat )
    {
        return at_.vjp(detail::scale_rows(w_hat, scale_));
    }

private:
    FeatureAt at_;
    Eigen::VectorXd scale_;
    Eigen::MatrixXd values_;
    bool have_values_ = false;
};

/// The whitened basis for one target: everything a fit holds fixed, with
/// theta_hat the only thing that varies.
///
/// `operator()` is the single callable the fitting layers consume -- one
/// callable rather than the eval/vjp/jac triple it replaced, because the three
/// share a theta_hat's worth of work that three independent closures could
/// not. Golub-Pereyra support is a CAPABILITY of the returned evaluation
/// (jac() being present), not a constructor option.
class WhitenedBasis
{
public:
    WhitenedBasis( Eigen::MatrixXd x, double target_mass, Eigen::VectorXd m2_diag,
                   std::vector<Mode> modes, Eigen::VectorXd mu0, MuMode mu_mode )
        : x_(std::move(x)), m2_diag_(std::move(m2_diag)), modes_(std::move(modes)),
          mu0_(std::move(mu0)), mu_mode_(mu_mode),
          scale_(detail::whitening_scale(target_mass, m2_diag_))
    {
        if ( x_.cols() != mu0_.size() )
        {
            throw std::invalid_argument(
                "lgpsf::WhitenedBasis: x has " + std::to_string(x_.cols())
                + " coordinate columns but mu0 has " + std::to_string(mu0_.size())
                + " entries");
        }
        if ( x_.rows() != m2_diag_.size() )
        {
            throw std::invalid_argument(
                "lgpsf::WhitenedBasis: x has " + std::to_string(x_.rows())
                + " points but m2_diag has " + std::to_string(m2_diag_.size())
                + " entries");
        }
    }

    int dim() const { return static_cast<int>(mu0_.size()); }
    int num_params() const { return theta_hat_size(dim(), mu_mode_); }
    std::size_t num_modes() const { return modes_.size(); }
    Eigen::Index num_points() const { return x_.rows(); }
    MuMode mu_mode() const { return mu_mode_; }

    WhitenedBasisAt operator()(
        const Eigen::Ref<const Eigen::VectorXd>& theta_hat ) const
    {
        return WhitenedBasisAt(FeatureAt(theta_hat, x_, modes_, mu0_, mu_mode_),
                               scale_);
    }

private:
    Eigen::MatrixXd x_;
    Eigen::VectorXd m2_diag_;
    std::vector<Mode> modes_;
    Eigen::VectorXd mu0_;
    MuMode mu_mode_;
    Eigen::VectorXd scale_;
};

/// phi_hat: (K, num_modes). Convenience wrapper around WhitenedBasis.
inline Eigen::MatrixXd whitened_eval_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x, double target_mass,
    const Eigen::Ref<const Eigen::VectorXd>& m2_diag,
    const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return WhitenedBasis(x, target_mass, m2_diag, modes, mu0, mu_mode)(theta_hat)
        .values();
}

/// Directional parameter derivative of phi_hat: (K, num_modes).
inline Eigen::MatrixXd whitened_jvp_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::VectorXd>& dtheta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x, double target_mass,
    const Eigen::Ref<const Eigen::VectorXd>& m2_diag,
    const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return WhitenedBasis(x, target_mass, m2_diag, modes, mu0, mu_mode)(theta_hat)
        .jvp(dtheta_hat);
}

/// The full parameter Jacobian of phi_hat: num_params of (K, num_modes).
inline std::vector<Eigen::MatrixXd> whitened_jac_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x, double target_mass,
    const Eigen::Ref<const Eigen::VectorXd>& m2_diag,
    const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return WhitenedBasis(x, target_mass, m2_diag, modes, mu0, mu_mode)(theta_hat)
        .jac();
}

/// <w_hat, d(phi_hat)/d(theta_hat)> per point: (K, num_params).
inline Eigen::MatrixXd whitened_vjp_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x, double target_mass,
    const Eigen::Ref<const Eigen::VectorXd>& m2_diag,
    const Eigen::Ref<const Eigen::MatrixXd>& w_hat,
    const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return WhitenedBasis(x, target_mass, m2_diag, modes, mu0, mu_mode)(theta_hat)
        .vjp(w_hat);
}

} // end namespace lgpsf
