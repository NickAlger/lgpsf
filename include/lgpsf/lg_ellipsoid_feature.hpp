#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The theta-dependent smooth feature phi_i(x) = psi_i(T(x)) and its
/// parameter derivatives.
///
/// Purely a composition of lg_functions.hpp (the LG modes and their spatial
/// gradient) with ellipsoid_transform.hpp (the pullback and its parameter
/// derivatives), by the chain rule. This header owns no mathematics of its
/// own: it evaluates the pullback once, hands it to the LG basis, and performs
/// the contraction that distinguishes the four quantities. All mode-level
/// sharing lives in LGBasisAt; all parameter knowledge lives here.
///
/// Deliberately mass-free -- it knows nothing of M1/M2. See whitening.hpp for
/// the layer above, and docs/varpro-whitening-notes.tex for why the two
/// compose without either one touching the other's concern.
///
/// Point batches are (K, N) coordinate-major; feature values come back
/// (K, num_modes), one column per mode, matching LGBasisAt::values().

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/ellipsoid_transform.hpp"
#include "lgpsf/lg_functions.hpp"

namespace lgpsf {

/// The feature basis evaluated at one theta_hat: the pullback and the LG
/// evaluation on it, shared between the values and every derivative.
///
/// The VarPro loop needs the values and a derivative at the SAME parameters
/// with a linear solve in between -- the Kaufman cotangent is built from the
/// values -- so the two cannot be fused into one value-and-gradient call. An
/// object with an explicit lifetime is what that dependency calls for instead:
/// build one per trial theta_hat, ask it for what the step needs, discard it.
///
/// It does not retain `x`. The pullback's derivatives depend on the points
/// only through u (see ellipsoid_transform.hpp), so once u is formed the
/// physical coordinates are no longer needed by anything here.
class FeatureAt
{
public:
    FeatureAt( const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
               const Eigen::Ref<const Eigen::MatrixXd>& x,
               std::vector<Mode> modes,
               const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
        : frame_(unpack_theta_hat(theta_hat, mu0, mu_mode)),
          u_(pullback(frame_, x)),
          lg_(std::move(modes), u_),
          mu_mode_(mu_mode),
          num_params_(theta_hat_size(frame_.dim(), mu_mode))
    {
    }

    int dim() const { return frame_.dim(); }
    int num_params() const { return num_params_; }
    std::size_t num_modes() const { return lg_.modes().size(); }
    Eigen::Index num_points() const { return u_.rows(); }

    /// The decoded ellipsoid and the pullback, for callers that want the
    /// geometry this evaluation was built on (the fit's containment checks,
    /// for one) without decoding theta_hat a second time.
    const EllipsoidFrame& frame() const { return frame_; }
    const Eigen::MatrixXd& u() const { return u_; }

    /// phi_i at every point: (K, num_modes), one column per mode.
    const Eigen::MatrixXd& values() { return lg_.values(); }

    /// Directional derivative with respect to theta_hat: (K, num_modes).
    ///
    /// The chain rule d/dtheta psi_i(T(theta, x)) = grad_u psi_i . dT/dtheta,
    /// contracted per mode over the spatial axis.
    Eigen::MatrixXd jvp( const Eigen::Ref<const Eigen::VectorXd>& dtheta_hat )
    {
        return contract(
            pullback_jvp(frame_, jvp_unpack_theta_hat(frame_, dtheta_hat, mu_mode_),
                         u_));
    }

    /// The full parameter Jacobian: num_params matrices of (K, num_modes),
    /// entry q being d(values)/d(theta_hat_q) -- exactly jvp() against the q-th
    /// unit direction.
    ///
    /// This is what the exact Golub-Pereyra VarPro variant consumes; its second
    /// term has no reverse-mode collapse, so it needs the uncontracted tensor.
    /// The default Kaufman variant goes through vjp() instead.
    ///
    /// Indexed by PARAMETER rather than by mode, because that is how the
    /// consumer slices it -- one Jacobian column at a time -- so no transpose
    /// is needed at the point of use. The per-mode spatial gradients do not
    /// depend on the direction, so they are computed once here rather than
    /// once per direction as a loop over jvp() would.
    const std::vector<Eigen::MatrixXd>& jac()
    {
        if ( have_jacobian_ )
        {
            return jacobian_;
        }
        have_jacobian_ = true;
        jacobian_.reserve(static_cast<std::size_t>(num_params_));

        Eigen::VectorXd direction = Eigen::VectorXd::Zero(num_params_);
        for ( int q = 0; q < num_params_; ++q )
        {
            direction(q) = 1.0;
            jacobian_.push_back(contract(
                pullback_jvp(frame_, jvp_unpack_theta_hat(frame_, direction, mu_mode_),
                             u_)));
            direction(q) = 0.0;
        }
        return jacobian_;
    }

    /// Reverse mode: <w, d(values)/d(theta_hat)> per point, for a cotangent w
    /// of shape (K, num_modes). Returns (K, num_params).
    ///
    /// Every mode's spatial gradient, weighted by its own cotangent entry,
    /// collapses into ONE u-space cotangent inside LGBasisAt::vjp (which
    /// regroups by shell, so the per-mode gradient tensor is never built), and
    /// that single cotangent is then pushed back through the pullback.
    Eigen::MatrixXd vjp( const Eigen::Ref<const Eigen::MatrixXd>& w )
    {
        return vjp_unpack_theta_hat(
            frame_, pullback_vjp(frame_, u_, lg_.vjp(w)), mu_mode_);
    }

private:
    /// Per mode, contract that mode's spatial gradient against a shared
    /// u-space tangent: out.col(i) = sum_d grad_i[:, d] * du[:, d].
    Eigen::MatrixXd contract( const Eigen::MatrixXd& du )
    {
        const std::vector<Eigen::MatrixXd>& grad = lg_.grad();
        Eigen::MatrixXd out(u_.rows(), static_cast<Eigen::Index>(grad.size()));
        for ( std::size_t i = 0; i < grad.size(); ++i )
        {
            out.col(static_cast<Eigen::Index>(i)) =
                (grad[i].array() * du.array()).rowwise().sum();
        }
        return out;
    }

    EllipsoidFrame frame_;
    Eigen::MatrixXd u_;
    LGBasisAt lg_;
    MuMode mu_mode_;
    int num_params_;

    std::vector<Eigen::MatrixXd> jacobian_;
    bool have_jacobian_ = false;
};

/// phi_i(x) = psi_i(T(x)) for each mode: (K, num_modes). Convenience wrapper;
/// use FeatureAt directly when values and a derivative are both needed.
inline Eigen::MatrixXd eval_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x, const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return FeatureAt(theta_hat, x, modes, mu0, mu_mode).values();
}

/// Directional parameter derivative of eval_feature: (K, num_modes).
inline Eigen::MatrixXd jvp_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::VectorXd>& dtheta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x, const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return FeatureAt(theta_hat, x, modes, mu0, mu_mode).jvp(dtheta_hat);
}

/// The full parameter Jacobian of eval_feature: num_params of (K, num_modes).
inline std::vector<Eigen::MatrixXd> jac_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x, const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return FeatureAt(theta_hat, x, modes, mu0, mu_mode).jac();
}

/// <w, d(eval_feature)/d(theta_hat)> per point, for w of shape
/// (K, num_modes). Returns (K, num_params).
inline Eigen::MatrixXd vjp_feature(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& w, const std::vector<Mode>& modes,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mu_mode )
{
    return FeatureAt(theta_hat, x, modes, mu0, mu_mode).vjp(w);
}

} // end namespace lgpsf
