#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Real Laguerre-Gaussian basis functions: the smooth part of the
/// LG-PSF kernel model, with no ellipsoid or theta dependence.
///
/// The mode (p, ell, m) is the product of three separable factors,
///
///     psi_{p,ell,m}(u) = C_{p,ell,N} Y_{ell,m}(u) L_p^alpha(r^2) exp(-r^2/2),
///     alpha = ell + N/2 - 1,
///
/// the eigenfunctions of the N-dimensional quantum harmonic oscillator,
/// orthonormal in L^2(R^N) with the Gaussian envelope part of the function
/// itself (as for Hermite functions), not a separate weight.
///
/// The mode index FACTORIZES -- the angular factor is independent of p, the
/// radial factor independent of m, the Gaussian independent of both -- so a
/// mode SET, not a single mode, is the natural unit of evaluation. LGBasisAt
/// is the production entry point; the one-at-a-time functions are the
/// readable reference it is pinned against.
///
/// Point batches are (K, N) throughout: points as ROWS, coordinates across.
/// (The Python bindings take (N, K); the two describe the same bytes. See
/// docs/api-guide.md.)

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/harmonic_polynomials.hpp"

namespace lgpsf {

/// One Laguerre-Gaussian mode: radial order p, angular degree ell, and the
/// index m of the harmonic within that degree. Oscillator level is 2p + ell.
struct Mode
{
    int p   = 0;
    int ell = 0;
    int m   = 0;

    friend bool operator==( const Mode& a, const Mode& b )
    {
        return a.p == b.p && a.ell == b.ell && a.m == b.m;
    }

    friend bool operator!=( const Mode& a, const Mode& b ) { return !(a == b); }

    /// Lexicographic by (p, ell, m) -- an arbitrary but total order, so modes
    /// can key an ordered set. Carries no meaning: energy order is 2p + ell.
    friend bool operator<( const Mode& a, const Mode& b )
    {
        if ( a.p != b.p ) return a.p < b.p;
        if ( a.ell != b.ell ) return a.ell < b.ell;
        return a.m < b.m;
    }
};

/// Generalized Laguerre polynomial L_p^alpha(x), evaluated elementwise, via
/// the standard three-term recurrence
///
///     L_0^alpha = 1,  L_1^alpha = 1 + alpha - x,
///     (k+1) L_{k+1}^alpha = (2k+1+alpha-x) L_k^alpha - (k+alpha) L_{k-1}^alpha.
///
/// Written out rather than called from a special-function library, because
/// C++ has none to call.
///
/// @param p     Polynomial order, >= 0.
/// @param alpha Order parameter; may be a half-integer.
/// @param x     Points to evaluate at.
/// @return      L_p^alpha(x), same length as @p x.
inline Eigen::VectorXd genlaguerre( int p, double alpha,
                                    const Eigen::Ref<const Eigen::VectorXd>& x )
{
    if ( p < 0 )
    {
        return Eigen::VectorXd::Zero(x.size());
    }
    Eigen::VectorXd previous = Eigen::VectorXd::Ones(x.size());
    if ( p == 0 )
    {
        return previous;
    }
    Eigen::VectorXd current = (1.0 + alpha) - x.array();
    for ( int k = 1; k < p; ++k )
    {
        Eigen::VectorXd next =
            (((2 * k + 1 + alpha) - x.array()) * current.array()
             - (k + alpha) * previous.array()) / (k + 1);
        previous = std::move(current);
        current = std::move(next);
    }
    return current;
}

/// The combining constant C_{p,ell,N} that makes psi_{p,ell,m} orthonormal in
/// L^2(R^N) given a surface-orthonormal Y_{ell,m}:
/// C = sqrt(2 p! / Gamma(p + alpha + 1)), alpha = ell + N/2 - 1. alpha is an
/// integer or a half-integer, so the gamma is elementary.
///
/// @param p    Radial order.
/// @param ell  Angular degree.
/// @param dim  Spatial dimension N.
/// @return     The scalar constant C_{p,ell,N}.
inline double lg_norm( int p, int ell, int dim )
{
    const double alpha = ell + dim / 2.0 - 1.0;
    return std::sqrt(2.0 * std::tgamma(p + 1.0) / std::tgamma(p + alpha + 1.0));
}

/// All (p, ell, m) with oscillator level 2p + ell <= max_level -- complete
/// shells in energy order, the single-knob mode family used for nested mode
/// ladders. `ell_max >= 0` caps the angular order (an ell-capped WEDGE: deep
/// radial, shallow angular); a negative value means no cap.
///
/// @param dim       Spatial dimension N, 1 to 4.
/// @param max_level Highest oscillator level 2p + ell to include.
/// @param ell_max   Cap on angular order; negative means no cap.
/// @return          The modes, in ascending level order.
/// @throws std::invalid_argument if the requested orders exceed the generated
///         harmonic table, rather than silently returning a truncated set.
inline std::vector<Mode> modes_up_to_level( int dim, int max_level,
                                            int ell_max = -1 )
{
    const int top = ( ell_max < 0 ) ? max_level : std::min(max_level, ell_max);
    if ( top > max_degree() )
    {
        throw std::invalid_argument(
            "lgpsf::modes_up_to_level: needs harmonics up to ell="
            + std::to_string(top) + ", but the generated table stops at ell="
            + std::to_string(max_degree())
            + "; lower max_level, set ell_max, or extend the table");
    }
    std::vector<Mode> modes;
    for ( int ell = 0; ell <= top; ++ell )
    {
        for ( int m = 0; m < num_harmonics(dim, ell); ++m )
        {
            for ( int p = 0; p <= (max_level - ell) / 2; ++p )
            {
                modes.push_back(Mode{p, ell, m});
            }
        }
    }
    return modes;
}

/// The LG basis for one mode set, evaluated at one point batch.
///
/// Holds everything independent of WHICH quantity is asked for -- r^2, the
/// Gaussian, the shell index, the harmonics and the Laguerre tables -- so
/// values, gradients and vector-Jacobian products at the same points share
/// them. That matters because the VarPro loop needs the values and a
/// derivative at the SAME theta with a linear solve in between (the Kaufman
/// cotangent is built from the values), so the two cannot be fused into a
/// single value-and-gradient call but can and should share one evaluation.
///
/// Construct one per point batch, ask it for what you need, discard it.
class LGBasisAt
{
public:
    LGBasisAt( std::vector<Mode> modes, Eigen::MatrixXd points )
        : modes_(std::move(modes)), u_(std::move(points))
    {
        dim_ = static_cast<int>(u_.cols());
        r2_ = u_.rowwise().squaredNorm();
        gauss_ = (-0.5 * r2_.array()).exp();
        for ( const Mode& mode : modes_ )
        {
            const std::pair<int, int> key{mode.ell, mode.m};
            auto found = shell_position_.find(key);
            if ( found == shell_position_.end() )
            {
                shell_position_.emplace(key, static_cast<int>(shells_.size()));
                shell_of_.push_back(static_cast<int>(shells_.size()));
                shells_.push_back(Shell{mode.ell, mode.m});
            }
            else
            {
                shell_of_.push_back(found->second);
            }
        }
    }

    /// Spatial dimension N.
    int dim() const { return dim_; }
    /// Number of points K in the batch.
    Eigen::Index num_points() const { return u_.rows(); }
    /// The mode set, in the order the columns of values() follow.
    const std::vector<Mode>& modes() const { return modes_; }

    /// @return psi_i at every point: (K, num_modes), one column per mode, in
    ///         mode order. Cached; repeated calls are free.
    const Eigen::MatrixXd& values()
    {
        if ( have_values_ )
        {
            return values_;
        }
        have_values_ = true;
        if ( modes_.empty() )
        {
            values_.resize(u_.rows(), 0);
            return values_;
        }
        ensure_harmonics(false);
        values_.resize(u_.rows(), static_cast<Eigen::Index>(modes_.size()));
        for ( std::size_t i = 0; i < modes_.size(); ++i )
        {
            const Mode& mode = modes_[i];
            values_.col(static_cast<Eigen::Index>(i)) =
                lg_norm(mode.p, mode.ell, dim_)
                * harmonics_.values.col(shell_of_[i]).cwiseProduct(
                      radial_value(mode.p, mode.ell));
        }
        return values_;
    }

    /// Per-mode spatial gradients, UNCONTRACTED.
    ///
    /// @return num_modes matrices of shape (K, N), in mode order. Cached.
    ///
    /// Needed by the exact Golub-Pereyra VarPro variant and by the
    /// theta-Jacobian; prefer vjp() when the result is only contracted, since
    /// this materializes the whole tensor.
    const std::vector<Eigen::MatrixXd>& grad()
    {
        if ( have_gradients_ )
        {
            return gradients_;
        }
        have_gradients_ = true;
        if ( modes_.empty() )
        {
            return gradients_;
        }
        ensure_harmonics(true);
        gradients_.clear();
        gradients_.reserve(modes_.size());
        for ( std::size_t i = 0; i < modes_.size(); ++i )
        {
            const Mode& mode = modes_[i];
            const Eigen::VectorXd& L = laguerre(mode.ell, mode.p)[mode.p];
            const Eigen::VectorXd& envelope = radial_envelope(mode.p, mode.ell);
            const Eigen::VectorXd& Y = harmonics_.values.col(shell_of_[i]);
            const Eigen::MatrixXd& dY = harmonics_.gradients[shell_of_[i]];
            const double prefactor = lg_norm(mode.p, mode.ell, dim_);

            Eigen::MatrixXd out =
                (dY.array().colwise() * L.array())
                - (u_.array().colwise() * Y.cwiseProduct(envelope).array());
            out.array().colwise() *= (prefactor * gauss_.array());
            gradients_.push_back(std::move(out));
        }
        return gradients_;
    }

    /// Vector-Jacobian product: the gradient of a weighted sum of modes.
    ///
    /// @param w Per-point weights, (K, num_modes).
    /// @return  grad_u sum_i w_i psi_i, shape (K, N).
    ///
    /// Regrouped by SHELL rather than accumulating per mode. Substituting the
    /// product rule and collecting terms:
    ///
    ///     sum_i w_i grad psi_i = g [ sum_s A_s grad_Y_s - u sum_s Y_s B_s ]
    ///     A_s = sum_{i in s} C_i w_i L_{p_i}
    ///     B_s = sum_{i in s} C_i w_i (L_{p_i} - 2 L'_{p_i})
    ///
    /// A_s and B_s are scalar fields, so only the harmonic gradients carry the
    /// N axis: one (K, N) multiply-add per SHELL instead of several per MODE,
    /// the u term collapses to a single multiply, and the per-mode gradient
    /// tensor is never materialized.
    ///
    /// NOT bit-identical to contracting grad() -- the regrouping reassociates
    /// the sum. Accuracy is a wash, not an improvement; the justification is
    /// operation count and memory.
    Eigen::MatrixXd vjp( const Eigen::Ref<const Eigen::MatrixXd>& w )
    {
        if ( modes_.empty() )
        {
            return Eigen::MatrixXd::Zero(u_.rows(), dim_);
        }
        if ( w.rows() != u_.rows()
             || w.cols() != static_cast<Eigen::Index>(modes_.size()) )
        {
            throw std::invalid_argument(
                "lgpsf::LGBasisAt::vjp: w must be (num_points, num_modes)");
        }
        ensure_harmonics(true);

        const Eigen::Index n_shells = static_cast<Eigen::Index>(shells_.size());
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(u_.rows(), n_shells);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(u_.rows(), n_shells);
        for ( std::size_t i = 0; i < modes_.size(); ++i )
        {
            const Mode& mode = modes_[i];
            const double prefactor = lg_norm(mode.p, mode.ell, dim_);
            const Eigen::VectorXd weight =
                prefactor * w.col(static_cast<Eigen::Index>(i));
            A.col(shell_of_[i]) +=
                weight.cwiseProduct(laguerre(mode.ell, mode.p)[mode.p]);
            B.col(shell_of_[i]) +=
                weight.cwiseProduct(radial_envelope(mode.p, mode.ell));
        }

        Eigen::MatrixXd angular = Eigen::MatrixXd::Zero(u_.rows(), dim_);
        Eigen::VectorXd radial = Eigen::VectorXd::Zero(u_.rows());
        for ( Eigen::Index s = 0; s < n_shells; ++s )
        {
            angular.array() +=
                harmonics_.gradients[static_cast<std::size_t>(s)].array().colwise()
                * A.col(s).array();
            radial += harmonics_.values.col(s).cwiseProduct(B.col(s));
        }
        Eigen::MatrixXd out = angular - (u_.array().colwise() * radial.array()).matrix();
        out.array().colwise() *= gauss_.array();
        return out;
    }

private:
    void ensure_harmonics( bool with_gradient )
    {
        if ( with_gradient )
        {
            if ( harmonics_.gradients.empty() )
            {
                // If values() already ran, tell harmonic_basis so it skips
                // recomputing a bit-identical set of values -- the
                // values-then-derivative order is exactly the VarPro pattern.
                const Eigen::MatrixXd* known =
                    harmonics_.values.size() != 0 ? &harmonics_.values : nullptr;
                HarmonicBasis fresh = harmonic_basis(shells_, u_, true, known);
                harmonics_ = std::move(fresh);
            }
            return;
        }
        if ( harmonics_.values.size() == 0 )
        {
            harmonics_ = harmonic_basis(shells_, u_, false);
        }
    }

    /// L_p^alpha(ell) for p = 0..max_p, extending any existing table in place.
    /// The recurrence continues from the stored values, so an extension gives
    /// exactly what building the deeper table up front would have.
    ///
    /// std::deque, not std::vector: callers hold references to entries across
    /// calls that may extend a table, and deque keeps element references valid
    /// on push_back where vector would reallocate them out from under us.
    const std::deque<Eigen::VectorXd>& laguerre( int ell, int max_p )
    {
        std::deque<Eigen::VectorXd>& table = laguerre_[ell];
        const double alpha = ell + dim_ / 2.0 - 1.0;
        if ( table.empty() )
        {
            table.push_back(Eigen::VectorXd::Ones(u_.rows()));
        }
        if ( max_p >= 1 && table.size() == 1u )
        {
            table.push_back(Eigen::VectorXd((1.0 + alpha) - r2_.array()));
        }
        for ( int k = static_cast<int>(table.size()) - 1; k < max_p; ++k )
        {
            table.push_back(Eigen::VectorXd(
                (((2 * k + 1 + alpha) - r2_.array()) * table[k].array()
                 - (k + alpha) * table[k - 1].array()) / (k + 1)));
        }
        return table;
    }

    /// L_p^alpha(ell)(r^2) * exp(-r^2/2), cached per (p, ell) since it is
    /// shared across every m in that shell.
    const Eigen::VectorXd& radial_value( int p, int ell )
    {
        const std::pair<int, int> key{p, ell};
        auto found = radial_value_.find(key);
        if ( found == radial_value_.end() )
        {
            found = radial_value_.emplace(
                key, laguerre(ell, p)[p].cwiseProduct(gauss_)).first;
        }
        return found->second;
    }

    /// L - 2 L', where d/dt L_p^alpha = -L_{p-1}^(alpha+1) and
    /// alpha(ell) + 1 = alpha(ell + 1) exactly -- so the derivative comes from
    /// the ell + 1 table and no separate recurrence family is needed.
    const Eigen::VectorXd& radial_envelope( int p, int ell )
    {
        const std::pair<int, int> key{p, ell};
        auto found = radial_envelope_.find(key);
        if ( found == radial_envelope_.end() )
        {
            Eigen::VectorXd envelope = laguerre(ell, p)[p];
            if ( p >= 1 )
            {
                envelope += 2.0 * laguerre(ell + 1, p - 1)[p - 1];
            }
            found = radial_envelope_.emplace(key, std::move(envelope)).first;
        }
        return found->second;
    }

    std::vector<Mode> modes_;
    Eigen::MatrixXd u_;
    int dim_ = 0;
    Eigen::VectorXd r2_;
    Eigen::VectorXd gauss_;

    std::vector<Shell> shells_;
    std::vector<int> shell_of_;
    std::map<std::pair<int, int>, int> shell_position_;

    HarmonicBasis harmonics_;
    std::map<int, std::deque<Eigen::VectorXd>> laguerre_;
    std::map<std::pair<int, int>, Eigen::VectorXd> radial_value_;
    std::map<std::pair<int, int>, Eigen::VectorXd> radial_envelope_;

    Eigen::MatrixXd values_;
    std::vector<Eigen::MatrixXd> gradients_;
    bool have_values_ = false;
    bool have_gradients_ = false;
};

/// Values of a whole mode set.
///
/// Convenience wrapper; construct an LGBasisAt directly when values and a
/// derivative are both needed at the same points, so they share the work.
///
/// @param modes The mode set.
/// @param u     Points, (K, N).
/// @return      (K, num_modes), one column per mode, in @p modes order.
inline Eigen::MatrixXd eval_lg_basis( const std::vector<Mode>& modes,
                                      const Eigen::Ref<const Eigen::MatrixXd>& u )
{
    return LGBasisAt(modes, u).values();
}

/// Per-mode spatial gradients of a whole mode set.
///
/// @param modes The mode set.
/// @param u     Points, (K, N).
/// @return      num_modes matrices of shape (K, N), in @p modes order.
inline std::vector<Eigen::MatrixXd> grad_lg_basis(
    const std::vector<Mode>& modes, const Eigen::Ref<const Eigen::MatrixXd>& u )
{
    return LGBasisAt(modes, u).grad();
}

/// Vector-Jacobian product: the gradient of a weighted sum of modes.
///
/// @param modes The mode set.
/// @param u     Points, (K, N).
/// @param w     Per-point weights, (K, num_modes).
/// @return      grad_u sum_i w_i psi_i, shape (K, N).
inline Eigen::MatrixXd vjp_lg_basis( const std::vector<Mode>& modes,
                                     const Eigen::Ref<const Eigen::MatrixXd>& u,
                                     const Eigen::Ref<const Eigen::MatrixXd>& w )
{
    return LGBasisAt(modes, u).vjp(w);
}

/// One mode at every point.
///
/// The readable reference the batched path is pinned against; prefer
/// eval_lg_basis() for more than one mode.
///
/// @param p,ell,m The mode.
/// @param u       Points, (K, N).
/// @return        psi_{p,ell,m} at each point, (K,).
inline Eigen::VectorXd eval_lg_nd( int p, int ell, int m,
                                   const Eigen::Ref<const Eigen::MatrixXd>& u )
{
    return LGBasisAt({Mode{p, ell, m}}, u).values().col(0);
}

/// Spatial gradient of one mode at every point.
///
/// @param p,ell,m The mode.
/// @param u       Points, (K, N).
/// @return        grad psi_{p,ell,m} at each point, (K, N).
inline Eigen::MatrixXd grad_eval_lg_nd( int p, int ell, int m,
                                        const Eigen::Ref<const Eigen::MatrixXd>& u )
{
    return LGBasisAt({Mode{p, ell, m}}, u).grad()[0];
}

/// p! / (p+a)!, as a product of reciprocals so every partial product stays in
/// [0, 1] -- an integer loop rather than a ratio of two overflowing gammas.
///
/// @param p Base.
/// @param a Non-negative offset.
/// @return  p! / (p+a)!.
inline double factorial_ratio( int p, int a )
{
    double ratio = 1.0;
    for ( int k = p + 1; k <= p + a; ++k )
    {
        ratio /= k;
    }
    return ratio;
}

/// The 2D-only reference mode, in the original research note's convention:
/// the sign of `ell` selects the real branch (ell > 0 -> cos(ell theta),
/// ell < 0 -> sin(|ell| theta), ell == 0 -> the angle-independent mode).
///
/// Kept as the independent definition of the N = 2 convention -- the
/// N-dimensional path indexes the same space by (ell >= 0, m) with no
/// canonical cos/sin labelling, and the two are cross-checked in the tests.
///
/// @param p      Radial order.
/// @param ell    Signed angular order; the sign picks the branch.
/// @param u1,u2  The two coordinates, equal length.
/// @return       The mode at each point, same length as @p u1.
inline Eigen::VectorXd eval_lg( int p, int ell,
                                const Eigen::Ref<const Eigen::VectorXd>& u1,
                                const Eigen::Ref<const Eigen::VectorXd>& u2 )
{
    if ( p < 0 )
    {
        throw std::invalid_argument("lgpsf::eval_lg: p must be >= 0, got "
                                    + std::to_string(p));
    }
    const int a = std::abs(ell);
    const Eigen::VectorXd r2 = u1.array().square() + u2.array().square();

    double norm = std::sqrt(factorial_ratio(p, a) / M_PI);
    if ( a > 0 )
    {
        norm *= std::sqrt(2.0);
    }

    const Eigen::VectorXd radial =
        r2.array().pow(a / 2.0) * genlaguerre(p, a, r2).array()
        * (-0.5 * r2.array()).exp();

    if ( ell == 0 )
    {
        return norm * radial;
    }
    Eigen::VectorXd angular(u1.size());
    for ( Eigen::Index i = 0; i < u1.size(); ++i )
    {
        const double theta = std::atan2(u2(i), u1(i));
        angular(i) = ( ell > 0 ) ? std::cos(a * theta) : std::sin(a * theta);
    }
    return norm * radial.cwiseProduct(angular);
}

} // end namespace lgpsf
