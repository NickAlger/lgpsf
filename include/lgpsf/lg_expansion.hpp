#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief `LGExpansion`: one target's model as a Laguerre-Gaussian expansion on
/// a fitted ellipsoid, plus its discrete correction.
///
/// The object a row fit produces, and the per-row content of an `LGOperator`.
/// "Expansion" rather than "function": this is a SUM over the LG basis, not a
/// basis element, and the distinction is worth keeping in the name.
///
/// **Two components, as everywhere else in this library:**
///
///     f(x)  =  sum_i c_i phi_i(x; theta)     +     sum_d s_d e_d
///              \______ smooth, a function ____/    \__ discrete __/
///
/// `c` and `s` are held together because they are FITTED JOINTLY -- the
/// projection in variable projection couples them, so separating them at the
/// type level would imply an independence the mathematics denies. It is the
/// documentation, not the type system, that carries the distinction: `s`
/// weights a discrete, dof-tied correction with no off-grid meaning, exactly as
/// `S` does in `H~ = M1 Phi~ M2 + M1 S`.
///
/// **One asymmetry to be aware of and not to "fix".** `c` is self-describing --
/// its mode list travels with it. `s` is NOT: it weights the caller's extra
/// basis, which an expansion does not carry, because a mode list is tens of
/// integer triples while an extra basis is a `(K, num_extra)` array the caller
/// already owns. Storing the batch here would make every candidate of every row
/// carry a copy of it.
///
/// `theta` is the PUBLIC absolute encoding, so an expansion decodes on its own:
/// no reference center, no mu mode. See ellipsoid_transform.hpp.

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/ellipsoid_transform.hpp"
#include "lgpsf/lg_ellipsoid_feature.hpp"
#include "lgpsf/lg_functions.hpp"

namespace lgpsf {

struct LGExpansion
{
    /// (N(N+3)/2,) ellipsoid parameters in the PUBLIC absolute encoding.
    Eigen::VectorXd theta;

    /// The mode list `c` corresponds to, index for index.
    std::vector<Mode> modes;

    /// (num_modes,) smooth coefficients.
    Eigen::VectorXd c;

    /// (num_extra,) coefficients of the caller's extra basis -- the spike, in
    /// the operator-row application. Empty when there is no extra basis.
    Eigen::VectorXd s;

    int dim() const { return dim_from_theta_size(static_cast<int>(theta.size())); }
    std::size_t num_modes() const { return modes.size(); }

    /// The ellipsoid this expansion lives on.
    EllipsoidFrame frame() const { return unpack_theta(theta); }
};

/// The smooth component at arbitrary points: `sum_i c_i phi_i(x; theta)`.
///
/// No masses, no extra basis, and no truncation -- an expansion has no window,
/// that being an operator-layer concept. See `lg_operator.hpp` for what
/// deployment does with it, and why it truncates.
///
/// @param expansion The model.
/// @param x_query   Points to evaluate at, (Q, N).
/// @return          (Q,) the smooth component. The spike is excluded: it is a
///                  discrete dof correction with no meaning off the mesh.
inline Eigen::VectorXd eval_expansion(
    const LGExpansion& expansion, const Eigen::Ref<const Eigen::MatrixXd>& x_query )
{
    const std::pair<Eigen::VectorXd, Eigen::VectorXd> split = freeze_mu(expansion.theta);
    return eval_feature(split.first, x_query, expansion.modes, split.second,
                        MuMode::Pinned)
           * expansion.c;
}

/// Structural problems with an expansion, one message each.
///
/// Shape only -- whether it approximates anything is not a question this can
/// answer.
///
/// @param expansion The model to check.
/// @return          One message per problem; empty means self-consistent.
inline std::vector<std::string> validate( const LGExpansion& expansion )
{
    std::vector<std::string> problems;
    if ( expansion.theta.size() == 0 )
    {
        problems.push_back("theta is empty");
        return problems;
    }
    int dim = 0;
    try
    {
        dim = expansion.dim();
    }
    catch ( const std::invalid_argument& )
    {
        problems.push_back("theta's length is not N(N+3)/2 for any N, so it is "
                           "not the public absolute encoding");
        return problems;
    }
    if ( expansion.c.size() != static_cast<Eigen::Index>(expansion.modes.size()) )
    {
        problems.push_back("c must have one coefficient per mode");
    }
    for ( const Mode& mode : expansion.modes )
    {
        if ( mode.p < 0 || mode.ell < 0 || mode.m < 0
             || mode.ell > max_degree() || mode.m >= num_harmonics(dim, mode.ell) )
        {
            problems.push_back("the mode list contains an index outside the "
                               "generated harmonic table");
            break;
        }
    }
    if ( !expansion.theta.allFinite() || !expansion.c.allFinite()
         || !expansion.s.allFinite() )
    {
        problems.push_back("theta, c and s must be finite");
    }
    return problems;
}

} // end namespace lgpsf
