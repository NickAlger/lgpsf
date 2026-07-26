#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief The ellipsoid pullback T(x) = L^{-1}(x - mu), its parameter
/// derivatives, and the two encodings the fit moves between.
///
/// T maps a physical point x into the normalized coordinate u that the LG
/// basis (lg_functions.hpp) expects, with Sigma = L L^T the local covariance
/// ellipsoid. Built as two composable stages, so that changing how the
/// parameters encode (mu, L) never touches the geometry:
///
///   Stage 1 (mu, L, x -> u): pullback / pullback_jvp / pullback_vjp. The only
///     code that touches the ellipsoid geometry. Solves against L through an
///     explicit inverse formed ONCE per EllipsoidFrame -- the parameters are
///     never batched, so this is negligible (N <= 4) and turns every solve
///     into a single batched contraction over the point axis.
///   Stage 2 (parameters -> (mu, L)): unpack_theta_hat and its JVP/VJP. A
///     log-Cholesky encoding: N log-diagonal entries so L_ii > 0 by
///     construction, N(N-1)/2 free strictly-lower entries.
///
/// eval_T / jvp_T / vjp_T chain the two exactly as an autodiff system composes
/// primitive rules -- forward mode pushes a tangent through stage 2 then stage
/// 1, reverse mode pulls a cotangent through stage 1 then stage 2.
///
/// ## The two encodings
///
/// **`theta` -- the public one.** `[mu, log-diag(L), strict-lower(L)]`, always
/// length `theta_size(N) = N(N+3)/2`, whether or not the fit that produced it
/// was free to move mu. It decodes with `unpack_theta(theta)` alone: no mu0,
/// no mode, nothing else stored alongside. That is what lets a whole operator
/// fit hand back one flat theta array that every row decodes uniformly, and it
/// is why the operator layer can default mu0 to the source points without the
/// returned parameters becoming meaningless.
///
/// **`theta_hat` -- the internal one**, and the only encoding the fitting core
/// ever sees. It is `theta` in coordinates centered on the caller's reference
/// center mu0: the leading block is the DISPLACEMENT `delta = mu - mu0` when
/// mu is fitted, and is absent entirely when mu is pinned. Two reasons:
///
///   1. Scale. Absolute centers carry the mesh's physical coordinates (of
///      order 1e6 m for the ice-sheet problem this method came from) while the
///      log-diagonal is of order 1 -- six orders of magnitude apart in one
///      vector the trust region and `xtol` both act on. Displacements are of
///      order the local ellipsoid, like everything else in theta_hat.
///   2. A pinned mu then has no parameters at all rather than N frozen ones,
///      so the reduced problem is genuinely smaller and its Jacobian has no
///      identically-zero columns for the LM's column-norm scaling to trip on.
///
/// The `_hat` suffix is this codebase's mark for "in the coordinates the
/// numerical core works in" (see whitening.hpp: z_hat, y_hat, e_hat, phi_hat),
/// so every input crossing into varpro.hpp is hatted and every conversion back
/// out is confined to a boundary. Note the transform differs -- mass whitening
/// there, an mu0-shift and mode reduction here -- but the role is the same.
///
/// Point batches are (K, N): coordinate-major, so `u.col(d)` is coordinate d
/// across the whole batch, contiguous under Eigen's column-major default. Same
/// bytes as the Python prototype's (N, K) row-major arrays; see
/// docs/design-notes.md. The parameter vectors are NEVER batched, so the loops
/// over N and P here stay plain scalar loops by design.

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace lgpsf {

/// Whether the ellipsoid center is a fitted parameter or held at the caller's
/// reference center mu0. Selects which `theta_hat` encoding is in play.
enum class MuMode
{
    Pinned,  ///< mu == mu0; theta_hat carries no center block.
    Fitted   ///< mu == mu0 + theta_hat.head(N); the leading block is a displacement.
};

/// Number of parameters in the public encoding `theta`: N (mu) + N (the
/// log-diagonal of L) + N(N-1)/2 (L's strictly-lower entries).
inline int theta_size( int dim )
{
    return 2 * dim + dim * (dim - 1) / 2;
}

/// Number of parameters in the internal encoding `theta_hat`: the same, minus
/// the N center parameters when mu is pinned.
inline int theta_hat_size( int dim, MuMode mode )
{
    return ( mode == MuMode::Fitted ) ? theta_size(dim)
                                      : dim + dim * (dim - 1) / 2;
}

/// The spatial dimension whose public encoding has `size` parameters.
///
/// `theta_size` is strictly increasing in N, so this inverts it; the public
/// encoding being self-describing is what makes `unpack_theta` need nothing
/// but the vector itself.
inline int dim_from_theta_size( int size )
{
    for ( int dim = 1; dim <= 64; ++dim )
    {
        if ( theta_size(dim) == size )
        {
            return dim;
        }
        if ( theta_size(dim) > size )
        {
            break;
        }
    }
    throw std::invalid_argument(
        "lgpsf::ellipsoid_transform: " + std::to_string(size)
        + " is not a valid theta length (N(N+3)/2 for some N >= 1)");
}

/// A decoded ellipsoid: center, Cholesky factor, and its inverse.
///
/// L_inv is formed once here rather than at each use. The parameters are never
/// batched, so one triangular inversion per decode is free next to the batched
/// work it serves, and it turns every subsequent "solve against L" into a
/// single contraction over the point axis.
struct EllipsoidFrame
{
    Eigen::VectorXd mu;     ///< (N,)
    Eigen::MatrixXd L;      ///< (N, N) lower-triangular, positive diagonal
    Eigen::MatrixXd L_inv;  ///< (N, N) lower-triangular

    int dim() const { return static_cast<int>(mu.size()); }
};

/// Build a frame from an explicit center and Cholesky factor, inverting L by
/// forward substitution (exact for the triangular structure, unlike a general
/// LU). Validates eagerly: L must be square, lower-triangular, and have a
/// finite positive diagonal.
inline EllipsoidFrame make_frame( Eigen::VectorXd mu, Eigen::MatrixXd L )
{
    const Eigen::Index n = mu.size();
    if ( L.rows() != n || L.cols() != n )
    {
        throw std::invalid_argument(
            "lgpsf::make_frame: L must be (" + std::to_string(n) + ", "
            + std::to_string(n) + ") to match mu, got (" + std::to_string(L.rows())
            + ", " + std::to_string(L.cols()) + ")");
    }
    for ( Eigen::Index i = 0; i < n; ++i )
    {
        if ( !(L(i, i) > 0.0) || !std::isfinite(L(i, i)) )
        {
            throw std::invalid_argument(
                "lgpsf::make_frame: L's diagonal must be finite and positive; "
                "entry " + std::to_string(i) + " is " + std::to_string(L(i, i)));
        }
        for ( Eigen::Index j = i + 1; j < n; ++j )
        {
            if ( L(i, j) != 0.0 )
            {
                throw std::invalid_argument(
                    "lgpsf::make_frame: L must be lower-triangular; entry ("
                    + std::to_string(i) + ", " + std::to_string(j)
                    + ") is nonzero (did you pass Sigma instead of its Cholesky "
                      "factor?)");
            }
        }
    }

    EllipsoidFrame frame;
    frame.mu = std::move(mu);
    frame.L  = std::move(L);
    frame.L_inv = frame.L.template triangularView<Eigen::Lower>().solve(
        Eigen::MatrixXd::Identity(n, n));
    return frame;
}

// --------------------------------------------------------------------------
// The encodings: decoding, and conversion between them.
// --------------------------------------------------------------------------

namespace detail {

/// Fill L from a log-diagonal block followed by strictly-lower entries in
/// row-major order (1,0), (2,0), (2,1), (3,0), ... -- the shared tail layout
/// of both encodings.
inline Eigen::MatrixXd cholesky_from_block(
    const Eigen::Ref<const Eigen::VectorXd>& block, int dim )
{
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(dim, dim);
    for ( int i = 0; i < dim; ++i )
    {
        L(i, i) = std::exp(block(i));
    }
    int idx = dim;
    for ( int i = 1; i < dim; ++i )
    {
        for ( int j = 0; j < i; ++j )
        {
            L(i, j) = block(idx++);
        }
    }
    return L;
}

inline void check_size( const char* what, Eigen::Index got, int want )
{
    if ( got != want )
    {
        throw std::invalid_argument(
            std::string("lgpsf::ellipsoid_transform: ") + what + " must have "
            + std::to_string(want) + " entries, got " + std::to_string(got));
    }
}

} // end namespace detail

/// Decode the public encoding. Needs nothing but the vector: N is recovered
/// from its length and the center is carried explicitly.
inline EllipsoidFrame unpack_theta( const Eigen::Ref<const Eigen::VectorXd>& theta )
{
    const int dim = dim_from_theta_size(static_cast<int>(theta.size()));
    return make_frame(theta.head(dim),
                      detail::cholesky_from_block(theta.tail(theta.size() - dim), dim));
}

/// Decode the internal encoding against the reference center mu0.
///
/// Throws if the log-diagonal has run away far enough to overflow L (an
/// extreme trial step, not a bug); the fitting core treats that the way it
/// treats a non-finite design matrix, by scoring the point as unusable and
/// letting the outer loop back off.
inline EllipsoidFrame unpack_theta_hat(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mode )
{
    const int dim = static_cast<int>(mu0.size());
    detail::check_size("theta_hat", theta_hat.size(), theta_hat_size(dim, mode));

    if ( mode == MuMode::Pinned )
    {
        return make_frame(mu0, detail::cholesky_from_block(theta_hat, dim));
    }
    return make_frame(mu0 + theta_hat.head(dim),
                      detail::cholesky_from_block(theta_hat.tail(theta_hat.size() - dim),
                                                  dim));
}

/// theta_hat -> theta. The two describe the same ellipsoid; only the center's
/// representation differs.
inline Eigen::VectorXd to_theta( const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
                                 const Eigen::Ref<const Eigen::VectorXd>& mu0,
                                 MuMode mode )
{
    const int dim = static_cast<int>(mu0.size());
    detail::check_size("theta_hat", theta_hat.size(), theta_hat_size(dim, mode));

    Eigen::VectorXd theta(theta_size(dim));
    if ( mode == MuMode::Pinned )
    {
        theta.head(dim) = mu0;
        theta.tail(theta_hat.size()) = theta_hat;
    }
    else
    {
        theta.head(dim) = mu0 + theta_hat.head(dim);
        theta.tail(theta_hat.size() - dim) = theta_hat.tail(theta_hat.size() - dim);
    }
    return theta;
}

/// theta -> theta_hat, relative to mu0.
///
/// In Pinned mode the center block is dropped rather than checked against mu0:
/// mu is carried by mu0 there, and re-pinning at a DIFFERENT center is a
/// legitimate operation (that is what freeze_mu does).
inline Eigen::VectorXd to_theta_hat( const Eigen::Ref<const Eigen::VectorXd>& theta,
                                     const Eigen::Ref<const Eigen::VectorXd>& mu0,
                                     MuMode mode )
{
    const int dim = static_cast<int>(mu0.size());
    detail::check_size("theta", theta.size(), theta_size(dim));

    if ( mode == MuMode::Pinned )
    {
        return theta.tail(theta.size() - dim);
    }
    Eigen::VectorXd theta_hat(theta.size());
    theta_hat.head(dim) = theta.head(dim) - mu0;
    theta_hat.tail(theta.size() - dim) = theta.tail(theta.size() - dim);
    return theta_hat;
}

/// Pinned theta_hat -> fitted theta_hat about the SAME center: the fitted
/// encoding is literally the pinned one with a zero displacement prepended, so
/// this is what warm-starts a released-mu stage from a pinned stage's result.
inline Eigen::VectorXd release_mu(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat_pinned, int dim )
{
    detail::check_size("theta_hat", theta_hat_pinned.size(),
                       theta_hat_size(dim, MuMode::Pinned));
    Eigen::VectorXd released(theta_hat_size(dim, MuMode::Fitted));
    released.head(dim).setZero();
    released.tail(theta_hat_pinned.size()) = theta_hat_pinned;
    return released;
}

/// theta -> (pinned theta_hat, its reference center): re-pin at the center
/// theta already describes. Inverse of release_mu composed with to_theta.
inline std::pair<Eigen::VectorXd, Eigen::VectorXd> freeze_mu(
    const Eigen::Ref<const Eigen::VectorXd>& theta )
{
    const int dim = dim_from_theta_size(static_cast<int>(theta.size()));
    return {theta.tail(theta.size() - dim), theta.head(dim)};
}

// --------------------------------------------------------------------------
// Stage 1: the pullback geometry. Never sees a parameter vector.
// --------------------------------------------------------------------------

/// A tangent on (mu, L). Only L's lower triangle is ever populated by stage 2,
/// but the identities below hold for any dL.
struct FrameTangent
{
    Eigen::VectorXd dmu;  ///< (N,)
    Eigen::MatrixXd dL;   ///< (N, N)
};

/// A cotangent on (mu, L), one per point.
///
/// The L part is rank one per point (`-outer(z, u)`), but stage 2 reads it one
/// (i, j) slot at a time, so it is materialized as a (K, N*N) matrix with the
/// (i, j) component in column `i*N + j` -- contiguous per component, and free
/// to build at N <= 4.
struct PullbackCotangent
{
    Eigen::MatrixXd w_mu;  ///< (K, N)
    Eigen::MatrixXd w_L;   ///< (K, N*N)
    int dim = 0;

    /// The (K,) vector of d/dL(i,j) components.
    Eigen::MatrixXd::ConstColXpr component( int i, int j ) const
    {
        return w_L.col(static_cast<Eigen::Index>(i) * dim + j);
    }
};

/// u = L^{-1}(x - mu), for x of shape (K, N). Returns (K, N).
inline Eigen::MatrixXd pullback( const EllipsoidFrame& frame,
                                 const Eigen::Ref<const Eigen::MatrixXd>& x )
{
    detail::check_size("x's dimension", x.cols(), frame.dim());
    return (x.rowwise() - frame.mu.transpose()) * frame.L_inv.transpose();
}

/// Directional derivative of the pullback with respect to (mu, L), at fixed x.
///
///     du = -L^{-1} (dL u + dmu),
///
/// from d(L^{-1}) = -L^{-1} dL L^{-1} and dv = -dmu, with L^{-1} v substituted
/// back as u. It depends on x ONLY through u, so u is what it takes -- the
/// caller already has it, and stage 1's own output is the natural input to its
/// derivative.
inline Eigen::MatrixXd pullback_jvp( const EllipsoidFrame& frame,
                                     const FrameTangent& tangent,
                                     const Eigen::Ref<const Eigen::MatrixXd>& u )
{
    detail::check_size("u's dimension", u.cols(), frame.dim());
    const Eigen::MatrixXd rhs =
        (u * tangent.dL.transpose()).rowwise() + tangent.dmu.transpose();
    return -(rhs * frame.L_inv.transpose());
}

/// Reverse mode: for a cotangent w on u (shape (K, N), one per point), the
/// cotangents on (mu, L) such that for every (dmu, dL) and every point,
///
///     sum_i w_i du_i == sum_i w_mu_i dmu_i + sum_ij w_L_ij dL_ij.
///
/// z = L^{-T} w, w_mu = -z, w_L = -outer(z, u) -- the adjoint of the linear map
/// (dmu, dL) -> du above. That identity is the adjoint-consistency test.
inline PullbackCotangent pullback_vjp( const EllipsoidFrame& frame,
                                       const Eigen::Ref<const Eigen::MatrixXd>& u,
                                       const Eigen::Ref<const Eigen::MatrixXd>& w )
{
    const int dim = frame.dim();
    detail::check_size("u's dimension", u.cols(), dim);
    detail::check_size("w's dimension", w.cols(), dim);
    if ( w.rows() != u.rows() )
    {
        throw std::invalid_argument(
            "lgpsf::pullback_vjp: w must have one cotangent per point");
    }

    PullbackCotangent out;
    out.dim = dim;
    const Eigen::MatrixXd z = w * frame.L_inv;
    out.w_mu = -z;
    out.w_L.resize(u.rows(), static_cast<Eigen::Index>(dim) * dim);
    for ( int i = 0; i < dim; ++i )
    {
        for ( int j = 0; j < dim; ++j )
        {
            out.w_L.col(static_cast<Eigen::Index>(i) * dim + j) =
                -z.col(i).cwiseProduct(u.col(j));
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// Stage 2: theta_hat -> (mu, L). Never sees a point.
// --------------------------------------------------------------------------

/// Directional derivative of unpack_theta_hat.
///
/// d(L_ii)/d(log-diag_i) = L_ii by the chain rule through exp; the off-diagonal
/// entries and the center displacement are parameters directly, so their
/// tangents are the corresponding dtheta_hat entries. The center enters as a
/// translation, so this is the SAME map for both encodings -- which is why the
/// displacement encoding needed no new derivative math.
inline FrameTangent jvp_unpack_theta_hat(
    const EllipsoidFrame& frame,
    const Eigen::Ref<const Eigen::VectorXd>& dtheta_hat, MuMode mode )
{
    const int dim = frame.dim();
    detail::check_size("dtheta_hat", dtheta_hat.size(), theta_hat_size(dim, mode));

    FrameTangent tangent;
    int idx = 0;
    if ( mode == MuMode::Fitted )
    {
        tangent.dmu = dtheta_hat.head(dim);
        idx = dim;
    }
    else
    {
        tangent.dmu = Eigen::VectorXd::Zero(dim);
    }
    tangent.dL = Eigen::MatrixXd::Zero(dim, dim);
    for ( int i = 0; i < dim; ++i )
    {
        tangent.dL(i, i) = frame.L(i, i) * dtheta_hat(idx++);
    }
    for ( int i = 1; i < dim; ++i )
    {
        for ( int j = 0; j < i; ++j )
        {
            tangent.dL(i, j) = dtheta_hat(idx++);
        }
    }
    return tangent;
}

/// Reverse mode through stage 2: a per-point cotangent on (mu, L) becomes a
/// per-point covector on theta_hat, shape (K, P).
///
/// Deliberately per-point rather than summed, so the caller can sum it (for the
/// gradient of a scalar objective) or keep it as a batched Jacobian's columns.
/// In Pinned mode w_mu is simply dropped: mu is not a function of theta_hat
/// there, so it has no component to accumulate into.
inline Eigen::MatrixXd vjp_unpack_theta_hat( const EllipsoidFrame& frame,
                                             const PullbackCotangent& cotangent,
                                             MuMode mode )
{
    const int dim = frame.dim();
    if ( cotangent.dim != dim )
    {
        throw std::invalid_argument(
            "lgpsf::vjp_unpack_theta_hat: cotangent dimension does not match the frame");
    }

    Eigen::MatrixXd out(cotangent.w_mu.rows(), theta_hat_size(dim, mode));
    int idx = 0;
    if ( mode == MuMode::Fitted )
    {
        for ( int i = 0; i < dim; ++i )
        {
            out.col(idx++) = cotangent.w_mu.col(i);
        }
    }
    for ( int i = 0; i < dim; ++i )
    {
        out.col(idx++) = cotangent.component(i, i) * frame.L(i, i);
    }
    for ( int i = 1; i < dim; ++i )
    {
        for ( int j = 0; j < i; ++j )
        {
            out.col(idx++) = cotangent.component(i, j);
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// Composition: T(theta_hat, x) and its parameter derivatives.
// --------------------------------------------------------------------------

/// u = T(theta_hat, x), for x of shape (K, N). Returns (K, N) -- feed it
/// straight into LGBasisAt.
inline Eigen::MatrixXd eval_T( const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
                               const Eigen::Ref<const Eigen::MatrixXd>& x,
                               const Eigen::Ref<const Eigen::VectorXd>& mu0,
                               MuMode mode )
{
    return pullback(unpack_theta_hat(theta_hat, mu0, mode), x);
}

/// Directional derivative of T with respect to theta_hat, at fixed x: (K, N).
inline Eigen::MatrixXd jvp_T( const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
                              const Eigen::Ref<const Eigen::VectorXd>& dtheta_hat,
                              const Eigen::Ref<const Eigen::MatrixXd>& x,
                              const Eigen::Ref<const Eigen::VectorXd>& mu0,
                              MuMode mode )
{
    const EllipsoidFrame frame = unpack_theta_hat(theta_hat, mu0, mode);
    return pullback_jvp(frame, jvp_unpack_theta_hat(frame, dtheta_hat, mode),
                        pullback(frame, x));
}

/// Reverse mode: <w, dT/dtheta_hat> per point, for a cotangent w of shape
/// (K, N). Returns (K, P).
inline Eigen::MatrixXd vjp_T( const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
                              const Eigen::Ref<const Eigen::MatrixXd>& x,
                              const Eigen::Ref<const Eigen::MatrixXd>& w,
                              const Eigen::Ref<const Eigen::VectorXd>& mu0,
                              MuMode mode )
{
    const EllipsoidFrame frame = unpack_theta_hat(theta_hat, mu0, mode);
    return vjp_unpack_theta_hat(frame, pullback_vjp(frame, pullback(frame, x), w),
                                mode);
}

/// dT/dtheta_hat_q at every point, by P forward sweeps: P matrices of (K, N),
/// entry q being exactly jvp_T against the q-th unit direction.
///
/// Indexed by PARAMETER, not by output component, because that is how every
/// consumer slices it (one Jacobian column at a time) and because it makes the
/// identity `jacobian_tensor_forward(...)[q] == jvp_T(..., unit_q, ...)`
/// literal.
inline std::vector<Eigen::MatrixXd> jacobian_tensor_forward(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mode )
{
    const EllipsoidFrame frame = unpack_theta_hat(theta_hat, mu0, mode);
    const Eigen::MatrixXd u = pullback(frame, x);
    const int n_params = theta_hat_size(frame.dim(), mode);

    std::vector<Eigen::MatrixXd> columns;
    columns.reserve(static_cast<std::size_t>(n_params));
    Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_params);
    for ( int q = 0; q < n_params; ++q )
    {
        direction(q) = 1.0;
        columns.push_back(
            pullback_jvp(frame, jvp_unpack_theta_hat(frame, direction, mode), u));
        direction(q) = 0.0;
    }
    return columns;
}

/// The same tensor by N reverse sweeps instead of P forward ones. Exists so the
/// two modes can be cross-checked against each other; forward is the one the
/// feature layer uses.
inline std::vector<Eigen::MatrixXd> jacobian_tensor_reverse(
    const Eigen::Ref<const Eigen::VectorXd>& theta_hat,
    const Eigen::Ref<const Eigen::MatrixXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& mu0, MuMode mode )
{
    const EllipsoidFrame frame = unpack_theta_hat(theta_hat, mu0, mode);
    const Eigen::MatrixXd u = pullback(frame, x);
    const int dim = frame.dim();
    const int n_params = theta_hat_size(dim, mode);

    std::vector<Eigen::MatrixXd> columns(
        static_cast<std::size_t>(n_params), Eigen::MatrixXd(x.rows(), dim));
    Eigen::MatrixXd cotangent = Eigen::MatrixXd::Zero(x.rows(), dim);
    for ( int i = 0; i < dim; ++i )
    {
        cotangent.col(i).setOnes();
        const Eigen::MatrixXd rows = vjp_unpack_theta_hat(
            frame, pullback_vjp(frame, u, cotangent), mode);
        for ( int q = 0; q < n_params; ++q )
        {
            columns[static_cast<std::size_t>(q)].col(i) = rows.col(q);
        }
        cotangent.col(i).setZero();
    }
    return columns;
}

} // end namespace lgpsf
