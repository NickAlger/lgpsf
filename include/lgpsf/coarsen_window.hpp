#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Graded quadrature coarsening of a fit window: a per-row work bound
/// that needs no estimate of the kernel's width.
///
/// A row is fitted on its window, and the fit's cost is linear in the window's
/// point count -- so a row whose a-priori ellipsoid is far too wide can cost a
/// hundred times the mean. The fit never regresses pointwise, though. Its
/// design matrix is a mass-weighted quadrature of the model's action on the
/// probes (probe_fit.hpp), `sum_j m_j z_jl phi_i(x_j)`, so "subsample the
/// window" is really "choose a coarser quadrature", and this header chooses
/// one that is adequate for EVERY kernel width at once:
///
///   - Cells graded by distance from the row's centre, size at most `eps`
///     times that distance, floored at single points. For a Gaussian of width
///     sigma the midpoint error of a cell of size h at distance r is about
///     `(h r / sigma^2) exp(-r^2 / 2 sigma^2)`, at most `0.74 h / r` over all
///     sigma; so `h <= eps r` bounds it by eps for every width simultaneously,
///     and the centroid rule makes it second order.
///   - Each cell becomes its mass-weighted centroid, its summed mass and the
///     mass-weighted mean of every probe field. The probe sum inside a cell,
///     `sum_j m_j z_jl`, is kept EXACTLY; only the basis function's variation
///     over the cell is approximated.
///   - Protected positions (the spike, the support of any extra column) and
///     the point farthest from the centre stay singleton cells, copied bit for
///     bit. The spike's design column is independent of the masses, so a
///     singleton reproduces it exactly and keeps the meaning of `s`; and the
///     farthest point keeps `window_radius` -- the admissibility bound and the
///     ladder's top rung -- exact, where a centroid would silently tighten it.
///
/// The cells are built in the coordinates of the WINDOW ellipsoid,
/// `u = L_w^{-1} (x - mu_w)`: the object that already encodes how much of the
/// prior's shape the library trusts. A ball window makes the grading
/// Euclidean; an anisotropic one grades in its own Mahalanobis metric for
/// free. The fit runs on the cells; the deployed operator, whose support is
/// the full window, is unchanged (operator_fit.hpp).
///
/// Point batches are (K, N) coordinate-major.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "lgpsf/ellipsoid_transform.hpp"

namespace lgpsf {

/// A coarsened window: one quadrature point per cell, row-aligned like the
/// window it came from, plus the provenance needed to remap positions.
struct CoarseWindow
{
    Eigen::MatrixXd  x;                ///< (Kc, N) mass-weighted cell centroids, PHYSICAL coordinates
    Eigen::VectorXd  m2;               ///< (Kc,) summed cell masses
    Eigen::MatrixXd  z;                ///< (Kc, k) mass-weighted mean probe fields
    std::vector<int> protected_cells;  ///< coarse position of each protected input position, in input order
    std::vector<int> cell_of;          ///< (K,) the cell each input position joined
};

namespace detail {

/// One node of the 2^N-tree: an axis-aligned box in whitened coordinates and
/// the ascending window positions of the points it holds.
struct CoarsenNode
{
    Eigen::VectorXd  lo;       ///< (N,) lower corner
    Eigen::VectorXd  hi;       ///< (N,) upper corner
    std::vector<int> members;  ///< ascending
    int              depth = 0;
};

/// Euclidean distance from the origin to the nearest point of the box, zero
/// when the box contains the origin.
inline double box_distance_to_origin( const Eigen::VectorXd& lo,
                                      const Eigen::VectorXd& hi )
{
    double sum = 0.0;
    for ( Eigen::Index a = 0; a < lo.size(); ++a )
    {
        double gap = 0.0;
        if ( lo(a) > 0.0 )
        {
            gap = lo(a);
        }
        else if ( hi(a) < 0.0 )
        {
            gap = -hi(a);
        }
        sum += gap * gap;
    }
    return std::sqrt(sum);
}

} // end namespace detail

/// Coarsen a window into graded quadrature cells.
///
/// **Algorithm**, in the whitened coordinates `u_j = pullback(window_frame,
/// x_j)`. A 2^N-tree on the axis-aligned bounding box of `{u_j}` (a quadtree
/// in 2D, an octree in 3D), splitting every axis at its midpoint; a point
/// exactly on a split plane goes to the upper child, empty children are
/// dropped, and child boxes are the sub-boxes, not re-tightened. A node is
/// split while it holds more than one point and either contains a protected
/// position or has diagonal `> eps * d`, where `d` is the whitened distance
/// from the origin to the box (zero if the box contains the origin). Every
/// leaf is a cell: mass `m_C = sum m_j`, centroid `x_C = sum m_j x_j / m_C`
/// in PHYSICAL coordinates and probe mean `z_C = sum m_j z_j / m_C`, each
/// accumulated in ascending position. Cells are emitted sorted by the smallest
/// position they contain; a singleton cell copies its rows bit for bit.
///
/// The protected set is `protected_positions` plus the point farthest from
/// @p centre in physical distance (ties to the smallest position). Note the
/// two centres: the grading measures distance from the frame's `mu`, the
/// farthest-point rule from @p centre. They normally coincide.
///
/// **Pure**: no threads, no randomness, no dependence on allocation order or
/// on anything outside its arguments. The same inputs give bit-identical
/// outputs, which is what keeps a distributed fit rank-independent -- the
/// window array is already canonical there, and this is a pure function of it.
///
/// **Cost** O(K log K) expected. The cell count is about `3 pi / eps^2` per
/// dyadic annulus in 2D, so `Kc ~ (3 pi / eps^2) log2(R / h)` for a window of
/// radius R on a mesh of spacing h -- logarithmic in the window's size, which
/// is what makes the fit's cost scale under refinement; in 3D about
/// `4 pi / eps^3` per shell. (The tests measure `Kc / law` between 0.7 and
/// 1.1 on a 10^4-point grid: grading on the box diagonal costs a constant,
/// and the singleton core, which the law counts as full annuli, gives one
/// back.) Near the centre the cells are singletons -- a box containing the
/// origin has `d = 0` and splits down to single points -- so `local_spacing`
/// is preserved.
///
/// **Resolution.** At distance r the cell is `eps * r`, and a mode of radial
/// degree p and angular order ell has its finest structure at scale
/// `sigma / (p + ell)` near `r ~ sigma`, so the uniform condition is
/// `eps * (p + ell)_max <~ 0.5`: at the ladder's top level of 5, `eps <= 0.1`.
/// A coarser quadrature aliases the high modes into noise, which the ladder's
/// cross-validation treats as the conservative failure (it stops lower).
///
/// **Caveats.**
///   - `window_shape` / `window_shape_ladder` on the coarse arrays lose the
///     within-cell second moments (parallel-axis theorem) and read smaller;
///     compute them on the full window.
///   - The cells are axis-aligned boxes in the frame's Cholesky coordinates,
///     which is what `make_frame` takes: parallelograms in physical space when
///     `L` has off-diagonal entries. The grading -- which points are close --
///     is the Mahalanobis metric of `L L^T` regardless.
///   - Points coincident in whitened coordinates cannot be separated, so a
///     protected point coincident with another shares its cell: a split is
///     abandoned once the box's diagonal is below 1e-12 of the root's or the
///     node is 60 levels deep. `protected_cells` then names a multi-point
///     cell, and the coarse set may hold fewer points than the caller expects.
///   - `eps == 0` and `K == 1` are the identity: the arrays are exact copies,
///     `cell_of` is 0..K-1 and `protected_cells == protected_positions`.
///
/// @param x_window            (K, N) window points, physical coordinates.
/// @param m2_window           (K,) their masses, finite and positive.
/// @param z                   (K, k) probe fields on the window.
/// @param protected_positions Window positions kept as singleton cells,
///                            distinct and in range.
/// @param centre              (N,) the row's centre, for the farthest-point
///                            rule.
/// @param window_frame        The window ellipsoid; the tree lives in its
///                            whitened coordinates.
/// @param eps                 Grading ratio, finite and >= 0.
/// @return                    The coarse window. `protected_cells[i]` is the
///                            cell of `protected_positions[i]`.
/// @throws std::invalid_argument on an empty window, a shape mismatch, a
///         non-finite or non-positive mass, a non-finite coordinate, a
///         non-finite or negative eps, or an out-of-range or repeated
///         protected position.
inline CoarseWindow coarsen_window(
    const Eigen::Ref<const Eigen::MatrixXd>& x_window,
    const Eigen::Ref<const Eigen::VectorXd>& m2_window,
    const Eigen::Ref<const Eigen::MatrixXd>& z,
    const std::vector<int>& protected_positions,
    const Eigen::Ref<const Eigen::VectorXd>& centre,
    const EllipsoidFrame& window_frame,
    double eps )
{
    const Eigen::Index count = x_window.rows();
    const Eigen::Index dim = centre.size();

    // ---- eager validation ------------------------------------------------
    if ( count == 0 )
    {
        throw std::invalid_argument("lgpsf::coarsen_window: the window is empty");
    }
    if ( dim < 1 || dim > 30 )
    {
        throw std::invalid_argument(
            "lgpsf::coarsen_window: the tree has 2^N children per node, so N must "
            "be between 1 and 30; centre has " + std::to_string(dim) + " entries");
    }
    if ( x_window.cols() != dim )
    {
        throw std::invalid_argument(
            "lgpsf::coarsen_window: x_window has " + std::to_string(x_window.cols())
            + " coordinate columns but centre has " + std::to_string(dim) + " entries");
    }
    if ( window_frame.dim() != dim )
    {
        throw std::invalid_argument(
            "lgpsf::coarsen_window: window_frame is " + std::to_string(window_frame.dim())
            + "-dimensional but centre has " + std::to_string(dim) + " entries");
    }
    if ( m2_window.size() != count )
    {
        throw std::invalid_argument(
            "lgpsf::coarsen_window: x_window has " + std::to_string(count)
            + " points but m2_window has " + std::to_string(m2_window.size()) + " entries");
    }
    if ( z.rows() != count )
    {
        throw std::invalid_argument(
            "lgpsf::coarsen_window: x_window has " + std::to_string(count)
            + " points but z has " + std::to_string(z.rows()) + " rows");
    }
    for ( Eigen::Index j = 0; j < count; ++j )
    {
        if ( !(m2_window(j) > 0.0) || !std::isfinite(m2_window(j)) )
        {
            throw std::invalid_argument(
                "lgpsf::coarsen_window: masses must be finite and positive; entry "
                + std::to_string(j) + " is " + std::to_string(m2_window(j)));
        }
    }
    if ( !x_window.allFinite() || !centre.allFinite() )
    {
        throw std::invalid_argument(
            "lgpsf::coarsen_window: x_window and centre must be finite");
    }
    if ( !(eps >= 0.0) || !std::isfinite(eps) )
    {
        throw std::invalid_argument(
            "lgpsf::coarsen_window: eps must be finite and >= 0, got "
            + std::to_string(eps));
    }
    for ( int p : protected_positions )
    {
        if ( p < 0 || p >= count )
        {
            throw std::invalid_argument(
                "lgpsf::coarsen_window: protected position " + std::to_string(p)
                + " is out of range for a window of " + std::to_string(count) + " points");
        }
    }
    {
        std::vector<int> sorted = protected_positions;
        std::sort(sorted.begin(), sorted.end());
        const auto duplicate = std::adjacent_find(sorted.begin(), sorted.end());
        if ( duplicate != sorted.end() )
        {
            throw std::invalid_argument(
                "lgpsf::coarsen_window: protected position " + std::to_string(*duplicate)
                + " is repeated");
        }
    }

    // ---- the identity cases ----------------------------------------------
    if ( count == 1 || eps == 0.0 )
    {
        CoarseWindow out;
        out.x = x_window;
        out.m2 = m2_window;
        out.z = z;
        out.protected_cells = protected_positions;
        out.cell_of.resize(static_cast<std::size_t>(count));
        std::iota(out.cell_of.begin(), out.cell_of.end(), 0);
        return out;
    }

    // ---- the protected set -----------------------------------------------
    std::vector<bool> is_protected(static_cast<std::size_t>(count), false);
    for ( int p : protected_positions )
    {
        is_protected[static_cast<std::size_t>(p)] = true;
    }
    {
        // The same expression window_radius() evaluates, so the singleton
        // this keeps is exactly the point that sets the radius.
        const Eigen::VectorXd distance =
            (x_window.rowwise() - centre.transpose()).rowwise().norm();
        Eigen::Index farthest = 0;
        for ( Eigen::Index j = 1; j < count; ++j )
        {
            if ( distance(j) > distance(farthest) )
            {
                farthest = j;
            }
        }
        is_protected[static_cast<std::size_t>(farthest)] = true;
    }

    // ---- the tree ----------------------------------------------------------
    const Eigen::MatrixXd u = pullback(window_frame, x_window);
    const int num_children = 1 << dim;

    detail::CoarsenNode root;
    root.lo = u.colwise().minCoeff().transpose();
    root.hi = u.colwise().maxCoeff().transpose();
    root.members.resize(static_cast<std::size_t>(count));
    std::iota(root.members.begin(), root.members.end(), 0);
    const double root_diagonal = (root.hi - root.lo).norm();

    std::vector<detail::CoarsenNode> stack;
    stack.push_back(std::move(root));
    std::vector<std::vector<int>> leaves;
    std::vector<std::vector<int>> child_members(static_cast<std::size_t>(num_children));
    Eigen::VectorXd mid(dim);

    while ( !stack.empty() )
    {
        detail::CoarsenNode node = std::move(stack.back());
        stack.pop_back();

        if ( node.members.size() <= 1 )
        {
            leaves.push_back(std::move(node.members));
            continue;
        }
        const double diagonal = (node.hi - node.lo).norm();
        // Coincident points: the box can shrink forever without separating
        // them. Give up at round-off scale (or depth), and accept the cell.
        if ( !(diagonal > 1e-12 * root_diagonal) || node.depth > 60 )
        {
            leaves.push_back(std::move(node.members));
            continue;
        }
        bool has_protected = false;
        for ( int j : node.members )
        {
            if ( is_protected[static_cast<std::size_t>(j)] )
            {
                has_protected = true;
                break;
            }
        }
        if ( !has_protected )
        {
            const double d = detail::box_distance_to_origin(node.lo, node.hi);
            if ( diagonal <= eps * d )
            {
                leaves.push_back(std::move(node.members));
                continue;
            }
        }

        // Split: midpoint on every axis, the upper child owning the plane.
        mid = 0.5 * (node.lo + node.hi);
        for ( int j : node.members )
        {
            int bits = 0;
            for ( Eigen::Index a = 0; a < dim; ++a )
            {
                if ( u(j, a) >= mid(a) )
                {
                    bits |= 1 << a;
                }
            }
            child_members[static_cast<std::size_t>(bits)].push_back(j);
        }
        for ( int c = 0; c < num_children; ++c )
        {
            std::vector<int>& members = child_members[static_cast<std::size_t>(c)];
            if ( members.empty() )
            {
                continue;
            }
            detail::CoarsenNode child;
            child.lo.resize(dim);
            child.hi.resize(dim);
            for ( Eigen::Index a = 0; a < dim; ++a )
            {
                const bool upper = ( c >> a ) & 1;
                child.lo(a) = upper ? mid(a) : node.lo(a);
                child.hi(a) = upper ? node.hi(a) : mid(a);
            }
            child.members = std::move(members);
            members = std::vector<int>();
            child.depth = node.depth + 1;
            stack.push_back(std::move(child));
        }
    }

    // ---- aggregation, in canonical order -----------------------------------
    // Leaves partition the positions, so their smallest members are distinct
    // and this order is total: the output does not depend on traversal order.
    std::vector<std::size_t> order(leaves.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&leaves]( std::size_t a, std::size_t b ) {
        return leaves[a].front() < leaves[b].front();
    });

    const Eigen::Index num_cells = static_cast<Eigen::Index>(leaves.size());
    const Eigen::Index num_probes = z.cols();
    CoarseWindow out;
    out.x.resize(num_cells, dim);
    out.m2.resize(num_cells);
    out.z.resize(num_cells, num_probes);
    out.cell_of.assign(static_cast<std::size_t>(count), -1);

    Eigen::RowVectorXd x_sum(dim);
    Eigen::RowVectorXd z_sum(num_probes);
    for ( Eigen::Index c = 0; c < num_cells; ++c )
    {
        const std::vector<int>& members = leaves[order[static_cast<std::size_t>(c)]];
        if ( members.size() == 1 )
        {
            const int j = members.front();
            out.x.row(c) = x_window.row(j);
            out.m2(c) = m2_window(j);
            out.z.row(c) = z.row(j);
        }
        else
        {
            double mass = 0.0;
            x_sum.setZero();
            z_sum.setZero();
            for ( int j : members )
            {
                const double m = m2_window(j);
                mass += m;
                x_sum += m * x_window.row(j);
                z_sum += m * z.row(j);
            }
            out.x.row(c) = x_sum / mass;
            out.m2(c) = mass;
            out.z.row(c) = z_sum / mass;
        }
        for ( int j : members )
        {
            out.cell_of[static_cast<std::size_t>(j)] = static_cast<int>(c);
        }
    }

    out.protected_cells.reserve(protected_positions.size());
    for ( int p : protected_positions )
    {
        out.protected_cells.push_back(out.cell_of[static_cast<std::size_t>(p)]);
    }
    return out;
}

} // end namespace lgpsf
