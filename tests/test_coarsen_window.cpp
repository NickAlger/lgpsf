// SPDX-License-Identifier: MIT
//
// Checks on the graded window coarsening: the invariants the fit relies on
// (conservation, protected singletons, the grading rule, purity, a valid
// partition), the identity cases, and the cell-count law, on structured 2D
// and 3D windows with uneven masses.
//
// All self-contained: nothing here is compared against a stored reference, so
// the suite cannot drift out of step with the code it tests.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/coarsen_window.hpp"
#include "test_helpers.hpp"

using lgpsf::CoarseWindow;
using lgpsf::EllipsoidFrame;
using lgpsf::coarsen_window;
using lgpsf::make_frame;
using lgpsf::pullback;

namespace {

struct Window
{
    Eigen::MatrixXd x;
    Eigen::VectorXd m2;
    Eigen::MatrixXd z;
    double spacing = 0.0;
};

/// A tensor grid on [-1, 1]^dim with `per_side` points per axis, each
/// coordinate jittered by up to `jitter` of the spacing so nothing sits on a
/// split plane; masses are the cell volume times an uneven factor in
/// [0.5, 1.5) so the mass weighting is exercised; probe fields are
/// standard normal. Coordinate 0 varies fastest.
Window grid_window( int per_side, int dim, int num_probes, unsigned seed,
                    double jitter )
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double h = 2.0 / (per_side - 1);
    Eigen::Index count = 1;
    for ( int d = 0; d < dim; ++d )
    {
        count *= per_side;
    }

    Window w;
    w.spacing = h;
    w.x.resize(count, dim);
    w.m2.resize(count);
    for ( Eigen::Index flat = 0; flat < count; ++flat )
    {
        Eigen::Index rest = flat;
        double volume = 1.0;
        for ( int d = 0; d < dim; ++d )
        {
            const Eigen::Index idx = rest % per_side;
            rest /= per_side;
            w.x(flat, d) = -1.0 + h * static_cast<double>(idx)
                           + jitter * h * (2.0 * unit(gen) - 1.0);
            volume *= h;
        }
        w.m2(flat) = volume * (0.5 + unit(gen));
    }
    w.z = test_helpers::randn_points(static_cast<int>(count), num_probes, gen);
    return w;
}

/// Points on a jittered grid inside the unit ball of the frame's whitened
/// coordinates, mapped to physical space through the frame: the shape a
/// window has when it IS the frame's ellipsoid, as a fit window is.
Window ellipsoid_window( const EllipsoidFrame& frame, int per_side, int num_probes,
                         unsigned seed )
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const int dim = frame.dim();
    const double h = 2.0 / (per_side - 1);
    Eigen::Index total = 1;
    for ( int d = 0; d < dim; ++d )
    {
        total *= per_side;
    }

    std::vector<Eigen::VectorXd> kept;
    for ( Eigen::Index flat = 0; flat < total; ++flat )
    {
        Eigen::Index rest = flat;
        Eigen::VectorXd u(dim);
        for ( int d = 0; d < dim; ++d )
        {
            const Eigen::Index idx = rest % per_side;
            rest /= per_side;
            u(d) = -1.0 + h * static_cast<double>(idx) + 0.3 * h * (2.0 * unit(gen) - 1.0);
        }
        if ( u.norm() <= 1.0 )
        {
            kept.push_back(std::move(u));
        }
    }

    Window w;
    w.spacing = h;
    const Eigen::Index count = static_cast<Eigen::Index>(kept.size());
    w.x.resize(count, dim);
    w.m2.resize(count);
    for ( Eigen::Index i = 0; i < count; ++i )
    {
        w.x.row(i) = (frame.mu + frame.L * kept[static_cast<std::size_t>(i)]).transpose();
        w.m2(i) = std::pow(h, dim) * (0.5 + unit(gen));
    }
    w.z = test_helpers::randn_points(static_cast<int>(count), num_probes, gen);
    return w;
}

EllipsoidFrame ball_frame( int dim, double radius )
{
    return make_frame(Eigen::VectorXd::Zero(dim),
                      radius * Eigen::MatrixXd::Identity(dim, dim));
}

/// A 2D frame with off-diagonal structure: a sheared, anisotropic ellipsoid.
EllipsoidFrame sheared_frame_2d()
{
    Eigen::MatrixXd L(2, 2);
    L << 2.0, 0.0,
         0.5, 0.3;
    return make_frame(Eigen::VectorXd::Zero(2), L);
}

/// True iff `call` throws std::invalid_argument with this header's prefix.
template <class F>
bool rejected( F&& call )
{
    try
    {
        call();
    }
    catch ( const std::invalid_argument& error )
    {
        return std::string(error.what()).rfind("lgpsf::coarsen_window:", 0) == 0;
    }
    return false;
}

template <class A, class B>
bool same_bits( const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b )
{
    return a.rows() == b.rows() && a.cols() == b.cols()
           && (a.array() == b.array()).all();
}

/// The members of every cell, in ascending position.
std::vector<std::vector<int>> members_of( const CoarseWindow& cw )
{
    std::vector<std::vector<int>> cells(static_cast<std::size_t>(cw.x.rows()));
    for ( std::size_t j = 0; j < cw.cell_of.size(); ++j )
    {
        cells[static_cast<std::size_t>(cw.cell_of[j])].push_back(static_cast<int>(j));
    }
    return cells;
}

/// The same rule the header uses to pick the farthest point: physical
/// distance to the centre, ties to the smallest position.
int farthest_position( const Eigen::MatrixXd& x, const Eigen::VectorXd& centre )
{
    const Eigen::VectorXd distance = (x.rowwise() - centre.transpose()).rowwise().norm();
    int best = 0;
    for ( Eigen::Index j = 1; j < distance.size(); ++j )
    {
        if ( distance(j) > distance(best) )
        {
            best = static_cast<int>(j);
        }
    }
    return best;
}

/// Bounding-box diagonal and distance from the origin of a set of whitened
/// points -- the grading quantities, recomputed from the members alone.
std::pair<double, double> grading_of( const Eigen::MatrixXd& u,
                                      const std::vector<int>& members )
{
    const Eigen::Index dim = u.cols();
    Eigen::VectorXd lo = Eigen::VectorXd::Constant(dim, std::numeric_limits<double>::infinity());
    Eigen::VectorXd hi = -lo;
    for ( int j : members )
    {
        lo = lo.cwiseMin(u.row(j).transpose());
        hi = hi.cwiseMax(u.row(j).transpose());
    }
    double gap2 = 0.0;
    for ( Eigen::Index a = 0; a < dim; ++a )
    {
        const double gap = ( lo(a) > 0.0 ) ? lo(a) : ( hi(a) < 0.0 ) ? -hi(a) : 0.0;
        gap2 += gap * gap;
    }
    return {(hi - lo).norm(), std::sqrt(gap2)};
}

void check_partition( const CoarseWindow& cw, Eigen::Index count )
{
    REQUIRE(cw.cell_of.size() == static_cast<std::size_t>(count));
    REQUIRE(cw.m2.size() == cw.x.rows());
    REQUIRE(cw.z.rows() == cw.x.rows());
    const Eigen::Index num_cells = cw.x.rows();
    for ( int c : cw.cell_of )
    {
        CHECK(c >= 0);
        CHECK(c < num_cells);
    }
    const std::vector<std::vector<int>> cells = members_of(cw);
    int previous_first = -1;
    std::size_t total = 0;
    for ( const std::vector<int>& members : cells )
    {
        REQUIRE_FALSE(members.empty());                  // every cell has a point
        CHECK(std::is_sorted(members.begin(), members.end()));
        CHECK(members.front() > previous_first);         // sorted by smallest member
        previous_first = members.front();
        total += members.size();
    }
    CHECK(total == static_cast<std::size_t>(count));     // every point exactly once
}

void check_conservation( const Window& w, const CoarseWindow& cw )
{
    const double total = w.m2.sum();
    CHECK(std::abs(cw.m2.sum() - total) <= 1e-12 * total);

    const Eigen::RowVectorXd fine = w.m2.transpose() * w.z;
    const Eigen::RowVectorXd coarse = cw.m2.transpose() * cw.z;
    const Eigen::RowVectorXd scale = w.m2.transpose() * w.z.cwiseAbs();
    for ( Eigen::Index l = 0; l < fine.size(); ++l )
    {
        CHECK(std::abs(coarse(l) - fine(l)) <= 1e-12 * scale(l));
    }

    // The first moment is conserved too: sum m_C x_C == sum m_j x_j.
    const Eigen::RowVectorXd fine_x = w.m2.transpose() * w.x;
    const Eigen::RowVectorXd coarse_x = cw.m2.transpose() * cw.x;
    const Eigen::RowVectorXd scale_x = w.m2.transpose() * w.x.cwiseAbs();
    for ( Eigen::Index a = 0; a < fine_x.size(); ++a )
    {
        CHECK(std::abs(coarse_x(a) - fine_x(a)) <= 1e-12 * scale_x(a));
    }

    // Each centroid lies inside its members' physical bounding box.
    const std::vector<std::vector<int>> cells = members_of(cw);
    for ( std::size_t c = 0; c < cells.size(); ++c )
    {
        for ( Eigen::Index a = 0; a < w.x.cols(); ++a )
        {
            double lo = std::numeric_limits<double>::infinity();
            double hi = -lo;
            for ( int j : cells[c] )
            {
                lo = std::min(lo, w.x(j, a));
                hi = std::max(hi, w.x(j, a));
            }
            const double slack = 1e-12 * (1.0 + std::max(std::abs(lo), std::abs(hi)));
            CHECK(cw.x(static_cast<Eigen::Index>(c), a) >= lo - slack);
            CHECK(cw.x(static_cast<Eigen::Index>(c), a) <= hi + slack);
        }
    }
}

/// Every multi-point cell obeys `diag <= eps * d` on its members' own box --
/// at least as tight as the node box the rule was applied to.
void check_grading( const Window& w, const CoarseWindow& cw,
                    const EllipsoidFrame& frame, double eps )
{
    const Eigen::MatrixXd u = pullback(frame, w.x);
    int multi = 0;
    for ( const std::vector<int>& members : members_of(cw) )
    {
        if ( members.size() < 2 )
        {
            continue;
        }
        ++multi;
        const auto [diag, d] = grading_of(u, members);
        CHECK(diag <= eps * d * (1.0 + 1e-12));
    }
    CHECK(multi > 0);  // the check has teeth only if something was merged
}

/// `position` is alone in its cell and its rows are copied bit for bit.
void check_singleton( const Window& w, const CoarseWindow& cw, int position )
{
    const std::vector<std::vector<int>> cells = members_of(cw);
    const int c = cw.cell_of[static_cast<std::size_t>(position)];
    REQUIRE(cells[static_cast<std::size_t>(c)].size() == 1u);
    CHECK(same_bits(cw.x.row(c), w.x.row(position)));
    CHECK(cw.m2(c) == w.m2(position));
    CHECK(same_bits(cw.z.row(c), w.z.row(position)));
}

bool same_result( const CoarseWindow& a, const CoarseWindow& b )
{
    return same_bits(a.x, b.x) && same_bits(a.m2, b.m2) && same_bits(a.z, b.z)
           && a.protected_cells == b.protected_cells && a.cell_of == b.cell_of;
}

} // end anonymous namespace

TEST_CASE("coarsen_window rejects malformed inputs eagerly, by name")
{
    const Window w = grid_window(5, 2, 3, 1, 0.3);
    const Eigen::Index count = w.x.rows();
    const EllipsoidFrame frame = ball_frame(2, 1.0);
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    const std::vector<int> protected_positions = {3};
    const double eps = 0.2;

    CHECK_NOTHROW(coarsen_window(w.x, w.m2, w.z, protected_positions, centre, frame, eps));

    // an empty window
    CHECK(rejected([&] {
        coarsen_window(Eigen::MatrixXd(0, 2), Eigen::VectorXd(0), Eigen::MatrixXd(0, 3),
                       {}, centre, frame, eps);
    }));
    // shape mismatches
    CHECK(rejected([&] {
        coarsen_window(w.x, w.m2.head(count - 1), w.z, protected_positions, centre, frame, eps);
    }));
    CHECK(rejected([&] {
        coarsen_window(w.x, w.m2, w.z.topRows(count - 1), protected_positions, centre, frame, eps);
    }));
    CHECK(rejected([&] {
        coarsen_window(w.x, w.m2, w.z, protected_positions, Eigen::VectorXd::Zero(3), frame, eps);
    }));
    CHECK(rejected([&] {
        coarsen_window(w.x, w.m2, w.z, protected_positions, centre, ball_frame(3, 1.0), eps);
    }));
    // masses: zero, negative, NaN
    for ( double bad : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()} )
    {
        Eigen::VectorXd m2 = w.m2;
        m2(7) = bad;
        CHECK(rejected([&] {
            coarsen_window(w.x, m2, w.z, protected_positions, centre, frame, eps);
        }));
    }
    // eps: negative, NaN, infinite
    for ( double bad : {-0.1, std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity()} )
    {
        CHECK(rejected([&] {
            coarsen_window(w.x, w.m2, w.z, protected_positions, centre, frame, bad);
        }));
    }
    // protected positions: out of range either side, repeated
    CHECK(rejected([&] {
        coarsen_window(w.x, w.m2, w.z, {-1}, centre, frame, eps);
    }));
    CHECK(rejected([&] {
        coarsen_window(w.x, w.m2, w.z, {static_cast<int>(count)}, centre, frame, eps);
    }));
    CHECK(rejected([&] {
        coarsen_window(w.x, w.m2, w.z, {3, 5, 3}, centre, frame, eps);
    }));
    // a non-finite coordinate would silently corrupt the tree
    {
        Eigen::MatrixXd x = w.x;
        x(4, 1) = std::numeric_limits<double>::quiet_NaN();
        CHECK(rejected([&] {
            coarsen_window(x, w.m2, w.z, protected_positions, centre, frame, eps);
        }));
    }
}

TEST_CASE("eps == 0 and a single point are the identity, bit for bit")
{
    const Window w = grid_window(9, 2, 3, 2, 0.3);
    const Eigen::Index count = w.x.rows();
    const EllipsoidFrame frame = sheared_frame_2d();
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    const std::vector<int> protected_positions = {40, 3, 77};

    const CoarseWindow cw =
        coarsen_window(w.x, w.m2, w.z, protected_positions, centre, frame, 0.0);
    CHECK(same_bits(cw.x, w.x));
    CHECK(same_bits(cw.m2, w.m2));
    CHECK(same_bits(cw.z, w.z));
    CHECK(cw.protected_cells == protected_positions);
    std::vector<int> iota(static_cast<std::size_t>(count));
    std::iota(iota.begin(), iota.end(), 0);
    CHECK(cw.cell_of == iota);

    // One point: nothing to coarsen, whatever eps says.
    const CoarseWindow one =
        coarsen_window(w.x.topRows(1), w.m2.head(1), w.z.topRows(1), {0}, centre, frame, 0.5);
    CHECK(same_bits(one.x, w.x.topRows(1)));
    CHECK(one.m2(0) == w.m2(0));
    CHECK(same_bits(one.z, w.z.topRows(1)));
    CHECK(one.protected_cells == std::vector<int>{0});
    CHECK(one.cell_of == std::vector<int>{0});
}

TEST_CASE("coarsening conserves mass and the probe sums, and centroids stay in their cells")
{
    const Window w = grid_window(41, 2, 4, 3, 0.3);
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    const int centre_position = 20 + 41 * 20;  // the grid point nearest the origin

    for ( const EllipsoidFrame& frame : {ball_frame(2, 1.0), sheared_frame_2d()} )
    {
        for ( double eps : {0.1, 0.25} )
        {
            const CoarseWindow cw =
                coarsen_window(w.x, w.m2, w.z, {centre_position}, centre, frame, eps);
            check_partition(cw, w.x.rows());
            check_conservation(w, cw);
            CHECK(cw.x.rows() < w.x.rows());
            CHECK(cw.x.rows() >= 2);
        }
    }
}

TEST_CASE("protected positions and the farthest point survive as bit-identical singletons")
{
    // A jittered square grid plus a tight far cluster on the +x axis, under a
    // frame that is wide in x: the cluster's farthest point in PHYSICAL
    // distance is well inside the whitened radius of the corners, so this
    // tells the physical rule from a whitened one.
    Window w = grid_window(41, 2, 3, 4, 0.3);
    const Eigen::Index base = w.x.rows();
    w.x.conservativeResize(base + 3, 2);
    w.m2.conservativeResize(base + 3);
    w.z.conservativeResize(base + 3, 3);
    w.x.row(base + 0) << 1.58, 0.01;
    w.x.row(base + 1) << 1.60, 0.00;
    w.x.row(base + 2) << 1.59, -0.02;
    for ( Eigen::Index j = base; j < base + 3; ++j )
    {
        w.m2(j) = 0.003;
        w.z.row(j) << 0.5, -1.0, 2.0;
    }
    const int farthest = static_cast<int>(base + 1);

    Eigen::MatrixXd L(2, 2);
    L << 4.0, 0.0,
         0.0, 1.0;
    const EllipsoidFrame frame = make_frame(Eigen::VectorXd::Zero(2), L);
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    const double eps = 0.15;

    CHECK(farthest_position(w.x, centre) == farthest);
    // and the corners are farther in WHITENED distance
    const Eigen::MatrixXd u = pullback(frame, w.x);
    CHECK(u.row(0).norm() > u.row(farthest).norm());

    // Two positions in the far field that would otherwise be merged.
    const int p1 = 36 + 41 * 6;   // near (0.8, -0.7)
    const int p2 = 10 + 41 * 30;  // near (-0.5, 0.5)
    {
        const CoarseWindow unprotected =
            coarsen_window(w.x, w.m2, w.z, {}, centre, frame, eps);
        const std::vector<std::vector<int>> cells = members_of(unprotected);
        CHECK(cells[static_cast<std::size_t>(unprotected.cell_of[p1])].size() > 1u);
        CHECK(cells[static_cast<std::size_t>(unprotected.cell_of[p2])].size() > 1u);
        // the farthest point is a singleton even with nothing protected
        check_singleton(w, unprotected, farthest);
        CHECK(unprotected.protected_cells.empty());

        // The rule follows `centre`, and does real work: moved to (0.5, 0),
        // the centre makes a left-hand corner the farthest point. That corner
        // becomes a singleton -- and it sits in a multi-point cell when it is
        // NOT the farthest, so the singleton is the rule's doing.
        Eigen::VectorXd shifted(2);
        shifted << 0.5, 0.0;
        const int corner = farthest_position(w.x, shifted);
        CHECK(corner != farthest);
        CHECK(cells[static_cast<std::size_t>(unprotected.cell_of[corner])].size() > 1u);
        const CoarseWindow recentred =
            coarsen_window(w.x, w.m2, w.z, {}, shifted, frame, eps);
        check_singleton(w, recentred, corner);
    }

    const std::vector<int> protected_positions = {p1, p2};
    const CoarseWindow cw =
        coarsen_window(w.x, w.m2, w.z, protected_positions, centre, frame, eps);
    check_partition(cw, w.x.rows());
    check_conservation(w, cw);
    REQUIRE(cw.protected_cells.size() == 2u);
    for ( std::size_t i = 0; i < protected_positions.size(); ++i )
    {
        const int position = protected_positions[i];
        check_singleton(w, cw, position);
        CHECK(cw.protected_cells[i] == cw.cell_of[static_cast<std::size_t>(position)]);
    }
    check_singleton(w, cw, farthest);

    // Protecting the farthest point itself is not a conflict.
    const CoarseWindow with_far =
        coarsen_window(w.x, w.m2, w.z, {farthest, p1}, centre, frame, eps);
    REQUIRE(with_far.protected_cells.size() == 2u);
    check_singleton(w, with_far, farthest);
    CHECK(with_far.protected_cells[0] == with_far.cell_of[static_cast<std::size_t>(farthest)]);
}

TEST_CASE("every multi-point cell obeys the grading rule on its own box")
{
    const Window w = grid_window(41, 2, 2, 5, 0.3);
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    for ( const EllipsoidFrame& frame : {ball_frame(2, 0.7), sheared_frame_2d()} )
    {
        for ( double eps : {0.1, 0.3} )
        {
            const CoarseWindow cw =
                coarsen_window(w.x, w.m2, w.z, {20 + 41 * 20}, centre, frame, eps);
            check_grading(w, cw, frame, eps);
        }
    }
}

TEST_CASE("coarsen_window is a pure function of its arguments")
{
    const Window w = grid_window(31, 2, 3, 6, 0.3);
    const Eigen::Index count = w.x.rows();
    const EllipsoidFrame frame = sheared_frame_2d();
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    const std::vector<int> protected_positions = {15 + 31 * 15, 4};
    const double eps = 0.2;

    const CoarseWindow first =
        coarsen_window(w.x, w.m2, w.z, protected_positions, centre, frame, eps);
    const CoarseWindow second =
        coarsen_window(w.x, w.m2, w.z, protected_positions, centre, frame, eps);
    CHECK(same_result(first, second));
    CHECK(first.x.rows() < count);

    // The same values in different storage: a copy, and a block of a larger
    // array bound through Ref without copying.
    const Eigen::MatrixXd x_copy = w.x;
    const Eigen::VectorXd m2_copy = w.m2;
    const Eigen::MatrixXd z_copy = w.z;
    const CoarseWindow copied =
        coarsen_window(x_copy, m2_copy, z_copy, protected_positions, centre, frame, eps);
    CHECK(same_result(first, copied));

    Eigen::MatrixXd big_x = Eigen::MatrixXd::Constant(count + 7, 2, 123.0);
    Eigen::VectorXd big_m2 = Eigen::VectorXd::Constant(count + 7, -5.0);
    Eigen::MatrixXd big_z = Eigen::MatrixXd::Constant(count + 7, 3, 99.0);
    big_x.middleRows(3, count) = w.x;
    big_m2.segment(3, count) = w.m2;
    big_z.middleRows(3, count) = w.z;
    const CoarseWindow blocked =
        coarsen_window(big_x.middleRows(3, count), big_m2.segment(3, count),
                       big_z.middleRows(3, count), protected_positions, centre, frame, eps);
    CHECK(same_result(first, blocked));
}

TEST_CASE("an anisotropic window frame elongates the cells the same way")
{
    // The window IS the frame's ellipsoid, as a fit window is. With L =
    // diag(4, 1) the whitened bounding box is a square, so the cells are
    // squares in u and 4:1 rectangles in physical space; under a ball frame
    // on a ball window they are squares.
    Eigen::MatrixXd L(2, 2);
    L << 4.0, 0.0,
         0.0, 1.0;
    const EllipsoidFrame wide = make_frame(Eigen::VectorXd::Zero(2), L);
    const EllipsoidFrame round = ball_frame(2, 1.0);
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    const double eps = 0.2;

    auto extent_ratio = []( const Window& w, const CoarseWindow& cw ) {
        double sum_x = 0.0, sum_y = 0.0;
        for ( const std::vector<int>& members : members_of(cw) )
        {
            if ( members.size() < 2 )
            {
                continue;
            }
            Eigen::Vector2d lo = w.x.row(members.front()).transpose();
            Eigen::Vector2d hi = lo;
            for ( int j : members )
            {
                lo = lo.cwiseMin(w.x.row(j).transpose());
                hi = hi.cwiseMax(w.x.row(j).transpose());
            }
            sum_x += hi(0) - lo(0);
            sum_y += hi(1) - lo(1);
        }
        return sum_x / sum_y;
    };

    const Window elongated = ellipsoid_window(wide, 61, 2, 7);
    const CoarseWindow cw_wide =
        coarsen_window(elongated.x, elongated.m2, elongated.z, {}, centre, wide, eps);
    check_partition(cw_wide, elongated.x.rows());
    check_conservation(elongated, cw_wide);
    check_grading(elongated, cw_wide, wide, eps);
    const double ratio_wide = extent_ratio(elongated, cw_wide);

    const Window ball = ellipsoid_window(round, 61, 2, 7);
    const CoarseWindow cw_round =
        coarsen_window(ball.x, ball.m2, ball.z, {}, centre, round, eps);
    const double ratio_round = extent_ratio(ball, cw_round);

    MESSAGE("physical x:y extent ratio of the multi-point cells: wide frame "
            << ratio_wide << ", ball frame " << ratio_round);
    CHECK(ratio_wide > 2.5);
    CHECK(ratio_wide < 6.0);
    CHECK(ratio_round > 0.7);
    CHECK(ratio_round < 1.4);
}

TEST_CASE("the cell count follows 3 pi / eps^2 log2(R / h) and falls with eps")
{
    // K ~ 1e4 on a regular grid with the centre ON a grid point, as a row's
    // centre is. The law counts about 3 pi / eps^2 cells per dyadic annulus;
    // grading on the diagonal runs above that constant, so the check is a
    // factor-of-two band, plus monotonicity and "far below K".
    const int per_side = 101;
    const Window w = grid_window(per_side, 2, 2, 8, 0.0);
    const Eigen::Index count = w.x.rows();
    const int centre_position = 50 + per_side * 50;
    const Eigen::VectorXd centre = w.x.row(centre_position).transpose();
    REQUIRE(centre.norm() == 0.0);
    const EllipsoidFrame frame = ball_frame(2, 1.0);
    const double radius = (w.x.rowwise() - centre.transpose()).rowwise().norm().maxCoeff();
    const double h = w.spacing;

    std::vector<Eigen::Index> counts;
    for ( double eps : {0.1, 0.2} )
    {
        const CoarseWindow cw =
            coarsen_window(w.x, w.m2, w.z, {centre_position}, centre, frame, eps);
        check_partition(cw, count);
        check_conservation(w, cw);
        check_grading(w, cw, frame, eps);
        check_singleton(w, cw, centre_position);
        const double law = 3.0 * M_PI / (eps * eps) * std::log2(radius / h);
        MESSAGE("eps = " << eps << ": " << cw.x.rows() << " cells of " << count
                         << " points; law predicts " << law << " (ratio "
                         << cw.x.rows() / law << ")");
        CHECK(cw.x.rows() >= 0.5 * law);
        CHECK(cw.x.rows() <= 2.0 * law);
        counts.push_back(cw.x.rows());
    }
    CHECK(counts[1] < counts[0]);
    CHECK(counts[0] < count);
    CHECK(counts[1] < count / 4);
}

TEST_CASE("cell_of is a partition sorted by smallest member")
{
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    for ( int per_side : {2, 3, 7, 25} )
    {
        const Window w = grid_window(per_side, 2, 1, 9, 0.3);
        for ( double eps : {0.05, 0.5, 2.0} )
        {
            const CoarseWindow cw =
                coarsen_window(w.x, w.m2, w.z, {0}, centre, ball_frame(2, 1.0), eps);
            check_partition(cw, w.x.rows());
            check_conservation(w, cw);
        }
    }
}

TEST_CASE("3D: conservation, protected singletons and the grading rule on an octree")
{
    const int per_side = 21;
    const Window w = grid_window(per_side, 3, 3, 10, 0.3);
    Eigen::MatrixXd L(3, 3);
    L <<  1.0, 0.0, 0.0,
          0.3, 0.7, 0.0,
         -0.2, 0.1, 1.5;
    const EllipsoidFrame frame = make_frame(Eigen::VectorXd::Zero(3), L);
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(3);
    const int centre_position = 10 + per_side * 10 + per_side * per_side * 10;
    const int far_position = 18 + per_side * 3 + per_side * per_side * 16;
    const std::vector<int> protected_positions = {centre_position, far_position};
    const int farthest = farthest_position(w.x, centre);
    CHECK(farthest != far_position);

    std::vector<Eigen::Index> counts;
    for ( double eps : {0.3, 0.6} )
    {
        const CoarseWindow cw =
            coarsen_window(w.x, w.m2, w.z, protected_positions, centre, frame, eps);
        counts.push_back(cw.x.rows());
        check_partition(cw, w.x.rows());
        check_conservation(w, cw);
        check_grading(w, cw, frame, eps);
        check_singleton(w, cw, centre_position);
        check_singleton(w, cw, far_position);
        check_singleton(w, cw, farthest);
        REQUIRE(cw.protected_cells.size() == 2u);
        CHECK(cw.protected_cells[0] == cw.cell_of[static_cast<std::size_t>(centre_position)]);
        CHECK(cw.protected_cells[1] == cw.cell_of[static_cast<std::size_t>(far_position)]);
        CHECK(cw.x.rows() < w.x.rows());
        MESSAGE("3D eps = " << eps << ": " << cw.x.rows() << " cells of " << w.x.rows());
    }
    CHECK(counts[1] < counts[0]);  // coarser with eps, as in 2D
}

TEST_CASE("coincident points share a cell and the split still terminates")
{
    // The documented caveat: a protected point coincident with another cannot
    // be isolated. The call must still return, deterministically, with the
    // pair in one cell and everything conserved.
    Window w;
    w.x.resize(4, 2);
    w.x << 0.5, 0.5,
           0.5, 0.5,
          -0.7, 0.1,
           0.2, -0.9;
    w.m2.resize(4);
    w.m2 << 1.0, 2.0, 3.0, 4.0;
    w.z.resize(4, 2);
    w.z << 1.0, 2.0,
           3.0, 4.0,
           5.0, 6.0,
           7.0, 8.0;
    const Eigen::VectorXd centre = Eigen::VectorXd::Zero(2);
    const EllipsoidFrame frame = ball_frame(2, 1.0);

    const CoarseWindow cw = coarsen_window(w.x, w.m2, w.z, {0}, centre, frame, 0.2);
    check_partition(cw, 4);
    check_conservation(w, cw);
    CHECK(cw.cell_of[0] == cw.cell_of[1]);
    CHECK(cw.protected_cells == std::vector<int>{cw.cell_of[0]});
    CHECK(cw.m2(cw.cell_of[0]) == 3.0);
    CHECK(same_result(cw, coarsen_window(w.x, w.m2, w.z, {0}, centre, frame, 0.2)));

    // Every point coincident: one cell, unless eps == 0 asks for the identity.
    const Eigen::MatrixXd same = Eigen::MatrixXd::Constant(3, 2, 0.25);
    const CoarseWindow one =
        coarsen_window(same, w.m2.head(3), w.z.topRows(3), {1}, centre, frame, 0.2);
    CHECK(one.x.rows() == 1);
    CHECK(one.m2(0) == 6.0);
    CHECK(one.protected_cells == std::vector<int>{0});
    const CoarseWindow all =
        coarsen_window(same, w.m2.head(3), w.z.topRows(3), {1}, centre, frame, 0.0);
    CHECK(all.x.rows() == 3);
}
