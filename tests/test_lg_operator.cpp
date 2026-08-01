// SPDX-License-Identifier: MIT
//
// Checks on LGOperator as a data structure independent of any fit.
//
// Every operator here is built BY HAND -- this file never calls fit_operator,
// and never includes operator_fit.hpp. That is the point: an operator from a
// physics-based approximation, a merge of chunk fits, or a file must work
// exactly as one that came from the fitter.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/lg_operator.hpp"
#include "test_helpers.hpp"

using lgpsf::LGExpansion;
using lgpsf::LGOperator;
using lgpsf::Mode;
using lgpsf::MuMode;
using lgpsf::assemble_sparse;
using lgpsf::concatenate_rows;
using lgpsf::eval_entries;
using lgpsf::eval_kernel;
using lgpsf::matvec;
using lgpsf::model_rows;
using lgpsf::modes_up_to_level;
using lgpsf::spike_measure;
using lgpsf::theta_size;
using lgpsf::to_theta;
using lgpsf::validate;

namespace {

/// An operator written down directly, the way a caller with a physics-based
/// approximation would: pick an ellipsoid and coefficients per row, pick a
/// window, done. No probes, no fitting, no diagnostics.
LGOperator hand_built( std::mt19937& gen, int per_side = 9, int num_modeled = 3,
                       double window_radius = 0.6 )
{
    LGOperator op;
    op.dim = 2;
    const int count = per_side * per_side;

    op.x_cols.resize(count, 2);
    int row = 0;
    for ( int i = 0; i < per_side; ++i )
    {
        for ( int j = 0; j < per_side; ++j )
        {
            op.x_cols(row, 0) = -1.0 + 2.0 * i / (per_side - 1);
            op.x_cols(row, 1) = -1.0 + 2.0 * j / (per_side - 1);
            ++row;
        }
    }
    op.m1_diag = test_helpers::uniform_points(count, 1, gen, 0.7, 1.5).col(0);
    op.m2_diag = test_helpers::uniform_points(count, 1, gen, 0.4, 1.1).col(0);
    op.spike = true;
    op.mode_sets = {modes_up_to_level(2, 1)};

    const int params = theta_size(2);
    const auto modes = op.mode_sets[0].size();
    op.theta = Eigen::MatrixXd::Constant(count, params,
                                         std::numeric_limits<double>::quiet_NaN());
    op.mu = Eigen::MatrixXd::Constant(count, 2, std::numeric_limits<double>::quiet_NaN());
    op.L = Eigen::MatrixXd::Constant(count, 4, std::numeric_limits<double>::quiet_NaN());
    op.c = Eigen::MatrixXd::Zero(count, static_cast<Eigen::Index>(modes));
    op.s = Eigen::VectorXd::Zero(count);
    op.mode_set_id.assign(static_cast<std::size_t>(count), -1);
    op.window_center = Eigen::MatrixXd::Constant(count, 2,
                                                 std::numeric_limits<double>::quiet_NaN());
    op.window_covariance = Eigen::MatrixXd::Constant(
        count, 4, std::numeric_limits<double>::quiet_NaN());
    op.window_indptr.assign(static_cast<std::size_t>(count) + 1, 0);

    const int stride = count / (num_modeled + 1);
    std::vector<int> modeled;
    for ( int m = 0; m < num_modeled; ++m )
    {
        modeled.push_back(stride * (m + 1));
    }

    for ( int rho = 0; rho < count; ++rho )
    {
        const bool has = std::find(modeled.begin(), modeled.end(), rho) != modeled.end();
        std::vector<int> window;
        if ( has )
        {
            const Eigen::VectorXd center = op.x_cols.row(rho).transpose();

            // a plainly-stated ellipsoid: axes 0.25 and 0.18, slightly tilted
            Eigen::VectorXd theta_hat(3);
            theta_hat << std::log(0.25), std::log(0.18), 0.04;
            op.theta.row(rho) = to_theta(theta_hat, center, MuMode::Pinned).transpose();
            const lgpsf::EllipsoidFrame frame =
                lgpsf::unpack_theta_hat(theta_hat, center, MuMode::Pinned);
            op.mu.row(rho) = frame.mu.transpose();
            for ( int i = 0; i < 2; ++i )
            {
                for ( int j = 0; j < 2; ++j )
                {
                    op.L(rho, i * 2 + j) = frame.L(i, j);
                }
            }
            op.c.row(rho) =
                test_helpers::randn_points(static_cast<int>(modes), 1, gen).col(0).transpose();
            op.s(rho) = 0.3;
            op.mode_set_id[static_cast<std::size_t>(rho)] = 0;

            // an isotropic window of the given radius
            op.window_center.row(rho) = center.transpose();
            for ( int i = 0; i < 2; ++i )
            {
                for ( int j = 0; j < 2; ++j )
                {
                    op.window_covariance(rho, i * 2 + j) =
                        ( i == j ) ? window_radius * window_radius : 0.0;
                }
            }
            for ( int j = 0; j < count; ++j )
            {
                if ( (op.x_cols.row(j).transpose() - center).norm() <= window_radius )
                {
                    window.push_back(j);
                }
            }
        }
        op.window_indptr[static_cast<std::size_t>(rho) + 1] =
            op.window_indptr[static_cast<std::size_t>(rho)]
            + static_cast<int>(window.size());
        op.window_indices.insert(op.window_indices.end(), window.begin(), window.end());
    }
    return op;
}

} // namespace

TEST_CASE("an operator written down by hand behaves like any other")
{
    std::mt19937 gen(0);
    const LGOperator op = hand_built(gen);

    CHECK(validate(op).empty());
    CHECK(model_rows(op).size() == 3u);
    for ( int rho : model_rows(op) )
    {
        CHECK(op.has_model(rho));
        CHECK(op.row_modes(rho).size() == modes_up_to_level(2, 1).size());
        CHECK(!op.row_window(rho).empty());
    }

    // it evaluates, applies and assembles with no fit anywhere in sight
    const Eigen::MatrixXd v =
        test_helpers::randn_points(static_cast<int>(op.num_cols()), 2, gen);
    const Eigen::MatrixXd applied = matvec(op, v);
    const Eigen::SparseMatrix<double> assembled = assemble_sparse(op, 40.0);
    CHECK((assembled * v - applied).cwiseAbs().maxCoeff() < 1e-10);
    CHECK(spike_measure(op)(model_rows(op).front())
          == doctest::Approx(op.m1_diag(model_rows(op).front()) * 0.3));
}

TEST_CASE("validate catches the ways a hand-built operator goes wrong")
{
    std::mt19937 gen(1);
    const LGOperator good = hand_built(gen);
    REQUIRE(validate(good).empty());

    // a mode-set id naming nothing
    LGOperator bad_id = good;
    bad_id.mode_set_id[static_cast<std::size_t>(model_rows(good).front())] = 7;
    CHECK(!validate(bad_id).empty());

    // coefficients too narrow to hold the row's mode set
    LGOperator narrow = good;
    narrow.c = narrow.c.leftCols(1).eval();
    CHECK(!validate(narrow).empty());

    // a window offset table that is not monotone
    LGOperator jumbled = good;
    jumbled.window_indptr[2] = jumbled.window_indptr[1] - 1;
    CHECK(!validate(jumbled).empty());

    // a window index off the end of the columns
    LGOperator stray = good;
    stray.window_indices.front() = static_cast<int>(good.num_cols()) + 5;
    CHECK(!validate(stray).empty());

    // theta in the wrong encoding width
    LGOperator wide = good;
    wide.theta = Eigen::MatrixXd::Zero(good.num_rows(), 3);
    CHECK(!validate(wide).empty());

    // a spike alongside separate row coordinates
    LGOperator rectangular = good;
    rectangular.x_rows = Eigen::MatrixXd(good.x_cols);
    CHECK(!validate(rectangular).empty());

    // and a row-count mismatch is reported rather than crashing
    LGOperator short_s = good;
    short_s.s = Eigen::VectorXd::Zero(3);
    CHECK(!validate(short_s).empty());
}

TEST_CASE("concatenating rows merges chunk operators without remapping by hand")
{
    // The motivating case: per-chunk fits gathered into a whole-operator one.
    // The two index remappings -- mode-set ids into a combined table, window
    // offsets by a running total -- are what this exists to get right.
    std::mt19937 gen(2);
    const LGOperator whole = hand_built(gen, 9, 3);

    // split it into two parts by row, the way chunked fitting would produce them
    const Eigen::Index split = whole.num_rows() / 2;
    std::vector<LGOperator> parts;
    for ( int half = 0; half < 2; ++half )
    {
        const Eigen::Index begin = ( half == 0 ) ? 0 : split;
        const Eigen::Index end = ( half == 0 ) ? split : whole.num_rows();
        LGOperator part = whole;
        part.m1_diag = whole.m1_diag.segment(begin, end - begin);
        part.theta = whole.theta.middleRows(begin, end - begin);
        part.mu = whole.mu.middleRows(begin, end - begin);
        part.L = whole.L.middleRows(begin, end - begin);
        part.c = whole.c.middleRows(begin, end - begin);
        part.s = whole.s.segment(begin, end - begin);
        part.window_center = whole.window_center.middleRows(begin, end - begin);
        part.window_covariance = whole.window_covariance.middleRows(begin, end - begin);
        part.mode_set_id.assign(whole.mode_set_id.begin() + begin,
                                whole.mode_set_id.begin() + end);
        part.window_indptr.assign(static_cast<std::size_t>(end - begin) + 1, 0);
        part.window_indices.clear();
        for ( Eigen::Index rho = begin; rho < end; ++rho )
        {
            const std::vector<int> window = whole.row_window(static_cast<int>(rho));
            const std::size_t local = static_cast<std::size_t>(rho - begin);
            part.window_indptr[local + 1] =
                part.window_indptr[local] + static_cast<int>(window.size());
            part.window_indices.insert(part.window_indices.end(), window.begin(),
                                       window.end());
        }
        parts.push_back(std::move(part));
    }

    const LGOperator merged = concatenate_rows(parts);
    CHECK(validate(merged).empty());
    CHECK(merged.num_rows() == whole.num_rows());
    CHECK(merged.mode_set_id == whole.mode_set_id);
    CHECK(merged.window_indptr == whole.window_indptr);
    CHECK(merged.window_indices == whole.window_indices);

    // the merged operator IS the original, as an operator
    const Eigen::MatrixXd v =
        test_helpers::randn_points(static_cast<int>(whole.num_cols()), 2, gen);
    CHECK((matvec(merged, v) - matvec(whole, v)).cwiseAbs().maxCoeff() < 1e-12);

    // mode sets are de-duplicated, not appended blindly
    CHECK(merged.mode_sets.size() == 1u);

    // parts that disagree about the column space are refused
    std::vector<LGOperator> mismatched = parts;
    mismatched[1].m2_diag(0) += 1.0;
    CHECK_THROWS_AS(concatenate_rows(mismatched), std::invalid_argument);
    CHECK_THROWS_AS(concatenate_rows({}), std::invalid_argument);
}

TEST_CASE("the row-parallel helpers are bit-identical across thread counts")
{
    std::mt19937 gen(3);
    const LGOperator op = hand_built(gen, 11, 6);
    const Eigen::MatrixXd v =
        test_helpers::randn_points(static_cast<int>(op.num_cols()), 3, gen);

    CHECK(matvec(op, v, std::numeric_limits<double>::infinity(), 1)
          == matvec(op, v, std::numeric_limits<double>::infinity(), 4));

    const Eigen::MatrixXd serial =
        Eigen::MatrixXd(assemble_sparse(op, 3.0, lgpsf::Symmetrize::None, 1));
    const Eigen::MatrixXd parallel =
        Eigen::MatrixXd(assemble_sparse(op, 3.0, lgpsf::Symmetrize::None, 4));
    CHECK(serial == parallel);

    const Eigen::MatrixXd weighted_serial =
        Eigen::MatrixXd(assemble_sparse(op, 3.0, lgpsf::Symmetrize::Weighted, 1));
    const Eigen::MatrixXd weighted_parallel =
        Eigen::MatrixXd(assemble_sparse(op, 3.0, lgpsf::Symmetrize::Weighted, 4));
    CHECK(weighted_serial == weighted_parallel);

    Eigen::MatrixXd HV = test_helpers::randn_points(static_cast<int>(op.num_rows()), 3, gen);
    const Eigen::VectorXd one = lgpsf::qc_map(op, v, HV, 1);
    const Eigen::VectorXd four = lgpsf::qc_map(op, v, HV, 4);
    for ( Eigen::Index rho = 0; rho < one.size(); ++rho )
    {
        CHECK(std::isnan(one(rho)) == std::isnan(four(rho)));
        if ( !std::isnan(one(rho)) )
        {
            CHECK(one(rho) == four(rho));
        }
    }
}

TEST_CASE("weighted symmetrization matches the formula and is exactly symmetric")
{
    std::mt19937 gen(9);
    const LGOperator op = hand_built(gen, 11, 6);

    const Eigen::MatrixXd A = Eigen::MatrixXd(assemble_sparse(op, 3.0));
    const Eigen::MatrixXd B =
        Eigen::MatrixXd(assemble_sparse(op, 3.0, lgpsf::Symmetrize::Weighted));

    // bitwise symmetric, not approximately: both orientations of an entry are
    // the same two products summed
    CHECK(B == Eigen::MatrixXd(B.transpose()));

    // the dense formula, written out independently of the sparse implementation
    std::vector<double> norms;
    for ( Eigen::Index i = 0; i < A.rows(); ++i )
    {
        if ( A.row(i).norm() > 0.0 )
        {
            norms.push_back(A.row(i).norm());
        }
    }
    std::sort(norms.begin(), norms.end());
    const std::size_t half = norms.size() / 2;
    const double median = ( norms.size() % 2 == 1 )
                              ? norms[half]
                              : 0.5 * (norms[half - 1] + norms[half]);
    Eigen::VectorXd w2(A.rows());
    for ( Eigen::Index i = 0; i < A.rows(); ++i )
    {
        w2(i) = 1.0 / (A.row(i).squaredNorm() + std::pow(1e-2 * median, 2));
    }
    Eigen::MatrixXd reference(A.rows(), A.cols());
    for ( Eigen::Index i = 0; i < A.rows(); ++i )
    {
        for ( Eigen::Index j = 0; j < A.cols(); ++j )
        {
            reference(i, j) =
                (w2(i) * A(i, j) + w2(j) * A(j, i)) / (w2(i) + w2(j));
        }
    }
    CHECK((B - reference).cwiseAbs().maxCoeff()
          < 1e-14 * reference.cwiseAbs().maxCoeff());

    // rectangular operators are refused, same as Average
    LGOperator rect = op;
    rect.spike = false;
    rect.x_rows = Eigen::MatrixXd(op.x_cols);
    CHECK_THROWS_AS(assemble_sparse(rect, 3.0, lgpsf::Symmetrize::Weighted),
                    std::invalid_argument);
}

TEST_CASE("weighted symmetrization lets the weak row own the disagreement")
{
    // Row 0 is weak (entries ~1e-3), row 1 strong (~1); row 1 grazes row 0 at
    // entry (1,0) with a value 250x row 0's own scale. Plain averaging would
    // drag the reconciled entry to ~0.25; the inverse-row-energy weights leave
    // it within ~0.1% of the weak row's opinion.
    std::vector<Eigen::Triplet<double>> entries = {
        {0, 0, 1e-3}, {0, 1, 2e-3}, {1, 0, 0.5}, {1, 1, 1.0}, {2, 2, 0.1}};
    Eigen::SparseMatrix<double> A(3, 3);
    A.setFromTriplets(entries.begin(), entries.end());

    const Eigen::MatrixXd B =
        Eigen::MatrixXd(lgpsf::detail::weighted_symmetrize(A));
    CHECK(B == Eigen::MatrixXd(B.transpose()));
    CHECK(std::abs(B(0, 1) - 2e-3) < 1e-5);

    // an already-symmetric matrix is a fixed point (up to roundoff)
    const Eigen::SparseMatrix<double> S =
        0.5 * (Eigen::SparseMatrix<double>(A)
               + Eigen::SparseMatrix<double>(A.transpose()));
    CHECK(Eigen::MatrixXd(lgpsf::detail::weighted_symmetrize(S))
              .isApprox(Eigen::MatrixXd(S), 1e-14));
}

TEST_CASE("an expansion decodes and evaluates on its own")
{
    std::mt19937 gen(4);
    const LGOperator op = hand_built(gen);
    const int rho = model_rows(op).front();

    const lgpsf::LGExpansion expansion = lgpsf::row_expansion(op, rho);
    CHECK(lgpsf::validate(expansion).empty());
    CHECK(expansion.dim() == 2);
    CHECK(expansion.num_modes() == op.row_modes(rho).size());

    // it carries its own ellipsoid: no reference center, no mu mode
    const lgpsf::EllipsoidFrame frame = expansion.frame();
    CHECK((frame.mu - op.mu.row(rho).transpose()).cwiseAbs().maxCoeff() < 1e-12);

    // and its smooth part is the operator's raw kernel for that row
    const Eigen::MatrixXd probes = test_helpers::uniform_points(20, 2, gen, -1.0, 1.0);
    CHECK((lgpsf::eval_expansion(expansion, probes)
           - lgpsf::eval_kernel_unrestricted(op, {rho}, probes).col(0))
              .cwiseAbs().maxCoeff() < 1e-12);

    // the spike travels with it, as the second component
    REQUIRE(expansion.s.size() == 1);
    CHECK(expansion.s(0) == op.s(rho));
}

TEST_CASE("validate catches a malformed expansion")
{
    std::mt19937 gen(5);
    const lgpsf::LGExpansion good = lgpsf::row_expansion(hand_built(gen), 40);
    REQUIRE(lgpsf::validate(good).empty());

    lgpsf::LGExpansion mismatched = good;
    mismatched.c = Eigen::VectorXd::Zero(good.c.size() + 1);
    CHECK(!lgpsf::validate(mismatched).empty());

    lgpsf::LGExpansion bad_theta = good;
    bad_theta.theta = Eigen::VectorXd::Zero(4);  // no N has N(N+3)/2 == 4
    CHECK(!lgpsf::validate(bad_theta).empty());

    lgpsf::LGExpansion off_table = good;
    off_table.modes[0] = lgpsf::Mode{0, 99, 0};
    CHECK(!lgpsf::validate(off_table).empty());

    lgpsf::LGExpansion infinite = good;
    infinite.c(0) = std::numeric_limits<double>::infinity();
    CHECK(!lgpsf::validate(infinite).empty());
}

TEST_CASE("an operator can be built from per-row expansions")
{
    // The hand-construction path: an operator row IS an expansion plus a window
    // plus the masses, so a caller with a physics-based approximation writes
    // expansions and lets build_operator do the bookkeeping -- mode-set
    // de-duplication, coefficient padding, window offsets.
    std::mt19937 gen(6);
    const LGOperator original = hand_built(gen);

    std::vector<std::optional<lgpsf::OperatorRow>> rows(
        static_cast<std::size_t>(original.num_rows()));
    for ( int rho : model_rows(original) )
    {
        lgpsf::OperatorRow row;
        row.model = lgpsf::row_expansion(original, rho);
        row.window_center = original.window_center.row(rho).transpose();
        row.window_covariance.resize(2, 2);
        for ( int i = 0; i < 2; ++i )
        {
            for ( int j = 0; j < 2; ++j )
            {
                row.window_covariance(i, j) = original.window_covariance(rho, i * 2 + j);
            }
        }
        row.window_columns = original.row_window(rho);
        rows[static_cast<std::size_t>(rho)] = std::move(row);
    }

    const LGOperator rebuilt =
        lgpsf::build_operator(original.x_cols, original.m1_diag, original.m2_diag,
                              original.spike, rows);

    CHECK(validate(rebuilt).empty());
    CHECK(rebuilt.mode_set_id == original.mode_set_id);
    CHECK(rebuilt.window_indptr == original.window_indptr);
    CHECK(rebuilt.window_indices == original.window_indices);
    CHECK(rebuilt.mode_sets.size() == 1u);  // de-duplicated, not appended per row

    // and it is the same operator
    const Eigen::MatrixXd v =
        test_helpers::randn_points(static_cast<int>(original.num_cols()), 2, gen);
    CHECK((matvec(rebuilt, v) - matvec(original, v)).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(Eigen::MatrixXd(assemble_sparse(rebuilt, 40.0))
              .isApprox(Eigen::MatrixXd(assemble_sparse(original, 40.0)), 1e-12));

    // a malformed expansion is refused rather than stored
    std::vector<std::optional<lgpsf::OperatorRow>> broken = rows;
    broken[static_cast<std::size_t>(model_rows(original).front())]->model.c =
        Eigen::VectorXd::Zero(1);
    CHECK_THROWS_AS(lgpsf::build_operator(original.x_cols, original.m1_diag,
                                          original.m2_diag, original.spike, broken),
                    std::invalid_argument);
}
