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
