// SPDX-License-Identifier: MIT
//
// The H_r-orthonormal mode block: merging keeps the representation exact,
// existing columns untouched, and the pencil eigenvalues analytic. Reference
// computations are dense and independent of the block's own arithmetic.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/hr_oracle.hpp"
#include "lgpsf/corrections/mode_block.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::HrOracle;
using lgpsf::corrections::ModeBlock;
using lgpsf::corrections::Provenance;
using lgpsf::corrections::apply_correction;
using lgpsf::corrections::empty_block;
using lgpsf::corrections::merge;
using lgpsf::corrections::pencil_eigenvalues;
using lgpsf::corrections::sparse_hr_oracle;

namespace {

Eigen::SparseMatrix<double> spd_sparse( int n )
{
    std::vector<Eigen::Triplet<double>> entries;
    for ( int i = 0; i < n; ++i )
    {
        entries.emplace_back(i, i, 2.5 + 0.01 * i);
        if ( i + 1 < n )
        {
            entries.emplace_back(i, i + 1, -1.0);
            entries.emplace_back(i + 1, i, -1.0);
        }
    }
    Eigen::SparseMatrix<double> Hr(n, n);
    Hr.setFromTriplets(entries.begin(), entries.end());
    return Hr;
}

Eigen::MatrixXd random_symmetric( int q, std::mt19937& gen )
{
    const Eigen::MatrixXd G = test_helpers::randn_points(q, q, gen);
    return 0.5 * (G + G.transpose());
}

/// The dense truth: Hr V C V^T Hr, no block machinery involved.
Eigen::MatrixXd dense_contribution( const Eigen::MatrixXd& Hr,
                                    const Eigen::MatrixXd& V,
                                    const Eigen::MatrixXd& C )
{
    return Hr * V * (0.5 * (C + C.transpose())) * V.transpose() * Hr;
}

} // namespace

TEST_CASE("a merged contribution is represented exactly")
{
    std::mt19937 gen(0);
    const int n = 40;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(Hr);
    const HrOracle oracle = sparse_hr_oracle(Hr);

    ModeBlock block = empty_block(n);
    CHECK(block.rank() == 0);
    CHECK(apply_correction(block, Eigen::MatrixXd::Identity(n, n)).isZero());

    const Eigen::MatrixXd V_new = test_helpers::randn_points(n, 5, gen);
    const Eigen::MatrixXd C_new = random_symmetric(5, gen);
    const auto report = merge(block, oracle, V_new, C_new,
                              Provenance::Deflation);
    CHECK(report.requested == 5);
    CHECK(report.added == 5);
    CHECK(block.rank() == 5);
    CHECK(lgpsf::corrections::validate(block).empty());
    CHECK(block.tags == std::vector<Provenance>(5, Provenance::Deflation));

    // the invariants: H_r-orthonormal columns, HrV consistent with V
    CHECK((block.V.transpose() * Hrd * block.V
           - Eigen::MatrixXd::Identity(5, 5)).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((block.HrV - Hrd * block.V).cwiseAbs().maxCoeff()
          < 1e-12 * block.HrV.cwiseAbs().maxCoeff());

    // the representation: E == Hr V_new C_new V_new^T Hr, densely
    const Eigen::MatrixXd expected = dense_contribution(Hrd, V_new, C_new);
    const Eigen::MatrixXd X = test_helpers::randn_points(n, 3, gen);
    CHECK((apply_correction(block, X) - expected * X).cwiseAbs().maxCoeff()
          < 1e-11 * (expected * X).cwiseAbs().maxCoeff());
}

TEST_CASE("merging the same directions twice adds coefficients, not columns")
{
    std::mt19937 gen(1);
    const int n = 30;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(Hr);
    const HrOracle oracle = sparse_hr_oracle(Hr);

    const Eigen::MatrixXd V1 = test_helpers::randn_points(n, 4, gen);
    const Eigen::MatrixXd C1 = random_symmetric(4, gen);
    const Eigen::MatrixXd C2 = random_symmetric(4, gen);

    ModeBlock block = empty_block(n);
    merge(block, oracle, V1, C1, Provenance::Flip);
    const auto second = merge(block, oracle, V1, C2, Provenance::Deflation);
    CHECK(second.requested == 4);
    CHECK(second.added == 0);
    CHECK(block.rank() == 4);
    // no new columns, so no new tags either
    CHECK(block.tags == std::vector<Provenance>(4, Provenance::Flip));

    const Eigen::MatrixXd expected = dense_contribution(Hrd, V1, C1 + C2);
    const Eigen::MatrixXd X = test_helpers::randn_points(n, 3, gen);
    CHECK((apply_correction(block, X) - expected * X).cwiseAbs().maxCoeff()
          < 1e-11 * (expected * X).cwiseAbs().maxCoeff());
}

TEST_CASE("overlapping candidates split into folded and added parts")
{
    std::mt19937 gen(2);
    const int n = 35;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(Hr);
    const HrOracle oracle = sparse_hr_oracle(Hr);

    const Eigen::MatrixXd V1 = test_helpers::randn_points(n, 3, gen);
    const Eigen::MatrixXd C1 = random_symmetric(3, gen);
    ModeBlock block = empty_block(n);
    merge(block, oracle, V1, C1, Provenance::PencilCache);

    // one candidate inside span(V1), two genuinely new
    Eigen::MatrixXd V2(n, 3);
    V2.col(0) = 2.0 * V1.col(0) - V1.col(2);
    V2.col(1) = test_helpers::randn_points(n, 1, gen).col(0);
    V2.col(2) = test_helpers::randn_points(n, 1, gen).col(0);
    const Eigen::MatrixXd C2 = random_symmetric(3, gen);

    const auto report = merge(block, oracle, V2, C2, Provenance::ValuePass);
    CHECK(report.requested == 3);
    CHECK(report.added == 2);
    CHECK(block.rank() == 5);
    CHECK(block.tags[2] == Provenance::PencilCache);
    CHECK(block.tags[3] == Provenance::ValuePass);

    const Eigen::MatrixXd expected =
        dense_contribution(Hrd, V1, C1) + dense_contribution(Hrd, V2, C2);
    const Eigen::MatrixXd X = test_helpers::randn_points(n, 4, gen);
    CHECK((apply_correction(block, X) - expected * X).cwiseAbs().maxCoeff()
          < 1e-10 * (expected * X).cwiseAbs().maxCoeff());
}

TEST_CASE("the block's pencil eigenvalues are exact eigenpairs against H_r")
{
    std::mt19937 gen(3);
    const int n = 30;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(Hr);
    const HrOracle oracle = sparse_hr_oracle(Hr);

    ModeBlock block = empty_block(n);
    merge(block, oracle, test_helpers::randn_points(n, 4, gen),
          random_symmetric(4, gen), Provenance::Flip);

    // E (V u_i) = theta_i H_r (V u_i) for every eigenpair (theta_i, u_i) of C
    const Eigen::VectorXd theta = pencil_eigenvalues(block);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(block.C);
    for ( int i = 0; i < 4; ++i )
    {
        const Eigen::VectorXd v = block.V * eigen.eigenvectors().col(i);
        const Eigen::VectorXd Ev =
            apply_correction(block, Eigen::MatrixXd(v)).col(0);
        CHECK((Ev - theta(i) * (Hrd * v)).norm() < 1e-11 * (Hrd * v).norm()
              + 1e-11 * std::abs(theta(i)) * (Hrd * v).norm());
    }
}

TEST_CASE("validate catches structural damage")
{
    std::mt19937 gen(4);
    const int n = 20;
    const HrOracle oracle = sparse_hr_oracle(spd_sparse(n));
    ModeBlock good = empty_block(n);
    merge(good, oracle, test_helpers::randn_points(n, 2, gen),
          random_symmetric(2, gen), Provenance::Deflation);
    REQUIRE(lgpsf::corrections::validate(good).empty());

    ModeBlock wrong_hrv = good;
    wrong_hrv.HrV = wrong_hrv.HrV.leftCols(1).eval();
    CHECK(!lgpsf::corrections::validate(wrong_hrv).empty());

    ModeBlock asym = good;
    asym.C(0, 1) += 1e-13;
    CHECK(!lgpsf::corrections::validate(asym).empty());

    ModeBlock short_tags = good;
    short_tags.tags.pop_back();
    CHECK(!lgpsf::corrections::validate(short_tags).empty());

    // misuse of merge is refused
    CHECK_THROWS_AS(merge(good, oracle, Eigen::MatrixXd::Zero(n + 1, 1),
                          Eigen::MatrixXd::Zero(1, 1), Provenance::Flip),
                    std::invalid_argument);
    CHECK_THROWS_AS(merge(good, oracle, Eigen::MatrixXd::Zero(n, 2),
                          Eigen::MatrixXd::Zero(1, 1), Provenance::Flip),
                    std::invalid_argument);
}
