// SPDX-License-Identifier: MIT
//
// The ShiftedOperator struct and its GLR deployment: the Woodbury inverse is
// exact at every shift with zero refactorization, its PD certificate is
// analytic and agrees with dense truth, and construction refuses what the
// contracts refuse. References are dense and independent.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/shifted_operator.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::BuildOptions;
using lgpsf::corrections::HrOracle;
using lgpsf::corrections::ModeBlock;
using lgpsf::corrections::ProbeArchive;
using lgpsf::corrections::Provenance;
using lgpsf::corrections::ShiftedOperator;
using lgpsf::corrections::SymmetricOp;
using lgpsf::corrections::apply;
using lgpsf::corrections::dense_op;
using lgpsf::corrections::glr_apply;
using lgpsf::corrections::glr_pd_floor;
using lgpsf::corrections::glr_solve;
using lgpsf::corrections::make_shifted_operator;
using lgpsf::corrections::merge;
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

Eigen::MatrixXd symmetric_matrix( int n, std::mt19937& gen )
{
    const Eigen::MatrixXd G = test_helpers::randn_points(n, n, gen);
    Eigen::MatrixXd A = 0.5 * (G + G.transpose());
    for ( int i = 0; i < n; ++i )
    {
        A(i, i) += 1.0;
    }
    return A;
}

/// A built struct with a rank-4 block of MIXED-SIGN coefficients -- the
/// regime the certificates exist for.
ShiftedOperator built_example( int n, std::mt19937& gen )
{
    ShiftedOperator A = make_shifted_operator(
        dense_op(symmetric_matrix(n, gen)), ProbeArchive{},
        sparse_hr_oracle(spd_sparse(n)), 1e-4);
    Eigen::MatrixXd C_new = Eigen::MatrixXd::Zero(4, 4);
    C_new.diagonal() << -0.3, -0.05, 0.8, 2.0;
    merge(A.block, A.hr, test_helpers::randn_points(n, 4, gen), C_new, C_new,
          Provenance::PencilCache);
    return A;
}

} // namespace

TEST_CASE("construction verifies symmetry and consistency")
{
    std::mt19937 gen(0);
    const int n = 30;
    const Eigen::MatrixXd S = symmetric_matrix(n, gen);

    // a symmetric operator passes
    const ShiftedOperator A = make_shifted_operator(
        dense_op(S), ProbeArchive{}, sparse_hr_oracle(spd_sparse(n)), 1e-4);
    CHECK(A.dim() == n);
    CHECK(lgpsf::corrections::validate(A).empty());
    CHECK(!A.lambda_floor.has_value());  // no flip pass has run

    // an unsymmetrized operator is refused, loudly
    Eigen::MatrixXd P = S;
    P(2, 7) += 0.05 * S.cwiseAbs().maxCoeff();
    CHECK_THROWS_AS(
        make_shifted_operator(dense_op(P), ProbeArchive{},
                              sparse_hr_oracle(spd_sparse(n)), 1e-4),
        std::invalid_argument);

    // so are a bad shift, a dim mismatch, and an inconsistent archive
    CHECK_THROWS_AS(
        make_shifted_operator(dense_op(S), ProbeArchive{},
                              sparse_hr_oracle(spd_sparse(n)), 0.0),
        std::invalid_argument);
    CHECK_THROWS_AS(
        make_shifted_operator(dense_op(S), ProbeArchive{},
                              sparse_hr_oracle(spd_sparse(n + 1)), 1e-4),
        std::invalid_argument);
    ProbeArchive lopsided;
    lopsided.Z = Eigen::MatrixXd::Zero(n, 3);
    lopsided.Y = Eigen::MatrixXd::Zero(n, 2);
    CHECK_THROWS_AS(
        make_shifted_operator(dense_op(S), lopsided,
                              sparse_hr_oracle(spd_sparse(n)), 1e-4),
        std::invalid_argument);
}

TEST_CASE("apply is B + E + a Hr, densely")
{
    std::mt19937 gen(1);
    const int n = 35;
    const Eigen::MatrixXd S = symmetric_matrix(n, gen);
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(spd_sparse(n));

    ShiftedOperator A = make_shifted_operator(
        dense_op(S), ProbeArchive{}, sparse_hr_oracle(spd_sparse(n)), 1e-4);
    const Eigen::MatrixXd V_new = test_helpers::randn_points(n, 3, gen);
    Eigen::MatrixXd C_new = Eigen::MatrixXd::Zero(3, 3);
    C_new.diagonal() << -0.4, 0.7, 1.5;
    merge(A.block, A.hr, V_new, C_new, C_new, Provenance::Flip);

    const Eigen::MatrixXd Ed = Hrd * V_new * C_new * V_new.transpose() * Hrd;
    const Eigen::MatrixXd X = test_helpers::randn_points(n, 3, gen);
    for ( double a : {0.0, 1e-3, 0.5} )
    {
        const Eigen::MatrixXd expected = (S + Ed + a * Hrd) * X;
        CHECK((apply(A, X, a) - expected).cwiseAbs().maxCoeff()
              < 1e-10 * expected.cwiseAbs().maxCoeff());
    }
}

TEST_CASE("the Woodbury inverse is exact at every shift, no refactorization")
{
    std::mt19937 gen(2);
    const int n = 40;
    ShiftedOperator A = built_example(n, gen);
    const double floor = glr_pd_floor(A);
    CHECK(floor > 0.0);  // the mixed-sign block really has a negative side

    const Eigen::MatrixXd X = test_helpers::randn_points(n, 3, gen);
    for ( double a : {1.05 * floor, 1.5 * floor, 3.0 * floor, 20.0 * floor} )
    {
        // both compositions recover the identity...
        const Eigen::MatrixXd there = glr_solve(A, glr_apply(A, X, a), a);
        const Eigen::MatrixXd back = glr_apply(A, glr_solve(A, X, a), a);
        const double scale = X.cwiseAbs().maxCoeff();
        CHECK((there - X).cwiseAbs().maxCoeff() < 1e-8 * scale);
        CHECK((back - X).cwiseAbs().maxCoeff() < 1e-8 * scale);
    }

    // ...and the solve matches a dense factorization of M(a) outright
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(spd_sparse(n));
    const Eigen::MatrixXd Ed =
        A.block.HrV * A.block.C_surr * A.block.HrV.transpose();
    const double a = 2.0 * floor;
    const Eigen::MatrixXd dense_solution =
        (a * Hrd + Ed).ldlt().solve(Eigen::MatrixXd(X));
    CHECK((glr_solve(A, X, a) - dense_solution).cwiseAbs().maxCoeff()
          < 1e-9 * dense_solution.cwiseAbs().maxCoeff());
}

TEST_CASE("the analytic PD certificate agrees with dense truth")
{
    std::mt19937 gen(3);
    const int n = 30;
    ShiftedOperator A = built_example(n, gen);
    const double floor = glr_pd_floor(A);
    REQUIRE(floor > 0.0);

    // dense M(a) changes definiteness exactly at the certified floor
    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(spd_sparse(n));
    const Eigen::MatrixXd Ed =
        A.block.HrV * A.block.C_surr * A.block.HrV.transpose();
    const auto min_eig = [&]( double a ) {
        return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(a * Hrd + Ed)
            .eigenvalues()
            .minCoeff();
    };
    CHECK(min_eig(1.02 * floor) > 0.0);
    CHECK(min_eig(0.98 * floor) < 0.0);

    // the solve enforces it
    const Eigen::MatrixXd x = Eigen::MatrixXd::Ones(n, 1);
    CHECK_THROWS_AS(glr_solve(A, x, 0.98 * floor), std::domain_error);
    CHECK_THROWS_AS(glr_solve(A, x, 0.0), std::domain_error);
    CHECK(glr_solve(A, x, 1.02 * floor).allFinite());
}

TEST_CASE("an empty block degenerates to pure prior preconditioning")
{
    std::mt19937 gen(4);
    const int n = 25;
    const ShiftedOperator A = make_shifted_operator(
        dense_op(symmetric_matrix(n, gen)), ProbeArchive{},
        sparse_hr_oracle(spd_sparse(n)), 1e-4);
    CHECK(glr_pd_floor(A) == 0.0);

    const Eigen::MatrixXd Hrd = Eigen::MatrixXd(spd_sparse(n));
    const Eigen::MatrixXd X = test_helpers::randn_points(n, 2, gen);
    const double a = 0.7;
    const Eigen::MatrixXd expected =
        Hrd.ldlt().solve(Eigen::MatrixXd(X)) / a;
    CHECK((glr_solve(A, X, a) - expected).cwiseAbs().maxCoeff()
          < 1e-9 * expected.cwiseAbs().maxCoeff());
}

TEST_CASE("extending the cache never changes the operator")
{
    // The reason the block carries TWO coefficient matrices: caching the
    // operator's own spectral content (for the GLR surrogate) must leave the
    // represented operator B + E bitwise alone, while still moving the
    // surrogate and its PD floor.
    std::mt19937 gen(5);
    const int n = 30;
    ShiftedOperator A = make_shifted_operator(
        dense_op(symmetric_matrix(n, gen)), ProbeArchive{},
        sparse_hr_oracle(spd_sparse(n)), 1e-4);

    const Eigen::MatrixXd X = test_helpers::randn_points(n, 3, gen);
    const Eigen::MatrixXd before = apply(A, X, 0.3);
    REQUIRE(glr_pd_floor(A) == 0.0);

    // a cache-only merge: zero correction, spectral content in the surrogate
    Eigen::MatrixXd lambdas = Eigen::MatrixXd::Zero(3, 3);
    lambdas.diagonal() << -0.2, 1.0, 4.0;
    merge(A.block, A.hr, test_helpers::randn_points(n, 3, gen),
          Eigen::MatrixXd::Zero(3, 3), lambdas, Provenance::PencilCache);

    CHECK(apply(A, X, 0.3) == before);      // the operator: bitwise unmoved
    CHECK(glr_pd_floor(A) > 0.0);           // the surrogate: moved
    CHECK(lgpsf::corrections::correction_eigenvalues(A.block).cwiseAbs()
              .maxCoeff() < 1e-12);
}
