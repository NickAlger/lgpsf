// SPDX-License-Identifier: MIT
//
// The operator boundary of the corrections layer: SymmetricOp, HrOracle,
// their adapters, and the symmetry measurement. Everything here is about the
// boundary's contracts -- shapes, ownership, symmetry, solve accuracy -- with
// no fit anywhere in sight, which is the point: the layer is operator-blind.

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/corrections/hr_oracle.hpp"
#include "lgpsf/corrections/symmetric_op.hpp"
#include "test_helpers.hpp"

using lgpsf::corrections::HrOracle;
using lgpsf::corrections::SymmetricOp;
using lgpsf::corrections::dense_op;
using lgpsf::corrections::sparse_hr_oracle;
using lgpsf::corrections::sparse_op;
using lgpsf::corrections::symmetry_defect;

namespace {

/// A dense symmetric test matrix with a decaying spectrum -- vaguely the shape
/// of a data-misfit operator, and dense enough that sparse and dense wrappings
/// of it exercise different code paths.
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

/// A sparse SPD matrix: a 1-D Laplacian plus a positive diagonal.
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

} // namespace

TEST_CASE("sparse and dense wrappings are the same operator")
{
    std::mt19937 gen(0);
    const Eigen::MatrixXd A = symmetric_matrix(40, gen);
    const SymmetricOp dense = dense_op(A);
    const SymmetricOp sparse = sparse_op(A.sparseView());

    CHECK(dense.dim() == 40);
    CHECK(sparse.dim() == 40);

    const Eigen::MatrixXd X = test_helpers::randn_points(40, 5, gen);
    const Eigen::MatrixXd from_dense = dense.apply(X);
    const Eigen::MatrixXd from_sparse = sparse.apply(X);
    CHECK((from_dense - A * X).cwiseAbs().maxCoeff() == 0.0);
    CHECK((from_sparse - from_dense).cwiseAbs().maxCoeff()
          < 1e-13 * from_dense.cwiseAbs().maxCoeff());
}

TEST_CASE("the handle owns its operator")
{
    // The wrapped matrix goes out of scope; the handle must keep working.
    std::mt19937 gen(1);
    SymmetricOp op = dense_op(Eigen::MatrixXd::Identity(3, 3));
    {
        const Eigen::MatrixXd doomed = symmetric_matrix(7, gen);
        op = dense_op(doomed);
    }
    const Eigen::MatrixXd X = test_helpers::randn_points(7, 2, gen);
    CHECK(op.apply(X).rows() == 7);
}

TEST_CASE("shape contracts are enforced on both sides of the boundary")
{
    const SymmetricOp op = dense_op(Eigen::MatrixXd::Identity(5, 5));

    // caller-side: wrong block height (a row-convention block, say)
    CHECK_THROWS_AS(op.apply(Eigen::MatrixXd::Zero(3, 5)),
                    std::invalid_argument);

    // callable-side: a wrapped function that breaks its own shape promise
    const SymmetricOp broken(
        4, []( const Eigen::Ref<const Eigen::MatrixXd>& )
        { return Eigen::MatrixXd::Zero(2, 2); });
    CHECK_THROWS_AS(broken.apply(Eigen::MatrixXd::Zero(4, 1)),
                    std::runtime_error);

    // construction-time misuse
    CHECK_THROWS_AS(SymmetricOp(0, nullptr), std::invalid_argument);
    CHECK_THROWS_AS(dense_op(Eigen::MatrixXd::Zero(3, 4)),
                    std::invalid_argument);
    CHECK_THROWS_AS(sparse_op(Eigen::SparseMatrix<double>(3, 4)),
                    std::invalid_argument);
}

TEST_CASE("symmetry_defect separates symmetric from nonsymmetric")
{
    std::mt19937 gen(2);
    const Eigen::MatrixXd S = symmetric_matrix(60, gen);

    // symmetric: rounding level
    const double clean = symmetry_defect(dense_op(S));
    CHECK(clean < 1e-13);

    // a few-percent asymmetric perturbation: measured at about its true size,
    // orders of magnitude above rounding -- no delicate threshold needed
    Eigen::MatrixXd P = S;
    const Eigen::MatrixXd noise = test_helpers::randn_points(60, 60, gen);
    P += 0.05 * S.cwiseAbs().maxCoeff() / noise.cwiseAbs().maxCoeff() * noise;
    const double dirty = symmetry_defect(dense_op(P));
    CHECK(dirty > 1e-4);
    CHECK(dirty > 1e6 * clean);

    // deterministic in the seed
    CHECK(symmetry_defect(dense_op(P), 8, 7) == symmetry_defect(dense_op(P), 8, 7));

    CHECK_THROWS_AS(symmetry_defect(dense_op(S), 0), std::invalid_argument);
}

TEST_CASE("the sparse H_r oracle applies, solves, and refuses indefiniteness")
{
    std::mt19937 gen(3);
    const int n = 50;
    const Eigen::SparseMatrix<double> Hr = spd_sparse(n);
    const HrOracle oracle = sparse_hr_oracle(Hr);
    CHECK(oracle.dim() == n);

    const Eigen::MatrixXd X = test_helpers::randn_points(n, 4, gen);

    // apply is the matrix product
    CHECK((oracle.apply(X) - Hr * X).cwiseAbs().maxCoeff() == 0.0);

    // solve(apply(X)) recovers X to direct-solve accuracy, any tol
    const Eigen::MatrixXd roundtrip = oracle.solve(oracle.apply(X), 1e-2);
    CHECK((roundtrip - X).cwiseAbs().maxCoeff()
          < 1e-10 * X.cwiseAbs().maxCoeff());

    // contracts
    CHECK_THROWS_AS(oracle.solve(X, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(oracle.apply(Eigen::MatrixXd::Zero(n + 1, 1)),
                    std::invalid_argument);

    // an indefinite matrix is refused at construction, not discovered later
    Eigen::SparseMatrix<double> indefinite = spd_sparse(n);
    indefinite.coeffRef(4, 4) = -50.0;
    CHECK_THROWS_AS(sparse_hr_oracle(indefinite), std::invalid_argument);
}

TEST_CASE("an oracle built from callables honors the same contracts")
{
    // The production shape: apply and solve are the consumer's own machinery.
    // Model it with a diagonal operator whose solve is exact.
    const int n = 12;
    Eigen::VectorXd d(n);
    for ( int i = 0; i < n; ++i )
    {
        d(i) = 1.0 + i;
    }
    const HrOracle oracle(
        n,
        [d]( const Eigen::Ref<const Eigen::MatrixXd>& X ) -> Eigen::MatrixXd
        { return d.asDiagonal() * X; },
        [d]( const Eigen::Ref<const Eigen::MatrixXd>& B, double )
            -> Eigen::MatrixXd { return d.cwiseInverse().asDiagonal() * B; });

    std::mt19937 gen(4);
    const Eigen::MatrixXd X = test_helpers::randn_points(n, 3, gen);
    CHECK((oracle.solve(oracle.apply(X), 1e-8) - X).cwiseAbs().maxCoeff()
          < 1e-14);

    CHECK_THROWS_AS(HrOracle(n, nullptr, nullptr), std::invalid_argument);
}
