// SPDX-License-Identifier: MIT
//
// Checks on the whole-operator layer: recovery of a synthetic operator with
// M1 != M2 throughout, the always-on baseline guard, the window geometries,
// the CSR window storage, and bit-identical results across thread counts.
//
// All self-contained -- nothing is compared against a stored reference or
// against the Python prototype.

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/operator_fit.hpp"
#include "test_helpers.hpp"

using lgpsf::FixedSet;
using lgpsf::Mode;
using lgpsf::MuMode;
using lgpsf::OperatorFit;
using lgpsf::OperatorFitConfig;
using lgpsf::RowStatus;
using lgpsf::WhitenedBasis;
using lgpsf::WindowShape;
using lgpsf::fit_operator;
using lgpsf::modes_up_to_level;
using lgpsf::theta_hat_size;
using lgpsf::unpack_theta;
using lgpsf::unpack_theta_hat;

namespace {

/// A synthetic operator built row by row from known ellipsoids and known
/// coefficients, with M1 != M2 everywhere so a mistake in the mass routing
/// cannot hide.
struct Synthetic
{
    Eigen::MatrixXd x_cols;   // (K, 2)
    Eigen::VectorXd m1, m2;   // (K,)
    Eigen::MatrixXd V, HV;    // (K, k), (K, k)
    std::vector<Eigen::MatrixXd> sigma;
    std::vector<Eigen::VectorXd> theta_hat_true;
    std::vector<Eigen::VectorXd> c_true;
    Eigen::VectorXd s_true;
    std::vector<Mode> modes;
    std::vector<char> gate;
    int fitted_rows = 0;
};

Synthetic make_operator( std::mt19937& gen, int per_side = 11, int num_probes = 30,
                         int fitted_rows = 4, double aspect = 1.6,
                         double prior_scale = 2.0 )
{
    Synthetic op;
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
    op.m1 = test_helpers::uniform_points(count, 1, gen, 0.7, 1.5).col(0);
    op.m2 = test_helpers::uniform_points(count, 1, gen, 0.4, 1.1).col(0);
    op.modes = modes_up_to_level(2, 2);
    op.fitted_rows = fitted_rows;

    // fit a few interior rows; the rest are gated out
    op.gate.assign(static_cast<std::size_t>(count), 0);
    const int stride = count / (fitted_rows + 1);
    std::vector<int> chosen;
    for ( int r = 0; r < fitted_rows; ++r )
    {
        const int rho = stride * (r + 1);
        chosen.push_back(rho);
        op.gate[static_cast<std::size_t>(rho)] = 1;
    }

    op.sigma.assign(static_cast<std::size_t>(count),
                    Eigen::MatrixXd::Identity(2, 2));
    op.theta_hat_true.assign(static_cast<std::size_t>(count), Eigen::VectorXd());
    op.c_true.assign(static_cast<std::size_t>(count), Eigen::VectorXd());
    op.s_true = Eigen::VectorXd::Zero(count);

    op.V = test_helpers::randn_points(count, num_probes, gen);
    op.HV = Eigen::MatrixXd::Zero(count, num_probes);

    for ( int rho : chosen )
    {
        Eigen::VectorXd theta_hat(theta_hat_size(2, MuMode::Pinned));
        theta_hat << std::log(0.26 * aspect), std::log(0.26), 0.03;
        op.theta_hat_true[static_cast<std::size_t>(rho)] = theta_hat;

        const Eigen::VectorXd center = op.x_cols.row(rho).transpose();
        const Eigen::MatrixXd L =
            unpack_theta_hat(theta_hat, center, MuMode::Pinned).L;
        // the caller's prior is deliberately WRONG by a scale factor, so the
        // baseline is beatable and the search has something to do
        op.sigma[static_cast<std::size_t>(rho)] =
            prior_scale * prior_scale * L * L.transpose();

        const Eigen::VectorXd c =
            test_helpers::randn_points(static_cast<int>(op.modes.size()), 1, gen).col(0);
        op.c_true[static_cast<std::size_t>(rho)] = c;
        op.s_true(rho) = 0.4;

        // H[rho, j] = m1 m2_j sum_i c_i phi_i(x_j) + m1 s [j == rho], with the
        // whitened basis giving phi_hat = sqrt(m1) sqrt(m2) phi
        const WhitenedBasis basis(op.x_cols, op.m1(rho), op.m2, op.modes, center,
                                  MuMode::Pinned);
        const Eigen::VectorXd phi_hat_c = basis(theta_hat).values() * c;
        Eigen::VectorXd h_row(op.x_cols.rows());
        for ( Eigen::Index j = 0; j < op.x_cols.rows(); ++j )
        {
            h_row(j) = std::sqrt(op.m1(rho)) * std::sqrt(op.m2(j)) * phi_hat_c(j);
        }
        h_row(rho) += op.m1(rho) * op.s_true(rho);
        op.HV.row(rho) = h_row.transpose() * op.V;
    }
    return op;
}

OperatorFitConfig config_for( const Synthetic& op )
{
    OperatorFitConfig config;
    config.tau_window = 10.0;
    config.spike = true;
    config.row.mode_policy = std::make_shared<FixedSet>(op.modes, "truth");
    config.row.target_score = std::nullopt;
    config.row.num_rungs = 3;
    return config;
}

/// Bit-equality that treats NaN as equal to NaN.
///
/// Plain `operator==` will not do: rows with no shipped model are NaN-padded
/// by design, and NaN != NaN, so an all-equal comparison reports a difference
/// for arrays that are in fact identical.
bool same( const Eigen::MatrixXd& a, const Eigen::MatrixXd& b )
{
    if ( a.rows() != b.rows() || a.cols() != b.cols() )
    {
        return false;
    }
    for ( Eigen::Index i = 0; i < a.rows(); ++i )
    {
        for ( Eigen::Index j = 0; j < a.cols(); ++j )
        {
            const bool a_nan = std::isnan(a(i, j));
            if ( a_nan != std::isnan(b(i, j)) )
            {
                return false;
            }
            if ( !a_nan && a(i, j) != b(i, j) )
            {
                return false;
            }
        }
    }
    return true;
}

OperatorFit run( const Synthetic& op, OperatorFitConfig config )
{
    return fit_operator(op.x_cols, op.m1, op.m2, op.V, op.HV, op.sigma, config,
                        std::nullopt, std::nullopt, op.gate);
}

} // namespace

TEST_CASE("a synthetic operator is recovered row by row")
{
    std::mt19937 gen(0);
    const Synthetic op = make_operator(gen);
    const OperatorFit fit = run(op, config_for(op));

    int shipped = 0;
    double worst_c = 0.0, worst_s = 0.0;
    for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
    {
        if ( !op.gate[static_cast<std::size_t>(rho)] )
        {
            CHECK(fit.status[static_cast<std::size_t>(rho)] == RowStatus::GatedOut);
            CHECK(fit.mode_set_id[static_cast<std::size_t>(rho)] == -1);
            continue;
        }
        REQUIRE(fit.status[static_cast<std::size_t>(rho)] != RowStatus::Failed);
        ++shipped;

        // the ellipsoid the data was built from
        const Eigen::VectorXd center = op.x_cols.row(rho).transpose();
        const Eigen::MatrixXd truth =
            unpack_theta_hat(op.theta_hat_true[static_cast<std::size_t>(rho)],
                             center, MuMode::Pinned).L;
        Eigen::MatrixXd recovered(2, 2);
        for ( int i = 0; i < 2; ++i )
        {
            for ( int j = 0; j < 2; ++j )
            {
                recovered(i, j) = fit.L(rho, i * 2 + j);
            }
        }
        CHECK((recovered * recovered.transpose() - truth * truth.transpose())
                  .cwiseAbs().maxCoeff() < 1e-5);
        CHECK((fit.mu.row(rho).transpose() - center).cwiseAbs().maxCoeff() < 1e-9);

        const Eigen::VectorXd& c_true = op.c_true[static_cast<std::size_t>(rho)];
        for ( Eigen::Index i = 0; i < c_true.size(); ++i )
        {
            worst_c = std::max(worst_c, std::abs(fit.c(rho, i) - c_true(i)));
        }
        worst_s = std::max(worst_s, std::abs(fit.s(rho) - op.s_true(rho)));
    }
    MESSAGE("recovered " << shipped << " rows; worst |dc| " << worst_c
                         << ", worst |ds| " << worst_s);
    CHECK(shipped == op.fitted_rows);
    CHECK(worst_c < 1e-4);
    CHECK(worst_s < 1e-4);
}

TEST_CASE("results are bit-identical across thread counts")
{
    // The M4 acceptance criterion, and the payoff of hoisting randomness: rows
    // write disjoint slots and every shared registry is built serially in row
    // order, so scheduling cannot reach the answer.
    std::mt19937 gen(1);
    const Synthetic op = make_operator(gen, 11, 30, 6);

    OperatorFitConfig serial = config_for(op);
    serial.num_threads = 1;
    OperatorFitConfig parallel = config_for(op);
    parallel.num_threads = 4;

    const OperatorFit a = run(op, serial);
    const OperatorFit b = run(op, parallel);

    CHECK(same(a.theta, b.theta));
    CHECK(same(a.mu, b.mu));
    CHECK(same(a.L, b.L));
    CHECK(same(a.c, b.c));
    CHECK(same(a.s, b.s));
    CHECK(same(a.score, b.score));
    CHECK(same(a.baseline_score, b.baseline_score));
    CHECK(a.mode_set_id == b.mode_set_id);
    CHECK(a.window_indptr == b.window_indptr);
    CHECK(a.window_indices == b.window_indices);
    CHECK(a.released == b.released);
    REQUIRE(a.mode_sets.size() == b.mode_sets.size());
    for ( std::size_t i = 0; i < a.mode_sets.size(); ++i )
    {
        CHECK(a.mode_sets[i] == b.mode_sets[i]);
    }
    for ( std::size_t i = 0; i < a.status.size(); ++i )
    {
        CHECK(a.status[i] == b.status[i]);
        CHECK(a.stop_reason[i] == b.stop_reason[i]);
    }
}

TEST_CASE("the baseline guard means a shipped row is never worse than the prior")
{
    // The whole point of the guard: whatever happens in the search, the row
    // that ships is at least as good as the a-priori-Gaussian status quo.
    std::mt19937 gen(2);
    const Synthetic op = make_operator(gen);
    const OperatorFit fit = run(op, config_for(op));

    int searched = 0, fell_back = 0;
    for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
    {
        const RowStatus status = fit.status[static_cast<std::size_t>(rho)];
        if ( status == RowStatus::Fit )
        {
            ++searched;
            // shipped only on a STRICT improvement
            CHECK(fit.score(rho) < fit.baseline_score(rho));
        }
        else if ( status == RowStatus::FallbackBaseline )
        {
            ++fell_back;
            CHECK(fit.score(rho) == fit.baseline_score(rho));
        }
    }
    MESSAGE("guard: " << searched << " searched fits shipped, " << fell_back
                      << " fell back to the baseline");
    CHECK(searched + fell_back == op.fitted_rows);
    // the prior here is deliberately wrong by 2x, so the search should win
    CHECK(searched > 0);
}

TEST_CASE("the ellipsoid window is the default and the ball is its bounding sphere")
{
    // The two differ only in anisotropy: same tau, same scale, so the ball
    // contains the ellipsoid and the point counts differ by the axis ratio.
    std::mt19937 gen(3);
    const Synthetic op = make_operator(gen, 11, 30, 3, 3.0);

    OperatorFitConfig ellipsoid = config_for(op);
    CHECK(ellipsoid.window == WindowShape::Ellipsoid);  // the design intent
    // tau must be small enough that the windows are smaller than the mesh,
    // or both shapes swallow everything and the comparison says nothing
    ellipsoid.tau_window = 2.0;
    OperatorFitConfig ball = ellipsoid;
    ball.window = WindowShape::Ball;

    const OperatorFit with_ellipsoid = run(op, ellipsoid);
    const OperatorFit with_ball = run(op, ball);

    std::size_t total_ellipsoid = 0, total_ball = 0;
    for ( Eigen::Index rho = 0; rho < with_ellipsoid.num_rows(); ++rho )
    {
        if ( !op.gate[static_cast<std::size_t>(rho)] )
        {
            continue;
        }
        const std::vector<int> a = with_ellipsoid.row_window(static_cast<int>(rho));
        const std::vector<int> b = with_ball.row_window(static_cast<int>(rho));
        total_ellipsoid += a.size();
        total_ball += b.size();

        // the ball is the circumscribed sphere, so it contains the ellipsoid
        for ( int column : a )
        {
            CHECK(std::find(b.begin(), b.end(), column) != b.end());
        }
        // both contain the row's own dof, so the spike has somewhere to live
        CHECK(std::find(a.begin(), a.end(), static_cast<int>(rho)) != a.end());
    }
    MESSAGE("window points: ellipsoid " << total_ellipsoid << ", ball " << total_ball
                                        << " (axis ratio 3)");
    CHECK(total_ellipsoid < total_ball);

    // both recover the operator; this test is about geometry, not accuracy
    for ( Eigen::Index rho = 0; rho < with_ellipsoid.num_rows(); ++rho )
    {
        if ( op.gate[static_cast<std::size_t>(rho)] )
        {
            CHECK(with_ellipsoid.status[static_cast<std::size_t>(rho)]
                  != RowStatus::Failed);
            CHECK(with_ball.status[static_cast<std::size_t>(rho)] != RowStatus::Failed);
        }
    }
}

TEST_CASE("the stored parameters and windows decode on their own")
{
    std::mt19937 gen(4);
    const Synthetic op = make_operator(gen);
    const OperatorFit fit = run(op, config_for(op));

    // CSR consistency
    REQUIRE(fit.window_indptr.size() == static_cast<std::size_t>(fit.num_rows()) + 1);
    CHECK(fit.window_indptr.front() == 0);
    CHECK(fit.window_indptr.back() == static_cast<int>(fit.window_indices.size()));
    for ( std::size_t i = 1; i < fit.window_indptr.size(); ++i )
    {
        CHECK(fit.window_indptr[i] >= fit.window_indptr[i - 1]);
    }

    for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
    {
        const std::vector<int> window = fit.row_window(static_cast<int>(rho));
        CHECK(std::is_sorted(window.begin(), window.end()));
        for ( int column : window )
        {
            CHECK(column >= 0);
            CHECK(column < fit.num_cols());
        }
        if ( !op.gate[static_cast<std::size_t>(rho)] )
        {
            CHECK(window.empty());
            CHECK_THROWS_AS(fit.row_modes(static_cast<int>(rho)), std::invalid_argument);
            continue;
        }

        // theta is the public absolute encoding: no mu0, no mode needed
        const Eigen::VectorXd theta = fit.theta.row(rho).transpose();
        const lgpsf::EllipsoidFrame frame = unpack_theta(theta);
        CHECK((frame.mu - fit.mu.row(rho).transpose()).cwiseAbs().maxCoeff() < 1e-12);
        for ( int i = 0; i < 2; ++i )
        {
            for ( int j = 0; j < 2; ++j )
            {
                CHECK(frame.L(i, j) == doctest::Approx(fit.L(rho, i * 2 + j)));
            }
        }
        CHECK(fit.row_modes(static_cast<int>(rho)).size() == op.modes.size());
    }
}

TEST_CASE("a row that cannot be windowed fails alone, without taking the fit down")
{
    std::mt19937 gen(5);
    Synthetic op = make_operator(gen);
    // one row's prior collapses to a point, so its window cannot be formed
    int victim = -1;
    for ( Eigen::Index rho = 0; rho < static_cast<Eigen::Index>(op.gate.size()); ++rho )
    {
        if ( op.gate[static_cast<std::size_t>(rho)] )
        {
            victim = static_cast<int>(rho);
            break;
        }
    }
    REQUIRE(victim >= 0);
    op.sigma[static_cast<std::size_t>(victim)] =
        1e-12 * Eigen::MatrixXd::Identity(2, 2);

    const OperatorFit fit = run(op, config_for(op));

    CHECK(fit.status[static_cast<std::size_t>(victim)] == RowStatus::Failed);
    CHECK(fit.failures.count(victim) == 1u);
    MESSAGE("failed row reported: " << fit.failures.at(victim));

    int survivors = 0;
    for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
    {
        if ( op.gate[static_cast<std::size_t>(rho)] && rho != victim )
        {
            CHECK(fit.status[static_cast<std::size_t>(rho)] != RowStatus::Failed);
            ++survivors;
        }
    }
    CHECK(survivors == op.fitted_rows - 1);
}

TEST_CASE("explicit windows override the derived ones")
{
    std::mt19937 gen(6);
    const Synthetic op = make_operator(gen);

    std::vector<std::vector<int>> windows(static_cast<std::size_t>(op.gate.size()));
    int overridden = -1;
    for ( Eigen::Index rho = 0; rho < static_cast<Eigen::Index>(op.gate.size()); ++rho )
    {
        if ( op.gate[static_cast<std::size_t>(rho)] )
        {
            overridden = static_cast<int>(rho);
            break;
        }
    }
    REQUIRE(overridden >= 0);
    // a deliberately odd window: the row's own dof plus a contiguous block
    std::vector<int> chosen{overridden};
    for ( int j = 0; j < 40; ++j )
    {
        if ( j != overridden )
        {
            chosen.push_back(j);
        }
    }
    std::sort(chosen.begin(), chosen.end());
    windows[static_cast<std::size_t>(overridden)] = chosen;

    const OperatorFit fit =
        fit_operator(op.x_cols, op.m1, op.m2, op.V, op.HV, op.sigma, config_for(op),
                     std::nullopt, std::nullopt, op.gate, windows);

    CHECK(fit.row_window(overridden) == chosen);
}

TEST_CASE("malformed inputs are rejected eagerly")
{
    std::mt19937 gen(7);
    const Synthetic op = make_operator(gen, 6, 20, 2);
    const OperatorFitConfig config = config_for(op);

    CHECK_THROWS_AS(fit_operator(op.x_cols, op.m1, op.m2, op.V,
                                 Eigen::MatrixXd::Zero(3, 3), op.sigma, config),
                    std::invalid_argument);
    CHECK_THROWS_AS(fit_operator(op.x_cols, op.m1, op.m2, op.V, op.HV,
                                 std::vector<Eigen::MatrixXd>(2), config),
                    std::invalid_argument);

    OperatorFitConfig no_policy = config;
    no_policy.row.mode_policy.reset();
    CHECK_THROWS_AS(fit_operator(op.x_cols, op.m1, op.m2, op.V, op.HV, op.sigma,
                                 no_policy),
                    std::invalid_argument);

    // the spike needs the square dof context
    CHECK_THROWS_AS(fit_operator(op.x_cols, op.m1, op.m2, op.V, op.HV, op.sigma,
                                 config, std::nullopt,
                                 Eigen::MatrixXd(op.x_cols)),
                    std::invalid_argument);
}
