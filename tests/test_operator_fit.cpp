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
#include <limits>
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
using lgpsf::capped_covariance;
using lgpsf::assemble_sparse;
using lgpsf::ellipsoid_field;
using lgpsf::eval_entries;
using lgpsf::eval_kernel;
using lgpsf::fit_operator;
using lgpsf::matvec;
using lgpsf::model_rows;
using lgpsf::qc_map;
using lgpsf::spike_measure;
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

TEST_CASE("the aspect cap spans the ball and the ellipsoid continuously")
{
    // The cap floors sigma's eigenvalues at lambda_max / cap^2, so cap = 1 is
    // isotropic, cap = infinity is the caller's ellipsoid untouched, and the
    // resulting axis ratio is min(the input's, cap). Orientation never moves.
    Eigen::MatrixXd sigma(2, 2);
    const double angle = 0.4;
    Eigen::Matrix2d rotation;
    rotation << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
    Eigen::Matrix2d axes = Eigen::Matrix2d::Zero();
    axes(0, 0) = 64.0;  // semi-axes 8 and 1: an 8:1 prior
    axes(1, 1) = 1.0;
    sigma = rotation * axes * rotation.transpose();

    const auto aspect_of = []( const Eigen::MatrixXd& a ) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(a);
        return std::sqrt(solver.eigenvalues()(1) / solver.eigenvalues()(0));
    };
    CHECK(aspect_of(sigma) == doctest::Approx(8.0));

    // the endpoints
    const Eigen::MatrixXd ball = capped_covariance(sigma, 1.0);
    CHECK(aspect_of(ball) == doctest::Approx(1.0));
    CHECK((ball - 64.0 * Eigen::MatrixXd::Identity(2, 2)).cwiseAbs().maxCoeff() < 1e-9);
    CHECK((capped_covariance(sigma, std::numeric_limits<double>::infinity()) - sigma)
              .cwiseAbs().maxCoeff() == 0.0);

    // everything between, and the cap is never exceeded
    for ( double cap : {1.0, 1.5, 2.0, 4.0, 8.0, 16.0} )
    {
        const Eigen::MatrixXd capped = capped_covariance(sigma, cap);
        CHECK(aspect_of(capped) == doctest::Approx(std::min(cap, 8.0)));
        // the largest axis is untouched: the cap moves shape, never scale
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(capped);
        CHECK(solver.eigenvalues()(1) == doctest::Approx(64.0));
        // and the orientation survives -- asked only where it is defined,
        // since cap = 1 is isotropic and an isotropic covariance has no
        // orientation to preserve (any orthonormal basis diagonalizes it)
        if ( cap > 1.0 )
        {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> original(sigma);
            CHECK(std::abs(std::abs(solver.eigenvectors().col(1).dot(
                      original.eigenvectors().col(1))) - 1.0) < 1e-9);
        }
    }

    CHECK_THROWS_AS(capped_covariance(sigma, 0.5), std::invalid_argument);
}

TEST_CASE("tightening the cap nests the windows and shrinks them")
{
    // cap = 1 is the bounding sphere, so it contains every tighter window; the
    // point count falls monotonically as the cap is raised toward the prior's
    // own shape.
    std::mt19937 gen(3);
    const Synthetic op = make_operator(gen, 11, 30, 3, 3.0);

    std::vector<double> caps{1.0, 1.5, 3.0, std::numeric_limits<double>::infinity()};
    std::vector<OperatorFit> fits;
    std::vector<std::size_t> totals;
    for ( double cap : caps )
    {
        OperatorFitConfig config = config_for(op);
        // tau must be small enough that windows are smaller than the mesh, or
        // every setting swallows the grid and the comparison says nothing
        config.tau_window = 1.5;
        config.window_aspect_cap = cap;
        fits.push_back(run(op, config));
        std::size_t total = 0;
        for ( Eigen::Index rho = 0; rho < fits.back().num_rows(); ++rho )
        {
            total += fits.back().row_window(static_cast<int>(rho)).size();
        }
        totals.push_back(total);
    }
    MESSAGE("window points by cap 1 / 1.5 / 3 / inf: " << totals[0] << " / "
            << totals[1] << " / " << totals[2] << " / " << totals[3]);

    // the default really is the uncapped ellipsoid
    CHECK(config_for(op).window_aspect_cap
          == std::numeric_limits<double>::infinity());
    // monotone non-increasing, and strictly smaller at the ends
    for ( std::size_t i = 1; i < totals.size(); ++i )
    {
        CHECK(totals[i] <= totals[i - 1]);
    }
    CHECK(totals.back() < totals.front());
    // the prior's aspect is 3, so capping AT 3 already gives the full ellipsoid
    CHECK(totals[2] == totals[3]);

    // nesting, row by row: a looser cap contains a tighter one
    for ( Eigen::Index rho = 0; rho < fits[0].num_rows(); ++rho )
    {
        if ( !op.gate[static_cast<std::size_t>(rho)] )
        {
            continue;
        }
        const std::vector<int> widest = fits[0].row_window(static_cast<int>(rho));
        for ( std::size_t i = 1; i < fits.size(); ++i )
        {
            for ( int column : fits[i].row_window(static_cast<int>(rho)) )
            {
                CHECK(std::find(widest.begin(), widest.end(), column) != widest.end());
            }
        }
        // every window keeps the row's own dof, so the spike has a home
        CHECK(std::find(widest.begin(), widest.end(), static_cast<int>(rho))
              != widest.end());
    }

    for ( const OperatorFit& fit : fits )
    {
        for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
        {
            if ( op.gate[static_cast<std::size_t>(rho)] )
            {
                CHECK(fit.status[static_cast<std::size_t>(rho)] != RowStatus::Failed);
            }
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

TEST_CASE("eval_kernel is the raw component; eval_entries is the deployed operator")
{
    // The distinction the helper table is built on. eval_kernel is the smooth
    // kernel at arbitrary points, unrestricted -- component access. Everything
    // in the dof context is window-restricted, because the deployed object has
    // to be the object that was scored.
    std::mt19937 gen(20);
    const Synthetic op = make_operator(gen, 11, 30, 3);
    OperatorFitConfig config = config_for(op);
    config.tau_window = 1.2;  // a window strictly inside the mesh
    const OperatorFit fit = run(op, config);

    const int rho = model_rows(fit).front();
    const std::vector<int> window = fit.row_window(rho);
    REQUIRE(!window.empty());
    REQUIRE(static_cast<Eigen::Index>(window.size()) < fit.num_cols());

    // find a column outside the window
    int outside = -1;
    for ( Eigen::Index j = 0; j < fit.num_cols() && outside < 0; ++j )
    {
        if ( !std::binary_search(window.begin(), window.end(), static_cast<int>(j)) )
        {
            outside = static_cast<int>(j);
        }
    }
    REQUIRE(outside >= 0);

    // the raw kernel is alive out there...
    const Eigen::MatrixXd probe = fit.x_cols.row(outside);
    CHECK(std::abs(eval_kernel(fit, {rho}, probe)(0, 0)) > 0.0);
    // ...and the deployed operator is exactly zero
    CHECK(eval_entries(fit, {rho}, {outside})(0) == 0.0);

    // inside the window it is m1 m2 times the kernel, plus the spike on the
    // diagonal
    const int inside = window.front();
    const double kernel =
        eval_kernel(fit, {rho}, Eigen::MatrixXd(fit.x_cols.row(inside)))(0, 0);
    double expected = fit.m1_diag(rho) * fit.m2_diag(inside) * kernel;
    if ( inside == rho )
    {
        expected += fit.m1_diag(rho) * fit.s(rho);
    }
    CHECK(eval_entries(fit, {rho}, {inside})(0) == doctest::Approx(expected));

    // the spike really is ADDITIVE on top of the unmodified smooth diagonal
    const double diagonal = eval_entries(fit, {rho}, {rho})(0);
    const double smooth_diagonal =
        fit.m1_diag(rho) * fit.m2_diag(rho)
        * eval_kernel(fit, {rho}, Eigen::MatrixXd(fit.x_cols.row(rho)))(0, 0);
    CHECK(diagonal - smooth_diagonal
          == doctest::Approx(fit.m1_diag(rho) * fit.s(rho)));

    // a row with no model contributes nothing
    int ungated = -1;
    for ( Eigen::Index r = 0; r < fit.num_rows() && ungated < 0; ++r )
    {
        if ( !op.gate[static_cast<std::size_t>(r)] )
        {
            ungated = static_cast<int>(r);
        }
    }
    REQUIRE(ungated >= 0);
    CHECK(eval_entries(fit, {ungated}, {ungated})(0) == 0.0);
    CHECK_THROWS_AS(eval_kernel(fit, {ungated}, fit.x_cols), std::invalid_argument);
}

TEST_CASE("matvec agrees with a dense assembly of the deployed operator")
{
    // The matrix-free path and the entry-by-entry definition must be the same
    // operator; matvec touches only the windows, so this also pins the
    // deployed-support invariant end to end.
    std::mt19937 gen(21);
    const Synthetic op = make_operator(gen, 11, 30, 3);
    OperatorFitConfig config = config_for(op);
    config.tau_window = 1.2;
    const OperatorFit fit = run(op, config);

    Eigen::MatrixXd dense =
        Eigen::MatrixXd::Zero(fit.num_rows(), fit.num_cols());
    for ( int rho : model_rows(fit) )
    {
        std::vector<int> rows_index, cols_index;
        for ( Eigen::Index j = 0; j < fit.num_cols(); ++j )
        {
            rows_index.push_back(rho);
            cols_index.push_back(static_cast<int>(j));
        }
        dense.row(rho) = eval_entries(fit, rows_index, cols_index).transpose();
    }

    const Eigen::MatrixXd v = test_helpers::randn_points(
        static_cast<int>(fit.num_cols()), 3, gen);
    const Eigen::MatrixXd applied = matvec(fit, v);
    CHECK((applied - dense * v).cwiseAbs().maxCoeff() < 1e-10);

    // rows with no model are exactly zero rows
    for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
    {
        if ( !op.gate[static_cast<std::size_t>(rho)] )
        {
            CHECK(applied.row(rho).cwiseAbs().maxCoeff() == 0.0);
        }
    }
    CHECK_THROWS_AS(matvec(fit, Eigen::MatrixXd::Zero(3, 1)), std::invalid_argument);
}

TEST_CASE("assemble_sparse decompresses the same operator matvec applies")
{
    std::mt19937 gen(22);
    const Synthetic op = make_operator(gen, 11, 30, 3);
    OperatorFitConfig config = config_for(op);
    config.tau_window = 1.2;
    const OperatorFit fit = run(op, config);

    // A generous tau: the kernel's tau-support then covers each window, so the
    // pattern IS the window and the assembly must reproduce matvec exactly.
    const Eigen::SparseMatrix<double> wide = assemble_sparse(fit, 40.0);
    CHECK(wide.isCompressed());
    CHECK(wide.rows() == fit.num_rows());
    CHECK(wide.cols() == fit.num_cols());

    const Eigen::MatrixXd v = test_helpers::randn_points(
        static_cast<int>(fit.num_cols()), 2, gen);
    CHECK((wide * v - matvec(fit, v)).cwiseAbs().maxCoeff() < 1e-10);

    // Every stored entry equals the deployed definition, and every one lies in
    // the row's fit window -- deployed support == fit support.
    for ( int k = 0; k < wide.outerSize(); ++k )
    {
        for ( Eigen::SparseMatrix<double>::InnerIterator it(wide, k); it; ++it )
        {
            const int rho = static_cast<int>(it.row());
            const int column = static_cast<int>(it.col());
            const std::vector<int> window = fit.row_window(rho);
            CHECK(std::binary_search(window.begin(), window.end(), column));
            CHECK(it.value()
                  == doctest::Approx(eval_entries(fit, {rho}, {column})(0)));
        }
    }

    // A tight tau trims the Gaussian tail INSIDE the window, so the pattern
    // shrinks but never leaves it.
    const Eigen::SparseMatrix<double> tight = assemble_sparse(fit, 1.0);
    CHECK(tight.nonZeros() < wide.nonZeros());
    for ( int k = 0; k < tight.outerSize(); ++k )
    {
        for ( Eigen::SparseMatrix<double>::InnerIterator it(tight, k); it; ++it )
        {
            const std::vector<int> window = fit.row_window(static_cast<int>(it.row()));
            CHECK(std::binary_search(window.begin(), window.end(),
                                     static_cast<int>(it.col())));
        }
    }
    MESSAGE("assemble_sparse nonzeros: tau=1 " << tight.nonZeros() << ", tau=40 "
                                               << wide.nonZeros() << " of "
                                               << fit.num_rows() * fit.num_cols());
    CHECK_THROWS_AS(assemble_sparse(fit, 0.0), std::invalid_argument);
}

TEST_CASE("symmetrization is an assembly policy applied after the fact")
{
    std::mt19937 gen(23);
    const Synthetic op = make_operator(gen, 11, 30, 3);
    OperatorFitConfig config = config_for(op);
    config.tau_window = 1.2;
    const OperatorFit fit = run(op, config);

    const Eigen::SparseMatrix<double> plain = assemble_sparse(fit, 40.0);
    const Eigen::SparseMatrix<double> averaged =
        assemble_sparse(fit, 40.0, lgpsf::Symmetrize::Average);
    CHECK(averaged.isCompressed());

    // row fits do not produce a symmetric operator...
    const Eigen::MatrixXd plain_dense = Eigen::MatrixXd(plain);
    CHECK((plain_dense - plain_dense.transpose()).cwiseAbs().maxCoeff() > 1e-8);
    // ...and averaging is exactly (A + A^T)/2
    const Eigen::MatrixXd averaged_dense = Eigen::MatrixXd(averaged);
    CHECK((averaged_dense - averaged_dense.transpose()).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((averaged_dense - 0.5 * (plain_dense + plain_dense.transpose()))
              .cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("the ellipsoid field, the quality map and the spike measure")
{
    std::mt19937 gen(24);
    const Synthetic op = make_operator(gen, 11, 30, 3);
    const OperatorFit fit = run(op, config_for(op));

    // Sigma = L L^T, row by row
    const lgpsf::EllipsoidField field = ellipsoid_field(fit);
    CHECK(field.mu.rows() == fit.num_rows());
    for ( int rho : model_rows(fit) )
    {
        Eigen::MatrixXd L(2, 2);
        for ( int i = 0; i < 2; ++i )
        {
            for ( int j = 0; j < 2; ++j )
            {
                L(i, j) = fit.L(rho, i * 2 + j);
            }
        }
        CHECK((field.sigma[static_cast<std::size_t>(rho)] - L * L.transpose())
                  .cwiseAbs().maxCoeff() < 1e-12);
        CHECK((field.mu.row(rho) - fit.mu.row(rho)).cwiseAbs().maxCoeff() == 0.0);
    }

    // the quality map is the relative residual against held-out probes
    std::mt19937 held(99);
    const Eigen::MatrixXd V_qc =
        test_helpers::randn_points(static_cast<int>(fit.num_cols()), 8, held);
    Eigen::MatrixXd HV_qc = Eigen::MatrixXd::Zero(fit.num_rows(), 8);
    for ( int rho : model_rows(fit) )
    {
        // the true operator row, from the synthetic construction
        const Eigen::VectorXd center = op.x_cols.row(rho).transpose();
        const WhitenedBasis basis(op.x_cols, op.m1(rho), op.m2, op.modes, center,
                                  MuMode::Pinned);
        const Eigen::VectorXd phi =
            basis(op.theta_hat_true[static_cast<std::size_t>(rho)]).values()
            * op.c_true[static_cast<std::size_t>(rho)];
        Eigen::VectorXd h_row(op.x_cols.rows());
        for ( Eigen::Index j = 0; j < op.x_cols.rows(); ++j )
        {
            h_row(j) = std::sqrt(op.m1(rho)) * std::sqrt(op.m2(j)) * phi(j);
        }
        h_row(rho) += op.m1(rho) * op.s_true(rho);
        HV_qc.row(rho) = h_row.transpose() * V_qc;
    }

    const Eigen::VectorXd quality = qc_map(fit, V_qc, HV_qc);
    double worst = 0.0;
    for ( Eigen::Index rho = 0; rho < fit.num_rows(); ++rho )
    {
        if ( op.gate[static_cast<std::size_t>(rho)] )
        {
            CHECK(std::isfinite(quality(rho)));
            worst = std::max(worst, quality(rho));
        }
        else
        {
            CHECK(std::isnan(quality(rho)));  // no model, no score
        }
    }
    MESSAGE("qc_map worst relative residual on held-out probes: " << worst);
    CHECK(worst < 1e-3);

    // and it agrees with computing the same thing by hand
    const Eigen::MatrixXd predicted = matvec(fit, V_qc);
    for ( int rho : model_rows(fit) )
    {
        CHECK(quality(rho)
              == doctest::Approx((predicted.row(rho) - HV_qc.row(rho)).norm()
                                 / HV_qc.row(rho).norm()));
    }

    // the Dirac mass field
    const Eigen::VectorXd measure = spike_measure(fit);
    for ( int rho : model_rows(fit) )
    {
        CHECK(measure(rho) == doctest::Approx(fit.m1_diag(rho) * fit.s(rho)));
    }
}
