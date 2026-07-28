// SPDX-License-Identifier: MIT
//
// Mesh scalability: what refinement costs
//
// The claim the method rests on is that a point-spread function is a property
// of the CONTINUUM operator, not of the mesh. If that is true, refining the
// mesh should not require more probes: the same k should buy the same
// accuracy, because it is buying the same shapes.
//
// This measures whether that holds, and what else moves with it. Per grid:
//
//   * relative Frobenius error against the dense truth, over a sweep of k
//   * the probe budget needed to reach fixed accuracy targets
//   * the density of the fitted H, and of the fitted H^T H
//   * CG iterations for (H^T H + alpha I) x = b at fixed relative alpha,
//     unpreconditioned and preconditioned by the fit
//
// Costs are quadratic in the dof count because the TRUTH is dense; the fit
// itself is not. Grid 48 means a 2304 x 2304 dense reference.
//
// Build (Release, and never a bare -j):
//
//   cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
//   cmake --build build-release --target mesh_scalability -j3
//   ./build-release/experiments/mesh_scalability > experiments/mesh_scalability.txt

#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/SparseCholesky>

#include "lgpsf/operator_fit.hpp"

#include "frog_kernel.hpp"

namespace
{

const std::vector<int> kGrids = {12, 16, 24, 32, 48};
const std::vector<int> kBudgets = {10, 20, 30, 45, 70, 110};
const std::vector<double> kTargets = {0.10, 0.05, 0.02};
constexpr double kRelativeAlpha = 1e-4;
constexpr int kMaxIterations = 20000;

Eigen::MatrixXd gaussian_matrix( Eigen::Index rows, Eigen::Index cols, unsigned seed )
{
    std::mt19937 generator(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::MatrixXd out(rows, cols);
    for ( Eigen::Index j = 0; j < cols; ++j )
    {
        for ( Eigen::Index i = 0; i < rows; ++i ) { out(i, j) = normal(generator); }
    }
    return out;
}

lgpsf::OperatorFit fit_at( const frog::Problem& problem, int num_probes )
{
    const Eigen::MatrixXd V = gaussian_matrix(problem.size(), num_probes, 0);
    const Eigen::MatrixXd HV = problem.H * V;

    lgpsf::OperatorFitConfig config;
    config.tau_window = 3.0;
    config.spike = false;
    config.row.mode_policy = std::make_shared<lgpsf::ShellLadder>(
        std::vector<int>{0, 1, 2, 3, 4, 5, 6});
    config.row.target_score = std::nullopt;
    return lgpsf::fit_operator(problem.x, problem.mass, problem.mass, V, HV,
                               problem.sigma, config);
}

template <typename Preconditioner>
int conjugate_gradients( const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                         Preconditioner apply, double tolerance = 1e-8 )
{
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    Eigen::VectorXd r = b;
    Eigen::VectorXd z = apply(r);
    Eigen::VectorXd p = z;
    double rz = r.dot(z);
    const double scale = b.norm();
    for ( int iteration = 1; iteration <= kMaxIterations; ++iteration )
    {
        const Eigen::VectorXd Ap = A * p;
        const double alpha = rz / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;
        if ( r.norm() / scale < tolerance ) { return iteration; }
        z = apply(r);
        const double rz_next = r.dot(z);
        p = z + (rz_next / rz) * p;
        rz = rz_next;
    }
    return kMaxIterations;
}

/// Linear interpolation, in log(error) against k, of the budget reaching
/// `target`. Reports -1 when the sweep never gets there.
double budget_for( const std::vector<int>& budgets,
                   const std::vector<double>& errors, double target )
{
    for ( std::size_t i = 0; i < errors.size(); ++i )
    {
        if ( errors[i] <= target )
        {
            if ( i == 0 ) { return budgets[0]; }
            const double t = (std::log(errors[i - 1]) - std::log(target))
                             / (std::log(errors[i - 1]) - std::log(errors[i]));
            return budgets[i - 1] + t * (budgets[i] - budgets[i - 1]);
        }
    }
    return -1.0;
}

} // namespace

/// Is the probe budget or the MODE LADDER the binding constraint at large k?
///
/// The counting rule allows m <= k/2 - P modes, which at k = 110 is 52; shells
/// to level 6 supply only 28. So the main sweep's large-k columns may be
/// reporting the ladder height rather than the probe budget, and the two have
/// to be separated before any of it means anything.
void ladder_study()
{
    std::printf("Is the ladder or the budget binding at large k? "
                "(k = 110, so the counting\nrule permits 52 modes)\n\n");
    std::printf("%5s %6s %7s %8s %10s\n", "grid", "dofs", "level", "modes",
                "rel. error");
    for ( int grid : {24, 32, 48} )
    {
        const frog::Problem problem = frog::build_problem(grid);
        const double truth_norm = problem.H.norm();
        for ( int max_level : {6, 8, 10} )
        {
            const Eigen::MatrixXd V = gaussian_matrix(problem.size(), 110, 0);
            const Eigen::MatrixXd HV = problem.H * V;

            std::vector<int> levels;
            for ( int i = 0; i <= max_level; ++i ) { levels.push_back(i); }
            lgpsf::OperatorFitConfig config;
            config.tau_window = 3.0;
            config.spike = false;
            config.row.mode_policy = std::make_shared<lgpsf::ShellLadder>(levels);
            config.row.target_score = std::nullopt;

            const lgpsf::OperatorFit fit = lgpsf::fit_operator(
                problem.x, problem.mass, problem.mass, V, HV, problem.sigma,
                config);
            const Eigen::SparseMatrix<double> A = lgpsf::assemble_sparse(
                fit.model, std::numeric_limits<double>::infinity());

            const std::vector<int> rows = lgpsf::model_rows(fit.model);
            double modes = 0.0;
            for ( int rho : rows )
            {
                modes += static_cast<double>(fit.model.row_modes(rho).size());
            }
            std::printf("%5d %6lld %7d %8.1f %10.4f\n", grid,
                        (long long)problem.size(), max_level,
                        rows.empty() ? 0.0 : modes / static_cast<double>(rows.size()),
                        (Eigen::MatrixXd(A) - problem.H).norm() / truth_norm);
        }
    }
}

int main( int argc, char** argv )
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if ( argc > 1 && std::string(argv[1]) == "ladder" )
    {
        ladder_study();
        return 0;
    }

    std::printf("Mesh scalability of the LG-PSF fit (rotating frog kernel)\n");
    std::printf("shells to level 6, tau_window 3, no spike, "
                "alpha = %.0e * trace(H^T H)/n\n\n", kRelativeAlpha);

    std::printf("%5s %6s", "grid", "dofs");
    for ( int k : kBudgets ) { std::printf(" %9s%-2d", "k=", k); }
    std::printf("\n");

    struct Summary
    {
        int grid, dofs;
        std::vector<double> errors;
        double k10, k05, k02;
        double h_density, g_density, g_error;
        int cg_plain, cg_precond;
        int k_used;
    };
    std::vector<Summary> summaries;

    for ( int grid : kGrids )
    {
        const frog::Problem problem = frog::build_problem(grid);
        const Eigen::Index n = problem.size();
        const double truth_norm = problem.H.norm();

        Summary summary;
        summary.grid = grid;
        summary.dofs = static_cast<int>(n);

        std::printf("%5d %6lld", grid, (long long)n);
        for ( int k : kBudgets )
        {
            const lgpsf::OperatorFit fit = fit_at(problem, k);
            const Eigen::SparseMatrix<double> A = lgpsf::assemble_sparse(
                fit.model, std::numeric_limits<double>::infinity());
            const double error =
                (Eigen::MatrixXd(A) - problem.H).norm() / truth_norm;
            summary.errors.push_back(error);
            std::printf("  %9.4f", error);
        }
        std::printf("\n");

        summary.k10 = budget_for(kBudgets, summary.errors, kTargets[0]);
        summary.k05 = budget_for(kBudgets, summary.errors, kTargets[1]);
        summary.k02 = budget_for(kBudgets, summary.errors, kTargets[2]);

        // Compression and CG, at the budget that reaches 5%.
        const int k_used = summary.k05 > 0
                               ? static_cast<int>(std::ceil(summary.k05))
                               : kBudgets.back();
        summary.k_used = k_used;

        const lgpsf::OperatorFit fit = fit_at(problem, k_used);
        const Eigen::SparseMatrix<double> H_approx = lgpsf::assemble_sparse(
            fit.model, std::numeric_limits<double>::infinity());
        const Eigen::SparseMatrix<double> G_approx =
            (H_approx.transpose() * H_approx).pruned();

        summary.h_density = 100.0 * H_approx.nonZeros() / (n * n);
        summary.g_density = 100.0 * G_approx.nonZeros() / (n * n);

        const Eigen::MatrixXd G = problem.H.transpose() * problem.H;
        summary.g_error = (Eigen::MatrixXd(G_approx) - G).norm() / G.norm();

        const double alpha = kRelativeAlpha * G.trace() / static_cast<double>(n);
        const Eigen::MatrixXd A = G + alpha * Eigen::MatrixXd::Identity(n, n);
        Eigen::SparseMatrix<double> M = G_approx;
        for ( Eigen::Index i = 0; i < n; ++i ) { M.coeffRef(i, i) += alpha; }
        M.makeCompressed();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> factorization(M);

        const Eigen::VectorXd b = gaussian_matrix(n, 1, 7).col(0);
        summary.cg_plain = conjugate_gradients(
            A, b, []( const Eigen::VectorXd& r ) { return r; });
        summary.cg_precond =
            factorization.info() == Eigen::Success
                ? conjugate_gradients(A, b, [&]( const Eigen::VectorXd& r ) {
                      return factorization.solve(r);
                  })
                : -1;
        summaries.push_back(std::move(summary));
    }

    std::printf("\n\nProbe budget to reach a fixed accuracy\n");
    std::printf("%5s %6s %10s %10s %10s\n", "grid", "dofs", "k @ 10%", "k @ 5%",
                "k @ 2%");
    for ( const Summary& s : summaries )
    {
        auto cell = []( double value ) {
            static char text[16];
            if ( value < 0 ) { std::snprintf(text, sizeof(text), "%10s", "> max"); }
            else { std::snprintf(text, sizeof(text), "%10.1f", value); }
            return std::string(text);
        };
        std::printf("%5d %6d %s %s %s\n", s.grid, s.dofs, cell(s.k10).c_str(),
                    cell(s.k05).c_str(), cell(s.k02).c_str());
    }

    std::printf("\n\nCompression and CG at the 5%% budget "
                "(alpha = %.0e relative)\n", kRelativeAlpha);
    std::printf("%5s %6s %5s %10s %12s %10s %9s %9s %8s\n", "grid", "dofs", "k",
                "fit(H) %", "fit(H^T H) %", "H^T H err", "CG plain", "CG prec",
                "speedup");
    for ( const Summary& s : summaries )
    {
        std::printf("%5d %6d %5d %10.1f %12.1f %10.4f %9d %9d %7.1fx\n", s.grid,
                    s.dofs, s.k_used, s.h_density, s.g_density, s.g_error,
                    s.cg_plain, s.cg_precond,
                    s.cg_precond > 0
                        ? static_cast<double>(s.cg_plain) / s.cg_precond
                        : 0.0);
    }

    return 0;
}
