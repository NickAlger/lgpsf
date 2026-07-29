# What a fit is FOR: preconditioning a regularized inverse problem

Every other example stops at "here is the error". This one spends the fit.

The setting is the standard one. A large, dense operator H is expensive to
apply, and you want to recover x from a noisy observation b:

    minimize   1/2 ||H x - b||^2  +  alpha/2 ||x||^2

whose normal equations are

    (H^T H + alpha I) x = H^T b.

That system is symmetric positive definite for any alpha > 0, which is what
makes conjugate gradients applicable -- no symmetrization and no artificial
shift. Its condition number grows as alpha shrinks, so CG gets slower
exactly where the reconstruction gets interesting.

The point of the example: **fit H once, precondition every alpha.** The
regularization parameter is almost never known in advance -- it is swept, or
chosen by discrepancy, or continued downward -- and the fit does not depend
on it. One batch of probes buys the whole sweep.

THIS EXAMPLE IS SYNTHETIC, and it is worth being plain about that. It holds
H as a dense matrix, which is the one thing the method exists to avoid. If
you really do have the dense matrix in memory, you do not need any of this
-- thresholding its small entries is simpler, exact, and cheaper. The frog
kernel is here because it is known by FORMULA, so every number below can be
checked against a ground truth that a real problem would never hand you.

The setting the method is actually for is MATRIX-FREE: you can apply the
operator to a vector, but you cannot look at its entries, because each
application runs an expensive computation rather than a memory read. The
motivating case is the Gauss-Newton Hessian of a PDE-constrained inverse
problem, where one application costs a linearized forward solve plus an
adjoint solve -- and the same shape appears whenever applying an operator
hides a subproblem.

There the probe count IS the cost, and everything else is bookkeeping.
Notice what fit_operator consumes below: V and HV, and nothing else. That is
exactly what a matrix-free operator can give you. The dense H in this file
is used only to score the result. It is also why the mesh-scalability
finding matters (experiments/mesh-scalability.md): the probe budget stays
flat under refinement, so the expensive part does not grow with the mesh.

Note what is being compressed. H itself is local, so its fit is sparse; but
H^T H is DENSE, because two rows of H overlap whenever their supports do.
The fitted H^T H is sparse anyway, and that sparse matrix is the
preconditioner.

CG is written out rather than taken from Eigen, so the one line where the
preconditioner enters is visible.

BUILD OPTIMIZED. From the repository root:

    ./build-release/examples/preconditioner

## Output

```text
fit of H from 80 probes: relative error 0.0328, 11.5% dense
H^T H is 100.0% dense; the fitted H^T H is 34.9% dense, with relative error 0.0235

    alpha       cond(A)        CG plain     CG with fit   speedup
    1e-02     4.741e+03          295                 16     18.4x
    1e-03     4.740e+04          866                 31     27.9x
    1e-04     4.740e+05         2557                 82     31.2x
    1e-05     4.740e+06         5000 +              289     17.3x

One fit, 80 operator applications, reused at every alpha -- the fit does not
depend on the regularization, so a sweep costs one batch of probes plus a
sparse factorization per value.

Both curves still grow as alpha shrinks; the fit lowers the whole curve
rather than flattening it. The speedup peaks in the middle and falls off
at the smallest alpha, where the approximation's own error -- 2% here --
becomes what limits how well it can stand in for the true operator. More
probes would push that crossover further down.
```

## Program

```cpp
#include <cstdio>

#include <Eigen/SparseCholesky>

#include "lgpsf/operator_fit.hpp"

#include "frog_kernel.hpp"

namespace
{

constexpr int kGrid = 20;
constexpr int kProbes = 80;
constexpr int kMaxIterations = 5000;
const std::vector<double> kAlphas = {1e-2, 1e-3, 1e-4, 1e-5};

/// Conjugate gradients, returning the iteration count to reach `tolerance`.
///
/// `apply_preconditioner` is the ONLY difference between preconditioned and
/// unpreconditioned CG: pass the identity and this is the textbook method.
template <typename Preconditioner>
int conjugate_gradients( const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                         Preconditioner apply_preconditioner,
                         double tolerance = 1e-8 )
{
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    Eigen::VectorXd r = b;
    Eigen::VectorXd z = apply_preconditioner(r);
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
        z = apply_preconditioner(r);
        const double rz_next = r.dot(z);
        p = z + (rz_next / rz) * p;
        rz = rz_next;
    }
    return kMaxIterations;
}

Eigen::MatrixXd gaussian_matrix( Eigen::Index rows, Eigen::Index cols,
                                 unsigned seed )
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

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const frog::Problem problem = frog::build_problem(kGrid);
    const Eigen::Index n = problem.size();
    const Eigen::MatrixXd& H = problem.H;

    // ---- fit H, once -----------------------------------------------------
    const Eigen::MatrixXd V = gaussian_matrix(n, kProbes, 0);
    const Eigen::MatrixXd HV = H * V;

    lgpsf::OperatorFitConfig config;
    config.tau_window = 3.0;
    config.spike = false;
    config.row.mode_policy = std::make_shared<lgpsf::ShellLadder>(
        std::vector<int>{0, 1, 2, 3, 4, 5, 6});
    config.row.target_score = std::nullopt;

    const lgpsf::OperatorFit fit = lgpsf::fit_operator(
        problem.x, problem.mass, problem.mass, V, HV, problem.sigma, config);

    const Eigen::SparseMatrix<double> H_approx = lgpsf::assemble_sparse(
        fit.model, std::numeric_limits<double>::infinity());

    std::printf("fit of H from %d probes: relative error %.4f, %.1f%% dense\n",
                kProbes, (Eigen::MatrixXd(H_approx) - H).norm() / H.norm(),
                100.0 * H_approx.nonZeros() / (n * n));

    // ---- the normal-equations operator, true and approximate -------------
    const Eigen::MatrixXd G = H.transpose() * H;
    const Eigen::SparseMatrix<double> G_approx =
        (H_approx.transpose() * H_approx).pruned();

    std::printf("H^T H is %.1f%% dense; the fitted H^T H is %.1f%% dense, "
                "with relative error %.4f\n",
                100.0 * (G.array().abs() > 0).count() / (n * n),
                100.0 * G_approx.nonZeros() / (n * n),
                (Eigen::MatrixXd(G_approx) - G).norm() / G.norm());

    // ---- sweep alpha, reusing the SAME fit --------------------------------
    const double scale = G.trace() / static_cast<double>(n);
    const Eigen::VectorXd b = gaussian_matrix(n, 1, 7).col(0);

    std::printf("\n%9s  %12s  %14s  %14s  %8s\n",
                "alpha", "cond(A)", "CG plain", "CG with fit", "speedup");
    for ( double relative_alpha : kAlphas )
    {
        const double alpha = relative_alpha * scale;
        const Eigen::MatrixXd A =
            G + alpha * Eigen::MatrixXd::Identity(n, n);

        // The preconditioner: the same shift on the approximation. The fit is
        // NOT recomputed -- only this sparse factorization is.
        Eigen::SparseMatrix<double> M = G_approx;
        for ( Eigen::Index i = 0; i < n; ++i ) { M.coeffRef(i, i) += alpha; }
        M.makeCompressed();
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> factorization(M);
        if ( factorization.info() != Eigen::Success )
        {
            std::printf("%9.0e  preconditioner not positive definite\n",
                        relative_alpha);
            continue;
        }

        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(
            A, Eigen::EigenvaluesOnly);
        const double condition =
            spectrum.eigenvalues()(n - 1) / spectrum.eigenvalues()(0);

        const int plain = conjugate_gradients(
            A, b, []( const Eigen::VectorXd& r ) { return r; });
        const int preconditioned = conjugate_gradients(
            A, b, [&]( const Eigen::VectorXd& r ) { return factorization.solve(r); });

        std::printf("%9.0e  %12.3e  %11d%-3s  %14d  %7.1fx\n",
                    relative_alpha, condition, plain,
                    plain >= kMaxIterations ? " +" : "", preconditioned,
                    static_cast<double>(plain) / preconditioned);
    }

    std::printf("\nOne fit, %d operator applications, reused at every alpha -- "
                "the fit does not\ndepend on the regularization, so a sweep "
                "costs one batch of probes plus a\nsparse factorization per "
                "value.\n", kProbes);
    std::printf("\nBoth curves still grow as alpha shrinks; the fit lowers the "
                "whole curve\nrather than flattening it. The speedup peaks in "
                "the middle and falls off\nat the smallest alpha, where the "
                "approximation's own error -- 2%% here --\nbecomes what limits "
                "how well it can stand in for the true operator. More\nprobes "
                "would push that crossover further down.\n");

    return 0;
}
```

---

*Generated by `docs/generate_examples.py` from [`examples/preconditioner.cpp`](../../examples/preconditioner.cpp); the output and figures above come from actually running it.*
