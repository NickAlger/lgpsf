# Fit ONE point-spread function from random probes -- the row layer in C++

The C++ counterpart of `fit_one_psf.py`, and the other main entry point
besides `fit_operator`. Reach for it when you want a single PSF, when you
are diagnosing a row a whole-operator fit found hard, or when your target is
not an operator row at all -- the row layer only assumes a function you can
take inner products with.

You never hand the fitter function values. You hand it

    z   (K, k)   probe fields, one per COLUMN
    y   (k)      the inner products <z_i, phi>, one scalar per probe

and it recovers phi. For an operator row, z is the probes and y is a row of
H*V -- which is why fitting an operator costs matvecs, not entries.

THREE THINGS DIFFER FROM THE PYTHON, and they are the reason this example
exists rather than just pointing at the .py:

  1. Points are (K, N) -- points as ROWS. That is the transpose of the
     Python boundary, which takes (N, K). Both describe the same bytes; the
     binding transposes so each language gets its natural contiguous layout.
  2. The mode policy is a std::shared_ptr<ModePolicy>, not a value.
  3. Optional inputs are std::optional, so `sigma0` and `target_mass` are
     passed as engaged optionals rather than by keyword.

BUILD OPTIMIZED -- see operator_fit_frog.cpp. From the repository root:

    ./build-release/examples/fit_one_psf

## Output

```text
target row 320 at (0.562, 0.354)

    k   modes   held-out   rel. L2 vs truth  stop
   10       1     0.2806             0.5049  Exhausted
   20       6     0.3545             0.4974  Exhausted
   45      15     0.1291             0.1074  Exhausted
  110      28     0.0243             0.0289  Exhausted

  #                 candidate   modes         cost     score
  0                    sigma0       3     5.14e-07    0.4117
  1           circle r=0.0417       3     5.14e-07    0.4117
  2            circle r=0.827       3     5.14e-07    0.4117
  3            warm(level<=1)      10     6.48e-08    0.1888
  4                    sigma0      10     6.48e-08    0.1888  <-
  5           circle r=0.0417      10     6.48e-08    0.1888
  6            circle r=0.827      10     1.79e-06    0.7748
     cost is IN-SAMPLE and is never the selector; the held-out score is.
```

## Program

```cpp
#include <cstdio>
#include <vector>

#include "lgpsf/probe_fit.hpp"

#include "frog_kernel.hpp"

namespace
{

constexpr int kGrid = 24;
constexpr int kRow = 13 * kGrid + 8;    // a target away from the boundary
const std::vector<int> kBudgets = {10, 20, 45, 110};

const char* stop_name( lgpsf::StopReason reason )
{
    switch ( reason )
    {
        case lgpsf::StopReason::Target:       return "Target";
        case lgpsf::StopReason::ModePatience: return "ModePatience";
        case lgpsf::StopReason::Exhausted:    return "Exhausted";
    }
    return "?";
}

} // namespace

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const frog::Problem problem = frog::build_problem(kGrid);

    // Everything the fitter is told about the target, besides the probes:
    // where the columns are, their quadrature weights, where the PSF is
    // centered, and its approximate shape.
    const Eigen::VectorXd mu0 = problem.x.row(kRow).transpose();
    const Eigen::MatrixXd sigma0 = problem.sigma[kRow];
    const Eigen::VectorXd truth = problem.H.row(kRow).transpose();
    const double truth_norm = truth.norm();

    std::printf("target row %d at (%.3f, %.3f)\n", kRow, mu0(0), mu0(1));
    std::printf("\n%5s  %6s  %9s  %17s  %s\n",
                "k", "modes", "held-out", "rel. L2 vs truth", "stop");

    std::mt19937 generator(0);
    std::normal_distribution<double> normal(0.0, 1.0);

    for ( int k : kBudgets )
    {
        // z is (K, k): probes as COLUMNS, matching the points-as-rows layout.
        Eigen::MatrixXd z(problem.size(), k);
        for ( Eigen::Index j = 0; j < z.cols(); ++j )
        {
            for ( Eigen::Index i = 0; i < z.rows(); ++i ) { z(i, j) = normal(generator); }
        }
        const Eigen::VectorXd y = z.transpose() * truth;

        lgpsf::ProbeFitConfig config;
        config.mode_policy = std::make_shared<lgpsf::ShellLadder>(
            std::vector<int>{0, 1, 2, 3, 4, 5, 6});
        config.target_score = std::nullopt;      // sweep the whole ladder

        const lgpsf::ProbeFitResult result = lgpsf::fit_from_probes(
            problem.x, problem.mass, z, y, mu0, /*spike_index=*/-1, config,
            sigma0, problem.mass(kRow));

        // result.model is an LGExpansion: a standalone, evaluable model.
        // eval_expansion gives the smooth CONTINUUM kernel, so multiply by the
        // masses to compare against a discrete operator row.
        const Eigen::VectorXd predicted =
            problem.mass(kRow)
            * (lgpsf::eval_expansion(result.model, problem.x).array()
               * problem.mass.array());

        std::printf("%5d  %6d  %9.4f  %17.4f  %s\n", k,
                    static_cast<int>(result.model.modes.size()), result.score,
                    (predicted - truth).norm() / truth_norm,
                    stop_name(result.stop_reason));
    }

    // ---- the audit trail --------------------------------------------------
    // A row fit is a SEARCH over candidates, and the result carries all of it.
    // This is the first place to look when a row disappoints.
    {
        Eigen::MatrixXd z(problem.size(), 45);
        for ( Eigen::Index j = 0; j < z.cols(); ++j )
        {
            for ( Eigen::Index i = 0; i < z.rows(); ++i ) { z(i, j) = normal(generator); }
        }
        const Eigen::VectorXd y = z.transpose() * truth;

        lgpsf::ProbeFitConfig config;
        config.mode_policy =
            std::make_shared<lgpsf::ShellLadder>(std::vector<int>{1, 3});
        config.num_rungs = 2;
        config.window_shape_rungs = false;
        config.target_score = std::nullopt;

        const lgpsf::ProbeFitResult result = lgpsf::fit_from_probes(
            problem.x, problem.mass, z, y, mu0, -1, config, sigma0,
            problem.mass(kRow));

        std::printf("\n%3s  %24s  %6s  %11s  %8s\n",
                    "#", "candidate", "modes", "cost", "score");
        for ( std::size_t i = 0; i < result.candidates.size(); ++i )
        {
            const lgpsf::CandidateFit& c = result.candidates[i];
            std::printf("%3zu  %24s  %6d  %11.2e  %8.4f%s\n", i, c.label.c_str(),
                        static_cast<int>(c.model.modes.size()), c.cost, c.score,
                        static_cast<int>(i) == result.winner ? "  <-" : "");
        }
        std::printf("     cost is IN-SAMPLE and is never the selector; the "
                    "held-out score is.\n");
    }

    return 0;
}
```

---

*Generated by `docs/generate_examples.py` from [`examples/fit_one_psf.cpp`](../../examples/fit_one_psf.cpp); the output and figures above come from actually running it.*
