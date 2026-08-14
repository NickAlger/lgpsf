// SPDX-License-Identifier: MIT
//
// Integration test of the fit -> operator-QC pipeline on a mesh where the
// mass matrix cannot hide: the frog kernel on a cell-centered POLAR mesh of
// a disk, whose quadrature weights m = r*dr*dtheta span a large and
// refinement-GROWING ratio.  Companion to docs/varpro-whitening-notes.tex
// ("Operator-level quality control") and the narrated python example
// examples/operator_qc_mesh_study.py.
//
// Asserted here (robust to seeds and config drift):
//   1. ESTIMATOR VALIDITY: the energy-ratio QC on held-out L^2-white-noise
//      probes (z = M^-1/2 randn), with residual/response measured by
//      lgpsf::dual_norm2, tracks the EXACT dense relative dual error.
//   2. MESH INDEPENDENCE: the exact dual error of the white-noise-probed
//      fit changes little between resolutions while the mass ratio nearly
//      doubles -- a misplaced power of M in probes or weighting drifts.
// The A-vs-B probe-family trade-off is narrated in the example rather than
// asserted: it is a property of the problem, not a contract of the library.

#include "doctest/doctest.h"

#include "lgpsf/lgpsf.hpp"
#include "../examples/frog_kernel.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace {

struct Disk
{
    Eigen::MatrixXd x;                    // (n, 2)
    Eigen::VectorXd mass;                 // (n)
    Eigen::MatrixXd H;                    // (n, n) dense truth
    std::vector<Eigen::MatrixXd> sigma;
};

Disk disk_problem( int nr, int ntheta )
{
    const double radius = 0.42;
    const Eigen::Vector2d center(0.5, 0.5);
    const double dr = radius / nr;
    const double dth = 2.0 * M_PI / ntheta;

    Disk p;
    const Eigen::Index n = static_cast<Eigen::Index>(nr) * ntheta;
    p.x.resize(n, 2);
    p.mass.resize(n);
    Eigen::Index k = 0;
    for ( int i = 0; i < nr; ++i )
    {
        const double r = (i + 0.5) * dr;
        for ( int j = 0; j < ntheta; ++j, ++k )
        {
            const double th = (j + 0.5) * dth;
            p.x(k, 0) = center(0) + r * std::cos(th);
            p.x(k, 1) = center(1) + r * std::sin(th);
            p.mass(k) = r * dr * dth;
        }
    }
    p.H.resize(n, n);
    p.sigma.reserve(static_cast<std::size_t>(n));
    for ( Eigen::Index i = 0; i < n; ++i )
    {
        const Eigen::Vector2d target = p.x.row(i).transpose();
        p.H.row(i) = p.mass(i) * frog::frog_row(target, p.x).array()
                     * p.mass.array();
        p.sigma.push_back(frog::frog_covariance(target));
    }
    return p;
}

double exact_rel_dual( const Eigen::MatrixXd& H, const Eigen::MatrixXd& B,
                       const Eigen::VectorXd& mass )
{
    const Eigen::VectorXd w = mass.cwiseSqrt().cwiseInverse();
    const Eigen::MatrixXd D = w.asDiagonal() * (H - B) * w.asDiagonal();
    const Eigen::MatrixXd Hd = w.asDiagonal() * H * w.asDiagonal();
    return D.norm() / Hd.norm();
}

} // namespace

TEST_CASE("operator QC pipeline: estimator validity + mesh independence "
          "on a strongly nonuniform mesh")
{
    const int k_fit = 20, k_qc = 8;
    std::mt19937_64 gen(20260813);
    std::normal_distribution<double> randn(0.0, 1.0);

    std::vector<double> exact_dual;
    for ( const auto [nr, ntheta] : {std::pair{8, 24}, std::pair{12, 36}} )
    {
        const Disk p = disk_problem(nr, ntheta);
        const Eigen::Index n = p.mass.size();
        // mass ratio grows with nr: the mesh the scaling cannot hide on
        CHECK(p.mass.maxCoeff() / p.mass.minCoeff() > 2.0 * nr - 2.0);

        // L^2-white-noise probes: z = M^-1/2 randn
        const Eigen::VectorXd sm = p.mass.cwiseSqrt().cwiseInverse();
        Eigen::MatrixXd Z(n, k_fit + k_qc);
        for ( Eigen::Index j = 0; j < Z.cols(); ++j )
        {
            for ( Eigen::Index i = 0; i < n; ++i )
            {
                Z(i, j) = randn(gen) * sm(i);
            }
        }
        const Eigen::MatrixXd Y = p.H * Z;

        lgpsf::OperatorFitConfig config;
        config.tau_window = 3.0;
        config.spike = false;
        std::vector<int> shells;
        for ( int l = 0; l <= 8; ++l ) shells.push_back(l);
        config.row.mode_policy = std::make_shared<lgpsf::ShellLadder>(shells);
        config.row.target_score.reset();

        const lgpsf::OperatorFit fit = lgpsf::fit_operator(
            p.x, p.mass, p.mass, Z.leftCols(k_fit), Y.leftCols(k_fit),
            p.sigma, config);
        const Eigen::MatrixXd B = Eigen::MatrixXd(lgpsf::assemble_sparse(
            fit.model, std::numeric_limits<double>::infinity(),
            lgpsf::Symmetrize::None));

        // the implemented estimator: energy ratio in the dual norms
        double num2 = 0.0, den2 = 0.0;
        for ( int q = 0; q < k_qc; ++q )
        {
            const Eigen::VectorXd r =
                B * Z.col(k_fit + q) - Y.col(k_fit + q);
            num2 += lgpsf::dual_norm2(r, p.mass);
            den2 += lgpsf::dual_norm2(Y.col(k_fit + q), p.mass);
        }
        const double qc = std::sqrt(num2 / den2);
        const double exact = exact_rel_dual(p.H, B, p.mass);
        exact_dual.push_back(exact);

        CHECK(exact > 0.0);
        CHECK(exact < 1.0);
        // 1. estimator tracks the exact dual error (k_qc-probe noise)
        CHECK(qc > 0.55 * exact);
        CHECK(qc < 1.8 * exact);
    }
    // 2. mesh independence: the dual error means the same thing on both
    //    meshes (a misplaced power of M drifts with the mass ratio instead)
    CHECK(std::abs(exact_dual[1] - exact_dual[0])
          < 0.25 * exact_dual[0]);
}
