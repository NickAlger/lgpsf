// SPDX-License-Identifier: MIT
//
// Gate G-L2 (Tier-B plan): the distributed lgpsf fit must reproduce the
// serial fit BITWISE at every rank count.
//
// Synthetic square problem: a jittered 2D grid of points (= rows), mildly
// anisotropic a-priori sigmas, hashed-RNG probes (a pure function of
// (gid, probe) — partition-independent), and responses from an analytic
// Gaussian-kernel operator evaluated in fixed ascending-gid order (so HV
// is bitwise identical no matter which rank computes it).  Rank 0 also
// runs the plain serial fit_operator/assemble_sparse on the full problem
// and broadcasts the reference; every rank then compares its distributed
// rows bitwise (values AND column sets).
//
// A second pass, `--coarsen` (S5 of dev/window-coarsening-plan.md), runs
// the same problem with the graded window coarsening on
// (`OperatorFitConfig::coarsen_above` / `coarsen_eps`) and demands the same
// bitwise agreement: the rank-independence claim for `coarsen_window`, made
// concrete.  It also compares `FitDiagnostics::fit_points` row by row and
// prints how many rows actually coarsened (fit_points < window size) on
// both sides.  `--coarsen-above N` and `--coarsen-eps X` override the pass's
// trigger (default 20, so most windows on this mesh trip it) and grading
// ratio (default 2.0).  The ratio is deliberately coarse: the window radius
// is ~3.5 mesh spacings here, so cells only merge at a large eps.  Measured
// rows-with-merged-cells / fit points (from 20305 window points, 600 rows):
// eps 0.2 -> 0 / 20305, 0.5 -> 10 / 20290, 1.0 -> 289 / 19592,
// 2.0 -> 564 / 13046, 4.0 -> 564 / 12056.  So 0.2 fires the trigger
// without merging a single cell (the copy path only); 2.0 is the value at
// which the accumulation path is exercised on most rows.  Without the flag
// the run is the original single pass.
//
// Not wired into CMake yet (needs an MPI toolchain); build + run:
//   mpicxx -O2 -std=c++17 -pthread -I../../include \
//       -I../../../ellipsoid_tree/include -I$EIGEN_INC \
//       test_dist_fit_mpi.cpp -o test_dist_fit_mpi
//   mpiexec -n 1 ./test_dist_fit_mpi && mpiexec -n 2 ./test_dist_fit_mpi \
//       && mpiexec -n 4 ./test_dist_fit_mpi
//   (same three with `--coarsen` for the second pass)

#include "lgpsf/mpi/dist_fit.hpp"
#include "lgpsf/mpi/dist_wsym.hpp"
#include "lgpsf/mode_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

// splitmix64 -> xorshift64* -> one Box-Muller draw: a standard normal as
// a pure function of (seed, a, b).
double hashed_normal( unsigned long seed, long a, long b )
{
    unsigned long z = seed + 0x9E3779B97F4A7C15UL * (unsigned long)(a + 1)
                      + 0xC2B2AE3D27D4EB4FUL * (unsigned long)(b + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z = z ^ (z >> 31);
    unsigned long s = z ? z : 0x853C49E6748FEA9BUL;
    for ( ;; )
    {
        unsigned long x = s;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s = x;
        const double u1 =
            (double)((x * 2685821657736338717UL) >> 11) / 9007199254740992.0;
        x = s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s = x;
        const double u2 =
            (double)((x * 2685821657736338717UL) >> 11) / 9007199254740992.0;
        const double v1 = 2.0 * u1 - 1.0, v2 = 2.0 * u2 - 1.0;
        const double r2 = v1 * v1 + v2 * v2;
        if ( r2 > 0.0 && r2 < 1.0 )
        {
            return v1 * std::sqrt(-2.0 * std::log(r2) / r2);
        }
    }
}

double hashed_uniform( unsigned long seed, long a, long b )
{
    unsigned long z = seed + 0xD6E8FEB86659FD93UL * (unsigned long)(a + 1)
                      + 0xCA5A826395121157UL * (unsigned long)(b + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    z = z ^ (z >> 31);
    return (double)(z >> 11) / 9007199254740992.0;
}

} // namespace

int main( int argc, char** argv )
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- command line: the coarsened second pass is opt-in --------------
    bool   coarsen_pass  = false;
    int    coarsen_above = 20;
    double coarsen_eps   = 2.0;
    for ( int a = 1; a < argc; ++a )
    {
        if ( std::strcmp(argv[a], "--coarsen") == 0 )
        {
            coarsen_pass = true;
        }
        else if ( std::strcmp(argv[a], "--coarsen-above") == 0 && a + 1 < argc )
        {
            coarsen_pass = true;
            coarsen_above = std::atoi(argv[++a]);
        }
        else if ( std::strcmp(argv[a], "--coarsen-eps") == 0 && a + 1 < argc )
        {
            coarsen_pass = true;
            coarsen_eps = std::atof(argv[++a]);
        }
        else
        {
            if ( rank == 0 )
            {
                std::fprintf(stderr, "unknown argument '%s' (accepted: --coarsen, "
                             "--coarsen-above N, --coarsen-eps X)\n", argv[a]);
            }
            MPI_Finalize();
            return 2;
        }
    }

    // ---- global synthetic problem (every rank can evaluate all of it) --
    const int  nx = 30, ny = 20, n = nx * ny, k = 12;
    const double h = 0.1;
    const unsigned long seed = 20260813UL;

    Eigen::MatrixXd x_all(n, 2);
    Eigen::VectorXd m_all(n);
    std::vector<Eigen::MatrixXd> sigma_all(
        static_cast<std::size_t>(n));
    for ( int g = 0; g < n; ++g )
    {
        const int ix = g % nx, iy = g / nx;
        x_all(g, 0) = h * ix + 0.25 * h * (hashed_uniform(seed, g, 1) - 0.5);
        x_all(g, 1) = h * iy + 0.25 * h * (hashed_uniform(seed, g, 2) - 0.5);
        m_all(g) = h * h * (0.9 + 0.2 * hashed_uniform(seed, g, 3));
        const double a  = 0.06 + 0.02 * hashed_uniform(seed, g, 4);
        const double b  = 0.04 + 0.01 * hashed_uniform(seed, g, 5);
        const double th = 360.0 * hashed_uniform(seed, g, 6);
        sigma_all[static_cast<std::size_t>(g)] =
            lgpsf::oriented_sigma(a, b, th);
    }

    // probes: pure function of (gid, probe)
    Eigen::MatrixXd V_all(n, k);
    for ( int g = 0; g < n; ++g )
    {
        for ( int j = 0; j < k; ++j )
        {
            V_all(g, j) = hashed_normal(seed + 7, g, j);
        }
    }

    // analytic operator: anisotropic Gaussian kernel row i over sigma_i,
    // mass-weighted; responses summed in ASCENDING gid order (bitwise
    // identical wherever computed)
    const auto response_row = [&]( int i, Eigen::Ref<Eigen::VectorXd> out )
    {
        const Eigen::Matrix2d Sinv =
            sigma_all[static_cast<std::size_t>(i)].inverse();
        out.setZero();
        for ( int j = 0; j < n; ++j )
        {
            const Eigen::Vector2d d =
                (x_all.row(j) - x_all.row(i)).transpose();
            const double w = std::exp(-0.5 * d.dot(Sinv * d)) * m_all(j);
            for ( int p = 0; p < k; ++p )
            {
                out(p) += w * V_all(j, p);
            }
        }
    };

    // ---- fit config: the production shape ------------------------------
    lgpsf::OperatorFitConfig config;
    config.tau_window = 5.0;
    config.window_aspect_cap = 1.0;   // production: ball windows
    config.spike = true;
    config.row.mode_policy = std::make_shared<lgpsf::WedgeLadder>(10, 2);
    config.row.mu = lgpsf::MuPolicy::Pinned;
    config.num_threads = 2;
    const double tau_assemble = 6.0;

    // ---- one pass: serial reference on rank 0, distributed fit on
    //      contiguous gid blocks, bitwise comparison.  Returns the total
    //      mismatch count (collective). ---------------------------------
    const auto run_pass = [&]( const lgpsf::OperatorFitConfig& cfg,
                               const char* label ) -> long
    {
        // ---- serial reference on rank 0, broadcast ---------------------
        // (square legacy path: columns in global order, identity own dofs)
        std::vector<double> ref_vals;   // dense n*n row-major, zeros elsewhere
        ref_vals.assign(static_cast<std::size_t>(n) * n, 0.0);
        std::vector<int> ref_fit_points(static_cast<std::size_t>(n), 0);
        std::vector<int> ref_window_size(static_cast<std::size_t>(n), 0);
        if ( rank == 0 )
        {
            Eigen::MatrixXd HV_all(n, k);
            Eigen::VectorXd row(k);
            for ( int i = 0; i < n; ++i )
            {
                response_row(i, row);
                HV_all.row(i) = row.transpose();
            }
            const lgpsf::OperatorFit ref_fit = lgpsf::fit_operator(
                x_all, m_all, m_all, V_all, HV_all, sigma_all, cfg);
            const Eigen::SparseMatrix<double> B_ref = lgpsf::assemble_sparse(
                ref_fit.model, tau_assemble, lgpsf::Symmetrize::None,
                cfg.num_threads);
            for ( int outer = 0; outer < B_ref.outerSize(); ++outer )
            {
                for ( Eigen::SparseMatrix<double>::InnerIterator it(B_ref, outer);
                      it; ++it )
                {
                    ref_vals[static_cast<std::size_t>(it.row()) * n
                             + static_cast<std::size_t>(it.col())] = it.value();
                }
            }
            for ( int i = 0; i < n; ++i )
            {
                ref_fit_points[static_cast<std::size_t>(i)] =
                    ref_fit.diagnostics.fit_points(i);
                ref_window_size[static_cast<std::size_t>(i)] =
                    static_cast<int>(ref_fit.model.row_window(i).size());
            }
        }
        MPI_Bcast(ref_vals.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(ref_fit_points.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(ref_window_size.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

        // ---- distributed fit on contiguous gid blocks ------------------
        const int rstart = (n * rank) / size;
        const int rend   = (n * (rank + 1)) / size;
        const int nloc   = rend - rstart;

        lgpsf::mpi::DistFitInput in;
        in.x_local = x_all.middleRows(rstart, nloc);
        in.m2_local = m_all.segment(rstart, nloc);
        in.V_local = V_all.middleRows(rstart, nloc);
        in.col_gids.resize(static_cast<std::size_t>(nloc));
        for ( int i = 0; i < nloc; ++i )
        {
            in.col_gids[static_cast<std::size_t>(i)] = rstart + i;
        }
        in.x_rows = in.x_local;
        in.m1_local = in.m2_local;
        in.sigma.assign(sigma_all.begin() + rstart,
                        sigma_all.begin() + rend);
        in.row_own_gid = in.col_gids;
        in.HV_local.resize(nloc, k);
        {
            Eigen::VectorXd row(k);
            for ( int i = 0; i < nloc; ++i )
            {
                response_row(rstart + i, row);
                in.HV_local.row(i) = row.transpose();
            }
        }

        const std::vector<ellipsoid_tree::Ellipsoid> windows =
            lgpsf::mpi::make_window_ellipsoids(in.x_rows, in.sigma, cfg);
        const lgpsf::mpi::HaloPlan plan =
            lgpsf::mpi::halo_plan(MPI_COMM_WORLD, windows, in.x_local,
                                  in.col_gids, /*k_cut=*/32);
        const lgpsf::mpi::DistFitResult res =
            lgpsf::mpi::dist_fit(plan, in, windows, cfg, tau_assemble);

        // ---- bitwise comparison against the reference ------------------
        long bad = 0, checked = 0;
        {
            // distributed entries must match the reference exactly...
            std::vector<double> mine(static_cast<std::size_t>(nloc) * n, 0.0);
            for ( int outer = 0; outer < res.B_local.outerSize(); ++outer )
            {
                for ( Eigen::SparseMatrix<double>::InnerIterator
                          it(res.B_local, outer); it; ++it )
                {
                    const long gcol =
                        res.col_gids[static_cast<std::size_t>(it.col())];
                    mine[static_cast<std::size_t>(it.row()) * n
                         + static_cast<std::size_t>(gcol)] = it.value();
                }
            }
            // ...and vice versa (no missing / extra entries): compare the
            // full dense row images
            for ( int i = 0; i < nloc; ++i )
            {
                for ( int j = 0; j < n; ++j )
                {
                    const double a =
                        mine[static_cast<std::size_t>(i) * n
                             + static_cast<std::size_t>(j)];
                    const double b =
                        ref_vals[static_cast<std::size_t>(rstart + i) * n
                                 + static_cast<std::size_t>(j)];
                    ++checked;
                    if ( a != b ) { ++bad; }   // BITWISE
                }
            }
        }
        // ---- the fit's quadrature, row by row: fit_points and window
        //      size must agree with the serial fit; count what coarsened --
        long fp_bad = 0, rows_coarsened = 0, rows_zero = 0;
        for ( int i = 0; i < nloc; ++i )
        {
            const int fp = res.fit.diagnostics.fit_points(i);
            const int ws = static_cast<int>(res.fit.model.row_window(i).size());
            if ( fp != ref_fit_points[static_cast<std::size_t>(rstart + i)]
                 || ws != ref_window_size[static_cast<std::size_t>(rstart + i)] )
            {
                ++fp_bad;
            }
            if ( fp == 0 ) { ++rows_zero; }            // gated / failed
            else if ( fp < ws ) { ++rows_coarsened; }  // the trigger fired AND cells merged
        }
        // ---- distributed weighted symmetrization vs serial, bitwise ----
        long wsym_bad = 0;
        {
            std::vector<double> ref_wsym(static_cast<std::size_t>(n) * n, 0.0);
            if ( rank == 0 )
            {
                Eigen::SparseMatrix<double> A(n, n);
                std::vector<Eigen::Triplet<double>> trip;
                for ( int i = 0; i < n; ++i )
                {
                    for ( int j = 0; j < n; ++j )
                    {
                        const double v =
                            ref_vals[static_cast<std::size_t>(i) * n
                                     + static_cast<std::size_t>(j)];
                        if ( v != 0.0 ) { trip.emplace_back(i, j, v); }
                    }
                }
                // NOTE: exact-zero stored entries are dropped by this dense
                // round trip on BOTH sides below, so the zeros-kept pattern
                // is not exercised here; values are.
                A.setFromTriplets(trip.begin(), trip.end());
                const Eigen::SparseMatrix<double> W =
                    lgpsf::detail::weighted_symmetrize(A);
                for ( int outer = 0; outer < W.outerSize(); ++outer )
                {
                    for ( Eigen::SparseMatrix<double>::InnerIterator it(W, outer);
                          it; ++it )
                    {
                        ref_wsym[static_cast<std::size_t>(it.row()) * n
                                 + static_cast<std::size_t>(it.col())] =
                            it.value();
                    }
                }
            }
            MPI_Bcast(ref_wsym.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<lgpsf::mpi::GlobalTriplet> rows_local;
            for ( int outer = 0; outer < res.B_local.outerSize(); ++outer )
            {
                for ( Eigen::SparseMatrix<double>::InnerIterator
                          it(res.B_local, outer); it; ++it )
                {
                    if ( it.value() == 0.0 ) { continue; }  // match the dense ref
                    rows_local.push_back(lgpsf::mpi::GlobalTriplet{
                        static_cast<long>(rstart + it.row()),
                        res.col_gids[static_cast<std::size_t>(it.col())],
                        it.value()});
                }
            }
            std::vector<long> row_ranges(static_cast<std::size_t>(size) + 1);
            for ( int r = 0; r <= size; ++r )
            {
                row_ranges[static_cast<std::size_t>(r)] = (long)(n * r) / size;
            }
            const std::vector<lgpsf::mpi::GlobalTriplet> mine =
                lgpsf::mpi::dist_weighted_symmetrize(MPI_COMM_WORLD, rows_local,
                                                     row_ranges);
            std::vector<double> got(static_cast<std::size_t>(nloc) * n, 0.0);
            for ( const lgpsf::mpi::GlobalTriplet& t : mine )
            {
                got[static_cast<std::size_t>(t.row - rstart) * n
                    + static_cast<std::size_t>(t.col)] = t.value;
            }
            for ( int i = 0; i < nloc; ++i )
            {
                for ( int j = 0; j < n; ++j )
                {
                    if ( got[static_cast<std::size_t>(i) * n
                             + static_cast<std::size_t>(j)]
                         != ref_wsym[static_cast<std::size_t>(rstart + i) * n
                                     + static_cast<std::size_t>(j)] )
                    {
                        ++wsym_bad;
                    }
                }
            }
        }

        long tot_bad = 0, tot_checked = 0, tot_wsym_bad = 0, tot_fp_bad = 0;
        MPI_Allreduce(&bad, &tot_bad, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&checked, &tot_checked, 1, MPI_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&wsym_bad, &tot_wsym_bad, 1, MPI_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&fp_bad, &tot_fp_bad, 1, MPI_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        long halo_tot = plan.candidates_received, halo_sum = 0;
        MPI_Allreduce(&halo_tot, &halo_sum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        // the coarsening counts: distributed (summed over ranks) vs serial
        long dist_counts[4] = { rows_coarsened, rows_zero,
                                res.window_candidates, res.fit_points_total };
        long dist_sum[4] = { 0, 0, 0, 0 };
        MPI_Allreduce(dist_counts, dist_sum, 4, MPI_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        const long total_bad = tot_bad + tot_wsym_bad + tot_fp_bad;
        if ( rank == 0 )
        {
            if ( cfg.coarsen_above > 0 )
            {
                long ser_coarsened = 0, ser_zero = 0, ser_window = 0, ser_fit = 0;
                for ( int i = 0; i < n; ++i )
                {
                    const int fp = ref_fit_points[static_cast<std::size_t>(i)];
                    const int ws = ref_window_size[static_cast<std::size_t>(i)];
                    ser_window += ws;
                    ser_fit += fp;
                    if ( fp == 0 ) { ++ser_zero; }
                    else if ( fp < ws ) { ++ser_coarsened; }
                }
                std::printf("[%s] coarsen_above=%d coarsen_eps=%g: serial "
                            "%ld/%d rows coarsened (%ld gated/failed), window "
                            "points %ld -> fit points %ld; distributed %ld/%d "
                            "rows coarsened (%ld gated/failed), window_candidates "
                            "%ld -> fit_points_total %ld\n",
                            label, cfg.coarsen_above, cfg.coarsen_eps,
                            ser_coarsened, n, ser_zero, ser_window, ser_fit,
                            dist_sum[0], n, dist_sum[1], dist_sum[2],
                            dist_sum[3]);
                std::printf("[%s] n=%d ranks=%d: %ld entries checked, %ld fit + "
                            "%ld wsym + %ld fit_points/window-size mismatches "
                            "(BITWISE), halo candidates total %ld\n",
                            label, n, size, tot_checked, tot_bad, tot_wsym_bad,
                            tot_fp_bad, halo_sum);
            }
            else
            {
                std::printf("[%s] n=%d ranks=%d: %ld entries checked, %ld fit + "
                            "%ld wsym mismatches (BITWISE), halo candidates total "
                            "%ld\n",
                            label, n, size, tot_checked, tot_bad, tot_wsym_bad,
                            halo_sum);
                if ( tot_fp_bad != 0 )
                {
                    std::printf("[%s] %ld fit_points/window-size mismatches\n",
                                label, tot_fp_bad);
                }
            }
            std::printf("[%s] %s\n", label, total_bad == 0 ? "PASS" : "FAIL");
        }
        return total_bad;
    };

    long failures = run_pass(config, "G-L2");
    if ( coarsen_pass )
    {
        lgpsf::OperatorFitConfig coarse_config = config;
        coarse_config.coarsen_above = coarsen_above;
        coarse_config.coarsen_eps = coarsen_eps;
        failures += run_pass(coarse_config, "G-L2 coarsened");
    }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
