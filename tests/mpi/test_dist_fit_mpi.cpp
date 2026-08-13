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
// Not wired into CMake yet (needs an MPI toolchain); build + run:
//   mpicxx -O2 -std=c++17 -I../../include -I../../../ellipsoid_tree/include \
//       -I$EIGEN_INC test_dist_fit_mpi.cpp -o test_dist_fit_mpi
//   mpiexec -n 1 ./test_dist_fit_mpi && mpiexec -n 2 ./test_dist_fit_mpi \
//       && mpiexec -n 4 ./test_dist_fit_mpi

#include "lgpsf/mpi/dist_fit.hpp"
#include "lgpsf/mpi/dist_wsym.hpp"
#include "lgpsf/mode_policy.hpp"

#include <cmath>
#include <cstdio>
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

    // ---- serial reference on rank 0, broadcast -------------------------
    // (square legacy path: columns in global order, identity own dofs)
    std::vector<double> ref_vals;   // dense n*n row-major, zeros elsewhere
    ref_vals.assign(static_cast<std::size_t>(n) * n, 0.0);
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
            x_all, m_all, m_all, V_all, HV_all, sigma_all, config);
        const Eigen::SparseMatrix<double> B_ref = lgpsf::assemble_sparse(
            ref_fit.model, tau_assemble, lgpsf::Symmetrize::None,
            config.num_threads);
        for ( int outer = 0; outer < B_ref.outerSize(); ++outer )
        {
            for ( Eigen::SparseMatrix<double>::InnerIterator it(B_ref, outer);
                  it; ++it )
            {
                ref_vals[static_cast<std::size_t>(it.row()) * n
                         + static_cast<std::size_t>(it.col())] = it.value();
            }
        }
    }
    MPI_Bcast(ref_vals.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ---- distributed fit on contiguous gid blocks ----------------------
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
        lgpsf::mpi::make_window_ellipsoids(in.x_rows, in.sigma, config);
    const lgpsf::mpi::HaloPlan plan =
        lgpsf::mpi::halo_plan(MPI_COMM_WORLD, windows, in.x_local,
                              in.col_gids, /*k_cut=*/32);
    const lgpsf::mpi::DistFitResult res =
        lgpsf::mpi::dist_fit(plan, in, windows, config, tau_assemble);

    // ---- bitwise comparison against the reference ----------------------
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
    // ---- distributed weighted symmetrization vs serial, bitwise --------
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

    long tot_bad = 0, tot_checked = 0, tot_wsym_bad = 0;
    MPI_Allreduce(&bad, &tot_bad, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&checked, &tot_checked, 1, MPI_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&wsym_bad, &tot_wsym_bad, 1, MPI_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    long halo_tot = plan.candidates_received, halo_sum = 0;
    MPI_Allreduce(&halo_tot, &halo_sum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if ( rank == 0 )
    {
        std::printf("[G-L2] n=%d ranks=%d: %ld entries checked, %ld fit + "
                    "%ld wsym mismatches (BITWISE), halo candidates total "
                    "%ld\n",
                    n, size, tot_checked, tot_bad, tot_wsym_bad, halo_sum);
        std::printf(tot_bad + tot_wsym_bad == 0 ? "[G-L2] PASS\n"
                                                : "[G-L2] FAIL\n");
    }
    MPI_Finalize();
    return tot_bad + tot_wsym_bad == 0 ? 0 : 1;
}
