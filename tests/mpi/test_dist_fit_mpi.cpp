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
// which the accumulation path is exercised on most rows.  With the flag
// every pass below runs twice, plain and coarsened.
//
// ---------------------------------------------------------------------------
// THE ROW-REDISTRIBUTION PASSES (dev/row-balance-plan.md, slice 3)
//
// `DistFitInput::balance_tolerance` moves the LM search of some rows to
// another rank and brings the answers back.  A row's fit is a pure function of
// the package that travels with it, so the operator must be BITWISE identical
// to the undelegated one -- that claim, and the failure discipline of the
// plan's section 9, are what these passes mechanize.  Each runs against the
// SAME serial reference as G-L2 and demands the same zero mismatches:
//
//   balanced        a tolerance of 0 with deliberately skewed weights (the
//                   first half of the rows 101x the rest, which is the
//                   production shape: expensive rows cluster spatially), so a
//                   large fraction of the rows actually migrate.
//   balanced-cap    the same plan under a byte cap far too small to hold it.
//                   The cap is enforced in `solve` on the packed problem's
//                   TRUE size, and a migration it drops is not abandoned: the
//                   owner fits that row itself.  So the pass asserts that the
//                   cap bit (globally -- a rank that sheds nothing caps
//                   nothing) and that the answer is still bit-identical.
//   cap-local       every row hosted by rank 0, under a cap sized as a
//                   fraction of the problem's exact payload, so a large part
//                   of a planned migration is fitted at home while the rest
//                   still moves.  At n >= 3 it is rank 0's INCOMING budget
//                   that binds, refused on the acknowledgement round; at n = 2
//                   it is the single sender's outgoing total.  Bitwise
//                   identity under a heavily capped plan is the claim.
//   perverse        every row hosted by (owner + 1) mod size.  The strongest
//                   form of the claim -- nothing stays home -- and the pass
//                   that catches a package missing a member the search reads.
//                   Vacuous at n = 1 (the identity), so it is SKIPPED there
//                   rather than passed.
//   failures        the perverse assignment plus two poisoned rows: one whose
//                   response carries a NaN, so its fit THROWS on the foreign
//                   host and the message must come home attributed to that
//                   row; and one whose sigma is not positive definite, so it
//                   is never attempted and must never be handed a slot.  The
//                   job must not hang, and the migrated count must equal the
//                   attempted rows exactly.
//   empty-rank      rank 0 owns no rows at all (it still owns columns, and
//                   still runs the reference).  It is then the emptiest host
//                   and receives; n >= 2 only.
//
// ---------------------------------------------------------------------------
//
// Not wired into CMake yet (needs an MPI toolchain); build + run:
//   mpicxx -O2 -std=c++17 -pthread -I../../include \
//       -I../../../ellipsoid_tree/include -I$EIGEN_INC \
//       test_dist_fit_mpi.cpp -o test_dist_fit_mpi
//   mpiexec -n 1 ./test_dist_fit_mpi && mpiexec -n 2 ./test_dist_fit_mpi \
//       && mpiexec -n 4 ./test_dist_fit_mpi
//   (same three with `--coarsen` for the coarsened half)

#include "lgpsf/mpi/dist_fit.hpp"
#include "lgpsf/mpi/dist_wsym.hpp"
#include "lgpsf/mode_policy.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
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

/// One comparison of the distributed fit against the serial reference. The
/// row-redistribution knobs are part of the spec, so every pass is the same
/// comparison against the same reference with only the schedule moved.
struct PassSpec
{
    const char* label = "G-L2";
    bool coarsen = false;

    /// Turn the redistribution on. `perverse` replaces the rule with "every
    /// row on (owner + 1) mod size"; otherwise the water-filling rule runs at
    /// `tolerance` over the skewed weights (or window sizes, if unset).
    bool balance = false;
    /// `dist_fit` reads 0 as OFF, so a pass that wants the rule to target a
    /// perfectly even split asks for the smallest tolerance that is still on.
    double tolerance = 0.0;
    bool skewed_weights = false;
    bool perverse = false;
    std::size_t bytes_cap = lgpsf::mpi::RowExchangeOptions().bytes_cap;
    /// A cap sized as a fraction of the problem's EXACT payload (the wire
    /// size of every row's package, from the reference's fit points), rather
    /// than an absolute byte count: what binds is then the same whatever the
    /// mesh, the probe count or the coarsening do.  0 = use `bytes_cap`.
    double cap_fraction = 0.0;
    /// Host every row on rank 0, so one rank receives from all the others.
    bool all_on_rank0 = false;

    /// Rank 0 owns no rows (rank 1 owns its share as well).
    bool empty_first_rank = false;

    /// A row whose response carries a NaN, so its FIT throws wherever it runs;
    /// and a row whose sigma is not SPD, so it is never attempted at all.
    int poison_row = -1;
    int gated_row = -1;

    /// Demand that rows actually moved, and (for the capped passes) that the
    /// byte cap actually bit -- both summed over ranks, since a rank that
    /// sheds nothing caps nothing.  Meaningless at one rank.
    bool expect_migration = false;
    bool expect_capping = false;
    /// Demand that the RECEIVE-side budget bit, i.e. that a host refused a
    /// peer on the acknowledgement round and that peer fitted those rows
    /// itself.  Needs at least two senders into one host, so n >= 3.
    bool expect_receiver_capping = false;
};

} // namespace

int main( int argc, char** argv )
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- command line: the coarsened half is opt-in ---------------------
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
    const auto run_pass = [&]( const PassSpec& spec ) -> long
    {
        const char* label = spec.label;
        lgpsf::OperatorFitConfig cfg = config;
        if ( spec.coarsen )
        {
            cfg.coarsen_above = coarsen_above;
            cfg.coarsen_eps = coarsen_eps;
        }

        // ---- the injected pathologies, identical on both sides ---------
        //
        // A row that FAILS must fail the same way in the reference and in the
        // distributed fit -- that is the point: a foreign host throwing is not
        // supposed to change the answer, only where the throw happened.
        std::vector<Eigen::MatrixXd> sigma_pass = sigma_all;
        if ( spec.gated_row >= 0 )
        {
            // Not positive definite: fit_operator's pre-pass never attempts
            // this row, so its window size is 0 and no assignment may hand it
            // a slot.
            Eigen::MatrixXd bad(2, 2);
            bad << 1e-4, 0.0, 0.0, -1e-4;
            sigma_pass[static_cast<std::size_t>(spec.gated_row)] = bad;
        }
        const auto poisoned_response =
            [&]( int i, Eigen::Ref<Eigen::VectorXd> out )
        {
            response_row(i, out);
            if ( i == spec.poison_row )
            {
                // Every mode set then scores NaN, no baseline is selected, and
                // phase B throws -- wherever phase B is running.
                out(0) = std::numeric_limits<double>::quiet_NaN();
            }
        };

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
                poisoned_response(i, row);
                HV_all.row(i) = row.transpose();
            }
            const lgpsf::OperatorFit ref_fit = lgpsf::fit_operator(
                x_all, m_all, m_all, V_all, HV_all, sigma_pass, cfg);
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

        // ---- the byte cap, sized from the payload that will actually exist -
        //
        // `wire::problem_doubles` is the layout the packer writes, so this is
        // the exact number of bytes the whole problem would put on the wire if
        // every row moved.  A cap given as a fraction of it binds the same way
        // whatever the mesh, the probe count or the coarsening do -- which an
        // absolute byte count does not.
        std::size_t bytes_cap = spec.bytes_cap;
        if ( spec.cap_fraction > 0.0 )
        {
            std::size_t payload = 0;
            for ( int i = 0; i < n; ++i )
            {
                payload += sizeof(double)
                           * lgpsf::mpi::wire::problem_doubles(
                               static_cast<std::size_t>(
                                   ref_fit_points[static_cast<std::size_t>(i)]),
                               2u, static_cast<std::size_t>(k));
            }
            bytes_cap = static_cast<std::size_t>(
                spec.cap_fraction * static_cast<double>(payload));
            if ( bytes_cap == 0 ) { bytes_cap = 1; }
        }

        // ---- the two partitions ----------------------------------------
        //
        // Columns always split evenly; ROWS may not, so that a rank owning no
        // rows -- and rows whose own column dof lives on another rank -- are
        // exercised.
        const int cstart = (n * rank) / size;
        const int cend   = (n * (rank + 1)) / size;
        const int nloc   = cend - cstart;

        std::vector<long> row_ranges(static_cast<std::size_t>(size) + 1);
        for ( int r = 0; r <= size; ++r )
        {
            row_ranges[static_cast<std::size_t>(r)] = (long)(n * r) / size;
        }
        if ( spec.empty_first_rank && size > 1 )
        {
            row_ranges[1] = 0;   // rank 0 owns nothing; rank 1 owns its share
        }
        const int rstart =
            static_cast<int>(row_ranges[static_cast<std::size_t>(rank)]);
        const int rend =
            static_cast<int>(row_ranges[static_cast<std::size_t>(rank) + 1]);
        const int nrows = rend - rstart;

        lgpsf::mpi::DistFitInput in;
        in.x_local = x_all.middleRows(cstart, nloc);
        in.m2_local = m_all.segment(cstart, nloc);
        in.V_local = V_all.middleRows(cstart, nloc);
        in.col_gids.resize(static_cast<std::size_t>(nloc));
        for ( int i = 0; i < nloc; ++i )
        {
            in.col_gids[static_cast<std::size_t>(i)] = cstart + i;
        }
        in.x_rows = x_all.middleRows(rstart, nrows);
        in.m1_local = m_all.segment(rstart, nrows);
        in.sigma.assign(sigma_pass.begin() + rstart, sigma_pass.begin() + rend);
        in.row_own_gid.resize(static_cast<std::size_t>(nrows));
        for ( int i = 0; i < nrows; ++i )
        {
            in.row_own_gid[static_cast<std::size_t>(i)] = rstart + i;
        }
        in.HV_local.resize(nrows, k);
        {
            Eigen::VectorXd row(k);
            for ( int i = 0; i < nrows; ++i )
            {
                poisoned_response(rstart + i, row);
                in.HV_local.row(i) = row.transpose();
            }
        }

        // The window ellipsoids go to `halo_plan` as well as to the fit, so
        // they are built from a SANITIZED sigma: a row the fit will refuse
        // still needs a sane footprint for the halo geometry, and what makes
        // the fit refuse it is `in.sigma`, which keeps the indefinite matrix.
        std::vector<Eigen::MatrixXd> sigma_window = in.sigma;
        if ( spec.gated_row >= rstart && spec.gated_row < rend )
        {
            sigma_window[static_cast<std::size_t>(spec.gated_row - rstart)] =
                sigma_all[static_cast<std::size_t>(spec.gated_row)];
        }
        const std::vector<ellipsoid_tree::Ellipsoid> windows =
            lgpsf::mpi::make_window_ellipsoids(in.x_rows, sigma_window, cfg);

        // ---- the redistribution knobs ----------------------------------
        if ( spec.balance )
        {
            in.balance_tolerance = spec.tolerance;
            in.balance_bytes_cap = bytes_cap;
            if ( spec.skewed_weights )
            {
                // The production shape: the expensive rows cluster spatially,
                // so the ranks owning the first half of the mesh are the ones
                // that have to shed.
                in.prev_evaluations = Eigen::VectorXd::Ones(nrows);
                for ( int i = 0; i < nrows; ++i )
                {
                    in.prev_evaluations(i) =
                        ( rstart + i < n / 2 ) ? 101.0 : 1.0;
                }
            }
            if ( spec.perverse )
            {
                const int self = rank, ranks = size;
                in.balance_assign =
                    [self, ranks]( const std::vector<int>& window_sizes )
                {
                    return std::vector<int>(window_sizes.size(),
                                            (self + 1) % ranks);
                };
            }
            if ( spec.all_on_rank0 )
            {
                // Every sender aims at one host, which is what makes the
                // host's INCOMING budget, not any sender's outgoing one, the
                // thing that binds.
                in.balance_assign =
                    []( const std::vector<int>& window_sizes )
                {
                    return std::vector<int>(window_sizes.size(), 0);
                };
            }
        }

        const lgpsf::mpi::HaloPlan plan =
            lgpsf::mpi::halo_plan(MPI_COMM_WORLD, windows, in.x_local,
                                  in.col_gids, /*k_cut=*/32);
        const lgpsf::mpi::DistFitResult res =
            lgpsf::mpi::dist_fit(plan, in, windows, cfg, tau_assemble);

        // ---- bitwise comparison against the reference ------------------
        long bad = 0, checked = 0;
        {
            // distributed entries must match the reference exactly...
            std::vector<double> mine(static_cast<std::size_t>(nrows) * n, 0.0);
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
            for ( int i = 0; i < nrows; ++i )
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
        for ( int i = 0; i < nrows; ++i )
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

        // ---- what the redistribution did, and what it must have done ---
        long migrated = 0, migration_bad = 0, attempted = 0;
        for ( int i = 0; i < nrows; ++i )
        {
            if ( res.fitted_on_rank[static_cast<std::size_t>(i)] != rank )
            {
                ++migrated;
            }
            if ( rstart + i != spec.gated_row ) { ++attempted; }
        }
        if ( static_cast<int>(res.fitted_on_rank.size()) != nrows )
        {
            ++migration_bad;   // the field must always be filled
        }
        if ( spec.balance && spec.perverse )
        {
            // The host's phase-B seconds must come home: `row_seconds` means
            // everything attributable to the row wherever it ran, so it can
            // never be less than the search alone, and the searches that ran
            // elsewhere cannot all have taken no time at all.
            double foreign_search = 0.0;
            for ( int i = 0; i < nrows; ++i )
            {
                const double row_s = res.fit.diagnostics.row_seconds(i);
                const double search_s = res.fit.diagnostics.search_seconds(i);
                if ( search_s > row_s ) { ++migration_bad; }
                if ( res.fitted_on_rank[static_cast<std::size_t>(i)] != rank )
                {
                    foreign_search += search_s;
                }
            }
            if ( size > 1 && migrated > 0 && !(foreign_search > 0.0) )
            {
                ++migration_bad;
            }
            // Nothing may stay home except what was never attempted: the
            // migrated count is DERIVED FROM THE DATA and must equal exactly
            // the rows phase A packaged.
            if ( size > 1 && res.balance.rows_migrated != attempted )
            {
                ++migration_bad;
                std::printf("[%s] rank %d: %ld rows migrated, %ld attempted\n",
                            label, rank, res.balance.rows_migrated, attempted);
            }
            // The gated row must never have been handed a slot.
            if ( spec.gated_row >= rstart && spec.gated_row < rend )
            {
                const std::size_t local =
                    static_cast<std::size_t>(spec.gated_row - rstart);
                if ( res.fitted_on_rank[local] != rank ) { ++migration_bad; }
                if ( res.fit.diagnostics.status[local]
                     != lgpsf::RowStatus::Failed )
                {
                    ++migration_bad;
                }
                const auto found =
                    res.fit.diagnostics.failures.find(spec.gated_row - rstart);
                if ( found == res.fit.diagnostics.failures.end()
                     || found->second.find("positive definite")
                            == std::string::npos )
                {
                    ++migration_bad;
                }
                else
                {
                    std::printf("[%s] rank %d: gated row %d never assigned a "
                                "slot, \"%s\"\n", label, rank, spec.gated_row,
                                found->second.c_str());
                }
            }
            // The poisoned row throws where it is FITTED; the message must
            // come home attributed to that row, and the job must not hang.
            if ( spec.poison_row >= rstart && spec.poison_row < rend )
            {
                const std::size_t local =
                    static_cast<std::size_t>(spec.poison_row - rstart);
                if ( size > 1 && res.fitted_on_rank[local] == rank )
                {
                    ++migration_bad;   // it was supposed to leave
                }
                if ( res.fit.diagnostics.status[local]
                     != lgpsf::RowStatus::Failed )
                {
                    ++migration_bad;
                }
                const auto found =
                    res.fit.diagnostics.failures.find(spec.poison_row - rstart);
                if ( found == res.fit.diagnostics.failures.end()
                     || found->second.empty() )
                {
                    ++migration_bad;
                }
                else
                {
                    if ( size > 1
                         && found->second.find("[fitted on rank ")
                                == std::string::npos )
                    {
                        ++migration_bad;   // it must say where it threw
                    }
                    std::printf("[%s] rank %d: poisoned row %d came home as "
                                "\"%s\"\n", label, rank, spec.poison_row,
                                found->second.c_str());
                }
            }
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
            const std::vector<lgpsf::mpi::GlobalTriplet> mine =
                lgpsf::mpi::dist_weighted_symmetrize(MPI_COMM_WORLD, rows_local,
                                                     row_ranges);
            std::vector<double> got(static_cast<std::size_t>(nrows) * n, 0.0);
            for ( const lgpsf::mpi::GlobalTriplet& t : mine )
            {
                got[static_cast<std::size_t>(t.row - rstart) * n
                    + static_cast<std::size_t>(t.col)] = t.value;
            }
            for ( int i = 0; i < nrows; ++i )
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
        // the redistribution counts
        long mig_counts[6] = { migrated, migration_bad, res.balance.rows_hosted,
                               res.balance.bytes_sent,
                               res.balance.rows_capped_sender,
                               res.balance.rows_capped_receiver };
        long mig_sum[6] = { 0, 0, 0, 0, 0, 0 };
        MPI_Allreduce(mig_counts, mig_sum, 6, MPI_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        if ( spec.expect_migration && size > 1 && mig_sum[0] == 0 )
        {
            mig_sum[1] += 1;   // the pass was supposed to move rows
        }
        // The cap is enforced per rank, so it is the SUM that has to be
        // non-zero: a rank with nothing to shed caps nothing.
        if ( spec.expect_capping && size > 1 && mig_sum[4] + mig_sum[5] == 0 )
        {
            mig_sum[1] += 1;   // the cap was supposed to keep rows at home
        }
        if ( spec.expect_receiver_capping && size > 2 && mig_sum[5] == 0 )
        {
            mig_sum[1] += 1;   // ... and on the receive side, specifically
        }
        const long total_bad =
            tot_bad + tot_wsym_bad + tot_fp_bad + mig_sum[1];
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
            }
            std::printf("[%s] n=%d ranks=%d: %ld entries checked, %ld fit + "
                        "%ld wsym + %ld fit_points/window-size mismatches "
                        "(BITWISE), halo candidates total %ld\n",
                        label, n, size, tot_checked, tot_bad, tot_wsym_bad,
                        tot_fp_bad, halo_sum);
            if ( spec.balance )
            {
                std::printf("[%s] redistribution: %ld/%d rows fitted elsewhere "
                            "(%.1f%%), %ld KB on the wire, %ld rows kept home "
                            "by the byte cap (%ld sender-side, %ld receiver-"
                            "side, cap %ld KB), predicted imbalance %.3f "
                            "(target %.4g), %ld protocol mismatches\n",
                            label, mig_sum[0], n,
                            100.0 * (double)mig_sum[0] / (double)n,
                            mig_sum[3] >> 10, mig_sum[4] + mig_sum[5],
                            mig_sum[4], mig_sum[5],
                            (long)(bytes_cap >> 10),
                            res.balance.predicted_imbalance, res.balance.target,
                            mig_sum[1]);
            }
            std::printf("[%s] %s\n", label, total_bad == 0 ? "PASS" : "FAIL");
        }
        return total_bad;
    };

    long failures = 0;
    // Reserved, not grown: `PassSpec::label` is a plain pointer into these.
    std::vector<std::string> labels;
    labels.reserve(14);
    for ( int coarse = 0; coarse < ( coarsen_pass ? 2 : 1 ); ++coarse )
    {
        const bool c = ( coarse == 1 );
        const std::string suffix = c ? " coarsened" : "";
        // The labels outlive the passes: `PassSpec::label` is a plain pointer.
        const std::size_t first = labels.size();
        labels.push_back("G-L2" + suffix);
        labels.push_back("balanced" + suffix);
        labels.push_back("balanced-cap" + suffix);
        labels.push_back("cap-local" + suffix);
        labels.push_back("perverse" + suffix);
        labels.push_back("failures" + suffix);
        labels.push_back("empty-rank" + suffix);

        {
            PassSpec spec;
            spec.label = labels[first + 0].c_str();
            spec.coarsen = c;
            failures += run_pass(spec);
        }
        {
            // A tolerance of 0 targets a perfectly even split; the weights are
            // skewed 101:1 across the mesh, so the low ranks must shed.
            PassSpec spec;
            spec.label = labels[first + 1].c_str();
            spec.coarsen = c;
            spec.balance = true;
            spec.tolerance = 1e-9;
            spec.skewed_weights = true;
            spec.expect_migration = true;
            failures += run_pass(spec);
        }
        {
            // The same plan under a cap that cannot hold it.  Nearly every
            // planned migration is fitted by its owner instead -- which is not
            // a failure and not a lost row: the answer must still be bitwise
            // the reference's.
            PassSpec spec;
            spec.label = labels[first + 2].c_str();
            spec.coarsen = c;
            spec.balance = true;
            spec.tolerance = 1e-9;
            spec.skewed_weights = true;
            spec.bytes_cap = 64u * 1024u;
            spec.expect_capping = true;
            failures += run_pass(spec);
        }
        {
            // A cap that keeps a large fraction of a planned migration at
            // home while the rest still moves: the mixed path, where one rank
            // fits its own capped rows in the same parallel_for as the foreign
            // rows it is hosting.  Every row is hosted by rank 0, so at n >= 3
            // rank 0's incoming budget binds and the refusal rides the
            // acknowledgement round; at n = 2 the one sender's outgoing total
            // binds.  0.30 of the whole problem's payload against per-sender
            // shares of about 1/(n-1) of it is what makes that so.
            PassSpec spec;
            spec.label = labels[first + 3].c_str();
            spec.coarsen = c;
            spec.balance = true;
            spec.all_on_rank0 = true;
            spec.cap_fraction = 0.30;
            spec.expect_migration = true;
            spec.expect_capping = true;
            spec.expect_receiver_capping = true;
            failures += run_pass(spec);
        }
        if ( size > 1 )
        {
            // (owner + 1) mod size is the identity at one rank, so this pass
            // is skipped there rather than passed vacuously.
            PassSpec spec;
            spec.label = labels[first + 4].c_str();
            spec.coarsen = c;
            spec.balance = true;
            spec.perverse = true;
            spec.bytes_cap = static_cast<std::size_t>(-1);
            spec.expect_migration = true;
            failures += run_pass(spec);
        }
        {
            PassSpec spec;
            spec.label = labels[first + 5].c_str();
            spec.coarsen = c;
            spec.balance = true;
            spec.perverse = true;
            spec.bytes_cap = static_cast<std::size_t>(-1);
            spec.poison_row = 7;    // throws in phase B, wherever it runs
            spec.gated_row = 13;    // never attempted at all
            spec.expect_migration = ( size > 1 );
            failures += run_pass(spec);
        }
        if ( size > 1 )
        {
            PassSpec spec;
            spec.label = labels[first + 6].c_str();
            spec.coarsen = c;
            spec.balance = true;
            spec.tolerance = 1e-9;
            spec.skewed_weights = true;
            spec.empty_first_rank = true;
            spec.expect_migration = true;
            failures += run_pass(spec);
        }
    }
    if ( rank == 0 )
    {
        std::printf("[gate] ranks=%d: %s (%ld mismatches)\n", size,
                    failures == 0 ? "ALL PASS" : "FAILURES", failures);
    }
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
