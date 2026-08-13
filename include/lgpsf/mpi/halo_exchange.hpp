// SPDX-License-Identifier: MIT
#ifndef LGPSF_MPI_HALO_EXCHANGE_HPP
#define LGPSF_MPI_HALO_EXCHANGE_HPP

/// \file halo_exchange.hpp
/// The distributed halo protocol of the Tier-B lgpsf fit (design record:
/// nicks_research_experiments/ellipsoid_psf_pig, "Tier-B halo design").
///
/// Roles: ROWS own window ellipsoids (footprints) living in the column
/// domain; COLUMNS own points.  Rank-local summaries (box-forest cuts of
/// each rank's footprint set) are allgathered; each rank pushes its
/// column points that touch a remote forest to that forest's owner; the
/// owner resolves candidates exactly with the usual dual-tree machinery.
/// Summary quality affects candidate-set size only, never correctness.
///
/// Determinism discipline: the received halo is SORTED BY GLOBAL ID, so
/// every downstream per-row structure has a partition-independent order.
///
/// This header knows MPI and ellipsoid_tree only — no PETSc, no
/// application types.  Include it from MPI-enabled translation units.

#include <ellipsoid_tree/aabb_tree.hpp>
#include <ellipsoid_tree/box_forest.hpp>
#include <ellipsoid_tree/geometry.hpp>

#include <Eigen/Dense>

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lgpsf {
namespace mpi {

/// The reusable geometry of one halo exchange: which of my columns each
/// neighbor needs, and where each received candidate lands in the sorted
/// halo.  Built once per footprint set (one lgpsf rebuild); any number of
/// per-column payloads (probe inputs, masses, smoothing fields) then
/// travel along it with `halo_push`.
struct HaloPlan
{
    MPI_Comm            comm = MPI_COMM_NULL;
    int                 nloc = 0;   ///< owned columns
    int                 dim  = 0;

    // send side: send_cols[i] = MY column indices neighbor send_ranks[i]
    // needs (ascending)
    std::vector<int>              send_ranks;
    std::vector<std::vector<int>> send_cols;

    // receive side (the halo): candidates from all sources, sorted by gid
    std::vector<long>             halo_gids;  ///< ascending, unique
    Eigen::MatrixXd               halo_x;     ///< (nhalo, dim)
    std::vector<int>              recv_ranks;
    std::vector<int>              recv_counts;
    std::vector<std::vector<int>> recv_perm;  ///< per source: slot -> halo row

    // telemetry (per-rank; reduce across comm for global numbers)
    long                candidates_sent     = 0;
    long                candidates_received = 0;
    int                 cut_boxes_local     = 0;

    int nhalo() const { return static_cast<int>(halo_gids.size()); }
};

/// Tight axis-aligned box of the ellipsoid {x : (x-mu)^T S^{-1} (x-mu) <= 1}:
/// half-widths sqrt(diag(S)).
inline void ellipsoid_aabb( const ellipsoid_tree::Ellipsoid& e,
                            Eigen::Ref<Eigen::VectorXd> lo,
                            Eigen::Ref<Eigen::VectorXd> hi )
{
    const Eigen::VectorXd w = e.Sigma.diagonal().cwiseMax(0.0).cwiseSqrt();
    lo = e.mu - w;
    hi = e.mu + w;
}

/// Build the halo plan.
///
/// `windows`: this rank's row footprints, PRE-SCALED (membership is
/// Mahalanobis <= 1) — the same objects later handed to `fit_operator` as
/// `window_ellipsoids`, so halo and fit windows agree by construction.
/// `x_local` (nloc, dim), `col_gids` (nloc, ascending): this rank's owned
/// column points.  `k_cut`: summary budget (boxes per rank).
inline HaloPlan halo_plan( MPI_Comm comm,
                           const std::vector<ellipsoid_tree::Ellipsoid>& windows,
                           const Eigen::Ref<const Eigen::MatrixXd>& x_local,
                           const std::vector<long>& col_gids,
                           int k_cut )
{
    HaloPlan plan;
    plan.comm = comm;
    plan.nloc = static_cast<int>(x_local.rows());
    plan.dim  = static_cast<int>(x_local.cols());
    if ( static_cast<int>(col_gids.size()) != plan.nloc )
    {
        throw std::invalid_argument("lgpsf::mpi::halo_plan: col_gids size");
    }
    if ( !std::is_sorted(col_gids.begin(), col_gids.end()) )
    {
        throw std::invalid_argument(
            "lgpsf::mpi::halo_plan: col_gids must be ascending "
            "(determinism discipline)");
    }
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const int dim = plan.dim;

    // ---- local footprint summary: box-forest cut ----------------------
    ellipsoid_tree::BoxForest cut;
    {
        const int nw = static_cast<int>(windows.size());
        Eigen::MatrixXd wlo(dim, nw), whi(dim, nw);
        for ( int i = 0; i < nw; ++i )
        {
            ellipsoid_aabb(windows[static_cast<std::size_t>(i)],
                           wlo.col(i), whi.col(i));
        }
        const ellipsoid_tree::AABBTree wtree(wlo, whi);
        cut = ellipsoid_tree::tree_cut(wtree, k_cut);
    }
    plan.cut_boxes_local = cut.count();

    // ---- allgather all ranks' cuts ------------------------------------
    std::vector<int> cut_counts(static_cast<std::size_t>(size), 0);
    {
        int mine = cut.count();
        MPI_Allgather(&mine, 1, MPI_INT, cut_counts.data(), 1, MPI_INT, comm);
    }
    std::vector<int> cut_displs(static_cast<std::size_t>(size) + 1, 0);
    for ( int r = 0; r < size; ++r )
    {
        cut_displs[static_cast<std::size_t>(r) + 1] =
            cut_displs[static_cast<std::size_t>(r)]
            + cut_counts[static_cast<std::size_t>(r)];
    }
    const int total_boxes = cut_displs[static_cast<std::size_t>(size)];
    // flattened per box: lo (dim) then hi (dim)
    std::vector<double> all_boxes(
        static_cast<std::size_t>(total_boxes) * 2 * dim, 0.0);
    {
        std::vector<double> mine(
            static_cast<std::size_t>(cut.count()) * 2 * dim);
        for ( int b = 0; b < cut.count(); ++b )
        {
            for ( int d = 0; d < dim; ++d )
            {
                mine[static_cast<std::size_t>(b) * 2 * dim + d] = cut.lo(d, b);
                mine[static_cast<std::size_t>(b) * 2 * dim + dim + d] =
                    cut.hi(d, b);
            }
        }
        std::vector<int> rc(static_cast<std::size_t>(size)),
            rd(static_cast<std::size_t>(size));
        for ( int r = 0; r < size; ++r )
        {
            rc[static_cast<std::size_t>(r)] =
                cut_counts[static_cast<std::size_t>(r)] * 2 * dim;
            rd[static_cast<std::size_t>(r)] =
                cut_displs[static_cast<std::size_t>(r)] * 2 * dim;
        }
        MPI_Allgatherv(mine.data(), static_cast<int>(mine.size()), MPI_DOUBLE,
                       all_boxes.data(), rc.data(), rd.data(), MPI_DOUBLE,
                       comm);
    }

    // ---- who needs which of my columns --------------------------------
    ellipsoid_tree::AABBTree point_tree;
    {
        Eigen::MatrixXd plo = x_local.transpose();  // (dim, nloc)
        point_tree.build(plo, plo);                 // degenerate boxes
    }
    std::vector<int> send_counts(static_cast<std::size_t>(size), 0);
    for ( int r = 0; r < size; ++r )
    {
        if ( r == rank || cut_counts[static_cast<std::size_t>(r)] == 0 )
        {
            continue;
        }
        ellipsoid_tree::BoxForest forest;
        const int nb = cut_counts[static_cast<std::size_t>(r)];
        forest.lo.resize(dim, nb);
        forest.hi.resize(dim, nb);
        for ( int b = 0; b < nb; ++b )
        {
            const std::size_t base = static_cast<std::size_t>(
                cut_displs[static_cast<std::size_t>(r)] + b) * 2 * dim;
            for ( int d = 0; d < dim; ++d )
            {
                forest.lo(d, b) = all_boxes[base + static_cast<std::size_t>(d)];
                forest.hi(d, b) =
                    all_boxes[base + static_cast<std::size_t>(dim + d)];
            }
        }
        std::vector<int> hits = ellipsoid_tree::forest_query(point_tree, forest);
        if ( !hits.empty() )
        {
            send_counts[static_cast<std::size_t>(r)] =
                static_cast<int>(hits.size());
            plan.send_ranks.push_back(r);
            plan.send_cols.push_back(std::move(hits));
            plan.candidates_sent +=
                static_cast<long>(plan.send_cols.back().size());
        }
    }

    // ---- exchange counts, then candidate (gid, coords) ----------------
    std::vector<int> recv_counts_all(static_cast<std::size_t>(size), 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts_all.data(), 1,
                 MPI_INT, comm);
    for ( int r = 0; r < size; ++r )
    {
        if ( recv_counts_all[static_cast<std::size_t>(r)] > 0 )
        {
            plan.recv_ranks.push_back(r);
            plan.recv_counts.push_back(
                recv_counts_all[static_cast<std::size_t>(r)]);
        }
    }

    const int nsend = static_cast<int>(plan.send_ranks.size());
    const int nrecv = static_cast<int>(plan.recv_ranks.size());
    std::vector<std::vector<long>>   sgid(static_cast<std::size_t>(nsend));
    std::vector<std::vector<double>> sxyz(static_cast<std::size_t>(nsend));
    std::vector<std::vector<long>>   rgid(static_cast<std::size_t>(nrecv));
    std::vector<std::vector<double>> rxyz(static_cast<std::size_t>(nrecv));
    std::vector<MPI_Request> reqs;
    reqs.reserve(2 * static_cast<std::size_t>(nsend + nrecv));
    for ( int i = 0; i < nrecv; ++i )
    {
        const int cnt = plan.recv_counts[static_cast<std::size_t>(i)];
        rgid[static_cast<std::size_t>(i)].resize(
            static_cast<std::size_t>(cnt));
        rxyz[static_cast<std::size_t>(i)].resize(
            static_cast<std::size_t>(cnt) * dim);
        reqs.emplace_back();
        MPI_Irecv(rgid[static_cast<std::size_t>(i)].data(), cnt, MPI_LONG,
                  plan.recv_ranks[static_cast<std::size_t>(i)], 71, comm,
                  &reqs.back());
        reqs.emplace_back();
        MPI_Irecv(rxyz[static_cast<std::size_t>(i)].data(), cnt * dim,
                  MPI_DOUBLE, plan.recv_ranks[static_cast<std::size_t>(i)],
                  72, comm, &reqs.back());
    }
    for ( int i = 0; i < nsend; ++i )
    {
        const std::vector<int>& cols =
            plan.send_cols[static_cast<std::size_t>(i)];
        std::vector<long>&   g = sgid[static_cast<std::size_t>(i)];
        std::vector<double>& x = sxyz[static_cast<std::size_t>(i)];
        g.resize(cols.size());
        x.resize(cols.size() * static_cast<std::size_t>(dim));
        for ( std::size_t k = 0; k < cols.size(); ++k )
        {
            g[k] = col_gids[static_cast<std::size_t>(cols[k])];
            for ( int d = 0; d < dim; ++d )
            {
                x[k * static_cast<std::size_t>(dim)
                  + static_cast<std::size_t>(d)] = x_local(cols[k], d);
            }
        }
        reqs.emplace_back();
        MPI_Isend(g.data(), static_cast<int>(g.size()), MPI_LONG,
                  plan.send_ranks[static_cast<std::size_t>(i)], 71, comm,
                  &reqs.back());
        reqs.emplace_back();
        MPI_Isend(x.data(), static_cast<int>(x.size()), MPI_DOUBLE,
                  plan.send_ranks[static_cast<std::size_t>(i)], 72, comm,
                  &reqs.back());
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(),
                MPI_STATUSES_IGNORE);

    // ---- sorted halo + per-source permutation -------------------------
    // (gids are globally unique: each column has one owner, and a source
    // sends a column at most once)
    struct Cand { long gid; int src; int slot; };
    std::vector<Cand> cands;
    for ( int i = 0; i < nrecv; ++i )
    {
        const std::vector<long>& g = rgid[static_cast<std::size_t>(i)];
        for ( std::size_t k = 0; k < g.size(); ++k )
        {
            cands.push_back(Cand{g[k], i, static_cast<int>(k)});
        }
    }
    std::sort(cands.begin(), cands.end(),
              []( const Cand& a, const Cand& b ) { return a.gid < b.gid; });
    plan.candidates_received = static_cast<long>(cands.size());
    plan.halo_gids.resize(cands.size());
    plan.halo_x.resize(static_cast<Eigen::Index>(cands.size()), dim);
    plan.recv_perm.assign(static_cast<std::size_t>(nrecv), {});
    for ( int i = 0; i < nrecv; ++i )
    {
        plan.recv_perm[static_cast<std::size_t>(i)].resize(
            static_cast<std::size_t>(
                plan.recv_counts[static_cast<std::size_t>(i)]));
    }
    for ( std::size_t h = 0; h < cands.size(); ++h )
    {
        const Cand& c = cands[h];
        plan.halo_gids[h] = c.gid;
        for ( int d = 0; d < dim; ++d )
        {
            plan.halo_x(static_cast<Eigen::Index>(h), d) =
                rxyz[static_cast<std::size_t>(c.src)]
                    [static_cast<std::size_t>(c.slot)
                     * static_cast<std::size_t>(dim)
                     + static_cast<std::size_t>(d)];
        }
        plan.recv_perm[static_cast<std::size_t>(c.src)]
                      [static_cast<std::size_t>(c.slot)] =
            static_cast<int>(h);
    }
    return plan;
}

/// Move a per-column payload along the plan: `local_vals` is (nloc, m);
/// returns (nhalo, m) in the plan's sorted-halo row order.  Reused for
/// probe inputs, masses, and smoothing fields — any number of times per
/// plan.
inline Eigen::MatrixXd halo_push( const HaloPlan& plan,
                                  const Eigen::Ref<const Eigen::MatrixXd>& local_vals )
{
    if ( local_vals.rows() != plan.nloc )
    {
        throw std::invalid_argument("lgpsf::mpi::halo_push: row count");
    }
    const int m = static_cast<int>(local_vals.cols());
    const int nsend = static_cast<int>(plan.send_ranks.size());
    const int nrecv = static_cast<int>(plan.recv_ranks.size());
    std::vector<std::vector<double>> sbuf(static_cast<std::size_t>(nsend));
    std::vector<std::vector<double>> rbuf(static_cast<std::size_t>(nrecv));
    std::vector<MPI_Request> reqs;
    reqs.reserve(static_cast<std::size_t>(nsend + nrecv));
    for ( int i = 0; i < nrecv; ++i )
    {
        rbuf[static_cast<std::size_t>(i)].resize(
            static_cast<std::size_t>(plan.recv_counts[static_cast<std::size_t>(i)])
            * static_cast<std::size_t>(m));
        reqs.emplace_back();
        MPI_Irecv(rbuf[static_cast<std::size_t>(i)].data(),
                  static_cast<int>(rbuf[static_cast<std::size_t>(i)].size()),
                  MPI_DOUBLE, plan.recv_ranks[static_cast<std::size_t>(i)],
                  73, plan.comm, &reqs.back());
    }
    for ( int i = 0; i < nsend; ++i )
    {
        const std::vector<int>& cols =
            plan.send_cols[static_cast<std::size_t>(i)];
        std::vector<double>& b = sbuf[static_cast<std::size_t>(i)];
        b.resize(cols.size() * static_cast<std::size_t>(m));
        for ( std::size_t k = 0; k < cols.size(); ++k )
        {
            for ( int j = 0; j < m; ++j )
            {
                b[k * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] =
                    local_vals(cols[k], j);
            }
        }
        reqs.emplace_back();
        MPI_Isend(b.data(), static_cast<int>(b.size()), MPI_DOUBLE,
                  plan.send_ranks[static_cast<std::size_t>(i)], 73, plan.comm,
                  &reqs.back());
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(),
                MPI_STATUSES_IGNORE);

    Eigen::MatrixXd halo_vals(plan.nhalo(), m);
    for ( int i = 0; i < nrecv; ++i )
    {
        const std::vector<int>& perm =
            plan.recv_perm[static_cast<std::size_t>(i)];
        for ( std::size_t k = 0; k < perm.size(); ++k )
        {
            for ( int j = 0; j < m; ++j )
            {
                halo_vals(perm[k], j) =
                    rbuf[static_cast<std::size_t>(i)]
                        [k * static_cast<std::size_t>(m)
                         + static_cast<std::size_t>(j)];
            }
        }
    }
    return halo_vals;
}

} // namespace mpi
} // namespace lgpsf

#endif // LGPSF_MPI_HALO_EXCHANGE_HPP
