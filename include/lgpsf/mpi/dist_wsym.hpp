// SPDX-License-Identifier: MIT
#ifndef LGPSF_MPI_DIST_WSYM_HPP
#define LGPSF_MPI_DIST_WSYM_HPP

/// \file dist_wsym.hpp
/// Distributed `weighted_symmetrize` (square dof context): every rank
/// holds its rows of A (local row indexing, GLOBAL column ids, ascending
/// row ownership blocks across ranks) and gets back its rows of
///
///   B_ij = (w_i^2 A_ij + w_j^2 A_ji) / (w_i^2 + w_j^2),
///   w_i^2 = 1 / (||A_i,:||^2 + (0.01 * median_r ||A_r,:||)^2),
///
/// exactly as `lgpsf::detail::weighted_symmetrize` computes it serially —
/// BITWISE: row energies are row-local; the median comes from an
/// allgathered row-norm vector sorted globally; and each (i, j) entry has
/// exactly TWO contributions (w_i^2 A_ij and w_j^2 A_ji), whose sum is
/// order-independent in IEEE arithmetic.  Zeros-kept convention: stored
/// zero entries contribute their (zero) products and their PATTERN, like
/// the serial code.
///
/// Note: the row-norm allgather is O(N) per rank — fine through mid
/// scale; at ~10^7 rows swap in a distributed median (localized change).

#include <Eigen/Sparse>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lgpsf {
namespace mpi {

/// One stored entry of the distributed matrix, global indexing.
struct GlobalTriplet
{
    long   row;
    long   col;
    double value;
};

/// `rows_local`: this rank's rows of A as (global row, global col, value)
/// triplets (any order; all rows in [row_start, row_start + nrows)).
/// `row_ranges`: (size+1) global row offsets per rank (ascending).
/// Returns this rank's rows of B, sorted by (row, col).
inline std::vector<GlobalTriplet> dist_weighted_symmetrize(
    MPI_Comm comm, const std::vector<GlobalTriplet>& rows_local,
    const std::vector<long>& row_ranges )
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if ( static_cast<int>(row_ranges.size()) != size + 1 )
    {
        throw std::invalid_argument(
            "lgpsf::mpi::dist_weighted_symmetrize: row_ranges size");
    }
    const long row_start = row_ranges[static_cast<std::size_t>(rank)];
    const long nrows =
        row_ranges[static_cast<std::size_t>(rank) + 1] - row_start;
    const long nglob = row_ranges[static_cast<std::size_t>(size)];

    // ---- local row energies -> global norms -> median -> w2 ------------
    std::vector<double> energy(static_cast<std::size_t>(nrows), 0.0);
    for ( const GlobalTriplet& t : rows_local )
    {
        energy[static_cast<std::size_t>(t.row - row_start)] +=
            t.value * t.value;
    }
    std::vector<double> energy_glob(static_cast<std::size_t>(nglob), 0.0);
    {
        std::vector<int> counts(static_cast<std::size_t>(size)),
            displs(static_cast<std::size_t>(size));
        for ( int r = 0; r < size; ++r )
        {
            counts[static_cast<std::size_t>(r)] = static_cast<int>(
                row_ranges[static_cast<std::size_t>(r) + 1]
                - row_ranges[static_cast<std::size_t>(r)]);
            displs[static_cast<std::size_t>(r)] =
                static_cast<int>(row_ranges[static_cast<std::size_t>(r)]);
        }
        MPI_Allgatherv(energy.data(), static_cast<int>(nrows), MPI_DOUBLE,
                       energy_glob.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, comm);
    }
    std::vector<double> norms;
    for ( long i = 0; i < nglob; ++i )
    {
        if ( energy_glob[static_cast<std::size_t>(i)] > 0.0 )
        {
            norms.push_back(std::sqrt(energy_glob[static_cast<std::size_t>(i)]));
        }
    }
    if ( norms.empty() )
    {
        std::vector<GlobalTriplet> out = rows_local;
        std::sort(out.begin(), out.end(),
                  []( const GlobalTriplet& a, const GlobalTriplet& b )
                  { return a.row != b.row ? a.row < b.row : a.col < b.col; });
        return out;
    }
    std::sort(norms.begin(), norms.end());
    const std::size_t half = norms.size() / 2;
    const double median = ( norms.size() % 2 == 1 )
                              ? norms[half]
                              : 0.5 * (norms[half - 1] + norms[half]);
    const double floor2 = (1e-2 * median) * (1e-2 * median);
    std::vector<double> w2(static_cast<std::size_t>(nglob));
    for ( long i = 0; i < nglob; ++i )
    {
        w2[static_cast<std::size_t>(i)] =
            1.0 / (energy_glob[static_cast<std::size_t>(i)] + floor2);
    }

    // ---- products: mine stay; transposed ones go to the row owner ------
    const auto owner = [&]( long grow )
    {
        const auto it = std::upper_bound(row_ranges.begin(), row_ranges.end(),
                                         grow);
        return static_cast<int>(it - row_ranges.begin()) - 1;
    };
    std::vector<GlobalTriplet> products;   // (i, j, w2_i A_ij) staying here
    products.reserve(rows_local.size());
    std::vector<std::vector<GlobalTriplet>> outbox(
        static_cast<std::size_t>(size));
    for ( const GlobalTriplet& t : rows_local )
    {
        const double product =
            w2[static_cast<std::size_t>(t.row)] * t.value;
        products.push_back(GlobalTriplet{t.row, t.col, product});
        const GlobalTriplet transposed{t.col, t.row, product};
        const int dst = owner(t.col);
        if ( dst == rank )
        {
            products.push_back(transposed);
        }
        else
        {
            outbox[static_cast<std::size_t>(dst)].push_back(transposed);
        }
    }
    // sparse exchange of transposed products
    std::vector<int> scnt(static_cast<std::size_t>(size), 0),
        rcnt(static_cast<std::size_t>(size), 0);
    for ( int r = 0; r < size; ++r )
    {
        scnt[static_cast<std::size_t>(r)] =
            static_cast<int>(outbox[static_cast<std::size_t>(r)].size());
    }
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, comm);
    {
        std::vector<MPI_Request> reqs;
        std::vector<std::vector<double>> rbuf(static_cast<std::size_t>(size));
        std::vector<std::vector<double>> sbuf(static_cast<std::size_t>(size));
        for ( int r = 0; r < size; ++r )
        {
            if ( rcnt[static_cast<std::size_t>(r)] > 0 )
            {
                rbuf[static_cast<std::size_t>(r)].resize(
                    static_cast<std::size_t>(rcnt[static_cast<std::size_t>(r)])
                    * 3);
                reqs.emplace_back();
                MPI_Irecv(rbuf[static_cast<std::size_t>(r)].data(),
                          rcnt[static_cast<std::size_t>(r)] * 3, MPI_DOUBLE,
                          r, 81, comm, &reqs.back());
            }
        }
        for ( int r = 0; r < size; ++r )
        {
            const std::vector<GlobalTriplet>& ob =
                outbox[static_cast<std::size_t>(r)];
            if ( ob.empty() ) { continue; }
            std::vector<double>& b = sbuf[static_cast<std::size_t>(r)];
            b.resize(ob.size() * 3);
            for ( std::size_t k = 0; k < ob.size(); ++k )
            {
                b[3 * k + 0] = static_cast<double>(ob[k].row);
                b[3 * k + 1] = static_cast<double>(ob[k].col);
                b[3 * k + 2] = ob[k].value;
            }
            reqs.emplace_back();
            MPI_Isend(b.data(), static_cast<int>(b.size()), MPI_DOUBLE, r,
                      81, comm, &reqs.back());
        }
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(),
                    MPI_STATUSES_IGNORE);
        for ( int r = 0; r < size; ++r )
        {
            const std::vector<double>& b = rbuf[static_cast<std::size_t>(r)];
            for ( std::size_t k = 0; k + 2 < b.size(); k += 3 )
            {
                products.push_back(GlobalTriplet{
                    static_cast<long>(b[k]), static_cast<long>(b[k + 1]),
                    b[k + 2]});
            }
        }
    }

    // ---- merge the (<= 2) contributions per entry, divide --------------
    std::sort(products.begin(), products.end(),
              []( const GlobalTriplet& a, const GlobalTriplet& b )
              { return a.row != b.row ? a.row < b.row : a.col < b.col; });
    std::vector<GlobalTriplet> out;
    out.reserve(products.size());
    for ( std::size_t k = 0; k < products.size(); )
    {
        const long i = products[k].row, j = products[k].col;
        double numer = products[k].value;
        ++k;
        if ( k < products.size() && products[k].row == i
             && products[k].col == j )
        {
            numer += products[k].value;   // two IEEE terms: order-free
            ++k;
        }
        out.push_back(GlobalTriplet{
            i, j,
            numer
                / (w2[static_cast<std::size_t>(i)]
                   + w2[static_cast<std::size_t>(j)])});
    }
    return out;
}

} // namespace mpi
} // namespace lgpsf

#endif // LGPSF_MPI_DIST_WSYM_HPP
