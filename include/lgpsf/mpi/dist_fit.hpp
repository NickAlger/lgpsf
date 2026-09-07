// SPDX-License-Identifier: MIT
#ifndef LGPSF_MPI_DIST_FIT_HPP
#define LGPSF_MPI_DIST_FIT_HPP

/// \file dist_fit.hpp
/// SPMD distributed lgpsf fit: every rank fits ITS rows against its own +
/// halo columns, with the halo delivered by `halo_exchange.hpp`.  Probes
/// are USER INPUTS distributed by layout — V by the column layout, HV by
/// the row layout — and the fit is probe-agnostic; ladder control, probe
/// generation, QC, and (weighted) symmetrization all stay with the
/// caller.  Rectangular-native: rows and columns are independent dof
/// families; the symmetric case is the specialization where each row
/// names its own column via `row_own_gid`.
///
/// Determinism: combined columns are merged in ascending global id, so
/// each row's internal window ordering — hence its design matrix, hence
/// the fitted values — is independent of the rank layout.  With
/// partition-independent probes and exact responses the fitted operator
/// is bitwise identical at every rank count (gate G-L2).

#include "lgpsf/mpi/halo_exchange.hpp"
#include "lgpsf/operator_fit.hpp"
#include "lgpsf/lg_operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

namespace lgpsf {
namespace mpi {

/// The PRE-SCALED window ellipsoids (membership = Mahalanobis <= 1) the
/// fit will use, built with fit_operator's own formula:
///     E_r = { mu_r,  tau_window^2 * capped_covariance(sigma_r, cap) }.
/// Handing THESE to both `halo_plan` and `fit_operator` (as
/// `window_ellipsoids`) makes halo and fit windows agree by construction.
inline std::vector<ellipsoid_tree::Ellipsoid>
make_window_ellipsoids( const Eigen::Ref<const Eigen::MatrixXd>& centers,
                        const std::vector<Eigen::MatrixXd>& sigma,
                        const OperatorFitConfig& config )
{
    const std::size_t n = sigma.size();
    if ( static_cast<std::size_t>(centers.rows()) != n )
    {
        throw std::invalid_argument(
            "lgpsf::mpi::make_window_ellipsoids: centers/sigma size mismatch");
    }
    std::vector<ellipsoid_tree::Ellipsoid> windows(n);
    for ( std::size_t r = 0; r < n; ++r )
    {
        windows[r].mu = centers.row(static_cast<Eigen::Index>(r)).transpose();
        windows[r].Sigma = config.tau_window * config.tau_window
                           * capped_covariance(sigma[r],
                                               config.window_aspect_cap);
    }
    return windows;
}

/// Everything one rank owns.  Columns: points, masses, probe inputs.
/// Rows: centers, masses, a-priori sigma, responses, and (symmetric case)
/// each row's own column gid (-1 = the row has no own column dof: no
/// spike for that row).
struct DistFitInput
{
    // columns (owned by this rank)
    Eigen::MatrixXd   x_local;     ///< (nloc, dim)
    std::vector<long> col_gids;    ///< (nloc) ascending
    Eigen::VectorXd   m2_local;    ///< (nloc) column masses
    Eigen::MatrixXd   V_local;     ///< (nloc, k) probe inputs

    // rows (owned by this rank)
    Eigen::MatrixXd              x_rows;      ///< (nrows, dim) centers
    Eigen::VectorXd              m1_local;    ///< (nrows) row masses
    std::vector<Eigen::MatrixXd> sigma;       ///< (nrows) a-priori covariances
    std::vector<long>            row_own_gid; ///< (nrows) own column gid or -1
    Eigen::MatrixXd              HV_local;    ///< (nrows, k) responses
};

/// What the rank gets back: the fit over (its rows) x (own + halo
/// columns), the assembled UNsymmetrized sparse block in LOCAL column
/// indexing, and the local -> global column map.  Symmetrization (square
/// case) is the caller's job on the globally assembled matrix.
struct DistFitResult
{
    OperatorFit                 fit;
    Eigen::SparseMatrix<double> B_local;    ///< (nrows, ncomb), Symmetrize::None
    std::vector<long>           col_gids;   ///< (ncomb) combined, ascending
    std::vector<int>            own_to_comb; ///< (nloc) owned col -> combined idx
    long                        window_candidates = 0; ///< sum of window sizes
    long                        fit_points_total = 0;  ///< sum of FitDiagnostics::fit_points: what the fits ran on
};

/// The SPMD fit.  `windows` must be the object handed to `halo_plan`
/// (see `make_window_ellipsoids`); `tau_assemble` is the deployment
/// support scale (production: 6).  Collective over plan.comm.
inline DistFitResult dist_fit( const HaloPlan& plan,
                               const DistFitInput& in,
                               const std::vector<ellipsoid_tree::Ellipsoid>& windows,
                               const OperatorFitConfig& config,
                               double tau_assemble )
{
    const int nloc  = static_cast<int>(in.x_local.rows());
    const int nrows = static_cast<int>(in.x_rows.rows());
    const int dim   = static_cast<int>(in.x_local.cols());
    const int k     = static_cast<int>(in.V_local.cols());
    if ( nloc != plan.nloc )
    {
        throw std::invalid_argument("lgpsf::mpi::dist_fit: plan/input nloc");
    }
    if ( static_cast<int>(windows.size()) != nrows
         || static_cast<int>(in.sigma.size()) != nrows
         || static_cast<int>(in.row_own_gid.size()) != nrows
         || in.HV_local.rows() != nrows || in.HV_local.cols() != k )
    {
        throw std::invalid_argument("lgpsf::mpi::dist_fit: row-side sizes");
    }

    // ---- halo payloads: probe inputs + column masses -------------------
    Eigen::MatrixXd Vm(nloc, k + 1);
    Vm.leftCols(k) = in.V_local;
    Vm.col(k)      = in.m2_local;
    const Eigen::MatrixXd Vm_halo = halo_push(plan, Vm);
    const int nhalo = plan.nhalo();
    const int ncomb = nloc + nhalo;

    // ---- merge own + halo columns in ascending gid ---------------------
    DistFitResult out;
    out.col_gids.resize(static_cast<std::size_t>(ncomb));
    out.own_to_comb.resize(static_cast<std::size_t>(nloc));
    Eigen::MatrixXd x_comb(ncomb, dim);
    Eigen::VectorXd m2_comb(ncomb);
    Eigen::MatrixXd V_comb(ncomb, k);
    {
        int io = 0, ih = 0;
        for ( int c = 0; c < ncomb; ++c )
        {
            const bool take_own =
                ih >= nhalo
                || ( io < nloc
                     && in.col_gids[static_cast<std::size_t>(io)]
                            < plan.halo_gids[static_cast<std::size_t>(ih)] );
            if ( take_own )
            {
                out.col_gids[static_cast<std::size_t>(c)] =
                    in.col_gids[static_cast<std::size_t>(io)];
                out.own_to_comb[static_cast<std::size_t>(io)] = c;
                x_comb.row(c)  = in.x_local.row(io);
                m2_comb(c)     = in.m2_local(io);
                V_comb.row(c)  = in.V_local.row(io);
                ++io;
            }
            else
            {
                out.col_gids[static_cast<std::size_t>(c)] =
                    plan.halo_gids[static_cast<std::size_t>(ih)];
                x_comb.row(c)  = plan.halo_x.row(ih);
                m2_comb(c)     = Vm_halo(ih, k);
                V_comb.row(c)  = Vm_halo.row(ih).head(k);
                ++ih;
            }
        }
    }

    // ---- each row's own column, in combined indexing -------------------
    std::vector<int> row_own_col(static_cast<std::size_t>(nrows), -1);
    for ( int r = 0; r < nrows; ++r )
    {
        const long g = in.row_own_gid[static_cast<std::size_t>(r)];
        if ( g < 0 ) { continue; }
        const auto it = std::lower_bound(out.col_gids.begin(),
                                         out.col_gids.end(), g);
        if ( it == out.col_gids.end() || *it != g )
        {
            throw std::invalid_argument(
                "lgpsf::mpi::dist_fit: row_own_gid not among combined "
                "columns (a row's own dof must be a local column)");
        }
        row_own_col[static_cast<std::size_t>(r)] =
            static_cast<int>(it - out.col_gids.begin());
    }

    // ---- the fit (windows explicit => halo/window agreement) -----------
    std::vector<std::optional<ellipsoid_tree::Ellipsoid>> window_opt(
        windows.begin(), windows.end());
    out.fit = fit_operator(x_comb, in.m1_local, m2_comb, V_comb, in.HV_local,
                           in.sigma, config, in.x_rows, in.x_rows, {},
                           window_opt, row_own_col);
    for ( int r = 0; r < nrows; ++r )
    {
        out.window_candidates +=
            static_cast<long>(out.fit.model.row_window(r).size());
        out.fit_points_total +=
            static_cast<long>(out.fit.diagnostics.fit_points(r));
    }

    // ---- assemble the UNsymmetrized local block ------------------------
    out.B_local = assemble_sparse(out.fit.model, tau_assemble,
                                  Symmetrize::None, config.num_threads);
    return out;
}

} // namespace mpi
} // namespace lgpsf

#endif // LGPSF_MPI_DIST_FIT_HPP
