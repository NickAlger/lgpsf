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
///
/// `balance_tolerance` (off by default) adds a fitting-only redistribution:
/// the LM search of the rows on the busiest ranks runs wherever there is
/// capacity and the answers come home, leaving the halo, the assembly and
/// every number in the result untouched — see `mpi/row_delegate.hpp` and
/// `dev/row-balance-plan.md`.  Wall time is the only thing it moves, and the
/// gate demands the same bitwise agreement with the redistribution on.

#include "lgpsf/mpi/halo_exchange.hpp"
#include "lgpsf/mpi/row_delegate.hpp"
#include "lgpsf/operator_fit.hpp"
#include "lgpsf/lg_operator.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
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

    // ---- fitting-only row redistribution (dev/row-balance-plan.md) ------
    //
    // Wall time only: the fit of a row is a pure function of the package that
    // travels with it, so the operator is BITWISE identical whether or not a
    // row was fitted somewhere else.  Every field here is collective -- it
    // must hold the same value on every rank, and `dist_fit` checks the ones
    // that decide whether the exchange happens at all, because a rank that
    // disagrees hangs the job rather than answering differently.

    /// The water-filling rule's imbalance tolerance.  0 (the default) is OFF:
    /// no delegate is built and the fit runs exactly as it did before this
    /// existed, bit for bit.  (One four-double reduction happens either way,
    /// to catch the ranks disagreeing about whether to redistribute at all --
    /// see the body.)  0.1 is a sane value when it is on; note that a
    /// tolerance of 0 would otherwise MEAN "target a perfectly even split",
    /// which is the meaning "off" displaces.
    double balance_tolerance = 0.0;

    /// (nrows) the previous rung's per-row WEIGHT for the rule, or empty on
    /// the first rung (where the window size is used, being the only thing
    /// available before any fit has run).  The plan's weight is
    /// `fit_points x evaluations` -- `diagnostics.fit_points(r) *
    /// diagnostics.evaluations(r)` from the previous rung's result, which
    /// correlates 0.94 in log with the row's seconds against 0.62 for points
    /// alone.  Both factors are exact and free there; only the evaluation
    /// count is a prediction, and a misprediction costs wall time, never
    /// correctness.  Any monotone proxy for the row's fit seconds is
    /// accepted, since only the order and the ratios enter the rule.
    Eigen::VectorXd prev_evaluations;

    /// Bytes one rank may pack, and bytes one rank may receive, per fit.
    /// The rule balances SECONDS and a row heavy in points but short in
    /// search is heavy in bytes and light in time, so a plan that is fine for
    /// the clock can still be an out-of-memory on the few ranks that are
    /// senders by construction.  See `RowExchangeOptions::bytes_cap`.
    std::size_t balance_bytes_cap =
        RowExchangeOptions().bytes_cap;

    /// Replace the assignment rule (diagnostics and the MPI gate, which needs
    /// a deliberately perverse assignment; production leaves it empty).
    /// Given every local row's window size, return every local row's host.
    /// Setting it turns the redistribution on even at `balance_tolerance` 0,
    /// and it must be set on every rank or on none.
    std::function<std::vector<int>(const std::vector<int>&)> balance_assign;
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
    long                        evaluations_total = 0; ///< sum of FitDiagnostics::evaluations
    double                      work_total = 0.0;      ///< sum of FitDiagnostics::work
    double                      work_max_row = 0.0;    ///< max over rows of FitDiagnostics::work
    double                      seconds_total = 0.0;   ///< sum of FitDiagnostics::row_seconds (telemetry, not deterministic)
    double                      seconds_max_row = 0.0; ///< max over rows of FitDiagnostics::row_seconds (telemetry)

    /// (nrows) which rank FITTED each of this rank's rows -- this rank for
    /// every row unless `balance_tolerance` moved it.  Always filled, so a
    /// consumer's report and dump column do not have to know whether the
    /// redistribution was on.  A row assigned away but never sent (gated out,
    /// or it threw in phase A, or the byte cap reverted it) reads as this
    /// rank, which is where it was in fact fitted.
    ///
    /// Note what `seconds_total` then is: the sum over the rows this rank
    /// OWNS, each including the search wherever it ran.  Per-rank wall time is
    /// the sum over the rows a rank FITTED, which is a different set.
    std::vector<int>            fitted_on_rank;

    /// How the redistribution went, when it was on.  Telemetry.
    RowExchangeStats            balance;
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

    // ---- who fits which row --------------------------------------------
    //
    // Off by default: `delegate` stays null and `fit_operator` takes exactly
    // the path it took before the hook existed, down to the bit.  The one
    // reduction below runs anyway, and deliberately -- the decision to
    // delegate must be UNANIMOUS, because a rank that opts out while its peers
    // opt in does not give a different answer, it hangs the job, and "off
    // here, on there" is precisely the disagreement that has to be caught.
    // Four doubles, once per rung, against a collective the caller is already
    // paying for.
    int rank = 0;
    MPI_Comm_rank(plan.comm, &rank);
    out.fitted_on_rank.assign(static_cast<std::size_t>(nrows), rank);

    const bool wants_balance =
        in.balance_tolerance > 0.0 || static_cast<bool>(in.balance_assign);
    {
        const double tolerance = in.balance_tolerance;
        double probe[4] = { tolerance, -tolerance,
                            wants_balance ? 1.0 : 0.0,
                            wants_balance ? 0.0 : 1.0 };
        double seen[4] = { 0.0, 0.0, 0.0, 0.0 };
        MPI_Allreduce(probe, seen, 4, MPI_DOUBLE, MPI_MAX, plan.comm);
        if ( seen[2] > 0.0 && seen[3] > 0.0 )
        {
            throw std::invalid_argument(
                "lgpsf::mpi::dist_fit: the row redistribution is requested on "
                "some ranks and not others (balance_tolerance / "
                "balance_assign); it is collective and must agree");
        }
        if ( wants_balance && seen[0] != -seen[1] )
        {
            throw std::invalid_argument(
                "lgpsf::mpi::dist_fit: balance_tolerance differs across ranks; "
                "it is collective and must agree");
        }
    }

    std::unique_ptr<RowExchange> exchange;
    RowDelegate delegate;
    const RowDelegate* delegate_ptr = nullptr;
    if ( wants_balance )
    {
        RowExchangeOptions options;
        options.tolerance = in.balance_tolerance;
        options.bytes_cap = in.balance_bytes_cap;
        options.coarsen_eps = config.coarsen_eps;
        options.num_threads = config.num_threads;
        options.dim = dim;
        options.num_probes = k;
        options.prev_evaluations = in.prev_evaluations;
        options.assign_override = in.balance_assign;
        exchange.reset(new RowExchange(plan.comm, std::move(options)));
        delegate = exchange->delegate();
        delegate_ptr = &delegate;
    }

    // ---- the fit (windows explicit => halo/window agreement) -----------
    std::vector<std::optional<ellipsoid_tree::Ellipsoid>> window_opt(
        windows.begin(), windows.end());
    out.fit = fit_operator(x_comb, in.m1_local, m2_comb, V_comb, in.HV_local,
                           in.sigma, config, in.x_rows, in.x_rows, {},
                           window_opt, row_own_col, delegate_ptr);
    if ( exchange )
    {
        out.fitted_on_rank = exchange->fitted_on_rank();
        out.balance = exchange->stats();
    }
    for ( int r = 0; r < nrows; ++r )
    {
        out.window_candidates +=
            static_cast<long>(out.fit.model.row_window(r).size());
        out.fit_points_total +=
            static_cast<long>(out.fit.diagnostics.fit_points(r));
        out.evaluations_total +=
            static_cast<long>(out.fit.diagnostics.evaluations(r));
        out.work_total += out.fit.diagnostics.work(r);
        out.work_max_row = std::max(out.work_max_row,
                                    out.fit.diagnostics.work(r));
        out.seconds_total += out.fit.diagnostics.row_seconds(r);
        out.seconds_max_row = std::max(out.seconds_max_row,
                                       out.fit.diagnostics.row_seconds(r));
    }

    // ---- assemble the UNsymmetrized local block ------------------------
    out.B_local = assemble_sparse(out.fit.model, tau_assemble,
                                  Symmetrize::None, config.num_threads);
    return out;
}

} // namespace mpi
} // namespace lgpsf

#endif // LGPSF_MPI_DIST_FIT_HPP
