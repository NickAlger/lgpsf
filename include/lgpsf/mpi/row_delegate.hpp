// SPDX-License-Identifier: MIT
#ifndef LGPSF_MPI_ROW_DELEGATE_HPP
#define LGPSF_MPI_ROW_DELEGATE_HPP

/// \file row_delegate.hpp
/// The MPI half of the fitting-only row redistribution: an implementation of
/// `lgpsf::RowDelegate` that decides where every row is FITTED with
/// `lgpsf::balance_rows` and then moves phase B -- the mode-set ladder and the
/// LM search -- to that rank and the answers back.
/// See `dev/row-balance-plan.md`; this is slice 3's second half.
///
/// Nothing about the operator moves. Phase A (the window, the quadrature) and
/// phase C (the full-window re-score, the guard, the assembly) stay on the
/// row's owner, because the deployed support is the full window and the host
/// never receives it. What crosses is a self-contained package -- the row's
/// fit quadrature and a handful of dense blocks -- and what comes back is two
/// finalists and the counters only the search knows. A row's fit is a pure
/// function of that package, so the result is BITWISE identical to fitting the
/// row at home; only wall time moves.
///
/// **The plan is identical on every rank by construction, not by broadcast.**
/// `assign` allgathers every rank's per-row weights in one canonical order (by
/// rank, then ascending local row) and then every rank runs the same pure
/// `balance_rows` on the same global arrays. No rank sends anyone a host, so
/// there is no window in which two ranks could hold different plans. The cost
/// is O(global rows) memory and arithmetic per rank -- one double per row
/// allgathered, a few megabytes at ten million rows -- which is the same scale
/// as `dist_wsym`'s row-norm allgather and fine through mid scale.
///
/// **What travels** (see `pack_problem` / `pack_candidates` below for the
/// literal layout; everything is `MPI_DOUBLE`, integers included, exactly as
/// `dist_wsym` ships its triplets):
///
///  - out: `fit_size`, `dim`, `num_probes`, `spike_fit`, `num_extra`, the
///    `coarsened` FLAG, `target_mass`, then `center`, `prior_L`, `sigma`, `y`,
///    `x_fit`, `m2_fit`, `z_fit`. That is every member of `RowFitProblem` the
///    search reads and nothing else. `sigma` rides even though `prior_L` is
///    its Cholesky factor: the LM stream is seeded from the raw covariance and
///    refactoring `L L^T` on the host does not return `L` bit for bit.
///  - back: a status word, then either a failure message or the two finalists
///    -- `baseline_index` (an index into the globally identical
///    `baseline_sets`, so it names the same modes anywhere), `theta_baseline`,
///    `baseline_c`, `baseline_s`, `baseline_score`, the searched model's
///    `theta` / `modes` / `c` / `s` / `score` / `released` / `stop_reason` --
///    plus `evaluations`, `candidates`, `max_modes` and the host's phase-B
///    seconds.
///
/// Two things deliberately do NOT travel, and both are RECOMPUTED rather than
/// reconstructed -- each is a pure function of something that did travel,
/// evaluated by the same code on the same doubles: `y_hat`, which the owner
/// refills from `y` and `target_mass`, and the winner's ellipsoid frame, which
/// `LGExpansion::frame()` unpacks from `theta`. The `coarsened` flag travels
/// instead of the config: the host rebuilds the coarsened row's
/// `ProbeFitConfig` from its OWN `RowFitContext::row_config` and
/// `coarsen_eps`, which SPMD makes identical, and which is also the only way
/// the mode policy (a virtual object) can be right on the host.
///
/// **The failure discipline is the whole design of `solve`**
/// (`dev/row-balance-plan.md`, section 9). A rank that throws while its peers
/// are inside an exchange hangs the job rather than crashing it, so:
///
///  - Every exchange count is derived from the DATA -- the buffers actually
///    packed -- never from the assignment. A row that threw in phase A is not
///    in the batch, and a receiver that sized itself from the globally known
///    plan would deadlock on one row failing on one rank.
///  - Every throw inside the migration region becomes a rank-local flag. The
///    flag is `MPI_Allreduce`d ONCE, after the last exchange has completed on
///    every rank, and only then does every rank throw together.
///  - Both directions run counts -> allocate -> "can you take it?" -> payload,
///    so a receiver that cannot allocate refuses and the sender skips it. That
///    is what makes an allocation failure on the RECEIVE side survivable:
///    without the acknowledgement round a receiver would have to hang or
///    abort, its peers having already been told how much is coming. The
///    receive-side byte budget rides that same round rather than adding one.
///  - A foreign throw is per-row and does not fail the job: the message rides
///    home in the candidate slot's `failure`, and the owner fails that row
///    with exactly the semantics of a local throw.
///
/// **The byte cap** is the other half of section 9. The rule balances SECONDS,
/// and a row with many points and a short search is heavy in bytes and light
/// in time, so a plan that is fine for the clock can be an out-of-memory: the
/// senders are few by construction, since being overloaded is what makes them
/// senders. `RowExchangeOptions::bytes_cap` bounds what one rank packs and
/// what one rank receives, and it is enforced in `solve`, on the TRUE size of
/// the packed problem -- never on an estimate, because there is no estimate
/// worth trusting here. Until 2026-09-09 it was enforced in `assign` instead,
/// on the row's WINDOW size, on the argument that coarsening only ever shrinks
/// the quadrature. It does -- by ten to fifty times, and hardest on exactly
/// the widest rows, which are the ones the rule wants to move. Measured on a
/// 192-rank continental run: 6763 of some 6900 planned migrations reverted,
/// for a predicted imbalance of 20.62 against an unbalanced 20.7. The cap saw
/// packages an order of magnitude larger than anything that would have been
/// sent, and so rejected precisely the rows worth moving.
///
///  - A SENDER, holding the packed problems, drops migrations in DECREASING
///    payload, ties by ascending row index, until its own outgoing total is
///    within the cap. Decreasing payload frees the cap in the fewest rows; the
///    tie-break makes the choice a function of the data alone.
///  - A RECEIVER whose incoming total would exceed the cap refuses whole
///    peers, largest first, ties by ascending rank, on the acknowledgement
///    round that is already there for allocation failure -- no new collective.
///    That round now carries three states rather than two: take it, refused
///    for the budget, refused because this rank has already failed. The third
///    exists so that a receiver's `bad_alloc` does not make every one of its
///    senders redo hundreds of LM searches for a job that is about to throw
///    anyway.
///  - A DROPPED ROW IS NOT A FAILURE AND IS NOT LEFT UNSOLVED. Its owner fits
///    it itself, calling `detail::fit_row_candidates` with the context it is
///    already holding, in the same `parallel_for` as the foreign rows it
///    hosts. A row's fit is a pure function of its package, so solving at home
///    is the same answer to the last bit: the cap costs wall time and nothing
///    else.
///
/// The drops are rank-local decisions and, unlike the plan, need no agreement
/// -- a drop moves work, never an answer. `assign` is therefore the makespan
/// plan and nothing else.

#include "lgpsf/operator_fit.hpp"
#include "lgpsf/row_balance.hpp"

#include <Eigen/Dense>

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lgpsf {
namespace mpi {

/// The wire: one `std::vector<double>` per peer, integers carried as exactly
/// representable doubles. Every read is bounds checked, so a truncated or
/// malformed message throws instead of walking off the buffer -- and that
/// throw is caught by the migration region and turned into the collective
/// failure flag rather than into a hang.
namespace wire {

/// A bounds-checked forward reader over one peer's buffer.
class Reader
{
public:
    Reader( const double* begin, const double* end ) : at_(begin), end_(end) {}

    bool done() const { return at_ >= end_; }

    double next()
    {
        if ( at_ >= end_ )
        {
            throw std::runtime_error(
                "lgpsf::mpi: the migration message is truncated");
        }
        return *at_++;
    }

    int next_int()
    {
        const double value = next();
        if ( !(value >= -2147483648.0) || !(value <= 2147483647.0)
             || value != std::floor(value) )
        {
            throw std::runtime_error(
                "lgpsf::mpi: the migration message carries a corrupt integer");
        }
        return static_cast<int>(value);
    }

    /// A non-negative count, additionally checked against what is left, so a
    /// corrupt length cannot ask for a gigabyte allocation.
    int next_count()
    {
        const int value = next_int();
        if ( value < 0 || static_cast<std::ptrdiff_t>(value) > end_ - at_ )
        {
            throw std::runtime_error(
                "lgpsf::mpi: the migration message carries a corrupt length");
        }
        return value;
    }

    void take( double* into, std::size_t count )
    {
        if ( static_cast<std::ptrdiff_t>(count) > end_ - at_ )
        {
            throw std::runtime_error(
                "lgpsf::mpi: the migration message is truncated");
        }
        std::memcpy(into, at_, count * sizeof(double));
        at_ += count;
    }

private:
    const double* at_ = nullptr;
    const double* end_ = nullptr;
};

inline void append( std::vector<double>& buffer, const double* first,
                    std::size_t count )
{
    buffer.insert(buffer.end(), first, first + count);
}

/// Doubles one packed problem occupies: the layout `pack_problem` writes,
/// counted. It is also what the byte cap measures -- on the package in hand,
/// so `fit_size` there is the COARSENED point count that will actually go on
/// the wire and not a bound on it. The per-point term
/// `fit_size * (dim + 1 + num_probes)` is the whole of it at any real size.
inline std::size_t problem_doubles( std::size_t fit_size, std::size_t dim,
                                    std::size_t num_probes )
{
    return 7u + dim + 2u * dim * dim + num_probes
           + fit_size * (dim + 1u + num_probes);
}

/// The bytes one built package will put on the wire. Exact: `pack_problem`
/// writes `problem_doubles` doubles and nothing more.
inline std::size_t problem_bytes( const lgpsf::detail::RowFitProblem& problem )
{
    return sizeof(double)
           * problem_doubles(
               static_cast<std::size_t>(problem.x_fit().rows()),
               static_cast<std::size_t>(problem.x_fit().cols()),
               static_cast<std::size_t>(problem.z_fit().cols()));
}

/// OWNER -> HOST. Exactly what `fit_row_candidates` reads, in the order
/// `problem_doubles` counts it.
inline void pack_problem( const lgpsf::detail::RowFitProblem& problem,
                          std::vector<double>& buffer )
{
    const Eigen::MatrixXd& x_fit = problem.x_fit();
    const Eigen::VectorXd& m2_fit = problem.m2_fit();
    const Eigen::MatrixXd& z_fit = problem.z_fit();
    const std::size_t fit_size = static_cast<std::size_t>(x_fit.rows());
    const std::size_t dim = static_cast<std::size_t>(x_fit.cols());
    const std::size_t num_probes = static_cast<std::size_t>(z_fit.cols());
    if ( m2_fit.size() != x_fit.rows() || z_fit.rows() != x_fit.rows()
         || problem.y.size() != z_fit.cols()
         || problem.center.size() != x_fit.cols()
         || problem.prior_L.rows() != x_fit.cols()
         || problem.prior_L.cols() != x_fit.cols()
         || problem.sigma.rows() != x_fit.cols()
         || problem.sigma.cols() != x_fit.cols() )
    {
        throw std::invalid_argument(
            "lgpsf::mpi::pack_problem: the package's blocks disagree on shape");
    }

    buffer.reserve(buffer.size() + problem_doubles(fit_size, dim, num_probes));
    buffer.push_back(static_cast<double>(fit_size));
    buffer.push_back(static_cast<double>(dim));
    buffer.push_back(static_cast<double>(num_probes));
    buffer.push_back(static_cast<double>(problem.spike_fit));
    buffer.push_back(static_cast<double>(problem.num_extra));
    buffer.push_back(problem.coarsened ? 1.0 : 0.0);
    buffer.push_back(problem.target_mass);
    append(buffer, problem.center.data(), dim);
    append(buffer, problem.prior_L.data(), dim * dim);
    append(buffer, problem.sigma.data(), dim * dim);
    append(buffer, problem.y.data(), num_probes);
    append(buffer, x_fit.data(), fit_size * dim);
    append(buffer, m2_fit.data(), fit_size);
    append(buffer, z_fit.data(), fit_size * num_probes);
}

/// The mirror. @p row_config is the host's own shared config and
/// @p coarse_config its copy with the released-centre resolution rule armed;
/// which one the row gets is the only thing the `coarsened` flag decides, and
/// rebuilding it here rather than shipping it is what keeps the (virtual,
/// un-serializable) mode policy correct on the host.
inline lgpsf::detail::RowFitProblem unpack_problem(
    Reader& reader, const ProbeFitConfig& row_config,
    const ProbeFitConfig& coarse_config, int expected_probes )
{
    const int fit_size = reader.next_int();
    const int dim = reader.next_int();
    const int num_probes = reader.next_int();
    if ( fit_size < 0 || dim <= 0 || num_probes < 0 )
    {
        throw std::runtime_error(
            "lgpsf::mpi::unpack_problem: implausible package shape");
    }
    if ( num_probes != expected_probes )
    {
        // SPMD is the premise of the whole scheme (the host fits against its
        // OWN context); a package from a differently configured fit would be
        // fitted wrong rather than fail, so it is refused here.
        throw std::runtime_error(
            "lgpsf::mpi::unpack_problem: the package carries "
            + std::to_string(num_probes) + " probes but this rank's fit has "
            + std::to_string(expected_probes)
            + " (every rank must run the same fit)");
    }

    lgpsf::detail::RowFitProblem problem;
    problem.spike_fit = reader.next_int();
    problem.num_extra = reader.next_int();
    problem.coarsened = reader.next() != 0.0;
    problem.target_mass = reader.next();
    problem.center.resize(dim);
    reader.take(problem.center.data(), static_cast<std::size_t>(dim));
    problem.prior_L.resize(dim, dim);
    reader.take(problem.prior_L.data(), static_cast<std::size_t>(dim) * dim);
    problem.sigma.resize(dim, dim);
    reader.take(problem.sigma.data(), static_cast<std::size_t>(dim) * dim);
    problem.y.resize(num_probes);
    reader.take(problem.y.data(), static_cast<std::size_t>(num_probes));

    Eigen::MatrixXd x_fit(fit_size, dim);
    reader.take(x_fit.data(), static_cast<std::size_t>(fit_size) * dim);
    Eigen::VectorXd m2_fit(fit_size);
    reader.take(m2_fit.data(), static_cast<std::size_t>(fit_size));
    Eigen::MatrixXd z_fit(fit_size, num_probes);
    reader.take(z_fit.data(), static_cast<std::size_t>(fit_size) * num_probes);
    problem.adopt(std::move(x_fit), std::move(m2_fit), std::move(z_fit),
                  problem.coarsened ? coarse_config : row_config);
    return problem;
}

/// HOST -> OWNER. Everything `select_row_fit` and the owner's bookkeeping
/// read, and nothing else: no `y_hat` (the owner refills it), no ellipsoid
/// frame (`frame()` unpacks it from `theta`), no candidate list.
inline void pack_candidates( const lgpsf::detail::RowFitCandidates& fit,
                             std::vector<double>& buffer )
{
    if ( !fit.solved() )
    {
        buffer.push_back(0.0);
        buffer.push_back(fit.search_seconds);
        buffer.push_back(static_cast<double>(fit.failure.size()));
        for ( char c : fit.failure )
        {
            buffer.push_back(static_cast<double>(
                static_cast<unsigned char>(c)));
        }
        return;
    }

    buffer.push_back(1.0);
    buffer.push_back(fit.search_seconds);
    buffer.push_back(static_cast<double>(fit.baseline_index));
    buffer.push_back(fit.baseline_score);
    buffer.push_back(static_cast<double>(fit.evaluations));
    buffer.push_back(static_cast<double>(fit.candidates));
    buffer.push_back(static_cast<double>(fit.max_modes));
    buffer.push_back(static_cast<double>(fit.theta_baseline.size()));
    buffer.push_back(static_cast<double>(fit.baseline_c.size()));
    buffer.push_back(static_cast<double>(fit.baseline_s.size()));
    append(buffer, fit.theta_baseline.data(),
           static_cast<std::size_t>(fit.theta_baseline.size()));
    append(buffer, fit.baseline_c.data(),
           static_cast<std::size_t>(fit.baseline_c.size()));
    append(buffer, fit.baseline_s.data(),
           static_cast<std::size_t>(fit.baseline_s.size()));

    if ( !fit.searched )
    {
        buffer.push_back(0.0);
        return;
    }
    const ProbeFitResult& searched = *fit.searched;
    buffer.push_back(1.0);
    buffer.push_back(searched.score);
    buffer.push_back(searched.released ? 1.0 : 0.0);
    buffer.push_back(static_cast<double>(static_cast<int>(searched.stop_reason)));
    buffer.push_back(static_cast<double>(searched.model.theta.size()));
    buffer.push_back(static_cast<double>(searched.model.modes.size()));
    buffer.push_back(static_cast<double>(searched.model.c.size()));
    buffer.push_back(static_cast<double>(searched.model.s.size()));
    append(buffer, searched.model.theta.data(),
           static_cast<std::size_t>(searched.model.theta.size()));
    for ( const Mode& mode : searched.model.modes )
    {
        buffer.push_back(static_cast<double>(mode.p));
        buffer.push_back(static_cast<double>(mode.ell));
        buffer.push_back(static_cast<double>(mode.m));
    }
    append(buffer, searched.model.c.data(),
           static_cast<std::size_t>(searched.model.c.size()));
    append(buffer, searched.model.s.data(),
           static_cast<std::size_t>(searched.model.s.size()));
}

/// The mirror, into a slot the delegate protocol left default-constructed.
inline void unpack_candidates( Reader& reader,
                               lgpsf::detail::RowFitCandidates& fit )
{
    const bool solved = reader.next() != 0.0;
    fit.search_seconds = reader.next();
    if ( !solved )
    {
        const int length = reader.next_count();
        std::string message;
        message.reserve(static_cast<std::size_t>(length));
        for ( int i = 0; i < length; ++i )
        {
            message.push_back(static_cast<char>(
                static_cast<unsigned char>(reader.next_int())));
        }
        fit.failure = std::move(message);
        return;  // baseline_index stays -1: the slot is UNSOLVED
    }

    const int baseline_index = reader.next_int();
    const double baseline_score = reader.next();
    fit.evaluations = reader.next_int();
    fit.candidates = reader.next_int();
    fit.max_modes = reader.next_int();
    const int n_theta_baseline = reader.next_count();
    const int n_baseline_c = reader.next_count();
    const int n_baseline_s = reader.next_count();
    fit.theta_baseline.resize(n_theta_baseline);
    reader.take(fit.theta_baseline.data(),
                static_cast<std::size_t>(n_theta_baseline));
    fit.baseline_c.resize(n_baseline_c);
    reader.take(fit.baseline_c.data(), static_cast<std::size_t>(n_baseline_c));
    fit.baseline_s.resize(n_baseline_s);
    reader.take(fit.baseline_s.data(), static_cast<std::size_t>(n_baseline_s));

    if ( reader.next() != 0.0 )
    {
        ProbeFitResult searched;
        searched.score = reader.next();
        searched.released = reader.next() != 0.0;
        const int stop = reader.next_int();
        if ( stop < static_cast<int>(StopReason::Target)
             || stop > static_cast<int>(StopReason::Exhausted) )
        {
            throw std::runtime_error(
                "lgpsf::mpi: the migration message carries an unknown stop "
                "reason");
        }
        searched.stop_reason = static_cast<StopReason>(stop);
        const int n_theta = reader.next_count();
        const int n_modes = reader.next_count();
        const int n_c = reader.next_count();
        const int n_s = reader.next_count();
        searched.model.theta.resize(n_theta);
        reader.take(searched.model.theta.data(),
                    static_cast<std::size_t>(n_theta));
        searched.model.modes.resize(static_cast<std::size_t>(n_modes));
        for ( int i = 0; i < n_modes; ++i )
        {
            Mode& mode = searched.model.modes[static_cast<std::size_t>(i)];
            mode.p = reader.next_int();
            mode.ell = reader.next_int();
            mode.m = reader.next_int();
        }
        searched.model.c.resize(n_c);
        reader.take(searched.model.c.data(), static_cast<std::size_t>(n_c));
        searched.model.s.resize(n_s);
        reader.take(searched.model.s.data(), static_cast<std::size_t>(n_s));
        fit.searched = std::move(searched);
    }

    // Set LAST: `solved()` is `baseline_index >= 0`, so a slot only becomes
    // solved once everything else in it is there.
    fit.baseline_score = baseline_score;
    fit.baseline_index = baseline_index;
}

} // namespace wire

/// How the redistribution is to be decided and bounded. Every field must hold
/// the same value on every rank: the fit is SPMD and the delegate is
/// collective, so a rank that disagrees about whether to migrate does not
/// produce a wrong answer, it hangs the job.
struct RowExchangeOptions
{
    /// The water-filling rule's imbalance tolerance (`dev/row-balance-plan.md`
    /// section 4). 0.1 is a sane value. Note that 0 here means "target a
    /// perfectly even split", NOT "off" -- `dist_fit` is where 0 disables the
    /// scheme, because a caller who wants no migration should not be paying
    /// for an exchange that moves nothing.
    double tolerance = 0.1;

    /// Bytes one rank may pack, and bytes one rank may receive, per fit.
    /// Enforced in `solve` on the TRUE size of each packed problem; a
    /// migration that does not fit is fitted by its owner instead, which
    /// costs wall time and nothing else. See the byte-cap paragraph at the
    /// top of this file.
    ///
    /// **Where 512 MiB comes from.** Two numbers bound it, and the default
    /// sits between them.
    ///
    /// What the traffic is. A package costs
    /// `8 * fit_points * (dim + 1 + num_probes)` bytes, essentially all of it
    /// the quadrature. On the 192-rank continental run the plan moves about
    /// 1.7% of 409545 rows -- the widest ones -- carrying roughly a third of
    /// the job's fit points, which at the top rung is a few hundred megabytes
    /// on the busiest senders; and the senders are few by construction, since
    /// being overloaded is what makes a rank one. The 64 MiB this field
    /// defaulted to before 2026-09-09 was some five times under that even
    /// with a correct size, so it would have bound on exactly the runs it was
    /// meant to leave alone.
    ///
    /// What a rank can afford. Production runs 48 ranks on a 192 GB node,
    /// about 4 GB each. A sender holds the packages phase A built plus the
    /// pack buffer's copy of them; a receiver holds its receive buffer plus
    /// the unpacked problems. So a cap of C is roughly 2C of peak, transient,
    /// and only on the handful of ranks that send or host at all. At 512 MiB
    /// that is 1 GiB -- a quarter of a rank, briefly -- against a requirement
    /// of a few hundred megabytes: room for a bigger mesh or a longer probe
    /// ladder without the cap quietly taking over the schedule, and still far
    /// below what would take a node down.
    ///
    /// It is a backstop, not a policy. It should not bind in the intended
    /// regime, and `RowExchangeStats::rows_capped_sender` /
    /// `::rows_capped_receiver` are there to say when it did.
    std::size_t bytes_cap = static_cast<std::size_t>(512) << 20;

    /// `OperatorFitConfig::coarsen_eps` of the fit this delegate serves. The
    /// host needs it to rebuild a coarsened row's `ProbeFitConfig`, which is
    /// the one thing the `coarsened` flag stands for.
    double coarsen_eps = 1.0;

    /// `OperatorFitConfig::num_threads`, for fitting the foreign rows.
    int num_threads = 0;

    /// (nrows) the previous rung's per-row WEIGHT, or empty on the first rung.
    /// `fit_points x evaluations` is what the plan calls for -- points alone
    /// correlate 0.62 in log with a row's seconds, the product 0.94 -- and
    /// both factors are exact and free from the previous rung's diagnostics.
    /// Any monotone proxy for the row's fit seconds will do, since only the
    /// order and the ratios enter the rule. Empty means rung 1, where the
    /// weight is the row's window size: exact, already computed by the
    /// pre-pass, and the only thing available before a fit has ever run.
    Eigen::VectorXd prev_evaluations;

    /// Replace the assignment rule entirely: given every LOCAL row's window
    /// size, return every local row's host. For diagnostics and for the gate
    /// (which needs a deliberately perverse assignment); production leaves it
    /// empty. Set it on every rank or on none -- a mismatch is detected and
    /// refused rather than left to hang. The byte cap still applies.
    std::function<std::vector<int>(const std::vector<int>&)> assign_override;
};

/// What one call's redistribution actually did. Telemetry: nothing reads it.
/// Every count is THIS RANK's -- reduce them if you want the job's.
struct RowExchangeStats
{
    long rows_migrated = 0;   ///< rows this rank sent away
    long rows_hosted = 0;     ///< foreign rows this rank fitted
    /// Planned migrations the byte cap sent home, where this rank's own
    /// outgoing total did not fit (`rows_capped_sender`) and where the host
    /// refused this rank's packages to stay inside its own incoming budget
    /// (`rows_capped_receiver`). Both were fitted here instead, bit for bit
    /// as if they had never been planned away; `rows_capped` is the sum.
    long rows_capped = 0;
    long rows_capped_sender = 0;
    long rows_capped_receiver = 0;
    long bytes_sent = 0;
    long bytes_received = 0;
    double target = 0.0;               ///< the rule's `T`
    /// The PLAN's max/mean, as `balance_rows` predicted it. It does not know
    /// about rows the cap later sent home; `rows_capped` is how you tell.
    double predicted_imbalance = 1.0;
};

/// The delegate itself. Construct one, hand `delegate()` to `fit_operator`,
/// and keep the object alive for the whole call -- the callbacks capture it.
/// One object may serve any number of fits (one rung each); every call resets
/// its state.
///
/// Collective on `comm`: both callbacks must be entered by every rank. That is
/// what `fit_operator` guarantees -- `solve` is called even with nothing to
/// solve -- provided the assignment is all-or-nothing across ranks, which it
/// is here because `assign` returns full length on every rank.
class RowExchange
{
public:
    RowExchange( MPI_Comm comm, RowExchangeOptions options )
        : comm_(comm), options_(std::move(options))
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }

    RowExchange( const RowExchange& ) = delete;
    RowExchange& operator=( const RowExchange& ) = delete;

    /// The hook `fit_operator` takes. The returned object holds pointers into
    /// this one; do not outlive it, and do not move this one.
    RowDelegate delegate()
    {
        RowDelegate hook;
        hook.self = rank_;
        hook.assign = [this]( const std::vector<int>& window_sizes )
        {
            return this->assign(window_sizes);
        };
        hook.solve =
            [this]( const lgpsf::detail::RowFitContext& context,
                    const std::vector<int>& rows,
                    const std::vector<lgpsf::detail::RowFitProblem>& problems,
                    std::vector<lgpsf::detail::RowFitCandidates>& out )
        {
            this->solve(context, rows, problems, out);
        };
        return hook;
    }

    /// (nrows) which rank FITTED each of this rank's rows: this rank for a row
    /// that stayed, including a row that was assigned away but never left
    /// (gated out, or it threw in phase A, or the byte cap sent it home to be
    /// fitted here). Valid after the fit; empty before it.
    const std::vector<int>& fitted_on_rank() const { return fitted_on_rank_; }

    const RowExchangeStats& stats() const { return stats_; }

private:
    // ---- the assignment ------------------------------------------------
    //
    // Called once, after the window pre-pass and before phase A: the weights
    // are known here and no package exists yet, which is the whole reason the
    // hook is two callbacks (`dev/row-balance-plan.md` section 1, fact 2).
    //
    // It produces the makespan plan and NOTHING ELSE. The byte cap used to
    // live here too, on the window size standing in for a payload that did
    // not exist yet; it was off by the coarsening ratio, which is largest on
    // exactly the rows worth moving, so it is now enforced in `solve` on the
    // real packages. See the byte-cap paragraph at the top of this file.
    std::vector<int> assign( const std::vector<int>& window_sizes )
    {
        const int nrows = static_cast<int>(window_sizes.size());
        fitted_on_rank_.assign(window_sizes.size(),
                               static_cast<int>(rank_));
        host_.assign(window_sizes.size(), rank_);
        stats_ = RowExchangeStats();

        // An override on some ranks and not others would delegate on some and
        // not others, and `solve` -- a collective -- would then be entered by
        // a subset. Decided globally, so the throw below is simultaneous.
        {
            const int mine = options_.assign_override ? 1 : 0;
            int probe[2] = { mine, 1 - mine };
            int seen[2] = { 0, 0 };
            MPI_Allreduce(probe, seen, 2, MPI_INT, MPI_MAX, comm_);
            if ( seen[0] == 1 && seen[1] == 1 )
            {
                throw std::invalid_argument(
                    "lgpsf::mpi::RowExchange: assign_override is set on some "
                    "ranks and not others; it is collective and must be set "
                    "on every rank or on none");
            }
        }

        // Rank-local problems become a FLAG, not a throw: a throw here, before
        // the collectives below, would hang every other rank. The flag is
        // reduced at the end of this function, after the last collective.
        int failed = 0;
        std::string why;

        // ---- the weights ------------------------------------------------
        std::vector<double> mine(static_cast<std::size_t>(nrows), 0.0);
        {
            const bool have_previous = options_.prev_evaluations.size() > 0;
            if ( have_previous
                 && options_.prev_evaluations.size() != nrows )
            {
                failed = 1;
                why = "prev_evaluations has "
                      + std::to_string(options_.prev_evaluations.size())
                      + " entries for " + std::to_string(nrows) + " rows";
            }
            const bool usable = have_previous && !failed;
            for ( int i = 0; i < nrows; ++i )
            {
                const int window = window_sizes[static_cast<std::size_t>(i)];
                // A row that will not be fitted (gated out, or its window
                // could not be formed) weighs nothing and, below, is pinned
                // home: it can never be handed a slot.
                double weight = 0.0;
                if ( window > 0 )
                {
                    weight = usable ? options_.prev_evaluations(i)
                                    : static_cast<double>(window);
                }
                if ( !(weight >= 0.0) || !std::isfinite(weight) )
                {
                    failed = 1;
                    if ( why.empty() )
                    {
                        why = "row " + std::to_string(i) + " has weight "
                              + std::to_string(weight)
                              + ", which is not finite and >= 0";
                    }
                    weight = 0.0;
                }
                mine[static_cast<std::size_t>(i)] = weight;
            }
        }

        // ---- one canonical global order: by rank, then ascending local row -
        std::vector<int> counts(static_cast<std::size_t>(size_), 0);
        MPI_Allgather(&nrows, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_);
        std::vector<int> displs(static_cast<std::size_t>(size_) + 1, 0);
        for ( int r = 0; r < size_; ++r )
        {
            displs[static_cast<std::size_t>(r) + 1] =
                displs[static_cast<std::size_t>(r)]
                + counts[static_cast<std::size_t>(r)];
        }
        const int nglobal = displs[static_cast<std::size_t>(size_)];
        offset_ = displs[static_cast<std::size_t>(rank_)];

        // One double per row: the weight. The window size used to ride along
        // for the cap's benefit and no longer needs to.
        std::vector<double> weights(static_cast<std::size_t>(nglobal), 0.0);
        MPI_Allgatherv(mine.data(), nrows, MPI_DOUBLE, weights.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, comm_);

        std::vector<int> owners(static_cast<std::size_t>(nglobal), 0);
        for ( int r = 0; r < size_; ++r )
        {
            for ( int i = displs[static_cast<std::size_t>(r)];
                  i < displs[static_cast<std::size_t>(r) + 1]; ++i )
            {
                owners[static_cast<std::size_t>(i)] = r;
            }
        }

        // ---- the plan: the same pure function of the same arrays, on every
        //      rank, so no host is ever communicated ------------------------
        const bool overridden = static_cast<bool>(options_.assign_override);
        std::vector<int> hosts;   // global; the rule's
        std::vector<int> local;   // this rank's; an override's
        if ( overridden )
        {
            // Used exactly as returned. It was allgathered while the byte cap
            // lived in `assign` and had to see what every receiver was being
            // sent; the cap is in `solve` now, where the sender and the
            // receiver each decide from what they are actually holding, so
            // nothing here needs the global picture and the collective is
            // gone with the need for it.
            local = options_.assign_override(window_sizes);
            if ( static_cast<int>(local.size()) != nrows )
            {
                failed = 1;
                if ( why.empty() )
                {
                    why = "assign_override returned "
                          + std::to_string(local.size()) + " hosts for "
                          + std::to_string(nrows) + " rows";
                }
                local.assign(static_cast<std::size_t>(nrows), rank_);
            }
            for ( int& host : local )
            {
                if ( host < 0 || host >= size_ )
                {
                    failed = 1;
                    if ( why.empty() )
                    {
                        why = "assign_override returned the out-of-range host "
                              + std::to_string(host);
                    }
                    host = rank_;
                }
            }
        }
        else
        {
            try
            {
                const BalancePlan plan =
                    balance_rows(weights, owners, size_, options_.tolerance);
                hosts = plan.host;
                stats_.target = plan.target;
                stats_.predicted_imbalance = plan.predicted_imbalance;
            }
            catch ( const std::exception& error )
            {
                // The rule is pure and its input is identical everywhere, so
                // this fires on every rank at once -- but it is flagged rather
                // than thrown, so the reduction below stays collective.
                failed = 1;
                if ( why.empty() ) { why = error.what(); }
                hosts = owners;
            }
        }

        // ---- my slice ----------------------------------------------------
        for ( int i = 0; i < nrows; ++i )
        {
            const std::size_t r = static_cast<std::size_t>(i);
            host_[r] = overridden
                           ? local[r]
                           : hosts[static_cast<std::size_t>(offset_ + i)];
            // A row nobody will fit is pinned home whatever the plan says, so
            // it can never be handed a slot. `fit_operator` also refuses to
            // delegate it; belt and braces, and it keeps `fitted_on_rank`
            // honest.
            if ( window_sizes[r] == 0 ) { host_[r] = rank_; }
        }

        // ---- the flag, after the last collective -------------------------
        int any_failed = 0;
        MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm_);
        if ( any_failed )
        {
            throw std::invalid_argument(
                "lgpsf::mpi::RowExchange: the row assignment failed on at "
                "least one rank (this rank: "
                + ( why.empty() ? std::string("no local failure") : why )
                + "). No row was exchanged.");
        }
        return host_;
    }

    // ---- the exchange --------------------------------------------------
    //
    // THE MIGRATION REGION. Every early return, every throw and every count in
    // here is load-bearing; read section 9 of the plan before changing any of
    // it. The invariants: counts come from the buffers, not from the plan; a
    // local failure never throws until the last exchange has completed on
    // every rank; and a receiver that cannot allocate refuses rather than
    // vanishing from an exchange its peers have already sized.
    //
    // It is also where the byte cap lives, because this is the first place a
    // rank holds the thing being sent rather than a guess at its size. A row
    // the cap keeps home is fitted here, in the same `parallel_for` as the
    // foreign rows, and is bit for bit the row it would have been anywhere.
    void solve( const lgpsf::detail::RowFitContext& context,
                const std::vector<int>& rows,
                const std::vector<lgpsf::detail::RowFitProblem>& problems,
                std::vector<lgpsf::detail::RowFitCandidates>& out )
    {
        const std::size_t peers = static_cast<std::size_t>(size_);
        int failed = 0;
        std::string why;

        // The host's copy of the coarsened row's config: rebuilt from ITS own
        // context, which is what the `coarsened` flag exists for.
        ProbeFitConfig coarse_config = context.row_config;
        coarse_config.resolution_eps = options_.coarsen_eps;

        // ---- the sender's byte budget, then pack -------------------------
        //
        // The packages exist now, so their size is a fact rather than an
        // estimate. Drop in DECREASING payload -- fewest rows to get under the
        // cap -- and break ties by ascending row, so the set dropped is a
        // function of the data and of nothing else. A dropped row is not a
        // failure: it goes in `at_home` and is fitted below.
        //
        // Both the budget and the packing are inside one try, because a
        // `bad_alloc` in either has to become the flag rather than a throw
        // through a collective. If either fails this rank sends nothing and
        // fits nothing extra: the job is going to throw at the reduction.
        std::vector<char> at_home(problems.size(), 0);
        std::vector<int> local_slots;   // slots this rank will fit itself
        std::vector<std::vector<double>> send_buffer(peers);
        std::vector<std::vector<int>> send_slots(peers);
        try
        {
            // Reserved HERE, inside the guard: every later `push_back` into it
            // happens outside a try, and a reallocation there would be a throw
            // in the middle of the exchange sequence.
            local_slots.reserve(problems.size());
            std::vector<std::size_t> bytes(problems.size(), 0u);
            std::size_t outgoing = 0u;
            for ( std::size_t k = 0; k < problems.size(); ++k )
            {
                bytes[k] = wire::problem_bytes(problems[k]);
                outgoing += bytes[k];
            }
            if ( outgoing > options_.bytes_cap )
            {
                std::vector<int> order(problems.size());
                for ( std::size_t k = 0; k < order.size(); ++k )
                {
                    order[k] = static_cast<int>(k);
                }
                std::sort(order.begin(), order.end(),
                          [&bytes, &rows]( int a, int b )
                          {
                              const std::size_t ba =
                                  bytes[static_cast<std::size_t>(a)];
                              const std::size_t bb =
                                  bytes[static_cast<std::size_t>(b)];
                              if ( ba != bb ) { return ba > bb; }
                              return rows[static_cast<std::size_t>(a)]
                                     < rows[static_cast<std::size_t>(b)];
                          });
                for ( int k : order )
                {
                    if ( outgoing <= options_.bytes_cap ) { break; }
                    at_home[static_cast<std::size_t>(k)] = 1;
                    outgoing -= bytes[static_cast<std::size_t>(k)];
                }
            }

            for ( std::size_t k = 0; k < problems.size(); ++k )
            {
                if ( at_home[k] )
                {
                    local_slots.push_back(static_cast<int>(k));
                    ++stats_.rows_capped_sender;
                    continue;
                }
                const int row = rows[k];
                const int host =
                    ( row >= 0 && row < static_cast<int>(host_.size()) )
                        ? host_[static_cast<std::size_t>(row)]
                        : rank_;
                if ( host == rank_ || host < 0 || host >= size_ )
                {
                    throw std::logic_error(
                        "lgpsf::mpi::RowExchange: row " + std::to_string(row)
                        + " was delegated but has no foreign host");
                }
                wire::pack_problem(problems[k],
                                   send_buffer[static_cast<std::size_t>(host)]);
                send_slots[static_cast<std::size_t>(host)].push_back(
                    static_cast<int>(k));
            }
        }
        catch ( const std::exception& error )
        {
            failed = 1;
            why = error.what();
            drop_everything(send_buffer, send_slots, local_slots, stats_);
        }
        catch ( ... )
        {
            failed = 1;
            why = "an unknown exception while packing";
            drop_everything(send_buffer, send_slots, local_slots, stats_);
        }

        // ---- counts, DERIVED FROM THE BUFFERS ----------------------------
        //
        // Not from the assignment. A row that threw in phase A is not in
        // `problems`, and a rank that failed to pack sends nothing at all; a
        // receiver sizing itself from the plan would wait forever for either.
        std::vector<int> send_counts(2 * peers, 0), recv_counts(2 * peers, 0);
        for ( std::size_t r = 0; r < peers; ++r )
        {
            // MPI counts are `int`. The byte cap keeps a message far below
            // that, but the cap is a knob, so an oversized buffer becomes a
            // clean collective failure rather than a truncated send.
            if ( !fits_a_count(send_buffer[r].size()) )
            {
                failed = 1;
                if ( why.empty() )
                {
                    why = "the package for rank " + std::to_string(r)
                          + " is too large for one message ("
                          + std::to_string(send_buffer[r].size())
                          + " doubles); lower balance_bytes_cap";
                }
                std::vector<double>().swap(send_buffer[r]);
                std::vector<int>().swap(send_slots[r]);
            }
            send_counts[2 * r] = static_cast<int>(send_slots[r].size());
            send_counts[2 * r + 1] = static_cast<int>(send_buffer[r].size());
        }
        MPI_Alltoall(send_counts.data(), 2, MPI_INT, recv_counts.data(), 2,
                     MPI_INT, comm_);

        // ---- the receiver's byte budget, allocate, then say whether we can
        //      take it -----------------------------------------------------
        //
        // The acknowledgement is what makes a receive-side allocation failure
        // survivable: the counts are already out, so a receiver that simply
        // threw would leave its senders blocked in an Isend nobody will match.
        // The INCOMING BYTE BUDGET rides that same round rather than adding
        // one of its own -- a receiver over the cap refuses whole peers,
        // largest first and ties by ascending rank, and each refused sender
        // then fits those rows itself.
        //
        // So the round carries three states, because the two refusals mean
        // different things to a sender:
        //    1  take it;
        //    0  refused for the budget -- fit those rows at home;
        //   -1  refused because this rank has already failed -- do not bother,
        //       the reduction at the end of this function throws on every rank
        //       and an LM search run for it would be minutes wasted.
        // Everything that reads these values therefore tests `> 0`, never
        // truthiness.
        std::vector<std::vector<double>> recv_buffer(peers);
        std::vector<int> accept(peers, 1), peer_accepts(peers, 0);
        try
        {
            std::size_t incoming = 0u;
            std::vector<int> order;
            order.reserve(peers);
            for ( std::size_t r = 0; r < peers; ++r )
            {
                if ( recv_counts[2 * r + 1] > 0 )
                {
                    incoming += sizeof(double)
                                * static_cast<std::size_t>(
                                    recv_counts[2 * r + 1]);
                    order.push_back(static_cast<int>(r));
                }
            }
            if ( incoming > options_.bytes_cap )
            {
                std::sort(order.begin(), order.end(),
                          [&recv_counts]( int a, int b )
                          {
                              const int ca = recv_counts[
                                  2 * static_cast<std::size_t>(a) + 1];
                              const int cb = recv_counts[
                                  2 * static_cast<std::size_t>(b) + 1];
                              if ( ca != cb ) { return ca > cb; }
                              return a < b;
                          });
                for ( int r : order )
                {
                    if ( incoming <= options_.bytes_cap ) { break; }
                    accept[static_cast<std::size_t>(r)] = 0;
                    incoming -= sizeof(double)
                                * static_cast<std::size_t>(
                                    recv_counts[2 * static_cast<std::size_t>(r)
                                                + 1]);
                }
            }
            for ( std::size_t r = 0; r < peers; ++r )
            {
                if ( accept[r] > 0 && recv_counts[2 * r + 1] > 0 )
                {
                    recv_buffer[r].resize(
                        static_cast<std::size_t>(recv_counts[2 * r + 1]));
                }
            }
        }
        catch ( ... )
        {
            failed = 1;
            if ( why.empty() )
            {
                why = "could not allocate the receive buffers for the "
                      "migrated rows";
            }
            clear_buffers(recv_buffer);
            accept.assign(peers, -1);
        }
        MPI_Alltoall(accept.data(), 1, MPI_INT, peer_accepts.data(), 1, MPI_INT,
                     comm_);

        // ---- the packages go out -----------------------------------------
        exchange(send_buffer, send_counts, recv_buffer, recv_counts,
                 peer_accepts, accept, /*tag=*/8301, /*stride=*/2);
        for ( std::size_t r = 0; r < peers; ++r )
        {
            if ( send_slots[r].empty() ) { continue; }
            if ( peer_accepts[r] > 0 )
            {
                for ( int slot : send_slots[r] )
                {
                    fitted_on_rank_[static_cast<std::size_t>(rows[
                        static_cast<std::size_t>(slot)])] =
                        static_cast<int>(r);
                }
                stats_.rows_migrated +=
                    static_cast<long>(send_slots[r].size());
                stats_.bytes_sent += static_cast<long>(
                    sizeof(double) * send_buffer[r].size());
            }
            else if ( peer_accepts[r] == 0 )
            {
                // Refused for the host's INCOMING budget. The rows are still
                // ours and are fitted below, exactly as if they had never been
                // planned away; `fitted_on_rank_` already says this rank. The
                // slot list is dropped with them, so a reply that cannot exist
                // could not overwrite an answer we are about to compute.
                for ( int slot : send_slots[r] )
                {
                    local_slots.push_back(slot);   // reserved above: no throw
                }
                stats_.rows_capped_receiver +=
                    static_cast<long>(send_slots[r].size());
                std::vector<int>().swap(send_slots[r]);
            }
            // peer_accepts[r] < 0: that rank has already failed and every rank
            // throws at the reduction. The rows come home unsolved, which is
            // what a failed row is.
        }
        // The packages are on the wire; the copies are dead weight from here,
        // and the peak is what the cap is about.
        clear_buffers(send_buffer);

        // ---- fit the foreign rows ----------------------------------------
        std::vector<std::vector<lgpsf::detail::RowFitProblem>> foreign(peers);
        std::vector<std::vector<lgpsf::detail::RowFitCandidates>> answer(peers);
        for ( std::size_t r = 0; r < peers; ++r )
        {
            const int expected = ( accept[r] > 0 ) ? recv_counts[2 * r] : 0;
            if ( expected <= 0 ) { continue; }
            // Pre-filled with a failure, so a row we cannot unpack still gets
            // an answer with a reason instead of silence.
            answer[r].resize(static_cast<std::size_t>(expected));
            for ( lgpsf::detail::RowFitCandidates& slot : answer[r] )
            {
                slot.failure = "the host rank " + std::to_string(rank_)
                               + " could not unpack this row's package";
            }
            try
            {
                wire::Reader reader(recv_buffer[r].data(),
                                    recv_buffer[r].data()
                                        + recv_buffer[r].size());
                while ( !reader.done()
                        && static_cast<int>(foreign[r].size()) < expected )
                {
                    foreign[r].push_back(wire::unpack_problem(
                        reader, context.row_config, coarse_config,
                        static_cast<int>(context.num_probes)));
                }
            }
            catch ( const std::exception& error )
            {
                failed = 1;
                if ( why.empty() ) { why = error.what(); }
            }
            catch ( ... )
            {
                failed = 1;
                if ( why.empty() ) { why = "an unknown exception while "
                                           "unpacking a package"; }
            }
            std::vector<double>().swap(recv_buffer[r]);
            stats_.rows_hosted += static_cast<long>(foreign[r].size());
            stats_.bytes_received += static_cast<long>(
                sizeof(double)
                * static_cast<std::size_t>(std::max(recv_counts[2 * r + 1], 0)));
        }

        // One flat job list, so the foreign rows of every source share one
        // parallel_for -- the same way a rank's own rows do. The rows the byte
        // cap kept at home (source -1, index a slot in `problems` / `out`) join
        // it: this rank has to fit them and there is no reason to do it in a
        // second pass.
        std::vector<std::pair<int, int>> jobs;
        for ( std::size_t r = 0; r < peers; ++r )
        {
            for ( std::size_t i = 0; i < foreign[r].size(); ++i )
            {
                jobs.emplace_back(static_cast<int>(r), static_cast<int>(i));
            }
        }
        for ( int slot : local_slots )
        {
            jobs.emplace_back(-1, slot);
        }
        try
        {
            ellipsoid_tree::detail::parallel_for(
                0, static_cast<std::ptrdiff_t>(jobs.size()),
                [&]( std::ptrdiff_t begin, std::ptrdiff_t end )
                {
                    for ( std::ptrdiff_t j = begin; j < end; ++j )
                    {
                        const int from =
                            jobs[static_cast<std::size_t>(j)].first;
                        const std::size_t index = static_cast<std::size_t>(
                            jobs[static_cast<std::size_t>(j)].second);
                        if ( from < 0 )
                        {
                            // A row the byte cap kept home. Same function, same
                            // context, the same package that would have gone on
                            // the wire, so the answer is the foreign host's to
                            // the last bit -- that identity is what makes the
                            // cap a wall-time knob rather than a correctness
                            // one. `y_hat` is kept rather than dropped: nothing
                            // has to travel, so there is nothing to refill.
                            lgpsf::detail::RowFitCandidates& home = out[index];
                            const auto home_start =
                                std::chrono::steady_clock::now();
                            try
                            {
                                home = lgpsf::detail::fit_row_candidates(
                                    problems[index], context);
                            }
                            catch ( const std::exception& error )
                            {
                                // A per-row throw here is the row throwing at
                                // home, so it reads exactly as it would with no
                                // delegate at all: no "[fitted on rank]".
                                home = lgpsf::detail::RowFitCandidates();
                                home.failure = error.what();
                            }
                            catch ( ... )
                            {
                                home = lgpsf::detail::RowFitCandidates();
                                home.failure = "an unknown exception";
                            }
                            home.search_seconds =
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now()
                                    - home_start).count();
                            continue;
                        }
                        const std::size_t source =
                            static_cast<std::size_t>(from);
                        lgpsf::detail::RowFitCandidates& slot =
                            answer[source][index];
                        const auto start = std::chrono::steady_clock::now();
                        try
                        {
                            // The receiving rank's OWN context: SPMD makes it
                            // identical, and the mode policy is a virtual
                            // object that could not have travelled.
                            slot = lgpsf::detail::fit_row_candidates(
                                foreign[source][index], context);
                            // The owner refills it, bit for bit, from data it
                            // already has. Not worth `num_probes` doubles.
                            slot.y_hat = Eigen::VectorXd();
                        }
                        catch ( const std::exception& error )
                        {
                            // A per-row failure is NOT a job failure: the
                            // message rides home and the owner fails that row
                            // exactly as a local throw would.
                            slot = lgpsf::detail::RowFitCandidates();
                            slot.failure = "[fitted on rank "
                                           + std::to_string(rank_) + "] "
                                           + error.what();
                        }
                        catch ( ... )
                        {
                            slot = lgpsf::detail::RowFitCandidates();
                            slot.failure = "[fitted on rank "
                                           + std::to_string(rank_)
                                           + "] an unknown exception";
                        }
                        slot.search_seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start)
                                .count();
                    }
                },
                options_.num_threads);
        }
        catch ( const std::exception& error )
        {
            // Not a per-row throw (those are caught above) -- a thread could
            // not be started, or something equally structural.
            failed = 1;
            if ( why.empty() ) { why = error.what(); }
        }
        catch ( ... )
        {
            failed = 1;
            if ( why.empty() )
            {
                why = "an unknown exception while fitting foreign rows";
            }
        }
        foreign.clear();

        // ---- the answers go back -----------------------------------------
        std::vector<std::vector<double>> reply_buffer(peers);
        std::vector<int> reply_send(2 * peers, 0), reply_recv_counts(2 * peers, 0);
        try
        {
            for ( std::size_t r = 0; r < peers; ++r )
            {
                for ( const lgpsf::detail::RowFitCandidates& slot : answer[r] )
                {
                    wire::pack_candidates(slot, reply_buffer[r]);
                }
            }
        }
        catch ( ... )
        {
            failed = 1;
            if ( why.empty() )
            {
                why = "could not pack the answers to the migrated rows";
            }
            clear_buffers(reply_buffer);
        }
        for ( std::size_t r = 0; r < peers; ++r )
        {
            if ( !fits_a_count(reply_buffer[r].size()) )
            {
                failed = 1;
                if ( why.empty() )
                {
                    why = "the answers for rank " + std::to_string(r)
                          + " are too large for one message";
                }
                std::vector<double>().swap(reply_buffer[r]);
            }
            // [0] records, [1] doubles -- both from the BUFFER, so a reply
            // that could not be packed advertises nothing and the owner waits
            // for nothing.
            reply_send[2 * r] =
                reply_buffer[r].empty() ? 0 : static_cast<int>(answer[r].size());
            reply_send[2 * r + 1] = static_cast<int>(reply_buffer[r].size());
        }
        answer.clear();
        MPI_Alltoall(reply_send.data(), 2, MPI_INT, reply_recv_counts.data(), 2,
                     MPI_INT, comm_);

        std::vector<std::vector<double>> reply_recv(peers);
        std::vector<int> accept_reply(peers, 1), peer_accepts_reply(peers, 0);
        try
        {
            for ( std::size_t r = 0; r < peers; ++r )
            {
                if ( reply_recv_counts[2 * r + 1] > 0 )
                {
                    reply_recv[r].resize(static_cast<std::size_t>(
                        reply_recv_counts[2 * r + 1]));
                }
            }
        }
        catch ( ... )
        {
            failed = 1;
            if ( why.empty() )
            {
                why = "could not allocate the buffers for the answers";
            }
            clear_buffers(reply_recv);
            accept_reply.assign(peers, 0);
        }
        MPI_Alltoall(accept_reply.data(), 1, MPI_INT,
                     peer_accepts_reply.data(), 1, MPI_INT, comm_);

        exchange(reply_buffer, reply_send, reply_recv, reply_recv_counts,
                 peer_accepts_reply, accept_reply, /*tag=*/8302, /*stride=*/2);
        clear_buffers(reply_buffer);

        // ---- unpack into the caller's slots ------------------------------
        //
        // Positional, in the order the rows were packed. A record that is not
        // there leaves its slot UNSET, which `fit_operator` turns into a failed
        // row -- never a silent zero -- and which only happens when some rank
        // set the flag below anyway.
        for ( std::size_t r = 0; r < peers; ++r )
        {
            if ( reply_recv[r].empty() ) { continue; }
            try
            {
                wire::Reader reader(reply_recv[r].data(),
                                    reply_recv[r].data()
                                        + reply_recv[r].size());
                for ( int slot : send_slots[r] )
                {
                    if ( reader.done() ) { break; }
                    wire::unpack_candidates(
                        reader, out[static_cast<std::size_t>(slot)]);
                }
            }
            catch ( const std::exception& error )
            {
                failed = 1;
                if ( why.empty() ) { why = error.what(); }
            }
            catch ( ... )
            {
                failed = 1;
                if ( why.empty() )
                {
                    why = "an unknown exception while unpacking an answer";
                }
            }
        }

        // What the cap cost this rank, in rows. Rank-local, like every other
        // count here: reduce them if you want the job's.
        stats_.rows_capped =
            stats_.rows_capped_sender + stats_.rows_capped_receiver;

        // ---- the flag, reduced AFTER the last exchange -------------------
        //
        // This is the line section 9 is about. Every rank has now completed
        // every collective in this function, so every rank can throw -- and
        // they all do, together, or none of them does.
        int any_failed = 0;
        MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, comm_);
        if ( any_failed )
        {
            throw std::runtime_error(
                "lgpsf::mpi::RowExchange: the row migration failed on at least "
                "one rank (this rank: "
                + ( why.empty() ? std::string("no local failure") : why )
                + ")");
        }
    }

    /// One `Irecv`/`Isend`/`Waitall` round. Sends only where the peer said it
    /// had room, receives only where we did -- the two conditions are the same
    /// acknowledgement seen from the two ends, so no message is ever posted
    /// without its match. The acknowledgement is a THREE-state int on the
    /// package round (see `solve`), so the test is `> 0`, not truthiness.
    void exchange( const std::vector<std::vector<double>>& send_buffer,
                   const std::vector<int>& send_counts,
                   std::vector<std::vector<double>>& recv_buffer,
                   const std::vector<int>& recv_counts,
                   const std::vector<int>& peer_accepts,
                   const std::vector<int>& accept, int tag, int stride )
    {
        const std::size_t peers = static_cast<std::size_t>(size_);
        std::vector<MPI_Request> requests;
        for ( std::size_t r = 0; r < peers; ++r )
        {
            const int count = recv_counts[stride * r + 1];
            if ( count > 0 && accept[r] > 0 && !recv_buffer[r].empty() )
            {
                requests.emplace_back();
                MPI_Irecv(recv_buffer[r].data(), count, MPI_DOUBLE,
                          static_cast<int>(r), tag, comm_, &requests.back());
            }
        }
        for ( std::size_t r = 0; r < peers; ++r )
        {
            const int count = send_counts[stride * r + 1];
            if ( count > 0 && peer_accepts[r] > 0 )
            {
                requests.emplace_back();
                MPI_Isend(const_cast<double*>(send_buffer[r].data()), count,
                          MPI_DOUBLE, static_cast<int>(r), tag, comm_,
                          &requests.back());
            }
        }
        if ( !requests.empty() )
        {
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                        MPI_STATUSES_IGNORE);
        }
    }

    /// Whether a buffer length fits the `int` every MPI count is.
    static bool fits_a_count( std::size_t length )
    {
        return length <= static_cast<std::size_t>(
                   std::numeric_limits<int>::max());
    }

    static void clear_buffers( std::vector<std::vector<double>>& buffers )
    {
        for ( std::vector<double>& buffer : buffers )
        {
            std::vector<double>().swap(buffer);
        }
    }

    /// A rank whose packing failed sends nothing AND fits nothing extra. The
    /// reduction at the end of `solve` throws on every rank, so an LM search
    /// run for a capped row here would be minutes of work for an answer that
    /// is never read.
    static void drop_everything( std::vector<std::vector<double>>& send_buffer,
                                 std::vector<std::vector<int>>& send_slots,
                                 std::vector<int>& local_slots,
                                 RowExchangeStats& stats )
    {
        clear_buffers(send_buffer);
        for ( std::vector<int>& slots : send_slots )
        {
            std::vector<int>().swap(slots);
        }
        std::vector<int>().swap(local_slots);
        stats.rows_capped_sender = 0;
    }

    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 1;
    int offset_ = 0;                    ///< my first row in the global order
    RowExchangeOptions options_;
    std::vector<int> host_;             ///< (nrows) where each of my rows fits
    std::vector<int> fitted_on_rank_;   ///< (nrows) where each one DID fit
    RowExchangeStats stats_;
};

} // namespace mpi
} // namespace lgpsf

#endif // LGPSF_MPI_ROW_DELEGATE_HPP
