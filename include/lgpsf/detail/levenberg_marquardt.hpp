#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief A trust-region Levenberg-Marquardt loop, reproducing MINPACK's
/// semantics.
///
/// Generic in the residual and Jacobian: it knows nothing about VarPro, the
/// LG basis, or ellipsoids, so it can be -- and is -- tested on problems whose
/// answers are known independently of anything else in this repository.
///
/// ## What is reproduced, and what is not
///
/// The *semantics* of MINPACK's lmder, not its iterate trajectory: fits here
/// are certified by cross-validation scores and recovery tolerances, never by
/// matching iterate paths. Specifically kept are the trust-region update rule
/// and its constants, `x_scale='jac'` column-norm scaling with MINPACK's
/// monotone-max update, the exact forms of the ftol/xtol/gtol tests, the
/// initial step bound `factor * ||D x0||`, and the rule that a run allowed
/// only one residual evaluation returns x0 untouched.
///
/// ## The trust-region subproblem: an SVD hook step, not lmpar
///
/// Each step solves, exactly,
///
///     min ||J p + f||  subject to  ||D p|| <= delta.
///
/// MINPACK's `lmpar` does this by repeatedly QR-factorizing [J; sqrt(par) D]
/// inside a root-find on the secular equation -- machinery that exists because
/// MINPACK targets large parameter counts. Here the parameter count is at most
/// a few dozen and the residual length is tens, so ONE singular value
/// decomposition of the scaled Jacobian J D^-1 serves every trial value of the
/// Levenberg parameter: with J D^-1 = U S V^T and g = U^T f,
///
///     p(lambda) = -D^-1 V diag( s_i / (s_i^2 + lambda) ) g,
///     ||D p(lambda)||^2 = sum_i ( s_i g_i / (s_i^2 + lambda) )^2,
///
/// so evaluating the constraint at any lambda costs O(P) and its derivative is
/// closed-form. The secular equation is then solved by Newton's method on
/// More's nearly-linear 1/||Dp|| - 1/delta, safeguarded by bisection.
///
/// This is the same exact subproblem solution lmpar computes, at a fraction of
/// the code, and rank deficiency needs no special case: zero singular values
/// contribute nothing to p(lambda) by construction, which is exactly what a
/// problem whose Jacobian is singular at the solution requires.

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include <Eigen/Dense>

namespace lgpsf {

/// How the trust region weights the parameters.
enum class LMScaling
{
    Jacobian,  ///< From the Jacobian's column norms, updated by monotone max.
    Fixed      ///< From a caller-supplied vector, held constant.
};

/// Why the loop stopped.
enum class LMStatus
{
    GradientTolerance,     ///< gtol: the residual is orthogonal to the columns of J.
    CostTolerance,         ///< ftol: the relative cost reduction became negligible.
    StepTolerance,         ///< xtol: the trust region shrank below the step scale.
    CostAndStepTolerance,  ///< both at once.
    MaxEvaluations,        ///< ran out of residual evaluations.
    ToleranceTooSmall      ///< no further progress is representable in double.
};

struct LevenbergMarquardtOptions
{
    double ftol = 1e-8;  ///< On relative cost reduction.
    double xtol = 1e-8;  ///< On relative step size.
    double gtol = 1e-8;  ///< On gradient orthogonality.

    /// Cap on residual evaluations. A value of 1 evaluates once and returns
    /// the starting point unchanged, with zero accepted steps.
    int max_evaluations = 50;

    /// MINPACK's `factor`: the initial trust radius is this times ||D x0||
    /// (or this alone, when that vanishes).
    double initial_bound_factor = 100.0;

    LMScaling scaling = LMScaling::Jacobian;
    Eigen::VectorXd fixed_scale;  ///< Required iff scaling == Fixed.
};

struct LevenbergMarquardtResult
{
    Eigen::VectorXd parameters;
    Eigen::VectorXd residual;
    Eigen::MatrixXd jacobian;  ///< At the returned parameters.
    double cost = 0.0;         ///< 0.5 * ||residual||^2.
    LMStatus status = LMStatus::MaxEvaluations;
    bool success = false;      ///< True for the three tolerance outcomes.
    int num_iterations = 0;    ///< Accepted steps.
    int num_residual_evaluations = 0;
    int num_jacobian_evaluations = 0;
    std::string message;
};

namespace detail {

struct TrustRegionStep
{
    Eigen::VectorXd step;         ///< p
    double parameter = 0.0;       ///< the Levenberg parameter lambda
    double scaled_norm = 0.0;     ///< ||D p||
};

/// Exactly solve the trust-region subproblem
/// min ||J p + f|| subject to ||scale .* p|| <= bound.
///
/// @param jacobian        J, (m, n).
/// @param f               The residual at the current point, (m,).
/// @param scale           Per-parameter scaling D, (n,), all positive.
/// @param bound           Trust-region radius, positive.
/// @param parameter_guess Warm start for the Levenberg parameter, taken from
///                        the previous step as MINPACK does. Affects only how
///                        fast the root-find converges, never where.
/// @return                The step, the Levenberg parameter that produced it,
///                        and ||D p||.
inline TrustRegionStep solve_trust_region(
    const Eigen::MatrixXd& jacobian, const Eigen::VectorXd& f,
    const Eigen::VectorXd& scale, double bound, double parameter_guess )
{
    const Eigen::Index n = jacobian.cols();

    Eigen::MatrixXd scaled = jacobian;
    for ( Eigen::Index j = 0; j < n; ++j )
    {
        scaled.col(j) /= scale(j);
    }
    Eigen::BDCSVD<Eigen::MatrixXd> svd(scaled,
                                       Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& sigma = svd.singularValues();
    const Eigen::VectorXd g = svd.matrixU().transpose() * f;
    const Eigen::Index rank_slots = sigma.size();

    const double sigma_max = ( rank_slots > 0 ) ? sigma(0) : 0.0;
    const double rank_tol = static_cast<double>(std::max(jacobian.rows(), n))
                            * std::numeric_limits<double>::epsilon() * sigma_max;

    // The unconstrained (Gauss-Newton) step, minimum-norm across any null
    // directions. If it already fits inside the trust region, it IS the
    // solution and the Levenberg parameter is zero.
    Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(rank_slots);
    for ( Eigen::Index i = 0; i < rank_slots; ++i )
    {
        if ( sigma(i) > rank_tol )
        {
            coefficients(i) = g(i) / sigma(i);
        }
    }
    Eigen::VectorXd q = -(svd.matrixV() * coefficients);

    TrustRegionStep out;
    if ( q.norm() <= bound || bound <= 0.0 )
    {
        out.parameter = 0.0;
        out.scaled_norm = q.norm();
        out.step = q.array() / scale.array();
        return out;
    }

    // ||D p(lambda)|| is a strictly decreasing function of lambda, so bracket
    // it and run safeguarded Newton on More's nearly-linear form.
    const Eigen::VectorXd weighted = sigma.cwiseProduct(g);
    const auto secular = [&]( double lambda ) {
        Eigen::VectorXd a(rank_slots);
        for ( Eigen::Index i = 0; i < rank_slots; ++i )
        {
            a(i) = weighted(i) / (sigma(i) * sigma(i) + lambda);
        }
        return a;
    };

    double lower = 0.0;
    // ||D p(lambda)|| <= ||sigma .* g|| / lambda, so this upper bound is exact.
    double upper = weighted.norm() / bound;
    double lambda = ( parameter_guess > 0.0 )
                        ? std::min(std::max(parameter_guess, 1e-3 * upper), upper)
                        : upper;

    Eigen::VectorXd a = secular(lambda);
    for ( int iteration = 0; iteration < 50; ++iteration )
    {
        a = secular(lambda);
        const double norm = a.norm();
        if ( std::abs(norm - bound) <= 0.1 * bound )
        {
            break;  // MINPACK's acceptance band on the trust-region boundary
        }
        if ( norm > bound )
        {
            lower = lambda;
        }
        else
        {
            upper = lambda;
        }

        // d/dlambda ||a||^2 = -2 sum a_i^2 / (s_i^2 + lambda), so Newton on
        // 1/||a|| - 1/bound gives this update, which is nearly exact in one
        // step because that function is nearly linear in lambda.
        double curvature = 0.0;
        for ( Eigen::Index i = 0; i < rank_slots; ++i )
        {
            curvature += a(i) * a(i) / (sigma(i) * sigma(i) + lambda);
        }
        double next = lambda;
        if ( curvature > 0.0 && norm > 0.0 )
        {
            next = lambda + (norm / bound - 1.0) * norm * norm / curvature;
        }
        if ( !(next > lower && next < upper) )
        {
            next = ( upper > lower ) ? 0.5 * (lower + upper) : lower;
        }
        if ( next == lambda )
        {
            break;
        }
        lambda = next;
    }

    q = -(svd.matrixV() * a);
    out.parameter = lambda;
    out.scaled_norm = q.norm();
    out.step = q.array() / scale.array();
    return out;
}

inline std::string lm_message( LMStatus status )
{
    switch ( status )
    {
        case LMStatus::GradientTolerance:
            return "converged: the residual is orthogonal to the Jacobian's "
                   "columns to within gtol";
        case LMStatus::CostTolerance:
            return "converged: relative cost reduction is below ftol";
        case LMStatus::StepTolerance:
            return "converged: relative step size is below xtol";
        case LMStatus::CostAndStepTolerance:
            return "converged: both ftol and xtol are satisfied";
        case LMStatus::MaxEvaluations:
            return "stopped: reached the residual-evaluation cap";
        case LMStatus::ToleranceTooSmall:
            return "stopped: the requested tolerance is below what double "
                   "precision can deliver here";
    }
    return "unknown";
}

} // end namespace detail

/// Minimize 0.5 ||residual(x)||^2 over x, unconstrained.
///
/// @param residual Callable: x -> (m,) residual vector.
/// @param jacobian Callable: x -> (m, n) matrix of partial derivatives.
/// @param start    Initial parameters, (n,).
/// @param options  Tolerances, evaluation cap and scaling mode.
/// @return         The best point found, its cost, the stopping status and
///                 the evaluation counts.
///
/// A non-finite residual at a trial point is not an error: it scores worse
/// than any finite cost, so the step is rejected and the trust region
/// contracts. That is what lets a caller signal "no model exists here"
/// without unwinding the search.
template <typename ResidualFn, typename JacobianFn>
LevenbergMarquardtResult levenberg_marquardt(
    ResidualFn&& residual, JacobianFn&& jacobian,
    const Eigen::Ref<const Eigen::VectorXd>& start,
    const LevenbergMarquardtOptions& options = LevenbergMarquardtOptions() )
{
    const double epsilon = std::numeric_limits<double>::epsilon();
    const Eigen::Index n = start.size();

    if ( options.scaling == LMScaling::Fixed && options.fixed_scale.size() != n )
    {
        throw std::invalid_argument(
            "lgpsf::levenberg_marquardt: scaling == Fixed needs a fixed_scale "
            "with one entry per parameter");
    }

    LevenbergMarquardtResult result;
    Eigen::VectorXd x = start;
    Eigen::VectorXd f = residual(x);
    result.num_residual_evaluations = 1;
    double f_norm = f.norm();

    Eigen::VectorXd scale = ( options.scaling == LMScaling::Fixed )
                                ? options.fixed_scale
                                : Eigen::VectorXd::Ones(n);
    double bound = 0.0;
    double x_norm = 0.0;
    double parameter = 0.0;
    bool first_iteration = true;

    const auto finish = [&]( LMStatus status ) {
        result.parameters = x;
        result.residual = f;
        result.cost = 0.5 * f.squaredNorm();
        result.status = status;
        result.success = ( status == LMStatus::GradientTolerance
                           || status == LMStatus::CostTolerance
                           || status == LMStatus::StepTolerance
                           || status == LMStatus::CostAndStepTolerance );
        result.message = detail::lm_message(status);
        return result;
    };

    Eigen::MatrixXd J;
    while ( true )
    {
        // Checked BEFORE any work at the current point, so a cap of one
        // evaluation returns the starting point untouched.
        if ( result.num_residual_evaluations >= options.max_evaluations )
        {
            if ( J.size() == 0 )
            {
                J = jacobian(x);
                ++result.num_jacobian_evaluations;
            }
            result.jacobian = J;
            return finish(LMStatus::MaxEvaluations);
        }

        J = jacobian(x);
        ++result.num_jacobian_evaluations;
        result.jacobian = J;

        Eigen::VectorXd column_norms(n);
        for ( Eigen::Index j = 0; j < n; ++j )
        {
            column_norms(j) = J.col(j).norm();
        }

        if ( options.scaling == LMScaling::Jacobian )
        {
            for ( Eigen::Index j = 0; j < n; ++j )
            {
                // Monotone max, as MINPACK does: the scaling may tighten as
                // the fit proceeds but never loosens, so the trust region
                // cannot be inflated by a temporarily flat direction.
                const double candidate =
                    ( column_norms(j) > 0.0 ) ? column_norms(j) : 1.0;
                scale(j) = first_iteration ? candidate
                                           : std::max(scale(j), candidate);
            }
        }

        if ( first_iteration )
        {
            x_norm = scale.cwiseProduct(x).norm();
            bound = options.initial_bound_factor
                    * ( (x_norm > 0.0) ? x_norm : 1.0 );
        }

        // gtol: the largest cosine between the residual and any Jacobian
        // column. Zero means the reduced gradient vanishes.
        double gradient_norm = 0.0;
        if ( f_norm > 0.0 )
        {
            for ( Eigen::Index j = 0; j < n; ++j )
            {
                if ( column_norms(j) > 0.0 )
                {
                    gradient_norm = std::max(
                        gradient_norm,
                        std::abs(J.col(j).dot(f)) / (column_norms(j) * f_norm));
                }
            }
        }
        if ( gradient_norm <= options.gtol )
        {
            return finish(LMStatus::GradientTolerance);
        }

        bool accepted = false;
        while ( !accepted )
        {
            const detail::TrustRegionStep trial =
                detail::solve_trust_region(J, f, scale, bound, parameter);
            parameter = trial.parameter;
            const double step_norm = trial.scaled_norm;

            if ( first_iteration )
            {
                bound = std::min(bound, step_norm);
            }

            const Eigen::VectorXd x_trial = x + trial.step;
            const Eigen::VectorXd f_trial = residual(x_trial);
            ++result.num_residual_evaluations;
            const double f_norm_trial = f_trial.norm();

            // Comparisons with a non-finite trial norm are all false, which
            // lands on actual_reduction = -1: rejected, region contracts.
            double actual_reduction = -1.0;
            if ( 0.1 * f_norm_trial < f_norm )
            {
                const double ratio_of_norms = f_norm_trial / f_norm;
                actual_reduction = 1.0 - ratio_of_norms * ratio_of_norms;
            }

            const double linear_term = (J * trial.step).norm() / f_norm;
            const double damping_term =
                std::sqrt(parameter) * step_norm / f_norm;
            const double predicted_reduction =
                linear_term * linear_term + 2.0 * damping_term * damping_term;
            const double directional_derivative =
                -(linear_term * linear_term + damping_term * damping_term);
            const double ratio = ( predicted_reduction != 0.0 )
                                     ? actual_reduction / predicted_reduction
                                     : 0.0;

            if ( ratio <= 0.25 )
            {
                double shrink;
                if ( actual_reduction >= 0.0 )
                {
                    shrink = 0.5;
                }
                else
                {
                    shrink = 0.5 * directional_derivative
                             / (directional_derivative + 0.5 * actual_reduction);
                }
                if ( 0.1 * f_norm_trial >= f_norm || shrink < 0.1 )
                {
                    shrink = 0.1;
                }
                bound = shrink * std::min(bound, 10.0 * step_norm);
                parameter /= shrink;
            }
            else if ( parameter == 0.0 || ratio >= 0.75 )
            {
                bound = 2.0 * step_norm;
                parameter *= 0.5;
            }

            if ( ratio >= 1e-4 )
            {
                x = x_trial;
                f = f_trial;
                f_norm = f_norm_trial;
                x_norm = scale.cwiseProduct(x).norm();
                ++result.num_iterations;
                accepted = true;
            }
            first_iteration = false;

            const bool cost_converged = std::abs(actual_reduction) <= options.ftol
                                        && predicted_reduction <= options.ftol
                                        && 0.5 * ratio <= 1.0;
            const bool step_converged = bound <= options.xtol * x_norm;
            if ( cost_converged && step_converged )
            {
                return finish(LMStatus::CostAndStepTolerance);
            }
            if ( cost_converged )
            {
                return finish(LMStatus::CostTolerance);
            }
            if ( step_converged )
            {
                return finish(LMStatus::StepTolerance);
            }

            if ( result.num_residual_evaluations >= options.max_evaluations )
            {
                return finish(LMStatus::MaxEvaluations);
            }
            if ( (std::abs(actual_reduction) <= epsilon
                  && predicted_reduction <= epsilon && 0.5 * ratio <= 1.0)
                 || bound <= epsilon * x_norm )
            {
                return finish(LMStatus::ToleranceTooSmall);
            }
        }
    }
}

} // end namespace lgpsf
