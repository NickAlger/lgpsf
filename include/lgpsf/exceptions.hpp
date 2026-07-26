#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief Exception types with meaning beyond "something went wrong".

#include <stdexcept>
#include <string>

namespace lgpsf {

/// Thrown when a basis cannot be evaluated at a given parameter vector -- not
/// because the caller made a mistake, but because that POINT IN PARAMETER
/// SPACE is unusable. The log-Cholesky diagonal running far enough for
/// exp() to overflow or underflow is the motivating case: an extreme trial
/// step during a fit, not a bug.
///
/// It has its own type precisely so the fitting core can catch it and nothing
/// else. A fit scores an infeasible trial point as "the smooth model
/// contributes nothing" -- the worst finite cost any parameter vector can
/// produce -- so the outer loop rejects the step and backs off instead of
/// crashing. Catching `std::invalid_argument` there instead would swallow
/// genuine caller errors, a mis-shaped array among them, and silently turn
/// them into a plausible-looking bad fit rather than a loud failure.
///
/// So: caller errors stay `std::invalid_argument`; "there is no valid basis
/// here" is this.
class InfeasibleParameters : public std::runtime_error
{
public:
    explicit InfeasibleParameters( const std::string& what )
        : std::runtime_error(what)
    {
    }
};

} // end namespace lgpsf
