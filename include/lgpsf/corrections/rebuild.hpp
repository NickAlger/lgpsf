#pragma once
// SPDX-License-Identifier: MIT
// Part of lgpsf — https://github.com/NickAlger/lgpsf

/// @file
/// @brief `rebuild_at`: re-anchor the contracts at a new build shift with
/// NO new H_d access.
///
/// The usage guidance is: build at the smallest shift you anticipate — the
/// guarantee covers the entire upward sweep for free — and `rebuild_at`
/// covers the unanticipated case. Everything it does is oracle solves and
/// rank-sized algebra over information already in the struct:
///
///   (i)   `make_pd` re-certifies at the new threshold -gamma * a1. The
///         block deflates everything already found, so only the modes in
///         the newly opened window are discovered and flipped; the floor,
///         gamma, and the warning-zone snapshot are renewed.
///   (ii)  if the block holds deflation modes, `deflate_free` re-runs in
///         the new metric from the archived residuals — which are taken
///         against the CURRENT corrected operator, so it estimates the
///         remaining error, not the original one.
///   (iii) archived value-pass pairs fold back in exactly
///         (`fold_value_pairs`): re-orthonormalized in the new metric by
///         linear combination, with H_d of the combinations known from the
///         stored images. Zero new applies — the pairs are secant
///         information, which is why the archive keeps them.

#include <stdexcept>
#include <string>

#include "lgpsf/corrections/deflation.hpp"
#include "lgpsf/corrections/pencil_lanczos.hpp"

namespace lgpsf::corrections {

struct RebuildOptions
{
    FlipMode flip_mode = FlipMode::Flip;
    LanczosOptions lanczos;   ///< budget for the re-certification
    DeflateOptions deflate;   ///< for the re-deflation and the fold
};

struct RebuildReport
{
    FlipReport flip;
    bool redeflated = false;
    DeflateReport deflate;
    bool refolded = false;
    DeflateReport value_fold;
};

/// Re-anchor at `a1` (typically a1 < a0: the downward move the original
/// guarantee cannot cover). If the flip budget runs out, the report's
/// `flip.certified` is false and the deflation steps are skipped — progress
/// is kept, call again. On success the struct's contracts (`a0`, `gamma`,
/// `lambda_floor`, the warning-zone snapshot) are those of a fresh build
/// at a1.
inline RebuildReport rebuild_at( ShiftedOperator& A, double a1,
                                 RebuildOptions opts = {} )
{
    if ( !(a1 > 0.0) )
    {
        throw std::invalid_argument(
            "lgpsf::corrections::rebuild_at: a1 must be positive, got "
            + std::to_string(a1));
    }
    RebuildReport report;
    A.a0 = a1;
    report.flip = make_pd(A, A.gamma, opts.flip_mode, opts.lanczos);
    if ( !report.flip.certified )
    {
        return report;  // resumable, like make_pd itself
    }
    bool has_deflation = false;
    for ( const Provenance tag : A.block.tags )
    {
        has_deflation = has_deflation || tag == Provenance::Deflation;
    }
    if ( has_deflation && A.archive.Z.size() > 0 )
    {
        report.deflate = deflate_free(A, opts.deflate);
        report.redeflated = true;
    }
    if ( A.archive.Q_vp.size() > 0 )
    {
        report.value_fold = fold_value_pairs(A, opts.deflate);
        report.refolded = true;
    }
    return report;
}

} // end namespace lgpsf::corrections
