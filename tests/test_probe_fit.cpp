// SPDX-License-Identifier: MIT
//
// Checks on the probe-fit engine: the cross-validation machinery, the
// selection rule and its guards, the early-stopping certificates, and an
// end-to-end recovery of a target the data was built from.
//
// All self-contained: nothing here is compared against a stored reference, so
// the suite cannot drift out of step with the code it tests.

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <vector>

#include "doctest/doctest.h"

#include "lgpsf/probe_fit.hpp"
#include "test_helpers.hpp"

using lgpsf::CandidateFit;
using lgpsf::CrossValidationSplit;
using lgpsf::CvFold;
using lgpsf::FixedSet;
using lgpsf::Mode;
using lgpsf::MuMode;
using lgpsf::MuPolicy;
using lgpsf::ProbeFitConfig;
using lgpsf::ProbeFitResult;
using lgpsf::ShellLadder;
using lgpsf::StopReason;
using lgpsf::WhitenedBasis;
using lgpsf::fit_from_probes;
using lgpsf::jitter_table;
using lgpsf::kfold_split;
using lgpsf::linear_cv_score;
using lgpsf::modes_up_to_level;
using lgpsf::theta_hat_size;
using lgpsf::unpack_theta;
using lgpsf::whiten_data;
using lgpsf::whiten_extra;
using lgpsf::whiten_probes;

namespace {

/// A synthetic target on a grid, built exactly from a known ellipsoid and
/// known coefficients, so the answer is available for comparison.
struct Target
{
    Eigen::MatrixXd x;        // (K, 2)
    Eigen::VectorXd m2_diag;  // (K,)
    Eigen::MatrixXd z;        // (K, k)
    Eigen::VectorXd y;        // (k,)
    Eigen::VectorXd mu0;      // (2,)
    Eigen::VectorXd theta_hat_true;
    Eigen::VectorXd c_true;
    Eigen::VectorXd s_true;
    std::vector<Mode> modes;
    int spike_index = 0;
    double mass = 1.0;
};

Target make_target( std::mt19937& gen, int level = 2, int num_probes = 40,
                    double noise = 0.0, bool oversized = false )
{
    Target target;
    const int per_side = 15;
    const int count = per_side * per_side;
    target.x.resize(count, 2);
    int row = 0;
    for ( int i = 0; i < per_side; ++i )
    {
        for ( int j = 0; j < per_side; ++j )
        {
            target.x(row, 0) = -1.0 + 2.0 * i / (per_side - 1);
            target.x(row, 1) = -1.0 + 2.0 * j / (per_side - 1);
            ++row;
        }
    }
    target.m2_diag =
        test_helpers::uniform_points(count, 1, gen, 0.5, 2.0).col(0);

    target.mu0 = Eigen::VectorXd::Zero(2);
    // the batch point closest to the origin carries the spike
    target.spike_index = (per_side / 2) * per_side + per_side / 2;
    target.mass = 1.3;

    // a modest, slightly anisotropic ellipsoid, well inside the batch
    target.theta_hat_true.resize(theta_hat_size(2, MuMode::Pinned));
    if ( oversized )
    {
        // Deliberately larger than the window, which the caller declares to be
        // conservative -- so any fit that finds it must be ruled inadmissible.
        target.theta_hat_true << std::log(3.0), std::log(2.5), 0.0;
    }
    else
    {
        target.theta_hat_true << std::log(0.30), std::log(0.22), 0.05;
    }

    target.modes = modes_up_to_level(2, level);
    const WhitenedBasis basis(target.x, target.mass, target.m2_diag, target.modes,
                              target.mu0, MuMode::Pinned);
    const Eigen::MatrixXd phi_hat =
        basis(target.theta_hat_true).values();  // (K, m), already whitened

    target.c_true = test_helpers::randn_points(
                        static_cast<int>(target.modes.size()), 1, gen).col(0);
    target.s_true = test_helpers::randn_points(1, 1, gen).col(0);

    target.z = test_helpers::randn_points(count, num_probes, gen);

    // Build y through the whitened identity, then unwhiten: y_hat = c . (phi_hat^T
    // z_hat) + s . (e_hat^T z_hat), and y = sqrt(mass) * y_hat.
    Eigen::MatrixXd extra = Eigen::MatrixXd::Zero(count, 1);
    extra(target.spike_index, 0) = 1.0;
    const Eigen::MatrixXd e_hat = whiten_extra(extra, target.mass, target.m2_diag);
    const Eigen::MatrixXd z_hat = whiten_probes(target.z, target.m2_diag);
    const Eigen::VectorXd y_hat =
        (phi_hat * target.c_true).transpose() * z_hat
        + (e_hat * target.s_true).transpose() * z_hat;
    target.y = std::sqrt(target.mass) * y_hat;

    if ( noise > 0.0 )
    {
        target.y += noise * target.y.norm()
                    * test_helpers::randn_points(num_probes, 1, gen).col(0)
                    / std::sqrt(static_cast<double>(num_probes));
    }
    return target;
}

ProbeFitConfig basic_config( std::shared_ptr<const lgpsf::ModePolicy> policy )
{
    ProbeFitConfig config;
    config.mode_policy = std::move(policy);
    config.target_score = std::nullopt;  // walk the whole grid unless asked otherwise
    return config;
}

} // namespace

TEST_CASE("the round-robin split partitions the probes and is deterministic")
{
    for ( int probes : {12, 37, 100} )
    {
        const CrossValidationSplit split = kfold_split(probes, 5);
        REQUIRE(split.size() == 5u);

        std::multiset<int> seen;
        for ( const CvFold& fold : split )
        {
            // train and validation are complementary and disjoint
            CHECK(fold.train.size() + fold.validation.size() == probes);
            std::set<int> validation(fold.validation.data(),
                                     fold.validation.data() + fold.validation.size());
            for ( Eigen::Index i = 0; i < fold.train.size(); ++i )
            {
                CHECK(validation.find(fold.train(i)) == validation.end());
            }
            for ( Eigen::Index i = 0; i < fold.validation.size(); ++i )
            {
                seen.insert(fold.validation(i));
            }
        }
        // every probe is validated exactly once
        CHECK(static_cast<int>(seen.size()) == probes);
        for ( int i = 0; i < probes; ++i )
        {
            CHECK(seen.count(i) == 1u);
        }
        // folds are balanced to within one
        std::size_t smallest = probes, largest = 0;
        for ( const CvFold& fold : split )
        {
            smallest = std::min(smallest, static_cast<std::size_t>(fold.validation.size()));
            largest = std::max(largest, static_cast<std::size_t>(fold.validation.size()));
        }
        CHECK(largest - smallest <= 1u);
    }

    // deterministic, and the fold count is clamped so folds stay well posed
    CHECK(kfold_split(20, 5)[0].validation == kfold_split(20, 5)[0].validation);
    CHECK(kfold_split(6, 5).size() == 3u);   // min(5, k/2)
    CHECK(kfold_split(3, 5).size() == 2u);   // never below 2
}

TEST_CASE("a seeded split is still a partition, and a different one")
{
    const CrossValidationSplit plain = kfold_split(40, 5);
    const CrossValidationSplit shuffled = kfold_split(40, 5, 12345u);
    REQUIRE(shuffled.size() == plain.size());

    std::multiset<int> seen;
    for ( const CvFold& fold : shuffled )
    {
        CHECK(fold.train.size() + fold.validation.size() == 40);
        for ( Eigen::Index i = 0; i < fold.validation.size(); ++i )
        {
            seen.insert(fold.validation(i));
        }
    }
    CHECK(seen.size() == 40u);
    for ( int i = 0; i < 40; ++i )
    {
        CHECK(seen.count(i) == 1u);
    }

    bool differs = false;
    for ( std::size_t f = 0; f < plain.size(); ++f )
    {
        differs = differs || plain[f].validation != shuffled[f].validation;
    }
    CHECK(differs);
    // and the same seed gives the same split
    CHECK(kfold_split(40, 5, 12345u)[2].validation == shuffled[2].validation);
}

TEST_CASE("the jitter table is deterministic, bounded, and shaped as asked")
{
    const std::vector<Eigen::VectorXd> table = jitter_table(5, 8);
    REQUIRE(table.size() == 8u);
    for ( const Eigen::VectorXd& row : table )
    {
        CHECK(row.size() == 5);
        CHECK(row.cwiseAbs().maxCoeff() <= 0.05);
    }
    CHECK(jitter_table(5, 8)[3] == table[3]);
    CHECK(jitter_table(5, 8, 99u)[3] != table[3]);
    // successive levels perturb differently, which is the point
    CHECK(table[0] != table[1]);
}

TEST_CASE("the cross-validation score is zero on an exact model and rises off it")
{
    std::mt19937 gen(0);
    const Target target = make_target(gen);
    const Eigen::MatrixXd z_hat = whiten_probes(target.z, target.m2_diag);
    const Eigen::VectorXd y_hat = whiten_data(target.y, target.mass);
    Eigen::MatrixXd extra = Eigen::MatrixXd::Zero(target.x.rows(), 1);
    extra(target.spike_index, 0) = 1.0;
    const Eigen::MatrixXd e_hat = whiten_extra(extra, target.mass, target.m2_diag);

    const WhitenedBasis basis(target.x, target.mass, target.m2_diag, target.modes,
                              target.mu0, MuMode::Pinned);
    const CrossValidationSplit split = kfold_split(static_cast<int>(target.y.size()), 5);

    const double exact =
        linear_cv_score(z_hat, y_hat, basis, target.theta_hat_true, e_hat, split);
    MESSAGE("cv score at the true parameters: " << exact);
    CHECK(exact < 1e-8);

    // a wrong ellipsoid scores worse -- this is the quantity selection uses
    Eigen::VectorXd wrong = target.theta_hat_true;
    wrong(0) += 1.5;
    CHECK(linear_cv_score(z_hat, y_hat, basis, wrong, e_hat, split) > 1e-3);

    // an unusable parameter vector scores as infinitely bad rather than throwing
    Eigen::VectorXd runaway = target.theta_hat_true;
    runaway(0) = -900.0;
    CHECK(std::isinf(linear_cv_score(z_hat, y_hat, basis, runaway, e_hat, split)));
}

TEST_CASE("a fit recovers the target it was built from")
{
    std::mt19937 gen(1);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    MESSAGE("recovery: " << result.candidates.size() << " candidates, score "
                         << result.score << ", winner '"
                         << result.candidates[result.winner].label << "'");
    CHECK(result.score < 1e-6);
    CHECK((result.model.c - target.c_true).cwiseAbs().maxCoeff() < 1e-5);
    CHECK((result.model.s - target.s_true).cwiseAbs().maxCoeff() < 1e-5);

    // the recovered ellipsoid is the one the data was built from
    const lgpsf::EllipsoidFrame truth =
        lgpsf::unpack_theta_hat(target.theta_hat_true, target.mu0, MuMode::Pinned);
    CHECK((result.model.frame().mu - truth.mu).cwiseAbs().maxCoeff() < 1e-6);
    CHECK((result.model.frame().L * result.model.frame().L.transpose() - truth.L * truth.L.transpose())
              .cwiseAbs().maxCoeff() < 1e-6);
}

TEST_CASE("the returned parameters decode on their own")
{
    // The public encoding is absolute, so a caller holding only `theta` can
    // recover the ellipsoid -- no mu0, no mode. That is what lets the operator
    // layer hand back one flat array every row decodes uniformly.
    std::mt19937 gen(2);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    const lgpsf::EllipsoidFrame decoded = unpack_theta(result.model.theta);
    CHECK((decoded.mu - result.model.frame().mu).cwiseAbs().maxCoeff() < 1e-12);
    CHECK((decoded.L - result.model.frame().L).cwiseAbs().maxCoeff() < 1e-12);
    CHECK(result.model.theta.size() == lgpsf::theta_size(2));
}

TEST_CASE("the default pins the center")
{
    std::mt19937 gen(3);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));
    CHECK(config.mu == MuPolicy::Pinned);

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);
    CHECK_FALSE(result.released);
    CHECK((result.model.frame().mu - target.mu0).cwiseAbs().maxCoeff() == 0.0);
    for ( const CandidateFit& candidate : result.candidates )
    {
        CHECK_FALSE(candidate.released);
    }
}

TEST_CASE("releasing the center is available on request")
{
    std::mt19937 gen(4);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));
    config.mu = MuPolicy::PinnedThenRelease;

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    int released = 0;
    for ( const CandidateFit& candidate : result.candidates )
    {
        released += candidate.released ? 1 : 0;
    }
    MESSAGE("release stage produced " << released << " candidates; winner released: "
                                      << result.released);
    CHECK(released > 0);
    // the tie-break prefers pinned, so an exactly-recovered target must not
    // ship a released fit merely for tying
    CHECK_FALSE(result.released);

    // fitting the center from the start also works, and its centers stay put
    config.mu = MuPolicy::Free;
    const ProbeFitResult free_fit =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);
    CHECK((free_fit.model.frame().mu - target.mu0).cwiseAbs().maxCoeff() < 1e-4);
}

TEST_CASE("admissibility is exactly the window-containment rule")
{
    // Every candidate's flag must agree with the rule it stands for, on both
    // the shape bound and (when the center moves) the center bound.
    std::mt19937 gen(5);
    const Target target = make_target(gen);
    const double radius = lgpsf::window_radius(target.x, target.mu0);

    ProbeFitConfig config =
        basic_config(std::make_shared<ShellLadder>(std::vector<int>{0, 2}));
    config.mu = MuPolicy::PinnedThenRelease;

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    for ( const CandidateFit& candidate : result.candidates )
    {
        if ( candidate.axes.size() == 0 )
        {
            continue;  // a start point the basis could not be evaluated at
        }
        bool expected = candidate.axes.maxCoeff() <= radius;
        if ( candidate.released )
        {
            expected = expected && candidate.model.theta.head(2).norm() <= radius;
        }
        CHECK(candidate.admissible == expected);
    }
}

TEST_CASE("a fit that outgrows the window is ruled inadmissible")
{
    // The guard, actually firing. The target is built from an ellipsoid three
    // times the window radius, which the conservativeness of the window says
    // cannot be the truth -- so however well such a fit scores, it must be
    // excluded. Few-equation validation cannot reject these on score alone.
    std::mt19937 gen(13);
    const Target target = make_target(gen, 2, 40, 0.0, true);
    const double radius = lgpsf::window_radius(target.x, target.mu0);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    int inadmissible = 0;
    for ( const CandidateFit& candidate : result.candidates )
    {
        inadmissible += candidate.admissible ? 0 : 1;
    }
    MESSAGE("oversized target: " << inadmissible << " of " << result.candidates.size()
                                 << " candidates ruled inadmissible (window radius "
                                 << radius << ")");
    CHECK(inadmissible > 0);

    // and the engine still returns a model rather than failing: when NOTHING is
    // admissible the rule falls back to the whole pool, so a caller always gets
    // an answer plus the evidence to judge it
    CHECK(result.winner >= 0);
    CHECK(std::isfinite(result.score));
}

TEST_CASE("the counting rule skips levels the probes cannot support")
{
    std::mt19937 gen(6);
    const Target target = make_target(gen, 2, 24);
    ProbeFitConfig config =
        basic_config(std::make_shared<ShellLadder>(std::vector<int>{0, 2, 8}));

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    MESSAGE("skipped: " << result.skipped.size() << " level(s)");
    REQUIRE(result.skipped.size() >= 1u);
    CHECK(result.skipped.back() == "level<=8");
    // and nothing over budget was ever fit
    const int budget = 24 / 2 - 1 - theta_hat_size(2, MuMode::Pinned);
    for ( const CandidateFit& candidate : result.candidates )
    {
        CHECK(candidate.num_modes() <= budget);
    }
}

TEST_CASE("the target certificate stops the search early")
{
    std::mt19937 gen(7);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));
    config.target_score = 0.05;

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    CHECK(result.stop_reason == StopReason::Target);
    CHECK(result.score <= 0.05);

    // and without it the same fit walks the whole initialization grid
    config.target_score = std::nullopt;
    const ProbeFitResult full =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);
    MESSAGE("certificate: " << result.candidates.size() << " candidates vs "
                            << full.candidates.size() << " without");
    CHECK(full.candidates.size() > result.candidates.size());
}

TEST_CASE("the simplicity tie-break prefers the smaller mode set")
{
    // Two nested levels both fit the target exactly, so their scores tie; the
    // winner must be the smaller one.
    std::mt19937 gen(8);
    const Target target = make_target(gen, 0);  // built from level 0 alone
    ProbeFitConfig config =
        basic_config(std::make_shared<ShellLadder>(std::vector<int>{0, 2}));
    config.tie_delta = 1e-6;

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    MESSAGE("tie-break chose " << result.model.modes.size() << " modes at score "
                               << result.score);
    CHECK(result.model.modes.size() == modes_up_to_level(2, 0).size());
}

TEST_CASE("the fit is a pure function of its inputs")
{
    // No hidden randomness anywhere: the split and the jitter are data, so the
    // same call twice gives bit-identical everything.
    std::mt19937 gen(9);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<ShellLadder>(std::vector<int>{0, 2}));

    const ProbeFitResult first =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);
    const ProbeFitResult second =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);

    REQUIRE(first.candidates.size() == second.candidates.size());
    CHECK(first.winner == second.winner);
    CHECK(first.model.theta == second.model.theta);
    CHECK(first.model.c == second.model.c);
    for ( std::size_t i = 0; i < first.candidates.size(); ++i )
    {
        CHECK(first.candidates[i].score == second.candidates[i].score);
        CHECK(first.candidates[i].model.theta == second.candidates[i].model.theta);
    }
}

TEST_CASE("supplying a different split changes the score, deterministically")
{
    std::mt19937 gen(10);
    const Target target = make_target(gen, 2, 40, 0.05);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));

    ProbeFitConfig permuted = config;
    permuted.split = kfold_split(static_cast<int>(target.y.size()), 5, 4242u);

    const ProbeFitResult plain =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {}, target.mass);
    const ProbeFitResult shuffled =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, permuted, {}, target.mass);

    MESSAGE("round-robin score " << plain.score << " vs permuted " << shuffled.score);
    CHECK(plain.score != shuffled.score);   // the split is genuinely in play
    CHECK(plain.score == doctest::Approx(shuffled.score).epsilon(0.5));
}

TEST_CASE("a supplied guess is tried before the default rungs")
{
    std::mt19937 gen(11);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));

    const lgpsf::EllipsoidFrame truth =
        lgpsf::unpack_theta_hat(target.theta_hat_true, target.mu0, MuMode::Pinned);
    lgpsf::InitialGuess prior;
    prior.sigma = truth.L * truth.L.transpose();
    prior.label = "sigma0";

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {prior}, target.mass);
    REQUIRE(!result.candidates.empty());
    CHECK(result.candidates.front().label == "sigma0");
    CHECK(result.score < 1e-6);

    // An unlabelled guess still identifies itself in the candidate table.
    lgpsf::InitialGuess unlabelled;
    unlabelled.sigma = prior.sigma;
    const ProbeFitResult anonymous =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {unlabelled}, target.mass);
    CHECK(anonymous.candidates.front().label == "guess[0]");

    // num_rungs = 0 means "only mine" -- and with nothing of mine, it is an
    // error rather than a silent fallback to the baseline.
    ProbeFitConfig only_mine = config;
    only_mine.num_rungs = 0;
    const ProbeFitResult alone =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, only_mine, {prior}, target.mass);
    CHECK(alone.candidates.size() == 1);
    CHECK(alone.candidates.front().label == "sigma0");
    CHECK_THROWS_AS(fit_from_probes(target.x, target.m2_diag, target.z, target.y,
                                    target.mu0, target.spike_index, only_mine, {},
                                    target.mass),
                    std::invalid_argument);
}

TEST_CASE("a guess carrying its own mu is pinned there, not at default_mu")
{
    // MuPolicy::Pinned means the optimizer does not move mu from its initial
    // guess -- not that mu is default_mu. A guess with its own center must
    // therefore ship a model centered there, and `theta_init` must record it.
    std::mt19937 gen(11);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));
    config.num_rungs = 0;

    const lgpsf::EllipsoidFrame truth =
        lgpsf::unpack_theta_hat(target.theta_hat_true, target.mu0, MuMode::Pinned);
    const int dim = static_cast<int>(target.mu0.size());
    const Eigen::VectorXd offset =
        target.mu0 + Eigen::VectorXd::Constant(dim, 0.05);

    lgpsf::InitialGuess elsewhere;
    elsewhere.sigma = truth.L * truth.L.transpose();
    elsewhere.mu = offset;
    elsewhere.label = "offset";

    const ProbeFitResult result =
        fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                        target.spike_index, config, {elsewhere}, target.mass);
    REQUIRE(result.candidates.size() == 1);
    const Eigen::VectorXd shipped_mu = result.model.frame().mu;
    CHECK(shipped_mu.isApprox(offset, 1e-12));
    CHECK(result.candidates.front().theta_init.head(dim).isApprox(offset, 1e-12));
    // and NOT at default_mu, which is what a row-wide origin would have given
    CHECK((shipped_mu - target.mu0).norm() > 1e-3);
}

TEST_CASE("the engine supplies adaptive feedback to a policy that asks for it")
{
    // No shipped policy consumes the margin-profit hook, so this checks it
    // directly through a policy that records what it was handed.
    //
    // The property asserted is Cauchy-Schwarz: a one-step reduction
    // (a.r)^2 / ||a||^2 cannot exceed the residual energy ||r||^2 it is
    // reducing. Note that asking for the profit of an ALREADY-ACTIVE mode is
    // outside the contract and returns a meaningless number -- its column
    // projects to roundoff, so the ratio is a 0/0 that the relative floor
    // cannot discriminate. Adaptive policies score margin groups, which are by
    // construction not active.
    struct Recording : lgpsf::ModePolicy
    {
        std::vector<Mode> first, second;
        mutable std::vector<double> already_active, newly_added;
        mutable double residual_norm_squared = -1.0;

        std::optional<lgpsf::ModeProposal> propose(
            const lgpsf::ModeSearchContext& ctx ) const override
        {
            if ( ctx.history.empty() )
            {
                return lgpsf::ModeProposal{"first", first};
            }
            if ( ctx.history.size() == 1u )
            {
                CHECK(static_cast<bool>(ctx.margin_profit));
                residual_norm_squared = ctx.residual_norm_squared;
                const Eigen::VectorXd inside = ctx.margin_profit(first);
                const Eigen::VectorXd outside = ctx.margin_profit(second);
                already_active.assign(inside.data(), inside.data() + inside.size());
                newly_added.assign(outside.data(), outside.data() + outside.size());
                return lgpsf::ModeProposal{"second", second};
            }
            return std::nullopt;
        }
    };

    std::mt19937 gen(14);
    const Target target = make_target(gen, 2, 40, 0.05);
    auto policy = std::make_shared<Recording>();
    policy->first = modes_up_to_level(2, 0);
    policy->second = target.modes;

    ProbeFitConfig config = basic_config(policy);
    fit_from_probes(target.x, target.m2_diag, target.z, target.y, target.mu0,
                    target.spike_index, config, {}, target.mass);

    REQUIRE(!policy->already_active.empty());
    REQUIRE(!policy->newly_added.empty());
    CHECK(policy->residual_norm_squared > 0.0);

    double added_max = 0.0;
    for ( double profit : policy->newly_added )
    {
        CHECK(profit >= 0.0);
        CHECK(std::isfinite(profit));
        // no single mode can remove more than all of the residual energy
        CHECK(profit <= policy->residual_norm_squared * (1.0 + 1e-9));
        added_max = std::max(added_max, profit);
    }
    MESSAGE("margin profit: best newly-added " << added_max << " against residual energy "
                                               << policy->residual_norm_squared);
    CHECK(added_max > 0.0);
}

TEST_CASE("malformed inputs and a missing policy are rejected eagerly")
{
    std::mt19937 gen(12);
    const Target target = make_target(gen);
    ProbeFitConfig config =
        basic_config(std::make_shared<FixedSet>(target.modes, "truth"));

    ProbeFitConfig no_policy = config;
    no_policy.mode_policy.reset();
    CHECK_THROWS_AS(fit_from_probes(target.x, target.m2_diag, target.z, target.y,
                                    target.mu0, target.spike_index, no_policy),
                    std::invalid_argument);

    CHECK_THROWS_AS(fit_from_probes(target.x, target.m2_diag, target.z,
                                    Eigen::VectorXd::Zero(3), target.mu0,
                                    target.spike_index, config),
                    std::invalid_argument);
    CHECK_THROWS_AS(fit_from_probes(target.x, target.m2_diag, target.z, target.y,
                                    Eigen::VectorXd::Zero(3), target.spike_index,
                                    config),
                    std::invalid_argument);
    CHECK_THROWS_AS(fit_from_probes(target.x, target.m2_diag, target.z, target.y,
                                    target.mu0, 99999, config),
                    std::invalid_argument);

    // every level over budget: nothing to fit, and that is an error
    ProbeFitConfig starved = config;
    starved.mode_policy = std::make_shared<ShellLadder>(std::vector<int>{8});
    CHECK_THROWS_AS(fit_from_probes(target.x, target.m2_diag,
                                    target.z.leftCols(12),
                                    target.y.head(12), target.mu0,
                                    target.spike_index, starved),
                    std::invalid_argument);
}
