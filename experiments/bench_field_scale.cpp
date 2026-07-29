// Timing profile at field scale: 6557 columns, 100 probes, 100 fitted rows.
//
// Mirrors the glaciology validation configuration so the numbers mean
// something for the replay: tau_window = 5, levels [0,1,2], mu pinned,
// num_rungs = 4, no window-shape rungs (they were a config flag then; the
// equivalent today is simply not passing `window_shape_ladder`), and the BALL
// window (aspect cap 1), which is what the prototype derives.
//
// Also dumps the generated problem to raw binary so the Python prototype runs
// on identical data.
//
// Build (Debug builds are meaningless for timing -- Eigen needs the optimizer):
//   g++ -O3 -march=native -std=c++17 -pthread \
//       -I include -I ../ellipsoid_tree/include \
//       -isystem build/_deps/eigen3_src-src \
//       -o experiments/bench_field_scale experiments/bench_field_scale.cpp

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include "lgpsf/operator_fit.hpp"

using namespace lgpsf;
using Clock = std::chrono::steady_clock;

static double seconds_since( Clock::time_point t0 )
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static void dump( const std::string& path, const double* data, std::size_t n )
{
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n * sizeof(double)));
}

int main()
{
    // ---- problem, field-shaped -------------------------------------------
    const int per_side = 81;                 // 6561, near the validation mesh
    const int num_cols = per_side * per_side;
    const int num_probes = 100;              // the validated probe budget
    const int num_fitted = 100;
    const int dim = 2;
    const double extent = 1.0;
    const double h = 2.0 * extent / (per_side - 1);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<double> jitter(-0.15 * h, 0.15 * h);
    std::uniform_real_distribution<double> mass(0.6, 1.4);
    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::MatrixXd x_cols(num_cols, dim);
    for ( int i = 0, r = 0; i < per_side; ++i )
    {
        for ( int j = 0; j < per_side; ++j, ++r )
        {
            x_cols(r, 0) = -extent + 2 * extent * i / (per_side - 1) + jitter(gen);
            x_cols(r, 1) = -extent + 2 * extent * j / (per_side - 1) + jitter(gen);
        }
    }
    Eigen::VectorXd m1(num_cols), m2(num_cols);
    for ( int j = 0; j < num_cols; ++j ) { m1(j) = mass(gen); m2(j) = mass(gen); }

    // an anisotropic sigma field: 1-sigma axes of about 2h and 1h, rotating
    std::vector<Eigen::MatrixXd> sigma(static_cast<std::size_t>(num_cols),
                                       Eigen::MatrixXd::Identity(dim, dim));
    std::vector<char> gate(static_cast<std::size_t>(num_cols), 0);
    std::vector<int> fitted;
    for ( int r = 0; r < num_fitted; ++r )
    {
        const int rho = (num_cols / (num_fitted + 1)) * (r + 1);
        fitted.push_back(rho);
        gate[static_cast<std::size_t>(rho)] = 1;
    }
    for ( int rho : fitted )
    {
        const double angle = 0.7 * rho;
        Eigen::Matrix2d rot;
        rot << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
        Eigen::Matrix2d axes = Eigen::Matrix2d::Zero();
        axes(0, 0) = (2.0 * h) * (2.0 * h);
        axes(1, 1) = (1.0 * h) * (1.0 * h);
        sigma[static_cast<std::size_t>(rho)] = rot * axes * rot.transpose();
    }

    Eigen::MatrixXd V(num_cols, num_probes);
    for ( int j = 0; j < num_cols; ++j )
        for ( int l = 0; l < num_probes; ++l ) V(j, l) = normal(gen);

    // truth: an LG expansion on a slightly different ellipsoid, plus a spike
    const std::vector<Mode> truth_modes = modes_up_to_level(dim, 2);
    Eigen::MatrixXd HV = Eigen::MatrixXd::Zero(num_cols, num_probes);
    for ( int rho : fitted )
    {
        const Eigen::VectorXd center = x_cols.row(rho).transpose();
        Eigen::VectorXd theta_hat(theta_hat_size(dim, MuMode::Pinned));
        theta_hat << std::log(1.6 * h), std::log(0.9 * h), 0.1 * h;
        const WhitenedBasis basis(x_cols, m1(rho), m2, truth_modes, center,
                                  MuMode::Pinned);
        Eigen::VectorXd c(truth_modes.size());
        for ( Eigen::Index i = 0; i < c.size(); ++i ) c(i) = normal(gen);
        const Eigen::VectorXd phi_hat = basis(theta_hat).values() * c;
        Eigen::VectorXd h_row(num_cols);
        for ( int j = 0; j < num_cols; ++j )
            h_row(j) = std::sqrt(m1(rho)) * std::sqrt(m2(j)) * phi_hat(j);
        h_row(rho) += m1(rho) * 0.4;
        HV.row(rho) = h_row.transpose() * V;
    }

    // ---- dump for the Python side --------------------------------------
    {
        std::vector<double> buf;
        buf.assign(x_cols.data(), x_cols.data() + x_cols.size());  // column-major
        dump("dev/bench_x.bin", buf.data(), buf.size());
        dump("dev/bench_m1.bin", m1.data(), static_cast<std::size_t>(m1.size()));
        dump("dev/bench_m2.bin", m2.data(), static_cast<std::size_t>(m2.size()));
        buf.assign(V.data(), V.data() + V.size());
        dump("dev/bench_V.bin", buf.data(), buf.size());
        buf.assign(HV.data(), HV.data() + HV.size());
        dump("dev/bench_HV.bin", buf.data(), buf.size());
        buf.clear();
        for ( int rho = 0; rho < num_cols; ++rho )
            for ( int i = 0; i < dim; ++i )
                for ( int j = 0; j < dim; ++j )
                    buf.push_back(sigma[static_cast<std::size_t>(rho)](i, j));
        dump("dev/bench_sigma.bin", buf.data(), buf.size());
        std::ofstream meta("dev/bench_meta.txt");
        meta << num_cols << " " << num_probes << " " << dim << "\n";
        for ( int rho : fitted ) meta << rho << " ";
        meta << "\n";
    }

    // ---- the configuration the validation runs -------------------------
    OperatorFitConfig config;
    config.tau_window = 5.0;
    config.window_aspect_cap = 1.0;   // the BALL, matching the prototype
    config.spike = true;
    config.row.mode_policy = std::make_shared<ShellLadder>(std::vector<int>{0, 1, 2});
    config.row.mu = MuPolicy::Pinned;
    config.row.num_rungs = 4;
    config.num_threads = 1;           // single-threaded, to compare with Python

    std::printf("Field-scale profile: %d columns, %d probes, %d fitted rows\n",
                num_cols, num_probes, num_fitted);

    auto t0 = Clock::now();
    const OperatorFit fit = fit_operator(x_cols, m1, m2, V, HV, sigma, config,
                                         std::nullopt, std::nullopt, gate);
    const double total_ball = seconds_since(t0);

    std::size_t window_total = 0;
    int shipped = 0;
    for ( int rho : fitted )
    {
        window_total += fit.model.row_window(rho).size();
        shipped += fit.model.has_model(rho) ? 1 : 0;
    }
    const double mean_window = static_cast<double>(window_total) / num_fitted;
    std::printf("  ball window (cap 1):   %8.3f s total, %7.1f ms/row, "
                "mean window %.0f pts, %d/%d shipped\n",
                total_ball, 1000.0 * total_ball / num_fitted, mean_window,
                shipped, num_fitted);

    OperatorFitConfig ellipse = config;
    ellipse.window_aspect_cap = std::numeric_limits<double>::infinity();
    t0 = Clock::now();
    const OperatorFit fit_e = fit_operator(x_cols, m1, m2, V, HV, sigma, ellipse,
                                           std::nullopt, std::nullopt, gate);
    const double total_ellipse = seconds_since(t0);
    std::size_t we = 0;
    for ( int rho : fitted ) we += fit_e.model.row_window(rho).size();
    std::printf("  ellipsoid window (inf):%8.3f s total, %7.1f ms/row, "
                "mean window %.0f pts\n",
                total_ellipse, 1000.0 * total_ellipse / num_fitted,
                static_cast<double>(we) / num_fitted);

    // threading, since the replay will use it
    for ( int threads : {2, 4} )
    {
        OperatorFitConfig par = config;
        par.num_threads = threads;
        t0 = Clock::now();
        fit_operator(x_cols, m1, m2, V, HV, sigma, par, std::nullopt, std::nullopt, gate);
        const double t = seconds_since(t0);
        std::printf("  ball, %d threads:      %8.3f s  (%.2fx)\n", threads, t,
                    total_ball / t);
    }

    // ---- where a single row's time goes ---------------------------------
    const int probe_row = fitted[num_fitted / 2];
    const std::vector<int> window = fit.model.row_window(probe_row);
    const Eigen::Index w = static_cast<Eigen::Index>(window.size());
    Eigen::MatrixXd x_w(w, dim);
    Eigen::VectorXd m2_w(w);
    Eigen::MatrixXd z(w, num_probes);
    for ( Eigen::Index i = 0; i < w; ++i )
    {
        x_w.row(i) = x_cols.row(window[static_cast<std::size_t>(i)]);
        m2_w(i) = m2(window[static_cast<std::size_t>(i)]);
        z.row(i) = V.row(window[static_cast<std::size_t>(i)]);
    }
    const Eigen::VectorXd y = HV.row(probe_row).transpose();
    const Eigen::VectorXd center = x_cols.row(probe_row).transpose();
    const double target_mass = m1(probe_row);

    ProbeFitConfig row_cfg = config.row;
    row_cfg.split = kfold_split(num_probes, row_cfg.cv_folds);
    row_cfg.jitter = jitter_table(theta_hat_size(dim, MuMode::Pinned), kMaxModeProposals);

    t0 = Clock::now();
    const ProbeFitResult one =
        fit_from_probes(x_w, m2_w, z, y, center, 0, row_cfg, sigma[static_cast<std::size_t>(probe_row)],
                        target_mass);
    const double row_time = seconds_since(t0);
    std::printf("\n  one row (window %ld pts): %.1f ms, %zu candidates\n",
                static_cast<long>(w), 1000.0 * row_time, one.candidates.size());

    const std::vector<Mode> modes = modes_up_to_level(dim, 2);
    const WhitenedBasis basis(x_w, target_mass, m2_w, modes, center, MuMode::Pinned);
    Eigen::VectorXd theta_hat(theta_hat_size(dim, MuMode::Pinned));
    theta_hat << std::log(1.6 * h), std::log(0.9 * h), 0.0;
    const Eigen::MatrixXd z_hat = whiten_probes(z, m2_w);
    const Eigen::VectorXd y_hat = whiten_data(y, target_mass);
    Eigen::MatrixXd extra = Eigen::MatrixXd::Zero(w, 1);
    extra(0, 0) = 1.0;
    const Eigen::MatrixXd e_hat = whiten_extra(extra, target_mass, m2_w);
    const Eigen::MatrixXd cot = Eigen::MatrixXd::Ones(w, static_cast<Eigen::Index>(modes.size()));

    const auto micro = [&]( const char* what, int reps, auto&& fn ) {
        fn();
        auto t = Clock::now();
        for ( int i = 0; i < reps; ++i ) fn();
        const double each = seconds_since(t) / reps;
        std::printf("    %-34s %8.1f us\n", what, 1e6 * each);
        return each;
    };
    std::printf("  components at that size (%zu modes):\n", modes.size());
    const double t_values = micro("basis(theta).values()", 2000,
                                  [&]{ volatile double s = basis(theta_hat).values()(0,0); (void)s; });
    const double t_vjp = micro("basis(theta).vjp(w)", 2000,
                               [&]{ volatile double s = basis(theta_hat).vjp(cot)(0,0); (void)s; });
    const double t_cv = micro("linear_cv_score", 500,
                              [&]{ volatile double s = linear_cv_score(z_hat, y_hat, basis, theta_hat, e_hat, row_cfg.split); (void)s; });
    const double t_lm = micro("fit_varpro (one candidate)", 200,
                              [&]{ volatile double s = fit_varpro(z_hat, y_hat, basis, theta_hat, e_hat, row_cfg.varpro).cost; (void)s; });
    std::printf("  -> a candidate costs about one fit_varpro (%.0f us) plus one\n"
                "     linear_cv_score (%.0f us); values/vjp are %.0f/%.0f us each\n",
                1e6 * t_lm, 1e6 * t_cv, 1e6 * t_values, 1e6 * t_vjp);
    return 0;
}
