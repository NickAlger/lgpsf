// How much does the SVD actually cost, and what would replacing it buy?
//
// The two SVD sites are `inner_solve` (once per LM residual evaluation, on the
// (k, m) residualized design matrix) and `linear_cv_score` (five times per
// candidate, on the (0.8k, m) training folds). Sizes below are the ones the
// field-scale profile actually sees at k = 20, 40, 100.
//
//   g++ -O3 -march=native -std=c++17 -I include \
//       -isystem build/_deps/eigen3_src-src -o dev/bench_solve dev/bench_solve.cpp

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

using Clock = std::chrono::steady_clock;

template <typename F>
double time_it( F&& f, int repeats )
{
    const auto t0 = Clock::now();
    for ( int i = 0; i < repeats; ++i ) { f(); }
    return 1e6 * std::chrono::duration<double>(Clock::now() - t0).count() / repeats;
}

int main()
{
    std::mt19937 gen(0);
    std::normal_distribution<double> normal;

    // `collinear` = how nearly the last column duplicates the first. LG design
    // matrices are ill-conditioned, but how ill decides whether the SVD's cost
    // is inherent or is slow Jacobi convergence, so both are measured.
    const double collinear = ( std::getenv("COLLINEAR" ) )
                                 ? std::atof(std::getenv("COLLINEAR")) : 0.0;
    std::printf("near-collinearity: %g (0 = generic random)\n", collinear);
    std::printf("%-12s %9s %9s %9s %9s %9s\n", "size (k x m)", "BDCSVD",
                "JacobiSVD", "ColPivQR", "HouseQR", "LLT-normal");
    for ( const auto& size : std::vector<std::pair<int, int>>{
              {20, 4}, {16, 4},        // k=20: LM inner solve, CV fold
              {40, 12}, {32, 12},      // k=40
              {100, 21}, {80, 21},     // k=100
              {100, 28}, {80, 28}} )   // k=100, shells-L6 width
    {
        const int rows = size.first, cols = size.second;
        Eigen::MatrixXd A(rows, cols);
        for ( int i = 0; i < rows; ++i )
            for ( int j = 0; j < cols; ++j ) A(i, j) = normal(gen);
        if ( collinear > 0.0 )
        {
            A.col(cols - 1) = A.col(0) + collinear * A.col(cols - 1);
        }
        const Eigen::VectorXd b = Eigen::VectorXd::NullaryExpr(
            rows, [&]( Eigen::Index ) { return normal(gen); });

        const int repeats = 20000;
        volatile double sink = 0.0;
        const double bdc = time_it([&] {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(
                A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            sink += svd.singularValues()(0);
        }, repeats);
        const double jac = time_it([&] {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                A, Eigen::ComputeThinU | Eigen::ComputeThinV);
            sink += svd.singularValues()(0);
        }, repeats);
        const double cpqr = time_it([&] {
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
            sink += qr.solve(b)(0);
        }, repeats);
        const double hqr = time_it([&] {
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
            sink += qr.solve(b)(0);
        }, repeats);
        const double llt = time_it([&] {
            const Eigen::MatrixXd normal_matrix = A.transpose() * A;
            Eigen::LLT<Eigen::MatrixXd> chol(normal_matrix);
            sink += chol.solve(A.transpose() * b)(0);
        }, repeats);

        std::printf("%3d x %-6d %8.2fus %8.2fus %8.2fus %8.2fus %8.2fus\n",
                    rows, cols, bdc, jac, cpqr, hqr, llt);
    }
    return 0;
}
