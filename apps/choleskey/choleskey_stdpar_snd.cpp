/*
 * MIT License
 *
 * Copyright (c) 2023 Chuanqiu He
 * Copyright (c) 2023 Weile Wei
 * Copyright (c) 2023 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of any
 * required approvals from the U.S. Dept. of Energy).All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
//
// This example provides a stdexec(senders/receivers) implementation for choleskey decomposition code.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexec/execution.hpp>
#include <vector>
#include "argparse/argparse.hpp"
#include "commons.hpp"
#include "exec/static_thread_pool.hpp"

#include "matrixutil.hpp"

using namespace std;

struct solver {

    using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

    template <typename T>
    std::vector<std::vector<T>> Cholesky_Decomposition(std::vector<T>& vec, int n, int np) {

        // test here first, scheduler from a thread pool
        exec::static_thread_pool pool(np);
        stdexec::scheduler auto sch = pool.get_scheduler();
        stdexec::sender auto begin = stdexec::schedule(sch);

        std::vector<std::vector<T>> lower(n, std::vector<T>(n, 0));

        auto matrix_ms = std::mdspan<T, view_2d, std::layout_right>(vec.data(), n, n);

        auto multiplier_lambda = [=](auto a, auto b) {
            return a * b;
        };

        for (int i = 0; i < matrix_ms.extent(0); i++) {
            for (int j = 0; j <= i; j++) {
                // avoid over parallelize
                if (j == 0) {
                    np = 1;
                } else if (j > 0 && np > j) {
                    np = j;
                }

                if (j == i)  // summation for diagonals
                {

                    if (i == 0 && j == 0) {
                        lower[j][j] = std::sqrt(matrix_ms(i, j));
                    } else {

                        std::vector<T> sum_vec(np);  // sub res for each piece
                        int size = j;                // there are j elements need to be calculated(power)

                        stdexec::sender auto send1 =
                            stdexec::bulk(begin, np,
                                          [&](int piece) {
                                              int start = piece * size / np;
                                              int chunk_size = size / np;
                                              int remaining = size % np;
                                              chunk_size += (piece == np - 1) ? remaining : 0;

                                              sum_vec[piece] = std::transform_reduce(
                                                  std::execution::par, counting_iterator(start),
                                                  counting_iterator(start + chunk_size), 0, std ::plus{},
                                                  [=](int val) { return lower[j][val] * lower[j][val]; });
                                          }) |
                            stdexec::then([&sum_vec]() {
                                return std::reduce(std::execution::par, sum_vec.begin(), sum_vec.end());
                            });

                        auto [sum1] = stdexec::sync_wait(std::move(send1)).value();

                        lower[j][j] = std::sqrt(matrix_ms(i, j) - sum1);
                    }

                } else {
                    // Evaluating L(i, j) using L(j, j)

                    if (j == 0) {
                        lower[i][j] = (matrix_ms(i, j)) / lower[j][j];
                    } else {

                        std::vector<T> sum_vec(np);  // sub_result for each par piece
                        int size_nondiag = j;

                        stdexec::sender auto send2 =
                            stdexec::bulk(begin, np,
                                          [&](int piece) {
                                              int start = piece * size_nondiag / np;
                                              int chunk_size = size_nondiag / np;
                                              int remaining = size_nondiag % np;
                                              chunk_size += (piece == np - 1) ? remaining : 0;

                                              sum_vec[piece] = std::transform_reduce(
                                                  std::execution::par, counting_iterator(start),
                                                  counting_iterator(start + chunk_size), 0, std ::plus{},
                                                  [=](int k) { return lower[j][k] * lower[i][k]; });
                                          }) |
                            stdexec::then([&sum_vec]() {
                                return std::reduce(std::execution::par, sum_vec.begin(), sum_vec.end());
                            });

                        auto [sum2] = stdexec::sync_wait(std::move(send2)).value();

                        lower[i][j] = (matrix_ms(i, j) - sum2) / lower[j][j];
                    }
                }
            }
        }
        return lower;
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {

    std::uint64_t nd = args.nd;  // Number of matrix dimension.
    std::uint64_t np = args.np;  // Number of parallel partitions.

    std::vector<int> inputMatrix = generate_pascal_matrix<int>(nd);

    // Create the solver object
    solver solve;

    // Measure execution time.
    Timer timer;

    // start decomposation
    auto res_matrix = solve.Cholesky_Decomposition(inputMatrix, nd, np);
    auto time = timer.stop();

    // Print the final results
    if (args.results) {
        // Displaying Lower Triangular and its Transpose
        fmt::print("{:>6} {:>30}\n", "Lower Triangular", "Transpose");
        for (int i = 0; i < nd; i++) {
            // Lower Triangular
            for (int j = 0; j < nd; j++)
                fmt::print("{:>6}\t", res_matrix[i][j]);
            fmt::print("\t");

            // Transpose of Lower Triangular
            for (int j = 0; j < nd; j++)
                fmt::print("{:>6}\t", res_matrix[j][i]);
            fmt::print("\n");
        }
    }

    if (args.time) {
        fmt::print("Duration: {:f} ms\n", time);
    }

    return 0;
}

// Driver Code for testing
int main(int argc, char* argv[]) {

    // parse params
    args_params_t args = argparse::parse<args_params_t>(argc, argv);
    // see if help wanted
    if (args.help) {
        args.print();  // prints all variables
        return 0;
    }

    benchmark(args);

    return 0;
}
