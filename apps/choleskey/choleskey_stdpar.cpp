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
// This example provides a stdpar implementation for choleskey decomposition code.

#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>
#include "argparse/argparse.hpp"
#include "commons.hpp"
#include "matrixutil.hpp"

using namespace std;

struct solver {

    using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

    template <typename T>
    std::vector<std::vector<T>> Cholesky_Decomposition(std::vector<T>& vec, int n) {
        std::vector<std::vector<T>> lower(n, std::vector<T>(n, 0));

        auto matrix_ms = std::mdspan<T, view_2d, std::layout_right>(vec.data(), n, n);

        auto multiplier_lambda = [=](auto a, auto b) {
            return a * b;
        };

        // Decomposing a matrix into Lower Triangular
        for (int i = 0; i < matrix_ms.extent(0); i++) {
            for (int j = 0; j <= i; j++) {
                T sum = 0;

                if (j == i)  // summation for diagonals
                {
                    sum = std::transform_reduce(std::execution::par, lower[j].cbegin(), lower[j].cbegin() + j, 0,
                                                std::plus{}, [=](int val) { return val * val; });

                    lower[j][j] = std::sqrt(matrix_ms(i, j) - sum);

                } else {  // Evaluating L(i, j) using L(j, j)

                    sum = std::transform_reduce(std::execution::par, lower[j].cbegin(), lower[j].cbegin() + j,
                                                lower[i].cbegin(), 0, std::plus<>(), multiplier_lambda);

                    lower[i][j] = (matrix_ms(i, j) - sum) / lower[j][j];
                }
            }
        }
        return lower;
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {

    std::uint64_t nd = args.nd;  // Number of matrix dimension.

    std::vector<int> inputMatrix = generate_pascal_matrix<int>(nd);

    // Create the solver object
    solver solve;
    // Measure execution time.
    Timer timer;

    // start decomposation
    auto res_matrix = solve.Cholesky_Decomposition(inputMatrix, nd);
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
