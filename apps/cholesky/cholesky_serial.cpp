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
// This example provides a serial(mdspan) implementation for cholesky decomposition code.

#include <bits/stdc++.h>
#include <vector>
#include "argparse/argparse.hpp"
#include "commons.hpp"
#include "matrixutil.hpp"

using namespace std;

struct solver {

    using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

    typedef std::mdspan<int, view_2d, std::layout_right> matrix_ms_t;

    template <typename T>
    matrix_ms_t Cholesky_Decomposition(std::vector<T>& vec, int n) {
        std::vector<T> lower(n * n, 0);

        auto matrix_ms = std::mdspan<T, view_2d, std::layout_right>(vec.data(), n, n);
        auto lower_ms = std::mdspan<T, view_2d, std::layout_right>(lower.data(), n, n);

        // Decomposing a matrix into Lower Triangular
        for (int i = 0; i < matrix_ms.extent(0); i++) {
            for (int j = 0; j <= i; j++) {
                T sum = 0;

                if (j == i) {
                    // summation for diagonals
                    for (int k = 0; k < j; k++)
                        sum += pow(lower_ms(j, k), 2);
                    lower_ms(j, j) = sqrt(matrix_ms(i, j) - sum);
                } else {
                    // Evaluating L(i, j) using L(j, j)
                    for (int k = 0; k < j; k++)
                        sum += (lower_ms(i, k) * lower_ms(j, k));
                    lower_ms(i, j) = (matrix_ms(i, j) - sum) / lower_ms(j, j);
                }
            }
        }
        return lower_ms;
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {

    std::uint64_t nd = args.nd;  // Number of matrix dimension.

    std::vector<int> inputMatrix = generate_pascal_matrix<int>(nd);

    // Create the solverobject
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
                fmt::print("{:>6}\t", res_matrix(i, j));
            fmt::print("\t");

            // Transpose of Lower Triangular
            for (int j = 0; j < nd; j++)
                fmt::print("{:>6}\t", res_matrix(j, i));
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
