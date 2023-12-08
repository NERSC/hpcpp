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
// This example provides a serial(mdspan) implementation for choleskey decomposition code.

#include <bits/stdc++.h>
#include <vector>
#include "matrixutil.hpp"

using data_type = float;

struct solver {

    using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

    typedef std::mdspan<data_type, view_2d, std::layout_right> matrix_ms_t;

    template <typename T>
    matrix_ms_t Cholesky_Decomposition(std::vector<T>& vec, const std::size_t n) {
        std::vector<T> lower(n * n, 0);

        auto matrix_ms = std::mdspan<T, view_2d, std::layout_right>(vec.data(), n, n);
        auto lower_ms = std::mdspan<T, view_2d, std::layout_right>(lower.data(), n, n);

        // Decomposing a matrix into Lower Triangular
        for (auto i = 0; i < matrix_ms.extent(0); i++) {
            for (auto j = 0; j <= i; j++) {
                auto sum = 0.0;
                for (int k = 0; k < j; k++) {
                    //summation
                    sum += (lower_ms(i, k) * lower_ms(j, k));
                }

                if (j == i) {
                    // for diagonals
                    lower_ms(j, j) = sqrt(matrix_ms(i, j) - sum);
                } else {
                    // for non-diagonals
                    lower_ms(i, j) = (matrix_ms(i, j) - sum) / lower_ms(j, j);
                }
            }
        }
        return lower_ms;
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {

    // obtain input_file args
    std::string inputFilePath = args.input_file;

    // check inputFilePath
    if (inputFilePath.empty() || !std::ifstream(inputFilePath)) {
        fmt::print("Error: File '{}' not found or input file not provided.\n", inputFilePath);
        return 1;
    }

    // read input file and store it to inputMatrix
    std::vector<data_type> inputMatrix = readDataFromFile<data_type>(inputFilePath);
    std::size_t size = inputMatrix.size();
    std::size_t nd = std::sqrt(size);  // Number of matrix dimension

    // Create the solverobject
    solver solve;
    // Measure execution time.
    Timer timer;
    // start decomposation
    auto res_matrix = solve.Cholesky_Decomposition(inputMatrix, nd);
    auto time = timer.stop();

    // Print the final results
    if (args.results) {
        // Displaying Lower Triangular
        fmt::print("{:>6}\n", "Lower Triangular after cholesky decomposition:");
        fmt::print("{:>30}\n", "");

        for (int i = 0; i < nd; i++) {
            // Lower Triangular
            for (int j = 0; j < nd; j++)
                fmt::print("{:>6}\t", res_matrix(i, j));

            fmt::print("\t\n");
        }
    }

    if (args.time) {
        fmt::print("\n");
        fmt::print("cholesky decomposition in serial, Duration: {:f} ms\n", time);
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
