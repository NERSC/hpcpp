/*
 * MIT License
 *
 * Copyright (c) 2023 Chuanqiu He
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

/*
 *  implememt task-graph based Cholesky decomposition: tiled Cholesky decomposition with OpenMP tasks.
 *  references:
 *  >Buttari, Alfredo, et al. "A class of parallel tiled linear algebra algorithms for multicore architectures." Parallel Computing 35.1 (2009): 38-53.
 *  >Dorris, Joseph, et al. "Task-based Cholesky decomposition on knights corner using OpenMP." High Performance Computing: ISC High Performance 2016 International Workshops, ExaComm, E-MuCoCoS, HPC-IODC, IXPUG, IWOPH, P^ 3MA, VHPC, WOPSSS, Frankfurt, Germany, June 19–23, 2016, Revised Selected Papers 31. Springer International Publishing, 2016.)
*/

#define CHOLESKY_OMP

#include "matrixutil.hpp"

void tiled_cholesky(data_type* matrix_split[], const std::size_t tile_size, const std::size_t num_tiles,
                    CBLAS_ORDER blasLay, const int lapackLay) {
    std::size_t m = 0, n = 0, k = 0;

    for (k = 0; k < num_tiles; ++k) {  //POTRF
#pragma omp task depend(inout : matrix_split[k * num_tiles + k])
        { int info = LAPACKE_dpotrf(lapackLay, 'L', tile_size, matrix_split[k * num_tiles + k], tile_size); }

        for (m = k + 1; m < num_tiles; ++m) {  //DTRSM
#pragma omp task depend(in : matrix_split[k * num_tiles + k]) depend(inout : matrix_split[m * num_tiles + k])
            {
                cblas_dtrsm(blasLay, CblasRight, CblasLower, CblasTrans, CblasNonUnit, tile_size, tile_size, 1.0,
                            matrix_split[k * num_tiles + k], tile_size, matrix_split[m * num_tiles + k], tile_size);
            }
        }

        for (n = k + 1; n < num_tiles; ++n) {  //DSYRK
#pragma omp task depend(in : matrix_split[n * num_tiles + k]) depend(inout : matrix_split[n * num_tiles + n])
            {
                cblas_dsyrk(blasLay, CblasLower, CblasNoTrans, tile_size, tile_size, -1.0,
                            matrix_split[n * num_tiles + k], tile_size, 1.0, matrix_split[n * num_tiles + n],
                            tile_size);
            }
            for (m = n + 1; m < num_tiles; ++m) {  //DGEMM
#pragma omp task depend(in : matrix_split[m * num_tiles + k], matrix_split[n * num_tiles + k]) \
    depend(inout : matrix_split[m * num_tiles + n])
                {
                    cblas_dgemm(blasLay, CblasNoTrans, CblasTrans, tile_size, tile_size, tile_size, -1.0,
                                matrix_split[m * num_tiles + k], tile_size, matrix_split[n * num_tiles + k], tile_size,
                                1.0, matrix_split[m * num_tiles + n], tile_size);
                }
            }
        }
    }
}

int main(int argc, char** argv) {

    // parse params
    args_params_t args = argparse::parse<args_params_t>(argc, argv);

    const std::size_t matrix_size = args.mat_size;    // Number of matrix dimension.
    const std::size_t num_tiles = args.num_tiles;     // matrix size MUST be divisible
    bool verifycorrectness = args.verifycorrectness;  // verify tiled_cholesky results with MKL cholesky
    bool layRow = args.layRow;                        // set the matrix in row-major order(default)
    int nthreads = args.nthreads;

    fmt::print("matrix_size = {}, num_tiles = {}\n\n", matrix_size, num_tiles);

    // Check mat_size is divisible by num_tiles
    if (matrix_size % num_tiles != 0) {
        fmt::print("matrix size must be divisible by num_tiles.. aborting\n");
        throw std::invalid_argument("Matrix size is not divisible by num_tiles");
    }

    if (matrix_size == 0) {
        fmt::print("0 is an illegal input matrix_size \n");
        std::exit(0);
    }

    const std::size_t tile_size = {matrix_size / num_tiles};
    const std::size_t tot_tiles = {num_tiles * num_tiles};

    data_type* A = new data_type[matrix_size * matrix_size];

    // Allocate memory for tiled_cholesky for the full matrix
    data_type* A_cholesky = new data_type[matrix_size * matrix_size];

    // Allocate memory for MKL cholesky for the full matrix
    data_type* A_MKL = new data_type[matrix_size * matrix_size];

    // Memory allocation for tiled matrix
    data_type** Asplit = new data_type*[tot_tiles];

    for (std::size_t i = 0; i < tot_tiles; ++i) {
        // Buffer per tile
        Asplit[i] = new data_type[tile_size * tile_size];
    }

    //Generate a symmetric positve-definite matrix
    A = generate_positiveDefinitionMatrix(matrix_size);

    //copying matrices into separate variables for tiled cholesky (A_cholesky)
    std::memcpy(A_cholesky, A, matrix_size * matrix_size * sizeof(data_type));
    //copying matrices into separate variables for MKL cholesky (A_MKL)
    std::memcpy(A_MKL, A, matrix_size * matrix_size * sizeof(data_type));

    // CBLAS_ORDER: Indicates whether a matrix is in row-major or column-major order.
    CBLAS_ORDER blasLay;
    int lapackLay;

    if (layRow) {
        blasLay = CblasRowMajor;
        lapackLay = LAPACK_ROW_MAJOR;
    } else {
        blasLay = CblasColMajor;
        lapackLay = LAPACK_COL_MAJOR;
    }

    //splits the input matrix into tiles
    split_into_tiles(A_cholesky, Asplit, num_tiles, tile_size, matrix_size, layRow);

    // Measure execution time of tiled_cholesky
    Timer timer;
    //run the tiled Cholesky function
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp master
        { tiled_cholesky(Asplit, tile_size, num_tiles, blasLay, lapackLay); }
    }

    auto time = timer.stop();

    if (args.time) {
        fmt::print("Time for tiled_cholesky decomposition(omp), Duration: {:f} ms\n", time);
    }

    //assembling seperated tiles back into full matrix
    assemble_tiles(Asplit, A_cholesky, num_tiles, tile_size, matrix_size, layRow);

    //calling LAPACKE_dpotrf cholesky for verification
    // Measure execution time of MKL Cholesky decomposition
    Timer timer2;
    int info = LAPACKE_dpotrf(lapackLay, 'L', matrix_size, A_MKL, matrix_size);
    auto time2 = timer2.stop();
    if (args.time) {
        fmt::print("Time for MKL Cholesky decomposition, Duration: {:f} ms\n", time2);
    }

    if (info != 0) {
        fmt::print("Error with dpotrf, info = {}\n", info);
    }

    if (verifycorrectness) {
        bool res = verify_results(A_cholesky, A_MKL, matrix_size * matrix_size);
        fmt::print("\n");
        if (res) {
            fmt::print("Tiled Cholesky decomposition successful.\n");
        } else {
            fmt::print("Tiled Cholesky decomposition failed.\n");
        }
        fmt::print("\n");
    }

    // print lower_matrix if tiled_cholesky sucessfull
    if (args.lower_matrix) {
        fmt::print("The lower matrix of input matrix after tiled_cholesky: \n");
        printLowerResults(A_cholesky, matrix_size);
    }

    //memory free
    delete[] A;
    delete[] A_cholesky;
    delete[] A_MKL;
    for (std::size_t i = 0; i < tot_tiles; ++i) {
        delete[] Asplit[i];
    }

    return 0;
}
