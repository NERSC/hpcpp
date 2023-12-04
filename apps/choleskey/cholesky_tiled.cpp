/*
1. Aim to implememt task-graph based Cholesky decomposition: tiled Cholesky decomposition with OpenMP tasks.
    references:
    >Buttari, Alfredo, et al. "A class of parallel tiled linear algebra algorithms for multicore architectures." Parallel Computing 35.1 (2009): 38-53.
    >Dorris, Joseph, et al. "Task-based Cholesky decomposition on knights corner using OpenMP." High Performance Computing: ISC High Performance 2016 International Workshops, ExaComm, E-MuCoCoS, HPC-IODC, IXPUG, IWOPH, P^ 3MA, VHPC, WOPSSS, Frankfurt, Germany, June 19–23, 2016, Revised Selected Papers 31. Springer International Publishing, 2016.)

2. Therefore, the first step is to implement tiled Cholesky decomposition algorithm.

3. This file is to implement tiled Cholesky decomposition algorithm. 
    reference the implementation from Intel open source project, hetero-streams which will no longer be maintained by Intel.
    https://github.com/intel/hetero-streams/tree/master/ref_code/cholesky

4. Additionally, include openblas library when build:
   or $ export OPENBLAS_DIR=/openblas/path
*/

#include <bits/stdc++.h>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "tiled_cholesky_help.hpp"

using data_type = double;

void tiled_cholesky(double* matrix_split[], const int tile_size, const int num_tiles, CBLAS_ORDER blasLay,
                    const int lapackLay) {
    unsigned int m, n, k;

    for (k = 0; k < num_tiles; ++k) {
        //POTRF
        int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', tile_size, matrix_split[k * num_tiles + k], tile_size);

        for (m = k + 1; m < num_tiles; ++m) {
            //DTRSM
            cblas_dtrsm(blasLay, CblasRight, CblasLower, CblasTrans, CblasNonUnit, tile_size, tile_size, 1.0,
                        matrix_split[k * num_tiles + k], tile_size, matrix_split[m * num_tiles + k], tile_size);
        }

        for (n = k + 1; n < num_tiles; ++n) {
            //DSYRK
            cblas_dsyrk(blasLay, CblasLower, CblasNoTrans, tile_size, tile_size, -1.0, matrix_split[n * num_tiles + k],
                        tile_size, 1.0, matrix_split[n * num_tiles + n], tile_size);

            for (m = n + 1; m < num_tiles; ++m) {
                //DGEMM
                cblas_dgemm(blasLay, CblasNoTrans, CblasTrans, tile_size, tile_size, tile_size, -1.0,
                            matrix_split[m * num_tiles + k], tile_size, matrix_split[n * num_tiles + k], tile_size, 1.0,
                            matrix_split[m * num_tiles + n], tile_size);
            }
        }
    }
}

int main(int argc, char** argv) {

    // TODO : introduce args
    int matrix_size, num_tiles, tile_size, tot_tiles;
    matrix_size = 40;  // must be an input
    num_tiles = 2;     // matrix size MUST be divisible
    int verifycorrectness = 1;
    bool layRow = true;

    fmt::print("matrix_size = {}, num_tiles = {}\n\n", matrix_size, num_tiles);

    // Check mat_size is divisible by num_tiles
    if (matrix_size % num_tiles != 0) {
        fmt::print("matrix size must be divisible by num_tiles.. aborting\n");
        throw std::invalid_argument("Matrix size is not divisible by num_tiles");
    }

    if (matrix_size == 0) {
        fmt::print(" 0 illegal input matrix_size \n");
        std::exit(0);
    }

    tile_size = matrix_size / num_tiles;
    tot_tiles = num_tiles * num_tiles;

    double* A = new double[matrix_size * matrix_size];

    // Allocate memory for tiled_cholesky for the full matrix
    double* A_lower = new double[matrix_size * matrix_size];

    // Allocate memory for MKL cholesky for the full matrix
    double* A_MKL = new double[matrix_size * matrix_size];

    // Memory allocation for tiled matrix
    double** Asplit = new double*[tot_tiles];

    for (int i = 0; i < tot_tiles; ++i) {
        // Buffer per tile
        Asplit[i] = new double[tile_size * tile_size];
    }

    //Generate a symmetric positve-definite matrix
    A = generate_positiveDefinitionMatrix(matrix_size);

    //printMatrix(A, mat_size_m);

    CBLAS_ORDER blasLay;
    int lapackLay;

    if (layRow) {
        blasLay = CblasRowMajor;
        lapackLay = LAPACK_ROW_MAJOR;
    } else {
        blasLay = CblasColMajor;
        lapackLay = LAPACK_COL_MAJOR;
    }

    //copying matrices into separate variables for tiled cholesky (A_lower) and MKL cholesky (A_MKL)
    std::memcpy(A_lower, A, matrix_size * matrix_size * sizeof(double));
    std::memcpy(A_MKL, A, matrix_size * matrix_size * sizeof(double));

    //splits the input matrix into tiles
    split_into_tiles(A_lower, Asplit, num_tiles, tile_size, matrix_size, layRow);

    //run the tiled Cholesky function
    tiled_cholesky(Asplit, tile_size, num_tiles, blasLay, lapackLay);

    //assembling seperated tiles back into full matrix
    assemble_tiles(Asplit, A_lower, num_tiles, tile_size, matrix_size, layRow);

    //calling LAPACKE_dpotrf cholesky for verification and timing comparison
    int info = LAPACKE_dpotrf(lapackLay, 'L', matrix_size, A_MKL, matrix_size);

    if (verifycorrectness == 1) {
        bool res = verify_results(A_lower, A_MKL, matrix_size * matrix_size);
        if (res) {
            fmt::print("Tiled Cholesky decomposition successful\n");
        } else {
            fmt::print("Tiled Cholesky decomposition failed\n");
        }
    }

    //memory free
    delete[] A;
    delete[] A_lower;
    delete[] A_MKL;
    for (int i = 0; i < tot_tiles; ++i) {
        delete[] Asplit[i];
    }

    return 0;
}