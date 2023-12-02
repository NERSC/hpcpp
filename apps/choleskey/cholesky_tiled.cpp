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
#include <experimental/linalg>
#include <iostream>
#include <utility>
#include <vector>
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
    int mat_size_m, num_tiles, tile_size, tot_tiles;
    mat_size_m = 4;  // must be an input
    num_tiles = 2;   // matrix size MUST be divisible
    int verify = 1;
    bool layRow = true;

    std::cout << "mat_size = " << mat_size_m << ", num_tiles = " << num_tiles << std::endl << std::endl;

    // Check that mat_size is divisible by num_tiles
    if (mat_size_m % num_tiles != 0) {
        std::cout << "matrix size must be divisible by num_tiles.. aborting" << std::endl;
        throw std::invalid_argument("Matrix size is not divisible by num_tiles");
    }

    if (mat_size_m == 0) {
        printf("mat_size_m is not defined\n");
        exit(0);
    }

    tile_size = mat_size_m / num_tiles;
    tot_tiles = num_tiles * num_tiles;

    double* A = new double[mat_size_m * mat_size_m];

    // Allocate memory for tiled_cholesky for the full matrix
    double* A_lower = new double[mat_size_m * mat_size_m];

    // Allocate memory for MKL cholesky for the full matrix
    double* A_MKL = new double[mat_size_m * mat_size_m];

    // Memory allocation for tiled matrix
    double** Asplit = new double*[tot_tiles];

    for (int i = 0; i < tot_tiles; ++i) {
        // Buffer per tile
        Asplit[i] = new double[tile_size * tile_size];
    }

    //Generate a symmetric positve-definite matrix
    A = generate_positiveDefinitionMatrix(mat_size_m);

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

    //copying matrices into separate variables for tiled cholesky (A_lower)
    //and MKL cholesky (A_MKL)
    //The output overwrites the matrices
    copy_matrix(A, A_lower, mat_size_m);
    copy_matrix(A, A_MKL, mat_size_m);

    //splits the input matrix into tiles
    split_into_tiles(A_lower, Asplit, num_tiles, tile_size, mat_size_m, layRow);

    //run the tiled Cholesky function
    tiled_cholesky(Asplit, tile_size, num_tiles, blasLay, lapackLay);

    //assembling seperated tiles back into full matrix
    assemble_tiles(Asplit, A_lower, num_tiles, tile_size, mat_size_m, layRow);

    //calling LAPACKE_dpotrf cholesky for verification and timing comparison
    int info = LAPACKE_dpotrf(lapackLay, 'L', mat_size_m, A_MKL, mat_size_m);

    if (verify == 1) {
        bool res = verify_results(A_lower, A_MKL, mat_size_m * mat_size_m);
        if (res == true) {
            printf("Tiled Cholesky decomposition successful\n");
        } else {
            printf("Tiled Chloesky decomposition failed\n");
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