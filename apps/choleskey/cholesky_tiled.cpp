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
#include "help.hpp"

using data_type = double;

//#include "time.hpp"
#define SWITCH_CHAR '-'

double* dpo_generate(size_t side_size) {
    unsigned int seed = side_size;
#ifdef _WIN32
    srand(seed);
#endif
    // M is a (very) pseudo-random symmetric matrix
    double* M_matrix = new double[side_size * side_size];
    for (size_t row = 0; row < side_size; ++row) {
        for (size_t col = row; col < side_size; ++col) {
            M_matrix[col * side_size + row] = M_matrix[row * side_size + col] = (double)
#ifdef _WIN32
                                                                                    rand()
#else
                                                                                    rand_r(&seed)
#endif
                                                                                / RAND_MAX;
        }
    }
    double* ret_matrix = (double*)malloc(side_size * side_size * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, side_size, side_size, side_size, 1.0, M_matrix, side_size,
                M_matrix, side_size, 0.0, ret_matrix, side_size);

    //adjust diagonals (diag = sum (row entries) + 1.0)
    for (size_t row = 0; row < side_size; ++row) {
        double diag = 1.0;  //start from 1.0
        for (size_t col = 0; col < side_size; ++col) {
            diag += ret_matrix[row * side_size + col];
        }
        //set the diag entry
        ret_matrix[row * side_size + row] = diag;
    }

    delete[] M_matrix;
    return ret_matrix;
}

void tiled_cholesky(double* mat_sp[], int tile_size, int num_tiles, CBLAS_ORDER blasLay, int lapackLay) {
    unsigned int m, n, k;
    int info;

    for (k = 0; k < num_tiles; ++k) {

        //POTRF - MKL call
        info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', tile_size, mat_sp[k * num_tiles + k], tile_size);
        std::cout << "info = " << info << std::endl;

        for (m = k + 1; m < num_tiles; ++m) {

            //DTRSM - MKL call
            cblas_dtrsm(blasLay, CblasRight, CblasLower, CblasTrans, CblasNonUnit, tile_size, tile_size, 1.0,
                        mat_sp[k * num_tiles + k], tile_size, mat_sp[m * num_tiles + k], tile_size);
        }

        for (n = k + 1; n < num_tiles; ++n) {
            //DSYRK - MKL call
            cblas_dsyrk(blasLay, CblasLower, CblasNoTrans, tile_size, tile_size, -1.0, mat_sp[n * num_tiles + k],
                        tile_size, 1.0, mat_sp[n * num_tiles + n], tile_size);

            for (m = n + 1; m < num_tiles; ++m) {

                //DGEMM - MKL call
                cblas_dgemm(blasLay, CblasNoTrans, CblasTrans, tile_size, tile_size, tile_size, -1.0,
                            mat_sp[m * num_tiles + k], tile_size, mat_sp[n * num_tiles + k], tile_size, 1.0,
                            mat_sp[m * num_tiles + n], tile_size);
            }
        }
    }
}

int main(int argc, char** argv) {
    int info;

    int mat_size_m, num_tiles, tile_size, tot_tiles;
    mat_size_m = 4;  // must be an input
    num_tiles = 2;   // matrix size MUST be divisible

    bool layRow = true;
    int verify = 1;

    std::cout << "mat_size = " << mat_size_m << ", num_tiles = " << num_tiles << std::endl << std::endl;

    // Check that mat_size is divisible by num_tiles
    if (mat_size_m % num_tiles != 0) {
        std::cout << "matrix size MUST be divisible by num_tiles.. aborting" << std::endl;
        throw std::invalid_argument("Matrix size is not divisible by num_tiles");
    }

    if (mat_size_m == 0) {
        printf("mat_size_m is not defined\n");
        exit(0);
    }

    tile_size = mat_size_m / num_tiles;
    tot_tiles = num_tiles * num_tiles;

    //allocating memory for input matrix (full matrix)
    double* A = (double*)malloc(mat_size_m * mat_size_m * sizeof(double));

    //allocating memory for tiled_cholesky for the full matrix
    double* A_lower = (double*)malloc(mat_size_m * mat_size_m * sizeof(double));

    //allocating memory for MKL cholesky for the full matrix
    double* A_MKL = (double*)malloc(mat_size_m * mat_size_m * sizeof(double));

    //memory allocation for tiled matrix
    double** Asplit = new double*[tot_tiles];

    for (int i = 0; i < tot_tiles; ++i) {
        //Buffer per tile
        Asplit[i] = (double*)malloc(tile_size * tile_size * sizeof(double));
    }

    //Generate a symmetric positve-definite matrix
    A = dpo_generate(mat_size_m);

    printMatrix(A, mat_size_m);

    CBLAS_ORDER blasLay;
    int lapackLay;

    if (layRow) {
        blasLay = CblasRowMajor;
        lapackLay = LAPACK_ROW_MAJOR;
    } else {
        blasLay = CblasColMajor;
        lapackLay = LAPACK_COL_MAJOR;
    }

    //copying matrices into separate variables for tiled cholesky (A_my)
    //and MKL cholesky (A_MKL)
    //The output overwrites the matrices
    std::cout << "A_my = " << std::endl;
    copy_mat(A, A_lower, mat_size_m);
    copy_mat(A, A_MKL, mat_size_m);
    printMatrix(A_lower, mat_size_m);

    //This splits the input matrix into tiles (or blocks)
    split_into_blocks(A_lower, Asplit, num_tiles, tile_size, mat_size_m, layRow);
    std::cout << "split matrix: " << std::endl;
    print_mat_split(Asplit, num_tiles, tile_size);

    //Calling the tiled Cholesky function. This does the factorization of the full matrix using a tiled implementation.
    tiled_cholesky(Asplit, tile_size, num_tiles, blasLay, lapackLay);

    //assembling of tiles back into full matrix
    assemble(Asplit, A_lower, num_tiles, tile_size, mat_size_m, layRow);
    std::cout << "Cholosky decomposition: lower matrix_A_lower \n";
    printMatrix(A_lower, mat_size_m);
    //tbegin = dtimeGet();

    //calling mkl cholesky for verification and timing comparison
    info = LAPACKE_dpotrf(lapackLay, 'L', mat_size_m, A_MKL, mat_size_m);
    std::cout << "A_MKL \n";
    printMatrix(A_MKL, mat_size_m);

    if (verify == 1) {
        bool res = verify_results(A_lower, A_MKL, mat_size_m * mat_size_m);
        if (res == true) {
            printf("Tiled Cholesky successful\n");
        } else {
            printf("Tiled Chloesky failed\n");
        }
    }

    //free
    free(A);
    free(A_lower);
    free(A_MKL);
    for (int i = 0; i < tot_tiles; ++i) {
        free(Asplit[i]);
    }

    return 0;
}