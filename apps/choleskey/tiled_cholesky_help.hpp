#pragma once

#include <cblas.h>
#include <lapacke.h>
#include <stddef.h>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

double* generate_positiveDefinitionMatrix(const size_t matrix_size) {
    double* A_matrix = new double[matrix_size * matrix_size];
    double* pd_matrix = (double*)malloc(matrix_size * matrix_size * sizeof(double));
    unsigned int seeds = matrix_size;

    // generate a random symmetric matrix
    for (size_t row = 0; row < matrix_size; ++row) {
        for (size_t col = row; col < matrix_size; ++col) {
            A_matrix[col * matrix_size + row] = A_matrix[row * matrix_size + col] = (double)rand_r(&seeds) / RAND_MAX;
        }
    }
    // compute the product of matrix A_matrix and its transpose, and storing the result in pd_matrix.
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, matrix_size, matrix_size, matrix_size, 1.0, A_matrix,
                matrix_size, A_matrix, matrix_size, 0.0, pd_matrix, matrix_size);

    // Adjust Diagonals
    for (size_t row = 0; row < matrix_size; ++row) {
        double diagonals = 1.0;  // from 1.0
        for (size_t col = 0; col < matrix_size; ++col) {
            diagonals += pd_matrix[row * matrix_size + col];
        }
        // Set the diag entry
        pd_matrix[row * matrix_size + row] = diagonals;
    }

    delete[] A_matrix;
    return pd_matrix;
}

void split_into_tiles(const double* matrix, double* matrix_split[], const int num_tiles, const int tile_size,
                      const int size, bool layRow) {

    int total_num_tiles = num_tiles * num_tiles;
    int offset_tile;

    //#pragma omp parallel for private(i, j, offset_tile) schedule(auto)
    for (int i_tile = 0; i_tile < total_num_tiles; ++i_tile) {
        if (layRow) {
            offset_tile =
                int(i_tile / num_tiles) * num_tiles * tile_size * tile_size + int(i_tile % num_tiles) * tile_size;
        } else {
            offset_tile =
                int(i_tile % num_tiles) * num_tiles * tile_size * tile_size + int(i_tile / num_tiles) * tile_size;
        }

        for (int i = 0; i < tile_size; ++i)
            //#pragma simd
            for (int j = 0; j < tile_size; ++j) {
                matrix_split[i_tile][i * tile_size + j] = matrix[offset_tile + i * size + j];
            }
    }
}

void assemble_tiles(double* matrix_split[], double* matrix, const int num_tiles, const int tile_size, const int size,
                    bool layRow) {
    int i_tile, j_tile, tile, i_local, j_local;
    //#pragma omp parallel for private(j, i_local, j_local, i_tile, j_tile, tile) \
    schedule(auto)
    for (int i = 0; i < size; ++i) {
        i_local = int(i % tile_size);
        i_tile = int(i / tile_size);
        //#pragma simd private(j_tile, tile, j_local)
        for (int j = 0; j < size; ++j) {
            j_tile = int(j / tile_size);
            if (layRow) {
                tile = i_tile * num_tiles + j_tile;
            } else {
                tile = j_tile * num_tiles + i_tile;
            }
            j_local = int(j % tile_size);
            matrix[i * size + j] = matrix_split[tile][i_local * tile_size + j_local];
        }
    }
}

void copy_matrix(const double* matrix, double* destination, const int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            destination[i * size + j] = matrix[i * size + j];
        }
}

bool verify_results(const double* lower_res, const double* dporft_res, const int totalsize) {
    bool res = true;
    double diff;
    for (int i = 0; i < totalsize; ++i) {
        diff = dporft_res[i] - lower_res[i];
        if (fabs(dporft_res[i]) > 1e-5) {
            diff /= dporft_res[i];
        }
        diff = fabs(diff);
        if (diff > 1.0e-5) {
            printf("\nError detected at i = %d: ref %g actual %g\n", i, dporft_res[i], lower_res[i]);
            res = false;
            break;
        }
    }
    return res;
}

void printMatrix(const double* matrix, size_t matrix_size) {
    for (size_t row = 0; row < matrix_size; ++row) {
        for (size_t col = 0; col <= row; ++col) {
            std::cout << matrix[row * matrix_size + col] << "\t";
        }
        std::cout << std::endl;
    }
}

void print_mat_split(double* matrix_split[], int num_tiles, int tile_size) {
    for (int itile = 0; itile < num_tiles * num_tiles; ++itile) {
        printf("Block %d:\n", itile);
        for (int i = 0; i < tile_size; ++i) {
            for (int j = 0; j < tile_size; ++j) {
                printf("%f ", matrix_split[itile][i * tile_size + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
