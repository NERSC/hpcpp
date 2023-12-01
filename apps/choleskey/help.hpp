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

// Function to read data from a text file and store it in a vector
template <typename T>
inline std::vector<T> readDataFromFile(const std::string& filename) {
  std::vector<T> data;

  // Open the file
  std::ifstream file(filename);

  // Check if the file is open successfully
  if (!file.is_open()) {
    std::cerr << "Failed to open the file: " << filename << std::endl;
    return data;  // Return an empty vector in case of failure
  }

  std::string line;
  while (std::getline(file, line)) {
    // Use std::istringstream to parse each line into doubles and store them in
    // the vector
    double value;
    std::istringstream iss(line);
    while (iss >> value) {
      data.push_back(value);
    }
  }

  // Close the file
  file.close();

  return data;
}

void split_into_blocks(double* mat, double* mat_split[], int num_tiles,
                       int tile_size, int size, bool layRow) {
  int itile, i, j, offset_tile;

  int tot_tiles = num_tiles * num_tiles;

  //#pragma omp parallel for private(i, j, offset_tile) schedule(auto)
  for (itile = 0; itile < tot_tiles; ++itile) {
    if (layRow) {
      offset_tile = int(itile / num_tiles) * num_tiles * tile_size * tile_size +
                    int(itile % num_tiles) * tile_size;
    } else {
      offset_tile = int(itile % num_tiles) * num_tiles * tile_size * tile_size +
                    int(itile / num_tiles) * tile_size;
    }

    for (i = 0; i < tile_size; ++i)
      //#pragma simd
      for (j = 0; j < tile_size; ++j) {
        mat_split[itile][i * tile_size + j] = mat[offset_tile + i * size + j];
      }
  }
}

void assemble(double* mat_split[], double* mat, int num_tiles, int tile_size,
              int size, bool layRow) {
  int i_tile, j_tile, tile, i, j, i_local, j_local;
  //#pragma omp parallel for private(j, i_local, j_local, i_tile, j_tile, tile) \
    schedule(auto)
  for (i = 0; i < size; ++i) {
    i_local = int(i % tile_size);
    i_tile = int(i / tile_size);
    //#pragma simd private(j_tile, tile, j_local)
    for (j = 0; j < size; ++j) {
      j_tile = int(j / tile_size);
      if (layRow) {
        tile = i_tile * num_tiles + j_tile;
      } else {
        tile = j_tile * num_tiles + i_tile;
      }
      j_local = int(j % tile_size);
      mat[i * size + j] = mat_split[tile][i_local * tile_size + j_local];
    }
  }
}

void copy_mat(double* A, double* B, int size) {
  int i, j;
  for (i = 0; i < size; ++i)
    for (j = 0; j < size; ++j) {
      B[i * size + j] = A[i * size + j];
    }
}

bool verify_results(double* act, double* ref, int totalsize) {
  double diff;

  bool res = true;
  for (int i = 0; i < totalsize; ++i) {
    diff = ref[i] - act[i];
    if (fabs(ref[i]) > 1e-5) {
      diff /= ref[i];
    }
    diff = fabs(diff);
    if (diff > 1.0e-5) {
      printf("\nError detected at i = %d: ref %g actual %g\n", i, ref[i],
             act[i]);
      res = false;
      break;
    }
  }
  return res;
}

void printMatrix(const double* matrix, size_t side_size) {
  for (size_t row = 0; row < side_size; ++row) {
    for (size_t col = 0; col <= row; ++col) {
      std::cout << matrix[row * side_size + col] << "\t";
    }
    std::cout << std::endl;
  }
}

void print_mat_split(double* mat_split[], int num_tiles, int tile_size) {
  for (int itile = 0; itile < num_tiles * num_tiles; ++itile) {
    printf("Block %d:\n", itile);
    for (int i = 0; i < tile_size; ++i) {
      for (int j = 0; j < tile_size; ++j) {
        printf("%f ", mat_split[itile][i * tile_size + j]);
      }
      printf("\n");
    }
    printf("\n");
  }
}
