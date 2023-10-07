/*
 * MIT License
 *
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
 * Simplified 2d heat equation example derived from amrex
 */

#include <cuda_runtime.h>

#include "heat-equation.hpp"

using namespace std;

// array to store PTM masses
__constant__ Real_t dx[2];

#define cudaErrorCheck(ans) check((ans), __FILE__, __LINE__)

// error checking function
template <typename T>
static inline void check(T result, const char* const file, const int line, bool is_fatal = true) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(result) << std::endl;

        if (is_fatal)
            exit(result);
    }
}

//
// initialize grid kernel
//
template <typename T>
__global__ void initialize(T* phi, int ncells, int ghost_cells) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int d_nghosts = nghosts;
    int phi_old_extent = ncells + d_nghosts;
    int gsize = ncells * ncells;

    for (; ind < gsize; ind += blockDim.x * gridDim.x) {
        int i = 1 + (ind / ncells);
        int j = 1 + (ind % ncells);

        Real_t x = pos(i, ghost_cells, dx[0]);
        Real_t y = pos(j, ghost_cells, dx[1]);

        // L2 distance (r2 from origin)
        Real_t r2 = (x * x + y * y) / (0.01);

        // phi(x,y) = 1 + exp(-r^2)
        phi[(i)*phi_old_extent + j] = 1 + exp(-r2);
    }
}

//
// fill boundary kernel
//
template <typename T>
__global__ void fillBoundary(T* phi_old, int ncells, int ghost_cells) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int d_nghosts = nghosts;
    int phi_old_extent = ncells + d_nghosts;
    int len = phi_old_extent;

    for (; pos < phi_old_extent - nghosts; pos += blockDim.x * gridDim.x) {
        int i = pos + ghost_cells;

        // fill boundary cells in phi_old
        phi_old[i] = phi_old[i + (ghost_cells * len)];

        phi_old[i + (len * (len - ghost_cells))] = phi_old[i + (len * (len - ghost_cells - 1))];

        phi_old[i * len] = phi_old[(ghost_cells * len) + i];

        phi_old[(len - ghost_cells) + (len * i)] = phi_old[(len - ghost_cells - 1) + (len * i)];
    }
}

//
// jacobi 2d stencil kernel
//
template <typename T>
__global__ void jacobi(T* phi_old, T* phi_new, int ncells, Real_t alpha, Real_t dt) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int d_nghosts = nghosts;
    int phi_old_extent = ncells + d_nghosts;
    int gsize = ncells * ncells;

    for (; pos < gsize; pos += blockDim.x * gridDim.x) {
        int i = 1 + (pos / ncells);
        int j = 1 + (pos % ncells);

        // Jacobi iteration
        phi_new[(i - 1) * ncells + j - 1] =
            phi_old[(i)*phi_old_extent + j] +
            alpha * dt *

                ((phi_old[(i + 1) * phi_old_extent + j] - 2.0 * phi_old[(i)*phi_old_extent + j] +
                  phi_old[(i - 1) * phi_old_extent + j]) /
                     (dx[0] * dx[0]) +

                 (phi_old[(i)*phi_old_extent + j + 1] - 2.0 * phi_old[(i)*phi_old_extent + j] +
                  phi_old[(i)*phi_old_extent + j - 1]) /
                     (dx[1] * dx[1]));
    }
}

//
// parallelCopy kernel
//
template <typename T>
__global__ void parallelCopy(T* phi_old, T* phi_new, int ncells) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    int d_nghosts = nghosts;
    int phi_old_extent = ncells + d_nghosts;
    int gsize = ncells * ncells;

    for (; pos < gsize; pos += blockDim.x * gridDim.x) {
        int i = 1 + (pos / ncells);
        int j = 1 + (pos % ncells);
        phi_old[(i)*phi_old_extent + j] = phi_new[(i - 1) * ncells + (j - 1)];
    }
}

//
// main simulation
//
int main(int argc, char* argv[]) {
    // parse params
    heat_params_t args = argparse::parse<heat_params_t>(argc, argv);

    // see if help wanted
    if (args.help) {
        args.print();  // prints all variables
        return 0;
    }

    // simulation variables
    int ncells = args.ncells;
    int nsteps = args.nsteps;
    Real_t dt = args.dt;
    Real_t alpha = args.alpha;

    // init simulation time
    Real_t time = 0.0;

    // initialize dx, dy, dz
    Real_t h_dx[dims];
    for (int i = 0; i < dims; ++i)
        h_dx[i] = 1.0 / (ncells - 1);

    cudaErrorCheck(cudaMemcpyToSymbol(dx, h_dx, sizeof(Real_t) * dims));

    // grid size
    int gsize = ncells * ncells;

    // host memory for printing
    Real_t* h_phi = nullptr;

    // simulation setup (2D)
    Real_t* phi_old = nullptr;
    Real_t* phi_new = nullptr;

    cudaErrorCheck(cudaMalloc(&phi_old, sizeof(Real_t) * ((ncells + nghosts) * (ncells + nghosts))));
    cudaErrorCheck(cudaMalloc(&phi_new, sizeof(Real_t) * ((ncells) * (ncells))));

    // setup grid
    int blockSize = std::min(1024, gsize);  // let's do at most 1024 threads.
    int nBlocks = (gsize + blockSize - 1) / blockSize;

    Timer timer;

    // initialize grid
    initialize<<<nBlocks, blockSize>>>(phi_old, ncells, ghost_cells);

    cudaErrorCheck(cudaDeviceSynchronize());

    // print initial grid if needed
    if (args.print_grid) {
        // copy initial grid to host
        h_phi = new Real_t[(ncells + nghosts) * (ncells + nghosts)];
        cudaErrorCheck(cudaMemcpy(h_phi, phi_old, sizeof(Real_t) * (ncells + nghosts) * (ncells + nghosts),
                                  cudaMemcpyDeviceToHost));

        printGrid(h_phi, ncells + nghosts);
    }

    // evolve the system
    for (auto step = 0; step < nsteps; step++) {
        static int fBblock = std::min(1024, ncells);              // let's do at most 1024 threads.
        static int fBnBlocks = (ncells + fBblock - 1) / fBblock;  // fillBoundary blocks

        // fillboundary
        fillBoundary<<<fBnBlocks, fBblock>>>(phi_old, ncells, ghost_cells);

        // jacobi
        jacobi<<<nBlocks, blockSize>>>(phi_old, phi_new, ncells, alpha, dt);

        // parallelCopy
        parallelCopy<<<nBlocks, blockSize>>>(phi_old, phi_new, ncells);

        cudaErrorCheck(cudaDeviceSynchronize());

        // update time
        time += dt;
    }

    auto elapsed = timer.stop();

    // print timing
    if (args.print_time) {
        std::cout << "Time: " << elapsed << " ms" << std::endl;
    }

    // print final grid if needed
    if (args.print_grid) {
        cudaErrorCheck(cudaMemcpy(h_phi, phi_new, sizeof(Real_t) * gsize, cudaMemcpyDeviceToHost));
        printGrid(h_phi, ncells);

        // free host memory
        delete[] h_phi;
        h_phi = nullptr;
    }

    // free device memory
    cudaErrorCheck(cudaFree(phi_old));
    cudaErrorCheck(cudaFree(phi_new));

    return 0;
}
