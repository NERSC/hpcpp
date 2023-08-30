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

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include <nvexec/multi_gpu_context.cuh>
#include <span>
#include <stdexec/execution.hpp>

#include "heat-equation.hpp"

namespace ex = stdexec;
using namespace nvexec;

//
// simulation
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
  // future if needed to split in multiple grids
  // int max_grid_size = args.max_grid_size;

  // init simulation time
  Real_t time = 0.0;

  // initialize dx, dy, dz
  thrust::universal_vector<Real_t> dx(dims);
  for (int i = 0; i < dims; ++i)
    dx[i] = 1.0 / (ncells - 1);

  // simulation setup (2D)
  thrust::universal_vector<Real_t> grid_old((ncells + nghosts) *
                                            (ncells + nghosts));
  thrust::universal_vector<Real_t> grid_new(ncells * ncells);

  // initialize grid
  auto phi_old = thrust::raw_pointer_cast(grid_old.data());
  auto phi_new = thrust::raw_pointer_cast(grid_new.data());

  // scheduler from gpu
  nvexec::multi_gpu_stream_context stream_ctx{};
  auto gpu = stream_ctx.get_scheduler();

  auto dx_span = std::span{thrust::raw_pointer_cast(dx.data()),
                           thrust::raw_pointer_cast(dx.data()) + dx.size()};
  auto phi_old_span = std::span{phi_old, phi_old + grid_old.size()};
  auto phi_new_span = std::span{phi_new, phi_new + grid_new.size()};
  auto phi_old_extent = ncells + nghosts;

  int gsize = ncells * ncells;
  auto heat_eq_init = ex::transfer_just(gpu, dx_span, phi_old_span) |
                      ex::bulk(gsize, [=](int pos, auto ds, auto phi) {
                        int i = 1 + (pos / ncells);
                        int j = 1 + (pos % ncells);

                        Real_t x = pos(i, ghost_cells, ds[0]);
                        Real_t y = pos(j, ghost_cells, ds[1]);

                        // L2 distance (r2 from origin)
                        Real_t r2 = (x * x + y * y) / (0.01);

                        // phi(x,y) = 1 + exp(-r^2)
                        phi[(i)*phi_old_extent + j] = 1 + exp(-r2);
                      });

  ex::sync_wait(std::move(heat_eq_init));
  if (args.print_grid)
    printGrid(phi_old, ncells + nghosts);

  auto tx = ex::transfer_just(gpu, dx_span, phi_old_span, phi_new_span);

  // evolve the system
  for (auto step = 0; step < nsteps; step++) {
    static auto evolve =
        tx |
        ex::bulk(phi_old_extent - nghosts,
                 [=](int pos, auto ds, auto phi_old, auto phi_new) {
                   int i = pos + ghost_cells;
                   int len = phi_old_extent;
                   // fill boundary cells in old_phi
                   phi_old[i] = phi_old[i + (ghost_cells * len)];
                   phi_old[i + (len * (len - ghost_cells))] =
                       phi_old[i + (len * (len - ghost_cells - 1))];
                   phi_old[i * len] = phi_old[(ghost_cells * len) + i];
                   phi_old[(len - ghost_cells) + (len * i)] =
                       phi_old[(len - ghost_cells - 1) + (len * i)];
                 }) |
        ex::bulk(gsize,
                 [=](int pos, auto ds, auto phi_old, auto phi_new) {
                   int i = 1 + (pos / ncells);
                   int j = 1 + (pos % ncells);

                   // Jacobi iteration
                   phi_new[(i - 1) * ncells + j - 1] =
                       phi_old[(i)*phi_old_extent + j] +
                       alpha * dt *
                           ((phi_old[(i + 1) * phi_old_extent + j] -
                             2.0 * phi_old[(i)*phi_old_extent + j] +
                             phi_old[(i - 1) * phi_old_extent + j]) /
                                (ds[0] * ds[0]) +
                            (phi_old[(i)*phi_old_extent + j + 1] -
                             2.0 * phi_old[(i)*phi_old_extent + j] +
                             phi_old[(i)*phi_old_extent + j - 1]) /
                                (ds[1] * ds[1]));
                 }) |
        ex::bulk(gsize, [=](int pos, auto ds, auto phi_old, auto phi_new) {
          int i = 1 + (pos / ncells);
          int j = 1 + (pos % ncells);
          phi_old[(i)*phi_old_extent + j] = phi_new[(i - 1) * ncells + (j - 1)];
        });

    ex::sync_wait(std::move(evolve));

    // update the simulation time
    time += dt;
  }

  auto finalize = ex::then(ex::just(), [&]() {
    if (args.print_grid)
      // print the final grid
      printGrid(phi_new, ncells);
  });

  // end the simulation
  ex::sync_wait(std::move(finalize));

  return 0;
}