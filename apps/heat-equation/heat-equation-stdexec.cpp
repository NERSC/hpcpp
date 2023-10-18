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

#define HEQ_STDEXEC
#include "heat-equation.hpp"

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
  int nthreads = args.nthreads;
  Real_t dt = args.dt;
  Real_t alpha = args.alpha;
  std::string sched = args.sch;

  // init simulation time
  Real_t time = 0.0;

  // initialize dx, dy, dz
  std::vector<Real_t> ds(dims);
  // simulation setup (2D)
  std::vector<Real_t> grid_old((ncells + nghosts) *
                                              (ncells + nghosts));
  std::vector<Real_t> grid_new(ncells * ncells);

  // data pointers
  Real_t *dx = ds.data();
  Real_t *phi_old = grid_old.data();
  Real_t *phi_new = grid_new.data();

  // 2D jacobi algorithm pipeline
  auto algorithm = [&](auto sch) {

    auto phi_old_extent = ncells + nghosts;
    int gsize = ncells * ncells;

    // initialize dx on CPU
    for (int i = 0; i < dims; ++i)
      dx[i] = 1.0 / (ncells - 1);

    auto heat_eq_init = ex::transfer_just(sch, phi_old)
                        | ex::bulk(gsize, [=](int pos, auto phi_old) {
                          int i = 1 + (pos / ncells);
                          int j = 1 + (pos % ncells);

                          Real_t x = pos(i, ghost_cells, dx[0]);
                          Real_t y = pos(j, ghost_cells, dx[1]);

                          // L2 distance (r2 from origin)
                          Real_t r2 = (x * x + y * y) / (0.01);

                          // phi(x,y) = 1 + exp(-r^2)
                          phi_old[(i)*phi_old_extent + j] = 1 + exp(-r2);
                        });

    ex::sync_wait(std::move(heat_eq_init));

    if (args.print_grid)
      printGrid(phi_old, ncells + nghosts);

    // evolve the system
    for (auto step = 0; step < nsteps; step++) {
      static auto evolve = ex::transfer_just(sch, phi_old, phi_new, dx)
          | ex::bulk(phi_old_extent - nghosts,
                   [=](int pos, auto phi_old, auto phi_new, auto dx) {
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
                   [=](int pos, auto phi_old, auto phi_new, auto dx) {
                     int i = 1 + (pos / ncells);
                     int j = 1 + (pos % ncells);

                     // Jacobi iteration
                     phi_new[(i - 1) * ncells + j - 1] =
                         phi_old[(i)*phi_old_extent + j] +
                         alpha * dt *
                             ((phi_old[(i + 1) * phi_old_extent + j] -
                               2.0 * phi_old[(i)*phi_old_extent + j] +
                               phi_old[(i - 1) * phi_old_extent + j]) /
                                  (dx[0] * dx[0]) +
                              (phi_old[(i)*phi_old_extent + j + 1] -
                               2.0 * phi_old[(i)*phi_old_extent + j] +
                               phi_old[(i)*phi_old_extent + j - 1]) /
                                  (dx[1] * dx[1]));
                   }) |
          ex::bulk(gsize, [=](int pos, auto phi_old, auto phi_new, auto dx) {
            int i = 1 + (pos / ncells);
            int j = 1 + (pos % ncells);
            phi_old[(i)*phi_old_extent + j] = phi_new[(i - 1) * ncells + (j - 1)];
          });

      ex::sync_wait(std::move(evolve));

      // update the simulation time
      time += dt;
    }
    return;
  };

  // initialize stdexec scheduler
  heq_sch_t sch(sched);

  // init timer
  Timer timer;

  // launch with appropriate stdexec scheduler
  switch (sch()) {
    case sch_type_t::CPU:
      algorithm(exec::static_thread_pool(nthreads).get_scheduler());
      break;
    case sch_type_t::GPU:
      algorithm(nvexec::stream_context().get_scheduler());
      break;
    case sch_type_t::MULTIGPU:
      algorithm(nvexec::multi_gpu_stream_context().get_scheduler());
      break;
    default:
      std::cerr << "FATAL: " << sched << " is not a stdexec scheduler." << std::endl;
      std::cerr << "Run: heat-equation-stdexec --help to see the list of available schedulers" << std::endl;
      std::cerr << "Exiting..." << std::endl;
      exit(1);
  }

  auto elapsed = timer.stop();

  // print timing
  if (args.print_time) {
    std::cout << "Time: " << elapsed << " ms" << std::endl;
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