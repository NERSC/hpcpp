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
#include "repeat_n/repeat_n.cuh"

// 2D jacobi algorithm pipeline
void heat_equation(scheduler auto sch, Real_t* phi_old, Real_t* phi_new, Real_t* dx, Real_t dt, Real_t alpha,
                   int nsteps, int ncells, bool print = false) {
    // init simulation time
    Real_t time = 0.0;
    auto phi_old_extent = ncells + nghosts;
    int gsize = ncells * ncells;

    // initialize dx on CPU
    for (int i = 0; i < dims; ++i)
        dx[i] = 1.0 / (ncells - 1);

    // set cout precision
    fmt::print("HEQ progress: ");

    ex::sender auto begin = schedule(sch);

    auto heat_eq_init = ex::bulk(begin, gsize, [=](int pos) {
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

    if (print)
        printGrid(phi_old, ncells + nghosts);

    auto fillBoundary = [=](int pos) {
        int i = pos + ghost_cells;
        int len = phi_old_extent;
        // fill boundary cells in old_phi
        phi_old[i] = phi_old[i + (ghost_cells * len)];
        phi_old[i + (len * (len - ghost_cells))] = phi_old[i + (len * (len - ghost_cells - 1))];
        phi_old[i * len] = phi_old[(ghost_cells * len) + i];
        phi_old[(len - ghost_cells) + (len * i)] = phi_old[(len - ghost_cells - 1) + (len * i)];
    };

    auto jacobi = [=](int pos) {
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
    };

    auto parallelCopy = [=](int pos) {
        int i = 1 + (pos / ncells);
        int j = 1 + (pos % ncells);
        phi_old[(i)*phi_old_extent + j] = phi_new[(i - 1) * ncells + (j - 1)];
    };

    // evolve the system
#if !defined(USE_GPU)
    for (auto iter = 0; iter < nsteps; iter++)
#endif
        stdexec::sync_wait(
#if defined(USE_GPU)
            ex::just() | exec::on(sch, repeat_n(nsteps,
#else
  stdexec::schedule(sch) |
#endif  // USE_GPU
                                                ex::bulk(phi_old_extent - nghosts, [=](int k) { fillBoundary(k); }) |
                                                    ex::bulk(gsize, [=](int k) { jacobi(k); }) |
                                                    ex::bulk(gsize, [=](int k) { parallelCopy(k); })
#if defined(USE_GPU)
                                                    ))
#endif  // USE_GPU
        );

    // update the simulation time
    time += nsteps * dt;

    // print final progress mark
    fmt::print("100% \n");

    return;
}

//
// simulation
//
int main(int argc, char* argv[]) {
    // parse params
    const heat_params_t args = argparse::parse<heat_params_t>(argc, argv);

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

    // initialize dx, dy, dz
    std::vector<Real_t> ds(dims);
    // simulation setup (2D)
    std::vector<Real_t> grid_old((ncells + nghosts) * (ncells + nghosts));
    std::vector<Real_t> grid_new(ncells * ncells);

    // data pointers
    Real_t* dx = ds.data();
    Real_t* phi_old = grid_old.data();
    Real_t* phi_new = grid_new.data();

    // initialize stdexec scheduler
    sch_t scheduler = get_sch_enum(sched);

    // init timer
    Timer timer;

    // launch with appropriate stdexec scheduler
    switch (scheduler) {
        case sch_t::CPU:
            heat_equation(exec::static_thread_pool{nthreads}.get_scheduler(), phi_old, phi_new, dx, dt, alpha, nsteps,
                          ncells, args.print_grid);
            break;
#if defined(USE_GPU)
        case sch_t::GPU:
            heat_equation(nvexec::stream_context().get_scheduler(), phi_old, phi_new, dx, dt, alpha, nsteps, ncells,
                          args.print_grid);
            break;
        case sch_t::MULTIGPU:
            heat_equation(nvexec::multi_gpu_stream_context().get_scheduler(), phi_old, phi_new, dx, dt, alpha, nsteps,
                          ncells, args.print_grid);
            break;
#endif  // USE_GPU
        default:
            throw std::runtime_error("Run: `heat-equation-stdexec --help` to see the list of available schedulers");
    }

    auto elapsed = timer.stop();

    // print timing
    if (args.print_time) {
        fmt::print("Duration: {:f} ms\n", elapsed);
    }

    if (args.print_grid)
        // print the final grid
        printGrid(phi_new, ncells);

    return 0;
}