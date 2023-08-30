/*
 * MIT License
 *
 * Copyright (c) 2023 Weile Wei 
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
//
// This example provides a stdpar implementation for the 1D stencil code.

#include <experimental/mdspan>

#include "argparse/argparse.hpp"
#include "commons.hpp"

// parameters
struct args_params_t : public argparse::Args {
  bool& results = kwarg("results", "print generated results (default: false)")
                      .set_default(false);
  std::uint64_t& nx =
      kwarg("nx", "Local x dimension (of each partition)").set_default(10);
  std::uint64_t& nt = kwarg("nt", "Number of time steps").set_default(45);
  std::uint64_t& np = kwarg("np", "Number of partitions").set_default(10);
  bool& k = kwarg("k", "Heat transfer coefficient").set_default(0.5);
  double& dt = kwarg("dt", "Timestep unit (default: 1.0[s])").set_default(1.0);
  double& dx = kwarg("dx", "Local x dimension").set_default(1.0);
  bool& no_header =
      kwarg("no-header", "Do not print csv header row (default: false)")
          .set_default(false);
  bool& help = flag("h, help", "print help");
  bool& time = kwarg("t, time", "print time").set_default(true);
};

///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true;  // print csv heading
double k = 0.5;      // heat transfer coefficient
double dt = 1.;      // time step
double dx = 1.;      // grid spacing

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper {
  // Our partition type
  typedef double partition;

  // Our data for one time step
  using view_1d = std::extents<int, std::dynamic_extent>;
  typedef std::mdspan<partition, view_1d, std::layout_right> space;

  // Our operator
  double heat(double left, double middle, double right, const double k = ::k,
              const double dt = ::dt, const double dx = ::dx) {
    return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
  }

  inline std::size_t idx(std::size_t id, int dir, std::size_t size) {
    if (id == 0 && dir == -1) {
      return size - 1;
    }

    if (id == size - 1 && dir == +1) {
      return (std::size_t)0;
    }
    assert(id < size);

    return id + dir;
  }

  // do all the work on 'nx' data points for 'nt' time steps
  space do_work(std::size_t np, std::size_t nx, std::size_t nt) {
    std::size_t size = np * nx;
    partition* current_ptr = new partition[size];
    partition* next_ptr = new partition[size];

    auto current = space(current_ptr, size);
    auto next = space(next_ptr, size);
    // parallel init
    std::for_each_n(std::execution::par, counting_iterator(0), np * nx,
                    [=](std::size_t i) { current_ptr[i] = (double)i; });

    // Actual time step loop
    for (std::size_t t = 0; t != nt; ++t) {
      std::for_each_n(std::execution::par, counting_iterator(0), np * nx,
                      [=, k = k, dt = dt, dx = dx](int32_t i) {
                        auto left = idx(i, -1, size);
                        auto right = idx(i, +1, size);
                        next[i] = heat(current[left], current[i],
                                       current[right], k, dt, dx);
                      });
      std::swap(current, next);
    }

    return current;
  }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {
  std::uint64_t np = args.np;  // Number of partitions.
  std::uint64_t nx = args.nx;  // Number of grid points.
  std::uint64_t nt = args.nt;  // Number of steps.

  // Create the stepper object
  stepper step;

  // Measure execution time.
  Timer timer;

  // Execute nt time steps on nx grid points.
  auto solution = step.do_work(np, nx, nt);
  auto time = timer.stop();

  // Print the final solution
  if (args.results) {
    for (std::size_t i = 0; i != np; ++i) {
      std::cout << "U[" << i << "] = {";
      for (std::size_t j = 0; j != nx; ++j) {
        std::cout << solution[i * nx + j] << " ";
      }
      std::cout << "}\n";
    }
  }

  if (args.time) {
    std::cout << "Duration: " << time << " ms."
              << "\n";
  }

  return 0;
}

int main(int argc, char* argv[]) {
  // parse params
  args_params_t args = argparse::parse<args_params_t>(argc, argv);
  // see if help wanted
  if (args.help) {
    args.print();  // prints all variables
    return 0;
  }

  benchmark(args);

  return 0;
}
