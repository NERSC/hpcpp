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

#include <exec/any_sender_of.hpp>
#include <exec/static_thread_pool.hpp>
#include <experimental/mdspan>
#include <stdexec/execution.hpp>

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

template <class... Ts>
using any_sender_of = typename exec::any_receiver_ref<
    stdexec::completion_signatures<Ts...>>::template any_sender<>;

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper {
  // Our partition type
  typedef double partition;

  // Our data for one time step
  using view_1d = std::extents<int, std::dynamic_extent>;
  typedef std::mdspan<partition, view_1d, std::layout_right> space;

  using any_space_sender =
      any_sender_of<stdexec::set_value_t(space), stdexec::set_stopped_t(),
                    stdexec::set_error_t(std::exception_ptr)>;

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

  partition* current_ptr = nullptr;
  partition* next_ptr = nullptr;
  space current;
  space next;

  // do all the work on 'nx' data points for 'nt' time steps
  auto do_work(std::size_t np, std::size_t nx, std::size_t nt)
      -> any_space_sender {
    if (nt == 0) {
      std::size_t size = np * nx;
      partition* current_ptr = new partition[size];
      partition* next_ptr = new partition[size];
      current = space(current_ptr, size);
      next = space(next_ptr, size);

      // parallel init
      std::for_each_n(std::execution::par, counting_iterator(0), np * nx,
                      [=](std::size_t i) { current_ptr[i] = (double)i; });

      return stdexec::just(current);
    }

    return stdexec::just(nt - 1) |
           stdexec::let_value([=](std::size_t nt_updated) {
             return do_work(np, nx, nt_updated);
           }) |
           stdexec::bulk(np,
                         [&, k = k, dt = dt, dx = dx, nx = nx, np = np](
                             std::size_t i, auto const& current) {
                           std::for_each_n(
                               std::execution::par, counting_iterator(0), nx,
                               [=, next = next](std::size_t j) {
                                 std::size_t id = i * nx + j;
                                 auto left = idx(id, -1, np * nx);
                                 auto right = idx(id, +1, np * nx);
                                 next[id] = heat(current[left], current[id],
                                                 current[right], k, dt, dx);
                               });
                         }) |
           stdexec::then([&](auto current) {
             // TODO: return next?
             std::swap(current, next);
             return current;
           });
  }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {
  std::uint64_t np = args.np;  // Number of partitions.
  std::uint64_t nx = args.nx;  // Number of grid points.
  std::uint64_t nt = args.nt;  // Number of steps.

  // Create the stepper object
  stepper step;

  exec::static_thread_pool pool(np);
  stdexec::scheduler auto sch = pool.get_scheduler();
  stdexec::sender auto begin = stdexec::schedule(sch);

  // Measure execution time.
  Timer timer;

  // Execute nt time steps on nx grid points.
  stdexec::sender auto sender =
      begin | stdexec::then([=]() { return nt; }) |
      stdexec::let_value(
          [=, &step](std::uint64_t nt) { return step.do_work(np, nx, nt); });

  auto [solution] = stdexec::sync_wait(std::move(sender)).value();

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
