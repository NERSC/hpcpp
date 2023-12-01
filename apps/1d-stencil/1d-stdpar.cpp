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
#include "argparse/argparse.hpp"
#include "commons.hpp"

// parameters
struct args_params_t : public argparse::Args {
    bool& results = kwarg("results", "print generated results (default: false)").set_default(false);
    std::uint64_t& nt = kwarg("nt", "Number of time steps").set_default(45);
    std::uint64_t& size = kwarg("size", "Number of elements").set_default(10);
    bool& k = kwarg("k", "Heat transfer coefficient").set_default(0.5);
    double& dt = kwarg("dt", "Timestep unit (default: 1.0[s])").set_default(1.0);
    double& dx = kwarg("dx", "Local x dimension").set_default(1.0);
    bool& help = flag("h, help", "print help");
    bool& time = kwarg("t, time", "print time").set_default(true);
};

using Real_t = double;
///////////////////////////////////////////////////////////////////////////////
// Command-line variables
constexpr Real_t k = 0.5;  // heat transfer coefficient
constexpr Real_t dt = 1.;  // time step
constexpr Real_t dx = 1.;  // grid spacing

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper {
    using view_1d = std::extents<int, std::dynamic_extent>;
    typedef std::mdspan<Real_t, view_1d, std::layout_right> space;

    // Our operator
    [[nodiscard]] Real_t heat(const Real_t left, const Real_t middle, const Real_t right, const Real_t k = ::k,
                              const Real_t dt = ::dt, const Real_t dx = ::dx) {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // do all the work on 'size' data points for 'nt' time steps
    [[nodiscard]] space do_work(const std::size_t size, const std::size_t nt) {
        Real_t* current_ptr = new Real_t[size];
        Real_t* next_ptr = new Real_t[size];

        auto current = space(current_ptr, size);
        auto next = space(next_ptr, size);

        // parallel init
        std::for_each_n(std::execution::par, counting_iterator(0), size,
                        [=](std::size_t i) { current[i] = (Real_t)i; });

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t) {
            std::for_each_n(std::execution::par, counting_iterator(0), size, [=, k = k, dt = dt, dx = dx](int32_t i) {
                std::size_t left = (i == 0) ? size - 1 : i - 1;
                std::size_t right = (i == size - 1) ? 0 : i + 1;
                next[i] = heat(current[left], current[i], current[right], k, dt, dx);
            });
            std::swap(current, next);
        }

        return current;
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {
    std::uint64_t size = args.size;  // Number of elements.
    std::uint64_t nt = args.nt;      // Number of steps.

    // Create the stepper object
    stepper step;

    // Measure execution time.
    Timer timer;

    // Execute nt time steps on nx grid points.
    auto solution = step.do_work(size, nt);
    auto time = timer.stop();

    // Print the final solution
    if (args.results) {
        fmt::println("{::f}", solution);
    }

    if (args.time) {
        fmt::print("Duration: {:f} ms\n", time);
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
