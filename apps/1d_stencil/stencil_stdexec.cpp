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
// This example provides a stdexec implementation for the 1D stencil code.
#include <exec/static_thread_pool.hpp>
#include <nvexec/multi_gpu_context.cuh>
#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

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
    bool& no_header = kwarg("no-header", "Do not print csv header row (default: false)").set_default(false);
    bool& help = flag("h, help", "print help");
    bool& time = kwarg("t, time", "print time").set_default(true);
    std::string& sch = kwarg("sch", "stdexec scheduler: [options: cpu, gpu, multigpu]").set_default("cpu");
    int& nthreads = kwarg("nthreads", "number of threads").set_default(std::thread::hardware_concurrency());
};

using Real_t = double;
///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true;  // print csv heading
Real_t k = 0.5;      // heat transfer coefficient
Real_t dt = 1.;      // time step
Real_t dx = 1.;      // grid spacing

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper {

    // Our operator
    Real_t heat(Real_t left, Real_t middle, Real_t right, const Real_t k = ::k, const Real_t dt = ::dt,
                const Real_t dx = ::dx) {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // do all the work on 'size' data points for 'nt' time steps
    [[nodiscard]] std::vector<Real_t> do_work(const auto& sch, std::size_t size, std::size_t nt) {
        std::vector<Real_t> current_vec(size);
        std::vector<Real_t> next_vec(size);

        auto current_ptr = current_vec.data();
        auto next_ptr = next_vec.data();

        stdexec::sender auto init = stdexec::transfer_just(sch, current_ptr) |
                                    stdexec::bulk(size, [&](int i, auto& current_ptr) { current_ptr[i] = (Real_t)i; });
        stdexec::sync_wait(std::move(init));

        for (std::size_t t = 0; t != nt; ++t) {
            auto sender =
                stdexec::transfer_just(sch, current_ptr, next_ptr, k, dt, dx, size) |
                stdexec::bulk(size, [&](int i, auto& current_ptr, auto& next_ptr, auto k, auto dt, auto dx, auto size) {
                    std::size_t left = (i == 0) ? size - 1 : i - 1;
                    std::size_t right = (i == size - 1) ? 0 : i + 1;
                    next_ptr[i] = heat(current_ptr[left], current_ptr[i], current_ptr[right], k, dt, dx);
                });
            stdexec::sync_wait(std::move(sender));
            std::swap(current_ptr, next_ptr);
        }

        if (nt % 2 == 0) {
            return current_vec;
        }
        return next_vec;
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {
    std::uint64_t size = args.size;  // Number of elements.
    std::uint64_t nt = args.nt;      // Number of steps.
    std::string sch_str = args.sch;  // scheduler type
    int nthreads = args.nthreads;    // number of threads for cpu scheduler type

    // Create the stepper object
    stepper step;

    // Measure execution time.
    Timer timer;

    // Execute nt time steps on size of elements.
    // launch with appropriate stdexec scheduler
    std::vector<Real_t> solution;
    try {
        sch_t schedulerType = get_sch_enum(sch_str);

        switch (schedulerType) {
            case sch_t::CPU:
                solution = step.do_work(exec::static_thread_pool(nthreads).get_scheduler(), size, nt);
                break;
            case sch_t::GPU:
                solution = step.do_work(nvexec::stream_context().get_scheduler(), size, nt);
                break;
            case sch_t::MULTIGPU:
                solution = step.do_work(nvexec::multi_gpu_stream_context().get_scheduler(), size, nt);
                break;
            default:
                std::cerr << "Unknown scheduler type encountered." << std::endl;
                break;
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    auto time = timer.stop();

    // Print the final solution
    if (args.results) {
        for (std::size_t i = 0; i != size; ++i) {
            std::cout << solution[i] << " ";
        }
        std::cout << "\n";
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
